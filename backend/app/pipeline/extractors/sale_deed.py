"""Sale Deed document extractor — text-based."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import PROMPTS_DIR, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_SALE_DEED_SCHEMA
from app.pipeline.sale_deed_preparse import (
    preparse_sale_deed,
    format_hints_for_prompt,
    PreparseHints,
    extract_party_names_from_section,
    _is_garbage_name,
)
from app.pipeline.utils import name_similarity

logger = logging.getLogger(__name__)


def _pick_richer(a, b):
    """Return the richer (more populated) of two values.

    For dicts: prefer the one with more non-empty values.
    For scalars: prefer non-empty/non-zero.
    NOTE: Lists (seller/buyer/witnesses) are handled separately via _union_parties.
    """
    if a is None or a == "" or a == 0:
        return b
    if b is None or b == "" or b == 0:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        a_count = sum(1 for v in a.values() if v)
        b_count = sum(1 for v in b.values() if v)
        return b if b_count > a_count else a
    if isinstance(a, list) and isinstance(b, list):
        return b if len(b) > len(a) else a
    return a  # keep first non-empty


# ── Party deduplication constants ──────────────────────────────────
_PARTY_LIST_FIELDS = {"seller", "buyer", "witnesses"}
_OWNERSHIP_HISTORY_FIELD = "ownership_history"
_PARTY_DEDUP_THRESHOLD = 0.65  # name_similarity threshold for dedup


def _union_parties(existing: list, incoming: list) -> list:
    """Union two party lists, deduplicating by fuzzy name match.

    For seller/buyer arrays (list of dicts with 'name' key), merges
    entries keeping the richer version when duplicates are found.
    For witness arrays (list of strings), deduplicates by similarity.
    """
    if not incoming:
        return existing
    if not existing:
        return incoming

    merged = list(existing)  # shallow copy

    for item in incoming:
        # Extract name from dict or use string directly
        if isinstance(item, dict):
            new_name = item.get("name", "")
        else:
            new_name = str(item)

        if not new_name:
            continue

        # Check if this name already exists in merged
        found_match = False
        for i, existing_item in enumerate(merged):
            if isinstance(existing_item, dict):
                ex_name = existing_item.get("name", "")
            else:
                ex_name = str(existing_item)

            if not ex_name:
                continue

            sim = name_similarity(new_name, ex_name)
            if sim >= _PARTY_DEDUP_THRESHOLD:
                found_match = True
                # Keep whichever has more populated fields
                if isinstance(item, dict) and isinstance(existing_item, dict):
                    new_filled = sum(1 for v in item.values() if v)
                    ex_filled = sum(1 for v in existing_item.values() if v)
                    if new_filled > ex_filled:
                        merged[i] = item
                break

        if not found_match:
            merged.append(item)

    return merged


def _union_ownership_history(existing: list, incoming: list) -> list:
    """Union ownership_history lists, deduplicating by owner+document_number.

    Keeps the richer entry when duplicates are found.  Entries are sorted
    chronologically (by document_date) at the end.
    """
    if not incoming:
        return existing
    if not existing:
        return incoming

    merged = list(existing)

    for item in incoming:
        if not isinstance(item, dict):
            continue
        new_owner = (item.get("owner") or "").strip()
        new_doc = (item.get("document_number") or "").strip()
        if not new_owner:
            continue

        found = False
        for i, ex in enumerate(merged):
            if not isinstance(ex, dict):
                continue
            ex_owner = (ex.get("owner") or "").strip()
            ex_doc = (ex.get("document_number") or "").strip()
            # Match on fuzzy owner name AND same doc number (if both present)
            owner_sim = name_similarity(new_owner, ex_owner) if ex_owner else 0.0
            doc_match = (new_doc and ex_doc and new_doc == ex_doc)
            if owner_sim >= _PARTY_DEDUP_THRESHOLD or doc_match:
                found = True
                new_filled = sum(1 for v in item.values() if v)
                ex_filled = sum(1 for v in ex.values() if v)
                if new_filled > ex_filled:
                    merged[i] = item
                break

        if not found:
            merged.append(item)

    # Sort chronologically by document_date (best effort)
    def _sort_key(entry):
        return entry.get("document_date") or "" if isinstance(entry, dict) else ""

    merged.sort(key=_sort_key)
    return merged


def _deep_merge_property(existing: dict, incoming: dict) -> dict:
    """Deep-merge two property dicts, keeping non-empty values per sub-field.

    Standard _pick_richer compares by total field count, which loses
    boundary values extracted in later chunks when the first chunk
    returned an empty boundaries object.  This merges field-by-field,
    and for the nested 'boundaries' dict, direction-by-direction.
    """
    if not existing:
        return incoming
    if not incoming:
        return existing

    merged = dict(existing)
    for key, val in incoming.items():
        ex_val = merged.get(key)
        if key == "boundaries" and isinstance(ex_val, dict) and isinstance(val, dict):
            # Merge direction-by-direction (north/south/east/west)
            merged_bounds = dict(ex_val)
            for direction, bval in val.items():
                if bval and (not merged_bounds.get(direction)):
                    merged_bounds[direction] = bval
            merged[key] = merged_bounds
        elif ex_val is None or ex_val == "" or ex_val == 0:
            if val is not None and val != "" and val != 0:
                merged[key] = val
        elif val is not None and val != "" and val != 0:
            # Both non-empty: keep the richer one
            merged[key] = _pick_richer(ex_val, val)
    return merged


def _merge_sale_deed_results(chunk_results: list, total_pages: int) -> dict:
    """Merge Sale Deed extraction results from multiple chunks.

    Unlike EC merge (concatenate transactions), Sale Deed merge:
    - Picks the richer/non-empty value for scalar fields
    - UNIONS party lists (seller, buyer, witnesses) with deduplication
    """
    merged: dict = {}
    notes: list[str] = []

    for i, result in enumerate(chunk_results):
        if isinstance(result, Exception):
            notes.append(f"Chunk {i + 1} failed: {str(result)}")
            continue
        if not isinstance(result, dict):
            continue

        for key in EXTRACT_SALE_DEED_SCHEMA.get("required", []):
            existing = merged.get(key)
            incoming = result.get(key)

            if key in _PARTY_LIST_FIELDS:
                # Union party lists with dedup
                if isinstance(incoming, list) and incoming:
                    merged[key] = _union_parties(
                        existing if isinstance(existing, list) else [],
                        incoming,
                    )
            elif key == _OWNERSHIP_HISTORY_FIELD:
                # Union ownership history with dedup by owner+doc
                if isinstance(incoming, list) and incoming:
                    merged[key] = _union_ownership_history(
                        existing if isinstance(existing, list) else [],
                        incoming,
                    )
            elif key == "property":
                # Deep-merge property dict to preserve boundaries from later chunks
                if isinstance(incoming, dict) and incoming:
                    merged[key] = _deep_merge_property(
                        existing if isinstance(existing, dict) else {},
                        incoming,
                    )
            elif existing is None:
                if incoming is not None:
                    merged[key] = incoming
            else:
                merged[key] = _pick_richer(existing, incoming)

        # Also pick up non-required fields
        for key, val in result.items():
            if key not in merged and val:
                merged[key] = val

    if notes:
        existing_notes = merged.get("remarks", "") or ""
        merged["remarks"] = (existing_notes + " | " + "; ".join(notes)).strip(" | ")

    return merged


def _names_overlap(extracted_names: list[str], hint_names: list[str], threshold: float = 0.55) -> bool:
    """Return True if any extracted name matches any hint name."""
    for en in extracted_names:
        for hn in hint_names:
            if name_similarity(en, hn) >= threshold:
                return True
    return False


def _validate_seller_buyer(merged: dict, hints: PreparseHints) -> dict:
    """Post-extraction guard: swap seller/buyer if they contradict preparse hints.

    If the preparse detected seller_names and buyer_names (from document
    structure), but the LLM placed them in the wrong arrays, swap them.

    Gracefully degrades: if >50% of hint names are garbage (location fragments),
    skip the validation entirely to avoid corrupting results with bad signals.
    """
    hint_sellers = hints.get("seller_names", [])
    hint_buyers = hints.get("buyer_names", [])
    if not hint_sellers or not hint_buyers:
        return merged  # no hint names → can't validate

    # Check hint quality: if most names are garbage, bail out
    all_hint_names = hint_sellers + hint_buyers
    garbage_count = sum(1 for n in all_hint_names if _is_garbage_name(n))
    if garbage_count > len(all_hint_names) * 0.5:
        logger.warning(
            "Seller/buyer hint names are mostly garbage (%d/%d) — skipping swap validation.",
            garbage_count, len(all_hint_names),
        )
        return merged

    extracted_sellers = merged.get("seller", [])
    extracted_buyers = merged.get("buyer", [])
    if not extracted_sellers or not extracted_buyers:
        return merged

    # Extract name strings from party dicts
    def _get_names(party_list: list) -> list[str]:
        names = []
        for p in party_list:
            if isinstance(p, dict):
                n = p.get("name", "")
            else:
                n = str(p)
            if n:
                names.append(n)
        return names

    ex_seller_names = _get_names(extracted_sellers)
    ex_buyer_names = _get_names(extracted_buyers)

    # Check if extracted sellers match hint sellers (correct assignment)
    sellers_correct = _names_overlap(ex_seller_names, hint_sellers)
    buyers_correct = _names_overlap(ex_buyer_names, hint_buyers)

    # Check if they're swapped: extracted sellers match hint buyers
    sellers_swapped = _names_overlap(ex_seller_names, hint_buyers)
    buyers_swapped = _names_overlap(ex_buyer_names, hint_sellers)

    if (sellers_swapped and buyers_swapped) and not (sellers_correct or buyers_correct):
        # Definite swap detected — fix it
        logger.warning(
            "PARTY SWAP DETECTED: LLM placed sellers in buyer[] and vice versa. "
            f"Extracted sellers={ex_seller_names}, hint sellers={hint_sellers}. Swapping."
        )
        merged["seller"], merged["buyer"] = merged["buyer"], merged["seller"]
        existing_remarks = merged.get("remarks", "") or ""
        merged["remarks"] = (
            existing_remarks + " | " +
            "Auto-corrected: seller/buyer arrays were swapped based on preparse section detection."
        ).strip(" | ")

    return merged


def _filter_current_sale_from_history(merged: dict) -> dict:
    """Remove ownership_history entries where the owner is a current buyer.

    The ownership_history should only contain PAST transfers showing how the
    property reached the current seller.  The current sale (seller→buyer) is
    NOT a history entry.  This post-processor enforces that rule.
    """
    history = merged.get("ownership_history", [])
    buyers = merged.get("buyer", [])
    if not history or not buyers:
        return merged

    # Build buyer name set
    buyer_names: list[str] = []
    for b in buyers:
        if isinstance(b, dict):
            n = b.get("name", "")
        else:
            n = str(b)
        if n:
            buyer_names.append(n)

    if not buyer_names:
        return merged

    filtered: list[dict] = []
    for entry in history:
        if not isinstance(entry, dict):
            filtered.append(entry)
            continue
        owner = (entry.get("owner") or "").strip()
        if not owner:
            filtered.append(entry)
            continue
        # Check if this owner matches any buyer name
        is_buyer = False
        for bn in buyer_names:
            if name_similarity(owner, bn) >= 0.55:
                is_buyer = True
                break
        if is_buyer:
            logger.info(
                "Removed ownership_history entry where owner='%s' matches a buyer (current sale).",
                owner,
            )
        else:
            filtered.append(entry)

    if len(filtered) != len(history):
        merged["ownership_history"] = filtered
    return merged


class SaleDeedExtractor(BaseExtractor):
    """Extract transaction and property details from Sale Deeds (text-based).

    For large documents (>MAX_CHUNK_PAGES pages), splits into page-aligned
    chunks and processes concurrently, then merges the richer values.

    Runs deterministic pre-parsing BEFORE LLM extraction to anchor key
    fields (registration number, survey number, village, seller/buyer
    sections) so the LLM doesn't confuse them.
    """

    _PROMPT_HINTS_PLACEHOLDER = "{preparse_hints}"

    def __init__(self):
        self._raw_system_prompt = (PROMPTS_DIR / "extract_sale_deed.txt").read_text(encoding="utf-8")

    def _build_system_prompt(self, hints: PreparseHints) -> str:
        """Build system prompt with pre-parse hints injected."""
        hints_text = format_hints_for_prompt(hints)
        if hints_text:
            hints_block = f"\n{hints_text}\n"
        else:
            hints_block = ""
        return self._raw_system_prompt.replace(self._PROMPT_HINTS_PLACEHOLDER, hints_block)

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None) -> dict:
        name = filename or "Sale Deed"
        pages = extracted_text.get("pages", [])
        total_pages = len(pages)
        full_text = extracted_text.get("full_text", "")

        # ── Run deterministic pre-parser ──
        hints = preparse_sale_deed(full_text)
        system_prompt = self._build_system_prompt(hints)

        # Small document — single-pass (existing behaviour)
        if total_pages <= MAX_CHUNK_PAGES:
            prompt = f"Extract all details from this Sale Deed:\n\n{full_text}"
            result = await call_llm(
                prompt=prompt,
                system_prompt=system_prompt,
                expect_json=EXTRACT_SALE_DEED_SCHEMA,
                task_label=f"{name} Sale Deed extraction ({len(full_text):,} chars)",
                on_progress=on_progress,
                think=True,
            )
            result = _validate_seller_buyer(result, hints)
            return _filter_current_sale_from_history(result)

        # Large document — chunk and merge
        chunks = self._create_chunks(pages)
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)
        logger.info(f"SaleDeedExtractor: {total_pages} pages → {len(chunks)} chunks, concurrency={LLM_MAX_CONCURRENT_CHUNKS}")

        # Sequential extraction with prior-chunk context injection:
        # Each chunk after the first receives a brief summary of the
        # previous chunk's extraction, so the LLM knows which fields
        # are already captured and can focus on new/missing data.
        chunk_results: list = []
        prior_context = ""
        for i, c in enumerate(chunks):
            try:
                result = await self._extract_chunk(
                    c["text"], i + 1, len(chunks), name, on_progress,
                    prior_context=prior_context,
                    system_prompt=system_prompt,
                )
                chunk_results.append(result)
                # Build context summary for the next chunk
                prior_context = self._summarize_chunk_result(result, i + 1)
            except Exception as e:
                chunk_results.append(e)
                prior_context = f"[Chunk {i + 1} extraction failed]"

        merged = _merge_sale_deed_results(chunk_results, total_pages)
        merged = _validate_seller_buyer(merged, hints)
        return _filter_current_sale_from_history(merged)

    # ── Section markers commonly found in Tamil Nadu Sale Deeds ──
    _SECTION_MARKERS = [
        # English markers (case-insensitive matching)
        "schedule", "property schedule", "schedule of property",
        "boundaries", "boundary description",
        "consideration", "sale consideration",
        "witness", "witnesses", "attestation",
        "conditions", "special conditions", "covenants",
        "recitals", "whereas",
        "encumbrances", "free from encumbrance",
        # Tamil markers (partial — common section headers)
        "அட்டவணை",      # Schedule
        "எல்லை",         # Boundary
        "விலை",          # Consideration/Price
        "சாட்சிகள்",     # Witnesses
    ]

    def _create_chunks(self, pages: list[dict]) -> list[dict]:
        """Split pages into processable chunks, preferring section boundaries.

        Attempts to detect Sale Deed section breaks (Schedule, Boundaries,
        Consideration, Witnesses) and split at those points so each chunk
        contains a semantically coherent section. Falls back to page-based
        splitting when no section markers are found or when sections span
        too many pages.
        """
        if len(pages) <= MAX_CHUNK_PAGES:
            # Small enough for a single chunk
            text = "\n\n".join(
                f"--- PAGE {p['page_number']} ---\n{p['text']}" for p in pages
            )
            return [{"index": 0, "start_page": pages[0]["page_number"],
                     "end_page": pages[-1]["page_number"], "text": text}]

        # Try to find section boundaries
        section_breaks = []  # List of page indices where sections start
        for pi, page in enumerate(pages):
            text_lower = page.get("text", "").lower()
            for marker in self._SECTION_MARKERS:
                if marker.lower() in text_lower:
                    section_breaks.append(pi)
                    break

        # Use section-aware splitting if we found meaningful breaks
        if len(section_breaks) >= 2:
            return self._split_by_sections(pages, section_breaks)

        # Fallback: page-based splitting
        return self._split_by_pages(pages)

    def _split_by_sections(self, pages: list[dict], breaks: list[int]) -> list[dict]:
        """Split at section boundaries, merging small sections."""
        # Deduplicate and sort break points
        breaks = sorted(set(breaks))
        # Always include page 0 as a break
        if 0 not in breaks:
            breaks = [0] + breaks

        chunks = []
        for bi in range(len(breaks)):
            start_idx = breaks[bi]
            end_idx = breaks[bi + 1] if bi + 1 < len(breaks) else len(pages)
            section_pages = pages[start_idx:end_idx]

            # If this section is too large, sub-split by page count
            if len(section_pages) > MAX_CHUNK_PAGES:
                for sub in self._split_by_pages(section_pages):
                    sub["index"] = len(chunks)
                    chunks.append(sub)
            else:
                text = "\n\n".join(
                    f"--- PAGE {p['page_number']} ---\n{p['text']}" for p in section_pages
                )
                chunks.append({
                    "index": len(chunks),
                    "start_page": section_pages[0]["page_number"],
                    "end_page": section_pages[-1]["page_number"],
                    "text": text,
                })

        # Merge very small trailing chunks (< 2 pages) into the previous one
        merged = []
        for chunk in chunks:
            if (merged and
                    chunk["start_page"] - merged[-1]["end_page"] <= 1 and
                    chunk["text"].count("--- PAGE") < 2 and
                    merged[-1]["text"].count("--- PAGE") < MAX_CHUNK_PAGES):
                merged[-1]["text"] += "\n\n" + chunk["text"]
                merged[-1]["end_page"] = chunk["end_page"]
            else:
                merged.append(chunk)

        # Re-index
        for i, c in enumerate(merged):
            c["index"] = i

        return merged

    @staticmethod
    def _split_by_pages(pages: list[dict]) -> list[dict]:
        """Fallback: split into page-count-based chunks."""
        chunks = []
        for i in range(0, len(pages), MAX_CHUNK_PAGES):
            chunk_pages = pages[i:i + MAX_CHUNK_PAGES]
            chunk_text = "\n\n".join(
                f"--- PAGE {p['page_number']} ---\n{p['text']}"
                for p in chunk_pages
            )
            chunks.append({
                "index": len(chunks),
                "start_page": chunk_pages[0]["page_number"],
                "end_page": chunk_pages[-1]["page_number"],
                "text": chunk_text,
            })
        return chunks

    async def _extract_chunk(self, text: str, chunk_num: int, total_chunks: int,
                             name: str, on_progress: LLMProgressCallback | None,
                             prior_context: str = "",
                             system_prompt: str = "") -> dict:
        """Extract from a single chunk of Sale Deed text."""
        label = f"{name} chunk {chunk_num}/{total_chunks} ({len(text):,} chars)"

        context_section = ""
        if prior_context:
            context_section = (
                f"\n\nCONTEXT FROM PREVIOUS CHUNKS:\n{prior_context}\n"
                f"Focus on extracting any NEW or UPDATED information from this "
                f"chunk. If a field was already extracted, only override it if "
                f"this chunk has a more complete or accurate value.\n\n"
            )

        prompt = (
            f"Extract all details from this section of a Sale Deed "
            f"(chunk {chunk_num} of {total_chunks}):"
            f"{context_section}\n{text}"
        )
        return await call_llm(
            prompt=prompt,
            system_prompt=system_prompt or self._raw_system_prompt,
            expect_json=EXTRACT_SALE_DEED_SCHEMA,
            task_label=label,
            on_progress=on_progress,
            think=True,
        )

    @staticmethod
    def _summarize_chunk_result(result: dict, chunk_num: int) -> str:
        """Build a compact context summary from a chunk extraction result.

        Provides the next chunk with awareness of what was already
        extracted, preventing redundant extraction and helping the LLM
        identify continuation data.
        """
        if not isinstance(result, dict):
            return f"[Chunk {chunk_num} produced no usable result]"

        parts = [f"Chunk {chunk_num} extracted:"]
        if result.get("document_number"):
            parts.append(f"  Doc No: {result['document_number']}")
        if result.get("registration_date"):
            parts.append(f"  Reg Date: {result['registration_date']}")
        if result.get("sro"):
            parts.append(f"  SRO: {result['sro']}")

        sellers = result.get("seller", [])
        if sellers and isinstance(sellers, list):
            names = [s.get("name", "?") if isinstance(s, dict) else str(s) for s in sellers[:3]]
            parts.append(f"  Sellers: {', '.join(names)}")

        buyers = result.get("buyer", [])
        if buyers and isinstance(buyers, list):
            names = [b.get("name", "?") if isinstance(b, dict) else str(b) for b in buyers[:3]]
            parts.append(f"  Buyers: {', '.join(names)}")

        prop = result.get("property", {})
        if isinstance(prop, dict):
            if prop.get("survey_number"):
                parts.append(f"  Survey No: {prop['survey_number']}")
            if prop.get("extent"):
                parts.append(f"  Extent: {prop['extent']}")
            if prop.get("village"):
                parts.append(f"  Village: {prop['village']}")

        fin = result.get("financials", {})
        if isinstance(fin, dict) and fin.get("consideration_amount"):
            parts.append(f"  Consideration: {fin['consideration_amount']}")

        return "\n".join(parts)
