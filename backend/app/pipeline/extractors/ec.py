"""EC (Encumbrance Certificate) extractor - handles large multi-page ECs.

Chunking strategy (v2 — semantic + char-budget):
  1. Detect transaction boundaries in raw OCR text via regex patterns
  2. Group complete transactions into chunks, each under MAX_CHUNK_CHARS
  3. Overlap = last transaction of previous chunk (semantic, not page-based)
  4. Fallback: if < 2 boundaries detected, use page-based + char budget
  5. EC header (page 1 context) injected into chunks 2+ via RAG query
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import (
    PROMPTS_DIR, MAX_CHUNK_PAGES, MAX_CHUNK_CHARS, LLM_MAX_CONCURRENT_CHUNKS,
)
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_EC_SCHEMA

if TYPE_CHECKING:
    from app.pipeline.rag_store import RAGStore

logger = logging.getLogger(__name__)

# ── Transaction boundary regex patterns ──────────────────────────────────
# These detect the start of a new EC transaction row in raw OCR text.
# Order matters: first match wins.
_TXN_BOUNDARY_PATTERNS = [
    # Row number + date:  "1  13-Dec-2012" or "12 01-Jan-2020"
    re.compile(r'^\s{0,4}(\d{1,4})\s{1,8}\d{1,2}-[A-Za-z]{3}-\d{4}', re.MULTILINE),
    # "Document No." or "Document No :" header
    re.compile(r'^\s{0,4}Document\s*No\.?\s*:?\s*\d+', re.MULTILINE | re.IGNORECASE),
    # Tamil: ஆவண எண் (Document Number) — supports both Latin and Tamil numerals
    re.compile(r'^\s{0,4}ஆவண\s*எண்\s*[:.]?\s*[\d௧-௯௦]+', re.MULTILINE),
]


def _detect_txn_starts(text: str) -> list[int]:
    """Find character offsets where new EC transactions begin.

    Returns sorted list of unique char offsets.  Each offset marks the
    start of a line that matches a transaction boundary pattern.
    """
    offsets: set[int] = set()
    for pattern in _TXN_BOUNDARY_PATTERNS:
        for m in pattern.finditer(text):
            offsets.add(m.start())
    return sorted(offsets)


def _txn_dedup_key(txn: dict) -> str:
    """Build a deduplication key from a transaction's immutable fields.

    Uses document_number + date + parties + survey_number + transaction_type
    so that the same transaction extracted from overlapping chunks is detected
    as a duplicate, while distinct transactions on the same date by the same
    parties (e.g., a SALE and a MORTGAGE) are kept separate.
    """
    doc_no = str(txn.get("document_number", "")).strip().lower()
    date = str(txn.get("date", "")).strip()
    seller = str(txn.get("seller_or_executant", "")).strip().lower()[:60]
    buyer = str(txn.get("buyer_or_claimant", "")).strip().lower()[:60]
    survey = str(txn.get("survey_number", "")).strip().lower()[:30]
    ttype = str(txn.get("transaction_type", "")).strip().lower()
    return f"{doc_no}|{date}|{seller}|{buyer}|{survey}|{ttype}"


def _txn_richness(txn: dict) -> int:
    """Score how many fields in a transaction are populated (non-empty)."""
    count = 0
    for k, v in txn.items():
        if k.startswith("_") or k == "row_number":
            continue
        if v is not None and v != "" and v != [] and v != 0:
            count += 1
    return count


def _dedup_ec_transactions(transactions: list[dict]) -> list[dict]:
    """Remove duplicate transactions that span chunk boundaries.

    When two transactions share the same document_number + date + parties,
    keep the one with more populated fields (richer extraction).
    Preserves original ordering.
    """
    seen: dict[str, tuple[int, dict]] = {}  # key → (richness, txn)
    order: list[str] = []                   # insertion order of keys

    for txn in transactions:
        key = _txn_dedup_key(txn)
        richness = _txn_richness(txn)
        if key in seen:
            prev_richness, _ = seen[key]
            if richness > prev_richness:
                seen[key] = (richness, txn)
            # else keep existing (richer or equal)
        else:
            seen[key] = (richness, txn)
            order.append(key)

    return [seen[k][1] for k in order]


def _assign_transaction_ids(transactions: list[dict]) -> None:
    """Assign stable `transaction_id` and sequential `row_number` in-place.

    ID format:  EC-{doc_number}/{doc_year}-{sro}
    Fallback:   EC-ROW{idx}  (when doc_number or doc_year missing)
    Collision:  append -a, -b, -c suffix if same ID produced twice
    """
    used_ids: dict[str, int] = {}  # base_id → count of times used

    for idx, txn in enumerate(transactions, 1):
        txn["row_number"] = idx

        doc_no = str(txn.get("document_number", "")).strip()
        doc_year = str(txn.get("document_year", "")).strip()
        sro = str(txn.get("sro", "")).strip()[:30]

        if doc_no and doc_year:
            base_id = f"EC-{doc_no}/{doc_year}"
            if sro:
                base_id += f"-{sro}"
        else:
            base_id = f"EC-ROW{idx}"

        # Handle collisions (extremely rare: same doc re-registered)
        if base_id in used_ids:
            count = used_ids[base_id]
            tid = f"{base_id}-{count}"  # numeric suffix: -2, -3, ...
            used_ids[base_id] = count + 1
        else:
            tid = base_id
            used_ids[base_id] = 2  # next collision gets -2

        txn["transaction_id"] = tid


class ECExtractor(BaseExtractor):
    """Extract transactions from Encumbrance Certificates.

    Handles large ECs (65+ pages) via semantic chunking with char budget:
      - Detects transaction boundaries in raw OCR text
      - Groups complete transactions into chunks ≤ MAX_CHUNK_CHARS
      - Injects EC header context (page 1) into later chunks via RAG
      - Falls back to page-based chunking if boundary detection fails
    """

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_ec.txt").read_text(encoding="utf-8")

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: "Path | None" = None,
        rag_store: "RAGStore | None" = None,
        embed_fn=None,
    ) -> dict:
        """Extract all EC transactions via chunked LLM processing."""
        pages = extracted_text["pages"]
        total_pages = len(pages)
        self._filename = filename or "EC"
        self._rag_store = rag_store
        self._embed_fn = embed_fn

        if total_pages <= MAX_CHUNK_PAGES:
            full_text = extracted_text["full_text"]
            if len(full_text) <= MAX_CHUNK_CHARS:
                # Small EC — process in one shot
                result = await self._extract_chunk(full_text, 1, 1, on_progress)
            else:
                # Small page count but dense text — use semantic chunking
                chunks = self._create_chunks(pages)
                result = await self._run_chunked(chunks, total_pages, on_progress)
        else:
            # Large EC — semantic chunking with char budget
            chunks = self._create_chunks(pages)
            result = await self._run_chunked(chunks, total_pages, on_progress)

        # ── Post-extraction: flag garbled Tamil names ──
        result = await self._vision_recheck_names(result, file_path, on_progress)

        return result

    async def _run_chunked(
        self, chunks: list[dict], total_pages: int,
        on_progress: LLMProgressCallback | None = None,
    ) -> dict:
        """Process chunks with concurrency limiter and merge results."""
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)
        logger.info(f"ECExtractor: {len(chunks)} chunks, concurrency={LLM_MAX_CONCURRENT_CHUNKS}")

        async def _limited_extract(chunk_text: str, idx: int):
            async with sem:
                return await self._extract_chunk(
                    chunk_text, idx + 1, len(chunks), on_progress,
                )

        tasks = [_limited_extract(chunk["text"], i) for i, chunk in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(chunk_results, total_pages)

    # ── Chunking (semantic boundaries + char budget) ───────────────────────

    def _create_chunks(self, pages: list[dict]) -> list[dict]:
        """Split pages into chunks using transaction boundaries + char budget.

        Strategy:
          1. Build full text from all pages
          2. Detect transaction boundary offsets via regex
          3. If ≥ 2 boundaries found → semantic chunking (group complete txns)
          4. If < 2 boundaries → fallback to page-based + char budget
          5. Each chunk ≤ MAX_CHUNK_CHARS; overlap = last txn of prev chunk
        """
        full_text = "\n\n".join(
            f"--- PAGE {p['page_number']} ---\n{p['text']}" for p in pages
        )
        boundaries = _detect_txn_starts(full_text)

        if len(boundaries) >= 2:
            chunks = self._semantic_chunks(full_text, boundaries, pages)
            logger.info(
                f"ECExtractor: semantic chunking — {len(boundaries)} txn boundaries "
                f"→ {len(chunks)} chunks"
            )
        else:
            chunks = self._page_chunks_with_budget(pages)
            logger.info(
                f"ECExtractor: fallback page chunking — {len(chunks)} chunks "
                f"(boundary detection found {len(boundaries)})"
            )
        return chunks

    def _semantic_chunks(
        self, full_text: str, boundaries: list[int], pages: list[dict],
    ) -> list[dict]:
        """Group complete transactions into chunks under MAX_CHUNK_CHARS.

        Each “segment” is the text between two consecutive boundary offsets.
        Segments are accumulated until adding the next would exceed the budget.
        Overlap: the last segment of the previous chunk is repeated at the
        start of the next chunk (semantic overlap = 1 transaction).
        """
        # ── Build char-offset → page_number mapping for accurate page tracking ──
        page_offsets: list[tuple[int, int]] = []  # (start_char, page_number)
        offset = 0
        for i, p in enumerate(pages):
            page_header = f"--- PAGE {p['page_number']} ---\n{p['text']}"
            page_offsets.append((offset, p["page_number"]))
            offset += len(page_header)
            if i < len(pages) - 1:
                offset += 2  # "\n\n" separator

        def _page_at(char_pos: int) -> int:
            """Return the page_number that contains the given char offset."""
            result = pages[0]["page_number"]
            for start, pnum in page_offsets:
                if start <= char_pos:
                    result = pnum
                else:
                    break
            return result

        # Build segments: text between consecutive boundaries
        # First segment = everything before the first boundary (header/metadata)
        segments: list[tuple[str, int]] = []  # (text, start_offset)
        seg_starts: list[int] = [0] + boundaries
        seg_ends: list[int] = boundaries + [len(full_text)]
        for s, e in zip(seg_starts, seg_ends):
            seg = full_text[s:e]
            if seg.strip():
                segments.append((seg, s))

        if not segments:
            return [{
                "index": 0,
                "start_page": pages[0]["page_number"],
                "end_page": pages[-1]["page_number"],
                "text": full_text,
            }]

        chunks: list[dict] = []
        current_segs: list[tuple[str, int]] = []  # (text, start_offset)
        current_len = 0

        for seg_text, seg_offset in segments:
            seg_len = len(seg_text)

            # If adding this segment would exceed budget AND we already
            # have content, close the current chunk.
            if current_len + seg_len > MAX_CHUNK_CHARS and current_segs:
                chunk_text = "".join(s[0] for s in current_segs)
                chunks.append({
                    "index": len(chunks),
                    "start_page": _page_at(current_segs[0][1]),
                    "end_page": _page_at(current_segs[-1][1] + len(current_segs[-1][0]) - 1),
                    "text": chunk_text,
                })
                # Overlap: carry last segment into next chunk
                last_seg = current_segs[-1]
                current_segs = [last_seg]
                current_len = len(last_seg[0])

            current_segs.append((seg_text, seg_offset))
            current_len += seg_len

        # Flush remaining
        if current_segs:
            chunk_text = "".join(s[0] for s in current_segs)
            chunks.append({
                "index": len(chunks),
                "start_page": _page_at(current_segs[0][1]),
                "end_page": _page_at(current_segs[-1][1] + len(current_segs[-1][0]) - 1),
                "text": chunk_text,
            })

        return chunks

    def _page_chunks_with_budget(self, pages: list[dict]) -> list[dict]:
        """Fallback: page-based chunking with char budget.

        Groups pages until MAX_CHUNK_PAGES or MAX_CHUNK_CHARS is reached.
        Overlap = 1 page from previous chunk.
        """
        chunks: list[dict] = []
        current_pages: list[dict] = []
        current_chars = 0

        for page in pages:
            page_text = page.get("text", "")
            page_chars = len(page_text) + 20  # +20 for PAGE header

            # Close chunk if adding this page would exceed either limit
            if current_pages and (
                len(current_pages) >= MAX_CHUNK_PAGES
                or current_chars + page_chars > MAX_CHUNK_CHARS
            ):
                chunk_text = "\n\n".join(
                    f"--- PAGE {p['page_number']} ---\n{p['text']}"
                    for p in current_pages
                )
                chunks.append({
                    "index": len(chunks),
                    "start_page": current_pages[0]["page_number"],
                    "end_page": current_pages[-1]["page_number"],
                    "text": chunk_text,
                })
                # Overlap: carry last page into next chunk
                overlap_page = current_pages[-1]
                current_pages = [overlap_page]
                current_chars = len(overlap_page.get("text", "")) + 20

            current_pages.append(page)
            current_chars += page_chars

        # Flush remaining
        if current_pages:
            chunk_text = "\n\n".join(
                f"--- PAGE {p['page_number']} ---\n{p['text']}"
                for p in current_pages
            )
            chunks.append({
                "index": len(chunks),
                "start_page": current_pages[0]["page_number"],
                "end_page": current_pages[-1]["page_number"],
                "text": chunk_text,
            })

        return chunks

    # ── Per-chunk extraction ──────────────────────────────────────────

    async def _extract_chunk(
        self, text: str, chunk_num: int = 1, total_chunks: int = 1,
        on_progress: LLMProgressCallback | None = None,
    ) -> dict:
        """Extract transactions from a single chunk of EC text.

        For chunks 2+ in multi-chunk ECs, injects EC header context
        (property description, survey, SRO, period) retrieved via RAG.
        """
        name = getattr(self, '_filename', 'EC')

        # ── EC header injection via RAG for later chunks ──
        header_context = ""
        if chunk_num > 1 and total_chunks > 1:
            header_context = await self._get_ec_header_context()

        if total_chunks > 1:
            label = f"{name} chunk {chunk_num}/{total_chunks} ({len(text):,} chars)"
        else:
            label = f"{name} extraction ({len(text):,} chars)"

        prompt_parts = []
        if header_context:
            prompt_parts.append(
                f"EC DOCUMENT CONTEXT (from header pages):\n{header_context}\n\n"
            )
        prompt_parts.append(f"Extract all transactions from this EC section:\n\n{text}")
        prompt = "".join(prompt_parts)

        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_EC_SCHEMA,
            task_label=label,
            on_progress=on_progress,
            think=True,
        )

    async def _get_ec_header_context(self) -> str:
        """Retrieve EC header info (property, survey, SRO, period) via RAG.

        Returns formatted context string, or empty string if RAG unavailable.
        """
        rag = getattr(self, '_rag_store', None)
        efn = getattr(self, '_embed_fn', None)
        fname = getattr(self, '_filename', None)
        if not rag or not efn:
            return ""
        try:
            chunks = await rag.query(
                question=(
                    "EC header property description survey number village "
                    "SRO sub-registrar period encumbrance certificate number"
                ),
                embed_fn=efn,
                n_results=2,
                filter_filename=fname,
            )
            if chunks:
                from app.pipeline.rag_store import RAGStore
                return RAGStore.format_evidence(chunks, max_chars=3000)
        except Exception as e:
            logger.debug(f"EC header RAG query failed (non-fatal): {e}")
        return ""

    def _merge_results(self, chunk_results: list, total_pages: int) -> dict:
        """Merge transaction results from multiple chunks.

        Deduplicates transactions that span chunk boundaries by matching on
        document_number + date + parties.  When a duplicate is found, the
        version with more populated fields wins.

        After dedup, generates a stable `transaction_id` per transaction:
          EC-{doc_number}/{doc_year}-{sro}  (e.g. EC-5909/2012-Vadavalli)
          Fallback: EC-ROW{idx}  when doc_number/year unavailable
        """
        all_transactions = []
        ec_number = None
        property_description = None
        period_from = None
        period_to = None
        village = None
        taluk = None
        notes = []

        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                notes.append(f"Chunk {i + 1} failed: {str(result)}")
                continue

            if not ec_number and result.get("ec_number"):
                ec_number = result["ec_number"]
            if not property_description and result.get("property_description"):
                property_description = result["property_description"]
            if not period_from and result.get("period_from"):
                period_from = result["period_from"]
            if not period_to and result.get("period_to"):
                period_to = result["period_to"]
            if not village and result.get("village"):
                village = result["village"]
            if not taluk and result.get("taluk"):
                taluk = result["taluk"]

            transactions = result.get("transactions", [])
            all_transactions.extend(transactions)

            if result.get("extraction_notes"):
                notes.append(result["extraction_notes"])

        # ── Dedup transactions that span chunk boundaries ────────────
        deduped = _dedup_ec_transactions(all_transactions)
        dedup_count = len(all_transactions) - len(deduped)
        if dedup_count > 0:
            logger.info(f"ECExtractor: Deduped {dedup_count} overlapping transaction(s)")
            notes.append(f"Deduped {dedup_count} overlapping transaction(s) from chunk boundaries")

        # ── Assign stable transaction_id + sequential row_number ─────
        _assign_transaction_ids(deduped)

        return {
            "ec_number": ec_number,
            "property_description": property_description,
            "village": village,
            "taluk": taluk,
            "period_from": period_from,
            "period_to": period_to,
            "total_entries_found": len(deduped),
            "transactions": deduped,
            "extraction_notes": "; ".join(notes) if notes else "",
            "pages_processed": total_pages,
        }

    async def _vision_recheck_names(
        self,
        result: dict,
        file_path: "Path | None",
        on_progress: LLMProgressCallback | None = None,
    ) -> dict:
        """Targeted vision re-read for garbled Tamil party names.

        DISABLED: qwen3-vl vision model removed from pipeline.
        Tamil name quality now depends on Sarvam OCR text quality.
        Garbled names are flagged for downstream matchers instead.
        """
        # Vision re-check disabled — no vision model available
        from app.pipeline.utils import detect_garbled_tamil

        # Find and flag transactions with garbled Tamil party names
        transactions = result.get("transactions", [])
        garbled_count = 0
        for txn in transactions:
            for field in ("seller_or_executant", "buyer_or_claimant"):
                val = txn.get(field, "")
                if val and isinstance(val, str) and len(val) >= 4:
                    is_garbled, _quality, _reason = detect_garbled_tamil(val)
                    if is_garbled:
                        txn[f"_{field}_confidence"] = "low"
                        garbled_count += 1

        if garbled_count:
            logger.info(
                f"ECExtractor: {garbled_count} garbled Tamil name(s) flagged as low-confidence"
            )

        return result