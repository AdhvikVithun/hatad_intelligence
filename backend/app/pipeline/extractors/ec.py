"""EC (Encumbrance Certificate) extractor - handles large multi-page ECs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from app.config import PROMPTS_DIR, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_EC_SCHEMA

logger = logging.getLogger(__name__)


def _txn_dedup_key(txn: dict) -> str:
    """Build a deduplication key from a transaction's immutable fields.

    Uses document_number + date + parties (normalized) so that the same
    transaction extracted from overlapping chunks is detected as a duplicate.
    """
    doc_no = str(txn.get("document_number", "")).strip().lower()
    date = str(txn.get("date", "")).strip()
    seller = str(txn.get("seller_or_executant", "")).strip().lower()[:30]
    buyer = str(txn.get("buyer_or_claimant", "")).strip().lower()[:30]
    return f"{doc_no}|{date}|{seller}|{buyer}"


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


class ECExtractor(BaseExtractor):
    """Extract transactions from Encumbrance Certificates.
    
    Handles large ECs (65+ pages) by chunking with concurrency-limited processing.
    """

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_ec.txt").read_text(encoding="utf-8")

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None) -> dict:
        """Extract all EC transactions via chunked LLM processing."""
        pages = extracted_text["pages"]
        total_pages = len(pages)
        self._filename = filename or "EC"

        if total_pages <= MAX_CHUNK_PAGES:
            # Small EC - process in one shot
            result = await self._extract_chunk(extracted_text["full_text"], 1, 1, on_progress)
        else:
            # Large EC — chunk with concurrency limiter to avoid overloading Ollama
            chunks = self._create_chunks(pages)
            sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)
            logger.info(f"ECExtractor: {len(chunks)} chunks, concurrency={LLM_MAX_CONCURRENT_CHUNKS}")

            async def _limited_extract(chunk_text: str, idx: int):
                async with sem:
                    return await self._extract_chunk(chunk_text, idx + 1, len(chunks), on_progress)

            tasks = [_limited_extract(chunk["text"], i) for i, chunk in enumerate(chunks)]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge all chunk results
            result = self._merge_results(chunk_results, total_pages)

        # ── Post-extraction: targeted Qwen vision re-read for garbled names ──
        result = await self._vision_recheck_names(result, file_path, on_progress)

        return result

    def _create_chunks(self, pages: list[dict]) -> list[dict]:
        """Split pages into processable chunks."""
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

    async def _extract_chunk(self, text: str, chunk_num: int = 1, total_chunks: int = 1, on_progress: LLMProgressCallback | None = None) -> dict:
        """Extract transactions from a single chunk of EC text."""
        name = getattr(self, '_filename', 'EC')
        if total_chunks > 1:
            label = f"{name} chunk {chunk_num}/{total_chunks} ({len(text):,} chars)"
        else:
            label = f"{name} extraction ({len(text):,} chars)"
        prompt = f"Extract all transactions from this EC section:\n\n{text}"
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_EC_SCHEMA,
            task_label=label,
            on_progress=on_progress,
            think=True,  # CoT helps with OCR artifacts and Tamil text
        )

    def _merge_results(self, chunk_results: list, total_pages: int) -> dict:
        """Merge transaction results from multiple chunks.

        Deduplicates transactions that span chunk boundaries by matching on
        document_number + date + parties.  When a duplicate is found, the
        version with more populated fields wins.
        """
        all_transactions = []
        ec_number = None
        property_description = None
        period_from = None
        period_to = None
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

        # Re-number transactions sequentially
        for idx, txn in enumerate(deduped, 1):
            txn["row_number"] = idx

        return {
            "ec_number": ec_number,
            "property_description": property_description,
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