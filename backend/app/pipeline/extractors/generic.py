"""Generic document extractor — text-based and vision-based.

Includes concurrent chunking for large documents to avoid context window overflow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from app.config import MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS, PROMPTS_DIR
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_GENERIC_SCHEMA

logger = logging.getLogger(__name__)

# If full_text exceeds this many chars, chunk the document
CHUNK_THRESHOLD = 40_000  # ~10K tokens


class GenericExtractor(BaseExtractor):
    """Catch-all extractor that asks LLM to extract whatever it can.
    
    Large documents are chunked by pages to stay within context limits.
    """

    SYSTEM_PROMPT = """You are an expert Tamil Nadu land document analyst.
Extract ALL relevant information from this land-related document.
Identify: names, dates, survey numbers, extent/area, amounts, document references, 
official designations, and any legal conditions or restrictions.

Return valid JSON with keys describing what you found. 
Always include: "document_summary", "key_parties", "property_details", "key_dates", "amounts", "notable_clauses"."""

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None, **kwargs) -> dict:
        full_text = extracted_text.get("full_text", "")
        pages = extracted_text.get("pages", [])
        name = filename or "document"

        # Small document — single LLM call
        if len(full_text) < CHUNK_THRESHOLD or len(pages) <= MAX_CHUNK_PAGES:
            prompt = f"Extract all relevant details from this document:\n\n{full_text}"
            return await call_llm(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                expect_json=EXTRACT_GENERIC_SCHEMA,
                task_label=f"{name} extraction ({len(full_text):,} chars)",
                on_progress=on_progress,
                think=True,  # CoT for better document understanding
            )

        # Large document — concurrent chunking
        logger.info(f"GenericExtractor: Chunking {len(pages)} pages ({len(full_text):,} chars)")
        total_chunks = (len(pages) + MAX_CHUNK_PAGES - 1) // MAX_CHUNK_PAGES
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)

        async def _extract_chunk(chunk_pages: list, chunk_num: int):
            chunk_text = "\n\n".join(
                f"--- Page {p.get('page_number', chunk_num)} ---\n{p.get('text', '')}"
                for p in chunk_pages
            )
            prompt = (
                f"Extract all relevant details from this document segment "
                f"(chunk {chunk_num}/{total_chunks}):\n\n{chunk_text}"
            )
            async with sem:
                return await call_llm(
                    prompt=prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    expect_json=EXTRACT_GENERIC_SCHEMA,
                    task_label=f"{name} chunk {chunk_num}/{total_chunks} ({len(chunk_text):,} chars)",
                    on_progress=on_progress,
                    think=True,
                )

        tasks = []
        for i in range(0, len(pages), MAX_CHUNK_PAGES):
            chunk_pages = pages[i : i + MAX_CHUNK_PAGES]
            chunk_num = i // MAX_CHUNK_PAGES + 1
            tasks.append(_extract_chunk(chunk_pages, chunk_num))

        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge chunk results
        return self._merge_chunks(chunk_results)

    @staticmethod
    def _merge_chunks(chunks: list[dict]) -> dict:
        """Merge multiple chunk results into a unified extraction."""
        merged = {
            "document_summary": "",
            "key_parties": [],
            "property_details": {},
            "key_dates": [],
            "amounts": [],
            "notable_clauses": [],
            "_chunked": True,
            "_chunk_count": len(chunks),
        }

        summaries = []
        for chunk in chunks:
            if isinstance(chunk, Exception):
                summaries.append(f"[chunk failed: {chunk}]")
                continue
            if not isinstance(chunk, dict):
                continue
            if isinstance(chunk.get("document_summary"), str):
                summaries.append(chunk["document_summary"])

            for field in ["key_parties", "key_dates", "amounts", "notable_clauses"]:
                val = chunk.get(field)
                if isinstance(val, list):
                    merged[field].extend(val)
                elif val:
                    merged[field].append(val)

            if isinstance(chunk.get("property_details"), dict):
                merged["property_details"].update(chunk["property_details"])

        merged["document_summary"] = " | ".join(summaries) if summaries else "Multiple chunks merged"

        # Deduplicate lists (preserve order)
        for field in ["key_parties", "key_dates", "amounts", "notable_clauses"]:
            seen = set()
            deduped = []
            for item in merged[field]:
                key = json.dumps(item, default=str, sort_keys=True) if isinstance(item, dict) else str(item)
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)
            merged[field] = deduped

        return merged
