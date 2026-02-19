"""Generic document extractor — text-based and vision-based.

Includes concurrent chunking for large documents to avoid context window overflow.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from app.config import MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS, PROMPTS_DIR, VISION_MODEL
from app.pipeline.llm_client import call_llm, call_vision_llm, LLMProgressCallback
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

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None) -> dict:
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


class VisionGenericExtractor(BaseExtractor):
    """Vision-based catch-all extractor for all non-EC, non-Patta, non-SaleDeed document types.

    Renders PDF pages as images and sends them to the vision model so it can
    read tables, stamps, handwritten notes, and complex layouts directly.
    Covers: ADANGAL, FMB, LAYOUT_APPROVAL, LEGAL_HEIR, POA, COURT_ORDER,
            WILL, PARTITION_DEED, GIFT_DEED, RELEASE_DEED, OTHER.

    Falls back to text-based GenericExtractor if vision is unavailable.
    """

    def __init__(self):
        self.vision_prompt = (PROMPTS_DIR / "extract_generic_vision.txt").read_text(encoding="utf-8")
        self._text_fallback = GenericExtractor()

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None, focus_fields: list[str] | None = None) -> dict:
        name = filename or "document"

        if not file_path or not Path(file_path).exists():
            logger.warning(f"[{name}] No file_path for vision extraction, falling back to text")
            return await self._text_fallback.extract(extracted_text, on_progress, filename, file_path)

        try:
            from app.pipeline.llm_client import check_ollama_status
            status = await check_ollama_status()
            if not any(VISION_MODEL in m for m in status.get("models", [])):
                logger.warning(f"[{name}] Vision model {VISION_MODEL} not available, falling back to text")
                return await self._text_fallback.extract(extracted_text, on_progress, filename, file_path)

            from app.pipeline.ingestion import render_pages_as_images
            images = render_pages_as_images(Path(file_path))
            if not images:
                logger.warning(f"[{name}] No images rendered, falling back to text")
                return await self._text_fallback.extract(extracted_text, on_progress, filename, file_path)

            # Build prompt — add focus-field hint when acting as a fallback helper
            focus_hint = ""
            if focus_fields:
                fields_str = ", ".join(focus_fields)
                focus_hint = (
                    f" The text-based extraction had LOW confidence for these fields: "
                    f"{fields_str}. Pay EXTRA attention to extracting accurate values for them."
                )

            prompt = (
                f"Extract all relevant details from this Tamil Nadu land document ({len(images)} page(s)). "
                "Identify the document type, all parties, property details, survey numbers, "
                f"extents, dates, amounts, legal conditions, and any other relevant information.{focus_hint}"
            )
            result = await call_vision_llm(
                prompt=prompt,
                images=images,
                system_prompt=self.vision_prompt,
                expect_json=EXTRACT_GENERIC_SCHEMA,
                task_label=f"{name} Vision extraction ({len(images)} pages)",
                on_progress=on_progress,
            )
            logger.info(f"[{name}] Vision extraction successful")
            return result

        except Exception as e:
            logger.warning(f"[{name}] Vision extraction failed ({e}), falling back to text")
            # Lazy OCR: enhance text quality before text fallback (Sarvam replaces Tesseract)
            if file_path:
                from app.pipeline.sarvam_ocr import run_sarvam_on_pages
                extracted_text = await run_sarvam_on_pages(Path(file_path), extracted_text)
            return await self._text_fallback.extract(extracted_text, on_progress, filename, file_path)
