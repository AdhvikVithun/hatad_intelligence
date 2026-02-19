"""Sale Deed document extractor — text-based and vision-based."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import PROMPTS_DIR, VISION_MODEL, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, call_vision_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_SALE_DEED_SCHEMA

logger = logging.getLogger(__name__)


def _pick_richer(a, b):
    """Return the richer (more populated) of two values.

    For dicts: prefer the one with more non-empty values.
    For lists: prefer the longer one.
    For scalars: prefer non-empty/non-zero.
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


def _merge_sale_deed_results(chunk_results: list, total_pages: int) -> dict:
    """Merge Sale Deed extraction results from multiple chunks.

    Unlike EC merge (concatenate transactions), Sale Deed merge picks
    the richer/non-empty value for each field since chunks overlap
    in which fields they extract with varying quality.
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
            if existing is None:
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


class SaleDeedExtractor(BaseExtractor):
    """Extract transaction and property details from Sale Deeds (text-based).

    For large documents (>MAX_CHUNK_PAGES pages), splits into page-aligned
    chunks and processes concurrently, then merges the richer values.
    """

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_sale_deed.txt").read_text(encoding="utf-8")

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None) -> dict:
        name = filename or "Sale Deed"
        pages = extracted_text.get("pages", [])
        total_pages = len(pages)

        # Small document — single-pass (existing behaviour)
        if total_pages <= MAX_CHUNK_PAGES:
            prompt = f"Extract all details from this Sale Deed:\n\n{extracted_text['full_text']}"
            return await call_llm(
                prompt=prompt,
                system_prompt=self.system_prompt,
                expect_json=EXTRACT_SALE_DEED_SCHEMA,
                task_label=f"{name} Sale Deed extraction ({len(extracted_text['full_text']):,} chars)",
                on_progress=on_progress,
                think=True,
            )

        # Large document — chunk and merge
        chunks = self._create_chunks(pages)
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)
        logger.info(f"SaleDeedExtractor: {total_pages} pages → {len(chunks)} chunks, concurrency={LLM_MAX_CONCURRENT_CHUNKS}")

        async def _limited_extract(chunk_text: str, idx: int):
            async with sem:
                return await self._extract_chunk(chunk_text, idx + 1, len(chunks), name, on_progress)

        tasks = [_limited_extract(c["text"], i) for i, c in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        return _merge_sale_deed_results(chunk_results, total_pages)

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

    async def _extract_chunk(self, text: str, chunk_num: int, total_chunks: int,
                             name: str, on_progress: LLMProgressCallback | None) -> dict:
        """Extract from a single chunk of Sale Deed text."""
        label = f"{name} chunk {chunk_num}/{total_chunks} ({len(text):,} chars)"
        prompt = (
            f"Extract all details from this section of a Sale Deed "
            f"(chunk {chunk_num} of {total_chunks}):\n\n{text}"
        )
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_SALE_DEED_SCHEMA,
            task_label=label,
            on_progress=on_progress,
            think=True,
        )


class VisionSaleDeedExtractor(BaseExtractor):
    """Extract Sale Deed details using vision model.

    Renders PDF pages as images and sends them to qwen3-vl so the model
    can read stamps, schedules, signatures, and tabular property details
    directly from the document layout.

    Falls back to text-based SaleDeedExtractor if vision is unavailable.
    """

    def __init__(self):
        self.vision_prompt = (PROMPTS_DIR / "extract_sale_deed_vision.txt").read_text(encoding="utf-8")
        self._text_fallback = SaleDeedExtractor()

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: "Path | None" = None, focus_fields: list[str] | None = None) -> dict:
        name = filename or "Sale Deed"

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
                f"Extract all details from this Sale Deed document ({len(images)} page(s)). "
                "Read carefully: registration number, date, SRO, seller/buyer names, "
                "property schedule, survey numbers, extent, boundaries, consideration amount, "
                f"stamp duty, witnesses, and any special conditions.{focus_hint}"
            )
            result = await call_vision_llm(
                prompt=prompt,
                images=images,
                system_prompt=self.vision_prompt,
                expect_json=EXTRACT_SALE_DEED_SCHEMA,
                task_label=f"{name} Vision Sale Deed extraction ({len(images)} pages)",
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
