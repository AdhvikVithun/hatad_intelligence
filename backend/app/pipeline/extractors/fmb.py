"""FMB (Field Measurement Book) extractor — text-based.

Includes a deterministic header pre-parser that anchors the subject
survey number from the FMB header ("Survey No : 317") so the LLM
doesn't confuse it with adjacent survey numbers from the sketch body.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from app.config import PROMPTS_DIR, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_FMB_SCHEMA

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# DETERMINISTIC FMB HEADER PRE-PARSER
# ═══════════════════════════════════════════════════

# "Survey No : 317" / "Survey No. : 317/1A" / "S.F.No. : 317"
# Tamil: "சர்வே எண் : 317" / "புல எண் : 317"
_RE_SURVEY_NO = re.compile(
    r'(?:'
    r'Survey\s*No\.?\s*[:\-]\s*|'
    r'S\.?F\.?\s*No\.?\s*[:\-]\s*|'
    r'T\.?S\.?\s*No\.?\s*[:\-]\s*|'
    r'R\.?S\.?\s*No\.?\s*[:\-]\s*|'
    r'சர்வே\s*(?:எண்|நம்பர்|விவரம்)\s*[:\-]\s*|'
    r'புல\s*எண்\s*[:\-]\s*'
    r')'
    r'([A-Za-z0-9\u0BE6-\u0BEF/\-\.]+)',
    re.IGNORECASE,
)

# "Village : Somayampalayam.(R)" / "கிராமம் : சோமயம்பாளையம்"
_RE_VILLAGE = re.compile(
    r'(?:Village\s*[:\-]\s*|கிராமம்\s*[:\-]\s*)([\w\s\.\(\)\u0B80-\u0BFF]+?)(?:\s{2,}|\n|$)',
    re.IGNORECASE,
)

# "Taluk : Coimbatore (N)" / "வட்டம் : கோயம்புத்தூர்"
_RE_TALUK = re.compile(
    r'(?:Taluk\s*[:\-]\s*|வட்டம்\s*[:\-]\s*)([\w\s\.\(\)\u0B80-\u0BFF]+?)(?:\s{2,}|\n|$)',
    re.IGNORECASE,
)

# "District : Coimbatore" / "மாவட்டம் : கோயம்புத்தூர்"
_RE_DISTRICT = re.compile(
    r'(?:District\s*[:\-]\s*|மாவட்டம்\s*[:\-]\s*)([\w\s\.\(\)\u0B80-\u0BFF]+?)(?:\s{2,}|\n|$)',
    re.IGNORECASE,
)

# "Area : Hect 00 Ares 92 Sqm 50" → "Hect 00 Ares 92 Sqm 50"
_RE_AREA = re.compile(
    r'Area\s*[:\-]\s*(.*?)(?:\s{2,}|\n|$)',
    re.IGNORECASE,
)


def preparse_fmb_header(text: str) -> dict:
    """Extract key fields deterministically from FMB header text.

    FMB headers have a standard format with labelled fields.
    This pre-parser anchors the subject survey number so the LLM
    doesn't pick up an adjacent survey number from the sketch body.

    Only examines the top ~800 characters (header region).
    """
    hints: dict = {}

    # Use only top portion of text (header typically in first ~800 chars)
    header_text = text[:800]

    m = _RE_SURVEY_NO.search(header_text)
    if m:
        hints["survey_number"] = m.group(1).strip().rstrip(".")

    m = _RE_VILLAGE.search(header_text)
    if m:
        hints["village"] = m.group(1).strip().rstrip(".")

    m = _RE_TALUK.search(header_text)
    if m:
        hints["taluk"] = m.group(1).strip().rstrip(".")

    m = _RE_DISTRICT.search(header_text)
    if m:
        hints["district"] = m.group(1).strip().rstrip(".")

    m = _RE_AREA.search(header_text)
    if m:
        hints["area_raw"] = m.group(1).strip()

    if hints:
        logger.info(f"FMB header pre-parse: {hints}")

    return hints


def _apply_header_overrides(result: dict, header: dict) -> dict:
    """Override LLM extraction with deterministic header values when they differ.

    The FMB header survey number is ground truth — it's printed in a
    standard format and is very reliably OCR'd.  If the LLM extracted a
    different survey number (likely an adjacent survey from the sketch),
    we override it and log a warning.
    """
    if not header:
        return result

    header_survey = header.get("survey_number")
    if header_survey:
        llm_survey = (result.get("survey_number") or "").strip()
        if llm_survey and llm_survey != header_survey:
            logger.warning(
                f"FMB survey override: LLM extracted '{llm_survey}' but header says "
                f"'{header_survey}' — using header value (LLM likely picked up "
                f"an adjacent survey from sketch body)"
            )
            # Save LLM's answer as a red flag for transparency
            flags = result.get("_extraction_red_flags") or []
            flags.append(
                f"Survey number corrected: LLM extracted '{llm_survey}' but FMB header "
                f"clearly shows '{header_survey}'. LLM value was likely an adjacent survey."
            )
            result["_extraction_red_flags"] = flags
            result["survey_number"] = header_survey
        elif not llm_survey:
            result["survey_number"] = header_survey

    # Override location fields only if LLM left them empty
    for key in ("village", "taluk", "district"):
        header_val = header.get(key)
        if header_val and not (result.get(key) or "").strip():
            result[key] = header_val

    return result


class FMBExtractor(BaseExtractor):
    """Extract survey geometry and boundary data from FMB sketches."""

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_fmb.txt").read_text(encoding="utf-8")

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
        **kwargs,
    ) -> dict:
        name = filename or "FMB"
        pages = extracted_text.get("pages", [])
        full_text = extracted_text.get("full_text", "")

        # ── Deterministic header pre-parse ──
        header = preparse_fmb_header(full_text)

        if len(pages) <= MAX_CHUNK_PAGES:
            result = await self._extract_single(
                full_text, name, on_progress, header=header,
            )
            return _apply_header_overrides(result, header)

        # Multi-page: chunk and merge
        chunks = self._create_chunks(pages)
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)

        async def _limited(chunk_text: str, idx: int):
            async with sem:
                return await self._extract_single(
                    chunk_text, f"{name} chunk {idx+1}/{len(chunks)}",
                    on_progress, header=header,
                )

        tasks = [_limited(c, i) for i, c in enumerate(chunks)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        merged = self._merge(results)
        return _apply_header_overrides(merged, header)

    async def _extract_single(
        self, text: str, label: str, on_progress,
        header: dict | None = None,
    ) -> dict:
        # Inject header hint so the LLM knows the correct survey number
        header_hint = ""
        if header and header.get("survey_number"):
            header_hint = (
                f"IMPORTANT — The FMB header says Survey No: {header['survey_number']}. "
                f"Use THIS as the survey_number field. Do NOT use adjacent survey "
                f"numbers from the sketch body.\n\n"
            )
        prompt = f"{header_hint}Extract all details from this FMB document:\n\n{text}"
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_FMB_SCHEMA,
            task_label=f"{label} extraction ({len(text):,} chars)",
            on_progress=on_progress,
            think=False,  # Simple sketch/table lookup — no CoT needed
        )

    @staticmethod
    def _create_chunks(pages: list[dict]) -> list[str]:
        chunks = []
        for i in range(0, len(pages), MAX_CHUNK_PAGES):
            chunk_pages = pages[i : i + MAX_CHUNK_PAGES]
            chunks.append(
                "\n\n".join(
                    f"--- PAGE {p['page_number']} ---\n{p['text']}"
                    for p in chunk_pages
                )
            )
        return chunks

    @staticmethod
    def _merge(results: list) -> dict:
        merged: dict = {}
        for r in results:
            if isinstance(r, Exception) or not isinstance(r, dict):
                continue
            for k, v in r.items():
                if v is None or v == "" or v == [] or v == {}:
                    continue
                existing = merged.get(k)
                if existing is None or existing == "" or existing == [] or existing == {}:
                    merged[k] = v
                elif isinstance(existing, list) and isinstance(v, list):
                    merged[k] = existing + v
        return merged
