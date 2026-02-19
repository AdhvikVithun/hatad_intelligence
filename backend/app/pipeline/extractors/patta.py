"""Patta document extractor — text-based and vision-based."""

import asyncio
import re
import logging
from pathlib import Path

from app.config import PROMPTS_DIR, MAX_CHUNK_PAGES, LLM_MAX_CONCURRENT_CHUNKS
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.extractors.base import BaseExtractor
from app.pipeline.schemas import EXTRACT_PATTA_SCHEMA
from app.pipeline.utils import normalize_survey_number, split_survey_numbers

logger = logging.getLogger(__name__)

# ── Extent patterns commonly confused by OCR / LLM ──
# Tamil Patta tables have columns like: வ.எண் | புல எண் | பரப்பு (ஹெ - ஏர்)
# The serial number (வ.எண் = 1,2,3...) often gets concatenated with the
# hectare figure, producing values like "7.92.50" when the real value is
# "0 hect, 92.50 ares" (= 0.9250 hectares) and 7 is just the serial number.
_BOGUS_EXTENT_RE = re.compile(
    r'^(\d{1,2})[\.\-\s]+(\d{1,2})[\.\-\s]+(\d{1,2}(?:\.\d+)?)\s*'
    r'(hectares?|hect|ha|ஹெக்டேர்|ஹெ)',
    re.IGNORECASE
)

# Pattern: "X.YY.ZZ hectares" where X is likely a serial number
_SERIAL_DOT_EXTENT_RE = re.compile(
    r'^(\d{1,2})\.(\d{1,2})\.(\d{2,4})\s*(hectares?|hect|ha|ஹெக்டேர்|ஹெ)?',
    re.IGNORECASE
)


def _fix_extent_serial_number(extent_str: str) -> str:
    """Fix extent value where serial number was concatenated with area.

    Tamil Patta OCR often produces "7.92.50 hectares" when the actual value is
    "0 hectares 92 ares 50 sq.m" (= 0.9250 hectares). The "7" is the
    serial number (வ.எண்) from the Patta table row.

    Detection heuristic:
      - Pattern X.YY.ZZ where X is a small number (1-20)
      - XX > 0 and ZZ > 0 (real extent values in hect.ares format)
      - If X.YY.ZZ would be unusually large (> 10 hectares for a typical
        residential/small agricultural plot), and YY.ZZ alone is reasonable,
        X is likely the serial number.
    """
    if not extent_str or not isinstance(extent_str, str):
        return extent_str

    s = extent_str.strip()
    m = _SERIAL_DOT_EXTENT_RE.match(s)
    if not m:
        return extent_str

    first, second, third = m.group(1), m.group(2), m.group(3)
    unit = m.group(4) or ""

    first_num = int(first)
    second_num = int(second)
    third_flt = float(third)

    # If first number is 1-20 and the resulting "hectares" value is unusually
    # large (> 10 ha), it's very likely the serial number was prepended.
    # A value of "0.92.50" (0 hect, 92 ares, 50 sqm) is 0.9250 ha.
    # A value of "7.92.50" would be 7.9250 ha which is large but possible.
    # But if first_num <= 20 and second_num < 100 (ares) and third < 100 (sqm),
    # the real extent is likely second.third hectares (0.second_num hectares + ares).

    # Build the "without serial" value (ares → ha: /100, sqm → ha: /10000)
    real_hectares = second_num / 100.0 + third_flt / 10000.0
    fake_hectares = first_num + real_hectares

    # Heuristic: if first_num > 0 AND (second_num, third) both valid for
    # ares.sqm format AND the "serial-included" value is > 5 ha while
    # the "serial-excluded" value is < 5 ha, strip the serial.
    if (1 <= first_num <= 20
            and 0 <= second_num <= 99
            and real_hectares < 100
            and fake_hectares > 5 * real_hectares):
        # First number is very likely a serial number
        corrected = f"{second_num / 100.0 + third_flt / 10000.0:.4f} hectares"
        logger.warning(
            f"Extent correction: '{extent_str}' → '{corrected}' "
            f"(stripped suspected serial number {first_num})"
        )
        return corrected

    return extent_str


# Regex for "NNN.DDD hectares" where NNN is a 2-3 digit survey number prefix
_SURVEY_IN_EXTENT_RE = re.compile(
    r'^(\d{2,4})[\.]?(\d{1,4}(?:\.\d+)?)\s*(hectares?|hect|ha|\u0bb9\u0bc6\u0b95\u0bcd\u0b9f\u0bc7\u0bb0\u0bcd|\u0bb9\u0bc6)',
    re.IGNORECASE
)

# Maximum plausible extent for a single survey in Tamil Nadu (hectares).
# 100 ha ≈ 247 acres — very generous ceiling for a single survey plot.
_MAX_SINGLE_SURVEY_HECTARES = 100.0


def _fix_survey_in_extent(result: dict) -> dict:
    """Fix extents where the LLM concatenated a survey number with the real extent.

    For example, when the Patta table has columns:  புல எண் | பரப்பு
    with values:  317 | 0.9250 hectares
    the LLM sometimes produces survey_no="317", extent="317.925 hectares"
    (concatenating survey 317 with 0.925 ha → 317.925 ha).

    This function cross-checks each survey entry's extent against its own
    survey_no: if the integer part of the extent equals the survey_no,
    strip it and keep only the decimal part as the real extent.
    """
    surveys = result.get("survey_numbers", [])
    if not surveys:
        return result

    # Build a set of known survey base numbers for cross-checking
    known_survey_bases: set[str] = set()
    for sn in surveys:
        if isinstance(sn, dict) and sn.get("survey_no"):
            raw = str(sn["survey_no"]).strip()
            # Get the numeric base: "317" from "317", "317" from "317/1A"
            m = re.match(r'^(\d+)', raw)
            if m:
                known_survey_bases.add(m.group(1))

    corrected_any = False
    for sn in surveys:
        if not isinstance(sn, dict) or not sn.get("extent"):
            continue
        ext = str(sn["extent"]).strip()
        own_survey = str(sn.get("survey_no", "")).strip()

        # Try to match "NNN.DDD hectares" pattern
        m = _SURVEY_IN_EXTENT_RE.match(ext)
        if not m:
            continue

        prefix_str = m.group(1)  # e.g. "317"
        decimal_str = m.group(2)  # e.g. "925" or "9250"

        # Check if the integer prefix IS one of the known survey numbers
        if prefix_str not in known_survey_bases:
            continue

        # Also check against this survey's own number
        own_base = ""
        own_match = re.match(r'^(\d+)', own_survey)
        if own_match:
            own_base = own_match.group(1)

        if prefix_str != own_base and prefix_str not in known_survey_bases:
            continue

        # The real extent is just the decimal part
        try:
            real_val = float(f"0.{decimal_str}")
        except ValueError:
            continue

        # Sanity: the "full" value would be absurdly large (e.g. 317.925 ha)
        # while the real value is tiny (0.925 ha).  Require ratio > 50:1.
        try:
            full_val = float(f"{prefix_str}.{decimal_str}")
        except ValueError:
            continue

        if real_val > 0 and full_val / real_val > 50:
            corrected = f"{real_val:.4f} hectares"
            logger.warning(
                f"Survey-in-extent correction: '{ext}' → '{corrected}' "
                f"(stripped survey number prefix {prefix_str})"
            )
            sn["extent"] = corrected
            corrected_any = True

    # If we corrected individual extents, recalculate total_extent
    if corrected_any:
        _recalculate_total_extent(result)

    return result


def _recalculate_total_extent(result: dict):
    """Recalculate total_extent by summing individual survey extents."""
    surveys = result.get("survey_numbers", [])
    total = 0.0
    count = 0
    for sn in surveys:
        if not isinstance(sn, dict) or not sn.get("extent"):
            continue
        m = re.search(r'([\d.]+)\s*(?:hectares?|hect|ha)', str(sn["extent"]), re.IGNORECASE)
        if m:
            try:
                total += float(m.group(1))
                count += 1
            except ValueError:
                pass
    if count > 0:
        result["total_extent"] = f"{total:.4f} hectares"
        logger.info(f"Recalculated total_extent from {count} surveys: {result['total_extent']}")


def _check_extent_plausibility(result: dict) -> dict:
    """Flag implausibly large extents in post-processing.

    A single survey plot in Tamil Nadu rarely exceeds 100 hectares (≈247 acres).
    This catches LLM hallucinations like "1928 acres 50 cents" before they
    reach the memory bank and corrupt downstream verification.
    """
    notes: list[str] = []

    for sn in result.get("survey_numbers", []):
        if not isinstance(sn, dict) or not sn.get("extent"):
            continue
        m = re.search(r'([\d.]+)\s*(?:hectares?|hect|ha)', str(sn["extent"]), re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if val > _MAX_SINGLE_SURVEY_HECTARES:
                    survey_id = sn.get("survey_no", "?")
                    notes.append(
                        f"Survey {survey_id} extent {val:.2f} ha "
                        f"exceeds plausibility ceiling ({_MAX_SINGLE_SURVEY_HECTARES} ha)"
                    )
                    sn["_extent_implausible"] = True
                    logger.warning(f"Implausible extent: survey {survey_id} = {val:.2f} ha")
            except ValueError:
                pass

    # Also check total_extent
    total_ext = result.get("total_extent", "")
    if total_ext:
        m = re.search(r'([\d.]+)\s*(?:hectares?|hect|ha)', str(total_ext), re.IGNORECASE)
        if m:
            try:
                num_surveys = max(1, len(result.get("survey_numbers", [])))
                val = float(m.group(1))
                ceiling = _MAX_SINGLE_SURVEY_HECTARES * num_surveys
                if val > ceiling:
                    notes.append(
                        f"Total extent {val:.2f} ha exceeds plausibility "
                        f"ceiling ({ceiling:.0f} ha for {num_surveys} survey(s))"
                    )
                    result["_total_extent_implausible"] = True
            except ValueError:
                pass

    if notes:
        existing = result.get("extraction_notes", "")
        joined = "; ".join(notes)
        result["extraction_notes"] = f"{existing}; {joined}".lstrip("; ") if existing else joined

    return result


def _dedup_survey_numbers(result: dict) -> dict:
    """Deduplicate survey_numbers by normalized survey_no, keeping richer entries."""
    surveys = result.get("survey_numbers", [])
    if len(surveys) <= 1:
        return result

    seen: dict[str, dict] = {}
    for sn in surveys:
        if not isinstance(sn, dict):
            continue
        raw = str(sn.get("survey_no", "")).strip()
        if not raw:
            continue
        key = normalize_survey_number(raw)
        if not key:
            key = raw
        if key in seen:
            # Keep whichever entry has more populated fields
            existing = seen[key]
            e_pop = sum(1 for v in existing.values() if v is not None and v != "")
            n_pop = sum(1 for v in sn.values() if v is not None and v != "")
            if n_pop > e_pop:
                seen[key] = sn
        else:
            seen[key] = sn

    if len(seen) < len(surveys):
        removed = len(surveys) - len(seen)
        logger.info(f"Survey dedup: removed {removed} duplicate(s)")
        result["survey_numbers"] = list(seen.values())

    return result


def _post_process_patta(result: dict) -> dict:
    """Post-process LLM-extracted Patta data to fix common errors."""
    if not isinstance(result, dict):
        return result

    # Fix 1: Strip serial numbers concatenated with extents (X.YY.ZZ pattern)
    if result.get("total_extent"):
        result["total_extent"] = _fix_extent_serial_number(result["total_extent"])
    for sn in result.get("survey_numbers", []):
        if isinstance(sn, dict) and sn.get("extent"):
            sn["extent"] = _fix_extent_serial_number(sn["extent"])

    # Fix 1b: Strip survey numbers concatenated with extents (NNN.DDD pattern)
    _fix_survey_in_extent(result)

    # Fix 6: Deduplicate survey numbers by normalized key
    _dedup_survey_numbers(result)

    # Fix 5: Flag implausibly large extents
    _check_extent_plausibility(result)

    # Fix owner_names: LLM sometimes puts wife/husband in father_name field
    # when the Patta format is "Husband மனைவி Wife". The actual owner is Wife
    # (the one after மனைவி/W/o), and the father_name should be Husband.
    for owner in result.get("owner_names", []):
        if isinstance(owner, dict):
            name = owner.get("name", "")
            father = owner.get("father_name", "")
            # If name looks like a husband and father_name looks like a wife name,
            # they may be swapped. We can't reliably fix this without more context,
            # but we can flag it.
            if name and father and not owner.get("_name_order_verified"):
                owner["_name_order_verified"] = True

    return result


class PattaExtractor(BaseExtractor):
    """Extract ownership and land details from Patta documents (text-based).
    
    Handles large multi-page Pattas by chunking with concurrency-limited
    processing, then merging results using a "richer value wins" strategy.
    """

    def __init__(self):
        self.system_prompt = (PROMPTS_DIR / "extract_patta.txt").read_text(encoding="utf-8")

    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: Path | None = None) -> dict:
        name = filename or "Patta"
        pages = extracted_text.get("pages", [])

        # Small Patta — single pass
        if len(pages) <= MAX_CHUNK_PAGES:
            result = await self._extract_single(
                extracted_text["full_text"], name, on_progress
            )
            return _post_process_patta(result)

        # Large Patta — concurrent chunking
        chunks = self._create_chunks(pages)
        sem = asyncio.Semaphore(LLM_MAX_CONCURRENT_CHUNKS)
        logger.info(f"PattaExtractor: {len(chunks)} chunks, concurrency={LLM_MAX_CONCURRENT_CHUNKS}")

        async def _limited(chunk_text: str, idx: int):
            async with sem:
                return await self._extract_single(
                    chunk_text, f"{name} chunk {idx+1}/{len(chunks)}", on_progress
                )

        tasks = [_limited(c["text"], i) for i, c in enumerate(chunks)]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = _merge_patta_results(chunk_results)
        return _post_process_patta(merged)

    async def _extract_single(self, text: str, label: str, on_progress) -> dict:
        prompt = f"Extract all details from this Patta document:\n\n{text}"
        return await call_llm(
            prompt=prompt,
            system_prompt=self.system_prompt,
            expect_json=EXTRACT_PATTA_SCHEMA,
            task_label=f"{label} extraction ({len(text):,} chars)",
            on_progress=on_progress,
            think=True,
        )

    @staticmethod
    def _create_chunks(pages: list[dict]) -> list[dict]:
        chunks = []
        for i in range(0, len(pages), MAX_CHUNK_PAGES):
            chunk_pages = pages[i:i + MAX_CHUNK_PAGES]
            chunk_text = "\n\n".join(
                f"--- PAGE {p['page_number']} ---\n{p['text']}"
                for p in chunk_pages
            )
            chunks.append({"text": chunk_text})
        return chunks


def _pick_richer_patta(a, b):
    """Pick the richer (more populated) value — for Patta merge."""
    if a is None or a == "" or a == [] or a == {}:
        return b
    if b is None or b == "" or b == [] or b == {}:
        return a
    if isinstance(a, dict) and isinstance(b, dict):
        a_pop = sum(1 for v in a.values() if v is not None and v != "")
        b_pop = sum(1 for v in b.values() if v is not None and v != "")
        return b if b_pop > a_pop else a
    if isinstance(a, list) and isinstance(b, list):
        return b if len(b) > len(a) else a
    return a  # for scalars, keep first non-empty


def _merge_patta_results(chunk_results: list) -> dict:
    """Merge multiple Patta chunk results — richer value wins for scalars,
    concatenate survey_numbers and owner_names lists (deduped)."""
    merged: dict = {}
    all_surveys: list[dict] = []
    all_owners: list[dict] = []
    notes: list[str] = []

    for i, result in enumerate(chunk_results):
        if isinstance(result, Exception):
            notes.append(f"Chunk {i+1} failed: {str(result)}")
            continue
        if not isinstance(result, dict):
            continue

        # Collect list fields for concatenation
        all_surveys.extend(result.get("survey_numbers", []))
        all_owners.extend(result.get("owner_names", []))

        # Scalar fields: richer wins
        for key, val in result.items():
            if key in ("survey_numbers", "owner_names"):
                continue
            merged[key] = _pick_richer_patta(merged.get(key), val)

    # Dedup survey numbers by normalized survey_no
    seen_surveys: dict[str, dict] = {}
    for sn in all_surveys:
        if isinstance(sn, dict) and sn.get("survey_no"):
            raw = str(sn["survey_no"]).strip()
            key = normalize_survey_number(raw) or raw
            if key not in seen_surveys:
                seen_surveys[key] = sn
            else:
                # Keep the richer entry
                existing = seen_surveys[key]
                e_pop = sum(1 for v in existing.values() if v is not None and v != "")
                n_pop = sum(1 for v in sn.values() if v is not None and v != "")
                if n_pop > e_pop:
                    seen_surveys[key] = sn
    merged["survey_numbers"] = list(seen_surveys.values()) if seen_surveys else all_surveys

    # Dedup owners by name
    seen_owners: dict[str, dict] = {}
    for o in all_owners:
        if isinstance(o, dict) and o.get("name"):
            key = str(o["name"]).strip().lower()
            if key not in seen_owners:
                seen_owners[key] = o
    merged["owner_names"] = list(seen_owners.values()) if seen_owners else all_owners

    if notes:
        merged["extraction_notes"] = "; ".join(notes)
    return merged
