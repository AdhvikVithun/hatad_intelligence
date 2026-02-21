"""Document classifier - identifies document type using LLM.

Text-primary: uses extracted text (from pdfplumber + Sarvam OCR) with
gpt-oss for classification.  The text model is faster and more reliable
for classification than the vision model, especially now that Sarvam
provides high-quality Tamil text extraction.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from app.config import PROMPTS_DIR
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.schemas import CLASSIFY_SCHEMA

logger = logging.getLogger(__name__)


# ── CID placeholder pattern ───────────────────────────────────────
# PDFs with embedded CID-encoded fonts produce garbled text like:
#   ப(cid:39)டா எ(cid:23): 637
# instead of பட்டா எண்: 637.  Stripping (cid:XX) reduces noise and
# makes Tamil keywords recognisable for both the LLM and heuristics.
_CID_RE = re.compile(r'\(cid:\d+\)')


def _strip_cid_placeholders(text: str) -> str:
    """Remove (cid:XX) font-encoding placeholders from pdfplumber text.

    After stripping, collapse runs of whitespace so the surviving Tamil
    characters merge back into readable tokens.
    """
    cleaned = _CID_RE.sub('', text)
    # Collapse multiple spaces (but keep newlines for structure)
    cleaned = re.sub(r'[^\S\n]+', ' ', cleaned)
    # Remove blank lines that were pure CID noise
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    return cleaned.strip()


# Pattern: 3+ consecutive repeats of a short token (1-20 chars)
_REPEAT_RE = re.compile(
    r'((?:\([^)]{1,10}\)|\S{1,20})(?:\s+|))\1{2,}',
    re.UNICODE,
)


def _collapse_repetitions(text: str, max_repeats: int = 3) -> str:
    """Collapse sequences where a short token repeats many times.

    OCR artefacts like '(R) (R) (R) (R) ...' (300 times) consume
    thousands of LLM tokens without adding information.  This collapses
    them to at most *max_repeats* copies.
    """
    def _replacer(m: re.Match) -> str:
        unit = m.group(1)
        return unit * max_repeats
    return _REPEAT_RE.sub(_replacer, text)


def _dedup_high_freq_tokens(text: str, max_occurrences: int = 6) -> str:
    """Remove excess occurrences of any token that appears too many times.

    Tamil Patta tables often have the same token (e.g. 'மொத்தம்') repeated
    across dozens of rows.  These non-adjacent repeats aren't caught by
    _collapse_repetitions but still overwhelm the LLM, causing it to
    generate runaway repetitive arrays.  This function keeps only the
    first *max_occurrences* of any token that appears more than that.
    """
    words = text.split()
    if len(words) < 30:          # Short text — don't bother
        return text
    counts: Counter = Counter(words)
    # Only target tokens that appear excessively (> max_occurrences)
    high_freq = {w for w, c in counts.items() if c > max_occurrences and len(w) >= 3}
    if not high_freq:
        return text
    seen: Counter = Counter()
    result: list[str] = []
    for w in words:
        if w in high_freq:
            seen[w] += 1
            if seen[w] > max_occurrences:
                continue
        result.append(w)
    deduped = " ".join(result)
    if len(deduped) < len(text):
        removed = len(words) - len(result)
        logger.debug(f"Deduped {removed} excess tokens from {len(high_freq)} high-freq types")
    return deduped


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# ── Quality-aware page selection ──────────────────────────────────
# Extraction method priority: Sarvam > OCR fallback > pdfplumber
_METHOD_PRIORITY = {
    "sarvam": 0,
    "ocr_fallback": 1,
    "ocr": 1,
    "pdfplumber": 2,
    "failed": 3,
}


def _page_sort_key(page: dict) -> tuple:
    """Sort key that ranks pages by quality then extraction method.

    Lower = better.  Returns (quality_rank, method_rank, page_number).
    Ties broken by page_number so earlier pages are preferred.
    """
    quality = page.get("quality", {}).get("quality", "LOW")
    quality_rank = 0 if quality == "HIGH" else 1
    method = page.get("extraction_method", "pdfplumber")
    method_rank = _METHOD_PRIORITY.get(method, 2)
    page_num = page.get("page_number", 999)
    return (quality_rank, method_rank, page_num)


def _select_best_pages(pages: list[dict], max_pages: int = 2) -> list[dict]:
    """Pick the best *max_pages* pages for classification.

    Strategy:
      1. Sort by quality (HIGH first), then by source (Sarvam > OCR > pdfplumber).
      2. Return up to *max_pages* pages, re-sorted by page number so the
         LLM sees them in document order.

    This avoids feeding garbled pdfplumber pages to the classifier when
    cleaner Sarvam/OCR alternatives exist for other pages.
    """
    if len(pages) <= max_pages:
        return pages

    ranked = sorted(pages, key=_page_sort_key)
    best = ranked[:max_pages]
    # Re-sort by page number for natural reading order
    best.sort(key=lambda p: p.get("page_number", 0))
    return best


async def classify_document(
    extracted_text: dict,
    filename: str = "",
    on_progress: LLMProgressCallback | None = None,
    file_path: Path | None = None,
) -> dict:
    """Classify a document using the text LLM (gpt-oss).

    Uses extracted text (pdfplumber + Sarvam OCR) for classification.
    This is faster and more reliable than vision-based classification.

    Args:
        extracted_text: Output from ingestion.extract_text_from_pdf()
        filename: Original filename for labeling
        on_progress: Async callback for LLM progress updates
        file_path: Path to original PDF (unused — kept for API compat)

    Returns:
        {"document_type": "EC", "confidence": 0.95, ...}
    """
    name_part = filename or "document"
    total_pages = extracted_text.get("total_pages", "?")

    # Select the best 2 pages for classification.
    # Prefer Sarvam-OCR'd pages (high-quality Tamil) over raw pdfplumber,
    # and HIGH-quality pages over LOW/garbled ones.  This ensures the
    # classifier sees clean text even when some pages are CID-garbled.
    pages = _select_best_pages(extracted_text.get("pages", []), max_pages=2)
    sample_text = "\n\n".join(p["text"] for p in pages if p.get("text"))

    # Strip (cid:XX) font-encoding placeholders — these indicate the PDF
    # uses CID-encoded fonts that pdfplumber cannot resolve.  Removing
    # them turns garbled "ப(cid:39)டா எ(cid:23):" into "படா எ:" which,
    # while imperfect, lets Tamil keywords become recognisable.
    sample_text = _strip_cid_placeholders(sample_text)

    # Collapse excessive repetitions (OCR artefacts like "(R) (R) (R)...")
    sample_text = _collapse_repetitions(sample_text)

    # Remove high-frequency non-adjacent repeated tokens (Tamil table rows)
    sample_text = _dedup_high_freq_tokens(sample_text)

    # Cap sample size — classification only needs header/title text.
    # Large pages (e.g. 68-page EC with 168K chars from 2 pages) overwhelm
    # the 64K context window, leaving no room for output tokens.
    _CLASSIFY_MAX_CHARS = 4000
    if len(sample_text) > _CLASSIFY_MAX_CHARS:
        logger.info(
            f"[{name_part}] Classification sample truncated: "
            f"{len(sample_text):,} → {_CLASSIFY_MAX_CHARS:,} chars"
        )
        sample_text = sample_text[:_CLASSIFY_MAX_CHARS]

    if not sample_text.strip():
        return {
            "document_type": "OTHER",
            "confidence": 0.0,
            "language": "unknown",
            "key_identifiers": [],
            "reasoning": "No text could be extracted for classification.",
        }

    system_prompt = _load_prompt("classify")
    result = await call_llm(
        prompt=f"Classify this document:\n\n{sample_text}",
        system_prompt=system_prompt,
        expect_json=CLASSIFY_SCHEMA,
        task_label=f"Classify {name_part} ({total_pages}p, {len(sample_text):,} chars)",
        on_progress=on_progress,
        think=True,
        max_tokens=2048,
    )

    from app.config import DOCUMENT_TYPES
    if result.get("document_type") not in DOCUMENT_TYPES:
        result["document_type"] = "OTHER"

    # Defensive: cap & deduplicate key_identifiers
    ids = result.get("key_identifiers", [])
    if ids:
        seen = set()
        unique: list[str] = []
        for v in ids:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        result["key_identifiers"] = unique[:10]

    logger.info(f"[{name_part}] Classification: {result.get('document_type')} "
                f"(confidence: {result.get('confidence', 0):.0%})")
    return result
