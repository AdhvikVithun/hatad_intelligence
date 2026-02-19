"""Document classifier - identifies document type using LLM.

Text-primary: uses extracted text (from pdfplumber + Sarvam OCR) with
gpt-oss for classification.  The text model is faster and more reliable
for classification than the vision model, especially now that Sarvam
provides high-quality Tamil text extraction.
"""

import logging
import re
from pathlib import Path
from app.config import PROMPTS_DIR
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.schemas import CLASSIFY_SCHEMA

logger = logging.getLogger(__name__)


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


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


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

    # Take first 2 pages of text for classification
    pages = extracted_text.get("pages", [])[:2]
    sample_text = "\n\n".join(p["text"] for p in pages if p.get("text"))

    # Collapse excessive repetitions (OCR artefacts like "(R) (R) (R)...")
    sample_text = _collapse_repetitions(sample_text)

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
        max_tokens=4096,
    )

    from app.config import DOCUMENT_TYPES
    if result.get("document_type") not in DOCUMENT_TYPES:
        result["document_type"] = "OTHER"

    logger.info(f"[{name_part}] Classification: {result.get('document_type')} "
                f"(confidence: {result.get('confidence', 0):.0%})")
    return result
