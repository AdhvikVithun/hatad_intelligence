"""Document classifier - identifies document type using LLM.

Vision-primary: always tries the vision model first (qwen3-vl reads
stamps, seals, Tamil headers directly from the page image). Falls back
to text-based classification only when no file_path is available.
"""

import logging
import re
from pathlib import Path
from app.config import PROMPTS_DIR, VISION_MODEL
from app.pipeline.llm_client import call_llm, call_vision_llm, LLMProgressCallback
from app.pipeline.schemas import CLASSIFY_SCHEMA

logger = logging.getLogger(__name__)


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


async def classify_document(
    extracted_text: dict,
    filename: str = "",
    on_progress: LLMProgressCallback | None = None,
    file_path: Path | None = None,
) -> dict:
    """Classify a document — vision-primary, text-fallback.
    
    Uses the vision model (qwen3-vl) as the primary classifier since it
    can read stamps, seals, headers, and Tamil text directly from images.
    Falls back to text-based classification only when no file_path is
    available for rendering.
    
    Args:
        extracted_text: Output from ingestion.extract_text_from_pdf()
        filename: Original filename for labeling
        on_progress: Async callback for LLM progress updates
        file_path: Path to original PDF (needed for vision)
    
    Returns:
        {"document_type": "EC", "confidence": 0.95, ...}
    """
    name_part = filename or "document"
    total_pages = extracted_text.get("total_pages", "?")

    # ── Vision-primary classification ──
    if file_path and Path(file_path).exists():
        return await _classify_with_vision(file_path, name_part, total_pages, on_progress)

    # ── Fallback: text-based classification (no file available) ──
    pages = extracted_text["pages"][:2]
    sample_text = "\n\n".join(p["text"] for p in pages if p["text"])

    if not sample_text.strip():
        return {
            "document_type": "OTHER",
            "confidence": 0.0,
            "language": "unknown",
            "key_identifiers": [],
            "reasoning": "No text could be extracted and no file available for vision.",
        }

    system_prompt = _load_prompt("classify")
    result = await call_llm(
        prompt=f"Classify this document:\n\n{sample_text}",
        system_prompt=system_prompt,
        expect_json=CLASSIFY_SCHEMA,
        task_label=f"Classify {name_part} ({total_pages}p, {len(sample_text):,} chars)",
        on_progress=on_progress,
        think=True,
        max_tokens=4096,  # Classification output is small — cap to save time
    )

    from app.config import DOCUMENT_TYPES
    if result.get("document_type") not in DOCUMENT_TYPES:
        result["document_type"] = "OTHER"

    return result


async def _classify_with_vision(
    file_path: Path,
    name_part: str,
    total_pages,
    on_progress: LLMProgressCallback | None,
) -> dict:
    """Classify a document by sending the first 2 page images to the vision model."""
    try:
        from app.pipeline.llm_client import check_ollama_status
        status = await check_ollama_status()
        if not any(VISION_MODEL in m for m in status.get("models", [])):
            logger.warning(f"[{name_part}] Vision model not available for classification")
            return {
                "document_type": "OTHER",
                "confidence": 0.0,
                "language": "unknown",
                "key_identifiers": [],
                "reasoning": "Vision model not available for classification.",
            }

        # Use shared render function (cached, 200 DPI)
        from app.pipeline.ingestion import render_pages_as_images
        all_images = render_pages_as_images(Path(file_path))
        # Only first 2 pages for classification
        images = all_images[:2]

        if not images:
            return {
                "document_type": "OTHER",
                "confidence": 0.0,
                "language": "unknown",
                "key_identifiers": [],
                "reasoning": "Could not render page images for classification.",
            }

        system_prompt = _load_prompt("classify_vision")
        result = await call_vision_llm(
            prompt=f"Classify this document ({len(images)} page(s) shown). What type of Tamil Nadu land document is this?",
            images=images,
            system_prompt=system_prompt,
            expect_json=CLASSIFY_SCHEMA,
            task_label=f"Vision classify {name_part} ({total_pages}p, {len(images)} images)",
            on_progress=on_progress,
        )

        from app.config import DOCUMENT_TYPES
        if result.get("document_type") not in DOCUMENT_TYPES:
            result["document_type"] = "OTHER"

        logger.info(f"[{name_part}] Vision classification: {result.get('document_type')} "
                     f"(confidence: {result.get('confidence', 0):.0%})")
        return result

    except Exception as e:
        logger.warning(f"[{name_part}] Vision classification failed: {e}")
        return {
            "document_type": "OTHER",
            "confidence": 0.0,
            "language": "unknown",
            "key_identifiers": [],
            "reasoning": f"Classification failed: {e}",
        }
