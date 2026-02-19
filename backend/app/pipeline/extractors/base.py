"""Base extractor interface and TextPrimaryExtractor wrapper.

Architecture (post qwen3-vl removal):
  - Sarvam AI provides high-quality OCR text (Tamil-optimised).
  - gpt-oss:20b does ALL reasoning/extraction from that text.
  - No vision model fallback — the pipeline is text-only.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.pipeline.llm_client import LLMProgressCallback

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base for document-type-specific extractors."""

    @abstractmethod
    async def extract(self, extracted_text: dict, on_progress: LLMProgressCallback | None = None, filename: str = "", file_path: Path | None = None) -> dict:
        """Extract structured data from document text.

        Args:
            extracted_text: Output from ingestion.extract_text_from_pdf()
            on_progress: Async callback for LLM progress updates
            filename: Original filename for labeling
            file_path: Path to original PDF (kept for API compat)

        Returns:
            Structured data dict specific to document type
        """
        pass


class TextPrimaryExtractor(BaseExtractor):
    """Thin wrapper that delegates to a text extractor with confidence metadata.

    With qwen3-vl removed, this simply runs the text extractor (GPT-OSS)
    and attaches confidence scores — no vision fallback, no merging.
    """

    def __init__(
        self,
        text_extractor: BaseExtractor,
        schema: dict,
        # Legacy kwargs accepted but ignored (for call-site compat)
        vision_extractor: BaseExtractor | None = None,
        confidence_threshold: float | None = None,
    ):
        self.text = text_extractor
        self.schema = schema

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
    ) -> dict:
        name = filename or "document"

        # ── Text extraction (GPT-OSS) ──────────────────────────────
        logger.info(f"[{name}] Text extraction starting (GPT-OSS)")
        result = await self.text.extract(
            extracted_text, on_progress=on_progress,
            filename=filename, file_path=file_path,
        )

        # ── Confidence assessment ───────────────────────────────────
        from app.pipeline.confidence import assess_extraction_confidence

        extraction_quality = extracted_text.get("extraction_quality", "HIGH")
        conf = assess_extraction_confidence(
            result, self.schema,
            extraction_quality=extraction_quality,
        )

        # Attach confidence metadata
        if isinstance(result, dict):
            result["_confidence"] = conf.score
            result["_field_confidences"] = conf.field_confidences
            result["_extraction_method"] = "text"

        logger.info(f"[{name}] Extraction complete — confidence {conf.score:.2f}")
        return result

