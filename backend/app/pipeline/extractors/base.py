"""Base extractor interface and TextPrimaryExtractor wrapper."""

from __future__ import annotations

import asyncio
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
            file_path: Path to original PDF (needed for vision-based extractors)
        
        Returns:
            Structured data dict specific to document type
        """
        pass


# Module-level vision semaphore — initialised lazily so it picks up the
# running event-loop.  Import VISION_MAX_CONCURRENT here to avoid circular
# imports at module load time.
_vision_semaphore: asyncio.Semaphore | None = None


def _get_vision_semaphore() -> asyncio.Semaphore:
    """Return (and lazily create) the module-level vision semaphore."""
    global _vision_semaphore
    if _vision_semaphore is None:
        from app.config import VISION_MAX_CONCURRENT
        _vision_semaphore = asyncio.Semaphore(VISION_MAX_CONCURRENT)
    return _vision_semaphore


class TextPrimaryExtractor(BaseExtractor):
    """Wraps a text extractor + vision extractor with confidence-gated fallback.

    Flow
    ────
    1.  Run ``text_extractor.extract()`` (GPT-OSS, structured outputs, CoT).
    2.  Assess confidence via ``confidence.assess_extraction_confidence()``.
    3.  If confidence is low **and** a file_path is available **and** vision
        fallback is enabled → call ``vision_extractor.extract()`` with a
        focus-fields hint, then merge the two results.
    4.  Return the (possibly merged) result with ``_extraction_method`` metadata.
    """

    def __init__(
        self,
        text_extractor: BaseExtractor,
        vision_extractor: BaseExtractor,
        schema: dict,
        confidence_threshold: float | None = None,
    ):
        self.text = text_extractor
        self.vision = vision_extractor
        self.schema = schema
        # Defer config import so module can be imported without side effects
        if confidence_threshold is not None:
            self.threshold = confidence_threshold
        else:
            from app.config import TEXT_EXTRACTION_CONFIDENCE_THRESHOLD
            self.threshold = TEXT_EXTRACTION_CONFIDENCE_THRESHOLD

    async def _extract_impl(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
    ) -> dict:
        name = filename or "document"

        # ── Step 1: Text extraction (GPT-OSS) ──────────────────────
        logger.info(f"[{name}] Text-primary extraction starting (GPT-OSS)")
        result = await self.text.extract(
            extracted_text, on_progress=on_progress,
            filename=filename, file_path=file_path,
        )

        # ── Step 2: Confidence assessment ───────────────────────────
        from app.config import VISION_FALLBACK_ENABLED, VISION_DOC_TYPES
        from app.pipeline.confidence import assess_extraction_confidence

        extraction_quality = extracted_text.get("extraction_quality", "HIGH")
        conf = assess_extraction_confidence(
            result, self.schema,
            extraction_quality=extraction_quality,
            threshold=self.threshold,
        )

        # Attach confidence metadata
        if isinstance(result, dict):
            result["_confidence"] = conf.score
            result["_field_confidences"] = conf.field_confidences
            result["_extraction_method"] = "text"

        if not conf.needs_vision:
            logger.info(
                f"[{name}] Confidence {conf.score:.2f} ≥ {self.threshold} — "
                f"skipping vision fallback"
            )
            return result

        # ── Step 3: Vision fallback (Qwen3-VL) ─────────────────────
        if not VISION_FALLBACK_ENABLED:
            logger.info(f"[{name}] Vision fallback disabled by config")
            return result

        # Determine document type for VISION_DOC_TYPES eligibility
        doc_type = extracted_text.get("_doc_type", "OTHER")
        if doc_type not in VISION_DOC_TYPES:
            logger.info(f"[{name}] Doc type '{doc_type}' not eligible for vision fallback")
            return result

        if not file_path or not Path(file_path).exists():
            logger.warning(f"[{name}] No file_path available for vision fallback")
            return result

        logger.info(
            f"[{name}] Confidence {conf.score:.2f} < {self.threshold} — "
            f"triggering vision fallback for {len(conf.weak_fields)} weak field(s): "
            f"{conf.weak_fields}"
        )

        try:
            sem = _get_vision_semaphore()
            async with sem:
                # Pass focus_fields hint if the vision extractor supports it
                vision_kwargs: dict = dict(
                    extracted_text=extracted_text,
                    on_progress=on_progress,
                    filename=filename,
                    file_path=file_path,
                )
                # Vision extractors accept optional focus_fields kwarg
                if hasattr(self.vision.extract, "__code__"):
                    import inspect
                    sig = inspect.signature(self.vision.extract)
                    if "focus_fields" in sig.parameters:
                        vision_kwargs["focus_fields"] = conf.weak_fields

                vision_result = await self.vision.extract(**vision_kwargs)

            # ── Step 4: Merge results ───────────────────────────────
            merged = self._merge_results(result, vision_result, conf.weak_fields)
            merged["_extraction_method"] = "text+vision"
            merged["_confidence"] = conf.score
            merged["_vision_reason"] = "; ".join(conf.reasons[:5])
            logger.info(f"[{name}] Merged text + vision results")
            return merged

        except Exception as e:
            logger.warning(f"[{name}] Vision fallback failed ({e}), keeping text result")
            result["_vision_fallback_error"] = str(e)
            return result

        finally:
            # ── Step 5: Targeted Tamil name vision re-check (Tier 2) ────
            # This runs AFTER the confidence-gated vision fallback (or when
            # it was skipped).  It only fires when Tamil names are detected
            # and a full vision fallback did NOT already run.
            # GPU cost: ONE batched call per document, max 3 pages.
            pass  # actual call is in the wrapper below

    async def extract(
        self,
        extracted_text: dict,
        on_progress: LLMProgressCallback | None = None,
        filename: str = "",
        file_path: Path | None = None,
    ) -> dict:
        result = await self._extract_impl(
            extracted_text, on_progress=on_progress,
            filename=filename, file_path=file_path,
        )

        # ── Tier 2: Targeted Tamil name vision re-check ────────────────
        # Only triggers when Tamil names exist AND full vision didn't run.
        doc_type = extracted_text.get("_doc_type", "OTHER")
        if isinstance(result, dict) and result.get("_extraction_method") != "text+vision":
            result = await vision_recheck_tamil_names(
                result, file_path, doc_type, on_progress
            )
        return result

    # ── Merge helpers ───────────────────────────────────────────────

    @staticmethod
    def _merge_results(
        text_result: dict,
        vision_result: dict,
        weak_fields: list[str],
    ) -> dict:
        """Merge text and vision extraction results.

        Strategy:
        - For weak fields: prefer the vision value if it is non-empty.
        - For all other fields: keep the text value (GPT-OSS with CoT is
          generally more reliable for well-OCR'd text).
        - Private metadata fields (starting with _) are never overwritten.
        """
        if not isinstance(vision_result, dict):
            return text_result

        merged = dict(text_result)  # shallow copy

        weak_set = set(weak_fields)

        for field_name in weak_set:
            # Handle dotted paths like "financials.consideration_amount"
            parts = field_name.split(".")
            if len(parts) == 2:
                parent, child = parts
                v_parent = vision_result.get(parent)
                if isinstance(v_parent, dict):
                    v_val = v_parent.get(child)
                    if v_val is not None and not _is_trivial(v_val):
                        if not isinstance(merged.get(parent), dict):
                            merged[parent] = {}
                        merged[parent][child] = v_val
            else:
                v_val = vision_result.get(field_name)
                if v_val is not None and not _is_trivial(v_val):
                    merged[field_name] = v_val

        return merged


def _is_trivial(val) -> bool:
    """True for empty / placeholder values that shouldn't override text results."""
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip() or val.strip().lower() in (
            "n/a", "unknown", "not available", "not found", "nil", "none",
        )
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        return len(val) == 0
    return False


# ═══════════════════════════════════════════════════
# SHARED VISION NAME RE-CHECK
# ═══════════════════════════════════════════════════
# Tiered name matching strategy (GPU-optimized):
#   Tier 1 (FREE):  _fix_orphan_vowel_signs + space-collapsed matching in utils.py
#   Tier 2 (RARE):  Targeted Qwen vision re-read — only when string matching
#                   STILL fails after Tier 1 fixes.  One batched call per doc.
#   Tier 3 (AUTO):  Full vision fallback via confidence scoring when garbled
#                   penalty exceeds threshold.
#
# This module provides the Tier 2 shared utility.  EC uses its own variant
# because EC has 65+ pages and needs row-based page estimation.

# Name field paths per document type — used by vision name re-check
_NAME_FIELD_PATHS: dict[str, list[tuple[str, str]]] = {
    # (array_field, name_subfield) tuples
    "PATTA":     [("owner_names", "name"), ("owner_names", "father_name")],
    "CHITTA":    [("owner_names", "name"), ("owner_names", "father_name")],
    "SALE_DEED": [("seller", "name"), ("seller", "father_name"),
                  ("buyer", "name"), ("buyer", "father_name")],
}


def _collect_tamil_names(result: dict, doc_type: str) -> list[tuple[str, str, int, str]]:
    """Collect name field values that contain Tamil text.

    Returns list of (value, array_field, index, sub_field) tuples.
    Only returns names that actually contain Tamil characters.
    """
    from app.pipeline.utils import has_tamil
    field_paths = _NAME_FIELD_PATHS.get(doc_type, [])
    tamil_names: list[tuple[str, str, int, str]] = []
    for array_field, sub_field in field_paths:
        items = result.get(array_field, [])
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                val = item.get(sub_field, "")
            elif isinstance(item, str) and sub_field == "name":
                val = item
            else:
                continue
            if val and isinstance(val, str) and len(val) >= 3 and has_tamil(val):
                tamil_names.append((val, array_field, idx, sub_field))
    return tamil_names


async def vision_recheck_tamil_names(
    result: dict,
    file_path: Path | None,
    doc_type: str,
    on_progress: LLMProgressCallback | None = None,
) -> dict:
    """Targeted Qwen vision re-read for Tamil party names.

    This is Tier 2 of the tiered name matching strategy.  It is called
    ONLY when:
      1. The document has Tamil names (detected by ``has_tamil()``).
      2. The full vision fallback was NOT already triggered (avoids
         double-processing).

    For GPU efficiency:
      - Sends only 1–3 pages (short docs like Patta / Sale Deed).
      - Batches ALL names into a SINGLE ``call_vision_llm()`` call.
      - Only triggers when Tamil text is present (non-Tamil docs skip).

    If vision fails or Qwen can’t read the name, keeps the text
    extraction and sets ``_name_confidence = "low"`` as a flag for
    downstream matchers to skip false-positive warnings.
    """
    from app.config import VISION_FALLBACK_ENABLED
    if not VISION_FALLBACK_ENABLED:
        return result
    if not file_path or not Path(file_path).exists():
        return result
    if doc_type not in _NAME_FIELD_PATHS:
        return result
    # Skip if full vision fallback already ran
    if isinstance(result, dict) and result.get("_extraction_method") == "text+vision":
        return result

    tamil_names = _collect_tamil_names(result, doc_type)
    if not tamil_names:
        return result

    logger.info(
        f"[{doc_type}] Found {len(tamil_names)} Tamil name(s) — "
        f"triggering targeted Qwen vision name re-check"
    )

    try:
        from app.pipeline.ingestion import render_pages_as_images
        from app.pipeline.llm_client import call_vision_llm

        all_images = render_pages_as_images(Path(file_path))
        if not all_images:
            logger.warning(f"[{doc_type}] No images rendered, skipping vision name re-check")
            return result

        # Short docs (Patta, Sale Deed): send first 2–3 pages
        # Names are typically on page 1 for these doc types.
        max_pages = min(3, len(all_images))
        images = all_images[:max_pages]

        # Build focused prompt with current extracted names
        name_list = "\n".join(
            f"  - {field}[{idx}].{sub}: \"{val}\""
            for val, field, idx, sub in tamil_names
        )
        doc_label = {
            "PATTA": "Patta (land ownership certificate)",
            "CHITTA": "Chitta (land record)",
            "SALE_DEED": "Sale Deed (property transfer document)",
        }.get(doc_type, doc_type)

        prompt = (
            f"This is a {doc_label} from Tamil Nadu. "
            "Some party names were extracted from OCR text and may be incorrect. "
            "Please read the document image carefully and provide the CORRECT "
            "Tamil party names.\n\n"
            f"Current extracted names:\n{name_list}\n\n"
            "Return a JSON object with this format:\n"
            '{"corrections": [{"field": "owner_names", "index": 0, '
            '"sub_field": "name", "corrected_value": "correct name"}, ...]}\n'
            "Only include entries where you can read the name more accurately "
            "than what was extracted. If a name looks correct, omit it."
        )

        sem = _get_vision_semaphore()
        async with sem:
            vision_result = await call_vision_llm(
                prompt=prompt,
                images=images,
                system_prompt="You are a Tamil document reader. Extract party names accurately.",
                expect_json=True,
                task_label=f"{doc_type} name re-check ({len(images)} pages, {len(tamil_names)} names)",
                on_progress=on_progress,
            )

        # Apply corrections
        corrections = (
            vision_result.get("corrections", [])
            if isinstance(vision_result, dict) else []
        )
        applied = 0
        for corr in corrections:
            field = corr.get("field")
            idx = corr.get("index")
            sub = corr.get("sub_field", "name")
            new_val = corr.get("corrected_value", "")
            if not field or idx is None or not new_val or len(new_val) < 2:
                continue
            items = result.get(field, [])
            if not isinstance(items, list) or idx >= len(items):
                continue
            item = items[idx]
            if isinstance(item, dict):
                old_val = item.get(sub, "")
                item[sub] = new_val
                logger.info(
                    f"[{doc_type}] Vision corrected {field}[{idx}].{sub}: "
                    f"'{old_val}' → '{new_val}'"
                )
                applied += 1
            elif isinstance(item, str) and sub == "name":
                old_val = item
                items[idx] = new_val
                logger.info(
                    f"[{doc_type}] Vision corrected {field}[{idx}]: "
                    f"'{old_val}' → '{new_val}'"
                )
                applied += 1

        if applied:
            notes = result.get("extraction_notes", "") or result.get("remarks", "")
            vision_note = (
                f"Vision re-checked {len(tamil_names)} Tamil name(s), "
                f"applied {applied} correction(s)"
            )
            if "extraction_notes" in result:
                result["extraction_notes"] = (
                    f"{notes}; {vision_note}" if notes else vision_note
                )
            else:
                result["remarks"] = (
                    f"{notes}; {vision_note}" if notes else vision_note
                )
            result["_extraction_method"] = "text+vision"
            logger.info(f"[{doc_type}] {vision_note}")
        else:
            # Vision didn't improve any names — flag as low-confidence
            for val, field, idx, sub in tamil_names:
                items = result.get(field, [])
                if isinstance(items, list) and idx < len(items):
                    item = items[idx]
                    if isinstance(item, dict):
                        item["_name_confidence"] = "low"
            logger.info(
                f"[{doc_type}] Vision name re-check returned no corrections — "
                f"flagging {len(tamil_names)} name(s) as low-confidence"
            )

    except Exception as e:
        logger.warning(
            f"[{doc_type}] Vision name re-check failed ({e}), "
            f"keeping text extraction"
        )
        # Flag names as low-confidence since vision also failed
        for val, field, idx, sub in tamil_names:
            items = result.get(field, [])
            if isinstance(items, list) and idx < len(items):
                item = items[idx]
                if isinstance(item, dict):
                    item["_name_confidence"] = "low"

    return result

