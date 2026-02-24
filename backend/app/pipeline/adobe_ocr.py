"""Adobe PDF Services integration for structured text extraction.

Adobe PDF Extract API returns structured JSON with per-element text, bounding
boxes, tables, and style information.  This serves as a **reliable cloud OCR
fallback** when Sarvam AI is unavailable or its queue is stalled.

Flow:
  1. Validate PDF (size, magic bytes).
  2. Authenticate with Adobe using Service Principal credentials.
  3. Upload PDF → submit Extract job.
  4. Poll until complete (SDK handles this internally).
  5. Download result ZIP → parse ``structuredData.json``.
  6. Build per-page text from structured elements.

Resilience features:
  - Circuit breaker: disables Adobe after N consecutive failures.
  - Timeout guard: aborts if job exceeds ADOBE_PDF_TIMEOUT seconds.
  - File-size guard: skips files > ADOBE_PDF_MAX_FILE_MB.
  - Every failure path returns None — never raises.

Feature-gated: when ``ADOBE_PDF_CLIENT_ID`` is empty the functions are no-ops.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.config import (
    ADOBE_PDF_CLIENT_ID,
    ADOBE_PDF_CLIENT_SECRET,
    ADOBE_PDF_ENABLED,
    ADOBE_PDF_TIMEOUT,
    ADOBE_PDF_MAX_FILE_MB,
)

logger = logging.getLogger(__name__)

# ── SDK availability check ─────────────────────────────────────────
_adobe_available: bool | None = None


def _check_adobe_sdk() -> bool:
    """Return True if the Adobe PDF Services SDK is importable."""
    global _adobe_available
    if _adobe_available is not None:
        return _adobe_available
    try:
        from adobe.pdfservices.operation.pdf_services import PDFServices  # noqa: F401
        _adobe_available = True
    except ImportError:
        logger.warning("pdfservices-sdk not installed — Adobe PDF OCR disabled")
        _adobe_available = False
    return _adobe_available


# ── Circuit breaker ────────────────────────────────────────────────
_CB_THRESHOLD = 3
_CB_COOLDOWN_SECS = 300  # 5 minutes
_cb_lock = threading.Lock()
_cb_consecutive_failures = 0
_cb_disabled_until: float = 0.0


def _cb_record_success() -> None:
    global _cb_consecutive_failures
    with _cb_lock:
        _cb_consecutive_failures = 0


def _cb_record_failure() -> None:
    global _cb_consecutive_failures, _cb_disabled_until
    with _cb_lock:
        _cb_consecutive_failures += 1
        if _cb_consecutive_failures >= _CB_THRESHOLD:
            _cb_disabled_until = time.monotonic() + _CB_COOLDOWN_SECS
            logger.warning(
                f"Adobe PDF circuit breaker OPEN: {_cb_consecutive_failures} consecutive "
                f"failures — disabled for {_CB_COOLDOWN_SECS}s"
            )


def _cb_is_open() -> bool:
    with _cb_lock:
        if _cb_consecutive_failures < _CB_THRESHOLD:
            return False
        if time.monotonic() >= _cb_disabled_until:
            return False
        return True


def reset_circuit_breaker() -> None:
    """Reset the circuit breaker (for testing or manual recovery)."""
    global _cb_consecutive_failures, _cb_disabled_until
    with _cb_lock:
        _cb_consecutive_failures = 0
        _cb_disabled_until = 0.0


# ── File validation ────────────────────────────────────────────────
_PDF_MAGIC = b"%PDF"
_MAX_FILE_BYTES = ADOBE_PDF_MAX_FILE_MB * 1024 * 1024


def _validate_pdf(file_path: Path) -> str | None:
    """Return error message if invalid for Adobe, else None."""
    if not file_path.exists():
        return f"file does not exist: {file_path}"
    try:
        size = file_path.stat().st_size
    except OSError as e:
        return f"cannot stat file: {e}"
    if size == 0:
        return "file is empty (0 bytes)"
    if size > _MAX_FILE_BYTES:
        return f"file too large ({size / 1024 / 1024:.1f} MB > {ADOBE_PDF_MAX_FILE_MB} MB limit)"
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        if not header.startswith(_PDF_MAGIC):
            return f"not a valid PDF (magic bytes: {header[:4]!r})"
    except OSError as e:
        return f"cannot read file: {e}"
    return None


# ── Sync extraction (runs in thread pool from async context) ──────
def _adobe_extract_sync(file_path: Path) -> dict | None:
    """Upload PDF to Adobe, extract text+tables, return result dict.

    This is a blocking call — intended to be run via ``asyncio.to_thread()``
    or a ``ThreadPoolExecutor``.

    Returns:
        dict matching ``extract_text_from_pdf()`` format, or None on failure.
    """
    from adobe.pdfservices.operation.auth.service_principal_credentials import (
        ServicePrincipalCredentials,
    )
    from adobe.pdfservices.operation.exception.exceptions import (
        ServiceApiException,
        ServiceUsageException,
        SdkException,
    )
    from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
    from adobe.pdfservices.operation.io.stream_asset import StreamAsset
    from adobe.pdfservices.operation.pdf_services import PDFServices
    from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
    from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import (
        ExtractElementType,
    )
    from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import (
        ExtractPDFParams,
    )
    from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import (
        ExtractPDFResult,
    )

    t0 = time.time()

    try:
        # Read input
        with open(file_path, "rb") as f:
            input_stream = f.read()

        # Authenticate
        credentials = ServicePrincipalCredentials(
            client_id=ADOBE_PDF_CLIENT_ID,
            client_secret=ADOBE_PDF_CLIENT_SECRET,
        )
        pdf_services = PDFServices(credentials=credentials)
        logger.info(f"Adobe PDF: authenticated ({time.time()-t0:.1f}s)")

        # Upload
        input_asset = pdf_services.upload(
            input_stream=input_stream, mime_type=PDFServicesMediaType.PDF
        )
        logger.info(f"Adobe PDF: uploaded {file_path.name} ({time.time()-t0:.1f}s)")

        # Configure extraction — TEXT + TABLES
        extract_params = ExtractPDFParams(
            elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES],
        )

        # Submit job
        extract_job = ExtractPDFJob(
            input_asset=input_asset, extract_pdf_params=extract_params
        )
        location = pdf_services.submit(extract_job)
        logger.info(f"Adobe PDF: job submitted for {file_path.name} ({time.time()-t0:.1f}s)")

        # Wait for result (SDK polls internally)
        response = pdf_services.get_job_result(location, ExtractPDFResult)
        logger.info(f"Adobe PDF: job completed for {file_path.name} ({time.time()-t0:.1f}s)")

        # Download result ZIP
        result_asset: CloudAsset = response.get_result().get_resource()
        stream_asset: StreamAsset = pdf_services.get_content(result_asset)
        zip_bytes = stream_asset.get_input_stream()

        if not zip_bytes:
            logger.error(f"Adobe PDF: empty result for {file_path.name}")
            return None

        # Parse structured data from ZIP
        result = _parse_adobe_zip(file_path, zip_bytes)
        elapsed = time.time() - t0
        if result:
            logger.info(
                f"Adobe PDF: extracted {result['total_pages']} pages, "
                f"{len(result['full_text']):,} chars from {file_path.name} in {elapsed:.1f}s"
            )
        return result

    except (ServiceApiException, ServiceUsageException, SdkException) as e:
        logger.error(f"Adobe PDF: API error for {file_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Adobe PDF: unexpected error for {file_path.name}: {e}", exc_info=True)
        return None


def _parse_adobe_zip(file_path: Path, zip_bytes: bytes) -> dict | None:
    """Parse Adobe Extract API ZIP output into our standard format.

    The ZIP contains ``structuredData.json`` with elements that have:
      - ``Path``: e.g. ``//Document/P``, ``//Document/Table``, ``//Document/H1``
      - ``Text``: the text content
      - ``Page``: 0-based page index
      - ``Bounds``: [x0, y0, x1, y1] coordinates

    We group elements by page and build per-page text.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            if "structuredData.json" not in zf.namelist():
                logger.error(f"Adobe PDF: no structuredData.json in output for {file_path.name}")
                return None

            data = json.loads(zf.read("structuredData.json"))
            elements = data.get("elements", [])

            if not elements:
                logger.warning(f"Adobe PDF: 0 elements extracted from {file_path.name}")
                return None

            # Group elements by page
            pages_dict: dict[int, list[str]] = {}
            tables_dict: dict[int, list[list[list[str]]]] = {}

            for elem in elements:
                text = elem.get("Text", "")
                page_idx = elem.get("Page", 0)
                path = elem.get("Path", "")

                if text and text.strip():
                    pages_dict.setdefault(page_idx, []).append(text.strip())

                # Extract table data if present
                if "Table" in path and elem.get("filePaths"):
                    for table_file in elem.get("filePaths", []):
                        if table_file in zf.namelist():
                            try:
                                import csv
                                table_content = zf.read(table_file).decode("utf-8", errors="replace")
                                reader = csv.reader(io.StringIO(table_content))
                                table_data = [row for row in reader]
                                if table_data:
                                    tables_dict.setdefault(page_idx, []).append(table_data)
                            except Exception:
                                pass

            if not pages_dict:
                logger.warning(f"Adobe PDF: no text elements for {file_path.name}")
                return None

            # Build result
            max_page = max(pages_dict.keys()) + 1
            pages = []
            full_text_parts = []
            qualities = []

            for i in range(max_page):
                texts = pages_dict.get(i, [])
                page_text = "\n".join(texts)
                tables = tables_dict.get(i, [])

                quality = _assess_page_quality(page_text)
                pages.append({
                    "page_number": i + 1,
                    "text": page_text,
                    "tables": tables,
                    "extraction_method": "adobe",
                    "quality": quality,
                })
                qualities.append(quality["quality"])
                full_text_parts.append(f"--- PAGE {i + 1} ---\n{page_text}")

            # Overall quality
            if not qualities or all(q == "LOW" for q in qualities):
                overall = "LOW" if qualities else "EMPTY"
            elif all(q == "HIGH" for q in qualities):
                overall = "HIGH"
            else:
                overall = "MIXED"

            try:
                file_size = file_path.stat().st_size
            except OSError:
                file_size = 0

            return {
                "total_pages": len(pages),
                "pages": pages,
                "full_text": "\n\n".join(full_text_parts),
                "extraction_quality": overall,
                "ocr_pages": 0,
                "adobe_pages": len(pages),
                "sarvam_pages": 0,
                "metadata": {
                    "filename": file_path.name,
                    "file_size": file_size,
                    "extraction_source": "adobe_pdf_services",
                    "element_count": len(elements),
                },
            }

    except zipfile.BadZipFile as e:
        logger.error(f"Adobe PDF: corrupt output ZIP for {file_path.name}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Adobe PDF: invalid JSON in output for {file_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Adobe PDF: failed to parse output for {file_path.name}: {e}")
        return None


def _assess_page_quality(text: str) -> dict:
    """Minimal quality assessment matching ingestion.py format."""
    char_count = len(text.strip())
    word_count = len(text.split())
    quality = "HIGH" if char_count >= 50 and word_count >= 10 else "LOW"
    return {
        "char_count": char_count,
        "word_count": word_count,
        "cid_count": 0,
        "cid_ratio": 0.0,
        "quality": quality,
        "reason": None if quality == "HIGH" else f"too few chars ({char_count})",
    }


# ── Thread pool for running sync Adobe SDK from async context ──────
_adobe_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _adobe_executor
    if _adobe_executor is None:
        _adobe_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="adobe-ocr")
    return _adobe_executor


# ── Public async API ───────────────────────────────────────────────
async def adobe_extract_text(file_path: str | Path, *, on_progress=None) -> dict | None:
    """Upload a PDF to Adobe PDF Services and return extracted text.

    Bulletproof guarantees:
      - Returns ``None`` on ANY failure (never raises).
      - Validates file before uploading.
      - Respects circuit breaker (skips if API is known-down).
      - Reports progress via on_progress callback.
      - Enforces timeout via ThreadPoolExecutor + wait.

    Args:
        file_path: Path to the PDF file.
        on_progress: Optional async callback ``(stage, message, detail)``.

    Returns:
        dict matching ``extract_text_from_pdf()`` format, or ``None``.
    """
    file_path = Path(file_path)
    if not ADOBE_PDF_ENABLED:
        return None
    if not _check_adobe_sdk():
        return None

    # Circuit breaker check
    if _cb_is_open():
        logger.info(f"Adobe PDF: circuit breaker OPEN — skipping {file_path.name}")
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"Adobe PDF OCR temporarily disabled (recent failures) — skipping {file_path.name}",
                    {"type": "adobe_circuit_breaker"},
                )
            except Exception:
                pass
        return None

    # Pre-flight validation
    validation_error = _validate_pdf(file_path)
    if validation_error:
        logger.warning(f"Adobe PDF: skipping {file_path.name} — {validation_error}")
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"Adobe PDF: skipping {file_path.name} ({validation_error})",
                    {"type": "adobe_validation_error", "error": validation_error},
                )
            except Exception:
                pass
        return None

    if on_progress:
        try:
            await on_progress(
                "extraction",
                f"Adobe PDF: extracting text from {file_path.name}...",
                {"type": "adobe_start"},
            )
        except Exception:
            pass

    try:
        loop = asyncio.get_event_loop()
        executor = _get_executor()

        # Run sync Adobe SDK in thread pool with timeout
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, _adobe_extract_sync, file_path),
            timeout=ADOBE_PDF_TIMEOUT,
        )

        if result is not None:
            _cb_record_success()
            if on_progress:
                try:
                    pages = result.get("total_pages", 0)
                    chars = len(result.get("full_text", ""))
                    await on_progress(
                        "extraction",
                        f"Adobe PDF: extracted {pages} pages ({chars:,} chars) from {file_path.name}",
                        {"type": "adobe_success", "pages": pages, "chars": chars},
                    )
                except Exception:
                    pass
        else:
            _cb_record_failure()

        return result

    except asyncio.TimeoutError:
        logger.error(f"Adobe PDF: timed out after {ADOBE_PDF_TIMEOUT}s for {file_path.name}")
        # Timeouts don't trip circuit breaker — just slow processing
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"Adobe PDF: timed out for {file_path.name} (>{ADOBE_PDF_TIMEOUT}s)",
                    {"type": "adobe_timeout"},
                )
            except Exception:
                pass
        return None

    except Exception as e:
        _cb_record_failure()
        logger.error(f"Adobe PDF: extraction failed for {file_path.name}: {e}", exc_info=True)
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"Adobe PDF: failed for {file_path.name}: {e}",
                    {"type": "adobe_error", "error": str(e)},
                )
            except Exception:
                pass
        return None


# ── Merge logic ────────────────────────────────────────────────────
def merge_adobe_with_existing(
    adobe_result: dict,
    existing_result: dict,
) -> dict:
    """Merge Adobe extraction with existing (pdfplumber/Sarvam) results.

    Decision logic per page:
      - If existing is HIGH quality → keep existing (preserves layout/tables).
      - If Adobe has significantly more text → Adobe wins.
      - If page has Tamil content and Adobe extracted it cleanly → Adobe wins.
      - Otherwise → keep existing.

    Returns:
        Merged result dict in standard format.
    """
    from app.pipeline.utils import has_tamil, detect_garbled_tamil

    existing_pages = existing_result.get("pages", [])
    adobe_pages = adobe_result.get("pages", [])

    merged_pages = []
    full_text_parts = []
    qualities = []
    adobe_used = 0

    max_pages = max(len(existing_pages), len(adobe_pages))

    for i in range(max_pages):
        existing_page = existing_pages[i] if i < len(existing_pages) else None
        adobe_page = adobe_pages[i] if i < len(adobe_pages) else None

        if existing_page and not adobe_page:
            merged_pages.append(existing_page)
            qualities.append(existing_page["quality"]["quality"])
            full_text_parts.append(f"--- PAGE {existing_page['page_number']} ---\n{existing_page['text']}")
            continue

        if adobe_page and not existing_page:
            merged_pages.append(adobe_page)
            qualities.append(adobe_page["quality"]["quality"])
            full_text_parts.append(f"--- PAGE {adobe_page['page_number']} ---\n{adobe_page['text']}")
            adobe_used += 1
            continue

        # Both have this page — decide
        existing_text = existing_page["text"]
        adobe_text = adobe_page["text"]
        existing_chars = existing_page["quality"]["char_count"]
        adobe_chars = adobe_page["quality"]["char_count"]
        existing_quality = existing_page["quality"]["quality"]
        adobe_quality = adobe_page["quality"]["quality"]

        use_adobe = False

        # Rule 1: Existing is LOW, Adobe is HIGH
        if adobe_quality == "HIGH" and existing_quality == "LOW":
            use_adobe = True

        # Rule 2: Adobe has significantly more text
        elif adobe_chars > existing_chars * 1.3 and adobe_chars > 50:
            use_adobe = True

        # Rule 3: Existing has garbled Tamil, Adobe doesn't
        elif has_tamil(existing_text):
            existing_garbled, _, _ = detect_garbled_tamil(existing_text)
            if existing_garbled and adobe_chars > 30:
                adobe_garbled = False
                if has_tamil(adobe_text):
                    adobe_garbled, _, _ = detect_garbled_tamil(adobe_text)
                if not adobe_garbled:
                    use_adobe = True

        if use_adobe:
            # Preserve existing tables if Adobe doesn't have them
            if not adobe_page.get("tables") and existing_page.get("tables"):
                adobe_page = {**adobe_page, "tables": existing_page["tables"]}
            merged_pages.append(adobe_page)
            qualities.append(adobe_quality)
            full_text_parts.append(f"--- PAGE {adobe_page['page_number']} ---\n{adobe_text}")
            adobe_used += 1
        else:
            merged_pages.append(existing_page)
            qualities.append(existing_quality)
            full_text_parts.append(f"--- PAGE {existing_page['page_number']} ---\n{existing_text}")

    # Overall quality
    if not qualities or all(q == "LOW" for q in qualities):
        overall = "LOW" if qualities else "EMPTY"
    elif all(q == "HIGH" for q in qualities):
        overall = "HIGH"
    else:
        overall = "MIXED"

    merged = {
        "total_pages": len(merged_pages),
        "pages": merged_pages,
        "full_text": "\n\n".join(full_text_parts),
        "extraction_quality": overall,
        "ocr_pages": existing_result.get("ocr_pages", 0),
        "sarvam_pages": existing_result.get("sarvam_pages", 0),
        "adobe_pages": adobe_used,
        "metadata": {
            **existing_result.get("metadata", {}),
            "adobe_pages_used": adobe_used,
        },
    }

    logger.info(
        f"Adobe merge: {adobe_used}/{len(merged_pages)} pages from Adobe, "
        f"rest from existing (overall quality: {overall})"
    )
    return merged
