"""Sarvam AI Document Intelligence integration for Tamil OCR.

Sarvam AI specialises in Indian-language document digitisation.  This module
wraps the ``sarvamai`` SDK to provide a drop-in OCR path that runs in
parallel with the existing pdfplumber + Tesseract pipeline.

Flow:
  1. Validate PDF (size, magic bytes) — skip if invalid.
  2. Upload PDF to Sarvam → async job starts server-side.
  3. Poll until complete (or timeout).
  4. Download HTML output → parse per-page text.
  5. Merge with pdfplumber result — best quality wins per page.

Resilience features:
  - Circuit breaker: disables Sarvam after N consecutive failures.
  - Exponential backoff between retries.
  - Per-step exception isolation (create_job / upload / start / poll / download).
  - File-size guard: skip files > SARVAM_MAX_FILE_MB.
  - ZIP integrity check before parsing.
  - Every failure path returns None — never raises.

Feature-gated: when ``SARVAM_API_KEY`` is empty the functions are no-ops.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import tempfile
import time
import threading
import zipfile
from pathlib import Path

from app.config import (
    SARVAM_API_KEY,
    SARVAM_ENABLED,
    SARVAM_LANGUAGE,
    SARVAM_MAX_RETRIES,
    SARVAM_POLL_INTERVAL,
    SARVAM_TIMEOUT,
    SARVAM_MAX_FILE_MB,
)

logger = logging.getLogger(__name__)

# ── Lazy SDK import ────────────────────────────────────────────────
_sarvam_available: bool | None = None


def _check_sarvam_sdk() -> bool:
    """Return True if the sarvamai package is importable."""
    global _sarvam_available
    if _sarvam_available is not None:
        return _sarvam_available
    try:
        import sarvamai  # noqa: F401
        _sarvam_available = True
    except ImportError:
        logger.warning("sarvamai package not installed — Sarvam OCR disabled")
        _sarvam_available = False
    return _sarvam_available


# ── Circuit breaker ────────────────────────────────────────────────
# After _CB_THRESHOLD consecutive failures, disable Sarvam for
# _CB_COOLDOWN_SECS seconds.  This avoids hammering a down API.
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
                f"Sarvam AI circuit breaker OPEN: {_cb_consecutive_failures} consecutive "
                f"failures — disabled for {_CB_COOLDOWN_SECS}s"
            )


def _cb_is_open() -> bool:
    with _cb_lock:
        if _cb_consecutive_failures < _CB_THRESHOLD:
            return False
        if time.monotonic() >= _cb_disabled_until:
            # Cooldown expired — allow a probe attempt
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
_MAX_FILE_BYTES = SARVAM_MAX_FILE_MB * 1024 * 1024


def _validate_pdf(file_path: Path) -> str | None:
    """Return an error message if the PDF is invalid for Sarvam, else None."""
    if not file_path.exists():
        return f"file does not exist: {file_path}"
    try:
        size = file_path.stat().st_size
    except OSError as e:
        return f"cannot stat file: {e}"
    if size == 0:
        return "file is empty (0 bytes)"
    if size > _MAX_FILE_BYTES:
        return f"file too large ({size / 1024 / 1024:.1f} MB > {SARVAM_MAX_FILE_MB} MB limit)"
    # Magic bytes check
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        if not header.startswith(_PDF_MAGIC):
            return f"not a valid PDF (magic bytes: {header[:4]!r})"
    except OSError as e:
        return f"cannot read file: {e}"
    return None


# ── HTML parsing ───────────────────────────────────────────────────
def _parse_sarvam_html(html_content: str) -> list[str]:
    """Extract per-page text from Sarvam HTML output.

    Sarvam's output format uses ``<div class="page-body-container">``
    wrappers — one per PDF page.  We try several strategies in order of
    specificity, falling back to treating the full HTML as a single page.

    Returns:
        list[str] — one text string per page.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Strategy 1: Sarvam page-body-container divs (current API format)
        # Each PDF page is wrapped in <div class="page-body-container">
        page_divs = soup.find_all("div", class_="page-body-container")
        if page_divs:
            logger.debug(f"Sarvam HTML: found {len(page_divs)} page-body-container divs")
            return [div.get_text(separator="\n", strip=True) for div in page_divs]

        # Strategy 2: explicit page containers via data attribute
        page_divs = soup.find_all("div", attrs={"data-page-number": True})
        if page_divs:
            return [div.get_text(separator="\n", strip=True) for div in page_divs]

        # Strategy 3: class="page" divs (older API / other providers)
        page_divs = soup.find_all("div", class_="page")
        if page_divs:
            return [div.get_text(separator="\n", strip=True) for div in page_divs]

        # Strategy 4: split on <hr> tags
        hr_parts = re.split(r'<hr\s*/?\s*>', html_content, flags=re.IGNORECASE)
        if len(hr_parts) > 1:
            pages = []
            for part in hr_parts:
                part_soup = BeautifulSoup(part, "html.parser")
                text = part_soup.get_text(separator="\n", strip=True)
                if text:
                    pages.append(text)
            if pages:
                return pages

        # Strategy 5: CSS page-break-before / page-break-after
        break_parts = re.split(
            r'<[^>]+style=["\'][^"\']*page-break-(?:before|after)\s*:\s*always[^"\']*["\'][^>]*>',
            html_content,
            flags=re.IGNORECASE,
        )
        if len(break_parts) > 1:
            pages = []
            for part in break_parts:
                part_soup = BeautifulSoup(part, "html.parser")
                text = part_soup.get_text(separator="\n", strip=True)
                if text:
                    pages.append(text)
            if pages:
                return pages

        # Fallback: treat entire HTML as one page
        full_text = soup.get_text(separator="\n", strip=True)
        return [full_text] if full_text else []

    except ImportError:
        logger.warning("beautifulsoup4 not installed — using regex HTML parsing")
        # Minimal regex fallback: strip tags
        text = re.sub(r'<[^>]+>', '\n', html_content)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        return [text] if text else []


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


# ── Core Sarvam extraction ────────────────────────────────────────
async def sarvam_extract_text(file_path: str | Path, *, on_progress=None) -> dict | None:
    """Upload a PDF to Sarvam AI and return extracted text.

    Runs the synchronous ``sarvamai`` SDK in a thread pool to avoid
    blocking the FastAPI event loop.

    Bulletproof guarantees:
      - Returns ``None`` on ANY failure (never raises).
      - Validates file before uploading.
      - Respects circuit breaker (skips if API is known-down).
      - Reports progress/errors via on_progress callback.

    Args:
        file_path: Path to the PDF file.
        on_progress: Optional async callback ``(stage, message, detail)``.

    Returns:
        dict matching ``extract_text_from_pdf()`` format, or ``None``
        if Sarvam is disabled / unavailable / fails.
    """
    file_path = Path(file_path)
    if not SARVAM_ENABLED:
        return None
    if not _check_sarvam_sdk():
        return None

    # Circuit breaker check
    if _cb_is_open():
        logger.info(f"Sarvam AI: circuit breaker OPEN — skipping {file_path.name}")
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"HATAD Vision temporarily disabled (recent failures) — skipping {file_path.name}",
                    {"type": "sarvam_circuit_breaker"},
                )
            except Exception:
                pass
        return None

    # Pre-flight file validation
    validation_error = _validate_pdf(file_path)
    if validation_error:
        logger.warning(f"Sarvam AI: skipping {file_path.name} — {validation_error}")
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"HATAD Vision: skipping {file_path.name} ({validation_error})",
                    {"type": "sarvam_validation_error", "error": validation_error},
                )
            except Exception:
                pass
        return None

    try:
        result = await asyncio.to_thread(
            _sarvam_extract_sync, file_path
        )
        if result is not None:
            _cb_record_success()
        else:
            _cb_record_failure()
        return result
    except Exception as e:
        _cb_record_failure()
        logger.error(f"HATAD Vision extraction failed for {file_path.name}: {e}", exc_info=True)
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"HATAD Vision failed for {file_path.name}: {e} (falling back to text extraction)",
                    {"type": "sarvam_error", "error": str(e)},
                )
            except Exception:
                pass
        return None


def _sarvam_extract_sync(file_path: str | Path) -> dict | None:
    """Synchronous Sarvam extraction — called via ``asyncio.to_thread``.

    Retries up to SARVAM_MAX_RETRIES times with exponential backoff.
    Each individual attempt is fully isolated — any exception in one
    attempt does not affect the next.
    """
    file_path = Path(file_path)
    max_attempts = SARVAM_MAX_RETRIES + 1
    last_error: str = ""

    for attempt in range(1, max_attempts + 1):
        try:
            result = _sarvam_extract_single_attempt(file_path, attempt)
            if result is not None:
                return result
            last_error = "extraction returned no data"
        except Exception as e:
            last_error = str(e)
            logger.warning(
                f"Sarvam AI: attempt {attempt}/{max_attempts} failed for "
                f"{file_path.name}: {e}"
            )

        # Exponential backoff before next retry (1s, 2s, 4s, …)
        if attempt < max_attempts:
            backoff = min(2 ** (attempt - 1), 30)  # cap at 30s
            logger.info(
                f"Sarvam AI: retrying {file_path.name} in {backoff}s "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(backoff)

    logger.error(
        f"Sarvam AI: all {max_attempts} attempts failed for {file_path.name}: {last_error}"
    )
    return None


def _sarvam_extract_single_attempt(file_path: str | Path, attempt: int = 1) -> dict | None:
    """Single attempt at Sarvam extraction with per-step error isolation."""
    try:
        from sarvamai import SarvamAI
    except ImportError:
        logger.error("sarvamai package not available")
        return None

    file_path = Path(file_path)
    logger.info(f"Sarvam AI: starting extraction for {file_path.name} (attempt {attempt})")

    # ── Step 1: Create client ──
    try:
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    except Exception as e:
        logger.error(f"Sarvam AI: client creation failed: {e}")
        return None

    # ── Step 2: Create job ──
    job = None
    try:
        job = client.document_intelligence.create_job(
            language=SARVAM_LANGUAGE,
            output_format="html",
        )
        logger.info(f"Sarvam AI: job created — {job.job_id}")
    except Exception as e:
        logger.error(f"Sarvam AI: create_job failed for {file_path.name}: {e}")
        return None

    # ── Step 3: Upload file ──
    try:
        job.upload_file(str(file_path))
        logger.info(f"Sarvam AI: uploaded {file_path.name}")
    except Exception as e:
        logger.error(f"Sarvam AI: upload_file failed for {file_path.name}: {e}")
        _try_cancel_job(job)
        return None

    # ── Step 4: Start processing ──
    try:
        job.start()
        logger.info(f"Sarvam AI: processing started for {file_path.name}")
    except Exception as e:
        logger.error(f"Sarvam AI: job.start() failed for {file_path.name}: {e}")
        _try_cancel_job(job)
        return None

    # ── Step 5: Wait for completion ──
    try:
        status = job.wait_until_complete(
            poll_interval=SARVAM_POLL_INTERVAL,
            timeout=SARVAM_TIMEOUT,
        )
    except TimeoutError:
        logger.error(
            f"Sarvam AI: job timed out after {SARVAM_TIMEOUT}s for {file_path.name}"
        )
        _try_cancel_job(job)
        return None
    except Exception as e:
        logger.error(f"Sarvam AI: wait_until_complete failed for {file_path.name}: {e}")
        _try_cancel_job(job)
        return None

    # ── Step 6: Evaluate job state ──
    state = (
        getattr(status, "job_state", None)
        or getattr(status, "state", None)
        or str(status)
    )
    state_lower = str(state).lower()
    logger.info(f"Sarvam AI: job {job.job_id} finished with state={state}")

    if state_lower == "failed":
        err = getattr(status, "error_message", None) or "unknown error"
        logger.error(f"Sarvam AI job failed for {file_path.name}: {err}")
        return None

    if state_lower not in ("completed", "partiallycompleted"):
        logger.error(
            f"Sarvam AI: unexpected state '{state}' after wait for {file_path.name}"
        )
        return None

    if state_lower == "partiallycompleted":
        try:
            metrics = job.get_page_metrics()
            logger.warning(
                f"Sarvam AI: partial completion for {file_path.name} — "
                f"{metrics.get('pages_succeeded', '?')}/{metrics.get('total_pages', '?')} pages OK, "
                f"{metrics.get('pages_failed', '?')} failed"
            )
        except Exception:
            logger.warning(f"Sarvam AI: partial completion for {file_path.name}")

    # ── Step 7: Download output ZIP ──
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.zip"
            try:
                job.download_output(str(output_path))
            except Exception as e:
                logger.error(
                    f"Sarvam AI: download_output failed for {file_path.name}: {e}"
                )
                return None

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error(
                    f"Sarvam AI: downloaded file is empty/missing for {file_path.name}"
                )
                return None

            logger.info(f"Sarvam AI: downloaded output for {file_path.name}")

            # ── Step 8: Extract HTML from ZIP ──
            html_content = _extract_html_from_zip(output_path)
            if not html_content:
                logger.warning(
                    f"Sarvam AI: no HTML content found in output for {file_path.name}"
                )
                return None

            # ── Step 8b: Also extract per-page metadata JSON (text fallback) ──
            metadata_texts = _extract_metadata_page_texts(output_path)
    except Exception as e:
        logger.error(f"Sarvam AI: output processing failed for {file_path.name}: {e}")
        return None

    # ── Step 9: Parse HTML → per-page text ──
    try:
        page_texts = _parse_sarvam_html(html_content)
    except Exception as e:
        logger.error(f"Sarvam AI: HTML parsing crashed for {file_path.name}: {e}")
        return None

    if not page_texts:
        logger.warning(
            f"Sarvam AI: HTML parsing produced no pages for {file_path.name}"
        )
        return None

    # If HTML produced only 1 "page" but metadata JSONs say there are more,
    # prefer the metadata-derived texts (they have correct page boundaries).
    if len(page_texts) == 1 and len(metadata_texts) > 1:
        logger.info(
            f"Sarvam AI: HTML fallback produced 1 page but metadata has "
            f"{len(metadata_texts)} — using metadata page texts for {file_path.name}"
        )
        page_texts = metadata_texts

    logger.info(f"Sarvam AI: extracted {len(page_texts)} pages from {file_path.name}")

    # ── Step 10: Build result dict ──
    return _build_result(file_path, page_texts)


def _try_cancel_job(job) -> None:
    """Best-effort job cancellation — swallows all errors."""
    try:
        if hasattr(job, "cancel"):
            job.cancel()
            logger.info(f"Sarvam AI: cancelled job {getattr(job, 'job_id', '?')}")
    except Exception:
        pass


def _build_result(file_path: Path, page_texts: list[str]) -> dict:
    """Build result dict matching extract_text_from_pdf() format."""
    pages = []
    full_text_parts = []
    qualities = []

    for i, text in enumerate(page_texts):
        quality = _assess_page_quality(text)
        pages.append({
            "page_number": i + 1,
            "text": text,
            "tables": [],
            "extraction_method": "sarvam",
            "quality": quality,
        })
        qualities.append(quality["quality"])
        full_text_parts.append(f"--- PAGE {i + 1} ---\n{text}")

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
        "sarvam_pages": len(pages),
        "metadata": {
            "filename": file_path.name,
            "file_size": file_size,
            "extraction_source": "sarvam_ai",
        },
    }


def _extract_html_from_zip(zip_path: Path) -> str | None:
    """Open a Sarvam output ZIP and return the first HTML file's content.

    Guards against corrupt ZIPs, empty archives, and encoding issues.
    """
    if not zip_path.exists():
        logger.error(f"Sarvam output ZIP missing: {zip_path}")
        return None

    try:
        file_size = zip_path.stat().st_size
        if file_size == 0:
            logger.error("Sarvam output ZIP is 0 bytes")
            return None
    except OSError as e:
        logger.error(f"Cannot stat Sarvam output ZIP: {e}")
        return None

    try:
        if not zipfile.is_zipfile(str(zip_path)):
            logger.error(f"Sarvam output is not a valid ZIP file ({file_size} bytes)")
            return None

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Check for ZIP corruption
            bad = zf.testzip()
            if bad is not None:
                logger.warning(f"Sarvam output ZIP has corrupt entry: {bad}")
                # Continue anyway — other entries may be fine

            html_files = [
                n for n in zf.namelist()
                if n.lower().endswith((".html", ".htm"))
            ]
            if not html_files:
                # Try any file with HTML-like content
                for name in zf.namelist():
                    try:
                        content = zf.read(name).decode("utf-8", errors="replace")
                        if "<html" in content.lower() or "<body" in content.lower():
                            return content
                    except Exception:
                        continue
                return None

            # Prefer the largest HTML file (main output)
            best = max(html_files, key=lambda n: zf.getinfo(n).file_size)
            raw = zf.read(best)

            # Try UTF-8, fall back to latin-1 (never fails)
            for encoding in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    return raw.decode(encoding)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")

    except zipfile.BadZipFile as e:
        logger.error(f"Corrupt Sarvam output ZIP: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to extract HTML from Sarvam output: {e}")
        return None


def _extract_metadata_page_texts(zip_path: Path) -> list[str]:
    """Extract per-page text from Sarvam metadata JSON files in the ZIP.

    Sarvam ZIPs contain ``metadata/page_001.json``, ``metadata/page_002.json``,
    etc.  Each has a ``blocks`` list with text content.  This serves as a
    reliable fallback when the HTML parser can't find page boundaries.

    Returns:
        list[str] — one text string per page, ordered by page number.
        Empty list if metadata files are missing or unparseable.
    """
    import json as _json

    try:
        if not zip_path.exists():
            return []

        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find metadata page files and sort by page number
            meta_files = sorted(
                [n for n in zf.namelist() if re.match(r'metadata/page_\d+\.json$', n)],
            )
            if not meta_files:
                return []

            page_texts: list[str] = []
            for meta_file in meta_files:
                try:
                    raw = zf.read(meta_file).decode("utf-8", errors="replace")
                    meta = _json.loads(raw)
                    blocks = meta.get("blocks", [])

                    # Concatenate text from all blocks on this page
                    texts: list[str] = []
                    for block in blocks:
                        # Each block may have 'text' or 'content' field
                        text = block.get("text", "") or block.get("content", "")
                        if text and text.strip():
                            texts.append(text.strip())

                    page_text = "\n".join(texts)
                    page_texts.append(page_text)
                except Exception as e:
                    logger.debug(f"Failed to parse {meta_file}: {e}")
                    page_texts.append("")

            # Remove trailing empty pages
            while page_texts and not page_texts[-1]:
                page_texts.pop()

            if page_texts:
                logger.info(
                    f"Sarvam metadata: extracted text from {len(page_texts)} "
                    f"page JSON files ({sum(len(t) for t in page_texts):,} chars total)"
                )
            return page_texts

    except Exception as e:
        logger.debug(f"Failed to extract metadata page texts: {e}")
        return []


# ── Drop-in replacement for Tesseract run_ocr_on_pages ─────────────
async def run_sarvam_on_pages(
    file_path: "str | Path",
    text_data: dict,
    *,
    on_progress=None,
) -> dict:
    """Run Sarvam OCR on LOW-quality pages (replaces Tesseract fallback).

    If Sarvam already processed this document (Stage 1b), pages with
    ``extraction_method='sarvam'`` are skipped — re-running would produce
    the same result.

    Mutates ``text_data`` in-place **and** returns it (matches the
    ``run_ocr_on_pages`` contract from ``ingestion.py``).

    Args:
        file_path: Path to the original PDF file.
        text_data: Output from ``extract_text_from_pdf()``.
        on_progress: Optional async callback ``(stage, message, detail)``.

    Returns:
        Updated ``text_data`` with Sarvam-enhanced pages where possible.
    """
    file_path = Path(file_path)
    pages = text_data.get("pages", [])

    # Find non-Sarvam LOW-quality pages
    low_indices = [
        i for i, p in enumerate(pages)
        if p.get("extraction_method") != "sarvam"
        and p.get("quality", {}).get("quality") == "LOW"
    ]
    if not low_indices:
        return text_data  # Nothing to improve

    # If HATAD Vision already ran (Stage 1b) and these pages are still LOW,
    # re-running won't help — it already gave its best.
    sarvam_count = sum(1 for p in pages if p.get("extraction_method") == "sarvam")
    if sarvam_count > 0:
        logger.info(
            f"HATAD Vision already processed {file_path.name} — "
            f"{len(low_indices)} LOW pages remain (cannot improve further)"
        )
        return text_data

    # Fresh HATAD Vision run for this document
    logger.info(
        f"Running HATAD Vision OCR on {file_path.name} "
        f"({len(low_indices)}/{len(pages)} LOW-quality pages)"
    )
    sarvam_result = await sarvam_extract_text(file_path, on_progress=on_progress)
    if sarvam_result is None:
        logger.warning(
            f"HATAD Vision unavailable for {file_path.name} — "
            f"LOW-quality pages will use pdfplumber text as-is"
        )
        return text_data

    # Merge: HATAD Vision replaces LOW-quality pages where it's better
    merged = merge_sarvam_with_pdfplumber(sarvam_result, text_data)

    # Copy merged pages back into text_data (in-place mutation)
    text_data["pages"] = merged["pages"]
    text_data["full_text"] = merged["full_text"]
    text_data["extraction_quality"] = merged["extraction_quality"]
    text_data["sarvam_pages"] = merged.get("sarvam_pages", 0)
    text_data["metadata"] = merged.get("metadata", text_data.get("metadata", {}))

    sarvam_used = merged.get("sarvam_pages", 0)
    if sarvam_used:
        logger.info(
            f"HATAD Vision enhanced {sarvam_used}/{len(pages)} pages for {file_path.name}"
        )
    return text_data


# ── Merge logic ────────────────────────────────────────────────────
def merge_sarvam_with_pdfplumber(
    sarvam_result: dict,
    pdfplumber_result: dict,
) -> dict:
    """Merge Sarvam and pdfplumber results, picking the best text per page.

    Decision logic per page:
      - If Sarvam produced significantly more text (>1.3× chars) → Sarvam wins.
      - If page contains Tamil and Sarvam text is non-garbled → Sarvam wins.
      - If Sarvam quality is HIGH and pdfplumber is LOW → Sarvam wins.
      - Otherwise → keep pdfplumber (it preserves tables and layout better).

    Returns:
        Merged result dict in the same format as ``extract_text_from_pdf()``.
    """
    from app.pipeline.utils import has_tamil, detect_garbled_tamil

    pdf_pages = pdfplumber_result.get("pages", [])
    sarvam_pages = sarvam_result.get("pages", [])

    merged_pages = []
    full_text_parts = []
    qualities = []
    sarvam_used = 0

    # Align by page number (Sarvam may have different page count)
    max_pages = max(len(pdf_pages), len(sarvam_pages))

    for i in range(max_pages):
        pdf_page = pdf_pages[i] if i < len(pdf_pages) else None
        sarvam_page = sarvam_pages[i] if i < len(sarvam_pages) else None

        if pdf_page and not sarvam_page:
            # Only pdfplumber has this page
            merged_pages.append(pdf_page)
            qualities.append(pdf_page["quality"]["quality"])
            full_text_parts.append(f"--- PAGE {pdf_page['page_number']} ---\n{pdf_page['text']}")
            continue

        if sarvam_page and not pdf_page:
            # Only Sarvam has this page
            merged_pages.append(sarvam_page)
            qualities.append(sarvam_page["quality"]["quality"])
            full_text_parts.append(f"--- PAGE {sarvam_page['page_number']} ---\n{sarvam_page['text']}")
            sarvam_used += 1
            continue

        # Both have this page — decide which is better
        pdf_text = pdf_page["text"]
        sarvam_text = sarvam_page["text"]
        pdf_chars = pdf_page["quality"]["char_count"]
        sarvam_chars = sarvam_page["quality"]["char_count"]
        pdf_quality = pdf_page["quality"]["quality"]
        sarvam_quality = sarvam_page["quality"]["quality"]

        use_sarvam = False

        # Rule 1: Sarvam HIGH, pdfplumber LOW
        if sarvam_quality == "HIGH" and pdf_quality == "LOW":
            use_sarvam = True

        # Rule 2: Sarvam has significantly more text
        elif sarvam_chars > pdf_chars * 1.3 and sarvam_chars > 50:
            use_sarvam = True

        # Rule 3: Tamil content — prefer Sarvam if it's not garbled
        elif has_tamil(sarvam_text) and sarvam_chars >= pdf_chars * 0.8:
            is_garbled, _quality, _reason = detect_garbled_tamil(sarvam_text)
            if not is_garbled:
                use_sarvam = True

        # Rule 4: pdfplumber has garbled Tamil, Sarvam doesn't
        elif has_tamil(pdf_text):
            pdf_garbled, _, _ = detect_garbled_tamil(pdf_text)
            if pdf_garbled and sarvam_chars > 30:
                sarvam_garbled, _, _ = detect_garbled_tamil(sarvam_text) if has_tamil(sarvam_text) else (False, 1.0, "")
                if not sarvam_garbled:
                    use_sarvam = True

        if use_sarvam:
            # Keep pdfplumber tables (Sarvam doesn't extract them)
            page = {**sarvam_page, "tables": pdf_page.get("tables", [])}
            merged_pages.append(page)
            qualities.append(sarvam_quality)
            full_text_parts.append(f"--- PAGE {sarvam_page['page_number']} ---\n{sarvam_text}")
            sarvam_used += 1
        else:
            merged_pages.append(pdf_page)
            qualities.append(pdf_quality)
            full_text_parts.append(f"--- PAGE {pdf_page['page_number']} ---\n{pdf_text}")

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
        "ocr_pages": pdfplumber_result.get("ocr_pages", 0),
        "sarvam_pages": sarvam_used,
        "metadata": {
            **pdfplumber_result.get("metadata", {}),
            "sarvam_pages_used": sarvam_used,
        },
    }

    logger.info(
        f"Sarvam merge: {sarvam_used}/{len(merged_pages)} pages from Sarvam AI, "
        f"rest from pdfplumber (overall quality: {overall})"
    )
    return merged
