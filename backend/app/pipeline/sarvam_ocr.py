"""Sarvam AI Document Intelligence integration for Tamil OCR.

Sarvam AI specialises in Indian-language document digitisation.  This module
wraps the ``sarvamai`` SDK to provide a drop-in OCR path that runs in
parallel with the existing pdfplumber + Tesseract pipeline.

Flow:
  1. Upload PDF to Sarvam → async job starts server-side.
  2. Poll until complete (or timeout).
  3. Download HTML output → parse per-page text.
  4. Merge with pdfplumber result — best quality wins per page.

Feature-gated: when ``SARVAM_API_KEY`` is empty the functions are no-ops.
"""

import asyncio
import io
import logging
import re
import tempfile
import zipfile
from pathlib import Path

from app.config import (
    SARVAM_API_KEY,
    SARVAM_ENABLED,
    SARVAM_LANGUAGE,
    SARVAM_POLL_INTERVAL,
    SARVAM_TIMEOUT,
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


# ── HTML parsing ───────────────────────────────────────────────────
def _parse_sarvam_html(html_content: str) -> list[str]:
    """Extract per-page text from Sarvam HTML output.

    Sarvam's output format uses page-break markers or ``<div>`` sections
    per page.  We try BeautifulSoup first (clean), then fall back to a
    simple regex-based splitter.

    Returns:
        list[str] — one text string per page.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Strategy 1: look for explicit page containers
        # Sarvam may use <div class="page"> or data-page-number attributes
        page_divs = soup.find_all("div", attrs={"data-page-number": True})
        if page_divs:
            return [div.get_text(separator="\n", strip=True) for div in page_divs]

        # Strategy 2: look for class="page" divs
        page_divs = soup.find_all("div", class_="page")
        if page_divs:
            return [div.get_text(separator="\n", strip=True) for div in page_divs]

        # Strategy 3: split on <hr> or page-break CSS
        # Many document converters insert <hr> between pages
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

        # Strategy 4: CSS page-break-before / page-break-after
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
async def sarvam_extract_text(file_path: Path, *, on_progress=None) -> dict | None:
    """Upload a PDF to Sarvam AI and return extracted text.

    Runs the synchronous ``sarvamai`` SDK in a thread pool to avoid
    blocking the FastAPI event loop.

    Args:
        file_path: Path to the PDF file.
        on_progress: Optional async callback ``(stage, message, detail)``.

    Returns:
        dict matching ``extract_text_from_pdf()`` format, or ``None``
        if Sarvam is disabled / unavailable / fails.
    """
    if not SARVAM_ENABLED:
        return None
    if not _check_sarvam_sdk():
        return None

    try:
        result = await asyncio.to_thread(
            _sarvam_extract_sync, file_path
        )
        return result
    except Exception as e:
        logger.error(f"Sarvam AI extraction failed for {file_path.name}: {e}")
        if on_progress:
            try:
                await on_progress(
                    "extraction",
                    f"Sarvam AI failed for {file_path.name}: {e} (falling back to local OCR)",
                    {"type": "sarvam_error", "error": str(e)},
                )
            except Exception:
                pass
        return None


def _sarvam_extract_sync(file_path: Path) -> dict | None:
    """Synchronous Sarvam extraction — called via ``asyncio.to_thread``."""
    from sarvamai import SarvamAI

    logger.info(f"Sarvam AI: starting extraction for {file_path.name}")

    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    # 1. Create job
    job = client.document_intelligence.create_job(
        language=SARVAM_LANGUAGE,
        output_format="html",
    )
    logger.info(f"Sarvam AI: job created — {job.job_id}")

    # 2. Upload file
    job.upload_file(str(file_path))
    logger.info(f"Sarvam AI: uploaded {file_path.name}")

    # 3. Start processing
    job.start()
    logger.info(f"Sarvam AI: processing started for {file_path.name}")

    # 4. Wait for completion (with timeout)
    import time
    elapsed = 0
    status = None
    while elapsed < SARVAM_TIMEOUT:
        status = job.get_status()
        state = getattr(status, "job_state", None) or getattr(status, "state", None) or str(status)
        logger.debug(f"Sarvam AI: job {job.job_id} state={state} ({elapsed}s)")

        if state in ("completed", "COMPLETED", "success", "SUCCESS"):
            break
        if state in ("failed", "FAILED", "error", "ERROR"):
            logger.error(f"Sarvam AI job failed: {state}")
            return None

        time.sleep(SARVAM_POLL_INTERVAL)
        elapsed += SARVAM_POLL_INTERVAL
    else:
        logger.error(f"Sarvam AI: timeout after {SARVAM_TIMEOUT}s for {file_path.name}")
        return None

    # 5. Download output ZIP
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "output.zip"
        job.download_output(str(output_path))
        logger.info(f"Sarvam AI: downloaded output for {file_path.name}")

        # 6. Extract HTML from ZIP
        html_content = _extract_html_from_zip(output_path)
        if not html_content:
            logger.warning(f"Sarvam AI: no HTML content found in output for {file_path.name}")
            return None

    # 7. Parse HTML → per-page text
    page_texts = _parse_sarvam_html(html_content)
    if not page_texts:
        logger.warning(f"Sarvam AI: HTML parsing produced no pages for {file_path.name}")
        return None

    logger.info(f"Sarvam AI: extracted {len(page_texts)} pages from {file_path.name}")

    # 8. Build result dict matching extract_text_from_pdf() format
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

    # Overall quality
    if not qualities or all(q == "LOW" for q in qualities):
        overall = "LOW" if qualities else "EMPTY"
    elif all(q == "HIGH" for q in qualities):
        overall = "HIGH"
    else:
        overall = "MIXED"

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text_parts),
        "extraction_quality": overall,
        "ocr_pages": 0,
        "sarvam_pages": len(pages),
        "metadata": {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            "extraction_source": "sarvam_ai",
        },
    }


def _extract_html_from_zip(zip_path: Path) -> str | None:
    """Open a Sarvam output ZIP and return the first HTML file's content."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            html_files = [
                n for n in zf.namelist()
                if n.lower().endswith((".html", ".htm"))
            ]
            if not html_files:
                # Try any file with HTML-like content
                for name in zf.namelist():
                    content = zf.read(name).decode("utf-8", errors="replace")
                    if "<html" in content.lower() or "<body" in content.lower():
                        return content
                return None

            # Prefer the largest HTML file (main output)
            best = max(html_files, key=lambda n: zf.getinfo(n).file_size)
            return zf.read(best).decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Failed to extract HTML from Sarvam output: {e}")
        return None


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

    # If Sarvam already ran (Stage 1b) and these pages are still LOW,
    # re-running won't help — Sarvam already gave its best.
    sarvam_count = sum(1 for p in pages if p.get("extraction_method") == "sarvam")
    if sarvam_count > 0:
        logger.info(
            f"Sarvam already processed {file_path.name} — "
            f"{len(low_indices)} LOW pages remain (Sarvam can't improve further)"
        )
        return text_data

    # Fresh Sarvam run for this document
    logger.info(
        f"Running Sarvam OCR on {file_path.name} "
        f"({len(low_indices)}/{len(pages)} LOW-quality pages)"
    )
    sarvam_result = await sarvam_extract_text(file_path, on_progress=on_progress)
    if sarvam_result is None:
        logger.warning(
            f"Sarvam OCR unavailable for {file_path.name} — "
            f"LOW-quality pages will use pdfplumber text as-is"
        )
        return text_data

    # Merge: Sarvam replaces LOW-quality pages where it's better
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
            f"Sarvam enhanced {sarvam_used}/{len(pages)} pages for {file_path.name}"
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
