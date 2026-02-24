"""PDF text extraction with pdfplumber + Tesseract OCR fallback.

Strategy:
  1. Try pdfplumber (fast, preserves layout for text-selectable PDFs).
  2. Score each page's text quality (chars, (cid:) ratio, word count).
  3. If a page is LOW quality → run Tesseract OCR on a rendered image.
  4. Return merged results with per-page extraction_method metadata.
"""

import base64
import io
import logging
import os
import re
import shutil
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

# ── OCR configuration ──────────────────────────────────────────────
# Tesseract binary path — auto-detect common Windows locations
_TESSERACT_CANDIDATES = [
    os.getenv("TESSERACT_CMD", ""),
    str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "tesseract",  # system PATH
]

# Poppler bin path for pdf2image
_POPPLER_CANDIDATES = [
    os.getenv("POPPLER_PATH", ""),
    str(Path(os.environ.get("USERPROFILE", "")) / "poppler" / "poppler-24.08.0" / "Library" / "bin"),
    str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "poppler" / "poppler-24.08.0" / "Library" / "bin"),
    r"C:\Program Files\poppler\Library\bin",
]

# Quality thresholds
MIN_CHARS_PER_PAGE = 50          # Pages with fewer chars → OCR
CID_RATIO_THRESHOLD = 0.15      # Pages with >15% (cid:XX) → OCR
MIN_WORDS_PER_PAGE = 10          # Pages with fewer words → OCR
OCR_DPI = 300                    # DPI for rendering pages to images
OCR_LANGUAGES = "eng+tam"        # English + Tamil


def _find_binary(candidates: list[str], name: str) -> str | None:
    """Return the first candidate path that exists or is in PATH."""
    for path in candidates:
        if not path:
            continue
        p = Path(path)
        if p.is_file():
            return str(p)
        # Check bare name in PATH
        if shutil.which(path):
            return path
    return None


def _init_poppler() -> bool:
    """Lazy-initialize poppler (pdf2image) for page rendering.

    Separated from Tesseract so that vision-based classification and
    extraction can render PDF pages even when Tesseract is not installed.
    """
    global _poppler_available, _poppler_path
    if hasattr(_init_poppler, "_done"):
        return _poppler_available

    _init_poppler._done = True
    _poppler_available = False
    _poppler_path = None

    try:
        from pdf2image import convert_from_path  # noqa: F401

        for candidate in _POPPLER_CANDIDATES:
            if not candidate:
                continue
            p = Path(candidate)
            if p.is_dir() and (p / "pdftoppm.exe").exists():
                _poppler_path = str(p)
                break
            elif p.is_dir() and (p / "pdftoppm").exists():
                _poppler_path = str(p)
                break
            elif p.is_file() and p.stem == "pdftoppm":
                _poppler_path = str(p.parent)
                break

        logger.info(f"Poppler (pdf2image) ready (poppler_path={_poppler_path})")
        _poppler_available = True
        return True

    except ImportError as e:
        logger.warning(f"pdf2image not installed ({e}) — page rendering disabled")
        return False
    except Exception as e:
        logger.warning(f"Poppler initialization failed: {e} — page rendering disabled")
        return False


def _init_ocr() -> bool:
    """Lazy-initialize Tesseract OCR.  Returns True if OCR is available.

    Requires both pytesseract and poppler (via _init_poppler).
    """
    global _ocr_available
    if hasattr(_init_ocr, "_done"):
        return _ocr_available

    _init_ocr._done = True
    _ocr_available = False

    # Poppler must be available first (shared with render_pages_as_images)
    if not _init_poppler():
        return False

    try:
        import pytesseract  # noqa: F401

        tess_cmd = _find_binary(_TESSERACT_CANDIDATES, "tesseract")
        if not tess_cmd:
            logger.warning("Tesseract binary not found — OCR fallback disabled")
            return False
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract OCR v{version} ready (cmd={tess_cmd})")
        _ocr_available = True
        return True

    except ImportError as e:
        logger.warning(f"pytesseract not installed ({e}) — OCR fallback disabled")
        return False
    except Exception as e:
        logger.warning(f"OCR initialization failed: {e} — OCR fallback disabled")
        return False


_poppler_available = False
_ocr_available = False
_poppler_path = None


# ── Text quality assessment ────────────────────────────────────────
def _assess_page_quality(text: str) -> dict:
    """Score a page's extracted text quality.

    Returns:
        {
            "char_count": int,
            "word_count": int,
            "cid_count": int,       # number of (cid:XX) placeholders
            "cid_ratio": float,     # fraction of text that is (cid:XX)
            "quality": "HIGH" | "LOW",
            "reason": str | None,   # why quality is LOW
        }
    """
    char_count = len(text.strip())
    words = text.split()
    word_count = len(words)

    # Count (cid:XX) placeholders — indicates broken font encoding
    cid_matches = re.findall(r'\(cid:\d+\)', text)
    cid_count = len(cid_matches)
    cid_chars = sum(len(m) for m in cid_matches)
    cid_ratio = cid_chars / max(char_count, 1)

    reason = None
    quality = "HIGH"

    if char_count < MIN_CHARS_PER_PAGE:
        quality = "LOW"
        reason = f"too few characters ({char_count})"
    elif cid_ratio > CID_RATIO_THRESHOLD:
        quality = "LOW"
        reason = f"high (cid:) ratio ({cid_ratio:.0%})"
    elif word_count < MIN_WORDS_PER_PAGE:
        quality = "LOW"
        reason = f"too few words ({word_count})"

    return {
        "char_count": char_count,
        "word_count": word_count,
        "cid_count": cid_count,
        "cid_ratio": round(cid_ratio, 3),
        "quality": quality,
        "reason": reason,
    }


def _ocr_page(file_path: Path, page_number: int) -> str:
    """Run Tesseract OCR on a single page of a PDF.

    Args:
        file_path: Path to the PDF file
        page_number: 1-based page number

    Returns:
        OCR'd text string, or empty string on failure
    """
    if not _ocr_available:
        return ""

    try:
        import pytesseract
        from pdf2image import convert_from_path

        # Convert just the one page to an image
        images = convert_from_path(
            str(file_path),
            first_page=page_number,
            last_page=page_number,
            dpi=OCR_DPI,
            poppler_path=_poppler_path,
        )

        if not images:
            logger.warning(f"pdf2image returned no images for page {page_number}")
            return ""

        # Run Tesseract
        ocr_text = pytesseract.image_to_string(
            images[0],
            lang=OCR_LANGUAGES,
            config="--psm 6",  # Assume uniform block of text
        )

        # Clean up: collapse excessive whitespace, strip control chars
        ocr_text = re.sub(r'[^\S\n]+', ' ', ocr_text)  # collapse spaces (keep newlines)
        ocr_text = re.sub(r'\n{3,}', '\n\n', ocr_text)  # max 2 consecutive newlines
        ocr_text = ocr_text.strip()

        return ocr_text

    except Exception as e:
        logger.warning(f"OCR failed on page {page_number}: {e}")
        return ""


def _ocr_full_document(file_path: Path, total_pages: int) -> list[str]:
    """Run Tesseract OCR on all pages at once (more efficient than page-by-page).

    Returns:
        List of OCR'd text strings, one per page.
    """
    if not _ocr_available:
        return [""] * total_pages

    try:
        import pytesseract
        from pdf2image import convert_from_path

        logger.info(f"Running full-document OCR on {file_path.name} ({total_pages} pages, {OCR_DPI} DPI)")

        images = convert_from_path(
            str(file_path),
            dpi=OCR_DPI,
            poppler_path=_poppler_path,
        )

        ocr_texts = []
        for i, img in enumerate(images):
            try:
                text = pytesseract.image_to_string(
                    img, lang=OCR_LANGUAGES, config="--psm 6"
                )
                text = re.sub(r'[^\S\n]+', ' ', text)
                text = re.sub(r'\n{3,}', '\n\n', text)
                ocr_texts.append(text.strip())
            except Exception as e:
                logger.warning(f"OCR failed on page {i + 1}: {e}")
                ocr_texts.append("")

        return ocr_texts

    except Exception as e:
        logger.error(f"Full-document OCR failed: {e}")
        return [""] * total_pages


def extract_text_from_pdf(file_path: str | Path, *, skip_ocr: bool = True) -> dict:
    """Extract all text from a PDF file, page by page.

    Strategy:
      1. Try pdfplumber first (fast, layout-preserving).
      2. Assess text quality per page.
      3. If skip_ocr is False AND a page is LOW quality, fallback to Tesseract OCR.
         When skip_ocr is True (default), pdfplumber text is kept as-is —
         vision extractors will re-render the PDF for extraction anyway,
         and the pdfplumber text is sufficient for classification & RAG.

    Args:
        file_path: Path to the PDF file.
        skip_ocr: If True (default), skip Tesseract OCR even for LOW quality
                  pages.  Set to False only for text-only extractors (EC) that
                  need high quality input.

    Returns:
        {
            "total_pages": int,
            "pages": [{"page_number": 1, "text": "...", "tables": [...],
                        "extraction_method": "pdfplumber"|"ocr"|"ocr_fallback",
                        "quality": {...}}],
            "full_text": "all pages concatenated",
            "extraction_quality": "HIGH"|"MIXED"|"LOW"|"EMPTY",
            "ocr_pages": int,  # number of pages that used OCR
            "metadata": {...}
        }
    """
    file_path = Path(file_path)
    pages = []
    full_text_parts = []
    qualities = []
    ocr_page_count = 0

    # ── Step 1: pdfplumber extraction ──
    pdfplumber_pages = []
    pdfplumber_ok = True
    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""

                # Try extracting tables
                tables = []
                try:
                    raw_tables = page.extract_tables() or []
                    for table in raw_tables:
                        cleaned = []
                        for row in table:
                            cleaned_row = [
                                cell.strip() if isinstance(cell, str) else (cell or "")
                                for cell in row
                            ]
                            cleaned.append(cleaned_row)
                        if cleaned:
                            tables.append(cleaned)
                except Exception as e:
                    logger.warning(f"Table extraction failed on page {i+1}: {e}")

                quality = _assess_page_quality(page_text)
                pdfplumber_pages.append({
                    "page_number": i + 1,
                    "text": page_text,
                    "tables": tables,
                    "quality": quality,
                })
    except Exception as e:
        logger.error(f"pdfplumber failed on {file_path.name}: {e}")
        pdfplumber_ok = False
        metadata = {}

    # ── Step 2: Determine which pages need OCR ──
    _init_ocr()

    if not pdfplumber_ok:
        # pdfplumber completely failed — try full OCR
        logger.info(f"pdfplumber failed entirely; attempting full OCR on {file_path.name}")
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(
                str(file_path), dpi=OCR_DPI, poppler_path=_poppler_path,
                first_page=1, last_page=1,  # Just to get page count
            )
            # Re-do with all pages
            total = len(convert_from_path(str(file_path), dpi=72, poppler_path=_poppler_path))
        except Exception:
            total = 0

        if total > 0:
            ocr_texts = _ocr_full_document(file_path, total)
            for i, ocr_text in enumerate(ocr_texts):
                quality = _assess_page_quality(ocr_text)
                pages.append({
                    "page_number": i + 1,
                    "text": ocr_text,
                    "tables": [],
                    "extraction_method": "ocr",
                    "quality": quality,
                })
                qualities.append(quality["quality"])
                full_text_parts.append(f"--- PAGE {i + 1} ---\n{ocr_text}")
                ocr_page_count += 1
        else:
            # Nothing works — return empty
            pages.append({
                "page_number": 1,
                "text": "",
                "tables": [],
                "extraction_method": "failed",
                "quality": _assess_page_quality(""),
            })
            qualities.append("LOW")
            full_text_parts.append("--- PAGE 1 ---\n")

    else:
        # Identify pages needing OCR
        low_quality_indices = [
            i for i, p in enumerate(pdfplumber_pages)
            if p["quality"]["quality"] == "LOW"
        ]

        # Batch OCR for all low-quality pages at once (more efficient)
        # When skip_ocr=True (default), we skip this entirely — vision model
        # will read the PDF images directly, making OCR redundant.
        ocr_results: dict[int, str] = {}
        if low_quality_indices and _ocr_available and not skip_ocr:
            low_count = len(low_quality_indices)
            total_count = len(pdfplumber_pages)
            logger.info(
                f"{file_path.name}: {low_count}/{total_count} pages have low quality text, "
                f"running OCR fallback"
            )

            if low_count == total_count:
                # All pages are low quality — do full-document OCR (faster)
                ocr_texts = _ocr_full_document(file_path, total_count)
                for i, ocr_text in enumerate(ocr_texts):
                    if ocr_text:
                        ocr_results[i] = ocr_text
            else:
                # Only OCR specific pages
                for idx in low_quality_indices:
                    page_num = pdfplumber_pages[idx]["page_number"]
                    ocr_text = _ocr_page(file_path, page_num)
                    if ocr_text:
                        ocr_results[idx] = ocr_text
        elif low_quality_indices and not _ocr_available and not skip_ocr:
            logger.warning(
                f"{file_path.name}: {len(low_quality_indices)} pages have low quality text "
                f"but OCR is not available"
            )
        elif low_quality_indices and skip_ocr:
            logger.info(
                f"{file_path.name}: {len(low_quality_indices)} LOW quality pages "
                f"(OCR skipped — vision model will handle extraction)"
            )

        # ── Step 3: Merge results ──
        for i, pp in enumerate(pdfplumber_pages):
            if i in ocr_results:
                ocr_text = ocr_results[i]
                ocr_quality = _assess_page_quality(ocr_text)

                # Decide: use OCR if it's genuinely better
                use_ocr = False
                reason_for_low = pp["quality"].get("reason", "")

                if "cid:" in reason_for_low:
                    # pdfplumber has garbled font encoding — OCR almost always better
                    use_ocr = ocr_quality["char_count"] > MIN_CHARS_PER_PAGE
                elif ocr_quality["char_count"] > pp["quality"]["char_count"] * 1.5:
                    # OCR produced significantly more text
                    use_ocr = True
                elif ocr_quality["quality"] == "HIGH" and pp["quality"]["quality"] == "LOW":
                    # OCR quality is better
                    use_ocr = True

                if use_ocr:
                    # OCR is significantly better — use it
                    pages.append({
                        "page_number": pp["page_number"],
                        "text": ocr_text,
                        "tables": pp["tables"],  # Keep pdfplumber tables if any
                        "extraction_method": "ocr_fallback",
                        "quality": ocr_quality,
                    })
                    ocr_page_count += 1
                    qualities.append(ocr_quality["quality"])
                    full_text_parts.append(f"--- PAGE {pp['page_number']} ---\n{ocr_text}")
                    logger.info(
                        f"Page {pp['page_number']}: OCR used "
                        f"(pdfplumber={pp['quality']['char_count']} chars → OCR={ocr_quality['char_count']} chars)"
                    )
                else:
                    # pdfplumber was actually fine or OCR wasn't better
                    pages.append({
                        "page_number": pp["page_number"],
                        "text": pp["text"],
                        "tables": pp["tables"],
                        "extraction_method": "pdfplumber",
                        "quality": pp["quality"],
                    })
                    qualities.append(pp["quality"]["quality"])
                    full_text_parts.append(f"--- PAGE {pp['page_number']} ---\n{pp['text']}")
            else:
                pages.append({
                    "page_number": pp["page_number"],
                    "text": pp["text"],
                    "tables": pp["tables"],
                    "extraction_method": "pdfplumber",
                    "quality": pp["quality"],
                })
                qualities.append(pp["quality"]["quality"])
                full_text_parts.append(f"--- PAGE {pp['page_number']} ---\n{pp['text']}")

    # ── Overall quality assessment ──
    if not qualities or all(q == "LOW" for q in qualities):
        overall_quality = "LOW" if qualities else "EMPTY"
    elif all(q == "HIGH" for q in qualities):
        overall_quality = "HIGH"
    else:
        overall_quality = "MIXED"

    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text_parts),
        "extraction_quality": overall_quality,
        "ocr_pages": ocr_page_count,
        "metadata": {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size,
            **{k: str(v) for k, v in metadata.items() if v},
        },
    }


def run_ocr_on_pages(file_path: str | Path, text_data: dict) -> dict:
    """Run Tesseract OCR on LOW-quality pages of a previously-extracted document.

    This is the *lazy* OCR path: call it only when a downstream consumer
    actually needs high-quality text (e.g. ECExtractor text-based pipeline,
    or a vision-extractor fallback path).

    Mutates ``text_data`` in-place **and** returns it.

    Args:
        file_path: Path to the original PDF file.
        text_data: Output from ``extract_text_from_pdf()``.

    Returns:
        Updated ``text_data`` with OCR-enhanced pages where possible.
    """
    file_path = Path(file_path)
    _init_ocr()
    if not _ocr_available:
        logger.warning(f"run_ocr_on_pages: OCR not available, returning unchanged text_data")
        return text_data

    pages = text_data.get("pages", [])
    low_indices = [
        i for i, p in enumerate(pages)
        if p.get("quality", {}).get("quality") == "LOW"
    ]
    if not low_indices:
        return text_data

    total = len(pages)
    logger.info(f"run_ocr_on_pages: {len(low_indices)}/{total} pages need OCR in {file_path.name}")

    ocr_page_count = text_data.get("ocr_pages", 0)

    if len(low_indices) == total:
        # All pages low quality — full document OCR (faster than page-by-page)
        ocr_texts = _ocr_full_document(file_path, total)
        for i, ocr_text in enumerate(ocr_texts):
            if not ocr_text:
                continue
            ocr_quality = _assess_page_quality(ocr_text)
            old_quality = pages[i].get("quality", {})
            if ocr_quality["char_count"] > old_quality.get("char_count", 0) * 1.2 or \
               (ocr_quality["quality"] == "HIGH" and old_quality.get("quality") == "LOW"):
                pages[i]["text"] = ocr_text
                pages[i]["extraction_method"] = "ocr_fallback"
                pages[i]["quality"] = ocr_quality
                ocr_page_count += 1
    else:
        for idx in low_indices:
            page_num = pages[idx]["page_number"]
            ocr_text = _ocr_page(file_path, page_num)
            if not ocr_text:
                continue
            ocr_quality = _assess_page_quality(ocr_text)
            old_quality = pages[idx].get("quality", {})
            reason = old_quality.get("reason", "")
            use_ocr = False
            if "cid:" in reason:
                use_ocr = ocr_quality["char_count"] > MIN_CHARS_PER_PAGE
            elif ocr_quality["char_count"] > old_quality.get("char_count", 0) * 1.5:
                use_ocr = True
            elif ocr_quality["quality"] == "HIGH" and old_quality.get("quality") == "LOW":
                use_ocr = True
            if use_ocr:
                pages[idx]["text"] = ocr_text
                pages[idx]["extraction_method"] = "ocr_fallback"
                pages[idx]["quality"] = ocr_quality
                ocr_page_count += 1
                logger.info(f"  Page {page_num}: OCR improved quality "
                            f"({old_quality.get('char_count', 0)} → {ocr_quality['char_count']} chars)")

    # Rebuild full_text and quality assessment
    text_data["ocr_pages"] = ocr_page_count
    full_parts = [
        f"--- PAGE {p['page_number']} ---\n{p['text']}" for p in pages
    ]
    text_data["full_text"] = "\n\n".join(full_parts)
    qualities = [p.get("quality", {}).get("quality", "LOW") for p in pages]
    if all(q == "HIGH" for q in qualities):
        text_data["extraction_quality"] = "HIGH"
    elif all(q == "LOW" for q in qualities):
        text_data["extraction_quality"] = "LOW"
    else:
        text_data["extraction_quality"] = "MIXED"

    return text_data


def extract_text_chunks(file_path: str | Path, chunk_size: int = 10) -> list[dict]:
    """Extract text in page chunks for large documents (e.g., 65-page EC).
    
    Returns list of chunks, each containing text from `chunk_size` pages.
    """
    result = extract_text_from_pdf(file_path)
    chunks = []
    
    for i in range(0, len(result["pages"]), chunk_size):
        chunk_pages = result["pages"][i:i + chunk_size]
        chunk_text = "\n\n".join(
            f"--- PAGE {p['page_number']} ---\n{p['text']}" 
            for p in chunk_pages
        )
        chunks.append({
            "chunk_index": len(chunks),
            "start_page": chunk_pages[0]["page_number"],
            "end_page": chunk_pages[-1]["page_number"],
            "text": chunk_text,
            "tables": [t for p in chunk_pages for t in p["tables"]],
        })
    
    return chunks


def render_pages_as_images(file_path: Path, dpi: int = 200) -> list[str]:
    """Render each page of a PDF as a base64-encoded PNG image.

    Uses poppler (pdf2image) to convert pages. Returns a list of base64
    strings suitable for sending to vision LLMs via the Ollama images API.

    Results are cached per (file_path, dpi) so the same document isn't
    rendered twice during classification → extraction.

    Args:
        file_path: Path to the PDF file.
        dpi: Resolution for rendering (default 200 — good balance of
             quality vs size for Tamil text).

    Returns:
        List of base64-encoded PNG strings, one per page.
    """
    cache_key = (str(file_path), dpi)
    if cache_key in _image_cache:
        logger.debug(f"Image cache hit for {file_path.name} (dpi={dpi})")
        return _image_cache[cache_key]
    if not _init_poppler():
        raise RuntimeError(
            "Vision extraction requires poppler (pdf2image). "
            "Install poppler and ensure it is in PATH or set POPPLER_PATH."
        )

    from pdf2image import convert_from_path

    logger.info(f"Rendering {file_path.name} as images (dpi={dpi})...")
    images = convert_from_path(
        str(file_path),
        dpi=dpi,
        poppler_path=_poppler_path,
        fmt="png",
    )

    base64_images = []
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        base64_images.append(b64)
        logger.debug(f"  Page {i + 1}: {len(b64):,} chars base64 ({len(buf.getvalue()):,} bytes PNG)")

    logger.info(f"Rendered {len(base64_images)} page(s) from {file_path.name}")
    _image_cache[cache_key] = base64_images
    return base64_images


# ── Image render cache ──
_image_cache: dict[tuple[str, int], list[str]] = {}


def clear_image_cache():
    """Clear the rendered image cache (call at pipeline end)."""
    _image_cache.clear()
