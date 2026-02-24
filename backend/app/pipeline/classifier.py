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
from app.config import (
    PROMPTS_DIR, DOCUMENT_TYPES,
    CLASSIFY_MAX_CHARS, CLASSIFY_MAX_PAGES, CLASSIFY_MAX_TOKENS,
    CLASSIFY_RETRY_MAX_CHARS, CLASSIFY_RETRY_MAX_PAGES,
)
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


# ── Deterministic keyword pre-classifier ──────────────────────────
# High-signal Tamil/English patterns mapped to document types.
# Each entry is (doc_type, compiled_regex).  Patterns are intentionally
# restrictive — false negatives go to the LLM (safe), false positives
# would bypass it (dangerous), so we only include very specific terms.

_KEYWORD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EC", re.compile(
        r"(?:சுமையின்மை\s*சான்றிதழ்|Encumbrance\s+Certificate|"
        r"Certificate\s+of\s+Encumbrance|வில்லங்கச்?\s*சான்று|"
        r"வில்லங்க\s*சான்றிதழ்|EC\s*(?:No|Number|எண்)[.:\s]|"
        r"ENCUMBRANCE\s+CERTIFICATE)",
        re.IGNORECASE | re.UNICODE)),
    ("PATTA", re.compile(
        r"(?:பட்டா\s*எண்|Patta\s*(?:No|Number|Chitta)[.:\s])",
        re.IGNORECASE | re.UNICODE)),
    ("A_REGISTER", re.compile(
        r"(?:அ[\-‐–]?\s*பதிவேடு|A[\-‐–]?\s*Register|A[\-‐–]?\s*Pathivedu)",
        re.IGNORECASE | re.UNICODE)),
    ("SALE_DEED", re.compile(
        r"(?:விற்பனை\s*பத்திரம்|Sale\s+Deed|SALE\s+DEED|"
        r"Deed\s+of\s+(?:Sale|Conveyance)|"
        r"விலை\s*ஆவணம்|"
        r"Certified\s+Copy\s+of\s+(?:R|Register)|"
        r"Book\s*(?:I|1)\s*/\s*\d+\s*/\s*\d{4})",
        re.IGNORECASE | re.UNICODE)),
    ("CHITTA", re.compile(
        r"(?:சிட்டா\s|Chitta\s*(?:No|Number|Extract)[.:\s])",
        re.IGNORECASE | re.UNICODE)),
    ("ADANGAL", re.compile(
        r"(?:அடங்கல்|Adangal|Village\s+Account)",
        re.IGNORECASE | re.UNICODE)),
    ("FMB", re.compile(
        r"(?:Field\s+Measurement\s+Book|FMB\s*(?:No|Sketch|Extract)|"
        r"வரைபட\s*எண்|நில\s*அளவை)",
        re.IGNORECASE | re.UNICODE)),
    ("LAYOUT_APPROVAL", re.compile(
        r"(?:Layout\s+Approval|CMDA|DTCP|வரைபட\s*ஒப்புதல்|"
        r"Planning\s+Permission|Approved\s+Layout)",
        re.IGNORECASE | re.UNICODE)),
    ("LEGAL_HEIR", re.compile(
        r"(?:Legal\s+Heir\s+Certificate|சட்ட\s*வாரிசு\s*சான்றிதழ்|"
        r"வாரிசு\s*சான்று)",
        re.IGNORECASE | re.UNICODE)),
    ("POA", re.compile(
        r"(?:Power\s+of\s+Attorney|அதிகார\s*பத்திரம்|General\s+Power|"
        r"Special\s+Power\s+of\s+Attorney)",
        re.IGNORECASE | re.UNICODE)),
    ("COURT_ORDER", re.compile(
        r"(?:Court\s+Order|நீதிமன்ற\s*ஆணை|Decree|"
        r"(?:District|High|Supreme)\s+Court|O\.?S\.?\s*No\.?)",
        re.IGNORECASE | re.UNICODE)),
    ("WILL", re.compile(
        r"(?:Last\s+Will|Testament|உயிலின்|உயில்\s)",
        re.IGNORECASE | re.UNICODE)),
    ("PARTITION_DEED", re.compile(
        r"(?:Partition\s+Deed|பாகப்\s*பிரிவினை\s*பத்திரம்|"
        r"Deed\s+of\s+Partition)",
        re.IGNORECASE | re.UNICODE)),
    ("GIFT_DEED", re.compile(
        r"(?:Gift\s+Deed|தான\s*பத்திரம்|Deed\s+of\s+Gift)",
        re.IGNORECASE | re.UNICODE)),
    ("RELEASE_DEED", re.compile(
        r"(?:Release\s+Deed|விடுதலை\s*பத்திரம்|"
        r"Deed\s+of\s+Release|Mortgage\s+Release)",
        re.IGNORECASE | re.UNICODE)),
]


def _get_keyword_hints(text: str, filename: str = "") -> tuple[str | None, list[str]]:
    """Return (single_match, all_matches) from keyword pre-classifier.

    Also checks filename for TNREGINET CCA patterns — these certified
    copies are almost always registered deeds (Sale Deed, Gift Deed, etc.).
    """
    matches: list[str] = []
    for doc_type, pattern in _KEYWORD_PATTERNS:
        if pattern.search(text):
            matches.append(doc_type)

    # TNREGINET Certified Copy Application filenames → hint at registered deed
    if filename and re.search(r'CCA[_\-]?Online', filename, re.IGNORECASE):
        if "SALE_DEED" not in matches:
            matches.append("SALE_DEED")

    # Disambiguation: EC documents naturally list sale deeds as transactions,
    # so SALE_DEED keywords often appear inside ECs.  When both match the
    # document header keywords ("Certificate of Encumbrance") are the real
    # signal — the SALE_DEED hit is just a transaction row.  EC wins.
    if "EC" in matches and "SALE_DEED" in matches:
        matches.remove("SALE_DEED")

    if len(matches) == 1:
        return matches[0], matches
    return None, matches


async def classify_document(
    extracted_text: dict,
    filename: str = "",
    on_progress: LLMProgressCallback | None = None,
    file_path: Path | None = None,
) -> dict:
    """Classify a document using deterministic keywords + LLM fallback.

    Strategy:
      1. Run keyword pre-classifier on cleaned text.
      2. If a single unambiguous match → return directly (skip LLM).
      3. Otherwise, call LLM with optional keyword hints.
      4. Compare LLM result vs keyword result — if they disagree,
         retry with expanded context (more pages, more chars) and an
         arbitration prompt.  This uses a real ambiguity signal rather
         than trusting LLM self-reported confidence.

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
    all_pages = extracted_text.get("pages", [])

    # ── Phase 1: Prepare text from best pages ──
    pages = _select_best_pages(all_pages, max_pages=CLASSIFY_MAX_PAGES)
    sample_text = "\n\n".join(p["text"] for p in pages if p.get("text"))
    sample_text = _strip_cid_placeholders(sample_text)
    sample_text = _collapse_repetitions(sample_text)
    sample_text = _dedup_high_freq_tokens(sample_text)

    if len(sample_text) > CLASSIFY_MAX_CHARS:
        logger.info(
            f"[{name_part}] Classification sample truncated: "
            f"{len(sample_text):,} → {CLASSIFY_MAX_CHARS:,} chars"
        )
        sample_text = sample_text[:CLASSIFY_MAX_CHARS]

    if not sample_text.strip():
        return {
            "document_type": "OTHER",
            "confidence": 0.0,
            "language": "unknown",
            "key_identifiers": [],
            "reasoning": "No text could be extracted for classification.",
            "keyword_classification": None,
        }

    # ── Phase 2: Keyword pre-classifier ──
    kw_single, kw_matches = _get_keyword_hints(sample_text, filename)

    if kw_single:
        # Unambiguous keyword match — skip LLM entirely
        logger.info(f"[{name_part}] Keyword pre-classifier: {kw_single} (skipping LLM)")
        if on_progress:
            await on_progress("classify", f"{name_part}: keyword match → {kw_single}", {
                "type": "classify_keyword_match",
                "filename": filename,
                "document_type": kw_single,
            })
        return {
            "document_type": kw_single,
            "confidence": 0.85,
            "language": "tamil_english",
            "key_identifiers": [],
            "reasoning": f"Deterministic keyword match: {kw_single}",
            "keyword_classification": kw_single,
        }

    # ── Phase 3: LLM classification ──
    system_prompt = _load_prompt("classify")

    # Include filename and keyword hints in prompt if available
    prompt_parts = []
    if filename:
        prompt_parts.append(f"Document filename: {filename}")
    if len(kw_matches) > 1:
        prompt_parts.append(
            f"Keyword analysis suggests possible types: {', '.join(kw_matches)}. "
            "Use these as hints but classify based on the actual content."
        )
    prompt_parts.append(f"Classify this document:\n\n{sample_text}")
    prompt = "\n\n".join(prompt_parts)

    result = await call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        expect_json=CLASSIFY_SCHEMA,
        task_label=f"Classify {name_part} ({total_pages}p, {len(sample_text):,} chars)",
        on_progress=on_progress,
        think=False,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )

    if result.get("document_type") not in DOCUMENT_TYPES:
        result["document_type"] = "OTHER"

    # Store keyword classification for downstream comparison
    result["keyword_classification"] = kw_matches[0] if len(kw_matches) == 1 else (
        kw_matches if kw_matches else None
    )

    # ── Phase 4: Disagreement-triggered retry ──
    # If keywords found matches and the LLM disagrees, retry with expanded
    # context.  This uses a real ambiguity signal instead of trusting the
    # LLM's self-reported confidence number.
    llm_type = result.get("document_type", "OTHER")
    llm_conf = result.get("confidence", 0)
    if isinstance(llm_conf, str):
        try:
            llm_conf = float(llm_conf)
        except (ValueError, TypeError):
            llm_conf = 0

    # Retry conditions:
    #   A) Keywords matched something but LLM disagrees (not OTHER)
    #   B) LLM says OTHER with low confidence and more pages are available
    needs_retry = (
        (kw_matches and llm_type not in kw_matches and llm_type != "OTHER")
        or (llm_type == "OTHER" and llm_conf < 0.80
            and len(all_pages) > CLASSIFY_MAX_PAGES)
    )
    if needs_retry:
        logger.info(
            f"[{name_part}] Classification disagreement: "
            f"keywords={kw_matches}, LLM={llm_type} — retrying with expanded context"
        )
        # Expand to more pages + more chars
        retry_pages = _select_best_pages(all_pages, max_pages=CLASSIFY_RETRY_MAX_PAGES)
        retry_text = "\n\n".join(p["text"] for p in retry_pages if p.get("text"))
        retry_text = _strip_cid_placeholders(retry_text)
        retry_text = _collapse_repetitions(retry_text)
        retry_text = _dedup_high_freq_tokens(retry_text)
        if len(retry_text) > CLASSIFY_RETRY_MAX_CHARS:
            retry_text = retry_text[:CLASSIFY_RETRY_MAX_CHARS]

        arbitration_prompt = (
            f"Document filename: {filename}\n\n"
            f"CLASSIFICATION DISAGREEMENT — please arbitrate:\n"
            f"- Keyword analysis suggests: {', '.join(kw_matches)}\n"
            f"- Initial classification was: {llm_type}\n\n"
            f"With this expanded context ({len(retry_pages)} pages, "
            f"{len(retry_text):,} chars), determine the correct document type.\n\n"
            f"Classify this document:\n\n{retry_text}"
        )

        retry_result = await call_llm(
            prompt=arbitration_prompt,
            system_prompt=system_prompt,
            expect_json=CLASSIFY_SCHEMA,
            task_label=f"Classify {name_part} retry ({len(retry_pages)}p, {len(retry_text):,} chars)",
            on_progress=on_progress,
            think=False,
            max_tokens=CLASSIFY_MAX_TOKENS,
        )

        if retry_result.get("document_type") not in DOCUMENT_TYPES:
            retry_result["document_type"] = "OTHER"

        retry_type = retry_result.get("document_type", "OTHER")
        logger.info(
            f"[{name_part}] Retry classification: {retry_type} "
            f"(was {llm_type}, keywords={kw_matches})"
        )
        # Use retry result — the LLM with more context is the final arbiter
        result = retry_result
        result["keyword_classification"] = kw_matches
        result["classification_retried"] = True
        result["original_classification"] = llm_type

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
