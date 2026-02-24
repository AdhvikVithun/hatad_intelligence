"""PDF report generation via Playwright (Chromium).

Renders the Jinja2 HTML template → converts to PDF with Playwright's
headless Chromium.  Produces a professional multi-page flowing
due-diligence report matching the HataD Plot Clearance Report design.
Pages auto-paginate based on content volume.
"""

import os
import tempfile
import logging
from pathlib import Path
from datetime import datetime

from jinja2 import ChainableUndefined, Environment, FileSystemLoader
from markupsafe import Markup
import mistune

from app.config import TEMPLATES_DIR, REPORTS_DIR, RISK_BANDS

import re as _re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text sanitisation — strip ALL pipeline / LLM internals for exec report
# ---------------------------------------------------------------------------

# Pass 1 — Self-Reflection amendment blocks (always appended at end)
_RE_SELF_REFLECT = _re.compile(
    r'\n*\[Self-Reflection Amendment\].*', _re.DOTALL,
)

# Pass 2 — LLM padding stubs
_RE_LLM_PADDING = _re.compile(
    r'LLM did not evaluate this check\s*[-—–]?\s*'
    r'insufficient document data or check was skipped\.?',
    _re.IGNORECASE,
)

# Pass 3 — Internal filenames
# Bracketed:  [filename_HEX.pdf]  or  [filename_HEX.pdf p.2]
_RE_FILE_BRACKET = _re.compile(
    r'\[[^\]]*[A-Fa-f0-9_-]{8,}\.pdf[^\]]*\]',
)
# "In FILENAME.pdf, " prefix — require filename chars (no spaces) right after "In "
_RE_FILE_IN_PREFIX = _re.compile(
    r'\bIn\s+\S*[A-Fa-f0-9_-]{8,}\.pdf\s*,\s*',
)
# Bare filename (NO spaces/parens — just word chars, underscores, hyphens)
_RE_FILE_BARE = _re.compile(
    r'\b[A-Za-z0-9_-]*[A-Fa-f0-9_-]{8,}\.pdf\b',
)

# Pass 4 — Identity evidence lines (✓ / → lines and header)
_RE_IDENTITY_LINES = _re.compile(
    r'(?:^|\n)\s*[✓→]\s*[^\n]*', _re.MULTILINE,
)
_RE_IDENTITY_HEADER = _re.compile(
    r'Identity Match:\s*[^\n]*resolved to the same person[^\n]*',
)
_RE_IDENTITY_XREF = _re.compile(
    r'(?:Buyer|Seller|EC Claimant)\s*↔\s*Patta:[^\n]*',
)

# Pass 5 — DET check references
_RE_DET_FULL = _re.compile(
    r'Deterministic check\s+DET_\w+\s*\([A-Z_/]+\)\s*',
    _re.IGNORECASE,
)
_RE_DET_BARE = _re.compile(r'\bDET_\w+\b')

# Pass 6 — Internal terminology
_RE_NORMALIZED = _re.compile(r'\(normalized:\s*[^)]*\)')
_RE_GARBLED = _re.compile(r'\[garbled OCR\]', _re.IGNORECASE)
_RE_TOOL_NAMES = _re.compile(
    r'\b(?:query_knowledge_base|search_documents)\b[^.]*',
)
_RE_CONFIDENCE_BAND = _re.compile(
    r'\(Confidence:\s*\d+%\s*[-—–]\s*\w+\)',
)
_RE_ISSUE_TAGS = _re.compile(
    r'\[(?:STATUS_EVIDENCE_CONTRADICTION|CROSS_GROUP_CONTRADICTION'
    r'|SEVERITY_UNDERESTIMATION|EVIDENCE_QUALITY)[^\]]*\]',
)
_RE_CONTRADICTION_SENT = _re.compile(
    r'This creates a (?:cross-group|status[- ]evidence) contradiction[^.]*\.?\s*',
    _re.IGNORECASE,
)
_RE_LLM_CHECK_REF = _re.compile(
    r'The LLM check\s+[^.]*\.\s*', _re.IGNORECASE,
)

# Pass 7 — Group references: "(Group 1)" but NOT "survey number groups"
_RE_GROUP_PAREN = _re.compile(r'\(Group\s+\d+\)', _re.IGNORECASE)

# Cleanup patterns
_RE_MULTI_SPACE = _re.compile(r'  +')
_RE_ORPHAN_PARENS = _re.compile(r'\(\s*[,and ]*\s*\)')
_RE_ORPHAN_VS = _re.compile(r'\bvs\s*\(\s*\)')
_RE_MULTI_NEWLINE = _re.compile(r'\n{3,}')
_RE_LEADING_CONNECTORS = _re.compile(
    r'(?:^|\n)\s*(?:and|vs)\s*(?=[.\n])', _re.IGNORECASE,
)
_RE_SPACE_BEFORE_PUNCT = _re.compile(r'\s+([.,;:])')


def _strip_paren_pdf_refs(text: str) -> str:
    """Remove any (...) group whose content contains '.pdf'.

    Uses a depth-tracking state machine so nested parens such as
    ``(download (4)_766e8f4386a3.pdf)`` are handled correctly.
    """
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '(':
            # Walk forward tracking depth to find matching ')'
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            group = text[i:j]  # includes outer ( and )
            if '.pdf' in group.lower():
                # Trim trailing whitespace that preceded the opening paren
                while result and result[-1] == ' ':
                    result.pop()
                i = j
                continue
            # Not a PDF ref — keep the paren group as-is
            result.append(text[i])
            i += 1
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _sanitize_explanation(text: str) -> str:
    """Strip ALL pipeline / LLM internals from check explanation text.

    Operates in sequential passes so earlier removals (like the entire
    Self-Reflection block) prevent later patterns from partially matching
    on already-removed content.
    """
    if not text:
        return text

    out = text

    # 1 — Strip self-reflection amendment blocks (everything after tag)
    out = _RE_SELF_REFLECT.sub('', out)

    # 2 — Humanise LLM padding stubs
    out = _RE_LLM_PADDING.sub(
        'This check was not evaluated due to insufficient document data.', out,
    )

    # 3 — Strip internal filenames
    #     a) Bracketed  [filename.pdf]
    out = _RE_FILE_BRACKET.sub('', out)
    #     b) Parenthesised (filename.pdf) — state machine for nested parens
    out = _strip_paren_pdf_refs(out)
    #     c) "In filename.pdf, " prefix
    out = _RE_FILE_IN_PREFIX.sub('', out)
    #     d) Bare filename (no spaces)
    out = _RE_FILE_BARE.sub('', out)

    # 4 — Strip identity evidence blocks
    out = _RE_IDENTITY_LINES.sub('', out)
    out = _RE_IDENTITY_HEADER.sub('', out)
    out = _RE_IDENTITY_XREF.sub('', out)

    # 5 — Strip DET references
    out = _RE_DET_FULL.sub('', out)
    out = _RE_DET_BARE.sub('', out)

    # 6 — Strip internal terminology
    out = _RE_NORMALIZED.sub('', out)
    out = _RE_GARBLED.sub('', out)
    out = _RE_TOOL_NAMES.sub('', out)
    out = _RE_CONFIDENCE_BAND.sub('', out)
    out = _RE_ISSUE_TAGS.sub('', out)
    out = _RE_CONTRADICTION_SENT.sub('', out)
    out = _RE_LLM_CHECK_REF.sub('', out)

    # 7 — Strip group references like (Group 1)
    out = _RE_GROUP_PAREN.sub('', out)

    # 8 — Humanise OCR language
    out = out.replace('OCR garble or extraction error',
                      'data entry or scanning error')

    # 9 — Cleanup
    out = _RE_ORPHAN_PARENS.sub('', out)
    out = _RE_ORPHAN_VS.sub('', out)
    out = _RE_LEADING_CONNECTORS.sub('', out)
    out = _RE_SPACE_BEFORE_PUNCT.sub(r'\1', out)
    out = _RE_MULTI_SPACE.sub(' ', out)
    out = _RE_MULTI_NEWLINE.sub('\n\n', out)
    out = out.strip()

    return out

# Markdown → HTML converter for narrative reports in PDF
_md = mistune.create_markdown(plugins=['table', 'strikethrough'])

def _markdown_filter(text: str) -> Markup:
    """Jinja2 filter: convert Markdown text to safe HTML."""
    if not text:
        return Markup('')
    return Markup(_md(text))

# Jinja2 environment for HTML template rendering
# ChainableUndefined allows safe nested attribute access (a.b.c) —
# if any level is missing it silently returns Undefined instead of raising.
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
    undefined=ChainableUndefined,
)
_env.filters['markdown'] = _markdown_filter


# -----------------------------------------------
# Property data extractor (merges across docs)
# -----------------------------------------------

def _extract_property_summary(session_data: dict) -> dict:
    """Scan all extracted_data and merge property fields across doc types.

    Survey numbers from Patta/Chitta are filtered to only include those
    that match surveys referenced in the EC or sale deed (the "transaction
    survey numbers"), preventing unrelated surveys from the same Patta
    being included in the property snapshot.
    """
    extracted = session_data.get("extracted_data", {})
    prop = {
        "survey_numbers": [],
        "village": "Not available",
        "taluk": "Not available",
        "district": "Not available",
        "extent": "Not available",
        "owners": [],
        "patta_number": "Not available",
        "ec_period": "Not available",
        "sro": "Not available",
        "boundaries": {},
        "land_classification": "Not available",
        "consideration_amount": "Not available",
        "property_type": "Not available",
    }

    # First pass: collect "anchor" survey numbers from EC and sale deed
    # (these are the surveys involved in the actual transaction)
    anchor_surveys: list[str] = []
    # Also collect Patta survey entries for second pass filtering
    patta_survey_entries: list[dict] = []
    patta_total_extents: list[str] = []

    for _filename, entry in extracted.items():
        doc_type = entry.get("document_type", "OTHER")
        data = entry.get("data")
        if not data or not isinstance(data, dict):
            continue

        if doc_type == "EC":
            if data.get("village") and prop["village"] == "Not available":
                prop["village"] = data["village"]
            if data.get("taluk") and prop["taluk"] == "Not available":
                prop["taluk"] = data["taluk"]
            period_from = data.get("period_from", "")
            period_to = data.get("period_to", "")
            if period_from or period_to:
                prop["ec_period"] = f"{period_from} to {period_to}"
            txns = data.get("transactions", [])
            if txns:
                sro = txns[0].get("sro", "")
                if sro and prop["sro"] == "Not available":
                    prop["sro"] = sro
                for txn in txns:
                    sn = txn.get("survey_number", "")
                    if sn and sn not in prop["survey_numbers"]:
                        prop["survey_numbers"].append(sn)
                        if sn not in anchor_surveys:
                            anchor_surveys.append(sn)

        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
            if data.get("village") and prop["village"] == "Not available":
                prop["village"] = data["village"]
            if data.get("taluk") and prop["taluk"] == "Not available":
                prop["taluk"] = data["taluk"]
            if data.get("district") and prop["district"] == "Not available":
                prop["district"] = data["district"]
            if data.get("patta_number") and prop["patta_number"] == "Not available":
                prop["patta_number"] = data["patta_number"]
            # Chitta uses chitta_number instead of patta_number
            if data.get("chitta_number") and prop["patta_number"] == "Not available":
                prop["patta_number"] = data["chitta_number"]
            if data.get("total_extent"):
                patta_total_extents.append(data["total_extent"])
            if data.get("land_classification") and prop["land_classification"] == "Not available":
                prop["land_classification"] = data["land_classification"]
            # Handle owner_names (list) or owner_name (string) for new schemas
            for owner in data.get("owner_names", []):
                name = owner.get("name", "")
                if name and name not in [o.get("name") for o in prop["owners"]]:
                    prop["owners"].append(owner)
            if not data.get("owner_names") and data.get("owner_name"):
                name = data["owner_name"]
                if name not in [o.get("name") for o in prop["owners"]]:
                    prop["owners"].append({"name": name, "father_name": data.get("father_name", "")})
            # Collect Patta surveys for deferred processing
            for sn_entry in data.get("survey_numbers", []):
                sn = sn_entry.get("survey_no", "")
                if sn:
                    patta_survey_entries.append(sn_entry)

        elif doc_type in ("FMB", "ADANGAL"):
            if data.get("village") and prop["village"] == "Not available":
                prop["village"] = data["village"]
            if data.get("taluk") and prop["taluk"] == "Not available":
                prop["taluk"] = data["taluk"]
            if data.get("district") and prop["district"] == "Not available":
                prop["district"] = data["district"]
            if data.get("land_classification") and prop["land_classification"] == "Not available":
                prop["land_classification"] = data["land_classification"]
            # FMB/Adangal have single survey_number
            sn = data.get("survey_number", "")
            if sn and sn not in prop["survey_numbers"]:
                prop["survey_numbers"].append(sn)
                if sn not in anchor_surveys:
                    anchor_surveys.append(sn)
            if data.get("extent") and prop["extent"] == "Not available":
                prop["extent"] = data["extent"]
            if data.get("area_acres") and prop["extent"] == "Not available":
                prop["extent"] = data["area_acres"]
            if data.get("boundaries") and not prop["boundaries"]:
                prop["boundaries"] = data["boundaries"]
            # Adangal owner_name
            if data.get("owner_name"):
                name = data["owner_name"]
                if name not in [o.get("name") for o in prop["owners"]]:
                    prop["owners"].append({"name": name, "father_name": data.get("father_name", "")})

        elif doc_type == "SALE_DEED":
            p = data.get("property", {})
            if p.get("village") and prop["village"] == "Not available":
                prop["village"] = p["village"]
            if p.get("taluk") and prop["taluk"] == "Not available":
                prop["taluk"] = p["taluk"]
            if p.get("district") and prop["district"] == "Not available":
                prop["district"] = p["district"]
            if p.get("survey_number"):
                sn = p["survey_number"]
                if sn not in prop["survey_numbers"]:
                    prop["survey_numbers"].append(sn)
                if sn not in anchor_surveys:
                    anchor_surveys.append(sn)
            if p.get("extent") and prop["extent"] == "Not available":
                prop["extent"] = p["extent"]
            if p.get("property_type") and prop["property_type"] == "Not available":
                prop["property_type"] = p["property_type"]
            if p.get("boundaries"):
                prop["boundaries"] = p["boundaries"]
            if data.get("sro") and prop["sro"] == "Not available":
                prop["sro"] = data["sro"]
            fin = data.get("financials", {})
            if fin.get("consideration_amount") and prop["consideration_amount"] == "Not available":
                prop["consideration_amount"] = _format_inr(fin["consideration_amount"])

        else:
            pd_data = data.get("property_details", {})
            if isinstance(pd_data, dict):
                for key in ("village", "taluk", "district"):
                    if pd_data.get(key) and prop[key] == "Not available":
                        prop[key] = pd_data[key]

    # ── Post-processing: filter Patta surveys against anchor surveys ──
    # Only include Patta survey numbers that match an EC/sale deed survey.
    # This prevents unrelated surveys (e.g., 543, 544 in the same Patta)
    # from being included in the property snapshot for a transaction
    # that only involves survey 317.
    if anchor_surveys and patta_survey_entries:
        matched_extent = None
        for sn_entry in patta_survey_entries:
            sn = sn_entry.get("survey_no", "")
            if not sn:
                continue
            # Check if this Patta survey matches any anchor survey
            sn_parts = [s.strip() for s in sn.replace(",", " ").split() if s.strip()]
            matches_anchor = any(
                anchor in sn_parts or sn in anchor
                for anchor in anchor_surveys
            )
            if matches_anchor:
                if sn not in prop["survey_numbers"]:
                    prop["survey_numbers"].append(sn)
                # Use the matched survey's extent if available
                if sn_entry.get("extent") and not matched_extent:
                    matched_extent = sn_entry["extent"]

        # If we found a matching survey extent, prefer it over total_extent
        # (total_extent includes ALL surveys in the Patta, which may be larger)
        if matched_extent and patta_total_extents:
            prop["extent"] = matched_extent
        elif patta_total_extents and prop["extent"] == "Not available":
            prop["extent"] = patta_total_extents[0]
    elif patta_survey_entries:
        # No anchor surveys (no EC/sale deed) — include all Patta surveys
        for sn_entry in patta_survey_entries:
            sn = sn_entry.get("survey_no", "")
            if sn and sn not in prop["survey_numbers"]:
                prop["survey_numbers"].append(sn)
        if patta_total_extents and prop["extent"] == "Not available":
            prop["extent"] = patta_total_extents[0]
    elif patta_total_extents and prop["extent"] == "Not available":
        prop["extent"] = patta_total_extents[0]

    return prop


# -----------------------------------------------
# Sale Deed details extractor (for dedicated report section)
# -----------------------------------------------

def _format_inr(amount) -> str:
    """Format a number as ₹ with Indian comma grouping."""
    if not amount:
        return ""
    try:
        n = int(float(str(amount).replace(",", "").replace("₹", "").strip()))
    except (ValueError, TypeError):
        return str(amount)
    # Indian grouping: last 3 digits, then groups of 2
    s = str(n)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while rest:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    return f"₹{','.join(groups)},{last3}"


def _extract_sale_deed_details(session_data: dict) -> dict | None:
    """Extract structured sale deed details for the report section.

    Returns a dict with keys: registration, sellers, buyers, financials,
    property_description, boundaries, ownership_history, conditions,
    witnesses, payment_mode, possession_date, encumbrance_declaration.
    Returns None if no SALE_DEED document found.
    """
    extracted = session_data.get("extracted_data", {})

    # Find the first (or only) sale deed
    sd_data = None
    for _fn, entry in extracted.items():
        if entry.get("document_type") == "SALE_DEED":
            sd_data = entry.get("data")
            break

    if not sd_data or not isinstance(sd_data, dict):
        return None

    result: dict = {}

    # ── Registration Info ──
    reg: dict = {}
    reg["document_number"] = sd_data.get("document_number", "")
    reg["registration_date"] = sd_data.get("registration_date", "")
    reg["execution_date"] = sd_data.get("execution_date", "")
    reg["sro"] = sd_data.get("sro", "")
    result["registration"] = {k: v for k, v in reg.items() if v}

    # ── Sellers & Buyers ──
    result["sellers"] = sd_data.get("seller", []) or []
    result["buyers"] = sd_data.get("buyer", []) or []

    # ── Financials ──
    fin_raw = sd_data.get("financials", {}) or {}
    fin: dict = {}
    for key in ("consideration_amount", "guideline_value", "stamp_duty",
                "registration_fee", "market_value"):
        val = fin_raw.get(key)
        if val:
            fin[key] = _format_inr(val)
    result["financials"] = fin

    # ── Payment & Possession ──
    result["payment_mode"] = sd_data.get("payment_mode", "")
    result["possession_date"] = sd_data.get("possession_date", "")
    result["encumbrance_declaration"] = sd_data.get("encumbrance_declaration", "")

    # ── Property Description ──
    result["property_description"] = sd_data.get("property_description", "")

    # ── Boundaries ──
    prop = sd_data.get("property", {}) or {}
    bounds = prop.get("boundaries", {}) or {}
    if bounds:
        result["boundaries"] = bounds

    # ── Ownership History ──
    oh = sd_data.get("ownership_history", []) or []
    result["ownership_history"] = oh

    # ── Conditions & Witnesses ──
    result["conditions"] = sd_data.get("conditions", []) or []
    result["witnesses"] = sd_data.get("witnesses", []) or []

    return result


# -----------------------------------------------
# Playwright HTML → PDF (sync, run in a thread from async code)
# -----------------------------------------------

_MAX_PLAYWRIGHT_RETRIES = 3


def _playwright_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """Convert an HTML file to PDF using Playwright Chromium.

    This is a **sync** function that must NOT be called from inside an
    asyncio event-loop directly.  The calling code should run it in a
    :class:`concurrent.futures.ProcessPoolExecutor` so Playwright gets
    its own clean process.

    Includes retry logic to handle transient ``TargetClosedError`` on
    Windows where the Chromium subprocess occasionally fails on first
    launch.
    """
    from playwright.sync_api import sync_playwright

    last_err: Exception | None = None
    for attempt in range(1, _MAX_PLAYWRIGHT_RETRIES + 1):
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-gpu",
                    ],
                )
                try:
                    page = browser.new_page()
                    page.goto(html_path.as_uri(), wait_until="networkidle")
                    page.pdf(
                        path=str(pdf_path),
                        format="A4",
                        print_background=True,
                        margin={
                            "top": "0",
                            "right": "0",
                            "bottom": "0",
                            "left": "0",
                        },
                    )
                finally:
                    browser.close()
            return  # success
        except Exception as exc:
            last_err = exc
            logger.warning(
                "Playwright PDF attempt %d/%d failed: %s",
                attempt,
                _MAX_PLAYWRIGHT_RETRIES,
                exc,
            )
    raise RuntimeError(
        f"Playwright PDF generation failed after {_MAX_PLAYWRIGHT_RETRIES} attempts"
    ) from last_err


# -----------------------------------------------
# Public API
# -----------------------------------------------

def generate_pdf_report(session_data: dict) -> Path:
    """Generate a PDF report from session data.

    Args:
        session_data: Complete session dict from AnalysisSession.to_dict()

    Returns:
        Path to generated PDF file
    """
    # Also generate HTML version for the HTML endpoint
    try:
        template = _env.get_template("report.html")
        risk_score = session_data.get("risk_score", 50)
        risk_band = session_data.get("risk_band", "MEDIUM")
        band_info = RISK_BANDS.get(risk_band, RISK_BANDS["MEDIUM"])
        verification = session_data.get("verification_result", {})
        checks = verification.get("checks", [])

        # Sanitize check explanations for executive report — strip
        # internal filenames, hashes, and pipeline references.
        for ck in checks:
            if ck.get("explanation"):
                ck["explanation"] = _sanitize_explanation(ck["explanation"])
            if ck.get("recommendation"):
                ck["recommendation"] = _sanitize_explanation(ck["recommendation"])

        # Property summary for the property identification block
        prop = _extract_property_summary(session_data)

        # Sale deed details for the dedicated sale deed section
        sale_deed = _extract_sale_deed_details(session_data)

        context = {
            "session_id": session_data["session_id"],
            "generated_at": datetime.now().strftime("%d %B %Y, %I:%M %p"),
            "risk_score": risk_score,
            "risk_band": risk_band,
            "risk_color": band_info["color"],
            "risk_label": band_info["label"],
            "executive_summary": verification.get("executive_summary", ""),
            "documents": session_data.get("documents", []),
            "prop": prop,
            "all_checks": checks,
            "group_bars": [],  # Kept for backward compat; no longer rendered
            "total_checks": len(checks),
            "total_pass": sum(1 for c in checks if c.get("status") == "PASS"),
            "total_fail": sum(1 for c in checks if c.get("status") == "FAIL"),
            "total_warn": sum(1 for c in checks if c.get("status") == "WARNING"),
            "total_na": sum(1 for c in checks if c.get("status") == "NOT_APPLICABLE"),
            "chain_of_title": verification.get("chain_of_title", []),
            "red_flags": verification.get("red_flags", []),
            "recommendations": verification.get("recommendations", []),
            "missing_documents": verification.get("missing_documents", []),
            "narrative_report": session_data.get("narrative_report", ""),
            "identity_clusters": session_data.get("identity_clusters", []) or [],
            "total_raw_names": sum(
                len(c.get("variants", []))
                for c in (session_data.get("identity_clusters", []) or [])
            ),
            "active_encumbrances": verification.get("active_encumbrances", []) or [],
            "sale_deed": sale_deed,
        }
        html_content = template.render(**context)
        html_path = REPORTS_DIR / f"{session_data['session_id']}_report.html"
        fd, tmp_path = tempfile.mkstemp(
            dir=str(REPORTS_DIR), suffix=".tmp", prefix="report_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(html_content)
            os.replace(tmp_path, str(html_path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.warning(f"HTML report generation failed (non-fatal): {e}")

    # Convert the rendered HTML to PDF via Playwright (Chromium)
    session_id = session_data["session_id"]
    html_path = REPORTS_DIR / f"{session_id}_report.html"
    pdf_path = REPORTS_DIR / f"{session_id}_report.pdf"

    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found at {html_path}")

    _playwright_html_to_pdf(html_path, pdf_path)
    logger.info(f"PDF report generated via Playwright: {pdf_path}")
    return pdf_path
