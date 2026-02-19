"""PDF report generation using fpdf2 with Unicode font support.

Produces a professional 5-6 page executive due-diligence report with:
  Page 1: Cover - risk score, GO/NO-GO, property snapshot
  Page 2: Executive summary, property details, document inventory
  Page 3: Verification results summary table
  Page 4: Chain of title, red flags, encumbrances
  Page 5: Critical findings detail, recommendations, missing docs
  Page 6: Risk breakdown, disclaimer
"""

import json
import os
import re
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Any

from jinja2 import Environment, FileSystemLoader
from fpdf import FPDF

from app.config import TEMPLATES_DIR, REPORTS_DIR, RISK_BANDS

logger = logging.getLogger(__name__)

# Jinja2 environment (still used for HTML report variant)
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=True,
)

# Font paths
_FONTS_DIR = Path(__file__).parent / "fonts"
_HAS_FONTS = (_FONTS_DIR / "NotoSans-Regular.ttf").exists()

# Tamil Unicode range for font switching
_TAMIL_RE = re.compile(r'[\u0B80-\u0BFF]+')
_INDIC_RE = re.compile(r'[\u0900-\u0DFF\u0E00-\u0FFF]+')


# -----------------------------------------------
# Text helpers
# -----------------------------------------------

def _sanitize(text: Any, strip_newlines: bool = False) -> str:
    """Clean text for PDF rendering. Preserves Tamil if Unicode fonts available."""
    if not text:
        return ""
    if not isinstance(text, str):
        if isinstance(text, list):
            text = "; ".join(str(item) for item in text)
        else:
            text = str(text)
    if strip_newlines:
        text = text.replace("\n", " ").replace("\r", " ")
    # Common typographic replacements
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")
    text = text.replace("\u2011", "-")
    text = text.replace("\u2022", "*").replace("\u2026", "...")
    text = text.replace("\u20b9", "Rs.").replace("\u00a0", " ")
    text = text.replace("\u202f", " ").replace("\u2010", "-")
    text = text.replace("\u2212", "-")
    text = text.replace("\u2264", "<=").replace("\u2265", ">=")
    text = text.replace("\u2714", "[Y]").replace("\u2716", "[X]")
    text = text.replace("\u2717", "[X]").replace("\u2713", "[Y]")
    text = text.replace("\u2192", "->").replace("\u2190", "<-")
    text = text.replace("\u2550", "=").replace("\u2551", "|")
    # Strip markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    if not _HAS_FONTS:
        text = _INDIC_RE.sub('', text)
        return text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def _truncate(text: str, max_len: int = 120) -> str:
    """Truncate text to max_len, adding ... if truncated."""
    text = _sanitize(text, strip_newlines=True)
    if len(text) <= max_len:
        return text
    return text[:max_len - 3].rstrip() + "..."


def _has_tamil(text: str) -> bool:
    """Check if text contains Tamil characters."""
    return bool(_TAMIL_RE.search(text)) if text else False


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#rrggbb' or color name to (R, G, B) tuple."""
    hex_color = hex_color.strip().lstrip("#")
    color_map = {
        "red": "ff0000", "green": "00cc44", "orange": "ff8800",
        "yellow": "ccaa00", "blue": "3388cc",
    }
    hex_color = color_map.get(hex_color.lower(), hex_color)
    if len(hex_color) == 6:
        return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (100, 100, 100)


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

        elif doc_type in ("PATTA", "CHITTA"):
            if data.get("village") and prop["village"] == "Not available":
                prop["village"] = data["village"]
            if data.get("taluk") and prop["taluk"] == "Not available":
                prop["taluk"] = data["taluk"]
            if data.get("district") and prop["district"] == "Not available":
                prop["district"] = data["district"]
            if data.get("patta_number") and prop["patta_number"] == "Not available":
                prop["patta_number"] = data["patta_number"]
            if data.get("total_extent"):
                patta_total_extents.append(data["total_extent"])
            if data.get("land_classification") and prop["land_classification"] == "Not available":
                prop["land_classification"] = data["land_classification"]
            for owner in data.get("owner_names", []):
                name = owner.get("name", "")
                if name and name not in [o.get("name") for o in prop["owners"]]:
                    prop["owners"].append(owner)
            # Collect Patta surveys for deferred processing
            for sn_entry in data.get("survey_numbers", []):
                sn = sn_entry.get("survey_no", "")
                if sn:
                    patta_survey_entries.append(sn_entry)

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
                prop["consideration_amount"] = str(fin["consideration_amount"])

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
# PDF Report class
# -----------------------------------------------

class _HATADReport(FPDF):
    """Professional PDF layout for HATAD due-diligence reports."""

    def __init__(self, session_id: str, risk_score: int, risk_band: str, risk_color: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.session_id = session_id
        self.risk_score = risk_score
        self.risk_band = risk_band
        self.risk_color = risk_color
        self.set_auto_page_break(auto=True, margin=22)

        # Register Unicode fonts
        if _HAS_FONTS:
            self.add_font("NotoSans", "", str(_FONTS_DIR / "NotoSans-Regular.ttf"))
            self.add_font("NotoSans", "B", str(_FONTS_DIR / "NotoSans-Bold.ttf"))
            if (_FONTS_DIR / "NotoSansTamil-Regular.ttf").exists():
                self.add_font("NotoSansTamil", "", str(_FONTS_DIR / "NotoSansTamil-Regular.ttf"))
            if (_FONTS_DIR / "NotoSansTamil-Bold.ttf").exists():
                self.add_font("NotoSansTamil", "B", str(_FONTS_DIR / "NotoSansTamil-Bold.ttf"))
            self._base_font = "NotoSans"
            self._tamil_font = "NotoSansTamil" if (_FONTS_DIR / "NotoSansTamil-Regular.ttf").exists() else "NotoSans"
            # Use Tamil font as fallback for Tamil glyphs not in NotoSans
            if self._tamil_font == "NotoSansTamil":
                self.set_fallback_fonts(["NotoSansTamil"])
        else:
            self._base_font = "Helvetica"
            self._tamil_font = "Helvetica"

    # -- Header / Footer --------------------------
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font(self._base_font, "B", 7)
        self.set_text_color(100, 100, 120)
        self.cell(95, 5, "HATAD LAND INTELLIGENCE PLATFORM", align="L")
        self.cell(95, 5, f"Report ID: {self.session_id[:8]}", align="R",
                  new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 210)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._base_font, "", 7)
        self.set_text_color(140, 140, 150)
        self.cell(0, 6, f"CONFIDENTIAL  |  Page {self.page_no()}/{{nb}}  |  "
                  f"Generated {datetime.now().strftime('%d-%b-%Y')}", align="C")

    # -- Smart text output (auto font-switch for Tamil) --
    def smart_cell(self, w, h, text, **kwargs):
        """Write cell text using base font."""
        clean = _sanitize(text, strip_newlines=True)
        self.cell(w, h, clean, **kwargs)

    def smart_multi_cell(self, w, h, text, **kwargs):
        """Write multi_cell text using base font."""
        clean = _sanitize(text)
        self.multi_cell(w, h, clean, **kwargs)

    # -- Section headers --------------------------
    def section_header(self, title: str):
        """Dark-background full-width section header."""
        if self.get_y() > self.h - 30:
            self.add_page()
        self.set_fill_color(26, 26, 46)
        self.set_text_color(255, 255, 255)
        self.set_font(self._base_font, "B", 10)
        self.cell(0, 8, f"  {title.upper()}", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(30, 30, 50)

    def sub_header(self, title: str):
        """Indented bold sub-section header."""
        self.set_font(self._base_font, "B", 9)
        self.set_text_color(26, 26, 46)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 50)
        self.ln(1)

    # -- Table helpers ----------------------------
    def table_header(self, columns: list, widths: list):
        """Draw a dark table header row."""
        self.set_fill_color(40, 40, 60)
        self.set_text_color(255, 255, 255)
        self.set_font(self._base_font, "B", 7.5)
        for col, w in zip(columns, widths):
            self.cell(w, 6, f" {col}", fill=True)
        self.ln()
        self.set_text_color(30, 30, 50)

    def table_row(self, cells: list, widths: list, alt: bool = False,
                  colors: list = None):
        """Draw a data row with optional alternating fill and per-cell colors."""
        if alt:
            self.set_fill_color(245, 245, 250)
        else:
            self.set_fill_color(255, 255, 255)

        self.set_font(self._base_font, "", 7.5)
        y_start = self.get_y()
        max_h = 5

        # Calculate row height by finding tallest cell
        for cell_text, w in zip(cells, widths):
            clean = _sanitize(str(cell_text), strip_newlines=True)
            char_per_line = max(1, int(w / 1.8))
            lines = max(1, (len(clean) + char_per_line - 1) // char_per_line)
            cell_h = lines * 4
            if cell_h > max_h:
                max_h = cell_h
        max_h = min(max_h, 20)

        # Page break check
        if self.get_y() + max_h > self.h - self.b_margin:
            self.add_page()
            y_start = self.get_y()

        x_start = self.get_x()
        for i, (cell_text, w) in enumerate(zip(cells, widths)):
            clean = _sanitize(str(cell_text), strip_newlines=True)
            x = x_start + sum(widths[:i])
            self.set_xy(x, y_start)

            if colors and i < len(colors) and colors[i]:
                self.set_text_color(*colors[i])

            self.rect(x, y_start, w, max_h, 'F')

            # For table cells, always use base font (NotoSans handles most scripts)
            # Tamil-only font lacks Latin glyphs, so avoid it for mixed content
            self.set_font(self._base_font, "", 7.5)

            self.set_xy(x + 1, y_start + 0.5)
            self.multi_cell(w - 2, 4, clean[:80] if len(clean) > 80 else clean)

            self.set_text_color(30, 30, 50)
            if _HAS_FONTS:
                self.set_font(self._base_font, "", 7.5)

        self.set_xy(x_start, y_start + max_h)
        self.set_draw_color(220, 220, 225)
        self.line(x_start, y_start + max_h, x_start + sum(widths), y_start + max_h)

    # -- Status/severity badges -------------------
    def status_badge(self, status: str, w: int = 18):
        """Colored status badge."""
        badge_colors = {
            "PASS": (0, 150, 70), "FAIL": (210, 40, 40),
            "WARNING": (210, 140, 20), "NOT_APPLICABLE": (130, 130, 145),
            "INFO": (70, 120, 190),
        }
        r, g, b = badge_colors.get(status, (100, 100, 100))
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font(self._base_font, "B", 6.5)
        label = status.replace("NOT_APPLICABLE", "N/A")
        self.cell(w, 5, label, fill=True, align="C")
        self.set_text_color(30, 30, 50)

    def severity_tag(self, severity: str, w: int = 16):
        """Colored severity tag."""
        tag_colors = {
            "CRITICAL": (210, 40, 40), "HIGH": (220, 110, 25),
            "MEDIUM": (190, 170, 25), "WARNING": (190, 170, 25),
            "LOW": (50, 150, 50), "INFO": (70, 120, 190),
        }
        r, g, b = tag_colors.get(severity.upper(), (100, 100, 100))
        self.set_text_color(r, g, b)
        self.set_font(self._base_font, "B", 6.5)
        self.cell(w, 5, severity)
        self.set_text_color(30, 30, 50)

    # -- Stat box ---------------------------------
    def stat_box(self, x: float, y: float, w: float, h: float,
                 label: str, value: str, color: tuple):
        """Draw a stat box with label and big number."""
        self.set_fill_color(*color)
        self.rect(x, y, w, h, 'F')
        self.set_xy(x, y + 2)
        self.set_font(self._base_font, "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(w, 8, str(value), align="C")
        self.set_xy(x, y + 11)
        self.set_font(self._base_font, "", 7)
        self.cell(w, 5, label, align="C")
        self.set_text_color(30, 30, 50)

    # -- Key-value pair ---------------------------
    def kv_line(self, key: str, value: str, key_w: int = 45):
        """Compact key:value line."""
        self.set_font(self._base_font, "B", 8)
        self.set_text_color(70, 70, 90)
        self.cell(key_w, 5, key + ":")
        self.set_font(self._base_font, "", 8)
        self.set_text_color(30, 30, 50)
        val_str = _sanitize(str(value), strip_newlines=True)
        self.multi_cell(0, 5, val_str, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    # -- Body text --------------------------------
    def body_text(self, text: str, size: float = 8.5):
        """Standard body text paragraph."""
        self.set_font(self._base_font, "", size)
        self.set_text_color(40, 40, 60)
        self.smart_multi_cell(0, 4.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bullet(self, text: str, size: float = 8):
        """Bulleted item."""
        self.set_font(self._base_font, "", size)
        self.set_text_color(40, 40, 60)
        self.cell(5, 4.5, "-")
        self.smart_multi_cell(0, 4.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)


# -----------------------------------------------
# PDF Builder
# -----------------------------------------------

def _build_pdf(session_data: dict) -> Path:
    """Build a professional 5-6 page executive PDF report."""
    # Suppress fpdf2 glyph fallback warnings (Tamil/Latin font substitution)
    logging.getLogger("fpdf.output").setLevel(logging.ERROR)

    session_id = session_data["session_id"]
    risk_score = session_data.get("risk_score", 50)
    risk_band = session_data.get("risk_band", "MEDIUM")
    band_info = RISK_BANDS.get(risk_band, RISK_BANDS["MEDIUM"])
    risk_color = band_info["color"]

    verification = session_data.get("verification_result", {})
    checks = verification.get("checks", [])
    docs = session_data.get("documents", [])
    prop = _extract_property_summary(session_data)

    pdf = _HATADReport(session_id, risk_score, risk_band, risk_color)
    pdf.alias_nb_pages()

    # ===================================================
    # PAGE 1: COVER
    # ===================================================
    pdf.add_page()
    pdf.ln(30)

    # Title block
    pdf.set_font(pdf._base_font, "B", 32)
    pdf.set_text_color(0, 200, 100)
    pdf.cell(0, 14, "HATAD", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(pdf._base_font, "", 11)
    pdf.set_text_color(120, 120, 140)
    pdf.cell(0, 7, "LAND INTELLIGENCE PLATFORM", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 210)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(6)
    pdf.set_font(pdf._base_font, "B", 16)
    pdf.set_text_color(30, 30, 50)
    pdf.cell(0, 10, "Land Due Diligence Report", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)

    # Risk score display
    rc = _hex_to_rgb(risk_color)
    pdf.set_font(pdf._base_font, "B", 44)
    pdf.set_text_color(*rc)
    pdf.cell(0, 20, f"{risk_score}/100", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(pdf._base_font, "B", 14)
    pdf.cell(0, 8, f"Risk: {risk_band}", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # GO / NO-GO recommendation box
    if risk_band == "LOW":
        go_color = (0, 160, 70)
        go_text = "PROCEED"
        go_sub = "Low risk identified - standard precautions apply"
    elif risk_band == "MEDIUM":
        go_color = (220, 150, 20)
        go_text = "PROCEED WITH CAUTION"
        go_sub = "Medium risk - additional verification recommended"
    else:
        go_color = (210, 40, 40)
        go_text = "DO NOT PROCEED"
        go_sub = "High/Critical risk - resolve issues before proceeding"

    box_x = 45
    box_w = 120
    box_y = pdf.get_y()
    pdf.set_fill_color(*go_color)
    pdf.rect(box_x, box_y, box_w, 22, 'F')
    pdf.set_xy(box_x, box_y + 3)
    pdf.set_font(pdf._base_font, "B", 18)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(box_w, 9, go_text, align="C")
    pdf.set_xy(box_x, box_y + 13)
    pdf.set_font(pdf._base_font, "", 8)
    pdf.cell(box_w, 5, go_sub, align="C")
    pdf.set_text_color(30, 30, 50)
    pdf.set_y(box_y + 28)

    # Property snapshot on cover
    pdf.set_draw_color(200, 200, 210)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())
    pdf.ln(5)
    pdf.set_font(pdf._base_font, "B", 9)
    pdf.set_text_color(70, 70, 90)
    pdf.cell(0, 5, "PROPERTY SNAPSHOT", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Two-column property snapshot
    snap_items = [
        ("Survey No.", ", ".join(prop["survey_numbers"]) if prop["survey_numbers"] else "Not available"),
        ("Village", prop["village"]),
        ("Taluk", prop["taluk"]),
        ("District", prop["district"]),
        ("Extent", prop["extent"]),
        ("SRO", prop["sro"]),
    ]
    x_left = 20
    x_right = 110
    for i, (key, val) in enumerate(snap_items):
        x = x_left if i % 2 == 0 else x_right
        if i % 2 == 0 and i > 0:
            pdf.ln(5)
        pdf.set_x(x)
        pdf.set_font(pdf._base_font, "B", 8)
        pdf.set_text_color(70, 70, 90)
        pdf.cell(25, 5, f"{key}:")
        pdf.set_font(pdf._base_font, "", 8)
        pdf.set_text_color(30, 30, 50)
        val_clean = _sanitize(val, strip_newlines=True)
        pdf.cell(65, 5, val_clean[:40])
    pdf.ln(8)

    # Footer info
    pdf.set_font(pdf._base_font, "", 8)
    pdf.set_text_color(100, 100, 120)
    gen_date = datetime.now().strftime('%d %B %Y, %I:%M %p')
    pdf.cell(0, 5, f"Session: {session_id}  |  Generated: {gen_date}  |  "
             f"Documents: {len(docs)}", align="C", new_x="LMARGIN", new_y="NEXT")

    # ===================================================
    # PAGE 2: EXECUTIVE SUMMARY + PROPERTY + DOCS
    # ===================================================
    pdf.add_page()

    # Executive Summary
    exec_summary = verification.get("executive_summary", "")
    if exec_summary:
        pdf.section_header("Executive Summary")
        pdf.body_text(exec_summary)
        pdf.ln(2)

    # Property Details
    pdf.section_header("Property Details")
    pdf.kv_line("Survey Number(s)",
                ", ".join(prop["survey_numbers"]) if prop["survey_numbers"] else "Not available")
    pdf.kv_line("Village / Taluk / District",
                f"{prop['village']} / {prop['taluk']} / {prop['district']}")
    pdf.kv_line("Total Extent", prop["extent"])
    pdf.kv_line("Land Classification", prop["land_classification"])
    pdf.kv_line("Property Type", prop["property_type"])
    pdf.kv_line("Patta Number", prop["patta_number"])
    pdf.kv_line("EC Period", prop["ec_period"])
    pdf.kv_line("SRO", prop["sro"])

    # Owners
    if prop["owners"]:
        parts = []
        for o in prop["owners"]:
            s = o.get("name", "?")
            if o.get("father_name"):
                s += f" (s/o {o['father_name']})"
            if o.get("share"):
                s += f" [{o['share']}]"
            parts.append(s)
        pdf.kv_line("Current Owner(s)", "; ".join(parts))

    # Boundaries
    if prop["boundaries"]:
        b = prop["boundaries"]
        bnd_str = (f"N: {b.get('north', '?')} | S: {b.get('south', '?')} | "
                   f"E: {b.get('east', '?')} | W: {b.get('west', '?')}")
        pdf.kv_line("Boundaries", bnd_str)

    pdf.ln(2)

    # Document Inventory
    if docs:
        pdf.section_header("Documents Analyzed")
        pdf.table_header(["Type", "Filename", "Pages", "Quality"],
                         [30, 100, 18, 42])
        for i, doc in enumerate(docs):
            dtype = doc.get("document_type", "OTHER")
            fname = doc.get("filename", doc.get("original_name", "?"))
            if len(fname) > 45:
                fname = fname[:42] + "..."
            pages = str(doc.get("pages", "?"))
            quality = doc.get("extraction_quality", "?")
            q_color = {"HIGH": (0, 150, 70), "MEDIUM": (210, 140, 20),
                       "LOW": (210, 40, 40)}.get(quality, (100, 100, 100))
            pdf.table_row([dtype, fname, pages, quality],
                          [30, 100, 18, 42], alt=i % 2 == 1,
                          colors=[None, None, None, q_color])
        pdf.ln(3)

    # ===================================================
    # PAGE 3: VERIFICATION RESULTS
    # ===================================================
    pdf.add_page()

    total_checks = len(checks)
    total_pass = sum(1 for c in checks if c.get("status") == "PASS")
    total_fail = sum(1 for c in checks if c.get("status") == "FAIL")
    total_warn = sum(1 for c in checks if c.get("status") == "WARNING")

    pdf.section_header("Verification Results")

    # Stats bar
    box_w = 44
    box_h = 18
    y = pdf.get_y()
    pdf.stat_box(12, y, box_w, box_h, "TOTAL CHECKS", str(total_checks), (60, 60, 80))
    pdf.stat_box(12 + box_w + 2, y, box_w, box_h, "PASSED", str(total_pass), (0, 150, 70))
    pdf.stat_box(12 + 2 * (box_w + 2), y, box_w, box_h, "FAILED", str(total_fail), (210, 40, 40))
    pdf.stat_box(12 + 3 * (box_w + 2), y, box_w, box_h, "WARNINGS", str(total_warn), (210, 140, 20))
    pdf.set_y(y + box_h + 4)

    # Sorted checks table
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "WARNING": 3, "LOW": 4, "INFO": 5}
    status_order = {"FAIL": 0, "WARNING": 1, "PASS": 2, "NOT_APPLICABLE": 3, "INFO": 4}
    sorted_checks = sorted(checks, key=lambda c: (
        severity_order.get(c.get("severity", "INFO"), 99),
        status_order.get(c.get("status", "INFO"), 99),
    ))

    # Table header
    pdf.set_fill_color(40, 40, 60)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font(pdf._base_font, "B", 7)
    pdf.cell(18, 5, " Status", fill=True)
    pdf.cell(16, 5, " Severity", fill=True)
    pdf.cell(50, 5, " Check", fill=True)
    pdf.cell(106, 5, " Finding", fill=True)
    pdf.ln()
    pdf.set_text_color(30, 30, 50)

    for i, chk in enumerate(sorted_checks):
        status = chk.get("status", "?")
        severity = chk.get("severity", "INFO")
        rule = chk.get("rule_name", chk.get("rule_code", ""))
        explanation = chk.get("explanation", "")

        if pdf.get_y() > pdf.h - 25:
            pdf.add_page()

        # Alternating row bg
        if i % 2 == 1:
            pdf.set_fill_color(248, 248, 252)
            pdf.rect(10, pdf.get_y(), 190, 5, 'F')

        pdf.status_badge(status, 18)
        pdf.severity_tag(severity, 16)
        pdf.set_font(pdf._base_font, "B", 7)
        pdf.cell(50, 5, _truncate(rule, 28))
        pdf.set_font(pdf._base_font, "", 7)
        pdf.cell(106, 5, _truncate(explanation, 60),
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)

    # Group performance summary
    group_summary = verification.get("group_results_summary", {})
    if group_summary:
        pdf.sub_header("Verification Group Summary")
        pdf.table_header(["Group", "Checks", "Deduction"],
                         [100, 35, 55])
        for gid in sorted(group_summary.keys(), key=lambda x: int(x) if x.isdigit() else 99):
            gs = group_summary[gid]
            name = gs.get("name", f"Group {gid}")
            count = str(gs.get("check_count", 0))
            ded = gs.get("deduction", 0)
            ded_str = f"-{ded} points" if ded > 0 else "No deduction"
            ded_color = (210, 40, 40) if ded > 10 else (210, 140, 20) if ded > 0 else (0, 150, 70)
            pdf.table_row([name, count, ded_str], [100, 35, 55],
                          colors=[None, None, ded_color])

    # ===================================================
    # PAGE 4: CHAIN OF TITLE + RED FLAGS + ENCUMBRANCES
    # ===================================================
    pdf.add_page()

    # Chain of Title
    chain = verification.get("chain_of_title", [])
    if chain:
        pdf.section_header("Chain of Title")
        pdf.table_header(["#", "Date", "From", "To", "Type", "Doc No.", "Valid"],
                         [8, 22, 48, 48, 24, 22, 18])
        for i, link in enumerate(chain):
            if isinstance(link, dict):
                seq = str(link.get("sequence", i + 1))
                date = str(link.get("date", "?"))
                fr = str(link.get("from", "?"))
                to = str(link.get("to", "?"))
                txn = str(link.get("transaction_type", "?"))
                doc_no = str(link.get("document_number", ""))
                valid = link.get("valid", True)
                valid_str = "Yes" if valid else "NO"
                valid_color = (0, 150, 70) if valid else (210, 40, 40)
                pdf.table_row(
                    [seq, date, fr[:25], to[:25], txn[:12], doc_no[:12], valid_str],
                    [8, 22, 48, 48, 24, 22, 18],
                    alt=i % 2 == 1,
                    colors=[None, None, None, None, None, None, valid_color]
                )
        pdf.ln(4)

    # Active Encumbrances
    encumbrances = verification.get("active_encumbrances", [])
    if encumbrances:
        pdf.section_header("Active Encumbrances")
        for enc in encumbrances:
            pdf.set_text_color(210, 40, 40)
            pdf.bullet(_sanitize(str(enc)), size=8)
        pdf.set_text_color(30, 30, 50)
        pdf.ln(3)

    # Red Flags
    red_flags = verification.get("red_flags", [])
    if red_flags:
        pdf.section_header("Red Flags")
        for rf in red_flags:
            text = rf.get("description", str(rf)) if isinstance(rf, dict) else str(rf)
            pdf.set_text_color(210, 40, 40)
            pdf.set_font(pdf._base_font, "B", 8)
            pdf.cell(5, 4.5, "!")
            pdf.set_font(pdf._base_font, "", 8)
            pdf.set_text_color(40, 40, 60)
            pdf.smart_multi_cell(0, 4.5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.ln(2)

    # ===================================================
    # PAGE 5: CRITICAL FINDINGS + RECOMMENDATIONS + MISSING DOCS
    # ===================================================
    if pdf.get_y() > pdf.h - 80:
        pdf.add_page()

    # Critical/High findings detail
    critical_checks = [c for c in sorted_checks
                       if c.get("severity") in ("CRITICAL", "HIGH")
                       and c.get("status") == "FAIL"]
    if critical_checks:
        pdf.section_header("Critical & High-Priority Findings (Detail)")
        sev_colors = {"CRITICAL": (210, 40, 40), "HIGH": (220, 110, 25)}
        for chk in critical_checks[:10]:
            severity = chk.get("severity", "")
            rule = chk.get("rule_name", chk.get("rule_code", ""))
            explanation = chk.get("explanation", "")
            rec = chk.get("recommendation", "")
            evidence = chk.get("evidence", "")

            if pdf.get_y() > pdf.h - 30:
                pdf.add_page()

            # Severity + Rule name
            pdf.set_font(pdf._base_font, "B", 8)
            sr, sg, sb = sev_colors.get(severity, (100, 100, 100))
            pdf.set_text_color(sr, sg, sb)
            pdf.cell(18, 5, f"[{severity}]")
            pdf.set_text_color(30, 30, 50)
            pdf.set_font(pdf._base_font, "B", 8)
            pdf.cell(0, 5, _sanitize(rule), new_x="LMARGIN", new_y="NEXT")

            # Explanation
            pdf.set_font(pdf._base_font, "", 7.5)
            pdf.set_text_color(50, 50, 70)
            pdf.smart_multi_cell(0, 4, explanation[:300],
                                 new_x="LMARGIN", new_y="NEXT")

            # Evidence
            if evidence:
                pdf.set_font(pdf._base_font, "", 6.5)
                pdf.set_text_color(100, 100, 130)
                pdf.smart_multi_cell(0, 3.5,
                                     f"Evidence: {_truncate(evidence, 200)}",
                                     new_x="LMARGIN", new_y="NEXT")

            # Recommendation
            if rec:
                pdf.set_font(pdf._base_font, "", 7.5)
                pdf.set_text_color(70, 120, 190)
                pdf.smart_multi_cell(0, 4,
                                     f"Action: {_sanitize(rec)[:200]}",
                                     new_x="LMARGIN", new_y="NEXT")

            pdf.set_text_color(30, 30, 50)
            pdf.ln(2)
            pdf.set_draw_color(230, 230, 235)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(2)

    # Recommendations
    recs = verification.get("recommendations", [])
    if recs:
        if pdf.get_y() > pdf.h - 40:
            pdf.add_page()
        pdf.section_header("Recommendations")
        for i, rec in enumerate(recs[:12], 1):
            text = rec.get("action", str(rec)) if isinstance(rec, dict) else str(rec)
            priority = rec.get("priority", "") if isinstance(rec, dict) else ""
            pdf.set_font(pdf._base_font, "B", 7.5)
            pdf.set_text_color(70, 70, 90)
            label = f"{i}."
            if priority:
                label = f"{i}. [{priority}]"
            pdf.cell(12, 4.5, label)
            pdf.set_font(pdf._base_font, "", 7.5)
            pdf.set_text_color(40, 40, 60)
            pdf.smart_multi_cell(0, 4.5, text[:200],
                                 new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        pdf.ln(2)

    # Missing Documents
    missing = verification.get("missing_documents", [])
    if missing:
        if pdf.get_y() > pdf.h - 30:
            pdf.add_page()
        pdf.section_header("Missing Documents")
        for md in missing:
            if isinstance(md, dict):
                text = f"{md.get('document_type', '?')} - {md.get('reason', '')}"
            else:
                text = str(md)
            pdf.set_text_color(210, 40, 40)
            pdf.set_font(pdf._base_font, "", 7.5)
            pdf.cell(4, 4.5, "-")
            pdf.set_text_color(40, 40, 60)
            pdf.smart_multi_cell(0, 4.5, text[:200],
                                 new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)
        pdf.ln(2)

    # ===================================================
    # PAGE 6: RISK BREAKDOWN + DISCLAIMER
    # ===================================================
    if pdf.get_y() > pdf.h - 80:
        pdf.add_page()

    # Risk Score Breakdown
    pdf.section_header("Risk Score Breakdown")
    pdf.set_font(pdf._base_font, "", 8)
    pdf.set_text_color(40, 40, 60)
    pdf.body_text(f"Starting score: 100 points. Deductions applied per verification group. "
                  f"Final score: {risk_score}/100 ({risk_band}).")

    if group_summary:
        pdf.table_header(["Verification Category", "Checks Run", "Points Deducted", "Impact"],
                         [65, 30, 35, 60])
        total_ded = 0
        for gid in sorted(group_summary.keys(), key=lambda x: int(x) if x.isdigit() else 99):
            gs = group_summary[gid]
            name = gs.get("name", f"Group {gid}")
            count = str(gs.get("check_count", 0))
            ded = gs.get("deduction", 0)
            total_ded += ded
            if ded >= 20:
                impact = "Severe"
                impact_c = (210, 40, 40)
            elif ded >= 10:
                impact = "Significant"
                impact_c = (220, 110, 25)
            elif ded > 0:
                impact = "Minor"
                impact_c = (210, 140, 20)
            else:
                impact = "None"
                impact_c = (0, 150, 70)
            ded_str = f"-{ded}" if ded > 0 else "0"
            pdf.table_row([name, count, ded_str, impact],
                          [65, 30, 35, 60],
                          colors=[None, None, (210, 40, 40) if ded > 0 else (0, 150, 70), impact_c])

        # Total row
        pdf.set_fill_color(30, 30, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font(pdf._base_font, "B", 8)
        pdf.cell(65, 6, " FINAL SCORE", fill=True)
        pdf.cell(30, 6, "", fill=True)
        pdf.cell(35, 6, f" -{total_ded}", fill=True)
        fin_rc = _hex_to_rgb(risk_color)
        pdf.set_text_color(*fin_rc)
        pdf.cell(60, 6, f" {risk_score}/100 ({risk_band})", fill=True,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(30, 30, 50)
        pdf.ln(4)

    # Risk Band Legend
    pdf.sub_header("Risk Band Definitions")
    bands = [
        ("LOW (80-100)", "Property title appears clear with minor or no issues.", (0, 150, 70)),
        ("MEDIUM (50-79)", "Some concerns identified; additional checks recommended.", (210, 140, 20)),
        ("HIGH (20-49)", "Significant issues found; legal consultation required.", (220, 110, 25)),
        ("CRITICAL (0-19)", "Severe defects; do not proceed without resolution.", (210, 40, 40)),
    ]
    for band_label, desc, color in bands:
        pdf.set_font(pdf._base_font, "B", 7.5)
        pdf.set_text_color(*color)
        pdf.cell(30, 4.5, band_label)
        pdf.set_font(pdf._base_font, "", 7.5)
        pdf.set_text_color(70, 70, 90)
        pdf.cell(0, 4.5, desc, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Disclaimer
    if pdf.get_y() > pdf.h - 50:
        pdf.add_page()

    pdf.section_header("Disclaimer")
    disclaimer = (
        "This report is generated by the HATAD Land Intelligence Platform using automated "
        "document analysis and AI-powered verification. It is intended solely for preliminary "
        "due diligence purposes and does not constitute legal advice or a legal opinion.\n\n"
        "The analysis is based exclusively on the documents provided. The absence of issues "
        "in this report does not guarantee a clear title. Physical site inspection, verification "
        "of original documents at the Sub-Registrar Office, and consultation with a qualified "
        "legal professional are strongly recommended before making any transaction decision.\n\n"
        "HATAD and its operators accept no liability for decisions made based on this report. "
        "All findings should be independently verified by qualified professionals."
    )
    pdf.set_font(pdf._base_font, "", 7.5)
    pdf.set_text_color(80, 80, 100)
    pdf.multi_cell(0, 4, disclaimer, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Final signature line
    pdf.set_draw_color(200, 200, 210)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font(pdf._base_font, "", 7)
    pdf.set_text_color(140, 140, 150)
    pdf.cell(0, 4,
             f"Report ID: {session_id}  |  "
             f"Analysis Date: {datetime.now().strftime('%d %B %Y %I:%M %p')}  |  "
             f"Checks: {total_checks}  |  Score: {risk_score}/100",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # -- Write PDF --------------------------------
    pdf_path = REPORTS_DIR / f"{session_id}_report.pdf"
    pdf.output(str(pdf_path))
    logger.info(f"PDF report generated: {pdf_path} ({pdf.page_no()} pages)")
    return pdf_path


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

        context = {
            "session_id": session_data["session_id"],
            "generated_at": datetime.now().strftime("%d %B %Y, %I:%M %p"),
            "risk_score": risk_score,
            "risk_band": risk_band,
            "risk_color": band_info["color"],
            "risk_label": band_info["label"],
            "executive_summary": verification.get("executive_summary", ""),
            "documents": session_data.get("documents", []),
            "extracted_data": session_data.get("extracted_data", {}),
            "critical_checks": [c for c in checks if c.get("severity") == "CRITICAL"],
            "high_checks": [c for c in checks if c.get("severity") == "HIGH"],
            "medium_checks": [c for c in checks if c.get("severity") == "MEDIUM"],
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

    # Build the executive PDF
    return _build_pdf(session_data)
