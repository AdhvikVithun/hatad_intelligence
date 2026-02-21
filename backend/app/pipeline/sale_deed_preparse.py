"""Deterministic pre-parser for Tamil Nadu Sale Deeds.

Runs BEFORE LLM extraction to anchor key fields using regex patterns.
Produces a ``PreparseHints`` dict that is injected into the LLM prompt,
giving the model strong structural anchoring so it doesn't confuse
document numbers with survey numbers, residential addresses with
property villages, or sellers with buyers.

All functions are pure (no I/O, no LLM calls) and fully unit-testable.
"""

from __future__ import annotations

import re
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════

class PreparseHints(TypedDict, total=False):
    """Hints extracted deterministically from OCR text."""
    registration_number: str          # e.g. "R/Vadavalli/Book1/5909/2012"
    registration_short: str           # e.g. "5909/2012"
    sro: str                         # Sub-Registrar Office name
    survey_numbers: list[str]         # e.g. ["317", "317/1A"]
    pan_numbers: list[str]            # e.g. ["AGUPP4291N", "ALCPS2485M"]
    consideration_amount: int | None  # e.g. 56792000
    property_village: str             # from கிராமம் keyword context
    property_taluk: str               # from தாலூக்கா keyword context
    property_district: str            # from மாவட்டம் keyword context
    seller_section: str               # text identified as seller portion
    buyer_section: str                # text identified as buyer portion
    previous_deed_date: str           # e.g. "20.01.1992"
    previous_owner: str               # e.g. "M. ஆறுமுகம்"
    property_type: str                # Agricultural / Residential from keywords
    stamp_value: int | None           # From stamp paper denomination
    ownership_chain_count: int        # How many prior transfers detected
    payment_mode: str                 # Cash / Cheque / DD / Bank Transfer
    has_encumbrance_declaration: bool  # Whether a free-from-encumbrance clause found


# ═══════════════════════════════════════════════════
# REGISTRATION NUMBER EXTRACTION
# ═══════════════════════════════════════════════════

# Pattern: R/SROName/Book1/5909/2012  (certified copy header)
_RE_REG_FULL = re.compile(
    r'R\s*/\s*([A-Za-z\u0B80-\u0BFF]+(?:\s+[A-Za-z\u0B80-\u0BFF]+)*)\s*/\s*'
    r'Book\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d{4})',
    re.IGNORECASE,
)

# Short form: "5909/12" or "5909/2012" from அஞ்சல் எண்
_RE_REG_SHORT = re.compile(
    r'(?:வருடம்|அஞ்சல்\s*எண்)[:\s]*(\d{3,5})\s*/?\s*(\d{2,4})',
)

# "No : 281" or "No.: 281" style on stamp paper
_RE_DOC_NO = re.compile(r'No\s*[.:]\s*(\d+)', re.IGNORECASE)


def extract_registration_number(text: str) -> tuple[str, str, str]:
    """Extract registration number, short form, and SRO from text.

    Returns:
        (full_reg_number, short_form, sro_name) — empty strings if not found.
    """
    # Try full form first
    m = _RE_REG_FULL.search(text)
    if m:
        sro = m.group(1).strip()
        book = m.group(2)
        num = m.group(3)
        year = m.group(4)
        full = f"R/{sro}/Book{book}/{num}/{year}"
        short = f"{num}/{year}"
        return full, short, sro

    # Try short form
    m = _RE_REG_SHORT.search(text)
    if m:
        num = m.group(1)
        year = m.group(2)
        short = f"{num}/{year}"
        return "", short, ""

    return "", "", ""


# ═══════════════════════════════════════════════════
# SURVEY NUMBER EXTRACTION
# ═══════════════════════════════════════════════════

# Tamil: க.ச. 317 or க.ச.எண். 317/1A
_RE_SURVEY_TAMIL = re.compile(
    r'(?:க\s*\.\s*ச\s*\.?\s*(?:எண்\s*\.?\s*)?|புல\s*(?:எண்|ம்\s*எண்)\s*\.?\s*)'
    r'(\d+(?:\s*/\s*\w+)*)',
)

# English: S.F.No. 317, S.F.No.317/1A, Survey No. 317
_RE_SURVEY_ENG = re.compile(
    r'(?:S\.?\s*F\.?\s*No\.?\s*|T\.?\s*S\.?\s*No\.?\s*|R\.?\s*S\.?\s*No\.?\s*|'
    r'Survey\s+No\.?\s*)'
    r'(\d+(?:\s*/\s*\w+)*)',
    re.IGNORECASE,
)


def extract_survey_numbers(text: str) -> list[str]:
    """Extract survey numbers from OCR text.

    Distinguishes survey numbers (க.ச., S.F.No.) from document/registration
    numbers (which use patterns like R/SRO/Book/NNNN/YYYY or NNNN/YY).
    """
    found: list[str] = []

    for pat in [_RE_SURVEY_TAMIL, _RE_SURVEY_ENG]:
        for m in pat.finditer(text):
            sn = m.group(1).strip()
            # Normalize whitespace around /
            sn = re.sub(r'\s*/\s*', '/', sn)
            if sn and sn not in found:
                found.append(sn)

    return found


# ═══════════════════════════════════════════════════
# PAN NUMBER EXTRACTION
# ═══════════════════════════════════════════════════

_RE_PAN = re.compile(r'(?:PAN\s*(?:NO\.?\s*)?)?([A-Z]{5}\s?\d{4}\s?[A-Z])', re.IGNORECASE)


def extract_pan_numbers(text: str) -> list[str]:
    """Extract Indian PAN numbers from text."""
    found: list[str] = []
    for m in _RE_PAN.finditer(text):
        pan = re.sub(r'\s+', '', m.group(1)).upper()
        if pan not in found:
            found.append(pan)
    return found


# ═══════════════════════════════════════════════════
# CONSIDERATION AMOUNT EXTRACTION
# ═══════════════════════════════════════════════════

# "ரூபாய் 5,67,92,000/-"  or "ரூ. 56,79,200" or "₹ 5,67,92,000"
_RE_AMOUNT_TAMIL = re.compile(
    r'(?:ரூபாய்|ரூ\.?|₹|Rs\.?)\s*([\d,]+(?:\.\d{1,2})?)\s*(?:/\s*-)?',
    re.IGNORECASE,
)

# "TWENTY FIVE THOUSAND RUPEES" on stamp paper
_RE_STAMP_VALUE = re.compile(
    r'(?:Rs\.?|ரூ\.?)\s*\n?\s*([\d,]+)\s*\n',
    re.IGNORECASE,
)


def extract_consideration_amount(text: str) -> int | None:
    """Extract the largest monetary amount (likely consideration) from text."""
    amounts: list[int] = []
    for m in _RE_AMOUNT_TAMIL.finditer(text):
        raw = m.group(1).replace(',', '').replace(' ', '')
        try:
            val = int(float(raw))
            if val >= 1000:  # Filter trivial amounts
                amounts.append(val)
        except ValueError:
            continue

    return max(amounts) if amounts else None


def extract_stamp_value(text: str) -> int | None:
    """Extract stamp paper denomination from first page."""
    m = _RE_STAMP_VALUE.search(text[:2000])  # stamp paper is on page 1
    if m:
        raw = m.group(1).replace(',', '')
        try:
            return int(raw)
        except ValueError:
            pass
    return None


# ═══════════════════════════════════════════════════
# SELLER / BUYER SECTION DETECTION
# ═══════════════════════════════════════════════════

# Key Tamil phrases that MARK the transition from seller to buyer context:
# "எழுதி வாங்குபவர்" = "the one who writes and takes" = BUYER
# "எழுதிக் கொடுப்பவர்" = "the one who writes and gives" = SELLER
# "கிரய சாசனம் / கிரயப் பத்திரம்" = sale deed
# "எழுதிக் கொடுத்த" = "who wrote and gave" (past tense, refers to sellers)
# "வசித்து வரும்" = "residing at" (marks person's address, not property)

# Buyer-start markers (buyer section begins here)
_BUYER_MARKERS = [
    'எழுதி வாங்குபவர்',        # buyer (formal)
    'வாங்குபவர்',               # buyer (short)
    'எழுதி வாங்கு',             # buyer variant
    'purchaser',                 # English
]

# Seller signature markers
_SELLER_SIG_MARKERS = [
    'எழுதிக் கொடுப்பவர்',      # seller (formal)
    'கொடுப்பவர்',               # giver
    'விற்பனையாளர்',             # seller
    'vendor',                    # English
]


def detect_seller_buyer_sections(text: str) -> tuple[str, str]:
    """Split text into seller-context and buyer-context sections.

    In a Tamil sale deed, the typical structure is:
      [Stamp paper + seller names/addresses] ... "எழுதிக் கொடுத்த ... கிரய சாசனம்"
      ... property description ...
      "எழுதி வாங்குபவர்" [buyer names] ...

    Returns:
        (seller_section, buyer_section) — may be empty if markers not found.
    """
    text_lower = text.lower()

    # Find earliest buyer marker
    buyer_pos = len(text)
    for marker in _BUYER_MARKERS:
        idx = text_lower.find(marker.lower())
        if idx != -1 and idx < buyer_pos:
            buyer_pos = idx

    if buyer_pos < len(text):
        seller_section = text[:buyer_pos].strip()
        buyer_section = text[buyer_pos:].strip()
        return seller_section, buyer_section

    return "", ""


# ═══════════════════════════════════════════════════
# PROPERTY LOCATION EXTRACTION  (village / taluk / district)
# ═══════════════════════════════════════════════════

# "கிராமம்" (village) keyword — property village appears near this
_RE_VILLAGE = re.compile(
    r'(?:கிராமம்|கிராம)\s*[,:.]?\s*'
    r'([\u0B80-\u0BFF][^\n,;]{2,40})',
)

# VillageName கிராமம் — the most common Tamil form: name BEFORE keyword
_RE_VILLAGE_BEFORE = re.compile(
    r'([\u0B80-\u0BFF][\u0B80-\u0BFF\w\s\.]{2,40}?)\s+கிராமம்',
)

# "கிராமத்தில்" = "in the village of" (locative case)
_RE_VILLAGE_LOC = re.compile(
    r'([\u0B80-\u0BFF][\u0B80-\u0BFF\s]{2,30})\s*கிராமத்தில்',
)

# "தாலூக்கா" / "தாலூக்" / "வட்டம்" (taluk)
_RE_TALUK = re.compile(
    r'(?:தாலூக்கா|தாலூக்|வட்டம்)\s*[,:.]?\s*'
    r'([\u0B80-\u0BFF][^\n,;]{2,40})',
)

# TalukName தாலூக்கா — name BEFORE keyword (most common Tamil form)
_RE_TALUK_BEFORE = re.compile(
    r'([\u0B80-\u0BFF][\u0B80-\u0BFF\w\s\.]{2,40}?)\s+(?:தாலூக்கா|தாலூக்|வட்டம்)',
)

# "மாவட்டம்" (district) — but NOT as part of an address
_RE_DISTRICT = re.compile(
    r'(?:மாவட்டம்)\s*[,:.]?\s*'
    r'([\u0B80-\u0BFF][^\n,;]{2,40})',
)

# DistrictName மாவட்டம் — name BEFORE keyword
_RE_DISTRICT_BEFORE = re.compile(
    r'([\u0B80-\u0BFF][\u0B80-\u0BFF\w\s\.]{2,40}?)\s+மாவட்டம்',
)

# English keywords near property schedule
_RE_VILLAGE_ENG = re.compile(
    r'(?:Village|Gramam)\s*[:\-]?\s*([A-Za-z\u0B80-\u0BFF][\w\s]{2,40})',
    re.IGNORECASE,
)


def extract_property_location(text: str) -> tuple[str, str, str]:
    """Extract property village, taluk, district from text.

    Avoids confusion with residential addresses by:
    - Preferring text near survey number markers (க.ச., S.F.No.)
    - Ignoring text near "வசித்து வரும்" (residing at) or "கதவு எண்" (door no.)
    """
    village = ""
    taluk = ""
    district = ""

    # Strategy 1: Look near survey number markers for property location
    # Find positions of survey markers
    survey_marker_positions: list[int] = []
    for pat in [_RE_SURVEY_TAMIL, _RE_SURVEY_ENG]:
        for m in pat.finditer(text):
            survey_marker_positions.append(m.start())

    # Search for location keywords near survey markers (within 500 chars)
    for spos in survey_marker_positions:
        window_start = max(0, spos - 500)
        window_end = min(len(text), spos + 500)
        window = text[window_start:window_end]

        if not village:
            # Try "VillageName கிராமம்" first (most common Tamil form)
            m = _RE_VILLAGE_BEFORE.search(window)
            if m:
                village = _clean_location(m.group(1))
            else:
                m = _RE_VILLAGE.search(window)
                if m:
                    village = _clean_location(m.group(1))
                else:
                    m = _RE_VILLAGE_LOC.search(window)
                    if m:
                        village = _clean_location(m.group(1))

        if not taluk:
            m = _RE_TALUK_BEFORE.search(window)
            if m:
                taluk = _clean_location(m.group(1))
            else:
                m = _RE_TALUK.search(window)
                if m:
                    taluk = _clean_location(m.group(1))
        if not district:
            m = _RE_DISTRICT_BEFORE.search(window)
            if m:
                district = _clean_location(m.group(1))
            else:
                m = _RE_DISTRICT.search(window)
                if m:
                    district = _clean_location(m.group(1))

    # Strategy 2: Global search but exclude address contexts
    address_markers = ['வசித்து வரும்', 'கதவு எண்', 'வீட்டில்', 'door no']
    lines = text.split('\n')

    for line in lines:
        line_lower = line.lower()
        # Skip lines that look like personal addresses
        if any(am.lower() in line_lower for am in address_markers):
            continue

        if not village:
            m = _RE_VILLAGE_BEFORE.search(line)
            if m:
                village = _clean_location(m.group(1))
            else:
                m = _RE_VILLAGE.search(line)
                if m:
                    village = _clean_location(m.group(1))
                else:
                    m = _RE_VILLAGE_LOC.search(line)
                    if m:
                        village = _clean_location(m.group(1))

        if not taluk:
            m = _RE_TALUK_BEFORE.search(line)
            if m:
                taluk = _clean_location(m.group(1))
            else:
                m = _RE_TALUK.search(line)
                if m:
                    taluk = _clean_location(m.group(1))

        if not district:
            m = _RE_DISTRICT_BEFORE.search(line)
            if m:
                district = _clean_location(m.group(1))
            else:
                m = _RE_DISTRICT.search(line)
                if m:
                    district = _clean_location(m.group(1))

    return village, taluk, district


def _clean_location(s: str) -> str:
    """Strip trailing punctuation and whitespace from location capture."""
    s = s.strip()
    s = re.sub(r'[\s,;.:\-]+$', '', s)
    return s.strip()


# ═══════════════════════════════════════════════════
# PREVIOUS OWNERSHIP EXTRACTION
# ═══════════════════════════════════════════════════

# "20.01.1992-ம் தேதியில்" or "dated 20.01.1992" near deed reference
_RE_PREV_DATE = re.compile(
    r'(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})\s*-?\s*(?:ம்\s*)?(?:தேதி|dated|நாள்)',
    re.IGNORECASE,
)

# "பத்திரம் எழுதிக் கொடுத்து" = "wrote and gave the deed"
_RE_PREV_DEED = re.compile(
    r'(?:பத்திரம்\s*எழுதிக்?\s*கொடுத்|deed\s*(?:dated|executed))',
    re.IGNORECASE,
)

# Person name before "அவர்களின் குமாரர்" (son of) or "S/o"
_RE_PREV_OWNER = re.compile(
    r'(?:திரு|Mr|Sri|Thiru)\.?\s*'
    r'([A-Za-z\u0B80-\u0BFF][\w\s.\u0B80-\u0BFF]{2,50}?)\s*'
    r'(?:அவர்கள|கவுண்டர்|S/o|D/o|W/o)',
    re.IGNORECASE,
)


def extract_previous_ownership(text: str) -> tuple[str, str]:
    """Extract previous deed date and previous owner name.

    Returns:
        (previous_deed_date, previous_owner_name) — empty if not found.
    """
    prev_date = ""
    prev_owner = ""

    # Look for previous deed context
    m = _RE_PREV_DEED.search(text)
    if m:
        # Search for date within 200 chars before the deed mention
        window_start = max(0, m.start() - 200)
        window = text[window_start:m.end() + 200]

        dm = _RE_PREV_DATE.search(window)
        if dm:
            prev_date = dm.group(1)

        om = _RE_PREV_OWNER.search(window)
        if om:
            prev_owner = om.group(1).strip()

    if not prev_date:
        # Global search for previous deed date
        m = _RE_PREV_DATE.search(text)
        if m:
            prev_date = m.group(1)

    return prev_date, prev_owner


# ═══════════════════════════════════════════════════
# OWNERSHIP CHAIN DETECTION (count of prior transfers)
# ═══════════════════════════════════════════════════

# Patterns that indicate a prior ownership transfer mention
_RE_DEED_TRANSFER = re.compile(
    r'(?:பத்திரம்\s*(?:எழுதி|வைத்|வழங்கி|கொடுத்)|'
    r'deed\s*(?:dated|no|number|executed)|'
    r'கிரய(?:(?:ப்\s*)?பத்திரம்|ம்)|'
    r'sale\s*deed\s*(?:dated|no)|'
    r'வாரிசு\s*(?:சான்|பட்ட)|'         # inheritance certificate
    r'பாகப்பிரிவினை|'                  # partition
    r'தான\s*(?:(?:ப்\s*)?பத்திரம்|ம்)|'  # gift deed
    r'settlement\s*deed)',
    re.IGNORECASE,
)


def count_ownership_transfers(text: str) -> int:
    """Count how many prior ownership transfers are mentioned in the recitals.

    Returns the number of distinct deed/transfer references found.
    """
    return len(_RE_DEED_TRANSFER.findall(text))


# ═══════════════════════════════════════════════════
# PAYMENT MODE DETECTION
# ═══════════════════════════════════════════════════

_PAYMENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'NEFT|RTGS|(?:online|bank)\s*transfer|வங்கி\s*பரிவர்த்தனை', re.IGNORECASE),
     "Bank Transfer"),
    (re.compile(r'(?:demand\s*)?draft|வங்கி\s*வரைவோலை|D\.?D\.?', re.IGNORECASE),
     "Demand Draft"),
    (re.compile(r'cheque|காசோலை|காசாளை|செக்', re.IGNORECASE),
     "Cheque"),
    (re.compile(r'ரொக்க(?:ம்|மாக)|cash|நகத', re.IGNORECASE),
     "Cash"),
]


def detect_payment_mode(text: str) -> str:
    """Detect payment mode from text.

    Returns the first matching payment mode or empty string.
    """
    for pat, mode in _PAYMENT_PATTERNS:
        if pat.search(text):
            return mode
    return ""


# ═══════════════════════════════════════════════════
# ENCUMBRANCE DECLARATION DETECTION
# ═══════════════════════════════════════════════════

_RE_ENCUMBRANCE_FREE = re.compile(
    r'(?:பாரமில்லை|'                   # sandhi form (no space)
    r'பாரம்?\s*(?:இல்லை|இல்ல)|'
    r'எவ்வித(?:மான)?\s*பாரம்?\s*(?:இல்லை|இல்ல)|'
    r'free\s*(?:from|of)\s*(?:all\s*)?encumbrance|'
    r'no\s*encumbrance|'
    r'ஈடு\s*கட்டு\s*இல்லை|'
    r'இந்தச?\s*சொத்து\s*எவ்வித)',
    re.IGNORECASE,
)


def detect_encumbrance_declaration(text: str) -> bool:
    """Detect whether the deed contains a free-from-encumbrance declaration."""
    return bool(_RE_ENCUMBRANCE_FREE.search(text))


# ═══════════════════════════════════════════════════
# PROPERTY TYPE DETECTION
# ═══════════════════════════════════════════════════

_AGRI_KEYWORDS = [
    'விவசாய', 'விவசாயம்', 'நிலம்', 'புன்செய்', 'புஞ்சை',
    'நன்செய்', 'agricultural', 'farm', 'dry land', 'wet land',
]

_RESI_KEYWORDS = [
    'வீட்டு', 'குடியிருப்பு', 'residential', 'house site',
    'plot', 'flat', 'apartment',
]

# "மனை" removed from _RESI_KEYWORDS — it's a substring of "மனைவி" (wife)
# and causes false positives. Use _RE_MANAI for bounded matching instead.
_RE_MANAI = re.compile(r'மனை(?!வி)')  # "மனை" not followed by "வி" (wife)

_COMM_KEYWORDS = [
    'வணிக', 'commercial', 'shop', 'office', 'industrial',
]


def detect_property_type(text: str) -> str:
    """Detect property type from keyword frequency in text."""
    text_lower = text.lower()
    agri_count = sum(1 for kw in _AGRI_KEYWORDS if kw.lower() in text_lower)
    resi_count = sum(1 for kw in _RESI_KEYWORDS if kw.lower() in text_lower)
    if _RE_MANAI.search(text):
        resi_count += 1
    comm_count = sum(1 for kw in _COMM_KEYWORDS if kw.lower() in text_lower)

    if agri_count > resi_count and agri_count > comm_count:
        return "Agricultural"
    if resi_count > agri_count and resi_count > comm_count:
        return "Residential"
    if comm_count > 0:
        return "Commercial"
    return ""


# ═══════════════════════════════════════════════════
# MAIN PRE-PARSER ENTRY POINT
# ═══════════════════════════════════════════════════

def preparse_sale_deed(full_text: str) -> PreparseHints:
    """Run all deterministic pre-parsing on sale deed OCR text.

    Returns a PreparseHints dict with all fields that could be extracted.
    Empty/None values mean the field could not be determined.
    """
    hints: PreparseHints = {}

    # Registration number
    reg_full, reg_short, sro = extract_registration_number(full_text)
    if reg_full:
        hints["registration_number"] = reg_full
    if reg_short:
        hints["registration_short"] = reg_short
    if sro:
        hints["sro"] = sro

    # Survey numbers
    surveys = extract_survey_numbers(full_text)
    if surveys:
        hints["survey_numbers"] = surveys

    # PAN numbers
    pans = extract_pan_numbers(full_text)
    if pans:
        hints["pan_numbers"] = pans

    # Consideration amount
    amount = extract_consideration_amount(full_text)
    if amount:
        hints["consideration_amount"] = amount

    # Stamp paper value
    stamp = extract_stamp_value(full_text)
    if stamp:
        hints["stamp_value"] = stamp

    # Property location
    village, taluk, district = extract_property_location(full_text)
    if village:
        hints["property_village"] = village
    if taluk:
        hints["property_taluk"] = taluk
    if district:
        hints["property_district"] = district

    # Seller/buyer section detection
    seller_sec, buyer_sec = detect_seller_buyer_sections(full_text)
    if seller_sec:
        hints["seller_section"] = seller_sec
    if buyer_sec:
        hints["buyer_section"] = buyer_sec

    # Previous ownership
    prev_date, prev_owner = extract_previous_ownership(full_text)
    if prev_date:
        hints["previous_deed_date"] = prev_date
    if prev_owner:
        hints["previous_owner"] = prev_owner

    # Property type
    prop_type = detect_property_type(full_text)
    if prop_type:
        hints["property_type"] = prop_type

    # Ownership chain count (for prompt guidance)
    chain_count = count_ownership_transfers(full_text)
    if chain_count:
        hints["ownership_chain_count"] = chain_count

    # Payment mode
    payment = detect_payment_mode(full_text)
    if payment:
        hints["payment_mode"] = payment

    # Encumbrance declaration
    has_encumbrance = detect_encumbrance_declaration(full_text)
    if has_encumbrance:
        hints["has_encumbrance_declaration"] = True

    logger.info(
        f"Sale deed pre-parse: {len(hints)} hints extracted "
        f"(reg={bool(reg_full or reg_short)}, survey={len(surveys)}, "
        f"pan={len(pans)}, village={bool(village)}, "
        f"seller_section={bool(seller_sec)}, amount={amount}, "
        f"chain={chain_count}, payment={payment})"
    )

    return hints


def format_hints_for_prompt(hints: PreparseHints) -> str:
    """Format pre-parse hints into a text block for LLM prompt injection.

    This gives the LLM strong anchoring to avoid confusing:
    - Document number vs survey number
    - Residential address vs property village
    - Seller vs buyer roles
    """
    if not hints:
        return ""

    lines = ["DETERMINISTIC PRE-PARSE HINTS (high confidence — use these to guide extraction):"]

    if hints.get("registration_number"):
        lines.append(f"  Registration Number: {hints['registration_number']}")
        lines.append(f"    → This is the DOCUMENT registration number, NOT a survey number")
    elif hints.get("registration_short"):
        lines.append(f"  Registration Number (short): {hints['registration_short']}")
        lines.append(f"    → This is the DOCUMENT registration number, NOT a survey number")

    if hints.get("sro"):
        lines.append(f"  SRO (Sub-Registrar Office): {hints['sro']}")

    if hints.get("survey_numbers"):
        lines.append(f"  Survey Number(s): {', '.join(hints['survey_numbers'])}")
        lines.append(f"    → These are the PROPERTY survey numbers (க.ச. / S.F.No.)")

    if hints.get("property_village"):
        lines.append(f"  Property Village: {hints['property_village']}")
        lines.append(f"    → This is the PROPERTY location village, NOT a person's residential address")

    if hints.get("property_taluk"):
        lines.append(f"  Property Taluk: {hints['property_taluk']}")

    if hints.get("property_district"):
        lines.append(f"  Property District: {hints['property_district']}")

    if hints.get("property_type"):
        lines.append(f"  Property Type: {hints['property_type']}")

    if hints.get("consideration_amount"):
        lines.append(f"  Consideration Amount: ₹{hints['consideration_amount']:,}")
        lines.append(f"    → Use this as financials.consideration_amount (integer: {hints['consideration_amount']})")
        lines.append(f"    → Do NOT return 0 — the amount is confirmed from the stamp paper/deed text")
    if hints.get("pan_numbers"):
        lines.append(f"  PAN Numbers Found: {', '.join(hints['pan_numbers'])}")

    if hints.get("seller_section") and hints.get("buyer_section"):
        lines.append(f"  Seller/Buyer Role Detection:")
        lines.append(f"    → Text BEFORE 'எழுதி வாங்குபவர்' contains SELLER names/addresses")
        lines.append(f"    → Text AFTER 'எழுதி வாங்குபவர்' contains BUYER names/addresses")

    if hints.get("previous_deed_date"):
        lines.append(f"  Previous Deed Date: {hints['previous_deed_date']}")
    if hints.get("previous_owner"):
        lines.append(f"  Previous Owner: {hints['previous_owner']}")

    if hints.get("ownership_chain_count"):
        lines.append(f"  Prior Ownership Transfers Detected: {hints['ownership_chain_count']}")
        lines.append(f"    → Extract EACH transfer as an ownership_history entry (oldest first)")

    if hints.get("payment_mode"):
        lines.append(f"  Payment Mode Detected: {hints['payment_mode']}")

    if hints.get("has_encumbrance_declaration"):
        lines.append(f"  Encumbrance Declaration: Found — extract the exact clause text")

    return "\n".join(lines)
