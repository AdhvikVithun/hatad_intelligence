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
    book_number: str                 # e.g. "Book 1" from registration
    survey_numbers: list[str]         # e.g. ["317", "317/1A"]
    pan_numbers: list[str]            # e.g. ["AGUPP4291N", "ALCPS2485M"]
    aadhaar_numbers: list[str]        # e.g. ["123456789012"]
    consideration_amount: int | None  # e.g. 56792000
    property_village: str             # from கிராமம் keyword context
    property_taluk: str               # from தாலூக்கா keyword context
    property_district: str            # from மாவட்டம் keyword context
    land_classification: str          # Wet/Dry/Maidan from keywords
    seller_section: str               # text identified as seller portion
    buyer_section: str                # text identified as buyer portion
    seller_names: list[str]           # deterministic seller name candidates
    buyer_names: list[str]            # deterministic buyer name candidates
    previous_deed_date: str           # e.g. "20.01.1992"
    previous_owner: str               # e.g. "M. ஆறுமுகம்"
    property_type: str                # Agricultural / Residential from keywords
    stamp_value: int | None           # From stamp paper denomination (per-sheet)
    stamp_sheet_count: int            # Number of physical stamp sheets detected
    stamp_serial_numbers: list[str]   # Stamp paper serial numbers detected
    stamp_vendor_name: str            # Stamp vendor name
    total_stamp_value: int | None     # stamp_value × stamp_sheet_count
    stamp_duty_from_text: int | None  # Explicit stamp duty figure from deed body
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


def extract_survey_numbers(
    text: str,
    schedule_zone: tuple[int, int] | None = None,
) -> list[str]:
    """Extract survey numbers from OCR text.

    Distinguishes survey numbers (க.ச., S.F.No.) from document/registration
    numbers (which use patterns like R/SRO/Book/NNNN/YYYY or NNNN/YY).

    When *schedule_zone* is provided, surveys are extracted from the schedule
    zone first.  Only if that yields nothing does the function fall back to a
    full-text scan (with recital-context filtering).
    """
    def _scan(region: str) -> list[str]:
        hits: list[str] = []
        for pat in [_RE_SURVEY_TAMIL, _RE_SURVEY_ENG]:
            for m in pat.finditer(region):
                sn = m.group(1).strip()
                sn = re.sub(r'\s*/\s*', '/', sn)
                if sn and sn not in hits:
                    hits.append(sn)
        return hits

    # Strategy 1: schedule-zone extraction (deterministic)
    if schedule_zone:
        sz_start, sz_end = schedule_zone
        zone_hits = _scan(text[sz_start:sz_end])
        if zone_hits:
            return zone_hits

    # Strategy 2: full-text with recital-context filtering
    found: list[str] = []
    for pat in [_RE_SURVEY_TAMIL, _RE_SURVEY_ENG]:
        for m in pat.finditer(text):
            sn = m.group(1).strip()
            sn = re.sub(r'\s*/\s*', '/', sn)
            if not sn or sn in found:
                continue
            # Filter: skip survey numbers near recital markers (historical refs)
            pos = m.start()
            window_start = max(0, pos - 300)
            window_end = min(len(text), pos + 100)
            window = text[window_start:window_end]
            if _RE_RECITAL_MARKER.search(window):
                # Still include if it's also near a schedule / property marker
                if not (_RE_SCHEDULE_START.search(window) or _RE_PROPERTY_INTRO.search(window)):
                    continue
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
# AADHAAR NUMBER EXTRACTION
# ═══════════════════════════════════════════════════

# 12-digit Aadhaar: "1234 5678 9012" or "123456789012" or "ஆதார் எண்: 1234 5678 9012"
_RE_AADHAAR = re.compile(
    r'(?:aadhaar|aadhar|UID|ஆதார்|ஆதா(?:ர்)?\s*எண்)\s*(?:NO\.?\s*)?[:\s]*'
    r'(\d{4}\s?\d{4}\s?\d{4})',
    re.IGNORECASE,
)

# Standalone 12-digit pattern (less strict — only use after keyword match fails)
_RE_AADHAAR_BARE = re.compile(r'\b(\d{4}\s\d{4}\s\d{4})\b')


def extract_aadhaar_numbers(text: str) -> list[str]:
    """Extract Aadhaar numbers (12 digits) from text.

    First tries keyword-anchored patterns (high confidence), then falls
    back to bare 12-digit spaced patterns near party sections.
    """
    found: list[str] = []
    for m in _RE_AADHAAR.finditer(text):
        aadhaar = re.sub(r'\s+', '', m.group(1))
        if len(aadhaar) == 12 and aadhaar not in found:
            found.append(aadhaar)

    # Bare pattern — only in first half (party sections) to avoid false positives
    if not found:
        party_zone = text[:len(text) // 2]
        for m in _RE_AADHAAR_BARE.finditer(party_zone):
            aadhaar = re.sub(r'\s+', '', m.group(1))
            if len(aadhaar) == 12 and aadhaar not in found:
                found.append(aadhaar)

    return found


# ═══════════════════════════════════════════════════
# LAND CLASSIFICATION DETECTION
# ═══════════════════════════════════════════════════

_LAND_CLASS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'நன்சை|நன்செய்|wet\s*land', re.IGNORECASE), "Wet"),
    (re.compile(r'புன்சை|புஞ்சை|புன்செய்|dry\s*land', re.IGNORECASE), "Dry"),
    (re.compile(r'மைதானம்|maidan', re.IGNORECASE), "Maidan"),
    (re.compile(r'தோட்டம்|garden', re.IGNORECASE), "Garden"),
]


def detect_land_classification(text: str) -> str:
    """Detect land classification from property schedule keywords.

    Returns the first matching classification: Wet, Dry, Maidan, Garden, or "".
    """
    # Prefer schedule zone if found
    schedule_zone = _find_schedule_zone(text)
    search_text = text[schedule_zone[0]:schedule_zone[1]] if schedule_zone else text

    for pat, classification in _LAND_CLASS_PATTERNS:
        if pat.search(search_text):
            return classification
    # Fallback: full text
    if schedule_zone:
        for pat, classification in _LAND_CLASS_PATTERNS:
            if pat.search(text):
                return classification
    return ""


# ═══════════════════════════════════════════════════
# STAMP VENDOR NAME EXTRACTION
# ═══════════════════════════════════════════════════

_RE_STAMP_VENDOR = re.compile(
    r'(?:stamp\s*vendor|முத்திரைத்தாள்\s*வணிகர்)\s*[:\-]?\s*'
    r'([A-Za-z\u0B80-\u0BFF][A-Za-z\u0B80-\u0BFF\s.]{3,60})',
    re.IGNORECASE,
)

# Alternative: Name appears after "Licensed" and before next line break
_RE_VENDOR_ALT = re.compile(
    r'licensed\s+(?:stamp\s+)?vendor\s*[:\-]?\s*'
    r'([A-Za-z][A-Za-z\s.]{3,60})',
    re.IGNORECASE,
)


def extract_stamp_vendor_name(text: str) -> str:
    """Extract stamp vendor name from stamp paper header."""
    header = text[:4000]  # Vendor info is on first page
    m = _RE_STAMP_VENDOR.search(header)
    if m:
        return m.group(1).strip().rstrip('.,;:-')
    m = _RE_VENDOR_ALT.search(header)
    if m:
        return m.group(1).strip().rstrip('.,;:-')
    return ""


# ═══════════════════════════════════════════════════
# STAMP SERIAL NUMBER COLLECTION
# ═══════════════════════════════════════════════════

def extract_stamp_serial_numbers(text: str) -> list[str]:
    """Extract distinct stamp paper serial numbers from text.

    Returns a list of normalized serial numbers (e.g. ['A123456', 'A123457']).
    """
    header = text[:8000]
    serials: list[str] = []
    for m in _RE_STAMP_SERIAL.finditer(header):
        serial = re.sub(r'\s+', '', m.group(0)).upper()
        if serial not in serials:
            serials.append(serial)
    return serials


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


# Stamp paper serial number patterns — OCR may drop/merge the space.
# "A 123456", "A123456", "a 123456"
_RE_STAMP_SERIAL = re.compile(
    r'[A-Za-z]\s*\d{5,8}',
)

# "Rs. 25,000" or "ரூ 25000" denomination on stamp paper
_RE_STAMP_DENOM = re.compile(
    r'(?:Rs\.?|ரூ\.?|₹)\s*([\d,]+)\s*(?:/\s*-)?',
    re.IGNORECASE,
)

# Explicit stamp duty amount as stated in the deed text body.
# "முத்திரைத் தீர்வை ரூ. 5,68,100" or "stamp duty Rs. 568100"
_RE_STAMP_DUTY_TEXT = re.compile(
    r'(?:முத்திரை(?:த்)?\s*தீர்வை|stamp\s*duty|முத்திரைத்தாள்\s*(?:கட்டணம்|விலை))'
    r'\s*(?:ரூ\.?|Rs\.?|₹)\s*([\d,]+)',
    re.IGNORECASE,
)


def extract_stamp_duty_from_text(text: str) -> int | None:
    """Extract explicit stamp duty amount from deed body text.

    Returns the parsed integer or None if no explicit mention found.
    """
    m = _RE_STAMP_DUTY_TEXT.search(text)
    if m:
        raw = m.group(1).replace(',', '')
        try:
            val = int(raw)
            if val >= 1000:  # filter trivial
                return val
        except ValueError:
            pass
    return None


def count_stamp_sheets(text: str) -> tuple[int, int | None]:
    """Count distinct stamp paper sheets and compute total stamp value.

    Tamil sale deeds printed on multiple stamp sheets each carry their own
    vendor serial number (e.g. "A 123456").  This function counts those
    serials and multiplies by the per-sheet denomination.

    Returns:
        (sheet_count, total_stamp_value) — sheet_count=0 if undetectable.
    """
    # Look in first ~8000 chars (multi-sheet deeds can have lengthy headers)
    header = text[:8000]

    serials = set()
    for m in _RE_STAMP_SERIAL.finditer(header):
        serial = m.group(0).strip()
        # Normalize: remove whitespace to deduplicate "A 123456" vs "A123456"
        serial_norm = re.sub(r'\s+', '', serial).upper()
        serials.add(serial_norm)

    sheet_count = len(serials)

    # Find denomination (typically on first stamp sheet)
    per_sheet: int | None = None
    dm = _RE_STAMP_DENOM.search(header)
    if dm:
        raw = dm.group(1).replace(',', '')
        try:
            val = int(raw)
            # Filter out tiny amounts that are NOT stamp denominations
            if val >= 100:
                per_sheet = val
        except ValueError:
            pass

    if sheet_count >= 2 and per_sheet:
        return sheet_count, sheet_count * per_sheet
    elif sheet_count == 1 and per_sheet:
        return 1, per_sheet
    elif per_sheet:
        return 0, per_sheet  # count unknown but denomination found
    return 0, None


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
# SELLER / BUYER NAME EXTRACTION FROM SECTIONS
# ═══════════════════════════════════════════════════

# Tamil honorific name pattern: திரு./திருமதி. + Initial + Name
_RE_TAMIL_NAME = re.compile(
    r'(?:திரு(?:மதி)?\.?\s*|Thiru(?:mathi)?\.?\s*|Mr\.?\s*|Mrs\.?\s*|Sri\.?\s*|Smt\.?\s*)'
    r'([A-Za-z\u0B80-\u0BFF][\w\s.\u0B80-\u0BFF]{1,40}?)(?:\s+அவர்கள்|\s+S/o|\s+D/o|\s+W/o|\s+மகன்|\s+மகள்|\s+மனைவி|\s*,|\s*\n)',
    re.IGNORECASE,
)

# Location keywords that should NEVER appear in a person's name.
# Names containing any of these are garbage captures from address text.
_LOCATION_BLOCKLIST: set[str] = {
    'மாவட்டம்', 'தாலூக்கா', 'தாலூக்', 'கிராமம்', 'கிராம',
    'நகரம்', 'டவுன்', 'வட்டம்', 'பேட்டை', 'நகர்', 'மண்டலம்',
    'ஊராட்சி', 'மாநகரம்', 'மாவட்ட', 'பகுதி', 'ரோடு', 'வீதி',
    'எக்ஸ்டென்சன்', 'லேஅவுட்', 'district', 'taluk', 'town',
    'village', 'nagar', 'road', 'street', 'extension', 'layout',
    'இலக்கமிட்ட', 'விலாசத்தில்', 'வசித்து',
}


def _is_garbage_name(name: str) -> bool:
    """Return True if *name* looks like an address fragment, not a person's name."""
    lower = name.lower()
    # Reject if any blocklist word appears anywhere in the name
    for bw in _LOCATION_BLOCKLIST:
        if bw.lower() in lower:
            return True
    # Reject very short captures that are honourific fragments (e.g. "மதி")
    # A real Tamil name has at least one initial-dot or multi-word structure.
    alpha_chars = sum(1 for c in name if c.isalpha() or '\u0B80' <= c <= '\u0BFF')
    if alpha_chars < 4:
        return True
    return False


def extract_party_names_from_section(section_text: str) -> list[str]:
    """Extract person names from a seller or buyer section using honorifics.

    Returns a list of cleaned names found in the text.  Filters out garbage
    captures that contain location keywords (மாவட்டம், டவுன், etc.).
    """
    names: list[str] = []
    for m in _RE_TAMIL_NAME.finditer(section_text):
        name = m.group(1).strip()
        # Clean trailing junk
        name = re.sub(r'[\s,;.\-]+$', '', name)
        if not name or len(name) <= 2 or name in names:
            continue
        if _is_garbage_name(name):
            continue
        names.append(name)
    return names


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


# Compiled once at module level for efficiency
_ADDRESS_MARKERS_RE = re.compile(
    r'வசித்து\s*வரும்|கதவு\s*எண்|வீட்டில்|door\s*no|residing\s*at|'
    r'முகவரி|address',
    re.IGNORECASE,
)


def _is_near_address_marker(text: str, match_start: int, radius: int = 500) -> bool:
    """Return True if ``match_start`` is within ``radius`` chars of an address marker.

    Address markers indicate a *person's residential address* — any village /
    taluk / district keyword found near them is NOT the property location.
    Radius default raised to 500 because Tamil deed address blocks often span
    300-400 chars across multiple lines (name + father + door + street + district).
    """
    window_start = max(0, match_start - radius)
    window_end = min(len(text), match_start + radius)
    return bool(_ADDRESS_MARKERS_RE.search(text[window_start:window_end]))


# Schedule-section markers that delimit the property description block
_RE_SCHEDULE_START = re.compile(
    r'(?:அட்டவணை|சொத்து\s*விவரம்|property\s*schedule|schedule\s*of\s*property)',
    re.IGNORECASE,
)

# Property-description intro markers (less formal than schedule headers)
_RE_PROPERTY_INTRO = re.compile(
    r'(?:கீழ்க்கண்ட\s*சொத்து|கீழ்கண்ட\s*சொத்து|below\s*mentioned\s*property|'
    r'property\s*more\s*fully\s*described|சொத்தின்\s*விவரம்)',
    re.IGNORECASE,
)

# Recital markers — survey numbers found near these are historical, not current
_RE_RECITAL_MARKER = re.compile(
    r'(?:பட்டா\s*எண்|patta\s*no|previously|earlier\s*deed|முன்பு|'
    r'முந்தைய|பழைய\s*பத்திரம்|கிரையப்\s*பத்திரம்\s*எழுதிக்)',
    re.IGNORECASE,
)


def _find_schedule_zone(text: str) -> tuple[int, int] | None:
    """Return (start, end) of the property schedule zone, or None.

    The schedule zone starts at an அட்டவணை/schedule marker and ends at the
    next clearly unrelated section (signatures, witnesses) or extends 2000
    chars, whichever comes first.
    """
    m = _RE_SCHEDULE_START.search(text)
    if not m:
        return None

    start = m.start()
    # End markers: signatures, witnesses, or next major section
    end_markers = re.compile(
        r'(?:சாட்சிகள்|எழுதிக் கொடுப்பவர்|witness|signature)',
        re.IGNORECASE,
    )
    em = end_markers.search(text[start + 100:])  # skip a bit past the header
    if em:
        end = start + 100 + em.start()
    else:
        end = min(len(text), start + 2000)
    return start, end


def extract_property_location(text: str) -> tuple[str, str, str]:
    """Extract property village, taluk, district from text.

    Avoids confusion with residential addresses by:
    - Strategy 0: If a property schedule zone is found, search ONLY there first.
    - Strategy 1: Preferring text near survey number markers (க.ச., S.F.No.)
      and filtering out matches within ±200 chars of address markers.
    - Strategy 2: Global search excluding address-context lines.
    """
    village = ""
    taluk = ""
    district = ""

    # ── Strategy 0: Property schedule zone ──
    schedule_zone = _find_schedule_zone(text)
    if schedule_zone:
        sz_start, sz_end = schedule_zone
        sz_text = text[sz_start:sz_end]
        for pat_v in [_RE_VILLAGE_BEFORE, _RE_VILLAGE, _RE_VILLAGE_LOC]:
            if not village:
                mv = pat_v.search(sz_text)
                if mv:
                    village = _clean_location(mv.group(1))
        for pat_t in [_RE_TALUK_BEFORE, _RE_TALUK]:
            if not taluk:
                mt = pat_t.search(sz_text)
                if mt:
                    taluk = _clean_location(mt.group(1))
        for pat_d in [_RE_DISTRICT_BEFORE, _RE_DISTRICT]:
            if not district:
                md = pat_d.search(sz_text)
                if md:
                    district = _clean_location(md.group(1))

    # ── Strategy 1: Near survey markers (with address proximity filter) ──
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
            if m and not _is_near_address_marker(text, window_start + m.start()):
                village = _clean_location(m.group(1))
            else:
                m = _RE_VILLAGE.search(window)
                if m and not _is_near_address_marker(text, window_start + m.start()):
                    village = _clean_location(m.group(1))
                else:
                    m = _RE_VILLAGE_LOC.search(window)
                    if m and not _is_near_address_marker(text, window_start + m.start()):
                        village = _clean_location(m.group(1))

        if not taluk:
            m = _RE_TALUK_BEFORE.search(window)
            if m and not _is_near_address_marker(text, window_start + m.start()):
                taluk = _clean_location(m.group(1))
            else:
                m = _RE_TALUK.search(window)
                if m and not _is_near_address_marker(text, window_start + m.start()):
                    taluk = _clean_location(m.group(1))
        if not district:
            m = _RE_DISTRICT_BEFORE.search(window)
            if m and not _is_near_address_marker(text, window_start + m.start()):
                district = _clean_location(m.group(1))
            else:
                m = _RE_DISTRICT.search(window)
                if m and not _is_near_address_marker(text, window_start + m.start()):
                    district = _clean_location(m.group(1))

    # ── Strategy 1.5: Property description intro zone ──
    # If no schedule marker, look for property-intro phrases
    if not (village and taluk and district) and not schedule_zone:
        pin = _RE_PROPERTY_INTRO.search(text)
        if pin:
            pz_start = pin.start()
            pz_end = min(len(text), pz_start + 1500)
            pz_text = text[pz_start:pz_end]
            for pat_v in [_RE_VILLAGE_BEFORE, _RE_VILLAGE, _RE_VILLAGE_LOC]:
                if not village:
                    mv = pat_v.search(pz_text)
                    if mv:
                        village = _clean_location(mv.group(1))
            for pat_t in [_RE_TALUK_BEFORE, _RE_TALUK]:
                if not taluk:
                    mt = pat_t.search(pz_text)
                    if mt:
                        taluk = _clean_location(mt.group(1))
            for pat_d in [_RE_DISTRICT_BEFORE, _RE_DISTRICT]:
                if not district:
                    md = pat_d.search(pz_text)
                    if md:
                        district = _clean_location(md.group(1))

    # Strategy 2: Global search but exclude address contexts (multi-line window)
    address_markers = ['வசித்து வரும்', 'கதவு எண்', 'வீட்டில்', 'door no']
    lines = text.split('\n')

    def _is_address_line(idx: int) -> bool:
        """Check line *idx* AND preceding 3 lines for address markers."""
        for offset in range(4):  # current line + 3 lines above
            li = idx - offset
            if li < 0:
                break
            ll = lines[li].lower()
            if any(am.lower() in ll for am in address_markers):
                return True
        return False

    for li, line in enumerate(lines):
        # Skip lines whose multi-line window contains address markers
        if _is_address_line(li):
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
        # Extract book number from full registration (R/SRO/BookN/...)
        bm = re.search(r'Book\s*(\d+)', reg_full, re.IGNORECASE)
        if bm:
            hints["book_number"] = f"Book {bm.group(1)}"
    if reg_short:
        hints["registration_short"] = reg_short
    if sro:
        hints["sro"] = sro

    # Survey numbers (prefer schedule zone)
    schedule_zone = _find_schedule_zone(full_text)
    surveys = extract_survey_numbers(full_text, schedule_zone=schedule_zone)
    if surveys:
        hints["survey_numbers"] = surveys

    # PAN numbers
    pans = extract_pan_numbers(full_text)
    if pans:
        hints["pan_numbers"] = pans

    # Aadhaar numbers
    aadhaars = extract_aadhaar_numbers(full_text)
    if aadhaars:
        hints["aadhaar_numbers"] = aadhaars

    # Consideration amount
    amount = extract_consideration_amount(full_text)
    if amount:
        hints["consideration_amount"] = amount

    # Stamp paper value
    stamp = extract_stamp_value(full_text)
    if stamp:
        hints["stamp_value"] = stamp

    # Stamp sheet counting (multi-sheet deeds)
    sheet_count, total_stamp = count_stamp_sheets(full_text)
    if sheet_count >= 2:
        hints["stamp_sheet_count"] = sheet_count
        if total_stamp:
            hints["total_stamp_value"] = total_stamp
    elif total_stamp and not stamp:
        # count_stamp_sheets found denomination but extract_stamp_value didn't
        hints["stamp_value"] = total_stamp

    # Stamp serial numbers
    stamp_serials = extract_stamp_serial_numbers(full_text)
    if stamp_serials:
        hints["stamp_serial_numbers"] = stamp_serials

    # Stamp vendor name
    vendor = extract_stamp_vendor_name(full_text)
    if vendor:
        hints["stamp_vendor_name"] = vendor

    # Explicit stamp duty from deed body text (highest priority)
    explicit_stamp_duty = extract_stamp_duty_from_text(full_text)
    if explicit_stamp_duty:
        hints["stamp_duty_from_text"] = explicit_stamp_duty

    # Property location
    village, taluk, district = extract_property_location(full_text)
    if village:
        hints["property_village"] = village
    if taluk:
        hints["property_taluk"] = taluk
    if district:
        hints["property_district"] = district

    # Land classification
    land_class = detect_land_classification(full_text)
    if land_class:
        hints["land_classification"] = land_class

    # Seller/buyer section detection
    seller_sec, buyer_sec = detect_seller_buyer_sections(full_text)
    if seller_sec:
        hints["seller_section"] = seller_sec
    if buyer_sec:
        hints["buyer_section"] = buyer_sec

    # Extract actual party names from sections (deterministic anchoring)
    if seller_sec:
        seller_names = extract_party_names_from_section(seller_sec)
        if seller_names:
            hints["seller_names"] = seller_names
    if buyer_sec:
        buyer_names = extract_party_names_from_section(buyer_sec)
        if buyer_names:
            hints["buyer_names"] = buyer_names

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
        f"pan={len(pans)}, aadhaar={len(aadhaars)}, village={bool(village)}, "
        f"land_class={land_class}, seller_section={bool(seller_sec)}, "
        f"amount={amount}, chain={chain_count}, payment={payment})"
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

    if hints.get("aadhaar_numbers"):
        lines.append(f"  Aadhaar Numbers Found: {', '.join(hints['aadhaar_numbers'])}")
        lines.append(f"    → Assign each to the corresponding seller/buyer party (match by proximity)")

    if hints.get("land_classification"):
        lines.append(f"  Land Classification: {hints['land_classification']}")
        lines.append(f"    → Use as property.land_classification")

    if hints.get("seller_section") and hints.get("buyer_section"):
        lines.append(f"  Seller/Buyer Role Detection:")
        lines.append(f"    → Text BEFORE 'எழுதி வாங்குபவர்' contains SELLER names/addresses")
        lines.append(f"    → Text AFTER 'எழுதி வாங்குபவர்' contains BUYER names/addresses")
        if hints.get("seller_names"):
            lines.append(f"  Seller Name Candidates: {', '.join(hints['seller_names'])}")
            lines.append(f"    → These names were found in the SELLER section — they MUST appear in seller[]")
        if hints.get("buyer_names"):
            lines.append(f"  Buyer Name Candidates: {', '.join(hints['buyer_names'])}")
            lines.append(f"    → These names were found in the BUYER section — they MUST appear in buyer[]")

    if hints.get("stamp_duty_from_text"):
        lines.append(f"  Stamp Duty (from deed text): \u20b9{hints['stamp_duty_from_text']:,}")
        lines.append(f"    \u2192 Use {hints['stamp_duty_from_text']} as financials.stamp_duty (explicitly stated in the deed)")
    elif hints.get("stamp_sheet_count") and hints["stamp_sheet_count"] >= 2:
        lines.append(f"  Stamp Sheets Detected: {hints['stamp_sheet_count']} physical stamp papers")
        if hints.get("stamp_value"):
            lines.append(f"    → Per-sheet denomination: ₹{hints['stamp_value']:,}")
        if hints.get("total_stamp_value"):
            lines.append(f"    → Total stamp duty = {hints['stamp_sheet_count']} × ₹{hints.get('stamp_value', 0):,} = ₹{hints['total_stamp_value']:,}")
            lines.append(f"    → Use {hints['total_stamp_value']} as financials.stamp_duty (NOT just one sheet)")
    elif hints.get("stamp_value"):
        lines.append(f"  Stamp Paper Denomination: ₹{hints['stamp_value']:,}")
        lines.append(f"    → If deed is printed on MULTIPLE stamp sheets, multiply by sheet count")

    if hints.get("stamp_serial_numbers"):
        lines.append(f"  Stamp Serial Numbers: {', '.join(hints['stamp_serial_numbers'])}")
        lines.append(f"    → Use these as stamp_paper.serial_numbers")
    if hints.get("stamp_vendor_name"):
        lines.append(f"  Stamp Vendor: {hints['stamp_vendor_name']}")
        lines.append(f"    → Use as stamp_paper.vendor_name")
    if hints.get("book_number"):
        lines.append(f"  Book Number: {hints['book_number']}")
        lines.append(f"    → Use as registration_details.book_number")

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
