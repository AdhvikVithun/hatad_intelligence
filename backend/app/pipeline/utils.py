"""Shared utility functions for the due diligence pipeline.

Consolidates logic used by both deterministic.py and memory_bank.py:
  - Amount parsing (lakh/crore aware)
  - Survey number normalization, parsing, and fuzzy matching
  - Village/location name normalization
  - Name normalization
"""

import re
import logging
import unicodedata
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 0. TAMIL NUMERAL NORMALIZATION
# ═══════════════════════════════════════════════════

# Tamil digits ௦–௯ (U+0BE6–U+0BEF) → ASCII 0–9
_TAMIL_DIGIT_TABLE = str.maketrans(
    "\u0BE6\u0BE7\u0BE8\u0BE9\u0BEA\u0BEB\u0BEC\u0BED\u0BEE\u0BEF",
    "0123456789",
)


def normalize_tamil_numerals(s: str) -> str:
    """Replace Tamil digits (௦–௯) with ASCII equivalents.

    Uses str.translate for single-pass O(n) performance.
    Examples:
      "௩௧௭"      → "317"
      "Survey ௩/௧" → "Survey 3/1"
      "₹௧௦,௦௦௦"   → "₹10,000"
    """
    if not s or not isinstance(s, str):
        return s or ""
    return s.translate(_TAMIL_DIGIT_TABLE)


# ── Tamil text quality detection ─────────────────────────────────────
# Tamil Unicode block: U+0B80–U+0BFF
# Valid Tamil syllable structure:
#   Independent vowel (அ–ஔ) | Consonant (க–ன) + optional vowel-sign/pulli
# Garbled text indicators:
#   - Orphan vowel signs (ா, ி, ீ …) not preceded by a consonant
#   - Other Indic scripts mixed in (Devanagari, Telugu, etc.)
#   - Excessive non-Tamil symbols within predominantly Tamil text

_TAMIL_CONSONANT_RANGE = set(range(0x0B95, 0x0BBA))   # க–ன (with gaps)
_TAMIL_VOWEL_SIGN_RANGE = set(range(0x0BBE, 0x0BD0))  # ா–ௌ + pulli ்
_TAMIL_INDEPENDENT_VOWELS = set(range(0x0B85, 0x0B95)) # அ–ஔ
_OTHER_INDIC_RANGES = [
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Odia
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
]


def _is_tamil_char(cp: int) -> bool:
    """Check if a codepoint is in the Tamil Unicode block (U+0B80–U+0BFF)."""
    return 0x0B80 <= cp <= 0x0BFF


def _is_other_indic(cp: int) -> bool:
    """Check if a codepoint belongs to a non-Tamil Indic script."""
    for lo, hi in _OTHER_INDIC_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def detect_garbled_tamil(text: str) -> tuple[bool, float, str]:
    """Detect if a string containing Tamil characters is garbled / corrupted.

    Uses Unicode structure analysis — no dictionary needed.
    Checks:
      1. Orphan vowel signs (ா, ி, ீ …) not preceded by a consonant
      2. Other Indic scripts mixed with Tamil (script confusion)
      3. High ratio of Tamil symbols with broken syllable structure

    Args:
        text: Any string that *may* contain Tamil characters.

    Returns:
        (is_garbled, quality_score, reason)
        - is_garbled: True if text appears corrupted
        - quality_score: 0.0 (completely garbled) to 1.0 (clean Tamil)
        - reason: Human-readable explanation
    """
    if not text or not isinstance(text, str):
        return False, 1.0, "no text"

    codepoints = [ord(ch) for ch in text]
    tamil_count = sum(1 for cp in codepoints if _is_tamil_char(cp))

    # Not a Tamil string — skip analysis
    if tamil_count < 3:
        return False, 1.0, "not Tamil text"

    total_chars = len(codepoints)
    orphan_vowel_signs = 0
    other_indic_count = 0
    valid_syllables = 0
    total_syllable_parts = 0

    prev_was_consonant = False
    for i, cp in enumerate(codepoints):
        if not _is_tamil_char(cp):
            if _is_other_indic(cp):
                other_indic_count += 1
            prev_was_consonant = False
            continue

        # Tamil vowel sign (dependent) without preceding consonant = orphan
        if cp in _TAMIL_VOWEL_SIGN_RANGE:
            total_syllable_parts += 1
            if not prev_was_consonant:
                orphan_vowel_signs += 1
            else:
                valid_syllables += 1
            prev_was_consonant = False
        elif cp in _TAMIL_CONSONANT_RANGE:
            total_syllable_parts += 1
            prev_was_consonant = True
        elif cp in _TAMIL_INDEPENDENT_VOWELS:
            total_syllable_parts += 1
            valid_syllables += 1
            prev_was_consonant = False
        else:
            # Tamil numerals, symbols, etc.
            prev_was_consonant = False

    # ── Scoring ──
    reasons = []

    # Orphan ratio: orphan vowel signs / total syllable parts
    orphan_ratio = orphan_vowel_signs / max(total_syllable_parts, 1)
    if orphan_ratio > 0.3:
        reasons.append(f"{orphan_vowel_signs} orphan vowel signs ({orphan_ratio:.0%})")

    # Script mixing: other Indic chars in a predominantly Tamil string
    indic_ratio = other_indic_count / max(tamil_count, 1)
    if indic_ratio > 0.1:
        reasons.append(f"{other_indic_count} non-Tamil Indic chars mixed in")

    # Overall quality: valid syllables / total syllable-forming parts
    syllable_quality = valid_syllables / max(total_syllable_parts, 1)

    # Combine into a single score
    quality_score = max(0.0, 1.0 - orphan_ratio * 1.5 - indic_ratio * 2.0)
    quality_score = round(quality_score, 3)

    is_garbled = quality_score < 0.5 or orphan_ratio > 0.3 or indic_ratio > 0.2

    reason = "; ".join(reasons) if reasons else "clean Tamil"
    return is_garbled, quality_score, reason


# ═══════════════════════════════════════════════════
# 1. AMOUNT PARSING (lakh / crore aware)
# ═══════════════════════════════════════════════════

def parse_amount(value: Any) -> Optional[float]:
    """Parse a monetary amount from various formats.

    Handles:
      - Numeric types (int, float)
      - Strings with Rs, ₹, INR prefixes
      - Lakh/crore text multipliers
      - Comma-separated numbers
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = normalize_tamil_numerals(value.strip())
    if not s:
        return None
    # Remove common prefixes/suffixes
    for prefix in ("Rs.", "Rs", "₹", "INR", "Rs "):
        s = s.replace(prefix, "")
    # Remove commas and spaces within the number
    s = s.replace(",", "").strip()
    # Handle lakh/crore text
    multiplier = 1
    s_lower = s.lower()
    if "crore" in s_lower or "cr" in s_lower:
        s = re.sub(r'(?i)\s*(crores?|crs?)\.?', '', s)
        multiplier = 10_000_000
    elif "lakh" in s_lower or "lac" in s_lower:
        s = re.sub(r'(?i)\s*(lakhs?|lacs?)\.?', '', s)
        multiplier = 100_000
    s = s.replace(" ", "").strip()
    try:
        return float(s) * multiplier
    except ValueError:
        return None


# ═══════════════════════════════════════════════════
# 2. SURVEY NUMBER NORMALIZATION & FUZZY MATCHING
# ═══════════════════════════════════════════════════

# Tamil Nadu survey number prefixes (English + Tamil)
_SURVEY_PREFIXES = re.compile(
    r'^(?:'
    r's\.?f\.?\s*no\.?|'          # S.F.No., SF No
    r'r\.?s\.?\s*no\.?|'          # R.S.No.
    r't\.?s\.?\s*no\.?|'          # T.S.No.
    r'o\.?s\.?\s*no\.?|'          # O.S.No.
    r'n\.?s\.?\s*no\.?|'          # N.S.No. (New Survey)
    r's\.?no\.?|'                 # S.No., SNo
    r'survey\s*no\.?|'            # Survey No
    r'sy\.?\s*no\.?|'             # Sy.No.
    r'புல\s*எண்\.?|'              # புல எண் (Tamil: Survey Number)
    r'நில\s*எண்\.?'               # நில எண் (Tamil: Land Number)
    r')\s*:?\s*',
    re.IGNORECASE | re.UNICODE,
)

# Extract the survey TYPE prefix (OS/TS/RS/SF/NS) before stripping it
_SURVEY_TYPE_RE = re.compile(
    r'^(s\.?f|r\.?s|t\.?s|o\.?s|n\.?s)\.?\s*no\.?',
    re.IGNORECASE,
)

# Canonical survey type labels
_SURVEY_TYPE_CANONICAL = {
    "sf": "SF",   # Sub-Field (modern)
    "rs": "RS",   # Re-Survey
    "ts": "TS",   # Town Survey
    "os": "OS",   # Old Survey
    "ns": "NS",   # New Survey
}


def extract_survey_type(sn: str) -> str:
    """Extract the survey type prefix from a survey number string.

    Returns canonical type ("SF", "RS", "TS", "OS", "NS") or empty string.

    Examples:
      "S.F.No. 311/1"  → "SF"
      "T.S.No. 45"     → "TS"
      "311/1"          → ""
    """
    if not sn or not isinstance(sn, str):
        return ""
    m = _SURVEY_TYPE_RE.match(sn.strip())
    if m:
        raw = m.group(1).replace(".", "").lower()
        return _SURVEY_TYPE_CANONICAL.get(raw, "")
    return ""


def normalize_survey_number(sn: str) -> str:
    """Normalize a survey number for comparison.

    Strips TN-specific prefixes (S.F.No., R.S.No., T.S.No., புல எண், etc.),
    normalizes separators, and removes whitespace.
    """
    if not sn or not isinstance(sn, str):
        return ""
    s = normalize_tamil_numerals(sn.strip().lower())
    # Remove known prefixes
    s = _SURVEY_PREFIXES.sub('', s)
    # Normalize separators: treat - and / equivalently
    s = s.replace("-", "/")
    # Remove extra whitespace (but keep internal structure)
    s = re.sub(r'\s+', '', s)
    return s.strip()


def parse_survey_components(normalized: str) -> tuple[str, str, str]:
    """Parse a normalized survey number into (base, subdivision, sub_subdivision).

    Examples:
      "311"     → ("311", "", "")
      "311/1"   → ("311", "1", "")
      "311/1a"  → ("311", "1", "a")
      "311/1/a" → ("311", "1", "a")
      "45/2b"   → ("45", "2", "b")
      "45/2b2"  → ("45", "2", "b2")
    """
    if not normalized:
        return ("", "", "")

    # Split on /
    parts = normalized.split("/")
    base = parts[0] if parts else ""
    sub = ""
    sub_sub = ""

    if len(parts) >= 2:
        # Second part: might be "1", "1a", "1A"
        sub_raw = parts[1]
        # Split numeric and alpha: "1a" → ("1", "a"), "1" → ("1", "")
        m = re.match(r'^(\d+)([a-z]\w*)?$', sub_raw)
        if m:
            sub = m.group(1)
            sub_sub = m.group(2) or ""
        else:
            sub = sub_raw

    if len(parts) >= 3:
        # Third part is the sub-subdivision
        sub_sub = parts[2]

    return (base, sub, sub_sub)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            ins = prev[j + 1] + 1
            dele = curr[j] + 1
            sub = prev[j] + (0 if c1 == c2 else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


def survey_numbers_match(a: str, b: str) -> tuple[bool, str]:
    """Check if two survey numbers match, with fuzzy logic.

    Returns:
      (True, "exact")        — normalized forms are identical
      (True, "subdivision")  — one is a parent of the other (311/1 contains 311/1A)
      (True, "ocr_fuzzy")    — Levenshtein distance ≤ 1 (OCR error tolerance)
      (False, "mismatch")    — genuinely different survey numbers
    """
    na = normalize_survey_number(a)
    nb = normalize_survey_number(b)

    if not na or not nb:
        return (False, "mismatch")

    # 1. Exact match
    if na == nb:
        return (True, "exact")

    # 2. Hierarchy / subdivision match
    ca = parse_survey_components(na)
    cb = parse_survey_components(nb)

    # Same base number
    if ca[0] == cb[0] and ca[0]:
        # One has subdivision, the other doesn't → parent-child
        if (ca[1] and not cb[1]) or (cb[1] and not ca[1]):
            return (True, "subdivision")
        # Same subdivision, one has sub-sub and the other doesn't
        if ca[1] == cb[1] and ca[1]:
            if (ca[2] and not cb[2]) or (cb[2] and not ca[2]):
                return (True, "subdivision")

    # 3. OCR fuzzy tolerance: Levenshtein distance ≤ 1
    #    Safety: only apply when strings are long enough (≥ 5 chars) AND
    #    the differing character is NOT a digit.  On short survey numbers
    #    like "311/1" vs "311/2" (distance = 1), a single digit change
    #    flips the actual subdivision — that's a real difference, not OCR.
    if len(na) >= 5 and len(nb) >= 5:
        dist = _levenshtein_distance(na, nb)
        if dist <= 1:
            # Find the differing character position and reject if it's a digit change
            if len(na) == len(nb):
                diff_positions = [i for i in range(len(na)) if na[i] != nb[i]]
                if diff_positions:
                    pos = diff_positions[0]
                    # If both characters at the diff position are digits,
                    # it's a real subdivision difference, not OCR noise
                    if na[pos].isdigit() and nb[pos].isdigit():
                        return (False, "mismatch")
            return (True, "ocr_fuzzy")

    return (False, "mismatch")


def split_survey_numbers(raw: str) -> list[str]:
    """Split a comma/semicolon-separated list of survey numbers.

    "311/1, 311/2, 312/3A" → ["311/1", "311/2", "312/3A"]

    Also handles dirty OCR extractions where extent data is appended:
    "760-2 50|17|1" → ["760-2"] (strips trailing non-survey junk)

    Filters out Patta soil classification codes (மண் வகையும் ரகமும்) like
    "4-3", "3-2" that are NOT survey numbers but soil type codes from a
    separate column in Tamil Patta tables.
    """
    if not raw or not isinstance(raw, str):
        return []
    raw = normalize_tamil_numerals(raw)
    parts = re.split(r'[,;]+', raw)
    results = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Skip Patta soil classification codes: single digit-dash-digit patterns
        # like "4-3", "3-2", "1-1" that are NOT survey numbers.
        # Real survey numbers with dashes have longer base numbers (e.g., "760-2").
        if re.match(r'^\d-\d$', p):
            continue
        # Extract just the survey number portion: digits followed by optional /- and more digits/letters
        # This strips trailing extent/area data like "2.0011" or "50|17|1"
        # Captures patterns like: 311, 311/1, 311/1A, 3111A, 760-2
        m = re.match(r'^(\d+[a-zA-Z]?(?:[/\-]\d+[a-zA-Z]?\d*)*)', p)
        if m:
            results.append(m.group(1))
        elif _SURVEY_PREFIXES.match(p):
            # Known survey-type prefix (S.F.No., T.S.No., etc.) — keep raw
            # so extract_survey_type() can identify the prefix downstream.
            results.append(p)
        else:
            # Fallback: OCR noise may prepend label text (e.g. "SE hag Survey 1/0:317").
            # Extract any 2+ digit survey number patterns from anywhere in the string.
            fallback = re.findall(r'\b(\d{2,}[a-zA-Z]?(?:[/\-]\d+[a-zA-Z]?\d*)*)', p)
            if fallback:
                results.extend(fallback)
            else:
                results.append(p)
    return results


def any_survey_match(surveys_a: list[str], surveys_b: list[str]) -> tuple[bool, str, str, str]:
    """Check if any survey number in list A matches any in list B.

    Returns:
      (True, match_type, matched_a, matched_b) — first match found
      (False, "mismatch", "", "")               — no match at all
    """
    for sa in surveys_a:
        for sb in surveys_b:
            matched, mtype = survey_numbers_match(sa, sb)
            if matched:
                return (True, mtype, sa, sb)
    return (False, "mismatch", "", "")


# ═══════════════════════════════════════════════════
# 3. VILLAGE / LOCATION NAME NORMALIZATION
# ═══════════════════════════════════════════════════

# ── Tamil-to-Latin approximate transliteration table ──
# Maps Tamil Unicode consonants/vowels to rough Latin equivalents.
# This is intentionally coarse — we only need enough to compare
# "வெள்ளலூர்" ≈ "vellalur" so cross-script village comparison works.
_TAMIL_VOWELS = {
    'அ': 'a', 'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ii', 'உ': 'u', 'ஊ': 'uu',
    'எ': 'e', 'ஏ': 'ee', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'oo', 'ஔ': 'au',
}

_TAMIL_CONSONANTS = {
    'க': 'k', 'ங': 'ng', 'ச': 'ch', 'ஞ': 'nj', 'ட': 't', 'ண': 'n',
    'த': 'th', 'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y', 'ர': 'r',
    'ல': 'l', 'வ': 'v', 'ழ': 'zh', 'ள': 'l', 'ற': 'r', 'ன': 'n',
    'ஜ': 'j', 'ஷ': 'sh', 'ஸ': 's', 'ஹ': 'h',
}

# Vowel sign (dependent vowel) modifiers — applied after consonant
_TAMIL_VOWEL_SIGNS = {
    '\u0bbe': 'a',   # ா
    '\u0bbf': 'i',   # ி
    '\u0bc0': 'ii',  # ீ
    '\u0bc1': 'u',   # ு
    '\u0bc2': 'uu',  # ூ
    '\u0bc6': 'e',   # ெ
    '\u0bc7': 'ee',  # ே
    '\u0bc8': 'ai',  # ை
    '\u0bca': 'o',   # ொ
    '\u0bcb': 'oo',  # ோ
    '\u0bcc': 'au',  # ௌ
}

_TAMIL_PULLI = '\u0bcd'  # ் — virama / pulli (suppresses inherent 'a')

# Vowel sign codepoint range for orphan detection (U+0BBE–U+0BCC, excludes pulli)
_VOWEL_SIGN_MIN = 0x0BBE
_VOWEL_SIGN_MAX = 0x0BCC
_CONSONANT_MIN = 0x0B95
_CONSONANT_MAX = 0x0BB9


def _fix_orphan_vowel_signs(text: str) -> str:
    """Fix OCR-garbled Tamil text where vowel signs appear before their consonant.

    Tamil OCR (especially from PDF text-layer extraction) sometimes produces
    orphan vowel signs — dependent vowels like ெ (U+0BC6) that appear BEFORE
    the consonant they should follow.  For example:

      Garbled: ெபான்அரசி  (ெ appears before ப)
      Fixed:   பொன்அரசி  (ப first, then ொ via NFC composition)

    Algorithm:
      1. Walk codepoints left to right.
      2. When a Tamil vowel sign (U+0BBE–U+0BCC) has no Tamil consonant
         immediately before it (i.e., it is "orphan"), check if the NEXT
         character is a Tamil consonant.
      3. If yes, swap them: emit consonant first, then the vowel sign.
      4. If no consonant follows, drop the orphan (unrecoverable).
      5. Apply ``unicodedata.normalize('NFC', ...)`` which composes
         adjacent vowel signs — e.g., ெ (U+0BC6) + ா (U+0BBE) → ொ (U+0BCA).

    Returns NFC-normalized text with orphan vowel signs repaired.
    """
    if not text:
        return text
    cps = list(text)
    n = len(cps)
    result: list[str] = []
    i = 0
    while i < n:
        cp = ord(cps[i])
        # Is this a Tamil vowel sign?
        if _VOWEL_SIGN_MIN <= cp <= _VOWEL_SIGN_MAX:
            # Check if previous char was a consonant (non-orphan)
            prev_is_consonant = (
                len(result) > 0 and
                _CONSONANT_MIN <= ord(result[-1]) <= _CONSONANT_MAX
            )
            # Also allow vowel sign after another vowel sign — this forms
            # compound vowels like ொ (ெ + ா), ோ (ே + ா), ௌ (ெ + ௗ).
            # NFC normalization will compose them correctly.
            prev_is_vowel_sign = (
                len(result) > 0 and
                _VOWEL_SIGN_MIN <= ord(result[-1]) <= _VOWEL_SIGN_MAX
            )
            if prev_is_consonant or prev_is_vowel_sign:
                # Normal: vowel sign follows its consonant or another vowel sign
                result.append(cps[i])
                i += 1
            else:
                # Orphan vowel sign — try to swap with next consonant
                if i + 1 < n and _CONSONANT_MIN <= ord(cps[i + 1]) <= _CONSONANT_MAX:
                    # Swap: emit consonant first, then vowel sign
                    result.append(cps[i + 1])  # consonant
                    result.append(cps[i])       # vowel sign
                    i += 2
                else:
                    # No consonant follows — drop the orphan
                    i += 1
        else:
            result.append(cps[i])
            i += 1
    # NFC composition: ெ (U+0BC6) + ா (U+0BBE) → ொ (U+0BCA)
    return unicodedata.normalize('NFC', ''.join(result))


def _has_tamil_chars(text: str) -> bool:
    """Check if text contains Tamil Unicode characters."""
    return any('\u0b80' <= ch <= '\u0bff' for ch in text)


def transliterate_tamil_to_latin(text: str) -> str:
    """Approximate transliteration of Tamil script to Latin characters.

    Produces a rough phonetic rendering suitable for fuzzy village name
    comparison — NOT a linguistically perfect transliteration.

    Pre-processes the input with ``_fix_orphan_vowel_signs()`` to handle
    OCR-garbled Tamil where vowel signs precede their consonant, then
    NFC-normalizes so compound vowel signs (ொ, ோ, ௌ) are single
    codepoints that the transliteration dict handles correctly.

    Examples:
      "வெள்ளலூர்" → "vellaluur"  (close enough to match "vellalur")
      "சென்னை"    → "chennai"
      "ெபான்அரசி" → "ponarachi"  (orphan ெ fixed → பொ)
    """
    if not text:
        return ""
    # Fix orphan vowel signs and NFC-normalize
    text = _fix_orphan_vowel_signs(text)
    result = []
    i = 0
    chars = list(text)
    n = len(chars)

    while i < n:
        ch = chars[i]

        # Independent vowel
        if ch in _TAMIL_VOWELS:
            result.append(_TAMIL_VOWELS[ch])
            i += 1
            continue

        # Consonant
        if ch in _TAMIL_CONSONANTS:
            consonant = _TAMIL_CONSONANTS[ch]
            i += 1

            # Check for vowel sign or pulli following the consonant
            if i < n:
                next_ch = chars[i]
                if next_ch == _TAMIL_PULLI:
                    # Pulli: suppress inherent 'a'
                    result.append(consonant)
                    i += 1
                elif next_ch in _TAMIL_VOWEL_SIGNS:
                    result.append(consonant + _TAMIL_VOWEL_SIGNS[next_ch])
                    i += 1
                else:
                    # Inherent 'a' vowel
                    result.append(consonant + 'a')
            else:
                result.append(consonant + 'a')
            continue

        # Non-Tamil character — pass through
        if ch.isalnum() or ch == ' ':
            result.append(ch.lower())
        i += 1

    return ''.join(result)


# Common Tamil village name suffix variants
_VILLAGE_SUFFIX_VARIANTS = {
    "pettai": ["pet", "petai", "pettai", "pete"],
    "puram": ["puram", "puram", "pur", "pore"],
    "nagar": ["nagar", "nagaram", "nager"],
    "ur": ["ur", "oor", "uru", "ooru"],
    "palayam": ["palayam", "paliam", "palayam"],
    "kuppam": ["kuppam", "kupam"],
    "mangalam": ["mangalam", "mangalm"],
}

# Build a combined reverse lookup: variant → canonical
_SUFFIX_CANONICAL: dict[str, str] = {}
for canonical, variants in _VILLAGE_SUFFIX_VARIANTS.items():
    for v in variants:
        _SUFFIX_CANONICAL[v] = canonical


def normalize_village_name(name: str) -> str:
    """Normalize a Tamil Nadu village name for fuzzy comparison.

    Handles:
      - Common transliteration variants (Chromepet/Chrompet/Chromepettai)
      - Tamil suffixes (பட்டி, நகர், புரம்)
      - Extra whitespace and punctuation
      - Administrative tags like (R), (Rural), (U), (Urban)
    """
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Remove administrative classification tags
    s = re.sub(r'\s*\(\s*(?:r|rural|u|urban|town|village|municipality|corp)\s*\)\s*', ' ', s)
    # Remove common filler
    s = re.sub(r'[,.\-\'\"()]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # Normalize known suffix variants to canonical form
    for variant, canonical in _SUFFIX_CANONICAL.items():
        if s.endswith(variant) and len(s) > len(variant):
            s = s[:-len(variant)] + canonical
            break

    # Remove double consonants that are common transliteration diffs
    # e.g. "chrompet" vs "chromepet" — we normalize by removing isolated 'e' between consonants
    # This is a light normalization; heavy matching uses Levenshtein
    return s


def _phonetic_normalize(s: str) -> str:
    """Collapse common Tamil→Latin transliteration divergences.

    Tamil transliteration produces long vowels (aa, oo, ee) and aspirated
    consonants (ch, th) that often differ from established English spellings.
    This normalizer reduces both sides to a common phonetic form.

    Examples:
      "somaiyampaalaiyam" → "somayampalayam"
      "chomaiyampalaiyanm" → "somayampalayam"
    """
    # Aspirated/alternate consonant collapses
    s = s.replace('ch', 's')    # ச can be "ch" or "s"
    s = s.replace('th', 't')    # த can be "th" or "t"
    s = s.replace('zh', 'l')    # ழ approximated as "l" in many spellings
    s = s.replace('sh', 's')    # ஷ often simplified to "s"
    # Long vowel → short vowel
    s = s.replace('aa', 'a')
    s = s.replace('oo', 'o')
    s = s.replace('ee', 'e')
    s = s.replace('ii', 'i')
    s = s.replace('uu', 'u')
    # Diphthong simplification
    s = s.replace('ai', 'a')
    s = s.replace('au', 'a')
    return s


def village_names_match(a: str, b: str) -> tuple[bool, str]:
    """Check if two village names match, with fuzzy tolerance.

    Handles cross-script comparison (Tamil vs Latin) by transliterating
    Tamil text to approximate Latin equivalents before comparing.

    Returns:
      (True, "exact")     — normalized forms are identical
      (True, "fuzzy")     — Levenshtein distance ≤ 2 (transliteration tolerance)
      (True, "cross_script") — matched after Tamil→Latin transliteration
      (False, "mismatch") — genuinely different villages
    """
    na = normalize_village_name(a)
    nb = normalize_village_name(b)

    if not na or not nb:
        return (False, "mismatch")

    if na == nb:
        return (True, "exact")

    # Allow Levenshtein ≤ 2 for transliteration variants
    dist = _levenshtein_distance(na, nb)
    if dist <= 2:
        return (True, "fuzzy")

    # Check if one is a substring of the other (e.g. "chromepet" in "chrompettai")
    if na in nb or nb in na:
        return (True, "fuzzy")

    # Cross-script comparison: if one has Tamil chars, transliterate both
    # and re-compare with fuzzy tolerance
    a_has_tamil = _has_tamil_chars(a)
    b_has_tamil = _has_tamil_chars(b)
    if a_has_tamil or b_has_tamil:
        ta = normalize_village_name(transliterate_tamil_to_latin(a)) if a_has_tamil else na
        tb = normalize_village_name(transliterate_tamil_to_latin(b)) if b_has_tamil else nb
        if ta and tb:
            if ta == tb:
                return (True, "cross_script")
            cross_dist = _levenshtein_distance(ta, tb)
            # More generous threshold for cross-script (transliteration is approximate)
            # Tamil→Latin is inherently imprecise (ச→"ch" vs English "s",
            # long vowels like "aa"/"oo" vs short "a"/"o").  We use
            # phonetic normalization + a wide Levenshtein threshold.
            ta_ph = _phonetic_normalize(ta)
            tb_ph = _phonetic_normalize(tb)
            if ta_ph == tb_ph:
                return (True, "cross_script")
            cross_dist = _levenshtein_distance(ta_ph, tb_ph)
            threshold = max(5, len(min(ta_ph, tb_ph, key=len)) // 2)
            if cross_dist <= threshold:
                return (True, "cross_script")
            if ta in tb or tb in ta:
                return (True, "cross_script")

    return (False, "mismatch")


# ═══════════════════════════════════════════════════
# 4. NAME NORMALIZATION
# ═══════════════════════════════════════════════════

_NAME_PREFIXES = (
    "mr.", "mrs.", "ms.", "sri.", "smt.", "thiru.", "thirumathi.",
    "selvi.", "selvan.", "dr.", "prof.",
    "s/o", "d/o", "w/o", "son of", "daughter of", "wife of",
)

# Regex to split multi-party name strings on delimiters
# Handles: "A and B", "A & B", "A, B", "A, B and C"
# Also handles Tamil conjunction: மற்றும் ("and")
_PARTY_SPLIT_RE = re.compile(
    r'\s*(?:,\s*(?:and|&|மற்றும்)\s+|,\s*|\s+(?:and|&|மற்றும்)\s+)\s*',
    re.IGNORECASE | re.UNICODE,
)


def split_party_names(raw: str) -> list[str]:
    """Split a multi-party name string into individual names.

    Handles delimiters: ",", "and", "&", Tamil "மற்றும்", and
    combinations like ", and".

    Filters out empty/whitespace-only fragments and avoids splitting
    around relation markers (S/o, D/o, W/o) which look like delimiters
    but are part of a single name.

    Examples:
      "Murugan and Lakshmi"           → ["Murugan", "Lakshmi"]
      "A, B, and C"                   → ["A", "B", "C"]
      "Ram S/o Krishna & Sita D/o Govind" → ["Ram S/o Krishna", "Sita D/o Govind"]
      "Single Name"                   → ["Single Name"]
    """
    if not raw or not isinstance(raw, str):
        return []
    parts = _PARTY_SPLIT_RE.split(raw.strip())
    return [p.strip() for p in parts if p and p.strip()]


def normalize_name(name: Any) -> str:
    """Normalize a person name for comparison.

    Strips honorific prefixes, relationship markers, single-letter initials,
    and extra whitespace.
    """
    if not name:
        return ""
    s = str(name).strip().lower()
    for prefix in _NAME_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # Strip single-letter initials like "r.", "a." (common in Tamil names)
    s = re.sub(r'\b[a-z]\.\s*', '', s)
    # Remove extra whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


# ── Relation-marker regexes for name splitting ──
_NAME_PREFIX_RE = re.compile(
    r'^(mr\.?|mrs\.?|ms\.?|dr\.?|sri\.?|smt\.?|thiru\.?|thirumathi\.?|'
    r'selvi\.?|shri\.?|shr\.?|kumari\.?|m/s\.?|messrs\.?)\s*',
    re.IGNORECASE
)
_RELATION_STRIP_RE = re.compile(
    r'\b(s/o|d/o|w/o|son of|daughter of|wife of|husband of|c/o|care of)\b.*$',
    re.IGNORECASE
)
_RELATION_SPLIT_RE = re.compile(
    r'\b(s/o|d/o|w/o|son of|daughter of|wife of|husband of|c/o|care of)\b',
    re.IGNORECASE,
)
_TAMIL_RELATION_RE = re.compile(
    r'(மனைவி|மகன்|மகள்|கணவர்)\s+',
    re.IGNORECASE
)


def has_tamil(s: str) -> bool:
    """Check if string contains Tamil Unicode characters."""
    return any('\u0B80' <= ch <= '\u0BFF' for ch in s)


def transliterate_for_comparison(s: str) -> str:
    """Transliterate Tamil text to Latin for cross-script name comparison.

    Also strips single-letter initials (e.g., "A." or "T.") since these
    are often added in English transliterations but absent in Tamil originals.
    """
    if has_tamil(s):
        s = transliterate_tamil_to_latin(s)
    # Strip single-letter initials like "a.", "t.", "r." at start
    s = re.sub(r'\b[a-z]\.\s*', '', s)
    return s.strip()


def split_name_parts(name: str) -> tuple[str, str]:
    """Split a name into (given_name, patronymic/father_name).

    "Murugan S/o Ramamoorthy" → ("murugan", "ramamoorthy")
    "Lakshmi"                 → ("lakshmi", "")
    "என்.துளசிராம் மனைவி சத்தயபாமா" → ("சத்தயபாமா", "என்.துளசிராம்")
    """
    if not name or not isinstance(name, str):
        return ("", "")
    s = name.strip().lower()
    # Remove honorific prefixes first
    s = _NAME_PREFIX_RE.sub("", s).strip()

    # Try Tamil relation markers first (மனைவி = wife, reversed order)
    tamil_match = _TAMIL_RELATION_RE.search(s)
    if tamil_match:
        relation = tamil_match.group(1)
        before = s[:tamil_match.start()].strip()
        after = s[tamil_match.end():].strip()
        if relation == 'மனைவி':  # wife of
            given = after if after else before
            patronymic = before if after else ""
        else:
            given = before
            patronymic = after
        given = " ".join(given.split())
        patronymic = " ".join(patronymic.split())
        return (given, patronymic)

    parts = _RELATION_SPLIT_RE.split(s, maxsplit=1)
    given = parts[0].strip()
    given = " ".join(given.split())  # collapse whitespace
    patronymic = ""
    if len(parts) >= 3:
        patronymic = parts[2].strip()
        patronymic = " ".join(patronymic.split())
    return (given, patronymic)


def base_name_similarity(n1: str, n2: str) -> float:
    """Raw similarity between two already-normalized name fragments.

    Handles cross-script comparison (Tamil vs Latin) by transliterating
    Tamil names to Latin before comparing.  Always strips single-letter
    initials (e.g., "r.", "a.") from both names.
    """
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    # Always strip single-letter initials (common in Tamil name transliterations)
    n1 = re.sub(r'\b[a-z]\.\s*', '', n1).strip()
    n2 = re.sub(r'\b[a-z]\.\s*', '', n2).strip()
    if n1 == n2:
        return 1.0

    # Fix orphan vowel signs in Tamil names before any comparison
    if has_tamil(n1):
        n1 = _fix_orphan_vowel_signs(n1)
    if has_tamil(n2):
        n2 = _fix_orphan_vowel_signs(n2)

    # Space-collapsed exact match (handles OCR space insertion/removal)
    n1_collapsed = n1.replace(' ', '')
    n2_collapsed = n2.replace(' ', '')
    if n1_collapsed == n2_collapsed:
        return 1.0

    # Cross-script detection: if one has Tamil and other doesn't (or both do),
    # transliterate both to Latin for a fair comparison
    if has_tamil(n1) or has_tamil(n2):
        n1_latin = transliterate_for_comparison(n1)
        n2_latin = transliterate_for_comparison(n2)
        if n1_latin and n2_latin:
            n1, n2 = n1_latin, n2_latin
            if n1 == n2:
                return 1.0
            # Also check space-collapsed after transliteration
            if n1.replace(' ', '') == n2.replace(' ', ''):
                return 1.0

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2) if (tokens1 and tokens2) else 0.0
    from difflib import SequenceMatcher
    seq_ratio = SequenceMatcher(None, n1, n2).ratio()
    token_score = 0.4 * jaccard + 0.6 * seq_ratio
    # Space-collapsed fallback: only when token counts differ (OCR space
    # insertion/removal).  Without this guard, single-token names like
    # "rajan" vs "rajam" would bypass the Jaccard penalty.
    if len(tokens1) != len(tokens2):
        collapsed_ratio = SequenceMatcher(
            None, n1.replace(' ', ''), n2.replace(' ', '')
        ).ratio()
        return max(token_score, collapsed_ratio)
    return token_score


def name_similarity(name1: str, name2: str) -> float:
    """Compute similarity between two names (0.0 to 1.0).

    Splits each name into (given_name, patronymic) around relation markers
    (S/o, D/o, W/o, etc.) and compares both parts separately.  This prevents
    "Murugan S/o Ramamoorthy" matching "Murugan S/o Sundaram" at 1.0.

    Scoring:
      - If both names have patronymics: 0.6 × given_sim + 0.4 × patron_sim
      - If only one has a patronymic: given_sim × 0.85 (slight penalty)
      - If neither has a patronymic: given_sim as before
    """
    g1, p1 = split_name_parts(name1)
    g2, p2 = split_name_parts(name2)

    if not g1 or not g2:
        return 0.0

    given_sim = base_name_similarity(g1, g2)

    if p1 and p2:
        patron_sim = base_name_similarity(p1, p2)
        return 0.6 * given_sim + 0.4 * patron_sim
    elif p1 or p2:
        return given_sim * 0.85
    else:
        return given_sim


def names_have_overlap(names_a: list[str], names_b: list[str],
                       threshold: float = 0.55) -> bool:
    """Check if any name in list A fuzzy-matches any name in list B.

    Uses name_similarity() for cross-script, initial-stripped comparison.
    Returns True if at least one pair exceeds the similarity threshold.
    """
    for na in names_a:
        for nb in names_b:
            if name_similarity(na, nb) >= threshold:
                return True
    return False


# ═══════════════════════════════════════════════════
# 6. AREA / EXTENT PARSING
# ═══════════════════════════════════════════════════

_AREA_TO_SQFT: dict[str, float] = {
    "sq.ft": 1.0, "sqft": 1.0, "sq ft": 1.0, "square feet": 1.0,
    "sq.m": 10.764, "sqm": 10.764, "sq m": 10.764, "square meters": 10.764,
    "cents": 435.6, "cent": 435.6,
    "acres": 43560.0, "acre": 43560.0,
    "hectares": 107639.0, "hectare": 107639.0,
    "ha": 107639.0,
    "ground": 2400.0, "grounds": 2400.0,
    "kuzhi": 144.0, "kuzhis": 144.0,
    "ares": 1076.39, "are": 1076.39,
    "guntha": 1089.0, "gunthas": 1089.0,
}

_AREA_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(' + '|'.join(
        re.escape(k) for k in sorted(_AREA_TO_SQFT.keys(), key=len, reverse=True)
    ) + ')',
    re.IGNORECASE
)


def parse_area_to_sqft(text: str) -> float | None:
    """Parse an area string and convert to square feet.

    Handles compound extents like "2 acres 50 cents" by summing all
    matched number+unit pairs.
    """
    if not text or not isinstance(text, str):
        return None
    total = 0.0
    found_any = False
    for match in _AREA_PATTERN.finditer(text):
        value = float(match.group(1))
        unit = match.group(2).lower()
        factor = _AREA_TO_SQFT.get(unit)
        if factor:
            total += value * factor
            found_any = True
    if found_any:
        return total
    # Try parsing bare number (assume sq.ft if no unit)
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


# ═══════════════════════════════════════════════════
# 7. TRANSACTION TYPE CATEGORIES
# ═══════════════════════════════════════════════════

# Types that transfer title (ownership changes)
TITLE_TRANSFER_TYPES = frozenset({
    "sale", "gift", "partition", "will", "exchange",
    "settlement", "relinquishment", "adoption",
    # Also match document-type variants
    "sale deed", "gift deed", "partition deed", "release deed",
})

# Types that create encumbrances (burden on title, no ownership change)
ENCUMBRANCE_TYPES = frozenset({
    "mortgage", "lease", "agreement", "surety",
})

# Administrative / procedural (no ownership or encumbrance change)
ADMINISTRATIVE_TYPES = frozenset({
    "rectification", "cancellation", "release", "reconveyance",
    "declaration", "receipt",
})

# Judicial / court-driven
JUDICIAL_TYPES = frozenset({
    "court_order",
})

# Other / structural
STRUCTURAL_TYPES = frozenset({
    "power_of_attorney", "trust", "amalgamation", "other",
})

# Combined: all types that should appear in ownership chain analysis
CHAIN_RELEVANT_TYPES = TITLE_TRANSFER_TYPES | {"release"}


def is_title_transfer(txn_type: str) -> bool:
    """Check if a transaction type transfers property title."""
    return (txn_type or "").strip().lower() in TITLE_TRANSFER_TYPES
