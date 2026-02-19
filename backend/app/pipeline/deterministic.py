"""Deterministic checks engine — Python-based verification that never uses LLMs.

These checks handle logic that LLMs are unreliable at:
  - Date arithmetic and temporal ordering
  - Financial calculations (stamp duty, area conversions)
  - Fuzzy name matching across documents
  - Survey number normalization and matching

The engine runs AFTER LLM verification and produces additional checks
that are appended to the verification results, or overrides LLM checks
where the deterministic result is more reliable.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

from app.pipeline.utils import (
    parse_amount as _parse_amount,
    normalize_survey_number as _normalize_survey_number,
    survey_numbers_match,
    split_survey_numbers,
    any_survey_match,
    normalize_village_name,
    village_names_match,
    split_party_names,
    extract_survey_type,
    transliterate_tamil_to_latin,
    has_tamil as _has_tamil_utils,
    transliterate_for_comparison as _transliterate_for_comparison_utils,
    split_name_parts as _split_name_parts_utils,
    base_name_similarity as _base_name_similarity_utils,
    name_similarity as _name_similarity_utils,
    parse_area_to_sqft as _parse_area_to_sqft_utils,
    detect_garbled_tamil as _detect_garbled_tamil,
    TITLE_TRANSFER_TYPES,
    CHAIN_RELEVANT_TYPES,
)
from app.config import TRACE_ENABLED

logger = logging.getLogger(__name__)


def _trace(msg: str):
    """Emit a trace-level debug message when HATAD_TRACE is enabled."""
    if TRACE_ENABLED:
        logger.debug(f"[TRACE] {msg}")


# ═══════════════════════════════════════════════════
# 1. TEMPORAL LOGIC ENGINE
# ═══════════════════════════════════════════════════

_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%Y/%m/%d", "%m/%d/%Y", "%d %b %Y", "%d %B %Y",
    "%B %d, %Y", "%b %d, %Y",
]


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse a date string trying multiple formats."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    # Try extracting yyyy from string
    match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if match:
        try:
            return datetime(int(match.group()), 1, 1)
        except ValueError:
            pass
    return None


def check_ec_period_coverage(extracted_data: dict) -> list[dict]:
    """Check that EC period covers relevant dates and is recent enough."""
    checks = []
    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "EC":
            continue

        d = data.get("data", {})
        period_from = _parse_date(d.get("period_from", ""))
        period_to = _parse_date(d.get("period_to", ""))
        now = datetime.now()
        _trace(f"EC_PERIOD [{filename}] from={d.get('period_from')} to={d.get('period_to')} parsed_from={period_from} parsed_to={period_to}")

        if period_from and period_to:
            # Check 1: EC period tiering (TN standard = 13+ years)
            span_days = (period_to - period_from).days
            span_years = span_days / 365.25
            if span_days < 0:
                checks.append(_make_check(
                    "DET_EC_PERIOD_INVALID", "EC Period Invalid",
                    "CRITICAL", "FAIL",
                    f"EC period end ({d.get('period_to')}) is before start ({d.get('period_from')}). "
                    f"This indicates a data error or fraudulent EC.",
                    "Obtain a corrected EC from the Sub-Registrar's office.",
                    f"[{filename}] period_from={d.get('period_from')}, period_to={d.get('period_to')}",
                ))
            elif span_years < 1:
                # Less than 1 year — critically insufficient
                checks.append(_make_check(
                    "DET_EC_SHORT_PERIOD", "EC Period Critically Short",
                    "CRITICAL", "FAIL",
                    f"EC covers only {span_days} days (~{span_years:.1f} years). "
                    f"Standard due diligence requires EC for at least 13 years. "
                    f"This period is far too short to reveal encumbrances or title defects.",
                    "Obtain a comprehensive EC covering 13+ years from the same SRO.",
                    f"[{filename}] span={span_days} days ({span_years:.1f} years)",
                ))
            elif span_years < 10:
                # 1-10 years — insufficient for TN due diligence
                checks.append(_make_check(
                    "DET_EC_SHORT_PERIOD", "EC Period Insufficient",
                    "HIGH", "FAIL",
                    f"EC covers {span_years:.1f} years ({d.get('period_from')} to {d.get('period_to')}). "
                    f"Standard due diligence requires EC for at least 13 years. "
                    f"Older encumbrances or title claims may be hidden.",
                    "Obtain a supplementary EC covering 13+ years from the same SRO.",
                    f"[{filename}] span={span_days} days ({span_years:.1f} years)",
                ))
            elif span_years < 13:
                # 10-13 years — borderline, advisory warning
                checks.append(_make_check(
                    "DET_EC_SHORT_PERIOD", "EC Period Below Recommended 13 Years",
                    "MEDIUM", "WARNING",
                    f"EC covers {span_years:.1f} years ({d.get('period_from')} to {d.get('period_to')}). "
                    f"TN standard due diligence recommends EC for 13+ years. "
                    f"Consider obtaining a longer-period EC for complete coverage.",
                    "Request a supplementary EC from the SRO to cover the full 13-year period.",
                    f"[{filename}] span={span_days} days ({span_years:.1f} years)",
                ))
            # 13+ years → no check emitted (PASS by absence)

            _trace(f"EC_PERIOD [{filename}] span={span_days}d, threshold=365d")

            # Check 2: EC recency — period_to should be within last 90 days
            ec_age_days = (now - period_to).days
            _trace(f"EC_RECENCY [{filename}] age={ec_age_days}d, threshold=90d")
            if ec_age_days > 90:
                checks.append(_make_check(
                    "DET_EC_STALE", "EC Not Recent",
                    "HIGH", "WARNING",
                    f"EC is {ec_age_days} days old (period ends {d.get('period_to')}). "
                    f"For transaction purposes, EC should be less than 30 days old.",
                    "Obtain a fresh EC dated within 30 days of the proposed transaction.",
                    f"[{filename}] ec_age={ec_age_days} days, period_to={d.get('period_to')}",
                ))

        # Check 3: Transaction chronological order
        transactions = d.get("transactions", [])
        if len(transactions) >= 2:
            sorted_issues = []
            prev_date = None
            for txn in transactions:
                txn_date = _parse_date(txn.get("date", ""))
                if txn_date and prev_date and txn_date < prev_date:
                    sorted_issues.append(
                        f"Row {txn.get('row_number', '?')}: {txn.get('date')} is before "
                        f"preceding transaction date"
                    )
                if txn_date:
                    prev_date = txn_date

            if sorted_issues:
                checks.append(_make_check(
                    "DET_EC_CHRONO_ORDER", "Transaction Date Order Issue",
                    "MEDIUM", "WARNING",
                    f"EC transactions are not in chronological order. "
                    f"{len(sorted_issues)} ordering issue(s) found.",
                    "Verify with the SRO that the EC is complete and correctly ordered.",
                    f"[{filename}] " + "; ".join(sorted_issues[:3]),
                ))

    return checks


def check_registration_within_ec(extracted_data: dict) -> list[dict]:
    """For sale deeds, verify registration date falls within EC period."""
    checks = []
    ec_periods = []
    sale_dates = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})
        if doc_type == "EC":
            period_from = _parse_date(d.get("period_from", ""))
            period_to = _parse_date(d.get("period_to", ""))
            if period_from and period_to:
                ec_periods.append((filename, period_from, period_to))
        elif doc_type == "SALE_DEED":
            reg_date = _parse_date(d.get("registration_date", ""))
            if reg_date:
                sale_dates.append((filename, reg_date, d.get("registration_date", "")))

    for sale_fn, reg_date, reg_str in sale_dates:
        for ec_fn, ec_from, ec_to in ec_periods:
            inside = ec_from <= reg_date <= ec_to
            _trace(f"REG_IN_EC [{sale_fn}] reg={reg_str} vs [{ec_fn}] {ec_from.strftime('%d-%m-%Y')}..{ec_to.strftime('%d-%m-%Y')} inside={inside}")
            if not inside:
                checks.append(_make_check(
                    "DET_REG_OUTSIDE_EC", "Registration Date Outside EC Period",
                    "HIGH", "FAIL",
                    f"Sale deed registration date ({reg_str}) falls outside EC period "
                    f"({ec_from.strftime('%d-%m-%Y')} to {ec_to.strftime('%d-%m-%Y')}). "
                    f"The EC does not cover the transaction date.",
                    "Obtain an EC that covers the registration date of the sale deed.",
                    f"[{sale_fn}] reg_date={reg_str}, [{ec_fn}] EC period={ec_from.strftime('%d-%m-%Y')} to {ec_to.strftime('%d-%m-%Y')}",
                ))

    return checks


def check_limitation_period(extracted_data: dict) -> list[dict]:
    """Check if any transactions are beyond the 12-year limitation period."""
    checks = []
    now = datetime.now()
    twelve_years = timedelta(days=12 * 365)

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "EC":
            continue

        d = data.get("data", {})
        stale_txns = []
        for txn in d.get("transactions", []):
            txn_date = _parse_date(txn.get("date", ""))
            if txn_date and (now - txn_date) > twelve_years:
                ttype = txn.get("transaction_type", "unknown")
                if ttype.lower() in CHAIN_RELEVANT_TYPES:
                    stale_txns.append(
                        f"Row {txn.get('row_number', '?')}: {ttype} on {txn.get('date')} "
                        f"({(now - txn_date).days // 365} years ago)"
                    )

        if stale_txns:
            checks.append(_make_check(
                "DET_LIMITATION_PERIOD", "Transactions Beyond Limitation Period",
                "MEDIUM", "INFO",
                f"{len(stale_txns)} transaction(s) in the EC are beyond the 12-year "
                f"limitation period under the Limitation Act. While these may be valid, "
                f"claims arising from them may be time-barred.",
                "For old transactions, verify that possession has been continuous and undisputed.",
                f"[{filename}] " + "; ".join(stale_txns[:5]),
            ))

    return checks


# ═══════════════════════════════════════════════════
# 2. STAMP DUTY CALCULATOR (Tamil Nadu)
# ═══════════════════════════════════════════════════

# Tamil Nadu stamp duty rates (as of 2024)
_TN_STAMP_DUTY_RATE = 0.07         # 7% of property value
_TN_REGISTRATION_FEE_RATE = 0.04   # 4% (capped at ₹4 lakhs for residential)
_TN_SURCHARGE_RATE = 0.01          # 1% transfer duty
# Total: ~12% for most transactions


# _parse_amount is now imported from utils.py (lakh/crore aware)


def check_stamp_duty(extracted_data: dict) -> list[dict]:
    """Verify stamp duty paid matches expected calculation."""
    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "SALE_DEED":
            continue

        d = data.get("data", {})
        financials = d.get("financials", {})
        if not isinstance(financials, dict):
            _trace(f"STAMP_DUTY [{filename}] SKIP: financials is not a dict")
            continue

        consideration = _parse_amount(financials.get("consideration_amount"))
        guideline_val = _parse_amount(financials.get("guideline_value"))
        stamp_duty_paid = _parse_amount(financials.get("stamp_duty"))
        _trace(f"STAMP_DUTY [{filename}] consideration={consideration} guideline={guideline_val} stamp_paid={stamp_duty_paid}")

        if not consideration:
            _trace(f"STAMP_DUTY [{filename}] SKIP: no consideration amount")
            continue

        # Stamp duty is calculated on the higher of consideration or guideline value
        assessable_value = max(consideration, guideline_val or 0)

        if assessable_value and stamp_duty_paid:
            expected_stamp = assessable_value * _TN_STAMP_DUTY_RATE
            # Allow 5% tolerance for rounding
            if stamp_duty_paid < expected_stamp * 0.95:
                shortfall = expected_stamp - stamp_duty_paid
                shortfall_pct = (shortfall / expected_stamp) * 100
                checks.append(_make_check(
                    "DET_STAMP_DUTY_SHORT", "Stamp Duty Shortfall Detected",
                    "HIGH", "FAIL",
                    f"Stamp duty paid (₹{stamp_duty_paid:,.0f}) appears insufficient. "
                    f"Expected ₹{expected_stamp:,.0f} (7% of ₹{assessable_value:,.0f}). "
                    f"Shortfall: ₹{shortfall:,.0f} ({shortfall_pct:.1f}%).",
                    "Verify stamp duty calculation with the registrar. Shortfall may result "
                    "in penalties or document being impounded under TN Stamp Act.",
                    f"[{filename}] consideration=₹{consideration:,.0f}, "
                    f"guideline=₹{guideline_val:,.0f}" if guideline_val else f"[{filename}] consideration=₹{consideration:,.0f}",
                ))

        # Check consideration vs guideline value
        if consideration and guideline_val and consideration < guideline_val * 0.8:
            underpay_pct = ((guideline_val - consideration) / guideline_val) * 100
            checks.append(_make_check(
                "DET_UNDERVALUATION", "Possible Undervaluation",
                "HIGH", "WARNING",
                f"Sale consideration (₹{consideration:,.0f}) is {underpay_pct:.0f}% below "
                f"guideline value (₹{guideline_val:,.0f}). This may indicate black money "
                f"component or incorrect guideline value lookup.",
                "Verify the current guideline value for this survey number and compare "
                "with actual transaction consideration.",
                f"[{filename}] consideration=₹{consideration:,.0f}, guideline=₹{guideline_val:,.0f}",
            ))

    return checks


# ═══════════════════════════════════════════════════
# 2b. PLAUSIBILITY RANGE CHECKS
# ═══════════════════════════════════════════════════

# Realistic bounds for Tamil Nadu land transactions
_PLAUSIBILITY_BOUNDS = {
    # field_key: (min, max, label)
    "consideration_amount": (10_000, 5_000_000_000, "Consideration Amount"),     # ₹10K – ₹500 Cr
    "guideline_value":      (10_000, 5_000_000_000, "Guideline Value"),
    "stamp_duty":           (100,    350_000_000,   "Stamp Duty"),               # ₹100 – ₹35 Cr
    "registration_fee":     (100,    400_000,       "Registration Fee"),          # TN cap ₹4 lakhs
}
_EXTENT_MIN_SQFT = 1.0
_EXTENT_MAX_SQFT = 10_000_000.0  # ~230 acres — TN ceiling limit


def check_plausibility_ranges(extracted_data: dict) -> list[dict]:
    """Flag financial and area values that fall outside realistic TN bounds."""
    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        # ── Financial plausibility (Sale Deed only) ──
        if doc_type == "SALE_DEED":
            financials = d.get("financials", {})
            if isinstance(financials, dict):
                for field_key, (lo, hi, label) in _PLAUSIBILITY_BOUNDS.items():
                    raw = financials.get(field_key)
                    amt = _parse_amount(raw)
                    if amt is not None and amt > 0:
                        if amt < lo:
                            checks.append(_make_check(
                                "DET_IMPLAUSIBLE_LOW", f"Suspiciously Low {label}",
                                "MEDIUM", "WARNING",
                                f"{label} of ₹{amt:,.0f} is unusually low for a TN property "
                                f"transaction (expected ≥ ₹{lo:,}).",
                                "Verify the amount with the registered document. This may "
                                "indicate an extraction error or an unusual transaction.",
                                f"[{filename}] {field_key}=₹{amt:,.0f}",
                            ))
                        elif amt > hi:
                            checks.append(_make_check(
                                "DET_IMPLAUSIBLE_HIGH", f"Suspiciously High {label}",
                                "MEDIUM", "WARNING",
                                f"{label} of ₹{amt:,.0f} exceeds the expected ceiling for a TN "
                                f"land transaction (expected ≤ ₹{hi:,}). This may indicate a "
                                f"data extraction error (extra zeros, misplaced decimal).",
                                "Verify the amount against the original document.",
                                f"[{filename}] {field_key}=₹{amt:,.0f}",
                            ))

        # ── Extent plausibility (Sale Deed, Patta, Chitta) ──
        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            extent_text = prop.get("extent") if isinstance(prop, dict) else None
        elif doc_type in ("PATTA", "CHITTA"):
            extent_text = d.get("total_extent") or d.get("extent")
        else:
            extent_text = None

        if extent_text:
            sqft = _parse_area_to_sqft(str(extent_text))
            if sqft is not None and sqft > 0:
                if sqft < _EXTENT_MIN_SQFT:
                    checks.append(_make_check(
                        "DET_IMPLAUSIBLE_EXTENT", "Suspiciously Small Property Extent",
                        "MEDIUM", "WARNING",
                        f"Extracted extent of {extent_text} ({sqft:.1f} sqft) is implausibly small. "
                        f"This likely indicates an extraction or unit conversion error.",
                        "Verify the extent and units in the original document.",
                        f"[{filename}] extent='{extent_text}' → {sqft:.1f} sqft",
                    ))
                elif sqft > _EXTENT_MAX_SQFT:
                    checks.append(_make_check(
                        "DET_IMPLAUSIBLE_EXTENT", "Suspiciously Large Property Extent",
                        "MEDIUM", "WARNING",
                        f"Extracted extent of {extent_text} ({sqft:,.0f} sqft / {sqft/43560:.1f} acres) "
                        f"exceeds the TN ceiling limit for single-person holdings (~230 acres). "
                        f"This may indicate an extraction error.",
                        "Verify the extent against the original document.",
                        f"[{filename}] extent='{extent_text}' → {sqft:,.0f} sqft",
                    ))

    return checks


# ═══════════════════════════════════════════════════
# 3. AREA UNIT CONVERTER & CROSS-CHECK
# ═══════════════════════════════════════════════════

# Tamil Nadu area conversion factors (to square feet as base unit)
_AREA_TO_SQFT = {
    "sqft": 1.0,
    "sq.ft": 1.0,
    "sq ft": 1.0,
    "square feet": 1.0,
    "square foot": 1.0,
    "sqm": 10.764,
    "sq.m": 10.764,
    "sq m": 10.764,
    "square meter": 10.764,
    "square metre": 10.764,
    "cent": 435.6,
    "cents": 435.6,
    "acre": 43560.0,
    "acres": 43560.0,
    "hectare": 107639.0,
    "hectares": 107639.0,
    "ha": 107639.0,
    "ground": 2400.0,
    "grounds": 2400.0,
    "kuzhi": 144.0,  # 1 kuzhi = 144 sq.ft in TN
    "kuzhis": 144.0,
    "ares": 1076.39,
    "are": 1076.39,
    "guntha": 1089.0,
    "gunthas": 1089.0,
}

# Regex pattern to match area values: "2.5 acres", "1500 sq.ft", "10 cents"
_AREA_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*(' + '|'.join(re.escape(k) for k in sorted(_AREA_TO_SQFT.keys(), key=len, reverse=True)) + ')',
    re.IGNORECASE
)


def _parse_area_to_sqft(text: str) -> Optional[float]:
    """Parse an area string and convert to square feet.

    Handles compound extents like \"2 acres 50 cents\" by summing all
    matched number+unit pairs via finditer (not just the first match).
    """
    return _parse_area_to_sqft_utils(text)


def check_area_consistency(extracted_data: dict) -> list[dict]:
    """Cross-check property extent across all documents after unit normalization."""
    checks = []
    area_records: list[tuple[str, str, float, str]] = []  # (filename, doc_type, sqft, original_text)

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        extent_text = None
        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                extent_text = prop.get("extent")
        elif doc_type in ("PATTA", "CHITTA"):
            extent_text = d.get("total_extent") or d.get("extent")
        elif doc_type == "EC":
            # EC may have extent per transaction; use from property description
            extent_text = d.get("property_description", "")

        if extent_text:
            sqft = _parse_area_to_sqft(str(extent_text))
            if sqft and sqft > 0:
                area_records.append((filename, doc_type, sqft, str(extent_text)))

    _trace(f"AREA_CHECK records={[(fn,dt,f'{sqft:.0f}sqft') for fn,dt,sqft,_ in area_records]}")
    # Compare all pairs for mismatches (>10% tolerance)
    if len(area_records) >= 2:
        mismatches = []
        for i in range(len(area_records)):
            for j in range(i + 1, len(area_records)):
                fn1, dt1, sqft1, txt1 = area_records[i]
                fn2, dt2, sqft2, txt2 = area_records[j]
                if sqft1 > 0 and sqft2 > 0:
                    diff_pct = abs(sqft1 - sqft2) / max(sqft1, sqft2) * 100
                    if diff_pct > 10:
                        mismatches.append(
                            f"{dt1}[{fn1}]: {txt1} ({sqft1:.0f} sqft) vs "
                            f"{dt2}[{fn2}]: {txt2} ({sqft2:.0f} sqft) — {diff_pct:.0f}% difference"
                        )
        if mismatches:
            checks.append(_make_check(
                "DET_AREA_MISMATCH", "Property Area Mismatch Across Documents",
                "HIGH", "FAIL",
                f"Property extent differs by >10% across documents. "
                f"{len(mismatches)} mismatch(es) found after unit normalization.",
                "Verify the actual property extent through physical survey. Area discrepancies "
                "may indicate encroachment, sub-division, or data errors.",
                " | ".join(mismatches[:3]),
            ))

    return checks


# ═══════════════════════════════════════════════════
# 4. FUZZY NAME MATCHER
# ═══════════════════════════════════════════════════

_NAME_PREFIXES = re.compile(
    r'^(mr\.?|mrs\.?|ms\.?|dr\.?|sri\.?|smt\.?|thiru\.?|thirumathi\.?|'
    r'selvi\.?|shri\.?|shr\.?|kumari\.?|m/s\.?|messrs\.?)\s*',
    re.IGNORECASE
)
_RELATION_PATTERNS = re.compile(
    r'\b(s/o|d/o|w/o|son of|daughter of|wife of|husband of|c/o|care of)\b.*$',
    re.IGNORECASE
)

# Tamil relation markers for name splitting
_TAMIL_RELATION_PATTERNS = re.compile(
    r'(மனைவி|மகன்|மகள்|கணவர்)\s+',
    re.IGNORECASE
)


# Regex to split a name into (given_name, relation, patronymic)
_RELATION_SPLIT = re.compile(
    r'\b(s/o|d/o|w/o|son of|daughter of|wife of|husband of|c/o|care of)\b',
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip titles, relation suffixes."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    # Remove prefixes
    s = _NAME_PREFIXES.sub("", s)
    # Remove relation suffixes (s/o, d/o, w/o and everything after)
    s = _RELATION_PATTERNS.sub("", s)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s.strip()


def _split_name_parts(name: str) -> tuple[str, str]:
    """Split a name into (given_name, patronymic/father_name).

    "Murugan S/o Ramamoorthy" → ("murugan", "ramamoorthy")
    "Lakshmi"                 → ("lakshmi", "")
    "என்.துளசிராம் மனைவி சத்தயபாமா" → ("சத்தயபாமா", "என்.துளசிராம்")
    """
    return _split_name_parts_utils(name)


def _has_tamil(s: str) -> bool:
    """Check if string contains Tamil Unicode characters."""
    return _has_tamil_utils(s)


def _transliterate_for_comparison(s: str) -> str:
    """Transliterate Tamil text to Latin for cross-script name comparison.

    Also strips single-letter initials (e.g., "A." or "T.") since these
    are often added in English transliterations but absent in Tamil originals.
    """
    return _transliterate_for_comparison_utils(s)


def _base_name_similarity(n1: str, n2: str) -> float:
    """Raw similarity between two already-normalized name fragments.

    Handles cross-script comparison (Tamil vs Latin) by transliterating
    Tamil names to Latin before comparing.
    """
    # Delegate to shared implementation but preserve trace logging
    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    # Cross-script detection with trace logging
    if _has_tamil(n1) or _has_tamil(n2):
        n1_latin = _transliterate_for_comparison(n1)
        n2_latin = _transliterate_for_comparison(n2)
        if n1_latin and n2_latin:
            _trace(f"CROSS_SCRIPT name match: '{n1}' → '{n1_latin}' vs '{n2}' → '{n2_latin}'")

    return _base_name_similarity_utils(n1, n2)


def _name_similarity(name1: str, name2: str) -> float:
    """Compute similarity between two names (0.0 to 1.0).

    Splits each name into (given_name, patronymic) around relation markers
    (S/o, D/o, W/o, etc.) and compares both parts separately.  This prevents
    "Murugan S/o Ramamoorthy" matching "Murugan S/o Sundaram" at 1.0.

    Scoring:
      - If both names have patronymics: 0.6 × given_sim + 0.4 × patron_sim
      - If only one has a patronymic: given_sim × 0.85 (slight penalty)
      - If neither has a patronymic: given_sim as before
    """
    return _name_similarity_utils(name1, name2)


def _check_names_via_identity(extracted_data: dict, resolver) -> list[dict]:
    """Identity-cluster-based party name verification.

    Instead of pairwise string matching, queries the IdentityResolver
    to check whether buyers/claimants share an identity cluster with
    patta owners.  Produces rich evidence reports.
    """
    checks: list[dict] = []

    # Check 1: Buyer ↔ Patta owner
    has_buyers = any(
        entry.get("document_type") == "SALE_DEED"
        and isinstance(entry.get("data", {}).get("buyer"), list)
        and entry["data"]["buyer"]
        for entry in extracted_data.values()
    )
    has_patta = any(
        entry.get("document_type") in ("PATTA", "CHITTA")
        and isinstance(entry.get("data", {}).get("owner_names"), list)
        and entry["data"]["owner_names"]
        for entry in extracted_data.values()
    )

    if has_buyers and has_patta:
        same, conf, evidence = resolver.roles_share_identity("buyer", "patta_owner")
        if same:
            checks.append(_make_check(
                "DET_BUYER_PATTA_MATCH", "Buyer-Patta Owner Identity Verified",
                "HIGH", "PASS",
                evidence,
                "No action needed — buyer and patta owner are the same person.",
                evidence,
            ))
        else:
            checks.append(_make_check(
                "DET_BUYER_PATTA_MISMATCH", "Buyer-Patta Owner Identity Mismatch",
                "HIGH", "WARNING",
                evidence,
                "Verify that patta has been transferred to the buyer's name. If not, "
                "ensure patta transfer is a condition of the transaction.",
                evidence,
            ))

    # Check 2: EC claimant ↔ Patta owner (when no sale deed buyers)
    if not has_buyers and has_patta:
        has_ec = any(
            entry.get("document_type") == "EC"
            for entry in extracted_data.values()
        )
        if has_ec:
            same, conf, evidence = resolver.roles_share_identity("ec_claimant", "patta_owner")
            if same:
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MATCH", "EC Claimant-Patta Owner Identity Verified",
                    "HIGH", "PASS",
                    evidence,
                    "No action needed — EC claimant and patta owner are the same person.",
                    evidence,
                ))
            else:
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MISMATCH", "EC Buyer-Patta Owner Identity Mismatch",
                    "HIGH", "WARNING",
                    evidence,
                    "Verify that patta has been transferred to the EC buyer's name.",
                    evidence,
                ))

    # Check 3: Last EC claimant ↔ Sale deed seller (chain continuity)
    has_sellers = any(
        entry.get("document_type") == "SALE_DEED"
        and isinstance(entry.get("data", {}).get("seller"), list)
        and entry["data"]["seller"]
        for entry in extracted_data.values()
    )
    has_ec_claimants = any(
        entry.get("document_type") == "EC"
        for entry in extracted_data.values()
    )

    if has_ec_claimants and has_sellers:
        same, conf, evidence = resolver.roles_share_identity("ec_claimant", "seller")
        if not same:
            checks.append(_make_check(
                "DET_CHAIN_NAME_GAP", "Chain of Title Name Gap (Identity)",
                "HIGH", "WARNING",
                evidence,
                "Verify that the sale deed seller acquired title through a valid "
                "transaction reflected in the EC.",
                evidence,
            ))

    return checks


def check_party_name_consistency(
    extracted_data: dict, *, identity_resolver=None
) -> list[dict]:
    """Cross-check party names across documents.

    When *identity_resolver* is provided (an :class:`IdentityResolver`
    that has already been resolved), uses multi-source identity clusters
    for evidence-based corroboration.  Falls back to pairwise fuzzy
    matching when the resolver is not available.
    """
    # ── New path: identity-cluster-based verification ──
    if identity_resolver is not None and identity_resolver._resolved:
        return _check_names_via_identity(extracted_data, identity_resolver)

    # ── Legacy path: pairwise fuzzy matching ──
    checks = []

    # Collect all parties by role
    sellers: list[tuple[str, str]] = []     # (name, filename)
    buyers: list[tuple[str, str]] = []      # (name, filename)
    patta_owners: list[tuple[str, str]] = []  # (name, filename)
    ec_parties: list[tuple[str, str, str]] = []  # (name, role, filename)

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        if doc_type == "SALE_DEED":
            for party in (d.get("seller") or []):
                if isinstance(party, dict):
                    sellers.append((party.get("name", ""), filename))
                elif isinstance(party, str):
                    sellers.append((party, filename))
            for party in (d.get("buyer") or []):
                if isinstance(party, dict):
                    buyers.append((party.get("name", ""), filename))
                elif isinstance(party, str):
                    buyers.append((party, filename))

        elif doc_type in ("PATTA", "CHITTA"):
            for owner in (d.get("owner_names") or []):
                if isinstance(owner, dict):
                    patta_owners.append((owner.get("name", ""), filename))
                elif isinstance(owner, str):
                    patta_owners.append((owner, filename))

        elif doc_type == "EC":
            for txn in d.get("transactions", []):
                seller = txn.get("seller_or_executant", "")
                buyer = txn.get("buyer_or_claimant", "")
                # Split multi-party strings: "A and B" → ["A", "B"]
                if seller:
                    for name in split_party_names(seller):
                        ec_parties.append((name, "executant", filename))
                if buyer:
                    for name in split_party_names(buyer):
                        ec_parties.append((name, "claimant", filename))

    # Check: Sale deed buyer should match patta owner
    if buyers and patta_owners:
        for buyer_name, buyer_fn in buyers:
            best_match = 0.0
            best_owner = ""
            for owner_name, _ in patta_owners:
                sim = _name_similarity(buyer_name, owner_name)
                if sim > best_match:
                    best_match = sim
                    best_owner = owner_name

            _trace(f"PARTY [{buyer_fn}] buyer='{buyer_name}' best_patta_owner='{best_owner}' sim={best_match:.2f}")
            if best_match < 0.5 and buyer_name.strip() and best_owner.strip():
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MISMATCH", "Buyer-Patta Owner Name Mismatch",
                    "HIGH", "WARNING",
                    f"Sale deed buyer '{buyer_name}' does not closely match any patta owner "
                    f"(closest: '{best_owner}', similarity: {best_match:.0%}). This may indicate "
                    f"that patta transfer has not been completed after the sale.",
                    "Verify that patta has been transferred to the buyer's name. If not, "
                    "ensure patta transfer is a condition of the transaction.",
                    f"[{buyer_fn}] buyer='{buyer_name}', patta_owner='{best_owner}' (match: {best_match:.0%})",
                ))

    # Check: EC claimant/buyer should match patta owner (when no sale deed)
    if ec_parties and patta_owners and not buyers:
        ec_claimants = [(name, fn) for name, role, fn in ec_parties if role == "claimant"]
        for claimant_name, claimant_fn in ec_claimants:
            best_match = 0.0
            best_owner = ""
            for owner_name, _ in patta_owners:
                sim = _name_similarity(claimant_name, owner_name)
                if sim > best_match:
                    best_match = sim
                    best_owner = owner_name

            _trace(f"PARTY [{claimant_fn}] ec_claimant='{claimant_name}' best_patta_owner='{best_owner}' sim={best_match:.2f}")
            if best_match < 0.5 and claimant_name.strip() and best_owner.strip():
                # Skip warning for garbled Tamil names — OCR corruption
                # can't be resolved by string matching; the vision
                # re-check pipeline is the correct fix for these.
                is_garbled, _quality, _reason = _detect_garbled_tamil(claimant_name)
                if is_garbled:
                    _trace(f"PARTY [{claimant_fn}] skipping mismatch warning — claimant name appears garbled: {_reason}")
                    continue
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MISMATCH", "EC Buyer-Patta Owner Name Mismatch",
                    "HIGH", "WARNING",
                    f"EC buyer/claimant '{claimant_name}' does not closely match any patta owner "
                    f"(closest: '{best_owner}', similarity: {best_match:.0%}). This may indicate "
                    f"that patta transfer has not been completed after the sale.",
                    "Verify that patta has been transferred to the buyer's name. If not, "
                    "ensure patta transfer is a condition of the transaction.",
                    f"[{claimant_fn}] ec_claimant='{claimant_name}', patta_owner='{best_owner}' (match: {best_match:.0%})",
                ))

    # Check: Last EC claimant should match sale deed seller (chain continuity)
    if ec_parties and sellers:
        # Get the most recent claimants from EC
        recent_claimants = [name for name, role, _ in ec_parties if role == "claimant"]
        if recent_claimants:
            last_claimant = recent_claimants[-1]
            for seller_name, seller_fn in sellers:
                sim = _name_similarity(last_claimant, seller_name)
                if sim < 0.5 and seller_name.strip() and last_claimant.strip():
                    # Skip warning for garbled Tamil names
                    is_garbled, _q, _r = _detect_garbled_tamil(last_claimant)
                    if is_garbled:
                        _trace(f"PARTY skipping chain gap warning — claimant appears garbled")
                        continue
                    checks.append(_make_check(
                        "DET_CHAIN_NAME_GAP", "Chain of Title Name Gap",
                        "HIGH", "WARNING",
                        f"Last EC claimant/buyer ('{last_claimant}') does not closely match "
                        f"sale deed seller ('{seller_name}', similarity: {sim:.0%}). "
                        f"There may be a gap in the chain of title.",
                        "Verify that the sale deed seller acquired title through a valid "
                        "transaction reflected in the EC.",
                        f"ec_claimant='{last_claimant}', [{seller_fn}] seller='{seller_name}' (match: {sim:.0%})",
                    ))

    return checks


# ═══════════════════════════════════════════════════
# 5. SURVEY NUMBER CROSS-CHECK (uses shared utils)
# ═══════════════════════════════════════════════════

def check_survey_number_consistency(extracted_data: dict) -> list[dict]:
    """Cross-check survey numbers across documents with fuzzy matching.

    Uses hierarchy-aware matching (311/1 ⊃ 311/1A) and OCR tolerance
    (Levenshtein ≤ 1) from utils.survey_numbers_match().
    """
    checks = []
    survey_records: list[tuple[str, str, str, str]] = []  # (original, normalized, filename, survey_type)

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                sn = prop.get("survey_number", "")
                if sn:
                    for s in split_survey_numbers(sn):
                        survey_records.append((s, _normalize_survey_number(s), filename, extract_survey_type(s)))

        elif doc_type in ("PATTA", "CHITTA"):
            for sn_obj in (d.get("survey_numbers") or []):
                if isinstance(sn_obj, dict):
                    sn = sn_obj.get("survey_no", "")
                    if sn:
                        survey_records.append((sn, _normalize_survey_number(sn), filename, extract_survey_type(sn)))
                elif isinstance(sn_obj, str) and sn_obj:
                    survey_records.append((sn_obj, _normalize_survey_number(sn_obj), sn_obj, extract_survey_type(sn_obj)))

        elif doc_type == "EC":
            # Check property description for survey no
            prop_desc = d.get("property_description", "")
            match = re.search(
                r'(?:'
                r's\.?f\.?\s*no\.?|'          # S.F.No., SF No
                r'r\.?s\.?\s*no\.?|'          # R.S.No.
                r't\.?s\.?\s*no\.?|'          # T.S.No.
                r'o\.?s\.?\s*no\.?|'          # O.S.No.
                r'n\.?s\.?\s*no\.?|'          # N.S.No.
                r's\.?no\.?|'                 # S.No., SNo
                r'survey\s*no\.?|'            # Survey No
                r'sy\.?\s*no\.?|'             # Sy.No.
                r'\u0BAA\u0BC1\u0BB2\s*\u0B8E\u0BA3\u0BCD\.?|'   # புல எண் (Tamil)
                r'\u0BA8\u0BBF\u0BB2\s*\u0B8E\u0BA3\u0BCD\.?'    # நில எண் (Tamil)
                r')\s*:?\s*([\d/\-A-Za-z,\s]+)',
                prop_desc, re.IGNORECASE | re.UNICODE
            )
            if match:
                for sn in split_survey_numbers(match.group(1).strip()):
                    survey_records.append((sn, _normalize_survey_number(sn), filename, extract_survey_type(sn)))
            # Also from individual transactions
            seen: set[str] = set()
            for txn in d.get("transactions", []):
                sn = txn.get("survey_number", "")
                if sn and sn not in seen:
                    seen.add(sn)
                    survey_records.append((sn, _normalize_survey_number(sn), filename, extract_survey_type(sn)))

    _trace(f"SURVEY_CHECK {len(survey_records)} records: {[(orig,fn) for orig,_,_,fn in survey_records]}")

    # ── Intra-EC consistency: header vs transactions ──
    # If a single EC's property_description survey differs from its
    # transaction survey numbers, flag an internal inconsistency.
    ec_files: dict[str, list[tuple[str, str]]] = {}  # filename → [(orig, norm)]
    for orig, norm, fn, _st in survey_records:
        # Only look at EC-sourced records
        for _filename, _data in extracted_data.items():
            if _filename == fn and _data.get("document_type") == "EC":
                ec_files.setdefault(fn, []).append((orig, norm))
                break
    for fn, recs in ec_files.items():
        if len(recs) < 2:
            continue
        # Build mini-clusters within this single EC
        ec_clusters: list[list[str]] = []
        for orig, _norm in recs:
            placed = False
            for ecl in ec_clusters:
                for existing in ecl:
                    m, _mt = survey_numbers_match(orig, existing)
                    if m:
                        ecl.append(orig)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                ec_clusters.append([orig])
        if len(ec_clusters) > 1:
            groups = " vs ".join(
                "{" + ", ".join(sorted(set(cl))) + "}" for cl in ec_clusters[:4]
            )
            checks.append(_make_check(
                "DET_EC_INTERNAL_SURVEY_INCONSISTENCY",
                "EC Internal Survey Number Inconsistency",
                "HIGH", "WARNING",
                f"Within the same EC ({fn}), the property description and "
                f"transaction entries reference {len(ec_clusters)} distinct "
                f"survey number groups: {groups}. This may indicate OCR errors, "
                f"a multi-property EC, or data extraction issues.",
                "Compare against the original EC document to confirm which "
                "survey numbers actually appear.",
                f"{fn}: {groups}",
            ))

    if len(survey_records) < 2:
        return checks

    # Build equivalence clusters using fuzzy matching
    clusters: list[list[tuple[str, str, str, str]]] = []  # (orig, norm, file, survey_type)
    for orig, norm, fn, stype in survey_records:
        if not norm:
            continue
        placed = False
        for cluster in clusters:
            # Check if this record matches any member of the cluster
            for c_orig, c_norm, _c_fn, _c_st in cluster:
                matched, _mtype = survey_numbers_match(orig, c_orig)
                if matched:
                    cluster.append((orig, norm, fn, stype))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([(orig, norm, fn, stype)])

    _trace(f"SURVEY_CLUSTERS {len(clusters)} cluster(s): {[[orig for orig,_,_,_ in c] for c in clusters]}")
    if len(clusters) <= 1:
        # Even with one cluster, check for survey type differences
        if clusters:
            types_seen = {st for _, _, _, st in clusters[0] if st}
            if len(types_seen) > 1:
                type_details = []
                for orig, _, fn, st in clusters[0]:
                    if st:
                        type_details.append(f"{fn}('{orig}' → {st})")
                checks.append(_make_check(
                    "DET_SURVEY_TYPE_DIFF", "Different Survey Type Prefixes",
                    "MEDIUM", "INFO",
                    f"Documents reference the same survey number but use different survey "
                    f"type prefixes: {', '.join(sorted(types_seen))}. In Tamil Nadu, the same "
                    f"plot may appear under Old Survey (OS), Town Survey (TS), Re-Survey (RS), "
                    f"or Sub-Field (SF) numbering depending on the era.",
                    "Verify the mapping between old and new survey numbers at the "
                    "village taluk office or using the TN survey records portal.",
                    " | ".join(type_details[:5]),
                ))
        return checks

    # Check match types within each cluster for informational messages
    # and flag cross-cluster mismatches
    subdivision_pairs = []
    ocr_pairs = []
    for cluster in clusters:
        for i, (oa, _na, fa, _sa) in enumerate(cluster):
            for ob, _nb, fb, _sb in cluster[i+1:]:
                _m, mtype = survey_numbers_match(oa, ob)
                if mtype == "subdivision":
                    subdivision_pairs.append((oa, ob, fa, fb))
                elif mtype == "ocr_fuzzy":
                    ocr_pairs.append((oa, ob, fa, fb))

    # Report subdivision matches as INFO (not errors)
    for oa, ob, fa, fb in subdivision_pairs:
        checks.append(_make_check(
            "DET_SURVEY_SUBDIVISION", "Survey Number Subdivision Match",
            "MEDIUM", "INFO",
            f"Survey numbers '{oa}' (in {fa}) and '{ob}' (in {fb}) appear to be "
            f"parent-child subdivisions of the same survey. This is normal when "
            f"a Patta lists subdivisions but the EC references the parent number.",
            "Confirm subdivision relationship with village revenue records.",
            f"{fa}('{oa}') ↔ {fb}('{ob}'): subdivision",
        ))

    # Report OCR fuzzy matches as WARNING
    for oa, ob, fa, fb in ocr_pairs:
        checks.append(_make_check(
            "DET_SURVEY_OCR_FUZZY", "Survey Number Near-Match (Possible OCR Error)",
            "HIGH", "WARNING",
            f"Survey numbers '{oa}' (in {fa}) and '{ob}' (in {fb}) differ by only "
            f"one character. This may be an OCR/data-entry error.",
            "Verify the correct survey number from the original document.",
            f"{fa}('{oa}') ↔ {fb}('{ob}'): Levenshtein ≤ 1",
        ))

    # If there are genuinely different clusters (after fuzzy consolidation), flag CRITICAL
    if len(clusters) > 1:
        # Collect survey types across all clusters
        all_types = {st for cl in clusters for _, _, _, st in cl if st}
        type_note = ""
        if len(all_types) > 1:
            type_note = (f" Note: documents use different survey type prefixes "
                         f"({', '.join(sorted(all_types))}), which may explain the "
                         f"difference if old/new survey numbers haven't been mapped.")

        detail_parts = []
        for i, cluster in enumerate(clusters):
            entries = []
            for orig, _n, fn, st in cluster:
                label = f"{fn}('{orig}')"
                if st:
                    label = f"{fn}('{orig}' [{st}])"
                entries.append(label)
            detail_parts.append(f"Group {i+1}: {', '.join(entries)}")

        checks.append(_make_check(
            "DET_SURVEY_MISMATCH", "Survey Number Inconsistency Across Documents",
            "CRITICAL", "FAIL",
            f"Found {len(clusters)} distinct survey number group(s) across documents. "
            f"Documents for the same property should reference the same survey number "
            f"(accounting for subdivisions).{type_note}",
            "Verify the correct survey number from the village revenue records. "
            "Mismatch may indicate wrong documents or mixed-up properties."
            + (" Check the old survey ↔ new survey mapping at the taluk office." if type_note else ""),
            " | ".join(detail_parts[:5]),
        ))

    return checks


# ═══════════════════════════════════════════════════
# 6. RAPID FLIPPING DETECTION
# ═══════════════════════════════════════════════════

def check_rapid_flipping(extracted_data: dict) -> list[dict]:
    """Detect suspiciously rapid property transactions in EC."""
    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "EC":
            continue

        d = data.get("data", {})
        sale_txns = []
        for txn in d.get("transactions", []):
            ttype = (txn.get("transaction_type") or "").lower()
            if ttype in ("sale", "sale deed"):
                txn_date = _parse_date(txn.get("date", ""))
                if txn_date:
                    sale_txns.append((txn_date, txn))

        sale_txns.sort(key=lambda x: x[0])

        _trace(f"RAPID_FLIP [{filename}] {len(sale_txns)} sale transactions found")
        critical_flips = []   # < 180 days (6 months)
        warning_flips = []    # 180-364 days (6-12 months)
        for i in range(1, len(sale_txns)):
            gap = (sale_txns[i][0] - sale_txns[i - 1][0]).days
            flip_desc = (
                f"Sale on {sale_txns[i-1][1].get('date')} → Sale on {sale_txns[i][1].get('date')} "
                f"({gap} days gap)"
            )
            if 0 <= gap < 180:
                critical_flips.append(flip_desc)
            elif gap < 365:
                warning_flips.append(flip_desc)

        if critical_flips:
            checks.append(_make_check(
                "DET_RAPID_FLIPPING", "Rapid Property Flipping Detected (< 6 months)",
                "HIGH", "FAIL",
                f"Property was sold {len(critical_flips) + 1} times within 6 months. "
                f"Rapid flipping within such a short period strongly suggests "
                f"speculative trading, price manipulation, or fraud.",
                "Investigate the reason for rapid resale. Verify that each transaction "
                "has legitimate business rationale and appropriate consideration.",
                f"[{filename}] " + "; ".join(critical_flips[:3]),
            ))
        if warning_flips:
            checks.append(_make_check(
                "DET_RAPID_FLIPPING", "Rapid Property Flipping Detected (< 12 months)",
                "HIGH", "WARNING",
                f"Property was sold {len(warning_flips) + 1} times within 6-12 months. "
                f"Rapid flipping may indicate speculative trading or fraud.",
                "Investigate the reason for rapid resale. Verify that each transaction "
                "has legitimate business reason and appropriate consideration.",
                f"[{filename}] " + "; ".join(warning_flips[:3]),
            ))

    return checks


# ═══════════════════════════════════════════════════
# 6b. MULTIPLE SALES — deterministic chain validation
# ═══════════════════════════════════════════════════

def check_multiple_sales(extracted_data: dict) -> list[dict]:
    """Detect multiple sales of the same property by the same seller to different buyers.

    Scans EC transactions for sale-type entries and checks whether a seller
    appears in more than one sale to different buyers for the same survey number.
    A valid resale chain (A→B, B→C) is normal; same seller selling twice
    (A→B, A→C) is the red flag.
    """
    from app.pipeline.utils import normalize_name

    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "EC":
            continue

        d = data.get("data", {})
        transactions = d.get("transactions", [])

        # Collect sale-type transactions with parties
        sale_txns = []
        for txn in transactions:
            ttype = (txn.get("transaction_type") or "").lower()
            if ttype in ("sale", "sale deed", "conveyance", "gift deed", "gift",
                         "settlement", "settlement deed", "exchange"):
                seller = normalize_name(txn.get("seller_or_executant") or "")
                buyer = normalize_name(txn.get("buyer_or_claimant") or "")
                if seller and buyer:
                    sale_txns.append({
                        "seller": seller,
                        "buyer": buyer,
                        "type": txn.get("transaction_type", "sale"),
                        "date": txn.get("date", "?"),
                        "doc_no": txn.get("document_number", "?"),
                    })

        _trace(f"MULTIPLE_SALES [{filename}] {len(sale_txns)} sale-type transactions")

        if len(sale_txns) < 2:
            continue

        # Group by seller: if same seller sells to 2+ different buyers, flag it
        from collections import defaultdict
        seller_sales: dict[str, list[dict]] = defaultdict(list)
        for txn in sale_txns:
            seller_sales[txn["seller"]].append(txn)

        for seller, txns in seller_sales.items():
            if len(txns) < 2:
                continue
            unique_buyers = set(t["buyer"] for t in txns)
            if len(unique_buyers) >= 2:
                # Same seller sold to different buyers — suspicious
                buyer_list = ", ".join(unique_buyers)
                txn_details = "; ".join(
                    f"{t['type']} to {t['buyer']} (Doc #{t['doc_no']}, {t['date']})"
                    for t in txns[:4]
                )
                checks.append(_make_check(
                    "DET_MULTIPLE_SALES",
                    "Multiple Sales by Same Seller Detected",
                    "CRITICAL", "FAIL",
                    f"Seller '{seller}' has sold the property to {len(unique_buyers)} "
                    f"different buyers: {buyer_list}. This indicates possible fraud — "
                    f"selling the same property multiple times to different parties.",
                    "Immediately verify the ownership chain. Confirm which transaction "
                    "is valid and whether subsequent sales were legally authorized.",
                    f"[{filename}] {txn_details}",
                ))
            # If same seller sold to same buyer multiple times, that's unusual but less critical
            elif len(txns) >= 3:
                checks.append(_make_check(
                    "DET_MULTIPLE_SALES",
                    "Repeated Sales Between Same Parties",
                    "HIGH", "WARNING",
                    f"Seller '{seller}' has {len(txns)} sale transactions to the same buyer "
                    f"'{txns[0]['buyer']}'. While not necessarily fraudulent, repeated "
                    f"buy-sell between the same parties may indicate benami transactions.",
                    "Verify the business rationale for repeated transactions between "
                    "the same parties.",
                    f"[{filename}] {len(txns)} transactions",
                ))

    return checks


# ═══════════════════════════════════════════════════
# 7. CROSS-DOCUMENT PLOT / PROPERTY IDENTITY
# ═══════════════════════════════════════════════════

_PLOT_RE = re.compile(
    r'(?:plot|door|flat|unit|block)\s*(?:no\.?|number|#)?\s*[:\-]?\s*(\w[\w\-/]*)',
    re.IGNORECASE,
)


def check_plot_identity_consistency(extracted_data: dict) -> list[dict]:
    """Detect when documents reference different plots, doors, or layout units.

    Catches the "Plot 27 vs Plot 26" problem — documents that reference the
    same survey number but different plot/door identifiers are likely for
    different physical properties.
    """
    checks = []
    plot_records: list[tuple[str, str, str]] = []  # (plot_id, source_field, filename)

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        # Extract plot/door numbers from property descriptions
        texts_to_search: list[tuple[str, str]] = []  # (text, field_name)

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                for field in ("plot_number", "door_number", "flat_number"):
                    v = prop.get(field)
                    if v and isinstance(v, str) and v.strip():
                        plot_records.append((v.strip().lower(), field, filename))
                desc = prop.get("description", "")
                if desc:
                    texts_to_search.append((desc, "property.description"))
                boundaries = prop.get("boundaries")
                if isinstance(boundaries, str) and boundaries:
                    texts_to_search.append((boundaries, "boundaries"))

        elif doc_type == "EC":
            prop_desc = d.get("property_description", "")
            if prop_desc:
                texts_to_search.append((prop_desc, "property_description"))

        # Regex-extract plot/door numbers from free text
        for text, field_name in texts_to_search:
            for m in _PLOT_RE.finditer(text):
                plot_id = m.group(1).strip().lower()
                if plot_id and len(plot_id) <= 10:  # Sanity bound
                    plot_records.append((plot_id, field_name, filename))

    # Compare plot identifiers across files
    if len(plot_records) >= 2:
        by_file: dict[str, set[str]] = {}
        for pid, _field, fn in plot_records:
            by_file.setdefault(fn, set()).add(pid)

        # Cross-file comparison: if any two files have non-overlapping plot IDs
        files = list(by_file.keys())
        for i, fa in enumerate(files):
            for fb in files[i+1:]:
                plots_a = by_file[fa]
                plots_b = by_file[fb]
                if plots_a and plots_b and not plots_a.intersection(plots_b):
                    checks.append(_make_check(
                        "DET_PLOT_IDENTITY_MISMATCH",
                        "Different Plot/Door Numbers Across Documents",
                        "CRITICAL", "FAIL",
                        f"Documents reference different plot/door identifiers: "
                        f"{fa} has {sorted(plots_a)} while {fb} has {sorted(plots_b)}. "
                        f"These may be documents for different properties.",
                        "Verify that all documents pertain to the same physical property. "
                        "Check the plot number, door number, and layout plan.",
                        f"{fa}={sorted(plots_a)} vs {fb}={sorted(plots_b)}",
                    ))

    return checks


# ═══════════════════════════════════════════════════
# 8. FINANCIAL SCALE ANOMALY DETECTION
# ═══════════════════════════════════════════════════

def check_financial_scale_anomalies(extracted_data: dict) -> list[dict]:
    """Detect suspicious jumps in transaction values across EC entries.

    Flags:
      - 10x+ scale jumps between consecutive transactions (lakhs → crores)
      - Mortgage amounts exceeding the last sale price by >2x
      - Active mortgages (no matching release) totalling more than sale price
    """
    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "EC":
            continue

        d = data.get("data", {})
        transactions = d.get("transactions", [])

        # Collect all transactions with parsed amounts and dates
        valued_txns: list[dict] = []
        mortgages: list[dict] = []
        releases: set[str] = set()
        last_sale_amount: Optional[float] = None

        for txn in transactions:
            ttype = (txn.get("transaction_type") or "").upper()
            amount = _parse_amount(txn.get("consideration_amount"))
            date = txn.get("date", "")
            doc_no = txn.get("document_number", "")

            if amount and amount > 0:
                valued_txns.append({
                    "type": ttype, "amount": amount, "date": date,
                    "row": txn.get("row_number", "?"), "doc_no": doc_no,
                })

            if ttype == "MORTGAGE" and amount and amount > 0:
                mortgages.append({
                    "amount": amount, "date": date, "doc_no": doc_no,
                    "row": txn.get("row_number", "?"),
                })
            elif ttype in ("RELEASE", "RECONVEYANCE"):
                # Track released mortgages by document number reference
                releases.add(doc_no)
                # Also check remarks for referenced doc numbers
                remarks = txn.get("remarks", "")
                if remarks:
                    ref_match = re.search(r'(?:doc|document)\s*(?:no\.?)?\s*(\d+)', remarks, re.IGNORECASE)
                    if ref_match:
                        releases.add(ref_match.group(1))

            if ttype == "SALE":
                last_sale_amount = amount

        # Check 1: Scale jumps between consecutive valued transactions
        if len(valued_txns) >= 2:
            for i in range(1, len(valued_txns)):
                prev = valued_txns[i - 1]
                curr = valued_txns[i]
                if prev["amount"] > 0 and curr["amount"] > 0:
                    ratio = curr["amount"] / prev["amount"]
                    inv_ratio = prev["amount"] / curr["amount"]
                    jump_ratio = max(ratio, inv_ratio)
                    if jump_ratio >= 10:
                        # 10x jump
                        def _fmt(v: float) -> str:
                            if v >= 1_00_00_000:
                                return f"₹{v/1_00_00_000:.1f} Cr"
                            elif v >= 1_00_000:
                                return f"₹{v/1_00_000:.1f} L"
                            return f"₹{v:,.0f}"

                        checks.append(_make_check(
                            "DET_FINANCIAL_SCALE_JUMP",
                            "Financial Scale Jump Detected",
                            "HIGH", "WARNING",
                            f"Transaction value jumped {jump_ratio:.0f}x between row "
                            f"{prev['row']} ({prev['type']} {_fmt(prev['amount'])}) and "
                            f"row {curr['row']} ({curr['type']} {_fmt(curr['amount'])}). "
                            f"A jump from lakhs to crores may indicate fraud, "
                            f"data-entry error, or a significant change in property value.",
                            "Verify the transaction amounts against the original documents. "
                            "Cross-check with guideline values for the respective years.",
                            f"[{filename}] Row {prev['row']}: {_fmt(prev['amount'])} → "
                            f"Row {curr['row']}: {_fmt(curr['amount'])} ({jump_ratio:.0f}x)",
                        ))

        # Check 2: Mortgage exceeds last sale price by >2x
        if last_sale_amount and last_sale_amount > 0:
            for mort in mortgages:
                if mort["amount"] > last_sale_amount * 2:
                    ratio = mort["amount"] / last_sale_amount
                    checks.append(_make_check(
                        "DET_MORTGAGE_EXCEEDS_SALE",
                        "Mortgage Amount Exceeds Sale Consideration",
                        "HIGH", "WARNING",
                        f"Mortgage of ₹{mort['amount']:,.0f} (row {mort['row']}) is "
                        f"{ratio:.1f}x the last sale price of ₹{last_sale_amount:,.0f}. "
                        f"This is unusual and may indicate over-leveraging or inflated "
                        f"property value for loan purposes.",
                        "Verify mortgage terms and assess if the property valuation "
                        "supports the loan amount. Check for potential loan fraud.",
                        f"[{filename}] Mortgage: ₹{mort['amount']:,.0f} vs "
                        f"Last sale: ₹{last_sale_amount:,.0f} ({ratio:.1f}x)",
                    ))

        # Check 3: Active (unreleased) mortgage total vs sale price
        active_mortgage_total = 0.0
        for mort in mortgages:
            if mort["doc_no"] not in releases:
                active_mortgage_total += mort["amount"]
        if active_mortgage_total > 0 and last_sale_amount and last_sale_amount > 0:
            if active_mortgage_total > last_sale_amount * 1.5:
                checks.append(_make_check(
                    "DET_ACTIVE_MORTGAGE_BURDEN",
                    "Active Mortgage Burden Exceeds Property Value",
                    "HIGH", "WARNING",
                    f"Total active (unreleased) mortgages of ₹{active_mortgage_total:,.0f} "
                    f"exceed the last sale consideration of ₹{last_sale_amount:,.0f} by "
                    f"{active_mortgage_total/last_sale_amount:.1f}x. The property may be "
                    f"over-encumbered.",
                    "Verify that all mortgage releases have been properly registered. "
                    "Obtain NOC from all lenders before purchase.",
                    f"[{filename}] Active mortgages: ₹{active_mortgage_total:,.0f}, "
                    f"Last sale: ₹{last_sale_amount:,.0f}",
                ))

    return checks


# ═══════════════════════════════════════════════════
# 9. MULTI-VILLAGE / GEOGRAPHICAL BOUNDARY DETECTION
# ═══════════════════════════════════════════════════

def check_multi_village(extracted_data: dict) -> list[dict]:
    """Detect when documents reference multiple villages, taluks, or districts.

    Documents for the same property should reference the same village.
    Different villages = likely different properties in the upload.
    Different taluks or districts = almost certainly wrong documents.
    """
    checks = []
    village_records: list[tuple[str, str]] = []  # (village_name, filename)
    taluk_records: list[tuple[str, str]] = []
    district_records: list[tuple[str, str]] = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        village = taluk = district = None

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                village = prop.get("village")
                taluk = prop.get("taluk")
                district = prop.get("district")

        elif doc_type in ("PATTA", "CHITTA"):
            village = d.get("village")
            taluk = d.get("taluk")
            district = d.get("district")

        elif doc_type == "EC":
            village = d.get("village")
            taluk = d.get("taluk")

        if village and isinstance(village, str) and village.strip():
            village_records.append((village.strip(), filename))
        if taluk and isinstance(taluk, str) and taluk.strip():
            taluk_records.append((taluk.strip(), filename))
        if district and isinstance(district, str) and district.strip():
            district_records.append((district.strip(), filename))

    # Check district consistency (most severe)
    if len(district_records) >= 2:
        district_groups: list[list[tuple[str, str]]] = []
        for name, fn in district_records:
            placed = False
            for group in district_groups:
                matched, _ = village_names_match(name, group[0][0])
                if matched:
                    group.append((name, fn))
                    placed = True
                    break
            if not placed:
                district_groups.append([(name, fn)])

        if len(district_groups) > 1:
            detail = " vs ".join(
                f"{g[0][0]} ({', '.join(fn for _, fn in g)})"
                for g in district_groups
            )
            checks.append(_make_check(
                "DET_MULTI_DISTRICT", "Documents Reference Different Districts",
                "CRITICAL", "FAIL",
                f"Documents reference {len(district_groups)} different districts: {detail}. "
                f"This almost certainly means the documents are for different properties.",
                "Verify that all uploaded documents belong to the same property. "
                "Remove documents from wrong districts.",
                detail,
            ))

    # Check taluk consistency
    if len(taluk_records) >= 2:
        taluk_groups: list[list[tuple[str, str]]] = []
        for name, fn in taluk_records:
            placed = False
            for group in taluk_groups:
                matched, _ = village_names_match(name, group[0][0])
                if matched:
                    group.append((name, fn))
                    placed = True
                    break
            if not placed:
                taluk_groups.append([(name, fn)])

        if len(taluk_groups) > 1:
            detail = " vs ".join(
                f"{g[0][0]} ({', '.join(fn for _, fn in g)})"
                for g in taluk_groups
            )
            checks.append(_make_check(
                "DET_MULTI_TALUK", "Documents Reference Different Taluks",
                "CRITICAL", "FAIL",
                f"Documents reference {len(taluk_groups)} different taluks: {detail}. "
                f"Documents for the same property should be in the same taluk.",
                "Verify taluk information. If correct, these are documents for "
                "different properties and should be analyzed separately.",
                detail,
            ))

    # Check village consistency (least severe of the three)
    if len(village_records) >= 2:
        village_groups: list[list[tuple[str, str]]] = []
        for name, fn in village_records:
            placed = False
            for group in village_groups:
                matched, _ = village_names_match(name, group[0][0])
                if matched:
                    group.append((name, fn))
                    placed = True
                    break
            if not placed:
                village_groups.append([(name, fn)])

        if len(village_groups) > 1:
            detail = " vs ".join(
                f"{g[0][0]} ({', '.join(fn for _, fn in g)})"
                for g in village_groups
            )
            checks.append(_make_check(
                "DET_MULTI_VILLAGE", "Documents Reference Multiple Villages",
                "HIGH", "WARNING",
                f"Documents reference {len(village_groups)} different villages: {detail}. "
                f"This may indicate documents for different properties, or a property "
                f"spanning village boundaries (which is unusual and requires verification).",
                "Confirm the village name with revenue records. If the property genuinely "
                "spans villages, obtain a combined patta or FMB from both villages.",
                detail,
            ))

    return checks


# ═══════════════════════════════════════════════════
# 8. AGE FRAUD — deterministic minor seller detection
# ═══════════════════════════════════════════════════

def check_age_fraud(extracted_data: dict) -> list[dict]:
    """Detect sellers/buyers who were minors (< 18) at the time of transaction.

    Uses Sale Deed party ages + registration date.  If age is stated in the
    document AND the party was < 18 at registration → FAIL (CRITICAL).
    Also checks for impossible ages (< 0, > 120) and age decreasing across
    transactions (if multiple Sale Deeds).
    """
    checks = []

    # Collect registration dates from Sale Deeds
    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        if doc_type != "SALE_DEED":
            continue

        d = data.get("data", {})
        reg_date_str = d.get("registration_date", "")
        reg_date = _parse_date(reg_date_str)

        # Check sellers
        for party_list, role in [(d.get("seller", []), "Seller"), (d.get("buyer", []), "Buyer")]:
            if not isinstance(party_list, list):
                continue
            for party in party_list:
                if not isinstance(party, dict):
                    continue
                name = party.get("name", "unknown")
                age_raw = party.get("age")

                # Parse age — could be int or string like "45", "unknown"
                age = None
                if isinstance(age_raw, (int, float)) and age_raw > 0:
                    age = int(age_raw)
                elif isinstance(age_raw, str):
                    try:
                        age = int(age_raw.strip())
                    except (ValueError, AttributeError):
                        pass

                if age is None:
                    continue

                _trace(f"AGE_FRAUD [{filename}] {role} '{name}' age={age} reg_date={reg_date_str}")

                # Check for impossible ages
                if age < 0 or age > 120:
                    checks.append(_make_check(
                        "DET_AGE_IMPOSSIBLE", "Impossible Age Detected",
                        "CRITICAL", "FAIL",
                        f"{role} '{name}' has an impossible age of {age} years "
                        f"in the Sale Deed registered on {reg_date_str}. "
                        f"This indicates a data error or deliberate falsification.",
                        "Verify the age of the party from independent identity documents "
                        "(Aadhaar, PAN, passport). Do not proceed until age is confirmed.",
                        f"[{filename}] {role}: {name}, stated age: {age}",
                    ))
                elif age < 18:
                    checks.append(_make_check(
                        "DET_MINOR_PARTY", "Minor Party in Transaction",
                        "CRITICAL", "FAIL",
                        f"{role} '{name}' was only {age} years old at the time of "
                        f"registration ({reg_date_str}). Minors cannot legally execute "
                        f"property transactions. The transaction may be void or voidable.",
                        "Verify the party's actual age with identity documents. "
                        "If confirmed as a minor, the transaction is legally void — "
                        "consult a property lawyer before proceeding.",
                        f"[{filename}] {role}: {name}, age: {age}, registration: {reg_date_str}",
                    ))

    return checks


# ═══════════════════════════════════════════════════
# POST-EXTRACTION FIELD FORMAT VALIDATION
# ═══════════════════════════════════════════════════

# A valid normalized survey number: digits, optional alpha, slashes/dashes
_VALID_SURVEY_RE = re.compile(r'^\d+[a-zA-Z]?(?:[/\-]\d+[a-zA-Z]?\d*)*$')

_MIN_NAME_LEN = 3
_MAX_NAME_LEN = 60


def check_field_format_validity(extracted_data: dict) -> list[dict]:
    """Detect fields with invalid format patterns that indicate extraction errors.

    Checks survey number format, village/name length, date parseability,
    and amount parseability to catch garbled OCR or LLM hallucinations.
    """
    checks = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        # ── Survey number format ──
        raw_surveys: list[str] = []
        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict) and prop.get("survey_number"):
                raw_surveys.append(str(prop["survey_number"]))
        elif doc_type in ("PATTA", "CHITTA"):
            for sn_obj in (d.get("survey_numbers") or []):
                if isinstance(sn_obj, dict) and sn_obj.get("survey_no"):
                    raw_surveys.append(str(sn_obj["survey_no"]))
                elif isinstance(sn_obj, str):
                    raw_surveys.append(sn_obj)

        for raw_sn in raw_surveys:
            norm = _normalize_survey_number(raw_sn)
            if norm and not _VALID_SURVEY_RE.match(norm):
                checks.append(_make_check(
                    "DET_INVALID_SURVEY_FORMAT", "Invalid Survey Number Format",
                    "MEDIUM", "WARNING",
                    f"Survey number '{raw_sn}' (normalized: '{norm}') does not match "
                    f"the expected TN format (digits with optional /- subdivisions). "
                    f"This may indicate OCR garble or extraction error.",
                    "Verify the survey number from the original document.",
                    f"[{filename}] survey='{raw_sn}'",
                ))

        # ── Village / Taluk / District name validity ──
        _name_fields: list[tuple[str, str]] = []
        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                for nf in ("village", "taluk", "district"):
                    v = prop.get(nf)
                    if v:
                        _name_fields.append((nf, str(v)))
        elif doc_type in ("PATTA", "CHITTA"):
            for nf in ("village", "taluk", "district"):
                v = d.get(nf)
                if v:
                    _name_fields.append((nf, str(v)))

        for field_name, name_val in _name_fields:
            stripped = name_val.strip()
            if len(stripped) < _MIN_NAME_LEN:
                checks.append(_make_check(
                    "DET_IMPLAUSIBLE_NAME", f"Suspiciously Short {field_name.title()}",
                    "MEDIUM", "WARNING",
                    f"{field_name.title()} '{stripped}' is only {len(stripped)} character(s). "
                    f"Valid TN location names are at least 3 characters. "
                    f"This may indicate garbled extraction.",
                    "Verify the location name from the original document.",
                    f"[{filename}] {field_name}='{stripped}'",
                ))
            elif len(stripped) > _MAX_NAME_LEN:
                checks.append(_make_check(
                    "DET_IMPLAUSIBLE_NAME", f"Suspiciously Long {field_name.title()}",
                    "MEDIUM", "WARNING",
                    f"{field_name.title()} '{stripped[:50]}...' is {len(stripped)} characters. "
                    f"This exceeds expected length and may indicate OCR noise "
                    f"concatenated with the actual name.",
                    "Verify the location name from the original document.",
                    f"[{filename}] {field_name} length={len(stripped)}",
                ))
            # Flag digit-only "names" (OCR garble)
            elif stripped.replace(" ", "").isdigit():
                checks.append(_make_check(
                    "DET_IMPLAUSIBLE_NAME", f"Digit-Only {field_name.title()}",
                    "MEDIUM", "WARNING",
                    f"{field_name.title()} '{stripped}' contains only digits. "
                    f"This is not a valid location name and likely indicates "
                    f"an extraction error.",
                    "Verify the location name from the original document.",
                    f"[{filename}] {field_name}='{stripped}'",
                ))

        # ── Date parseability ──
        _date_fields: list[tuple[str, str]] = []
        if doc_type == "SALE_DEED":
            for df in ("registration_date", "execution_date"):
                v = d.get(df)
                if v and isinstance(v, str) and v.strip():
                    _date_fields.append((df, v.strip()))
        elif doc_type == "EC":
            for df in ("period_from", "period_to"):
                v = d.get(df)
                if v and isinstance(v, str) and v.strip():
                    _date_fields.append((df, v.strip()))

        for field_name, date_val in _date_fields:
            parsed = _parse_date(date_val)
            if parsed is None:
                checks.append(_make_check(
                    "DET_UNPARSEABLE_DATE", f"Unparseable {field_name.replace('_', ' ').title()}",
                    "MEDIUM", "WARNING",
                    f"{field_name.replace('_', ' ').title()} '{date_val}' could not be "
                    f"parsed as a valid date. This may indicate OCR corruption or "
                    f"an unusual date format.",
                    "Verify the date from the original document.",
                    f"[{filename}] {field_name}='{date_val}'",
                ))

    return checks


# ═══════════════════════════════════════════════════
# 16. GARBLED TAMIL DETECTION
# ═══════════════════════════════════════════════════

def check_garbled_tamil(extracted_data: dict) -> list[dict]:
    """Detect garbled/corrupted Tamil text in extracted fields.

    Uses Unicode syllable-structure analysis from utils.detect_garbled_tamil()
    to identify fields where Tamil text is corrupted — orphan vowel signs,
    script mixing, broken syllable sequences.

    This is a single-document check: it examines each document independently.
    """
    from app.pipeline.utils import detect_garbled_tamil

    checks: list[dict] = []

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        d = doc.get("data", {})
        if not isinstance(d, dict):
            continue
        doc_type = doc.get("document_type", "OTHER")

        garbled_fields: list[tuple[str, str, float, str]] = []

        def _scan(field_path: str, val):
            """Recursively scan values for garbled Tamil."""
            if isinstance(val, str) and len(val) >= 4:
                is_garbled, quality, reason = detect_garbled_tamil(val)
                if is_garbled:
                    garbled_fields.append((field_path, val[:60], quality, reason))
            elif isinstance(val, dict):
                for k, v in val.items():
                    if not str(k).startswith("_"):
                        _scan(f"{field_path}.{k}", v)
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    _scan(f"{field_path}[{i}]", item)

        for key, val in d.items():
            if str(key).startswith("_"):
                continue
            _scan(key, val)

        for field_path, preview, quality, reason in garbled_fields:
            checks.append(_make_check(
                "DET_GARBLED_TAMIL", f"Garbled Tamil in {field_path.split('.')[0]}",
                "MEDIUM", "WARNING",
                f"Field '{field_path}' contains garbled Tamil text "
                f"(quality={quality:.2f}): {reason}. "
                f"Preview: '{preview}'",
                "Verify this field from the original document. The text may have "
                "been corrupted during OCR or extraction.",
                f"[{filename}] {field_path}='{preview}'",
            ))

    return checks


# ═══════════════════════════════════════════════════
# 17. HALLUCINATION / SUSPICIOUSLY PERFECT DETECTION
# ═══════════════════════════════════════════════════

def _is_suspiciously_round(amount: float) -> bool:
    """Check if an amount is 'too round' — exact lakhs/crores with no paise."""
    if amount <= 0:
        return False
    # Exact multiples of ₹1 lakh with no remainder
    if amount >= 100_000 and amount % 100_000 == 0:
        return True
    # Exact multiples of ₹10,000 (for smaller amounts)
    if amount >= 10_000 and amount % 10_000 == 0 and amount < 100_000:
        return True
    return False


def check_hallucination_signs(extracted_data: dict) -> list[dict]:
    """Detect signs of LLM hallucination / fabrication.

    Heuristics:
      1. ALL financial amounts are suspiciously round (exact lakh/crore multiples)
      2. Duplicate exact values across different fields (copy-paste)
      3. No extraction_notes/remarks despite complex document (zero uncertainty)
      4. EC: all transactions have identical consideration_amount
      5. Repeated party names in unrelated transactions
    """
    checks: list[dict] = []

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        d = doc.get("data", {})
        if not isinstance(d, dict):
            continue
        doc_type = doc.get("document_type", "OTHER")

        # ── Heuristic 1: All financials suspiciously round ──
        if doc_type == "SALE_DEED":
            financials = d.get("financials", {})
            if isinstance(financials, dict):
                amounts = []
                for amt_field in ("consideration_amount", "guideline_value",
                                  "stamp_duty", "registration_fee"):
                    raw = financials.get(amt_field)
                    if raw is not None and raw != "" and raw != 0:
                        parsed = _parse_amount(raw)
                        if parsed and parsed > 0:
                            amounts.append((amt_field, parsed))
                if len(amounts) >= 3:
                    round_count = sum(1 for _, a in amounts if _is_suspiciously_round(a))
                    if round_count == len(amounts):
                        checks.append(_make_check(
                            "DET_ALL_AMOUNTS_ROUND",
                            "All Amounts Suspiciously Round",
                            "MEDIUM", "WARNING",
                            f"All {len(amounts)} financial amounts are exact round numbers "
                            f"({', '.join(f'{f}=₹{a:,.0f}' for f,a in amounts)}). "
                            f"This pattern is unusual for real transactions and may indicate "
                            f"fabricated values.",
                            "Cross-verify all financial figures from the original Sale Deed.",
                            f"[{filename}] {len(amounts)} round amounts",
                        ))

        # ── Heuristic 2: Duplicate exact field values ──
        if doc_type == "SALE_DEED":
            financials = d.get("financials", {})
            if isinstance(financials, dict):
                vals = {}
                for k, v in financials.items():
                    if v is not None and v != "" and v != 0:
                        sv = str(v).strip()
                        if sv:
                            vals.setdefault(sv, []).append(k)
                for sv, fields in vals.items():
                    if len(fields) >= 3:
                        checks.append(_make_check(
                            "DET_DUPLICATE_VALUES",
                            "Identical Values Across Fields",
                            "LOW", "WARNING",
                            f"Fields {', '.join(fields)} all have the identical value "
                            f"'{sv}'. This suggests the LLM may have copied one value "
                            f"to multiple fields.",
                            "Verify each amount independently from the original document.",
                            f"[{filename}] value='{sv}' in {len(fields)} fields",
                        ))

        # ── Heuristic 3: EC — identical consideration across all transactions ──
        if doc_type == "EC":
            txns = d.get("transactions", [])
            if len(txns) >= 5:
                consideration_values = []
                for txn in txns:
                    if isinstance(txn, dict):
                        ca = txn.get("consideration_amount", "")
                        if ca and str(ca).strip() and str(ca).strip() != "0":
                            consideration_values.append(str(ca).strip())
                if len(consideration_values) >= 5:
                    unique = set(consideration_values)
                    if len(unique) == 1:
                        checks.append(_make_check(
                            "DET_EC_IDENTICAL_AMOUNTS",
                            "All EC Transactions Same Amount",
                            "MEDIUM", "WARNING",
                            f"All {len(consideration_values)} EC transactions have the "
                            f"identical consideration amount '{consideration_values[0]}'. "
                            f"This is highly unusual and may indicate the LLM fabricated "
                            f"or repeated a single value.",
                            "Verify transaction amounts from the original EC document.",
                            f"[{filename}] {len(consideration_values)} identical amounts",
                        ))

        # ── Heuristic 4: Zero uncertainty in complex docs ──
        if doc_type in ("EC", "SALE_DEED"):
            notes = d.get("extraction_notes", "")
            remarks = d.get("remarks", "")
            pages = d.get("pages_processed", 0) or 0
            has_notes = bool(
                (isinstance(notes, str) and notes.strip()) or
                (isinstance(remarks, str) and remarks.strip())
            )
            # Large doc (10+ pages for EC, any for Sale Deed) with zero notes
            # is suspicious — extraction always has some ambiguity
            txn_count = len(d.get("transactions", []))
            if doc_type == "EC" and txn_count >= 15 and not has_notes:
                checks.append(_make_check(
                    "DET_ZERO_UNCERTAINTY",
                    "No Extraction Notes Despite Complexity",
                    "LOW", "INFO",
                    f"EC with {txn_count} transactions has no extraction_notes or "
                    f"remarks. Complex documents typically have some ambiguous "
                    f"entries. This may indicate the LLM did not attempt to flag "
                    f"uncertainties.",
                    "Review extraction for potential silent errors.",
                    f"[{filename}] {txn_count} transactions, no notes",
                ))

        # ── Heuristic 5: EC — repeated parties in unrelated transactions ──
        if doc_type == "EC":
            txns = d.get("transactions", [])
            if len(txns) >= 5:
                seller_counts: dict[str, int] = {}
                buyer_counts: dict[str, int] = {}
                for txn in txns:
                    if isinstance(txn, dict):
                        s = str(txn.get("seller_or_executant", "")).strip().lower()
                        b = str(txn.get("buyer_or_claimant", "")).strip().lower()
                        if s and len(s) > 3:
                            seller_counts[s] = seller_counts.get(s, 0) + 1
                        if b and len(b) > 3:
                            buyer_counts[b] = buyer_counts.get(b, 0) + 1

                # If a single party appears in > 80% of transactions as the
                # same role, it may be hallucinated
                for party, count in {**seller_counts, **buyer_counts}.items():
                    ratio = count / len(txns)
                    if count >= 5 and ratio >= 0.8:
                        checks.append(_make_check(
                            "DET_REPEATED_PARTY",
                            "Same Party in Most Transactions",
                            "LOW", "WARNING",
                            f"Party '{party}' appears in {count}/{len(txns)} "
                            f"transactions ({ratio:.0%}). While this can happen "
                            f"(e.g. a developer), it may also indicate the LLM "
                            f"copied the same name across transactions.",
                            "Verify party names from the original EC.",
                            f"[{filename}] '{party}' in {count}/{len(txns)} txns",
                        ))

    return checks


# ═══════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════

def _make_check(rule_code: str, rule_name: str, severity: str, status: str,
                explanation: str, recommendation: str, evidence: str) -> dict:
    """Create a standardized check result dict."""
    return {
        "rule_code": rule_code,
        "rule_name": rule_name,
        "severity": severity,
        "status": status,
        "explanation": explanation,
        "recommendation": recommendation,
        "evidence": evidence,
        "source": "deterministic",  # Mark as Python-computed, not LLM
    }


# ═══════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════

def run_deterministic_checks(
    extracted_data: dict, *, identity_resolver=None
) -> list[dict]:
    """Run all deterministic checks and return a flat list of check results.

    Args:
        extracted_data: dict of {filename: {document_type, data}} from the pipeline
        identity_resolver: optional resolved :class:`IdentityResolver` for
            evidence-based name verification and chain continuity checks.

    Returns:
        List of check dicts ready to be merged into verification results
    """
    all_checks = []

    check_functions = [
        ("Temporal: EC period", check_ec_period_coverage),
        ("Temporal: Registration in EC", check_registration_within_ec),
        ("Temporal: Limitation period", check_limitation_period),
        ("Financial: Stamp duty", check_stamp_duty),
        ("Financial: Plausibility ranges", check_plausibility_ranges),
        ("Property: Area consistency", check_area_consistency),
        ("Property: Survey numbers", check_survey_number_consistency),
        ("Property: Plot identity", check_plot_identity_consistency),
        ("Party: Name consistency", check_party_name_consistency),
        ("Party: Age fraud", check_age_fraud),
        ("Pattern: Rapid flipping", check_rapid_flipping),
        ("Pattern: Multiple sales", check_multiple_sales),
        ("Financial: Scale anomalies", check_financial_scale_anomalies),
        ("Geography: Multi-village", check_multi_village),
        ("Validation: Field formats", check_field_format_validity),
        ("Validation: Garbled Tamil", check_garbled_tamil),
        ("Validation: Hallucination signs", check_hallucination_signs),
    ]

    for label, fn in check_functions:
        try:
            # Pass identity_resolver to name consistency check
            if fn is check_party_name_consistency:
                results = fn(extracted_data, identity_resolver=identity_resolver)
            else:
                results = fn(extracted_data)
            if results:
                logger.info(f"Deterministic [{label}]: {len(results)} check(s) generated")
            all_checks.extend(results)
        except Exception as e:
            logger.error(f"Deterministic [{label}] failed: {e}")

    # ── Identity-based chain continuity check ──
    if identity_resolver is not None and identity_resolver._resolved:
        try:
            chain_checks = identity_resolver.check_chain_continuity(extracted_data)
            if chain_checks:
                logger.info(f"Deterministic [Identity: Chain continuity]: "
                            f"{len(chain_checks)} check(s) generated")
            all_checks.extend(chain_checks)
        except Exception as e:
            logger.error(f"Deterministic [Identity: Chain continuity] failed: {e}")

    logger.info(f"Deterministic engine: {len(all_checks)} total checks generated")
    if TRACE_ENABLED:
        for c in all_checks:
            _trace(f"RESULT {c['rule_code']} status={c['status']} severity={c['severity']}")
    return all_checks
