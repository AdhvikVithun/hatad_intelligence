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
from app.config import TRACE_ENABLED, TN_STAMP_DUTY_RATE, TN_REGISTRATION_FEE_RATE

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
                        f"{txn.get('transaction_id', 'Row ' + str(txn.get('row_number', '?')))}: "
                        f"{txn.get('date')} is before "
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
                        f"{txn.get('transaction_id', 'Row ' + str(txn.get('row_number', '?')))}: "
                        f"{ttype} on {txn.get('date')} "
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

# Tamil Nadu stamp duty rates (from config, env-overridable)
_TN_STAMP_DUTY_RATE = TN_STAMP_DUTY_RATE
_TN_REGISTRATION_FEE_RATE = TN_REGISTRATION_FEE_RATE
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

                # Common single-sheet stamp denominations in Tamil Nadu
                _COMMON_STAMP_DENOMS = {100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 50000}
                multi_sheet_note = ""
                if stamp_duty_paid in _COMMON_STAMP_DENOMS and shortfall_pct > 80:
                    # Likely the extractor only counted one stamp sheet
                    possible_sheets = round(expected_stamp / stamp_duty_paid) if stamp_duty_paid else 0
                    multi_sheet_note = (
                        f" NOTE: The stamp duty paid (₹{stamp_duty_paid:,.0f}) matches a common "
                        f"single-sheet denomination. If the deed is printed on ~{possible_sheets} "
                        f"stamp sheets, total stamp duty may actually be "
                        f"₹{stamp_duty_paid * possible_sheets:,.0f}. "
                        f"Verify the physical stamp sheet count before concluding shortfall."
                    )

                checks.append(_make_check(
                    "DET_STAMP_DUTY_SHORT", "Stamp Duty Shortfall Detected",
                    "HIGH", "FAIL",
                    f"Stamp duty paid (₹{stamp_duty_paid:,.0f}) appears insufficient. "
                    f"Expected ₹{expected_stamp:,.0f} (7% of ₹{assessable_value:,.0f}). "
                    f"Shortfall: ₹{shortfall:,.0f} ({shortfall_pct:.1f}%)."
                    + multi_sheet_note,
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
        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
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
    """Cross-check property extent across all documents after unit normalization.

    Patta-portfolio aware: A Patta is an *owner-level* document listing ALL
    surveys held by one owner.  For area comparison we use only the Patta
    survey rows whose survey numbers match a survey in any non-Patta document
    (EC ∪ Sale Deed ∪ FMB).  The aggregate ``total_extent`` is preserved for
    Urban Land Ceiling Act / ceiling-surplus compliance checks.
    """
    checks = []

    # ── Pass 1: collect target survey numbers from every non-Patta doc ──
    target_surveys: list[str] = []
    for _fn, fdata in extracted_data.items():
        dt = fdata.get("document_type", "")
        d = fdata.get("data", {})
        if dt in ("PATTA", "CHITTA", "A_REGISTER"):
            continue
        if dt == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                sn = prop.get("survey_number", "")
                if sn:
                    target_surveys.extend(split_survey_numbers(sn))
        elif dt == "EC":
            prop_desc = d.get("property_description", "")
            m = re.search(
                r'(?:s\.?f\.?\s*no\.?|r\.?s\.?\s*no\.?|t\.?s\.?\s*no\.?|'
                r'o\.?s\.?\s*no\.?|n\.?s\.?\s*no\.?|s\.?no\.?|survey\s*no\.?|'
                r'sy\.?\s*no\.?)\s*:?\s*([\d/\-A-Za-z,\s]+)',
                prop_desc, re.IGNORECASE,
            )
            if m:
                target_surveys.extend(split_survey_numbers(m.group(1).strip()))
            for txn in d.get("transactions", []):
                sn = txn.get("survey_number", "")
                if sn:
                    target_surveys.extend(split_survey_numbers(sn))
        else:
            # FMB, A-certificate, etc.
            sn = d.get("survey_number", "")
            if sn:
                target_surveys.extend(split_survey_numbers(sn))

    target_surveys = list(set(filter(None, target_surveys)))

    # ── Pass 2: collect area records (Patta-portfolio filtered) ──
    area_records: list[tuple[str, str, float, str]] = []  # (filename, doc_type, sqft, text)

    for filename, fdata in extracted_data.items():
        doc_type = fdata.get("document_type", "")
        d = fdata.get("data", {})

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                extent_text = prop.get("extent")
                if extent_text:
                    sqft = _parse_area_to_sqft(str(extent_text))
                    if sqft and sqft > 0:
                        area_records.append((filename, doc_type, sqft, str(extent_text)))

        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
            survey_numbers = d.get("survey_numbers", [])
            if isinstance(survey_numbers, list) and survey_numbers and target_surveys:
                # Only sum extents for Patta surveys matching a target survey
                matched_sqft = 0.0
                matched_texts: list[str] = []
                for sn_obj in survey_numbers:
                    if not isinstance(sn_obj, dict):
                        continue
                    sn = sn_obj.get("survey_no", "")
                    sn_ext = sn_obj.get("extent", "")
                    if not sn or not sn_ext:
                        continue
                    for tgt in target_surveys:
                        m_ok, _ = survey_numbers_match(sn, tgt)
                        if m_ok:
                            sqft = _parse_area_to_sqft(str(sn_ext))
                            if sqft and sqft > 0:
                                matched_sqft += sqft
                                matched_texts.append(f"{sn}: {sn_ext}")
                            break
                if matched_sqft > 0:
                    area_records.append((filename, doc_type, matched_sqft,
                                        " + ".join(matched_texts)))
                else:
                    _trace(f"AREA_CHECK Patta {filename}: no surveys match "
                           f"targets {target_surveys}, skipping area comparison")
            elif not target_surveys:
                # No non-Patta surveys known — fall back to total_extent
                extent_text = d.get("total_extent") or d.get("extent")
                if extent_text:
                    sqft = _parse_area_to_sqft(str(extent_text))
                    if sqft and sqft > 0:
                        area_records.append((filename, doc_type, sqft, str(extent_text)))

        elif doc_type == "EC":
            extent_text = d.get("property_description", "")
            if extent_text:
                sqft = _parse_area_to_sqft(str(extent_text))
                if sqft and sqft > 0:
                    area_records.append((filename, doc_type, sqft, str(extent_text)))

        else:
            # FMB, A-certificate, etc.
            extent_text = d.get("extent") or d.get("area")
            if extent_text:
                sqft = _parse_area_to_sqft(str(extent_text))
                if sqft and sqft > 0:
                    area_records.append((filename, doc_type, sqft, str(extent_text)))

    _trace(f"AREA_CHECK records={[(fn,dt,f'{sqft:.0f}sqft') for fn,dt,sqft,_ in area_records]}")

    # ── Pairwise comparison (>10 % tolerance) ──
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

    # ── Patta-portfolio INFO when Patta covers more surveys than the target ──
    for filename, fdata in extracted_data.items():
        doc_type = fdata.get("document_type", "")
        d = fdata.get("data", {})
        if doc_type not in ("PATTA", "CHITTA", "A_REGISTER"):
            continue
        total_ext = d.get("total_extent")
        if not total_ext or not target_surveys:
            continue
        total_sqft = _parse_area_to_sqft(str(total_ext))
        if not total_sqft or total_sqft <= 0:
            continue
        matched_rec = [r for r in area_records if r[0] == filename]
        if matched_rec:
            matched_sqft = matched_rec[0][2]
            if abs(total_sqft - matched_sqft) / max(total_sqft, matched_sqft) > 0.10:
                sn_list = [s.get("survey_no", "?") for s in d.get("survey_numbers", [])
                           if isinstance(s, dict)]
                checks.append(_make_check(
                    "DET_PATTA_PORTFOLIO",
                    "Patta Lists Additional Surveys (Owner Portfolio)",
                    "LOW", "INFO",
                    f"Patta ({filename}) lists {len(sn_list)} surveys with total extent "
                    f"{total_ext} ({total_sqft:.0f} sqft), but only the matching survey(s) "
                    f"({matched_rec[0][3]}, {matched_sqft:.0f} sqft) were used for comparison. "
                    f"The Patta is an owner-level document covering this owner's entire "
                    f"land portfolio, not just the property under due diligence.",
                    "This is normal. Patta reflects the owner's full land holdings. "
                    "The total_extent is relevant for Urban Land Ceiling Act checks.",
                    f"Patta surveys: {', '.join(sn_list)} | Total: {total_ext} "
                    f"| Matched: {matched_rec[0][3]}",
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
        entry.get("document_type") in ("PATTA", "CHITTA", "A_REGISTER")
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
            # Before flagging, check if SELLER matches patta owner.
            # In TN, the Patta stays in the seller's name until mutation.
            # If seller == patta_owner, the buyer not matching is EXPECTED
            # (pending mutation), not a title defect.
            has_sellers = any(
                entry.get("document_type") == "SALE_DEED"
                and isinstance(entry.get("data", {}).get("seller"), list)
                and entry["data"]["seller"]
                for entry in extracted_data.values()
            )
            seller_is_patta_owner = False
            if has_sellers:
                seller_same, _, seller_ev = resolver.roles_share_identity(
                    "seller", "patta_owner"
                )
                seller_is_patta_owner = seller_same

            if seller_is_patta_owner:
                # Seller matches patta owner → buyer not matching is normal
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MATCH",
                    "Patta Mutation Pending (Normal)",
                    "MEDIUM", "INFO",
                    f"Buyer does not yet appear as Patta owner, but the seller "
                    f"matches the current Patta owner — this is normal in Tamil Nadu. "
                    f"The Patta will be mutated to the buyer's name after registration.\n"
                    f"Buyer ↔ Patta: {evidence}\n"
                    f"Seller ↔ Patta: {seller_ev}",
                    "Buyer should apply for patta transfer (mutation) at the taluk office "
                    "after registration. This is a routine procedural step.",
                    evidence,
                ))
            else:
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MISMATCH",
                    "Buyer-Patta Owner Identity Mismatch",
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

    # ── Fallback: pairwise fuzzy matching ──
    return _check_names_fuzzy(extracted_data)


def _check_names_fuzzy(extracted_data: dict) -> list[dict]:
    """Pairwise fuzzy name matching fallback (no IdentityResolver).

    Checks:
      1. Buyer ↔ Patta owner (DET_BUYER_PATTA_MISMATCH)
      2. Last EC claimant ↔ Sale deed seller (DET_CHAIN_NAME_GAP)
    """
    checks: list[dict] = []

    # ── Collect names from documents ────────────────────────────
    buyers: list[str] = []
    sellers: list[str] = []
    patta_owners: list[str] = []
    ec_claimants: list[str] = []  # last claimant per transaction, chronologically

    for _filename, entry in extracted_data.items():
        doc_type = entry.get("document_type", "")
        d = entry.get("data") or {}

        if doc_type == "SALE_DEED":
            for b in (d.get("buyer") or []):
                n = b.get("name", "") if isinstance(b, dict) else str(b)
                if n:
                    buyers.append(n)
            for s in (d.get("seller") or []):
                n = s.get("name", "") if isinstance(s, dict) else str(s)
                if n:
                    sellers.append(n)

        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
            for o in (d.get("owner_names") or []):
                n = o.get("name", "") if isinstance(o, dict) else str(o)
                if n:
                    patta_owners.append(n)

        elif doc_type == "EC":
            for txn in (d.get("transactions") or []):
                claimant = txn.get("buyer_or_claimant", "")
                if claimant:
                    ec_claimants.append(claimant)

    # ── Check 1: Buyer ↔ Patta owner ──────────────────────────
    if buyers and patta_owners:
        # Check if any buyer matches any patta owner
        any_match = False
        for buyer in buyers:
            if _detect_garbled_tamil(buyer)[0]:
                continue  # skip garbled names
            for owner in patta_owners:
                if _detect_garbled_tamil(owner)[0]:
                    continue
                sim = _name_similarity_utils(buyer, owner)
                if sim >= 0.5:
                    any_match = True
                    break
            if any_match:
                break

        if not any_match:
            # Before flagging, check if SELLER matches patta owner.
            # In TN, the Patta stays in the seller's name until mutation.
            seller_matches_patta = False
            if sellers and patta_owners:
                for seller in sellers:
                    if _detect_garbled_tamil(seller)[0]:
                        continue
                    for owner in patta_owners:
                        if _detect_garbled_tamil(owner)[0]:
                            continue
                        sim = _name_similarity_utils(seller, owner)
                        if sim >= 0.5:
                            seller_matches_patta = True
                            break
                    if seller_matches_patta:
                        break

            buyer_str = ", ".join(buyers[:3])
            owner_str = ", ".join(patta_owners[:3])

            if seller_matches_patta:
                seller_str = ", ".join(sellers[:3])
                evidence = (
                    f"Buyer(s): {buyer_str} do not match Patta owner(s): {owner_str}, "
                    f"but Seller(s): {seller_str} match the Patta owner — patta mutation pending (normal)."
                )
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MATCH",
                    "Patta Mutation Pending (Normal)",
                    "MEDIUM", "INFO",
                    evidence,
                    "Buyer should apply for patta transfer (mutation) at the taluk office "
                    "after registration. This is a routine procedural step.",
                    evidence,
                ))
            else:
                evidence = (f"Buyer(s): {buyer_str} | Patta owner(s): {owner_str} — "
                            f"no fuzzy match found (threshold 0.5).")
                checks.append(_make_check(
                    "DET_BUYER_PATTA_MISMATCH",
                    "Buyer-Patta Owner Name Mismatch",
                    "HIGH", "WARNING",
                    evidence,
                    "Verify that patta has been transferred to the buyer's name. "
                    "If not, ensure patta transfer is a condition of the transaction.",
                    evidence,
                ))

    # ── Check 2: Last EC claimant ↔ Sale deed seller ──────────
    if ec_claimants and sellers:
        # Use the last EC claimant (most recent transaction)
        last_claimant = ec_claimants[-1]
        if not _detect_garbled_tamil(last_claimant)[0]:
            any_match = False
            for seller in sellers:
                if _detect_garbled_tamil(seller)[0]:
                    continue
                sim = _name_similarity_utils(last_claimant, seller)
                if sim >= 0.5:
                    any_match = True
                    break

            if not any_match:
                evidence = (f"Last EC claimant: '{last_claimant}' does not match "
                            f"sale deed seller(s): {', '.join(sellers[:3])} — "
                            f"possible chain-of-title break.")
                checks.append(_make_check(
                    "DET_CHAIN_NAME_GAP",
                    "Chain of Title Name Gap",
                    "HIGH", "WARNING",
                    evidence,
                    "Verify that the sale deed seller acquired title through a valid "
                    "transaction reflected in the EC.",
                    evidence,
                ))

    return checks


# ═══════════════════════════════════════════════════
# 5. SURVEY NUMBER CROSS-CHECK (uses shared utils)
# ═══════════════════════════════════════════════════

def check_survey_number_consistency(extracted_data: dict) -> list[dict]:
    """Cross-check survey numbers across documents with fuzzy matching.

    Uses hierarchy-aware matching (311/1 ⊃ 311/1A) and OCR tolerance
    (Levenshtein ≤ 1) from utils.survey_numbers_match().

    Patta-portfolio aware: A Patta is an *owner-level* document listing ALL
    surveys held by one owner.  Surveys appearing only in the Patta (but in
    no EC / Sale Deed / FMB) are classified as "patta-only" (the owner's
    other holdings) and downgraded to INFO instead of CRITICAL FAIL.  Only
    genuine mismatches among non-Patta documents are flagged as failures.
    """
    checks = []
    # 5-tuple: (original, normalized, filename, survey_type, doc_type)
    survey_records: list[tuple[str, str, str, str, str]] = []

    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "")
        d = data.get("data", {})

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                sn = prop.get("survey_number", "")
                if sn:
                    for s in split_survey_numbers(sn):
                        survey_records.append((s, _normalize_survey_number(s), filename,
                                               extract_survey_type(s), doc_type))

        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
            for sn_obj in (d.get("survey_numbers") or []):
                if isinstance(sn_obj, dict):
                    sn = sn_obj.get("survey_no", "")
                    if sn:
                        survey_records.append((sn, _normalize_survey_number(sn), filename,
                                               extract_survey_type(sn), doc_type))
                elif isinstance(sn_obj, str) and sn_obj:
                    survey_records.append((sn_obj, _normalize_survey_number(sn_obj), filename,
                                           extract_survey_type(sn_obj), doc_type))

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
                    survey_records.append((sn, _normalize_survey_number(sn), filename,
                                           extract_survey_type(sn), doc_type))
            # Also from individual transactions
            seen: set[str] = set()
            for txn in d.get("transactions", []):
                sn = txn.get("survey_number", "")
                if sn and sn not in seen:
                    seen.add(sn)
                    survey_records.append((sn, _normalize_survey_number(sn), filename,
                                           extract_survey_type(sn), doc_type))

        else:
            # FMB, A-certificate, etc.
            sn = d.get("survey_number", "")
            if sn:
                for s in split_survey_numbers(sn):
                    survey_records.append((s, _normalize_survey_number(s), filename,
                                           extract_survey_type(s), doc_type))

    _trace(f"SURVEY_CHECK {len(survey_records)} records: "
           f"{[(orig, fn, dt) for orig, _, fn, _, dt in survey_records]}")

    # ── Intra-EC consistency: header vs transactions ──
    ec_files: dict[str, list[tuple[str, str]]] = {}  # filename → [(orig, norm)]
    for orig, norm, fn, _st, _dt in survey_records:
        for _filename, _data in extracted_data.items():
            if _filename == fn and _data.get("document_type") == "EC":
                ec_files.setdefault(fn, []).append((orig, norm))
                break
    for fn, recs in ec_files.items():
        if len(recs) < 2:
            continue
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

    # ── Build equivalence clusters (5-tuple aware) ──
    clusters: list[list[tuple[str, str, str, str, str]]] = []
    for orig, norm, fn, stype, dtype in survey_records:
        if not norm:
            continue
        placed = False
        for cluster in clusters:
            for c_orig, _c_norm, _c_fn, _c_st, _c_dt in cluster:
                matched, _mtype = survey_numbers_match(orig, c_orig)
                if matched:
                    cluster.append((orig, norm, fn, stype, dtype))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            clusters.append([(orig, norm, fn, stype, dtype)])

    _trace(f"SURVEY_CLUSTERS {len(clusters)} cluster(s): "
           f"{[[orig for orig, *_ in c] for c in clusters]}")

    if len(clusters) <= 1:
        if clusters:
            types_seen = {st for _, _, _, st, _ in clusters[0] if st}
            if len(types_seen) > 1:
                type_details = []
                for orig, _, fn, st, _ in clusters[0]:
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

    # ── Within-cluster match-type analysis (subdivision / OCR fuzzy) ──
    subdivision_pairs = []
    ocr_pairs = []
    for cluster in clusters:
        for i, (oa, _na, fa, _sa, _da) in enumerate(cluster):
            for ob, _nb, fb, _sb, _db in cluster[i + 1:]:
                _m, mtype = survey_numbers_match(oa, ob)
                if mtype == "subdivision":
                    subdivision_pairs.append((oa, ob, fa, fb))
                elif mtype == "ocr_fuzzy":
                    ocr_pairs.append((oa, ob, fa, fb))

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

    for oa, ob, fa, fb in ocr_pairs:
        checks.append(_make_check(
            "DET_SURVEY_OCR_FUZZY", "Survey Number Near-Match (Possible OCR Error)",
            "HIGH", "WARNING",
            f"Survey numbers '{oa}' (in {fa}) and '{ob}' (in {fb}) differ by only "
            f"one character. This may be an OCR/data-entry error.",
            "Verify the correct survey number from the original document.",
            f"{fa}('{oa}') ↔ {fb}('{ob}'): Levenshtein ≤ 1",
        ))

    # ═══ Patta-portfolio-aware mismatch classification ═══
    #
    # Classify each cluster:
    #   "shared"     — contains ≥1 record from a non-Patta doc
    #   "patta_only" — contains ONLY Patta/Chitta records
    #
    # FAIL only when:
    #   • >1 shared cluster exists (non-Patta docs disagree with each other)
    # INFO when:
    #   • patta-only clusters exist (owner's other holdings, not the
    #     property under due diligence)

    _PATTA_TYPES = {"PATTA", "CHITTA", "A_REGISTER"}
    shared_clusters: list[list[tuple]] = []
    patta_only_clusters: list[list[tuple]] = []

    for cluster in clusters:
        doc_types_in_cluster = {dt for _, _, _, _, dt in cluster}
        if doc_types_in_cluster - _PATTA_TYPES:
            # At least one non-Patta record
            shared_clusters.append(cluster)
        else:
            patta_only_clusters.append(cluster)

    _trace(f"SURVEY_CLASSIFY shared={len(shared_clusters)} patta_only={len(patta_only_clusters)}")

    # Emit INFO for patta-only clusters (owner's other surveys)
    if patta_only_clusters:
        po_details = []
        for cl in patta_only_clusters:
            nums = sorted({orig for orig, *_ in cl})
            files = sorted({fn for _, _, fn, _, _ in cl})
            po_details.append(f"{', '.join(files)}: {{{', '.join(nums)}}}")
        checks.append(_make_check(
            "DET_PATTA_ONLY_SURVEYS",
            "Patta Lists Additional Surveys (Owner Portfolio)",
            "LOW", "INFO",
            f"The Patta lists {sum(len(cl) for cl in patta_only_clusters)} survey(s) "
            f"that do not appear in any other document (EC, Sale Deed, FMB). "
            f"This is expected — a Patta is an owner-level document covering the "
            f"owner's entire land portfolio, not just the property under due diligence.",
            "No action required. These are the owner's other land holdings.",
            " | ".join(po_details[:5]),
        ))

    # Only flag CRITICAL FAIL when non-Patta documents genuinely disagree
    if len(shared_clusters) > 1:
        all_types = {st for cl in shared_clusters for _, _, _, st, _ in cl if st}
        type_note = ""
        if len(all_types) > 1:
            type_note = (f" Note: documents use different survey type prefixes "
                         f"({', '.join(sorted(all_types))}), which may explain the "
                         f"difference if old/new survey numbers haven't been mapped.")

        detail_parts = []
        for i, cluster in enumerate(shared_clusters):
            entries = []
            for orig, _n, fn, st, _dt in cluster:
                label = f"{fn}('{orig}')"
                if st:
                    label = f"{fn}('{orig}' [{st}])"
                entries.append(label)
            detail_parts.append(f"Group {i + 1}: {', '.join(entries)}")

        checks.append(_make_check(
            "DET_SURVEY_MISMATCH", "Survey Number Inconsistency Across Documents",
            "CRITICAL", "FAIL",
            f"Found {len(shared_clusters)} distinct survey number group(s) across "
            f"non-Patta documents. Documents for the same property should reference "
            f"the same survey number (accounting for subdivisions).{type_note}",
            "Verify the correct survey number from the village revenue records. "
            "Mismatch may indicate wrong documents or mixed-up properties."
            + (" Check the old survey ↔ new survey mapping at the taluk office."
               if type_note else ""),
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
                    "row": txn.get("transaction_id", txn.get("row_number", "?")),
                    "doc_no": doc_no,
                })

            if ttype == "MORTGAGE" and amount and amount > 0:
                mortgages.append({
                    "amount": amount, "date": date, "doc_no": doc_no,
                    "row": txn.get("transaction_id", txn.get("row_number", "?")),
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
            def _fmt(v: float) -> str:
                if v >= 1_00_00_000:
                    return f"₹{v/1_00_00_000:.1f} Cr"
                elif v >= 1_00_000:
                    return f"₹{v/1_00_000:.1f} L"
                return f"₹{v:,.0f}"

            for i in range(1, len(valued_txns)):
                prev = valued_txns[i - 1]
                curr = valued_txns[i]
                if prev["amount"] > 0 and curr["amount"] > 0:
                    ratio = curr["amount"] / prev["amount"]
                    inv_ratio = prev["amount"] / curr["amount"]
                    jump_ratio = max(ratio, inv_ratio)
                    if jump_ratio >= 10:
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
# 8b. DETERMINISTIC SRO JURISDICTION CHECK
# ═══════════════════════════════════════════════════

def check_sro_jurisdiction(extracted_data: dict) -> list[dict]:
    """Deterministic SRO jurisdiction check using the known SRO→District mapping.

    The LLM's SRO_JURISDICTION check frequently fires false positives (e.g.,
    confusing the seller's residential address with the property location).
    This deterministic check calls the same verify_sro_jurisdiction() tool
    and, when the mapping is conclusive, emits a PASS or FAIL that supersedes
    the LLM result via the dedup mechanism.
    """
    from app.pipeline.tools import verify_sro_jurisdiction

    checks: list[dict] = []

    # Collect SRO, district, and village from extracted data
    sro_values: list[str] = []
    district_values: list[str] = []
    village_values: list[str] = []

    for filename, entry in extracted_data.items():
        doc_type = entry.get("document_type", "")
        d = entry.get("data", {})

        if doc_type == "SALE_DEED":
            prop = d.get("property", {}) or {}
            sro = d.get("sro") or d.get("registration_office") or ""
            if not sro:
                # Try to extract SRO from document_number (e.g. R/Vadavalli/...)
                doc_num = d.get("document_number", "") or ""
                parts = doc_num.split("/")
                if len(parts) >= 2 and parts[0] in ("R", "r"):
                    sro = parts[1]
            if sro and isinstance(sro, str):
                sro_values.append(sro.strip())
            dist = prop.get("district", "")
            if dist and isinstance(dist, str):
                district_values.append(dist.strip())
            vill = prop.get("village", "")
            if vill and isinstance(vill, str):
                village_values.append(vill.strip())

        elif doc_type == "EC":
            sro = d.get("sro") or d.get("issuing_office") or ""
            if sro and isinstance(sro, str):
                sro_values.append(sro.strip())
            dist = d.get("district", "")
            if dist and isinstance(dist, str):
                district_values.append(dist.strip())
            vill = d.get("village", "")
            if vill and isinstance(vill, str):
                village_values.append(vill.strip())

        elif doc_type in ("PATTA", "A_REGISTER"):
            dist = d.get("district", "")
            if dist and isinstance(dist, str):
                district_values.append(dist.strip())
            vill = d.get("village", "")
            if vill and isinstance(vill, str):
                village_values.append(vill.strip())

    if not sro_values or not district_values:
        return checks

    # Use the first non-empty SRO and district
    sro = sro_values[0]
    district = district_values[0]
    village = village_values[0] if village_values else ""

    try:
        result = verify_sro_jurisdiction(
            sro_name=sro, district=district, village=village,
        )
    except Exception as e:
        logger.warning(f"check_sro_jurisdiction: tool call failed: {e}")
        return checks

    valid = result.get("jurisdiction_valid")
    confidence = (result.get("confidence") or "").lower()

    if valid is True and confidence in ("high", "medium"):
        checks.append(_make_check(
            "DET_SRO_JURISDICTION",
            "SRO Jurisdiction Check (Deterministic)",
            "CRITICAL", "PASS",
            f"SRO '{sro}' is confirmed to have jurisdiction over "
            f"{village or district} ({district} district). "
            f"Confidence: {confidence}. {result.get('note', '')}",
            "No action needed — SRO jurisdiction verified.",
            f"SRO={sro}, District={district}, Village={village}, "
            f"jurisdiction_valid=True, confidence={confidence}",
        ))
    elif valid is False and confidence == "high":
        checks.append(_make_check(
            "DET_SRO_JURISDICTION",
            "SRO Jurisdiction Check (Deterministic)",
            "CRITICAL", "FAIL",
            f"SRO '{sro}' does NOT have jurisdiction over {district} district. "
            f"{result.get('note', '')}",
            "Verify that documents are registered at the correct SRO "
            "for the property's location.",
            f"SRO={sro}, District={district}, Village={village}, "
            f"jurisdiction_valid=False, confidence={confidence}",
        ))
    # Low-confidence / inconclusive: don't emit — let LLM handle

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

        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
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
        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
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
        elif doc_type in ("PATTA", "CHITTA", "A_REGISTER"):
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
# 9. BOUNDARY ADJACENCY CHECK — FMB ↔ Sale Deed
# ═══════════════════════════════════════════════════

# Physical features that constitute undisclosed encumbrances / risk
_PHYSICAL_FEATURES = [
    "water channel", "canal", "nala", "nallah", "drainage", "drain",
    "river", "stream", "tank", "pond", "lake", "well",
    "road", "pathway", "cart track", "footpath", "highway",
    "railway", "rail line", "power line", "ht line", "electric",
    "government land", "govt land", "poramboke", "poramboku",
    "burial ground", "cremation", "temple", "church", "mosque",
    "sewage", "sewer", "pipeline",
]

# Tamil equivalents of physical features
_PHYSICAL_FEATURES_TAMIL = [
    "வாய்க்கால்", "கால்வாய்", "நீர்வழி", "ஓடை",
    "சாலை", "பாதை", "தெரு", "ரோடு",
    "ரயில்", "மின்சார", "அரசு நிலம்", "புறம்போக்கு",
    "குளம்", "ஏரி", "கிணறு", "ஆறு",
]

_DIRECTIONS = ["north", "south", "east", "west"]


def _normalize_boundary(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for boundary comparison."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _has_physical_feature(text: str) -> list[str]:
    """Return list of physical feature keywords found in the text."""
    norm = _normalize_boundary(text)
    found = []
    for feat in _PHYSICAL_FEATURES:
        if feat in norm:
            found.append(feat)
    # Check Tamil features too
    if isinstance(text, str):
        for feat in _PHYSICAL_FEATURES_TAMIL:
            if feat in text:
                found.append(feat)
    return found


def check_boundary_adjacency(extracted_data: dict) -> list[dict]:
    """Cross-check boundaries between FMB and Sale Deed.

    Detects:
      1. Contradictory boundaries (same direction has incompatible descriptions)
      2. Undisclosed physical encumbrances (FMB shows water channel / road /
         Government land that Sale Deed boundary omits)
    """
    checks: list[dict] = []

    # Collect boundaries from Sale Deed(s) and FMB(s)
    sd_boundaries: list[tuple[str, dict]] = []  # (filename, {north, south, east, west})
    fmb_boundaries: list[tuple[str, dict]] = []

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        doc_type = doc.get("document_type", "")
        d = doc.get("data", {})

        if doc_type == "SALE_DEED":
            prop = d.get("property", {})
            if isinstance(prop, dict):
                bounds = prop.get("boundaries", {})
                if isinstance(bounds, dict) and any(bounds.get(dir_) for dir_ in _DIRECTIONS):
                    sd_boundaries.append((filename, bounds))

        elif doc_type == "FMB":
            bounds = d.get("boundaries", {})
            if isinstance(bounds, dict) and any(bounds.get(dir_) for dir_ in _DIRECTIONS):
                fmb_boundaries.append((filename, bounds))

    # If we don't have both FMB and Sale Deed boundaries, skip
    if not sd_boundaries or not fmb_boundaries:
        return checks

    # Compare each FMB ↔ Sale Deed pair
    for sd_fn, sd_bounds in sd_boundaries:
        for fmb_fn, fmb_bounds in fmb_boundaries:
            mismatches: list[str] = []

            for direction in _DIRECTIONS:
                sd_val = _normalize_boundary(sd_bounds.get(direction, ""))
                fmb_val = _normalize_boundary(fmb_bounds.get(direction, ""))

                if not sd_val or not fmb_val:
                    continue

                # Check for contradictory descriptions:
                # Road vs owner name, river vs survey number, etc.
                sd_features = _has_physical_feature(sd_bounds.get(direction, ""))
                fmb_features = _has_physical_feature(fmb_bounds.get(direction, ""))

                # If both have content but one mentions a physical feature the other
                # doesn't, and the other mentions a person/survey instead — flag
                if fmb_features and not sd_features:
                    # FMB shows physical feature, Sale Deed doesn't mention it at all
                    # This is a boundary description mismatch
                    mismatches.append(
                        f"{direction.title()}: FMB shows '{fmb_bounds.get(direction, '')}' "
                        f"(contains {', '.join(fmb_features)}), "
                        f"Sale Deed shows '{sd_bounds.get(direction, '')}'"
                    )

            if mismatches:
                checks.append(_make_check(
                    "DET_BOUNDARY_MISMATCH",
                    "FMB ↔ Sale Deed Boundary Mismatch",
                    "HIGH", "WARNING",
                    f"Boundary descriptions differ between FMB and Sale Deed: "
                    f"{'; '.join(mismatches)}. "
                    f"FMB is the government survey record and is more authoritative "
                    f"for physical features.",
                    "Physically inspect the property to confirm actual boundaries. "
                    "Obtain a recent FMB sketch and verify all four boundaries "
                    "match the Sale Deed schedule.",
                    f"[{sd_fn} vs {fmb_fn}] {len(mismatches)} direction(s) differ",
                ))

            # ── Check 2: Undisclosed physical encumbrances from FMB ──
            all_fmb_features: list[str] = []
            all_sd_features: list[str] = []
            for direction in _DIRECTIONS:
                all_fmb_features.extend(_has_physical_feature(fmb_bounds.get(direction, "")))
                all_sd_features.extend(_has_physical_feature(sd_bounds.get(direction, "")))

            # Features in FMB but not in Sale Deed = undisclosed
            undisclosed = set(all_fmb_features) - set(all_sd_features)
            if undisclosed:
                checks.append(_make_check(
                    "DET_UNDISCLOSED_ENCUMBRANCE",
                    "Physical Feature in FMB Not Disclosed in Sale Deed",
                    "MEDIUM", "WARNING",
                    f"FMB boundaries mention physical feature(s) not found in "
                    f"Sale Deed: {', '.join(sorted(undisclosed))}. "
                    f"Features like water channels, roads, or Government land "
                    f"adjacent to the property may affect usage, access, or value.",
                    "Verify actual boundary conditions with a site inspection. "
                    "Confirm whether the physical feature(s) create any easement "
                    "rights or access restrictions.",
                    f"[{fmb_fn}] undisclosed features: {', '.join(sorted(undisclosed))}",
                ))

    return checks


# ═══════════════════════════════════════════════════
# 10. CONSIDERATION CONSISTENCY — Sale Deed ↔ EC
# ═══════════════════════════════════════════════════

def check_consideration_consistency(extracted_data: dict) -> list[dict]:
    """Match Sale Deed consideration with EC transaction by document number.

    Finds the Sale Deed's document_number in EC transactions and compares
    the consideration amounts with a 5% tolerance (to account for rounding
    and extraction variations).
    """
    checks: list[dict] = []

    # Collect Sale Deed info
    sd_docs: list[tuple[str, str, float]] = []  # (filename, doc_number, amount)
    ec_transactions: list[tuple[str, dict]] = []  # (filename, txn_dict)

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        doc_type = doc.get("document_type", "")
        d = doc.get("data", {})

        if doc_type == "SALE_DEED":
            doc_num = str(d.get("document_number", "")).strip()
            financials = d.get("financials", {})
            if isinstance(financials, dict):
                amt = _parse_amount(financials.get("consideration_amount"))
                if doc_num and amt and amt > 0:
                    sd_docs.append((filename, doc_num, amt))

        elif doc_type == "EC":
            for txn in d.get("transactions", []):
                if isinstance(txn, dict):
                    ec_transactions.append((filename, txn))

    if not sd_docs or not ec_transactions:
        return checks

    # For each Sale Deed, try to find matching EC transaction by doc number
    for sd_fn, sd_doc_num, sd_amt in sd_docs:
        # Normalize doc number: strip year suffix, leading zeros
        sd_num_norm = re.sub(r'[/\-\s]', '', sd_doc_num.lower().strip())

        matched = False
        for ec_fn, txn in ec_transactions:
            txn_doc_num = str(txn.get("document_number", "")).strip()
            if not txn_doc_num:
                continue
            txn_num_norm = re.sub(r'[/\-\s]', '', txn_doc_num.lower().strip())

            # Match: exact or partial (one contains the other)
            if sd_num_norm == txn_num_norm or sd_num_norm in txn_num_norm or txn_num_norm in sd_num_norm:
                matched = True
                ec_amt = _parse_amount(txn.get("consideration_amount"))
                if ec_amt and ec_amt > 0:
                    # Compare with 5% tolerance
                    diff = abs(sd_amt - ec_amt)
                    tolerance = max(sd_amt, ec_amt) * 0.05
                    if diff > tolerance:
                        pct = (diff / max(sd_amt, ec_amt)) * 100
                        checks.append(_make_check(
                            "DET_CONSIDERATION_MISMATCH",
                            "Sale Deed ↔ EC Consideration Mismatch",
                            "HIGH", "FAIL",
                            f"Sale Deed (Doc #{sd_doc_num}) shows consideration ₹{sd_amt:,.0f} "
                            f"but the matching EC transaction (Doc #{txn_doc_num}) shows "
                            f"₹{ec_amt:,.0f} — a {pct:.1f}% difference. "
                            f"These should match as they record the same transaction.",
                            "Verify both documents against the original registration record "
                            "at the SRO. A mismatch may indicate extraction error or "
                            "undervaluation in one record.",
                            f"[{sd_fn} vs {ec_fn}] SD=₹{sd_amt:,.0f}, EC=₹{ec_amt:,.0f}, "
                            f"diff={pct:.1f}%",
                        ))
                break  # Only need first match per Sale Deed

        if not matched and sd_doc_num:
            _trace(f"CONSIDERATION: SD doc #{sd_doc_num} not found in EC transactions")

    return checks


# ═══════════════════════════════════════════════════
# 11. PAN CROSS-VERIFICATION
# ═══════════════════════════════════════════════════

_PAN_PATTERN = re.compile(r'^[A-Z]{5}[0-9]{4}[A-Z]$')


def check_pan_consistency(extracted_data: dict) -> list[dict]:
    """Verify PAN numbers across Sale Deed parties.

    Checks:
      1. PAN format validity (ABCDE1234F)
      2. Duplicate PANs across different parties (same PAN on seller + buyer = red flag)
      3. PAN availability — flags if parties lack PAN (advisory only)
    """
    checks: list[dict] = []

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        doc_type = doc.get("document_type", "")
        if doc_type != "SALE_DEED":
            continue
        d = doc.get("data", {})

        # Collect all PANs with their party roles
        pan_registry: dict[str, list[str]] = {}  # PAN → [list of "role: name"]
        parties_without_pan: list[str] = []
        invalid_pans: list[tuple[str, str]] = []  # (party_label, pan_value)

        for role, party_list in [("Seller", d.get("seller", [])),
                                  ("Buyer", d.get("buyer", []))]:
            if not isinstance(party_list, list):
                continue
            for i, party in enumerate(party_list):
                if not isinstance(party, dict):
                    continue
                name = party.get("name", f"Unknown {role} {i+1}")
                pan = str(party.get("pan", "")).strip().upper()
                party_label = f"{role}: {name}"

                if not pan or pan in ("", "NONE", "N/A", "NA", "NOT AVAILABLE",
                                       "NOT PROVIDED", "NIL"):
                    parties_without_pan.append(party_label)
                    continue

                # Validate format
                if not _PAN_PATTERN.match(pan):
                    invalid_pans.append((party_label, pan))
                else:
                    pan_registry.setdefault(pan, []).append(party_label)

        # Check 1: Invalid PAN formats
        if invalid_pans:
            for party_label, pan_val in invalid_pans:
                checks.append(_make_check(
                    "DET_PAN_FORMAT_INVALID",
                    "Invalid PAN Format",
                    "MEDIUM", "WARNING",
                    f"PAN '{pan_val}' for {party_label} does not match the standard "
                    f"Indian PAN format (ABCDE1234F — 5 letters, 4 digits, 1 letter). "
                    f"This may indicate an extraction error or invalid PAN.",
                    "Verify the PAN from the original Sale Deed document.",
                    f"[{filename}] {party_label}: '{pan_val}'",
                ))

        # Check 2: Duplicate PANs across different parties
        for pan, parties in pan_registry.items():
            if len(parties) >= 2:
                # Check if duplicate is across roles (seller+buyer = fraud risk)
                roles = set(p.split(":")[0].strip() for p in parties)
                if len(roles) >= 2:
                    # Same PAN on both seller and buyer side — high risk
                    checks.append(_make_check(
                        "DET_PAN_DUPLICATE_CROSS",
                        "Same PAN on Seller and Buyer",
                        "HIGH", "FAIL",
                        f"PAN {pan} appears on both sides of the transaction: "
                        f"{', '.join(parties)}. This is a serious red flag — "
                        f"the same person cannot be both seller and buyer.",
                        "Investigate the transaction for possible fraud, benami "
                        "transaction, or data extraction error.",
                        f"[{filename}] PAN={pan}: {', '.join(parties)}",
                    ))
                else:
                    # Same PAN on same side (two sellers with same PAN)
                    checks.append(_make_check(
                        "DET_PAN_DUPLICATE",
                        "Duplicate PAN Within Same Party",
                        "MEDIUM", "WARNING",
                        f"PAN {pan} is shared by multiple parties on the same side: "
                        f"{', '.join(parties)}. This may indicate a data entry error "
                        f"or extraction issue.",
                        "Verify each party's PAN from the original document.",
                        f"[{filename}] PAN={pan}: {', '.join(parties)}",
                    ))

        # Check 3: Missing PANs (advisory) — only flag if some have PAN but others don't
        if parties_without_pan and pan_registry:
            checks.append(_make_check(
                "DET_PAN_MISSING",
                "Some Parties Lack PAN",
                "LOW", "INFO",
                f"{len(parties_without_pan)} party/parties have no PAN recorded: "
                f"{', '.join(parties_without_pan[:5])}. "
                f"PAN is mandatory for property transactions above ₹10 lakh.",
                "Verify PAN details from the original Sale Deed or request "
                "PAN cards from the parties for verification.",
                f"[{filename}] {len(parties_without_pan)} missing, "
                f"{len(pan_registry)} present",
            ))

    return checks


# ═══════════════════════════════════════════════════
# 12. PRE-EC PERIOD GAP CHECK
# ═══════════════════════════════════════════════════

def check_pre_ec_gap(extracted_data: dict) -> list[dict]:
    """Flag when the Sale Deed references ownership history before EC start.

    If the Sale Deed's previous_ownership document_date or earliest
    ownership_history entry is significantly before the EC period_from,
    the title chain before the EC start is unverified by the EC.
    """
    checks: list[dict] = []

    # Collect EC period start
    ec_starts: list[tuple[str, datetime]] = []
    # Collect Sale Deed previous ownership dates
    sd_prev_dates: list[tuple[str, datetime, str]] = []  # (fn, date, context)

    for filename, doc in extracted_data.items():
        if not isinstance(doc, dict):
            continue
        doc_type = doc.get("document_type", "")
        d = doc.get("data", {})

        if doc_type == "EC":
            pf = _parse_date(d.get("period_from", ""))
            if pf:
                ec_starts.append((filename, pf))

        elif doc_type == "SALE_DEED":
            # Check previous_ownership.document_date
            prev = d.get("previous_ownership", {})
            if isinstance(prev, dict):
                prev_date_str = prev.get("document_date", "")
                prev_date = _parse_date(prev_date_str)
                if prev_date:
                    prev_owner = prev.get("previous_owner", "unknown")
                    sd_prev_dates.append((
                        filename, prev_date,
                        f"previous_ownership doc #{prev.get('document_number', '?')} "
                        f"dated {prev_date_str} (owner: {prev_owner})"
                    ))

            # Check ownership_history entries
            history = d.get("ownership_history", [])
            if isinstance(history, list):
                for entry in history:
                    if not isinstance(entry, dict):
                        continue
                    doc_date = _parse_date(entry.get("document_date", ""))
                    if doc_date:
                        sd_prev_dates.append((
                            filename, doc_date,
                            f"ownership_history: {entry.get('owner', '?')} acquired via "
                            f"{entry.get('acquisition_mode', '?')} on {entry.get('document_date', '?')}"
                        ))

    if not ec_starts or not sd_prev_dates:
        return checks

    # Use the earliest EC start
    earliest_ec_fn, earliest_ec_start = min(ec_starts, key=lambda x: x[1])

    # Check if any previous ownership date falls before EC start
    for sd_fn, prev_date, context in sd_prev_dates:
        gap_days = (earliest_ec_start - prev_date).days
        if gap_days > 365:  # More than 1 year before EC start
            gap_years = gap_days / 365.25
            checks.append(_make_check(
                "DET_PRE_EC_GAP",
                "Title History Before EC Period",
                "MEDIUM", "WARNING",
                f"Sale Deed references ownership history from {context}, "
                f"which is {gap_years:.1f} years before the EC start date "
                f"({earliest_ec_start.strftime('%d-%m-%Y')}). "
                f"The title chain during this gap period is NOT verified by "
                f"the EC and could conceal prior encumbrances or disputes.",
                f"Obtain a supplementary EC from the SRO covering the period "
                f"from {prev_date.strftime('%Y')} to "
                f"{earliest_ec_start.strftime('%Y')} to close the gap.",
                f"[{sd_fn}] prev_date={prev_date.strftime('%d-%m-%Y')}, "
                f"ec_start={earliest_ec_start.strftime('%d-%m-%Y')}, "
                f"gap={gap_years:.1f}yr",
            ))

    return checks


# ═══════════════════════════════════════════════════
# CHAIN OF TITLE BUILDER
# ═══════════════════════════════════════════════════

# Acquisition-mode label → canonical TRANSACTION_TYPES key
_ACQUISITION_TO_TXN_TYPE = {
    "sale": "SALE", "gift": "GIFT", "inheritance": "INHERITANCE",
    "partition": "PARTITION", "settlement": "SETTLEMENT", "unknown": "OTHER",
    "will": "WILL", "release": "RELEASE", "exchange": "EXCHANGE",
    "relinquishment": "RELINQUISHMENT", "adoption": "ADOPTION",
}


def _join_party_names(parties: list | str) -> str:
    """Join party names from a list-of-dicts, list-of-strings, or a raw string."""
    if isinstance(parties, str):
        return parties.strip()
    if isinstance(parties, list):
        names = []
        for p in parties:
            if isinstance(p, dict):
                n = p.get("name", "")
            else:
                n = str(p)
            n = n.strip()
            if n:
                names.append(n)
        return ", ".join(names)
    return ""


def _chain_link(*, from_party: str, to_party: str, date: str = "",
                transaction_type: str = "", document_number: str = "",
                source: str = "", valid: bool = True, notes: str = "",
                transaction_id: str = "") -> dict | None:
    """Build a single chain-link dict.  Returns None if from/to is empty or 'Unknown'."""
    f = (from_party or "").strip()
    t = (to_party or "").strip()
    if not f or not t:
        return None
    # Filter out placeholder names that produce meaningless chain links
    if f.lower() in ("unknown", "not available", "n/a", "na", "-"):
        return None
    if t.lower() in ("unknown", "not available", "n/a", "na", "-"):
        return None
    return {
        "sequence": 0,  # assigned later
        "date": (date or "").strip(),
        "from": f,
        "to": t,
        "transaction_type": (transaction_type or "OTHER").strip().upper(),
        "document_number": (document_number or "").strip(),
        "valid": valid,
        "notes": (notes or "").strip(),
        "source": source,
        "transaction_id": (transaction_id or "").strip(),
    }


def _extract_ec_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from EC transactions."""
    links: list[dict] = []
    for txn in (data.get("transactions") or []):
        txn_type = (txn.get("transaction_type") or "").strip().lower()
        if txn_type not in CHAIN_RELEVANT_TYPES:
            continue
        doc_num = txn.get("document_number", "")
        doc_year = txn.get("document_year", "")
        doc_ref = f"{doc_num}/{doc_year}" if doc_num and doc_year else doc_num
        link = _chain_link(
            from_party=txn.get("seller_or_executant", ""),
            to_party=txn.get("buyer_or_claimant", ""),
            date=txn.get("date", ""),
            transaction_type=txn_type.upper(),
            document_number=doc_ref,
            source="EC",
            transaction_id=txn.get("transaction_id", ""),
        )
        if link:
            links.append(link)
    return links


def _extract_sale_deed_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from Sale Deed — the sale itself + previous_ownership + ownership_history."""
    links: list[dict] = []
    sellers = _join_party_names(data.get("seller", []))
    buyers = _join_party_names(data.get("buyer", []))

    # The main sale transaction
    link = _chain_link(
        from_party=sellers,
        to_party=buyers,
        date=data.get("registration_date", data.get("execution_date", "")),
        transaction_type="SALE",
        document_number=data.get("document_number", ""),
        source="Sale Deed",
    )
    if link:
        links.append(link)

    # previous_ownership (the acquisition that preceded this sale)
    prev = data.get("previous_ownership") or {}
    if prev.get("previous_owner"):
        # The previous owner transferred to the first seller
        first_seller = sellers.split(",")[0].strip() if sellers else ""
        mode = _ACQUISITION_TO_TXN_TYPE.get(
            (prev.get("acquisition_mode") or "Unknown").lower(), "OTHER"
        )
        link = _chain_link(
            from_party=prev["previous_owner"],
            to_party=first_seller,
            date=prev.get("document_date", ""),
            transaction_type=mode,
            document_number=prev.get("document_number", ""),
            source="Sale Deed",
            notes="Previous ownership link",
        )
        if link:
            links.append(link)

    # ownership_history
    for hist in (data.get("ownership_history") or []):
        acquired_from = (hist.get("acquired_from") or "").strip()
        owner = (hist.get("owner") or "").strip()
        if not acquired_from or not owner:
            continue
        mode = _ACQUISITION_TO_TXN_TYPE.get(
            (hist.get("acquisition_mode") or "Unknown").lower(), "OTHER"
        )
        link = _chain_link(
            from_party=acquired_from,
            to_party=owner,
            date=hist.get("document_date", ""),
            transaction_type=mode,
            document_number=hist.get("document_number", ""),
            source="Sale Deed",
            notes="Ownership history",
        )
        if link:
            links.append(link)

    return links


def _extract_a_register_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from A-Register mutation entries."""
    links: list[dict] = []
    for mut in (data.get("mutation_entries") or []):
        from_owner = (mut.get("from_owner") or "").strip()
        to_owner = (mut.get("to_owner") or "").strip()
        if not to_owner:
            continue
        # If from_owner missing, use the register's primary owner
        if not from_owner:
            from_owner = (data.get("owner_name") or "").strip()
        if not from_owner:
            continue
        reason = (mut.get("reason") or "OTHER").strip()
        mode = _ACQUISITION_TO_TXN_TYPE.get(reason.lower(), reason.upper())
        link = _chain_link(
            from_party=from_owner,
            to_party=to_owner,
            date=mut.get("date", ""),
            transaction_type=mode,
            document_number=mut.get("order_number", ""),
            source="A-Register",
        )
        if link:
            links.append(link)
    return links


def _extract_gift_deed_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain link from Gift Deed — donor → donee."""
    donor = data.get("donor") or {}
    donee = data.get("donee") or {}
    link = _chain_link(
        from_party=donor.get("name", ""),
        to_party=donee.get("name", ""),
        date=data.get("registration_date", ""),
        transaction_type="GIFT",
        document_number=data.get("registration_number", ""),
        source="Gift Deed",
    )
    return [link] if link else []


def _extract_partition_deed_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from Partition Deed — joint owners → each partitioned share."""
    links: list[dict] = []
    joint_names = _join_party_names(data.get("joint_owners", []))
    if not joint_names:
        return links
    for share in (data.get("partitioned_shares") or []):
        to_name = (share.get("name") or "").strip()
        if not to_name:
            continue
        link = _chain_link(
            from_party=joint_names,
            to_party=to_name,
            date=data.get("registration_date", ""),
            transaction_type="PARTITION",
            document_number=data.get("registration_number", ""),
            source="Partition Deed",
        )
        if link:
            links.append(link)
    return links


def _extract_release_deed_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain link from Release Deed — releasing party → beneficiary."""
    rel = data.get("releasing_party") or {}
    ben = data.get("beneficiary") or {}
    link = _chain_link(
        from_party=rel.get("name", ""),
        to_party=ben.get("name", ""),
        date=data.get("registration_date", ""),
        transaction_type="RELEASE",
        document_number=data.get("registration_number", ""),
        source="Release Deed",
    )
    return [link] if link else []


def _extract_will_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from Will — testator → each beneficiary."""
    links: list[dict] = []
    testator = data.get("testator") or {}
    from_name = (testator.get("name") or "").strip()
    if not from_name:
        return links
    for b in (data.get("beneficiaries") or []):
        to_name = (b.get("name") or "").strip()
        if not to_name:
            continue
        share = b.get("share", "")
        notes = f"Bequest: {share}" if share else ""
        link = _chain_link(
            from_party=from_name,
            to_party=to_name,
            date=data.get("execution_date", data.get("registration_date", "")),
            transaction_type="WILL",
            document_number=data.get("registration_number", ""),
            source="Will",
            notes=notes,
        )
        if link:
            links.append(link)
    return links


def _extract_legal_heir_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain links from Legal Heir Certificate — deceased → each heir."""
    links: list[dict] = []
    from_name = (data.get("deceased_name") or "").strip()
    if not from_name:
        return links
    for heir in (data.get("heirs") or []):
        to_name = (heir.get("name") or "").strip()
        if not to_name:
            continue
        share = heir.get("share_percentage", "")
        rel = heir.get("relationship", "")
        notes_parts = [p for p in [rel, f"{share}%" if share else ""] if p]
        link = _chain_link(
            from_party=from_name,
            to_party=to_name,
            date=data.get("date_of_death", data.get("certificate_date", "")),
            transaction_type="INHERITANCE",
            document_number=data.get("certificate_number", ""),
            source="Legal Heir",
            notes=", ".join(notes_parts),
        )
        if link:
            links.append(link)
    return links


def _extract_court_order_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain link from Court Order — only for decree / attachment types."""
    order_type = (data.get("order_type") or "").strip().lower()
    if order_type not in ("decree", "attachment"):
        return []
    link = _chain_link(
        from_party=data.get("petitioner", ""),
        to_party=data.get("respondent", ""),
        date=data.get("order_date", ""),
        transaction_type="COURT_ORDER",
        document_number=data.get("case_number", ""),
        source="Court Order",
        notes=f"Order type: {order_type}, status: {data.get('status', 'unknown')}",
    )
    return [link] if link else []


def _extract_poa_chain_links(data: dict, filename: str) -> list[dict]:
    """Extract chain link from Power of Attorney — principal → agent."""
    principal = data.get("principal") or {}
    agent = data.get("agent") or {}
    link = _chain_link(
        from_party=principal.get("name", "") if isinstance(principal, dict) else str(principal),
        to_party=agent.get("name", "") if isinstance(agent, dict) else str(agent),
        date=data.get("registration_date", ""),
        transaction_type="POWER_OF_ATTORNEY",
        document_number=data.get("registration_number", ""),
        source="POA",
        notes="Authorisation (not title transfer)",
    )
    return [link] if link else []


# Per-document-type extractor dispatch
_CHAIN_EXTRACTORS: dict[str, callable] = {
    "EC": _extract_ec_chain_links,
    "SALE_DEED": _extract_sale_deed_chain_links,
    "A_REGISTER": _extract_a_register_chain_links,
    "GIFT_DEED": _extract_gift_deed_chain_links,
    "PARTITION_DEED": _extract_partition_deed_chain_links,
    "RELEASE_DEED": _extract_release_deed_chain_links,
    "WILL": _extract_will_chain_links,
    "LEGAL_HEIR": _extract_legal_heir_chain_links,
    "COURT_ORDER": _extract_court_order_chain_links,
    "POA": _extract_poa_chain_links,
}


def _dedup_chain_key(link: dict) -> str:
    """Generate a dedup key from document_number + type + to (to handle partitions)."""
    doc = (link.get("document_number") or "").strip().lower()
    txn = (link.get("transaction_type") or "").strip().lower()
    t = (link.get("to") or "").strip().lower()
    if doc:
        return f"{doc}|{txn}|{t}"
    f = (link.get("from") or "").strip().lower()
    return f"{f}>{t}|{txn}"


def _sort_key_for_link(link: dict) -> tuple:
    """Sort key: (parsed_date or far-future, document_number)."""
    d = _parse_date(link.get("date", ""))
    if d is None:
        # Try extracting a year from the document_number (e.g. "5909/2012")
        doc = link.get("document_number", "")
        m = re.search(r'/((?:19|20)\d{2})$', doc)
        if m:
            d = datetime(int(m.group(1)), 1, 1)
    ts = d.timestamp() if d else 9999999999.0  # unknown dates go last
    return (ts, link.get("document_number", ""))


def build_chain_of_title(
    extracted_data: dict,
    *,
    llm_chain: list[dict] | None = None,
) -> list[dict]:
    """Build a unified chain-of-title from all ownership-transfer documents.

    Extracts chain links from EC transactions, Sale Deed ownership history,
    A-Register mutations, Gift/Partition/Release Deeds, Wills, Legal Heir
    Certificates, Court Orders, and POAs.  Deduplicates, merges with any
    LLM-generated chain (preserving LLM notes/validity), sorts
    chronologically, and assigns sequence numbers.

    Args:
        extracted_data: dict of {filename: {document_type, data}} from pipeline
        llm_chain: optional chain_of_title list from Group 1 LLM verification

    Returns:
        List of chain-link dicts compatible with ``_CHAIN_LINK_SCHEMA``
        (extended with ``source`` and ``transaction_id`` fields).
    """
    raw_links: list[dict] = []

    # ── 1. Extract links from every document ──
    for filename, entry in (extracted_data or {}).items():
        doc_type = entry.get("document_type", "")
        data = entry.get("data")
        if not data or not isinstance(data, dict):
            continue
        extractor = _CHAIN_EXTRACTORS.get(doc_type)
        if extractor:
            try:
                links = extractor(data, filename)
                raw_links.extend(links)
                if links:
                    _trace(f"CHAIN_BUILDER [{doc_type}] [{filename}]: {len(links)} link(s)")
            except Exception as e:
                logger.warning(f"Chain builder: {doc_type} extractor failed for {filename}: {e}")

    # ── 2. Deduplicate deterministic links ──
    seen: dict[str, dict] = {}
    for link in raw_links:
        key = _dedup_chain_key(link)
        if key in seen:
            # Keep the one with more information (longer notes, has date, etc.)
            existing = seen[key]
            if (not existing.get("date") and link.get("date")):
                seen[key] = link
            elif len(link.get("notes", "")) > len(existing.get("notes", "")):
                seen[key] = link
        else:
            seen[key] = link
    deduped = list(seen.values())

    # ── 3. Merge LLM chain links ──
    if llm_chain:
        # Build lookup of deterministic links by dedup key
        det_keys = {_dedup_chain_key(l) for l in deduped}
        for llm_link in llm_chain:
            llm_key = _dedup_chain_key(llm_link)
            if llm_key in det_keys:
                # Find matching det link and merge LLM notes/valid
                for dl in deduped:
                    if _dedup_chain_key(dl) == llm_key:
                        if llm_link.get("notes") and not dl.get("notes"):
                            dl["notes"] = llm_link["notes"]
                        if "valid" in llm_link:
                            dl["valid"] = llm_link["valid"]
                        break
            else:
                # LLM link has no deterministic match — keep it
                deduped.append({
                    "sequence": 0,
                    "date": llm_link.get("date", ""),
                    "from": llm_link.get("from", ""),
                    "to": llm_link.get("to", ""),
                    "transaction_type": llm_link.get("transaction_type", ""),
                    "document_number": llm_link.get("document_number", ""),
                    "valid": llm_link.get("valid", True),
                    "notes": llm_link.get("notes", ""),
                    "source": "LLM",
                    "transaction_id": llm_link.get("transaction_id", ""),
                })

    # ── 4. Sort chronologically and assign sequence numbers ──
    deduped.sort(key=_sort_key_for_link)
    for i, link in enumerate(deduped, 1):
        link["sequence"] = i

    _trace(f"CHAIN_BUILDER total: {len(deduped)} link(s)")
    return deduped


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
        ("Temporal: Pre-EC gap", check_pre_ec_gap),
        ("Financial: Stamp duty", check_stamp_duty),
        ("Financial: Plausibility ranges", check_plausibility_ranges),
        ("Financial: Consideration consistency", check_consideration_consistency),
        ("Property: Area consistency", check_area_consistency),
        ("Property: Survey numbers", check_survey_number_consistency),
        ("Property: Plot identity", check_plot_identity_consistency),
        ("Property: Boundary adjacency", check_boundary_adjacency),
        ("Party: Name consistency", check_party_name_consistency),
        ("Party: Age fraud", check_age_fraud),
        ("Party: PAN consistency", check_pan_consistency),
        ("Pattern: Rapid flipping", check_rapid_flipping),
        ("Pattern: Multiple sales", check_multiple_sales),
        ("Financial: Scale anomalies", check_financial_scale_anomalies),
        ("Geography: Multi-village", check_multi_village),
        ("Geography: SRO jurisdiction", check_sro_jurisdiction),
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
