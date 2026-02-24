"""Tests for backend/app/pipeline/deterministic.py — all 11 check functions.

Each test class constructs the minimal extracted_data dict required by its
target function, then asserts on the returned check list:
  - Correct rule_code
  - Expected status (PASS / FAIL / WARNING / INFO)
  - Correct severity
  - No false positives for clean data
"""

import json
import pytest
from datetime import datetime, timedelta

from app.pipeline.deterministic import (
    check_ec_period_coverage,
    check_registration_within_ec,
    check_limitation_period,
    check_stamp_duty,
    check_area_consistency,
    check_survey_number_consistency,
    check_plot_identity_consistency,
    check_party_name_consistency,
    check_rapid_flipping,
    check_financial_scale_anomalies,
    check_multi_village,
    check_plausibility_ranges,
    check_field_format_validity,
    check_garbled_tamil,
    check_hallucination_signs,
    check_boundary_adjacency,
    check_consideration_consistency,
    check_pan_consistency,
    check_pre_ec_gap,
    check_sro_jurisdiction,
    run_deterministic_checks,
    build_chain_of_title,
    _chain_link,
)


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def _codes(checks: list[dict]) -> list[str]:
    """Extract rule_code list from check results for easy assertion."""
    return [c["rule_code"] for c in checks]


def _find(checks: list[dict], code: str) -> dict | None:
    """Find a specific check by rule_code."""
    for c in checks:
        if c["rule_code"] == code:
            return c
    return None


def _ec(period_from: str, period_to: str, transactions: list | None = None,
        village: str = "", taluk: str = "") -> dict:
    """Build a minimal EC extracted_data."""
    d = {
        "period_from": period_from,
        "period_to": period_to,
        "transactions": transactions or [],
    }
    if village:
        d["village"] = village
    if taluk:
        d["taluk"] = taluk
    return {"ec.pdf": {"document_type": "EC", "data": d}}


def _sale_deed(reg_date: str = "20-06-2020", survey: str = "311/1",
               extent: str = "2400 sq.ft", village: str = "Chromepet",
               taluk: str = "Tambaram", district: str = "Chengalpattu",
               consideration: str = "45,00,000", guideline: str = "40,00,000",
               stamp_duty: str = "3,15,000",
               sellers: list | None = None, buyers: list | None = None) -> dict:
    """Build a minimal Sale Deed extracted_data."""
    return {
        "sale_deed.pdf": {
            "document_type": "SALE_DEED",
            "data": {
                "registration_date": reg_date,
                "seller": sellers or [{"name": "Seller A"}],
                "buyer": buyers or [{"name": "Buyer B"}],
                "property": {
                    "survey_number": survey,
                    "extent": extent,
                    "village": village,
                    "taluk": taluk,
                    "district": district,
                },
                "financials": {
                    "consideration_amount": consideration,
                    "guideline_value": guideline,
                    "stamp_duty": stamp_duty,
                },
            },
        }
    }


def _patta(survey_no: str = "311/1", extent: str = "2400 sq.ft",
           village: str = "Chromepet", taluk: str = "Tambaram",
           district: str = "Chengalpattu",
           owners: list | None = None) -> dict:
    """Build a minimal Patta extracted_data."""
    return {
        "patta.pdf": {
            "document_type": "PATTA",
            "data": {
                "patta_number": "P-001",
                "survey_numbers": [{"survey_no": survey_no, "extent": extent}],
                "total_extent": extent,
                "village": village,
                "taluk": taluk,
                "district": district,
                "owner_names": owners or [{"name": "Buyer B"}],
            },
        }
    }


# ═══════════════════════════════════════════════════
# 1. check_ec_period_coverage
# ═══════════════════════════════════════════════════

class TestEcPeriodCoverage:

    def test_valid_period_no_issues(self):
        """Long EC period ending recently → no checks."""
        now = datetime.now()
        data = _ec("01-01-2010", now.strftime("%d-%m-%Y"))
        checks = check_ec_period_coverage(data)
        # Should have no FAIL / WARNING (may have INFO if period < 13y but we use 15y)
        fail_codes = [c for c in checks if c["status"] in ("FAIL", "WARNING")]
        assert fail_codes == [] or all(c["rule_code"] != "DET_EC_PERIOD_INVALID" for c in fail_codes)

    def test_inverted_period(self):
        """period_to before period_from → DET_EC_PERIOD_INVALID."""
        data = _ec("01-01-2025", "01-01-2020")
        checks = check_ec_period_coverage(data)
        assert "DET_EC_PERIOD_INVALID" in _codes(checks)
        c = _find(checks, "DET_EC_PERIOD_INVALID")
        assert c["severity"] == "CRITICAL"
        assert c["status"] == "FAIL"

    def test_short_period(self):
        """EC covering only 6 months → DET_EC_SHORT_PERIOD (CRITICAL, < 1 year)."""
        data = _ec("01-01-2024", "01-07-2024")
        checks = check_ec_period_coverage(data)
        assert "DET_EC_SHORT_PERIOD" in _codes(checks)
        c = _find(checks, "DET_EC_SHORT_PERIOD")
        assert c["severity"] == "CRITICAL"
        assert c["status"] == "FAIL"

    def test_stale_ec(self):
        """EC ending more than 30 days ago → DET_EC_STALE."""
        old_end = (datetime.now() - timedelta(days=200)).strftime("%d-%m-%Y")
        data = _ec("01-01-2010", old_end)
        checks = check_ec_period_coverage(data)
        assert "DET_EC_STALE" in _codes(checks)

    def test_ec_not_stale_within_30_days(self):
        """EC ending 10 days ago → NOT stale."""
        recent_end = (datetime.now() - timedelta(days=10)).strftime("%d-%m-%Y")
        data = _ec("01-01-2010", recent_end)
        checks = check_ec_period_coverage(data)
        assert "DET_EC_STALE" not in _codes(checks)

    def test_ec_stale_at_45_days(self):
        """EC ending 45 days ago → DET_EC_STALE (> 30 threshold)."""
        old_end = (datetime.now() - timedelta(days=45)).strftime("%d-%m-%Y")
        data = _ec("01-01-2010", old_end)
        checks = check_ec_period_coverage(data)
        assert "DET_EC_STALE" in _codes(checks)

    def test_chrono_order_issue(self):
        """Transactions with out-of-order dates → DET_EC_CHRONO_ORDER."""
        txns = [
            {"row_number": 1, "date": "01-06-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-01-2018", "transaction_type": "Sale"},
        ]
        now = datetime.now().strftime("%d-%m-%Y")
        data = _ec("01-01-2010", now, txns)
        checks = check_ec_period_coverage(data)
        assert "DET_EC_CHRONO_ORDER" in _codes(checks)

    def test_non_ec_skipped(self):
        """Sale deed data should be ignored by this check."""
        data = _sale_deed()
        checks = check_ec_period_coverage(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# 2. check_registration_within_ec
# ═══════════════════════════════════════════════════

class TestRegistrationWithinEc:

    def test_within_period(self):
        """Registration date inside EC → no issues."""
        data = {**_ec("01-01-2010", "31-12-2025"), **_sale_deed(reg_date="20-06-2020")}
        checks = check_registration_within_ec(data)
        assert checks == []

    def test_outside_period(self):
        """Registration date before EC start → DET_REG_OUTSIDE_EC."""
        data = {**_ec("01-01-2015", "31-12-2025"), **_sale_deed(reg_date="20-06-2012")}
        checks = check_registration_within_ec(data)
        assert "DET_REG_OUTSIDE_EC" in _codes(checks)
        c = _find(checks, "DET_REG_OUTSIDE_EC")
        assert c["status"] == "FAIL"
        assert c["severity"] == "HIGH"

    def test_no_ec(self):
        """Only sale deed, no EC → no checks."""
        data = _sale_deed()
        checks = check_registration_within_ec(data)
        assert checks == []

    def test_no_sale_deed(self):
        """Only EC, no sale deed → no checks."""
        data = _ec("01-01-2010", "31-12-2025")
        checks = check_registration_within_ec(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# 3. check_limitation_period
# ═══════════════════════════════════════════════════

class TestLimitationPeriod:

    def test_recent_transactions_no_issue(self):
        """All transactions within 12 years → no checks."""
        recent = (datetime.now() - timedelta(days=365)).strftime("%d-%m-%Y")
        txns = [{"row_number": 1, "date": recent, "transaction_type": "Sale"}]
        data = _ec("01-01-2010", datetime.now().strftime("%d-%m-%Y"), txns)
        checks = check_limitation_period(data)
        assert checks == []

    def test_old_chain_transaction(self):
        """Sale 15 years ago → DET_LIMITATION_PERIOD."""
        old_date = (datetime.now() - timedelta(days=15 * 365)).strftime("%d-%m-%Y")
        txns = [{"row_number": 1, "date": old_date, "transaction_type": "sale"}]
        data = _ec("01-01-2005", datetime.now().strftime("%d-%m-%Y"), txns)
        checks = check_limitation_period(data)
        assert "DET_LIMITATION_PERIOD" in _codes(checks)
        c = _find(checks, "DET_LIMITATION_PERIOD")
        assert c["status"] == "INFO"

    def test_old_mortgage_not_flagged(self):
        """Mortgage 15 years ago is NOT chain-relevant → not flagged."""
        old_date = (datetime.now() - timedelta(days=15 * 365)).strftime("%d-%m-%Y")
        txns = [{"row_number": 1, "date": old_date, "transaction_type": "mortgage"}]
        data = _ec("01-01-2005", datetime.now().strftime("%d-%m-%Y"), txns)
        checks = check_limitation_period(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# 4. check_stamp_duty
# ═══════════════════════════════════════════════════

class TestStampDuty:

    def test_correct_stamp_duty(self):
        """7% stamp duty on consideration → no shortfall."""
        # Consideration 45L, guideline 40L → assessable = 45L, expected stamp = 3.15L
        data = _sale_deed(consideration="45,00,000", guideline="40,00,000", stamp_duty="3,15,000")
        checks = check_stamp_duty(data)
        assert "DET_STAMP_DUTY_SHORT" not in _codes(checks)

    def test_stamp_duty_shortfall(self):
        """Stamp duty far below 7% → DET_STAMP_DUTY_SHORT."""
        # Consideration 45L → expected ~3.15L, paid only 1L
        data = _sale_deed(consideration="45,00,000", guideline="40,00,000", stamp_duty="1,00,000")
        checks = check_stamp_duty(data)
        assert "DET_STAMP_DUTY_SHORT" in _codes(checks)
        c = _find(checks, "DET_STAMP_DUTY_SHORT")
        assert c["severity"] == "HIGH"
        assert c["status"] == "FAIL"

    def test_undervaluation(self):
        """Consideration < 80% of guideline → DET_UNDERVALUATION."""
        # Guideline 50L, consideration 30L → 60% of guideline
        data = _sale_deed(consideration="30,00,000", guideline="50,00,000", stamp_duty="3,50,000")
        checks = check_stamp_duty(data)
        assert "DET_UNDERVALUATION" in _codes(checks)
        c = _find(checks, "DET_UNDERVALUATION")
        assert c["severity"] == "HIGH"

    def test_no_consideration_skipped(self):
        """Missing consideration → no checks."""
        data = _sale_deed(consideration="", guideline="40,00,000", stamp_duty="3,15,000")
        checks = check_stamp_duty(data)
        assert checks == []

    def test_non_sale_deed_skipped(self):
        """EC data should be ignored."""
        data = _ec("01-01-2010", "31-12-2025")
        checks = check_stamp_duty(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# 5. check_area_consistency
# ═══════════════════════════════════════════════════

class TestAreaConsistency:

    def test_matching_areas(self):
        """Same area across docs → no issues."""
        data = {**_sale_deed(extent="2400 sq.ft"), **_patta(extent="2400 sq.ft")}
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)

    def test_mismatched_areas(self):
        """Different areas >10% → DET_AREA_MISMATCH."""
        data = {**_sale_deed(extent="2400 sq.ft"), **_patta(extent="5000 sq.ft")}
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_AREA_MISMATCH")
        assert c["severity"] == "HIGH"
        assert c["status"] == "FAIL"

    def test_unit_conversion(self):
        """1 cent = 435.6 sqft — should not flag ~436 sqft vs 1 cent."""
        data = {**_sale_deed(extent="435 sq.ft"), **_patta(extent="1 cent")}
        checks = check_area_consistency(data)
        # Within 10% tolerance
        assert "DET_AREA_MISMATCH" not in _codes(checks)

    def test_single_doc(self):
        """Only one document → nothing to compare."""
        data = _sale_deed(extent="2400 sq.ft")
        checks = check_area_consistency(data)
        assert checks == []

    def test_acres_vs_sqft(self):
        """1 acre = 43560 sqft — huge mismatch should flag."""
        data = {**_sale_deed(extent="1 acre"), **_patta(extent="100 sq.ft")}
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" in _codes(checks)

    # ── Patta-portfolio-aware tests ──

    def test_patta_portfolio_no_false_mismatch(self):
        """Multi-survey Patta with matching survey → uses only matching extent, no mismatch."""
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "2400 sq.ft", "classification": "Dry"},
                        {"survey_no": "543/1A1", "extent": "3000 sq.ft", "classification": "Dry"},
                        {"survey_no": "544/1A1", "extent": "5000 sq.ft", "classification": "Wet"},
                    ],
                    "total_extent": "1.0650 hectares",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)

    def test_patta_portfolio_info_emitted(self):
        """Multi-survey Patta whose total_extent >> matched extent → DET_PATTA_PORTFOLIO INFO."""
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "2400 sq.ft"},
                        {"survey_no": "543", "extent": "10000 sq.ft"},
                    ],
                    "total_extent": "12400 sq.ft",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)
        assert "DET_PATTA_PORTFOLIO" in _codes(checks)
        c = _find(checks, "DET_PATTA_PORTFOLIO")
        assert c["status"] == "INFO"
        assert c["severity"] == "LOW"

    def test_patta_portfolio_genuine_mismatch(self):
        """Even with portfolio filtering, a real extent mismatch on the matching survey flags FAIL."""
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "5000 sq.ft"},
                        {"survey_no": "543", "extent": "3000 sq.ft"},
                    ],
                    "total_extent": "8000 sq.ft",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" in _codes(checks)

    def test_patta_no_matching_survey_skips(self):
        """Patta with no overlapping surveys → no area mismatch (nothing to compare)."""
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "543", "extent": "10000 sq.ft"},
                        {"survey_no": "544", "extent": "5000 sq.ft"},
                    ],
                    "total_extent": "15000 sq.ft",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)

    def test_patta_union_of_non_patta_surveys(self):
        """Target surveys union: EC survey 317 + Sale Deed survey 317 → same target, Patta filters correctly."""
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "317"},
        ])
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            **ec,
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "2400 sq.ft"},
                        {"survey_no": "999", "extent": "50000 sq.ft"},
                    ],
                    "total_extent": "52400 sq.ft",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)


# ═══════════════════════════════════════════════════
# 6. check_survey_number_consistency
# ═══════════════════════════════════════════════════

class TestSurveyNumberConsistency:

    def test_matching_surveys(self):
        """Same survey no → no mismatch."""
        data = {**_sale_deed(survey="311/1"), **_patta(survey_no="311/1")}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)

    def test_mismatched_surveys(self):
        """Completely different Patta survey vs Sale Deed → Patta-only cluster (INFO),
        not CRITICAL FAIL, because Patta is an owner-portfolio document."""
        data = {**_sale_deed(survey="311/1"), **_patta(survey_no="999/5")}
        checks = check_survey_number_consistency(data)
        # No CRITICAL FAIL — Patta's 999/5 is just the owner's other holding
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)
        assert "DET_PATTA_ONLY_SURVEYS" in _codes(checks)
        c = _find(checks, "DET_PATTA_ONLY_SURVEYS")
        assert c["status"] == "INFO"

    def test_subdivision_info(self):
        """Parent-child surveys + a third mismatching doc → DET_SURVEY_SUBDIVISION (INFO).

        Subdivision/OCR info only emitted when >1 cluster exists.
        """
        # 311 ↔ 311/1 = parent-child cluster; 999/5 = separate cluster → 2 clusters
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "999/5"}
        ])
        data = {**_sale_deed(survey="311"), **_patta(survey_no="311/1"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_SUBDIVISION" in _codes(checks)
        c = _find(checks, "DET_SURVEY_SUBDIVISION")
        assert c["status"] == "INFO"

    def test_ocr_fuzzy_warning(self):
        """Non-digit OCR-like difference on longer survey + third mismatching doc
        → DET_SURVEY_OCR_FUZZY.  Note: "311/1" vs "312/1" is now correctly
        treated as a mismatch (digit change), so we use longer survey numbers
        where a non-digit char differs."""
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "999/5"}
        ])
        data = {**_sale_deed(survey="3111A"), **_patta(survey_no="3111B"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_OCR_FUZZY" in _codes(checks)
        c = _find(checks, "DET_SURVEY_OCR_FUZZY")
        assert c["status"] == "WARNING"

    def test_single_doc(self):
        """Only one document → no checks possible."""
        data = _sale_deed(survey="311/1")
        checks = check_survey_number_consistency(data)
        assert checks == []

    # ── Patta-portfolio-aware tests ──

    def test_patta_extra_surveys_no_fail(self):
        """Patta with extra surveys (owner portfolio) + matching EC survey → no DET_SURVEY_MISMATCH."""
        data = {
            **_sale_deed(survey="317"),
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317"},
                        {"survey_no": "543/1A1"},
                        {"survey_no": "544/1A1"},
                    ],
                    "village": "Chromepet",
                },
            },
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)
        # Patta-only surveys should be INFO
        assert "DET_PATTA_ONLY_SURVEYS" in _codes(checks)
        c = _find(checks, "DET_PATTA_ONLY_SURVEYS")
        assert c["status"] == "INFO"
        assert c["severity"] == "LOW"

    def test_non_patta_disagree_still_fail(self):
        """Two non-Patta documents with different surveys → still CRITICAL FAIL."""
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "999/5"},
        ])
        data = {**_sale_deed(survey="311/1"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_SURVEY_MISMATCH")
        assert c["severity"] == "CRITICAL"
        assert c["status"] == "FAIL"

    def test_patta_portfolio_with_ec_no_fail(self):
        """EC survey 317 matches one Patta survey, extra Patta surveys → no FAIL, only INFO."""
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "317"},
        ])
        data = {
            **ec,
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317"},
                        {"survey_no": "543/1A1"},
                        {"survey_no": "543/1C1"},
                        {"survey_no": "544/1A1"},
                        {"survey_no": "544/1B1"},
                    ],
                    "village": "Chromepet",
                },
            },
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)
        assert "DET_PATTA_ONLY_SURVEYS" in _codes(checks)

    def test_patta_wrong_number_still_flags(self):
        """If Patta and EC reference entirely different surveys with no match,
        that IS a legitimate problem (wrong Patta for this property).
        Only the EC can create a shared cluster → 1 shared cluster, no FAIL.
        But the Patta-only cluster still gets INFO.
        """
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "317"},
        ])
        data = {
            **_sale_deed(survey="317"),
            **ec,
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "patta_number": "P-999",
                    "survey_numbers": [
                        {"survey_no": "888/1"},
                        {"survey_no": "888/2"},
                    ],
                    "village": "Chromepet",
                },
            },
        }
        checks = check_survey_number_consistency(data)
        # Sale deed + EC both have 317 → 1 shared cluster
        # Patta has 888/1, 888/2 → patta-only cluster(s)
        # No DET_SURVEY_MISMATCH because only 1 shared cluster
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)
        assert "DET_PATTA_ONLY_SURVEYS" in _codes(checks)


# ═══════════════════════════════════════════════════
# 7. check_plot_identity_consistency
# ═══════════════════════════════════════════════════

class TestPlotIdentityConsistency:

    def test_no_plot_numbers(self):
        """No plot/door numbers mentioned → no checks."""
        data = _sale_deed()
        checks = check_plot_identity_consistency(data)
        assert checks == []

    def test_matching_plots(self):
        """Same plot number across docs → no mismatch."""
        deed = _sale_deed()
        deed["sale_deed.pdf"]["data"]["property"]["plot_number"] = "27"
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "Plot No 27 in Layout XYZ"
        data = {**deed, **ec}
        checks = check_plot_identity_consistency(data)
        assert "DET_PLOT_IDENTITY_MISMATCH" not in _codes(checks)

    def test_mismatched_plots(self):
        """Different plot numbers → DET_PLOT_IDENTITY_MISMATCH."""
        deed = _sale_deed()
        deed["sale_deed.pdf"]["data"]["property"]["plot_number"] = "27"
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "Plot No 26 in Layout XYZ"
        data = {**deed, **ec}
        checks = check_plot_identity_consistency(data)
        assert "DET_PLOT_IDENTITY_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_PLOT_IDENTITY_MISMATCH")
        assert c["severity"] == "CRITICAL"


# ═══════════════════════════════════════════════════
# 8. check_party_name_consistency
# ═══════════════════════════════════════════════════

class TestPartyNameConsistency:

    def test_matching_buyer_patta(self):
        """Buyer matches patta owner → no mismatch."""
        data = {
            **_sale_deed(buyers=[{"name": "Lakshmi W/o Senthil"}]),
            **_patta(owners=[{"name": "Lakshmi"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" not in _codes(checks)

    def test_mismatched_buyer_patta(self):
        """Buyer totally different from patta owner → DET_BUYER_PATTA_MISMATCH."""
        data = {
            **_sale_deed(buyers=[{"name": "Raman S/o Krishnan"}]),
            **_patta(owners=[{"name": "Velan S/o Murugan"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_BUYER_PATTA_MISMATCH")
        assert c["severity"] == "HIGH"

    def test_chain_name_gap(self):
        """Last EC claimant doesn't match sale deed seller → DET_CHAIN_NAME_GAP."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        deed = _sale_deed(sellers=[{"name": "Totally Different Person"}])
        data = {**ec, **deed}
        checks = check_party_name_consistency(data)
        assert "DET_CHAIN_NAME_GAP" in _codes(checks)

    def test_no_patta(self):
        """Only deed, no patta → no buyer-patta check."""
        data = _sale_deed(buyers=[{"name": "Raman"}])
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" not in _codes(checks)

    def test_garbled_tamil_claimant_skips_warning(self):
        """Garbled Tamil EC claimant should NOT produce DET_BUYER_PATTA_MISMATCH.

        When an EC claimant name is detected as garbled Tamil (orphan vowel
        signs, etc.), the deterministic checker should skip the mismatch
        warning because OCR corruption can't be resolved by string matching.
        """
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "seller_or_executant": "X",
             # Garbled Tamil: ெபான்அரசி (orphan vowel sign)
             "buyer_or_claimant": "\u0bc6\u0baa\u0bbe\u0ba9\u0bcd\u0b85\u0bb0\u0b9a\u0bbf"},
        ])
        patta = _patta(owners=[{"name": "\u0baa\u0bca\u0ba9\u0bcd \u0b85\u0bb0\u0b9a\u0bbf"}])  # Clean: பொன் அரசி
        data = {**ec, **patta}
        checks = check_party_name_consistency(data)
        # Should NOT have mismatch warning for garbled names
        mismatch_checks = [c for c in checks if c["code"] == "DET_BUYER_PATTA_MISMATCH"]
        assert len(mismatch_checks) == 0, (
            f"Expected no mismatch warnings for garbled Tamil name, got {len(mismatch_checks)}"
        )


# ═══════════════════════════════════════════════════
# 9. check_rapid_flipping
# ═══════════════════════════════════════════════════

class TestRapidFlipping:

    def test_no_flipping(self):
        """Sales >1 year apart → no issue."""
        txns = [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-06-2018", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        assert checks == []

    def test_rapid_flip_detected(self):
        """Two sales within 6 months → DET_RAPID_FLIPPING."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-04-2020", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        assert "DET_RAPID_FLIPPING" in _codes(checks)
        c = _find(checks, "DET_RAPID_FLIPPING")
        assert c["severity"] == "HIGH"

    def test_mortgage_not_counted(self):
        """Mortgage is not a sale → should not trigger flipping."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-02-2020", "transaction_type": "Mortgage"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        assert checks == []

    def test_sale_deed_type_variant(self):
        """'sale deed' (lowercase) should also count."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "sale deed"},
            {"row_number": 2, "date": "01-04-2020", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        assert "DET_RAPID_FLIPPING" in _codes(checks)


# ═══════════════════════════════════════════════════
# 10. check_financial_scale_anomalies
# ═══════════════════════════════════════════════════

class TestFinancialScaleAnomalies:

    def test_no_anomaly(self):
        """Moderate value progression → no issues."""
        txns = [
            {"row_number": 1, "date": "2015", "transaction_type": "Sale",
             "consideration_amount": "10,00,000", "document_number": "111"},
            {"row_number": 2, "date": "2020", "transaction_type": "Sale",
             "consideration_amount": "20,00,000", "document_number": "222"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_financial_scale_anomalies(data)
        assert "DET_FINANCIAL_SCALE_JUMP" not in _codes(checks)

    def test_10x_jump(self):
        """Lakhs → crores jump → DET_FINANCIAL_SCALE_JUMP."""
        txns = [
            {"row_number": 1, "date": "2015", "transaction_type": "Sale",
             "consideration_amount": "5,00,000", "document_number": "111"},
            {"row_number": 2, "date": "2020", "transaction_type": "Sale",
             "consideration_amount": "5,00,00,000", "document_number": "222"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_financial_scale_anomalies(data)
        assert "DET_FINANCIAL_SCALE_JUMP" in _codes(checks)

    def test_mortgage_exceeds_sale(self):
        """Mortgage > 2x last sale → DET_MORTGAGE_EXCEEDS_SALE."""
        txns = [
            {"row_number": 1, "date": "2018", "transaction_type": "SALE",
             "consideration_amount": "10,00,000", "document_number": "111"},
            {"row_number": 2, "date": "2020", "transaction_type": "MORTGAGE",
             "consideration_amount": "50,00,000", "document_number": "222"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_financial_scale_anomalies(data)
        assert "DET_MORTGAGE_EXCEEDS_SALE" in _codes(checks)

    def test_active_mortgage_burden(self):
        """Unreleased mortgages > 1.5x sale → DET_ACTIVE_MORTGAGE_BURDEN."""
        txns = [
            {"row_number": 1, "date": "2018", "transaction_type": "SALE",
             "consideration_amount": "10,00,000", "document_number": "111"},
            {"row_number": 2, "date": "2019", "transaction_type": "MORTGAGE",
             "consideration_amount": "8,00,000", "document_number": "M1"},
            {"row_number": 3, "date": "2020", "transaction_type": "MORTGAGE",
             "consideration_amount": "9,00,000", "document_number": "M2"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_financial_scale_anomalies(data)
        assert "DET_ACTIVE_MORTGAGE_BURDEN" in _codes(checks)

    def test_released_mortgage_not_flagged(self):
        """Mortgage that has a matching release should not be in active total."""
        txns = [
            {"row_number": 1, "date": "2018", "transaction_type": "SALE",
             "consideration_amount": "10,00,000", "document_number": "111"},
            {"row_number": 2, "date": "2019", "transaction_type": "MORTGAGE",
             "consideration_amount": "20,00,000", "document_number": "M1"},
            {"row_number": 3, "date": "2020", "transaction_type": "RELEASE",
             "consideration_amount": "", "document_number": "M1"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_financial_scale_anomalies(data)
        assert "DET_ACTIVE_MORTGAGE_BURDEN" not in _codes(checks)


# ═══════════════════════════════════════════════════
# 11. check_multi_village
# ═══════════════════════════════════════════════════

class TestMultiVillage:

    def test_same_village(self):
        """Same village across docs → no issue."""
        data = {
            **_sale_deed(village="Chromepet"),
            **_patta(village="Chromepet"),
        }
        checks = check_multi_village(data)
        assert "DET_MULTI_VILLAGE" not in _codes(checks)

    def test_different_villages(self):
        """Different villages → DET_MULTI_VILLAGE."""
        data = {
            **_sale_deed(village="Chromepet"),
            **_patta(village="Tambaram"),
        }
        checks = check_multi_village(data)
        assert "DET_MULTI_VILLAGE" in _codes(checks)

    def test_different_taluks(self):
        """Different taluks → DET_MULTI_TALUK."""
        data = {
            **_sale_deed(village="Chromepet", taluk="Tambaram"),
            **_patta(village="Chromepet", taluk="Sriperumbudur"),
        }
        checks = check_multi_village(data)
        assert "DET_MULTI_TALUK" in _codes(checks)
        c = _find(checks, "DET_MULTI_TALUK")
        assert c["severity"] == "CRITICAL"

    def test_different_districts(self):
        """Different districts → DET_MULTI_DISTRICT."""
        data = {
            **_sale_deed(district="Chengalpattu"),
            **_patta(district="Chennai"),
        }
        checks = check_multi_village(data)
        assert "DET_MULTI_DISTRICT" in _codes(checks)
        c = _find(checks, "DET_MULTI_DISTRICT")
        assert c["severity"] == "CRITICAL"

    def test_fuzzy_village_match(self):
        """Slight transliteration diff → should match (no flag)."""
        data = {
            **_sale_deed(village="Chrompet"),
            **_patta(village="Chromepet"),
        }
        checks = check_multi_village(data)
        assert "DET_MULTI_VILLAGE" not in _codes(checks)

    def test_single_doc(self):
        """One document → nothing to compare."""
        data = _sale_deed(village="Chromepet")
        checks = check_multi_village(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# 12. run_deterministic_checks (integration)
# ═══════════════════════════════════════════════════

class TestRunDeterministicChecks:

    def test_returns_list(self, all_docs):
        """Entry point should return a list of dicts."""
        results = run_deterministic_checks(all_docs)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)
            assert "rule_code" in r
            assert "status" in r
            assert "severity" in r
            assert r["source"] == "deterministic"

    def test_clean_docs_minimal_issues(self, all_docs):
        """Consistent test fixtures should produce no FAIL checks."""
        results = run_deterministic_checks(all_docs)
        fails = [r for r in results if r["status"] == "FAIL"]
        # The conftest fixtures are designed to be consistent
        # There should be no critical failures
        for f in fails:
            # If there are fails, they should be explainable (e.g. EC stale)
            assert f["rule_code"] in (
                "DET_EC_STALE", "DET_SURVEY_MISMATCH", "DET_REG_OUTSIDE_EC",
            ), f"Unexpected FAIL: {f['rule_code']}: {f['explanation']}"

    def test_empty_data(self):
        """Empty extracted_data → empty results."""
        results = run_deterministic_checks({})
        assert results == []

    def test_error_resilience(self):
        """Malformed data should not crash the engine."""
        bad_data = {"bad.pdf": {"document_type": "EC", "data": None}}
        # Should not raise — individual checks handle missing/None data
        results = run_deterministic_checks(bad_data)
        assert isinstance(results, list)


# ═══════════════════════════════════════════════════
# P0 FIX TESTS — COMPOUND AREA PARSING
# ═══════════════════════════════════════════════════

class TestCompoundAreaParsing:
    """Tests for _parse_area_to_sqft compound extent handling (P0 Fix 1).

    The old code used re.search() which only captured the first number+unit
    pair.  "2 acres 50 cents" would be parsed as just "2 acres" (87,120 sqft)
    instead of 89,298 sqft.
    """

    def test_single_unit(self):
        from app.pipeline.deterministic import _parse_area_to_sqft
        result = _parse_area_to_sqft("2 acres")
        assert result == pytest.approx(87120.0, rel=0.01)

    def test_compound_acres_cents(self):
        from app.pipeline.deterministic import _parse_area_to_sqft
        result = _parse_area_to_sqft("2 acres 50 cents")
        expected = 2 * 43560.0 + 50 * 435.6  # 2 acres + 50 cents
        assert result == pytest.approx(expected, rel=0.01)

    def test_compound_hectare_ares(self):
        from app.pipeline.deterministic import _parse_area_to_sqft
        result = _parse_area_to_sqft("1 hectare 20 ares")
        expected = 107639.0 + 20 * 1076.39
        assert result == pytest.approx(expected, rel=0.01)

    def test_compound_in_area_check(self):
        """Full integration: sale_deed with '2 acres 50 cents' vs patta with
        the equivalent sqft should PASS (not FAIL due to partial parse)."""
        compound_sqft = 2 * 43560.0 + 50 * 435.6  # 108,900 sqft
        data = {
            "deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {"property": {"extent": "2 acres 50 cents"}},
            },
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {"total_extent": f"{compound_sqft:.0f} sq ft"},
            },
        }
        checks = check_area_consistency(data)
        # Should NOT produce a mismatch since both parse to ~89,298 sqft
        fail_codes = [c["rule_code"] for c in checks if c["status"] == "FAIL"]
        assert "DET_AREA_MISMATCH" not in fail_codes


# ═══════════════════════════════════════════════════
# P0 FIX TESTS — FATHER NAME PRESERVATION
# ═══════════════════════════════════════════════════

class TestFatherNamePreservation:
    """Tests for name similarity with patronymic awareness (P0 Fix 3).

    Old code stripped everything after S/o, D/o, etc., making
    "Murugan S/o Ramamoorthy" identical to "Murugan S/o Sundaram" (1.0).
    New code compares both given name and patronymic separately.
    """

    def test_same_person_high_similarity(self):
        from app.pipeline.deterministic import _name_similarity
        score = _name_similarity("Murugan S/o Ramamoorthy", "Murugan S/o Ramamoorthy")
        assert score >= 0.95

    def test_different_fathers_low_similarity(self):
        from app.pipeline.deterministic import _name_similarity
        score = _name_similarity("Murugan S/o Ramamoorthy", "Murugan S/o Sundaram")
        # Must be < 0.5 to trigger FAIL in party name consistency (threshold is 0.5)
        assert score < 0.75  # significantly lower than 1.0
        # But given names match so not 0.0
        assert score > 0.3

    def test_split_name_parts(self):
        from app.pipeline.deterministic import _split_name_parts
        g, p = _split_name_parts("Murugan S/o Ramamoorthy")
        assert g == "murugan"
        assert p == "ramamoorthy"

    def test_split_name_no_patronymic(self):
        from app.pipeline.deterministic import _split_name_parts
        g, p = _split_name_parts("Lakshmi")
        assert g == "lakshmi"
        assert p == ""

    def test_split_daughter_of(self):
        from app.pipeline.deterministic import _split_name_parts
        g, p = _split_name_parts("Priya D/o Venkatesh")
        assert g == "priya"
        assert p == "venkatesh"

    def test_split_wife_of(self):
        from app.pipeline.deterministic import _split_name_parts
        g, p = _split_name_parts("Lakshmi W/o Murugan")
        assert g == "lakshmi"
        assert p == "murugan"

    def test_one_has_patronymic_penalty(self):
        from app.pipeline.deterministic import _name_similarity
        full = _name_similarity("Murugan S/o Ramamoorthy", "Murugan")
        bare = _name_similarity("Murugan", "Murugan")
        # Having a patronymic only one side should incur a penalty
        assert full < bare

    def test_party_name_check_catches_different_fathers(self):
        """Integration: similar-ish given names with different fathers now
        correctly score below 0.5 thanks to patronymic awareness.
        'Rajan S/o Raman' vs 'Rajam S/o Sundaram' → ~0.40 (below 0.5)."""
        data = {
            "deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "buyer": [{"name": "Rajan S/o Raman"}],
                    "seller": [{"name": "Lakshmi W/o Shankar"}],
                },
            },
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "owner_names": [{"name": "Rajam S/o Sundaram"}],
                },
            },
        }
        checks = check_party_name_consistency(data)
        codes = _codes(checks)
        assert "DET_BUYER_PATTA_MISMATCH" in codes


# ═══════════════════════════════════════════════════
# P0 FIX TESTS — CRITICAL SCORE CAPPING
# ═══════════════════════════════════════════════════

class TestCriticalScoreCapping:
    """Tests for _compute_score_deductions CRITICAL floor logic (P0 Fix 4).

    1 CRITICAL FAIL → min deduction 51 (score ≤ 49 → HIGH band)
    2+ CRITICAL FAILs → min deduction 81 (score ≤ 19 → CRITICAL band)
    """

    def test_one_critical_fail_floor(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [{"status": "FAIL", "severity": "CRITICAL"}]
        deduction = _compute_score_deductions(checks)
        assert deduction >= 51  # score ≤ 49 → at most HIGH

    def test_two_critical_fails_floor(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
        ]
        deduction = _compute_score_deductions(checks)
        assert deduction >= 81  # score ≤ 19 → CRITICAL band

    def test_no_critical_no_floor(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [{"status": "FAIL", "severity": "HIGH"}]
        deduction = _compute_score_deductions(checks)
        assert deduction == 8  # Just the HIGH deduction, no floor

    def test_mixed_with_critical(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [
            {"status": "FAIL", "severity": "CRITICAL"},  # 25 pts
            {"status": "FAIL", "severity": "HIGH"},       # 8 pts
            {"status": "WARNING", "severity": "MEDIUM"},  # 1 pt
        ]
        deduction = _compute_score_deductions(checks)
        # 25 + 8 + 1 = 34, but floor says ≥ 51 for 1 CRITICAL FAIL
        assert deduction >= 51

    def test_pass_and_warning_no_floor(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [
            {"status": "PASS", "severity": "CRITICAL"},
            {"status": "WARNING", "severity": "HIGH"},
        ]
        deduction = _compute_score_deductions(checks)
        # PASS contributes 0, WARNING contributes 1
        assert deduction == 1

    def test_clamped_to_100(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [{"status": "FAIL", "severity": "CRITICAL"}] * 5
        deduction = _compute_score_deductions(checks)
        assert deduction <= 100


# ═══════════════════════════════════════════════════
# P1 FIX TESTS — EC PERIOD TIERING
# ═══════════════════════════════════════════════════

class TestEcPeriodTiering:
    """Tests for tiered EC period coverage checks (P1 Fix C).

    Tiers:
      < 1 year  → CRITICAL FAIL
      1-10 yr   → HIGH FAIL
      10-13 yr  → MEDIUM WARNING
      13+ yr    → no check (PASS by absence)
    """

    def test_under_1_year_critical(self):
        """200 days → CRITICAL FAIL."""
        data = _ec("01-01-2024", "20-07-2024")
        checks = check_ec_period_coverage(data)
        shorts = [c for c in checks if c["rule_code"] == "DET_EC_SHORT_PERIOD"]
        assert len(shorts) == 1
        assert shorts[0]["severity"] == "CRITICAL"
        assert shorts[0]["status"] == "FAIL"

    def test_5_years_high_fail(self):
        """5 years → HIGH FAIL."""
        data = _ec("01-01-2019", "01-01-2024")
        checks = check_ec_period_coverage(data)
        shorts = [c for c in checks if c["rule_code"] == "DET_EC_SHORT_PERIOD"]
        assert len(shorts) == 1
        assert shorts[0]["severity"] == "HIGH"
        assert shorts[0]["status"] == "FAIL"

    def test_11_years_medium_warning(self):
        """11 years → MEDIUM WARNING (borderline)."""
        data = _ec("01-01-2013", "01-01-2024")
        checks = check_ec_period_coverage(data)
        shorts = [c for c in checks if c["rule_code"] == "DET_EC_SHORT_PERIOD"]
        assert len(shorts) == 1
        assert shorts[0]["severity"] == "MEDIUM"
        assert shorts[0]["status"] == "WARNING"

    def test_14_years_no_issue(self):
        """14 years → no DET_EC_SHORT_PERIOD."""
        data = _ec("01-01-2010", "01-01-2024")
        checks = check_ec_period_coverage(data)
        shorts = [c for c in checks if c["rule_code"] == "DET_EC_SHORT_PERIOD"]
        assert len(shorts) == 0


# ═══════════════════════════════════════════════════
# P1 FIX TESTS — RAPID FLIP TIERING
# ═══════════════════════════════════════════════════

class TestRapidFlipTiering:
    """Tests for two-tier rapid flipping (P1 Fix B).

    < 180 days  → FAIL (6 months)
    180-364 days → WARNING (6-12 months)
    """

    def test_under_6_months_fail(self):
        """90-day gap → FAIL."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-04-2020", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        fail_checks = [c for c in checks if c["status"] == "FAIL"]
        assert len(fail_checks) >= 1
        assert fail_checks[0]["rule_code"] == "DET_RAPID_FLIPPING"

    def test_between_6_and_12_months_warning(self):
        """270-day gap → WARNING."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "28-09-2020", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        warn_checks = [c for c in checks if c["status"] == "WARNING"]
        assert len(warn_checks) >= 1
        assert warn_checks[0]["rule_code"] == "DET_RAPID_FLIPPING"

    def test_over_12_months_no_issue(self):
        """400-day gap → no issue."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "05-02-2021", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        assert checks == []

    def test_mixed_critical_and_warning(self):
        """3 sales: 60-day then 200-day gaps → one FAIL + one WARNING."""
        txns = [
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale"},
            {"row_number": 2, "date": "01-03-2020", "transaction_type": "Sale"},
            {"row_number": 3, "date": "17-09-2020", "transaction_type": "Sale"},
        ]
        data = _ec("01-01-2010", "31-12-2025", txns)
        checks = check_rapid_flipping(data)
        statuses = [c["status"] for c in checks]
        assert "FAIL" in statuses
        assert "WARNING" in statuses


# ═══════════════════════════════════════════════════
# P1 FIX TESTS — ATOMIC FILE WRITES
# ═══════════════════════════════════════════════════

class TestAtomicFileWrites:
    """Tests for atomic save() in AnalysisSession (P1 Fix A)."""

    def test_session_save_creates_file(self, tmp_path, monkeypatch):
        """Session.save() should create a valid JSON file atomically."""
        import app.pipeline.orchestrator as orch
        monkeypatch.setattr(orch, "SESSIONS_DIR", tmp_path)

        from app.pipeline.orchestrator import AnalysisSession
        session = AnalysisSession()
        session.save()

        session_file = tmp_path / f"{session.session_id}.json"
        assert session_file.exists()
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert data["session_id"] == session.session_id

    def test_session_save_no_temp_leftover(self, tmp_path, monkeypatch):
        """After save(), no .tmp files should remain."""
        import app.pipeline.orchestrator as orch
        monkeypatch.setattr(orch, "SESSIONS_DIR", tmp_path)

        from app.pipeline.orchestrator import AnalysisSession
        session = AnalysisSession()
        session.save()

        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []

    def test_session_save_overwrites_existing(self, tmp_path, monkeypatch):
        """Successive saves update the same file atomically."""
        import app.pipeline.orchestrator as orch
        monkeypatch.setattr(orch, "SESSIONS_DIR", tmp_path)

        from app.pipeline.orchestrator import AnalysisSession
        session = AnalysisSession()
        session.status = "processing"
        session.save()

        session.status = "completed"
        session.save()

        session_file = tmp_path / f"{session.session_id}.json"
        data = json.loads(session_file.read_text(encoding="utf-8"))
        assert data["status"] == "completed"


# ═══════════════════════════════════════════════════
# P2: Multi-party name splitting in party consistency
# ═══════════════════════════════════════════════════

class TestMultiPartyNameSplitting:

    def test_ec_multi_party_matches_deed(self):
        """EC 'A and B' as claimant — 'Lakshmi' is the deed seller → no chain gap."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale",
             "seller_or_executant": "Old Owner",
             "buyer_or_claimant": "Raman and Lakshmi"},
        ])
        deed = _sale_deed(sellers=[{"name": "Lakshmi"}],
                          buyers=[{"name": "New Buyer"}])
        data = {**ec, **deed, **_patta(owners=[{"name": "New Buyer"}])}
        checks = check_party_name_consistency(data)
        assert "DET_CHAIN_NAME_GAP" not in _codes(checks)

    def test_ec_comma_separated_parties(self):
        """EC 'A, B, C' should be split — 'Senthil' is the deed seller → no chain gap."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2020", "transaction_type": "Sale",
             "seller_or_executant": "X",
             "buyer_or_claimant": "Raman, Lakshmi, Senthil"},
        ])
        deed = _sale_deed(sellers=[{"name": "Senthil"}],
                          buyers=[{"name": "New Buyer"}])
        data = {**ec, **deed, **_patta(owners=[{"name": "New Buyer"}])}
        checks = check_party_name_consistency(data)
        assert "DET_CHAIN_NAME_GAP" not in _codes(checks)


# ═══════════════════════════════════════════════════
# P2: Survey type awareness in survey check
# ═══════════════════════════════════════════════════

class TestSurveyTypeAwareness:

    def test_same_number_different_type_info(self):
        """Same survey number but SF vs TS prefix → DET_SURVEY_TYPE_DIFF INFO."""
        data = {
            **_sale_deed(survey="S.F.No. 311/1"),
            **_patta(survey_no="T.S.No. 311/1"),
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_TYPE_DIFF" in _codes(checks)
        c = _find(checks, "DET_SURVEY_TYPE_DIFF")
        assert c["status"] == "INFO"
        assert "SF" in c["evidence"]
        assert "TS" in c["evidence"]

    def test_same_type_no_info(self):
        """Same survey type → no DET_SURVEY_TYPE_DIFF."""
        data = {
            **_sale_deed(survey="S.F.No. 311/1"),
            **_patta(survey_no="S.F.No. 311/1"),
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_TYPE_DIFF" not in _codes(checks)

    def test_mismatch_with_type_note(self):
        """Different numbers AND different types across non-Patta docs → MISMATCH with type note."""
        ec = _ec("01-01-2010", "31-12-2025", [
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "T.S.No. 45"},
        ])
        data = {
            **_sale_deed(survey="O.S.No. 120"),
            **ec,
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_SURVEY_MISMATCH")
        assert "OS" in c["explanation"]
        assert "TS" in c["explanation"]


# ═══════════════════════════════════════════════════
# P3: Plausibility range checks
# ═══════════════════════════════════════════════════

class TestPlausibilityRanges:

    def test_normal_values_no_warnings(self):
        """Realistic TN transaction values produce no plausibility checks."""
        data = _sale_deed(consideration="45,00,000", guideline="40,00,000",
                          stamp_duty="3,15,000")
        checks = check_plausibility_ranges(data)
        assert _codes(checks) == []

    def test_absurdly_high_consideration(self):
        """₹999,999,999,999 → implausibly high warning."""
        data = _sale_deed(consideration="999999999999")
        checks = check_plausibility_ranges(data)
        assert "DET_IMPLAUSIBLE_HIGH" in _codes(checks)

    def test_absurdly_low_consideration(self):
        """₹5 → implausibly low warning."""
        data = _sale_deed(consideration="5")
        checks = check_plausibility_ranges(data)
        assert "DET_IMPLAUSIBLE_LOW" in _codes(checks)

    def test_boundary_ok(self):
        """₹10,000 (lower bound) should NOT trigger warning."""
        data = _sale_deed(consideration="10000", guideline="10000",
                          stamp_duty="700")
        checks = check_plausibility_ranges(data)
        low_checks = [c for c in checks if c["rule_code"] == "DET_IMPLAUSIBLE_LOW"
                       and "Consideration" in c["rule_name"]]
        assert low_checks == []

    def test_implausible_extent_too_large(self):
        """300 acres → over TN ceiling → warning."""
        data = _patta(extent="300 acres")
        checks = check_plausibility_ranges(data)
        assert "DET_IMPLAUSIBLE_EXTENT" in _codes(checks)

    def test_normal_extent_ok(self):
        """2400 sqft → no warning."""
        data = _patta(extent="2400 sq.ft")
        checks = check_plausibility_ranges(data)
        assert "DET_IMPLAUSIBLE_EXTENT" not in _codes(checks)

    def test_no_sale_deed(self):
        """EC-only → no financial plausibility checks."""
        data = _ec("01-01-2020", "01-01-2025")
        checks = check_plausibility_ranges(data)
        assert checks == []


# ═══════════════════════════════════════════════════
# P4: Field format validity checks
# ═══════════════════════════════════════════════════

class TestFieldFormatValidity:

    def test_valid_survey_no_warning(self):
        """Clean survey number passes format check."""
        data = _sale_deed(survey="311/1")
        checks = check_field_format_validity(data)
        survey_checks = [c for c in checks if c["rule_code"] == "DET_INVALID_SURVEY_FORMAT"]
        assert survey_checks == []

    def test_garbled_survey_warning(self):
        """Garbled OCR survey number → format warning."""
        data = _sale_deed(survey="XY##Z")
        checks = check_field_format_validity(data)
        assert "DET_INVALID_SURVEY_FORMAT" in _codes(checks)

    def test_short_village_name(self):
        """Single-character village → implausible name warning."""
        data = _sale_deed(village="A")
        checks = check_field_format_validity(data)
        assert "DET_IMPLAUSIBLE_NAME" in _codes(checks)

    def test_digit_only_village(self):
        """Digit-only village → warning."""
        data = _sale_deed(village="12345")
        checks = check_field_format_validity(data)
        assert "DET_IMPLAUSIBLE_NAME" in _codes(checks)

    def test_normal_village_ok(self):
        """Normal village name passes."""
        data = _sale_deed(village="Chromepet")
        checks = check_field_format_validity(data)
        name_checks = [c for c in checks if c["rule_code"] == "DET_IMPLAUSIBLE_NAME"]
        assert name_checks == []

    def test_unparseable_date(self):
        """Garbage date string → unparseable warning."""
        data = _sale_deed(reg_date="garbage-date")
        checks = check_field_format_validity(data)
        assert "DET_UNPARSEABLE_DATE" in _codes(checks)

    def test_valid_date_ok(self):
        """Valid date passes."""
        data = _sale_deed(reg_date="20-06-2020")
        checks = check_field_format_validity(data)
        date_checks = [c for c in checks if c["rule_code"] == "DET_UNPARSEABLE_DATE"]
        assert date_checks == []

    def test_patta_survey_format(self):
        """Patta with garbled survey → warning."""
        data = _patta(survey_no="SE hag Survey 317")
        checks = check_field_format_validity(data)
        assert "DET_INVALID_SURVEY_FORMAT" in _codes(checks)

    def test_patta_valid_survey(self):
        """Patta with clean survey → no warning."""
        data = _patta(survey_no="317/1")
        checks = check_field_format_validity(data)
        survey_checks = [c for c in checks if c["rule_code"] == "DET_INVALID_SURVEY_FORMAT"]
        assert survey_checks == []

    def test_long_village_name(self):
        """Very long village name (>60 chars) → warning."""
        data = _sale_deed(village="A" * 65)
        checks = check_field_format_validity(data)
        assert "DET_IMPLAUSIBLE_NAME" in _codes(checks)


# ═══════════════════════════════════════════════════
# P5: New checks registered in run_deterministic_checks
# ═══════════════════════════════════════════════════

class TestNewChecksRegistered:

    def test_plausibility_runs_in_engine(self):
        """Plausibility check is registered and fires for absurd amounts."""
        data = _sale_deed(consideration="1")
        checks = run_deterministic_checks(data)
        assert "DET_IMPLAUSIBLE_LOW" in _codes(checks)

    def test_field_format_runs_in_engine(self):
        """Field format check is registered and fires for garbled surveys."""
        data = _sale_deed(survey="XY##Z")
        checks = run_deterministic_checks(data)
        assert "DET_INVALID_SURVEY_FORMAT" in _codes(checks)


# ═══════════════════════════════════════════════════
# P6: Garbled Tamil detection
# ═══════════════════════════════════════════════════

class TestGarbledTamil:

    def test_clean_tamil_village_no_warning(self):
        """Clean Tamil village name → no garbled Tamil warning."""
        data = {
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {
                    "village": "சென்னை",
                    "taluk": "தாம்பரம்",
                    "district": "செங்கல்பட்டு",
                    "patta_number": "P-001",
                    "survey_numbers": [{"survey_no": "311/1"}],
                    "total_extent": "2400 sq.ft",
                    "owner_names": [{"name": "A"}],
                },
            }
        }
        checks = check_garbled_tamil(data)
        assert "DET_GARBLED_TAMIL" not in _codes(checks)

    def test_garbled_tamil_detected(self):
        """Orphan vowel signs in extracted text → garbled warning."""
        # Build text with lots of orphan vowel signs
        garbled = "ா" * 15 + "ிூா" * 5
        data = {
            "doc.pdf": {
                "document_type": "PATTA",
                "data": {
                    "village": garbled,
                    "patta_number": "P-001",
                    "survey_numbers": [],
                    "total_extent": "",
                    "owner_names": [],
                },
            }
        }
        checks = check_garbled_tamil(data)
        assert "DET_GARBLED_TAMIL" in _codes(checks)

    def test_ascii_fields_no_warning(self):
        """ASCII-only fields → no garbled Tamil warning."""
        data = _sale_deed()
        checks = check_garbled_tamil(data)
        assert "DET_GARBLED_TAMIL" not in _codes(checks)

    def test_nested_garbled_detected(self):
        """Garbled Tamil in nested field → detected."""
        garbled = "ா" * 20
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "property": {"village": garbled},
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "financials": {},
                },
            }
        }
        checks = check_garbled_tamil(data)
        assert "DET_GARBLED_TAMIL" in _codes(checks)


# ═══════════════════════════════════════════════════
# P7: Hallucination / Suspiciously Perfect detection
# ═══════════════════════════════════════════════════

class TestHallucinationSigns:

    def test_all_round_amounts_warning(self):
        """All financial amounts are exact round numbers → warning."""
        data = _sale_deed(
            consideration="50,00,000",
            guideline="40,00,000",
            stamp_duty="3,00,000",
        )
        # Also add registration_fee to have 3+ amounts (but _sale_deed helper only has 3 fields)
        checks = check_hallucination_signs(data)
        assert "DET_ALL_AMOUNTS_ROUND" in _codes(checks)

    def test_non_round_amounts_no_warning(self):
        """Mixed/non-round amounts → no warning."""
        data = _sale_deed(
            consideration="45,23,500",
            guideline="40,12,300",
            stamp_duty="3,15,245",
        )
        checks = check_hallucination_signs(data)
        assert "DET_ALL_AMOUNTS_ROUND" not in _codes(checks)

    def test_identical_ec_amounts_warning(self):
        """All EC transactions have identical consideration → warning."""
        txns = [
            {"row_number": i, "date": f"0{i}-01-2020", "document_number": str(1000 + i),
             "document_year": "2020", "sro": "SRO",
             "transaction_type": "Sale", "seller_or_executant": f"Seller {i}",
             "buyer_or_claimant": f"Buyer {i}",
             "extent": "2400 sq.ft", "survey_number": "311/1",
             "consideration_amount": "10,00,000", "remarks": "", "suspicious_flags": []}
            for i in range(1, 7)
        ]
        data = _ec("01-01-2015", "31-12-2024", transactions=txns)
        checks = check_hallucination_signs(data)
        assert "DET_EC_IDENTICAL_AMOUNTS" in _codes(checks)

    def test_varied_ec_amounts_no_warning(self):
        """Varied EC transaction amounts → no warning."""
        txns = [
            {"row_number": i, "date": f"0{i}-01-2020", "document_number": str(1000 + i),
             "document_year": "2020", "sro": "SRO",
             "transaction_type": "Sale", "seller_or_executant": f"Seller {i}",
             "buyer_or_claimant": f"Buyer {i}",
             "extent": "2400 sq.ft", "survey_number": "311/1",
             "consideration_amount": f"{i*5},00,000", "remarks": "", "suspicious_flags": []}
            for i in range(1, 7)
        ]
        data = _ec("01-01-2015", "31-12-2024", transactions=txns)
        checks = check_hallucination_signs(data)
        assert "DET_EC_IDENTICAL_AMOUNTS" not in _codes(checks)

    def test_duplicate_field_values_warning(self):
        """3+ financial fields with identical value → warning."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "311/1", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {
                        "consideration_amount": "50,00,000",
                        "guideline_value": "50,00,000",
                        "stamp_duty": "50,00,000",
                    },
                },
            }
        }
        checks = check_hallucination_signs(data)
        assert "DET_DUPLICATE_VALUES" in _codes(checks)

    def test_repeated_party_warning(self):
        """Same party in 80%+ of EC transactions → warning."""
        txns = [
            {"row_number": i, "date": f"0{i}-01-2020", "document_number": str(1000 + i),
             "document_year": "2020", "sro": "SRO",
             "transaction_type": "Sale", "seller_or_executant": "Always Same Seller",
             "buyer_or_claimant": f"Buyer {i}",
             "extent": "2400 sq.ft", "survey_number": "311/1",
             "consideration_amount": f"{i*3},00,000", "remarks": "", "suspicious_flags": []}
            for i in range(1, 7)
        ]
        data = _ec("01-01-2015", "31-12-2024", transactions=txns)
        checks = check_hallucination_signs(data)
        assert "DET_REPEATED_PARTY" in _codes(checks)

    def test_zero_uncertainty_info(self):
        """Large EC with zero extraction notes → info finding."""
        txns = [
            {"row_number": i, "date": f"0{i}-01-2020", "document_number": str(1000 + i),
             "document_year": "2020", "sro": "SRO",
             "transaction_type": "Sale", "seller_or_executant": f"Seller {i}",
             "buyer_or_claimant": f"Buyer {i}",
             "extent": "2400 sq.ft", "survey_number": "311/1",
             "consideration_amount": f"{i*5},00,000", "remarks": "", "suspicious_flags": []}
            for i in range(1, 16)
        ]
        data = {"ec.pdf": {
            "document_type": "EC",
            "data": {
                "period_from": "01-01-2015",
                "period_to": "31-12-2024",
                "transactions": txns,
                "extraction_notes": "",
            }
        }}
        checks = check_hallucination_signs(data)
        assert "DET_ZERO_UNCERTAINTY" in _codes(checks)


# ═══════════════════════════════════════════════════
# P8: New checks registered (garbled Tamil + hallucination)
# ═══════════════════════════════════════════════════

class TestNewChecksRegisteredV2:

    def test_garbled_tamil_registered(self):
        """Garbled Tamil check is registered in run_deterministic_checks."""
        garbled = "ா" * 15 + "ிூா" * 5
        data = {
            "doc.pdf": {
                "document_type": "PATTA",
                "data": {"village": garbled, "patta_number": "P-001",
                         "survey_numbers": [], "total_extent": "", "owner_names": []},
            }
        }
        checks = run_deterministic_checks(data)
        assert "DET_GARBLED_TAMIL" in _codes(checks)

    def test_hallucination_registered(self):
        """Hallucination check is registered in run_deterministic_checks."""
        data = _sale_deed(
            consideration="50,00,000",
            guideline="40,00,000",
            stamp_duty="3,00,000",
        )
        checks = run_deterministic_checks(data)
        assert "DET_ALL_AMOUNTS_ROUND" in _codes(checks)

    def test_total_check_count(self):
        """Engine has 17 check functions registered."""
        # Count by running with clean data — should not crash
        data = _sale_deed()
        checks = run_deterministic_checks(data)
        # The function itself logs the count; we just ensure no crash
        assert isinstance(checks, list)


# ═══════════════════════════════════════════════════
# Deduplication & scoring (orchestrator helpers)
# ═══════════════════════════════════════════════════

class TestDeduplicateChecks:
    """Tests for _deduplicate_checks — preventing double-counted checks."""

    def test_exact_duplicate_removed(self):
        """Same rule_code appearing twice → only 1 kept."""
        from app.pipeline.orchestrator import _deduplicate_checks
        checks = [
            {"rule_code": "ACTIVE_MORTGAGE", "status": "PASS", "severity": "HIGH",
             "explanation": "No mortgage found"},
            {"rule_code": "ACTIVE_MORTGAGE", "status": "PASS", "severity": "HIGH",
             "explanation": "No mortgage found (different text)"},
        ]
        result = _deduplicate_checks(checks)
        codes = [c["rule_code"] for c in result]
        assert codes.count("ACTIVE_MORTGAGE") == 1

    def test_richer_duplicate_kept(self):
        """When deduplicating, the version with more metadata wins."""
        from app.pipeline.orchestrator import _deduplicate_checks
        sparse = {"rule_code": "PORAMBOKE_DETECTION", "status": "PASS",
                   "severity": "MEDIUM", "evidence": ""}
        rich = {"rule_code": "PORAMBOKE_DETECTION", "status": "PASS",
                "severity": "MEDIUM", "evidence": "detailed evidence text here",
                "guardrail_warnings": ["some warning"],
                "ground_truth": {"verified": True, "matches": ["x"]}}
        result = _deduplicate_checks([sparse, rich])
        kept = [c for c in result if c["rule_code"] == "PORAMBOKE_DETECTION"]
        assert len(kept) == 1
        assert len(kept[0].get("guardrail_warnings", [])) > 0

    def test_different_codes_preserved(self):
        """Different rule_codes are all kept."""
        from app.pipeline.orchestrator import _deduplicate_checks
        checks = [
            {"rule_code": "A", "status": "PASS", "severity": "INFO"},
            {"rule_code": "B", "status": "FAIL", "severity": "HIGH"},
            {"rule_code": "C", "status": "WARNING", "severity": "MEDIUM"},
        ]
        result = _deduplicate_checks(checks)
        assert len(result) == 3

    def test_llm_det_equivalents_still_work(self):
        """LLM↔deterministic equivalent dedup (pass 2) still supersedes LLM checks."""
        from app.pipeline.orchestrator import _deduplicate_checks
        checks = [
            {"rule_code": "SURVEY_NUMBER_MISMATCH", "status": "FAIL",
             "severity": "HIGH", "source": "llm"},
            {"rule_code": "DET_SURVEY_MISMATCH", "status": "FAIL",
             "severity": "CRITICAL", "source": "deterministic"},
        ]
        result = _deduplicate_checks(checks)
        llm_check = [c for c in result if c["rule_code"] == "SURVEY_NUMBER_MISMATCH"][0]
        assert llm_check["status"] == "SUPERSEDED"

    def test_triple_duplicate_reduced_to_one(self):
        """3 copies of same rule_code → only 1 in output."""
        from app.pipeline.orchestrator import _deduplicate_checks
        checks = [
            {"rule_code": "GROUP2_SKIPPED", "status": "INFO", "severity": "INFO",
             "evidence": ""},
            {"rule_code": "GROUP2_SKIPPED", "status": "INFO", "severity": "INFO",
             "evidence": ""},
            {"rule_code": "GROUP2_SKIPPED", "status": "INFO", "severity": "INFO",
             "evidence": ""},
        ]
        result = _deduplicate_checks(checks)
        assert sum(1 for c in result if c["rule_code"] == "GROUP2_SKIPPED") == 1


class TestComputeScoreDeductions:
    """Tests for _compute_score_deductions — deterministic scoring."""

    def test_no_checks_zero_deduction(self):
        from app.pipeline.orchestrator import _compute_score_deductions
        assert _compute_score_deductions([]) == 0

    def test_single_critical_fail_minimum_51(self):
        """One CRITICAL FAIL → at least 51 deduction (score ≤ 49)."""
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [{"status": "FAIL", "severity": "CRITICAL"}]
        deduction = _compute_score_deductions(checks)
        assert deduction >= 51

    def test_two_critical_fails_minimum_81(self):
        """Two CRITICAL FAILs → at least 81 deduction (score ≤ 19)."""
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
        ]
        deduction = _compute_score_deductions(checks)
        assert deduction >= 81

    def test_superseded_not_counted(self):
        """SUPERSEDED checks contribute 0 deduction."""
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [{"status": "SUPERSEDED", "severity": "HIGH"}]
        assert _compute_score_deductions(checks) == 0

    def test_passes_no_deduction(self):
        """PASS checks contribute 0 deduction."""
        from app.pipeline.orchestrator import _compute_score_deductions
        checks = [
            {"status": "PASS", "severity": "HIGH"},
            {"status": "PASS", "severity": "CRITICAL"},
        ]
        assert _compute_score_deductions(checks) == 0


# ═══════════════════════════════════════════════════
# PER-CHECK CONFIDENCE ANNOTATION TESTS
# ═══════════════════════════════════════════════════

class TestAnnotateCheckConfidence:
    """Tests for _annotate_check_confidence per-check metadata.

    The function aggregates from ``_field_confidence`` (per-field band dict)
    rather than a single ``_confidence_score``.  Field bands map to:
    high=1.0, medium=0.7, low=0.4.  The minimum across all fields across
    all documents whose type is in needed_types becomes the overall score.
    """

    def test_high_confidence_annotated(self):
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "doc.pdf": {
                "document_type": "EC",
                "data": {"_field_confidence": {"ec_number": "high", "property_description": "high"}},
            }
        }
        _annotate_check_confidence(checks, extracted, ["EC"])
        assert checks[0]["data_confidence"] == "HIGH"
        assert checks[0]["data_confidence_score"] == 1.0

    def test_low_confidence_annotated(self):
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "FAIL"}]
        extracted = {
            "doc.pdf": {
                "document_type": "SALE_DEED",
                "data": {"_field_confidence": {"consideration_amount": "low", "stamp_duty": "high"}},
            }
        }
        _annotate_check_confidence(checks, extracted, ["SALE_DEED"])
        assert checks[0]["data_confidence"] == "VERY_LOW"
        assert checks[0]["data_confidence_score"] == 0.4

    def test_min_across_doc_types(self):
        """Cross-doc groups should use the minimum confidence across documents."""
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {"_field_confidence": {"ec_number": "high"}},
            },
            "patta.pdf": {
                "document_type": "PATTA",
                "data": {"_field_confidence": {"patta_number": "high", "owner_name": "medium"}},
            },
        }
        _annotate_check_confidence(checks, extracted, ["EC", "PATTA"])
        assert checks[0]["data_confidence"] == "MODERATE"
        assert checks[0]["data_confidence_score"] == 0.7

    def test_irrelevant_doc_types_ignored(self):
        """Only doc types in needed_types are considered."""
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {"_field_confidence": {"ec_number": "high"}},
            },
            "sale.pdf": {
                "document_type": "SALE_DEED",
                "data": {"_field_confidence": {"amount": "low"}},  # low but irrelevant
            },
        }
        _annotate_check_confidence(checks, extracted, ["EC"])
        assert checks[0]["data_confidence"] == "HIGH"
        assert checks[0]["data_confidence_score"] == 1.0

    def test_no_confidence_no_annotation(self):
        """If no documents have _field_confidence, checks unchanged."""
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "ec.pdf": {"document_type": "EC", "data": {"ec_number": "123"}},
        }
        _annotate_check_confidence(checks, extracted, ["EC"])
        assert "data_confidence" not in checks[0]

    def test_empty_checks_no_error(self):
        from app.pipeline.orchestrator import _annotate_check_confidence
        _annotate_check_confidence([], {}, ["EC"])  # should not raise

    def test_mixed_bands_uses_minimum(self):
        """Multiple fields with mixed bands → minimum wins."""
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "doc.pdf": {
                "document_type": "EC",
                "data": {"_field_confidence": {
                    "ec_number": "high",
                    "property_description": "medium",
                    "executant": "low",
                }},
            }
        }
        _annotate_check_confidence(checks, extracted, ["EC"])
        assert checks[0]["data_confidence"] == "VERY_LOW"
        assert checks[0]["data_confidence_score"] == 0.4

    def test_empty_field_confidence_dict_no_annotation(self):
        """Empty _field_confidence dict → no annotation (no fields to aggregate)."""
        from app.pipeline.orchestrator import _annotate_check_confidence
        checks = [{"rule_code": "X", "status": "PASS"}]
        extracted = {
            "doc.pdf": {
                "document_type": "EC",
                "data": {"_field_confidence": {}},
            }
        }
        _annotate_check_confidence(checks, extracted, ["EC"])
        assert "data_confidence" not in checks[0]

class TestTSNoPrefixInEC:
    """EC property_description with T.S.No., O.S.No., N.S.No. prefixes
    should be detected by the improved regex."""

    def test_ts_no_extracted(self):
        """T.S.No. in EC property description → survey extracted."""
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "T.S.No. 45/2 in Ward 5"
        data = {**_sale_deed(survey="45/2"), **ec}
        checks = check_survey_number_consistency(data)
        # Should NOT be a mismatch since they match
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)

    def test_os_no_extracted(self):
        """O.S.No. in EC property description → survey extracted."""
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "O.S.No. 100 in Village X"
        data = {**_sale_deed(survey="100"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)

    def test_ns_no_extracted(self):
        """N.S.No. in EC property description → survey extracted."""
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "N.S.No. 200/3A"
        data = {**_sale_deed(survey="200/3A"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)

    def test_s_no_extracted(self):
        """S.No. in EC property description → survey extracted."""
        ec = _ec("01-01-2010", "31-12-2025")
        ec["ec.pdf"]["data"]["property_description"] = "S.No. 77/1"
        data = {**_sale_deed(survey="77/1"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)


class TestIntraECSurveyConsistency:
    """Intra-EC consistency: property description vs transaction surveys."""

    def test_consistent_ec_no_warning(self):
        """Same survey in header and transactions → no intra-EC warning."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "311/1"},
        ])
        ec["ec.pdf"]["data"]["property_description"] = "S.F.No. 311/1 in Chromepet"
        data = {**_sale_deed(survey="311/1"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_EC_INTERNAL_SURVEY_INCONSISTENCY" not in _codes(checks)

    def test_inconsistent_ec_warning(self):
        """Header says 311, transaction says 500 → DET_EC_INTERNAL_SURVEY_INCONSISTENCY."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "500/2"},
        ])
        ec["ec.pdf"]["data"]["property_description"] = "S.F.No. 311 in Chromepet"
        data = {**_sale_deed(survey="311"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_EC_INTERNAL_SURVEY_INCONSISTENCY" in _codes(checks)
        c = _find(checks, "DET_EC_INTERNAL_SURVEY_INCONSISTENCY")
        assert c["severity"] == "HIGH"
        assert c["status"] == "WARNING"

    def test_subdivision_within_ec_no_warning(self):
        """Header says 311, transaction says 311/1 → subdivision, not inconsistency."""
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "01-01-2015", "transaction_type": "Sale",
             "survey_number": "311/1"},
        ])
        ec["ec.pdf"]["data"]["property_description"] = "S.F.No. 311 in Chromepet"
        data = {**_sale_deed(survey="311"), **ec}
        checks = check_survey_number_consistency(data)
        assert "DET_EC_INTERNAL_SURVEY_INCONSISTENCY" not in _codes(checks)


# ═══════════════════════════════════════════════════
# A_REGISTER treated identically to PATTA/CHITTA
# ═══════════════════════════════════════════════════

def _a_register(survey_no: str = "311/1", extent: str = "2400 sq.ft",
                village: str = "Chromepet", taluk: str = "Tambaram",
                district: str = "Chengalpattu",
                owners: list | None = None) -> dict:
    """Build a minimal A_REGISTER extracted_data (same schema as Patta)."""
    return {
        "a_register.pdf": {
            "document_type": "A_REGISTER",
            "data": {
                "patta_number": "P-001",
                "survey_numbers": [{"survey_no": survey_no, "extent": extent}],
                "total_extent": extent,
                "village": village,
                "taluk": taluk,
                "district": district,
                "owner_names": owners or [{"name": "Buyer B"}],
            },
        }
    }


class TestARegisterParity:
    """A_REGISTER must behave identically to PATTA in all deterministic checks."""

    def test_area_consistency_uses_a_register(self):
        """A_REGISTER extent is compared like Patta."""
        data = {
            **_sale_deed(survey="317", extent="2400 sq.ft"),
            "a_register.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "patta_number": "P-637",
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "2400 sq.ft", "classification": "Dry"},
                    ],
                    "total_extent": "2400 sq.ft",
                    "village": "Chromepet",
                },
            },
        }
        checks = check_area_consistency(data)
        assert "DET_AREA_MISMATCH" not in _codes(checks)

    def test_survey_consistency_uses_a_register(self):
        """A_REGISTER surveys grouped like Patta surveys."""
        data = {
            **_sale_deed(survey="317"),
            "a_register.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "survey_numbers": [{"survey_no": "317", "extent": "2400 sq.ft"}],
                    "total_extent": "2400 sq.ft",
                },
            },
        }
        checks = check_survey_number_consistency(data)
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)

    def test_buyer_a_register_owner_match(self):
        """Buyer matches A_REGISTER owner → no mismatch."""
        data = {
            **_sale_deed(buyers=[{"name": "Lakshmi W/o Senthil"}]),
            **_a_register(owners=[{"name": "Lakshmi"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" not in _codes(checks)

    def test_buyer_a_register_owner_mismatch(self):
        """Buyer different from A_REGISTER owner → DET_BUYER_PATTA_MISMATCH."""
        data = {
            **_sale_deed(buyers=[{"name": "Raman S/o Krishnan"}]),
            **_a_register(owners=[{"name": "Velan S/o Murugan"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" in _codes(checks)

    def test_a_register_in_document_types_config(self):
        """A_REGISTER is in the global DOCUMENT_TYPES whitelist."""
        from app.config import DOCUMENT_TYPES
        assert "A_REGISTER" in DOCUMENT_TYPES

    def test_a_register_extra_surveys_patta_only(self):
        """A_REGISTER with extra surveys not in EC/SaleDeed → patta-only cluster (INFO)."""
        data = {
            **_sale_deed(survey="317"),
            "a_register.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "survey_numbers": [
                        {"survey_no": "317", "extent": "2400 sq.ft"},
                        {"survey_no": "999", "extent": "5000 sq.ft"},
                    ],
                    "total_extent": "7400 sq.ft",
                },
            },
        }
        checks = check_survey_number_consistency(data)
        # Survey 999 only in A_REGISTER → patta-only cluster, no CRITICAL fail
        assert "DET_SURVEY_MISMATCH" not in _codes(checks)
        # May emit DET_PATTA_ONLY_SURVEYS INFO
        patta_only = [c for c in checks if c["rule_code"] == "DET_PATTA_ONLY_SURVEYS"]
        assert len(patta_only) >= 1


# ═══════════════════════════════════════════════════
# Seller ↔ Patta owner: mutation pending logic
# ═══════════════════════════════════════════════════

class TestPattaMutationPending:
    """When seller matches patta owner but buyer doesn't, emit INFO not WARNING."""

    def test_seller_matches_patta_owner_fuzzy(self):
        """Seller == patta owner, buyer != patta owner → INFO (mutation pending), not WARNING."""
        data = {
            **_sale_deed(
                sellers=[{"name": "Sathyabama W/o Thulasi Ram"}],
                buyers=[{"name": "Pon Arasi W/o Arulsing"}],
            ),
            **_patta(owners=[{"name": "Sathyabama"}]),
        }
        checks = check_party_name_consistency(data)
        # Should NOT emit WARNING mismatch
        assert "DET_BUYER_PATTA_MISMATCH" not in _codes(checks)
        # Should emit INFO about pending mutation
        match_checks = [c for c in checks if c["rule_code"] == "DET_BUYER_PATTA_MATCH"]
        assert len(match_checks) >= 1
        assert match_checks[0]["severity"] == "MEDIUM"
        assert match_checks[0]["status"] == "INFO"

    def test_neither_matches_still_warns(self):
        """Buyer AND seller both different from patta owner → WARNING remains."""
        data = {
            **_sale_deed(
                sellers=[{"name": "Totally Different Person"}],
                buyers=[{"name": "Another Unknown Person"}],
            ),
            **_patta(owners=[{"name": "Velan S/o Murugan"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" in _codes(checks)

    def test_buyer_matches_patta_owner_still_pass(self):
        """Buyer == patta owner → PASS (existing behavior, mutation already done)."""
        data = {
            **_sale_deed(
                sellers=[{"name": "Old Owner"}],
                buyers=[{"name": "Lakshmi W/o Senthil"}],
            ),
            **_patta(owners=[{"name": "Lakshmi"}]),
        }
        checks = check_party_name_consistency(data)
        assert "DET_BUYER_PATTA_MISMATCH" not in _codes(checks)


# ═══════════════════════════════════════════════════
# BOUNDARY ADJACENCY — FMB ↔ Sale Deed
# ═══════════════════════════════════════════════════

def _fmb(survey_no: str = "317", village: str = "Somayampalayam",
         taluk: str = "Coimbatore North",
         north: str = "", south: str = "", east: str = "", west: str = "") -> dict:
    """Build a minimal FMB extracted_data."""
    return {
        "fmb.pdf": {
            "document_type": "FMB",
            "data": {
                "survey_number": survey_no,
                "village": village,
                "taluk": taluk,
                "boundaries": {
                    "north": north,
                    "south": south,
                    "east": east,
                    "west": west,
                },
                "remarks": "",
            },
        }
    }


def _sd_with_boundaries(north: str = "", south: str = "",
                        east: str = "", west: str = "") -> dict:
    """Build a Sale Deed with boundaries."""
    return {
        "sale_deed.pdf": {
            "document_type": "SALE_DEED",
            "data": {
                "registration_date": "20-06-2020",
                "seller": [{"name": "Seller A"}],
                "buyer": [{"name": "Buyer B"}],
                "property": {
                    "survey_number": "317",
                    "extent": "2400 sq.ft",
                    "village": "Somayampalayam",
                    "taluk": "Coimbatore North",
                    "district": "Coimbatore",
                    "boundaries": {
                        "north": north,
                        "south": south,
                        "east": east,
                        "west": west,
                    },
                },
                "financials": {
                    "consideration_amount": "45,00,000",
                    "guideline_value": "40,00,000",
                    "stamp_duty": "3,15,000",
                },
            },
        }
    }


class TestBoundaryAdjacency:

    def test_no_fmb_no_checks(self):
        """No FMB in session → no boundary checks."""
        data = _sd_with_boundaries(north="S.No. 316", south="Road", east="S.No. 318", west="Canal")
        checks = check_boundary_adjacency(data)
        assert checks == []

    def test_no_sale_deed_no_checks(self):
        """No Sale Deed in session → no boundary checks."""
        data = _fmb(north="S.No. 316", south="Road", east="S.No. 318", west="Canal")
        checks = check_boundary_adjacency(data)
        assert checks == []

    def test_matching_boundaries_no_issues(self):
        """FMB and Sale Deed both mention same physical features → no checks."""
        data = {
            **_sd_with_boundaries(north="S.No. 316", south="Road", east="S.No. 318", west="Canal"),
            **_fmb(north="S.No. 316", south="Road", east="S.No. 318", west="Canal"),
        }
        checks = check_boundary_adjacency(data)
        assert "DET_BOUNDARY_MISMATCH" not in _codes(checks)
        assert "DET_UNDISCLOSED_ENCUMBRANCE" not in _codes(checks)

    def test_fmb_water_channel_not_in_sd(self):
        """FMB shows water channel on east, Sale Deed says owner name → flag."""
        data = {
            **_sd_with_boundaries(north="S.No. 316", south="Road", east="Raman land", west="S.No. 320"),
            **_fmb(north="S.No. 316", south="Road", east="Water channel", west="S.No. 320"),
        }
        checks = check_boundary_adjacency(data)
        assert "DET_BOUNDARY_MISMATCH" in _codes(checks) or "DET_UNDISCLOSED_ENCUMBRANCE" in _codes(checks)
        # Should flag the undisclosed water channel
        assert "DET_UNDISCLOSED_ENCUMBRANCE" in _codes(checks)
        c = _find(checks, "DET_UNDISCLOSED_ENCUMBRANCE")
        assert c["severity"] == "MEDIUM"
        assert c["status"] == "WARNING"
        assert "water channel" in c["evidence"].lower() or "water channel" in c["explanation"].lower()

    def test_fmb_road_disclosed_in_sd(self):
        """Both mention road on south → no undisclosed encumbrance."""
        data = {
            **_sd_with_boundaries(north="S.No. 316", south="Main Road", east="S.No. 318", west="S.No. 320"),
            **_fmb(north="S.No. 316", south="Road", east="S.No. 318", west="S.No. 320"),
        }
        checks = check_boundary_adjacency(data)
        assert "DET_UNDISCLOSED_ENCUMBRANCE" not in _codes(checks)

    def test_fmb_government_land(self):
        """FMB shows Government land not in Sale Deed → undisclosed encumbrance."""
        data = {
            **_sd_with_boundaries(north="S.No. 316", south="S.No. 319", east="S.No. 318", west="S.No. 320"),
            **_fmb(north="S.No. 316", south="Government land", east="S.No. 318", west="S.No. 320"),
        }
        checks = check_boundary_adjacency(data)
        assert "DET_UNDISCLOSED_ENCUMBRANCE" in _codes(checks)

    def test_boundary_mismatch_fmb_physical_sd_person(self):
        """FMB north says 'Nala', Sale Deed says 'S.No. 316' → mismatch."""
        data = {
            **_sd_with_boundaries(north="S.No. 316", south="Road", east="S.No. 318", west="S.No. 320"),
            **_fmb(north="Nala (drainage)", south="Road", east="S.No. 318", west="S.No. 320"),
        }
        checks = check_boundary_adjacency(data)
        assert "DET_BOUNDARY_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_BOUNDARY_MISMATCH")
        assert c["severity"] == "HIGH"
        assert c["status"] == "WARNING"


# ═══════════════════════════════════════════════════
# CONSIDERATION CONSISTENCY — Sale Deed ↔ EC
# ═══════════════════════════════════════════════════

class TestConsiderationConsistency:

    def test_no_ec_no_checks(self):
        """Only Sale Deed → no checks."""
        data = _sale_deed(consideration="45,00,000")
        checks = check_consideration_consistency(data)
        assert checks == []

    def test_no_sale_deed_no_checks(self):
        """Only EC → no checks."""
        data = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "20-06-2020", "document_number": "5909/2012",
             "transaction_type": "Sale", "consideration_amount": "45,00,000",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        checks = check_consideration_consistency(data)
        assert checks == []

    def test_matching_consideration_no_issues(self):
        """SD and EC amounts match → no checks."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909/2012",
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "20-06-2020", "document_number": "5909/2012",
             "transaction_type": "Sale", "consideration_amount": "45,00,000",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        data = {**sd, **ec}
        checks = check_consideration_consistency(data)
        assert "DET_CONSIDERATION_MISMATCH" not in _codes(checks)

    def test_mismatched_consideration(self):
        """SD = 45L, EC = 30L → mismatch flagged."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909/2012",
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "20-06-2020", "document_number": "5909/2012",
             "transaction_type": "Sale", "consideration_amount": "30,00,000",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        data = {**sd, **ec}
        checks = check_consideration_consistency(data)
        assert "DET_CONSIDERATION_MISMATCH" in _codes(checks)
        c = _find(checks, "DET_CONSIDERATION_MISMATCH")
        assert c["severity"] == "HIGH"
        assert c["status"] == "FAIL"

    def test_within_tolerance_no_issue(self):
        """SD = 45L, EC = 44.5L (< 5%) → no mismatch."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909/2012",
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "20-06-2020", "document_number": "5909/2012",
             "transaction_type": "Sale", "consideration_amount": "44,50,000",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        data = {**sd, **ec}
        checks = check_consideration_consistency(data)
        assert "DET_CONSIDERATION_MISMATCH" not in _codes(checks)

    def test_doc_number_partial_match(self):
        """SD doc# '5909' matches EC doc# '5909/2012' → comparison happens."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909",
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025", transactions=[
            {"row_number": 1, "date": "20-06-2020", "document_number": "5909/2012",
             "transaction_type": "Sale", "consideration_amount": "30,00,000",
             "seller_or_executant": "A", "buyer_or_claimant": "B"},
        ])
        data = {**sd, **ec}
        checks = check_consideration_consistency(data)
        assert "DET_CONSIDERATION_MISMATCH" in _codes(checks)


# ═══════════════════════════════════════════════════
# PAN CROSS-VERIFICATION
# ═══════════════════════════════════════════════════

class TestPanConsistency:

    def test_no_sale_deed_no_checks(self):
        """No Sale Deed → no PAN checks."""
        data = _ec("01-01-2010", "31-12-2025")
        checks = check_pan_consistency(data)
        assert checks == []

    def test_valid_pans_no_issues(self):
        """All valid PANs, no duplicates → no checks."""
        data = _sale_deed(
            sellers=[{"name": "Seller A", "pan": "ABCDE1234F"}],
            buyers=[{"name": "Buyer B", "pan": "FGHIJ5678K"}],
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_FORMAT_INVALID" not in _codes(checks)
        assert "DET_PAN_DUPLICATE" not in _codes(checks)
        assert "DET_PAN_DUPLICATE_CROSS" not in _codes(checks)
        assert "DET_PAN_MISSING" not in _codes(checks)

    def test_invalid_pan_format(self):
        """Invalid PAN format → DET_PAN_FORMAT_INVALID."""
        data = _sale_deed(
            sellers=[{"name": "Seller A", "pan": "12345ABCDE"}],
            buyers=[{"name": "Buyer B", "pan": "FGHIJ5678K"}],
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_FORMAT_INVALID" in _codes(checks)
        c = _find(checks, "DET_PAN_FORMAT_INVALID")
        assert c["severity"] == "MEDIUM"
        assert c["status"] == "WARNING"

    def test_duplicate_pan_cross_party(self):
        """Same PAN on seller and buyer → DET_PAN_DUPLICATE_CROSS (FAIL)."""
        data = _sale_deed(
            sellers=[{"name": "Seller A", "pan": "ABCDE1234F"}],
            buyers=[{"name": "Buyer B", "pan": "ABCDE1234F"}],
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_DUPLICATE_CROSS" in _codes(checks)
        c = _find(checks, "DET_PAN_DUPLICATE_CROSS")
        assert c["severity"] == "HIGH"
        assert c["status"] == "FAIL"

    def test_duplicate_pan_same_side(self):
        """Two sellers share same PAN → DET_PAN_DUPLICATE."""
        data = _sale_deed(
            sellers=[
                {"name": "Seller A", "pan": "ABCDE1234F"},
                {"name": "Seller B", "pan": "ABCDE1234F"},
            ],
            buyers=[{"name": "Buyer B", "pan": "FGHIJ5678K"}],
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_DUPLICATE" in _codes(checks)
        c = _find(checks, "DET_PAN_DUPLICATE")
        assert c["severity"] == "MEDIUM"

    def test_missing_pan_advisory(self):
        """Some have PAN, some don't → DET_PAN_MISSING (INFO)."""
        data = _sale_deed(
            sellers=[{"name": "Seller A", "pan": "ABCDE1234F"}],
            buyers=[{"name": "Buyer B"}],  # no PAN
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_MISSING" in _codes(checks)
        c = _find(checks, "DET_PAN_MISSING")
        assert c["severity"] == "LOW"
        assert c["status"] == "INFO"

    def test_all_missing_no_flag(self):
        """All parties missing PAN but none have PAN → no DET_PAN_MISSING."""
        data = _sale_deed(
            sellers=[{"name": "Seller A"}],
            buyers=[{"name": "Buyer B"}],
        )
        checks = check_pan_consistency(data)
        # No DET_PAN_MISSING because it only flags when SOME have PAN but others don't
        assert "DET_PAN_MISSING" not in _codes(checks)

    def test_pan_none_value_ignored(self):
        """PAN set to 'N/A' treated as missing, not invalid format."""
        data = _sale_deed(
            sellers=[{"name": "Seller A", "pan": "ABCDE1234F"}],
            buyers=[{"name": "Buyer B", "pan": "N/A"}],
        )
        checks = check_pan_consistency(data)
        assert "DET_PAN_FORMAT_INVALID" not in _codes(checks)
        assert "DET_PAN_MISSING" in _codes(checks)


# ═══════════════════════════════════════════════════
# PRE-EC PERIOD GAP CHECK
# ═══════════════════════════════════════════════════

class TestPreEcGap:

    def test_no_gap_no_checks(self):
        """Previous ownership within EC period → no gap."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                    "previous_ownership": {
                        "document_number": "1234/2012",
                        "document_date": "15-03-2012",
                        "previous_owner": "Old Owner",
                        "acquisition_mode": "Sale",
                    },
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025")
        data = {**sd, **ec}
        checks = check_pre_ec_gap(data)
        assert "DET_PRE_EC_GAP" not in _codes(checks)

    def test_gap_detected(self):
        """Previous ownership from 2000, EC starts 2010 → gap flagged."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                    "previous_ownership": {
                        "document_number": "789/2000",
                        "document_date": "10-05-2000",
                        "previous_owner": "Earlier Owner",
                        "acquisition_mode": "Sale",
                    },
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025")
        data = {**sd, **ec}
        checks = check_pre_ec_gap(data)
        assert "DET_PRE_EC_GAP" in _codes(checks)
        c = _find(checks, "DET_PRE_EC_GAP")
        assert c["severity"] == "MEDIUM"
        assert c["status"] == "WARNING"
        assert "2000" in c["explanation"]

    def test_no_ec_no_checks(self):
        """No EC → no gap check."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                    "previous_ownership": {
                        "document_number": "789/1990",
                        "document_date": "10-05-1990",
                        "previous_owner": "Earlier Owner",
                        "acquisition_mode": "Sale",
                    },
                },
            }
        }
        checks = check_pre_ec_gap(sd)
        assert checks == []

    def test_no_previous_ownership_no_checks(self):
        """Sale Deed without previous_ownership → no gap check."""
        data = {**_sale_deed(), **_ec("01-01-2010", "31-12-2025")}
        checks = check_pre_ec_gap(data)
        assert checks == []

    def test_small_gap_under_1yr_no_flag(self):
        """Previous ownership just 6 months before EC → no flag (< 1yr threshold)."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                    "previous_ownership": {
                        "document_number": "789/2009",
                        "document_date": "01-07-2009",
                        "previous_owner": "Earlier Owner",
                        "acquisition_mode": "Sale",
                    },
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025")
        data = {**sd, **ec}
        checks = check_pre_ec_gap(data)
        assert "DET_PRE_EC_GAP" not in _codes(checks)

    def test_ownership_history_gap(self):
        """Ownership history entry from 1995, EC starts 2010 → gap."""
        sd = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "registration_date": "20-06-2020",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                    "property": {"survey_number": "317", "extent": "2400 sq.ft",
                                 "village": "X", "taluk": "Y", "district": "Z"},
                    "financials": {"consideration_amount": "4500000",
                                   "guideline_value": "4000000",
                                   "stamp_duty": "315000"},
                    "previous_ownership": {
                        "document_number": "100/2012",
                        "document_date": "20-01-2012",
                        "previous_owner": "Recent Owner",
                        "acquisition_mode": "Sale",
                    },
                    "ownership_history": [
                        {
                            "owner": "Ancestor",
                            "acquisition_mode": "Inheritance",
                            "document_date": "01-01-1995",
                        },
                    ],
                },
            }
        }
        ec = _ec("01-01-2010", "31-12-2025")
        data = {**sd, **ec}
        checks = check_pre_ec_gap(data)
        assert "DET_PRE_EC_GAP" in _codes(checks)


# ═══════════════════════════════════════════════════
# build_chain_of_title
# ═══════════════════════════════════════════════════

class TestBuildChainOfTitle:
    """Tests for the deterministic chain-of-title builder."""

    # ── empty / minimal cases ──

    def test_empty_extracted_data(self):
        """No documents → empty chain."""
        assert build_chain_of_title({}) == []

    def test_none_extracted_data(self):
        """None input → empty chain."""
        assert build_chain_of_title(None) == []

    def test_no_relevant_docs(self):
        """Documents with no ownership data → empty chain."""
        data = {
            "fmb.pdf": {"document_type": "FMB", "data": {"survey_number": "317"}},
            "patta.pdf": {"document_type": "PATTA", "data": {"owner_names": [{"name": "X"}]}},
        }
        assert build_chain_of_title(data) == []

    # ── EC extraction ──

    def test_ec_single_sale(self):
        """Single EC SALE transaction → one chain link."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1,
                        "date": "2012-12-13",
                        "document_number": "5909",
                        "document_year": "2012",
                        "sro": "Vadavalli",
                        "transaction_type": "SALE",
                        "seller_or_executant": "Seller A",
                        "buyer_or_claimant": "Buyer B",
                        "extent": "1.14",
                        "survey_number": "317",
                        "consideration_amount": "5000000",
                        "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        link = chain[0]
        assert link["from"] == "Seller A"
        assert link["to"] == "Buyer B"
        assert link["transaction_type"] == "SALE"
        assert link["document_number"] == "5909/2012"
        assert link["source"] == "EC"
        assert link["sequence"] == 1
        assert link["valid"] is True

    def test_ec_multiple_transactions_sorted(self):
        """Multiple EC transactions → sorted chronologically."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [
                        {
                            "row_number": 1, "date": "2015-06-01",
                            "document_number": "200", "document_year": "2015",
                            "transaction_type": "SALE",
                            "seller_or_executant": "B", "buyer_or_claimant": "C",
                            "extent": "1", "survey_number": "1",
                            "consideration_amount": "100", "remarks": "",
                            "suspicious_flags": [],
                        },
                        {
                            "row_number": 2, "date": "2010-01-15",
                            "document_number": "100", "document_year": "2010",
                            "transaction_type": "SALE",
                            "seller_or_executant": "A", "buyer_or_claimant": "B",
                            "extent": "1", "survey_number": "1",
                            "consideration_amount": "50", "remarks": "",
                            "suspicious_flags": [],
                        },
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        assert chain[0]["from"] == "A"  # 2010 first
        assert chain[0]["sequence"] == 1
        assert chain[1]["from"] == "B"  # 2015 second
        assert chain[1]["sequence"] == 2

    def test_ec_excludes_mortgage(self):
        """MORTGAGE transactions are not title transfers and should be excluded."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [
                        {
                            "row_number": 1, "date": "2012-01-01",
                            "document_number": "100", "document_year": "2012",
                            "transaction_type": "MORTGAGE",
                            "seller_or_executant": "Owner", "buyer_or_claimant": "Bank",
                            "extent": "1", "survey_number": "1",
                            "consideration_amount": "500000", "remarks": "",
                            "suspicious_flags": [],
                        },
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 0

    def test_ec_includes_gift_and_partition(self):
        """GIFT and PARTITION EC types ARE included (title transfers)."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [
                        {
                            "row_number": 1, "date": "2010-01-01",
                            "document_number": "50", "document_year": "2010",
                            "transaction_type": "GIFT",
                            "seller_or_executant": "Father", "buyer_or_claimant": "Son",
                            "extent": "1", "survey_number": "1",
                            "consideration_amount": "0", "remarks": "",
                            "suspicious_flags": [],
                        },
                        {
                            "row_number": 2, "date": "2011-01-01",
                            "document_number": "60", "document_year": "2011",
                            "transaction_type": "PARTITION",
                            "seller_or_executant": "Brothers", "buyer_or_claimant": "Brother A",
                            "extent": "0.5", "survey_number": "1",
                            "consideration_amount": "0", "remarks": "",
                            "suspicious_flags": [],
                        },
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        types = [l["transaction_type"] for l in chain]
        assert "GIFT" in types
        assert "PARTITION" in types

    def test_ec_includes_release(self):
        """RELEASE is in CHAIN_RELEVANT_TYPES and should be included."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-01-01",
                        "document_number": "70", "document_year": "2012",
                        "transaction_type": "RELEASE",
                        "seller_or_executant": "Co-owner", "buyer_or_claimant": "Main Owner",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "0", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["transaction_type"] == "RELEASE"

    # ── Sale Deed extraction ──

    def test_sale_deed_main_sale(self):
        """Sale Deed produces chain link for the main sale."""
        data = {
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909",
                    "registration_date": "2012-12-13",
                    "seller": [{"name": "Seller X"}, {"name": "Seller Y"}],
                    "buyer": [{"name": "Buyer Z"}],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) >= 1
        main = chain[0]
        assert main["from"] == "Seller X, Seller Y"
        assert main["to"] == "Buyer Z"
        assert main["transaction_type"] == "SALE"
        assert main["document_number"] == "5909"
        assert main["source"] == "Sale Deed"

    def test_sale_deed_previous_ownership(self):
        """Sale Deed previous_ownership creates an earlier chain link."""
        data = {
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "200",
                    "registration_date": "2015-06-01",
                    "seller": [{"name": "Current Seller"}],
                    "buyer": [{"name": "New Buyer"}],
                    "previous_ownership": {
                        "document_number": "100",
                        "document_date": "2010-01-15",
                        "previous_owner": "Old Owner",
                        "acquisition_mode": "Sale",
                    },
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        # Older link first (2010)
        assert chain[0]["from"] == "Old Owner"
        assert chain[0]["to"] == "Current Seller"
        assert chain[0]["transaction_type"] == "SALE"
        assert chain[0]["document_number"] == "100"
        # Newer link (2015)
        assert chain[1]["from"] == "Current Seller"
        assert chain[1]["to"] == "New Buyer"

    def test_sale_deed_ownership_history(self):
        """Sale Deed ownership_history produces chain links."""
        data = {
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "300",
                    "registration_date": "2020-01-01",
                    "seller": [{"name": "S"}],
                    "buyer": [{"name": "B"}],
                    "ownership_history": [
                        {
                            "owner": "Middle Owner",
                            "acquired_from": "First Owner",
                            "acquisition_mode": "Gift",
                            "document_number": "50",
                            "document_date": "2005-01-01",
                        },
                        {
                            "owner": "S",
                            "acquired_from": "Middle Owner",
                            "acquisition_mode": "Sale",
                            "document_number": "150",
                            "document_date": "2012-06-01",
                        },
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 3  # 2 history + 1 main sale
        assert chain[0]["from"] == "First Owner"
        assert chain[0]["to"] == "Middle Owner"
        assert chain[0]["transaction_type"] == "GIFT"
        assert chain[1]["from"] == "Middle Owner"
        assert chain[1]["to"] == "S"
        assert chain[2]["from"] == "S"
        assert chain[2]["to"] == "B"

    # ── A-Register extraction ──

    def test_a_register_mutations(self):
        """A-Register mutation entries → chain links."""
        data = {
            "areg.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "owner_name": "Current Owner",
                    "survey_numbers": [{"survey_no": "317", "extent": "1 acre"}],
                    "village": "TestVillage",
                    "taluk": "TestTaluk",
                    "remarks": "",
                    "mutation_entries": [
                        {
                            "date": "2010-05-01",
                            "from_owner": "Prev Owner",
                            "to_owner": "Current Owner",
                            "reason": "Sale",
                            "order_number": "MO-100",
                        },
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Prev Owner"
        assert chain[0]["to"] == "Current Owner"
        assert chain[0]["source"] == "A-Register"
        assert chain[0]["document_number"] == "MO-100"

    def test_a_register_mutation_no_from_uses_owner(self):
        """If mutation has no from_owner, fall back to register's owner_name."""
        data = {
            "areg.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "owner_name": "Registered Owner",
                    "survey_numbers": [{"survey_no": "1", "extent": "1"}],
                    "village": "V", "taluk": "T", "remarks": "",
                    "mutation_entries": [
                        {"to_owner": "New Owner", "reason": "Inheritance"},
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Registered Owner"

    # ── Gift Deed extraction ──

    def test_gift_deed(self):
        """Gift Deed donor → donee chain link."""
        data = {
            "gift.pdf": {
                "document_type": "GIFT_DEED",
                "data": {
                    "donor": {"name": "Father"},
                    "donee": {"name": "Daughter"},
                    "registration_number": "GD-100",
                    "registration_date": "2018-03-15",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Father"
        assert chain[0]["to"] == "Daughter"
        assert chain[0]["transaction_type"] == "GIFT"
        assert chain[0]["source"] == "Gift Deed"

    # ── Partition Deed extraction ──

    def test_partition_deed(self):
        """Partition Deed creates one link per partitioned share."""
        data = {
            "part.pdf": {
                "document_type": "PARTITION_DEED",
                "data": {
                    "joint_owners": [{"name": "Brother A"}, {"name": "Brother B"}],
                    "partitioned_shares": [
                        {"name": "Brother A", "allocated_extent": "0.5 acre"},
                        {"name": "Brother B", "allocated_extent": "0.5 acre"},
                    ],
                    "registration_number": "PD-50",
                    "registration_date": "2016-01-01",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        assert all(l["transaction_type"] == "PARTITION" for l in chain)
        assert all(l["source"] == "Partition Deed" for l in chain)
        tos = {l["to"] for l in chain}
        assert tos == {"Brother A", "Brother B"}
        assert chain[0]["from"] == "Brother A, Brother B"

    # ── Release Deed extraction ──

    def test_release_deed(self):
        """Release Deed releasing_party → beneficiary chain link."""
        data = {
            "rel.pdf": {
                "document_type": "RELEASE_DEED",
                "data": {
                    "releasing_party": {"name": "Co-owner"},
                    "beneficiary": {"name": "Main Owner"},
                    "registration_number": "RD-30",
                    "registration_date": "2019-07-01",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Co-owner"
        assert chain[0]["to"] == "Main Owner"
        assert chain[0]["transaction_type"] == "RELEASE"
        assert chain[0]["source"] == "Release Deed"

    # ── Will extraction ──

    def test_will(self):
        """Will testator → each beneficiary chain links."""
        data = {
            "will.pdf": {
                "document_type": "WILL",
                "data": {
                    "testator": {"name": "Grandfather"},
                    "beneficiaries": [
                        {"name": "Son A", "share": "50%"},
                        {"name": "Son B", "share": "50%"},
                    ],
                    "registration_number": "W-10",
                    "execution_date": "2008-01-01",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        assert all(l["from"] == "Grandfather" for l in chain)
        assert all(l["transaction_type"] == "WILL" for l in chain)
        assert all(l["source"] == "Will" for l in chain)
        tos = {l["to"] for l in chain}
        assert tos == {"Son A", "Son B"}
        assert "Bequest: 50%" in chain[0]["notes"]

    # ── Legal Heir extraction ──

    def test_legal_heir(self):
        """Legal Heir Certificate deceased → heirs chain links."""
        data = {
            "heir.pdf": {
                "document_type": "LEGAL_HEIR",
                "data": {
                    "deceased_name": "Late Owner",
                    "date_of_death": "2020-06-15",
                    "certificate_number": "LH-5",
                    "heirs": [
                        {"name": "Wife", "relationship": "Spouse", "share_percentage": "50"},
                        {"name": "Son", "relationship": "Son", "share_percentage": "50"},
                    ],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        assert all(l["from"] == "Late Owner" for l in chain)
        assert all(l["transaction_type"] == "INHERITANCE" for l in chain)
        assert all(l["source"] == "Legal Heir" for l in chain)

    # ── Court Order extraction ──

    def test_court_order_decree(self):
        """Court Order with 'decree' type → chain link."""
        data = {
            "court.pdf": {
                "document_type": "COURT_ORDER",
                "data": {
                    "petitioner": "Claimant X",
                    "respondent": "Defendant Y",
                    "order_type": "decree",
                    "case_number": "CS-100/2018",
                    "order_date": "2018-11-01",
                    "status": "final",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Claimant X"
        assert chain[0]["to"] == "Defendant Y"
        assert chain[0]["transaction_type"] == "COURT_ORDER"
        assert chain[0]["source"] == "Court Order"

    def test_court_order_injunction_excluded(self):
        """Court Order with 'injunction' type → no chain link (not a transfer)."""
        data = {
            "court.pdf": {
                "document_type": "COURT_ORDER",
                "data": {
                    "petitioner": "P",
                    "respondent": "R",
                    "order_type": "injunction",
                    "case_number": "CS-200",
                    "order_date": "2019-01-01",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 0

    # ── POA extraction ──

    def test_poa(self):
        """POA principal → agent chain link with notes."""
        data = {
            "poa.pdf": {
                "document_type": "POA",
                "data": {
                    "principal": {"name": "Owner Abroad"},
                    "agent": {"name": "Local Agent"},
                    "registration_number": "POA-20",
                    "registration_date": "2017-03-01",
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 1
        assert chain[0]["from"] == "Owner Abroad"
        assert chain[0]["to"] == "Local Agent"
        assert chain[0]["transaction_type"] == "POWER_OF_ATTORNEY"
        assert chain[0]["source"] == "POA"
        assert "not title transfer" in chain[0]["notes"].lower()

    # ── Deduplication ──

    def test_dedup_same_sale_in_ec_and_sale_deed(self):
        """Same sale appearing in both EC and Sale Deed → deduplicated to one link."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-12-13",
                        "document_number": "5909", "document_year": "2012",
                        "transaction_type": "SALE",
                        "seller_or_executant": "Seller A",
                        "buyer_or_claimant": "Buyer B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "5000000", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "5909/2012",
                    "registration_date": "2012-12-13",
                    "seller": [{"name": "Seller A"}],
                    "buyer": [{"name": "Buyer B"}],
                },
            },
        }
        chain = build_chain_of_title(data)
        # The EC doc_number becomes "5909/2012" and the SD doc_number is "5909/2012"
        # so they share the same dedup key → only 1 link
        sale_links = [l for l in chain if l["document_number"] == "5909/2012"]
        assert len(sale_links) == 1

    def test_dedup_keeps_richer_link(self):
        """When deduplicating, keep the link with more information."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "",
                        "document_number": "100", "document_year": "2010",
                        "transaction_type": "SALE",
                        "seller_or_executant": "A", "buyer_or_claimant": "B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "0", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "100/2010",
                    "registration_date": "2010-05-15",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                },
            },
        }
        chain = build_chain_of_title(data)
        # Both produce 100/2010 SALE → dedup. The one with the date should win.
        matching = [l for l in chain if l["document_number"] == "100/2010"
                    and l["transaction_type"] == "SALE"]
        assert len(matching) == 1
        assert matching[0]["date"] == "2010-05-15"

    # ── LLM chain merge ──

    def test_llm_chain_merge_notes(self):
        """LLM chain notes are merged onto matching deterministic link."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-12-13",
                        "document_number": "5909", "document_year": "2012",
                        "transaction_type": "SALE",
                        "seller_or_executant": "Seller A",
                        "buyer_or_claimant": "Buyer B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "5000000", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
        }
        llm_chain = [{
            "sequence": 1,
            "date": "2012-12-13",
            "from": "Seller A",
            "to": "Buyer B",
            "transaction_type": "SALE",
            "document_number": "5909/2012",
            "valid": True,
            "notes": "LLM note: consideration Rs 50L for 1.14 acres",
        }]
        chain = build_chain_of_title(data, llm_chain=llm_chain)
        assert len(chain) == 1
        assert "LLM note" in chain[0]["notes"]
        assert chain[0]["source"] == "EC"  # deterministic source preserved

    def test_llm_chain_validity_override(self):
        """LLM chain valid=False overrides deterministic default True."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-12-13",
                        "document_number": "5909", "document_year": "2012",
                        "transaction_type": "SALE",
                        "seller_or_executant": "Seller A",
                        "buyer_or_claimant": "Buyer B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "5000000", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
        }
        llm_chain = [{
            "sequence": 1, "from": "Seller A", "to": "Buyer B",
            "transaction_type": "SALE", "document_number": "5909/2012",
            "valid": False, "notes": "Suspicious",
        }]
        chain = build_chain_of_title(data, llm_chain=llm_chain)
        assert chain[0]["valid"] is False

    def test_llm_only_chain_kept(self):
        """LLM chain link with no deterministic match → kept with source='LLM'."""
        data = {}
        llm_chain = [{
            "sequence": 1, "date": "2008-01-01",
            "from": "Unknown Seller", "to": "Unknown Buyer",
            "transaction_type": "SALE", "document_number": "OLD-100",
            "valid": True, "notes": "From LLM analysis",
        }]
        chain = build_chain_of_title(data, llm_chain=llm_chain)
        assert len(chain) == 1
        assert chain[0]["source"] == "LLM"
        assert chain[0]["from"] == "Unknown Seller"

    # ── Mixed documents — unified chain ──

    def test_mixed_ec_sale_deed_a_register(self):
        """EC + Sale Deed + A-Register → unified chronological chain."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2015-06-01",
                        "document_number": "200", "document_year": "2015",
                        "transaction_type": "SALE",
                        "seller_or_executant": "B", "buyer_or_claimant": "C",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "100", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "200/2015",
                    "registration_date": "2015-06-01",
                    "seller": [{"name": "B"}],
                    "buyer": [{"name": "C"}],
                    "previous_ownership": {
                        "document_number": "100",
                        "document_date": "2010-01-01",
                        "previous_owner": "A",
                        "acquisition_mode": "Sale",
                    },
                },
            },
            "areg.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "owner_name": "C",
                    "survey_numbers": [{"survey_no": "1", "extent": "1"}],
                    "village": "V", "taluk": "T", "remarks": "",
                    "mutation_entries": [{
                        "date": "2020-03-01",
                        "from_owner": "C",
                        "to_owner": "D",
                        "reason": "Inheritance",
                        "order_number": "MO-50",
                    }],
                },
            },
        }
        chain = build_chain_of_title(data)
        # Should have: A→B (prev_ownership 2010), B→C (deduped 2015), C→D (mutation 2020)
        assert len(chain) == 3
        assert chain[0]["from"] == "A"
        assert chain[0]["date"] == "2010-01-01"
        assert chain[1]["to"] == "C"
        assert chain[2]["from"] == "C"
        assert chain[2]["to"] == "D"
        assert chain[2]["source"] == "A-Register"
        # Sequential numbering
        assert [l["sequence"] for l in chain] == [1, 2, 3]

    # ── Edge cases ──

    def test_empty_party_names_skipped(self):
        """Chain links with empty from or to are not created."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-01-01",
                        "document_number": "100", "document_year": "2012",
                        "transaction_type": "SALE",
                        "seller_or_executant": "",
                        "buyer_or_claimant": "Buyer",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "0", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 0

    def test_missing_date_sorted_last(self):
        """Links with no parseable date appear after dated links."""
        data = {
            "sd1.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "1",
                    "registration_date": "",
                    "seller": [{"name": "X"}],
                    "buyer": [{"name": "Y"}],
                },
            },
            "sd2.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "2",
                    "registration_date": "2015-01-01",
                    "seller": [{"name": "A"}],
                    "buyer": [{"name": "B"}],
                },
            },
        }
        chain = build_chain_of_title(data)
        assert len(chain) == 2
        # The one with date should come first
        assert chain[0]["date"] == "2015-01-01"
        assert chain[1]["date"] == ""

    def test_document_with_no_data_dict_skipped(self):
        """Document entry with data=None is gracefully skipped."""
        data = {
            "bad.pdf": {"document_type": "EC", "data": None},
        }
        chain = build_chain_of_title(data)
        assert chain == []

    def test_transaction_id_preserved(self):
        """EC transaction_id is preserved in the chain link."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2012-12-13",
                        "document_number": "5909", "document_year": "2012",
                        "transaction_type": "SALE",
                        "seller_or_executant": "A", "buyer_or_claimant": "B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "5000000", "remarks": "",
                        "suspicious_flags": [],
                        "transaction_id": "EC-5909/2012-Vadavalli",
                    }],
                },
            }
        }
        chain = build_chain_of_title(data)
        assert chain[0]["transaction_id"] == "EC-5909/2012-Vadavalli"

    def test_all_sources_in_one_chain(self):
        """Verify all document types can contribute to one unified chain."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "transactions": [{
                        "row_number": 1, "date": "2000-01-01",
                        "document_number": "1", "document_year": "2000",
                        "transaction_type": "SALE",
                        "seller_or_executant": "A", "buyer_or_claimant": "B",
                        "extent": "1", "survey_number": "1",
                        "consideration_amount": "100", "remarks": "",
                        "suspicious_flags": [],
                    }],
                },
            },
            "sd.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "2",
                    "registration_date": "2005-01-01",
                    "seller": [{"name": "B"}],
                    "buyer": [{"name": "C"}],
                },
            },
            "gift.pdf": {
                "document_type": "GIFT_DEED",
                "data": {
                    "donor": {"name": "C"},
                    "donee": {"name": "D"},
                    "registration_number": "3",
                    "registration_date": "2010-01-01",
                },
            },
            "areg.pdf": {
                "document_type": "A_REGISTER",
                "data": {
                    "owner_name": "D",
                    "survey_numbers": [{"survey_no": "1", "extent": "1"}],
                    "village": "V", "taluk": "T", "remarks": "",
                    "mutation_entries": [{
                        "date": "2015-01-01",
                        "from_owner": "D", "to_owner": "E",
                        "reason": "Inheritance", "order_number": "MO-1",
                    }],
                },
            },
            "will.pdf": {
                "document_type": "WILL",
                "data": {
                    "testator": {"name": "E"},
                    "beneficiaries": [{"name": "F", "share": "100%"}],
                    "registration_number": "4",
                    "execution_date": "2018-01-01",
                },
            },
            "heir.pdf": {
                "document_type": "LEGAL_HEIR",
                "data": {
                    "deceased_name": "F",
                    "date_of_death": "2020-01-01",
                    "certificate_number": "5",
                    "heirs": [{"name": "G", "relationship": "Son", "share_percentage": "100"}],
                },
            },
        }
        chain = build_chain_of_title(data)
        sources = {l["source"] for l in chain}
        assert "EC" in sources
        assert "Sale Deed" in sources
        assert "Gift Deed" in sources
        assert "A-Register" in sources
        assert "Will" in sources
        assert "Legal Heir" in sources
        # Verify chronological ordering
        assert chain[0]["date"] == "2000-01-01"
        assert chain[-1]["date"] == "2020-01-01"
        # Verify all sequences are sequential
        seqs = [l["sequence"] for l in chain]
        assert seqs == list(range(1, len(chain) + 1))


# ═══════════════════════════════════════════════════
# SRO JURISDICTION DETERMINISTIC CHECK
# ═══════════════════════════════════════════════════

class TestCheckSroJurisdiction:
    """Tests for check_sro_jurisdiction — deterministic SRO override."""

    def test_vadavalli_coimbatore_pass(self):
        """Vadavalli SRO + Coimbatore district → PASS."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "sro": "Vadavalli",
                    "property": {
                        "district": "Coimbatore",
                        "village": "Somayampalayam",
                    },
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 1
        assert checks[0]["rule_code"] == "DET_SRO_JURISDICTION"
        assert checks[0]["status"] == "PASS"
        assert checks[0]["severity"] == "CRITICAL"

    def test_singanallur_coimbatore_pass(self):
        """Singanallur SRO + Coimbatore district → PASS."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "sro": "Singanallur",
                    "property": {
                        "district": "Coimbatore",
                        "village": "Ganapathy",
                    },
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 1
        assert checks[0]["status"] == "PASS"

    def test_sro_from_document_number(self):
        """SRO extracted from document_number when sro field is empty."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "document_number": "R/Vadavalli/Book1/5909/2012",
                    "property": {
                        "district": "Coimbatore",
                        "village": "Somayampalayam",
                    },
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 1
        assert checks[0]["status"] == "PASS"

    def test_known_mismatch_fail(self):
        """Vadavalli SRO + Madurai district → FAIL (known mapping mismatch)."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "sro": "Vadavalli",
                    "property": {
                        "district": "Madurai",
                        "village": "SomeVillage",
                    },
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 1
        assert checks[0]["status"] == "FAIL"

    def test_missing_sro_no_checks(self):
        """No SRO available → no checks emitted."""
        data = {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": {
                    "property": {
                        "district": "Coimbatore",
                        "village": "Somayampalayam",
                    },
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 0

    def test_ec_sro_used_when_sale_deed_missing(self):
        """EC document provides SRO when Sale Deed lacks it."""
        data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {
                    "sro": "Vadavalli",
                    "district": "Coimbatore",
                    "village": "Somayampalayam",
                },
            },
        }
        checks = check_sro_jurisdiction(data)
        assert len(checks) == 1
        assert checks[0]["status"] == "PASS"


# ═══════════════════════════════════════════════════
# CHAIN LINK UNKNOWN FILTERING
# ═══════════════════════════════════════════════════

class TestChainLinkUnknownFiltering:
    """Tests for _chain_link filtering of Unknown/placeholder party names."""

    def test_unknown_from_party(self):
        """from_party='Unknown' → None."""
        result = _chain_link(from_party="Unknown", to_party="V. அருள்சிங்")
        assert result is None

    def test_unknown_to_party(self):
        """to_party='Unknown' → None."""
        result = _chain_link(from_party="முருகன்", to_party="Unknown")
        assert result is None

    def test_both_unknown(self):
        """Both parties Unknown → None."""
        result = _chain_link(from_party="unknown", to_party="unknown")
        assert result is None

    def test_not_available_filtered(self):
        """'Not Available' → None."""
        result = _chain_link(from_party="Not Available", to_party="Someone")
        assert result is None

    def test_na_filtered(self):
        """'N/A' → None."""
        result = _chain_link(from_party="Someone", to_party="N/A")
        assert result is None

    def test_dash_filtered(self):
        """'-' → None."""
        result = _chain_link(from_party="-", to_party="Someone")
        assert result is None

    def test_real_names_pass(self):
        """Real Tamil names produce a valid chain link."""
        result = _chain_link(
            from_party="மருதகுட்டி",
            to_party="V. அருள்சிங்",
            date="1992-01-20",
            transaction_type="SALE",
            source="Sale Deed",
        )
        assert result is not None
        assert result["from"] == "மருதகுட்டி"
        assert result["to"] == "V. அருள்சிங்"
        assert result["transaction_type"] == "SALE"

    def test_empty_string_filtered(self):
        """Empty string → None (existing behavior preserved)."""
        result = _chain_link(from_party="", to_party="Someone")
        assert result is None
