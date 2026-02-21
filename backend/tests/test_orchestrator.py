"""Unit tests for orchestrator pure functions.

Tests _compute_score_deductions, _deduplicate_checks,
_validate_group_result, _annotate_check_confidence, and _confidence_band.
"""

import pytest
from app.pipeline.orchestrator import (
    _compute_score_deductions,
    _deduplicate_checks,
    _validate_group_result,
    _annotate_check_confidence,
    _confidence_band,
    _compute_check_deduction,
)


# ═══════════════════════════════════════════════════
# _confidence_band
# ═══════════════════════════════════════════════════


class TestConfidenceBand:

    def test_high(self):
        assert _confidence_band(0.90) == "HIGH"
        assert _confidence_band(0.85) == "HIGH"
        assert _confidence_band(1.0) == "HIGH"

    def test_moderate(self):
        assert _confidence_band(0.70) == "MODERATE"
        assert _confidence_band(0.65) == "MODERATE"

    def test_low(self):
        assert _confidence_band(0.50) == "LOW"
        assert _confidence_band(0.45) == "LOW"

    def test_very_low(self):
        assert _confidence_band(0.30) == "VERY_LOW"
        assert _confidence_band(0.0) == "VERY_LOW"

    def test_boundary_exact(self):
        # 0.85 is HIGH, 0.8499 is MODERATE
        assert _confidence_band(0.8499) == "MODERATE"
        assert _confidence_band(0.6499) == "LOW"
        assert _confidence_band(0.4499) == "VERY_LOW"


# ═══════════════════════════════════════════════════
# _compute_check_deduction  (single check)
# ═══════════════════════════════════════════════════


class TestComputeCheckDeduction:

    def test_fail_critical(self):
        assert _compute_check_deduction({"status": "FAIL", "severity": "CRITICAL"}) == 25

    def test_fail_high(self):
        assert _compute_check_deduction({"status": "FAIL", "severity": "HIGH"}) == 8

    def test_fail_medium(self):
        assert _compute_check_deduction({"status": "FAIL", "severity": "MEDIUM"}) == 3

    def test_fail_info(self):
        assert _compute_check_deduction({"status": "FAIL", "severity": "INFO"}) == 1

    def test_warning_any_severity(self):
        assert _compute_check_deduction({"status": "WARNING", "severity": "CRITICAL"}) == 1
        assert _compute_check_deduction({"status": "WARNING", "severity": "INFO"}) == 1

    def test_pass(self):
        assert _compute_check_deduction({"status": "PASS", "severity": "CRITICAL"}) == 0

    def test_superseded(self):
        assert _compute_check_deduction({"status": "SUPERSEDED", "severity": "CRITICAL"}) == 0

    def test_not_applicable(self):
        assert _compute_check_deduction({"status": "NOT_APPLICABLE", "severity": "HIGH"}) == 0


# ═══════════════════════════════════════════════════
# _compute_score_deductions  (aggregate scoring)
# ═══════════════════════════════════════════════════


class TestComputeScoreDeductions:

    def test_all_pass(self):
        checks = [
            {"status": "PASS", "severity": "CRITICAL"},
            {"status": "PASS", "severity": "HIGH"},
        ]
        assert _compute_score_deductions(checks) == 0

    def test_single_fail_medium(self):
        checks = [{"status": "FAIL", "severity": "MEDIUM"}]
        assert _compute_score_deductions(checks) == 3

    def test_mixed(self):
        checks = [
            {"status": "FAIL", "severity": "HIGH"},    # 8
            {"status": "WARNING", "severity": "HIGH"},  # 1
            {"status": "PASS", "severity": "MEDIUM"},    # 0
        ]
        assert _compute_score_deductions(checks) == 9

    def test_critical_floor_one(self):
        """1 CRITICAL FAIL → minimum deduction 51."""
        checks = [{"status": "FAIL", "severity": "CRITICAL"}]  # base = 25
        assert _compute_score_deductions(checks) == 51

    def test_critical_floor_two(self):
        """2+ CRITICAL FAILs → minimum deduction 81."""
        checks = [
            {"status": "FAIL", "severity": "CRITICAL"},  # 25
            {"status": "FAIL", "severity": "CRITICAL"},  # 25 → base=50, floor=81
        ]
        assert _compute_score_deductions(checks) == 81

    def test_superseded_ignored(self):
        """SUPERSEDED checks contribute 0."""
        checks = [
            {"status": "SUPERSEDED", "severity": "CRITICAL"},
            {"status": "PASS", "severity": "HIGH"},
        ]
        assert _compute_score_deductions(checks) == 0

    def test_clamped_to_100(self):
        """Total can't exceed 100."""
        checks = [
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
            {"status": "FAIL", "severity": "CRITICAL"},
        ]
        # 5 × 25 = 125 but floor 81, max is min(125,100) = 100
        assert _compute_score_deductions(checks) == 100

    def test_empty(self):
        assert _compute_score_deductions([]) == 0

    def test_warnings_only(self):
        checks = [{"status": "WARNING", "severity": "MEDIUM"}] * 5
        assert _compute_score_deductions(checks) == 5


# ═══════════════════════════════════════════════════
# _deduplicate_checks
# ═══════════════════════════════════════════════════


class TestDeduplicateChecks:

    def test_no_duplicates(self):
        checks = [
            {"rule_code": "A", "status": "PASS", "evidence": ""},
            {"rule_code": "B", "status": "FAIL", "evidence": ""},
        ]
        result = _deduplicate_checks(checks)
        assert len(result) == 2

    def test_exact_duplicate_keeps_richer(self):
        """When same rule_code appears twice, keep the one with more metadata."""
        checks = [
            {"rule_code": "A", "status": "FAIL", "evidence": "short"},
            {"rule_code": "A", "status": "FAIL", "evidence": "much longer evidence text"},
        ]
        result = _deduplicate_checks(checks)
        codes = [c["rule_code"] for c in result]
        assert codes.count("A") == 1
        # The richer one (longer evidence) should be kept
        assert result[0]["evidence"] == "much longer evidence text"

    def test_llm_det_equivalent_superseded(self):
        """LLM check FAIL superseded by deterministic FAIL on same risk."""
        checks = [
            {"rule_code": "SURVEY_NUMBER_MISMATCH", "status": "FAIL", "source": "llm"},
            {"rule_code": "DET_SURVEY_MISMATCH", "status": "FAIL", "source": "deterministic"},
        ]
        result = _deduplicate_checks(checks)
        llm_check = [c for c in result if c["rule_code"] == "SURVEY_NUMBER_MISMATCH"][0]
        assert llm_check["status"] == "SUPERSEDED"

    def test_llm_det_no_supersede_if_det_passes(self):
        """LLM FAIL not superseded if deterministic equivalent PASSes."""
        checks = [
            {"rule_code": "SURVEY_NUMBER_MISMATCH", "status": "FAIL", "source": "llm"},
            {"rule_code": "DET_SURVEY_MISMATCH", "status": "PASS", "source": "deterministic"},
        ]
        result = _deduplicate_checks(checks)
        llm_check = [c for c in result if c["rule_code"] == "SURVEY_NUMBER_MISMATCH"][0]
        assert llm_check["status"] == "FAIL"  # Not superseded

    def test_empty_rule_code_preserved(self):
        """Checks with empty rule_code are always kept."""
        checks = [
            {"rule_code": "", "status": "FAIL", "evidence": "a"},
            {"rule_code": "", "status": "PASS", "evidence": "b"},
        ]
        result = _deduplicate_checks(checks)
        assert len(result) == 2

    def test_verified_ground_truth_preferred(self):
        """Check with verified ground_truth is preferred over one without."""
        checks = [
            {"rule_code": "X", "status": "FAIL", "evidence": "",
             "ground_truth": {"verified": True}, "guardrail_warnings": []},
            {"rule_code": "X", "status": "FAIL", "evidence": "",
             "ground_truth": {"verified": False}, "guardrail_warnings": []},
        ]
        result = _deduplicate_checks(checks)
        assert len([c for c in result if c["rule_code"] == "X"]) == 1
        assert result[0]["ground_truth"]["verified"] is True


# ═══════════════════════════════════════════════════
# _annotate_check_confidence
# ═══════════════════════════════════════════════════


class TestAnnotateCheckConfidence:

    def test_annotates_with_min_score(self):
        checks = [{"status": "FAIL", "severity": "HIGH"}]
        extracted_data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {"_confidence_score": 0.72},
            },
            "sale.pdf": {
                "document_type": "SALE_DEED",
                "data": {"_confidence_score": 0.55},
            },
        }
        _annotate_check_confidence(checks, extracted_data, ["EC", "SALE_DEED"])
        assert checks[0]["data_confidence"] == "LOW"
        assert checks[0]["data_confidence_score"] == 0.55

    def test_filters_by_needed_types(self):
        checks = [{"status": "PASS"}]
        extracted_data = {
            "ec.pdf": {
                "document_type": "EC",
                "data": {"_confidence_score": 0.30},
            },
            "sale.pdf": {
                "document_type": "SALE_DEED",
                "data": {"_confidence_score": 0.90},
            },
        }
        # Only SALE_DEED needed — should use 0.90 (not 0.30)
        _annotate_check_confidence(checks, extracted_data, ["SALE_DEED"])
        assert checks[0]["data_confidence"] == "HIGH"
        assert checks[0]["data_confidence_score"] == 0.9

    def test_no_confidence_info_skips(self):
        checks = [{"status": "FAIL"}]
        extracted_data = {
            "file.pdf": {"document_type": "EC", "data": {"field": "value"}},
        }
        _annotate_check_confidence(checks, extracted_data, ["EC"])
        assert "data_confidence" not in checks[0]

    def test_empty_extracted_data(self):
        checks = [{"status": "PASS"}]
        _annotate_check_confidence(checks, {}, ["EC"])
        assert "data_confidence" not in checks[0]


# ═══════════════════════════════════════════════════
# _validate_group_result
# ═══════════════════════════════════════════════════


class TestValidateGroupResult:

    @pytest.fixture
    def group_ec(self):
        return {"id": 1, "name": "EC-Only Checks"}

    def test_normalizes_invalid_status(self, group_ec):
        result = {
            "checks": [
                {"status": "INVALID", "severity": "HIGH", "rule_code": "X",
                 "explanation": "test", "evidence": "ev", "recommendation": "r"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        assert out["checks"][0]["status"] == "WARNING"

    def test_normalizes_invalid_severity(self, group_ec):
        result = {
            "checks": [
                {"status": "FAIL", "severity": "ULTRA", "rule_code": "X",
                 "explanation": "test", "evidence": "ev", "recommendation": "r"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        assert out["checks"][0]["severity"] == "MEDIUM"

    def test_fail_info_upgraded_to_medium(self, group_ec):
        """Guardrail 1: FAIL+INFO severity → upgraded to MEDIUM."""
        result = {
            "checks": [
                {"status": "FAIL", "severity": "INFO", "rule_code": "Y",
                 "explanation": "bad", "evidence": "some evidence", "recommendation": "fix"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        assert out["checks"][0]["severity"] == "MEDIUM"

    def test_pads_missing_expected_rules(self, group_ec):
        """Missing expected rules are padded as NOT_APPLICABLE."""
        result = {"checks": []}
        out = _validate_group_result(result, group_ec)
        codes = [c["rule_code"] for c in out["checks"]]
        # Group 1 expects ACTIVE_MORTGAGE, MULTIPLE_SALES, LIS_PENDENS
        assert "ACTIVE_MORTGAGE" in codes
        assert "MULTIPLE_SALES" in codes
        assert "LIS_PENDENS" in codes
        for c in out["checks"]:
            assert c["status"] == "NOT_APPLICABLE"

    def test_evidence_filename_guardrail(self, group_ec):
        """Guardrail 2: FAIL evidence not citing any known file."""
        result = {
            "checks": [
                {"status": "FAIL", "severity": "HIGH", "rule_code": "Z",
                 "explanation": "problem found", "evidence": "some evidence but no file ref",
                 "recommendation": "fix"},
            ]
        }
        out = _validate_group_result(result, group_ec, filenames=["ec.pdf", "sale.pdf"])
        check = out["checks"][0]
        assert any("does not cite any uploaded filename" in w
                    for w in check.get("guardrail_warnings", []))

    def test_pass_with_fail_signal_guardrail(self, group_ec):
        """Guardrail 3: PASS but explanation contains fail signals."""
        result = {
            "checks": [
                {"status": "PASS", "severity": "HIGH", "rule_code": "Q",
                 "explanation": "Active mortgage found on the property",
                 "evidence": "mortgage details here",
                 "recommendation": "ok"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        check = out["checks"][0]
        assert any("active mortgage" in w.lower()
                    for w in check.get("guardrail_warnings", []))

    def test_ground_truth_not_set_without_memory_bank(self, group_ec):
        """Without memory_bank, ground_truth.verified = False."""
        result = {
            "checks": [
                {"status": "PASS", "severity": "HIGH", "rule_code": "W",
                 "explanation": "ok", "evidence": "good evidence",
                 "recommendation": "none"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        gt = out["checks"][0]["ground_truth"]
        assert gt["verified"] is False

    def test_unknowable_checks_flagged(self, group_ec):
        """Guardrail 5: unknowable checks marked unreliable."""
        result = {
            "checks": [
                {"status": "PASS", "severity": "MEDIUM", "rule_code": "FLOOD_ZONE",
                 "explanation": "no flood risk", "evidence": "looks fine",
                 "recommendation": "none"},
            ]
        }
        out = _validate_group_result(result, group_ec)
        check = out["checks"][0]
        assert check.get("unreliable") is True

    def test_verification_meta_present(self, group_ec):
        result = {"checks": []}
        out = _validate_group_result(result, group_ec)
        meta = out.get("_verification_meta", {})
        assert "guardrail_flags" in meta
        assert "tool_call_count" in meta

    def test_non_dict_checks_filtered(self, group_ec):
        """Non-dict entries in checks list are silently dropped."""
        result = {
            "checks": ["a string", 42, None, {"status": "PASS", "severity": "HIGH",
                                                "rule_code": "OK",
                                                "explanation": "fine",
                                                "evidence": "evidence here",
                                                "recommendation": "r"}]
        }
        out = _validate_group_result(result, group_ec)
        dict_checks = [c for c in out["checks"] if isinstance(c, dict)]
        assert len(dict_checks) >= 1  # At least our valid one + padding
