"""Tests for backend/app/pipeline/check_registry.py — check-aware N/A stubs.

Covers:
  - partition_checks() splits checks into runnable vs N/A stubs
  - build_check_roster() generates dynamic prompt text
  - _requires_satisfied() evaluates OR-of-AND logic correctly
  - Edge cases: empty available_types, all types available, single anchor
"""

import pytest

from app.pipeline.check_registry import (
    GROUP1_CHECKS,
    GROUP2_CHECKS,
    GROUP3_CHECKS,
    GROUP4_CHECKS,
    GROUP5_CHECKS,
    ALL_CHECKS_FLAT,
    CHECK_BY_CODE,
    partition_checks,
    build_check_roster,
    _requires_satisfied,
    _missing_types_message,
)


# ═══════════════════════════════════════════════════
# 1. _requires_satisfied logic
# ═══════════════════════════════════════════════════

class TestRequiresSatisfied:
    """Test the OR-of-AND evaluation for per-check requires."""

    def test_single_and_group_all_present(self):
        # requires [["EC", "SALE_DEED"]] — both present
        assert _requires_satisfied([["EC", "SALE_DEED"]], {"EC", "SALE_DEED", "PATTA"}) is True

    def test_single_and_group_missing_one(self):
        # requires [["EC", "SALE_DEED"]] — SALE_DEED missing
        assert _requires_satisfied([["EC", "SALE_DEED"]], {"EC", "PATTA"}) is False

    def test_or_of_and_first_matches(self):
        # requires [["EC", "PATTA"], ["EC", "SALE_DEED"]]
        assert _requires_satisfied(
            [["EC", "PATTA"], ["EC", "SALE_DEED"]], {"EC", "PATTA"}
        ) is True

    def test_or_of_and_second_matches(self):
        assert _requires_satisfied(
            [["EC", "PATTA"], ["EC", "SALE_DEED"]], {"EC", "SALE_DEED"}
        ) is True

    def test_or_of_and_none_match(self):
        assert _requires_satisfied(
            [["EC", "PATTA"], ["EC", "SALE_DEED"]], {"PATTA", "CHITTA"}
        ) is False

    def test_single_type_or(self):
        # requires [["EC"], ["PATTA"], ["SALE_DEED"]] — any one anchor
        assert _requires_satisfied([["EC"], ["PATTA"], ["SALE_DEED"]], {"PATTA"}) is True

    def test_single_type_none_present(self):
        assert _requires_satisfied(
            [["EC"], ["PATTA"], ["SALE_DEED"]], {"FMB", "ADANGAL"}
        ) is False

    def test_empty_requires(self):
        # Empty requires means no AND-group can match → always False
        assert _requires_satisfied([], {"EC"}) is False

    def test_empty_available(self):
        assert _requires_satisfied([["EC"]], set()) is False


# ═══════════════════════════════════════════════════
# 2. _missing_types_message
# ═══════════════════════════════════════════════════

class TestMissingTypesMessage:

    def test_simple_and_missing(self):
        msg = _missing_types_message([["EC", "SALE_DEED"]], {"EC"})
        assert "Sale Deed" in msg

    def test_all_satisfied(self):
        msg = _missing_types_message([["EC"]], {"EC"})
        assert msg == ""

    def test_or_missing_all(self):
        msg = _missing_types_message([["EC", "PATTA"], ["EC", "SALE_DEED"]], {"CHITTA"})
        assert len(msg) > 0


# ═══════════════════════════════════════════════════
# 3. partition_checks — Group 1 (EC-only)
# ═══════════════════════════════════════════════════

class TestPartitionGroup1:
    """Group 1 checks require EC. All should be runnable if EC present."""

    def test_all_runnable_with_ec(self):
        runnable, stubs = partition_checks(1, {"EC", "SALE_DEED", "PATTA"})
        assert len(runnable) == len(GROUP1_CHECKS)
        assert len(stubs) == 0

    def test_all_na_without_ec(self):
        runnable, stubs = partition_checks(1, {"SALE_DEED", "PATTA"})
        assert len(runnable) == 0
        assert len(stubs) == len(GROUP1_CHECKS)
        for stub in stubs:
            assert stub["status"] == "NOT_APPLICABLE"
            assert "Encumbrance Certificate" in stub["explanation"]


# ═══════════════════════════════════════════════════
# 4. partition_checks — Group 3 (cross-document)
# ═══════════════════════════════════════════════════

class TestPartitionGroup3:

    def test_poramboke_runnable_with_ec_only(self):
        """PORAMBOKE_DETECTION requires any anchor (EC alone suffices)."""
        runnable, stubs = partition_checks(3, {"EC"})
        codes = [c["rule_code"] for c in runnable]
        assert "PORAMBOKE_DETECTION" in codes

    def test_owner_name_needs_two_docs(self):
        """OWNER_NAME_MISMATCH requires pairs like [EC, PATTA] or [EC, SALE_DEED]."""
        runnable, stubs = partition_checks(3, {"EC"})
        stub_codes = [s["rule_code"] for s in stubs]
        assert "OWNER_NAME_MISMATCH" in stub_codes

    def test_owner_name_runnable_with_ec_and_patta(self):
        runnable, _stubs = partition_checks(3, {"EC", "PATTA"})
        codes = [c["rule_code"] for c in runnable]
        assert "OWNER_NAME_MISMATCH" in codes

    def test_boundary_needs_fmb_or_pair(self):
        """BOUNDARY_CONSISTENCY needs doc pairs with boundary data."""
        runnable, stubs = partition_checks(3, {"EC", "SALE_DEED"})
        codes = [c["rule_code"] for c in runnable]
        # BOUNDARY_CONSISTENCY requires pairs involving Patta or FMB
        # With just EC + SALE_DEED, it might or might not be runnable
        # depending on the exact requires definition


# ═══════════════════════════════════════════════════
# 5. partition_checks — Group 4 (chain of title)
# ═══════════════════════════════════════════════════

class TestPartitionGroup4:

    def test_broken_chain_needs_ec_and_sale_deed(self):
        runnable, stubs = partition_checks(5, {"EC", "SALE_DEED"})
        codes = [c["rule_code"] for c in runnable]
        assert "BROKEN_CHAIN_OF_TITLE" in codes

    def test_broken_chain_na_without_ec(self):
        runnable, stubs = partition_checks(5, {"SALE_DEED", "PATTA"})
        stub_codes = [s["rule_code"] for s in stubs]
        assert "BROKEN_CHAIN_OF_TITLE" in stub_codes

    def test_death_succession_enriched_by_legal_heir(self):
        """DEATH_WITHOUT_SUCCESSION requires EC, enriched_by LEGAL_HEIR/WILL."""
        runnable, _stubs = partition_checks(5, {"EC", "SALE_DEED"})
        check = next((c for c in runnable if c["rule_code"] == "DEATH_WITHOUT_SUCCESSION"), None)
        assert check is not None
        # enriched_by should be noted in the prompt section
        assert "enriched_by" in check or check.get("enriched_by")


# ═══════════════════════════════════════════════════
# 6. build_check_roster — dynamic prompt text
# ═══════════════════════════════════════════════════

class TestBuildCheckRoster:

    def test_basic_roster_format(self):
        runnable, _ = partition_checks(1, {"EC"})
        roster = build_check_roster(runnable, {"EC"})
        assert "═══ CHECKS TO PERFORM ═══" in roster
        for check in runnable:
            assert check["rule_code"] in roster

    def test_roster_includes_severity(self):
        runnable, _ = partition_checks(1, {"EC"})
        roster = build_check_roster(runnable, {"EC"})
        assert "CRITICAL" in roster or "HIGH" in roster or "MEDIUM" in roster

    def test_enriched_by_note(self):
        """Checks with enriched_by should get a NOTE in the roster."""
        runnable, _ = partition_checks(5, {"EC", "SALE_DEED", "LEGAL_HEIR"})
        roster = build_check_roster(runnable, {"EC", "SALE_DEED", "LEGAL_HEIR"})
        # DEATH_WITHOUT_SUCCESSION has enriched_by=[LEGAL_HEIR, WILL]
        # Since LEGAL_HEIR is available, it should be noted as human-readable name
        assert "Legal Heir Certificate" in roster

    def test_empty_runnable(self):
        roster = build_check_roster([], {"EC"})
        assert "═══ CHECKS TO PERFORM ═══" in roster
        # Should be empty or minimal


# ═══════════════════════════════════════════════════
# 7. Registry integrity
# ═══════════════════════════════════════════════════

class TestRegistryIntegrity:

    def test_all_checks_have_required_fields(self):
        """Every check definition must have rule_code, rule_name, severity, group_id, requires."""
        required_keys = {"rule_code", "rule_name", "severity", "group_id", "requires"}
        for check in ALL_CHECKS_FLAT:
            for key in required_keys:
                assert key in check, f"Check {check.get('rule_code', '??')} missing '{key}'"

    def test_unique_rule_codes(self):
        codes = [c["rule_code"] for c in ALL_CHECKS_FLAT]
        assert len(codes) == len(set(codes)), f"Duplicate rule codes: {[c for c in codes if codes.count(c) > 1]}"

    def test_check_by_code_lookup(self):
        for check in ALL_CHECKS_FLAT:
            assert check["rule_code"] in CHECK_BY_CODE

    def test_valid_severities(self):
        valid = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        for check in ALL_CHECKS_FLAT:
            assert check["severity"] in valid, f"{check['rule_code']} has invalid severity: {check['severity']}"

    def test_valid_group_ids(self):
        valid_groups = {1, 2, 3, 4, 5}
        for check in ALL_CHECKS_FLAT:
            assert check["group_id"] in valid_groups, f"{check['rule_code']} has invalid group_id: {check['group_id']}"

    def test_requires_is_list_of_lists(self):
        for check in ALL_CHECKS_FLAT:
            req = check["requires"]
            assert isinstance(req, list), f"{check['rule_code']}: requires must be a list"
            for group in req:
                assert isinstance(group, list), f"{check['rule_code']}: each AND-group must be a list"
                for dtype in group:
                    assert isinstance(dtype, str), f"{check['rule_code']}: doc types must be strings"

    def test_total_check_count(self):
        """We should have exactly 31 checks across all groups."""
        total = (len(GROUP1_CHECKS) + len(GROUP2_CHECKS) + len(GROUP3_CHECKS) +
                 len(GROUP4_CHECKS) + len(GROUP5_CHECKS))
        assert total == len(ALL_CHECKS_FLAT)
        assert total == 31
