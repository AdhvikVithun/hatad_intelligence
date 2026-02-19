"""Tests for backend/app/pipeline/memory_bank.py — cross-document fact store.

Covers:
  - Fact creation and serialization
  - MemoryBank ingest (EC, SALE_DEED, PATTA, generic)
  - Conflict detection (property, party, financial)
  - Cross-references
  - Serialization round-trip (to_dict / from_dict)
  - Summary generation
  - Verification context building
"""

import pytest

from app.pipeline.memory_bank import Fact, Conflict, MemoryBank


# ═══════════════════════════════════════════════════
# 1. Fact basics
# ═══════════════════════════════════════════════════

class TestFact:

    def test_creation(self):
        f = Fact("property", "survey_number", "311/1", "deed.pdf", "SALE_DEED")
        assert f.category == "property"
        assert f.key == "survey_number"
        assert f.value == "311/1"
        assert f.source_file == "deed.pdf"
        assert f.source_type == "SALE_DEED"
        assert f.confidence == 1.0

    def test_to_dict(self):
        f = Fact("financial", "amount", 5000, "ec.pdf", "EC", context="row 1")
        d = f.to_dict()
        assert d["category"] == "financial"
        assert d["key"] == "amount"
        assert d["value"] == 5000
        assert d["context"] == "row 1"
        assert "timestamp" in d

    def test_repr(self):
        f = Fact("party", "seller", "Raman", "deed.pdf", "SALE_DEED")
        r = repr(f)
        assert "party" in r
        assert "seller" in r
        assert "Raman" in r


# ═══════════════════════════════════════════════════
# 2. MemoryBank ingestion
# ═══════════════════════════════════════════════════

class TestIngestion:

    def test_ingest_ec(self):
        bank = MemoryBank()
        ec_data = {
            "ec_number": "EC-001",
            "property_description": "Land S.F.No. 311/1",
            "period_from": "01-01-2010",
            "period_to": "31-12-2025",
            "village": "Chromepet",
            "taluk": "Tambaram",
            "transactions": [
                {
                    "row_number": 1,
                    "transaction_type": "Sale",
                    "seller_or_executant": "Raman",
                    "buyer_or_claimant": "Muthu",
                    "document_number": "1234",
                    "date": "2015-03-15",
                    "consideration_amount": "10,00,000",
                },
            ],
        }
        count = bank.ingest_document("ec.pdf", "EC", ec_data)
        assert count > 0
        assert "ec.pdf" in bank._ingested_files
        # Should have property, party, reference, timeline, financial facts
        cats = {f.category for f in bank.facts}
        assert "property" in cats
        assert "party" in cats
        assert "reference" in cats

    def test_ingest_sale_deed(self):
        bank = MemoryBank()
        deed_data = {
            "document_number": "5678/2020",
            "registration_date": "20-06-2020",
            "sro": "Tambaram SRO",
            "seller": [{"name": "Muthu"}],
            "buyer": [{"name": "Lakshmi"}],
            "property": {
                "survey_number": "311/1",
                "extent": "2400 sq.ft",
                "village": "Chromepet",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
            },
            "financials": {
                "consideration_amount": "45,00,000",
                "guideline_value": "40,00,000",
                "stamp_duty": "3,15,000",
            },
        }
        count = bank.ingest_document("deed.pdf", "SALE_DEED", deed_data)
        assert count > 0
        # Property facts
        survey_facts = bank.get_facts_by_key("survey_number")
        assert len(survey_facts) >= 1
        assert any(f["value"] == "311/1" for f in survey_facts)

    def test_ingest_patta(self):
        bank = MemoryBank()
        patta_data = {
            "patta_number": "P-001",
            "survey_numbers": [
                {"survey_no": "311/1", "extent": "2400 sq.ft", "classification": "Dry"},
            ],
            "total_extent": "2400 sq.ft",
            "village": "Chromepet",
            "owner_names": [{"name": "Lakshmi", "share": "Full"}],
        }
        count = bank.ingest_document("patta.pdf", "PATTA", patta_data)
        assert count > 0
        owners = [f for f in bank.facts if f.key == "patta_owner"]
        assert len(owners) >= 1

    def test_ingest_none_data(self):
        bank = MemoryBank()
        count = bank.ingest_document("bad.pdf", "EC", None)
        assert count == 0

    def test_ingest_generic(self):
        bank = MemoryBank()
        generic = {
            "document_number": "COURT-001",
            "key_parties": [{"role": "petitioner", "name": "Raman"}],
        }
        count = bank.ingest_document("order.pdf", "COURT_ORDER", generic)
        assert count > 0


# ═══════════════════════════════════════════════════
# 3. Conflict detection
# ═══════════════════════════════════════════════════

class TestConflictDetection:

    def _bank_with_deed_and_patta(self, deed_survey="311/1", patta_survey="311/1",
                                   deed_village="Chromepet", patta_village="Chromepet",
                                   buyer_name="Lakshmi", patta_owner="Lakshmi",
                                   deed_amount="45,00,000", ec_amount=None):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "seller": [{"name": "Seller"}],
            "buyer": [{"name": buyer_name}],
            "property": {"survey_number": deed_survey, "extent": "2400 sq.ft",
                         "village": deed_village},
            "financials": {"consideration_amount": deed_amount},
        })
        bank.ingest_document("patta.pdf", "PATTA", {
            "survey_numbers": [{"survey_no": patta_survey}],
            "village": patta_village,
            "owner_names": [{"name": patta_owner}],
        })
        if ec_amount:
            bank.ingest_document("ec.pdf", "EC", {
                "transactions": [
                    {"transaction_type": "Sale", "consideration_amount": ec_amount,
                     "seller_or_executant": "X", "buyer_or_claimant": "Y",
                     "document_number": "999", "date": "2020"},
                ],
            })
        return bank

    def test_no_conflicts_clean_data(self):
        """Matching survey, village, and party → no conflicts."""
        bank = self._bank_with_deed_and_patta()
        conflicts = bank.detect_conflicts()
        survey_conflicts = [c for c in conflicts if c.key == "survey_number"]
        village_conflicts = [c for c in conflicts if c.key == "village"]
        assert len(survey_conflicts) == 0
        assert len(village_conflicts) == 0

    def test_survey_conflict(self):
        """Different survey numbers → property conflict."""
        bank = self._bank_with_deed_and_patta(deed_survey="311/1", patta_survey="999/5")
        conflicts = bank.detect_conflicts()
        survey_conflicts = [c for c in conflicts if c.key == "survey_number"]
        assert len(survey_conflicts) >= 1
        assert survey_conflicts[0].severity == "HIGH"

    def test_survey_ocr_fuzzy_conflict(self):
        """OCR-fuzzy survey match (non-digit char diff, ≥ 5 chars) → WARNING conflict."""
        bank = self._bank_with_deed_and_patta(deed_survey="3111A", patta_survey="3111B")
        conflicts = bank.detect_conflicts()
        survey_conflicts = [c for c in conflicts if c.key == "survey_number"]
        assert len(survey_conflicts) >= 1
        assert any(c.severity == "WARNING" for c in survey_conflicts)

    def test_survey_digit_change_is_hard_conflict(self):
        """Digit change on short survey (311/1 vs 312/1) → HIGH conflict, not fuzzy WARNING."""
        bank = self._bank_with_deed_and_patta(deed_survey="311/1", patta_survey="312/1")
        conflicts = bank.detect_conflicts()
        survey_conflicts = [c for c in conflicts if c.key == "survey_number"]
        assert len(survey_conflicts) >= 1
        assert any(c.severity == "HIGH" for c in survey_conflicts)

    def test_village_conflict(self):
        """Different villages → property conflict."""
        bank = self._bank_with_deed_and_patta(deed_village="Chromepet", patta_village="Tambaram")
        conflicts = bank.detect_conflicts()
        village_conflicts = [c for c in conflicts if c.key == "village"]
        assert len(village_conflicts) >= 1

    def test_party_conflict(self):
        """Buyer name doesn't match patta owner → party conflict."""
        bank = self._bank_with_deed_and_patta(buyer_name="Raman", patta_owner="Velan")
        conflicts = bank.detect_conflicts()
        party_conflicts = [c for c in conflicts if c.category == "party"]
        assert len(party_conflicts) >= 1

    def test_financial_conflict(self):
        """Deed consideration doesn't match EC amount → financial conflict."""
        bank = self._bank_with_deed_and_patta(deed_amount="45,00,000", ec_amount="30,00,000")
        conflicts = bank.detect_conflicts()
        fin_conflicts = [c for c in conflicts if c.category == "financial"]
        assert len(fin_conflicts) >= 1

    def test_village_cross_script_severity_medium(self):
        """Tamil vs English village name mismatch → MEDIUM, not HIGH."""
        bank = self._bank_with_deed_and_patta(
            deed_village="Tambaram", patta_village="தாம்பரம்OTHER"
        )
        conflicts = bank.detect_conflicts()
        village_conflicts = [c for c in conflicts if c.key == "village"]
        # Should have a conflict but severity should be MEDIUM for cross-script
        if village_conflicts:
            assert village_conflicts[0].severity == "MEDIUM"
            assert "cross-script" in village_conflicts[0].description

    def test_village_same_script_stays_high(self):
        """Same-script village mismatch → severity stays HIGH."""
        bank = self._bank_with_deed_and_patta(
            deed_village="Chromepet", patta_village="Tambaram"
        )
        conflicts = bank.detect_conflicts()
        village_conflicts = [c for c in conflicts if c.key == "village"]
        assert len(village_conflicts) >= 1
        assert village_conflicts[0].severity == "HIGH"


# ═══════════════════════════════════════════════════
# 4. Query & cross-references
# ═══════════════════════════════════════════════════

class TestQueryAndCrossRef:

    def _populated_bank(self):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "seller": [{"name": "Seller"}], "buyer": [{"name": "Buyer"}],
            "property": {"survey_number": "311/1", "village": "Chromepet"},
            "financials": {"consideration_amount": "45,00,000"},
        })
        bank.ingest_document("patta.pdf", "PATTA", {
            "survey_numbers": [{"survey_no": "311/1"}],
            "village": "Chromepet",
            "owner_names": [{"name": "Buyer"}],
        })
        return bank

    def test_get_facts_by_category(self):
        bank = self._populated_bank()
        props = bank.get_facts_by_category("property")
        assert len(props) > 0
        assert all(f["category"] == "property" for f in props)

    def test_get_facts_by_key(self):
        bank = self._populated_bank()
        surveys = bank.get_facts_by_key("survey_number")
        assert len(surveys) >= 2  # From deed and patta

    def test_get_facts_by_source(self):
        bank = self._populated_bank()
        deed_facts = bank.get_facts_by_source("deed.pdf")
        assert len(deed_facts) > 0
        assert all(f["source_file"] == "deed.pdf" for f in deed_facts)

    def test_cross_references(self):
        bank = self._populated_bank()
        xrefs = bank.get_cross_references()
        assert len(xrefs) > 0
        # survey_number should be cross-referenced (appears in both docs)
        survey_xref = [x for x in xrefs if x["key"] == "survey_number"]
        assert len(survey_xref) >= 1
        assert len(survey_xref[0]["sources"]) >= 2


# ═══════════════════════════════════════════════════
# 5. Serialization round-trip
# ═══════════════════════════════════════════════════

class TestSerialization:

    def test_to_dict_from_dict_roundtrip(self):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "seller": [{"name": "Seller"}], "buyer": [{"name": "Buyer"}],
            "property": {"survey_number": "311/1"},
            "financials": {},
        })
        d = bank.to_dict()
        assert "facts" in d
        assert "ingested_files" in d

        restored = MemoryBank.from_dict(d)
        assert len(restored.facts) == len(bank.facts)
        assert restored._ingested_files == bank._ingested_files

    def test_get_summary(self):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "seller": [{"name": "Seller"}], "buyer": [{"name": "Buyer"}],
            "property": {"survey_number": "311/1"},
            "financials": {"consideration_amount": "10,00,000"},
        })
        summary = bank.get_summary()
        assert "total_facts" in summary
        assert summary["total_facts"] > 0
        assert "categories" in summary
        assert "ingested_files" in summary


# ═══════════════════════════════════════════════════
# 6. Verification context
# ═══════════════════════════════════════════════════

class TestVerificationContext:

    def test_context_string(self):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "seller": [{"name": "Seller"}], "buyer": [{"name": "Buyer"}],
            "property": {"survey_number": "311/1", "village": "Chromepet"},
            "financials": {"consideration_amount": "45,00,000"},
        })
        ctx = bank.get_verification_context()
        assert isinstance(ctx, str)
        assert "MEMORY BANK" in ctx
        assert "311/1" in ctx
        assert "Chromepet" in ctx

    def test_context_includes_conflicts(self):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "property": {"survey_number": "311/1", "village": "Chromepet"},
            "financials": {},
        })
        bank.ingest_document("patta.pdf", "PATTA", {
            "survey_numbers": [{"survey_no": "999/5"}],
            "village": "Tambaram",
        })
        bank.detect_conflicts()
        ctx = bank.get_verification_context()
        assert "CONFLICTS DETECTED" in ctx


# ═══════════════════════════════════════════════════
# 7. Fuzzy party name matching
# ═══════════════════════════════════════════════════

class TestFuzzyPartyMatching:
    """Party matching now uses fuzzy cross-script name_similarity."""

    def _bank_with_parties(self, buyer_name, patta_owner):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "buyer": [{"name": buyer_name}],
            "property": {"survey_number": "311/1"},
            "financials": {},
        })
        bank.ingest_document("patta.pdf", "PATTA", {
            "survey_numbers": [{"survey_no": "311/1"}],
            "owner_names": [{"name": patta_owner}],
        })
        return bank

    def test_exact_match_no_conflict(self):
        """Exact name match → no party conflict."""
        bank = self._bank_with_parties("Rani", "Rani")
        conflicts = bank.detect_conflicts()
        party_conflicts = [c for c in conflicts if c.category == "party"]
        assert len(party_conflicts) == 0

    def test_initial_stripped_no_conflict(self):
        """'r. Rani' should match 'Rani' after initial stripping."""
        bank = self._bank_with_parties("r. Rani", "Rani")
        conflicts = bank.detect_conflicts()
        party_conflicts = [c for c in conflicts if c.category == "party"]
        assert len(party_conflicts) == 0

    def test_genuinely_different_names_conflict(self):
        """Genuinely different names → party conflict."""
        bank = self._bank_with_parties("Murugan", "Velan")
        conflicts = bank.detect_conflicts()
        party_conflicts = [c for c in conflicts if c.category == "party"]
        assert len(party_conflicts) >= 1

    def test_case_insensitive_no_conflict(self):
        """Case difference should not cause a conflict."""
        bank = self._bank_with_parties("RANI", "rani")
        conflicts = bank.detect_conflicts()
        party_conflicts = [c for c in conflicts if c.category == "party"]
        assert len(party_conflicts) == 0


# ═══════════════════════════════════════════════════
# 8. Extent unit-aware comparison
# ═══════════════════════════════════════════════════

class TestExtentUnitAware:
    """Extent comparison should normalize units before flagging conflicts."""

    def _bank_with_extents(self, deed_extent, patta_extent):
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", {
            "property": {"survey_number": "311/1", "extent": deed_extent},
            "financials": {},
        })
        bank.ingest_document("patta.pdf", "PATTA", {
            "survey_numbers": [{"survey_no": "311/1"}],
            "total_extent": patta_extent,
        })
        return bank

    def test_same_unit_no_conflict(self):
        """Same extent in same units → no conflict."""
        bank = self._bank_with_extents("2400 sq.ft", "2400 sq.ft")
        conflicts = bank.detect_conflicts()
        extent_conflicts = [c for c in conflicts if c.key == "extent"]
        assert len(extent_conflicts) == 0

    def test_different_units_same_area_no_conflict(self):
        """Same area in different units → no conflict (within 10%)."""
        bank = self._bank_with_extents("1 acres", "100 cents")
        conflicts = bank.detect_conflicts()
        extent_conflicts = [c for c in conflicts if c.key == "extent"]
        assert len(extent_conflicts) == 0

    def test_genuinely_different_extents_conflict(self):
        """Genuinely different areas → extent conflict."""
        bank = self._bank_with_extents("5 acres", "10 acres")
        conflicts = bank.detect_conflicts()
        extent_conflicts = [c for c in conflicts if c.key == "extent"]
        assert len(extent_conflicts) >= 1

    def test_hectares_vs_acres_same_property(self):
        """0.9250 hectares ≈ 2.29 acres → same property, no conflict."""
        bank = self._bank_with_extents("0.9250 hectares", "2.29 acres")
        conflicts = bank.detect_conflicts()
        extent_conflicts = [c for c in conflicts if c.key == "extent"]
        # These should be approximately equal (within 10%)
        assert len(extent_conflicts) == 0
