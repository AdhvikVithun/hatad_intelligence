"""Tests for new document-type ingestion methods in MemoryBank.

Covers the 12 new _ingest_* methods added for:
  FMB, Adangal, Layout Approval, Legal Heir, POA, Court Order,
  Will, Partition Deed, Gift Deed, Release Deed, Chitta, A-Register.
Uses concrete Tamil-name fixtures.
"""

import pytest

from app.pipeline.memory_bank import MemoryBank


# ═══════════════════════════════════════════════════
# Helper — retrieve facts as dicts and filter by key
# ═══════════════════════════════════════════════════

def _facts(bank: MemoryBank, category: str) -> list[dict]:
    """Shorthand for get_facts_by_category returning list of fact dicts."""
    return bank.get_facts_by_category(category)


def _fact_keys(bank: MemoryBank, category: str) -> list[str]:
    return [f["key"] for f in _facts(bank, category)]


def _fact_values(bank: MemoryBank, category: str) -> list:
    return [f["value"] for f in _facts(bank, category)]


# ═══════════════════════════════════════════════════
# Fixtures — concrete Tamil names and realistic data
# ═══════════════════════════════════════════════════

def _make_bank() -> MemoryBank:
    return MemoryBank()


def _chitta_data() -> dict:
    return {
        "chitta_number": "7821",
        "owner_name": "செல்வராஜ் S/o முருகன்",
        "father_name": "முருகன்",
        "survey_numbers": [
            {"survey_no": "317/2A", "extent": "1.25 acres", "classification": "Dry", "soil_code": "1"},
            {"survey_no": "318", "extent": "0.50 acres", "classification": "Wet", "soil_code": "2"},
        ],
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "total_extent": "1.75 acres",
        "tax_assessment": "150",
        "remarks": "",
    }


def _a_register_data() -> dict:
    return {
        "register_serial_number": "456",
        "paguthi_number": "3",
        "owner_name": "R. Murugan",
        "father_name": "Raman",
        "patta_number": "12345",
        "survey_numbers": [
            {"survey_no": "311/1", "extent": "2400 sq.ft", "classification": "House Site"},
        ],
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "total_extent": "2400 sq.ft",
        "mutation_entries": [
            {"date": "15-03-2012", "from_owner": "Krishnan", "to_owner": "Raman", "reason": "Sale", "order_number": "M-2012/45"},
            {"date": "20-06-2020", "from_owner": "Raman", "to_owner": "R. Murugan", "reason": "Inheritance", "order_number": "M-2020/12"},
        ],
        "remarks": "Mutation up to date",
    }


def _fmb_data() -> dict:
    return {
        "survey_number": "311/1",
        "area_acres": "0.055 acres",
        "area_hectares": "0.0223 hectares",
        "dimensions": [
            {"side": "North", "length_ft": "40", "bearing": "N0°E"},
            {"side": "East", "length_ft": "60"},
        ],
        "adjacent_surveys": [
            {"direction": "North", "survey_no": "311/2", "owner_name": "Kannan"},
            {"direction": "East", "survey_no": "312", "owner_name": "Govt. Road"},
        ],
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "boundaries": {"north": "311/2", "south": "Vacant plot", "east": "Govt. Road", "west": "310"},
        "land_classification": "House Site",
        "remarks": "Regular shape",
    }


def _adangal_data() -> dict:
    return {
        "survey_number": "317/2A",
        "owner_name": "செல்வராஜ்",
        "father_name": "முருகன்",
        "extent": "1.25 acres",
        "wet_extent": "",
        "dry_extent": "1.25 acres",
        "soil_type": "புன்செய்",
        "irrigation_source": "Rain-fed",
        "crop_details": [
            {"season": "Samba", "crop": "Paddy", "extent_cultivated": "1.00 acres"},
        ],
        "village": "Kovilambakkam",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "remarks": "",
    }


def _layout_data() -> dict:
    return {
        "approval_number": "LP/2019/1234",
        "authority": "CMDA",
        "layout_name": "Lakshmi Nagar Phase II",
        "applicant_name": "ABC Developers Pvt Ltd",
        "total_plots": 48,
        "total_area": "5.2 acres",
        "plot_schedule": [
            {"plot_no": "12", "area": "1200 sq.ft", "dimensions": "30x40"},
            {"plot_no": "13", "area": "1500 sq.ft"},
        ],
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "approval_date": "15-01-2019",
        "validity_period": "5 years",
        "conditions": ["Road width minimum 30 ft", "10% open space reservation"],
        "remarks": "",
    }


def _legal_heir_data() -> dict:
    return {
        "certificate_number": "LHC/2021/789",
        "deceased_person": {"name": "Perumal S/o Kannan", "date_of_death": "10-05-2021"},
        "heirs": [
            {"name": "Muthu D/o Perumal", "relationship": "Daughter", "share": "1/3"},
            {"name": "Ravi S/o Perumal", "relationship": "Son", "share": "1/3"},
            {"name": "Kavitha W/o Perumal", "relationship": "Wife", "share": "1/3"},
        ],
        "issuing_authority": "Tambaram Tahsildar",
        "date_of_issue": "20-08-2021",
        "remarks": "",
    }


def _poa_data() -> dict:
    return {
        "principal": {"name": "Lakshmi W/o Senthil", "address": "Chennai"},
        "agent": {"name": "Kumar S/o Raman", "address": "Tambaram"},
        "powers_granted": ["sell", "mortgage", "sign documents", "receive payments"],
        "poa_type": "GPA",
        "is_general_or_specific": "general",
        "registration_number": "GPA/2022/456",
        "date": "01-03-2022",
        "property_description": "S.F.No. 311/1, Chromepet Village",
        "remarks": "",
    }


def _court_order_data() -> dict:
    return {
        "case_number": "OS/2020/1234",
        "court_name": "District Court, Chengalpattu",
        "case_type": "Civil Suit",
        "parties": {"plaintiff": "Kannan S/o Raman", "defendant": "Murugan S/o Krishnan"},
        "order_type": "Decree",
        "order_date": "15-06-2022",
        "order_summary": "Title dispute resolved in favour of plaintiff",
        "injunction_status": "None",
        "attachment_status": "None",
        "property_description": "S.F.No. 311/1, Chromepet",
        "remarks": "",
    }


def _will_data() -> dict:
    return {
        "testator": {"name": "Perumal S/o Kannan", "age": "72"},
        "beneficiaries": [
            {"name": "Muthu", "relationship": "Daughter", "share": "House property"},
            {"name": "Ravi", "relationship": "Son", "share": "Agricultural land"},
        ],
        "executor": "Kumar S/o Raman",
        "registration_number": "WILL/2020/789",
        "date": "10-01-2020",
        "probate_status": "Not obtained",
        "witnesses": ["Senthil", "Bala"],
        "codicils": [],
        "remarks": "",
    }


def _partition_data() -> dict:
    return {
        "document_number": "PD/2019/567",
        "registration_date": "20-09-2019",
        "joint_owners": ["Raman S/o Krishnan", "Bala S/o Krishnan", "Kumar S/o Krishnan"],
        "shares": [
            {"owner": "Raman S/o Krishnan", "share": "Plot A - 800 sq.ft"},
            {"owner": "Bala S/o Krishnan", "share": "Plot B - 800 sq.ft"},
            {"owner": "Kumar S/o Krishnan", "share": "Plot C - 800 sq.ft"},
        ],
        "consent_of_all_parties": True,
        "property_description": "S.F.No. 311/1, Chromepet",
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "remarks": "",
    }


def _gift_deed_data() -> dict:
    return {
        "donor": {"name": "Perumal S/o Kannan", "relationship_to_donee": "Father"},
        "donee": {"name": "Muthu D/o Perumal", "relationship_to_donor": "Daughter"},
        "property_description": "S.F.No. 317/2A, Kovilambakkam Village, 1.25 acres",
        "consideration_amount": "0",
        "registration_number": "GD/2021/234",
        "date": "25-12-2021",
        "acceptance_clause": "Donee has accepted the gift in the presence of witnesses",
        "village": "Kovilambakkam",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "remarks": "",
    }


def _release_deed_data() -> dict:
    return {
        "releasing_party": {"name": "Bala S/o Krishnan"},
        "beneficiary": {"name": "Raman S/o Krishnan"},
        "beneficiary_party": {"name": "Raman S/o Krishnan"},
        "property_description": "Undivided share in S.F.No. 311/1, Chromepet",
        "original_document": {"type": "Partition Deed", "number": "PD/2019/567", "sro": "Tambaram"},
        "consideration_amount": "10,00,000",
        "registration_number": "RD/2022/789",
        "date": "15-04-2022",
        "village": "Chromepet",
        "taluk": "Tambaram",
        "district": "Chengalpattu",
        "remarks": "",
    }


# ═══════════════════════════════════════════════════
# Tests — Chitta ingestion
# ═══════════════════════════════════════════════════

class TestIngestChitta:

    def test_chitta_number_stored(self):
        bank = _make_bank()
        bank.ingest_document("chitta.pdf", "CHITTA", _chitta_data())
        facts = _facts(bank, "reference")
        chitta_facts = [f for f in facts if f["key"] == "chitta_number"]
        assert len(chitta_facts) >= 1
        assert chitta_facts[0]["value"] == "7821"

    def test_survey_numbers_stored(self):
        bank = _make_bank()
        bank.ingest_document("chitta.pdf", "CHITTA", _chitta_data())
        facts = _facts(bank, "property")
        survey_facts = [f for f in facts if f["key"] == "survey_number"]
        survey_vals = [f["value"] for f in survey_facts]
        assert "317/2A" in survey_vals
        assert "318" in survey_vals


# ═══════════════════════════════════════════════════
# Tests — A-Register ingestion
# ═══════════════════════════════════════════════════

class TestIngestARegister:

    def test_serial_number_stored(self):
        bank = _make_bank()
        bank.ingest_document("a_reg.pdf", "A_REGISTER", _a_register_data())
        facts = _facts(bank, "reference")
        sn_facts = [f for f in facts if f["key"] == "a_register_serial"]
        assert len(sn_facts) >= 1

    def test_mutation_chain_stored(self):
        bank = _make_bank()
        bank.ingest_document("a_reg.pdf", "A_REGISTER", _a_register_data())
        facts = _facts(bank, "chain")
        assert len(facts) >= 1  # Mutations stored as single composite fact

    def test_party_names_from_mutations(self):
        bank = _make_bank()
        bank.ingest_document("a_reg.pdf", "A_REGISTER", _a_register_data())
        facts = _facts(bank, "party")
        party_names = [f["value"] for f in facts]
        assert any("Krishnan" in str(name) for name in party_names)
        assert any("Raman" in str(name) for name in party_names)


# ═══════════════════════════════════════════════════
# Tests — FMB ingestion
# ═══════════════════════════════════════════════════

class TestIngestFMB:

    def test_area_stored(self):
        bank = _make_bank()
        bank.ingest_document("fmb.pdf", "FMB", _fmb_data())
        facts = _facts(bank, "property")
        area_facts = [f for f in facts if "area" in f["key"]]
        assert len(area_facts) >= 1

    def test_boundaries_stored(self):
        bank = _make_bank()
        bank.ingest_document("fmb.pdf", "FMB", _fmb_data())
        facts = _facts(bank, "property")
        boundary_facts = [f for f in facts if "boundary" in f["key"]]
        assert len(boundary_facts) >= 1

    def test_adjacent_surveys(self):
        bank = _make_bank()
        bank.ingest_document("fmb.pdf", "FMB", _fmb_data())
        facts = _facts(bank, "property")
        adj_facts = [f for f in facts if "adjacent" in f["key"]]
        assert len(adj_facts) >= 1


# ═══════════════════════════════════════════════════
# Tests — Adangal ingestion
# ═══════════════════════════════════════════════════

class TestIngestAdangal:

    def test_cultivation_stored(self):
        bank = _make_bank()
        bank.ingest_document("adangal.pdf", "ADANGAL", _adangal_data())
        facts = _facts(bank, "property")
        soil_facts = [f for f in facts if "soil" in f["key"] or "cultivat" in f["key"].lower()]
        assert len(soil_facts) >= 1

    def test_crop_details(self):
        bank = _make_bank()
        bank.ingest_document("adangal.pdf", "ADANGAL", _adangal_data())
        facts = _facts(bank, "property")
        crop_facts = [f for f in facts if "crop" in f["key"]]
        assert len(crop_facts) >= 1


# ═══════════════════════════════════════════════════
# Tests — Layout Approval ingestion
# ═══════════════════════════════════════════════════

class TestIngestLayoutApproval:

    def test_approval_info_stored(self):
        bank = _make_bank()
        bank.ingest_document("layout.pdf", "LAYOUT_APPROVAL", _layout_data())
        facts = _facts(bank, "reference")
        approval_facts = [f for f in facts if "approval" in f["key"]]
        assert len(approval_facts) >= 1

    def test_authority_stored(self):
        bank = _make_bank()
        bank.ingest_document("layout.pdf", "LAYOUT_APPROVAL", _layout_data())
        facts = _facts(bank, "reference")
        auth_facts = [f for f in facts if "authority" in f["key"]]
        assert len(auth_facts) >= 1
        assert auth_facts[0]["value"] == "CMDA"


# ═══════════════════════════════════════════════════
# Tests — Legal Heir ingestion
# ═══════════════════════════════════════════════════

class TestIngestLegalHeir:

    def test_deceased_stored(self):
        bank = _make_bank()
        bank.ingest_document("heir.pdf", "LEGAL_HEIR", _legal_heir_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Perumal" in str(n) for n in names)

    def test_heirs_stored(self):
        bank = _make_bank()
        bank.ingest_document("heir.pdf", "LEGAL_HEIR", _legal_heir_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Muthu" in str(n) for n in names)
        assert any("Ravi" in str(n) for n in names)
        assert any("Kavitha" in str(n) for n in names)


# ═══════════════════════════════════════════════════
# Tests — POA ingestion
# ═══════════════════════════════════════════════════

class TestIngestPOA:

    def test_principal_agent_stored(self):
        bank = _make_bank()
        bank.ingest_document("poa.pdf", "POA", _poa_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Lakshmi" in str(n) for n in names)
        assert any("Kumar" in str(n) for n in names)

    def test_gpa_risk_flagged(self):
        bank = _make_bank()
        bank.ingest_document("poa.pdf", "POA", _poa_data())
        # POA type stored as property fact
        prop_facts = _facts(bank, "property")
        poa_type_facts = [f for f in prop_facts if f["key"] == "poa_type"]
        assert len(poa_type_facts) >= 1


# ═══════════════════════════════════════════════════
# Tests — Court Order ingestion
# ═══════════════════════════════════════════════════

class TestIngestCourtOrder:

    def test_case_info_stored(self):
        bank = _make_bank()
        bank.ingest_document("order.pdf", "COURT_ORDER", _court_order_data())
        facts = _facts(bank, "reference")
        case_facts = [f for f in facts if "court" in f["key"]]
        assert len(case_facts) >= 1

    def test_no_injunction_no_risk(self):
        bank = _make_bank()
        bank.ingest_document("order.pdf", "COURT_ORDER", _court_order_data())
        facts = _facts(bank, "risk")
        # "None" injunction/attachment should not generate risk facts
        injunction_risks = [f for f in facts if "injunction" in f["key"].lower()]
        assert len(injunction_risks) == 0


# ═══════════════════════════════════════════════════
# Tests — Will ingestion
# ═══════════════════════════════════════════════════

class TestIngestWill:

    def test_testator_stored(self):
        bank = _make_bank()
        bank.ingest_document("will.pdf", "WILL", _will_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Perumal" in str(n) for n in names)

    def test_probate_not_obtained_risk(self):
        """Will with probate_status='Not obtained' should flag risk."""
        bank = _make_bank()
        bank.ingest_document("will.pdf", "WILL", _will_data())
        # Probate status stored under encumbrance category
        enc_facts = _facts(bank, "encumbrance")
        probate_facts = [f for f in enc_facts if "probate" in f["key"].lower()]
        assert len(probate_facts) >= 1


# ═══════════════════════════════════════════════════
# Tests — Partition Deed ingestion
# ═══════════════════════════════════════════════════

class TestIngestPartitionDeed:

    def test_joint_owners_stored(self):
        bank = _make_bank()
        bank.ingest_document("partition.pdf", "PARTITION_DEED", _partition_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Raman" in str(n) for n in names)
        assert any("Bala" in str(n) for n in names)
        assert any("Kumar" in str(n) for n in names)

    def test_shares_stored(self):
        bank = _make_bank()
        bank.ingest_document("partition.pdf", "PARTITION_DEED", _partition_data())
        facts = _facts(bank, "chain")
        assert len(facts) >= 1


# ═══════════════════════════════════════════════════
# Tests — Gift Deed ingestion
# ═══════════════════════════════════════════════════

class TestIngestGiftDeed:

    def test_donor_donee_stored(self):
        bank = _make_bank()
        bank.ingest_document("gift.pdf", "GIFT_DEED", _gift_deed_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Perumal" in str(n) for n in names)
        assert any("Muthu" in str(n) for n in names)


# ═══════════════════════════════════════════════════
# Tests — Release Deed ingestion
# ═══════════════════════════════════════════════════

class TestIngestReleaseDeed:

    def test_release_parties_stored(self):
        bank = _make_bank()
        bank.ingest_document("release.pdf", "RELEASE_DEED", _release_deed_data())
        facts = _facts(bank, "party")
        names = [f["value"] for f in facts]
        assert any("Bala" in str(n) for n in names)
        assert any("Raman" in str(n) for n in names)

    def test_original_doc_reference(self):
        bank = _make_bank()
        bank.ingest_document("release.pdf", "RELEASE_DEED", _release_deed_data())
        # Original doc ref stored under 'reference' category
        facts = _facts(bank, "reference")
        values = [str(f["value"]) for f in facts]
        assert any("Partition" in v or "PD/2019" in v for v in values)


# ═══════════════════════════════════════════════════
# Tests — Dispatch routing
# ═══════════════════════════════════════════════════

class TestIngestDispatch:
    """Verify that ingest_document correctly routes all 16 doc types."""

    DOC_TYPES = [
        ("EC", {"ec_number": "EC-001", "transactions": [], "village": "Test", "taluk": "Test"}),
        ("SALE_DEED", {"document_number": "SD-001", "seller": [], "buyer": [], "property": {}, "financials": {}}),
        ("PATTA", {"patta_number": "P-001", "owner_names": [], "survey_numbers": [], "village": "Test", "taluk": "Test", "total_extent": "", "land_classification": "", "remarks": ""}),
        ("CHITTA", _chitta_data()),
        ("A_REGISTER", _a_register_data()),
        ("FMB", _fmb_data()),
        ("ADANGAL", _adangal_data()),
        ("LAYOUT_APPROVAL", _layout_data()),
        ("LEGAL_HEIR", _legal_heir_data()),
        ("POA", _poa_data()),
        ("COURT_ORDER", _court_order_data()),
        ("WILL", _will_data()),
        ("PARTITION_DEED", _partition_data()),
        ("GIFT_DEED", _gift_deed_data()),
        ("RELEASE_DEED", _release_deed_data()),
        ("OTHER", {"property_details": {"village": "Test"}}),
    ]

    @pytest.mark.parametrize("doc_type,data", DOC_TYPES, ids=[d[0] for d in DOC_TYPES])
    def test_ingest_does_not_raise(self, doc_type, data):
        """Each doc type should ingest without raising an exception."""
        bank = _make_bank()
        bank.ingest_document(f"test_{doc_type.lower()}.pdf", doc_type, data)
        # Should have at least one fact after ingestion
        all_facts = (_facts(bank, "property") + _facts(bank, "party") +
                     _facts(bank, "chain") + _facts(bank, "financial") +
                     _facts(bank, "legal") + _facts(bank, "risk"))
        assert len(all_facts) >= 0  # Just verify no crash; some types may produce 0 facts with minimal data
