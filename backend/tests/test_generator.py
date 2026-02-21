"""Tests for report generator — Sale Deed details extraction and INR formatting."""

import pytest

from app.reports.generator import _extract_sale_deed_details, _format_inr


# ═══════════════════════════════════════════════════════════
# _format_inr
# ═══════════════════════════════════════════════════════════

class TestFormatINR:
    def test_none(self):
        assert _format_inr(None) == ""

    def test_zero(self):
        assert _format_inr(0) == ""

    def test_small(self):
        assert _format_inr(500) == "₹500"

    def test_thousands(self):
        assert _format_inr(5000) == "₹5,000"

    def test_lakhs(self):
        assert _format_inr(250000) == "₹2,50,000"

    def test_crores(self):
        assert _format_inr(15000000) == "₹1,50,00,000"

    def test_string_input(self):
        assert _format_inr("250000") == "₹2,50,000"

    def test_rupee_prefixed_string(self):
        assert _format_inr("₹5,000") == "₹5,000"

    def test_empty_string(self):
        assert _format_inr("") == ""


# ═══════════════════════════════════════════════════════════
# _extract_sale_deed_details
# ═══════════════════════════════════════════════════════════

def _session(sd_data: dict) -> dict:
    """Build minimal session_data with a single sale deed."""
    return {
        "extracted_data": {
            "sale_deed.pdf": {
                "document_type": "SALE_DEED",
                "data": sd_data,
            }
        }
    }


class TestExtractSaleDeedDetails:
    def test_returns_none_when_no_sale_deed(self):
        session = {"extracted_data": {"ec.pdf": {"document_type": "EC", "data": {}}}}
        assert _extract_sale_deed_details(session) is None

    def test_returns_none_for_empty_extracted(self):
        assert _extract_sale_deed_details({"extracted_data": {}}) is None

    def test_registration_info(self):
        sd = {
            "document_number": "5909/2012",
            "registration_date": "13.12.2012",
            "execution_date": "10.12.2012",
            "sro": "Vadavalli",
        }
        result = _extract_sale_deed_details(_session(sd))
        assert result is not None
        assert result["registration"]["document_number"] == "5909/2012"
        assert result["registration"]["execution_date"] == "10.12.2012"

    def test_sellers_and_buyers(self):
        sd = {
            "seller": [{"name": "Raman", "age": 45}],
            "buyer": [{"name": "Kumar", "age": 30}],
        }
        result = _extract_sale_deed_details(_session(sd))
        assert len(result["sellers"]) == 1
        assert result["sellers"][0]["name"] == "Raman"
        assert len(result["buyers"]) == 1

    def test_financials_formatted(self):
        sd = {
            "financials": {
                "consideration_amount": 250000,
                "stamp_duty": 15000,
            }
        }
        result = _extract_sale_deed_details(_session(sd))
        assert result["financials"]["consideration_amount"] == "₹2,50,000"
        assert result["financials"]["stamp_duty"] == "₹15,000"

    def test_ownership_history(self):
        sd = {
            "ownership_history": [
                {"owner": "A", "acquisition_mode": "Purchase", "document_date": "2010"},
                {"owner": "B", "acquisition_mode": "Inheritance", "document_date": "2005"},
            ]
        }
        result = _extract_sale_deed_details(_session(sd))
        assert len(result["ownership_history"]) == 2

    def test_payment_possession_encumbrance(self):
        sd = {
            "payment_mode": "Cash",
            "possession_date": "15.12.2012",
            "encumbrance_declaration": "Free from encumbrance",
        }
        result = _extract_sale_deed_details(_session(sd))
        assert result["payment_mode"] == "Cash"
        assert result["possession_date"] == "15.12.2012"
        assert result["encumbrance_declaration"] == "Free from encumbrance"

    def test_property_description_and_boundaries(self):
        sd = {
            "property_description": "Land in S.F.No. 317, extent 1 acre 14.5 cents",
            "property": {
                "boundaries": {
                    "north": "Road", "south": "Kumar land",
                    "east": "Canal", "west": "Raman land",
                },
                "survey_number": "317",
            },
        }
        result = _extract_sale_deed_details(_session(sd))
        assert result["property_description"] == "Land in S.F.No. 317, extent 1 acre 14.5 cents"
        assert result["boundaries"]["north"] == "Road"

    def test_empty_fields_default(self):
        sd = {"document_number": ""}  # minimal non-empty dict
        result = _extract_sale_deed_details(_session(sd))
        assert result is not None
        assert result["sellers"] == []
        assert result["buyers"] == []
        assert result["ownership_history"] == []
        assert result["payment_mode"] == ""

    def test_empty_data_returns_none(self):
        sd = {}
        assert _extract_sale_deed_details(_session(sd)) is None
