"""Shared fixtures for HATAD due-diligence test suite."""

import pytest
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════
# Extracted-data fixtures (dicts shaped like real pipeline output)
# ═══════════════════════════════════════════════════

@pytest.fixture
def ec_basic():
    """Minimal EC with 2 transactions and valid period."""
    now = datetime.now()
    return {
        "ec.pdf": {
            "document_type": "EC",
            "data": {
                "ec_number": "EC-2024-001",
                "property_description": "Land in S.F.No. 311/1, Chromepet village",
                "period_from": "01-01-2010",
                "period_to": now.strftime("%d-%m-%Y"),
                "village": "Chromepet",
                "taluk": "Tambaram",
                "transactions": [
                    {
                        "row_number": 1,
                        "date": "15-03-2012",
                        "transaction_type": "Sale",
                        "seller_or_executant": "Raman S/o Krishnan",
                        "buyer_or_claimant": "Muthu D/o Perumal",
                        "document_number": "1234/2012",
                        "consideration_amount": "15,00,000",
                        "survey_number": "311/1",
                    },
                    {
                        "row_number": 2,
                        "date": "20-06-2020",
                        "transaction_type": "Sale",
                        "seller_or_executant": "Muthu D/o Perumal",
                        "buyer_or_claimant": "Lakshmi W/o Senthil",
                        "document_number": "5678/2020",
                        "consideration_amount": "45,00,000",
                        "survey_number": "311/1",
                    },
                ],
            },
        }
    }


@pytest.fixture
def sale_deed_basic():
    """Minimal sale deed with property + financials."""
    return {
        "sale_deed.pdf": {
            "document_type": "SALE_DEED",
            "data": {
                "document_number": "5678/2020",
                "registration_date": "20-06-2020",
                "sro": "Tambaram SRO",
                "seller": [{"name": "Muthu D/o Perumal", "father_name": "Perumal"}],
                "buyer": [{"name": "Lakshmi W/o Senthil", "father_name": "Senthil"}],
                "property": {
                    "survey_number": "311/1",
                    "extent": "2400 sq.ft",
                    "village": "Chromepet",
                    "taluk": "Tambaram",
                    "district": "Chengalpattu",
                    "boundaries": "North: Road, South: Plot 28",
                },
                "financials": {
                    "consideration_amount": "45,00,000",
                    "guideline_value": "40,00,000",
                    "stamp_duty": "3,15,000",
                    "registration_fee": "1,80,000",
                },
            },
        }
    }


@pytest.fixture
def patta_basic():
    """Minimal patta doc."""
    return {
        "patta.pdf": {
            "document_type": "PATTA",
            "data": {
                "patta_number": "12345",
                "survey_numbers": [
                    {"survey_no": "311/1", "extent": "2400 sq.ft", "classification": "Dry"},
                ],
                "total_extent": "2400 sq.ft",
                "village": "Chromepet",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
                "owner_names": [{"name": "Lakshmi", "father_name": "Senthil", "share": "Full"}],
            },
        }
    }


@pytest.fixture
def all_docs(ec_basic, sale_deed_basic, patta_basic):
    """Combined extracted_data with EC + Sale Deed + Patta — consistent data."""
    merged = {}
    merged.update(ec_basic)
    merged.update(sale_deed_basic)
    merged.update(patta_basic)
    return merged
