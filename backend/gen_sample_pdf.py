"""Generate a sample PDF with realistic data to preview the new template."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.reports.generator import generate_pdf_report

session_data = {
    "session_id": "preview01",
    "risk_score": 78,
    "risk_band": "MEDIUM",
    "documents": [
        {"filename": "ec_chromepet.pdf", "classification": {"doc_type": "EC", "confidence": "0.95"}},
        {"filename": "patta_311.pdf", "classification": {"doc_type": "PATTA", "confidence": "0.92"}},
        {"filename": "sale_deed_2020.pdf", "classification": {"doc_type": "SALE_DEED", "confidence": "0.97"}},
    ],
    "extracted_data": {
        "ec_chromepet.pdf": {
            "document_type": "EC",
            "data": {
                "village": "Chromepet",
                "taluk": "Tambaram",
                "period_from": "01-01-2010",
                "period_to": "31-12-2024",
                "transactions": [
                    {"survey_number": "311/1", "sro": "Tambaram",
                     "document_number": "1234/2012", "date": "15-03-2012",
                     "transaction_type": "Sale", "from": "Raman S/o Krishnan",
                     "to": "Muthu D/o Perumal", "consideration": "15,00,000"},
                    {"survey_number": "311/1",
                     "document_number": "5678/2020", "date": "20-06-2020",
                     "transaction_type": "Sale", "from": "Muthu D/o Perumal",
                     "to": "Lakshmi W/o Senthil", "consideration": "45,00,000"},
                    {"survey_number": "311/1",
                     "document_number": "8901/2018", "date": "10-01-2018",
                     "transaction_type": "Mortgage", "from": "Muthu D/o Perumal",
                     "to": "State Bank of India", "consideration": "20,00,000"},
                    {"survey_number": "311/1",
                     "document_number": "8902/2019", "date": "05-11-2019",
                     "transaction_type": "Release", "from": "State Bank of India",
                     "to": "Muthu D/o Perumal", "consideration": "--"},
                ],
            },
        },
        "patta_311.pdf": {
            "document_type": "PATTA",
            "data": {
                "village": "Chromepet",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
                "patta_number": "54321",
                "total_extent": "2400 sq.ft (3.30 cents)",
                "land_classification": "Dry Land (Punjai)",
                "owner_names": [
                    {"name": "Lakshmi W/o Senthil", "share": "Full"},
                ],
                "survey_numbers": [
                    {"survey_no": "311/1", "extent": "2400 sq.ft"},
                ],
            },
        },
        "sale_deed_2020.pdf": {
            "document_type": "SALE_DEED",
            "data": {
                "document_number": "5678/2020",
                "registration_date": "20-06-2020",
                "execution_date": "15-06-2020",
                "sro": "Tambaram",
                "property": {
                    "village": "Chromepet",
                    "taluk": "Tambaram",
                    "district": "Chengalpattu",
                    "survey_number": "311/1",
                    "extent": "2400 sq.ft",
                    "property_type": "Residential Plot",
                    "boundaries": {
                        "north": "30 ft Road",
                        "south": "S.F.No. 312 - Vacant land",
                        "east": "Irrigation Canal",
                        "west": "S.F.No. 310 - House of Kumar",
                    },
                },
                "seller": [
                    {"name": "Muthu D/o Perumal", "age": "55",
                     "relation": "D/o Late Perumal"},
                ],
                "buyer": [
                    {"name": "Lakshmi W/o Senthil", "age": "42",
                     "relation": "W/o Senthil Kumar"},
                ],
                "financials": {
                    "consideration_amount": 4500000,
                    "guideline_value": 4000000,
                    "stamp_duty": 315000,
                    "registration_fee": 45000,
                },
                "payment_mode": "Demand Draft (DD No. 445566, Indian Bank)",
                "possession_date": "20-06-2020",
                "encumbrance_declaration": "The Seller hereby declares that the "
                    "Schedule Property is free from all encumbrances, liens, "
                    "charges, mortgages, claims, and litigation whatsoever, and "
                    "the Seller has absolute right and title to sell and convey "
                    "the same.",
                "property_description": "All that piece and parcel of land "
                    "bearing Old S.F.No. 311/1 (New S.F.No. 311/1A), situate "
                    "at Chromepet Village, Tambaram Taluk, Chengalpattu District, "
                    "measuring an extent of 2400 sq.ft (3.30 cents) bounded on "
                    "all sides as described in the Schedule hereunder.",
                "ownership_history": [
                    {"owner": "State (Original Grant)",
                     "acquired_from": "--", "acquisition_mode": "Grant",
                     "document_number": "--", "document_date": "Pre-2000"},
                    {"owner": "Raman S/o Krishnan",
                     "acquired_from": "State", "acquisition_mode": "Patta Transfer",
                     "document_number": "Rev. 45/2001", "document_date": "12-04-2001"},
                    {"owner": "Muthu D/o Perumal",
                     "acquired_from": "Raman S/o Krishnan", "acquisition_mode": "Sale",
                     "document_number": "1234/2012", "document_date": "15-03-2012"},
                    {"owner": "Lakshmi W/o Senthil",
                     "acquired_from": "Muthu D/o Perumal", "acquisition_mode": "Sale",
                     "document_number": "5678/2020", "document_date": "20-06-2020"},
                ],
                "witnesses": [
                    {"name": "Arun Kumar", "address": "12, Anna Nagar, Tambaram"},
                    {"name": "Priya Devi", "address": "45, Gandhi Road, Chromepet"},
                ],
            },
        },
    },
    "verification_result": {
        "executive_summary": "Property at S.F.No. 311/1, Chromepet village shows "
            "a clear chain of title from State grant through Raman to Muthu to "
            "current owner Lakshmi. All EC transactions reconcile. Mortgage by "
            "SBI was fully released prior to the 2020 sale. Minor stamp duty "
            "variance detected (7% vs 7% guideline -- within tolerance). "
            "No poramboke or government land indicators found. Recommend "
            "standard legal verification before closing.",
        "checks": [
            {"rule_name": "EC Period Coverage", "rule_code": "DET_EC_PERIOD",
             "status": "PASS", "severity": "HIGH",
             "explanation": "EC covers 2010-2024, exceeding the required 13-year lookback period.",
             "evidence": "Period: 01-01-2010 to 31-12-2024 (14 years)", "data_confidence": "HIGH"},
            {"rule_name": "Chain of Title Continuity", "rule_code": "DET_CHAIN",
             "status": "PASS", "severity": "CRITICAL",
             "explanation": "All ownership transfers form an unbroken chain from original grant to current owner.",
             "evidence": "State > Raman > Muthu > Lakshmi", "data_confidence": "HIGH"},
            {"rule_name": "Survey Number Consistency", "rule_code": "LLM_SURVEY",
             "status": "PASS", "severity": "HIGH",
             "explanation": "Survey number 311/1 is consistent across EC, Patta, and Sale Deed.",
             "evidence": "EC: 311/1, Patta: 311/1, Sale Deed: 311/1", "data_confidence": "HIGH"},
            {"rule_name": "Stamp Duty Compliance", "rule_code": "DET_STAMP_DUTY",
             "status": "WARNING", "severity": "MEDIUM",
             "explanation": "Stamp duty paid is Rs.3,15,000 which is 7% of consideration. Matches statutory rate but guideline value is lower than consideration.",
             "evidence": "Paid: 3,15,000; Consideration: 45,00,000; Rate: 7%",
             "data_confidence": "MEDIUM", "recommendation": "Verify stamp duty calculation with SRO records."},
            {"rule_name": "Poramboke Land Check", "rule_code": "LLM_PORAMBOKE",
             "status": "PASS", "severity": "CRITICAL",
             "explanation": "No government land, poramboke, or waterway indicators found in any document.",
             "evidence": "Classification: Dry Land (Punjai)", "data_confidence": "HIGH"},
            {"rule_name": "Active Encumbrance Check", "rule_code": "DET_ENCUMBRANCE",
             "status": "PASS", "severity": "CRITICAL",
             "explanation": "SBI mortgage (2018) was formally released in 2019. No active encumbrances at time of sale.",
             "evidence": "Mortgage: 8901/2018, Release: 8902/2019", "data_confidence": "HIGH"},
            {"rule_name": "Owner Name Match (EC vs Patta)", "rule_code": "DET_NAME_MATCH",
             "status": "PASS", "severity": "HIGH",
             "explanation": "Current owner Lakshmi W/o Senthil appears in both EC (latest claimant) and Patta.",
             "evidence": "EC: Lakshmi W/o Senthil, Patta: Lakshmi W/o Senthil", "data_confidence": "HIGH"},
            {"rule_name": "Extent Consistency", "rule_code": "DET_EXTENT",
             "status": "PASS", "severity": "MEDIUM",
             "explanation": "Property extent 2400 sq.ft is consistent across all three documents.",
             "evidence": "EC: implied, Patta: 2400 sq.ft, Sale Deed: 2400 sq.ft", "data_confidence": "HIGH"},
            {"rule_name": "Guideline Value Compliance", "rule_code": "DET_GUIDELINE",
             "status": "PASS", "severity": "MEDIUM",
             "explanation": "Consideration (45L) exceeds guideline value (40L). No undervaluation detected.",
             "evidence": "Consideration: 45,00,000; Guideline: 40,00,000", "data_confidence": "HIGH"},
            {"rule_name": "Litigation / Lis Pendens", "rule_code": "LLM_LITIGATION",
             "status": "PASS", "severity": "CRITICAL",
             "explanation": "No references to court orders, suits, or lis pendens found in any document.",
             "evidence": "No litigation markers in EC or Sale Deed", "data_confidence": "MEDIUM"},
        ],
        "group_results_summary": {
            "1": {"name": "EC Hygiene", "check_count": 3, "deduction": 0},
            "2": {"name": "Sale Deed", "check_count": 4, "deduction": 4},
            "3": {"name": "Cross-Doc", "check_count": 2, "deduction": 0},
            "4": {"name": "Chain", "check_count": 1, "deduction": 0},
        },
        "chain_of_title": [
            {"sequence": 1, "from": "State (Original Grant)", "to": "Raman S/o Krishnan",
             "date": "Pre-2010", "transaction_type": "Grant", "document_number": "--",
             "consideration": "--", "valid": True},
            {"sequence": 2, "from": "Raman S/o Krishnan", "to": "Muthu D/o Perumal",
             "date": "15-03-2012", "transaction_type": "Sale", "document_number": "1234/2012",
             "consideration": "15,00,000", "valid": True},
            {"sequence": 3, "from": "Muthu D/o Perumal", "to": "State Bank of India",
             "date": "10-01-2018", "transaction_type": "Mortgage", "document_number": "8901/2018",
             "consideration": "20,00,000", "valid": True},
            {"sequence": 4, "from": "State Bank of India", "to": "Muthu D/o Perumal",
             "date": "05-11-2019", "transaction_type": "Release", "document_number": "8902/2019",
             "consideration": "--", "valid": True},
            {"sequence": 5, "from": "Muthu D/o Perumal", "to": "Lakshmi W/o Senthil",
             "date": "20-06-2020", "transaction_type": "Sale", "document_number": "5678/2020",
             "consideration": "45,00,000", "valid": True},
        ],
        "red_flags": [],
        "recommendations": [
            "Obtain original Sale Deed and verify physical document matches extracted data.",
            "Confirm current Patta holder name matches with Tambaram Taluk office records.",
        ],
        "missing_documents": ["FMB (Field Measurement Book)", "Approved Layout Plan"],
        "active_encumbrances": [],
    },
    "identity_clusters": [
        {"consensus_name": "Lakshmi W/o Senthil", "cluster_id": 1,
         "variants": ["Lakshmi W/o Senthil", "Lakshmi", "Lakshmi Senthil"],
         "mention_count": 8, "confidence": 0.96},
        {"consensus_name": "Muthu D/o Perumal", "cluster_id": 2,
         "variants": ["Muthu D/o Perumal", "Muthu", "Muthu Perumal"],
         "mention_count": 6, "confidence": 0.94},
        {"consensus_name": "Raman S/o Krishnan", "cluster_id": 3,
         "variants": ["Raman S/o Krishnan", "Raman"],
         "mention_count": 3, "confidence": 0.91},
        {"consensus_name": "State Bank of India", "cluster_id": 4,
         "variants": ["State Bank of India", "SBI"],
         "mention_count": 2, "confidence": 0.99},
    ],
    "narrative_report": "",
}

pdf_path = generate_pdf_report(session_data)
print(f"PDF generated: {pdf_path}")
