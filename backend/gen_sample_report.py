"""Generate a sample PDF report with mock session data.

Usage:
    python gen_sample_report.py

Produces:
    temp/reports/mock_demo_report.html
    temp/reports/mock_demo_report.pdf
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.reports.generator import generate_pdf_report

MOCK_SESSION = {
    "session_id": "mock_demo",
    "status": "completed",
    "created_at": "2026-02-23T10:30:00",
    "risk_score": 32,
    "risk_band": "MEDIUM",
    "documents": [
        {
            "doc_id": "doc_1",
            "filename": "Sale_Deed_2019.pdf",
            "doc_type": "SALE_DEED",
            "confidence": 0.95,
            "total_pages": 8,
            "extracted_data": {
                "document_number": "1234/2019",
                "registration_date": "15-03-2019",
                "sub_registrar_office": "Tambaram SRO",
                "seller_name": "Rajesh Kumar S",
                "buyer_name": "Anitha Devi M",
                "property_description": "Plot No. 45, Door No. 12/3, Velachery Main Road, Ward 173, Block 28",
                "survey_number": "318/1A",
                "village": "Velachery",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
                "extent": "2400 sq.ft",
                "consideration_amount": 4500000,
                "market_value": 4200000,
                "stamp_duty_paid": 315000,
                "registration_fee_paid": 45000,
                "boundaries": {
                    "north": "Plot No. 44 belonging to Suresh",
                    "south": "20 feet road",
                    "east": "Plot No. 46 belonging to Meena",
                    "west": "30 feet road"
                },
                "witnesses": ["Karthik R", "Priya S"],
                "schedule_property": "All that piece and parcel of land measuring 2400 sq.ft situated at Survey No. 318/1A, Velachery Village, Tambaram Taluk, Chengalpattu District.",
            },
        },
        {
            "doc_id": "doc_2",
            "filename": "EC_2014_2024.pdf",
            "doc_type": "EC",
            "confidence": 0.92,
            "total_pages": 12,
            "extracted_data": {
                "ec_number": "EC/2024/56789",
                "issuing_office": "Tambaram SRO",
                "property_description": "Survey No. 318/1A, Velachery Village",
                "period_from": "01-01-2014",
                "period_to": "31-12-2024",
                "transactions": [
                    {
                        "document_number": "890/2014",
                        "date": "20-06-2014",
                        "nature": "Sale Deed",
                        "executant": "Murthy K",
                        "claimant": "Rajesh Kumar S",
                        "consideration": 2800000,
                    },
                    {
                        "document_number": "1234/2019",
                        "date": "15-03-2019",
                        "nature": "Sale Deed",
                        "executant": "Rajesh Kumar S",
                        "claimant": "Anitha Devi M",
                        "consideration": 4500000,
                    },
                ],
                "encumbrances_found": False,
            },
        },
        {
            "doc_id": "doc_3",
            "filename": "Patta_318_1A.pdf",
            "doc_type": "PATTA",
            "confidence": 0.88,
            "total_pages": 2,
            "extracted_data": {
                "patta_number": "12345",
                "survey_number": "318/1A",
                "village": "Velachery",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
                "owner_name": "Anitha Devi M",
                "extent_hectare": "0.0223",
                "extent_sqft": "2400",
                "classification": "Dry",
                "tax_paid_upto": "2025-2026",
            },
        },
        {
            "doc_id": "doc_4",
            "filename": "Chitta_318_1A.pdf",
            "doc_type": "CHITTA",
            "confidence": 0.85,
            "total_pages": 1,
            "extracted_data": {
                "survey_number": "318/1A",
                "village": "Velachery",
                "owner_name": "Anitha Devi M",
                "extent": "2400 sq.ft",
                "classification": "Dry",
            },
        },
        {
            "doc_id": "doc_5",
            "filename": "FMB_318.pdf",
            "doc_type": "FMB",
            "confidence": 0.80,
            "total_pages": 1,
            "extracted_data": {
                "survey_number": "318/1A",
                "village": "Velachery",
                "extent": "2400 sq.ft",
                "boundaries": {
                    "north": "Sy. No. 317",
                    "south": "20 ft road",
                    "east": "Sy. No. 318/1B",
                    "west": "30 ft road"
                },
            },
        },
    ],
    "verification_result": {
        "executive_summary": (
            "The property at Survey No. 318/1A, Velachery Village, Tambaram Taluk, "
            "Chengalpattu District has been analysed across 5 documents. The chain of "
            "title shows two clean sale transactions (2014 and 2019) with no encumbrances "
            "found in the 10-year EC period. Patta and Chitta records confirm current "
            "ownership in favour of Anitha Devi M. Minor discrepancies in boundary "
            "descriptions between the Sale Deed and FMB warrant physical verification."
        ),
        "checks": [
            # Group 1 – EC Checks
            {"check_id": "EC_COVERAGE", "group": "EC Integrity", "label": "EC covers ≥13 years", "status": "PASS",
             "detail": "EC covers 2014–2024 (10 years). Adequate coverage for recent transactions.", "confidence": 0.90},
            {"check_id": "EC_NO_ENCUMBRANCE", "group": "EC Integrity", "label": "No active encumbrances", "status": "PASS",
             "detail": "No mortgages, liens, or pending litigation found in EC.", "confidence": 0.95},
            {"check_id": "EC_CONTINUITY", "group": "EC Integrity", "label": "Transaction continuity in EC", "status": "PASS",
             "detail": "All transactions in EC form a continuous chain from Murthy K → Rajesh Kumar S → Anitha Devi M.", "confidence": 0.92},

            # Group 2 – Sale Deed
            {"check_id": "SD_STAMP_DUTY", "group": "Sale Deed Compliance", "label": "Stamp duty ≥ 7% of max(consideration, market value)", "status": "PASS",
             "detail": "Stamp duty ₹3,15,000 = 7.0% of consideration ₹45,00,000. Meets TN requirement.", "confidence": 0.94},
            {"check_id": "SD_REG_FEE", "group": "Sale Deed Compliance", "label": "Registration fee paid", "status": "PASS",
             "detail": "Registration fee ₹45,000 = 1% of consideration. Compliant.", "confidence": 0.93},
            {"check_id": "SD_WITNESSES", "group": "Sale Deed Compliance", "label": "Two witnesses present", "status": "PASS",
             "detail": "Two witnesses recorded: Karthik R and Priya S.", "confidence": 0.97},

            # Group 3 – Cross-document
            {"check_id": "CROSS_SURVEY_MATCH", "group": "Cross-Document Consistency", "label": "Survey number consistent across docs", "status": "PASS",
             "detail": "Survey No. 318/1A matches across Sale Deed, Patta, Chitta, and FMB.", "confidence": 0.96},
            {"check_id": "CROSS_EXTENT_MATCH", "group": "Cross-Document Consistency", "label": "Extent consistent across docs", "status": "PASS",
             "detail": "Extent 2400 sq.ft consistent across all documents.", "confidence": 0.95},
            {"check_id": "CROSS_OWNER_MATCH", "group": "Cross-Document Consistency", "label": "Current owner matches across docs", "status": "PASS",
             "detail": "Anitha Devi M listed as owner in Sale Deed (buyer), Patta, and Chitta.", "confidence": 0.94},
            {"check_id": "CROSS_BOUNDARY_MATCH", "group": "Cross-Document Consistency", "label": "Boundary descriptions consistent", "status": "WARNING",
             "detail": "Sale Deed mentions 'Plot No. 44 belonging to Suresh' on north; FMB says 'Sy. No. 317'. These likely refer to the same parcel but physical verification recommended.", "confidence": 0.65},

            # Group 4 – Chain of Title
            {"check_id": "CHAIN_COMPLETE", "group": "Chain of Title", "label": "Unbroken chain of ownership", "status": "PASS",
             "detail": "Chain: Murthy K (pre-2014) → Rajesh Kumar S (2014) → Anitha Devi M (2019). No gaps.", "confidence": 0.91},
            {"check_id": "CHAIN_EC_SD_MATCH", "group": "Chain of Title", "label": "EC transactions match Sale Deed", "status": "PASS",
             "detail": "Sale Deed doc no. 1234/2019 found in EC with matching parties and consideration.", "confidence": 0.96},

            # Group 5 – Meta / Completeness
            {"check_id": "META_ALL_DOCS", "group": "Completeness", "label": "All essential documents present", "status": "WARNING",
             "detail": "Missing: Legal Opinion, Tax Receipts, Building Approval (if applicable). Recommended for comprehensive due diligence.", "confidence": 0.70},
            {"check_id": "META_PATTA_TAX", "group": "Completeness", "label": "Property tax current", "status": "PASS",
             "detail": "Patta shows tax paid up to 2025-2026. Current.", "confidence": 0.88},
        ],
        "chain_of_title": [
            {"year": "Before 2014", "owner": "Murthy K", "doc_ref": "Prior ownership (referenced in EC)"},
            {"year": "2014", "owner": "Rajesh Kumar S", "doc_ref": "Sale Deed 890/2014 dated 20-06-2014"},
            {"year": "2019", "owner": "Anitha Devi M", "doc_ref": "Sale Deed 1234/2019 dated 15-03-2019"},
        ],
        "red_flags": [
            "Boundary description mismatch between Sale Deed and FMB — recommend physical site inspection.",
            "EC coverage is 10 years (2014–2024); ideally should be 13+ years for full clearance.",
        ],
        "recommendations": [
            "Conduct physical site inspection to verify boundaries match FMB sketch.",
            "Obtain a fresh EC extending to February 2026 to cover the gap from Dec 2024.",
            "Request legal opinion from an advocate for completeness.",
            "Verify latest property tax receipts (beyond Patta record).",
            "Check with local body (municipality/panchayat) for any planning restrictions.",
        ],
        "missing_documents": [
            "Legal Opinion from Advocate",
            "Latest Property Tax Receipts (2024-2025 onwards)",
            "Building Approval Plan (if construction planned)",
            "NOC from Layout Authority (if part of approved layout)",
        ],
        "active_encumbrances": [],
    },
    "identity_clusters": [
        {
            "canonical": "Anitha Devi M",
            "variants": ["Anitha Devi M", "M. Anitha Devi", "Anitha Devi"],
            "doc_refs": ["Sale_Deed_2019.pdf", "Patta_318_1A.pdf", "Chitta_318_1A.pdf"],
        },
        {
            "canonical": "Rajesh Kumar S",
            "variants": ["Rajesh Kumar S", "S. Rajesh Kumar", "Rajesh Kumar"],
            "doc_refs": ["Sale_Deed_2019.pdf", "EC_2014_2024.pdf"],
        },
        {
            "canonical": "Murthy K",
            "variants": ["Murthy K", "K. Murthy"],
            "doc_refs": ["EC_2014_2024.pdf"],
        },
    ],
    "narrative_report": """## Property Due Diligence Summary

### Property Identification
The subject property is a **residential plot** measuring **2,400 sq.ft** at **Survey No. 318/1A**, Velachery Village, Tambaram Taluk, Chengalpattu District, Tamil Nadu.

### Ownership History
The property has changed hands twice in the last decade:

| Year | Transaction | Seller → Buyer | Consideration |
|------|------------|----------------|---------------|
| 2014 | Sale Deed 890/2014 | Murthy K → Rajesh Kumar S | ₹28,00,000 |
| 2019 | Sale Deed 1234/2019 | Rajesh Kumar S → Anitha Devi M | ₹45,00,000 |

The current owner is **Anitha Devi M** as confirmed by the Sale Deed, Patta, and Chitta records.

### Encumbrance Status
The Encumbrance Certificate covering **2014–2024** reveals **no active encumbrances** — no mortgages, liens, court attachments, or pending litigation are recorded against this property.

### Document Consistency
Cross-verification across all 5 documents shows **strong consistency** in survey number (318/1A), extent (2,400 sq.ft), and ownership. A **minor discrepancy** exists in boundary descriptions between the Sale Deed (which uses plot numbers) and the FMB (which uses survey numbers). This is common and not a concern, but physical verification is recommended.

### Compliance
- **Stamp duty**: ₹3,15,000 paid (7% of ₹45,00,000) — **compliant** with Tamil Nadu rates
- **Registration fee**: ₹45,000 (1%) — **compliant**
- **Witnesses**: Two witnesses recorded — **compliant**
- **Property tax**: Paid up to 2025-2026 — **current**

### Risk Assessment
Overall risk score: **32/100 (MEDIUM)**. The property has a clean title with no encumbrances. The two areas requiring attention are:
1. Boundary description discrepancy (low severity, resolvable via site visit)
2. EC coverage gap from Dec 2024 to present (procedural, obtain fresh EC)

### Recommendations
1. Physical site inspection to verify boundaries
2. Obtain fresh EC extending to current date
3. Engage advocate for formal legal opinion
4. Verify latest tax payment receipts
5. Check municipal/panchayat planning restrictions
""",
    "chat_history": [],
}


if __name__ == "__main__":
    print("Generating report with mock data...")
    print(f"  Session ID: {MOCK_SESSION['session_id']}")
    print(f"  Risk: {MOCK_SESSION['risk_band']} ({MOCK_SESSION['risk_score']}/100)")
    print(f"  Documents: {len(MOCK_SESSION['documents'])}")
    print(f"  Checks: {len(MOCK_SESSION['verification_result']['checks'])}")
    print()

    pdf_path = generate_pdf_report(MOCK_SESSION)

    print(f"PDF generated: {pdf_path}")
    print(f"PDF size: {pdf_path.stat().st_size:,} bytes")

    html_path = pdf_path.with_suffix(".html")
    if html_path.exists():
        print(f"HTML generated: {html_path}")
        print(f"HTML size: {html_path.stat().st_size:,} bytes")
