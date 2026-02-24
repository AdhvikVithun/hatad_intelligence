"""JSON Schema definitions for Ollama structured outputs.

Each schema enforces exact field names, types, and enums at the API level,
eliminating parse errors and schema drift in LLM responses.

Used via Ollama's `format: { JSON Schema }` parameter.
"""

from app.config import DOCUMENT_TYPES

# ═══════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": DOCUMENT_TYPES,
        },
        "confidence": {
            "type": "number",
        },
        "language": {
            "type": "string",
        },
        "key_identifiers": {
            "type": "array",
            "items": {"type": "string", "maxLength": 100},
            "maxItems": 8,
        },
    },
    "required": ["document_type", "confidence", "language", "key_identifiers"],
}

# ═══════════════════════════════════════════════════
# TRANSACTION TYPES (shared enum)
# ═══════════════════════════════════════════════════

TRANSACTION_TYPES = [
    "SALE", "MORTGAGE", "RELEASE", "PARTITION", "GIFT", "WILL",
    "COURT_ORDER", "LEASE", "AGREEMENT", "POWER_OF_ATTORNEY",
    "RECTIFICATION", "CANCELLATION", "SETTLEMENT",
    # Extended types for TN land records
    "RECEIPT", "EXCHANGE", "RELINQUISHMENT", "RECONVEYANCE",
    "DECLARATION", "TRUST", "SURETY", "ADOPTION", "AMALGAMATION",
    "OTHER",
]

# ═══════════════════════════════════════════════════
# EC EXTRACTION
# ═══════════════════════════════════════════════════

_EC_TRANSACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "row_number": {"type": "integer"},
        "date": {"type": "string", "description": "Primary date — use registration date"},
        "document_number": {"type": "string"},
        "document_year": {"type": "string"},
        "sro": {"type": "string"},
        "transaction_type": {
            "type": "string",
            "enum": TRANSACTION_TYPES,
        },
        "seller_or_executant": {"type": "string"},
        "buyer_or_claimant": {"type": "string"},
        "extent": {"type": "string"},
        "survey_number": {"type": "string"},
        "consideration_amount": {"type": "string"},
        "remarks": {"type": "string", "description": "Document Remarks / ஆவணக் குறிப்புகள்"},
        "suspicious_flags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        # ── New fields for complete EC capture ──
        "execution_date": {"type": "string", "description": "Date of Execution / எழுதிக் கொ டுத்த நாள்"},
        "presentation_date": {"type": "string", "description": "Date of Presentation / தாக்கல் நாள்"},
        "registration_date": {"type": "string", "description": "Date of Registration / பதிவு நாள்"},
        "market_value": {"type": "string", "description": "Market/Guideline Value / சந்தை மதிப்பு"},
        "pr_number": {"type": "string", "description": "Previous Registration / PR Number / முன்றாவ ஆவண எண்"},
        "plot_number": {"type": "string", "description": "Plot No / மனை எண் from schedule"},
        "door_number": {"type": "string", "description": "Door No / கதவு எண் (new or old) from schedule"},
        "schedule_remarks": {"type": "string", "description": "Schedule Remarks / சொத்து விவரம் தொடர்பான குறிப்பு — property details, cross-references"},
        "additional_details": {"type": "string", "description": "Any other information in this EC entry not captured above"},
        # ── Stable identity (generated post-extraction, not by LLM) ──
        "transaction_id": {"type": "string", "description": "Stable unique ID, e.g. EC-5909/2012-Vadavalli. Generated post-extraction."},
    },
    "required": [
        "row_number", "date", "document_number",
        "transaction_type", "seller_or_executant", "buyer_or_claimant",
    ],
}

EXTRACT_EC_SCHEMA = {
    "type": "object",
    "properties": {
        "ec_number": {"type": "string"},
        "property_description": {"type": "string"},
        "village": {"type": "string", "description": "Village name from EC header (e.g. Sholinganallur)"},
        "taluk": {"type": "string", "description": "Taluk name from EC header (e.g. Tambaram)"},
        "period_from": {"type": "string"},
        "period_to": {"type": "string"},
        "total_entries_found": {"type": "integer"},
        "transactions": {
            "type": "array",
            "items": _EC_TRANSACTION_SCHEMA,
            "maxItems": 200,
        },
        "extraction_notes": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Document-level red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": [
        "ec_number", "property_description", "period_from", "period_to",
        "total_entries_found", "transactions", "extraction_notes",
    ],
}

# ═══════════════════════════════════════════════════
# PATTA EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_PATTA_SCHEMA = {
    "type": "object",
    "properties": {
        "patta_number": {"type": "string"},
        "old_patta_number": {"type": "string"},
        "owner_names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "father_name": {"type": "string"},
                    "share": {"type": "string"},
                },
                "required": ["name", "father_name", "share"],
            },
            "maxItems": 20,
        },
        "survey_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "survey_no": {"type": "string"},
                    "extent": {"type": "string"},
                    "classification": {"type": "string"},
                },
                "required": ["survey_no", "extent", "classification"],
            },
            "maxItems": 50,
        },
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "total_extent": {"type": "string"},
        "land_classification": {"type": "string"},
        "thandaper_number": {"type": "string"},
        "tax_details": {"type": "string"},
        "irrigation_source": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": [
        "patta_number", "owner_names", "survey_numbers",
        "village", "taluk", "district", "total_extent", "land_classification",
        "remarks",
    ],
}

# ═══════════════════════════════════════════════════
# SALE DEED EXTRACTION
# ═══════════════════════════════════════════════════

_PARTY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "father_name": {"type": "string"},
        "age": {"type": "string"},
        "address": {"type": "string"},
        "pan": {"type": "string"},
        "aadhaar": {"type": "string", "description": "Aadhaar number (12 digits) if mentioned"},
        "share_percentage": {"type": "string", "description": "Ownership share e.g. '50%', '1/3rd'"},
        "identification_proof": {"type": "string", "description": "ID proof type and number other than PAN/Aadhaar"},
        "relationship": {"type": "string", "description": "Relationship to other parties e.g. 'wife of Seller 1', 'son of X'"},
    },
    "required": ["name"],
}

_WITNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "father_name": {"type": "string"},
        "age": {"type": "string"},
        "address": {"type": "string"},
    },
    "required": ["name"],
}

_OWNERSHIP_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "owner": {"type": "string"},
        "acquired_from": {"type": "string"},
        "acquisition_mode": {
            "type": "string",
            "enum": ["Sale", "Gift", "Inheritance", "Partition", "Settlement", "Unknown"],
        },
        "document_number": {"type": "string"},
        "document_date": {"type": "string"},
        "remarks": {"type": "string"},
    },
    "required": ["owner", "acquisition_mode"],
}

EXTRACT_SALE_DEED_SCHEMA = {
    "type": "object",
    "properties": {
        "document_number": {"type": "string"},
        "registration_date": {"type": "string"},
        "execution_date": {"type": "string"},
        "sro": {"type": "string"},
        "seller": {"type": "array", "items": _PARTY_SCHEMA, "maxItems": 10},
        "buyer": {"type": "array", "items": _PARTY_SCHEMA, "maxItems": 10},
        "property": {
            "type": "object",
            "properties": {
                "survey_number": {"type": "string"},
                "village": {"type": "string"},
                "taluk": {"type": "string"},
                "district": {"type": "string"},
                "extent": {"type": "string"},
                "boundaries": {
                    "type": "object",
                    "properties": {
                        "north": {"type": "string"},
                        "south": {"type": "string"},
                        "east": {"type": "string"},
                        "west": {"type": "string"},
                    },
                    "required": ["north", "south", "east", "west"],
                },
                "property_type": {
                    "type": "string",
                    "enum": ["Residential", "Agricultural", "Commercial", "Industrial", "Mixed", "Unknown"],
                },
                "land_classification": {
                    "type": "string",
                    "description": "Land type: Wet (நன்சை), Dry (புன்சை), Maidan, Garden, or Unknown",
                },
                "plot_number": {"type": "string", "description": "Plot/sub-division number if applicable"},
                "door_number": {"type": "string", "description": "Property door number (கதவு எண்) if built-up property"},
                "measurement_reference": {"type": "string", "description": "FMB/sketch measurement reference if mentioned"},
                "assessment_number": {"type": "string", "description": "Municipal assessment/tax number if mentioned"},
            },
            "required": ["survey_number", "village", "taluk", "district", "extent", "boundaries", "property_type"],
        },
        "property_description": {"type": "string"},
        "financials": {
            "type": "object",
            "properties": {
                "consideration_amount": {"type": "number"},
                "guideline_value": {"type": "number"},
                "stamp_duty": {"type": "number"},
                "registration_fee": {"type": "number"},
            },
            "required": ["consideration_amount", "guideline_value", "stamp_duty", "registration_fee"],
        },
        "stamp_paper": {
            "type": "object",
            "description": "Stamp paper details used for this deed",
            "properties": {
                "serial_numbers": {"type": "array", "items": {"type": "string"}, "maxItems": 20,
                                   "description": "Stamp paper serial numbers (e.g. 'A 123456')"},
                "vendor_name": {"type": "string", "description": "Stamp paper vendor name"},
                "denomination_per_sheet": {"type": "integer", "description": "Denomination of each sheet in rupees"},
                "sheet_count": {"type": "integer", "description": "Number of physical stamp sheets"},
                "total_stamp_value": {"type": "integer", "description": "Total stamp paper value (denomination × sheets)"},
            },
        },
        "registration_details": {
            "type": "object",
            "description": "Book/volume/page registration office metadata",
            "properties": {
                "book_number": {"type": "string"},
                "volume_number": {"type": "string"},
                "page_number": {"type": "string"},
                "certified_copy_number": {"type": "string"},
                "certified_copy_date": {"type": "string"},
            },
        },
        "payment_mode": {"type": "string"},
        "previous_ownership": {
            "type": "object",
            "properties": {
                "document_number": {"type": "string"},
                "document_date": {"type": "string"},
                "previous_owner": {"type": "string"},
                "acquisition_mode": {
                    "type": "string",
                    "enum": ["Sale", "Gift", "Inheritance", "Partition", "Settlement", "Unknown"],
                },
            },
            "required": ["document_number", "document_date", "previous_owner", "acquisition_mode"],
        },
        "ownership_history": {
            "type": "array",
            "items": _OWNERSHIP_LINK_SCHEMA,
            "maxItems": 10,
        },
        "encumbrance_declaration": {"type": "string"},
        "possession_date": {"type": "string"},
        "witnesses": {"type": "array", "items": _WITNESS_SCHEMA, "maxItems": 10},
        "special_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "power_of_attorney": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": [
        "document_number", "registration_date", "execution_date", "sro",
        "seller", "buyer", "property", "property_description", "financials",
        "payment_mode", "previous_ownership", "ownership_history",
        "encumbrance_declaration", "possession_date",
        "witnesses", "special_conditions", "power_of_attorney", "remarks",
    ],
}

# ═══════════════════════════════════════════════════
# SALE DEED SUMMARIZATION
# ═══════════════════════════════════════════════════

SUMMARIZE_SALE_DEED_SCHEMA = {
    "type": "object",
    "properties": {
        "document_number": {"type": "string"},
        "registration_date": {"type": "string"},
        "sro": {"type": "string"},
        "transaction_summary": {
            "type": "string",
            "description": "One-paragraph human-readable summary of the entire transaction",
        },
        "seller_names": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "buyer_names": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "property_summary": {
            "type": "string",
            "description": "Compact property description: survey, village, extent, classification",
        },
        "ownership_chain": {
            "type": "array", "items": {"type": "string"}, "maxItems": 20,
            "description": "Chronological ownership chain: 'A → B (Sale, Doc 123/1990)'",
        },
        "financial_summary": {
            "type": "object",
            "properties": {
                "consideration_amount": {"type": "number"},
                "guideline_value": {"type": "number"},
                "stamp_duty": {"type": "number"},
                "registration_fee": {"type": "number"},
                "stamp_duty_adequate": {"type": "boolean"},
            },
        },
        "risk_flags": {
            "type": "array", "items": {"type": "string"}, "maxItems": 20,
            "description": "Identified risks: undervaluation, GPA sale, missing encumbrance cert, etc.",
        },
        "completeness_score": {
            "type": "string",
            "description": "Assessment: COMPLETE / MOSTLY_COMPLETE / PARTIAL / INCOMPLETE",
        },
        "summary_notes": {"type": "string"},
    },
    "required": [
        "document_number", "registration_date", "sro",
        "transaction_summary", "seller_names", "buyer_names",
        "property_summary", "ownership_chain", "financial_summary",
        "risk_flags", "completeness_score", "summary_notes",
    ],
}

# ═══════════════════════════════════════════════════
# EC SUMMARIZATION
# ═══════════════════════════════════════════════════

SUMMARIZE_EC_SCHEMA = {
    "type": "object",
    "properties": {
        "ec_number": {"type": "string"},
        "period": {"type": "string"},
        "total_transactions": {"type": "integer"},
        "transaction_table": {"type": "string"},
        "ownership_chain": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
        "active_encumbrances": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
        "suspicious_patterns": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "key_statistics": {
            "type": "object",
            "properties": {
                "sales": {"type": "integer"},
                "mortgages": {"type": "integer"},
                "releases": {"type": "integer"},
                "partitions": {"type": "integer"},
                "others": {"type": "integer"},
                "date_range": {"type": "string"},
                "unique_parties": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
                "survey_numbers": {"type": "array", "items": {"type": "string"}, "maxItems": 50},
            },
            "required": ["sales", "mortgages", "releases", "partitions", "others",
                         "date_range", "unique_parties", "survey_numbers"],
        },
        "summary_notes": {"type": "string"},
    },
    "required": [
        "ec_number", "period", "total_transactions", "transaction_table",
        "ownership_chain", "active_encumbrances", "suspicious_patterns",
        "key_statistics", "summary_notes",
    ],
}

# ═══════════════════════════════════════════════════
# VERIFICATION GROUPS — Shared check schema
# ═══════════════════════════════════════════════════

CHECK_STATUS_ENUM = ["PASS", "FAIL", "WARNING", "NOT_APPLICABLE", "INFO"]
SEVERITY_ENUM = ["CRITICAL", "HIGH", "MEDIUM", "INFO"]

_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "rule_code": {"type": "string"},
        "rule_name": {"type": "string"},
        "severity": {"type": "string", "enum": SEVERITY_ENUM},
        "status": {"type": "string", "enum": CHECK_STATUS_ENUM},
        "explanation": {"type": "string"},
        "recommendation": {"type": "string"},
        "evidence": {"type": "string", "minLength": 10},
    },
    "required": ["rule_code", "rule_name", "severity", "status", "explanation", "recommendation", "evidence"],
}

_CHAIN_LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "sequence": {"type": "integer"},
        "date": {"type": "string"},
        "from": {"type": "string"},
        "to": {"type": "string"},
        "transaction_type": {"type": "string"},
        "document_number": {"type": "string"},
        "valid": {"type": "boolean"},
        "notes": {"type": "string"},
        "source": {"type": "string", "description": "Origin document type: EC, Sale Deed, A-Register, Gift Deed, etc."},
    },
    "required": ["sequence", "from", "to", "transaction_type"],
}

# Group 1: EC-Only
VERIFY_GROUP1_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA, "maxItems": 25},
        "chain_of_title": {"type": "array", "items": _CHAIN_LINK_SCHEMA, "maxItems": 100},
        "active_encumbrances": {"type": "array", "items": {"type": "string"}, "maxItems": 30},
    },
    "required": ["group", "group_score_deduction", "checks", "chain_of_title", "active_encumbrances"],
}

# Group 2: Sale Deed
VERIFY_GROUP2_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA, "maxItems": 25},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 3: Cross-Document
VERIFY_GROUP3_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA, "maxItems": 25},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 4: Chain & Patterns
VERIFY_GROUP4_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA, "maxItems": 25},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 5: Meta — includes risk score, red flags, recommendations
VERIFY_GROUP5_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "risk_score": {"type": "integer"},
        "risk_band": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
        "executive_summary": {"type": "string"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA, "maxItems": 25},
        "missing_documents": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
        "red_flags": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "recommendations": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
    },
    "required": [
        "group", "risk_score", "risk_band", "executive_summary",
        "checks", "missing_documents", "red_flags", "recommendations",
    ],
}

# Map group ID → schema (for orchestrator lookup)
VERIFY_GROUP_SCHEMAS = {
    1: VERIFY_GROUP1_SCHEMA,
    2: VERIFY_GROUP2_SCHEMA,
    3: VERIFY_GROUP3_SCHEMA,
    4: VERIFY_GROUP3_SCHEMA,  # Cross-doc compliance — same structure as cross-doc property
    5: VERIFY_GROUP4_SCHEMA,  # Chain & Pattern (was id=4)
    6: VERIFY_GROUP5_SCHEMA,  # Meta Assessment (was id=5)
}

# Generic extractor schema (loose — accepts any keys) — used only for OTHER
EXTRACT_GENERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "document_summary": {"type": "string"},
        "key_parties": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "property_details": {"type": "object"},
        "key_dates": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "amounts": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "notable_clauses": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
    },
    "required": ["document_summary", "key_parties", "property_details",
                  "key_dates", "amounts", "notable_clauses"],
}

# ═══════════════════════════════════════════════════
# FMB (Field Measurement Book) EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_FMB_SCHEMA = {
    "type": "object",
    "properties": {
        "survey_number": {"type": "string", "description": "S.F.No. / T.S.No. / R.S.No."},
        "subdivision": {"type": "string", "description": "Subdivision details if any"},
        "area_acres": {"type": "string", "description": "Area in acres-cents (e.g. 2 acres 50 cents)"},
        "area_hectares": {"type": "string", "description": "Area in hectares-ares (e.g. 0.9250 hectares)"},
        "dimensions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "description": "Side label (A-B, North, etc.)"},
                    "length_ft": {"type": "string", "description": "Length in feet or links"},
                    "bearing": {"type": "string", "description": "Compass bearing (e.g. N30°E)"},
                },
                "required": ["side", "length_ft"],
            },
            "maxItems": 20,
        },
        "adjacent_surveys": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "description": "N/S/E/W or side label"},
                    "survey_no": {"type": "string", "description": "Adjacent survey number"},
                    "owner_name": {"type": "string", "description": "Owner of adjacent survey"},
                },
                "required": ["direction", "survey_no"],
            },
            "maxItems": 20,
        },
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "boundaries": {
            "type": "object",
            "properties": {
                "north": {"type": "string"},
                "south": {"type": "string"},
                "east": {"type": "string"},
                "west": {"type": "string"},
            },
            "required": ["north", "south", "east", "west"],
        },
        "land_classification": {"type": "string", "description": "Wet/Dry/Garden/House site"},
        "sketch_notes": {"type": "string", "description": "Any notes from the FMB sketch"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["survey_number", "village", "taluk", "boundaries", "remarks"],
}

# ═══════════════════════════════════════════════════
# ADANGAL (Village Account) EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_ADANGAL_SCHEMA = {
    "type": "object",
    "properties": {
        "survey_number": {"type": "string"},
        "subdivision": {"type": "string"},
        "owner_name": {"type": "string"},
        "father_name": {"type": "string"},
        "extent": {"type": "string", "description": "Total extent with unit"},
        "wet_extent": {"type": "string", "description": "Wet (நன்செய்) extent"},
        "dry_extent": {"type": "string", "description": "Dry (புன்செய்) extent"},
        "soil_type": {
            "type": "string",
            "description": "நன்செய் (wet) / புன்செய் (dry) / தோட்டம் (garden) / மனை (house site)",
        },
        "irrigation_source": {"type": "string", "description": "Well/Canal/Tank/Rain-fed"},
        "crop_details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "season": {"type": "string", "description": "Kuruvai/Samba/Navarai/Annual"},
                    "crop": {"type": "string"},
                    "extent_cultivated": {"type": "string"},
                },
                "required": ["season", "crop"],
            },
            "maxItems": 10,
        },
        "assessment_amount": {"type": "string", "description": "Revenue assessment in rupees"},
        "tenant_name": {"type": "string", "description": "Tenant/cultivator if different from owner"},
        "cultivation_status": {"type": "string", "description": "Cultivated/Fallow/Waste"},
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["survey_number", "owner_name", "extent", "soil_type",
                  "village", "taluk", "remarks"],
}

# ═══════════════════════════════════════════════════
# LAYOUT APPROVAL EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "approval_number": {"type": "string", "description": "LP No. / PP No. / Approval No."},
        "authority": {
            "type": "string",
            "description": "Approving authority — CMDA / DTCP / Local Planning Authority",
        },
        "layout_name": {"type": "string"},
        "applicant_name": {"type": "string"},
        "total_plots": {"type": "integer"},
        "total_area": {"type": "string", "description": "Total layout area with unit"},
        "plot_schedule": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "plot_no": {"type": "string"},
                    "area": {"type": "string"},
                    "dimensions": {"type": "string"},
                },
                "required": ["plot_no", "area"],
            },
            "maxItems": 100,
        },
        "road_specifications": {"type": "string"},
        "common_area": {"type": "string"},
        "approval_date": {"type": "string"},
        "validity_period": {"type": "string", "description": "Validity in years or expiry date"},
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
        },
        "survey_number": {"type": "string"},
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["approved", "expired", "revoked", "unknown"],
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["approval_number", "authority", "approval_date", "village", "remarks"],
}

# ═══════════════════════════════════════════════════
# LEGAL HEIR CERTIFICATE EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_LEGAL_HEIR_SCHEMA = {
    "type": "object",
    "properties": {
        "certificate_number": {"type": "string"},
        "issuing_authority": {"type": "string", "description": "RDO / Court / Tahsildar"},
        "certificate_date": {"type": "string"},
        "court_or_revenue": {
            "type": "string",
            "enum": ["court", "revenue", "notarized", "unknown"],
            "description": "Whether issued by court (succession certificate) or revenue authority",
        },
        "deceased_name": {"type": "string"},
        "deceased_father_name": {"type": "string"},
        "date_of_death": {"type": "string"},
        "heirs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relationship": {"type": "string", "description": "மகன்/மகள்/மனைவி/தாய்/தந்தை"},
                    "share_percentage": {"type": "number"},
                    "age": {"type": "string"},
                    "address": {"type": "string"},
                },
                "required": ["name", "relationship"],
            },
            "maxItems": 20,
        },
        "property_description": {"type": "string"},
        "survey_number": {"type": "string"},
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["deceased_name", "date_of_death", "heirs", "remarks"],
}

# ═══════════════════════════════════════════════════
# POWER OF ATTORNEY EXTRACTION
# ═══════════════════════════════════════════════════

_POA_PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "father_name": {"type": "string"},
        "address": {"type": "string"},
    },
    "required": ["name"],
}

EXTRACT_POA_SCHEMA = {
    "type": "object",
    "properties": {
        "registration_number": {"type": "string"},
        "sro": {"type": "string"},
        "registration_date": {"type": "string"},
        "principal": _POA_PERSON_SCHEMA,
        "agent": _POA_PERSON_SCHEMA,
        "powers_granted": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 20,
            "description": "List of specific powers granted (sell, mortgage, lease, etc.)",
        },
        "property_description": {"type": "string"},
        "survey_number": {"type": "string"},
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "is_general_or_specific": {
            "type": "string",
            "enum": ["general", "specific", "unknown"],
        },
        "validity_period": {"type": "string"},
        "revocation_status": {
            "type": "string",
            "enum": ["active", "revoked", "expired", "unknown"],
        },
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["principal", "agent", "powers_granted", "is_general_or_specific",
                  "revocation_status", "remarks"],
}

# ═══════════════════════════════════════════════════
# COURT ORDER EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_COURT_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        "case_number": {"type": "string", "description": "O.S. No. / C.S. No. / W.P. No. / A.S. No."},
        "court_name": {"type": "string"},
        "court_type": {
            "type": "string",
            "enum": ["district_munsif", "sub_court", "district_court",
                     "high_court", "revenue_court", "tribunal", "other"],
        },
        "order_date": {"type": "string"},
        "order_type": {
            "type": "string",
            "enum": ["injunction", "attachment", "decree", "stay",
                     "vacating", "dismissal", "other"],
        },
        "petitioner": {"type": "string"},
        "respondent": {"type": "string"},
        "property_affected": {
            "type": "object",
            "properties": {
                "survey_number": {"type": "string"},
                "village": {"type": "string"},
                "description": {"type": "string"},
            },
        },
        "order_terms": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 15,
        },
        "status": {
            "type": "string",
            "enum": ["interim", "final", "vacated", "appealed", "unknown"],
        },
        "next_hearing_date": {"type": "string"},
        "related_documents": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["case_number", "court_name", "order_type", "petitioner",
                  "respondent", "status", "remarks"],
}

# ═══════════════════════════════════════════════════
# WILL / TESTAMENT EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_WILL_SCHEMA = {
    "type": "object",
    "properties": {
        "testator": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "father_name": {"type": "string"},
                "age": {"type": "string"},
                "address": {"type": "string"},
            },
            "required": ["name"],
        },
        "execution_date": {"type": "string"},
        "registration_number": {"type": "string"},
        "sro": {"type": "string"},
        "beneficiaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relationship": {"type": "string"},
                    "share": {"type": "string"},
                    "property_bequeathed": {"type": "string"},
                },
                "required": ["name", "relationship"],
            },
            "maxItems": 20,
        },
        "executor": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"},
            },
        },
        "probate_status": {
            "type": "string",
            "enum": ["granted", "pending", "not_applied", "unknown"],
        },
        "probate_court": {"type": "string"},
        "codicils": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "witnesses": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "property_description": {"type": "string"},
        "survey_number": {"type": "string"},
        "village": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["testator", "beneficiaries", "probate_status", "remarks"],
}

# ═══════════════════════════════════════════════════
# PARTITION DEED EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_PARTITION_SCHEMA = {
    "type": "object",
    "properties": {
        "registration_number": {"type": "string"},
        "sro": {"type": "string"},
        "registration_date": {"type": "string"},
        "original_property": {
            "type": "object",
            "properties": {
                "survey_number": {"type": "string"},
                "village": {"type": "string"},
                "total_extent": {"type": "string"},
            },
            "required": ["survey_number"],
        },
        "joint_owners": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "original_share": {"type": "string"},
                },
                "required": ["name"],
            },
            "maxItems": 20,
        },
        "partitioned_shares": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "allocated_survey": {"type": "string"},
                    "allocated_extent": {"type": "string"},
                    "boundaries": {"type": "string"},
                },
                "required": ["name", "allocated_extent"],
            },
            "maxItems": 20,
        },
        "consent_all_parties": {"type": "boolean"},
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "witnesses": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["joint_owners", "partitioned_shares", "remarks"],
}

# ═══════════════════════════════════════════════════
# GIFT DEED EXTRACTION
# ═══════════════════════════════════════════════════

_GIFT_PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "father_name": {"type": "string"},
        "address": {"type": "string"},
        "relationship_to_other": {"type": "string", "description": "Relationship donor↔donee"},
    },
    "required": ["name"],
}

EXTRACT_GIFT_DEED_SCHEMA = {
    "type": "object",
    "properties": {
        "registration_number": {"type": "string"},
        "sro": {"type": "string"},
        "registration_date": {"type": "string"},
        "donor": _GIFT_PERSON_SCHEMA,
        "donee": _GIFT_PERSON_SCHEMA,
        "property": {
            "type": "object",
            "properties": {
                "survey_number": {"type": "string"},
                "village": {"type": "string"},
                "taluk": {"type": "string"},
                "extent": {"type": "string"},
                "boundaries": {
                    "type": "object",
                    "properties": {
                        "north": {"type": "string"},
                        "south": {"type": "string"},
                        "east": {"type": "string"},
                        "west": {"type": "string"},
                    },
                },
            },
        },
        "is_revocable": {"type": "boolean"},
        "acceptance_clause": {"type": "boolean", "description": "Whether donee has accepted"},
        "consideration_amount": {"type": "number", "description": "Should be 0 for valid gift"},
        "conditions": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "witnesses": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["donor", "donee", "is_revocable", "acceptance_clause", "remarks"],
}

# ═══════════════════════════════════════════════════
# RELEASE DEED EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_RELEASE_DEED_SCHEMA = {
    "type": "object",
    "properties": {
        "registration_number": {"type": "string"},
        "sro": {"type": "string"},
        "registration_date": {"type": "string"},
        "releasing_party": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string", "description": "Bank / co-owner / lender / etc."},
            },
            "required": ["name"],
        },
        "beneficiary": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["name"],
        },
        "original_document": {
            "type": "object",
            "properties": {
                "document_type": {"type": "string", "description": "Mortgage / Lien / Co-owner share"},
                "document_number": {"type": "string"},
                "document_date": {"type": "string"},
                "sro": {"type": "string"},
            },
            "required": ["document_type", "document_number"],
        },
        "claim_released": {
            "type": "string",
            "enum": ["mortgage", "lien", "co_owner_share", "encumbrance", "other"],
        },
        "property": {
            "type": "object",
            "properties": {
                "survey_number": {"type": "string"},
                "village": {"type": "string"},
                "extent": {"type": "string"},
            },
        },
        "release_terms": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "consideration_amount": {"type": "number"},
        "original_loan_amount": {"type": "number", "description": "Original mortgage/loan amount if mortgage release"},
        "outstanding_at_release": {"type": "number", "description": "Amount outstanding at time of release"},
        "repayment_confirmed": {"type": "boolean", "description": "Whether full repayment is confirmed in the deed"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["releasing_party", "beneficiary", "original_document",
                  "claim_released", "remarks"],
}

# ═══════════════════════════════════════════════════
# A-REGISTER EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_A_REGISTER_SCHEMA = {
    "type": "object",
    "properties": {
        "register_serial_number": {"type": "string", "description": "Register entry serial number"},
        "paguthi_number": {"type": "string", "description": "Paguthi / ward number"},
        "owner_name": {"type": "string"},
        "father_name": {"type": "string"},
        "patta_number": {"type": "string"},
        "survey_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "survey_no": {"type": "string"},
                    "extent": {"type": "string"},
                    "classification": {"type": "string"},
                },
                "required": ["survey_no", "extent"],
            },
            "maxItems": 50,
        },
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "total_extent": {"type": "string"},
        "tax_details": {"type": "string"},
        "mutation_entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "from_owner": {"type": "string"},
                    "to_owner": {"type": "string"},
                    "reason": {"type": "string", "description": "Sale/Inheritance/Gift/Partition"},
                    "order_number": {"type": "string"},
                },
                "required": ["to_owner", "reason"],
            },
            "maxItems": 30,
        },
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["owner_name", "survey_numbers", "village", "taluk", "remarks"],
}

# ═══════════════════════════════════════════════════
# CHITTA EXTRACTION
# ═══════════════════════════════════════════════════

EXTRACT_CHITTA_SCHEMA = {
    "type": "object",
    "properties": {
        "chitta_number": {"type": "string"},
        "owner_name": {"type": "string"},
        "father_name": {"type": "string"},
        "survey_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "survey_no": {"type": "string"},
                    "extent": {"type": "string"},
                    "classification": {"type": "string"},
                    "soil_code": {"type": "string", "description": "1=புன்செய் 2=நன்செய் 3=தோட்டம் 4=மனை"},
                },
                "required": ["survey_no", "extent"],
            },
            "maxItems": 50,
        },
        "village": {"type": "string"},
        "taluk": {"type": "string"},
        "district": {"type": "string"},
        "total_extent": {"type": "string"},
        "tax_assessment": {"type": "string", "description": "Annual tax assessment amount"},
        "irrigation_source": {"type": "string"},
        "soil_classification_details": {"type": "string"},
        "remarks": {"type": "string"},
        "_field_confidence": {
            "type": "object",
            "description": "Per-field confidence scores (0.0-1.0). Include any field where confidence < 0.9.",
            "additionalProperties": {"type": "number"},
        },
        "_extraction_red_flags": {
            "type": "array",
            "description": "Red flags, anomalies, or concerns noticed during extraction.",
            "items": {"type": "string"},
            "maxItems": 15,
        },
    },
    "required": ["owner_name", "survey_numbers", "village", "taluk", "remarks"],
}
