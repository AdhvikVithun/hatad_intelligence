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
        "row_number", "date", "document_number", "document_year", "sro",
        "transaction_type", "seller_or_executant", "buyer_or_claimant",
        "extent", "survey_number", "consideration_amount", "remarks",
        "suspicious_flags",
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
    },
    "required": ["name", "father_name", "age", "address"],
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
            },
            "required": ["survey_number", "village", "taluk", "district", "extent", "boundaries", "property_type"],
        },
        "property_description": {"type": "string"},
        "financials": {
            "type": "object",
            "properties": {
                "consideration_amount": {"type": "integer"},
                "guideline_value": {"type": "integer"},
                "stamp_duty": {"type": "integer"},
                "registration_fee": {"type": "integer"},
            },
            "required": ["consideration_amount", "guideline_value", "stamp_duty", "registration_fee"],
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
        "witnesses": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "special_conditions": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
        "power_of_attorney": {"type": "string"},
        "remarks": {"type": "string"},
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

# Generic extractor schema (loose — accepts any keys)
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
