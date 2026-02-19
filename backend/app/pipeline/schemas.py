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
            "items": {"type": "string"},
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
        "date": {"type": "string"},
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
        "remarks": {"type": "string"},
        "suspicious_flags": {
            "type": "array",
            "items": {"type": "string"},
        },
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
    },
    "required": ["name", "father_name", "age", "address"],
}

EXTRACT_SALE_DEED_SCHEMA = {
    "type": "object",
    "properties": {
        "document_number": {"type": "string"},
        "registration_date": {"type": "string"},
        "sro": {"type": "string"},
        "seller": {"type": "array", "items": _PARTY_SCHEMA},
        "buyer": {"type": "array", "items": _PARTY_SCHEMA},
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
        "witnesses": {"type": "array", "items": {"type": "string"}},
        "special_conditions": {"type": "array", "items": {"type": "string"}},
        "power_of_attorney": {"type": "string"},
        "remarks": {"type": "string"},
    },
    "required": [
        "document_number", "registration_date", "sro", "seller", "buyer",
        "property", "financials", "previous_ownership", "witnesses",
        "special_conditions", "power_of_attorney", "remarks",
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
        "ownership_chain": {"type": "array", "items": {"type": "string"}},
        "active_encumbrances": {"type": "array", "items": {"type": "string"}},
        "suspicious_patterns": {"type": "array", "items": {"type": "string"}},
        "key_statistics": {
            "type": "object",
            "properties": {
                "sales": {"type": "integer"},
                "mortgages": {"type": "integer"},
                "releases": {"type": "integer"},
                "partitions": {"type": "integer"},
                "others": {"type": "integer"},
                "date_range": {"type": "string"},
                "unique_parties": {"type": "array", "items": {"type": "string"}},
                "survey_numbers": {"type": "array", "items": {"type": "string"}},
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
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
        "chain_of_title": {"type": "array", "items": _CHAIN_LINK_SCHEMA},
        "active_encumbrances": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 2: Sale Deed
VERIFY_GROUP2_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 3: Cross-Document
VERIFY_GROUP3_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
    },
    "required": ["group", "group_score_deduction", "checks"],
}

# Group 4: Chain & Patterns
VERIFY_GROUP4_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string"},
        "group_score_deduction": {"type": "integer"},
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
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
        "checks": {"type": "array", "items": _CHECK_SCHEMA},
        "missing_documents": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
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
    4: VERIFY_GROUP4_SCHEMA,
    5: VERIFY_GROUP5_SCHEMA,
}

# Generic extractor schema (loose — accepts any keys)
EXTRACT_GENERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "document_summary": {"type": "string"},
        "key_parties": {"type": "array", "items": {"type": "string"}},
        "property_details": {"type": "object"},
        "key_dates": {"type": "array", "items": {"type": "string"}},
        "amounts": {"type": "array", "items": {"type": "string"}},
        "notable_clauses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["document_summary", "key_parties", "property_details",
                  "key_dates", "amounts", "notable_clauses"],
}
