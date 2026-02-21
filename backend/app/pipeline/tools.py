"""Tool/function definitions for Ollama tool calling.

Each tool is registered as an Ollama-compatible function definition and
implemented as a plain Python function that the LLM client executes locally.

Tool calling enables the model to look up external data during verification
(e.g., guideline values, SRO jurisdictions, document age calculations).
"""

import json
import logging
import asyncio
from datetime import datetime, date
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Per-session context — each asyncio task gets its own RAG store / memory bank
# This replaces the unsafe module-level globals that broke concurrent sessions
_active_rag_store: ContextVar = ContextVar("_active_rag_store", default=None)
_embed_fn: ContextVar = ContextVar("_embed_fn", default=None)
_active_memory_bank: ContextVar = ContextVar("_active_memory_bank", default=None)


def set_rag_store(store, embed_fn=None):
    """Set the active RAG store for tool calls. Called by orchestrator."""
    _active_rag_store.set(store)
    _embed_fn.set(embed_fn)


def set_memory_bank(bank):
    """Set the active Memory Bank for tool calls. Called by orchestrator."""
    _active_memory_bank.set(bank)


def clear_rag_store():
    """Clear the RAG store reference after analysis."""
    _active_rag_store.set(None)
    _embed_fn.set(None)
    _active_memory_bank.set(None)


# ═══════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS (plain Python functions)
# ═══════════════════════════════════════════════════

def lookup_guideline_value(
    district: str,
    taluk: str,
    village: str,
    survey_number: str,
    year: int | None = None,
) -> dict:
    """Lookup government guideline value for a property location.
    
    Returns indicative ranges based on location classification.
    The LLM should cross-check this against the actual guideline value
    mentioned in the uploaded documents.
    """
    # Heuristic: urban taluks tend to have higher values
    urban_taluks = [
        "ambattur", "tambaram", "sriperumbudur", "alandur",
        "sholinganallur", "madhavaram", "perungudi", "guindy",
        "chennai", "avadi", "pallavaram", "chromepet",
    ]
    is_urban = taluk.lower().strip() in urban_taluks

    if is_urban:
        per_sqft = {"min": 3000, "max": 25000, "typical": 8000}
        per_cent = {"min": 500000, "max": 5000000, "typical": 1500000}
    else:
        per_sqft = {"min": 200, "max": 5000, "typical": 1200}
        per_cent = {"min": 50000, "max": 1000000, "typical": 300000}

    return {
        "district": district,
        "taluk": taluk,
        "village": village,
        "survey_number": survey_number,
        "year": year or datetime.now().year,
        "guideline_value_per_sqft": per_sqft,
        "guideline_value_per_cent": per_cent,
        "classification": "urban" if is_urban else "rural",
        "source": "Indicative range — verify against document data",
        "note": "Use search_documents or query_knowledge_base to find the actual guideline value stated in the uploaded documents.",
    }



# ── Known SRO → District mapping for Tamil Nadu ──
# SROs are named after localities, NOT districts/taluks.
# This mapping prevents false jurisdiction mismatch flags.
_SRO_DISTRICT_MAP: dict[str, str] = {
    # Coimbatore district
    "vadavalli": "coimbatore",
    "singanallur": "coimbatore",
    "sulur": "coimbatore",
    "pollachi": "coimbatore",
    "mettupalayam": "coimbatore",
    "annur": "coimbatore",
    "kinathukadavu": "coimbatore",
    "valparai": "coimbatore",
    "perur": "coimbatore",
    "thondamuthur": "coimbatore",
    "karamadai": "coimbatore",
    "madukkarai": "coimbatore",
    "coimbatore north": "coimbatore",
    "coimbatore south": "coimbatore",
    # Chennai district
    "t. nagar": "chennai",
    "anna nagar": "chennai",
    "adyar": "chennai",
    "mylapore": "chennai",
    "tambaram": "chennai",
    "sholinganallur": "chennai",
    "ambattur": "chennai",
    "velachery": "chennai",
    "perambur": "chennai",
    "saidapet": "chennai",
    "guindy": "chennai",
    "kodambakkam": "chennai",
    "egmore": "chennai",
    "kilpauk": "chennai",
    "tondiarpet": "chennai",
    "triplicane": "chennai",
    "washermanpet": "chennai",
    "madhavaram": "chennai",
    "tiruvottiyur": "chennai",
    # Chengalpattu district
    "chengalpattu": "chengalpattu",
    "maraimalai nagar": "chengalpattu",
    "pallavaram": "chengalpattu",
    "vandalur": "chengalpattu",
    "guduvanchery": "chengalpattu",
    "kelambakkam": "chengalpattu",
    "thiruporur": "chengalpattu",
    "maduranthakam": "chengalpattu",
    "chromepet": "chengalpattu",
    # Kancheepuram district
    "kancheepuram": "kancheepuram",
    "sriperumbudur": "kancheepuram",
    "uthiramerur": "kancheepuram",
    "walajabad": "kancheepuram",
    # Tiruvallur district
    "tiruvallur": "tiruvallur",
    "avadi": "tiruvallur",
    "poonamallee": "tiruvallur",
    "gummidipoondi": "tiruvallur",
    "tiruttani": "tiruvallur",
    "ponneri": "tiruvallur",
    "ennore": "tiruvallur",
    # Madurai district
    "madurai north": "madurai",
    "madurai south": "madurai",
    "melur": "madurai",
    "usilampatti": "madurai",
    "vadipatti": "madurai",
    "thirumangalam": "madurai",
    # Tiruchirappalli district
    "trichy": "tiruchirappalli",
    "srirangam": "tiruchirappalli",
    "lalgudi": "tiruchirappalli",
    "musiri": "tiruchirappalli",
    "manapparai": "tiruchirappalli",
    "thuraiyur": "tiruchirappalli",
    # Salem district
    "salem": "salem",
    "attur": "salem",
    "mettur": "salem",
    "omalur": "salem",
    # Tirunelveli district
    "tirunelveli": "tirunelveli",
    "palayamkottai": "tirunelveli",
    "tenkasi": "tirunelveli",
    "ambasamudram": "tirunelveli",
    # Erode district
    "erode": "erode",
    "gobichettipalayam": "erode",
    "bhavani": "erode",
    "sathyamangalam": "erode",
    "perundurai": "erode",
    # Thanjavur district
    "thanjavur": "thanjavur",
    "kumbakonam": "thanjavur",
    "papanasam": "thanjavur",
    "pattukkottai": "thanjavur",
    # Vellore district
    "vellore": "vellore",
    "ambur": "vellore",
    "ranipet": "vellore",
    "arcot": "vellore",
    "gudiyatham": "vellore",
    # Tiruppur district
    "tiruppur": "tiruppur",
    "avinashi": "tiruppur",
    "palladam": "tiruppur",
    "dharapuram": "tiruppur",
    "kangeyam": "tiruppur",
    "udumalpet": "tiruppur",
    # Cuddalore district
    "cuddalore": "cuddalore",
    "chidambaram": "cuddalore",
    "virudhachalam": "cuddalore",
    # Tuticorin / Thoothukudi district
    "tuticorin": "thoothukudi",
    "thoothukudi": "thoothukudi",
    "kovilpatti": "thoothukudi",
    # Villupuram district
    "villupuram": "villupuram",
    "tindivanam": "villupuram",
    "gingee": "villupuram",
}

# District alias normalization (handles Tamil vs English, old vs new names)
_DISTRICT_ALIASES: dict[str, str] = {
    "coimbatore": "coimbatore",
    "kovai": "coimbatore",
    "koyampattur": "coimbatore",
    "koyamputtur": "coimbatore",
    "chennai": "chennai",
    "madras": "chennai",
    "chengalpattu": "chengalpattu",
    "chengalpet": "chengalpattu",
    "kancheepuram": "kancheepuram",
    "kanchipuram": "kancheepuram",
    "tiruvallur": "tiruvallur",
    "thiruvallur": "tiruvallur",
    "tiruchirappalli": "tiruchirappalli",
    "trichy": "tiruchirappalli",
    "tiruchirapalli": "tiruchirappalli",
    "madurai": "madurai",
    "salem": "salem",
    "tirunelveli": "tirunelveli",
    "erode": "erode",
    "thanjavur": "thanjavur",
    "tanjore": "thanjavur",
    "vellore": "vellore",
    "tiruppur": "tiruppur",
    "cuddalore": "cuddalore",
    "thoothukudi": "thoothukudi",
    "tuticorin": "thoothukudi",
    "villupuram": "villupuram",
}


def _normalize_district(name: str) -> str:
    """Normalize a district name to a canonical English form."""
    n = name.strip().lower()
    # Remove common Tamil Unicode transliteration noise
    n = n.replace("கோயம்புத்தூர்", "coimbatore")
    n = n.replace("சென்னை", "chennai")
    n = n.replace("செங்கல்பட்டு", "chengalpattu")
    n = n.replace("திருவள்ளூர்", "tiruvallur")
    n = n.replace("காஞ்சிபுரம்", "kancheepuram")
    n = n.replace("மதுரை", "madurai")
    n = n.replace("திருச்சிராப்பள்ளி", "tiruchirappalli")
    n = n.replace("சேலம்", "salem")
    n = n.replace("திருநெல்வேலி", "tirunelveli")
    n = n.replace("ஈரோடு", "erode")
    n = n.replace("தஞ்சாவூர்", "thanjavur")
    n = n.replace("வேலூர்", "vellore")
    n = n.replace("திருப்பூர்", "tiruppur")
    n = n.replace("கடலூர்", "cuddalore")
    n = n.replace("தூத்துக்குடி", "thoothukudi")
    n = n.replace("விழுப்புரம்", "villupuram")
    return _DISTRICT_ALIASES.get(n, n)


def verify_sro_jurisdiction(
    sro_name: str,
    district: str,
    village: str,
) -> dict:
    """Verify whether a Sub-Registrar Office has jurisdiction over a village.
    
    Uses a known SRO→District mapping for Tamil Nadu plus name-matching
    heuristics. Falls back to "inconclusive" when the SRO is not in the
    known mapping (avoids false negatives).
    """
    sro = sro_name.strip().lower()
    dist = district.strip().lower()
    vill = village.strip().lower()
    norm_dist = _normalize_district(district)

    # 1. Check known SRO → district mapping
    mapped_district = _SRO_DISTRICT_MAP.get(sro)
    if mapped_district:
        if mapped_district == norm_dist:
            return {
                "sro_name": sro_name,
                "district": district,
                "village": village,
                "jurisdiction_valid": True,
                "confidence": "high",
                "note": (
                    f"{sro_name} SRO is a known Sub-Registrar Office within "
                    f"{district} district. Jurisdiction confirmed."
                ),
                "source": "Known TN SRO-District mapping",
            }
        else:
            return {
                "sro_name": sro_name,
                "district": district,
                "village": village,
                "jurisdiction_valid": False,
                "confidence": "high",
                "note": (
                    f"{sro_name} SRO is mapped to {mapped_district.title()} district, "
                    f"but the property district is {district}. Possible jurisdiction issue."
                ),
                "source": "Known TN SRO-District mapping",
            }

    # 2. Fallback: name-matching heuristic
    likely_match = (
        norm_dist in sro
        or any(part in sro for part in vill.split())
        or any(part in sro for part in norm_dist.split())
        or sro in norm_dist
    )

    if likely_match:
        return {
            "sro_name": sro_name,
            "district": district,
            "village": village,
            "jurisdiction_valid": True,
            "confidence": "medium",
            "note": (
                "Jurisdiction appears valid based on naming convention."
            ),
            "source": "Heuristic — cross-check with uploaded document data",
        }

    # 3. Unknown SRO — return inconclusive instead of false negative
    return {
        "sro_name": sro_name,
        "district": district,
        "village": village,
        "jurisdiction_valid": True,
        "confidence": "low",
        "note": (
            f"Cannot definitively confirm or deny jurisdiction for {sro_name} SRO "
            f"in {district} district. SRO names are locality-based and do not always "
            f"match the district or taluk name. No evidence of jurisdiction conflict found. "
            f"Use search_documents to verify from source documents if needed."
        ),
        "source": "Inconclusive — SRO not in known mapping, name mismatch is not evidence of conflict",
    }


def check_document_age(
    document_date: str,
    reference_date: str | None = None,
) -> dict:
    """Calculate document age and check validity periods.
    
    Useful for EC validity (typically 30 days for transactions),
    Patta currency, and limitation period checks.
    """
    ref = reference_date or datetime.now().strftime("%Y-%m-%d")

    try:
        # Try multiple date formats
        doc_dt = None
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d.%m.%Y"]:
            try:
                doc_dt = datetime.strptime(document_date, fmt).date()
                break
            except ValueError:
                continue
        
        if doc_dt is None:
            return {
                "document_date": document_date,
                "error": "Could not parse date format",
                "parsed": False,
            }

        ref_dt = None
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"]:
            try:
                ref_dt = datetime.strptime(ref, fmt).date()
                break
            except ValueError:
                continue
        
        if ref_dt is None:
            ref_dt = date.today()

        delta = ref_dt - doc_dt
        days = delta.days
        years = days / 365.25

        return {
            "document_date": document_date,
            "reference_date": ref,
            "age_days": days,
            "age_years": round(years, 1),
            "parsed": True,
            "validity_checks": {
                "ec_30_day_validity": days <= 30,
                "within_limitation_12_years": years <= 12,
                "within_limitation_3_years": years <= 3,
                "is_recent_1_year": years <= 1,
                "is_old_15_plus_years": years > 15,
            },
            "notes": _age_notes(days, years),
        }

    except Exception as e:
        return {
            "document_date": document_date,
            "reference_date": ref,
            "error": str(e),
            "parsed": False,
        }


def _age_notes(days: int, years: float) -> str:
    """Generate human-readable notes about document age."""
    notes = []
    if days <= 30:
        notes.append("Document is within 30-day EC validity window.")
    if years > 12:
        notes.append("WARNING: Document is older than 12-year limitation period.")
    if years > 15:
        notes.append("WARNING: Document >15 years old -- verify chain continuity carefully.")
    if years <= 1:
        notes.append("Recent document (< 1 year).")
    return " ".join(notes) if notes else f"Document is {years:.1f} years old."


def query_knowledge_base(
    category: str | None = None,
    key: str | None = None,
    source_file: str | None = None,
) -> dict:
    """Query the document knowledge base (Memory Bank) for extracted facts.
    
    The knowledge base is built from ALL uploaded documents during analysis.
    It contains structured facts: property details, party names, financial data,
    timeline events, encumbrances, and ownership chain information.
    
    Use this to cross-verify information between documents without needing
    to re-read the raw text.
    
    Categories: property, party, financial, timeline, encumbrance, chain, reference, risk
    """
    mb = _active_memory_bank.get()
    if mb is None:
        return {
            "error": "Knowledge base not available",
            "results": [],
            "note": "Memory Bank not initialized for this session",
        }

    try:
        results = []

        if source_file:
            # Get all facts from a specific document
            facts = mb.get_facts_by_source(source_file)
            results = facts
        elif category:
            # Get all facts in a category
            facts = mb.get_facts_by_category(category)
            if key:
                # Further filter by key
                facts = [f for f in facts if f.get("key") == key]
            results = facts
        else:
            # Return summary: categories, conflicts, cross-references
            summary = mb.get_summary()
            return {
                "total_facts": summary["total_facts"],
                "ingested_files": summary["ingested_files"],
                "categories": summary["categories"],
                "conflict_count": summary["conflict_count"],
                "conflicts": summary["conflicts"],
                "cross_references": summary["cross_references"],
            }

        return {
            "category": category,
            "key": key,
            "source_file": source_file,
            "result_count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"query_knowledge_base failed: {e}")
        return {
            "error": f"Knowledge base query failed: {str(e)}",
            "results": [],
        }


def search_documents(
    query: str,
    filename: str | None = None,
    transaction_type: str | None = None,
) -> dict:
    """Search the uploaded documents for relevant passages using semantic search.
    
    Uses RAG vector store to find the most relevant text chunks from the
    original uploaded documents. Returns passages with page citations.
    """
    rag = _active_rag_store.get()
    efn = _embed_fn.get()
    if rag is None or efn is None:
        return {
            "error": "Knowledge base not available",
            "results": [],
            "note": "RAG store not initialized for this session",
        }

    try:
        # Get embedding for the query — this function runs in a thread
        # (via loop.run_in_executor in call_llm), so asyncio.run() is safe
        # here as it creates a new event loop in this thread.
        embeddings = asyncio.run(efn([query]))

        if not embeddings or not embeddings[0]:
            return {
                "error": "Failed to generate query embedding",
                "results": [],
            }

        # Use synchronous MMR query for diverse results
        chunks = rag.query_mmr_sync(
            question_embedding=embeddings[0],
            filter_filename=filename,
            filter_transaction_type=transaction_type,
            query_text=query,
        )

        if not chunks:
            return {
                "query": query,
                "results": [],
                "note": "No relevant passages found",
            }

        # Format results with citations
        results = []
        for chunk in chunks:
            results.append({
                "text": chunk.text,
                "source": f"{chunk.filename} page {chunk.page_number}",
                "filename": chunk.filename,
                "page": chunk.page_number,
                "relevance_score": round(1 - chunk.score, 4),  # Convert distance to similarity
            })

        return {
            "query": query,
            "result_count": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error(f"search_documents failed: {e}")
        return {
            "error": f"Search failed: {str(e)}",
            "results": [],
        }


# ═══════════════════════════════════════════════════
# TOOL REGISTRY — maps function names to implementations
# ═══════════════════════════════════════════════════

TOOL_IMPLEMENTATIONS = {
    "lookup_guideline_value": lookup_guideline_value,
    "verify_sro_jurisdiction": verify_sro_jurisdiction,
    "check_document_age": check_document_age,
    "query_knowledge_base": query_knowledge_base,
    "search_documents": search_documents,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns JSON string."""
    fn = TOOL_IMPLEMENTATIONS.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result, default=str, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


# ═══════════════════════════════════════════════════
# OLLAMA TOOL DEFINITIONS (OpenAI-compatible format)
# ═══════════════════════════════════════════════════

OLLAMA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_guideline_value",
            "description": (
                "Look up the government guideline value (market value) for a property "
                "based on its location in Tamil Nadu. Use this when you need to verify "
                "whether the sale consideration is reasonable compared to guideline value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "District name (e.g., 'Chennai', 'Kancheepuram')",
                    },
                    "taluk": {
                        "type": "string",
                        "description": "Taluk name (e.g., 'Tambaram', 'Sriperumbudur')",
                    },
                    "village": {
                        "type": "string",
                        "description": "Village name",
                    },
                    "survey_number": {
                        "type": "string",
                        "description": "Survey number of the property",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year for guideline value lookup (defaults to current year)",
                    },
                },
                "required": ["district", "taluk", "village", "survey_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_sro_jurisdiction",
            "description": (
                "Verify whether a Sub-Registrar Office (SRO) has jurisdiction over "
                "a given village/area. Use this to check if a document was registered "
                "at the correct SRO."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sro_name": {
                        "type": "string",
                        "description": "Name of the Sub-Registrar Office",
                    },
                    "district": {
                        "type": "string",
                        "description": "District where the property is located",
                    },
                    "village": {
                        "type": "string",
                        "description": "Village where the property is located",
                    },
                },
                "required": ["sro_name", "district", "village"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_document_age",
            "description": (
                "Calculate the age of a document and check various validity periods "
                "(30-day EC validity, 12-year limitation period, etc.). "
                "Use this for any date-based validity checks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_date": {
                        "type": "string",
                        "description": "Date of the document (YYYY-MM-DD or DD-MM-YYYY format)",
                    },
                    "reference_date": {
                        "type": "string",
                        "description": "Reference date to calculate age against (defaults to today)",
                    },
                },
                "required": ["document_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": (
                "Query the document knowledge base for structured facts extracted from uploaded documents. "
                "The knowledge base contains property details, party names, financial data, timelines, "
                "encumbrances, ownership chain, and cross-document references. Use this to cross-verify "
                "information between documents (e.g., check if survey number in EC matches Patta, "
                "compare seller name in deed with EC entries, verify consideration amounts). "
                "Call without arguments to get a summary of all facts and any detected conflicts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Fact category to query: property, party, financial, timeline, "
                            "encumbrance, chain, reference, risk"
                        ),
                    },
                    "key": {
                        "type": "string",
                        "description": (
                            "Specific fact key within the category (e.g., 'survey_number', 'seller', "
                            "'consideration_amount', 'ec_number'). Optional — omit to get all facts in category."
                        ),
                    },
                    "source_file": {
                        "type": "string",
                        "description": "Filter facts by source document filename. Omit to search across all documents.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the uploaded source documents for relevant passages using semantic search. "
                "Use this to find exact text, verify claims, locate specific clauses, or retrieve "
                "original wording from any uploaded document. Returns matching passages with page "
                "numbers for citation. Use this tool whenever you need to verify a fact against "
                "the original document text or find specific details. You can filter by transaction "
                "type (e.g. 'mortgage', 'sale', 'lease') to retrieve only EC entries of that type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language search query describing what to find. "
                            "Be specific, e.g., 'seller name and address in sale deed' or "
                            "'survey number boundaries in EC' or 'mortgage or lien entries'"
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Optional: restrict search to a specific document filename. "
                            "Omit to search across all uploaded documents."
                        ),
                    },
                    "transaction_type": {
                        "type": "string",
                        "description": (
                            "Optional: restrict to EC transaction entries of this type. "
                            "Valid values: sale, mortgage, release, lease, gift, partition, "
                            "will, agreement, court_order, attachment, other. "
                            "Omit to search all chunk types."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]
