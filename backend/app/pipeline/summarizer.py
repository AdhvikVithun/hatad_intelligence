"""Document summarizer — creates compact LLM-friendly summaries of extracted data.

Used between Stage 3 (extraction) and Stage 4 (verification) to keep
downstream prompts within context window limits.
"""

import json
from app.config import PROMPTS_DIR
from app.pipeline.llm_client import call_llm, LLMProgressCallback
from app.pipeline.schemas import SUMMARIZE_EC_SCHEMA


# Threshold: if extracted JSON exceeds this many chars, run LLM summarizer
EC_SUMMARY_THRESHOLD = 50_000  # ~12,500 tokens — medium ECs fit in 64K context, skip summarization


async def summarize_document(
    doc_type: str,
    extracted_data: dict,
    on_progress: LLMProgressCallback | None = None,
) -> dict:
    """Create a compact summary of extracted data for downstream LLM calls.

    Returns a smaller dict that preserves legally significant information
    but fits within context windows.
    """
    if doc_type == "EC":
        return await _summarize_ec(extracted_data, on_progress)
    elif doc_type == "SALE_DEED":
        return _compact_sale_deed(extracted_data)
    else:
        # Patta, Chitta, FMB, Generic — already small, pass through
        return extracted_data


async def _summarize_ec(data: dict, on_progress: LLMProgressCallback | None = None) -> dict:
    """Summarize a large EC via LLM — condenses 200+ transactions into key findings."""
    raw_json = json.dumps(data, indent=2, default=str, ensure_ascii=False)

    # If already small enough, no need to summarize
    if len(raw_json) < EC_SUMMARY_THRESHOLD:
        return data

    system_prompt = (PROMPTS_DIR / "summarize_ec.txt").read_text(encoding="utf-8")

    txn_count = data.get("total_entries_found", len(data.get("transactions", [])))
    prompt = (
        f"Summarize this EC with {txn_count} transactions. "
        f"Preserve ALL ownership transfers, active encumbrances, and suspicious patterns.\n\n"
        f"{raw_json}"
    )

    summary = await call_llm(
        prompt=prompt,
        system_prompt=system_prompt,
        expect_json=SUMMARIZE_EC_SCHEMA,
        task_label=f"Summarize EC ({txn_count} transactions)",
        on_progress=on_progress,
        think=False,  # Summarization is compaction, not reasoning — disable CoT
    )

    # Attach metadata so downstream knows this is a summary
    summary["_is_summary"] = True
    summary["_original_transaction_count"] = txn_count
    return summary


def _compact_sale_deed(data: dict) -> dict:
    """Trim verbose Sale Deed fields — no LLM call needed."""
    compact = {}
    # Keep essential fields — must match schema keys from extraction
    for key in [
        "document_number", "registration_date", "sro",
        "seller", "buyer", "property_schedule",
        "property",  # nested object: survey_number, village, boundaries etc.
        "financials",  # nested object: consideration, guideline_value, stamp_duty
        "consideration_amount", "guideline_value", "stamp_duty",
        "survey_number", "extent", "boundaries",
        "encumbrances", "conditions",
        "previous_ownership", "execution_date",
    ]:
        if key in data:
            compact[key] = data[key]

    # Truncate verbose fields
    if "special_conditions" in data:
        sc = data["special_conditions"]
        if isinstance(sc, list) and len(sc) > 5:
            compact["special_conditions"] = sc[:5] + [f"... and {len(sc) - 5} more"]
        else:
            compact["special_conditions"] = sc

    if "witnesses" in data:
        compact["witness_count"] = len(data["witnesses"]) if isinstance(data["witnesses"], list) else 1

    return compact


def build_compact_summary(extracted_data: dict, summaries: dict) -> str:
    """Build a compact consolidated text for verification/narrative stages.

    Uses summaries where available, falls back to raw data for small docs.
    """
    parts = []
    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "UNKNOWN")
        content = summaries.get(filename) or data.get("data")

        if content:
            # Use compact JSON (no indent) to save space
            json_str = json.dumps(content, separators=(",", ":"), default=str, ensure_ascii=False)
            parts.append(f"═══ {doc_type}: {filename} ═══\n{json_str}")
        else:
            error = data.get("error", "No data extracted")
            parts.append(f"═══ {doc_type}: {filename} ═══\n[Extraction failed: {error}]")

    return "\n\n".join(parts)
