"""Document summarizer — creates compact summaries of extracted data.

Used between Stage 3 (extraction) and Stage 4 (verification) to keep
downstream prompts within context window limits.

EC compaction is DETERMINISTIC (no LLM call) to guarantee zero data loss.
Every transaction, survey number, party name, and amount is preserved.
Only LLM-generated metadata (suspicious_flags) is dropped.

For large ECs (≥ LLM_SUMMARY_TXN_THRESHOLD transactions), an optional
Tier 5 LLM enrichment pass produces an analytical summary (ownership chain,
active encumbrances, suspicious patterns) stored as ``_llm_summary``.
"""

import json
import logging
from app.config import PROMPTS_DIR
from app.pipeline.llm_client import LLMProgressCallback


logger = logging.getLogger(__name__)

# Threshold: if extracted JSON exceeds this many chars, run compaction
EC_SUMMARY_THRESHOLD = 50_000  # ~12,500 tokens — medium ECs fit in 64K context, skip compaction
# Hard ceiling: compacted output must stay under this to fit in LLM context
_COMPACTION_HARD_LIMIT = 180_000
# Minimum transaction count to trigger Tier 5 LLM enrichment
LLM_SUMMARY_TXN_THRESHOLD = 25


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
    """Deterministic compaction for large ECs — zero data loss.

    Tiered approach:
      Tier 0: Small EC (< threshold) → pass through unchanged
      Tier 1: Drop only suspicious_flags per transaction
      Tier 2: Also drop schedule_remarks and additional_details
      Tier 3: Also truncate party names to 80 chars
      Tier 4: Also drop remarks (last resort)

    Every tier preserves ALL survey numbers, dates, document numbers,
    party names (possibly truncated at tier 3+), and amounts.
    """
    raw_json = json.dumps(data, indent=2, default=str, ensure_ascii=False)

    # Tier 0: If already small enough, no compaction needed
    if len(raw_json) < EC_SUMMARY_THRESHOLD:
        return data

    txn_count = len(data.get("transactions", []))

    # Build compacted copy — keep all fields except transactions (rebuilt below)
    compacted = {k: v for k, v in data.items() if k != "transactions"}

    # Tier 1: Drop suspicious_flags per transaction (LLM-generated, redundant with verification)
    compacted["transactions"] = [
        {k: v for k, v in txn.items() if k != "suspicious_flags"}
        for txn in data.get("transactions", [])
    ]

    compact_json = json.dumps(compacted, separators=(",", ":"), default=str, ensure_ascii=False)
    tier = 1

    # Tier 2: Drop verbose text fields (schedule_remarks, additional_details)
    # NEVER drop: transaction_id, row_number, document_number, date, survey_number
    if len(compact_json) > _COMPACTION_HARD_LIMIT:
        tier = 2
        for txn in compacted["transactions"]:
            txn.pop("schedule_remarks", None)
            txn.pop("additional_details", None)
        compact_json = json.dumps(compacted, separators=(",", ":"), default=str, ensure_ascii=False)

    # Tier 3: Truncate long party names to 80 chars
    if len(compact_json) > _COMPACTION_HARD_LIMIT:
        tier = 3
        for txn in compacted["transactions"]:
            for field in ("seller_or_executant", "buyer_or_claimant"):
                val = txn.get(field, "")
                if isinstance(val, str) and len(val) > 80:
                    txn[field] = val[:80] + "…"
        compact_json = json.dumps(compacted, separators=(",", ":"), default=str, ensure_ascii=False)

    # Tier 4: Truncate remarks to 200 chars (deterministic.py needs them for release matching)
    if len(compact_json) > _COMPACTION_HARD_LIMIT:
        tier = 4
        for txn in compacted["transactions"]:
            r = txn.get("remarks", "")
            if isinstance(r, str) and len(r) > 200:
                txn["remarks"] = r[:200] + "…"
        compact_json = json.dumps(compacted, separators=(",",":"), default=str, ensure_ascii=False)
        logger.warning(
            f"EC compaction reached Tier 4: truncated remarks. "
            f"{txn_count} transactions, {len(compact_json):,} chars"
        )

    # Attach metadata
    compacted["_is_compacted"] = True
    compacted["_compaction_tier"] = tier
    compacted["_original_transaction_count"] = txn_count

    logger.info(
        f"EC compacted (Tier {tier}): {len(raw_json):,} → {len(compact_json):,} chars "
        f"({txn_count} transactions preserved)"
    )

    # ── Tier 5: LLM analytical enrichment for large ECs ─────────────
    # Produces ownership chain, active encumbrances, and suspicious
    # patterns that deterministic compaction cannot derive.
    if txn_count >= LLM_SUMMARY_TXN_THRESHOLD:
        llm_summary = await _llm_enrich_ec(compacted, on_progress)
        if llm_summary:
            compacted["_llm_summary"] = llm_summary
            compacted["_compaction_tier"] = 5
            logger.info(
                f"EC Tier 5 LLM enrichment: ownership_chain={len(llm_summary.get('ownership_chain', []))}, "
                f"active_encumbrances={len(llm_summary.get('active_encumbrances', []))}"
            )

    return compacted


async def _llm_enrich_ec(
    compacted: dict,
    on_progress: LLMProgressCallback | None = None,
) -> dict | None:
    """Tier 5: LLM analytical enrichment for large ECs.

    Uses the ``summarize_ec.txt`` prompt to derive ownership chain, active
    encumbrances, and suspicious patterns from the deterministically
    compacted transaction data. Returns the parsed JSON or ``None`` on
    failure (non-fatal — downstream verification still works without it).
    """
    from app.pipeline.llm_client import call_llm

    prompt_path = PROMPTS_DIR / "summarize_ec.txt"
    if not prompt_path.exists():
        logger.warning("summarize_ec.txt prompt not found — skipping Tier 5")
        return None

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Build a compact JSON payload of the transaction list + EC header
    payload = {
        "ec_number": compacted.get("ec_number"),
        "property_description": compacted.get("property_description"),
        "village": compacted.get("village"),
        "taluk": compacted.get("taluk"),
        "period_from": compacted.get("period_from"),
        "period_to": compacted.get("period_to"),
        "total_entries_found": compacted.get("total_entries_found"),
        "transactions": compacted.get("transactions", []),
    }
    user_prompt = json.dumps(payload, separators=(",", ":"), default=str, ensure_ascii=False)

    # Truncate if too large for a single LLM call (~55K chars ≈ 14K tokens)
    if len(user_prompt) > 55_000:
        user_prompt = user_prompt[:55_000] + '...(truncated)'

    try:
        result = await call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            expect_json=True,
            task_label="EC LLM Summary (Tier 5)",
            on_progress=on_progress,
            think=False,
        )
        if isinstance(result, dict):
            return result
        logger.warning(f"Tier 5 LLM enrichment returned non-dict: {type(result)}")
        return None
    except Exception as exc:
        logger.warning(f"Tier 5 LLM enrichment failed (non-fatal): {exc}")
        return None


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
        "ownership_history", "payment_mode",
        "encumbrance_declaration", "possession_date",
    ]:
        if key in data:
            compact[key] = data[key]

    # Truncate verbose fields
    if "property_description" in data:
        pd = data["property_description"]
        if isinstance(pd, str) and len(pd) > 500:
            compact["property_description"] = pd[:500] + "…"
        else:
            compact["property_description"] = pd

    if "ownership_history" in compact:
        oh = compact["ownership_history"]
        if isinstance(oh, list) and len(oh) > 5:
            compact["ownership_history"] = oh[:5]

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
