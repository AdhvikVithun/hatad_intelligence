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
        return await _summarize_sale_deed(extracted_data, on_progress)
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
            think=True,  # CoT improves analytical summary (ownership chain, suspicious patterns)
        )
        if isinstance(result, dict):
            return result
        logger.warning(f"Tier 5 LLM enrichment returned non-dict: {type(result)}")
        return None
    except Exception as exc:
        logger.warning(f"Tier 5 LLM enrichment failed (non-fatal): {exc}")
        return None


# ── Sale Deed summarization: Tiered approach ──
# Thresholds mirror EC approach but with Sale Deed-specific logic.
_SD_SUMMARY_THRESHOLD = 30_000  # chars — most sale deeds are smaller than ECs
_SD_LLM_ENRICHMENT_THRESHOLD = 3  # ownership_history entries — trigger LLM for complex chains


async def _summarize_sale_deed(data: dict, on_progress: LLMProgressCallback | None = None) -> dict:
    """Tiered compaction for Sale Deed extracted data.

    Tier 0: Small deed (< threshold) → pass through unchanged
    Tier 1: Deterministic trimming — cap ownership_history, special_conditions, witnesses
    Tier 2: LLM enrichment for complex deeds (ownership chain analysis, risk flags)
    """
    raw_json = json.dumps(data, indent=2, default=str, ensure_ascii=False)

    # Tier 0: Small enough — pass through
    if len(raw_json) < _SD_SUMMARY_THRESHOLD:
        # Still check if LLM enrichment would add value
        oh = data.get("ownership_history", [])
        if isinstance(oh, list) and len(oh) >= _SD_LLM_ENRICHMENT_THRESHOLD:
            llm_summary = await _llm_enrich_sale_deed(data, on_progress)
            if llm_summary:
                data["_llm_summary"] = llm_summary
                data["_compaction_tier"] = 2
                logger.info("Sale Deed Tier 2 LLM enrichment applied (small doc but complex chain)")
        return data

    # Tier 1: Deterministic trimming
    compact = {}
    for key in [
        "document_number", "registration_date", "execution_date", "sro",
        "seller", "buyer", "property", "property_description",
        "financials", "stamp_paper", "registration_details",
        "payment_mode", "previous_ownership",
        "ownership_history", "encumbrance_declaration", "possession_date",
        "witnesses", "special_conditions", "power_of_attorney", "remarks",
    ]:
        if key in data:
            compact[key] = data[key]

    # Cap ownership_history to 15
    if "ownership_history" in compact:
        oh = compact["ownership_history"]
        if isinstance(oh, list) and len(oh) > 15:
            compact["ownership_history"] = oh[:15]

    # Cap special_conditions to 10
    if "special_conditions" in compact:
        sc = compact["special_conditions"]
        if isinstance(sc, list) and len(sc) > 10:
            compact["special_conditions"] = sc[:10] + [f"... and {len(sc) - 10} more"]

    # Preserve full witness list (capped at 10)
    if "witnesses" in compact:
        witnesses = compact["witnesses"]
        if isinstance(witnesses, list):
            compact["witnesses"] = witnesses[:10]
            compact["witness_count"] = len(witnesses)

    # Truncate property_description to 800 chars
    pd = compact.get("property_description", "")
    if isinstance(pd, str) and len(pd) > 800:
        compact["property_description"] = pd[:800] + "…"

    # Preserve extraction-time analysis fields
    for meta_key in ("_extraction_red_flags", "_field_confidence"):
        if meta_key in data:
            compact[meta_key] = data[meta_key]

    compact["_is_compacted"] = True
    compact["_compaction_tier"] = 1

    compact_json = json.dumps(compact, separators=(",", ":"), default=str, ensure_ascii=False)
    logger.info(
        f"Sale Deed compacted (Tier 1): {len(raw_json):,} → {len(compact_json):,} chars"
    )

    # ── Tier 2: LLM analytical enrichment ────────
    oh = data.get("ownership_history", [])
    if isinstance(oh, list) and len(oh) >= _SD_LLM_ENRICHMENT_THRESHOLD:
        llm_summary = await _llm_enrich_sale_deed(compact, on_progress)
        if llm_summary:
            compact["_llm_summary"] = llm_summary
            compact["_compaction_tier"] = 2
            logger.info(
                f"Sale Deed Tier 2 LLM enrichment: "
                f"risk_flags={len(llm_summary.get('risk_flags', []))}, "
                f"chain={len(llm_summary.get('ownership_chain', []))}"
            )

    return compact


async def _llm_enrich_sale_deed(
    data: dict,
    on_progress: LLMProgressCallback | None = None,
) -> dict | None:
    """Tier 2: LLM analytical enrichment for Sale Deeds.

    Uses the ``summarize_sale_deed.txt`` prompt to derive transaction summary,
    ownership chain analysis, risk flags, and completeness assessment.
    Returns the parsed JSON or ``None`` on failure (non-fatal).
    """
    from app.pipeline.llm_client import call_llm
    from app.pipeline.schemas import SUMMARIZE_SALE_DEED_SCHEMA

    prompt_path = PROMPTS_DIR / "summarize_sale_deed.txt"
    if not prompt_path.exists():
        logger.warning("summarize_sale_deed.txt prompt not found — skipping Sale Deed LLM enrichment")
        return None

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()

    # Build compact JSON payload
    user_prompt = json.dumps(data, separators=(",", ":"), default=str, ensure_ascii=False)
    if len(user_prompt) > 55_000:
        user_prompt = user_prompt[:55_000] + "...(truncated)"

    try:
        result = await call_llm(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            expect_json=SUMMARIZE_SALE_DEED_SCHEMA,
            task_label="Sale Deed LLM Summary (Tier 2)",
            on_progress=on_progress,
            think=True,
        )
        if isinstance(result, dict):
            return result
        logger.warning(f"Sale Deed Tier 2 LLM enrichment returned non-dict: {type(result)}")
        return None
    except Exception as exc:
        logger.warning(f"Sale Deed Tier 2 LLM enrichment failed (non-fatal): {exc}")
        return None


def build_compact_summary(extracted_data: dict, summaries: dict) -> str:
    """Build a compact consolidated text for verification/narrative stages.

    Uses summaries where available, falls back to raw data for small docs.
    """
    parts = []
    for filename, data in extracted_data.items():
        doc_type = data.get("document_type", "UNKNOWN")
        content = summaries.get(filename) or data.get("data")

        if content:
            # Use indented JSON for better LLM comprehension
            json_str = json.dumps(content, indent=2, default=str, ensure_ascii=False)
            parts.append(f"═══ {doc_type}: {filename} ═══\n{json_str}")
        else:
            error = data.get("error", "No data extracted")
            parts.append(f"═══ {doc_type}: {filename} ═══\n[Extraction failed: {error}]")

    return "\n\n".join(parts)
