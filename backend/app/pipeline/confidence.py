"""Extraction confidence assessor — decides when vision fallback is needed.

Analyses a GPT-OSS text extraction result and the source OCR quality to
produce a confidence score.  When confidence is low the orchestrator asks
Qwen3-VL to visually re-read specific weak fields from the PDF images.

Signals evaluated
─────────────────
1. OCR quality (from ingestion page-level metrics)
2. Required-field emptiness (empty / placeholder / "unknown" values)
3. GPT self-reported uncertainty (phrases in remarks / extraction_notes)
4. Numeric sanity (zero amounts, impossible dates, negative values)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── Placeholder / low-confidence value patterns ──────────────────────
_PLACEHOLDER_RE = re.compile(
    r"^("
    r"n/?a|not\s*available|not\s*found|not\s*provided|not\s*mentioned"
    r"|unknown|unclear|illegible|unreadable|unable\s*to\s*(extract|read|determine)"
    r"|nil|none|empty|\-{2,}|\.{3,}|\?"
    r"|0{4}[-/]0{1,2}[-/]0{1,2}"   # 0000-00-00 style dates
    r")$",
    re.IGNORECASE,
)

_UNCERTAINTY_RE = re.compile(
    r"unclear|illegible|could\s*not\s*(determine|read|extract)"
    r"|unable\s*to|unreadable|uncertain|ambiguous|partially\s*visible"
    r"|low.?quality|garbled|corrupted|missing|not\s*legible",
    re.IGNORECASE,
)

_INVALID_DATE_RE = re.compile(
    r"^(0{4}|9{4})[-/](0{1,2}|9{1,2})[-/](0{1,2}|9{1,2})$"
)

# Valid normalized survey number: digits with optional /- subdivisions and alpha
_VALID_SURVEY_RE = re.compile(r'^\d+[a-zA-Z]?(?:[/\-]\d+[a-zA-Z]?\d*)*$')


def _validate_field_patterns(result: dict) -> tuple[list[str], list[str]]:
    """Check extracted values against expected format patterns.

    Returns (weak_fields, reasons) for fields that fail pattern checks.
    Triggers vision fallback for garbled survey numbers, invalid dates,
    implausible names, and unparseable amounts.
    """
    from app.pipeline.utils import normalize_survey_number, normalize_tamil_numerals, parse_amount

    weak: list[str] = []
    reasons: list[str] = []

    # ── Survey number format ──
    # Check top-level survey number fields and nested property.survey_number
    _survey_sources = []
    prop = result.get("property", {})
    if isinstance(prop, dict) and prop.get("survey_number"):
        _survey_sources.append(("property.survey_number", str(prop["survey_number"])))
    for sn_obj in (result.get("survey_numbers") or []):
        if isinstance(sn_obj, dict) and sn_obj.get("survey_no"):
            _survey_sources.append(("survey_numbers", str(sn_obj["survey_no"])))

    for field_name, raw_sn in _survey_sources:
        norm = normalize_survey_number(raw_sn)
        if norm and not _VALID_SURVEY_RE.match(norm):
            weak.append(field_name)
            reasons.append(f"'{field_name}' has invalid survey format: '{raw_sn}'")

    # ── Location name validity ──
    for nf in ("village", "taluk", "district"):
        val = result.get(nf) or (prop.get(nf) if isinstance(prop, dict) else None)
        if val and isinstance(val, str):
            stripped = val.strip()
            if len(stripped) < 3:
                weak.append(nf)
                reasons.append(f"'{nf}' is suspiciously short: '{stripped}'")
            elif len(stripped) > 60:
                weak.append(nf)
                reasons.append(f"'{nf}' is suspiciously long ({len(stripped)} chars)")
            elif stripped.replace(" ", "").isdigit():
                weak.append(nf)
                reasons.append(f"'{nf}' contains only digits: '{stripped}'")

    # ── Amount parseability ──
    financials = result.get("financials", {})
    if isinstance(financials, dict):
        for amt_field in ("consideration_amount", "guideline_value", "stamp_duty"):
            raw = financials.get(amt_field)
            if raw is not None and raw != "" and raw != 0:
                parsed = parse_amount(raw)
                if parsed is None:
                    weak.append(f"financials.{amt_field}")
                    reasons.append(f"'financials.{amt_field}' is unparseable: '{raw}'")

    return weak, reasons


def _detect_garbled_tamil_fields(result: dict) -> tuple[list[str], list[str]]:
    """Check extracted string fields for garbled Tamil text.

    Walks all string values in the result and checks each Tamil-containing
    string for syllable-structure corruption.

    Returns (weak_fields, reasons).
    """
    from app.pipeline.utils import detect_garbled_tamil

    weak: list[str] = []
    reasons: list[str] = []

    def _check(field_path: str, val: str) -> None:
        if not val or not isinstance(val, str) or len(val) < 4:
            return
        is_garbled, quality, reason = detect_garbled_tamil(val)
        if is_garbled:
            weak.append(field_path)
            reasons.append(
                f"'{field_path}' contains garbled Tamil (quality={quality:.2f}): {reason}"
            )

    # Top-level string fields
    for key, val in result.items():
        if key.startswith("_"):
            continue
        if isinstance(val, str):
            _check(key, val)
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, str):
                    _check(f"{key}.{sub_key}", sub_val)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, dict):
                    for sub_key, sub_val in item.items():
                        if isinstance(sub_val, str):
                            _check(f"{key}[{i}].{sub_key}", sub_val)
                elif isinstance(item, str):
                    _check(f"{key}[{i}]", item)

    return weak, reasons


@dataclass
class ConfidenceResult:
    """Outcome of extraction confidence assessment."""
    score: float                      # 0.0 (no confidence) – 1.0 (fully confident)
    needs_vision: bool                # True when score < threshold
    weak_fields: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    field_confidences: dict[str, float] = field(default_factory=dict)  # per-field scores 0.0–1.0


def _get_required_fields(schema: dict) -> list[str]:
    """Return the top-level required field names from a JSON Schema."""
    return list(schema.get("required", []))


def _is_empty_value(val) -> bool:
    """Check whether a value is empty or a known placeholder."""
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip() or bool(_PLACEHOLDER_RE.match(val.strip()))
    if isinstance(val, list):
        return len(val) == 0
    if isinstance(val, dict):
        return len(val) == 0
    return False


def _is_zero_numeric(val) -> bool:
    """Check if a numeric field is suspiciously zero."""
    if isinstance(val, (int, float)):
        return val == 0
    if isinstance(val, str):
        s = val.strip()
        return s in ("0", "0.0", "0.00", "₹0", "Rs.0", "Rs. 0")
    return False


def _walk_weak_fields(result: dict, schema: dict) -> tuple[list[str], list[str]]:
    """Walk the extraction result and find fields with empty/placeholder values.

    Returns (weak_field_names, reasons).
    """
    required = _get_required_fields(schema)
    properties = schema.get("properties", {})
    weak: list[str] = []
    reasons: list[str] = []

    for field_name in required:
        val = result.get(field_name)

        # Get the field's schema to check nested structures
        field_schema = properties.get(field_name, {})
        field_type = field_schema.get("type")

        if _is_empty_value(val):
            weak.append(field_name)
            reasons.append(f"'{field_name}' is empty or placeholder")
            continue

        # For nested objects, check if all sub-fields are empty
        if field_type == "object" and isinstance(val, dict):
            sub_required = field_schema.get("required", [])
            sub_empty = sum(1 for k in sub_required if _is_empty_value(val.get(k)))
            if sub_required and sub_empty / len(sub_required) > 0.5:
                weak.append(field_name)
                reasons.append(f"'{field_name}' has {sub_empty}/{len(sub_required)} empty sub-fields")

        # For arrays of objects, check if items are mostly empty
        if field_type == "array" and isinstance(val, list):
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "object" and val:
                item_required = items_schema.get("required", [])
                if item_required:
                    total_fields = len(item_required) * len(val)
                    empty_fields = 0
                    for item in val:
                        if isinstance(item, dict):
                            empty_fields += sum(
                                1 for k in item_required if _is_empty_value(item.get(k))
                            )
                    if total_fields and empty_fields / total_fields > 0.4:
                        weak.append(field_name)
                        reasons.append(
                            f"'{field_name}' array items have {empty_fields}/{total_fields} empty sub-fields"
                        )

        # Zero-value sanity for financial / numeric fields
        _FINANCIAL_KEYS = {
            "consideration_amount", "guideline_value", "stamp_duty",
            "registration_fee", "total_entries_found",
        }
        if field_name in _FINANCIAL_KEYS:
            if _is_zero_numeric(val):
                weak.append(field_name)
                reasons.append(f"'{field_name}' is zero (suspicious)")
        elif field_name == "financials" and isinstance(val, dict):
            # financials is a nested object — check its sub-fields
            for k, v in val.items():
                if _is_zero_numeric(v) and k in _FINANCIAL_KEYS:
                    weak.append(f"{field_name}.{k}")
                    reasons.append(f"'{field_name}.{k}' is zero (suspicious)")

    return weak, reasons


def assess_extraction_confidence(
    result: dict,
    schema: dict,
    extraction_quality: str = "HIGH",
    *,
    threshold: float = 0.7,
) -> ConfidenceResult:
    """Assess confidence in a text-based extraction result.

    Args:
        result: Extracted data dict from GPT-OSS.
        schema: The JSON Schema used for this document type.
        extraction_quality: Overall OCR quality ("HIGH", "LOW", "MIXED", "EMPTY").
        threshold: Score below which ``needs_vision`` is set.

    Returns:
        ConfidenceResult with score, needs_vision flag, weak fields, and reasons.
    """
    if not isinstance(result, dict) or result.get("_fallback"):
        req_fields = list(schema.get("required", []))
        return ConfidenceResult(
            score=0.0,
            needs_vision=True,
            weak_fields=req_fields,
            reasons=["Extraction returned fallback/empty result"],
            field_confidences={f: 0.0 for f in req_fields},
        )

    score = 1.0
    all_weak: list[str] = []
    all_reasons: list[str] = []

    # ── Signal 1: OCR quality ────────────────────────────────────────
    if extraction_quality == "EMPTY":
        score -= 0.5
        all_reasons.append("OCR quality is EMPTY — no readable text")
    elif extraction_quality == "LOW":
        score -= 0.35
        all_reasons.append("OCR quality is LOW — garbled / insufficient text")
    elif extraction_quality == "MIXED":
        score -= 0.15
        all_reasons.append("OCR quality is MIXED — some pages have poor text")

    # ── Signal 2: Field emptiness ────────────────────────────────────
    required = _get_required_fields(schema)
    if required:
        weak, reasons = _walk_weak_fields(result, schema)
        all_weak.extend(weak)
        all_reasons.extend(reasons)

        empty_ratio = len(weak) / max(len(required), 1)
        score -= empty_ratio * 0.5  # Up to −0.5 if ALL fields are empty

    # ── Signal 3: GPT self-reported uncertainty ──────────────────────
    for note_field in ("remarks", "extraction_notes"):
        text = result.get(note_field, "")
        if isinstance(text, str) and _UNCERTAINTY_RE.search(text):
            score -= 0.15
            all_reasons.append(f"GPT flagged uncertainty in '{note_field}'")
            break  # Only penalise once

    # ── Signal 4: Date sanity ────────────────────────────────────────
    for date_field in ("registration_date", "period_from", "period_to"):
        val = result.get(date_field, "")
        if isinstance(val, str) and _INVALID_DATE_RE.match(val.strip()):
            if date_field not in all_weak:
                all_weak.append(date_field)
            score -= 0.1
            all_reasons.append(f"'{date_field}' has an invalid date")

    # ── Signal 5: Field pattern validation ───────────────────────────
    pattern_weak, pattern_reasons = _validate_field_patterns(result)
    if pattern_weak:
        all_weak.extend(pattern_weak)
        all_reasons.extend(pattern_reasons)
        # Deduct up to −0.2 based on ratio of invalid fields
        score -= min(0.2, 0.05 * len(pattern_weak))

    # ── Signal 6: Garbled Tamil detection ────────────────────────────
    tamil_weak, tamil_reasons = _detect_garbled_tamil_fields(result)
    if tamil_weak:
        all_weak.extend(tamil_weak)
        all_reasons.extend(tamil_reasons)
        score -= min(0.40, 0.12 * len(tamil_weak))

    # Clamp
    score = max(0.0, min(1.0, score))

    needs_vision = score < threshold

    # ── Build per-field confidence scores ────────────────────────────
    # Start each required field at 1.0, deduct for each signal that flags it.
    deduped_weak = list(dict.fromkeys(all_weak))
    field_conf: dict[str, float] = {}
    for f in required:
        fc = 1.0
        # Count how many times this field was flagged as weak
        hit_count = sum(1 for w in all_weak if w == f or w.startswith(f + ".") or w.startswith(f + "["))
        fc -= min(0.8, hit_count * 0.25)  # Each hit deducts 0.25, max 0.8
        field_conf[f] = round(max(0.0, fc), 3)

    # Also track non-required fields that were flagged
    for w in deduped_weak:
        base_field = w.split(".")[0].split("[")[0]
        if base_field not in field_conf:
            hit_count = sum(1 for x in all_weak if x == w or x.startswith(w + "."))
            field_conf[w] = round(max(0.0, 1.0 - hit_count * 0.25), 3)

    cr = ConfidenceResult(
        score=round(score, 3),
        needs_vision=needs_vision,
        weak_fields=deduped_weak,
        reasons=all_reasons,
        field_confidences=field_conf,
    )

    logger.info(
        f"Confidence: {cr.score:.2f}, needs_vision={cr.needs_vision}, "
        f"weak_fields={cr.weak_fields}"
    )
    return cr
