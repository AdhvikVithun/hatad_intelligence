"""Self-reflection loop — post-verification LLM consistency checker.

After all 5 verification groups complete, this module sends the full
set of check results back to the LLM and asks it to identify:
  - Internal contradictions (PASS on mortgage check but evidence mentions active mortgage)
  - Status-evidence mismatches (FAIL with weak evidence, PASS with damning evidence)
  - Missing cross-references between groups
  - Severity downgrades that seem unjustified

The reflection can AMEND check statuses (upgrade WARNING→FAIL) or add
advisory notes. It does NOT add new checks — that's the LLM's job in
the original pass.

Enhanced: Now uses tools (query_knowledge_base, search_documents) to verify
suspicious findings against source documents, rather than reasoning in isolation.
"""

import json
import logging

from app.pipeline.llm_client import call_llm, LLMProgressCallback

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM_PROMPT = """\
You are a senior legal auditor reviewing the output of a junior analyst's land due diligence verification.

Your ONLY job is to find CONTRADICTIONS and ERRORS in the check results below. You are NOT performing new checks.

Look for these specific issues:

1. STATUS-EVIDENCE CONTRADICTION: A check is marked PASS but the evidence text describes a problem (e.g., "PASS" on mortgage check but evidence mentions "mortgage in favour of SBI"). Conversely, a check marked FAIL but evidence shows the issue was resolved.

2. CROSS-GROUP CONTRADICTION: Two checks from different groups contradict each other (e.g., Group 1 says no mortgage exists but Group 3's evidence mentions a mortgage).

3. SEVERITY UNDERESTIMATION: A check with genuinely serious findings (active mortgage, broken chain, poramboke land) is rated as WARNING or MEDIUM when it should be FAIL or CRITICAL.

4. EVIDENCE QUALITY: A check marked FAIL or WARNING has evidence that is too vague, lacks document citations, or appears fabricated (generic text not matching any real document).

TOOL USAGE: You have access to tools to verify suspicious findings:
- query_knowledge_base: Look up structured facts (survey numbers, names, amounts, dates) to cross-check evidence claims. Call without arguments for a full summary.
- search_documents: Semantic search in the original uploaded documents. Use this to verify quoted text or find contradicting passages.

When you find a suspicious check result, USE these tools to verify before proposing an amendment. Your amendments will be MORE credible with tool-verified evidence.

For each issue found, return an amendment. If no issues are found, return an empty amendments list.

═══ FEW-SHOT EXAMPLE ═══

Input check:
{
  "rule_code": "ACTIVE_MORTGAGE",
  "status": "PASS",
  "evidence": "[EC.pdf] page 4: 'Mortgage in favour of Indian Bank dated 12.05.2018, Document No. 1234/2018'. No discharge entry found."
}

Correct amendment:
{
  "rule_code": "ACTIVE_MORTGAGE",
  "issue": "STATUS_EVIDENCE_CONTRADICTION",
  "original_status": "PASS",
  "amended_status": "FAIL",
  "reason": "Evidence explicitly states 'Mortgage in favour of Indian Bank' with 'No discharge entry found', which means the mortgage is ACTIVE. This contradicts the PASS status and should be FAIL."
}

Return ONLY valid JSON:
{
  "amendments": [
    {
      "rule_code": "RULE_CODE_HERE",
      "issue": "STATUS_EVIDENCE_CONTRADICTION|CROSS_GROUP_CONTRADICTION|SEVERITY_UNDERESTIMATION|EVIDENCE_QUALITY",
      "original_status": "PASS|FAIL|WARNING",
      "amended_status": "PASS|FAIL|WARNING",
      "reason": "explanation of why the amendment is needed"
    }
  ],
  "reflection_notes": "Brief summary of overall quality assessment"
}
"""


async def run_self_reflection(
    all_checks: list[dict],
    on_progress: LLMProgressCallback | None = None,
    tools: list[dict] | None = None,
) -> dict:
    """Run self-reflection over the completed check results.

    Args:
        all_checks: List of check result dicts from all verification groups
        on_progress: Async callback for progress updates
        tools: Ollama tool definitions (query_knowledge_base, search_documents)
               for verifying suspicious findings against source documents

    Returns:
        dict with 'amendments' (list) and 'reflection_notes' (str)
    """
    # Build a compact representation of checks for the LLM
    # Skip system-generated fallback checks (errors/skipped) — they are
    # not LLM outputs and confuse the reflection model
    compact_checks = []
    for check in all_checks:
        code = check.get("rule_code", "")
        if code.endswith("_ERROR") or code.endswith("_SKIPPED"):
            continue
        # Also skip deterministic checks (source=deterministic) — pure Python, no LLM to audit
        if check.get("source") == "deterministic":
            continue
        compact_checks.append({
            "rule_code": code,
            "rule_name": check.get("rule_name", ""),
            "severity": check.get("severity", ""),
            "status": check.get("status", ""),
            "explanation": check.get("explanation", "")[:300],
            "evidence": check.get("evidence", "")[:400],
        })

    prompt = (
        f"Review these {len(compact_checks)} verification check results for internal "
        f"contradictions, status-evidence mismatches, and cross-group inconsistencies.\n\n"
        f"CHECK RESULTS:\n{json.dumps(compact_checks, indent=2, ensure_ascii=False)}"
    )

    try:
        # Build tool hint if tools are available
        tool_hint = ""
        if tools:
            tool_hint = (
                "\n\nBefore proposing amendments, use the available tools to verify your findings:\n"
                "- Call query_knowledge_base to check structured facts\n"
                "- Call search_documents to find exact text in source documents\n"
                "Only propose amendments you can back with tool-verified evidence."
            )

        result = await call_llm(
            prompt=prompt + tool_hint,
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            expect_json=True,
            temperature=0.1,
            task_label="Self-Reflection: Consistency Check",
            on_progress=on_progress,
            think=True,
            tools=tools,  # Now can query KB and search documents
        )

        amendments = result.get("amendments", [])
        notes = result.get("reflection_notes", "")

        logger.info(f"Self-reflection: {len(amendments)} amendment(s), notes: {notes[:100]}")
        return {"amendments": amendments, "reflection_notes": notes}

    except Exception as e:
        logger.error(f"Self-reflection failed: {e}")
        return {"amendments": [], "reflection_notes": f"Reflection failed: {e}"}


def apply_amendments(all_checks: list[dict], amendments: list[dict]) -> int:
    """Apply self-reflection amendments to the check results in place.

    Only applies status changes — does not change severity, evidence, or explanation.
    Adds a 'reflected' flag and appends the amendment reason to the explanation.

    Returns:
        Number of amendments applied
    """
    valid_statuses = {"PASS", "FAIL", "WARNING", "NOT_APPLICABLE", "INFO"}
    applied = 0

    # Build lookup: rule_code → check
    checks_by_code: dict[str, dict] = {}
    for check in all_checks:
        code = check.get("rule_code", "")
        checks_by_code[code] = check  # Last one wins if duplicates

    for amendment in amendments:
        code = amendment.get("rule_code", "")
        new_status = amendment.get("amended_status", "").upper()
        reason = amendment.get("reason", "")
        issue_type = amendment.get("issue", "")

        if code not in checks_by_code:
            logger.warning(f"Self-reflection: amendment for unknown rule_code '{code}'")
            continue

        if new_status not in valid_statuses:
            logger.warning(f"Self-reflection: invalid amended_status '{new_status}' for {code}")
            continue

        check = checks_by_code[code]
        old_status = check.get("status", "")

        # Only apply if status actually changes
        if old_status == new_status:
            continue

        # Safety: never downgrade FAIL to PASS (could mask real issues)
        if old_status == "FAIL" and new_status == "PASS":
            logger.warning(
                f"Self-reflection: blocked FAIL→PASS downgrade for {code}: {reason}"
            )
            check.setdefault("reflection_blocked", []).append(
                f"Reflection attempted FAIL→PASS: {reason}"
            )
            continue

        check["status"] = new_status
        check["reflected"] = True
        check["reflection_reason"] = (
            f"[{issue_type}] Status changed from {old_status} to {new_status}: {reason}"
        )
        # Append to explanation
        check["explanation"] = (
            check.get("explanation", "") +
            f"\n\n[Self-Reflection Amendment] {reason}"
        )

        logger.info(f"Self-reflection: {code} {old_status} → {new_status}")
        applied += 1

    return applied
