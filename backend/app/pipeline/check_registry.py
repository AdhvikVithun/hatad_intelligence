"""Check registry — single source of truth for all verification checks.

Each check defines:
  - rule_code / rule_name / severity: identity & classification
  - group_id: which verification group owns it
  - requires: list of OR-groups; each inner list is AND.
              E.g. [["EC","SALE_DEED"]] = EC AND SALE_DEED
                   [["EC"],["PATTA"]]   = EC OR PATTA
  - enriched_by: optional doc types that improve the check (never gate it)
  - prompt_section: text injected into the dynamic prompt when this check runs

The orchestrator uses ``partition_checks()`` to split checks into
**runnable** (sent to LLM) and **na_stubs** (pre-generated NOT_APPLICABLE)
based on which document types are actually uploaded.
"""

from __future__ import annotations
from typing import Any

# ───────────────────────────────────────────────────────
# Check definitions — grouped by verification group
# ───────────────────────────────────────────────────────

GROUP1_CHECKS: list[dict[str, Any]] = [
    {
        "rule_code": "ACTIVE_MORTGAGE",
        "rule_name": "Active Mortgage Check",
        "severity": "CRITICAL",
        "group_id": 1,
        "requires": [["EC"]],
        "enriched_by": ["RELEASE_DEED"],
        "prompt_section": (
            "Are there mortgage entries WITHOUT corresponding release/discharge entries? "
            "An active mortgage means the property is pledged to a lender. "
            "Look for: mortgage, hypothecation, equitable mortgage, simple mortgage. "
            "Then check for: release, discharge, satisfaction. "
            "If mortgage exists without release = FAIL."
        ),
    },
    {
        "rule_code": "MULTIPLE_SALES",
        "rule_name": "Multiple Sales Check",
        "severity": "CRITICAL",
        "group_id": 1,
        "requires": [["EC"]],
        "enriched_by": [],
        "prompt_section": (
            "Has the same property (same survey number) been sold to MULTIPLE DIFFERENT "
            "buyers? This is outright fraud. Look for multiple sale deeds to different "
            "people for the same survey number."
        ),
    },
    {
        "rule_code": "LIS_PENDENS",
        "rule_name": "Lis Pendens / Litigation Check",
        "severity": "HIGH",
        "group_id": 1,
        "requires": [["EC"]],
        "enriched_by": ["COURT_ORDER"],
        "prompt_section": (
            "Are there any court attachments, pending litigation, or lis pendens entries? "
            "Look for: attachment, court order, lis pendens, injunction, stay order, suit, "
            "decree. Any such entry = WARNING or FAIL depending on whether it's been vacated."
        ),
    },
    {
        "rule_code": "MULTIPLE_PATTA",
        "rule_name": "Multiple Patta Reference Check",
        "severity": "MEDIUM",
        "group_id": 1,
        "requires": [["EC"]],
        "enriched_by": [],
        "prompt_section": (
            "Are there indications of multiple/duplicate Patta references "
            "for the same survey number?"
        ),
    },
    {
        "rule_code": "SURVEY_SUBDIVISION",
        "rule_name": "Survey Subdivision Check",
        "severity": "MEDIUM",
        "group_id": 1,
        "requires": [["EC"]],
        "enriched_by": [],
        "prompt_section": (
            "Has the survey number been recently subdivided? Look for subdivision entries. "
            "Note the original survey number and new subdivided numbers."
        ),
    },
]

GROUP2_CHECKS: list[dict[str, Any]] = [
    {
        "rule_code": "POA_SALE",
        "rule_name": "Power of Attorney Sale Check",
        "severity": "HIGH",
        "group_id": 2,
        "requires": [["SALE_DEED"]],
        "enriched_by": ["POA"],
        "prompt_section": (
            "Is the property being sold through Power of Attorney? Check if the seller "
            "is an attorney holder or if POA is mentioned. POA sales carry higher risk. "
            "If a standalone POA document is provided, verify: (a) principal in POA matches "
            "seller in Sale Deed, (b) agent matches the person executing, (c) POA scope "
            "covers sale transactions, (d) POA is not revoked/expired."
        ),
    },
    {
        "rule_code": "LAYOUT_APPROVAL",
        "rule_name": "Layout Approval Check",
        "severity": "MEDIUM",
        "group_id": 2,
        "requires": [["SALE_DEED"]],
        "enriched_by": ["LAYOUT_APPROVAL"],
        "prompt_section": (
            "If this is a residential plot in a layout, is there valid CMDA/DTCP/local body "
            "layout approval referenced? If a Layout Approval document is provided, check "
            "approval number, validity period, and whether the property survey is within the "
            "approved layout."
        ),
    },
    {
        "rule_code": "BUILDING_VIOLATION",
        "rule_name": "Building Violation Check",
        "severity": "MEDIUM",
        "group_id": 2,
        "requires": [["SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "If built-up property, are there any building plan violations "
            "or unauthorized constructions mentioned?"
        ),
    },
    {
        "rule_code": "SELLER_CAPACITY",
        "rule_name": "Seller Legal Capacity Check",
        "severity": "MEDIUM",
        "group_id": 2,
        "requires": [["SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "Does the seller appear to have legal capacity to sell? Check for "
            "minor sellers, mental capacity issues, or unauthorized representatives."
        ),
    },
]

GROUP3_CHECKS: list[dict[str, Any]] = [
    {
        "rule_code": "PORAMBOKE_DETECTION",
        "rule_name": "Government / Poramboke Land Detection",
        "severity": "CRITICAL",
        "group_id": 3,
        "requires": [["EC"], ["PATTA"], ["SALE_DEED"]],       # any one anchor
        "enriched_by": ["ADANGAL", "FMB"],
        "prompt_section": (
            "Is this government/poramboke land? Search ALL documents for: poramboke, "
            "government, temple, wakf, trust, inam, assessed waste, waterway, roadway. "
            "These lands CANNOT be privately sold. If Adangal is provided, check its "
            "classification field directly."
        ),
    },
    {
        "rule_code": "OWNER_NAME_MISMATCH",
        "rule_name": "Owner Name Cross-Check",
        "severity": "CRITICAL",
        "group_id": 3,
        # Needs ≥2 doc types to cross-check — coded as OR of 2-type AND pairs
        "requires": [
            ["EC", "PATTA"], ["EC", "SALE_DEED"], ["EC", "CHITTA"],
            ["EC", "A_REGISTER"], ["PATTA", "SALE_DEED"],
            ["CHITTA", "SALE_DEED"], ["A_REGISTER", "SALE_DEED"],
        ],
        "enriched_by": [],
        "prompt_section": (
            "Do owner/seller names match across documents? Account for common Tamil "
            "name variations (Murugan vs Muruganandam, S/o vs Son of, initial expansions) "
            "but flag genuine mismatches. Check both seller and buyer names appear "
            "consistently."
        ),
    },
    {
        "rule_code": "SRO_JURISDICTION",
        "rule_name": "SRO Jurisdiction Check",
        "severity": "CRITICAL",
        "group_id": 3,
        "requires": [["EC", "SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "Does the Sub-Registrar Office match the correct jurisdiction for the "
            "village/taluk? Check SRO mentioned in Sale Deed matches the one in EC entries.\n"
            "IMPORTANT — SRO names and Taluk names are DIFFERENT levels of the "
            "administrative tree. Do NOT flag SRO name vs taluk name differences as "
            "mismatches — only flag if the SRO is clearly in a DIFFERENT geographic region.\n"
            "RULE: If the pre-computed SRO JURISDICTION CHECK shows jurisdiction_valid=True, "
            "you MUST mark SRO_JURISDICTION as PASS."
        ),
    },
    {
        "rule_code": "BOUNDARY_CONSISTENCY",
        "rule_name": "Boundary Consistency Check",
        "severity": "MEDIUM",
        "group_id": 3,
        "requires": [["SALE_DEED"]],
        "enriched_by": ["FMB", "PATTA"],
        "prompt_section": (
            "Do property boundaries in Sale Deed match those in Patta/FMB? Compare "
            "North/South/East/West boundaries. If FMB data is provided, compare FMB "
            "boundary holders (adjacent survey owners) with Sale Deed boundary references. "
            "FMB dimensions can validate whether the extent claimed is physically consistent."
        ),
    },
    {
        "rule_code": "PATTA_CURRENT",
        "rule_name": "Patta Current Owner Check",
        "severity": "MEDIUM",
        "group_id": 3,
        "requires": [
            ["PATTA", "EC"], ["PATTA", "SALE_DEED"],
            ["CHITTA", "EC"], ["CHITTA", "SALE_DEED"],
            ["A_REGISTER", "EC"], ["A_REGISTER", "SALE_DEED"],
        ],
        "enriched_by": [],
        "prompt_section": (
            "Is the Patta in the current seller's name? If not, mutation may be pending.\n"
            "IMPORTANT — Patta mutation delays are EXTREMELY COMMON in Tamil Nadu. "
            "If the Patta owner does not match the current EC buyer, this is a "
            "PENDING MUTATION issue (WARNING), not a title defect. Only flag as FAIL "
            "if there is an UNRELATED third party in the Patta."
        ),
    },
    {
        "rule_code": "ACCESS_ROAD",
        "rule_name": "Access Road Check",
        "severity": "MEDIUM",
        "group_id": 3,
        "requires": [["SALE_DEED"], ["FMB"]],
        "enriched_by": [],
        "prompt_section": (
            "Is there described road access to the property? Check for 'road' or 'way' "
            "in boundaries (Sale Deed or FMB)."
        ),
    },
]

GROUP4_CHECKS: list[dict[str, Any]] = [
    {
        "rule_code": "RESTRICTED_LAND",
        "rule_name": "Restricted Land Check",
        "severity": "HIGH",
        "group_id": 4,
        "requires": [["EC"], ["PATTA"], ["SALE_DEED"]],
        "enriched_by": ["ADANGAL"],
        "prompt_section": (
            "Is this trust/Wakf/temple/SC-ST restricted land? Check for trust deeds, "
            "Wakf registration, temple land references, SC/ST land grant conditions. "
            "These restrictions may prohibit private sale."
        ),
    },
    {
        "rule_code": "CEILING_SURPLUS",
        "rule_name": "Land Ceiling Check",
        "severity": "HIGH",
        "group_id": 4,
        "requires": [["PATTA"], ["CHITTA"], ["A_REGISTER"]],
        "enriched_by": ["ADANGAL"],
        "prompt_section": (
            "Does the land holding exceed Tamil Nadu ceiling limits "
            "(15 acres wet / 30 acres dry)? Check Patta/Chitta total extent and "
            "land classification."
        ),
    },
    {
        "rule_code": "CONVERSION_STATUS",
        "rule_name": "Land Use Conversion Check",
        "severity": "HIGH",
        "group_id": 4,
        "requires": [["PATTA"], ["SALE_DEED"]],
        "enriched_by": ["ADANGAL", "LAYOUT_APPROVAL"],
        "prompt_section": (
            "If the property is agricultural land being used for non-agricultural "
            "purposes, has NA (Non-Agricultural) conversion been obtained? If Adangal "
            "shows active cultivation but Sale Deed says Residential/Commercial, "
            "flag missing NA conversion. If Layout Approval is provided, check its validity."
        ),
    },
    {
        "rule_code": "TAX_ARREARS",
        "rule_name": "Tax Arrears Check",
        "severity": "MEDIUM",
        "group_id": 4,
        "requires": [["PATTA"], ["CHITTA"], ["A_REGISTER"]],
        "enriched_by": [],
        "prompt_section": (
            "Are there indications of unpaid property tax from Patta/Chitta/A-Register "
            "tax details fields?"
        ),
    },
    {
        "rule_code": "ENCROACHMENT_RISK",
        "rule_name": "Encroachment Risk Check",
        "severity": "MEDIUM",
        "group_id": 4,
        "requires": [["EC"], ["PATTA"], ["SALE_DEED"]],
        "enriched_by": ["FMB", "ADANGAL"],
        "prompt_section": (
            "Is there risk of encroachment on government land or waterbody? Check for "
            "references to adjacent government lands, water bodies, or roads in boundaries."
        ),
    },
    {
        "rule_code": "AGRICULTURAL_ZONE",
        "rule_name": "Agricultural Zone Check",
        "severity": "MEDIUM",
        "group_id": 4,
        "requires": [["PATTA"], ["SALE_DEED"]],
        "enriched_by": ["ADANGAL", "LAYOUT_APPROVAL"],
        "prompt_section": (
            "Is the property in an agricultural zone with restrictions on non-agricultural "
            "use? Check land classification in Patta and Adangal."
        ),
    },
]

GROUP5_CHECKS: list[dict[str, Any]] = [
    {
        "rule_code": "BROKEN_CHAIN_OF_TITLE",
        "rule_name": "Chain of Title Continuity",
        "severity": "CRITICAL",
        "group_id": 5,
        "requires": [["EC", "SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "Is the ownership chain unbroken from the earliest EC entry to the current "
            "seller? Trace each transfer: the buyer in one transaction must be the "
            "seller in the next. Account for Tamil name variations."
        ),
    },
    {
        "rule_code": "FAMILY_PARTITION_DODGE",
        "rule_name": "Suspicious Family Partition Check",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["EC", "SALE_DEED"]],
        "enriched_by": ["PARTITION_DEED"],
        "prompt_section": (
            "Are there suspicious family partitions shortly before the sale? Look for "
            "partition deeds registered within 6 months before the sale deed date. "
            "If a Partition Deed document is provided, verify all joint owners match "
            "prior EC chain parties and shares are mathematically consistent."
        ),
    },
    {
        "rule_code": "JOINT_OWNERSHIP_CONSENT",
        "rule_name": "Joint Ownership Consent Check",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["SALE_DEED"]],
        "enriched_by": ["PATTA"],
        "prompt_section": (
            "If there are multiple co-owners (from Patta or EC), have ALL co-owners "
            "signed the Sale Deed? Missing co-owner consent makes the sale voidable."
        ),
    },
    {
        "rule_code": "UNREGISTERED_GAP",
        "rule_name": "Unregistered Transfer Gap Check",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["EC", "SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "Are there gaps in the registered chain suggesting unregistered intermediate "
            "transfers? If the EC chain jumps from A→C without A→B→C, there may be "
            "an unregistered transfer."
        ),
    },
    {
        "rule_code": "DEATH_WITHOUT_SUCCESSION",
        "rule_name": "Death Without Succession Certificate",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["EC"]],
        "enriched_by": ["LEGAL_HEIR", "WILL"],
        "prompt_section": (
            "Do any EC entries show inheritance/will/succession transfers without a "
            "corresponding Legal Heir Certificate or probate order? If a Legal Heir "
            "Certificate is provided, verify: (a) the deceased matches a prior owner "
            "in the EC chain, (b) the current seller is listed as an heir, (c) heir "
            "shares are consistent with claimed ownership. If a Will is provided, "
            "check probate status — Tamil Nadu requires probate for registered wills "
            "covering immovable property."
        ),
    },
    {
        "rule_code": "REVENUE_DISPUTE",
        "rule_name": "Revenue Court Dispute Check",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["EC"]],
        "enriched_by": ["COURT_ORDER"],
        "prompt_section": (
            "Are there revenue court proceedings or disputes in the EC? Look for "
            "revenue court case numbers, RDO references, or tahsildar order entries. "
            "If a Court Order document is provided, check its status — is it "
            "final/vacated or still pending?"
        ),
    },
    {
        "rule_code": "GOVERNMENT_ACQUISITION",
        "rule_name": "Government Acquisition Check",
        "severity": "HIGH",
        "group_id": 5,
        "requires": [["EC"]],
        "enriched_by": ["COURT_ORDER"],
        "prompt_section": (
            "Are there Land Acquisition Act proceedings affecting this property? "
            "Look for LA Act notifications, Section 4/6 proceedings, acquisition "
            "awards, or compensation entries in the EC."
        ),
    },
    {
        "rule_code": "PRICE_ANOMALY",
        "rule_name": "Transaction Price Anomaly Check",
        "severity": "MEDIUM",
        "group_id": 5,
        "requires": [["EC", "SALE_DEED"]],
        "enriched_by": [],
        "prompt_section": (
            "Is the sale price significantly above or below (>30%) comparable "
            "transactions in the EC? Compare consideration amounts across sale "
            "entries for the same survey."
        ),
    },
    {
        "rule_code": "TENANT_OCCUPANCY",
        "rule_name": "Tenant Occupancy Check",
        "severity": "MEDIUM",
        "group_id": 5,
        "requires": [["EC"]],
        "enriched_by": ["ADANGAL"],
        "prompt_section": (
            "Are there active lease entries in the EC without a corresponding "
            "surrender/cancellation? If Adangal is provided, check if a tenant "
            "name is listed."
        ),
    },
    {
        "rule_code": "POWER_LINE_EASEMENT",
        "rule_name": "Power Line / Easement Check",
        "severity": "MEDIUM",
        "group_id": 5,
        "requires": [["EC"]],
        "enriched_by": ["FMB"],
        "prompt_section": (
            "Are there electricity/power line easements or other utility easements "
            "registered in the EC? If FMB is provided, check for easement markers "
            "in the sketch notes."
        ),
    },
]

# ───────────────────────────────────────────────────────
# All checks indexed by group
# ───────────────────────────────────────────────────────

ALL_CHECK_DEFS: dict[int, list[dict[str, Any]]] = {
    1: GROUP1_CHECKS,
    2: GROUP2_CHECKS,
    3: GROUP3_CHECKS,
    4: GROUP4_CHECKS,
    5: GROUP5_CHECKS,
}

# Flat list for quick lookup
ALL_CHECKS_FLAT: list[dict[str, Any]] = [
    c for checks in ALL_CHECK_DEFS.values() for c in checks
]

# rule_code → check def
CHECK_BY_CODE: dict[str, dict[str, Any]] = {
    c["rule_code"]: c for c in ALL_CHECKS_FLAT
}


# ───────────────────────────────────────────────────────
# Partition logic
# ───────────────────────────────────────────────────────

def _requires_satisfied(requires: list[list[str]], available: set[str]) -> bool:
    """Check if at least one AND-group in ``requires`` is fully satisfied.

    ``requires`` is a list of OR-groups: [[A, B], [C]] means (A AND B) OR (C).
    Returns True if any inner group is a subset of *available*.
    """
    for and_group in requires:
        if all(dt in available for dt in and_group):
            return True
    return False


def _missing_types_message(requires: list[list[str]], available: set[str]) -> str:
    """Human-readable explanation of which doc types are missing.

    Picks the AND-group closest to being satisfied and reports its gaps.
    """
    best_group = None
    best_missing: list[str] = []
    fewest_missing = 999

    for and_group in requires:
        missing = [dt for dt in and_group if dt not in available]
        if len(missing) < fewest_missing:
            fewest_missing = len(missing)
            best_group = and_group
            best_missing = missing

    if not best_missing:
        return ""

    type_names = {
        "EC": "Encumbrance Certificate",
        "SALE_DEED": "Sale Deed",
        "PATTA": "Patta",
        "CHITTA": "Chitta",
        "A_REGISTER": "A-Register",
        "FMB": "Field Measurement Book",
        "ADANGAL": "Adangal",
        "LAYOUT_APPROVAL": "Layout Approval",
        "LEGAL_HEIR": "Legal Heir Certificate",
        "POA": "Power of Attorney",
        "COURT_ORDER": "Court Order",
        "WILL": "Will / Testament",
        "PARTITION_DEED": "Partition Deed",
        "GIFT_DEED": "Gift Deed",
        "RELEASE_DEED": "Release Deed",
    }

    readable = [type_names.get(t, t) for t in best_missing]
    if len(readable) == 1:
        return f"{readable[0]} not provided."
    return f"{', '.join(readable[:-1])} and {readable[-1]} not provided."


def partition_checks(
    group_id: int,
    available_types: set[str],
) -> tuple[list[dict], list[dict]]:
    """Split a group's checks into runnable and N/A stubs.

    Args:
        group_id: Verification group (1-5).
        available_types: Set of document types actually uploaded.

    Returns:
        (runnable, na_stubs) — runnable checks go to LLM, na_stubs are
        pre-generated NOT_APPLICABLE check results.
    """
    check_defs = ALL_CHECK_DEFS.get(group_id, [])
    runnable: list[dict] = []
    na_stubs: list[dict] = []

    for check in check_defs:
        if _requires_satisfied(check["requires"], available_types):
            runnable.append(check)
        else:
            msg = _missing_types_message(check["requires"], available_types)
            na_stubs.append({
                "rule_code": check["rule_code"],
                "rule_name": check["rule_name"],
                "severity": check["severity"],
                "status": "NOT_APPLICABLE",
                "explanation": (
                    f"This check could not be performed. {msg}"
                ),
                "recommendation": (
                    f"Provide the required document(s) for this check to be evaluated."
                ),
                "evidence": f"Required documents not in session: {msg}",
            })

    return runnable, na_stubs


def build_check_roster(
    runnable_checks: list[dict],
    available_types: set[str],
) -> str:
    """Build the dynamic '═══ CHECKS TO PERFORM ═══' section for the LLM prompt.

    For each runnable check, emits:
      N. RULE_CODE (SEVERITY): prompt_section
      [Enrichment note if relevant doc is available]
    """
    lines = ["═══ CHECKS TO PERFORM ═══", ""]

    for i, check in enumerate(runnable_checks, 1):
        lines.append(
            f"{i}. {check['rule_code']} ({check['severity']}): "
            f"{check['prompt_section']}"
        )

        # Add enrichment note if any enriched_by doc is actually present
        enrichments = [dt for dt in check.get("enriched_by", []) if dt in available_types]
        if enrichments:
            type_names = {
                "RELEASE_DEED": "Release Deed", "COURT_ORDER": "Court Order",
                "POA": "Power of Attorney", "LAYOUT_APPROVAL": "Layout Approval",
                "FMB": "Field Measurement Book", "ADANGAL": "Adangal",
                "LEGAL_HEIR": "Legal Heir Certificate", "WILL": "Will / Testament",
                "PARTITION_DEED": "Partition Deed", "GIFT_DEED": "Gift Deed",
                "PATTA": "Patta", "CHITTA": "Chitta", "A_REGISTER": "A-Register",
            }
            names = [type_names.get(dt, dt) for dt in enrichments]
            lines.append(
                f"   NOTE: {', '.join(names)} data is available — use it to "
                f"strengthen this analysis."
            )

        lines.append("")

    return "\n".join(lines)
