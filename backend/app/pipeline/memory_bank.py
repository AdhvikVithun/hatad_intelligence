"""Memory Bank - cross-document fact store for due diligence verification.

Extracts key facts from each document and enables cross-referencing:
  - Property identifiers (survey numbers, extent, boundaries)
  - Party names (buyers, sellers, owners across documents)
  - Financial data (consideration, guideline value, stamp duty)
  - Dates and timelines (registration dates, EC periods, ownership changes)
  - Encumbrances and restrictions
  - Document references (cross-linking between documents)

Facts are categorised, deduplicated, and checked for consistency
across all uploaded documents for a single property.
"""

import json
import logging
from datetime import datetime
from typing import Any

from app.pipeline.utils import (
    parse_amount,
    normalize_survey_number,
    survey_numbers_match,
    split_survey_numbers,
    any_survey_match,
    normalize_village_name,
    village_names_match,
    normalize_name,
    split_party_names,
    name_similarity,
    names_have_overlap,
    parse_area_to_sqft,
    TITLE_TRANSFER_TYPES,
    CHAIN_RELEVANT_TYPES,
)

logger = logging.getLogger(__name__)


# Fact categories
FACT_CATEGORIES = [
    "property",       # Survey number, extent, location, boundaries
    "party",          # Names of buyers, sellers, owners, witnesses
    "financial",      # Amounts, stamp duty, guideline values
    "timeline",       # Dates, periods, sequences
    "encumbrance",    # Mortgages, liens, restrictions
    "chain",          # Ownership chain / title flow
    "reference",      # Cross-document references (doc numbers, SRO, etc.)
    "risk",           # Red flags, anomalies, concerns
]


class Fact:
    """A single verifiable fact extracted from a document."""

    def __init__(
        self,
        category: str,
        key: str,
        value: Any,
        source_file: str,
        source_type: str,
        confidence: float = 1.0,
        context: str = "",
    ):
        self.category = category
        self.key = key
        self.value = value
        self.source_file = source_file
        self.source_type = source_type
        self.confidence = confidence
        self.context = context
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    def __repr__(self):
        return f"Fact({self.category}/{self.key}={self.value} from {self.source_file})"


class Conflict:
    """A detected inconsistency between facts from different sources."""

    def __init__(
        self,
        category: str,
        key: str,
        facts: list[Fact],
        severity: str = "WARNING",
        description: str = "",
    ):
        self.category = category
        self.key = key
        self.facts = facts
        self.severity = severity
        self.description = description

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "key": self.key,
            "severity": self.severity,
            "description": self.description,
            "conflicting_values": [
                {
                    "value": f.value,
                    "source": f.source_file,
                    "type": f.source_type,
                }
                for f in self.facts
            ],
        }


class MemoryBank:
    """Cross-document fact store with conflict detection.

    Usage:
        bank = MemoryBank()
        bank.ingest_document("deed.pdf", "SALE_DEED", extracted_data)
        bank.ingest_document("ec.pdf", "EC", extracted_data)
        conflicts = bank.detect_conflicts()
        summary = bank.get_summary()
    """

    def __init__(self):
        self.facts: list[Fact] = []
        self.conflicts: list[Conflict] = []
        self._ingested_files: list[str] = []

    def ingest_document(self, filename: str, doc_type: str, data: dict) -> int:
        """Extract facts from a document's extracted data and store them.

        Returns the number of facts extracted.
        """
        if data is None:
            return 0

        before = len(self.facts)

        if doc_type == "EC":
            self._ingest_ec(filename, data)
        elif doc_type == "SALE_DEED":
            self._ingest_sale_deed(filename, data)
        elif doc_type in ("PATTA", "CHITTA"):
            self._ingest_patta(filename, data)
        else:
            self._ingest_generic(filename, doc_type, data)

        self._ingested_files.append(filename)
        added = len(self.facts) - before
        logger.info(f"MemoryBank: {added} facts from {filename} ({doc_type})")
        return added

    # ── EC ingestion ──

    def _ingest_ec(self, filename: str, data: dict):
        src = filename
        stype = "EC"

        self._add_fact("reference", "ec_number", data.get("ec_number"), src, stype)
        self._add_fact("property", "property_description", data.get("property_description"), src, stype)
        self._add_fact("timeline", "ec_period_from", data.get("period_from"), src, stype)
        self._add_fact("timeline", "ec_period_to", data.get("period_to"), src, stype)

        # Village / taluk from EC header (if available)
        village = data.get("village")
        if village and isinstance(village, str) and village.strip():
            self._add_fact("property", "village", village.strip(), src, stype)
        taluk = data.get("taluk")
        if taluk and isinstance(taluk, str) and taluk.strip():
            self._add_fact("property", "taluk", taluk.strip(), src, stype)

        transactions = data.get("transactions", [])
        self._add_fact("encumbrance", "total_ec_entries", len(transactions), src, stype)

        # Extract parties and ownership chain from transactions
        # Schema field mapping: seller_or_executant, buyer_or_claimant,
        #   transaction_type (not nature_of_document), date (not date_of_document),
        #   consideration_amount (not consideration_value)
        owners_seen = set()
        for txn in transactions:
            # Parties — schema uses seller_or_executant / buyer_or_claimant
            for schema_field, fact_label in [
                ("seller_or_executant", "ec_executant"),
                ("buyer_or_claimant", "ec_claimant"),
            ]:
                party = txn.get(schema_field)
                if party and isinstance(party, str) and party.strip():
                    # Split multi-party strings: "A and B" → ["A", "B"]
                    for individual in split_party_names(party):
                        if individual not in owners_seen:
                            owners_seen.add(individual)
                            self._add_fact(
                                "party", fact_label,
                                individual, src, stype,
                                context=f"Transaction #{txn.get('row_number', '?')}: {txn.get('transaction_type', '')}"
                            )

            # Document references within EC
            doc_num = txn.get("document_number")
            if doc_num:
                self._add_fact(
                    "reference", "ec_doc_number",
                    doc_num, src, stype,
                    context=f"{txn.get('transaction_type', '')} dated {txn.get('date', '?')}"
                )

            # Financial data from EC transactions — schema uses consideration_amount
            consideration = txn.get("consideration_amount")
            if consideration:
                self._add_fact(
                    "financial", "ec_consideration",
                    consideration, src, stype,
                    context=f"Doc #{doc_num or '?'}: {txn.get('transaction_type', '')}"
                )

            # Encumbrance types — schema uses transaction_type
            nature = txn.get("transaction_type")
            if nature and isinstance(nature, str) and nature.strip():
                self._add_fact(
                    "encumbrance", "ec_transaction_type",
                    nature.strip(), src, stype,
                    context=f"#{txn.get('row_number', '?')} dated {txn.get('date', '?')}"
                )

        # Build ownership chain summary
        if transactions:
            chain = []
            for txn in transactions:
                ttype = (txn.get("transaction_type") or "").lower()
                if ttype in CHAIN_RELEVANT_TYPES:
                    chain.append({
                        "from": txn.get("seller_or_executant", "?"),
                        "to": txn.get("buyer_or_claimant", "?"),
                        "type": txn.get("transaction_type"),
                        "date": txn.get("date"),
                        "doc_no": txn.get("document_number"),
                    })
            if chain:
                self._add_fact("chain", "ownership_transfers", chain, src, stype)

    # ── Sale Deed ingestion ──

    def _ingest_sale_deed(self, filename: str, data: dict):
        src = filename
        stype = "SALE_DEED"

        self._add_fact("reference", "sale_deed_number", data.get("document_number"), src, stype)
        self._add_fact("reference", "sro", data.get("sro"), src, stype)
        self._add_fact("timeline", "registration_date", data.get("registration_date"), src, stype)

        # Parties — schema: seller/buyer are arrays of {name, father_name, age, address}
        seller = data.get("seller")
        buyer = data.get("buyer")
        if seller:
            if isinstance(seller, list):
                for s in seller:
                    name = s.get("name", s) if isinstance(s, dict) else str(s)
                    self._add_fact("party", "seller", name, src, stype)
            else:
                self._add_fact("party", "seller", str(seller), src, stype)
        if buyer:
            if isinstance(buyer, list):
                for b in buyer:
                    name = b.get("name", b) if isinstance(b, dict) else str(b)
                    self._add_fact("party", "buyer", name, src, stype)
            else:
                self._add_fact("party", "buyer", str(buyer), src, stype)

        # Property — schema nests under data["property"]
        prop = data.get("property", {})
        if isinstance(prop, dict):
            self._add_fact("property", "survey_number", prop.get("survey_number"), src, stype)
            self._add_fact("property", "extent", prop.get("extent"), src, stype)
            self._add_fact("property", "boundaries", prop.get("boundaries"), src, stype)
            self._add_fact("property", "village", prop.get("village"), src, stype)
            self._add_fact("property", "taluk", prop.get("taluk"), src, stype)
            self._add_fact("property", "district", prop.get("district"), src, stype)
            self._add_fact("property", "property_type", prop.get("property_type"), src, stype)

        # Financial — schema nests under data["financials"]
        fin = data.get("financials", {})
        if isinstance(fin, dict):
            self._add_fact("financial", "consideration_amount", fin.get("consideration_amount"), src, stype)
            self._add_fact("financial", "guideline_value", fin.get("guideline_value"), src, stype)
            self._add_fact("financial", "stamp_duty", fin.get("stamp_duty"), src, stype)
            self._add_fact("financial", "registration_fee", fin.get("registration_fee"), src, stype)

        # Previous ownership
        prev = data.get("previous_ownership")
        if prev:
            self._add_fact("chain", "previous_ownership", prev, src, stype)

    # ── Patta/Chitta ingestion ──

    def _ingest_patta(self, filename: str, data: dict):
        src = filename
        stype = "PATTA"

        self._add_fact("reference", "patta_number", data.get("patta_number"), src, stype)

        # Survey numbers — schema: array of {survey_no, extent, classification}
        survey_numbers = data.get("survey_numbers", [])
        if isinstance(survey_numbers, list) and survey_numbers:
            survey_nos = [s.get("survey_no", "") for s in survey_numbers if isinstance(s, dict)]
            self._add_fact("property", "survey_number", ", ".join(filter(None, survey_nos)), src, stype)
            # Also add individual survey details
            for sn in survey_numbers:
                if isinstance(sn, dict) and sn.get("survey_no"):
                    self._add_fact("property", f"survey_{sn['survey_no']}_extent",
                                   sn.get("extent"), src, stype)
                    self._add_fact("property", f"survey_{sn['survey_no']}_class",
                                   sn.get("classification"), src, stype)
        else:
            # Fallback for non-schema data
            self._add_fact("property", "survey_number", data.get("survey_number"), src, stype)

        # Extent — schema uses total_extent
        self._add_fact("property", "extent", data.get("total_extent") or data.get("extent") or data.get("area"), src, stype)
        self._add_fact("property", "village", data.get("village"), src, stype)
        self._add_fact("property", "taluk", data.get("taluk"), src, stype)
        self._add_fact("property", "district", data.get("district"), src, stype)
        self._add_fact("property", "classification", data.get("land_classification") or data.get("classification"), src, stype)

        # Owners — schema: owner_names is array of {name, father_name, share}
        owners = data.get("owner_names", []) or data.get("owner") or data.get("pattadar")
        if owners:
            if isinstance(owners, list):
                for o in owners:
                    name = o.get("name", o) if isinstance(o, dict) else str(o)
                    self._add_fact("party", "patta_owner", name, src, stype,
                                   context=f"Share: {o.get('share', '?')}" if isinstance(o, dict) else None)
            else:
                self._add_fact("party", "patta_owner", str(owners), src, stype)

    # ── Generic ingestion ──

    def _ingest_generic(self, filename: str, doc_type: str, data: dict):
        src = filename
        stype = doc_type

        # Try to extract common fields
        for key_field in ("document_number", "registration_number", "case_number"):
            val = data.get(key_field)
            if val:
                self._add_fact("reference", key_field, val, src, stype)

        # Parties
        parties = data.get("key_parties", [])
        if isinstance(parties, list):
            for p in parties:
                if isinstance(p, dict):
                    self._add_fact("party", p.get("role", "party"), p.get("name", str(p)), src, stype)
                elif isinstance(p, str):
                    self._add_fact("party", "party", p, src, stype)

        # Property details
        prop = data.get("property_details", {})
        if isinstance(prop, dict):
            for k, v in prop.items():
                if v:
                    self._add_fact("property", k, v, src, stype)

        # Dates
        dates = data.get("key_dates", [])
        if isinstance(dates, list):
            for d in dates:
                if isinstance(d, dict):
                    self._add_fact("timeline", d.get("label", "date"), d.get("date", str(d)), src, stype)
                elif isinstance(d, str):
                    self._add_fact("timeline", "date", d, src, stype)

        # Amounts
        amounts_list = data.get("amounts", [])
        if isinstance(amounts_list, list):
            for a in amounts_list:
                if isinstance(a, dict):
                    self._add_fact("financial", a.get("label", "amount"), a.get("amount", str(a)), src, stype)
                elif a:
                    self._add_fact("financial", "amount", a, src, stype)

    # ── Core helpers ──

    def _add_fact(self, category: str, key: str, value: Any, src: str, stype: str,
                  confidence: float = 1.0, context: str = ""):
        """Add a fact if the value is non-empty."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        self.facts.append(Fact(category, key, value, src, stype, confidence, context))

    # ── Conflict detection ──

    def detect_conflicts(self) -> list[Conflict]:
        """Find inconsistencies between facts from different documents.

        Checks:
          - Same property field, different values across docs
          - Party name mismatches (seller in deed vs owner in patta)
          - Financial discrepancies
          - Timeline gaps or overlaps
        """
        self.conflicts.clear()

        # Group facts by (category, key)
        groups: dict[tuple[str, str], list[Fact]] = {}
        for f in self.facts:
            groups.setdefault((f.category, f.key), []).append(f)

        # Check property consistency
        self._check_property_conflicts(groups)

        # Check party consistency
        self._check_party_conflicts(groups)

        # Check financial consistency
        self._check_financial_conflicts(groups)

        return self.conflicts

    def _check_property_conflicts(self, groups: dict):
        """Check for property detail mismatches across documents.

        Uses fuzzy survey-number matching (hierarchy + OCR tolerance)
        instead of exact string equality.
        """
        # --- Survey number (fuzzy) ---
        survey_facts = groups.get(("property", "survey_number"), [])
        if len(survey_facts) >= 2:
            by_source: dict[str, list] = {}
            for f in survey_facts:
                by_source.setdefault(f.source_file, []).append(f)

            if len(by_source) >= 2:
                source_items = list(by_source.items())
                for i, (src_a, facts_a) in enumerate(source_items):
                    for src_b, facts_b in source_items[i + 1:]:
                        # Collect all survey numbers from each source
                        nums_a: list[str] = []
                        for f in facts_a:
                            nums_a.extend(split_survey_numbers(str(f.value)))
                        nums_b: list[str] = []
                        for f in facts_b:
                            nums_b.extend(split_survey_numbers(str(f.value)))

                        if nums_a and nums_b:
                            has_match, match_type, _, _ = any_survey_match(nums_a, nums_b)
                            if not has_match:
                                self.conflicts.append(Conflict(
                                    category="property",
                                    key="survey_number",
                                    facts=facts_a + facts_b,
                                    severity="HIGH",
                                    description=(
                                        f"Survey numbers differ across documents: "
                                        f"{src_a}={[f.value for f in facts_a]} vs "
                                        f"{src_b}={[f.value for f in facts_b]}"
                                    ),
                                ))
                            elif match_type == "ocr_fuzzy":
                                self.conflicts.append(Conflict(
                                    category="property",
                                    key="survey_number",
                                    facts=facts_a + facts_b,
                                    severity="WARNING",
                                    description=(
                                        f"Survey numbers match only by OCR-fuzzy tolerance "
                                        f"(possible typo): {src_a}={[f.value for f in facts_a]} "
                                        f"vs {src_b}={[f.value for f in facts_b]}"
                                    ),
                                ))

        # --- Extent (unit-aware) ---
        extent_facts = groups.get(("property", "extent"), [])
        if len(extent_facts) >= 2:
            by_source_ext: dict[str, "Fact"] = {}
            for f in extent_facts:
                by_source_ext.setdefault(f.source_file, f)
            if len(by_source_ext) >= 2:
                ext_facts_list = list(by_source_ext.values())
                # Try numeric comparison via parse_area_to_sqft
                sqft_values: list[tuple["Fact", float | None]] = []
                for f in ext_facts_list:
                    sqft_values.append((f, parse_area_to_sqft(str(f.value))))

                parsed = [(f, v) for f, v in sqft_values if v is not None and v > 0]
                if len(parsed) >= 2:
                    # All parseable — compare numerically with 10% tolerance
                    ref_fact, ref_sqft = parsed[0]
                    conflict_found = False
                    for other_fact, other_sqft in parsed[1:]:
                        diff_ratio = abs(ref_sqft - other_sqft) / max(ref_sqft, other_sqft)
                        if diff_ratio > 0.10:  # >10% difference
                            conflict_found = True
                            break
                    if conflict_found:
                        self.conflicts.append(Conflict(
                            category="property",
                            key="extent",
                            facts=ext_facts_list,
                            severity="WARNING",
                            description=(
                                f"Property extent differs across documents: "
                                f"{', '.join(f'{f.source_file}={f.value}' for f in ext_facts_list)}"
                            ),
                        ))
                else:
                    # Fallback to string comparison when units can't be parsed
                    values = {str(f.value).strip().lower() for f in ext_facts_list}
                    if len(values) > 1:
                        self.conflicts.append(Conflict(
                            category="property",
                            key="extent",
                            facts=ext_facts_list,
                            severity="WARNING",
                            description=(
                                f"Property extent differs across documents: "
                                f"{', '.join(f'{f.source_file}={f.value}' for f in ext_facts_list)}"
                            ),
                        ))

        # --- Village consistency (fuzzy) ---
        village_facts = groups.get(("property", "village"), [])
        if len(village_facts) >= 2:
            by_source_vill: dict[str, "Fact"] = {}
            for f in village_facts:
                by_source_vill.setdefault(f.source_file, f)
            if len(by_source_vill) >= 2:
                src_list = list(by_source_vill.items())
                for i, (src_a, fact_a) in enumerate(src_list):
                    for src_b, fact_b in src_list[i + 1:]:
                        matched, match_type = village_names_match(str(fact_a.value), str(fact_b.value))
                        if not matched:
                            # Downgrade to MEDIUM when one name is Tamil and the
                            # other is Latin — cross-script transliteration is
                            # imprecise and false alarms are common.
                            val_a, val_b = str(fact_a.value), str(fact_b.value)
                            a_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in val_a)
                            b_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in val_b)
                            cross_script = (a_tamil != b_tamil)
                            severity = "MEDIUM" if cross_script else "HIGH"
                            self.conflicts.append(Conflict(
                                category="property",
                                key="village",
                                facts=[fact_a, fact_b],
                                severity=severity,
                                description=(
                                    f"Village differs across documents: "
                                    f"{src_a}={fact_a.value} vs {src_b}={fact_b.value}"
                                    + (" (cross-script comparison — may be transliteration variant)" if cross_script else "")
                                ),
                            ))

    def _check_party_conflicts(self, groups: dict):
        """Check for party name mismatches between documents.

        Uses fuzzy cross-script name matching (Tamil↔English, initials
        stripped, relation-marker splitting) instead of exact intersection.
        """
        # Collect all seller/buyer/owner names by source
        party_groups = {}
        for (cat, key), facts in groups.items():
            if cat != "party":
                continue
            for f in facts:
                role = key.replace("ec_", "").replace("patta_", "")
                party_groups.setdefault(role, []).append(f)

        # Check: last buyer in EC should match patta owner
        buyers = party_groups.get("buyer", []) + party_groups.get("claimant", [])
        owners = party_groups.get("owner", [])

        if buyers and owners:
            buyer_names = [str(f.value) for f in buyers if f.value]
            owner_names = [str(f.value) for f in owners if f.value]
            if buyer_names and owner_names and not names_have_overlap(buyer_names, owner_names, threshold=0.55):
                # Format display names using normalize_name for readability
                display_buyers = ', '.join({normalize_name(n) for n in buyer_names} - {''})
                display_owners = ', '.join({normalize_name(n) for n in owner_names} - {''})
                self.conflicts.append(Conflict(
                    category="party",
                    key="buyer_vs_owner",
                    facts=buyers + owners,
                    severity="HIGH",
                    description=f"Buyer/claimant names in EC/Deed ({display_buyers}) do not match Patta owner ({display_owners})",
                ))

    def _check_financial_conflicts(self, groups: dict):
        """Check for financial discrepancies."""
        considerations = groups.get(("financial", "consideration_amount"), [])
        ec_considerations = groups.get(("financial", "ec_consideration"), [])

        if considerations and ec_considerations:
            deed_vals = set()
            ec_vals = set()
            for f in considerations:
                v = self._parse_amount(f.value)
                if v:
                    deed_vals.add(v)
            for f in ec_considerations:
                v = self._parse_amount(f.value)
                if v:
                    ec_vals.add(v)

            if deed_vals and ec_vals and not deed_vals.intersection(ec_vals):
                self.conflicts.append(Conflict(
                    category="financial",
                    key="consideration_mismatch",
                    facts=considerations + ec_considerations,
                    severity="WARNING",
                    description=f"Consideration amount in Sale Deed ({deed_vals}) differs from EC ({ec_vals})",
                ))

    @staticmethod
    def _normalize_name(name: Any) -> str:
        """Normalize a name for comparison (delegates to shared utils)."""
        return normalize_name(name)

    @staticmethod
    def _parse_amount(val: Any) -> float | None:
        """Parse a monetary amount (delegates to shared utils — lakh/crore aware)."""
        return parse_amount(val)

    # ── Query & summary ──

    def get_facts_by_category(self, category: str) -> list[dict]:
        """Get all facts in a category."""
        return [f.to_dict() for f in self.facts if f.category == category]

    def get_facts_by_key(self, key: str) -> list[dict]:
        """Get all facts matching a specific key (e.g. 'survey_number', 'seller_name').
        
        Used by _validate_group_result to cross-check LLM evidence against
        ground-truth facts extracted from documents.
        """
        key_lower = key.lower().replace(" ", "_").replace("-", "_")
        return [f.to_dict() for f in self.facts if f.key.lower().replace(" ", "_").replace("-", "_") == key_lower]

    def get_facts_by_source(self, filename: str) -> list[dict]:
        """Get all facts from a specific document."""
        return [f.to_dict() for f in self.facts if f.source_file == filename]

    def get_cross_references(self) -> list[dict]:
        """Find facts that appear in multiple documents (potential cross-verification points)."""
        groups: dict[tuple[str, str], list[Fact]] = {}
        for f in self.facts:
            groups.setdefault((f.category, f.key), []).append(f)

        cross_refs = []
        for (cat, key), facts in groups.items():
            sources = set(f.source_file for f in facts)
            if len(sources) > 1:
                cross_refs.append({
                    "category": cat,
                    "key": key,
                    "sources": list(sources),
                    "values": [{"value": f.value, "source": f.source_file, "type": f.source_type} for f in facts],
                    "consistent": len(set(str(f.value).strip().lower() for f in facts)) == 1,
                })

        return cross_refs

    def get_summary(self) -> dict:
        """Get a complete summary of the memory bank state."""
        cats = {}
        for f in self.facts:
            cats.setdefault(f.category, []).append(f.to_dict())

        return {
            "total_facts": len(self.facts),
            "ingested_files": self._ingested_files,
            "categories": {cat: len(facts) for cat, facts in cats.items()},
            "facts_by_category": cats,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "conflict_count": len(self.conflicts),
            "cross_references": self.get_cross_references(),
        }

    def to_dict(self) -> dict:
        """Serialize entire memory bank for session persistence."""
        return {
            "facts": [f.to_dict() for f in self.facts],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "ingested_files": self._ingested_files,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryBank":
        """Restore memory bank from serialized dict."""
        bank = cls()
        bank._ingested_files = data.get("ingested_files", [])
        for fd in data.get("facts", []):
            bank.facts.append(Fact(
                category=fd["category"],
                key=fd["key"],
                value=fd["value"],
                source_file=fd["source_file"],
                source_type=fd["source_type"],
                confidence=fd.get("confidence", 1.0),
                context=fd.get("context", ""),
            ))
        # Conflicts are re-detected, not restored
        return bank

    def get_verification_context(self) -> str:
        """Build a compact text summary for injection into verification prompts.

        This gives the LLM verifier access to cross-document facts
        so it can check consistency without re-parsing raw data.
        """
        lines = ["=== MEMORY BANK: Cross-Document Facts ==="]
        lines.append(f"Documents ingested: {', '.join(self._ingested_files)}")
        lines.append(f"Total facts: {len(self.facts)}")

        if self.conflicts:
            lines.append(f"\n--- CONFLICTS DETECTED ({len(self.conflicts)}) ---")
            for c in self.conflicts:
                lines.append(f"  [{c.severity}] {c.description}")

        cross_refs = self.get_cross_references()
        if cross_refs:
            lines.append(f"\n--- CROSS-REFERENCES ({len(cross_refs)}) ---")
            for cr in cross_refs:
                status = "CONSISTENT" if cr["consistent"] else "INCONSISTENT"
                lines.append(f"  {cr['category']}/{cr['key']}: {status} across {', '.join(cr['sources'])}")
                for v in cr["values"]:
                    val_str = str(v["value"])[:100]
                    lines.append(f"    {v['source']} ({v['type']}): {val_str}")

        # Key property facts
        prop_facts = self.get_facts_by_category("property")
        if prop_facts:
            lines.append("\n--- PROPERTY FACTS ---")
            for f in prop_facts:
                lines.append(f"  {f['key']}: {str(f['value'])[:120]} (from {f['source_file']})")

        # Key party facts
        party_facts = self.get_facts_by_category("party")
        if party_facts:
            lines.append("\n--- PARTIES ---")
            seen = set()
            for f in party_facts:
                sig = f"{f['key']}:{f['value']}"
                if sig not in seen:
                    seen.add(sig)
                    lines.append(f"  {f['key']}: {f['value']} (from {f['source_file']})")

        # Financial summary
        fin_facts = self.get_facts_by_category("financial")
        if fin_facts:
            lines.append("\n--- FINANCIAL ---")
            for f in fin_facts:
                lines.append(f"  {f['key']}: {f['value']} (from {f['source_file']})")

        return "\n".join(lines)
