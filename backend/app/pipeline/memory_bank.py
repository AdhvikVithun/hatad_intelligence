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
        transaction_id: str = "",
    ):
        self.category = category
        self.key = key
        self.value = value
        self.source_file = source_file
        self.source_type = source_type
        self.confidence = confidence
        self.context = context
        self.transaction_id = transaction_id
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        d = {
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "confidence": self.confidence,
            "context": self.context,
            "timestamp": self.timestamp,
        }
        if self.transaction_id:
            d["transaction_id"] = self.transaction_id
        return d

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
        elif doc_type == "PATTA":
            self._ingest_patta(filename, data)
        elif doc_type == "CHITTA":
            self._ingest_chitta(filename, data)
        elif doc_type == "A_REGISTER":
            self._ingest_a_register(filename, data)
        elif doc_type == "FMB":
            self._ingest_fmb(filename, data)
        elif doc_type == "ADANGAL":
            self._ingest_adangal(filename, data)
        elif doc_type == "LAYOUT_APPROVAL":
            self._ingest_layout_approval(filename, data)
        elif doc_type == "LEGAL_HEIR":
            self._ingest_legal_heir(filename, data)
        elif doc_type == "POA":
            self._ingest_poa(filename, data)
        elif doc_type == "COURT_ORDER":
            self._ingest_court_order(filename, data)
        elif doc_type == "WILL":
            self._ingest_will(filename, data)
        elif doc_type == "PARTITION_DEED":
            self._ingest_partition_deed(filename, data)
        elif doc_type == "GIFT_DEED":
            self._ingest_gift_deed(filename, data)
        elif doc_type == "RELEASE_DEED":
            self._ingest_release_deed(filename, data)
        else:
            self._ingest_generic(filename, doc_type, data)

        # Common: ingest _extraction_red_flags from any doc type
        # These are the model's extraction-time observations about
        # anomalies and concerns — store as risk facts
        red_flags = data.get("_extraction_red_flags")
        if red_flags and isinstance(red_flags, list):
            for flag in red_flags:
                if isinstance(flag, str) and flag.strip():
                    self._add_fact(
                        "risk", "extraction_red_flag",
                        flag.strip()[:400], filename, doc_type,
                        confidence=0.8,
                        context="Flagged by model during structured data extraction",
                    )

        # Common: ingest _field_confidence for low-confidence fields
        field_conf = data.get("_field_confidence")
        if field_conf and isinstance(field_conf, dict):
            low_conf_fields = [
                f"{k} (conf={v})" for k, v in field_conf.items()
                if isinstance(v, (int, float)) and v < 0.7
            ]
            if low_conf_fields:
                self._add_fact(
                    "risk", "low_confidence_fields",
                    f"Low extraction confidence on: {', '.join(low_conf_fields)}",
                    filename, doc_type,
                    confidence=0.9,
                    context="Model self-assessed confidence during extraction",
                )

        self._ingested_files.append(filename)
        added = len(self.facts) - before
        logger.info(f"MemoryBank: {added} facts from {filename} ({doc_type})")
        return added

    def ingest_risk_classification(self, filename: str, ec_data: dict) -> int:
        """Store transaction risk classification facts from EC analysis.

        Iterates EC transactions and stores a ``risk`` fact for each
        CRITICAL (encumbrance/judicial) transaction so verification can
        query high-risk items from the knowledge base.
        """
        from app.pipeline.utils import ENCUMBRANCE_TYPES, JUDICIAL_TYPES

        before = len(self.facts)
        for txn in (ec_data.get("transactions") or []):
            txn_type = (txn.get("transaction_type") or "").strip().lower()
            tid = txn.get("transaction_id", "")
            if txn_type in ENCUMBRANCE_TYPES or txn_type in JUDICIAL_TYPES:
                self._add_fact(
                    "risk", "high_risk_transaction",
                    f"{txn_type} (Doc {txn.get('document_number', '?')}, "
                    f"{txn.get('date', '?')})",
                    filename, "EC",
                    context=f"Row #{txn.get('row_number', '?')}: "
                            f"{txn.get('seller_or_executant', '?')} → "
                            f"{txn.get('buyer_or_claimant', '?')}",
                    transaction_id=tid,
                )
        added = len(self.facts) - before
        if added:
            logger.info(f"MemoryBank: {added} risk facts from {filename}")
        return added

    def ingest_risk_classification_sale_deed(self, filename: str, sd_data: dict) -> int:
        """Store risk classification facts from Sale Deed analysis.

        Flags high-risk patterns: undervaluation, GPA-based sale, missing
        encumbrance declaration, inadequate stamp duty, missing witnesses,
        incomplete ownership history, minor parties.
        """
        before = len(self.facts)
        fin = sd_data.get("financials", {}) or {}
        consideration = fin.get("consideration_amount", 0) or 0
        guideline = fin.get("guideline_value", 0) or 0
        stamp_duty = fin.get("stamp_duty", 0) or 0

        # Undervaluation check
        if guideline > 0 and consideration > 0 and consideration < guideline * 0.7:
            self._add_fact(
                "risk", "undervaluation",
                f"Consideration ₹{consideration:,} is below guideline ₹{guideline:,} "
                f"({consideration/guideline:.0%})",
                filename, "SALE_DEED",
            )

        # Stamp duty adequacy (TN: ~7% of max(consideration, guideline))
        if consideration > 0:
            expected_duty = max(consideration, guideline) * 0.07
            if stamp_duty > 0 and stamp_duty < expected_duty * 0.5:
                self._add_fact(
                    "risk", "inadequate_stamp_duty",
                    f"Stamp duty ₹{stamp_duty:,} appears low vs expected ~₹{int(expected_duty):,}",
                    filename, "SALE_DEED",
                )

        # GPA-based sale
        poa = sd_data.get("power_of_attorney", "")
        if poa and isinstance(poa, str) and len(poa.strip()) > 5:
            self._add_fact(
                "risk", "gpa_sale",
                f"Sale executed via Power of Attorney: {poa[:200]}",
                filename, "SALE_DEED",
            )

        # Missing encumbrance declaration
        enc = sd_data.get("encumbrance_declaration", "")
        if not enc or (isinstance(enc, str) and len(enc.strip()) < 10):
            self._add_fact(
                "risk", "missing_encumbrance_declaration",
                "No encumbrance declaration found in the Sale Deed",
                filename, "SALE_DEED",
            )

        # Missing or insufficient witnesses
        witnesses = sd_data.get("witnesses", [])
        if isinstance(witnesses, list) and len(witnesses) < 2:
            self._add_fact(
                "risk", "insufficient_witnesses",
                f"Only {len(witnesses)} witness(es) found — minimum 2 recommended",
                filename, "SALE_DEED",
            )

        # Minor party check
        for role in ("seller", "buyer"):
            parties = sd_data.get(role, [])
            if isinstance(parties, list):
                for p in parties:
                    if isinstance(p, dict):
                        age_str = p.get("age", "")
                        if age_str:
                            try:
                                age = int(str(age_str).strip().split()[0])
                                if 0 < age < 18:
                                    self._add_fact(
                                        "risk", "minor_party",
                                        f"{role.title()} '{p.get('name', '?')}' appears to be a minor (age {age})",
                                        filename, "SALE_DEED",
                                    )
                            except (ValueError, IndexError):
                                pass

        # Incomplete ownership history
        oh = sd_data.get("ownership_history", [])
        if isinstance(oh, list) and len(oh) == 0:
            self._add_fact(
                "risk", "no_ownership_history",
                "No ownership history extracted from deed recitals",
                filename, "SALE_DEED",
            )

        # Red flags from extraction
        red_flags = sd_data.get("_extraction_red_flags", [])
        if isinstance(red_flags, list):
            for flag in red_flags[:10]:
                if isinstance(flag, str) and flag.strip():
                    self._add_fact(
                        "risk", "extraction_red_flag",
                        flag.strip()[:300],
                        filename, "SALE_DEED",
                    )

        added = len(self.facts) - before
        if added:
            logger.info(f"MemoryBank: {added} Sale Deed risk facts from {filename}")
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
            tid = txn.get("transaction_id", "")
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
                                context=f"Transaction #{txn.get('row_number', '?')}: {txn.get('transaction_type', '')}",
                                transaction_id=tid,
                            )

            # Document references within EC
            doc_num = txn.get("document_number")
            if doc_num:
                self._add_fact(
                    "reference", "ec_doc_number",
                    doc_num, src, stype,
                    context=f"{txn.get('transaction_type', '')} dated {txn.get('date', '?')}",
                    transaction_id=tid,
                )

            # Financial data from EC transactions — schema uses consideration_amount
            consideration = txn.get("consideration_amount")
            if consideration:
                self._add_fact(
                    "financial", "ec_consideration",
                    consideration, src, stype,
                    context=f"Doc #{doc_num or '?'}: {txn.get('transaction_type', '')}",
                    transaction_id=tid,
                )

            # Encumbrance types — schema uses transaction_type
            nature = txn.get("transaction_type")
            if nature and isinstance(nature, str) and nature.strip():
                self._add_fact(
                    "encumbrance", "ec_transaction_type",
                    nature.strip(), src, stype,
                    context=f"#{txn.get('row_number', '?')} dated {txn.get('date', '?')}",
                    transaction_id=tid,
                )

            # Per-transaction survey numbers — critical for cross-document verification
            survey = txn.get("survey_number")
            if survey and isinstance(survey, str) and survey.strip():
                self._add_fact(
                    "property", "ec_survey_number",
                    survey.strip(), src, stype,
                    context=f"Transaction #{txn.get('row_number', '?')}: "
                            f"{txn.get('transaction_type', '')} - "
                            f"Doc {txn.get('document_number', '?')}",
                    transaction_id=tid,
                )

            # Per-transaction dates — enables temporal queries in verification
            txn_date = txn.get("date")
            if txn_date and isinstance(txn_date, str) and txn_date.strip():
                self._add_fact(
                    "timeline", "ec_transaction_date",
                    txn_date.strip(), src, stype,
                    context=f"#{txn.get('row_number', '?')}: "
                            f"{txn.get('transaction_type', '')} Doc {txn.get('document_number', '?')}",
                    transaction_id=tid,
                )

            # Suspicious flags — model's extraction-time red-flag analysis
            # Stored BEFORE compaction strips them (Tier 1 drops suspicious_flags)
            suspicious_flags = txn.get("suspicious_flags")
            if suspicious_flags:
                flags_list = suspicious_flags if isinstance(suspicious_flags, list) else [suspicious_flags]
                for flag in flags_list:
                    if flag and isinstance(flag, str) and flag.strip():
                        self._add_fact(
                            "risk", "ec_suspicious_flag",
                            flag.strip(), src, stype,
                            context=f"Transaction #{txn.get('row_number', '?')}: "
                                    f"{txn.get('transaction_type', '')} Doc {txn.get('document_number', '?')}",
                            transaction_id=tid,
                        )

            # Market value — government guideline reference from EC
            market_val = txn.get("market_value")
            if market_val:
                self._add_fact(
                    "financial", "ec_market_value",
                    market_val, src, stype,
                    context=f"Doc #{doc_num or '?'}: {txn.get('transaction_type', '')}",
                    transaction_id=tid,
                )

            # PR number — previous registration reference
            pr_num = txn.get("pr_number")
            if pr_num:
                self._add_fact(
                    "reference", "ec_pr_number",
                    pr_num, src, stype,
                    context=f"Doc #{doc_num or '?'}: {txn.get('transaction_type', '')}",
                    transaction_id=tid,
                )

            # Schedule remarks — property description details
            schedule_remarks = txn.get("schedule_remarks")
            if schedule_remarks and isinstance(schedule_remarks, str) and schedule_remarks.strip():
                self._add_fact(
                    "property", "ec_schedule_remarks",
                    schedule_remarks.strip()[:500], src, stype,
                    context=f"Transaction #{txn.get('row_number', '?')}",
                    transaction_id=tid,
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
                        "transaction_id": txn.get("transaction_id", ""),
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

        # Parties — schema: seller/buyer are arrays of {name, father_name, age, address, pan, aadhaar, share_percentage, identification_proof, relationship}
        seller = data.get("seller")
        buyer = data.get("buyer")
        if seller:
            if isinstance(seller, list):
                for s in seller:
                    if isinstance(s, dict):
                        name = s.get("name", "")
                        self._add_fact("party", "seller", name, src, stype)
                        if s.get("aadhaar"):
                            self._add_fact("party", "seller_aadhaar", s["aadhaar"], src, stype,
                                           context=f"Aadhaar of seller {name}")
                        if s.get("share_percentage"):
                            self._add_fact("party", "seller_share", s["share_percentage"], src, stype,
                                           context=f"Share of seller {name}")
                        if s.get("pan"):
                            self._add_fact("party", "seller_pan", s["pan"], src, stype,
                                           context=f"PAN of seller {name}")
                    else:
                        self._add_fact("party", "seller", str(s), src, stype)
            else:
                self._add_fact("party", "seller", str(seller), src, stype)
        if buyer:
            if isinstance(buyer, list):
                for b in buyer:
                    if isinstance(b, dict):
                        name = b.get("name", "")
                        self._add_fact("party", "buyer", name, src, stype)
                        if b.get("aadhaar"):
                            self._add_fact("party", "buyer_aadhaar", b["aadhaar"], src, stype,
                                           context=f"Aadhaar of buyer {name}")
                        if b.get("share_percentage"):
                            self._add_fact("party", "buyer_share", b["share_percentage"], src, stype,
                                           context=f"Share of buyer {name}")
                        if b.get("pan"):
                            self._add_fact("party", "buyer_pan", b["pan"], src, stype,
                                           context=f"PAN of buyer {name}")
                    else:
                        self._add_fact("party", "buyer", str(b), src, stype)
            else:
                self._add_fact("party", "buyer", str(buyer), src, stype)

        # Property — schema nests under data["property"]
        prop = data.get("property", {})
        if isinstance(prop, dict):
            self._add_fact("property", "survey_number", prop.get("survey_number"), src, stype)
            self._add_fact("property", "extent", prop.get("extent"), src, stype)
            self._add_fact("property", "boundaries", prop.get("boundaries"), src, stype)
            # Parse individual boundary directions for cross-doc comparison
            boundaries = prop.get("boundaries")
            if isinstance(boundaries, dict):
                for direction in ("north", "south", "east", "west"):
                    val = boundaries.get(direction)
                    if val:
                        self._add_fact("property", f"boundary_{direction}", val, src, stype)
            self._add_fact("property", "village", prop.get("village"), src, stype)
            self._add_fact("property", "taluk", prop.get("taluk"), src, stype)
            self._add_fact("property", "district", prop.get("district"), src, stype)
            self._add_fact("property", "property_type", prop.get("property_type"), src, stype)
            # New schema fields
            self._add_fact("property", "land_classification", prop.get("land_classification"), src, stype)
            self._add_fact("property", "plot_number", prop.get("plot_number"), src, stype)
            self._add_fact("property", "door_number", prop.get("door_number"), src, stype)
            self._add_fact("property", "measurement_reference", prop.get("measurement_reference"), src, stype)
            self._add_fact("property", "assessment_number", prop.get("assessment_number"), src, stype)

        # Financial — schema nests under data["financials"]
        fin = data.get("financials", {})
        if isinstance(fin, dict):
            self._add_fact("financial", "consideration_amount", fin.get("consideration_amount"), src, stype)
            self._add_fact("financial", "guideline_value", fin.get("guideline_value"), src, stype)
            self._add_fact("financial", "stamp_duty", fin.get("stamp_duty"), src, stype)
            self._add_fact("financial", "registration_fee", fin.get("registration_fee"), src, stype)

        # Witnesses — store as party facts for identity resolution
        # Schema: witnesses is now array of {name, father_name, age, address}
        witnesses = data.get("witnesses", [])
        if isinstance(witnesses, list):
            for w in witnesses:
                if isinstance(w, dict):
                    wname = w.get("name", "")
                    if wname and isinstance(wname, str) and wname.strip():
                        detail = wname.strip()
                        if w.get("father_name"):
                            detail += f" s/o {w['father_name']}"
                        self._add_fact("party", "witness", detail, src, stype)
                elif isinstance(w, str) and w.strip():
                    self._add_fact("party", "witness", w.strip(), src, stype)

        # Stamp paper details
        stamp_paper = data.get("stamp_paper")
        if isinstance(stamp_paper, dict):
            serials = stamp_paper.get("serial_numbers")
            if isinstance(serials, list):
                for sn in serials:
                    if sn and isinstance(sn, str) and sn.strip():
                        self._add_fact("reference", "stamp_serial", sn.strip(), src, stype)
            if stamp_paper.get("vendor_name"):
                self._add_fact("reference", "stamp_vendor", stamp_paper["vendor_name"], src, stype)
            if stamp_paper.get("total_stamp_value"):
                self._add_fact("financial", "total_stamp_value",
                               stamp_paper["total_stamp_value"], src, stype)

        # Registration details (book/volume/page)
        reg_details = data.get("registration_details")
        if isinstance(reg_details, dict):
            self._add_fact("reference", "book_number", reg_details.get("book_number"), src, stype)
            self._add_fact("reference", "volume_number", reg_details.get("volume_number"), src, stype)
            self._add_fact("reference", "page_number", reg_details.get("page_number"), src, stype)
            self._add_fact("reference", "certified_copy_number",
                           reg_details.get("certified_copy_number"), src, stype)
            self._add_fact("timeline", "certified_copy_date",
                           reg_details.get("certified_copy_date"), src, stype)

        # Previous ownership (single immediate predecessor — backward compat)
        prev = data.get("previous_ownership")
        if prev:
            self._add_fact("chain", "previous_ownership", prev, src, stype)

        # Full ownership history (story of the land — from deed recitals)
        ownership_history = data.get("ownership_history")
        if ownership_history and isinstance(ownership_history, list):
            # Store the complete chain as a single structured fact for queries
            self._add_fact("chain", "ownership_history", ownership_history, src, stype)

            # Also store individual chain links for fine-grained cross-referencing
            for i, link in enumerate(ownership_history):
                if not isinstance(link, dict):
                    continue
                owner = link.get("owner", "")
                acquired_from = link.get("acquired_from", "")
                mode = link.get("acquisition_mode", "")
                doc_num = link.get("document_number", "")
                doc_date = link.get("document_date", "")
                remarks = link.get("remarks", "")

                summary = f"{acquired_from} → {owner}" if acquired_from else owner
                if mode:
                    summary += f" ({mode})"
                if doc_num:
                    summary += f" [Doc {doc_num}"
                    if doc_date:
                        summary += f", {doc_date}"
                    summary += "]"
                if remarks:
                    summary += f" — {remarks}"

                self._add_fact(
                    "chain", "recital_transfer", summary, src, stype,
                    context=f"Ownership history entry {i + 1}/{len(ownership_history)}",
                )

                # Store prior owner names as party facts for identity resolution
                if owner and isinstance(owner, str) and owner.strip():
                    self._add_fact("party", "prior_owner", owner.strip(), src, stype,
                                   context=f"Recital chain link {i + 1}: {mode or 'transfer'}")

        # Special conditions — restrictive covenants, developer obligations
        special_conditions = data.get("special_conditions")
        if special_conditions and isinstance(special_conditions, list):
            for i, cond in enumerate(special_conditions[:15]):
                if isinstance(cond, str) and cond.strip():
                    self._add_fact("encumbrance", "special_condition",
                                   cond.strip()[:300], src, stype,
                                   context=f"Condition {i + 1}/{len(special_conditions)}")

        # Encumbrance declaration — seller's statement about existing encumbrances
        enc_decl = data.get("encumbrance_declaration")
        if enc_decl and isinstance(enc_decl, str) and enc_decl.strip():
            self._add_fact("encumbrance", "encumbrance_declaration",
                           enc_decl.strip()[:500], src, stype)

        # Payment mode — cash/cheque/NEFT (legally significant for black money checks)
        payment_mode = data.get("payment_mode")
        if payment_mode and isinstance(payment_mode, str) and payment_mode.strip():
            self._add_fact("financial", "payment_mode",
                           payment_mode.strip(), src, stype)

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
            # Add individual survey details + per-survey extent facts
            for sn in survey_numbers:
                if isinstance(sn, dict) and sn.get("survey_no"):
                    sn_no = sn["survey_no"]
                    sn_ext = sn.get("extent")
                    self._add_fact("property", f"survey_{sn_no}_extent",
                                   sn_ext, src, stype)
                    self._add_fact("property", f"survey_{sn_no}_class",
                                   sn.get("classification"), src, stype)
                    # Per-survey extent as a first-class extent fact w/ survey context
                    # so that conflict detection can compare survey-by-survey.
                    if sn_ext:
                        self._add_fact("property", "extent", sn_ext, src, stype,
                                       context=f"Survey {sn_no}")
        else:
            # Fallback for non-schema data
            self._add_fact("property", "survey_number", data.get("survey_number"), src, stype)

        # Aggregate total_extent — stored as separate key for ceiling/compliance checks.
        # NOT stored as generic "extent" to avoid false-positive mismatch with
        # single-survey extents from EC / Sale Deed.
        self._add_fact("property", "total_patta_extent",
                       data.get("total_extent") or data.get("extent") or data.get("area"),
                       src, stype)
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

    # ── Chitta ingestion ──

    def _ingest_chitta(self, filename: str, data: dict):
        src = filename
        stype = "CHITTA"

        self._add_fact("reference", "chitta_number", data.get("chitta_number"), src, stype)
        self._add_fact("property", "village", data.get("village"), src, stype)
        self._add_fact("property", "taluk", data.get("taluk"), src, stype)
        self._add_fact("property", "district", data.get("district"), src, stype)
        self._add_fact("party", "chitta_owner", data.get("owner_name"), src, stype)
        self._add_fact("reference", "patta_number", data.get("patta_number"), src, stype)
        self._add_fact("financial", "tax_assessment", data.get("tax_assessment"), src, stype)
        self._add_fact("property", "irrigation_source", data.get("irrigation_source"), src, stype)

        # Survey-level soil codes
        surveys = data.get("survey_numbers", [])
        if isinstance(surveys, list):
            for sn in surveys:
                if isinstance(sn, dict) and sn.get("survey_no"):
                    self._add_fact("property", "survey_number", sn["survey_no"], src, stype)
                    self._add_fact("property", f"survey_{sn['survey_no']}_soil_code",
                                   sn.get("soil_code"), src, stype)

        # Poramboke detection — critical red flag
        if data.get("is_poramboke"):
            self._add_fact("risk", "poramboke_detected",
                           "Chitta records this land as poramboke (government land)",
                           src, stype)

    # ── A-Register ingestion ──

    def _ingest_a_register(self, filename: str, data: dict):
        src = filename
        stype = "A_REGISTER"

        self._add_fact("reference", "a_register_serial", data.get("register_serial_number"), src, stype)
        self._add_fact("reference", "paguthi_number", data.get("paguthi_number"), src, stype)
        self._add_fact("property", "village", data.get("village"), src, stype)
        self._add_fact("property", "taluk", data.get("taluk"), src, stype)
        self._add_fact("property", "district", data.get("district"), src, stype)
        self._add_fact("party", "a_register_owner", data.get("owner_name"), src, stype)
        self._add_fact("reference", "patta_number", data.get("patta_number"), src, stype)

        # Survey numbers
        surveys = data.get("survey_numbers", [])
        if isinstance(surveys, list):
            for sn in surveys:
                val = sn.get("survey_no", sn) if isinstance(sn, dict) else str(sn)
                self._add_fact("property", "survey_number", val, src, stype)

        # Mutation entries — the core value of A-Register
        mutations = data.get("mutation_entries", [])
        if isinstance(mutations, list):
            chain = []
            for m in mutations:
                if not isinstance(m, dict):
                    continue
                self._add_fact("party", "a_register_from", m.get("from_owner"), src, stype,
                               context=f"Mutation #{m.get('entry_number', '?')}: {m.get('reason', '')}")
                self._add_fact("party", "a_register_to", m.get("to_owner"), src, stype,
                               context=f"Mutation #{m.get('entry_number', '?')}: {m.get('reason', '')}")
                chain.append({
                    "from": m.get("from_owner", "?"),
                    "to": m.get("to_owner", "?"),
                    "type": m.get("reason"),
                    "date": m.get("date"),
                    "order": m.get("order_number"),
                })
            if chain:
                self._add_fact("chain", "a_register_mutations", chain, src, stype)

    # ── FMB ingestion ──

    def _ingest_fmb(self, filename: str, data: dict):
        src = filename
        stype = "FMB"

        self._add_fact("property", "survey_number", data.get("survey_number"), src, stype)
        self._add_fact("property", "village", data.get("village"), src, stype)
        self._add_fact("property", "taluk", data.get("taluk"), src, stype)
        self._add_fact("property", "fmb_area_acres", data.get("area_acres"), src, stype)
        self._add_fact("property", "fmb_area_hectares", data.get("area_hectares"), src, stype)
        self._add_fact("property", "land_classification", data.get("land_classification"), src, stype)

        # Boundaries from FMB
        boundaries = data.get("boundaries")
        if isinstance(boundaries, dict):
            for direction in ("north", "south", "east", "west"):
                val = boundaries.get(direction)
                if val:
                    self._add_fact("property", f"boundary_{direction}", val, src, stype)

        # Dimensions and adjacent surveys
        dims = data.get("dimensions", [])
        if isinstance(dims, list) and dims:
            self._add_fact("property", "fmb_dimensions", dims, src, stype)
        adj = data.get("adjacent_surveys", [])
        if isinstance(adj, list) and adj:
            self._add_fact("property", "adjacent_surveys", adj, src, stype)

    # ── Adangal ingestion ──

    def _ingest_adangal(self, filename: str, data: dict):
        src = filename
        stype = "ADANGAL"

        self._add_fact("property", "survey_number", data.get("survey_number"), src, stype)
        self._add_fact("property", "village", data.get("village"), src, stype)
        self._add_fact("property", "taluk", data.get("taluk"), src, stype)
        self._add_fact("property", "district", data.get("district"), src, stype)
        self._add_fact("party", "adangal_owner", data.get("owner_name"), src, stype)
        self._add_fact("party", "adangal_tenant", data.get("tenant_name"), src, stype)
        self._add_fact("property", "extent", data.get("extent"), src, stype)
        self._add_fact("property", "wet_extent", data.get("wet_extent"), src, stype)
        self._add_fact("property", "dry_extent", data.get("dry_extent"), src, stype)
        self._add_fact("property", "soil_type", data.get("soil_type"), src, stype)
        self._add_fact("property", "irrigation_source", data.get("irrigation_source"), src, stype)
        self._add_fact("property", "cultivation_status", data.get("cultivation_status"), src, stype)
        self._add_fact("financial", "assessment_amount", data.get("assessment_amount"), src, stype)

        # Crop details
        crops = data.get("crop_details", [])
        if isinstance(crops, list) and crops:
            self._add_fact("property", "crop_details", crops, src, stype)

        # Poramboke detection from Adangal
        soil = (data.get("soil_type") or "").lower()
        if "போரம்போக்கு" in soil or "poramboke" in soil or "அரசு நிலம்" in soil:
            self._add_fact("risk", "poramboke_detected",
                           f"Adangal classifies this land as: {data.get('soil_type')}",
                           src, stype)

    # ── Helper: extract name from party field (may be dict or plain string) ──

    @staticmethod
    def _name_from_party(val) -> str | None:
        """Extract name from a party field that may be a dict or a plain string."""
        if isinstance(val, dict):
            return val.get("name")
        if isinstance(val, str) and val.strip():
            return val.strip()
        return None

    # ── Layout Approval ingestion ──

    def _ingest_layout_approval(self, filename: str, data: dict):
        src = filename
        stype = "LAYOUT_APPROVAL"

        self._add_fact("reference", "layout_approval_number", data.get("approval_number"), src, stype)
        self._add_fact("reference", "layout_authority", data.get("authority"), src, stype)
        self._add_fact("timeline", "layout_approval_date", data.get("approval_date"), src, stype)
        self._add_fact("timeline", "layout_validity_period", data.get("validity_period"), src, stype)
        self._add_fact("property", "layout_status", data.get("status"), src, stype)
        self._add_fact("property", "layout_total_plots", data.get("total_plots"), src, stype)

        # Survey reference — schema field is "survey_number" (not "parent_survey_numbers")
        parent_surveys = data.get("survey_number") or data.get("parent_survey_numbers")
        if parent_surveys:
            self._add_fact("property", "survey_number", parent_surveys, src, stype)

        # Conditions
        conditions = data.get("conditions", [])
        if isinstance(conditions, list) and conditions:
            self._add_fact("encumbrance", "layout_conditions", conditions, src, stype)

        # Expired layout is a risk
        if data.get("status") == "expired":
            self._add_fact("risk", "layout_expired",
                           "Layout approval has expired — plots may not be legally transferable",
                           src, stype)

    # ── Legal Heir Certificate ingestion ──

    def _ingest_legal_heir(self, filename: str, data: dict):
        src = filename
        stype = "LEGAL_HEIR"

        self._add_fact("reference", "legal_heir_cert_number", data.get("certificate_number"), src, stype)
        self._add_fact("reference", "legal_heir_authority", data.get("issuing_authority"), src, stype)
        self._add_fact("party", "deceased_person", data.get("deceased_name"), src, stype)
        self._add_fact("timeline", "date_of_death", data.get("date_of_death"), src, stype)

        # All heirs
        heirs = data.get("heirs", [])
        if isinstance(heirs, list):
            for h in heirs:
                if isinstance(h, dict):
                    self._add_fact("party", "legal_heir", h.get("name"), src, stype,
                                   context=f"Relationship: {h.get('relationship', '?')}, "
                                           f"Share: {h.get('share_percentage', '?')}")
                    if h.get("minor"):
                        self._add_fact("risk", "minor_heir",
                                       f"Minor heir: {h.get('name')} (guardian: {h.get('guardian', '?')})",
                                       src, stype)

    # ── POA ingestion ──

    def _ingest_poa(self, filename: str, data: dict):
        src = filename
        stype = "POA"

        self._add_fact("reference", "poa_document_number",
                      data.get("registration_number") or data.get("document_number"), src, stype)
        self._add_fact("reference", "sro", data.get("sro"), src, stype)
        self._add_fact("timeline", "poa_registration_date", data.get("registration_date"), src, stype)
        self._add_fact("party", "poa_principal", self._name_from_party(data.get("principal")), src, stype)
        self._add_fact("party", "poa_agent", self._name_from_party(data.get("agent")), src, stype)
        poa_kind = data.get("is_general_or_specific") or data.get("poa_type")
        self._add_fact("property", "poa_type", poa_kind, src, stype)
        self._add_fact("property", "poa_revocation_status", data.get("revocation_status"), src, stype)

        # Powers granted
        powers = data.get("powers_granted", [])
        if isinstance(powers, list) and powers:
            self._add_fact("encumbrance", "poa_powers", powers, src, stype)

        # GPA risk flag
        poa_kind_lower = (str(poa_kind) if poa_kind else "").lower()
        if poa_kind_lower in ("general", "gpa"):
            self._add_fact("risk", "general_poa",
                           "General Power of Attorney detected — GPA sales are not legally recognized "
                           "(Suraj Lamp Industries v. State of Haryana, 2012)",
                           src, stype)

    # ── Court Order ingestion ──

    def _ingest_court_order(self, filename: str, data: dict):
        src = filename
        stype = "COURT_ORDER"

        self._add_fact("reference", "court_case_number", data.get("case_number"), src, stype)
        self._add_fact("reference", "court_name", data.get("court_name"), src, stype)
        self._add_fact("timeline", "court_order_date", data.get("order_date"), src, stype)
        self._add_fact("party", "court_petitioner", data.get("petitioner"), src, stype)
        self._add_fact("party", "court_respondent", data.get("respondent"), src, stype)
        self._add_fact("encumbrance", "court_order_type", data.get("order_type"), src, stype)
        self._add_fact("encumbrance", "court_order_status", data.get("status"), src, stype)
        self._add_fact("property", "court_property_affected", data.get("property_affected"), src, stype)
        self._add_fact("encumbrance", "court_restriction", data.get("restriction_type"), src, stype)

        # Critical risk flags
        order_type = (data.get("order_type") or "").lower()
        if order_type in ("injunction", "temporary_injunction", "permanent_injunction"):
            self._add_fact("risk", "active_injunction",
                           f"Court injunction ({order_type}) on property — transfer may be prohibited",
                           src, stype)
        if order_type == "attachment":
            self._add_fact("risk", "attachment_order",
                           "Court attachment order — property is seized/frozen",
                           src, stype)

    # ── Will ingestion ──

    def _ingest_will(self, filename: str, data: dict):
        src = filename
        stype = "WILL"

        self._add_fact("reference", "will_registration", data.get("registration_number"), src, stype)
        self._add_fact("party", "testator", self._name_from_party(data.get("testator")), src, stype)
        self._add_fact("party", "executor", self._name_from_party(data.get("executor")), src, stype)
        self._add_fact("timeline", "will_date",
                      data.get("execution_date") or data.get("will_date"), src, stype)
        self._add_fact("encumbrance", "probate_status", data.get("probate_status"), src, stype)

        # Beneficiaries
        beneficiaries = data.get("beneficiaries", [])
        if isinstance(beneficiaries, list):
            for b in beneficiaries:
                if isinstance(b, dict):
                    self._add_fact("party", "will_beneficiary", b.get("name"), src, stype,
                                   context=f"Share: {b.get('share', '?')}, "
                                           f"Property: {b.get('property_bequeathed', '?')}")

        # Witnesses
        witnesses = data.get("witnesses", [])
        if isinstance(witnesses, list):
            for w in witnesses:
                if isinstance(w, str) and w.strip():
                    self._add_fact("party", "witness", w.strip(), src, stype)
            if len(witnesses) < 2:
                self._add_fact("risk", "insufficient_will_witnesses",
                               f"Will has only {len(witnesses)} witness(es) — minimum 2 required",
                               src, stype)

        # Unregistered Will risk
        if data.get("is_registered") is False:
            self._add_fact("risk", "unregistered_will",
                           "Will is not registered — harder to prove validity",
                           src, stype)

    # ── Partition Deed ingestion ──

    def _ingest_partition_deed(self, filename: str, data: dict):
        src = filename
        stype = "PARTITION_DEED"

        self._add_fact("reference", "partition_deed_number",
                      data.get("registration_number") or data.get("document_number"), src, stype)
        self._add_fact("reference", "sro", data.get("sro"), src, stype)
        self._add_fact("timeline", "partition_date", data.get("registration_date"), src, stype)
        self._add_fact("property", "original_property", data.get("original_property"), src, stype)
        # survey_number & village are nested inside original_property in schema
        orig_prop = data.get("original_property") or {}
        orig_prop = orig_prop if isinstance(orig_prop, dict) else {}
        self._add_fact("property", "survey_number",
                      orig_prop.get("survey_number") or data.get("original_survey_numbers"), src, stype)
        self._add_fact("property", "village",
                      orig_prop.get("village") or data.get("village"), src, stype)

        # Joint owners — handle list of dicts or list of strings
        joint_owners = data.get("joint_owners", [])
        if isinstance(joint_owners, list):
            for o in joint_owners:
                if isinstance(o, dict):
                    self._add_fact("party", "partition_owner", o.get("name"), src, stype,
                                   context=f"Original share: {o.get('original_share', '?')}")
                elif isinstance(o, str) and o.strip():
                    self._add_fact("party", "partition_owner", o, src, stype)

        # Partitioned shares — check both key names
        shares = data.get("partitioned_shares") or data.get("shares", [])
        if isinstance(shares, list):
            for s in shares:
                if isinstance(s, dict):
                    name = s.get("name") or s.get("owner")
                    survey = s.get("allocated_survey") or s.get("survey")
                    extent = s.get("allocated_extent") or s.get("extent") or s.get("share")
                    self._add_fact("chain", "partition_allocation",
                                   {
                                       "to": name,
                                       "survey": survey,
                                       "extent": extent,
                                   }, src, stype)

        # Consent flag
        if data.get("consent_all_parties") is False:
            self._add_fact("risk", "partition_without_consent",
                           "Not all parties consented to the partition — may be voidable",
                           src, stype)

    # ── Gift Deed ingestion ──

    def _ingest_gift_deed(self, filename: str, data: dict):
        src = filename
        stype = "GIFT_DEED"

        self._add_fact("reference", "gift_deed_number",
                      data.get("registration_number") or data.get("document_number"), src, stype)
        self._add_fact("reference", "sro", data.get("sro"), src, stype)
        self._add_fact("timeline", "gift_deed_date", data.get("registration_date"), src, stype)
        self._add_fact("party", "donor", self._name_from_party(data.get("donor")), src, stype)
        self._add_fact("party", "donee", self._name_from_party(data.get("donee")), src, stype)
        self._add_fact("property", "gift_property", data.get("property"), src, stype)
        # survey_number is nested inside property in schema
        prop = data.get("property") or {}
        prop = prop if isinstance(prop, dict) else {}
        self._add_fact("property", "survey_number",
                      prop.get("survey_number") or data.get("survey_number"), src, stype)

        # Consideration should be 0 for genuine gift
        consideration = data.get("consideration_amount")
        if consideration and str(consideration).strip() not in ("0", "nil", ""):
            self._add_fact("risk", "gift_with_consideration",
                           f"Gift deed has non-zero consideration ({consideration}) — may be disguised sale",
                           src, stype)

        # Acceptance clause required
        if not data.get("acceptance_clause"):
            self._add_fact("risk", "gift_no_acceptance",
                           "Gift deed missing acceptance clause — gift may be incomplete",
                           src, stype)

    # ── Release Deed ingestion ──

    def _ingest_release_deed(self, filename: str, data: dict):
        src = filename
        stype = "RELEASE_DEED"

        self._add_fact("reference", "release_deed_number",
                      data.get("registration_number") or data.get("document_number"), src, stype)
        self._add_fact("reference", "sro", data.get("sro"), src, stype)
        self._add_fact("timeline", "release_deed_date", data.get("registration_date"), src, stype)
        self._add_fact("party", "releasing_party",
                      self._name_from_party(data.get("releasing_party")), src, stype)
        self._add_fact("party", "release_beneficiary",
                      self._name_from_party(data.get("beneficiary") or data.get("beneficiary_party")),
                      src, stype)
        self._add_fact("encumbrance", "claim_released", data.get("claim_released"), src, stype)

        # Original document reference — important for EC cross-check
        orig_raw = data.get("original_document")
        if isinstance(orig_raw, dict):
            orig = orig_raw
        elif isinstance(orig_raw, str) and orig_raw.strip():
            # Plain string — store as-is
            orig = {"type": orig_raw, "number": None, "sro": None}
        else:
            orig = {}
        if orig:
            self._add_fact("reference", "release_original_doc_type",
                          orig.get("document_type") or orig.get("type"), src, stype)
            self._add_fact("reference", "release_original_doc_number", orig.get("number"), src, stype)
            self._add_fact("reference", "release_original_sro", orig.get("sro"), src, stype)

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
                  confidence: float = 1.0, context: str = "", transaction_id: str = ""):
        """Add a fact if the value is non-empty."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return
        self.facts.append(Fact(
            category, key, value, src, stype,
            confidence, context, transaction_id=transaction_id,
        ))

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

        # --- Extent (unit-aware, Patta-portfolio filtered) ---
        # Patta now stores per-survey extent facts with context="Survey NNN".
        # Only compare Patta extents whose survey matches a non-Patta document's
        # survey, avoiding false conflicts from the owner's other holdings.
        extent_facts = groups.get(("property", "extent"), [])
        if len(extent_facts) >= 2:
            _PATTA_STYPES = {"PATTA", "CHITTA"}

            # Gather non-Patta survey numbers for matching
            non_patta_surveys: list[str] = []
            for sf in groups.get(("property", "survey_number"), []):
                if sf.source_type not in _PATTA_STYPES:
                    non_patta_surveys.extend(split_survey_numbers(str(sf.value)))

            # Filter extent facts: keep non-Patta as-is; for Patta, keep only
            # those whose survey context matches a non-Patta survey.
            comparable: list["Fact"] = []
            for f in extent_facts:
                if f.source_type not in _PATTA_STYPES:
                    comparable.append(f)
                elif f.context and f.context.startswith("Survey "):
                    patta_sn = f.context[len("Survey "):]
                    for nps in non_patta_surveys:
                        m_ok, _ = survey_numbers_match(patta_sn, nps)
                        if m_ok:
                            comparable.append(f)
                            break
                # Patta extent w/o survey context → skip (old aggregate)

            if len(comparable) >= 2:
                by_source_ext: dict[str, "Fact"] = {}
                for f in comparable:
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

        # --- Taluk consistency (fuzzy, cross-script aware) ---
        taluk_facts = groups.get(("property", "taluk"), [])
        if len(taluk_facts) >= 2:
            by_source_taluk: dict[str, "Fact"] = {}
            for f in taluk_facts:
                by_source_taluk.setdefault(f.source_file, f)
            if len(by_source_taluk) >= 2:
                src_list = list(by_source_taluk.items())
                for i, (src_a, fact_a) in enumerate(src_list):
                    for src_b, fact_b in src_list[i + 1:]:
                        matched, _ = village_names_match(
                            str(fact_a.value), str(fact_b.value)
                        )
                        if not matched:
                            # Downgrade cross-script taluk mismatches
                            val_a, val_b = str(fact_a.value), str(fact_b.value)
                            a_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in val_a)
                            b_tamil = any('\u0B80' <= ch <= '\u0BFF' for ch in val_b)
                            cross_script = (a_tamil != b_tamil)
                            severity = "LOW" if cross_script else "MEDIUM"
                            self.conflicts.append(Conflict(
                                category="property",
                                key="taluk",
                                facts=[fact_a, fact_b],
                                severity=severity,
                                description=(
                                    f"Taluk differs across documents: "
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
                transaction_id=fd.get("transaction_id", ""),
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

        # Timeline facts — dates, periods, sequences
        timeline_facts = self.get_facts_by_category("timeline")
        if timeline_facts:
            lines.append("\n--- TIMELINE ---")
            for f in timeline_facts:
                ctx = f" [{f['context']}]" if f.get("context") else ""
                lines.append(f"  {f['key']}: {f['value']} (from {f['source_file']}){ctx}")

        # Encumbrance facts — mortgages, liens, restrictions
        enc_facts = self.get_facts_by_category("encumbrance")
        if enc_facts:
            lines.append("\n--- ENCUMBRANCES & RESTRICTIONS ---")
            for f in enc_facts:
                ctx = f" [{f['context']}]" if f.get("context") else ""
                lines.append(f"  {f['key']}: {f['value']} (from {f['source_file']}){ctx}")

        # Chain of title facts — ownership transfers
        chain_facts = self.get_facts_by_category("chain")
        if chain_facts:
            lines.append("\n--- CHAIN OF TITLE ---")
            for f in chain_facts:
                val = f['value']
                if isinstance(val, list):
                    # Structured chain — format each link
                    lines.append(f"  {f['key']} (from {f['source_file']}):")
                    for link in val[:20]:  # Cap display at 20 links
                        if isinstance(link, dict):
                            from_p = link.get("from", "?")
                            to_p = link.get("to", "?")
                            ltype = link.get("type", "")
                            ldate = link.get("date", "")
                            lines.append(f"    {from_p} → {to_p} ({ltype}, {ldate})")
                        else:
                            lines.append(f"    {link}")
                    if len(val) > 20:
                        lines.append(f"    ... and {len(val) - 20} more transfers")
                else:
                    ctx = f" [{f['context']}]" if f.get("context") else ""
                    lines.append(f"  {f['key']}: {val} (from {f['source_file']}){ctx}")

        # Cross-document references — EC numbers, deed numbers, SRO, patta numbers
        ref_facts = self.get_facts_by_category("reference")
        if ref_facts:
            lines.append("\n--- REFERENCES ---")
            for f in ref_facts:
                ctx = f" [{f['context']}]" if f.get("context") else ""
                lines.append(f"  {f['key']}: {f['value']} (from {f['source_file']}){ctx}")

        # Risk flags — red flags, anomalies, concerns
        risk_facts = self.get_facts_by_category("risk")
        if risk_facts:
            lines.append("\n--- RISK FLAGS ---")
            for f in risk_facts:
                ctx = f" [{f['context']}]" if f.get("context") else ""
                lines.append(f"  ⚠ {f['key']}: {f['value']} (from {f['source_file']}){ctx}")

        return "\n".join(lines)
