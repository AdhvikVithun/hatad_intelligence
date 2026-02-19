"""Forensic Identity Resolution Engine.

Instead of pairwise string matching ("do these two names match?"),
this module treats each document as a *witness* providing partial,
possibly corrupted evidence about a person.  It collects every person
mention across all uploaded documents, clusters them into identity
groups using agglomerative clustering, and for each group computes a
corroboration-based confidence score with a human-readable evidence
report.

Usage in the pipeline::

    resolver = IdentityResolver()
    resolver.collect_mentions(session.extracted_data)
    clusters = resolver.resolve()

    # Query: does the buyer share an identity with the patta owner?
    same, conf, evidence = resolver.roles_share_identity("buyer", "patta_owner")

Architecture notes:
  - Reuses existing utilities from ``utils.py`` (``name_similarity``,
    ``split_name_parts``, ``detect_garbled_tamil``, etc.)
  - Complete-linkage agglomerative clustering prevents transitive
    chaining errors (A ~ B and B ~ C does NOT force A ~ C)
  - The 0.50 merge threshold matches the existing deterministic-check
    threshold for behavioural consistency
  - JSON-serializable via ``to_dict()`` for session persistence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.utils import (
    name_similarity,
    split_name_parts,
    split_party_names,
    has_tamil,
    detect_garbled_tamil,
)
from app.config import TRACE_ENABLED

logger = logging.getLogger(__name__)

# ── Constants ──
_MERGE_THRESHOLD = 0.50  # same as deterministic check threshold
_ROLE_LABELS = {
    "buyer": "Buyer",
    "seller": "Seller",
    "patta_owner": "Patta Owner",
    "ec_claimant": "EC Claimant/Buyer",
    "ec_executant": "EC Executant/Seller",
    "party": "Party",
}
_CONFIDENCE_BANDS = [
    (0.85, "HIGH"),
    (0.70, "MODERATE"),
    (0.50, "LOW"),
    (0.00, "VERY LOW"),
]


def _trace(msg: str):
    if TRACE_ENABLED:
        logger.debug(f"[TRACE] {msg}")


def _confidence_band(score: float) -> str:
    for threshold, label in _CONFIDENCE_BANDS:
        if score >= threshold:
            return label
    return "VERY LOW"


# ═══════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════

@dataclass
class PersonMention:
    """A single mention of a person in one document."""
    name: str                 # raw name as extracted
    role: str                 # buyer | seller | patta_owner | ec_claimant | ec_executant
    source_file: str          # filename
    source_type: str          # EC | SALE_DEED | PATTA | CHITTA | OTHER
    date: str = ""            # transaction date (if available)
    ocr_quality: float = 1.0  # 1.0 = clean, 0.3 = garbled

    # Computed during collection
    given: str = ""           # given name component (from split_name_parts)
    patronymic: str = ""      # patronymic component

    def __post_init__(self):
        g, p = split_name_parts(self.name)
        self.given = g
        self.patronymic = p
        # Assess OCR quality via garbled Tamil detection
        is_garbled, quality, _reason = detect_garbled_tamil(self.name)
        if is_garbled:
            self.ocr_quality = max(0.3, 1.0 - 0.7)  # garbled → 0.3

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "source_file": self.source_file,
            "source_type": self.source_type,
            "date": self.date,
            "ocr_quality": self.ocr_quality,
            "given": self.given,
            "patronymic": self.patronymic,
        }


@dataclass
class IdentityCluster:
    """A group of mentions believed to refer to the same person."""
    cluster_id: str                               # e.g. "PERSON_1"
    mentions: list[PersonMention] = field(default_factory=list)
    consensus_name: str = ""                       # best representative name
    confidence: float = 0.0                        # 0.0–1.0 corroboration score
    evidence_lines: list[str] = field(default_factory=list)
    roles: set[str] = field(default_factory=set)
    source_files: set[str] = field(default_factory=set)
    source_types: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "consensus_name": self.consensus_name,
            "confidence": round(self.confidence, 2),
            "confidence_band": _confidence_band(self.confidence),
            "evidence_lines": self.evidence_lines,
            "roles": sorted(self.roles),
            "source_files": sorted(self.source_files),
            "source_types": sorted(self.source_types),
            "mention_count": len(self.mentions),
            "mentions": [m.to_dict() for m in self.mentions],
        }


# ═══════════════════════════════════════════════════
# IDENTITY RESOLVER
# ═══════════════════════════════════════════════════

class IdentityResolver:
    """Collects person mentions from extracted data and clusters them
    into identity groups using complete-linkage agglomerative clustering."""

    def __init__(self):
        self.mentions: list[PersonMention] = []
        self.clusters: list[IdentityCluster] = []
        self._resolved = False

    # ── Collection ──

    def collect_mentions(self, extracted_data: dict) -> int:
        """Extract all person mentions from pipeline output.

        Args:
            extracted_data: dict of ``{filename: {document_type, data}}``

        Returns:
            Total number of mentions collected.
        """
        self.mentions.clear()
        self._resolved = False

        for filename, entry in extracted_data.items():
            doc_type = entry.get("document_type", "OTHER")
            d = entry.get("data")
            if not d or not isinstance(d, dict):
                continue

            if doc_type == "SALE_DEED":
                self._collect_sale_deed(d, filename, doc_type)
            elif doc_type in ("PATTA", "CHITTA"):
                self._collect_patta(d, filename, doc_type)
            elif doc_type == "EC":
                self._collect_ec(d, filename, doc_type)
            else:
                self._collect_generic(d, filename, doc_type)

        _trace(f"IDENTITY collected {len(self.mentions)} person mentions "
               f"from {len(extracted_data)} documents")
        return len(self.mentions)

    def _collect_sale_deed(self, d: dict, filename: str, doc_type: str):
        date = ""
        if isinstance(d.get("registration"), dict):
            date = d["registration"].get("registration_date", "")

        for party in (d.get("seller") or []):
            name = party.get("name", party) if isinstance(party, dict) else str(party)
            if name and name.strip():
                self.mentions.append(PersonMention(
                    name=name.strip(), role="seller",
                    source_file=filename, source_type=doc_type, date=date,
                ))

        for party in (d.get("buyer") or []):
            name = party.get("name", party) if isinstance(party, dict) else str(party)
            if name and name.strip():
                self.mentions.append(PersonMention(
                    name=name.strip(), role="buyer",
                    source_file=filename, source_type=doc_type, date=date,
                ))

    def _collect_patta(self, d: dict, filename: str, doc_type: str):
        for owner in (d.get("owner_names") or []):
            name = owner.get("name", owner) if isinstance(owner, dict) else str(owner)
            if name and name.strip():
                self.mentions.append(PersonMention(
                    name=name.strip(), role="patta_owner",
                    source_file=filename, source_type=doc_type,
                ))

    def _collect_ec(self, d: dict, filename: str, doc_type: str):
        for txn in (d.get("transactions") or []):
            date = txn.get("registration_date", "")
            seller = txn.get("seller_or_executant", "")
            buyer = txn.get("buyer_or_claimant", "")

            if seller:
                for name in split_party_names(seller):
                    if name.strip():
                        self.mentions.append(PersonMention(
                            name=name.strip(), role="ec_executant",
                            source_file=filename, source_type=doc_type, date=date,
                        ))
            if buyer:
                for name in split_party_names(buyer):
                    if name.strip():
                        self.mentions.append(PersonMention(
                            name=name.strip(), role="ec_claimant",
                            source_file=filename, source_type=doc_type, date=date,
                        ))

    def _collect_generic(self, d: dict, filename: str, doc_type: str):
        for party in (d.get("key_parties") or []):
            if isinstance(party, dict):
                name = party.get("name", "")
            else:
                name = str(party)
            if name and name.strip():
                self.mentions.append(PersonMention(
                    name=name.strip(), role="party",
                    source_file=filename, source_type=doc_type,
                ))

    # ── Resolution ──

    def resolve(self) -> list[IdentityCluster]:
        """Cluster mentions into identity groups.

        Uses complete-linkage agglomerative clustering:
        a mention only joins a cluster if it is similar to ALL
        existing members — prevents transitive chaining errors.

        Returns:
            List of ``IdentityCluster`` objects.
        """
        if not self.mentions:
            self.clusters = []
            self._resolved = True
            return []

        n = len(self.mentions)
        _trace(f"IDENTITY resolving {n} mentions")

        # ── Step 1: Build similarity matrix ──
        sim_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                s = name_similarity(self.mentions[i].name, self.mentions[j].name)
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        # ── Step 2: Agglomerative clustering (complete linkage) ──
        # Start: each mention is its own cluster
        cluster_members: list[list[int]] = [[i] for i in range(n)]
        active = set(range(n))  # indices into cluster_members

        while len(active) > 1:
            # Find the pair with highest minimum (complete-linkage) similarity
            best_sim = -1.0
            best_pair = (-1, -1)

            active_list = sorted(active)
            for idx_a in range(len(active_list)):
                for idx_b in range(idx_a + 1, len(active_list)):
                    ca = active_list[idx_a]
                    cb = active_list[idx_b]

                    # Complete linkage: minimum similarity between all member pairs
                    min_sim = 1.0
                    for mi in cluster_members[ca]:
                        for mj in cluster_members[cb]:
                            if sim_matrix[mi][mj] < min_sim:
                                min_sim = sim_matrix[mi][mj]

                    if min_sim > best_sim:
                        best_sim = min_sim
                        best_pair = (ca, cb)

            if best_sim < _MERGE_THRESHOLD:
                break  # No more merges above threshold

            # Merge the best pair
            ca, cb = best_pair
            cluster_members[ca].extend(cluster_members[cb])
            active.discard(cb)
            _trace(f"IDENTITY merged clusters → size {len(cluster_members[ca])} "
                   f"(min_sim={best_sim:.2f})")

        # ── Step 3: Build IdentityCluster objects ──
        self.clusters = []
        cluster_idx = 1
        for ci in sorted(active):
            members = cluster_members[ci]
            mentions = [self.mentions[mi] for mi in members]
            cluster = IdentityCluster(cluster_id=f"PERSON_{cluster_idx}")
            cluster.mentions = mentions
            cluster.roles = {m.role for m in mentions}
            cluster.source_files = {m.source_file for m in mentions}
            cluster.source_types = {m.source_type for m in mentions}
            cluster.consensus_name = self._pick_consensus(mentions)
            cluster.confidence = self._compute_confidence(cluster)
            cluster.evidence_lines = self._build_evidence(cluster)
            self.clusters.append(cluster)
            cluster_idx += 1

        self._resolved = True
        _trace(f"IDENTITY resolved into {len(self.clusters)} identity clusters")
        return self.clusters

    # ── Consensus name selection ──

    @staticmethod
    def _pick_consensus(mentions: list[PersonMention]) -> str:
        """Select the best representative name from a set of mentions.

        Preference order:
          1. Non-garbled name (OCR quality > 0.5)
          2. Longest name (most complete)
          3. Name with patronymic (most specific)
        """
        # Filter to non-garbled mentions
        clean = [m for m in mentions if m.ocr_quality > 0.5]
        candidates = clean if clean else mentions

        # Sort: prefer with patronymic, then longest
        def sort_key(m: PersonMention) -> tuple:
            return (
                1 if m.patronymic else 0,     # prefer names with patronymic
                len(m.name),                   # prefer longer (more complete)
                -m.ocr_quality,                # tiebreak: highest quality
            )

        candidates_sorted = sorted(candidates, key=sort_key, reverse=True)
        return candidates_sorted[0].name if candidates_sorted else mentions[0].name

    # ── Confidence scoring ──

    @staticmethod
    def _compute_confidence(cluster: IdentityCluster) -> float:
        """Compute corroboration-based confidence for an identity cluster.

        Signals:
          - Base: 0.50 for a single isolated mention
          - +0.10 per additional corroborating source file (cap +0.30)
          - +0.10 if source types are diverse (≥2 distinct doc types)
          - +0.05 if given-name component is consistent across all mentions
          - OCR quality: weighted average as a multiplier
        """
        mentions = cluster.mentions
        if not mentions:
            return 0.0

        # Base confidence
        conf = 0.50

        # Additional source files corroboration
        unique_files = len(cluster.source_files)
        conf += min(0.30, 0.10 * (unique_files - 1))

        # Source type diversity
        if len(cluster.source_types) >= 2:
            conf += 0.10

        # Given-name consistency check
        givens = [m.given for m in mentions if m.given]
        if len(givens) >= 2:
            # Check if all given names are similar
            from itertools import combinations
            all_similar = True
            from app.pipeline.utils import base_name_similarity
            for g1, g2 in combinations(givens, 2):
                if base_name_similarity(g1, g2) < 0.5:
                    all_similar = False
                    break
            if all_similar:
                conf += 0.05

        # OCR quality multiplier (weighted avg)
        avg_quality = sum(m.ocr_quality for m in mentions) / len(mentions)
        if avg_quality < 0.7:
            conf *= (0.6 + 0.4 * avg_quality)  # scale down for poor quality

        return min(1.0, conf)

    # ── Evidence generation ──

    @staticmethod
    def _build_evidence(cluster: IdentityCluster) -> list[str]:
        """Generate human-readable evidence lines for an identity cluster."""
        lines = []

        # List each mention
        for m in cluster.mentions:
            role_label = _ROLE_LABELS.get(m.role, m.role)
            quality_tag = ""
            if m.ocr_quality < 0.5:
                quality_tag = " [garbled OCR]"
            date_tag = f" ({m.date})" if m.date else ""
            lines.append(
                f"  ✓ '{m.name}' — {m.source_type} [{m.source_file}]{date_tag} "
                f"as {role_label}{quality_tag}"
            )

        # Corroboration summary
        unique_files = len(cluster.source_files)
        unique_types = len(cluster.source_types)

        if unique_files > 1:
            lines.append(
                f"  → Name appears across {unique_files} documents "
                f"({unique_types} document type{'s' if unique_types > 1 else ''})"
            )

        # Given-name consistency
        givens = set(m.given for m in cluster.mentions if m.given)
        if len(givens) > 1:
            lines.append(f"  → Given name variants: {', '.join(repr(g) for g in givens)}")
        elif len(givens) == 1:
            given = next(iter(givens))
            if any(m.given == given for m in cluster.mentions):
                count = sum(1 for m in cluster.mentions if m.given == given)
                lines.append(f"  → Given name '{given}' consistent in {count}/{len(cluster.mentions)} mentions")

        # Confidence
        band = _confidence_band(cluster.confidence)
        lines.append(
            f"  → Confidence: {cluster.confidence:.0%} ({band})"
        )

        return lines

    # ── Query methods ──

    def find_clusters_for_role(self, role: str) -> list[IdentityCluster]:
        """Return all identity clusters containing a mention with the given role."""
        if not self._resolved:
            raise RuntimeError("Call resolve() before querying clusters")
        return [c for c in self.clusters if role in c.roles]

    def roles_share_identity(
        self, role_a: str, role_b: str
    ) -> tuple[bool, float, str]:
        """Check if any identity cluster contains mentions with both roles.

        Returns:
            ``(is_same_person, confidence, evidence_report)``
        """
        if not self._resolved:
            raise RuntimeError("Call resolve() before querying clusters")

        label_a = _ROLE_LABELS.get(role_a, role_a)
        label_b = _ROLE_LABELS.get(role_b, role_b)

        for cluster in self.clusters:
            if role_a in cluster.roles and role_b in cluster.roles:
                evidence = (
                    f"Identity Match: {label_a} and {label_b} resolved to the same person "
                    f"'{cluster.consensus_name}' (Confidence: {cluster.confidence:.0%} "
                    f"— {_confidence_band(cluster.confidence)})\n"
                    + "\n".join(cluster.evidence_lines)
                )
                return (True, cluster.confidence, evidence)

        # Not found in any shared cluster — build explanation
        clusters_a = self.find_clusters_for_role(role_a)
        clusters_b = self.find_clusters_for_role(role_b)

        if not clusters_a and not clusters_b:
            return (False, 0.0, f"No {label_a} or {label_b} mentions found in any document.")
        if not clusters_a:
            return (False, 0.0, f"No {label_a} mentions found in any document.")
        if not clusters_b:
            return (False, 0.0, f"No {label_b} mentions found in any document.")

        # Show why they didn't match
        lines = [f"Identity Mismatch: {label_a} and {label_b} resolved to DIFFERENT persons."]
        for c in clusters_a:
            if role_a in c.roles:
                lines.append(f"  {label_a} cluster: '{c.consensus_name}' "
                             f"(Confidence: {c.confidence:.0%})")
        for c in clusters_b:
            if role_b in c.roles:
                lines.append(f"  {label_b} cluster: '{c.consensus_name}' "
                             f"(Confidence: {c.confidence:.0%})")
        return (False, 0.0, "\n".join(lines))

    def get_cluster_for_name(self, name: str, threshold: float = 0.50) -> IdentityCluster | None:
        """Find the cluster that best matches a given name string."""
        if not self._resolved:
            raise RuntimeError("Call resolve() before querying clusters")
        best_cluster = None
        best_sim = threshold
        for cluster in self.clusters:
            for m in cluster.mentions:
                sim = name_similarity(name, m.name)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster
        return best_cluster

    # ── Chain-of-title helpers ──

    def check_chain_continuity(self, extracted_data: dict) -> list[dict]:
        """Deterministic chain-of-title continuity check using identity clusters.

        For each consecutive pair of EC transactions, checks whether the
        buyer/claimant in Tx N shares an identity cluster with the
        seller/executant in Tx N+1.

        Returns:
            List of check dicts (``CheckResult``-compatible).
        """
        if not self._resolved:
            raise RuntimeError("Call resolve() before checking chain continuity")

        checks: list[dict] = []
        ec_transactions = []

        # Collect EC transactions with dates
        for filename, entry in extracted_data.items():
            if entry.get("document_type") != "EC":
                continue
            d = entry.get("data")
            if not d or not isinstance(d, dict):
                continue
            for txn in (d.get("transactions") or []):
                ec_transactions.append({
                    "buyer": txn.get("buyer_or_claimant", ""),
                    "seller": txn.get("seller_or_executant", ""),
                    "date": txn.get("registration_date", ""),
                    "doc_no": txn.get("document_number", ""),
                    "filename": filename,
                })

        if len(ec_transactions) < 2:
            return checks  # Can't check continuity with < 2 transactions

        # Check consecutive pairs
        gaps_found = 0
        for i in range(len(ec_transactions) - 1):
            tx_curr = ec_transactions[i]
            tx_next = ec_transactions[i + 1]

            buyer_name = tx_curr["buyer"]
            seller_name = tx_next["seller"]

            if not buyer_name.strip() or not seller_name.strip():
                continue

            # Split multi-party names and check if ANY buyer matches ANY next seller
            buyer_names = split_party_names(buyer_name)
            seller_names = split_party_names(seller_name)

            link_found = False
            for bn in buyer_names:
                buyer_cluster = self.get_cluster_for_name(bn)
                if not buyer_cluster:
                    continue
                for sn in seller_names:
                    seller_cluster = self.get_cluster_for_name(sn)
                    if seller_cluster and seller_cluster.cluster_id == buyer_cluster.cluster_id:
                        link_found = True
                        break
                if link_found:
                    break

            if not link_found:
                # Also do a direct name_similarity check in case clustering
                # missed something (e.g. name only appeared once)
                direct_match = False
                for bn in buyer_names:
                    for sn in seller_names:
                        if name_similarity(bn, sn) >= _MERGE_THRESHOLD:
                            direct_match = True
                            break
                    if direct_match:
                        break

                if not direct_match:
                    gaps_found += 1
                    date_curr = tx_curr["date"] or "unknown date"
                    date_next = tx_next["date"] or "unknown date"
                    checks.append({
                        "rule_code": "DET_CHAIN_BREAK",
                        "rule_name": "Chain of Title Break (Identity)",
                        "severity": "HIGH",
                        "status": "WARNING",
                        "explanation": (
                            f"Buyer/claimant in transaction #{i+1} ({date_curr}) "
                            f"'{buyer_name}' does not match seller/executant in "
                            f"transaction #{i+2} ({date_next}) '{seller_name}'. "
                            f"No identity cluster links these names — this may indicate "
                            f"a gap in the chain of title."
                        ),
                        "recommendation": (
                            "Verify that the property was transferred through a valid "
                            "transaction between these parties. Check for missing "
                            "intermediate transactions (inheritance, partition, gift)."
                        ),
                        "evidence": (
                            f"Tx #{i+1} buyer='{buyer_name}' ({date_curr}) → "
                            f"Tx #{i+2} seller='{seller_name}' ({date_next}): "
                            f"No identity link found"
                        ),
                        "source": "deterministic",
                    })

        if ec_transactions and gaps_found == 0:
            # All chain links verified
            chain_len = len(ec_transactions)
            checks.append({
                "rule_code": "DET_CHAIN_CONTINUITY",
                "rule_name": "Chain of Title Continuity Verified",
                "severity": "INFO",
                "status": "PASS",
                "explanation": (
                    f"Chain of title is continuous across {chain_len} EC transactions. "
                    f"Each buyer/claimant in one transaction matches the seller/executant "
                    f"in the next, verified through multi-source identity resolution."
                ),
                "recommendation": "No action needed.",
                "evidence": (
                    f"All {chain_len - 1} consecutive transaction pairs verified: "
                    f"buyer→seller identity links confirmed."
                ),
                "source": "deterministic",
            })

        return checks

    # ── Serialization ──

    def to_dict(self) -> list[dict]:
        """JSON-serializable representation of all identity clusters."""
        return [c.to_dict() for c in self.clusters]

    def get_summary(self) -> str:
        """One-line summary for progress updates."""
        if not self.clusters:
            return "No person mentions found"
        return (
            f"{len(self.clusters)} unique person(s) identified "
            f"from {len(self.mentions)} mentions across "
            f"{len(set(m.source_file for m in self.mentions))} documents"
        )

    def get_llm_context(self) -> str:
        """Build a compact identity summary for LLM prompt injection.

        Format::

            IDENTITY CLUSTERS (from multi-source analysis):
            - PERSON_1: "K. Rajesh Kumar" — appears as: Buyer (Sale Deed),
              Claimant (EC), Owner (Patta) — confidence: 88% (HIGH)
        """
        if not self.clusters:
            return ""

        lines = ["IDENTITY CLUSTERS (from multi-source analysis):"]
        for c in self.clusters:
            roles_desc = []
            for m in c.mentions:
                role_label = _ROLE_LABELS.get(m.role, m.role)
                roles_desc.append(f"{role_label} ({m.source_type})")
            # Deduplicate while preserving order
            seen = set()
            unique_roles = []
            for rd in roles_desc:
                if rd not in seen:
                    seen.add(rd)
                    unique_roles.append(rd)

            band = _confidence_band(c.confidence)
            lines.append(
                f"- {c.cluster_id}: \"{c.consensus_name}\" — appears as: "
                f"{', '.join(unique_roles)} — confidence: {c.confidence:.0%} ({band})"
            )

        return "\n".join(lines)
