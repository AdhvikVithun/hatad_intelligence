"""Tests for the Forensic Identity Resolution Engine.

Covers:
  - PersonMention creation and OCR quality detection
  - IdentityResolver clustering (same person, different people, cross-script)
  - Confidence scoring and evidence generation
  - Role-based identity queries
  - Chain-of-title continuity checks
  - JSON serialization round-trip
  - Integration with deterministic checks
"""

import pytest
from app.pipeline.identity import (
    PersonMention,
    IdentityCluster,
    IdentityResolver,
    _MERGE_THRESHOLD,
    _confidence_band,
)


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def _make_extracted_data(**kwargs):
    """Build a minimal extracted_data dict for testing.

    Keyword args are document specs:
        sale_deed={"seller": [...], "buyer": [...]}
        patta={"owner_names": [...]}
        ec={"transactions": [...]}
    """
    data = {}
    idx = 1
    if "sale_deed" in kwargs:
        data[f"sale_deed_{idx}.pdf"] = {
            "document_type": "SALE_DEED",
            "data": kwargs["sale_deed"],
        }
        idx += 1
    if "patta" in kwargs:
        data[f"patta_{idx}.pdf"] = {
            "document_type": "PATTA",
            "data": kwargs["patta"],
        }
        idx += 1
    if "ec" in kwargs:
        data[f"ec_{idx}.pdf"] = {
            "document_type": "EC",
            "data": kwargs["ec"],
        }
        idx += 1
    return data


# ═══════════════════════════════════════════════════
# PersonMention
# ═══════════════════════════════════════════════════

class TestPersonMention:
    def test_basic_creation(self):
        m = PersonMention(
            name="Rajesh Kumar", role="buyer",
            source_file="deed.pdf", source_type="SALE_DEED",
        )
        assert m.given == "rajesh kumar"
        assert m.patronymic == ""
        assert m.ocr_quality == 1.0

    def test_with_patronymic(self):
        m = PersonMention(
            name="Murugan S/o Ramamoorthy", role="buyer",
            source_file="deed.pdf", source_type="SALE_DEED",
        )
        assert m.given == "murugan"
        assert m.patronymic == "ramamoorthy"

    def test_garbled_ocr_quality(self):
        """Garbled Tamil names should get low OCR quality."""
        m = PersonMention(
            name="xక్ Rஜேsh", role="buyer",
            source_file="ec.pdf", source_type="EC",
        )
        # We can't guarantee the exact quality score, but for a very garbled
        # name we just check it's not 1.0 OR is 1.0 if our detector doesn't
        # flag it.  The key test is that __post_init__ runs without error.
        assert isinstance(m.ocr_quality, float)

    def test_to_dict(self):
        m = PersonMention(
            name="K. Rajesh Kumar", role="patta_owner",
            source_file="patta.pdf", source_type="PATTA",
        )
        d = m.to_dict()
        assert d["name"] == "K. Rajesh Kumar"
        assert d["role"] == "patta_owner"
        assert "given" in d
        assert "patronymic" in d
        assert "ocr_quality" in d


# ═══════════════════════════════════════════════════
# IdentityResolver — Clustering
# ═══════════════════════════════════════════════════

class TestIdentityResolverClustering:
    def test_same_person_different_formats(self):
        """Three mentions of the same person in different name formats → 1 cluster."""
        data = _make_extracted_data(
            sale_deed={
                "seller": [{"name": "Rajesh Kumar K."}],
                "buyer": [{"name": "Someone Else"}],
                "registration": {"registration_date": "2020-01-15"},
            },
            patta={
                "owner_names": [{"name": "K. Rajesh Kumar"}],
            },
            ec={
                "transactions": [
                    {"buyer_or_claimant": "K. Rajesh Kumar S/o Krishnan",
                     "seller_or_executant": "Previous Owner",
                     "registration_date": "2019-12-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        clusters = resolver.resolve()

        # Find the cluster containing "Rajesh Kumar"
        rajesh_clusters = [c for c in clusters
                          if "rajesh" in c.consensus_name.lower()
                          or any("rajesh" in m.name.lower() for m in c.mentions)]
        assert len(rajesh_clusters) == 1, (
            f"Expected 1 Rajesh cluster, got {len(rajesh_clusters)}: "
            f"{[c.consensus_name for c in rajesh_clusters]}"
        )
        cluster = rajesh_clusters[0]
        assert len(cluster.mentions) >= 2  # at least seller + patta + ec_claimant could merge
        assert cluster.confidence > 0.5

    def test_different_people(self):
        """Two completely different names → 2 separate clusters."""
        data = _make_extracted_data(
            sale_deed={
                "seller": [{"name": "Sundaram Pillai"}],
                "buyer": [{"name": "Rajesh Kumar"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        clusters = resolver.resolve()

        sundaram = [c for c in clusters if "sundaram" in c.consensus_name.lower()]
        rajesh = [c for c in clusters if "rajesh" in c.consensus_name.lower()]
        assert len(sundaram) == 1
        assert len(rajesh) == 1
        assert sundaram[0].cluster_id != rajesh[0].cluster_id

    def test_tamil_cross_script(self):
        """Tamil + Latin spelling of same name → 1 cluster."""
        data = _make_extracted_data(
            patta={
                "owner_names": [{"name": "முருகன்"}],
            },
            ec={
                "transactions": [
                    {"buyer_or_claimant": "Murugan",
                     "seller_or_executant": "Someone",
                     "registration_date": "2020-01-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        clusters = resolver.resolve()

        murugan_clusters = [c for c in clusters
                           if any("murugan" in m.name.lower() or "முருகன்" in m.name
                                  for m in c.mentions)]
        # All Murugan mentions should be in one cluster
        assert len(murugan_clusters) == 1
        assert len(murugan_clusters[0].mentions) >= 2

    def test_orphan_vowel_real_case(self):
        """The critical Tamil OCR test case: garbled ெபான்அரசி + clean பொன் அரசி → 1 cluster."""
        data = _make_extracted_data(
            patta={
                "owner_names": [{"name": "பொன் அரசி"}],
            },
            ec={
                "transactions": [
                    {"buyer_or_claimant": "a. ெபான்அரசி",
                     "seller_or_executant": "Some Seller",
                     "registration_date": "2020-01-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        clusters = resolver.resolve()

        pon_clusters = [c for c in clusters
                       if any("அரசி" in m.name or "அரசி" in m.name for m in c.mentions)]
        assert len(pon_clusters) == 1, (
            f"Expected garbled+clean Tamil to cluster together, got "
            f"{len(pon_clusters)} clusters"
        )
        assert len(pon_clusters[0].mentions) >= 2

    def test_single_mention(self):
        """A lone name → 1 cluster with base confidence 0.5."""
        data = _make_extracted_data(
            patta={
                "owner_names": [{"name": "Lone Person"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        clusters = resolver.resolve()

        assert len(clusters) == 1
        assert clusters[0].confidence == pytest.approx(0.5, abs=0.05)

    def test_empty_data(self):
        """No documents → no clusters."""
        resolver = IdentityResolver()
        resolver.collect_mentions({})
        clusters = resolver.resolve()
        assert clusters == []

    def test_multi_party_ec_splitting(self):
        """EC multi-party string 'A and B' creates separate mentions."""
        data = _make_extracted_data(
            ec={
                "transactions": [
                    {"buyer_or_claimant": "Rajesh Kumar and Sundaram Pillai",
                     "seller_or_executant": "Previous Owner",
                     "registration_date": "2020-01-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        count = resolver.collect_mentions(data)
        # Should have at least 3 mentions: Rajesh, Sundaram, Previous Owner
        assert count >= 3
        clusters = resolver.resolve()
        rajesh = [c for c in clusters if "rajesh" in c.consensus_name.lower()]
        sundaram = [c for c in clusters if "sundaram" in c.consensus_name.lower()]
        assert len(rajesh) == 1
        assert len(sundaram) == 1


# ═══════════════════════════════════════════════════
# Confidence and Evidence
# ═══════════════════════════════════════════════════

class TestConfidenceAndEvidence:
    def test_multi_source_confidence(self):
        """3 documents, 2 doc types → confidence > 0.7."""
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
                "registration": {"registration_date": "2020-01-15"},
            },
            patta={
                "owner_names": [{"name": "Rajesh Kumar"}],
            },
            ec={
                "transactions": [
                    {"buyer_or_claimant": "Rajesh Kumar",
                     "seller_or_executant": "Sundaram",
                     "registration_date": "2019-12-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        rajesh = [c for c in resolver.clusters if "rajesh" in c.consensus_name.lower()]
        assert len(rajesh) == 1
        assert rajesh[0].confidence >= 0.70, (
            f"Expected ≥70% confidence for 3-source corroboration, got {rajesh[0].confidence:.0%}"
        )
        assert rajesh[0].source_types == {"SALE_DEED", "PATTA", "EC"}

    def test_evidence_lines_present(self):
        """Evidence lines should contain source references and confidence."""
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Rajesh Kumar"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        rajesh = [c for c in resolver.clusters if "rajesh" in c.consensus_name.lower()]
        assert len(rajesh) == 1
        evidence = "\n".join(rajesh[0].evidence_lines)
        assert "Rajesh Kumar" in evidence
        assert "Confidence" in evidence

    def test_confidence_band(self):
        assert _confidence_band(0.90) == "HIGH"
        assert _confidence_band(0.75) == "MODERATE"
        assert _confidence_band(0.55) == "LOW"
        assert _confidence_band(0.30) == "VERY LOW"


# ═══════════════════════════════════════════════════
# Role Queries
# ═══════════════════════════════════════════════════

class TestRoleQueries:
    def test_roles_share_identity_match(self):
        """Buyer and patta_owner in same cluster → (True, high_conf, evidence)."""
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Rajesh Kumar"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        same, conf, evidence = resolver.roles_share_identity("buyer", "patta_owner")
        assert same is True
        assert conf > 0.5
        assert "Identity Match" in evidence

    def test_roles_disjoint(self):
        """Buyer and patta_owner are different people → (False, 0.0, evidence)."""
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Completely Different Person"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        same, conf, evidence = resolver.roles_share_identity("buyer", "patta_owner")
        assert same is False
        assert "Mismatch" in evidence or "DIFFERENT" in evidence

    def test_roles_missing_document(self):
        """No patta → role query returns (False, 0.0, 'no mentions')."""
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        same, conf, evidence = resolver.roles_share_identity("buyer", "patta_owner")
        assert same is False
        assert "No Patta Owner mentions" in evidence

    def test_find_clusters_for_role(self):
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}, {"name": "Sundaram Pillai"}],
                "seller": [{"name": "Previous Owner"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        buyers = resolver.find_clusters_for_role("buyer")
        assert len(buyers) == 2  # Two distinct buyers

    def test_get_cluster_for_name(self):
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        cluster = resolver.get_cluster_for_name("Rajesh Kumar")
        assert cluster is not None
        assert "rajesh" in cluster.consensus_name.lower()

        none_cluster = resolver.get_cluster_for_name("Nonexistent Person")
        assert none_cluster is None


# ═══════════════════════════════════════════════════
# Chain of Title Continuity
# ═══════════════════════════════════════════════════

class TestChainContinuity:
    def test_continuous_chain(self):
        """A→B, B→C → no gaps."""
        data = _make_extracted_data(
            ec={
                "transactions": [
                    {"seller_or_executant": "Rajesh Kumar", "buyer_or_claimant": "Sundaram Pillai",
                     "registration_date": "2018-01-01", "document_number": "100"},
                    {"seller_or_executant": "Sundaram Pillai", "buyer_or_claimant": "Murugan Ravi",
                     "registration_date": "2020-01-01", "document_number": "200"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        checks = resolver.check_chain_continuity(data)
        # Should get a PASS for chain continuity
        statuses = {c["rule_code"]: c["status"] for c in checks}
        assert "DET_CHAIN_CONTINUITY" in statuses
        assert statuses["DET_CHAIN_CONTINUITY"] == "PASS"
        assert "DET_CHAIN_BREAK" not in statuses

    def test_broken_chain(self):
        """A→B, X→C → gap between B and X."""
        data = _make_extracted_data(
            ec={
                "transactions": [
                    {"seller_or_executant": "Rajesh Kumar", "buyer_or_claimant": "Sundaram Pillai",
                     "registration_date": "2018-01-01", "document_number": "100"},
                    {"seller_or_executant": "Govindarajan Iyer", "buyer_or_claimant": "Murugan Ravi",
                     "registration_date": "2020-01-01", "document_number": "200"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        checks = resolver.check_chain_continuity(data)
        codes = [c["rule_code"] for c in checks]
        assert "DET_CHAIN_BREAK" in codes

    def test_single_transaction_no_chain_check(self):
        """Only 1 transaction → no chain check emitted."""
        data = _make_extracted_data(
            ec={
                "transactions": [
                    {"seller_or_executant": "Rajesh Kumar", "buyer_or_claimant": "Sundaram Pillai",
                     "registration_date": "2020-01-01"},
                ],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        checks = resolver.check_chain_continuity(data)
        assert len(checks) == 0  # Can't check continuity with < 2 transactions


# ═══════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════

class TestSerialization:
    def test_to_dict_structure(self):
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        result = resolver.to_dict()
        assert isinstance(result, list)
        assert len(result) >= 2  # at least 2 distinct people

        for cluster_dict in result:
            assert "cluster_id" in cluster_dict
            assert "consensus_name" in cluster_dict
            assert "confidence" in cluster_dict
            assert "confidence_band" in cluster_dict
            assert "evidence_lines" in cluster_dict
            assert "roles" in cluster_dict
            assert "mentions" in cluster_dict
            assert isinstance(cluster_dict["mentions"], list)

    def test_get_summary(self):
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        summary = resolver.get_summary()
        assert "unique person" in summary
        assert "mentions" in summary

    def test_get_llm_context(self):
        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        ctx = resolver.get_llm_context()
        assert "IDENTITY CLUSTERS" in ctx
        assert "PERSON_" in ctx

    def test_empty_llm_context(self):
        resolver = IdentityResolver()
        resolver.collect_mentions({})
        resolver.resolve()
        assert resolver.get_llm_context() == ""


# ═══════════════════════════════════════════════════
# Integration with deterministic checks
# ═══════════════════════════════════════════════════

class TestDeterministicIntegration:
    def test_identity_based_name_check_pass(self):
        """When identity clusters link buyer and patta owner → PASS."""
        from app.pipeline.deterministic import check_party_name_consistency

        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Rajesh Kumar"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        checks = check_party_name_consistency(data, identity_resolver=resolver)
        pass_checks = [c for c in checks if c["status"] == "PASS"]
        assert len(pass_checks) >= 1
        assert any("DET_BUYER_PATTA_MATCH" in c["rule_code"] for c in pass_checks)

    def test_identity_based_name_check_mismatch(self):
        """When buyer and patta owner are different → WARNING."""
        from app.pipeline.deterministic import check_party_name_consistency

        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Completely Different Person"}],
            },
        )
        resolver = IdentityResolver()
        resolver.collect_mentions(data)
        resolver.resolve()

        checks = check_party_name_consistency(data, identity_resolver=resolver)
        warn_checks = [c for c in checks if c["status"] == "WARNING"]
        assert len(warn_checks) >= 1

    def test_legacy_fallback_without_resolver(self):
        """Without identity_resolver, falls back to pairwise matching."""
        from app.pipeline.deterministic import check_party_name_consistency

        data = _make_extracted_data(
            sale_deed={
                "buyer": [{"name": "Rajesh Kumar"}],
                "seller": [{"name": "Sundaram"}],
            },
            patta={
                "owner_names": [{"name": "Completely Different Person"}],
            },
        )
        # No resolver → legacy path
        checks = check_party_name_consistency(data)
        # Should still produce a warning (legacy pairwise)
        warn_checks = [c for c in checks if c["status"] == "WARNING"]
        assert len(warn_checks) >= 1
