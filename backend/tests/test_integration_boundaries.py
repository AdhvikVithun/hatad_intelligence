"""Integration-boundary tests — exercise REAL libraries instead of mocks.

These tests target the exact boundary points where production bugs appeared:

1. ChromaDB + numpy arrays: Real ChromaDB returns numpy arrays; bare
   ``if array:`` raises ValueError.  These tests use a REAL in-memory
   ChromaDB client and verify our code never does bare truthiness checks
   on numpy arrays.

2. Classifier repetition guards: Tamil Patta text with repeated tokens
   (non-adjacent) and FMB text with long repeated phrases.  These tests
   feed real-world-shaped text through _collapse_repetitions,
   _dedup_high_freq_tokens, and the defensive post-processing.

3. LLM repetition detection: _is_repetitive_output() must catch both
   short-token loops (Tamil word ×50) and long-phrase loops (FMB survey
   description ×10).

4. RAG store end-to-end: Index real text → query with real embeddings →
   verify MMR re-ranking returns correct chunks without numpy crashes.

5. Report generation: Render the HTML template with realistic context
   to catch Jinja2 attribute/filter errors that only appear with
   production-shaped data.
"""

import asyncio
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

# ═══════════════════════════════════════════════════
# 1. ChromaDB + numpy — real library, real arrays
# ═══════════════════════════════════════════════════

class TestChromaDBNumpyBoundary:
    """Verify that ChromaDB's numpy return values are handled safely."""

    @pytest.fixture(autouse=True)
    def _setup_chroma(self, tmp_path):
        """Create a real ChromaDB client with test data."""
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(
            path=str(tmp_path / "chroma_test"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="test_numpy",
            metadata={"hnsw:space": "cosine"},
        )
        # Insert 5 documents with known 8-dim embeddings
        self.dim = 8
        docs = [
            "Encumbrance Certificate for S.F. No 311/1 Chromepet",
            "Sale deed registered at Tambaram SRO in 2020",
            "Patta issued to Lakshmi W/o Senthil for survey 311/1",
            "Property located in Chromepet village Kancheepuram district",
            "Mortgage document for Raman S/o Krishnan",
        ]
        embeddings = []
        for i in range(len(docs)):
            # Create simple orthogonal-ish embeddings
            vec = [0.0] * self.dim
            vec[i % self.dim] = 1.0
            vec[(i + 1) % self.dim] = 0.5
            # Normalize
            norm = math.sqrt(sum(x * x for x in vec))
            vec = [x / norm for x in vec]
            embeddings.append(vec)

        self.collection.upsert(
            ids=[f"doc_{i}" for i in range(len(docs))],
            embeddings=embeddings,
            documents=docs,
            metadatas=[{"filename": f"doc{i}.pdf", "page_number": 1, "chunk_index": 0}
                       for i in range(len(docs))],
        )

    def test_query_returns_numpy_arrays(self):
        """ChromaDB actually returns numpy arrays for embeddings."""
        query_vec = [0.0] * self.dim
        query_vec[0] = 1.0
        norm = math.sqrt(sum(x * x for x in query_vec))
        query_vec = [x / norm for x in query_vec]

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        # Verify embeddings are numpy arrays (the root cause of the bug)
        embeds = results["embeddings"]
        assert embeds is not None
        assert len(embeds) > 0
        assert len(embeds[0]) > 0

        # The actual type check — this is what caused the production bug
        first_embed = embeds[0][0]
        assert hasattr(first_embed, '__len__'), "Embedding should be array-like"

        # THIS is the exact pattern that crashed in production:
        # ``if embeds:`` on a numpy array → ValueError
        # Our fix uses ``embeds is not None and len(embeds) > 0``
        # Verify the safe pattern works:
        embeds_raw = results.get("embeddings")
        safe_check = (
            embeds_raw is not None
            and len(embeds_raw) > 0
            and len(embeds_raw[0]) > 0
        )
        assert safe_check is True

    def test_bare_truthiness_raises_on_numpy(self):
        """Prove that bare ``if array:`` raises ValueError on multi-element numpy arrays."""
        arr = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="ambiguous"):
            bool(arr)

    def test_none_embedding_safe(self):
        """query without 'embeddings' in include should return None safely."""
        query_vec = [0.0] * self.dim
        query_vec[0] = 1.0
        norm = math.sqrt(sum(x * x for x in query_vec))
        query_vec = [x / norm for x in query_vec]

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=2,
            include=["documents", "distances"],  # No embeddings!
        )
        embeds_raw = results.get("embeddings")
        # Should be None or empty — safe to check with ``is None``
        safe = embeds_raw is None or len(embeds_raw) == 0
        assert safe or all(e is None for e in embeds_raw)

    def test_empty_collection_query(self):
        """Querying an empty collection should not crash."""
        empty_col = self.client.get_or_create_collection(
            name="test_empty",
            metadata={"hnsw:space": "cosine"},
        )
        query_vec = [0.0] * self.dim
        query_vec[0] = 1.0
        norm = math.sqrt(sum(x * x for x in query_vec))
        query_vec = [x / norm for x in query_vec]

        results = empty_col.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        # ChromaDB may return empty lists or None
        docs = results.get("documents", [[]])
        assert docs == [[]] or docs is None


# ═══════════════════════════════════════════════════
# 2. RAG Store end-to-end (real ChromaDB, no mocks)
# ═══════════════════════════════════════════════════

class TestRAGStoreRealChroma:
    """Index and query through RAGStore with real ChromaDB + embeddings."""

    @pytest.fixture
    def rag_store(self, tmp_path, monkeypatch):
        """Create a RAGStore with a temporary ChromaDB directory."""
        from app.pipeline.rag_store import RAGStore
        # Point CHROMA_DIR to temp
        monkeypatch.setattr("app.pipeline.rag_store.CHROMA_DIR", tmp_path / "vectordb")
        (tmp_path / "vectordb").mkdir()
        sid = uuid.uuid4().hex[:8]
        store = RAGStore(sid)
        return store

    @staticmethod
    def _make_embed_fn(dim: int = 16):
        """Create a deterministic embedding function (no Ollama needed).
        
        Returns consistent embeddings based on text hash so similar 
        queries return similar vectors.
        """
        import hashlib

        async def embed_fn(texts: list[str]) -> list[list[float]]:
            embeddings = []
            for text in texts:
                # Hash-based deterministic embedding
                h = hashlib.sha256(text.encode()).digest()
                vec = [float(b) / 255.0 for b in h[:dim]]
                norm = math.sqrt(sum(x * x for x in vec))
                vec = [x / norm for x in vec]
                embeddings.append(vec)
            return embeddings

        return embed_fn

    def test_index_and_query_no_crash(self, rag_store):
        """Index documents → query → no numpy crash."""
        pages = [
            {"page_number": 1, "text": "Encumbrance Certificate EC-2024-001 for SF No 311/1 "
                                        "Chromepet village Tambaram taluk. Period: 2010-2025. "
                                        "Transaction 1: Sale by Raman to Muthu dated 2012."},
            {"page_number": 2, "text": "Transaction 2: Sale by Muthu to Lakshmi dated 2020. "
                                        "Document number 5678/2020 at Tambaram SRO. "
                                        "Consideration Rs 45,00,000."},
        ]

        embed_fn = self._make_embed_fn(dim=16)

        # Index
        count = asyncio.get_event_loop().run_until_complete(
            rag_store.index_document("ec.pdf", pages, embed_fn)
        )
        assert count > 0

        # Query — this is where the numpy bug would crash
        query_text = "Who is the seller in 2012 transaction?"
        query_embed = asyncio.get_event_loop().run_until_complete(
            embed_fn([query_text])
        )
        chunks = rag_store.query_mmr_sync(
            question_embedding=query_embed[0],
            n_results=2,
            lambda_param=0.7,
            query_text=query_text,
        )
        assert isinstance(chunks, list)
        # Should return results from indexed data
        assert len(chunks) > 0
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'filename') for c in chunks)

    def test_query_with_embeddings_returned(self, rag_store):
        """Verify MMR works when ChromaDB returns embedding arrays (the numpy path)."""
        pages = [
            {"page_number": 1, "text": "Land survey document for plot 42A in Madurai district. " * 5},
            {"page_number": 2, "text": "Registration details for property transfer in 2023. " * 5},
            {"page_number": 3, "text": "Patta issued to Venkatesh S/o Ramasamy in Madurai. " * 5},
        ]
        embed_fn = self._make_embed_fn(dim=16)

        asyncio.get_event_loop().run_until_complete(
            rag_store.index_document("multi.pdf", pages, embed_fn)
        )

        query_embed = asyncio.get_event_loop().run_until_complete(
            embed_fn(["property transfer registration"])
        )

        # This exercises the MMR path that iterates over numpy embeddings
        chunks = rag_store.query_mmr_sync(
            question_embedding=query_embed[0],
            n_results=3,
            lambda_param=0.7,
        )
        assert isinstance(chunks, list)
        # Verify no crash and valid output
        for chunk in chunks:
            assert isinstance(chunk.text, str)
            assert len(chunk.text) > 0
            assert chunk.score >= 0

    def test_empty_store_query(self, rag_store):
        """Querying an empty store should return [] not crash."""
        embed_fn = self._make_embed_fn(dim=16)
        query_embed = asyncio.get_event_loop().run_until_complete(
            embed_fn(["anything"])
        )
        chunks = rag_store.query_mmr_sync(
            question_embedding=query_embed[0],
            n_results=3,
        )
        assert chunks == []

    def test_format_evidence(self, rag_store):
        """format_evidence should handle real RetrievedChunk objects."""
        from app.pipeline.rag_store import RAGStore, RetrievedChunk

        chunks = [
            RetrievedChunk(
                text="Sale deed for 311/1 executed on 2020-06-20",
                filename="deed.pdf", page_number=1, chunk_index=0, score=0.12,
            ),
            RetrievedChunk(
                text="EC shows no encumbrances after 2020",
                filename="ec.pdf", page_number=3, chunk_index=1, score=0.25,
            ),
        ]
        evidence = RAGStore.format_evidence(chunks)
        assert "SOURCE EVIDENCE" in evidence
        assert "deed.pdf" in evidence
        assert "ec.pdf" in evidence


# ═══════════════════════════════════════════════════
# 3. Classifier repetition guards — real text patterns
# ═══════════════════════════════════════════════════

class TestClassifierRepetitionGuards:
    """Test _collapse_repetitions and _dedup_high_freq_tokens with real-world patterns."""

    def test_tamil_patta_repeated_token(self):
        """Tamil Patta text with 'மொத்தம்' repeated 50+ times (non-adjacent)."""
        from app.pipeline.classifier import _dedup_high_freq_tokens

        # Build realistic Tamil Patta text with repeated token
        base_words = ["நிலம்", "புல எண்", "311/1", "பரப்பு", "ஹெக்டேர்"]
        lines = []
        for i in range(60):
            row_words = base_words + ["மொத்தம்", str(i + 1)]  # 'மொத்தம்' in every row
            lines.append(" ".join(row_words))
        text = "\n".join(lines)

        result = _dedup_high_freq_tokens(text, max_occurrences=6)

        # Count occurrences of 'மொத்தம்' — should be capped
        count = result.split().count("மொத்தம்")
        assert count <= 6, f"Expected ≤6 occurrences, got {count}"
        # Text should be significantly shorter
        assert len(result) < len(text)

    def test_collapse_adjacent_repeats(self):
        """OCR artefact: '(R) (R) (R) ...' repeated 200 times."""
        from app.pipeline.classifier import _collapse_repetitions

        text = "(R) " * 200
        result = _collapse_repetitions(text, max_repeats=3)
        # Should collapse to at most 3 copies
        assert result.count("(R)") <= 3

    def test_fmb_long_phrase_in_dedup(self):
        """FMB text with long repeated phrase should be handled by dedup."""
        from app.pipeline.classifier import _dedup_high_freq_tokens

        # Each "row" has a long phrase token that repeats
        phrase = "Survey_No:_317_Scale:_1:1158_Area:_Hect_00_Ares_92"
        lines = [f"{phrase} row_{i}" for i in range(40)]
        text = " ".join(lines)

        result = _dedup_high_freq_tokens(text, max_occurrences=6)
        count = result.split().count(phrase)
        assert count <= 6

    def test_normal_text_not_affected(self):
        """Normal document text should pass through unchanged."""
        from app.pipeline.classifier import _collapse_repetitions, _dedup_high_freq_tokens

        text = (
            "This is a normal encumbrance certificate for property located in "
            "Chromepet village, Tambaram taluk, Chengalpattu district. "
            "The EC covers the period from 01-01-2010 to 31-12-2025. "
            "Transaction 1: Sale deed executed on 15-03-2012 by Raman to Muthu "
            "for a consideration of Rs 15,00,000. Document registered at SRO."
        )
        collapsed = _collapse_repetitions(text)
        deduped = _dedup_high_freq_tokens(collapsed)
        # Normal text should not be modified (or trivially modified)
        assert len(deduped) >= len(text) * 0.95  # Allow minimal change

    def test_key_identifiers_dedup_cap(self):
        """Verify that key_identifiers post-processing caps and deduplicates."""
        # Simulate what classify_document does after LLM returns
        ids = ["மொத்தம்", "311/1", "மொத்தம்", "EC-001", "மொத்தம்",
               "Chromepet", "மொத்தம்", "311/1", "Tambaram", "மொத்தம்",
               "SRO", "deed", "patta", "village"]
        seen = set()
        unique: list[str] = []
        for v in ids:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        result = unique[:10]

        # Should have no duplicates
        assert len(result) == len(set(result))
        # Should be capped at 10
        assert len(result) <= 10
        # 'மொத்தம்' should appear exactly once
        assert result.count("மொத்தம்") == 1

    def test_truncation_after_dedup(self):
        """Large text should be truncated to 4000 chars after dedup."""
        from app.pipeline.classifier import _collapse_repetitions, _dedup_high_freq_tokens

        # Create text that's 10K chars
        text = "word " * 2000  # ~10K chars
        # Pipeline: collapse → dedup → truncate
        text = _collapse_repetitions(text)
        text = _dedup_high_freq_tokens(text, max_occurrences=6)
        # In classify_document, text gets truncated:
        _CLASSIFY_MAX_CHARS = 4000
        if len(text) > _CLASSIFY_MAX_CHARS:
            text = text[:_CLASSIFY_MAX_CHARS]
        assert len(text) <= _CLASSIFY_MAX_CHARS


# ═══════════════════════════════════════════════════
# 4. LLM repetition detection — realistic degenerate outputs
# ═══════════════════════════════════════════════════

class TestLLMRepetitionDetection:
    """Test _is_repetitive_output with realistic pathological LLM outputs."""

    def test_short_token_10x(self):
        """Short Tamil token repeated 10+ times → detected."""
        from app.pipeline.llm_client import _is_repetitive_output

        text = '"மொத்தம்", ' * 15
        assert _is_repetitive_output(text) is True

    def test_short_token_50x(self):
        """Short token repeated 50 times — clearly degenerate."""
        from app.pipeline.llm_client import _is_repetitive_output

        text = '"total", ' * 50
        assert _is_repetitive_output(text) is True

    def test_long_phrase_4x(self):
        """Long FMB phrase repeated 4+ times → detected by _LONG_REPEAT_RE."""
        from app.pipeline.llm_client import _is_repetitive_output

        phrase = "Survey No: 317, Scale: 1: 1158 mm, Area: Hect 00 Ares 92 Sqm 50"
        text = phrase * 5
        assert _is_repetitive_output(text) is True

    def test_long_phrase_in_json_array(self):
        """Long phrase inside a JSON array with commas — still detected."""
        from app.pipeline.llm_client import _is_repetitive_output

        phrase = "Survey No: 317, Scale: 1: 1158 mm, Area: Hect 00 Ares 92 Sqm 50"
        # Simulate what the LLM actually produces
        items = [f'"{phrase}"' for _ in range(6)]
        text = '{"key_identifiers": [' + ", ".join(items) + ']}'
        assert _is_repetitive_output(text) is True

    def test_normal_json_not_flagged(self):
        """Normal LLM JSON output should NOT be flagged as repetitive."""
        from app.pipeline.llm_client import _is_repetitive_output

        text = '''{"document_type": "EC", "confidence": 0.95, "language": "Tamil",
        "key_identifiers": ["EC-2024-001", "S.F.No. 311/1", "Chromepet",
        "Raman", "Muthu", "Lakshmi"]}'''
        assert _is_repetitive_output(text) is False

    def test_normal_long_ec_not_flagged(self):
        """A legitimate long EC extraction with many transactions should not be flagged."""
        from app.pipeline.llm_client import _is_repetitive_output

        transactions = []
        for i in range(30):
            transactions.append(
                f'{{"row_number": {i+1}, "date": "2020-0{(i%9)+1}-15", '
                f'"transaction_type": "Sale", "seller": "Person_{i}", '
                f'"buyer": "Person_{i+1}", "amount": "{(i+1)*100000}"}}'
            )
        text = '{"transactions": [' + ",\n".join(transactions) + ']}'
        assert _is_repetitive_output(text) is False

    def test_empty_and_short_not_flagged(self):
        """Empty or very short strings should return False."""
        from app.pipeline.llm_client import _is_repetitive_output

        assert _is_repetitive_output("") is False
        assert _is_repetitive_output("short") is False
        assert _is_repetitive_output(None) is False

    def test_regex_patterns_compiled(self):
        """Verify both regex patterns exist and are compiled."""
        from app.pipeline.llm_client import _REPETITIVE_RE, _LONG_REPEAT_RE
        import re

        assert isinstance(_REPETITIVE_RE, re.Pattern)
        assert isinstance(_LONG_REPEAT_RE, re.Pattern)


# ═══════════════════════════════════════════════════
# 5. Schema maxItems/maxLength enforcement
# ═══════════════════════════════════════════════════

class TestSchemaConstraints:
    """Verify all schema array fields have maxItems and string fields have maxLength."""

    def test_classify_schema_has_max_items(self):
        from app.pipeline.schemas import CLASSIFY_SCHEMA

        ki = CLASSIFY_SCHEMA["properties"]["key_identifiers"]
        assert "maxItems" in ki, "key_identifiers must have maxItems"
        assert ki["maxItems"] <= 20, "maxItems should be reasonable"

    def test_classify_key_identifiers_max_length(self):
        from app.pipeline.schemas import CLASSIFY_SCHEMA

        items = CLASSIFY_SCHEMA["properties"]["key_identifiers"]["items"]
        assert "maxLength" in items, "key_identifiers items must have maxLength"
        assert items["maxLength"] <= 200

    def test_all_schemas_arrays_have_max_items(self):
        """Scan ALL exported schemas for any array without maxItems."""
        from app.pipeline import schemas
        import json

        def _check_schema(obj, path=""):
            """Recursively check all array-typed fields for maxItems."""
            violations = []
            if isinstance(obj, dict):
                if obj.get("type") == "array":
                    if "maxItems" not in obj:
                        violations.append(f"{path}: array without maxItems")
                for key, val in obj.items():
                    violations.extend(_check_schema(val, f"{path}.{key}"))
            elif isinstance(obj, list):
                for i, val in enumerate(obj):
                    violations.extend(_check_schema(val, f"{path}[{i}]"))
            return violations

        # Inspect all module-level dicts that look like schemas
        for name in dir(schemas):
            obj = getattr(schemas, name)
            if isinstance(obj, dict) and "properties" in obj:
                violations = _check_schema(obj, name)
                assert not violations, f"Schema violations:\n" + "\n".join(violations)


# ═══════════════════════════════════════════════════
# 6. Report HTML template rendering
# ═══════════════════════════════════════════════════

class TestReportTemplateRendering:
    """Render the HTML template with production-shaped data — catch Jinja2 errors."""

    @pytest.fixture
    def template_env(self):
        """Load the real Jinja2 template environment."""
        import jinja2
        from jinja2 import ChainableUndefined
        from markupsafe import Markup
        from app.config import TEMPLATES_DIR
        import mistune

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
            undefined=ChainableUndefined,
        )
        _md = mistune.create_markdown(plugins=['table', 'strikethrough'])
        env.filters['markdown'] = lambda text: Markup(_md(text)) if text else Markup('')
        return env

    def _make_context(self, *, include_docs=True, include_checks=True,
                      include_chain=True, include_identity=True):
        """Build a production-shaped template context."""
        docs = []
        if include_docs:
            docs = [
                {
                    "filename": "ec.pdf",
                    "classification": {"confidence": "0.95"},  # String confidence — tests |float
                    "doc_type": "EC",
                    "pages": 3,
                },
                {
                    "filename": "sale_deed.pdf",
                    "classification": {"confidence": 0.92},  # Numeric — should also work
                    "doc_type": "SALE_DEED",
                    "pages": 5,
                },
                {
                    "filename": "patta.pdf",
                    # No 'classification' key — tests the attribute guard
                    "doc_type": "PATTA",
                    "pages": 2,
                },
            ]

        checks = []
        if include_checks:
            checks = [
                {
                    "check_name": "EC Period Coverage",
                    "status": "PASS",
                    "severity": "HIGH",
                    "explanation": "EC covers the required 13-year period.",
                    "group": "EC Hygiene",
                    "source": "deterministic",
                    "rule_code": "DET_EC_PERIOD",
                },
                {
                    "check_name": "Stamp Duty",
                    "status": "FAIL",
                    "severity": "CRITICAL",
                    "explanation": "Stamp duty is 2% below the statutory rate.",
                    "group": "Sale Deed",
                    "source": "deterministic",
                    "rule_code": "DET_STAMP_DUTY",
                },
                {
                    "check_name": "Survey Consistency",
                    "status": "WARNING",
                    "severity": "MEDIUM",
                    "explanation": "Minor survey number discrepancy between EC and Patta.",
                    "group": "Cross-Document",
                    "source": "llm",
                    "rule_code": "LLM_SURVEY_CHECK",
                },
                {
                    "check_name": "Poramboke Detection",
                    "status": "PASS",
                    "severity": "CRITICAL",
                    "explanation": "No poramboke/government land indicators found.",
                    "group": "Cross-Document",
                    "source": "llm",
                    "rule_code": "LLM_PORAMBOKE",
                },
            ]

        chain = []
        if include_chain:
            chain = [
                {"sequence": "1", "from": "State (Original Grant)", "to": "Raman S/o Krishnan",
                 "date": "Pre-2010", "transaction_type": "Unknown", "document_number": "—",
                 "consideration": "—", "valid": True},
                {"sequence": "2", "from": "Raman S/o Krishnan", "to": "Muthu D/o Perumal",
                 "date": "15-03-2012", "transaction_type": "Sale", "document_number": "1234/2012",
                 "consideration": "15,00,000", "valid": True},
                {"sequence": "3", "from": "Muthu D/o Perumal", "to": "Lakshmi W/o Senthil",
                 "date": "20-06-2020", "transaction_type": "Sale", "document_number": "5678/2020",
                 "consideration": "45,00,000", "valid": True},
            ]

        identity_clusters = []
        if include_identity:
            identity_clusters = [
                {
                    "consensus_name": "Muthu D/o Perumal",
                    "cluster_id": "1",  # String — tests |int filter
                    "variants": ["Muthu", "Muthu Perumal", "முத்து"],
                    "source_files": ["ec.pdf", "sale_deed.pdf"],
                    "mention_count": 5,
                },
            ]

        return {
            "session_id": "test123",
            "generated_at": "20 February 2026, 10:00 AM",
            "risk_score": 72,
            "risk_band": "MEDIUM",
            "risk_color": "#ffaa00",
            "risk_label": "Medium Risk",
            "executive_summary": "Property has clear title chain with minor stamp duty concern.",
            "documents": docs,
            "prop": {
                "survey_no": "S.F.No. 311/1",
                "survey_numbers": ["311/1"],
                "village": "Chromepet",
                "taluk": "Tambaram",
                "district": "Chengalpattu",
                "extent": "2400 sq.ft",
                "owner": "Lakshmi W/o Senthil",
                "patta_number": "12345",
                "ec_period": "2010–2024",
                "land_classification": "Dry Land",
                "sro": "Tambaram",
                "consideration_amount": "₹45,00,000",
                "property_type": "Residential Plot",
                "owners": [{"name": "Lakshmi W/o Senthil"}],
            },
            "all_checks": checks,
            "group_bars": [
                {"name": "EC Hygiene", "pct": 100},
                {"name": "Sale Deed", "pct": 40},
                {"name": "Cross-Document", "pct": 75},
            ],
            "total_checks": len(checks),
            "total_pass": sum(1 for c in checks if c.get("status") == "PASS"),
            "total_fail": sum(1 for c in checks if c.get("status") == "FAIL"),
            "total_warn": sum(1 for c in checks if c.get("status") == "WARNING"),
            "total_na": 0,
            "total_raw_names": 3 if include_identity else 0,
            "chain_of_title": chain,
            "red_flags": ["Stamp duty underpayment detected"],
            "recommendations": ["Verify stamp duty payment with SRO records"],
            "missing_documents": ["FMB (Field Measurement Book)"],
            "narrative_report": "The property at S.F.No. 311/1, Chromepet village, "
                                "shows a clear chain of title from Raman to Lakshmi.",
            "identity_clusters": identity_clusters,
            "active_encumbrances": [],
            "sale_deed": None,
        }

    def test_render_full_context(self, template_env):
        """Template renders without error with full production-shaped data."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        html = template.render(**ctx)
        assert len(html) > 500
        assert "TEST123" in html  # session_id[:8]|upper
        assert "Chromepet" in html

    def test_render_empty_checks(self, template_env):
        """Template handles empty checks list without crashing."""
        template = template_env.get_template("report.html")
        ctx = self._make_context(include_checks=False)
        html = template.render(**ctx)
        assert len(html) > 100

    def test_render_no_chain(self, template_env):
        """Template handles empty chain_of_title without crashing."""
        template = template_env.get_template("report.html")
        ctx = self._make_context(include_chain=False)
        html = template.render(**ctx)
        assert len(html) > 100

    def test_render_doc_without_classification(self, template_env):
        """Document without 'classification' attribute should not crash template."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        # One doc explicitly has no classification — the actual bug we fixed
        has_no_class = [d for d in ctx["documents"] if "classification" not in d]
        assert len(has_no_class) > 0, "Test should include a doc without classification"
        html = template.render(**ctx)
        assert len(html) > 100

    def test_render_empty_identity_clusters(self, template_env):
        """Template handles empty identity clusters gracefully."""
        template = template_env.get_template("report.html")
        ctx = self._make_context(include_identity=False)
        html = template.render(**ctx)
        assert len(html) > 100

    def test_render_minimal_context(self, template_env):
        """Template renders with absolutely minimal context."""
        template = template_env.get_template("report.html")
        ctx = self._make_context(
            include_docs=False,
            include_checks=False,
            include_chain=False,
            include_identity=False,
        )
        html = template.render(**ctx)
        assert len(html) > 100

    def test_render_string_typed_numbers(self, template_env):
        """Template handles string-typed sequence/cluster_id/confidence (the %d bug)."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        # Verify chain_of_title has string sequence values
        for link in ctx["chain_of_title"]:
            assert isinstance(link["sequence"], str), "Test data must use string sequence"
        # Verify identity_clusters has string cluster_id
        for cluster in ctx["identity_clusters"]:
            assert isinstance(cluster["cluster_id"], str), "Test data must use string cluster_id"
        # Verify classification confidence is string on first doc
        assert isinstance(ctx["documents"][0]["classification"]["confidence"], str)
        # This is the exact line that caused: %d format: a real number is required, not str
        html = template.render(**ctx)
        assert len(html) > 500
        # Verify the formatted values appear correctly
        assert "01" in html  # sequence "1" → "%02d" → "01"

    def test_render_with_sale_deed(self, template_env):
        """Template renders sale deed §06 section with full sale deed data."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        ctx["sale_deed"] = {
            "registration": {
                "document_number": "1234/2020",
                "registration_date": "20-06-2020",
                "execution_date": "15-06-2020",
                "sro": "Tambaram",
            },
            "financials": {
                "consideration_amount": "₹45,00,000",
                "guideline_value": "₹40,00,000",
                "stamp_duty": "₹3,15,000",
                "registration_fee": "₹45,000",
            },
            "sellers": [{"name": "Muthu D/o Perumal", "age": "55"}],
            "buyers": [{"name": "Lakshmi W/o Senthil", "relation": "W/o Senthil"}],
            "payment_mode": "Demand Draft",
            "possession_date": "20-06-2020",
            "encumbrance_declaration": "The Seller hereby declares that the property is free from all encumbrances.",
            "property_description": "All that piece and parcel of land bearing S.F.No. 311/1, Chromepet Village.",
            "boundaries": {"north": "Road", "south": "S.F.No. 312", "east": "Canal", "west": "S.F.No. 310"},
            "ownership_history": [
                {
                    "owner": "Raman S/o Krishnan",
                    "acquired_from": "State (Grant)",
                    "acquisition_mode": "Original Grant",
                    "document_number": "—",
                    "document_date": "Pre-2010",
                },
                {
                    "owner": "Muthu D/o Perumal",
                    "acquired_from": "Raman S/o Krishnan",
                    "acquisition_mode": "Sale",
                    "document_number": "567/2012",
                    "document_date": "15-03-2012",
                },
            ],
            "conditions": [],
            "witnesses": [],
        }
        html = template.render(**ctx)
        assert "1234/2020" in html
        assert "₹45,00,000" in html
        assert "Muthu D/o Perumal" in html
        assert "Lakshmi W/o Senthil" in html
        assert "Road" in html  # boundary north
        assert "Demand Draft" in html
        assert "encumbrance" in html.lower()
        assert "Raman S/o Krishnan" in html  # ownership history

    def test_render_sale_deed_partial_data(self, template_env):
        """Template handles sale deed with only registration (no financials, etc.)."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        ctx["sale_deed"] = {
            "registration": {"document_number": "999/2021"},
            "financials": {},
            "sellers": [],
            "buyers": [],
            "payment_mode": "",
            "possession_date": "",
            "encumbrance_declaration": "",
            "property_description": "",
            "ownership_history": [],
            "conditions": [],
            "witnesses": [],
        }
        html = template.render(**ctx)
        assert "999/2021" in html
        assert len(html) > 500

    def test_render_missing_documents(self, template_env):
        """Template renders missing documents callout when present."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        ctx["missing_documents"] = ["FMB (Field Measurement Book)", "Approved Layout Plan"]
        html = template.render(**ctx)
        assert "Missing Documents" in html
        assert "FMB" in html
        assert "Approved Layout Plan" in html

    def test_render_witnesses(self, template_env):
        """Template renders sale deed witnesses when present."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        ctx["sale_deed"] = {
            "registration": {"document_number": "123/2020"},
            "financials": {},
            "sellers": [{"name": "Seller A"}],
            "buyers": [{"name": "Buyer B"}],
            "payment_mode": "",
            "possession_date": "",
            "encumbrance_declaration": "",
            "property_description": "",
            "ownership_history": [],
            "conditions": [],
            "witnesses": [
                {"name": "Arun Kumar", "address": "12, Anna Nagar"},
                {"name": "Priya Devi", "address": "45, Gandhi Road"},
            ],
        }
        html = template.render(**ctx)
        assert "Arun Kumar" in html
        assert "Priya Devi" in html
        assert "Anna Nagar" in html

    def test_render_no_system_internals(self, template_env):
        """Template should not contain system-internal elements."""
        template = template_env.get_template("report.html")
        ctx = self._make_context()
        html = template.render(**ctx)
        # Cluster IDs removed
        assert "CLU-" not in html
        # Confidence badges removed
        assert "ck-conf" not in html
        # Evidence source snippets removed
        assert "ck-src" not in html
        # AI deduplication label removed
        assert "AI Name Deduplication" not in html
        # Flag IDs removed
        assert "RED FLAG-" not in html
        assert "POSITIVE FINDING" not in html
        # Page 2 of 2 removed (flowing layout)
        assert "Page 2 of 2" not in html
        assert "Page 1 of 2" not in html


# ═══════════════════════════════════════════════════
# 7. Generator helper: _extract_property_summary
# ═══════════════════════════════════════════════════

class TestExtractPropertySummary:
    """Test _extract_property_summary with realistic session data shapes."""

    def test_extracts_from_sale_deed(self):
        from app.reports.generator import _extract_property_summary

        session = {
            "extracted_data": {
                "sale_deed.pdf": {
                    "document_type": "SALE_DEED",
                    "data": {
                        "property": {
                            "survey_number": "311/1",
                            "village": "Chromepet",
                            "taluk": "Tambaram",
                            "district": "Chengalpattu",
                            "extent": "2400 sq.ft",
                        },
                        "buyer": [{"name": "Lakshmi"}],
                    },
                },
            },
        }
        prop = _extract_property_summary(session)
        assert isinstance(prop, dict)
        # Should extract at least some property fields
        assert any(v for v in prop.values() if v)

    def test_extracts_from_ec(self):
        from app.reports.generator import _extract_property_summary

        session = {
            "extracted_data": {
                "ec.pdf": {
                    "document_type": "EC",
                    "data": {
                        "property_description": "Land in S.F.No. 311/1",
                        "village": "Chromepet",
                        "taluk": "Tambaram",
                    },
                },
            },
        }
        prop = _extract_property_summary(session)
        assert isinstance(prop, dict)

    def test_handles_empty_extracted_data(self):
        from app.reports.generator import _extract_property_summary

        session = {"extracted_data": {}}
        prop = _extract_property_summary(session)
        assert isinstance(prop, dict)

    def test_handles_missing_extracted_data(self):
        from app.reports.generator import _extract_property_summary

        session = {}
        prop = _extract_property_summary(session)
        assert isinstance(prop, dict)


# ═══════════════════════════════════════════════════
# 8. Numpy safe comparison patterns
# ═══════════════════════════════════════════════════

class TestNumpySafePatterns:
    """Verify safe patterns for comparing numpy arrays used throughout the codebase."""

    def test_is_not_none_on_numpy(self):
        """``x is not None`` is safe for numpy arrays."""
        arr = np.array([0.1, 0.2, 0.3])
        assert (arr is not None) is True

    def test_is_none_on_none(self):
        """``x is None`` correctly identifies None."""
        assert (None is None) is True
        arr = np.array([0.1, 0.2])
        assert (arr is None) is False

    def test_len_check_on_numpy(self):
        """``len(arr) > 0`` is safe for numpy arrays."""
        arr = np.array([0.1, 0.2, 0.3])
        assert len(arr) > 0
        empty = np.array([])
        assert len(empty) == 0

    def test_indexing_numpy_safely(self):
        """Indexing numpy arrays works the same as lists."""
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        assert len(arr[0]) == 2
        # But bool(arr[0]) would fail:
        with pytest.raises(ValueError):
            bool(arr[0])

    def test_iterating_numpy_for_dot_product(self):
        """zip-based dot product (used in MMR) works with numpy arrays."""
        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])
        dot = sum(x * y for x, y in zip(a, b))
        assert isinstance(float(dot), float)
        assert dot > 0


# ═══════════════════════════════════════════════════
# 9. Finalize hint includes all schema properties
# ═══════════════════════════════════════════════════

class TestFinalizeHintAllProperties:
    """Verify _build_finalize_hint includes ALL properties, not just required."""

    def test_hint_includes_chain_of_title(self):
        """chain_of_title must appear in the finalize hint for Group 1."""
        from app.pipeline.llm_client import _build_finalize_hint
        from app.pipeline.schemas import VERIFY_GROUP1_SCHEMA

        hint = _build_finalize_hint(VERIFY_GROUP1_SCHEMA)
        assert "chain_of_title" in hint
        assert "active_encumbrances" in hint

    def test_hint_includes_checks(self):
        """Standard required fields must also be present."""
        from app.pipeline.llm_client import _build_finalize_hint
        from app.pipeline.schemas import VERIFY_GROUP1_SCHEMA

        hint = _build_finalize_hint(VERIFY_GROUP1_SCHEMA)
        assert "checks" in hint
        assert "group_score_deduction" in hint
        assert "group" in hint

    def test_hint_with_no_schema(self):
        """Fallback hint when schema is None."""
        from app.pipeline.llm_client import _build_finalize_hint

        hint = _build_finalize_hint(None)
        assert "checks" in hint
        assert "group_score_deduction" in hint

    def test_hint_with_empty_schema(self):
        """Empty schema returns fallback hint."""
        from app.pipeline.llm_client import _build_finalize_hint

        hint = _build_finalize_hint({})
        assert isinstance(hint, str)
        assert hint.startswith("{")

    def test_group1_schema_requires_chain_of_title(self):
        """VERIFY_GROUP1_SCHEMA must have chain_of_title in required list."""
        from app.pipeline.schemas import VERIFY_GROUP1_SCHEMA

        required = VERIFY_GROUP1_SCHEMA.get("required", [])
        assert "chain_of_title" in required
        assert "active_encumbrances" in required


# ═══════════════════════════════════════════════════
# 10. Sale deed merger deep-merges boundaries
# ═══════════════════════════════════════════════════

class TestSaleDeedMergerBoundaries:
    """Verify _merge_sale_deed_results deep-merges property.boundaries."""

    def test_boundaries_from_later_chunk_preserved(self):
        """Boundaries extracted in chunk 2 are preserved when chunk 1 has empties."""
        from app.pipeline.extractors.sale_deed import _merge_sale_deed_results

        chunk1 = {
            "document_number": "R/Vadavalli/Book1/5909/2012",
            "registration_date": "2012-12-13",
            "execution_date": "2012-12-13",
            "sro": "Vadavalli",
            "seller": [{"name": "V. Test"}],
            "buyer": [{"name": "T. Test"}],
            "property": {
                "survey_number": "317",
                "village": "Somayampalayam",
                "taluk": "Coimbatore North",
                "district": "Tiruppur",
                "extent": "2.29 acres",
                "boundaries": {"north": "", "south": "", "east": "", "west": ""},
                "property_type": "Agricultural",
            },
            "property_description": "test",
            "financials": {"consideration_amount": 56792000, "guideline_value": 0,
                          "stamp_duty": 0, "registration_fee": 0},
            "payment_mode": "",
            "previous_ownership": {"document_number": "", "document_date": "",
                                   "previous_owner": "", "acquisition_mode": "Unknown"},
            "ownership_history": [],
            "encumbrance_declaration": "",
            "possession_date": "",
            "witnesses": [],
            "special_conditions": [],
            "power_of_attorney": "",
            "remarks": "",
        }

        chunk2 = {
            "document_number": "",
            "registration_date": "",
            "execution_date": "",
            "sro": "",
            "seller": [],
            "buyer": [],
            "property": {
                "survey_number": "317",
                "village": "",
                "taluk": "",
                "district": "",
                "extent": "",
                "boundaries": {
                    "north": "துளசிராம் பூமிக்கும்",
                    "south": "எங்களுக்கு பாத்தியப்பட்ட பூமிக்கும்",
                    "east": "கண்ணப்பன் பூமிக்கும்",
                    "west": "ஸ்ரீபால் மகேஸ்வரி பூமிக்கும்",
                },
                "property_type": "Agricultural",
            },
            "property_description": "",
            "financials": {"consideration_amount": 0, "guideline_value": 0,
                          "stamp_duty": 0, "registration_fee": 0},
            "payment_mode": "",
            "previous_ownership": {"document_number": "", "document_date": "",
                                   "previous_owner": "", "acquisition_mode": "Unknown"},
            "ownership_history": [],
            "encumbrance_declaration": "",
            "possession_date": "",
            "witnesses": [],
            "special_conditions": [],
            "power_of_attorney": "",
            "remarks": "",
        }

        merged = _merge_sale_deed_results([chunk1, chunk2], 20)

        # Boundaries from chunk 2 should be preserved
        bounds = merged["property"]["boundaries"]
        assert bounds["north"] == "துளசிராம் பூமிக்கும்"
        assert bounds["south"] == "எங்களுக்கு பாத்தியப்பட்ட பூமிக்கும்"
        assert bounds["east"] == "கண்ணப்பன் பூமிக்கும்"
        assert bounds["west"] == "ஸ்ரீபால் மகேஸ்வரி பூமிக்கும்"

        # Other property fields from chunk 1 should be preserved
        assert merged["property"]["village"] == "Somayampalayam"
        assert merged["property"]["extent"] == "2.29 acres"

    def test_earlier_boundaries_not_overwritten(self):
        """If chunk 1 has boundaries, chunk 2 empties should NOT overwrite."""
        from app.pipeline.extractors.sale_deed import _merge_sale_deed_results

        chunk1 = {
            "document_number": "5909/2012",
            "registration_date": "2012-12-13",
            "execution_date": "2012-12-13",
            "sro": "Vadavalli",
            "seller": [],
            "buyer": [],
            "property": {
                "survey_number": "317",
                "village": "Somayampalayam",
                "taluk": "",
                "district": "",
                "extent": "2.29 acres",
                "boundaries": {"north": "A land", "south": "B land",
                              "east": "C land", "west": "D land"},
                "property_type": "Agricultural",
            },
            "property_description": "",
            "financials": {"consideration_amount": 0, "guideline_value": 0,
                          "stamp_duty": 0, "registration_fee": 0},
            "payment_mode": "",
            "previous_ownership": {"document_number": "", "document_date": "",
                                   "previous_owner": "", "acquisition_mode": "Unknown"},
            "ownership_history": [],
            "encumbrance_declaration": "",
            "possession_date": "",
            "witnesses": [],
            "special_conditions": [],
            "power_of_attorney": "",
            "remarks": "",
        }

        chunk2 = {
            "document_number": "",
            "registration_date": "",
            "execution_date": "",
            "sro": "",
            "seller": [],
            "buyer": [],
            "property": {
                "survey_number": "",
                "village": "",
                "taluk": "Coimbatore North",
                "district": "Tiruppur",
                "extent": "",
                "boundaries": {"north": "", "south": "", "east": "", "west": ""},
                "property_type": "Agricultural",
            },
            "property_description": "",
            "financials": {"consideration_amount": 0, "guideline_value": 0,
                          "stamp_duty": 0, "registration_fee": 0},
            "payment_mode": "",
            "previous_ownership": {"document_number": "", "document_date": "",
                                   "previous_owner": "", "acquisition_mode": "Unknown"},
            "ownership_history": [],
            "encumbrance_declaration": "",
            "possession_date": "",
            "witnesses": [],
            "special_conditions": [],
            "power_of_attorney": "",
            "remarks": "",
        }

        merged = _merge_sale_deed_results([chunk1, chunk2], 20)

        # Original boundaries should be preserved
        bounds = merged["property"]["boundaries"]
        assert bounds["north"] == "A land"
        assert bounds["south"] == "B land"

        # New non-empty property fields should be merged in
        assert merged["property"]["taluk"] == "Coimbatore North"
        assert merged["property"]["district"] == "Tiruppur"

    def test_deep_merge_property_function(self):
        """Direct test of _deep_merge_property."""
        from app.pipeline.extractors.sale_deed import _deep_merge_property

        existing = {
            "survey_number": "317",
            "village": "Somayampalayam",
            "taluk": "",
            "district": "",
            "extent": "2.29 acres",
            "boundaries": {"north": "", "south": "", "east": "", "west": ""},
            "property_type": "Agricultural",
        }
        incoming = {
            "survey_number": "",
            "village": "",
            "taluk": "Coimbatore North",
            "district": "Tiruppur",
            "extent": "",
            "boundaries": {"north": "X land", "south": "", "east": "Y land", "west": ""},
            "property_type": "Agricultural",
        }

        result = _deep_merge_property(existing, incoming)
        assert result["survey_number"] == "317"
        assert result["village"] == "Somayampalayam"
        assert result["taluk"] == "Coimbatore North"
        assert result["district"] == "Tiruppur"
        assert result["extent"] == "2.29 acres"
        assert result["boundaries"]["north"] == "X land"
        assert result["boundaries"]["south"] == ""
        assert result["boundaries"]["east"] == "Y land"
        assert result["boundaries"]["west"] == ""

    def test_deep_merge_property_empty_existing(self):
        """_deep_merge_property returns incoming when existing is empty."""
        from app.pipeline.extractors.sale_deed import _deep_merge_property

        incoming = {"survey_number": "317", "boundaries": {"north": "X"}}
        assert _deep_merge_property({}, incoming) == incoming
        assert _deep_merge_property(None, incoming) == incoming


# ═══════════════════════════════════════════════════
# 11. Consideration amount preparse
# ═══════════════════════════════════════════════════

class TestConsiderationAmountPreparse:
    """Verify extract_consideration_amount parses Indian number formats."""

    def test_indian_format_with_dashes(self):
        """Standard Tamil stamp paper format: ரூபாய் 5,67,92,000/-"""
        from app.pipeline.sale_deed_preparse import extract_consideration_amount

        text = "ரூபாய் 5,67,92,000/-க்கு கிரயம்"
        result = extract_consideration_amount(text)
        assert result == 56792000

    def test_rs_prefix(self):
        """Rs. prefix format."""
        from app.pipeline.sale_deed_preparse import extract_consideration_amount

        text = "Rs. 25,00,000/- consideration"
        result = extract_consideration_amount(text)
        assert result == 2500000

    def test_rupee_symbol(self):
        """₹ symbol format."""
        from app.pipeline.sale_deed_preparse import extract_consideration_amount

        text = "₹ 10,50,000"
        result = extract_consideration_amount(text)
        assert result == 1050000

    def test_no_amount(self):
        """Returns None when no amount found."""
        from app.pipeline.sale_deed_preparse import extract_consideration_amount

        result = extract_consideration_amount("No monetary amount here")
        assert result is None

    def test_picks_largest_amount(self):
        """When multiple amounts present, picks the largest (likely consideration)."""
        from app.pipeline.sale_deed_preparse import extract_consideration_amount

        text = "ரூ. 25,000 stamp duty... ரூபாய் 5,67,92,000/- total"
        result = extract_consideration_amount(text)
        assert result == 56792000
