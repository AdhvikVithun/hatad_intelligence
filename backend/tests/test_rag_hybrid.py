"""Tests for RAG hybrid keyword+semantic retrieval helpers."""

import pytest
from app.pipeline.rag_store import RAGStore


# ── _extract_keywords ──

class TestExtractKeywords:
    def test_basic_extraction(self):
        kws = RAGStore._extract_keywords("survey number 311/1 owner Raman")
        assert "311/1" in kws
        assert "survey" in kws
        assert "raman" in kws

    def test_stopwords_removed(self):
        kws = RAGStore._extract_keywords("check whether the document has this value")
        # all are stopwords or <3 chars
        assert "the" not in kws
        assert "check" not in kws
        assert "whether" not in kws
        assert "value" not in kws

    def test_numeric_tokens_always_kept(self):
        kws = RAGStore._extract_keywords("document 5678/2020 search the value")
        assert "5678/2020" in kws  # numeric — always kept regardless of stopwords

    def test_empty_query(self):
        assert RAGStore._extract_keywords("") == []

    def test_short_tokens_skipped(self):
        kws = RAGStore._extract_keywords("in on at of")
        assert kws == []

    def test_deduplication(self):
        kws = RAGStore._extract_keywords("sale SALE Sale deed")
        assert kws.count("sale") == 1

    def test_identifier_patterns(self):
        kws = RAGStore._extract_keywords("EC-2024-001 SF.No.311/1A Tambaram")
        assert any("2024" in k for k in kws)
        assert "tambaram" in kws


# ── _keyword_score ──

class TestKeywordScore:
    def test_perfect_match(self):
        kws = ["survey", "311/1", "raman"]
        doc = "Survey number 311/1 executed by Raman"
        score = RAGStore._keyword_score(kws, doc)
        assert score == 1.0

    def test_no_match(self):
        kws = ["survey", "311/1"]
        doc = "This document contains only unrelated text about cooking."
        score = RAGStore._keyword_score(kws, doc)
        assert score == 0.0

    def test_partial_match(self):
        kws = ["survey", "311/1", "raman"]
        # Only "survey" matches (weight=1), "311/1" (weight=2) and "raman" (weight=1) don't
        doc = "The survey indicates the land is fertile."
        score = RAGStore._keyword_score(kws, doc)
        # hit_weight=1, total_weight=1+2+1=4
        assert abs(score - 0.25) < 0.01

    def test_numeric_double_weight(self):
        kws = ["survey", "311/1"]
        # "311/1" is numeric → weight 2; "survey" → weight 1; total=3
        # Only "311/1" matches → hit_weight=2 → score=2/3≈0.667
        doc = "Land extent 311/1 is in the record"
        score = RAGStore._keyword_score(kws, doc)
        assert abs(score - 2.0 / 3.0) < 0.01

    def test_empty_keywords(self):
        assert RAGStore._keyword_score([], "any text") == 0.0

    def test_case_insensitive(self):
        kws = ["raman", "chromepet"]
        doc = "RAMAN purchased land in CHROMEPET village"
        assert RAGStore._keyword_score(kws, doc) == 1.0


# ── Integration: query_sync with keyword re-ranking ──

class TestQuerySyncHybrid:
    """Test that query_text parameter correctly adjusts scores."""

    def test_query_sync_accepts_query_text_param(self):
        """query_sync should accept query_text as a keyword argument."""
        store = RAGStore("test_hybrid_sync")
        # With 0 indexed docs, should return [] without error
        result = store.query_sync(
            question_embedding=[0.0] * 768,
            query_text="survey 311/1",
        )
        assert result == []
        store.cleanup()

    def test_query_mmr_sync_accepts_query_text_param(self):
        """query_mmr_sync should accept query_text as a keyword argument."""
        store = RAGStore("test_hybrid_mmr")
        result = store.query_mmr_sync(
            question_embedding=[0.0] * 768,
            query_text="survey 311/1",
        )
        assert result == []
        store.cleanup()


class TestRetrievedChunkMethods:
    """Test RetrievedChunk.to_citation() and to_dict() are accessible."""

    def test_to_citation(self):
        from app.pipeline.rag_store import RetrievedChunk
        chunk = RetrievedChunk(
            text="some text", filename="doc.pdf",
            page_number=3, chunk_index=0, score=0.15,
        )
        assert chunk.to_citation() == "[doc.pdf p.3]"

    def test_to_dict(self):
        from app.pipeline.rag_store import RetrievedChunk
        chunk = RetrievedChunk(
            text="hello", filename="ec.pdf",
            page_number=1, chunk_index=2, score=0.1234,
            transaction_id="txn_5",
        )
        d = chunk.to_dict()
        assert d["filename"] == "ec.pdf"
        assert d["transaction_id"] == "txn_5"
        assert d["score"] == 0.1234

    def test_format_evidence_uses_to_citation(self):
        """RAGStore.format_evidence should call to_citation without error."""
        from app.pipeline.rag_store import RetrievedChunk
        chunks = [
            RetrievedChunk(
                text="Survey 311/1 land details",
                filename="ec.pdf", page_number=2,
                chunk_index=0, score=0.2,
            )
        ]
        evidence = RAGStore.format_evidence(chunks)
        assert "[ec.pdf p.2]" in evidence
        assert "Survey 311/1" in evidence
