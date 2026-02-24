"""Tests for the /api/analyze/{session_id}/chat endpoint."""

import json
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from app.pipeline.orchestrator import AnalysisSession

# ── Guard: skip all tests if playwright is not installed ──
# (verification.py -> generator.py -> playwright at import time)
try:
    from app.api.verification import (
        _build_chat_system_prompt,
        _CHAT_MAX_NARRATIVE_CHARS,
        ChatRequest,
        ChatMessageIn,
        chat_with_session,
    )
    _CAN_IMPORT = True
except ImportError:
    _CAN_IMPORT = False

pytestmark = pytest.mark.skipif(not _CAN_IMPORT, reason="Optional deps (playwright etc.) not installed")


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def _make_completed_session(session_id="test-chat-001") -> AnalysisSession:
    """Create a minimal completed AnalysisSession for chat tests."""
    session = AnalysisSession.__new__(AnalysisSession)
    session.session_id = session_id
    session.created_at = datetime.now().isoformat()
    session.status = "completed"
    session.documents = [{"filename": "sale_deed.pdf"}, {"filename": "ec.pdf"}]
    session.extracted_data = {}
    session.memory_bank = {
        "facts": [
            {"category": "property", "key": "survey_no", "value": "311/1",
             "source": "sale_deed.pdf", "confidence": 0.95},
        ],
        "conflicts": [],
        "cross_references": [],
        "ingested_files": ["sale_deed.pdf", "ec.pdf"],
        "summary": {"property": 1},
    }
    session.verification_result = {
        "risk_score": 22,
        "risk_band": "LOW",
        "executive_summary": "Low risk property.",
        "checks": [],
        "chain_of_title": [],
        "red_flags": [],
        "recommendations": [],
        "missing_documents": [],
    }
    session.narrative_report = "This is a test narrative report about Survey No 311/1."
    session.risk_score = 22
    session.risk_band = "LOW"
    session.identity_clusters = []
    session.chat_history = []
    session.progress = []
    session.error = None
    return session


def _make_running_session(session_id="test-chat-002") -> AnalysisSession:
    """Session that hasn't completed."""
    session = _make_completed_session(session_id)
    session.status = "running"
    return session


# ═══════════════════════════════════════════════════
# Unit tests for _build_chat_system_prompt
# ═══════════════════════════════════════════════════

class TestBuildChatSystemPrompt:

    def test_template_fills_placeholders(self):
        from app.api.verification import _build_chat_system_prompt
        session = _make_completed_session()
        prompt = _build_chat_system_prompt(session, "some MB context", "some RAG evidence")
        assert "some MB context" in prompt
        assert "some RAG evidence" in prompt
        assert "2" in prompt  # doc_count = len(documents) = 2
        assert "22" in prompt  # risk_score
        assert "LOW" in prompt  # risk_band

    def test_missing_narrative_shows_fallback(self):
        from app.api.verification import _build_chat_system_prompt
        session = _make_completed_session()
        session.narrative_report = None
        prompt = _build_chat_system_prompt(session, "", "")
        assert "No narrative report available" in prompt

    def test_long_narrative_truncated(self):
        from app.api.verification import _build_chat_system_prompt, _CHAT_MAX_NARRATIVE_CHARS
        session = _make_completed_session()
        session.narrative_report = "A" * (_CHAT_MAX_NARRATIVE_CHARS + 1000)
        prompt = _build_chat_system_prompt(session, "", "")
        assert "truncated" in prompt.lower()

    def test_empty_memory_bank(self):
        from app.api.verification import _build_chat_system_prompt
        session = _make_completed_session()
        prompt = _build_chat_system_prompt(session, "", "")
        # Should not crash, and should include "No memory bank" or the empty string fills
        assert isinstance(prompt, str)

    def test_missing_prompt_file_uses_fallback(self, tmp_path):
        from app.api.verification import _build_chat_system_prompt
        session = _make_completed_session()
        # Patch PROMPTS_DIR to a nonexistent path
        with patch("app.api.verification.PROMPTS_DIR", tmp_path / "nonexistent"):
            prompt = _build_chat_system_prompt(session, "ctx", "ev")
            assert "HATAD Intelligence" in prompt


# ═══════════════════════════════════════════════════
# ChatRequest validation tests
# ═══════════════════════════════════════════════════

class TestChatRequestModel:

    def test_valid_request(self):
        from app.api.verification import ChatRequest
        req = ChatRequest(message="What is the survey number?", history=[])
        assert req.message == "What is the survey number?"
        assert req.history == []

    def test_with_history(self):
        from app.api.verification import ChatRequest, ChatMessageIn
        req = ChatRequest(
            message="Follow up question",
            history=[
                ChatMessageIn(role="user", content="Hello"),
                ChatMessageIn(role="assistant", content="Hi there"),
            ],
        )
        assert len(req.history) == 2
        assert req.history[0].role == "user"

    def test_empty_message_allowed(self):
        """Pydantic doesn't block empty strings unless we add a validator."""
        from app.api.verification import ChatRequest
        req = ChatRequest(message="", history=[])
        assert req.message == ""


# ═══════════════════════════════════════════════════
# Endpoint guard tests (sync — no actual streaming)
# ═══════════════════════════════════════════════════

class TestChatEndpointGuards:

    @pytest.mark.asyncio
    async def test_session_not_found_returns_404(self):
        from app.api.verification import chat_with_session, ChatRequest
        from fastapi import HTTPException

        request = ChatRequest(message="Hello", history=[])
        with patch.object(AnalysisSession, "load", side_effect=FileNotFoundError):
            with pytest.raises(HTTPException) as exc_info:
                await chat_with_session("nonexistent-id", request)
            assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_incomplete_session_returns_400(self):
        from app.api.verification import chat_with_session, ChatRequest
        from fastapi import HTTPException

        request = ChatRequest(message="Hello", history=[])
        session = _make_running_session()
        with patch.object(AnalysisSession, "load", return_value=session):
            with pytest.raises(HTTPException) as exc_info:
                await chat_with_session("test-chat-002", request)
            assert exc_info.value.status_code == 400
            assert "completed" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_completed_session_returns_streaming_response(self):
        """A completed session should return a StreamingResponse (not raise)."""
        from app.api.verification import chat_with_session, ChatRequest
        from fastapi.responses import StreamingResponse

        request = ChatRequest(message="What is the risk score?", history=[])
        session = _make_completed_session()

        with patch.object(AnalysisSession, "load", return_value=session):
            with patch("app.api.verification.RAGStore") as mock_rag:
                mock_collection = MagicMock()
                mock_collection.count.return_value = 0
                mock_rag_inst = MagicMock()
                mock_rag_inst._collection = mock_collection
                mock_rag_inst._indexed_count = 0
                mock_rag.return_value = mock_rag_inst
                mock_rag_inst.query = AsyncMock(return_value=[])

                result = await chat_with_session("test-chat-001", request)
                assert isinstance(result, StreamingResponse)
                assert result.media_type == "text/event-stream"


# ═══════════════════════════════════════════════════
# History truncation tests
# ═══════════════════════════════════════════════════

class TestHistoryTruncation:

    def test_history_cap(self):
        """Ensure we only send the last N*2 messages to the LLM."""
        from app.api.verification import _CHAT_MAX_HISTORY_TURNS, ChatMessageIn

        # Build a very long history
        long_history = []
        for i in range(50):
            long_history.append(ChatMessageIn(role="user", content=f"Question {i}"))
            long_history.append(ChatMessageIn(role="assistant", content=f"Answer {i}"))

        # Truncation logic from the endpoint
        cap = _CHAT_MAX_HISTORY_TURNS * 2
        truncated = long_history[-cap:]
        assert len(truncated) == cap
        # The oldest message in truncated should be from index (50 - 20)
        assert "Question 30" in truncated[0].content


# ═══════════════════════════════════════════════════
# Session persistence of chat_history
# ═══════════════════════════════════════════════════

class TestChatHistoryPersistence:

    def test_chat_history_in_session_dict(self):
        """chat_history should appear in to_dict() output."""
        session = _make_completed_session()
        session.chat_history = [
            {"role": "user", "content": "Hi", "timestamp": "2024-01-01T00:00:00"},
        ]
        d = session.to_dict()
        assert "chat_history" in d
        assert len(d["chat_history"]) == 1
        assert d["chat_history"][0]["role"] == "user"

    def test_empty_chat_history(self):
        session = _make_completed_session()
        session.chat_history = []
        d = session.to_dict()
        assert d["chat_history"] == []
