"""Tests for results endpoint retry / resilience logic.

Tests cover:
  - _load_session_with_retry: transient file errors, persistent failures, 404
  - _safe_json_response: datetime/Path safety, round-trip fidelity
  - get_analysis_results: completed sessions, status guard, failed sessions
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.pipeline.orchestrator import AnalysisSession

# Guard: skip if playwright transitive import fails
try:
    from app.api.verification import (
        _load_session_with_retry,
        _safe_json_response,
        _LOAD_MAX_RETRIES,
        _LOAD_BACKOFF_BASE,
    )
    _CAN_IMPORT = True
except ImportError:
    _CAN_IMPORT = False

pytestmark = pytest.mark.skipif(not _CAN_IMPORT, reason="Optional deps not installed")


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def _make_session(
    session_id: str = "test-res-001",
    status: str = "completed",
) -> AnalysisSession:
    """Create a minimal AnalysisSession for testing."""
    s = AnalysisSession.__new__(AnalysisSession)
    s.session_id = session_id
    s.created_at = datetime.now().isoformat()
    s.status = status
    s.progress = [{"stage": "complete", "message": "done"}]
    s.documents = [{"filename": "sale.pdf", "document_type": "SALE_DEED"}]
    s.extracted_data = {"sale.pdf": {"document_type": "SALE_DEED", "data": {}}}
    s.memory_bank = {"facts": [], "conflicts": [], "cross_references": [],
                     "ingested_files": ["sale.pdf"], "summary": {}}
    s.verification_result = {
        "risk_score": 72, "risk_band": "MEDIUM",
        "checks": [], "executive_summary": "Test",
        "red_flags": [], "recommendations": [],
        "missing_documents": [], "chain_of_title": [],
    }
    s.narrative_report = "Test narrative"
    s.risk_score = 72
    s.risk_band = "MEDIUM"
    s.identity_clusters = []
    s.error = None
    s.chat_history = []
    return s


# ═══════════════════════════════════════════════════
# _load_session_with_retry
# ═══════════════════════════════════════════════════

class TestLoadSessionWithRetry:
    """Test the retry helper for transient file errors."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        """Normal case — loads on first try."""
        session = _make_session()
        with patch.object(AnalysisSession, "load", return_value=session):
            result = await _load_session_with_retry("test-001")
        assert result.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_file_not_found_raises_404(self):
        """FileNotFoundError immediately raises HTTPException 404."""
        from fastapi import HTTPException
        with patch.object(AnalysisSession, "load", side_effect=FileNotFoundError):
            with pytest.raises(HTTPException) as exc_info:
                await _load_session_with_retry("missing-id")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_transient_permission_error_recovers(self):
        """PermissionError on first attempt, success on second."""
        session = _make_session()
        call_count = 0

        def flaky_load(sid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise PermissionError("File locked by antivirus")
            return session

        with patch.object(AnalysisSession, "load", side_effect=flaky_load):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                result = await _load_session_with_retry("test-perm")
        assert result.session_id == session.session_id
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_transient_os_error_recovers(self):
        """OSError on first attempt, success on second."""
        session = _make_session()
        call_count = 0

        def flaky_load(sid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Disk I/O error")
            return session

        with patch.object(AnalysisSession, "load", side_effect=flaky_load):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                result = await _load_session_with_retry("test-os")
        assert result.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_transient_json_decode_error_recovers(self):
        """JSONDecodeError (corrupted file) on first attempt recovers."""
        session = _make_session()
        call_count = 0

        def flaky_load(sid):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return session

        with patch.object(AnalysisSession, "load", side_effect=flaky_load):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                result = await _load_session_with_retry("test-json")
        assert result.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_persistent_error_raises_503(self):
        """All retries exhausted → HTTPException 503."""
        from fastapi import HTTPException
        with patch.object(AnalysisSession, "load",
                          side_effect=PermissionError("locked")):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(HTTPException) as exc_info:
                    await _load_session_with_retry("test-locked")
        assert exc_info.value.status_code == 503
        assert "retry" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_retries_correct_count(self):
        """Verify exactly _LOAD_MAX_RETRIES attempts are made."""
        call_count = 0

        def always_fail(sid):
            nonlocal call_count
            call_count += 1
            raise PermissionError("locked")

        from fastapi import HTTPException
        with patch.object(AnalysisSession, "load", side_effect=always_fail):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(HTTPException):
                    await _load_session_with_retry("test-count")
        assert call_count == _LOAD_MAX_RETRIES

    @pytest.mark.asyncio
    async def test_unicode_error_is_retried(self):
        """UnicodeDecodeError is a transient error and should be retried."""
        session = _make_session()
        call_count = 0

        def flaky_load(sid):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise UnicodeDecodeError("utf-8", b"", 0, 1, "bad byte")
            return session

        with patch.object(AnalysisSession, "load", side_effect=flaky_load):
            with patch("app.api.verification.asyncio.sleep", new_callable=AsyncMock):
                result = await _load_session_with_retry("test-unicode")
        assert result.session_id == session.session_id
        assert call_count == 3


# ═══════════════════════════════════════════════════
# _safe_json_response
# ═══════════════════════════════════════════════════

class TestSafeJsonResponse:
    """Test the safe JSON response helper."""

    def test_plain_dict_passthrough(self):
        """Simple dict passes through unchanged."""
        data = {"key": "value", "number": 42}
        resp = _safe_json_response(data)
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body == data

    def test_datetime_converted_to_string(self):
        """datetime objects are stringified (matching session.save behavior)."""
        dt = datetime(2026, 2, 24, 10, 30, 0)
        data = {"created": dt, "nested": {"ts": dt}}
        resp = _safe_json_response(data)
        body = json.loads(resp.body)
        assert isinstance(body["created"], str)
        assert "2026" in body["created"]
        assert isinstance(body["nested"]["ts"], str)

    def test_path_converted_to_string(self):
        """Path objects are stringified."""
        data = {"file": Path("/tmp/test.pdf")}
        resp = _safe_json_response(data)
        body = json.loads(resp.body)
        assert isinstance(body["file"], str)
        assert "test.pdf" in body["file"]

    def test_custom_status_code(self):
        """Status code parameter is respected."""
        resp = _safe_json_response({"ok": True}, status_code=201)
        assert resp.status_code == 201

    def test_full_session_round_trip(self):
        """A complete session dict serializes cleanly."""
        session = _make_session()
        data = session.to_dict()
        resp = _safe_json_response(data)
        body = json.loads(resp.body)
        assert body["session_id"] == "test-res-001"
        assert body["risk_score"] == 72
        assert body["status"] == "completed"

    def test_none_values_preserved(self):
        """None values remain null in JSON."""
        data = {"field": None, "list": [None, 1]}
        resp = _safe_json_response(data)
        body = json.loads(resp.body)
        assert body["field"] is None
        assert body["list"][0] is None

    def test_unicode_preserved(self):
        """Tamil/non-ASCII characters survive the round-trip."""
        data = {"text": "தமிழ் நிலம்", "village": "நாகப்பட்டினம்"}
        resp = _safe_json_response(data)
        body = json.loads(resp.body)
        assert body["text"] == "தமிழ் நிலம்"
        assert body["village"] == "நாகப்பட்டினம்"


# ═══════════════════════════════════════════════════
# Endpoint integration tests (light, no HTTP server)
# ═══════════════════════════════════════════════════

class TestGetAnalysisResults:
    """Test the get_analysis_results endpoint function directly."""

    @pytest.mark.asyncio
    async def test_completed_session_returns_200(self):
        """Completed session returns full results dict."""
        from app.api.verification import get_analysis_results
        session = _make_session(status="completed")
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_analysis_results("test-001")
        body = json.loads(resp.body)
        assert body["status"] == "completed"
        assert "incomplete" not in body

    @pytest.mark.asyncio
    async def test_failed_session_has_incomplete_flag(self):
        """Failed session includes incomplete=True."""
        from app.api.verification import get_analysis_results
        session = _make_session(status="failed")
        session.error = "Some error"
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_analysis_results("test-fail")
        body = json.loads(resp.body)
        assert body["incomplete"] is True

    @pytest.mark.asyncio
    async def test_processing_session_raises_400(self):
        """Session still processing raises 400."""
        from fastapi import HTTPException
        from app.api.verification import get_analysis_results
        session = _make_session(status="processing")
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            with pytest.raises(HTTPException) as exc_info:
                await get_analysis_results("test-proc")
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_results_contain_all_required_fields(self):
        """Verify the response contains all SessionData-expected fields."""
        from app.api.verification import get_analysis_results
        session = _make_session()
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_analysis_results("test-fields")
        body = json.loads(resp.body)
        required = [
            "session_id", "status", "documents", "extracted_data",
            "memory_bank", "verification_result", "narrative_report",
            "risk_score", "risk_band",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_zero_risk_score_serializes_correctly(self):
        """Risk score of 0 (CRITICAL) serializes as integer 0, not null."""
        from app.api.verification import get_analysis_results
        session = _make_session()
        session.risk_score = 0
        session.risk_band = "CRITICAL"
        session.verification_result["risk_score"] = 0
        session.verification_result["risk_band"] = "CRITICAL"
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_analysis_results("test-zero")
        body = json.loads(resp.body)
        assert body["risk_score"] == 0
        assert body["verification_result"]["risk_score"] == 0


class TestGetAnalysisStatus:
    """Test the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_safe_json(self):
        """Status endpoint returns safe JSON response."""
        from app.api.verification import get_analysis_status
        session = _make_session()
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_analysis_status("test-stat")
        body = json.loads(resp.body)
        assert body["status"] == "completed"
        assert body["session_id"] == "test-res-001"


class TestGetMemoryBank:
    """Test the memory-bank endpoint."""

    @pytest.mark.asyncio
    async def test_returns_memory_bank(self):
        """Memory bank endpoint returns the bank dict."""
        from app.api.verification import get_memory_bank
        session = _make_session()
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            resp = await get_memory_bank("test-mb")
        body = json.loads(resp.body)
        assert "facts" in body

    @pytest.mark.asyncio
    async def test_no_memory_bank_raises_400(self):
        """Missing memory bank raises 400."""
        from fastapi import HTTPException
        from app.api.verification import get_memory_bank
        session = _make_session()
        session.memory_bank = None
        with patch("app.api.verification._load_session_with_retry",
                    new_callable=AsyncMock, return_value=session):
            with pytest.raises(HTTPException) as exc_info:
                await get_memory_bank("test-no-mb")
        assert exc_info.value.status_code == 400
