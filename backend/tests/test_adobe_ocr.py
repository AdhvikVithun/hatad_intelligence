"""Tests for Adobe PDF Services OCR integration.

These tests mock the Adobe SDK entirely — no network calls.
"""

import asyncio
import io
import json
import os
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from app.pipeline.adobe_ocr import (
    _validate_pdf,
    _assess_page_quality,
    _parse_adobe_zip,
    _cb_record_failure,
    _cb_record_success,
    _cb_is_open,
    reset_circuit_breaker,
    merge_adobe_with_existing,
    adobe_extract_text,
    _adobe_extract_sync,
    _CB_THRESHOLD,
)


# ═══════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════

def _make_adobe_zip(elements: list[dict], tables: dict[str, bytes] | None = None) -> bytes:
    """Build a fake Adobe Extract API ZIP with structuredData.json."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("structuredData.json", json.dumps({"elements": elements}))
        if tables:
            for name, content in tables.items():
                zf.writestr(name, content)
    return buf.getvalue()


def _make_elements(pages_text: dict[int, list[str]]) -> list[dict]:
    """Build Adobe-style elements from {page_idx: [texts]}."""
    elements = []
    for page_idx, texts in sorted(pages_text.items()):
        for text in texts:
            elements.append({
                "Path": "//Document/P",
                "Text": text,
                "Page": page_idx,
                "Bounds": [0.0, 0.0, 100.0, 20.0],
            })
    return elements


def _make_result(pages_text: dict[int, str], quality: str = "HIGH") -> dict:
    """Build a standard extraction result dict."""
    pages = []
    full_parts = []
    for i, (_, text) in enumerate(sorted(pages_text.items())):
        pages.append({
            "page_number": i + 1,
            "text": text,
            "tables": [],
            "extraction_method": "pdfplumber",
            "quality": _assess_page_quality(text),
        })
        full_parts.append(f"--- PAGE {i + 1} ---\n{text}")
    return {
        "total_pages": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_parts),
        "extraction_quality": quality,
        "ocr_pages": 0,
        "sarvam_pages": 0,
        "adobe_pages": 0,
        "metadata": {"filename": "test.pdf", "file_size": 1000},
    }


# ═══════════════════════════════════════════════════
# FILE VALIDATION
# ═══════════════════════════════════════════════════

class TestValidatePdf:
    def test_nonexistent_file(self, tmp_path):
        err = _validate_pdf(tmp_path / "ghost.pdf")
        assert err is not None
        assert "does not exist" in err

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        assert "empty" in _validate_pdf(f)

    def test_not_pdf(self, tmp_path):
        f = tmp_path / "notpdf.pdf"
        f.write_bytes(b"This is not a PDF")
        assert "not a valid PDF" in _validate_pdf(f)

    def test_valid_pdf(self, tmp_path):
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        assert _validate_pdf(f) is None

    def test_file_too_large(self, tmp_path):
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"%PDF-1.4" + b"\x00" * 10)
        with patch("app.pipeline.adobe_ocr._MAX_FILE_BYTES", 5):
            err = _validate_pdf(f)
            assert err is not None
            assert "too large" in err


# ═══════════════════════════════════════════════════
# QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════

class TestAssessPageQuality:
    def test_high_quality(self):
        text = "This is a page with enough content to be considered high quality text"
        q = _assess_page_quality(text)
        assert q["quality"] == "HIGH"
        assert q["char_count"] > 50

    def test_low_quality_few_chars(self):
        q = _assess_page_quality("short")
        assert q["quality"] == "LOW"
        assert "too few chars" in q["reason"]

    def test_empty_text(self):
        q = _assess_page_quality("")
        assert q["quality"] == "LOW"
        assert q["char_count"] == 0


# ═══════════════════════════════════════════════════
# ZIP PARSING
# ═══════════════════════════════════════════════════

class TestParseAdobeZip:
    def test_basic_two_pages(self, tmp_path):
        elements = _make_elements({
            0: ["Page 1 line 1", "Page 1 line 2 with lot more text to reach HIGH quality threshold"],
            1: ["Page 2 content is here with enough words to count as a high quality extraction"],
        })
        zip_bytes = _make_adobe_zip(elements)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")

        result = _parse_adobe_zip(pdf_path, zip_bytes)
        assert result is not None
        assert result["total_pages"] == 2
        assert "Page 1 line 1" in result["pages"][0]["text"]
        assert "Page 2 content" in result["pages"][1]["text"]
        assert result["pages"][0]["extraction_method"] == "adobe"
        assert result["metadata"]["extraction_source"] == "adobe_pdf_services"

    def test_empty_elements(self, tmp_path):
        zip_bytes = _make_adobe_zip([])
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        result = _parse_adobe_zip(pdf_path, zip_bytes)
        assert result is None

    def test_no_structured_data(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.json", "{}")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        result = _parse_adobe_zip(pdf_path, buf.getvalue())
        assert result is None

    def test_corrupt_zip(self, tmp_path):
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        result = _parse_adobe_zip(pdf_path, b"not a zip")
        assert result is None

    def test_invalid_json(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("structuredData.json", "not json{{{")
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        result = _parse_adobe_zip(pdf_path, buf.getvalue())
        assert result is None

    def test_whitespace_only_text(self, tmp_path):
        elements = [
            {"Path": "//Document/P", "Text": "   \n  \t  ", "Page": 0},
        ]
        zip_bytes = _make_adobe_zip(elements)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF")
        result = _parse_adobe_zip(pdf_path, zip_bytes)
        # Whitespace-only == no content
        assert result is None

    def test_overall_quality_mixed(self, tmp_path):
        elements = _make_elements({
            0: ["A very long page with lots of text that should pass as HIGH quality content for testing purposes"],
            1: ["x"],  # Very short = LOW
        })
        zip_bytes = _make_adobe_zip(elements)
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 content")
        result = _parse_adobe_zip(pdf_path, zip_bytes)
        assert result is not None
        assert result["extraction_quality"] == "MIXED"


# ═══════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════

class TestCircuitBreaker:
    def setup_method(self):
        reset_circuit_breaker()

    def teardown_method(self):
        reset_circuit_breaker()

    def test_initially_closed(self):
        assert not _cb_is_open()

    def test_opens_after_threshold_failures(self):
        for _ in range(_CB_THRESHOLD):
            _cb_record_failure()
        assert _cb_is_open()

    def test_success_resets(self):
        for _ in range(_CB_THRESHOLD - 1):
            _cb_record_failure()
        _cb_record_success()
        _cb_record_failure()  # Only 1 after reset
        assert not _cb_is_open()

    def test_reset_clears(self):
        for _ in range(_CB_THRESHOLD):
            _cb_record_failure()
        assert _cb_is_open()
        reset_circuit_breaker()
        assert not _cb_is_open()

    def test_closes_after_cooldown(self):
        for _ in range(_CB_THRESHOLD):
            _cb_record_failure()
        assert _cb_is_open()
        # Simulate cooldown by setting disabled_until to the past
        import app.pipeline.adobe_ocr as mod
        mod._cb_disabled_until = 0.0
        assert not _cb_is_open()


# ═══════════════════════════════════════════════════
# MERGE LOGIC
# ═══════════════════════════════════════════════════

class TestMergeAdobeWithExisting:
    def test_adobe_replaces_low_quality(self):
        existing = _make_result({0: "x"}, quality="LOW")
        adobe = _make_result({0: "Adobe extracted this full page of content that is quite detailed and long"})
        adobe["pages"][0]["extraction_method"] = "adobe"

        merged = merge_adobe_with_existing(adobe, existing)
        assert merged["pages"][0]["extraction_method"] == "adobe"
        assert merged["adobe_pages"] >= 1

    def test_keeps_high_quality_existing(self):
        existing = _make_result({
            0: "This is a perfectly good pdfplumber extraction with lots of detail and content"
        })
        adobe = _make_result({
            0: "Adobe also extracted this but existing is already good enough to use"
        })
        adobe["pages"][0]["extraction_method"] = "adobe"

        merged = merge_adobe_with_existing(adobe, existing)
        # Both are HIGH — existing should be preferred (no 1.3x advantage)
        assert merged["pages"][0]["extraction_method"] == "pdfplumber"

    def test_adobe_wins_with_more_text(self):
        existing = _make_result({0: "Short existing text only thirty chars or so"})
        existing["pages"][0]["quality"]["quality"] = "LOW"
        existing["pages"][0]["quality"]["char_count"] = 30
        long_text = "Adobe got much more content: " + "word " * 50
        adobe_pages = [{
            "page_number": 1,
            "text": long_text,
            "tables": [],
            "extraction_method": "adobe",
            "quality": _assess_page_quality(long_text),
        }]
        adobe = {
            "total_pages": 1,
            "pages": adobe_pages,
            "full_text": long_text,
            "extraction_quality": "HIGH",
            "ocr_pages": 0,
            "sarvam_pages": 0,
            "adobe_pages": 1,
            "metadata": {},
        }

        merged = merge_adobe_with_existing(adobe, existing)
        assert merged["pages"][0]["extraction_method"] == "adobe"

    def test_handles_different_page_counts(self):
        existing = _make_result({0: "Page 1 only with enough content to be considered high quality"})
        adobe = _make_result({
            0: "Adobe page 1 with enough content for high quality extraction test",
            1: "Adobe page 2 extra content that existing doesn't have at all"
        })
        adobe["pages"][0]["extraction_method"] = "adobe"
        adobe["pages"][1]["extraction_method"] = "adobe"

        merged = merge_adobe_with_existing(adobe, existing)
        assert merged["total_pages"] == 2
        # Page 2 only in Adobe — should be Adobe
        assert merged["pages"][1]["extraction_method"] == "adobe"


# ═══════════════════════════════════════════════════
# ASYNC WRAPPER
# ═══════════════════════════════════════════════════

class TestAdobeExtractText:
    def setup_method(self):
        reset_circuit_breaker()

    def teardown_method(self):
        reset_circuit_breaker()

    @pytest.mark.asyncio
    async def test_returns_none_when_disabled(self):
        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", False):
            result = await adobe_extract_text(Path("test.pdf"))
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_sdk_missing(self):
        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=False):
            result = await adobe_extract_text(Path("test.pdf"))
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_circuit_breaker_open(self):
        for _ in range(_CB_THRESHOLD):
            _cb_record_failure()
        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True):
            result = await adobe_extract_text(Path("test.pdf"))
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_file(self, tmp_path):
        f = tmp_path / "bad.pdf"
        f.write_bytes(b"not a pdf")
        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True):
            result = await adobe_extract_text(f)
            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_timeout(self, tmp_path):
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 content")

        async def slow_sync(*args, **kwargs):
            await asyncio.sleep(10)

        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True), \
             patch("app.pipeline.adobe_ocr.ADOBE_PDF_TIMEOUT", 0.1), \
             patch("app.pipeline.adobe_ocr._get_executor") as mock_exec:
            # Make run_in_executor raise timeout
            import asyncio as _asyncio
            loop = _asyncio.get_event_loop()
            with patch.object(loop, "run_in_executor", side_effect=lambda *a: asyncio.sleep(10)):
                result = await adobe_extract_text(f)
                assert result is None

    @pytest.mark.asyncio
    async def test_success_flow(self, tmp_path):
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 content")

        mock_result = {
            "total_pages": 1,
            "pages": [{"page_number": 1, "text": "Hello", "quality": {"quality": "HIGH", "char_count": 5, "word_count": 1, "cid_count": 0, "cid_ratio": 0.0, "reason": None}}],
            "full_text": "Hello",
            "extraction_quality": "HIGH",
            "metadata": {},
        }

        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True), \
             patch("app.pipeline.adobe_ocr._adobe_extract_sync", return_value=mock_result):
            result = await adobe_extract_text(f)
            assert result is not None
            assert result["total_pages"] == 1

    @pytest.mark.asyncio
    async def test_failure_trips_circuit_breaker(self, tmp_path):
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 content")

        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True), \
             patch("app.pipeline.adobe_ocr._adobe_extract_sync", return_value=None):
            for _ in range(_CB_THRESHOLD):
                await adobe_extract_text(f)
            assert _cb_is_open()

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, tmp_path):
        f = tmp_path / "valid.pdf"
        f.write_bytes(b"%PDF-1.4 content")

        mock_result = {
            "total_pages": 1,
            "pages": [],
            "full_text": "test",
            "extraction_quality": "HIGH",
            "metadata": {},
        }

        progress_calls = []
        async def mock_progress(stage, msg, detail):
            progress_calls.append((stage, msg))

        with patch("app.pipeline.adobe_ocr.ADOBE_PDF_ENABLED", True), \
             patch("app.pipeline.adobe_ocr._check_adobe_sdk", return_value=True), \
             patch("app.pipeline.adobe_ocr._adobe_extract_sync", return_value=mock_result):
            await adobe_extract_text(f, on_progress=mock_progress)
            assert len(progress_calls) >= 2  # start + success
            assert any("extracting" in msg.lower() for _, msg in progress_calls)
            assert any("extracted" in msg.lower() for _, msg in progress_calls)
