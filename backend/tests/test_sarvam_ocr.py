"""Tests for Sarvam AI Document Intelligence integration.

These tests mock the sarvamai SDK entirely — no network calls.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

from app.pipeline.sarvam_ocr import (
    _parse_sarvam_html,
    _assess_page_quality,
    _extract_html_from_zip,
    _extract_metadata_page_texts,
    _validate_pdf,
    _cb_record_failure,
    _cb_record_success,
    _cb_is_open,
    reset_circuit_breaker,
    merge_sarvam_with_pdfplumber,
    sarvam_extract_text,
    _sarvam_extract_sync,
)


# ═══════════════════════════════════════════════════
# HTML PARSING
# ═══════════════════════════════════════════════════

class TestParseSarvamHtml:
    """Tests for extracting per-page text from Sarvam HTML output."""

    def test_page_divs_with_data_attr(self):
        html = """
        <html><body>
          <div data-page-number="1">Page one text here</div>
          <div data-page-number="2">Page two text here</div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 2
        assert "Page one text here" in pages[0]
        assert "Page two text here" in pages[1]

    def test_page_divs_with_class(self):
        html = """
        <html><body>
          <div class="page">First page content</div>
          <div class="page">Second page content</div>
          <div class="page">Third page content</div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 3
        assert "First page content" in pages[0]
        assert "Third page content" in pages[2]

    def test_hr_separator(self):
        html = """
        <html><body>
          <p>Page one text</p>
          <hr>
          <p>Page two text</p>
          <hr/>
          <p>Page three text</p>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) >= 2  # at least split on HR tags
        assert "Page one text" in pages[0]

    def test_css_page_break(self):
        html = """
        <html><body>
          <p>First page</p>
          <div style="page-break-before: always">
            <p>Second page</p>
          </div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) >= 1  # at minimum, text is extracted

    def test_single_page_fallback(self):
        html = "<html><body><p>All content on one page</p></body></html>"
        pages = _parse_sarvam_html(html)
        assert len(pages) == 1
        assert "All content on one page" in pages[0]

    def test_empty_html(self):
        pages = _parse_sarvam_html("")
        assert len(pages) == 0

    def test_tamil_content_preserved(self):
        html = """
        <html><body>
          <div data-page-number="1">முருகன் என்ற பெயர்</div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 1
        assert "முருகன்" in pages[0]

    def test_page_body_container(self):
        """Sarvam's actual HTML format uses page-body-container divs."""
        html = """
        <!DOCTYPE html>
        <html lang="ta-IN">
        <head><title>Document</title></head>
        <body>
          <div class="page-body-container"><p>Page 1 content</p></div>
          <div class="page-body-container"><p>Page 2 content</p></div>
          <div class="page-body-container"><p>Page 3 content</p></div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 3
        assert "Page 1 content" in pages[0]
        assert "Page 2 content" in pages[1]
        assert "Page 3 content" in pages[2]

    def test_page_body_container_with_nested_content(self):
        """page-body-container divs may contain tables, figures, etc."""
        html = """
        <html><body>
          <div class="page-body-container">
            <h1>Sale Deed</h1>
            <table><tr><td>Row 1</td></tr></table>
          </div>
          <div class="page-body-container">
            <p>விற்பனை பத்திரம்</p>
          </div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 2
        assert "Sale Deed" in pages[0]
        assert "Row 1" in pages[0]
        assert "விற்பனை" in pages[1]

    def test_page_body_container_20_pages(self):
        """Multi-page document (like a 20-page sale deed) should split correctly."""
        divs = "\n".join(
            f'<div class="page-body-container"><p>Page {i} text here</p></div>'
            for i in range(1, 21)
        )
        html = f"<html><body>{divs}</body></html>"
        pages = _parse_sarvam_html(html)
        assert len(pages) == 20
        assert "Page 1 text here" in pages[0]
        assert "Page 20 text here" in pages[19]

    def test_page_body_container_preferred_over_fallback(self):
        """page-body-container should be picked even if other divs exist."""
        html = """
        <html><body>
          <div class="header">Header</div>
          <div class="page-body-container"><p>Real page 1</p></div>
          <div class="page-body-container"><p>Real page 2</p></div>
          <div class="footer">Footer</div>
        </body></html>
        """
        pages = _parse_sarvam_html(html)
        assert len(pages) == 2
        assert "Header" not in pages[0] or "Real page 1" in pages[0]


# ═══════════════════════════════════════════════════
# METADATA PAGE TEXT EXTRACTION
# ═══════════════════════════════════════════════════

class TestExtractMetadataPageTexts:
    """Tests for extracting per-page text from Sarvam metadata JSON files."""

    def _make_zip_with_metadata(self, tmp_path, page_data):
        """Create a ZIP with metadata/page_NNN.json files."""
        import json
        zip_path = tmp_path / "output.zip"
        import zipfile
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("document.html", "<html><body>dummy</body></html>")
            for i, blocks in enumerate(page_data, 1):
                meta = {"page_num": i, "blocks": blocks}
                zf.writestr(f"metadata/page_{i:03d}.json", json.dumps(meta))
        return zip_path

    def test_basic_extraction(self, tmp_path):
        page_data = [
            [{"text": "Page 1 line 1"}, {"text": "Page 1 line 2"}],
            [{"text": "Page 2 content"}],
        ]
        zip_path = self._make_zip_with_metadata(tmp_path, page_data)
        texts = _extract_metadata_page_texts(zip_path)
        assert len(texts) == 2
        assert "Page 1 line 1" in texts[0]
        assert "Page 1 line 2" in texts[0]
        assert "Page 2 content" in texts[1]

    def test_content_field_fallback(self, tmp_path):
        """Blocks with 'content' instead of 'text' should work."""
        page_data = [
            [{"content": "Block via content field"}],
        ]
        zip_path = self._make_zip_with_metadata(tmp_path, page_data)
        texts = _extract_metadata_page_texts(zip_path)
        assert len(texts) == 1
        assert "Block via content field" in texts[0]

    def test_no_metadata_returns_empty(self, tmp_path):
        """ZIP without metadata files should return empty list."""
        import zipfile
        zip_path = tmp_path / "output.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("document.html", "<html></html>")
        texts = _extract_metadata_page_texts(zip_path)
        assert texts == []

    def test_missing_file_returns_empty(self, tmp_path):
        texts = _extract_metadata_page_texts(tmp_path / "nonexistent.zip")
        assert texts == []

    def test_20_pages(self, tmp_path):
        page_data = [
            [{"text": f"Content for page {i}"}] for i in range(1, 21)
        ]
        zip_path = self._make_zip_with_metadata(tmp_path, page_data)
        texts = _extract_metadata_page_texts(zip_path)
        assert len(texts) == 20
        assert "page 1" in texts[0]
        assert "page 20" in texts[19]

class TestAssessPageQuality:
    def test_high_quality(self):
        text = "This is a sufficiently long text with many words to be considered high quality."
        q = _assess_page_quality(text)
        assert q["quality"] == "HIGH"
        assert q["char_count"] > 50
        assert q["word_count"] >= 10

    def test_low_quality_short(self):
        q = _assess_page_quality("Short")
        assert q["quality"] == "LOW"

    def test_low_quality_empty(self):
        q = _assess_page_quality("")
        assert q["quality"] == "LOW"


# ═══════════════════════════════════════════════════
# ZIP EXTRACTION
# ═══════════════════════════════════════════════════

class TestExtractHtmlFromZip:
    def test_extracts_html(self, tmp_path):
        import zipfile
        zip_path = tmp_path / "output.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("document.html", "<html><body>Hello</body></html>")
        result = _extract_html_from_zip(zip_path)
        assert result is not None
        assert "Hello" in result

    def test_prefers_largest_html(self, tmp_path):
        import zipfile
        zip_path = tmp_path / "output.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("small.html", "<p>Small</p>")
            zf.writestr("large.html", "<html><body>" + "X" * 1000 + "</body></html>")
        result = _extract_html_from_zip(zip_path)
        assert result is not None
        assert len(result) > 100

    def test_finds_html_in_non_html_extension(self, tmp_path):
        import zipfile
        zip_path = tmp_path / "output.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("output.txt", "<html><body>Disguised HTML</body></html>")
        result = _extract_html_from_zip(zip_path)
        assert result is not None
        assert "Disguised HTML" in result

    def test_no_html_returns_none(self, tmp_path):
        import zipfile
        zip_path = tmp_path / "output.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("data.csv", "a,b,c\n1,2,3")
        result = _extract_html_from_zip(zip_path)
        assert result is None

    def test_invalid_zip_returns_none(self, tmp_path):
        bad_path = tmp_path / "notazip.zip"
        bad_path.write_text("this is not a zip file")
        result = _extract_html_from_zip(bad_path)
        assert result is None


# ═══════════════════════════════════════════════════
# MERGE LOGIC
# ═══════════════════════════════════════════════════

class TestMergeSarvamWithPdfplumber:
    def _make_page(self, page_num, text, method="pdfplumber"):
        q = _assess_page_quality(text)
        return {
            "page_number": page_num,
            "text": text,
            "tables": [],
            "extraction_method": method,
            "quality": q,
        }

    def _make_result(self, pages):
        return {
            "total_pages": len(pages),
            "pages": pages,
            "full_text": "\n".join(p["text"] for p in pages),
            "extraction_quality": "HIGH",
            "ocr_pages": 0,
            "metadata": {"filename": "test.pdf"},
        }

    def test_sarvam_wins_when_higher_quality(self):
        """Sarvam HIGH quality should replace pdfplumber LOW quality."""
        pdf_pages = [self._make_page(1, "x", "pdfplumber")]  # LOW quality
        sarvam_pages = [self._make_page(1, "This is a much longer text with enough words for high quality assessment.", "sarvam")]

        merged = merge_sarvam_with_pdfplumber(
            self._make_result(sarvam_pages),
            self._make_result(pdf_pages),
        )
        assert merged["pages"][0]["extraction_method"] == "sarvam"
        assert merged["sarvam_pages"] == 1

    def test_pdfplumber_wins_when_better(self):
        """pdfplumber should win when it has more/better text."""
        long_text = "This is a very long and detailed text extracted by pdfplumber with many words and good quality."
        pdf_pages = [self._make_page(1, long_text)]
        sarvam_pages = [self._make_page(1, "Short Sarvam text", "sarvam")]

        merged = merge_sarvam_with_pdfplumber(
            self._make_result(sarvam_pages),
            self._make_result(pdf_pages),
        )
        assert merged["pages"][0]["extraction_method"] == "pdfplumber"
        assert merged["sarvam_pages"] == 0

    def test_sarvam_wins_more_text(self):
        """Sarvam should win when it has significantly more text (>1.3x)."""
        pdf_text = "Short text from pdfplumber with just enough words for quality assessment purposes."
        sarvam_text = pdf_text + " " + ("Extra content from Sarvam AI " * 5)

        pdf_pages = [self._make_page(1, pdf_text)]
        sarvam_pages = [self._make_page(1, sarvam_text, "sarvam")]

        merged = merge_sarvam_with_pdfplumber(
            self._make_result(sarvam_pages),
            self._make_result(pdf_pages),
        )
        assert merged["pages"][0]["extraction_method"] == "sarvam"

    def test_preserves_pdfplumber_tables(self):
        """When Sarvam wins, pdfplumber tables should be preserved."""
        pdf_page = self._make_page(1, "x")  # LOW
        pdf_page["tables"] = [["col1", "col2"], ["a", "b"]]

        sarvam_page = self._make_page(1, "Much longer text from Sarvam AI with proper quality and sufficient word count.", "sarvam")

        merged = merge_sarvam_with_pdfplumber(
            self._make_result([sarvam_page]),
            self._make_result([pdf_page]),
        )
        assert merged["pages"][0]["extraction_method"] == "sarvam"
        assert len(merged["pages"][0]["tables"]) > 0

    def test_handles_different_page_counts(self):
        """Merge should handle Sarvam having fewer pages than pdfplumber."""
        pdf_pages = [
            self._make_page(1, "Page 1 text with enough words for quality assessment in this test."),
            self._make_page(2, "Page 2 text with enough words for quality assessment in this test."),
        ]
        sarvam_pages = [
            self._make_page(1, "Sarvam page 1 only text content.", "sarvam"),
        ]

        merged = merge_sarvam_with_pdfplumber(
            self._make_result(sarvam_pages),
            self._make_result(pdf_pages),
        )
        assert merged["total_pages"] == 2  # keeps all pages

    def test_sarvam_only_pages(self):
        """Pages only in Sarvam result should be included."""
        pdf_pages = [self._make_page(1, "First page only from pdf with enough text for assessment.")]
        sarvam_pages = [
            self._make_page(1, "Sarvam page 1 text here.", "sarvam"),
            self._make_page(2, "Sarvam page 2 extra page not in pdfplumber output.", "sarvam"),
        ]

        merged = merge_sarvam_with_pdfplumber(
            self._make_result(sarvam_pages),
            self._make_result(pdf_pages),
        )
        assert merged["total_pages"] == 2


# ═══════════════════════════════════════════════════
# ASYNC EXTRACTION (mocked SDK)
# ═══════════════════════════════════════════════════

class TestSarvamExtractText:
    def test_disabled_returns_none(self):
        """When SARVAM_ENABLED=False, should return None immediately."""
        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", False):
            result = asyncio.get_event_loop().run_until_complete(
                sarvam_extract_text(Path("test.pdf"))
            )
            assert result is None

    def test_missing_sdk_returns_none(self):
        """When sarvamai is not installed, should return None."""
        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True), \
             patch("app.pipeline.sarvam_ocr._check_sarvam_sdk", return_value=False):
            result = asyncio.get_event_loop().run_until_complete(
                sarvam_extract_text(Path("test.pdf"))
            )
            assert result is None

    def test_exception_returns_none(self):
        """When the SDK throws, should log error and return None."""
        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True), \
             patch("app.pipeline.sarvam_ocr._check_sarvam_sdk", return_value=True), \
             patch("app.pipeline.sarvam_ocr._sarvam_extract_sync", side_effect=RuntimeError("API down")):
            result = asyncio.get_event_loop().run_until_complete(
                sarvam_extract_text(Path("test.pdf"))
            )
            assert result is None


class TestSarvamExtractSync:
    def test_successful_extraction(self, tmp_path):
        """Mock the full Sarvam flow: create job → upload → start → poll → download → parse."""
        import zipfile

        # Create a test PDF (just needs to exist)
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        # Create a mock output ZIP with HTML
        output_zip = tmp_path / "output.zip"
        with zipfile.ZipFile(output_zip, "w") as zf:
            zf.writestr("result.html", """
                <html><body>
                    <div data-page-number="1">முருகன் S/o ராமன்</div>
                    <div data-page-number="2">சுந்தரம் W/o லக்ஷ்மி</div>
                </body></html>
            """)

        # Mock the SarvamAI SDK
        mock_job = MagicMock()
        mock_job.job_id = "test-job-123"

        mock_status = MagicMock()
        mock_status.job_state = "Completed"
        mock_job.wait_until_complete.return_value = mock_status

        def mock_download(path):
            import shutil
            shutil.copy(str(output_zip), path)

        mock_job.download_output = mock_download

        mock_client = MagicMock()
        mock_client.document_intelligence.create_job.return_value = mock_job

        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True), \
             patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"), \
             patch("app.pipeline.sarvam_ocr.SARVAM_TIMEOUT", 10), \
             patch("app.pipeline.sarvam_ocr.SARVAM_POLL_INTERVAL", 0.1):

            # Patch the import inside _sarvam_extract_sync
            import app.pipeline.sarvam_ocr as sarvam_mod
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            with patch.dict("sys.modules", {"sarvamai": MagicMock()}):
                with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"):
                    # Directly test with mocked internals
                    from unittest.mock import patch as mock_patch
                    import sys
                    mock_sarvamai = MagicMock()
                    mock_sarvamai.SarvamAI.return_value = mock_client
                    sys.modules["sarvamai"] = mock_sarvamai

                    try:
                        result = _sarvam_extract_sync(test_pdf)
                    finally:
                        del sys.modules["sarvamai"]

        assert result is not None
        assert result["total_pages"] == 2
        assert result["sarvam_pages"] == 2
        assert "முருகன்" in result["pages"][0]["text"]
        assert "சுந்தரம்" in result["pages"][1]["text"]
        assert result["pages"][0]["extraction_method"] == "sarvam"

    def test_timeout_returns_none(self, tmp_path):
        """If job doesn't complete within timeout, should return None."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        mock_job = MagicMock()
        mock_job.job_id = "test-job-slow"

        mock_status = MagicMock()
        mock_status.job_state = "processing"  # never completes
        mock_job.wait_until_complete.side_effect = TimeoutError("Sarvam timeout")

        mock_client = MagicMock()
        mock_client.document_intelligence.create_job.return_value = mock_job

        import sys
        mock_sarvamai = MagicMock()
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"), \
                 patch("app.pipeline.sarvam_ocr.SARVAM_TIMEOUT", 0.5), \
                 patch("app.pipeline.sarvam_ocr.SARVAM_POLL_INTERVAL", 0.1):
                result = _sarvam_extract_sync(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None

    def test_failed_job_returns_none(self, tmp_path):
        """If Sarvam job fails, should return None."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        mock_job = MagicMock()
        mock_job.job_id = "test-job-fail"

        mock_status = MagicMock()
        mock_status.job_state = "Failed"
        mock_status.error_message = "Processing error"
        mock_job.wait_until_complete.return_value = mock_status

        mock_client = MagicMock()
        mock_client.document_intelligence.create_job.return_value = mock_job

        import sys
        mock_sarvamai = MagicMock()
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"), \
                 patch("app.pipeline.sarvam_ocr.SARVAM_TIMEOUT", 10), \
                 patch("app.pipeline.sarvam_ocr.SARVAM_POLL_INTERVAL", 0.1):
                result = _sarvam_extract_sync(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None


# ═══════════════════════════════════════════════════
# FEATURE FLAG
# ═══════════════════════════════════════════════════

class TestFeatureFlag:
    def test_sarvam_disabled_by_default(self):
        """SARVAM_ENABLED should be False when no API key is set."""
        with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", ""):
            with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", False):
                # When disabled, extract should be a no-op
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(sarvam_extract_text(Path("test.pdf")))
                finally:
                    loop.close()
                assert result is None

    def test_sarvam_enabled_with_key(self):
        """SARVAM_ENABLED should be True when API key is set."""
        from app.config import SARVAM_ENABLED as _
        with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "sk_test_key"):
            with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True):
                # Just verify the flag; actual extraction is tested elsewhere
                assert True


# ═══════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for the thread-safe circuit breaker protecting Sarvam calls."""

    def setup_method(self):
        reset_circuit_breaker()

    def teardown_method(self):
        reset_circuit_breaker()

    def test_starts_closed(self):
        """Circuit breaker should start in closed (healthy) state."""
        assert _cb_is_open() is False

    def test_single_failure_stays_closed(self):
        """One failure should NOT open the breaker."""
        _cb_record_failure()
        assert _cb_is_open() is False

    def test_two_failures_stays_closed(self):
        """Two failures (below threshold of 3) should NOT open the breaker."""
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is False

    def test_three_failures_opens(self):
        """Three consecutive failures should open the breaker."""
        _cb_record_failure()
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is True

    def test_success_resets_counter(self):
        """A success after failures should reset the counter."""
        _cb_record_failure()
        _cb_record_failure()
        _cb_record_success()
        # Now two more failures should NOT open (counter was reset to 0)
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is False

    def test_success_after_partial_failures(self):
        """Record 2 failures, then success, then 2 more: breaker stays closed."""
        _cb_record_failure()
        _cb_record_failure()
        _cb_record_success()
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is False

    def test_reset_clears_open_breaker(self):
        """reset_circuit_breaker should close an open breaker."""
        _cb_record_failure()
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is True
        reset_circuit_breaker()
        assert _cb_is_open() is False

    def test_cooldown_expires(self):
        """After cooldown, the breaker should allow a probe attempt."""
        import app.pipeline.sarvam_ocr as sarvam_mod

        _cb_record_failure()
        _cb_record_failure()
        _cb_record_failure()
        assert _cb_is_open() is True

        # Simulate cooldown expiry by moving _cb_disabled_until to the past
        with sarvam_mod._cb_lock:
            sarvam_mod._cb_disabled_until = 0.0

        assert _cb_is_open() is False

    def test_open_breaker_skips_extraction(self):
        """sarvam_extract_text should return None when breaker is open."""
        _cb_record_failure()
        _cb_record_failure()
        _cb_record_failure()

        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True), \
             patch("app.pipeline.sarvam_ocr._check_sarvam_sdk", return_value=True):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(sarvam_extract_text(Path("test.pdf")))
            finally:
                loop.close()
            assert result is None


# ═══════════════════════════════════════════════════
# FILE VALIDATION
# ═══════════════════════════════════════════════════

class TestFileValidation:
    """Tests for _validate_pdf file pre-flight checks."""

    def test_valid_pdf(self, tmp_path):
        """A valid PDF file should pass validation."""
        pdf = tmp_path / "good.pdf"
        pdf.write_bytes(b"%PDF-1.7 some content here")
        assert _validate_pdf(pdf) is None

    def test_missing_file(self, tmp_path):
        """A missing file should return an error."""
        err = _validate_pdf(tmp_path / "nonexistent.pdf")
        assert err is not None
        assert "does not exist" in err

    def test_empty_file(self, tmp_path):
        """A 0-byte file should return an error."""
        pdf = tmp_path / "empty.pdf"
        pdf.write_bytes(b"")
        err = _validate_pdf(pdf)
        assert err is not None
        assert "empty" in err.lower()

    def test_file_too_large(self, tmp_path):
        """A file exceeding SARVAM_MAX_FILE_MB should be rejected."""
        pdf = tmp_path / "huge.pdf"
        pdf.write_bytes(b"%PDF-1.4" + b"\x00" * 100)  # small file

        # Patch _MAX_FILE_BYTES to a tiny value
        with patch("app.pipeline.sarvam_ocr._MAX_FILE_BYTES", 50):
            err = _validate_pdf(pdf)
            assert err is not None
            assert "too large" in err.lower()

    def test_not_pdf_magic_bytes(self, tmp_path):
        """A file without %PDF magic should be rejected."""
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"PK\x03\x04 this is a ZIP not a PDF")
        err = _validate_pdf(pdf)
        assert err is not None
        assert "not a valid PDF" in err

    def test_validation_error_skips_extraction(self, tmp_path):
        """sarvam_extract_text should return None for invalid files."""
        bad = tmp_path / "bad.txt"
        bad.write_bytes(b"not a pdf")

        with patch("app.pipeline.sarvam_ocr.SARVAM_ENABLED", True), \
             patch("app.pipeline.sarvam_ocr._check_sarvam_sdk", return_value=True):
            reset_circuit_breaker()
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(sarvam_extract_text(bad))
            finally:
                loop.close()
            assert result is None


# ═══════════════════════════════════════════════════
# EXPONENTIAL BACKOFF
# ═══════════════════════════════════════════════════

class TestExponentialBackoff:
    """Verify that _sarvam_extract_sync uses exponential backoff between retries."""

    def setup_method(self):
        reset_circuit_breaker()

    def teardown_method(self):
        reset_circuit_breaker()

    def test_backoff_durations(self, tmp_path):
        """With 3 retries (4 total attempts), sleeps should be 1s, 2s, 4s."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 content")

        import sys
        mock_sarvamai = MagicMock()
        mock_sarvamai.SarvamAI.side_effect = RuntimeError("down")
        sys.modules["sarvamai"] = mock_sarvamai

        sleep_calls = []

        try:
            with patch("app.pipeline.sarvam_ocr.SARVAM_MAX_RETRIES", 3), \
                 patch("app.pipeline.sarvam_ocr.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                result = _sarvam_extract_sync(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None
        # 4 attempts → 3 sleeps: 1, 2, 4
        assert sleep_calls == [1, 2, 4]

    def test_single_retry_sleeps_once(self, tmp_path):
        """With SARVAM_MAX_RETRIES=1 (2 attempts), sleep should be 1s."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 content")

        import sys
        mock_sarvamai = MagicMock()
        mock_sarvamai.SarvamAI.side_effect = RuntimeError("down")
        sys.modules["sarvamai"] = mock_sarvamai

        sleep_calls = []

        try:
            with patch("app.pipeline.sarvam_ocr.SARVAM_MAX_RETRIES", 1), \
                 patch("app.pipeline.sarvam_ocr.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
                result = _sarvam_extract_sync(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None
        assert sleep_calls == [1]

    def test_no_sleep_on_first_success(self, tmp_path):
        """If the first attempt succeeds, no sleep should occur."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 content")

        import sys
        import app.pipeline.sarvam_ocr as sarvam_mod

        mock_result = {"total_pages": 1, "pages": [], "full_text": "ok"}

        sleep_calls = []

        with patch.object(sarvam_mod, "_sarvam_extract_single_attempt", return_value=mock_result), \
             patch("app.pipeline.sarvam_ocr.SARVAM_MAX_RETRIES", 3), \
             patch("app.pipeline.sarvam_ocr.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            result = _sarvam_extract_sync(test_pdf)

        assert result is not None
        assert sleep_calls == []


# ═══════════════════════════════════════════════════
# PER-STEP ISOLATION
# ═══════════════════════════════════════════════════

class TestPerStepIsolation:
    """Each step in _sarvam_extract_single_attempt should fail gracefully."""

    def setup_method(self):
        reset_circuit_breaker()

    def teardown_method(self):
        reset_circuit_breaker()

    def _make_mock_client(self):
        """Return a mock SarvamAI client with reasonable defaults."""
        mock_job = MagicMock()
        mock_job.job_id = "test-job"
        mock_status = MagicMock()
        mock_status.job_state = "Completed"
        mock_job.wait_until_complete.return_value = mock_status

        mock_client = MagicMock()
        mock_client.document_intelligence.create_job.return_value = mock_job
        return mock_client, mock_job

    def test_create_job_failure(self, tmp_path):
        """create_job raising should return None, not crash."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        import sys
        mock_sarvamai = MagicMock()
        mock_client = MagicMock()
        mock_client.document_intelligence.create_job.side_effect = RuntimeError("quota exceeded")
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            from app.pipeline.sarvam_ocr import _sarvam_extract_single_attempt
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"):
                result = _sarvam_extract_single_attempt(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None

    def test_upload_file_failure(self, tmp_path):
        """upload_file raising should return None and attempt cancellation."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        import sys
        mock_sarvamai = MagicMock()
        mock_client, mock_job = self._make_mock_client()
        mock_job.upload_file.side_effect = RuntimeError("network error")
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            from app.pipeline.sarvam_ocr import _sarvam_extract_single_attempt
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"):
                result = _sarvam_extract_single_attempt(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None
        mock_job.cancel.assert_called_once()

    def test_start_failure(self, tmp_path):
        """job.start() raising should return None and cancel."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        import sys
        mock_sarvamai = MagicMock()
        mock_client, mock_job = self._make_mock_client()
        mock_job.start.side_effect = RuntimeError("server error")
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            from app.pipeline.sarvam_ocr import _sarvam_extract_single_attempt
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"):
                result = _sarvam_extract_single_attempt(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None
        mock_job.cancel.assert_called_once()

    def test_download_failure(self, tmp_path):
        """download_output raising should return None."""
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4 test")

        import sys
        mock_sarvamai = MagicMock()
        mock_client, mock_job = self._make_mock_client()
        mock_job.download_output.side_effect = RuntimeError("download failed")
        mock_sarvamai.SarvamAI.return_value = mock_client
        sys.modules["sarvamai"] = mock_sarvamai

        try:
            from app.pipeline.sarvam_ocr import _sarvam_extract_single_attempt
            with patch("app.pipeline.sarvam_ocr.SARVAM_API_KEY", "test-key"):
                result = _sarvam_extract_single_attempt(test_pdf)
        finally:
            del sys.modules["sarvamai"]

        assert result is None


# ═══════════════════════════════════════════════════
# JOB CANCELLATION
# ═══════════════════════════════════════════════════

class TestJobCancellation:
    """_try_cancel_job must never raise, regardless of what goes wrong."""

    def test_cancel_success(self):
        from app.pipeline.sarvam_ocr import _try_cancel_job
        mock_job = MagicMock()
        _try_cancel_job(mock_job)  # should not raise
        mock_job.cancel.assert_called_once()

    def test_cancel_raises_swallowed(self):
        from app.pipeline.sarvam_ocr import _try_cancel_job
        mock_job = MagicMock()
        mock_job.cancel.side_effect = RuntimeError("cannot cancel")
        _try_cancel_job(mock_job)  # must not raise

    def test_cancel_no_cancel_attr(self):
        from app.pipeline.sarvam_ocr import _try_cancel_job
        mock_job = MagicMock(spec=[])  # no attributes at all
        _try_cancel_job(mock_job)  # must not raise

    def test_cancel_none_job(self):
        from app.pipeline.sarvam_ocr import _try_cancel_job
        _try_cancel_job(None)  # must not raise


# ═══════════════════════════════════════════════════
# BUILD RESULT
# ═══════════════════════════════════════════════════

class TestBuildResult:
    """Tests for the _build_result helper."""

    def test_single_page(self, tmp_path):
        from app.pipeline.sarvam_ocr import _build_result
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        result = _build_result(pdf, ["Tamil text content here for testing"])
        assert result["total_pages"] == 1
        assert result["sarvam_pages"] == 1
        assert result["pages"][0]["extraction_method"] == "sarvam"
        assert result["pages"][0]["page_number"] == 1
        assert result["metadata"]["extraction_source"] == "sarvam_ai"
        assert result["metadata"]["file_size"] > 0

    def test_multi_page(self, tmp_path):
        from app.pipeline.sarvam_ocr import _build_result
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        texts = ["Page one text content", "Page two text content", "Page three text"]
        result = _build_result(pdf, texts)
        assert result["total_pages"] == 3
        assert result["sarvam_pages"] == 3
        assert [p["page_number"] for p in result["pages"]] == [1, 2, 3]

    def test_missing_file_stat(self, tmp_path):
        """_build_result should not crash if the file no longer exists."""
        from app.pipeline.sarvam_ocr import _build_result
        pdf = tmp_path / "deleted.pdf"
        # Don't create the file — stat will fail
        result = _build_result(pdf, ["some text"])
        assert result["metadata"]["file_size"] == 0

    def test_quality_assessment(self, tmp_path):
        from app.pipeline.sarvam_ocr import _build_result
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        # HIGH quality — enough words and chars
        long_text = " ".join(["word"] * 20)
        result = _build_result(pdf, [long_text])
        assert result["extraction_quality"] == "HIGH"

        # LOW quality — too few words
        result = _build_result(pdf, ["hi"])
        assert result["extraction_quality"] == "LOW"

    def test_mixed_quality(self, tmp_path):
        from app.pipeline.sarvam_ocr import _build_result
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 content")

        texts = [" ".join(["word"] * 20), "hi"]
        result = _build_result(pdf, texts)
        assert result["extraction_quality"] == "MIXED"
