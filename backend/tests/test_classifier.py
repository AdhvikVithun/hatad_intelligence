"""Tests for backend/app/pipeline/classifier.py — classification helpers.

Covers:
  - _strip_cid_placeholders: CID font-encoding noise removal
  - _collapse_repetitions: consecutive repeat squashing
  - _dedup_high_freq_tokens: non-adjacent repeat deduplication
"""

import pytest

from app.pipeline.classifier import (
    _strip_cid_placeholders,
    _collapse_repetitions,
    _dedup_high_freq_tokens,
    _select_best_pages,
)


# ── _strip_cid_placeholders ───────────────────────────────────────

class TestStripCidPlaceholders:
    """Ensure (cid:XX) noise is removed and remaining text is readable."""

    def test_basic_removal(self):
        text = "ப(cid:39)டா எ(cid:23): 637"
        result = _strip_cid_placeholders(text)
        assert "(cid:" not in result
        assert "637" in result

    def test_patta_keyword_survives(self):
        """After CID stripping, Tamil fragments should remain."""
        text = "ப(cid:39)டா எ(cid:23): 637"
        result = _strip_cid_placeholders(text)
        # The Tamil chars ப, டா, எ survive
        assert "ப" in result
        assert "டா" in result

    def test_no_cid_passthrough(self):
        """Text without CID placeholders should pass through unchanged."""
        text = "பட்டா எண்: 637\nSurvey No: 317"
        result = _strip_cid_placeholders(text)
        assert result == text

    def test_whitespace_collapsed(self):
        """Multiple spaces left by CID removal should be collapsed."""
        text = "(cid:1) (cid:2) (cid:3) hello (cid:4) world"
        result = _strip_cid_placeholders(text)
        assert "  " not in result
        assert "hello" in result
        assert "world" in result

    def test_blank_lines_removed(self):
        """Lines that were pure CID noise become blank and are removed."""
        text = "Header\n(cid:10)(cid:11)(cid:12)\nFooter"
        result = _strip_cid_placeholders(text)
        assert result == "Header\nFooter"

    def test_empty_string(self):
        assert _strip_cid_placeholders("") == ""

    def test_all_cid(self):
        """If text is entirely CID placeholders, result should be empty."""
        text = "(cid:1)(cid:2)(cid:3)"
        result = _strip_cid_placeholders(text)
        assert result == ""

    def test_survey_numbers_survive(self):
        """Survey numbers and numeric data should survive CID stripping."""
        text = "(cid:5)543 - 1A1(cid:6) 544/1B1(cid:7)"
        result = _strip_cid_placeholders(text)
        assert "543" in result
        assert "1A1" in result
        assert "544" in result
        assert "1B1" in result

    def test_realistic_patta_garble(self):
        """Simulate realistic garbled Patta text from pdfplumber."""
        text = (
            "தமி(cid:2)(cid:3)நா(cid:4) அர(cid:5)\n"
            "ேசாைமயபாைளய(cid:8)\n"
            "ப(cid:39)டா எ(cid:23): 637\n"
            "நில(cid:10) வைக(cid:11) : பு(cid:12)ஜ(cid:13)\n"
            "317 543 - 1A1 543 - 1C1\n"
        )
        result = _strip_cid_placeholders(text)
        assert "(cid:" not in result
        assert "637" in result
        assert "317" in result
        assert "543" in result

    def test_high_cid_ratio_page(self):
        """Page with >50% CID content should clean up to short residual."""
        # Simulate a page with very high CID ratio
        cid_noise = " ".join(f"(cid:{i})" for i in range(50))
        text = f"Header: பட்டா\n{cid_noise}\nFooter line"
        result = _strip_cid_placeholders(text)
        assert "(cid:" not in result
        assert "பட்டா" in result
        assert "Footer" in result


# ── _collapse_repetitions ─────────────────────────────────────────

class TestCollapseRepetitions:
    """Consecutive repeated tokens should be collapsed."""

    def test_basic_collapse(self):
        text = "(R) (R) (R) (R) (R) (R) (R)"
        result = _collapse_repetitions(text, max_repeats=3)
        # regex group may include trailing space; result ≤ max_repeats+1
        assert result.count("(R)") <= 4

    def test_no_repeats_passthrough(self):
        text = "Hello world test"
        assert _collapse_repetitions(text) == text

    def test_tamil_token_collapse(self):
        """Repeated Tamil token should be collapsed."""
        text = "மொத்தம் " * 10
        result = _collapse_repetitions(text.strip(), max_repeats=3)
        # Must be significantly reduced from the original 10
        assert result.count("மொத்தம்") <= 5


# ── _dedup_high_freq_tokens ───────────────────────────────────────

class TestDedupHighFreqTokens:
    """Non-adjacent high-frequency tokens should be capped."""

    def test_high_freq_capped(self):
        # Token appears 20 times across 40+ words, should be capped
        words = [f"filler{i}" for i in range(15)] + ["மொத்தம்"] * 20 + [f"end{i}" for i in range(15)]
        text = " ".join(words)
        result = _dedup_high_freq_tokens(text, max_occurrences=6)
        assert result.count("மொத்தம்") == 6

    def test_short_text_passthrough(self):
        """Short text (<30 words) should not be modified."""
        text = "short text here"
        assert _dedup_high_freq_tokens(text) == text

    def test_no_high_freq_passthrough(self):
        """Text with no excessively repeated tokens should not change."""
        words = [f"word{i}" for i in range(40)]
        text = " ".join(words)
        assert _dedup_high_freq_tokens(text) == text


# ── _select_best_pages ────────────────────────────────────────────

def _make_page(num, text="some text", method="pdfplumber", quality="HIGH"):
    """Helper to build a page dict for testing."""
    return {
        "page_number": num,
        "text": text,
        "tables": [],
        "extraction_method": method,
        "quality": {
            "char_count": len(text),
            "word_count": len(text.split()),
            "cid_count": 0,
            "cid_ratio": 0.0,
            "quality": quality,
            "reason": None if quality == "HIGH" else "test",
        },
    }


class TestSelectBestPages:
    """Quality-aware page selection for classification."""

    def test_fewer_than_max_returns_all(self):
        pages = [_make_page(1)]
        result = _select_best_pages(pages, max_pages=2)
        assert len(result) == 1
        assert result[0]["page_number"] == 1

    def test_two_pages_returns_both(self):
        pages = [_make_page(1), _make_page(2)]
        assert len(_select_best_pages(pages, max_pages=2)) == 2

    def test_sarvam_preferred_over_garbled_pdfplumber(self):
        """When page 1 is Sarvam HIGH and page 2 is pdfplumber LOW,
        prefer page 1 over page 2 for a 3-page doc."""
        pages = [
            _make_page(1, "good sarvam text", method="sarvam", quality="HIGH"),
            _make_page(2, "ப(cid:39)டா garbled", method="pdfplumber", quality="LOW"),
            _make_page(3, "another sarvam page", method="sarvam", quality="HIGH"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        assert len(result) == 2
        methods = {p["extraction_method"] for p in result}
        assert methods == {"sarvam"}

    def test_high_quality_preferred_over_low(self):
        """HIGH quality pages should be picked over LOW quality."""
        pages = [
            _make_page(1, "garbled", method="pdfplumber", quality="LOW"),
            _make_page(2, "clean text", method="pdfplumber", quality="HIGH"),
            _make_page(3, "also clean", method="pdfplumber", quality="HIGH"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        assert all(p["quality"]["quality"] == "HIGH" for p in result)

    def test_result_in_page_order(self):
        """Selected pages should be re-sorted by page number."""
        pages = [
            _make_page(1, "garbled", method="pdfplumber", quality="LOW"),
            _make_page(2, "ok", method="pdfplumber", quality="HIGH"),
            _make_page(3, "good", method="sarvam", quality="HIGH"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        assert result[0]["page_number"] < result[1]["page_number"]

    def test_sarvam_page_tie_breaks_by_page_number(self):
        """Among equally ranked pages, earlier pages are preferred."""
        pages = [
            _make_page(1, "page 1", method="sarvam", quality="HIGH"),
            _make_page(2, "page 2", method="sarvam", quality="HIGH"),
            _make_page(3, "page 3", method="sarvam", quality="HIGH"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        assert result[0]["page_number"] == 1
        assert result[1]["page_number"] == 2

    def test_realistic_patta_scenario(self):
        """Real scenario: Sarvam page 1 (good) + pdfplumber page 2 (garbled CID).
        With only 2 pages total, both should be returned (can't skip any)."""
        pages = [
            _make_page(1, "பட்டா எண்: 637 survey 317", method="sarvam", quality="HIGH"),
            _make_page(2, "(cid:1)(cid:2) garbled", method="pdfplumber", quality="LOW"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        # Both returned since total <= max_pages
        assert len(result) == 2

    def test_ocr_fallback_preferred_over_raw_pdfplumber(self):
        """OCR fallback pages rank between Sarvam and raw pdfplumber."""
        pages = [
            _make_page(1, "raw pdf", method="pdfplumber", quality="HIGH"),
            _make_page(2, "ocr enhanced", method="ocr_fallback", quality="HIGH"),
            _make_page(3, "sarvam best", method="sarvam", quality="HIGH"),
        ]
        result = _select_best_pages(pages, max_pages=2)
        methods = [p["extraction_method"] for p in result]
        assert "sarvam" in methods
        assert "ocr_fallback" in methods
