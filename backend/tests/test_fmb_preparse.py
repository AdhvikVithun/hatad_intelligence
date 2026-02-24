"""Tests for FMB header pre-parser and survey number override logic.

Covers:
  - preparse_fmb_header() — deterministic regex extraction from header
  - _apply_header_overrides() — LLM result correction using header ground truth
"""

import pytest

from app.pipeline.extractors.fmb import preparse_fmb_header, _apply_header_overrides


# ═══════════════════════════════════════════════════
# 1. preparse_fmb_header — English headers
# ═══════════════════════════════════════════════════

class TestPreparseFmbHeaderEnglish:

    def test_standard_header(self):
        """Standard English FMB header — District, Taluk, Village, Survey No, Area."""
        text = (
            "District : Coimbatore                  Survey No : 317\n"
            "Taluk : Coimbatore (N)                  Area : Hect 00 Ares 92 Sqm 50\n"
            "Village : Somayampalayam.(R)            Scale : 1 : 1158 mm\n"
            "\n"
            "--- sketch body with adjacent surveys 543/1A1, 318/2 ---"
        )
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "317"
        assert "Coimbatore" in h["district"]
        assert "Coimbatore" in h["taluk"]
        assert "Somayampalayam" in h["village"]
        assert "Hect" in h.get("area_raw", "")

    def test_sf_no_format(self):
        """S.F.No. : 311/1A style header."""
        text = "S.F.No. : 311/1A\nVillage : Vadavalli\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "311/1A"

    def test_ts_no_format(self):
        """T.S.No. : 45/2 style header."""
        text = "T.S.No. : 45/2\nVillage : Sholinganallur\nTaluk : Tambaram\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "45/2"
        assert "Sholinganallur" in h["village"]

    def test_rs_no_format(self):
        """R.S.No. : 99 style header."""
        text = "R.S.No. : 99\nDistrict : Salem\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "99"

    def test_survey_no_dot_format(self):
        """Survey No. : 500/3B1 (with dot after No)."""
        text = "Survey No. : 500/3B1\nVillage : Perungudi\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "500/3B1"

    def test_dash_separator(self):
        """Survey No - 317 (dash instead of colon)."""
        text = "Survey No - 317\nVillage : Test\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "317"


# ═══════════════════════════════════════════════════
# 2. preparse_fmb_header — Tamil headers
# ═══════════════════════════════════════════════════

class TestPreparseFmbHeaderTamil:

    def test_tamil_survey_keyword(self):
        """சர்வே எண் : 317 (Tamil survey number label)."""
        text = "சர்வே எண் : 317\nகிராமம் : சோமயம்பாளையம்\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "317"
        assert "சோமயம்பாளையம்" in h["village"]

    def test_tamil_pula_number(self):
        """புல எண் : 45/2 (Tamil field number label)."""
        text = "புல எண் : 45/2\nவட்டம் : கோயம்புத்தூர்\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "45/2"
        assert "கோயம்புத்தூர்" in h["taluk"]

    def test_tamil_district(self):
        """மாவட்டம் : கோயம்புத்தூர் (Tamil district label)."""
        text = "மாவட்டம் : கோயம்புத்தூர்\nSurvey No : 100\n"
        h = preparse_fmb_header(text)
        assert "கோயம்புத்தூர்" in h["district"]
        assert h["survey_number"] == "100"


# ═══════════════════════════════════════════════════
# 3. preparse_fmb_header — Edge cases
# ═══════════════════════════════════════════════════

class TestPreparseFmbHeaderEdge:

    def test_empty_text(self):
        """Empty text → empty hints dict."""
        assert preparse_fmb_header("") == {}

    def test_garbled_text(self):
        """Garbled OCR text with no recognizable labels → empty dict."""
        text = "abc123 xyz789 random noise without labels"
        assert preparse_fmb_header(text) == {}

    def test_only_body_surveys_not_picked(self):
        """Text with survey numbers only in body, not in header format → no survey hint."""
        text = (
            "Some random FMB text\n"
            "Adjacent plots: 543/1A1, 318/2, 320\n"
            "Boundary: north 543/1A1\n"
        )
        h = preparse_fmb_header(text)
        assert "survey_number" not in h

    def test_header_beyond_800_chars_ignored(self):
        """Survey No in text beyond 800 chars should be ignored."""
        padding = "x" * 900
        text = padding + "\nSurvey No : 999\n"
        h = preparse_fmb_header(text)
        assert "survey_number" not in h

    def test_survey_number_with_subdivision(self):
        """Survey No : 317/1A1 — subdivision preserved."""
        text = "Survey No : 317/1A1\nVillage : Test\n"
        h = preparse_fmb_header(text)
        assert h["survey_number"] == "317/1A1"


# ═══════════════════════════════════════════════════
# 4. _apply_header_overrides
# ═══════════════════════════════════════════════════

class TestApplyHeaderOverrides:

    def test_override_wrong_survey(self):
        """LLM gives 543/1A1 but header says 317 → result corrected to 317 with red flag."""
        result = {"survey_number": "543/1A1", "village": "Somayampalayam"}
        header = {"survey_number": "317"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["survey_number"] == "317"
        flags = corrected.get("_extraction_red_flags", [])
        assert any("543/1A1" in f and "317" in f for f in flags)

    def test_no_override_when_matching(self):
        """LLM and header agree → no red flag added."""
        result = {"survey_number": "317", "village": "Somayampalayam"}
        header = {"survey_number": "317"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["survey_number"] == "317"
        flags = corrected.get("_extraction_red_flags", [])
        assert not any("corrected" in f.lower() for f in flags)

    def test_fill_empty_survey(self):
        """LLM gives empty survey → header fills it in."""
        result = {"survey_number": "", "village": "Somayampalayam"}
        header = {"survey_number": "317"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["survey_number"] == "317"

    def test_fill_missing_survey(self):
        """LLM gives no survey_number key → header fills it."""
        result = {"village": "Somayampalayam"}
        header = {"survey_number": "317"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["survey_number"] == "317"

    def test_fill_empty_village(self):
        """LLM leaves village empty → header fills it."""
        result = {"survey_number": "317", "village": ""}
        header = {"survey_number": "317", "village": "Somayampalayam"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["village"] == "Somayampalayam"

    def test_no_override_village_when_present(self):
        """LLM has village → header doesn't overwrite it."""
        result = {"survey_number": "317", "village": "Somayam Palayam"}
        header = {"survey_number": "317", "village": "Somayampalayam.(R)"}
        corrected = _apply_header_overrides(result, header)
        assert corrected["village"] == "Somayam Palayam"  # LLM value preserved

    def test_empty_header(self):
        """Empty header → result unchanged."""
        result = {"survey_number": "543/1A1", "village": "Test"}
        corrected = _apply_header_overrides(result, {})
        assert corrected["survey_number"] == "543/1A1"

    def test_none_header(self):
        """None header → result unchanged (should not crash)."""
        result = {"survey_number": "543/1A1"}
        corrected = _apply_header_overrides(result, None)
        assert corrected["survey_number"] == "543/1A1"

    def test_existing_red_flags_preserved(self):
        """Existing red flags should be preserved when adding override flag."""
        result = {
            "survey_number": "543/1A1",
            "_extraction_red_flags": ["Area doesn't match dimensions"],
        }
        header = {"survey_number": "317"}
        corrected = _apply_header_overrides(result, header)
        flags = corrected["_extraction_red_flags"]
        assert len(flags) == 2
        assert "Area doesn't match dimensions" in flags[0]
        assert "317" in flags[1]
