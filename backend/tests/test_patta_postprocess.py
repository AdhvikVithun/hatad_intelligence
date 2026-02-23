"""Tests for Patta post-processing — husband-wife owner merging.

Covers:
  - _merge_wife_husband_owners: raw text based merging
  - _merge_wife_husband_owners: fallback inference from owner list
  - _post_process_patta: integration with full post-processing pipeline
"""

import pytest

from app.pipeline.extractors.patta import (
    _post_process_patta,
    _merge_wife_husband_owners,
    _normalize_owner_key,
)


# ═══════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════

def _make_owner(name: str, father_name: str = "", share: str = "") -> dict:
    return {"name": name, "father_name": father_name, "share": share}


# ═══════════════════════════════════════════════════
# _normalize_owner_key
# ═══════════════════════════════════════════════════

class TestNormalizeOwnerKey:

    def test_strips_ascii_initials(self):
        """Strips single-letter ASCII initials like 'A.'."""
        assert _normalize_owner_key("A. Ponnarasi") == "ponnarasi"

    def test_strips_spaces(self):
        assert _normalize_owner_key("  பொன் அரசி  ") == "பொன்அரசி"

    def test_lowercase(self):
        assert _normalize_owner_key("PONNARASI") == "ponnarasi"

    def test_empty(self):
        assert _normalize_owner_key("") == ""

    def test_tamil_initials_kept_for_consistent_comparison(self):
        """Tamil multi-codepoint initials like 'என்.' are kept but both sides match."""
        # Both normalized forms will contain the Tamil initial, so comparison works
        key = _normalize_owner_key("என்.துளசிராம்")
        assert "துளசிராம்" in key


# ═══════════════════════════════════════════════════
# _merge_wife_husband_owners (raw text based)
# ═══════════════════════════════════════════════════

class TestMergeWifeHusbandFromRawText:
    """Test merging when raw_text contains 'Husband மனைவி Wife' patterns."""

    def test_three_pairs_merged_to_three_wives(self):
        """6 owners (3 husband + 3 wife) → 3 wife owners."""
        raw_text = (
            "என்.துளசிராம் மனைவி சத்தயபாமா\n"
            "ராதாகிருஷ்ணன் மனைவி ராணி\n"
            "அருள்சிங் மனைவி பொன் அரசி\n"
        )
        result = {
            "owner_names": [
                _make_owner("என்.துளசிராம்"),
                _make_owner("சத்தயபாமா"),
                _make_owner("ராதாகிருஷ்ணன்"),
                _make_owner("ராணி"),
                _make_owner("அருள்சிங்"),
                _make_owner("பொன் அரசி"),
            ]
        }
        _merge_wife_husband_owners(result, raw_text)
        owners = result["owner_names"]
        assert len(owners) == 3
        names = {o["name"] for o in owners}
        assert "சத்தயபாமா" in names
        assert "ராணி" in names
        assert "பொன் அரசி" in names
        # Husbands should NOT be in names
        assert "என்.துளசிராம்" not in names
        assert "ராதாகிருஷ்ணன்" not in names
        assert "அருள்சிங்" not in names

    def test_wife_gets_husband_as_father_name(self):
        """After merge, wife's father_name is set to husband."""
        raw_text = "துளசிராம் மனைவி சத்தயபாமா\n"
        result = {
            "owner_names": [
                _make_owner("துளசிராம்"),
                _make_owner("சத்தயபாமா"),
            ]
        }
        _merge_wife_husband_owners(result, raw_text)
        assert len(result["owner_names"]) == 1
        wife = result["owner_names"][0]
        assert wife["name"] == "சத்தயபாமா"
        assert wife["father_name"] == "துளசிராம்"

    def test_w_o_english_pattern(self):
        """'W/o' English pattern also works."""
        raw_text = "Tulsiram W/o Sathyabama\n"
        result = {
            "owner_names": [
                _make_owner("Tulsiram"),
                _make_owner("Sathyabama"),
            ]
        }
        _merge_wife_husband_owners(result, raw_text)
        assert len(result["owner_names"]) == 1
        assert result["owner_names"][0]["name"] == "Sathyabama"

    def test_already_correct_no_change(self):
        """If owners are already correctly extracted (3 wives only), no merge needed."""
        raw_text = "துளசிராம் மனைவி சத்தயபாமா\n"
        result = {
            "owner_names": [
                _make_owner("சத்தயபாமா", father_name="துளசிராம்"),
            ]
        }
        _merge_wife_husband_owners(result, raw_text)
        assert len(result["owner_names"]) == 1

    def test_single_owner_no_change(self):
        """Single owner → no change."""
        result = {
            "owner_names": [_make_owner("முருகன்")]
        }
        _merge_wife_husband_owners(result, "முருகன் S/o கந்தன்")
        assert len(result["owner_names"]) == 1

    def test_no_manaivi_pattern_no_change(self):
        """No மனைவி pattern in text, mixed owners → no change."""
        raw_text = "முருகன் மகன் கந்தன்\n"
        result = {
            "owner_names": [
                _make_owner("முருகன்"),
                _make_owner("கந்தன்"),
            ]
        }
        _merge_wife_husband_owners(result, raw_text)
        assert len(result["owner_names"]) == 2


# ═══════════════════════════════════════════════════
# _merge_wife_husband_owners (fallback inference)
# ═══════════════════════════════════════════════════

class TestMergeWifeHusbandInference:
    """Test inference-based merging when no raw text is available."""

    def test_infer_from_father_name_field(self):
        """Owner B has father_name matching Owner A's name → merge."""
        result = {
            "owner_names": [
                _make_owner("துளசிராம்"),
                _make_owner("சத்தயபாமா", father_name="துளசிராம்"),
            ]
        }
        _merge_wife_husband_owners(result, "")
        assert len(result["owner_names"]) == 1
        assert result["owner_names"][0]["name"] == "சத்தயபாமா"

    def test_no_inference_when_no_father_match(self):
        """Two owners without father_name relationship → no merge."""
        result = {
            "owner_names": [
                _make_owner("முருகன்"),
                _make_owner("கந்தன்"),
            ]
        }
        _merge_wife_husband_owners(result, "")
        assert len(result["owner_names"]) == 2


# ═══════════════════════════════════════════════════
# _post_process_patta integration
# ═══════════════════════════════════════════════════

class TestPostProcessPattaIntegration:
    """Integration: _post_process_patta applies wife-husband merging."""

    def test_full_pipeline_merges_owners(self):
        raw_text = "துளசிராம் மனைவி சத்தயபாமா\nராதாகிருஷ்ணன் மனைவி ராணி\n"
        result = {
            "owner_names": [
                _make_owner("துளசிராம்"),
                _make_owner("சத்தயபாமா"),
                _make_owner("ராதாகிருஷ்ணன்"),
                _make_owner("ராணி"),
            ],
            "survey_numbers": [],
        }
        processed = _post_process_patta(result, raw_text=raw_text)
        assert len(processed["owner_names"]) == 2
        names = {o["name"] for o in processed["owner_names"]}
        assert names == {"சத்தயபாமா", "ராணி"}

    def test_non_dict_passthrough(self):
        assert _post_process_patta("not a dict") == "not a dict"
        assert _post_process_patta(None) is None
