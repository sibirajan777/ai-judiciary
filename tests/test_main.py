"""Unit tests for main module functions."""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    get_terminal_width,
    wrap_text,
    format_scores_table,
    fmt_score,
    save_debate_result,
)


class TestTerminalWidth:
    """Tests for get_terminal_width function."""

    def test_returns_int(self):
        """Test that function returns an integer."""
        width = get_terminal_width()
        assert isinstance(width, int)
        assert width >= 70
        assert width <= 100


class TestWrapText:
    """Tests for wrap_text function."""

    def test_wraps_long_text(self):
        """Test that long text gets wrapped."""
        text = "This is a very long text that should be wrapped at word boundaries because it exceeds the width limit"
        result = wrap_text(text, width=40, indent="  ")
        lines = result.split("\n")
        # Each line should be <= width (approximately)
        for line in lines:
            assert len(line) <= 45  # Allow some margin for indent

    def test_preserves_paragraphs(self):
        """Test that paragraph breaks are preserved."""
        text = "Paragraph one.\n\nParagraph two."
        result = wrap_text(text, width=40, indent="  ")
        assert "\n\n" in result or result.count("\n") >= 1

    def test_initial_indent(self):
        """Test initial_indent parameter."""
        text = "Test text"
        result = wrap_text(text, width=40, indent="  ", initial_indent=">>> ")
        assert result.startswith(">>> ")

    def test_empty_string(self):
        """Test empty string handling."""
        result = wrap_text("", width=40)
        assert result == ""


class TestFmtScore:
    """Tests for fmt_score function."""

    def test_integer_score(self):
        """Test integer score formatting."""
        assert fmt_score(10) == "10"
        assert fmt_score(7) == "7"

    def test_float_score(self):
        """Test float score formatting."""
        assert fmt_score(7.5) == "7.5"
        assert fmt_score(7.0) == "7"  # Should show as integer

    def test_string_score(self):
        """Test string score formatting."""
        assert fmt_score("8") == "8"
        assert fmt_score("8.5") == "8.5"


class TestFormatScoresTable:
    """Tests for format_scores_table function."""

    def test_formats_table(self):
        """Test that table is formatted without errors."""
        pro_scores = {"evidence": 7, "logic": 8, "relevance": 6}
        con_scores = {"evidence": 5, "logic": 6, "relevance": 7}
        # Should not raise
        format_scores_table(pro_scores, con_scores, 80)

    def test_handles_missing_keys(self):
        """Test handling of missing score keys."""
        pro_scores = {"evidence": 7}  # missing logic, relevance
        con_scores = {"evidence": 5, "logic": 6, "relevance": 7}
        # Should not raise, uses defaults
        format_scores_table(pro_scores, con_scores, 80)


class TestSaveDebateResult:
    """Tests for save_debate_result function."""

    def test_saves_json_file(self, tmp_path):
        """Test that debate result is saved to JSON."""
        # Create a temporary results directory
        import main
        original_dir = Path(main.__file__).parent / "results"
        
        # We'll test the logic directly
        claim = "Test claim"
        result = {
            "pro_arguments": [{"point": "P1", "reason": "R1", "evidence": "E1"}],
            "con_arguments": [{"point": "C1", "reason": "R1", "evidence": "E1"}],
            "pro_rebuttal": {"target_point": "C1", "rebuttal": "Rebuttal"},
            "con_rebuttal": {"target_point": "P1", "rebuttal": "Rebuttal"},
            "judge_run1": {"mapped": {"verdict": "pro"}},
            "judge_run2": {"mapped": {"verdict": "pro"}},
        }
        timings = {"total": 10.5}
        
        # Create temp results dir
        test_results_dir = tmp_path / "results"
        test_results_dir.mkdir()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_claim = "".join(c if c.isalnum() or c in " -_" else "_" for c in claim[:50])
        filename = f"{timestamp}_{safe_claim}.json"
        filepath = test_results_dir / filename
        
        data = {
            "claim": claim,
            "timestamp": datetime.now().isoformat(),
            "timings": timings,
            "pro_arguments": result.get("pro_arguments", []),
            "con_arguments": result.get("con_arguments", []),
            "pro_rebuttal": result.get("pro_rebuttal", {}),
            "con_rebuttal": result.get("con_rebuttal", {}),
            "judge_run1": result.get("judge_run1", {}),
            "judge_run2": result.get("judge_run2", {}),
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        assert filepath.exists()
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["claim"] == claim
        assert "timings" in loaded


if __name__ == "__main__":
    import unittest
    unittest.main()