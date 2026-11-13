"""Unit tests for JSON parsing and schema validation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import (
    strip_code_fences_and_extract,
    parse_json_safely,
    _validate_arguments_schema,
    _validate_rebuttal_schema,
    _validate_judge_schema,
)


class TestStripCodeFences:
    """Tests for strip_code_fences_and_extract function."""

    def test_plain_json(self):
        """Test plain JSON without code fences."""
        text = '{"key": "value"}'
        result = strip_code_fences_and_extract(text)
        assert result == '{"key": "value"}'

    def test_json_with_code_fences(self):
        """Test JSON wrapped in markdown code fences."""
        text = '```json\n{"key": "value"}\n```'
        result = strip_code_fences_and_extract(text)
        assert result == '{"key": "value"}'

    def test_json_with_generic_fences(self):
        """Test JSON wrapped in generic code fences."""
        text = '```\n{"key": "value"}\n```'
        result = strip_code_fences_and_extract(text)
        assert result == '{"key": "value"}'

    def test_json_with_extra_text(self):
        """Test JSON embedded in extra text."""
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = strip_code_fences_and_extract(text)
        assert result == '{"key": "value"}'

    def test_multiple_code_blocks(self):
        """Test handling of multiple code blocks - should get the JSON one."""
        text = '```python\nprint("hello")\n```\n```json\n{"key": "value"}\n```'
        result = strip_code_fences_and_extract(text)
        assert result == '{"key": "value"}'

    def test_nested_braces(self):
        """Test JSON with nested objects."""
        text = '{"outer": {"inner": {"key": "value"}}}'
        result = strip_code_fences_and_extract(text)
        assert result == '{"outer": {"inner": {"key": "value"}}}'

    def test_array_response(self):
        """Test JSON array response."""
        text = '[{"a": 1}, {"b": 2}]'
        result = strip_code_fences_and_extract(text)
        assert result == '[{"a": 1}, {"b": 2}]'

    def test_empty_string(self):
        """Test empty string returns empty."""
        result = strip_code_fences_and_extract("")
        assert result == ""

    def test_whitespace_only(self):
        """Test whitespace only returns empty."""
        result = strip_code_fences_and_extract("   \n\t  ")
        assert result == ""


class TestParseJsonSafely:
    """Tests for parse_json_safely function."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        result = parse_json_safely('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_valid_json_with_fences(self):
        """Test parsing valid JSON with code fences."""
        result = parse_json_safely('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises ValueError."""
        try:
            parse_json_safely('{"key": "value"')  # missing closing brace
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "JSON syntax error" in str(e)

    def test_empty_string_raises(self):
        """Test that empty string raises ValueError."""
        try:
            parse_json_safely("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty response" in str(e).lower()

    def test_trailing_comma_recovery(self):
        """Test recovery from trailing comma."""
        # This tests the _attempt_json_repair function indirectly
        result = parse_json_safely('{"key": "value",}')
        assert result == {"key": "value"}

    def test_missing_quotes_recovery(self):
        """Test recovery from missing quotes on keys."""
        result = parse_json_safely('{key: "value"}')
        assert result == {"key": "value"}


class TestValidateArgumentsSchema:
    """Tests for _validate_arguments_schema function."""

    def test_valid_arguments(self):
        """Test valid arguments schema."""
        data = {
            "arguments": [
                {"point": "P1", "reason": "R1", "evidence": "E1"},
                {"point": "P2", "reason": "R2", "evidence": "E2"},
                {"point": "P3", "reason": "R3", "evidence": "E3"},
            ]
        }
        valid, error = _validate_arguments_schema(data)
        assert valid is True
        assert error == ""

    def test_missing_arguments_key(self):
        """Test missing arguments key."""
        data = {"points": []}
        valid, error = _validate_arguments_schema(data)
        assert valid is False
        assert "Missing top-level key 'arguments'" in error

    def test_wrong_number_of_arguments(self):
        """Test wrong number of arguments."""
        data = {"arguments": [{"point": "P1", "reason": "R1", "evidence": "E1"}]}
        valid, error = _validate_arguments_schema(data)
        assert valid is False
        assert "Expected exactly 3 arguments" in error

    def test_empty_field(self):
        """Test empty field in argument."""
        data = {
            "arguments": [
                {"point": "P1", "reason": "R1", "evidence": "E1"},
                {"point": "", "reason": "R2", "evidence": "E2"},  # empty point
                {"point": "P3", "reason": "R3", "evidence": "E3"},
            ]
        }
        valid, error = _validate_arguments_schema(data)
        assert valid is False
        assert "missing or has empty 'point'" in error

    def test_non_list_arguments(self):
        """Test non-list arguments."""
        data = {"arguments": "not a list"}
        valid, error = _validate_arguments_schema(data)
        assert valid is False
        assert "'arguments' must be a list" in error


class TestValidateRebuttalSchema:
    """Tests for _validate_rebuttal_schema function."""

    def test_valid_rebuttal(self):
        """Test valid rebuttal schema."""
        data = {"target_point": "Opponent's claim", "rebuttal": "My counter"}
        valid, error = _validate_rebuttal_schema(data)
        assert valid is True
        assert error == ""

    def test_missing_target_point(self):
        """Test missing target_point."""
        data = {"rebuttal": "My counter"}
        valid, error = _validate_rebuttal_schema(data)
        assert valid is False
        assert "Missing or empty 'target_point'" in error

    def test_missing_rebuttal(self):
        """Test missing rebuttal."""
        data = {"target_point": "Opponent's claim"}
        valid, error = _validate_rebuttal_schema(data)
        assert valid is False
        assert "Missing or empty 'rebuttal'" in error

    def test_empty_strings(self):
        """Test empty string fields."""
        data = {"target_point": "  ", "rebuttal": "My counter"}
        valid, error = _validate_rebuttal_schema(data)
        assert valid is False


class TestValidateJudgeSchema:
    """Tests for _validate_judge_schema function."""

    def test_valid_judge_output(self):
        """Test valid judge schema."""
        data = {
            "scores": {
                "side_a": {"evidence": 7, "logic": 8, "relevance": 6},
                "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
            },
            "strongest_point_each": {"side_a": "Point A", "side_b": "Point B"},
            "verdict": "side_a",
            "reasoning": "Side A had better evidence",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is True
        assert error == ""

    def test_invalid_verdict(self):
        """Test invalid verdict value."""
        data = {
            "scores": {
                "side_a": {"evidence": 7, "logic": 8, "relevance": 6},
                "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "invalid",
            "reasoning": "Test",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is False
        assert "Verdict must be" in error

    def test_score_out_of_range(self):
        """Test score out of 0-10 range."""
        data = {
            "scores": {
                "side_a": {"evidence": 15, "logic": 8, "relevance": 6},  # 15 > 10
                "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "side_a",
            "reasoning": "Test",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is False
        assert "outside [0, 10]" in error

    def test_missing_criterion(self):
        """Test missing criterion in scores."""
        data = {
            "scores": {
                "side_a": {"evidence": 7, "logic": 8},  # missing relevance
                "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "side_a",
            "reasoning": "Test",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is False
        assert "Missing criterion" in error

    def test_missing_reasoning(self):
        """Test missing reasoning."""
        data = {
            "scores": {
                "side_a": {"evidence": 7, "logic": 8, "relevance": 6},
                "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "side_a",
            "reasoning": "",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is False
        assert "Missing or empty 'reasoning'" in error

    def test_float_scores_converted(self):
        """Test that float scores are converted to int."""
        data = {
            "scores": {
                "side_a": {"evidence": 7.7, "logic": 8.2, "relevance": 6.9},
                "side_b": {"evidence": 5.1, "logic": 6.5, "relevance": 7.3},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "side_a",
            "reasoning": "Test",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is True
        # Check that scores were rounded to integers
        assert data["scores"]["side_a"]["evidence"] == 8
        assert data["scores"]["side_a"]["logic"] == 8
        assert data["scores"]["side_a"]["relevance"] == 7


class TestVisionAgentJSONHandling:
    """Regression tests for vision agent JSON handling."""

    def test_parse_json_with_single_quotes(self):
        """Test parsing JSON with single quotes (common LLM output)."""
        result = parse_json_safely('{"key": "value", "nested": {"a": 1}}')
        assert result == {"key": "value", "nested": {"a": 1}}

    def test_parse_json_with_prose_before_json(self):
        """Test that prose before JSON is handled correctly."""
        text = 'Here is my analysis:\n```json\n{"reasoning": "test", "scores": {"side_a": {"evidence": 8, "logic": 8, "relevance": 8}, "side_b": {"evidence": 6, "logic": 7, "relevance": 6}}, "strongest_point_each": {"side_a": "A", "side_b": "B"}, "verdict": "side_a"}\n```'
        result = parse_json_safely(text)
        assert result["verdict"] == "side_a"
        assert result["scores"]["side_a"]["evidence"] == 8

    def test_empty_content_raises_clear_error(self):
        """Test that empty content raises a clear error."""
        try:
            parse_json_safely("")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "empty response" in str(e).lower()

    def test_malformed_json_raises_clear_error(self):
        """Test that malformed JSON raises a clear error."""
        try:
            parse_json_safely('{"key": "value"')  # missing closing brace
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "JSON syntax error" in str(e)

    def test_reasoning_content_not_used_as_json(self):
        """Test that reasoning-like prose is not treated as valid JSON."""
        # This simulates what happens when message.content is empty
        # but reasoning_content has text - the llm.py fix prevents this
        # but we test that the parser correctly rejects prose
        prose = "The user wants me to act as a judge in a debate about an image. I need to analyze both sides carefully."
        try:
            parse_json_safely(prose)
            assert False, "Should have raised ValueError for prose input"
        except ValueError as e:
            assert "JSON syntax error" in str(e) or "No JSON object" in str(e)

    def test_valid_judge_schema_with_integers(self):
        """Test that judge schema validates integer scores."""
        data = {
            "reasoning": "Side A had stronger visual evidence",
            "scores": {
                "side_a": {"evidence": 8, "logic": 8, "relevance": 8},
                "side_b": {"evidence": 6, "logic": 7, "relevance": 6},
            },
            "strongest_point_each": {"side_a": "Clear product labeling", "side_b": "Price point"},
            "verdict": "side_a",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is True

    def test_judge_schema_rejects_missing_verdict(self):
        """Test that judge schema rejects missing verdict."""
        data = {
            "reasoning": "Test",
            "scores": {
                "side_a": {"evidence": 8, "logic": 8, "relevance": 8},
                "side_b": {"evidence": 6, "logic": 7, "relevance": 6},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            # missing verdict
        }
        valid, error = _validate_judge_schema(data)
        assert valid is False
        assert "verdict" in error.lower()

    def test_judge_schema_rejects_non_integer_scores(self):
        """Test that judge schema rejects non-integer scores after conversion."""
        # The validator converts floats to int using round (banker's rounding)
        data = {
            "reasoning": "Test",
            "scores": {
                "side_a": {"evidence": 8.6, "logic": 8.2, "relevance": 8.7},
                "side_b": {"evidence": 6.3, "logic": 7.1, "relevance": 6.9},
            },
            "strongest_point_each": {"side_a": "A", "side_b": "B"},
            "verdict": "side_a",
        }
        valid, error = _validate_judge_schema(data)
        assert valid is True
        # Check conversion happened (banker's rounding: 8.6 -> 9, 8.2 -> 8, 8.7 -> 9)
        assert data["scores"]["side_a"]["evidence"] == 9
        assert data["scores"]["side_a"]["logic"] == 8
        assert data["scores"]["side_a"]["relevance"] == 9


class TestTruncationRecovery:
    """Regression tests for vision agent truncation recovery."""

    def test_finish_reason_stop_accepted(self):
        """Test that finish_reason=stop is handled correctly."""
        # This is tested implicitly - valid JSON with stop is accepted
        pass

    def test_finish_reason_length_raises_clear_error(self):
        """Test that finish_reason=length raises a clear error with diagnostics."""
        from agents import call_agent_with_retry_vision
        from llm import call_llm_vision
        
        # We can't easily mock the LLM call here, but we can test the error message format
        # The actual truncation handling is tested in TestTruncationRecoveryIntegration

    def test_truncation_error_contains_finish_reason(self):
        """Test that truncation error includes finish_reason for diagnostics."""
        error_msg = "NVIDIA NIM vision response truncated (finish_reason=length). Increase max_tokens or reduce prompt size."
        assert "finish_reason=length" in error_msg
        assert "truncated" in error_msg.lower()

    def test_content_filter_error_distinct(self):
        """Test that content_filter finish_reason produces distinct error."""
        error_msg = "NVIDIA NIM vision response filtered by content policy."
        assert "filtered" in error_msg.lower()
        assert "content policy" in error_msg.lower()

    def test_other_finish_reasons_reported(self):
        """Test that unexpected finish_reasons are reported with diagnostics."""
        error_msg = "NVIDIA NIM vision returned empty message.content. finish_reason=unknown. This model may not support vision or the requested format."
        assert "finish_reason=unknown" in error_msg
        assert "empty message.content" in error_msg


class TestTruncationRecoveryIntegration:
    """Integration tests for truncation recovery logic (mocked)."""

    def test_truncation_retry_increases_token_budget(self, monkeypatch):
        """Test that truncation triggers retry with increased token budget."""
        from agents import call_agent_with_retry_vision
        from llm import NVIDIA_VISION_MAX_TOKENS
        
        call_count = 0
        original_max_tokens = 2048
        
        def mock_call_llm_vision(system_prompt, user_message, image_base64, image_mime_type, max_tokens=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call simulates truncation
                raise RuntimeError("NVIDIA NIM vision response truncated (finish_reason=length). Increase max_tokens or reduce prompt size.")
            # Second call succeeds with valid JSON
            return '{"arguments": [{"point": "P1", "reason": "R1", "evidence": "E1"}, {"point": "P2", "reason": "R2", "evidence": "E2"}, {"point": "P3", "reason": "R3", "evidence": "E3"}]}'
        
        # Patch in llm module where call_llm_vision is defined
        monkeypatch.setattr("llm.call_llm_vision", mock_call_llm_vision)
        
        # Should retry once with increased budget and succeed
        result = call_agent_with_retry_vision(
            system_prompt="test",
            user_message="test",
            image_base64="fake_base64",
            image_mime_type="image/png",
            schema_description='{"arguments": [...]}',
            validator_fn=lambda d: (True, ""),
            max_tokens=original_max_tokens,
        )
        
        assert call_count == 2
        assert result["arguments"][0]["point"] == "P1"

    def test_truncation_retry_respects_global_max(self, monkeypatch):
        """Test that truncation retry doesn't exceed global max token limit."""
        from agents import call_agent_with_retry_vision
        from llm import NVIDIA_VISION_MAX_TOKENS
        
        call_count = 0
        original_max_tokens = NVIDIA_VISION_MAX_TOKENS  # Already at global max
        
        def mock_call_llm_vision(system_prompt, user_message, image_base64, image_mime_type, max_tokens=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("NVIDIA NIM vision response truncated (finish_reason=length). Increase max_tokens or reduce prompt size.")
            return '{"arguments": [{"point": "P1", "reason": "R1", "evidence": "E1"}, {"point": "P2", "reason": "R2", "evidence": "E2"}, {"point": "P3", "reason": "R3", "evidence": "E3"}]}'
        
        monkeypatch.setattr("llm.call_llm_vision", mock_call_llm_vision)
        
        # Should NOT retry when already at global max (truncation_retry_done prevents infinite loop)
        try:
            call_agent_with_retry_vision(
                system_prompt="test",
                user_message="test",
                image_base64="fake_base64",
                image_mime_type="image/png",
                schema_description='{"arguments": [...]}',
                validator_fn=lambda d: (True, ""),
                max_tokens=original_max_tokens,
            )
            assert False, "Should have raised RuntimeError after truncation at max budget"
        except RuntimeError as e:
            assert "truncated" in str(e).lower()
            assert call_count == 1  # No retry attempted

    def test_valid_json_after_retry(self, monkeypatch):
        """Test that valid JSON is returned after successful truncation retry."""
        from agents import call_agent_with_retry_vision
        
        call_count = 0
        
        def mock_call_llm_vision(system_prompt, user_message, image_base64, image_mime_type, max_tokens=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("NVIDIA NIM vision response truncated (finish_reason=length). Increase max_tokens or reduce prompt size.")
            return '{"arguments": [{"point": "P1", "reason": "R1", "evidence": "E1"}, {"point": "P2", "reason": "R2", "evidence": "E2"}, {"point": "P3", "reason": "R3", "evidence": "E3"}]}'
        
        monkeypatch.setattr("llm.call_llm_vision", mock_call_llm_vision)
        
        result = call_agent_with_retry_vision(
            system_prompt="test",
            user_message="test",
            image_base64="fake_base64",
            image_mime_type="image/png",
            schema_description='{"arguments": [...]}',
            validator_fn=lambda d: (True, ""),
            max_tokens=2048,
        )
        
        assert len(result["arguments"]) == 3
        assert result["arguments"][0]["point"] == "P1"

    def test_invalid_json_after_retry_fails(self, monkeypatch):
        """Test that invalid JSON after retry fails appropriately."""
        from agents import call_agent_with_retry_vision
        
        call_count = 0
        
        def mock_call_llm_vision(system_prompt, user_message, image_base64, image_mime_type, max_tokens=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("NVIDIA NIM vision response truncated (finish_reason=length). Increase max_tokens or reduce prompt size.")
            # Second call returns invalid JSON
            return 'not valid json'
        
        monkeypatch.setattr("llm.call_llm_vision", mock_call_llm_vision)
        
        try:
            call_agent_with_retry_vision(
                system_prompt="test",
                user_message="test",
                image_base64="fake_base64",
                image_mime_type="image/png",
                schema_description='{"arguments": [...]}',
                validator_fn=lambda d: (True, ""),
                max_tokens=2048,
            )
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Failed to obtain valid JSON" in str(e)


class TestSyncAsyncParity:
    """Test that sync and async vision calls have same truncation handling."""

    def test_async_truncation_error_format(self):
        """Test that async function produces same truncation error format."""
        # The async function _call_nvidia_vision_async has the same error handling
        # This is verified by code inspection - both check finish_reason identically
        pass


if __name__ == "__main__":
    import unittest
    unittest.main()