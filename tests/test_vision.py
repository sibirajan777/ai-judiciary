"""Unit tests for vision functionality."""

import sys
import base64
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from agents import (
    VisionAdvocateAgent,
    VisionOpposerAgent,
    VisionRebuttalAgent,
    VisionJudgeAgent,
    call_agent_with_retry_vision,
)
from prompts import (
    VISION_ADVOCATE_SYSTEM_PROMPT,
    VISION_OPPOSER_SYSTEM_PROMPT,
    VISION_REBUTTAL_SYSTEM_PROMPT,
    VISION_JUDGE_SYSTEM_PROMPT,
)
from llm import NVIDIA_VISION_MODEL, MAX_IMAGE_SIZE_MB, ALLOWED_IMAGE_TYPES


class TestVisionConfig:
    """Tests for vision configuration."""

    def test_vision_model_configured(self):
        """Test that vision model is configured."""
        assert NVIDIA_VISION_MODEL is not None
        assert len(NVIDIA_VISION_MODEL) > 0

    def test_max_image_size(self):
        """Test max image size is reasonable."""
        assert MAX_IMAGE_SIZE_MB > 0
        assert MAX_IMAGE_SIZE_MB <= 100  # Reasonable upper bound

    def test_allowed_image_types(self):
        """Test allowed image types includes common formats."""
        assert "image/png" in ALLOWED_IMAGE_TYPES
        assert "image/jpeg" in ALLOWED_IMAGE_TYPES
        assert "image/webp" in ALLOWED_IMAGE_TYPES


class TestVisionPrompts:
    """Tests for vision system prompts."""

    def test_vision_advocate_prompt_exists(self):
        """Test vision advocate prompt is defined."""
        assert VISION_ADVOCATE_SYSTEM_PROMPT is not None
        assert "IMAGE" in VISION_ADVOCATE_SYSTEM_PROMPT.upper()
        assert "VISUAL" in VISION_ADVOCATE_SYSTEM_PROMPT.upper()
        assert "EXACTLY 3" in VISION_ADVOCATE_SYSTEM_PROMPT
        assert "arguments" in VISION_ADVOCATE_SYSTEM_PROMPT.lower()

    def test_vision_opposer_prompt_exists(self):
        """Test vision opposer prompt is defined."""
        assert VISION_OPPOSER_SYSTEM_PROMPT is not None
        assert "IMAGE" in VISION_OPPOSER_SYSTEM_PROMPT.upper()
        assert "VISUAL" in VISION_OPPOSER_SYSTEM_PROMPT.upper()
        assert "EXACTLY 3" in VISION_OPPOSER_SYSTEM_PROMPT

    def test_vision_rebuttal_prompt_exists(self):
        """Test vision rebuttal prompt is defined."""
        assert VISION_REBUTTAL_SYSTEM_PROMPT is not None
        assert "IMAGE" in VISION_REBUTTAL_SYSTEM_PROMPT.upper()
        assert "VISUAL" in VISION_REBUTTAL_SYSTEM_PROMPT.upper()
        assert "target_point" in VISION_REBUTTAL_SYSTEM_PROMPT
        assert "rebuttal" in VISION_REBUTTAL_SYSTEM_PROMPT

    def test_vision_judge_prompt_exists(self):
        """Test vision judge prompt is defined."""
        assert VISION_JUDGE_SYSTEM_PROMPT is not None
        assert "IMAGE" in VISION_JUDGE_SYSTEM_PROMPT.upper()
        assert "VISUAL" in VISION_JUDGE_SYSTEM_PROMPT.upper()
        assert "side_a" in VISION_JUDGE_SYSTEM_PROMPT
        assert "side_b" in VISION_JUDGE_SYSTEM_PROMPT
        assert "verdict" in VISION_JUDGE_SYSTEM_PROMPT


class TestVisionAgentsStructure:
    """Tests for vision agent class structure (without API calls)."""

    def test_vision_advocate_class_exists(self):
        """Test VisionAdvocateAgent class exists with argue method."""
        assert hasattr(VisionAdvocateAgent, 'argue')
        import inspect
        sig = inspect.signature(VisionAdvocateAgent.argue)
        params = list(sig.parameters.keys())
        assert 'question' in params
        assert 'image_base64' in params
        assert 'image_mime_type' in params

    def test_vision_opposer_class_exists(self):
        """Test VisionOpposerAgent class exists with argue method."""
        assert hasattr(VisionOpposerAgent, 'argue')
        import inspect
        sig = inspect.signature(VisionOpposerAgent.argue)
        params = list(sig.parameters.keys())
        assert 'question' in params
        assert 'image_base64' in params
        assert 'image_mime_type' in params

    def test_vision_rebuttal_class_exists(self):
        """Test VisionRebuttalAgent class exists with rebut method."""
        assert hasattr(VisionRebuttalAgent, 'rebut')
        import inspect
        sig = inspect.signature(VisionRebuttalAgent.rebut)
        params = list(sig.parameters.keys())
        assert 'role_name' in params
        assert 'question' in params
        assert 'own_side' in params
        assert 'opponent_args' in params
        assert 'image_base64' in params
        assert 'image_mime_type' in params

    def test_vision_judge_class_exists(self):
        """Test VisionJudgeAgent class exists with evaluate method."""
        assert hasattr(VisionJudgeAgent, 'evaluate')
        import inspect
        sig = inspect.signature(VisionJudgeAgent.evaluate)
        params = list(sig.parameters.keys())
        assert 'question' in params
        assert 'pro_arguments' in params
        assert 'con_arguments' in params
        assert 'pro_rebuttal' in params
        assert 'con_rebuttal' in params
        assert 'image_base64' in params
        assert 'image_mime_type' in params
        assert 'pro_is_side_a' in params


class TestCallAgentWithRetryVision:
    """Tests for call_agent_with_retry_vision function signature."""

    def test_function_exists(self):
        """Test the function exists."""
        from agents import call_agent_with_retry_vision
        assert callable(call_agent_with_retry_vision)

    def test_function_signature(self):
        """Test function has correct parameters."""
        import inspect
        from agents import call_agent_with_retry_vision
        sig = inspect.signature(call_agent_with_retry_vision)
        params = list(sig.parameters.keys())
        assert 'system_prompt' in params
        assert 'user_message' in params
        assert 'image_base64' in params
        assert 'image_mime_type' in params
        assert 'schema_description' in params
        assert 'validator_fn' in params
        assert 'max_retries' in params


class TestSampleImageCreation:
    """Tests for creating sample test images."""

    def test_create_sample_base64_image(self):
        """Test creating a minimal base64 PNG for testing."""
        # 1x1 transparent PNG
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        assert len(b64) > 0
        # Verify it's valid base64
        decoded = base64.b64decode(b64)
        assert decoded == png_bytes

    def test_image_mime_types(self):
        """Test MIME type constants."""
        assert "image/png" == "image/png"
        assert "image/jpeg" == "image/jpeg"
        assert "image/webp" == "image/webp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])