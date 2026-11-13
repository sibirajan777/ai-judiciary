"""Unit tests for LLM configuration and timeout handling."""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# We need to test the module-level configuration loading
# So we'll import after setting env vars


class TestLLMConfiguration:
    """Tests for LLM timeout configuration."""

    def test_default_timeouts(self):
        """Test default timeout values when env vars not set."""
        # Clear any existing env vars
        for key in ["LLM_CONNECT_TIMEOUT", "LLM_READ_TIMEOUT", "LLM_MAX_RETRIES", "LLM_RATE_LIMIT_BACKOFF"]:
            if key in os.environ:
                del os.environ[key]
        
        # Re-import to pick up defaults
        import importlib
        import llm
        importlib.reload(llm)
        
        from llm import CONNECT_TIMEOUT, READ_TIMEOUT, MAX_RETRIES, RATE_LIMIT_BASE_BACKOFF
        
        assert CONNECT_TIMEOUT == 15.0
        assert READ_TIMEOUT == 90.0
        assert MAX_RETRIES == 3
        assert RATE_LIMIT_BASE_BACKOFF == 10.0

    def test_custom_timeouts(self):
        """Test custom timeout values from env vars."""
        os.environ["LLM_CONNECT_TIMEOUT"] = "20.0"
        os.environ["LLM_READ_TIMEOUT"] = "120.0"
        os.environ["LLM_MAX_RETRIES"] = "5"
        os.environ["LLM_RATE_LIMIT_BACKOFF"] = "15.0"
        
        import importlib
        import llm
        importlib.reload(llm)
        
        from llm import CONNECT_TIMEOUT, READ_TIMEOUT, MAX_RETRIES, RATE_LIMIT_BASE_BACKOFF
        
        assert CONNECT_TIMEOUT == 20.0
        assert READ_TIMEOUT == 120.0
        assert MAX_RETRIES == 5
        assert RATE_LIMIT_BASE_BACKOFF == 15.0
        
        # Cleanup
        for key in ["LLM_CONNECT_TIMEOUT", "LLM_READ_TIMEOUT", "LLM_MAX_RETRIES", "LLM_RATE_LIMIT_BACKOFF"]:
            del os.environ[key]

    def test_invalid_timeout_fallbacks(self):
        """Test that invalid values fall back to defaults."""
        os.environ["LLM_CONNECT_TIMEOUT"] = "invalid"
        os.environ["LLM_MAX_RETRIES"] = "not_a_number"
        
        import importlib
        import llm
        importlib.reload(llm)
        
        from llm import CONNECT_TIMEOUT, MAX_RETRIES
        
        # Should fall back to defaults on ValueError
        assert CONNECT_TIMEOUT == 15.0
        assert MAX_RETRIES == 3
        
        # Cleanup
        for key in ["LLM_CONNECT_TIMEOUT", "LLM_MAX_RETRIES"]:
            del os.environ[key]


class TestSplitTimeoutHTTPSConnection:
    """Tests for SplitTimeoutHTTPSConnection class."""

    def test_initialization(self):
        """Test connection initialization with custom timeouts."""
        from llm import SplitTimeoutHTTPSConnection
        
        conn = SplitTimeoutHTTPSConnection("example.com", connect_timeout=10.0, read_timeout=60.0)
        assert conn.connect_timeout == 10.0
        assert conn.read_timeout == 60.0

    def test_default_timeouts(self):
        """Test connection uses module-level defaults."""
        from llm import SplitTimeoutHTTPSConnection, CONNECT_TIMEOUT, READ_TIMEOUT
        
        conn = SplitTimeoutHTTPSConnection("example.com")
        assert conn.connect_timeout == CONNECT_TIMEOUT
        assert conn.read_timeout == READ_TIMEOUT


if __name__ == "__main__":
    import unittest
    unittest.main()