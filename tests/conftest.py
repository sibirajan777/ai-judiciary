"""Shared test fixtures and configuration."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_claim():
    """Sample debate claim for testing."""
    return "Remote work is better than office work"


@pytest.fixture
def sample_arguments():
    """Sample argument structure for testing."""
    return [
        {"point": "Productivity", "reason": "Fewer distractions", "evidence": "Studies show 13% increase"},
        {"point": "Work-life balance", "reason": "Flexible schedule", "evidence": "Employee surveys"},
        {"point": "Cost savings", "reason": "Less office space", "evidence": "Real estate costs reduced"},
    ]


@pytest.fixture
def sample_rebuttal():
    """Sample rebuttal structure for testing."""
    return {"target_point": "Productivity", "rebuttal": "But collaboration suffers"}


@pytest.fixture
def sample_judge_output():
    """Sample judge output for testing."""
    return {
        "scores": {
            "side_a": {"evidence": 7, "logic": 8, "relevance": 6},
            "side_b": {"evidence": 5, "logic": 6, "relevance": 7},
        },
        "strongest_point_each": {"side_a": "Productivity", "side_b": "Collaboration"},
        "verdict": "side_a",
        "reasoning": "Side A had stronger evidence",
    }