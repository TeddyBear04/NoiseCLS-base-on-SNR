"""
Pytest fixtures and configuration for testing single-backbone CED MoE architecture.
"""

import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_waveform():
    """Batch of 2 synthetic audio clips at 16kHz for 4.0 seconds (64,000 samples)."""
    torch.manual_seed(42)
    return torch.randn(2, 64000)
