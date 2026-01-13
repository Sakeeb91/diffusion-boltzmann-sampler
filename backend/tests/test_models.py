"""Tests for diffusion model components."""

import pytest
import torch

from backend.ml.models import (
    ScoreNetwork,
    SinusoidalTimeEmbedding,
    ConvBlock,
    DiffusionProcess,
    LinearNoiseSchedule,
    CosineNoiseSchedule,
    SigmoidNoiseSchedule,
    get_schedule,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def score_network():
    """Create a small score network for testing."""
    return ScoreNetwork(
        in_channels=1,
        base_channels=16,
        time_embed_dim=32,
        num_blocks=2,
    )


@pytest.fixture
def diffusion():
    """Create a diffusion process for testing."""
    return DiffusionProcess(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def batch_data():
    """Create sample batch data."""
    batch_size = 4
    size = 16
    x = torch.randn(batch_size, 1, size, size)
    t = torch.rand(batch_size)
    return x, t


@pytest.fixture
def linear_schedule():
    """Create a linear noise schedule."""
    return LinearNoiseSchedule(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def cosine_schedule():
    """Create a cosine noise schedule."""
    return CosineNoiseSchedule(s=0.008)


@pytest.fixture
def sigmoid_schedule():
    """Create a sigmoid noise schedule."""
    return SigmoidNoiseSchedule(beta_min=0.1, beta_max=20.0)
