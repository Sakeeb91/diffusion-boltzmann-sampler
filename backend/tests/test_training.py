"""Tests for training module including losses, trainer, and API endpoints."""

import pytest
import torch
import tempfile
import os
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.models.score_network import ScoreNetwork
from backend.ml.models.diffusion import DiffusionProcess
from backend.ml.training import (
    Trainer,
    ScoreMatchingLoss,
    denoising_score_matching_loss,
    sigma_weighted_loss,
    snr_weighted_loss,
    importance_sampled_loss,
    compute_loss,
    reduce_loss,
)
from backend.ml.training.trainer import (
    EarlyStopping,
    EMA,
    create_scheduler,
    compute_gradient_norm,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_score_network():
    """Create a small score network for fast testing."""
    return ScoreNetwork(
        in_channels=1,
        base_channels=8,
        time_embed_dim=16,
        num_blocks=1,
    )


@pytest.fixture
def diffusion():
    """Create a diffusion process."""
    return DiffusionProcess(beta_min=0.1, beta_max=20.0)


@pytest.fixture
def sample_batch():
    """Create sample batch data for testing."""
    return torch.randn(4, 1, 8, 8)


@pytest.fixture
def sample_dataloader(sample_batch):
    """Create a dataloader from sample batch."""
    dataset = TensorDataset(sample_batch)
    return DataLoader(dataset, batch_size=2, shuffle=True)


@pytest.fixture
def trainer(small_score_network, diffusion):
    """Create a trainer instance."""
    return Trainer(
        score_network=small_score_network,
        diffusion=diffusion,
        learning_rate=1e-3,
    )
