"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from backend.api.main import app
    return TestClient(app)


@pytest.fixture
def ising_model():
    """Create a small Ising model for testing."""
    from backend.ml.systems.ising import IsingModel
    return IsingModel(size=8, J=1.0, h=0.0)


@pytest.fixture
def mcmc_sampler(ising_model):
    """Create a Metropolis-Hastings sampler at critical temperature."""
    from backend.ml.samplers.mcmc import MetropolisHastings
    return MetropolisHastings(ising_model, temperature=2.27)


@pytest.fixture
def low_temp_sampler(ising_model):
    """Create a low-temperature sampler (ordered phase)."""
    from backend.ml.samplers.mcmc import MetropolisHastings
    return MetropolisHastings(ising_model, temperature=1.0)


@pytest.fixture
def high_temp_sampler(ising_model):
    """Create a high-temperature sampler (disordered phase)."""
    from backend.ml.samplers.mcmc import MetropolisHastings
    return MetropolisHastings(ising_model, temperature=5.0)


@pytest.fixture
def random_spins(ising_model):
    """Generate random spin configuration."""
    return ising_model.random_configuration(batch_size=1).squeeze(0)


@pytest.fixture
def all_up_spins():
    """Create all-up spin configuration."""
    return torch.ones(8, 8)


@pytest.fixture
def all_down_spins():
    """Create all-down spin configuration."""
    return -torch.ones(8, 8)


@pytest.fixture
def checkerboard_spins():
    """Create checkerboard spin configuration."""
    spins = torch.ones(8, 8)
    spins[::2, 1::2] = -1
    spins[1::2, ::2] = -1
    return spins


# Temperature fixtures for phase transition testing
@pytest.fixture
def critical_temperature():
    """Critical temperature for 2D Ising model."""
    return 2.269


@pytest.fixture
def subcritical_temperature():
    """Temperature below T_c (ordered phase)."""
    return 1.5


@pytest.fixture
def supercritical_temperature():
    """Temperature above T_c (disordered phase)."""
    return 4.0


# Sampling parameter fixtures
@pytest.fixture
def default_mcmc_params():
    """Default parameters for MCMC sampling API requests."""
    return {
        "temperature": 2.27,
        "lattice_size": 16,
        "n_samples": 5,
        "n_sweeps": 10,
        "burn_in": 50,
    }


@pytest.fixture
def fast_mcmc_params():
    """Fast parameters for quick testing."""
    return {
        "temperature": 2.27,
        "lattice_size": 8,
        "n_samples": 2,
        "n_sweeps": 5,
        "burn_in": 10,
    }
