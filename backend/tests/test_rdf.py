"""Tests for radial distribution function analysis."""

import pytest
import torch
import numpy as np

from backend.ml.analysis.rdf import (
    compute_rdf,
    compute_rdf_from_system,
    rdf_statistics,
    compare_rdfs,
)
from backend.ml.systems.lennard_jones import LennardJonesSystem


class TestComputeRDF:
    """Tests for RDF computation."""

    def test_rdf_shape(self):
        """RDF should return correct shapes."""
        positions = torch.rand(10, 16, 2) * 5.0
        r, g_r = compute_rdf(positions, box_size=5.0, n_bins=50)

        assert len(r) == 50
        assert len(g_r) == 50

    def test_rdf_single_sample(self):
        """RDF should work with single sample (no batch dim)."""
        positions = torch.rand(16, 2) * 5.0
        r, g_r = compute_rdf(positions, box_size=5.0, n_bins=30)

        assert len(r) == 30
        assert np.all(np.isfinite(g_r))

    def test_rdf_positive(self):
        """RDF values should be non-negative."""
        # Generate many random samples
        positions = torch.rand(100, 25, 2) * 10.0
        r, g_r = compute_rdf(positions, box_size=10.0, n_bins=50)

        # g(r) should be non-negative
        assert np.all(g_r >= 0)
        # Should have some non-zero values
        assert np.any(g_r > 0)

    def test_rdf_max_distance(self):
        """r_max should limit maximum distance."""
        positions = torch.rand(10, 16, 2) * 5.0
        r, g_r = compute_rdf(positions, box_size=5.0, n_bins=50, r_max=2.0)

        assert r.max() <= 2.0

    def test_rdf_with_system(self):
        """compute_rdf_from_system should work correctly."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration(batch_size=5)

        r, g_r = compute_rdf_from_system(positions, system, n_bins=30)

        assert len(r) == 30
        assert np.all(np.isfinite(g_r))


class TestRDFStatistics:
    """Tests for RDF statistics extraction."""

    def test_statistics_keys(self):
        """Statistics should contain expected keys."""
        r = np.linspace(0.1, 2.5, 50)
        # Create a fake RDF with a peak
        g_r = np.exp(-((r - 1.2) ** 2) / 0.1) + 0.5

        stats = rdf_statistics(r, g_r, sigma=1.0)

        assert "first_peak_r" in stats
        assert "first_peak_height" in stats
        assert "coordination_number" in stats

    def test_statistics_peak_detection(self):
        """Should correctly identify peak position."""
        r = np.linspace(0.5, 3.0, 100)
        # Create RDF with peak at r = 1.5
        g_r = 2.0 * np.exp(-((r - 1.5) ** 2) / 0.1) + 0.2

        stats = rdf_statistics(r, g_r, sigma=1.0)

        # Peak should be near 1.5
        assert 1.4 < stats["first_peak_r"] < 1.6
        # Peak height should be close to 2.2
        assert 1.8 < stats["first_peak_height"] < 2.5

    def test_statistics_empty_region(self):
        """Should handle RDF with g(r)=0 at small r."""
        r = np.linspace(0.1, 3.0, 100)
        g_r = np.zeros_like(r)
        g_r[r > 1.0] = np.exp(-((r[r > 1.0] - 1.5) ** 2) / 0.1)

        stats = rdf_statistics(r, g_r, sigma=1.0)

        # Should still find a peak
        assert stats["first_peak_r"] > 1.0


class TestCompareRDFs:
    """Tests for RDF comparison."""

    def test_compare_identical(self):
        """Identical RDFs should have zero difference."""
        r = np.linspace(0.5, 2.5, 50)
        g_r = np.exp(-((r - 1.2) ** 2) / 0.1) + 0.5

        comparison = compare_rdfs(r, g_r, r, g_r)

        assert comparison["mse"] < 1e-10
        assert comparison["max_diff"] < 1e-10
        assert comparison["peak_diff"] < 1e-10

    def test_compare_different(self):
        """Different RDFs should have non-zero difference."""
        r = np.linspace(0.5, 2.5, 50)
        g_r1 = np.exp(-((r - 1.2) ** 2) / 0.1) + 0.5
        g_r2 = np.exp(-((r - 1.5) ** 2) / 0.1) + 0.3

        comparison = compare_rdfs(r, g_r1, r, g_r2)

        assert comparison["mse"] > 0
        assert comparison["max_diff"] > 0
        assert comparison["peak_diff"] > 0

    def test_compare_interpolation(self):
        """Should interpolate when grids differ."""
        r1 = np.linspace(0.5, 2.5, 50)
        r2 = np.linspace(0.5, 2.5, 100)
        g_r1 = np.exp(-((r1 - 1.2) ** 2) / 0.1)
        g_r2 = np.exp(-((r2 - 1.2) ** 2) / 0.1)

        comparison = compare_rdfs(r1, g_r1, r2, g_r2)

        # Should be small since underlying functions are the same
        assert comparison["mse"] < 0.01


class TestRDFIntegration:
    """Integration tests with actual L-J samples."""

    @pytest.mark.slow
    def test_rdf_from_langevin_samples(self):
        """RDF from Langevin samples should show liquid structure."""
        from backend.ml.samplers.langevin import LangevinDynamics

        system = LennardJonesSystem(n_particles=25, box_size=6.0)
        sampler = LangevinDynamics(system, temperature=1.0, dt=0.001)

        # Generate samples
        samples = sampler.sample(n_samples=20, n_steps=50, burn_in=200)

        # Compute RDF
        r, g_r = compute_rdf_from_system(samples, system, n_bins=50)

        # Check basic properties
        assert len(r) == 50
        assert np.all(np.isfinite(g_r))

        # For liquid, g(r) should have a peak around r ≈ 1.1-1.3σ
        stats = rdf_statistics(r, g_r, sigma=system.sigma)
        assert 0.9 < stats["first_peak_r"] < 2.0


@pytest.fixture
def sample_rdf():
    """Create a sample RDF for testing."""
    r = np.linspace(0.5, 2.5, 50)
    g_r = np.exp(-((r - 1.2) ** 2) / 0.1) + 0.5
    return r, g_r
