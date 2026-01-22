"""Tests for Lennard-Jones system."""

import pytest
import torch
import numpy as np

from backend.ml.systems.lennard_jones import LennardJonesSystem


class TestLennardJonesEnergy:
    """Tests for energy computation."""

    def test_lattice_energy_finite(self):
        """Lattice configuration should have finite energy."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()
        energy = system.energy(positions)

        assert torch.isfinite(energy).all()
        assert energy.item() < 0  # Attractive at equilibrium spacing

    def test_energy_batch(self):
        """Energy should work with batch dimension."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration(batch_size=4)
        energies = system.energy(positions)

        assert energies.shape == (4,)
        assert torch.allclose(energies, energies[0].expand(4))

    def test_energy_increases_with_overlap(self):
        """Energy should increase when particles overlap."""
        system = LennardJonesSystem(n_particles=2, box_size=10.0)

        # Well-separated particles
        pos_far = torch.tensor([[[1.0, 1.0], [5.0, 5.0]]])
        E_far = system.energy(pos_far)

        # Close particles (but not overlapping core)
        pos_close = torch.tensor([[[1.0, 1.0], [2.2, 1.0]]])
        E_close = system.energy(pos_close)

        # Very close particles (repulsive regime)
        pos_overlap = torch.tensor([[[1.0, 1.0], [1.5, 1.0]]])
        E_overlap = system.energy(pos_overlap)

        assert E_close < E_far  # Attraction at moderate distance
        assert E_overlap > E_close  # Repulsion at short distance

    def test_energy_cutoff(self):
        """Particles beyond cutoff should not interact."""
        system = LennardJonesSystem(n_particles=2, box_size=10.0, cutoff=2.5)

        # Particles beyond cutoff (2.5σ = 2.5)
        pos = torch.tensor([[[0.0, 0.0], [3.0, 0.0]]])
        E = system.energy(pos)

        assert torch.abs(E) < 1e-6  # No interaction beyond cutoff


class TestLennardJonesForces:
    """Tests for force computation."""

    def test_force_shape(self):
        """Forces should have correct shape."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()
        forces = system.forces(positions)

        assert forces.shape == positions.shape

    def test_force_consistency_with_energy(self):
        """Forces should be consistent with numerical gradient of energy."""
        system = LennardJonesSystem(n_particles=4, box_size=5.0)
        positions = system.lattice_configuration().squeeze(0)

        # Analytical force
        forces = system.forces(positions.unsqueeze(0)).squeeze(0)

        # Numerical gradient
        eps = 1e-5
        numerical_forces = torch.zeros_like(forces)

        for i in range(system.n_particles):
            for d in range(2):
                pos_plus = positions.clone()
                pos_minus = positions.clone()
                pos_plus[i, d] += eps
                pos_minus[i, d] -= eps

                E_plus = system.energy(pos_plus.unsqueeze(0))
                E_minus = system.energy(pos_minus.unsqueeze(0))

                # F = -dE/dx
                numerical_forces[i, d] = -(E_plus - E_minus) / (2 * eps)

        # Check agreement
        assert torch.allclose(forces, numerical_forces, atol=1e-3, rtol=1e-3)

    def test_force_symmetry(self):
        """Force on particle i from j should be opposite to force on j from i."""
        system = LennardJonesSystem(n_particles=2, box_size=10.0)
        positions = torch.tensor([[[1.0, 1.0], [2.5, 1.0]]])

        forces = system.forces(positions)

        # F_01 should be -F_10 (Newton's third law)
        assert torch.allclose(forces[0, 0], -forces[0, 1], atol=1e-6)


class TestLennardJonesMinimumImage:
    """Tests for periodic boundary conditions."""

    def test_minimum_image_wrapping(self):
        """Minimum image should wrap large displacements."""
        system = LennardJonesSystem(n_particles=2, box_size=5.0)

        # Displacement of 4.0 should wrap to -1.0
        dr = torch.tensor([[4.0, 0.0]])
        wrapped = system.minimum_image(dr)

        assert torch.allclose(wrapped, torch.tensor([[-1.0, 0.0]]))

    def test_minimum_image_no_change(self):
        """Small displacements should not be changed."""
        system = LennardJonesSystem(n_particles=2, box_size=5.0)

        dr = torch.tensor([[1.0, -1.5]])
        wrapped = system.minimum_image(dr)

        assert torch.allclose(wrapped, dr)

    def test_periodic_interaction(self):
        """Particles should interact across periodic boundaries."""
        system = LennardJonesSystem(n_particles=2, box_size=5.0)

        # Particles at opposite edges (distance 1.0 across boundary)
        pos = torch.tensor([[[0.5, 2.5], [4.5, 2.5]]])
        E = system.energy(pos)

        # Should interact (distance = 1.0 < cutoff = 2.5)
        assert torch.isfinite(E)
        assert E.item() != 0


class TestLennardJonesScore:
    """Tests for score function."""

    def test_score_shape(self):
        """Score should have correct shape."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()
        score = system.score(positions, temperature=1.0)

        assert score.shape == positions.shape

    def test_score_scales_with_temperature(self):
        """Score should be inversely proportional to temperature."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()

        score_T1 = system.score(positions, temperature=1.0)
        score_T2 = system.score(positions, temperature=2.0)

        assert torch.allclose(score_T1, 2 * score_T2)


class TestLennardJonesConfiguration:
    """Tests for configuration generation."""

    def test_lattice_no_overlap(self):
        """Lattice configuration should have no overlapping particles."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()
        dist, _ = system.pairwise_distances(positions)

        # Minimum non-self distance should be greater than sigma
        mask = dist > 0.1
        min_dist = dist[mask].min()

        assert min_dist > system.sigma * 0.9

    def test_random_in_box(self):
        """Random configuration should be within box bounds."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.random_configuration(batch_size=10)

        assert (positions >= 0).all()
        assert (positions <= system.box_size).all()

    def test_continuous_conversion_roundtrip(self):
        """Continuous conversion should be invertible."""
        system = LennardJonesSystem(n_particles=16, box_size=5.0)
        positions = system.lattice_configuration()

        continuous = system.to_continuous(positions)
        back = system.from_continuous(continuous)

        assert torch.allclose(positions, back, atol=1e-6)


class TestLennardJonesProperties:
    """Tests for system properties."""

    def test_density(self):
        """Density should be N/V."""
        system = LennardJonesSystem(n_particles=16, box_size=4.0)
        expected = 16 / (4.0 ** 2)

        assert np.isclose(system.density, expected)

    def test_reduced_density(self):
        """Reduced density should be ρσ²."""
        system = LennardJonesSystem(n_particles=16, box_size=4.0, sigma=1.0)
        expected = 16 / (4.0 ** 2) * 1.0 ** 2

        assert np.isclose(system.reduced_density, expected)

    def test_configuration_shape(self):
        """Configuration shape should be (n_particles, 2)."""
        system = LennardJonesSystem(n_particles=25, box_size=6.0)

        assert system.configuration_shape == (25, 2)
        assert system.num_particles == 25


@pytest.fixture
def lj_system():
    """Create a standard LJ system for testing."""
    return LennardJonesSystem(n_particles=16, box_size=5.0)


@pytest.fixture
def lj_lattice(lj_system):
    """Create a lattice configuration."""
    return lj_system.lattice_configuration()
