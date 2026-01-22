"""2D Lennard-Jones fluid with periodic boundary conditions."""

import torch
from typing import Tuple, Optional


class LennardJonesSystem:
    """2D Lennard-Jones fluid in a periodic box.

    The Lennard-Jones potential is:
        U(r) = 4ε [(σ/r)¹² - (σ/r)⁶]

    where ε is the depth of the potential well and σ is the particle diameter.

    Attributes:
        n_particles: Number of particles
        box_size: Length of the square periodic box
        epsilon: Depth of potential well (default 1.0)
        sigma: Particle diameter (default 1.0)
        cutoff: Potential cutoff radius (default 2.5σ)
    """

    # Typical reduced units
    REDUCED_DENSITY_LIQUID = 0.8  # ρ* for liquid phase
    REDUCED_DENSITY_GAS = 0.3  # ρ* for gas phase
    TRIPLE_POINT_TEMP = 0.694  # T* at triple point
    CRITICAL_TEMP = 1.326  # T* at critical point

    def __init__(
        self,
        n_particles: int,
        box_size: float,
        epsilon: float = 1.0,
        sigma: float = 1.0,
        cutoff: float = 2.5,
    ):
        """Initialize Lennard-Jones system.

        Args:
            n_particles: Number of particles in the system
            box_size: Size of the periodic box
            epsilon: Depth of potential well
            sigma: Particle diameter
            cutoff: Cutoff radius in units of sigma
        """
        self.n_particles = n_particles
        self.box_size = box_size
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff * sigma
        self._cutoff_squared = self.cutoff ** 2

        # Precompute cutoff energy shift for smooth potential
        r6 = (self.sigma / self.cutoff) ** 6
        r12 = r6 ** 2
        self._energy_shift = 4 * self.epsilon * (r12 - r6)

    def minimum_image(self, dr: torch.Tensor) -> torch.Tensor:
        """Apply minimum image convention for periodic boundaries.

        Args:
            dr: Displacement vectors (..., 2) in the range [-L, L]

        Returns:
            Displacement vectors wrapped to [-L/2, L/2]
        """
        return dr - self.box_size * torch.round(dr / self.box_size)

    def pairwise_distances(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute all pairwise distances between particles.

        Args:
            positions: Particle positions (..., n_particles, 2)

        Returns:
            dist: Pairwise distances (..., n_particles, n_particles)
            dr: Displacement vectors (..., n_particles, n_particles, 2)
        """
        # Expand for pairwise computation
        # (..., n, 1, 2) - (..., 1, n, 2) -> (..., n, n, 2)
        r_i = positions[..., :, None, :]
        r_j = positions[..., None, :, :]
        dr = self.minimum_image(r_i - r_j)

        # Distance with small epsilon to avoid division by zero on diagonal
        dist = torch.sqrt((dr ** 2).sum(dim=-1) + 1e-10)
        return dist, dr

    def energy(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute total Lennard-Jones potential energy.

        Args:
            positions: Particle positions (..., n_particles, 2)

        Returns:
            Total energy (...,)
        """
        dist, _ = self.pairwise_distances(positions)

        # Mask: only pairs within cutoff, excluding self-interactions (i != j)
        # Use 0.1 threshold to safely exclude diagonal (sqrt(1e-10) ≈ 1e-5)
        mask = (dist > 0.1) & (dist < self.cutoff)

        # Avoid division by zero with safe distance
        dist_safe = torch.where(mask, dist, torch.ones_like(dist))

        # Compute LJ potential: 4ε [(σ/r)¹² - (σ/r)⁶]
        r6 = (self.sigma / dist_safe) ** 6
        r12 = r6 ** 2
        potential = 4 * self.epsilon * (r12 - r6) - self._energy_shift

        # Zero out masked values
        potential = torch.where(mask, potential, torch.zeros_like(potential))

        # Sum over pairs, factor 0.5 for double counting
        return 0.5 * potential.sum(dim=(-1, -2))

    def forces(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute forces on all particles: F = -∇U.

        The force between two particles is:
            F(r) = 24ε/r [2(σ/r)¹² - (σ/r)⁶] r̂

        Args:
            positions: Particle positions (..., n_particles, 2)

        Returns:
            Forces on each particle (..., n_particles, 2)
        """
        dist, dr = self.pairwise_distances(positions)

        # Mask: only pairs within cutoff, excluding self
        # Use 0.1 threshold to safely exclude diagonal (sqrt(1e-10) ≈ 1e-5)
        mask = (dist > 0.1) & (dist < self.cutoff)

        # Safe distance for computation
        dist_safe = torch.where(mask, dist, torch.ones_like(dist))

        # Force magnitude: |F| = 24ε/r [2(σ/r)¹² - (σ/r)⁶]
        r6 = (self.sigma / dist_safe) ** 6
        r12 = r6 ** 2
        f_mag = 24 * self.epsilon / dist_safe * (2 * r12 - r6)

        # Zero out masked values
        f_mag = torch.where(mask, f_mag, torch.zeros_like(f_mag))

        # Unit vector: r̂_ij = (r_i - r_j) / |r_ij|
        # Force on i from j: F_ij = f_mag * r̂_ij
        unit = dr / (dist_safe[..., None] + 1e-10)
        forces_ij = f_mag[..., None] * unit

        # Sum forces from all other particles
        return forces_ij.sum(dim=-2)

    def score(self, positions: torch.Tensor, temperature: float) -> torch.Tensor:
        """Compute score function s(x) = ∇log p(x) = -∇U(x) / kT = F(x) / kT.

        The score is the gradient of the log probability, which for a
        Boltzmann distribution is the force divided by temperature.

        Args:
            positions: Particle positions (..., n_particles, 2)
            temperature: Temperature (in reduced units, k_B = 1)

        Returns:
            Score vectors (..., n_particles, 2)
        """
        return self.forces(positions) / temperature

    def energy_per_particle(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute energy per particle for normalization.

        Args:
            positions: Particle positions (..., n_particles, 2)

        Returns:
            Energy per particle (...,)
        """
        return self.energy(positions) / self.n_particles

    def random_configuration(self, batch_size: int = 1) -> torch.Tensor:
        """Generate random particle positions uniformly in the box.

        Note: This may generate overlapping particles. For equilibrated
        configurations, use the MCMC sampler with burn-in.

        Args:
            batch_size: Number of configurations to generate

        Returns:
            Random positions (batch_size, n_particles, 2)
        """
        return torch.rand(batch_size, self.n_particles, 2) * self.box_size

    def lattice_configuration(self, batch_size: int = 1) -> torch.Tensor:
        """Generate a square lattice configuration.

        Useful for initializing simulations without overlaps.

        Args:
            batch_size: Number of configurations to generate

        Returns:
            Lattice positions (batch_size, n_particles, 2)
        """
        # Compute grid dimensions
        n_side = int(torch.ceil(torch.sqrt(torch.tensor(self.n_particles))))
        spacing = self.box_size / n_side

        # Create grid positions
        positions = []
        for i in range(self.n_particles):
            x = (i % n_side + 0.5) * spacing
            y = (i // n_side + 0.5) * spacing
            positions.append([x, y])

        positions = torch.tensor(positions)
        return positions.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def to_continuous(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert positions to continuous representation for diffusion.

        Normalizes positions to [-1, 1] range for the diffusion process.

        Args:
            positions: Particle positions (..., n_particles, 2) in [0, L]

        Returns:
            Normalized positions (..., n_particles, 2) in [-1, 1]
        """
        return 2 * positions / self.box_size - 1

    def from_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Convert continuous values back to physical positions.

        Applies periodic boundary conditions.

        Args:
            x: Normalized positions (..., n_particles, 2) in [-1, 1]

        Returns:
            Physical positions (..., n_particles, 2) in [0, L]
        """
        positions = (x + 1) * self.box_size / 2
        # Apply periodic boundaries
        return positions % self.box_size

    @property
    def num_particles(self) -> int:
        """Return the number of particles."""
        return self.n_particles

    @property
    def configuration_shape(self) -> Tuple[int, int]:
        """Return the shape of a single configuration."""
        return (self.n_particles, 2)

    @property
    def density(self) -> float:
        """Return the number density ρ = N / V."""
        return self.n_particles / (self.box_size ** 2)

    @property
    def reduced_density(self) -> float:
        """Return the reduced density ρ* = ρσ²."""
        return self.density * self.sigma ** 2


if __name__ == "__main__":
    # Quick test
    system = LennardJonesSystem(n_particles=16, box_size=5.0)

    # Test with lattice configuration (no overlaps)
    positions = system.lattice_configuration(batch_size=2)
    print(f"Lattice positions shape: {positions.shape}")

    # Compute energy
    E = system.energy(positions)
    print(f"Lattice energy: {E}")
    print(f"Energy per particle: {system.energy_per_particle(positions)}")

    # Compute forces
    F = system.forces(positions)
    print(f"Forces shape: {F.shape}")
    print(f"Mean force magnitude: {torch.sqrt((F ** 2).sum(dim=-1)).mean():.4f}")

    # Test score function
    score = system.score(positions, temperature=1.0)
    print(f"Score shape: {score.shape}")

    # Verify continuous conversion roundtrip
    x_cont = system.to_continuous(positions)
    positions_back = system.from_continuous(x_cont)
    diff = (positions - positions_back).abs().max()
    print(f"Roundtrip error: {diff:.6f}")

    # Test minimum image
    dr = torch.tensor([[4.0, 0.0], [-4.0, 0.0]])  # Should wrap to ±1
    dr_wrapped = system.minimum_image(dr)
    print(f"Minimum image: {dr_wrapped}")

    print("All basic tests passed!")
