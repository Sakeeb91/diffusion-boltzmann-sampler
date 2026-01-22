"""Langevin dynamics MCMC sampler for continuous systems like Lennard-Jones."""

import torch
import numpy as np
from typing import Generator, Optional
from ..systems.lennard_jones import LennardJonesSystem


class LangevinDynamics:
    """Langevin dynamics sampler for continuous particle systems.

    Implements overdamped Langevin dynamics (Brownian dynamics) with
    Metropolis-Hastings correction for exact sampling from the Boltzmann
    distribution.

    The update rule is:
        x' = x + dt * F(x) / (k_B T) + sqrt(2 * dt) * η

    where F(x) is the force, T is temperature, and η is Gaussian noise.
    """

    def __init__(
        self,
        system: LennardJonesSystem,
        temperature: float,
        dt: float = 0.001,
        use_mh_correction: bool = True,
    ):
        """Initialize Langevin sampler.

        Args:
            system: LennardJonesSystem instance
            temperature: Temperature (in reduced units, k_B = 1)
            dt: Time step for integration
            use_mh_correction: Whether to use Metropolis-Hastings correction
        """
        self.system = system
        self.temperature = temperature
        self.dt = dt
        self.use_mh_correction = use_mh_correction
        self.beta = 1.0 / temperature

        # Precompute noise scale
        self.noise_scale = np.sqrt(2 * dt)

    def step(self, positions: torch.Tensor) -> torch.Tensor:
        """Perform single Langevin dynamics step.

        Args:
            positions: Current positions (n_particles, 2)

        Returns:
            Updated positions
        """
        # Compute forces
        forces = self.system.forces(positions.unsqueeze(0)).squeeze(0)

        # Propose new positions
        noise = torch.randn_like(positions) * self.noise_scale
        positions_new = positions + self.dt * forces / self.temperature + noise

        # Apply periodic boundary conditions
        positions_new = positions_new % self.system.box_size

        if self.use_mh_correction:
            # Metropolis-Hastings correction for detailed balance
            E_old = self.system.energy(positions.unsqueeze(0)).item()
            E_new = self.system.energy(positions_new.unsqueeze(0)).item()

            # Compute forward and backward proposal probabilities
            forces_new = self.system.forces(positions_new.unsqueeze(0)).squeeze(0)

            # Forward: x -> x'
            drift_fwd = positions + self.dt * forces / self.temperature
            log_q_fwd = -0.5 * ((positions_new - drift_fwd) ** 2).sum() / (2 * self.dt)

            # Backward: x' -> x
            drift_bwd = positions_new + self.dt * forces_new / self.temperature
            log_q_bwd = -0.5 * ((positions - drift_bwd) ** 2).sum() / (2 * self.dt)

            # Acceptance probability
            log_alpha = -self.beta * (E_new - E_old) + log_q_bwd - log_q_fwd

            if np.log(np.random.random()) < log_alpha.item():
                return positions_new
            else:
                return positions
        else:
            return positions_new

    def sample(
        self,
        n_samples: int,
        n_steps: int = 100,
        initial: Optional[torch.Tensor] = None,
        burn_in: int = 1000,
    ) -> torch.Tensor:
        """Generate samples from equilibrium distribution.

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of steps between samples
            initial: Initial configuration (lattice if None)
            burn_in: Number of burn-in steps

        Returns:
            Tensor of shape (n_samples, n_particles, 2)
        """
        # Initialize
        if initial is None:
            positions = self.system.lattice_configuration(batch_size=1).squeeze(0)
        else:
            positions = initial.clone()

        # Burn-in
        for _ in range(burn_in):
            positions = self.step(positions)

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_steps):
                positions = self.step(positions)
            samples.append(positions.clone())

        return torch.stack(samples)

    def sample_with_trajectory(
        self,
        n_steps: int,
        initial: Optional[torch.Tensor] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """Generate samples yielding each configuration.

        Useful for animation.

        Args:
            n_steps: Total number of steps
            initial: Initial configuration

        Yields:
            Position configuration after each step
        """
        if initial is None:
            positions = self.system.lattice_configuration(batch_size=1).squeeze(0)
        else:
            positions = initial.clone()

        yield positions.clone()

        for _ in range(n_steps):
            positions = self.step(positions)
            yield positions.clone()


class UnderdampedLangevin:
    """Underdamped Langevin dynamics (Hamiltonian Monte Carlo variant).

    Includes momentum for better exploration of phase space.
    Uses leapfrog integration for energy conservation.
    """

    def __init__(
        self,
        system: LennardJonesSystem,
        temperature: float,
        dt: float = 0.001,
        mass: float = 1.0,
        friction: float = 1.0,
    ):
        """Initialize underdamped Langevin sampler.

        Args:
            system: LennardJonesSystem instance
            temperature: Temperature (in reduced units)
            dt: Time step
            mass: Particle mass
            friction: Friction coefficient (gamma)
        """
        self.system = system
        self.temperature = temperature
        self.dt = dt
        self.mass = mass
        self.friction = friction
        self.beta = 1.0 / temperature

        # Precompute integration coefficients
        # For BAOAB integrator (velocity Verlet with Ornstein-Uhlenbeck thermostat)
        self.c1 = np.exp(-friction * dt)
        self.c2 = np.sqrt(1 - self.c1 ** 2) * np.sqrt(temperature / mass)

    def step(
        self, positions: torch.Tensor, velocities: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform single underdamped Langevin step (BAOAB integrator).

        Args:
            positions: Current positions (n_particles, 2)
            velocities: Current velocities (n_particles, 2)

        Returns:
            (new_positions, new_velocities)
        """
        # B: Half-step velocity update from forces
        forces = self.system.forces(positions.unsqueeze(0)).squeeze(0)
        velocities = velocities + 0.5 * self.dt * forces / self.mass

        # A: Half-step position update
        positions = positions + 0.5 * self.dt * velocities

        # O: Ornstein-Uhlenbeck thermostat
        noise = torch.randn_like(velocities)
        velocities = self.c1 * velocities + self.c2 * noise

        # A: Half-step position update
        positions = positions + 0.5 * self.dt * velocities

        # Apply periodic boundaries
        positions = positions % self.system.box_size

        # B: Half-step velocity update from new forces
        forces = self.system.forces(positions.unsqueeze(0)).squeeze(0)
        velocities = velocities + 0.5 * self.dt * forces / self.mass

        return positions, velocities

    def sample(
        self,
        n_samples: int,
        n_steps: int = 100,
        initial: Optional[torch.Tensor] = None,
        burn_in: int = 1000,
    ) -> torch.Tensor:
        """Generate samples from equilibrium distribution.

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of steps between samples
            initial: Initial positions (lattice if None)
            burn_in: Number of burn-in steps

        Returns:
            Tensor of shape (n_samples, n_particles, 2)
        """
        # Initialize positions
        if initial is None:
            positions = self.system.lattice_configuration(batch_size=1).squeeze(0)
        else:
            positions = initial.clone()

        # Initialize velocities from Maxwell-Boltzmann distribution
        velocities = torch.randn_like(positions) * np.sqrt(
            self.temperature / self.mass
        )

        # Burn-in
        for _ in range(burn_in):
            positions, velocities = self.step(positions, velocities)

        # Collect samples
        samples = []
        for _ in range(n_samples):
            for _ in range(n_steps):
                positions, velocities = self.step(positions, velocities)
            samples.append(positions.clone())

        return torch.stack(samples)


if __name__ == "__main__":
    # Test Langevin sampler
    system = LennardJonesSystem(n_particles=16, box_size=5.0)

    # Test overdamped Langevin
    sampler = LangevinDynamics(system, temperature=1.0, dt=0.001)
    samples = sampler.sample(n_samples=5, n_steps=100, burn_in=500)
    print(f"Overdamped samples shape: {samples.shape}")

    # Compute energies
    energies = system.energy_per_particle(samples)
    print(f"Energy per particle: {energies.mean():.4f} ± {energies.std():.4f}")

    # Test underdamped Langevin
    sampler_ud = UnderdampedLangevin(system, temperature=1.0, dt=0.001)
    samples_ud = sampler_ud.sample(n_samples=5, n_steps=100, burn_in=500)
    print(f"Underdamped samples shape: {samples_ud.shape}")

    energies_ud = system.energy_per_particle(samples_ud)
    print(f"Energy per particle (UD): {energies_ud.mean():.4f} ± {energies_ud.std():.4f}")

    # Test trajectory generation
    trajectory = list(sampler.sample_with_trajectory(n_steps=10))
    print(f"Trajectory length: {len(trajectory)}")

    print("Langevin sampler tests passed!")
