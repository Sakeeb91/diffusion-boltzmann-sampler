"""Noise schedules for diffusion models.

Provides various beta schedules that control the rate of noise addition
during the forward diffusion process.
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules.

    A noise schedule defines how the noise rate β(t) varies over time t ∈ [0, 1].
    For the VP-SDE, the forward process is:
        dx = -0.5 β(t) x dt + √β(t) dW
    """

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Compute noise rate β(t) at time t.

        Args:
            t: Time values in [0, 1], shape (batch,) or scalar

        Returns:
            β(t) values with same shape as t
        """
        pass

    def noise_level(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute signal and noise coefficients at time t.

        For the VP-SDE, we have:
            x_t = α_t x_0 + σ_t ε,  where ε ~ N(0, I)

        Args:
            t: Time values in [0, 1]

        Returns:
            (α_t, σ_t) coefficients
        """
        # Default numerical integration (can be overridden for analytic forms)
        integral = self._integrate_beta(t)
        alpha_t = torch.exp(-0.5 * integral)
        sigma_t = torch.sqrt(torch.clamp(1 - alpha_t**2, min=1e-8))
        return alpha_t, sigma_t

    def _integrate_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Integrate β(s) from 0 to t.

        Default implementation uses numerical integration.
        Subclasses should override with analytic forms when available.
        """
        # Simple trapezoidal integration
        n_steps = 100
        dt = t / n_steps
        integral = torch.zeros_like(t)

        for i in range(n_steps):
            t_i = i * dt
            t_ip1 = (i + 1) * dt
            integral = integral + 0.5 * (self.beta(t_i) + self.beta(t_ip1)) * dt

        return integral
