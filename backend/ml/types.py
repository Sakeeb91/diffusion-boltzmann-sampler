"""Common type definitions for the ML module."""

from typing import TypeVar, Callable, Protocol, Tuple, Union, List
import torch
from torch import Tensor

# Type aliases for clarity
SpinConfiguration = Tensor  # Shape: (..., size, size) with values in {-1, +1}
ParticleConfiguration = Tensor  # Shape: (..., num_particles, dim)
EnergyTensor = Tensor  # Shape: (...,) containing energy values
ScoreTensor = Tensor  # Shape: same as input, gradient of log probability

# Temperature type (must be positive)
Temperature = float

# Time step for diffusion (in [0, 1])
DiffusionTime = Union[float, Tensor]

# Batch size type
BatchSize = int


class Sampler(Protocol):
    """Protocol for samplers that generate configurations."""

    def sample(self, n_samples: int, **kwargs) -> Tensor:
        """Generate samples from the target distribution."""
        ...


class ScoreFunction(Protocol):
    """Protocol for score functions (gradient of log probability)."""

    def __call__(self, x: Tensor, t: DiffusionTime) -> Tensor:
        """Compute score at configuration x and time t."""
        ...


# Callback types for training
LossCallback = Callable[[int, float], None]
SampleCallback = Callable[[int, Tensor], None]

# Device type
Device = Union[str, torch.device]

# Learning rate schedule
LRSchedule = Callable[[int], float]
