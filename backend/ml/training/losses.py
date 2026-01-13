"""Loss functions for denoising score matching training.

This module provides various loss functions and weighting schemes
for training score-based diffusion models.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.diffusion import DiffusionProcess

# Numerical stability constant
EPS = 1e-8
