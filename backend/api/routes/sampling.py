"""Sampling API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch

from ...ml.systems.ising import IsingModel
from ...ml.samplers.mcmc import MetropolisHastings


router = APIRouter()


class MCMCSampleRequest(BaseModel):
    """Request model for MCMC sampling."""

    temperature: float = Field(2.27, ge=0.1, le=10.0, description="Temperature")
    lattice_size: int = Field(32, ge=8, le=128, description="Lattice size")
    n_samples: int = Field(10, ge=1, le=1000, description="Number of samples")
    n_sweeps: int = Field(10, ge=1, le=100, description="Sweeps between samples")
    burn_in: int = Field(100, ge=0, le=1000, description="Burn-in sweeps")


class DiffusionSampleRequest(BaseModel):
    """Request model for diffusion sampling."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(32, ge=8, le=64)
    n_samples: int = Field(1, ge=1, le=10)
    num_steps: int = Field(100, ge=10, le=500)


class SampleResponse(BaseModel):
    """Response model for sampling."""

    samples: List[List[List[float]]]  # (n_samples, size, size)
    energies: List[float]
    magnetizations: List[float]
    temperature: float
    lattice_size: int


@router.post("/mcmc", response_model=SampleResponse)
async def sample_mcmc(request: MCMCSampleRequest) -> SampleResponse:
    """Generate samples using Metropolis-Hastings MCMC.

    This is the gold-standard baseline for Ising model sampling.
    """
    try:
        # Create model and sampler
        model = IsingModel(size=request.lattice_size)
        sampler = MetropolisHastings(model, temperature=request.temperature)

        # Generate samples
        samples = sampler.sample(
            n_samples=request.n_samples,
            n_sweeps=request.n_sweeps,
            burn_in=request.burn_in,
        )

        # Compute observables
        energies = model.energy_per_spin(samples).tolist()
        magnetizations = model.magnetization(samples).tolist()

        return SampleResponse(
            samples=samples.tolist(),
            energies=energies,
            magnetizations=magnetizations,
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diffusion", response_model=SampleResponse)
async def sample_diffusion(request: DiffusionSampleRequest) -> SampleResponse:
    """Generate samples using diffusion model.

    Uses the trained score network for reverse diffusion sampling.
    """
    try:
        from ...ml.samplers.diffusion import PretrainedDiffusionSampler

        # Create sampler
        sampler = PretrainedDiffusionSampler(
            lattice_size=request.lattice_size,
            num_steps=request.num_steps,
        )

        # Generate samples using heuristic (demo mode)
        samples = sampler.sample_heuristic(
            batch_size=request.n_samples,
            temperature=request.temperature,
        )

        # Remove channel dimension if present
        if len(samples.shape) == 4:
            samples = samples.squeeze(1)

        # Create model for observables
        model = IsingModel(size=request.lattice_size)
        energies = model.energy_per_spin(samples).tolist()
        magnetizations = model.magnetization(samples).tolist()

        return SampleResponse(
            samples=samples.tolist(),
            energies=energies,
            magnetizations=magnetizations,
            temperature=request.temperature,
            lattice_size=request.lattice_size,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random")
async def get_random_configuration(
    lattice_size: int = 32,
) -> dict:
    """Get a random spin configuration."""
    model = IsingModel(size=lattice_size)
    spins = model.random_configuration(batch_size=1).squeeze(0)

    return {
        "spins": spins.tolist(),
        "energy": model.energy_per_spin(spins.unsqueeze(0)).item(),
        "magnetization": model.magnetization(spins.unsqueeze(0)).item(),
        "lattice_size": lattice_size,
    }


@router.get("/ground_state")
async def get_ground_state(
    lattice_size: int = 32,
    positive: bool = True,
) -> dict:
    """Get a ground state configuration (all +1 or all -1)."""
    model = IsingModel(size=lattice_size)

    if positive:
        spins = torch.ones(lattice_size, lattice_size)
    else:
        spins = -torch.ones(lattice_size, lattice_size)

    return {
        "spins": spins.tolist(),
        "energy": model.energy_per_spin(spins.unsqueeze(0)).item(),
        "magnetization": model.magnetization(spins.unsqueeze(0)).item(),
        "lattice_size": lattice_size,
    }


class CompareRequest(BaseModel):
    """Request model for comparing samplers."""

    temperature: float = Field(2.27, ge=0.1, le=10.0)
    lattice_size: int = Field(16, ge=8, le=64)
    n_samples: int = Field(50, ge=10, le=200)
    mcmc_sweeps: int = Field(20, ge=5, le=100)
    mcmc_burn_in: int = Field(200, ge=50, le=1000)
    diffusion_steps: int = Field(100, ge=50, le=500)


class CompareResponse(BaseModel):
    """Response model for sampler comparison."""

    temperature: float
    lattice_size: int
    n_samples: int
    summary: dict
    basic_statistics: dict
    kl_divergence: dict
    wasserstein: dict
    correlation: dict


@router.post("/compare", response_model=CompareResponse)
async def compare_samplers(request: CompareRequest) -> CompareResponse:
    """Compare diffusion sampler against MCMC baseline.

    Generates samples from both samplers and computes comprehensive
    comparison metrics including KL divergence, Wasserstein distance,
    and correlation function analysis.
    """
    try:
        from ...ml.samplers.diffusion import PretrainedDiffusionSampler
        from ...ml.analysis import comprehensive_comparison

        # Create model and MCMC sampler
        model = IsingModel(size=request.lattice_size)
        mcmc_sampler = MetropolisHastings(model, temperature=request.temperature)

        # Generate MCMC samples (reference)
        mcmc_samples = mcmc_sampler.sample(
            n_samples=request.n_samples,
            n_sweeps=request.mcmc_sweeps,
            burn_in=request.mcmc_burn_in,
        )

        # Generate diffusion samples
        diffusion_sampler = PretrainedDiffusionSampler(
            lattice_size=request.lattice_size,
            num_steps=request.diffusion_steps,
        )
        diffusion_samples = diffusion_sampler.sample_heuristic(
            batch_size=request.n_samples,
            temperature=request.temperature,
        )

        # Remove channel dimension if present
        if len(diffusion_samples.shape) == 4:
            diffusion_samples = diffusion_samples.squeeze(1)

        # Compute comprehensive comparison
        comparison = comprehensive_comparison(
            model=model,
            samples_a=mcmc_samples,
            samples_b=diffusion_samples,
            name_a="MCMC",
            name_b="Diffusion",
        )

        return CompareResponse(
            temperature=request.temperature,
            lattice_size=request.lattice_size,
            n_samples=request.n_samples,
            summary=comparison["summary"],
            basic_statistics=comparison["basic_statistics"],
            kl_divergence=comparison["kl_divergence"],
            wasserstein=comparison["wasserstein"],
            correlation=comparison["correlation"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
