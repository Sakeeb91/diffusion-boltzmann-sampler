"""Lennard-Jones sampling API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch

from ...ml.systems.lennard_jones import LennardJonesSystem
from ...ml.samplers.langevin import LangevinDynamics, UnderdampedLangevin
from ...ml.analysis.rdf import compute_rdf_from_system, rdf_statistics


router = APIRouter(prefix="/lj", tags=["lennard-jones"])


class LJSampleRequest(BaseModel):
    """Request model for Lennard-Jones sampling."""

    n_particles: int = Field(16, ge=4, le=100, description="Number of particles")
    box_size: float = Field(5.0, ge=2.0, le=20.0, description="Box size")
    temperature: float = Field(1.0, ge=0.1, le=5.0, description="Temperature (reduced units)")
    n_samples: int = Field(10, ge=1, le=100, description="Number of samples")
    n_steps: int = Field(100, ge=10, le=1000, description="Steps between samples")
    burn_in: int = Field(500, ge=0, le=5000, description="Burn-in steps")
    dt: float = Field(0.001, ge=0.0001, le=0.01, description="Time step")
    sampler_type: str = Field(
        "overdamped", description="Sampler type: overdamped or underdamped"
    )


class LJSampleResponse(BaseModel):
    """Response model for Lennard-Jones sampling."""

    positions: List[List[List[float]]]  # (n_samples, n_particles, 2)
    energies: List[float]
    energies_per_particle: List[float]
    temperature: float
    n_particles: int
    box_size: float
    density: float


@router.post("/sample", response_model=LJSampleResponse)
async def sample_lj(request: LJSampleRequest) -> LJSampleResponse:
    """Generate samples from Lennard-Jones fluid using Langevin dynamics.

    Uses Metropolis-adjusted Langevin dynamics for exact sampling
    from the Boltzmann distribution.
    """
    try:
        # Create system
        system = LennardJonesSystem(
            n_particles=request.n_particles,
            box_size=request.box_size,
        )

        # Create sampler
        if request.sampler_type == "underdamped":
            sampler = UnderdampedLangevin(
                system=system,
                temperature=request.temperature,
                dt=request.dt,
            )
        else:
            sampler = LangevinDynamics(
                system=system,
                temperature=request.temperature,
                dt=request.dt,
            )

        # Generate samples
        samples = sampler.sample(
            n_samples=request.n_samples,
            n_steps=request.n_steps,
            burn_in=request.burn_in,
        )

        # Compute observables
        energies = system.energy(samples).tolist()
        energies_per_particle = system.energy_per_particle(samples).tolist()

        return LJSampleResponse(
            positions=samples.tolist(),
            energies=energies,
            energies_per_particle=energies_per_particle,
            temperature=request.temperature,
            n_particles=request.n_particles,
            box_size=request.box_size,
            density=system.density,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LJRDFRequest(BaseModel):
    """Request model for RDF computation."""

    n_particles: int = Field(25, ge=9, le=100)
    box_size: float = Field(6.0, ge=2.0, le=20.0)
    temperature: float = Field(1.0, ge=0.1, le=5.0)
    n_samples: int = Field(50, ge=10, le=200)
    n_steps: int = Field(100, ge=10, le=500)
    burn_in: int = Field(1000, ge=100, le=5000)
    n_bins: int = Field(50, ge=20, le=200)


class LJRDFResponse(BaseModel):
    """Response model for RDF computation."""

    r: List[float]
    g_r: List[float]
    statistics: Dict[str, float]
    n_samples: int
    temperature: float


@router.post("/rdf", response_model=LJRDFResponse)
async def compute_rdf(request: LJRDFRequest) -> LJRDFResponse:
    """Compute radial distribution function g(r) from Lennard-Jones samples.

    Generates samples using Langevin dynamics and computes the RDF,
    which characterizes the liquid structure.
    """
    try:
        # Create system and sampler
        system = LennardJonesSystem(
            n_particles=request.n_particles,
            box_size=request.box_size,
        )
        sampler = LangevinDynamics(
            system=system,
            temperature=request.temperature,
            dt=0.001,
        )

        # Generate samples
        samples = sampler.sample(
            n_samples=request.n_samples,
            n_steps=request.n_steps,
            burn_in=request.burn_in,
        )

        # Compute RDF
        r, g_r = compute_rdf_from_system(
            positions=samples,
            system=system,
            n_bins=request.n_bins,
        )

        # Compute statistics
        stats = rdf_statistics(r, g_r, sigma=system.sigma)

        return LJRDFResponse(
            r=r.tolist(),
            g_r=g_r.tolist(),
            statistics=stats,
            n_samples=request.n_samples,
            temperature=request.temperature,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lattice")
async def get_lattice_configuration(
    n_particles: int = 16,
    box_size: float = 5.0,
) -> dict:
    """Get a square lattice configuration (useful for initialization)."""
    system = LennardJonesSystem(n_particles=n_particles, box_size=box_size)
    positions = system.lattice_configuration(batch_size=1).squeeze(0)

    return {
        "positions": positions.tolist(),
        "energy": system.energy(positions.unsqueeze(0)).item(),
        "energy_per_particle": system.energy_per_particle(positions.unsqueeze(0)).item(),
        "n_particles": n_particles,
        "box_size": box_size,
        "density": system.density,
    }


@router.get("/random")
async def get_random_configuration(
    n_particles: int = 16,
    box_size: float = 5.0,
) -> dict:
    """Get a random configuration (may have overlaps)."""
    system = LennardJonesSystem(n_particles=n_particles, box_size=box_size)
    positions = system.random_configuration(batch_size=1).squeeze(0)

    return {
        "positions": positions.tolist(),
        "energy": system.energy(positions.unsqueeze(0)).item(),
        "energy_per_particle": system.energy_per_particle(positions.unsqueeze(0)).item(),
        "n_particles": n_particles,
        "box_size": box_size,
        "density": system.density,
    }


class LJSystemInfo(BaseModel):
    """Information about the Lennard-Jones system."""

    critical_temperature: float
    triple_point_temperature: float
    liquid_density: float
    gas_density: float


@router.get("/info", response_model=LJSystemInfo)
async def get_system_info() -> LJSystemInfo:
    """Get physical constants for the Lennard-Jones system."""
    return LJSystemInfo(
        critical_temperature=LennardJonesSystem.CRITICAL_TEMP,
        triple_point_temperature=LennardJonesSystem.TRIPLE_POINT_TEMP,
        liquid_density=LennardJonesSystem.REDUCED_DENSITY_LIQUID,
        gas_density=LennardJonesSystem.REDUCED_DENSITY_GAS,
    )
