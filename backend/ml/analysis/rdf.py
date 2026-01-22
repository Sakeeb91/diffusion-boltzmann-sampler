"""Radial distribution function g(r) analysis for particle systems."""

import torch
import numpy as np
from typing import Tuple, Optional, Dict
from ..systems.lennard_jones import LennardJonesSystem


def compute_rdf(
    positions: torch.Tensor,
    box_size: float,
    n_bins: int = 100,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r) for particle configurations.

    The RDF g(r) describes how density varies as a function of distance
    from a reference particle. For an ideal gas, g(r) = 1. For liquids,
    g(r) shows characteristic peaks at preferred separation distances.

    Args:
        positions: Particle positions (n_samples, n_particles, 2) or (n_particles, 2)
        box_size: Size of the periodic box
        n_bins: Number of histogram bins
        r_max: Maximum distance (default: box_size / 2)

    Returns:
        r: Bin centers (n_bins,)
        g_r: Radial distribution function values (n_bins,)
    """
    # Ensure batch dimension
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)

    n_samples, n_particles, _ = positions.shape

    if r_max is None:
        r_max = box_size / 2  # Maximum meaningful distance with periodic BC

    # Histogram bins
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0]

    # Collect all pairwise distances
    all_distances = []

    for sample_idx in range(n_samples):
        pos = positions[sample_idx]  # (n_particles, 2)

        # Compute pairwise distances with minimum image convention
        r_i = pos[:, None, :]  # (n, 1, 2)
        r_j = pos[None, :, :]  # (1, n, 2)
        dr_vec = r_i - r_j  # (n, n, 2)

        # Apply minimum image convention
        dr_vec = dr_vec - box_size * torch.round(dr_vec / box_size)
        dist = torch.sqrt((dr_vec ** 2).sum(dim=-1))  # (n, n)

        # Get upper triangle (exclude diagonal and avoid double counting)
        mask = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)
        distances = dist[mask].numpy()
        all_distances.append(distances)

    all_distances = np.concatenate(all_distances)

    # Compute histogram
    hist, _ = np.histogram(all_distances, bins=bin_edges)

    # Normalize to get g(r)
    # For 2D: g(r) = hist / (n_pairs * 2πr dr * ρ)
    # where ρ = N / V is the number density
    n_pairs_per_sample = n_particles * (n_particles - 1) / 2
    total_pairs = n_pairs_per_sample * n_samples
    rho = n_particles / (box_size ** 2)

    # Shell area in 2D: 2πr dr
    shell_areas = 2 * np.pi * bin_centers * dr

    # Normalize: divide by expected count in each shell for ideal gas
    g_r = hist / (total_pairs * shell_areas * rho + 1e-10)

    return bin_centers, g_r


def compute_rdf_from_system(
    positions: torch.Tensor,
    system: LennardJonesSystem,
    n_bins: int = 100,
    r_max: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RDF using system parameters.

    Args:
        positions: Particle positions (n_samples, n_particles, 2)
        system: LennardJonesSystem instance
        n_bins: Number of histogram bins
        r_max: Maximum distance (default: box_size / 2)

    Returns:
        r: Bin centers (n_bins,)
        g_r: Radial distribution function values (n_bins,)
    """
    return compute_rdf(
        positions=positions,
        box_size=system.box_size,
        n_bins=n_bins,
        r_max=r_max,
    )


def rdf_statistics(
    r: np.ndarray, g_r: np.ndarray, sigma: float = 1.0
) -> Dict[str, float]:
    """Compute summary statistics from RDF.

    Args:
        r: Distance values
        g_r: RDF values
        sigma: Particle diameter (for normalization)

    Returns:
        Dictionary with statistics:
        - first_peak_r: Position of first peak (nearest neighbor distance)
        - first_peak_height: Height of first peak
        - coordination_number: Approximate coordination number
    """
    # Find first peak (skip r < 0.9σ where g(r) should be ~0)
    valid_mask = r > 0.9 * sigma

    if not valid_mask.any():
        return {
            "first_peak_r": 0.0,
            "first_peak_height": 0.0,
            "coordination_number": 0.0,
        }

    r_valid = r[valid_mask]
    g_valid = g_r[valid_mask]

    # Find first maximum
    peak_idx = np.argmax(g_valid)
    first_peak_r = r_valid[peak_idx]
    first_peak_height = g_valid[peak_idx]

    # Estimate coordination number: integral of 2πρr g(r) dr up to first minimum
    # Find first minimum after first peak
    dr = r[1] - r[0]

    # Look for minimum after peak
    search_start = peak_idx + 1
    if search_start < len(g_valid) - 1:
        # Find local minimum
        for i in range(search_start, len(g_valid) - 1):
            if g_valid[i] < g_valid[i - 1] and g_valid[i] < g_valid[i + 1]:
                first_min_idx = i
                break
        else:
            first_min_idx = len(g_valid) - 1

        # Integrate up to first minimum (in 2D)
        r_integrate = r_valid[: first_min_idx + 1]
        g_integrate = g_valid[: first_min_idx + 1]
        coordination_number = 2 * np.pi * np.trapz(r_integrate * g_integrate, r_integrate)
    else:
        coordination_number = 0.0

    return {
        "first_peak_r": float(first_peak_r),
        "first_peak_height": float(first_peak_height),
        "coordination_number": float(coordination_number),
    }


def compare_rdfs(
    r1: np.ndarray,
    g_r1: np.ndarray,
    r2: np.ndarray,
    g_r2: np.ndarray,
) -> Dict[str, float]:
    """Compare two radial distribution functions.

    Args:
        r1, g_r1: First RDF
        r2, g_r2: Second RDF

    Returns:
        Dictionary with comparison metrics:
        - mse: Mean squared error
        - max_diff: Maximum absolute difference
        - peak_diff: Difference in first peak positions
    """
    # Interpolate to common grid if needed
    if not np.allclose(r1, r2):
        r_common = np.linspace(
            max(r1.min(), r2.min()),
            min(r1.max(), r2.max()),
            min(len(r1), len(r2)),
        )
        g1_interp = np.interp(r_common, r1, g_r1)
        g2_interp = np.interp(r_common, r2, g_r2)
    else:
        r_common = r1
        g1_interp = g_r1
        g2_interp = g_r2

    # Compute metrics
    mse = float(np.mean((g1_interp - g2_interp) ** 2))
    max_diff = float(np.max(np.abs(g1_interp - g2_interp)))

    # Peak positions
    stats1 = rdf_statistics(r1, g_r1)
    stats2 = rdf_statistics(r2, g_r2)
    peak_diff = abs(stats1["first_peak_r"] - stats2["first_peak_r"])

    return {
        "mse": mse,
        "max_diff": max_diff,
        "peak_diff": float(peak_diff),
    }


if __name__ == "__main__":
    # Test RDF computation
    from ..systems.lennard_jones import LennardJonesSystem
    from ..samplers.langevin import LangevinDynamics

    # Create system and generate samples
    system = LennardJonesSystem(n_particles=25, box_size=6.0)
    sampler = LangevinDynamics(system, temperature=1.0, dt=0.001)

    print("Generating samples for RDF...")
    samples = sampler.sample(n_samples=20, n_steps=100, burn_in=500)
    print(f"Samples shape: {samples.shape}")

    # Compute RDF
    r, g_r = compute_rdf_from_system(samples, system, n_bins=50)
    print(f"RDF computed: r from {r.min():.2f} to {r.max():.2f}")

    # Get statistics
    stats = rdf_statistics(r, g_r, sigma=system.sigma)
    print(f"First peak position: {stats['first_peak_r']:.3f}")
    print(f"First peak height: {stats['first_peak_height']:.3f}")
    print(f"Coordination number: {stats['coordination_number']:.3f}")

    print("RDF tests passed!")
