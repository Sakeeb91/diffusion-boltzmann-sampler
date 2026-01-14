"""Statistical analysis functions for comparing samplers."""

import torch
import numpy as np
from typing import Any, Dict, List


def pair_correlation(samples: torch.Tensor) -> Dict[str, List[float]]:
    """Compute spin-spin correlation function C(r) = <s_0 s_r>.

    Args:
        samples: Tensor of shape (n_samples, size, size)

    Returns:
        Dictionary with 'r' (distances) and 'C_r' (correlations)
    """
    samples_np = samples.numpy() if isinstance(samples, torch.Tensor) else samples
    n_samples, size, _ = samples_np.shape
    max_r = size // 2

    correlations = np.zeros(max_r)
    counts = np.zeros(max_r)

    # Sample a subset of reference points for efficiency
    n_ref = min(100, size * size)
    ref_indices = np.random.choice(size * size, n_ref, replace=False)

    for sample in samples_np:
        for idx in ref_indices:
            i, j = divmod(idx, size)
            ref_spin = sample[i, j]

            # Compute correlations at all distances
            for di in range(-max_r, max_r + 1):
                for dj in range(-max_r, max_r + 1):
                    r = int(np.sqrt(di**2 + dj**2))
                    if 0 < r < max_r:
                        ni, nj = (i + di) % size, (j + dj) % size
                        correlations[r] += ref_spin * sample[ni, nj]
                        counts[r] += 1

    # Normalize
    correlations = np.divide(
        correlations, counts, where=counts > 0, out=np.zeros_like(correlations)
    )

    return {"r": list(range(max_r)), "C_r": correlations.tolist()}


def magnetization_distribution(
    samples: torch.Tensor, n_bins: int = 50
) -> Dict[str, List[float]]:
    """Compute magnetization distribution P(M).

    Args:
        samples: Tensor of shape (n_samples, size, size)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with 'M' (magnetization values) and 'P_M' (probabilities)
    """
    # Compute magnetization for each sample
    if len(samples.shape) == 4:  # (batch, channel, h, w)
        mags = samples.mean(dim=(-1, -2, -3)).numpy()
    else:  # (batch, h, w)
        mags = samples.mean(dim=(-1, -2)).numpy()

    # Create histogram
    hist, bin_edges = np.histogram(mags, bins=n_bins, density=True, range=(-1, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {"M": bin_centers.tolist(), "P_M": hist.tolist()}


def autocorrelation_time(
    samples: torch.Tensor, observable: str = "magnetization"
) -> float:
    """Estimate integrated autocorrelation time.

    Args:
        samples: Time-ordered sequence of samples
        observable: Which observable to use ("magnetization" or "energy")

    Returns:
        Integrated autocorrelation time τ_int
    """
    # Compute observable time series
    if observable == "magnetization":
        if len(samples.shape) == 4:
            obs = samples.mean(dim=(-1, -2, -3)).numpy()
        else:
            obs = samples.mean(dim=(-1, -2)).numpy()
    else:
        raise ValueError(f"Unknown observable: {observable}")

    n = len(obs)
    if n < 10:
        return 1.0

    mean = obs.mean()
    var = obs.var()

    if var < 1e-10:
        return 1.0

    # Compute autocorrelation function
    obs_centered = obs - mean
    autocorr = np.correlate(obs_centered, obs_centered, mode="full")
    autocorr = autocorr[n - 1 :]  # Keep only non-negative lags
    autocorr = autocorr / (var * np.arange(n, 0, -1))  # Normalize

    # Integrate until first negative value (or use window)
    tau_int = 0.5  # Include lag 0 contribution
    for i in range(1, min(n, 100)):
        if autocorr[i] < 0:
            break
        tau_int += autocorr[i]

    return float(tau_int)


def energy_histogram(
    samples: torch.Tensor, ising_model, n_bins: int = 50
) -> Dict[str, List[float]]:
    """Compute energy histogram.

    Args:
        samples: Tensor of shape (n_samples, size, size) or (n_samples, 1, size, size)
        ising_model: IsingModel instance
        n_bins: Number of histogram bins

    Returns:
        Dictionary with 'E' (energies) and 'P_E' (probabilities)
    """
    # Handle channel dimension
    if len(samples.shape) == 4:
        samples = samples.squeeze(1)

    # Compute energies
    energies = ising_model.energy_per_spin(samples).numpy()

    # Create histogram
    hist, bin_edges = np.histogram(energies, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return {"E": bin_centers.tolist(), "P_E": hist.tolist()}


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Reference distribution (e.g., MCMC samples)
        q: Approximating distribution (e.g., diffusion samples)
        epsilon: Small value for numerical stability

    Returns:
        KL divergence value (non-negative, 0 if identical)
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # Normalize to ensure valid probability distributions
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)

    # Add epsilon for numerical stability
    p = p + epsilon
    q = q + epsilon

    # Re-normalize
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Compute symmetric KL divergence (Jensen-Shannon-like).

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value for numerical stability

    Returns:
        Symmetric KL divergence: 0.5 * (D_KL(P||Q) + D_KL(Q||P))
    """
    return 0.5 * (kl_divergence(p, q, epsilon) + kl_divergence(q, p, epsilon))


def magnetization_kl_divergence(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    n_bins: int = 50,
) -> Dict[str, float]:
    """Compute KL divergence between magnetization distributions.

    Args:
        samples1: Reference samples (e.g., MCMC)
        samples2: Test samples (e.g., diffusion)
        n_bins: Number of histogram bins

    Returns:
        Dictionary with kl_divergence and symmetric_kl_divergence
    """
    dist1 = magnetization_distribution(samples1, n_bins)
    dist2 = magnetization_distribution(samples2, n_bins)

    p = np.array(dist1["P_M"])
    q = np.array(dist2["P_M"])

    return {
        "kl_divergence": kl_divergence(p, q),
        "symmetric_kl_divergence": symmetric_kl_divergence(p, q),
    }


def energy_kl_divergence(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
    n_bins: int = 50,
) -> Dict[str, float]:
    """Compute KL divergence between energy distributions.

    Args:
        samples1: Reference samples (e.g., MCMC)
        samples2: Test samples (e.g., diffusion)
        ising_model: IsingModel instance
        n_bins: Number of histogram bins

    Returns:
        Dictionary with kl_divergence and symmetric_kl_divergence
    """
    dist1 = energy_histogram(samples1, ising_model, n_bins)
    dist2 = energy_histogram(samples2, ising_model, n_bins)

    p = np.array(dist1["P_E"])
    q = np.array(dist2["P_E"])

    return {
        "kl_divergence": kl_divergence(p, q),
        "symmetric_kl_divergence": symmetric_kl_divergence(p, q),
    }


def wasserstein_distance_1d(
    samples1: np.ndarray,
    samples2: np.ndarray,
) -> float:
    """Compute 1D Wasserstein distance (Earth Mover's Distance).

    Uses the efficient formula for 1D: W_1 = integral |F(x) - G(x)| dx
    where F and G are the cumulative distribution functions.

    Args:
        samples1: First set of 1D samples
        samples2: Second set of 1D samples

    Returns:
        Wasserstein-1 distance
    """
    samples1 = np.sort(np.asarray(samples1).flatten())
    samples2 = np.sort(np.asarray(samples2).flatten())

    # Combine and sort all unique values
    all_values = np.sort(np.unique(np.concatenate([samples1, samples2])))

    # Compute CDFs at each value
    cdf1 = np.searchsorted(samples1, all_values, side="right") / len(samples1)
    cdf2 = np.searchsorted(samples2, all_values, side="right") / len(samples2)

    # Integrate |CDF1 - CDF2|
    deltas = np.diff(np.concatenate([[all_values[0]], all_values]))
    return float(np.sum(np.abs(cdf1 - cdf2) * deltas))


def magnetization_wasserstein(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
) -> float:
    """Compute Wasserstein distance between magnetization distributions.

    Args:
        samples1: First set of spin samples
        samples2: Second set of spin samples

    Returns:
        Wasserstein-1 distance between magnetization distributions
    """
    # Compute magnetizations
    if len(samples1.shape) == 4:
        mag1 = samples1.mean(dim=(-1, -2, -3)).numpy()
        mag2 = samples2.mean(dim=(-1, -2, -3)).numpy()
    else:
        mag1 = samples1.mean(dim=(-1, -2)).numpy()
        mag2 = samples2.mean(dim=(-1, -2)).numpy()

    return wasserstein_distance_1d(mag1, mag2)


def energy_wasserstein(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
) -> float:
    """Compute Wasserstein distance between energy distributions.

    Args:
        samples1: First set of spin samples
        samples2: Second set of spin samples
        ising_model: IsingModel instance

    Returns:
        Wasserstein-1 distance between energy distributions
    """
    # Handle channel dimension
    if len(samples1.shape) == 4:
        s1 = samples1.squeeze(1)
        s2 = samples2.squeeze(1)
    else:
        s1, s2 = samples1, samples2

    # Compute energies
    e1 = ising_model.energy_per_spin(s1).numpy()
    e2 = ising_model.energy_per_spin(s2).numpy()

    return wasserstein_distance_1d(e1, e2)


def correlation_function_comparison(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
) -> Dict[str, Any]:
    """Compare pair correlation functions between sample sets.

    Args:
        samples1: First set of samples (e.g., MCMC)
        samples2: Second set of samples (e.g., diffusion)

    Returns:
        Dictionary with:
        - r: Distance values
        - C1_r: Correlations for samples1
        - C2_r: Correlations for samples2
        - rmse: Root mean squared error between correlations
        - max_diff: Maximum absolute difference
        - correlation_length_ratio: Ratio of correlation lengths
    """
    corr1 = pair_correlation(samples1)
    corr2 = pair_correlation(samples2)

    c1 = np.array(corr1["C_r"])
    c2 = np.array(corr2["C_r"])

    # Compute comparison metrics
    rmse = float(np.sqrt(np.mean((c1 - c2) ** 2)))
    max_diff = float(np.max(np.abs(c1 - c2)))

    # Estimate correlation length (distance where C(r) drops to 1/e)
    threshold = 1.0 / np.e

    def estimate_correlation_length(c):
        for i, val in enumerate(c):
            if val < threshold:
                return i
        return len(c)

    xi1 = estimate_correlation_length(c1)
    xi2 = estimate_correlation_length(c2)
    xi_ratio = xi2 / xi1 if xi1 > 0 else float("inf")

    return {
        "r": corr1["r"],
        "C1_r": corr1["C_r"],
        "C2_r": corr2["C_r"],
        "rmse": rmse,
        "max_diff": max_diff,
        "correlation_length_1": xi1,
        "correlation_length_2": xi2,
        "correlation_length_ratio": xi_ratio,
    }


def compare_distributions(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
) -> Dict[str, float]:
    """Compare two sets of samples statistically.

    Args:
        samples1: First set of samples (e.g., MCMC)
        samples2: Second set of samples (e.g., diffusion)
        ising_model: IsingModel instance

    Returns:
        Dictionary of comparison metrics
    """
    # Magnetization statistics
    mag1 = magnetization_distribution(samples1)
    mag2 = magnetization_distribution(samples2)

    # Energy statistics
    e1 = energy_histogram(samples1, ising_model)
    e2 = energy_histogram(samples2, ising_model)

    # Compute mean and variance
    if len(samples1.shape) == 4:
        m1 = samples1.mean(dim=(-1, -2, -3))
        m2 = samples2.mean(dim=(-1, -2, -3))
    else:
        m1 = samples1.mean(dim=(-1, -2))
        m2 = samples2.mean(dim=(-1, -2))

    return {
        "mag_mean_diff": abs(m1.mean().item() - m2.mean().item()),
        "mag_var_diff": abs(m1.var().item() - m2.var().item()),
        "samples1_mean_mag": m1.mean().item(),
        "samples2_mean_mag": m2.mean().item(),
        "samples1_var_mag": m1.var().item(),
        "samples2_var_mag": m2.var().item(),
    }


def comprehensive_comparison(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    ising_model,
    n_bins: int = 50,
) -> Dict[str, Any]:
    """Perform comprehensive comparison between two sample sets.

    Combines all comparison metrics into a single report.

    Args:
        samples1: Reference samples (e.g., MCMC gold standard)
        samples2: Test samples (e.g., diffusion model)
        ising_model: IsingModel instance
        n_bins: Number of histogram bins

    Returns:
        Dictionary with all comparison metrics organized by category
    """
    # Basic distribution comparison
    basic = compare_distributions(samples1, samples2, ising_model)

    # KL divergence
    mag_kl = magnetization_kl_divergence(samples1, samples2, n_bins)
    energy_kl = energy_kl_divergence(samples1, samples2, ising_model, n_bins)

    # Wasserstein distance
    mag_w = magnetization_wasserstein(samples1, samples2)
    energy_w = energy_wasserstein(samples1, samples2, ising_model)

    # Correlation functions
    corr_comp = correlation_function_comparison(samples1, samples2)

    return {
        "basic_statistics": basic,
        "kl_divergence": {
            "magnetization": mag_kl,
            "energy": energy_kl,
        },
        "wasserstein": {
            "magnetization": mag_w,
            "energy": energy_w,
        },
        "correlation": {
            "rmse": corr_comp["rmse"],
            "max_diff": corr_comp["max_diff"],
            "correlation_length_ratio": corr_comp["correlation_length_ratio"],
        },
        "summary": {
            "mag_kl": mag_kl["symmetric_kl_divergence"],
            "energy_kl": energy_kl["symmetric_kl_divergence"],
            "mag_wasserstein": mag_w,
            "energy_wasserstein": energy_w,
            "correlation_rmse": corr_comp["rmse"],
        },
    }


if __name__ == "__main__":
    from ..systems.ising import IsingModel
    from ..samplers.mcmc import MetropolisHastings

    # Generate test samples
    model = IsingModel(size=16)
    sampler = MetropolisHastings(model, temperature=2.27)
    samples = sampler.sample(n_samples=100, n_sweeps=10, burn_in=200)

    print("Testing correlation functions...")
    corr = pair_correlation(samples)
    print(f"Distances: {corr['r'][:5]}")
    print(f"Correlations: {[f'{c:.3f}' for c in corr['C_r'][:5]]}")

    print("\nTesting magnetization distribution...")
    mag_dist = magnetization_distribution(samples)
    print(f"M range: [{min(mag_dist['M']):.2f}, {max(mag_dist['M']):.2f}]")

    print("\nTesting autocorrelation time...")
    tau = autocorrelation_time(samples)
    print(f"τ_int = {tau:.2f}")

    print("\nTesting energy histogram...")
    e_hist = energy_histogram(samples, model)
    print(f"E/N range: [{min(e_hist['E']):.2f}, {max(e_hist['E']):.2f}]")

    print("\nAll analysis tests passed!")
