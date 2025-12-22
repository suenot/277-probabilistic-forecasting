"""
Proper Scoring Rules for Probabilistic Forecasts
================================================

Implements CRPS, Log Score, calibration metrics, and other
proper scoring rules for evaluating probabilistic forecasts.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
import warnings


def crps_empirical(
    samples: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute CRPS (Continuous Ranked Probability Score) from samples.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Where X, X' are independent samples from the forecast distribution.

    Args:
        samples: Forecast samples [n_samples, n_observations] or [n_samples]
        observation: Observed values [n_observations] or scalar

    Returns:
        CRPS values for each observation
    """
    samples = np.atleast_2d(samples)
    observation = np.atleast_1d(observation)

    n_samples = samples.shape[0]

    # E|X - y|
    abs_diff = np.abs(samples - observation).mean(axis=0)

    # E|X - X'| using all pairs
    # More efficient: sort samples and use formula
    sorted_samples = np.sort(samples, axis=0)

    # For sorted samples x_1 <= x_2 <= ... <= x_n:
    # E|X - X'| = (2/n^2) * sum_{i=1}^n (2i - 1 - n) * x_i
    n = n_samples
    weights = (2 * np.arange(1, n + 1) - 1 - n) / (n ** 2)
    spread = 2 * np.sum(weights[:, np.newaxis] * sorted_samples, axis=0)

    crps = abs_diff - 0.5 * spread

    return crps


def crps_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute CRPS for Gaussian forecasts (closed form).

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    Where z = (y - mu) / sigma

    Args:
        mu: Predicted means
        sigma: Predicted standard deviations
        observation: Observed values

    Returns:
        CRPS values
    """
    z = (observation - mu) / sigma

    crps = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) +
        2 * stats.norm.pdf(z) -
        1 / np.sqrt(np.pi)
    )

    return crps


def log_score(
    samples: np.ndarray,
    observation: np.ndarray,
    bandwidth: Optional[float] = None
) -> np.ndarray:
    """
    Compute Log Score using kernel density estimation.

    Log Score = log(p(y))

    Args:
        samples: Forecast samples [n_samples, n_observations]
        observation: Observed values
        bandwidth: KDE bandwidth (auto if None)

    Returns:
        Log score values (higher is better)
    """
    samples = np.atleast_2d(samples)
    observation = np.atleast_1d(observation)

    n_obs = observation.shape[0]
    log_scores = np.zeros(n_obs)

    for i in range(n_obs):
        if bandwidth is None:
            # Scott's rule
            bw = 1.06 * np.std(samples[:, i]) * (len(samples[:, i]) ** (-1/5))
        else:
            bw = bandwidth

        bw = max(bw, 1e-6)  # Prevent zero bandwidth

        # KDE estimate of density at observation
        kernel_vals = stats.norm.pdf(
            observation[i],
            loc=samples[:, i],
            scale=bw
        )
        density = np.mean(kernel_vals)

        log_scores[i] = np.log(density + 1e-10)

    return log_scores


def log_score_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute Log Score for Gaussian forecasts.

    Args:
        mu: Predicted means
        sigma: Predicted standard deviations
        observation: Observed values

    Returns:
        Log score values
    """
    return stats.norm.logpdf(observation, loc=mu, scale=sigma)


def quantile_score(
    quantile_pred: np.ndarray,
    observation: np.ndarray,
    tau: float
) -> np.ndarray:
    """
    Compute Quantile Score (Pinball Loss).

    L_tau(y, q) = (tau - I(y < q)) * (y - q)

    Args:
        quantile_pred: Predicted quantile values
        observation: Observed values
        tau: Quantile level (0-1)

    Returns:
        Quantile score values (lower is better)
    """
    error = observation - quantile_pred
    score = np.where(
        error >= 0,
        tau * error,
        (tau - 1) * error
    )
    return score


def interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    observation: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Compute Interval Score for prediction intervals.

    IS = (upper - lower) + (2/alpha) * (lower - y) * I(y < lower)
                        + (2/alpha) * (y - upper) * I(y > upper)

    Args:
        lower: Lower bound of interval
        upper: Upper bound of interval
        observation: Observed values
        alpha: Nominal coverage level (e.g., 0.1 for 90% interval)

    Returns:
        Interval score values (lower is better)
    """
    width = upper - lower
    below_penalty = (2 / alpha) * (lower - observation) * (observation < lower)
    above_penalty = (2 / alpha) * (observation - upper) * (observation > upper)

    return width + below_penalty + above_penalty


def pit_values(
    samples: np.ndarray,
    observation: np.ndarray
) -> np.ndarray:
    """
    Compute PIT (Probability Integral Transform) values.

    For well-calibrated forecasts, PIT values should be uniform on [0, 1].

    Args:
        samples: Forecast samples [n_samples, n_observations]
        observation: Observed values

    Returns:
        PIT values [n_observations]
    """
    samples = np.atleast_2d(samples)
    observation = np.atleast_1d(observation)

    # PIT = proportion of samples <= observation
    pit = (samples <= observation).mean(axis=0)

    return pit


def calibration_error(
    samples: np.ndarray,
    observation: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute calibration error metrics.

    Args:
        samples: Forecast samples [n_samples, n_observations]
        observation: Observed values
        n_bins: Number of bins for histogram

    Returns:
        Dictionary with calibration metrics
    """
    pit = pit_values(samples, observation)

    # Expected vs observed coverage at different levels
    quantile_levels = np.linspace(0.1, 0.9, 9)
    coverage_errors = []

    for level in quantile_levels:
        expected_coverage = level
        actual_coverage = (pit <= level).mean()
        coverage_errors.append(abs(actual_coverage - expected_coverage))

    # Histogram-based calibration
    hist, _ = np.histogram(pit, bins=n_bins, range=(0, 1))
    uniform_expected = len(pit) / n_bins
    chi_squared = ((hist - uniform_expected) ** 2 / uniform_expected).sum()

    return {
        'mean_calibration_error': np.mean(coverage_errors),
        'max_calibration_error': np.max(coverage_errors),
        'chi_squared': chi_squared,
        'pit_mean': pit.mean(),  # Should be ~0.5
        'pit_std': pit.std(),    # Should be ~0.289 for uniform
    }


def sharpness(
    samples: np.ndarray,
    quantiles: List[float] = [0.05, 0.95]
) -> Dict[str, float]:
    """
    Compute sharpness metrics (width of prediction intervals).

    Args:
        samples: Forecast samples [n_samples, n_observations]
        quantiles: Quantile levels for interval

    Returns:
        Dictionary with sharpness metrics
    """
    samples = np.atleast_2d(samples)

    lower_q, upper_q = quantiles[0], quantiles[-1]
    lower = np.percentile(samples, lower_q * 100, axis=0)
    upper = np.percentile(samples, upper_q * 100, axis=0)

    interval_widths = upper - lower

    return {
        'mean_interval_width': interval_widths.mean(),
        'median_interval_width': np.median(interval_widths),
        'std_interval_width': interval_widths.std(),
    }


def coverage(
    samples: np.ndarray,
    observation: np.ndarray,
    nominal_coverage: float = 0.90
) -> float:
    """
    Compute empirical coverage of prediction intervals.

    Args:
        samples: Forecast samples [n_samples, n_observations]
        observation: Observed values
        nominal_coverage: Nominal coverage level

    Returns:
        Empirical coverage (fraction of observations in interval)
    """
    samples = np.atleast_2d(samples)
    alpha = 1 - nominal_coverage

    lower = np.percentile(samples, (alpha / 2) * 100, axis=0)
    upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=0)

    in_interval = (observation >= lower) & (observation <= upper)

    return in_interval.mean()


def compute_all_metrics(
    samples: np.ndarray,
    observation: np.ndarray,
    mu: Optional[np.ndarray] = None,
    sigma: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all probabilistic forecast metrics.

    Args:
        samples: Forecast samples [n_samples, n_observations]
        observation: Observed values
        mu: Predicted means (optional, for Gaussian CRPS)
        sigma: Predicted std devs (optional, for Gaussian CRPS)

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # CRPS
    crps = crps_empirical(samples, observation)
    metrics['crps_mean'] = crps.mean()
    metrics['crps_std'] = crps.std()

    if mu is not None and sigma is not None:
        crps_gauss = crps_gaussian(mu, sigma, observation)
        metrics['crps_gaussian'] = crps_gauss.mean()

    # Log Score
    log_scores = log_score(samples, observation)
    metrics['log_score_mean'] = log_scores.mean()
    metrics['log_score_std'] = log_scores.std()

    if mu is not None and sigma is not None:
        log_scores_gauss = log_score_gaussian(mu, sigma, observation)
        metrics['log_score_gaussian'] = log_scores_gauss.mean()

    # Calibration
    cal_metrics = calibration_error(samples, observation)
    metrics.update(cal_metrics)

    # Sharpness
    sharp_metrics = sharpness(samples)
    metrics.update(sharp_metrics)

    # Coverage at different levels
    for cov_level in [0.50, 0.80, 0.90, 0.95]:
        metrics[f'coverage_{int(cov_level*100)}'] = coverage(
            samples, observation, cov_level
        )

    # Quantile scores
    for tau in [0.05, 0.25, 0.5, 0.75, 0.95]:
        q_pred = np.percentile(samples, tau * 100, axis=0)
        q_score = quantile_score(q_pred, observation, tau)
        metrics[f'quantile_score_{int(tau*100)}'] = q_score.mean()

    return metrics


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Generate synthetic forecasts and observations
    n_obs = 100
    n_samples = 500

    # True distribution: N(0, 1)
    true_mu = np.zeros(n_obs)
    true_sigma = np.ones(n_obs)

    # Observations from true distribution
    observations = np.random.randn(n_obs)

    # Well-calibrated forecast
    good_samples = np.random.randn(n_samples, n_obs)

    # Overconfident forecast (too narrow)
    overconfident_samples = np.random.randn(n_samples, n_obs) * 0.5

    # Biased forecast
    biased_samples = np.random.randn(n_samples, n_obs) + 0.5

    print("=" * 60)
    print("Well-Calibrated Forecast:")
    print("=" * 60)
    metrics_good = compute_all_metrics(
        good_samples, observations, true_mu, true_sigma
    )
    for k, v in metrics_good.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Overconfident Forecast:")
    print("=" * 60)
    metrics_over = compute_all_metrics(overconfident_samples, observations)
    for k, v in metrics_over.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Biased Forecast:")
    print("=" * 60)
    metrics_biased = compute_all_metrics(biased_samples, observations)
    for k, v in metrics_biased.items():
        print(f"  {k}: {v:.4f}")
