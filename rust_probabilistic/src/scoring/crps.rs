//! CRPS (Continuous Ranked Probability Score)

use crate::distributions::forecast::ForecastDistribution;
use crate::distributions::gaussian::GaussianDistribution;

/// Compute CRPS from samples (empirical)
///
/// CRPS = E|X - y| - 0.5 * E|X - X'|
pub fn compute_crps(forecast: &ForecastDistribution, observation: f64) -> f64 {
    let samples = &forecast.samples;
    let n = samples.len() as f64;

    // E|X - y|
    let abs_diff: f64 = samples.iter().map(|&x| (x - observation).abs()).sum::<f64>() / n;

    // E|X - X'| using sorted samples
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // For sorted samples, E|X - X'| = (2/n^2) * sum_{i=1}^n (2i - 1 - n) * x_i
    let spread: f64 = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let weight = (2.0 * (i as f64 + 1.0) - 1.0 - n) / (n * n);
            2.0 * weight * x
        })
        .sum();

    abs_diff - 0.5 * spread
}

/// Compute CRPS for Gaussian distribution (closed form)
///
/// CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))
pub fn compute_crps_gaussian(mu: f64, sigma: f64, observation: f64) -> f64 {
    let dist = GaussianDistribution::new(mu, sigma);
    dist.crps(observation)
}

/// Compute CRPS for multiple forecasts and observations
pub fn compute_crps_batch(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
) -> Vec<f64> {
    forecasts
        .iter()
        .zip(observations.iter())
        .map(|(f, &o)| compute_crps(f, o))
        .collect()
}

/// Compute mean CRPS over multiple forecasts
pub fn mean_crps(forecasts: &[ForecastDistribution], observations: &[f64]) -> f64 {
    if forecasts.is_empty() {
        return 0.0;
    }

    let crps_values = compute_crps_batch(forecasts, observations);
    crps_values.iter().sum::<f64>() / crps_values.len() as f64
}

/// CRPS skill score relative to a baseline (climatology)
///
/// CRPSS = 1 - CRPS_forecast / CRPS_baseline
pub fn crps_skill_score(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
    baseline: &ForecastDistribution,
) -> f64 {
    let crps_forecast = mean_crps(forecasts, observations);

    let crps_baseline: f64 = observations
        .iter()
        .map(|&o| compute_crps(baseline, o))
        .sum::<f64>()
        / observations.len() as f64;

    if crps_baseline > 0.0 {
        1.0 - crps_forecast / crps_baseline
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crps_perfect_forecast() {
        // Perfect forecast: all samples equal observation
        let samples = vec![1.0; 100];
        let forecast = ForecastDistribution::from_samples(samples);
        let crps = compute_crps(&forecast, 1.0);

        // CRPS should be 0 for perfect forecast
        assert!(crps.abs() < 0.01);
    }

    #[test]
    fn test_crps_gaussian() {
        // Compare empirical and closed-form CRPS
        let mu = 0.0;
        let sigma = 1.0;

        let forecast = ForecastDistribution::from_gaussian(mu, sigma, 10000);
        let observation = 0.5;

        let crps_empirical = compute_crps(&forecast, observation);
        let crps_closed = compute_crps_gaussian(mu, sigma, observation);

        // Should be close (within Monte Carlo error)
        assert!((crps_empirical - crps_closed).abs() < 0.05);
    }

    #[test]
    fn test_crps_worse_for_bad_forecast() {
        // Good forecast centered on observation
        let good_forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 1000);
        let crps_good = compute_crps(&good_forecast, 0.0);

        // Bad forecast far from observation
        let bad_forecast = ForecastDistribution::from_gaussian(5.0, 1.0, 1000);
        let crps_bad = compute_crps(&bad_forecast, 0.0);

        assert!(crps_bad > crps_good);
    }

    #[test]
    fn test_crps_penalizes_overconfidence() {
        let observation = 1.0;

        // Well-calibrated forecast
        let calibrated = ForecastDistribution::from_gaussian(1.0, 1.0, 1000);

        // Overconfident forecast (too narrow)
        let overconfident = ForecastDistribution::from_gaussian(0.0, 0.1, 1000);

        let crps_calibrated = compute_crps(&calibrated, observation);
        let crps_overconfident = compute_crps(&overconfident, observation);

        // Overconfident forecast should have worse CRPS when wrong
        assert!(crps_overconfident > crps_calibrated);
    }
}
