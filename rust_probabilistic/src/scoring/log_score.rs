//! Log Score (Logarithmic Scoring Rule)

use crate::distributions::forecast::ForecastDistribution;
use std::f64::consts::PI;

/// Compute log score using kernel density estimation
///
/// Log Score = log(p(y))
pub fn compute_log_score(forecast: &ForecastDistribution, observation: f64) -> f64 {
    let samples = &forecast.samples;
    let n = samples.len() as f64;

    // Bandwidth using Scott's rule
    let bandwidth = 1.06 * forecast.std * n.powf(-0.2);
    let bandwidth = bandwidth.max(1e-8);

    // KDE estimate of density at observation
    let density: f64 = samples
        .iter()
        .map(|&x| gaussian_kernel(observation, x, bandwidth))
        .sum::<f64>()
        / n;

    (density + 1e-10).ln()
}

/// Gaussian kernel for KDE
fn gaussian_kernel(x: f64, center: f64, bandwidth: f64) -> f64 {
    let z = (x - center) / bandwidth;
    (-0.5 * z * z).exp() / (bandwidth * (2.0 * PI).sqrt())
}

/// Compute log score for Gaussian distribution (closed form)
pub fn compute_log_score_gaussian(mu: f64, sigma: f64, observation: f64) -> f64 {
    let z = (observation - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * PI).ln()
}

/// Compute log scores for multiple forecasts
pub fn compute_log_score_batch(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
) -> Vec<f64> {
    forecasts
        .iter()
        .zip(observations.iter())
        .map(|(f, &o)| compute_log_score(f, o))
        .collect()
}

/// Compute mean log score
pub fn mean_log_score(forecasts: &[ForecastDistribution], observations: &[f64]) -> f64 {
    if forecasts.is_empty() {
        return f64::NEG_INFINITY;
    }

    let scores = compute_log_score_batch(forecasts, observations);
    scores.iter().sum::<f64>() / scores.len() as f64
}

/// Compute ignorance score (negative log score)
/// Lower is better
pub fn ignorance_score(forecast: &ForecastDistribution, observation: f64) -> f64 {
    -compute_log_score(forecast, observation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_score_at_mean() {
        let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);
        let log_score = compute_log_score(&forecast, 0.0);

        // Log score at mean of N(0,1) should be around -log(sqrt(2*pi)) ≈ -0.919
        let closed_form = compute_log_score_gaussian(0.0, 1.0, 0.0);
        assert!((log_score - closed_form).abs() < 0.1);
    }

    #[test]
    fn test_log_score_penalizes_outliers() {
        let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);

        let score_at_mean = compute_log_score(&forecast, 0.0);
        let score_at_tail = compute_log_score(&forecast, 3.0);

        // Score at tail should be worse (more negative)
        assert!(score_at_tail < score_at_mean);
    }

    #[test]
    fn test_gaussian_log_score() {
        let mu = 5.0;
        let sigma = 2.0;
        let observation = 6.0;

        let score = compute_log_score_gaussian(mu, sigma, observation);

        // Manual calculation
        let z = (observation - mu) / sigma; // 0.5
        let expected = -0.5 * z * z - sigma.ln() - 0.5 * (2.0 * PI).ln();

        assert!((score - expected).abs() < 0.001);
    }
}
