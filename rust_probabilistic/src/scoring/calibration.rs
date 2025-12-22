//! Calibration metrics for probabilistic forecasts

use crate::distributions::forecast::ForecastDistribution;

/// Compute PIT (Probability Integral Transform) values
///
/// For well-calibrated forecasts, PIT values should be uniform on [0, 1]
pub fn compute_pit(forecasts: &[ForecastDistribution], observations: &[f64]) -> Vec<f64> {
    forecasts
        .iter()
        .zip(observations.iter())
        .map(|(forecast, &obs)| {
            let count = forecast.samples.iter().filter(|&&x| x <= obs).count();
            count as f64 / forecast.samples.len() as f64
        })
        .collect()
}

/// Check calibration at multiple levels
pub fn calibration_error(forecasts: &[ForecastDistribution], observations: &[f64]) -> f64 {
    let pit_values = compute_pit(forecasts, observations);
    let n = pit_values.len() as f64;

    if n == 0.0 {
        return 0.0;
    }

    // Check coverage at different nominal levels
    let levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let mut total_error = 0.0;

    for &level in &levels {
        let expected = level;
        let actual = pit_values.iter().filter(|&&p| p <= level).count() as f64 / n;
        total_error += (actual - expected).abs();
    }

    total_error / levels.len() as f64
}

/// Compute coverage at a given nominal level
pub fn coverage(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
    nominal_coverage: f64,
) -> f64 {
    if forecasts.is_empty() {
        return 0.0;
    }

    let alpha = 1.0 - nominal_coverage;
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;

    let in_interval: usize = forecasts
        .iter()
        .zip(observations.iter())
        .filter(|(forecast, &obs)| {
            let lower = forecast.quantile(lower_q);
            let upper = forecast.quantile(upper_q);
            obs >= lower && obs <= upper
        })
        .count();

    in_interval as f64 / forecasts.len() as f64
}

/// Compute average interval width (sharpness)
pub fn sharpness(forecasts: &[ForecastDistribution], coverage_level: f64) -> f64 {
    if forecasts.is_empty() {
        return 0.0;
    }

    let alpha = 1.0 - coverage_level;
    let lower_q = alpha / 2.0;
    let upper_q = 1.0 - alpha / 2.0;

    let total_width: f64 = forecasts
        .iter()
        .map(|f| {
            let lower = f.quantile(lower_q);
            let upper = f.quantile(upper_q);
            upper - lower
        })
        .sum();

    total_width / forecasts.len() as f64
}

/// Compute reliability diagram data
/// Returns (nominal_level, actual_coverage) pairs
pub fn reliability_diagram(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
    num_bins: usize,
) -> Vec<(f64, f64)> {
    let pit_values = compute_pit(forecasts, observations);
    let n = pit_values.len() as f64;

    if n == 0.0 {
        return Vec::new();
    }

    (1..=num_bins)
        .map(|i| {
            let level = i as f64 / num_bins as f64;
            let actual = pit_values.iter().filter(|&&p| p <= level).count() as f64 / n;
            (level, actual)
        })
        .collect()
}

/// Compute Brier score for probability of event
/// Event is defined as observation > threshold
pub fn brier_score(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
    threshold: f64,
) -> f64 {
    if forecasts.is_empty() {
        return 0.0;
    }

    let total: f64 = forecasts
        .iter()
        .zip(observations.iter())
        .map(|(forecast, &obs)| {
            let prob = forecast.prob_greater_than(threshold);
            let actual = if obs > threshold { 1.0 } else { 0.0 };
            (prob - actual).powi(2)
        })
        .sum();

    total / forecasts.len() as f64
}

/// Summary of calibration metrics
#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    pub mean_calibration_error: f64,
    pub pit_mean: f64,
    pub pit_std: f64,
    pub coverage_50: f64,
    pub coverage_90: f64,
    pub coverage_95: f64,
    pub sharpness_90: f64,
}

/// Compute all calibration metrics
pub fn calibration_summary(
    forecasts: &[ForecastDistribution],
    observations: &[f64],
) -> CalibrationSummary {
    let pit_values = compute_pit(forecasts, observations);
    let n = pit_values.len() as f64;

    let pit_mean = if n > 0.0 {
        pit_values.iter().sum::<f64>() / n
    } else {
        0.5
    };

    let pit_std = if n > 1.0 {
        let variance =
            pit_values.iter().map(|&p| (p - pit_mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    } else {
        0.0
    };

    CalibrationSummary {
        mean_calibration_error: calibration_error(forecasts, observations),
        pit_mean,
        pit_std,
        coverage_50: coverage(forecasts, observations, 0.50),
        coverage_90: coverage(forecasts, observations, 0.90),
        coverage_95: coverage(forecasts, observations, 0.95),
        sharpness_90: sharpness(forecasts, 0.90),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_well_calibrated_coverage() {
        // Create well-calibrated forecasts
        let mut forecasts = Vec::new();
        let mut observations = Vec::new();

        for _ in 0..1000 {
            let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 500);
            let obs = forecast.samples[rand::random::<usize>() % forecast.samples.len()];
            forecasts.push(forecast);
            observations.push(obs);
        }

        let cov_90 = coverage(&forecasts, &observations, 0.90);

        // Should be close to 90%
        assert!((cov_90 - 0.90).abs() < 0.1);
    }

    #[test]
    fn test_pit_uniform() {
        // Well-calibrated forecasts should have uniform PIT
        let mut forecasts = Vec::new();
        let mut observations = Vec::new();

        for _ in 0..1000 {
            let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 500);
            let obs = forecast.samples[rand::random::<usize>() % forecast.samples.len()];
            forecasts.push(forecast);
            observations.push(obs);
        }

        let pit = compute_pit(&forecasts, &observations);
        let pit_mean: f64 = pit.iter().sum::<f64>() / pit.len() as f64;

        // Mean of uniform should be ~0.5
        assert!((pit_mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_calibration_error() {
        // Well-calibrated should have low error
        let mut forecasts = Vec::new();
        let mut observations = Vec::new();

        for _ in 0..500 {
            let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 500);
            let obs = forecast.samples[rand::random::<usize>() % forecast.samples.len()];
            forecasts.push(forecast);
            observations.push(obs);
        }

        let error = calibration_error(&forecasts, &observations);
        assert!(error < 0.1);
    }
}
