//! Kelly Criterion for optimal position sizing

use crate::distributions::forecast::ForecastDistribution;

/// Compute Kelly fraction for binary outcomes
///
/// f* = (p * b - q) / b
///
/// where:
///   p = probability of win
///   q = 1 - p = probability of loss
///   b = win/loss ratio (net odds)
pub fn kelly_fraction_binary(prob_win: f64, win_return: f64, loss_return: f64) -> f64 {
    if loss_return >= 0.0 || win_return <= 0.0 {
        return 0.0;
    }

    let q = 1.0 - prob_win;
    let b = win_return / loss_return.abs();

    let kelly = (prob_win * b - q) / b;

    kelly
}

/// Compute Kelly fraction for continuous distribution
///
/// f* = argmax_f E[log(1 + f * R)]
///
/// Solved numerically using samples
pub fn kelly_fraction(forecast: &ForecastDistribution) -> f64 {
    // Grid search for optimal fraction
    let mut best_f = 0.0;
    let mut best_growth = f64::NEG_INFINITY;

    // Search from -2 to 2 (allowing short positions)
    for i in -200..=200 {
        let f = i as f64 / 100.0;
        let growth = expected_log_growth(forecast, f);

        if growth > best_growth {
            best_growth = growth;
            best_f = f;
        }
    }

    // Refine with finer search around best
    let start = (best_f - 0.1).max(-2.0);
    let end = (best_f + 0.1).min(2.0);

    for i in 0..=100 {
        let f = start + (end - start) * i as f64 / 100.0;
        let growth = expected_log_growth(forecast, f);

        if growth > best_growth {
            best_growth = growth;
            best_f = f;
        }
    }

    best_f
}

/// Compute expected log growth rate
fn expected_log_growth(forecast: &ForecastDistribution, f: f64) -> f64 {
    let n = forecast.samples.len() as f64;

    let total: f64 = forecast
        .samples
        .iter()
        .map(|&r| {
            let wealth_ratio = 1.0 + f * r;
            if wealth_ratio > 0.0 {
                wealth_ratio.ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .filter(|x| x.is_finite())
        .sum();

    total / n
}

/// Compute fractional Kelly (safer than full Kelly)
pub fn fractional_kelly(forecast: &ForecastDistribution, fraction: f64) -> f64 {
    kelly_fraction(forecast) * fraction
}

/// Compute Kelly fraction with leverage constraint
pub fn kelly_with_constraint(
    forecast: &ForecastDistribution,
    max_leverage: f64,
    min_leverage: f64,
) -> f64 {
    let kelly = kelly_fraction(forecast);
    kelly.clamp(min_leverage, max_leverage)
}

/// Compute Kelly fraction considering parameter uncertainty
///
/// When uncertain about the distribution, use more conservative sizing
pub fn robust_kelly(forecast: &ForecastDistribution, confidence: f64) -> f64 {
    // Use a more pessimistic distribution
    // Shift mean down and increase variance based on uncertainty

    let mean = forecast.mean;
    let std = forecast.std;

    // Adjust for uncertainty (lower confidence = more conservative)
    let adjusted_mean = mean * confidence;
    let adjusted_std = std / confidence.sqrt();

    let adjusted_forecast = ForecastDistribution::from_gaussian(
        adjusted_mean,
        adjusted_std,
        forecast.samples.len(),
    );

    kelly_fraction(&adjusted_forecast)
}

/// Expected growth rate at given Kelly fraction
pub fn expected_growth_rate(forecast: &ForecastDistribution, kelly_f: f64) -> f64 {
    expected_log_growth(forecast, kelly_f)
}

/// Expected growth rate using full Kelly
pub fn kelly_growth_rate(forecast: &ForecastDistribution) -> f64 {
    let f = kelly_fraction(forecast);
    expected_log_growth(forecast, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_binary_favorable() {
        // 60% win, 2:1 payoff
        let kelly = kelly_fraction_binary(0.6, 1.0, -0.5);
        // f* = (0.6 * 2 - 0.4) / 2 = 0.4
        assert!((kelly - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_kelly_binary_unfavorable() {
        // 40% win, 1:1 payoff
        let kelly = kelly_fraction_binary(0.4, 1.0, -1.0);
        // f* = (0.4 * 1 - 0.6) / 1 = -0.2 (should not bet)
        assert!(kelly < 0.0);
    }

    #[test]
    fn test_kelly_continuous_positive_mean() {
        // Distribution with positive mean
        let forecast = ForecastDistribution::from_gaussian(0.05, 0.1, 10000);
        let kelly = kelly_fraction(&forecast);

        // Should be positive (bet on upside)
        assert!(kelly > 0.0);
    }

    #[test]
    fn test_kelly_continuous_negative_mean() {
        // Distribution with negative mean
        let forecast = ForecastDistribution::from_gaussian(-0.05, 0.1, 10000);
        let kelly = kelly_fraction(&forecast);

        // Should be negative (bet on downside)
        assert!(kelly < 0.0);
    }

    #[test]
    fn test_kelly_zero_mean() {
        // Fair game with zero mean
        let forecast = ForecastDistribution::from_gaussian(0.0, 0.1, 10000);
        let kelly = kelly_fraction(&forecast);

        // Should be close to zero
        assert!(kelly.abs() < 0.1);
    }

    #[test]
    fn test_fractional_kelly() {
        let forecast = ForecastDistribution::from_gaussian(0.02, 0.05, 10000);

        let full_kelly = kelly_fraction(&forecast);
        let half_kelly = fractional_kelly(&forecast, 0.5);

        assert!((half_kelly - full_kelly * 0.5).abs() < 0.001);
    }

    #[test]
    fn test_kelly_with_constraint() {
        let forecast = ForecastDistribution::from_gaussian(0.1, 0.05, 10000);

        let constrained = kelly_with_constraint(&forecast, 0.5, -0.5);

        assert!(constrained <= 0.5);
        assert!(constrained >= -0.5);
    }
}
