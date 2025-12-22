//! Quantile regression model

use crate::distributions::forecast::ForecastDistribution;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Configuration for quantile regression
#[derive(Debug, Clone)]
pub struct QuantileConfig {
    /// Quantile levels to predict
    pub quantiles: Vec<f64>,
    /// Number of iterations for gradient descent
    pub max_iterations: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// L2 regularization
    pub l2_reg: f64,
}

impl Default for QuantileConfig {
    fn default() -> Self {
        Self {
            quantiles: vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            max_iterations: 1000,
            learning_rate: 0.01,
            l2_reg: 0.001,
        }
    }
}

/// Linear quantile regression model
#[derive(Debug, Clone)]
pub struct QuantileRegressor {
    /// Configuration
    pub config: QuantileConfig,
    /// Weights for each quantile [num_features + 1, num_quantiles]
    /// (includes bias term)
    weights: Option<Array2<f64>>,
    /// Feature means for normalization
    feature_means: Option<Array1<f64>>,
    /// Feature stds for normalization
    feature_stds: Option<Array1<f64>>,
}

impl QuantileRegressor {
    /// Create a new quantile regressor
    pub fn new(config: QuantileConfig) -> Self {
        Self {
            config,
            weights: None,
            feature_means: None,
            feature_stds: None,
        }
    }

    /// Fit the model to data
    pub fn fit(&mut self, features: &Array2<f64>, targets: &Array1<f64>) {
        let (n_samples, n_features) = features.dim();
        let n_quantiles = self.config.quantiles.len();

        // Compute normalization parameters
        let means = features.mean_axis(ndarray::Axis(0)).unwrap();
        let stds = features.std_axis(ndarray::Axis(0), 0.0);

        // Normalize features
        let mut normalized = features.clone();
        for j in 0..n_features {
            let std = if stds[j] > 1e-8 { stds[j] } else { 1.0 };
            for i in 0..n_samples {
                normalized[[i, j]] = (features[[i, j]] - means[j]) / std;
            }
        }

        // Initialize weights randomly
        let mut rng = rand::thread_rng();
        let mut weights = Array2::zeros((n_features + 1, n_quantiles));
        for i in 0..(n_features + 1) {
            for j in 0..n_quantiles {
                weights[[i, j]] = rng.gen_range(-0.1..0.1);
            }
        }

        // Gradient descent for each quantile
        for q_idx in 0..n_quantiles {
            let tau = self.config.quantiles[q_idx];

            for _iter in 0..self.config.max_iterations {
                let mut gradient = Array1::zeros(n_features + 1);

                for i in 0..n_samples {
                    // Compute prediction
                    let mut pred = weights[[n_features, q_idx]]; // bias
                    for j in 0..n_features {
                        pred += normalized[[i, j]] * weights[[j, q_idx]];
                    }

                    // Compute gradient of pinball loss
                    let error = targets[i] - pred;
                    let indicator = if error < 0.0 { 1.0 } else { 0.0 };
                    let grad_multiplier = tau - indicator;

                    // Update gradient
                    gradient[n_features] += grad_multiplier; // bias
                    for j in 0..n_features {
                        gradient[j] += grad_multiplier * normalized[[i, j]];
                    }
                }

                // Average gradient and add regularization
                for j in 0..=n_features {
                    gradient[j] = -gradient[j] / n_samples as f64;
                    if j < n_features {
                        gradient[j] += self.config.l2_reg * weights[[j, q_idx]];
                    }
                }

                // Update weights
                for j in 0..=n_features {
                    weights[[j, q_idx]] -= self.config.learning_rate * gradient[j];
                }
            }
        }

        self.weights = Some(weights);
        self.feature_means = Some(means);
        self.feature_stds = Some(stds);
    }

    /// Predict quantiles for new data
    pub fn predict_quantiles(&self, features: &Array2<f64>) -> Option<Array2<f64>> {
        let weights = self.weights.as_ref()?;
        let means = self.feature_means.as_ref()?;
        let stds = self.feature_stds.as_ref()?;

        let (n_samples, n_features) = features.dim();
        let n_quantiles = self.config.quantiles.len();

        // Normalize features
        let mut normalized = features.clone();
        for j in 0..n_features {
            let std = if stds[j] > 1e-8 { stds[j] } else { 1.0 };
            for i in 0..n_samples {
                normalized[[i, j]] = (features[[i, j]] - means[j]) / std;
            }
        }

        // Compute predictions
        let mut predictions = Array2::zeros((n_samples, n_quantiles));

        for i in 0..n_samples {
            for q_idx in 0..n_quantiles {
                let mut pred = weights[[n_features, q_idx]]; // bias
                for j in 0..n_features {
                    pred += normalized[[i, j]] * weights[[j, q_idx]];
                }
                predictions[[i, q_idx]] = pred;
            }
        }

        // Ensure monotonicity (sort quantiles)
        for i in 0..n_samples {
            let mut row: Vec<f64> = (0..n_quantiles).map(|j| predictions[[i, j]]).collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for j in 0..n_quantiles {
                predictions[[i, j]] = row[j];
            }
        }

        Some(predictions)
    }

    /// Predict and return a ForecastDistribution
    pub fn predict(&self, features: &Array2<f64>) -> Option<Vec<ForecastDistribution>> {
        let quantile_preds = self.predict_quantiles(features)?;
        let (n_samples, n_quantiles) = quantile_preds.dim();

        let mut forecasts = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Generate samples by interpolating quantiles
            let quantile_values: Vec<f64> = (0..n_quantiles).map(|j| quantile_preds[[i, j]]).collect();

            let mut samples = Vec::with_capacity(200);
            let mut rng = rand::thread_rng();

            for _ in 0..200 {
                let u: f64 = rng.gen();
                // Linear interpolation
                let sample = interpolate_quantile(u, &self.config.quantiles, &quantile_values);
                samples.push(sample);
            }

            forecasts.push(ForecastDistribution::from_samples(samples));
        }

        Some(forecasts)
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.weights.is_some()
    }
}

/// Interpolate to get value at given quantile level
fn interpolate_quantile(level: f64, quantiles: &[f64], values: &[f64]) -> f64 {
    if level <= quantiles[0] {
        return values[0];
    }
    if level >= quantiles[quantiles.len() - 1] {
        return values[values.len() - 1];
    }

    for i in 0..(quantiles.len() - 1) {
        if level >= quantiles[i] && level <= quantiles[i + 1] {
            let t = (level - quantiles[i]) / (quantiles[i + 1] - quantiles[i]);
            return values[i] + t * (values[i + 1] - values[i]);
        }
    }

    values[values.len() / 2]
}

/// Compute pinball loss for a single quantile
pub fn pinball_loss(predictions: &[f64], actuals: &[f64], tau: f64) -> f64 {
    let n = predictions.len();
    if n == 0 {
        return 0.0;
    }

    let total: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| {
            let error = actual - pred;
            if error >= 0.0 {
                tau * error
            } else {
                (tau - 1.0) * error
            }
        })
        .sum();

    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_regressor() {
        // Create synthetic data
        let n = 100;
        let mut rng = rand::thread_rng();

        let features = Array2::from_shape_fn((n, 3), |_| rng.gen::<f64>());
        let targets = Array1::from_shape_fn(n, |i| {
            features[[i, 0]] + 0.5 * features[[i, 1]] + rng.gen::<f64>() * 0.1
        });

        // Train model
        let mut model = QuantileRegressor::new(QuantileConfig::default());
        model.fit(&features, &targets);

        assert!(model.is_trained());

        // Predict
        let preds = model.predict_quantiles(&features).unwrap();
        assert_eq!(preds.nrows(), n);

        // Check monotonicity
        for i in 0..n {
            for j in 1..preds.ncols() {
                assert!(preds[[i, j]] >= preds[[i, j - 1]]);
            }
        }
    }

    #[test]
    fn test_pinball_loss() {
        let predictions = vec![1.0, 2.0, 3.0];
        let actuals = vec![1.1, 1.9, 3.2];

        let loss_50 = pinball_loss(&predictions, &actuals, 0.5);
        assert!(loss_50 > 0.0);

        // For tau=0.5, pinball loss equals 0.5 * MAE
        let mae: f64 = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (a - p).abs())
            .sum::<f64>()
            / 3.0;
        assert!((loss_50 - 0.5 * mae).abs() < 0.001);
    }

    #[test]
    fn test_interpolate_quantile() {
        let quantiles = vec![0.1, 0.5, 0.9];
        let values = vec![-1.0, 0.0, 1.0];

        assert!((interpolate_quantile(0.1, &quantiles, &values) - (-1.0)).abs() < 0.001);
        assert!((interpolate_quantile(0.5, &quantiles, &values) - 0.0).abs() < 0.001);
        assert!((interpolate_quantile(0.9, &quantiles, &values) - 1.0).abs() < 0.001);
        assert!((interpolate_quantile(0.3, &quantiles, &values) - (-0.5)).abs() < 0.001);
    }
}
