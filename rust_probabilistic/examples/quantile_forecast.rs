//! Example: Quantile Regression Forecasting
//!
//! Demonstrates how to train and use a quantile regression model
//! for probabilistic forecasting.
//!
//! Run with: cargo run --example quantile_forecast

use probabilistic_forecasting::prelude::*;
use probabilistic_forecasting::models::quantile::{QuantileConfig, QuantileRegressor, pinball_loss};
use ndarray::{Array1, Array2};
use rand::Rng;

fn main() {
    println!("Probabilistic Forecasting - Quantile Regression Example");
    println!("========================================================\n");

    // Generate synthetic data
    println!("Generating synthetic data...");
    let (features, targets) = generate_synthetic_data(500);
    println!("Generated {} samples with {} features\n", features.nrows(), features.ncols());

    // Split into train/test
    let train_size = 400;
    let x_train = features.slice(ndarray::s![..train_size, ..]).to_owned();
    let y_train = targets.slice(ndarray::s![..train_size]).to_owned();
    let x_test = features.slice(ndarray::s![train_size.., ..]).to_owned();
    let y_test = targets.slice(ndarray::s![train_size..]).to_owned();

    println!("Train size: {}", train_size);
    println!("Test size:  {}\n", features.nrows() - train_size);

    // Configure model
    let config = QuantileConfig {
        quantiles: vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        max_iterations: 500,
        learning_rate: 0.01,
        l2_reg: 0.001,
    };

    println!("Quantile levels: {:?}\n", config.quantiles);

    // Train model
    println!("Training quantile regression model...");
    let mut model = QuantileRegressor::new(config.clone());
    model.fit(&x_train, &y_train);
    println!("Training complete!\n");

    // Make predictions
    println!("Generating predictions on test set...");
    let forecasts = model.predict(&x_test).expect("Prediction failed");

    // Evaluate forecasts
    println!("\n{}", "=".repeat(60));
    println!("FORECAST EVALUATION");
    println!("{}\n", "=".repeat(60));

    // CRPS
    let crps_values: Vec<f64> = forecasts
        .iter()
        .zip(y_test.iter())
        .map(|(f, &y)| compute_crps(f, y))
        .collect();
    let avg_crps = crps_values.iter().sum::<f64>() / crps_values.len() as f64;
    println!("Average CRPS:      {:.6}", avg_crps);

    // Calibration check (coverage)
    let coverage_90 = check_coverage(&forecasts, &y_test.to_vec(), 0.05, 0.95);
    let coverage_50 = check_coverage(&forecasts, &y_test.to_vec(), 0.25, 0.75);
    println!("90% CI Coverage:   {:.2}% (target: 90%)", coverage_90 * 100.0);
    println!("50% CI Coverage:   {:.2}% (target: 50%)", coverage_50 * 100.0);

    // Sharpness (interval width)
    let widths_90: Vec<f64> = forecasts
        .iter()
        .map(|f| f.quantile(0.95) - f.quantile(0.05))
        .collect();
    let avg_width_90 = widths_90.iter().sum::<f64>() / widths_90.len() as f64;
    println!("Avg 90% Interval:  {:.6}", avg_width_90);

    // Sample predictions
    println!("\n{}", "=".repeat(60));
    println!("SAMPLE PREDICTIONS");
    println!("{}\n", "=".repeat(60));

    println!("{:>8} {:>12} {:>12} {:>12} {:>12}",
             "Sample", "Actual", "Q(0.05)", "Q(0.50)", "Q(0.95)");
    println!("{}", "-".repeat(60));

    for i in 0..10.min(forecasts.len()) {
        let f = &forecasts[i];
        println!(
            "{:>8} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
            i + 1,
            y_test[i],
            f.quantile(0.05),
            f.quantile(0.50),
            f.quantile(0.95)
        );
    }

    // Pinball loss evaluation
    println!("\n{}", "=".repeat(60));
    println!("PINBALL LOSS BY QUANTILE");
    println!("{}\n", "=".repeat(60));

    let quantile_preds = model.predict_quantiles(&x_test).expect("Prediction failed");

    for (i, &tau) in config.quantiles.iter().enumerate() {
        let preds: Vec<f64> = (0..x_test.nrows())
            .map(|j| quantile_preds[[j, i]])
            .collect();
        let loss = pinball_loss(&preds, &y_test.to_vec(), tau);
        println!("  tau = {:.2}: {:.6}", tau, loss);
    }

    println!("\n{}", "=".repeat(60));
    println!("Example complete!");
    println!("{}", "=".repeat(60));
}

/// Generate synthetic data for demonstration
fn generate_synthetic_data(n: usize) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::thread_rng();

    let features = Array2::from_shape_fn((n, 5), |_| rng.gen::<f64>() * 2.0 - 1.0);

    let targets = Array1::from_shape_fn(n, |i| {
        let x = &features;
        // Linear combination with noise
        0.5 * x[[i, 0]] + 0.3 * x[[i, 1]] - 0.2 * x[[i, 2]] + rng.gen::<f64>() * 0.3 - 0.15
    });

    (features, targets)
}

/// Check empirical coverage of prediction intervals
fn check_coverage(
    forecasts: &[ForecastDistribution],
    actuals: &[f64],
    lower_q: f64,
    upper_q: f64,
) -> f64 {
    let in_interval: usize = forecasts
        .iter()
        .zip(actuals.iter())
        .filter(|(f, &a)| {
            let lower = f.quantile(lower_q);
            let upper = f.quantile(upper_q);
            a >= lower && a <= upper
        })
        .count();

    in_interval as f64 / forecasts.len() as f64
}
