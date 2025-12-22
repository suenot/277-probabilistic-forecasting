//! Example: Proper Scoring Rules for Probabilistic Forecasts
//!
//! Demonstrates CRPS, Log Score, and calibration metrics for evaluating
//! probabilistic forecasts.
//!
//! Run with: cargo run --example scoring_rules

use probabilistic_forecasting::prelude::*;
use probabilistic_forecasting::scoring::crps::{compute_crps, compute_crps_gaussian, mean_crps, crps_skill_score};
use probabilistic_forecasting::scoring::log_score::compute_log_score;
use probabilistic_forecasting::scoring::calibration::{compute_pit, calibration_summary, coverage};
use rand::Rng;

fn main() {
    println!("Probabilistic Forecasting - Scoring Rules Example");
    println!("=================================================\n");

    // Part 1: CRPS Demonstration
    println!("{}", "=".repeat(60));
    println!("PART 1: CRPS (Continuous Ranked Probability Score)");
    println!("{}\n", "=".repeat(60));

    println!("CRPS measures the distance between the predicted distribution");
    println!("and the observed outcome. Lower CRPS is better.\n");

    // Perfect forecast
    let perfect_samples = vec![1.0; 1000];
    let perfect_forecast = ForecastDistribution::from_samples(perfect_samples);
    let crps_perfect = compute_crps(&perfect_forecast, 1.0);
    println!("Perfect forecast (all samples = observation):");
    println!("  CRPS = {:.6} (should be ~0)\n", crps_perfect);

    // Good forecast
    let good_forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);
    let crps_good = compute_crps(&good_forecast, 0.0);
    let crps_good_closed = compute_crps_gaussian(0.0, 1.0, 0.0);
    println!("Good forecast (N(0,1), observation=0):");
    println!("  CRPS (empirical)    = {:.6}", crps_good);
    println!("  CRPS (closed form)  = {:.6}\n", crps_good_closed);

    // Biased forecast
    let biased_forecast = ForecastDistribution::from_gaussian(2.0, 1.0, 10000);
    let crps_biased = compute_crps(&biased_forecast, 0.0);
    println!("Biased forecast (N(2,1), observation=0):");
    println!("  CRPS = {:.6} (worse than good forecast)\n", crps_biased);

    // Overconfident forecast
    let overconfident_forecast = ForecastDistribution::from_gaussian(0.0, 0.1, 10000);
    let crps_overconf_correct = compute_crps(&overconfident_forecast, 0.0);
    let crps_overconf_wrong = compute_crps(&overconfident_forecast, 0.5);
    println!("Overconfident forecast (N(0, 0.1)):");
    println!("  CRPS when correct (obs=0):   {:.6}", crps_overconf_correct);
    println!("  CRPS when wrong (obs=0.5):   {:.6} (heavily penalized!)\n", crps_overconf_wrong);

    // Part 2: Log Score
    println!("{}", "=".repeat(60));
    println!("PART 2: LOG SCORE");
    println!("{}\n", "=".repeat(60));

    println!("Log Score = log(p(y)) where p(y) is the predicted density");
    println!("at the observed value. Higher is better.\n");

    let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);

    for obs in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
        let log_score = compute_log_score(&forecast, *obs);
        println!("  Observation = {:>5.1}: Log Score = {:.4}", obs, log_score);
    }

    println!("\nNote: Log Score heavily penalizes observations in the tails.\n");

    // Part 3: Calibration
    println!("{}", "=".repeat(60));
    println!("PART 3: CALIBRATION");
    println!("{}\n", "=".repeat(60));

    println!("A well-calibrated forecast has PIT values uniformly distributed.");
    println!("PIT = F(y), where F is the predicted CDF and y is observed.\n");

    // Generate test forecasts and observations
    let mut rng = rand::thread_rng();
    let n_forecasts = 500;

    // Well-calibrated forecasts (observations from same distribution)
    let well_calibrated: Vec<(ForecastDistribution, f64)> = (0..n_forecasts)
        .map(|_| {
            let mean = rng.gen::<f64>() * 2.0 - 1.0;
            let std = 0.5 + rng.gen::<f64>() * 0.5;
            let forecast = ForecastDistribution::from_gaussian(mean, std, 200);
            // Observation from the forecast distribution
            let obs = mean + rng.gen::<f64>() * std * 2.0 - std;
            (forecast, obs)
        })
        .collect();

    let well_forecasts: Vec<_> = well_calibrated.iter().map(|(f, _)| f.clone()).collect();
    let well_obs: Vec<_> = well_calibrated.iter().map(|(_, o)| *o).collect();

    let _pit_well = compute_pit(&well_forecasts, &well_obs);
    let metrics_well = calibration_summary(&well_forecasts, &well_obs);

    println!("Well-Calibrated Forecasts:");
    println!("  PIT mean:     {:.4} (ideal: 0.5)", metrics_well.pit_mean);
    println!("  PIT std:      {:.4} (ideal: 0.289)", metrics_well.pit_std);
    println!("  Mean cal err: {:.4}", metrics_well.mean_calibration_error);
    println!("  90% coverage: {:.4}\n", metrics_well.coverage_90);

    // Overconfident forecasts (std too small)
    let overconfident: Vec<(ForecastDistribution, f64)> = (0..n_forecasts)
        .map(|_| {
            let mean = rng.gen::<f64>() * 2.0 - 1.0;
            let true_std = 1.0;
            let predicted_std = 0.3; // Too confident!
            let forecast = ForecastDistribution::from_gaussian(mean, predicted_std, 200);
            let obs = mean + (rng.gen::<f64>() - 0.5) * true_std * 2.0;
            (forecast, obs)
        })
        .collect();

    let over_forecasts: Vec<_> = overconfident.iter().map(|(f, _)| f.clone()).collect();
    let over_obs: Vec<_> = overconfident.iter().map(|(_, o)| *o).collect();

    let _pit_over = compute_pit(&over_forecasts, &over_obs);
    let metrics_over = calibration_summary(&over_forecasts, &over_obs);

    println!("Overconfident Forecasts:");
    println!("  PIT mean:     {:.4} (ideal: 0.5)", metrics_over.pit_mean);
    println!("  PIT std:      {:.4} (ideal: 0.289)", metrics_over.pit_std);
    println!("  Mean cal err: {:.4}", metrics_over.mean_calibration_error);
    println!("  90% coverage: {:.4}\n", metrics_over.coverage_90);

    // Part 4: Coverage Analysis
    println!("{}", "=".repeat(60));
    println!("PART 4: COVERAGE ANALYSIS");
    println!("{}\n", "=".repeat(60));

    println!("Coverage = fraction of observations within prediction interval.\n");

    println!("Well-Calibrated Forecasts:");
    for nominal in &[0.50, 0.80, 0.90, 0.95] {
        let actual = coverage(&well_forecasts, &well_obs, *nominal);
        let diff = (actual - nominal).abs();
        let status = if diff < 0.05 { "OK" } else { "MISS" };
        println!("  {:.0}% interval: {:.1}% actual ({}) ", nominal * 100.0, actual * 100.0, status);
    }

    println!("\nOverconfident Forecasts:");
    for nominal in &[0.50, 0.80, 0.90, 0.95] {
        let actual = coverage(&over_forecasts, &over_obs, *nominal);
        let diff = (actual - nominal).abs();
        let status = if diff < 0.05 { "OK" } else { "MISS" };
        println!("  {:.0}% interval: {:.1}% actual ({})", nominal * 100.0, actual * 100.0, status);
    }

    // Part 5: CRPS Skill Score
    println!("\n{}", "=".repeat(60));
    println!("PART 5: CRPS SKILL SCORE");
    println!("{}\n", "=".repeat(60));

    println!("CRPSS measures improvement over a baseline (e.g., climatology).");
    println!("CRPSS = 1 - CRPS_forecast / CRPS_baseline\n");

    // Baseline: unconditional distribution (climatology)
    let all_obs_mean = well_obs.iter().sum::<f64>() / well_obs.len() as f64;
    let all_obs_var = well_obs.iter().map(|o| (o - all_obs_mean).powi(2)).sum::<f64>() / well_obs.len() as f64;
    let baseline = ForecastDistribution::from_gaussian(all_obs_mean, all_obs_var.sqrt(), 500);

    let skill_score = crps_skill_score(&well_forecasts, &well_obs, &baseline);
    let mean_crps_val = mean_crps(&well_forecasts, &well_obs);

    println!("Baseline: N({:.3}, {:.3})", all_obs_mean, all_obs_var.sqrt());
    println!("Mean CRPS:        {:.6}", mean_crps_val);
    println!("CRPS Skill Score: {:.4}", skill_score);

    if skill_score > 0.0 {
        println!("\nForecasts are {:.1}% better than climatology!", skill_score * 100.0);
    } else {
        println!("\nForecasts are worse than climatology!");
    }

    // Part 6: Scoring Rule Comparison
    println!("\n{}", "=".repeat(60));
    println!("PART 6: SCORING RULE COMPARISON");
    println!("{}\n", "=".repeat(60));

    println!("Compare different scoring rules for the same forecasts:\n");

    let test_cases = vec![
        ("Perfect", ForecastDistribution::from_gaussian(1.0, 0.001, 1000), 1.0),
        ("Good", ForecastDistribution::from_gaussian(1.0, 0.5, 1000), 1.0),
        ("Wide", ForecastDistribution::from_gaussian(1.0, 2.0, 1000), 1.0),
        ("Biased", ForecastDistribution::from_gaussian(2.0, 0.5, 1000), 1.0),
        ("Very Biased", ForecastDistribution::from_gaussian(5.0, 0.5, 1000), 1.0),
    ];

    println!("{:>15} {:>12} {:>12} {:>12}", "Forecast", "CRPS", "Log Score", "Prob(+)");
    println!("{}", "-".repeat(55));

    for (name, forecast, obs) in test_cases {
        let crps = compute_crps(&forecast, obs);
        let log_score = compute_log_score(&forecast, obs);
        let prob_correct = forecast.prob_greater_than(obs - 0.5) - forecast.prob_greater_than(obs + 0.5);

        println!(
            "{:>15} {:>12.4} {:>12.4} {:>12.1}%",
            name, crps, log_score, prob_correct * 100.0
        );
    }

    println!("\n{}", "=".repeat(60));
    println!("Scoring rules example complete!");
    println!("{}", "=".repeat(60));
}
