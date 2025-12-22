//! Example: Kelly Criterion for Position Sizing
//!
//! Demonstrates how to use the Kelly Criterion with probabilistic forecasts
//! for optimal position sizing.
//!
//! Run with: cargo run --example kelly_sizing

use probabilistic_forecasting::prelude::*;
use probabilistic_forecasting::strategy::kelly::{
    kelly_fraction_binary,
    kelly_fraction,
    fractional_kelly,
    kelly_with_constraint,
    robust_kelly,
    kelly_growth_rate,
};

fn main() {
    println!("Probabilistic Forecasting - Kelly Criterion Example");
    println!("===================================================\n");

    // Part 1: Binary Kelly
    println!("{}", "=".repeat(60));
    println!("PART 1: BINARY KELLY CRITERION");
    println!("{}\n", "=".repeat(60));

    println!("The classic Kelly formula for binary outcomes:");
    println!("  f* = (p * b - q) / b");
    println!("  where p = prob(win), q = prob(loss), b = win/loss ratio\n");

    // Favorable bet: 60% win rate, 2:1 payoff
    let p = 0.60;
    let win = 0.10;   // 10% gain
    let loss = -0.05; // 5% loss

    let kelly = kelly_fraction_binary(p, win, loss);
    println!("Example 1: Favorable Bet");
    println!("  Probability of win:  {:.0}%", p * 100.0);
    println!("  Win return:          +{:.0}%", win * 100.0);
    println!("  Loss return:         {:.0}%", loss * 100.0);
    println!("  Kelly fraction:      {:.1}%\n", kelly * 100.0);

    // Unfavorable bet: 40% win rate, 1:1 payoff
    let p = 0.40;
    let win = 0.10;
    let loss = -0.10;

    let kelly = kelly_fraction_binary(p, win, loss);
    println!("Example 2: Unfavorable Bet");
    println!("  Probability of win:  {:.0}%", p * 100.0);
    println!("  Win return:          +{:.0}%", win * 100.0);
    println!("  Loss return:         {:.0}%", loss * 100.0);
    println!("  Kelly fraction:      {:.1}% (negative = don't bet)\n", kelly * 100.0);

    // Part 2: Continuous Kelly
    println!("{}", "=".repeat(60));
    println!("PART 2: CONTINUOUS KELLY CRITERION");
    println!("{}\n", "=".repeat(60));

    println!("For continuous distributions:");
    println!("  f* = argmax_f E[log(1 + f * R)]\n");

    // Strong positive forecast
    let strong_forecast = ForecastDistribution::from_gaussian(0.02, 0.03, 10000);
    let kelly_strong = kelly_fraction(&strong_forecast);
    let growth_strong = kelly_growth_rate(&strong_forecast);

    println!("Example 1: Strong Bullish Forecast");
    println!("  Mean return:     +{:.2}%", strong_forecast.mean * 100.0);
    println!("  Std deviation:   {:.2}%", strong_forecast.std * 100.0);
    println!("  Prob(+):         {:.1}%", strong_forecast.prob_greater_than(0.0) * 100.0);
    println!("  Kelly fraction:  {:.1}%", kelly_strong * 100.0);
    println!("  Expected growth: {:.4}% per period\n", growth_strong * 100.0);

    // Weak positive forecast
    let weak_forecast = ForecastDistribution::from_gaussian(0.005, 0.03, 10000);
    let kelly_weak = kelly_fraction(&weak_forecast);
    let growth_weak = kelly_growth_rate(&weak_forecast);

    println!("Example 2: Weak Bullish Forecast");
    println!("  Mean return:     +{:.2}%", weak_forecast.mean * 100.0);
    println!("  Std deviation:   {:.2}%", weak_forecast.std * 100.0);
    println!("  Prob(+):         {:.1}%", weak_forecast.prob_greater_than(0.0) * 100.0);
    println!("  Kelly fraction:  {:.1}%", kelly_weak * 100.0);
    println!("  Expected growth: {:.4}% per period\n", growth_weak * 100.0);

    // Bearish forecast
    let bearish_forecast = ForecastDistribution::from_gaussian(-0.015, 0.02, 10000);
    let kelly_bearish = kelly_fraction(&bearish_forecast);

    println!("Example 3: Bearish Forecast");
    println!("  Mean return:     {:.2}%", bearish_forecast.mean * 100.0);
    println!("  Std deviation:   {:.2}%", bearish_forecast.std * 100.0);
    println!("  Prob(+):         {:.1}%", bearish_forecast.prob_greater_than(0.0) * 100.0);
    println!("  Kelly fraction:  {:.1}% (negative = short)\n", kelly_bearish * 100.0);

    // Part 3: Fractional Kelly
    println!("{}", "=".repeat(60));
    println!("PART 3: FRACTIONAL KELLY (SAFETY)");
    println!("{}\n", "=".repeat(60));

    println!("Full Kelly maximizes growth but has high variance.");
    println!("Fractional Kelly reduces risk at cost of some growth.\n");

    let forecast = ForecastDistribution::from_gaussian(0.02, 0.04, 10000);
    let _full_kelly = kelly_fraction(&forecast);

    println!("Forecast: Mean=+{:.2}%, Std={:.2}%\n", forecast.mean * 100.0, forecast.std * 100.0);
    println!("{:>15} {:>15}", "Fraction", "Position Size");
    println!("{}", "-".repeat(35));

    for frac in &[1.0, 0.75, 0.5, 0.25, 0.1] {
        let size = fractional_kelly(&forecast, *frac);
        println!("{:>15.0}% {:>15.1}%", frac * 100.0, size * 100.0);
    }

    // Part 4: Kelly with Constraints
    println!("\n{}", "=".repeat(60));
    println!("PART 4: KELLY WITH CONSTRAINTS");
    println!("{}\n", "=".repeat(60));

    println!("In practice, we often constrain position sizes.\n");

    let aggressive_forecast = ForecastDistribution::from_gaussian(0.05, 0.03, 10000);
    let unconstrained = kelly_fraction(&aggressive_forecast);
    let constrained = kelly_with_constraint(&aggressive_forecast, 0.25, -0.25);

    println!("Aggressive forecast: Mean=+{:.1}%, Std={:.1}%",
             aggressive_forecast.mean * 100.0, aggressive_forecast.std * 100.0);
    println!("  Unconstrained Kelly: {:.1}%", unconstrained * 100.0);
    println!("  Constrained [-25%, +25%]: {:.1}%\n", constrained * 100.0);

    // Part 5: Robust Kelly
    println!("{}", "=".repeat(60));
    println!("PART 5: ROBUST KELLY (PARAMETER UNCERTAINTY)");
    println!("{}\n", "=".repeat(60));

    println!("When uncertain about forecast accuracy, be more conservative.\n");

    let forecast = ForecastDistribution::from_gaussian(0.02, 0.03, 10000);

    println!("Forecast: Mean=+{:.2}%, Std={:.2}%\n", forecast.mean * 100.0, forecast.std * 100.0);
    println!("{:>15} {:>15}", "Confidence", "Position Size");
    println!("{}", "-".repeat(35));

    for conf in &[1.0, 0.8, 0.6, 0.4, 0.2] {
        let size = robust_kelly(&forecast, *conf);
        println!("{:>15.0}% {:>15.1}%", conf * 100.0, size * 100.0);
    }

    // Part 6: Kelly Growth Analysis
    println!("\n{}", "=".repeat(60));
    println!("PART 6: GROWTH RATE ANALYSIS");
    println!("{}\n", "=".repeat(60));

    let forecast = ForecastDistribution::from_gaussian(0.015, 0.025, 10000);
    let kelly = kelly_fraction(&forecast);

    println!("Forecast: Mean=+{:.2}%, Std={:.2}%", forecast.mean * 100.0, forecast.std * 100.0);
    println!("Kelly fraction: {:.1}%\n", kelly * 100.0);

    println!("Growth rate at different position sizes:");
    println!("{:>15} {:>20} {:>15}", "Position", "Expected Growth", "vs Kelly");
    println!("{}", "-".repeat(55));

    for mult in &[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
        let pos = kelly * mult;
        // Simple growth rate approximation
        let growth = forecast.mean * pos - 0.5 * forecast.std.powi(2) * pos.powi(2);
        let kelly_growth = forecast.mean * kelly - 0.5 * forecast.std.powi(2) * kelly.powi(2);
        let vs_kelly = if kelly_growth != 0.0 { growth / kelly_growth * 100.0 } else { 0.0 };

        println!(
            "{:>15.1}% {:>20.4}% {:>15.0}%",
            pos * 100.0,
            growth * 100.0,
            vs_kelly
        );
    }

    println!("\nNote: Over-betting (>Kelly) reduces growth quickly!");
    println!("      Under-betting reduces growth linearly.\n");

    println!("{}", "=".repeat(60));
    println!("Kelly Criterion example complete!");
    println!("{}", "=".repeat(60));
}
