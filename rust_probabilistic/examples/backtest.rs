//! Example: Backtesting a Probabilistic Trading Strategy
//!
//! Demonstrates a complete backtest workflow using probabilistic forecasts.
//!
//! Run with: cargo run --example backtest

use probabilistic_forecasting::prelude::*;
use probabilistic_forecasting::backtest::{BacktestConfig, BacktestEngine};
use probabilistic_forecasting::api::types::Kline;
use rand::Rng;

fn main() {
    println!("Probabilistic Forecasting - Backtest Example");
    println!("============================================\n");

    // Generate synthetic market data
    println!("Generating synthetic market data...");
    let klines = generate_synthetic_klines(500, 50000.0);
    println!("Generated {} klines\n", klines.len());

    // Generate forecasts
    println!("Generating probabilistic forecasts...");
    let forecasts = generate_forecasts(&klines);
    println!("Generated {} forecasts\n", forecasts.len());

    // Configure strategy
    let strategy_config = StrategyConfig {
        confidence_threshold: 0.60,
        min_expected_return: 0.005,
        kelly_fraction: 0.25,  // Use quarter-Kelly for safety
        max_position_size: 0.15,
        var_limit: 0.02,
        transaction_cost: 0.001,
    };

    println!("Strategy Configuration:");
    println!("  Confidence threshold: {:.0}%", strategy_config.confidence_threshold * 100.0);
    println!("  Min expected return:  {:.2}%", strategy_config.min_expected_return * 100.0);
    println!("  Kelly fraction:       {:.0}%", strategy_config.kelly_fraction * 100.0);
    println!("  Max position size:    {:.0}%", strategy_config.max_position_size * 100.0);
    println!("  VaR limit:            {:.2}%", strategy_config.var_limit * 100.0);
    println!("  Transaction cost:     {:.2}%\n", strategy_config.transaction_cost * 100.0);

    // Configure backtest
    let backtest_config = BacktestConfig {
        initial_capital: 10000.0,
        transaction_cost: 0.001,
        slippage_bps: 5.0,
    };

    println!("Backtest Configuration:");
    println!("  Initial capital:  ${:.2}", backtest_config.initial_capital);
    println!("  Transaction cost: {:.2}%", backtest_config.transaction_cost * 100.0);
    println!("  Slippage:         {} bps\n", backtest_config.slippage_bps);

    // Create strategy and backtest engine
    let strategy = ProbabilisticStrategy::new(strategy_config);
    let mut engine = BacktestEngine::new(strategy, backtest_config);

    // Run backtest
    println!("Running backtest...\n");
    let report = engine.run(&klines, &forecasts, "BTCUSDT");

    // Print results
    println!("{}", report);

    // Additional analysis
    println!("\n{}", "=".repeat(60));
    println!("ADDITIONAL ANALYSIS");
    println!("{}\n", "=".repeat(60));

    // Equity curve statistics
    let equity = engine.equity_curve();
    let min_equity = equity.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_equity = equity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("Equity Curve:");
    println!("  Starting:  ${:.2}", equity[0]);
    println!("  Final:     ${:.2}", equity[equity.len() - 1]);
    println!("  Minimum:   ${:.2}", min_equity);
    println!("  Maximum:   ${:.2}", max_equity);

    // Trade analysis
    let trades = engine.trades();
    if !trades.is_empty() {
        println!("\nTrade Analysis:");

        let long_trades: Vec<_> = trades.iter()
            .filter(|t| t.direction == SignalType::Long)
            .collect();
        let short_trades: Vec<_> = trades.iter()
            .filter(|t| t.direction == SignalType::Short)
            .collect();

        println!("  Long trades:  {}", long_trades.len());
        println!("  Short trades: {}", short_trades.len());

        // Analyze correlation between forecast confidence and outcome
        let mut high_conf_wins = 0;
        let mut high_conf_total = 0;
        let mut low_conf_wins = 0;
        let mut low_conf_total = 0;

        for trade in trades {
            if let Some(pnl) = trade.pnl {
                let conf = (trade.forecast_prob_positive - 0.5).abs() * 2.0;
                if conf > 0.3 {
                    high_conf_total += 1;
                    if pnl > 0.0 { high_conf_wins += 1; }
                } else {
                    low_conf_total += 1;
                    if pnl > 0.0 { low_conf_wins += 1; }
                }
            }
        }

        if high_conf_total > 0 {
            println!("\n  High confidence trades (>65%):");
            println!("    Count:    {}", high_conf_total);
            println!("    Win rate: {:.1}%", high_conf_wins as f64 / high_conf_total as f64 * 100.0);
        }

        if low_conf_total > 0 {
            println!("\n  Low confidence trades (<=65%):");
            println!("    Count:    {}", low_conf_total);
            println!("    Win rate: {:.1}%", low_conf_wins as f64 / low_conf_total as f64 * 100.0);
        }

        // Sample trades
        println!("\n\nSample Trades (first 10):");
        println!("{}", "-".repeat(90));
        println!(
            "{:>12} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
            "Entry Time", "Direction", "Size", "Entry Price", "Exit Price", "P&L", "Prob(+)"
        );
        println!("{}", "-".repeat(90));

        for trade in trades.iter().take(10) {
            let dir_str = match trade.direction {
                SignalType::Long => "LONG",
                SignalType::Short => "SHORT",
                SignalType::Hold => "HOLD",
            };
            let exit_price_str = trade.exit_price
                .map(|p| format!("{:.2}", p))
                .unwrap_or_else(|| "---".to_string());
            let pnl_str = trade.pnl
                .map(|p| format!("{:.4}", p))
                .unwrap_or_else(|| "---".to_string());

            println!(
                "{:>12} {:>10} {:>10.2}% {:>12.2} {:>12} {:>10} {:>10.1}%",
                trade.entry_time,
                dir_str,
                trade.position_size * 100.0,
                trade.entry_price,
                exit_price_str,
                pnl_str,
                trade.forecast_prob_positive * 100.0
            );
        }
    }

    println!("\n{}", "=".repeat(60));
    println!("Backtest complete!");
    println!("{}", "=".repeat(60));
}

/// Generate synthetic kline data
fn generate_synthetic_klines(n: usize, initial_price: f64) -> Vec<Kline> {
    let mut rng = rand::thread_rng();
    let mut price = initial_price;
    let mut klines = Vec::with_capacity(n);

    for i in 0..n {
        // Random walk with slight drift
        let return_pct = rng.gen::<f64>() * 0.04 - 0.02 + 0.0002; // Slight positive drift
        price *= 1.0 + return_pct;

        let volatility = 0.01 + rng.gen::<f64>() * 0.01;
        let high = price * (1.0 + volatility);
        let low = price * (1.0 - volatility);
        let open = price * (1.0 + (rng.gen::<f64>() - 0.5) * volatility);
        let close = price;

        klines.push(Kline::new(
            i as i64 * 3600000, // Hourly timestamps
            open,
            high,
            low,
            close,
            1000.0 + rng.gen::<f64>() * 500.0,
        ));
    }

    klines
}

/// Generate forecasts based on kline data
fn generate_forecasts(klines: &[Kline]) -> Vec<ForecastDistribution> {
    let mut rng = rand::thread_rng();
    let mut forecasts = Vec::with_capacity(klines.len());

    for i in 0..klines.len() {
        // Simple momentum-based forecast
        let momentum = if i > 10 {
            let recent_return = klines[i].log_return(klines[i - 10].close) / 10.0;
            recent_return * 0.5 // Mean reversion factor
        } else {
            0.0
        };

        // Add some noise to make it realistic
        let forecast_mean = momentum + (rng.gen::<f64>() - 0.5) * 0.005;
        let forecast_std = 0.015 + rng.gen::<f64>() * 0.01;

        forecasts.push(ForecastDistribution::from_gaussian(
            forecast_mean,
            forecast_std,
            200,
        ));
    }

    forecasts
}
