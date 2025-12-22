//! Backtesting engine for probabilistic forecasting strategies

use crate::api::types::Kline;
use crate::distributions::forecast::ForecastDistribution;
use crate::strategy::probabilistic::ProbabilisticStrategy;
use crate::strategy::signal::{Signal, SignalType};
use crate::backtest::report::BacktestReport;
use std::collections::HashMap;

/// Trade record
#[derive(Debug, Clone)]
pub struct Trade {
    /// Entry timestamp
    pub entry_time: i64,
    /// Exit timestamp
    pub exit_time: Option<i64>,
    /// Symbol traded
    pub symbol: String,
    /// Trade direction
    pub direction: SignalType,
    /// Entry price
    pub entry_price: f64,
    /// Exit price
    pub exit_price: Option<f64>,
    /// Position size (-1 to 1)
    pub position_size: f64,
    /// Realized P&L
    pub pnl: Option<f64>,
    /// Forecast mean at entry
    pub forecast_mean: f64,
    /// Forecast std at entry
    pub forecast_std: f64,
    /// Probability of positive return at entry
    pub forecast_prob_positive: f64,
}

impl Trade {
    /// Create a new trade
    pub fn new(
        entry_time: i64,
        symbol: String,
        direction: SignalType,
        entry_price: f64,
        position_size: f64,
        forecast: &ForecastDistribution,
    ) -> Self {
        Self {
            entry_time,
            exit_time: None,
            symbol,
            direction,
            entry_price,
            exit_price: None,
            position_size,
            pnl: None,
            forecast_mean: forecast.mean,
            forecast_std: forecast.std,
            forecast_prob_positive: forecast.prob_greater_than(0.0),
        }
    }

    /// Close the trade
    pub fn close(&mut self, exit_time: i64, exit_price: f64) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);
        self.pnl = Some(
            (exit_price - self.entry_price) / self.entry_price * self.position_size,
        );
    }

    /// Check if trade is closed
    pub fn is_closed(&self) -> bool {
        self.exit_time.is_some()
    }

    /// Get return (if closed)
    pub fn return_pct(&self) -> Option<f64> {
        self.exit_price.map(|exit| {
            let direction = if self.direction == SignalType::Long { 1.0 } else { -1.0 };
            direction * (exit - self.entry_price) / self.entry_price
        })
    }
}

/// Backtest configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Initial capital
    pub initial_capital: f64,
    /// Transaction cost per trade (fraction)
    pub transaction_cost: f64,
    /// Slippage in basis points
    pub slippage_bps: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001,
            slippage_bps: 5.0,
        }
    }
}

/// Backtesting engine
pub struct BacktestEngine {
    /// Strategy to backtest
    pub strategy: ProbabilisticStrategy,
    /// Configuration
    pub config: BacktestConfig,
    /// Current capital
    capital: f64,
    /// Position sizes by symbol
    positions: HashMap<String, f64>,
    /// Open trades
    open_trades: HashMap<String, Trade>,
    /// Completed trades
    trades: Vec<Trade>,
    /// Equity curve
    equity_curve: Vec<f64>,
    /// Returns
    returns: Vec<f64>,
    /// CRPS values for forecast evaluation
    crps_values: Vec<f64>,
    /// Coverage checks (90% interval)
    coverage_values: Vec<bool>,
}

impl BacktestEngine {
    /// Create a new backtest engine
    pub fn new(strategy: ProbabilisticStrategy, config: BacktestConfig) -> Self {
        let capital = config.initial_capital;
        Self {
            strategy,
            config,
            capital,
            positions: HashMap::new(),
            open_trades: HashMap::new(),
            trades: Vec::new(),
            equity_curve: vec![capital],
            returns: Vec::new(),
            crps_values: Vec::new(),
            coverage_values: Vec::new(),
        }
    }

    /// Run backtest on single symbol
    pub fn run(
        &mut self,
        klines: &[Kline],
        forecasts: &[ForecastDistribution],
        symbol: &str,
    ) -> BacktestReport {
        let n = klines.len().min(forecasts.len());

        for i in 0..n {
            let kline = &klines[i];
            let forecast = &forecasts[i];

            // Generate signal
            let signal = self.strategy.generate_signal(forecast);

            // Execute trades
            self.execute_signal(symbol, kline, &signal, forecast);

            // Mark to market
            if i > 0 {
                self.mark_to_market(symbol, klines[i - 1].close, kline.close);
            }

            // Track equity
            self.equity_curve.push(self.capital);

            // Evaluate forecast quality (if not last period)
            if i < n - 1 {
                let actual_return = klines[i + 1].log_return(kline.close);

                // CRPS
                let crps = self.compute_crps(forecast, actual_return);
                self.crps_values.push(crps);

                // Coverage check
                let (low, high) = forecast.interval_90();
                let in_interval = actual_return >= low && actual_return <= high;
                self.coverage_values.push(in_interval);
            }
        }

        // Close remaining positions
        if let Some(last_kline) = klines.last() {
            self.close_all_positions(symbol, last_kline.timestamp, last_kline.close);
        }

        // Generate report
        self.generate_report()
    }

    /// Execute a trading signal
    fn execute_signal(
        &mut self,
        symbol: &str,
        kline: &Kline,
        signal: &Signal,
        forecast: &ForecastDistribution,
    ) {
        let current_position = *self.positions.get(symbol).unwrap_or(&0.0);
        let target_position = signal.position_size;

        // Check if position change is significant
        if (target_position - current_position).abs() < 0.01 {
            return;
        }

        let slippage = self.config.slippage_bps / 10000.0;

        // Close existing position if direction changed
        if let Some(mut trade) = self.open_trades.remove(symbol) {
            if target_position.signum() != current_position.signum() || target_position == 0.0 {
                let exit_price = kline.close * (1.0 - slippage * current_position.signum());
                trade.close(kline.timestamp, exit_price);
                self.trades.push(trade);

                // Apply transaction cost
                self.capital -= self.config.transaction_cost * current_position.abs() * self.capital;
            }
        }

        // Open new position
        if signal.signal_type != SignalType::Hold && target_position.abs() > 0.01 {
            let entry_price = kline.close * (1.0 + slippage * target_position.signum());

            let trade = Trade::new(
                kline.timestamp,
                symbol.to_string(),
                signal.signal_type,
                entry_price,
                target_position,
                forecast,
            );

            self.open_trades.insert(symbol.to_string(), trade);

            // Apply transaction cost
            self.capital -= self.config.transaction_cost * target_position.abs() * self.capital;
        }

        self.positions.insert(symbol.to_string(), target_position);
    }

    /// Mark positions to market
    fn mark_to_market(&mut self, symbol: &str, prev_price: f64, current_price: f64) {
        if let Some(&position) = self.positions.get(symbol) {
            let ret = (current_price - prev_price) / prev_price;
            let pnl = position * self.capital * ret;

            let prev_capital = self.capital;
            self.capital += pnl;

            if prev_capital > 0.0 {
                self.returns.push(pnl / prev_capital);
            }
        }
    }

    /// Close all open positions
    fn close_all_positions(&mut self, symbol: &str, timestamp: i64, price: f64) {
        if let Some(mut trade) = self.open_trades.remove(symbol) {
            trade.close(timestamp, price);
            self.trades.push(trade);
        }
        self.positions.remove(symbol);
    }

    /// Compute CRPS for forecast evaluation
    fn compute_crps(&self, forecast: &ForecastDistribution, actual: f64) -> f64 {
        let samples = &forecast.samples;
        let n = samples.len() as f64;

        // E|X - y|
        let abs_diff: f64 = samples.iter().map(|&x| (x - actual).abs()).sum::<f64>() / n;

        // E|X - X'| using sorted samples
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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

    /// Generate backtest report
    fn generate_report(&self) -> BacktestReport {
        BacktestReport::from_results(
            self.config.initial_capital,
            self.capital,
            &self.equity_curve,
            &self.returns,
            &self.trades,
            &self.crps_values,
            &self.coverage_values,
        )
    }

    /// Get current capital
    pub fn current_capital(&self) -> f64 {
        self.capital
    }

    /// Get equity curve
    pub fn equity_curve(&self) -> &[f64] {
        &self.equity_curve
    }

    /// Get completed trades
    pub fn trades(&self) -> &[Trade] {
        &self.trades
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::probabilistic::StrategyConfig;

    fn create_test_klines(n: usize, base_price: f64) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let price = base_price * (1.0 + (i as f64 * 0.001));
                Kline::new(
                    i as i64 * 3600000,
                    price,
                    price * 1.01,
                    price * 0.99,
                    price * 1.005,
                    1000.0,
                )
            })
            .collect()
    }

    fn create_test_forecasts(n: usize) -> Vec<ForecastDistribution> {
        (0..n)
            .map(|_| ForecastDistribution::from_gaussian(0.005, 0.02, 200))
            .collect()
    }

    #[test]
    fn test_backtest_engine() {
        let strategy = ProbabilisticStrategy::new(StrategyConfig::default());
        let config = BacktestConfig::default();
        let mut engine = BacktestEngine::new(strategy, config);

        let klines = create_test_klines(100, 50000.0);
        let forecasts = create_test_forecasts(100);

        let report = engine.run(&klines, &forecasts, "BTCUSDT");

        assert!(report.num_trades >= 0);
        assert!(report.total_return.is_finite());
    }

    #[test]
    fn test_trade_creation() {
        let forecast = ForecastDistribution::from_gaussian(0.01, 0.02, 200);
        let trade = Trade::new(
            1000000,
            "BTCUSDT".to_string(),
            SignalType::Long,
            50000.0,
            0.1,
            &forecast,
        );

        assert_eq!(trade.symbol, "BTCUSDT");
        assert!(!trade.is_closed());
    }

    #[test]
    fn test_trade_close() {
        let forecast = ForecastDistribution::from_gaussian(0.01, 0.02, 200);
        let mut trade = Trade::new(
            1000000,
            "BTCUSDT".to_string(),
            SignalType::Long,
            50000.0,
            0.1,
            &forecast,
        );

        trade.close(2000000, 51000.0);

        assert!(trade.is_closed());
        assert!(trade.pnl.is_some());
        assert!(trade.return_pct().unwrap() > 0.0);
    }
}
