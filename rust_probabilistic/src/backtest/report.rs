//! Backtest report generation and analysis

use crate::backtest::engine::Trade;
#[allow(unused_imports)]
use crate::strategy::signal::SignalType;

/// Backtest results report
#[derive(Debug, Clone)]
pub struct BacktestReport {
    // Performance metrics
    /// Total return (fractional)
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized)
    pub sortino_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Calmar ratio (ann. return / max dd)
    pub calmar_ratio: f64,

    // Trade statistics
    /// Number of completed trades
    pub num_trades: usize,
    /// Win rate
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// Average winning trade return
    pub avg_win: f64,
    /// Average losing trade return
    pub avg_loss: f64,

    // Forecast quality
    /// Average CRPS
    pub avg_crps: f64,
    /// 90% interval coverage
    pub coverage_90: f64,
    /// Calibration error (|coverage - 0.90|)
    pub calibration_error: f64,

    // Raw data
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Period returns
    pub returns: Vec<f64>,
    /// Final capital
    pub final_capital: f64,
    /// Initial capital
    pub initial_capital: f64,
}

impl BacktestReport {
    /// Create report from backtest results
    pub fn from_results(
        initial_capital: f64,
        final_capital: f64,
        equity_curve: &[f64],
        returns: &[f64],
        trades: &[Trade],
        crps_values: &[f64],
        coverage_values: &[bool],
    ) -> Self {
        // Performance metrics
        let total_return = if initial_capital > 0.0 {
            final_capital / initial_capital - 1.0
        } else {
            0.0
        };

        let n_periods = returns.len();
        let periods_per_year = 252.0 * 24.0; // Assuming hourly data

        let annualized_return = if n_periods > 0 {
            (1.0 + total_return).powf(periods_per_year / n_periods as f64) - 1.0
        } else {
            0.0
        };

        let (sharpe_ratio, sortino_ratio) = compute_risk_adjusted_returns(returns, periods_per_year);
        let max_drawdown = compute_max_drawdown(equity_curve);
        let calmar_ratio = if max_drawdown > 0.0 {
            annualized_return / max_drawdown
        } else {
            0.0
        };

        // Trade statistics
        let completed_trades: Vec<&Trade> = trades.iter().filter(|t| t.is_closed()).collect();
        let num_trades = completed_trades.len();

        let (win_rate, profit_factor, avg_win, avg_loss) = compute_trade_stats(&completed_trades);

        // Forecast quality
        let avg_crps = if crps_values.is_empty() {
            0.0
        } else {
            crps_values.iter().sum::<f64>() / crps_values.len() as f64
        };

        let coverage_90 = if coverage_values.is_empty() {
            0.0
        } else {
            coverage_values.iter().filter(|&&v| v).count() as f64 / coverage_values.len() as f64
        };

        let calibration_error = (coverage_90 - 0.90).abs();

        Self {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            num_trades,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            avg_crps,
            coverage_90,
            calibration_error,
            equity_curve: equity_curve.to_vec(),
            returns: returns.to_vec(),
            final_capital,
            initial_capital,
        }
    }

    /// Get equity curve as percentage of initial capital
    pub fn equity_curve_pct(&self) -> Vec<f64> {
        self.equity_curve
            .iter()
            .map(|&e| e / self.initial_capital * 100.0)
            .collect()
    }

    /// Get drawdown series
    pub fn drawdown_series(&self) -> Vec<f64> {
        let mut max_equity = self.equity_curve[0];
        self.equity_curve
            .iter()
            .map(|&e| {
                if e > max_equity {
                    max_equity = e;
                }
                (max_equity - e) / max_equity
            })
            .collect()
    }

    /// Check if strategy is profitable
    pub fn is_profitable(&self) -> bool {
        self.total_return > 0.0
    }

    /// Check if strategy beats benchmark
    pub fn beats_benchmark(&self, benchmark_return: f64) -> bool {
        self.total_return > benchmark_return
    }

    /// Get summary statistics as string
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results Summary\n\
            ========================\n\
            \n\
            Performance:\n\
              Total Return:      {:>10.2}%\n\
              Annualized Return: {:>10.2}%\n\
              Sharpe Ratio:      {:>10.2}\n\
              Sortino Ratio:     {:>10.2}\n\
              Max Drawdown:      {:>10.2}%\n\
              Calmar Ratio:      {:>10.2}\n\
            \n\
            Trades:\n\
              Number of Trades:  {:>10}\n\
              Win Rate:          {:>10.2}%\n\
              Profit Factor:     {:>10.2}\n\
              Avg Win:           {:>10.4}\n\
              Avg Loss:          {:>10.4}\n\
            \n\
            Forecast Quality:\n\
              Avg CRPS:          {:>10.6}\n\
              90% Coverage:      {:>10.2}%\n\
              Calibration Error: {:>10.4}\n",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.sortino_ratio,
            self.max_drawdown * 100.0,
            self.calmar_ratio,
            self.num_trades,
            self.win_rate * 100.0,
            self.profit_factor,
            self.avg_win,
            self.avg_loss,
            self.avg_crps,
            self.coverage_90 * 100.0,
            self.calibration_error,
        )
    }
}

impl std::fmt::Display for BacktestReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Compute risk-adjusted return metrics
fn compute_risk_adjusted_returns(returns: &[f64], periods_per_year: f64) -> (f64, f64) {
    if returns.is_empty() {
        return (0.0, 0.0);
    }

    let n = returns.len() as f64;
    let mean_return = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / n;
    let std_return = variance.sqrt();

    let sharpe = if std_return > 0.0 {
        mean_return / std_return * periods_per_year.sqrt()
    } else {
        0.0
    };

    // Sortino (downside deviation)
    let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
    let downside_variance = if downside_returns.is_empty() {
        0.0
    } else {
        let n_down = downside_returns.len() as f64;
        downside_returns.iter().map(|r| r.powi(2)).sum::<f64>() / n_down
    };
    let downside_std = downside_variance.sqrt();

    let sortino = if downside_std > 0.0 {
        mean_return / downside_std * periods_per_year.sqrt()
    } else {
        0.0
    };

    (sharpe, sortino)
}

/// Compute maximum drawdown
fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }

    let mut max_equity = equity_curve[0];
    let mut max_dd = 0.0;

    for &equity in equity_curve {
        if equity > max_equity {
            max_equity = equity;
        }
        let dd = (max_equity - equity) / max_equity;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    max_dd
}

/// Compute trade statistics
fn compute_trade_stats(trades: &[&Trade]) -> (f64, f64, f64, f64) {
    if trades.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let pnls: Vec<f64> = trades
        .iter()
        .filter_map(|t| t.pnl)
        .collect();

    if pnls.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let wins: Vec<f64> = pnls.iter().filter(|&&p| p > 0.0).copied().collect();
    let losses: Vec<f64> = pnls.iter().filter(|&&p| p <= 0.0).copied().collect();

    let win_rate = wins.len() as f64 / pnls.len() as f64;

    let gross_profit: f64 = wins.iter().sum();
    let gross_loss: f64 = losses.iter().map(|l| l.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let avg_win = if wins.is_empty() {
        0.0
    } else {
        wins.iter().sum::<f64>() / wins.len() as f64
    };

    let avg_loss = if losses.is_empty() {
        0.0
    } else {
        losses.iter().sum::<f64>() / losses.len() as f64
    };

    (win_rate, profit_factor, avg_win, avg_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::forecast::ForecastDistribution;

    fn create_test_trades() -> Vec<Trade> {
        let forecast = ForecastDistribution::from_gaussian(0.01, 0.02, 100);

        let mut trades = Vec::new();

        // Winning trade
        let mut t1 = Trade::new(
            1000000,
            "BTCUSDT".to_string(),
            SignalType::Long,
            50000.0,
            0.1,
            &forecast,
        );
        t1.close(2000000, 51000.0);
        trades.push(t1);

        // Losing trade
        let mut t2 = Trade::new(
            3000000,
            "BTCUSDT".to_string(),
            SignalType::Long,
            50000.0,
            0.1,
            &forecast,
        );
        t2.close(4000000, 49500.0);
        trades.push(t2);

        trades
    }

    #[test]
    fn test_report_creation() {
        let trades = create_test_trades();
        let equity_curve = vec![10000.0, 10100.0, 10050.0, 10200.0];
        let returns = vec![0.01, -0.005, 0.015];
        let crps_values = vec![0.01, 0.012, 0.011];
        let coverage_values = vec![true, true, false];

        let report = BacktestReport::from_results(
            10000.0,
            10200.0,
            &equity_curve,
            &returns,
            &trades,
            &crps_values,
            &coverage_values,
        );

        assert_eq!(report.num_trades, 2);
        assert!(report.total_return > 0.0);
        assert!((report.win_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_max_drawdown() {
        let equity_curve = vec![100.0, 110.0, 100.0, 90.0, 95.0, 105.0];
        let max_dd = compute_max_drawdown(&equity_curve);

        // Max DD should be (110 - 90) / 110 = 0.182
        assert!((max_dd - 0.182).abs() < 0.01);
    }

    #[test]
    fn test_risk_adjusted_returns() {
        let returns = vec![0.01, 0.02, -0.01, 0.015, -0.005, 0.01];
        let (sharpe, sortino) = compute_risk_adjusted_returns(&returns, 252.0);

        assert!(sharpe > 0.0);
        assert!(sortino > sharpe); // Sortino should be higher when upside > downside
    }

    #[test]
    fn test_report_display() {
        let trades = create_test_trades();
        let report = BacktestReport::from_results(
            10000.0,
            10200.0,
            &[10000.0, 10200.0],
            &[0.02],
            &trades,
            &[0.01],
            &[true],
        );

        let summary = report.summary();
        assert!(summary.contains("Total Return"));
        assert!(summary.contains("Sharpe Ratio"));
    }
}
