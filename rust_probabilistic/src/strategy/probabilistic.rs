//! Probabilistic trading strategy

use crate::distributions::forecast::ForecastDistribution;
use crate::strategy::kelly::kelly_fraction;
use crate::strategy::signal::{Signal, SignalType};

/// Strategy configuration
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    /// Minimum probability threshold for signal
    pub confidence_threshold: f64,
    /// Minimum expected return to trade
    pub min_expected_return: f64,
    /// Fraction of Kelly to use (e.g., 0.5 for half-Kelly)
    pub kelly_fraction: f64,
    /// Maximum position size (as fraction of portfolio)
    pub max_position_size: f64,
    /// Maximum VaR exposure
    pub var_limit: f64,
    /// Transaction cost per trade
    pub transaction_cost: f64,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.60,
            min_expected_return: 0.005,
            kelly_fraction: 0.5,
            max_position_size: 0.25,
            var_limit: 0.02,
            transaction_cost: 0.001,
        }
    }
}

/// Probabilistic trading strategy
#[derive(Debug, Clone)]
pub struct ProbabilisticStrategy {
    pub config: StrategyConfig,
}

impl ProbabilisticStrategy {
    /// Create a new strategy with default config
    pub fn new(config: StrategyConfig) -> Self {
        Self { config }
    }

    /// Generate trading signal from forecast distribution
    pub fn generate_signal(&self, forecast: &ForecastDistribution) -> Signal {
        // Compute key statistics
        let expected_return = forecast.mean;
        let prob_positive = forecast.prob_greater_than(0.0);
        let var_95 = forecast.var(0.95);

        // Compute Kelly-optimal position
        let full_kelly = kelly_fraction(forecast);
        let target_position = full_kelly * self.config.kelly_fraction;

        // Determine signal direction
        let (signal_type, mut position_size) = if prob_positive > self.config.confidence_threshold
            && expected_return > self.config.min_expected_return
        {
            (SignalType::Long, target_position.abs())
        } else if prob_positive < (1.0 - self.config.confidence_threshold)
            && expected_return < -self.config.min_expected_return
        {
            (SignalType::Short, -target_position.abs())
        } else {
            (SignalType::Hold, 0.0)
        };

        // Apply position constraints
        if signal_type != SignalType::Hold {
            // Cap at max position
            position_size = position_size.abs().min(self.config.max_position_size);

            // VaR constraint
            if var_95 < 0.0 {
                let var_constrained = self.config.var_limit / var_95.abs();
                position_size = position_size.min(var_constrained);
            }

            // Apply sign
            if signal_type == SignalType::Short {
                position_size = -position_size;
            }
        }

        // Confidence metric
        let confidence = (prob_positive - 0.5).abs() * 2.0;

        Signal {
            signal_type,
            position_size,
            confidence,
            expected_return,
            var_95,
            prob_positive,
            timestamp: None,
            symbol: None,
        }
    }

    /// Generate signals for multiple forecasts
    pub fn generate_signals(&self, forecasts: &[ForecastDistribution]) -> Vec<Signal> {
        forecasts
            .iter()
            .map(|f| self.generate_signal(f))
            .collect()
    }

    /// Compute expected P&L for a position
    pub fn expected_pnl(&self, forecast: &ForecastDistribution, position_size: f64) -> ExpectedPnL {
        let gross_pnl: Vec<f64> = forecast
            .samples
            .iter()
            .map(|&r| position_size * r)
            .collect();

        let net_pnl: Vec<f64> = gross_pnl
            .iter()
            .map(|&p| p - 2.0 * self.config.transaction_cost * position_size.abs())
            .collect();

        let n = net_pnl.len() as f64;
        let mean_pnl = net_pnl.iter().sum::<f64>() / n;
        let variance = net_pnl.iter().map(|&p| (p - mean_pnl).powi(2)).sum::<f64>() / n;
        let std_pnl = variance.sqrt();

        let mut sorted_pnl = net_pnl.clone();
        sorted_pnl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_95 = sorted_pnl[(n * 0.05) as usize];

        let tail: Vec<f64> = sorted_pnl.iter().filter(|&&p| p <= var_95).copied().collect();
        let cvar_95 = if tail.is_empty() {
            var_95
        } else {
            tail.iter().sum::<f64>() / tail.len() as f64
        };

        let prob_profit = net_pnl.iter().filter(|&&p| p > 0.0).count() as f64 / n;

        let profits: Vec<f64> = net_pnl.iter().filter(|&&p| p > 0.0).copied().collect();
        let losses: Vec<f64> = net_pnl.iter().filter(|&&p| p <= 0.0).copied().collect();

        let expected_profit = if profits.is_empty() {
            0.0
        } else {
            profits.iter().sum::<f64>() / profits.len() as f64
        };

        let expected_loss = if losses.is_empty() {
            0.0
        } else {
            losses.iter().sum::<f64>() / losses.len() as f64
        };

        ExpectedPnL {
            expected_pnl: mean_pnl,
            pnl_std: std_pnl,
            pnl_var_95: var_95,
            pnl_cvar_95: cvar_95,
            prob_profit,
            expected_profit_if_profit: expected_profit,
            expected_loss_if_loss: expected_loss,
        }
    }
}

/// Expected P&L statistics
#[derive(Debug, Clone)]
pub struct ExpectedPnL {
    pub expected_pnl: f64,
    pub pnl_std: f64,
    pub pnl_var_95: f64,
    pub pnl_cvar_95: f64,
    pub prob_profit: f64,
    pub expected_profit_if_profit: f64,
    pub expected_loss_if_loss: f64,
}

impl std::fmt::Display for ExpectedPnL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Expected P&L Statistics:")?;
        writeln!(f, "  Expected P&L:    {:.4}", self.expected_pnl)?;
        writeln!(f, "  P&L Std Dev:     {:.4}", self.pnl_std)?;
        writeln!(f, "  VaR (95%):       {:.4}", self.pnl_var_95)?;
        writeln!(f, "  CVaR (95%):      {:.4}", self.pnl_cvar_95)?;
        writeln!(f, "  Prob Profit:     {:.2}%", self.prob_profit * 100.0)?;
        writeln!(f, "  Exp Profit|Win:  {:.4}", self.expected_profit_if_profit)?;
        write!(f, "  Exp Loss|Loss:   {:.4}", self.expected_loss_if_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_long_signal() {
        let strategy = ProbabilisticStrategy::new(StrategyConfig::default());

        // Forecast with positive mean and high probability of positive return
        let forecast = ForecastDistribution::from_gaussian(0.02, 0.01, 10000);
        let signal = strategy.generate_signal(&forecast);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert!(signal.position_size > 0.0);
        assert!(signal.prob_positive > 0.5);
    }

    #[test]
    fn test_generate_short_signal() {
        let strategy = ProbabilisticStrategy::new(StrategyConfig::default());

        // Forecast with negative mean
        let forecast = ForecastDistribution::from_gaussian(-0.02, 0.01, 10000);
        let signal = strategy.generate_signal(&forecast);

        assert_eq!(signal.signal_type, SignalType::Short);
        assert!(signal.position_size < 0.0);
        assert!(signal.prob_positive < 0.5);
    }

    #[test]
    fn test_generate_hold_signal() {
        let strategy = ProbabilisticStrategy::new(StrategyConfig::default());

        // Uncertain forecast (near 50-50)
        let forecast = ForecastDistribution::from_gaussian(0.001, 0.02, 10000);
        let signal = strategy.generate_signal(&forecast);

        // Should be hold due to low confidence
        assert_eq!(signal.signal_type, SignalType::Hold);
        assert_eq!(signal.position_size, 0.0);
    }

    #[test]
    fn test_position_size_constraint() {
        let config = StrategyConfig {
            max_position_size: 0.10, // 10% max
            ..Default::default()
        };
        let strategy = ProbabilisticStrategy::new(config);

        // Strong signal that would suggest large position
        let forecast = ForecastDistribution::from_gaussian(0.1, 0.01, 10000);
        let signal = strategy.generate_signal(&forecast);

        assert!(signal.position_size.abs() <= 0.10);
    }

    #[test]
    fn test_expected_pnl() {
        let strategy = ProbabilisticStrategy::new(StrategyConfig::default());
        let forecast = ForecastDistribution::from_gaussian(0.01, 0.02, 10000);

        let pnl = strategy.expected_pnl(&forecast, 0.1);

        // Expected P&L should be roughly 0.1 * 0.01 = 0.001 minus costs
        assert!(pnl.expected_pnl > 0.0);
        assert!(pnl.prob_profit > 0.5);
    }
}
