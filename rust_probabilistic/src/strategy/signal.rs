//! Trading signals

use serde::{Deserialize, Serialize};

/// Signal type (direction)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    /// Long position (buy)
    Long,
    /// Short position (sell)
    Short,
    /// No position (hold/neutral)
    Hold,
}

impl SignalType {
    /// Get sign for position sizing
    pub fn sign(&self) -> f64 {
        match self {
            SignalType::Long => 1.0,
            SignalType::Short => -1.0,
            SignalType::Hold => 0.0,
        }
    }
}

/// Trading signal with probabilistic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    /// Signal direction
    pub signal_type: SignalType,
    /// Recommended position size (-1 to 1)
    pub position_size: f64,
    /// Confidence in the signal (0 to 1)
    pub confidence: f64,
    /// Expected return
    pub expected_return: f64,
    /// Value at Risk (95%)
    pub var_95: f64,
    /// Probability of positive return
    pub prob_positive: f64,
    /// Timestamp (optional)
    pub timestamp: Option<i64>,
    /// Symbol (optional)
    pub symbol: Option<String>,
}

impl Signal {
    /// Create a new hold signal
    pub fn hold() -> Self {
        Self {
            signal_type: SignalType::Hold,
            position_size: 0.0,
            confidence: 0.0,
            expected_return: 0.0,
            var_95: 0.0,
            prob_positive: 0.5,
            timestamp: None,
            symbol: None,
        }
    }

    /// Create a long signal
    pub fn long(
        position_size: f64,
        confidence: f64,
        expected_return: f64,
        var_95: f64,
        prob_positive: f64,
    ) -> Self {
        Self {
            signal_type: SignalType::Long,
            position_size: position_size.abs(),
            confidence,
            expected_return,
            var_95,
            prob_positive,
            timestamp: None,
            symbol: None,
        }
    }

    /// Create a short signal
    pub fn short(
        position_size: f64,
        confidence: f64,
        expected_return: f64,
        var_95: f64,
        prob_positive: f64,
    ) -> Self {
        Self {
            signal_type: SignalType::Short,
            position_size: -position_size.abs(),
            confidence,
            expected_return,
            var_95,
            prob_positive,
            timestamp: None,
            symbol: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: i64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: String) -> Self {
        self.symbol = Some(symbol);
        self
    }

    /// Check if this is an actionable signal (not hold)
    pub fn is_actionable(&self) -> bool {
        self.signal_type != SignalType::Hold
    }

    /// Get signed position size
    pub fn signed_size(&self) -> f64 {
        self.position_size
    }

    /// Get risk-reward ratio (expected return / VaR)
    pub fn risk_reward_ratio(&self) -> f64 {
        if self.var_95.abs() > 0.0 {
            self.expected_return / self.var_95.abs()
        } else {
            0.0
        }
    }
}

impl std::fmt::Display for Signal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let direction = match self.signal_type {
            SignalType::Long => "LONG",
            SignalType::Short => "SHORT",
            SignalType::Hold => "HOLD",
        };

        write!(
            f,
            "{} | Size: {:.2}% | Conf: {:.1}% | E[R]: {:.3}% | Prob(+): {:.1}%",
            direction,
            self.position_size * 100.0,
            self.confidence * 100.0,
            self.expected_return * 100.0,
            self.prob_positive * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_types() {
        assert_eq!(SignalType::Long.sign(), 1.0);
        assert_eq!(SignalType::Short.sign(), -1.0);
        assert_eq!(SignalType::Hold.sign(), 0.0);
    }

    #[test]
    fn test_signal_creation() {
        let signal = Signal::long(0.1, 0.8, 0.02, -0.03, 0.65);

        assert_eq!(signal.signal_type, SignalType::Long);
        assert_eq!(signal.position_size, 0.1);
        assert!(signal.is_actionable());
    }

    #[test]
    fn test_hold_signal() {
        let signal = Signal::hold();

        assert_eq!(signal.signal_type, SignalType::Hold);
        assert!(!signal.is_actionable());
    }

    #[test]
    fn test_signal_display() {
        let signal = Signal::long(0.15, 0.75, 0.025, -0.02, 0.70);
        let display = format!("{}", signal);

        assert!(display.contains("LONG"));
        assert!(display.contains("15.00%"));
    }
}
