"""
Trading Strategy with Probabilistic Forecasts
=============================================

Implements risk-aware trading strategies using probabilistic
forecasts and the Kelly criterion.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG = 1
    SHORT = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """Trading signal with probabilistic information."""
    signal_type: SignalType
    position_size: float
    confidence: float
    expected_return: float
    var_95: float
    prob_positive: float
    timestamp: Optional[str] = None


@dataclass
class ForecastDistribution:
    """Container for forecast distribution information."""
    samples: np.ndarray  # [n_samples]
    mean: float
    std: float
    quantiles: Dict[float, float]  # {0.05: value, 0.5: value, ...}

    @classmethod
    def from_samples(cls, samples: np.ndarray):
        """Create from samples."""
        quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        quantiles = {q: np.percentile(samples, q * 100) for q in quantile_levels}
        return cls(
            samples=samples,
            mean=samples.mean(),
            std=samples.std(),
            quantiles=quantiles
        )

    def prob_greater_than(self, threshold: float) -> float:
        """Probability that value exceeds threshold."""
        return (self.samples > threshold).mean()

    def prob_less_than(self, threshold: float) -> float:
        """Probability that value is below threshold."""
        return (self.samples < threshold).mean()

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk at given confidence level."""
        return np.percentile(self.samples, (1 - confidence) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        var_level = self.var(confidence)
        return self.samples[self.samples <= var_level].mean()


def kelly_fraction_continuous(forecast: ForecastDistribution) -> float:
    """
    Compute Kelly fraction for continuous distribution.

    f* = argmax_f E[log(1 + f * R)]

    Uses numerical optimization over samples.

    Args:
        forecast: Forecast distribution

    Returns:
        Optimal Kelly fraction
    """
    from scipy.optimize import minimize_scalar

    samples = forecast.samples

    def neg_expected_log_return(f):
        # Clip to prevent log(0) or log(negative)
        returns = np.clip(1 + f * samples, 1e-10, None)
        return -np.mean(np.log(returns))

    # Find optimal fraction in [-2, 2] range
    result = minimize_scalar(
        neg_expected_log_return,
        bounds=(-2, 2),
        method='bounded'
    )

    return result.x


def kelly_fraction_simple(
    prob_win: float,
    win_return: float,
    loss_return: float
) -> float:
    """
    Simple Kelly fraction for binary outcomes.

    f* = (p * b - q) / b

    where:
        p = probability of win
        q = 1 - p = probability of loss
        b = win/loss ratio

    Args:
        prob_win: Probability of winning
        win_return: Return if win (e.g., 0.1 for 10%)
        loss_return: Return if loss (e.g., -0.05 for -5%, should be negative)

    Returns:
        Kelly fraction (can be negative for short)
    """
    if loss_return >= 0:
        return 0  # No risk of loss

    q = 1 - prob_win
    b = win_return / abs(loss_return)

    kelly = (prob_win * b - q) / b

    return kelly


class ProbabilisticStrategy:
    """
    Trading strategy using probabilistic forecasts.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        min_expected_return: float = 0.005,
        kelly_fraction: float = 0.5,
        max_position_size: float = 0.25,
        var_limit: float = 0.02,
        transaction_cost: float = 0.001
    ):
        """
        Initialize strategy.

        Args:
            confidence_threshold: Minimum probability for signal
            min_expected_return: Minimum expected return to trade
            kelly_fraction: Fraction of Kelly to use (e.g., 0.5 for half-Kelly)
            max_position_size: Maximum position as fraction of portfolio
            var_limit: Maximum VaR(95%) exposure
            transaction_cost: Cost per trade as fraction
        """
        self.confidence_threshold = confidence_threshold
        self.min_expected_return = min_expected_return
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        self.var_limit = var_limit
        self.transaction_cost = transaction_cost

    def generate_signal(
        self,
        forecast: ForecastDistribution,
        current_position: float = 0.0
    ) -> TradingSignal:
        """
        Generate trading signal from probabilistic forecast.

        Args:
            forecast: Forecast distribution
            current_position: Current position (-1 to 1)

        Returns:
            Trading signal
        """
        # Compute key statistics
        expected_return = forecast.mean
        prob_positive = forecast.prob_greater_than(0)
        var_95 = forecast.var(0.95)

        # Compute Kelly-optimal position
        full_kelly = kelly_fraction_continuous(forecast)
        target_position = full_kelly * self.kelly_fraction

        # Determine signal direction
        if prob_positive > self.confidence_threshold and expected_return > self.min_expected_return:
            signal_type = SignalType.LONG
        elif prob_positive < (1 - self.confidence_threshold) and expected_return < -self.min_expected_return:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.HOLD

        # Adjust position size
        if signal_type == SignalType.HOLD:
            position_size = 0.0
        else:
            # Start with Kelly-optimal
            position_size = abs(target_position)

            # Cap at max position
            position_size = min(position_size, self.max_position_size)

            # VaR constraint
            if var_95 < 0:
                var_constrained = self.var_limit / abs(var_95)
                position_size = min(position_size, var_constrained)

            # Adjust sign for direction
            if signal_type == SignalType.SHORT:
                position_size = -position_size

        # Confidence metric
        confidence = abs(prob_positive - 0.5) * 2

        return TradingSignal(
            signal_type=signal_type,
            position_size=position_size,
            confidence=confidence,
            expected_return=expected_return,
            var_95=var_95,
            prob_positive=prob_positive
        )

    def compute_expected_pnl(
        self,
        forecast: ForecastDistribution,
        position_size: float
    ) -> Dict[str, float]:
        """
        Compute expected P&L statistics.

        Args:
            forecast: Forecast distribution
            position_size: Position size (negative for short)

        Returns:
            Dictionary with P&L statistics
        """
        # Gross P&L
        gross_pnl = position_size * forecast.samples

        # Net of transaction costs (entry + exit)
        net_pnl = gross_pnl - 2 * self.transaction_cost * abs(position_size)

        return {
            'expected_pnl': net_pnl.mean(),
            'pnl_std': net_pnl.std(),
            'pnl_var_95': np.percentile(net_pnl, 5),
            'pnl_cvar_95': net_pnl[net_pnl <= np.percentile(net_pnl, 5)].mean(),
            'prob_profit': (net_pnl > 0).mean(),
            'expected_profit_if_profit': net_pnl[net_pnl > 0].mean() if (net_pnl > 0).any() else 0,
            'expected_loss_if_loss': net_pnl[net_pnl <= 0].mean() if (net_pnl <= 0).any() else 0,
        }


class PortfolioManager:
    """
    Manages portfolio with multiple assets using probabilistic forecasts.
    """

    def __init__(
        self,
        strategy: ProbabilisticStrategy,
        initial_capital: float = 10000.0
    ):
        """
        Initialize portfolio manager.

        Args:
            strategy: Trading strategy
            initial_capital: Starting capital
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> position_value
        self.history: List[Dict] = []

    def update(
        self,
        forecasts: Dict[str, ForecastDistribution],
        current_prices: Dict[str, float],
        timestamp: str
    ) -> Dict[str, TradingSignal]:
        """
        Update portfolio based on new forecasts.

        Args:
            forecasts: Forecasts for each symbol
            current_prices: Current prices for each symbol
            timestamp: Current timestamp

        Returns:
            Signals for each symbol
        """
        signals = {}

        for symbol, forecast in forecasts.items():
            current_position = self.positions.get(symbol, 0.0) / self.capital

            signal = self.strategy.generate_signal(forecast, current_position)
            signal.timestamp = timestamp
            signals[symbol] = signal

            # Update position
            target_value = signal.position_size * self.capital
            current_value = self.positions.get(symbol, 0.0)

            trade_value = target_value - current_value

            if abs(trade_value) > 0:
                # Apply transaction cost
                cost = self.strategy.transaction_cost * abs(trade_value)
                self.capital -= cost

                self.positions[symbol] = target_value

                logger.info(
                    f"{timestamp} | {symbol} | "
                    f"Signal: {signal.signal_type.name} | "
                    f"Position: {signal.position_size:.2%} | "
                    f"Prob(+): {signal.prob_positive:.2%}"
                )

        # Record state
        self.history.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'positions': self.positions.copy(),
            'signals': signals
        })

        return signals

    def mark_to_market(
        self,
        returns: Dict[str, float]
    ):
        """
        Mark positions to market with realized returns.

        Args:
            returns: Realized returns for each symbol
        """
        pnl = 0.0

        for symbol, position_value in self.positions.items():
            if symbol in returns:
                pnl += position_value * returns[symbol]

        self.capital += pnl

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Compute portfolio performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.history:
            return {}

        capitals = [h['capital'] for h in self.history]
        returns = np.diff(capitals) / capitals[:-1]

        metrics = {
            'total_return': (self.capital / self.initial_capital - 1),
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252),
            'max_drawdown': self._compute_max_drawdown(capitals),
            'win_rate': (returns > 0).mean() if len(returns) > 0 else 0,
            'num_trades': len([h for h in self.history if h.get('signals')])
        }

        return metrics

    def _compute_max_drawdown(self, capitals: List[float]) -> float:
        """Compute maximum drawdown."""
        peak = capitals[0]
        max_dd = 0.0

        for capital in capitals:
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)

        return max_dd


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Create synthetic forecast
    samples = np.random.randn(1000) * 0.02 + 0.01  # Mean 1%, std 2%
    forecast = ForecastDistribution.from_samples(samples)

    print("Forecast Statistics:")
    print(f"  Mean: {forecast.mean:.4f}")
    print(f"  Std: {forecast.std:.4f}")
    print(f"  Prob(+): {forecast.prob_greater_than(0):.2%}")
    print(f"  VaR(95%): {forecast.var(0.95):.4f}")
    print(f"  CVaR(95%): {forecast.cvar(0.95):.4f}")

    # Kelly fraction
    kelly = kelly_fraction_continuous(forecast)
    print(f"\nKelly fraction: {kelly:.2%}")

    # Generate signal
    strategy = ProbabilisticStrategy(
        confidence_threshold=0.55,
        kelly_fraction=0.5,
        max_position_size=0.20
    )

    signal = strategy.generate_signal(forecast)
    print(f"\nSignal: {signal.signal_type.name}")
    print(f"Position size: {signal.position_size:.2%}")
    print(f"Confidence: {signal.confidence:.2%}")

    # Expected P&L
    pnl_stats = strategy.compute_expected_pnl(forecast, signal.position_size)
    print(f"\nExpected P&L:")
    for k, v in pnl_stats.items():
        print(f"  {k}: {v:.4f}")
