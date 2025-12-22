"""
Backtesting Engine for Probabilistic Forecasting Strategies
==========================================================

Implements backtesting with proper handling of probabilistic forecasts.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from datetime import datetime
import logging

from strategy import (
    ProbabilisticStrategy,
    ForecastDistribution,
    TradingSignal,
    SignalType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: str
    exit_time: Optional[str]
    symbol: str
    direction: SignalType
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    pnl: Optional[float] = None
    forecast_mean: float = 0.0
    forecast_std: float = 0.0
    forecast_prob_positive: float = 0.5


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Forecast quality
    avg_crps: float
    avg_calibration_error: float
    avg_coverage_90: float

    # Time series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: List[Trade]


class BacktestEngine:
    """
    Backtesting engine for probabilistic forecasting strategies.
    """

    def __init__(
        self,
        strategy: ProbabilisticStrategy,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        slippage_bps: float = 5.0
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Trading strategy
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            slippage_bps: Slippage in basis points
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage_bps / 10000

    def run(
        self,
        prices: pd.DataFrame,
        forecasts: Dict[str, List[ForecastDistribution]],
        forecast_timestamps: List[str]
    ) -> BacktestResult:
        """
        Run backtest.

        Args:
            prices: DataFrame with OHLCV data, index is timestamp
            forecasts: Dictionary mapping symbol to list of forecasts
            forecast_timestamps: Timestamps corresponding to forecasts

        Returns:
            BacktestResult with all metrics
        """
        capital = self.initial_capital
        positions: Dict[str, float] = {}  # symbol -> position_value
        position_sizes: Dict[str, float] = {}  # symbol -> size (-1 to 1)

        equity_curve = []
        returns_list = []
        position_history = []
        trades: List[Trade] = []
        open_trades: Dict[str, Trade] = {}

        # Forecast quality metrics
        crps_values = []
        coverage_90_values = []

        for i, timestamp in enumerate(forecast_timestamps):
            if i >= len(prices):
                break

            current_prices = prices.iloc[i]

            # Get forecasts for this timestamp
            current_forecasts = {}
            for symbol in forecasts.keys():
                if i < len(forecasts[symbol]):
                    current_forecasts[symbol] = forecasts[symbol][i]

            # Generate signals
            signals = {}
            for symbol, forecast in current_forecasts.items():
                if symbol not in current_prices.index:
                    continue

                current_pos = position_sizes.get(symbol, 0.0)
                signal = self.strategy.generate_signal(forecast, current_pos)
                signals[symbol] = signal

                # Check forecast quality (if we have next period's return)
                if i < len(prices) - 1:
                    next_prices = prices.iloc[i + 1]
                    if symbol in next_prices.index:
                        actual_return = (next_prices[symbol] - current_prices[symbol]) / current_prices[symbol]

                        # CRPS approximation
                        crps = self._compute_crps(forecast.samples, actual_return)
                        crps_values.append(crps)

                        # Coverage check
                        q05 = forecast.quantiles.get(0.05, forecast.samples.min())
                        q95 = forecast.quantiles.get(0.95, forecast.samples.max())
                        in_interval = q05 <= actual_return <= q95
                        coverage_90_values.append(float(in_interval))

            # Execute trades
            for symbol, signal in signals.items():
                price = current_prices.get(symbol)
                if price is None:
                    continue

                current_pos_size = position_sizes.get(symbol, 0.0)
                target_pos_size = signal.position_size

                if abs(target_pos_size - current_pos_size) > 0.01:
                    # Close existing position if direction changed
                    if symbol in open_trades and np.sign(target_pos_size) != np.sign(current_pos_size):
                        trade = open_trades[symbol]
                        exit_price = price * (1 - self.slippage * np.sign(current_pos_size))
                        trade.exit_time = timestamp
                        trade.exit_price = exit_price
                        trade.pnl = (exit_price - trade.entry_price) / trade.entry_price * trade.position_size
                        trades.append(trade)
                        del open_trades[symbol]

                        # Apply transaction cost
                        capital -= self.transaction_cost * abs(current_pos_size) * capital

                    # Open new position
                    if abs(target_pos_size) > 0.01 and signal.signal_type != SignalType.HOLD:
                        entry_price = price * (1 + self.slippage * np.sign(target_pos_size))

                        open_trades[symbol] = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            symbol=symbol,
                            direction=signal.signal_type,
                            entry_price=entry_price,
                            exit_price=None,
                            position_size=target_pos_size,
                            forecast_mean=signal.expected_return,
                            forecast_std=current_forecasts[symbol].std if symbol in current_forecasts else 0,
                            forecast_prob_positive=signal.prob_positive
                        )

                        # Apply transaction cost
                        capital -= self.transaction_cost * abs(target_pos_size) * capital

                    position_sizes[symbol] = target_pos_size
                    positions[symbol] = target_pos_size * capital

            # Mark to market
            if i > 0:
                prev_prices = prices.iloc[i - 1]
                pnl = 0.0

                for symbol, pos_value in positions.items():
                    if symbol in current_prices.index and symbol in prev_prices.index:
                        ret = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                        pnl += position_sizes.get(symbol, 0) * capital * ret

                capital += pnl

                if len(equity_curve) > 0:
                    returns_list.append(pnl / equity_curve[-1])
                else:
                    returns_list.append(0)

            equity_curve.append(capital)
            position_history.append(position_sizes.copy())

        # Close remaining positions
        for symbol, trade in open_trades.items():
            final_price = prices.iloc[-1].get(symbol, trade.entry_price)
            trade.exit_time = forecast_timestamps[-1]
            trade.exit_price = final_price
            trade.pnl = (final_price - trade.entry_price) / trade.entry_price * trade.position_size
            trades.append(trade)

        # Compute metrics
        equity_series = pd.Series(equity_curve, index=forecast_timestamps[:len(equity_curve)])
        returns_series = pd.Series(returns_list, index=forecast_timestamps[1:len(returns_list)+1])

        result = self._compute_metrics(
            equity_series,
            returns_series,
            trades,
            position_history,
            crps_values,
            coverage_90_values
        )

        return result

    def _compute_crps(self, samples: np.ndarray, actual: float) -> float:
        """Compute CRPS from samples."""
        abs_diff = np.abs(samples - actual).mean()
        sorted_samples = np.sort(samples)
        n = len(samples)
        weights = (2 * np.arange(1, n + 1) - 1 - n) / (n ** 2)
        spread = 2 * np.sum(weights * sorted_samples)
        return abs_diff - 0.5 * spread

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Trade],
        positions: List[Dict],
        crps_values: List[float],
        coverage_values: List[float]
    ) -> BacktestResult:
        """Compute all backtest metrics."""

        # Handle empty returns
        if len(returns) == 0:
            returns = pd.Series([0.0])

        # Performance metrics
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        n_periods = len(returns)
        periods_per_year = 252 * 24  # Assuming hourly data

        annualized_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1

        returns_std = returns.std() if len(returns) > 1 else 1e-10
        sharpe_ratio = (returns.mean() / returns_std * np.sqrt(periods_per_year)) if returns_std > 0 else 0

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else 1e-10
        sortino_ratio = (returns.mean() / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (cummax - equity) / cummax
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        completed_trades = [t for t in trades if t.pnl is not None]
        num_trades = len(completed_trades)

        if num_trades > 0:
            pnls = [t.pnl for t in completed_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / num_trades
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        # Forecast quality
        avg_crps = np.mean(crps_values) if crps_values else 0
        avg_coverage = np.mean(coverage_values) if coverage_values else 0

        positions_df = pd.DataFrame(positions)

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_crps=avg_crps,
            avg_calibration_error=abs(0.90 - avg_coverage),
            avg_coverage_90=avg_coverage,
            equity_curve=equity,
            returns=returns,
            positions=positions_df,
            trades=trades
        )


def print_backtest_results(result: BacktestResult):
    """Pretty print backtest results."""
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\nPerformance Metrics:")
    print(f"  Total Return:      {result.total_return:>10.2%}")
    print(f"  Annualized Return: {result.annualized_return:>10.2%}")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
    print(f"  Sortino Ratio:     {result.sortino_ratio:>10.2f}")
    print(f"  Max Drawdown:      {result.max_drawdown:>10.2%}")
    print(f"  Calmar Ratio:      {result.calmar_ratio:>10.2f}")

    print("\nTrade Statistics:")
    print(f"  Number of Trades:  {result.num_trades:>10d}")
    print(f"  Win Rate:          {result.win_rate:>10.2%}")
    print(f"  Profit Factor:     {result.profit_factor:>10.2f}")
    print(f"  Avg Win:           {result.avg_win:>10.4f}")
    print(f"  Avg Loss:          {result.avg_loss:>10.4f}")

    print("\nForecast Quality:")
    print(f"  Avg CRPS:          {result.avg_crps:>10.6f}")
    print(f"  90% Coverage:      {result.avg_coverage_90:>10.2%}")
    print(f"  Calibration Error: {result.avg_calibration_error:>10.4f}")

    print("=" * 60)


if __name__ == "__main__":
    # Example backtest with synthetic data
    np.random.seed(42)

    # Generate synthetic price data
    n_periods = 500
    timestamps = [f"2024-01-01T{i:04d}" for i in range(n_periods)]

    price_data = {
        'BTC/USDT': 50000 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.01)),
        'ETH/USDT': 3000 * np.exp(np.cumsum(np.random.randn(n_periods) * 0.015)),
    }
    prices = pd.DataFrame(price_data, index=timestamps)

    # Generate synthetic forecasts
    forecasts = {}
    for symbol in ['BTC/USDT', 'ETH/USDT']:
        forecasts[symbol] = []
        for i in range(n_periods):
            # Forecast with slight positive bias
            samples = np.random.randn(200) * 0.02 + 0.002
            forecasts[symbol].append(ForecastDistribution.from_samples(samples))

    # Create strategy and run backtest
    strategy = ProbabilisticStrategy(
        confidence_threshold=0.55,
        kelly_fraction=0.25,
        max_position_size=0.15
    )

    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=10000,
        transaction_cost=0.001
    )

    result = engine.run(prices, forecasts, timestamps)
    print_backtest_results(result)
