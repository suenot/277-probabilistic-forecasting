"""
Example: Running a Backtest with Probabilistic Forecasts
=======================================================

This example demonstrates a complete backtest workflow:
1. Fetch historical data
2. Train a probabilistic model
3. Generate forecasts
4. Run backtest with Kelly-optimal sizing
5. Analyze results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

from data_fetcher import BybitDataFetcher
from features import compute_all_features, select_features
from models.quantile_regression import QuantileRegressor, QuantileRegressorTrainer
from strategy import ProbabilisticStrategy, ForecastDistribution
from backtest import BacktestEngine, print_backtest_results


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(X_train, y_train, X_val, y_val, config):
    """Train quantile regression model."""
    model_config = config['model']['quantile_regression']

    model = QuantileRegressor(
        input_dim=X_train.shape[1],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        quantiles=model_config['quantiles']
    )

    trainer = QuantileRegressorTrainer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    trainer.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    return trainer


def generate_forecasts(trainer, X, quantiles, n_samples=200):
    """Generate forecast distributions from model predictions."""
    predictions = trainer.predict(X)
    forecasts = []

    for i in range(len(X)):
        # Generate samples by interpolating quantiles
        u = np.random.rand(n_samples)
        samples = np.interp(u, quantiles, predictions[i])
        forecasts.append(ForecastDistribution.from_samples(samples))

    return forecasts


def main():
    print("=" * 60)
    print("Probabilistic Forecasting Backtest")
    print("=" * 60)

    # Load configuration
    config = load_config()

    # Fetch data
    print("\nFetching data...")
    fetcher = BybitDataFetcher()

    symbol = "BTC/USDT"
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe="1h",
        days=60  # Use 60 days for this example
    )

    print(f"Fetched {len(df)} candles")

    # Compute features
    print("Computing features...")
    df = compute_all_features(df)
    feature_cols = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'volume', 'symbol'
    ]]
    features = df[feature_cols].copy()

    # Target: next period return
    target_horizon = 4
    target = np.log(df['close'].shift(-target_horizon) / df['close'])

    # Remove NaN
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_mask]
    target = target[valid_mask]
    prices_aligned = df.loc[valid_mask, ['close']].copy()
    prices_aligned.columns = [symbol]
    timestamps = list(prices_aligned.index.astype(str))

    # Split: use first 70% for training, rest for backtest
    n = len(features)
    train_end = int(n * 0.5)
    val_end = int(n * 0.65)

    X_train = features.iloc[:train_end]
    y_train = target.iloc[:train_end]
    X_val = features.iloc[train_end:val_end]
    y_val = target.iloc[train_end:val_end]
    X_test = features.iloc[val_end:]

    # Normalize
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8

    X_train_norm = ((X_train - train_mean) / train_std).values
    X_val_norm = ((X_val - train_mean) / train_std).values
    X_test_norm = ((X_test - train_mean) / train_std).values

    print(f"\nTraining model on {len(X_train)} samples...")
    trainer = train_model(
        X_train_norm, y_train.values,
        X_val_norm, y_val.values,
        config
    )

    # Generate forecasts for test period
    print(f"\nGenerating forecasts for {len(X_test)} test samples...")
    quantiles = config['model']['quantile_regression']['quantiles']
    forecasts_list = generate_forecasts(trainer, X_test_norm, quantiles)

    # Prepare data for backtest
    test_timestamps = timestamps[val_end:]
    test_prices = prices_aligned.iloc[val_end:]

    forecasts_dict = {symbol: forecasts_list}

    # Create strategy
    strategy_config = config['strategy']
    strategy = ProbabilisticStrategy(
        confidence_threshold=strategy_config['confidence_threshold'],
        min_expected_return=strategy_config['min_expected_return'],
        kelly_fraction=strategy_config['kelly_fraction'],
        max_position_size=strategy_config['max_position_size'],
        var_limit=strategy_config['var_limit'],
        transaction_cost=strategy_config['transaction_cost']
    )

    # Run backtest
    print("\nRunning backtest...")
    backtest_config = config['backtest']
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=backtest_config['initial_capital'],
        transaction_cost=strategy_config['transaction_cost'],
        slippage_bps=backtest_config['slippage_bps']
    )

    result = engine.run(
        prices=test_prices,
        forecasts=forecasts_dict,
        forecast_timestamps=test_timestamps
    )

    # Print results
    print_backtest_results(result)

    # Additional analysis
    print("\n" + "=" * 60)
    print("TRADE ANALYSIS")
    print("=" * 60)

    if result.trades:
        print(f"\nFirst 10 trades:")
        print("-" * 80)
        print(f"{'Entry Time':<20} {'Direction':<8} {'Size':>8} {'P&L':>10} {'Prob(+)':>10}")
        print("-" * 80)

        for trade in result.trades[:10]:
            if trade.pnl is not None:
                print(
                    f"{trade.entry_time:<20} "
                    f"{trade.direction.name:<8} "
                    f"{trade.position_size:>8.2%} "
                    f"{trade.pnl:>10.4f} "
                    f"{trade.forecast_prob_positive:>10.2%}"
                )

        # Analyze trade quality
        profitable_trades = [t for t in result.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in result.trades if t.pnl and t.pnl <= 0]

        print(f"\n{'Profitable trades:':<25} {len(profitable_trades)}")
        print(f"{'Losing trades:':<25} {len(losing_trades)}")

        if profitable_trades:
            avg_profit_prob = np.mean([t.forecast_prob_positive for t in profitable_trades])
            print(f"{'Avg prob(+) on wins:':<25} {avg_profit_prob:.2%}")

        if losing_trades:
            avg_loss_prob = np.mean([t.forecast_prob_positive for t in losing_trades])
            print(f"{'Avg prob(+) on losses:':<25} {avg_loss_prob:.2%}")

    # Equity curve summary
    print("\n" + "=" * 60)
    print("EQUITY CURVE SUMMARY")
    print("=" * 60)

    equity = result.equity_curve
    print(f"  Start:  ${equity.iloc[0]:,.2f}")
    print(f"  End:    ${equity.iloc[-1]:,.2f}")
    print(f"  Min:    ${equity.min():,.2f}")
    print(f"  Max:    ${equity.max():,.2f}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
