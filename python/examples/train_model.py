"""
Example: Training a Probabilistic Forecasting Model
===================================================

This example demonstrates how to:
1. Fetch data from Bybit
2. Compute features
3. Train a quantile regression model
4. Evaluate probabilistic forecasts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from data_fetcher import (
    BybitDataFetcher,
    compute_returns,
    compute_volatility,
    prepare_features,
    split_data
)
from features import compute_all_features, select_features
from models.quantile_regression import QuantileRegressor, QuantileRegressorTrainer
from scoring import compute_all_metrics


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    print("=" * 60)
    print("Probabilistic Forecasting Model Training")
    print("=" * 60)

    # Load configuration
    config = load_config()
    print(f"\nConfiguration loaded.")

    # Fetch data
    print(f"\nFetching data from Bybit...")
    fetcher = BybitDataFetcher()

    symbol = config['data']['symbols'][0]  # Use first symbol
    df = fetcher.fetch_historical_data(
        symbol=symbol,
        timeframe=config['data']['timeframe'],
        days=config['data']['history_days']
    )

    print(f"Fetched {len(df)} candles for {symbol}")

    # Compute features
    print("\nComputing features...")
    df = compute_all_features(df)
    features = select_features(df)

    # Prepare target (next period return)
    target_horizon = config['training']['prediction_horizon']
    target = np.log(df['close'].shift(-target_horizon) / df['close'])

    # Remove NaN rows
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_mask]
    target = target[valid_mask]

    print(f"Feature shape: {features.shape}")
    print(f"Features: {list(features.columns)[:10]}...")

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        features, target,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio']
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Normalize features
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8

    X_train_norm = ((X_train - train_mean) / train_std).values
    X_val_norm = ((X_val - train_mean) / train_std).values
    X_test_norm = ((X_test - train_mean) / train_std).values

    y_train_arr = y_train.values
    y_val_arr = y_val.values
    y_test_arr = y_test.values

    # Create model
    model_config = config['model']['quantile_regression']
    quantiles = model_config['quantiles']

    print(f"\nCreating Quantile Regression model...")
    print(f"  Quantiles: {quantiles}")
    print(f"  Hidden size: {model_config['hidden_size']}")
    print(f"  Num layers: {model_config['num_layers']}")

    model = QuantileRegressor(
        input_dim=X_train_norm.shape[1],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout'],
        quantiles=quantiles
    )

    # Train model
    print("\nTraining model...")
    trainer = QuantileRegressorTrainer(
        model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    history = trainer.fit(
        X_train_norm, y_train_arr,
        X_val=X_val_norm, y_val=y_val_arr,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    print(f"\nTraining completed.")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")

    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    predictions = trainer.predict(X_test_norm)

    # Convert quantile predictions to samples for scoring
    n_samples_scoring = 500
    samples = np.zeros((n_samples_scoring, len(y_test_arr)))

    for i in range(len(y_test_arr)):
        # Interpolate to get samples from predicted quantiles
        u = np.random.rand(n_samples_scoring)
        samples[:, i] = np.interp(u, quantiles, predictions[i])

    # Compute metrics
    print("\nComputing forecast quality metrics...")
    metrics = compute_all_metrics(samples, y_test_arr)

    print("\n" + "=" * 60)
    print("FORECAST QUALITY METRICS")
    print("=" * 60)

    print("\nProbabilistic Accuracy:")
    print(f"  CRPS:              {metrics['crps_mean']:.6f}")
    print(f"  Log Score:         {metrics['log_score_mean']:.4f}")

    print("\nCalibration:")
    print(f"  Mean Cal. Error:   {metrics['mean_calibration_error']:.4f}")
    print(f"  PIT Mean:          {metrics['pit_mean']:.4f} (ideal: 0.5)")
    print(f"  PIT Std:           {metrics['pit_std']:.4f} (ideal: 0.289)")

    print("\nCoverage:")
    print(f"  50% CI Coverage:   {metrics['coverage_50']:.2%} (target: 50%)")
    print(f"  90% CI Coverage:   {metrics['coverage_90']:.2%} (target: 90%)")
    print(f"  95% CI Coverage:   {metrics['coverage_95']:.2%} (target: 95%)")

    print("\nSharpness:")
    print(f"  Mean Interval:     {metrics['mean_interval_width']:.6f}")

    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    for i in range(min(5, len(y_test_arr))):
        print(f"\nSample {i + 1}:")
        print(f"  Actual return:     {y_test_arr[i]:.4f}")
        print(f"  Predicted quantiles:")
        for j, q in enumerate(quantiles):
            print(f"    {q*100:.0f}th percentile: {predictions[i, j]:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
