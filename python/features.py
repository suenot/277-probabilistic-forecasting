"""
Feature Engineering for Probabilistic Forecasting
=================================================

Technical indicators and feature computation.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.

    Args:
        prices: Price series
        period: RSI period

    Returns:
        RSI values (0-100)
    """
    delta = prices.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Compute MACD indicator.

    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'macd': macd_line,
        'macd_signal': signal_line,
        'macd_hist': histogram
    })


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Args:
        prices: Price series
        period: Moving average period
        std_dev: Number of standard deviations

    Returns:
        DataFrame with upper, middle, lower bands and position
    """
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # Position within bands (0 = lower, 1 = upper)
    position = (prices - lower) / (upper - lower + 1e-10)

    return pd.DataFrame({
        'bb_upper': upper,
        'bb_middle': middle,
        'bb_lower': lower,
        'bb_position': position
    })


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period

    Returns:
        ATR values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_volume_features(df: pd.DataFrame, periods: List[int] = [12, 24]) -> pd.DataFrame:
    """
    Compute volume-based features.

    Args:
        df: DataFrame with volume column
        periods: Periods for rolling calculations

    Returns:
        DataFrame with volume features
    """
    result = pd.DataFrame(index=df.index)

    for period in periods:
        result[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        result[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()
        result[f'volume_ratio_{period}'] = df['volume'] / result[f'volume_ma_{period}']

    # Volume trend
    result['volume_trend'] = df['volume'].diff(12)

    # Relative volume
    result['relative_volume'] = df['volume'] / df['volume'].rolling(window=24).mean()

    return result


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price-based features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with price features
    """
    result = pd.DataFrame(index=df.index)

    # Candle features
    result['body'] = (df['close'] - df['open']) / df['open']
    result['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    result['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
    result['range'] = (df['high'] - df['low']) / df['open']

    # Price position
    result['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Gap
    result['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

    return result


def compute_momentum_features(
    prices: pd.Series,
    periods: List[int] = [4, 12, 24, 48]
) -> pd.DataFrame:
    """
    Compute momentum features.

    Args:
        prices: Price series
        periods: Momentum periods

    Returns:
        DataFrame with momentum features
    """
    result = pd.DataFrame(index=prices.index)

    for period in periods:
        # Price momentum (log return)
        result[f'momentum_{period}'] = np.log(prices / prices.shift(period))

        # Rate of change
        result[f'roc_{period}'] = (prices - prices.shift(period)) / prices.shift(period)

    return result


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all features for probabilistic forecasting.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all features
    """
    result = df.copy()

    # Returns at multiple horizons
    for period in [1, 4, 12, 24, 48]:
        result[f'return_{period}'] = np.log(result['close'] / result['close'].shift(period))

    # Volatility
    for window in [12, 24, 48, 168]:
        result[f'volatility_{window}'] = result['return_1'].rolling(window=window).std()

    # RSI
    result['rsi_14'] = compute_rsi(result['close'], period=14)
    result['rsi_7'] = compute_rsi(result['close'], period=7)

    # MACD
    macd = compute_macd(result['close'])
    result = pd.concat([result, macd], axis=1)

    # Bollinger Bands
    bb = compute_bollinger_bands(result['close'])
    result = pd.concat([result, bb], axis=1)

    # ATR
    result['atr_14'] = compute_atr(result['high'], result['low'], result['close'], period=14)
    result['atr_pct'] = result['atr_14'] / result['close']

    # Volume features
    volume_features = compute_volume_features(result)
    result = pd.concat([result, volume_features], axis=1)

    # Price features
    price_features = compute_price_features(result)
    result = pd.concat([result, price_features], axis=1)

    # Momentum
    momentum = compute_momentum_features(result['close'])
    result = pd.concat([result, momentum], axis=1)

    # Time features (cyclical)
    result['hour'] = result.index.hour
    result['day_of_week'] = result.index.dayofweek
    result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
    result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
    result['dow_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
    result['dow_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)

    # Drop intermediate columns
    result = result.drop(columns=['hour', 'day_of_week'], errors='ignore')

    return result


def select_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Select features for model training (exclude raw OHLCV).

    Args:
        df: DataFrame with all features
        exclude_cols: Additional columns to exclude

    Returns:
        DataFrame with selected features
    """
    exclude = ['open', 'high', 'low', 'close', 'volume', 'symbol']
    if exclude_cols:
        exclude.extend(exclude_cols)

    feature_cols = [col for col in df.columns if col not in exclude]
    return df[feature_cols]


if __name__ == "__main__":
    # Example usage
    from data_fetcher import BybitDataFetcher

    fetcher = BybitDataFetcher()
    df = fetcher.fetch_historical_data("BTC/USDT", "1h", days=30)

    # Compute all features
    df_features = compute_all_features(df)
    print(f"Total features: {len(df_features.columns)}")
    print(f"Feature columns: {list(df_features.columns)}")

    # Select features for modeling
    features = select_features(df_features)
    print(f"\nSelected features: {len(features.columns)}")
    print(features.describe())
