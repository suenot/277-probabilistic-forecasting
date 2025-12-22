"""
Data Fetcher for Probabilistic Forecasting
==========================================

Fetches cryptocurrency data from Bybit exchange using CCXT library.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BybitDataFetcher:
    """
    Fetches OHLCV data from Bybit exchange using CCXT.
    """

    def __init__(self, api_key: Optional[str] = None, secret: Optional[str] = None):
        """
        Initialize Bybit client.

        Args:
            api_key: Optional API key for authenticated requests
            secret: Optional API secret
        """
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',  # or 'linear' for perpetual futures
            }
        })

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[int] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1m', '5m', '1h', '1d')
            since: Start timestamp in milliseconds
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )

            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            raise

    def fetch_historical_data(
        self,
        symbol: str,
        timeframe: str = '1h',
        days: int = 90
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for multiple days.

        Args:
            symbol: Trading pair
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            DataFrame with complete historical data
        """
        # Calculate milliseconds per candle
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }

        ms_per_candle = timeframe_ms.get(timeframe, 60 * 60 * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        all_data = []
        current_time = start_time

        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")

        while current_time < end_time:
            df = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_time,
                limit=1000
            )

            if df.empty:
                break

            all_data.append(df)

            # Move to next batch
            last_timestamp = int(df.index[-1].timestamp() * 1000)
            current_time = last_timestamp + ms_per_candle

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data)
        result = result[~result.index.duplicated(keep='first')]
        result = result.sort_index()

        logger.info(f"Fetched {len(result)} candles for {symbol}")

        return result

    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        days: int = 90
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of trading pairs
            timeframe: Candle timeframe
            days: Number of days of history

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}

        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            data[symbol] = self.fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                days=days
            )
            time.sleep(0.5)  # Rate limiting between symbols

        return data

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of trading pairs

        Returns:
            Dictionary mapping symbol to current price
        """
        prices = {}

        for symbol in symbols:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                prices[symbol] = ticker['last']
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = None

        return prices


def compute_returns(df: pd.DataFrame, periods: List[int] = [1, 4, 12, 24]) -> pd.DataFrame:
    """
    Compute log returns for multiple periods.

    Args:
        df: DataFrame with 'close' column
        periods: List of periods for return calculation

    Returns:
        DataFrame with return columns added
    """
    result = df.copy()

    for period in periods:
        result[f'return_{period}'] = np.log(
            result['close'] / result['close'].shift(period)
        )

    return result


def compute_volatility(
    df: pd.DataFrame,
    windows: List[int] = [12, 24, 168]
) -> pd.DataFrame:
    """
    Compute rolling volatility for multiple windows.

    Args:
        df: DataFrame with return column
        windows: List of window sizes

    Returns:
        DataFrame with volatility columns added
    """
    result = df.copy()

    # Ensure we have returns
    if 'return_1' not in result.columns:
        result['return_1'] = np.log(result['close'] / result['close'].shift(1))

    for window in windows:
        result[f'volatility_{window}'] = result['return_1'].rolling(
            window=window
        ).std() * np.sqrt(window)

    return result


def prepare_features(
    df: pd.DataFrame,
    target_horizon: int = 4
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.

    Args:
        df: DataFrame with OHLCV and derived features
        target_horizon: Prediction horizon in periods

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Compute target: future return
    target = np.log(
        df['close'].shift(-target_horizon) / df['close']
    )

    # Select feature columns (exclude raw OHLCV and target-related)
    feature_cols = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'volume', 'symbol'
    ] and not col.startswith('future_')]

    features = df[feature_cols].copy()

    # Add time features
    features['hour'] = df.index.hour
    features['day_of_week'] = df.index.dayofweek

    # Cyclical encoding for time
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

    # Drop raw time columns
    features = features.drop(columns=['hour', 'day_of_week'])

    # Remove rows with NaN
    valid_mask = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_mask]
    target = target[valid_mask]

    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train, validation, and test sets.
    Uses time-based splitting to avoid look-ahead bias.

    Args:
        features: Feature DataFrame
        target: Target Series
        train_ratio: Fraction for training
        val_ratio: Fraction for validation

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    n = len(features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = features.iloc[:train_end]
    y_train = target.iloc[:train_end]

    X_val = features.iloc[train_end:val_end]
    y_val = target.iloc[train_end:val_end]

    X_test = features.iloc[val_end:]
    y_test = target.iloc[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Example usage
    fetcher = BybitDataFetcher()

    # Fetch data for BTC/USDT
    df = fetcher.fetch_historical_data(
        symbol="BTC/USDT",
        timeframe="1h",
        days=30
    )

    print(f"Fetched {len(df)} candles")
    print(df.head())

    # Compute features
    df = compute_returns(df, periods=[1, 4, 12, 24])
    df = compute_volatility(df, windows=[12, 24, 168])

    # Prepare for modeling
    features, target = prepare_features(df, target_horizon=4)
    print(f"\nFeature shape: {features.shape}")
    print(f"Features: {list(features.columns)}")

    # Split data
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        features, target
    )
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
