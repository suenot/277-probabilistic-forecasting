//! Feature engineering for time series

use crate::api::types::Kline;
use ndarray::{Array1, Array2};

/// Feature engineering engine
pub struct FeatureEngine {
    /// Return periods to compute
    pub return_periods: Vec<usize>,
    /// Volatility windows
    pub volatility_windows: Vec<usize>,
}

impl Default for FeatureEngine {
    fn default() -> Self {
        Self {
            return_periods: vec![1, 4, 12, 24],
            volatility_windows: vec![12, 24, 48],
        }
    }
}

impl FeatureEngine {
    /// Create new feature engine
    pub fn new(return_periods: Vec<usize>, volatility_windows: Vec<usize>) -> Self {
        Self {
            return_periods,
            volatility_windows,
        }
    }

    /// Compute all features from kline data
    pub fn compute(&self, klines: &[Kline]) -> Array2<f64> {
        let n = klines.len();
        if n == 0 {
            return Array2::zeros((0, 0));
        }

        // Compute individual feature arrays
        let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        let highs: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let lows: Vec<f64> = klines.iter().map(|k| k.low).collect();

        let mut feature_vecs: Vec<Vec<f64>> = Vec::new();

        // Returns at different periods
        for &period in &self.return_periods {
            let returns = compute_log_returns(&closes, period);
            feature_vecs.push(returns);
        }

        // Volatility at different windows
        let base_returns = compute_log_returns(&closes, 1);
        for &window in &self.volatility_windows {
            let volatility = compute_rolling_std(&base_returns, window);
            feature_vecs.push(volatility);
        }

        // RSI
        let rsi = compute_rsi(&closes, 14);
        feature_vecs.push(rsi);

        // MACD components
        let (macd, signal) = compute_macd(&closes, 12, 26, 9);
        feature_vecs.push(macd);
        feature_vecs.push(signal);

        // Bollinger Band position
        let bb_pos = compute_bb_position(&closes, 20, 2.0);
        feature_vecs.push(bb_pos);

        // Volume features
        let vol_ma = compute_sma(&volumes, 20);
        let vol_ratio: Vec<f64> = volumes
            .iter()
            .zip(vol_ma.iter())
            .map(|(v, ma)| if *ma > 0.0 { v / ma } else { 1.0 })
            .collect();
        feature_vecs.push(vol_ratio);

        // Price features
        let body_ratio: Vec<f64> = klines.iter().map(|k| k.body_ratio()).collect();
        feature_vecs.push(body_ratio);

        // ATR
        let atr = compute_atr(klines, 14);
        let atr_pct: Vec<f64> = atr
            .iter()
            .zip(closes.iter())
            .map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 })
            .collect();
        feature_vecs.push(atr_pct);

        // High-Low range
        let hl_range: Vec<f64> = highs
            .iter()
            .zip(lows.iter())
            .zip(closes.iter())
            .map(|((h, l), c)| if *c > 0.0 { (h - l) / c } else { 0.0 })
            .collect();
        feature_vecs.push(hl_range);

        // Convert to 2D array
        let num_features = feature_vecs.len();
        let mut features = Array2::zeros((n, num_features));

        for (j, fvec) in feature_vecs.iter().enumerate() {
            for (i, &val) in fvec.iter().enumerate() {
                features[[i, j]] = if val.is_finite() { val } else { 0.0 };
            }
        }

        features
    }

    /// Get feature names
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for &period in &self.return_periods {
            names.push(format!("return_{}", period));
        }

        for &window in &self.volatility_windows {
            names.push(format!("volatility_{}", window));
        }

        names.push("rsi_14".to_string());
        names.push("macd".to_string());
        names.push("macd_signal".to_string());
        names.push("bb_position".to_string());
        names.push("volume_ratio".to_string());
        names.push("body_ratio".to_string());
        names.push("atr_pct".to_string());
        names.push("hl_range".to_string());

        names
    }
}

/// Compute log returns
fn compute_log_returns(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut returns = vec![0.0; n];

    for i in period..n {
        if prices[i - period] > 0.0 {
            returns[i] = (prices[i] / prices[i - period]).ln();
        }
    }

    returns
}

/// Compute rolling standard deviation
fn compute_rolling_std(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![0.0; n];

    for i in window..n {
        let slice = &values[(i - window)..i];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
        result[i] = variance.sqrt();
    }

    result
}

/// Compute simple moving average
fn compute_sma(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![0.0; n];

    for i in window..n {
        let sum: f64 = values[(i - window)..i].iter().sum();
        result[i] = sum / window as f64;
    }

    // Fill initial values with first computed value
    if n > window {
        for i in 0..window {
            result[i] = result[window];
        }
    }

    result
}

/// Compute exponential moving average
fn compute_ema(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![0.0; n];
    let alpha = 2.0 / (period as f64 + 1.0);

    if n > 0 {
        result[0] = values[0];
        for i in 1..n {
            result[i] = alpha * values[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// Compute RSI
fn compute_rsi(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    let mut rsi = vec![50.0; n];

    if n < period + 1 {
        return rsi;
    }

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    let avg_gains = compute_ema(&gains, period);
    let avg_losses = compute_ema(&losses, period);

    for i in period..n {
        let ag = avg_gains[i];
        let al = avg_losses[i];

        if al > 0.0 {
            let rs = ag / al;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        } else if ag > 0.0 {
            rsi[i] = 100.0;
        } else {
            rsi[i] = 50.0;
        }
    }

    rsi
}

/// Compute MACD
fn compute_macd(
    prices: &[f64],
    fast: usize,
    slow: usize,
    signal: usize,
) -> (Vec<f64>, Vec<f64>) {
    let fast_ema = compute_ema(prices, fast);
    let slow_ema = compute_ema(prices, slow);

    let macd: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(f, s)| f - s)
        .collect();

    let signal_line = compute_ema(&macd, signal);

    (macd, signal_line)
}

/// Compute Bollinger Band position
fn compute_bb_position(prices: &[f64], period: usize, num_std: f64) -> Vec<f64> {
    let n = prices.len();
    let mut position = vec![0.5; n];

    let sma = compute_sma(prices, period);
    let std = compute_rolling_std(prices, period);

    for i in period..n {
        let upper = sma[i] + num_std * std[i];
        let lower = sma[i] - num_std * std[i];
        let range = upper - lower;

        if range > 0.0 {
            position[i] = (prices[i] - lower) / range;
        }
    }

    position
}

/// Compute Average True Range
fn compute_atr(klines: &[Kline], period: usize) -> Vec<f64> {
    let n = klines.len();
    let mut tr = vec![0.0; n];

    if n > 0 {
        tr[0] = klines[0].high - klines[0].low;
    }

    for i in 1..n {
        let prev_close = klines[i - 1].close;
        let hl = klines[i].high - klines[i].low;
        let hc = (klines[i].high - prev_close).abs();
        let lc = (klines[i].low - prev_close).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    compute_ema(&tr, period)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_klines(n: usize) -> Vec<Kline> {
        (0..n)
            .map(|i| {
                let base = 100.0 + (i as f64 * 0.1);
                Kline::new(
                    i as i64 * 3600000,
                    base,
                    base + 1.0,
                    base - 1.0,
                    base + 0.5,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_computation() {
        let klines = create_test_klines(100);
        let engine = FeatureEngine::default();
        let features = engine.compute(&klines);

        assert_eq!(features.nrows(), 100);
        assert!(features.ncols() > 0);
    }

    #[test]
    fn test_log_returns() {
        let prices = vec![100.0, 101.0, 102.01, 103.0303];
        let returns = compute_log_returns(&prices, 1);

        // ln(101/100) ≈ 0.00995
        assert!((returns[1] - 0.00995).abs() < 0.001);
    }

    #[test]
    fn test_rsi() {
        let mut prices = vec![100.0; 50];
        for i in 0..50 {
            prices[i] = 100.0 + (i as f64 * 0.5);
        }
        let rsi = compute_rsi(&prices, 14);

        // Trending up should have RSI > 50
        assert!(rsi[49] > 50.0);
    }
}
