//! Data types for Bybit API responses

use serde::{Deserialize, Serialize};

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start timestamp in milliseconds
    pub timestamp: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Create a new Kline
    pub fn new(
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            turnover: close * volume,
        }
    }

    /// Compute log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        (self.close / prev_close).ln()
    }

    /// Compute typical price
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Compute true range
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let hl = self.high - self.low;
        let hc = (self.high - prev_close).abs();
        let lc = (self.low - prev_close).abs();
        hl.max(hc).max(lc)
    }

    /// Check if candle is bullish
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Compute body size as fraction of range
    pub fn body_ratio(&self) -> f64 {
        let range = self.high - self.low;
        if range > 0.0 {
            (self.close - self.open).abs() / range
        } else {
            0.0
        }
    }
}

/// Ticker data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Trading symbol
    pub symbol: String,
    /// Last traded price
    pub last_price: f64,
    /// 24h high price
    pub high_24h: f64,
    /// 24h low price
    pub low_24h: f64,
    /// 24h trading volume
    pub volume_24h: f64,
    /// 24h turnover
    pub turnover_24h: f64,
    /// Price change percentage
    pub price_change_pct: f64,
}

/// Order book entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level
    pub price: f64,
    /// Quantity at this level
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Trading symbol
    pub symbol: String,
    /// Timestamp
    pub timestamp: i64,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Get best bid price
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Compute bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Compute spread in basis points
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid + ask) / 2.0;
                Some((ask - bid) / mid * 10000.0)
            }
            _ => None,
        }
    }

    /// Compute mid price
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Compute depth imbalance at top N levels
    pub fn depth_imbalance(&self, levels: usize) -> f64 {
        let bid_depth: f64 = self.bids.iter().take(levels).map(|l| l.quantity).sum();
        let ask_depth: f64 = self.asks.iter().take(levels).map(|l| l.quantity).sum();
        let total = bid_depth + ask_depth;

        if total > 0.0 {
            (bid_depth - ask_depth) / total
        } else {
            0.0
        }
    }
}

/// Bybit API response wrapper
#[derive(Debug, Deserialize)]
pub struct BybitResponse<T> {
    pub ret_code: i32,
    pub ret_msg: String,
    pub result: T,
    pub time: i64,
}

/// Kline API response result
#[derive(Debug, Deserialize)]
pub struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_log_return() {
        let kline = Kline::new(0, 100.0, 105.0, 98.0, 102.0, 1000.0);
        let prev_close = 100.0;
        let ret = kline.log_return(prev_close);
        assert!((ret - 0.0198).abs() < 0.001);
    }

    #[test]
    fn test_order_book_spread() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 49990.0, quantity: 1.0 },
                OrderBookLevel { price: 49980.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 50010.0, quantity: 1.0 },
                OrderBookLevel { price: 50020.0, quantity: 2.0 },
            ],
        };

        assert_eq!(book.best_bid(), Some(49990.0));
        assert_eq!(book.best_ask(), Some(50010.0));
        assert_eq!(book.spread(), Some(20.0));
    }

    #[test]
    fn test_depth_imbalance() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 0,
            bids: vec![
                OrderBookLevel { price: 49990.0, quantity: 3.0 },
                OrderBookLevel { price: 49980.0, quantity: 2.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 50010.0, quantity: 1.0 },
                OrderBookLevel { price: 50020.0, quantity: 1.0 },
            ],
        };

        let imbalance = book.depth_imbalance(2);
        // (3+2 - 1+1) / (3+2+1+1) = 3/7 ≈ 0.428
        assert!((imbalance - 0.428).abs() < 0.01);
    }
}
