//! Bybit API client for fetching market data

use crate::api::types::{BybitResponse, Kline, KlineResult, OrderBook, OrderBookLevel, Ticker};
use anyhow::{Context, Result};
use reqwest::Client;
use std::time::Duration;

/// Bybit API base URL
const BYBIT_API_BASE: &str = "https://api.bybit.com";

/// Bybit API client
#[derive(Debug, Clone)]
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: BYBIT_API_BASE.to_string(),
        }
    }

    /// Create client with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.to_string(),
        }
    }

    /// Fetch kline (candlestick) data
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        // Bybit returns newest first, reverse for chronological order
        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch klines synchronously (blocking)
    pub fn get_klines_blocking(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url, symbol, interval, limit
        );

        let response: BybitResponse<KlineResult> = reqwest::blocking::get(&url)
            .context("Failed to send request")?
            .json()
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let klines: Vec<Kline> = response
            .result
            .list
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 6 {
                    Some(Kline {
                        timestamp: row[0].parse().ok()?,
                        open: row[1].parse().ok()?,
                        high: row[2].parse().ok()?,
                        low: row[3].parse().ok()?,
                        close: row[4].parse().ok()?,
                        volume: row[5].parse().ok()?,
                        turnover: row.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0),
                    })
                } else {
                    None
                }
            })
            .collect();

        let mut klines = klines;
        klines.reverse();

        Ok(klines)
    }

    /// Fetch ticker data
    pub async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let url = format!(
            "{}/v5/market/tickers?category=spot&symbol={}",
            self.base_url, symbol
        );

        #[derive(serde::Deserialize)]
        struct TickerResult {
            list: Vec<TickerData>,
        }

        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TickerData {
            symbol: String,
            last_price: String,
            high_price_24h: String,
            low_price_24h: String,
            volume_24h: String,
            turnover_24h: String,
            price_24h_pcnt: String,
        }

        let response: BybitResponse<TickerResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let ticker_data = response
            .result
            .list
            .into_iter()
            .next()
            .context("No ticker data")?;

        Ok(Ticker {
            symbol: ticker_data.symbol,
            last_price: ticker_data.last_price.parse()?,
            high_24h: ticker_data.high_price_24h.parse()?,
            low_24h: ticker_data.low_price_24h.parse()?,
            volume_24h: ticker_data.volume_24h.parse()?,
            turnover_24h: ticker_data.turnover_24h.parse()?,
            price_change_pct: ticker_data.price_24h_pcnt.parse()?,
        })
    }

    /// Fetch order book
    pub async fn get_order_book(&self, symbol: &str, limit: usize) -> Result<OrderBook> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url, symbol, limit
        );

        #[derive(serde::Deserialize)]
        struct OrderBookResult {
            s: String,
            b: Vec<Vec<String>>,
            a: Vec<Vec<String>>,
            ts: i64,
        }

        let response: BybitResponse<OrderBookResult> = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to send request")?
            .json()
            .await
            .context("Failed to parse response")?;

        if response.ret_code != 0 {
            anyhow::bail!("API error: {}", response.ret_msg);
        }

        let result = response.result;

        let bids: Vec<OrderBookLevel> = result
            .b
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        let asks: Vec<OrderBookLevel> = result
            .a
            .into_iter()
            .filter_map(|row| {
                if row.len() >= 2 {
                    Some(OrderBookLevel {
                        price: row[0].parse().ok()?,
                        quantity: row[1].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(OrderBook {
            symbol: result.s,
            timestamp: result.ts,
            bids,
            asks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_get_klines() {
        let client = BybitClient::new();
        let klines = client.get_klines("BTCUSDT", "60", 10).await.unwrap();
        assert!(!klines.is_empty());
        assert!(klines[0].close > 0.0);
    }

    #[tokio::test]
    #[ignore] // Requires network access
    async fn test_get_ticker() {
        let client = BybitClient::new();
        let ticker = client.get_ticker("BTCUSDT").await.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(ticker.last_price > 0.0);
    }
}
