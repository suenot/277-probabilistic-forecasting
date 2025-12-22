//! # Probabilistic Forecasting for Trading
//!
//! This library provides implementations for probabilistic forecasting models
//! applied to cryptocurrency trading using data from Bybit exchange.
//!
//! ## Core Concepts
//!
//! - **Probabilistic Forecasts**: Full probability distributions instead of point estimates
//! - **Quantile Regression**: Predict specific quantiles of the return distribution
//! - **Proper Scoring Rules**: CRPS, Log Score for evaluating probabilistic forecasts
//! - **Kelly Criterion**: Optimal position sizing with uncertainty
//!
//! ## Modules
//!
//! - `api` - Bybit API client for fetching market data
//! - `distributions` - Probability distributions and operations
//! - `models` - Forecasting models (quantile regression, etc.)
//! - `scoring` - Proper scoring rules for forecast evaluation
//! - `strategy` - Trading strategy with probabilistic signals
//! - `backtest` - Backtesting framework
//!
//! ## Example
//!
//! ```rust,no_run
//! use probabilistic_forecasting::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch market data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "1h", 500).await?;
//!
//!     // Compute features
//!     let features = FeatureEngine::compute(&klines);
//!
//!     // Make probabilistic forecast
//!     let model = QuantileRegressor::new(QuantileConfig::default());
//!     let forecast = model.predict(&features);
//!
//!     // Generate trading signal
//!     let strategy = ProbabilisticStrategy::new(StrategyConfig::default());
//!     let signal = strategy.generate_signal(&forecast);
//!
//!     println!("Signal: {:?}", signal);
//!     println!("Position size: {:.2}%", signal.position_size * 100.0);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod backtest;
pub mod distributions;
pub mod models;
pub mod scoring;
pub mod strategy;

// Re-export commonly used types
pub use api::client::BybitClient;
pub use api::types::{Kline, OrderBook, Ticker};
pub use backtest::engine::BacktestEngine;
pub use backtest::report::BacktestReport;
pub use distributions::forecast::ForecastDistribution;
pub use distributions::gaussian::GaussianDistribution;
pub use distributions::student_t::StudentTDistribution;
pub use models::features::FeatureEngine;
pub use models::quantile::QuantileRegressor;
pub use scoring::crps::compute_crps;
pub use scoring::log_score::compute_log_score;
pub use strategy::kelly::kelly_fraction;
pub use strategy::signal::{Signal, SignalType};
pub use strategy::probabilistic::ProbabilisticStrategy;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default trading symbols for examples
pub const DEFAULT_SYMBOLS: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "MATICUSDT", "DOTUSDT",
];

/// Default quantile levels for quantile regression
pub const DEFAULT_QUANTILES: &[f64] = &[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::api::client::BybitClient;
    pub use crate::api::types::{Kline, OrderBook, Ticker};
    pub use crate::backtest::engine::BacktestEngine;
    pub use crate::backtest::report::BacktestReport;
    pub use crate::distributions::forecast::ForecastDistribution;
    pub use crate::distributions::gaussian::GaussianDistribution;
    pub use crate::models::features::FeatureEngine;
    pub use crate::models::quantile::{QuantileConfig, QuantileRegressor};
    pub use crate::scoring::crps::compute_crps;
    pub use crate::scoring::log_score::compute_log_score;
    pub use crate::strategy::kelly::kelly_fraction;
    pub use crate::strategy::signal::{Signal, SignalType};
    pub use crate::strategy::probabilistic::{ProbabilisticStrategy, StrategyConfig};
    pub use crate::DEFAULT_QUANTILES;
    pub use crate::DEFAULT_SYMBOLS;
}
