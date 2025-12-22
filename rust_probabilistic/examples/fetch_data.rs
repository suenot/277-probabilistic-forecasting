//! Example: Fetching data from Bybit
//!
//! Demonstrates how to use the Bybit API client to fetch market data.
//!
//! Run with: cargo run --example fetch_data

use probabilistic_forecasting::prelude::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Probabilistic Forecasting - Data Fetching Example");
    println!("=================================================\n");

    // Create Bybit client
    let client = BybitClient::new();

    // Fetch kline data for BTC/USDT
    println!("Fetching BTC/USDT 1-hour klines...");
    let klines = client.get_klines("BTCUSDT", "60", 100).await?;

    println!("Fetched {} klines\n", klines.len());

    // Display first few klines
    println!("First 5 klines:");
    println!("{:-<80}", "");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Timestamp", "Open", "High", "Low", "Close", "Volume"
    );
    println!("{:-<80}", "");

    for kline in klines.iter().take(5) {
        println!(
            "{:<15} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>12.2}",
            kline.timestamp, kline.open, kline.high, kline.low, kline.close, kline.volume
        );
    }

    // Compute basic statistics
    println!("\n\nBasic Statistics:");
    println!("{:-<40}", "");

    let closes: Vec<f64> = klines.iter().map(|k| k.close).collect();
    let mean_price = closes.iter().sum::<f64>() / closes.len() as f64;
    let min_price = closes.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_price = closes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Compute returns
    let returns: Vec<f64> = klines
        .windows(2)
        .map(|w| w[1].log_return(w[0].close))
        .collect();

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>() / returns.len() as f64;
    let volatility = variance.sqrt();

    println!("Mean Price:     ${:.2}", mean_price);
    println!("Min Price:      ${:.2}", min_price);
    println!("Max Price:      ${:.2}", max_price);
    println!("Mean Return:    {:.4}%", mean_return * 100.0);
    println!("Volatility:     {:.4}%", volatility * 100.0);

    // Fetch ticker data
    println!("\n\nCurrent Ticker:");
    println!("{:-<40}", "");

    let ticker = client.get_ticker("BTCUSDT").await?;
    println!("Symbol:         {}", ticker.symbol);
    println!("Last Price:     ${:.2}", ticker.last_price);
    println!("24h High:       ${:.2}", ticker.high_24h);
    println!("24h Low:        ${:.2}", ticker.low_24h);
    println!("24h Volume:     {:.2}", ticker.volume_24h);
    println!("24h Change:     {:.2}%", ticker.price_change_pct * 100.0);

    // Fetch order book
    println!("\n\nOrder Book (Top 5 levels):");
    println!("{:-<60}", "");

    let orderbook = client.get_order_book("BTCUSDT", 5).await?;
    println!(
        "{:>20} {:^10} {:>20}",
        "Bid Qty", "Price", "Ask Qty"
    );
    println!("{:-<60}", "");

    for i in 0..5 {
        let bid = orderbook.bids.get(i);
        let ask = orderbook.asks.get(i);

        let bid_str = bid.map(|b| format!("{:.4}", b.quantity)).unwrap_or_default();
        let ask_str = ask.map(|a| format!("{:.4}", a.quantity)).unwrap_or_default();
        let bid_price = bid.map(|b| format!("{:.2}", b.price)).unwrap_or_default();
        let ask_price = ask.map(|a| format!("{:.2}", a.price)).unwrap_or_default();

        println!(
            "{:>20} {:>10} | {:<10} {:>20}",
            bid_str, bid_price, ask_price, ask_str
        );
    }

    if let (Some(spread), Some(spread_bps)) = (orderbook.spread(), orderbook.spread_bps()) {
        println!("\nSpread: ${:.2} ({:.2} bps)", spread, spread_bps);
    }

    if let Some(imbalance) = Some(orderbook.depth_imbalance(5)) {
        println!("Depth Imbalance (top 5): {:.2}%", imbalance * 100.0);
    }

    println!("\n\nData fetching complete!");

    Ok(())
}
