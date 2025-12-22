# Chapter 329: Probabilistic Forecasting for Trading

## Overview

Probabilistic forecasting goes beyond traditional point forecasts by providing full probability distributions over future outcomes. Instead of predicting "the price will be $100", probabilistic methods predict "the price will be between $95 and $105 with 90% confidence, with the most likely value being $100". This paradigm shift enables risk-aware decision-making, proper uncertainty quantification, and more robust trading strategies.

## Why Probabilistic Forecasting for Trading?

### The Problem with Point Forecasts

Traditional forecasting models output a single number (point forecast):

```
Input: Historical prices, features
Model: LSTM, XGBoost, Linear Regression
Output: Price tomorrow = $45,250  (single number)
```

**Problems:**
1. **No uncertainty measure** - How confident should we be in this prediction?
2. **Overconfidence** - Point forecasts appear more precise than they are
3. **Poor risk management** - Cannot properly size positions without uncertainty
4. **Ignores distribution shape** - Asymmetric risks (fat tails) are invisible

### The Probabilistic Solution

```
Input: Historical prices, features
Model: DeepAR, Quantile Regression, Gaussian Process
Output: Full probability distribution
        - Mean: $45,250
        - 10th percentile: $44,100
        - 90th percentile: $46,500
        - 95% CI: [$43,800, $46,900]
        - Distribution shape: Slightly right-skewed
```

## Core Concepts

### 1. Point vs Probabilistic Forecasts

```
POINT FORECAST:
┌─────────────────────────────────────────┐
│                 │                        │
│                 ●  $45,250               │
│                 │                        │
│    Yesterday    Today    Tomorrow        │
└─────────────────────────────────────────┘

PROBABILISTIC FORECAST:
┌─────────────────────────────────────────┐
│                       ╭───────╮          │
│                      ╱ 90% CI  ╲         │
│                     ╱ ┌─────┐   ╲        │
│                    ╱  │ 50% │    ╲       │
│         ●─────────╱───│  CI │─────╲      │
│                       └─────┘            │
│    Yesterday    Today    Tomorrow        │
│                                          │
│  We see: Most likely value + uncertainty │
└─────────────────────────────────────────┘
```

### 2. Quantile Regression

Quantile regression estimates specific percentiles of the conditional distribution:

```python
# Standard regression: E[Y|X] = f(X)  (mean only)
# Quantile regression: Q_τ[Y|X] = f_τ(X)  (any quantile τ)

# Example quantiles for trading:
τ = 0.05  → 5th percentile (extreme downside)
τ = 0.25  → 25th percentile (moderate downside)
τ = 0.50  → 50th percentile (median)
τ = 0.75  → 75th percentile (moderate upside)
τ = 0.95  → 95th percentile (extreme upside)
```

**Quantile Loss (Pinball Loss):**

```
L_τ(y, q) = (τ - 1{y < q}) × (y - q)

Where:
- y = actual value
- q = predicted quantile
- τ = target quantile level
- 1{y < q} = indicator function

For τ = 0.5: Equivalent to MAE (median regression)
For τ = 0.9: Penalizes under-prediction more heavily
```

### 3. Distributional Forecasting

Instead of predicting quantiles separately, model the entire distribution:

```
Parametric approach:
┌────────────────────────────────────────────────┐
│ Assume Y ~ Distribution(θ₁, θ₂, ...)           │
│                                                 │
│ Normal:     Y ~ N(μ, σ²)    → predict μ, σ     │
│ Student-t:  Y ~ t(μ, σ, ν)  → predict μ, σ, ν  │
│ Mixture:    Y ~ Σᵢ πᵢ N(μᵢ, σᵢ²)               │
│                                                 │
│ Non-parametric:                                 │
│ - Histogram forecasts                           │
│ - Kernel density estimation                     │
│ - Normalizing flows                             │
└────────────────────────────────────────────────┘
```

### 4. DeepAR: Deep Autoregressive Recurrent Networks

DeepAR (Amazon) is a state-of-the-art probabilistic forecasting method:

```
Architecture:
┌────────────────────────────────────────────────────────────┐
│                        DeepAR                               │
│                                                             │
│  Encoder (LSTM/GRU):                                        │
│  ┌────┐    ┌────┐    ┌────┐    ┌────┐                       │
│  │ h₁ │ → │ h₂ │ → │ h₃ │ → │ hₜ │                       │
│  └────┘    └────┘    └────┘    └────┘                       │
│     ↑         ↑         ↑         ↑                         │
│  [x₁,z₁]   [x₂,z₂]   [x₃,z₃]   [xₜ,zₜ]                     │
│                                                             │
│  Decoder (Autoregressive):                                  │
│  ┌────┐    ┌────┐    ┌────┐                                 │
│  │ hₜ │ → │hₜ₊₁│ → │hₜ₊₂│ → ...                            │
│  └────┘    └────┘    └────┘                                 │
│     ↓         ↓         ↓                                   │
│  θₜ₊₁      θₜ₊₂      θₜ₊₃                                   │
│     ↓         ↓         ↓                                   │
│  Sample   Sample   Sample   → Monte Carlo forecasts         │
│                                                             │
│  xᵢ = target values (price, returns)                        │
│  zᵢ = covariates (volume, indicators, time features)        │
│  θᵢ = distribution parameters (μ, σ for Gaussian)           │
└────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Learns from multiple related time series simultaneously
- Handles missing values and varying history lengths
- Produces coherent probabilistic forecasts via sampling
- Captures cross-series patterns (e.g., correlated assets)

## Proper Scoring Rules

### Why Scoring Rules Matter

To evaluate probabilistic forecasts, we need **proper scoring rules** - metrics that are optimized when the forecaster reports their true belief:

```
Proper scoring rule: S(P, y)
- P = predicted probability distribution
- y = observed outcome
- Maximized (or minimized) when P equals true distribution

Key property: Cannot "game" the metric by reporting something
             other than your true belief
```

### CRPS (Continuous Ranked Probability Score)

CRPS is the gold standard for evaluating probabilistic forecasts of continuous variables:

```
CRPS(F, y) = ∫_{-∞}^{∞} (F(x) - 1{y ≤ x})² dx

Where:
- F(x) = predicted CDF at point x
- y = observed value
- 1{y ≤ x} = step function (0 before y, 1 after y)

Interpretation:
- Measures the "distance" between predicted CDF and perfect forecast
- Lower is better
- Reduces to MAE when forecast is deterministic
- Decomposes into: CRPS = Reliability + Resolution - Uncertainty
```

**CRPS Visualization:**

```
┌────────────────────────────────────────────────────┐
│  1.0 ─────────────────────────●━━━━━━━━━━━━━━━━━━  │
│       │                     ╱                      │
│       │                    ╱                       │
│  CDF  │                  ╱╱     ┌──────────────┐   │
│       │                ╱╱       │ Shaded area  │   │
│  0.0 ─●═══════════════╱        │ = CRPS       │   │
│       │              ↑         └──────────────┘   │
│       └──────────────┼────────────────────────    │
│                    y=observed                      │
│                                                    │
│  ── = Predicted CDF (F)                           │
│  ── = Step function at observed value              │
└────────────────────────────────────────────────────┘
```

### Log Score (Logarithmic Score)

```
LogScore(p, y) = log(p(y))

Where:
- p(y) = predicted probability density at observed value y
- Higher is better (or -LogScore: lower is better)

Properties:
- Heavily penalizes confident wrong predictions
- Sensitive to tail behavior
- Also called "negative log-likelihood" when minimized
```

### Comparison of Scoring Rules

| Scoring Rule | Sensitivity to Tails | Locality | Decomposable |
|--------------|---------------------|----------|--------------|
| CRPS | Moderate | Yes | Yes |
| Log Score | High | Yes | No |
| Brier Score | N/A (classification) | Yes | Yes |
| Quantile Score | Quantile-specific | Yes | Yes |

## Calibration

### What is Calibration?

A probabilistic forecast is **calibrated** if predicted probabilities match observed frequencies:

```
Calibration Definition:
If we predict "80% chance of positive return" many times,
then positive returns should occur ~80% of those times.

Formally: P(Y ≤ y | F(y) = p) = p  for all p ∈ [0,1]
```

### PIT (Probability Integral Transform)

```
For a well-calibrated forecast:
- Transform observations using predicted CDF: u = F(y)
- u should be uniformly distributed on [0, 1]

Checking calibration:
1. Compute PIT values: uᵢ = Fᵢ(yᵢ) for each prediction
2. Plot histogram of uᵢ values
3. Should be flat (uniform) if well-calibrated

Common issues:
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  ████████   │ │   ██████    │ │ ██      ██  │
│  ████████   │ │  ████████   │ │████    ████ │
│  ████████   │ │ ██████████  │ │████████████ │
└─────────────┘ └─────────────┘ └─────────────┘
  Well-calibrated  Overconfident   Underconfident
  (uniform)        (U-shaped)      (inverse U)
```

### Calibration vs. Sharpness

```
Good forecasts need BOTH:

CALIBRATION: Probabilities are honest
SHARPNESS: Predictions are precise (narrow intervals)

Trade-off:
┌─────────────────────────────────────────────────┐
│                                                  │
│  Calibrated but not sharp: "50% chance up/down" │
│  (Always predict 50-50 → calibrated but useless)│
│                                                  │
│  Sharp but not calibrated: "99% confident"      │
│  (Always confident → precise but wrong)         │
│                                                  │
│  GOAL: Sharp AND calibrated                     │
└─────────────────────────────────────────────────┘
```

## Decision Making Under Uncertainty

### Expected Utility Theory

```
Optimal Action = argmax_a E[U(outcome | action a)]

Where:
- U = utility function (e.g., profit, risk-adjusted return)
- Expectation taken over predictive distribution

Example:
- Predictive distribution: Price ~ N($100, $5²)
- Action: Buy if expected profit > 0
- Utility: U = Position × (Price - Entry) - Transaction_costs
```

### Value at Risk (VaR) from Probabilistic Forecasts

```
VaR_α = quantile_α of predicted return distribution

Example:
Predicted return distribution for tomorrow:
- Mean: +0.5%
- Std: 2.0%
- VaR(95%): -2.8% (5% chance of losing more than 2.8%)

Use for:
- Position sizing
- Risk limits
- Margin requirements
```

### Expected Shortfall (CVaR)

```
ES_α = E[Loss | Loss > VaR_α]

More conservative than VaR:
- Averages over all losses beyond VaR
- Better captures tail risk
- Coherent risk measure (VaR is not)

From probabilistic forecast:
ES = (1/(1-α)) × ∫_{-∞}^{VaR_α} x × f(x) dx
```

## Kelly Criterion with Probabilistic Forecasts

### Classic Kelly Criterion

```
f* = (p × b - q) / b

Where:
- f* = optimal fraction of capital to bet
- p = probability of winning
- q = 1 - p = probability of losing
- b = odds (net profit per unit bet if win)

Example:
- 60% chance of +10% return (p=0.6, b=0.10)
- 40% chance of -5% return (q=0.4)
- Kelly fraction: (0.6 × 0.10 - 0.4 × 0.05) / 0.10 = 40%
```

### Kelly with Full Distribution

```
For continuous distributions:

f* = argmax_f E[log(1 + f × R)]

Where:
- R = return random variable from predictive distribution
- Expectation over predicted distribution
- Solved numerically when distribution is complex

Practical implementation:
1. Sample N returns from predictive distribution
2. For each candidate f: compute mean log return
3. Find f that maximizes mean log return
4. Apply "fractional Kelly" (e.g., f*/2) for safety
```

### Kelly with Uncertainty in Probabilities

```
When probabilities themselves are uncertain:

Bayesian Kelly:
- p has a posterior distribution (not a point estimate)
- Integrate over uncertainty in p

f*_robust = E_p[Kelly(p)] or max_p min_outcome Kelly(p)

This naturally leads to more conservative positions
when we're uncertain about our predictions!
```

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               PROBABILISTIC FORECASTING MODEL                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Time Series Features:                                     │   │
│  │   - Historical returns (1m, 5m, 15m, 1h, 4h, 1d)         │   │
│  │   - Volume profile and VWAP                               │   │
│  │   - Volatility measures (realized, implied)               │   │
│  │   - Order book imbalance                                  │   │
│  │ Calendar Features:                                        │   │
│  │   - Hour of day, day of week (cyclical encoding)         │   │
│  │   - Time since market events                              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  TEMPORAL ENCODER (LSTM/Transformer)                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Captures temporal patterns and dependencies               │   │
│  │ Output: Context vector h_t                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  DISTRIBUTION HEAD                                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Option A: Parametric Distribution                         │   │
│  │   h_t → Linear → [μ, log(σ), ν, ...]                     │   │
│  │   Return distribution: Student-t(μ, σ, ν)                │   │
│  │                                                           │   │
│  │ Option B: Quantile Outputs                                │   │
│  │   h_t → Linear → [q_0.05, q_0.25, q_0.50, q_0.75, q_0.95]│   │
│  │                                                           │   │
│  │ Option C: Mixture Density Network                         │   │
│  │   h_t → Linear → [π_k, μ_k, σ_k] for k=1..K components   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              ↓                                   │
│  OUTPUT: Full Predictive Distribution                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ - PDF/CDF at any point                                    │   │
│  │ - Samples via Monte Carlo                                 │   │
│  │ - Quantiles at any level                                  │   │
│  │ - Mean, variance, skewness, kurtosis                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Trading Strategy

### Signal Generation with Uncertainty

```python
def generate_probabilistic_signals(forecast_distribution, current_price):
    """
    Generate trading signals using probabilistic forecasts.
    """
    # Extract distribution statistics
    mean_return = forecast_distribution.mean()
    std_return = forecast_distribution.std()
    var_95 = forecast_distribution.quantile(0.05)
    upside_prob = forecast_distribution.prob_greater_than(0)

    # Kelly-optimal position sizing
    kelly_fraction = compute_kelly(forecast_distribution)

    # Risk-adjusted signal
    signal = SignalType.HOLD
    position_size = 0.0

    if upside_prob > 0.60 and mean_return > 0.005:
        # Strong bullish signal
        signal = SignalType.LONG
        # Scale position by confidence and Kelly
        position_size = min(kelly_fraction * 0.5, MAX_POSITION)

    elif upside_prob < 0.40 and mean_return < -0.005:
        # Strong bearish signal
        signal = SignalType.SHORT
        position_size = min(-kelly_fraction * 0.5, MAX_POSITION)

    # Adjust for forecast uncertainty
    uncertainty_penalty = std_return / BASELINE_VOLATILITY
    position_size *= max(0.5, 1.0 - uncertainty_penalty)

    return Signal(
        direction=signal,
        size=position_size,
        confidence=abs(upside_prob - 0.5) * 2,
        var_95=var_95,
        expected_return=mean_return
    )
```

### Position Sizing with VaR Constraints

```python
def size_position_with_var(forecast, risk_budget, current_portfolio):
    """
    Size position such that portfolio VaR stays within budget.
    """
    portfolio_var = compute_portfolio_var(current_portfolio, forecast)

    max_size = risk_budget / forecast.var_95
    kelly_size = compute_kelly(forecast)

    # Take minimum of Kelly and VaR-constrained size
    position_size = min(kelly_size, max_size)

    return position_size
```

## Key Metrics

### Forecast Quality

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| CRPS | See above | Lower = better probabilistic accuracy |
| Log Score | log(p(y)) | Higher = better; sensitive to tails |
| Calibration Error | |F(y) - U| | Lower = better calibrated |
| Sharpness | Avg interval width | Lower = more precise |
| Coverage | % within 90% CI | Should equal 90% |

### Trading Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 2.0 | Risk-adjusted return |
| Sortino Ratio | > 2.5 | Downside risk-adjusted |
| Max Drawdown | < 15% | Largest peak-to-trough |
| Win Rate | > 55% | Profitable trade percentage |
| Profit Factor | > 1.5 | Gross profit / Gross loss |
| Kelly Growth | > 10% | Annualized Kelly-optimal growth |

## Implementation Details

### Data Requirements

```
Cryptocurrency Market Data:
├── OHLCV data (1-minute resolution minimum)
│   └── Multiple timeframes for features
├── Order book snapshots
│   └── Depth and imbalance metrics
├── Trade flow data
│   └── Buy/sell pressure indicators
└── Volatility data
    ├── Historical realized volatility
    └── Implied volatility (if available)

Feature Engineering:
├── Returns at multiple horizons
├── Rolling volatility measures
├── Volume-weighted features
├── Technical indicators (RSI, MACD, BB)
└── Calendar/seasonal features
```

### Training Configuration

```yaml
model:
  type: "deepar"  # or "quantile_regression", "mixture_density"
  hidden_size: 128
  num_layers: 2
  dropout: 0.1
  distribution: "student_t"  # or "gaussian", "negative_binomial"

quantiles:  # if using quantile regression
  - 0.05
  - 0.10
  - 0.25
  - 0.50
  - 0.75
  - 0.90
  - 0.95

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

data:
  sequence_length: 120  # 2 hours of 1-minute data
  prediction_horizons: [5, 15, 60]  # minutes ahead
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
```

## Advantages of Probabilistic Forecasting

| Aspect | Point Forecasts | Probabilistic Forecasts |
|--------|-----------------|------------------------|
| Uncertainty | Hidden | Explicit |
| Risk Management | Manual VaR | Integrated |
| Position Sizing | Heuristic | Principled (Kelly) |
| Decision Making | Threshold-based | Expected utility |
| Calibration | N/A | Measurable |
| Tail Risk | Ignored | Captured |
| Model Selection | MSE/MAE only | CRPS, Log Score |

## Directory Structure

```
329_probabilistic_forecasting/
├── README.md                    # This file
├── README.ru.md                 # Russian translation
├── readme.simple.md             # Beginner-friendly explanation
├── readme.simple.ru.md          # Russian beginner version
├── python/                      # Python implementation
│   ├── requirements.txt
│   ├── config.yaml
│   ├── data_fetcher.py          # Bybit data via CCXT
│   ├── features.py              # Feature engineering
│   ├── models/
│   │   ├── quantile_regression.py
│   │   ├── deepar.py
│   │   └── mixture_density.py
│   ├── scoring.py               # CRPS, Log Score, calibration
│   ├── strategy.py              # Trading strategy
│   ├── backtest.py              # Backtesting engine
│   └── examples/
│       ├── train_model.py
│       ├── evaluate_forecasts.py
│       └── run_backtest.py
└── rust_probabilistic/          # Rust implementation
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── api/                 # Bybit API client
    │   ├── distributions/       # Probability distributions
    │   ├── models/              # Forecasting models
    │   ├── scoring/             # Proper scoring rules
    │   ├── strategy/            # Trading strategy
    │   └── backtest/            # Backtesting engine
    └── examples/
        ├── fetch_data.rs
        ├── quantile_forecast.rs
        └── backtest.rs
```

## References

1. **Probabilistic Forecasting** (Gneiting & Katzfuss, 2014)
   - Annual Review of Statistics: Foundations of probabilistic forecasting

2. **DeepAR: Probabilistic Forecasting with Autoregressive RNNs** (Salinas et al., 2020)
   - https://arxiv.org/abs/1704.04110

3. **Strictly Proper Scoring Rules, Prediction, and Estimation** (Gneiting & Raftery, 2007)
   - Journal of the American Statistical Association

4. **Quantile Regression** (Koenker, 2005)
   - Cambridge University Press

5. **Kelly Criterion in Portfolio Optimization** (MacLean et al., 2011)
   - The Kelly Capital Growth Investment Criterion

6. **Calibration of Probabilistic Forecasts** (Gneiting et al., 2007)
   - https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf

## Difficulty Level

**Advanced** - Requires understanding of:
- Probability theory and statistics
- Time series analysis
- Bayesian inference concepts
- Risk management principles
- Deep learning architectures

## Disclaimer

This chapter is for **educational purposes only**. Cryptocurrency trading involves substantial risk. The strategies described here have not been validated in live trading and should be thoroughly tested before any real-world application. Past performance does not guarantee future results. Probabilistic forecasts provide uncertainty estimates but cannot eliminate risk.
