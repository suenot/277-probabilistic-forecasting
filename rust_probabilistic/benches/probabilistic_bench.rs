//! Benchmarks for Probabilistic Forecasting
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use probabilistic_forecasting::prelude::*;
use probabilistic_forecasting::scoring::crps::{compute_crps, compute_crps_gaussian};
use rand::Rng;

/// Benchmark ForecastDistribution creation from samples
fn bench_forecast_from_samples(c: &mut Criterion) {
    let mut group = c.benchmark_group("forecast_creation");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let samples: Vec<f64> = (0..*size).map(|i| i as f64 * 0.001).collect();

        group.bench_with_input(BenchmarkId::new("from_samples", size), size, |b, _| {
            b.iter(|| ForecastDistribution::from_samples(black_box(samples.clone())))
        });
    }

    group.finish();
}

/// Benchmark Gaussian distribution creation
fn bench_gaussian_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_creation");

    for n_samples in [100, 500, 1000, 5000].iter() {
        group.bench_with_input(
            BenchmarkId::new("from_gaussian", n_samples),
            n_samples,
            |b, &n| {
                b.iter(|| {
                    ForecastDistribution::from_gaussian(
                        black_box(0.0),
                        black_box(1.0),
                        black_box(n),
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark CRPS computation
fn bench_crps(c: &mut Criterion) {
    let mut group = c.benchmark_group("crps");

    // Empirical CRPS
    for size in [100, 500, 1000, 5000].iter() {
        let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, *size);

        group.bench_with_input(BenchmarkId::new("empirical", size), size, |b, _| {
            b.iter(|| compute_crps(black_box(&forecast), black_box(0.5)))
        });
    }

    // Closed-form Gaussian CRPS
    group.bench_function("gaussian_closed_form", |b| {
        b.iter(|| {
            compute_crps_gaussian(black_box(0.0), black_box(1.0), black_box(0.5))
        })
    });

    group.finish();
}

/// Benchmark quantile computation
fn bench_quantiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantiles");

    let forecast = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);
    let quantiles = vec![0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95];

    group.bench_function("single_quantile", |b| {
        b.iter(|| forecast.quantile(black_box(0.50)))
    });

    group.bench_function("multiple_quantiles", |b| {
        b.iter(|| {
            for &q in &quantiles {
                black_box(forecast.quantile(q));
            }
        })
    });

    group.finish();
}

/// Benchmark VaR and CVaR computation
fn bench_risk_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_metrics");

    let forecast = ForecastDistribution::from_gaussian(-0.01, 0.03, 10000);

    group.bench_function("var_95", |b| {
        b.iter(|| forecast.var(black_box(0.95)))
    });

    group.bench_function("cvar_95", |b| {
        b.iter(|| forecast.cvar(black_box(0.95)))
    });

    group.finish();
}

/// Benchmark probability computations
fn bench_probabilities(c: &mut Criterion) {
    let mut group = c.benchmark_group("probabilities");

    let forecast = ForecastDistribution::from_gaussian(0.01, 0.02, 10000);

    group.bench_function("prob_greater_than", |b| {
        b.iter(|| forecast.prob_greater_than(black_box(0.0)))
    });

    group.bench_function("prob_between", |b| {
        b.iter(|| forecast.prob_between(black_box(-0.01), black_box(0.01)))
    });

    group.finish();
}

/// Benchmark Kelly fraction computation
fn bench_kelly(c: &mut Criterion) {
    use probabilistic_forecasting::strategy::kelly::{kelly_fraction, fractional_kelly};

    let mut group = c.benchmark_group("kelly");

    let forecast = ForecastDistribution::from_gaussian(0.02, 0.03, 10000);

    group.bench_function("kelly_fraction", |b| {
        b.iter(|| kelly_fraction(black_box(&forecast)))
    });

    group.bench_function("fractional_kelly", |b| {
        b.iter(|| fractional_kelly(black_box(&forecast), black_box(0.5)))
    });

    group.finish();
}

/// Benchmark batch CRPS computation
fn bench_batch_crps(c: &mut Criterion) {
    use probabilistic_forecasting::scoring::crps::mean_crps;

    let mut group = c.benchmark_group("batch_crps");

    for n_forecasts in [10, 50, 100, 500].iter() {
        let forecasts: Vec<_> = (0..*n_forecasts)
            .map(|_| ForecastDistribution::from_gaussian(0.0, 1.0, 500))
            .collect();
        let observations: Vec<f64> = (0..*n_forecasts)
            .map(|i| (i as f64 - *n_forecasts as f64 / 2.0) * 0.1)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("mean_crps", n_forecasts),
            n_forecasts,
            |b, _| {
                b.iter(|| mean_crps(black_box(&forecasts), black_box(&observations)))
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_forecast_from_samples,
    bench_gaussian_creation,
    bench_crps,
    bench_quantiles,
    bench_risk_metrics,
    bench_probabilities,
    bench_kelly,
    bench_batch_crps,
);

criterion_main!(benches);
