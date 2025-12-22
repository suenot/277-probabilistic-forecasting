//! Gaussian (Normal) distribution

use std::f64::consts::PI;

/// Gaussian distribution
#[derive(Debug, Clone, Copy)]
pub struct GaussianDistribution {
    /// Mean
    pub mu: f64,
    /// Standard deviation
    pub sigma: f64,
}

impl GaussianDistribution {
    /// Create a new Gaussian distribution
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "Sigma must be positive");
        Self { mu, sigma }
    }

    /// Standard normal distribution N(0, 1)
    pub fn standard() -> Self {
        Self { mu: 0.0, sigma: 1.0 }
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        (-0.5 * z * z).exp() / (self.sigma * (2.0 * PI).sqrt())
    }

    /// Cumulative distribution function
    pub fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        0.5 * (1.0 + erf(z / (2.0_f64).sqrt()))
    }

    /// Log probability density
    pub fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - self.sigma.ln() - 0.5 * (2.0 * PI).ln()
    }

    /// Quantile function (inverse CDF)
    pub fn quantile(&self, p: f64) -> f64 {
        assert!((0.0..=1.0).contains(&p), "p must be in [0, 1]");
        self.mu + self.sigma * inv_erf(2.0 * p - 1.0) * (2.0_f64).sqrt()
    }

    /// Generate samples
    pub fn sample(&self, n: usize) -> Vec<f64> {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let normal = Normal::new(self.mu, self.sigma).unwrap();
        let mut rng = thread_rng();

        (0..n).map(|_| normal.sample(&mut rng)).collect()
    }

    /// Compute CRPS for this distribution (closed form)
    pub fn crps(&self, observation: f64) -> f64 {
        let z = (observation - self.mu) / self.sigma;
        let phi_z = standard_normal_pdf(z);
        let big_phi_z = standard_normal_cdf(z);

        self.sigma * (z * (2.0 * big_phi_z - 1.0) + 2.0 * phi_z - 1.0 / PI.sqrt())
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Horner form of approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Inverse error function approximation
fn inv_erf(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }

    let a = 0.147;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let ln_term = (1.0 - x * x).ln();
    let term1 = 2.0 / (PI * a) + ln_term / 2.0;
    let term2 = ln_term / a;

    sign * (term1 * term1 - term2).sqrt().sqrt() - term1.sqrt()
}

/// Standard normal PDF
fn standard_normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Standard normal CDF
fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / (2.0_f64).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_normal() {
        let n = GaussianDistribution::standard();

        // PDF at 0 should be 1/sqrt(2*pi) ≈ 0.3989
        assert!((n.pdf(0.0) - 0.3989).abs() < 0.001);

        // CDF at 0 should be 0.5
        assert!((n.cdf(0.0) - 0.5).abs() < 0.001);

        // Quantile at 0.5 should be 0
        assert!((n.quantile(0.5)).abs() < 0.001);
    }

    #[test]
    fn test_crps() {
        let n = GaussianDistribution::new(0.0, 1.0);

        // CRPS for observation at mean should be around 0.564
        let crps = n.crps(0.0);
        assert!((crps - 0.564).abs() < 0.01);
    }

    #[test]
    fn test_sample() {
        let n = GaussianDistribution::new(100.0, 10.0);
        let samples = n.sample(10000);

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 100.0).abs() < 1.0);
    }
}
