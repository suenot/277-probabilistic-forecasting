//! Student-t distribution for heavy-tailed forecasts

use std::f64::consts::PI;

/// Student-t distribution
#[derive(Debug, Clone, Copy)]
pub struct StudentTDistribution {
    /// Location parameter
    pub mu: f64,
    /// Scale parameter
    pub sigma: f64,
    /// Degrees of freedom
    pub df: f64,
}

impl StudentTDistribution {
    /// Create a new Student-t distribution
    pub fn new(mu: f64, sigma: f64, df: f64) -> Self {
        assert!(sigma > 0.0, "Sigma must be positive");
        assert!(df > 0.0, "Degrees of freedom must be positive");
        Self { mu, sigma, df }
    }

    /// Standard Student-t with given degrees of freedom
    pub fn standard(df: f64) -> Self {
        Self {
            mu: 0.0,
            sigma: 1.0,
            df,
        }
    }

    /// Probability density function
    pub fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        let v = self.df;

        let coef = gamma((v + 1.0) / 2.0) / (gamma(v / 2.0) * (v * PI).sqrt());
        let term = (1.0 + z * z / v).powf(-(v + 1.0) / 2.0);

        coef * term / self.sigma
    }

    /// Log probability density
    pub fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        let v = self.df;

        ln_gamma((v + 1.0) / 2.0) - ln_gamma(v / 2.0) - 0.5 * (v * PI).ln() - self.sigma.ln()
            - (v + 1.0) / 2.0 * (1.0 + z * z / v).ln()
    }

    /// Get the mean (only defined for df > 1)
    pub fn mean(&self) -> Option<f64> {
        if self.df > 1.0 {
            Some(self.mu)
        } else {
            None
        }
    }

    /// Get the variance (only defined for df > 2)
    pub fn variance(&self) -> Option<f64> {
        if self.df > 2.0 {
            Some(self.sigma * self.sigma * self.df / (self.df - 2.0))
        } else {
            None
        }
    }

    /// Generate samples using rejection sampling
    pub fn sample(&self, n: usize) -> Vec<f64> {
        use rand::thread_rng;
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = thread_rng();
        let normal = StandardNormal;

        (0..n)
            .map(|_| {
                // Generate using ratio of normal to chi-squared
                let z: f64 = normal.sample(&mut rng);
                let chi2: f64 = (0..self.df.round() as usize)
                    .map(|_| {
                        let u: f64 = normal.sample(&mut rng);
                        u * u
                    })
                    .sum();

                let t = z / (chi2 / self.df).sqrt();
                self.mu + self.sigma * t
            })
            .collect()
    }
}

/// Gamma function approximation using Lanczos approximation
fn gamma(x: f64) -> f64 {
    ln_gamma(x).exp()
}

/// Log gamma function (Lanczos approximation)
fn ln_gamma(x: f64) -> f64 {
    let g = 7;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for i in 1..=(g + 1) {
            a += c[i] / (x + i as f64);
        }
        let t = x + g as f64 + 0.5;
        0.5 * (2.0 * PI).ln() + (t.ln() * (x + 0.5)) - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_t() {
        // With high df, should approach standard normal
        let t = StudentTDistribution::standard(100.0);
        let pdf_0 = t.pdf(0.0);

        // PDF at 0 for standard normal is ~0.3989
        assert!((pdf_0 - 0.3989).abs() < 0.01);
    }

    #[test]
    fn test_heavy_tails() {
        let t3 = StudentTDistribution::standard(3.0);
        let normal_approx = StudentTDistribution::standard(100.0);

        // Student-t should have heavier tails
        let t3_tail = t3.pdf(3.0);
        let norm_tail = normal_approx.pdf(3.0);

        assert!(t3_tail > norm_tail);
    }

    #[test]
    fn test_variance() {
        let t = StudentTDistribution::new(0.0, 1.0, 5.0);

        // Variance = sigma^2 * df / (df - 2) = 1 * 5/3 = 1.667
        let var = t.variance().unwrap();
        assert!((var - 1.667).abs() < 0.01);
    }

    #[test]
    fn test_sample() {
        let t = StudentTDistribution::new(0.0, 1.0, 10.0);
        let samples = t.sample(10000);

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.2);
    }
}
