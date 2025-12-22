//! Forecast distribution container

use std::collections::HashMap;

/// Container for a probabilistic forecast distribution
#[derive(Debug, Clone)]
pub struct ForecastDistribution {
    /// Samples from the distribution
    pub samples: Vec<f64>,
    /// Mean of the distribution
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Pre-computed quantiles
    pub quantiles: HashMap<i32, f64>, // key is quantile * 100 (e.g., 5 for 0.05)
}

impl ForecastDistribution {
    /// Create a new forecast distribution from samples
    pub fn from_samples(samples: Vec<f64>) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut quantiles = HashMap::new();
        for q in [5, 10, 25, 50, 75, 90, 95] {
            let idx = ((q as f64 / 100.0) * (n - 1.0)) as usize;
            quantiles.insert(q, sorted_samples[idx]);
        }

        Self {
            samples,
            mean,
            std,
            quantiles,
        }
    }

    /// Create from mean and standard deviation (generates samples)
    pub fn from_gaussian(mean: f64, std: f64, num_samples: usize) -> Self {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let normal = Normal::new(mean, std).unwrap();
        let mut rng = thread_rng();

        let samples: Vec<f64> = (0..num_samples).map(|_| normal.sample(&mut rng)).collect();

        Self::from_samples(samples)
    }

    /// Get probability that value is greater than threshold
    pub fn prob_greater_than(&self, threshold: f64) -> f64 {
        let count = self.samples.iter().filter(|&&x| x > threshold).count();
        count as f64 / self.samples.len() as f64
    }

    /// Get probability that value is less than threshold
    pub fn prob_less_than(&self, threshold: f64) -> f64 {
        let count = self.samples.iter().filter(|&&x| x < threshold).count();
        count as f64 / self.samples.len() as f64
    }

    /// Get Value at Risk at given confidence level (e.g., 0.95)
    pub fn var(&self, confidence: f64) -> f64 {
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((1.0 - confidence) * (sorted.len() - 1) as f64) as usize;
        sorted[idx]
    }

    /// Get Conditional VaR (Expected Shortfall)
    pub fn cvar(&self, confidence: f64) -> f64 {
        let var_level = self.var(confidence);
        let tail_samples: Vec<f64> = self
            .samples
            .iter()
            .filter(|&&x| x <= var_level)
            .copied()
            .collect();

        if tail_samples.is_empty() {
            var_level
        } else {
            tail_samples.iter().sum::<f64>() / tail_samples.len() as f64
        }
    }

    /// Get quantile at specified level (0-1)
    pub fn quantile(&self, level: f64) -> f64 {
        let key = (level * 100.0) as i32;
        if let Some(&q) = self.quantiles.get(&key) {
            return q;
        }

        // Compute if not pre-computed
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = (level * (sorted.len() - 1) as f64) as usize;
        sorted[idx]
    }

    /// Get 90% prediction interval
    pub fn interval_90(&self) -> (f64, f64) {
        (self.quantile(0.05), self.quantile(0.95))
    }

    /// Get 50% prediction interval
    pub fn interval_50(&self) -> (f64, f64) {
        (self.quantile(0.25), self.quantile(0.75))
    }

    /// Get median
    pub fn median(&self) -> f64 {
        self.quantile(0.50)
    }

    /// Get skewness
    pub fn skewness(&self) -> f64 {
        let n = self.samples.len() as f64;
        let m3: f64 = self
            .samples
            .iter()
            .map(|x| ((x - self.mean) / self.std).powi(3))
            .sum();
        m3 / n
    }

    /// Get excess kurtosis
    pub fn kurtosis(&self) -> f64 {
        let n = self.samples.len() as f64;
        let m4: f64 = self
            .samples
            .iter()
            .map(|x| ((x - self.mean) / self.std).powi(4))
            .sum();
        m4 / n - 3.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_samples() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let dist = ForecastDistribution::from_samples(samples);

        assert!((dist.mean - 0.5).abs() < 0.01);
        assert!((dist.median() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prob_greater_than() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let dist = ForecastDistribution::from_samples(samples);

        let prob = dist.prob_greater_than(50.0);
        assert!((prob - 0.49).abs() < 0.02);
    }

    #[test]
    fn test_var() {
        let samples: Vec<f64> = (-100..100).map(|i| i as f64 / 100.0).collect();
        let dist = ForecastDistribution::from_samples(samples);

        let var = dist.var(0.95);
        assert!(var < -0.8); // 5th percentile should be around -0.9
    }

    #[test]
    fn test_gaussian() {
        let dist = ForecastDistribution::from_gaussian(0.0, 1.0, 10000);

        assert!((dist.mean).abs() < 0.1);
        assert!((dist.std - 1.0).abs() < 0.1);
    }
}
