//! Sensor model:
//! ω̃(t) = (1 + s)·ω(t) + b(t) + n(t)
//! b(t) = b₀ + b_drift(t)
//! ḃ_drift(t) = −(1/τ_c)·b_drift(t) + w(t)
//! β = exp(−Δt/τ_c)
//! b_drift[k] = β · b_drift[k−1] + √(σ²_b·(1 − β²)) · ξ[k],     ξ ~ N(0,1)

use nalgebra::{Vector3, vector};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};

pub mod ins;

fn std_normal(rng: &mut StdRng) -> Vector3<f64> {
    Vector3::from_fn(|_, _| StandardNormal.sample(rng))
}

fn update_drift(drift: &mut Vector3<f64>, sigma: f64, beta: f64, rng: &mut StdRng) {
    let vol = (sigma.powi(2) * (1. - beta.powi(2))).sqrt();
    *drift *= beta;
    *drift += vol * std_normal(rng);
}

fn generate_measurement(
    true_value: &Vector3<f64>,
    white_noise_density: f64,
    scale: f64,
    b0: &Vector3<f64>,
    b_drift: &Vector3<f64>,
    dt: f64,
    rng: &mut StdRng,
) -> Vector3<f64> {
    let noise = white_noise_density / dt.sqrt() * std_normal(rng);
    (1.0 + scale) * true_value + b0 + b_drift + noise
}

/// Per-sensor error parameters
pub struct ImuErrorParams {
    pub scale: f64,               // s, dimensionless
    pub turn_on_bias_sigma: f64,  // 1σ of b₀, drawn once per IMU
    pub bias_instab_sigma: f64,   // σ_b, stationary std
    pub bias_corr_time: f64,      // τ_c, correlation time
    pub white_noise_density: f64, // N for n(t) process
}

impl ImuErrorParams {
    pub fn new(
        scale: f64,               // s, dimensionless
        turn_on_bias_sigma: f64,  // 1σ of b₀, drawn once per IMU
        bias_instab_sigma: f64,   // σ_b, stationary std
        bias_corr_time: f64,      // τ_c, correlation time
        white_noise_density: f64, // n(t) process
    ) -> Self {
        Self {
            scale,
            turn_on_bias_sigma,
            bias_instab_sigma,
            bias_corr_time,
            white_noise_density,
        }
    }
}

/// One IMU reading: the gyro and accelerometer measurement vectors.
#[derive(Debug, Clone, Copy)]
pub struct ImuSample {
    pub gyro: Vector3<f64>,
    pub accel: Vector3<f64>,
}

pub struct SyntheticImu {
    pub gyro: ImuErrorParams,
    pub accel: ImuErrorParams,
    pub dt: f64, // Δt

    gyro_b0: Vector3<f64>,
    accel_b0: Vector3<f64>,
    gyro_drift: Vector3<f64>,
    accel_drift: Vector3<f64>,
    beta_gyro: f64,
    beta_accel: f64,
    rng: StdRng,
}

impl SyntheticImu {
    pub fn new(gyro: ImuErrorParams, accel: ImuErrorParams, dt: f64, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let gyro_b0 =
            gyro.turn_on_bias_sigma * Vector3::from_fn(|_, _| StandardNormal.sample(&mut rng));

        let accel_b0 =
            accel.turn_on_bias_sigma * Vector3::from_fn(|_, _| StandardNormal.sample(&mut rng));

        let gyro_drift = vector![0.0, 0.0, 0.0];
        let accel_drift = vector![0.0, 0.0, 0.0];

        let beta_gyro = (-dt / gyro.bias_corr_time).exp();
        let beta_accel = (-dt / accel.bias_corr_time).exp();

        Self {
            gyro,
            accel,
            dt,
            gyro_b0,
            accel_b0,
            gyro_drift,
            accel_drift,
            beta_gyro,
            beta_accel,
            rng,
        }
    }

    pub fn sample(&mut self, gyro_true: Vector3<f64>, accel_true: Vector3<f64>) -> ImuSample {
        update_drift(
            &mut self.gyro_drift,
            self.gyro.bias_instab_sigma,
            self.beta_gyro,
            &mut self.rng,
        );

        update_drift(
            &mut self.accel_drift,
            self.accel.bias_instab_sigma,
            self.beta_accel,
            &mut self.rng,
        );

        let gyro_meas = generate_measurement(
            &gyro_true,
            self.gyro.white_noise_density,
            self.gyro.scale,
            &self.gyro_b0,
            &self.gyro_drift,
            self.dt,
            &mut self.rng,
        );

        let accel_meas = generate_measurement(
            &accel_true,
            self.accel.white_noise_density,
            self.accel.scale,
            &self.accel_b0,
            &self.accel_drift,
            self.dt,
            &mut self.rng,
        );

        ImuSample {
            gyro: gyro_meas,
            accel: accel_meas,
        }
    }

    /// Current true additive accelerometer bias, `b₀ + b_drift(t)` (m/s²) — the
    /// quantity an estimator's accel-bias state should track. For validating a
    /// downstream filter's bias estimate against ground truth.
    pub fn accel_bias(&self) -> Vector3<f64> {
        self.accel_b0 + self.accel_drift
    }

    /// Current true additive gyro bias, `b₀ + b_drift(t)` (rad/s).
    pub fn gyro_bias(&self) -> Vector3<f64> {
        self.gyro_b0 + self.gyro_drift
    }
}

pub mod allan {
    use std::iter::once;

    /// One point on an Allan-deviation curve.
    #[derive(Debug, Clone, Copy)]
    pub struct AllanPoint {
        pub tau: f64,       // averaging time τ (s)
        pub deviation: f64, // Allan deviation σ_A(τ)
    }

    #[inline]
    fn cluster_mean(arr: &[f64], start: usize, size: usize) -> f64 {
        (arr[start + size] - arr[start]) / size as f64
    }

    pub fn allan_deviation(data: &[f64], dt: f64) -> Vec<AllanPoint> {
        let cum_sum: Vec<f64> = once(&0.0)
            .chain(data.iter())
            .scan(0., |sum, &x| {
                *sum += x;
                Some(*sum)
            })
            .collect();

        // Log-spaced averaging lengths m
        let n = data.len();
        let max_m = n / 2;
        let mut ms = Vec::new();
        let mut m = 1usize;
        while m <= max_m {
            ms.push(m);
            let next = ((m as f64) * 1.2).ceil() as usize;
            m = next.max(m + 1);
        }

        let mut out = Vec::with_capacity(ms.len());
        for &m in &ms {
            let tau = m as f64 * dt;

            let count = n - 2 * m + 1;
            let mut acc = 0.0;
            for j in 0..count {
                let diff = cluster_mean(&cum_sum, j + m, m) - cluster_mean(&cum_sum, j, m);
                acc += diff.powi(2);
            }
            out.push(AllanPoint {
                tau,
                deviation: (acc / (2.0 * count as f64)).sqrt(),
            })
        }
        out
    }

    /// Read the white-noise coefficient
    /// σ_A(τ) = N / √τ    =>    σ_A(1) = N
    pub fn read_arw(curve: &[AllanPoint]) -> f64 {
        curve
            .iter()
            .min_by(|a, b| {
                (a.tau - 1.)
                    .abs()
                    .partial_cmp(&(b.tau - 1.).abs())
                    .expect("cannot compare")
            })
            .map(|p| p.deviation)
            .expect("no value found")
    }

    /// Read the bias-instability coefficient B: the curve's minimum
    /// equals 0.664·B, so B = min(σ_A) / 0.664.
    pub fn read_bias_instability(curve: &[AllanPoint]) -> f64 {
        // The floor is the bathtub *bottom*: the first local minimum, where the
        // −1/2 white-noise arm turns up into the bias hump. A global minimum
        // would instead catch the noisy long-τ tail (e.g. a Gauss-Markov bias,
        // whose Allan deviation decays again at large τ). Fall back to the global
        // minimum when the curve only decreases (pure white noise).
        let sigmas: Vec<f64> = curve.iter().map(|p| p.deviation).collect();
        let floor = (1..sigmas.len().saturating_sub(1))
            .find(|&i| sigmas[i] <= sigmas[i - 1] && sigmas[i] <= sigmas[i + 1])
            .map(|i| sigmas[i])
            .unwrap_or_else(|| sigmas.iter().copied().fold(f64::INFINITY, f64::min));
        floor / 0.664
    }
}

#[cfg(test)]
mod lib_test {
    use super::*;

    #[test]
    fn allan_recovers_white_noise() {
        let dt: f64 = 0.01; // 100 Hz
        let n = 1_000_000; // long record — Allan needs samples
        let arw_in = 0.02; // the N you put in
        // generate pure white rate noise of density arw_in:
        let mut rng = StdRng::seed_from_u64(1234);
        let sigma_step: f64 = arw_in / dt.sqrt(); // per-sample std (Step 0.2)
        let data: Vec<f64> = (0..n)
            .map(|_| {
                let sample: f64 = StandardNormal.sample(&mut rng);
                sigma_step * sample
            })
            .collect();

        let curve = allan::allan_deviation(&data, dt);
        let arw_out = allan::read_arw(&curve);
        // recovered within a few percent (Allan estimates are themselves noisy):
        assert!(
            (arw_out - arw_in).abs() / arw_in < 0.1,
            "ARW recovery: put in {arw_in}, got {arw_out}"
        );
    }
}
