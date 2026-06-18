//! GPS/INS consistency tooling — the imu-forge → ESKF NEES validation.
//!
//! A known trajectory is flown, an Allan-characterized [`SyntheticImu`] (Phase
//! 3) corrupts the inertial stream, the [`EskfState`] (Phase 4) dead-reckons it
//! and folds in periodic position fixes, and the 15-state estimation error is
//! scored against the filter's own covariance as NEES. Averaged over an
//! independent-seed Monte-Carlo ensemble, `M·η̄ ~ χ²_{15M}`, so a *consistent*
//! filter sits on `E[η̄] = 15` inside a tight χ² band.
//!
//! Why this is a real proof, not a demo — every knob is matched on purpose:
//!
//! * **Same σ both sides.** The IMU's white-noise densities and Gauss–Markov
//!   bias parameters are fed *verbatim* to the filter's process model
//!   (`sigma_a = n_a`, `sigma_ba = σ_ba`, `tau_a`, …). That identity is the
//!   whole point of Phase 3 feeding Phase 4; a √dt slip or a units mismatch
//!   shows up immediately as NEES leaving the band.
//! * **No unmodeled error.** Scale error and constant turn-on bias are set to
//!   zero — the ESKF estimates an additive Gauss–Markov bias, not scale, so
//!   leaving them in would be an honest *model mismatch*, not an inconsistency.
//! * **Matched dynamics.** Truth is integrated with the *same* nominal
//!   equations as the filter ([`truth_step`]), so under zero noise the filter
//!   reproduces truth exactly and only sensor noise drives the error.
//! * **Honest P₀.** The initial estimate is offset from truth by a draw from
//!   P₀ itself, so the error is in-distribution from `t = 0` (no warm-up
//!   transient masking a bad covariance).
//!
//! The attitude error uses the filter's **right/body** convention
//! `δθ = Log(q_estᵀ · q_true)` — matching `predict`'s `F_θθ = Exp(−ω̂ dt)` and
//! the `q ← q ⊗ Exp(δθ)` injection — *not* a nav-frame error.

use nalgebra::{Matrix3, SVector, Vector3};
use nav_attitude::quat::UnitQuat;
use nav_attitude::rotvec::RotVec;
use nav_imu::{ImuErrorParams, SyntheticImu};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::{EskfState, G};

// ----- Allan-characterized IMU (tactical grade), shared by truth and filter -----
/// Accelerometer velocity-random-walk (VRW) spectral coefficient, m/s²/√Hz.
pub const N_A: f64 = 0.02;
/// Gyro angle-random-walk (ARW) spectral coefficient, rad/s/√Hz.
pub const N_G: f64 = 5e-4;
/// Accelerometer bias-instability stationary σ (Gauss–Markov), m/s².
pub const SIGMA_BA: f64 = 0.01;
/// Gyro bias-instability stationary σ (Gauss–Markov), rad/s.
pub const SIGMA_BG: f64 = 1e-4;
/// Bias correlation time τ for both sensors, s.
pub const TAU: f64 = 100.0;

// ----- Initial uncertainty (P₀ 1σ) — also the spread of the initial error -----
pub const INIT_SIGMA_P: f64 = 1.0; // m
pub const INIT_SIGMA_V: f64 = 0.1; // m/s
pub const INIT_SIGMA_THETA: f64 = 0.02; // rad (~1.1°)

// ----- Trajectory: a level coordinated turn (constant speed, constant yaw) -----
pub const SPEED: f64 = 10.0; // m/s along body-x
pub const TURN_RATE: f64 = 0.2; // rad/s yaw  ⇒ radius = SPEED/TURN_RATE = 50 m

// ----- Position fixes (GPS) -----
pub const SIGMA_GPS: f64 = 0.5; // m, per axis (isotropic R)

/// Monte-Carlo run configuration (resolution / length / ensemble size).
#[derive(Clone, Copy)]
pub struct McConfig {
    pub runs: usize,
    pub dt: f64,
    pub n_steps: usize,
    pub fix_every: usize, // steps between position fixes (= NEES sample spacing)
    pub seed_base: u64,
}

impl McConfig {
    /// Full-resolution config for the portfolio plot: 50 runs, 120 s @ 100 Hz,
    /// a fix every 1 s.
    pub fn portfolio() -> Self {
        Self {
            runs: 50,
            dt: 0.01,
            n_steps: 12_000,
            fix_every: 100,
            seed_base: 0,
        }
    }

    /// Lighter config for the in-CI consistency assertion: 24 runs, 20 s @
    /// 50 Hz, 100 fixes — enough for a tight band and a meaningful in-band
    /// fraction, while staying quick in a debug `cargo test` run.
    pub fn test() -> Self {
        Self {
            runs: 24,
            dt: 0.02,
            n_steps: 1_000,
            fix_every: 10,
            seed_base: 1_000,
        }
    }
}

/// One Monte-Carlo result: the seed-averaged NEES time series with its χ² band,
/// plus run-0 diagnostics for the plot panels.
pub struct NeesResult {
    pub t: Vec<f64>,         // sample times (s)
    pub mean_nees: Vec<f64>, // η̄(t), averaged over runs
    pub band_lo: f64,        // χ²₁₅ 95% band on the η̄ scale
    pub band_hi: f64,
    pub dof: usize, // 15
    pub runs: usize,
    // --- run-0 diagnostics (one realization) ---
    pub truth_xy: Vec<(f64, f64)>, // truth horizontal track
    pub est_xy: Vec<(f64, f64)>,   // ESKF estimate track
    pub fix_xy: Vec<(f64, f64)>,   // GPS fixes
    pub pos_err: Vec<f64>,         // east position error e_p.x
    pub pos_sig: Vec<f64>,         // √P for east position
    pub ba_err: Vec<f64>,          // accel-bias error e_ba.x
    pub ba_sig: Vec<f64>,          // √P for accel-bias x
}

impl NeesResult {
    /// Fraction of post-burn-in samples whose seed-averaged NEES lies in band.
    pub fn fraction_in_band(&self, burn_in_s: f64) -> f64 {
        let kept: Vec<f64> = self
            .t
            .iter()
            .zip(&self.mean_nees)
            .filter(|(t, _)| **t >= burn_in_s)
            .map(|(_, v)| *v)
            .collect();
        let inb = kept
            .iter()
            .filter(|&&v| v >= self.band_lo && v <= self.band_hi)
            .count();
        inb as f64 / kept.len() as f64
    }

    /// Mean of the post-burn-in seed-averaged NEES (should sit near `dof`).
    pub fn mean_after(&self, burn_in_s: f64) -> f64 {
        let kept: Vec<f64> = self
            .t
            .iter()
            .zip(&self.mean_nees)
            .filter(|(t, _)| **t >= burn_in_s)
            .map(|(_, v)| *v)
            .collect();
        kept.iter().sum::<f64>() / kept.len() as f64
    }
}

/// Two-sided χ² acceptance band for a Monte-Carlo-averaged NEES, on the η̄
/// scale (`M·η̄ ~ χ²_{M·dof}`). Same construction as the Phase-2 tooling.
pub fn chi2_band(runs: usize, dof: usize, confidence: f64) -> (f64, f64) {
    let m = runs as f64;
    let chi = ChiSquared::new((runs * dof) as f64).expect("runs * dof must be positive");
    let tail = (1.0 - confidence) / 2.0;
    (chi.inverse_cdf(tail) / m, chi.inverse_cdf(1.0 - tail) / m)
}

fn randn3(rng: &mut StdRng) -> Vector3<f64> {
    Vector3::from_fn(|_, _| StandardNormal.sample(rng))
}

/// Truth kinematic step — mirrors [`EskfState::predict`]'s *nominal* state
/// update exactly (same equations, same order), so zero-noise / perfect-bias
/// inputs reproduce the filter trajectory and only sensor noise drives error.
fn truth_step(
    p: &mut Vector3<f64>,
    v: &mut Vector3<f64>,
    q: &mut UnitQuat,
    f_b: &Vector3<f64>, // true specific force, body frame
    w: &Vector3<f64>,   // true body rate
    dt: f64,
) {
    let g = Vector3::new(0.0, 0.0, -G);
    let r = *q.to_dcm().matrix();
    let a_nav = r * f_b + g;
    *p += *v * dt + 0.5 * a_nav * dt * dt;
    *v += a_nav * dt;
    *q = q.compose(&UnitQuat::from_rotvec(RotVec::new(w * dt)));
}

/// The 15-state estimation error in the filter's own error coordinates:
/// `[δp; δv; δθ; δb_a; δb_g]`, each = truth − estimate. Attitude uses the
/// right/body convention `δθ = Log(q_estᵀ q_true)`.
fn error_vector(
    f: &EskfState,
    p_t: &Vector3<f64>,
    v_t: &Vector3<f64>,
    q_t: &UnitQuat,
    imu: &SyntheticImu,
) -> SVector<f64, 15> {
    let e_p = p_t - f.p;
    let e_v = v_t - f.v;
    let e_th = f.q.inverse().compose(q_t).to_rotvec().to_vector();
    let e_ba = imu.accel_bias() - f.ba;
    let e_bg = imu.gyro_bias() - f.bg;

    let mut e = SVector::<f64, 15>::zeros();
    e.fixed_rows_mut::<3>(0).copy_from(&e_p);
    e.fixed_rows_mut::<3>(3).copy_from(&e_v);
    e.fixed_rows_mut::<3>(6).copy_from(&e_th);
    e.fixed_rows_mut::<3>(9).copy_from(&e_ba);
    e.fixed_rows_mut::<3>(12).copy_from(&e_bg);
    e
}

fn gyro_params() -> ImuErrorParams {
    // scale = 0, turn-on bias = 0 (unmodeled by the ESKF — keep them out).
    ImuErrorParams::new(0.0, 0.0, SIGMA_BG, TAU, N_G)
}
fn accel_params() -> ImuErrorParams {
    ImuErrorParams::new(0.0, 0.0, SIGMA_BA, TAU, N_A)
}

/// Run the full imu-forge → ESKF NEES Monte-Carlo and return the seed-averaged
/// series plus its χ² band.
pub fn run_nees_mc(cfg: &McConfig) -> NeesResult {
    let n_samples = cfg.n_steps / cfg.fix_every;
    let mut nees_sum = vec![0.0f64; n_samples];

    // run-0 diagnostics
    let mut truth_xy = Vec::with_capacity(n_samples);
    let mut est_xy = Vec::with_capacity(n_samples);
    let mut fix_xy = Vec::with_capacity(n_samples);
    let mut pos_err = Vec::with_capacity(n_samples);
    let mut pos_sig = Vec::with_capacity(n_samples);
    let mut ba_err = Vec::with_capacity(n_samples);
    let mut ba_sig = Vec::with_capacity(n_samples);

    let omega = Vector3::new(0.0, 0.0, TURN_RATE);
    let g_vec = Vector3::new(0.0, 0.0, -G);

    for run in 0..cfg.runs {
        // Independent streams: one rng for init-error + GPS, one for the IMU.
        let mut rng = StdRng::seed_from_u64(cfg.seed_base + run as u64);
        let mut imu = SyntheticImu::new(
            gyro_params(),
            accel_params(),
            cfg.dt,
            cfg.seed_base + 10_000 + run as u64,
        );

        // Truth initial state: on the circle, heading east, level.
        let mut p_t = Vector3::zeros();
        let mut v_t = Vector3::new(SPEED, 0.0, 0.0);
        let mut q_t = UnitQuat::IDENTITY;

        // Filter starts offset from truth by a draw from P₀ (truth − est = δx₀),
        // so the error is in-distribution from t = 0. Truth bias is 0 at t = 0,
        // so e_ba(0) = -filter.ba = δba₀, consistent with the P₀ bias block.
        let dp0 = INIT_SIGMA_P * randn3(&mut rng);
        let dv0 = INIT_SIGMA_V * randn3(&mut rng);
        let dth0 = INIT_SIGMA_THETA * randn3(&mut rng);
        let dba0 = SIGMA_BA * randn3(&mut rng);
        let dbg0 = SIGMA_BG * randn3(&mut rng);

        let mut filter = EskfState::new(
            Vector3::zeros(),
            Vector3::zeros(),
            TAU,
            TAU,
            N_A,
            N_G,
            SIGMA_BA,
            SIGMA_BG,
            INIT_SIGMA_P,
            INIT_SIGMA_V,
            INIT_SIGMA_THETA,
        );
        filter.p = p_t - dp0;
        filter.v = v_t - dv0;
        filter.q = q_t.compose(&UnitQuat::from_rotvec(RotVec::new(-dth0)));
        filter.ba = -dba0;
        filter.bg = -dbg0;

        let r_cov = Matrix3::identity() * SIGMA_GPS.powi(2);
        let mut si = 0;

        for k in 1..=cfg.n_steps {
            // True specific force / rate for this step, from the current truth.
            let r_true = *q_t.to_dcm().matrix();
            let a_nav = omega.cross(&v_t); // centripetal, constant-speed turn
            let f_b = r_true.transpose() * (a_nav - g_vec);

            let meas = imu.sample(omega, f_b); // (gyro, accel), corrupted
            filter.predict(&meas.accel, &meas.gyro, cfg.dt);
            truth_step(&mut p_t, &mut v_t, &mut q_t, &f_b, &omega, cfg.dt);

            if k % cfg.fix_every == 0 {
                let gps = p_t + SIGMA_GPS * randn3(&mut rng);
                filter.update_position(&gps, &r_cov);

                let e = error_vector(&filter, &p_t, &v_t, &q_t, &imu);
                let pinv = filter.cov.try_inverse().expect("P singular");
                nees_sum[si] += e.dot(&(pinv * e));

                if run == 0 {
                    truth_xy.push((p_t.x, p_t.y));
                    est_xy.push((filter.p.x, filter.p.y));
                    fix_xy.push((gps.x, gps.y));
                    pos_err.push(p_t.x - filter.p.x);
                    pos_sig.push(filter.cov[(0, 0)].sqrt());
                    ba_err.push(imu.accel_bias().x - filter.ba.x);
                    ba_sig.push(filter.cov[(9, 9)].sqrt());
                }
                si += 1;
            }
        }
    }

    let t: Vec<f64> = (1..=n_samples)
        .map(|i| (i * cfg.fix_every) as f64 * cfg.dt)
        .collect();
    let mean_nees: Vec<f64> = nees_sum.iter().map(|s| s / cfg.runs as f64).collect();
    let (band_lo, band_hi) = chi2_band(cfg.runs, 15, 0.95);

    NeesResult {
        t,
        mean_nees,
        band_lo,
        band_hi,
        dof: 15,
        runs: cfg.runs,
        truth_xy,
        est_xy,
        fix_xy,
        pos_err,
        pos_sig,
        ba_err,
        ba_sig,
    }
}
