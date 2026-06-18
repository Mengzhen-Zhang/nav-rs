//! Strapdown inertial navigation (mechanization) in a local flat-Earth nav
//! frame — **no Earth rotation, no transport rate, constant gravity**. This
//! is the simplest mechanization that still shows the real story: with no
//! external reference, the integrated solution drifts without bound.
//!
//! Convention: ENU nav frame (x east, y north, z up), gravity points down,
//! `g_n = (0, 0, −g)`. Attitude is body→nav.
//!
//! ```text
//! attitude: R[k+1] = R[k] · Exp(ω̃ Δt)        (RotVec::exp)
//! velocity: v[k+1] = v[k] + (R[k]·f̃ + g_n) Δt
//! position: p[k+1] = p[k] + v[k] Δt
//! ```
//! where ω̃ is the measured body rate and f̃ the measured specific force.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use nav_attitude::dcm::Dcm;
use nav_attitude::rotvec::RotVec;

/// Standard gravity magnitude (m/s²).
pub const G: f64 = 9.80665;

/// Nav-frame gravity vector (ENU: points down).
pub fn gravity() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -G)
}

/// Strapdown navigation state in the nav frame.
#[derive(Debug, Clone)]
pub struct NavState {
    pub attitude: Dcm,     // body -> nav
    pub vel: Vector3<f64>, // nav frame (m/s)
    pub pos: Vector3<f64>, // nav frame (m)
}

impl NavState {
    /// One mechanization step. `gyro` (rad/s) and `accel` (specific force,
    /// m/s²) are body-frame measurements held over `dt`.
    pub fn propagate(&self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) -> NavState {
        let delta = RotVec::new(gyro * dt).exp(); // body-frame incremental rotation
        let attitude = self.attitude.compose(&delta); // R[k] · Exp(ω̃ Δt)
        let accel_nav = self.attitude.transform(accel) + gravity();
        let vel = self.vel + accel_nav * dt;
        let pos = self.pos + self.vel * dt; // Euler (exact while stationary)
        NavState { attitude, vel, pos }
    }
}

/// The specific force the IMU would read for a given nav-frame acceleration
/// and attitude: `f^b = Rᵀ (a^n − g_n)`. Exact inverse of [`NavState::propagate`]'s
/// velocity update, so zero-error integration reproduces the truth.
pub fn true_specific_force(attitude: &Dcm, accel_nav: Vector3<f64>) -> Vector3<f64> {
    attitude.transpose().transform(accel_nav - gravity())
}

/// Skew-symmetric (cross-product) matrix `[v]×`.
pub fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// 9-state INS error transition `F = exp(A Δt)`, state `[δθ; δv; δp]` (all
/// nav frame). The continuous error dynamics (flat-Earth, no Earth rate) are
///
/// ```text
/// δθ̇ = -R δω^b              (gyro error tilts the attitude)
/// δv̇ = -[f^n]× δθ + R δf^b  (tilt mis-resolves specific force)
/// δṗ = δv
/// ```
///
/// `A` is nilpotent (`A³ = 0`), so the matrix exponential is the exact finite
/// series `I + AΔt + ½A²Δt²`. `f_nav` is the specific force in the nav frame.
pub fn error_transition(f_nav: Vector3<f64>, dt: f64) -> SMatrix<f64, 9, 9> {
    let mut f = SMatrix::<f64, 9, 9>::identity();
    let s = skew(f_nav);
    // δv ← δθ : -[f^n]× Δt
    f.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-s * dt));
    // δp ← δθ : -½ [f^n]× Δt²
    f.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-0.5 * s * dt * dt));
    // δp ← δv : I Δt
    f.fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(Matrix3::identity() * dt));
    f
}

/// Discrete INS process noise from the gyro/accel white-noise densities
/// (`arw`, `vrw`, same units as `ImuErrorParams::white_noise_density`).
/// First-order `Q_d = G Q_c Gᵀ Δt`: angle-random-walk into `δθ`,
/// velocity-random-walk into `δv`.
pub fn error_process_noise(arw: f64, vrw: f64, dt: f64) -> SMatrix<f64, 9, 9> {
    let mut q = SMatrix::<f64, 9, 9>::zeros();
    let qa = arw * arw * dt;
    let qv = vrw * vrw * dt;
    for i in 0..3 {
        q[(i, i)] = qa;
        q[(3 + i, 3 + i)] = qv;
    }
    q
}

/// The 9-state error `[δθ; δv; δp]` of an estimate against truth.
/// `δθ = Log(R_est · R_trueᵀ)` is the nav-frame attitude error, matching the
/// sign convention of [`error_transition`].
pub fn nav_error(est: &NavState, truth: &NavState) -> SVector<f64, 9> {
    let dtheta = est
        .attitude
        .compose(&truth.attitude.transpose())
        .log()
        .to_vector();
    let dv = est.vel - truth.vel;
    let dp = est.pos - truth.pos;
    let mut e = SVector::<f64, 9>::zeros();
    e.fixed_rows_mut::<3>(0).copy_from(&dtheta);
    e.fixed_rows_mut::<3>(3).copy_from(&dv);
    e.fixed_rows_mut::<3>(6).copy_from(&dp);
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Zero-error mechanization must reproduce a stationary truth exactly:
    /// no rates, specific force exactly cancels gravity, nothing moves.
    #[test]
    fn zero_error_stationary_stays_put() {
        let dt = 0.01;
        let truth = NavState {
            attitude: Dcm::new(Matrix3::identity()),
            vel: Vector3::zeros(),
            pos: Vector3::new(10.0, -5.0, 100.0),
        };
        // a stationary, level IMU reads zero rate and +g specific force (up)
        let gyro_true = Vector3::zeros();
        let accel_true = true_specific_force(&truth.attitude, Vector3::zeros());
        assert_relative_eq!(accel_true, Vector3::new(0.0, 0.0, G), epsilon = 1e-12);

        let mut s = truth.clone();
        for _ in 0..10_000 {
            s = s.propagate(gyro_true, accel_true, dt);
        }
        assert_relative_eq!(s.pos, truth.pos, epsilon = 1e-9);
        assert_relative_eq!(s.vel, Vector3::zeros(), epsilon = 1e-9);
    }

    /// A constant-velocity, level, non-rotating truth is also reproduced:
    /// position advances linearly with no spurious drift.
    #[test]
    fn zero_error_constant_velocity_tracks() {
        let dt = 0.001;
        let v = Vector3::new(3.0, -1.0, 0.0);
        let attitude = Dcm::new(Matrix3::identity());
        let accel_true = true_specific_force(&attitude, Vector3::zeros()); // no accel
        let gyro_true = Vector3::zeros();

        let mut s = NavState {
            attitude,
            vel: v,
            pos: Vector3::zeros(),
        };
        let n = 5000;
        for _ in 0..n {
            s = s.propagate(gyro_true, accel_true, dt);
        }
        let t = n as f64 * dt;
        assert_relative_eq!(s.pos, v * t, epsilon = 1e-9);
        assert_relative_eq!(s.vel, v, epsilon = 1e-12);
    }
}
