use nalgebra::{Matrix3, SMatrix, Vector3};
use nav_attitude::{quat::UnitQuat, rotvec::RotVec};

pub mod sim;

/// Standard gravity magnitude (m/s²). Must match the value imu-forge
/// (`nav_imu::ins::G`) uses to synthesize specific force, otherwise a
/// stationary feed accelerates at the difference. See the static-gravity test.
pub const G: f64 = 9.80665;

/// Error-state block layout (consistent across `F`, `Q`, `H`, and injection):
///   [0:3] = δp, [3:6] = δv, [6:9] = δθ, [9:12] = δb_a, [12:15] = δb_g.
/// NB: the derivation doc orders the state differently ([δθ, δv, δp, δb_g, δb_a])
/// — do not cross-reference block indices between the doc and this code.
pub struct EskfState {
    pub p: Vector3<f64>,
    pub v: Vector3<f64>,
    pub q: UnitQuat,
    pub ba: Vector3<f64>,
    pub bg: Vector3<f64>,

    pub cov: SMatrix<f64, 15, 15>,

    pub s_a: Matrix3<f64>,
    pub s_g: Matrix3<f64>,

    pub tau_a: f64,
    pub tau_g: f64,

    pub sigma_a: f64,
    pub sigma_g: f64,
    pub sigma_ba: f64,
    pub sigma_bg: f64,
}

/// Block-diagonal initial covariance P₀ from per-block 1σ uncertainties, in
/// real units. Each diagonal block is σ²·I₃:
///   p: σ_p² [m²], v: σ_v² [(m/s)²], θ: σ_θ² [rad²],
///   b_a: σ_ba² [(m/s²)²], b_g: σ_bg² [(rad/s)²].
///
/// The bias blocks are seeded with the *stationary* Gauss–Markov variances
/// σ_ba², σ_bg² — the honest prior on bias before any measurement. An
/// overconfident (e.g. isotropic 1e-4) P₀ causes slow convergence or startup
/// divergence and fails NEES immediately.
fn initial_covariance(
    sigma_p: f64,
    sigma_v: f64,
    sigma_theta: f64,
    sigma_ba: f64,
    sigma_bg: f64,
) -> SMatrix<f64, 15, 15> {
    let mut p0 = SMatrix::<f64, 15, 15>::zeros();
    let vars = [
        sigma_p * sigma_p,
        sigma_v * sigma_v,
        sigma_theta * sigma_theta,
        sigma_ba * sigma_ba,
        sigma_bg * sigma_bg,
    ];
    for (block, &var) in vars.iter().enumerate() {
        p0.view_mut((3 * block, 3 * block), (3, 3))
            .copy_from(&(Matrix3::identity() * var));
    }
    p0
}

impl EskfState {
    /// `sigma_a` / `sigma_g` are the **continuous white-noise spectral
    /// coefficients** (VRW m/s²/√Hz, ARW rad/s/√Hz) — the same N read off the
    /// Allan bathtub and passed to imu-forge as `white_noise_density`. They are
    /// NOT per-sample standard deviations; passing a per-sample σ instead is
    /// off by √dt. The discrete `Q = σ²·dt` below is correct only for the
    /// spectral form.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s_a: Vector3<f64>,
        s_g: Vector3<f64>,
        tau_a: f64,
        tau_g: f64,
        sigma_a: f64,
        sigma_g: f64,
        sigma_ba: f64,
        sigma_bg: f64,
        init_sigma_p: f64,
        init_sigma_v: f64,
        init_sigma_theta: f64,
    ) -> Self {
        Self {
            p: Vector3::zeros(),
            v: Vector3::zeros(),
            q: UnitQuat::IDENTITY,
            ba: Vector3::zeros(),
            bg: Vector3::zeros(),
            cov: initial_covariance(
                init_sigma_p,
                init_sigma_v,
                init_sigma_theta,
                sigma_ba,
                sigma_bg,
            ),
            s_a: Matrix3::identity() + Matrix3::from_diagonal(&s_a),
            s_g: Matrix3::identity() + Matrix3::from_diagonal(&s_g),
            tau_a,
            tau_g,
            sigma_a,
            sigma_g,
            sigma_ba,
            sigma_bg,
        }
    }

    fn skew(&self, v: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    pub fn predict(&mut self, imu_a: &Vector3<f64>, imu_w: &Vector3<f64>, dt: f64) {
        let g = Vector3::new(0.0, 0.0, -G); // nav-frame gravity (ENU: points down)
        let r_mat = *self.q.to_dcm().matrix();

        let a_hat = self.s_a * (imu_a - self.ba);
        let w_hat = self.s_g * (imu_w - self.bg);

        let p_next = self.p + self.v * dt + 0.5 * (r_mat * a_hat + g) * dt * dt;
        let v_next = self.v + (r_mat * a_hat + g) * dt;

        let decay_a = (-dt / self.tau_a).exp();
        let decay_g = (-dt / self.tau_g).exp();

        let ba_next = self.ba * decay_a;
        let bg_next = self.bg * decay_g;

        let rot_vec = RotVec::new(w_hat * dt);
        let dq = UnitQuat::from_rotvec_borrowed(&rot_vec);
        let q_next = self.q.compose(&dq);

        let mut f = SMatrix::<f64, 15, 15>::identity();

        // δp ← δv : I·dt
        f.view_mut((0, 3), (3, 3))
            .copy_from(&(Matrix3::identity() * dt));

        let f_v_theta = -r_mat * self.skew(&a_hat) * dt;
        let f_v_ba = -r_mat * self.s_a * dt;
        f.view_mut((3, 6), (3, 3)).copy_from(&f_v_theta);
        f.view_mut((3, 9), (3, 3)).copy_from(&f_v_ba);

        let rot_vec = RotVec::new(-w_hat * dt);
        let f_theta_theta = *UnitQuat::from_rotvec(rot_vec).to_dcm().matrix();
        let f_theta_bg = -self.s_g * dt;
        f.view_mut((6, 6), (3, 3)).copy_from(&f_theta_theta);
        f.view_mut((6, 12), (3, 3)).copy_from(&f_theta_bg);
        f.view_mut((9, 9), (3, 3))
            .copy_from(&(Matrix3::identity() * decay_a));
        f.view_mut((12, 12), (3, 3))
            .copy_from(&(Matrix3::identity() * decay_g));

        let mut g_mat = SMatrix::<f64, 15, 12>::zeros();
        g_mat
            .view_mut((3, 0), (3, 3))
            .copy_from(&(-r_mat * self.s_a));
        g_mat.view_mut((6, 3), (3, 3)).copy_from(&(-self.s_g));
        g_mat
            .view_mut((9, 6), (3, 3))
            .copy_from(&Matrix3::identity());
        g_mat
            .view_mut((12, 9), (3, 3))
            .copy_from(&Matrix3::identity());

        let mut q_discrete = SMatrix::<f64, 12, 12>::zeros();
        // σ_a, σ_g are continuous spectral densities ⇒ discrete variance σ²·dt.
        let q_a = self.sigma_a.powi(2) * dt;
        let q_g = self.sigma_g.powi(2) * dt;
        let q_ba = self.sigma_ba.powi(2) * (1.0 - (-2.0 * dt / self.tau_a).exp());
        let q_bg = self.sigma_bg.powi(2) * (1.0 - (-2.0 * dt / self.tau_g).exp());

        for i in 0..3 {
            q_discrete[(i, i)] = q_a;
            q_discrete[(i + 3, i + 3)] = q_g;
            q_discrete[(i + 6, i + 6)] = q_ba;
            q_discrete[(i + 9, i + 9)] = q_bg;
        }

        self.cov = f * self.cov * f.transpose() + g_mat * q_discrete * g_mat.transpose();

        self.p = p_next;
        self.v = v_next;
        self.q = q_next;

        self.ba = ba_next;
        self.bg = bg_next;
    }

    pub fn update_position(&mut self, meas_p: &Vector3<f64>, r_cov: &Matrix3<f64>) {
        let residual = meas_p - self.p;

        let mut h = SMatrix::<f64, 3, 15>::zeros();
        h.view_mut((0, 0), (3, 3)).copy_from(&Matrix3::identity());

        let s = h * self.cov * h.transpose() + r_cov;
        let k = self.cov * h.transpose() * s.try_inverse().expect("Inverse failed");

        let delta_x: SMatrix<f64, 15, 1> = k * residual;

        self.p += delta_x.fixed_rows(0);
        self.v += delta_x.fixed_rows(3);

        let delta_theta: Vector3<f64> = delta_x.fixed_rows::<3>(6).into_owned();
        // δθ is a full rotation vector; from_rotvec applies the half-angle.
        let dq = UnitQuat::from_rotvec(RotVec::new(delta_theta));
        self.q = self.q.compose(&dq);

        self.ba += delta_x.fixed_rows(9);
        self.bg += delta_x.fixed_rows(12);

        let kh = k * h;
        let idx = SMatrix::<f64, 15, 15>::identity() - kh;
        self.cov = idx * self.cov * idx.transpose() + k * r_cov * k.transpose();

        let mut g_reset = SMatrix::<f64, 15, 15>::identity();
        let skew_delta_theta = self.skew(&delta_theta);
        let rotation_reset_block = Matrix3::identity() - 0.5 * skew_delta_theta;
        g_reset
            .view_mut((6, 6), (3, 3))
            .copy_from(&rotation_reset_block);

        self.cov = g_reset * self.cov * g_reset.transpose();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Static-gravity test — the #1 silent ESKF bug. Initialize at rest, feed
    /// the stationary specific force a level IMU reads in imu-forge's ENU
    /// convention — `(0, 0, +G)` (up) — with zero rate, and step for 100 s.
    /// If gravity sign and specific-force convention disagree, the body
    /// accelerates (a sign error diverges within the first hundred steps).
    /// Here `R·â + g = (0,0,+G) + (0,0,−G) = 0`, so p and v must stay put.
    #[test]
    fn static_gravity_stays_put() {
        let mut eskf = EskfState::new(
            Vector3::zeros(), // s_a scale error
            Vector3::zeros(), // s_g scale error
            100.0,            // tau_a
            100.0,            // tau_g
            0.0,              // sigma_a (no driving noise — deterministic check)
            0.0,              // sigma_g
            0.0,              // sigma_ba
            0.0,              // sigma_bg
            1.0,              // init_sigma_p
            0.1,              // init_sigma_v
            0.01,             // init_sigma_theta
        );

        let accel = Vector3::new(0.0, 0.0, G); // specific force, +g up (at rest)
        let gyro = Vector3::zeros();
        let dt = 0.01;

        for _ in 0..10_000 {
            eskf.predict(&accel, &gyro, dt);
        }

        assert!(
            eskf.p.norm() < 1e-9,
            "position drifted under static gravity: {}",
            eskf.p
        );
        assert!(
            eskf.v.norm() < 1e-9,
            "velocity drifted under static gravity: {}",
            eskf.v
        );
    }

    /// P₀ is block-diagonal with the requested per-block variances on the
    /// diagonal — not a blanket isotropic value — and the bias blocks carry the
    /// stationary GM variances σ_ba², σ_bg².
    #[test]
    fn initial_covariance_is_per_block() {
        let (sp, sv, st, sba, sbg) = (2.0, 0.5, 0.05, 0.02, 0.001);
        let eskf = EskfState::new(
            Vector3::zeros(),
            Vector3::zeros(),
            100.0,
            100.0,
            0.0,
            0.0,
            sba,
            sbg,
            sp,
            sv,
            st,
        );
        let expected = [sp * sp, sv * sv, st * st, sba * sba, sbg * sbg];
        for (block, &var) in expected.iter().enumerate() {
            for i in 0..3 {
                let d = 3 * block + i;
                assert!(
                    (eskf.cov[(d, d)] - var).abs() < 1e-18,
                    "P0 diag[{d}] = {}, expected {var}",
                    eskf.cov[(d, d)]
                );
            }
        }
        // off-diagonal stays zero (block 0 vs block 1, say)
        assert_eq!(eskf.cov[(0, 4)], 0.0);
    }
}
