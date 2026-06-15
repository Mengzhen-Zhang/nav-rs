use nalgebra::{Matrix2, Matrix4, SMatrix, SVector, Vector2, Vector4};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use statrs::distribution::{ChiSquared, ContinuousCDF};

/// Two-sided χ² acceptance band for a Monte-Carlo-averaged NEES or NIS.
///
/// With `runs` independent runs, the per-step average η̄ = (1/M)·Σᵢ ηⁱ
/// satisfies `M·η̄ ~ χ²_{M·dof}`, where M = `runs` and `dof` is the
/// per-sample degrees of freedom — state dimension N for NEES, measurement
/// dimension M for NIS. Returns the `(low, high)` band **on the η̄ scale**,
/// so a consistent filter has `E[η̄] = dof` sitting inside it.
///
/// Pass `runs = 1` for the band on a single raw per-step η (no averaging).
pub fn chi2_consistency_band(runs: usize, dof: usize, confidence: f64) -> (f64, f64) {
    let m = runs as f64;
    let chi = ChiSquared::new((runs * dof) as f64).expect("runs * dof must be positive");
    let tail = (1.0 - confidence) / 2.0;
    (chi.inverse_cdf(tail) / m, chi.inverse_cdf(1.0 - tail) / m)
}

/// Normalized Innovation Squared
pub fn nis<const M: usize>(
    y: SVector<f64, M>,    // innovation
    s: SMatrix<f64, M, M>, // innovation covariane
) -> f64 {
    let s_inv = s.try_inverse().expect("S is singular");
    y.dot(&(s_inv * y))
}

/// Normalized Estimation Error Squard
pub fn nees<const N: usize>(
    x: SVector<f64, N>,     // ground truth
    x_hat: SVector<f64, N>, // state estimate
    p: SMatrix<f64, N, N>,  // posterior covariance
) -> f64 {
    let p_inv = p.try_inverse().expect("P is singular");
    let e = x - x_hat;
    e.dot(&(p_inv * e))
}

pub struct ConstVelocity2D {
    pub x: SVector<f64, 4>,
    pub var_a: f64, // power spectral density of accelaration
    pub var_m: f64, // power spectral density of measurement
    pub dt: f64,
    pub q: SMatrix<f64, 4, 4>,
    pub r: SMatrix<f64, 2, 2>,
    pub f: SMatrix<f64, 4, 4>,
    pub h: SMatrix<f64, 2, 4>,
    pub q_cholesky: SMatrix<f64, 4, 4>,
    pub r_cholesky: SMatrix<f64, 2, 2>,
    pub rng: StdRng,
}

impl ConstVelocity2D {
    pub fn new(
        var_a: f64, // power spectral density of accelaration
        var_m: f64, // power spectral density of measurement
        dt: f64,
        seed: u64,
    ) -> Self {
        let x = Vector4::new(0.0, 0.0, 1.0, 0.5);
        let q = {
            let a = dt.powi(3) / 3.0;
            let b = dt.powi(2) / 2.0;
            Matrix4::new(
                a, 0.0, b, 0.0, 0.0, a, 0.0, b, b, 0.0, dt, 0.0, 0.0, b, 0.0, dt,
            ) * var_a
        };
        let f = Matrix4::new(
            1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, dt, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let h = SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let r = var_m * Matrix2::identity();
        let rng = StdRng::seed_from_u64(seed);

        let q_cholesky = q.cholesky().expect("q is not positive definite").l();

        let r_cholesky = r.cholesky().expect("r is not positive definite").l();

        Self {
            x,
            var_a,
            var_m,
            dt,
            q,
            r,
            f,
            h,
            q_cholesky,
            r_cholesky,
            rng,
        }
    }

    pub fn update(&mut self) -> (Vector4<f64>, Vector2<f64>) {
        let w = self.q_cholesky
            * SVector::<f64, 4>::from_fn(|_, _| StandardNormal.sample(&mut self.rng));
        let v = self.r_cholesky
            * SVector::<f64, 2>::from_fn(|_, _| StandardNormal.sample(&mut self.rng));

        // update x
        self.x = self.f * self.x + w;

        // generate measurement
        let z = self.h * self.x + v;

        (self.x, z)
    }
}
