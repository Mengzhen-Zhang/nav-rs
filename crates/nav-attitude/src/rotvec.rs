//! Rotation vectors — coordinates on the Lie algebra so(3), and the
//! exponential map onto the group.
//!
//! # Where this module sits in the crate's Lie-theoretic diagram
//!
//! ```text
//!   RotVec  (ℝ³ ≅ so(3) via numerics::hat)
//!     │  exp  (Rodrigues, this module)          │ UnitQuat::from_rotvec
//!     ▼                                         ▼
//!    Dcm  (SO(3))  ◄── covering map to_dcm ──  UnitQuat  (SU(2) = Spin(3))
//!     │
//!     └─ log = to_quat ∘ to_rotvec  (back to so(3), principal branch)
//! ```
//!
//! # Conventions
//!
//! | Decision        | Choice                                                        |
//! |-----------------|---------------------------------------------------------------|
//! | Meaning         | φ = θ·û: angle θ = `angle()` radians about unit axis û        |
//! | exp target      | **Active** DCM, consistent with the rest of the crate         |
//! | Principal branch| `Dcm::log` and `UnitQuat::to_rotvec` return the **minimal** vector, \|φ\| ≤ π; inputs of any magnitude are accepted by `exp` (it wraps) |
//! | Small angles    | All series branch at the shared `numerics::SMALL_ANGLE_EPS`   |
//! | Cancellation    | `1 − cosθ` is **never computed directly** — the half-angle identity `2·sin²(θ/2)` is used instead; see the `rodrigues_coeffs_seam` test for the failure it prevents |
//!
//! # ⚠ Rotation vectors do not add
//!
//! `exp(φ₁)·exp(φ₂) ≠ exp(φ₁ + φ₂)` except to first order — the
//! correction begins with the Lie bracket, ½·φ₁×φ₂
//! (Baker–Campbell–Hausdorff). Compose rotations through `exp`, never by
//! summing rotation vectors; the `rotation_vectors_do_not_add` test below
//! quantifies the error naive addition commits.

use nalgebra::{Matrix3, Vector3};

use crate::{dcm::Dcm, numerics::SMALL_ANGLE_EPS};

/// Taylor branches of the Rodrigues coefficients, valid for
/// θ < [`SMALL_ANGLE_EPS`]:
///
/// - `a(θ) = sinθ/θ        = 1 − θ²/6  + θ⁴/120 − O(θ⁶/5040)`
/// - `b(θ) = (1−cosθ)/θ²   = ½ − θ²/24 + θ⁴/720 − O(θ⁶/40320)`
///
/// Truncation error at the seam (θ = 10⁻⁶) is ~10⁻³⁹ relative — both
/// branches agree to full f64 precision where they meet (the
/// `rodrigues_coeffs_seam` test certifies this).
///
/// Takes θ² rather than θ so that callers may branch on θ² against
/// `EPS²` and skip the square root entirely in the small-angle path.
fn rodrigues_coeffs(theta_sq: f64) -> (f64, f64) {
    let a = 1.0 - theta_sq / 6.0 + theta_sq * theta_sq / 120.0;
    let b = 0.5 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0;
    (a, b)
}

/// A rotation vector φ = θ·û — the coordinates of an so(3) element
/// under the `numerics::hat` isomorphism.
///
/// This is the representation in which rotations are a **vector space**:
/// the natural home for error states, covariances, and Gaussians (the
/// Phase-4 filter keeps its attitude error here). The price is that the
/// group operation is *not* addition — see the module-level warning.
///
/// Any finite vector is a valid `RotVec`; magnitudes above π denote the
/// long way around and are accepted by [`RotVec::exp`], but never
/// *returned* by the principal-branch logarithms.
pub struct RotVec(Vector3<f64>);

impl RotVec {
    /// Wraps a vector as so(3) coordinates. No constraints: every finite
    /// vector is a valid algebra element.
    pub fn new(v: Vector3<f64>) -> Self {
        Self(v)
    }

    /// The underlying coordinates (a copy).
    pub fn to_vector(&self) -> Vector3<f64> {
        self.0
    }

    /// The underlying coordinates (a copy).
    pub fn vector(&self) -> &Vector3<f64> {
        &self.0
    }

    /// The rotation angle θ = |φ|, in radians. Always ≥ 0; the axis
    /// carries the sign.
    pub fn angle(&self) -> f64 {
        self.0.norm()
    }
}

impl RotVec {
    /// The exponential map so(3) → SO(3): the **active** DCM rotating by
    /// θ = |φ| about û = φ/θ.
    ///
    /// Computed via Rodrigues in division-free form,
    ///
    /// ```text
    /// R = I + a(θ)·[φ]ₓ + b(θ)·[φ]ₓ²,   a = sinθ/θ,   b = (1−cosθ)/θ²
    /// ```
    ///
    /// where the 1/θ of axis normalization has been absorbed into two
    /// smooth scalar coefficients — so θ → 0 needs only a Taylor branch
    /// ([`rodrigues_coeffs`], below [`SMALL_ANGLE_EPS`]), not a special
    /// case, and exp(0) = I falls out exactly.
    ///
    /// Numerics: in the closed-form branch, `b` is evaluated as
    /// `2·sin²(θ/2)/θ²` rather than `(1−cosθ)/θ²` — the direct form
    /// suffers catastrophic cancellation for small-to-moderate θ (the
    /// difference sits a few hundred ulps below 1.0). The half-angle
    /// identity has full relative precision everywhere.
    ///
    /// The result satisfies `Dcm`'s SO(3) trust contract by
    /// construction, up to rounding.
    pub fn exp(&self) -> Dcm {
        use crate::numerics::hat;

        let theta = self.angle();

        let (a, b) = if theta < SMALL_ANGLE_EPS {
            rodrigues_coeffs(theta * theta)
        } else {
            let half_sin = (0.5 * theta).sin();
            (
                theta.sin() / theta,
                2.0 * half_sin * half_sin / (theta * theta),
            )
        };

        let phi = self.to_vector();
        let phi_hat = hat(&phi);

        let r = Matrix3::identity() + a * phi_hat + b * phi_hat * phi_hat;
        Dcm::new(r)
    }
}

// ============================================================
// Day 6 test suite
// ============================================================
//
// Note on access: these tests live in rotvec.rs, so they can call the
// private `rodrigues_coeffs` and the pub(crate) `hat`/`vee`, but NOT
// Dcm's private inner matrix. Where a raw matrix is needed (naive-log
// baseline), it is reconstructed through the public API: the columns
// of R are R·e₁, R·e₂, R·e₃. If that ever feels too clever, the
// alternative is a `pub(crate) fn matrix(&self)` accessor on Dcm.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::{SMALL_ANGLE_EPS, hat, vee};
    use crate::quat::UnitQuat;
    use approx::relative_eq;
    use nalgebra::{Matrix3, Unit, Vector3};
    use proptest::prelude::*;
    use std::f64::consts::PI;

    // ---------------- strategies & helpers ----------------

    /// Random rotation vector with angle in (min, max).
    fn arb_phi(min: f64, max: f64) -> impl Strategy<Value = Vector3<f64>> {
        (prop::array::uniform3(-1.0f64..1.0), min..max).prop_filter_map(
            "axis too short to normalize",
            |(a, theta)| {
                let v = Vector3::from(a);
                (v.norm() > 1e-3).then(|| v.normalize() * theta)
            },
        )
    }

    fn arb_vec3() -> impl Strategy<Value = Vector3<f64>> {
        prop::array::uniform3(-10.0f64..10.0).prop_map(Vector3::from)
    }

    /// Reconstruct the raw matrix of a Dcm through its public action:
    /// column i of R is R·eᵢ.
    fn matrix_of(d: &Dcm) -> Matrix3<f64> {
        Matrix3::from_columns(&[
            d.transform(Vector3::x()),
            d.transform(Vector3::y()),
            d.transform(Vector3::z()),
        ])
    }

    /// The naive matrix log everyone writes first: θ from the trace,
    /// axis from the antisymmetric part. Divides by sin θ → collapses
    /// near θ = π. Test-only baseline for the degradation comparison.
    fn naive_log(d: &Dcm) -> Vector3<f64> {
        let r = matrix_of(d);
        let cos_theta = ((r.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let axis_scaled = vee(&r); // = sinθ · û
        let s = theta.sin();
        if s.abs() < 1e-300 {
            return Vector3::zeros(); // θ = 0 (or exactly π, where this method is hopeless anyway)
        }
        axis_scaled * (theta / s)
    }

    // ---------------- the so(3) algebra itself ----------------

    proptest! {
        /// Defining property of the hat map: [v]ₓ·w = v × w.
        #[test]
        fn hat_action(v in arb_vec3(), w in arb_vec3()) {
            prop_assert!(relative_eq!(hat(&v) * w, v.cross(&w), epsilon = 1e-9));
        }
    }

    proptest! {
        /// [u]ₓ² = u·uᵀ − |u|²·I — the identity that collapses the
        /// exponential series into Rodrigues.
        #[test]
        fn hat_square_identity(u in arb_vec3()) {
            let lhs = hat(&u) * hat(&u);
            let rhs = u * u.transpose() - u.norm_squared() * Matrix3::identity();
            prop_assert!(relative_eq!(lhs, rhs, epsilon = 1e-9));
        }
    }

    proptest! {
        /// vee ∘ hat = id on ℝ³.
        #[test]
        fn vee_inverts_hat(v in arb_vec3()) {
            prop_assert!(relative_eq!(vee(&hat(&v)), v, epsilon = 1e-12));
        }
    }

    // ---------------- exp / log ----------------

    proptest! {
        /// φ → exp → log → φ across the full principal domain.
        /// (log returns the minimal vector, so θ stays below π here;
        /// the π boundary is exercised deterministically below.)
        #[test]
        fn exp_log_roundtrip(phi in arb_phi(1e-9, PI - 1e-6)) {
            let back = RotVec::new(phi).exp().log();
            prop_assert!(relative_eq!(back.to_vector(), phi, epsilon = 1e-9));
        }
    }

    proptest! {
        /// THE CONSISTENCY TRIANGLE: the matrix exponential and the
        /// quaternion exponential must describe the same rotation.
        /// Routes through all three modules (rotvec → dcm → Shepperd →
        /// quat vs rotvec → quat), so no single-module slip survives it.
        /// This test fails for every θ under the half-angle bug in
        /// `from_rotvec` — run it once before fixing, as the ritual demands.
        #[test]
        fn exp_agrees_with_quaternion_path(phi in arb_phi(1e-9, PI - 1e-6)) {
            let via_matrix = RotVec::new(phi).exp().to_quat();
            let via_quat = UnitQuat::from_rotvec(RotVec::new(phi));
            prop_assert!(via_matrix.approx_eq_rotation(&via_quat, 1e-11));
        }
    }

    proptest! {
        /// from_rotvec must agree with the independently-implemented
        /// from_axis_angle for all angles — this sweeps both branches of
        /// the sin(θ/2)/θ coefficient, seam included, against an oracle
        /// that never branches.
        #[test]
        fn from_rotvec_matches_from_axis_angle(phi in arb_phi(1e-9, PI - 1e-9)) {
            let q1 = UnitQuat::from_rotvec(RotVec::new(phi));
            let q2 = UnitQuat::from_axis_angle(Unit::new_normalize(phi), phi.norm());
            prop_assert!(q1.approx_eq_rotation(&q2, 1e-12));
        }
    }

    proptest! {
        /// φ → quaternion → φ, vector equality on the open domain.
        #[test]
        fn quat_rotvec_roundtrip(phi in arb_phi(1e-9, PI - 1e-6)) {
            let back = UnitQuat::from_rotvec(RotVec::new(phi)).to_rotvec();
            prop_assert!(relative_eq!(back.to_vector(), phi, epsilon = 1e-9));
        }
    }

    // ---------------- deterministic seams & boundaries ----------------

    /// Both branches of the Rodrigues coefficients agree across the
    /// SMALL_ANGLE_EPS seam — the Day-6 deliverable from the schedule.
    #[test]
    fn rodrigues_coeffs_seam() {
        for theta in [
            SMALL_ANGLE_EPS * 0.5,
            SMALL_ANGLE_EPS * 0.99,
            SMALL_ANGLE_EPS * 1.01,
            SMALL_ANGLE_EPS * 2.0,
        ] {
            let (a_series, b_series) = rodrigues_coeffs(theta * theta);
            let a_exact = theta.sin() / theta;
            let half_sin = (0.5 * theta).sin();
            let b_exact = 2.0 * half_sin * half_sin / (theta * theta);

            assert!(
                relative_eq!(a_series, a_exact, epsilon = 1e-14),
                "a-coefficient seam mismatch at θ = {theta:e}"
            );
            assert!(
                relative_eq!(b_series, b_exact, epsilon = 1e-14),
                "b-coefficient seam mismatch at θ = {theta:e}"
            );
        }
    }

    /// θ = π about each coordinate axis: the boundary of the principal
    /// domain, where ±û are the same rotation. Compared as rotations.
    /// One action spot-check pins exp(π·x̂) = diag(1, −1, −1).
    #[test]
    fn pi_axis_cases() {
        for axis in [Vector3::x_axis(), Vector3::y_axis(), Vector3::z_axis()] {
            let phi = axis.into_inner() * PI;
            let via_exp = RotVec::new(phi).exp().to_quat();
            let oracle = UnitQuat::from_axis_angle(axis, PI);
            assert!(
                via_exp.approx_eq_rotation(&oracle, 1e-11),
                "π rotation about {axis:?} disagrees with oracle"
            );
        }
        let r = RotVec::new(Vector3::x() * PI).exp();
        assert!(relative_eq!(
            r.transform(Vector3::x()),
            Vector3::x(),
            epsilon = 1e-12
        ));
        assert!(relative_eq!(
            r.transform(Vector3::y()),
            -Vector3::y(),
            epsilon = 1e-12
        ));
        assert!(relative_eq!(
            r.transform(Vector3::z()),
            -Vector3::z(),
            epsilon = 1e-12
        ));
    }

    // ---------------- the Lie-theoretic signature moves ----------------

    proptest! {
        /// BCH to second order: log(exp(a)·exp(b)) = a + b + ½ a×b + O(θ³).
        /// The cross product is so(3)'s bracket, so this certifies the
        /// *algebra*, not just the maps. Tolerance is adaptive: the
        /// neglected terms are cubic in the total angle.
        #[test]
        fn bch_second_order(
            a in arb_phi(1e-4, 1e-2),
            b in arb_phi(1e-4, 1e-2),
        ) {
            let composed = RotVec::new(a).exp().compose(&RotVec::new(b).exp());
            let z = composed.log().to_vector();
            let bch2 = a + b + 0.5 * a.cross(&b);
            let residual = (z - bch2).norm();
            let cubic_scale = (a.norm() + b.norm()).powi(3);
            prop_assert!(
                residual < cubic_scale,
                "BCH residual {residual:e} exceeds cubic scale {cubic_scale:e}"
            );
        }
    }

    /// And the negative result the BCH doc-sentence warns about:
    /// naive addition of rotation vectors misses by exactly the bracket
    /// term. a = 0.01·x̂, b = 0.01·ŷ: the ½ a×b term is 5·10⁻⁵ ẑ —
    /// naive summing eats that error; second-order BCH does not.
    #[test]
    fn rotation_vectors_do_not_add() {
        let a = Vector3::x() * 0.01;
        let b = Vector3::y() * 0.01;
        let z = RotVec::new(a)
            .exp()
            .compose(&RotVec::new(b).exp())
            .log()
            .to_vector();

        let naive_err = (z - (a + b)).norm();
        let bch2_err = (z - (a + b + 0.5 * a.cross(&b))).norm();

        assert!(
            naive_err > 4e-5,
            "naive sum unexpectedly good: {naive_err:e}"
        );
        assert!(
            bch2_err < 5e-6,
            "second-order BCH unexpectedly bad: {bch2_err:e}"
        );
    }

    // ---------------- why log routes through the quaternion ----------------

    /// Degradation comparison near θ = π: the naive trace/sin log loses
    /// digits as 1/(π−θ); the quaternion-routed log must not. Run with
    /// `cargo test naive -- --nocapture` to print the README-plot data.
    #[test]
    fn naive_log_degrades_near_pi_quat_path_does_not() {
        let axis = Vector3::new(1.0, 2.0, 3.0).normalize();
        let mut worst_naive: f64 = 0.0;
        let mut worst_quat: f64 = 0.0;

        for k in 4..=10 {
            let theta = PI - 10f64.powi(-k);
            let phi = axis * theta;
            let d = RotVec::new(phi).exp();

            let err_quat = (d.log().to_vector() - phi).norm();
            let err_naive = (naive_log(&d) - phi).norm();
            println!("π−θ = 1e-{k}:  quat-path {err_quat:.3e}   naive {err_naive:.3e}");

            worst_quat = worst_quat.max(err_quat);
            worst_naive = worst_naive.max(err_naive);
        }

        assert!(
            worst_quat < 1e-12,
            "quaternion-routed log degraded near π: {worst_quat:e}"
        );
        assert!(
            worst_naive > 100.0 * worst_quat,
            "naive log suspiciously healthy ({worst_naive:e}) — is the baseline still naive?"
        );
    }
}
