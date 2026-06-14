//! Euler angles, 321 (yaw–pitch–roll) intrinsic sequence — the
//! human-I/O representation, kept under strict contract.
//!
//! # Conventions
//!
//! | Decision        | Choice                                                       |
//! |-----------------|--------------------------------------------------------------|
//! | Sequence        | **321 intrinsic**: yaw ψ about z, pitch θ about the new y, roll φ about the newest x |
//! | As active ops   | R(ψ,θ,φ) = **Rz(ψ)·Ry(θ)·Rx(φ)** (intrinsic steps append on the right; as a vector operator, roll acts first) |
//! | Units / ranges  | Radians. Extraction returns the canonical box: pitch ∈ [−π/2, π/2], yaw and roll ∈ (−π, π]. Constructors accept anything (angles wrap; rotations don't) |
//! | Gimbal lock     | Gate at \|sinθ\| > 1 − [`GIMBAL_LOCK_EPS`]; gauge choice **roll = 0**, yaw carries the observable ψ−φ (+lock) or ψ+φ (−lock) |
//! | Composition     | **Deliberately absent.** Euler angles do not compose; convert to `UnitQuat`. The missing method is a design statement |
//!
//! # Why filters don't integrate Euler angles
//!
//! The matrix E mapping Euler rates (ψ̇, θ̇, φ̇) to body angular velocity
//! has det E = cosθ: recovering Euler rates from gyro data means
//! inverting a matrix that is *singular* at θ = ±π/2 and ill-conditioned
//! near it, amplifying gyro noise like 1/cosθ. Rewriting with tan does
//! not help — tanθ carries the same pole. The singularity lives in the
//! parametrization's Jacobian, not the notation (the same way a walker
//! crossing the north pole needs infinite longitude rate). Hence: this
//! crate propagates quaternions and treats Euler angles as a display
//! and specification format only.

use crate::{dcm::Dcm, quat::UnitQuat};
use nalgebra::Vector3;

/// Gate for the gimbal-lock branch of [`Euler321::from_dcm`], applied
/// to |sinθ|.
///
/// The gate handles the **singular point**, not the sick neighborhood:
/// extraction accuracy degrades *continuously* approaching the lock
/// (error amplification ~1/cosθ — see the `conditioning_sweep` test),
/// and no threshold can repair that; 1e-9 is the conventional choice
/// for where the formulas stop being meaningful at all.
const GIMBAL_LOCK_EPS: f64 = 1e-9;

/// Yaw–pitch–roll attitude, 321 intrinsic sequence (see module docs).
///
/// Fields are public: this is a human-I/O type, and the range comments
/// describe what *extraction* produces, not constructor preconditions —
/// any finite angles are accepted, and out-of-range values simply wrap
/// onto the same rotation (verified by the
/// `out_of_range_angles_same_rotation` test).
#[derive(Debug, Clone, Copy)]
pub struct Euler321 {
    pub yaw: f64,   // (-π,π]
    pub pitch: f64, // [-π/2, π/2]
    pub roll: f64,  // (-π,π]
}

impl Euler321 {
    /// Bundles three angles. No validation or normalization — see the
    /// struct docs for range semantics.
    pub fn new(
        yaw: f64,   // (-π,π]
        pitch: f64, // [-π/2, π/2]
        roll: f64,  // (-π,π]
    ) -> Self {
        Self { yaw, pitch, roll }
    }
}

impl Euler321 {
    /// The quaternion of this attitude: qz(ψ) ⊗ qy(θ) ⊗ qx(φ).
    ///
    /// **Composition, not transcription**: built from three
    /// `from_axis_angle` calls on already-certified code, so it is
    /// correct by construction. The closed-form 9-entry matrix — a
    /// transcription minefield — exists only as a test oracle
    /// (`closed_form_dcm` below), where a slip costs a red test
    /// instead of a wrong answer.
    pub fn to_quat(&self) -> UnitQuat {
        let qx = UnitQuat::from_axis_angle(&Vector3::x_axis(), self.roll);
        let qy = UnitQuat::from_axis_angle(&Vector3::y_axis(), self.pitch);
        let qz = UnitQuat::from_axis_angle(&Vector3::z_axis(), self.yaw);
        qz.compose(&qy.compose(&qx))
    }

    /// The DCM of this attitude, routed through the quaternion — the
    /// crate's privileged chart.
    pub fn to_dcm(&self) -> Dcm {
        self.to_quat().to_dcm()
    }

    /// Extraction from a DCM, with the gimbal lock under contract.
    ///
    /// Main branch (|sinθ| below the gate): θ = −asin(r₃₁),
    /// ψ = atan2(r₂₁, r₁₁), φ = atan2(r₃₂, r₃₃) — every atan2 argument
    /// carries a healthy cosθ. The clamp before `asin` absorbs the
    /// ±1±ε that floating point produces at representation boundaries.
    ///
    /// Lock branch: at θ = ±π/2 the rotation genuinely contains one
    /// fewer observable degree of freedom in this parametrization —
    /// yaw and roll have become the same physical motion, and only
    /// ψ−φ (+lock) or ψ+φ (−lock) survives in the matrix. The
    /// returned **gauge** is: roll = 0, pitch = ±π/2 exactly, and yaw
    /// carries the whole observable angle. Any (ψ, φ) pair with the
    /// same difference (resp. sum) extracts to the identical result —
    /// the `lock_gauge_*` tests state this as executable fact.
    pub fn from_dcm(dcm: &Dcm) -> Self {
        use std::f64::consts::FRAC_PI_2;
        let m = dcm.matrix();
        let s = (-m.m31).clamp(-1.0, 1.0);
        let c = m.m11.hypot(m.m21);

        if c > GIMBAL_LOCK_EPS {
            Euler321 {
                yaw: m.m21.atan2(m.m11),
                pitch: s.asin(),
                roll: m.m32.atan2(m.m33),
            }
        } else {
            let yaw = if s > 0.0 {
                (-m.m12).atan2(m.m13)
            } else {
                (-m.m12).atan2(-m.m13)
            };
            let pitch = FRAC_PI_2.copysign(s);
            Euler321::new(yaw, pitch, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::relative_eq;
    use nalgebra::{Matrix3, Vector3};
    use proptest::prelude::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    // ---------------- helpers & strategies ----------------

    /// Reconstruct the raw matrix through the public action (columns are
    /// R·eᵢ) — independent of the `matrix()` accessor's exact signature.
    fn matrix_of(d: &Dcm) -> Matrix3<f64> {
        Matrix3::from_columns(&[
            d.transform(Vector3::x()),
            d.transform(Vector3::y()),
            d.transform(Vector3::z()),
        ])
    }

    /// The hand-derived closed form of Rz(ψ)·Ry(θ)·Rx(φ) — the Step-0(b)
    /// derivation, living here as an oracle so a transcription slip
    /// turns a test red instead of an answer wrong.
    fn closed_form_dcm(e: &Euler321) -> Matrix3<f64> {
        let (sy, cy) = e.yaw.sin_cos();
        let (sp, cp) = e.pitch.sin_cos();
        let (sr, cr) = e.roll.sin_cos();
        Matrix3::new(
            cy * cp,
            -sy * cr + cy * sp * sr,
            sy * sr + cy * sp * cr,
            sy * cp,
            cy * cr + sy * sp * sr,
            -cy * sr + sy * sp * cr,
            -sp,
            cp * sr,
            cp * cr,
        )
    }

    fn angle_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    /// Angles strictly inside the canonical box, edges trimmed: the ±π
    /// wrap on yaw/roll and the lock on pitch are representation
    /// artifacts, exercised by their own dedicated tests below.
    fn arb_euler_safe() -> impl Strategy<Value = Euler321> {
        (
            -PI + 0.05..PI - 0.05,
            -FRAC_PI_2 + 0.05..FRAC_PI_2 - 0.05,
            -PI + 0.05..PI - 0.05,
        )
            .prop_map(|(yaw, pitch, roll)| Euler321::new(yaw, pitch, roll))
    }

    /// Anything finite, including outside the canonical ranges —
    /// rotations don't wrap even when angles do.
    fn arb_euler_any() -> impl Strategy<Value = Euler321> {
        (
            -2.0 * PI..2.0 * PI,
            -2.0 * PI..2.0 * PI,
            -2.0 * PI..2.0 * PI,
        )
            .prop_map(|(yaw, pitch, roll)| Euler321::new(yaw, pitch, roll))
    }

    // ---------------- forward map: oracle & anchors ----------------

    proptest! {
        /// Production (composition through the quaternion) against the
        /// hand-derived closed form: paper and code certify each other.
        #[test]
        fn closed_form_matches_production(e in arb_euler_any()) {
            let produced = matrix_of(&e.to_dcm());
            prop_assert!(relative_eq!(produced, closed_form_dcm(&e), epsilon = 1e-12));
        }
    }

    /// Single-axis anchors assert the *action* — the asymmetric checks
    /// that catch sign and transposition errors every symmetric
    /// property forgives. Active convention: +90° yaw sends x̂ → ŷ;
    /// +90° pitch sends x̂ → −ẑ; +90° roll sends ŷ → ẑ.
    #[test]
    fn single_axis_anchors() {
        let yaw90 = Euler321::new(FRAC_PI_2, 0.0, 0.0).to_dcm();
        assert!(relative_eq!(
            yaw90.transform(Vector3::x()),
            Vector3::y(),
            epsilon = 1e-12
        ));

        let pitch90 = Euler321::new(0.0, FRAC_PI_2, 0.0).to_dcm();
        assert!(relative_eq!(
            pitch90.transform(Vector3::x()),
            -Vector3::z(),
            epsilon = 1e-12
        ));

        let roll90 = Euler321::new(0.0, 0.0, FRAC_PI_2).to_dcm();
        assert!(relative_eq!(
            roll90.transform(Vector3::y()),
            Vector3::z(),
            epsilon = 1e-12
        ));
    }

    // ---------------- round trips ----------------

    proptest! {
        /// Angle-level round trip on the safe box: every angle comes
        /// back elementwise.
        #[test]
        fn angle_roundtrip_safe_box(e in arb_euler_safe()) {
            let back = Euler321::from_dcm(&e.to_dcm());
            prop_assert!(angle_eq(back.yaw, e.yaw, 1e-9),   "yaw {} → {}", e.yaw, back.yaw);
            prop_assert!(angle_eq(back.pitch, e.pitch, 1e-9), "pitch {} → {}", e.pitch, back.pitch);
            prop_assert!(angle_eq(back.roll, e.roll, 1e-9),  "roll {} → {}", e.roll, back.roll);
        }
    }

    proptest! {
        /// Rotation-level round trip everywhere — no range restrictions,
        /// because rotations don't wrap even when angles do. This is the
        /// test that catches a swapped lock branch instantly.
        #[test]
        fn rotation_roundtrip_everywhere(e in arb_euler_any()) {
            let q1 = e.to_quat();
            let q2 = Euler321::from_dcm(&e.to_dcm()).to_quat();
            prop_assert!(q1.approx_eq_rotation(&q2, 1e-11));
        }
    }

    proptest! {
        /// Out-of-range inputs are valid: angles wrap onto the same
        /// rotation (the constructor's documented non-contract).
        #[test]
        fn out_of_range_angles_same_rotation(e in arb_euler_any()) {
            let shifted = Euler321::new(e.yaw + 2.0 * PI, e.pitch, e.roll - 2.0 * PI);
            prop_assert!(e.to_quat().approx_eq_rotation(&shifted.to_quat(), 1e-11));
        }
    }

    // ---------------- the lock contract ----------------

    /// +lock observability: only ψ − φ survives. Pairs with the same
    /// difference must extract to the *identical* gauge
    /// (yaw = ψ−φ, pitch = +π/2, roll = 0) — and round-trip as rotations.
    #[test]
    fn lock_gauge_difference_at_plus_lock() {
        let pairs = [(0.7, 0.2), (1.2, 0.7), (-0.3, -0.8)]; // all ψ−φ = 0.5
        for (yaw, roll) in pairs {
            let e = Euler321::new(yaw, FRAC_PI_2, roll);
            let got = Euler321::from_dcm(&e.to_dcm());
            assert!(
                angle_eq(got.yaw, 0.5, 1e-9),
                "yaw gauge: ψ−φ=0.5, got {}",
                got.yaw
            );
            assert!(angle_eq(got.pitch, FRAC_PI_2, 1e-9));
            assert!(angle_eq(got.roll, 0.0, 1e-9));
            assert!(e.to_quat().approx_eq_rotation(&got.to_quat(), 1e-11));
        }
    }

    /// −lock observability: only ψ + φ survives. Same statement with sums.
    #[test]
    fn lock_gauge_sum_at_minus_lock() {
        let pairs = [(0.7, -0.2), (0.2, 0.3), (-0.4, 0.9)]; // all ψ+φ = 0.5
        for (yaw, roll) in pairs {
            let e = Euler321::new(yaw, -FRAC_PI_2, roll);
            let got = Euler321::from_dcm(&e.to_dcm());
            assert!(
                angle_eq(got.yaw, 0.5, 1e-9),
                "yaw gauge: ψ+φ=0.5, got {}",
                got.yaw
            );
            assert!(angle_eq(got.pitch, -FRAC_PI_2, 1e-9));
            assert!(angle_eq(got.roll, 0.0, 1e-9));
            assert!(e.to_quat().approx_eq_rotation(&got.to_quat(), 1e-11));
        }
    }

    /// Machine-exact lock inputs return finite angles (the clamp test).
    #[test]
    fn exact_lock_is_finite() {
        for pitch in [FRAC_PI_2, -FRAC_PI_2] {
            for (yaw, roll) in [(0.0, 0.0), (1.0, -0.5), (-2.5, 2.0)] {
                let got = Euler321::from_dcm(&Euler321::new(yaw, pitch, roll).to_dcm());
                assert!(
                    got.yaw.is_finite() && got.pitch.is_finite() && got.roll.is_finite(),
                    "non-finite extraction at pitch {pitch}, ψ={yaw}, φ={roll}"
                );
            }
        }
    }

    // ---------------- the conditioning finding ----------------

    /// Approaching the lock, *angle* accuracy degrades like 1/cosθ while
    /// the *rotation* stays exact — the empirical half of the module's
    /// "why filters don't integrate Euler angles" paragraph. Run with
    /// `-- --nocapture` for the README-plot data.
    #[test]
    fn conditioning_sweep() {
        let mut worst_angle_err: f64 = 0.0;
        for k in 1..=12 {
            let pitch = FRAC_PI_2 - 10f64.powi(-k);
            let e = Euler321::new(0.4, pitch, -0.3);
            let back = Euler321::from_dcm(&e.to_dcm());

            let angle_err = (back.yaw - e.yaw)
                .abs()
                .max((back.pitch - e.pitch).abs())
                .max((back.roll - e.roll).abs());
            println!("π/2−θ = 1e-{k}:  worst angle err {angle_err:.3e}");
            worst_angle_err = worst_angle_err.max(angle_err);

            // The rotation itself must stay machine-exact throughout.
            assert!(
                e.to_quat().approx_eq_rotation(&back.to_quat(), 1e-11),
                "rotation drifted at π/2−θ = 1e-{k}"
            );
        }
        // Angle error is *allowed* to grow — that's the finding. Assert
        // only that the growth is real, so the test fails if someone
        // "fixes" the sweep into meaninglessness.
        assert!(
            worst_angle_err > 1e-9,
            "expected visible 1/cosθ degradation, saw only {worst_angle_err:e}"
        );
    }
}
