//! Direction cosine matrix (rotation matrix) representation.
//!
//! # Conventions
//!
//! | Decision          | Choice                                                  |
//! |-------------------|---------------------------------------------------------|
//! | Operator sense    | **Active** — `transform` rotates the vector, frame fixed |
//! | Composition order | `a.compose(&b)` applies **`b` first**, then `a` (i.e. the matrix product `A·B`) — identical to `quat::UnitQuat::compose` |
//! | Storage           | Wraps `nalgebra::Matrix3<f64>` (column-major internally; irrelevant through this API, relevant if you ever touch raw storage) |
//! | Validity          | `new` **trusts the caller** to supply a member of SO(3); use [`Dcm::is_orthonormal`] to validate untrusted input |
//!
//! # Relationship to `quat`
//!
//! `UnitQuat::to_dcm` and [`Dcm::to_quat`] are mutual inverses *as
//! rotations*; round trips may land on the opposite quaternion hemisphere
//! (see [`Dcm::to_quat`]), so compare results with
//! `UnitQuat::approx_eq_rotation`, never componentwise.

use nalgebra::{Matrix3, Vector3};

use crate::rotvec::RotVec;

/// A rotation in SO(3) represented as a 3×3 direction cosine matrix.
///
/// Invariant (by caller contract, not enforced): orthonormal with
/// determinant +1. All methods assume it; [`Dcm::is_orthonormal`] checks it.
pub struct Dcm(Matrix3<f64>);

impl Dcm {
    /// Wraps a matrix **without validation**.
    ///
    /// The caller guarantees `dcm ∈ SO(3)` (orthonormal columns,
    /// determinant +1). Feeding a non-rotation here makes every other
    /// method silently wrong — when in doubt, gate on
    /// [`Dcm::is_orthonormal`] first.
    pub fn new(dcm: Matrix3<f64>) -> Self {
	Self(dcm)
    }

    pub fn matrix(&self) -> &Matrix3<f64> {
	&self.0
    }
}

impl Dcm {
    /// Active rotation of `v` (frame fixed, vector rotated): returns `R·v`.
    ///
    /// Agrees with `UnitQuat::transform` of the corresponding quaternion;
    /// pinned by the `action_agreement` property test.
    pub fn transform(&self, v: Vector3<f64>) -> Vector3<f64> {
	self.0 * v
    }

    /// Composition `self · other`: applies `other` first, then `self` —
    /// the same order contract as `UnitQuat::compose`, so conversions
    /// commute with composition (the `homomorphism` property test).
    pub fn compose(&self, other: &Self) -> Self {
	Self(self.0 * other.0)
    }

    /// The transpose — which on SO(3) **is** the inverse, since
    /// orthonormality means `RᵀR = I`. This is why inverting a DCM is
    /// free of arithmetic beyond a relabeling.
    pub fn transpose(&self) -> Self {
	Self(self.0.transpose())
    }

    /// Membership test for SO(3) within tolerance.
    ///
    /// Checks `max |(RᵀR − I)ᵢⱼ| < tol` (orthonormality) **and**
    /// `det R > 0` (excludes reflections; given orthonormality the
    /// determinant is ±1, so the sign test is all that remains).
    ///
    /// `tol` bounds the worst single entry of the Gram defect; for
    /// matrices produced by composing valid rotations, defects sit near
    /// machine epsilon, so `1e-9` is a generous gate.
    pub fn is_orthonormal(&self, tol: f64) -> bool {
	(self.0.transpose() * self.0 - Matrix3::identity()).abs().max() < tol
        && self.0.determinant() > 0.0
    }

    /// Quaternion of the same rotation, via **Shepperd's method**.
    ///
    /// # Why four branches
    ///
    /// The matrix entries determine the quaternion through
    /// `4w² = 1 + tr R`, `4x² = 1 + 2r₁₁ − tr R` (and cyclic), plus
    /// pair products `4wx = r₃₂ − r₂₃`, `4xy = r₂₁ + r₁₂` (and cyclic).
    /// Solving any one quadratic and dividing for the rest is a valid
    /// chart on S³ — but each chart is singular where its component
    /// vanishes (the naive trace-only method loses ~half its digits as
    /// θ → π, where w → 0).
    ///
    /// Shepperd selects the chart of the **largest** component. The
    /// selector costs nothing: `tr R − r₁₁ = 2(w² − x²)` and
    /// `r₁₁ − r₂₂ = 2(x² − y²)`, so comparing the trace against the
    /// diagonal entries identifies `max(w², x², y², z²)` exactly. Since
    /// the squares sum to 1, the winner is ≥ ¼: its square-root argument
    /// is ≥ 1 (no cancellation) and the divisor 4|q| ≥ 2 (no error
    /// amplification). Accuracy is uniform over SO(3), including θ = π —
    /// see the `four_branch_pi_rotations` and `near_pi_roundtrip` tests.
    ///
    /// # Hemisphere
    ///
    /// Each branch takes the positive root of *its* component, so the
    /// output sign convention depends on which branch fired; the result
    /// is **not** canonicalized to `w ≥ 0`. Compare with
    /// `approx_eq_rotation`, which absorbs the double cover.
    ///
    /// Reference: S. W. Shepperd, *J. Guidance and Control* 1(3), 1978.
    pub fn to_quat(&self) -> crate::quat::UnitQuat {
	let t = self.0.trace();
	let r11 = self.0.m11;
	let r22 = self.0.m22;
	let r33 = self.0.m33;
	let r12 = self.0.m12;
	let r13 = self.0.m13;
	let r21 = self.0.m21;
	let r23 = self.0.m23;
	let r31 = self.0.m31;
	let r32 = self.0.m32;
	
	if t >= r11.max(r22).max(r33) {
	    let w = (1.0 + t).sqrt() * 0.5;
	    let x = (r32 - r23) / w * 0.25;
	    let y = (r13 - r31) / w * 0.25;
	    let z = (r21 - r12) / w * 0.25;
	    crate::quat::UnitQuat::new(w, x, y, z)
	} else if r11 >= t.max(r22).max(r33) {
	    let x = (1.0 + 2.0*r11 - t).sqrt() * 0.5;
	    let y = (r21 + r12) / x * 0.25;
	    let z = (r13 + r31) / x * 0.25;
	    let w = (r32 - r23) / x * 0.25;
	    crate::quat::UnitQuat::new(w, x, y, z)
	} else if r22 >= t.max(r11).max(r33) {
	    let y = (1.0 + 2.0*r22 - t).sqrt() * 0.5;
	    let x = (r21 + r12) / y * 0.25;
	    let z = (r32 + r23) / y * 0.25;
	    let w = (r13 - r31) / y * 0.25;
	    crate::quat::UnitQuat::new(w, x, y, z)
	} else {
	    let z = (1.0 + 2.0*r33 - t).sqrt() * 0.5;
	    let x = (r13 + r31) / z * 0.25;
	    let y = (r32 + r23) / z * 0.25;
	    let w = (r21 - r12) / z * 0.25;
	    crate::quat::UnitQuat::new(w, x, y, z)
	}
    }

    /// Rotation vector (so(3) coordinates) of this rotation — the inverse
    /// of `RotVec::exp` on the principal branch, |φ| ≤ π.
    ///
    /// One line by design: routed through Shepperd (`to_quat`, uniformly
    /// accurate over SO(3), no bad region at θ = π) and `to_rotvec`
    /// (atan2 + Taylor). The quaternion is the numerically privileged
    /// chart; conversions transit through it. The direct matrix log —
    /// θ from the trace, axis from the antisymmetric part — divides by
    /// sinθ and collapses near π; see rotvec's
    /// `naive_log_degrades_near_pi_quat_path_does_not` for the comparison.
    pub fn log(&self) -> RotVec {
	self.to_quat().to_rotvec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat::UnitQuat;
    use approx::relative_eq;
    use nalgebra::{Unit, Vector3};
    use proptest::prelude::*;
    use std::f64::consts::PI;

    /// Uniform-ish random rotations: random axis, angle in (−π, π).
    /// Statistically exercises all four Shepperd branches (each fires
    /// when its component dominates).
    fn arb_rotation() -> impl Strategy<Value = UnitQuat> {
        (prop::array::uniform3(-1.0f64..1.0), -PI..PI).prop_filter_map(
            "axis too short to normalize",
            |(a, angle)| {
                let v = Vector3::from(a);
                (v.norm() > 1e-3)
                    .then(|| UnitQuat::from_axis_angle(Unit::new_normalize(v), angle))
            },
        )
    }

    /// Rotations within 10⁻³ of θ = π: the regime where the naive
    /// (trace-only) extraction loses digits and Shepperd must not.
    fn arb_near_pi_rotation() -> impl Strategy<Value = UnitQuat> {
        (
            prop::array::uniform3(-1.0f64..1.0),
            (PI - 1e-3)..(PI - 1e-12),
            prop::bool::ANY,
        )
            .prop_filter_map("axis too short to normalize", |(a, angle, neg)| {
                let v = Vector3::from(a);
                let angle = if neg { -angle } else { angle };
                (v.norm() > 1e-3)
                    .then(|| UnitQuat::from_axis_angle(Unit::new_normalize(v), angle))
            })
    }

    fn arb_vec3() -> impl Strategy<Value = Vector3<f64>> {
        prop::array::uniform3(-10.0f64..10.0).prop_map(Vector3::from)
    }

    // ---------------- property tests ----------------

    proptest! {
        /// q → DCM → q′ must be the same rotation. Compared with
        /// `approx_eq_rotation`: Shepperd's output hemisphere is
        /// branch-dependent, so componentwise comparison is *wrong by
        /// design* here.
        #[test]
        fn roundtrip_is_same_rotation(q in arb_rotation()) {
            let q2 = q.to_dcm().to_quat();
            prop_assert!(q.approx_eq_rotation(&q2, 1e-11));
        }
    }

    proptest! {
        /// The two representations act identically on vectors.
        #[test]
        fn action_agreement(q in arb_rotation(), v in arb_vec3()) {
            let via_dcm = q.to_dcm().transform(v);
            let via_quat = q.transform(&v);
            prop_assert!(relative_eq!(via_dcm, via_quat, epsilon = 1e-9));
        }
    }

    proptest! {
        /// Conversion is a group homomorphism:
        /// R(q₂ ⊗ q₁) = R(q₂) · R(q₁).
        #[test]
        fn homomorphism(q1 in arb_rotation(), q2 in arb_rotation()) {
            let lhs = q2.compose(&q1).to_dcm();
            let rhs = q2.to_dcm().compose(&q1.to_dcm());
            prop_assert!(relative_eq!(lhs.0, rhs.0, epsilon = 1e-9));
        }
    }

    proptest! {
        /// SO(3) is closed under composition, and the transpose is the
        /// inverse.
        #[test]
        fn closure_and_transpose_inverse(q1 in arb_rotation(), q2 in arb_rotation()) {
            let r = q1.to_dcm().compose(&q2.to_dcm());
            prop_assert!(r.is_orthonormal(1e-9));

            let should_be_identity = r.compose(&r.transpose());
            prop_assert!(relative_eq!(
                should_be_identity.0,
                Matrix3::identity(),
                epsilon = 1e-9
            ));
        }
    }

    proptest! {
        /// Shepperd's stress regime: θ within 10⁻³ of π. The naive
        /// extraction degrades like 1/(π−θ) here; Shepperd must stay at
        /// machine precision.
        #[test]
        fn near_pi_roundtrip(q in arb_near_pi_rotation()) {
            let q2 = q.to_dcm().to_quat();
            prop_assert!(q.approx_eq_rotation(&q2, 1e-11));
        }
    }

    // ---------------- deterministic branch & contract tests ----------------

    /// One rotation per Shepperd branch, each at that branch's *worst
    /// case*: θ = π about a coordinate axis zeroes w and two of the three
    /// vector components, so every divisor except the branch's own would
    /// be exactly zero. The pre-fix z-branch bug (dividing by y) panics
    /// on the third case below — this test is its regression lock.
    #[test]
    fn four_branch_pi_rotations() {
        let cases = [
            Vector3::x_axis(), // x-branch
            Vector3::y_axis(), // y-branch
            Vector3::z_axis(), // z-branch
        ];
        for axis in cases {
            let q = UnitQuat::from_axis_angle(axis, PI);
            let q2 = q.to_dcm().to_quat();
            assert!(
                q.approx_eq_rotation(&q2, 1e-12),
                "π rotation about {:?} failed round trip",
                axis
            );
        }
        // w-branch: identity (t = 3 dominates).
        let id = UnitQuat::IDENTITY;
        assert!(id.approx_eq_rotation(&id.to_dcm().to_quat(), 1e-12));
    }

    /// Physical anchor for the conventions table: an active +90° yaw
    /// (about +z) sends x̂ to ŷ. A transposed or passive-convention
    /// matrix sends it to −ŷ; every symmetric property above would pass
    /// anyway, which is why this test exists.
    #[test]
    fn yaw_90_sends_x_to_y() {
        let q = UnitQuat::from_axis_angle(Vector3::z_axis(), PI / 2.0);
        let rotated = q.to_dcm().transform(Vector3::x());
        assert!(relative_eq!(rotated, Vector3::y(), epsilon = 1e-12));
    }

    /// `is_orthonormal` must reject det = +1 shears (orthonormality
    /// clause) and orthonormal reflections (determinant clause).
    #[test]
    fn is_orthonormal_rejects_non_rotations() {
        let shear = Dcm::new(Matrix3::new(
            1.0, 5.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        )); // det = 1, not orthogonal
        assert!(!shear.is_orthonormal(1e-9));

        let reflection = Dcm::new(Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, -1.0,
        )); // orthonormal, det = −1
        assert!(!reflection.is_orthonormal(1e-9));

        let rotation = UnitQuat::from_axis_angle(Vector3::z_axis(), 0.3).to_dcm();
        assert!(rotation.is_orthonormal(1e-9));
    }
}
