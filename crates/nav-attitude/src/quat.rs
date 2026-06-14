//! Unit quaternion attitude representation.
//!
//! # Conventions (the contract for this module)
//!
//! | Decision            | Choice                                              |
//! |---------------------|-----------------------------------------------------|
//! | Quaternion algebra  | **Hamilton** (`ij = k`, right-handed)               |
//! | Storage order       | **Scalar-first**: `[w, x, y, z]`                    |
//! | Operator sense      | **Active**: `transform` rotates the vector, frame fixed |
//! | Composition order   | `a.compose(&b)` applies **`b` first**, then `a` (matches matrix convention `R_a · R_b`) |
//! | Double cover        | `q` and `-q` denote the same rotation; `PartialEq`, the `approx` impls, and `approx_eq_rotation` all identify them |
//! | Angle units         | Radians, right-hand rule about the axis             |
//!
//! As `dcm`, `euler`, and `rotvec` mature, the crate-wide version of this
//! table lives in `lib.rs`; entries here are scoped to quaternions.
//!
//! # Invariant
//!
//! Every `UnitQuat` reachable through this API has unit norm: constructors
//! normalize (`new`) or produce unit values by construction
//! (`from_axis_angle`, `IDENTITY`), and `compose` renormalizes its result.

use approx::{AbsDiffEq, RelativeEq};
use nalgebra::{Matrix3, Unit, Vector3};

/// A unit quaternion representing a rotation in SO(3).
///
/// Stored **scalar-first** as `[w, x, y, z]`, Hamilton convention, used as
/// an **active** rotation operator (see module docs for the full table).
///
/// Unit quaternions form SU(2), the double cover of SO(3): `q` and `-q`
/// encode the same rotation. All equality and comparison provided here
/// (`PartialEq`, `AbsDiffEq`, `RelativeEq`, [`Self::approx_eq_rotation`])
/// respect that identification.
#[derive(Clone, Copy, Debug)]
pub struct UnitQuat([f64; 4]);

/// Exact component equality **up to global sign**: `q == p` iff their
/// components are bitwise-equal floats, or exactly negated.
///
/// This makes `==` agree with rotation semantics while remaining an exact
/// comparison — useful for identities and constants, not for numerics
/// (use [`RelativeEq`] or [`UnitQuat::approx_eq_rotation`] for results of
/// floating-point computation). Any `NaN` component makes `eq` return
/// `false`, so `Eq` is deliberately not implemented.
impl PartialEq for UnitQuat {
    fn eq(&self, other: &Self) -> bool {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = other.0;
        let same = w1 == w2 && x1 == x2 && y1 == y2 && z1 == z2;
        let neg = w1 == -w2 && x1 == -x2 && y1 == -y2 && z1 == -z2;
        same || neg
    }
}

impl UnitQuat {
    /// The identity rotation, `[1, 0, 0, 0]`.
    pub const IDENTITY: Self = Self([1.0, 0., 0., 0.]);

    /// Rotation by `angle` radians about `axis`, right-hand rule.
    ///
    /// The axis is unit-length by type (`Unit<Vector3>`). Any finite angle
    /// is accepted; the half-angle trigonometry wraps it onto the double
    /// cover (e.g. `angle = 2π` yields `-IDENTITY`, the same rotation as
    /// `IDENTITY`).
    pub fn from_axis_angle(axis: &Unit<Vector3<f64>>, angle: f64) -> Self {
        let half_angle = angle * 0.5;
        let sin = half_angle.sin();
        let cos = half_angle.cos();

        let n = axis.into_inner() * sin;
        let x = n.x;
        let y = n.y;
        let z = n.z;

        Self([cos, x, y, z])
    }

    /// Normalizing constructor from raw components (scalar first).
    ///
    /// Restores the unit invariant after numerical drift, and is the
    /// renormalization step inside [`Self::compose`].
    ///
    /// # Panics
    ///
    /// Panics if all four components are exactly zero. Inputs are otherwise
    /// not validated: non-finite components, or magnitudes extreme enough
    /// that `w² + x² + y² + z²` overflows or vanishes in `f64`, propagate
    /// non-finite values into the result. Callers supply finite,
    /// non-degenerate components.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        let norm_sq = w * w + x * x + y * y + z * z;
        assert!(norm_sq > 0.0);

        let norm = norm_sq.sqrt();

        Self([w, x, y, z].map(|n| n / norm))
    }

    /// Hamilton product `self ⊗ other`.
    ///
    /// **Order contract:** as an active rotation, the result applies
    /// `other` *first*, then `self` — i.e.
    /// `a.compose(&b).transform(v) == a.transform(&b.transform(v))`,
    /// mirroring matrix composition `R_a · R_b`. This is pinned by the
    /// `composition_matches_sequential_application` property test.
    ///
    /// The result is renormalized (via [`Self::new`]) so that repeated
    /// composition cannot drift off the unit sphere; the cost is one
    /// square root per call.
    pub fn compose(&self, other: &Self) -> Self {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = other.0;

        let w = w1 * w2 - (x1 * x2 + y1 * y2 + z1 * z2);
        let x = w1 * x2 + w2 * x1 + y1 * z2 - z1 * y2;
        let y = w1 * y2 + w2 * y1 + z1 * x2 - x1 * z2;
        let z = w1 * z2 + w2 * z1 + x1 * y2 - y1 * x2;

        Self::new(w, x, y, z)
    }

    /// The inverse rotation.
    ///
    /// Implemented as the conjugate `[w, -x, -y, -z]`, which equals the
    /// inverse precisely because of the unit-norm invariant. Does not
    /// renormalize.
    pub fn inverse(&self) -> Self {
        let [w, x, y, z] = self.0;
        Self([w, -x, -y, -z])
    }

    /// Vector (imaginary) part `[x, y, z]`.
    fn v(&self) -> Vector3<f64> {
        let [_, x, y, z] = self.0;
        Vector3::new(x, y, z)
    }

    /// Active rotation of `v`: returns `q v q⁻¹` with `v` embedded as a
    /// pure quaternion.
    ///
    /// Uses the expanded form `v + w·t + q_v × t` with `t = 2 q_v × v`,
    /// which is algebraically identical to the sandwich product but
    /// cheaper (no intermediate quaternion).
    ///
    /// Under the passive reading, this maps coordinates *from* the frame
    /// this quaternion rotates *to* — if you need that direction
    /// explicitly, prefer going through `to_dcm` with labeled frames once
    /// `frames.rs` lands.
    pub fn transform(&self, v: &Vector3<f64>) -> Vector3<f64> {
        let w = self.0[0];
        let qv = self.v();

        let t = 2.0 * qv.cross(v);

        v + w * t + qv.cross(&t)
    }

    /// Four-component inner product ⟨self, other⟩.
    ///
    /// For unit quaternions this equals `cos(Δθ/2)`, where `Δθ` is the
    /// angle of the relative rotation between the two.
    fn dot(&self, other: &Self) -> f64 {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = other.0;

        w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2
    }

    /// True if `self` and `other` represent the same rotation within `tol`,
    /// honoring the double cover (SU(2) → SO(3): `q` ≡ `-q`).
    ///
    /// The metric is `1 − |⟨self, other⟩| < tol`. Since
    /// `⟨q₁, q₂⟩ = cos(Δθ/2)`, small relative angles satisfy
    /// `1 − |cos(Δθ/2)| ≈ Δθ²/8`, so a tolerance maps to an angular
    /// disagreement of about `Δθ ≈ √(8·tol)` — e.g. `tol = 1e-12`
    /// accepts rotations within ≈ 2.8·10⁻⁶ rad (≈ 0.6 arcsec).
    pub fn approx_eq_rotation(&self, other: &Self, tol: f64) -> bool {
        (1.0 - self.dot(other).abs()) < tol
    }

    /// The direction cosine matrix of the same **active** rotation:
    /// `q.to_dcm() · v == q.transform(&v)` for all `v`.
    ///
    /// Orthonormal with determinant +1 up to floating-point error
    /// inherited from the unit invariant.
    pub fn to_dcm(&self) -> crate::dcm::Dcm {
        let [w, x, y, z] = self.0;
        let r11 = 1.0 - 2.0 * (y * y + z * z);
        let r12 = 2.0 * (x * y - w * z);
        let r13 = 2.0 * (x * z + w * y);

        let r21 = 2.0 * (x * y + w * z);
        let r22 = 1.0 - 2.0 * (x * x + z * z);
        let r23 = 2.0 * (y * z - w * x);

        let r31 = 2.0 * (x * z - w * y);
        let r32 = 2.0 * (y * z + w * x);
        let r33 = 1.0 - 2.0 * (x * x + y * y);

        let matrix = Matrix3::new(r11, r12, r13, r21, r22, r23, r31, r32, r33);

        crate::dcm::Dcm::new(matrix)
    }

    /// The **minimal rotation vector** φ = θ·û of this rotation, with
    /// the guarantee `|φ| ≤ π`.
    ///
    /// The quaternion is first canonicalized onto the `w ≥ 0` hemisphere
    /// (double cover), so a near-2π rotation comes back as its short-way
    /// equivalent rather than a vector of length ≈ 2π. Downstream code —
    /// in particular the error-state filter in Phase 4 — relies on this
    /// minimality contract.
    ///
    /// Numerics: the scale factor θ/sin(θ/2) is evaluated with
    /// `2·atan2(s, w)/s` (well-conditioned everywhere, unlike `acos`
    /// near `|w| = 1`), switching to its Taylor series below
    /// `SMALL_ANGLE_EPS`; the zero rotation maps to the zero vector.
    pub fn to_rotvec(&self) -> crate::rotvec::RotVec {
        let w = self.0[0];
        let v = self.v();
        let s = v.norm();

        let (w, v) = if w < 0.0 { (-w, v * -1.0) } else { (w, v) };

        let scale = if s < crate::numerics::SMALL_ANGLE_EPS {
            2.0 / w * (1.0 - s * s / (3.0 * w * w))
        } else {
            2.0 * s.atan2(w) / s
        };

        crate::rotvec::RotVec::new(v * scale)
    }

    pub fn from_rotvec(v: crate::rotvec::RotVec) -> Self {
        let theta = v.angle();

        let w = (0.5 * theta).cos();
        let c = if theta < crate::numerics::SMALL_ANGLE_EPS {
            let theta_sq = theta * theta;
            0.5 - theta_sq / 48.0 + theta_sq * theta_sq / 3840.0
        } else {
            (0.5 * theta).sin() / theta
        };

        let u = c * v.to_vector();

        Self([w, u.x, u.y, u.z])
    }
}

/// Component-wise absolute comparison **up to global sign**, consistent
/// with the double cover (matches [`PartialEq`] semantics, with slack).
impl AbsDiffEq for UnitQuat {
    type Epsilon = f64;

    fn default_epsilon() -> Self::Epsilon {
        f64::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let same = self
            .0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| (*a - *b).abs() <= epsilon);

        let neg = self
            .0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| (*a + *b).abs() <= epsilon);

        same || neg
    }
}

/// Component-wise relative comparison **up to global sign**; see
/// [`AbsDiffEq`] above. Prefer [`UnitQuat::approx_eq_rotation`] when the
/// quantity of interest is the rotation itself rather than the
/// representation.
impl RelativeEq for UnitQuat {
    fn default_max_relative() -> Self::Epsilon {
        f64::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        let same = self.0.iter().zip(other.0.iter()).all(|(a, b)| {
            approx::relative_eq!(*a, *b, epsilon = epsilon, max_relative = max_relative)
        });

        let neg = self.0.iter().zip(other.0.iter()).all(|(a, b)| {
            approx::relative_eq!(*a, -*b, epsilon = epsilon, max_relative = max_relative)
        });

        same || neg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use nalgebra::{Unit, Vector3};
    use proptest::prelude::*;

    fn nonzero_vec3(v: [f64; 3]) -> Option<Vector3<f64>> {
        let v = Vector3::from(v);
        (v.norm() > 1e-8).then_some(v)
    }

    proptest! {
    #[test]
    fn rotation_preserves_norm(
            axis in prop::array::uniform3(-10.0f64..10.0),
        vec  in prop::array::uniform3(-10.0f64..10.0),
        angle in -std::f64::consts::PI..std::f64::consts::PI,
    ) {
            let Some(axis) = nonzero_vec3(axis) else {
        return Ok(());
            };

            let q = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis),
        angle,
            );

            let v = Vector3::from(vec);
            let r = q.transform(&v);

            prop_assert!(
        relative_eq!(
                    v.norm(),
                    r.norm(),
                    epsilon = 1e-10,
        )
            );
    }
    }

    proptest! {
    #[test]
    fn inverse_recovers_vector(
            axis in prop::array::uniform3(-10.0f64..10.0),
        vec  in prop::array::uniform3(-10.0f64..10.0),
        angle in -std::f64::consts::PI..std::f64::consts::PI,
    ) {
            let Some(axis) = nonzero_vec3(axis) else {
        return Ok(());
            };

            let q = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis),
        angle,
            );

            let v = Vector3::from(vec);

            let recovered =
        q.inverse()
        .transform(&q.transform(&v));

            prop_assert!(
        relative_eq!(
                    recovered,
                    v,
                    epsilon = 1e-10,
        )
            );
    }
    }

    proptest! {
    #[test]
    fn composition_matches_sequential_application(
        axis1 in prop::array::uniform3(-10.0f64..10.0),
        angle1 in -std::f64::consts::PI..std::f64::consts::PI,
        axis2 in prop::array::uniform3(-10.0f64..10.0),
        angle2 in -std::f64::consts::PI..std::f64::consts::PI,
        vec  in prop::array::uniform3(-10.0f64..10.0),
    ) {
            let Some(axis1) = nonzero_vec3(axis1) else {
        return Ok(());
            };

            let Some(axis2) = nonzero_vec3(axis2) else {
        return Ok(());
            };

            let q1 = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis1),
        angle1,
            );

            let q2 = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis2),
        angle2,
            );

            let v = Vector3::from(vec);

            let sequential =
        q2.transform(
                    &q1.transform(&v)
        );

            let composed =
        q2.compose(&q1)
        .transform(&v);

            prop_assert!(
        relative_eq!(
                    sequential,
                    composed,
                    epsilon = 1e-10,
        )
            );
    }
    }

    proptest! {
    #[test]
    fn identity_is_neutral(
        axis in prop::array::uniform3(-10.0f64..10.0),
            angle in -10.0f64..10.0,
    ) {
            let Some(axis) = nonzero_vec3(axis) else {
        return Ok(());
            };

            let q = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis),
        angle,
            );

            prop_assert!(
        q.compose(&UnitQuat::IDENTITY)
            .approx_eq_rotation(&q, 1e-12)
            );

            prop_assert!(
        UnitQuat::IDENTITY
                    .compose(&q)
                    .approx_eq_rotation(&q, 1e-12)
            );
    }
    }

    proptest! {
    #[test]
    fn q_and_minus_q_give_same_rotation(
            axis in prop::array::uniform3(-10.0f64..10.0),
        vec  in prop::array::uniform3(-10.0f64..10.0),
        angle in -std::f64::consts::PI..std::f64::consts::PI,
    ) {
            let Some(axis) = nonzero_vec3(axis) else {
        return Ok(());
            };

            let q = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis),
        angle,
            );

            let neg_q = UnitQuat([
        -q.0[0],
        -q.0[1],
        -q.0[2],
        -q.0[3],
            ]);

            let v = Vector3::from(vec);

            let r1 = q.transform(&v);
            let r2 = neg_q.transform(&v);

            prop_assert!(
        relative_eq!(
                    r1,
                    r2,
                    epsilon = 1e-10,
        )
            );

        prop_assert!(q.approx_eq_rotation(&neg_q, 1e-12))
    }
    }

    proptest! {
    #[test]
    fn composition_produces_unit_quaternion(
        axis1 in prop::array::uniform3(-10.0f64..10.0),
        angle1 in -std::f64::consts::PI..std::f64::consts::PI,
        axis2 in prop::array::uniform3(-10.0f64..10.0),
        angle2 in -std::f64::consts::PI..std::f64::consts::PI,
    ) {
            let Some(axis1) = nonzero_vec3(axis1) else {
        return Ok(());
            };

            let Some(axis2) = nonzero_vec3(axis2) else {
        return Ok(());
            };

            let q1 = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis1),
        angle1,
            );

            let q2 = UnitQuat::from_axis_angle(
        &Unit::new_normalize(axis2),
        angle2,
            );

            let q = q1.compose(&q2);

            let norm =
        q.0[0]*q.0[0]
        + q.0[1]*q.0[1]
        + q.0[2]*q.0[2]
        + q.0[3]*q.0[3];

            prop_assert!(
        relative_eq!(norm, 1.0, epsilon = 1e-12)
            );
    }
    }
}
