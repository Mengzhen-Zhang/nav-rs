use nalgebra::{Matrix3, Vector3};

/// Shared branch point for every small-angle Taylor series in the
/// crate: θ/sin(θ/2) in `UnitQuat::to_rotvec`, sin(θ/2)/θ in
/// `UnitQuat::from_rotvec`, and both Rodrigues coefficients in
/// `RotVec::exp`.
///
/// Worst truncation error among them at the seam is ~s⁴/5 ≈ 2·10⁻²⁵
/// at θ = 1e-6 — nine orders below f64 epsilon, so both branches of
/// every series agree to full precision where they meet. Certified by
/// the seam tests in `quat.rs` and `rotvec.rs`; tests should straddle
/// this constant.
pub(crate) const SMALL_ANGLE_EPS: f64 = 1e-6;

pub(crate) fn hat(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
pub(crate) fn vee(mat: &Matrix3<f64>) -> Vector3<f64> {
    let x = mat.m32 - mat.m23;
    let y = mat.m13 - mat.m31;
    let z = mat.m21 - mat.m12;
    Vector3::new(x, y, z) * 0.5
}
