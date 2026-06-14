//! Cross-checks against `nalgebra`, used as an independent oracle.
//!
//! `nalgebra` is also Hamilton, but stores quaternions **scalar-last**
//! `[x, y, z, w]` while this crate stores them **scalar-first**. So
//! quaternions are compared by their *action* on a vector, never by raw
//! components — a component-wise check would spuriously fail even when
//! the rotations are identical. Rotation matrices have no storage
//! ambiguity (column-major 3x3 either way), so those are compared
//! entry-wise.

use approx::assert_relative_eq;
use nalgebra::{Rotation3, Unit, UnitQuaternion, Vector3};

use nav_attitude::dcm::Dcm;
use nav_attitude::euler::Euler321;
use nav_attitude::quat::UnitQuat;
use nav_attitude::rotvec::RotVec;

/// Agreement tolerance. Loose enough to survive the worst-conditioned
/// case here (the `log` near θ = 3 rad), tight enough that any genuine
/// convention mismatch — which flips signs or reorders axes — produces
/// an O(1) difference that blows clean through it.
const EPS: f64 = 1e-9;

fn axes() -> [Unit<Vector3<f64>>; 5] {
    [
        Vector3::x_axis(),
        Vector3::y_axis(),
        Vector3::z_axis(),
        Unit::new_normalize(Vector3::new(1.0, 2.0, 3.0)),
        Unit::new_normalize(Vector3::new(-2.0, 0.5, 1.5)),
    ]
}

/// Test angles, all kept in (0, π) so that `log` / `scaled_axis` both
/// land on the same principal branch (the minimal rotation vector).
const ANGLES: [f64; 5] = [0.1, 0.7, 1.5, 2.5, 3.0];

fn probes() -> [Vector3<f64>; 3] {
    [
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.5, -1.0, 2.0),
        Vector3::new(-3.0, 1.0, 0.2),
    ]
}

/// 1. Axis-angle agreement — compared by action on a vector, since the
///    two libraries disagree on quaternion storage order.
#[test]
fn axis_angle_matches_nalgebra() {
    for axis in &axes() {
        for &angle in &ANGLES {
            let mine = UnitQuat::from_axis_angle(axis, angle);
            let theirs = UnitQuaternion::from_axis_angle(axis, angle);
            for v in &probes() {
                assert_relative_eq!(mine.transform(v), theirs.transform_vector(v), epsilon = EPS);
            }
        }
    }
}

/// 2. Composition order. `a.compose(&b)` applies `b` first; nalgebra's
///    `qa * qb` also applies `qb` first. If this fails, the crate's
///    compose order disagrees with the ecosystem — a real finding.
#[test]
fn composition_order_matches_nalgebra() {
    let a_axis = Vector3::x_axis();
    let b_axis = Vector3::y_axis();
    let c_axis = Unit::new_normalize(Vector3::new(1.0, 1.0, 1.0));

    let mine_a = UnitQuat::from_axis_angle(&a_axis, 0.6);
    let mine_b = UnitQuat::from_axis_angle(&b_axis, 1.1);
    let mine_c = UnitQuat::from_axis_angle(&c_axis, 0.9);

    let na_a = UnitQuaternion::from_axis_angle(&a_axis, 0.6);
    let na_b = UnitQuaternion::from_axis_angle(&b_axis, 1.1);
    let na_c = UnitQuaternion::from_axis_angle(&c_axis, 0.9);

    // Two-term and three-term composites, to pin associativity too.
    let mine_ab = mine_a.compose(&mine_b);
    let na_ab = na_a * na_b;

    let mine_abc = mine_a.compose(&mine_b.compose(&mine_c));
    let na_abc = na_a * (na_b * na_c);

    for v in &probes() {
        assert_relative_eq!(
            mine_ab.transform(v),
            na_ab.transform_vector(v),
            epsilon = EPS
        );
        assert_relative_eq!(
            mine_abc.transform(v),
            na_abc.transform_vector(v),
            epsilon = EPS
        );
    }
}

/// 3. DCM from quaternion. Rotation matrices have no storage ambiguity,
///    so this is an entry-wise check of the whole `to_dcm` formula.
///    `nalgebra`'s quaternion is built from the same axis-angle (the
///    crate exposes no component accessor), then turned into a matrix
///    via `Rotation3::from`.
#[test]
fn to_dcm_matches_nalgebra() {
    for axis in &axes() {
        for &angle in &ANGLES {
            let mine = UnitQuat::from_axis_angle(axis, angle).to_dcm();
            let theirs = Rotation3::from(UnitQuaternion::from_axis_angle(axis, angle));
            assert_relative_eq!(*mine.matrix(), *theirs.matrix(), epsilon = EPS);
        }
    }
}

/// 4a. `exp`: `RotVec::new(φ).exp()` is the rotation of the scaled axis
///     φ, which is exactly `Rotation3::new(φ)`.
#[test]
fn exp_matches_nalgebra() {
    for axis in &axes() {
        for &angle in &ANGLES {
            let phi = axis.into_inner() * angle;
            let mine = RotVec::new(phi).exp();
            let theirs = Rotation3::new(phi);
            assert_relative_eq!(*mine.matrix(), *theirs.matrix(), epsilon = EPS);
        }
    }
}

/// 4b. `log`: `dcm.log()` is the rotation vector, which is nalgebra's
///     `scaled_axis()`. Angles stay in (0, π) so both pick the minimal
///     vector on the principal branch.
#[test]
fn log_matches_nalgebra() {
    for axis in &axes() {
        for &angle in &ANGLES {
            let phi = axis.into_inner() * angle;
            let theirs = Rotation3::new(phi);
            let mine = Dcm::new(*theirs.matrix());
            assert_relative_eq!(mine.log().to_vector(), theirs.scaled_axis(), epsilon = EPS);
        }
    }
}

/// 5. Euler 321. nalgebra's `from_euler_angles` takes (roll, pitch, yaw)
///    — the reverse of the sequence's name — so the argument order is
///    the easy place for a silently-passing bug. Pitch is kept away from
///    ±π/2 to avoid gimbal lock.
#[test]
fn euler321_matches_nalgebra() {
    let cases = [
        (0.0, 0.0, 0.0),
        (0.3, -0.4, 0.9),
        (1.2, 0.5, -0.7),
        (-2.1, -0.8, 2.4),
        (2.9, 1.0, -1.5),
    ];
    for &(yaw, pitch, roll) in &cases {
        let mine = Euler321::new(yaw, pitch, roll).to_dcm();
        let theirs = Rotation3::from_euler_angles(roll, pitch, yaw);
        assert_relative_eq!(*mine.matrix(), *theirs.matrix(), epsilon = EPS);
    }
}
