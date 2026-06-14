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

pub mod dcm;
pub mod euler;
pub mod frames;
mod numerics;
pub mod quat;
pub mod rotvec;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
