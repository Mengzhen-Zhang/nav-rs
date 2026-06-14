//! Rigid-body attitude representations for navigation and estimation.
//!
//! Four interchangeable charts on the rotation group SO(3) —
//! [`quat::UnitQuat`], [`dcm::Dcm`], [`euler::Euler321`], and
//! [`rotvec::RotVec`] — with explicit, tested conversions between them and
//! a single set of conventions enforced across all four.
//!
//! # Conventions — the crate-wide contract
//!
//! Every choice below holds for *all* representations; the per-module
//! rustdoc carries the same row scoped to that type, with the derivation.
//! When a reviewer asks "which convention?", this table is the answer.
//!
//! | Decision             | Choice                                                                                  | Defined in |
//! |----------------------|-----------------------------------------------------------------------------------------|------------|
//! | Quaternion algebra   | **Hamilton** (`ij = k`, right-handed) — *not* JPL                                        | [`quat`]   |
//! | Quaternion storage   | **Scalar-first**: `[w, x, y, z]`                                                         | [`quat`]   |
//! | Operator sense       | **Active**: `transform` rotates the vector; the frame stays fixed                        | all        |
//! | Composition order    | `a.compose(&b)` applies **`b` first**, then `a` — the matrix product `A·B`               | [`quat`], [`dcm`] |
//! | Double cover         | `q` and `−q` are the same rotation; every `==` / `approx` / `approx_eq_rotation` agrees  | [`quat`]   |
//! | Angle units          | **Radians**, right-hand rule about the axis                                              | all        |
//! | Euler sequence       | **321 intrinsic** (yaw ψ → pitch θ → roll φ); note `nalgebra` takes `(roll, pitch, yaw)` | [`euler`]  |
//! | Rotation vector      | `φ = θ·û`; `exp` is Rodrigues, `log` returns the **principal branch** `‖φ‖ ≤ π`          | [`rotvec`] |
//! | Small-angle seam     | every Taylor series branches at the shared `numerics::SMALL_ANGLE_EPS`                   | [`rotvec`] |
//!
//! ## Not yet pinned (forward declarations)
//!
//! Two rows of the eventual contract are deliberately left open until the
//! code that needs them lands — recorded here so the gap is visible rather
//! than silently assumed:
//!
//! - **Frame-direction labelling** (e.g. `C_b^n` vs `C_n^b`): the
//!   [`frames`] module is a stub. Until it ships, this crate speaks only in
//!   active operators on coordinate vectors — no named frames.
//! - **Filter perturbation side** (local vs global error state for the
//!   `δθ` linearization): deferred to Phase 4, where the estimator is
//!   built.
//!
//! # Lie-theoretic map
//!
//! ```text
//!   RotVec  (ℝ³ ≅ so(3))
//!     │  exp                                      │ UnitQuat::from_rotvec
//!     ▼                                           ▼
//!    Dcm  (SO(3))  ◄── covering map to_dcm ──  UnitQuat  (SU(2) = Spin(3))
//!     │
//!     └─ log = to_quat ∘ to_rotvec  (back to so(3), principal branch)
//! ```

pub mod dcm;
pub mod euler;
pub mod frames;
mod numerics;
pub mod quat;
pub mod rotvec;
