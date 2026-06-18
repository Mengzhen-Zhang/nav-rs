//! imu-forge → ESKF consistency, the Monte-Carlo way.
//!
//! Fly a known trajectory, corrupt the inertial stream with an
//! Allan-characterized IMU, fold in periodic position fixes, and require the
//! seed-averaged 15-state NEES to sit inside the χ²₁₅ acceptance band across
//! the post-burn-in window. Averaging is over independent *runs* (the identity
//! that makes `M·η̄ ~ χ²_{15M}` hold), and the per-step series is scanned in
//! time so a transient or time-localized inconsistency can't hide inside a
//! single aggregate number. This is the proof behind the portfolio plot.

use nav_eskf::sim::{McConfig, run_nees_mc};

#[test]
fn imu_forge_to_eskf_nees_is_consistent() {
    let cfg = McConfig::test();
    let res = run_nees_mc(&cfg);

    let burn_in_s = 5.0; // let the P₀ draw and the first few fixes settle
    let frac = res.fraction_in_band(burn_in_s);
    let mean = res.mean_after(burn_in_s);

    println!(
        "ESKF NEES: mean η̄ = {mean:.3} (expect 15.0); {:.0}% of post-burn-in \
         samples in 95% band [{:.2}, {:.2}]  ({} runs)",
        frac * 100.0,
        res.band_lo,
        res.band_hi,
        res.runs,
    );

    // E[η̄] = 15 within a generous tolerance — a gross bias-block, attitude
    // convention, or Q-scaling error pushes this far off 15.
    assert!(
        (mean - 15.0).abs() < 3.0,
        "mean NEES {mean:.3} is far from the 15-DOF expectation — filter inconsistent"
    );

    // The 95% band should hold ~95% of post-burn-in samples; 0.85 leaves margin
    // for Monte-Carlo scatter at this ensemble size.
    assert!(
        frac >= 0.85,
        "only {:.0}% of post-burn-in samples inside the 95% χ² band [{:.2}, {:.2}] \
         — filter likely inconsistent",
        frac * 100.0,
        res.band_lo,
        res.band_hi,
    );
}
