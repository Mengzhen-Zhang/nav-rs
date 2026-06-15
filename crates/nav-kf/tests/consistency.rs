use nalgebra::{Matrix4, Vector4};
use nav_kf::KalmanFilter;
use nav_kf::sim::{ConstVelocity2D, chi2_consistency_band, nees, nis};

/// Matched-filter consistency, the Monte-Carlo way.
///
/// Run an ensemble of independent simulations, average NEES/NIS over runs
/// at each step, and require that almost every post-transient step lands
/// inside the χ² acceptance band. Averaging is over the M independent
/// *runs* (which is what makes `M·η̄ ~ χ²_{M·dof}` hold); the per-step
/// series is then scanned in time, so a transient or time-localized
/// inconsistency can't hide inside a single aggregate number.
#[test]
fn matched_filter_nees_and_nis_are_consistent() {
    let dt = 1.0;
    let var_a = 0.1;
    let var_m = 1.0;

    let n_runs: usize = 50;
    let steps = 500;
    let burn_in = 20; // let the P = I initialization wash out first

    let mut nees_acc = vec![0.0; steps];
    let mut nis_acc = vec![0.0; steps];

    for run in 0..n_runs {
        let mut sim = ConstVelocity2D::new(var_a, var_m, dt, run as u64);
        let (f, q, h, r) = (sim.f, sim.q, sim.h, sim.r);

        let mut kf: KalmanFilter<4, 2> = KalmanFilter {
            x: Vector4::zeros(),
            p: Matrix4::identity(),
        };

        for k in 0..steps {
            let (x_true, z) = sim.update();
            kf.predict(&f, &q);
            let (y, s) = kf.update(&z, &h, &r);
            nees_acc[k] += nees(x_true, kf.x, kf.p);
            nis_acc[k] += nis(y, s);
        }
    }

    let confidence = 0.99;
    let min_in_band = 0.90; // 99% band -> expect ~99% in; 0.90 leaves margin

    for (name, acc, dof) in [("NEES", &nees_acc, 4usize), ("NIS", &nis_acc, 2usize)] {
        let (low, high) = chi2_consistency_band(n_runs, dof, confidence);
        let avg: Vec<f64> = acc.iter().map(|s| s / n_runs as f64).collect();
        let window = &avg[burn_in..];

        let in_band = window.iter().filter(|&&v| v >= low && v <= high).count();
        let frac = in_band as f64 / window.len() as f64;
        let mean = window.iter().sum::<f64>() / window.len() as f64;

        println!(
            "{name}: mean η̄ = {mean:.3} (expect {dof}.0); {:.0}% of steps in {:.0}% band [{low:.2}, {high:.2}]",
            frac * 100.0,
            confidence * 100.0,
        );

        assert!(
            frac >= min_in_band,
            "{name}: only {:.0}% of post-burn-in steps inside the {:.0}% χ² band \
             [{low:.2}, {high:.2}] — filter likely inconsistent",
            frac * 100.0,
            confidence * 100.0,
        );
    }
}
