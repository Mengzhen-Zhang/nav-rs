use nalgebra::{Matrix4, Vector4};
use nav_kf::KalmanFilter;
use nav_kf::sim::{ConstVelocity2D, chi2_consistency_band, nees, nis};

/// Single-run constant-velocity tracking, streamed to rerun: truth,
/// measurement, estimate, 1σ position ellipse, velocity arrow, and the
/// NEES/NIS traces with their single-run χ² acceptance bands.
///
/// This is a *visualization* — one realization, so the NEES/NIS traces are
/// noisy and individual points routinely poke outside the band. The
/// statistical consistency verdict lives in `tests/consistency.rs`, which
/// averages over an ensemble.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rec = rerun::RecordingStreamBuilder::new("cv_tracking").spawn()?;

    let dt = 1.0;
    let var_a = 0.1;
    let var_m = 1.0;
    let seed = 0;
    let mut sim = ConstVelocity2D::new(var_a, var_m, dt, seed);

    let (f, q, h, r) = (sim.f, sim.q, sim.h, sim.r);

    let mut kf: KalmanFilter<4, 2> = KalmanFilter {
        x: Vector4::zeros(),
        p: Matrix4::identity(),
    };

    // Single-run (M = 1) acceptance bands: NEES ~ χ²₄, NIS ~ χ²₂.
    let (nees_lo, nees_hi) = chi2_consistency_band(1, 4, 0.95);
    let (nis_lo, nis_hi) = chi2_consistency_band(1, 2, 0.95);

    let steps = 100;
    for k in 0..steps {
        rec.set_time_sequence("step", k as i64);

        let (x_true, z) = sim.update();
        kf.predict(&f, &q);
        let (y, s) = kf.update(&z, &h, &r);

        let nees_k = nees(x_true, kf.x, kf.p);
        let nis_k = nis(y, s);

        // the noisy measurement: a single grey 2-D point
        rec.log(
            "world/measurement",
            &rerun::Points2D::new([(z[0] as f32, z[1] as f32)])
                .with_colors([rerun::Color::from_rgb(150, 150, 150)])
                .with_radii([0.15]),
        )?;

        // the true position: a green point
        rec.log(
            "world/truth",
            &rerun::Points2D::new([(x_true[0] as f32, x_true[1] as f32)])
                .with_colors([rerun::Color::from_rgb(0, 200, 0)])
                .with_radii([0.2]),
        )?;

        // the estimate: a blue point
        rec.log(
            "world/estimate",
            &rerun::Points2D::new([(kf.x[0] as f32, kf.x[1] as f32)])
                .with_colors([rerun::Color::from_rgb(0, 100, 255)])
                .with_radii([0.2]),
        )?;

        // the 1σ covariance ellipse on position
        let p_pos = kf.p.fixed_view::<2, 2>(0, 0).into_owned();
        let eig = p_pos.symmetric_eigen();
        let half_x = (eig.eigenvalues[0].max(0.0)).sqrt() as f32;
        let half_y = (eig.eigenvalues[1].max(0.0)).sqrt() as f32;
        rec.log(
            "world/estimate/uncertainty",
            &rerun::Ellipses2D::from_centers_and_half_sizes(
                [(kf.x[0] as f32, kf.x[1] as f32)],
                [(half_x, half_y)],
            )
            .with_colors([rerun::Color::from_rgb(0, 100, 255)]),
        )?;

        // the velocity estimate as an arrow from the estimate point
        rec.log(
            "world/estimate/velocity",
            &rerun::Arrows2D::from_vectors([(kf.x[2] as f32, kf.x[3] as f32)])
                .with_origins([(kf.x[0] as f32, kf.x[1] as f32)])
                .with_colors([rerun::Color::from_rgb(255, 150, 0)]),
        )?;

        // NEES / NIS traces with their 95% single-run acceptance bands
        rec.log("plots/nees", &rerun::Scalars::new([nees_k]))?;
        rec.log("plots/nees_band", &rerun::Scalars::new([nees_lo, nees_hi]))?;
        rec.log("plots/nis", &rerun::Scalars::new([nis_k]))?;
        rec.log("plots/nis_band", &rerun::Scalars::new([nis_lo, nis_hi]))?;
    }

    Ok(())
}
