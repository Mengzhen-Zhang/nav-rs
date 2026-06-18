//! Pure inertial navigation, no corrections — watch it diverge.
//!
//! A stationary, level IMU (the cleanest case) is corrupted through
//! [`SyntheticImu`] and integrated forward with the strapdown mechanization
//! in [`nav_imu::ins`]. With no external reference the solution walks away
//! without bound — this is the GPS-denied problem, made visible.
//!
//! Three deliverables, written to .dat and drawn by gnuplot as a 2×2 board:
//! 1. walk-off track — the dead-reckoned position leaving the origin.
//! 2. drift vs time — RMS position error (log-log) with t²/t³ slope lines;
//!    whichever the data parallels names the dominant error source.
//! 3. NEES vs time — error against the propagated INS covariance, inside the
//!    χ² band, showing the covariance is consistent.
//!
//! NIS is deliberately absent: pure dead-reckoning has no measurements, hence
//! no innovations. That absence *is* why it diverges and why you need GPS.
//!
//! Run:
//! ```text
//! cargo run --release --example ins_deadreckon
//! (cd crates/nav-imu && gnuplot ins_deadreckon.gp)   # -> ins_deadreckon.png
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::{Matrix3, SMatrix, Vector3};
use nav_attitude::dcm::Dcm;
use nav_imu::ins::{
    self, NavState, error_process_noise, error_transition, nav_error, true_specific_force,
};
use nav_imu::{ImuErrorParams, SyntheticImu};
use statrs::distribution::{ChiSquared, ContinuousCDF};

const DT: f64 = 0.01; // 100 Hz
const T: f64 = 300.0; // 5 minutes
const N_G: f64 = 5e-4; // gyro white-noise density (ARW), rad/√s
const N_A: f64 = 0.02; // accel white-noise density (VRW), m/s/√s

// Full error model — biases + noise, the realistic divergence driver.
fn gyro_full() -> ImuErrorParams {
    ImuErrorParams::new(0.0, 2e-4, 1e-4, 100.0, N_G)
}
fn accel_full() -> ImuErrorParams {
    ImuErrorParams::new(0.0, 0.02, 0.01, 100.0, N_A)
}
// White-noise-only — what the propagated covariance models exactly.
fn gyro_wn() -> ImuErrorParams {
    ImuErrorParams::new(0.0, 0.0, 0.0, 100.0, N_G)
}
fn accel_wn() -> ImuErrorParams {
    ImuErrorParams::new(0.0, 0.0, 0.0, 100.0, N_A)
}

/// Stationary, level truth, and the (constant) measurements a perfect IMU
/// would report: zero rate, specific force +g (up).
fn stationary_truth() -> (NavState, Vector3<f64>, Vector3<f64>) {
    let attitude = Dcm::new(Matrix3::identity());
    let state = NavState {
        attitude,
        vel: Vector3::zeros(),
        pos: Vector3::zeros(),
    };
    let gyro_true = Vector3::zeros();
    let accel_true = true_specific_force(&attitude, Vector3::zeros());
    (state, gyro_true, accel_true)
}

/// Log-spaced step indices in `[first, n_steps]`.
fn sample_steps(n_steps: usize, n_pts: usize, first: usize) -> Vec<usize> {
    let mut v = Vec::new();
    for i in 0..n_pts {
        let frac = i as f64 / (n_pts - 1) as f64;
        let s = (first as f64 * (n_steps as f64 / first as f64).powf(frac)).round() as usize;
        let s = s.clamp(first, n_steps);
        if v.last() != Some(&s) {
            v.push(s);
        }
    }
    v
}

/// Two-sided χ² acceptance band for a Monte-Carlo-averaged NEES (per the
/// `M·η̄ ~ χ²_{M·dof}` identity), on the η̄ scale.
fn chi2_band(runs: usize, dof: usize, conf: f64) -> (f64, f64) {
    let m = runs as f64;
    let chi = ChiSquared::new((runs * dof) as f64).unwrap();
    let tail = (1.0 - conf) / 2.0;
    (chi.inverse_cdf(tail) / m, chi.inverse_cdf(1.0 - tail) / m)
}

fn main() -> std::io::Result<()> {
    let n_steps = (T / DT) as usize;
    let samples = sample_steps(n_steps, 140, (1.0 / DT) as usize); // from t = 1 s

    // ---------- 1 & 2. Drift (full error model), Monte-Carlo ----------
    let m_drift = 40;
    let mut sumsq = vec![0.0f64; samples.len()]; // Σ |δp|² over runs
    let mut track: Vec<(f64, f64)> = Vec::new(); // run 0's horizontal walk-off
    for run in 0..m_drift {
        let (truth, gyro_true, accel_true) = stationary_truth();
        let mut imu = SyntheticImu::new(gyro_full(), accel_full(), DT, run as u64);
        let mut st = truth.clone();
        let mut si = 0;
        for k in 1..=n_steps {
            let meas = imu.sample(gyro_true, accel_true);
            st = st.propagate(meas.gyro, meas.accel, DT);
            if si < samples.len() && samples[si] == k {
                sumsq[si] += st.pos.norm_squared(); // truth.pos = 0
                if run == 0 {
                    track.push((st.pos.x, st.pos.y));
                }
                si += 1;
            }
        }
    }
    let rms: Vec<f64> = sumsq.iter().map(|s| (s / m_drift as f64).sqrt()).collect();

    // ---------- 3. NEES (white-noise model), Monte-Carlo ----------
    // Propagate the 9-state INS error covariance once (deterministic), cache
    // its inverse at each sample time. Stationary => specific force f^n = (0,0,g).
    let f_nav = Vector3::new(0.0, 0.0, ins::G);
    let f_mat = error_transition(f_nav, DT);
    let q = error_process_noise(N_G, N_A, DT);
    let mut p = SMatrix::<f64, 9, 9>::zeros();
    let mut pinvs = Vec::with_capacity(samples.len());
    {
        let mut si = 0;
        for k in 1..=n_steps {
            p = f_mat * p * f_mat.transpose() + q;
            if si < samples.len() && samples[si] == k {
                pinvs.push(p.try_inverse());
                si += 1;
            }
        }
    }

    let m_nees = 60;
    let mut nees_sum = vec![0.0f64; samples.len()];
    for run in 0..m_nees {
        let (truth, gyro_true, accel_true) = stationary_truth();
        let mut imu = SyntheticImu::new(gyro_wn(), accel_wn(), DT, 1000 + run as u64);
        let mut st = truth.clone();
        let mut si = 0;
        for k in 1..=n_steps {
            let meas = imu.sample(gyro_true, accel_true);
            st = st.propagate(meas.gyro, meas.accel, DT);
            if si < samples.len() && samples[si] == k {
                if let Some(pinv) = pinvs[si] {
                    let e = nav_error(&st, &truth);
                    nees_sum[si] += e.dot(&(pinv * e));
                }
                si += 1;
            }
        }
    }
    let nees: Vec<f64> = nees_sum.iter().map(|s| s / m_nees as f64).collect();
    let (band_lo, band_hi) = chi2_band(m_nees, 9, 0.95);

    println!(
        "INS dead-reckoning, stationary truth, {T} s @ {} Hz",
        1.0 / DT
    );
    println!(
        "  full-error drift: RMS |δp| at {T} s = {:.0} m  ({} runs)",
        rms.last().unwrap(),
        m_drift
    );
    println!(
        "  white-noise NEES at {T} s = {:.2}  (χ² 95% band [{band_lo:.2}, {band_hi:.2}], expect 9)",
        nees.last().unwrap()
    );

    // ---------- write data ----------
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    {
        let mut w = BufWriter::new(File::create(dir.join("ins_drift.dat"))?);
        writeln!(w, "# t(s)  rms_pos_err(m)")?;
        for (i, &s) in samples.iter().enumerate() {
            writeln!(w, "{:.4e} {:.6e}", s as f64 * DT, rms[i])?;
        }
    }
    {
        let mut w = BufWriter::new(File::create(dir.join("ins_track.dat"))?);
        writeln!(w, "# x(m)  y(m)  (dead-reckoned, run 0)")?;
        for (x, y) in &track {
            writeln!(w, "{x:.6e} {y:.6e}")?;
        }
    }
    {
        let mut w = BufWriter::new(File::create(dir.join("ins_nees.dat"))?);
        writeln!(w, "# t(s)  nees  band_lo  band_hi")?;
        for (i, &s) in samples.iter().enumerate() {
            writeln!(
                w,
                "{:.4e} {:.6e} {band_lo:.6e} {band_hi:.6e}",
                s as f64 * DT,
                nees[i]
            )?;
        }
    }

    // reference-slope anchors for the drift log-log (pass through the tail)
    let t_last = *samples.last().unwrap() as f64 * DT;
    let c2 = rms.last().unwrap() / t_last.powi(2);
    let c3 = rms.last().unwrap() / t_last.powi(3);
    {
        let mut w = BufWriter::new(File::create(dir.join("ins_deadreckon.gp"))?);
        writeln!(w, "c2 = {c2:.6e}")?;
        writeln!(w, "c3 = {c3:.6e}")?;
        writeln!(w, "expect = 9.0")?;
        writeln!(w, "band_lo = {band_lo:.6e}")?;
        writeln!(w, "band_hi = {band_hi:.6e}")?;
        w.write_all(GP_BODY.as_bytes())?;
    }

    println!("wrote ins_drift.dat, ins_track.dat, ins_nees.dat, ins_deadreckon.gp");
    println!("plot: (cd {} && gnuplot ins_deadreckon.gp)", dir.display());
    Ok(())
}

const GP_BODY: &str = r#"
set terminal pngcairo size 1150,860 enhanced font 'Sans,10'
set output 'ins_deadreckon.png'
set grid
set multiplot layout 2,2 title 'Pure inertial navigation diverges — stationary IMU, 5 min' font 'Sans,14' margins 0.09,0.97,0.08,0.90 spacing 0.13,0.12

# --- 1. walk-off track ---
set title 'dead-reckoned horizontal track (one run)'
set xlabel 'east (m)'; set ylabel 'north (m)'
set key off
plot 'ins_track.dat' u 1:2 w l lw 2 lc rgb '#2471a3', \
     '<echo 0 0' u 1:2 w p pt 7 ps 1.5 lc rgb '#27ae60'

# --- 2. position error vs time (linear) ---
set title 'position error vs time'
set xlabel 'time (s)'; set ylabel 'RMS |{/Symbol d}p| (m)'
set xrange [0:300]
unset logscale
set key left top
plot 'ins_drift.dat' u 1:2 w lp pt 7 ps 0.4 lc rgb '#c0392b' title 'RMS over runs'

# --- 3. drift vs time (log-log, slope diagnosis) ---
set title 'drift growth law'
set xlabel 'time (s)'; set ylabel 'RMS |{/Symbol d}p| (m)'
set xrange [1:300]
set logscale xy
set key left top
plot 'ins_drift.dat' u 1:2 w lp pt 7 ps 0.4 lc rgb '#c0392b' title 'RMS |{/Symbol d}p|', \
     c3*x**3 w l dt 2 lw 2 lc rgb 'black'    title 't^3  (gyro bias {/Symbol \264} gravity)', \
     c2*x**2 w l dt 3 lw 2 lc rgb '#888888'  title 't^2  (accel bias)'

# --- 4. NEES consistency ---
set title 'NEES vs propagated INS covariance (white-noise only)'
set xlabel 'time (s)'; set ylabel 'NEES (9 dof)'
unset logscale
set xrange [0:300]
set yrange [4:16]
set key left top
plot 'ins_nees.dat' u 1:2 w lp pt 7 ps 0.4 lc rgb '#8e44ad' title 'mean NEES', \
     expect  w l lw 1 dt 2 lc rgb 'black'    title 'expected (9)', \
     band_lo w l lw 1 dt 3 lc rgb '#c0392b'  title '95% band', \
     band_hi w l lw 1 dt 3 lc rgb '#c0392b'  notitle

unset multiplot
"#;
