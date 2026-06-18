//! imu-forge → ESKF NEES validation — the GPS-denied nav-filter centerpiece.
//!
//! A known coordinated-turn trajectory is flown; an Allan-characterized
//! [`SyntheticImu`](nav_imu::SyntheticImu) (Phase 3) corrupts the inertial
//! stream with the *same* white-noise densities and Gauss–Markov bias the ESKF
//! (Phase 4) assumes; the filter dead-reckons and folds in a position fix every
//! second. Over a 50-seed Monte-Carlo ensemble the 15-state NEES is scored
//! against the filter's own covariance — and lands on χ²₁₅, proving the filter
//! is consistent: it knows exactly how wrong it is.
//!
//! Four panels, written to .dat and drawn by gnuplot as a 2×2 board:
//! 1. NEES vs time inside the χ²₁₅ 95% band — the consistency proof.
//! 2. truth track vs ESKF estimate vs GPS fixes — the filter doing its job.
//! 3. east-position error against its ±3σ envelope — covariance honesty.
//! 4. accel-bias error against its ±3σ envelope — a hidden state, observed.
//!
//! Run:
//! ```text
//! cargo run --release --example eskf_nees
//! (cd crates/nav-eskf && gnuplot eskf_nees.gp)   # -> eskf_nees.png
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nav_eskf::sim::{McConfig, run_nees_mc};

fn main() -> std::io::Result<()> {
    let cfg = McConfig::portfolio();
    let res = run_nees_mc(&cfg);

    let burn_in_s = 5.0;
    println!(
        "imu-forge → ESKF NEES validation: {} s @ {} Hz, fix every {} s, {} runs",
        cfg.n_steps as f64 * cfg.dt,
        1.0 / cfg.dt,
        cfg.fix_every as f64 * cfg.dt,
        res.runs,
    );
    println!(
        "  mean NEES (post {burn_in_s}s) = {:.2}  (expect {}, χ² 95% band [{:.2}, {:.2}])",
        res.mean_after(burn_in_s),
        res.dof,
        res.band_lo,
        res.band_hi,
    );
    println!(
        "  {:.0}% of post-burn-in samples in band",
        res.fraction_in_band(burn_in_s) * 100.0,
    );

    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    {
        let mut w = BufWriter::new(File::create(dir.join("eskf_nees.dat"))?);
        writeln!(w, "# t(s)  mean_nees  band_lo  band_hi")?;
        for (i, &t) in res.t.iter().enumerate() {
            writeln!(
                w,
                "{:.4e} {:.6e} {:.6e} {:.6e}",
                t, res.mean_nees[i], res.band_lo, res.band_hi
            )?;
        }
    }
    {
        let mut w = BufWriter::new(File::create(dir.join("eskf_track.dat"))?);
        writeln!(w, "# truth_x truth_y  est_x est_y  fix_x fix_y   (run 0)")?;
        for i in 0..res.truth_xy.len() {
            let (tx, ty) = res.truth_xy[i];
            let (ex, ey) = res.est_xy[i];
            let (fx, fy) = res.fix_xy[i];
            writeln!(w, "{tx:.6e} {ty:.6e} {ex:.6e} {ey:.6e} {fx:.6e} {fy:.6e}")?;
        }
    }
    {
        let mut w = BufWriter::new(File::create(dir.join("eskf_envelope.dat"))?);
        writeln!(w, "# t(s)  pos_err  pos_sig  ba_err  ba_sig   (run 0)")?;
        for (i, &t) in res.t.iter().enumerate() {
            writeln!(
                w,
                "{:.4e} {:.6e} {:.6e} {:.6e} {:.6e}",
                t, res.pos_err[i], res.pos_sig[i], res.ba_err[i], res.ba_sig[i]
            )?;
        }
    }
    {
        let mut w = BufWriter::new(File::create(dir.join("eskf_nees.gp"))?);
        writeln!(w, "expect = {}.0", res.dof)?;
        writeln!(w, "band_lo = {:.6e}", res.band_lo)?;
        writeln!(w, "band_hi = {:.6e}", res.band_hi)?;
        w.write_all(GP_BODY.as_bytes())?;
    }

    println!("wrote eskf_nees.dat, eskf_track.dat, eskf_envelope.dat, eskf_nees.gp");
    println!("plot: (cd {} && gnuplot eskf_nees.gp)", dir.display());
    Ok(())
}

const GP_BODY: &str = r#"
set terminal pngcairo size 1200,900 enhanced font 'Sans,10'
set output 'eskf_nees.png'
set grid
set multiplot layout 2,2 title 'GPS-denied ESKF — imu-forge stream, position fixes, 15-state NEES validated' font 'Sans,14' margins 0.08,0.97,0.07,0.90 spacing 0.13,0.13

# --- 1. NEES consistency (the proof) ---
set title 'NEES vs {/Symbol c}^2_{15} (50-seed Monte-Carlo mean)'
set xlabel 'time (s)'; set ylabel 'mean NEES (15 dof)'
set yrange [10:20]
set key right top
plot 'eskf_nees.dat' u 1:2 w lp pt 7 ps 0.4 lc rgb '#8e44ad' title 'mean NEES', \
     expect  w l lw 1 dt 2 lc rgb 'black'   title 'expected (15)', \
     band_lo w l lw 1 dt 3 lc rgb '#c0392b' title '95% band', \
     band_hi w l lw 1 dt 3 lc rgb '#c0392b' notitle

# --- 2. trajectory: truth vs estimate vs fixes ---
set title 'coordinated turn — truth, ESKF estimate, GPS fixes (one run)'
set xlabel 'east (m)'; set ylabel 'north (m)'
set size ratio -1
unset yrange
set key right bottom
plot 'eskf_track.dat' u 1:2 w l lw 2 lc rgb '#2471a3' title 'truth', \
     'eskf_track.dat' u 3:4 w l lw 1.5 dt 2 lc rgb '#e67e22' title 'ESKF estimate', \
     'eskf_track.dat' u 5:6 w p pt 6 ps 0.5 lc rgb '#27ae60' title 'GPS fixes'
set size noratio

# --- 3. position error vs ±3-sigma envelope ---
set title 'east-position error vs filter ±3{/Symbol s} (one run)'
set xlabel 'time (s)'; set ylabel 'east error (m)'
set key right top
plot 'eskf_envelope.dat' u 1:($3*3)  w l lw 1 dt 2 lc rgb '#c0392b' title '+3{/Symbol s}', \
     'eskf_envelope.dat' u 1:(-$3*3) w l lw 1 dt 2 lc rgb '#c0392b' notitle, \
     'eskf_envelope.dat' u 1:2 w lp pt 7 ps 0.4 lc rgb '#2c3e50' title 'error'

# --- 4. accel-bias error vs ±3-sigma envelope ---
set title 'accel-bias_x error vs filter ±3{/Symbol s} (one run)'
set xlabel 'time (s)'; set ylabel 'accel-bias error (m/s^2)'
set key right top
plot 'eskf_envelope.dat' u 1:($5*3)  w l lw 1 dt 2 lc rgb '#c0392b' title '+3{/Symbol s}', \
     'eskf_envelope.dat' u 1:(-$5*3) w l lw 1 dt 2 lc rgb '#c0392b' notitle, \
     'eskf_envelope.dat' u 1:4 w lp pt 7 ps 0.4 lc rgb '#16a085' title 'error'

unset multiplot
"#;
