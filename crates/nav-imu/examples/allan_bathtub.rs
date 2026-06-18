//! Allan-deviation "bathtub" round-trip for [`SyntheticImu`].
//!
//! Drive the sensor stationary (true rate = 0) with white noise + a
//! Gauss-Markov bias, compute the overlapping Allan deviation with the
//! crate's [`nav_imu::allan`] module, and read the two spec coefficients
//! straight off the curve:
//!
//!   * **N** — angle random walk, the −1/2 arm: `σ(τ) = N/√τ`, so
//!     `read_arw` reports `σ(1 s)`.
//!   * **B** — bias instability, the floor: `read_bias_instability` uses the
//!     IEEE convention `σ_min = 0.664·B`.
//!
//! Those recovered N and B are written *as the reference-line coefficients*
//! into a gnuplot script: if the round trip worked the `N/√τ` line lies
//! along the left arm and the `0.664·B` line sits at the bathtub floor.
//!
//! Run:
//! ```text
//! cargo run --release --example allan_bathtub
//! (cd crates/nav-imu && gnuplot allan_bathtub.gp)   # writes allan_bathtub.png
//! ```

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use nalgebra::Vector3;
use nav_imu::{ImuErrorParams, SyntheticImu, allan};

fn main() -> std::io::Result<()> {
    // ---- known truth (what we hope to read back off the curve) ----
    let dt = 0.01; // 100 Hz
    let n_density = 0.003; // white-noise density N (rate·√s) -> ARW
    let sigma_b = 0.005; // bias-instability stationary σ_b (rate)
    let tau_c = 50.0; // bias correlation time (s)
    let n_samples = 1_000_000usize; // ~2.8 h at 100 Hz

    // gyro: white noise + Gauss-Markov bias. Turn-on bias and scale stay zero
    // — they don't affect Allan variance (the double difference kills any
    // constant) and the true rate is zero anyway. accel: ideal, unused.
    let gyro = ImuErrorParams::new(0.0, 0.0, sigma_b, tau_c, n_density);
    let ideal = ImuErrorParams::new(0.0, 0.0, 0.0, tau_c, 0.0);
    let mut imu = SyntheticImu::new(gyro, ideal, dt, 0xA11A);

    // ---- stationary record of the gyro x-channel rate ----
    let zero = Vector3::zeros();
    let mut data = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let gyro_meas = imu.sample(zero, zero).gyro;
        data.push(gyro_meas[0]);
    }

    // ---- Allan deviation + coefficient recovery (crate API) ----
    let curve = allan::allan_deviation(&data, dt);
    let n_est = allan::read_arw(&curve);
    let b_est = allan::read_bias_instability(&curve);

    println!("input     N = {n_density:.5}   σ_b = {sigma_b:.5}   τ_c = {tau_c} s");
    println!("recovered N = {n_est:.5}   B = {b_est:.5}");
    println!(
        "N round-trip error: {:.1}%",
        100.0 * (n_est - n_density).abs() / n_density
    );

    // ---- write data + a self-contained gnuplot script next to the crate ----
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dat_path = out_dir.join("allan_bathtub.dat");
    let gp_path = out_dir.join("allan_bathtub.gp");

    {
        let mut w = BufWriter::new(File::create(&dat_path)?);
        writeln!(w, "# tau(s)   allan_deviation")?;
        for p in &curve {
            writeln!(w, "{:.6e} {:.6e}", p.tau, p.deviation)?;
        }
    }

    {
        // Inject only the numbers via format!, then emit the body verbatim so
        // gnuplot's own {/Symbol ...} braces are left untouched.
        let mut w = BufWriter::new(File::create(&gp_path)?);
        writeln!(w, "N = {n_est:.6e}")?;
        writeln!(w, "B = {b_est:.6e}")?;
        w.write_all(GP_BODY.as_bytes())?;
    }

    println!("wrote {} and {}", dat_path.display(), gp_path.display());
    println!(
        "plot: (cd {} && gnuplot allan_bathtub.gp)",
        out_dir.display()
    );
    Ok(())
}

/// Static gnuplot body. References the `N` and `B` variables written above;
/// every `{...}` here is gnuplot enhanced-text markup, not Rust formatting.
const GP_BODY: &str = r#"
set terminal pngcairo size 900,650 enhanced font 'Sans,11'
set output 'allan_bathtub.png'
set logscale xy
set grid xtics ytics mxtics mytics lc rgb '#cccccc'
set xlabel 'averaging time {/Symbol t} (s)'
set ylabel 'Allan deviation {/Symbol s}({/Symbol t})'
set title 'SyntheticImu gyro — Allan-deviation bathtub (round-trip)'
set key left bottom
set label 1 sprintf("N = %.3e  (slope -1/2)", N) at graph 0.04,0.30 tc rgb '#c0392b'
set label 2 sprintf("B = %.3e  (bias instability)", B) at graph 0.46,0.10 tc rgb '#2471a3'
plot \
  'allan_bathtub.dat' u 1:2 w lp pt 7 ps 0.5 lc rgb 'black' title 'Allan deviation', \
  N/sqrt(x) w l lw 2 dt 2 lc rgb '#c0392b' title 'N {/Symbol t}^{-1/2}', \
  0.664*B w l lw 2 dt 2 lc rgb '#2471a3' title '0.664 B'
"#;
