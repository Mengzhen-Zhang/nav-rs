use nalgebra::Vector3;
use nav_imu::{ImuErrorParams, ImuSample, SyntheticImu};

fn zero_params() -> ImuErrorParams {
    // scale, turn_on_bias_sigma, bias_instab_sigma, bias_corr_time, white_noise_density
    ImuErrorParams::new(0.0, 0.0, 0.0, 0.0, 0.0)
}

/// With every error parameter zero, the sensor is ideal: `sample` must
/// return the truth bit-for-bit, on every call (scale = 0, b0 = 0, the
/// drift volatility `√(σ_b²(1−β²))` = 0, and the white-noise term
/// `0/√dt · ξ` = 0).
#[test]
fn returns_truth_exactly_when_all_params_zero() {
    let mut imu = SyntheticImu::new(zero_params(), zero_params(), 0.01, 42);

    let gyro_true = Vector3::new(0.1, -0.2, 0.3);
    let accel_true = Vector3::new(9.81, 0.0, -1.0);

    for _ in 0..100 {
        let ImuSample { gyro, accel } = imu.sample(gyro_true, accel_true);
        assert_eq!(gyro, gyro_true, "gyro perturbed despite zero params");
        assert_eq!(accel, accel_true, "accel perturbed despite zero params");
    }
}

/// With only white noise enabled, each measurement component is
/// `N(true, σ_n²/dt)`. Over many draws the sample mean must converge to
/// truth and the sample variance to `σ_n²/dt`. Tolerances are 5× the
/// theoretical standard errors (SEM for the mean, `σ²√(2/M)` for the
/// variance), so the check is both meaningful and effectively non-flaky.
#[test]
fn white_noise_only_mean_and_variance_match_theory() {
    let sigma = 0.05; // white_noise_density, σ_n
    let dt = 0.01;

    // gyro: white noise only. accel: ideal (unused here).
    let gyro = ImuErrorParams::new(0.0, 0.0, 0.0, 0.0, sigma);
    let mut imu = SyntheticImu::new(gyro, zero_params(), dt, 7);

    let gyro_true = Vector3::new(0.10, -0.20, 0.30);
    let accel_true = Vector3::zeros();

    let m = 400_000usize;
    let mut sum = Vector3::zeros();
    let mut sum_sq = Vector3::zeros();
    for _ in 0..m {
        let gyro = imu.sample(gyro_true, accel_true).gyro;
        sum += gyro;
        sum_sq += gyro.component_mul(&gyro);
    }
    let mean = sum / m as f64;

    let expected_std = sigma / dt.sqrt(); // 0.5
    let expected_var = expected_std * expected_std; // 0.25
    let sem = expected_std / (m as f64).sqrt(); // SE of the mean
    let var_se = expected_var * (2.0 / m as f64).sqrt(); // SE of the sample variance

    // unbiased sample variance per component: (Σx² − (Σx)²/M) / (M−1)
    let var = (sum_sq - sum.component_mul(&sum) / m as f64) / (m as f64 - 1.0);

    for i in 0..3 {
        let mean_err = (mean[i] - gyro_true[i]).abs();
        assert!(
            mean_err < 5.0 * sem,
            "component {i}: |mean − true| = {mean_err:.3e} exceeds 5·SEM = {:.3e}",
            5.0 * sem
        );

        let var_err = (var[i] - expected_var).abs();
        assert!(
            var_err < 5.0 * var_se,
            "component {i}: sample var {:.5} vs expected {expected_var:.5} \
             (err {var_err:.3e} > 5·SE {:.3e})",
            var[i],
            5.0 * var_se
        );
    }
}
