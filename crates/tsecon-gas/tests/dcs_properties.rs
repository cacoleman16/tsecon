//! Property tests for the DCS robust local level — the Monte-Carlo grade
//! the Student-t and Laplace filters are honestly held to, since no
//! runnable third-party reference implements them (DCS reference code is
//! R/Matlab; the same grade the roadmap assigns SETAR-class estimators).
//!
//! Four claims, mirroring the lab study (`lab/REPORT.md`, experiment 3)
//! that graduated this estimator:
//!
//! 1. **Parameter recovery** on data simulated from the DCS-t model
//!    itself (seeded Philox, 200 deterministic parallel replications).
//! 2. **Robustness**: under 5%/10% additive 8-sigma outliers the
//!    Student-t (and Laplace) filtered level beats the Gaussian filter's
//!    RMSE against the *clean* truth by a stated margin.
//! 3. **No clean-data tax + nesting**: on clean Gaussian data the DCS-t
//!    fit collapses onto the Gaussian filter (`nu` runs toward its
//!    Gaussian boundary, honestly flagged as non-converged), and the
//!    fitted Gaussian `kappa` matches the steady-state gain implied by
//!    the statsmodels UC-MLE up to the finite-sample transient.
//! 4. **Observed-information SEs** cover the truth at conventional
//!    multiples on a well-posed fit.

use serde_json::Value;
use tsecon_bootstrap::par_replicate;
use tsecon_gas::{DcsDensity, DcsModel};
use tsecon_rng::Stream;
use tsecon_stats::{ContinuousDist, StdNormal, StudentT};

fn load_fixture() -> Value {
    let path = format!(
        "{}/../../fixtures/tsecon-dcs.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read fixture");
    serde_json::from_str(&text).expect("parse fixture")
}

fn as_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Uniform draw strictly inside (0, 1) for inverse-CDF sampling
/// (`uniform_f64` can return exactly 0 with probability 2^-53).
fn uniform_pos(stream: &mut Stream) -> f64 {
    loop {
        let u = stream.uniform_f64();
        if u > 0.0 {
            return u;
        }
    }
}

/// Simulate the DCS-t local level at its own true parameters:
/// `y_t = mu_t + scale * t_nu`, `mu_{t+1} = mu_t + kappa u_t`.
fn simulate_dcs_t(n: usize, kappa: f64, scale: f64, nu: f64, stream: &mut Stream) -> Vec<f64> {
    let t_dist = StudentT::new(nu).expect("nu > 0");
    let mut y = vec![0.0; n];
    let mut m = 0.0;
    for yt in y.iter_mut() {
        let e = scale * t_dist.ppf(uniform_pos(stream)).expect("ppf");
        *yt = m + e;
        let z2 = e * e / (scale * scale);
        let u = (nu + 1.0) * e / (nu + z2);
        m += kappa * u;
    }
    y
}

fn rmse_vs(level: &[f64], truth: &[f64]) -> f64 {
    let n = level.len() as f64;
    (level
        .iter()
        .zip(truth)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Claim 1 — Monte-Carlo parameter recovery on simulated DCS-t data.
///
/// Design: T = 400, true (kappa, scale, nu) = (0.15, 1.0, 5.0), 200
/// seeded replications (Philox substreams; `par_replicate` is
/// thread-count invariant, so the numbers below are exactly reproducible).
///
/// Measured over the 200 replications (this exact seed):
///
/// ```text
/// kappa: bias -0.0026, RMSE 0.0331
/// scale: bias +0.0011, RMSE 0.0578
/// nu   : median 5.17, within (3.5, 8) in 183/200 reps
///        (the dof of a T = 400 t-sample is genuinely noisy; the
///        occasional Gaussian-looking path pushes nu_hat far upward,
///        which is why the median, not the mean, is the statistic to
///        grade)
/// convergence: 200/200 fits certified
/// ```
///
/// The asserted bounds below sit with a margin around those measured
/// values so seed-stable regressions are caught without flaking.
#[test]
fn mc_recovery_on_simulated_dcs_t_data() {
    const REPS: usize = 200;
    const T: usize = 400;
    const KAPPA: f64 = 0.15;
    const SCALE: f64 = 1.0;
    const NU: f64 = 5.0;

    let fits: Vec<(f64, f64, f64, bool)> = par_replicate(20260818, REPS, |_, stream| {
        let y = simulate_dcs_t(T, KAPPA, SCALE, NU, stream);
        let res = DcsModel::new(&y, DcsDensity::StudentT)
            .expect("model")
            .fit()
            .expect("fit");
        (
            res.params.kappa,
            res.params.scale,
            res.params.nu,
            res.converged,
        )
    })
    .expect("replicate");

    let n = REPS as f64;
    let bias_k = fits.iter().map(|x| x.0).sum::<f64>() / n - KAPPA;
    let rmse_k = (fits
        .iter()
        .map(|x| (x.0 - KAPPA) * (x.0 - KAPPA))
        .sum::<f64>()
        / n)
        .sqrt();
    let bias_s = fits.iter().map(|x| x.1).sum::<f64>() / n - SCALE;
    let rmse_s = (fits
        .iter()
        .map(|x| (x.1 - SCALE) * (x.1 - SCALE))
        .sum::<f64>()
        / n)
        .sqrt();
    let mut nus: Vec<f64> = fits.iter().map(|x| x.2).collect();
    nus.sort_unstable_by(f64::total_cmp);
    let med_nu = 0.5 * (nus[REPS / 2 - 1] + nus[REPS / 2]);
    let nu_in_band = nus.iter().filter(|&&v| (3.5..8.0).contains(&v)).count();
    let n_conv = fits.iter().filter(|x| x.3).count();

    eprintln!(
        "[dcs mc-recovery] kappa bias {bias_k:+.4} rmse {rmse_k:.4} | \
         scale bias {bias_s:+.4} rmse {rmse_s:.4} | nu median {med_nu:.2} \
         in-band {nu_in_band}/{REPS} | converged {n_conv}/{REPS}"
    );

    assert!(bias_k.abs() < 0.02, "kappa bias {bias_k}");
    assert!(rmse_k < 0.08, "kappa RMSE {rmse_k}");
    assert!(bias_s.abs() < 0.02, "scale bias {bias_s}");
    assert!(rmse_s < 0.08, "scale RMSE {rmse_s}");
    assert!(
        (4.0..6.5).contains(&med_nu),
        "median nu {med_nu} (true {NU})"
    );
    assert!(
        nu_in_band >= (0.70 * n) as usize,
        "nu in (3.5, 8) only {nu_in_band}/{REPS}"
    );
    assert!(
        n_conv >= (0.90 * n) as usize,
        "only {n_conv}/{REPS} fits certified convergence"
    );
}

/// Simulate the lab's exp03 design: a Gaussian local level
/// (`sigma_eta`, `sigma_eps`) whose *observations* are contaminated with
/// additive `size`-sigma outliers at rate `frac` (Bernoulli positions,
/// random sign); the level path stays clean and is what RMSE is measured
/// against.
fn simulate_contaminated_ll(
    n: usize,
    sigma_eta: f64,
    sigma_eps: f64,
    frac: f64,
    size: f64,
    stream: &mut Stream,
) -> (Vec<f64>, Vec<f64>) {
    let mut y = vec![0.0; n];
    let mut mu = vec![0.0; n];
    let mut m = 0.0_f64;
    for t in 0..n {
        m += sigma_eta * StdNormal.ppf(uniform_pos(stream)).expect("ppf");
        mu[t] = m;
        y[t] = m + sigma_eps * StdNormal.ppf(uniform_pos(stream)).expect("ppf");
        if frac > 0.0 && stream.uniform_f64() < frac {
            let sign = if stream.uniform_f64() < 0.5 {
                -1.0
            } else {
                1.0
            };
            y[t] += sign * size * sigma_eps;
        }
    }
    (y, mu)
}

/// Claim 2 — robustness under additive outliers, replicated Monte-Carlo
/// on the lab's exp03 design (sigma_eta = 0.1, sigma_eps = 1.0, T = 500,
/// 8-sigma additive outliers at 0/5/10%, RMSE of the one-step-predicted
/// level against the CLEAN truth, 20 seeded reps per contamination
/// level).
///
/// Measured mean RMSE ratios vs the fitted DCS-Gaussian control (this
/// exact seed set — deterministic at any thread count):
///
/// ```text
/// frac   t/gauss   laplace/gauss   mean kappa: gauss    t
///  0%     0.999        1.103                   0.086   0.087
///  5%     0.774        0.811                   0.048   0.140
/// 10%     0.688        0.741                   0.034   0.122
/// ```
///
/// — reproducing the lab's exp03 verdict on the shipped implementation:
/// no clean-data tax, -23%/-31% for DCS-t under 5/10% contamination
/// (lab: -22%/-31% against the Kalman-pipeline control), the Laplace
/// filter robust but dominated by DCS-t, and the *mechanism* visible in
/// the fitted gains — the contaminated Gaussian MLE collapses its gain
/// (0.086 -> 0.034, going blind to real level shifts) while DCS-t raises
/// it (0.087 -> 0.122) because its bounded score already discounts the
/// outliers. Bounds asserted with margin below.
#[test]
fn robust_densities_beat_gaussian_under_contamination() {
    const REPS: usize = 20;
    const T: usize = 500;

    // (frac, upper bound on mean t/gauss, upper bound on mean lap/gauss)
    let designs = [
        (0.00_f64, 1.05_f64, 1.20_f64),
        (0.05, 0.85, 0.92),
        (0.10, 0.80, 0.88),
    ];
    let mut kappa_gauss_clean = 0.0;
    let mut kappa_gauss_contam10 = 0.0;
    let mut kappa_t_contam10 = 0.0;

    for (lvl, &(frac, t_bound, l_bound)) in designs.iter().enumerate() {
        let per_rep: Vec<(f64, f64, f64, f64, f64)> =
            par_replicate(20260817 + lvl as u64, REPS, |_, stream| {
                let (y, mu_true) = simulate_contaminated_ll(T, 0.1, 1.0, frac, 8.0, stream);
                let mut rmse = [0.0; 3];
                let mut kap = [0.0; 3];
                for (i, density) in [
                    DcsDensity::Gaussian,
                    DcsDensity::StudentT,
                    DcsDensity::Laplace,
                ]
                .into_iter()
                .enumerate()
                {
                    let res = DcsModel::new(&y, density)
                        .expect("model")
                        .fit()
                        .expect("fit");
                    rmse[i] = rmse_vs(&res.level, &mu_true);
                    kap[i] = res.params.kappa;
                }
                (rmse[0], rmse[1], rmse[2], kap[0], kap[1])
            })
            .expect("replicate");

        let n = REPS as f64;
        let ratio_t = per_rep.iter().map(|x| x.1 / x.0).sum::<f64>() / n;
        let ratio_l = per_rep.iter().map(|x| x.2 / x.0).sum::<f64>() / n;
        let mean_kg = per_rep.iter().map(|x| x.3).sum::<f64>() / n;
        let mean_kt = per_rep.iter().map(|x| x.4).sum::<f64>() / n;
        eprintln!(
            "[dcs robustness {:.0}%] mean RMSE ratio t/gauss {ratio_t:.3} \
             laplace/gauss {ratio_l:.3} | mean kappa gauss {mean_kg:.3} \
             t {mean_kt:.3}",
            100.0 * frac
        );
        assert!(
            ratio_t < t_bound,
            "{frac}: mean t/gauss RMSE ratio {ratio_t} not below {t_bound}"
        );
        assert!(
            ratio_l < l_bound,
            "{frac}: mean laplace/gauss RMSE ratio {ratio_l} not below {l_bound}"
        );
        if frac == 0.0 {
            // No robustness tax the other way either.
            assert!(ratio_t > 0.95, "t beat gaussian on clean data? {ratio_t}");
            kappa_gauss_clean = mean_kg;
        }
        if frac == 0.10 {
            kappa_gauss_contam10 = mean_kg;
            kappa_t_contam10 = mean_kt;
        }
    }

    // The gain-collapse mechanism (lab failure mode #5): contamination
    // makes the Gaussian MLE shrink its gain while DCS-t raises its own.
    assert!(
        kappa_gauss_contam10 < 0.6 * kappa_gauss_clean,
        "no Gaussian gain collapse: {kappa_gauss_contam10} vs clean \
         {kappa_gauss_clean}"
    );
    assert!(
        kappa_t_contam10 > kappa_gauss_contam10,
        "DCS-t gain {kappa_t_contam10} not above collapsed Gaussian gain \
         {kappa_gauss_contam10}"
    );
}

/// The fixture-frozen contaminated series (the exact arrays the Python
/// binding tests re-pin) reproduce the same verdict one seed at a time.
#[test]
fn fixture_contaminated_series_verdict() {
    let fx = load_fixture();
    for (key, t_bound) in [("sim_contam5", 0.90), ("sim_contam10", 0.85)] {
        let case = &fx[key];
        let y = as_f64_vec(&case["y"]);
        let mu_true = as_f64_vec(&case["mu_true"]);
        let g = DcsModel::new(&y, DcsDensity::Gaussian)
            .expect("model")
            .fit()
            .expect("fit");
        let t = DcsModel::new(&y, DcsDensity::StudentT)
            .expect("model")
            .fit()
            .expect("fit");
        let (rg, rt) = (rmse_vs(&g.level, &mu_true), rmse_vs(&t.level, &mu_true));
        eprintln!(
            "[dcs fixture {key}] level RMSE gauss {rg:.4} t {rt:.4} ratio {:.3}",
            rt / rg
        );
        assert!(
            rt < t_bound * rg,
            "{key}: DCS-t RMSE {rt} not below {t_bound} x Gaussian {rg}"
        );
    }
}

/// Claim 3 — no clean-data tax, and Gaussian nesting in both directions.
///
/// On the clean fixture series (the same one the statsmodels golden runs
/// on): (i) the DCS-t level path pays no measurable robustness tax
/// against the Gaussian filter's; (ii) `nu` runs toward the Gaussian
/// boundary and the optimizer honestly reports non-convergence (the same
/// certificate semantics as the volatility model on Gaussian data);
/// (iii) the fitted Gaussian `kappa` matches the steady-state gain mapped
/// from the statsmodels UC-MLE up to the finite-sample transient
/// (measured |diff| 1.3e-3 here; the exact Kalman filter's gain varies
/// during its diffuse transient, so equality is only asymptotic).
#[test]
fn clean_data_nesting_and_no_robustness_tax() {
    let fx = load_fixture();
    let case = &fx["gaussian_ss"][0];
    let y = as_f64_vec(&case["y"]);
    let mu_true = as_f64_vec(&case["mu_true"]);
    let kappa_ss = case["map"]["kappa"].as_f64().expect("kappa");

    let gauss = DcsModel::new(&y, DcsDensity::Gaussian)
        .expect("model")
        .fit()
        .expect("fit");
    let t = DcsModel::new(&y, DcsDensity::StudentT)
        .expect("model")
        .fit()
        .expect("fit");

    // (i) no tax: identical level accuracy on clean data (measured ratio
    // 1.0002 on this seed).
    let (rg, rt) = (rmse_vs(&gauss.level, &mu_true), rmse_vs(&t.level, &mu_true));
    eprintln!(
        "[dcs clean] level RMSE gauss {rg:.4} t {rt:.4} ratio {:.4}; \
         nu_hat {:.1}; kappa gauss {:.5} vs steady-state {:.5}",
        rt / rg,
        t.params.nu,
        gauss.params.kappa,
        kappa_ss
    );
    assert!(rt < 1.02 * rg, "robustness tax on clean data: {rt} vs {rg}");
    // ...and the paths themselves nearly coincide.
    let path_gap = rmse_vs(&t.level, &gauss.level);
    assert!(path_gap < 0.05, "t-vs-gaussian path RMSE {path_gap}");

    // (ii) nu runs to the Gaussian boundary; the certificate is honest.
    assert!(
        t.params.nu > 30.0,
        "expected nu at the Gaussian boundary, got {}",
        t.params.nu
    );
    assert!(
        !t.converged,
        "Student-t fit on clean Gaussian data claimed convergence at \
         nu = {:e} — there is no interior optimum to certify",
        t.params.nu
    );
    // The Gaussian control on the same data does converge: same series,
    // same code path, opposite certificate — that is the signal.
    assert!(gauss.converged);

    // (iii) nesting against the independent statsmodels UC-MLE, through
    // the steady-state mapping, up to the transient.
    assert!(
        (gauss.params.kappa - kappa_ss).abs() < 0.02,
        "fitted kappa {} vs steady-state gain {}",
        gauss.params.kappa,
        kappa_ss
    );
}

/// Claim 4 — observed-information SEs are usable: on one long well-posed
/// DCS-t path (T = 2000) the SEs are finite and positive and the truth
/// lies within conventional multiples of them.
#[test]
fn observed_information_ses_cover_truth_on_a_long_path() {
    let (kappa, scale, nu) = (0.15, 1.0, 5.0);
    let mut stream = Stream::new(97);
    let y = simulate_dcs_t(2000, kappa, scale, nu, &mut stream);
    let res = DcsModel::new(&y, DcsDensity::StudentT)
        .expect("model")
        .fit()
        .expect("fit");

    eprintln!(
        "[dcs se] kappa {:.4} (se {:.4}) scale {:.4} (se {:.4}) nu {:.2} \
         (se {:.2})",
        res.params.kappa, res.se.kappa, res.params.scale, res.se.scale, res.params.nu, res.se.nu
    );
    for (name, se) in [
        ("kappa", res.se.kappa),
        ("scale", res.se.scale),
        ("nu", res.se.nu),
    ] {
        assert!(se.is_finite() && se > 0.0, "se({name}) = {se}");
    }
    assert!(
        (res.params.kappa - kappa).abs() < 4.0 * res.se.kappa,
        "kappa {} vs true {kappa}, se {}",
        res.params.kappa,
        res.se.kappa
    );
    assert!(
        (res.params.scale - scale).abs() < 4.0 * res.se.scale,
        "scale {} vs true {scale}, se {}",
        res.params.scale,
        res.se.scale
    );
    assert!(
        (res.params.nu - nu).abs() < 4.0 * res.se.nu,
        "nu {} vs true {nu}, se {}",
        res.params.nu,
        res.se.nu
    );
    // The SE scale is sane: a T = 2000 gain SE in the low hundredths, not
    // zero-collapsed and not parameter-sized.
    assert!(res.se.kappa < 0.1, "se(kappa) {}", res.se.kappa);
}

/// The Laplace sign filter recovers its own parameters, fitted by the
/// exact hard-sign likelihood (no smoothing — a deliberate divergence
/// from the lab prototype, whose L-BFGS-B needed a tanh-smoothed sign;
/// derivative-free Nelder-Mead does not).
///
/// Twelve seeded reps of T = 1200: single-path kappa is genuinely noisy
/// for a sign filter (each step carries only the *sign* of the
/// innovation, so the likelihood is locally flat in kappa; single-path
/// errors near 0.09 were observed while calibrating), but the mean
/// recovers. Measured on this seed set: mean kappa 0.1970 (true 0.2),
/// mean scale 0.8061 (true 0.8), per-rep max |kappa error| 0.0406.
#[test]
fn laplace_sign_filter_recovers_its_own_parameters() {
    const REPS: usize = 12;
    let (kappa, b) = (0.2, 0.8);
    let fits: Vec<(f64, f64)> = par_replicate(31, REPS, |_, stream| {
        // Laplace draws by inverse CDF:
        // e = -b sgn(u - 1/2) ln(1 - 2 |u - 1/2|).
        let n = 1200;
        let mut y = vec![0.0; n];
        let mut m = 0.0_f64;
        for yt in y.iter_mut() {
            let u = uniform_pos(stream) - 0.5;
            let e = -b * u.signum() * (1.0 - 2.0 * u.abs()).ln();
            *yt = m + e;
            let sgn = if e == 0.0 { 0.0 } else { e.signum() };
            m += kappa * b * sgn;
        }
        let res = DcsModel::new(&y, DcsDensity::Laplace)
            .expect("model")
            .fit()
            .expect("fit");
        (res.params.kappa, res.params.scale)
    })
    .expect("replicate");

    let n = REPS as f64;
    let mean_k = fits.iter().map(|x| x.0).sum::<f64>() / n;
    let mean_b = fits.iter().map(|x| x.1).sum::<f64>() / n;
    let max_k_err = fits
        .iter()
        .map(|x| (x.0 - kappa).abs())
        .fold(0.0_f64, f64::max);
    eprintln!(
        "[dcs laplace] mean kappa {mean_k:.4} (true {kappa}) mean scale \
         {mean_b:.4} (true {b}) max |kappa err| {max_k_err:.4}"
    );
    assert!((mean_k - kappa).abs() < 0.05, "mean kappa {mean_k}");
    assert!((mean_b - b).abs() < 0.03, "mean scale {mean_b}");
}
