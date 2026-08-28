//! Property tests for the STAR family: statistical properties a golden
//! transcription cannot prove.
//!
//! * Seeded MC size of the LM3 F-form linearity test under an AR null
//!   (~ nominal at 5% and 10%; measured rate and MC standard error
//!   printed — run with `--nocapture`).
//! * Seeded MC power under an LSTAR alternative, plus the H-sequence's
//!   LSTAR/ESTAR selection frequencies on both DGPs.
//! * Seeded MC parameter recovery for the estimator (bias/RMSE for the
//!   phis and `c` at T = 250 and T = 500; gamma is reported honestly —
//!   large-gamma flatness makes it the literature's known weak spot,
//!   Terasvirta 1994).
//! * The LSTAR -> SETAR limit: at enormous fixed gamma the concentrated
//!   fit equals a hard-threshold two-regime OLS computed *in this test*
//!   (an independent transcription — never against `tsecon::setar`, no
//!   circularity), and the Gauss-Newton SEs honestly report
//!   `se_valid = false` (the gamma Jacobian column vanishes).
//! * Scale/location equivariance of the grid stage (exact) and of the
//!   refined fit (loose — the refinement's initial simplex is not
//!   shift-invariant).
//! * `gamma_at_boundary` fires on hard-threshold data; `converged` is
//!   reported; degenerate inputs raise the documented teaching errors.

use tsecon_bootstrap::WildWeights;
use tsecon_regime::{star, star_eval, star_test, RegimeError, StarModel};
use tsecon_rng::Stream;

// ------------------------------------------------------------ simulation

/// LSTAR(1), delay 1: `y_t = c1 + a1 y_{t-1} + G (c2 + a2 y_{t-1}) + e_t`
/// with `G = 1/(1 + exp(-gamma (y_{t-1} - c)))`, standard normal errors.
fn sim_lstar1(
    stream: &mut Stream,
    t: usize,
    phi1: [f64; 2],
    phi2: [f64; 2],
    gamma: f64,
    c: f64,
) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        let g = 1.0 / (1.0 + (-gamma * (y[i - 1] - c)).exp());
        y[i] = phi1[0]
            + phi1[1] * y[i - 1]
            + g * (phi2[0] + phi2[1] * y[i - 1])
            + WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// ESTAR(1), delay 1, with `G = 1 - exp(-gamma (y_{t-1} - c)^2)`.
fn sim_estar1(
    stream: &mut Stream,
    t: usize,
    phi1: [f64; 2],
    phi2: [f64; 2],
    gamma: f64,
    c: f64,
) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        let d = y[i - 1] - c;
        let g = 1.0 - (-gamma * d * d).exp();
        y[i] = phi1[0]
            + phi1[1] * y[i - 1]
            + g * (phi2[0] + phi2[1] * y[i - 1])
            + WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// Linear AR(1) `y_t = phi y_{t-1} + e_t`.
fn sim_ar1(stream: &mut Stream, t: usize, phi: f64) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        y[i] = phi * y[i - 1] + WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// Hard-threshold SETAR(1) with small noise, for the boundary flag.
fn sim_setar1(
    stream: &mut Stream,
    t: usize,
    low: [f64; 2],
    high: [f64; 2],
    gamma: f64,
    sigma: f64,
) -> Vec<f64> {
    let burn = 100;
    let mut y = vec![0.0_f64; t + burn + 1];
    for i in 1..y.len() {
        let c = if y[i - 1] <= gamma { low } else { high };
        y[i] = c[0] + c[1] * y[i - 1] + sigma * WildWeights::Normal.draw(stream);
    }
    y[(burn + 1)..].to_vec()
}

/// Direct OLS of `y` on `[1, x]` by 2x2 normal equations — the test's own
/// independent check implementation.
fn direct_ols2(x: &[f64], y: &[f64]) -> [f64; 2] {
    let n = y.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&v| v * v).sum();
    let sxy: f64 = x.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let det = n * sxx - sx * sx;
    [(sy * sxx - sx * sxy) / det, (n * sxy - sx * sy) / det]
}

// -------------------------------------------------------------- size MC

#[test]
fn lm3_null_rejection_rate_is_near_nominal() {
    // Linear-AR(1) null at T = 200 and T = 500. The LM3 F-form is known
    // to be slightly conservative in small samples (that conservatism is
    // why Terasvirta recommends it over the chi-squared form, which
    // over-rejects); the bands below allow it while still catching a
    // broken statistic. Binomial 3-sigma at 5% with 400 draws: +/- 0.033.
    for &t in &[200usize, 500] {
        let n_series = 400;
        let mut streams = Stream::substreams(20260827 + t as u64, n_series).expect("substreams");
        let mut reject05 = 0usize;
        let mut reject10 = 0usize;
        let mut psum = 0.0_f64;
        for stream in streams.iter_mut() {
            let y = sim_ar1(stream, t, 0.5);
            let r = star_test(&y, 1, &[1]).expect("test runs");
            let p = r.tests[0].lm3_f_p_value;
            if p <= 0.05 {
                reject05 += 1;
            }
            if p <= 0.10 {
                reject10 += 1;
            }
            psum += p;
        }
        let n = n_series as f64;
        let rate05 = reject05 as f64 / n;
        let rate10 = reject10 as f64 / n;
        let se05 = (0.05_f64 * 0.95 / n).sqrt();
        let se10 = (0.10_f64 * 0.90 / n).sqrt();
        println!(
            "LM3 null MC (T={t}, {n_series} reps): reject@5% = {rate05:.4} \
             (MC se {se05:.4}), reject@10% = {rate10:.4} (MC se {se10:.4}), \
             mean p = {:.3}",
            psum / n
        );
        assert!(
            (0.005..=0.10).contains(&rate05),
            "T={t}: 5% rejection rate {rate05} far from nominal"
        );
        assert!(
            (0.03..=0.17).contains(&rate10),
            "T={t}: 10% rejection rate {rate10} far from nominal"
        );
        assert!(
            (0.40..=0.60).contains(&(psum / n)),
            "T={t}: mean null p-value {} far from 0.5",
            psum / n
        );
    }
}

// ------------------------------------------------------------- power MC

#[test]
fn lm3_power_and_model_selection_on_star_alternatives() {
    // LSTAR alternative: intercept-and-slope switch, T = 250.
    let n_series = 400;
    let mut streams = Stream::substreams(777, n_series).expect("substreams");
    let mut reject = 0usize;
    let mut chose_lstar = 0usize;
    for stream in streams.iter_mut() {
        let y = sim_lstar1(stream, 250, [1.0, 0.6], [-2.0, -0.4], 2.0, 0.0);
        let r = star_test(&y, 1, &[1]).expect("test runs");
        let t = &r.tests[0];
        if t.lm3_f_p_value <= 0.05 {
            reject += 1;
            if t.suggested == StarModel::Lstar {
                chose_lstar += 1;
            }
        }
    }
    let n = n_series as f64;
    let power = reject as f64 / n;
    let se = (power * (1.0 - power) / n).sqrt();
    let sel = chose_lstar as f64 / reject.max(1) as f64;
    println!(
        "LM3 power MC (LSTAR, T=250, {n_series} reps): reject@5% = {power:.3} \
         (MC se {se:.3}); H-sequence chose LSTAR in {sel:.3} of rejections"
    );
    assert!(power >= 0.60, "power {power} too low on a strong LSTAR");
    assert!(
        sel >= 0.50,
        "H-sequence chose LSTAR in only {sel} of rejections"
    );

    // ESTAR alternative: random-walk inner band, reverting outside.
    let mut streams = Stream::substreams(778, n_series).expect("substreams");
    let mut reject_e = 0usize;
    let mut chose_estar = 0usize;
    for stream in streams.iter_mut() {
        let y = sim_estar1(stream, 250, [0.0, 1.0], [0.0, -0.9], 0.25, 0.0);
        let r = star_test(&y, 1, &[1]).expect("test runs");
        let t = &r.tests[0];
        if t.lm3_f_p_value <= 0.05 {
            reject_e += 1;
            if t.suggested == StarModel::Estar {
                chose_estar += 1;
            }
        }
    }
    let power_e = reject_e as f64 / n;
    let sel_e = chose_estar as f64 / reject_e.max(1) as f64;
    println!(
        "LM3 power MC (ESTAR, T=250, {n_series} reps): reject@5% = {power_e:.3}; \
         H-sequence chose ESTAR in {sel_e:.3} of rejections"
    );
    assert!(power_e >= 0.30, "power {power_e} too low on the ESTAR DGP");
    assert!(
        sel_e >= 0.50,
        "H-sequence chose ESTAR in only {sel_e} of rejections"
    );
}

// --------------------------------------------------------- recovery MC

#[test]
fn mc_parameter_recovery_lstar() {
    // The model-card evidence: seeded replications of
    //   y_t = 1 + 0.6 y_{t-1} + G(2, 0; y_{t-1}) (-2 - 0.4 y_{t-1}) + e_t
    // at T = 250 and T = 500. c and the phis recover in the median; gamma
    // is wildly noisy — many draws run to a gamma boundary — which is the
    // literature's known result (Terasvirta 1994: accurate estimation of
    // gamma needs many observations near c; tsDyn's lstar routinely warns
    // that gamma hit its bound). Mean bias/RMSE are printed for the
    // record but gated on MEDIANS: on boundary draws the phi2 block
    // trades off against gamma, so a handful of unidentified fits
    // dominate any mean.
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    };
    for &t in &[250usize, 500] {
        let n_reps = 200;
        let mut streams = Stream::substreams(9000 + t as u64, n_reps).expect("substreams");
        let truth = [1.0_f64, 0.6, -2.0, -0.4]; // phi1, then phi2
        let mut errs: [Vec<f64>; 4] = Default::default();
        let mut c_est: Vec<f64> = Vec::with_capacity(n_reps);
        let mut gamma_std: Vec<f64> = Vec::with_capacity(n_reps);
        let mut n_boundary = 0usize;
        let mut n_converged = 0usize;
        let mut interior_errs: [Vec<f64>; 4] = Default::default();
        for stream in streams.iter_mut() {
            let y = sim_lstar1(stream, t, [1.0, 0.6], [-2.0, -0.4], 2.0, 0.0);
            let fit = star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25).expect("fit runs");
            let est = [
                fit.eval.coefs_linear[0],
                fit.eval.coefs_linear[1],
                fit.eval.coefs_nonlinear[0],
                fit.eval.coefs_nonlinear[1],
            ];
            for j in 0..4 {
                errs[j].push(est[j] - truth[j]);
                if !fit.gamma_at_boundary {
                    interior_errs[j].push(est[j] - truth[j]);
                }
            }
            c_est.push(fit.c);
            gamma_std.push(fit.gamma_standardized);
            if fit.gamma_at_boundary {
                n_boundary += 1;
            }
            if fit.converged {
                n_converged += 1;
            }
        }
        let nf = n_reps as f64;
        let mut bias = [0.0_f64; 4];
        let mut rmse = [0.0_f64; 4];
        let mut med = [0.0_f64; 4];
        let mut med_int = [0.0_f64; 4];
        for j in 0..4 {
            bias[j] = errs[j].iter().sum::<f64>() / nf;
            rmse[j] = (errs[j].iter().map(|e| e * e).sum::<f64>() / nf).sqrt();
            med[j] = median(&mut errs[j]);
            med_int[j] = median(&mut interior_errs[j]);
        }
        let n_interior = n_reps - n_boundary;
        let c_mae: f64 = c_est.iter().map(|v| v.abs()).sum::<f64>() / nf;
        let mut c_abs: Vec<f64> = c_est.iter().map(|v| v.abs()).collect();
        let c_abs_med = median(&mut c_abs);
        let c_med = median(&mut c_est);
        let g_med = median(&mut gamma_std);
        let g_q1 = gamma_std[n_reps / 4];
        let g_q3 = gamma_std[3 * n_reps / 4];
        println!(
            "LSTAR recovery MC (T={t}, {n_reps} reps, true std gamma ~2.9): \
             bias/RMSE/median-err phi1_0 {:+.3}/{:.3}/{:+.3}, \
             phi1_1 {:+.3}/{:.3}/{:+.3}, phi2_0 {:+.3}/{:.3}/{:+.3}, \
             phi2_1 {:+.3}/{:.3}/{:+.3}; interior-only medians \
             [{:+.3}, {:+.3}, {:+.3}, {:+.3}] over {n_interior} fits; \
             c median {:+.3} (true 0), median |c| {:.3}, mean |c| {:.3}; \
             standardized gamma median {:.2} [IQR {:.2}, {:.2}]; \
             gamma at boundary {}/{}; converged {}/{}",
            bias[0],
            rmse[0],
            med[0],
            bias[1],
            rmse[1],
            med[1],
            bias[2],
            rmse[2],
            med[2],
            bias[3],
            rmse[3],
            med[3],
            med_int[0],
            med_int[1],
            med_int[2],
            med_int[3],
            c_med,
            c_abs_med,
            c_mae,
            g_med,
            g_q1,
            g_q3,
            n_boundary,
            n_reps,
            n_converged,
            n_reps
        );
        // Gates. Overall medians are polluted by the boundary draws
        // (gamma at a bound attenuates the phi2 block — the honest
        // finite-sample story told in the model card), so the tight gate
        // is on the *interior* fits; the overall medians get a loose
        // sanity band only.
        assert!(
            n_interior >= n_reps / 4,
            "T={t}: only {n_interior}/{n_reps} fits kept gamma off the boundary"
        );
        for (j, m) in med_int.iter().enumerate() {
            assert!(
                m.abs() < 0.35,
                "T={t} interior coefficient median error [{j}] = {m}"
            );
        }
        for (j, m) in med.iter().enumerate() {
            assert!(
                m.abs() < 1.2,
                "T={t} overall coefficient median error [{j}] = {m}"
            );
        }
        // c is centered but disperse: with a smooth transition, c and
        // gamma trade off, so |c| spreads to ~0.3-0.5 sd(s) at these T
        // (measured; quoted in the model card).
        assert!(c_med.abs() < 0.20, "T={t} median c error {c_med}");
        assert!(c_abs_med < 0.60, "T={t} median |c| error {c_abs_med}");
        // Gamma: no tight pin — the honest claim is only that the median
        // standardized gamma is in a broad sane band around the truth
        // (the measured spread is quoted in the model card).
        assert!(
            (0.5..=100.0).contains(&g_med),
            "T={t} median standardized gamma {g_med} outside the sane band"
        );
        assert!(
            n_converged >= n_reps * 9 / 10,
            "T={t}: only {n_converged}/{n_reps} refinements converged"
        );
    }
}

// ------------------------------------------------- LSTAR -> SETAR limit

#[test]
fn lstar_limit_is_a_hard_threshold_fit_and_ses_go_invalid() {
    // At enormous gamma the logistic is exactly 0/1 at every observation
    // (c is placed between two adjacent order statistics), so the
    // concentrated fit must equal the two-regime split OLS computed here
    // directly — this test's own transcription, not tsecon::setar.
    let mut stream = Stream::new(314);
    let y = sim_lstar1(&mut stream, 400, [1.0, 0.6], [-2.0, -0.4], 8.0, 0.0);

    // Place c strictly between the two order statistics of y_{t-1}
    // nearest 0, so no point sits in the sigmoid's blur zone.
    let mut z: Vec<f64> = y[..y.len() - 1].to_vec();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let i = z.partition_point(|&v| v <= 0.0);
    let c = 0.5 * (z[i - 1] + z[i]);
    let gamma = 1e6;

    let r = star_eval(&y, 1, 1, StarModel::Lstar, gamma, c, true).expect("eval runs");

    // Split OLS on the same sample (t = 1..T-1; regressor y_{t-1}).
    let mut lo_x = Vec::new();
    let mut lo_y = Vec::new();
    let mut hi_x = Vec::new();
    let mut hi_y = Vec::new();
    for t in 1..y.len() {
        if y[t - 1] <= c {
            lo_x.push(y[t - 1]);
            lo_y.push(y[t]);
        } else {
            hi_x.push(y[t - 1]);
            hi_y.push(y[t]);
        }
    }
    let bl = direct_ols2(&lo_x, &lo_y);
    let bh = direct_ols2(&hi_x, &hi_y);
    // phi1 = low-regime coefficients; phi1 + phi2 = high-regime.
    for j in 0..2 {
        assert!(
            (r.coefs_linear[j] - bl[j]).abs() < 1e-7,
            "linear part [{j}]: {} vs split OLS {}",
            r.coefs_linear[j],
            bl[j]
        );
        assert!(
            (r.coefs_linear[j] + r.coefs_nonlinear[j] - bh[j]).abs() < 1e-7,
            "regime sum [{j}]: {} vs split OLS {}",
            r.coefs_linear[j] + r.coefs_nonlinear[j],
            bh[j]
        );
    }
    // Every G_t is numerically 0 or 1.
    assert!(r.transition.iter().all(|&g| g < 1e-12 || (1.0 - g) < 1e-12));
    // And the Gauss-Newton SEs honestly refuse: the gamma column of the
    // Jacobian is identically ~0, J'J is singular.
    assert!(!r.se_valid, "SEs must be invalid in the step limit");
    assert!(r.se_gamma.is_nan() && r.se_c.is_nan());
}

// --------------------------------------------------------- equivariance

#[test]
fn grid_stage_is_scale_and_location_equivariant() {
    let mut stream = Stream::new(17);
    let y = sim_lstar1(&mut stream, 300, [1.0, 0.6], [-2.0, -0.4], 2.0, 0.0);
    let (a, b) = (2.5_f64, 3.0_f64);
    let yt: Vec<f64> = y.iter().map(|&v| a * v + b).collect();

    let f0 = star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 12, 11).expect("fit runs");
    let f1 = star(&yt, 1, &[1], StarModel::Lstar, 0.15, true, 12, 11).expect("fit runs");

    let rel = |x: f64, e: f64| ((x - e) / e).abs();
    // The grid is built in standardized units: raw gammas scale by 1/a,
    // c candidates map affinely, the SSR surface scales by a^2, and the
    // best cell is identical.
    assert!(rel(f1.s_sd, a * f0.s_sd) < 1e-12);
    for (g1, g0) in f1.grid_gamma.iter().zip(&f0.grid_gamma) {
        assert!(rel(*g1, g0 / a) < 1e-10, "gamma grid {g1} vs {}", g0 / a);
    }
    for (c1, c0) in f1.grid_c.iter().zip(&f0.grid_c) {
        assert!(
            (c1 - (a * c0 + b)).abs() < 1e-10,
            "c grid {c1} vs {}",
            a * c0 + b
        );
    }
    for (s1, s0) in f1.ssr_grid.iter().zip(&f0.ssr_grid) {
        if s0.is_nan() {
            assert!(s1.is_nan());
        } else {
            assert!(
                rel(*s1, a * a * s0) < 1e-8,
                "ssr grid {s1} vs {}",
                a * a * s0
            );
        }
    }
    assert_eq!(f1.best_cell, f0.best_cell);

    // The refined optimum of the transformed problem is the transformed
    // optimum; the refinement's initial simplex is not shift-invariant,
    // so only loose agreement is claimed.
    assert!(
        rel(f1.gamma_standardized, f0.gamma_standardized) < 1e-2,
        "standardized gamma {} vs {}",
        f1.gamma_standardized,
        f0.gamma_standardized
    );
    assert!(
        (f1.c - (a * f0.c + b)).abs() < 1e-2 * f0.s_sd * a,
        "refined c {} vs {}",
        f1.c,
        a * f0.c + b
    );
    assert!(rel(f1.eval.ssr, a * a * f0.eval.ssr) < 1e-6);
}

// ----------------------------------------------------- honesty flagging

#[test]
fn hard_threshold_data_sets_the_gamma_boundary_flag() {
    // A sharply separated SETAR with small noise: the best LSTAR gamma
    // runs off to the cap, and the fit must say so instead of reporting
    // a precise-looking gamma.
    let mut stream = Stream::new(2718);
    let y = sim_setar1(&mut stream, 400, [1.0, 0.5], [-1.0, 0.3], 0.0, 0.25);
    let fit = star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25).expect("fit runs");
    assert!(
        fit.gamma_at_boundary,
        "standardized gamma {} did not flag the boundary on step data",
        fit.gamma_standardized
    );
    // The step fit is still a *good* fit — c near the true threshold.
    assert!(
        fit.c.abs() < 0.3,
        "threshold location {} far from the true 0",
        fit.c
    );
}

#[test]
fn smooth_data_does_not_set_the_boundary_flag() {
    let mut stream = Stream::new(161803);
    let y = sim_lstar1(&mut stream, 400, [1.0, 0.6], [-2.0, -0.4], 2.0, 0.0);
    let fit = star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25).expect("fit runs");
    assert!(
        !fit.gamma_at_boundary,
        "smooth LSTAR data flagged the gamma boundary (standardized gamma {})",
        fit.gamma_standardized
    );
    assert!(fit.converged, "refinement did not converge on clean data");
    // Self-consistency: the reported eval is the concentrated fit at the
    // reported (gamma, c).
    let e = star_eval(&y, 1, 1, StarModel::Lstar, fit.gamma, fit.c, true).expect("eval runs");
    assert!((e.ssr - fit.eval.ssr).abs() / fit.eval.ssr < 1e-12);
    assert_eq!(e.coefs_linear, fit.eval.coefs_linear);
}

// ------------------------------------------------------------ degeneracy

#[test]
fn degenerate_inputs_raise_teaching_errors() {
    let mut stream = Stream::new(1);
    let y = sim_ar1(&mut stream, 120, 0.4);

    // Constant series.
    let c = vec![1.0; 100];
    assert!(matches!(
        star(&c, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        star_test(&c, 1, &[1]),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // Near-constant transition variable: one blip in an otherwise flat
    // series (the series is not constant, but sd(y_{t-d}) ~ 0 relative
    // to its level).
    let mut flat = vec![5.0; 120];
    flat[60] += 5e-10;
    assert!(matches!(
        star(&flat, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // Too short.
    assert!(matches!(
        star(&y[..6], 1, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InsufficientData { .. })
    ));
    assert!(matches!(
        star_test(&y[..6], 1, &[1]),
        Err(RegimeError::InsufficientData { .. })
    ));

    // p = 0, delay 0, empty delays.
    assert!(matches!(
        star(&y, 0, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        star(&y, 1, &[0], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InvalidParameter { name: "delay", .. })
    ));
    assert!(matches!(
        star(&y, 1, &[], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InvalidSpec { .. })
    ));
    assert!(matches!(
        star_test(&y, 1, &[]),
        Err(RegimeError::InvalidSpec { .. })
    ));

    // trim outside (0, 0.5); grid dimensions below 2.
    assert!(matches!(
        star(&y, 1, &[1], StarModel::Lstar, 0.5, true, 25, 25),
        Err(RegimeError::InvalidParameter { name: "trim", .. })
    ));
    assert!(matches!(
        star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 1, 25),
        Err(RegimeError::InvalidParameter {
            name: "n_gamma",
            ..
        })
    ));
    assert!(matches!(
        star(&y, 1, &[1], StarModel::Lstar, 0.15, true, 25, 1),
        Err(RegimeError::InvalidParameter { name: "n_c", .. })
    ));

    // NaN observation.
    let mut bad = y.clone();
    bad[10] = f64::NAN;
    assert!(matches!(
        star(&bad, 1, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::NonFinite { .. })
    ));

    // star_eval: gamma <= 0, non-finite c.
    assert!(matches!(
        star_eval(&y, 1, 1, StarModel::Lstar, 0.0, 0.0, true),
        Err(RegimeError::InvalidParameter { name: "gamma", .. })
    ));
    assert!(matches!(
        star_eval(&y, 1, 1, StarModel::Lstar, 1.0, f64::NAN, true),
        Err(RegimeError::NonFinite { .. })
    ));

    // A transition numerically constant over the sample (huge gamma, c
    // far below every s_t: G = 1 everywhere) makes [x, Gx] collinear;
    // the refusal must name the data property and a fix, not just the
    // internal solver (audit round 10, finding 3d).
    let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let err = star_eval(&y, 1, 1, StarModel::Lstar, 1e8, ymin - 100.0, true).unwrap_err();
    assert!(matches!(err, RegimeError::Singular { .. }), "{err:?}");
    let msg = err.to_string();
    assert!(
        msg.contains("collinear") && msg.contains("numerically constant"),
        "singular STAR OLS must name the collinear/constant design: {msg}"
    );
    assert!(
        msg.contains("reduce p") || msg.contains("move (gamma, c)"),
        "singular STAR OLS must offer a fix: {msg}"
    );
}

/// Audit round 10, finding 1: `star_test` must refuse an empty series or
/// a delay at/past the end of the sample with the same
/// [`RegimeError::InsufficientData`] contract as every sibling estimator
/// — before this fix, `build_design` ran first and the usable-row count
/// wrapped (a capacity-overflow panic), and the `T - 1` / `T` boundary
/// was miscategorized as a near-constant transition variable.
#[test]
fn star_test_refuses_out_of_sample_delays_as_insufficiency() {
    let mut stream = Stream::new(7);
    let t = 50usize;
    let y = sim_ar1(&mut stream, t, 0.4);
    let p = 2usize;

    // Empty series.
    assert!(matches!(
        star_test(&[], p, &[1]),
        Err(RegimeError::InsufficientData { .. })
    ));

    // delay = T - 1, T, T + 1, T + 50: all insufficiency, never the
    // near-constant-transition category, never a panic.
    for d in [t - 1, t, t + 1, t + 50] {
        let err = star_test(&y, p, &[d]).unwrap_err();
        match err {
            RegimeError::InsufficientData { needed, got } => {
                // q = p + 1 (d > p), k0 = 1 + q, rows = k0 + 3q + 1 = 14,
                // measured from start = d.
                assert_eq!(got, t, "delay {d}: got must be the supplied length");
                assert_eq!(needed, d + 14, "delay {d}: sibling-contract minimum");
            }
            other => panic!("delay {d}: expected InsufficientData, got {other:?}"),
        }
        // The message keeps the shared teaching form.
        let msg = star_test(&y, p, &[d]).unwrap_err().to_string();
        assert!(
            msg.contains("insufficient data") && msg.contains("required"),
            "delay {d}: message lost the sibling form: {msg}"
        );
    }

    // A delay list mixing a valid and an out-of-sample delay refuses too
    // (each battery runs on its own usable sample).
    assert!(matches!(
        star_test(&y, p, &[1, t + 50]),
        Err(RegimeError::InsufficientData { .. })
    ));

    // The estimator and fixed-parameter surfaces share the contract.
    assert!(matches!(
        star(&[], p, &[1], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InsufficientData { .. })
    ));
    assert!(matches!(
        star(&y, p, &[1, t + 50], StarModel::Lstar, 0.15, true, 25, 25),
        Err(RegimeError::InsufficientData { .. })
    ));
    assert!(matches!(
        star_eval(&y, p, t + 50, StarModel::Lstar, 2.0, 0.0, true),
        Err(RegimeError::InsufficientData { .. })
    ));
}

// ---------------------------------------------- flag constructibility

/// Deterministic LCG noise in [-0.5, 0.5) — self-contained so the flag
/// constructions below are bit-reproducible.
fn lcg_noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
}

/// Audit round 10, finding 4 (sweep C's OPEN item): the BOTTOM-wall
/// `gamma_at_boundary` is reachable from data. A true LSTAR whose
/// standardized gamma sits below the grid bottom (0.5) makes the in-box
/// SSR decrease toward the wall; this construction pins the refinement
/// there, inside the 1e-9-relative detection band.
#[test]
fn bottom_wall_gamma_boundary_is_constructible() {
    let mut seed = 29u64;
    let mut y = vec![0.3_f64];
    for _ in 1..500 {
        let prev = *y.last().unwrap();
        let g = 1.0 / (1.0 + (-0.2 * prev).exp());
        let v: f64 = 0.2 + 0.6 * prev + g * (-0.9 * prev - 0.4) + 0.02 * lcg_noise(&mut seed);
        y.push(v.clamp(-50.0, 50.0));
    }
    let fit = star(&y, 1, &[1], StarModel::Lstar, 0.1, true, 25, 25).expect("fit runs");
    assert_eq!(
        fit.best_cell.0, 0,
        "the best grid cell must sit in the bottom gamma row (got {:?})",
        fit.best_cell
    );
    assert!(
        fit.gamma_standardized <= 0.5 * (1.0 + 1e-9),
        "refined standardized gamma {} left the bottom wall",
        fit.gamma_standardized
    );
    assert!(
        fit.gamma_at_boundary,
        "bottom-wall fit did not set gamma_at_boundary (standardized gamma {})",
        fit.gamma_standardized
    );
}

/// Audit round 10, finding 4 (sweep C's OPEN item): `converged = false`
/// is reachable from data. The Nelder-Mead refinement certifies an
/// absolute f-spread of 1e-8; once the SSR magnitude exceeds the float
/// resolution floor `f_tol / (4 eps)` (~1.1e7 — e.g. a series measured
/// in "large" units), the optimizer honestly terminates
/// ObjectiveResolution and reports `converged = false` rather than
/// certifying a tolerance it could not verify.
#[test]
fn converged_false_is_constructible() {
    let mut seed = 5u64;
    let mut y = vec![0.3_f64];
    for _ in 1..300 {
        let prev = *y.last().unwrap();
        let g = 1.0 / (1.0 + (-8.0 * prev).exp());
        let v: f64 = 0.2 + 0.6 * prev + g * (-0.9 * prev - 0.4) + 0.3 * lcg_noise(&mut seed);
        y.push(v.clamp(-50.0, 50.0));
    }
    // Sanity: in ordinary units the same series converges.
    let base = star(&y, 1, &[1], StarModel::Lstar, 0.1, true, 25, 25).expect("fit runs");
    assert!(base.converged, "base-scale fit unexpectedly unconverged");

    let big: Vec<f64> = y.iter().map(|v| v * 1e6).collect();
    let fit = star(&big, 1, &[1], StarModel::Lstar, 0.1, true, 25, 25).expect("fit runs");
    assert!(
        fit.eval.ssr > 1.1e7,
        "construction lost its premise: SSR {} below the resolution floor",
        fit.eval.ssr
    );
    assert!(
        !fit.converged,
        "SSR {} beyond the f_tol resolution floor must report converged = false",
        fit.eval.ssr
    );
}

// --------------------------------------------------------- delay search

#[test]
fn delay_search_recovers_the_true_delay() {
    // LSTAR switching on the second lag: both the test battery's rule
    // (smallest LM3-F p-value) and the estimator's rule (smallest refined
    // SSR) should land on d = 2.
    let burn = 100;
    let t = 400;
    let mut stream = Stream::new(55);
    let mut y = vec![0.0_f64; t + burn + 2];
    for i in 2..y.len() {
        let g = 1.0 / (1.0 + (-3.0 * (y[i - 2] - 0.0)).exp());
        y[i] = 1.0
            + 0.5 * y[i - 1]
            + g * (-2.0 - 0.3 * y[i - 1])
            + WildWeights::Normal.draw(&mut stream);
    }
    let y = y[(burn + 2)..].to_vec();

    let r = star_test(&y, 1, &[1, 2, 3]).expect("test runs");
    assert_eq!(r.tests[r.best].delay, 2, "battery picked delay != 2");

    let fit = star(&y, 1, &[1, 2, 3], StarModel::Lstar, 0.15, true, 25, 25).expect("fit runs");
    assert_eq!(fit.delay, 2, "estimator picked delay != 2");
}
