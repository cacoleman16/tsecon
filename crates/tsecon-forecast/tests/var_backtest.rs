//! Golden, hand-computed, property (Monte Carlo size/power), and guardrail
//! tests for the VaR backtest battery ([`tsecon_forecast::var_backtest`]).
//!
//! Goldens come from `fixtures/var_backtest.json`
//! (`fixtures/generate_var_backtest_fixtures.py`, NumPy 2.4.6 / SciPy /
//! statsmodels 0.14.6): the Kupiec and Christoffersen LR statistics are
//! first-principles closed-form NumPy references, and the DQ statistic is a
//! statsmodels-OLS construction (fitted'fitted / (alpha(1-alpha))) on the
//! identical hit sequences — a genuine third-party pin for the regression
//! algebra. Two hand cases are re-derived in comments below; the Jorion
//! (Value at Risk, ch. 6) J.P. Morgan 1998 example (20 breaches in 252
//! days at 95%, LR_uc = 3.91) is the published-example pin.

use serde_json::Value;
use tsecon_forecast::{var_backtest, var_backtest_hits, ForecastError};

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/var_backtest.json"
    );
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Relative comparison; falls back to absolute when the reference is 0.
fn assert_close(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    if expected == 0.0 {
        assert!(
            actual.abs() <= rtol,
            "{ctx}: actual {actual}, expected 0 (atol {rtol})"
        );
    } else {
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel <= rtol,
            "{ctx}: actual {actual}, expected {expected}, rel err {rel:e} > {rtol:e}"
        );
    }
}

/// p-value tolerance: 1e-8 relative normally, loosened in the deep tail
/// where this crate's chi-squared survival function and SciPy agree to
/// ~1e-9 relative per unit of magnitude (the same policy as the GW golden).
fn assert_pvalue(actual: f64, expected: f64, ctx: &str) {
    let rtol = if expected >= 1e-10 { 1e-8 } else { 1e-4 };
    assert_close(actual, expected, rtol, ctx);
}

fn check_case(case: &Value, r: &tsecon_forecast::VarBacktestResult, name: &str) {
    assert_eq!(r.n, case["n"].as_u64().unwrap() as usize, "{name}: n");
    assert_eq!(
        r.n_violations,
        case["n_violations"].as_u64().unwrap() as usize,
        "{name}: n_violations"
    );
    assert_close(
        r.hit_rate,
        case["hit_rate"].as_f64().unwrap(),
        1e-14,
        &format!("{name}: hit_rate"),
    );
    assert_close(
        r.lr_uc,
        case["lr_uc"].as_f64().unwrap(),
        1e-12,
        &format!("{name}: lr_uc"),
    );
    assert_pvalue(
        r.p_uc,
        case["p_uc"].as_f64().unwrap(),
        &format!("{name}: p_uc"),
    );
    for (field, got) in [
        ("n00", r.n00),
        ("n01", r.n01),
        ("n10", r.n10),
        ("n11", r.n11),
    ] {
        assert_eq!(
            got,
            case[field].as_u64().unwrap() as usize,
            "{name}: {field}"
        );
    }
    assert_close(
        r.pi01,
        case["pi01"].as_f64().unwrap(),
        1e-14,
        &format!("{name}: pi01"),
    );
    assert_close(
        r.pi11,
        case["pi11"].as_f64().unwrap(),
        1e-14,
        &format!("{name}: pi11"),
    );
    assert_close(
        r.lr_ind,
        case["lr_ind"].as_f64().unwrap(),
        1e-12,
        &format!("{name}: lr_ind"),
    );
    assert_pvalue(
        r.p_ind,
        case["p_ind"].as_f64().unwrap(),
        &format!("{name}: p_ind"),
    );
    assert_close(
        r.lr_cc,
        case["lr_cc"].as_f64().unwrap(),
        1e-12,
        &format!("{name}: lr_cc"),
    );
    assert_pvalue(
        r.p_cc,
        case["p_cc"].as_f64().unwrap(),
        &format!("{name}: p_cc"),
    );
    // DQ: statsmodels-OLS third-party golden. The route differs (modified
    // Gram-Schmidt projection here, pinv/SVD in statsmodels), so 1e-9
    // relative.
    assert_eq!(
        r.dq_df,
        case["dq_df"].as_u64().unwrap() as usize,
        "{name}: dq_df"
    );
    assert_eq!(
        r.dq_includes_var,
        case["includes_var"].as_bool().unwrap(),
        "{name}: dq_includes_var"
    );
    assert_eq!(
        r.dq_var_dropped,
        case["var_dropped"].as_bool().unwrap(),
        "{name}: dq_var_dropped"
    );
    assert_close(
        r.dq_stat,
        case["dq_stat"].as_f64().unwrap(),
        1e-9,
        &format!("{name}: dq_stat"),
    );
    assert_pvalue(
        r.p_dq,
        case["p_dq"].as_f64().unwrap(),
        &format!("{name}: p_dq"),
    );
}

#[test]
fn golden_hit_cases_match_numpy_and_statsmodels() {
    let fx = fixture();
    for case in fx["hit_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let hits = f64s(&case["hits"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let lags = case["dq_lags"].as_u64().unwrap() as usize;
        let r = var_backtest_hits(&hits, None, alpha, lags).unwrap();
        check_case(case, &r, name);
    }
}

#[test]
fn golden_return_cases_pin_the_sign_convention() {
    let fx = fixture();
    for case in fx["return_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let returns = f64s(&case["returns"]);
        let var = f64s(&case["var_forecasts"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let lags = case["dq_lags"].as_u64().unwrap() as usize;
        // The generator computed hits as returns < var (return scale, VaR
        // negative); n_violations matching pins the convention end to end.
        let r = var_backtest(&returns, &var, alpha, lags).unwrap();
        check_case(case, &r, name);
        // And the hits entry point with the VaR forecasts must agree
        // exactly with the returns entry point.
        let hits: Vec<f64> = returns
            .iter()
            .zip(var.iter())
            .map(|(&x, &q)| if x < q { 1.0 } else { 0.0 })
            .collect();
        let r2 = var_backtest_hits(&hits, Some(&var), alpha, lags).unwrap();
        assert_eq!(r, r2, "{name}: hits entry point must match returns");

        // The constant-VaR case exercises the documented rank rule: the
        // VaR column is collinear with the intercept, gets dropped, and
        // the verdict explains the df reduction.
        if name == "iid_true_var_a025" {
            assert!(r.dq_var_dropped, "{name}: constant VaR must be dropped");
            assert!(!r.dq_includes_var);
            assert_eq!(r.dq_df, r.dq_lags + 1);
            assert!(
                r.verdict.contains("constant"),
                "{name}: verdict explains the drop: {}",
                r.verdict
            );
        }
    }
}

#[test]
fn golden_hand_cases() {
    let fx = fixture();
    for case in fx["hand_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let hits = f64s(&case["hits"]);
        let alpha = case["alpha"].as_f64().unwrap();
        let lags = case["dq_lags"].as_u64().unwrap() as usize;
        let r = var_backtest_hits(&hits, None, alpha, lags).unwrap();
        check_case(case, &r, name);
    }
}

#[test]
fn hand_computed_kupiec_n250_x5() {
    // n = 250, 5 violations at alpha = 0.05 (12.5 expected, pi_hat = 0.02).
    // By hand:
    //   ll_null = 245 ln(0.95) + 5 ln(0.05) = -12.5668571 - 14.9786614
    //   ll_alt  = 245 ln(0.98) + 5 ln(0.02) =  -4.9496633 - 19.5601150
    //   LR_uc   = -2 (ll_null - ll_alt)     =   6.0714803
    // SciPy cross-check: LR_uc = 6.07148034557369, p = 0.013738177260985.
    // Rejection at 5% *for too few violations*: the VaR is too conservative.
    let mut hits = vec![0.0; 250];
    for t in [40usize, 90, 140, 190, 240] {
        hits[t] = 1.0;
    }
    let r = var_backtest_hits(&hits, None, 0.05, 4).unwrap();
    assert_close(r.lr_uc, 6.07148034557369, 1e-12, "hand LR_uc");
    assert_close(r.p_uc, 0.013738177260985241, 1e-9, "hand p_uc");
    // No two violations are consecutive: the n11 = 0 continuity cell.
    assert_eq!((r.n00, r.n01, r.n10, r.n11), (239, 5, 5, 0));
    assert_eq!(r.pi11, 0.0);
    assert!(r.lr_ind > 0.0 && r.lr_ind < 1.0, "tiny LR_ind, not NaN");
    assert!(
        r.verdict.contains("too conservative"),
        "verdict teaches the rejection direction: {}",
        r.verdict
    );
    assert!(
        !r.verdict.contains("Caution"),
        "the small-sample caution appears only when expected violations \
         < 5; here 12.5 were expected — got: {}",
        r.verdict
    );
}

#[test]
fn published_jorion_jpm_1998_kupiec() {
    // Jorion, "Value at Risk" (3rd ed.), ch. 6: J.P. Morgan disclosed 20
    // 95%-VaR breaches in the 252 trading days of 1998. LR_uc = 3.91 as
    // printed (exact value 3.9125508275532184, SciPy p = 0.0479268):
    // a borderline rejection against the chi-squared(1) 3.84 critical
    // value. Only the (T, N, alpha) triple is published; the placement
    // below is arbitrary and LR_uc depends only on the count.
    let hits: Vec<f64> = (0..252)
        .map(|t| if t % 12 == 6 && t < 240 { 1.0 } else { 0.0 })
        .collect();
    assert_eq!(hits.iter().filter(|&&h| h == 1.0).count(), 20);
    let r = var_backtest_hits(&hits, None, 0.05, 4).unwrap();
    assert_close(r.lr_uc, 3.9125508275532184, 1e-12, "Jorion LR_uc");
    assert_close(r.p_uc, 0.04792680100008753, 1e-9, "Jorion p_uc");
    assert!(r.p_uc < 0.05, "borderline rejection, as published");
    assert!(r.verdict.contains("Reject unconditional coverage"));
    assert!(r.verdict.contains("too aggressive"));
}

// ---------------------------------------------------------------- MC rng

/// SplitMix64: deterministic, well-mixed uniforms for the Monte Carlo
/// property tests (statistical quality is ample for rejection counting).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn bernoulli_hits(n: usize, alpha: f64, rng: &mut SplitMix64) -> Vec<f64> {
    (0..n)
        .map(|_| if rng.uniform() < alpha { 1.0 } else { 0.0 })
        .collect()
}

/// First-order Markov hits with P(1|0) = pi01, P(1|1) = pi11; the
/// stationary violation rate is pi01 / (pi01 + 1 - pi11).
fn markov_hits(n: usize, pi01: f64, pi11: f64, rng: &mut SplitMix64) -> Vec<f64> {
    let mut state = 0u8;
    (0..n)
        .map(|_| {
            let p = if state == 1 { pi11 } else { pi01 };
            state = u8::from(rng.uniform() < p);
            f64::from(state)
        })
        .collect()
}

/// Rejection rates at the 5% test size over `reps` seeded replications.
/// Degenerate draws (no/all violations, singular DQ design) are skipped
/// and counted; they must stay rare or the assertion is meaningless.
fn rejection_rates<F: FnMut(&mut SplitMix64) -> Vec<f64>>(
    mut gen: F,
    reps: usize,
    alpha: f64,
    lags: usize,
    seed: u64,
) -> ([f64; 4], usize) {
    let mut rng = SplitMix64(seed);
    let mut rej = [0usize; 4];
    let mut used = 0usize;
    for _ in 0..reps {
        let hits = gen(&mut rng);
        match var_backtest_hits(&hits, None, alpha, lags) {
            Ok(r) => {
                used += 1;
                for (slot, p) in rej.iter_mut().zip([r.p_uc, r.p_ind, r.p_cc, r.p_dq]) {
                    if p < 0.05 {
                        *slot += 1;
                    }
                }
            }
            Err(
                ForecastError::NoViolations { .. }
                | ForecastError::AllViolations { .. }
                | ForecastError::SingularDqDesign { .. },
            ) => {}
            Err(e) => panic!("unexpected error in MC rep: {e}"),
        }
    }
    let u = used as f64;
    (
        [
            rej[0] as f64 / u,
            rej[1] as f64 / u,
            rej[2] as f64 / u,
            rej[3] as f64 / u,
        ],
        used,
    )
}

#[test]
fn mc_size_iid_hits_rejects_near_nominal() {
    // Under iid Bernoulli(0.05) hits every test's null is true. NumPy
    // calibration (2000 reps, n = 500, L = 4, identical formulas):
    // uc 4.9%, ind 3.3% (discrete-cell conservatism), cc 4.0%, dq 5.7%.
    // Bands = calibrated value +/- ~3 MC standard errors at 500 reps.
    let n = 500;
    let (rates, used) = rejection_rates(
        |rng| bernoulli_hits(n, 0.05, rng),
        500,
        0.05,
        4,
        0x5EED_0001,
    );
    assert!(used >= 495, "degenerate reps must be rare, used {used}");
    let [uc, ind, cc, dq] = rates;
    assert!(
        (0.020..=0.080).contains(&uc),
        "Kupiec size {uc} outside [0.02, 0.08]"
    );
    assert!(
        (0.010..=0.065).contains(&ind),
        "LR_ind size {ind} outside [0.01, 0.065]"
    );
    assert!(
        (0.015..=0.070).contains(&cc),
        "LR_cc size {cc} outside [0.015, 0.07]"
    );
    assert!(
        (0.025..=0.090).contains(&dq),
        "DQ size {dq} outside [0.025, 0.09]"
    );
}

#[test]
fn mc_power_clustered_hits_separates_the_tests() {
    // Markov-dependent hits with pi11 = 0.4 and the stationary rate held
    // at exactly alpha = 0.05 (pi01 = 0.05 * 0.6 / 0.95): the violation
    // RATE is right, the violations CLUSTER. This is the whole point of
    // the battery: LR_ind and DQ must reject overwhelmingly while Kupiec,
    // which only sees the count, stays far below (it drifts modestly above
    // its nominal size because dependence inflates the count variance —
    // NumPy calibration at 2000 reps: uc 18.8%, ind 98.4%, dq 98.1%).
    let n = 500;
    let pi11 = 0.4;
    let pi01 = 0.05 * (1.0 - pi11) / (1.0 - 0.05);
    let (rates, used) = rejection_rates(
        |rng| markov_hits(n, pi01, pi11, rng),
        500,
        0.05,
        4,
        0x5EED_0002,
    );
    assert!(used >= 495, "degenerate reps must be rare, used {used}");
    let [uc, ind, cc, dq] = rates;
    assert!(ind > 0.90, "LR_ind power {ind} <= 0.90 on clustered hits");
    assert!(dq > 0.90, "DQ power {dq} <= 0.90 on clustered hits");
    assert!(cc > 0.90, "LR_cc power {cc} <= 0.90 on clustered hits");
    assert!(
        uc < 0.35,
        "Kupiec {uc} should stay far below the dependence tests when the \
         unconditional rate is correct"
    );
    assert!(
        ind - uc > 0.50 && dq - uc > 0.50,
        "the separation is the point: ind {ind}, dq {dq}, uc {uc}"
    );
}

// ------------------------------------------------------------ guardrails

#[test]
fn guardrails_teach() {
    let good: Vec<f64> = (0..100)
        .map(|t| if t % 25 == 7 { 1.0 } else { 0.0 })
        .collect();

    // alpha outside (0, 1).
    for bad_alpha in [0.0, 1.0, -0.2, 1.7] {
        assert!(matches!(
            var_backtest_hits(&good, None, bad_alpha, 4).unwrap_err(),
            ForecastError::InvalidAlpha { .. }
        ));
    }

    // Length mismatch between returns and VaR forecasts.
    let r: Vec<f64> = vec![0.1; 50];
    let q: Vec<f64> = vec![-1.0; 49];
    assert!(matches!(
        var_backtest(&r, &q, 0.05, 4).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));
    // ... and between hits and VaR forecasts.
    assert!(matches!(
        var_backtest_hits(&good, Some(&q), 0.05, 4).unwrap_err(),
        ForecastError::LengthMismatch { .. }
    ));

    // Non-finite input.
    let mut nan = r.clone();
    nan[3] = f64::NAN;
    assert!(matches!(
        var_backtest(&nan, &vec![-1.0; 50], 0.05, 4).unwrap_err(),
        ForecastError::NonFinite { index: 3, .. }
    ));

    // A hit series must be exactly 0/1 — a raw return series is caught.
    let err = var_backtest_hits(&[0.0, 1.0, 0.3, 0.0], None, 0.05, 1).unwrap_err();
    assert!(matches!(
        err,
        ForecastError::InvalidHitValue { index: 2, .. }
    ));
    let msg = err.to_string();
    assert!(
        msg.contains("return series") && msg.contains("VaR"),
        "teaches the likely mistake: {msg}"
    );

    // Zero violations: a teaching error that reports the continuity-limit
    // LR and the expected count instead of silently degenerating.
    let err = var_backtest_hits(&vec![0.0; 250], None, 0.05, 4).unwrap_err();
    assert!(matches!(err, ForecastError::NoViolations { n: 250, .. }));
    let msg = err.to_string();
    assert!(msg.contains("12.5"), "expected count in message: {msg}");
    // Continuity limit -2 * 250 * ln(0.95) = 25.647 appears in the message.
    assert!(msg.contains("25.647"), "continuity-limit LR: {msg}");
    assert!(msg.contains("too conservative"), "teaches direction: {msg}");

    // All violations: teaches the sign convention.
    let err = var_backtest_hits(&vec![1.0; 250], None, 0.05, 4).unwrap_err();
    assert!(matches!(err, ForecastError::AllViolations { .. }));
    assert!(err.to_string().contains("sign-convention"));

    // dq_lags = 0 or too large for the sample.
    assert!(matches!(
        var_backtest_hits(&good, None, 0.05, 0).unwrap_err(),
        ForecastError::InvalidDqLags { lags: 0, .. }
    ));
    let short: Vec<f64> = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let err = var_backtest_hits(&short, None, 0.05, 4).unwrap_err();
    assert!(matches!(err, ForecastError::InvalidDqLags { .. }));
    assert!(err.to_string().contains("Engle-Manganelli default is 4"));

    // Singular DQ design: one violation cannot identify 4 lag
    // coefficients (the lagged-hit columns are collinear with the
    // constant), and the error says which knob to turn.
    let mut lonely = vec![0.0; 60];
    lonely[59] = 1.0;
    let err = var_backtest_hits(&lonely, None, 0.05, 4).unwrap_err();
    assert!(matches!(err, ForecastError::SingularDqDesign { .. }));
    assert!(err.to_string().contains("reduce dq_lags"));
}

#[test]
fn verdict_reads_like_a_report() {
    // Clustered violations with the rate right: the verdict must name the
    // clustering and keep the coverage claim intact.
    let mut rng = SplitMix64(0xC0FFEE);
    let pi11 = 0.5;
    let pi01 = 0.05 * (1.0 - pi11) / (1.0 - 0.05);
    // Draw until the rate is close to nominal so the UC leg stays quiet.
    let hits = loop {
        let h = markov_hits(2000, pi01, pi11, &mut rng);
        let n1 = h.iter().filter(|&&x| x == 1.0).count();
        if (90..=110).contains(&n1) {
            break h;
        }
    };
    let r = var_backtest_hits(&hits, None, 0.05, 4).unwrap();
    assert!(r.p_ind < 0.05, "clustered draw should reject independence");
    assert!(r.verdict.contains("violations in 2000 observations"));
    assert!(r.verdict.contains("Reject independence"));
    assert!(r.verdict.contains("violations cluster"));

    // The sign-convention tripwire: alpha = 0.05 but almost every
    // observation a "violation".
    let mostly_ones: Vec<f64> = (0..100)
        .map(|t| if t % 10 == 0 { 0.0 } else { 1.0 })
        .collect();
    let r = var_backtest_hits(&mostly_ones, None, 0.05, 2).unwrap();
    assert!(r.verdict.contains("sign-convention"), "{}", r.verdict);

    // Few expected violations: the small-sample caution appears.
    let mut sparse = vec![0.0; 250];
    sparse[10] = 1.0;
    sparse[100] = 1.0;
    sparse[200] = 1.0;
    let r = var_backtest_hits(&sparse, None, 0.01, 2).unwrap();
    assert!(r.verdict.contains("Caution"), "{}", r.verdict);
    assert!(r.verdict.contains("2.5"), "{}", r.verdict);
}
