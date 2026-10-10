//! Audit round 16: size budgets, keyword-named refusals and extreme-scale
//! impact matrices in the structural-identification layer.
//!
//! * A horizon or draw count of `2^31` / `2^40` used to abort the process in
//!   `cholesky_irf` / `Stream::substreams`; the samplers, the robust bounds,
//!   the non-Gaussian SVAR and the proxy AR reduced-form variances now refuse
//!   it as `BayesError::TooLarge` (shown without the "Bayesian foundations
//!   failure" prefix), naming the keyword, before any substream is spawned.
//! * The narrative weight loop is a work budget (`NARRATIVE_WEIGHT_WORK`).
//! * Refusals name the Python keyword (`n_draws`, `max_tries`, `lags`).
//! * `structural_fevd_from_theta` and the impact singularity test work on
//!   power-of-two-scaled values: the statsmodels FEVD golden is reproduced
//!   at any finite scale of `Theta`, bit for bit under a power-of-two
//!   scale, and the Blanchard-Quah impact golden of the historical
//!   decomposition is reproduced at scales whose rank numpy reports full
//!   (`1e155` used to be called singular).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use serde_json::Value;
use tsecon_bayes::{BayesError, MinnesotaNiwPrior, NiwPosterior};
use tsecon_ident::histdecomp::decompose_with_impact;
use tsecon_ident::proxy_ar::{
    check_ar_sets_budget, psi_reduced_form_cov, ArRfBudget, PROXY_AR_DELTA_WORK,
};
use tsecon_ident::structural_fevd::structural_fevd_from_theta;
use tsecon_ident::{
    narrative_weight_draw_work, robust_svar_bounds_default, IdentError, NarrativeRestriction,
    NarrativeRestrictionSet, NarrativeSampler, Sign, SignRestriction, SignRestrictionSet,
    SignSampler, NARRATIVE_WEIGHT_WORK,
};
use tsecon_linalg::faer::Mat;

const BIG: usize = 1 << 31;
const HUGE: usize = 1 << 40;

fn load(name: &str) -> Value {
    let path = format!("{}/../../fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

fn mat(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn scale(m: &Mat<f64>, s: f64) -> Mat<f64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] * s)
}

fn mats(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().unwrap().iter().map(mat).collect()
}

/// A deterministic 3-variable VAR(1) sample (xorshift, no tsecon RNG).
fn toy_data(t: usize) -> Mat<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut data = Mat::<f64>::zeros(t, 3);
    for r in 1..t {
        for c in 0..3 {
            data[(r, c)] = 0.4 * data[(r - 1, c)] + next();
        }
    }
    data
}

fn posterior(data: &Mat<f64>) -> NiwPosterior {
    MinnesotaNiwPrior::new(data.as_ref(), 1, 100.0, 0.2, 1.0, 0.0)
        .unwrap()
        .posterior(data.as_ref())
        .unwrap()
}

fn signs(horizon: usize) -> SignRestrictionSet {
    SignRestrictionSet::new(
        vec![
            SignRestriction::at(0, 0, 0, Sign::Positive),
            SignRestriction::at(1, 0, 0, Sign::Positive),
        ],
        3,
        horizon,
    )
    .unwrap()
}

/// The refusal is a named `TooLarge`, shown without the Bayesian-layer
/// prefix; returns `(arg, value)`.
fn named(e: IdentError) -> (&'static str, usize) {
    let msg = e.to_string();
    assert!(!msg.contains("Bayesian foundations failure"), "{msg}");
    assert!(msg.contains("before anything is allocated"), "{msg}");
    match e {
        IdentError::Bayes(BayesError::TooLarge {
            arg,
            value,
            max: Some(_),
            ..
        }) => (arg, value),
        other => panic!("expected a named TooLarge with a quoted maximum, got {other:?}"),
    }
}

#[test]
fn the_samplers_name_horizon_or_n_draws_before_spawning_streams() {
    let data = toy_data(200);
    let post = posterior(&data);
    for d in [BIG, HUGE] {
        let e = SignSampler::new(4, d, 100)
            .unwrap()
            .run(&post, &signs(4), 0)
            .unwrap_err();
        assert_eq!(named(e), ("n_draws", d));
        let e = NarrativeSampler::new(4, d, 100, 200)
            .unwrap()
            .run(&post, data.as_ref(), Some(&signs(4)), None, 0)
            .unwrap_err();
        assert_eq!(named(e), ("n_draws", d));
        let e = robust_svar_bounds_default(&post, &signs(4), 4, d, 0, 0.1).unwrap_err();
        assert_eq!(named(e), ("n_draws", d));
    }
    for h in [BIG, HUGE] {
        let e = SignSampler::new(h, 30, 100)
            .unwrap()
            .run(&post, &signs(h), 0)
            .unwrap_err();
        assert_eq!(named(e), ("horizon", h));
        let e = robust_svar_bounds_default(&post, &signs(h), h, 30, 0, 0.1).unwrap_err();
        assert_eq!(named(e), ("horizon", h));
    }
}

#[test]
fn the_narrative_weight_loop_is_a_named_work_budget() {
    let data = toy_data(200);
    let post = posterior(&data);
    let nr = NarrativeRestrictionSet::new(
        vec![NarrativeRestriction::ShockSign {
            shock: 0,
            period: 10,
            sign: Sign::Positive,
        }],
        3,
        199,
    )
    .unwrap();
    for k in [BIG, HUGE] {
        let e = NarrativeSampler::new(4, 20, 200, k)
            .unwrap()
            .run(&post, data.as_ref(), Some(&signs(4)), Some(&nr), 0)
            .unwrap_err();
        assert!(
            e.to_string().contains("2^35 multiply-add work budget"),
            "{e}"
        );
        assert_eq!(named(e), ("n_weight_draws", k));
    }
    // The documented count: the default 500 x 200 on this sample is far
    // inside the budget.
    let unit = narrative_weight_draw_work(3, 199, 4);
    assert_eq!(unit, 199 * 9 + 7 * 27);
    assert!(500 * 200 * unit < NARRATIVE_WEIGHT_WORK / 100);
}

#[test]
fn zero_counts_name_the_python_keywords() {
    for (e, want) in [
        (
            SignSampler::new(4, 0, 10).unwrap_err(),
            "n_draws must be at least 1",
        ),
        (
            SignSampler::new(4, 10, 0).unwrap_err(),
            "max_tries must be at least 1",
        ),
        (
            NarrativeSampler::new(4, 0, 10, 10).unwrap_err(),
            "n_draws must be at least 1",
        ),
    ] {
        let msg = e.to_string();
        assert!(
            msg.contains(want) && !msg.contains("n_posterior_draws"),
            "{msg}"
        );
    }
}

/// `(arg, value, max)` of a named `TooLarge` refusal.
fn quoted(e: IdentError) -> (&'static str, usize, Option<usize>) {
    match e {
        IdentError::Bayes(BayesError::TooLarge {
            arg, value, max, ..
        }) => (arg, value, max),
        other => panic!("expected a named TooLarge, got {other:?}"),
    }
}

#[test]
fn a_quoted_maximum_passes_every_budget_of_the_call() {
    // proxy_ar_sets, delta method, n = 3, lags = 2: the memory budget alone
    // admits horizons in the millions, but the Jacobian work
    // H (H + 1) / 2 * 72 + 1350 H (np = 6, d = 18: per pair
    // np * n (n + 1) = 72, per horizon n d^2 + n^2 d + np^3 = 1350) caps it
    // far lower; the quote is that closed-form boundary.
    let work = |h: u128| h * (h + 1) / 2 * 72 + 1350 * h;
    let mut h_star = 0u128;
    while work(h_star + 1) <= PROXY_AR_DELTA_WORK {
        h_star += 1;
    }
    let h_star = h_star as usize;
    for h in [BIG, HUGE, h_star + 1] {
        let e = check_ar_sets_budget(3, 2, h, ArRfBudget::Delta).unwrap_err();
        assert!(
            e.to_string().contains("2^32 multiply-add work budget"),
            "{e}"
        );
        assert_eq!(quoted(e), ("horizon", h, Some(h_star)));
    }
    check_ar_sets_budget(3, 2, h_star, ArRfBudget::Delta).unwrap();
    // The second-order simulation: the quoted rf_draws is even (the
    // simulation works in antithetic pairs) and passes.
    let e = check_ar_sets_budget(3, 2, 12, ArRfBudget::Simulation { draws: HUGE }).unwrap_err();
    let (arg, _, max) = quoted(e);
    let max = max.unwrap();
    assert_eq!((arg, max % 2), ("rf_draws", 0));
    check_ar_sets_budget(3, 2, 12, ArRfBudget::Simulation { draws: max }).unwrap();
    check_ar_sets_budget(3, 2, 12, ArRfBudget::Simulation { draws: max + 2 }).unwrap_err();
    // And a quoted horizon passes with the simulation's rows counted.
    let e = check_ar_sets_budget(3, 2, HUGE, ArRfBudget::Simulation { draws: 256 }).unwrap_err();
    let (_, _, max) = quoted(e);
    check_ar_sets_budget(3, 2, max.unwrap(), ArRfBudget::Simulation { draws: 256 }).unwrap();

    // The narrative sampler with narrative restrictions: an absurd n_draws
    // is quoted at the work boundary 2^35 / (n_weight_draws * unit), which
    // is below the memory boundary, never at a value the work budget then
    // refuses.
    let data = toy_data(200);
    let post = posterior(&data);
    let nr = NarrativeRestrictionSet::new(
        vec![NarrativeRestriction::ShockSign {
            shock: 0,
            period: 10,
            sign: Sign::Positive,
        }],
        3,
        199,
    )
    .unwrap();
    let unit = narrative_weight_draw_work(3, 199, 4);
    let work_max = (NARRATIVE_WEIGHT_WORK / (200 * unit)) as usize;
    let e = NarrativeSampler::new(4, HUGE, 100, 200)
        .unwrap()
        .run(&post, data.as_ref(), Some(&signs(4)), Some(&nr), 0)
        .unwrap_err();
    assert!(
        e.to_string().contains("2^35 multiply-add work budget"),
        "{e}"
    );
    assert_eq!(quoted(e), ("n_draws", HUGE, Some(work_max)));
    // Sized for a historical decomposition, each draw also keeps its T_eff
    // contribution matrices, so the memory boundary drops below the plain
    // one and is quoted as such.
    let plain = NarrativeSampler::new(4, HUGE, 100, 200)
        .unwrap()
        .run(&post, data.as_ref(), Some(&signs(4)), None, 0)
        .unwrap_err();
    let hd = NarrativeSampler::new(4, HUGE, 100, 200)
        .unwrap()
        .sized_for_hd()
        .run(&post, data.as_ref(), Some(&signs(4)), None, 0)
        .unwrap_err();
    let (_, _, plain_max) = quoted(plain);
    let (arg, _, hd_max) = quoted(hd);
    let hd_max = hd_max.unwrap();
    assert!(
        arg == "n_draws" && hd_max < plain_max.unwrap() / 10,
        "{hd_max}"
    );
    let e = NarrativeSampler::new(4, hd_max + 1, 100, 200)
        .unwrap()
        .sized_for_hd()
        .run(&post, data.as_ref(), Some(&signs(4)), None, 0)
        .unwrap_err();
    assert_eq!(quoted(e), ("n_draws", hd_max + 1, Some(hd_max)));
}

#[test]
fn the_delta_method_horizon_is_a_named_work_budget() {
    let n = 3;
    let coefs = vec![Mat::from_fn(n, n, |i, j| if i == j { 0.5 } else { 0.0 })];
    let cov_alpha = Mat::<f64>::identity(n * n, n * n);
    let psi: Vec<Mat<f64>> = (0..=200_000).map(|_| Mat::<f64>::identity(n, n)).collect();
    let e =
        psi_reduced_form_cov(&psi, &coefs, cov_alpha.as_ref(), &[1.0, 0.5, 0.2], 100).unwrap_err();
    assert!(
        e.to_string().contains("2^32 multiply-add work budget"),
        "{e}"
    );
    assert_eq!(named(e), ("horizon", 200_000));
}

#[test]
fn fevd_shares_reproduce_the_statsmodels_golden_at_any_scale() {
    let fx = load("structural_fevd.json");
    let theta = mats(&fx["theta_chol"]);
    let want = mats(&fx["fevd_statsmodels"]);
    let base = structural_fevd_from_theta(&theta).unwrap();
    // A power-of-two scale is exact: the shares are bit-identical.
    for k in [-1000i32, -500, 500, 1000] {
        let s = 2f64.powi(k / 2) * 2f64.powi(k - k / 2);
        let scaled: Vec<Mat<f64>> = theta.iter().map(|m| scale(m, s)).collect();
        let got = structural_fevd_from_theta(&scaled).unwrap();
        for (g, b) in got.iter().zip(&base) {
            assert_eq!(g, b, "2^{k}");
        }
    }
    // Any finite scale reproduces the statsmodels shares (they used to be
    // NaN past ~1e154 and wrong, then "degenerate", below ~1e-154).
    for s in [1e-300, 1e-200, 1e-160, 1e160, 1e200, 1e300] {
        let scaled: Vec<Mat<f64>> = theta.iter().map(|m| scale(m, s)).collect();
        let got = structural_fevd_from_theta(&scaled).unwrap();
        for (h, (g, w)) in got.iter().zip(&want).enumerate() {
            for i in 0..g.nrows() {
                for j in 0..g.ncols() {
                    let d = (g[(i, j)] - w[(i, j)]).abs();
                    assert!(d < 1e-10, "scale {s:e} h={h} ({i},{j}): {d:e}");
                }
            }
        }
    }
    // A row of exact zeros is still degenerate.
    let mut zero_row = theta.clone();
    for m in &mut zero_row {
        for j in 0..m.ncols() {
            m[(0, j)] = 0.0;
        }
    }
    assert!(matches!(
        structural_fevd_from_theta(&zero_row),
        Err(IdentError::InvalidArgument { .. })
    ));
}

#[test]
fn hd_impact_golden_holds_where_numpy_reports_full_rank() {
    let fx = load("historical_decomposition_impact.json");
    let case = &fx["bq_var8"];
    let p = case["p"].as_u64().unwrap() as usize;
    let t_eff = case["t_eff"].as_u64().unwrap() as usize;
    let data = mat(&case["data"]);
    let b = mat(&case["b"]);
    let a0 = mat(&case["impact"]);
    let hd_want = mats(&case["hd"]);
    let cov_want = mat(&case["shock_covariance"]);
    // 1e155 used to be refused as "singular" (faer's SVD overflowed).
    for s in [1e-100, 1e100, 1e155] {
        let scaled = scale(&a0, s);
        let hd = decompose_with_impact(data.as_ref(), b.as_ref(), scaled.as_ref(), p, t_eff - 1)
            .unwrap_or_else(|e| panic!("scale {s:e}: {e}"));
        for (t, w) in hd_want.iter().enumerate() {
            for i in 0..w.nrows() {
                for j in 0..w.ncols() {
                    let (g, w) = (hd.hd()[t][(i, j)], w[(i, j)]);
                    assert!(
                        (g - w).abs() <= 1e-10 + 1e-8 * w.abs(),
                        "scale {s:e} hd[{t}]"
                    );
                }
            }
        }
        if s < 1e150 {
            let cov = hd.shock_covariance();
            for i in 0..cov.nrows() {
                for j in 0..cov.ncols() {
                    let (g, w) = (cov[(i, j)] * s * s, cov_want[(i, j)]);
                    assert!((g - w).abs() <= 1e-10 + 1e-8 * w.abs(), "scale {s:e} cov");
                }
            }
        }
    }
    // A rank-deficient impact is singular at every scale.
    let n = a0.nrows();
    for s in [1.0, 1e200, 1e-200] {
        let sing = Mat::from_fn(n, n, |_, _| s);
        let e = decompose_with_impact(data.as_ref(), b.as_ref(), sing.as_ref(), p, t_eff - 1)
            .unwrap_err();
        assert!(e.to_string().contains("impact is singular"), "{e}");
    }
}
