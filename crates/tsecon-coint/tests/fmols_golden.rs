//! Golden-value tests for the single-equation cointegrating regressions
//! (`fmols`, `dols`, `ccr`) against `fixtures/fmols.json`, generated from
//! `arch.unitroot.cointegration.{FullyModifiedOLS, DynamicOLS,
//! CanonicalCointegratingReg}` (arch 8.0.0) by
//! `fixtures/generate_fmols_fixtures.py`.
//!
//! Every case pins the coefficients, the parameter covariance (hence the
//! standard errors and t-statistics), the residuals, `R^2`, the bandwidth
//! actually used and — for DOLS — the selected leads and lags, across
//! three seeded systems (`k_x` = 1, 3 and a `T = 60` small sample), all
//! four deterministic specifications, the three kernels, explicit /
//! automatic / forced-integer bandwidths, `df_adjust`, `diff`, `x_trend`,
//! both DOLS covariance types, the three information criteria, the
//! `common` restriction and the search caps. Two blocks are
//! documented-formula rather than third-party goldens and the fixture
//! records them as such: the Andrews (1991) bandwidth rule (the value is
//! the closed form; the estimates are arch's at that bandwidth) and CCR
//! under `df_adjust` (arch 8.0 scales only `omega_11`; the documented
//! `T/(T - k)` scaling of the conditional long-run variance is pinned,
//! and arch's raw value is asserted to differ). The kernel long-run
//! covariances and arch's automatic bandwidth are pinned separately
//! against `arch.covariance.kernel`.

mod common;

use serde_json::Value;
use tsecon_coint::tsecon_hac::{self, Kernel};
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{
    automatic_bandwidth, ccr, dols, fmols, long_run_covariance, parse_kernel, BandwidthRule,
    CointError, CointRegOptions, CointRegResult, CointTrend, DolsCovType, DolsIc, DolsOptions,
};

use common::{as_mat, as_vec, load_fixture, num};

const TOL: f64 = 1e-10;

// ------------------------------------------------------------- helpers

fn trend_of(code: &str) -> CointTrend {
    CointTrend::parse(code).unwrap_or_else(|| panic!("unknown trend {code:?}"))
}

fn kernel_of(code: &str) -> Kernel {
    parse_kernel(code).unwrap_or_else(|| panic!("unknown kernel {code:?}"))
}

fn rule_of(v: &Value) -> BandwidthRule {
    match v.as_str() {
        None => BandwidthRule::NeweyWest,
        Some(code) => BandwidthRule::parse(code).unwrap_or_else(|| panic!("unknown rule {code:?}")),
    }
}

fn opt_f64(v: &Value) -> Option<f64> {
    v.as_f64()
}

fn opt_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|u| u as usize)
}

fn system(fx: &Value, name: &str) -> (Vec<f64>, Mat<f64>) {
    let s = &fx["systems"][name];
    (as_vec(&s["y"]), as_mat(&s["x"]))
}

/// `|a - e| <= tol * max(1, |e|)`.
fn close(actual: f64, expected: f64, tol: f64, ctx: &str) {
    let scale = expected.abs().max(1.0);
    let rel = (actual - expected).abs() / scale;
    assert!(
        rel <= tol,
        "{ctx}: actual {actual}, expected {expected}, rel {rel:e} > {tol:e}"
    );
}

fn close_vec(actual: &[f64], expected: &Value, tol: f64, ctx: &str) {
    let e = as_vec(expected);
    assert_eq!(actual.len(), e.len(), "{ctx}: length");
    for (i, (a, e)) in actual.iter().zip(e.iter()).enumerate() {
        close(*a, *e, tol, &format!("{ctx}[{i}]"));
    }
}

fn close_mat(actual: &[Vec<f64>], expected: &Value, tol: f64, ctx: &str) {
    let rows = expected.as_array().expect("rows");
    assert_eq!(actual.len(), rows.len(), "{ctx}: rows");
    for (i, row) in rows.iter().enumerate() {
        close_vec(&actual[i], row, tol, &format!("{ctx}[{i}]"));
    }
}

/// p-values: relative, with an absolute floor for the saturated tail.
fn close_p(actual: &[f64], expected: &Value, ctx: &str) {
    let e = as_vec(expected);
    for (i, (a, e)) in actual.iter().zip(e.iter()).enumerate() {
        let tol = 1e-14 + 1e-9 * e.abs();
        assert!((a - e).abs() <= tol, "{ctx}[{i}]: actual {a}, expected {e}");
    }
}

fn coint_opts(case: &Value) -> CointRegOptions {
    CointRegOptions {
        trend: trend_of(case["trend"].as_str().expect("trend")),
        x_trend: case["x_trend"].as_str().map(trend_of),
        kernel: kernel_of(case["kernel"].as_str().expect("kernel")),
        bandwidth: opt_f64(&case["bandwidth_arg"]),
        bandwidth_rule: rule_of(&case["bandwidth_rule"]),
        force_int: case["force_int"].as_bool().expect("force_int"),
        diff: case["diff"].as_bool().expect("diff"),
        df_adjust: case["df_adjust"].as_bool().expect("df_adjust"),
    }
}

fn check_coint_reg(r: &CointRegResult, case: &Value, ctx: &str) {
    let names: Vec<&str> = case["param_names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|v| v.as_str().expect("name"))
        .collect();
    assert_eq!(r.param_names, names, "{ctx}: param_names");
    assert_eq!(
        r.n_x,
        case["k_x"].as_u64().expect("k_x") as usize,
        "{ctx}: n_x"
    );
    close(
        r.bandwidth,
        num(&case["bandwidth"]),
        TOL,
        &format!("{ctx}: bandwidth"),
    );
    close_vec(&r.params, &case["params"], TOL, &format!("{ctx}: params"));
    close_vec(&r.se, &case["se"], TOL, &format!("{ctx}: se"));
    close_vec(
        &r.tvalues,
        &case["tvalues"],
        TOL,
        &format!("{ctx}: tvalues"),
    );
    close_p(&r.pvalues, &case["pvalues"], &format!("{ctx}: pvalues"));
    close_mat(&r.cov, &case["cov"], TOL, &format!("{ctx}: cov"));
    close_vec(&r.resid, &case["resid"], TOL, &format!("{ctx}: resid"));
    close(
        r.rsquared,
        num(&case["rsquared"]),
        TOL,
        &format!("{ctx}: rsquared"),
    );
    close(
        r.rsquared_adj,
        num(&case["rsquared_adj"]),
        TOL,
        &format!("{ctx}: rsquared_adj"),
    );
    close(
        r.long_run_variance,
        num(&case["long_run_variance"]),
        TOL,
        &format!("{ctx}: long_run_variance"),
    );
    close_vec(
        &r.ols_params,
        &case["ols_params"],
        TOL,
        &format!("{ctx}: ols_params"),
    );
    // Self-consistency of the reported inference.
    for j in 0..r.params.len() {
        close(
            r.se[j],
            r.cov[j][j].sqrt(),
            1e-14,
            &format!("{ctx}: se = sqrt(diag cov)"),
        );
        close(
            r.tvalues[j],
            r.params[j] / r.se[j],
            1e-14,
            &format!("{ctx}: t = b/se"),
        );
    }
    match case["bandwidth_arg"].as_f64() {
        Some(_) => assert!(
            r.bandwidth_rule.is_none(),
            "{ctx}: explicit bandwidth has no rule"
        ),
        None => assert_eq!(
            r.bandwidth_rule.map(|b| b.code()),
            Some(case["bandwidth_rule"].as_str().unwrap_or("newey-west")),
            "{ctx}: bandwidth_rule"
        ),
    }
}

// --------------------------------------------------------------- golden

#[test]
fn golden_fmols_matches_arch() {
    let fx = load_fixture("fmols.json");
    let cases = fx["cases"].as_array().expect("cases");
    let mut n = 0;
    for case in cases.iter().filter(|c| c["estimator"] == "fmols") {
        let name = case["system"].as_str().expect("system");
        let (y, x) = system(&fx, name);
        let opts = coint_opts(case);
        let ctx = format!(
            "fmols/{name}/{}/{}/bw={}/rule={}/fi={}/diff={}/df={}/x_trend={}",
            case["trend"],
            case["kernel"],
            case["bandwidth_arg"],
            case["bandwidth_rule"],
            case["force_int"],
            case["diff"],
            case["df_adjust"],
            case["x_trend"]
        );
        let r = fmols(&y, x.as_ref(), &opts).unwrap_or_else(|e| panic!("{ctx}: {e}"));
        assert_eq!(r.nobs, y.len());
        check_coint_reg(&r, case, &ctx);
        n += 1;
    }
    assert!(n >= 50, "fixture lost FM-OLS cases: {n}");
}

#[test]
fn golden_ccr_matches_arch() {
    let fx = load_fixture("fmols.json");
    let cases = fx["cases"].as_array().expect("cases");
    let mut n = 0;
    let mut n_deviation = 0;
    for case in cases.iter().filter(|c| c["estimator"] == "ccr") {
        let name = case["system"].as_str().expect("system");
        let (y, x) = system(&fx, name);
        let opts = coint_opts(case);
        let ctx = format!(
            "ccr/{name}/{}/{}/bw={}/rule={}/fi={}/diff={}/df={}/x_trend={}",
            case["trend"],
            case["kernel"],
            case["bandwidth_arg"],
            case["bandwidth_rule"],
            case["force_int"],
            case["diff"],
            case["df_adjust"],
            case["x_trend"]
        );
        let r = ccr(&y, x.as_ref(), &opts).unwrap_or_else(|e| panic!("{ctx}: {e}"));
        check_coint_reg(&r, case, &ctx);
        if let Some(arch_cov_00) = case["arch_cov_00"].as_f64() {
            // The documented df_adjust deviation: arch's raw value differs
            // from the pinned (documented) one, and the pinned one is
            // exactly T/(T - k) times the unadjusted covariance.
            n_deviation += 1;
            let m = (y.len() - 1) as f64;
            let k = r.params.len() as f64;
            let unadj = ccr(
                &y,
                x.as_ref(),
                &CointRegOptions {
                    df_adjust: false,
                    ..opts.clone()
                },
            )
            .expect("unadjusted");
            close(
                r.cov[0][0],
                m / (m - k) * unadj.cov[0][0],
                1e-12,
                &format!("{ctx}: documented df scaling"),
            );
            assert!(
                (r.cov[0][0] - arch_cov_00).abs() > 1e-9 * r.cov[0][0],
                "{ctx}: arch's precedence slip no longer differs"
            );
        }
        n += 1;
    }
    assert!(n >= 50, "fixture lost CCR cases: {n}");
    assert!(
        n_deviation >= 2,
        "fixture lost the df_adjust deviation cases"
    );
}

#[test]
fn golden_dols_matches_arch() {
    let fx = load_fixture("fmols.json");
    let cases = fx["cases"].as_array().expect("cases");
    let mut n = 0;
    for case in cases.iter().filter(|c| c["estimator"] == "dols") {
        let name = case["system"].as_str().expect("system");
        let (y, x) = system(&fx, name);
        let opts = DolsOptions {
            trend: trend_of(case["trend"].as_str().expect("trend")),
            lags: opt_usize(&case["lags_arg"]),
            leads: opt_usize(&case["leads_arg"]),
            common: case["common"].as_bool().expect("common"),
            max_lag: opt_usize(&case["max_lag_arg"]),
            max_lead: opt_usize(&case["max_lead_arg"]),
            ic: DolsIc::parse(case["ic"].as_str().expect("ic")).expect("ic"),
            cov_type: DolsCovType::parse(case["cov_type"].as_str().expect("cov_type"))
                .expect("cov_type"),
            kernel: kernel_of(case["kernel"].as_str().expect("kernel")),
            bandwidth: opt_f64(&case["bandwidth_arg"]),
            bandwidth_rule: rule_of(&case["bandwidth_rule"]),
            force_int: case["force_int"].as_bool().expect("force_int"),
            df_adjust: case["df_adjust"].as_bool().expect("df_adjust"),
        };
        let ctx = format!(
            "dols/{name}/{}/lags={}/leads={}/common={}/max={}/{}/ic={}/{}/{}/bw={}/rule={}/fi={}/df={}",
            case["trend"], case["lags_arg"], case["leads_arg"], case["common"],
            case["max_lag_arg"], case["max_lead_arg"], case["ic"], case["cov_type"],
            case["kernel"], case["bandwidth_arg"], case["bandwidth_rule"], case["force_int"],
            case["df_adjust"]
        );
        let r = dols(&y, x.as_ref(), &opts).unwrap_or_else(|e| panic!("{ctx}: {e}"));
        assert_eq!(
            r.lags,
            case["lags"].as_u64().expect("lags") as usize,
            "{ctx}: lags"
        );
        assert_eq!(
            r.leads,
            case["leads"].as_u64().expect("leads") as usize,
            "{ctx}: leads"
        );
        assert_eq!(
            r.nobs,
            case["nobs"].as_u64().expect("nobs") as usize,
            "{ctx}: nobs"
        );
        assert_eq!(r.n_total, y.len());
        let names: Vec<&str> = case["param_names"]
            .as_array()
            .expect("names")
            .iter()
            .map(|v| v.as_str().expect("name"))
            .collect();
        assert_eq!(r.param_names, names, "{ctx}: param_names");
        assert_eq!(
            r.full_params.len(),
            case["full_params"].as_array().expect("fp").len()
        );
        assert_eq!(r.full_param_names.len(), r.full_params.len());
        close(
            r.bandwidth,
            num(&case["bandwidth"]),
            TOL,
            &format!("{ctx}: bandwidth"),
        );
        close_vec(&r.params, &case["params"], TOL, &format!("{ctx}: params"));
        close_vec(&r.se, &case["se"], TOL, &format!("{ctx}: se"));
        close_vec(
            &r.tvalues,
            &case["tvalues"],
            TOL,
            &format!("{ctx}: tvalues"),
        );
        close_p(&r.pvalues, &case["pvalues"], &format!("{ctx}: pvalues"));
        close_mat(&r.cov, &case["cov"], TOL, &format!("{ctx}: cov"));
        close_vec(
            &r.full_params,
            &case["full_params"],
            TOL,
            &format!("{ctx}: full_params"),
        );
        close_vec(
            &r.full_se,
            &case["full_se"],
            TOL,
            &format!("{ctx}: full_se"),
        );
        close_mat(
            &r.full_cov,
            &case["full_cov"],
            TOL,
            &format!("{ctx}: full_cov"),
        );
        close_vec(&r.resid, &case["resid"], TOL, &format!("{ctx}: resid"));
        close(
            r.rsquared,
            num(&case["rsquared"]),
            TOL,
            &format!("{ctx}: rsquared"),
        );
        close(
            r.rsquared_adj,
            num(&case["rsquared_adj"]),
            TOL,
            &format!("{ctx}: rsquared_adj"),
        );
        close(
            r.long_run_variance,
            num(&case["long_run_variance"]),
            TOL,
            &format!("{ctx}: long_run_variance"),
        );
        close_vec(
            &r.ols_params,
            &case["ols_params"],
            TOL,
            &format!("{ctx}: ols_params"),
        );
        assert_eq!(r.selected, !(opts.lags.is_some() && opts.leads.is_some()));
        if r.selected {
            assert!(r.ic_value.is_finite(), "{ctx}: ic_value");
        } else {
            assert!(
                r.ic_value.is_nan(),
                "{ctx}: ic_value is NaN when both are fixed"
            );
        }
        n += 1;
    }
    assert!(n >= 38, "fixture lost DOLS cases: {n}");
}

/// The residual-system long-run covariances and both automatic
/// bandwidths against `arch.covariance.kernel` (and the documented
/// Andrews closed form).
#[test]
fn golden_long_run_covariance_matches_arch_kernels() {
    let fx = load_fixture("fmols.json");
    let block = &fx["kernel"];
    let eta = as_mat(&block["eta"]);
    let cases = block["cases"].as_array().expect("cases");
    assert!(cases.len() >= 15);
    for case in cases {
        let kernel = kernel_of(case["kernel"].as_str().expect("kernel"));
        let force_int = case["force_int"].as_bool().expect("force_int");
        let ctx = format!(
            "kernel/{}/bw={}/fi={force_int}",
            case["kernel"], case["bandwidth_arg"]
        );
        let bw = match case["bandwidth_arg"].as_f64() {
            Some(b) => b,
            None => {
                let nw =
                    automatic_bandwidth(eta.as_ref(), kernel, BandwidthRule::NeweyWest, force_int)
                        .expect("newey-west");
                close(
                    nw,
                    num(&case["bandwidth"]),
                    TOL,
                    &format!("{ctx}: arch bandwidth"),
                );
                close(
                    nw,
                    num(&case["newey_west_bandwidth"]),
                    TOL,
                    &format!("{ctx}: newey-west transcription"),
                );
                let an =
                    automatic_bandwidth(eta.as_ref(), kernel, BandwidthRule::Andrews, force_int)
                        .expect("andrews");
                close(
                    an,
                    num(&case["andrews_bandwidth"]),
                    TOL,
                    &format!("{ctx}: andrews closed form"),
                );
                nw
            }
        };
        let lr = long_run_covariance(eta.as_ref(), kernel, bw).expect("lrcov");
        assert_eq!(lr.k, eta.ncols());
        assert_eq!(lr.nobs, eta.nrows());
        // arch's weight vector has n_weights entries including lag 0.
        assert_eq!(
            lr.n_lags + 1,
            case["n_weights"].as_u64().expect("n_weights") as usize,
            "{ctx}: window length"
        );
        if let Some(w) = case["weights"].as_array() {
            for (j, wj) in w.iter().enumerate() {
                let mine = if j <= lr.n_lags {
                    kernel.weight(j, bw)
                } else {
                    0.0
                };
                close(mine, num(wj), 1e-12, &format!("{ctx}: weight[{j}]"));
            }
        }
        close_mat(
            &lr.sigma,
            &case["short_run"],
            TOL,
            &format!("{ctx}: short_run"),
        );
        close_mat(
            &lr.lambda,
            &case["one_sided"],
            TOL,
            &format!("{ctx}: one_sided"),
        );
        close_mat(
            &lr.omega,
            &case["long_run"],
            TOL,
            &format!("{ctx}: long_run"),
        );
    }
}

/// With an integer bandwidth the module's window is exactly
/// `tsecon_hac::lrv`'s: the univariate long-run variance agrees with the
/// shared crate to round-off (the two must never disagree on the same
/// settings — one HAC owner).
#[test]
fn univariate_long_run_variance_agrees_with_tsecon_hac() {
    let fx = load_fixture("fmols.json");
    let eta = as_mat(&fx["kernel"]["eta"]);
    let n = eta.nrows();
    let col0: Vec<f64> = (0..n).map(|i| eta[(i, 0)]).collect();
    let m = Mat::from_fn(n, 1, |i, _| col0[i]);
    for kernel in [Kernel::Bartlett, Kernel::Parzen, Kernel::QuadraticSpectral] {
        for bw in [0.0, 1.0, 4.0, 9.0] {
            let mine = long_run_covariance(m.as_ref(), kernel, bw)
                .expect("lrcov")
                .omega[0][0];
            let shared = tsecon_hac::lrv(&col0, kernel, bw).expect("lrv");
            close(mine, shared, 1e-12, &format!("{kernel:?}/bw={bw}"));
        }
    }
}

// ------------------------------------------------------------- refusals

/// The specification arch silently runs underdetermined (see the fixture
/// `refusals` block) is refused here, naming the caps.
#[test]
fn underdetermined_dols_search_is_refused_with_the_caps_named() {
    let fx = load_fixture("fmols.json");
    for case in fx["refusals"].as_array().expect("refusals") {
        let (y, x) = system(&fx, case["system"].as_str().expect("system"));
        let err = dols(
            &y,
            x.as_ref(),
            &DolsOptions {
                trend: trend_of(case["trend"].as_str().expect("trend")),
                ..DolsOptions::default()
            },
        )
        .unwrap_err();
        assert!(matches!(err, CointError::InvalidSpec { .. }), "{err:?}");
        let msg = err.to_string();
        for name in case["must_name"].as_array().expect("must_name") {
            let name = name.as_str().expect("name");
            assert!(msg.contains(name), "message must name {name}: {msg}");
        }
        assert!(
            msg.contains(&format!("{}", case["rows"].as_u64().expect("rows"))),
            "message must state the rows: {msg}"
        );
        // Capping the search makes the same data estimable.
        let r = dols(
            &y,
            x.as_ref(),
            &DolsOptions {
                max_lag: Some(2),
                max_lead: Some(2),
                ..DolsOptions::default()
            },
        )
        .expect("capped search runs");
        assert!(r.lags <= 2 && r.leads <= 2);
    }
}

#[test]
fn every_refusal_names_its_parameter() {
    let fx = load_fixture("fmols.json");
    let (y, x) = system(&fx, "sim_k1");
    let base = CointRegOptions::default();

    // Length mismatch names x and y.
    let err = fmols(&y[..y.len() - 1], x.as_ref(), &base).unwrap_err();
    assert!(matches!(err, CointError::Dimension { .. }), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("x") && msg.contains("y"), "{msg}");

    // Empty x.
    let empty = Mat::<f64>::zeros(y.len(), 0);
    let err = ccr(&y, empty.as_ref(), &base).unwrap_err();
    assert!(
        err.to_string()
            .contains("x must hold at least one regressor"),
        "{err}"
    );

    // Negative / non-finite bandwidth names bandwidth and its value.
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        for est in ["fmols", "ccr", "dols"] {
            let msg = match est {
                "fmols" => fmols(
                    &y,
                    x.as_ref(),
                    &CointRegOptions {
                        bandwidth: Some(bad),
                        ..base.clone()
                    },
                )
                .unwrap_err()
                .to_string(),
                "ccr" => ccr(
                    &y,
                    x.as_ref(),
                    &CointRegOptions {
                        bandwidth: Some(bad),
                        ..base.clone()
                    },
                )
                .unwrap_err()
                .to_string(),
                _ => dols(
                    &y,
                    x.as_ref(),
                    &DolsOptions {
                        bandwidth: Some(bad),
                        ..DolsOptions::default()
                    },
                )
                .unwrap_err()
                .to_string(),
            };
            assert!(
                msg.starts_with(&format!("bandwidth = {bad}")),
                "{est}: {msg}"
            );
        }
    }

    // x_trend smaller than trend.
    let err = fmols(
        &y,
        x.as_ref(),
        &CointRegOptions {
            trend: CointTrend::ConstantTrend,
            x_trend: Some(CointTrend::Constant),
            ..base.clone()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("x_trend = \"c\"") && msg.contains("trend = \"ct\""),
        "{msg}"
    );

    // The truncated kernel.
    let err = fmols(
        &y,
        x.as_ref(),
        &CointRegOptions {
            kernel: Kernel::Truncated,
            ..base.clone()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("kernel = \"truncated\""), "{err}");

    // NaN in y names y and the index; NaN in x names x and the position.
    let mut y_nan = y.clone();
    y_nan[7] = f64::NAN;
    let err = fmols(&y_nan, x.as_ref(), &base).unwrap_err();
    assert!(
        matches!(err, CointError::NonFiniteSeries { index: 7, .. }),
        "{err:?}"
    );
    let mut x_nan = x.clone();
    x_nan[(3, 0)] = f64::INFINITY;
    let err = dols(&y, x_nan.as_ref(), &DolsOptions::default()).unwrap_err();
    assert!(
        matches!(
            err,
            CointError::NonFinite {
                at: Some((3, 0)),
                ..
            }
        ),
        "{err:?}"
    );

    // Too-short samples name T and the requirement.
    let err = fmols(
        &y[..4],
        Mat::from_fn(4, 1, |i, _| x[(i, 0)]).as_ref(),
        &base,
    )
    .unwrap_err();
    assert!(err.to_string().contains("T = 4"), "{err}");

    // DOLS: common with unequal fixed lags/leads, and unequal caps.
    let err = dols(
        &y,
        x.as_ref(),
        &DolsOptions {
            lags: Some(1),
            leads: Some(2),
            common: true,
            ..DolsOptions::default()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("common = true") && msg.contains("lags = 1") && msg.contains("leads = 2"),
        "{msg}"
    );
    let err = dols(
        &y,
        x.as_ref(),
        &DolsOptions {
            max_lag: Some(1),
            max_lead: Some(2),
            common: true,
            ..DolsOptions::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("max_lag = Some(1)"), "{err}");

    // DOLS: fixed lags/leads too large for T.
    let err = dols(
        &y,
        x.as_ref(),
        &DolsOptions {
            lags: Some(100),
            leads: Some(100),
            ..DolsOptions::default()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("lags = 100") && msg.contains("leads = 100") && msg.contains("T = 200"),
        "{msg}"
    );
}

/// The three estimators agree with plain OLS on the regressor
/// coefficient to the order of the correction (they are all
/// super-consistent for the same beta) — a sanity check on the fixture
/// systems, not a golden.
#[test]
fn corrected_estimators_track_the_true_cointegrating_vector() {
    let fx = load_fixture("fmols.json");
    for name in ["sim_k1", "sim_k3"] {
        let (y, x) = system(&fx, name);
        let beta = as_vec(&fx["systems"][name]["truth"]["beta"]);
        let f = fmols(&y, x.as_ref(), &CointRegOptions::default()).expect("fmols");
        let c = ccr(&y, x.as_ref(), &CointRegOptions::default()).expect("ccr");
        let d = dols(&y, x.as_ref(), &DolsOptions::default()).expect("dols");
        for (j, b) in beta.iter().enumerate() {
            for (est, r) in [
                ("fmols", &f.params),
                ("ccr", &c.params),
                ("dols", &d.params),
            ] {
                assert!(
                    (r[j] - b).abs() < 0.15,
                    "{name}/{est}: beta[{j}] = {} vs true {b}",
                    r[j]
                );
            }
        }
    }
}
