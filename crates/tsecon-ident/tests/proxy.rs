//! Independent-reference golden and property tests for the proxy SVAR
//! (external-instrument SVAR-IV) identification.
//!
//! `fixtures/proxy_svar.json` is produced by
//! `fixtures/generate_proxy_svar_fixtures.py`, whose every number comes from
//! statsmodels (the reduced-form VAR fit and its MA representation) and plain
//! NumPy (the method-of-moments identification, the first stage, and the
//! shock series) — never from this crate. The fixture feeds the reduced-form
//! quantities (`resid`, `sigma_u`, `psi`) and the aligned proxy straight into
//! [`tsecon_ident::proxy_svar`], so reproducing the expected outputs is a
//! genuine cross-implementation check of the identification algebra in
//! isolation, exactly at the crate boundary (proxy_svar consumes reduced-form
//! inputs and does not fit a VAR).
//!
//! Tolerance: `rtol = 1e-9`, `atol = 1e-11` — only faer-vs-NumPy OLS /
//! Cholesky rounding separates the two implementations of the same closed
//! form.

use serde_json::Value;
use tsecon_ident::proxy_svar;
use tsecon_linalg::faer::Mat;

const RTOL: f64 = 1e-9;
const ATOL: f64 = 1e-11;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/proxy_svar.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Like [`f64s`], but a JSON `null` decodes to `NaN` — the proxy encodes its
/// unavailability mask (the dropped observations) as null.
fn proxy_f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| {
            if x.is_null() {
                f64::NAN
            } else {
                x.as_f64().expect("number")
            }
        })
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn mat(v: &Value) -> Mat<f64> {
    let r = rows(v);
    let nr = r.len();
    let nc = r[0].len();
    Mat::from_fn(nr, nc, |i, j| r[i][j])
}

fn psis(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().expect("array").iter().map(mat).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn close_at(actual: f64, expected: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < ATOL || rel < RTOL,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn close_slice(actual: &[f64], expected: &[f64], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close_at(*a, *e, &format!("{what}[{i}]"));
    }
}

#[test]
fn golden_matches_numpy_reference() {
    let fx = load();
    let cases = fx["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let resid = mat(&case["resid"]);
        let sigma_u = mat(&case["sigma_u"]);
        let psi = psis(&case["psi"]);
        let proxy = proxy_f64s(&case["proxy_aligned"]);
        let norm_var = u(&case["norm_var"]);
        let unit = case["unit"].as_f64().expect("unit");
        let robust_f = case["robust_f"].as_bool().expect("robust_f");

        let r = proxy_svar(
            resid.as_ref(),
            &proxy,
            &psi,
            sigma_u.as_ref(),
            norm_var,
            unit,
            robust_f,
        )
        .unwrap_or_else(|e| panic!("[{name}] proxy_svar failed: {e}"));

        let exp = &case["expected"];

        // n_proxy is an exact integer.
        assert_eq!(r.n_proxy, u(&exp["n_proxy"]), "[{name}] n_proxy");

        // The unit-effect normalization is exact.
        assert_eq!(
            r.impact[norm_var], unit,
            "[{name}] impact[norm_var] must equal unit exactly"
        );
        assert_eq!(
            r.relative_impact[norm_var], 1.0,
            "[{name}] relative_impact[norm_var] must equal 1 exactly"
        );
        assert_eq!(
            r.irf[0], r.impact,
            "[{name}] irf[0] must equal the impact vector"
        );

        close_slice(
            &r.relative_impact,
            &f64s(&exp["relative_impact"]),
            &format!("{name} relative_impact"),
        );
        close_slice(&r.impact, &f64s(&exp["impact"]), &format!("{name} impact"));
        close_slice(&r.cov_um, &f64s(&exp["cov_um"]), &format!("{name} cov_um"));
        close_at(
            r.first_stage_f,
            exp["first_stage_f"].as_f64().expect("f"),
            &format!("{name} first_stage_f"),
        );
        close_at(
            r.reliability,
            exp["reliability"].as_f64().expect("rel"),
            &format!("{name} reliability"),
        );

        // IRF, row by row.
        let exp_irf = rows(&exp["irf"]);
        assert_eq!(r.irf.len(), exp_irf.len(), "[{name}] irf horizons");
        for (h, (a, e)) in r.irf.iter().zip(exp_irf.iter()).enumerate() {
            close_slice(a, e, &format!("{name} irf[{h}]"));
        }

        close_slice(&r.shock, &f64s(&exp["shock"]), &format!("{name} shock"));
    }
}

#[test]
fn parameter_recovery_matches_population_truth() {
    // The population relative impact is exact: H[:, 0] / H[norm_var, 0]. For
    // the baseline (norm_var = 0, the largest denominator) the estimator lands
    // within ~0.05 at T=2000, proving the algebra is the correct estimator of
    // the structural object (not merely that two formula implementations
    // agree). The generator asserts the same bound; we re-check it here from
    // the crate's own output.
    let fx = load();
    let case = fx["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["name"].as_str() == Some("baseline_robust"))
        .expect("baseline case present");

    let resid = mat(&case["resid"]);
    let sigma_u = mat(&case["sigma_u"]);
    let psi = psis(&case["psi"]);
    let proxy = proxy_f64s(&case["proxy_aligned"]);
    let norm_var = u(&case["norm_var"]);
    let unit = case["unit"].as_f64().expect("unit");
    let robust_f = case["robust_f"].as_bool().expect("robust_f");
    let rho_true = f64s(&case["rho_true"]);

    let r = proxy_svar(
        resid.as_ref(),
        &proxy,
        &psi,
        sigma_u.as_ref(),
        norm_var,
        unit,
        robust_f,
    )
    .expect("proxy_svar ok");

    for (j, (&est, &tru)) in r.relative_impact.iter().zip(rho_true.iter()).enumerate() {
        assert!(
            (est - tru).abs() < 0.05,
            "relative_impact[{j}] = {est} vs population {tru} (err {:.4} >= 0.05)",
            (est - tru).abs()
        );
    }

    // Projecting the residuals onto the estimated shock reproduces the impact
    // vector: b_j = Cov(u_j, eps1_hat) / Var(eps1_hat) by construction of the
    // minimum-variance shock. Check this identity on the full sample.
    let t = r.shock.len();
    let n = r.impact.len();
    let sbar = r.shock.iter().sum::<f64>() / t as f64;
    let mut ubar = vec![0.0; n];
    for j in 0..n {
        for i in 0..t {
            ubar[j] += resid[(i, j)];
        }
        ubar[j] /= t as f64;
    }
    let mut var_s = 0.0;
    for &s in &r.shock {
        var_s += (s - sbar) * (s - sbar);
    }
    for j in 0..n {
        let mut cov = 0.0;
        for i in 0..t {
            cov += (resid[(i, j)] - ubar[j]) * (r.shock[i] - sbar);
        }
        let recovered = cov / var_s;
        assert!(
            (recovered - r.impact[j]).abs() < 1e-8,
            "shock projection[{j}] = {recovered} vs impact {} (identity broken)",
            r.impact[j]
        );
    }
}
