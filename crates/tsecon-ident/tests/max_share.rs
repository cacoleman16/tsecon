//! Golden and property tests for max-share / maximum-FEV identification.
//!
//! `fixtures/max_share_svar.json` is produced by
//! `fixtures/generate_max_share_svar_fixtures.py` — every expected number
//! comes from an INDEPENDENT NumPy pipeline (`numpy.linalg.lstsq` OLS,
//! `numpy.linalg.cholesky`, `numpy.linalg.eigh`), never from this crate, so
//! reproducing them is a genuine cross-implementation check. The eigenvector is
//! defined only up to sign; both sides pin it with the identical `cumsum` rule,
//! which is what makes the golden reproducible across two independent
//! self-adjoint eigensolvers.
//!
//! * **`core_matches_numpy_from_theta`** feeds the orthogonalized MA
//!   coefficients straight to [`max_share_shock`] and bit-matches the whole
//!   result (irf, impact, q, share, fev profile, eigenvalues) to ~1e-10 — the
//!   strong validation of the novel eigen-identification core.
//! * **`pipeline_matches_numpy_from_reduced_form`** rebuilds the orthogonalized
//!   MA from the stored reduced form via `cholesky_irf` (the exact call the
//!   binding makes) and matches at 1e-8.
//! * consistency checks (fev in `[0,1]`, share == windowed FEV ratio,
//!   eigenvalues ascending, a clear spectral gap) and a favorable-DGP recovery
//!   property (a single dominant shock is recovered) round it out.

use serde_json::Value;
use tsecon_bayes::cholesky_irf;
use tsecon_ident::{max_share_shock, MaxShareSign, MaxShareWeighting};
use tsecon_linalg::faer::Mat;

const TOL_CORE: f64 = 1e-10; // observed agreement is < 1e-13; 1e-10 leaves platform margin
const TOL_PIPE: f64 = 1e-8; // reduced-form-fed: adds one Cholesky round-trip

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/max_share_svar.json",
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

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().expect("array").iter().map(f64s).collect()
}

fn u(v: &Value) -> usize {
    v.as_u64().expect("uint") as usize
}

fn mat(v: &Value) -> Mat<f64> {
    let r = rows(v);
    let nr = r.len();
    let nc = r[0].len();
    Mat::from_fn(nr, nc, |i, j| r[i][j])
}

fn theta_mats(v: &Value) -> Vec<Mat<f64>> {
    v.as_array().expect("array").iter().map(mat).collect()
}

fn weighting_of(s: &str) -> MaxShareWeighting {
    match s {
        "window" => MaxShareWeighting::Window,
        "cumulative" => MaxShareWeighting::Cumulative,
        other => panic!("unknown weighting {other:?}"),
    }
}

fn sign_of(s: &str) -> MaxShareSign {
    match s {
        "cumsum" => MaxShareSign::Cumsum,
        "impact" => MaxShareSign::Impact,
        "none" => MaxShareSign::None,
        other => panic!("unknown sign {other:?}"),
    }
}

fn close_at(actual: f64, expected: f64, tol: f64, what: &str) {
    let err = (actual - expected).abs();
    let rel = err / (1.0 + expected.abs());
    assert!(
        err < tol || rel < tol,
        "{what}: actual={actual:.15e} expected={expected:.15e} abs_err={err:.3e} rel={rel:.3e}"
    );
}

fn close_slice(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} length");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        close_at(*a, *e, tol, &format!("{what}[{i}]"));
    }
}

/// Runs `max_share_shock` on the given theta with the case parameters and
/// checks every returned field against the fixture's expected block.
fn check_case(name: &str, theta: &[Mat<f64>], case: &Value, tol: f64) {
    let p = &case["params"];
    let exp = &case["expected"];
    let r = max_share_shock(
        theta,
        u(&p["target"]),
        u(&p["h0"]),
        u(&p["h1"]),
        p["exclude_impact"].as_bool().expect("bool"),
        weighting_of(p["weighting"].as_str().expect("str")),
        sign_of(p["sign"].as_str().expect("str")),
    )
    .unwrap_or_else(|e| panic!("{name}: max_share_shock failed: {e}"));

    let exp_irf = rows(&exp["irf"]);
    assert_eq!(r.irf.len(), exp_irf.len(), "{name} irf horizon length");
    for (h, (got, want)) in r.irf.iter().zip(exp_irf.iter()).enumerate() {
        close_slice(got, want, tol, &format!("{name} irf[{h}]"));
    }
    close_slice(
        &r.impact,
        &f64s(&exp["impact"]),
        tol,
        &format!("{name} impact"),
    );
    close_slice(&r.q, &f64s(&exp["q"]), tol, &format!("{name} q"));
    close_at(
        r.share_window,
        exp["share_window"].as_f64().expect("f64"),
        tol,
        &format!("{name} share_window"),
    );
    close_slice(
        &r.fev_share,
        &f64s(&exp["fev_share"]),
        tol,
        &format!("{name} fev_share"),
    );
    close_slice(
        &r.eigenvalues,
        &f64s(&exp["eigenvalues"]),
        tol,
        &format!("{name} eigenvalues"),
    );
}

#[test]
fn core_matches_numpy_from_theta() {
    let fx = load();
    let theta = theta_mats(&fx["theta"]);
    let cases = fx["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty());
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        check_case(name, &theta, case, TOL_CORE);
    }
}

#[test]
fn pipeline_matches_numpy_from_reduced_form() {
    // Rebuild the orthogonalized MA from the stored reduced form exactly the
    // way the Python binding will: cholesky_irf(reg_coefs, sigma, lags, horizon).
    let fx = load();
    let b = mat(&fx["reg_coefs"]);
    let sigma = mat(&fx["sigma"]);
    let cases = fx["cases"].as_array().expect("cases array");
    for case in cases {
        let name = case["name"].as_str().unwrap_or("?");
        let p = &case["params"];
        let lags = u(&p["lags"]);
        let horizon = u(&p["horizon"]);
        let theta = cholesky_irf(b.as_ref(), sigma.as_ref(), lags, horizon)
            .unwrap_or_else(|e| panic!("{name}: cholesky_irf failed: {e:?}"));
        check_case(name, &theta, case, TOL_PIPE);
    }
}

#[test]
fn fev_share_is_a_probability() {
    let fx = load();
    let theta = theta_mats(&fx["theta"]);
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap_or("?");
        let p = &case["params"];
        let r = max_share_shock(
            &theta,
            u(&p["target"]),
            u(&p["h0"]),
            u(&p["h1"]),
            p["exclude_impact"].as_bool().unwrap(),
            weighting_of(p["weighting"].as_str().unwrap()),
            sign_of(p["sign"].as_str().unwrap()),
        )
        .unwrap();
        for (h, s) in r.fev_share.iter().enumerate() {
            assert!(
                s.is_finite() && (-1e-12..=1.0 + 1e-12).contains(s),
                "{name} fev_share[{h}] = {s} not in [0, 1]"
            );
        }
        // exclude_impact => zero impact FEV at horizon 0.
        if p["exclude_impact"].as_bool().unwrap() {
            assert!(
                r.fev_share[0].abs() < 1e-12,
                "{name} exclude_impact fev_share[0] should be 0, got {}",
                r.fev_share[0]
            );
            assert!(
                r.impact[u(&p["target"])].abs() < 1e-10,
                "{name} exclude_impact target impact should be 0"
            );
        }
    }
}

#[test]
fn share_equals_windowed_fev_ratio() {
    // For "window" weighting, share_window is exactly the identified shock's
    // windowed FEV fraction recomputed from the returned IRF and theta.
    let fx = load();
    let theta = theta_mats(&fx["theta"]);
    for case in fx["cases"].as_array().expect("cases") {
        let p = &case["params"];
        if p["weighting"].as_str().unwrap() != "window" {
            continue;
        }
        let name = case["name"].as_str().unwrap_or("?");
        let target = u(&p["target"]);
        let (h0, h1) = (u(&p["h0"]), u(&p["h1"]));
        let r = max_share_shock(
            &theta,
            target,
            h0,
            h1,
            p["exclude_impact"].as_bool().unwrap(),
            MaxShareWeighting::Window,
            sign_of(p["sign"].as_str().unwrap()),
        )
        .unwrap();
        let mut num = 0.0;
        let mut den = 0.0;
        for (irf_s, theta_s) in r.irf[h0..=h1].iter().zip(theta[h0..=h1].iter()) {
            num += irf_s[target] * irf_s[target];
            for j in 0..theta_s.ncols() {
                let v = theta_s[(target, j)];
                den += v * v;
            }
        }
        close_at(
            r.share_window,
            num / den,
            1e-10,
            &format!("{name} share == windowed FEV ratio"),
        );
    }
}

#[test]
fn eigenvalues_ascending_with_a_spectral_gap() {
    let fx = load();
    let theta = theta_mats(&fx["theta"]);
    for case in fx["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap_or("?");
        let p = &case["params"];
        let r = max_share_shock(
            &theta,
            u(&p["target"]),
            u(&p["h0"]),
            u(&p["h1"]),
            p["exclude_impact"].as_bool().unwrap(),
            weighting_of(p["weighting"].as_str().unwrap()),
            sign_of(p["sign"].as_str().unwrap()),
        )
        .unwrap();
        for w in r.eigenvalues.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-12,
                "{name} eigenvalues must be ascending: {} then {}",
                w[0],
                w[1]
            );
            assert!(w[0] >= -1e-10, "{name} PSD eigenvalue negative: {}", w[0]);
        }
        let n = r.eigenvalues.len();
        assert!(n >= 2, "{name} need >= 2 eigenvalues for a gap diagnostic");
        // Designed DGP: the top eigenvalue is strictly separated.
        assert!(
            r.eigenvalues[n - 1] > r.eigenvalues[n - 2] + 1e-6,
            "{name} expected a clear spectral gap; got {:?}",
            &r.eigenvalues[n - 2..]
        );
    }
}

/// Favorable-DGP recovery (the honest MODERATE property). A single persistent
/// structural shock is engineered to dominate the target's FEV over a long
/// window; max-share should recover it. This validates correctness + wiring,
/// NOT economic interpretation: max-share is a STATISTICAL identification that
/// equals the economic shock only when one shock dominates the band.
#[test]
fn recovers_the_dominant_shock_on_a_favorable_dgp() {
    // Same structural DGP as the fixture generator (population, not estimated).
    let a1 = [[0.30, 0.00, 0.55], [0.00, 0.20, 0.00], [0.00, 0.00, 0.90]];
    let a2 = [[-0.10, 0.00, 0.00], [0.00, 0.00, 0.00], [0.00, 0.00, -0.05]];
    let p_true = [[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.3, 1.2]];
    let k = 3usize;
    let lags = 2usize;
    let horizon = 40usize;

    // Packed reduced-form coefficients b ((1 + k*lags) x k), intercept zero.
    // cholesky_irf reads A_l[(i, j)] = b[(1 + (l-1)*k + j, i)].
    let mut b = Mat::<f64>::zeros(1 + k * lags, k);
    for i in 0..k {
        for j in 0..k {
            b[(1 + j, i)] = a1[i][j]; // lag 1
            b[(1 + k + j, i)] = a2[i][j]; // lag 2
        }
    }
    // sigma = P P' (so cholesky(sigma) = P_true exactly).
    let sigma = Mat::from_fn(k, k, |i, j| {
        (0..k).map(|l| p_true[i][l] * p_true[j][l]).sum::<f64>()
    });
    let theta = cholesky_irf(b.as_ref(), sigma.as_ref(), lags, horizon).expect("theta");

    let r = max_share_shock(
        &theta,
        0, // target
        20,
        40,
        false,
        MaxShareWeighting::Window,
        MaxShareSign::Cumsum,
    )
    .expect("max share");

    // (a) the identified shock explains almost all of the target's long-window FEV.
    assert!(
        r.share_window > 0.80,
        "recovered share {} should exceed 0.80 on the favorable DGP",
        r.share_window
    );
    // (b) the recovered impact aligns with the true dominant column P[:, 2].
    let col2 = [p_true[0][2], p_true[1][2], p_true[2][2]];
    let dot: f64 = r.impact.iter().zip(col2.iter()).map(|(a, b)| a * b).sum();
    let nb: f64 = r.impact.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nc: f64 = col2.iter().map(|v| v * v).sum::<f64>().sqrt();
    let cosine = dot.abs() / (nb * nc);
    assert!(
        cosine > 0.95,
        "recovered impact cosine with the true dominant shock was {cosine}, expected > 0.95"
    );
}

#[test]
fn rejects_invalid_arguments() {
    let fx = load();
    let theta = theta_mats(&fx["theta"]);
    let k = u(&fx["k"]);
    let horizon = u(&fx["horizon"]);

    // target out of range.
    assert!(max_share_shock(
        &theta,
        k,
        0,
        10,
        false,
        MaxShareWeighting::Window,
        MaxShareSign::Cumsum
    )
    .is_err());
    // h0 > h1.
    assert!(max_share_shock(
        &theta,
        0,
        10,
        5,
        false,
        MaxShareWeighting::Window,
        MaxShareSign::Cumsum
    )
    .is_err());
    // h1 beyond the IRF horizon.
    assert!(max_share_shock(
        &theta,
        0,
        0,
        horizon + 1,
        false,
        MaxShareWeighting::Window,
        MaxShareSign::Cumsum
    )
    .is_err());
    // sign="impact" together with exclude_impact.
    assert!(max_share_shock(
        &theta,
        0,
        0,
        10,
        true,
        MaxShareWeighting::Window,
        MaxShareSign::Impact
    )
    .is_err());
    // A valid call still succeeds (guards do not over-reject).
    assert!(max_share_shock(
        &theta,
        0,
        0,
        horizon,
        false,
        MaxShareWeighting::Window,
        MaxShareSign::Cumsum
    )
    .is_ok());
}
