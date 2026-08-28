//! Golden, identity, and refusal tests for the Ornstein-Uhlenbeck spread
//! utilities (`ou_fit`, `spread_zscore`) against `fixtures/ou.json`.
//!
//! Grade of the golden (see the validation matrix): the AR(1) leg
//! (`c`, `phi`, `eta2`, both standard errors, the `(c, phi)` covariance,
//! and the conditional log-likelihood) is pinned against statsmodels
//! `AutoReg(x, lags=1, trend='c').fit()` — an independent package
//! computing the same estimator through lstsq — at 1e-10 relative
//! (achieved ~3e-15 across all five cells on the shipped fixture). The
//! OU mapping layer on top is a documented-formula golden: the fixture
//! generator transcribes the published inverse discretization + delta
//! method into NumPy, and the crate must reproduce it at 1e-10 relative
//! (achieved ~1e-12; the two sides sum the same data in different
//! orders, so bit-for-bit is not claimed *across* implementations — the
//! bit-for-bit claim lives in
//! `ou_equals_closed_form_ar1_bit_for_bit` below, where the summation
//! order is controlled).

mod common;

use common::{as_vec, assert_rel_close, load_fixture, num};
use tsecon_coint::{ou_fit, spread_zscore, CointError};

fn cell<'a>(fx: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    fx["cells"]
        .as_array()
        .expect("cells array")
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("no fixture cell {name}"))
}

/// Statsmodels `AutoReg` golden on the AR(1) discretization leg, all four
/// mean-reverting and explosive cells: `c`, `phi`, `sigma2` (= `eta2`),
/// both standard errors, and the conditional log-likelihood at 1e-10
/// relative.
#[test]
fn golden_ar1_leg_matches_statsmodels_autoreg() {
    let fx = load_fixture("ou.json");
    for name in [
        "daily_fast",
        "daily_slow",
        "monthly",
        "daily_weak",
        "explosive",
    ] {
        let c = cell(&fx, name);
        let x = as_vec(&c["x"]);
        let dt = num(&c["dt"]);
        let r = ou_fit(&x, dt, 0.95).unwrap();
        let sm = &c["statsmodels"];
        assert_rel_close(r.phi, num(&sm["phi"]), 1e-10, &format!("{name}: phi"));
        assert_rel_close(r.c, num(&sm["c"]), 1e-10, &format!("{name}: c"));
        assert_rel_close(r.eta2, num(&sm["sigma2"]), 1e-10, &format!("{name}: eta2"));
        assert_rel_close(r.loglik, num(&sm["llf"]), 1e-10, &format!("{name}: loglik"));
        assert_rel_close(
            r.phi_se,
            num(&sm["phi_se"]),
            1e-10,
            &format!("{name}: phi_se"),
        );
        assert_rel_close(r.c_se, num(&sm["c_se"]), 1e-10, &format!("{name}: c_se"));
        assert_eq!(r.n_obs, x.len() - 1, "{name}: n_obs");
    }
}

/// Documented-formula golden on the OU mapping layer: kappa, mu, sigma,
/// their delta-method standard errors, half-life, the level-scale CI
/// (the Monte-Carlo-vetted construction), and the stationary sd, against
/// the NumPy transcription in the generator, at 1e-10 relative.
#[test]
fn golden_ou_mapping_matches_documented_formulas() {
    let fx = load_fixture("ou.json");
    // daily_weak pins the +inf upper CI branch (a positive kappa whose
    // level-scale interval crosses zero).
    for name in ["daily_fast", "daily_slow", "monthly", "daily_weak"] {
        let c = cell(&fx, name);
        let x = as_vec(&c["x"]);
        let r = ou_fit(&x, num(&c["dt"]), num(&c["level"])).unwrap();
        let ou = &c["ou"];
        assert_rel_close(r.kappa, num(&ou["kappa"]), 1e-10, &format!("{name}: kappa"));
        assert_rel_close(r.mu, num(&ou["mu"]), 1e-10, &format!("{name}: mu"));
        assert_rel_close(r.sigma, num(&ou["sigma"]), 1e-10, &format!("{name}: sigma"));
        assert_rel_close(
            r.kappa_se,
            num(&ou["kappa_se"]),
            1e-10,
            &format!("{name}: kappa_se"),
        );
        assert_rel_close(r.mu_se, num(&ou["mu_se"]), 1e-10, &format!("{name}: mu_se"));
        assert_rel_close(
            r.sigma_se,
            num(&ou["sigma_se"]),
            1e-10,
            &format!("{name}: sigma_se"),
        );
        assert_rel_close(
            r.half_life,
            num(&ou["half_life"]),
            1e-10,
            &format!("{name}: half_life"),
        );
        let ci = r.half_life_ci.unwrap_or_else(|| panic!("{name}: no CI"));
        let fci = ou["half_life_ci"].as_array().expect("ci array");
        // inv_norm_cdf vs scipy norm.ppf agree far tighter than 1e-10; the
        // CI inherits that. A JSON null upper endpoint encodes +inf (the
        // kappa interval crossed zero).
        assert_rel_close(ci.0, num(&fci[0]), 1e-10, &format!("{name}: ci lo"));
        if fci[1].is_null() {
            assert!(
                ci.1.is_infinite() && ci.1 > 0.0,
                "{name}: ci hi must be +inf"
            );
        } else {
            assert_rel_close(ci.1, num(&fci[1]), 1e-10, &format!("{name}: ci hi"));
        }
        assert_rel_close(
            r.stationary_sd.unwrap_or(f64::NAN),
            num(&ou["stationary_sd"]),
            1e-10,
            &format!("{name}: stationary_sd"),
        );
        assert!(r.mean_reverting, "{name}: mean_reverting");
        assert!(
            ci.0 < r.half_life && r.half_life < ci.1,
            "{name}: CI brackets"
        );
    }
}

/// The estimator IS the closed-form AR(1) OLS/MLE mapping — recomputing
/// the same centered two-pass sums in the same order in the test must
/// reproduce every parameter **bit-for-bit** (no tolerance). Bit-for-bit
/// is claimable here because the summation order is identical; the
/// cross-implementation comparisons above use 1e-10 because NumPy's
/// pairwise summation orders the same arithmetic differently.
#[test]
fn ou_equals_closed_form_ar1_bit_for_bit() {
    let fx = load_fixture("ou.json");
    let c = cell(&fx, "daily_fast");
    let x = as_vec(&c["x"]);
    let dt = num(&c["dt"]);
    let r = ou_fit(&x, dt, 0.95).unwrap();

    let n = x.len() - 1;
    let nf = n as f64;
    let (lag, lead) = (&x[..n], &x[1..]);
    let m_lag = lag.iter().sum::<f64>() / nf;
    let m_lead = lead.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for t in 0..n {
        let d = lag[t] - m_lag;
        sxx += d * d;
        sxy += d * (lead[t] - m_lead);
    }
    let phi = sxy / sxx;
    let c0 = m_lead - phi * m_lag;
    let mut rss = 0.0;
    for t in 0..n {
        let e = lead[t] - c0 - phi * lag[t];
        rss += e * e;
    }
    let eta2 = rss / nf;

    assert_eq!(r.phi, phi, "phi not bit-identical to the closed form");
    assert_eq!(r.c, c0, "c not bit-identical to the closed form");
    assert_eq!(r.eta2, eta2, "eta2 not bit-identical to the closed form");
    assert_eq!(
        r.kappa,
        -phi.ln() / dt,
        "kappa not the closed-form map of phi"
    );
    assert_eq!(
        r.mu,
        c0 / (1.0 - phi),
        "mu not the closed-form map of (c, phi)"
    );
    let a = -2.0 * phi.ln() / (dt * (1.0 - phi * phi));
    assert_eq!(r.sigma, (eta2 * a).sqrt(), "sigma not the closed-form map");
}

/// The three published identities, at f64 identity/1-ulp strength:
/// `half_life * kappa = ln 2` (exact division round trip),
/// `phi = exp(-kappa dt)` (round trip through the log), and the
/// stationary-variance identity `eta2 = sigma^2 (1 - phi^2) / (2 kappa)`.
#[test]
fn identities_half_life_discretization_stationary_variance() {
    let fx = load_fixture("ou.json");
    for name in ["daily_fast", "daily_slow", "monthly", "daily_weak"] {
        let c = cell(&fx, name);
        let x = as_vec(&c["x"]);
        let dt = num(&c["dt"]);
        let r = ou_fit(&x, dt, 0.95).unwrap();
        // half_life = ln2 / kappa is a single division; multiplying back is
        // one rounding each way.
        assert_rel_close(
            r.half_life * r.kappa,
            core::f64::consts::LN_2,
            1e-15,
            &format!("{name}: half_life * kappa = ln 2"),
        );
        assert_rel_close(
            (-r.kappa * dt).exp(),
            r.phi,
            1e-14,
            &format!("{name}: exp(-kappa dt) = phi"),
        );
        assert_rel_close(
            r.sigma * r.sigma * (1.0 - r.phi * r.phi) / (2.0 * r.kappa),
            r.eta2,
            1e-12,
            &format!("{name}: stationary-variance identity"),
        );
        // stationary_sd^2 = sigma^2 / (2 kappa)
        let sd = r.stationary_sd.unwrap_or(f64::NAN);
        assert_rel_close(
            sd * sd * 2.0 * r.kappa,
            r.sigma * r.sigma,
            1e-12,
            &format!("{name}: stationary sd identity"),
        );
    }
}

/// The `phi_hat > 1` cell reports the failure honestly instead of
/// erroring: `kappa < 0`, `mean_reverting = false`, `half_life = +inf`,
/// no CI, no stationary sd — and the AR(1) leg is still golden-pinned
/// (checked in `golden_ar1_leg_matches_statsmodels_autoreg`).
#[test]
fn explosive_cell_reports_not_mean_reverting() {
    let fx = load_fixture("ou.json");
    let c = cell(&fx, "explosive");
    let x = as_vec(&c["x"]);
    let r = ou_fit(&x, num(&c["dt"]), 0.95).unwrap();
    assert!(r.phi > 1.0, "fixture must keep phi_hat > 1 (got {})", r.phi);
    assert!(r.kappa < 0.0);
    assert_rel_close(r.kappa, num(&c["ou"]["kappa"]), 1e-10, "explosive kappa");
    assert!(!r.mean_reverting);
    assert!(r.half_life.is_infinite() && r.half_life > 0.0);
    assert!(r.half_life_ci.is_none());
    assert!(r.stationary_sd.is_none());
    // mu is still finite here (phi != 1 exactly) and matches the formula.
    assert_rel_close(r.mu, num(&c["ou"]["mu"]), 1e-10, "explosive mu");
    // ... but scoring against a nonexistent stationary law is refused.
    let err = r.zscore(&x).unwrap_err();
    assert!(matches!(err, CointError::InvalidArgument { .. }));
}

/// `spread_zscore` reproduces the documented `(x - mu) / (sigma /
/// sqrt(2 kappa))` head from the fixture, and `OuFit::zscore` is the
/// same computation.
#[test]
fn golden_spread_zscore() {
    let fx = load_fixture("ou.json");
    let c = cell(&fx, "daily_fast");
    let x = as_vec(&c["x"]);
    let r = ou_fit(&x, num(&c["dt"]), 0.95).unwrap();
    let want = as_vec(&fx["zscore"]["zscore_head"]);
    let z = spread_zscore(&x[..want.len()], r.kappa, r.mu, r.sigma).unwrap();
    for (i, (zi, wi)) in z.iter().zip(&want).enumerate() {
        assert_rel_close(*zi, *wi, 1e-10, &format!("zscore[{i}]"));
    }
    let z2 = r.zscore(&x[..want.len()]).unwrap();
    assert_eq!(z, z2, "OuFit::zscore must be spread_zscore exactly");
}

/// Refusals that teach: each invalid input gets a named error, not a
/// panic and not a silently wrong number.
#[test]
fn refusals() {
    let ok = [0.4, 0.1, 0.3, 0.2, 0.35, 0.15, 0.35, 0.1];
    // too short
    assert!(matches!(
        ou_fit(&[1.0, 2.0, 3.0], 1.0, 0.95),
        Err(CointError::InvalidArgument { .. })
    ));
    // bad dt / level
    assert!(ou_fit(&ok, 0.0, 0.95).is_err());
    assert!(ou_fit(&ok, -1.0, 0.95).is_err());
    assert!(ou_fit(&ok, f64::NAN, 0.95).is_err());
    assert!(ou_fit(&ok, 1.0, 0.0).is_err());
    assert!(ou_fit(&ok, 1.0, 1.0).is_err());
    // non-finite data, with the index named and an OU-appropriate
    // consequence — not the multivariate-cointegration eigenvalue text
    // (audit round 10, finding 3f; the variant moved to NonFiniteSeries
    // so the AR(1) surfaces carry their own teaching).
    let mut bad = ok;
    bad[3] = f64::NAN;
    let err = ou_fit(&bad, 1.0, 0.95).unwrap_err();
    assert!(matches!(err, CointError::NonFiniteSeries { index: 3, .. }));
    let msg = err.to_string();
    assert!(msg.contains("index 3"), "index must be named: {msg}");
    assert!(
        msg.contains("AR(1)"),
        "consequence must be the AR(1) fit: {msg}"
    );
    assert!(
        !msg.contains("eigenvalue"),
        "the cointegration-crate NaN text is wrong for an AR(1) fit: {msg}"
    );
    let zerr = spread_zscore(&bad, 1.0, 0.0, 1.0).unwrap_err();
    assert!(matches!(zerr, CointError::NonFiniteSeries { index: 3, .. }));
    assert!(
        !zerr.to_string().contains("eigenvalue"),
        "spread_zscore NaN text must be OU-appropriate: {zerr}"
    );
    // constant series: the lagged regressor has zero variance
    assert!(matches!(
        ou_fit(&[2.0; 16], 1.0, 0.95),
        Err(CointError::Singular { .. })
    ));
    // anti-persistent (phi_hat < 0): no real kappa exists
    let alternating: Vec<f64> = (0..64)
        .map(|t| if t % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let err = ou_fit(&alternating, 1.0, 0.95).unwrap_err();
    // (an exactly deterministic alternation has rss == 0, so it is refused
    //  as degenerate; jitter it to reach the phi <= 0 branch)
    assert!(matches!(err, CointError::InvalidArgument { .. }));
    let jittered: Vec<f64> = alternating
        .iter()
        .enumerate()
        .map(|(t, v)| v + 1e-3 * ((t * 2654435761 % 97) as f64 / 97.0 - 0.5))
        .collect();
    let err = ou_fit(&jittered, 1.0, 0.95).unwrap_err();
    assert!(
        matches!(err, CointError::InvalidArgument { .. }),
        "expected the phi <= 0 refusal, got {err:?}"
    );
    assert!(err.to_string().contains("anti-persistent"), "{err}");
    // spread_zscore parameter domain
    assert!(spread_zscore(&ok, 0.0, 0.0, 1.0).is_err());
    assert!(spread_zscore(&ok, -1.0, 0.0, 1.0).is_err());
    assert!(spread_zscore(&ok, 1.0, f64::NAN, 1.0).is_err());
    assert!(spread_zscore(&ok, 1.0, 0.0, 0.0).is_err());
    assert!(spread_zscore(&[], 1.0, 0.0, 1.0).is_err());
    // a NON-FINITE kappa/sigma is refused for finiteness, not sign
    // (audit round 10, finding 3g: "requires kappa > 0" misdescribed
    // kappa = inf, which satisfies that inequality)
    for bad_kappa in [f64::INFINITY, f64::NAN] {
        let err = spread_zscore(&ok, bad_kappa, 0.0, 1.0).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("finite kappa"),
            "kappa = {bad_kappa} must be refused for finiteness: {msg}"
        );
        assert!(
            !msg.contains("kappa <= 0"),
            "kappa = {bad_kappa} is not a sign violation: {msg}"
        );
    }
    let err = spread_zscore(&ok, 1.0, 0.0, f64::INFINITY).unwrap_err();
    assert!(
        err.to_string().contains("finite sigma"),
        "sigma = inf must be refused for finiteness: {err}"
    );
}

/// Estimating on a simulated path with known parameters recovers them
/// within loose sanity bands (the strong recovery/bias/coverage claims
/// live in the seeded Monte Carlo,
/// `docs/examples/coverage/experiments/ou_kappa_bias_coverage.py`, and in
/// the Python test suite; this is the crate-level smoke that the mapping
/// points the right way).
#[test]
fn recovery_smoke() {
    let fx = load_fixture("ou.json");
    for name in ["daily_fast", "monthly"] {
        let c = cell(&fx, name);
        let x = as_vec(&c["x"]);
        let r = ou_fit(&x, num(&c["dt"]), 0.95).unwrap();
        let t = &c["true"];
        let (k, mu, s) = (num(&t["kappa"]), num(&t["mu"]), num(&t["sigma"]));
        assert!(
            (r.kappa - k).abs() < 3.0 * k.max(1.0),
            "{name}: kappa {} vs true {k}",
            r.kappa
        );
        assert!((r.mu - mu).abs() < 5.0 * r.mu_se + 0.5, "{name}: mu");
        assert!((r.sigma - s).abs() / s < 0.2, "{name}: sigma within 20%");
        assert!(r.mean_reverting, "{name}");
    }
}
