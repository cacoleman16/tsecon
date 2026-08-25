//! Fixture-driven validation of the multivariate GARCH crate against
//! `fixtures/mgarch.json`.
//!
//! CRITICAL: the fixture is *simulated* DCC-GARCH(1,1) data with the true
//! parameters attached — there is **no external DCC reference** in this
//! project. The DCC dynamics are therefore validated by internal properties
//! (positive-definiteness of every `R_t` and `H_t`, correlation targeting)
//! and by a loose single-realization simulation-recovery bound, not by a
//! golden third-party comparison. Only the univariate stage is `arch`-pinned
//! (through `tsecon-garch`).

use serde_json::Value;
use tsecon_garch::{DistSpec, GarchSpec, MeanSpec, VolSpec};
use tsecon_mgarch::faer::{Mat, MatRef};
use tsecon_mgarch::{constant_correlation_test, CccGarch, CorrDist, DccGarch, DccVariant};

fn load() -> Value {
    let path = format!("{}/../../fixtures/mgarch.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// The fixture's `returns` are stored series-major (`k` rows of `T`).
fn returns(fx: &Value) -> Vec<Vec<f64>> {
    fx["returns"]
        .as_array()
        .expect("returns array")
        .iter()
        .map(|s| {
            s.as_array()
                .expect("series array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

fn f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn spec() -> GarchSpec {
    GarchSpec {
        mean: MeanSpec::Zero,
        vol: VolSpec::Garch { p: 1, q: 1 },
        dist: DistSpec::Normal,
    }
}

/// Smallest eigenvalue of a symmetric `k x k` matrix via power iteration on
/// `cI - M` (test-local; k is tiny). Positive return => positive-definite.
fn min_eig_sym(m: MatRef<'_, f64>) -> f64 {
    let k = m.nrows();
    // Gershgorin upper bound on the spectrum.
    let mut c = 0.0_f64;
    for i in 0..k {
        let mut row = m[(i, i)];
        for j in 0..k {
            if i != j {
                row += m[(i, j)].abs();
            }
        }
        c = c.max(row);
    }
    // Power-iterate B = cI - M for its top eigenvalue lambda_max(B); then
    // lambda_min(M) = c - lambda_max(B).
    let mut v = vec![1.0_f64; k];
    let mut lambda = 0.0;
    for _ in 0..2000 {
        let mut w = vec![0.0_f64; k];
        for i in 0..k {
            let mut s = c * v[i];
            for j in 0..k {
                s -= m[(i, j)] * v[j];
            }
            w[i] = s;
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            break;
        }
        for x in &mut w {
            *x /= norm;
        }
        lambda = norm;
        v = w;
    }
    c - lambda
}

fn is_symmetric(m: MatRef<'_, f64>, tol: f64) -> bool {
    let k = m.nrows();
    (0..k).all(|i| (0..i).all(|j| (m[(i, j)] - m[(j, i)]).abs() <= tol))
}

/// The univariate stage recovers each series' true GARCH(1,1) parameters
/// reasonably well (this is the `arch`-pinned leg of the estimator).
#[test]
fn univariate_params_near_truth() {
    let fx = load();
    let series = returns(&fx);
    let omega = f64_vec(&fx["true"]["omega"]);
    let alpha = f64_vec(&fx["true"]["alpha"]);
    let beta = f64_vec(&fx["true"]["beta"]);

    let fit = CccGarch::new(spec()).fit(&series).unwrap();
    assert_eq!(fit.k(), 3);
    // Loose QMLE bounds on a single 2400-point realization; params are
    // [omega, alpha, beta].
    for i in 0..fit.k() {
        let p = &fit.stage.univariate[i].params;
        assert!(
            (p[0] - omega[i]).abs() < 0.05,
            "series {i} omega {} vs {}",
            p[0],
            omega[i]
        );
        assert!(
            (p[1] - alpha[i]).abs() < 0.05,
            "series {i} alpha {} vs {}",
            p[1],
            alpha[i]
        );
        assert!(
            (p[2] - beta[i]).abs() < 0.06,
            "series {i} beta {} vs {}",
            p[2],
            beta[i]
        );
    }
}

/// The CCC correlation matrix is positive-definite and symmetric with unit
/// diagonal, and its off-diagonals are near the true `Qbar` correlations.
#[test]
fn ccc_correlation_pd() {
    let fx = load();
    let series = returns(&fx);
    let fit = CccGarch::new(spec()).fit(&series).unwrap();
    let r = fit.correlation.as_ref();
    assert!(is_symmetric(r, 1e-14));
    for i in 0..fit.k() {
        assert!((r[(i, i)] - 1.0).abs() <= 1e-14);
    }
    assert!(min_eig_sym(r) > 1e-8, "R not PD");

    // True correlation targets (Qbar in the fixture is already a correlation).
    let qbar = &fx["true"]["Qbar"];
    for (i, ri) in (0..fit.k()).zip(qbar.as_array().unwrap()) {
        let row = f64_vec(ri);
        for (j, &target) in row.iter().enumerate() {
            if i != j {
                assert!(
                    (r[(i, j)] - target).abs() < 0.08,
                    "R[{i}][{j}] {} vs target {target}",
                    r[(i, j)]
                );
            }
        }
    }
}

/// Every conditional covariance `H_t = D_t R D_t` on the CCC fit is symmetric
/// and positive-definite (checked on a stride through the sample).
#[test]
fn ccc_covariance_pd_path() {
    let fx = load();
    let series = returns(&fx);
    let fit = CccGarch::new(spec()).fit(&series).unwrap();
    for t in (0..fit.nobs()).step_by(37) {
        let h = fit.conditional_covariance(t).unwrap();
        assert!(is_symmetric(h.as_ref(), 1e-12), "H_{t} asymmetric");
        assert!(min_eig_sym(h.as_ref()) > 0.0, "H_{t} not PD");
    }
}

/// CCC analytic multi-step covariance forecasts are symmetric and PD, and
/// converge toward the unconditional covariance implied by `R` and the
/// per-series unconditional variances.
#[test]
fn ccc_forecast_covariance() {
    let fx = load();
    let series = returns(&fx);
    let fit = CccGarch::new(spec()).fit(&series).unwrap();
    let horizon = 50;
    let fc = fit.forecast_covariance(horizon).unwrap();
    assert_eq!(fc.len(), horizon);
    for h in &fc {
        assert!(is_symmetric(h.as_ref(), 1e-12));
        assert!(min_eig_sym(h.as_ref()) > 0.0);
    }
    // Long-horizon diagonal approaches omega / (1 - alpha - beta).
    let omega = f64_vec(&fx["true"]["omega"]);
    let alpha = f64_vec(&fx["true"]["alpha"]);
    let beta = f64_vec(&fx["true"]["beta"]);
    let last = fc.last().unwrap();
    for i in 0..fit.k() {
        let uncond = omega[i] / (1.0 - alpha[i] - beta[i]);
        assert!(
            (last[(i, i)] / uncond - 1.0).abs() < 0.35,
            "series {i} forecast var {} vs uncond {uncond}",
            last[(i, i)]
        );
    }
    let err = fit.forecast_covariance(0).unwrap_err();
    assert!(matches!(err, tsecon_mgarch::MgarchError::InvalidHorizon));
}

/// DCC simulation recovery: on the fixture's single simulated realization
/// (truth a = 0.03, b = 0.95, persistence 0.98), the estimated persistence
/// lands within a loose Monte-Carlo tolerance. This is a sanity bound on one
/// realization, NOT a precision claim (there is no golden DCC reference).
#[test]
fn dcc_simulation_recovery() {
    let fx = load();
    let series = returns(&fx);
    let fit = DccGarch::new(spec()).fit(&series).unwrap();

    let true_a = fx["true"]["a_dcc"].as_f64().unwrap();
    let true_b = fx["true"]["b_dcc"].as_f64().unwrap();
    let true_pers = true_a + true_b;

    assert!(fit.a >= 0.0 && fit.b >= 0.0);
    assert!(fit.persistence() < 1.0);
    // Documented loose bar: persistence within 0.05 of the true 0.98.
    assert!(
        (fit.persistence() - true_pers).abs() < 0.05,
        "persistence {} vs true {true_pers} (a={}, b={})",
        fit.persistence(),
        fit.a,
        fit.b
    );
    // The DCC likelihood should not be worse than the CCC (a=b=0) special
    // case it nests.
    let ccc = CccGarch::new(spec()).fit(&series).unwrap();
    assert!(
        fit.loglik >= ccc.loglik - 1e-6,
        "DCC loglik {} < CCC loglik {}",
        fit.loglik,
        ccc.loglik
    );
}

/// Every dynamic correlation `R_t` and conditional covariance `H_t` on the
/// fitted DCC path is symmetric and positive-definite.
#[test]
fn dcc_pd_path() {
    let fx = load();
    let series = returns(&fx);
    let fit = DccGarch::new(spec()).fit(&series).unwrap();
    for t in (0..fit.nobs()).step_by(41) {
        let r: MatRef<'_, f64> = fit.correlation_path[t].as_ref();
        assert!(is_symmetric(r, 1e-12), "R_{t} asymmetric");
        assert!((r[(0, 0)] - 1.0).abs() <= 1e-12);
        assert!(min_eig_sym(r) > 1e-10, "R_{t} not PD");

        let h = fit.conditional_covariance(t).unwrap();
        assert!(is_symmetric(h.as_ref(), 1e-10), "H_{t} asymmetric");
        assert!(min_eig_sym(h.as_ref()) > 0.0, "H_{t} not PD");
    }
}

/// The one-step DCC covariance forecast is symmetric and PD (multi-step
/// requires simulation; only one step is analytic).
#[test]
fn dcc_one_step_forecast() {
    let fx = load();
    let series = returns(&fx);
    let fit = DccGarch::new(spec()).fit(&series).unwrap();
    let h: Mat<f64> = fit.forecast_covariance_one_step().unwrap();
    assert!(is_symmetric(h.as_ref(), 1e-12));
    assert!(min_eig_sym(h.as_ref()) > 0.0);
}

/// A cheaper bivariate slice of the fixture (first two series, first 1200
/// observations) for the variant fits that would otherwise double the
/// suite's optimization time; the precision claims for the variants live in
/// the release-built Python Monte-Carlo tests, not here.
fn subset(series: &[Vec<f64>]) -> Vec<Vec<f64>> {
    series[..2].iter().map(|s| s[..1200].to_vec()).collect()
}

/// cDCC (Aielli 2013) on the fixture data: valid parameters, persistence in
/// the same loose recovery band as DCC (the DGP is plain DCC, and at these
/// parameter values the two recursions are near-coincident — the
/// correction matters for *consistency*, not for this finite sample), the
/// target `S` close to `Qbar` (documented magnitude), and h-step forecasts
/// that are PD and converge to `corr(S)`.
#[test]
fn cdcc_fixture_recovery_and_forecast() {
    let fx = load();
    let series = returns(&fx);
    let dcc = DccGarch::new(spec()).fit(&series).unwrap();
    let fit = DccGarch::new(spec())
        .with_variant(DccVariant::Cdcc)
        .fit(&series)
        .unwrap();

    let true_pers = fx["true"]["a_dcc"].as_f64().unwrap() + fx["true"]["b_dcc"].as_f64().unwrap();
    assert!(fit.a >= 0.0 && fit.b >= 0.0 && fit.persistence() < 1.0);
    assert!(
        (fit.persistence() - true_pers).abs() < 0.05,
        "cDCC persistence {} vs true {true_pers}",
        fit.persistence()
    );
    // The cDCC/DCC contrast on this DGP is small: parameters land close to
    // the plain-DCC estimates and S is within 0.02 of Qbar entrywise.
    assert!((fit.a - dcc.a).abs() < 0.02, "a {} vs {}", fit.a, dcc.a);
    assert!((fit.b - dcc.b).abs() < 0.05, "b {} vs {}", fit.b, dcc.b);
    for i in 0..fit.k() {
        assert!((fit.qbar[(i, i)] - 1.0).abs() <= 1e-12, "S diagonal");
        for j in 0..fit.k() {
            assert!(
                (fit.qbar[(i, j)] - dcc.qbar[(i, j)]).abs() < 0.02,
                "S[{i}][{j}] {} vs Qbar {}",
                fit.qbar[(i, j)],
                dcc.qbar[(i, j)]
            );
        }
    }

    // h-step forecasts: PD every step, converging to corr(S).
    let fc = fit.forecast(60).unwrap();
    assert_eq!(fc.correlation.len(), 60);
    for r in [&fc.correlation[0], &fc.correlation[59]] {
        assert!(is_symmetric(r.as_ref(), 1e-12));
        assert!(min_eig_sym(r.as_ref()) > 0.0);
    }
    for h in [&fc.covariance[0], &fc.covariance[59]] {
        assert!(is_symmetric(h.as_ref(), 1e-10));
        assert!(min_eig_sym(h.as_ref()) > 0.0);
    }
    // Geometric convergence: || R_60 - corr(S) || <= (a+b)^59 * || R_1 - corr(S) || + dust.
    let k = fit.k();
    let d: Vec<f64> = (0..k).map(|i| fit.qbar[(i, i)].sqrt()).collect();
    let mut dist1 = 0.0_f64;
    let mut dist60 = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            let rbar = fit.qbar[(i, j)] / (d[i] * d[j]);
            dist1 = dist1.max((fc.correlation[0][(i, j)] - rbar).abs());
            dist60 = dist60.max((fc.correlation[59][(i, j)] - rbar).abs());
        }
    }
    // Near-geometric contraction at rate (a + b): exact in Q-space, and
    // within ~1% of geometric after the nonlinear correlation
    // normalization (measured 0.1751 vs (a+b)^59 = 0.1730 on this
    // fixture); 1.1x is the documented cushion for that nonlinearity.
    assert!(
        dist60 <= 1.1 * fit.persistence().powi(59) * dist1 + 1e-12,
        "forecast not contracting geometrically: h=1 {dist1}, h=60 {dist60}, a+b = {}",
        fit.persistence()
    );
}

/// ADCC (Cappiello-Engle-Sheppard 2006) nests DCC: on the same (symmetric,
/// no-leverage DGP) data the fitted ADCC log-likelihood is no worse than
/// DCC's up to optimizer slack, the fitted `g` is small, and the
/// stationarity/positivity constraint holds at the estimates.
#[test]
fn adcc_nests_dcc_on_fixture() {
    let fx = load();
    let series = subset(&returns(&fx));
    let dcc = DccGarch::new(spec()).fit(&series).unwrap();
    let adcc = DccGarch::new(spec())
        .with_variant(DccVariant::Adcc)
        .fit(&series)
        .unwrap();

    assert!(adcc.a >= 0.0 && adcc.b >= 0.0 && adcc.g >= 0.0);
    // The feasible set contains every DCC point (g = 0), so the ADCC
    // optimum cannot be materially worse.
    assert!(
        adcc.loglik >= dcc.loglik - 1e-2,
        "ADCC loglik {} vs DCC {}",
        adcc.loglik,
        dcc.loglik
    );
    // The DGP has no asymmetry: g should be small (loose one-realization bar).
    assert!(adcc.g < 0.08, "spurious asymmetry g = {}", adcc.g);
    // Nbar is stored for the fit and the constraint holds strictly.
    assert!(adcc.nbar.is_some());
    assert!(adcc.persistence() < 1.0);
}

/// The Student-t second stage on (Gaussian-innovation) fixture data: the
/// estimated degrees of freedom are large (the t nests the normal as
/// `nu -> infinity`), and the correlation dynamics agree with the Gaussian
/// fit to a loose band.
#[test]
fn student_t_second_stage_on_gaussian_data() {
    let fx = load();
    let series = subset(&returns(&fx));
    let gauss = DccGarch::new(spec()).fit(&series).unwrap();
    let t = DccGarch::new(spec())
        .with_dist(CorrDist::StudentT)
        .fit(&series)
        .unwrap();

    let nu = t.nu.expect("StudentT fit carries nu");
    assert!(nu > 10.0, "Gaussian data should push nu up, got {nu}");
    assert!((t.a - gauss.a).abs() < 0.02, "a {} vs {}", t.a, gauss.a);
    assert!((t.b - gauss.b).abs() < 0.05, "b {} vs {}", t.b, gauss.b);
    // The Gaussian fit reports a Gaussian loglik; the t fit a t loglik.
    assert!(t.loglik.is_finite());
    assert_eq!(t.dist, CorrDist::StudentT);
    assert_eq!(gauss.nu, None);
}

/// Engle-Sheppard (2001) constant-correlation test on the fixture's DCC
/// DGP (a = 0.03, b = 0.95): the diagnostic should reject constant
/// correlation on this single long realization (T = 2400, three pairs).
/// The size/power calibration lives in the Python Monte-Carlo tests; this
/// pins the end-to-end wiring on real fixture data.
#[test]
fn engle_sheppard_rejects_on_dcc_fixture() {
    let fx = load();
    let series = returns(&fx);
    let r = constant_correlation_test(&series, spec(), 5).unwrap();
    assert_eq!(r.df, 6);
    assert_eq!(r.nobs, 2400);
    assert_eq!(r.n_stacked, (2400 - 5) * 3);
    assert!(r.stat.is_finite() && r.stat >= 0.0);
    assert!(
        r.p_value < 0.05,
        "expected rejection on the DCC DGP, got p = {} (stat {})",
        r.p_value,
        r.stat
    );
}
