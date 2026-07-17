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
use tsecon_mgarch::{CccGarch, DccGarch};

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
