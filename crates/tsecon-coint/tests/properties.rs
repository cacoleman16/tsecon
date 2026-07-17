//! Property tests: cointegration-rank recovery and error-correction signs
//! on the simulated system, the VECM-to-level-VAR round trip, and the
//! Engle-Granger two-step on the same data.

mod common;

use common::{as_endog, load_fixture};
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{
    engle_granger, fit_vecm, johansen, EngleGrangerTrend, SignificanceLevel, VecmResult,
};
use tsecon_diag::AdfLagSelection;

/// The system is two cointegrated I(1) series plus one stationary series,
/// so the true cointegration rank is 2. Both the trace and
/// maximum-eigenvalue sequential tests recover rank 2 at the 1% level, and
/// the eigenvalue spectrum is clearly "two large, one small".
#[test]
fn johansen_recovers_rank_two() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let res = johansen(endog.as_ref(), 2).unwrap();

    assert_eq!(res.rank_trace(SignificanceLevel::One), Some(2));
    assert_eq!(res.rank_max_eig(SignificanceLevel::One), Some(2));
    // At 5% the evidence is at least as strong (the third eigenvalue is
    // borderline for this sample).
    assert!(res.rank_trace(SignificanceLevel::Five).unwrap() >= 2);

    // Two dominant eigenvalues, one near the unit-root boundary.
    assert!(res.eig[1] > 0.05, "second eigenvalue should be sizeable");
    assert!(res.eig[2] < 0.02, "third eigenvalue should be near zero");
}

/// The rank-1 error-correction loading on the first (normalized) variable
/// is negative — the classic stabilizing sign: when the cointegrating
/// relation is positive, the first series is pulled back down.
#[test]
fn alpha_has_error_correction_sign() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let res = fit_vecm(endog.as_ref(), 2, 1).unwrap();
    assert!(
        res.alpha[(0, 0)] < 0.0,
        "alpha[0] = {} should be negative (error correction)",
        res.alpha[(0, 0)]
    );
    // beta is normalized so the leading block is the identity.
    assert!((res.beta[(0, 0)] - 1.0).abs() < 1e-12);
}

/// The VECM-to-level-VAR mapping inverts exactly on the fitted system:
/// summing the recovered level coefficients returns `I + Pi`, and the
/// telescoping partial sums return the short-run `Gamma_i`.
#[test]
fn vecm_to_var_round_trip_on_fit() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let res = fit_vecm(endog.as_ref(), 2, 1).unwrap();
    let k = res.neqs;
    let p = res.k_ar_diff + 1;
    let coefs = res.var_coefs();
    assert_eq!(coefs.len(), p);

    // Pi = sum_j A_j - I.
    let pi = res.pi();
    let mut pi_rt = Mat::<f64>::zeros(k, k);
    for a in &coefs {
        pi_rt += a;
    }
    for i in 0..k {
        pi_rt[(i, i)] -= 1.0;
    }
    for i in 0..k {
        for j in 0..k {
            assert!((pi_rt[(i, j)] - pi[(i, j)]).abs() < 1e-10);
        }
    }

    // Gamma_i = -(A_{i+1} + ... + A_p).
    for gi in 1..=res.k_ar_diff {
        let gamma_i = res.gamma_lag(gi).unwrap();
        let mut rt = Mat::<f64>::zeros(k, k);
        for a in coefs.iter().skip(gi) {
            rt += a;
        }
        for i in 0..k {
            for j in 0..k {
                assert!((-rt[(i, j)] - gamma_i[(i, j)]).abs() < 1e-10);
            }
        }
    }

    // Companion has the right shape.
    let comp = res.companion().unwrap();
    assert_eq!(comp.nrows(), k * p);
    assert_eq!(comp.ncols(), k * p);
}

/// The VECM-to-VAR mapping inverts on a hand-built system with known
/// `alpha`, `beta`, `Gamma` (so the level coefficients are known too).
#[test]
fn vecm_to_var_round_trip_synthetic() {
    // k = 2, one cointegrating vector, one lagged difference.
    let alpha = Mat::from_fn(2, 1, |i, _| [-0.4, 0.1][i]);
    let beta = Mat::from_fn(2, 1, |i, _| [1.0, -0.8][i]);
    let gamma = Mat::from_fn(2, 2, |i, j| [[0.3, -0.1], [0.05, 0.2]][i][j]);
    let res = VecmResult {
        neqs: 2,
        nobs: 100,
        k_ar_diff: 1,
        coint_rank: 1,
        alpha: alpha.clone(),
        beta: beta.clone(),
        gamma: gamma.clone(),
        sigma_u: Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 0.0 }),
        eig: vec![0.3, 0.0],
        llf: 0.0,
    };
    let coefs = res.var_coefs();
    // A_1 = I + Pi + Gamma_1, A_2 = -Gamma_1.
    let pi = &alpha * beta.transpose();
    for i in 0..2 {
        for j in 0..2 {
            let ident = if i == j { 1.0 } else { 0.0 };
            let a1 = ident + pi[(i, j)] + gamma[(i, j)];
            assert!((coefs[0][(i, j)] - a1).abs() < 1e-12);
            assert!((coefs[1][(i, j)] + gamma[(i, j)]).abs() < 1e-12);
        }
    }
}

/// Engle-Granger step 1 fits a sensible cointegrating vector and step 2
/// finds strong evidence against a unit root in the residuals (a large
/// negative ADF statistic) on this cointegrated system.
#[test]
fn engle_granger_runs_and_rejects_unit_root() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let res = engle_granger(
        endog.as_ref(),
        EngleGrangerTrend::Constant,
        AdfLagSelection::Aic(None),
    )
    .unwrap();

    // [const, coef on series 1, coef on series 2].
    assert_eq!(res.coint_coefs.len(), 3);
    assert_eq!(res.resid.len(), endog.nrows());
    // Residuals are stationary here, so the ADF tau is strongly negative
    // (well past even the conservative standard-ADF 1% level ~ -3.4).
    assert!(
        res.statistic() < -3.5,
        "Engle-Granger ADF statistic {} should be strongly negative",
        res.statistic()
    );
}
