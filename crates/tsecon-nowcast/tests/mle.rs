//! Golden and property tests for the one-step Gaussian MLE
//! (`Nowcaster::fit_mle`) against `fixtures/nowcast_mle.json`.
//!
//! The fixture is generated independently by fitting statsmodels'
//! `DynamicFactor(k_factors=1, factor_order=p, error_order=0)` MLE on a
//! centred simulated panel (`fixtures/generate_nowcast_mle_fixtures.py`).
//!
//! Two regimes, exactly as documented on [`tsecon_nowcast::mle`]:
//!
//! * **Reference-exact (tight, ~1e-6).** Given statsmodels' *fitted*
//!   parameters, the crate's `smooth_fixed` reproduces statsmodels' maximised
//!   `llf` and its smoothed factor on the same centred panel. This confirms the
//!   Kalman path at the MLE optimum, not just at the DGP parameters.
//! * **Optimum (honest gap).** `fit_mle` and statsmodels maximise the same
//!   function; the achieved llf lands within a small, reported tolerance of
//!   statsmodels' `llf`. Parameter vectors are *not* tolerance-matched (sign /
//!   optimiser ambiguity); instead the crate asserts `llf(MLE) >= llf(two-step)`
//!   and, as a property, that the MLE smoothed factor tracks the simulated
//!   truth (`|corr| > 0.9`) and recovers the AR persistence.

mod common;

use std::sync::OnceLock;

use common::{as_mat, as_vec, assert_rel_close, load_fixture, pearson};
use tsecon_linalg::faer::Mat;
use tsecon_nowcast::{smooth_fixed, DfmParams, Nowcaster};

const FIXTURE: &str = "nowcast_mle.json";

/// The one-step MLE fit on the fixture panel, computed once and shared across
/// tests (the optimisation is the expensive part; it is identical for every
/// test that consumes it).
fn fitted_mle() -> &'static Nowcaster {
    static FIT: OnceLock<Nowcaster> = OnceLock::new();
    FIT.get_or_init(|| {
        let y = panel();
        let p = factor_order();
        Nowcaster::fit_mle(y.as_ref(), p).expect("fit_mle should succeed on the fixture panel")
    })
}

/// The raw balanced panel `Y` (T x N).
fn panel() -> Mat<f64> {
    as_mat(&load_fixture(FIXTURE)["panel"])
}

/// The stored column means (length N).
fn center() -> Vec<f64> {
    as_vec(&load_fixture(FIXTURE)["center"])
}

/// The centred panel `Yc = Y - center` (the panel statsmodels fit on and the
/// panel `fit_mle` centres to internally).
fn centered_panel() -> Mat<f64> {
    let y = panel();
    let c = center();
    Mat::from_fn(y.nrows(), y.ncols(), |i, j| y[(i, j)] - c[j])
}

/// statsmodels' fitted parameters as a single-factor [`DfmParams`].
fn mle_fitted_params() -> DfmParams {
    let fx = load_fixture(FIXTURE);
    let loadings = as_vec(&fx["mle_fitted"]["loadings"]);
    let ar = as_vec(&fx["mle_fitted"]["factor_ar"]);
    let idio = as_vec(&fx["mle_fitted"]["idiosyncratic"]);
    let n = loadings.len();
    let p = ar.len();
    DfmParams {
        loadings: Mat::from_fn(n, 1, |i, _| loadings[i]),
        factor_ar: Mat::from_fn(1, p, |_, j| ar[j]),
        factor_cov: Mat::from_fn(1, 1, |_, _| 1.0),
        idiosyncratic: idio,
    }
}

fn factor_order() -> usize {
    load_fixture(FIXTURE)["dims"]["factor_order"]
        .as_u64()
        .unwrap() as usize
}

// -------------------------------------------------------------------------
// Reference-exact: the Kalman step at statsmodels' MLE-fitted parameters.
// -------------------------------------------------------------------------

#[test]
fn smooth_fixed_reproduces_statsmodels_mle_loglik() {
    let fx = load_fixture(FIXTURE);
    let params = mle_fitted_params();
    let yc = centered_panel();
    let out = smooth_fixed(&params, yc.as_ref()).unwrap();
    let expected = fx["mle_fitted"]["loglik"].as_f64().unwrap();
    assert_rel_close(out.loglik, expected, 1e-6, "mle_fitted loglik");
}

#[test]
fn smooth_fixed_reproduces_statsmodels_mle_smoothed_factor() {
    let fx = load_fixture(FIXTURE);
    let params = mle_fitted_params();
    let yc = centered_panel();
    let out = smooth_fixed(&params, yc.as_ref()).unwrap();
    let expected = as_vec(&fx["mle_fitted"]["smoothed_factor"]);
    assert_eq!(out.smoothed_factors.len(), expected.len());
    for (t, &e) in expected.iter().enumerate() {
        assert_rel_close(
            out.smoothed_factors[t][0],
            e,
            1e-6,
            &format!("mle_fitted smoothed_factor[{t}]"),
        );
    }
}

// -------------------------------------------------------------------------
// The MLE optimum: fit_mle vs statsmodels (honest gap) and properties.
// -------------------------------------------------------------------------

#[test]
fn fit_mle_reaches_statsmodels_loglik() {
    let fx = load_fixture(FIXTURE);
    let nc = fitted_mle();

    let sm_llf = fx["mle_fitted"]["loglik"].as_f64().unwrap();
    let gap = nc.loglik() - sm_llf;
    println!(
        "fit_mle loglik = {}, statsmodels llf = {}, gap = {:.3e}",
        nc.loglik(),
        sm_llf,
        gap
    );
    // Same likelihood, different optimisers: the crate should land within a
    // small tolerance of statsmodels' maximum (and may exceed it slightly).
    let scale = sm_llf.abs().max(1.0);
    assert!(
        gap.abs() <= 1e-2 * scale,
        "fit_mle llf {} should be within 1e-2 (rel) of statsmodels {} (gap {:.3e})",
        nc.loglik(),
        sm_llf,
        gap
    );
}

#[test]
fn fit_mle_loglik_at_least_two_step() {
    // The MLE is the maximum of the same likelihood and is started from the
    // two-step estimate, so on the same centred panel it cannot do worse.
    let nc = fitted_mle();
    let two_step_ref = nc
        .two_step_reference_loglik()
        .expect("an MLE fit exposes the two-step reference loglik");
    println!(
        "fit_mle loglik = {}, two-step reference = {}, improvement = {:.3e}",
        nc.loglik(),
        two_step_ref,
        nc.loglik() - two_step_ref
    );
    assert!(
        nc.loglik() >= two_step_ref - 1e-6,
        "MLE loglik {} must be >= two-step loglik {} on the same centred panel",
        nc.loglik(),
        two_step_ref
    );
}

#[test]
fn fit_mle_smoothed_factor_tracks_true_factor() {
    let fx = load_fixture(FIXTURE);
    let nc = fitted_mle();

    let true_f = as_vec(&fx["true_factor"]);
    let sm: Vec<f64> = nc.smoothed_factors().iter().map(|s| s[0]).collect();
    assert_eq!(sm.len(), true_f.len());
    // Factor identified only up to sign; compare absolute correlation.
    let corr = pearson(&sm, &true_f).abs();
    println!("|corr(MLE smoothed factor, true factor)| = {corr:.4}");
    assert!(
        corr > 0.9,
        "MLE smoothed factor should track the true factor: |corr| = {corr}"
    );
}

#[test]
fn fit_mle_recovers_ar_persistence() {
    // The sum of the fitted AR coefficients (spectral persistence at lag 0)
    // should be near the DGP's; this is a Monte-Carlo band, not a tight match.
    let fx = load_fixture(FIXTURE);
    let p = factor_order();
    let nc = fitted_mle();

    let dgp_phi = as_vec(&fx["dgp"]["phi"]);
    let dgp_sum: f64 = dgp_phi.iter().sum();
    let ar = &nc.params().factor_ar;
    let fit_sum: f64 = (0..p).map(|j| ar[(0, j)]).sum();
    println!("AR persistence: fitted sum = {fit_sum:.4}, DGP sum = {dgp_sum:.4}");
    assert!(
        (fit_sum - dgp_sum).abs() < 0.2,
        "fitted AR persistence {fit_sum} should be within a MC band of DGP {dgp_sum}"
    );
}

#[test]
fn fit_mle_nowcast_is_finite_and_reasonable() {
    // The MLE nowcaster produces finite edge nowcasts (balanced and ragged).
    let y = panel();
    let p = factor_order();
    let nc = fitted_mle();
    assert_eq!(nc.n_factors(), 1);
    assert_eq!(nc.factor_order(), p);

    let res = nc.nowcast_panel(y.as_ref()).unwrap();
    assert_eq!(res.values.len(), nc.n_series());
    for (i, v) in res.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "balanced nowcast for series {i} must be finite"
        );
        assert!(v.abs() < 1e3, "nowcast unreasonably large: {v}");
    }

    // Ragged edge: drop the final observation of half the series.
    let t_last = y.nrows() - 1;
    let mut ragged = y.clone();
    for j in (0..y.ncols()).step_by(2) {
        ragged[(t_last, j)] = f64::NAN;
    }
    let rres = nc.nowcast_panel(ragged.as_ref()).unwrap();
    assert!(rres.values.iter().all(|v| v.is_finite()));
    assert!(rres.edge_factor.iter().all(|f| f.is_finite()));
}

// -------------------------------------------------------------------------
// Guardrails.
// -------------------------------------------------------------------------

#[test]
fn fit_mle_rejects_ragged_training_panel() {
    let mut y = panel();
    y[(0, 0)] = f64::NAN;
    let err = Nowcaster::fit_mle(y.as_ref(), 2);
    assert!(err.is_err());
}

#[test]
fn fit_mle_rejects_zero_factor_order() {
    let y = panel();
    let err = Nowcaster::fit_mle(y.as_ref(), 0);
    assert!(err.is_err());
}
