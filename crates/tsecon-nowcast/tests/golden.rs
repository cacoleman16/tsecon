//! Golden and structural tests for `tsecon-nowcast` against
//! `fixtures/tsecon-nowcast.json`.
//!
//! Two regimes, as documented in the crate:
//!
//! * **Reference-exact (Kalman step).** With the true DGP parameters plugged
//!   into the state space on the raw balanced panel, `smooth_fixed` must
//!   reproduce statsmodels `DynamicFactor`'s Kalman log-likelihood
//!   (`results.llf` from `mod.smooth(params)`) and smoothed states
//!   (`results.states.smoothed`) to ~1e-8. This isolates and validates the
//!   crate's Kalman/state-space step against statsmodels.
//! * **Structural (two-step DGR estimator).** The DGR two-step estimator is a
//!   *different* estimator from one-step MLE, so it is validated by properties,
//!   not by tolerance: its smoothed factor tracks the simulated true factor
//!   (corr > 0.9), the balanced nowcast is finite, and the ragged-edge nowcast
//!   moves in the expected direction.

mod common;

use common::{as_mat, as_vec, assert_rel_close, load_fixture, pearson};
use tsecon_linalg::faer::Mat;
use tsecon_nowcast::{smooth_fixed, DfmParams, Nowcaster};

const FIXTURE: &str = "tsecon-nowcast.json";

/// The raw balanced panel `Y` (T x N).
fn panel() -> Mat<f64> {
    as_mat(&load_fixture(FIXTURE)["panel"])
}

/// The true-DGP parameters as a single-factor [`DfmParams`].
fn dgp_params() -> DfmParams {
    let fx = load_fixture(FIXTURE);
    let loadings_v = as_vec(&fx["dgp"]["loadings"]);
    let phi = as_vec(&fx["dgp"]["phi"]);
    let idio = as_vec(&fx["dgp"]["idiosyncratic"]);
    let n = loadings_v.len();
    let p = phi.len();
    DfmParams {
        loadings: Mat::from_fn(n, 1, |i, _| loadings_v[i]),
        factor_ar: Mat::from_fn(1, p, |_, j| phi[j]),
        factor_cov: Mat::from_fn(1, 1, |_, _| 1.0),
        idiosyncratic: idio,
    }
}

// -------------------------------------------------------------------------
// Reference-exact: the Kalman step at fixed parameters.
// -------------------------------------------------------------------------

#[test]
fn kalman_fixed_loglik_matches_statsmodels() {
    let fx = load_fixture(FIXTURE);
    let params = dgp_params();
    let y = panel();
    let out = smooth_fixed(&params, y.as_ref()).unwrap();
    let expected = fx["kalman_fixed"]["loglik"].as_f64().unwrap();
    assert_rel_close(out.loglik, expected, 1e-8, "kalman_fixed loglik");
}

#[test]
fn kalman_fixed_smoothed_state_matches_statsmodels() {
    let fx = load_fixture(FIXTURE);
    let params = dgp_params();
    let y = panel();
    let out = smooth_fixed(&params, y.as_ref()).unwrap();
    let expected = as_mat(&fx["kalman_fixed"]["smoothed_state"]); // T x P
    assert_eq!(out.smoothed_state.len(), expected.nrows());
    let m = expected.ncols();
    for t in 0..expected.nrows() {
        assert_eq!(out.smoothed_state[t].len(), m);
        for j in 0..m {
            assert_rel_close(
                out.smoothed_state[t][j],
                expected[(t, j)],
                1e-8,
                &format!("smoothed_state[{t}][{j}]"),
            );
        }
    }
}

#[test]
fn kalman_fixed_smoothed_factor_is_state_column_zero() {
    // The smoothed factor is state-column 0, matching statsmodels' column 0.
    let fx = load_fixture(FIXTURE);
    let params = dgp_params();
    let y = panel();
    let out = smooth_fixed(&params, y.as_ref()).unwrap();
    let expected = as_mat(&fx["kalman_fixed"]["smoothed_state"]);
    for t in 0..expected.nrows() {
        assert_rel_close(
            out.smoothed_factors[t][0],
            expected[(t, 0)],
            1e-8,
            &format!("smoothed_factor[{t}]"),
        );
    }
}

// -------------------------------------------------------------------------
// Structural: the two-step DGR estimator.
// -------------------------------------------------------------------------

#[test]
fn two_step_smoothed_factor_tracks_true_factor() {
    let fx = load_fixture(FIXTURE);
    let y = panel();
    let p = as_vec(&fx["dgp"]["phi"]).len();
    let nc = Nowcaster::fit_two_step(y.as_ref(), 1, p).unwrap();

    let true_f = as_vec(&fx["true_factor"]);
    let sm: Vec<f64> = nc.smoothed_factors().iter().map(|s| s[0]).collect();
    assert_eq!(sm.len(), true_f.len());
    // Factor identified only up to sign; compare the absolute correlation.
    let corr = pearson(&sm, &true_f).abs();
    assert!(
        corr > 0.9,
        "two-step smoothed factor should track the true factor: corr = {corr}"
    );
    assert!(nc.loglik().is_finite(), "training loglik must be finite");
}

#[test]
fn two_step_balanced_nowcast_is_finite() {
    let y = panel();
    let nc = Nowcaster::fit_two_step(y.as_ref(), 1, 2).unwrap();
    let res = nc.nowcast_panel(y.as_ref()).unwrap();
    assert_eq!(res.values.len(), nc.n_series());
    for (i, v) in res.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "balanced nowcast for series {i} must be finite"
        );
    }
    // The edge nowcasts should be in the same ballpark as the last raw
    // observations (levels, de-standardized): finite and not absurd.
    for v in &res.values {
        assert!(v.abs() < 1e3, "nowcast unreasonably large: {v}");
    }
}

#[test]
fn ragged_edge_nowcast_moves_in_expected_direction() {
    // Dropping (vs. keeping) a large positive reading on a positive-loading
    // "driver" series at the edge lowers a positive-loading target's nowcast:
    // the missing observation removes its upward pull on the smoothed factor.
    let y = panel();
    let nc = Nowcaster::fit_two_step(y.as_ref(), 1, 2).unwrap();

    // Driver series 6 has loading +1.1, target series 0 has loading +1.0.
    let driver = 6usize;
    let target = 0usize;
    assert!(nc.params().loadings[(driver, 0)] > 0.0);
    assert!(nc.params().loadings[(target, 0)] > 0.0);

    let t_last = y.nrows() - 1;

    // "Present": inflate the driver's final reading to a large positive value.
    let mut present = y.clone();
    present[(t_last, driver)] = 25.0;
    let present_nc = nc.nowcast_series(present.as_ref(), target).unwrap();

    // "Dropped": the driver's final reading is missing (ragged edge).
    let mut dropped = y.clone();
    dropped[(t_last, driver)] = f64::NAN;
    let dropped_nc = nc.nowcast_series(dropped.as_ref(), target).unwrap();

    assert!(present_nc.is_finite() && dropped_nc.is_finite());
    assert!(
        present_nc > dropped_nc,
        "keeping a large positive driver reading should raise the target \
         nowcast relative to dropping it: present {present_nc} vs dropped {dropped_nc}"
    );
}

#[test]
fn ragged_filter_runs_with_multiple_missing_at_edge() {
    // The whole point of nowcasting: the filter still runs and yields finite
    // nowcasts when several series are missing their final observations.
    let y = panel();
    let nc = Nowcaster::fit_two_step(y.as_ref(), 1, 2).unwrap();
    let t_last = y.nrows() - 1;
    let mut ragged = y.clone();
    // Half the panel has not reported the final month yet.
    for j in (0..y.ncols()).step_by(2) {
        ragged[(t_last, j)] = f64::NAN;
    }
    let res = nc.nowcast_panel(ragged.as_ref()).unwrap();
    for (i, v) in res.values.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ragged nowcast for series {i} must be finite"
        );
    }
    // The smoothed edge factor is still finite.
    assert!(res.edge_factor.iter().all(|f| f.is_finite()));
}

// -------------------------------------------------------------------------
// Guardrails.
// -------------------------------------------------------------------------

#[test]
fn smooth_fixed_rejects_wrong_column_count() {
    let params = dgp_params();
    let bad: Mat<f64> = Mat::zeros(10, 3); // params expect N = 8 columns
    let err = smooth_fixed(&params, bad.as_ref());
    assert!(err.is_err());
}

#[test]
fn nowcast_rejects_out_of_range_series() {
    let y = panel();
    let nc = Nowcaster::fit_two_step(y.as_ref(), 1, 2).unwrap();
    let n = nc.n_series();
    let err = nc.nowcast_series(y.as_ref(), n);
    assert!(err.is_err());
}
