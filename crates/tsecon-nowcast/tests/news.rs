//! Tests for the Banbura-Modugno (2014) NEWS / update decomposition.
//!
//! Two regimes:
//!
//! * **Self-validating exact identity.** The decomposition is an EXACT
//!   identity: `new_nowcast - old_nowcast == Σ_j contribution_j`, and each
//!   `news_j == actual_j - (old-vintage Kalman forecast of cell j)`. This is
//!   the strongest possible check and needs no external golden — it is asserted
//!   to machine precision (~1e-10) both on a two-step-fitted [`Nowcaster`] and
//!   on the fixture design.
//! * **Independent finite-difference cross-check.** The crate computes the
//!   Kalman weights ANALYTICALLY (a unit-impulse smoother pass). The fixture
//!   `nowcast_news.json` supplies weights computed the DIRECT way — a central
//!   finite difference on an independent NumPy Kalman/RTS smoother — and the
//!   analytic weights must match them to ~1e-6.

mod common;

use common::{as_mat_nan, as_vec, assert_rel_close, load_fixture};
use tsecon_linalg::faer::{Mat, MatRef};
use tsecon_nowcast::{news_decomposition_at, DfmParams, Nowcaster};

const FIXTURE: &str = "nowcast_news.json";

/// The single-factor [`DfmParams`] stored in the fixture.
fn fixture_params() -> DfmParams {
    let fx = load_fixture(FIXTURE);
    let loadings_v = as_vec(&fx["params"]["loadings"]);
    let phi = as_vec(&fx["params"]["phi"]);
    let idio = as_vec(&fx["params"]["idiosyncratic"]);
    let q = fx["params"]["factor_innovation_var"].as_f64().unwrap();
    let n = loadings_v.len();
    let p = phi.len();
    DfmParams {
        loadings: Mat::from_fn(n, 1, |i, _| loadings_v[i]),
        factor_ar: Mat::from_fn(1, p, |_, j| phi[j]),
        factor_cov: Mat::from_fn(1, 1, |_, _| q),
        idiosyncratic: idio,
    }
}

// -------------------------------------------------------------------------
// Reference cross-check against the independent NumPy Kalman/RTS smoother.
// -------------------------------------------------------------------------

#[test]
fn news_decomposition_matches_finite_difference_reference() {
    let fx = load_fixture(FIXTURE);
    let params = fixture_params();
    let center = as_vec(&fx["center"]);
    let scale = as_vec(&fx["scale"]);
    let old_v = as_mat_nan(&fx["old_vintage"]);
    let new_v = as_mat_nan(&fx["new_vintage"]);
    let target_series = fx["target_series"].as_u64().unwrap() as usize;
    let target_period = fx["target_period"].as_u64().unwrap() as usize;

    let dec = news_decomposition_at(
        &params,
        &center,
        &scale,
        old_v.as_ref(),
        new_v.as_ref(),
        target_series,
        target_period,
    )
    .unwrap();

    let g = &fx["golden"];
    assert_rel_close(
        dec.old_nowcast,
        g["old_nowcast"].as_f64().unwrap(),
        1e-7,
        "old_nowcast",
    );
    assert_rel_close(
        dec.new_nowcast,
        g["new_nowcast"].as_f64().unwrap(),
        1e-7,
        "new_nowcast",
    );
    assert_rel_close(
        dec.total_revision,
        g["total_revision"].as_f64().unwrap(),
        1e-7,
        "total_revision",
    );

    let golden = g["contributions"].as_array().unwrap();
    assert_eq!(
        dec.contributions.len(),
        golden.len(),
        "number of newly-observed cells"
    );
    for (c, gj) in dec.contributions.iter().zip(golden) {
        assert_eq!(c.series, gj["series"].as_u64().unwrap() as usize);
        assert_eq!(c.period, gj["period"].as_u64().unwrap() as usize);
        assert_rel_close(c.actual, gj["actual"].as_f64().unwrap(), 1e-9, "actual");
        assert_rel_close(
            c.forecast,
            gj["forecast"].as_f64().unwrap(),
            1e-7,
            "forecast",
        );
        assert_rel_close(c.news, gj["news"].as_f64().unwrap(), 1e-7, "news");
        // The crate's ANALYTIC Kalman weight vs. the reference FINITE-DIFFERENCE
        // weight: the headline cross-check.
        assert_rel_close(c.weight, gj["weight"].as_f64().unwrap(), 1e-6, "weight");
        assert_rel_close(
            c.contribution,
            gj["contribution"].as_f64().unwrap(),
            1e-6,
            "contribution",
        );
    }
}

// -------------------------------------------------------------------------
// The self-validating exact identities.
// -------------------------------------------------------------------------

#[test]
fn adding_up_identity_holds_on_fixture_design() {
    let fx = load_fixture(FIXTURE);
    let params = fixture_params();
    let center = as_vec(&fx["center"]);
    let scale = as_vec(&fx["scale"]);
    let old_v = as_mat_nan(&fx["old_vintage"]);
    let new_v = as_mat_nan(&fx["new_vintage"]);
    let target_series = fx["target_series"].as_u64().unwrap() as usize;
    let target_period = fx["target_period"].as_u64().unwrap() as usize;

    let dec = news_decomposition_at(
        &params,
        &center,
        &scale,
        old_v.as_ref(),
        new_v.as_ref(),
        target_series,
        target_period,
    )
    .unwrap();

    // EXACT identity: total revision == sum of contributions (~machine eps).
    let sum: f64 = dec.contributions.iter().map(|c| c.contribution).sum();
    assert!(
        (sum - dec.total_revision).abs() <= 1e-10,
        "adding-up identity: Σcontribution {sum} vs total_revision {}",
        dec.total_revision
    );
    // Each contribution == weight * news, exactly by construction.
    for c in &dec.contributions {
        assert!((c.contribution - c.weight * c.news).abs() <= 1e-12);
    }
}

/// Builds a small balanced training panel and fits a two-step Nowcaster, then
/// checks the identity on ragged old/new vintages derived from a held-out row.
#[test]
fn adding_up_identity_holds_on_fitted_nowcaster() {
    // Balanced training panel from the main nowcast fixture.
    let train = as_mat_nan(&load_fixture("tsecon-nowcast.json")["panel"]);
    let nc = Nowcaster::fit_two_step(train.as_ref(), 1, 2).unwrap();
    let n = nc.n_series();

    // A short ragged panel: reuse the last rows of the training data and carve
    // out a ragged edge. New reveals more of the final row than old.
    let t = train.nrows();
    let sub_rows = 12;
    let base: Mat<f64> = Mat::from_fn(sub_rows, n, |i, j| train[(t - sub_rows + i, j)]);

    let last = sub_rows - 1;
    // New vintage: final row observes series 0..n-2, last series missing.
    let mut new_v = base.clone();
    new_v[(last, n - 1)] = f64::NAN;
    // Old vintage: final row observes only series 0..n/2, the rest missing.
    let mut old_v = new_v.clone();
    for j in (n / 2)..n {
        old_v[(last, j)] = f64::NAN;
    }

    let target_series = n - 1; // missing at the edge in both -> a true nowcast
    let dec = nc
        .news_decomposition(old_v.as_ref(), new_v.as_ref(), target_series, last)
        .unwrap();

    assert!(
        !dec.contributions.is_empty(),
        "expected newly-observed cells"
    );
    let sum: f64 = dec.contributions.iter().map(|c| c.contribution).sum();
    assert!(
        (sum - dec.total_revision).abs() <= 1e-10,
        "adding-up identity on fitted model: Σcontribution {sum} vs revision {}",
        dec.total_revision
    );

    // Each news equals actual minus the old-vintage nowcast (common-component
    // projection) of that same cell -- verified independently here by nowcasting
    // each newly-observed series at its period on the OLD vintage.
    let old_res = nc.nowcast_panel(old_v.as_ref()).unwrap();
    let old_factors = &old_res.smoothing.smoothed_factors;
    for c in &dec.contributions {
        // Independent recomputation of the old-vintage forecast of the cell.
        let f = &old_factors[c.period];
        let mut acc = 0.0;
        for (k, &fk) in f.iter().enumerate() {
            acc += nc.params().loadings[(c.series, k)] * fk;
        }
        let forecast = nc.center()[c.series] + nc.scale()[c.series] * acc;
        assert_rel_close(c.forecast, forecast, 1e-10, "old-vintage forecast");
        assert_rel_close(
            c.news,
            c.actual - forecast,
            1e-10,
            "news = actual - forecast",
        );
    }
}

// -------------------------------------------------------------------------
// Guardrails.
// -------------------------------------------------------------------------

#[test]
fn rejects_mismatched_vintage_shapes() {
    let params = fixture_params();
    let n = params.n_series();
    let center = vec![0.0; n];
    let scale = vec![1.0; n];
    let old_v: Mat<f64> = Mat::zeros(5, n);
    let new_v: Mat<f64> = Mat::zeros(6, n);
    let err = news_decomposition_at(
        &params,
        &center,
        &scale,
        old_v.as_ref(),
        new_v.as_ref(),
        0,
        0,
    );
    assert!(err.is_err());
}

#[test]
fn rejects_changed_old_observation() {
    let params = fixture_params();
    let n = params.n_series();
    let center = vec![0.0; n];
    let scale = vec![1.0; n];
    let old_v: Mat<f64> = Mat::from_fn(4, n, |i, j| (i + j) as f64);
    // Change a cell that is observed in the old vintage -> not pure news.
    let mut new_v = old_v.clone();
    new_v[(0, 0)] += 1.0;
    let err = news_decomposition_at(
        &params,
        &center,
        &scale,
        old_v.as_ref(),
        new_v.as_ref(),
        0,
        3,
    );
    assert!(err.is_err());
}

#[test]
fn rejects_out_of_range_target() {
    let params = fixture_params();
    let n = params.n_series();
    let center = vec![0.0; n];
    let scale = vec![1.0; n];
    let v: Mat<f64> = Mat::zeros(4, n);
    // target_series out of range.
    assert!(bad(&params, &center, &scale, v.as_ref(), n, 0).is_err());
    // target_period out of range.
    assert!(bad(&params, &center, &scale, v.as_ref(), 0, 4).is_err());
    // non-positive scale.
    let mut bad_scale = scale.clone();
    bad_scale[0] = 0.0;
    assert!(bad(&params, &center, &bad_scale, v.as_ref(), 0, 0).is_err());
}

fn bad(
    params: &DfmParams,
    center: &[f64],
    scale: &[f64],
    v: MatRef<'_, f64>,
    ts: usize,
    tp: usize,
) -> Result<tsecon_nowcast::NewsDecomposition, tsecon_nowcast::NowcastError> {
    news_decomposition_at(params, center, scale, v, v, ts, tp)
}
