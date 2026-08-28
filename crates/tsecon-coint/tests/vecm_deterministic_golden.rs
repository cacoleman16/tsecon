//! Golden and reconciliation tests for the VECM deterministic-terms cases
//! (field item 12 and its restricted-cases follow-up) against
//! `fixtures/vecm_deterministic.json`.
//!
//! Dataset 1 (top level): seeded *drifting* cointegrated data (rank 1)
//! where statsmodels `VECM(deterministic = "n")` and
//! `VECM(deterministic = "co")` pin the two original cases, and
//! `coint_johansen(det_order = 0)` arbitrates that the
//! unrestricted-constant fit spans exactly the cointegrating space the
//! Johansen rank test works in — while the no-deterministic fit visibly
//! does not (the reporter's beta-cosine divergence, pinned as documented
//! behavior).
//!
//! Dataset 2 (`trending` block): *trending* cointegrated data whose
//! equilibrium relation is stationary around a constant plus a linear
//! trend — statsmodels pins **every** deterministic case (`"n"`, `"co"`,
//! `"ci"`, `"lo"`, `"li"`, `"colo"`, `"coli"`, `"cilo"`, `"cili"`),
//! including the restricted rows `det_coef_coint` of the widened
//! cointegrating matrix, plus `coint_johansen(det_order = 1)` and the
//! measured cross-case beta cosines (the case choice changes the answer).
//!
//! Dataset 3 (`seasonal` block): a quarterly cointegrated pair pinning
//! the centered-seasonal-dummy machinery (`seasons = 4`, including a
//! nonzero `first_season`).

mod common;

use common::{as_endog, as_mat, as_vec, assert_mat_close, assert_rel_close, load_fixture, num};
use serde_json::Value;
use tsecon_coint::tsecon_linalg::faer::Mat;
use tsecon_coint::{
    fit_vecm, fit_vecm_det, fit_vecm_seasonal, johansen, CointError, VecmDeterministic,
};

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn beta_col0(beta: &Mat<f64>) -> Vec<f64> {
    (0..beta.nrows()).map(|i| beta[(i, 0)]).collect()
}

/// `deterministic = "n"`: alpha, beta, gamma, sigma_u, llf match
/// statsmodels `VECM(..., deterministic = "n")` to 1e-6 relative, and
/// `det_coef` is `k x 0`. Also pins that `fit_vecm` (the historical
/// three-argument entry point) is exactly this case.
#[test]
fn golden_vecm_deterministic_none() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let vb = &fx["vecm_n"];

    let res = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::None).unwrap();
    assert_eq!(res.neqs, 3);
    assert_eq!(res.nobs, 398);
    assert_eq!(res.deterministic, VecmDeterministic::None);
    assert_eq!(res.det_coef.ncols(), 0);
    assert_mat_close(&res.alpha, &vb["alpha"], 1e-6, "alpha (n)");
    assert_mat_close(&res.beta, &vb["beta"], 1e-6, "beta (n)");
    assert_mat_close(&res.gamma, &vb["gamma"], 1e-6, "gamma (n)");
    assert_mat_close(&res.sigma_u, &vb["sigma_u"], 1e-6, "sigma_u (n)");
    assert_rel_close(res.llf, num(&vb["llf"]), 1e-6, "llf (n)");

    // The default entry point is bit-identical to the explicit "n" case.
    let default = fit_vecm(endog.as_ref(), 1, 1).unwrap();
    for i in 0..3 {
        assert_eq!(default.beta[(i, 0)], res.beta[(i, 0)], "fit_vecm != n");
    }
    assert_eq!(default.llf, res.llf, "fit_vecm llf != n llf");
}

/// `deterministic = "co"` (unrestricted constant): alpha, beta, gamma,
/// det_coef, sigma_u, llf match statsmodels `VECM(..., deterministic =
/// "co")` to 1e-6 relative.
#[test]
fn golden_vecm_deterministic_constant() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let vb = &fx["vecm_co"];

    let res = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    assert_eq!(res.deterministic, VecmDeterministic::Constant);
    assert_mat_close(&res.alpha, &vb["alpha"], 1e-6, "alpha (co)");
    assert_mat_close(&res.beta, &vb["beta"], 1e-6, "beta (co)");
    assert_mat_close(&res.gamma, &vb["gamma"], 1e-6, "gamma (co)");
    assert_mat_close(&res.det_coef, &vb["det_coef"], 1e-6, "det_coef (co)");
    assert_mat_close(&res.sigma_u, &vb["sigma_u"], 1e-6, "sigma_u (co)");
    assert_rel_close(res.llf, num(&vb["llf"]), 1e-6, "llf (co)");
}

/// The reconciliation the field reporter asked for: the
/// unrestricted-constant VECM solves the same reduced-rank eigenproblem
/// as the Johansen rank test (`det_order = 0`), so its eigenvalues equal
/// the test's to 1e-8 and its beta spans the same direction as the
/// normalized first Johansen eigenvector (cosine 1 to 1e-10) — while the
/// no-deterministic default's beta diverges on this drifting draw exactly
/// as the fixture pins (cosine ~0.63; the reporter measured ~0.57 on
/// their data).
#[test]
fn vecm_constant_reconciles_with_johansen() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);

    let joh = johansen(endog.as_ref(), 1).unwrap();
    let co = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    let none = fit_vecm(endog.as_ref(), 1, 1).unwrap();

    // Same canonical-correlation eigenvalues.
    for i in 0..3 {
        assert_rel_close(
            co.eig[i],
            joh.eig[i],
            1e-8,
            &format!("eig[{i}] co vs johansen"),
        );
    }
    // Same cointegrating direction: normalize the first Johansen
    // eigenvector as the VECM does (leading entry 1).
    let joh_beta: Vec<f64> = (0..3)
        .map(|i| joh.evec[(i, 0)] / joh.evec[(0, 0)])
        .collect();
    let cos_co = cosine(&beta_col0(&co.beta), &joh_beta);
    assert!(
        cos_co.abs() > 1.0 - 1e-10,
        "cosine(beta_co, beta_johansen) = {cos_co}, expected ~1"
    );

    // And the documented divergence of the "n" default (the defect the
    // field report demonstrated): pinned by the fixture.
    let cos_n = cosine(&beta_col0(&none.beta), &beta_col0(&co.beta));
    let pinned = num(&fx["beta_cosine_n_co"]);
    assert_rel_close(cos_n, pinned, 1e-6, "cosine(beta_n, beta_co)");
    assert!(
        cos_n < 0.8,
        "cosine(beta_n, beta_co) = {cos_n}: the two deterministic cases \
         should visibly diverge on this drifting draw"
    );

    // The Johansen eigenvectors the fixture pins are reproduced too —
    // per column up to sign (both are S_11-orthonormal, but eigensolvers
    // pick column signs arbitrarily).
    let evec_fx = as_mat(&fx["johansen"]["evec"]);
    for j in 0..3 {
        let dot: f64 = (0..3).map(|i| joh.evec[(i, j)] * evec_fx[(i, j)]).sum();
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        for i in 0..3 {
            assert_rel_close(
                sign * joh.evec[(i, j)],
                evec_fx[(i, j)],
                1e-6,
                &format!("evec[({i},{j})] (sign-aligned)"),
            );
        }
    }
    // Johansen eigenvalues against the fixture.
    let eig_fx = as_vec(&fx["johansen"]["eig"]);
    for (i, &e) in eig_fx.iter().enumerate() {
        assert_rel_close(joh.eig[i], e, 1e-8, &format!("johansen eig[{i}]"));
    }
}

/// The degenerate short-run case `k_ar_diff = 0` under the unrestricted
/// constant: the regression block is the constant alone; gamma is `k x 0`,
/// det_coef is `k x 1` and finite, and the fit succeeds.
#[test]
fn vecm_constant_k_ar_diff_zero() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["data"]);
    let res = fit_vecm_det(endog.as_ref(), 0, 1, VecmDeterministic::Constant).unwrap();
    assert_eq!(res.gamma.ncols(), 0);
    assert_eq!(res.det_coef.ncols(), 1);
    for i in 0..3 {
        assert!(res.det_coef[(i, 0)].is_finite());
    }
    assert!(res.llf.is_finite());
}

// ------------------------------------------------------------------
// The restricted cases (the 0.7.0 follow-up).

/// [`assert_mat_close`] that also accepts a `0 x c` expected block, which
/// JSON stores as `[]` (no rows to take a column count from).
fn assert_mat_close_maybe_empty(actual: &Mat<f64>, expected: &Value, tol: f64, what: &str) {
    let rows = expected.as_array().expect("expected JSON array of rows");
    if rows.is_empty() {
        assert_eq!(actual.nrows(), 0, "{what}: expected zero rows");
        return;
    }
    assert_mat_close(actual, expected, tol, what);
}

/// Fits one pinned case and compares every estimate against statsmodels.
fn check_case(endog: &Mat<f64>, k_ar_diff: usize, rank: usize, code: &str, block: &Value) {
    let det = VecmDeterministic::from_code(code)
        .unwrap_or_else(|| panic!("fixture case {code:?} not parseable"));
    assert_eq!(det.code(), code, "code round-trip");
    let res = fit_vecm_det(endog.as_ref(), k_ar_diff, rank, det).unwrap();
    assert_eq!(res.deterministic, det);
    assert_mat_close(
        &res.alpha,
        &block["alpha"],
        1e-6,
        &format!("alpha ({code})"),
    );
    assert_mat_close(&res.beta, &block["beta"], 1e-6, &format!("beta ({code})"));
    assert_mat_close_maybe_empty(
        &res.det_coef_coint,
        &block["det_coef_coint"],
        1e-6,
        &format!("det_coef_coint ({code})"),
    );
    assert_mat_close(
        &res.gamma,
        &block["gamma"],
        1e-6,
        &format!("gamma ({code})"),
    );
    assert_mat_close_maybe_empty(
        &res.det_coef,
        &block["det_coef"],
        1e-6,
        &format!("det_coef ({code})"),
    );
    assert_mat_close(
        &res.sigma_u,
        &block["sigma_u"],
        1e-6,
        &format!("sigma_u ({code})"),
    );
    assert_rel_close(res.llf, num(&block["llf"]), 1e-6, &format!("llf ({code})"));
}

/// Every statsmodels deterministic case — `"n"`, `"co"`, `"ci"`, `"lo"`,
/// `"li"`, `"colo"`, `"coli"`, `"cilo"`, `"cili"` — matches statsmodels
/// `VECM(k_ar_diff = 2, coint_rank = 1, deterministic = ...)` to 1e-6 on
/// the trending draw: alpha, beta, the restricted rows `det_coef_coint`
/// (constant first, then trend), gamma, det_coef, sigma_u, llf.
#[test]
fn golden_vecm_every_deterministic_case_trending() {
    let fx = load_fixture("vecm_deterministic.json");
    let tr = &fx["trending"];
    let endog = as_endog(&tr["data"]);
    let k_ar_diff = tr["k_ar_diff"].as_u64().unwrap() as usize;
    let rank = tr["coint_rank"].as_u64().unwrap() as usize;
    let cases = tr["cases"].as_object().unwrap();
    assert_eq!(cases.len(), 9, "the fixture pins all nine cases");
    for (code, block) in cases {
        check_case(&endog, k_ar_diff, rank, code, block);
    }
}

/// The restricted-case shape contract on the trending draw: `"cili"`
/// widens the cointegrating matrix by two rows (constant, then trend —
/// split into `det_coef_coint`, statsmodels' `VECMResults` layout), the
/// short-run equations get no deterministic term, and the widened-beta
/// normalization keeps `beta[:r, :r] = I`.
#[test]
fn vecm_restricted_shapes_and_normalization() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["trending"]["data"]);
    let res = fit_vecm_det(
        endog.as_ref(),
        2,
        1,
        VecmDeterministic::RestrictedConstantRestrictedTrend,
    )
    .unwrap();
    assert_eq!(res.beta.nrows(), 3);
    assert_eq!(res.beta.ncols(), 1);
    assert_eq!(res.det_coef_coint.nrows(), 2); // constant row, then trend row
    assert_eq!(res.det_coef_coint.ncols(), 1);
    assert_eq!(res.det_coef.ncols(), 0); // nothing outside the relation
                                         // The normalization beta[:r,:r] = I holds to float round-off (the top
                                         // block is multiplied by its own inverse, as in statsmodels).
    assert!(
        (res.beta[(0, 0)] - 1.0).abs() < 1e-12,
        "normalization beta[:r,:r] = I: {}",
        res.beta[(0, 0)]
    );
    // The widened problem has k + 2 eigenvalues, at most k of them nonzero.
    assert_eq!(res.eig.len(), 5);
    assert!(res.eig[3].abs() < 1e-10 && res.eig[4].abs() < 1e-10);
}

/// The Johansen `det_order` correspondence on the trending draw, exactly
/// as the fixture measures it in statsmodels itself: `"colo"`'s beta and
/// `coint_johansen(det_order = 1)`'s first eigenvector are the same
/// direction *asymptotically* but NOT identical in finite samples
/// (det_order = 1 detrends the levels over the full sample before
/// partialling — a different projection), so the pinned cosine is ~1
/// minus ~6e-9 rather than exact — unlike the exact `"co"` <->
/// `det_order = 0` identity of `vecm_constant_reconciles_with_johansen`.
/// The cross-case cosines pin that the deterministic case visibly moves
/// beta on this draw.
#[test]
fn vecm_trending_johansen_correspondence_and_case_divergence() {
    let fx = load_fixture("vecm_deterministic.json");
    let tr = &fx["trending"];
    let endog = as_endog(&tr["data"]);
    let cos_pins = &tr["beta_cosines"];

    let colo = fit_vecm_det(endog.as_ref(), 2, 1, VecmDeterministic::ConstantTrend).unwrap();
    let joh1_evec = as_mat(&tr["johansen_det1"]["evec"]);
    let joh1_beta: Vec<f64> = (0..3)
        .map(|i| joh1_evec[(i, 0)] / joh1_evec[(0, 0)])
        .collect();
    let cos_colo = cosine(&beta_col0(&colo.beta), &joh1_beta);
    assert_rel_close(
        cos_colo,
        num(&cos_pins["colo_joh1"]),
        1e-6,
        "cosine(beta_colo, joh det_order=1)",
    );
    assert!(cos_colo.abs() > 0.9999, "colo ~ johansen det_order=1");

    let co = fit_vecm_det(endog.as_ref(), 2, 1, VecmDeterministic::Constant).unwrap();
    let coli = fit_vecm_det(
        endog.as_ref(),
        2,
        1,
        VecmDeterministic::ConstantRestrictedTrend,
    )
    .unwrap();
    let ci = fit_vecm_det(endog.as_ref(), 2, 1, VecmDeterministic::RestrictedConstant).unwrap();
    let cili = fit_vecm_det(
        endog.as_ref(),
        2,
        1,
        VecmDeterministic::RestrictedConstantRestrictedTrend,
    )
    .unwrap();
    let cos_co_coli = cosine(&beta_col0(&co.beta), &beta_col0(&coli.beta));
    let cos_ci_cili = cosine(&beta_col0(&ci.beta), &beta_col0(&cili.beta));
    assert_rel_close(
        cos_co_coli,
        num(&cos_pins["co_coli"]),
        1e-6,
        "cosine(beta_co, beta_coli)",
    );
    assert_rel_close(
        cos_ci_cili,
        num(&cos_pins["ci_cili"]),
        1e-6,
        "cosine(beta_ci, beta_cili)",
    );
    // On this trending draw, adding the restricted trend visibly moves
    // beta (the case choice changes the answer).
    assert!(
        cos_co_coli < 0.999,
        "co vs coli should differ: {cos_co_coli}"
    );
    assert!(
        cos_ci_cili < 0.999,
        "ci vs cili should differ: {cos_ci_cili}"
    );
}

/// Centered seasonal dummies match statsmodels `seasons = 4` to 1e-6 —
/// both with an unrestricted constant and a nonzero `first_season`
/// (season phase of the presample start pinned) and with a restricted
/// constant (`"ci"` + seasons).
#[test]
fn golden_vecm_seasonal() {
    let fx = load_fixture("vecm_deterministic.json");
    let se = &fx["seasonal"];
    let endog = as_endog(&se["data"]);
    let k_ar_diff = se["k_ar_diff"].as_u64().unwrap() as usize;
    let rank = se["coint_rank"].as_u64().unwrap() as usize;
    let seasons = se["seasons"].as_u64().unwrap() as usize;

    for (key, det) in [
        ("co_s4_fs2", VecmDeterministic::Constant),
        ("ci_s4_fs0", VecmDeterministic::RestrictedConstant),
    ] {
        let block = &se[key];
        let first_season = block["first_season"].as_u64().unwrap() as usize;
        let res =
            fit_vecm_seasonal(endog.as_ref(), k_ar_diff, rank, det, seasons, first_season).unwrap();
        assert_eq!(res.seasons, 4);
        assert_eq!(res.first_season, first_season);
        assert_mat_close(&res.alpha, &block["alpha"], 1e-6, &format!("alpha ({key})"));
        assert_mat_close(&res.beta, &block["beta"], 1e-6, &format!("beta ({key})"));
        assert_mat_close_maybe_empty(
            &res.det_coef_coint,
            &block["det_coef_coint"],
            1e-6,
            &format!("det_coef_coint ({key})"),
        );
        assert_mat_close(&res.gamma, &block["gamma"], 1e-6, &format!("gamma ({key})"));
        assert_mat_close(
            &res.det_coef,
            &block["det_coef"],
            1e-6,
            &format!("det_coef ({key})"),
        );
        assert_mat_close(
            &res.sigma_u,
            &block["sigma_u"],
            1e-6,
            &format!("sigma_u ({key})"),
        );
        assert_rel_close(res.llf, num(&block["llf"]), 1e-6, &format!("llf ({key})"));
    }
}

/// `fit_vecm_det` is exactly `fit_vecm_seasonal` with `seasons = 0`, and
/// `seasons = 1` is refused with the teaching error (a one-period cycle
/// has no dummies).
#[test]
fn vecm_seasonal_delegation_and_seasons_one_refusal() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["seasonal"]["data"]);
    let a = fit_vecm_det(endog.as_ref(), 1, 1, VecmDeterministic::Constant).unwrap();
    let b = fit_vecm_seasonal(endog.as_ref(), 1, 1, VecmDeterministic::Constant, 0, 0).unwrap();
    assert_eq!(a.llf.to_bits(), b.llf.to_bits(), "seasons=0 delegation");

    let err = fit_vecm_seasonal(endog.as_ref(), 1, 1, VecmDeterministic::Constant, 1, 0);
    assert!(
        matches!(err, Err(CointError::InvalidArgument { .. })),
        "seasons=1 must be refused: {err:?}"
    );
}

/// Audit round 10, finding 3h: when `seasons` is large enough that the
/// dummies alone exhaust the degrees of freedom, the insufficiency
/// message must charge them for it — the regressor-count sentence and the
/// hint both account for the seasonal columns instead of blaming
/// `k_ar_diff` alone.
#[test]
fn vecm_seasonal_insufficiency_accounts_for_dummy_columns() {
    let fx = load_fixture("vecm_deterministic.json");
    let endog = as_endog(&fx["seasonal"]["data"]);
    let t = endog.nrows();
    let k = endog.ncols();

    // seasons ~ T: the seasons - 1 dummies leave no residual df.
    let err = fit_vecm_seasonal(endog.as_ref(), 1, 1, VecmDeterministic::Constant, t, 0)
        .expect_err("seasons ~ T must refuse");
    match err {
        CointError::InsufficientObservations {
            n_det, n_seasonal, ..
        } => {
            assert_eq!(n_seasonal, t - 1, "seasonal column count");
            assert_eq!(n_det, 1, "the unrestricted constant");
        }
        other => panic!("expected InsufficientObservations, got {other:?}"),
    }
    let msg = fit_vecm_seasonal(endog.as_ref(), 1, 1, VecmDeterministic::Constant, t, 0)
        .expect_err("seasons ~ T must refuse")
        .to_string();
    assert!(
        msg.contains("seasonal-dummy column(s)"),
        "message must name the seasonal dummies: {msg}"
    );
    assert!(
        !msg.contains("Try k_ar_diff <="),
        "no k_ar_diff can rescue a dummy-exhausted sample: {msg}"
    );
    assert!(
        msg.contains("reduce seasons"),
        "message must offer the seasons lever: {msg}"
    );
    // The regressor-count sentence sums lags (k_ar_diff = 1) + levels +
    // deterministic + seasonal columns.
    let per_eq = k + k + 1 + (t - 1);
    assert!(
        msg.contains(&format!("= {per_eq} in total")),
        "per-equation count must include the dummy columns: {msg}"
    );

    // The hint's k_ar_diff bound stays exact WITH seasonal columns: on a
    // truncated sample with quarterly dummies, the claimed d_max fits
    // and d_max + 1 refuses. The bound inverts
    // n >= d*(k+1) + k + n_det + n_seasonal + 2 (n_det = 1 for "co",
    // n_seasonal = 3).
    let small_t = 8 * (k + 1) + k; // roomy enough for a d_max >= 1
    let small = Mat::from_fn(small_t, k, |i, j| endog[(i, j)]);
    let d_max = (small_t - k - 2 - 1 - 3) / (k + 1);
    let too_big = d_max + 1 + 2; // clearly refused
    let err = fit_vecm_seasonal(
        small.as_ref(),
        too_big,
        1,
        VecmDeterministic::Constant,
        4,
        0,
    )
    .expect_err("oversized k_ar_diff with quarterly dummies must refuse")
    .to_string();
    assert!(
        err.contains(&format!("Try k_ar_diff <= {d_max}")),
        "hint must subtract the dummy columns before inverting (d_max = {d_max}): {err}"
    );
    assert!(
        fit_vecm_seasonal(small.as_ref(), d_max, 1, VecmDeterministic::Constant, 4, 0).is_ok(),
        "the hinted k_ar_diff = {d_max} must actually fit"
    );
    assert!(
        fit_vecm_seasonal(
            small.as_ref(),
            d_max + 1,
            1,
            VecmDeterministic::Constant,
            4,
            0
        )
        .is_err(),
        "k_ar_diff = {} must still refuse",
        d_max + 1
    );
}

/// The statsmodels-string round trip covers all nine cases, and
/// unparseable strings — including the statsmodels conflicts and legacy
/// aliases — return `None`.
#[test]
fn vecm_deterministic_code_round_trip() {
    let codes = ["n", "co", "ci", "lo", "li", "colo", "coli", "cilo", "cili"];
    for code in codes {
        let det =
            VecmDeterministic::from_code(code).unwrap_or_else(|| panic!("{code:?} should parse"));
        assert_eq!(det.code(), code);
    }
    for bad in ["", "nc", "c", "coci", "lilo", "cico", "loli", "seasons"] {
        assert!(
            VecmDeterministic::from_code(bad).is_none(),
            "{bad:?} should not parse"
        );
    }
}
