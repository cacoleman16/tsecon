//! Golden-value tests for the asymptotic (Lütkepohl 1990 delta-method)
//! VAR IRF standard errors against the statsmodels fixture
//! (`var_irf_bands.json`): a stable VAR(2), k = 3, n = 300, fitted with a
//! constant. The delta-method standard errors are a deterministic closed
//! form, so they match statsmodels `IRAnalysis.stderr` /
//! `cum_effect_stderr` tightly — the tolerance here is 1e-6.

mod common;

use common::{as_mat, assert_mat_close, load_fixture};
use tsecon_var::irf_asymptotic::irf_asymptotic_se;
use tsecon_var::{Trend, VarSpec};

const H: usize = 10;
const RTOL: f64 = 1e-6;

fn fitted() -> (serde_json::Value, tsecon_var::VarResults) {
    let fx = load_fixture("var_irf_bands.json");
    let data = as_mat(&fx["data"]);
    let res = VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    (fx, res)
}

fn check_se(fx_key: &str, orth: bool, cumulative: bool) {
    let (fx, res) = fitted();
    let se = irf_asymptotic_se(&res, H, orth, cumulative).unwrap();
    let expected = fx[fx_key].as_array().unwrap();
    assert_eq!(se.len(), H + 1, "{fx_key}: horizon count");
    assert_eq!(expected.len(), H + 1, "{fx_key}: fixture horizon count");
    for h in 0..=H {
        assert_mat_close(&se[h], &expected[h], RTOL, &format!("{fx_key}[{h}]"));
    }
}

/// The point IRFs of the fitted VAR reproduce the statsmodels fixture
/// (a prerequisite: the delta-method SEs are anchored on these). Both
/// the reduced-form `Phi_h` and the orthogonalized `Theta_h = Phi_h P`.
#[test]
fn golden_point_matches_statsmodels() {
    let (fx, res) = fitted();
    let irf = res.irf(H).unwrap();
    let e_nonorth = fx["point_nonorth"].as_array().unwrap();
    let e_orth = fx["point_orth"].as_array().unwrap();
    for h in 0..=H {
        assert_mat_close(
            &irf.irfs[h],
            &e_nonorth[h],
            1e-8,
            &format!("point_nonorth[{h}]"),
        );
        assert_mat_close(
            &irf.orth_irfs[h],
            &e_orth[h],
            1e-8,
            &format!("point_orth[{h}]"),
        );
    }
}

/// Reduced-form (non-orthogonalized) delta-method SEs match
/// `irf.stderr(orth=False)` to 1e-6. Horizon 0 is exactly zero (the IRF
/// at h=0 is the identity with no parameter uncertainty).
#[test]
fn golden_stderr_nonorth() {
    check_se("stderr_nonorth", false, false);
    // Horizon 0 carries no coefficient uncertainty.
    let (_, res) = fitted();
    let se = irf_asymptotic_se(&res, H, false, false).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!(se[0][(i, j)] == 0.0, "non-orth se[0][{i}][{j}] should be 0");
        }
    }
}

/// Orthogonalized (Cholesky) delta-method SEs match
/// `irf.stderr(orth=True)` to 1e-6, including the nonzero horizon-0
/// standard errors that arise from the `vech(Sigma_u)` derivative term.
#[test]
fn golden_stderr_orth() {
    check_se("stderr_orth", true, false);
}

/// Cumulative reduced-form delta-method SEs match
/// `irf.cum_effect_stderr(orth=False)` to 1e-6.
#[test]
fn golden_cum_stderr_nonorth() {
    check_se("cum_stderr_nonorth", false, true);
}

/// Cumulative orthogonalized delta-method SEs match
/// `irf.cum_effect_stderr(orth=True)` to 1e-6.
#[test]
fn golden_cum_stderr_orth() {
    check_se("cum_stderr_orth", true, true);
}
