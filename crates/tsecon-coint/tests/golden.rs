//! Golden-value tests against the statsmodels fixture (coint.json): a
//! 3-variable system (two cointegrated I(1) series plus one stationary
//! series) simulated for statsmodels' `coint_johansen(det_order = 0,
//! k_ar_diff = 2)` and `VECM(k_ar_diff = 2, coint_rank = 1,
//! deterministic = "n")`.

mod common;

use common::{as_endog, as_vec, assert_mat_close, assert_rel_close, load_fixture, num};
use tsecon_coint::{fit_vecm, johansen};

/// The Johansen eigenvalues match statsmodels to 1e-8 and both
/// likelihood-ratio statistics to 1e-6 relative.
#[test]
fn golden_johansen_statistics() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let jb = &fx["johansen"];

    let res = johansen(endog.as_ref(), num(&jb["k_ar_diff"]) as usize).unwrap();
    assert_eq!(res.neqs, 3);
    assert_eq!(res.nobs, 397);

    let eig = as_vec(&jb["eig"]);
    let trace = as_vec(&jb["trace_stat"]);
    let maxe = as_vec(&jb["max_eig_stat"]);
    for r in 0..3 {
        assert_rel_close(res.eig[r], eig[r], 1e-8, &format!("eig[{r}]"));
        assert_rel_close(res.trace_stat[r], trace[r], 1e-6, &format!("trace[{r}]"));
        assert_rel_close(res.max_eig_stat[r], maxe[r], 1e-6, &format!("max_eig[{r}]"));
    }
}

/// The shipped MacKinnon-Haug-Michelis critical values (det_order = 0)
/// reproduce the tabulated rows the fixture pins, exactly.
#[test]
fn golden_johansen_critical_values() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let jb = &fx["johansen"];
    let res = johansen(endog.as_ref(), 2).unwrap();

    let trace_cv = &jb["trace_crit_90_95_99"];
    let maxe_cv = &jb["max_eig_crit_90_95_99"];
    for r in 0..3 {
        for c in 0..3 {
            assert_rel_close(
                res.trace_crit[r][c],
                num(&trace_cv[r][c]),
                0.0,
                &format!("trace_crit[{r}][{c}]"),
            );
            assert_rel_close(
                res.max_eig_crit[r][c],
                num(&maxe_cv[r][c]),
                0.0,
                &format!("max_eig_crit[{r}][{c}]"),
            );
        }
    }
}

/// Rank-1 VECM: `alpha`, `beta`, `gamma`, and the log-likelihood match
/// statsmodels to 1e-6 relative (statsmodels `beta[:r, :r] = I`
/// normalization).
#[test]
fn golden_vecm_rank1() {
    let fx = load_fixture("coint.json");
    let endog = as_endog(&fx["data"]);
    let vb = &fx["vecm_rank1"];

    let res = fit_vecm(endog.as_ref(), 2, 1).unwrap();
    assert_eq!(res.neqs, 3);
    assert_eq!(res.nobs, 397);
    assert_eq!(res.coint_rank, 1);

    assert_mat_close(&res.alpha, &vb["alpha"], 1e-6, "alpha");
    assert_mat_close(&res.beta, &vb["beta"], 1e-6, "beta");
    assert_mat_close(&res.gamma, &vb["gamma"], 1e-6, "gamma");
    assert_rel_close(res.llf, num(&vb["llf"]), 1e-6, "llf");
}
