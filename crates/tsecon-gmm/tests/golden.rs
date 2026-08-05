//! Golden-value tests against `fixtures/gmm.json` (linearmodels 7.0).
//!
//! The fixture DGP is a linear IV regression of `y` on `[const, w, x]` where
//! `x` is endogenous and `[const, w]` are exogenous; the instrument set is
//! `[z1, z2]` plus the included exogenous regressors, so the full instrument
//! matrix is `Z = [const, w, z1, z2]` (4 columns) against a `k = 3` regressor
//! matrix `X = [const, w, x]` — one over-identifying restriction.
//!
//! The golden numbers come from `linearmodels.iv.IVGMM(...).fit()` with the
//! default `weight_type="robust"` and `cov_type="robust"` (2-step efficient
//! GMM, heteroskedasticity-robust weighting and robust sandwich covariance).
//! We reproduce, via [`tsecon_gmm::two_step_gmm`] with
//! [`tsecon_gmm::GmmWeight::Robust`]:
//!
//! * `params` (const, w, x) to `1e-9` (fixture printed at full precision;
//!   the point estimate matches to machine precision);
//! * `bse` (const, w, x) to `1e-6` — the full GMM sandwich covariance with
//!   the step-2 estimation weight `W = S(u1)^{-1}` and the moment covariance
//!   `S` recomputed at the step-2 residuals (the collapsed efficient form
//!   only reaches ~5e-5);
//! * `j_stat` to `1e-6` and `j_pval` to `1e-6` — the Hansen J uses the
//!   step-2 estimation weight evaluated at the step-2 residuals.
//!
//! A second fixture, `fixtures/gmm_first_stage.json`, pins the two surfaces
//! the interval-coverage audit found missing or broken:
//!
//! * the **first-stage F** on the excluded instruments, for a one-endogenous
//!   and a two-endogenous design, against `linearmodels`
//!   `IV2SLS(...).fit(cov_type="robust").first_stage.diagnostics`. The
//!   conventions differ by an exact factor (`linearmodels` reports an
//!   undivided HC0 Wald against `chi2(q)`, this crate the Stata-convention
//!   HC1 `F(q, n - L)`), so the fixture stores both the raw statistic and the
//!   converted one and the test checks against each; achieved agreement is
//!   ~2e-15 relative on the statistic and ~1e-13 on the p-value (versus
//!   `scipy.stats.f.sf`), so they are pinned at `1e-12` / `1e-11`. The
//!   fixture's `lm_chi2_pval` (linearmodels' own `f.pval`) is checked too, and
//!   it is what makes the conversion *auditable* rather than asserted: it
//!   confirms directly that their statistic is referred to `chi2(q)`, which is
//!   the premise the whole `f.stat = fstat * q * n / (n - L)` identity rests
//!   on;
//! * the **HAC weighting path at a nonzero bandwidth**, against
//!   `IVGMM(weight_type="kernel", kernel="bartlett", bandwidth=m)`. This had
//!   never been validated: the only HAC test used `bandwidth = 0`, which is
//!   the White estimator in disguise. Agreement is ~2e-16 absolute, pinned
//!   at `1e-12`.

use serde_json::Value;
use tsecon_gmm::{two_step_gmm, GmmWeight};
use tsecon_hac::Kernel;
use tsecon_stats::chi2_sf;

fn load() -> Value {
    let path = format!("{}/../../fixtures/gmm.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn load_first_stage() -> Value {
    let path = format!(
        "{}/../../fixtures/gmm_first_stage.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture file readable");
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn assert_close(actual: f64, expected: f64, atol: f64, ctx: &str) {
    let err = (actual - expected).abs();
    assert!(
        err <= atol,
        "{ctx}: actual {actual}, expected {expected}, abs err {err:e} > tol {atol:e}"
    );
}

fn assert_rel(actual: f64, expected: f64, rtol: f64, ctx: &str) {
    let err = ((actual - expected) / expected).abs();
    assert!(
        err <= rtol,
        "{ctx}: actual {actual}, expected {expected}, rel err {err:e} > tol {rtol:e}"
    );
}

#[test]
fn ivgmm_two_step_robust_matches_linearmodels() {
    let fx = load();
    let y = f64s(&fx["y"]);
    let x = f64s(&fx["x"]);
    let w = f64s(&fx["w"]);
    let z1 = f64s(&fx["z1"]);
    let z2 = f64s(&fx["z2"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];

    // X = [const, w, x] (x endogenous); Z = [const, w, z1, z2].
    let x_cols = vec![cst.clone(), w.clone(), x];
    let z_cols = vec![cst, w, z1, z2];

    let fit = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).expect("two-step GMM fits");

    // Parameter order in the fixture is const, w, x.
    let gp = &fx["ivgmm"]["params"];
    let gb = &fx["ivgmm"]["bse"];
    let names = ["const", "w", "x"];
    for (i, name) in names.iter().enumerate() {
        assert_close(
            fit.params[i],
            gp[*name].as_f64().unwrap(),
            1e-9,
            &format!("param[{name}]"),
        );
        assert_close(
            fit.bse[i],
            gb[*name].as_f64().unwrap(),
            1e-6,
            &format!("bse[{name}]"),
        );
    }

    // Hansen J-test: over-identified with dof = 4 - 3 = 1.
    let jtest = fit
        .jtest
        .expect("over-identified model has a Hansen J-test");
    assert_eq!(jtest.dof, 1);
    assert_close(
        jtest.stat,
        fx["ivgmm"]["j_stat"].as_f64().unwrap(),
        1e-6,
        "j_stat",
    );
    assert_close(
        jtest.pval,
        fx["ivgmm"]["j_pval"].as_f64().unwrap(),
        1e-6,
        "j_pval",
    );

    assert_eq!(fit.steps, 2);
    assert_eq!(fit.nobs, n);
    assert_eq!(fit.nmoments, 4);
    assert_eq!(fit.nparams, 3);
}

// ---------------------------------------------------------------------------
// fixtures/gmm_first_stage.json — the first-stage F and the HAC path.
// ---------------------------------------------------------------------------

/// Check one design's first-stage diagnostics against the fixture.
///
/// The fixture's `f_hc1` is `linearmodels`' robust `f.stat` (an *undivided*
/// HC0 Wald referred to `chi2(q)`) converted by the exact identity
/// `F_HC1 = f.stat / q * (n - L) / n` into the Stata-convention HC1 F this
/// crate reports; see [`tsecon_gmm::FirstStageF`] and the fixture generator
/// `fixtures/generate_gmm_first_stage_fixtures.py` for why the conventions
/// differ and why the map is exact. We re-assert the identity here from the
/// raw `lm_wald_chi2_hc0` so the test fails loudly if the conversion is ever
/// silently re-derived, and check the p-value against `scipy.stats.f.sf`.
fn check_first_stage(design: &Value, x_cols: &[Vec<f64>], z_cols: &[Vec<f64>], y: &[f64]) {
    let fit = two_step_gmm(x_cols, z_cols, y, GmmWeight::Robust).expect("two-step GMM fits");
    let expected = design["first_stage"].as_array().expect("array");
    let endog_index: Vec<usize> = design["endog_index"]
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_u64().expect("index") as usize)
        .collect();
    assert_eq!(
        fit.first_stage.len(),
        expected.len(),
        "one first-stage F per endogenous regressor"
    );
    let n = y.len();
    let ell = z_cols.len();
    for (i, fs) in fit.first_stage.iter().enumerate() {
        let name = expected[i]["name"].as_str().expect("name");
        assert_eq!(fs.regressor, endog_index[i], "{name}: regressor index");
        assert_eq!(
            fs.dof_num,
            expected[i]["dof_num"].as_u64().expect("dof") as usize,
            "{name}: numerator dof (excluded instruments)"
        );
        assert_eq!(fs.dof_den, n - ell, "{name}: denominator dof");

        // Direct comparison against the converted linearmodels statistic.
        assert_rel(
            fs.fstat,
            expected[i]["f_hc1"].as_f64().expect("f"),
            1e-12,
            &format!("{name}: first-stage F (HC1)"),
        );
        // And against the raw chi2 Wald through the documented identity.
        let wald = expected[i]["lm_wald_chi2_hc0"].as_f64().expect("f");
        let as_chi2 = fs.fstat * fs.dof_num as f64 * n as f64 / (n - ell) as f64;
        assert_rel(
            as_chi2,
            wald,
            1e-12,
            &format!("{name}: linearmodels chi2 Wald via F * q * n/(n-L)"),
        );
        // The identity above only makes sense if linearmodels really refers
        // its `f.stat` to chi2(q) rather than to an F. Its own `f.pval` column
        // is stored in the fixture, so check that reading directly: our
        // chi2(q) survival function at their statistic must reproduce their
        // p-value. If linearmodels ever switched conventions this fails here,
        // where the cause is legible, instead of silently shifting `f_hc1`.
        let lm_pval = expected[i]["lm_chi2_pval"].as_f64().expect("f");
        let ours = chi2_sf(wald, fs.dof_num as f64).expect("chi2 sf");
        if lm_pval > 0.0 {
            assert_rel(
                ours,
                lm_pval,
                1e-10,
                &format!("{name}: linearmodels f.pval is chi2(q), not F(q, n-L)"),
            );
            // The check is only worth making if the two conventions are
            // actually distinguishable on this row — otherwise it would pass
            // just as happily against `F(q, n - L)`. On the weakly
            // instrumented regressor they differ by ~42% (6.78e-6 vs 1.18e-5).
            let spread = ((lm_pval - fs.pval) / fs.pval).abs();
            assert!(
                spread > 0.1,
                "{name}: chi2(q) and F(q, n-L) p-values must be far enough \
                 apart for this check to have bite, got {lm_pval} vs {} \
                 (spread {spread:e})",
                fs.pval
            );
        } else {
            // linearmodels/scipy underflow the strongly-instrumented rows to
            // exactly 0. All that is checkable there is that we also vanish.
            assert!(
                ours < 1e-40,
                "{name}: linearmodels reports f.pval = 0, ours is {ours:e}"
            );
        }
        assert_rel(
            fs.pval,
            expected[i]["f_pval"].as_f64().expect("f"),
            1e-11,
            &format!("{name}: F(q, n-L) p-value"),
        );
    }
}

#[test]
fn first_stage_f_matches_linearmodels_one_endogenous() {
    let fx = load_first_stage();
    let d = &fx["design_a"];
    let y = f64s(&d["y"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];
    let (x, w, z1, z2) = (f64s(&d["x"]), f64s(&d["w"]), f64s(&d["z1"]), f64s(&d["z2"]));
    // X = [const, w, x]; Z = [const, w, z1, z2] => x endogenous, q = 2.
    let x_cols = vec![cst.clone(), w.clone(), x];
    let z_cols = vec![cst, w, z1, z2];
    check_first_stage(d, &x_cols, &z_cols, &y);
}

#[test]
fn first_stage_f_matches_linearmodels_two_endogenous() {
    let fx = load_first_stage();
    let d = &fx["design_b"];
    let y = f64s(&d["y"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];
    let (x1, x2, w) = (f64s(&d["x1"]), f64s(&d["x2"]), f64s(&d["w"]));
    let (z1, z2, z3) = (f64s(&d["z1"]), f64s(&d["z2"]), f64s(&d["z3"]));
    // X = [const, w, x1, x2]; Z = [const, w, z1, z2, z3] => q = 3, two
    // instrumented regressors, so the per-regressor loop is exercised.
    let x_cols = vec![cst.clone(), w.clone(), x1, x2];
    let z_cols = vec![cst, w, z1, z2, z3];
    check_first_stage(d, &x_cols, &z_cols, &y);

    // The generator gives x1 a large loading on the instruments and x2 a small
    // one, and the reported statistics separate the two by orders of magnitude.
    // That is all this asserts: the number responds to the first-stage fit and
    // is not a constant.
    //
    // It is NOT an endorsement of reading the naive per-regressor F as a
    // weak-identification test here. With two endogenous regressors it is not
    // one: all of them can clear 10 while the system is under-identified,
    // because the instruments may predict only a single common combination of
    // x1 and x2. The statistics that answer that question are the
    // Angrist-Pischke per-regressor F and Cragg-Donald / Kleibergen-Paap
    // against Stock-Yogo critical values, none of which this crate implements
    // (see `tsecon_gmm::FirstStageF`). Read "F = 0.7 for x2" as "the excluded
    // instruments barely move x2", not as "x2's coefficient is weakly
    // identified and x1's is not".
    let fit = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).expect("fits");
    assert!(
        fit.first_stage[0].fstat > 100.0,
        "the instruments explain x1 strongly, got F = {}",
        fit.first_stage[0].fstat
    );
    assert!(
        fit.first_stage[1].fstat < 10.0,
        "the instruments barely explain x2, got F = {}",
        fit.first_stage[1].fstat
    );
}

/// The HAC weighting path at a **nonzero** bandwidth, against
/// `linearmodels` `IVGMM(weight_type="kernel", ...)`.
///
/// Before the bandwidth fix this path had no golden at all: the only test
/// covering it used `bandwidth = 0`, which is the White estimator in
/// disguise, so the kernel lag sum itself was never validated against a
/// reference. The bandwidth here is the automatic Newey-West (1994) rule
/// value at `n = 300` (5 lags), which is also what [`GmmWeight::HacAuto`]
/// selects — so this pins the automatic rule end to end.
#[test]
fn hac_two_step_matches_linearmodels_kernel_gmm() {
    let fx = load_first_stage();
    let d = &fx["design_a"];
    let y = f64s(&d["y"]);
    let n = y.len();
    let cst = vec![1.0_f64; n];
    let (x, w, z1, z2) = (f64s(&d["x"]), f64s(&d["w"]), f64s(&d["z1"]), f64s(&d["z2"]));
    let x_cols = vec![cst.clone(), w.clone(), x];
    let z_cols = vec![cst, w, z1, z2];

    let g = &d["hac_two_step"];
    let bandwidth = g["bandwidth"].as_f64().expect("bandwidth");
    assert_eq!(
        GmmWeight::auto_bandwidth(n),
        bandwidth,
        "the fixture bandwidth is the automatic rule's choice at n = {n}"
    );

    let fit = two_step_gmm(
        &x_cols,
        &z_cols,
        &y,
        GmmWeight::HacAuto {
            kernel: Kernel::Bartlett,
        },
    )
    .expect("HAC two-step GMM fits");
    assert_eq!(fit.hac_bandwidth, Some(bandwidth));

    let names = ["const", "w", "x"];
    for (i, name) in names.iter().enumerate() {
        assert_close(
            fit.params[i],
            g["params"][*name].as_f64().unwrap(),
            1e-12,
            &format!("hac param[{name}]"),
        );
        assert_close(
            fit.bse[i],
            g["bse"][*name].as_f64().unwrap(),
            1e-12,
            &format!("hac bse[{name}]"),
        );
    }
    let j = fit.jtest.expect("over-identified");
    assert_close(j.stat, g["j_stat"].as_f64().unwrap(), 1e-12, "hac j_stat");
    assert_close(j.pval, g["j_pval"].as_f64().unwrap(), 1e-12, "hac j_pval");

    // The HAC answer must differ materially from the White one — the bug the
    // audit found was that it did not.
    let robust = two_step_gmm(&x_cols, &z_cols, &y, GmmWeight::Robust).expect("fits");
    let gap = fit
        .bse
        .iter()
        .zip(robust.bse.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        gap > 1e-3,
        "HAC and robust standard errors should differ on a serially correlated \
         design, max gap {gap:e}"
    );
}
