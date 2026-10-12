//! Golden tests for the Blanchard-Quah long-run SVAR bands
//! (`long_run_bootstrap_bands`, `long_run_asymptotic_bands`).
//!
//! `fixtures/long_run_svar_bands.json` is produced by
//! `fixtures/generate_long_run_svar_bands_fixtures.py`, which never imports
//! tsecon:
//!
//! * the BOOTSTRAP cases replay the documented resampling stream in NumPy
//!   (replication `b` = `Philox(SeedSequence(seed).spawn(n_boot)[b])`, iid
//!   indices by bitmask rejection, one raw draw per attempt), rebuild each
//!   pseudo-sample, refit by `numpy.linalg.lstsq`, and re-run the
//!   Blanchard-Quah map per draw; the Kilian branch is a transcription of
//!   tsecon's own `estimate_bias` / `correct_coefs` (its external anchor is
//!   the Monte Carlo in `long_run_bands_props.rs`);
//! * the DELTA cases use statsmodels 0.15's `IRAnalysis` building blocks with
//!   an analytic Blanchard-Quah Jacobian that the generator checks against
//!   central finite differences and, at `B = chol(Sigma_u)`, against
//!   `IRAnalysis.stderr(orth=True)`.
//!
//! Tolerance: `rtol = 1e-10` with an absolute floor of `1e-12` (the imposed
//! long-run zeros are exactly `±0` on both sides). The measured
//! cross-implementation gap is ~1e-14 (statsmodels / LAPACK vs faer OLS).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::{as_mat, load_fixture};
use serde_json::Value;
use tsecon_bootstrap::{indices, BlockScheme};
use tsecon_linalg::faer::Mat;
use tsecon_rng::Stream;
use tsecon_var::irf_asymptotic::{BandMethod, IrfBandScope};
use tsecon_var::{
    bootstrap_irf_bands, long_run_asymptotic_bands, long_run_bootstrap_bands, LongRunBandSpec,
    Trend, VarSpec,
};

const RTOL: f64 = 1e-10;
const ATOL: f64 = 1e-12;

fn close(actual: f64, expected: f64, what: &str) {
    let tol = ATOL + RTOL * expected.abs();
    assert!(
        (actual - expected).abs() <= tol,
        "{what}: {actual:.17e} vs {expected:.17e} (|diff| {:.3e} > tol {tol:.3e})",
        (actual - expected).abs()
    );
}

fn close_mat(actual: &Mat<f64>, expected: &Value, what: &str) {
    let e = as_mat(expected);
    assert_eq!(
        (actual.nrows(), actual.ncols()),
        (e.nrows(), e.ncols()),
        "{what} shape"
    );
    for i in 0..e.nrows() {
        for j in 0..e.ncols() {
            close(actual[(i, j)], e[(i, j)], &format!("{what}[{i}][{j}]"));
        }
    }
}

fn close_cube(actual: &[Mat<f64>], expected: &Value, what: &str) {
    let e = expected.as_array().unwrap();
    assert_eq!(actual.len(), e.len(), "{what} horizons");
    for (h, (a, ev)) in actual.iter().zip(e.iter()).enumerate() {
        close_mat(a, ev, &format!("{what}[h={h}]"));
    }
}

fn pairs(v: &Value) -> Option<Vec<(usize, usize)>> {
    v.as_array().map(|a| {
        a.iter()
            .map(|p| {
                let p = p.as_array().unwrap();
                (
                    p[0].as_u64().unwrap() as usize,
                    p[1].as_u64().unwrap() as usize,
                )
            })
            .collect()
    })
}

fn spec_from(s: &Value) -> LongRunBandSpec {
    let mut spec = LongRunBandSpec::new(
        s["lags"].as_u64().unwrap() as usize,
        s["horizon"].as_u64().unwrap() as usize,
    );
    spec.alpha = s["alpha"].as_f64().unwrap();
    spec.n_boot = s["n_boot"].as_u64().unwrap() as usize;
    spec.seed = s["seed"].as_u64().unwrap();
    spec.long_run_zeros = pairs(&s["restrictions"]);
    spec.impact_signs = pairs(&s["impact_signs"]);
    spec.normalize_impact = s["normalize"].as_str().unwrap() == "impact";
    spec.cumulative = s["cumulative"].as_bool().unwrap();
    spec.bias_correct = s
        .get("bias_correct")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if s.get("band").and_then(Value::as_str) == Some("sup-t") {
        spec.band = Some((BandMethod::SupT, IrfBandScope::Horizon));
    }
    spec
}

/// The first indices of replication 0 straight from the documented stream
/// (`Stream::substreams(seed, n_boot)[0]`, iid bitmask rejection): a stream
/// mismatch fails here, legibly, before any band is compared.
fn check_first_indices(case: &Value, spec: &LongRunBandSpec, t_eff: usize, name: &str) {
    let want: Vec<usize> = case["first_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mut streams = Stream::substreams(spec.seed, spec.n_boot).unwrap();
    let idx = indices(BlockScheme::Iid, t_eff, &mut streams[0]).unwrap();
    assert_eq!(&idx[..want.len()], &want[..], "{name} first_indices");
}

fn check_bootstrap_case(fx: &Value, name: &str, data_key: &str) {
    let case = &fx[name];
    let data = as_mat(&fx[data_key]);
    let spec = spec_from(&case["spec"]);
    check_first_indices(case, &spec, data.nrows() - spec.lags, name);
    let b = long_run_bootstrap_bands(data.as_ref(), &spec).unwrap();
    close_cube(&b.point, &case["point"], &format!("{name} point"));
    close_cube(&b.se, &case["se"], &format!("{name} se"));
    close_cube(&b.lower, &case["lower"], &format!("{name} lower"));
    close_cube(&b.upper, &case["upper"], &format!("{name} upper"));
    close_mat(
        &b.impact_point,
        &case["impact_point"],
        &format!("{name} impact"),
    );
    close_mat(
        &b.long_run_point,
        &case["long_run_point"],
        &format!("{name} LR"),
    );
    close_mat(
        &b.long_run_se,
        &case["long_run_se"],
        &format!("{name} LR se"),
    );
    close_mat(
        &b.long_run_lower,
        &case["long_run_lower"],
        &format!("{name} LR lower"),
    );
    close_mat(
        &b.long_run_upper,
        &case["long_run_upper"],
        &format!("{name} LR upper"),
    );
    assert_eq!(
        b.n_unstable_draws,
        Some(case["n_unstable_draws"].as_u64().unwrap() as usize),
        "{name} n_unstable_draws"
    );
    match case.get("sign_anchor_mismatches") {
        Some(m) => {
            let want: Vec<usize> = m
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(b.sign_anchor_mismatches, Some(want), "{name} mismatches");
        }
        None => assert_eq!(b.sign_anchor_mismatches, None),
    }
    if let Some(cv) = case.get("critical_value") {
        let got = b.simultaneous.as_ref().expect("sup-t band");
        let want = as_mat(cv);
        for i in 0..want.nrows() {
            for j in 0..want.ncols() {
                close(
                    got.values[i][j],
                    want[(i, j)],
                    &format!("{name} cv[{i}][{j}]"),
                );
            }
        }
        close_cube(
            b.sim_lower.as_ref().unwrap(),
            &case["sim_lower"],
            &format!("{name} sim_lower"),
        );
        close_cube(
            b.sim_upper.as_ref().unwrap(),
            &case["sim_upper"],
            &format!("{name} sim_upper"),
        );
    }
}

fn check_delta_case(fx: &Value, name: &str, data_key: &str) {
    let case = &fx[name];
    let data = as_mat(&fx[data_key]);
    let spec = spec_from(&case["spec"]);
    let res = VarSpec::new(spec.lags, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .unwrap();
    // The delta arm reads no band settings from the golden spec.
    let spec = LongRunBandSpec { band: None, ..spec };
    let b = long_run_asymptotic_bands(&res, &spec).unwrap();
    close_cube(&b.point, &case["point"], &format!("{name} point"));
    close_cube(&b.se, &case["se"], &format!("{name} se"));
    close_cube(&b.lower, &case["lower"], &format!("{name} lower"));
    close_cube(&b.upper, &case["upper"], &format!("{name} upper"));
    close_mat(
        &b.long_run_point,
        &case["long_run_point"],
        &format!("{name} LR"),
    );
    close_mat(
        &b.long_run_se,
        &case["long_run_se"],
        &format!("{name} LR se"),
    );
    close_mat(
        &b.long_run_lower,
        &case["long_run_lower"],
        &format!("{name} LR lower"),
    );
    close_mat(
        &b.long_run_upper,
        &case["long_run_upper"],
        &format!("{name} LR upper"),
    );
    assert_eq!(b.n_boot, None);
    assert_eq!(b.n_unstable_draws, None);
}

#[test]
fn bq_var8_bootstrap_matches_the_numpy_replay() {
    let fx = load_fixture("long_run_svar_bands.json");
    check_bootstrap_case(&fx, "bq_var8", "bq_data");
    check_bootstrap_case(&fx, "bq_var8_cum", "bq_data");
}

#[test]
fn bq_var8_kilian_bootstrap_matches_the_transcription() {
    let fx = load_fixture("long_run_svar_bands.json");
    check_bootstrap_case(&fx, "bq_var8_bc", "bq_data");
    check_bootstrap_case(&fx, "bq_var8_bc_cum", "bq_data");
}

#[test]
fn permuted_restrictions_impact_normalization_and_draw_sup_t() {
    let fx = load_fixture("long_run_svar_bands.json");
    check_bootstrap_case(&fx, "var3_perm", "var3_data");
}

#[test]
fn delta_method_matches_the_analytic_jacobian_reference() {
    let fx = load_fixture("long_run_svar_bands.json");
    check_delta_case(&fx, "bq_var8_delta", "bq_data");
    check_delta_case(&fx, "bq_var8_delta_cum", "bq_data");
    check_delta_case(&fx, "var3_perm_delta", "var3_data");
}

/// The shared Kilian path of `var_irf_bands` (orth, bias_correct) pinned to
/// the same independent transcription: before this case the bootstrap arm of
/// `var_irf_bands` had no external golden, and its bias-corrected path no CI
/// guard at all.
#[test]
fn var_irf_bands_kilian_bootstrap_matches_the_transcription() {
    let fx = load_fixture("long_run_svar_bands.json");
    let case = &fx["var_irf_bands_bc"];
    let s = &case["spec"];
    let data = as_mat(&fx["macro3_data"]);
    let b = bootstrap_irf_bands(
        data.as_ref(),
        s["lags"].as_u64().unwrap() as usize,
        Trend::Constant,
        s["horizon"].as_u64().unwrap() as usize,
        true,
        false,
        s["alpha"].as_f64().unwrap(),
        s["n_boot"].as_u64().unwrap() as usize,
        s["seed"].as_u64().unwrap(),
        true,
    )
    .unwrap();
    let spec = LongRunBandSpec {
        n_boot: s["n_boot"].as_u64().unwrap() as usize,
        seed: s["seed"].as_u64().unwrap(),
        ..LongRunBandSpec::new(2, 0)
    };
    check_first_indices(case, &spec, data.nrows() - 2, "var_irf_bands_bc");
    close_cube(&b.point, &case["point"], "var_irf_bands_bc point");
    close_cube(&b.se, &case["se"], "var_irf_bands_bc se");
    close_cube(&b.lower, &case["lower"], "var_irf_bands_bc lower");
    close_cube(&b.upper, &case["upper"], "var_irf_bands_bc upper");
}

/// The generator's own self-checks, recorded in the fixture, are the margins
/// that keep every pinned decision away from its threshold.
#[test]
fn fixture_records_its_margins_and_jacobian_checks() {
    let fx = load_fixture("long_run_svar_bands.json");
    assert!(fx["margins"]["radius"].as_f64().unwrap() > 1e-6);
    assert!(fx["margins"]["anchor"].as_f64().unwrap() > 1e-8);
    assert!(fx["margins"]["diag"].as_f64().unwrap() > 1e-8);
    assert!(fx["fd_jacobian_worst_rel"].as_f64().unwrap() < 1e-6);
    for k in ["bq_var8", "var3_perm"] {
        assert!(fx["chol_reduction_rel"][k].as_f64().unwrap() < 1e-12);
    }
    for k in ["numpy", "scipy", "statsmodels", "python"] {
        assert!(fx["_meta"][k].is_string(), "_meta.{k}");
    }
}
