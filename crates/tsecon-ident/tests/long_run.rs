//! Golden and property tests for the Blanchard-Quah long-run SVAR.
//!
//! `fixtures/long_run_svar.json` is produced by
//! `fixtures/generate_long_run_svar_fixtures.py`, a NumPy-only reference
//! (never imports tsecon) that transcribes the documented closed form
//! `B = D * chol_lower(C1 Sigma_u C1')`. Because both sides run the same
//! deterministic f64 algebra (LU inverse, lower Cholesky), the golden
//! tolerance is tight: 1e-10. The property tests (`B B' = Sigma_u`,
//! `C1 B = LR`, FEVD rows sum to 1, the strict-upper-triangle zeros) verify
//! the identifying restriction holds mechanically.

use serde_json::Value;
use tsecon_ident::long_run::{long_run_multiplier, long_run_svar};
use tsecon_ident::IdentError;
use tsecon_linalg::faer::{Mat, MatRef};

const TOL: f64 = 1e-10;

fn load() -> Value {
    let path = format!(
        "{}/../../fixtures/long_run_svar.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str(&text).expect("valid JSON")
}

fn mat_from(v: &Value) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .expect("2-D array")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect();
    let nr = rows.len();
    let nc = rows[0].len();
    Mat::from_fn(nr, nc, |i, j| rows[i][j])
}

fn coefs_from(v: &Value) -> Vec<Mat<f64>> {
    v.as_array()
        .expect("coefs array")
        .iter()
        .map(mat_from)
        .collect()
}

fn close_mat(actual: MatRef<'_, f64>, expected: MatRef<'_, f64>, what: &str) {
    assert_eq!(actual.nrows(), expected.nrows(), "{what} rows");
    assert_eq!(actual.ncols(), expected.ncols(), "{what} cols");
    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let a = actual[(i, j)];
            let e = expected[(i, j)];
            let err = (a - e).abs();
            assert!(
                err < TOL,
                "{what}[{i}][{j}]: actual={a:.15e} expected={e:.15e} abs_err={err:.3e}"
            );
        }
    }
}

/// Runs `long_run_svar` on a fixture case and returns the result.
fn run_case(case: &Value, normalize_impact: bool) -> tsecon_ident::LongRunSvar {
    let coefs = coefs_from(&case["coefs"]);
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let sigma = mat_from(&case["sigma_u"]);
    let horizon = case["horizon"].as_u64().expect("horizon") as usize;
    long_run_svar(&refs, sigma.as_ref(), horizon, None, normalize_impact).expect("identified")
}

fn assert_matches_fixture(case: &Value, res: &tsecon_ident::LongRunSvar) {
    close_mat(
        res.impact.as_ref(),
        mat_from(&case["impact"]).as_ref(),
        "impact",
    );
    close_mat(
        res.long_run.as_ref(),
        mat_from(&case["long_run"]).as_ref(),
        "long_run",
    );
    close_mat(
        res.long_run_multiplier.as_ref(),
        mat_from(&case["long_run_multiplier"]).as_ref(),
        "long_run_multiplier",
    );
    let irf_expected = case["irf"].as_array().expect("irf array");
    assert_eq!(res.irf.len(), irf_expected.len(), "irf length");
    for (h, ev) in irf_expected.iter().enumerate() {
        close_mat(
            res.irf[h].as_ref(),
            mat_from(ev).as_ref(),
            &format!("irf[{h}]"),
        );
    }
}

#[test]
fn algebra_2var_matches_closed_form() {
    let fx = load();
    let case = &fx["algebra_2var"];
    assert_matches_fixture(case, &run_case(case, false));
}

#[test]
fn algebra_3var_matches_closed_form() {
    let fx = load();
    let case = &fx["algebra_3var"];
    assert_matches_fixture(case, &run_case(case, false));
}

#[test]
fn estimated_2var_matches_closed_form() {
    // The reduced form here comes from a NumPy OLS fit of a simulated stable
    // VAR(2); this checks the identification map on a realistic (non-toy)
    // reduced form, and pins the fixture the end-to-end binding test reuses.
    let fx = load();
    let case = &fx["estimated_2var"];
    assert_matches_fixture(case, &run_case(case, false));
}

#[test]
fn impact_normalization_flips_negative_diagonal_columns() {
    let fx = load();
    let base = &fx["flip_2var"];
    let default = run_case(base, false);
    // The default impact has a negative diagonal entry (D[0,0] < 0).
    assert!(
        default.impact[(0, 0)] < 0.0,
        "fixture should trigger a flip"
    );

    // Default matches its stored expectation.
    assert_matches_fixture(base, &default);
    // Impact-normalized matches its own stored expectation.
    let impact = run_case(&fx["flip_2var_impact"], true);
    assert_matches_fixture(&fx["flip_2var_impact"], &impact);

    // Relationship: every diagonal entry is non-negative, and each column is
    // either identical or exactly negated.
    let k = impact.impact.nrows();
    for j in 0..k {
        assert!(impact.impact[(j, j)] >= 0.0, "diag {j} non-negative");
        let negated = (impact.impact[(0, j)] + default.impact[(0, j)]).abs() < TOL;
        let same = (impact.impact[(0, j)] - default.impact[(0, j)]).abs() < TOL;
        assert!(same || negated, "column {j} is default or its negation");
    }
}

#[test]
fn long_run_multiplier_helper_matches_fixture() {
    let fx = load();
    let case = &fx["algebra_3var"];
    let coefs = coefs_from(&case["coefs"]);
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let c1 = long_run_multiplier(&refs).expect("multiplier");
    close_mat(
        c1.as_ref(),
        mat_from(&case["long_run_multiplier"]).as_ref(),
        "long_run_multiplier helper",
    );
}

#[test]
fn identifying_restriction_holds_mechanically() {
    // P1: long_run is lower-triangular (strict upper entries are 0);
    // P2: B B' == Sigma_u; P3: C1 B == long_run and
    // long_run long_run' == C1 Sigma_u C1'; P4: irf[0] == impact.
    let fx = load();
    for name in ["algebra_2var", "algebra_3var", "estimated_2var"] {
        let case = &fx[name];
        let res = run_case(case, false);
        let k = res.impact.nrows();
        let sigma = mat_from(&case["sigma_u"]);

        // P1
        for i in 0..k {
            for j in (i + 1)..k {
                assert!(
                    res.long_run[(i, j)].abs() < 1e-12,
                    "{name}: long_run[{i}][{j}] should be 0"
                );
            }
        }
        // P2: B B'
        let bbt = res.impact.as_ref() * res.impact.transpose();
        close_mat(bbt.as_ref(), sigma.as_ref(), &format!("{name} B B'"));
        // P3a: C1 B == long_run
        let c1b = res.long_run_multiplier.as_ref() * res.impact.as_ref();
        close_mat(
            c1b.as_ref(),
            res.long_run.as_ref(),
            &format!("{name} C1 B == LR"),
        );
        // P3b: LR LR' == C1 Sigma C1'
        let lrlrt = res.long_run.as_ref() * res.long_run.transpose();
        let c1s = res.long_run_multiplier.as_ref() * sigma.as_ref();
        let c1sc1t = c1s.as_ref() * res.long_run_multiplier.transpose();
        close_mat(
            lrlrt.as_ref(),
            c1sc1t.as_ref(),
            &format!("{name} LR LR' == C1 Sigma C1'"),
        );
        // P4: irf[0] == impact
        close_mat(
            res.irf[0].as_ref(),
            res.impact.as_ref(),
            &format!("{name} irf[0] == impact"),
        );
    }
}

#[test]
fn structural_fevd_rows_sum_to_one() {
    // P5: the structural forecast-error-variance shares at each horizon sum
    // to 1 across shocks (structural shocks are orthonormal).
    let fx = load();
    let case = &fx["algebra_3var"];
    let res = run_case(case, false);
    let k = res.impact.nrows();
    let mut contrib = vec![vec![0.0_f64; k]; k];
    for theta in &res.irf {
        for i in 0..k {
            for j in 0..k {
                contrib[i][j] += theta[(i, j)] * theta[(i, j)];
            }
        }
        for (i, row) in contrib.iter().enumerate() {
            let total: f64 = row.iter().sum();
            let share_sum: f64 = row.iter().map(|c| c / total).sum();
            assert!(
                (share_sum - 1.0).abs() < 1e-12,
                "fevd row {i} sums to {share_sum}"
            );
        }
    }
}

#[test]
fn explicit_default_pattern_equals_none() {
    // restrictions [(0,1)] IS the classic recursive lower-triangular pattern
    // for k=2, so it must reproduce the None (default) identification.
    let fx = load();
    let case = &fx["algebra_2var"];
    let coefs = coefs_from(&case["coefs"]);
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let sigma = mat_from(&case["sigma_u"]);
    let none = long_run_svar(&refs, sigma.as_ref(), 4, None, false).expect("none");
    let explicit =
        long_run_svar(&refs, sigma.as_ref(), 4, Some(&[(0, 1)]), false).expect("explicit");
    close_mat(explicit.impact.as_ref(), none.impact.as_ref(), "impact eq");
    close_mat(explicit.long_run.as_ref(), none.long_run.as_ref(), "LR eq");
}

#[test]
fn column_permutation_swaps_shocks() {
    // A column permutation of the default pattern relabels shocks. For k=2,
    // restriction [(0,0)] (shock 0 long-run-neutral for variable 0) is the
    // swap of the default [(0,1)], so the impact columns swap.
    let fx = load();
    let case = &fx["algebra_2var"];
    let coefs = coefs_from(&case["coefs"]);
    let refs: Vec<MatRef<'_, f64>> = coefs.iter().map(Mat::as_ref).collect();
    let sigma = mat_from(&case["sigma_u"]);
    let default = long_run_svar(&refs, sigma.as_ref(), 4, None, false).expect("default");
    let swapped = long_run_svar(&refs, sigma.as_ref(), 4, Some(&[(0, 0)]), false).expect("swapped");
    // Swapped long_run has its zero at [0][0] (shock 0 neutral for var 0).
    assert!(swapped.long_run[(0, 0)].abs() < 1e-12);
    for i in 0..2 {
        assert!(
            (swapped.impact[(i, 0)] - default.impact[(i, 1)]).abs() < TOL,
            "col 0 == default col 1"
        );
        assert!(
            (swapped.impact[(i, 1)] - default.impact[(i, 0)]).abs() < TOL,
            "col 1 == default col 0"
        );
    }
}

#[test]
fn singular_long_run_matrix_errors() {
    // A_1 = I makes D = I - A_1 = 0, singular: a unit root at frequency zero.
    let a1 = Mat::<f64>::identity(2, 2);
    let refs = [a1.as_ref()];
    let sigma = Mat::<f64>::identity(2, 2);
    let err = long_run_svar(&refs, sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");
}

#[test]
fn non_positive_definite_sigma_errors() {
    let a1 = Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 });
    let refs = [a1.as_ref()];
    // Indefinite "covariance" (negative eigenvalue).
    let sigma = Mat::from_fn(2, 2, |i, j| if i == j { 1.0 } else { 2.0 });
    let err = long_run_svar(&refs, sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");
}

#[test]
fn fewer_than_two_variables_errors() {
    let a1 = Mat::from_fn(1, 1, |_, _| 0.5);
    let refs = [a1.as_ref()];
    let sigma = Mat::from_fn(1, 1, |_, _| 1.0);
    let err = long_run_svar(&refs, sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");
}

#[test]
fn sigma_dimension_mismatch_errors() {
    let a1 = Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 });
    let refs = [a1.as_ref()];
    let sigma = Mat::<f64>::identity(3, 3);
    let err = long_run_svar(&refs, sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::Dimension { .. }), "{err:?}");
}

#[test]
fn nonfinite_inputs_error() {
    let a1 = Mat::from_fn(2, 2, |i, j| if i == 0 && j == 0 { f64::NAN } else { 0.1 });
    let refs = [a1.as_ref()];
    let sigma = Mat::<f64>::identity(2, 2);
    let err = long_run_svar(&refs, sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::NonFinite { .. }), "{err:?}");

    let a_ok = Mat::from_fn(2, 2, |i, j| if i == j { 0.3 } else { 0.0 });
    let refs2 = [a_ok.as_ref()];
    let bad_sigma = Mat::from_fn(2, 2, |i, _| if i == 0 { f64::INFINITY } else { 1.0 });
    let err2 = long_run_svar(&refs2, bad_sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err2, IdentError::NonFinite { .. }), "{err2:?}");
}

#[test]
fn empty_coefs_errors() {
    let sigma = Mat::<f64>::identity(2, 2);
    let err = long_run_svar(&[], sigma.as_ref(), 4, None, false).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");
}

#[test]
fn out_of_range_restriction_errors() {
    let a1 = Mat::from_fn(2, 2, |i, j| if i == j { 0.5 } else { 0.0 });
    let refs = [a1.as_ref()];
    let sigma = Mat::<f64>::identity(2, 2);
    let err = long_run_svar(&refs, sigma.as_ref(), 4, Some(&[(0, 5)]), false).unwrap_err();
    assert!(
        matches!(err, IdentError::RestrictionOutOfRange { .. }),
        "{err:?}"
    );
}

#[test]
fn non_triangularizable_pattern_errors() {
    let a1 = Mat::from_fn(3, 3, |i, j| if i == j { 0.4 } else { 0.0 });
    let refs = [a1.as_ref()];
    let sigma = Mat::<f64>::identity(3, 3);
    // Two shocks with the same (size-1) zero set: not a column permutation of
    // the strict-upper-triangular pattern.
    let err = long_run_svar(&refs, sigma.as_ref(), 4, Some(&[(0, 0), (0, 1)]), false).unwrap_err();
    assert!(matches!(err, IdentError::InvalidArgument { .. }), "{err:?}");

    // A zero set that is not a prefix {0..m-1}: shock 0 zero only at variable
    // 1 (for k=2).
    let a2 = Mat::from_fn(2, 2, |i, j| if i == j { 0.4 } else { 0.0 });
    let refs2 = [a2.as_ref()];
    let sigma2 = Mat::<f64>::identity(2, 2);
    let err2 = long_run_svar(&refs2, sigma2.as_ref(), 4, Some(&[(1, 0)]), false).unwrap_err();
    assert!(
        matches!(err2, IdentError::InvalidArgument { .. }),
        "{err2:?}"
    );
}
