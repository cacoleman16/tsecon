//! Refusals that must say what is actually wrong.
//!
//! * A sample whose residual degrees of freedom `T - m` fall below `k` has a
//!   singular residual covariance whatever the data; the fit used to blame a
//!   duplicated column (audit round 15, B15-VAR-1). It is now refused by
//!   sample size, with the largest lag order that fits.
//! * A bootstrap pseudo-sample whose iid resample drew too few distinct
//!   residual rows has a singular refit covariance, with positive probability
//!   in very small samples, although the full sample is fine. The first such
//!   replication is now refused by index, by counting the rows rather than by
//!   whatever rounding does to a Cholesky factorization (audit round 16,
//!   SVF-2).
//! * The non-finite refusal of a residual matrix explained itself with the
//!   VAR estimator's coefficients, which a normality test does not have
//!   (audit round 16, DOC16-5).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::Lcg;
use tsecon_bootstrap::{indices, par_replicate, BlockScheme};
use tsecon_linalg::faer::Mat;
use tsecon_var::{
    bootstrap_irf_bands, long_run_bootstrap_bands, ma_rep, normality_test_cholesky, select_order,
    LongRunBandSpec, Trend, VarError, VarSpec,
};

fn gaussian(rows: usize, k: usize, seed: u64) -> Mat<f64> {
    let mut rng = Lcg::new(seed);
    Mat::from_fn(rows, k, |_, _| rng.gaussian())
}

#[test]
fn too_few_residual_df_is_refused_by_sample_size_not_blamed_on_a_column() {
    // k = 3, lags = 2, a constant: m = 7 regressors, so T - m = n - 9.
    for n in [10usize, 11] {
        let e = VarSpec::new(2, Trend::Constant)
            .unwrap()
            .fit(gaussian(n, 3, 5).as_ref())
            .unwrap_err();
        assert!(
            matches!(
                e,
                VarError::SingularResidualCovariance {
                    needed: 12,
                    lags: 2,
                    neqs: 3,
                    ..
                }
            ),
            "{e:?}"
        );
        let text = e.to_string();
        assert!(
            text.contains("Sigma_u, the residual covariance, is singular"),
            "{text}"
        );
        assert!(text.contains(&format!("df_resid = {}", n - 9)), "{text}");
        assert!(
            text.contains("at least 12 rows") && text.contains("Try lags <= 1"),
            "{text}"
        );
        assert!(!text.contains("redundant column"), "{text}");
    }
    // T - m = k: fits.
    assert!(VarSpec::new(2, Trend::Constant)
        .unwrap()
        .fit(gaussian(12, 3, 5).as_ref())
        .is_ok());
    // No lag order fits: the hint says so.
    let e = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(gaussian(7, 3, 5).as_ref())
        .unwrap_err();
    assert!(
        e.to_string()
            .contains("Even lags=1 would need 8 rows on k=3 series"),
        "{e}"
    );
    // select_order's common subsample (here the lags = 2 candidate, one row
    // offset under max_lags = 3) gets the generic advice, naming the Python
    // keyword of var_select_order.
    let e = select_order(gaussian(11, 3, 5).as_ref(), 3, Trend::Constant).unwrap_err();
    assert!(
        matches!(e, VarError::SingularResidualCovariance { .. }),
        "{e:?}"
    );
    assert!(
        e.to_string()
            .contains("every larger one more: reduce max_lags"),
        "{e}"
    );
}

/// The replications of `SeedSequence(seed).spawn(n_boot)` whose iid resample
/// of `te` residual rows has at most `k` distinct rows (an intercept: the
/// refit residuals then have rank <= k - 1, so `sigma_u` is singular).
fn singular_replications(seed: u64, n_boot: usize, te: usize, k: usize) -> Vec<usize> {
    let draws: Vec<Vec<usize>> = par_replicate(seed, n_boot, |_, stream| {
        indices(BlockScheme::Iid, te, stream).unwrap()
    })
    .unwrap();
    (0..n_boot)
        .filter(|&b| {
            let mut d = draws[b].clone();
            d.sort_unstable();
            d.dedup();
            d.len() <= k
        })
        .collect()
}

#[test]
fn a_singular_pseudo_sample_is_reported_by_replication_index() {
    // k = 2, lags = 1, T = 6: T_eff - m = 2 = k, so the full sample fits, but
    // a resample of the 5 residual rows with at most 2 distinct rows (about
    // one draw in ten) has a singular refit covariance. It is refused by
    // count: exactly the first such replication, whatever the rounding.
    let data = gaussian(6, 2, 11);
    assert!(VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(data.as_ref())
        .is_ok());
    for seed in 0..4u64 {
        let singular = singular_replications(seed, 50, 5, 2);
        let first = *singular.first().expect("no singular resample in 50 draws");
        let irf = bootstrap_irf_bands(
            data.as_ref(),
            1,
            Trend::Constant,
            3,
            true,
            false,
            0.1,
            50,
            seed,
            false,
        );
        let mut spec = LongRunBandSpec::new(1, 3);
        spec.n_boot = 50;
        spec.seed = seed;
        let lr = long_run_bootstrap_bands(data.as_ref(), &spec);
        for e in [irf.unwrap_err(), lr.unwrap_err()] {
            let VarError::Replication { index, .. } = e else {
                panic!("expected a replication error, got {e:?}");
            };
            assert_eq!(index, first, "seed {seed}: {e}");
            let text = e.to_string();
            assert!(
                text.contains(&format!("bootstrap replication {first}"))
                    && text.contains("too few distinct residual rows")
                    && text.contains("Sigma_u is singular"),
                "{text}"
            );
            assert!(!text.contains("redundant column"), "{text}");
        }
        // Reduced-form responses never read sigma_u: the same draws all count.
        let reduced = bootstrap_irf_bands(
            data.as_ref(),
            1,
            Trend::Constant,
            3,
            false,
            false,
            0.1,
            50,
            seed,
            false,
        )
        .unwrap();
        assert_eq!(reduced.n_boot, 50);
    }
}

#[test]
fn non_finite_refusals_explain_their_own_consequence() {
    let mut x = gaussian(30, 2, 3);
    x[(3, 1)] = f64::NAN;
    let text = normality_test_cholesky(x.as_ref()).unwrap_err().to_string();
    assert!(
        text.contains("resid") && text.contains("row 3, column 1"),
        "{text}"
    );
    assert!(text.contains("normality statistic"), "{text}");
    assert!(
        !text.contains("VAR estimator") && !text.contains("coefficient"),
        "{text}"
    );

    let text = VarSpec::new(1, Trend::Constant)
        .unwrap()
        .fit(x.as_ref())
        .unwrap_err()
        .to_string();
    assert!(
        text.contains("the data matrix at row 3, column 1"),
        "{text}"
    );
    assert!(
        text.contains("VAR estimator has no missing-value handling"),
        "{text}"
    );

    let mut a = Mat::<f64>::zeros(2, 2);
    a[(1, 0)] = f64::INFINITY;
    let text = ma_rep(&[a], 3).unwrap_err().to_string();
    assert!(
        text.contains("coefficient matrices at row 1, column 0"),
        "{text}"
    );
    assert!(!text.contains("rows first"), "{text}");
}
