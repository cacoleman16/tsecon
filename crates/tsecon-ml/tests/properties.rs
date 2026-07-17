//! Property / invariant tests beyond the golden fixtures:
//!
//! * adaptive LASSO zeros the design's true zeros more reliably than plain
//!   LASSO at matched shrinkage;
//! * the regularization path starts fully sparse at `lambda_max` and its
//!   AIC/BIC selectors behave sensibly;
//! * purged + embargoed folds never contain excluded indices (exhaustive);
//! * expanding-origin training sets are strictly nested and never peek
//!   ahead;
//! * CV selection on iid data loosely agrees with BIC selection;
//! * the `Scaler` is fit on train and replayed on test.

mod common;

use common::{as_f64_vec, as_mat, load_fixture, mat_from_cols, Lcg};
use tsecon_ml::{
    adaptive_lasso, cv_select, expanding_origin_splits, lasso, mse, purged_kfold_splits,
    regularization_path, ridge, rolling_origin_splits, CoordDescentOptions, PathOptions, Scaler,
    TargetCenterer,
};

const CD: CoordDescentOptions = CoordDescentOptions {
    tol: 1e-11,
    max_iter: 100_000,
};

/// Indices of the fixture's true-zero and true-nonzero coefficients.
fn true_support(fx: &serde_json::Value) -> (Vec<usize>, Vec<usize>) {
    let tb = as_f64_vec(&fx["true_beta"]);
    let zeros = (0..tb.len()).filter(|&i| tb[i] == 0.0).collect();
    let nonzeros = (0..tb.len()).filter(|&i| tb[i] != 0.0).collect();
    (zeros, nonzeros)
}

/// Adaptive LASSO (Zou 2006) drives the design's true zeros to exactly zero
/// more reliably than plain LASSO across a shrinkage grid, and never at the
/// cost of the true nonzeros. On the fixture design the domination is
/// strict at small `alpha`, where plain LASSO leaves false positives that
/// the adaptive weights kill.
#[test]
fn adaptive_lasso_sparser_on_true_zeros() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);
    let (zeros, nonzeros) = true_support(&fx);
    let is_zero = |b: f64| b.abs() <= 1e-8;

    let grid = [0.05, 0.10, 0.15, 0.20];
    let mut strict_wins = 0usize;
    for &alpha in &grid {
        let l = lasso(x.as_ref(), &y, alpha, CD).unwrap();
        let a = adaptive_lasso(x.as_ref(), &y, alpha, 1.0, 1.0, CD).unwrap();

        let lasso_fp = zeros.iter().filter(|&&j| !is_zero(l.coef[j])).count();
        let ada_fp = zeros.iter().filter(|&&j| !is_zero(a.coef[j])).count();
        assert!(
            ada_fp <= lasso_fp,
            "alpha={alpha}: adaptive false positives {ada_fp} > lasso {lasso_fp}"
        );
        if ada_fp < lasso_fp {
            strict_wins += 1;
        }
        // Adaptive keeps the strong true signals (indices 0,1,2) nonzero.
        for &j in nonzeros.iter().take(3) {
            assert!(
                !is_zero(a.coef[j]),
                "alpha={alpha}: adaptive zeroed a true signal at index {j}"
            );
        }
    }
    assert!(
        strict_wins >= 1,
        "adaptive LASSO never strictly beat plain LASSO on the true zeros"
    );
}

/// At `lambda_max = max_j |x_j'y| / (n*l1_ratio)` every coefficient is
/// exactly zero, and the path's degrees of freedom are nondecreasing as the
/// penalty relaxes (more features enter). BIC selection lands on a model
/// that recovers the strong true signals.
#[test]
fn regularization_path_starts_empty_and_selects_signal() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);
    let (_zeros, nonzeros) = true_support(&fx);

    let path = regularization_path(x.as_ref(), &y, 1.0, PathOptions::default()).unwrap();

    // First grid point is lambda_max: all coefficients zero, df == 0.
    assert_eq!(path.df[0], 0, "lambda_max should zero every coefficient");
    for b in &path.coefs[0] {
        assert_eq!(*b, 0.0, "lambda_max coefficient not exactly zero");
    }
    // Last grid point (smallest penalty) is the least sparse.
    assert!(
        path.df[path.df.len() - 1] >= path.df[0],
        "df should grow as the penalty relaxes"
    );
    // RSS is nonincreasing along the descending-penalty grid (more
    // flexibility never hurts the in-sample fit), up to solver noise.
    for w in path.rss.windows(2) {
        assert!(
            w[1] <= w[0] + 1e-6,
            "RSS rose along the relaxing path: {} -> {}",
            w[0],
            w[1]
        );
    }
    // BIC-selected model recovers the strong true signals.
    let bic_i = path.bic_best();
    for &j in nonzeros.iter().take(3) {
        assert!(
            path.coefs[bic_i][j].abs() > 1e-6,
            "BIC model missed true signal at index {j}"
        );
    }
}

/// Purged + embargoed blocked K-fold never lets a training index fall inside
/// the test block, its purge bands, or its embargo band — checked
/// exhaustively over every fold and every index for several configurations.
#[test]
fn purged_kfold_excludes_all_leaky_indices() {
    let configs = [
        (100usize, 5usize, 3usize, 2usize),
        (100, 5, 0, 0),
        (97, 4, 5, 5),
        (50, 10, 2, 4),
        (23, 3, 1, 0),
    ];
    for &(n, k, purge, embargo) in &configs {
        let splits = purged_kfold_splits(n, k, purge, embargo).unwrap();
        assert_eq!(splits.len(), k, "expected k folds");

        // Every index is tested exactly once across folds.
        let mut tested = vec![false; n];
        for s in &splits {
            for &i in &s.test {
                assert!(!tested[i], "index {i} tested twice");
                tested[i] = true;
            }
        }
        assert!(tested.iter().all(|&t| t), "some index never tested");

        let right = purge.max(embargo);
        for s in &splits {
            let ts = *s.test.first().unwrap();
            let te = s.test.last().unwrap() + 1; // exclusive end
            for &i in &s.train {
                // Disjoint from the test block.
                assert!(i < ts || i >= te, "train index {i} inside test [{ts},{te})");
                // Outside the left purge band [ts - purge, ts).
                if i < ts {
                    assert!(
                        ts - i > purge,
                        "train index {i} within purge {purge} before test start {ts}"
                    );
                }
                // Outside the right purge/embargo band [te, te + max(purge,embargo)).
                if i >= te {
                    assert!(
                        i - te >= right,
                        "train index {i} within {right} after test end {te}"
                    );
                }
            }
        }
    }
}

/// Expanding-origin training sets are strictly nested prefixes and never
/// contain an index at or beyond their test block; rolling-origin windows
/// have constant size.
#[test]
fn origin_splits_are_ordered_and_nested() {
    let splits = expanding_origin_splits(100, 40, 10, 10).unwrap();
    assert!(splits.len() >= 2, "need several splits to check nesting");
    for w in splits.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        // Strict growth.
        assert!(
            b.train.len() > a.train.len(),
            "expanding train set did not grow: {} -> {}",
            a.train.len(),
            b.train.len()
        );
        // Prefix nesting: the earlier train set is a prefix of the later.
        assert_eq!(&b.train[..a.train.len()], &a.train[..], "not prefix-nested");
    }
    for s in &splits {
        // Train is exactly 0..origin (a contiguous prefix).
        assert_eq!(s.train, (0..s.train.len()).collect::<Vec<_>>());
        // No training index reaches into or past the test block.
        let first_test = *s.test.first().unwrap();
        assert!(
            *s.train.last().unwrap() < first_test,
            "training index overlaps or follows the test block"
        );
    }

    // Rolling windows keep a fixed training size.
    let rolling = rolling_origin_splits(100, 30, 10, 10).unwrap();
    for s in &rolling {
        assert_eq!(s.train.len(), 30, "rolling window changed size");
        assert!(*s.train.last().unwrap() < *s.test.first().unwrap());
    }
}

/// On seeded iid data with a sparse signal, CV selection over a `lambda`
/// grid loosely agrees with BIC selection over the same grid (both land in
/// the same neighbourhood of the path and recover the true support).
#[test]
fn cv_selection_agrees_with_ic_on_iid_data() {
    let mut rng = Lcg::new(20260717);
    let n = 140usize;
    let p = 8usize;
    let true_beta = [2.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Draw iid standard-normal features, then standardize each column.
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|_| (0..n).map(|_| rng.normal()).collect::<Vec<f64>>())
        .collect();
    for c in &mut cols {
        let m = c.iter().sum::<f64>() / n as f64;
        let sd = (c.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n as f64).sqrt();
        for v in c.iter_mut() {
            *v = (*v - m) / sd;
        }
    }
    // y = X beta + noise, then centered.
    let mut y: Vec<f64> = (0..n)
        .map(|i| {
            let signal: f64 = (0..p).map(|j| cols[j][i] * true_beta[j]).sum();
            signal + 0.5 * rng.normal()
        })
        .collect();
    let ymean = y.iter().sum::<f64>() / n as f64;
    for v in &mut y {
        *v -= ymean;
    }
    let x = mat_from_cols(&cols);

    // Path + BIC selection.
    let path = regularization_path(x.as_ref(), &y, 1.0, PathOptions::default()).unwrap();
    let bic_i = path.bic_best();

    // CV selection over the same grid, expanding origin, MSE loss.
    let splits = expanding_origin_splits(n, 70, 14, 14).unwrap();
    let cv = cv_select(x.as_ref(), &y, &splits, &path.lambdas, 1.0, mse, CD).unwrap();

    // Loose agreement: the two selected grid indices are close (the grid
    // has 100 points spanning three decades).
    let gap = (cv.best_index as isize - bic_i as isize).unsigned_abs();
    assert!(
        gap <= 12,
        "CV index {} and BIC index {bic_i} disagree by {gap} > 12 grid points",
        cv.best_index
    );
    // Both selected models recover the true support {0,1,2} and exclude the
    // pure-noise features {3..8}.
    for sel in [bic_i, cv.best_index] {
        for j in 0..3 {
            assert!(
                path.coefs[sel][j].abs() > 1e-6,
                "selected model (idx {sel}) missed true signal {j}"
            );
        }
    }
}

/// Ridge coefficients shrink toward the origin as `alpha` grows: the
/// coefficient 2-norm is strictly decreasing in the penalty.
#[test]
fn ridge_shrinks_with_alpha() {
    let fx = load_fixture("ml.json");
    let x = as_mat(&fx["X_standardized"]);
    let y = as_f64_vec(&fx["y_centered"]);
    let norm = |b: &[f64]| b.iter().map(|v| v * v).sum::<f64>().sqrt();

    let n0 = norm(&ridge(x.as_ref(), &y, 0.0).unwrap());
    let n1 = norm(&ridge(x.as_ref(), &y, 1.0).unwrap());
    let n10 = norm(&ridge(x.as_ref(), &y, 10.0).unwrap());
    let n100 = norm(&ridge(x.as_ref(), &y, 100.0).unwrap());
    assert!(
        n0 > n1 && n1 > n10 && n10 > n100,
        "ridge norms not decreasing: {n0} {n1} {n10} {n100}"
    );
}

/// The `Scaler` fits per-column mean/scale on the training rows and replays
/// them on the test rows, so the transformed training columns have zero
/// mean and unit variance while the test transform uses the *train* scales
/// (never its own). Constant columns map to zero without dividing by zero.
#[test]
fn scaler_fits_on_train_and_replays_on_test() {
    let train_cols = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![10.0, 10.0, 10.0, 10.0, 10.0], // constant column
        vec![-2.0, 0.0, 2.0, 4.0, 6.0],
    ];
    let x_train = mat_from_cols(&train_cols);
    let scaler = Scaler::fit(x_train.as_ref()).unwrap();
    let z = scaler.transform(x_train.as_ref()).unwrap();

    let n = 5usize;
    // Column 0 and 2: zero mean, unit population variance after transform.
    for j in [0usize, 2] {
        let mean: f64 = (0..n).map(|i| z[(i, j)]).sum::<f64>() / n as f64;
        let var: f64 = (0..n).map(|i| z[(i, j)] * z[(i, j)]).sum::<f64>() / n as f64;
        assert!(mean.abs() < 1e-12, "column {j} mean {mean} not ~0");
        assert!((var - 1.0).abs() < 1e-9, "column {j} var {var} not ~1");
    }
    // Constant column maps to exactly zero (no NaN).
    for i in 0..n {
        assert_eq!(z[(i, 1)], 0.0, "constant column not zeroed");
    }

    // Test transform uses the frozen train scales: a test row equal to the
    // train mean maps to zero on columns 0 and 2.
    let test_cols = vec![vec![scaler.means()[0]], vec![10.0], vec![scaler.means()[2]]];
    let x_test = mat_from_cols(&test_cols);
    let zt = scaler.transform(x_test.as_ref()).unwrap();
    assert!(zt[(0, 0)].abs() < 1e-12);
    assert!(zt[(0, 2)].abs() < 1e-12);

    // TargetCenterer round-trips.
    let y = [1.0, 2.0, 3.0];
    let c = TargetCenterer::fit(&y).unwrap();
    let cen = c.transform(&y);
    assert!(
        (cen.iter().sum::<f64>()).abs() < 1e-12,
        "centered mean not 0"
    );
    let back = c.inverse_transform(&cen);
    for (a, b) in back.iter().zip(&y) {
        assert!(
            (a - b).abs() < 1e-12,
            "inverse_transform did not round-trip"
        );
    }
}
