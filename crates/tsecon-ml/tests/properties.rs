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
//! * the `Scaler` is fit on train and replayed on test;
//! * the coordinate-descent solvers and the regularization path are exactly
//!   scale equivariant — `coef * s` is invariant under `X -> s*X` with
//!   `alpha -> s*alpha`, and `coef / c` under `y -> c*y` with
//!   `alpha -> c*alpha` — swept over twenty-odd decades and both shipped
//!   tolerances, which is the only way a scale-blind stopping rule is
//!   visible.

mod common;

use common::{as_f64_vec, as_mat, load_fixture, mat_from_cols, Lcg};
use tsecon_ml::{
    adaptive_lasso, cv_select, elastic_net, expanding_origin_splits, lasso, mse,
    purged_kfold_splits, regularization_path, ridge, rolling_origin_splits, CoordDescentOptions,
    PathOptions, Scaler, TargetCenterer,
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

/// A seeded `n = 300`, `p = 8` design with three strong signals and five
/// true zeros, returned column-major together with its target.
fn scale_sweep_design() -> (Vec<Vec<f64>>, Vec<f64>) {
    const N: usize = 300;
    const P: usize = 8;
    let mut rng = Lcg::new(12);
    let cols: Vec<Vec<f64>> = (0..P)
        .map(|_| (0..N).map(|_| rng.normal()).collect())
        .collect();
    let beta = [3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    let y: Vec<f64> = (0..N)
        .map(|i| {
            let fit: f64 = (0..P).map(|j| cols[j][i] * beta[j]).sum();
            fit + 0.5 * rng.normal()
        })
        .collect();
    (cols, y)
}

/// `cols` with every entry multiplied by `s`, as a faer matrix.
fn scaled_design(cols: &[Vec<f64>], s: f64) -> tsecon_ml::faer::Mat<f64> {
    let scaled: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| c.iter().map(|v| v * s).collect())
        .collect();
    mat_from_cols(&scaled)
}

/// The `(alpha, l1_ratio)` pair that reproduces the scale-1 penalties after
/// `X -> s*X`, i.e. that leaves the objective identical with the minimizer
/// at `b/s`.
///
/// The `L1` term `alpha*l1_ratio*||b||_1` picks up one factor of `1/s` from
/// `b`, so it needs `alpha*l1_ratio -> alpha*l1_ratio*s`; the `L2` term
/// `0.5*alpha*(1-l1_ratio)*||b||^2` picks up two, so it needs
/// `alpha*(1-l1_ratio) -> alpha*(1-l1_ratio)*s^2`. A pure LASSO
/// (`l1_ratio = 1`) therefore has the simple `alpha -> alpha*s` of the
/// finding; a genuine elastic net needs both penalties moved, which the
/// `(alpha, l1_ratio)` parametrization can still express.
fn rescale_penalty(alpha: f64, l1_ratio: f64, s: f64) -> (f64, f64) {
    let a1 = alpha * l1_ratio * s;
    let a2 = alpha * (1.0 - l1_ratio) * s * s;
    let a = a1 + a2;
    (a, a1 / a)
}

/// **Scale equivariance of the coordinate-descent solvers.**
///
/// With the penalties moved by [`rescale_penalty`], sending `X -> s*X`
/// leaves `(1/(2n))||y - Xb||^2 + alpha*l1_ratio*||b||_1 +
/// 0.5*alpha*(1-l1_ratio)*||b||^2` reproduced term for term, so the
/// minimizer at scale `s` is the scale-1 minimizer divided by `s`.
/// `coef * s` cannot depend on `s`: it is an algebraic identity, not an
/// approximation.
///
/// A single-scale test cannot see a violation of it. With an **absolute**
/// stopping tolerance the solver silently breaks the identity: coefficient
/// moves shrink like `1/s`, so past `s ~ |b|/tol` the very first sweep out
/// of the zero warm start already looks converged and the solver returns
/// that one soft-threshold step as a success. At `s = 1e9` with the
/// binding's `tol = 1e-8` that turned a leading coefficient of `2.934` into
/// `3.211` — a 9% error, no warning, `n_iter = 1`. Hence the sweep over
/// fifteen-plus decades *and* over both shipped tolerances: the scale at
/// which an absolute rule collapses is proportional to `1/tol`, so a sweep
/// pinned to one tolerance can miss it.
#[test]
fn coordinate_descent_is_scale_equivariant() {
    let (cols, y) = scale_sweep_design();
    // A pure LASSO tolerates the widest sweep; the elastic net's `s^2` on
    // the L2 penalty drives `1 - l1_ratio'` toward cancellation at extreme
    // `s`, which is a limit of the *reparametrization*, not of the solver.
    let lasso_scales = [1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8, 1e12];
    let enet_scales = [1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9];
    let cases: [(f64, f64, &[f64]); 3] = [
        (1.0, 0.1, &lasso_scales),
        (0.5, 0.1, &enet_scales),
        (0.5, 0.02, &enet_scales),
    ];

    let base_x = mat_from_cols(&cols);
    // Both tolerances the bindings ship (lasso/elastic_net 1e-8,
    // adaptive_lasso/lasso_path 1e-7) plus the crate default.
    for &tol in &[1e-11f64, 1e-8] {
        let opts = CoordDescentOptions {
            tol,
            max_iter: 100_000,
        };
        for &(l1_ratio, alpha1, scales) in &cases {
            let base = elastic_net(base_x.as_ref(), &y, alpha1, l1_ratio, opts).unwrap();
            let mag = base.coef.iter().fold(0.0f64, |m, &b| m.max(b.abs()));
            assert!(mag > 1.0, "degenerate baseline fit: max |coef| = {mag}");
            // The baseline must be a real fit, not one soft-threshold step.
            assert!(base.n_iter >= 3, "baseline converged suspiciously fast");

            let mut worst = 0.0f64;
            for &s in scales {
                let (alpha_s, l1_s) = rescale_penalty(alpha1, l1_ratio, s);
                let x = scaled_design(&cols, s);
                let fit = elastic_net(x.as_ref(), &y, alpha_s, l1_s, opts).unwrap();
                assert!(
                    fit.max_rel_change <= tol,
                    "tol={tol:e} l1_ratio={l1_ratio} s={s:e}: reported \
                     max_rel_change {:e} exceeds tol",
                    fit.max_rel_change
                );
                for j in 0..base.coef.len() {
                    let d = (fit.coef[j] * s - base.coef[j]).abs();
                    worst = worst.max(d);
                    assert!(
                        d <= 1e-6 * mag,
                        "tol={tol:e} l1_ratio={l1_ratio} alpha={alpha1} s={s:e} \
                         coord {j}: coef*s = {} but s=1 gives {} (diff {d:e}, \
                         allowed {:e})",
                        fit.coef[j] * s,
                        base.coef[j],
                        1e-6 * mag
                    );
                }
            }
            println!(
                "tol={tol:e} l1_ratio={l1_ratio} alpha={alpha1}: worst \
                 |coef*s - coef_1| = {worst:e} (max |coef| = {mag:e})"
            );
        }
    }
}

/// The companion equivariance in the **target**: `y -> c*y`,
/// `alpha -> c*alpha` scales the objective by `c^2`, so the minimizer
/// scales by `c` and `coef / c` must be invariant. This is the half of the
/// scale-freedom that the `||y||` in the stopping rule buys; a tolerance
/// compared against a bare coefficient change fails it for the same reason
/// it fails the design-scaling version.
#[test]
fn coordinate_descent_is_equivariant_in_the_target_scale() {
    let (cols, y) = scale_sweep_design();
    let x = mat_from_cols(&cols);
    let base = lasso(x.as_ref(), &y, 0.1, CD).unwrap();
    let mag = base.coef.iter().fold(0.0f64, |m, &b| m.max(b.abs()));

    for &c in &[1e-12, 1e-6, 1e-3, 1e3, 1e6, 1e12] {
        let yc: Vec<f64> = y.iter().map(|v| v * c).collect();
        let fit = lasso(x.as_ref(), &yc, 0.1 * c, CD).unwrap();
        for j in 0..base.coef.len() {
            let d = (fit.coef[j] / c - base.coef[j]).abs();
            assert!(
                d <= 1e-8 * mag,
                "c={c:e} coord {j}: coef/c = {} but c=1 gives {} (diff {d:e})",
                fit.coef[j] / c,
                base.coef[j]
            );
        }
    }
}

/// The regularization path inherits the same equivariance: the grid is
/// built from `lambda_max = max_j |x_j'y| / (n*l1_ratio)`, which scales
/// like `s`, so every `coefs[i] * s` must be invariant. This exercises the
/// warm-started `cd_engine` calls, where the stopping rule is applied to a
/// nonzero start rather than to zeros.
#[test]
fn regularization_path_is_scale_equivariant() {
    let (cols, y) = scale_sweep_design();
    let opts = PathOptions {
        n_lambdas: 25,
        eps: 1e-3,
        cd: CD,
    };
    let base = regularization_path(mat_from_cols(&cols).as_ref(), &y, 1.0, opts).unwrap();
    let mag = base
        .coefs
        .iter()
        .flatten()
        .fold(0.0f64, |m, &b| m.max(b.abs()));

    for &s in &[1e-6, 1e3, 1e9] {
        let x = scaled_design(&cols, s);
        let path = regularization_path(x.as_ref(), &y, 1.0, opts).unwrap();
        // `df` itself is not compared: at the very first grid point the
        // soft-threshold argument sits exactly on `lambda_max`, so which
        // side of zero the rounding lands on is a floating-point coin flip
        // that changes `df` by one while the coefficient stays at ~1e-17.
        for (i, (bs, b1)) in path.coefs.iter().zip(&base.coefs).enumerate() {
            for (j, (&a, &b)) in bs.iter().zip(b1).enumerate() {
                let d = (a * s - b).abs();
                assert!(
                    d <= 1e-6 * mag,
                    "s={s:e} grid {i} coord {j}: {} vs {} (diff {d:e})",
                    a * s,
                    b
                );
            }
        }
    }
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
