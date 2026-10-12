//! Size arguments are refused by name, with their largest admissible value,
//! before anything is allocated (audit round 16, findings A4, A6, A7, A10,
//! B1, B3).
//!
//! Every count that sizes an allocation — a path, a grid, a network width,
//! an ensemble, a fold set — used to reach the allocator unchecked, and a
//! value of `2^31` aborted the process instead of raising. Each is now
//! checked against [`ML_MAX_VALUES`] with checked arithmetic. The largest
//! admissible value each refusal quotes is pinned here against a closed
//! form of the working set the estimator documents, computed independently
//! of the crate's bisection.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use tsecon_ml::faer::Mat;
use tsecon_ml::{
    boosting, expanding_origin_splits, kernel_regression, kernel_ridge, mlp_regression, pds_lasso,
    purged_kfold_splits, random_forest, regression_tree, regularization_path,
    rolling_origin_splits, BandwidthSpec, BoostStop, BoostingOptions, CoordDescentOptions,
    ForestOptions, Importance, KernelRegressionOptions, KernelRidgeOptions, KernelType, MlError,
    MlpOptions, MlpWeights, PathOptions, PdsAlpha, RegressionKernel, RegressionKind, TreeOptions,
    ML_MAX_VALUES,
};

const HUGE: [usize; 3] = [1 << 31, 1 << 40, usize::MAX];

/// Uniform draws on `[-0.5, 0.5)` from a 64-bit LCG.
fn draws(n: usize, mut s: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

fn design(n: usize, p: usize) -> (Mat<f64>, Vec<f64>) {
    let cols: Vec<Vec<f64>> = (0..p).map(|j| draws(n, 10 + j as u64)).collect();
    let x = Mat::from_fn(n, p, |i, j| cols[j][i]);
    let e = draws(n, 99);
    let y = (0..n)
        .map(|i| cols[0][i] - 0.5 * cols[1][i] + 0.3 * e[i])
        .collect();
    (x, y)
}

/// The refusal names `keyword = value` and quotes `can be at most {max}`.
fn assert_named(e: &MlError, keyword: &str, value: usize, max: usize) {
    let msg = e.to_string();
    assert!(
        matches!(e, MlError::InvalidValue { .. }),
        "{keyword}: not InvalidValue: {msg}"
    );
    assert!(
        msg.contains(&format!("{keyword} = {value} is too large")),
        "{msg}"
    );
    assert!(
        msg.contains(&format!("{keyword} can be at most {max} here")),
        "{msg}"
    );
    assert!(msg.contains("2^28-value (2 GiB) memory budget"), "{msg}");
}

#[test]
fn boosting_n_steps_is_budgeted() {
    let (n, p) = (200, 3);
    let (x, y) = design(n, p);
    // n_steps (n + p + 10) + n p + p^2 + 4 n + 5 p values.
    let max = (ML_MAX_VALUES - (n * p + p * p + 4 * n + 5 * p)) / (n + p + 10);
    assert_eq!(max, 1_260_253);
    for steps in HUGE {
        let opts = BoostingOptions {
            n_steps: steps,
            ..BoostingOptions::default()
        };
        let e = boosting(x.as_ref(), &y, opts, None).unwrap_err();
        assert_named(&e, "n_steps", steps, max);
    }
    let fit = boosting(
        x.as_ref(),
        &y,
        BoostingOptions {
            n_steps: 50,
            learning_rate: 0.5,
            stop: BoostStop::None,
        },
        None,
    )
    .unwrap();
    assert_eq!(fit.selected.len(), 50);
}

#[test]
fn lasso_path_n_lambdas_is_budgeted_and_cd_options_are_validated() {
    let (n, p) = (200, 3);
    let (x, y) = design(n, p);
    // n_lambdas (p + 8) + n p + 2 n values.
    let max = (ML_MAX_VALUES - (n * p + 2 * n)) / (p + 8);
    assert_eq!(max, 24_403_132);
    for m in HUGE {
        let opts = PathOptions {
            n_lambdas: m,
            ..PathOptions::default()
        };
        let e = regularization_path(x.as_ref(), &y, 1.0, opts).unwrap_err();
        assert_named(&e, "n_lambdas", m, max);
    }
    // max_iter = 0 is refused by name, not reported as non-convergence
    // after one sweep (and not silently accepted on a one-point path).
    for n_lambdas in [1, 20] {
        let opts = PathOptions {
            n_lambdas,
            cd: CoordDescentOptions {
                max_iter: 0,
                ..CoordDescentOptions::default()
            },
            ..PathOptions::default()
        };
        let e = regularization_path(x.as_ref(), &y, 1.0, opts).unwrap_err();
        assert_eq!(
            e.to_string(),
            "invalid argument: max_iter must be at least 1"
        );
    }
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let opts = PathOptions {
            cd: CoordDescentOptions {
                tol,
                ..CoordDescentOptions::default()
            },
            ..PathOptions::default()
        };
        let e = regularization_path(x.as_ref(), &y, 1.0, opts).unwrap_err();
        assert_eq!(
            e.to_string(),
            "invalid argument: tol must be finite and positive"
        );
    }
}

#[test]
fn pds_lasso_validates_cd_options_up_front() {
    let (x, y) = design(120, 4);
    let d = draws(120, 7);
    for alpha in [PdsAlpha::Bic, PdsAlpha::Fixed(0.05)] {
        let zero = CoordDescentOptions {
            max_iter: 0,
            ..CoordDescentOptions::default()
        };
        let e = pds_lasso(&y, &d, x.as_ref(), alpha, None, zero).unwrap_err();
        assert_eq!(
            e.to_string(),
            "invalid argument: max_iter must be at least 1"
        );
        let neg = CoordDescentOptions {
            tol: -1.0,
            ..CoordDescentOptions::default()
        };
        let e = pds_lasso(&y, &d, x.as_ref(), alpha, None, neg).unwrap_err();
        assert_eq!(
            e.to_string(),
            "invalid argument: tol must be finite and positive"
        );
    }
}

#[test]
fn kernel_ridge_rff_features_is_budgeted() {
    let (n, p) = (200, 3);
    let (x, y) = design(n, p);
    // For D >= n: D (p + 5) + D (2 n) + 2 n (n + D) values.
    let max = (ML_MAX_VALUES - 2 * n * n) / (p + 5 + 4 * n);
    assert_eq!(max, 332_123);
    for d in HUGE {
        let opts = KernelRidgeOptions {
            rff_features: Some(d),
            ..KernelRidgeOptions::default()
        };
        let e = kernel_ridge(x.as_ref(), &y, None, &opts).unwrap_err();
        assert_named(&e, "rff_features", d, max);
    }
}

#[test]
fn kernel_ridge_refuses_a_non_finite_polynomial_kernel_by_name() {
    let (n, p) = (40, 3);
    let (x, y) = design(n, p);
    let poly = |degree: f64, coef0: f64| KernelRidgeOptions {
        kernel: KernelType::Polynomial,
        degree,
        coef0,
        ..KernelRidgeOptions::default()
    };
    // (gamma <x, y> + 1)^(2^31) overflows: it used to be reported as a
    // rank-deficient kernel that a larger alpha would fix.
    let e = kernel_ridge(x.as_ref(), &y, None, &poly(2_147_483_648.0, 1.0)).unwrap_err();
    let msg = e.to_string();
    assert!(matches!(e, MlError::InvalidValue { .. }), "{msg}");
    assert!(msg.contains("polynomial kernel"), "{msg}");
    assert!(msg.contains("degree=2147483648"), "{msg}");
    assert!(msg.contains("Lower degree"), "{msg}");
    // A negative base under a non-integer degree is NaN: refused, whatever
    // alpha is.
    let e = kernel_ridge(x.as_ref(), &y, None, &poly(2.5, -1.0)).unwrap_err();
    assert!(e.to_string().contains("non-integer degree"), "{e}");
    // A finite training kernel with a test row whose base is negative used to
    // return a silent NaN in `predicted`.
    let ok = poly(2.5, 2.0);
    assert!(kernel_ridge(x.as_ref(), &y, None, &ok).is_ok());
    let xt = Mat::from_fn(2, p, |i, _| if i == 0 { -40.0 } else { 0.1 });
    let e = kernel_ridge(x.as_ref(), &y, Some(xt.as_ref()), &ok).unwrap_err();
    let msg = e.to_string();
    assert!(msg.contains("of the test kernel matrix"), "{msg}");
    assert!(msg.contains("NaN"), "{msg}");
}

#[test]
fn mlp_hidden_n_seeds_and_max_epochs_are_budgeted() {
    let (n, p) = (60, 2);
    let (x, y) = design(n, p);
    let base = MlpOptions {
        max_epochs: 3,
        n_seeds: 1,
        ..MlpOptions::default()
    };
    // One layer of width h, one member, no recorded path: weights
    // L = h (p + 2) + 1; values 8 L + 29 + 2 (h + 1) (3 n) + n (p + 1) + 3 n.
    let max_h = (ML_MAX_VALUES - (8 + 29 + 6 * n + n * (p + 1) + 3 * n)) / (8 * (p + 2) + 6 * n);
    for h in HUGE {
        let opts = MlpOptions {
            hidden: vec![h],
            ..base.clone()
        };
        let e = mlp_regression(x.as_ref(), &y, None, &opts).unwrap_err();
        assert_named(&e, "hidden[0]", h, max_h);
    }
    // The widest layer is the one named, with the other width held fixed.
    let e = mlp_regression(
        x.as_ref(),
        &y,
        None,
        &MlpOptions {
            hidden: vec![16, 1 << 40],
            ..base.clone()
        },
    )
    .unwrap_err();
    assert!(e.to_string().contains("hidden[1] = 1099511627776"), "{e}");

    // n_seeds members of hidden = [16]: L = 16 (p + 2) + 1 = 65, W = 17.
    let l = 16 * (p + 2) + 1;
    let fixed = 7 * l + 2 * 17 * (3 * n) + n * (p + 1) + 3 * n;
    let max_s = (ML_MAX_VALUES - fixed) / (l + 29);
    for s in HUGE {
        let opts = MlpOptions {
            n_seeds: s,
            ..base.clone()
        };
        let e = mlp_regression(x.as_ref(), &y, None, &opts).unwrap_err();
        assert_named(&e, "n_seeds", s, max_s);
    }

    // With no validation split every epoch runs and is recorded: max_epochs
    // is budgeted (each epoch adds 2 values per member).
    let max_e = (ML_MAX_VALUES - fixed - (l + 29)) / 2;
    for epochs in HUGE {
        let opts = MlpOptions {
            max_epochs: epochs,
            validation_fraction: 0.0,
            ..base.clone()
        };
        let e = mlp_regression(x.as_ref(), &y, None, &opts).unwrap_err();
        assert_named(&e, "max_epochs", epochs, max_e);
    }
    // With early stopping a huge max_epochs means "until the rule fires":
    // it is not budgeted, and nothing is reserved from it any more.
    let fit = mlp_regression(
        x.as_ref(),
        &y,
        None,
        &MlpOptions {
            max_epochs: 1 << 31,
            patience: Some(3),
            learning_rate: Some(0.05),
            ..base.clone()
        },
    )
    .unwrap();
    assert!(fit.converged[0]);
    assert!(fit.train_loss_path[0].len() < 100_000);
    // But every member records at least min(max_epochs, patience + 1)
    // epochs: a patience that leaves the rule no room to fire before an
    // unreachable max_epochs is refused by name (it used to run for hours
    // and then abort on the growing paths).
    for (max_epochs, patience, keyword, value, max) in [
        (
            1usize << 31,
            1usize << 31,
            "max_epochs",
            1usize << 31,
            max_e,
        ),
        (1 << 40, 1 << 31, "patience", 1 << 31, max_e - 1),
        (usize::MAX, 1 << 40, "patience", 1 << 40, max_e - 1),
    ] {
        let opts = MlpOptions {
            max_epochs,
            patience: Some(patience),
            ..base.clone()
        };
        let e = mlp_regression(x.as_ref(), &y, None, &opts).unwrap_err();
        assert_named(&e, keyword, value, max);
    }

    // Overflowing layer widths are an error, not a wrapped offset.
    assert!(MlpWeights::from_flat(&[usize::MAX, 4, 1], &[]).is_err());
}

#[test]
fn forest_n_trees_and_n_permutations_are_budgeted() {
    let (n, p) = (100, 3);
    let (x, y) = design(n, p);
    // Per tree 2 n + p + 29 values, fixed 9 + n (p + 4).
    let fixed = 9 + n * (p + 4);
    let per_tree = 2 * n + p + 29;
    let max_t = (ML_MAX_VALUES - fixed) / per_tree;
    for t in HUGE {
        let opts = ForestOptions {
            n_trees: t,
            ..ForestOptions::default()
        };
        let e = random_forest(x.as_ref(), &y, &opts, None).unwrap_err();
        assert_named(&e, "n_trees", t, max_t);
    }
    // 3 units x n_permutations permutations of n + 4 values on top of the
    // 20-tree forest.
    let forest = fixed + 20 * per_tree;
    let max_r = (ML_MAX_VALUES - forest) / (3 * (n + 4));
    for r in HUGE {
        let opts = ForestOptions {
            n_trees: 20,
            importance: Importance::BlockPermutation {
                permutation_block: Some(5),
                n_permutations: r,
            },
            ..ForestOptions::default()
        };
        let e = random_forest(x.as_ref(), &y, &opts, None).unwrap_err();
        assert_named(&e, "n_permutations", r, max_r);
    }
}

#[test]
fn tree_leaf_and_split_refusals_name_the_keyword() {
    let (x, y) = design(50, 2);
    let tree = |leaf: usize, split: usize| TreeOptions {
        max_depth: Some(3),
        min_samples_leaf: leaf,
        min_samples_split: split,
    };
    for leaf in [26usize, 1 << 31, usize::MAX] {
        let e = regression_tree(x.as_ref(), &y, tree(leaf, 2), None).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains(&format!("min_samples_leaf = {leaf}")), "{msg}");
        assert!(msg.contains("min_samples_leaf can be at most 25"), "{msg}");
        assert!(msg.contains("insufficient data: 50 observations"), "{msg}");
        let forest = ForestOptions {
            n_trees: 3,
            min_samples_leaf: leaf,
            ..ForestOptions::default()
        };
        let e = random_forest(x.as_ref(), &y, &forest, None).unwrap_err();
        assert!(
            e.to_string().contains("min_samples_leaf can be at most 25"),
            "{e}"
        );
    }
    for split in [51usize, 1 << 31, usize::MAX] {
        let e = regression_tree(x.as_ref(), &y, tree(1, split), None).unwrap_err();
        let msg = e.to_string();
        assert!(
            msg.contains(&format!("min_samples_split = {split}")),
            "{msg}"
        );
        assert!(msg.contains("min_samples_split can be at most 50"), "{msg}");
    }
    assert!(regression_tree(x.as_ref(), &y, tree(25, 50), None).is_ok());
}

#[test]
fn kernel_regression_block_refusal_names_block() {
    let xs = draws(40, 5);
    let x = Mat::from_fn(40, 1, |i, _| xs[i]);
    let y: Vec<f64> = xs
        .iter()
        .zip(draws(40, 6))
        .map(|(a, e)| a.sin() + 0.2 * e)
        .collect();
    let opts = |block: usize| KernelRegressionOptions {
        kind: RegressionKind::LocalLinear,
        kernel: RegressionKernel::Gaussian,
        bandwidth: BandwidthSpec::BlockCv { block: Some(block) },
    };
    // Local linear with k = 1 needs 2 + 2 block + 1 rows: block <= 18 at n = 40.
    for block in [19usize, 1 << 31, usize::MAX] {
        let e = kernel_regression(x.as_ref(), &y, None, &opts(block)).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains(&format!("block = {block}")), "{msg}");
        assert!(
            msg.contains("block can be at most 18 for these 40 rows"),
            "{msg}"
        );
    }
    // block = 18 passes the sample check (whatever the search then says).
    if let Err(e) = kernel_regression(x.as_ref(), &y, None, &opts(18)) {
        assert!(!e.to_string().contains("block can be at most"), "{e}");
        assert!(!matches!(e, MlError::InsufficientData { .. }), "{e}");
    }
}

#[test]
fn cv_splits_are_budgeted_on_n() {
    // Every index costs 6 values (its usize and the boxed integer and list
    // slot the Python binding returns it as) and every fold 80 (two Vec
    // headers, its dict and two lists).
    let budget = ML_MAX_VALUES as u128;
    // Expanding (train 50, horizon 2, step 5): F folds hold
    // 6 (52 F + 5 F (F - 1) / 2) + 80 F values; the largest n has the most
    // folds that fit, plus step - 1 rows that start no new fold.
    let expanding = |f: u128| 6 * (52 * f + 5 * f * (f - 1) / 2) + 80 * f;
    let mut folds = 1u128;
    while expanding(folds + 1) <= budget {
        folds += 1;
    }
    let max_n = 52 + (folds as usize - 1) * 5 + 4;
    assert_eq!(max_n, 21_136);
    for n in HUGE {
        let e = expanding_origin_splits(n, 50, 2, 5, 0).unwrap_err();
        assert_named(&e, "n", n, max_n);
    }
    assert!(expanding_origin_splits(max_n + 1, 50, 2, 5, 0).is_err());
    // Rolling (window 50, horizon 2, step 5): 6 * 52 + 80 = 392 values per
    // fold.
    let max_r = 52 + ((ML_MAX_VALUES / 392) - 1) * 5 + 4;
    for n in HUGE {
        let e = rolling_origin_splits(n, 50, 2, 5, 0).unwrap_err();
        assert_named(&e, "n", n, max_r);
    }
    // Purged k-fold: k (6 n + 80) + k + 1 values.
    let max_k = (ML_MAX_VALUES - 5 * 80 - 6) / 30;
    for n in HUGE {
        let e = purged_kfold_splits(n, 5, 0, 0).unwrap_err();
        assert_named(&e, "n", n, max_k);
    }
    assert_eq!(expanding_origin_splits(100, 50, 2, 5, 0).unwrap().len(), 10);
    assert_eq!(purged_kfold_splits(100, 5, 2, 1).unwrap().len(), 5);
}

#[test]
fn cv_splits_name_the_window_or_k_when_no_n_can_fit() {
    // One fold at n = train + horizon holds 6 (train + horizon) + 80 values,
    // so no n helps once train alone is past (2^28 - 80) / 6 - horizon.
    let max_train = (ML_MAX_VALUES - 80) / 6 - 2;
    for (train, n) in [(1usize << 31, 1usize << 32), (1 << 40, 1 << 41)] {
        let e = expanding_origin_splits(n, train, 2, 1, 0).unwrap_err();
        assert_named(&e, "train", train, max_train);
        let e = rolling_origin_splits(n, train, 2, 1, 0).unwrap_err();
        assert_named(&e, "train", train, max_train);
    }
    // A horizon wider than the training window is the one named.
    let e = rolling_origin_splits(1 << 32, 50, 1 << 31, 1, 0).unwrap_err();
    assert_named(&e, "horizon", 1 << 31, (ML_MAX_VALUES - 80) / 6 - 50);
    // Purged k-fold at n = k: k (6 k + 80) + k + 1 values.
    let mut max_folds = 2usize;
    while (max_folds + 1) * (6 * (max_folds + 1) + 80) + max_folds + 2 <= ML_MAX_VALUES {
        max_folds += 1;
    }
    for k in [1usize << 31, 1 << 40] {
        let e = purged_kfold_splits(k, k, 0, 0).unwrap_err();
        assert_named(&e, "k", k, max_folds);
    }
}
