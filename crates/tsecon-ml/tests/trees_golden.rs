//! Golden-value tests against `fixtures/trees.json`: scikit-learn 1.9.0
//! `DecisionTreeRegressor(criterion="squared_error", splitter="best",
//! max_features=None)` on float32-representable Friedman #1 data, several
//! `(max_depth, min_samples_leaf, min_samples_split)` settings, each
//! proven tie-free by the generator (see its header for why that is what
//! makes exact matching possible).
//!
//! Asserted: training fit and test predictions at 1e-12, `n_nodes` /
//! `n_leaves` / `depth` exact, `feature_importances_` at 1e-10, and the
//! sorted `(feature, threshold)` multiset — features exact, thresholds
//! 1e-12. The achieved figures are printed. The single-tree bridge
//! (`random_forest` with no resampling, all features, one tree) is then
//! held bit-for-bit to `regression_tree` on every fixture case, which is
//! how the forest inherits the golden.

mod common;

use common::{as_f64_vec, as_mat, assert_slice_close, load_fixture};
use serde_json::Value;
use tsecon_ml::{
    random_forest, regression_tree, ForestOptions, Importance, MaxFeatures, Resampling, TreeOptions,
};

fn tree_opts(case: &Value) -> TreeOptions {
    let params = &case["params"];
    TreeOptions {
        max_depth: params["max_depth"].as_u64().map(|d| d as usize),
        min_samples_leaf: params["min_samples_leaf"].as_u64().unwrap() as usize,
        min_samples_split: params["min_samples_split"].as_u64().unwrap() as usize,
    }
}

#[test]
fn golden_regression_tree_matches_sklearn() {
    let fx = load_fixture("trees.json");
    let x = as_mat(&fx["X"]);
    let x_test = as_mat(&fx["X_test"]);
    let y = as_f64_vec(&fx["y"]);
    let cases = fx["cases"].as_array().unwrap();
    assert!(cases.len() >= 7, "fixture must carry several settings");

    let mut worst_pred = 0.0f64;
    let mut worst_imp = 0.0f64;
    let mut worst_thr = 0.0f64;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let opts = tree_opts(case);
        let fit = regression_tree(x.as_ref(), &y, opts, Some(x_test.as_ref())).unwrap();

        assert_eq!(
            fit.n_nodes,
            case["n_nodes"].as_u64().unwrap() as usize,
            "{name}: n_nodes"
        );
        assert_eq!(
            fit.n_leaves,
            case["n_leaves"].as_u64().unwrap() as usize,
            "{name}: n_leaves"
        );
        assert_eq!(
            fit.depth,
            case["depth"].as_u64().unwrap() as usize,
            "{name}: depth"
        );

        let d = assert_slice_close(&fit.fitted, &as_f64_vec(&case["fitted"]), 1e-12, name);
        worst_pred = worst_pred.max(d);
        let predicted = fit.predicted.as_ref().unwrap();
        let d = assert_slice_close(predicted, &as_f64_vec(&case["predicted"]), 1e-12, name);
        worst_pred = worst_pred.max(d);

        let d = assert_slice_close(
            &fit.feature_importance,
            &as_f64_vec(&case["feature_importances"]),
            1e-10,
            name,
        );
        worst_imp = worst_imp.max(d);

        let expected: Vec<(usize, f64)> = case["splits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let a = pair.as_array().unwrap();
                (a[0].as_u64().unwrap() as usize, a[1].as_f64().unwrap())
            })
            .collect();
        assert_eq!(fit.splits.len(), expected.len(), "{name}: number of splits");
        for (k, ((f, t), (ef, et))) in fit.splits.iter().zip(&expected).enumerate() {
            assert_eq!(f, ef, "{name}: split {k} feature");
            let d = (t - et).abs();
            assert!(
                d <= 1e-12,
                "{name}: split {k} threshold {t} vs {et} (diff {d:e})"
            );
            worst_thr = worst_thr.max(d);
        }
    }
    println!(
        "regression_tree achieved: predictions {worst_pred:e}, importances {worst_imp:e}, \
         thresholds {worst_thr:e}"
    );
    assert!(worst_pred <= 1e-12 && worst_imp <= 1e-10 && worst_thr <= 1e-12);
}

/// `random_forest(bootstrap="none", max_features="all", n_trees=1)` is the
/// deterministic tree, bit for bit, on every golden setting — so the
/// forest's tree grower is the one scikit-learn pins. (`min_samples_split`
/// is fixed at 2 inside the forest, so only the fixture cases with that
/// value are bridged; the rest are covered by the tree test above.)
#[test]
fn golden_forest_single_tree_bridge_is_bit_identical() {
    let fx = load_fixture("trees.json");
    let x = as_mat(&fx["X"]);
    let x_test = as_mat(&fx["X_test"]);
    let y = as_f64_vec(&fx["y"]);
    let mut bridged = 0usize;
    for case in fx["cases"].as_array().unwrap() {
        let opts = tree_opts(case);
        if opts.min_samples_split != 2 {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let tree = regression_tree(x.as_ref(), &y, opts, Some(x_test.as_ref())).unwrap();
        let fopts = ForestOptions {
            n_trees: 1,
            max_features: MaxFeatures::All,
            max_depth: opts.max_depth,
            min_samples_leaf: opts.min_samples_leaf,
            resampling: Resampling::None,
            seed: 0,
            quantiles: None,
            importance: Importance::Impurity,
            importance_groups: None,
        };
        let forest = random_forest(x.as_ref(), &y, &fopts, Some(x_test.as_ref())).unwrap();
        assert_eq!(forest.fitted, tree.fitted, "{name}: fitted");
        assert_eq!(forest.predicted, tree.predicted, "{name}: predicted");
        assert_eq!(
            forest.importance.as_ref().unwrap(),
            &tree.feature_importance,
            "{name}: impurity importance"
        );
        assert!(forest.oob_prediction.is_none() && forest.oob_mse.is_none());
        assert_eq!(forest.max_features_resolved, x.ncols());
        bridged += 1;
    }
    assert!(
        bridged >= 5,
        "expected several bridged cases, got {bridged}"
    );
}

/// The bridge also holds at the settings the fixture cannot store —
/// unbounded depth with `min_samples_leaf = 1`, where two-row nodes tie
/// on every feature — because both paths break ties toward the lowest
/// feature index.
#[test]
fn forest_single_tree_bridge_holds_under_ties() {
    let fx = load_fixture("trees.json");
    let x = as_mat(&fx["X"]);
    let x_test = as_mat(&fx["X_test"]);
    let y = as_f64_vec(&fx["y"]);
    let tree = regression_tree(
        x.as_ref(),
        &y,
        TreeOptions::default(),
        Some(x_test.as_ref()),
    )
    .unwrap();
    assert!(tree.n_leaves > 100, "unbounded tree should be deep");
    let fopts = ForestOptions {
        n_trees: 1,
        max_features: MaxFeatures::All,
        max_depth: None,
        min_samples_leaf: 1,
        resampling: Resampling::None,
        seed: 7,
        quantiles: None,
        importance: Importance::None,
        importance_groups: None,
    };
    let forest = random_forest(x.as_ref(), &y, &fopts, Some(x_test.as_ref())).unwrap();
    assert_eq!(forest.fitted, tree.fitted);
    assert_eq!(forest.predicted, tree.predicted);
    // Interpolation: with min_samples_leaf = 1 and distinct rows the fit is
    // exact on the training set.
    for (f, v) in tree.fitted.iter().zip(&y) {
        assert!((f - v).abs() <= 1e-12);
    }
}
