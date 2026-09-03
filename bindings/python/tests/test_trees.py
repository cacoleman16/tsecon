"""Golden and behavioral tests for the tree and forest bindings.

Re-pins fixtures/trees.json (scikit-learn 1.9.0 `DecisionTreeRegressor`,
an independent-package golden — see the generator header for the
tie-break argument that makes exact matching possible) through the Python
surface, checks the single-tree forest bridge, the reproducibility
contract, the teaching errors with their house wording, the sentinel
refusals of inert kwargs, pandas coercion, the exact returned key sets,
and docstring/key consistency. The full forest's Monte-Carlo properties
live in crates/tsecon-ml/tests/trees_properties.rs; here they are only
smoke-checked so the binding layer stays cheap.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
TREES = json.loads((FIX / "trees.json").read_text())
X = np.array(TREES["X"])
X_TEST = np.array(TREES["X_test"])
Y = np.array(TREES["y"])

TREE_KEYS = {
    "fitted", "predicted", "n_nodes", "n_leaves", "depth", "feature_importance", "splits",
}
FOREST_KEYS = {
    "fitted", "predicted", "oob_prediction", "oob_mse", "importance",
    "importance_groups_resolved", "quantile_predictions", "n_trees", "max_features_resolved",
}


def _friedman(n, p, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, p))
    y = (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2
         + 10 * x[:, 3] + 5 * x[:, 4] + rng.standard_normal(n))
    return x, y


# ------------------------------------------------------------------ golden

@pytest.mark.parametrize("case", TREES["cases"], ids=lambda c: c["name"])
def test_regression_tree_matches_sklearn_fixture(case):
    p = case["params"]
    r = tsecon.regression_tree(
        X, Y,
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        min_samples_split=p["min_samples_split"],
        x_test=X_TEST,
    )
    assert set(r.keys()) == TREE_KEYS
    assert r["n_nodes"] == case["n_nodes"]
    assert r["n_leaves"] == case["n_leaves"]
    assert r["depth"] == case["depth"]
    np.testing.assert_allclose(r["fitted"], case["fitted"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["predicted"], case["predicted"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        r["feature_importance"], case["feature_importances"], rtol=0, atol=1e-10
    )
    assert abs(float(np.sum(r["feature_importance"])) - 1.0) < 1e-12
    got = r["splits"]
    assert len(got) == len(case["splits"])
    for (f, t), (ef, et) in zip(got, case["splits"]):
        assert f == ef
        assert abs(t - et) <= 1e-12
    # Sorted by (feature, threshold), as documented.
    assert got == sorted(got)


def test_fixture_meta_states_reference_and_tie_break():
    meta = TREES["_meta"]
    assert meta["sklearn"] == "1.9.0"
    assert "random_state" in meta["tie_break"]
    assert meta["excluded_settings"], "the generator records the tie-exercising settings"


@pytest.mark.parametrize("case", TREES["cases"], ids=lambda c: c["name"])
def test_forest_single_tree_bridge_reproduces_the_tree(case):
    p = case["params"]
    if p["min_samples_split"] != 2:
        pytest.skip("the forest fixes min_samples_split at 2")
    tree = tsecon.regression_tree(
        X, Y, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], x_test=X_TEST
    )
    forest = tsecon.random_forest(
        X, Y, n_trees=1, max_features="all", max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"], bootstrap="none", x_test=X_TEST,
        importance="impurity",
    )
    assert set(forest.keys()) == FOREST_KEYS
    np.testing.assert_array_equal(forest["fitted"], tree["fitted"])
    np.testing.assert_array_equal(forest["predicted"], tree["predicted"])
    np.testing.assert_array_equal(forest["importance"], tree["feature_importance"])
    np.testing.assert_array_equal(forest["importance_groups_resolved"], np.arange(X.shape[1]))
    assert forest["oob_prediction"] is None and forest["oob_mse"] is None
    assert forest["quantile_predictions"] is None
    assert forest["n_trees"] == 1
    assert forest["max_features_resolved"] == X.shape[1]
    # ... and hence sklearn.
    np.testing.assert_allclose(forest["predicted"], case["predicted"], rtol=0, atol=1e-12)


# --------------------------------------------------------------- behaviour

def test_forest_keys_shapes_and_defaults():
    x, y = _friedman(120, 6, 1)
    r = tsecon.random_forest(x, y, n_trees=40, x_test=x[:7], quantiles=[0.1, 0.5, 0.9])
    assert set(r.keys()) == FOREST_KEYS
    assert r["fitted"].shape == (120,)
    assert r["predicted"].shape == (7,)
    assert r["oob_prediction"].shape == (120,)
    assert np.isfinite(r["oob_mse"])
    assert r["importance"] is None and r["importance_groups_resolved"] is None
    assert r["quantile_predictions"].shape == (7, 3)
    assert np.all(np.diff(r["quantile_predictions"], axis=1) >= 0), "quantiles never cross"
    assert r["n_trees"] == 40
    assert r["max_features_resolved"] == 2  # "third" of 6
    # Without x_test the prediction slots are None.
    r0 = tsecon.random_forest(x, y, n_trees=5)
    assert r0["predicted"] is None and r0["quantile_predictions"] is None


def test_same_seed_bit_identical_different_seed_differs():
    x, y = _friedman(100, 5, 2)
    a = tsecon.random_forest(x, y, n_trees=30, seed=3, x_test=x[:5])
    b = tsecon.random_forest(x, y, n_trees=30, seed=3, x_test=x[:5])
    c = tsecon.random_forest(x, y, n_trees=30, seed=4, x_test=x[:5])
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    np.testing.assert_array_equal(a["predicted"], b["predicted"])
    assert a["oob_mse"] == b["oob_mse"]
    assert not np.array_equal(a["fitted"], c["fitted"])


def test_one_tree_leaves_in_bag_rows_without_oob_prediction():
    x, y = _friedman(80, 5, 5)
    r = tsecon.random_forest(x, y, n_trees=1, seed=1)
    nan = np.isnan(r["oob_prediction"])
    assert 0 < nan.sum() < 80
    assert np.isfinite(r["oob_mse"])


@pytest.mark.parametrize("bootstrap", ["block", "stationary"])
def test_block_bootstraps_run_and_max_features_variants(bootstrap):
    x, y = _friedman(90, 6, 6)
    r = tsecon.random_forest(x, y, n_trees=20, bootstrap=bootstrap, block_length=8, seed=2)
    assert np.isfinite(r["oob_mse"])
    for mf, resolved in (("sqrt", 2), ("third", 2), ("all", 6), (4, 4)):
        r = tsecon.random_forest(x, y, n_trees=3, max_features=mf, seed=2)
        assert r["max_features_resolved"] == resolved


def test_importance_modes_and_groups():
    x, y = _friedman(400, 8, 7)
    imp = tsecon.random_forest(x, y, n_trees=150, importance="impurity", seed=1)
    assert imp["importance"].shape == (8,)
    assert abs(float(np.sum(imp["importance"])) - 1.0) < 1e-12
    np.testing.assert_array_equal(imp["importance_groups_resolved"], np.arange(8))
    # The five relevant Friedman columns dominate the three noise columns.
    assert set(np.argsort(imp["importance"])[-5:]) == {0, 1, 2, 3, 4}

    perm = tsecon.random_forest(
        x, y, n_trees=150, importance="block_permutation", permutation_block=5,
        n_permutations=3, seed=1,
    )
    assert perm["importance"].shape == (8,)
    assert set(np.argsort(perm["importance"])[-5:]) == {0, 1, 2, 3, 4}

    # importance_groups is an integer LABEL vector, not data: the package-
    # level coercion layer would float64-ify it (random_forest is not yet in
    # _coerce._EXEMPT — that table is owned by another slice this wave), so
    # this test calls the compiled function directly; the integrator adds
    # `"random_forest": frozenset({"importance_groups"})` to _EXEMPT.
    groups = [0, 0, 0, 1, 1, 2, 2, 2]
    g = tsecon._core.random_forest(
        x, y, n_trees=150, importance="impurity", importance_groups=groups, seed=1
    )
    np.testing.assert_array_equal(g["importance_groups_resolved"], [0, 1, 2])
    assert g["importance"].shape == (3,)
    np.testing.assert_allclose(g["importance"][0], imp["importance"][:3].sum(), atol=1e-12)
    np.testing.assert_allclose(g["importance"][2], imp["importance"][5:].sum(), atol=1e-12)
    with pytest.raises(ValueError, match=r"expected 8, got 3"):
        tsecon._core.random_forest(
            x, y, n_trees=5, importance="impurity", importance_groups=[0, 1, 2]
        )


def test_pandas_inputs_are_coerced():
    pd = pytest.importorskip("pandas")
    x, y = _friedman(100, 5, 8)
    df = pd.DataFrame(x, columns=list("abcde"))
    s = pd.Series(y)
    t_np = tsecon.regression_tree(x, y, max_depth=3, x_test=x[:4])
    t_pd = tsecon.regression_tree(df, s, max_depth=3, x_test=df.iloc[:4])
    np.testing.assert_array_equal(t_np["predicted"], t_pd["predicted"])
    f_np = tsecon.random_forest(x, y, n_trees=10, x_test=x[:4])
    f_pd = tsecon.random_forest(df, s, n_trees=10, x_test=df.iloc[:4])
    np.testing.assert_array_equal(f_np["predicted"], f_pd["predicted"])
    # An integer data matrix is data (coerced), not a label vector.
    xi = (x * 1000).astype(np.int64)
    assert "fitted" in tsecon.regression_tree(xi, y, max_depth=2)


# ---------------------------------------------------------- teaching errors

def test_teaching_errors():
    x, y = _friedman(60, 5, 9)

    xn = x.copy()
    xn[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x$"):
        tsecon.regression_tree(xn, y)
    yn = y.copy()
    yn[0] = np.inf
    with pytest.raises(ValueError, match=r"in y$"):
        tsecon.random_forest(x, yn, n_trees=3)
    xt = x[:3].copy()
    xt[1, 0] = np.nan
    with pytest.raises(ValueError, match=r"in x_test$"):
        tsecon.random_forest(x, y, n_trees=3, x_test=xt)
    with pytest.raises(ValueError, match=r"x_test must have the same number of columns"):
        tsecon.regression_tree(x, y, x_test=x[:3, :2])

    # House insufficiency wording.
    with pytest.raises(ValueError, match=r"insufficient data: 7 observations, at least 10 required"):
        tsecon.random_forest(x[:7], y[:7], n_trees=3)
    with pytest.raises(ValueError, match=r"insufficient data: 3 observations, at least 4 required"):
        tsecon.regression_tree(x[:3], y[:3], min_samples_leaf=2)

    # Unknown string options list the accepted values.
    with pytest.raises(ValueError, match=r'expected "sqrt", "third", "all"'):
        tsecon.random_forest(x, y, n_trees=3, max_features="half")
    with pytest.raises(ValueError, match=r"max_features=9 is outside 1..=5"):
        tsecon.random_forest(x, y, n_trees=3, max_features=9)
    with pytest.raises(ValueError, match=r'expected "iid", "block", "stationary", or "none"'):
        tsecon.random_forest(x, y, n_trees=3, bootstrap="circular")
    with pytest.raises(ValueError, match=r'expected "none", "impurity", or "block_permutation"'):
        tsecon.random_forest(x, y, n_trees=3, importance="shap")

    # quantiles: domain and order, naming the fix.
    with pytest.raises(ValueError, match=r"strictly inside \(0, 1\).*quantiles=\[0.1, 0.5, 0.9\]"):
        tsecon.random_forest(x, y, n_trees=3, x_test=x[:2], quantiles=[0.0, 0.5])
    with pytest.raises(ValueError, match=r"strictly increasing"):
        tsecon.random_forest(x, y, n_trees=3, x_test=x[:2], quantiles=[0.9, 0.1])
    with pytest.raises(ValueError, match=r"quantiles were given but x_test was not"):
        tsecon.random_forest(x, y, n_trees=3, quantiles=[0.5])

    # Block lengths.
    with pytest.raises(ValueError, match=r"block_length=61 is outside 1..=60"):
        tsecon.random_forest(x, y, n_trees=3, bootstrap="block", block_length=61)
    with pytest.raises(ValueError, match=r"permutation_block=0 is outside 1..=60"):
        tsecon.random_forest(
            x, y, n_trees=3, importance="block_permutation", permutation_block=0
        )
    with pytest.raises(ValueError, match=r"n_trees"):
        tsecon.random_forest(x, y, n_trees=0)
    with pytest.raises(ValueError, match=r"min_samples_split must be at least 2"):
        tsecon.regression_tree(x, y, min_samples_split=1)
    with pytest.raises(ValueError, match=r"min_samples_leaf must be at least 1"):
        tsecon.regression_tree(x, y, min_samples_leaf=0)
    # Negative counts get the coercion layer's teaching upgrade.
    with pytest.raises(ValueError, match=r"nonnegative integer"):
        tsecon.random_forest(x, y, n_trees=-1)


def test_inert_kwargs_are_refused_and_live_where_documented():
    """Audit-round-10 sentinel convention: explicit-where-inert raises, the
    default call is bit-identical, and the kwarg is live where documented."""
    x, y = _friedman(80, 5, 10)
    base = tsecon.random_forest(x, y, n_trees=10, seed=1)

    # block_length: required for block/stationary, refused for iid/none.
    with pytest.raises(ValueError, match=r'bootstrap="block" needs block_length'):
        tsecon.random_forest(x, y, n_trees=10, bootstrap="block")
    with pytest.raises(ValueError, match=r'bootstrap="stationary" needs block_length'):
        tsecon.random_forest(x, y, n_trees=10, bootstrap="stationary")
    with pytest.raises(ValueError, match=r'block_length=8 has no effect under bootstrap="iid"'):
        tsecon.random_forest(x, y, n_trees=10, bootstrap="iid", block_length=8)
    with pytest.raises(ValueError, match=r'block_length=8 has no effect under bootstrap="none"'):
        tsecon.random_forest(x, y, n_trees=10, bootstrap="none", block_length=8)
    live = tsecon.random_forest(x, y, n_trees=10, seed=1, bootstrap="block", block_length=8)
    assert not np.array_equal(live["fitted"], base["fitted"])

    # importance_groups only acts with an importance.
    with pytest.raises(ValueError, match=r'importance_groups \(5 labels\) has no effect under importance="none"'):
        tsecon._core.random_forest(x, y, n_trees=10, importance_groups=[0, 0, 1, 1, 2])

    # permutation_block / n_permutations only act under block_permutation.
    for kw in ({"permutation_block": 4}, {"n_permutations": 3}):
        for mode in ("none", "impurity"):
            with pytest.raises(ValueError, match=rf'has no effect under importance="{mode}"'):
                tsecon.random_forest(x, y, n_trees=10, importance=mode, **kw)
    a = tsecon.random_forest(
        x, y, n_trees=10, seed=1, importance="block_permutation", permutation_block=1,
        n_permutations=2,
    )
    b = tsecon.random_forest(
        x, y, n_trees=10, seed=1, importance="block_permutation", permutation_block=8,
        n_permutations=2,
    )
    assert not np.array_equal(a["importance"], b["importance"]), "permutation_block is live"
    # ... and the forest itself is unchanged by the importance request.
    np.testing.assert_array_equal(a["fitted"], base["fitted"])

    # block_permutation needs out-of-bag rows.
    with pytest.raises(ValueError, match=r"bootstrap='none' has none"):
        tsecon.random_forest(x, y, n_trees=10, bootstrap="none", importance="block_permutation")

    # The default call is exactly the documented defaults.
    explicit = tsecon.random_forest(
        x, y, n_trees=10, max_features="third", max_depth=None, min_samples_leaf=5,
        bootstrap="iid", seed=1, importance="none",
    )
    np.testing.assert_array_equal(explicit["fitted"], base["fitted"])
    assert explicit["oob_mse"] == base["oob_mse"]


def test_docstrings_name_every_returned_key_and_the_gotchas():
    x, y = _friedman(60, 5, 11)

    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    t = tsecon.regression_tree(x, y, max_depth=2, x_test=x[:2])
    missing = set(t.keys()) - tokens(tsecon.regression_tree)
    assert not missing, f"regression_tree.__doc__ missing keys: {sorted(missing)}"

    f = tsecon.random_forest(
        x, y, n_trees=5, x_test=x[:2], quantiles=[0.5], importance="impurity"
    )
    missing = set(f.keys()) - tokens(tsecon.random_forest)
    assert not missing, f"random_forest.__doc__ missing keys: {sorted(missing)}"

    flat = re.sub(r"\s+", " ", tsecon.random_forest.__doc__)
    assert "OPTIMISTIC" in flat and "pseudo-out-of-sample" in flat
    assert "block permutation does NOT remove" in flat
    assert "scikit-learn 1.9.0" in flat
