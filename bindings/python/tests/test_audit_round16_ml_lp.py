"""Regression pins for audit round 16: the local-projection and ML size
arguments.

Round 16's house-wide sweep found that a count of 2**31 or 2**40 in these
arguments reached a Rust allocation before any check and aborted the
interpreter (SIGABRT, uncatchable): the local-projection ``horizons`` (A2),
the ML grid, path, network and epoch sizes (A6, A10), the ensemble and
replicate counts ``n_seeds``, ``n_trees`` and ``n_permutations`` (A4), and
``cv_splits``' ``n`` (A7). Each is now refused with a ValueError that names
the keyword and its largest admissible value, before anything is allocated,
so these tests run in-process. The same round found refusals that blamed
the wrong cause (B3) or named no keyword (B1); those are pinned here too.

The largest admissible values are closed forms of the per-horizon row
count (pinned below) or of the documented working sets (pinned in the Rust
suite, ``crates/tsecon-ml/tests/size_budgets.rs``).
"""
import numpy as np
import pytest

import tsecon

HUGE = (2**31, 2**40)


def _series(n=200, seed=0):
    r = np.random.default_rng(seed)
    y = r.standard_normal(n)
    s = r.standard_normal(n)
    z = s + 0.5 * r.standard_normal(n)
    state = (np.arange(n) // 7 % 2).astype(float)
    return y, s, z, state


def _design(n=200, p=3, seed=1):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, p))
    y = x[:, 0] - 0.5 * x[:, 1] + 0.3 * r.standard_normal(n)
    return x, y


# ------------------------------------------------------------------ A2: LP
# Horizon h of each design keeps nobs(h) rows for k regressors and needs
# nobs > k; the refusal quotes the last horizon that passes. With n = 200:
#   lp (lag-augmented, p = 4): 200 - h - max(4, h) rows, 6 + h regressors -> 64
#   lp (se="hac", p = 2):      198 - h rows, 4 regressors                -> 193
#   lp_iv (p = 2):             198 - h rows, 4 regressors                -> 193
#   lp_multiplier (p = 2):     198 - h rows, 2 + 2p = 6 regressors       -> 191
#   lp_state (hac, p = 2):     198 - h rows, 2 (2 + p) = 8 regressors    -> 189
#   flp (K = 2, p = 2):        198 - h rows, 1 + K + p = 5 parameters    -> 192
def _lp_calls(h):
    y, s, z, state = _series()
    scores = np.column_stack([s, z])
    r = np.random.default_rng(3)
    curves = np.column_stack([s, z, s - z, r.standard_normal(200)])
    delta = np.array([1.0, 0.5, 0.25, 0.0])
    return {
        "lp": (lambda: tsecon.lp(y, s, horizons=h), 64),
        "lp hac": (lambda: tsecon.lp(y, s, horizons=h, n_lag_controls=2, se="hac"), 193),
        "lp sup-t band": (lambda: tsecon.lp(y, s, horizons=h, band="sup-t"), 64),
        "lp_iv": (lambda: tsecon.lp_iv(y, s, z, horizons=h, n_lag_controls=2), 193),
        "lp_multiplier": (lambda: tsecon.lp_multiplier(y, s, z, horizons=h, n_lag_controls=2), 191),
        "lp_state": (
            lambda: tsecon.lp_state(y, s, state, horizons=h, n_lag_controls=2, se="hac"),
            189,
        ),
        "flp": (lambda: tsecon.flp(y, scores, horizons=h, n_lag_controls=2), 192),
        "flp_scenario": (
            lambda: tsecon.flp_scenario(y, curves, delta, n_factors=2, horizons=h, n_lag_controls=2),
            192,
        ),
    }


@pytest.mark.parametrize("h", HUGE)
@pytest.mark.parametrize("name", sorted(_lp_calls(0)))
def test_lp_family_refuses_unreachable_horizons_by_name(name, h):
    call, largest = _lp_calls(h)[name]
    with pytest.raises(ValueError, match=rf"horizons can be at most {largest} for this sample"):
        call()


def test_lp_largest_admissible_horizon_is_admissible():
    y, s, z, _ = _series()
    assert len(tsecon.lp(y, s, horizons=64)["irf"]) == 65
    assert len(tsecon.lp_iv(y, s, z, horizons=193, n_lag_controls=2)["irf"]) == 194
    with pytest.raises(ValueError, match="horizons can be at most 64"):
        tsecon.lp(y, s, horizons=65)


@pytest.mark.parametrize("h", HUGE)
def test_quantile_lp_refuses_unreachable_horizons_by_name(h):
    # k = 2 + 2 n_lag_controls = 6 parameters, 198 - h rows: horizons <= 191.
    y, s, _, _ = _series()
    with pytest.raises(ValueError, match=rf"horizons = {h} reaches past the sample") as e:
        tsecon.quantile_lp(y, s, horizons=h, n_lag_controls=2, taus=[0.25, 0.5, 0.75])
    assert "horizons can be at most 191 for this series" in str(e.value)


@pytest.mark.parametrize("p", [66, 2**31])
def test_quantile_lp_names_n_lag_controls_when_horizon_zero_is_short(p):
    # B3: was "horizon 0 leaves nobs = 0 ... lower the maximum horizon".
    # Horizon 0 keeps 200 - p rows for 2 + 2p parameters: p <= 65.
    y, s, _, _ = _series()
    with pytest.raises(ValueError, match=rf"^n_lag_controls = {p} leaves no estimable horizon") as e:
        tsecon.quantile_lp(y, s, horizons=3, n_lag_controls=p)
    assert "n_lag_controls can be at most 65 here" in str(e.value)


# ------------------------------------------------------- A6 / A10: networks
@pytest.mark.parametrize(
    "hidden, named",
    [
        ((2**31,), "hidden[0] = 2147483648 is too large"),
        ([2**31], "hidden[0] = 2147483648 is too large"),
        ([2**40], "hidden[0] = 1099511627776 is too large"),
        (2**36, "hidden[0] = 68719476736 is too large"),
        ([16, 2**40], "hidden[1] = 1099511627776 is too large"),
        (np.array([2**62]), "hidden[0] = 4611686018427387904 is too large"),
        (np.array([2**31], dtype=np.int64), "hidden[0] = 2147483648 is too large"),
    ],
)
def test_mlp_hidden_widths_are_budgeted_by_name(hidden, named):
    x, y = _design(50, 2)
    with pytest.raises(ValueError) as e:
        tsecon.mlp_regression(x, y, hidden=hidden, max_epochs=2, n_seeds=1)
    assert named in str(e.value)
    assert "can be at most" in str(e.value)


@pytest.mark.parametrize("hidden", [[2**62], [2**62, 4], (2**62,), [np.int64(2**62)]])
def test_mlp_hidden_list_beyond_2_48_is_refused_not_a_panic(hidden):
    # A10: the list form used to be float64-ified past the 2**48 pre-flight
    # and raise a PanicException (a BaseException `except Exception` misses).
    x, y = _design(50, 2)
    with pytest.raises(ValueError, match=r"hidden=.*4611686018427387904.* at or beyond 2\*\*48"):
        tsecon.mlp_regression(x, y, hidden=hidden, max_epochs=2, n_seeds=1)


@pytest.mark.parametrize(
    "hidden, msg",
    [
        ((-1,), r"hidden\[0\] = -1 is not a non-negative integer"),
        ([4, -2], r"hidden\[1\] = -2 is not a non-negative integer"),
        ([1e300], r"hidden\[0\] = 1e\+300 is at or beyond 2\*\*48"),
    ],
)
def test_mlp_bad_hidden_elements_are_named(hidden, msg):
    x, y = _design(50, 2)
    with pytest.raises(ValueError, match=msg):
        tsecon.mlp_regression(x, y, hidden=hidden, max_epochs=2, n_seeds=1)


def test_mlp_hidden_integral_float_widths_still_accepted():
    x, y = _design(80, 2)
    for hidden in ([8.0, 4.0], (8.0,), np.array([8.0])):
        r = tsecon.mlp_regression(x, y, hidden=hidden, max_epochs=2, n_seeds=1)
        assert r["weights"][0]["coefs"][0].shape == (2, 8)


@pytest.mark.parametrize("n", HUGE)
def test_mlp_n_seeds_is_budgeted(n):
    # A4: one Philox substream per member used to be allocated first.
    x, y = _design(60, 2)
    with pytest.raises(ValueError, match=rf"n_seeds = {n} is too large.*n_seeds can be at most \d+ here"):
        tsecon.mlp_regression(x, y, hidden=(4,), max_epochs=2, n_seeds=n)


def test_mlp_max_epochs_runs_until_early_stopping_and_is_budgeted_without_it():
    x, y = _design(60, 2)
    # With a validation split a huge max_epochs means "until the rule fires";
    # the loss paths are no longer reserved from it.
    r = tsecon.mlp_regression(
        x, y, hidden=(4,), max_epochs=2**31, patience=3, learning_rate=0.05, n_seeds=1
    )
    assert r["converged"] == [True]
    assert len(r["train_loss_path"][0]) < 100_000
    # With no split every epoch runs and is recorded: refused by name.
    for e in HUGE:
        with pytest.raises(ValueError, match=rf"max_epochs = {e} is too large.*validation_fraction = 0"):
            tsecon.mlp_regression(
                x, y, hidden=(4,), max_epochs=e, validation_fraction=0.0, n_seeds=1
            )


# ------------------------------------------------ A6: paths, grids, features
@pytest.mark.parametrize("v", HUGE)
def test_boosting_n_steps_is_budgeted(v):
    # n_steps (n + p + 10) + n p + p^2 + 4 n + 5 p values with n = 200, p = 3.
    x, y = _design()
    with pytest.raises(ValueError, match=rf"n_steps = {v} is too large.*n_steps can be at most 1260253 here"):
        tsecon.boosting(x, y, n_steps=v)


@pytest.mark.parametrize("v", HUGE)
def test_lasso_path_n_lambdas_is_budgeted(v):
    # n_lambdas (p + 8) + n p + 2 n values with n = 200, p = 3.
    x, y = _design()
    with pytest.raises(ValueError, match=rf"n_lambdas = {v} is too large.*n_lambdas can be at most 24403132 here"):
        tsecon.lasso_path(x, y, n_lambdas=v)


@pytest.mark.parametrize("v", HUGE)
def test_kernel_ridge_rff_features_is_budgeted(v):
    # D (p + 5) + 2 n D + 2 n (n + D) values for D >= n = 200, p = 3.
    x, y = _design()
    with pytest.raises(ValueError, match=rf"rff_features = {v} is too large.*rff_features can be at most 332123 here"):
        tsecon.kernel_ridge(x, y, rff_features=v, seed=1)


# ---------------------------------------------------- A4: forest replicates
@pytest.mark.parametrize("v", HUGE)
def test_random_forest_n_trees_and_n_permutations_are_budgeted(v):
    x, y = _design(200, 3)
    with pytest.raises(ValueError, match=rf"n_trees = {v} is too large.*n_trees can be at most \d+ here"):
        tsecon.random_forest(x, y, n_trees=v, max_features=2, max_depth=4)
    with pytest.raises(ValueError, match=rf"n_permutations = {v} is too large.*n_permutations can be at most \d+ here"):
        tsecon.random_forest(
            x, y, n_trees=20, importance="block_permutation", n_permutations=v, permutation_block=5
        )


# ----------------------------------------------------------- A7: cv_splits
@pytest.mark.parametrize("v", HUGE)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scheme="expanding", train=50, horizon=2, step=5),
        dict(scheme="rolling", train=50, horizon=2, step=5),
        dict(scheme="purged_kfold", k=5),
    ],
)
def test_cv_splits_n_is_budgeted(v, kwargs):
    with pytest.raises(ValueError, match=rf"n = {v} is too large.*n can be at most \d+ here"):
        tsecon.cv_splits(n=v, **kwargs)


def test_cv_splits_purged_kfold_largest_n_closed_form():
    # k (n + 6) + k + 1 values: n <= (2**28 - 6) // 5 - 6 for k = 5.
    with pytest.raises(ValueError, match=rf"n can be at most {(2**28 - 6) // 5 - 6} here"):
        tsecon.cv_splits(n=2**31, scheme="purged_kfold", k=5)


# -------------------------------------------- B3: refusals with the wrong cause
def test_kernel_ridge_overflowing_polynomial_degree_names_degree():
    x, y = _design(60, 3)
    with pytest.raises(ValueError, match=r"polynomial kernel .*degree=2147483648") as e:
        tsecon.kernel_ridge(x, y, kernel="polynomial", degree=2**31)
    assert "Increase alpha" not in str(e.value)


def test_kernel_ridge_nan_polynomial_test_kernel_is_refused():
    # A negative base under degree 2.5 used to come back as a NaN prediction.
    x, y = _design(60, 3)
    xt = np.array([[-40.0, -40.0, -40.0], [0.1, 0.2, 0.3]])
    with pytest.raises(ValueError, match=r"NaN at entry \(0, \d+\) of the test kernel matrix"):
        tsecon.kernel_ridge(x, y, kernel="polynomial", degree=2.5, coef0=10.0, x_test=xt)


@pytest.mark.parametrize("n_lambdas", [1, 20])
def test_lasso_path_max_iter_zero_is_refused_by_name(n_lambdas):
    x, y = _design()
    with pytest.raises(ValueError, match=r"max_iter must be at least 1"):
        tsecon.lasso_path(x, y, n_lambdas=n_lambdas, max_iter=0)


@pytest.mark.parametrize("tol", [0.0, -1.0, float("nan")])
def test_lasso_path_and_pds_lasso_refuse_bad_tol_by_name(tol):
    x, y = _design()
    d = x[:, 2] + 0.1 * y
    with pytest.raises(ValueError, match=r"tol must be finite and positive"):
        tsecon.lasso_path(x, y, tol=tol)
    with pytest.raises(ValueError, match=r"tol must be finite and positive"):
        tsecon.pds_lasso(y, d, x, tol=tol)


def test_pds_lasso_max_iter_zero_is_refused_by_name():
    x, y = _design()
    d = x[:, 2] + 0.1 * y
    with pytest.raises(ValueError, match=r"max_iter must be at least 1"):
        tsecon.pds_lasso(y, d, x, hac_lags=3, max_iter=0)


# ------------------------------------------------- B1: refusals with no keyword
def test_kernel_regression_block_refusal_names_block():
    # Local linear on k = 1 needs 2 + 2 block + 1 rows: block <= 98 at n = 200.
    x, y = _design(200, 2)
    with pytest.raises(ValueError, match=r"block = 2147483648 .*block can be at most 98 for these 200 rows"):
        tsecon.kernel_regression(x[:, 0], y, bandwidth_method="block_cv", block=2**31)


def test_tree_refusals_name_min_samples_leaf_and_split():
    x, y = _design(200, 3)
    with pytest.raises(ValueError, match=r"min_samples_leaf = 2147483648 .*min_samples_leaf can be at most 100"):
        tsecon.regression_tree(x, y, max_depth=3, min_samples_leaf=2**31)
    with pytest.raises(ValueError, match=r"min_samples_split = 2147483648 .*min_samples_split can be at most 200"):
        tsecon.regression_tree(x, y, max_depth=3, min_samples_split=2**31)
    with pytest.raises(ValueError, match=r"min_samples_leaf = 2147483648 .*min_samples_leaf can be at most 100"):
        tsecon.random_forest(x, y, n_trees=20, max_features=2, max_depth=4, min_samples_leaf=2**31)
    # The house insufficiency wording is kept inside the named refusal.
    with pytest.raises(ValueError, match=r"insufficient data: 7 observations, at least 10 required"):
        tsecon.random_forest(x[:7], y[:7], n_trees=3)
