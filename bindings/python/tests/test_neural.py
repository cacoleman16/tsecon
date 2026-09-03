"""Binding tests for the neural slice: `mlp_regression` and
`echo_state_network`.

The golden mechanics (forward pass, objective, gradient at scikit-learn's
fitted weights; the reservoir state path and readout on an explicit
reservoir) are pinned on the Rust side against fixtures/neural.json. This
file exercises the Python surface: marshalling and the exact returned key
sets, the `hidden` tuple/list/int/array contract through the coercion
layer, ValueError propagation with the teaching text (array naming,
accepted-value lists, the two-layer limit, the lbfgs sentinels, the
insufficiency wording with the validation split / washout counted),
pandas coercion, the seed and leakage contracts, and the Monte-Carlo /
property claims re-checked through Python on the release wheel — where
the wall-clock budgets are measured too.
"""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
NEURAL = json.loads((FIX / "neural.json").read_text())

MLP_KEYS = {
    "fitted", "predicted", "member_predictions", "train_loss_path",
    "validation_loss_path", "best_epoch", "converged", "n_parameters", "weights",
    "n_train", "n_validation", "x_mean", "x_scale", "y_mean", "y_scale", "solver",
    "activation",
}
ESN_KEYS = {
    "fitted", "predicted", "readout", "spectral_radius_achieved", "reservoir_size",
    "n_washout", "n_train",
}

ACT = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
}


# ------------------------------------------------------------------ data


def sin_ar1(seed, n, sigma=0.2):
    """y_t = sin(2 y_{t-1}) + sigma e_t as (lag matrix, target)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n + 101)
    e = rng.standard_normal(n + 101)
    for t in range(1, y.size):
        y[t] = np.sin(2.0 * y[t - 1]) + sigma * e[t]
    y = y[100:]
    return y[:n].reshape(-1, 1), y[1:n + 1]


def narma10(seed, n):
    rng = np.random.default_rng(seed)
    u = 0.5 * rng.uniform(size=n)
    y = np.zeros(n)
    for t in range(9, n - 1):
        y[t + 1] = 0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum() + 1.5 * u[t - 9] * u[t] + 0.1
    return u.reshape(-1, 1), y


def r2(pred, truth, benchmark=None):
    """Out-of-sample R^2; with `benchmark` (the training mean) it is the
    Campbell-Thompson (2008) definition, which stays meaningful when the
    test window sits in one basin of the bistable sin(2y) map."""
    b = truth.mean() if benchmark is None else benchmark
    return 1.0 - np.sum((pred - truth) ** 2) / np.sum((truth - b) ** 2)


def nrmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2) / np.var(truth))


def forward(weights, act, x):
    a = x
    coefs, intercepts = weights["coefs"], weights["intercepts"]
    for i in range(len(coefs)):
        z = a @ coefs[i] + intercepts[i]
        a = act(z) if i < len(coefs) - 1 else z
    return a[:, 0]


# --------------------------------------------------------- mlp: surface


def test_mlp_keys_shapes_and_weight_layout_reproduce_fitted():
    x, y = sin_ar1(1, 260)
    x_test = x[-20:]
    r = tsecon.mlp_regression(x[:240], y[:240], hidden=(8,), max_epochs=30, n_seeds=3,
                              x_test=x_test)
    assert set(r) == MLP_KEYS
    assert r["fitted"].shape == (240,)
    assert r["predicted"].shape == (20,)
    assert r["member_predictions"].shape == (3, 20)
    assert len(r["train_loss_path"]) == 3 and len(r["validation_loss_path"]) == 3
    assert all(p.shape == (30,) for p in r["train_loss_path"])
    assert all(p.shape == (30,) for p in r["validation_loss_path"])
    assert r["best_epoch"] == [int(e) for e in r["best_epoch"]]
    assert all(isinstance(c, bool) for c in r["converged"])
    assert r["n_parameters"] == 1 * 8 + 8 + 8 * 1 + 1
    assert (r["n_train"], r["n_validation"]) == (192, 48)
    assert r["x_mean"].shape == (1,) and r["x_scale"].shape == (1,)
    assert (r["solver"], r["activation"]) == ("adam", "tanh")
    # The returned weights are in sklearn's (fan_in, fan_out) layout and,
    # with the returned training-row scaler, reproduce the predictions
    # exactly: the ensemble mean of each member's forward pass.
    assert len(r["weights"]) == 3
    w0 = r["weights"][0]
    assert w0["coefs"][0].shape == (1, 8) and w0["coefs"][1].shape == (8, 1)
    assert w0["intercepts"][0].shape == (8,) and w0["intercepts"][1].shape == (1,)
    xs = (x[:240] - r["x_mean"]) / r["x_scale"]
    members = np.array([forward(w, np.tanh, xs) * r["y_scale"] + r["y_mean"]
                        for w in r["weights"]])
    np.testing.assert_allclose(members.mean(0), r["fitted"], rtol=0, atol=1e-10)
    xts = (x_test - r["x_mean"]) / r["x_scale"]
    members_t = np.array([forward(w, np.tanh, xts) * r["y_scale"] + r["y_mean"]
                          for w in r["weights"]])
    np.testing.assert_allclose(members_t, r["member_predictions"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(members_t.mean(0), r["predicted"], rtol=0, atol=1e-10)
    # The scaler is the training rows' population moments.
    np.testing.assert_allclose(r["x_mean"], x[:192].mean(0))
    np.testing.assert_allclose(r["x_scale"], x[:192].std(0))
    assert r["y_mean"] == pytest.approx(y[:192].mean())
    assert r["y_scale"] == pytest.approx(y[:192].std())


def test_mlp_without_x_test_returns_none_predictions():
    x, y = sin_ar1(2, 120)
    r = tsecon.mlp_regression(x, y, max_epochs=5, n_seeds=2)
    assert r["predicted"] is None and r["member_predictions"] is None
    assert set(r) == MLP_KEYS


@pytest.mark.parametrize("hidden", [(8, 4), [8, 4], np.array([8, 4]), np.array([8, 4], dtype=np.int32)])
def test_mlp_hidden_accepts_tuple_list_and_int_arrays(hidden):
    x, y = sin_ar1(3, 120)
    r = tsecon.mlp_regression(x, y, hidden=hidden, max_epochs=3, n_seeds=1)
    assert r["n_parameters"] == 1 * 8 + 8 + 8 * 4 + 4 + 4 * 1 + 1
    assert r["weights"][0]["coefs"][1].shape == (8, 4)


def test_mlp_hidden_int_is_one_layer_and_bad_values_teach():
    x, y = sin_ar1(3, 120)
    r = tsecon.mlp_regression(x, y, hidden=6, max_epochs=3, n_seeds=1)
    assert r["n_parameters"] == 6 + 6 + 6 + 1
    with pytest.raises(ValueError, match="hidden lists 3 layers"):
        tsecon.mlp_regression(x, y, hidden=(4, 4, 4), max_epochs=3, n_seeds=1)
    with pytest.raises(ValueError, match="hidden lists no hidden layer"):
        tsecon.mlp_regression(x, y, hidden=(), max_epochs=3, n_seeds=1)
    with pytest.raises(ValueError, match=r"hidden\[1\] is 0 units"):
        tsecon.mlp_regression(x, y, hidden=(4, 0), max_epochs=3, n_seeds=1)
    with pytest.raises(ValueError, match="positive integer"):
        tsecon.mlp_regression(x, y, hidden=[4.5], max_epochs=3, n_seeds=1)
    with pytest.raises(ValueError, match="hidden must be a tuple or list"):
        tsecon.mlp_regression(x, y, hidden="16", max_epochs=3, n_seeds=1)


def test_mlp_teaching_errors_name_arrays_choices_and_counts():
    x, y = sin_ar1(4, 80)
    quick = dict(max_epochs=2, n_seeds=1)
    xn = x.copy()
    xn[5, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x$"):
        tsecon.mlp_regression(xn, y, **quick)
    yn = y.copy()
    yn[5] = np.inf
    with pytest.raises(ValueError, match="in y"):
        tsecon.mlp_regression(x, yn, **quick)
    with pytest.raises(ValueError, match="in x_test"):
        tsecon.mlp_regression(x, y, x_test=xn, **quick)
    with pytest.raises(ValueError, match='unknown activation "swish"; expected one of "tanh", "relu", "logistic"'):
        tsecon.mlp_regression(x, y, activation="swish", **quick)
    with pytest.raises(ValueError, match='unknown solver "sgd"; expected one of "adam", "lbfgs"'):
        tsecon.mlp_regression(x, y, solver="sgd", **quick)
    # Insufficiency counts the temporal validation split: with 20% held
    # out the smallest feasible sample is 5 (1 validation + 4 training rows).
    with pytest.raises(ValueError, match="insufficient data: 4 observations, at least 5 required"):
        tsecon.mlp_regression(x[:4], y[:4], **quick)
    tsecon.mlp_regression(x[:5], y[:5], **quick)
    with pytest.raises(ValueError, match="insufficient data: 3 observations, at least 10 required"):
        tsecon.mlp_regression(x[:3], y[:3], validation_fraction=0.1, **quick)
    with pytest.raises(ValueError, match="validation_fraction"):
        tsecon.mlp_regression(x, y, validation_fraction=0.7, **quick)
    with pytest.raises(ValueError, match="batch_size=1000 exceeds the 64 training rows"):
        tsecon.mlp_regression(x, y, batch_size=1000, **quick)
    with pytest.raises(ValueError, match="alpha"):
        tsecon.mlp_regression(x, y, alpha=-1.0, **quick)
    with pytest.raises(ValueError, match="learning_rate"):
        tsecon.mlp_regression(x, y, learning_rate=0.0, **quick)
    with pytest.raises(ValueError, match="n_seeds"):
        tsecon.mlp_regression(x, y, n_seeds=0, max_epochs=2)
    with pytest.raises(ValueError, match="max_epochs"):
        tsecon.mlp_regression(x, y, max_epochs=0, n_seeds=1)
    with pytest.raises(TypeError):
        tsecon.mlp_regression(y, y, **quick)  # 1-D x: rank error, not a panic


def test_mlp_lbfgs_sentinels_refuse_epoch_arguments_and_defaults_are_identical():
    x, y = sin_ar1(5, 100)
    for kw, text in [
        ({"learning_rate": 0.01}, 'learning_rate=0.01 has no effect under solver="lbfgs"'),
        ({"batch_size": 8}, 'batch_size=8 has no effect under solver="lbfgs"'),
        ({"patience": 3}, 'patience=3 has no effect under solver="lbfgs"'),
    ]:
        with pytest.raises(ValueError, match=text):
            tsecon.mlp_regression(x, y, solver="lbfgs", max_epochs=5, n_seeds=1, **kw)
    r = tsecon.mlp_regression(x, y, solver="lbfgs", max_epochs=50, n_seeds=2)
    assert r["solver"] == "lbfgs"
    assert all(p.shape == (2,) for p in r["train_loss_path"])
    assert all(p.shape == (2,) for p in r["validation_loss_path"])
    assert all(p[1] <= p[0] for p in r["train_loss_path"])
    # Under adam the None sentinels resolve to the documented defaults, and
    # passing those defaults explicitly is bit-identical.
    a = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2)
    b = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, learning_rate=1e-3, patience=20)
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    for pa, pb in zip(a["train_loss_path"], b["train_loss_path"]):
        np.testing.assert_array_equal(pa, pb)
    # ...and they are live: a different learning rate changes the fit.
    c = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, learning_rate=1e-2)
    assert not np.array_equal(a["fitted"], c["fitted"])


def test_mlp_seed_contract_and_leakage_safe_scaler():
    x, y = sin_ar1(6, 250, sigma=0.3)
    a = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, x_test=x[:10])
    b = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, x_test=x[:10])
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    np.testing.assert_array_equal(a["member_predictions"], b["member_predictions"])
    for wa, wb in zip(a["weights"], b["weights"]):
        for ca, cb in zip(wa["coefs"], wb["coefs"]):
            np.testing.assert_array_equal(ca, cb)
    c = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, seed=1, x_test=x[:10])
    assert not np.array_equal(a["fitted"], c["fitted"])
    # Perturbing the validation rows (the LAST 20%) leaves the scaler
    # bit-identical: it was fit on the training rows only.
    x2, y2 = x.copy(), y.copy()
    x2[200:] += 100.0
    y2[200:] -= 50.0
    p = tsecon.mlp_regression(x2, y2, max_epochs=20, n_seeds=2)
    assert (a["n_train"], a["n_validation"]) == (200, 50)
    np.testing.assert_array_equal(a["x_mean"], p["x_mean"])
    np.testing.assert_array_equal(a["x_scale"], p["x_scale"])
    assert a["y_mean"] == p["y_mean"] and a["y_scale"] == p["y_scale"]
    # ...whereas perturbing a training row moves it.
    x3 = x.copy()
    x3[0] += 100.0
    m = tsecon.mlp_regression(x3, y, max_epochs=20, n_seeds=2)
    assert not np.array_equal(a["x_mean"], m["x_mean"])
    raw = tsecon.mlp_regression(x, y, max_epochs=20, n_seeds=2, standardize=False)
    np.testing.assert_array_equal(raw["x_mean"], [0.0])
    np.testing.assert_array_equal(raw["x_scale"], [1.0])
    assert (raw["y_mean"], raw["y_scale"]) == (0.0, 1.0)


def test_mlp_pandas_coercion_matches_numpy():
    pd = pytest.importorskip("pandas")
    x, y = sin_ar1(7, 150)
    a = tsecon.mlp_regression(x, y, max_epochs=10, n_seeds=2, x_test=x[:5])
    b = tsecon.mlp_regression(
        pd.DataFrame(x, columns=["lag1"]), pd.Series(y), max_epochs=10, n_seeds=2,
        x_test=pd.DataFrame(x[:5], columns=["lag1"]),
    )
    np.testing.assert_array_equal(a["fitted"], b["fitted"])
    np.testing.assert_array_equal(a["predicted"], b["predicted"])
    # float32 / non-contiguous inputs are coerced too.
    c = tsecon.mlp_regression(x.astype(np.float32), y[::-1][::-1], max_epochs=10, n_seeds=2,
                              x_test=x[:5])
    np.testing.assert_allclose(c["fitted"], a["fitted"], rtol=1e-5)


# ---------------------------------------------- mlp: MC / property claims


def test_mlp_recovers_nonlinear_ar1_out_of_sample_both_solvers():
    """sigma = 0.3, 600 / 100 rows, Campbell-Thompson R^2. Across six data
    seeds on the release wheel: oracle map 0.78-0.90, mini-batch Adam and
    L-BFGS 0.76-0.90, all-defaults 0.59-0.82, linear AR(1) 0.46-0.76."""
    x, y = sin_ar1(11, 700, sigma=0.3)
    xtr, ytr, xte, yte = x[:600], y[:600], x[600:], y[600:]
    adam = tsecon.mlp_regression(xtr, ytr, batch_size=32, learning_rate=1e-2, max_epochs=200,
                                 x_test=xte)
    lbfgs = tsecon.mlp_regression(xtr, ytr, solver="lbfgs", max_epochs=300, x_test=xte)
    default = tsecon.mlp_regression(xtr, ytr, x_test=xte)
    slope = np.cov(xtr[:, 0], ytr)[0, 1] / np.var(xtr[:, 0], ddof=1)
    lin = ytr.mean() + slope * (xte[:, 0] - xtr[:, 0].mean())
    scores = {k: float(r2(v, yte, ytr.mean())) for k, v in [
        ("adam", adam["predicted"]), ("lbfgs", lbfgs["predicted"]),
        ("defaults", default["predicted"]), ("linear", lin),
        ("oracle", np.sin(2.0 * xte[:, 0]))]}
    print("sin-AR(1) out-of-sample R^2:", scores)
    assert scores["adam"] > 0.6 and scores["lbfgs"] > 0.6
    assert scores["adam"] > scores["linear"] + 0.1
    assert scores["defaults"] > scores["linear"]


def test_mlp_ensemble_beats_mean_member_always_and_median_member_mostly():
    """The documented overfitting DGP (60 rows, sigma 0.7, (32, 16), lbfgs,
    alpha 1e-6, no validation split): the ensemble beats the MEAN member
    MSE in every replication (Jensen) and the MEDIAN member in most
    (measured 10/10 here on the release wheel; asserted >= 7/10)."""
    wins_median = wins_mean = 0
    for seed in range(1, 11):
        x, y = sin_ar1(100 + seed, 260, sigma=0.7)
        r = tsecon.mlp_regression(x[:60], y[:60], hidden=(32, 16), solver="lbfgs", alpha=1e-6,
                                  validation_fraction=0.0, max_epochs=500, n_seeds=9,
                                  x_test=x[60:])
        truth = y[60:]
        ens = np.mean((r["predicted"] - truth) ** 2)
        members = np.mean((r["member_predictions"] - truth) ** 2, axis=1)
        wins_mean += ens < members.mean()
        wins_median += ens < np.median(members)
    print(f"ensemble beats mean member {wins_mean}/10, median member {wins_median}/10")
    assert wins_mean == 10
    assert wins_median >= 7


def test_mlp_early_stopping_fires_on_easy_problem_and_not_at_one_epoch():
    rng = np.random.default_rng(12)
    x = rng.standard_normal((300, 2))
    y = 1.0 + 2.0 * x[:, 0] - x[:, 1] + 0.01 * rng.standard_normal(300)
    r = tsecon.mlp_regression(x, y, batch_size=32, learning_rate=1e-2, patience=10, max_epochs=500)
    assert all(r["converged"])
    assert all(p.shape[0] < 500 for p in r["train_loss_path"])
    assert all(1 <= e <= p.shape[0] for e, p in zip(r["best_epoch"], r["train_loss_path"]))
    assert r2(r["fitted"], y) > 0.99
    one = tsecon.mlp_regression(x, y, batch_size=32, learning_rate=1e-2, patience=10, max_epochs=1)
    assert not any(one["converged"])
    assert all(p.shape == (1,) for p in one["train_loss_path"])
    assert one["best_epoch"] == [1] * 5
    # No validation split: no early stopping, empty validation paths.
    none = tsecon.mlp_regression(x, y, validation_fraction=0.0, max_epochs=5, n_seeds=2)
    assert none["n_validation"] == 0 and not any(none["converged"])
    assert all(p.shape == (0,) for p in none["validation_loss_path"])
    assert none["best_epoch"] == [5, 5]


def test_mlp_default_call_wall_clock_budget():
    rng = np.random.default_rng(21)
    x = rng.standard_normal((500, 5))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] * x[:, 2] + 0.3 * rng.standard_normal(500)
    t0 = time.perf_counter()
    r = tsecon.mlp_regression(x, y)
    secs = time.perf_counter() - t0
    print(f"mlp_regression default call on n=500, p=5: {secs:.3f} s")
    assert r["n_parameters"] == 5 * 16 + 16 + 16 + 1
    # Budget ~3 s (measured ~1.6-2.3 s on the release wheel); 2x headroom
    # so a loaded CI runner does not flake.
    assert secs < 6.0, f"default call took {secs:.2f} s"


# ------------------------------------------------------------------- esn


def test_esn_keys_shapes_and_radius():
    x, y = narma10(1, 400)
    r = tsecon.echo_state_network(x[:300], y[:300], reservoir_size=50, x_test=x[300:])
    assert set(r) == ESN_KEYS
    assert r["fitted"].shape == (250,)
    assert r["predicted"].shape == (100,)
    assert r["readout"].shape == (1 + 1 + 50,)
    assert (r["reservoir_size"], r["n_washout"], r["n_train"]) == (50, 50, 250)
    assert abs(r["spectral_radius_achieved"] - 0.9) < 1e-6
    for target in (0.5, 1.2):
        s = tsecon.echo_state_network(x[:300], y[:300], reservoir_size=50, spectral_radius=target)
        assert abs(s["spectral_radius_achieved"] - target) < 1e-6
        assert s["predicted"] is None
    assert all(np.isfinite(r["fitted"])) and all(np.isfinite(r["predicted"]))


def test_esn_narma10_nrmse_bars():
    """Documented NARMA-10 bars: input_scaling=0.3 with otherwise default
    settings on 1000 training rows, mean over two data seeds < 0.4
    (measured ~0.3); reservoir_size=400 on 2000 rows < 0.3 (~0.19-0.24)."""
    small = []
    for seed in (1, 2):
        x, y = narma10(seed, 1200)
        r = tsecon.echo_state_network(x[:1000], y[:1000], input_scaling=0.3, x_test=x[1000:])
        small.append(nrmse(r["predicted"], y[1000:]))
    x, y = narma10(3, 2200)
    big = tsecon.echo_state_network(x[:2000], y[:2000], reservoir_size=400, input_scaling=0.3,
                                    x_test=x[2000:])
    big_nrmse = nrmse(big["predicted"], y[2000:])
    d = tsecon.echo_state_network(x[:1000], y[:1000], x_test=x[1000:1200])
    print(f"NARMA-10 NRMSE: input_scaling 0.3 {small}, N=400/2000 rows {big_nrmse:.4f}, "
          f"all defaults {nrmse(d['predicted'], y[1000:1200]):.4f}")
    assert np.mean(small) < 0.4
    assert big_nrmse < 0.3


def test_esn_seed_contract_and_pandas():
    x, y = narma10(4, 300)
    a = tsecon.echo_state_network(x, y, reservoir_size=40, x_test=x[:20])
    b = tsecon.echo_state_network(x, y, reservoir_size=40, x_test=x[:20])
    np.testing.assert_array_equal(a["readout"], b["readout"])
    np.testing.assert_array_equal(a["predicted"], b["predicted"])
    c = tsecon.echo_state_network(x, y, reservoir_size=40, seed=3, x_test=x[:20])
    assert not np.array_equal(a["readout"], c["readout"])
    pd = pytest.importorskip("pandas")
    p = tsecon.echo_state_network(pd.DataFrame(x), pd.Series(y), reservoir_size=40,
                                  x_test=pd.DataFrame(x[:20]))
    np.testing.assert_array_equal(a["fitted"], p["fitted"])
    np.testing.assert_array_equal(a["predicted"], p["predicted"])


def test_esn_teaching_errors():
    x, y = narma10(5, 40)
    small = dict(reservoir_size=10, washout=5)
    tsecon.echo_state_network(x, y, **small)
    with pytest.raises(ValueError, match="washout=40 discards every row.*n=40 rows.*washout < n - 2"):
        tsecon.echo_state_network(x, y, reservoir_size=10, washout=40)
    with pytest.raises(ValueError, match="insufficient data: 40 observations, at least 41 required"):
        tsecon.echo_state_network(x, y, reservoir_size=10, washout=39)
    xn = x.copy()
    xn[2, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x$"):
        tsecon.echo_state_network(xn, y, **small)
    with pytest.raises(ValueError, match="in x_test"):
        tsecon.echo_state_network(x, y, x_test=xn, **small)
    yn = y.copy()
    yn[0] = np.inf
    with pytest.raises(ValueError, match="in y"):
        tsecon.echo_state_network(x, yn, **small)
    for kw, key in [
        ({"leak_rate": 0.0}, "leak_rate"), ({"leak_rate": 1.5}, "leak_rate"),
        ({"sparsity": 0.0}, "sparsity"), ({"reservoir_size": 0}, "reservoir_size"),
        ({"spectral_radius": 0.0}, "spectral_radius"), ({"input_scaling": 0.0}, "input_scaling"),
        ({"ridge_alpha": -1.0}, "ridge_alpha"),
    ]:
        args = {**small, **kw}
        with pytest.raises(ValueError, match=key):
            tsecon.echo_state_network(x, y, **args)
    with pytest.raises(ValueError, match="column count must match"):
        tsecon.echo_state_network(x, y, x_test=np.hstack([x, x]), **small)
    with pytest.raises(TypeError):
        tsecon.echo_state_network(y, y, **small)  # 1-D x: rank error, not a panic


def test_esn_default_call_wall_clock_budget():
    rng = np.random.default_rng(22)
    x = rng.standard_normal((500, 5))
    y = x[:, 0] + 0.1 * rng.standard_normal(500)
    t0 = time.perf_counter()
    r = tsecon.echo_state_network(x, y)
    secs = time.perf_counter() - t0
    print(f"echo_state_network default call on n=500, p=5: {secs:.3f} s")
    assert r["readout"].shape == (206,)
    # Budget ~1 s (measured ~0.1-0.2 s on the release wheel).
    assert secs < 2.0, f"default call took {secs:.2f} s"


# ---------------------------------------------------------- fixture meta


def test_fixture_states_its_grades():
    meta = NEURAL["_meta"]
    assert meta["sklearn"] == "1.9.0"
    assert "independent package" in meta["mlp"]["grade"]
    rp = meta["esn"]["reservoirpy"]
    assert isinstance(rp["installed"], bool)
    if rp["installed"]:
        assert rp["max_abs_state_diff"] < 1e-12
        assert "third-party" in meta["esn"]["grade"]
    else:
        assert "transcription" in meta["esn"]["grade"]


def test_docstrings_and_stub_document_every_key():
    for fn, keys in [(tsecon.mlp_regression, MLP_KEYS), (tsecon.echo_state_network, ESN_KEYS)]:
        doc = fn.__doc__
        for k in keys:
            assert k in doc, f"{fn.__name__}: key {k} undocumented"
    stub = (Path(__file__).parents[1] / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    for name in ("mlp_regression", "echo_state_network"):
        assert f"def {name}(" in stub
    for k in MLP_KEYS | ESN_KEYS:
        assert k in stub
