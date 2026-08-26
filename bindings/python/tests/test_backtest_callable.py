"""Python-callable forecasters through the leakage-safe backtest engine
(field-report item 9).

``backtest(forecaster=...)`` and the split/aci conformal ``base=`` now accept
any Python callable ``f(train, horizon) -> array-like of horizon point
forecasts`` alongside the built-in string names. These tests pin the contract:

1.  Leakage: a spy callable records every training window it is handed; each
    window must be exactly the engine's documented slice and must end at the
    forecast origin — strictly before every target it is scored against. A
    perturbation test drives the same point without trusting the spy:
    changing an observation cannot move any forecast whose origin precedes it.
2.  Plumbing: a Python reimplementation of ``naive`` is bit-identical to
    ``forecaster="naive"`` through both window schemes and an infrequent
    refit cadence.
3.  A real model: statsmodels ``AutoReg(lags=p, trend="c")`` — the same
    OLS-AR(p)-with-constant, iterated-multi-step spec as tsecon's own
    ``"ar"`` conformal base — runs through the engine and (a) matches a
    hand-rolled rolling-origin loop exactly, (b) reproduces the Rust
    ``"ar"`` base through the conformal entry points at cross-implementation
    tolerance (stated at the assertion).
4.  Errors: exceptions re-raise with the failing origin and window named and
    the original chained as ``__cause__``; wrong-length / non-finite /
    non-coercible returns get teaching errors naming the callable and origin.
5.  Bit-identity: the pre-existing string-forecaster paths reproduce the
    pre-change snapshot (``fixtures/backtest_string_snapshot.json``,
    captured from the build immediately before this feature landed)
    float-bit for float-bit.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
SNAP = json.loads((FIXTURES / "backtest_string_snapshot.json").read_text())


def unhex(seq):
    return [float.fromhex(s) for s in seq]


def make_y(n=60, seed=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n)) + 20.0


# ---------------------------------------------------------------- leakage


def test_spy_callable_sees_only_training_slices_expanding():
    y = make_y(50)
    seen = []

    def spy(train, h):
        assert train.dtype == np.float64
        assert not train.flags.writeable, "training window must be read-only"
        seen.append((np.array(train), h))
        return np.full(h, train[-1])

    res = tsecon.backtest(y, window="expanding", train=10, horizon=2, forecaster=spy)
    # refit_every=1: one call per origin, asked exactly `horizon` steps.
    assert len(seen) == res["n_origins"]
    for (tr, h), t in zip(seen, res["origins"]):
        assert h == 2
        # The exact documented window: y[0..=t]. It ends AT the origin,
        # strictly before its first test target y[t+1].
        np.testing.assert_array_equal(tr, y[: t + 1])
        assert len(tr) == t + 1


def test_spy_callable_sees_only_training_slices_rolling():
    y = make_y(48)
    width = 12
    seen = []

    def spy(train, h):
        seen.append(np.array(train))
        return np.full(h, train.mean())

    res = tsecon.backtest(y, window="rolling", train=width, horizon=3, forecaster=spy)
    assert len(seen) == res["n_origins"]
    for tr, t in zip(seen, res["origins"]):
        # The exact documented window: the `width` most recent obs ending at t.
        np.testing.assert_array_equal(tr, y[t + 1 - width : t + 1])


def test_spy_callable_refit_block_walk_and_requested_steps():
    y = make_y(40)
    train, horizon, refit_every = 10, 2, 4
    calls = []

    def spy(tr, h):
        calls.append((len(tr), h))
        return np.full(h, tr[-1])

    res = tsecon.backtest(
        y, window="expanding", train=train, horizon=horizon,
        refit_every=refit_every, forecaster=spy,
    )
    # Expected refit-block walk: origins t0, t0+k, ... with the requested
    # steps covering each block: (block_len - 1) + horizon.
    t0 = train - 1
    p = res["n_origins"]
    expected = []
    i = 0
    while i < p:
        block = min(refit_every, p - i)
        expected.append((t0 + i + 1, (block - 1) + horizon))  # (win len, steps)
        i += block
    assert calls == expected


def test_perturbing_an_observation_cannot_move_earlier_origin_forecasts():
    y = make_y(60)

    def full_info(train, h):
        # Deliberately uses EVERY training observation, so any leaked
        # future point would move the forecast.
        return np.full(h, train.mean())

    cfg = dict(window="expanding", train=15, horizon=1, forecaster=full_info)
    base = tsecon.backtest(y, **cfg)
    origins = np.array(base["origins"])

    # y[-1] enters no training window (it is only ever a target): every
    # forecast must be bit-identical after perturbing it.
    y_last = y.copy()
    y_last[-1] += 1000.0
    pert = tsecon.backtest(y_last, **cfg)
    np.testing.assert_array_equal(base["forecasts"][0], pert["forecasts"][0])

    # Perturbing a middle point k moves NO forecast whose origin precedes k
    # (those training windows end before k) and DOES move later ones (the
    # test can tell a leakage guard from a no-op).
    k = 30
    y_mid = y.copy()
    y_mid[k] += 1000.0
    pert_mid = tsecon.backtest(y_mid, **cfg)
    before = origins < k
    np.testing.assert_array_equal(
        np.asarray(base["forecasts"][0])[before],
        np.asarray(pert_mid["forecasts"][0])[before],
    )
    assert not np.allclose(
        np.asarray(base["forecasts"][0])[~before],
        np.asarray(pert_mid["forecasts"][0])[~before],
    )


# ------------------------------------------------------------- plumbing


def test_python_naive_bit_identical_to_string_naive():
    y = make_y(70)

    def py_naive(train, h):
        return np.full(h, train[-1])

    for cfg in (
        dict(window="expanding", train=15, horizon=3, refit_every=1),
        dict(window="rolling", train=20, horizon=2, refit_every=3),
    ):
        rs = tsecon.backtest(y, forecaster="naive", **cfg)
        rc = tsecon.backtest(y, forecaster=py_naive, **cfg)
        assert list(rs["origins"]) == list(rc["origins"])
        for h in range(cfg["horizon"]):
            assert [float(v) for v in rs["forecasts"][h]] == [
                float(v) for v in rc["forecasts"][h]
            ]
            assert [float(v) for v in rs["targets"][h]] == [
                float(v) for v in rc["targets"][h]
            ]
        for row_s, row_c in zip(rs["accuracy"], rc["accuracy"]):
            assert row_s == row_c


def test_default_forecaster_is_naive():
    y = make_y(40)
    r_default = tsecon.backtest(y, train=12, horizon=2)
    r_naive = tsecon.backtest(y, train=12, horizon=2, forecaster="naive")
    for h in range(2):
        assert [float(v) for v in r_default["forecasts"][h]] == [
            float(v) for v in r_naive["forecasts"][h]
        ]


# ------------------------------------------- a real model: statsmodels AR

statsmodels = pytest.importorskip("statsmodels")
from statsmodels.tsa.ar_model import AutoReg  # noqa: E402

AR_LAGS = 2


def sm_autoreg(train, h):
    """statsmodels AutoReg(p, trend='c'): OLS AR(p) with constant, iterated
    multi-step — the same spec as tsecon's `"ar"` base."""
    fit = AutoReg(np.asarray(train), lags=AR_LAGS, trend="c").fit()
    return fit.predict(start=len(train), end=len(train) + h - 1)


def _ar2_series(n=90, seed=11):
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.2 * y[t - 2] + rng.standard_normal()
    return y + 5.0


def test_autoreg_callable_matches_manual_rolling_origin_loop():
    y = _ar2_series(80)
    res = tsecon.backtest(
        y, window="expanding", train=40, horizon=2, forecaster=sm_autoreg
    )
    assert res["n_origins"] == 80 - 2 - 40 + 1
    for i, t in enumerate(res["origins"]):
        fc = np.asarray(sm_autoreg(y[: t + 1], 2))
        for h in (1, 2):
            # Same library, same slice: agreement is numerical identity up
            # to the float64 round trip — 1e-12 relative.
            assert res["forecasts"][h - 1][i] == pytest.approx(fc[h - 1], rel=1e-12)
            assert res["targets"][h - 1][i] == y[t + h]


def test_autoreg_callable_reproduces_rust_ar_base_in_conformal_forecast():
    y = _ar2_series(90)
    r_str = tsecon.conformal_forecast(y, horizon=3, base="ar", lags=AR_LAGS, calib=20)
    r_call = tsecon.conformal_forecast(y, horizon=3, base=sm_autoreg, calib=20)
    # Cross-implementation tolerance: both are OLS AR(2)+constant iterated
    # multi-step, but the least-squares routes differ (tsecon: normal
    # equations + Cholesky; statsmodels: QR/lstsq) — 1e-6 relative on the
    # point path, 1e-6 absolute on the calibrated bounds (which accumulate
    # per-origin refit differences across the whole calibration grid).
    np.testing.assert_allclose(r_call["mean"], r_str["mean"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_call["lower"], r_str["lower"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r_call["upper"], r_str["upper"], rtol=1e-6, atol=1e-6)
    assert r_call["n_calib"] == r_str["n_calib"]
    assert r_call["base"] == "<callable sm_autoreg>"


def test_autoreg_callable_reproduces_rust_ar_base_in_conformal_backtest():
    y = _ar2_series(90)
    kw = dict(horizon=1, method="split", alpha=0.2, calib=15, n_eval=10)
    r_str = tsecon.conformal_backtest(y, base="ar", lags=AR_LAGS, **kw)
    r_call = tsecon.conformal_backtest(y, base=sm_autoreg, **kw)
    assert list(r_str["origins"]) == list(r_call["origins"])
    np.testing.assert_allclose(r_call["mean"][0], r_str["mean"][0], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_call["lower"][0], r_str["lower"][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(r_call["upper"][0], r_str["upper"][0], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        r_call["realized_coverage"], r_str["realized_coverage"]
    )


# ------------------------------------------------------ error propagation


def test_callable_exception_reraises_with_origin_and_window_context():
    y = make_y(40)

    def boom(train, h):
        if len(train) == 25:  # expanding: origin 24
            raise ValueError("boom at 25")
        return np.full(h, train[-1])

    with pytest.raises(RuntimeError) as ei:
        tsecon.backtest(y, window="expanding", train=20, horizon=1, forecaster=boom)
    msg = str(ei.value)
    assert "boom" in msg or "origin 24" in msg
    assert "origin 24" in msg
    assert "y[0..=24]" in msg
    assert "25 observation" in msg
    # The user's own exception survives, type and message intact, as the
    # __cause__ (the gmm_nonlinear re-raise pattern, plus context).
    assert isinstance(ei.value.__cause__, ValueError)
    assert "boom at 25" in str(ei.value.__cause__)


def test_callable_exception_names_rolling_window():
    y = make_y(40)

    def boom(train, h):
        raise KeyError("bad key")

    with pytest.raises(RuntimeError) as ei:
        tsecon.backtest(y, window="rolling", train=10, horizon=1, forecaster=boom)
    # First rolling origin is width-1 = 9, window y[0..=9].
    assert "origin 9" in str(ei.value)
    assert "y[0..=9]" in str(ei.value)
    assert isinstance(ei.value.__cause__, KeyError)


def test_wrong_length_return_teaching_error_names_origin():
    y = make_y(40)

    def too_many(train, h):
        return np.zeros(h + 1)

    with pytest.raises(ValueError, match="asked for exactly"):
        tsecon.backtest(y, window="expanding", train=20, horizon=2, forecaster=too_many)
    try:
        tsecon.backtest(y, window="expanding", train=20, horizon=2, forecaster=too_many)
    except ValueError as e:
        assert "too_many" in str(e)
        assert "origin 19" in str(e)  # first expanding origin: train-1
        assert "returned 3 forecast(s)" in str(e)


def test_non_finite_return_teaching_error_names_step_and_origin():
    y = make_y(40)

    def has_nan(train, h):
        out = np.full(h, train[-1])
        out[-1] = np.nan
        return out

    with pytest.raises(ValueError, match="non-finite"):
        tsecon.backtest(y, window="expanding", train=20, horizon=2, forecaster=has_nan)
    try:
        tsecon.backtest(y, window="expanding", train=20, horizon=2, forecaster=has_nan)
    except ValueError as e:
        assert "has_nan" in str(e)
        assert "step 2 of 2" in str(e)
        assert "origin 19" in str(e)


def test_scalar_return_teaching_error():
    y = make_y(40)

    def scalar(train, h):
        return float(train[-1])

    with pytest.raises(TypeError, match="1-D float sequence"):
        tsecon.backtest(y, train=20, forecaster=scalar)


def test_non_string_non_callable_forecaster_is_type_error():
    y = make_y(40)
    with pytest.raises(TypeError, match="forecaster must be"):
        tsecon.backtest(y, forecaster=3.14)
    with pytest.raises(TypeError, match="base must be"):
        tsecon.conformal_forecast(y, base=3.14)
    # Unknown *names* still raise the pre-existing ValueError.
    with pytest.raises(ValueError, match="unknown forecaster"):
        tsecon.backtest(y, forecaster="nope")


# ------------------------------------------------------ conformal callable


def test_conformal_callable_windows_are_prefixes_plus_one_forward_call():
    y = make_y(60)
    seen = []

    def spy(train, h):
        assert not train.flags.writeable
        seen.append(np.array(train))
        return np.full(h, train[-1])

    r = tsecon.conformal_forecast(y, horizon=2, base=spy, calib=15)
    n = len(y)
    # Every calibration window is a strict prefix of y ending inside the
    # sample; exactly one call — the forward forecast — sees the full series.
    assert sum(len(tr) == n for tr in seen) == 1
    assert len(seen[-1]) == n
    for tr in seen:
        np.testing.assert_array_equal(tr, y[: len(tr)])
    assert r["base"] == f"<callable {spy.__qualname__}>"


def test_conformal_callable_exception_context_and_cause():
    y = make_y(60)

    def boom(train, h):
        raise ArithmeticError("diverged")

    with pytest.raises(RuntimeError) as ei:
        tsecon.conformal_forecast(y, base=boom)
    assert "conformal base origin" in str(ei.value)
    assert isinstance(ei.value.__cause__, ArithmeticError)


def test_enbpi_rejects_callable_base():
    y = make_y(60)
    with pytest.raises(ValueError, match="enbpi"):
        tsecon.conformal_forecast(
            y, method="enbpi", base=lambda tr, h: np.zeros(h), lags=2
        )
    with pytest.raises(ValueError, match="enbpi"):
        tsecon.conformal_backtest(
            y, method="enbpi", base=lambda tr, h: np.zeros(h), lags=2
        )


# ------------------------------------------------- bit-identity snapshot


def _snapshot_series():
    rng = np.random.default_rng(20260826)
    return np.cumsum(rng.standard_normal(90)) + 50.0


def test_backtest_string_paths_bit_identical_to_pre_callable_snapshot():
    y = _snapshot_series()
    for name, blk in SNAP["backtest"].items():
        r = tsecon.backtest(y, forecaster=name, **blk["config"])
        assert [int(t) for t in r["origins"]] == blk["origins"], name
        for h, fx in enumerate(blk["forecasts"]):
            assert [float(v) for v in r["forecasts"][h]] == unhex(fx), (name, h)
        for h, tx in enumerate(blk["targets"]):
            assert [float(v) for v in r["targets"][h]] == unhex(tx), (name, h)
        assert [row["rmse"] for row in r["accuracy"]] == unhex(blk["accuracy_rmse"]), name
        assert [row["mase"] for row in r["accuracy"]] == unhex(blk["accuracy_mase"]), name


def test_conformal_string_paths_bit_identical_to_pre_callable_snapshot():
    y = _snapshot_series()

    cf = tsecon.conformal_forecast(
        y, horizon=3, method="split", base="theta", alpha=0.1, mode="asymmetric"
    )
    b = SNAP["conformal_forecast_split"]
    assert [float(v) for v in cf["mean"]] == unhex(b["mean"])
    assert [float(v) for v in cf["lower"]] == unhex(b["lower"])
    assert [float(v) for v in cf["upper"]] == unhex(b["upper"])
    assert int(cf["n_calib"]) == b["n_calib"]
    assert cf["base"] == "theta"

    ca = tsecon.conformal_forecast(
        y, horizon=2, method="aci", base="ar", lags=2, alpha=0.1, gamma=0.01
    )
    b = SNAP["conformal_forecast_aci"]
    assert [float(v) for v in ca["mean"]] == unhex(b["mean"])
    assert [float(v) for v in ca["lower"]] == unhex(b["lower"])
    assert [float(v) for v in ca["upper"]] == unhex(b["upper"])
    assert [float(v) for v in ca["alpha_final"]] == unhex(b["alpha_final"])

    ce = tsecon.conformal_forecast(
        y, horizon=2, method="enbpi", base="ar", lags=2, n_boot=10, seed=7
    )
    b = SNAP["conformal_forecast_enbpi"]
    assert [float(v) for v in ce["mean"]] == unhex(b["mean"])
    assert [float(v) for v in ce["lower"]] == unhex(b["lower"])
    assert [float(v) for v in ce["upper"]] == unhex(b["upper"])
    assert [float(ce["beta"])] == unhex(b["beta"])

    cb = tsecon.conformal_backtest(
        y, horizon=2, method="split", base="drift", alpha=0.2, calib=15, n_eval=12
    )
    b = SNAP["conformal_backtest_split"]
    assert [float(v) for v in cb["realized_coverage"]] == unhex(b["realized_coverage"])
    assert [float(v) for v in cb["mean"][0]] == unhex(b["mean_h1"])
    assert [float(v) for v in cb["lower"][0]] == unhex(b["lower_h1"])
    assert [float(v) for v in cb["upper"][0]] == unhex(b["upper_h1"])
    assert cb["base"] == "drift"
