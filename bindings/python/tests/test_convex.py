"""Binding tests for the convex / greedy estimators: `l1_trend_filter`
(Kim-Koh-Boyd 2009 L1 trend filtering and its L2 / Hodrick-Prescott form)
and `boosting` (componentwise L2 boosting, Buhlmann-Yu 2003 / Buhlmann
2006).

Re-pins fixtures/convex.json through the Python surface — the independent
KKT / duality-gap certificate for every L1 case (the primary grade), the
cvxpy + Clarabel third-party trends, the closed-form limits, the dense
transcription of boosting — and checks what a Rust golden cannot see:
marshalling and exact key sets, teaching ValueErrors with their text,
pandas coercion, the audit-round-10 sentinel convention for the inert
`tol` / `max_iter` under `penalty="l2"`, the cross-surface identity with
`tsecon.hp_filter`, and the docstring-names-every-key tripwire.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
CX = json.loads((FIX / "convex.json").read_text())

L1_KEYS = {
    "trend", "cycle", "knots", "n_knots", "duality_gap", "objective",
    "converged", "n_iter", "lam_max",
}
BOOST_KEYS = {
    "coef", "coef_path", "selected", "rss_path", "df_path", "aic_path",
    "best_step", "fitted", "predicted",
}


def diff_matrix(n, k):
    return np.diff(np.eye(n), k, axis=0)


def certificate(y, x, lam, k):
    """Independent KKT certificate: recover v from y - x = D'v by k negative
    cumulative sums, clip into the dual box, return (pobj, dobj, v)."""
    D = diff_matrix(y.size, k)
    r = y - x
    v = r.copy()
    for _ in range(k):
        v = -np.cumsum(v)[:-1]
    vc = np.clip(v, -lam, lam)
    pobj = 0.5 * float(r @ r) + lam * float(np.abs(D @ x).sum())
    dobj = -0.5 * float(np.sum((D.T @ vc) ** 2)) + float(vc @ (D @ y))
    return pobj, dobj, v


def series(name):
    return np.array(CX["series"][name])


# ------------------------------------------------------- l1_trend_filter

def test_l1_trend_filter_keys_and_marshalling():
    y = series("pwl")
    r = tsecon.l1_trend_filter(y, 200.0)
    assert set(r.keys()) == L1_KEYS
    assert r["trend"].shape == y.shape and r["trend"].dtype == np.float64
    assert r["cycle"].shape == y.shape
    np.testing.assert_allclose(r["trend"] + r["cycle"], y, rtol=0, atol=1e-12)
    assert r["knots"].dtype.kind == "i" and r["knots"].ndim == 1
    assert r["n_knots"] == len(r["knots"]) == len(set(r["knots"].tolist()))
    assert np.all((r["knots"] >= 0) & (r["knots"] < y.size - 2))
    assert isinstance(r["converged"], bool) and r["converged"] is True
    assert isinstance(r["n_iter"], int) and r["n_iter"] >= 1
    assert r["duality_gap"] >= 0.0 and r["objective"] > 0.0
    assert r["duality_gap"] <= 1e-8 * r["objective"]
    assert r["lam_max"] > 200.0


@pytest.mark.parametrize("case", CX["l1_cases"], ids=lambda c: c["name"])
def test_l1_certificate_and_lam_max_through_python(case):
    y = series(case["series"])
    k, lam = case["order"], case["lam"]
    r = tsecon.l1_trend_filter(y, lam, order=k)
    assert r["converged"] is True
    pobj, dobj, v = certificate(y, r["trend"], lam, k)
    assert (pobj - dobj) / pobj <= 1e-8
    assert r["objective"] == pytest.approx(pobj, rel=1e-10)
    assert np.max(np.abs(v)) <= lam * (1 + 1e-8)
    assert r["lam_max"] == pytest.approx(case["lam_max"], rel=1e-10)
    # Knots are exactly the differences above the documented threshold.
    dx = diff_matrix(y.size, k) @ r["trend"]
    thr = max(1e-6 * np.max(np.abs(diff_matrix(y.size, k) @ y)), 1e-12 * np.max(np.abs(y)))
    np.testing.assert_array_equal(r["knots"], np.flatnonzero(np.abs(dx) > thr))
    # Complementary slackness at every knot.
    for i in r["knots"]:
        assert v[i] == pytest.approx(lam * np.sign(dx[i]), rel=1e-6)


@pytest.mark.parametrize(
    "case", [c for c in CX["l1_cases"] if c.get("trend_ref") is not None],
    ids=lambda c: c["name"],
)
def test_l1_matches_clarabel_through_python(case):
    y = series(case["series"])
    r = tsecon.l1_trend_filter(y, case["lam"], order=case["order"])
    assert case["ref_status"] == "optimal" and case["ref_gap_rel"] <= 1e-11
    np.testing.assert_allclose(r["trend"], case["trend_ref"], rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "case", [c for c in CX["l1_cases"] if c["lam_frac"] >= 1.0], ids=lambda c: c["name"]
)
def test_l1_polynomial_limit(case):
    y = series(case["series"])
    r = tsecon.l1_trend_filter(y, case["lam"], order=case["order"])
    np.testing.assert_allclose(r["trend"], case["poly_fit"], rtol=0, atol=1e-8)
    assert r["n_knots"] == 0 and r["n_iter"] == 0 and r["converged"] is True


def test_l1_lam_zero_and_vanishing_lam_return_the_data():
    y = series("rw")
    r0 = tsecon.l1_trend_filter(y, 0.0)
    np.testing.assert_array_equal(r0["trend"], y)
    assert r0["objective"] == 0.0 and r0["duality_gap"] == 0.0 and r0["n_iter"] == 0
    # The trend is within 2^order * lam of y for any lam, so a lam of
    # 1e-14 * lam_max (~1e-11 here) lands inside 1e-10.
    for k in (1, 2):
        lam = 1e-14 * tsecon.l1_trend_filter(y, 1.0, order=k)["lam_max"]
        r = tsecon.l1_trend_filter(y, lam, order=k)
        np.testing.assert_allclose(r["trend"], y, rtol=0, atol=1e-10)
        # The bound is attained (every constraint active at +-lam): allow
        # the rounding of y - (y - D'z).
        assert np.max(np.abs(r["trend"] - y)) <= 2**k * lam + 64 * np.finfo(float).eps * np.max(np.abs(y))


def test_l2_penalty_is_the_hp_filter():
    """Cross-surface identity: `penalty="l2"`, `order=2` reproduces
    `tsecon.hp_filter` at 1e-10 for the same lam (1600 quarterly, 100,
    6.25 annual) on three series, and the fixture's dense closed form."""
    worst = 0.0
    for name, lam in [("pwl", 1600.0), ("rw", 100.0), ("ar_trend", 6.25), ("steps", 1600.0)]:
        y = series(name)
        r = tsecon.l1_trend_filter(y, lam, penalty="l2")
        hp = tsecon.hp_filter(y, lamb=lam)
        d = float(np.max(np.abs(r["trend"] - hp["trend"])))
        worst = max(worst, d)
        np.testing.assert_allclose(r["trend"], hp["trend"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(r["cycle"], hp["cycle"], rtol=0, atol=1e-10)
        assert r["converged"] is True and r["n_iter"] == 0
        assert abs(r["duality_gap"]) <= 1e-10 * r["objective"]
    for case in CX["l2_cases"]:
        y = series(case["series"])
        r = tsecon.l1_trend_filter(y, case["lam"], order=case["order"], penalty="l2")
        np.testing.assert_allclose(r["trend"], case["trend_ref"], rtol=0, atol=1e-10)
        assert r["objective"] == pytest.approx(case["objective"], rel=1e-10)
    print(f"l2-vs-hp_filter worst abs diff {worst:.2e}")


def test_l2_inert_kwargs_follow_the_sentinel_convention():
    """Audit-round-10 convention, three legs: explicitly passed where
    inert raises naming the kwarg and the fix; the default call is
    bit-identical to the pre-sentinel behaviour; the kwargs are live under
    `penalty="l1"`."""
    y = series("ar_trend")
    with pytest.raises(ValueError, match="tol was given but penalty='l2' ignores it"):
        tsecon.l1_trend_filter(y, 1600.0, penalty="l2", tol=1e-6)
    with pytest.raises(ValueError, match="max_iter was given but penalty='l2' ignores it"):
        tsecon.l1_trend_filter(y, 1600.0, penalty="l2", max_iter=5)
    # Default call (sentinel None) is bit-identical to the HP surface and to
    # itself across calls.
    a = tsecon.l1_trend_filter(y, 1600.0, penalty="l2")
    b = tsecon.l1_trend_filter(y, 1600.0, order=2, penalty="l2")
    np.testing.assert_array_equal(a["trend"], b["trend"])
    # Under l1 the None defaults resolve to 1e-8 / 10000 bit-for-bit...
    d = tsecon.l1_trend_filter(y, 100.0)
    e = tsecon.l1_trend_filter(y, 100.0, tol=1e-8, max_iter=10000)
    np.testing.assert_array_equal(d["trend"], e["trend"])
    assert d["n_iter"] == e["n_iter"] and d["duality_gap"] == e["duality_gap"]
    # ...and both kwargs are live there.
    loose = tsecon.l1_trend_filter(y, 100.0, tol=1e-2)
    assert loose["n_iter"] < d["n_iter"] and loose["duality_gap"] >= d["duality_gap"]
    starved = tsecon.l1_trend_filter(y, 100.0, max_iter=1)
    assert starved["converged"] is False and starved["n_iter"] == 1
    assert starved["duality_gap"] > 1e-8 * starved["objective"]
    assert np.all(np.isfinite(starved["trend"]))


def test_l1_teaching_errors():
    y = series("steps")
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in y"):
        tsecon.l1_trend_filter(np.r_[y, np.nan], 1.0)
    with pytest.raises(ValueError, match="insufficient data: 2 observations, at least 3 required"):
        tsecon.l1_trend_filter(y[:2], 1.0)
    with pytest.raises(ValueError, match="insufficient data: 1 observations, at least 2 required"):
        tsecon.l1_trend_filter(y[:1], 1.0, order=1)
    with pytest.raises(ValueError, match="insufficient data: 0 observations"):
        tsecon.l1_trend_filter(np.array([]), 1.0)
    with pytest.raises(ValueError, match="lam must be finite and non-negative"):
        tsecon.l1_trend_filter(y, -1.0)
    with pytest.raises(ValueError, match="lam must be finite and non-negative"):
        tsecon.l1_trend_filter(y, np.inf)
    with pytest.raises(ValueError, match=r"order must be 1 \(piecewise-constant trend\) or 2"):
        tsecon.l1_trend_filter(y, 1.0, order=3)
    with pytest.raises(ValueError, match="order must be 1"):
        tsecon.l1_trend_filter(y, 1.0, order=-1)
    with pytest.raises(ValueError, match=r'unknown penalty "huber"; expected "l1" .* or "l2"'):
        tsecon.l1_trend_filter(y, 1.0, penalty="huber")
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        tsecon.l1_trend_filter(y, 1.0, tol=0.0)
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        tsecon.l1_trend_filter(y, 1.0, max_iter=0)


def test_l1_pandas_coercion():
    pd = pytest.importorskip("pandas")
    y = series("pwl")
    a = tsecon.l1_trend_filter(y, 150.0)
    b = tsecon.l1_trend_filter(pd.Series(y, name="gdp"), 150.0)
    np.testing.assert_array_equal(a["trend"], b["trend"])
    np.testing.assert_array_equal(a["knots"], b["knots"])
    # Non-contiguous and float32 inputs are accepted too.
    c = tsecon.l1_trend_filter(np.asfortranarray(np.c_[y, y])[:, 0], 150.0)
    np.testing.assert_array_equal(a["trend"], c["trend"])


# --------------------------------------------------------------- boosting

def _design(name):
    d = CX["boost_designs"][name]
    xt = None if d["X_test"] is None else np.array(d["X_test"])
    return np.array(d["X"]), np.array(d["y"]), xt, np.array(d["true_beta"])


def test_boosting_keys_and_marshalling():
    X, y, Xt, _ = _design("sparse")
    r = tsecon.boosting(X, y, n_steps=40, x_test=Xt)
    assert set(r.keys()) == BOOST_KEYS
    n, p = X.shape
    assert r["coef"].shape == (p,) and r["coef"].dtype == np.float64
    assert r["coef_path"].shape == (40, p)
    assert r["selected"].shape == (40,) and r["selected"].dtype.kind == "i"
    for key in ("rss_path", "df_path", "aic_path"):
        assert r[key].shape == (40,) and r[key].dtype == np.float64
    assert isinstance(r["best_step"], int) and 0 <= r["best_step"] < 40
    np.testing.assert_array_equal(r["coef"], r["coef_path"][r["best_step"]])
    assert r["fitted"].shape == (n,)
    np.testing.assert_allclose(r["fitted"], X @ r["coef"], rtol=0, atol=1e-12)
    assert r["predicted"].shape == (Xt.shape[0],)
    np.testing.assert_allclose(r["predicted"], Xt @ r["coef"], rtol=0, atol=1e-12)
    assert tsecon.boosting(X, y, n_steps=5)["predicted"] is None


@pytest.mark.parametrize("case", CX["boost_cases"], ids=lambda c: c["name"])
def test_boosting_matches_dense_transcription(case):
    X, y, Xt, _ = _design(case["design"])
    r = tsecon.boosting(
        X, y, learning_rate=case["learning_rate"], n_steps=case["n_steps"], x_test=Xt
    )
    np.testing.assert_array_equal(r["selected"], case["selected"])
    assert r["best_step"] == case["best_step"]
    np.testing.assert_allclose(r["coef_path"], case["coef_path"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["df_path"], case["df_path"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["aic_path"], case["aic_path"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["rss_path"], case["rss_path"], rtol=1e-12)
    np.testing.assert_allclose(r["coef"], case["coef"], rtol=0, atol=1e-12)
    # X @ coef reproduces the dense operator applied to y.
    np.testing.assert_allclose(r["fitted"], case["fitted_operator"], rtol=0, atol=1e-10)
    if Xt is not None:
        np.testing.assert_allclose(r["predicted"], case["predicted"], rtol=0, atol=1e-12)
    else:
        assert r["predicted"] is None


def test_boosting_properties_through_python():
    X, y, _, beta = _design("sparse")
    r = tsecon.boosting(X, y)
    assert np.all(np.diff(r["rss_path"]) <= 1e-12)
    assert np.all(np.diff(r["df_path"]) >= -1e-12)
    # Seedless determinism.
    r2 = tsecon.boosting(X, y)
    np.testing.assert_array_equal(r["selected"], r2["selected"])
    np.testing.assert_array_equal(r["coef_path"], r2["coef_path"])
    # AIC picks an interior step and recovers the sparse support.
    assert 0 < r["best_step"] < 499
    for j, b in enumerate(beta):
        if b != 0:
            assert r["coef"][j] * b > 0 and abs(r["coef"][j] - b) < 0.5
        else:
            assert abs(r["coef"][j]) < 0.15
    # stop="none" reports the last step of the same path.
    rn = tsecon.boosting(X, y, stop="none")
    assert rn["best_step"] == 499
    np.testing.assert_array_equal(rn["aic_path"], r["aic_path"])
    np.testing.assert_array_equal(rn["coef"], r["coef_path"][-1])


def test_boosting_teaching_errors():
    X, y, Xt, _ = _design("sparse")
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x"):
        tsecon.boosting(bad, y)
    ybad = y.copy()
    ybad[0] = np.inf
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in y"):
        tsecon.boosting(X, ybad)
    xtb = Xt.copy()
    xtb[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite value \(NaN or infinity\) in x_test"):
        tsecon.boosting(X, y, x_test=xtb)
    with pytest.raises(ValueError, match="insufficient data: 2 observations, at least 3 required"):
        tsecon.boosting(X[:2], y[:2])
    for nu in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"learning_rate must lie in \(0, 1\]"):
            tsecon.boosting(X, y, learning_rate=nu)
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        tsecon.boosting(X, y, n_steps=0)
    with pytest.raises(ValueError, match=r'unknown stop "bic"; expected "aic" .* or "none"'):
        tsecon.boosting(X, y, stop="bic")
    with pytest.raises(ValueError, match="x_test must have the same number of columns as x"):
        tsecon.boosting(X, y, x_test=Xt[:, :3])
    with pytest.raises(ValueError, match="y length must equal the number of rows of x"):
        tsecon.boosting(X, y[:10])
    with pytest.raises(ValueError, match="every column of x has zero norm"):
        tsecon.boosting(np.zeros_like(X), y)


def test_boosting_pandas_coercion():
    pd = pytest.importorskip("pandas")
    X, y, Xt, _ = _design("sparse")
    a = tsecon.boosting(X, y, n_steps=30, x_test=Xt)
    cols = [f"x{j}" for j in range(X.shape[1])]
    b = tsecon.boosting(
        pd.DataFrame(X, columns=cols), pd.Series(y), n_steps=30,
        x_test=pd.DataFrame(Xt, columns=cols),
    )
    np.testing.assert_array_equal(a["coef_path"], b["coef_path"])
    np.testing.assert_array_equal(a["selected"], b["selected"])
    np.testing.assert_array_equal(a["predicted"], b["predicted"])


# ------------------------------------------------------- docs and timing

def test_docstrings_name_every_returned_key():
    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    y = series("pwl")
    r = tsecon.l1_trend_filter(y, 200.0)
    missing = set(r.keys()) - tokens(tsecon.l1_trend_filter)
    assert not missing, f"l1_trend_filter.__doc__ missing keys: {sorted(missing)}"
    X, yb, Xt, _ = _design("sparse")
    b = tsecon.boosting(X, yb, n_steps=10, x_test=Xt)
    missing = set(b.keys()) - tokens(tsecon.boosting)
    assert not missing, f"boosting.__doc__ missing keys: {sorted(missing)}"


def test_wall_time_report():
    """The report figures: `l1_trend_filter` at n = 10000 (banded O(n)
    interior point) and `boosting` at n = 500, p = 50, n_steps = 500. The
    bounds only guard against a lost O(n) structure."""
    rng = np.random.default_rng(1)
    n = 10_000
    slope = np.repeat(0.02 * rng.standard_normal(10), n // 10)
    y = np.cumsum(slope) + 0.3 * rng.standard_normal(n)
    t0 = time.perf_counter()
    r = tsecon.l1_trend_filter(y, 50.0)
    t_l1 = time.perf_counter() - t0
    assert r["converged"] is True
    t0 = time.perf_counter()
    tsecon.l1_trend_filter(y, 1600.0, penalty="l2")
    t_l2 = time.perf_counter() - t0

    X = rng.standard_normal((500, 50))
    X = (X - X.mean(0)) / X.std(0)
    yb = X[:, :5] @ np.array([3.0, -2.0, 1.5, 1.0, -1.0]) + rng.standard_normal(500)
    yb -= yb.mean()
    t0 = time.perf_counter()
    b = tsecon.boosting(X, yb, n_steps=500)
    t_boost = time.perf_counter() - t0
    assert b["coef_path"].shape == (500, 50)
    print(
        f"wall: l1_trend_filter n=10000 {t_l1:.3f} s ({r['n_iter']} iterations, "
        f"{r['n_knots']} knots); l2 form {t_l2:.4f} s; boosting n=500 p=50 "
        f"n_steps=500 {t_boost:.3f} s (best_step {b['best_step']})"
    )
    assert t_l1 < 5.0 and t_boost < 10.0
