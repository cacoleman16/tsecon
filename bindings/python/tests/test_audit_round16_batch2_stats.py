"""Regression pins for adversarial audit round 16, statistics findings on the
batch-2 surface (`dfm_smooth`, `adl_midas`, `umidas`, `var_conditional_forecast`)
and the nowcast-crate refusal wording.

* DF16-1 — `dfm_smooth`: more than r zero idiosyncratic variances observed
  together gave a roundoff-driven log-likelihood (+20 points); the value is now
  the density on the support, pinned against statsmodels on the reduced panel
  and an exact Kalman filter (`fixtures/dfm_smooth.json`, `singular_cases`).
* DF16-2 — `adl_midas`: OLS by QR of the design instead of Cholesky on X'X;
  level targets pinned against an exact rational least-squares solution
  (`fixtures/adl_midas.json`, `level_cases`).
* H16-11 — finite inputs at extreme magnitudes give the rescaled finite fit.
* H16-05 — `dfm_smooth` at N = 8192 no longer runs an O(N^3) factorization of
  the diagonal observation covariance (18 s and 3.1 GB before).
* H16-06 — complex arrays and cells are refused by name, not truncated.
* API16-4 — `umidas` refuses an inert `maxlags` under `se_type="nonrobust"`.
* B2 (nowcast-crate cells) — refusals name `factor_order`, `target_series`,
  `n_factors`.

Only numpy, scipy and pytest are needed (the CI wheel jobs); the statsmodels
and exact references are read from the fixtures.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import tsecon

REPO = Path(__file__).resolve().parents[3]
DFM_FIX = json.loads((REPO / "fixtures" / "dfm_smooth.json").read_text(encoding="utf-8"))
ADL_FIX = json.loads((REPO / "fixtures" / "adl_midas.json").read_text(encoding="utf-8"))


def _panel(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)


def _rel(a, e):
    a, e = np.asarray(a, float), np.asarray(e, float)
    return float(np.max(np.abs(a - e) / np.abs(e)))


# --------------------------------------------------------------------------
# DF16-1: zero idiosyncratic variances observed together
# --------------------------------------------------------------------------

def _singular_orders():
    for case in DFM_FIX["singular_cases"]:
        for o in case["orders"]:
            yield pytest.param(case, o, id=f"{case['name']}-{''.join(map(str, o['order']))}")


@pytest.mark.parametrize("case, order", list(_singular_orders()))
def test_dfm_smooth_singular_density_is_the_density_on_the_support(case, order):
    perm = order["order"]
    y = _panel(case["panel"])[:, perm]
    lam = np.asarray(case["loadings"])[perm]
    idio = np.asarray(case["idiosyncratic"])[perm]
    kw = dict(factor_ar=case["factor_ar"], factor_cov=case["factor_cov"])
    full = tsecon.dfm_smooth(y, lam, idiosyncratic=idio, **kw)
    yr = y.copy()
    yr[:, order["redundant"]] = np.nan
    reduced = tsecon.dfm_smooth(yr, lam, idiosyncratic=idio, **kw)
    # A skipped element leaves the recursion untouched: bitwise.
    assert full["loglik"] == reduced["loglik"]
    np.testing.assert_array_equal(full["smoothed_state"], reduced["smoothed_state"])
    # statsmodels on the reduced panel, and the exact rational / decimal filter.
    assert abs(full["loglik"] - order["loglik"]) <= 1e-10 * abs(order["loglik"])
    assert abs(full["loglik"] - order["loglik_exact"]) <= 1e-12 * abs(order["loglik_exact"])
    np.testing.assert_allclose(full["smoothed_state"], order["smoothed_state"], rtol=0, atol=1e-9)


def test_dfm_smooth_audit_repro_value():
    # The audit's repro: 18.657 before, the density on the support now.
    z = np.array([[0.03666137558761732, 0.5527422219007221, -0.04042486581736605], [np.nan] * 3])
    lam = [[-0.43272912833939525], [0.7196862105052038], [-0.05263433357820673]]
    r = tsecon.dfm_smooth(z, lam, [[-0.8642299294783886, -0.8484056219079968]],
                          [0.8355556816599208, 0.0, 0.0], factor_cov=[[0.13302995088165312]])
    assert abs(r["loglik"] - (-1.7368851005097792)) <= 1e-12
    # Mathematically the same point a few ulp away: the same answer.
    for k in (-3, 3):
        zz = z.copy()
        zz[0, 2] += k * np.spacing(zz[0, 2])
        rr = tsecon.dfm_smooth(zz, lam, [[-0.8642299294783886, -0.8484056219079968]],
                               [0.8355556816599208, 0.0, 0.0], factor_cov=[[0.13302995088165312]])
        assert rr["loglik"] == r["loglik"]


# --------------------------------------------------------------------------
# DF16-2: adl_midas on persistent level targets
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", ADL_FIX["level_cases"], ids=lambda c: c["name"])
def test_adl_midas_level_target_matches_exact_and_statsmodels(case):
    y = np.asarray(case["y"])
    hf = np.asarray(case["hf_lags"])
    p = case["ar_lags"]
    ex = case["exact"]
    nr = tsecon.adl_midas(y, hf, ar_lags=p, se_type="nonrobust")
    hac = tsecon.adl_midas(y, hf, ar_lags=p)
    assert hac["maxlags"] == case["default_maxlags"]
    # Against the exact least-squares solution of the same float64 design
    # (the Cholesky normal equations were 2.2e-8 / 4.9e-10 off in params).
    assert _rel(nr["params"], ex["params"]) <= 1e-11
    assert _rel(nr["bse"], ex["bse_nonrobust"]) <= 1e-12
    assert _rel(hac["bse"], ex["bse_hac_default"]) <= 1e-11
    # Against statsmodels at the documented 1e-10 (HAC: within statsmodels'
    # own distance from the exact answer).
    assert _rel(nr["params"], case["params"]) <= 1e-10
    assert _rel(nr["bse"], case["fits"]["nonrobust"]["bse"]) <= 1e-10
    assert _rel(hac["bse"], case["fits"]["hac_default"]["bse"]) <= (
        case["sm_vs_exact"]["bse_hac_default"] + 1e-11)
    assert abs(nr["rsquared"] - case["rsquared"]) <= 1e-10


# --------------------------------------------------------------------------
# H16-11: finite inputs at extreme magnitudes
# --------------------------------------------------------------------------

def _adl_data(n=120, seed=5):
    rng = np.random.default_rng(seed)
    hf = rng.standard_normal((n, 3))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + hf[t] @ [0.5, 0.2, 0.1] + rng.standard_normal()
    return y, hf


@pytest.mark.parametrize("sy, sx", [(1e77, 1.0), (1e154, 1.0), (1e155, 1.0), (1.0, 1e154),
                                    (1e300, 1.0), (1e-300, 1.0)])
@pytest.mark.parametrize("se_type", ["hac", "nonrobust"])
def test_adl_midas_extreme_magnitudes_give_the_rescaled_fit(sy, sx, se_type):
    y, hf = _adl_data()
    base = tsecon.adl_midas(y, hf, ar_lags=1, se_type=se_type)
    r = tsecon.adl_midas(y * sy, hf * sx, ar_lags=1, se_type=se_type)
    factor = np.array([sy, 1.0, sy / sx, sy / sx, sy / sx])
    for key in ("params", "bse"):
        got = np.asarray(r[key])
        assert np.all(np.isfinite(got)), (key, got)
        np.testing.assert_allclose(got, np.asarray(base[key]) * factor, rtol=1e-11)
    np.testing.assert_allclose(r["tvalues"], base["tvalues"], rtol=1e-11)
    assert abs(r["rsquared"] - base["rsquared"]) <= 1e-12


@pytest.mark.parametrize("sy, sx", [(1e154, 1.0), (1.0, 1e154), (1e155, 1.0)])
def test_umidas_extreme_magnitudes_give_the_rescaled_fit(sy, sx):
    y, hf = _adl_data()
    base = tsecon.umidas(y, hf)
    r = tsecon.umidas(y * sy, hf * sx)
    factor = np.array([sy, sy / sx, sy / sx, sy / sx])
    np.testing.assert_allclose(r["params"], np.asarray(base["params"]) * factor, rtol=1e-11)
    np.testing.assert_allclose(r["bse"], np.asarray(base["bse"]) * factor, rtol=1e-11)
    assert abs(r["rsquared"] - base["rsquared"]) <= 1e-12


# --------------------------------------------------------------------------
# API16-4: umidas maxlags under nonrobust
# --------------------------------------------------------------------------

def test_umidas_refuses_maxlags_under_nonrobust():
    y, hf = _adl_data()
    for ml in (3, 10**6):
        with pytest.raises(ValueError, match=r"umidas: maxlags = \d+ has no effect under "
                                             r"se_type=\"nonrobust\""):
            tsecon.umidas(y, hf, se_type="nonrobust", maxlags=ml)
    a = tsecon.umidas(y, hf, se_type="nonrobust")
    b = tsecon.umidas(y, hf, se_type="nonrobust")
    assert np.array_equal(a["bse"], b["bse"])
    # maxlags still matters where it is a bandwidth.
    h1 = tsecon.umidas(y, hf, se_type="hac", maxlags=1)
    h8 = tsecon.umidas(y, hf, se_type="hac", maxlags=8)
    assert not np.array_equal(h1["bse"], h8["bse"])
    # An unknown se_type is still reported first.
    with pytest.raises(ValueError, match="unknown se_type"):
        tsecon.umidas(y, hf, se_type="hc9", maxlags=3)


# --------------------------------------------------------------------------
# H16-05: dfm_smooth at the widest N
# --------------------------------------------------------------------------

@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RLIMIT_AS is Linux-only")
def test_dfm_smooth_widest_panel_fits_in_a_few_gib():
    # N = 8192, the largest the N x N guard admits: the diagonal observation
    # covariance used to go through a dense jittered Cholesky (O(N^3), about
    # six N x N buffers: 18 s and 3.1 GB RSS) and failed under 3 GiB of
    # address space with an allocation error naming no argument.
    code = textwrap.dedent("""
        import resource, numpy as np, tsecon
        resource.setrlimit(resource.RLIMIT_AS, (3 * 2**30, 3 * 2**30))
        N = 8192
        r = tsecon.dfm_smooth(np.zeros((1, N)), np.ones((N, 1)), [[0.5]], np.ones(N))
        print("ok", np.isfinite(r["loglik"]))
    """)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.split() == ["ok", "True"]


# --------------------------------------------------------------------------
# H16-06: complex input is refused, not truncated
# --------------------------------------------------------------------------

def _dfm_args():
    rng = np.random.default_rng(3)
    return (rng.standard_normal((60, 5)), rng.standard_normal((5, 1)), np.array([[0.7]]),
            np.full(5, 0.5))


@pytest.mark.parametrize("name", ["loadings", "factor_ar", "factor_cov"])
@pytest.mark.parametrize("form", ["ndarray", "np_cells", "py_cells", "complex64"])
def test_dfm_smooth_refuses_complex_matrices_by_name(name, form):
    P, L, A, ID = _dfm_args()
    kw = dict(loadings=L, factor_ar=A, idiosyncratic=ID, factor_cov=np.eye(1))
    real = np.asarray(kw[name], float)
    bad = real + 0.5j
    if form == "np_cells":
        bad = [[np.complex128(v) for v in row] for row in bad]
    elif form == "py_cells":
        bad = [[complex(v) for v in row] for row in bad]
    elif form == "complex64":
        bad = bad.astype(np.complex64)
    kw[name] = bad
    with pytest.raises(TypeError, match=rf"dfm_smooth: {name} must hold real numbers.*"
                                        r"not truncated to its real part"):
        tsecon.dfm_smooth(P, **kw)


def test_dfm_smooth_still_takes_real_lists_and_integer_arrays():
    P, L, A, ID = _dfm_args()
    base = tsecon.dfm_smooth(P, L, A, ID)
    as_lists = tsecon.dfm_smooth(P, L.tolist(), A.tolist(), ID, factor_cov=[[1]])
    assert as_lists["loglik"] == base["loglik"]
    ints = tsecon.dfm_smooth(P, L, A, ID, factor_cov=np.eye(1, dtype=int))
    assert ints["loglik"] == base["loglik"]


def _var2(n=200, seed=3):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 2))
    for t in range(1, n):
        X[t] = 0.5 * X[t - 1] + rng.standard_normal(2)
    return X


@pytest.mark.parametrize("which", ["conditions", "condition_sd"])
@pytest.mark.parametrize("form", ["ndarray", "np_cell", "py_cell", "complex64_cell"])
def test_var_conditional_forecast_refuses_complex_by_name(which, form):
    X = _var2()
    cond = [[1.0, None]]
    sd = [[0.3, None]]
    value = {"conditions": 1.0 + 7j, "condition_sd": 0.3 + 9j}[which]
    if form == "ndarray":
        bad = np.array([[value, np.nan]])
    elif form == "np_cell":
        bad = [[np.complex128(value), None]]
    elif form == "py_cell":
        bad = [[complex(value), None]]
    else:
        bad = [[np.complex64(value), None]]
    kw = dict(conditions=cond, condition_sd=sd, soft="signal")
    kw[which] = bad
    with pytest.raises(TypeError, match=rf"var_conditional_forecast: {which} must hold real "
                                        r"numbers or None.*not truncated to its real part"):
        tsecon.var_conditional_forecast(X, **kw)


def test_var_conditional_forecast_still_takes_none_cells_and_object_arrays():
    X = _var2()
    a = tsecon.var_conditional_forecast(X, [[1.0, None]])
    b = tsecon.var_conditional_forecast(X, np.array([[1.0, None]], dtype=object))
    c = tsecon.var_conditional_forecast(X, np.array([[1.0, np.nan]]))
    np.testing.assert_array_equal(a["point"], b["point"])
    np.testing.assert_array_equal(a["point"], c["point"])


def test_impact_still_refuses_complex_by_name():
    X = _var2()
    with pytest.raises(ValueError, match=r"impact must be a real numeric \(n, n\) matrix, got "
                                         r"dtype complex128: a complex value is not truncated"):
        tsecon.structural_fevd(X, lags=1, impact=np.eye(2) + 1j)


# --------------------------------------------------------------------------
# B2 (nowcast crate): refusals name the keyword
# --------------------------------------------------------------------------

def _dfm_panel(t=80, n=5, seed=11):
    rng = np.random.default_rng(seed)
    f = np.zeros(t)
    for s in range(1, t):
        f[s] = 0.6 * f[s - 1] + rng.standard_normal()
    return np.outer(f, rng.uniform(0.5, 1.5, n)) + 0.5 * rng.standard_normal((t, n))


def test_dfm_factor_order_too_large_names_factor_order():
    X = _dfm_panel()
    X[-1, :2] = np.nan  # a ragged edge: the training block is 79 rows
    for big in (40, 2**31):
        with pytest.raises(ValueError, match=rf"dfm_nowcast: factor_order = {big} is too large "
                                             r"for the 79-row balanced training block.*"
                                             r"pass factor_order <= 39"):
            tsecon.dfm_nowcast(X, factor_order=big)
    new = X.copy()
    new[-1, :2] = 0.0
    with pytest.raises(ValueError, match=r"dfm_news: factor_order = 2147483648 is too large"):
        tsecon.dfm_news(X, new, factor_order=2**31)
    # The largest order the hint names passes the row check; on this short
    # block its fitted VAR(39) is explosive, which is refused by name too.
    with pytest.raises(ValueError, match=r"dfm_nowcast: the VAR\(39\) fitted to the n_factors = 1 "
                                         r"factor\(s\) is not stationary .* Lower factor_order"):
        tsecon.dfm_nowcast(X, factor_order=39)
    assert tsecon.dfm_nowcast(X, factor_order=2)["factor_order"] == 2


def test_dfm_news_target_series_out_of_range_names_target_series():
    X = _dfm_panel()
    old = X.copy()
    old[-1, :2] = np.nan
    for bad in (5, 2**31):
        with pytest.raises(ValueError, match=rf"dfm_news: target_series = {bad} is out of range: "
                                             r"the vintages have N = 5 series .* 0\.\.=4"):
            tsecon.dfm_news(old, X, target_series=bad)


def test_factor_model_n_factors_out_of_range_names_n_factors():
    X = _dfm_panel()
    for bad in (0, 6):
        with pytest.raises(ValueError, match=rf"factor_model: n_factors = {bad} is out of range: "
                                             r".* pass 1 <= n_factors <= 5"):
            tsecon.factor_model(X, n_factors=bad, kmax=3)
