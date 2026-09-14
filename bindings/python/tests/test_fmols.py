"""Golden and behavioral tests for the single-equation cointegrating
regressions ``tsecon.fmols`` / ``tsecon.dols`` / ``tsecon.ccr``.

Re-pins ``fixtures/fmols.json`` (arch 8.0 ``FullyModifiedOLS`` /
``DynamicOLS`` / ``CanonicalCointegratingReg`` — see the generator header
for the two documented deviations: the Andrews bandwidth rule and CCR
under ``df_adjust``) through the Python surface, calls ``arch`` directly on
fresh data when it is importable, re-downloads the two Rdatasets of the
real-data illustration (skipped offline) and pins arch's numbers on them,
and exercises the sentinel refusals, the teaching errors, determinism and
the stub / docstring contracts.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIX / "fmols.json").read_text())
SYSTEMS = {k: (np.asarray(v["y"], float), np.asarray(v["x"], float)) for k, v in FX["systems"].items()}


def _coint_kwargs(case):
    kw = dict(trend=case["trend"], kernel=case["kernel"], bandwidth=case["bandwidth_arg"],
              force_int=case["force_int"], df_adjust=case["df_adjust"])
    if case["bandwidth_rule"] is not None:
        kw["bandwidth_rule"] = case["bandwidth_rule"]
    if case["diff"]:
        kw["diff"] = True
    if case["x_trend"] is not None:
        kw["x_trend"] = case["x_trend"]
    return kw


def _dols_kwargs(case):
    kw = dict(trend=case["trend"], lags=case["lags_arg"], leads=case["leads_arg"],
              cov_type=case["cov_type"], kernel=case["kernel"], bandwidth=case["bandwidth_arg"],
              force_int=case["force_int"], df_adjust=case["df_adjust"])
    if case["ic"] != "bic":  # the default; passing it with fixed lags/leads is refused
        kw["ic"] = case["ic"]
    if case["common"]:
        kw["common"] = True
    if case["max_lag_arg"] is not None:
        kw["max_lag"] = case["max_lag_arg"]
    if case["max_lead_arg"] is not None:
        kw["max_lead"] = case["max_lead_arg"]
    if case["bandwidth_rule"] is not None:
        kw["bandwidth_rule"] = case["bandwidth_rule"]
    return kw


def _case_id(c):
    bits = [c["estimator"], c["system"], c["trend"], c["kernel"]]
    if c["estimator"] == "dols":
        bits += [f"l{c['lags_arg']}", f"L{c['leads_arg']}", c["cov_type"], c["ic"]]
    bits += [f"bw{c['bandwidth_arg']}", f"fi{int(c['force_int'])}", f"df{int(c['df_adjust'])}"]
    return "-".join(bits)


CASES_FM = [c for c in FX["cases"] if c["estimator"] == "fmols"]
CASES_CCR = [c for c in FX["cases"] if c["estimator"] == "ccr"]
CASES_DOLS = [c for c in FX["cases"] if c["estimator"] == "dols"]


def _check_common(r, case):
    assert r["param_names"] == case["param_names"]
    assert r["bandwidth"] == pytest.approx(case["bandwidth"], rel=1e-10)
    np.testing.assert_allclose(r["params"], case["params"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["se"], case["se"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["tvalues"], case["tvalues"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["pvalues"], case["pvalues"], rtol=1e-9, atol=1e-14)
    np.testing.assert_allclose(np.asarray(r["cov"]), np.asarray(case["cov"]), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["resid"], case["resid"], rtol=1e-10, atol=1e-10)
    assert r["rsquared"] == pytest.approx(case["rsquared"], rel=1e-10)
    assert r["rsquared_adj"] == pytest.approx(case["rsquared_adj"], rel=1e-10)
    assert r["long_run_variance"] == pytest.approx(case["long_run_variance"], rel=1e-10)
    np.testing.assert_allclose(r["ols_params"], case["ols_params"], rtol=1e-10, atol=1e-10)
    if case["bandwidth_arg"] is None:
        assert r["bandwidth_rule"] == (case["bandwidth_rule"] or "newey-west")
    else:
        assert r["bandwidth_rule"] is None


# ---------------------------------------------------------------- golden

@pytest.mark.parametrize("case", CASES_FM, ids=_case_id)
def test_fmols_matches_fixture(case):
    y, x = SYSTEMS[case["system"]]
    r = tsecon.fmols(y, x, **_coint_kwargs(case))
    assert r["estimator"] == "Fully Modified OLS"
    assert r["nobs"] == len(y) and r["n_x"] == x.shape[1]
    _check_common(r, case)
    omega = np.asarray(r["omega"])
    assert omega.shape == (1 + x.shape[1], 1 + x.shape[1])
    np.testing.assert_allclose(omega, omega.T, rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", CASES_CCR, ids=_case_id)
def test_ccr_matches_fixture(case):
    y, x = SYSTEMS[case["system"]]
    r = tsecon.ccr(y, x, **_coint_kwargs(case))
    assert r["estimator"] == "Canonical Cointegrating Regression"
    _check_common(r, case)
    if "arch_cov_00" in case:
        # The documented df_adjust deviation: the pinned value is the
        # documented (T-1)/(T-1-k) scaling of the unadjusted covariance,
        # and arch's raw value differs from it.
        base = tsecon.ccr(y, x, **{**_coint_kwargs(case), "df_adjust": False})
        m, k = len(y) - 1, len(r["params"])
        assert r["cov"][0][0] == pytest.approx(m / (m - k) * base["cov"][0][0], rel=1e-12)
        assert abs(r["cov"][0][0] - case["arch_cov_00"]) > 1e-9 * r["cov"][0][0]


@pytest.mark.parametrize("case", CASES_DOLS, ids=_case_id)
def test_dols_matches_fixture(case):
    y, x = SYSTEMS[case["system"]]
    r = tsecon.dols(y, x, **_dols_kwargs(case))
    assert (r["lags"], r["leads"], r["nobs"]) == (case["lags"], case["leads"], case["nobs"])
    assert r["n_total"] == len(y)
    _check_common(r, case)
    np.testing.assert_allclose(r["full_params"], case["full_params"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["full_se"], case["full_se"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r["full_cov"]), np.asarray(case["full_cov"]), rtol=1e-10, atol=1e-10)
    assert r["full_param_names"] and len(r["full_param_names"]) == len(r["full_params"])
    assert r["selected"] == (case["lags_arg"] is None or case["leads_arg"] is None)
    assert math.isfinite(r["ic_value"]) == r["selected"]
    assert r["cov_type"] == case["cov_type"] and r["ic"] == case["ic"]


def test_fixture_covers_the_option_grid():
    for cases in (CASES_FM, CASES_CCR):
        assert {c["trend"] for c in cases} == {"n", "c", "ct", "ctt"}
        assert {c["kernel"] for c in cases} == {"bartlett", "parzen", "quadratic-spectral"}
        assert any(c["bandwidth_rule"] == "andrews" for c in cases)
        assert any(c["diff"] for c in cases) and any(c["x_trend"] for c in cases)
        assert any(c["df_adjust"] for c in cases) and any(not c["force_int"] for c in cases)
    assert {c["cov_type"] for c in CASES_DOLS} == {"unadjusted", "robust"}
    assert {c["ic"] for c in CASES_DOLS} == {"aic", "bic", "hqic"}
    assert any(c["common"] for c in CASES_DOLS)
    assert any(c["max_lag_arg"] is not None for c in CASES_DOLS)


# ------------------------------------------------------ arch, directly

def _arch():
    return pytest.importorskip("arch.unitroot.cointegration")


def _fresh_system(seed=7, t=180, kx=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((t, kx))
    x = np.cumsum(u, axis=0)
    e = np.zeros(t)
    eps = rng.standard_normal(t)
    for i in range(1, t):
        e[i] = 0.4 * e[i - 1] + eps[i] + 0.5 * u[i, 0]
    y = 1.0 + x @ np.array([1.0, -0.5])[:kx] + e
    return y, x


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic-spectral"])
def test_fmols_and_ccr_agree_with_arch_on_fresh_data(trend, kernel):
    coint = _arch()
    y, x = _fresh_system()
    arch_kernel = kernel.replace("-", "")  # arch's FM-OLS fit rejects the hyphenated spelling
    for name, cls, fn in (("fmols", coint.FullyModifiedOLS, tsecon.fmols),
                          ("ccr", coint.CanonicalCointegratingReg, tsecon.ccr)):
        ref = cls(y, x, trend=trend).fit(kernel=arch_kernel)
        r = fn(y, x, trend=trend, kernel=kernel)
        assert r["bandwidth"] == pytest.approx(ref.bandwidth, rel=1e-10), name
        np.testing.assert_allclose(r["params"], np.asarray(ref.params), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.asarray(r["cov"]), np.asarray(ref.cov), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r["resid"], np.asarray(ref.resid), rtol=1e-10, atol=1e-10)
        assert r["rsquared"] == pytest.approx(ref.rsquared, rel=1e-10)
        assert r["param_names"] == list(ref.params.index)


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
@pytest.mark.parametrize("ic", ["aic", "bic", "hqic"])
def test_dols_agrees_with_arch_on_fresh_data(cov_type, ic):
    coint = _arch()
    y, x = _fresh_system(seed=11)
    ref = coint.DynamicOLS(y, x, trend="c", method=ic).fit(cov_type=cov_type)
    r = tsecon.dols(y, x, trend="c", ic=ic, cov_type=cov_type)
    assert (r["lags"], r["leads"]) == (ref.lags, ref.leads)
    assert r["bandwidth"] == pytest.approx(ref.bandwidth, rel=1e-10)
    np.testing.assert_allclose(r["full_params"], np.asarray(ref.full_params), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(r["full_cov"]), np.asarray(ref.full_cov), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["params"], np.asarray(ref.params), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(r["resid"], np.asarray(ref.resid), rtol=1e-10, atol=1e-10)
    assert r["rsquared_adj"] == pytest.approx(ref.rsquared_adj, rel=1e-10)


def test_long_run_pieces_agree_with_arch_kernels():
    kern = pytest.importorskip("arch.covariance.kernel")
    y, x = _fresh_system(seed=3)
    r = tsecon.fmols(y, x, trend="c", kernel="parzen", bandwidth=6.0)
    # Rebuild eta as the estimator does and hand it to arch's Parzen.
    z = np.column_stack([x, np.ones(len(y))])
    eta1 = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    tr = np.ones((len(y), 1))
    eta2 = np.diff(x - tr @ np.linalg.lstsq(tr, x, rcond=None)[0], axis=0)
    eta = np.column_stack([eta1[1:], eta2])
    est = kern.Parzen(eta, bandwidth=6.0, center=False)
    np.testing.assert_allclose(np.asarray(r["omega"]), np.asarray(est.cov.long_run), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(r["lambda"]), np.asarray(est.cov.one_sided), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(r["sigma"]), np.asarray(est.cov.short_run), rtol=1e-10)
    assert r["n_lags"] == 6


# ----------------------------------------------------------- real data

def _rdataset(item, package):
    sm = pytest.importorskip("statsmodels.api")
    try:
        return sm.datasets.get_rdataset(item, package).data
    except Exception as exc:  # pragma: no cover - offline container
        pytest.skip(f"{item}/{package} not reachable: {exc!r}")


@pytest.mark.parametrize("key", sorted(FX["real"]))
def test_real_data_illustration_reproduces_arch(key):
    rec = FX["real"][key]
    d = _rdataset(rec["item"], rec["package"])
    y = d[rec["y"]].to_numpy(dtype=float)
    x = d[rec["x"]].to_numpy(dtype=float)
    assert len(y) == rec["T"]
    fm = tsecon.fmols(y, x, trend="c")
    cc = tsecon.ccr(y, x, trend="c")
    do = tsecon.dols(y, x, trend="c")
    d11 = tsecon.dols(y, x, trend="c", lags=1, leads=1)
    for name, r in (("fmols", fm), ("ccr", cc), ("dols", do), ("dols_11", d11)):
        want = rec[name]
        np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r["se"], want["se"], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(r["tvalues"], want["tvalues"], rtol=1e-10, atol=1e-10)
        assert r["bandwidth"] == pytest.approx(want["bandwidth"], rel=1e-10)
        assert r["rsquared"] == pytest.approx(want["rsquared"], rel=1e-10)
        if name.startswith("dols"):
            assert (r["lags"], r["leads"]) == (want["lags"], want["leads"])
    np.testing.assert_allclose(fm["ols_params"], rec["ols_params"], rtol=1e-10)
    # And against arch itself when it is importable.
    coint = pytest.importorskip("arch.unitroot.cointegration")
    ref = coint.FullyModifiedOLS(y, x, trend="c").fit()
    np.testing.assert_allclose(fm["params"], np.asarray(ref.params), rtol=1e-10)
    np.testing.assert_allclose(fm["se"], np.asarray(ref.std_errors), rtol=1e-10)


# ------------------------------------------------------ sentinel rules

def test_bandwidth_rule_with_explicit_bandwidth_is_refused():
    y, x = SYSTEMS["sim_k1"]
    for fn in (tsecon.fmols, tsecon.ccr, tsecon.dols):
        with pytest.raises(ValueError, match="bandwidth_rule"):
            fn(y, x, bandwidth=3.0, bandwidth_rule="andrews")
        # ... and acts when the bandwidth is automatic.
        nw = fn(y, x)
        an = fn(y, x, bandwidth_rule="andrews")
        assert nw["bandwidth_rule"] == "newey-west" and an["bandwidth_rule"] == "andrews"
        assert nw["bandwidth"] != an["bandwidth"]
        ex = fn(y, x, bandwidth=an["bandwidth"])
        assert ex["bandwidth_rule"] is None
        np.testing.assert_allclose(ex["params"], an["params"], rtol=1e-12)


def test_diff_without_a_trend_is_refused():
    y, x = SYSTEMS["sim_k1"]
    for fn in (tsecon.fmols, tsecon.ccr):
        with pytest.raises(ValueError, match=r"diff = (True|False) was given"):
            fn(y, x, trend="c", diff=True)
        with pytest.raises(ValueError, match="diff"):
            fn(y, x, trend="n", diff=False)
        # A trend on either side makes it act.
        a = fn(y, x, trend="ct", diff=True)
        b = fn(y, x, trend="ct")
        assert a["diff"] and not b["diff"]
        assert not np.allclose(a["params"], b["params"], rtol=1e-8)
        c = fn(y, x, trend="c", x_trend="ct", diff=True)
        assert c["x_trend"] == "ct" and c["trend"] == "c"


def test_dols_inert_search_options_are_refused():
    y, x = SYSTEMS["sim_k1"]
    with pytest.raises(ValueError, match='ic = "aic"'):
        tsecon.dols(y, x, lags=1, leads=1, ic="aic")
    with pytest.raises(ValueError, match="max_lag = 3"):
        tsecon.dols(y, x, lags=1, max_lag=3)
    with pytest.raises(ValueError, match="max_lead = 3"):
        tsecon.dols(y, x, leads=1, max_lead=3)
    with pytest.raises(ValueError, match="common = True"):
        tsecon.dols(y, x, lags=1, leads=1, common=True)
    # Each acts in its mode.
    a = tsecon.dols(y, x, lags=1, ic="aic")
    b = tsecon.dols(y, x, lags=1, ic="bic")
    assert a["ic"] == "aic" and b["ic"] == "bic" and a["lags"] == b["lags"] == 1
    capped = tsecon.dols(y, x, max_lag=1, max_lead=1)
    assert capped["lags"] <= 1 and capped["leads"] <= 1 and capped["max_lag"] == 1
    common = tsecon.dols(y, x, common=True, max_lag=2, max_lead=2)
    assert common["lags"] == common["leads"] and common["common"]


# ------------------------------------------------------ teaching errors

def test_refusals_name_the_parameter():
    y, x = SYSTEMS["sim_k1"]
    for fn in (tsecon.fmols, tsecon.ccr, tsecon.dols):
        with pytest.raises(ValueError, match=r"x .*rows.*expected 200, got 199"):
            fn(y, x[:-1])
        with pytest.raises(ValueError, match=r"bandwidth = -1"):
            fn(y, x, bandwidth=-1.0)
        with pytest.raises(ValueError, match=r"bandwidth = NaN"):
            fn(y, x, bandwidth=float("nan"))
        with pytest.raises(ValueError, match=r'unknown trend "quadratic"'):
            fn(y, x, trend="quadratic")
        with pytest.raises(ValueError, match=r'unknown kernel "tukey"'):
            fn(y, x, kernel="tukey")
        with pytest.raises(ValueError, match=r'unknown bandwidth_rule "silverman"'):
            fn(y, x, bandwidth_rule="silverman")
        with pytest.raises(ValueError, match=r"x must hold at least one regressor"):
            fn(y, np.empty((200, 0)))
        with pytest.raises(ValueError, match=r"non-finite value .* y"):
            yy = y.copy(); yy[5] = np.nan
            fn(yy, x)
        with pytest.raises(ValueError, match=r"non-finite value .* x at row 4, column 0"):
            xx = x.copy(); xx[4, 0] = np.inf
            fn(y, xx)
        with pytest.raises(TypeError):
            fn(y, x, trend=3)  # non-string trend
    with pytest.raises(ValueError, match=r'x_trend = "c" .* trend = "ct"'):
        tsecon.fmols(y, x, trend="ct", x_trend="c")
    with pytest.raises(ValueError, match=r"lags = 150 and leads = 150"):
        tsecon.dols(y, x, lags=150, leads=150)
    with pytest.raises(ValueError, match=r"max_lag = 120"):
        tsecon.dols(y, x, max_lag=120, max_lead=0)
    with pytest.raises(ValueError, match=r'unknown ic "hq"'):
        tsecon.dols(y, x, ic="hq")
    with pytest.raises(ValueError, match=r'unknown cov_type "hc0"'):
        tsecon.dols(y, x, cov_type="hc0")
    with pytest.raises(ValueError, match=r"T = 4"):
        tsecon.fmols(y[:4], x[:4])
    # The underdetermined default search arch runs silently is refused.
    ref = FX["refusals"][0]
    ys, xs = SYSTEMS[ref["system"]]
    with pytest.raises(ValueError, match=r"max_lag = 11 .*max_lead = 11") as exc:
        tsecon.dols(ys, xs, trend=ref["trend"])
    assert str(ref["rows"]) in str(exc.value)


# ---------------------------------------------------------- behaviour

def test_corrections_move_the_estimate_toward_the_truth_on_the_fixture_dgp():
    """On the fixture DGP (endogenous regressors, AR(1) equilibrium error)
    the corrected estimators all sit closer to the true beta than plain
    OLS on the k_x = 3 system — the second-order OLS bias the corrections
    exist for. The quantitative claim is the property-test MC."""
    y, x = SYSTEMS["sim_k3"]
    beta = np.asarray(FX["systems"]["sim_k3"]["truth"]["beta"])
    fm = tsecon.fmols(y, x)
    ols_err = np.abs(np.asarray(fm["ols_params"][:3]) - beta).sum()
    for r in (fm, tsecon.ccr(y, x), tsecon.dols(y, x)):
        assert np.abs(np.asarray(r["params"][:3]) - beta).sum() < ols_err


def test_determinism_and_pandas_inputs():
    pd = pytest.importorskip("pandas")
    y, x = SYSTEMS["sim_k3"]
    a = tsecon.fmols(y, x, trend="ct")
    b = tsecon.fmols(pd.Series(y), pd.DataFrame(x), trend="ct")
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), key
        else:
            assert va == vb, key
    d1 = tsecon.dols(y, x)
    d2 = tsecon.dols(list(y), np.asarray(x, dtype=np.float32))  # a list and a float32 array
    np.testing.assert_allclose(d1["full_params"], d2["full_params"], rtol=1e-4)  # float32 input
    d3 = tsecon.dols(list(y), pd.DataFrame(x))
    assert np.array_equal(d1["full_params"], d3["full_params"])
    assert d1["lags"] == d2["lags"] and d1["leads"] == d2["leads"]


def test_bandwidth_zero_and_force_int_conventions():
    y, x = SYSTEMS["sim_k1"]
    r0 = tsecon.fmols(y, x, kernel="quadratic-spectral", bandwidth=0.0)
    np.testing.assert_allclose(np.asarray(r0["omega"]), np.asarray(r0["sigma"]), rtol=1e-14)
    assert r0["n_lags"] == len(y) - 2  # T - 1 residual rows, every positive lag
    r = tsecon.fmols(y, x, bandwidth=3.7)          # force_int=True ceils it
    assert r["bandwidth"] == 4.0 and r["n_lags"] == 4
    r = tsecon.fmols(y, x, bandwidth=3.7, force_int=False)
    assert r["bandwidth"] == 3.7 and r["n_lags"] == 3  # arch's floor(bw) window
    d = tsecon.dols(y, x, lags=1, leads=1)
    assert d["bandwidth"] != math.ceil(d["bandwidth"]) or d["bandwidth"] == 0
    d = tsecon.dols(y, x, lags=1, leads=1, force_int=True)
    assert d["bandwidth"] == math.ceil(d["bandwidth"])


# --------------------------------------------------------- contracts

@pytest.mark.parametrize("name", ["fmols", "dols", "ccr"])
def test_docstrings_name_every_returned_key(name):
    import re
    y, x = SYSTEMS["sim_k1"]
    fn = getattr(tsecon, name)
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))
    keys = set(fn(y, x).keys())
    missing = keys - tokens
    assert not missing, f"{name}.__doc__ does not name returned keys: {sorted(missing)}"


@pytest.mark.parametrize("name", ["fmols", "dols", "ccr"])
def test_no_ellipsis_defaults(name):
    import inspect
    sig = inspect.signature(getattr(tsecon._core, name))
    for p in sig.parameters.values():
        assert p.default is not Ellipsis, p
