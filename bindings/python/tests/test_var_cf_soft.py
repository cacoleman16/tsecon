"""Golden, parity and behavioural tests for the 0.11.0 soft (Gaussian,
stated-variance) conditions of `tsecon.var_conditional_forecast`
(`condition_sd=`, `soft="density"` / `soft="signal"`).

Re-pins fixtures/var_cf_soft.json through the Python surface (the NumPy
closed form of both readings at 1e-10; statsmodels VARMAX — signal: a
time-varying measurement-noise covariance, density: the hard smoother plus
an exact perturbation gain — at 1e-8; the APW structural form for density),
then runs the statsmodels legs DIRECTLY on systems the fixture does not
store, checks that no soft cell IS the hard path (dict-equal, and the
var_cf.json pins), the spellings of `condition_sd`, determinism, the
refusals, and the docstring / signature / docs contracts.
"""
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import tsecon

try:  # extras-gated: the live-parity tests need statsmodels
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.statespace.varmax import VARMAX

    HAVE_SM = True
except ImportError:  # pragma: no cover - extras absent
    HAVE_SM = False

ROOT = Path(__file__).parents[3]
FIX = ROOT / "fixtures"
CF = json.loads((FIX / "var_cf.json").read_text(encoding="utf-8"))
SOFT = json.loads((FIX / "var_cf_soft.json").read_text(encoding="utf-8"))
CASES = SOFT["cases"]
IDS = [f"{c['name']}-{c['mode']}" for c in CASES]
OLD_KEYS = ("point", "unconditional", "cov", "se", "unconditional_se", "lower", "upper",
            "shocks", "orth_shocks", "constrained", "n_constrained", "mahalanobis",
            "mahalanobis_pvalue", "steps", "alpha")
NEW_KEYS = ("mode", "condition_sd", "n_soft")

needs_sm = pytest.mark.skipif(not HAVE_SM, reason="statsmodels not installed")


def _series(name):
    return np.array(CF["series"][name])


def _call(case, **override):
    kw = dict(
        conditions=case["conditions"],
        lags=case["p"],
        trend=case["trend"],
        steps=case["steps"],
        alpha=case["alpha"],
        condition_sd=case["condition_sd_input"],
        soft=case["mode"],
    )
    kw.update(override)
    return tsecon.var_conditional_forecast(_series(case["series"]), **kw)


def _close(a, b, tol):
    np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                               rtol=tol, atol=tol)


def _y3():
    return _series("sim_var2_k3")


# ------------------------------------------------------------------ golden

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_soft_repins_the_closed_form(case):
    r = _call(case)
    assert set(r) == set(OLD_KEYS) | set(NEW_KEYS)
    assert r["mode"] == case["mode"] and r["n_soft"] == case["n_soft"]
    assert r["condition_sd"] == case["condition_sd"]
    assert r["constrained"] == case["constrained"]
    assert r["n_constrained"] == case["n_constrained"]
    assert r["steps"] == case["steps"] and r["alpha"] == case["alpha"]
    for key in ("point", "unconditional", "cov", "se", "unconditional_se", "lower",
                "upper", "shocks", "orth_shocks"):
        _close(r[key], case[key], 1e-10)
    assert r["mahalanobis"] == pytest.approx(case["mahalanobis"], rel=1e-10, abs=1e-12)
    assert r["mahalanobis_pvalue"] == pytest.approx(case["mahalanobis_pvalue"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_soft_matches_the_statsmodels_leg(case):
    r = _call(case)
    _close(r["point"], case["bgl_point"], 1e-8)
    _close(r["cov"], case["bgl_cov"], 1e-8)
    _close(r["shocks"], case["bgl_shocks"], 1e-8)
    # statsmodels' future log-likelihood is log N(r; 0, M): a check on the
    # returned mahalanobis.
    m = r["n_constrained"]
    ll = -0.5 * (m * math.log(2 * math.pi) + case["ln_det_m"] + r["mahalanobis"])
    assert ll == pytest.approx(case["bgl_llf_future"], rel=1e-8, abs=1e-8)
    if case["mode"] == "density":
        _close(r["point"], case["apw_point"], 1e-8)
        _close(r["cov"], case["apw_cov"], 1e-8)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_exact_cells_per_mode(case):
    r = _call(case)
    point, se, cov = np.asarray(r["point"]), np.asarray(r["se"]), np.asarray(r["cov"])
    sd = np.asarray(r["condition_sd"])
    for h, row in enumerate(case["conditions"]):
        for j, v in enumerate(row):
            if v is None:
                assert sd[h, j] == 0.0 and not r["constrained"][h][j]
                continue
            assert r["constrained"][h][j]
            if sd[h, j] == 0.0:  # hard: pinned, zero row and column
                assert point[h, j] == v and se[h, j] == 0.0
                assert np.all(cov[h, j, :] == 0.0) and np.all(cov[h, :, j] == 0.0)
            elif r["mode"] == "density":
                assert point[h, j] == v and se[h, j] == sd[h, j]
                assert cov[h, j, j] == sd[h, j] ** 2
                others = [l for l in range(cov.shape[1]) if l != j and r["constrained"][h][l]]
                assert all(cov[h, j, l] == 0.0 == cov[h, l, j] for l in others)
            else:  # signal: the posterior, not pinned
                assert se[h, j] <= sd[h, j] and se[h, j] < r["unconditional_se"][h][j]
                if sd[h, j] > 1e-6:
                    assert point[h, j] != v


# -------------------------------------------- no soft cell = the hard path

@pytest.mark.parametrize("case", CF["cases"], ids=[c["name"] for c in CF["cases"]])
def test_no_soft_cell_is_the_hard_path(case):
    y = _series(case["series"])
    kw = dict(lags=case["p"], trend=case["trend"], steps=case["steps"], alpha=case["alpha"])
    k = y.shape[1]
    rows = len(case["conditions"])
    base = tsecon.var_conditional_forecast(y, case["conditions"], **kw)
    assert base["mode"] == "hard" and base["n_soft"] == 0
    assert base["condition_sd"] == [[0.0] * k for _ in range(case["steps"])]
    for sd in ([[None] * k] * rows, np.full((rows, k), np.nan), np.zeros((rows, k))):
        r = tsecon.var_conditional_forecast(y, case["conditions"], condition_sd=sd, **kw)
        assert r == base
    for key in ("point", "unconditional", "cov", "se", "unconditional_se", "lower",
                "upper", "shocks", "orth_shocks"):
        _close(base[key], case[key], 1e-10)
    assert base["constrained"] == case["constrained"]


def test_spellings_of_condition_sd_agree():
    case = next(c for c in CASES if c["name"] == "macro_infl_band" and c["mode"] == "signal")
    ref = _call(case)
    assert len(case["conditions"]) < case["steps"]  # a short `conditions` ...
    rows, k = len(case["conditions"]), 3
    grid = np.full((rows, k), np.nan)  # ... with condition_sd of the same row count
    for h, row in enumerate(case["condition_sd_input"]):
        for j, v in enumerate(row):
            if v is not None:
                grid[h, j] = v
    assert _call(case, condition_sd=grid) == ref
    pd = pytest.importorskip("pandas")
    assert _call(case, condition_sd=pd.DataFrame(grid)) == ref
    cgrid = np.full((rows, k), np.nan)
    for h, row in enumerate(case["conditions"]):
        for j, v in enumerate(row):
            if v is not None:
                cgrid[h, j] = v
    assert _call(case, conditions=pd.DataFrame(cgrid), condition_sd=pd.DataFrame(grid)) == ref
    # Hard cells spelled 0 instead of NaN/None change nothing.
    assert _call(case, condition_sd=np.nan_to_num(grid, nan=0.0)) == ref
    # Integer standard deviations are fine.
    r = tsecon.var_conditional_forecast(_y3(), [[None, 0.3, None]], condition_sd=[[None, 1, None]],
                                        soft="density")
    assert r["se"][0][1] == 1.0 and r["n_soft"] == 1


def test_soft_is_deterministic():
    for case in CASES[:3]:
        assert _call(case) == _call(case)


def test_the_two_readings_differ_where_they_should():
    case_d = next(c for c in CASES if c["name"] == "var2c_soft_path" and c["mode"] == "density")
    d = _call(case_d)
    s = _call(case_d, soft="signal")
    hard = _call(case_d, condition_sd=None, soft=None)
    # density: the hard path at the central values, a wider covariance
    assert d["point"] == hard["point"] and d["shocks"] == hard["shocks"]
    assert d["mahalanobis"] == hard["mahalanobis"]
    assert np.all(np.asarray(d["se"]) >= np.asarray(hard["se"]) * (1 - 1e-12))
    # signal: the readings are not pinned, and the plausibility uses Occ + Om
    assert s["point"][0][1] != 0.3 and s["mahalanobis"] < hard["mahalanobis"]


# ------------------------------------------------------- live statsmodels

def _macro_levels():
    md = sm.datasets.macrodata.load_pandas().data
    g = 100.0 * np.diff(np.log(md["realgdp"].to_numpy(dtype=float)))
    return np.column_stack([g, md["infl"].to_numpy(dtype=float)[1:],
                            md["tbilrate"].to_numpy(dtype=float)[1:]])


def _varmax_smooth(y, res, H, cond, trend, obs_var=None):
    """statsmodels VARMAX(...NaN future...).smooth at the OLS fit; obs_var
    (H x k, 0 = none) is installed as a time-varying obs_cov."""
    k, p, T = res.neqs, res.k_ar, y.shape[0]
    mod = VARMAX(np.vstack([y, cond]), order=(p, 0), trend=trend, error_cov_type="unstructured")
    P = np.linalg.cholesky(np.asarray(res.sigma_u))
    params = np.zeros(len(mod.param_names))
    for i, nm in enumerate(mod.param_names):
        if nm.startswith("intercept."):
            params[i] = res.intercept[int(nm.split(".y")[1]) - 1]
        elif nm.startswith("L"):
            lag_s, src, eq = nm.split(".")
            params[i] = res.coefs[int(lag_s[1:]) - 1][int(eq[1:]) - 1, int(src[1:]) - 1]
        elif nm.startswith("sqrt.var."):
            j = int(nm.split(".y")[1]) - 1
            params[i] = P[j, j]
        else:
            a, b = nm[len("sqrt.cov."):].split(".")
            ia, ib = int(a[1:]) - 1, int(b[1:]) - 1
            params[i] = P[max(ia, ib), min(ia, ib)]
    if obs_var is not None:
        oc = np.zeros((k, k, T + H))
        for h in range(H):
            for j in range(k):
                oc[j, j, T + h] = obs_var[h, j]
        mod.update(params)
        mod.ssm["obs_cov"] = oc
        sres = mod.smooth(params)
        # VARMAX.update writes obs_cov only with measurement_error=True: the
        # installed array must be what the RESULTS object filtered with.
        assert np.array_equal(np.asarray(sres.filter_results.obs_cov), oc)
    else:
        sres = mod.smooth(params)
    path = np.asarray(sres.smoothed_state[:k, T:T + H].T)
    cov = np.transpose(np.asarray(sres.smoothed_state_cov[:k, :k, T:T + H]), (2, 0, 1))
    shocks = np.asarray(sres.smoothed_state_disturbance[:k, T - 1:T + H - 1].T)
    return path, cov, shocks


def _fit(y, p, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VAR(y).fit(p, trend=trend)


@needs_sm
def test_live_signal_leg_on_macrodata_levels():
    y = _macro_levels()  # growth, inflation, T-bill LEVEL: not a fixture system
    res = _fit(y, 2, "c")
    H, last = 8, y[-1]
    cond = np.full((H, 3), np.nan)
    sd = np.zeros((H, 3))
    cond[:3, 1], sd[:3, 1] = last[1], [0.5, 1.0, 1.5]   # inflation read with noise
    cond[:2, 2] = last[2]                               # the T-bill hard for two
    r = tsecon.var_conditional_forecast(y, cond, lags=2, steps=H, condition_sd=sd, soft="signal")
    path, cov, shocks = _varmax_smooth(y, res, H, cond, "c", obs_var=sd ** 2)
    _close(r["point"], path, 1e-8)
    _close(r["cov"], cov, 1e-8)
    _close(r["shocks"], shocks, 1e-8)
    assert r["mode"] == "signal" and r["n_soft"] == 3 and r["n_constrained"] == 5


@needs_sm
def test_live_signal_leg_on_a_fresh_var3_without_constant():
    rng = np.random.default_rng(20261009)
    k, n = 2, 250
    A = np.array([[0.4, 0.15], [-0.1, 0.5]])
    L = np.array([[1.0, 0.0], [-0.6, 0.7]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + L @ rng.standard_normal(k)
    res = _fit(y, 3, "n")
    H = 6
    cond = np.full((H, k), np.nan)
    sd = np.zeros((H, k))
    cond[0, 0], cond[3, 1], cond[5, 0] = 0.5, -0.2, 0.0
    sd[0, 0], sd[3, 1] = 0.3, 0.8
    r = tsecon.var_conditional_forecast(y, cond, lags=3, trend="n", condition_sd=sd, soft="signal")
    path, cov, shocks = _varmax_smooth(y, res, H, cond, "n", obs_var=sd ** 2)
    _close(r["point"], path, 1e-8)
    _close(r["cov"], cov, 1e-8)
    _close(r["shocks"], shocks, 1e-8)


@needs_sm
def test_live_density_leg_hard_smoother_plus_perturbation_gain():
    y = _macro_levels()
    res = _fit(y, 2, "c")
    H, last = 6, y[-1]
    cond = np.full((H, 3), np.nan)
    sd = np.zeros((H, 3))
    cond[:3, 1], sd[:3, 1] = last[1], [0.5, 0.75, 1.0]
    cond[1, 2] = last[2]
    r = tsecon.var_conditional_forecast(y, cond, lags=2, steps=H, condition_sd=sd, soft="density")
    path0, cov0, shocks0 = _varmax_smooth(y, res, H, cond, "c")
    cov = cov0.copy()
    for h in range(H):
        for j in range(3):
            if sd[h, j] > 0:
                c2 = cond.copy()
                c2[h, j] += 1.0
                kcol = _varmax_smooth(y, res, H, c2, "c")[0] - path0
                for hh in range(H):
                    cov[hh] += sd[h, j] ** 2 * np.outer(kcol[hh], kcol[hh])
    _close(r["point"], path0, 1e-8)
    _close(r["cov"], cov, 1e-8)
    _close(r["shocks"], shocks0, 1e-8)


# ---------------------------------------------------------------- refusals

def test_refusals_name_the_parameter_and_cell():
    y = _y3()
    cond = [[None, 0.3, None], [None, None, -0.2]]
    ok = [[None, 0.2, None], [None, None, None]]
    tsecon.var_conditional_forecast(y, cond, condition_sd=ok, soft="signal")
    bad = lambda sd, **kw: tsecon.var_conditional_forecast(y, cond, condition_sd=sd, soft="signal", **kw)
    with pytest.raises(ValueError, match=re.escape(
            "condition_sd[0][1] = -0.1: a standard deviation must be finite and >= 0 "
            "(0 or NaN means a hard condition)")):
        bad([[None, -0.1, None], [None, None, None]])
    with pytest.raises(ValueError, match=re.escape("condition_sd[1][2] = inf")):
        bad([[None, None, None], [None, None, np.inf]])
    with pytest.raises(ValueError, match=re.escape("condition_sd[0][1] = 1e200: its square")):
        bad([[None, 1e200, None], [None, None, None]])
    with pytest.raises(ValueError, match=re.escape(
            "condition_sd[0][0] = 0.5 but conditions[0][0] is free: give the cell a value "
            "or drop the sd")):
        bad([[0.5, 0.2, None], [None, None, None]])
    with pytest.raises(ValueError, match=r"condition_sd has 1 rows but conditions has 2.*expected 2, got 1"):
        bad([[None, 0.2, None]])
    with pytest.raises(ValueError, match=r"condition_sd\[1\] has 2 entries.*expected 3, got 2"):
        bad([[None, 0.2, None], [None, None]])
    # soft must name the reading whenever any sd > 0 ...
    with pytest.raises(ValueError, match=re.escape(
            "condition_sd gives 1 cell a standard deviation > 0, so soft must say which "
            "reading you mean: soft='density'")):
        tsecon.var_conditional_forecast(y, cond, condition_sd=ok)
    # ... and is refused as inert without one.
    for sd in (None, [[0.0, 0.0, 0.0], [None, None, np.nan]]):
        with pytest.raises(ValueError, match=r"soft='density' has nothing to act on"):
            tsecon.var_conditional_forecast(y, cond, condition_sd=sd, soft="density")
    with pytest.raises(ValueError, match=r"soft='tilt' is not a soft-condition reading: pass "
                                         r"soft='density'.*soft='signal'"):
        tsecon.var_conditional_forecast(y, cond, condition_sd=ok, soft="tilt")
    # A non-str soft is named, never the int `steps` beside it.
    with pytest.raises(TypeError) as info:
        tsecon.var_conditional_forecast(y, cond, steps=4, condition_sd=ok, soft=1)
    msg = str(info.value)
    assert "soft=1 is of type int" in msg and "steps" not in msg.split("soft takes")[0], msg
    # The existing refusals are unchanged (and checked through the soft path).
    with pytest.raises(ValueError, match="conditions contains an infinite value"):
        tsecon.var_conditional_forecast(y, [[None, np.inf, None]], condition_sd=[[None, 0.1, None]],
                                        soft="density")
    with pytest.raises(ValueError, match="steps = 0"):
        bad(ok, steps=0)
    with pytest.raises(ValueError, match="alpha"):
        bad(ok, alpha=0.0)
    for big in (2**40, 2**47):
        with pytest.raises(ValueError, match="steps"):
            bad(ok, steps=big)


def test_a_bad_cell_is_named_by_its_position_and_soft_is_not_blamed():
    """A str cell is named "condition_sd[0][1]='x'" — and with soft="density"
    passed beside it (also a str), soft is not named too."""
    y = _y3()
    cond = [[None, 0.3, None]]
    with pytest.raises(TypeError) as info:
        tsecon.var_conditional_forecast(y, cond, condition_sd=[[None, "x", None]], soft="density")
    msg = str(info.value)
    assert "condition_sd[0][1]='x'" in msg and "soft=" not in msg, msg
    with pytest.raises(TypeError) as info:
        tsecon.var_conditional_forecast(y, [["x", None, None]], condition_sd=[[None, 0.1, None]],
                                        soft="signal")
    msg = str(info.value)
    assert "conditions[0][0]='x'" in msg and "soft=" not in msg and "condition_sd" not in msg, msg
    for bad, frag in (("abc", "condition_sd='abc' is of type str"),
                      (0.5, "condition_sd=0.5 is of type float"),
                      ([0.5, None, None], "condition_sd[0]=0.5 is of type float")):
        with pytest.raises(TypeError, match=re.escape(frag)):
            tsecon.var_conditional_forecast(y, cond, condition_sd=bad, soft="density")
    with pytest.raises(TypeError, match=re.escape("conditions=None is of type NoneType")):
        tsecon.var_conditional_forecast(y, None)


# ------------------------------------------------------ contract tripwires

def _doc_tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _stub_doc(name):
    stub = (ROOT / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    m = re.search(rf"def {name}\(.*?\"\"\"(.*?)\"\"\"", stub, re.S)
    assert m, name
    return m.group(1)


def test_docstrings_name_all_18_keys_and_the_signature_is_extended_at_the_end():
    import inspect
    case = CASES[0]
    out = _call(case)
    assert len(out) == 18
    for text in (tsecon.var_conditional_forecast.__doc__, _stub_doc("var_conditional_forecast")):
        missing = set(out) - _doc_tokens(text)
        assert not missing, missing
        flat = re.sub(r"\s+", " ", text)
        assert "QUADRATIC in `steps`" in flat
        assert "Andersson-Palmqvist-Waggoner 2010" in flat and "Monti 2010" in flat
    sig = inspect.signature(tsecon.var_conditional_forecast)
    assert list(sig.parameters) == ["data", "conditions", "lags", "trend", "steps", "alpha",
                                    "condition_sd", "soft"]
    assert sig.parameters["condition_sd"].default is None
    assert sig.parameters["soft"].default is None


# ---------------------------------------------------------- docs examples

CARD_EXPECTED = (
    "mode: density  soft cells: 4  plausibility p = 0.294\n"
    "series 1, h=1..4  point: [0.5 0.5 0.5 0.5]  se: [0.2 0.2 0.2 0.2]\n"
    "series 0, h=1..4  point: [-0.129 -0.015  0.012  0.006]  se: [0.294 0.333 0.342 0.344]\n"
    "mode: signal  soft cells: 4  plausibility p = 0.438\n"
    "series 1, h=1..4  point: [0.39  0.414 0.41  0.387]  se: [0.162 0.164 0.164 0.167]\n"
    "series 0, h=1..4  point: [-0.149 -0.044 -0.014 -0.016]  se: [0.294 0.332 0.341 0.344]\n"
)


def test_var_svar_card_soft_example_prints_what_it_says(capsys):
    rng = np.random.default_rng(0)
    k, n = 3, 400
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
    cond = np.full((8, k), np.nan)
    cond[:4, 1] = 0.5
    cond[1, 2] = 0.0
    sd = np.zeros((8, k))
    sd[:4, 1] = 0.2      # series 1 "at 0.5, give or take 0.2"; series 2 stays hard
    for soft in ("density", "signal"):
        cf = tsecon.var_conditional_forecast(Y, cond, lags=2, steps=8, condition_sd=sd, soft=soft)
        p, s = np.array(cf["point"]), np.array(cf["se"])
        print("mode:", cf["mode"], " soft cells:", cf["n_soft"],
              " plausibility p =", round(cf["mahalanobis_pvalue"], 3))
        print("series 1, h=1..4  point:", np.round(p[:4, 1], 3), " se:", np.round(s[:4, 1], 3))
        print("series 0, h=1..4  point:", np.round(p[:4, 0], 3), " se:", np.round(s[:4, 0], 3))
    out = capsys.readouterr().out
    assert out == CARD_EXPECTED
    card = (ROOT / "docs" / "reference" / "model-cards" / "var-svar.md").read_text(encoding="utf-8")
    for line in CARD_EXPECTED.splitlines():
        assert line in card, f"card does not show: {line!r}"
    assert 'sd[:4, 1] = 0.2      # series 1 "at 0.5, give or take 0.2"; series 2 stays hard' in card


def _guide_chapter_7_data():
    rng = np.random.default_rng(42)
    T, burn = 400, 100
    A1 = np.array([[0.5, 0.0, -0.2], [0.3, 0.4, -0.1], [0.2, 0.1, 0.7]])
    shocks = rng.normal(size=(T + burn, 3)) * np.array([1.0, 0.8, 0.5])
    y = np.zeros((T + burn, 3))
    for t in range(1, T + burn):
        y[t] = A1 @ y[t - 1] + shocks[t]
    return y[burn:]


def test_guide_chapter_7_soft_snippet_runs_as_written():
    """The guide's soft-condition snippet runs verbatim on the chapter's
    `data`, and the paragraph under it says what the numbers say."""
    data = _guide_chapter_7_data()
    H = 8
    cond = np.full((H, data.shape[1]), np.nan)
    cond[:4, 2] = data[-1, 2]
    cond[0, 0] = 0.5
    sd = np.zeros((H, data.shape[1]))
    sd[:4, 2] = 0.25
    hard = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H)
    dens = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H, condition_sd=sd, soft="density")
    sig = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H, condition_sd=sd, soft="signal")
    assert dens["point"] == hard["point"]
    assert np.all(np.array(dens["se"])[:4, 2] == 0.25)
    assert np.all(np.array(sig["se"])[:4, 2] < 0.25)
    out_h = np.array(hard["se"])[:, 1]
    out_d = np.array(dens["se"])[:, 1]
    out_s = np.array(sig["se"])[:, 1]
    assert np.all(out_d >= out_h) and np.all(out_s >= out_h)
    assert np.all(np.array(dens["upper"])[:4, 2] - np.array(dens["lower"])[:4, 2]
                  == pytest.approx(2 * 1.959963984540054 * 0.25, rel=1e-12))
    # The numbers the paragraph quotes.
    assert (round(out_h[2], 3), round(out_d[2], 3)) == (0.885, 0.894)
    sig_rate_se = np.round(np.array(sig["se"])[:4, 2], 3)
    assert (sig_rate_se.min(), sig_rate_se.max()) == (0.215, 0.23)
    moved = np.abs(np.array(sig["point"])[:4, 2] - data[-1, 2])
    assert round(moved.max(), 2) == 0.06 and int(np.argmax(moved)) == 3
    guide = (ROOT / "docs" / "guide" / "07-multivariate.md").read_text(encoding="utf-8")
    for line in ("sd = np.zeros((H, data.shape[1]))",
                 "sd[:4, 2] = 0.25",
                 'dens = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H, condition_sd=sd, soft="density")',
                 'sig = tsecon.var_conditional_forecast(data, cond, lags=1, steps=H, condition_sd=sd, soft="signal")'):
        assert line in guide, line
    assert "with a 25bp standard deviation" in guide
    assert "±25bp" not in guide
