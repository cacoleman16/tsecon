"""`tsecon.dfm_smooth` — the fixed-parameter dynamic-factor Kalman smoother
(0.11; crate tsecon-nowcast `smooth_standardized`).

Re-pins fixtures/dfm_smooth.json (statsmodels DynamicFactor.smooth and the
generic KalmanSmoother), checks the vintage re-smoothing round trip against
dfm_nowcast on both estimation routes BIT FOR BIT, a live statsmodels parity
on a fresh design, the input forms (nested lists, pandas), every refusal by
the keyword it names, the size caps, and the docstring/stub contracts.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

REPO = Path(__file__).resolve().parents[3]
FIX = json.loads((REPO / "fixtures" / "dfm_smooth.json").read_text(encoding="utf-8"))
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
KEYS = {"loglik", "smoothed_factors", "smoothed_state", "smoothed_state_cov", "edge_factor",
        "nowcast", "n_factors", "factor_order"}
MODEL_KEYS = ("loadings", "factor_ar", "factor_cov", "idiosyncratic", "center", "scale")


def _case(name):
    return next(c for c in FIX["cases"] if c["name"] == name)


def _panel(c):
    return np.array([[np.nan if v is None else v for v in row] for row in c["panel"]], dtype=float)


def _args(c):
    return dict(loadings=np.asarray(c["loadings"]), factor_ar=np.asarray(c["factor_ar"]),
                idiosyncratic=np.asarray(c["idiosyncratic"]), factor_cov=np.asarray(c["factor_cov"]))


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _tokens(text):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", text or ""))


def _ragged_factor_panel(T=160, N=7, seed=11):
    rng = np.random.default_rng(seed)
    f = np.zeros(T)
    for t in range(2, T):
        f[t] = 0.6 * f[t - 1] - 0.15 * f[t - 2] + rng.standard_normal()
    lam = rng.uniform(0.5, 1.5, N)
    x = np.outer(f, lam) + 0.5 * rng.standard_normal((T, N)) + rng.uniform(-3, 3, N)
    x[-1, :4] = np.nan
    x[-2, :2] = np.nan
    return x


# --------------------------------------------------------------- goldens
@pytest.mark.parametrize("name", ["dynamic_factor_r2_p2", "kalman_smoother_q"])
def test_raw_panel_matches_statsmodels(name):
    c = _case(name)
    out = tsecon.dfm_smooth(_panel(c), **_args(c))
    assert set(out) == KEYS
    assert out["loglik"] == pytest.approx(c["loglik"], rel=1e-8)
    np.testing.assert_allclose(out["smoothed_state"], c["smoothed_state"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["smoothed_state_cov"], c["smoothed_state_cov"], rtol=0, atol=1e-8)
    r, p = c["n_factors"], c["factor_order"]
    assert (out["n_factors"], out["factor_order"]) == (r, p)
    T = len(c["panel"])
    assert out["smoothed_factors"].shape == (T, r)
    assert out["smoothed_state"].shape == (T, r * p)
    assert out["smoothed_state_cov"].shape == (T, r * p, r * p)
    np.testing.assert_array_equal(out["smoothed_factors"], out["smoothed_state"][:, :r])
    np.testing.assert_array_equal(out["edge_factor"], out["smoothed_factors"][-1])
    # Default center/scale: nowcast is loadings @ edge_factor on the raw scale.
    np.testing.assert_allclose(out["nowcast"], np.asarray(c["loadings"]) @ out["edge_factor"],
                               rtol=1e-14, atol=1e-14)


def test_standardized_panel_and_edge_nowcast_match_reference():
    c = _case("standardized_r1_p3")
    out = tsecon.dfm_smooth(_panel(c), **_args(c), center=np.asarray(c["center"]),
                            scale=np.asarray(c["scale"]))
    assert out["loglik"] == pytest.approx(c["loglik"], rel=1e-8)
    np.testing.assert_allclose(out["smoothed_state"], c["smoothed_state"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["smoothed_state_cov"], c["smoothed_state_cov"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["edge_factor"], c["edge_factor"], rtol=0, atol=1e-8)
    np.testing.assert_allclose(out["nowcast"], c["nowcast"], rtol=1e-8)


def test_live_statsmodels_parity_on_a_fresh_design():
    sm_df = pytest.importorskip("statsmodels.tsa.statespace.dynamic_factor")
    rng = np.random.default_rng(5)
    T, N = 90, 4
    f = np.zeros(T)
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    lam = rng.uniform(0.5, 1.2, N)
    idio = rng.uniform(0.3, 0.8, N)
    y = np.outer(f, lam) + rng.standard_normal((T, N)) * np.sqrt(idio)
    y[-1, 1:] = np.nan
    mod = sm_df.DynamicFactor(y, k_factors=1, factor_order=1, error_order=0)
    res = mod.smooth(np.r_[lam, idio, 0.7], transformed=True)
    out = tsecon.dfm_smooth(y, loadings=lam.reshape(-1, 1), factor_ar=[[0.7]], idiosyncratic=idio)
    assert out["loglik"] == pytest.approx(float(res.llf), rel=1e-9)
    np.testing.assert_allclose(out["smoothed_state"][:, 0], np.asarray(res.smoothed_state)[0],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(out["smoothed_state_cov"][:, 0, 0],
                               np.asarray(res.smoothed_state_cov)[0, 0], rtol=0, atol=1e-9)


# --------------------------------------------- the vintage round trip
@pytest.mark.parametrize("method,r", [("two_step", 2), ("mle", 1)])
def test_resmoothing_reproduces_dfm_nowcast_bit_for_bit(method, r):
    x = _ragged_factor_panel()
    fit = tsecon.dfm_nowcast(x, n_factors=r, factor_order=2, method=method)
    out = tsecon.dfm_smooth(x, **{k: fit[k] for k in MODEL_KEYS})
    assert out["loglik"] == fit["loglik"]
    np.testing.assert_array_equal(out["edge_factor"], fit["edge_factor"])
    np.testing.assert_array_equal(out["nowcast"], fit["nowcast"])
    # dfm_nowcast's smoothed_factors is the BALANCED training block, a nested
    # list; dfm_smooth's covers the full panel, as an ndarray.
    T_b = len(fit["smoothed_factors"])
    assert isinstance(fit["smoothed_factors"], list)
    assert T_b == x.shape[0] - 2
    assert out["smoothed_factors"].shape == (x.shape[0], r)


def test_a_new_vintage_moves_the_nowcast_the_way_the_data_do():
    x = _ragged_factor_panel()
    fit = tsecon.dfm_nowcast(x, n_factors=1, factor_order=2)
    model = {k: fit[k] for k in MODEL_KEYS}
    old = tsecon.dfm_smooth(x, **model)
    new_x = x.copy()
    lam = np.asarray(fit["loadings"])[:, 0]
    j = int(np.argmax(np.abs(lam)))
    j = j if j < 4 else 0
    # Release a large reading in the direction of a series' loading.
    new_x[-1, j] = fit["center"][j] + 3.0 * fit["scale"][j] * np.sign(lam[j])
    new = tsecon.dfm_smooth(new_x, **model)
    assert new["edge_factor"][0] * 1.0 > old["edge_factor"][0]
    # The released cell's nowcast moves towards the release.
    assert abs(new["nowcast"][j] - new_x[-1, j]) < abs(old["nowcast"][j] - new_x[-1, j])


def test_defaults_are_identity_zero_and_one_bitwise():
    c = _case("dynamic_factor_r2_p2")
    y = _panel(c)
    a = tsecon.dfm_smooth(y, loadings=c["loadings"], factor_ar=c["factor_ar"],
                          idiosyncratic=c["idiosyncratic"])
    n = len(c["idiosyncratic"])
    b = tsecon.dfm_smooth(y, loadings=c["loadings"], factor_ar=c["factor_ar"],
                          idiosyncratic=c["idiosyncratic"], factor_cov=np.eye(2),
                          center=np.zeros(n), scale=np.ones(n))
    for k in KEYS:
        np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]))


def test_nested_lists_arrays_and_pandas_are_interchangeable():
    pd = pytest.importorskip("pandas")
    c = _case("kalman_smoother_q")
    y = _panel(c)
    lists = tsecon.dfm_smooth(y, loadings=c["loadings"], factor_ar=c["factor_ar"],
                              idiosyncratic=c["idiosyncratic"], factor_cov=c["factor_cov"])
    frames = tsecon.dfm_smooth(pd.DataFrame(y), loadings=pd.DataFrame(c["loadings"]),
                               factor_ar=np.asfortranarray(c["factor_ar"]),
                               idiosyncratic=pd.Series(c["idiosyncratic"]),
                               factor_cov=np.asfortranarray(c["factor_cov"]))
    for k in KEYS:
        np.testing.assert_array_equal(np.asarray(lists[k]), np.asarray(frames[k]))


def test_an_all_missing_period_is_predicted_not_refused():
    c = _case("dynamic_factor_r2_p2")
    y = _panel(c)
    y[40] = np.nan
    out = tsecon.dfm_smooth(y, **_args(c))
    assert np.isfinite(out["loglik"])
    assert np.all(np.isfinite(out["smoothed_state"]))
    # Less information at the gap: its factor posterior variance is larger.
    v = out["smoothed_state_cov"][:, 0, 0]
    assert v[40] > v[39] and v[40] > v[41]


# --------------------------------------------------------------- refusals
def _good():
    c = _case("standardized_r1_p3")
    return _panel(c), dict(**_args(c), center=np.asarray(c["center"]), scale=np.asarray(c["scale"]))


def _bad(**over):
    y, kw = _good()
    if "data" in over:
        y = over.pop("data")
    kw.update(over)
    return y, kw


@pytest.mark.parametrize(
    "over,needle",
    [
        (dict(loadings=np.ones(5)), "loadings is 1-D"),
        (dict(loadings=np.ones((5, 0))), "loadings has shape"),
        (dict(loadings=np.full((5, 1), np.nan)), "loadings contains a non-finite"),
        (dict(loadings=[[1.0], [1.0, 2.0]]), "loadings is a ragged"),
        (dict(loadings="abc"), "loadings must be a 2-D"),
        (dict(factor_ar=np.ones((2, 3))), "factor_ar has 2 row"),
        (dict(factor_ar=np.ones((1, 0))), "factor_ar has 0 column"),
        (dict(factor_ar=np.array([[1.2, 0.0, 0.0]])), "factor_ar describes a non-stationary"),
        (dict(factor_ar=np.array([[np.inf, 0.0, 0.0]])), "factor_ar contains a non-finite"),
        (dict(factor_cov=np.eye(2)), "factor_cov has shape"),
        (dict(factor_cov=np.array([[-1.0]])), "factor_cov is not a symmetric positive semidefinite"),
        (dict(idiosyncratic=np.array([0.5, 0.5, -0.1, 0.5, 0.5])), "idiosyncratic[2] = -0.1"),
        (dict(idiosyncratic=np.ones(4)), "idiosyncratic has 4 entries"),
        (dict(idiosyncratic=np.array([0.5, np.nan, 0.5, 0.5, 0.5])), "idiosyncratic[1] = NaN"),
        (dict(center=np.zeros(3)), "center has 3 entries"),
        (dict(center=np.array([0.0, np.inf, 0.0, 0.0, 0.0])), "center[1] = inf"),
        (dict(scale=np.ones(6)), "scale has 6 entries"),
        (dict(scale=np.array([1.0, 1.0, 0.0, 1.0, 1.0])), "scale[2] = 0"),
        (dict(scale=np.array([1.0, -2.0, 1.0, 1.0, 1.0])), "scale[1] = -2"),
        (dict(data=np.zeros((0, 5))), "data has zero rows"),
        (dict(data=np.zeros((10, 4))), "data has 4 column"),
    ],
)
def test_every_refusal_names_the_keyword(over, needle):
    y, kw = _bad(**over)
    with pytest.raises(ValueError) as exc:
        tsecon.dfm_smooth(y, **kw)
    msg = str(exc.value)
    assert needle in msg, msg
    # The engine's internal names never leak: the caller typed no "Q" or "H".
    assert "Q must" not in msg and " H " not in msg and "transition matrix" not in msg


def test_infinite_data_is_refused_but_nan_is_missing():
    y, kw = _good()
    y = y.copy()
    y[3, 2] = np.inf
    with pytest.raises(ValueError, match=r"data contains an infinite value \(inf\) at \[3, 2\]"):
        tsecon.dfm_smooth(y, **kw)


def test_one_dimensional_data_is_refused_naming_data():
    y, kw = _good()
    with pytest.raises((TypeError, ValueError), match="data"):
        tsecon.dfm_smooth(y[:, 0], **kw)


def test_size_caps_refuse_before_allocating():
    # A long AR order for one factor: T * m^2 past 2^26 cells.
    y = np.zeros((300, 1))
    with pytest.raises(ValueError, match=r"T \* m\^2"):
        tsecon.dfm_smooth(y, loadings=[[1.0]], factor_ar=np.zeros((1, 600)), idiosyncratic=[1.0])
    # N^2 observation covariance past 2^26 cells.
    n = 9000
    with pytest.raises(ValueError, match="8192 series"):
        tsecon.dfm_smooth(np.zeros((2, n)), loadings=np.ones((n, 1)), factor_ar=[[0.5]],
                          idiosyncratic=np.ones(n))
    # Work: T * (2 m^3 + 20 N m^2 + N^2 + 1000 N + 12000) + 8 m^3 past 2^34
    # (both size caps pass): 3.01e7 units per period at N = 5000, m = 1, so
    # 600 periods are refused.
    n = 5000
    with pytest.raises(ValueError, match=r"T \* \(2 m\^3 \+ 20 N m\^2 \+ N\^2 \+ 1000 N \+ 12000\)"):
        tsecon.dfm_smooth(np.zeros((600, n)), loadings=np.ones((n, 1)), factor_ar=[[0.5]],
                          idiosyncratic=np.ones(n))
    # The stationary initialization is paid once per call whatever T is: a
    # single period with m = 8 * 150 = 1200 states costs 10 m^3 + ... =
    # 1.73e10 units (the companion eigenvalue check and the Lyapunov
    # doubling), refused.
    with pytest.raises(ValueError, match=r"\+ 8 m\^3 = 1\.73e10 work units"):
        tsecon.dfm_smooth(np.zeros((1, 2)), loadings=np.ones((2, 8)),
                          factor_ar=np.zeros((8, 1200)), idiosyncratic=np.ones(2))
    # Inside every cap the call runs (300 periods: 9.0e9 units).
    out = tsecon.dfm_smooth(np.zeros((300, n)), loadings=np.ones((n, 1)), factor_ar=[[0.5]],
                            idiosyncratic=np.ones(n))
    assert out["smoothed_factors"].shape == (300, 1)


def test_a_constant_in_place_of_the_model_is_named():
    y, kw = _good()
    kw["factor_ar"] = 0.5
    with pytest.raises(ValueError, match="factor_ar must be a 2-D"):
        tsecon.dfm_smooth(y, **kw)


# --------------------------------------------------------------- contracts
def test_every_returned_key_is_named_in_both_docstrings():
    y, kw = _good()
    keys = set(tsecon.dfm_smooth(y, **kw))
    assert keys == KEYS
    assert not keys - _tokens(tsecon.dfm_smooth.__doc__)
    assert not keys - _tokens(_stub_doc("dfm_smooth"))


def test_docs_state_the_round_trip_and_the_smoothed_factors_difference():
    for doc in (tsecon.dfm_smooth.__doc__, _stub_doc("dfm_smooth")):
        flat = re.sub(r"\s+", " ", doc)
        assert "bit for bit" in flat
        assert "(T_b, r)" in flat and "FULL panel" in flat
        assert "STANDARDIZED" in flat
    flat = re.sub(r"\s+", " ", tsecon.dfm_nowcast.__doc__)
    assert "dfm_smooth" in flat


def test_the_cost_caps_are_documented_with_their_measurement():
    for doc in (tsecon.dfm_smooth.__doc__, _stub_doc("dfm_smooth")):
        flat = re.sub(r"\s+", " ", doc)
        assert "2^26" in flat and "2^34" in flat and "8 m^3" in flat
    card = (REPO / "docs" / "reference" / "model-cards" / "nowcasting-midas.md").read_text(encoding="utf-8")
    assert "**What it costs.**" in card
    assert "lab/audit/wave11/dfm_smooth_cost.py" in card and "lab/audit/wave11/adl_midas_cost.py" in card
    assert (REPO / "lab" / "audit" / "wave11" / "out" / "dfm_smooth_cost.txt").exists()
