"""Regression pins for adversarial audit round 14.

Round 14 swept the thirteen callables the 0.10.0 wave added
(`unobserved_components`, `tvp_regression`, `ets_fit`, `auto_ets`,
`var_conditional_forecast`, `var_diagnostics`, `var_select_order`, `spa_test`,
`model_confidence_set`, `stepm_test`, `fmols`, `dols`, `ccr`) plus the `mask=`
parameter added to the four panel callables. Each test below pins one
confirmed finding; the report is `docs/roadmap/29-audit-round-14-findings.md`
and the sweep scripts are under `lab/audit/round14/`.
"""
import ast
import inspect
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import tsecon

REPO = Path(__file__).resolve().parents[3]
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
CARDS = REPO / "docs" / "reference" / "model-cards"


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _ets_series(n=200, period=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 10.0 + 0.02 * t + 2.0 * np.sin(2 * np.pi * t / period) + rng.standard_normal(n) * 0.3


def _var3(n=200, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(3)
    return y


def _coint(n=200, k=2, seed=0):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal((n, k)), axis=0)
    return 1.5 * x[:, 0] - 0.5 * x[:, 1] + rng.standard_normal(n), x


def _dl_panel(N=6, T=200, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(20.0, 5.0, N)
    temp = mu[:, None] + rng.standard_normal((N, T))
    growth = (
        rng.normal(0.0, 1.0, N)[:, None]
        + rng.normal(0.0, 0.5, T)[None, :]
        + 0.30 * temp
        - 0.0075 * temp**2
        + rng.standard_normal((N, T))
    )
    return growth, temp[None]


def _ragged(N, T):
    m = np.ones((N, T))
    m[0, : T // 10] = 0.0
    m[1, -(T // 10) :] = 0.0
    return m


# --------------------------------------------------------------- L1 (sweep S)
@pytest.mark.parametrize(
    "call, offender",
    [
        (lambda: tsecon.ets_fit(_ets_series(), seasonal="add", seasonal_periods=4,
                                initialization="heuristic", smoothing_params=[0.3]),
         "smoothing_params"),
        (lambda: tsecon.ets_fit(_ets_series(), seasonal="add", seasonal_periods=4,
                                initialization="known", initial_states=[1.0]),
         "initial_states"),
        (lambda: tsecon.ets_fit(_ets_series(), seasonal="add", seasonal_periods=2**31),
         "seasonal_periods"),
        (lambda: tsecon.ets_fit(_ets_series(), seasonal="add", seasonal_periods=250),
         "seasonal_periods"),
        (lambda: tsecon.auto_ets(_ets_series(), seasonal_periods=4, horizon=0, seed=3),
         "seed"),
        (lambda: tsecon.auto_ets(_ets_series(), seasonal_periods=4, horizon=0, n_sim=99),
         "n_sim"),
    ],
)
def test_ets_refusals_name_the_offending_parameter(call, offender):
    """Round 14, sweep S: the length and sample-size refusals of `ets_fit`
    described the concept ("smoothing parameters", "fitting ETS(A,A,A) with
    2147483653 parameters") and never the keyword the user typed."""
    with pytest.raises(ValueError) as info:
        call()
    msg = str(info.value)
    assert re.search(rf"(?<![\w.]){offender}(?![\w])", msg), msg


def test_a_seasonal_period_longer_than_the_sample_is_refused_by_name():
    """The guard is on the PERIOD, not on the parameter count it inflates."""
    y = _ets_series(n=200)
    with pytest.raises(ValueError, match=r"seasonal_periods = 201"):
        tsecon.ets_fit(y, seasonal="add", seasonal_periods=201)
    ok = tsecon.ets_fit(y, seasonal="add", seasonal_periods=4)
    assert ok["seasonal_periods"] == 4


# --------------------------------------------------------------- L2 (sweep S)
@pytest.mark.parametrize("fn", ["model_confidence_set", "spa_test", "stepm_test"])
def test_automatic_block_length_failure_names_losses_and_reads_as_a_sentence(fn):
    """Round 14, sweep S: a degenerate loss column was refused as
    "block_size = None is invalid: requires the automatic Politis-White block
    length could not be computed ..." — ungrammatical, and it blamed a
    parameter the caller never passed."""
    # a sample too short for the Politis-White rule: the automatic block
    # length cannot be computed, which is the path whose message was garbled
    rng = np.random.default_rng(0)
    short = rng.standard_normal((5, 3)) ** 2
    with pytest.raises(ValueError) as info:
        if fn == "model_confidence_set":
            tsecon.model_confidence_set(short, reps=50)
        else:
            getattr(tsecon, fn)(short[:, 0], short[:, 1:], reps=50)
    msg = str(info.value)
    assert "losses" in msg, msg
    assert "requires a block length" in msg, msg
    assert "requires the automatic" not in msg, msg


# --------------------------------------------------------------- L3 (sweep S)
def test_a_bad_cell_inside_a_nested_list_argument_is_named_by_its_position():
    """Round 14, sweep S: a string inside `conditions` produced
    "an argument of type str is of type str" — a tautology that named no
    parameter, because the offender scan only looked one level into a list."""
    with pytest.raises(TypeError) as info:
        tsecon.var_conditional_forecast(_var3(), [["x", None, None]])
    msg = str(info.value)
    assert "conditions[0][0]='x'" in msg, msg
    assert "of type str is of type str" not in msg, msg
    assert "an argument of type" not in msg, msg


def test_round_13s_shallow_list_message_shape_is_unchanged():
    """The positional spelling is used only two levels down, so round 13's
    pinned `delays=[1.5]` wording still holds."""
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError) as info:
        tsecon.setar(rng.standard_normal(200), 1, delays=[1.5])
    assert "delays=[1.5]" in str(info.value)


# --------------------------------------------------------------- L4 (sweep F)
@pytest.mark.parametrize("fn", ["fmols", "ccr"])
@pytest.mark.parametrize("trend, n_det", [("n", 0), ("c", 1), ("ct", 2)])
def test_df_adjust_scales_by_the_estimated_coefficient_count_not_just_k(fn, trend, n_det):
    """Round 14, sweep F: every surface said the factor is `(T-1)/(T-1-k)`
    with `k` the I(1) regressors; it counts the DETERMINISTICS too."""
    y, x = _coint()
    base = getattr(tsecon, fn)(y, x, trend=trend)
    adj = getattr(tsecon, fn)(y, x, trend=trend, df_adjust=True)
    m = base["nobs"] - 1
    p = len(base["params"])
    assert p == base["n_x"] + base["n_det"] and base["n_det"] == n_det
    want = m / (m - p)
    got = np.asarray(adj["cov"])[0, 0] / np.asarray(base["cov"])[0, 0]
    assert got == pytest.approx(want, rel=1e-12)
    assert got != pytest.approx(m / (m - base["n_x"]), rel=1e-12) or n_det == 0


@pytest.mark.parametrize("fn", ["fmols", "ccr"])
def test_every_surface_states_the_df_adjust_factor_over_the_coefficient_count(fn):
    doc = getattr(tsecon, fn).__doc__
    surfaces = {"__doc__": doc, "stub": _stub_doc(fn)}
    for label, text in surfaces.items():
        flat = re.sub(r"\s+", " ", text)
        assert "(T-1)/(T-1-p)" in flat, (label, flat[:400])
        assert "(T-1)/(T-1-k)" not in flat, label
    card = (CARDS / "cointegration-regime.md").read_text(encoding="utf-8")
    assert "(T−1)/(T−1−p)" in card
    assert "`T/(T−k)`" not in card


# --------------------------------------------------------------- L5 (sweep F)
@pytest.mark.parametrize("fn", ["fmols", "ccr"])
def test_diff_false_is_refused_where_the_default_runs_and_the_doc_says_so(fn):
    """`diff` defaults to the `None` sentinel; the docstring used to call the
    default `False`, which is a value that RAISES at the default trend."""
    y, x = _coint()
    getattr(tsecon, fn)(y, x)                       # the default runs
    with pytest.raises(ValueError, match="diff"):
        getattr(tsecon, fn)(y, x, diff=False)
    with pytest.raises(ValueError, match="diff"):
        getattr(tsecon, fn)(y, x, diff=True)
    getattr(tsecon, fn)(y, x, trend="ct", diff=True)  # live with a trend term
    # `ccr` defers its option prose to `fmols` ("Same inputs, options and keys
    # as `fmols`"), so the sentence lives there; both tails state the sentinel
    for text in (tsecon.fmols.__doc__, _stub_doc("fmols")):
        flat = re.sub(r"\s+", " ", text)
        assert "`diff` (default None, which behaves as False)" in flat, flat[:400]
        assert "`diff` (default False)" not in flat
    for text in (getattr(tsecon, fn).__doc__, _stub_doc(fn)):
        assert "`diff` (None: False)" in re.sub(r"\s+", " ", text)


# --------------------------------------------------------------- L6 (sweep F)
def test_the_guide_panel_bullets_are_calls_that_run():
    """Round 14, sweep F: guide 14 advertised
    `panel_fe(..., se_type="cluster", bandwidth=4.0)`, a call that raises
    because `bandwidth` is refused under any `se_type` but Driscoll-Kraay."""
    guide = (REPO / "docs" / "guide" / "14-panel-time-series.md").read_text(encoding="utf-8")
    bullet = re.search(r"`tsecon\.panel_fe\(([^`]*)\)`", guide).group(1)
    kwargs = dict(re.findall(r"(\w+)=(\"[a-z_]+\"|None|True|False|[\d.]+)", bullet))
    assert kwargs.get("bandwidth") == "None" and kwargs.get("mask") == "None", kwargs
    y, x = _dl_panel(6, 60)
    tsecon.panel_fe(y, x, se_type=kwargs["se_type"].strip('"'), bandwidth=None, mask=None)
    with pytest.raises(ValueError, match="bandwidth"):
        tsecon.panel_fe(y, x, se_type="cluster", bandwidth=4.0)
    for name in ("panel_distributed_lag", "panel_lp"):
        sig = re.search(rf"`tsecon\.{name}\(([^`]*)\)`", guide).group(1)
        assert "mask=None" in sig, name


# --------------------------------------------------------------- L7 (sweep S)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, 1.0, 1e300])
def test_band_alpha_is_validated_even_without_a_band(bad):
    """Round 14, sweep S: `band_alpha` has a concrete default, so it cannot be
    refused as an inert keyword — but it was not validated either, and
    `band_alpha=nan` with `band=None` returned in silence."""
    rng = np.random.default_rng(0)
    y, shock = rng.standard_normal((6, 120)), rng.standard_normal(120)
    with pytest.raises(ValueError, match="band_alpha"):
        tsecon.panel_lp(y, shock, horizon=3, band_alpha=bad)
    with pytest.raises(ValueError, match="band_alpha"):
        tsecon.lp(rng.standard_normal(200), rng.standard_normal(200), band_alpha=bad)
    tsecon.panel_lp(y, shock, horizon=3)            # the default still runs


# --------------------------------------------------------------- M1 (sweep G)
def test_the_unbalanced_two_way_panel_cost_is_documented_and_bounded():
    """Round 14, sweep G: with a mask AND time effects `panel_distributed_lag`
    is cubic in T (38 s at T=3200, N=6) against 3 ms unmasked. The algorithm
    is the exact FWL route the PanelOLS goldens pin, so the fix is to say so
    — and to keep a ceiling far enough above the measurement to catch a
    regression without being flaky."""
    flat = re.sub(r"\s+", " ", tsecon.panel_distributed_lag.__doc__)
    assert "CUBIC in the" in flat and "time_effects=True" in flat, flat[:300]
    assert "CUBIC in the" in re.sub(r"\s+", " ", _stub_doc("panel_distributed_lag"))
    card = (CARDS / "panel.md").read_text(encoding="utf-8")
    assert "**What it costs.**" in card and "cube of the number of periods" in card
    y, x = _dl_panel(6, 400)
    mask = _ragged(6, 400)
    t0 = time.perf_counter()
    r = tsecon.panel_distributed_lag(y, x, lags=1, powers=2, mask=mask)
    slow = time.perf_counter() - t0
    t0 = time.perf_counter()
    fast = tsecon.panel_distributed_lag(y, x, lags=1, powers=2, mask=mask, time_effects=False)
    quick = time.perf_counter() - t0
    assert r["nobs"] > 0 and fast["nobs"] > 0
    assert slow < 10.0, slow          # measured 0.11 s; 90x headroom
    assert quick < 2.0, quick         # measured 0.0005 s


# --------------------------------------------------------------- M2 (sweep G)
def test_the_conditional_forecast_steps_cost_is_documented_and_bounded():
    """Round 14, sweep G: `steps` is guarded by a MEMORY budget while the work
    is quadratic, so `steps=200_000` (well inside the budget) did not finish
    inside the sweep's 60 s deadline."""
    for text in (tsecon.var_conditional_forecast.__doc__, _stub_doc("var_conditional_forecast")):
        flat = re.sub(r"\s+", " ", text)
        assert "QUADRATIC in `steps`" in flat, flat[:400]
    card = (CARDS / "var-svar.md").read_text(encoding="utf-8")
    assert "That budget is on **memory**, not time" in card
    data = _var3()
    cond = [[None, 0.5, None]]
    t0 = time.perf_counter()
    r = tsecon.var_conditional_forecast(data, cond, lags=2, steps=2000)
    dt = time.perf_counter() - t0
    assert r["steps"] == 2000 and len(r["point"]) == 2000
    assert dt < 5.0, dt               # measured 0.12 s
    with pytest.raises(ValueError, match="steps"):
        tsecon.var_conditional_forecast(data, cond, lags=2, steps=2**31)


# --------------------------------------------------------------- L8 (sweep C)
def test_the_unobserved_components_card_example_prints_what_it_claims():
    """Round 14, sweep C: the card's example wrapped its numbers in
    `np.float64(...)` under NumPy 2, so the "expected output" comment was not
    what the committed code produced."""
    pytest.importorskip("statsmodels.api")
    card = (CARDS / "unobserved-components.md").read_text(encoding="utf-8")
    block = next(b for b in re.findall(r"```python\n(.*?)```", card, re.S)
                 if "unobserved_components(nile" in b)
    claimed = [l[2:] for l in block.splitlines() if l.startswith("# {")]
    assert claimed, block
    out = subprocess.run([sys.executable, "-c", block], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-600:]
    printed = out.stdout.splitlines()
    for line in claimed:
        assert line in printed, (line, printed)


# --------------------------------------------------------------- sweep H
def test_the_seed_contract_of_the_new_surface():
    """`ets_fit`/`auto_ets` take a `None` seed that means 0 and SAY so (round
    13's M3 lesson); the three multiple-comparison tests take a plain integer
    and refuse `None` by name."""
    y = _ets_series()
    a = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=4,
                       horizon=4, n_sim=200, seed=None)
    b = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=4,
                       horizon=4, n_sim=200, seed=0)
    c = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=4,
                       horizon=4, n_sim=200, seed=1)
    assert a["seed"] == 0
    np.testing.assert_array_equal(a["forecast_variance"], b["forecast_variance"])
    assert not np.array_equal(a["forecast_variance"], c["forecast_variance"])
    for text in (tsecon.ets_fit.__doc__, tsecon.auto_ets.__doc__):
        assert "`seed` (None: 0" in re.sub(r"\s+", " ", text)
    rng = np.random.default_rng(0)
    losses = (rng.standard_normal((200, 4)) * np.array([1.0, 1.01, 1.02, 1.03])) ** 2
    with pytest.raises(TypeError, match="seed"):
        tsecon.model_confidence_set(losses, reps=50, seed=None)
    with pytest.raises(TypeError, match="seed"):
        tsecon.spa_test(losses[:, 0], losses[:, 1:], reps=50, seed=None)
    with pytest.raises(TypeError, match="seed"):
        tsecon.stepm_test(losses[:, 0], losses[:, 1:], reps=50, seed=None)


# --------------------------------------------------------------- sweep F
def test_no_default_renders_as_ellipsis_anywhere_in_the_surface():
    """The hygiene slice took eleven Ellipsis defaults to zero; round 14
    re-checks the whole surface, the thirteen new callables included."""
    offenders = []
    for name in dir(tsecon):
        if name.startswith("_"):
            continue
        fn = getattr(tsecon, name)
        if not callable(fn):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        offenders += [f"{name}.{p.name}" for p in sig.parameters.values()
                      if p.default is Ellipsis]
    assert offenders == [], offenders
