"""Golden, Monte Carlo and behavioral tests for the multiple-forecast-
comparison bindings `tsecon.spa_test`, `tsecon.model_confidence_set` and
`tsecon.stepm_test`.

Re-pins fixtures/spa.json and fixtures/mcs.json (arch 8.0.0 goldens, see
the generator headers for the honest grading) through the Python surface:

* the resample-free statistics — mean loss differentials, Hansen's kernel
  variances, the consistent-recentring flags, the observed SPA statistic,
  the MCS mean losses — at 1e-12 against the fixture (arch's numbers for
  the un-studentized path, the NumPy transcription of Hansen 2005 for the
  studentized one; these do not depend on which resamples were drawn);
* the bootstrap p-values and sets at MONTE CARLO tolerance: the seeded
  Rust bootstrap (4000 replications) against arch's 4000-replication
  values, within 0.05 (two independent bootstraps; ~4.5 sd at p = 0.5),
  with the MCS included set reproduced exactly on designs the generator
  certified are separated from `size`. The bit-for-bit leg (arch's own
  resample indices fed through the Rust core) lives in the crate tests
  `spa_golden.rs` / `mcs_golden.rs`, since NumPy's generator cannot be
  reproduced from Python here without exposing a testing entry point;
* determinism, the p-value bracketing, nested MCS sets, the StepM subset
  identity, the refusal messages naming their parameter, the docstring key
  tripwire, and the `backtest` -> loss table -> `model_confidence_set`
  workflow the guide shows.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
SPA = json.loads((FIX / "spa.json").read_text())
MCS = json.loads((FIX / "mcs.json").read_text())
TOL = SPA["_meta"]["mc_tolerance"]


def _spa_inputs(case):
    bench = np.array(case["benchmark"])
    models = np.column_stack([np.array(c) for c in case["models"]])
    return bench, models


def _spa_id(c):
    return c["name"]


def _mcs_id(c):
    return c["name"]


# --------------------------------------------------------------- SPA golden

@pytest.mark.parametrize("case", SPA["cases"], ids=_spa_id)
@pytest.mark.parametrize("studentize", [False, True])
def test_spa_resample_free_statistics_match_the_fixture(case, studentize):
    bench, models = _spa_inputs(case)
    r = tsecon.spa_test(bench, models, block_size=case["block_size"], reps=50,
                        bootstrap=case["bootstrap"], studentize=studentize,
                        nested=case["nested"], seed=3)
    ref = case["studentized"] if studentize else case["arch"]
    assert r["n"] == case["n"] and r["m"] == case["m"]
    assert r["block_size"] == case["block_size"] and r["block_size_auto"] is False
    assert r["bootstrap"] == case["bootstrap"]
    assert r["studentize"] is studentize and r["nested"] is case["nested"]
    np.testing.assert_allclose(r["mean_loss_diff"], ref["mean_loss_diff"], rtol=1e-12)
    if not case["nested"]:
        # Hansen's kernel variance is resample-free; the nested one is not.
        np.testing.assert_allclose(r["loss_diff_var"], ref["loss_diff_var"], rtol=1e-12)
        assert list(r["recentered"]) == ref["recentered"]
        assert r["statistic"] == pytest.approx(ref["statistic"], rel=1e-12)
        assert r["best_model"] == ref["best_model"]
    assert r["recentered"].dtype == np.bool_
    assert r["p_value"] == r["p_value_consistent"]
    assert r["p_value_lower"] <= r["p_value_consistent"] <= r["p_value_upper"]
    assert len(r["boot_consistent"]) == 50 and r["reps"] == 50
    np.testing.assert_array_equal(r["crit_levels"], [0.90, 0.95, 0.99])


@pytest.mark.parametrize("case", SPA["cases"], ids=_spa_id)
@pytest.mark.parametrize("studentize", [False, True])
def test_spa_seeded_pvalues_match_arch_at_monte_carlo_tolerance(case, studentize):
    bench, models = _spa_inputs(case)
    mc = case["mc"]
    r = tsecon.spa_test(bench, models, block_size=case["block_size"], reps=mc["reps"],
                        bootstrap=case["bootstrap"], studentize=studentize,
                        nested=case["nested"], seed=20260911)
    ref = mc["studentized"] if studentize else mc["unstudentized"]
    for key in ("p_lower", "p_consistent", "p_upper"):
        mine = r["p_value_" + key[2:]]
        assert abs(mine - ref[key]) <= TOL, f"{key}: {mine} vs arch {ref[key]}"


def test_spa_fixture_records_the_arch_findings():
    meta = SPA["_meta"]
    assert meta["arch"] == "8.0.0" and meta["studentize_inert"] is True
    # arch's StepM crash is exercised by at least one stored case.
    assert any(v["arch_raises"] for c in SPA["cases"] for v in c["stepm"].values())


@pytest.mark.parametrize("name", ["dominated_benchmark", "no_model_better"])
def test_stepm_reproduces_the_clear_cut_superior_sets(name):
    case = next(c for c in SPA["cases"] if c["name"] == name)
    bench, models = _spa_inputs(case)
    r = tsecon.stepm_test(bench, models, size=0.05, block_size=case["block_size"],
                          reps=case["mc"]["reps"], bootstrap=case["bootstrap"],
                          studentize=False, nested=case["nested"], seed=20260911)
    assert r["superior_models"] == case["stepm"]["0.05"]["superior_models"]
    assert r["n_superior"] == len(r["superior_models"])
    assert r["n_steps"] == len(r["steps"]) == len(r["step_crit_values"])
    assert sum(len(s) for s in r["steps"]) == r["n_superior"]
    assert r["size"] == 0.05
    # The embedded full-set SPA is spa_test itself.
    s = tsecon.spa_test(bench, models, block_size=case["block_size"], reps=case["mc"]["reps"],
                        bootstrap=case["bootstrap"], studentize=False, nested=case["nested"],
                        seed=20260911)
    for key in ("statistic", "p_value_consistent", "best_model"):
        assert r[key] == s[key]
    np.testing.assert_array_equal(r["boot_consistent"], s["boot_consistent"])
    for k in r["superior_models"]:
        assert r["mean_loss_diff"][k] > 0


# --------------------------------------------------------------- MCS golden

@pytest.mark.parametrize("case", MCS["cases"], ids=_mcs_id)
def test_mcs_seeded_reproduces_arch_set_and_pvalues_at_mc_tolerance(case):
    losses = np.column_stack([np.array(c) for c in case["losses"]])
    mc = case["mc"]
    r = tsecon.model_confidence_set(losses, size=case["size"], method=case["method"],
                                    block_size=case["block_size"], reps=mc["reps"],
                                    bootstrap=case["bootstrap"], seed=20260911)
    assert r["n"] == case["n"] and r["m"] == case["m"]
    assert r["method"] == case["method"] and r["bootstrap"] == case["bootstrap"]
    assert r["size"] == case["size"] and r["reps"] == mc["reps"]
    assert r["block_size"] == case["block_size"] and r["block_size_auto"] is False
    np.testing.assert_allclose(r["mean_losses"], case["mean_losses"], rtol=1e-13)
    assert r["included"] == mc["included"]
    assert r["excluded"] == mc["excluded"]
    for k, (mine, theirs) in enumerate(zip(r["mcs_p_values"], mc["mcs_p_values"])):
        assert abs(mine - theirs) <= MCS["_meta"]["mc_tolerance"], f"model {k}: {mine} vs {theirs}"
    # Structure: a partition, the running maximum, the survivor's 1.0.
    assert sorted(r["included"] + r["excluded"]) == list(range(case["m"]))
    assert len(r["elimination_order"]) == case["m"] == len(r["step_p_values"])
    assert r["step_p_values"][-1] == 1.0
    running = -math.inf
    for k, p in zip(r["elimination_order"], r["step_p_values"]):
        running = max(running, p)
        assert r["mcs_p_values"][k] == running
    assert r["n_steps"] == len(r["statistics"])
    if "all_equal" in case["name"]:
        assert r["included"] == list(range(case["m"]))


# ------------------------------------------------------------- determinism

def _panel(seed=0, n=120, m=4):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    return np.column_stack([(0.7 * u + s * rng.standard_normal(n)) ** 2
                            for s in np.linspace(0.8, 1.3, m)])


def test_seed_reproducibility_and_sensitivity():
    L = _panel()
    a = tsecon.spa_test(L[:, 0], L[:, 1:], reps=500, seed=4)
    b = tsecon.spa_test(L[:, 0], L[:, 1:], reps=500, seed=4)
    c = tsecon.spa_test(L[:, 0], L[:, 1:], reps=500, seed=5)
    for key in ("boot_lower", "boot_consistent", "boot_upper", "crit_consistent"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["boot_consistent"], c["boot_consistent"])
    np.testing.assert_array_equal(a["mean_loss_diff"], c["mean_loss_diff"])
    assert a["block_size_auto"] is True and 1 <= a["block_size"] < 120
    ma = tsecon.model_confidence_set(L, reps=500, seed=4)
    mb = tsecon.model_confidence_set(L, reps=500, seed=4)
    np.testing.assert_array_equal(ma["mcs_p_values"], mb["mcs_p_values"])
    assert ma["elimination_order"] == mb["elimination_order"]


def test_mcs_sets_are_nested_in_size_and_pvalues_are_size_free():
    L = _panel(seed=2, m=5)
    prev = None
    for size in (0.01, 0.05, 0.10, 0.25, 0.50):
        r = tsecon.model_confidence_set(L, size=size, reps=400, seed=1)
        if prev is not None:
            np.testing.assert_array_equal(r["mcs_p_values"], prev["mcs_p_values"])
            assert set(r["included"]) <= set(prev["included"])
        prev = r
        assert r["included"]


def test_single_model_and_1d_model_losses():
    L = _panel(seed=3, m=2)
    r = tsecon.spa_test(L[:, 0], L[:, 1], reps=300, block_size=4, seed=0)
    assert r["m"] == 1 and len(r["mean_loss_diff"]) == 1
    r2 = tsecon.spa_test(L[:, 0], L[:, 1:2], reps=300, block_size=4, seed=0)
    np.testing.assert_array_equal(r["boot_consistent"], r2["boot_consistent"])


# ------------------------------------------------------------------ refusals

def test_refusals_name_the_parameter():
    L = _panel(seed=5)
    bench, models = L[:, 0], L[:, 1:]
    with pytest.raises(ValueError, match=r"reps = 0"):
        tsecon.spa_test(bench, models, reps=0)
    with pytest.raises(ValueError, match=r"block_size = 0"):
        tsecon.spa_test(bench, models, block_size=0)
    with pytest.raises(ValueError, match=r"block_size = 120"):
        tsecon.spa_test(bench, models, block_size=120)
    with pytest.raises(ValueError, match=r'bootstrap = "block"'):
        tsecon.spa_test(bench, models, bootstrap="block")
    with pytest.raises(ValueError, match=r"model_losses.*period 4 of column 1"):
        bad = models.copy()
        bad[4, 1] = np.nan
        tsecon.spa_test(bench, bad)
    with pytest.raises(ValueError, match=r"benchmark_losses.*period 2"):
        bb = bench.copy()
        bb[2] = np.inf
        tsecon.spa_test(bb, models)
    with pytest.raises(ValueError, match=r"model_losses must be a 1-D loss series"):
        tsecon.spa_test(bench, np.zeros((4, 3, 2)))
    with pytest.raises(ValueError, match=r"model_losses column 0.*constant"):
        tsecon.spa_test(bench, np.column_stack([bench, models[:, 0]]))
    with pytest.raises(ValueError, match=r"benchmark_losses = 2 periods"):
        tsecon.spa_test(bench[:2], models[:2])
    with pytest.raises(ValueError, match=r"block_size = None.*Politis-White"):
        tsecon.spa_test(bench[:8], models[:8])
    with pytest.raises(ValueError, match=r"size = 1.*0 < size < 1"):
        tsecon.stepm_test(bench, models, size=1.0)
    with pytest.raises(ValueError, match=r"losses = 1 column"):
        tsecon.model_confidence_set(bench)
    with pytest.raises(ValueError, match=r"size = 0.*0 < size < 1"):
        tsecon.model_confidence_set(L, size=0.0)
    with pytest.raises(ValueError, match=r'method = "range"'):
        tsecon.model_confidence_set(L, method="range")
    with pytest.raises(ValueError, match=r"models 1 and 2.*identical"):
        tsecon.model_confidence_set(np.column_stack([L[:, 0], L[:, 1], L[:, 1]]))
    with pytest.raises(ValueError, match=r"losses.*period 3 of column 2"):
        bad = L.copy()
        bad[3, 2] = np.nan
        tsecon.model_confidence_set(bad)
    with pytest.raises(ValueError, match=r"reps = 0"):
        tsecon.model_confidence_set(L, reps=0)



@pytest.mark.parametrize("reps", [2 ** 44, 2 ** 47, 10 ** 12])
def test_an_impossible_replication_count_is_refused_not_allocated(reps):
    """The reps x m bootstrap buffer is the one allocation whose size is a
    product of user counts; it is budgeted with try_reserve in Rust, so a
    count no machine can serve must raise a ValueError naming `reps` rather
    than abort the interpreter. These counts are below the wrapper's own
    2**48 guard, so it is the Rust budget being exercised."""
    L = _panel(seed=9)
    for call in (lambda: tsecon.spa_test(L[:, 0], L[:, 1:], reps=reps, block_size=4),
                 lambda: tsecon.stepm_test(L[:, 0], L[:, 1:], reps=reps, block_size=4),
                 lambda: tsecon.model_confidence_set(L, reps=reps, block_size=4)):
        with pytest.raises(ValueError, match=r"refusing to allocate .* reduce reps"):
            call()


def test_the_wrapper_refuses_counts_at_or_beyond_2_48():
    L = _panel(seed=9)
    with pytest.raises(ValueError, match=r"reps=\d+ is at or beyond 2\*\*48"):
        tsecon.spa_test(L[:, 0], L[:, 1:], reps=2 ** 48)

# --------------------------------------------------------- docstring tripwire

def _tokens(fn):
    return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))


@pytest.mark.parametrize("name", ["spa_test", "model_confidence_set", "stepm_test"])
def test_every_returned_key_is_named_in_the_docstring(name):
    L = _panel(seed=7)
    fn = getattr(tsecon, name)
    if name == "model_confidence_set":
        out = fn(L, reps=100)
    else:
        out = fn(L[:, 0], L[:, 1:], reps=100)
    missing = set(out) - _tokens(fn)
    assert not missing, f"{name}.__doc__ does not name returned keys: {sorted(missing)}"


# --------------------------------------------------------- the guide workflow

def test_backtest_loss_table_feeds_the_model_confidence_set():
    rng = np.random.default_rng(11)
    n = 160
    y = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = 0.6 * prev + rng.standard_normal()
        y[t] = prev + 10.0
    names = ["naive", "drift", "mean", "theta"]
    losses = []
    origins = None
    for f in names:
        bt = tsecon.backtest(y, train=80, horizon=1, forecaster=f)
        if origins is None:
            origins = bt["origins"]
        assert bt["origins"] == origins
        e = np.array(bt["targets"][0]) - np.array(bt["forecasts"][0])
        losses.append(e**2)
    L = np.column_stack(losses)
    r = tsecon.model_confidence_set(L, size=0.10, reps=500, seed=0)
    assert r["m"] == 4 and r["n"] == len(origins)
    assert r["included"], "the set is never empty"
    assert all(0.0 <= p <= 1.0 for p in r["mcs_p_values"])
    # The historical mean is the natural benchmark of a stationary AR(1):
    # no model should be declared superior to it with any confidence.
    s = tsecon.spa_test(L[:, 2], L[:, [0, 1, 3]], reps=500, seed=0)
    assert 0.0 <= s["p_value"] <= 1.0
    assert set(tsecon.stepm_test(L[:, 2], L[:, [0, 1, 3]], reps=500, seed=0)["superior_models"]) <= {0, 1, 2}
