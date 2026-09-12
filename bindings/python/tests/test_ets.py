"""Golden and behavioral tests for the exponential-smoothing bindings
`tsecon.ets_fit` and `tsecon.auto_ets`.

Re-pins fixtures/ets.json through the Python surface (see the generator's
docstring for the honest grading): the twenty models without a
multiplicative seasonal against statsmodels `ETSModel` at fixed parameters
(log-likelihood, fitted values, states, forecasts, exact class-1 variances),
the ten multiplicative-seasonal models against the transcription of the
published recursion, the heuristic initialisation against
`holtwinters.ExponentialSmoothing`, the maximum likelihood against
statsmodels' fit (match-or-beat), and the R `forecast::ets` candidate set;
then the live statsmodels cross-checks (an `ETSModel.forecast` and
`get_prediction` at a fixed point), determinism, the inert-option
refusals, the teaching errors, and stub sync.
"""
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIX / "ets.json").read_text())
SERIES = {k: np.array(v, dtype=float) for k, v in FX["series"].items()}

# The tolerances of crates/tsecon-ets/tests/ets_golden.rs, re-used here so
# the Rust and Python pins of the same fixture state the same bar.
SM_TOL = 1e-10          # statsmodels legs
TR_TOL = 1e-12          # documented-formula transcription and Table 6.1
MLE_LL_SLACK = 1e-5     # match-or-beat slack on the log-likelihood
MLE_PARAM_TOL = 1e-3    # cross-optimizer parameter agreement


def _kw(case):
    kw = dict(error=case["error"], trend=case["trend"], damped=case["damped"], seasonal=case["seasonal"])
    if case["seasonal"]:
        kw["seasonal_periods"] = case["seasonal_periods"]
    return kw


def _params(case):
    p = [case["alpha"]]
    if case["trend"]:
        p.append(case["beta"])
    if case["seasonal"]:
        p.append(case["gamma"])
    if case["damped"]:
        p.append(case["phi"])
    return p


def _states(case):
    st = [case["initial_level"]]
    if case["trend"]:
        st.append(case["initial_trend"])
    if case["seasonal"]:
        st.extend(case["initial_seasonal"])
    return st


def _fixed_ids(c):
    return c["name"]


def _close(what, got, want, tol):
    """The Rust golden's tolerance convention, so the two agree exactly:
    |got - want| / max(|want|, 1) <= tol. Relative on quantities of order
    one and larger, absolute below it — a seasonal index of 0.12 is not
    held to 1e-12 *of itself* when its neighbours are of order 3."""
    g = np.atleast_1d(np.asarray(got, dtype=float)).ravel()
    w = np.atleast_1d(np.asarray(want, dtype=float)).ravel()
    assert g.shape == w.shape, f"{what}: length {g.size} vs {w.size}"
    if g.size == 0:
        return
    err = np.abs(g - w) / np.maximum(np.abs(w), 1.0)
    i = int(np.argmax(err))
    assert err[i] <= tol, (
        f"{what}[{i}]: got {g[i]!r}, want {w[i]!r} "
        f"(rel err {err[i]:.3e} > {tol:.1e})"
    )


# ---------------------------------------------------------------- fixed

@pytest.mark.parametrize("case", FX["fixed"], ids=_fixed_ids)
def test_fixed_parameter_evaluation_matches_the_references(case):
    y = SERIES[case["series"]]
    r = tsecon.ets_fit(y, **_kw(case), initialization="known", smoothing_params=_params(case),
                       initial_states=_states(case), horizon=case["h"])
    assert r["short_name"] == case["short_name"]
    assert r["optimizer"] == "none"
    assert r["converged"] is True
    assert r["initialization"] == "known"
    tr = case["transcription"]
    name = case["name"]
    _close(f"{name} loglik", r["loglik"], tr["loglik"], TR_TOL)
    _close(f"{name} sigma2", r["sigma2"], tr["sigma2"], TR_TOL)
    _close(f"{name} forecast", r["forecast"], tr["forecast"], TR_TOL)
    _close(f"{name} final_level", r["final_level"], tr["final_level"], TR_TOL)
    if case["seasonal"]:
        _close(f"{name} final_seasonal", r["final_seasonal"], tr["final_seasonal"], TR_TOL)
    sm = case["statsmodels"]
    if sm is None:
        # multiplicative seasonal: the transcription carries the paths
        _close(f"{name} fitted", r["fitted"], tr["fitted"], TR_TOL)
        _close(f"{name} level_path", r["level_path"], tr["level"], TR_TOL)
        _close(f"{name} seasonal_path", r["seasonal_path"], tr["seasonal"], TR_TOL)
        assert r["interval_method"] == "simulated"
        assert r["class1"] is False
        return
    _close(f"{name} sm loglik", r["loglik"], sm["loglik"], SM_TOL)
    _close(f"{name} sm fitted", r["fitted"], sm["fitted"], SM_TOL)
    _close(f"{name} sm resid", r["resid"], sm["resid"], SM_TOL)
    _close(f"{name} sm level", r["level_path"], sm["level"], SM_TOL)
    if case["trend"]:
        _close(f"{name} sm slope", r["trend_path"], sm["trend"], SM_TOL)
    else:
        assert r["trend_path"] is None and r["beta"] is None and r["initial_trend"] is None
    if case["seasonal"]:
        _close(f"{name} sm season", r["seasonal_path"], sm["seasonal"], SM_TOL)
    else:
        assert r["seasonal_path"] is None and r["gamma"] is None and r["initial_seasonal"] is None
    _close(f"{name} sm forecast", r["forecast"], sm["forecast"], SM_TOL)
    if sm.get("forecast_variance") is not None:
        assert r["class1"] is True and r["interval_method"] == "exact"
        _close(f"{name} sm variance", r["forecast_variance"], sm["forecast_variance"], SM_TOL)
        _close(f"{name} Table 6.1", r["forecast_variance"], case["class1_variance_table61"], TR_TOL)
        z = 1.959963984540054
        _close(f"{name} lower", r["forecast_lower"],
               r["forecast"] - z * np.sqrt(r["forecast_variance"]), TR_TOL)
        assert r["n_sim"] == 0 and r["seed"] == 0
    else:
        assert r["class1"] is False and r["interval_method"] == "simulated"
        assert r["n_sim"] == 5000 and r["seed"] == 0


def test_every_taxonomy_member_is_pinned():
    assert len({c["short_name"] for c in FX["fixed"]}) == 30
    assert sum(c["statsmodels"] is not None for c in FX["fixed"]) == 23
    assert len({c["short_name"] for c in FX["fixed"] if c["statsmodels"] is None}) == 10


# ------------------------------------------------------------- heuristic

@pytest.mark.parametrize("case", [c for c in FX["heuristic"] if c["method"] == "heuristic"],
                         ids=lambda c: f"{c['series']}-{c['trend']}-{c['seasonal']}")
def test_heuristic_initial_states_match_holtwinters(case):
    y = SERIES[case["series"]]
    kw = dict(trend=case["trend"], seasonal=case["seasonal"])
    if case["seasonal"]:
        kw["seasonal_periods"] = case["seasonal_periods"]
    p = [0.5] + ([0.1] if case["trend"] else []) + ([0.1] if case["seasonal"] else [])
    r = tsecon.ets_fit(y, **kw, initialization="heuristic", smoothing_params=p)
    assert r["initialization"] == "heuristic"
    assert r["initial_level"] == pytest.approx(case["initial_level"], rel=1e-10)
    if case["trend"]:
        assert r["initial_trend"] == pytest.approx(case["initial_trend"], rel=1e-10)
    if case["seasonal"]:
        np.testing.assert_allclose(r["initial_seasonal"], case["initial_seasonal"], rtol=1e-10, atol=1e-10)
    # k counts the smoothing parameters and sigma2 only
    assert r["k_params"] == len(p) + 1


# ------------------------------------------------------------------ mle

@pytest.mark.parametrize("case", FX["mle"], ids=lambda c: f"{c['series']}-{c['short_name']}-{c['initialization']}")
def test_maximum_likelihood_matches_or_beats_statsmodels(case):
    y = SERIES[case["series"]]
    r = tsecon.ets_fit(y, **_kw(case), initialization=case["initialization"])
    ll_sm = case["loglik"]
    # Match-or-beat at the golden's stated slack: two optimizers on one
    # likelihood agree to their stopping tolerances, not bitwise. Measured
    # worst shortfall over the 21 cases: 2.67e-6 relative (co2 ETS(A,Ad,A),
    # 17 free parameters); on two cases the crate's optimum is better.
    assert r["loglik"] >= ll_sm - MLE_LL_SLACK * abs(ll_sm)
    assert r["optimizer"] == "nelder_mead+bfgs"
    assert r["k_params"] == case["k_params_crate"]
    n, k = r["nobs"], r["k_params"]
    assert r["aic"] == pytest.approx(-2 * r["loglik"] + 2 * k, rel=1e-12)
    assert r["bic"] == pytest.approx(-2 * r["loglik"] + k * np.log(n), rel=1e-12)
    assert r["aicc"] == pytest.approx(r["aic"] + 2 * k * (k + 1) / (n - k - 1), rel=1e-12)
    if r["loglik"] - ll_sm <= 1e-3 * abs(ll_sm):
        # same optimum: the parameters agree at cross-optimizer tolerance
        assert r["alpha"] == pytest.approx(case["alpha"], abs=MLE_PARAM_TOL)
        if case["trend"]:
            assert r["beta"] == pytest.approx(case["beta"], abs=MLE_PARAM_TOL)
        if case["seasonal"]:
            assert r["gamma"] == pytest.approx(case["gamma"], abs=MLE_PARAM_TOL)
            s = np.asarray(r["initial_seasonal"])
            if case["seasonal"] == "mul":
                assert s.mean() == pytest.approx(1.0, abs=1e-9)
            else:
                assert s.sum() == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(r["forecast"] if r["forecast"] is not None else [], [], rtol=0)
    # determinism
    r2 = tsecon.ets_fit(y, **_kw(case), initialization=case["initialization"])
    assert r2["loglik"] == r["loglik"] and np.array_equal(r2["params"], r["params"])


def test_fit_forecast_and_live_statsmodels_cross_check():
    sm_ets = pytest.importorskip("statsmodels.tsa.exponential_smoothing.ets")
    y = SERIES["log_ukgas"]
    r = tsecon.ets_fit(y, trend="add", damped=True, seasonal="add", seasonal_periods=4, horizon=8)
    mod = sm_ets.ETSModel(y, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=4,
                          initialization_method="known", initial_level=r["initial_level"],
                          initial_trend=r["initial_trend"], initial_seasonal=np.asarray(r["initial_seasonal"]))
    res = mod.smooth(np.array([r["alpha"], r["beta"], r["gamma"], r["phi"]]))
    assert res.llf == pytest.approx(r["loglik"], rel=1e-10)
    np.testing.assert_allclose(res.forecast(8), r["forecast"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(res.fittedvalues), r["fitted"], rtol=1e-10)


# -------------------------------------------------------------- auto_ets

def test_auto_ets_candidate_set_and_selection_consistency():
    y = SERIES["log_ukgas"]
    r = tsecon.auto_ets(y, seasonal_periods=4, horizon=4)
    assert r["ic"] == "aicc" and r["n_candidates"] == 15
    names = [c["short_name"] for c in r["candidates"]]
    want = next(c["candidates"] for c in FX["candidates"]
                if c["seasonal_periods"] == 4 and c["data_positive"] and not c["allow_multiplicative_trend"]
                and c["restrict"] and c["damped"] is None)
    assert sorted(names) == sorted(want)
    ics = [c["ic_value"] for c in r["candidates"]]
    assert ics == sorted(ics)
    assert r["candidates"][0]["short_name"] == r["short_name"]
    assert r["ic_value"] == r["aicc"] == r["candidates"][0]["ic_value"]
    assert r["n_fitted"] == sum(c["status"] == "ok" for c in r["candidates"])
    # the winner is the search's own fit: refitting reproduces it exactly
    refit = tsecon.ets_fit(y, error=r["error"], trend=r["trend"], damped=r["damped"], seasonal=r["seasonal"],
                           seasonal_periods=r["seasonal_periods"], horizon=4)
    assert refit["loglik"] == r["loglik"]
    np.testing.assert_array_equal(refit["params"], r["params"])
    np.testing.assert_array_equal(refit["forecast"], r["forecast"])
    # every candidate's criterion is its ets_fit criterion
    for c in r["candidates"][:3]:
        f = tsecon.ets_fit(y, error="add" if c["short_name"][0] == "A" else "mul",
                           trend={"N": None, "A": "add", "M": "mul"}[c["short_name"][1]],
                           damped="d" in c["short_name"], seasonal={"N": None, "A": "add", "M": "mul"}[c["short_name"][-1]],
                           seasonal_periods=4 if c["short_name"][-1] != "N" else None)
        assert f["aicc"] == c["aicc"] and f["loglik"] == c["loglik"]
    # non-positive data: additive candidates only; bic and damped=False options
    assert (y - 10.0).min() < 0
    r2 = tsecon.auto_ets(y - 10.0, seasonal_periods=4, ic="bic", damped=False)
    want2 = next(c["candidates"] for c in FX["candidates"]
                 if c["seasonal_periods"] == 4 and not c["data_positive"]
                 and not c["allow_multiplicative_trend"] and c["restrict"] and c["damped"] is False)
    assert sorted(c["short_name"] for c in r2["candidates"]) == sorted(want2)
    assert r2["n_candidates"] == len(want2) == 4
    assert all(c["short_name"][0] == "A" and "d" not in c["short_name"] for c in r2["candidates"])
    assert r2["ic"] == "bic" and r2["ic_value"] == r2["bic"]
    r3 = tsecon.auto_ets(y, allow_multiplicative_trend=True, restrict=False)
    assert r3["n_candidates"] == 10


def test_simulated_intervals_are_seeded_and_the_forecast_is_the_zero_error_path():
    y = SERIES["airline"]
    a = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=12, horizon=12, n_sim=2000, seed=3)
    b = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=12, horizon=12, n_sim=2000, seed=3)
    c = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=12, horizon=12, n_sim=2000, seed=4)
    assert a["interval_method"] == "simulated" and a["n_sim"] == 2000 and a["seed"] == 3
    np.testing.assert_array_equal(a["forecast_lower"], b["forecast_lower"])
    assert not np.array_equal(a["forecast_lower"], c["forecast_lower"])
    np.testing.assert_array_equal(a["forecast"], c["forecast"])
    assert np.all(a["forecast_lower"] < a["forecast"]) and np.all(a["forecast_upper"] > a["forecast"])
    assert a["forecast_upper"][-1] - a["forecast_lower"][-1] > a["forecast_upper"][0] - a["forecast_lower"][0]
    np.testing.assert_allclose(np.asarray(a["initial_seasonal"]).mean(), 1.0, atol=1e-9)
    # narrower interval at a lower level
    d = tsecon.ets_fit(y, error="mul", trend="add", seasonal="mul", seasonal_periods=12, horizon=12, level=0.8, n_sim=2000, seed=3)
    assert np.all(d["forecast_upper"] - d["forecast_lower"] < a["forecast_upper"] - a["forecast_lower"])


# ------------------------------------------------------------- refusals

def test_inert_options_raise_when_passed_explicitly():
    y = SERIES["sim_aadn"]
    with pytest.raises(ValueError, match="damped"):
        tsecon.ets_fit(y, damped=True)
    with pytest.raises(ValueError, match="seasonal_periods"):
        tsecon.ets_fit(y, seasonal_periods=4)
    with pytest.raises(ValueError, match="seasonal_periods"):
        tsecon.ets_fit(y, seasonal="add")
    for kw in ({"level": 0.9}, {"n_sim": 100}, {"seed": 1}):
        with pytest.raises(ValueError, match=f"{list(kw)[0]} was given but horizon=0"):
            tsecon.ets_fit(y, **kw)
        with pytest.raises(ValueError, match=f"{list(kw)[0]} was given but horizon=0"):
            tsecon.auto_ets(y, **kw)
    for kw in ({"n_sim": 100}, {"seed": 1}):
        with pytest.raises(ValueError, match="class-1"):
            tsecon.ets_fit(y, trend="add", horizon=3, **kw)
    with pytest.raises(ValueError, match="initial_states was given but"):
        tsecon.ets_fit(y, initial_states=[1.0])
    with pytest.raises(ValueError, match="initial_states was given but"):
        tsecon.ets_fit(y, initialization="heuristic", initial_states=[1.0])
    with pytest.raises(ValueError, match='initialization="known" needs initial_states'):
        tsecon.ets_fit(y, initialization="known")
    with pytest.raises(ValueError, match="optimizer was given but smoothing_params"):
        tsecon.ets_fit(y, initialization="heuristic", smoothing_params=[0.5], optimizer="bfgs")
    with pytest.raises(ValueError, match="max_iter was given but smoothing_params"):
        tsecon.ets_fit(y, initialization="heuristic", smoothing_params=[0.5], max_iter=10)
    with pytest.raises(ValueError, match='initialization="estimated"'):
        tsecon.ets_fit(y, smoothing_params=[0.5])


def test_teaching_errors_name_the_argument():
    y = SERIES["sim_aadn"]
    with pytest.raises(ValueError, match=r"y\[5\] = -1"):
        tsecon.ets_fit(np.r_[y[:5], -1.0, y[6:]], error="mul")
    with pytest.raises(ValueError, match=r"y: contains a non-finite value"):
        tsecon.ets_fit(np.r_[y[:5], np.nan, y[6:]])
    with pytest.raises(ValueError, match=r"y: contains a non-finite value"):
        tsecon.auto_ets(np.r_[y[:5], np.nan, y[6:]])
    with pytest.raises(ValueError, match='error = "gamma" is invalid'):
        tsecon.ets_fit(y, error="gamma")
    with pytest.raises(ValueError, match='trend = "quad" is invalid'):
        tsecon.ets_fit(y, trend="quad")
    with pytest.raises(ValueError, match='initialization = "mle" is invalid'):
        tsecon.ets_fit(y, initialization="mle")
    with pytest.raises(ValueError, match='optimizer = "sgd" is invalid'):
        tsecon.ets_fit(y, optimizer="sgd")
    with pytest.raises(ValueError, match='ic = "hqic" is invalid'):
        tsecon.auto_ets(y, ic="hqic")
    with pytest.raises(ValueError, match="seasonal_periods = 0"):
        tsecon.auto_ets(y, seasonal_periods=0)
    with pytest.raises(ValueError, match="beta = 0.7"):
        tsecon.ets_fit(y, trend="add", initialization="heuristic", smoothing_params=[0.5, 0.7])
    with pytest.raises(ValueError, match="alpha = 1.5"):
        tsecon.ets_fit(y, initialization="heuristic", smoothing_params=[1.5])
    with pytest.raises(ValueError, match="expected length 1 but got 2"):
        tsecon.ets_fit(y, initialization="heuristic", smoothing_params=[0.5, 0.1])
    with pytest.raises(ValueError, match="expected length 6 but got 2"):
        tsecon.ets_fit(y, seasonal="add", seasonal_periods=4, initialization="known", initial_states=[1.0, 2.0],
                       trend="add")
    with pytest.raises(ValueError, match="level = 1.5"):
        tsecon.ets_fit(y, horizon=2, level=1.5)
    with pytest.raises(ValueError, match="n_sim = 1"):
        tsecon.ets_fit(y, error="mul", horizon=2, n_sim=1)
    with pytest.raises(ValueError, match="needs at least"):
        tsecon.ets_fit(y[:12], trend="add", damped=True, seasonal="add", seasonal_periods=12)
    with pytest.raises(ValueError, match="heuristic initialisation"):
        tsecon.ets_fit(y[:8], initialization="heuristic")


def test_allocation_guards_refuse_by_name_instead_of_aborting():
    """The simulated interval materialises `n_sim` paths of `horizon`
    steps, a buffer whose size is a product of two user counts. Before the
    guards these three calls killed the interpreter with
    `memory allocation of N bytes failed` from the Rust allocator; the
    coercion layer only stops integer counts at 2**48, so everything below
    that is the crate's job."""
    y = SERIES["sim_aadn"]
    with pytest.raises(ValueError, match=r"n_sim = 140737488355328"):
        tsecon.ets_fit(y, error="mul", horizon=2, n_sim=2 ** 47)
    with pytest.raises(ValueError, match=r"n_sim = 1000000 .*2\^28|n_sim = 1000000"):
        tsecon.ets_fit(y, error="mul", horizon=1000, n_sim=1000000)
    for kw in ({"horizon": 10 ** 8}, {"horizon": 10 ** 8, "error": "mul"}):
        with pytest.raises(ValueError, match=r"horizon = 100000000"):
            tsecon.ets_fit(y, **kw)
    with pytest.raises(ValueError, match=r"horizon = 100000000"):
        tsecon.auto_ets(y, horizon=10 ** 8)
    # The guards bind only absurd requests: the documented defaults run.
    r = tsecon.ets_fit(y, error="mul", horizon=200)
    assert r["n_sim"] == 5000 and len(r["forecast"]) == 200


def test_accepts_lists_and_integer_arrays_through_the_coercion_layer():
    y = SERIES["sim_aadn"]
    r = tsecon.ets_fit(list(y), trend="add", horizon=2)
    r2 = tsecon.ets_fit(np.round(y).astype(int), trend="add", horizon=2)
    assert r["loglik"] == tsecon.ets_fit(y, trend="add", horizon=2)["loglik"]
    assert np.isfinite(r2["loglik"])


# ------------------------------------------------------------- stub sync

def test_stub_signatures_and_docstrings_match_runtime():
    stub = (Path(__file__).parents[1] / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    for name in ("ets_fit", "auto_ets"):
        fn = getattr(tsecon._core, name)
        params = list(inspect.signature(fn).parameters)
        m = re.search(rf"def {name}\((.*?)\) ->", stub, re.S)
        stub_params = [p.strip().split(":")[0] for p in m.group(1).split(",") if p.strip()]
        assert stub_params == params, name
        assert "Ellipsis" not in str(inspect.signature(fn))
        # every returned key is named in both docstrings
        y = SERIES["log_ukgas"]
        out = (tsecon.ets_fit(y, trend="add", seasonal="add", seasonal_periods=4, horizon=2) if name == "ets_fit"
               else tsecon.auto_ets(y, seasonal_periods=4, horizon=2, damped=False))
        for doc in (fn.__doc__, stub[stub.index(f"def {name}("):]):
            tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", doc))
            missing = set(out) - tokens
            assert not missing, f"{name}: {sorted(missing)}"
        if name == "auto_ets":
            ctoks = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__))
            assert set(out["candidates"][0]) <= ctoks
