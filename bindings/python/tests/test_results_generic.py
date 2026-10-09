"""The generic `summarize` / `Result` renderer for any estimator output."""
import json
import pickle

import numpy as np
import pytest
import tsecon
from tsecon.results import Result, Results, VARResults, summarize


def _y(n=120, seed=0):
    return np.cumsum(np.random.default_rng(seed).standard_normal(n))


def _data(n=150, k=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, k))


def test_summarize_wraps_a_plain_dict():
    r = summarize(tsecon.adf(_y()), title="adf")
    assert isinstance(r, Result)
    text = r.summary()
    assert "generic Result" in text
    # every key is faithfully present, nothing invented or hidden
    assert set(r) == set(tsecon.adf(_y()))
    for key in ("statistic", "p_value", "nobs", "crit"):
        assert key in text


def test_summarize_renders_scalars_matrices_and_nested_dicts():
    r = summarize(tsecon.var_fit(_data(), lags=2), title="var_fit")
    text = r.summary()
    assert "is_stable" in text                 # scalar
    assert "params" in text and "sigma_u" in text  # matrices, shaped tables
    r2 = summarize(tsecon.adf(_y()))
    assert "crit  (dict, 3 entries)" in r2.summary()   # nested dict one level


def test_dict_contract_preserved():
    d = tsecon.adf(_y())
    r = summarize(d)
    assert isinstance(r, dict)
    assert json.loads(json.dumps(r)) == {k: v for k, v in d.items()}
    back = pickle.loads(pickle.dumps(r))
    assert dict(back) == dict(r) and back._title == r._title
    assert r["p_value"] == d["p_value"]
    assert {**r} == {**d}


def test_bespoke_object_passes_through_unchanged():
    vr = VARResults.fit(_data(), lags=2)
    assert summarize(vr) is vr                 # keep the good bespoke summary
    assert isinstance(vr, Results)


def test_summarize_is_idempotent():
    r = summarize(tsecon.adf(_y()))
    assert summarize(r) is r


def test_wrap_generic_forces_structural_dump():
    vr = VARResults.fit(_data(), lags=2)
    forced = summarize(vr, wrap="generic")
    assert isinstance(forced, Result)
    assert "generic Result" in forced.summary()


def test_summarize_handles_nondict_returns():
    # var_irf returns a nested list, not a dict
    r = summarize(tsecon.var_irf(_data(), lags=2, horizon=4), title="irf")
    assert isinstance(r, Result) and "value" in r


def test_large_arrays_summarized_not_dumped():
    # a long vector must be summarized by shape/range, not printed in full
    r = summarize({"resid": np.arange(500.0)}, title="big")
    text = r.summary()
    assert "(500,)" in text and "min" in text and "max" in text
    assert text.count("\n") < 15   # not a 500-line dump


def test_repr_is_the_summary():
    r = summarize(tsecon.adf(_y()), title="adf")
    assert repr(r) == r.summary()


# ------------------------------------------- labels: coefficient vectors only
def _line(text, key):
    return next(ln for ln in text.splitlines() if ln.startswith(f"{key}  "))


def test_only_a_paired_coefficient_vector_takes_the_name_list():
    r = summarize({"param_names": ["a", "b"], "params": np.array([1.0, 2.0]),
                   "other": np.array([3.0, 4.0]), "names": ["p", "q"]})
    text = r.summary()
    assert _line(text, "params") == "params  (2,)  [a=1.000000, b=2.000000]"
    assert "=" not in _line(text, "other")


def test_ardl_bounds_are_not_labelled_with_uecm_names():
    """Audit round 15: the basic ARDL(1,1) at the default case has exactly
    four UECM parameters, so the four bound levels used to read
    `crit_levels (4,) [const=0.9, y.L1=0.95, ...]`."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
    y = 0.5 * x[:, 0] + rng.standard_normal(200)
    out = tsecon.ardl_bounds_test(y, x, lags=1, order=1)
    assert len(out["param_names"]) == 4 == len(out["crit_levels"])
    text = summarize(out).summary()
    for key in ("crit_levels", "crit_lower", "crit_upper"):
        assert "=" not in _line(text, key), _line(text, key)
    assert "const=" in _line(text, "params") and "y.L1=" in _line(text, "se")
    for key in ("long_run", "long_run_se"):
        assert all(f"{n}=" in _line(text, key) for n in out["long_run_names"]), key


def test_fama_macbeth_period_flags_are_not_labelled_when_t_equals_k():
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((2, 20, 3))
    out = tsecon.panel_fama_macbeth(xs[0] + rng.standard_normal((20, 3)), xs)
    assert len(out["names"]) == len(out["periods_used"]) == 3
    text = summarize(out).summary()
    assert "=" not in _line(text, "periods_used")
    assert "const=" in _line(text, "params") and "x0=" in _line(text, "bse")


def test_uc_forecasts_are_not_labelled_with_parameter_names():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.standard_normal(150)) + rng.standard_normal(150)
    out = tsecon.unobserved_components(y, level="llevel", autoregressive=1, forecast_steps=4)
    assert len(out["param_names"]) == 4 == len(out["forecast"])
    text = summarize(out).summary()
    for key in ("forecast", "forecast_var"):
        assert "=" not in _line(text, key), _line(text, key)
    assert all(f"{n}=" in _line(text, "params") for n in out["param_names"])


def test_per_coefficient_flags_and_states_keep_their_labels():
    """The explicit pairing must not drop a label length-matching got right:
    garch/arima's per-parameter `se_valid`/`boundary` flags, tvp_regression's
    per-coefficient `sigma2_beta`/`pile_up` (named by `coef_names`), and the
    ETS `final_states` (the `initial_states` layout)."""
    rng = np.random.default_rng(0)
    g = tsecon.garch_fit(rng.standard_normal(600) * 0.01)
    a = tsecon.arima_fit(np.cumsum(rng.standard_normal(200)), p=1, d=1, q=1, forecast_steps=4)
    for out in (g, a):
        text = summarize(out).summary()
        for key in ("se_valid", "boundary"):
            assert all(f"{n}=" in _line(text, key) for n in out["param_names"]), _line(text, key)
    # arima's four forecasts beside four parameters stay unlabelled
    assert len(a["param_names"]) == len(a["forecast_mean"]) == 4
    assert "=" not in _line(summarize(a).summary(), "forecast_mean")
    t = tsecon.tvp_regression(rng.standard_normal(150), rng.standard_normal((150, 2)))
    text = summarize(t).summary()
    for key in ("sigma2_beta", "pile_up"):
        assert all(f"{n}=" in _line(text, key) for n in t["coef_names"]), _line(text, key)
    y = np.abs(rng.standard_normal(80)) + 10 + np.sin(np.arange(80) * 2 * np.pi / 4)
    e = tsecon.ets_fit(y, trend="add", seasonal="add", seasonal_periods=4)
    line = _line(summarize(e).summary(), "final_states")
    assert line.startswith("final_states  (6,)  [level=") and "seasonal[3]=" in line, line
