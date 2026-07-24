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
