"""Golden and behavioral tests for the threshold-VAR bindings.

Re-pins fixtures/tvar.json (an independent NumPy transcription of the
concentrated ln-det scan and the Eicker-White score-form linearity
statistic — see the generator header for the honest grading: no
third-party TVAR was runnable in the fixture container) through the
Python surface, and checks the bootstrap test's seeded determinism and
its verdicts on threshold vs. linear data.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
TVAR = json.loads((FIX / "tvar.json").read_text())


@pytest.mark.parametrize("case", TVAR["fit"], ids=lambda c: (
    f"{c['series']}-p{c['p']}-tv{c['threshold_index']}-"
    f"d{'.'.join(map(str, c['delays']))}-t{c['trim']}"
))
def test_tvar_fit_matches_fixture(case):
    y = np.array(TVAR["series"][case["series"]])
    r = tsecon.threshold_var(
        y,
        p=case["p"],
        threshold_index=case["threshold_index"],
        delays=case["delays"],
        trim=case["trim"],
        constant=case["constant"],
    )
    assert r["threshold"] == pytest.approx(case["threshold"], rel=1e-12)
    assert r["delay"] == case["delay"]
    assert r["threshold_index"] == case["threshold_index"]
    np.testing.assert_allclose(r["params_low"], case["coefs_low"], rtol=1e-10)
    np.testing.assert_allclose(r["params_high"], case["coefs_high"], rtol=1e-10)
    np.testing.assert_allclose(r["bse_low"], case["se_low"], rtol=1e-10)
    np.testing.assert_allclose(r["bse_high"], case["se_high"], rtol=1e-10)
    assert r["n_low"] == case["n_low"]
    assert r["n_high"] == case["n_high"]
    assert r["nobs"] == case["nobs"]
    np.testing.assert_allclose(r["sigma"], case["sigma"], rtol=1e-10)
    np.testing.assert_allclose(r["sigma_low"], case["sigma_low"], rtol=1e-10)
    np.testing.assert_allclose(r["sigma_high"], case["sigma_high"], rtol=1e-10)
    assert r["log_det_sigma"] == pytest.approx(case["log_det_sigma"], rel=1e-10)
    assert r["llf"] == pytest.approx(case["llf"], rel=1e-10)
    assert r["aic"] == pytest.approx(case["aic"], rel=1e-10)
    assert r["bic"] == pytest.approx(case["bic"], rel=1e-10)
    np.testing.assert_allclose(r["thresholds"], case["thresholds"], rtol=1e-12)
    np.testing.assert_allclose(r["logdet_path"], case["logdet_path"], rtol=1e-10)
    assert r["min_regime"] == case["min_regime"]
    assert r["n_regressors"] == case["n_regressors"]
    assert r["neqs"] == y.shape[1]


@pytest.mark.parametrize("case", TVAR["test"], ids=lambda c: (
    f"{c['series']}-p{c['p']}-d{c['delay']}-g{c['n_grid']}"
))
def test_tvar_supwald_statistic_matches_fixture(case):
    y = np.array(TVAR["series"][case["series"]])
    r = tsecon.threshold_var_test(
        y, p=case["p"], threshold_index=case["threshold_index"],
        delay=case["delay"], trim=case["trim"], n_grid=case["n_grid"],
        n_boot=19, seed=0, constant=case["constant"],
    )
    assert r["stat"] == pytest.approx(case["stat"], rel=1e-10)
    assert r["threshold"] == pytest.approx(case["threshold"], rel=1e-12)
    assert r["nobs"] == case["nobs"]
    assert r["min_regime"] == case["min_regime"]
    assert r["n_regressors"] == case["n_regressors"]
    np.testing.assert_allclose(r["thresholds"], case["thresholds"], rtol=1e-12)
    np.testing.assert_allclose(r["wald_path"], case["wald_path"], rtol=1e-10)
    assert len(r["boot_stats"]) == 19
    assert 0.0 < r["p_value"] <= 1.0


def test_delay_search_picks_the_true_delay():
    y = np.array(TVAR["series"]["tvar_d2"])
    r = tsecon.threshold_var(y, p=1, delays=[1, 2, 3], trim=0.15)
    assert r["delay"] == 2
    # `delays` overrides `delay`; a single delay via either spelling agrees.
    a = tsecon.threshold_var(y, p=1, delay=2, trim=0.15)
    b = tsecon.threshold_var(y, p=1, delays=[2], trim=0.15)
    assert a["threshold"] == b["threshold"]


def test_bootstrap_is_seed_deterministic_and_verdicts_split():
    strong = np.array(TVAR["series"]["tvar_strong"])
    linear = np.array(TVAR["series"]["tvar_linear"])
    a = tsecon.threshold_var_test(strong, p=1, n_grid=50, n_boot=199, seed=42)
    b = tsecon.threshold_var_test(strong, p=1, n_grid=50, n_boot=199, seed=42)
    np.testing.assert_array_equal(a["boot_stats"], b["boot_stats"])
    assert a["p_value"] == b["p_value"]
    assert a["p_value"] <= 0.01
    c = tsecon.threshold_var_test(linear, p=1, n_grid=50, n_boot=199, seed=42)
    assert c["p_value"] > 0.10


def test_refit_criterion_is_scan_minimum_and_split_is_consistent():
    y = np.array(TVAR["series"]["tvar_strong"])
    r = tsecon.threshold_var(y, p=1)
    assert r["log_det_sigma"] == pytest.approx(min(r["logdet_path"]), rel=1e-10)
    assert r["n_low"] + r["n_high"] == r["nobs"]
    assert min(r["n_low"], r["n_high"]) >= r["min_regime"]
    # Pooled sigma is the regime-size-weighted mix.
    mix = (np.array(r["sigma_low"]) * r["n_low"]
           + np.array(r["sigma_high"]) * r["n_high"]) / r["nobs"]
    np.testing.assert_allclose(np.array(r["sigma"]), mix, rtol=1e-12)


def test_pandas_input_accepted():
    pd = pytest.importorskip("pandas")
    y = np.array(TVAR["series"]["tvar_strong"])
    df = pd.DataFrame(y, columns=["output", "rate"])
    a = tsecon.threshold_var(df, p=1)
    b = tsecon.threshold_var(y, p=1)
    assert a["threshold"] == b["threshold"]
    np.testing.assert_array_equal(a["params_low"], b["params_low"])


def test_teaching_errors():
    y = np.array(TVAR["series"]["tvar_strong"])
    with pytest.raises(ValueError, match="at least two series"):
        tsecon.threshold_var(y[:, :1].reshape(-1, 1), p=1)
    with pytest.raises(ValueError, match="p >= 1"):
        tsecon.threshold_var(y, p=0)
    with pytest.raises(ValueError, match="threshold_index"):
        tsecon.threshold_var(y, p=1, threshold_index=2)
    with pytest.raises(ValueError, match="trim"):
        tsecon.threshold_var(y, p=1, trim=0.5)
    with pytest.raises(ValueError, match="delay"):
        tsecon.threshold_var(y, p=1, delay=0)
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[3, 1] = np.inf
        tsecon.threshold_var(bad, p=1)
    with pytest.raises(ValueError, match="insufficient"):
        tsecon.threshold_var(y[:8], p=1)
    with pytest.raises(ValueError, match="n_boot"):
        tsecon.threshold_var_test(y, p=1, n_boot=0)
    with pytest.raises(ValueError, match="n_grid"):
        tsecon.threshold_var_test(y, p=1, n_grid=1)


def test_docstrings_name_every_returned_key():
    import re

    y = np.array(TVAR["series"]["tvar_strong"])
    for fn, res in [
        (tsecon.threshold_var, tsecon.threshold_var(y, p=1)),
        (tsecon.threshold_var_test,
         tsecon.threshold_var_test(y, p=1, n_grid=40, n_boot=9)),
    ]:
        tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__))
        missing = set(res.keys()) - tokens
        assert not missing, f"{fn.__name__} docstring misses {sorted(missing)}"
