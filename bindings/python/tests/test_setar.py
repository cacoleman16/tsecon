"""Golden and behavioral tests for the SETAR bindings.

Re-pins fixtures/setar.json (an independent NumPy transcription of Tong-Lim
1980 concentrated LS / Hansen 1996-1997 sup-F — see the generator header
for the honest grading) through the Python surface, and checks the
bootstrap test's seeded determinism and its verdicts on threshold vs.
linear data.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
SETAR = json.loads((FIX / "setar.json").read_text())


@pytest.mark.parametrize("case", SETAR["fit"], ids=lambda c: (
    f"{c['series']}-p{c['p']}-d{'.'.join(map(str, c['delays']))}-t{c['trim']}"
))
def test_setar_fit_matches_fixture(case):
    y = np.array(SETAR["series"][case["series"]])
    r = tsecon.setar(
        y,
        p=case["p"],
        delays=case["delays"],
        trim=case["trim"],
        constant=case["constant"],
    )
    assert r["threshold"] == pytest.approx(case["threshold"], rel=1e-12)
    assert r["delay"] == case["delay"]
    np.testing.assert_allclose(r["params_low"], case["coefs_low"], rtol=1e-10)
    np.testing.assert_allclose(r["params_high"], case["coefs_high"], rtol=1e-10)
    np.testing.assert_allclose(r["bse_low"], case["se_low"], rtol=1e-10)
    np.testing.assert_allclose(r["bse_high"], case["se_high"], rtol=1e-10)
    assert r["n_low"] == case["n_low"]
    assert r["n_high"] == case["n_high"]
    assert r["nobs"] == case["nobs"]
    assert r["min_regime"] == case["min_regime"]
    assert r["k"] == case["k"]
    assert r["ssr"] == pytest.approx(case["ssr"], rel=1e-10)
    assert r["sigma2"] == pytest.approx(case["sigma2"], rel=1e-10)
    assert r["sigma2_low"] == pytest.approx(case["sigma2_low"], rel=1e-10)
    assert r["sigma2_high"] == pytest.approx(case["sigma2_high"], rel=1e-10)
    assert r["aic"] == pytest.approx(case["aic"], rel=1e-10)
    assert r["bic"] == pytest.approx(case["bic"], rel=1e-10)
    np.testing.assert_allclose(r["thresholds"], case["thresholds"], rtol=1e-12)
    np.testing.assert_allclose(r["ssr_path"], case["ssr_path"], rtol=1e-10)


@pytest.mark.parametrize("case", SETAR["test"], ids=lambda c: (
    f"{c['series']}-p{c['p']}-d{c['delay']}-t{c['trim']}"
))
def test_setar_supf_statistic_matches_fixture(case):
    y = np.array(SETAR["series"][case["series"]])
    r = tsecon.setar_test(
        y, p=case["p"], delay=case["delay"], trim=case["trim"], n_boot=19, seed=0
    )
    assert r["stat"] == pytest.approx(case["stat"], rel=1e-10)
    assert r["threshold"] == pytest.approx(case["threshold"], rel=1e-12)
    assert r["nobs"] == case["nobs"]
    assert r["ssr_linear"] == pytest.approx(case["ssr_linear"], rel=1e-10)
    assert r["ssr_setar"] == pytest.approx(case["ssr_setar"], rel=1e-10)
    np.testing.assert_allclose(r["f_path"], case["f_path"], rtol=1e-10)
    assert len(r["boot_stats"]) == 19
    assert 0.0 < r["p_value"] <= 1.0


def test_ic_selects_the_reported_criterion():
    y = np.array(SETAR["series"]["setar_strong"])
    a = tsecon.setar(y, p=1, ic="aic")
    b = tsecon.setar(y, p=1, ic="bic")
    assert a["ic"] == a["aic"] and a["ic_used"] == "aic"
    assert b["ic"] == b["bic"] and b["ic_used"] == "bic"
    # ic only changes what is reported, never the fit.
    assert a["threshold"] == b["threshold"]
    with pytest.raises(ValueError, match="ic"):
        tsecon.setar(y, p=1, ic="hqic")


def test_bootstrap_is_seeded_and_deterministic():
    y = np.array(SETAR["series"]["linear_ar1"])
    r1 = tsecon.setar_test(y, p=1, n_boot=199, seed=42)
    r2 = tsecon.setar_test(y, p=1, n_boot=199, seed=42)
    np.testing.assert_array_equal(r1["boot_stats"], r2["boot_stats"])
    assert r1["p_value"] == r2["p_value"]
    # p-values live on the bootstrap lattice {1/200, ..., 200/200}.
    assert abs(r1["p_value"] * 200 - round(r1["p_value"] * 200)) < 1e-12


def test_verdicts_threshold_vs_linear():
    strong = np.array(SETAR["series"]["setar_strong"])
    linear = np.array(SETAR["series"]["linear_ar1"])
    assert tsecon.setar_test(strong, p=1, n_boot=199, seed=1)["p_value"] <= 0.01
    assert tsecon.setar_test(linear, p=1, n_boot=199, seed=1)["p_value"] > 0.05


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError, match="constant"):
        tsecon.setar(np.ones(100), p=1)
    with pytest.raises(ValueError, match="trim"):
        tsecon.setar(np.array(SETAR["series"]["linear_ar1"]), p=1, trim=0.6)
    with pytest.raises(ValueError, match="delay"):
        tsecon.setar(np.array(SETAR["series"]["linear_ar1"]), p=1, delay=0)
    with pytest.raises(ValueError, match="insufficient"):
        tsecon.setar(np.array([0.1, -0.2, 0.3, 0.05]), p=1)
    with pytest.raises(ValueError, match="n_boot"):
        tsecon.setar_test(np.array(SETAR["series"]["linear_ar1"]), p=1, n_boot=0)
