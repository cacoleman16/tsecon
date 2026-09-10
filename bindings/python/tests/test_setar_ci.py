"""Golden and behavioral tests for the Hansen (1997/2000) threshold
confidence-set binding `tsecon.setar_threshold_ci`.

Re-pins fixtures/setar_ci.json through the Python surface — the closed-form
critical values and p-value function (documented-formula golden, Hansen 2000
Table 1 to the printed decimals) and the cross-implementation NumPy
transcription of the LR profile, Hansen's eta^2 convention, the interval /
hull construction and the slope unions (see the generator header for the
honest grading: no third-party threshold-CI code runs in the fixture
container) — and checks bit-identity of the fit with `tsecon.setar`,
seed-free determinism, the refusal of the inert keyword, and the teaching
errors for bad levels and nulls.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
CI = json.loads((FIX / "setar_ci.json").read_text())


def _call(case, **override):
    y = np.array(CI["series"][case["series"]])
    kw = dict(
        p=case["p"],
        delays=case["delays"],
        trim=case["trim"],
        constant=case["constant"],
        level=case["level"],
        het_robust=case["het_robust"],
        null_threshold=case["null_threshold"],
    )
    if case["slope_level"] is not None:
        kw["slope_level"] = case["slope_level"]
        kw["slope_region_level"] = case["slope_region_level"]
    kw.update(override)
    return tsecon.setar_threshold_ci(y, **kw)


def _case_id(c):
    return (
        f"{c['series']}-p{c['p']}-d{'.'.join(map(str, c['delays']))}-"
        f"lvl{c['level']}-het{int(c['het_robust'])}"
    )


# ------------------------------------------------------------ closed forms

def test_hansen_2000_table_1_through_lr_crit():
    y = np.array(CI["series"]["setar_strong"])
    for level_str, printed in CI["_meta"]["hansen_2000_table_1"].items():
        r = tsecon.setar_threshold_ci(y, p=1, level=float(level_str))
        assert round(r["lr_crit"], 2) == printed
        assert r["lr_crit"] == pytest.approx(
            -2.0 * math.log(1.0 - math.sqrt(float(level_str))), rel=1e-14
        )
        assert r["lr_crit_scaled"] == r["lr_crit"]  # eta2 == 1 exactly
        assert r["eta2"] == 1.0
    for row in CI["critical_values"]:
        r = tsecon.setar_threshold_ci(y, p=1, level=row["level"])
        assert r["lr_crit"] == pytest.approx(row["crit"], rel=1e-14)


def test_pvalue_function_is_the_documented_closed_form():
    # p(LR at the null) = 1 - (1 - exp(-LR/2))^2, exercised through the
    # null-threshold inversion on a case where the null is far from gamma^.
    case = next(c for c in CI["cases"] if c["series"] == "setar_strong"
                and c["null_threshold"] == 0.1)
    r = _call(case)
    x = r["lr_at_null"] / r["eta2"]
    assert r["pvalue_at_threshold"] == pytest.approx(
        1.0 - (1.0 - math.exp(-x / 2.0)) ** 2, abs=1e-12
    )
    assert r["pvalue_at_threshold"] == pytest.approx(case["pvalue_at_threshold"], rel=1e-10)
    # At the estimate itself LR = 0 and the p-value is exactly one.
    r0 = _call(case, null_threshold=r["threshold"])
    assert r0["lr_at_null"] == 0.0
    assert r0["pvalue_at_threshold"] == 1.0
    assert r0["null_threshold_used"] == r["threshold"]


# ---------------------------------------------------------------- golden

@pytest.mark.parametrize("case", CI["cases"], ids=_case_id)
def test_threshold_ci_matches_fixture(case):
    r = _call(case)
    assert r["threshold"] == pytest.approx(case["threshold"], rel=1e-12)
    assert r["delay"] == case["delay"]
    assert r["nobs"] == case["nobs"]
    assert r["k"] == case["k"]
    assert r["level"] == case["level"]
    assert r["het_robust"] == case["het_robust"]
    assert r["lr_crit"] == pytest.approx(case["lr_crit"], rel=1e-14)
    assert r["eta2"] == pytest.approx(case["eta2"], rel=1e-10)
    assert r["lr_crit_scaled"] == pytest.approx(case["lr_crit_scaled"], rel=1e-10)
    np.testing.assert_allclose(r["thresholds"], case["thresholds"], rtol=1e-12)
    np.testing.assert_allclose(r["ssr_path"], case["ssr_path"], rtol=1e-10)
    np.testing.assert_allclose(r["lr_stat"], case["lr_stat"], rtol=1e-10, atol=1e-10)
    assert r["in_set"].dtype == np.bool_
    assert list(r["in_set"]) == case["in_set"]
    assert r["n_intervals"] == len(case["intervals"])
    assert len(r["intervals"]) == len(case["intervals"])
    for got, want in zip(r["intervals"], case["intervals"]):
        assert got[0] == pytest.approx(want[0], rel=1e-12)
        assert got[1] == pytest.approx(want[1], rel=1e-12)
    assert r["is_connected"] == case["is_connected"]
    assert r["ci_low"] == pytest.approx(case["ci_low"], rel=1e-12)
    assert r["ci_high"] == pytest.approx(case["ci_high"], rel=1e-12)
    assert r["n_in_set"] == case["n_in_set"]
    if case["null_threshold"] is None:
        assert r["null_threshold_used"] is None
        assert r["lr_at_null"] is None
        assert r["pvalue_at_threshold"] is None
    else:
        assert r["null_threshold_used"] == pytest.approx(case["null_threshold_used"], rel=1e-12)
        assert r["lr_at_null"] == pytest.approx(case["lr_at_null"], rel=1e-10, abs=1e-10)
        assert r["pvalue_at_threshold"] == pytest.approx(case["pvalue_at_threshold"], rel=1e-10)
    if case["slope_level"] is None:
        for key in ("slope_level", "slope_region_level", "slope_region_low",
                    "slope_region_high", "slope_n_region", "slope_ci_low",
                    "slope_ci_high"):
            assert r[key] is None
    else:
        s = case["slope"]
        assert r["slope_level"] == case["slope_level"]
        assert r["slope_region_level"] == case["slope_region_level"]
        assert r["slope_region_low"] == pytest.approx(s["region_low"], rel=1e-12)
        assert r["slope_region_high"] == pytest.approx(s["region_high"], rel=1e-12)
        assert r["slope_n_region"] == s["n_region"]
        np.testing.assert_allclose(r["slope_ci_low"][0], s["low_lower"], rtol=1e-10)
        np.testing.assert_allclose(r["slope_ci_low"][1], s["high_lower"], rtol=1e-10)
        np.testing.assert_allclose(r["slope_ci_high"][0], s["low_upper"], rtol=1e-10)
        np.testing.assert_allclose(r["slope_ci_high"][1], s["high_upper"], rtol=1e-10)


def test_fixture_exercises_a_disjoint_set_and_the_correction():
    assert any(len(c["intervals"]) > 1 for c in CI["cases"])
    assert any(c["het_robust"] and c["eta2"] != 1.0 for c in CI["cases"])
    assert any(c["slope_level"] is not None for c in CI["cases"])


# ------------------------------------------------- bit-identity with setar

@pytest.mark.parametrize("case", CI["cases"], ids=_case_id)
def test_fit_is_bit_identical_to_setar(case):
    y = np.array(CI["series"][case["series"]])
    r = _call(case)
    fit = tsecon.setar(y, p=case["p"], delays=case["delays"], trim=case["trim"],
                       constant=case["constant"])
    assert r["threshold"] == fit["threshold"]
    assert r["delay"] == fit["delay"]
    assert r["nobs"] == fit["nobs"]
    assert r["k"] == fit["k"]
    assert np.array_equal(r["thresholds"], fit["thresholds"])
    assert np.array_equal(r["ssr_path"], fit["ssr_path"])
    # The estimate sits on the grid at LR exactly zero and is in the set.
    i = int(np.flatnonzero(r["thresholds"] == r["threshold"])[0])
    assert r["lr_stat"][i] == 0.0
    assert r["in_set"][i]
    assert r["ci_low"] <= r["threshold"] <= r["ci_high"]


def test_slope_union_contains_the_conventional_interval_at_the_estimate():
    from scipy.stats import norm

    y = np.array(CI["series"]["setar_strong"])
    for het in (False, True):
        r = tsecon.setar_threshold_ci(y, p=1, slope_level=0.95, het_robust=het)
        assert r["slope_region_low"] <= r["threshold"] <= r["slope_region_high"]
        assert r["slope_n_region"] >= 1
        if not het:
            # Classical per-regime SEs: the interval at gamma^ is exactly
            # setar's params +/- z * bse, which the union must contain.
            fit = tsecon.setar(y, p=1)
            z = norm.ppf(0.975)
            for reg, params, bse in ((0, fit["params_low"], fit["bse_low"]),
                                     (1, fit["params_high"], fit["bse_high"])):
                lo = np.asarray(r["slope_ci_low"][reg])
                hi = np.asarray(r["slope_ci_high"][reg])
                assert np.all(lo <= params - z * bse + 1e-12)
                assert np.all(hi >= params + z * bse - 1e-12)
        assert len(r["slope_ci_low"]) == 2 and len(r["slope_ci_low"][0]) == r["k"]


# ---------------------------------------------------------- determinism

def test_seed_free_determinism():
    y = np.array(CI["series"]["setar_het"])
    kw = dict(p=1, delays=[1, 2], trim=0.10, het_robust=True, slope_level=0.9,
              null_threshold=0.0)
    a = tsecon.setar_threshold_ci(y, **kw)
    b = tsecon.setar_threshold_ci(y, **kw)
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), key
        else:
            assert va == vb, key
    # Nesting in the level, through the surface.
    r90 = tsecon.setar_threshold_ci(y, p=1, level=0.90)
    r95 = tsecon.setar_threshold_ci(y, p=1, level=0.95)
    assert np.all(~r90["in_set"] | r95["in_set"])
    assert r95["ci_low"] <= r90["ci_low"] and r95["ci_high"] >= r90["ci_high"]


def test_delays_overrides_delay_and_accepts_plain_lists():
    y = list(CI["series"]["setar_d2"])
    r = tsecon.setar_threshold_ci(y, p=1, delay=3, delays=[1, 2])
    assert r["delay"] in (1, 2)
    fit = tsecon.setar(np.asarray(y), p=1, delays=[1, 2])
    assert r["delay"] == fit["delay"]
    assert r["threshold"] == fit["threshold"]


# ---------------------------------------------------------- refusals

def test_inert_keyword_is_refused():
    y = np.array(CI["series"]["setar_strong"])
    with pytest.raises(ValueError, match="slope_region_level"):
        tsecon.setar_threshold_ci(y, p=1, slope_region_level=0.8)
    # ... and acts when slope intervals are requested.
    r = tsecon.setar_threshold_ci(y, p=1, slope_level=0.95, slope_region_level=0.99)
    assert r["slope_region_level"] == 0.99
    r80 = tsecon.setar_threshold_ci(y, p=1, slope_level=0.95)
    assert r80["slope_region_level"] == 0.80
    assert r["slope_n_region"] >= r80["slope_n_region"]


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf")])
def test_bad_level_is_a_teaching_error(level):
    y = np.array(CI["series"]["setar_strong"])
    with pytest.raises(ValueError, match=r"level = .*0 < level < 1"):
        tsecon.setar_threshold_ci(y, p=1, level=level)
    with pytest.raises(ValueError, match=r"slope_level = .*0 < level < 1"):
        tsecon.setar_threshold_ci(y, p=1, slope_level=level)
    with pytest.raises(ValueError, match=r"slope_region_level = .*0 < level < 1"):
        tsecon.setar_threshold_ci(y, p=1, slope_level=0.95, slope_region_level=level)


def test_bad_null_threshold_and_unidentified_eta2():
    y = np.array(CI["series"]["setar_strong"])
    fit = tsecon.setar(y, p=1)
    with pytest.raises(ValueError, match="null_threshold"):
        tsecon.setar_threshold_ci(y, p=1, null_threshold=fit["thresholds"][0] - 1.0)
    with pytest.raises(ValueError, match="null_threshold"):
        tsecon.setar_threshold_ci(y, p=1, null_threshold=fit["thresholds"][-1] + 1.0)
    with pytest.raises(ValueError, match="null_threshold"):
        tsecon.setar_threshold_ci(y, p=1, null_threshold=float("nan"))
    # The grid endpoints are admissible; a value between candidates uses
    # the lower one (LR is a step function).
    g = fit["thresholds"]
    r = tsecon.setar_threshold_ci(y, p=1, null_threshold=0.5 * (g[3] + g[4]))
    assert r["null_threshold_used"] == g[3]
    # No threshold effect: eta^2 not identified, refused with the way out.
    lin = np.array(CI["series"]["linear_ar2"])
    with pytest.raises(ValueError, match=r"eta\^2.*het_robust = false"):
        tsecon.setar_threshold_ci(lin, p=2, delays=[1, 2], constant=False,
                                  level=0.80, het_robust=True)
    # The SETAR input errors pass through.
    with pytest.raises(ValueError, match="p >= 1"):
        tsecon.setar_threshold_ci(y, p=0)
    with pytest.raises(ValueError, match="trim"):
        tsecon.setar_threshold_ci(y, p=1, trim=0.5)
    with pytest.raises(ValueError, match="insufficient"):
        tsecon.setar_threshold_ci(y[:4], p=1)
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[5] = np.nan
        tsecon.setar_threshold_ci(bad, p=1)


def test_docstring_names_every_returned_key():
    import re

    y = np.array(CI["series"]["setar_strong"])
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.setar_threshold_ci.__doc__))
    for kw in ({}, {"slope_level": 0.95, "het_robust": True, "null_threshold": 0.0}):
        keys = set(tsecon.setar_threshold_ci(y, p=1, **kw).keys())
        missing = keys - tokens
        assert not missing, f"docstring does not name returned keys: {sorted(missing)}"
