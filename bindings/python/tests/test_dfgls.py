"""Python-surface tests for the DF-GLS unit-root test.

Re-pins the crate's golden fixture (fixtures/dfgls.json, generated from
arch 8.0.0) end to end through the compiled binding, and checks the
binding contract (argument mapping, dict keys, error surface). The heavy
numerical validation lives in the crate goldens
(crates/tsecon-diag/tests/dfgls_golden.rs).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIXTURES / "dfgls.json").read_text())


def _run_case(case):
    y = np.asarray(FX["series"][case["series"]], dtype=float)
    return tsecon.dfgls(
        y,
        regression=case["trend"],
        lags=case["lags"],
        max_lags=case["max_lags"],
        method=case["method"],
    )


@pytest.mark.parametrize(
    "case",
    FX["cases"],
    ids=[
        f"{c['series']}-{c['trend']}-lags{c['lags']}-max{c['max_lags']}-{c['method']}"
        for c in FX["cases"]
    ],
)
def test_dfgls_matches_arch_fixture(case):
    got = _run_case(case)
    assert got["statistic"] == pytest.approx(case["stat"], rel=1e-10)
    assert got["p_value"] == pytest.approx(case["pvalue"], rel=1e-8, abs=1e-12)
    assert got["used_lag"] == case["lags_used"]
    assert got["nobs"] == case["nobs"]
    assert got["trend"] == case["trend"]
    for k in ("1%", "5%", "10%"):
        assert got["crit"][k] == pytest.approx(case["crit"][k], rel=1e-12)


def test_dfgls_live_parity_with_arch_if_available():
    arch_ur = pytest.importorskip("arch.unitroot")
    rng = np.random.default_rng(123)
    y = np.cumsum(rng.standard_normal(180))
    for trend in ("c", "ct"):
        got = tsecon.dfgls(y, regression=trend)
        ref = arch_ur.DFGLS(y, trend=trend)
        assert got["statistic"] == pytest.approx(ref.stat, rel=1e-10)
        assert got["p_value"] == pytest.approx(ref.pvalue, rel=1e-8, abs=1e-12)
        assert got["used_lag"] == ref.lags


def test_dfgls_contract_and_errors():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.standard_normal(100))
    r = tsecon.dfgls(y)
    assert {"statistic", "p_value", "used_lag", "nobs", "crit", "trend"} <= set(r)
    assert r["trend"] == "c"
    assert set(r["crit"]) == {"1%", "5%", "10%"}
    # nobs bookkeeping.
    assert r["nobs"] == len(y) - 1 - r["used_lag"]

    # Fixed lags override method/max_lags.
    fixed = tsecon.dfgls(y, lags=r["used_lag"], method="bic", max_lags=1)
    assert fixed["statistic"] == r["statistic"]

    # Unknown regression / method teach.
    with pytest.raises(ValueError, match="expected \"c\" or \"ct\""):
        tsecon.dfgls(y, regression="n")
    with pytest.raises(ValueError, match="aic"):
        tsecon.dfgls(y, method="nope")

    # Degenerate inputs raise instead of returning garbage.
    with pytest.raises(ValueError):
        tsecon.dfgls(np.full(50, 3.0))               # constant
    with pytest.raises(ValueError):
        tsecon.dfgls(np.arange(4, dtype=float), regression="ct")  # too short
    bad = y.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError):
        tsecon.dfgls(bad)


def test_dfgls_rejects_noise_not_walk():
    # The fixture's noise / rw0 series, whose arch verdicts are pinned in
    # the golden (a fresh random walk can reject by chance — rw2 does).
    noise = np.asarray(FX["series"]["noise"], dtype=float)
    walk = np.asarray(FX["series"]["rw0"], dtype=float)
    assert tsecon.dfgls(noise)["p_value"] < 0.01
    assert tsecon.dfgls(walk)["p_value"] > 0.10
