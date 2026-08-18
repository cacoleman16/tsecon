"""Golden and behavioral tests for the `var_backtest` binding.

The statistical heavy lifting is validated Rust-side against
fixtures/var_backtest.json (first-principles NumPy Kupiec/Christoffersen,
statsmodels-OLS DQ). Here we re-pin a fixture case through the Python
surface, recompute Kupiec from scratch with SciPy in-test, verify the
documented sign convention end to end, and check the teaching errors.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
VB = json.loads((FIX / "var_backtest.json").read_text())


def _case(group, name):
    return next(c for c in VB[group] if c["name"] == name)


def test_hit_case_matches_fixture():
    c = _case("hit_cases", "markov_n1000_s5")
    r = tsecon.var_backtest(np.array(c["hits"], float),
                            alpha=c["alpha"], dq_lags=c["dq_lags"])
    assert r["lr_uc"] == pytest.approx(c["lr_uc"], rel=1e-12)
    assert r["lr_ind"] == pytest.approx(c["lr_ind"], rel=1e-12)
    assert r["lr_cc"] == pytest.approx(c["lr_cc"], rel=1e-12)
    assert r["dq_stat"] == pytest.approx(c["dq_stat"], rel=1e-9)
    assert r["p_uc"] == pytest.approx(c["p_uc"], rel=1e-8)
    assert r["n_violations"] == c["n_violations"]
    assert (r["n00"], r["n01"], r["n10"], r["n11"]) == (
        c["n00"], c["n01"], c["n10"], c["n11"])
    assert r["dq_df"] == c["dq_df"] and not r["dq_includes_var"]
    # The clustered case is the battery's reason to exist: independence
    # and DQ reject decisively.
    assert r["p_ind"] < 0.001 and r["p_dq"] < 0.001
    assert "Reject independence" in r["verdict"]


def test_return_case_and_sign_convention():
    c = _case("return_cases", "garch_flat_var_a05")
    ret = np.array(c["returns"])
    var = np.array(c["var_forecasts"])
    r = tsecon.var_backtest(ret, var, alpha=c["alpha"], dq_lags=c["dq_lags"])
    # Sign convention: violation = return < VaR quantile (both on the
    # return scale, VaR negative here).
    assert r["n_violations"] == int((ret < var).sum()) == c["n_violations"]
    assert r["dq_stat"] == pytest.approx(c["dq_stat"], rel=1e-9)
    # This VaR path is constant (an unconditional VaR), so the documented
    # rank rule drops it from the DQ regression with honest df.
    assert r["dq_var_dropped"] and r["dq_df"] == c["dq_lags"] + 1
    # A flat VaR on heteroskedastic returns passes coverage but fails
    # the dependence tests — the textbook pattern.
    assert r["p_uc"] > 0.05 and r["p_ind"] < 0.05 and r["p_dq"] < 0.05


def test_kupiec_recomputed_from_scratch_with_scipy():
    from scipy.stats import chi2
    rng = np.random.default_rng(7)
    hits = (rng.random(600) < 0.05).astype(float)
    n, n1 = hits.size, int(hits.sum())
    n0 = n - n1
    p = n1 / n
    lr = -2 * ((n0 * np.log(0.95) + n1 * np.log(0.05))
               - (n0 * np.log(1 - p) + n1 * np.log(p)))
    r = tsecon.var_backtest(hits, alpha=0.05, dq_lags=4)
    assert r["lr_uc"] == pytest.approx(lr, rel=1e-12)
    assert r["p_uc"] == pytest.approx(chi2.sf(lr, 1), rel=1e-8)
    assert r["expected_violations"] == pytest.approx(30.0)


def test_jorion_published_example():
    # Jorion, Value at Risk ch. 6 (J.P. Morgan 1998): 20 exceptions in
    # 252 days at 95% VaR -> LR_uc = 3.91 (exact 3.9126), borderline
    # rejection vs the chi2(1) 5% critical value 3.84.
    hits = np.zeros(252)
    hits[6:240:12] = 1.0
    assert hits.sum() == 20
    r = tsecon.var_backtest(hits, alpha=0.05, dq_lags=4)
    assert r["lr_uc"] == pytest.approx(3.9125508275532184, rel=1e-12)
    assert 0.0479 < r["p_uc"] < 0.05


def test_hits_with_var_forecasts_keeps_dq_var_regressor():
    c = _case("return_cases", "garch_true_var_a05")
    ret = np.array(c["returns"])
    var = np.array(c["var_forecasts"])
    hits = (ret < var).astype(float)
    r = tsecon.var_backtest(hits, var, alpha=c["alpha"],
                            dq_lags=c["dq_lags"], input="hits")
    assert r["dq_includes_var"]
    assert r["dq_stat"] == pytest.approx(c["dq_stat"], rel=1e-9)


def test_constant_var_is_dropped_with_honest_df():
    c = _case("return_cases", "iid_true_var_a025")
    r = tsecon.var_backtest(np.array(c["returns"]),
                            np.array(c["var_forecasts"]),
                            alpha=c["alpha"], dq_lags=c["dq_lags"])
    assert r["dq_var_dropped"] and not r["dq_includes_var"]
    assert r["dq_df"] == c["dq_lags"] + 1 == c["dq_df"]
    assert r["dq_stat"] == pytest.approx(c["dq_stat"], rel=1e-9)
    assert "constant" in r["verdict"]


def test_errors_teach():
    hits = np.zeros(250)
    with pytest.raises(ValueError, match="too conservative"):
        tsecon.var_backtest(hits, alpha=0.05)          # zero violations
    with pytest.raises(ValueError, match="sign-convention"):
        tsecon.var_backtest(np.ones(250), alpha=0.05)  # all violations
    with pytest.raises(ValueError, match="alpha"):
        tsecon.var_backtest(np.r_[np.ones(5), np.zeros(95)], alpha=1.5)
    with pytest.raises(ValueError, match="exactly 0"):
        tsecon.var_backtest(np.array([0.0, 1.0, 0.3, 0.0] * 20))
    with pytest.raises(ValueError, match="index-aligned"):
        tsecon.var_backtest(np.zeros(50), np.full(49, -1.6), alpha=0.05)
    with pytest.raises(ValueError, match="var_forecasts"):
        tsecon.var_backtest(np.zeros(50), alpha=0.05, input="returns")
    with pytest.raises(ValueError, match="unknown input"):
        tsecon.var_backtest(np.zeros(50), alpha=0.05, input="losses")


def test_verdict_is_summarizable():
    # The dict flows through tsecon.summarize like every other result.
    c = _case("hand_cases", "kupiec_hand_n250_x5")
    r = tsecon.var_backtest(np.array(c["hits"], float), alpha=0.05, dq_lags=4)
    text = tsecon.summarize(r)
    assert "lr_uc" in text and "verdict" in text
    assert "too conservative" in r["verdict"]
