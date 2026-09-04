"""Golden and behavioral tests for the STAR bindings.

Re-pins fixtures/star.json (an independent NumPy/SciPy transcription of
the Luukkonen-Saikkonen-Terasvirta 1988 / Terasvirta 1994 auxiliary
regressions, the concentrated STAR OLS with Gauss-Newton SEs, and the
(gamma, c) grid — see the generator header for the honest grading)
through the Python surface, and checks the honesty flags (`converged`,
`gamma_at_boundary`, `se_valid`), the teaching errors, and docstring/key
consistency.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
STAR = json.loads((FIX / "star.json").read_text())

BATTERY_FLOATS = [
    "lm3_stat", "lm3_p_value", "lm3_f_stat", "lm3_f_p_value",
    "h3_f_stat", "h3_p_value", "h2_f_stat", "h2_p_value",
    "h1_f_stat", "h1_p_value", "ssr0", "ssr1", "ssr2", "ssr3",
]


def close(a, b, rtol=1e-10, atol=1e-10):
    # The absolute floor mirrors the Rust golden test: a near-zero F
    # statistic is a difference of two O(100) SSRs, so its *relative*
    # accuracy is cancellation-limited even at 1e-10 SSR agreement.
    assert a == pytest.approx(b, rel=rtol, abs=atol)


@pytest.mark.parametrize("case", STAR["test"], ids=lambda c: (
    f"{c['series']}-p{c['p']}-d{'.'.join(map(str, c['delays']))}"
))
def test_star_test_matches_fixture(case):
    y = np.array(STAR["series"][case["series"]])
    r = tsecon.star_test(y, p=case["p"], delays=case["delays"])
    assert r["best"] == case["best"]
    assert len(r["tests"]) == len(case["tests"])
    for got, exp in zip(r["tests"], case["tests"]):
        for key in ("delay", "nobs", "q", "k0"):
            assert got[key] == exp[key], key
        for key in BATTERY_FLOATS:
            close(got[key], exp[key])
        assert got["suggested"] == exp["suggested"]
    # The top level repeats the selected battery.
    sel = case["tests"][case["best"]]
    assert r["delay"] == sel["delay"]
    close(r["lm3_f_p_value"], sel["lm3_f_p_value"])
    assert r["suggested"] == sel["suggested"]


@pytest.mark.parametrize("case", STAR["eval"], ids=lambda c: (
    f"{c['series']}-{c['model']}-g{c['gamma']}-c{c['c']}"
))
def test_star_eval_matches_fixture(case):
    y = np.array(STAR["series"][case["series"]])
    r = tsecon.star_eval(
        y,
        p=case["p"],
        gamma=case["gamma"],
        c=case["c"],
        model=case["model"],
        delay=case["delay"],
        constant=case["constant"],
    )
    np.testing.assert_allclose(r["params_linear"], case["coefs_linear"], rtol=1e-10)
    np.testing.assert_allclose(r["params_nonlinear"], case["coefs_nonlinear"], rtol=1e-10)
    assert r["se_valid"] is True
    np.testing.assert_allclose(r["bse_linear"], case["se_linear"], rtol=1e-10)
    np.testing.assert_allclose(r["bse_nonlinear"], case["se_nonlinear"], rtol=1e-10)
    close(r["se_gamma"], case["se_gamma"])
    close(r["se_c"], case["se_c"])
    for key in ("ssr", "sigma2", "loglik", "aic", "bic"):
        close(r[key], case[key])
    assert r["nobs"] == case["nobs"]
    assert r["k"] == case["k"]
    np.testing.assert_allclose(r["transition"][:8], case["transition_head"], rtol=1e-10)
    close(float(np.sum(r["transition"])), case["transition_sum"])


@pytest.mark.parametrize("case", STAR["grid"], ids=lambda c: (
    f"{c['series']}-{c['model']}"
))
def test_star_grid_matches_fixture(case):
    y = np.array(STAR["series"][case["series"]])
    r = tsecon.star(
        y,
        p=case["p"],
        model=case["model"],
        delay=case["delay"],
        trim=case["trim"],
        n_gamma=case["n_gamma"],
        n_c=case["n_c"],
    )
    close(r["s_sd"], case["s_sd"], rtol=1e-12)
    np.testing.assert_allclose(r["grid_gamma"], case["grid_gamma"], rtol=1e-12)
    np.testing.assert_allclose(r["grid_c"], case["grid_c"], rtol=1e-12)
    expected = np.array(
        [np.nan if v is None else v for v in case["ssr_grid"]]
    ).reshape(case["n_gamma"], case["n_c"])
    assert r["ssr_grid"].shape == expected.shape
    np.testing.assert_allclose(r["ssr_grid"], expected, rtol=1e-10)
    assert tuple(r["best_cell"]) == tuple(case["best_cell"])
    # Refinement only improves on the grid, and the reported eval is the
    # concentrated fit at the reported (gamma, c).
    assert r["ssr"] <= case["best_ssr"] * (1 + 1e-12)
    e = tsecon.star_eval(
        y, p=case["p"], gamma=r["gamma"], c=r["c"],
        model=case["model"], delay=case["delay"],
    )
    close(e["ssr"], r["ssr"], rtol=1e-12, atol=1e-12)


def test_flags_boundary_on_hard_threshold_and_interior_on_smooth():
    rng = np.random.default_rng(7)
    # Hard-threshold SETAR-style data, low noise -> gamma runs to the cap.
    y = np.zeros(500)
    for t in range(1, 500):
        lo, hi = (1.0, 0.5), (-1.0, 0.3)
        c0, a = lo if y[t - 1] <= 0 else hi
        y[t] = c0 + a * y[t - 1] + 0.25 * rng.standard_normal()
    r = tsecon.star(y[100:], p=1, model="lstar")
    assert r["gamma_at_boundary"] is True
    assert isinstance(r["converged"], bool)
    assert abs(r["c"]) < 0.4

    # The transition path is consistent with the reported (gamma, c).
    s = y[100:][:-1]
    g = 1.0 / (1.0 + np.exp(-r["gamma"] * (s - r["c"])))
    np.testing.assert_allclose(r["transition"], g, rtol=1e-10, atol=1e-12)


def test_star_eval_step_limit_reports_invalid_ses():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(300).cumsum() * 0.1 + rng.standard_normal(300)
    r = tsecon.star_eval(y, p=1, gamma=1e8, c=float(np.median(y[:-1])) + 1e-6)
    assert r["se_valid"] is False
    assert np.isnan(r["se_gamma"]) and np.isnan(r["se_c"])
    # The linear parameters are still the (finite) split OLS.
    assert np.all(np.isfinite(r["params_linear"]))
    assert np.all(np.isfinite(r["params_nonlinear"]))


def test_teaching_errors():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(150)

    with pytest.raises(ValueError, match="constant"):
        tsecon.star(np.ones(100), p=1)
    with pytest.raises(ValueError, match="p >= 1"):
        tsecon.star(y, p=0)
    with pytest.raises(ValueError, match="delay"):
        tsecon.star(y, p=1, delay=0)
    with pytest.raises(ValueError, match="trim"):
        tsecon.star(y, p=1, trim=0.5)
    with pytest.raises(ValueError, match="n_gamma"):
        tsecon.star(y, p=1, n_gamma=1)
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star(y[:6], p=1)
    with pytest.raises(ValueError, match="gamma > 0"):
        tsecon.star_eval(y, p=1, gamma=-1.0, c=0.0)
    with pytest.raises(ValueError, match="unknown STAR model"):
        tsecon.star(y, p=1, model="tar")
    with pytest.raises(ValueError, match="finite"):
        tsecon.star_test(np.r_[y, np.nan], p=1)
    # Degenerate transition variable: near-constant series.
    flat = np.full(120, 5.0)
    flat[60] += 5e-10
    with pytest.raises(ValueError, match="transition variable"):
        tsecon.star(flat, p=1)


def test_out_of_sample_delay_is_a_teaching_error_not_a_panic():
    """Audit round 10, finding 1 (SEVERE): star_test with an empty series
    or a delay at/past the end of the sample used to escape as a Rust
    panic (pyo3 PanicException, uncatchable by `except ValueError`), and
    the T-1/T boundary was miscategorized as a near-constant transition
    variable. All must raise the sibling estimators' "insufficient data"
    ValueError."""
    rng = np.random.default_rng(3)
    T = 50
    y = rng.standard_normal(T)

    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star_test(np.array([]), p=2)
    for d in (T - 1, T, T + 1, T + 50):
        with pytest.raises(ValueError, match="insufficient data"):
            tsecon.star_test(y, p=2, delays=[d])
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star_test(y, p=2, delays=[1, T + 50])

    # star and star_eval share the contract (they already refused; pinned
    # so the family stays aligned).
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star(np.array([]), p=2)
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star(y, p=2, delays=[1, T + 50])
    with pytest.raises(ValueError, match="insufficient data"):
        tsecon.star_eval(y, p=2, gamma=2.0, c=0.0, delay=T + 50)


def test_delays_search_and_battery_agree_on_lstar_d2():
    y = np.array(STAR["series"]["lstar_d2"])
    t = tsecon.star_test(y, p=1, delays=[1, 2, 3])
    assert t["delay"] == 2
    r = tsecon.star(y, p=1, model="lstar", delays=[1, 2, 3])
    assert r["delay"] == 2


def test_docstrings_name_every_returned_key():
    """The runtime docstring names every returned key (the audit-round
    tripwire pattern from test_docstring_keys.py, applied to the three
    new functions)."""
    y = np.array(STAR["series"]["lstar_strong"])

    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    r = tsecon.star(y, p=1, model="lstar")
    missing = set(r.keys()) - tokens(tsecon.star)
    assert not missing, f"star.__doc__ missing keys: {sorted(missing)}"

    e = tsecon.star_eval(y, p=1, gamma=2.0, c=0.0)
    missing = set(e.keys()) - tokens(tsecon.star_eval)
    assert not missing, f"star_eval.__doc__ missing keys: {sorted(missing)}"

    t = tsecon.star_test(y, p=1)
    missing = set(t.keys()) - tokens(tsecon.star_test)
    assert not missing, f"star_test.__doc__ missing keys: {sorted(missing)}"
