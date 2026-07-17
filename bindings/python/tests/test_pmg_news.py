"""Tests for the PMG panel estimator and the DFM news/update decomposition
bindings.

panel_pmg is checked tightly against the crate's documented-formula golden
(fixtures/pmg.json); dfm_news is checked by its exact adding-up identity
(the strongest possible validation, and what the crate itself pins).
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
PMG = json.loads((FIXTURES / "pmg.json").read_text())


def _pmg_units():
    ys = [np.array(u) for u in PMG["y"]]           # N response vectors
    x = PMG["x"]                                    # [K][N][T]
    n = PMG["design"]["N"]
    xs = [np.column_stack([x[0][i], x[1][i]]) for i in range(n)]
    return ys, xs


def test_pmg_matches_golden():
    ys, xs = _pmg_units()
    r = tsecon.panel_pmg(ys, xs)
    np.testing.assert_allclose(r["theta"], PMG["pmg"]["theta"], atol=1e-7)
    np.testing.assert_allclose(r["theta_se"], PMG["pmg"]["theta_se"], atol=1e-7)
    assert abs(r["phi_bar"] - PMG["pmg"]["phi_bar"]) < 1e-7
    np.testing.assert_allclose(r["phi"], PMG["pmg"]["phi"], atol=1e-7)
    assert r["n_units"] == PMG["design"]["N"]
    assert r["k"] == PMG["design"]["K"]
    # Stable long-run adjustment.
    assert r["phi_bar"] < 0


def test_pmg_pools_the_long_run_near_truth():
    ys, xs = _pmg_units()
    r = tsecon.panel_pmg(ys, xs)
    # Recovers the true common long-run coefficient theta0...
    np.testing.assert_allclose(r["theta"], PMG["theta0"], atol=0.1)
    # ...and pools far tighter than a free mean-group of per-unit long runs.
    assert np.all(np.array(r["theta_se"]) < np.array(PMG["free_mg"]["cross_unit_sd"]))


# ---------------------------------------------------- DFM news decomposition
def _factor_panel(n=140, big_n=10, seed=7):
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = (1.1 * f[t - 1] - 0.3 * f[t - 2] if t > 1 else 0.6 * f[t - 1])
        f[t] += rng.standard_normal()
    load = rng.uniform(0.6, 1.4, big_n)
    x = np.outer(f, load) + 0.5 * rng.standard_normal((n, big_n))
    return x


def test_dfm_news_adds_up_exactly():
    panel = _factor_panel()
    # Old vintage: the last row's faster series (0..4) not yet observed.
    old = panel.copy()
    old[-1, :5] = np.nan
    new = panel.copy()  # those five cells now revealed

    res = tsecon.dfm_news(old, new, target_series=0, n_factors=1, factor_order=2)
    contribs = res["contributions"]
    # Exact adding-up identity: the contributions sum to the total revision.
    total = sum(c["contribution"] for c in contribs)
    assert abs(total - res["total_revision"]) < 1e-9
    assert abs(res["total_revision"] - (res["new_nowcast"] - res["old_nowcast"])) < 1e-9
    # Each contribution is weight * news.
    for c in contribs:
        assert abs(c["contribution"] - c["weight"] * c["news"]) < 1e-9
    # The five newly-revealed cells are the source of the news.
    assert len(contribs) == 5
