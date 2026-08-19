"""Python-level tests for proxy_first_stage and the `first_stage` diagnostics
stamped into proxy_svar.

The regression algebra and the noncentral chi-square critical values are
golden-pinned in the Rust crate suite against statsmodels and scipy
(crates/tsecon-ident/tests/first_stage.rs). These tests pin what the Python
caller can be silently wrong about:

* the golden fixture reproduces END TO END through the binding path — a
  lags=0, trend="c" VAR's residuals are the demeaned data, and the first
  stage demeans over the overlap anyway, so tsecon.proxy_first_stage on the
  fixture's raw residual matrix must reproduce statsmodels' numbers;
* proxy_svar's `first_stage` dict and proxy_first_stage agree, and both agree
  with the scalar `first_stage_f` (the HC1 effective F equivalence);
* the stamped MOP critical values are the published weakivtest table;
* the audit-shape guards fire (hac_lags under a non-HAC variance, unknown
  variance) instead of silently ignoring an argument;
* weak/strong DGPs land on the right side of the tau=10% bar, and the
  verdict fields are consistent with the stamped thresholds.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"

EXPECTED_KEYS = {
    "beta",
    "se",
    "effective_f",
    "f_classical",
    "f_hc1",
    "reliability",
    "n_proxy",
    "hac_lags",
    "mop_cv_tau5",
    "mop_cv_tau10",
    "mop_cv_tau20",
    "mop_cv_tau30",
    "tau_bound",
    "weak_mop_tau10",
    "weak_folklore",
}


def _fixture():
    with open(FIXTURES / "proxy_first_stage.json", encoding="utf-8") as f:
        return json.load(f)


def _proxy_var(n=400, seed=3, strength=1.0, noise=0.7):
    """Stable VAR(1) DGP plus a proxy for structural shock 0."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    h = np.array([[1.0, 0.4, 0.2], [0.5, 1.2, 0.3], [0.3, 0.5, 0.9]])
    eps = rng.standard_normal((n, 3))
    u = eps @ h.T
    y = np.zeros((n, 3))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + u[t]
    proxy = strength * eps[:, 0] + noise * rng.standard_normal(n)
    return y, proxy


def test_golden_fixture_reproduces_through_the_binding():
    """lags=0 + trend='c' residuals are the demeaned data, and the first stage
    demeans over the overlap again, so the binding must reproduce the
    statsmodels-pinned fixture numbers on the raw residual matrices."""
    fx = _fixture()
    for case in fx["cases"]:
        u = np.asarray(case["u"])
        proxy = np.array(
            [np.nan if v is None else v for v in case["proxy"]], dtype=float
        )
        kwargs = {"lags": 0, "norm_var": case["norm_var"]}
        if case["hac_lags"] is not None:
            kwargs.update(variance="hac", hac_lags=case["hac_lags"])
        d = tsecon.proxy_first_stage(u, proxy, **kwargs)
        e = case["expected"]
        assert set(d) == EXPECTED_KEYS, case["name"]
        for key in ("beta", "se", "effective_f", "f_classical", "f_hc1", "reliability"):
            assert d[key] == pytest.approx(e[key], rel=1e-9), f"{case['name']}.{key}"
        assert d["n_proxy"] == e["n_proxy"]
        if e["tau_bound"] is None:
            assert np.isinf(d["tau_bound"])
        else:
            assert d["tau_bound"] == pytest.approx(e["tau_bound"], rel=1e-5)


def test_mop_critical_values_are_the_published_table():
    _, proxy = _proxy_var()
    y, _ = _proxy_var()
    d = tsecon.proxy_first_stage(y, proxy, lags=2)
    # weakivtest single-instrument table (tau=30% differs in the 3rd decimal
    # because weakivtest rounds 1/tau to 3.33; tsecon uses the exact value).
    assert d["mop_cv_tau5"] == pytest.approx(37.418, abs=5e-3)
    assert d["mop_cv_tau10"] == pytest.approx(23.109, abs=5e-3)
    assert d["mop_cv_tau20"] == pytest.approx(15.062, abs=5e-3)
    assert d["mop_cv_tau30"] == pytest.approx(12.046, abs=5e-3)


def test_proxy_svar_stamps_the_same_diagnostics():
    y, proxy = _proxy_var()
    proxy[:60] = np.nan
    pr = tsecon.proxy_svar(y, proxy, lags=2, horizon=8, norm_var=0)
    fs = tsecon.proxy_first_stage(y, proxy, lags=2, norm_var=0)
    assert set(pr["first_stage"]) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert pr["first_stage"][key] == fs[key], key
    # The scalar first_stage_f IS the HC1 effective F (robust_f=True default).
    assert pr["first_stage_f"] == fs["effective_f"] == fs["f_hc1"]


def test_strong_and_weak_land_on_the_right_side_of_the_bar():
    y, strong = _proxy_var(strength=1.0, noise=0.5)
    d = tsecon.proxy_first_stage(y, strong, lags=2)
    assert not d["weak_mop_tau10"] and not d["weak_folklore"]
    assert d["effective_f"] > d["mop_cv_tau10"]
    assert d["tau_bound"] < 0.10

    y2, weak = _proxy_var(strength=0.05, noise=1.0)
    w = tsecon.proxy_first_stage(y2, weak, lags=2)
    assert w["weak_mop_tau10"]
    assert w["effective_f"] < d["effective_f"]
    # Verdicts are literally consistent with the stamped thresholds.
    assert w["weak_mop_tau10"] == (w["effective_f"] <= w["mop_cv_tau10"])
    assert w["weak_folklore"] == (w["effective_f"] < 10.0)


def test_variance_choices_and_guards():
    y, proxy = _proxy_var()
    hc1 = tsecon.proxy_first_stage(y, proxy, lags=2)
    cl = tsecon.proxy_first_stage(y, proxy, lags=2, variance="classical")
    hac = tsecon.proxy_first_stage(y, proxy, lags=2, variance="hac", hac_lags=6)
    assert cl["effective_f"] == cl["f_classical"]
    assert hc1["effective_f"] == hc1["f_hc1"]
    assert hac["hac_lags"] == 6 and hc1["hac_lags"] is None
    # hac without hac_lags falls back to the Newey-West rule and still runs.
    nw = tsecon.proxy_first_stage(y, proxy, lags=2, variance="hac")
    assert nw["hac_lags"] is not None and nw["hac_lags"] > 0
    # hac_lags under a non-HAC variance is an error, not a silent no-op.
    with pytest.raises(Exception, match="hac_lags"):
        tsecon.proxy_first_stage(y, proxy, lags=2, variance="hc1", hac_lags=3)
    with pytest.raises(Exception, match="variance"):
        tsecon.proxy_first_stage(y, proxy, lags=2, variance="hc9")


def test_nan_proxy_dropped_by_date_not_compacted():
    """Prepending NaNs must not shift the finite observations against the
    residual rows: the diagnostics on the masked series equal those computed
    on the truncated-sample equivalents only if alignment is by date."""
    y, proxy = _proxy_var()
    masked = proxy.copy()
    masked[:100] = np.nan
    d = tsecon.proxy_first_stage(y, masked, lags=2)
    # Residual sample is n-2 rows; the first 98 of them carry a NaN proxy
    # (100 masked observations minus the 2 presample rows).
    assert d["n_proxy"] == (len(proxy) - 2) - 98
    # Shifting the mask start changes the answer (alignment is real).
    masked2 = proxy.copy()
    masked2[:150] = np.nan
    d2 = tsecon.proxy_first_stage(y, masked2, lags=2)
    assert d2["n_proxy"] == (len(proxy) - 2) - 148
    assert d2["effective_f"] != d["effective_f"]
