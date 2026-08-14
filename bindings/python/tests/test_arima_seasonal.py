"""Seasonal ARIMA (SARIMA) through the Python surface.

Golden values come from fixtures/sarima.json (statsmodels SARIMAX with
simple_differencing=True; see fixtures/generate_sarima_fixtures.py).
The heavy numeric gates (1e-8 fixed-parameter log-likelihoods, bse
parity, levels-forecast parity) live in the Rust suite
(crates/tsecon-arima/tests/seasonal_golden.rs); these tests pin the
Python wiring: argument parsing, naming, output shape, and the
fit-level statsmodels parity a user would check first.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
SARIMA = json.loads((FIXTURES / "sarima.json").read_text())


# Box-Jenkins Series G, embedded once for this test module (public
# data; identical to the generator's copy).
AIRLINE = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
]


def test_airline_model_matches_statsmodels():
    fx = SARIMA["airline_011_011_12"]
    y = np.log(np.asarray(AIRLINE, dtype=float))
    r = tsecon.arima_fit(
        y, p=0, d=1, q=1, seasonal=(0, 1, 1, 12), constant=False, forecast_steps=24
    )
    # Match or beat the (polished) statsmodels maximum, and land on the
    # textbook airline parameters.
    assert r["loglik"] >= fx["fit_loglike"] - 1e-6 * abs(fx["fit_loglike"])
    np.testing.assert_allclose(r["params"], fx["fit_params"], rtol=5e-3)
    assert list(r["param_names"]) == ["ma.L1", "ma.S.L12", "sigma2"]
    assert r["bse"] is not None and np.all(np.isfinite(r["bse"]))
    assert len(r["forecast_mean"]) == 24
    # 13 observations lost to differencing: 144 - 1 - 12 = 131 residuals.
    assert len(r["residuals"]) == 131


def test_seasonal_accepts_list_and_none():
    y = np.log(np.asarray(AIRLINE, dtype=float))
    fit_tuple = tsecon.arima_fit(y, p=0, d=1, q=1, seasonal=(0, 1, 1, 12), constant=False)
    fit_list = tsecon.arima_fit(y, p=0, d=1, q=1, seasonal=[0, 1, 1, 12], constant=False)
    np.testing.assert_array_equal(fit_tuple["params"], fit_list["params"])

    rng = np.random.default_rng(7)
    z = rng.standard_normal(150)
    plain = tsecon.arima_fit(z, p=1, d=0, q=0, constant=False)
    with_none = tsecon.arima_fit(z, p=1, d=0, q=0, seasonal=None, constant=False)
    # All-zero seasonal orders are the non-seasonal model at any period.
    with_zeros = tsecon.arima_fit(z, p=1, d=0, q=0, seasonal=(0, 0, 0, 12), constant=False)
    np.testing.assert_array_equal(plain["params"], with_none["params"])
    np.testing.assert_array_equal(plain["params"], with_zeros["params"])


def test_seasonal_argument_errors_teach():
    y = np.log(np.asarray(AIRLINE, dtype=float))
    with pytest.raises(ValueError, match=r"\(P, D, Q, s\)"):
        tsecon.arima_fit(y, seasonal=(0, 1, 1))
    with pytest.raises(ValueError, match="period s >= 2"):
        tsecon.arima_fit(y, seasonal=(0, 1, 1, 1))
    with pytest.raises(ValueError, match=r"non-negative"):
        tsecon.arima_fit(y, seasonal=(0, 1, -1, 12))
    # A non-sequence-of-ints is rejected at the boundary as a TypeError
    # (PyO3's conversion error). The message text varies with the Python
    # version — 3.9's abi3 build says "Can't extract `str` to `Vec`"
    # while newer Pythons prefix the argument name — so only the
    # exception type is pinned.
    with pytest.raises(TypeError):
        tsecon.arima_fit(y, seasonal="monthly")


def test_seasonal_random_walk_forecast_law():
    s = 12
    rng = np.random.default_rng(11)
    n = 120
    y = np.zeros(n)
    for t in range(n):
        y[t] = (y[t - s] if t >= s else 0.0) + rng.standard_normal()
    r = tsecon.arima_fit(
        y, p=0, d=0, q=0, seasonal=(0, 1, 0, s), constant=False, forecast_steps=2 * s
    )
    # Point forecast repeats the last season; se = sigma * sqrt(ceil(h/s)).
    np.testing.assert_allclose(r["forecast_mean"][:s], y[-s:], rtol=1e-10)
    np.testing.assert_allclose(r["forecast_mean"][s:], r["forecast_mean"][:s], rtol=1e-10)
    sigma = np.sqrt(r["params"][-1])
    h = np.arange(1, 2 * s + 1)
    np.testing.assert_allclose(
        r["forecast_se"], sigma * np.sqrt(np.ceil(h / s)), rtol=1e-8
    )


def test_quarterly_sar_matches_statsmodels():
    fx = SARIMA["quarterly_sar_c"]
    y = np.asarray(fx["y"])
    r = tsecon.arima_fit(y, p=1, d=0, q=0, seasonal=(1, 0, 0, 4), constant=True)
    assert r["loglik"] >= fx["fit_loglike"] - 1e-6 * abs(fx["fit_loglike"])
    np.testing.assert_allclose(r["params"], fx["fit_params"], rtol=5e-3)
    assert list(r["param_names"]) == ["const", "ar.L1", "ar.S.L4", "sigma2"]
