"""Golden tests for the extra realized-volatility estimators:

  * realized_quarticity / tripower_quarticity — integrated-quarticity
    estimators (Barndorff-Nielsen & Shephard 2002, 2004),
  * bns_jump_test — the studentized (RV-BV)/RV ratio jump statistic
    (BNS 2004; Huang & Tauchen 2005),
  * realized_range — Parkinson (1980) and Garman-Klass (1980) OHLC
    range variances.

The quarticity / jump tests reuse fixtures/realized.json's small return
vector (measures_small.returns) and assert against the documented
closed-form formulas; the range tests simulate OHLC bars and check the
analytic expressions directly.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
REAL = json.loads((FIXTURES / "realized.json").read_text())
RETURNS = REAL["measures_small"]["returns"]


# ------------------------------------------------ integrated quarticity
def test_realized_quarticity_matches_documented_formula():
    r = np.array(RETURNS)
    n = len(r)
    # RQ = (n/3) sum r_i^4  (BNS 2002).
    expected = n / 3.0 * float(np.sum(r**4))
    got = tsecon.realized_quarticity(r)
    assert abs(got - expected) < 1e-12
    assert got >= 0.0


def test_tripower_quarticity_matches_documented_formula():
    r = np.array(RETURNS)
    n = len(r)
    p = 4.0 / 3.0
    # mu_{4/3} = 2^{2/3} Gamma(7/6) / Gamma(1/2) = E|Z|^{4/3}.
    mu = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
    mu_inv_cubed = mu ** (-3)
    # TQ = n mu_{4/3}^{-3} sum_{i=3}^n |r_i|^{4/3}|r_{i-1}|^{4/3}|r_{i-2}|^{4/3}.
    s = sum(
        abs(r[i]) ** p * abs(r[i - 1]) ** p * abs(r[i - 2]) ** p
        for i in range(2, n)
    )
    expected = n * mu_inv_cubed * s
    got = tsecon.tripower_quarticity(r)
    assert abs(got - expected) < 1e-12
    assert got >= 0.0


# ------------------------------------------------------- BNS jump test
def test_bns_jump_test_matches_documented_formula():
    r = np.array(RETURNS)
    n = len(r)
    # Rebuild RV, BV, TQ exactly as the crate does.
    rv = float(np.sum(r**2))
    bv = (math.pi / 2.0) * sum(abs(r[i]) * abs(r[i - 1]) for i in range(1, n))
    p = 4.0 / 3.0
    mu = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
    tq = (
        n
        * mu ** (-3)
        * sum(
            abs(r[i]) ** p * abs(r[i - 1]) ** p * abs(r[i - 2]) ** p
            for i in range(2, n)
        )
    )
    theta = math.pi**2 / 4.0 + math.pi - 5.0
    # z = sqrt(n)(RV-BV)/RV / sqrt(theta * max(1, TQ/BV^2)); note the floor
    # at 1 (TQ/BV^2 < 1 for this series, so the max is binding).
    denom = math.sqrt(theta * max(1.0, tq / (bv * bv)))
    expected = math.sqrt(n) * ((rv - bv) / rv) / denom

    res = tsecon.bns_jump_test(r)
    assert set(res) == {"ratio"}
    assert abs(res["ratio"] - expected) < 1e-12


# ------------------------------------------------------- range variance
# A few clean OHLC bars with high >= max(open, close) >= min >= low > 0.
_HIGH = np.array([10.5, 11.2, 10.9])
_LOW = np.array([9.8, 10.1, 10.3])
_OPEN = np.array([10.0, 10.4, 10.5])
_CLOSE = np.array([10.3, 10.9, 10.6])


def test_realized_range_parkinson_closed_form():
    # P = (1/(4 ln 2)) sum (ln(H/L))^2  (Parkinson 1980).
    expected = (1.0 / (4.0 * math.log(2.0))) * float(
        np.sum(np.log(_HIGH / _LOW) ** 2)
    )
    got = tsecon.realized_range(_HIGH, _LOW, method="parkinson")
    assert abs(got - expected) < 1e-12
    assert got >= 0.0
    # Parkinson is the default method.
    assert abs(tsecon.realized_range(_HIGH, _LOW) - expected) < 1e-12


def test_realized_range_garman_klass_closed_form():
    # GK = sum[ 0.5 (ln(H/L))^2 - (2 ln2 - 1)(ln(C/O))^2 ]  (Garman-Klass 1980).
    c2 = 2.0 * math.log(2.0) - 1.0
    expected = float(
        np.sum(
            0.5 * np.log(_HIGH / _LOW) ** 2 - c2 * np.log(_CLOSE / _OPEN) ** 2
        )
    )
    got = tsecon.realized_range(
        _HIGH, _LOW, method="garman_klass", open=_OPEN, close=_CLOSE
    )
    assert abs(got - expected) < 1e-12
    assert got >= 0.0


def test_realized_range_rejects_unknown_method():
    with pytest.raises(ValueError):
        tsecon.realized_range(_HIGH, _LOW, method="rogers_satchell")


def test_realized_range_garman_klass_requires_open_close():
    # Without open/close the Garman-Klass branch cannot be evaluated.
    with pytest.raises(ValueError):
        tsecon.realized_range(_HIGH, _LOW, method="garman_klass")
