"""Python-level contract + parity tests for var_irf_bands.

The heavy numerical validation (delta-method SE vs statsmodels to ~1e-15,
bootstrap reproducibility/coverage) lives in the tsecon-var Rust golden and
property suites. These tests pin the *binding* contract from Python: the dict
shape, that the point path equals var_irf exactly, that asymptotic bands are
point +/- z_{1-alpha/2} * se, and that the bootstrap branch is reproducible at
a fixed seed. Where statsmodels is installed, one parity assertion re-checks
the asymptotic SE end to end through the Python surface.
"""
import numpy as np
import pytest
import tsecon
from scipy.stats import norm


def _stable_var(n=300, k=3, seed=42):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(k)
    return y


DATA = _stable_var()
H = 10


def _shape(bands, key, k=3):
    arr = np.asarray(bands[key])
    assert arr.shape == (H + 1, k, k), f"{key} shape {arr.shape}"
    return arr


def test_asymptotic_contract_and_band_arithmetic():
    b = tsecon.var_irf_bands(DATA, lags=2, horizon=H, method="asymptotic", alpha=0.1)
    assert b["method"] == "asymptotic"
    assert b["alpha"] == pytest.approx(0.1)
    assert b["n_boot"] is None
    point, se = _shape(b, "point"), _shape(b, "se")
    lower, upper = _shape(b, "lower"), _shape(b, "upper")
    zc = norm.ppf(1.0 - 0.1 / 2.0)
    assert np.allclose(lower, point - zc * se, atol=1e-10)
    assert np.allclose(upper, point + zc * se, atol=1e-10)
    assert np.all(se >= 0.0)


def test_point_equals_var_irf_exactly():
    for orth in (True, False):
        for cumulative in (False, True):
            b = tsecon.var_irf_bands(
                DATA, lags=2, horizon=H, orth=orth, cumulative=cumulative,
                method="asymptotic",
            )
            ref = np.asarray(
                tsecon.var_irf(DATA, lags=2, horizon=H, orth=orth, cumulative=cumulative)
            )
            assert np.array_equal(np.asarray(b["point"]), ref), (orth, cumulative)


def test_alpha_widens_bands():
    narrow = tsecon.var_irf_bands(DATA, lags=2, horizon=H, method="asymptotic", alpha=0.32)
    wide = tsecon.var_irf_bands(DATA, lags=2, horizon=H, method="asymptotic", alpha=0.01)
    nw = np.asarray(wide["upper"]) - np.asarray(wide["lower"])
    nn = np.asarray(narrow["upper"]) - np.asarray(narrow["lower"])
    # 99% band strictly wider than 68% everywhere the se is nonzero
    se = np.asarray(narrow["se"])
    assert np.all(nw[se > 0] > nn[se > 0])


def test_bootstrap_is_reproducible_and_brackets_point():
    kw = dict(lags=2, horizon=H, method="bootstrap", alpha=0.1, n_boot=200, seed=7)
    b1 = tsecon.var_irf_bands(DATA, **kw)
    b2 = tsecon.var_irf_bands(DATA, **kw)
    assert b1["n_boot"] == 200
    for key in ("point", "se", "lower", "upper"):
        assert np.array_equal(np.asarray(b1[key]), np.asarray(b2[key])), key
    point = np.asarray(b1["point"])
    assert np.all(np.asarray(b1["lower"]) <= point + 1e-12)
    assert np.all(np.asarray(b1["upper"]) >= point - 1e-12)


def test_bootstrap_different_seed_differs():
    a = tsecon.var_irf_bands(DATA, lags=2, horizon=H, method="bootstrap", n_boot=200, seed=1)
    b = tsecon.var_irf_bands(DATA, lags=2, horizon=H, method="bootstrap", n_boot=200, seed=2)
    assert not np.array_equal(np.asarray(a["se"]), np.asarray(b["se"]))


@pytest.mark.parametrize("orth", [False, True])
def test_asymptotic_se_matches_statsmodels(orth):
    sm = pytest.importorskip("statsmodels.tsa.api")
    res = sm.VAR(DATA).fit(2, trend="c")
    ref = np.asarray(res.irf(H).stderr(orth=orth))
    got = np.asarray(
        tsecon.var_irf_bands(DATA, lags=2, horizon=H, orth=orth, method="asymptotic")["se"]
    )
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-10)
