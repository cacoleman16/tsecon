"""Python-surface tests for the Ng-Perron (2001) M unit-root tests.

No runnable independent implementation of the M tests exists (statsmodels
0.14.6 and arch 8.0.0 do not ship them, and no CRAN package does), so this
file carries an independent NumPy re-implementation of the documented
pipeline — GLS detrending via numpy.linalg.lstsq, MAIC on the detrended
series, the AR spectral density at frequency zero, the four statistics —
and re-pins the compiled binding against it end to end. That is a
cross-implementation check of the arithmetic (different linear algebra,
different language), not an independent authority; the statistical claim
is carried by the crate's seeded Monte-Carlo size/power tests
(crates/tsecon-diag/tests/ng_perron_properties.rs).

It also demonstrates the shared-detrending contract with dfgls through the
public boundary: one NumPy GLS-detrended series reproduces BOTH the DF-GLS
tau (trendless ADF t-ratio on it) and the M statistics, so the two tests
provably consume the same detrended series.
"""
import numpy as np
import pytest
import tsecon

# Ng-Perron (2001) Table 1, transcribed independently of the Rust source.
TABLE1 = {
    "c": {
        "mza": {"1%": -13.8, "5%": -8.1, "10%": -5.7},
        "mzt": {"1%": -2.58, "5%": -1.98, "10%": -1.62},
        "msb": {"1%": 0.174, "5%": 0.233, "10%": 0.275},
        "mpt": {"1%": 1.78, "5%": 3.17, "10%": 4.45},
    },
    "ct": {
        "mza": {"1%": -23.8, "5%": -17.3, "10%": -14.2},
        "mzt": {"1%": -3.42, "5%": -2.91, "10%": -2.62},
        "msb": {"1%": 0.143, "5%": 0.168, "10%": 0.185},
        "mpt": {"1%": 4.03, "5%": 5.48, "10%": 6.67},
    },
}


# --------------------------------------------------- NumPy reference pipeline

def _gls_detrend(y, trend):
    """ERS GLS detrending at cbar = -7 ('c') / -13.5 ('ct')."""
    n = len(y)
    cbar = -7.0 if trend == "c" else -13.5
    a = 1.0 + cbar / n
    if trend == "c":
        z = np.ones((n, 1))
    else:
        z = np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=float)])
    yq = np.concatenate([[y[0]], y[1:] - a * y[:-1]])
    zq = np.vstack([z[:1], z[1:] - a * z[:-1]])
    beta, *_ = np.linalg.lstsq(zq, yq, rcond=None)
    return y - z @ beta


def _adf_design(yd, k, trim):
    """Trendless ADF design trimmed at `trim` lags: response dy_t and
    columns [y_{t-1}, dy_{t-1}, .., dy_{t-k}] for t = trim+1..n-1."""
    n = len(yd)
    dy = np.diff(yd)
    resp = dy[trim:]
    cols = [yd[trim: n - 1]]
    for j in range(1, k + 1):
        cols.append(dy[trim - j: n - 1 - j])
    return np.column_stack(cols), resp


def _maic(yd, kmax):
    """Ng-Perron MAIC on the common sample trimmed at kmax."""
    n = len(yd)
    rows = n - 1 - kmax
    lev = yd[kmax: n - 1]
    sum_lev2 = lev @ lev
    best, best_ic = 0, np.inf
    for k in range(kmax + 1):
        x, resp = _adf_design(yd, k, kmax)
        beta, *_ = np.linalg.lstsq(x, resp, rcond=None)
        resid = resp - x @ beta
        sigma2 = (resid @ resid) / rows
        tau = beta[0] ** 2 * sum_lev2 / sigma2
        ic = np.log(sigma2) + 2.0 * (tau + k) / rows
        if ic < best_ic:
            best_ic, best = ic, k
    return best


def _ng_perron_ref(y, trend, lags=None, max_lags=None):
    """The full documented pipeline, independently in NumPy."""
    y = np.asarray(y, dtype=float)
    yd = _gls_detrend(y, trend)
    n = len(yd)
    if lags is None:
        if max_lags is None:
            max_lags = min(
                int(np.ceil(12.0 * (n / 100.0) ** 0.25)), (n - 1) // 2 - 1
            )
        k = _maic(yd, max_lags)
    else:
        k = lags
    x, resp = _adf_design(yd, k, k)
    beta, *_ = np.linalg.lstsq(x, resp, rcond=None)
    resid = resp - x @ beta
    rows = n - 1 - k
    sigma2_e = (resid @ resid) / rows
    b1 = beta[1:].sum()
    s2_ar = sigma2_e / (1.0 - b1) ** 2
    kappa = (yd[:-1] @ yd[:-1]) / n**2
    w = yd[-1] ** 2 / n
    cbar = -7.0 if trend == "c" else -13.5
    if trend == "c":
        mpt = (cbar**2 * kappa - cbar * w) / s2_ar
    else:
        mpt = (cbar**2 * kappa + (1.0 - cbar) * w) / s2_ar
    return {
        "mza": (w - s2_ar) / (2.0 * kappa),
        "msb": np.sqrt(kappa / s2_ar),
        "mzt": (w - s2_ar) / (2.0 * np.sqrt(kappa * s2_ar)),
        "mpt": mpt,
        "used_lag": k,
        "nobs": rows,
        "s2_ar": s2_ar,
        "yd": yd,
    }


def _series(name):
    rng = np.random.default_rng(42)
    walk = np.cumsum(rng.standard_normal(240))
    trended = walk + 3.0 + 0.15 * np.arange(240)
    ar = np.empty(240)
    ar[0] = 0.0
    for t in range(1, 240):
        ar[t] = 0.85 * ar[t - 1] + rng.standard_normal()
    noise = rng.standard_normal(240)
    return {"walk": walk, "trended": trended, "ar085": ar, "noise": noise}[name]


CASES = [
    ("walk", "c", None, None),
    ("walk", "ct", None, None),
    ("walk", "c", 0, None),
    ("walk", "ct", 3, None),
    ("walk", "c", None, 6),
    ("trended", "ct", None, None),
    ("ar085", "c", None, None),
    ("ar085", "c", 2, None),
    ("noise", "c", 0, None),
    ("noise", "ct", None, 4),
]


@pytest.mark.parametrize(
    "series,trend,lags,max_lags",
    CASES,
    ids=[f"{s}-{t}-lags{l}-max{m}" for s, t, l, m in CASES],
)
def test_matches_independent_numpy_reference(series, trend, lags, max_lags):
    y = _series(series)
    got = tsecon.ng_perron(y, trend=trend, lags=lags, max_lags=max_lags)
    ref = _ng_perron_ref(y, trend, lags=lags, max_lags=max_lags)
    assert got["used_lag"] == ref["used_lag"]
    assert got["nobs"] == ref["nobs"]
    for k in ("mza", "mzt", "msb", "mpt", "s2_ar"):
        assert got[k] == pytest.approx(ref[k], rel=1e-9), k


def test_mzt_identity_through_the_boundary():
    for name in ("walk", "ar085", "noise"):
        y = _series(name)
        for trend in ("c", "ct"):
            r = tsecon.ng_perron(y, trend=trend)
            assert r["mzt"] == pytest.approx(r["mza"] * r["msb"], rel=1e-13)


def test_shared_detrending_with_dfgls_through_the_boundary():
    """One NumPy GLS-detrended series reproduces BOTH dfgls' tau and
    ng_perron's M statistics: through the public boundary, the two tests
    demonstrably consume the same detrended series (in the crate they call
    the identical engine; the crate test pins that bitwise)."""
    y = _series("walk")
    for trend, k in [("c", 2), ("ct", 2)]:
        yd = _gls_detrend(y, trend)
        # dfgls leg: trendless ADF t-ratio on the detrended series.
        x, resp = _adf_design(yd, k, k)
        beta, *_ = np.linalg.lstsq(x, resp, rcond=None)
        resid = resp - x @ beta
        dof = len(resp) - x.shape[1]
        s2 = (resid @ resid) / dof
        xtx_inv = np.linalg.inv(x.T @ x)
        tau = beta[0] / np.sqrt(s2 * xtx_inv[0, 0])
        d = tsecon.dfgls(y, regression=trend, lags=k)
        assert d["statistic"] == pytest.approx(tau, rel=1e-9)
        # ng_perron leg: the M statistics from the same detrended series.
        ref = _ng_perron_ref(y, trend, lags=k)
        got = tsecon.ng_perron(y, trend=trend, lags=k)
        for key in ("mza", "mzt", "msb", "mpt"):
            assert got[key] == pytest.approx(ref[key], rel=1e-9)


def test_table1_critical_values_and_contract():
    y = _series("walk")
    r = tsecon.ng_perron(y)
    assert {
        "mza", "mzt", "msb", "mpt", "used_lag", "nobs", "s2_ar", "crit", "trend",
    } <= set(r)
    assert r["trend"] == "c"
    assert r["nobs"] == len(y) - 1 - r["used_lag"]
    assert r["crit"] == TABLE1["c"]
    assert tsecon.ng_perron(y, trend="ct")["crit"] == TABLE1["ct"]
    # lags="maic" is the spelled-out default.
    assert tsecon.ng_perron(y, lags="maic") == r


def test_verdicts_walk_vs_stationary():
    walk = _series("walk")
    r = tsecon.ng_perron(walk)
    assert r["mza"] > r["crit"]["mza"]["10%"]  # cannot reject the null
    ar = _series("ar085")
    r = tsecon.ng_perron(ar)
    for k in ("mza", "mzt", "msb", "mpt"):
        assert r[k] < r["crit"][k]["5%"], k  # rejects


def test_errors_teach():
    y = _series("walk")
    with pytest.raises(ValueError, match="expected \"c\" or \"ct\""):
        tsecon.ng_perron(y, trend="n")
    with pytest.raises(ValueError, match="maic"):
        tsecon.ng_perron(y, lags="aic")
    with pytest.raises(ValueError, match="non-negative integer"):
        tsecon.ng_perron(y, lags=2.5)
    with pytest.raises(ValueError):
        tsecon.ng_perron(np.full(50, 3.0))  # constant series
    with pytest.raises(ValueError):
        tsecon.ng_perron(np.arange(4, dtype=float), trend="ct")  # too short
    bad = y.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError):
        tsecon.ng_perron(bad)
    # Exact linear trend: a teaching error, not a garbage number.
    det = 2.0 + 0.5 * np.arange(60, dtype=float)
    with pytest.raises(ValueError):
        tsecon.ng_perron(det, trend="ct")


def test_statsmodels_and_arch_do_not_implement_ng_perron():
    """The validation record: the reference venv's statsmodels and arch
    genuinely lack the M tests, so no runnable independent golden exists.
    If either ever grows one, this canary fails and the fixture should be
    upgraded to a reference-run golden."""
    sm_tsa = pytest.importorskip("statsmodels.tsa.stattools")
    arch_ur = pytest.importorskip("arch.unitroot")
    for mod in (sm_tsa, arch_ur):
        names = [n.lower() for n in dir(mod)]
        assert not any("ng_perron" in n or "ngperron" in n for n in names), (
            f"{mod.__name__} now ships a Ng-Perron implementation - "
            "upgrade the validation to a reference-run golden"
        )
