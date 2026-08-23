"""Golden + property tests for the ACM (Adrian-Crump-Moench 2013)
regression-based term premium binding.

``fixtures/acm.json`` is a documented-formula golden: the ENTIRE three-step
pipeline (PCA factors, VAR(1), excess-return regressions, lambda0/lambda1,
affine recursions) is transcribed independently into NumPy by
``fixtures/generate_acm_fixtures.py`` — never calling tsecon — on (a) a
simulated affine DGP with known prices of risk and (b) the real 1961-2014
monthly GSW zero-coupon panel. The binding must reproduce every stored
quantity to 1e-8, recover the DGP's true premium, and track the NY Fed's
published ACM 10-year term premium in level and shape.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
ACM = json.loads((FIXTURES / "acm.json").read_text())

GOLDEN_KEYS_1D = [
    ("mu", "mu"),
    ("a", "a"),
    ("lambda0", "lambda0"),
    ("delta1", "delta1"),
    ("A", "A"),
    ("A_rn", "A_rn"),
    ("var_rsquared", "var_rsquared"),
    ("rx_rsquared", "rx_rsquared"),
    ("yield_rsquared", "yield_rsquared"),
]
GOLDEN_KEYS_2D = [
    ("factors", "factors"),
    ("factor_loadings", "loadings"),
    ("phi", "phi"),
    ("sigma", "sigma"),
    ("beta", "beta"),
    ("c", "c"),
    ("lambda1", "lambda1"),
    ("B", "B"),
    ("B_rn", "B_rn"),
]


def _fit(leg):
    case = ACM[leg]
    return case, tsecon.acm_term_premium(
        np.asarray(case["yields"], float),
        case["maturities"],
        n_factors=case["n_factors"],
        periods_per_year=case["periods_per_year"],
    )


@pytest.mark.parametrize("leg", ["sim", "gsw"])
def test_acm_golden_matches_numpy_pipeline(leg):
    case, res = _fit(leg)
    golden = case["golden"]

    assert list(res["rx_maturities"]) == golden["rx_maturities"]
    assert res["sigma2"] == pytest.approx(golden["sigma2"], abs=1e-8)
    assert res["delta0"] == pytest.approx(golden["delta0"], abs=1e-8)
    assert res["short_rate_rsquared"] == pytest.approx(
        golden["short_rate_rsquared"], abs=1e-8
    )
    for res_key, gold_key in GOLDEN_KEYS_1D:
        np.testing.assert_allclose(
            np.asarray(res[res_key], float),
            np.asarray(golden[gold_key], float),
            atol=1e-8,
            rtol=0,
            err_msg=f"{leg} {res_key}",
        )
    for res_key, gold_key in GOLDEN_KEYS_2D:
        np.testing.assert_allclose(
            np.asarray(res[res_key], float),
            np.asarray(golden[gold_key], float),
            atol=1e-8,
            rtol=0,
            err_msg=f"{leg} {res_key}",
        )

    # The decomposition itself: full first/last rows plus whole time series
    # at the stored report maturities.
    fitted = np.asarray(res["fitted"], float)
    rn = np.asarray(res["risk_neutral"], float)
    tp = np.asarray(res["term_premium"], float)
    np.testing.assert_allclose(fitted[0], golden["fitted_row0"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(fitted[-1], golden["fitted_row_last"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(tp[0], golden["term_premium_row0"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        tp[-1], golden["term_premium_row_last"], atol=1e-8, rtol=0
    )
    mats = list(case["maturities"])
    for n in mats:
        for key, arr in (("fitted", fitted), ("risk_neutral", rn), ("term_premium", tp)):
            field = f"{key}_{n}"
            if field in golden:
                np.testing.assert_allclose(
                    arr[:, mats.index(n)],
                    golden[field],
                    atol=1e-8,
                    rtol=0,
                    err_msg=f"{leg} {field}",
                )

    # Internal identity: fitted = risk-neutral + term premium, exactly.
    np.testing.assert_allclose(fitted, rn + tp, atol=1e-12, rtol=0)


def test_acm_recovers_the_true_term_premium_of_the_affine_dgp():
    case, res = _fit("sim")
    mats = list(case["maturities"])
    tp = np.asarray(res["term_premium"], float)
    true60 = np.asarray(case["true"]["term_premium_60"], float)
    est60 = tp[:, mats.index(60)]
    assert np.corrcoef(est60, true60)[0, 1] > 0.95
    assert np.abs(est60 - true60).mean() < 0.0060  # < 60bp vs a ~367bp premium


def test_acm_tracks_the_published_ny_fed_series():
    case, res = _fit("gsw")
    mats = list(case["maturities"])
    idx = np.asarray(case["published"]["quarter_row_idx"], int)
    pub_tp10 = np.asarray(case["published"]["acmtp10"], float)
    tp10 = np.asarray(res["term_premium"], float)[idx, mats.index(120)] * 100.0
    assert np.corrcoef(tp10, pub_tp10)[0, 1] > 0.97
    assert abs((tp10 - pub_tp10).mean()) < 0.35  # level gap, pp
    assert np.sqrt(((tp10 - pub_tp10) ** 2).mean()) < 0.5  # RMSE, pp


def test_acm_default_arguments_are_the_acm_baseline():
    # n_factors defaults to 5 (ACM's baseline), periods_per_year to monthly.
    case = ACM["gsw"]
    res = tsecon.acm_term_premium(np.asarray(case["yields"], float), case["maturities"])
    assert res["n_factors"] == 5
    assert res["periods_per_year"] == 12.0
    np.testing.assert_allclose(
        np.asarray(res["lambda0"], float), ACM["gsw"]["golden"]["lambda0"], atol=1e-8
    )


def test_acm_rejects_degenerate_inputs():
    case = ACM["sim"]
    y = np.asarray(case["yields"], float)
    mats = case["maturities"]

    with pytest.raises(ValueError, match="one-period"):
        tsecon.acm_term_premium(y[:, 1:], mats[1:], n_factors=3)
    with pytest.raises(ValueError, match="ascending"):
        tsecon.acm_term_premium(y, [1] + mats[:-1], n_factors=3)
    with pytest.raises(ValueError, match="n_factors"):
        tsecon.acm_term_premium(y, mats, n_factors=0)
    with pytest.raises(ValueError, match="periods_per_year"):
        tsecon.acm_term_premium(y, mats, n_factors=3, periods_per_year=-12.0)
    with pytest.raises(ValueError, match="residual degrees of freedom"):
        tsecon.acm_term_premium(y[:5], mats, n_factors=3)
    with pytest.raises(ValueError, match="non-finite"):
        bad = y.copy()
        bad[3, 4] = np.nan
        tsecon.acm_term_premium(bad, mats, n_factors=3)


def test_acm_docstring_names_every_returned_key():
    """Audit round 8: the returns enumeration in ``__doc__`` must cover every
    key actually returned (the echoed inputs ``maturities``/``n_factors``/
    ``periods_per_year`` were missing)."""
    import re

    case = ACM["sim"]
    res = tsecon.acm_term_premium(
        np.asarray(case["yields"], float),
        case["maturities"],
        n_factors=case["n_factors"],
        periods_per_year=case["periods_per_year"],
    )
    words = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", tsecon.acm_term_premium.__doc__ or ""))
    missing = set(res.keys()) - words
    assert not missing, f"acm_term_premium.__doc__ does not name returned keys: {sorted(missing)}"
