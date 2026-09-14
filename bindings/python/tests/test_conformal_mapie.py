"""MAPIE cross-check of the EnbPI and ACI conformal forecasters (VM-01).

Re-pins fixtures/conformal_mapie.json (`mapie.regression.TimeSeriesRegressor`
1.5.0 on the test_conformal.py AR(1) DGP; see the generator header for what
each leg is and is not):

* ACI — EXACT: with the same prefit linear point forecaster (handed to
  tsecon as a Python callable rebuilt from the stored coefficients), the
  same 75-residual sliding window, the same step size and MAPIE's
  `AbsoluteConformityScore(sym=True)` (its `TimeSeriesRegressor` default
  is `sym=False`, an asymmetric signed-residual pair — see the generator
  header), tsecon's online run reproduces MAPIE's per-origin bounds at
  1e-12 relative, its miss indicators exactly, and the `alpha_t` used at
  every origin exactly.
* EnbPI — STATISTICAL: the two implementations draw their bootstrap
  ensembles from different generators and use different quantile
  conventions (MAPIE's +1 finite-sample correction vs the paper's
  empirical quantile; different `beta` grids), so the comparison is in
  distribution: realized coverage, mean width and ensemble centres agree
  within stated bounds, and the measured gaps are printed.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
FX = json.loads((FIX / "conformal_mapie.json").read_text())
S = FX["series"]
Y = np.array(S["y"])
LAGS, CALIB, N_EVAL, ALPHA = S["lags"], S["calib"], S["n_eval"], S["alpha"]


def _prefit_base(coef, intercept):
    """The stored scikit-learn LinearRegression on `lags` lagged values, as
    a tsecon callable base(train, horizon): features [y_{t-1}, ..., y_{t-L}]."""
    coef = np.asarray(coef)

    def base(train, horizon):
        assert horizon == 1
        x = np.array([train[-l] for l in range(1, LAGS + 1)])
        return np.array([float(x @ coef + intercept)])

    return base


@pytest.mark.parametrize("case", FX["aci"], ids=lambda c: f"gamma={c['gamma']}")
def test_aci_reproduces_mapie_bounds_misses_and_alpha_trajectory(case):
    r = tsecon.conformal_backtest(
        Y, horizon=1, method="aci", base=_prefit_base(case["coef"], case["intercept"]),
        alpha=ALPHA, gamma=case["gamma"], calib=CALIB, n_eval=N_EVAL,
    )
    assert r["method"] == "aci" and r["n_eval"] == N_EVAL
    assert [int(o) for o in r["origins"]] == case["origins"]
    # The point forecasts are the same linear predictor to machine precision.
    np.testing.assert_allclose(r["mean"][0], case["point"], rtol=1e-12, atol=1e-12)
    # alpha_t used at each origin: MAPIE's current_alpha before the row's
    # update equals tsecon's trajectory entry (both start at alpha).
    np.testing.assert_allclose(r["alpha_trajectory"][0], case["alpha_used"], rtol=0, atol=1e-12)
    # Bounds: finite ones at 1e-12 relative; MAPIE's infinite ones (index
    # beyond the window) are tsecon's -inf/+inf.
    lo_m = np.array([(-np.inf if v is None else v) for v in case["lower"]])
    up_m = np.array([(np.inf if v is None else v) for v in case["upper"]])
    lo_t, up_t = np.asarray(r["lower"][0]), np.asarray(r["upper"][0])
    fin = np.isfinite(lo_m)
    assert np.array_equal(np.isfinite(lo_t), fin) and np.array_equal(np.isfinite(up_t), fin)
    np.testing.assert_allclose(lo_t[fin], lo_m[fin], rtol=1e-12)
    np.testing.assert_allclose(up_t[fin], up_m[fin], rtol=1e-12)
    # Misses exactly, hence the realized coverage and the final level.
    assert [bool(e) for e in r["err"][0]] == case["miss"]
    assert r["realized_coverage"][0] == pytest.approx(case["realized_coverage"], abs=1e-12)
    # tsecon's alpha after the last update equals MAPIE's current_alpha.
    last = r["alpha_trajectory"][0][-1] + case["gamma"] * (ALPHA - float(case["miss"][-1]))
    assert last == pytest.approx(case["alpha_final"], abs=1e-12)


def test_the_fixture_runs_stay_inside_the_window_and_the_infinite_convention_holds():
    """Both fixture runs keep `alpha_t` high enough that the
    `ceil((m + 1)(1 - alpha_t))`-th order statistic exists in the
    `calib`-residual window, so neither implementation produces an
    infinite bound (the per-origin parity test already asserts the
    finite/infinite PATTERN agrees). Where the two libraries part company
    is what happens when the index DOES run past the window: MAPIE returns
    an infinite bound under `allow_infinite_bounds=True`, tsecon refuses
    the call up front with a message naming `calib` and `alpha` and the
    number of residuals the level needs. Both are exercised below."""
    import math

    for case in FX["aci"]:
        assert all(v is not None for v in case["lower"]), case["gamma"]
        assert all(v is not None for v in case["upper"]), case["gamma"]
        worst = max(math.ceil((CALIB + 1) * (1 - a)) for a in case["alpha_used"])
        assert worst <= CALIB, (case["gamma"], worst)
    # alpha below 1 - m/(m + 1) = 1/76 cannot be delivered by 75 residuals.
    assert 0.005 < 1.0 / (CALIB + 1)
    case = FX["aci"][0]
    with pytest.raises(ValueError, match=r"alpha = 0\.005.*at least 199 calibration residuals"):
        tsecon.conformal_backtest(
            Y, horizon=1, method="aci", base=_prefit_base(case["coef"], case["intercept"]),
            alpha=0.005, gamma=0.0, calib=CALIB, n_eval=N_EVAL,
        )


@pytest.mark.parametrize("case", FX["enbpi"], ids=lambda c: f"beta={c['optimize_beta']}")
def test_enbpi_agrees_with_mapie_in_distribution(case):
    """Same algorithm, different bootstrap generators and quantile grids:
    coverage within 0.06, mean width within 10%, centres within 0.05 on
    average. The measured gaps are printed for the matrix row."""
    r = tsecon.conformal_backtest(
        Y, horizon=1, method="enbpi", base="ar", lags=LAGS, n_boot=case["n_boot"], seed=0,
        alpha=ALPHA, n_eval=N_EVAL, batch=1, optimize_beta=case["optimize_beta"],
    )
    assert [int(o) for o in r["origins"]] == case["origins"]
    cov_t = r["realized_coverage"][0]
    width_t = float(np.mean(np.asarray(r["upper"][0]) - np.asarray(r["lower"][0])))
    center_gap = float(np.mean(np.abs(np.asarray(r["mean"][0]) - np.asarray(case["center"]))))
    print(
        f"enbpi optimize_beta={case['optimize_beta']}: coverage tsecon {cov_t:.4f} vs mapie "
        f"{case['realized_coverage']:.4f}; mean width {width_t:.4f} vs {case['mean_width']:.4f} "
        f"(ratio {width_t / case['mean_width']:.4f}); mean |centre gap| {center_gap:.4f}"
    )
    assert abs(cov_t - case["realized_coverage"]) <= 0.06
    assert 0.9 <= width_t / case["mean_width"] <= 1.1
    assert center_gap < 0.05
