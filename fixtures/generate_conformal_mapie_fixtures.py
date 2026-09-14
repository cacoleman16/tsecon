"""MAPIE cross-check fixture for the EnbPI and ACI conformal forecasters
(`conformal_backtest(method="aci")` / `(method="enbpi")`), closing the
validation-matrix follow-up VM-01: `mapie.regression.TimeSeriesRegressor`
(version in `_meta`) run on the `test_conformal.py` AR(1) DGP.

Run with the project venv (this script never imports tsecon):
    .venv/bin/python fixtures/generate_conformal_mapie_fixtures.py

What is compared, and the honest grade of each leg:

  * ACI — EXACT cross-check (independent package). Both implementations
    run the Gibbs-Candès recursion `alpha_{t+1} = alpha_t + gamma (alpha -
    err_t)` over a sliding window of the last `calib` absolute residuals
    with the `ceil((m + 1)(1 - alpha_t))`-th smallest score as the
    half-width (MAPIE: `_compute_regression_quantile`, level
    `ceil((1 - a)(n + 1)) / n` with numpy `method="lower"`, which is that
    same order statistic; infinite when the index exceeds the window).
    MAPIE is a regression library, so the point forecaster is handed in as
    a PREFIT scikit-learn `LinearRegression` on `lags` lagged values, fit
    once on the training block (`coef`/`intercept` stored, so the tsecon
    side reproduces it as a Python callable `base(train, horizon)` to
    machine precision), `cv="prefit"`, and — this one matters —
    `conformity_score=AbsoluteConformityScore(sym=True)`: MAPIE's
    `TimeSeriesRegressor` DEFAULTS to `sym=False`, which builds the
    interval from a pair of signed-residual quantiles (`beta = alpha_t/2`
    below, `1 - alpha_t + beta` above) and is asymmetric about the point
    forecast, i.e. not the construction ACI's own paper specifies with an
    absolute score. With `sym=True` MAPIE's half-width is the same
    `ceil((m + 1)(1 - alpha_t))`-th smallest absolute residual tsecon uses;
    the calibration block is the `calib` rows before the evaluation window
    and each evaluation row is scored with `predict(allow_infinite_bounds=
    True)` at the current `alpha_t`, then `adapt_conformal_inference(gamma)`
    and `update` (which rolls the oldest score out). The stored per-origin
    lower/upper bounds, miss indicators and the `alpha_t` used at each
    origin are pinned by `bindings/python/tests/test_conformal_mapie.py`
    at 1e-12 relative (bounds) / exact (misses, trajectory); measured on
    the stored runs: gamma = 0.05 reaches coverage 0.8933 and
    `alpha_final` 0.0750, gamma = 0.005 coverage 0.9067 and 0.1025, the
    two sides agreeing origin for origin. Three known differences, none
    of which fires on these runs: MAPIE clips `alpha_t` to [0, 1] where
    tsecon keeps the paper's unclipped recursion; MAPIE counts a target
    exactly on a bound as a miss where tsecon counts it covered (measure
    zero on continuous data); and when `alpha_t` falls far enough that
    `ceil((m + 1)(1 - alpha_t))` exceeds the window, MAPIE returns an
    infinite bound under `allow_infinite_bounds=True` while tsecon
    REFUSES the call, naming the level and the number of residuals it
    would need (both behaviours are pinned by the test — `alpha_t` stays
    inside the window on both stored runs, its worst order index being 74
    of 75).

  * EnbPI — STATISTICAL cross-check only, stated as such. The two
    implementations are the same algorithm (Xu & Xie 2021 Algorithm 1:
    an iid bootstrap ensemble of least-squares AR fits, leave-one-out
    aggregated residuals, a sliding residual window updated online) but
    NOT the same arithmetic: the bootstrap draws come from different
    generators (MAPIE `BlockBootstrap(length=1)` on a NumPy RandomState;
    tsecon Philox substreams), MAPIE applies the `+1` finite-sample
    correction to the residual quantile where tsecon follows the paper's
    ordinary empirical quantile, and the `beta` line search uses a
    different grid (MAPIE: `n` points from `alpha/(n + 1)` to `alpha`
    with mixed `higher`/`lower` quantiles; tsecon: 21 points on
    `[0, alpha]`, type-1 quantiles). So the stored MAPIE online run
    (ensemble centre, symmetric bounds, misses) is compared to tsecon's
    on the same series in DISTRIBUTION: realized coverage within 0.06,
    mean interval width within 10%, mean absolute centre difference below
    0.05 (the ensemble-averaging noise at `n_boot = 25`). The measured
    numbers are printed by the test and quoted on the matrix row.

DGP (seed 20260911): AR(1), phi = 0.6, unit innovations, n = 300; lag
features `lags = 2`; training rows target y[2:150), calibration rows
y[150:225) (`calib = 75`), evaluation rows y[225:300) (`n_eval = 75`);
alpha = 0.1; ACI gamma = 0.05 (the shift-scenario step of the card) and
0.005 (the paper's default); EnbPI `n_boot = 25`, `random_state = 0`.
"""
import json
import platform
from pathlib import Path

import numpy as np
import mapie
import sklearn
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import TimeSeriesRegressor
from mapie.subsample import BlockBootstrap
from sklearn.linear_model import LinearRegression

OUT = Path(__file__).parent
META = {
    "mapie": mapie.__version__,
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "python": platform.python_version(),
}
SEED = 20260911
N, PHI, LAGS = 300, 0.6, 2
T_TRAIN, CALIB, N_EVAL = 150, 75, 75
ALPHA = 0.1


def ar1(rng, n, phi=0.6, sigma=1.0):
    y = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = phi * prev + sigma * rng.standard_normal()
        y[t] = prev
    return y


def lag_design(y, lags):
    """Row t (t >= lags): features [y_{t-1}, ..., y_{t-lags}], target y_t."""
    rows = np.arange(lags, len(y))
    X = np.column_stack([y[rows - l] for l in range(1, lags + 1)])
    return rows, X, y[rows]


def aci_run(y, gamma):
    rows, X, target = lag_design(y, LAGS)
    tr = rows < T_TRAIN
    ca = (rows >= T_TRAIN) & (rows < T_TRAIN + CALIB)
    ev = rows >= T_TRAIN + CALIB
    lr = LinearRegression().fit(X[tr], target[tr])
    # sym=True is REQUIRED for an apples-to-apples comparison: MAPIE's
    # TimeSeriesRegressor defaults to AbsoluteConformityScore(sym=False),
    # which builds the interval from a pair of SIGNED-residual quantiles
    # (beta = alpha_t/2 below, 1 - alpha_t + beta above) and is therefore
    # asymmetric about the point forecast. tsecon's ACI follows the
    # Gibbs-Candes construction with the ABSOLUTE score, i.e. a symmetric
    # half-width; sym=True selects exactly that in MAPIE.
    ts = TimeSeriesRegressor(
        estimator=lr, method="aci", cv="prefit",
        conformity_score=AbsoluteConformityScore(sym=True),
    )
    ts.fit(X[ca], target[ca])
    alpha_used, lower, upper, miss, point = [], [], [], [], []
    for x_row, y_row in zip(X[ev], target[ev]):
        x = x_row[None, :]
        ts._get_alpha()
        alpha_t = float(ts.current_alpha.setdefault(ALPHA, ALPHA))
        pred, bounds = ts.predict(x, confidence_level=1 - ALPHA, allow_infinite_bounds=True)
        lo, up = float(bounds[0, 0, 0]), float(bounds[0, 1, 0])
        alpha_used.append(alpha_t)
        point.append(float(pred[0]))
        lower.append(lo)
        upper.append(up)
        miss.append(not (lo < y_row < up))
        ts.adapt_conformal_inference(x, np.array([y_row]), gamma=gamma, confidence_level=1 - ALPHA)
        ts.update(x, np.array([y_row]))
    return {
        "gamma": gamma,
        "coef": lr.coef_.tolist(),
        "intercept": float(lr.intercept_),
        "origins": [int(r - 1) for r in rows[ev]],
        "initial_scores": [float(v) for v in np.abs(target[ca] - lr.predict(X[ca]))],
        "alpha_used": alpha_used,
        "point": point,
        "lower": [v if np.isfinite(v) else None for v in lower],
        "upper": [v if np.isfinite(v) else None for v in upper],
        "miss": [bool(m) for m in miss],
        "realized_coverage": 1.0 - float(np.mean(miss)),
        "alpha_final": float(ts.current_alpha[ALPHA]),
    }


def enbpi_run(y, n_boot, random_state, optimize_beta):
    rows, X, target = lag_design(y, LAGS)
    tr = rows < T_TRAIN + CALIB
    ev = rows >= T_TRAIN + CALIB
    cv = BlockBootstrap(n_resamplings=n_boot, length=1, overlapping=False, random_state=random_state)
    ts = TimeSeriesRegressor(estimator=LinearRegression(), method="enbpi", cv=cv, agg_function="mean")
    ts.fit(X[tr], target[tr])
    center, lower, upper, miss = [], [], [], []
    for x_row, y_row in zip(X[ev], target[ev]):
        x = x_row[None, :]
        pred, bounds = ts.predict(x, ensemble=True, confidence_level=1 - ALPHA, optimize_beta=optimize_beta)
        center.append(float(pred[0]))
        lower.append(float(bounds[0, 0, 0]))
        upper.append(float(bounds[0, 1, 0]))
        miss.append(not (lower[-1] < y_row < upper[-1]))
        ts.update(x, np.array([y_row]), ensemble=True)
    width = np.array(upper) - np.array(lower)
    return {
        "n_boot": n_boot,
        "random_state": random_state,
        "optimize_beta": optimize_beta,
        "n_train_rows": int(tr.sum()),
        "origins": [int(r - 1) for r in rows[ev]],
        "center": center,
        "lower": lower,
        "upper": upper,
        "miss": [bool(m) for m in miss],
        "realized_coverage": 1.0 - float(np.mean(miss)),
        "mean_width": float(width.mean()),
    }


def main():
    rng = np.random.default_rng(SEED)
    y = ar1(rng, N, PHI)
    fx = {
        "_meta": META,
        "series": {"seed": SEED, "phi": PHI, "n": N, "lags": LAGS, "t_train": T_TRAIN,
                   "calib": CALIB, "n_eval": N_EVAL, "alpha": ALPHA, "y": y.tolist()},
        "aci": [aci_run(y, 0.05), aci_run(y, 0.005)],
        "enbpi": [enbpi_run(y, 25, 0, False), enbpi_run(y, 25, 0, True)],
    }
    path = OUT / "conformal_mapie.json"
    path.write_text(json.dumps(fx, separators=(",", ":")))
    for a in fx["aci"]:
        print(f"aci gamma={a['gamma']}: coverage {a['realized_coverage']:.4f}, alpha_final {a['alpha_final']:.4f}, "
              f"{sum(v is None for v in a['lower'])} infinite")
    for e in fx["enbpi"]:
        print(f"enbpi optimize_beta={e['optimize_beta']}: coverage {e['realized_coverage']:.4f}, "
              f"mean width {e['mean_width']:.4f}")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
