"""User-facing API for prophet_lite: ``fit`` and ``ProphetLiteResult``.

Follows the tsecon results convention: the result IS a plain ``dict`` of
documented keys (JSON-, pickle- and ``**``-friendly; values are floats,
ints, strings, plain dicts and numpy arrays), with forecasting methods
layered on top.  Every method also exists as a module-level function taking
the dict, so a round-tripped plain dict stays fully usable.

Deterministic seeding: every stochastic path takes an explicit ``seed``;
there is no module-level RNG state.

Provenance: from-scratch implementation of Taylor & Letham (2018),
"Forecasting at Scale", Amer. Statist. 72(1) 37-45.  See model.py and
README.md for the estimation details and the honest-omissions list.
"""

from __future__ import annotations

import numpy as np

from . import model as _m
from . import uncertainty as _u

__all__ = ["fit", "ProphetLiteResult", "forecast_from_result",
           "components_from_result", "predictive_draws_from_result"]

_SECONDS_PER_DAY = 86400.0


# ----------------------------------------------------------------------------
# time-index handling
# ----------------------------------------------------------------------------

def _parse_index(y, dates, seasonality):
    """Resolve the time index and the seasonality spec.

    ``dates`` may be:
      * None                       -> integer index 0..n-1;
      * a list/tuple of (period, K) pairs -> integer index with the explicit
        seasonalities (periods in index units);
      * datetime-like array (DatetimeIndex, datetime64, parseable strings)
        -> u = days since first observation; regular spacing required.

    Auto-seasonality for dated series (Prophet's defaults and enablement
    rules, simplified to what this lab needs):
      * yearly (P=365.25 d, K=10) if the span is >= 730 days;
      * weekly (P=7 d, K=3) if the spacing is daily and the span >= 14 days.
    An explicit ``seasonality`` list of (period, K) always overrides auto
    (periods in DAYS for dated series, in index steps otherwise).
    """
    n = len(y)

    def _pairs_to_spec(pairs, unit):
        spec = {}
        for period, K in pairs:
            period = float(period)
            K = int(K)
            if period <= 0 or K <= 0:
                raise ValueError(f"invalid seasonality pair ({period}, {K})")
            spec[f"seasonal_p{period:g}"] = {"period": period, "K": K, "unit": unit}
        return spec

    # form 1/2: no dates
    is_pair_list = (
        isinstance(dates, (list, tuple))
        and len(dates) > 0
        and isinstance(dates[0], (list, tuple))
        and len(dates[0]) == 2
    )
    if dates is None or is_pair_list:
        u = np.arange(n, dtype=float)
        pairs = list(dates) if is_pair_list else []
        if seasonality is not None:
            pairs = list(seasonality)
        return {
            "index_kind": "integer",
            "u": u,
            "step_units": 1.0,
            "last_date_ns": None,
            "seasonalities": _pairs_to_spec(pairs, "index"),
        }

    # form 3: datetime-like
    d = np.asarray(dates)
    if not np.issubdtype(d.dtype, np.datetime64):
        d = np.array(d, dtype="datetime64[ns]")
    d = d.astype("datetime64[ns]")
    if d.shape[0] != n:
        raise ValueError("dates and y must have equal length")
    sec = (d - d[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9
    u = sec / _SECONDS_PER_DAY  # days since first obs
    du = np.diff(u)
    if np.any(du <= 0):
        raise ValueError("dates must be strictly increasing")
    step = float(np.median(du))
    if np.max(np.abs(du - step)) > 1e-6 * max(step, 1.0):
        raise ValueError("prophet_lite requires regularly spaced dates")
    if seasonality is not None:
        spec = _pairs_to_spec(seasonality, "days")
    else:
        spec = {}
        span = float(u[-1])
        if span >= 730.0:
            spec["yearly"] = {"period": 365.25, "K": 10, "unit": "days"}
        if abs(step - 1.0) < 1e-6 and span >= 14.0:
            spec["weekly"] = {"period": 7.0, "K": 3, "unit": "days"}
    return {
        "index_kind": "dates",
        "u": u,
        "step_units": step,
        "last_date_ns": int(d[-1].astype(np.int64)),
        "seasonalities": spec,
    }


def _seasonal_design(u, seasonalities):
    """Stack Fourier blocks; return (n, 2*sum K) matrix and column slices."""
    blocks, slices, pos = [], {}, 0
    for name, s in seasonalities.items():
        Xb = _m.fourier_features(u, s["period"], s["K"])
        blocks.append(Xb)
        slices[name] = (pos, pos + Xb.shape[1])
        pos += Xb.shape[1]
    X = np.hstack(blocks) if blocks else np.empty((len(u), 0))
    return X, slices


# ----------------------------------------------------------------------------
# fit
# ----------------------------------------------------------------------------

def fit(y, dates=None, X=None, n_changepoints=25, changepoint_range=0.8,
        tau=0.05, seasonality=None, max_sigma_iter=50):
    """MAP-fit the decomposable model y = trend + seasonality + beta'x + eps.

    Parameters
    ----------
    y : (n,) array-like
        Observations.
    dates : None, datetime-like array, or list of (period, K) pairs
        None -> integer index, no seasonality (unless ``seasonality`` given).
        Datetime array -> time in days; yearly/weekly seasonality
        auto-enabled per the rules in ``_parse_index``.
        List of (period, K) pairs -> integer index with those Fourier
        seasonalities (periods in index units).
    X : (n, r) array-like, optional
        Extra regressors; standardized internally (means/stds stored).
        ``forecast``/``predictive_draws`` then require ``X_future``.
    n_changepoints : int, default 25
        Candidate trend changepoints, uniform over the first
        ``changepoint_range`` of the sample (Prophet defaults).
    changepoint_range : float, default 0.8
    tau : float, default 0.05
        Laplace prior scale on changepoint rate adjustments
        (Prophet's ``changepoint_prior_scale``).  LARGER tau = weaker
        L1 penalty = more active changepoints; tau -> 0 kills all of them.
    seasonality : list of (period, K), optional
        Explicit override of the seasonal spec (days for dated series,
        index steps otherwise).
    max_sigma_iter : int
        Cap on the (lasso <-> sigma) block-descent iterations; the fit
        reports ``converged`` honestly instead of hiding non-convergence.

    Returns
    -------
    ProphetLiteResult
        dict subclass; see its docstring for the key inventory and the
        ``.forecast`` / ``.components`` / ``.predictive_draws`` methods.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = y.shape[0]
    if n < 5:
        raise ValueError("need at least 5 observations")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values")

    idx = _parse_index(y, dates, seasonality)
    u = idx["u"]
    u_span = float(u[-1])
    t = u / u_span

    y_scale = float(np.max(np.abs(y)))
    if y_scale == 0.0:
        y_scale = 1.0
    y_s = y / y_scale

    cp_t, cp_idx = _m.make_changepoint_times(t, n_changepoints, changepoint_range)

    X_seas, seas_slices = _seasonal_design(u, idx["seasonalities"])

    if X is not None:
        Xr = np.asarray(X, dtype=float)
        if Xr.ndim == 1:
            Xr = Xr[:, None]
        if Xr.shape[0] != n:
            raise ValueError("X must have the same number of rows as y")
        x_mean = Xr.mean(axis=0)
        x_std = Xr.std(axis=0)
        x_std = np.where(x_std > 0, x_std, 1.0)
        X_extra = (Xr - x_mean) / x_std
    else:
        x_mean = x_std = None
        X_extra = np.empty((n, 0))

    X_unpen = np.column_stack([np.ones(n), t, X_seas, X_extra])
    fitres = _m.map_fit(y_s, t, cp_t, X_unpen, tau, max_sigma_iter=max_sigma_iter)

    b = fitres["b_unpen"]
    q_seas = X_seas.shape[1]
    m_, k_ = float(b[0]), float(b[1])
    beta_season = b[2:2 + q_seas]
    beta_extra = b[2 + q_seas:]

    trend_s = _m.piecewise_linear_trend(t, m_, k_, fitres["delta"], cp_t)
    seas_s = X_seas @ beta_season if q_seas else np.zeros(n)
    extra_s = X_extra @ beta_extra if X_extra.shape[1] else np.zeros(n)
    fitted = (trend_s + seas_s + extra_s) * y_scale

    res = ProphetLiteResult(
        # --- parameters (scaled space: y/y_scale, t in [0,1]) ---
        m=m_, k=k_,
        delta=fitres["delta"],
        changepoints_t=cp_t,
        changepoint_indices=cp_idx,
        beta_season=beta_season,
        beta_extra=beta_extra,
        sigma=float(fitres["sigma_scaled"] * y_scale),
        sigma_scaled=fitres["sigma_scaled"],
        # --- scaling & index state (all that's needed to forecast) ---
        y_scale=y_scale,
        u=u, u_span=u_span,
        index_kind=idx["index_kind"],
        step_units=idx["step_units"],
        last_date_ns=idx["last_date_ns"],
        seasonalities=idx["seasonalities"],
        seasonal_slices=seas_slices,
        x_mean=x_mean, x_std=x_std,
        # --- data & diagnostics ---
        y=y,
        fitted=fitted,
        residuals=y - fitted,
        n=n,
        tau=float(tau),
        lam=fitres["lam"],
        rss_scaled=fitres["rss"],
        n_changepoints=int(cp_t.shape[0]),
        n_active=fitres["n_active"],
        changepoint_range=float(changepoint_range),
        n_sigma_iter=fitres["n_sigma_iter"],
        converged=fitres["converged"],
        cd_converged=fitres["cd_converged"],
        kkt_gap=fitres["kkt_gap"],
    )
    return res


# ----------------------------------------------------------------------------
# forecasting from a (plain-dict) result
# ----------------------------------------------------------------------------

def _future_frame(res, h):
    """Future u, scaled t and seasonal design for horizon h."""
    h = int(h)
    if h < 1:
        raise ValueError("h must be >= 1")
    u = np.asarray(res["u"], dtype=float)
    step = float(res["step_units"])
    u_f = u[-1] + step * np.arange(1, h + 1)
    t_f = u_f / float(res["u_span"])
    X_seas_f, _ = _seasonal_design(u_f, res["seasonalities"])
    return u_f, t_f, X_seas_f


def _deterministic_scaled(res, h, X_future):
    """Scaled deterministic parts for the future: trend, seasonal+regressors."""
    u_f, t_f, X_seas_f = _future_frame(res, h)
    trend_s = _m.piecewise_linear_trend(t_f, res["m"], res["k"],
                                        res["delta"], res["changepoints_t"])
    det = X_seas_f @ np.asarray(res["beta_season"], dtype=float) \
        if X_seas_f.shape[1] else np.zeros(len(t_f))
    beta_extra = np.asarray(res["beta_extra"], dtype=float)
    if beta_extra.size:
        if X_future is None:
            raise ValueError("model was fit with extra regressors; "
                             "forecasting requires X_future of shape (h, r)")
        Xf = np.asarray(X_future, dtype=float)
        if Xf.ndim == 1:
            Xf = Xf[:, None]
        if Xf.shape != (int(h), beta_extra.size):
            raise ValueError(f"X_future must have shape ({h}, {beta_extra.size})")
        Xf = (Xf - res["x_mean"]) / res["x_std"]
        det = det + Xf @ beta_extra
    elif X_future is not None:
        raise ValueError("model was fit without extra regressors; drop X_future")
    return u_f, t_f, trend_s, det


def predictive_draws_from_result(res, h, n_draws=1000, seed=0, X_future=None):
    """Simulation draws from the predictive distribution (original y scale).

    Draws combine (i) Prophet's future-changepoint trend bootstrap and
    (ii) Gaussian observation noise — see uncertainty.py for the scheme and
    its deliberate omissions.  Deterministic given ``seed``.

    Returns
    -------
    (n_draws, h) ndarray.
    """
    _, t_f, _, det = _deterministic_scaled(res, h, X_future)
    return _u.sample_predictive_draws(
        t_f, det, res["m"], res["k"], res["delta"], res["changepoints_t"],
        res["sigma_scaled"], res["y_scale"], n_draws, seed)


def forecast_from_result(res, h, level=(0.8, 0.95), X_future=None,
                         n_draws=1000, seed=0):
    """Point forecast plus simulation intervals.

    The point forecast ("mean") is the deterministic MAP extrapolation
    (historical changepoints only, no noise), exactly Prophet's ``yhat``;
    intervals are empirical quantiles of ``predictive_draws``.

    Returns
    -------
    dict with keys
        "mean", "trend", "seasonal" : (h,) arrays, original scale;
        "lower", "upper" : dicts keyed by str(level) -> (h,) arrays;
        "level" : list of float levels;
        "h", "n_draws", "seed" : ints;
        "dates" : (h,) datetime64[ns] array for dated fits, else None.
    """
    if np.isscalar(level):
        level = (float(level),)
    levels = [float(lv) for lv in level]
    u_f, t_f, trend_s, det = _deterministic_scaled(res, h, X_future)
    y_scale = float(res["y_scale"])
    mean = (trend_s + det) * y_scale
    draws = _u.sample_predictive_draws(
        t_f, det, res["m"], res["k"], res["delta"], res["changepoints_t"],
        res["sigma_scaled"], res["y_scale"], n_draws, seed)
    lower, upper = _u.intervals_from_draws(draws, levels)
    if res["index_kind"] == "dates":
        base = np.datetime64(int(res["last_date_ns"]), "ns")
        offs = (u_f - float(res["u"][-1])) * _SECONDS_PER_DAY
        dates = base + np.round(offs * 1e9).astype("timedelta64[ns]")
    else:
        dates = None
    # seasonal part reported separately for convenience
    X_seas_f, _ = _seasonal_design(u_f, res["seasonalities"])
    seas = (X_seas_f @ np.asarray(res["beta_season"], dtype=float)
            if X_seas_f.shape[1] else np.zeros(len(u_f))) * y_scale
    return {
        "mean": mean,
        "trend": trend_s * y_scale,
        "seasonal": seas,
        "lower": lower,
        "upper": upper,
        "level": levels,
        "h": int(h),
        "n_draws": int(n_draws),
        "seed": int(seed),
        "dates": dates,
    }


def components_from_result(res):
    """Historical decomposition on the original scale.

    Returns
    -------
    dict with "trend", "seasonal" (total), one "seasonal_<name>" per
    component, "regressors", "fitted", "residual" — each an (n,) array.
    """
    u = np.asarray(res["u"], dtype=float)
    t = u / float(res["u_span"])
    y_scale = float(res["y_scale"])
    out = {}
    out["trend"] = _m.piecewise_linear_trend(
        t, res["m"], res["k"], res["delta"], res["changepoints_t"]) * y_scale
    X_seas, slices = _seasonal_design(u, res["seasonalities"])
    beta_s = np.asarray(res["beta_season"], dtype=float)
    total = np.zeros(len(u))
    for name, (a, b) in slices.items():
        part = X_seas[:, a:b] @ beta_s[a:b]
        out[f"seasonal_{name}"] = part * y_scale
        total += part
    out["seasonal"] = total * y_scale
    beta_x = np.asarray(res["beta_extra"], dtype=float)
    if beta_x.size:
        # reconstruct standardized regressor contribution from the fit
        out["regressors"] = res["fitted"] - out["trend"] - out["seasonal"]
    else:
        out["regressors"] = np.zeros(len(u))
    out["fitted"] = res["fitted"]
    out["residual"] = res["residuals"]
    return out


class ProphetLiteResult(dict):
    """Fit result: a plain dict of documented keys plus forecast methods.

    Keys (tsecon results convention — ``res["delta"]``, ``json``/``pickle``
    and ``**res`` all work):

    Parameters (scaled space: y/y_scale, time t in [0,1] over history)
        ``m``, ``k``               intercept and base slope
        ``delta``                  (S,) changepoint rate adjustments
                                   (exact zeros = inactive)
        ``changepoints_t``         (S,) candidate locations in scaled time
        ``changepoint_indices``    (S,) sample indices of the candidates
        ``beta_season``            Fourier coefficients (scaled y units;
                                   original-scale coefficient = beta*y_scale)
        ``beta_extra``             extra-regressor coefficients (on
                                   standardized regressors, scaled y units)
        ``sigma``                  observation noise sd, ORIGINAL scale
        ``sigma_scaled``           same, scaled space

    Scaling / index state (everything needed to forecast from the bare dict)
        ``y_scale``, ``u``, ``u_span``, ``index_kind`` ("integer"/"dates"),
        ``step_units``, ``last_date_ns``, ``seasonalities``,
        ``seasonal_slices``, ``x_mean``, ``x_std``

    Data & diagnostics
        ``y``, ``fitted``, ``residuals``, ``n``, ``tau``, ``lam`` (final L1
        weight sigma^2/tau), ``rss_scaled``, ``n_changepoints``,
        ``n_active``, ``changepoint_range``, ``n_sigma_iter``,
        ``converged``, ``cd_converged``, ``kkt_gap``
    """

    def forecast(self, h, level=(0.8, 0.95), X_future=None, n_draws=1000, seed=0):
        """See :func:`forecast_from_result`."""
        return forecast_from_result(self, h, level=level, X_future=X_future,
                                    n_draws=n_draws, seed=seed)

    def components(self):
        """See :func:`components_from_result`."""
        return components_from_result(self)

    def predictive_draws(self, h, n_draws=1000, seed=0, X_future=None):
        """See :func:`predictive_draws_from_result`."""
        return predictive_draws_from_result(self, h, n_draws=n_draws,
                                            seed=seed, X_future=X_future)
