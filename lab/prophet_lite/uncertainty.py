"""Prophet-style simulation intervals for prophet_lite.

Scheme (Taylor & Letham 2017, sec. 3.1 "Trend Uncertainty"; mirrors
``sample_predictive_trend`` in the MIT-licensed reference implementation —
re-implemented from the published description, no code copied):

The fitted history has S candidate changepoints on scaled time [0, 1] with
MAP rate adjustments delta_hat.  The generative assumption for the future is
that changepoints keep arriving "with the same average frequency" as in
history and with magnitudes matching the historical ones:

* number of new changepoints over the forecast window
      n_new ~ Poisson(S * (T - 1)),      T = max scaled forecast time (> 1),
  i.e. rate S per unit of (scaled) history length.  NOTE an honest wrinkle we
  reproduce deliberately: the S candidates actually sit in the first
  ``changepoint_range`` (80%) of history, so the literal historical frequency
  would be S / 0.8 per unit time; the reference implementation uses rate S
  and we follow it exactly for comparability.
* locations uniform on (1, T);
* magnitudes delta_new ~ Laplace(0, b) with b = mean(|delta_hat|) + 1e-8
  (the MLE of the Laplace scale given the fitted magnitudes; exact zeros
  from the L1 MAP damp b just as Stan's near-zeros do upstream).

Each trend path is the deterministic MAP trend extrapolation plus the new
hinge terms (continuous at t = 1 because (t - s)_+ vanishes for t <= s).
Observation noise N(0, sigma_hat^2) is added per draw and per step; interval
endpoints are empirical quantiles across draws.

Deliberately omitted relative to full Prophet: parameter uncertainty in
(k, m, delta, beta) — upstream that requires MCMC (``mcmc_samples > 0``); in
MAP mode Prophet's intervals carry exactly the two ingredients implemented
here.  Seasonality and regressor effects are treated as deterministic, as in
upstream MAP mode.

All randomness flows through a caller-supplied ``numpy.random.Generator`` —
seeded, no hidden state.
"""

from __future__ import annotations

import numpy as np

from .model import piecewise_linear_trend

__all__ = [
    "sample_future_trend_paths",
    "sample_predictive_draws",
    "intervals_from_draws",
]


def sample_future_trend_paths(t_future, m, k, delta, changepoints_t, n_draws, rng):
    """Simulate future trend paths (scaled units).

    Parameters
    ----------
    t_future : (h,) scaled future times (entries > 1 lie beyond the history).
    m, k, delta, changepoints_t : fitted MAP trend parameters.
    n_draws : int
    rng : numpy.random.Generator

    Returns
    -------
    (n_draws, h) ndarray of trend paths.
    """
    t_future = np.asarray(t_future, dtype=float)
    delta = np.asarray(delta, dtype=float)
    S = delta.shape[0]
    T = float(t_future.max()) if t_future.size else 1.0
    base = piecewise_linear_trend(t_future, m, k, delta, changepoints_t)
    paths = np.tile(base, (int(n_draws), 1))
    if S == 0 or T <= 1.0:
        return paths
    b = float(np.mean(np.abs(delta))) + 1e-8
    rate = S * (T - 1.0)
    n_new = rng.poisson(rate, size=int(n_draws))
    for i in range(int(n_draws)):
        if n_new[i] == 0:
            continue
        s_new = 1.0 + rng.random(n_new[i]) * (T - 1.0)
        d_new = rng.laplace(0.0, b, n_new[i])
        paths[i] += np.maximum(t_future[:, None] - s_new[None, :], 0.0) @ d_new
    return paths


def sample_predictive_draws(t_future, det_scaled, m, k, delta, changepoints_t,
                            sigma_scaled, y_scale, n_draws, seed):
    """Full predictive draws on the ORIGINAL y scale.

    draw = (trend_path + det_scaled + N(0, sigma_scaled)) * y_scale

    where ``det_scaled`` is the deterministic seasonal + regressor part in
    scaled units.

    Returns
    -------
    (n_draws, h) ndarray.
    """
    rng = np.random.default_rng(seed)
    paths = sample_future_trend_paths(t_future, m, k, delta, changepoints_t,
                                      n_draws, rng)
    noise = rng.normal(0.0, float(sigma_scaled), size=paths.shape)
    return (paths + np.asarray(det_scaled, dtype=float)[None, :] + noise) * float(y_scale)


def intervals_from_draws(draws, levels):
    """Equal-tailed empirical intervals from predictive draws.

    Parameters
    ----------
    draws : (n_draws, h) ndarray
    levels : iterable of floats in (0, 1), e.g. (0.8, 0.95)

    Returns
    -------
    (lower, upper) : two dicts keyed by ``str(level)`` (JSON-friendly),
    each value an (h,) ndarray.
    """
    lower, upper = {}, {}
    for lv in levels:
        lv = float(lv)
        if not 0.0 < lv < 1.0:
            raise ValueError(f"interval level must be in (0,1), got {lv}")
        a = (1.0 - lv) / 2.0
        lower[str(lv)] = np.quantile(draws, a, axis=0)
        upper[str(lv)] = np.quantile(draws, 1.0 - a, axis=0)
    return lower, upper
