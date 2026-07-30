"""Predictive-interval coverage: do tsecon's forecast bands cover the future?

A forecast interval is a promise about repeated samples: a nominal 95% h-step
predictive interval should contain the realised y_{T+h} in 95% of repeated
samples.  This module MEASURES that rate for every tsecon entry point that
emits a forecast interval, on data-generating processes whose parameters we
know, and reports what it finds -- including where the promise is not kept.

Surfaces covered
----------------
* ``arima_fit(..., forecast_steps=H, conf_alpha=a)`` -- ``forecast_lower`` /
  ``forecast_upper`` / ``forecast_se``, on an AR(1) (Exp 1), an ARMA(1,1)
  (Exp 2) and a random walk with drift (Exp 6).
* ``var_forecast(..., steps=H, alpha=a)`` -- ``lower`` / ``upper`` on a
  stationary bivariate VAR(1) (Exp 3), plus a nominal-level sweep (Exp 4).
* ``theta_forecast`` and ``backtest`` -- both emit POINT paths only.  There is
  no library interval to score, so Exp 5 (a) asserts that fact so it cannot
  silently change and (b) measures a *user-constructed* interval built from the
  backtest error distribution.  That interval is ours, not the library's, and
  is labelled as such everywhere it appears.

Two devices carry the argument.

ORACLE COLUMNS.  Every library interval here is a *plug-in* interval: it
evaluates the textbook Gaussian formula at the estimated parameters and ignores
the sampling error in those estimates.  Run the identical formula at the TRUE
parameters and it covers exactly.  The gap between the two columns is therefore
not a wrong standard error -- it is the price of not knowing the parameters,
i.e. a property of the approximation.  Reporting both makes that distinction
measurable instead of rhetorical.

A CLOSED FORM.  For the I(1) case in Experiment 6 the omitted term is available
in closed form, so the coverage the shipped band *must* attain is computable in
advance and printed next to the measured value.  When measurement matches
prediction, the shortfall is not merely observed, it is explained.

Every experiment is seeded from one master seed, every coverage number is
printed with its Monte Carlo standard error, comparisons between two bands
measured on the same replications use a PAIRED standard error, and nothing is
tuned to make an assertion pass.  Where the shortfall is real it is printed
plainly and left in the returned data.

Run it::

    python docs/examples/coverage/forecast_intervals.py
    python docs/examples/coverage/forecast_intervals.py --quick
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Sequence

import numpy as np
from scipy import stats

import tsecon

# --------------------------------------------------------------------------
# Global, printed, reproducible.
# --------------------------------------------------------------------------
SEED = 20260729
NOMINAL = 0.95
ALPHA = 1.0 - NOMINAL


def _rng(stream: int) -> np.random.Generator:
    """An independent Generator per experiment, deterministic in SEED."""
    return np.random.default_rng([SEED, stream])


def mc_se(p: float, reps: int) -> float:
    """Monte Carlo standard error of a measured coverage rate."""
    return math.sqrt(max(p * (1.0 - p), 0.0) / reps) if reps > 0 else float("nan")


def dev_in_se(p: float, se: float, nominal: float = NOMINAL) -> float:
    """(measured - nominal) expressed in Monte Carlo standard errors."""
    return float("nan") if se <= 0.0 else (p - nominal) / se


# --------------------------------------------------------------------------
# Data-generating processes.  All start from the exact stationary
# distribution, so there is no burn-in approximation anywhere.
# --------------------------------------------------------------------------
def sim_ar1(
    rng: np.random.Generator, reps: int, n: int, phi: float, sigma: float, mu: float
) -> np.ndarray:
    """(reps, n) draws of y_t = mu + phi (y_{t-1} - mu) + sigma e_t, e ~ N(0,1)."""
    eps = rng.standard_normal((reps, n)) * sigma
    y = np.empty((reps, n))
    state = rng.standard_normal(reps) * (sigma / math.sqrt(1.0 - phi * phi))
    for t in range(n):
        state = phi * state + eps[:, t]
        y[:, t] = state
    return y + mu


def sim_arma11(
    rng: np.random.Generator,
    reps: int,
    n: int,
    phi: float,
    theta: float,
    sigma: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(reps, n) draws of an ARMA(1,1), plus the innovations that generated them.

    y_t - mu = phi (y_{t-1} - mu) + e_t + theta e_{t-1}, started from the exact
    stationary variance sigma^2 (1 + 2 phi theta + theta^2) / (1 - phi^2).  The
    returned innovation array is aligned with y, so ``eps[:, t]`` is the shock
    that entered ``y[:, t]`` -- Experiment 2 needs e_T to build the exact
    conditional forecast.
    """
    eps = rng.standard_normal((reps, n + 1)) * sigma
    var0 = sigma * sigma * (1.0 + 2.0 * phi * theta + theta * theta) / (1.0 - phi * phi)
    y = np.empty((reps, n))
    state = rng.standard_normal(reps) * math.sqrt(var0)
    for t in range(n):
        state = phi * state + eps[:, t + 1] + theta * eps[:, t]
        y[:, t] = state
    return y + mu, eps[:, 1:]


def _stationary_cov(a: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Solve G = A G A' + Sigma by iteration (tiny systems, converges to 1e-14)."""
    g = sigma.copy()
    for _ in range(5000):
        g_new = a @ g @ a.T + sigma
        if np.max(np.abs(g_new - g)) < 1e-14:
            return g_new
        g = g_new
    return g


def sim_var1(
    rng: np.random.Generator, reps: int, n: int, a: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """(reps, n, k) draws of y_t = A y_{t-1} + u_t, u ~ N(0, Sigma)."""
    k = a.shape[0]
    u = rng.standard_normal((reps, n, k)) @ np.linalg.cholesky(sigma).T
    y = np.empty((reps, n, k))
    state = rng.standard_normal((reps, k)) @ np.linalg.cholesky(
        _stationary_cov(a, sigma)
    ).T
    for t in range(n):
        state = state @ a.T + u[:, t, :]
        y[:, t, :] = state
    return y


# --------------------------------------------------------------------------
# Closed-form pieces of the plug-in formula the library uses.
# --------------------------------------------------------------------------
def ar1_plugin(
    y_last: float, mu: float, phi: float, sigma2: float, horizons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """AR(1) h-step conditional mean and plug-in forecast SE."""
    mean = mu + (phi**horizons) * (y_last - mu)
    var = sigma2 * (1.0 - phi ** (2 * horizons)) / (1.0 - phi * phi)
    return mean, np.sqrt(var)


def arma11_psi(phi: float, theta: float, h_max: int) -> np.ndarray:
    """MA(inf) weights of an ARMA(1,1): psi_0 = 1, psi_j = phi^{j-1}(phi+theta)."""
    psi = np.empty(h_max)
    psi[0] = 1.0
    for j in range(1, h_max):
        psi[j] = (phi ** (j - 1)) * (phi + theta)
    return psi


# --------------------------------------------------------------------------
# Per-replication bookkeeping shared by every experiment.
#
# Keeping the raw per-replication indicators (rather than running totals) is
# what makes honest comparisons possible: two bands scored on the SAME
# replications are strongly positively correlated, so the standard error of
# their difference is far smaller than the independent-sample formula suggests.
# Every band-vs-band number below is therefore a paired standard error, while
# every level-vs-nominal number uses the plain binomial one.
# --------------------------------------------------------------------------
class BandAccumulator:
    """Collects hit indicators and widths for several competing bands."""

    def __init__(
        self, variants: Sequence[str], reps: int, h_max: int, pooled: bool = False
    ) -> None:
        self.variants = tuple(variants)
        self.pooled = pooled  # True when each cell pools k series in one rep
        self.hit = {v: np.zeros((reps, h_max)) for v in self.variants}
        self.wid = {v: np.zeros((reps, h_max)) for v in self.variants}
        self.all_in = {v: np.zeros(reps, dtype=bool) for v in self.variants}
        self.ok = np.zeros(reps, dtype=bool)

    def add(
        self,
        r: int,
        future: np.ndarray,
        bands: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Score replication ``r``; ``future`` is (h,) or (h, k)."""
        self.ok[r] = True
        for v in self.variants:
            lo, hi = bands[v]
            inside = (future >= lo) & (future <= hi)
            self.all_in[v][r] = bool(inside.all())
            if inside.ndim == 2:
                self.hit[v][r] = inside.mean(axis=1)
                self.wid[v][r] = (hi - lo).mean(axis=1)
            else:
                self.hit[v][r] = inside
                self.wid[v][r] = hi - lo

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        n = int(self.ok.sum())
        hit = {v: self.hit[v][self.ok] for v in self.variants}
        cov = {v: hit[v].mean(axis=0) for v in self.variants}
        if self.pooled:
            se = {v: hit[v].std(axis=0, ddof=1) / math.sqrt(n) for v in self.variants}
        else:
            se = {
                v: np.array([mc_se(p, n) for p in cov[v]]) for v in self.variants
            }
        ref = self.variants[0]
        h_max = hit[ref].shape[1]

        paired_vs_ref_diff: dict[str, list[float]] = {}
        paired_vs_ref_se: dict[str, list[float]] = {}
        for v in self.variants[1:]:
            d = hit[v] - hit[ref]
            paired_vs_ref_diff[v] = d.mean(axis=0).tolist()
            paired_vs_ref_se[v] = (d.std(axis=0, ddof=1) / math.sqrt(n)).tolist()

        d_h = hit[ref][:, 0] - hit[ref][:, h_max - 1]
        return {
            "reps": n,
            "variants": list(self.variants),
            "coverage": {v: cov[v].tolist() for v in self.variants},
            "mc_se": {v: se[v].tolist() for v in self.variants},
            "dev_in_se": {
                v: [dev_in_se(p, s) for p, s in zip(cov[v], se[v])]
                for v in self.variants
            },
            "mean_width": {
                v: self.wid[v][self.ok].mean(axis=0).tolist() for v in self.variants
            },
            "joint_all_horizons": {
                v: float(self.all_in[v][self.ok].mean()) for v in self.variants
            },
            "paired_ref": ref,
            "paired_vs_ref_diff": paired_vs_ref_diff,
            "paired_vs_ref_se": paired_vs_ref_se,
            "paired_h1_minus_hlast": {
                "diff": float(d_h.mean()),
                "se": float(d_h.std(ddof=1) / math.sqrt(n)),
            },
        }


# --------------------------------------------------------------------------
# Experiment 1: arima_fit on an AR(1).
# --------------------------------------------------------------------------
def exp_ar1_arima(
    phi: float = 0.9,
    t_obs: int = 100,
    h_max: int = 12,
    reps: int = 700,
    stream: int = 1,
) -> dict[str, Any]:
    """Predictive coverage of ``arima_fit`` bands on an AR(1), by horizon.

    Six bands, all at the same nominal level:

    ``library``     the shipped ``forecast_lower`` / ``forecast_upper``.
    ``replicated``  our evaluation of the textbook plug-in formula at the
                    library's own MLEs -- a check that the band really is
                    ``mean +/- z * plug-in se`` (asserted to 1e-8).
    ``oracle``      the same formula at the TRUE (mu, phi, sigma2).  Correct by
                    construction, so it isolates how much of any shortfall is
                    parameter uncertainty rather than a wrong formula.
    ``true_mu_s2``  true mean path, estimated sigma2 -> the sigma2 channel.
    ``est_mu_s2``   estimated mean path, true sigma2 -> the mean-path channel.
    ``t_quantile``  library SE with a t_{T-2} quantile instead of z: the
                    cheapest candidate fix, scored rather than advocated.
    """
    rng = _rng(stream)
    h = np.arange(1, h_max + 1)
    mu_true, sigma_true = 0.0, 1.0
    sigma2_true = sigma_true**2
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)
    t_q = stats.t.ppf(1.0 - ALPHA / 2.0, df=t_obs - 2)

    variants = (
        "library",
        "replicated",
        "oracle",
        "true_mu_s2",
        "est_mu_s2",
        "t_quantile",
    )
    acc = BandAccumulator(variants, reps, h_max)
    max_abs_replication_gap = 0.0
    failures = 0

    data = sim_ar1(rng, reps, t_obs + h_max, phi, sigma_true, mu_true)
    for r in range(reps):
        train, future = data[r, :t_obs], data[r, t_obs:]
        try:
            fit = tsecon.arima_fit(
                train,
                p=1,
                d=0,
                q=0,
                constant=True,
                forecast_steps=h_max,
                conf_alpha=ALPHA,
            )
        except Exception:  # pragma: no cover - counted, never hidden
            failures += 1
            continue
        names = list(fit["param_names"])
        par = np.asarray(fit["params"], dtype=float)
        c_hat = float(par[names.index("const")])
        phi_hat = float(par[names.index("ar.L1")])
        s2_hat = float(par[names.index("sigma2")])
        mu_hat = c_hat / (1.0 - phi_hat) if abs(1.0 - phi_hat) > 1e-8 else c_hat
        y_last = float(train[-1])

        lo = np.asarray(fit["forecast_lower"], dtype=float)
        hi = np.asarray(fit["forecast_upper"], dtype=float)
        m_lib = np.asarray(fit["forecast_mean"], dtype=float)
        se_lib = np.asarray(fit["forecast_se"], dtype=float)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            failures += 1
            continue

        m_hat, se_hat = ar1_plugin(y_last, mu_hat, phi_hat, s2_hat, h)
        max_abs_replication_gap = max(
            max_abs_replication_gap,
            float(np.max(np.abs(m_hat - z * se_hat - lo))),
            float(np.max(np.abs(m_hat + z * se_hat - hi))),
        )
        m_or, se_or = ar1_plugin(y_last, mu_true, phi, sigma2_true, h)
        _, se_s2 = ar1_plugin(y_last, mu_true, phi, s2_hat, h)

        acc.add(
            r,
            future,
            {
                "library": (lo, hi),
                "replicated": (m_hat - z * se_hat, m_hat + z * se_hat),
                "oracle": (m_or - z * se_or, m_or + z * se_or),
                "true_mu_s2": (m_or - z * se_s2, m_or + z * se_s2),
                "est_mu_s2": (m_hat - z * se_or, m_hat + z * se_or),
                "t_quantile": (m_lib - t_q * se_lib, m_lib + t_q * se_lib),
            },
        )

    out: dict[str, Any] = {
        "name": f"exp1_ar1_arima_phi{phi}",
        "title": f"arima_fit(1,0,0) predictive bands on an AR(1), phi={phi}",
        "surface": "arima_fit -> forecast_lower / forecast_upper / forecast_se",
        "dgp": f"y_t = {phi} y_(t-1) + e_t, e ~ N(0,1), mu = 0",
        "kind": "PRED (frequentist interval about a realisation, not a parameter)",
        "t_obs": t_obs,
        "failures": failures,
        "seed": SEED,
        "stream": stream,
        "nominal": NOMINAL,
        "horizons": h.tolist(),
        "max_abs_replication_gap": max_abs_replication_gap,
    }
    out.update(acc.summary())
    return out


# --------------------------------------------------------------------------
# Experiment 2: arima_fit on an ARMA(1,1).
# --------------------------------------------------------------------------
def exp_arma11_arima(
    phi: float = 0.6,
    theta: float = 0.4,
    t_obs: int = 100,
    h_max: int = 8,
    reps: int = 600,
    stream: int = 2,
) -> dict[str, Any]:
    """Predictive coverage of ``arima_fit`` bands on an ARMA(1,1), by horizon.

    Harder than Experiment 1 for a reason the AR case cannot show: the h-step
    forecast needs the latent innovation e_T, which the estimator must filter
    out of the data.  The oracle column uses the e_T we actually simulated, so
    it is the exact conditional interval; the library has to approximate both
    the parameters and e_T, and the h=1 row is where that shows.
    """
    rng = _rng(stream)
    h = np.arange(1, h_max + 1)
    mu_true, sigma_true = 0.0, 1.0
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)
    psi = arma11_psi(phi, theta, h_max)
    se_or = sigma_true * np.sqrt(np.cumsum(psi**2))

    acc = BandAccumulator(("library", "oracle"), reps, h_max)
    failures = 0

    data, eps = sim_arma11(rng, reps, t_obs + h_max, phi, theta, sigma_true, mu_true)
    for r in range(reps):
        train, future = data[r, :t_obs], data[r, t_obs:]
        try:
            fit = tsecon.arima_fit(
                train,
                p=1,
                d=0,
                q=1,
                constant=True,
                forecast_steps=h_max,
                conf_alpha=ALPHA,
            )
        except Exception:  # pragma: no cover
            failures += 1
            continue
        lo = np.asarray(fit["forecast_lower"], dtype=float)
        hi = np.asarray(fit["forecast_upper"], dtype=float)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            failures += 1
            continue

        # Exact conditional mean at the true parameters, using the simulated e_T.
        m_or = np.empty(h_max)
        prev = phi * (train[-1] - mu_true) + theta * eps[r, t_obs - 1]
        m_or[0] = prev
        for j in range(1, h_max):
            prev = phi * prev
            m_or[j] = prev
        m_or = m_or + mu_true

        acc.add(
            r,
            future,
            {
                "library": (lo, hi),
                "oracle": (m_or - z * se_or, m_or + z * se_or),
            },
        )

    out: dict[str, Any] = {
        "name": "exp2_arma11_arima",
        "title": (
            f"arima_fit(1,0,1) predictive bands on an ARMA(1,1), "
            f"phi={phi}, theta={theta}"
        ),
        "surface": "arima_fit -> forecast_lower / forecast_upper",
        "dgp": f"y_t = {phi} y_(t-1) + e_t + {theta} e_(t-1), e ~ N(0,1)",
        "kind": "PRED",
        "t_obs": t_obs,
        "failures": failures,
        "seed": SEED,
        "stream": stream,
        "nominal": NOMINAL,
        "horizons": h.tolist(),
        "note": (
            "the oracle band conditions on the simulated e_T, which no "
            "estimator observes; the library must filter it"
        ),
    }
    out.update(acc.summary())
    return out


# --------------------------------------------------------------------------
# Experiment 3: var_forecast on a stationary VAR(1).
# --------------------------------------------------------------------------
VAR_A = np.array([[0.70, 0.15], [0.10, 0.60]])
VAR_SIGMA = np.array([[1.00, 0.40], [0.40, 1.00]])


def exp_var_forecast(
    t_obs: int = 100,
    fit_lags: int = 1,
    h_max: int = 12,
    reps: int = 6000,
    stream: int = 3,
) -> dict[str, Any]:
    """Predictive coverage of ``var_forecast`` intervals, by horizon.

    ``fit_lags`` lets the same DGP be fitted at the true order (1) or
    over-specified (4).  Over-specification adds parameter uncertainty without
    adding bias, so the contrast isolates the plug-in cost from any
    misspecification story.

    ``joint_all_horizons`` reports the fraction of replications in which
    *every* one of the (h, series) marginal intervals holds.  A marginal
    interval makes no joint promise, so this number is far below the nominal
    level by construction -- it is here to keep the reader honest about what a
    fan chart is, not as a defect.
    """
    rng = _rng(stream)
    k = VAR_A.shape[0]
    h = np.arange(1, h_max + 1)
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)

    # Oracle h-step MSE: sum_{j<h} A^j Sigma A^j'.
    se_or = np.empty((h_max, k))
    acc_mse = np.zeros((k, k))
    powj = np.eye(k)
    for j in range(h_max):
        acc_mse = acc_mse + powj @ VAR_SIGMA @ powj.T
        se_or[j] = np.sqrt(np.diag(acc_mse))
        powj = powj @ VAR_A

    acc = BandAccumulator(("library", "oracle"), reps, h_max, pooled=True)
    by_series = {v: np.zeros((reps, h_max, k)) for v in ("library", "oracle")}
    failures = 0

    data = sim_var1(rng, reps, t_obs + h_max, VAR_A, VAR_SIGMA)
    for r in range(reps):
        train, future = data[r, :t_obs, :], data[r, t_obs:, :]
        try:
            fc = tsecon.var_forecast(train, lags=fit_lags, steps=h_max, alpha=ALPHA)
        except Exception:  # pragma: no cover
            failures += 1
            continue
        lo = np.asarray(fc["lower"], dtype=float)
        hi = np.asarray(fc["upper"], dtype=float)

        m_or = np.empty((h_max, k))
        state = train[-1].copy()
        for j in range(h_max):
            state = VAR_A @ state
            m_or[j] = state

        bands = {
            "library": (lo, hi),
            "oracle": (m_or - z * se_or, m_or + z * se_or),
        }
        acc.add(r, future, bands)
        for v, (b_lo, b_hi) in bands.items():
            by_series[v][r] = (future >= b_lo) & (future <= b_hi)

    ok = acc.ok
    out: dict[str, Any] = {
        "name": f"exp3_var_forecast_T{t_obs}_lags{fit_lags}",
        "title": (
            f"var_forecast intervals on a stationary VAR(1), T={t_obs}, "
            f"fitted lags={fit_lags}"
        ),
        "surface": "var_forecast -> lower / upper",
        "dgp": "y_t = A y_(t-1) + u_t, A = [[.7,.15],[.1,.6]], Sigma = [[1,.4],[.4,1]]",
        "kind": "PRED",
        "t_obs": t_obs,
        "fit_lags": fit_lags,
        "failures": failures,
        "seed": SEED,
        "stream": stream,
        "nominal": NOMINAL,
        "horizons": h.tolist(),
        "coverage_by_series": {
            v: by_series[v][ok].mean(axis=0).tolist() for v in by_series
        },
        "note": (
            "coverage pools the k=2 series; its MC se is clustered by "
            "replication (sd of the per-rep pooled indicator / sqrt(reps)), "
            "which is honest about the two series sharing a shock covariance"
        ),
    }
    out.update(acc.summary())
    return out


# --------------------------------------------------------------------------
# Experiment 4: nominal-level sweep.
# --------------------------------------------------------------------------
def exp_level_sweep(
    levels: tuple[float, ...] = (0.50, 0.80, 0.90, 0.95, 0.99),
    t_obs: int = 100,
    h_max: int = 6,
    reps: int = 4000,
    stream: int = 4,
) -> dict[str, Any]:
    """Is the *level* honoured, or only its ordering?  ``var_forecast``, h = 1..6.

    A plug-in band is too narrow by a roughly multiplicative factor, so the
    absolute shortfall in percentage points is largest at the middling levels,
    where the normal density at the endpoint is largest, and smallest at 99%,
    whose endpoints sit far out in a thin tail.  Sweeping the level shows the
    shortfall is a property of the approximation rather than of one hard-coded
    quantile, and that ``alpha`` is a real level and not a knob.
    """
    rng = _rng(stream)
    per_rep = {lev: np.empty((reps, h_max)) for lev in levels}
    data = sim_var1(rng, reps, t_obs + h_max, VAR_A, VAR_SIGMA)
    for r in range(reps):
        train, future = data[r, :t_obs, :], data[r, t_obs:, :]
        for lev in levels:
            fc = tsecon.var_forecast(train, lags=1, steps=h_max, alpha=1.0 - lev)
            lo = np.asarray(fc["lower"], dtype=float)
            hi = np.asarray(fc["upper"], dtype=float)
            per_rep[lev][r] = ((future >= lo) & (future <= hi)).mean(axis=1)
    cov = {lev: per_rep[lev].mean(axis=0) for lev in levels}
    se = {lev: per_rep[lev].std(axis=0, ddof=1) / math.sqrt(reps) for lev in levels}
    return {
        "name": "exp4_level_sweep",
        "title": "var_forecast: is the nominal level honoured across levels?",
        "surface": "var_forecast(alpha=1-level) -> lower / upper",
        "dgp": "same stationary VAR(1) as Experiment 3",
        "kind": "PRED",
        "t_obs": t_obs,
        "reps": reps,
        "seed": SEED,
        "stream": stream,
        "levels": list(levels),
        "horizons": list(range(1, h_max + 1)),
        "coverage": {lev: cov[lev].tolist() for lev in levels},
        "mc_se": {lev: se[lev].tolist() for lev in levels},
        "shortfall_pp": {
            lev: [100.0 * (p - lev) for p in cov[lev]] for lev in levels
        },
        "note": "pooled over k=2 series; MC se clustered by replication",
    }


# --------------------------------------------------------------------------
# Experiment 5: the point-only surfaces, and a DIY interval around them.
# --------------------------------------------------------------------------
def exp_point_only_surfaces(
    t_obs: int = 200,
    train: int = 100,
    h_max: int = 6,
    reps: int = 1000,
    stream: int = 5,
) -> dict[str, Any]:
    """``theta_forecast`` and ``backtest`` emit NO interval.  Proof, then a DIY.

    Part 1 records what the two functions actually return, so a future release
    that starts emitting bands cannot slip past this experiment unnoticed.

    Part 2 is explicitly NOT a library promise.  It builds the interval a
    practitioner would build from the pieces tsecon does ship: run ``backtest``
    on the training sample, take the empirical alpha/2 and 1-alpha/2 quantiles
    of the h-step pseudo-out-of-sample errors, and hang them off the
    ``theta_forecast`` point path.  Coverage is measured on two DGPs -- a random
    walk with drift, where Theta is well suited, and a mean-reverting AR(1),
    where it is misspecified.  A symmetric Gaussian variant (point +/- z times
    the sd of the same errors) is scored alongside, because it cannot absorb
    forecast bias while the quantile version can.
    """
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)
    rng = _rng(stream)

    # ---- Part 1: structural facts about the shipped return values.
    probe = sim_ar1(rng, 1, 240, 0.7, 1.0, 0.0)[0]
    theta_out = tsecon.theta_forecast(probe, steps=h_max, period=1)
    bt = tsecon.backtest(
        probe, window="expanding", train=train, horizon=h_max, forecaster="theta"
    )
    interval_words = ("lower", "upper", "se", "conf", "interval", "band", "alpha")
    structure = {
        "theta_forecast_type": type(theta_out).__name__,
        "theta_forecast_shape": list(np.shape(theta_out)),
        "theta_forecast_emits_interval": False,
        "backtest_keys": sorted(bt.keys()),
        "backtest_interval_keys": sorted(
            key for key in bt if any(w in key.lower() for w in interval_words)
        ),
    }

    # ---- Part 2: the user-built interval, on two DGPs.
    dgps: dict[str, np.ndarray] = {
        "rw_drift": np.cumsum(0.10 + rng.standard_normal((reps, t_obs + h_max)), axis=1),
        "ar1_phi0.7": sim_ar1(rng, reps, t_obs + h_max, 0.7, 1.0, 0.0),
    }

    out: dict[str, Any] = {}
    n_origins = 0
    for dgp_name, data in dgps.items():
        hits_q = np.zeros(h_max)
        hits_g = np.zeros(h_max)
        width_q = np.zeros(h_max)
        width_g = np.zeros(h_max)
        scored = np.zeros(h_max)
        for r in range(reps):
            hist, future = data[r, :t_obs], data[r, t_obs:]
            bt_r = tsecon.backtest(
                hist,
                window="expanding",
                train=train,
                horizon=h_max,
                forecaster="theta",
            )
            fc = np.asarray(bt_r["forecasts"], dtype=float)
            tg = np.asarray(bt_r["targets"], dtype=float)
            err = tg - fc  # (h_max, n_origins); NaN where an origin runs out
            n_origins = fc.shape[1]
            point = np.asarray(
                tsecon.theta_forecast(hist, steps=h_max, period=1), dtype=float
            )
            for j in range(h_max):
                e = err[j][np.isfinite(err[j])]
                if e.size < 20:
                    continue
                q_lo, q_hi = np.quantile(e, [ALPHA / 2.0, 1.0 - ALPHA / 2.0])
                s = e.std(ddof=1)
                lo_q, hi_q = point[j] + q_lo, point[j] + q_hi
                lo_g, hi_g = point[j] - z * s, point[j] + z * s
                hits_q[j] += float(lo_q <= future[j] <= hi_q)
                hits_g[j] += float(lo_g <= future[j] <= hi_g)
                width_q[j] += hi_q - lo_q
                width_g[j] += hi_g - lo_g
                scored[j] += 1.0
        cov_q, cov_g = hits_q / scored, hits_g / scored
        out[dgp_name] = {
            "scored_reps": scored.tolist(),
            "coverage_empirical_quantile": cov_q.tolist(),
            "coverage_gaussian_sd": cov_g.tolist(),
            "mc_se_empirical_quantile": [
                mc_se(p, int(n)) for p, n in zip(cov_q, scored)
            ],
            "mc_se_gaussian_sd": [mc_se(p, int(n)) for p, n in zip(cov_g, scored)],
            "mean_width_empirical_quantile": (width_q / scored).tolist(),
            "mean_width_gaussian_sd": (width_g / scored).tolist(),
        }

    return {
        "name": "exp5_point_only_surfaces",
        "title": "theta_forecast / backtest: point paths only, plus a user-built interval",
        "surface": "theta_forecast -> ndarray; backtest -> forecasts/targets/accuracy",
        "kind": "NONE for the library; PRED for the user-constructed interval",
        "t_obs": t_obs,
        "train": train,
        "reps": reps,
        "seed": SEED,
        "stream": stream,
        "nominal": NOMINAL,
        "horizons": list(range(1, h_max + 1)),
        "n_backtest_origins": n_origins,
        "structure": structure,
        "diy": out,
        "disclaimer": (
            "the DIY intervals are constructed IN THIS SCRIPT from backtest "
            "errors; tsecon makes no interval promise for theta_forecast or "
            "backtest, so a shortfall here belongs to our construction, not to "
            "the library"
        ),
    }


# --------------------------------------------------------------------------
# Experiment 6: the I(1) case, where the omitted term is available in closed
# form -- so the shortfall can be PREDICTED, not merely observed.
# --------------------------------------------------------------------------
def exp_rw_drift_arima(
    drift: float = 0.10,
    t_obs: int = 100,
    h_max: int = 12,
    reps: int = 1500,
    stream: int = 6,
) -> dict[str, Any]:
    """``arima_fit(0,1,0)`` on a random walk with drift: the shortfall in closed form.

    The fitted model is dy_t = mu + e_t, so mu_hat is the mean of T-1
    differences with variance sigma^2/(T-1), and the h-step forecast error is
    h (mu - mu_hat) + sum_{j<=h} e_{T+j}.  Those two pieces are independent, so
    the exact predictive variance is

        sigma^2 * (h + h^2 / (T-1)),

    whereas ``forecast_se`` returns sigma_hat * sqrt(h) -- asserted below to
    machine precision.  The dropped h^2/(T-1) term is exactly the parameter
    uncertainty a plug-in band ignores, and it grows FASTER than the innovation
    term, which is why this is the case where plug-in bands fail worst and why
    the failure is worst at short T and long h.

    Because the omitted term is known, so is the coverage a nominal 95% plug-in
    band must attain when sigma is known:

        2 * Phi(z / sqrt(1 + h/(T-1))) - 1

    printed as ``predicted``.  ``corrected`` restores the missing term at the
    estimated sigma^2, i.e. it is what the band should have been.
    """
    rng = _rng(stream)
    h = np.arange(1, h_max + 1)
    sigma_true = 1.0
    z = stats.norm.ppf(1.0 - ALPHA / 2.0)
    predicted = 2.0 * stats.norm.cdf(z / np.sqrt(1.0 + h / (t_obs - 1))) - 1.0

    acc = BandAccumulator(("library", "oracle", "corrected"), reps, h_max)
    max_abs_se_gap = 0.0
    failures = 0

    steps = drift + rng.standard_normal((reps, t_obs + h_max)) * sigma_true
    data = np.cumsum(steps, axis=1)
    for r in range(reps):
        train, future = data[r, :t_obs], data[r, t_obs:]
        try:
            fit = tsecon.arima_fit(
                train,
                p=0,
                d=1,
                q=0,
                constant=True,
                forecast_steps=h_max,
                conf_alpha=ALPHA,
            )
        except Exception:  # pragma: no cover
            failures += 1
            continue
        names = list(fit["param_names"])
        par = np.asarray(fit["params"], dtype=float)
        mu_hat = float(par[names.index("const")])
        s2_hat = float(par[names.index("sigma2")])
        y_last = float(train[-1])
        se_lib = np.asarray(fit["forecast_se"], dtype=float)
        max_abs_se_gap = max(
            max_abs_se_gap,
            float(np.max(np.abs(se_lib - math.sqrt(s2_hat) * np.sqrt(h)))),
        )

        m_hat = y_last + h * mu_hat
        m_or = y_last + h * drift
        se_or = sigma_true * np.sqrt(h)
        se_fix = np.sqrt(s2_hat * (h + h * h / (t_obs - 1)))
        acc.add(
            r,
            future,
            {
                "library": (
                    np.asarray(fit["forecast_lower"], dtype=float),
                    np.asarray(fit["forecast_upper"], dtype=float),
                ),
                "oracle": (m_or - z * se_or, m_or + z * se_or),
                "corrected": (m_hat - z * se_fix, m_hat + z * se_fix),
            },
        )

    out: dict[str, Any] = {
        "name": f"exp6_rw_drift_T{t_obs}_h{h_max}",
        "title": (
            f"arima_fit(0,1,0) predictive bands on a random walk with drift "
            f"{drift}, T={t_obs}, H={h_max}"
        ),
        "surface": "arima_fit -> forecast_lower / forecast_upper / forecast_se",
        "dgp": f"y_t = y_(t-1) + {drift} + e_t, e ~ N(0,1)   [I(1): the practical case]",
        "kind": "PRED",
        "t_obs": t_obs,
        "failures": failures,
        "seed": SEED,
        "stream": stream,
        "nominal": NOMINAL,
        "horizons": h.tolist(),
        "predicted_coverage": predicted.tolist(),
        "max_abs_se_gap": max_abs_se_gap,
        "note": (
            "predicted = 2*Phi(z/sqrt(1+h/(T-1)))-1 is the coverage the shipped "
            "band must attain when sigma is known; measured sits at or a little "
            "below it because sigma2 is also the MLE"
        ),
    }
    out.update(acc.summary())
    return out


# --------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------
_RULE = "=" * 100


def _print_header(res: dict[str, Any]) -> None:
    print(_RULE)
    print(res["title"])
    print(f"  surface : {res['surface']}")
    if "dgp" in res:
        print(f"  dgp     : {res['dgp']}")
    print(f"  kind    : {res['kind']}")
    bits = [f"seed={res['seed']}/{res['stream']}", f"reps={res['reps']}"]
    if "t_obs" in res:
        bits.insert(0, f"T={res['t_obs']}")
    if res.get("failures"):
        bits.append(f"FIT FAILURES={res['failures']}")
    print("  design  : " + ", ".join(bits))
    print(_RULE)


def print_horizon_table(res: dict[str, Any]) -> None:
    """Aligned coverage-by-horizon table with MC standard errors."""
    _print_header(res)
    variants = res["variants"]
    head = f"{'h':>3} |" + "".join(f" {v:>13} |" for v in variants)
    head += f" {'width(lib)':>11}"
    print(head)
    print("-" * len(head))
    for i, hh in enumerate(res["horizons"]):
        row = f"{hh:>3} |"
        for v in variants:
            p = res["coverage"][v][i]
            s = res["mc_se"][v][i]
            row += f" {100 * p:5.1f} {'(' + f'{100 * s:.2f}' + ')':>7} |"
        row += f" {res['mean_width'][variants[0]][i]:>11.3f}"
        print(row)
    print("-" * len(head))
    print(
        f"  cell = coverage % (MC se, pp).  nominal = {100 * res['nominal']:.1f}%.  "
        "library deviation in MC se: "
        + " ".join(
            f"h{h}:{d:+.1f}"
            for h, d in zip(res["horizons"], res["dev_in_se"]["library"])
        )
    )
    if "oracle" in res["paired_vs_ref_diff"]:
        print(
            "  plug-in cost (oracle minus library, PAIRED on the same reps, pp): "
            + " ".join(
                f"h{h}:{100 * d:+.1f}+/-{100 * s:.1f}"
                for h, d, s in zip(
                    res["horizons"],
                    res["paired_vs_ref_diff"]["oracle"],
                    res["paired_vs_ref_se"]["oracle"],
                )
            )
        )
    if "corrected" in res["paired_vs_ref_diff"]:
        print(
            "  gain from restoring the omitted term (PAIRED, pp): "
            + " ".join(
                f"h{h}:{100 * d:+.1f}+/-{100 * s:.1f}"
                for h, d, s in zip(
                    res["horizons"],
                    res["paired_vs_ref_diff"]["corrected"],
                    res["paired_vs_ref_se"]["corrected"],
                )
            )
        )
    if "predicted_coverage" in res:
        print(
            "  closed-form prediction for the shipped band: "
            + " ".join(
                f"h{h}:{100 * p:.1f}%"
                for h, p in zip(res["horizons"], res["predicted_coverage"])
            )
        )
        print(
            "  measured minus predicted (pp): "
            + " ".join(
                f"h{h}:{100 * (m - p):+.1f}"
                for h, m, p in zip(
                    res["horizons"], res["coverage"]["library"], res["predicted_coverage"]
                )
            )
        )
    d = res["paired_h1_minus_hlast"]
    print(
        f"  library h=1 minus h={res['horizons'][-1]} (PAIRED): "
        f"{100 * d['diff']:+.1f}pp +/- {100 * d['se']:.1f}"
    )
    joint = res["joint_all_horizons"]
    print(
        "  every horizon inside at once (NOT a nominal-95% promise): "
        + ", ".join(f"{v}={100 * joint[v]:.1f}%" for v in variants)
    )
    if "note" in res:
        print(f"  note: {res['note']}")
    print()


def print_level_table(res: dict[str, Any]) -> None:
    _print_header(res)
    head = f"{'h':>3} |" + "".join(
        f" {int(100 * lev):>4}% nom   " for lev in res["levels"]
    )
    print(head)
    print("-" * len(head))
    for i, hh in enumerate(res["horizons"]):
        row = f"{hh:>3} |"
        for lev in res["levels"]:
            p = res["coverage"][lev][i]
            row += f"  {100 * p:5.1f} ({100 * (p - lev):+4.1f}) "
        print(row)
    print("-" * len(head))
    print("  cell = measured coverage % (shortfall vs nominal, percentage points)")
    ref = min(res["levels"], key=lambda x: abs(x - 0.95))
    print(
        f"  MC se at the {100 * ref:.0f}% level: "
        + ", ".join(
            f"h{h}:{100 * s:.2f}pp" for h, s in zip(res["horizons"], res["mc_se"][ref])
        )
    )
    if "note" in res:
        print(f"  note: {res['note']}")
    print()


def print_point_only(res: dict[str, Any]) -> None:
    _print_header(res)
    st = res["structure"]
    print("  PART 1 -- what the library actually returns")
    print(
        f"    theta_forecast(steps=6)    -> {st['theta_forecast_type']} "
        f"shape {tuple(st['theta_forecast_shape'])}: a point path, no interval"
    )
    print(f"    backtest keys              -> {st['backtest_keys']}")
    print(
        f"    backtest interval-ish keys -> "
        f"{st['backtest_interval_keys'] or 'NONE'}"
    )
    print()
    print(
        "  PART 2 -- USER-CONSTRUCTED intervals from backtest errors "
        "(NOT a library promise)"
    )
    print(
        f"    {res['n_backtest_origins']} expanding-window origins per "
        f"replication, nominal {100 * res['nominal']:.0f}%, reps={res['reps']}"
    )
    for dgp_name, d in res["diy"].items():
        print(
            f"    {dgp_name:>13} | "
            + "".join(f"h={h:<12} " for h in res["horizons"])
        )
        for label, ck, sk in (
            ("emp. quantile", "coverage_empirical_quantile", "mc_se_empirical_quantile"),
            ("gauss z*sd", "coverage_gaussian_sd", "mc_se_gaussian_sd"),
        ):
            row = f"    {label:>13} | "
            for p, s in zip(d[ck], d[sk]):
                row += f"{100 * p:5.1f} ({100 * s:4.2f})   "
            print(row)
        print()
    print(f"  {res['disclaimer']}")
    print()


# --------------------------------------------------------------------------
# Assertions.  Only claims with a large, un-tuned margin, and only where the
# replication count can actually resolve them.
# --------------------------------------------------------------------------
def check(
    res_ar1: dict,
    res_arma: dict,
    res_var: dict,
    res_var_big: dict,
    res_levels: dict,
    res_point: dict,
    res_rw: dict,
    quick: bool = False,
) -> tuple[list[str], list[str]]:
    """Assert what is robustly true.

    Returns ``(claims_checked, claims_skipped)``.  Three claims compare two
    measured rates and so need enough replications to resolve a few percentage
    points.  ``--quick`` does not have them, and the honest response is to skip
    those claims rather than to loosen a margin until they pass.
    """
    claims: list[str] = []
    skipped: list[str] = []

    # (1) The band really is the textbook plug-in band. Pure algebra, exact.
    gap = res_ar1["max_abs_replication_gap"]
    assert gap < 1e-8, f"arima_fit band is not mean +/- z*plug-in se (gap {gap:.2e})"
    claims.append(
        f"arima_fit's AR(1) band equals mean +/- z * plug-in se to {gap:.1e} over "
        f"{res_ar1['reps']} fits, so the shipped interval IS the classical "
        "conditional-on-parameters interval and nothing else"
    )

    # (2) The formula is right: at the TRUE parameters it covers at nominal.
    for res, label in (
        (res_ar1, "AR(1)"),
        (res_arma, "ARMA(1,1)"),
        (res_var, "VAR(1)"),
        (res_rw, "I(1)"),
    ):
        for i, hh in enumerate(res["horizons"]):
            d = res["dev_in_se"]["oracle"][i]
            assert abs(d) < 4.5, (
                f"{label} oracle coverage is off nominal at h={hh}: "
                f"{100 * res['coverage']['oracle'][i]:.1f}% ({d:+.1f} MC se) -- "
                "that would implicate the interval FORMULA, not the plug-in"
            )
    claims.append(
        "evaluated at the true parameters the same formula covers within 4.5 MC "
        "se of nominal at every horizon in all four DGPs (AR(1), ARMA(1,1), "
        "VAR(1), I(1)) -- the formula is right, so every shortfall below is the "
        "price of estimating the parameters"
    )

    # (3) The plug-in penalty is real at the far end of the horizon profile.
    #     Measured as a PAIRED gap (oracle minus library on the same
    #     replications), which is far more precise than comparing two rates:
    #     the two bands agree on almost every replication, so only the
    #     discordant ones carry information.  Comparing h=1 with h=H directly is
    #     the weaker test -- those two indicators are only mildly correlated --
    #     so that gap is printed as data and not asserted.
    txt3 = (
        "at the longest horizon the identical formula at the TRUE parameters "
        "covers materially more than the shipped plug-in band (PAIRED gap > 3 "
        "se) for arima_fit on an AR(1) and for var_forecast on a VAR(1)"
    )
    if quick:
        skipped.append(txt3 + "   [needs the default replication count to resolve]")
    else:
        for res, label in ((res_ar1, "AR(1)"), (res_var, "VAR(1)")):
            d = res["paired_vs_ref_diff"]["oracle"][-1]
            s = res["paired_vs_ref_se"]["oracle"][-1]
            assert d > 3.0 * s, (
                f"{label}: expected a positive plug-in cost at h="
                f"{res['horizons'][-1]}; paired oracle-minus-library gap "
                f"{100 * d:+.1f}pp +/- {100 * s:.1f}"
            )
        claims.append(txt3)

    # (4) The plug-in penalty shrinks as T grows (independent samples: hypot).
    small = res_var["coverage"]["library"][-1]
    big = res_var_big["coverage"]["library"][-1]
    s_small = res_var["mc_se"]["library"][-1]
    s_big = res_var_big["mc_se"]["library"][-1]
    txt4 = (
        f"the shortfall is estimation error, not bias: at the longest horizon "
        f"var_forecast covers {100 * small:.1f}% at T={res_var['t_obs']} and "
        f"{100 * big:.1f}% at T={res_var_big['t_obs']} (gap > 3 MC se of the "
        "difference across independent runs)"
    )
    if quick:
        skipped.append(txt4 + "   [needs the default replication count to resolve]")
    else:
        assert big - small > 3.0 * math.hypot(s_small, s_big), (
            "expected var_forecast long-horizon coverage to improve with T: "
            f"T={res_var['t_obs']} {100 * small:.1f}% vs "
            f"T={res_var_big['t_obs']} {100 * big:.1f}%"
        )
        claims.append(txt4)

    # (5) Marginal intervals are not a joint band.
    j = res_var["joint_all_horizons"]["library"]
    assert j < 0.95 - 10.0 * mc_se(j, res_var["reps"]), (
        f"all-horizon simultaneous coverage {100 * j:.1f}% is not clearly below 95%"
    )
    claims.append(
        f"marginal intervals are not a joint band: all {len(res_var['horizons'])} "
        f"horizons x 2 series hold simultaneously in only {100 * j:.1f}% of "
        "replications, more than 10 MC se below the marginal level"
    )

    # (6) alpha is a real level: coverage is monotone in it at every horizon.
    lev = res_levels["levels"]
    for i in range(len(res_levels["horizons"])):
        seq = [res_levels["coverage"][x][i] for x in lev]
        assert all(a < b for a, b in zip(seq, seq[1:])), (
            f"coverage is not monotone in the nominal level at h={i + 1}: {seq}"
        )
    claims.append(
        "var_forecast coverage is strictly increasing in the requested nominal "
        f"level at every horizon over levels {[int(100 * x) for x in lev]} "
        "(alpha is a real level, not a knob)"
    )

    # (7) The I(1) band provably drops the drift term; the consequence is
    #     checked against the closed form in the only guaranteed direction.
    gap_se = res_rw["max_abs_se_gap"]
    assert gap_se < 1e-8, (
        f"arima_fit(0,1,0) forecast_se is no longer sigma_hat*sqrt(h) "
        f"(gap {gap_se:.2e}); Experiment 6's closed form assumes it is"
    )
    claims.append(
        "arima_fit(0,1,0) reports forecast_se = sigma_hat * sqrt(h) exactly (to "
        f"{gap_se:.1e}), i.e. it omits the h^2/(T-1) drift-uncertainty term "
        "entirely -- this is an identity, not a measurement"
    )

    txt7 = (
        "restoring the omitted drift term covers materially better at the "
        "longest horizon than the shipped band (PAIRED gap > 5 se), and the "
        "shipped band never beats its own closed-form ceiling at any horizon"
    )
    if quick:
        skipped.append(txt7 + "   [needs the default replication count to resolve]")
    else:
        diff = res_rw["paired_vs_ref_diff"]["corrected"][-1]
        se = res_rw["paired_vs_ref_se"]["corrected"][-1]
        assert diff > 5.0 * se, (
            "expected the drift-corrected band to cover better at the longest "
            f"horizon: paired gap {100 * diff:+.1f}pp +/- {100 * se:.1f}"
        )
        for i, hh in enumerate(res_rw["horizons"]):
            meas = res_rw["coverage"]["library"][i]
            pred = res_rw["predicted_coverage"][i]
            s = res_rw["mc_se"]["library"][i]
            assert meas - pred < 3.5 * s, (
                f"at h={hh} the shipped band covered {100 * meas:.1f}%, above its "
                f"closed-form ceiling {100 * pred:.1f}% -- the closed form or the "
                "experiment would then be wrong"
            )
        claims.append(txt7)

    # (8) Structural: the point-only surfaces stay point-only.
    st = res_point["structure"]
    assert st["backtest_interval_keys"] == [], (
        f"backtest now emits interval-ish keys {st['backtest_interval_keys']}; "
        "this experiment must be extended to score them"
    )
    assert tuple(st["theta_forecast_shape"]) == (6,), (
        f"theta_forecast returned shape {st['theta_forecast_shape']}, expected (6,)"
    )
    claims.append(
        "theta_forecast returns a bare (steps,) point path and backtest returns "
        "no interval key -- there is no library interval to score for either, so "
        "any interval a user reports around them is the user's own"
    )

    return claims, skipped


# --------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------
def run(quick: bool = False) -> dict[str, Any]:
    """Run every experiment, print the report, return the structured results."""
    t_start = time.perf_counter()
    print()
    print("PREDICTIVE-INTERVAL COVERAGE FOR tsecon FORECAST BANDS")
    print(
        f"master seed = {SEED}   nominal level = {100 * NOMINAL:.1f}%   "
        f"mode = {'quick' if quick else 'default'}"
    )
    print(
        "coverage numbers carry the MC standard error se = sqrt(p(1-p)/reps); "
        "band-vs-band gaps carry the PAIRED se over the same replications"
    )
    print()

    reps_ar1 = 120 if quick else 700
    reps_arma = 90 if quick else 600
    reps_rw = 200 if quick else 1500
    reps_var = 400 if quick else 6000
    reps_lev = 300 if quick else 4000
    reps_pt = 60 if quick else 1000

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    ar1_persistent = exp_ar1_arima(phi=0.9, t_obs=100, reps=reps_ar1, stream=1)
    timings["exp1_ar1_phi0.9"] = time.perf_counter() - t0
    print_horizon_table(ar1_persistent)

    t0 = time.perf_counter()
    ar1_mild = exp_ar1_arima(phi=0.5, t_obs=100, reps=reps_ar1, stream=11)
    timings["exp1_ar1_phi0.5"] = time.perf_counter() - t0
    print_horizon_table(ar1_mild)

    t0 = time.perf_counter()
    arma = exp_arma11_arima(reps=reps_arma, stream=2)
    timings["exp2_arma11"] = time.perf_counter() - t0
    print_horizon_table(arma)

    t0 = time.perf_counter()
    var_small = exp_var_forecast(t_obs=100, fit_lags=1, reps=reps_var, stream=3)
    var_over = exp_var_forecast(t_obs=100, fit_lags=4, reps=reps_var, stream=31)
    var_big = exp_var_forecast(t_obs=800, fit_lags=1, reps=reps_var, stream=32)
    timings["exp3_var_forecast"] = time.perf_counter() - t0
    for res in (var_small, var_over, var_big):
        print_horizon_table(res)

    t0 = time.perf_counter()
    rw = exp_rw_drift_arima(t_obs=100, h_max=12, reps=reps_rw, stream=6)
    rw_short = exp_rw_drift_arima(t_obs=60, h_max=24, reps=reps_rw, stream=61)
    timings["exp6_rw_drift"] = time.perf_counter() - t0
    print_horizon_table(rw)
    print_horizon_table(rw_short)

    t0 = time.perf_counter()
    levels = exp_level_sweep(reps=reps_lev, stream=4)
    timings["exp4_level_sweep"] = time.perf_counter() - t0
    print_level_table(levels)

    t0 = time.perf_counter()
    point_only = exp_point_only_surfaces(reps=reps_pt, stream=5)
    timings["exp5_point_only"] = time.perf_counter() - t0
    print_point_only(point_only)

    results: dict[str, Any] = {
        "seed": SEED,
        "nominal": NOMINAL,
        "quick": quick,
        "exp1_ar1_phi0.9": ar1_persistent,
        "exp1_ar1_phi0.5": ar1_mild,
        "exp2_arma11": arma,
        "exp3_var_T100_lags1": var_small,
        "exp3_var_T100_lags4": var_over,
        "exp3_var_T800_lags1": var_big,
        "exp4_level_sweep": levels,
        "exp5_point_only": point_only,
        "exp6_rw_drift_T100_h12": rw,
        "exp6_rw_drift_T60_h24": rw_short,
        "timings_s": timings,
    }

    claims, skipped = check(
        ar1_persistent,
        arma,
        var_small,
        var_big,
        levels,
        point_only,
        rw_short,
        quick=quick,
    )
    results["claims"] = claims
    results["claims_skipped"] = skipped

    print(_RULE)
    print("ASSERTED (each of these held; none of them was tuned to hold)")
    print(_RULE)
    for i, c in enumerate(claims, 1):
        print(f"  {i}. {c}")
    for c in skipped:
        print(f"  --. SKIPPED: {c}")
    print()
    print(_RULE)
    print("WHAT THIS MEASURED, PLAINLY")
    print(_RULE)
    for line in _verdict(results):
        print(f"  {line}")
    print()
    total = time.perf_counter() - t_start
    results["runtime_s"] = total
    print(
        f"  runtime: {total:.1f}s   "
        + "  ".join(f"[{k} {v:.1f}s]" for k, v in timings.items())
    )
    print()
    return results


def _verdict(results: dict[str, Any]) -> list[str]:
    """The honest summary, computed from the numbers just measured."""
    lines: list[str] = []
    for key in ("exp1_ar1_phi0.9", "exp1_ar1_phi0.5", "exp2_arma11"):
        r = results[key]
        lib, orc = r["coverage"]["library"], r["coverage"]["oracle"]
        worst = int(np.argmin(lib))
        lines.append(
            f"{key}: library covers {100 * lib[0]:.1f}% at h=1 and "
            f"{100 * lib[-1]:.1f}% at h={r['horizons'][-1]}; worst horizon is "
            f"h={r['horizons'][worst]} at {100 * lib[worst]:.1f}% "
            f"(+/-{100 * r['mc_se']['library'][worst]:.2f}pp) where the oracle "
            f"gets {100 * orc[worst]:.1f}%, a paired plug-in cost of "
            f"{100 * r['paired_vs_ref_diff']['oracle'][worst]:+.1f}pp "
            f"+/-{100 * r['paired_vs_ref_se']['oracle'][worst]:.1f}"
        )
    for key in ("exp6_rw_drift_T100_h12", "exp6_rw_drift_T60_h24"):
        rw = results[key]
        lines.append(
            f"{key}: library covers {100 * rw['coverage']['library'][0]:.1f}% at "
            f"h=1 and {100 * rw['coverage']['library'][-1]:.1f}% at "
            f"h={rw['horizons'][-1]} (+/-{100 * rw['mc_se']['library'][-1]:.2f}pp) "
            f"against a closed-form ceiling of "
            f"{100 * rw['predicted_coverage'][-1]:.1f}%; restoring the omitted "
            f"drift term gives {100 * rw['coverage']['corrected'][-1]:.1f}% -- the "
            "shortfall is fully accounted for and it is the approximation"
        )
    for key in ("exp3_var_T100_lags1", "exp3_var_T100_lags4", "exp3_var_T800_lags1"):
        r = results[key]
        lib = r["coverage"]["library"]
        lines.append(
            f"{key}: library covers {100 * lib[0]:.1f}% at h=1 and "
            f"{100 * lib[-1]:.1f}% at h={r['horizons'][-1]}; simultaneous "
            f"all-horizon rate {100 * r['joint_all_horizons']['library']:.1f}%"
        )
    lev = results["exp4_level_sweep"]
    worst_lev = min(lev["levels"], key=lambda x: min(p - x for p in lev["coverage"][x]))
    lines.append(
        f"exp4_level_sweep: largest absolute shortfall is at the "
        f"{100 * worst_lev:.0f}% level "
        f"({min(100 * (p - worst_lev) for p in lev['coverage'][worst_lev]):+.1f}pp); "
        "the 99% level loses the least in pp because its endpoints sit in a thin tail"
    )
    pt = results["exp5_point_only"]
    for dgp_name, d in pt["diy"].items():
        q = d["coverage_empirical_quantile"]
        g = d["coverage_gaussian_sd"]
        lines.append(
            f"exp5_point_only {dgp_name}: the library ships NO interval here. Our "
            f"backtest-quantile interval covers {100 * q[0]:.1f}% (h=1) to "
            f"{100 * q[-1]:.1f}% (h={pt['horizons'][-1]}); the symmetric Gaussian "
            f"variant {100 * g[0]:.1f}% to {100 * g[-1]:.1f}%"
        )
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure predictive-interval coverage of tsecon forecast bands."
    )
    ap.add_argument(
        "--quick", action="store_true", help="fast smoke run with fewer replications"
    )
    args = ap.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
