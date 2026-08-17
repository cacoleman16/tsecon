"""Score-driven (DCS) robust local level: Student-t and Laplace observation
densities, with the Gaussian local level as the nested control.

Model
-----
Observation: y_t = mu_t + eps_t, eps_t iid with density p(.; scale, [nu]).
Level recursion driven by the conditional score of the observation
density (Creal-Koopman-Lucas 2013 GAS; Harvey 2013 DCS; the local-level
case is Harvey & Luati 2014, "Filtering with heavy tails", JASA 109):

    e_t     = y_t - mu_t                        (prediction error)
    u_t     = scale^2 * d log p(y_t | mu_t) / d mu_t
    mu_{t+1} = mu_t + kappa * u_t,   kappa >= 0.

The common scaling u_t = scale^2 * score (a fixed multiple of the
inverse-information scaling, absorbed into kappa) gives:

- Gaussian:  log p = -.5 log(2 pi s^2) - e^2/(2 s^2)
             score = e/s^2            ->  u_t = e_t
             mu_{t+1} = mu_t + kappa e_t  — EXACTLY the steady-state
             (innovations-form) Kalman filter for the Gaussian local
             level with constant gain kappa.  This is the nested
             control: the DCS-t driver u_t = (nu+1) e /(nu + e^2/s^2)
             -> e_t as nu -> inf.
- Student-t: log p = lgamma((nu+1)/2) - lgamma(nu/2) - .5 log(nu pi s^2)
                     - (nu+1)/2 log(1 + e^2/(nu s^2))
             score = (nu+1) e / (nu s^2 + e^2)
             u_t   = (nu+1) e_t / (nu + e_t^2/s^2).
             u is redescending in e: a huge outlier moves the level by
             ~ kappa (nu+1) s^2 / e -> 0.  This is the DCS-t local level
             of Harvey & Luati (2014) — the robust alternative to the
             Gaussian Kalman local level under additive outliers.
- Laplace:   log p = -log(2 b) - |e|/b
             score = sign(e)/b       ->  u_t = b * sign(e_t)
             mu_{t+1} = mu_t + kappa b sign(e_t): a bounded, constant-
             magnitude "sign filter" step — the level tracks a local
             MEDIAN.  Laplace is GED(1); the sign driver is the DCS-GED
             case at nu_GED = 1 (Harvey 2013, ch. 3).

Estimation
----------
MLE by prediction-error decomposition: sum_t log p(y_t | mu_t) with
mu_1 = median of the first 10 observations (robust initialisation; the
first ``burn_loglik`` terms can be excluded from the criterion to reduce
initialisation effects — default 0, i.e. exact conditional likelihood
given mu_1).  Parameters (kappa, scale, [nu]) via L-BFGS-B: kappa box-
bounded >= 0, scale and nu on the log scale (nu in [0.8, 200]), small
deterministic multi-start over kappa.

Honest notes / simplifications
------------------------------
* The Laplace likelihood is piecewise smooth in kappa (sign flips
  propagate); by default sign(e) is smoothed as tanh(e/h) with a fixed
  data-driven h = 0.1 * MAD(diff(y))/ (sqrt(2)*0.6745) for optimiser
  stability (``smooth=0`` gives the hard sign; fitting and filtering use
  the same h, so the reported path is the exact filter of the estimated
  model).
* mu_t here is the PREDICTIVE level E-type estimate given y_{1:t-1};
  there is no smoother (the DCS literature filters; smoothing exists but
  is not implemented).
* Only the level is time-varying: constant scale (no DCS volatility) and
  no slope/seasonal components.
* Gaussian nesting is at the STEADY-STATE filter: the exact Kalman
  filter has a time-varying gain during its transient, so on finite
  clean samples the DCS-Gaussian MLE gain matches the steady-state gain
  implied by the Kalman local-level MLE up to O(transient) differences —
  verified numerically in tests.py against statsmodels
  UnobservedComponents / tsecon.local_level_smooth.

``steady_state_gain(sigma2_eta, sigma2_eps)`` converts local-level
variance estimates to the steady-state Kalman gain
    p = (s + sqrt(s^2 + 4 s))/2,  s = sigma2_eta/sigma2_eps,
    gain = p/(1 + p),
for the nesting check.

References
----------
- Creal, Koopman & Lucas (2013), JAE 28(5).
- Harvey (2013), "Dynamic Models for Volatility and Heavy Tails", CUP.
- Harvey & Luati (2014), "Filtering with Heavy Tails", JASA 109(507),
  1112-1122.  (DCS-t local level.)
- Durbin & Koopman (2012), "Time Series Analysis by State Space
  Methods", 2nd ed. (steady-state Kalman filter for the local level).

Provenance: implemented from the published literature above; no
proprietary code consulted.  Research code, not part of the public
tsecon API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

_LOG2PI = np.log(2.0 * np.pi)


def steady_state_gain(sigma2_eta: float, sigma2_eps: float) -> float:
    """Steady-state Kalman gain of the Gaussian local level.

    Solves the Riccati fixed point P = P - P^2/(P + s_eps^2) + s_eta^2:
    with q = s_eta^2/s_eps^2, p = P/s_eps^2 = (q + sqrt(q^2 + 4q))/2 and
    the predictive-recursion gain is K = p/(1+p), i.e.
    mu_{t+1|t} = mu_{t|t-1} + K (y_t - mu_{t|t-1}).
    """
    q = sigma2_eta / sigma2_eps
    p = 0.5 * (q + np.sqrt(q * q + 4.0 * q))
    return float(p / (1.0 + p))


def _robust_scale_diff(y: np.ndarray) -> float:
    """MAD-based scale of the innovation-ish first differences."""
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    s = mad / 0.6745 / np.sqrt(2.0)
    return float(s) if s > 0 else float(np.std(y) + 1e-12)


def dcs_filter(
    y: np.ndarray,
    density: str,
    kappa: float,
    scale: float,
    nu: float | None = None,
    mu0: float | None = None,
    smooth_h: float = 0.0,
):
    """Run the DCS local-level filter; returns (mu, loglik_terms).

    mu[t] is the one-step-ahead (predictive) level for y_t.
    """
    y = np.asarray(y, float)
    T = y.shape[0]
    mu = np.empty(T)
    ll = np.empty(T)
    m = float(np.median(y[: min(10, T)])) if mu0 is None else float(mu0)
    s2 = scale * scale
    if density == "gaussian":
        c = -0.5 * (_LOG2PI + np.log(s2))
        for t in range(T):
            mu[t] = m
            e = y[t] - m
            ll[t] = c - 0.5 * e * e / s2
            m = m + kappa * e
    elif density == "t":
        c = (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
             - 0.5 * np.log(nu * np.pi * s2))
        for t in range(T):
            mu[t] = m
            e = y[t] - m
            z2 = e * e / s2
            ll[t] = c - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)
            u = (nu + 1.0) * e / (nu + z2)
            m = m + kappa * u
    elif density == "laplace":
        c = -np.log(2.0 * scale)
        for t in range(T):
            mu[t] = m
            e = y[t] - m
            ll[t] = c - abs(e) / scale
            sgn = np.tanh(e / smooth_h) if smooth_h > 0.0 else np.sign(e)
            m = m + kappa * scale * sgn
        # NB: driver u = b * sign(e) = scale^2 * score
    else:
        raise ValueError("density must be 'gaussian', 't' or 'laplace'")
    return mu, ll


@dataclass
class DCSResult:
    density: str
    kappa: float
    scale: float
    nu: float | None
    mu: np.ndarray = field(repr=False)
    loglik: float
    aic: float
    bic: float
    converged: bool
    n_obs: int
    smooth_h: float


def fit_dcs_local_level(
    y: np.ndarray,
    density: str = "t",
    smooth: float = 0.1,
    burn_loglik: int = 0,
    maxiter: int = 300,
) -> DCSResult:
    """MLE of the DCS local level with Gaussian / Student-t / Laplace errors.

    Parameters
    ----------
    y : (T,) series.
    density : 'gaussian' (nested steady-state Kalman control), 't'
        (Harvey-Luati DCS-t), or 'laplace' (sign filter).
    smooth : Laplace only — sign(e) ~ tanh(e/h) with
        h = smooth * MAD-scale(diff y); 0 -> hard sign.
    burn_loglik : number of initial likelihood terms to drop (default 0).
    """
    y = np.asarray(y, float)
    T = y.shape[0]
    if T < 30:
        raise ValueError("need at least 30 observations")
    s_rob = _robust_scale_diff(y)
    h = smooth * s_rob if density == "laplace" else 0.0

    def unpack(psi):
        if density == "gaussian":
            return psi[0], np.exp(psi[1]), None
        if density == "laplace":
            return psi[0], np.exp(psi[1]), None
        return psi[0], np.exp(psi[1]), np.exp(psi[2])

    def neg(psi):
        kap, sc, nu = unpack(psi)
        _, ll = dcs_filter(y, density, kap, sc, nu, smooth_h=h)
        return -np.sum(ll[burn_loglik:]) / T

    bounds = [(0.0, 5.0), (np.log(1e-8 + s_rob * 1e-4), np.log(s_rob * 1e4))]
    if density == "t":
        bounds.append((np.log(0.8), np.log(200.0)))

    starts = []
    for k0 in (0.05, 0.3, 0.8):
        psi = [k0, np.log(s_rob)]
        if density == "t":
            psi.append(np.log(8.0))
        starts.append(np.array(psi))

    best = None
    for psi0 in starts:
        res = minimize(neg, psi0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    kap, sc, nu = unpack(best.x)
    mu, ll = dcs_filter(y, density, kap, sc, nu, smooth_h=h)
    loglik = float(np.sum(ll[burn_loglik:]))
    npar = 3 if density == "t" else 2
    n_used = T - burn_loglik
    return DCSResult(
        density=density, kappa=float(kap), scale=float(sc),
        nu=(float(nu) if nu is not None else None), mu=mu, loglik=loglik,
        aic=2.0 * npar - 2.0 * loglik,
        bic=np.log(n_used) * npar - 2.0 * loglik,
        converged=bool(best.success), n_obs=T, smooth_h=h,
    )


def simulate_local_level(
    T: int,
    sigma_eta: float,
    sigma_eps: float,
    outlier_frac: float = 0.0,
    outlier_size: float = 8.0,
    seed: int = 0,
):
    """Gaussian local level, optionally contaminated with additive outliers.

    Outliers are added to the OBSERVATION only (additive outliers), the
    level path is clean.  Returns (y, mu_true, outlier_mask).
    """
    rng = np.random.default_rng(seed)
    mu = np.cumsum(rng.normal(0.0, sigma_eta, T))
    y = mu + rng.normal(0.0, sigma_eps, T)
    mask = np.zeros(T, bool)
    if outlier_frac > 0.0:
        idx = rng.choice(T, size=int(round(outlier_frac * T)), replace=False)
        mask[idx] = True
        y[idx] += rng.choice([-1.0, 1.0], idx.size) * outlier_size * sigma_eps
    return y, mu, mask


def _demo():
    y, mu_true, _ = simulate_local_level(
        800, sigma_eta=0.1, sigma_eps=1.0, outlier_frac=0.07,
        outlier_size=9.0, seed=11,
    )
    for d in ("gaussian", "t", "laplace"):
        r = fit_dcs_local_level(y, d)
        rmse = float(np.sqrt(np.mean((r.mu - mu_true) ** 2)))
        extra = f" nu={r.nu:.2f}" if r.nu is not None else ""
        print(f"{d:8s} kappa={r.kappa:.4f} scale={r.scale:.4f}{extra} "
              f"loglik={r.loglik:.1f}  level-RMSE={rmse:.4f}")


if __name__ == "__main__":
    _demo()
