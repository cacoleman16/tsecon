"""Score-driven (GAS/DCS) time-varying quantiles with an asymmetric-Laplace
working likelihood.

Model
-----
Observation working density: asymmetric Laplace (AL) at quantile level
``tau`` with time-varying location ``q_t`` (the conditional tau-quantile)
and static scale ``sigma``::

    f(y | q_t, sigma, tau) = tau (1 - tau) / sigma * exp( -rho_tau(y - q_t) / sigma )
    rho_tau(u) = u (tau - 1{u < 0})              (Koenker "check"/pinball loss)

Log-density and its score with respect to the location::

    l_t          = log(tau(1-tau)) - log sigma - rho_tau(y_t - q_t) / sigma
    d l_t / d q  = (tau - 1{y_t <= q_t}) / sigma        (a.e.)

The AL location score is *bounded* (it only depends on which side of q_t
the observation falls), which is exactly what makes the filter robust.
The conditional Fisher information w.r.t. q is E[(tau - 1)^2]/sigma^2 =
tau(1-tau)/sigma^2, a constant, so any inverse-information scaling of the
score (Creal-Koopman-Lucas 2013 choices S = I^{-1}, I^{-1/2}, Id) is a
constant multiple and is absorbed into the loading ``a``.  We therefore
drive the recursion with the unscaled indicator score

    s_t = tau - 1{y_t <= q_t}            in  [tau - 1, tau],

and the GAS(1,1)-type updating equation

    q_{t+1} = omega + b q_t + a s_t,     |b| < 1,  a >= 0.

With a > 0 a "violation" (y_t <= q_t for a low quantile) pushes q down,
otherwise q drifts up by a*tau: this is precisely the *adaptive* CAViaR
of Engle & Manganelli (2004) with an added mean-reversion term, obtained
here as the score-driven (DCS) recursion under the AL working density.

Estimation
----------
MLE of (omega, a, b) with the AL scale ``sigma`` profiled out in closed
form: given the path q_t(psi),

    sigma_hat(psi) = (1/T) sum_t rho_tau(y_t - q_t(psi))
    profile loglik = T [ log(tau(1-tau)) - log sigma_hat(psi) - 1 ].

Maximising the profile likelihood is therefore *identical* to minimising
average pinball loss, the (elicitable) scoring rule for quantiles
(Koenker & Machado 1999; Gneiting 2011).  This makes explicit that the AL
density is a working likelihood: consistency for the true conditional
quantile does not require the data to be AL distributed (same logic as
quantile regression, Yu & Moyeed 2001, and CAViaR).  ``sigma`` can also
be estimated jointly (``profile_scale=False``); the argmax is the same by
construction, the option exists as a numerical cross-check.

Optimisation detail (honest notes)
----------------------------------
* The indicator makes the criterion piecewise-constant-in-jumps as psi
  varies (a step in s_t propagates through the whole future path), so
  quasi-Newton optimisers can stall on kinks.  We optionally smooth the
  indicator with a logistic cdf, 1{y<=q} ~ expit((q-y)/h), with a small
  data-driven bandwidth h (default 0.05 * IQR(y)); ``bandwidth=0`` gives
  the pure indicator model.  This is the standard smoothed-CAViaR device
  (cf. the smoothed criterion in Engle & Manganelli's estimation practice
  and kernel-smoothed quantile objectives).  Fitting and filtering use
  the *same* h, so the reported path is the exact filter of the model
  actually estimated.
* |b| < 1 is enforced by the reparameterisation b = tanh(b_raw); a >= 0
  by an L-BFGS-B box bound; a small deterministic multi-start grid guards
  against local optima.
* q_1 is initialised at the empirical tau-quantile of the first
  max(25, T//10) observations (no full-sample look-ahead).

Multi-tau fitting
-----------------
``fit_al_gas_multi`` fits each tau separately (one AL working likelihood
per tau, as in per-tau quantile regression) and reports the fraction of
time points where fitted quantile paths cross.  Optionally the paths are
monotonised by pointwise rearrangement (sorting across tau at each t),
following Chernozhukov, Fernandez-Val & Galichon (2010, Ecta).  Joint
non-crossing estimation is NOT implemented — that is a real
simplification relative to joint dynamic-quantile models.

References
----------
- Creal, Koopman & Lucas (2013), "Generalized Autoregressive Score models
  with applications", J. Applied Econometrics 28(5).
- Harvey (2013), "Dynamic Models for Volatility and Heavy Tails", CUP.
- Engle & Manganelli (2004), "CAViaR: Conditional autoregressive Value at
  Risk by regression quantiles", JBES 22(4).  (Adaptive CAViaR = this
  driver.)
- Koenker & Machado (1999, JASA); Yu & Moyeed (2001, Stat. & Prob.
  Letters): the AL working-likelihood <-> quantile-loss equivalence.
- Catania & Luati, "Semiparametric modeling of multiple quantiles"
  (Journal of Econometrics, 2022; earlier drafts circulated ~2019) study
  score-driven dynamic quantiles of this type, including multi-quantile
  systems.  What is implemented here is the single-quantile AL-score
  recursion with per-tau estimation, i.e. a simplified member of that
  family, not a replication of their joint estimator.
- Chernozhukov, Fernandez-Val & Galichon (2010, Econometrica): quantile
  rearrangement.

Provenance: implemented from the published literature above; no
proprietary code consulted.  Research code, not part of the public
tsecon API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def pinball_loss(u: np.ndarray, tau: float) -> np.ndarray:
    """rho_tau(u) = u * (tau - 1{u<0}), elementwise."""
    u = np.asarray(u, float)
    return u * (tau - (u < 0.0))


def _default_bandwidth(y: np.ndarray) -> float:
    q75, q25 = np.percentile(y, [75.0, 25.0])
    iqr = q75 - q25
    if iqr <= 0.0:
        iqr = np.std(y) + 1e-12
    return 0.05 * iqr


def al_gas_filter(
    y: np.ndarray,
    tau: float,
    omega: float,
    a: float,
    b: float,
    q0: float,
    bandwidth: float = 0.0,
):
    """Run the AL score-driven quantile recursion.

    Returns
    -------
    q : (T,) array — filtered quantile path, q[t] = quantile of y_t given
        y_{1:t-1} (one-step-ahead, no look-ahead beyond q0's init window).
    q_next : float — one-step-ahead forecast for t = T+1.
    """
    y = np.asarray(y, float)
    T = y.shape[0]
    q = np.empty(T)
    qt = float(q0)
    h = float(bandwidth)
    if h > 0.0:
        for t in range(T):
            q[t] = qt
            s = tau - expit((qt - y[t]) / h)
            qt = omega + b * qt + a * s
    else:
        for t in range(T):
            q[t] = qt
            s = tau - (1.0 if y[t] <= qt else 0.0)
            qt = omega + b * qt + a * s
    return q, qt


@dataclass
class ALGASResult:
    tau: float
    omega: float
    a: float
    b: float
    sigma: float
    q: np.ndarray = field(repr=False)
    q_next: float
    loglik: float
    avg_pinball: float
    hit_rate: float
    bandwidth: float
    converged: bool
    n_obs: int

    @property
    def aic(self) -> float:
        # omega, a, b + profiled sigma
        return 2.0 * 4 - 2.0 * self.loglik

    @property
    def bic(self) -> float:
        return np.log(self.n_obs) * 4 - 2.0 * self.loglik


def _al_loglik_from_path(y, q, tau, sigma=None):
    """AL log-likelihood of a quantile path; profiles sigma if not given."""
    rho = pinball_loss(y - q, tau)
    T = y.shape[0]
    if sigma is None:
        sigma = float(np.mean(rho))
        if sigma <= 0.0:
            sigma = 1e-12
        ll = T * (np.log(tau * (1.0 - tau)) - np.log(sigma) - 1.0)
        return ll, sigma
    ll = T * np.log(tau * (1.0 - tau) / sigma) - np.sum(rho) / sigma
    return ll, sigma


def fit_al_gas(
    y: np.ndarray,
    tau: float,
    bandwidth: float | None = None,
    profile_scale: bool = True,
    maxiter: int = 300,
) -> ALGASResult:
    """MLE of the AL score-driven quantile model at level ``tau``.

    Parameters
    ----------
    y : (T,) series.
    tau : quantile level in (0, 1).
    bandwidth : logistic smoothing bandwidth for the indicator score;
        None -> 0.05 * IQR(y) (default), 0.0 -> hard indicator.
    profile_scale : profile sigma in closed form (default) or estimate it
        jointly on the log scale (numerical cross-check; same argmax).
    """
    y = np.asarray(y, float)
    T = y.shape[0]
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0,1)")
    if T < 50:
        raise ValueError("need at least 50 observations")
    h = _default_bandwidth(y) if bandwidth is None else float(bandwidth)

    n0 = max(25, T // 10)
    q0 = float(np.quantile(y[:n0], tau))
    q_unc = float(np.quantile(y, tau))
    scale_y = float(np.std(y)) + 1e-12

    def path(psi):
        omega, a, b_raw = psi[0], psi[1], psi[2]
        b = np.tanh(b_raw)
        return al_gas_filter(y, tau, omega, a, b, q0, h)

    if profile_scale:
        def neg(psi):
            q, _ = path(psi)
            ll, _ = _al_loglik_from_path(y, q, tau, sigma=None)
            return -ll / T
        nfree = 3
    else:
        def neg(psi):
            q, _ = path(psi[:3])
            ll, _ = _al_loglik_from_path(y, q, tau, sigma=np.exp(psi[3]))
            return -ll / T
        nfree = 4

    # deterministic multi-start grid
    starts = []
    for b0 in (0.90, 0.97):
        for a0 in (0.05 * scale_y, 0.25 * scale_y):
            psi = [q_unc * (1.0 - b0), a0, np.arctanh(b0)]
            if not profile_scale:
                psi.append(np.log(np.mean(pinball_loss(y - q_unc, tau)) + 1e-12))
            starts.append(np.array(psi))

    bounds = [(None, None), (0.0, None), (-6.0, 6.0)]
    if not profile_scale:
        bounds.append((None, None))

    best = None
    for psi0 in starts:
        res = minimize(neg, psi0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter})
        if best is None or res.fun < best.fun:
            best = res

    omega, a, b_raw = best.x[0], best.x[1], best.x[2]
    b = float(np.tanh(b_raw))
    q, q_next = al_gas_filter(y, tau, omega, a, b, q0, h)
    if profile_scale:
        ll, sigma = _al_loglik_from_path(y, q, tau, sigma=None)
    else:
        sigma = float(np.exp(best.x[3]))
        ll, _ = _al_loglik_from_path(y, q, tau, sigma=sigma)
    return ALGASResult(
        tau=tau, omega=float(omega), a=float(a), b=b, sigma=float(sigma),
        q=q, q_next=float(q_next), loglik=float(ll),
        avg_pinball=float(np.mean(pinball_loss(y - q, tau))),
        hit_rate=float(np.mean(y <= q)), bandwidth=h,
        converged=bool(best.success), n_obs=T,
    )


@dataclass
class ALGASMultiResult:
    taus: tuple
    results: dict            # tau -> ALGASResult
    q_matrix: np.ndarray = field(repr=False)      # (T, K) raw per-tau paths
    q_monotone: np.ndarray = field(repr=False)    # (T, K) rearranged paths
    crossing_frac: float     # fraction of t with any adjacent-pair crossing (raw)


def fit_al_gas_multi(
    y: np.ndarray,
    taus,
    bandwidth: float | None = None,
    rearrange: bool = True,
) -> ALGASMultiResult:
    """Fit AL-GAS quantile models at several tau levels (independently).

    Non-crossing is *checked*, not imposed, at the estimation stage;
    ``q_monotone`` applies the Chernozhukov-Fernandez-Val-Galichon (2010)
    pointwise rearrangement (sort across tau at each t), which is proven
    to weakly improve every quantile estimate in finite samples.
    """
    taus = tuple(sorted(float(t) for t in taus))
    results = {t: fit_al_gas(y, t, bandwidth=bandwidth) for t in taus}
    Q = np.column_stack([results[t].q for t in taus])
    crossing = np.any(np.diff(Q, axis=1) < 0.0, axis=1)
    Qm = np.sort(Q, axis=1) if rearrange else Q.copy()
    return ALGASMultiResult(
        taus=taus, results=results, q_matrix=Q, q_monotone=Qm,
        crossing_frac=float(np.mean(crossing)),
    )


def _demo(seed: int = 7):
    """GARCH(1,1) DGP with a known conditional-quantile path."""
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    T = 3000
    om, al, be = 0.02, 0.10, 0.88
    sig2 = np.empty(T)
    y = np.empty(T)
    s2 = om / (1 - al - be)
    for t in range(T):
        sig2[t] = s2
        y[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + al * y[t] ** 2 + be * s2
    tau = 0.05
    q_true = np.sqrt(sig2) * norm.ppf(tau)

    r = fit_al_gas(y, tau)
    rmse_gas = float(np.sqrt(np.mean((r.q - q_true) ** 2)))
    q_static = np.quantile(y, tau)
    rmse_static = float(np.sqrt(np.mean((q_static - q_true) ** 2)))
    print(f"tau={tau}  omega={r.omega:.4f} a={r.a:.4f} b={r.b:.4f} "
          f"sigma={r.sigma:.4f}  hit={r.hit_rate:.4f} loglik={r.loglik:.1f}")
    print(f"tracking RMSE  AL-GAS={rmse_gas:.4f}  static={rmse_static:.4f} "
          f"ratio={rmse_gas / rmse_static:.3f}")

    m = fit_al_gas_multi(y, (0.05, 0.25, 0.5, 0.75, 0.95))
    print(f"multi-tau crossing_frac (raw paths) = {m.crossing_frac:.4f}")


if __name__ == "__main__":
    _demo()
