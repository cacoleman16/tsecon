#!/usr/bin/env python
"""Golden fixtures for the DCS robust local level (tsecon-gas crate,
``dcs_local_level``): score-driven level filtering with Gaussian /
Student-t / Laplace observation densities (Harvey 2013; Creal-Koopman-Lucas
2013; Harvey & Luati 2014 JASA — the DCS-t local level).

Reference implementations (this venv):
  * Gaussian limit -> statsmodels 0.14.6
      statsmodels.tsa.statespace.structural.UnobservedComponents(y, 'llevel')
      — an INDEPENDENT-PACKAGE golden, pinned THROUGH the steady-state
      mapping derived below.
  * Student-t / Laplace -> documented-formula goldens: the recursion and
      densities written out below, applied literally in NumPy (no package
      implements the DCS-t/Laplace local level; the DCS reference code is
      R/Matlab). The Student-t observation density is additionally
      cross-checked against scipy.stats.t (independent reference).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  /home/user/tsecon/.venv/bin/python fixtures/generate_tsecon-dcs_fixtures.py

MODEL (time-varying level ``mu_t``, observations ``y_t``)
---------------------------------------------------------
Observation: y_t = mu_t + eps_t, eps_t iid with density p(.; scale[, nu]).
Level recursion driven by the conditional score of the observation density:

    e_t      = y_t - mu_t                       (one-step prediction error)
    u_t      = scale^2 * d log p(y_t|mu_t)/d mu_t
    mu_{t+1} = mu_t + kappa * u_t,              kappa >= 0.

The scale^2 factor is a constant multiple of the inverse-information
scaling (absorbed into kappa), giving:

    Gaussian : u_t = e_t
    Student-t: u_t = (nu+1) e_t / (nu + e_t^2/scale^2)   (redescending)
    Laplace  : u_t = scale * sign(e_t)                   (sign filter)

Initialization: mu_1 = median(y_1..y_min(10,T)) (robust); the likelihood is
the exact conditional one given mu_1 (no burn).

THE GAUSSIAN LIMIT AND THE kappa <-> q MAPPING
----------------------------------------------
With Gaussian errors the recursion mu_{t+1} = mu_t + kappa e_t is EXACTLY
the steady-state (constant-gain, innovations-form) Kalman filter of the
Gaussian local level

    y_t = mu_t + eps_t,  eps_t ~ N(0, sigma2_eps),
    mu_{t+1} = mu_t + eta_t,  eta_t ~ N(0, sigma2_eta).

statsmodels parameterizes that model by the two variances; the DCS filter
by (kappa, scale). The exact mapping: with the signal-to-noise ratio
q = sigma2_eta / sigma2_eps, the steady-state Riccati fixed point
P = P - P^2/(P + sigma2_eps) + sigma2_eta has scaled solution

    p = P / sigma2_eps = (q + sqrt(q^2 + 4 q)) / 2,

the constant predictive-recursion Kalman gain is

    kappa = p / (1 + p),           (inverse: q = kappa^2 / (1 - kappa))

and the one-step prediction-error variance is

    scale^2 = F = P + sigma2_eps = sigma2_eps * (1 + p)
            = sigma2_eps / (1 - kappa).

The fixture pins THROUGH this mapping: statsmodels' UC MLE variances are
mapped to (kappa, scale), statsmodels itself is re-run at those variances
with KNOWN initialization at the steady state (initial state mu_1, initial
state variance P = p * sigma2_eps) — its Kalman gain is then constant and
equal to kappa at every t, its predicted-state path is the DCS level path,
and the sum of its per-observation log-likelihoods equals the DCS
log-likelihood. All three claims are asserted below at ~1e-10 before the
values are written. (NB statsmodels' reported `llf` excludes the first
observation — UnobservedComponents keeps `loglikelihood_burn = 1` even
under known initialization — so the golden pins `sum(llf_obs)`, all T
terms.)

FITTED-PARAMETER GOLDEN (two optimizers)
----------------------------------------
`dcs_mle` per series is the maximizer of the SAME criterion the Rust
implementation optimizes (mean negative conditional log-likelihood given
mu_1, kappa = exp(z0), scale = exp(z1)), found here by scipy L-BFGS-B from
three deterministic kappa starts and polished by a high-precision
Nelder-Mead. Rust (tsecon-optim Nelder-Mead multistart) must land on the
same interior optimum to 1e-4 relative in the parameters — a
two-optimizer, one-criterion pin.

VALIDATION LEGS (asserted in this file before writing)
------------------------------------------------------
1. NumPy innovations recursion == statsmodels known-init filter: level
   path to machine precision, loglik to ~1e-10, Kalman gain == kappa.
2. Mapping round-trip: q == kappa^2/(1-kappa) to 1e-14.
3. Student-t observation log-density == scipy.stats.t.logpdf(e, df=nu,
   scale=scale) to 1e-12.
4. The scipy DCS-Gaussian MLE has zero gradient (central differences) to
   1e-6 in working space (interior optimum).
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln
from statsmodels.tsa.statespace.structural import UnobservedComponents

OUT = Path(__file__).resolve().parent / "tsecon-dcs.json"

META = {
    "statsmodels": statsmodels.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "python": platform.python_version(),
}

LOG2PI = np.log(2.0 * np.pi)


# --------------------------------------------------------------- series

def simulate_local_level(T, sigma_eta, sigma_eps, outlier_frac, outlier_size,
                         seed):
    """Gaussian local level, optionally with additive outliers on the
    observation only (the level path stays clean)."""
    rng = np.random.default_rng(seed)
    mu = np.cumsum(rng.normal(0.0, sigma_eta, T))
    y = mu + rng.normal(0.0, sigma_eps, T)
    mask = np.zeros(T, bool)
    if outlier_frac > 0.0:
        idx = rng.choice(T, size=int(round(outlier_frac * T)), replace=False)
        mask[idx] = True
        y[idx] += rng.choice([-1.0, 1.0], idx.size) * outlier_size * sigma_eps
    return y, mu, mask


def nile_series():
    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)


# ------------------------------------------------- DCS filter (documented)

def mu_init(y):
    return float(np.median(y[: min(10, len(y))]))


def dcs_filter(y, density, kappa, scale, nu=None):
    """The documented recursion, applied literally. Returns
    (mu path, per-obs loglik, mu_{T+1})."""
    T = len(y)
    mu = np.empty(T)
    ll = np.empty(T)
    m = mu_init(y)
    s2 = scale * scale
    for t in range(T):
        mu[t] = m
        e = y[t] - m
        if density == "gaussian":
            ll[t] = -0.5 * (LOG2PI + np.log(s2)) - 0.5 * e * e / s2
            u = e
        elif density == "t":
            c = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
                 - 0.5 * np.log(nu * np.pi * s2))
            ll[t] = c - 0.5 * (nu + 1.0) * np.log1p(e * e / (nu * s2))
            u = (nu + 1.0) * e / (nu + e * e / s2)
        elif density == "laplace":
            ll[t] = -np.log(2.0 * scale) - abs(e) / scale
            u = scale * np.sign(e)
        else:
            raise ValueError(density)
        m = m + kappa * u
    return mu, ll, m


def cross_check_t_density(nu, scale, e):
    ours = (gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi * scale * scale)
            - 0.5 * (nu + 1.0) * np.log1p(e * e / (nu * scale * scale)))
    ref = stats.t.logpdf(e, df=nu, loc=0.0, scale=scale)
    assert abs(ours - ref) < 1e-12, f"t density mismatch {ours} vs {ref}"


# ------------------------------------ Gaussian limit: mapping + statsmodels

def steady_state_map(sigma2_eps, sigma2_eta):
    q = sigma2_eta / sigma2_eps
    p = 0.5 * (q + np.sqrt(q * q + 4.0 * q))
    kappa = p / (1.0 + p)
    scale = float(np.sqrt(sigma2_eps * (1.0 + p)))
    # round-trip: q = kappa^2 / (1 - kappa)
    assert abs(q - kappa * kappa / (1.0 - kappa)) < 1e-13 * max(q, 1.0)
    return q, p, float(kappa), scale


def gaussian_case(y, name):
    y = np.asarray(y, float)
    T = len(y)
    mu0 = mu_init(y)

    # 1) statsmodels UC MLE (its own parameterization + diffuse-ish init).
    uc = UnobservedComponents(y, "llevel")
    fit = uc.fit(disp=0)
    s2eps = float(fit.params[uc.param_names.index("sigma2.irregular")])
    s2eta = float(fit.params[uc.param_names.index("sigma2.level")])
    q, p, kappa, scale = steady_state_map(s2eps, s2eta)

    # 2) statsmodels AGAIN, at the mapped params, with known steady-state
    #    initialization: constant gain == kappa, predicted path == DCS path.
    ss = UnobservedComponents(y, "llevel")
    ss.ssm.initialize_known(np.array([mu0]), np.array([[p * s2eps]]))
    res = ss.filter([s2eps, s2eta])
    llf_obs = np.asarray(res.filter_results.llf_obs)
    loglik_ss = float(llf_obs.sum())
    pred = np.asarray(res.predicted_state[0])
    level = pred[:T]
    next_level = float(pred[T])
    gain = np.asarray(res.filter_results.kalman_gain)[0, 0, :]
    assert np.max(np.abs(gain - kappa)) < 1e-12, name
    fvar = np.asarray(res.filter_results.forecasts_error_cov)[0, 0, :]
    assert np.max(np.abs(fvar - scale * scale)) < 1e-8 * scale * scale, name

    # 3) NumPy innovations recursion agrees with statsmodels (leg 1).
    mu_np, ll_np, mnext_np = dcs_filter(y, "gaussian", kappa, scale)
    assert np.max(np.abs(mu_np - level)) < 1e-9 * max(np.max(np.abs(level)), 1.0)
    assert abs(ll_np.sum() - loglik_ss) < 1e-10 * abs(loglik_ss)
    assert abs(mnext_np - next_level) < 1e-9 * max(abs(next_level), 1.0)

    # 4) the DCS-Gaussian MLE of the same criterion Rust optimizes.
    mle = fit_dcs_gaussian_mle(y)

    return {
        "series": name,
        "y": y.tolist(),
        "mu0": mu0,
        "uc_mle": {"sigma2_eps": s2eps, "sigma2_eta": s2eta,
                   "llf": float(fit.llf)},
        "map": {"q": float(q), "p": float(p), "kappa": kappa, "scale": scale},
        "ss_filter": {"loglik": loglik_ss, "level": level.tolist(),
                      "next_level": next_level},
        "dcs_mle": mle,
    }


def robust_scale_diff(y):
    d = np.diff(y)
    mad = np.median(np.abs(d - np.median(d)))
    s = mad / 0.6745 / np.sqrt(2.0)
    return float(s) if s > 0 else float(np.std(y))


def fit_dcs_gaussian_mle(y):
    """Maximize the exact conditional DCS-Gaussian likelihood (given mu_1)
    over (kappa, scale) — the identical criterion the Rust `fit` optimizes,
    solved here with scipy (L-BFGS-B + high-precision Nelder-Mead polish)."""
    y = np.asarray(y, float)
    T = len(y)
    s_rob = robust_scale_diff(y)

    def neg(z):
        _, ll, _ = dcs_filter(y, "gaussian", np.exp(z[0]), np.exp(z[1]))
        return -ll.sum() / T

    best = None
    for k0 in (0.05, 0.3, 0.8):
        z0 = np.array([np.log(k0), np.log(s_rob)])
        r = minimize(neg, z0, method="L-BFGS-B")
        r = minimize(neg, r.x, method="Nelder-Mead",
                     options={"xatol": 1e-13, "fatol": 1e-15,
                              "maxiter": 8000, "maxfev": 8000})
        if best is None or r.fun < best.fun:
            best = r
    # interior optimum: central-difference gradient ~ 0 in working space
    h = 1e-6
    for i in range(2):
        zp, zm = best.x.copy(), best.x.copy()
        zp[i] += h
        zm[i] -= h
        g = (neg(zp) - neg(zm)) / (2 * h)
        assert abs(g) < 1e-6, f"non-interior optimum: grad[{i}] = {g}"
    kappa, scale = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    _, ll, _ = dcs_filter(y, "gaussian", kappa, scale)
    return {"kappa": kappa, "scale": scale, "loglik": float(ll.sum())}


# ----------------------------------------------------------------- main

def main():
    fixtures = {"meta": META}

    # --- Gaussian-limit goldens: two seeded local levels + the Nile ------
    y1, mu1, _ = simulate_local_level(500, 0.1, 1.0, 0.0, 0.0, seed=11)
    y2, _, _ = simulate_local_level(400, 0.3, 1.0, 0.0, 0.0, seed=3)
    cases = [
        gaussian_case(y1, "sim_ll_sn01_t500_seed11"),
        gaussian_case(y2, "sim_ll_sn03_t400_seed3"),
        gaussian_case(nile_series(), "nile"),
    ]
    # keep the clean level path of case 1 for the property tests
    cases[0]["mu_true"] = mu1.tolist()
    fixtures["gaussian_ss"] = cases

    # --- Student-t / Laplace fixed-parameter filter goldens --------------
    # (documented-formula: the recursion above applied literally; Laplace
    # uses the exact hard sign). Contaminated series so the redescending /
    # sign drivers are actually exercised.
    yc, _, _ = simulate_local_level(300, 0.15, 1.0, 0.05, 8.0, seed=17)
    t_kappa, t_scale, t_nu = 0.12, 0.9, 5.0
    mu_t, ll_t, next_t = dcs_filter(yc, "t", t_kappa, t_scale, t_nu)
    for i in (0, 41, 137, 299):
        cross_check_t_density(t_nu, t_scale, float(yc[i] - mu_t[i]))
    l_kappa, l_scale = 0.15, 0.8
    mu_l, ll_l, next_l = dcs_filter(yc, "laplace", l_kappa, l_scale)
    g_kappa, g_scale = 0.2, 1.1
    mu_g, ll_g, next_g = dcs_filter(yc, "gaussian", g_kappa, g_scale)
    fixtures["filter_golden"] = {
        "y": yc.tolist(),
        "mu0": mu_init(yc),
        "student_t": {"params": {"kappa": t_kappa, "scale": t_scale,
                                 "nu": t_nu},
                      "level": mu_t.tolist(), "loglik": float(ll_t.sum()),
                      "next_level": next_t},
        "laplace": {"params": {"kappa": l_kappa, "scale": l_scale},
                    "level": mu_l.tolist(), "loglik": float(ll_l.sum()),
                    "next_level": next_l},
        "gaussian": {"params": {"kappa": g_kappa, "scale": g_scale},
                     "level": mu_g.tolist(), "loglik": float(ll_g.sum()),
                     "next_level": next_g},
    }

    # --- contaminated property-test designs (data, not goldens) ----------
    # The lab's exp03 design: sigma_eta = 0.1, sigma_eps = 1.0, additive
    # 8-sigma outliers on the observation only; RMSE is measured against
    # the CLEAN mu_true.
    for frac, seed, key in [(0.05, 23, "sim_contam5"),
                            (0.10, 29, "sim_contam10")]:
        y, mu, mask = simulate_local_level(500, 0.1, 1.0, frac, 8.0, seed)
        fixtures[key] = {
            "design": {"T": 500, "sigma_eta": 0.1, "sigma_eps": 1.0,
                       "outlier_frac": frac, "outlier_size": 8.0,
                       "seed": seed},
            "y": y.tolist(),
            "mu_true": mu.tolist(),
            "outlier_mask": mask.astype(int).tolist(),
        }

    OUT.write_text(json.dumps(fixtures))
    print(f"wrote {OUT}")
    for c in fixtures["gaussian_ss"]:
        print(f"  {c['series']}: kappa_ss={c['map']['kappa']:.6f} "
              f"scale_ss={c['map']['scale']:.6f} "
              f"ss loglik={c['ss_filter']['loglik']:.6f} | "
              f"dcs_mle kappa={c['dcs_mle']['kappa']:.6f} "
              f"scale={c['dcs_mle']['scale']:.6f} "
              f"loglik={c['dcs_mle']['loglik']:.6f}")


if __name__ == "__main__":
    main()
