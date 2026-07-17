#!/usr/bin/env python
"""Golden fixtures for the tsecon-gas crate (GAS/DCS score-driven volatility).

This is a DOCUMENTED-FORMULA golden. The filtered variance path ``f_t`` and
the total log-likelihood are computed here by literally applying the
score-driven recursion and the observation density written out below, in
plain NumPy. The Rust crate must reproduce both to ~1e-10. Nothing here
calls the Rust code, so the golden is non-circular.

MODEL (time-varying variance ``f_t``, observations ``y_t``)
------------------------------------------------------------
Recursion (Creal-Koopman-Lucas 2013, inverse-information scaling
``S_t = I_t^{-1}``):

    f_{t+1} = omega + a * s_t + b * f_t,      s_t = S_t * nabla_t
    nabla_t = d/df_t  log p(y_t | f_t)         (score of the density)

Initialization: f_1 = omega / (1 - b)          (stationary mean of f_t).

GAUSSIAN density,  y_t ~ N(0, f_t):

    log p(y|f) = -0.5 ln(2*pi) - 0.5 ln f - 0.5 y^2 / f
    nabla_t    = 0.5 (y^2 - f) / f^2,   I_t = 0.5 / f^2,   S_t = 2 f^2
    => s_t     = y^2 - f
    => f_{t+1} = omega + a (y^2 - f) + b f.

STUDENT-T density (unit-variance standardized t, nu > 2 dof):

    y_t = sqrt(f_t) * eps_t,  eps_t ~ unit-variance t with density
        f_eps(e) = G((nu+1)/2) / (sqrt((nu-2) pi) G(nu/2))
                   * (1 + e^2/(nu-2))^{-(nu+1)/2}
    log p(y|f) = c(nu) - 0.5 ln f - (nu+1)/2 ln(1 + y^2/((nu-2) f)),
        c(nu)  = lnG((nu+1)/2) - lnG(nu/2) - 0.5 ln((nu-2) pi).

    With eps^2 = y^2/f and g(eps) = (nu eps^2 - (nu-2)) / ((nu-2) + eps^2):
        nabla_t = g(eps) / (2 f)
        I_t     = E[g^2] / (4 f^2),   E[g^2] = 2 nu / (nu+3)   [validated below]
        S_t     = 4 f^2 (nu+3) / (2 nu)
    => s_t = ((nu+3)/nu) * f * (nu y^2 - (nu-2) f) / ((nu-2) f + y^2).

    As nu -> inf, s_t -> y^2 - f, recovering the Gaussian update.

VALIDATION
----------
1. The scaling constant E[g^2] = 2 nu/(nu+3) is confirmed by direct
   numerical integration against f_eps (scipy.integrate.quad), and E[g] = 0.
2. The Student-t log density log p(y|f) is cross-checked against
   scipy.stats.t: y ~ t(df=nu, loc=0, scale=sqrt(f (nu-2)/nu)) has exactly
   this density (independent reference).
"""

import json
import os

import numpy as np
from scipy import integrate, stats
from scipy.special import gammaln

HALF_LN_2PI = 0.5 * np.log(2.0 * np.pi)


def scaling_constant_check(nu):
    """Confirm E[g] = 0 and E[g^2] = 2 nu/(nu+3) by numerical integration."""
    sd = np.sqrt(nu / (nu - 2.0))  # eps = z / sd, z ~ standard t_nu

    def peps(e):
        return sd * stats.t.pdf(sd * e, df=nu)

    def g(e):
        return (nu * e * e - (nu - 2.0)) / ((nu - 2.0) + e * e)

    eg, _ = integrate.quad(lambda e: g(e) * peps(e), -np.inf, np.inf)
    eg2, _ = integrate.quad(lambda e: g(e) ** 2 * peps(e), -np.inf, np.inf)
    analytic = 2.0 * nu / (nu + 3.0)
    assert abs(eg) < 1e-9, f"E[g] != 0: {eg}"
    assert abs(eg2 - analytic) < 1e-8, f"E[g^2] {eg2} != {analytic}"
    return eg2, analytic


def log_density(density, nu, y, f):
    if density == "gaussian":
        return -HALF_LN_2PI - 0.5 * np.log(f) - 0.5 * y * y / f
    a = nu - 2.0
    c = gammaln(0.5 * (nu + 1.0)) - gammaln(0.5 * nu) - 0.5 * np.log(a * np.pi)
    return c - 0.5 * np.log(f) - 0.5 * (nu + 1.0) * np.log1p(y * y / (a * f))


def scaled_score(density, nu, y, f):
    if density == "gaussian":
        return y * y - f
    a = nu - 2.0
    y2 = y * y
    return ((nu + 3.0) / nu) * f * (nu * y2 - a * f) / (a * f + y2)


def filter_gas(density, y, omega, a, b, nu):
    """Apply the documented recursion; return (f_path, loglik, f_next)."""
    n = len(y)
    f = np.empty(n)
    ft = omega / (1.0 - b)
    loglik = 0.0
    for t in range(n):
        f[t] = ft
        loglik += log_density(density, nu, y[t], ft)
        st = scaled_score(density, nu, y[t], ft)
        ft = omega + a * st + b * ft
    return f, loglik, ft


def cross_check_student_density(nu, y, f):
    """Independent reference: scipy.stats.t on the rescaled variable."""
    ours = log_density("student-t", nu, y, f)
    scale = np.sqrt(f * (nu - 2.0) / nu)
    ref = stats.t.logpdf(y, df=nu, loc=0.0, scale=scale)
    assert abs(ours - ref) < 1e-12, f"t density mismatch {ours} vs {ref}"


def simulate_gas(density, n, omega, a, b, nu, seed):
    """Simulate a GAS series for the ML-recovery property test (data only,
    not a golden to match bit-for-bit)."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    ft = omega / (1.0 - b)
    for t in range(n):
        if density == "gaussian":
            y[t] = rng.normal(0.0, np.sqrt(ft))
        else:
            # unit-variance standardized t innovation, scaled by sqrt(ft)
            z = rng.standard_t(nu)
            eps = z / np.sqrt(nu / (nu - 2.0))
            y[t] = np.sqrt(ft) * eps
        st = scaled_score(density, nu, y[t], ft)
        ft = omega + a * st + b * ft
        if ft < 1e-12:
            ft = 1e-12
    return y


def main():
    fixtures = {}

    # --- validate the Student-t scaling constant for the nu we use --------
    fixtures["scaling_check"] = []
    for nu in (5.0, 6.0, 8.0, 15.0):
        eg2, analytic = scaling_constant_check(nu)
        fixtures["scaling_check"].append(
            {"nu": nu, "e_g2_numeric": eg2, "e_g2_analytic": analytic}
        )

    # --- a fixed, deterministic return series -----------------------------
    rng = np.random.default_rng(20260717)
    y = rng.standard_normal(60) * 1.1
    y = y.tolist()

    # --- Gaussian golden --------------------------------------------------
    g_omega, g_a, g_b = 0.05, 0.04, 0.92
    gf, gll, gnext = filter_gas("gaussian", y, g_omega, g_a, g_b, np.nan)
    # h-step forecast: v1 = f_{N+1}; v_k = omega + b v_{k-1}
    gfc = [gnext]
    for _ in range(9):
        gfc.append(g_omega + g_b * gfc[-1])
    fixtures["gaussian_golden"] = {
        "params": {"omega": g_omega, "a": g_a, "b": g_b},
        "y": y,
        "variance": gf.tolist(),
        "loglik": gll,
        "next_variance": gnext,
        "forecast": gfc,
    }

    # --- Student-t golden -------------------------------------------------
    t_omega, t_a, t_b, t_nu = 0.03, 0.05, 0.90, 6.0
    tf, tll, tnext = filter_gas("student-t", y, t_omega, t_a, t_b, t_nu)
    # cross-check the density piece at a couple of points against scipy
    for i in (0, 17, 41):
        cross_check_student_density(t_nu, y[i], tf[i])
    tfc = [tnext]
    for _ in range(9):
        tfc.append(t_omega + t_b * tfc[-1])
    fixtures["student_t_golden"] = {
        "params": {"omega": t_omega, "a": t_a, "b": t_b, "nu": t_nu},
        "y": y,
        "variance": tf.tolist(),
        "loglik": tll,
        "next_variance": tnext,
        "forecast": tfc,
    }

    # --- longer simulated series for the ML-recovery property test --------
    sim_g = simulate_gas("gaussian", 4000, 0.05, 0.06, 0.90, np.nan, seed=7)
    fixtures["sim_gaussian"] = {
        "true_params": {"omega": 0.05, "a": 0.06, "b": 0.90},
        "y": sim_g.tolist(),
    }
    sim_t = simulate_gas("student-t", 4000, 0.04, 0.07, 0.90, 6.0, seed=11)
    fixtures["sim_student_t"] = {
        "true_params": {"omega": 0.04, "a": 0.07, "b": 0.90, "nu": 6.0},
        "y": sim_t.tolist(),
    }

    out_path = os.path.join(os.path.dirname(__file__), "tsecon-gas.json")
    with open(out_path, "w") as fh:
        json.dump(fixtures, fh)
    print(f"wrote {out_path}")
    print(f"gaussian loglik = {gll:.10f}, student-t loglik = {tll:.10f}")


if __name__ == "__main__":
    main()
