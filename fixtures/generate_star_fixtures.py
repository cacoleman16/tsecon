"""Golden fixtures for the smooth-transition autoregression (STAR) family:
`star_test` (the Terasvirta modeling-cycle battery), `star_eval` (the
concentrated fit at fixed transition parameters), and the `(gamma, c)` grid
stage of `star`.

Reference: NO third-party STAR implementation is reachable from this
environment (R/tsDyn requires CRAN, which the sandbox's egress policy
denies; statsmodels has no STAR), so the reference is built HERE, in NumPy
+ SciPy, as an independent transcription of the published closed forms:

  * `star_test` — the Luukkonen-Saikkonen-Terasvirta (1988, Biometrika
    75(3)) third-order Taylor auxiliary regression and the Terasvirta
    (1994, JASA 89(425)) H03/H02/H01 sequence.  Auxiliary regression:

        y_t = b0' w_t + b1'(x~ s_t) + b2'(x~ s_t^2) + b3'(x~ s_t^3) + u_t,

    w_t = (1, y_{t-1}, ..., y_{t-p})', x~ the lag block, s_t = y_{t-d};
    when d > p both w_t and x~ are augmented with y_{t-d} (Terasvirta's
    redefinition), so q = len(x~) is p or p + 1.  LM3 chi-squared form:
    n (SSR0 - SSR3)/SSR0 ~ chi2(3q); F form (small-sample recommended):
    ((SSR0 - SSR3)/3q)/(SSR3/(n - k0 - 3q)) ~ F(3q, n - k0 - 3q).
    H03: b3 = 0 in the full regression, F(q, n - k0 - 3q); H02: b2 = 0
    given b3 = 0, F(q, n - k0 - 2q); H01: b1 = 0 given b2 = b3 = 0,
    F(q, n - k0 - q).  Verdict: ESTAR iff the H02 p-value is strictly the
    smallest, else LSTAR.  Delay selection: smallest F-form LM3 p-value.

  * `star_eval` — at fixed (gamma, c) the model is linear:  OLS of y_t on
    [x_t, G_t x_t] with
        LSTAR: G = 1/(1 + exp(-gamma (s - c)))          (raw gamma — the
        ESTAR: G = 1 - exp(-gamma (s - c)^2)             tsDyn convention;
    Terasvirta's standardized gamma is gamma * sd(s) resp. gamma * var(s),
    population sd over the usable sample).  Gauss-Newton standard errors
    from the full Jacobian J = [x, Gx, (phi2'x) dG/dgamma, (phi2'x) dG/dc]:
    se = sqrt(sigma2 * diag[(J'J)^{-1}]), sigma2 = SSR/(n - 2k - 2);
    loglik = -n/2 (ln(2 pi SSR/n) + 1); AIC/BIC = n ln(SSR/n) + pen * m,
    m = 2k + 2.

  * the `star` grid — gamma log-spaced over standardized [0.5, 100]
    divided by the standardizer; c on n_c equally spaced order statistics
    of s between the trim and 1 - trim quantile positions (index =
    floor(pos + 0.5)); per-cell concentrated OLS SSR.  The Nelder-Mead
    refinement is deliberately NOT pinned (optimizer-dependent); the crate
    pins refined-SSR <= grid-SSR and self-consistency by property.

Grade (honest): documented-ALGORITHM transcription validated against an
independent NumPy implementation (this file) — NOT a third-party golden.
What the pin proves: the Rust STAR machinery (designs, augmentation rule,
nested-F battery, chi2/F tails, transition functions, concentrated OLS,
Gauss-Newton SEs, grid construction) agrees with a direct dense NumPy
implementation of the same published closed forms at 1e-10.  Statistical
correctness (test size/power, parameter recovery, the LSTAR->SETAR limit)
is established separately by the crate's seeded Monte Carlo property
tests, whose numbers are quoted in the model card.

This generator NEVER imports tsecon.  Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).  Non-finite grid cells are written as
null.

Run:  .venv-wt/bin/python fixtures/generate_star_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(__file__).resolve().parent / "star.json"

GAMMA_STD_GRID_LO = 0.5
GAMMA_STD_GRID_HI = 100.0


# ------------------------------------------------------------------ series

def g_lstar(gamma, c, s):
    return 1.0 / (1.0 + np.exp(-gamma * (s - c)))


def g_estar(gamma, c, s):
    return 1.0 - np.exp(-gamma * (s - c) ** 2)


def sim_star(seed, T, d, model, gamma, c, phi1, phi2, sigma=1.0, burn=100):
    """Simulate a STAR: phi vectors are [const, lag1, ...]; raw gamma."""
    rng = np.random.default_rng(seed)
    p = len(phi1) - 1
    m = max(p, d)
    g = g_lstar if model == "lstar" else g_estar
    y = np.zeros(T + burn + m)
    e = sigma * rng.standard_normal(T + burn + m)
    for t in range(m, T + burn + m):
        x = np.array([1.0] + [y[t - 1 - l] for l in range(p)])
        gt = g(gamma, c, y[t - d])
        y[t] = x @ phi1 + gt * (x @ phi2) + e[t]
    return y[burn + m:]


def sim_ar(seed, T, coefs, sigma=1.0, burn=100):
    rng = np.random.default_rng(seed)
    p = len(coefs) - 1
    y = np.zeros(T + burn + p)
    e = sigma * rng.standard_normal(T + burn + p)
    for t in range(p, T + burn + p):
        y[t] = coefs[0] + sum(coefs[1 + l] * y[t - 1 - l] for l in range(p)) + e[t]
    return y[burn + p:]


def make_series():
    return {
        # Strongly separated LSTAR(1), d=1: intercepts push the process
        # back across c = 0, both regimes well populated.
        "lstar_strong": sim_star(20260827, 300, 1, "lstar", 2.0, 0.0,
                                 phi1=[1.0, 0.6], phi2=[-2.0, -0.4]),
        # ESTAR(1), d=1: random-walk inner regime, strongly mean-reverting
        # outer band — the classic real-exchange-rate shape. (gamma = 0.25
        # keeps the band wide relative to the wander, which is where the
        # LM3 battery has power and H02 dominates.)
        "estar_mid": sim_star(42, 300, 1, "estar", 0.25, 0.0,
                              phi1=[0.0, 1.0], phi2=[0.0, -0.9]),
        # LSTAR(1) switching on the SECOND lag (delay selection).
        "lstar_d2": sim_star(97, 320, 2, "lstar", 3.0, 0.2,
                             phi1=[0.8, 0.5], phi2=[-1.6, -0.3]),
        # Linear nulls.
        "linear_ar1": sim_ar(11, 200, [0.0, 0.5]),
        "linear_ar2": sim_ar(13, 250, [0.2, 0.4, 0.25]),
    }


# ------------------------------------- the transcription (see module doc)

def design(y, p, delay, start, constant=True):
    T = y.size
    n = T - start
    cols = []
    if constant:
        cols.append(np.ones(n))
    for lag in range(1, p + 1):
        cols.append(y[start - lag:T - lag])
    X = np.column_stack(cols)
    resp = y[start:]
    s = y[start - delay:T - delay]
    return X, resp, s


def ols_ssr(X, yy):
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r = yy - X @ b
    return b, float(r @ r)


def battery(y, p, delay):
    start = max(p, delay)
    X, yy, s = design(y, p, delay, start, constant=True)
    n = yy.size
    # w = [1, lags, (s if d > p)]; x~ = lags (+ s if d > p).
    w = X
    xt = X[:, 1:]
    if delay > p:
        w = np.column_stack([w, s])
        xt = np.column_stack([xt, s])
    q = xt.shape[1]
    k0 = w.shape[1]

    def block(m):
        return xt * (s ** m)[:, None]

    _, ssr0 = ols_ssr(w, yy)
    a1 = np.column_stack([w, block(1)])
    _, ssr1 = ols_ssr(a1, yy)
    a2 = np.column_stack([a1, block(2)])
    _, ssr2 = ols_ssr(a2, yy)
    a3 = np.column_stack([a2, block(3)])
    _, ssr3 = ols_ssr(a3, yy)

    lm3 = n * (ssr0 - ssr3) / ssr0
    lm3_p = float(stats.chi2.sf(lm3, 3 * q))

    def nested_f(ssr_r, ssr_f, r, df2):
        f = ((ssr_r - ssr_f) / r) / (ssr_f / df2)
        return float(f), float(stats.f.sf(f, r, df2))

    lm3_f, lm3_f_p = nested_f(ssr0, ssr3, 3 * q, n - k0 - 3 * q)
    h3_f, h3_p = nested_f(ssr2, ssr3, q, n - k0 - 3 * q)
    h2_f, h2_p = nested_f(ssr1, ssr2, q, n - k0 - 2 * q)
    h1_f, h1_p = nested_f(ssr0, ssr1, q, n - k0 - q)
    suggested = "estar" if (h2_p < h1_p and h2_p < h3_p) else "lstar"
    return {
        "delay": int(delay),
        "nobs": int(n),
        "q": int(q),
        "k0": int(k0),
        "lm3_stat": float(lm3),
        "lm3_p_value": lm3_p,
        "lm3_f_stat": lm3_f,
        "lm3_f_p_value": lm3_f_p,
        "h3_f_stat": h3_f,
        "h3_p_value": h3_p,
        "h2_f_stat": h2_f,
        "h2_p_value": h2_p,
        "h1_f_stat": h1_f,
        "h1_p_value": h1_p,
        "ssr0": float(ssr0),
        "ssr1": float(ssr1),
        "ssr2": float(ssr2),
        "ssr3": float(ssr3),
        "suggested": suggested,
    }


def star_test(y, p, delays):
    tests = [battery(y, p, d) for d in delays]
    best = int(np.argmin([t["lm3_f_p_value"] for t in tests]))
    return {"tests": tests, "best": best}


def transition(model, gamma, c, s):
    return (g_lstar if model == "lstar" else g_estar)(gamma, c, s)


def dg(model, gamma, c, s):
    if model == "lstar":
        g = g_lstar(gamma, c, s)
        gg = g * (1.0 - g)
        return gg * (s - c), -gamma * gg
    e = np.exp(-gamma * (s - c) ** 2)
    return (s - c) ** 2 * e, -2.0 * gamma * (s - c) * e


def star_eval(y, p, delay, model, gamma, c, constant=True):
    start = max(p, delay)
    X, yy, s = design(y, p, delay, start, constant)
    n, k = X.shape
    G = transition(model, gamma, c, s)
    Z = np.column_stack([X, X * G[:, None]])
    beta, ssr = ols_ssr(Z, yy)
    m = 2 * k + 2
    sigma2 = ssr / (n - m)
    loglik = -0.5 * n * (math.log(2.0 * math.pi * ssr / n) + 1.0)
    aic = n * math.log(ssr / n) + 2.0 * m
    bic = n * math.log(ssr / n) + m * math.log(n)
    phi2 = beta[k:]
    phi2x = X @ phi2
    dgg, dgc = dg(model, gamma, c, s)
    J = np.column_stack([Z, phi2x * dgg, phi2x * dgc])
    se = np.sqrt(np.diag(np.linalg.inv(J.T @ J)) * sigma2)
    return {
        "coefs_linear": beta[:k].tolist(),
        "coefs_nonlinear": phi2.tolist(),
        "se_linear": se[:k].tolist(),
        "se_nonlinear": se[k:2 * k].tolist(),
        "se_gamma": float(se[2 * k]),
        "se_c": float(se[2 * k + 1]),
        "ssr": float(ssr),
        "sigma2": float(sigma2),
        "loglik": float(loglik),
        "aic": float(aic),
        "bic": float(bic),
        "nobs": int(n),
        "k": int(k),
        "transition_head": G[:8].tolist(),
        "transition_sum": float(G.sum()),
    }


def star_grid(y, p, delay, model, trim, n_gamma, n_c, constant=True):
    start = max(p, delay)
    X, yy, s = design(y, p, delay, start, constant)
    n = yy.size
    sd = float(np.std(s))  # population sd (ddof = 0)
    scale = sd if model == "lstar" else sd * sd
    lo, hi = math.log(GAMMA_STD_GRID_LO), math.log(GAMMA_STD_GRID_HI)
    gammas = [math.exp(lo + (hi - lo) * j / (n_gamma - 1)) / scale
              for j in range(n_gamma)]
    zs = np.sort(s)
    i_lo = math.ceil(trim * (n - 1))
    i_hi = math.floor((1.0 - trim) * (n - 1))
    cs = []
    for j in range(n_c):
        pos = i_lo + (i_hi - i_lo) * j / (n_c - 1)
        cs.append(float(zs[int(math.floor(pos + 0.5))]))
    ssr_grid = []
    best = None
    for i, g in enumerate(gammas):
        for j, c in enumerate(cs):
            G = transition(model, g, c, s)
            Z = np.column_stack([X, X * G[:, None]])
            _, ssr = ols_ssr(Z, yy)
            ssr_grid.append(float(ssr) if np.isfinite(ssr) else None)
            if np.isfinite(ssr) and (best is None or ssr < best[2]):
                best = (i, j, float(ssr))
    return {
        "s_sd": sd,
        "grid_gamma": gammas,
        "grid_c": cs,
        "ssr_grid": ssr_grid,
        "best_cell": [best[0], best[1]],
        "best_ssr": best[2],
    }


# ------------------------------------------------------------------- main

def main():
    series = make_series()

    test_cases = []
    for name, p, delays in [
        ("lstar_strong", 1, [1, 2]),
        ("estar_mid", 1, [1]),
        ("linear_ar1", 1, [1]),
        ("linear_ar2", 2, [1, 2, 3]),   # d = 3 > p = 2: augmentation path
        ("lstar_d2", 1, [1, 2, 3]),     # delay selection -> should pick 2
    ]:
        case = star_test(series[name], p, delays)
        case.update(series=name, p=p, delays=delays)
        test_cases.append(case)

    eval_cases = []
    for name, p, delay, model, gamma, c, constant in [
        ("lstar_strong", 1, 1, "lstar", 2.0, 0.0, True),
        ("lstar_strong", 1, 1, "lstar", 8.0, 0.25, True),
        ("estar_mid", 1, 1, "estar", 1.0, 0.0, True),
        ("estar_mid", 1, 1, "estar", 0.3, -0.4, True),
        ("linear_ar2", 2, 2, "lstar", 1.3, 0.1, False),  # no constant
        ("lstar_d2", 1, 2, "lstar", 3.0, 0.2, True),
    ]:
        case = star_eval(series[name], p, delay, model, gamma, c, constant)
        case.update(series=name, p=p, delay=delay, model=model,
                    gamma=gamma, c=c, constant=constant)
        eval_cases.append(case)

    grid_cases = []
    for name, p, delay, model, trim, n_gamma, n_c in [
        ("lstar_strong", 1, 1, "lstar", 0.15, 8, 7),
        ("estar_mid", 1, 1, "estar", 0.15, 8, 7),
        ("lstar_d2", 1, 2, "lstar", 0.10, 6, 9),
    ]:
        case = star_grid(series[name], p, delay, model, trim, n_gamma, n_c)
        case.update(series=name, p=p, delay=delay, model=model, trim=trim,
                    n_gamma=n_gamma, n_c=n_c)
        grid_cases.append(case)

    import scipy
    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "note": (
                "STAR family: documented-algorithm transcription "
                "(Luukkonen-Saikkonen-Terasvirta 1988; Terasvirta 1994) "
                "validated against this independent NumPy/SciPy "
                "implementation — no third-party STAR is reachable from "
                "this environment (CRAN egress denied, so no R tsDyn; "
                "statsmodels has no STAR). The Nelder-Mead refinement of "
                "(gamma, c) is checked by property (refined SSR <= grid "
                "SSR; seeded MC recovery), not pinned here."
            ),
        },
        "series": {k: v.tolist() for k, v in series.items()},
        "test": test_cases,
        "eval": eval_cases,
        "grid": grid_cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")
    for c in test_cases:
        t = c["tests"][c["best"]]
        print(f"  test {c['series']} p={c['p']} delays={c['delays']}: "
              f"best d={t['delay']} LM3-F p={t['lm3_f_p_value']:.3g} "
              f"suggested={t['suggested']}")
    for c in grid_cases:
        print(f"  grid {c['series']} {c['model']}: best cell {c['best_cell']} "
              f"ssr={c['best_ssr']:.4f}")


if __name__ == "__main__":
    main()
