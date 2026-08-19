"""Golden fixtures for the proxy-SVAR first-stage strength diagnostics
(`proxy_first_stage`): the Montiel Olea-Pflueger effective F under classical /
HC1 / HAC-Bartlett variance, and the MOP tau-based critical values.

VALIDATION STRATEGY
===================
Every number here is produced by an INDEPENDENT reference — statsmodels OLS
with classical, HC1 and HAC(Bartlett) covariance for the regression algebra,
and scipy.stats.ncx2 for the critical values — and NEVER by the tsecon Rust
crate. All data are DERIVED from a seeded NumPy DGP; nothing is redistributed.

THE STATISTIC
-------------
With one instrument the MOP (2013, JBES) effective first-stage F coincides
with the robust F: the squared robust t-statistic of the slope in the OLS
regression of the target residual u_nv on [1, m] over the overlap O (the
finite-proxy rows). Reference: Windmeijer (2025, J. Econometrics, "The robust
F-statistic as a test for weak instruments"): in the just-identified case the
effective F and its critical values are identical to the robust F and its
critical values. The variance conventions follow Stata `weakivtest`
(Pflueger & Wang 2015, Stata Journal 15(1)):

* classical:  Var(b) = s^2 / Smm, s^2 = SSE / (|O| - 2)
* HC1:        Var(b) = (|O|/(|O|-2)) * sum(md^2 e^2) / Smm^2
* HAC:        Var(b) = (|O|/(|O|-2)) * [G0 + sum_{j<=L} w_j (Gj + Gj')] / Smm^2,
              w_j = 1 - j/(L+1) (Bartlett), scores s_t = md_t e_t

which statsmodels reproduces exactly via cov_type="HC1" and
cov_type="HAC" (kernel=bartlett, use_correction=True; its correction is
n/(n-k) = |O|/(|O|-2) for the two-regressor design).

THE CRITICAL VALUES
-------------------
MOP test the null "worst-case (Nagar-benchmark) relative bias > tau" at level
alpha. With one instrument the effective degrees of freedom equal 1 and the
critical value is the noncentral chi-square quantile

    cv(tau, alpha) = ncx2.ppf(1 - alpha, df=1, nc=1/tau)

(the `weakivtest` construction `invnchi2(K_eff, K_eff*x, 1-alpha)/K_eff` with
K_eff = 1, x = 1/tau). At alpha = 5%: tau 5% -> 37.42, 10% -> 23.11,
20% -> 15.06, 30% -> 12.05 (weakivtest prints 12.039 for the last row only
because it rounds x to 3.33). The fixture also pins the tau-bound inversion:
the lambda solving ncx2.ppf(0.95, 1, lambda) = F, reported as tau = 1/lambda.

The crate test feeds each case's (u, proxy) straight into
tsecon_ident::proxy_first_stage and must reproduce beta, se, effective_f,
f_classical, f_hc1, reliability, n_proxy to rtol=1e-9, and the critical
values / tau bounds to atol=1e-6 (the Rust side inverts an exact closed-form
df-1 CDF by bisection; scipy inverts the same distribution by its own path).

Run with the project venv:
    .venv/bin/python fixtures/generate_proxy_first_stage_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats

OUT = "fixtures/proxy_first_stage.json"


def nan_to_null(arr):
    """1-D list with non-finite entries as None (strict JSON; Rust reads null as NaN)."""
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr)]


def simulate(seed, t, strength, ar_score=0.0, nan_prefix=0):
    """Residual matrix u (t x 3) and a proxy for shock 0 of chosen strength.

    `ar_score` > 0 gives the proxy an AR(1) measurement-noise component so the
    first-stage score m~_t e_t is serially correlated and the HAC variance has
    something real to correct.
    """
    rng = np.random.default_rng(seed)
    h = np.array([[1.0, 0.4, 0.2], [0.5, 1.2, 0.3], [0.3, 0.5, 0.9]])
    eps = rng.standard_normal((t, 3))
    u = eps @ h.T
    noise = rng.standard_normal(t)
    if ar_score > 0.0:
        for i in range(1, t):
            noise[i] += ar_score * noise[i - 1]
    proxy = strength * eps[:, 0] + noise
    if nan_prefix:
        proxy[:nan_prefix] = np.nan
    return u, proxy


def reference_first_stage(u, proxy, norm_var, hac_lags=None):
    """statsmodels OLS reference for one case; returns every pinned number."""
    mask = np.isfinite(proxy)
    m = proxy[mask]
    y = u[mask, norm_var]
    n_o = int(mask.sum())
    x = sm.add_constant(m)

    fit_classical = sm.OLS(y, x).fit()
    fit_hc1 = sm.OLS(y, x).fit(cov_type="HC1")
    b = float(fit_classical.params[1])

    f_classical = b * b / float(fit_classical.bse[1]) ** 2
    f_hc1 = b * b / float(fit_hc1.bse[1]) ** 2
    out = {
        "beta": b,
        "f_classical": f_classical,
        "f_hc1": f_hc1,
        "reliability": float(np.corrcoef(m, y)[0, 1] ** 2),
        "n_proxy": n_o,
    }
    if hac_lags is None:
        out["se"] = float(fit_hc1.bse[1])
        out["effective_f"] = f_hc1
    else:
        fit_hac = sm.OLS(y, x).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": hac_lags, "kernel": "bartlett", "use_correction": True},
        )
        out["se"] = float(fit_hac.bse[1])
        out["effective_f"] = b * b / float(fit_hac.bse[1]) ** 2
    return out


def mop_cv(tau, alpha=0.05):
    return float(stats.ncx2.ppf(1.0 - alpha, 1, 1.0 / tau))


def mop_tau_bound(f_eff, alpha=0.05):
    """Smallest tau rejected at alpha: invert ncx2.ppf(1-alpha, 1, 1/tau) = F."""
    if f_eff <= stats.chi2.ppf(1.0 - alpha, 1):
        return None  # +inf on the Rust side
    from scipy.optimize import brentq

    g = lambda lam: stats.ncx2.ppf(1.0 - alpha, 1, lam) - f_eff
    lam = brentq(g, 1e-12, 1e6, xtol=1e-12, rtol=1e-13)
    return 1.0 / lam


def build_case(name, seed, t, strength, norm_var, hac_lags, ar_score, nan_prefix):
    u, proxy = simulate(seed, t, strength, ar_score=ar_score, nan_prefix=nan_prefix)
    ref = reference_first_stage(u, proxy, norm_var, hac_lags=hac_lags)
    ref["tau_bound"] = mop_tau_bound(ref["effective_f"])
    return {
        "name": name,
        "norm_var": norm_var,
        "hac_lags": hac_lags,
        "u": u.tolist(),
        "proxy": nan_to_null(proxy),
        "expected": ref,
    }


def main():
    seed = 20260819
    cases = [
        # Strong instrument, HC1 (the default path; also pins f_classical).
        build_case("strong_hc1", seed, 500, 1.0, 0, None, 0.0, 120),
        # Weak instrument, HC1: effective F must fail the tau=10% bar.
        build_case("weak_hc1", seed + 1, 500, 0.08, 0, None, 0.0, 0),
        # Serially correlated score, HAC(6): HAC and HC1 must genuinely
        # differ here, so the case pins the kernel, not just the plumbing.
        build_case("strong_hac6", seed + 2, 600, 0.8, 1, 6, 0.7, 0),
        # HAC on a gap-free short overlap with a different bandwidth.
        build_case("mid_hac3", seed + 3, 300, 0.35, 0, 3, 0.4, 60),
    ]

    # Guard: the weak case is really weak and the strong cases really strong.
    assert cases[0]["expected"]["effective_f"] > mop_cv(0.10)
    assert cases[1]["expected"]["effective_f"] < mop_cv(0.30)
    hac = cases[2]["expected"]
    assert abs(hac["effective_f"] - hac["f_hc1"]) > 0.02 * hac["f_hc1"], (
        "HAC case fails to separate HAC from HC1; increase ar_score"
    )

    taus = [0.05, 0.10, 0.20, 0.30, 0.50]
    alphas = [0.05, 0.10]
    critical_values = [
        {"tau": tau, "alpha": a, "cv": mop_cv(tau, a)} for tau in taus for a in alphas
    ]
    # Round-trip checks for the tau bound at assorted F values.
    tau_bounds = [
        {"f": f, "alpha": 0.05, "tau": mop_tau_bound(f)}
        for f in [3.0, 3.85, 5.0, 10.0, 15.062, 21.55, 23.109, 37.418, 100.0]
    ]

    fixture = {
        "_meta": {
            "description": "Golden fixtures for proxy-SVAR first-stage strength "
            "diagnostics: the Montiel Olea-Pflueger effective F (== robust F in "
            "the just-identified single-instrument case; Windmeijer 2025) under "
            "classical/HC1/HAC-Bartlett variance, and the MOP tau-based critical "
            "values via the noncentral chi-square (weakivtest construction, "
            "K_eff = 1).",
            "references": {
                "regression": "statsmodels OLS(u_nv ~ [1, m]) over the finite-proxy "
                'rows; cov_type in {nonrobust, "HC1", "HAC"(bartlett, '
                "use_correction=True)}; effective F = (b / bse)^2",
                "critical_values": "scipy.stats.ncx2.ppf(1 - alpha, df=1, nc=1/tau)",
                "tau_bound": "brentq-invert ncx2.ppf(0.95, 1, lambda) = F; tau = 1/lambda; "
                "None (Rust: +inf) when F <= chi2.ppf(0.95, 1)",
            },
            "tolerance": {"rtol": 1e-9, "cv_atol": 1e-6},
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "seed": seed,
        },
        "cases": cases,
        "critical_values": critical_values,
        "tau_bounds": tau_bounds,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    for c in cases:
        e = c["expected"]
        print(
            f"  {c['name']}: n_proxy={e['n_proxy']} beta={e['beta']:+.4f} "
            f"F_eff={e['effective_f']:.3f} F_HC1={e['f_hc1']:.3f} "
            f"F_cl={e['f_classical']:.3f} tau_bound="
            + (f"{e['tau_bound']:.4f}" if e["tau_bound"] is not None else "inf")
        )
    print("  cv(tau=0.10, alpha=0.05) =", mop_cv(0.10))


if __name__ == "__main__":
    main()
