"""Golden fixtures for tsecon-evt: peaks-over-threshold GPD tails (with
McNeil-Frey VaR/ES) and GEV block-maxima fitting (with return levels).

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference:
scipy.stats.genpareto / scipy.stats.genextreme (scipy 1.17.1 in the shared
reference venv) plus plain-numpy blocking/thresholding. Nothing here imports
tsecon, so reproducing these numbers in Rust is a genuine
cross-implementation check, never circular.

SIGN CONVENTIONS (verified numerically below, not assumed)
----------------------------------------------------------
* scipy's genpareto shape `c` IS the tail index xi used by tsecon
  (density (1/beta)(1 + xi z / beta)^(-1/xi - 1)).
* scipy's genextreme shape is `c = -xi` (scipy uses the "Weibull-positive"
  sign; the EVT literature's xi is its negation).
Both identities are asserted against hand-coded log-densities at 1e-12
before any fixture is written.

ESTIMATES AND THEIR GRADING
---------------------------
1. GPD parameters.  REFERENCE: scipy.stats.genpareto.fit(z, floc=0) —
   BUT scipy's rv_continuous.fit stops its Nelder-Mead at xtol=ftol=1e-4,
   far looser than the 1e-6 pin this fixture wants. The raw scipy fit is
   therefore POLISHED in-generator with scipy.optimize.minimize
   (Nelder-Mead, xatol=1e-12, fatol=1e-13, run twice) on the exact
   negative log-likelihood sum(-genpareto.logpdf) over (c, ln scale); the
   polish is asserted to only improve the likelihood. Both raw and
   polished params are stored; the Rust golden pins the POLISHED ones at
   1e-6 and, the honest optimizer comparison, its own log-likelihood at
   its own optimum against `loglik` (= sum(genpareto.logpdf) at the
   polished params) at 1e-10 absolute.
2. GEV parameters.  REFERENCE: scipy.stats.genextreme.fit(maxima),
   polished identically over (c, loc, ln scale); stored as
   xi = -c, mu = loc, sigma = scale.
3. Standard errors.  REFERENCE: observed information computed HERE with
   numpy — four-point central cross-differences (the statsmodels
   approx_hess3 stencil) of the negative log-likelihood at the polished
   optimum, steps h_i = eps^(1/4) * s_i with the same unit-safe scales the
   Rust side documents (shape: max(|xi|, 0.1); scale/location: the fitted
   scale), inverted with numpy.linalg.inv. Graded at 1e-4 relative.
4. VaR / ES (McNeil-Frey 2000).  GRADING: documented closed form,
   computed here from scipy's fitted params — VaR_p as
   u + genpareto.ppf(1 - (1-p)/rate, c, 0, beta) (scipy's own quantile
   route), asserted at 1e-12 against the transcribed closed form
   u + (beta/xi)((( 1-p)/rate)^(-xi) - 1); ES_p by the closed form
   (VaR_p + beta - xi u)/(1 - xi), cross-checked against numerical
   integration of the quantile curve (ES_p = (1/(1-p)) int_p^1 VaR_q dq)
   at 1e-6 relative. Graded at 1e-5 relative in Rust (the propagation of
   the 1e-6 parameter pin through the tail formulas).
5. Return levels.  REFERENCE: scipy.stats.genextreme.ppf(1 - 1/T),
   asserted at 1e-12 against the transcribed Coles (2001, eq. 3.4) closed
   form. Graded at 1e-5 relative in Rust.

CASE NOTES
----------
* "bounded_negxi" uses draws with true xi = -0.25, INSIDE the Smith (1985)
  regularity region xi > -0.5. A raw uniform tail has xi = -1, where the
  GPD MLE does not exist (the likelihood supremum is +inf at the boundary
  beta -> -xi max(z)) — that case cannot be pinned against any optimizer
  and is exercised as a Rust property test (irregularity flagged, SEs not
  certified) instead.
* "yield_abs_lr" / "yield_annual_max" reuse the committed
  fixtures/yield_curve_recession.csv (FRED GS10, monthly): absolute log
  returns x100 — a real, persistent, heavy-ish tailed series. NEVER
  fetched from the network.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  /home/user/tsecon/.venv/bin/python fixtures/generate_tsecon-evt_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import genextreme, genpareto

HERE = Path(__file__).resolve().parent
OUT = HERE / "tsecon-evt.json"

EPS = np.finfo(float).eps


# ------------------------------------------------------- sign conventions
def verify_sign_conventions() -> None:
    z = np.array([0.05, 0.3, 1.2, 2.5, 7.0])
    for xi in (0.3, -0.25):
        beta = 1.4
        with np.errstate(invalid="ignore"):
            hand = -np.log(beta) - (1.0 + 1.0 / xi) * np.log1p(xi * z / beta)
        ok = np.isfinite(hand)  # points inside the support for this xi
        ref = genpareto.logpdf(z[ok], xi, loc=0.0, scale=beta)
        assert np.allclose(hand[ok], ref, rtol=0, atol=1e-12), "genpareto c != xi ?!"
    x = np.array([-0.5, 0.2, 1.1, 3.0])
    for xi in (0.2, -0.2):
        mu, sig = 0.1, 1.3
        t = (x - mu) / sig
        a = np.log1p(xi * t) / xi
        hand = -np.log(sig) - (1.0 + xi) * a - np.exp(-a)
        ref = genextreme.logpdf(x, -xi, loc=mu, scale=sig)
        ok = np.isfinite(hand)
        assert np.allclose(hand[ok], ref[ok], rtol=0, atol=1e-12), "genextreme c != -xi ?!"


# ------------------------------------------------------------- utilities
def polish(nll, x0):
    """Two tight Nelder-Mead passes from x0; asserts monotone improvement."""
    f0 = nll(x0)
    opts = dict(xatol=1e-12, fatol=1e-13, maxiter=100_000, maxfev=100_000)
    r = minimize(nll, x0, method="Nelder-Mead", options=opts)
    r = minimize(nll, r.x, method="Nelder-Mead", options=opts)
    assert nll(r.x) <= f0 + 1e-12, "polish made the likelihood worse?!"
    return np.asarray(r.x, float)


def hess_central4(nll, x, scales):
    """statsmodels approx_hess3 stencil: four-point central cross
    differences with h_i = eps^(1/4) * scales[i] — the same scheme the
    Rust side uses, so both converge to the same observed information."""
    x = np.asarray(x, float)
    h = EPS ** 0.25 * np.asarray(scales, float)
    k = len(x)
    H = np.empty((k, k))
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k)
            ej = np.zeros(k)
            ei[i] = h[i]
            ej[j] = h[j]
            v = (
                nll(x + ei + ej)
                - nll(x + ei - ej)
                - nll(x - ei + ej)
                + nll(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    return H


def yield_abs_log_returns():
    lines = [
        ln.strip()
        for ln in (HERE / "yield_curve_recession.csv").read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    header = lines[0].split(",")
    col = header.index("gs10")
    gs10 = np.array([float(ln.split(",")[col]) for ln in lines[1:]])
    return 100.0 * np.abs(np.diff(np.log(gs10)))


# ------------------------------------------------------------------- GPD
def gpd_case(name, y, quantile=None, threshold=None, p_tail=(0.99, 0.995, 0.999)):
    y = np.asarray(y, float)
    n = len(y)
    if threshold is None:
        u = float(np.quantile(y, quantile))
        q_reported = quantile
    else:
        u = float(threshold)
        q_reported = float(np.sum(y <= u)) / n
    z = y[y > u] - u
    n_u = len(z)
    assert n_u >= 10, f"{name}: only {n_u} exceedances"
    rate = n_u / n

    c_raw, loc_raw, s_raw = genpareto.fit(z, floc=0.0)
    assert loc_raw == 0.0

    def nll(p):
        c, ls = p
        ll = genpareto.logpdf(z, c, loc=0.0, scale=np.exp(ls))
        return -ll.sum() if np.all(np.isfinite(ll)) else np.inf

    c_pol, ls_pol = polish(nll, [c_raw, np.log(s_raw)])
    xi, beta = float(c_pol), float(np.exp(ls_pol))
    loglik = float(genpareto.logpdf(z, xi, loc=0.0, scale=beta).sum())

    def nll_orig(p):
        c, s = p
        if s <= 0:
            return np.inf
        ll = genpareto.logpdf(z, c, loc=0.0, scale=s)
        return -ll.sum() if np.all(np.isfinite(ll)) else np.inf

    H = hess_central4(nll_orig, [xi, beta], [max(abs(xi), 0.1), beta])
    cov = np.linalg.inv(H)
    assert np.all(np.diag(cov) > 0), f"{name}: observed information not PD"
    se_xi, se_beta = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))

    var_list, es_list = [], []
    for p in p_tail:
        q_gpd = 1.0 - (1.0 - p) / rate
        v_scipy = u + float(genpareto.ppf(q_gpd, xi, loc=0.0, scale=beta))
        v_closed = u + beta / xi * (((1.0 - p) / rate) ** (-xi) - 1.0)
        assert abs(v_scipy - v_closed) <= 1e-12 * max(1.0, abs(v_scipy)), (
            f"{name}: ppf route and closed form disagree at p={p}"
        )
        e_closed = (v_scipy + beta - xi * u) / (1.0 - xi)
        # Independent cross-check: ES_p = (1/(1-p)) * int_p^1 VaR_q dq,
        # via the substitution q = 1 - (1-p)s (s in (0,1]).
        e_num = quad(
            lambda s: u
            + float(genpareto.ppf(1.0 - (1.0 - p) * s / rate, xi, 0.0, beta)),
            0.0,
            1.0,
            limit=200,
        )[0]
        assert abs(e_num - e_closed) <= 1e-6 * abs(e_closed), (
            f"{name}: ES closed form vs quad: {e_closed} vs {e_num}"
        )
        var_list.append(v_scipy)
        es_list.append(e_closed)

    return {
        "name": name,
        "y": [float(v) for v in y],
        "threshold_arg": None if threshold is None else float(threshold),
        "quantile_arg": None if quantile is None else float(quantile),
        "threshold": u,
        "threshold_quantile": float(q_reported),
        "n": n,
        "n_exceed": n_u,
        "exceed_rate": rate,
        "scipy_fit_raw": [float(c_raw), float(s_raw)],
        "xi": xi,
        "beta": beta,
        "loglik": loglik,
        "se_xi": se_xi,
        "se_beta": se_beta,
        "p_tail": list(p_tail),
        "var": var_list,
        "es": es_list,
    }


# ------------------------------------------------------------------- GEV
def gev_case(name, y, block_size=None, return_periods=(10.0, 50.0, 100.0)):
    y = np.asarray(y, float)
    if block_size is None:
        maxima = y.copy()
    else:
        nb = len(y) // block_size
        maxima = y[: nb * block_size].reshape(nb, block_size).max(axis=1)
    nm = len(maxima)
    assert nm >= 10, f"{name}: only {nm} maxima"

    c_raw, loc_raw, s_raw = genextreme.fit(maxima)

    def nll(p):
        c, loc, ls = p
        ll = genextreme.logpdf(maxima, c, loc=loc, scale=np.exp(ls))
        return -ll.sum() if np.all(np.isfinite(ll)) else np.inf

    c_pol, loc_pol, ls_pol = polish(nll, [c_raw, loc_raw, np.log(s_raw)])
    xi, mu, sigma = float(-c_pol), float(loc_pol), float(np.exp(ls_pol))
    loglik = float(genextreme.logpdf(maxima, -xi, loc=mu, scale=sigma).sum())

    def nll_orig(p):
        x, m, s = p
        if s <= 0:
            return np.inf
        ll = genextreme.logpdf(maxima, -x, loc=m, scale=s)
        return -ll.sum() if np.all(np.isfinite(ll)) else np.inf

    H = hess_central4(
        nll_orig, [xi, mu, sigma], [max(abs(xi), 0.1), sigma, sigma]
    )
    cov = np.linalg.inv(H)
    assert np.all(np.diag(cov) > 0), f"{name}: observed information not PD"
    se = np.sqrt(np.diag(cov))

    rl = []
    for T in return_periods:
        q = 1.0 - 1.0 / T
        z_scipy = float(genextreme.ppf(q, -xi, loc=mu, scale=sigma))
        z_closed = mu + sigma / xi * ((-np.log(q)) ** (-xi) - 1.0)
        assert abs(z_scipy - z_closed) <= 1e-12 * max(1.0, abs(z_scipy)), (
            f"{name}: return level ppf vs closed form at T={T}"
        )
        rl.append(z_scipy)

    return {
        "name": name,
        "y": [float(v) for v in y],
        "block_size": block_size,
        "n_maxima": nm,
        "scipy_fit_raw": [float(c_raw), float(loc_raw), float(s_raw)],
        "xi": xi,
        "mu": mu,
        "sigma": sigma,
        "loglik": loglik,
        "se_xi": float(se[0]),
        "se_mu": float(se[1]),
        "se_sigma": float(se[2]),
        "return_periods": list(return_periods),
        "return_levels": rl,
    }


def main() -> None:
    verify_sign_conventions()
    yc = yield_abs_log_returns()

    gpd_cases = [
        # Heavy tail: Student-t(3), true tail index xi = 1/3.
        gpd_case("t3_heavy", np.random.default_rng(42).standard_t(3, 1500), 0.90),
        # xi = 0 boundary: exponential draws.
        gpd_case("exponential", np.random.default_rng(7).exponential(1.0, 1500), 0.90),
        # Bounded tail, true xi = -0.25 (inside the regular region; see header).
        gpd_case(
            "bounded_negxi",
            genpareto.rvs(-0.25, size=1500, random_state=np.random.default_rng(11)),
            0.90,
        ),
        # Real series: FRED GS10 absolute log returns x100 (committed CSV).
        gpd_case("yield_abs_lr", yc, 0.90),
        # Explicit-threshold path on exponential draws.
        gpd_case(
            "expon_explicit_u",
            np.random.default_rng(13).exponential(1.0, 1200),
            threshold=2.0,
        ),
    ]

    gev_cases = [
        # Heavy tail: block maxima of t(3), 100 blocks of 30.
        gev_case("t3_blocks", np.random.default_rng(5).standard_t(3, 3000), 30),
        # Gumbel-limit region: block maxima of exponentials, 100 blocks of 50.
        gev_case(
            "exponential_blocks", np.random.default_rng(9).exponential(1.0, 5000), 50
        ),
        # Pre-computed-maxima path: direct GEV(xi=-0.25) draws.
        gev_case(
            "gev_negxi_maxima",
            genextreme.rvs(0.25, size=200, random_state=np.random.default_rng(21)),
            None,
        ),
        # Real series: annual maxima of monthly absolute log returns.
        gev_case("yield_annual_max", yc, 12),
    ]

    out = {
        "_meta": {
            "generator": "fixtures/generate_tsecon-evt_fixtures.py",
            "reference": (
                "scipy.stats.genpareto.fit(z, floc=0) / genextreme.fit(maxima), "
                "Nelder-Mead-polished in-generator (rv_continuous.fit stops at "
                "xtol=1e-4); observed-information SEs from a numpy central-4 "
                "Hessian; VaR/ES/return levels from genpareto.ppf / "
                "genextreme.ppf + the documented McNeil-Frey / Coles closed "
                "forms (asserted equal at 1e-12; ES cross-checked by "
                "quadrature at 1e-6)"
            ),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "gpd": gpd_cases,
        "gev": gev_cases,
    }
    OUT.write_text(json.dumps(out) + "\n")
    print(f"wrote {OUT}")
    for c in gpd_cases:
        print(
            f"  gpd {c['name']:>16}: xi={c['xi']:+.6f} beta={c['beta']:.6f} "
            f"ll={c['loglik']:.6f} se_xi={c['se_xi']:.5f} n_u={c['n_exceed']}"
        )
    for c in gev_cases:
        print(
            f"  gev {c['name']:>18}: xi={c['xi']:+.6f} mu={c['mu']:.6f} "
            f"sigma={c['sigma']:.6f} ll={c['loglik']:.6f}"
        )


if __name__ == "__main__":
    main()
