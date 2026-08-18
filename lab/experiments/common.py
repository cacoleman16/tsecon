"""Shared helpers for the lab's end-to-end simulation study.

Everything here is deliberately small and boring: DGPs, loss functions,
calibration tests, and table/markdown writers used by exp01-exp05.  All
randomness flows through explicit numpy Generators seeded by the caller.

Not part of the public tsecon API.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../lab
RESULTS = os.path.join(LAB, "experiments", "results")

# make lab/ importable (prophet_lite package) and lab/laplace importable
for p in (LAB, os.path.join(LAB, "laplace")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------

def piecewise_seasonal(T, seed, slope1=0.5, slope2=-0.3, cp_frac=0.5,
                       period=12, amp=(4.0, 2.0), noise_sd=1.0,
                       outlier_frac=0.03, outlier_lo=6.0, outlier_hi=10.0,
                       outlier_until=None):
    """Piecewise-linear trend + fixed Fourier seasonal + N(0,1) noise
    + sparse additive outliers (prophet_lite's home-turf DGP).

    outlier_until : outliers only occur at t < outlier_until (None = anywhere).
    Returns (y, clean) with clean = trend + seasonal (no noise, no outliers).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=float)
    cp = int(round(cp_frac * T))
    trend = np.where(t < cp, slope1 * t, slope1 * cp + slope2 * (t - cp))
    seas = amp[0] * np.sin(2 * np.pi * t / period) \
        + amp[1] * np.cos(4 * np.pi * t / period)
    clean = trend + seas
    y = clean + noise_sd * rng.standard_normal(T)
    mask = rng.random(T) < outlier_frac
    if outlier_until is not None:
        mask[outlier_until:] = False
    n_out = int(mask.sum())
    if n_out:
        signs = rng.choice([-1.0, 1.0], n_out)
        sizes = rng.uniform(outlier_lo, outlier_hi, n_out)
        y[mask] += signs * sizes * noise_sd
    return y, clean


def simulate_garch_t(T, seed, omega=0.05, alpha=0.10, beta=0.85, nu=5.0):
    """GARCH(1,1) with standardized Student-t(nu) innovations.

    Returns (y, sigma) where sigma is the true conditional sd path
    (sigma[t] is sd of y[t] given the past).
    """
    rng = np.random.default_rng(seed)
    scale = np.sqrt((nu - 2.0) / nu)          # standardize t to unit variance
    z = rng.standard_t(nu, T) * scale
    y = np.empty(T)
    sig2 = np.empty(T)
    s2 = omega / (1.0 - alpha - beta)
    for t in range(T):
        sig2[t] = s2
        y[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return y, np.sqrt(sig2)


def garch_sigma_path(y, omega, alpha, beta):
    """One-step-ahead conditional sd path of a fitted GARCH(1,1) with the
    parameters FROZEN (zero-mean convention): s2[t] = sd^2 of y[t] | past."""
    T = len(y)
    sig2 = np.empty(T)
    s2 = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(T):
        sig2[t] = s2
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return np.sqrt(sig2)


# ---------------------------------------------------------------------------
# losses & calibration tests
# ---------------------------------------------------------------------------

def rmse(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(e):
    e = np.asarray(e, float)
    return float(np.mean(np.abs(e)))


def pinball(y, q, tau):
    """Mean pinball (quantile) score of quantile path q for series y."""
    u = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(u * (tau - (u < 0.0))))


def kupiec(hits, tau):
    """Kupiec (1995) unconditional-coverage LR test.

    hits : boolean array, 1{y_t <= q_t}.  Returns (hit_rate, LR, p_value).
    """
    from scipy.stats import chi2
    hits = np.asarray(hits, bool)
    n = hits.size
    n1 = int(hits.sum())
    n0 = n - n1
    p_hat = n1 / n
    def ll(p):
        with np.errstate(divide="ignore", invalid="ignore"):
            return n0 * np.log1p(-p) + n1 * np.log(p)
    if n1 in (0, n):
        lr = -2.0 * ll(tau)             # saturated ll is 0 at the boundary
    else:
        lr = -2.0 * (ll(tau) - ll(p_hat))
    return p_hat, float(lr), float(chi2.sf(lr, 1))


def nw_tstat(d, lags=None):
    """Newey-West t-stat that mean(d) = 0 (loss-differential test, DM-style).

    Used where tsecon.dm_test does not apply (non-squared/absolute losses,
    e.g. pinball differentials).  Returns (tstat, p_value).
    """
    from scipy.stats import norm
    d = np.asarray(d, float)
    n = d.size
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    dbar = d.mean()
    u = d - dbar
    s = float(u @ u) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        s += 2.0 * w * float(u[k:] @ u[:-k]) / n
    se = np.sqrt(max(s, 1e-300) / n)
    t = dbar / se
    return float(t), float(2.0 * norm.sf(abs(t)))


# ---------------------------------------------------------------------------
# markdown table + results writers
# ---------------------------------------------------------------------------

def md_table(headers, rows, floatfmt="{:.4f}"):
    """Render a GitHub markdown table; floats formatted, rest str()'d."""
    def cell(x):
        if isinstance(x, float):
            return floatfmt.format(x)
        return str(x)
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(x) for x in r) + " |")
    return "\n".join(lines)


def write_results(name, markdown, payload=None):
    """Write results/<name>.md (and .json if payload given); echo the md."""
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".md"), "w") as f:
        f.write(markdown.rstrip() + "\n")
    if payload is not None:
        def default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(type(o))
        with open(os.path.join(RESULTS, name + ".json"), "w") as f:
            json.dump(payload, f, indent=1, default=default)
    print(markdown)
