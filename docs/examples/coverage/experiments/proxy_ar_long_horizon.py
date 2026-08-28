"""Can any correction recover `proxy_ar_sets`' long-horizon coverage? (audit round 6, finding 8)

    .venv/bin/python docs/examples/coverage/experiments/proxy_ar_long_horizon.py            # full, ~15 min
    .venv/bin/python docs/examples/coverage/experiments/proxy_ar_long_horizon.py --quick    # smoke, ~1 min

THE PROBLEM BEING MEASURED
--------------------------
Round 6 measured `proxy_ar_sets`' propagated coverage continuing to decline
past the card's published table: 0.876-0.894 at the default `horizon=12` on
the card's own VAR(2) DGP, 0.80-0.85 on a routine VAR(1) at T=250, one-sided
(the truth lands ABOVE the set), fading in T. The recorded mechanism: the
delta-method reduced-form variance is evaluated at the estimated coefficients,
so when the fitted VAR draws less persistent than the truth, Psi_hat_h shrinks
AND the variance propagated for it shrinks with it -- the two errors are
positively coupled, and at long h the coupling is strong enough to push a 95%
set to ~0.85-0.88.

This script measures that mechanism directly and then measures five candidate
corrections ON THE SAME DRAWS, so differences between arms are differences in
the method, not in the noise:

  delta    the shipped construction (first-order delta at theta_hat) -- baseline
  delta2   second-order propagation: S Gaussian coefficient draws
           alpha* ~ N(alpha_hat, Cov(alpha_hat)); psi_var = T_O Cov*(Psi(alpha*) gamma_hat).
           Equal to `delta` to first order; additionally carries the convexity
           of alpha -> Psi_h, which first-order delta drops and which grows
           with h. (candidate direction 1: a second-order delta correction)
           THE MEASURED WINNER -- shipped as
           proxy_ar_sets(..., rf_method="second_order") via
           tsecon_ident::proxy_ar::psi_reduced_form_cov_mc; the full verdict
           table is in docs/roadmap/21-long-horizon-and-joint-inference.md.
  bcvar    the shipped delta variance evaluated at Pope (1990) bias-corrected
           coefficients instead of alpha_hat -- decoupling the variance from
           the downward-biased estimate without touching the estimate itself
           (candidate direction 2: decouple the variance from the estimate)
  delta2bc the combination arm note 21's verdict named as the natural next
           candidate for the residual ~2pp: the second-order propagation
           (delta2) with the Gaussian coefficient draws CENTERED AT the
           Pope-bias-corrected coefficients -- the convexity channel and the
           evaluation-point channel at once. Like every other arm it changes
           only psi_var (v0), so the boundedness statistic and the weak-IV
           argument are untouched. (added 2026-08: the follow-up
           investigation of the residual gap; verdict appended to note 21)
  floor    a variance floor tied to the estimate's own scale: the RELATIVE
           reduced-form variance rel_h^2 = psi_var[h]_ii / (Psi_hat_h gamma_hat)_i^2
           made monotone non-decreasing in h (error compounds with the
           horizon; the delta estimate can violate this exactly in the draws
           that miss) (candidate direction 2: a scale-tied floor)
  boot-v   parametric-bootstrap variance: B refits of the fitted VAR;
           psi_var[h] = T_O Cov*(Psi*_h gamma_hat) (candidate direction 3)
  boot-c   parametric-bootstrap CALIBRATION of the whole statistic: the same B
           refits, but used to replace the chi2(1) critical value with the
           per-cell 1-alpha quantile of AR*(lambda_true*), where the bootstrap
           truth lambda* = lambda(theta_hat) is known by construction
           (candidate direction 3, applied to the statistic rather than to
           one variance term)

Every arm's psi_var is quadratic in gamma_hat (or, for boot-c, its critical
value is a calibration of the same statistic), so each candidate keeps the
weak-instrument robustness argument intact: the correction still vanishes with
the instrument's relevance. The weak arm is measured anyway, not asserted.

VALIDATION BEFORE ANY COVERAGE IS COUNTED (the round-6 harness rule)
--------------------------------------------------------------------
1. The estimand is validated against a T=200,000 fit: the plug-in
   lambda_hat(h,i) from one huge-sample fit must agree with the closed-form
   truth at every cell through h=12, or nothing below is a coverage number.
2. When `tsecon` is importable, this file's NumPy transcription of the AR set
   (identical to fixtures/generate_proxy_ar_fixtures.py's) is cross-checked
   against `tsecon.proxy_ar_sets` endpoint-for-endpoint on one dataset.
   This file never uses tsecon for the experiment itself.

DGPs (both with a strong instrument, phi = 1.0, sig_nu = 1.5, norm_var 0)
-------------------------------------------------------------------------
  card_var2     the card's own DGP: the 3-variable VAR(2) of
                fixtures/generate_proxy_ar_fixtures.py, T = 300 (spectral
                radius 0.68) -- the configuration behind the published table
  routine_var1  a routine 3-variable VAR(1), T = 250, own lags .65/.55/.45,
                mild cross terms (spectral radius ~0.70)

plus a weak-instrument guard arm (phi = 0.06) on card_var2, because any
candidate that buys long-horizon coverage by breaking weak-IV conservatism is
disqualified no matter what its h=12 number says.

WHAT IS REPORTED PER ARM
------------------------
Coverage by horizon (excluding the degenerate (norm_var, h=0) cell, which
covers with probability 1 by construction), the h=8 and h=12 rows called out,
the DIRECTION of the misses at h=12 (above/below), the paired median width
ratio to the delta baseline at h in {8, 12} (the price), the bounded-cell
fraction, and per-arm runtime. Nominal is 0.95 throughout; MC standard error
at 500 reps is ~0.010 on a by-horizon row.

Seeded end to end; nothing here imports the Rust crate for the measurement.
"""

import argparse
import sys
import time

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------

CARD_H = np.array(
    [
        [1.0, 0.4, 0.2],
        [0.5, 1.2, 0.3],
        [0.3, 0.5, 0.9],
    ]
)
CARD_A = [
    np.array(
        [
            [0.50, 0.10, 0.00],
            [0.00, 0.40, 0.10],
            [0.10, 0.00, 0.30],
        ]
    ),
    np.array(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.00, 0.00, 0.10],
        ]
    ),
]

ROUTINE_H = np.array(
    [
        [1.0, 0.3, 0.1],
        [0.4, 1.1, 0.2],
        [0.2, 0.4, 0.8],
    ]
)
ROUTINE_A = [
    np.array(
        [
            [0.65, 0.15, 0.00],
            [0.00, 0.55, 0.15],
            [0.10, 0.00, 0.45],
        ]
    )
]

DGPS = {
    "card_var2": {"H": CARD_H, "A": CARD_A, "T": 300, "lags": 2},
    "routine_var1": {"H": ROUTINE_H, "A": ROUTINE_A, "T": 250, "lags": 1},
}


def simulate(rng, n_obs, coefs, h_mat, phi, sig_nu, burn=500):
    """VAR(p) y_t = sum A_i y_{t-i} + H eps_t with a proxy for shock 0."""
    p = len(coefs)
    n = h_mat.shape[0]
    total = n_obs + burn
    eps = rng.standard_normal((total, n))
    u = eps @ h_mat.T
    y = np.zeros((total, n))
    for t in range(p, total):
        acc = u[t].copy()
        for i, a in enumerate(coefs):
            acc += a @ y[t - 1 - i]
        y[t] = acc
    m = phi * eps[burn:, 0] + sig_nu * rng.standard_normal(n_obs)
    return y[burn:], m


def ma_from_coefs(coefs, horizon):
    """Psi_0 = I, Psi_h = sum_{i=1..min(h,p)} Psi_{h-i} A_i."""
    p = len(coefs)
    n = coefs[0].shape[0]
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc += psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return np.array(psi)


def true_lambda(psi_true, h_mat, norm_var, unit, horizon):
    """gamma = E[m u'] = phi H[:,0]; phi cancels in the ratio."""
    hcol = h_mat[:, 0]
    return np.array(
        [unit * (psi_true[h] @ hcol) / hcol[norm_var] for h in range(horizon + 1)]
    )


# ---------------------------------------------------------------------------
# AR-set machinery (transcribed from fixtures/generate_proxy_ar_fixtures.py;
# cross-checked against tsecon.proxy_ar_sets below when tsecon is importable)
# ---------------------------------------------------------------------------


def fit_var_ols(y, lags):
    """OLS VAR with a constant: residuals, coefs (p,n,n), lag-block cov, zz_inv, sigma_u."""
    t_all, n = y.shape
    t = t_all - lags
    x = np.ones((t, 1 + n * lags))
    for i in range(1, lags + 1):
        x[:, 1 + (i - 1) * n : 1 + i * n] = y[lags - i : t_all - i]
    zz_inv = np.linalg.inv(x.T @ x)
    beta = zz_inv @ x.T @ y[lags:]
    resid = y[lags:] - x @ beta
    coefs = np.array([beta[1 + (i - 1) * n : 1 + i * n].T for i in range(1, lags + 1)])
    sigma_u = resid.T @ resid / (t - (1 + n * lags))
    cov_alpha = np.kron(zz_inv[1:, 1:], sigma_u)
    return resid, coefs, cov_alpha, sigma_u


def moment_pieces(uu, m):
    """Overlap moments and the HC0 moment covariance Omega."""
    ok = np.isfinite(m)
    mo, uo = m[ok], uu[ok]
    t_o = mo.size
    md = mo - mo.mean()
    ud = uo - uo.mean(axis=0)
    g = md[:, None] * ud
    gamma = g.mean(axis=0)
    gt = g - gamma
    return t_o, gamma, gt.T @ gt / t_o


def companion(coefs):
    p, n, _ = coefs.shape
    c = np.zeros((n * p, n * p))
    for i in range(p):
        c[:n, i * n : (i + 1) * n] = coefs[i]
    if p > 1:
        c[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return c


def psi_reduced_form_cov(psi, coefs, cov_alpha, gamma, t_o):
    """T_O * Cov(Psi_hat_h gamma) per horizon by the first-order delta method."""
    p, n, _ = coefs.shape
    horizon = psi.shape[0] - 1
    at = companion(coefs).T
    kp = at.shape[0]
    atpow, cur = [], np.eye(kp)
    for _ in range(horizon):
        atpow.append(cur[:n, :].copy())
        cur = cur @ at
    out = [np.zeros((n, n))]
    for h in range(1, horizon + 1):
        gh = np.zeros((n * n, n * kp))
        for m in range(h):
            gh += np.kron(atpow[h - 1 - m], psi[m])
        gm = np.zeros((n, n * kp))
        for i in range(n):
            for j in range(n):
                gm[i] += gamma[j] * gh[j * n + i]
        out.append(t_o * (gm @ cov_alpha @ gm.T))
    return out


def stable_roots(a, b, c, d):
    sq = np.sqrt(max(d, 0.0))
    s = -b - (-1.0 if b < 0 else 1.0) * sq
    r1 = s / (2.0 * a)
    r2 = 2.0 * c / s
    return (r1, r2) if r1 <= r2 else (r2, r1)


def classify(a, b, c, point, tau_a, tau_b, tau_c, tau_d):
    if abs(a) <= tau_a:
        if abs(b) <= tau_b:
            return ("whole", None, None) if c <= tau_c else ("empty", None, None)
        root = -c / b
        return ("ray_below", None, root) if b > 0 else ("ray_above", root, None)
    d = b * b - 4.0 * a * c
    if a > 0:
        if d > tau_d:
            lo, hi = stable_roots(a, b, c, d)
            return ("interval", lo, hi)
        if d >= -tau_d:
            return ("point", point, point)
        raise AssertionError("negative discriminant with A > 0")
    if d > tau_d:
        lo, hi = stable_roots(a, b, c, d)
        return ("exterior", lo, hi)
    return ("whole", None, None)


def contains(kind, lo, hi, lam):
    if kind == "interval":
        return lo <= lam <= hi
    if kind == "point":
        return lam == lo
    if kind == "ray_below":
        return lam <= hi
    if kind == "ray_above":
        return lam >= lo
    if kind == "exterior":
        return lam <= lo or lam >= hi
    return kind == "whole"


def ar_sets(uu, m, psi, norm_var, unit, crit, psi_var=None, crit_cells=None):
    """Every cell's AR set.

    `psi_var` is None (moment-only) or the per-horizon T_O Cov(Psi_hat_h gamma)
    list (the reduced-form correction; the cross term is zero here, exactly as
    the shipped binding passes it). `crit_cells` optionally overrides the
    critical value per (h, i) cell -- the boot-c arm; `crit` is then unused
    inside the loop but still reported.
    """
    t_o, gamma, omega = moment_pieces(uu, m)
    n = uu.shape[1]
    horizon = psi.shape[0] - 1
    q0 = float(gamma[norm_var])
    v2 = float(omega[norm_var, norm_var])
    b_impact = unit * (gamma / q0)
    cells = []
    for h in range(horizon + 1):
        p_h = psi[h] @ omega
        row = []
        for i in range(n):
            c_hi = crit if crit_cells is None else float(crit_cells[h][i])
            q1 = unit * float(psi[h][i] @ gamma)
            v0 = unit * unit * float(p_h[i] @ psi[h][i])
            v1 = unit * float(p_h[i, norm_var])
            if psi_var is not None:
                v0 += unit * unit * float(psi_var[h][i, i])
            a = t_o * q0 * q0 - c_hi * v2
            b = 2.0 * (c_hi * v1 - t_o * q1 * q0)
            c = t_o * q1 * q1 - c_hi * v0
            point = float(psi[h][i] @ b_impact)
            tau_a = 1e-12 * max(t_o * q0 * q0, c_hi * v2)
            tau_b = 1e-12 * max(2.0 * abs(c_hi * v1), 2.0 * abs(t_o * q1 * q0))
            tau_c = 1e-12 * max(t_o * q1 * q1, c_hi * v0)
            tau_d = 1e-12 * max(b * b, abs(4.0 * a * c))
            kind, lo, hi = classify(a, b, c, point, tau_a, tau_b, tau_c, tau_d)
            row.append({"kind": kind, "lo": lo, "hi": hi, "point": point})
        cells.append(row)
    return {"t_o": t_o, "gamma": gamma, "omega": omega, "cells": cells}


# ---------------------------------------------------------------------------
# Candidate psi_var producers
# ---------------------------------------------------------------------------


def psi_var_delta2(rng, coefs, cov_alpha, gamma, t_o, horizon, s_draws=256):
    """Second-order propagation: Gaussian coefficient draws, exact Psi map.

    Antithetic pairs alpha_hat +/- L z. The alpha layout matches cov_alpha:
    index (l*n + j)*n + e  <->  A_l[e, j].
    """
    p, n, _ = coefs.shape
    dim = p * n * n
    chol = np.linalg.cholesky(cov_alpha + 1e-30 * np.eye(dim))
    half = s_draws // 2
    z = rng.standard_normal((half, dim))
    dev = z @ chol.T
    alpha_hat = np.empty(dim)
    for l in range(p):
        for j in range(n):
            for e in range(n):
                alpha_hat[(l * n + j) * n + e] = coefs[l][e, j]
    draws = np.concatenate([alpha_hat + dev, alpha_hat - dev])  # (S, dim)
    s = draws.shape[0]
    coefs_s = np.empty((s, p, n, n))
    for l in range(p):
        block = draws[:, l * n * n : (l + 1) * n * n].reshape(s, n, n)  # [j, e]
        coefs_s[:, l] = np.transpose(block, (0, 2, 1))  # [e, j]
    psi_s = np.empty((s, horizon + 1, n, n))
    psi_s[:, 0] = np.eye(n)
    for h in range(1, horizon + 1):
        acc = np.zeros((s, n, n))
        for i in range(1, min(h, p) + 1):
            acc += np.einsum("sij,sjk->sik", psi_s[:, h - i], coefs_s[:, i - 1])
        psi_s[:, h] = acc
    w = np.einsum("shij,j->shi", psi_s, gamma)  # (S, H+1, n)
    wc = w - w.mean(axis=0)
    out = [np.zeros((n, n))]
    for h in range(1, horizon + 1):
        out.append(t_o * (wc[:, h].T @ wc[:, h]) / (s - 1))
    return out


def pope_bias_correct(coefs, sigma_u, t_eff):
    """Pope (1990) first-order bias correction on the companion matrix,
    with Kilian's (1998) stationarity adjustment.

    b = -B/T, B = Sigma_U [ (I-A')^-1 + A'(I-A'^2)^-1 + sum_l l (I - l A')^-1 ] G0^-1,
    G0 the companion variance. Applied only when the fitted companion is
    stable; the corrected companion is shrunk back toward the fit if the
    correction pushes it explosive.
    """
    p, n, _ = coefs.shape
    a = companion(coefs)
    kp = a.shape[0]
    lam = np.linalg.eigvals(a)
    if np.max(np.abs(lam)) >= 1.0:
        return coefs  # no correction for an unstable fit
    sig_c = np.zeros((kp, kp))
    sig_c[:n, :n] = sigma_u
    # G0 = A G0 A' + Sigma_c  (vectorized discrete Lyapunov)
    g0 = np.linalg.solve(np.eye(kp * kp) - np.kron(a, a), sig_c.reshape(-1)).reshape(
        kp, kp
    )
    at = a.T
    eye = np.eye(kp)
    core = np.linalg.inv(eye - at) + at @ np.linalg.inv(eye - at @ at)
    for l in lam:
        core = core + l * np.linalg.inv(eye - l * at)
    bias = -(sig_c @ core.real @ np.linalg.inv(g0)) / t_eff
    corr = a[:n, :] - bias[:n, :]  # A_tilde = A_hat - bias(A_hat)
    # Kilian stationarity adjustment: shrink the correction until stable.
    delta = 1.0
    while delta > 0.0:
        cand = a.copy()
        cand[:n, :] = a[:n, :] + delta * (corr - a[:n, :])
        if np.max(np.abs(np.linalg.eigvals(cand))) < 1.0:
            break
        delta -= 0.05
    else:
        return coefs
    out = np.array([cand[:n, i * n : (i + 1) * n] for i in range(p)])
    return out


def psi_var_floor(psi_var, psi, gamma):
    """Monotone relative-variance floor on the diagonal.

    rel_h(i) = psi_var[h][i,i] / (Psi_hat_h gamma)_i^2 forced non-decreasing
    in h; cells whose (Psi_hat_h gamma)_i passes through zero are left alone
    (the relative scale is meaningless there).
    """
    horizon = len(psi_var) - 1
    n = psi_var[0].shape[0]
    out = [pv.copy() for pv in psi_var]
    rel_prev = np.zeros(n)
    for h in range(1, horizon + 1):
        w = psi[h] @ gamma
        for i in range(n):
            denom = w[i] * w[i]
            if denom <= 0.0:
                continue
            rel = out[h][i, i] / denom
            if rel < rel_prev[i]:
                out[h][i, i] = rel_prev[i] * denom
                rel = rel_prev[i]
            rel_prev[i] = rel
    return out


def simulate_boot_batch(rng, b_draws, t_all, coefs, intercept, joint_cov, burn=100):
    """B parametric-bootstrap series from the fitted VAR, with the proxy drawn
    jointly Gaussian with the innovations (Cov(m, u) = gamma_hat).

    Returns y* (B, T, n) and m* (B, T).
    """
    p, n, _ = coefs.shape
    total = t_all + burn
    chol = np.linalg.cholesky(joint_cov + 1e-12 * np.trace(joint_cov) * np.eye(n + 1))
    zdraw = rng.standard_normal((b_draws, total, n + 1)) @ chol.T
    u = zdraw[:, :, :n]
    m = zdraw[:, :, n]
    y = np.zeros((b_draws, total, n))
    for t in range(p, total):
        acc = u[:, t] + intercept
        for i, a in enumerate(coefs):
            acc = acc + y[:, t - 1 - i] @ a.T
        y[:, t] = acc
    return y[:, burn:], m[:, burn:]


def fit_var_ols_batch(y, lags):
    """Batched OLS VAR with constant over the leading axis.

    y is (B, T, n); returns resid (B, T-p, n), coefs (B, p, n, n),
    zz_inv (B, k, k), sigma_u (B, n, n).
    """
    b_draws, t_all, n = y.shape
    t = t_all - lags
    k = 1 + n * lags
    x = np.ones((b_draws, t, k))
    for i in range(1, lags + 1):
        x[:, :, 1 + (i - 1) * n : 1 + i * n] = y[:, lags - i : t_all - i]
    xtx = np.einsum("btk,btl->bkl", x, x)
    xty = np.einsum("btk,btn->bkn", x, y[:, lags:])
    zz_inv = np.linalg.inv(xtx)
    beta = zz_inv @ xty  # (B, k, n)
    resid = y[:, lags:] - np.einsum("btk,bkn->btn", x, beta)
    coefs = np.empty((b_draws, lags, n, n))
    for i in range(1, lags + 1):
        coefs[:, i - 1] = np.transpose(beta[:, 1 + (i - 1) * n : 1 + i * n], (0, 2, 1))
    sigma_u = np.einsum("btn,btm->bnm", resid, resid) / (t - k)
    return resid, coefs, zz_inv, sigma_u


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

ARMS = ["delta", "delta2", "bcvar", "delta2bc", "floor", "boot-v", "boot-c"]


def run_dgp(name, spec, phi, seed, reps, horizon, level, boot_b, delta2_s, arms):
    coefs_true = spec["A"]
    h_mat = spec["H"]
    t_obs = spec["T"]
    lags = spec["lags"]
    n = h_mat.shape[0]
    norm_var, unit, sig_nu = 0, 1.0, 1.5
    crit = float(stats.chi2.ppf(level, 1))
    psi_true = ma_from_coefs(coefs_true, horizon)
    truth = true_lambda(psi_true, h_mat, norm_var, unit, horizon)

    # -- estimand validation against a huge-sample fit (round-6 harness rule)
    rng = np.random.default_rng(seed)
    y_big, _ = simulate(rng, 200_000, coefs_true, h_mat, phi=1.0, sig_nu=sig_nu)
    _, coefs_big, _, _ = fit_var_ols(y_big, lags)
    lam_big = true_lambda(ma_from_coefs(list(coefs_big), horizon), h_mat, norm_var, unit, horizon)
    # (uses H's col 0 as gamma-direction; the plug-in from the fit only replaces Psi)
    scale = np.abs(truth).max()
    est_err = np.abs(lam_big - truth).max() / scale
    assert est_err < 0.02, f"estimand validation failed: rel err {est_err:.4f}"
    print(f"[{name}] estimand validated: T=200k plug-in max rel err {est_err:.5f}")

    acc = {
        a: {
            "cov": np.zeros((horizon + 1, n)),
            "miss_above": np.zeros((horizon + 1, n)),
            "miss_below": np.zeros((horizon + 1, n)),
            "bounded": 0,
            "total": 0,
            "widths": {8: [], horizon: []},
            "wratio": {8: [], horizon: []},
            "time": 0.0,
        }
        for a in arms
    }
    diag_sd, diag_err = [], []  # h=12, i=0: hat-sd vs point error (delta arm)
    used = 0
    for r in range(reps):
        rng = np.random.default_rng(seed + 1000 + r)
        y, m = simulate(rng, t_obs, coefs_true, h_mat, phi, sig_nu)
        uu, coefs, cov_alpha, sigma_u = fit_var_ols(y, lags)
        mm = m[lags:]
        psi = ma_from_coefs(list(coefs), horizon)
        t_o, gamma, _ = moment_pieces(uu, mm)

        producers = {}
        t0 = time.perf_counter()
        pv_delta = psi_reduced_form_cov(psi, coefs, cov_alpha, gamma, t_o)
        t_delta = time.perf_counter() - t0
        producers["delta"] = (pv_delta, None, t_delta)

        if "delta2" in arms:
            t0 = time.perf_counter()
            pv = psi_var_delta2(rng, coefs, cov_alpha, gamma, t_o, horizon, delta2_s)
            producers["delta2"] = (pv, None, time.perf_counter() - t0)
        if "bcvar" in arms or "delta2bc" in arms:
            t0 = time.perf_counter()
            coefs_bc = pope_bias_correct(coefs, sigma_u, t_o)
            t_bc = time.perf_counter() - t0
            if "bcvar" in arms:
                t0 = time.perf_counter()
                psi_bc = ma_from_coefs(list(coefs_bc), horizon)
                pv = psi_reduced_form_cov(psi_bc, coefs_bc, cov_alpha, gamma,
                                          t_o)
                producers["bcvar"] = (pv, None,
                                      time.perf_counter() - t0 + t_bc + t_delta)
            if "delta2bc" in arms:
                t0 = time.perf_counter()
                # A dedicated substream: the arm was added after the note-21
                # tables were published, and drawing from the shared `rng`
                # here would shift the boot arms' streams and silently break
                # the reproducibility of those published numbers.
                rng_bc = np.random.default_rng(seed + 424242 + r)
                pv = psi_var_delta2(rng_bc, coefs_bc, cov_alpha, gamma, t_o,
                                    horizon, delta2_s)
                producers["delta2bc"] = (pv, None,
                                         time.perf_counter() - t0 + t_bc)
        if "floor" in arms:
            t0 = time.perf_counter()
            pv = psi_var_floor(pv_delta, psi, gamma)
            producers["floor"] = (pv, None, time.perf_counter() - t0 + t_delta)

        if "boot-v" in arms or "boot-c" in arms:
            t0 = time.perf_counter()
            # joint innovation covariance over the overlap, one divisor
            ok = np.isfinite(mm)
            zjoint = np.column_stack([uu[ok], mm[ok]])
            zjoint = zjoint - zjoint.mean(axis=0)
            joint_cov = zjoint.T @ zjoint / zjoint.shape[0]
            intercept = np.zeros(n)  # DGP and fit are mean-zero; intercept ~ 0
            yb, mb = simulate_boot_batch(rng, boot_b, t_obs, coefs, intercept, joint_cov)
            residb, coefsb, zz_invb, sigma_ub = fit_var_ols_batch(yb, lags)
            mbb = mb[:, lags:]
            w_star = np.empty((boot_b, horizon + 1, n))
            ar_star = np.empty((boot_b, horizon + 1, n))
            for bidx in range(boot_b):
                psib = ma_from_coefs(list(coefsb[bidx]), horizon)
                w_star[bidx] = np.einsum("hij,j->hi", psib, gamma)
                if "boot-c" in arms:
                    ub, mbi = residb[bidx], mbb[bidx]
                    t_ob, gammab, omegab = moment_pieces(ub, mbi)
                    cov_alphab = np.kron(zz_invb[bidx][1:, 1:], sigma_ub[bidx])
                    pvb = psi_reduced_form_cov(psib, coefsb[bidx], cov_alphab, gammab, t_ob)
                    q0b = gammab[norm_var]
                    v2b = omegab[norm_var, norm_var]
                    for h in range(horizon + 1):
                        p_h = psib[h] @ omegab
                        for i in range(n):
                            # the bootstrap-world truth is the original fit
                            lam0 = float(psi[h][i] @ gamma) / gamma[norm_var] * unit
                            q1b = unit * float(psib[h][i] @ gammab)
                            v0b = unit * unit * (float(p_h[i] @ psib[h][i]) + float(pvb[h][i, i]))
                            v1b = unit * float(p_h[i, norm_var])
                            gmom = q1b - lam0 * q0b
                            vv = v0b - 2.0 * lam0 * v1b + lam0 * lam0 * v2b
                            ar_star[bidx, h, i] = t_ob * gmom * gmom / vv if vv > 0 else np.inf
            t_boot_shared = time.perf_counter() - t0
            if "boot-v" in arms:
                t0 = time.perf_counter()
                wc = w_star - w_star.mean(axis=0)
                pv = [t_o * (wc[:, h].T @ wc[:, h]) / (boot_b - 1) for h in range(horizon + 1)]
                pv[0] = np.zeros((n, n))
                producers["boot-v"] = (pv, None, time.perf_counter() - t0 + t_boot_shared)
            if "boot-c" in arms:
                t0 = time.perf_counter()
                # the degenerate (0, norm_var) cell is 0/0 in every draw; it is
                # overwritten with the chi2 value below, so zero it pre-quantile
                ar_star[:, 0, norm_var] = 0.0
                crit_cells = np.quantile(ar_star, level, axis=0)
                crit_cells[0] = crit  # Psi_0 = I: the bootstrap has nothing to add
                producers["boot-c"] = (pv_delta, crit_cells, time.perf_counter() - t0 + t_boot_shared + t_delta)

        used += 1
        for arm in arms:
            pv, crit_cells, dt = producers[arm]
            t0 = time.perf_counter()
            res = ar_sets(uu, mm, psi, norm_var, unit, crit, pv, crit_cells)
            acc[arm]["time"] += dt + (time.perf_counter() - t0)
            d = acc[arm]
            base = producers["delta"]
            res_base = None
            for h in range(horizon + 1):
                for i in range(n):
                    cell = res["cells"][h][i]
                    d["total"] += 1
                    if cell["kind"] in ("interval", "point"):
                        d["bounded"] += 1
                    if contains(cell["kind"], cell["lo"], cell["hi"], truth[h, i]):
                        d["cov"][h, i] += 1
                    else:
                        if cell["kind"] == "interval":
                            if truth[h, i] > cell["hi"]:
                                d["miss_above"][h, i] += 1
                            else:
                                d["miss_below"][h, i] += 1
                        elif cell["kind"] == "exterior":
                            d["miss_below"][h, i] += 1  # truth inside the rejected middle
                    if h in d["widths"] and cell["kind"] == "interval":
                        d["widths"][h].append(cell["hi"] - cell["lo"])
            if arm != "delta":
                res_base = ar_sets(uu, mm, psi, norm_var, unit, crit, base[0], None)
                for h in d["wratio"]:
                    for i in range(n):
                        cb = res_base["cells"][h][i]
                        ca = res["cells"][h][i]
                        if cb["kind"] == "interval" and ca["kind"] == "interval":
                            d["wratio"][h].append(
                                (ca["hi"] - ca["lo"]) / (cb["hi"] - cb["lo"])
                            )
            if arm == "delta":
                cell = res["cells"][horizon][1]
                if cell["kind"] == "interval":
                    diag_sd.append(np.sqrt(pv_delta[horizon][1, 1] / t_o) / abs(gamma[norm_var]))
                    diag_err.append(cell["point"] - truth[horizon, 1])

    # ---- report
    keep = [i for i in range(n) if i != norm_var]
    print(f"\n[{name}] phi={phi} T={t_obs} lags={lags} reps used {used}; nominal {level}")
    if diag_sd:
        rho_corr = np.corrcoef(diag_sd, np.abs(diag_err))[0, 1]
        sd_arr, err_arr = np.array(diag_sd), np.array(diag_err)
        print(
            f"  mechanism check (delta, h={horizon}, var 1): corr(hat-sd, |point err|) = {rho_corr:+.3f}; "
            f"mean hat-sd {sd_arr.mean():.4g} vs empirical sd of point {err_arr.std():.4g} "
            f"(ratio {sd_arr.mean()/err_arr.std():.3f}); mean point err {err_arr.mean():+.4g}"
        )
    header = "  arm      " + " ".join(f"h={h:<4d}" for h in range(horizon + 1))
    print(header + "  | mean(h>=1) h8    h12   above below bnd    wr8    wr12  s/rep")
    out = {}
    for arm in arms:
        d = acc[arm]
        by_h = np.empty(horizon + 1)
        by_h[0] = d["cov"][0, keep].mean() / used
        by_h[1:] = d["cov"][1:].mean(axis=1) / used
        mean_cov = float(
            (d["cov"][0, keep].sum() + d["cov"][1:].sum())
            / (used * (len(keep) + horizon * n))
        )
        mean_h1 = float(d["cov"][1:].sum() / (used * horizon * n))
        row_a = d["miss_above"][horizon].sum()
        row_b = d["miss_below"][horizon].sum()
        wr8 = float(np.median(d["wratio"][8])) if d["wratio"][8] else 1.0
        wr12 = float(np.median(d["wratio"][horizon])) if d["wratio"][horizon] else 1.0
        cells = " ".join(f"{x:.3f}" for x in by_h)
        print(
            f"  {arm:8s} {cells}  |   {mean_h1:.3f}   {by_h[8]:.3f} {by_h[horizon]:.3f} "
            f"{int(row_a):5d} {int(row_b):5d} {d['bounded']/max(d['total'],1):.3f} "
            f"{wr8:6.3f} {wr12:6.3f} {d['time']/used:6.3f}"
        )
        out[arm] = {
            "by_h": by_h.tolist(),
            "mean": mean_cov,
            "h8": float(by_h[8]),
            "h12": float(by_h[horizon]),
            "miss_above_h12": int(row_a),
            "miss_below_h12": int(row_b),
            "wr8": wr8,
            "wr12": wr12,
            "bounded": d["bounded"] / max(d["total"], 1),
            "s_per_rep": d["time"] / max(used, 1),
        }
    return out


def cross_check_against_tsecon(seed=20260818):
    """When tsecon is importable, pin this file's transcription to the crate."""
    try:
        import tsecon
    except ImportError:
        print("cross-check: tsecon not importable here; skipped (the fixture "
              "generator pins the same transcription)")
        return
    spec = DGPS["card_var2"]
    rng = np.random.default_rng(seed)
    y, m = simulate(rng, spec["T"], spec["A"], spec["H"], 1.0, 1.5)
    lags, horizon = spec["lags"], 12
    ts = tsecon.proxy_ar_sets(y, m, lags=lags, horizon=horizon, norm_var=0,
                              unit=1.0, alpha=0.05)
    uu, coefs, cov_alpha, _ = fit_var_ols(y, lags)
    mm = m[lags:]
    psi = ma_from_coefs(list(coefs), horizon)
    t_o, gamma, _ = moment_pieces(uu, mm)
    pv = psi_reduced_form_cov(psi, coefs, cov_alpha, gamma, t_o)
    crit = float(stats.chi2.ppf(0.95, 1))
    mine = ar_sets(uu, mm, psi, 0, 1.0, crit, pv)
    worst = 0.0
    for h in range(horizon + 1):
        for i in range(3):
            c_ts, c_my = ts["cells"][h][i], mine["cells"][h][i]
            assert c_ts["kind"] == c_my["kind"], (h, i, c_ts["kind"], c_my["kind"])
            if c_my["kind"] == "interval":
                for a, b in ((c_ts["lower"], c_my["lo"]), (c_ts["upper"], c_my["hi"])):
                    worst = max(worst, abs(a - b) / max(abs(b), 1e-12))
    assert worst < 1e-7, worst
    print(f"cross-check vs tsecon.proxy_ar_sets: all kinds equal, "
          f"endpoint max rel err {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--skip-weak", action="store_true")
    args = ap.parse_args()
    reps = args.reps or (40 if args.quick else 500)
    boot_b = 64 if args.quick else 256
    delta2_s = 64 if args.quick else 256
    arms = [a for a in args.arms.split(",") if a]
    horizon, level = 12, 0.95

    print(f"proxy_ar_sets long-horizon candidates | reps={reps} boot_b={boot_b} "
          f"delta2_s={delta2_s} arms={arms}")
    cross_check_against_tsecon()

    t0 = time.time()
    results = {}
    results["card_var2"] = run_dgp(
        "card_var2", DGPS["card_var2"], 1.0, 20260818, reps, horizon, level,
        boot_b, delta2_s, arms,
    )
    results["routine_var1"] = run_dgp(
        "routine_var1", DGPS["routine_var1"], 1.0, 20260819, reps, horizon, level,
        boot_b, delta2_s, arms,
    )
    if not args.skip_weak:
        wreps = max(reps // 2, 20)
        results["card_var2_weak"] = run_dgp(
            "card_var2_weak", DGPS["card_var2"], 0.06, 20260820, wreps, horizon,
            level, boot_b, delta2_s, arms,
        )
    print(f"\ntotal wall time {time.time()-t0:.0f}s")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
