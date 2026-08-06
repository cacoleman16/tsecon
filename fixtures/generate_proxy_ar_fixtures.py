"""Golden fixtures for weak-instrument-robust (Anderson-Rubin) confidence sets
for proxy-SVAR impulse responses.

VALIDATION STRATEGY
===================
No external Python package exposes an Anderson-Rubin confidence set for the
proxy-SVAR estimand, so there is no third-party implementation to diff
against. The validation therefore rests on three legs, none of which touches
the tsecon Rust crate (this file NEVER imports tsecon):

1. DOCUMENTED-FORMULA GOLDEN. A plain-NumPy transcription of the moment
   condition and its variance, written straight from the algebra rather than
   from the Rust source. The reduced form comes from statsmodels VAR (an
   independent implementation of the OLS fit and its MA representation).

2. BRUTE-FORCE GRID PROOF OF THE CLOSED-FORM INVERSION. This is the strongest
   check available here and it is cheap. For every (horizon, variable) cell we
   evaluate the AR statistic directly on a fine grid of candidate values and
   compare the accepted grid points to what the closed-form quadratic says the
   set is. Where the grid brackets a boundary we bisect AR(lam) - c to machine
   precision and compare the refined crossing to the closed-form root. Every
   one of the shapes the set can take -- bounded interval, single point, union
   of two rays, half-line, whole line -- is exercised.

3. SEEDED MONTE-CARLO COVERAGE, STRONG *AND* WEAK. The weak arm is the entire
   point: a Wald interval under-covers badly there, and the AR set must not.
   Both arms are run with the MA matrices known (which is what the moment-only
   HC0 variance assumes) and with the VAR estimated from simulated data. The
   estimated-VAR runs are done TWICE inside one replication loop -- once with
   the reduced-form (Psi_h) delta-method term omitted and once with it
   propagated -- so the cost of the omission, and the width the honest set
   needs, are measured on the same draws rather than asserted. The measured
   numbers are written into the fixture's `_meta` so they can be audited
   without rerunning anything.

   Every by-horizon figure is reported twice: once over all n cells and once
   EXCLUDING the (norm_var, h=0) cell, which is the single point {unit} and
   covers with probability exactly 1 by construction. Averaging it in inflates
   the h=0 column and hides where a collapse actually starts.

THE ESTIMAND
------------
lambda_{i,h} = unit * (e_i' Psi_h gamma) / (e_k' gamma), the unit-effect
normalized structural impulse response of variable i at horizon h, with
gamma = E[m_t u_t] the residual-instrument covariance and k = norm_var. This
is exactly `proxy_svar`'s irf[h][i]; the object here is a confidence set for
it that stays valid when gamma_k is close to zero.

THE ALGEBRA (transcribed here from the moment condition, not from Rust)
----------------------------------------------------------------------
Overlap O = {t : m_t finite}, T_O = |O|, everything demeaned over O.
  g_t      = m~_t * u~_t                    (n-vector), gamma = mean_t g_t
  Omega    = (1/T_O) sum_t (g_t - gamma)(g_t - gamma)'      [HC0]
             + Bartlett lag terms over CALENDAR-time pairs  [HAC]
  q1 = unit * (Psi_h gamma)_i     q0 = gamma_k
  v0 = unit^2 (Psi_h Omega Psi_h')[i,i]     v1 = unit (Psi_h Omega)[i,k]
  v2 = Omega[k,k]
  AR(lam) = T_O (q1 - lam q0)^2 / (v0 - 2 lam v1 + lam^2 v2)

REDUCED-FORM (Psi_h) ESTIMATION ERROR
-------------------------------------
Psi_h is estimated, so the numerator moment carries a second source of error:
  w_{h,i} = e_i' (Psi_hat_h - Psi_h) gamma = Gamma_h (alpha_hat - alpha),
  Gamma_h = (gamma' kron I_n) G_h,
  G_h     = sum_{m<h} J (A')^{h-1-m} kron Psi_m   (Luetkepohl 1990),
with J = [I_n 0] and A the companion matrix. Since w carries no lam, the
statistic stays quadratic and only two constants move:
  v0 += unit^2 (T_O Var(w_{h,i}) + 2 sum_j Psi_h[i,j] T_O Cov(w_{h,i}, gamma_j))
  v1 += unit * T_O Cov(w_{h,i}, gamma_k)
The cross-covariance is zero in the lag block for OLS with an intercept
((Z'Z)^-1 Z'1 is the unit vector on the intercept), and measures 2-5% of its
Cauchy-Schwarz bound in simulation; it is nevertheless carried through the
fixture so the Rust path for it is exercised against a reference.
The set {lam : AR(lam) <= c} is {lam : A lam^2 + B lam + C <= 0} with
  A = T_O q0^2 - c v2   B = 2(c v1 - T_O q1 q0)   C = T_O q1^2 - c v0.
A carries neither i nor h, so `ar_bound_stat = T_O q0^2 / v2 > c` decides
boundedness for the whole grid at once.

Run with the project venv:
    .venv/bin/python fixtures/generate_proxy_ar_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.tsa.api import VAR

OUT = "fixtures/proxy_ar.json"

def nan_to_null(arr):
    """A 1-D list with non-finite entries as None (JSON null).

    json.dump emits a bare `NaN` for float nan, which is not valid JSON and
    which serde_json rejects; the proxy's unavailability mask travels as null.
    """
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr)]


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

H_TRUE = np.array(
    [
        [1.0, 0.4, 0.2],
        [0.5, 1.2, 0.3],
        [0.3, 0.5, 0.9],
    ]
)
A1 = np.array(
    [
        [0.50, 0.10, 0.00],
        [0.00, 0.40, 0.10],
        [0.10, 0.00, 0.30],
    ]
)
A2 = np.array(
    [
        [0.10, 0.00, 0.00],
        [0.00, 0.10, 0.00],
        [0.00, 0.00, 0.10],
    ]
)


def ma_from_coefs(coefs, horizon):
    """Psi_0 = I, Psi_h = sum_{i=1..min(h,p)} Psi_{h-i} A_i."""
    p, n, _ = coefs.shape
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc += psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return np.array(psi)


def simulate(seed, n_obs, phi, sig_nu):
    """VAR(2) with a known impact matrix H and a proxy for structural shock 0.

    The proxy is relevant only for eps_0 (`phi`) and carries independent
    measurement noise (`sig_nu`); shrinking phi weakens the instrument without
    changing the estimand, since the common factor cancels in the ratio.
    """
    rng = np.random.default_rng(seed)
    n = 3
    burn = 500
    total = n_obs + burn
    eps = rng.standard_normal((total, n))
    u = eps @ H_TRUE.T
    y = np.zeros((total, n))
    for t in range(2, total):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + u[t]
    y = y[burn:]
    eps = eps[burn:]
    u = u[burn:]
    m = phi * eps[:, 0] + sig_nu * rng.standard_normal(n_obs)
    return y, u, m


def true_lambda(psi_true, norm_var, unit, horizon):
    """The population estimand.

    gamma = E[m u'] = phi * H[:, 0], so the relevance factor phi cancels in
    lambda and the truth is IDENTICAL in the strong and weak arms -- which is
    what makes a coverage comparison across arms meaningful.
    """
    hcol = H_TRUE[:, 0]
    return np.array(
        [unit * (psi_true[h] @ hcol) / hcol[norm_var] for h in range(horizon + 1)]
    )


# ---------------------------------------------------------------------------
# Reference AR machinery (pure NumPy, written from the algebra above)
# ---------------------------------------------------------------------------


def moment_pieces(uu, m, hac_lags=0):
    """Overlap moments and the moment covariance Omega."""
    t = uu.shape[0]
    assert m.shape[0] == t
    ok = np.isfinite(m)
    idx = np.where(ok)[0]
    t_o = idx.size
    mo = m[idx]
    uo = uu[idx]
    md = mo - mo.mean()
    ud = uo - uo.mean(axis=0)
    g = md[:, None] * ud  # (T_O, n)
    gamma = g.mean(axis=0)
    gt = g - gamma
    omega = gt.T @ gt / t_o
    if hac_lags > 0:
        # Calendar-time pairing: t and t-j must BOTH carry a finite proxy.
        pos = -np.ones(t, dtype=int)
        pos[idx] = np.arange(t_o)
        for j in range(1, hac_lags + 1):
            w = 1.0 - j / (hac_lags + 1.0)
            lead, lag = [], []
            for p, r in enumerate(idx):
                if r >= j and pos[r - j] >= 0:
                    lead.append(p)
                    lag.append(pos[r - j])
            if not lead:
                continue
            gj = gt[lead].T @ gt[lag] / t_o
            omega = omega + w * (gj + gj.T)
    return idx, t_o, gamma, omega


def first_stage_f_hc1(uu, m, norm_var):
    """The HC1-robust first-stage F that proxy_svar reports."""
    ok = np.isfinite(m)
    mo = m[ok]
    yo = uu[ok, norm_var]
    md = mo - mo.mean()
    yd = yo - yo.mean()
    smm = float(md @ md)
    beta = float(md @ yd) / smm
    e = yd - beta * md
    n_o = mo.size
    var_hc1 = (n_o / (n_o - 2.0)) * float(np.sum(md * md * e * e)) / (smm * smm)
    return beta * beta / var_hc1


def companion(coefs):
    """Companion matrix of a VAR(p): first block row is [A_1 .. A_p]."""
    p, n, _ = coefs.shape
    c = np.zeros((n * p, n * p))
    for i in range(p):
        c[:n, i * n : (i + 1) * n] = coefs[i]
    if p > 1:
        c[n:, : n * (p - 1)] = np.eye(n * (p - 1))
    return c


def psi_reduced_form_cov(psi, coefs, cov_alpha, gamma, t_o):
    """T_O * Cov(Psi_hat_h gamma) per horizon, by the delta method.

    `cov_alpha` is Cov(vec alpha_hat) for the LAG BLOCK ONLY, indexed
    r = a*n + e over the n*p stacked lag regressors and the n equations --
    i.e. kron((Z'Z)^-1 lag block, Sigma_u). psi_var[0] is zero: Psi_0 = I.
    """
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
        # G_h (n^2 x n*kp), then Gamma_h row i = sum_j gamma_j G_h[j*n+i, :].
        gh = np.zeros((n * n, n * kp))
        for m in range(h):
            gh += np.kron(atpow[h - 1 - m], psi[m])
        gm = np.zeros((n, n * kp))
        for i in range(n):
            for j in range(n):
                gm[i] += gamma[j] * gh[j * n + i]
        out.append(t_o * (gm @ cov_alpha @ gm.T))
    return out


def cell_coefs(psi_h, omega, gamma, i, norm_var, unit, t_o, crit, rf=None, h=0):
    """(q1, q0, v0, v1, v2, A, B, C) for one (horizon, variable) cell.

    `rf` is None (moment covariance only) or (psi_var, psi_gamma_cov) with
    psi_gamma_cov possibly None.
    """
    p_h = psi_h @ omega
    q1 = unit * float(psi_h[i] @ gamma)
    q0 = float(gamma[norm_var])
    v0 = unit * unit * float(p_h[i] @ psi_h[i])
    v1 = unit * float(p_h[i, norm_var])
    v2 = float(omega[norm_var, norm_var])
    if rf is not None:
        psi_var, cross = rf
        s2 = float(psi_var[h][i, i])
        if cross is None:
            cnum, cden = 0.0, 0.0
        else:
            cnum = float(psi_h[i] @ cross[h][i])
            cden = float(cross[h][i, norm_var])
        v0 += unit * unit * (s2 + 2.0 * cnum)
        v1 += unit * cden
    a = t_o * q0 * q0 - crit * v2
    b = 2.0 * (crit * v1 - t_o * q1 * q0)
    c = t_o * q1 * q1 - crit * v0
    return q1, q0, v0, v1, v2, a, b, c


def _classify(a, b, c, point, tau_a, tau_b, tau_c, tau_d):
    """The closed-form inversion of A lam^2 + B lam + C <= 0.

    Returns (kind, lo, hi) with lo/hi the payload of that shape (None where
    the shape carries no endpoint); for "exterior" they bound the REJECTED
    open middle, not the set.
    """
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
        raise AssertionError("negative discriminant with A > 0: Omega is not PSD")
    if d > tau_d:
        lo, hi = stable_roots(a, b, c, d)
        return ("exterior", lo, hi)
    return ("whole", None, None)


def stable_roots(a, b, c, d):
    """Cancellation-free quadratic roots, sorted ascending (Vieta pair)."""
    sq = np.sqrt(max(d, 0.0))
    s = -b - (-1.0 if b < 0 else 1.0) * sq
    r1 = s / (2.0 * a)
    r2 = 2.0 * c / s
    return (r1, r2) if r1 <= r2 else (r2, r1)


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
    if kind == "whole":
        return True
    return False


def ar_sets(uu, m, psi, norm_var, unit, crit, hac_lags=0, rf=None):
    """Every cell's AR confidence set, plus the shared diagnostics.

    `rf` is None or (psi_var, psi_gamma_cov): the reduced-form correction.
    """
    idx, t_o, gamma, omega = moment_pieces(uu, m, hac_lags)
    n = uu.shape[1]
    horizon = psi.shape[0] - 1
    q0 = float(gamma[norm_var])
    v2 = float(omega[norm_var, norm_var])
    assert q0 != 0.0 and v2 > 0.0
    b_impact = unit * (gamma / q0)
    a_shared = t_o * q0 * q0 - crit * v2
    tau_a = 1e-12 * max(t_o * q0 * q0, crit * v2)
    cells = []
    for h in range(horizon + 1):
        row = []
        for i in range(n):
            q1, q0_, v0, v1, v2_, a, b, c = cell_coefs(
                psi[h], omega, gamma, i, norm_var, unit, t_o, crit, rf, h
            )
            assert a == a_shared, "A must not depend on the cell"
            # PSD sanity of the projected 2x2 block.
            scale = max(abs(v0), abs(v1), abs(v2_))
            assert v1 * v1 <= v0 * v2_ + 1e-9 * scale * scale
            point = float(psi[h][i] @ b_impact)
            tau_b = 1e-12 * max(2.0 * abs(crit * v1), 2.0 * abs(t_o * q1 * q0_))
            tau_c = 1e-12 * max(t_o * q1 * q1, crit * v0)
            tau_d = 1e-12 * max(b * b, abs(4.0 * a * c))
            kind, lo, hi = _classify(a, b, c, point, tau_a, tau_b, tau_c, tau_d)
            row.append(
                {
                    "kind": kind,
                    "lo": None if lo is None else float(lo),
                    "hi": None if hi is None else float(hi),
                    "point": point,
                    "a": a,
                    "b": b,
                    "c": c,
                    "q1": q1,
                    "q0": q0_,
                    "v0": v0,
                    "v1": v1,
                    "v2": v2_,
                    # Read off the SET, not off c: at the knife edge the set is
                    # the whole line for c <= tau_c, and `c > 0.0` would then
                    # claim a set containing every real number excludes zero.
                    "excludes_zero": not contains(kind, lo, hi, 0.0),
                }
            )
        cells.append(row)
    return {
        "n_proxy": int(t_o),
        "cov_um": gamma,
        "impact": b_impact,
        "omega": omega,
        "critical_value": float(crit),
        "ar_bound_stat": float(t_o * q0 * q0 / v2),
        "ar_bounded_all": bool(a_shared > tau_a),
        "first_stage_f": first_stage_f_hc1(uu, m, norm_var),
        "reduced_form_uncertainty": rf is not None,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# LEG 2: brute-force grid proof of the closed-form inversion
# ---------------------------------------------------------------------------


def ar_stat(cell, lam, t_o):
    g = cell["q1"] - lam * cell["q0"]
    v = cell["v0"] - 2.0 * lam * cell["v1"] + lam * lam * cell["v2"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return t_o * g * g / v


def bisect_crossing(cell, t_o, crit, x_in, x_out):
    """Refine the AR(lam) = c crossing bracketed by (accepted, rejected)."""
    for _ in range(200):
        mid = 0.5 * (x_in + x_out)
        if mid in (x_in, x_out):
            break
        if ar_stat(cell, mid, t_o) <= crit:
            x_in = mid
        else:
            x_out = mid
    return 0.5 * (x_in + x_out)


def grid_proof(res, t_o, crit, span, npts=40001):
    """Re-derive every cell's set by direct evaluation and compare.

    Two comparisons, per cell:

    * MEMBERSHIP. Evaluate AR(lam) directly at every grid point and compare
      `AR <= c` to what the closed-form set says. A disagreement is tolerated
      only where the grid point sits on a boundary, judged by the two sides of
      the inequality: |T_O g(lam)^2 - c V(lam)| small RELATIVE TO
      T_O g(lam)^2 + c V(lam). That relative scale matters -- at the
      boundedness knife edge the whole quadratic is float noise, and an
      absolute scale would either pass everything or fail everything.
    * ENDPOINTS. Bisect AR(lam) - c at every accept/reject transition down to
      machine precision and match the crossing against the closed-form root.
    """
    report = {"cells_checked": 0, "membership_points": 0, "max_endpoint_relerr": 0.0}
    for row in res["cells"]:
        for cell in row:
            report["cells_checked"] += 1
            kind, lo, hi = cell["kind"], cell["lo"], cell["hi"]
            # Centre the grid on the action: the point estimate and any roots.
            anchors = [cell["point"]]
            for x in (lo, hi):
                if x is not None and np.isfinite(x):
                    anchors.append(x)
            centre = float(np.mean(anchors))
            half = max(span, 4.0 * (max(anchors) - min(anchors) + 1e-12))
            lams = np.linspace(centre - half, centre + half, npts)
            # The (norm_var, h=0) cell has AR = 0/0 exactly at lam = unit; a
            # grid-based check must not evaluate the ratio there.
            vv = cell["v0"] - 2.0 * lams * cell["v1"] + lams**2 * cell["v2"]
            keep = np.abs(vv) > 1e-14 * max(abs(cell["v0"]), abs(cell["v2"]), 1.0)
            lams = lams[keep]
            vv = vv[keep]
            gg = cell["q1"] - lams * cell["q0"]
            acc_grid = t_o * gg * gg <= crit * vv
            acc_form = np.array([contains(kind, lo, hi, x) for x in lams])
            report["membership_points"] += lams.size

            # A grid point counts as "on the boundary" when the two sides of
            # AR(lam) <= c agree to within 1e-10 of their own magnitude.
            resid = t_o * gg * gg - crit * vv
            magn = t_o * gg * gg + crit * np.abs(vv)
            near = np.abs(resid) <= 1e-10 * np.maximum(magn, 1e-300)
            bad = (acc_grid != acc_form) & ~near
            assert not bad.any(), (
                f"grid and closed form disagree away from a boundary: "
                f"kind={kind} n_bad={int(bad.sum())} of {lams.size}"
            )

            if kind == "whole":
                assert np.all(acc_grid | near), "kind=whole but the grid rejects"
                continue
            if kind == "point":
                stray = acc_grid & ~near
                if stray.any():
                    assert np.abs(lams[stray] - cell["point"]).max() < 1e-6 * max(
                        abs(cell["point"]), 1.0
                    ), "kind=point but the grid accepts away from the point"
                continue

            # Endpoint refinement: bisect every accept/reject transition and
            # compare the crossing to the closed-form root.
            roots_form = [x for x in (lo, hi) if x is not None and np.isfinite(x)]
            if not roots_form:
                continue
            trans = np.where(acc_grid[:-1] != acc_grid[1:])[0]
            for j in trans:
                if acc_grid[j]:
                    x = bisect_crossing(cell, t_o, crit, lams[j], lams[j + 1])
                else:
                    x = bisect_crossing(cell, t_o, crit, lams[j + 1], lams[j])
                err = min(abs(x - r) / max(abs(r), 1.0) for r in roots_form)
                report["max_endpoint_relerr"] = max(report["max_endpoint_relerr"], err)
                assert err < 1e-7, (
                    f"grid crossing {x} does not match a closed-form root "
                    f"{roots_form} (rel err {err}, kind={kind})"
                )
    return report


# ---------------------------------------------------------------------------
# LEG 3: Monte-Carlo coverage
# ---------------------------------------------------------------------------


def fit_var_ols(y, lags):
    """Plain OLS VAR with a constant.

    Returns (resid, coefs, cov_alpha) where cov_alpha is the LAG-BLOCK
    coefficient covariance kron((Z'Z)^-1[1:, 1:], Sigma_u) in the r = a*n + e
    layout the delta-method Jacobian expects.
    """
    t_all, n = y.shape
    t = t_all - lags
    x = np.ones((t, 1 + n * lags))
    for i in range(1, lags + 1):
        x[:, 1 + (i - 1) * n : 1 + i * n] = y[lags - i : t_all - i]
    yy = y[lags:]
    zz_inv = np.linalg.inv(x.T @ x)
    beta = zz_inv @ x.T @ yy  # (1+n*p, n)
    resid = yy - x @ beta
    coefs = np.array(
        [beta[1 + (i - 1) * n : 1 + i * n].T for i in range(1, lags + 1)]
    )  # (p, n, n)
    sigma_u = resid.T @ resid / (t - (1 + n * lags))
    cov_alpha = np.kron(zz_inv[1:, 1:], sigma_u)
    return resid, coefs, cov_alpha


def _summarize(cov, cov_wald, bounded, wald_bounded, total, used, norm_var, widths):
    """Coverage summaries, with and without the degenerate (norm_var, h=0) cell.

    That cell is the single point {unit} and covers with probability exactly 1
    by construction, so it inflates every h=0 average; a three-variable system
    reads 0.968 where the informative two-cell average is 0.951.
    """
    horizon = cov.shape[0] - 1
    by_h = cov.mean(axis=1) / used
    keep = [i for i in range(cov.shape[1]) if i != norm_var]
    by_h_excl = np.concatenate([[cov[0, keep].mean() / used], by_h[1:]])
    n_excl = cov[0, keep].sum() + cov[1:].sum()
    tot_excl = used * (len(keep) + horizon * cov.shape[1])
    return {
        "reps_used": int(used),
        "ar_coverage_mean": float(cov.mean() / used),
        "ar_coverage_mean_excl_degenerate": float(n_excl / tot_excl),
        "ar_coverage_h0": float(by_h[0]),
        "ar_coverage_h0_excl_degenerate": float(by_h_excl[0]),
        "ar_coverage_hmax": float(by_h[-1]),
        # The horizon profile is the diagnostic: a FLAT profile means the
        # variance is capturing what moves; a monotone decay is the signature
        # of omitted reduced-form estimation error.
        "ar_coverage_by_horizon": [float(x) for x in by_h],
        "ar_coverage_by_horizon_excl_degenerate": [float(x) for x in by_h_excl],
        "wald_coverage_mean": float(cov_wald.mean() / used),
        "wald_bounded_fraction": float(wald_bounded / max(total, 1)),
        "bounded_fraction": float(bounded / max(total, 1)),
        "median_width_hmax": (
            float(np.median(widths)) if len(widths) else None
        ),
    }


def mc_coverage(
    seed, reps, n_obs, phi, sig_nu, level, horizon, norm_var, unit, fit_var
):
    """Coverage of the AR set, and of a delta-method Wald interval for scale.

    `fit_var=False` feeds the TRUE innovations and the TRUE MA matrices, which
    is exactly the setting the moment-only variance assumes -- so a shortfall
    there is an error in the set, not an omitted term.

    `fit_var=True` estimates both, and then runs the SAME replication twice:
    once with the reduced-form (Psi_h) delta-method term omitted and once with
    it propagated. Paired on the same draws, the difference between the two
    IS the cost of the omission, and `width_ratio_hmax_median` is the factor by
    which the honest set is wider at the longest horizon.
    """
    crit = float(stats.chi2.ppf(level, 1))
    zq = float(stats.norm.ppf(0.5 + 0.5 * level))
    psi_true = ma_from_coefs(np.array([A1, A2]), horizon)
    truth = true_lambda(psi_true, norm_var, unit, horizon)  # (H+1, n)
    n = 3
    arms = ["omitted", "propagated"] if fit_var else ["omitted"]
    acc = {
        a: {
            "cov": np.zeros((horizon + 1, n)),
            "cov_wald": np.zeros((horizon + 1, n)),
            "bounded": 0,
            "wald_bounded": 0,
            "total": 0,
            "widths": [],
        }
        for a in arms
    }
    width_ratio = []
    used = 0
    for r in range(reps):
        y, u, m = simulate(seed + r, n_obs, phi, sig_nu)
        if fit_var:
            uu, coefs, cov_alpha = fit_var_ols(y, 2)
            psi = ma_from_coefs(coefs, horizon)
            mm = m[2:]
            _, t_o, gamma, _ = moment_pieces(uu, mm)
            psi_var = psi_reduced_form_cov(psi, coefs, cov_alpha, gamma, t_o)
            rfs = {"omitted": None, "propagated": (psi_var, None)}
        else:
            uu, psi, mm = u, psi_true, m
            rfs = {"omitted": None}
        try:
            res = {a: ar_sets(uu, mm, psi, norm_var, unit, crit, 0, rfs[a]) for a in arms}
        except AssertionError:
            continue
        used += 1
        for a in arms:
            d = acc[a]
            t_o = res[a]["n_proxy"]
            for h in range(horizon + 1):
                for i in range(n):
                    cell = res[a]["cells"][h][i]
                    if contains(cell["kind"], cell["lo"], cell["hi"], truth[h, i]):
                        d["cov"][h, i] += 1
                    d["total"] += 1
                    if cell["kind"] in ("interval", "point"):
                        d["bounded"] += 1
                    # Delta-method Wald interval for the same ratio: the
                    # variance frozen at the point estimate. This is the object
                    # the AR set replaces, kept here so the weak arm shows the
                    # difference. It is bounded whenever that variance is
                    # finite, which is measured rather than assumed.
                    lam = cell["point"]
                    v = cell["v0"] - 2.0 * lam * cell["v1"] + lam * lam * cell["v2"]
                    se = np.sqrt(max(v, 0.0) / t_o) / abs(cell["q0"])
                    if np.isfinite(se):
                        d["wald_bounded"] += 1
                    if abs(truth[h, i] - lam) <= zq * se:
                        d["cov_wald"][h, i] += 1
            for i in range(n):
                cell = res[a]["cells"][horizon][i]
                if cell["kind"] == "interval":
                    d["widths"].append(cell["hi"] - cell["lo"])
        if fit_var:
            for i in range(n):
                lo = res["omitted"]["cells"][horizon][i]
                hi = res["propagated"]["cells"][horizon][i]
                if lo["kind"] == "interval" and hi["kind"] == "interval":
                    width_ratio.append((hi["hi"] - hi["lo"]) / (lo["hi"] - lo["lo"]))
    if used == 0:
        raise AssertionError("every replication failed")
    out = {
        a: _summarize(
            acc[a]["cov"], acc[a]["cov_wald"], acc[a]["bounded"],
            acc[a]["wald_bounded"], acc[a]["total"], used, norm_var,
            acc[a]["widths"],
        )
        for a in arms
    }
    out["reps"] = reps
    out["reps_used"] = int(used)
    if width_ratio:
        out["width_ratio_hmax_median"] = float(np.median(width_ratio))
    return out


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def build_case(
    name, seed, n_obs, phi, sig_nu, lags, horizon, norm_var, unit, nan_prefix,
    variance, critical_kind, level, grid_span, reduced_form=None,
):
    y, _u, m_raw = simulate(seed, n_obs, phi, sig_nu)
    res_var = VAR(y).fit(lags, trend="c")
    uu = np.asarray(res_var.resid)
    psi = np.asarray(res_var.ma_rep(horizon))
    m_full = m_raw.copy()
    m_full[:lags] = np.nan
    m_full[lags : lags + nan_prefix] = np.nan
    m_aligned = m_full[lags:]

    hac_lags = variance.get("hac_lags", 0)
    if critical_kind == "chi2":
        crit = float(stats.chi2.ppf(level, 1))
        crit_spec = {"kind": "chi2", "level": level}
    elif critical_kind == "f":
        t_o = int(np.sum(np.isfinite(m_aligned)))
        crit = float(stats.f.ppf(level, 1, t_o - 2))
        crit_spec = {"kind": "f", "level": level}
    elif critical_kind == "knife_edge":
        # Place the test EXACTLY at the boundedness knife edge by supplying
        # c = ar_bound_stat, so A = 0 up to rounding and every cell degenerates
        # to a half-line. Measure-zero in the data, reachable only this way.
        _, t_o, gamma, omega = moment_pieces(uu, m_aligned, hac_lags)
        crit = float(t_o * gamma[norm_var] ** 2 / omega[norm_var, norm_var])
        crit_spec = {"kind": "value", "value": crit}
    else:
        raise ValueError(critical_kind)

    # The reduced-form (Psi_h) delta-method correction, when the case asks for
    # it. The reduced form here is statsmodels' fit, so (Z'Z)^-1 is rebuilt
    # from the same design matrix statsmodels uses (constant, then lag 1..p).
    rf = None
    rf_json = None
    if reduced_form is not None:
        t_all = y.shape[0]
        nvar = y.shape[1]
        x = np.ones((t_all - lags, 1 + nvar * lags))
        for i in range(1, lags + 1):
            x[:, 1 + (i - 1) * nvar : 1 + i * nvar] = y[lags - i : t_all - i]
        cov_alpha = np.kron(
            np.linalg.inv(x.T @ x)[1:, 1:], np.asarray(res_var.sigma_u)
        )
        _, t_o_rf, gamma_rf, omega_rf = moment_pieces(uu, m_aligned, hac_lags)
        psi_var = psi_reduced_form_cov(
            psi, np.asarray(res_var.coefs), cov_alpha, gamma_rf, t_o_rf
        )
        cross = None
        rho = reduced_form.get("cross_rho", 0.0)
        if rho:
            # A synthetic but Cauchy-Schwarz-admissible cross-covariance, so
            # the psi_gamma_cov code path is exercised against a reference
            # rather than being dead in both implementations. The real object
            # is zero in the lag block for OLS with an intercept (see the
            # module header); rho scales it to a plausible magnitude.
            cross = [
                np.array(
                    [
                        [
                            rho
                            * np.sqrt(max(pv[i, i], 0.0) * omega_rf[j, j])
                            for j in range(nvar)
                        ]
                        for i in range(nvar)
                    ]
                )
                for pv in psi_var
            ]
        rf = (psi_var, cross)
        rf_json = {
            "psi_var": [p.tolist() for p in psi_var],
            "psi_gamma_cov": None if cross is None else [c.tolist() for c in cross],
        }

    res = ar_sets(uu, m_aligned, psi, norm_var, unit, crit, hac_lags, rf)
    grid = grid_proof(res, res["n_proxy"], crit, grid_span)

    kinds = sorted({c["kind"] for row in res["cells"] for c in row})

    return {
        "name": name,
        "lags": lags,
        "horizon": horizon,
        "norm_var": norm_var,
        "unit": unit,
        "variance": {"kind": "hac" if hac_lags else "hc0", "lags": hac_lags},
        "critical": crit_spec,
        "resid": uu.tolist(),
        "psi": psi.tolist(),
        "proxy_aligned": nan_to_null(m_aligned),
        "reduced_form": rf_json,
        "grid_proof": {
            "cells_checked": grid["cells_checked"],
            "membership_points": grid["membership_points"],
            "max_endpoint_relerr": grid["max_endpoint_relerr"],
        },
        "kinds_present": kinds,
        "expected": {
            "n_proxy": res["n_proxy"],
            "cov_um": res["cov_um"].tolist(),
            "impact": res["impact"].tolist(),
            "omega": res["omega"].tolist(),
            "critical_value": res["critical_value"],
            "ar_bound_stat": res["ar_bound_stat"],
            "ar_bounded_all": res["ar_bounded_all"],
            "first_stage_f": res["first_stage_f"],
            "reduced_form_uncertainty": res["reduced_form_uncertainty"],
            # `level` is only claimed when the sets earn it: the moment-only
            # variance conditions on the reduced form, so it reports None.
            "level": (
                level
                if (res["reduced_form_uncertainty"] and critical_kind != "knife_edge")
                else None
            ),
            "cells": res["cells"],
        },
    }


def main():
    seed = 20260805
    horizon = 8
    cases = [
        # Strong instrument: A > 0, so every cell is a bounded interval except
        # the (norm_var, h=0) cell, which is the single point {unit}.
        build_case(
            "strong_hc0", seed, 400, 1.0, 1.5, 2, horizon, 0, 1.0, 40,
            {"hac_lags": 0}, "chi2", 0.95, 5.0,
        ),
        # Weak instrument: A < 0, so no cell is bounded -- exterior unions and
        # whole lines. This is the arm the method exists for.
        build_case(
            "weak_hc0", 20260811, 250, 0.06, 1.5, 2, horizon, 0, 1.0, 0,
            {"hac_lags": 0}, "chi2", 0.95, 40.0,
        ),
        # Bartlett HAC variance, a different normalizing variable, a negative
        # `unit`, and the finite-sample F critical value.
        build_case(
            "strong_hac_altnorm", seed + 23, 400, 1.0, 1.0, 2, horizon, 1, -2.0, 25,
            {"hac_lags": 4}, "f", 0.90, 8.0,
        ),
        # The knife edge: c set exactly to ar_bound_stat, so the quadratic
        # degenerates and every cell is a half-line (and the (norm_var, h=0)
        # cell must come back as the whole line, NOT empty).
        build_case(
            "knife_edge", seed + 37, 300, 0.35, 1.5, 2, horizon, 0, 1.0, 0,
            {"hac_lags": 0}, "knife_edge", 0.95, 20.0,
        ),
        # The same knife edge with `unit` negated. Flipping `unit` flips the
        # sign of q1 and v1, hence of B, hence of which way every half-line
        # points -- so this case supplies the mirror-image shape and doubles
        # as the `unit`-equivariance check on a degenerate configuration.
        build_case(
            "knife_edge_flipped", seed + 37, 300, 0.35, 1.5, 2, horizon, 0, -1.0, 0,
            {"hac_lags": 0}, "knife_edge", 0.95, 20.0,
        ),
        # The reduced form PROPAGATED: the same strong-instrument data with the
        # Psi_h delta-method term added to v0. This is the configuration a real
        # caller with an estimated VAR must use, and the one whose omission
        # takes measured coverage to 0.119 by h=8.
        build_case(
            "strong_hc0_reduced_form", seed, 400, 1.0, 1.5, 2, horizon, 0, 1.0, 40,
            {"hac_lags": 0}, "chi2", 0.95, 5.0, reduced_form={},
        ),
        # The same, with a nonzero psi_gamma_cov, so the cross-covariance
        # branch is not dead code in either implementation.
        build_case(
            "strong_hc0_reduced_form_cross", seed, 400, 1.0, 1.5, 2, horizon, 0, 1.0, 40,
            {"hac_lags": 0}, "chi2", 0.95, 5.0, reduced_form={"cross_rho": 0.05},
        ),
    ]

    shapes = sorted({k for c in cases for k in c["kinds_present"]})
    for needed in ("interval", "point", "exterior", "whole", "ray_below", "ray_above"):
        assert needed in shapes, f"no case produces shape {needed}: {shapes}"

    mc = {
        "strong_known_psi": mc_coverage(
            991, 2000, 300, 1.0, 1.5, 0.95, 8, 0, 1.0, fit_var=False
        ),
        "weak_known_psi": mc_coverage(
            2991, 2000, 300, 0.06, 1.5, 0.95, 8, 0, 1.0, fit_var=False
        ),
        "strong_estimated_psi": mc_coverage(
            4991, 1000, 300, 1.0, 1.5, 0.95, 8, 0, 1.0, fit_var=True
        ),
        "weak_estimated_psi": mc_coverage(
            6991, 1000, 300, 0.06, 1.5, 0.95, 8, 0, 1.0, fit_var=True
        ),
    }

    fixture = {
        "_meta": {
            "description": "Golden fixtures for weak-instrument-robust "
            "(Anderson-Rubin / Fieller) confidence sets for proxy-SVAR impulse "
            "responses: one set per (horizon, variable) cell, obtained by "
            "inverting AR(lam) <= c in closed form.",
            "references": {
                "reduced_form": 'statsmodels VAR(y).fit(lags, trend="c"): resid and '
                "ma_rep(horizon) (Psi_0 = I)",
                "moment": "gamma = mean_O (m-mbar)(u-ubar); "
                "Omega = HC0 (or Bartlett HAC over calendar-time pairs) of "
                "g_t = m~_t u~_t",
                "statistic": "AR(lam) = T_O (q1 - lam q0)^2 / (v0 - 2 lam v1 + lam^2 v2) "
                "with the null imposed in the variance",
                "inversion": "A lam^2 + B lam + C <= 0, A = T_O q0^2 - c v2, "
                "B = 2(c v1 - T_O q1 q0), C = T_O q1^2 - c v0",
                "critical_value": "scipy.stats.chi2.ppf(level, 1) or "
                "scipy.stats.f.ppf(level, 1, T_O - 2)",
            },
            "validation": {
                "grid_proof": "for every cell, AR(lam) is evaluated directly on a "
                "40001-point grid and the accepted points are compared to the "
                "closed-form set; every accept/reject transition is bisected to "
                "machine precision and matched against the closed-form root",
                "monte_carlo": mc,
                "monte_carlo_note": "the known-psi arms are the setting the "
                "moment-only variance assumes. The estimated-psi arms are run "
                "twice on the SAME draws: `omitted` leaves the Psi_h "
                "delta-method term out (the old behaviour) and `propagated` "
                "includes it. `width_ratio_hmax_median` is how much wider the "
                "honest set is at the longest horizon. Every by-horizon list "
                "is given twice; the `_excl_degenerate` form drops the "
                "(norm_var, h=0) cell, which is the point {unit} and covers "
                "with probability exactly 1 by construction",
            },
            "tolerance": {"rtol": 1e-9, "atol": 1e-11},
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "seed": seed,
        },
        "cases": cases,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    for c in cases:
        e = c["expected"]
        print(
            f"  case {c['name']}: T_O={e['n_proxy']} c={e['critical_value']:.4f} "
            f"ar_bound_stat={e['ar_bound_stat']:.4f} bounded_all={e['ar_bounded_all']} "
            f"F={e['first_stage_f']:.2f} shapes={c['kinds_present']} "
            f"grid_pts={c['grid_proof']['membership_points']} "
            f"max_endpoint_relerr={c['grid_proof']['max_endpoint_relerr']:.2e}"
        )
    print("  Monte-Carlo coverage (nominal 0.95; h=0 excludes the degenerate cell):")
    for k, v in mc.items():
        for arm in ("omitted", "propagated"):
            if arm not in v:
                continue
            a = v[arm]
            by_h = " ".join(
                f"{x:.3f}" for x in a["ar_coverage_by_horizon_excl_degenerate"]
            )
            print(
                f"    {k:22s} rf={arm:11s} AR mean={a['ar_coverage_mean_excl_degenerate']:.4f} "
                f"| Wald={a['wald_coverage_mean']:.4f} "
                f"| bounded={a['bounded_fraction']:.3f} "
                f"| wald_bounded={a['wald_bounded_fraction']:.3f} "
                f"(used={a['reps_used']}/{v['reps']})"
            )
            print(f"      by_h {by_h}")
        if "width_ratio_hmax_median" in v:
            print(
                f"      median width ratio at h=hmax "
                f"(propagated/omitted) = {v['width_ratio_hmax_median']:.3f}"
            )


if __name__ == "__main__":
    main()
