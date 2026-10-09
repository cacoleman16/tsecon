"""Golden fixtures for SOFT (Gaussian, stated-variance) conditions in
`var_conditional_forecast(condition_sd=..., soft=...)` (crate `tsecon-var`,
module `conditional.rs`, `VarResults::conditional_forecast_soft`).

Notation as in generate_var_cf_fixtures.py: `Psi_h` the MA coefficients of
the OLS-fitted VAR(p), `Sigma` the df-adjusted `sigma_u`, `yhat` the
unconditional path, `R` the block-lower-triangular MA matrix, `S = I_H kron
Sigma`, and the m conditioned cells (hard and soft together, row-major in
(h, j)) giving `B` (their rows of R), `r = value - yhat`, `W = S B'`,
`Occ = B W`, `G = R W`, `Omega_y = R S R'`. Conditioned cell i carries a
standard deviation sd_i (0 for a hard cell), `Om = diag(sd_i^2)`.

Three reference legs, graded honestly:

  * CLOSED FORM (documented formula, transcribed here in NumPy):

      SIGNAL (Monti 2010, JMCB 42(8): judgment as a noisy observation
      z = B y + e, e ~ N(0, Om) independent of the innovations), M = Occ + Om:
          u* = W M^{-1} r,   y = yhat + R u*,   V = Omega_y - G M^{-1} G',
          mahalanobis = r' M^{-1} r ~ chi2(m);
        hard cells (sd = 0) are pinned bitwise with a zero row and column;
        soft cells are the posterior mean, not pinned, and the soft x soft
        block of a horizon is taken from the exact identity
        Cov(B y | z) = Om M^{-1} Occ (= Om - Om M^{-1} Om), which has no
        cancellation when sd is tiny (asserted equal to V's block first).

      DENSITY (Andersson, Palmqvist and Waggoner 2010, Riksbank WP 247: the
      conditioned cells' forecast distribution IS N(value, Om), the mixture
      over c ~ N(value, Om) of the hard conditional at c), M = Occ:
          u*, y, mahalanobis = the hard ones at the central values,
          V = Omega_y - G Occ^{-1} G' + K Om K',   K = G Occ^{-1};
        then the exact overrides the crate applies (asserted to hold to
        roundoff first): every conditioned cell's point is its value, the
        conditioned x conditioned block of cov[h] is Om exactly (sd^2 on the
        diagonal, 0.0 between distinct conditioned cells, a zero row and
        column at hard cells), and a soft cell's se is sd bitwise.

  * KALMAN (statsmodels 0.15 VARMAX, an independent package), the
    Banbura-Giannone-Lenza route of the hard fixture extended:

      SIGNAL: `VARMAX(endog_with_NaN_future, order=(p, 0), trend,
        error_cov_type="unstructured")`, parameters set BY NAME to the OLS
        fit, `mod.update(params)`, then a TIME-VARYING `mod.ssm["obs_cov"]`
        of shape (k, k, T + H) installed: zero in-sample, diag sd^2 at the
        future conditioned cells. `VARMAX.update` writes obs_cov only when
        measurement_error=True, so the array survives `smooth(params)`; the
        generator asserts it on the RESULTS object
        (`sres.filter_results.obs_cov` equals the installed array exactly).
        Path = smoothed_state[:k, T:T+H], cov = smoothed_state_cov[:k, :k,
        T:T+H], shocks = smoothed_state_disturbance[:k, T-1:T+H-1].
        `llf_obs[T:].sum()` equals -0.5 (m ln 2 pi + ln det M + mahalanobis)
        and is stored as `bgl_llf_future`, a statsmodels check on the
        returned mahalanobis.

      DENSITY: the hard smoother at the central values plus the law of total
        variance with an exact perturbation gain: for each soft cell i the
        smoother is rerun with value_i + 1.0 and the path difference is the
        gain column K_i (the smoother is affine in the observations), so
        cov[h] = smoothed_state_cov[h] + sum_i sd_i^2 K_i[h] K_i[h]'.
        Statsmodels does the conditioning, NumPy assembles the total
        variance: semi-independent, and only valid for sd = O(1) because the
        differenced gain's ~1e-16 roundoff is multiplied by sd^2 (1.2e-4
        absolute at sd = 1e6), so the extreme-sd case is signal-only.

  * APW STRUCTURAL FORM (density only; a third, formula-level
    transcription): with Rb = B (I_H kron P), P = chol_lower(Sigma), the
    structural shocks are eps ~ N(Rb^+ r, Rb^+ Om Rb^+' + I - Rb^+ Rb)
    (Moore-Penrose pseudo-inverse), and y = yhat + R (I kron P) eps:
    `apw_point`, `apw_cov`.

All legs are asserted to agree at 1e-10 before anything is stored. The
series are var_cf.json's stored `series` (by name, not duplicated here):
they are re-simulated with the var_cf generator's own helpers, imported by
module (the generate_var_diag_fixtures.py precedent), and asserted bitwise
equal to the stored arrays.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden tests parse to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_var_cf_soft_fixtures.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_var_cf_fixtures import (  # noqa: E402
    ma_rep,
    macro_system,
    sim_var,
    varmax_params_by_name,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "var_cf_soft.json"
CF_JSON = HERE / "var_cf.json"


# ------------------------------------------------------------------ series

def series():
    """The two var_cf.json series, re-derived and asserted bitwise equal."""
    chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]]
    y = sim_var(
        20260910, 300, c=[0.5, -0.2, 0.1],
        a_list=[[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]],
                [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]],
        chol=chol)
    macro = macro_system()
    stored = json.loads(CF_JSON.read_text(encoding="utf-8"))["series"]
    assert np.array_equal(y, np.array(stored["sim_var2_k3"]))
    assert np.array_equal(macro, np.array(stored["macro_growth_infl_dtbill"]))
    return {"sim_var2_k3": y, "macro_growth_infl_dtbill": macro}


def fit(y, p, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return VAR(y).fit(p, trend=trend)


# --------------------------------------------------------- closed-form leg

def pieces(res, H):
    k = res.neqs
    p = res.k_ar
    coefs = [np.asarray(res.coefs[i]) for i in range(p)]
    sigma = np.asarray(res.sigma_u)
    yhat = np.asarray(res.forecast(res.endog[-p:], H))
    psi = ma_rep(coefs, H - 1)
    assert np.allclose(np.asarray(res.irf(H - 1).irfs), np.asarray(psi), atol=1e-12)
    n = H * k
    R = np.zeros((n, n))
    for h in range(H):
        for s in range(h + 1):
            R[h * k:(h + 1) * k, s * k:(s + 1) * k] = psi[h - s]
    S = np.kron(np.eye(H), sigma)
    return k, sigma, yhat, R, S


def grids(steps, k, cond_rows, sd_rows):
    cond = np.full((steps, k), np.nan)
    sd = np.zeros((steps, k))
    for h, row in enumerate(cond_rows):
        for j, v in enumerate(row):
            if v is not None:
                cond[h, j] = v
    for h, row in enumerate(sd_rows):
        for j, v in enumerate(row):
            if v is not None and np.isfinite(v) and v > 0:
                assert np.isfinite(cond[h, j]), "sd at a free cell"
                sd[h, j] = v
    return cond, sd


def closed_form(res, H, cond, sd, mode, alpha):
    """mode in {"signal", "density"}; cond NaN = free; sd 0 = hard."""
    k, sigma, yhat, R, S = pieces(res, H)
    idx = [h * k + j for h in range(H) for j in range(k) if np.isfinite(cond[h, j])]
    m = len(idx)
    r = np.array([cond[i // k, i % k] - yhat[i // k, i % k] for i in idx])
    sdv = np.array([sd[i // k, i % k] for i in idx])
    om = sdv ** 2
    B = R[idx, :]
    W = S @ B.T
    Occ = B @ W
    G = R @ W
    Omega_y = R @ S @ R.T
    if mode == "signal":
        M = Occ + np.diag(om)
        lam = np.linalg.solve(M, r)
        V = Omega_y - G @ np.linalg.solve(M, G.T)
    elif mode == "density":
        M = Occ
        lam = np.linalg.solve(Occ, r)
        K = np.linalg.solve(Occ, G.T).T          # n x m gain G Occ^{-1}
        V = Omega_y - G @ np.linalg.solve(Occ, G.T) + K @ np.diag(om) @ K.T
    else:
        raise ValueError(mode)
    u = W @ lam
    ycond = (yhat.ravel() + R @ u).reshape(H, k)
    covs = np.array([V[h * k:(h + 1) * k, h * k:(h + 1) * k] for h in range(H)])
    covs = 0.5 * (covs + np.transpose(covs, (0, 2, 1)))
    raw_point, raw_cov = ycond.copy(), covs.copy()
    P = np.linalg.cholesky(sigma)
    eps = np.linalg.solve(np.kron(np.eye(H), P), u).reshape(H, k)
    maha = float(r @ lam)
    sign, ln_det_m = np.linalg.slogdet(M)
    assert sign > 0
    mask = np.isfinite(cond)
    hard = mask & (sd == 0.0)
    soft = mask & (sd > 0.0)
    # Exact overrides, each checked to hold to roundoff first.
    pinned = mask if mode == "density" else hard
    assert np.max(np.abs(ycond[pinned] - cond[pinned]), initial=0.0) < 1e-10
    ycond = np.where(pinned, cond, ycond)
    for h in range(H):
        cj = [j for j in range(k) if mask[h, j]]
        for j in range(k):
            if hard[h, j]:
                assert np.max(np.abs(covs[h, j, :])) < 1e-10
                covs[h, j, :] = 0.0
                covs[h, :, j] = 0.0
        if mode == "signal":
            # The soft x soft block from the cancellation-free identity
            # Cov(B y | z) = Om M^{-1} Occ (= Om - Om M^{-1} Om), which the
            # crate uses too: MSE - G M^{-1} G' leaves a nearly hard soft
            # cell's variance at roundoff (sd = 1e-9 below).
            Q = np.linalg.solve(M, Occ)
            pos = {i: a for a, i in enumerate(idx)}
            for j in cj:
                for l in cj:
                    if soft[h, j] and soft[h, l]:
                        a, b = pos[h * k + j], pos[h * k + l]
                        v = 0.5 * (om[a] * Q[a, b] + om[b] * Q[b, a])
                        assert abs(covs[h, j, l] - v) < 1e-10 * max(1.0, abs(v))
                        covs[h, j, l] = v
                covs[h, j, j] = max(covs[h, j, j], 0.0)
        if mode == "density":
            for j in cj:
                if soft[h, j]:
                    for l in cj:
                        target = sd[h, j] ** 2 if l == j else 0.0
                        assert abs(covs[h, j, l] - target) < 1e-10 * max(1.0, sd[h, j] ** 2)
                        covs[h, j, l] = target
                        covs[h, l, j] = target
    se = np.sqrt(np.maximum(np.einsum("hjj->hj", covs), 0.0))
    if mode == "density":
        assert np.max(np.abs(se[soft] - sd[soft]), initial=0.0) < 1e-10
        se = np.where(soft, sd, se)
    mse = np.asarray(res.mse(H))
    unc_se = np.sqrt(np.einsum("hjj->hj", mse))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    out = {
        "unconditional": yhat.tolist(),
        "point": ycond.tolist(),
        "cov": covs.tolist(),
        "se": se.tolist(),
        "unconditional_se": unc_se.tolist(),
        "lower": (ycond - z * se).tolist(),
        "upper": (ycond + z * se).tolist(),
        "shocks": u.reshape(H, k).tolist(),
        "orth_shocks": eps.tolist(),
        "constrained": mask.tolist(),
        "n_constrained": m,
        "mahalanobis": maha,
        "mahalanobis_pvalue": float(stats.chi2.sf(maha, m)),
        "mode": mode,
        "condition_sd": np.where(soft, sd, 0.0).tolist(),
        "n_soft": int(soft.sum()),
        "steps": H,
        "alpha": alpha,
        "ln_det_m": float(ln_det_m),
    }
    aux = {"raw_point": raw_point, "raw_cov": raw_cov, "B": B, "r": r, "om": om,
           "R": R, "S": S, "yhat": yhat, "P": P, "m": m, "maha": maha,
           "ln_det_m": float(ln_det_m), "mse": mse}
    return out, aux


# ------------------------------------------------------- Kalman (BGL) leg

def varmax_smooth(y, res, H, cond, trend, obs_var=None):
    """statsmodels VARMAX smoother over the future with NaN free cells.
    obs_var: H x k measurement-noise variances on the future rows (0 = none),
    installed as a time-varying obs_cov (zero in-sample)."""
    k = res.neqs
    p = res.k_ar
    T = y.shape[0]
    mod = VARMAX(np.vstack([y, cond]), order=(p, 0), trend=trend,
                 error_cov_type="unstructured")
    params = varmax_params_by_name(mod, res, trend)
    if obs_var is not None:
        oc = np.zeros((k, k, T + H))
        for h in range(H):
            for j in range(k):
                oc[j, j, T + h] = obs_var[h, j]
        mod.update(params)
        mod.ssm["obs_cov"] = oc
        sres = mod.smooth(params)
        got = np.asarray(sres.filter_results.obs_cov)
        assert got.shape == (k, k, T + H), got.shape
        assert np.max(np.abs(got - oc)) == 0.0
    else:
        sres = mod.smooth(params)
    sc = np.asarray(mod.ssm["state_cov"])
    sc = sc if sc.ndim == 2 else sc[:, :, 0]
    assert np.max(np.abs(sc[:k, :k] - np.asarray(res.sigma_u))) < 1e-12
    path = np.asarray(sres.smoothed_state[:k, T:T + H].T)
    cov = np.transpose(np.asarray(sres.smoothed_state_cov[:k, :k, T:T + H]), (2, 0, 1))
    shocks = np.asarray(sres.smoothed_state_disturbance[:k, T - 1:T + H - 1].T)
    llf_future = float(np.asarray(sres.llf_obs)[T:].sum())
    return path, cov, shocks, llf_future


def kalman_leg(y, res, H, cond, sd, trend, mode):
    k = res.neqs
    if mode == "signal":
        return varmax_smooth(y, res, H, cond, trend, obs_var=sd ** 2)
    path0, cov0, shocks0, llf0 = varmax_smooth(y, res, H, cond, trend)
    cov = cov0.copy()
    for h in range(H):
        for j in range(k):
            if sd[h, j] > 0:
                c2 = cond.copy()
                c2[h, j] += 1.0
                path1, _, _, _ = varmax_smooth(y, res, H, c2, trend)
                kcol = path1 - path0                  # H x k gain column
                for hh in range(H):
                    cov[hh] += sd[h, j] ** 2 * np.outer(kcol[hh], kcol[hh])
    return path0, cov, shocks0, llf0


# ------------------------------------------------------------- APW leg

def apw_leg(aux, H, k):
    """Andersson-Palmqvist-Waggoner structural form with Moore-Penrose."""
    B, r, om, R, P, yhat = aux["B"], aux["r"], aux["om"], aux["R"], aux["P"], aux["yhat"]
    IP = np.kron(np.eye(H), P)
    Rb = B @ IP
    Rbp = np.linalg.pinv(Rb)
    n = H * k
    mean_eps = Rbp @ r
    cov_eps = Rbp @ np.diag(om) @ Rbp.T + np.eye(n) - Rbp @ Rb
    path = (yhat.ravel() + R @ IP @ mean_eps).reshape(H, k)
    Vy = R @ IP @ cov_eps @ IP.T @ R.T
    cov = np.array([Vy[h * k:(h + 1) * k, h * k:(h + 1) * k] for h in range(H)])
    return path, 0.5 * (cov + np.transpose(cov, (0, 2, 1)))


# ------------------------------------------------------------------ cases

def build_case(name, series_name, y, p, trend, steps, cond_rows, sd_rows, mode, alpha=0.05):
    res = fit(y, p, trend)
    k = res.neqs
    cond, sd = grids(steps, k, cond_rows, sd_rows)
    cf, aux = closed_form(res, steps, cond, sd, mode, alpha)
    path, cov, shocks, llf_future = kalman_leg(y, res, steps, cond, sd, trend, mode)
    dev_path = float(np.max(np.abs(path - np.asarray(cf["point"]))))
    dev_cov = float(np.max(np.abs(cov - np.asarray(cf["cov"]))))
    dev_shocks = float(np.max(np.abs(shocks - np.asarray(cf["shocks"]))))
    assert dev_path < 1e-10 and dev_cov < 1e-10 and dev_shocks < 1e-10, \
        (name, mode, dev_path, dev_cov, dev_shocks)
    # statsmodels' future log-likelihood is log N(r; 0, M): a check on the
    # mahalanobis statistic (and ln det M) of the closed form.
    m = aux["m"]
    ll_closed = -0.5 * (m * math.log(2.0 * math.pi) + aux["ln_det_m"] + aux["maha"])
    assert abs(llf_future - ll_closed) < 1e-10 * max(1.0, abs(ll_closed)), (llf_future, ll_closed)
    case = {
        "name": name,
        "series": series_name,
        "p": p,
        "trend": trend,
        "conditions": cond_rows,
        "condition_sd_input": sd_rows,
        "bgl_point": path.tolist(),
        "bgl_cov": cov.tolist(),
        "bgl_shocks": shocks.tolist(),
        "bgl_max_abs_dev": {"point": dev_path, "cov": dev_cov, "shocks": dev_shocks},
        "bgl_llf_future": llf_future,
    }
    if mode == "density":
        apath, acov = apw_leg(aux, steps, k)
        dev_ap = float(np.max(np.abs(apath - aux["raw_point"])))
        dev_ac = float(np.max(np.abs(acov - aux["raw_cov"])))
        assert dev_ap < 1e-10 and dev_ac < 1e-10, (name, dev_ap, dev_ac)
        case["apw_point"] = apath.tolist()
        case["apw_cov"] = acov.tolist()
        case["apw_max_abs_dev"] = {"point": dev_ap, "cov": dev_ac}
    case.update(cf)
    return case, res, aux


def main():
    ser = series()
    y = ser["sim_var2_k3"]
    macro = ser["macro_growth_infl_dtbill"]
    res2 = fit(y, 2, "c")
    yhat2 = np.asarray(res2.forecast(y[-2:], 6))
    mse2 = np.asarray(res2.mse(6))
    N = None
    S = "sim_var2_k3"
    cases = []

    path_cond = [[N, 0.3, N], [N, 0.2, N], [N, 0.1, N], [N, 0.0, N],
                 [N, N, N], [N, N, N], [N, N, N], [1.0, N, N]]
    path_sd = [[N, 0.1, N], [N, 0.2, N], [N, 0.3, N], [N, 0.4, N],
               [N, N, N], [N, N, N], [N, N, N], [0.0, N, N]]
    mixed_cond = [[0.6, -0.1, 0.2], [N, N, N], [N, 0.05, N]]
    mixed_sd = [[0.0, 0.5, N], [N, N, N], [N, 1.0, N]]
    for mode in ("density", "signal"):
        # Series-1 path held softly for four quarters, a hard cell at h = 8.
        cases.append(build_case("var2c_soft_path", S, y, 2, "c", 8, path_cond, path_sd, mode)[0])
        # Hard and soft cells at the SAME horizon; VAR(3) without constant.
        cases.append(build_case("var3n_mixed_same_h", S, y, 3, "n", 6, mixed_cond, mixed_sd, mode)[0])
    # Every series soft at h = 1, a hard cell later; alpha 0.10.
    cases.append(build_case(
        "var1c_all_soft_h1", S, y, 1, "c", 5,
        [[0.6, -0.1, 0.2], [N, N, N], [N, N, N], [N, 0.05, N]],
        [[0.3, 0.2, 0.25], [N, N, N], [N, N, N], [N, N, N]], "signal", alpha=0.10)[0])
    # ONE cell at its unconditional forecast with sd = its unconditional se:
    # the density forecast IS the unconditional one (cov == MSE). One cell
    # only: with several the identity needs a full Om = Occ.
    sd_id = float(math.sqrt(mse2[1][1, 1]))
    case, res_id, _ = build_case(
        "var2c_density_identity", S, y, 2, "c", 6,
        [[N, N, N], [N, float(yhat2[1, 1]), N]], [[N, N, N], [N, sd_id, N]], "density")
    cov_id = np.asarray(case["cov"])
    assert np.max(np.abs(cov_id - mse2)) < 1e-14, np.max(np.abs(cov_id - mse2))
    assert np.max(np.abs(np.asarray(case["point"]) - yhat2)) < 1e-14
    cases.append(case)
    # Extreme standard deviations (1e-9: essentially hard; 1e6: essentially
    # ignored) beside a hard cell. SIGNAL ONLY: the density perturbation leg
    # loses ~1e-16 * sd^2 absolute.
    cases.append(build_case(
        "var2c_extreme_sd", S, y, 2, "c", 4,
        [[N, 0.3, N], [0.5, N, N], [N, N, -0.2]],
        [[N, 1e-9, N], [1e6, N, N], [N, N, 0.0]], "signal")[0])
    # Two soft cells at the same horizon plus a hard cell: the soft x soft
    # covariance is exactly 0 (computed: ~1e-17).
    cases.append(build_case(
        "var2c_two_soft_same_h", S, y, 2, "c", 4,
        [[N, N, N], [float(yhat2[1, 0] + 0.4), float(yhat2[1, 1] - 0.2), N],
         [N, N, N], [N, N, float(yhat2[3, 2] + 0.1)]],
        [[N, N, N], [0.3, 0.15, N], [N, N, N], [N, N, 0.0]], "density")[0])
    # Macro: inflation held at its last value for four quarters with a
    # widening band (sd 0.5 -> 1.25), the T-bill on hold (hard) for two.
    m_last = macro[-1]
    infl = float(m_last[1])
    macro_cond = [[N, infl, 0.0], [N, infl, 0.0], [N, infl, N], [N, infl, N]]
    macro_sd = [[N, 0.5, N], [N, 0.75, N], [N, 1.0, N], [N, 1.25, N]]
    for mode in ("density", "signal"):
        cases.append(build_case("macro_infl_band", "macro_growth_infl_dtbill", macro, 2, "c", 8,
                                macro_cond, macro_sd, mode)[0])

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "series_source": "var_cf.json (`series`, by name; asserted bitwise equal at generation)",
            "note": (
                "Soft (Gaussian, stated-variance) conditions: closed form in NumPy "
                "(signal: Monti 2010, M = Occ + Om; density: Andersson-Palmqvist-Waggoner "
                "2010, the hard path plus K Om K', with the exact overrides the crate "
                "applies), statsmodels VARMAX (signal: time-varying obs_cov asserted on "
                "the results object; density: the hard smoother plus an exact "
                "perturbation gain; `bgl_*`), and the APW Moore-Penrose structural form "
                "(`apw_*`, density). All legs asserted equal at 1e-10. `ln_det_m` is "
                "ln det M of the closed form (M = Occ + Om in signal mode, Occ "
                "otherwise); -0.5 (m ln 2 pi + ln_det_m + mahalanobis) equals "
                "statsmodels' `bgl_llf_future`."
            ),
        },
        "cases": cases,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} cases")
    for c in cases:
        d = c["bgl_max_abs_dev"]
        extra = ""
        if "apw_max_abs_dev" in c:
            a = c["apw_max_abs_dev"]
            extra = f" apw dev path={a['point']:.1e} cov={a['cov']:.1e}"
        print(f"  {c['name']:24s} {c['mode']:7s} p={c['p']} {c['trend']} H={c['steps']} "
              f"m={c['n_constrained']} soft={c['n_soft']} maha={c['mahalanobis']:.4f} "
              f"bgl dev path={d['point']:.1e} cov={d['cov']:.1e} shocks={d['shocks']:.1e}{extra}")


if __name__ == "__main__":
    main()
