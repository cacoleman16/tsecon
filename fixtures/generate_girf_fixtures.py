"""Golden fixtures for the Koop-Pesaran-Potter generalized impulse-response
engine (`var_girf` on the linear VAR, `threshold_var_girf` on the two-regime
threshold VAR).

Reference, graded honestly per block:

  * LINEAR VAR, orthogonal shock — statsmodels `VARResults.irf(H).orth_irfs`
    (independent package): with common random numbers the paired
    with/without difference of a linear model is `Psi_h P e_j * size` for
    EVERY draw and every history, so the engine must reproduce the Cholesky
    IRF at 1e-12 with a single draw (any n_draws gives the same path).
  * LINEAR VAR, generalized shock — the Pesaran-Shin (1998, Economics
    Letters 58, eq. 10) closed form, transcribed here in NumPy (documented
    formula golden; statsmodels has no GIRF):

        GIRF_h = Phi_h Sigma e_j / sqrt(sigma_jj) * size,
        Phi_0 = I,  Phi_h = sum_{i=1}^{min(h,p)} Phi_{h-i} A_i,

    with Sigma the df-adjusted residual covariance statsmodels calls
    `sigma_u` (the covariance `var_irf(orth=True)` factors). Phi_h is
    computed by the recursion here and asserted equal to statsmodels'
    `irf.irfs` before anything is stored.
  * THRESHOLD VAR — no third-party TVAR GIRF runs in this container (no
    tsDyn: CRAN unreachable through the egress proxy). The reference is an
    independent NumPy transcription of the DOCUMENTED engine (the module
    docs of crates/tsecon-var/src/girf.rs and
    crates/tsecon-regime/src/tvar_girf.rs), reproducing its random streams
    exactly through NumPy's own SeedSequence/Philox/Generator.random (which
    tsecon-rng is bit-compatible with), so the pinned numbers are the
    engine's exact simulation, not just its expectation:

      histories   every lag window t >= L = max(p, d) of the sample (the
                  shock hits period t), in time order; regime of window =
                  0 if y[t-d, tv] <= gamma else 1; regime="low"/"high"
                  keep one regime; `histories=m` keeps a subsample chosen
                  by a partial Fisher-Yates shuffle driven by
                  Generator(Philox(seed + 0xD1B54A32D192ED03 mod 2^64)):
                  j = i + floor(U * (n - i)), swap, first m, sorted.
      streams     SeedSequence(seed).spawn(n_hist)[i].spawn(n_streams)[r],
                  n_streams = n_draws/2 if antithetic else n_draws;
                  Philox(child) -> Generator.random() 53-bit uniforms;
                  normals by Box-Muller z = sqrt(-2 ln(1-U1)) cos(2 pi U2)
                  on consecutive uniforms; (H+1)*k normals per stream,
                  period-major; draw 2r = +z, draw 2r+1 = -z (antithetic).
      shock       s0 = regime of the history; orthogonal: size * L_{s0}[:, j]
                  (L lower Cholesky of the regime ML covariance sigma_low /
                  sigma_high); generalized: size * Sigma_{s0}[:, j] /
                  sqrt(Sigma_{s0}[j, j]).
      paths       both start from the window; at period h each path reads
                  its own regime s_h from its own window, u_h = L_{s_h} z_h
                  (the common draw), the shocked path adds the shock at
                  h = 0; y_h = c_s + sum_lag A_s[:, lag] y_{h-lag} + u_h
                  (coefficient columns [const?, y_{t-1}.., y_{t-p}..]);
                  diff_h = y^with_h - y^without_h.
      summaries   per_history = mean over draws; girf = mean over
                  histories; lower/upper = numpy.percentile(linear) at
                  bands across histories; mc_se = sqrt(sum_i var_i /
                  n_eff) / n_hist with var_i the (n_eff - 1)-divisor
                  variance of the pair means (n_eff = n_streams); draw_sd
                  = sqrt(mean_i var(draws_i)) with the (n_draws - 1)
                  divisor; draw_lower/draw_upper = mean over histories of
                  the per-history across-draw percentiles; girf_low /
                  girf_high = means over the used histories of each regime.

    The TVAR fit itself is the NumPy transcription of
    generate_tvar_fixtures.py (imported from it), already pinned by
    tvar.json. Grade: documented-algorithm transcription, pinned at 1e-10
    (the engine and this file share only the RNG contract and the
    conventions above). Statistical properties (asymmetry, regime
    dependence, 1/sqrt(n) convergence, antithetic variance reduction) are
    carried by the crate's seeded property tests.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden tests parse to identical
bits (serde_json `float_roundtrip`).

Run:  .venv-wt/bin/python fixtures/generate_girf_fixtures.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import statsmodels
from statsmodels.tsa.api import VAR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_tvar_fixtures import fit_tvar, sim_tvar  # noqa: E402

OUT = Path(__file__).resolve().parent / "girf.json"
HISTORY_SEED_OFFSET = 0xD1B54A32D192ED03


# ------------------------------------------------------------------ series

def sim_var(seed, T, c, a_list, chol, burn=100):
    """VAR(p) with intercept c, lag matrices a_list, innovation covariance
    chol @ chol.T."""
    rng = np.random.default_rng(seed)
    k = len(c)
    p = len(a_list)
    n = T + burn + p
    y = np.zeros((n, k))
    c = np.asarray(c)
    a_list = [np.asarray(a) for a in a_list]
    chol = np.asarray(chol)
    for t in range(p, n):
        v = c.copy()
        for i, a in enumerate(a_list, start=1):
            v = v + a @ y[t - i]
        y[t] = v + chol @ rng.standard_normal(k)
    return y[burn + p:]


# --------------------------------------------------------- linear closed forms

def ma_rep(coefs, H):
    p = len(coefs)
    k = coefs[0].shape[0]
    phi = [np.eye(k)]
    for h in range(1, H + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc += phi[h - i] @ coefs[i - 1]
        phi.append(acc)
    return phi


def linear_case(y, p, trend, shock_var, size, H):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(p, trend=trend)
        irf = res.irf(H)
    coefs = [np.asarray(res.coefs[i]) for i in range(p)]
    sigma = np.asarray(res.sigma_u)
    phi = ma_rep(coefs, H)
    # The recursion must agree with statsmodels' own MA representation.
    assert np.allclose(np.asarray(irf.irfs), np.asarray(phi), atol=1e-12)
    L = np.linalg.cholesky(sigma)
    orth = [np.asarray(irf.orth_irfs)[h][:, shock_var] * size for h in range(H + 1)]
    # Cross-check the statsmodels orthogonalized column against Phi_h P e_j.
    assert np.allclose(orth, [phi[h] @ L[:, shock_var] * size for h in range(H + 1)],
                       atol=1e-12)
    gen = [phi[h] @ sigma[:, shock_var] / math.sqrt(sigma[shock_var, shock_var]) * size
           for h in range(H + 1)]
    return {
        "p": p,
        "trend": trend,
        "shock_var": shock_var,
        "size": size,
        "horizon": H,
        "girf_orthogonal": np.asarray(orth).tolist(),
        "girf_generalized": np.asarray(gen).tolist(),
        "shock_vector_orthogonal": (L[:, shock_var] * size).tolist(),
        "shock_vector_generalized": (sigma[:, shock_var]
                                     / math.sqrt(sigma[shock_var, shock_var]) * size).tolist(),
        "n_histories_all": int(y.shape[0] - p),
    }


# ------------------------------------------- the engine transcription (TVAR)

def normals(gen, n):
    out = np.empty(n)
    for i in range(n):
        u1 = gen.random()
        u2 = gen.random()
        out[i] = math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)
    return out


def subsample(n, m, seed):
    if m >= n:
        return list(range(n))
    gen = np.random.Generator(np.random.Philox((seed + HISTORY_SEED_OFFSET) % 2**64))
    idx = list(range(n))
    for i in range(m):
        j = i + min(int(gen.random() * (n - i)), n - i - 1)
        idx[i], idx[j] = idx[j], idx[i]
    return sorted(idx[:m])


def tvar_regime(window, L, d, tv, gamma):
    return 0 if window[L - d, tv] <= gamma else 1


def tvar_step(coefs, constant, window, p, k, u):
    a = coefs
    c0 = 1 if constant else 0
    out = u.copy()
    if constant:
        out = out + a[:, 0]
    L = window.shape[0]
    for lag in range(1, p + 1):
        out = out + a[:, c0 + (lag - 1) * k:c0 + lag * k] @ window[L - lag]
    return out


def tvar_girf_transcribed(y, fit, p, constant, shock, shock_var, size, H,
                          n_draws, seed, antithetic, bands, regime, histories):
    T, k = y.shape
    d = fit["delay"]
    tv = fit["threshold_index"]
    gamma = fit["threshold"]
    L = max(p, d)
    coefs = [np.asarray(fit["coefs_low"]), np.asarray(fit["coefs_high"])]
    sigma = [np.asarray(fit["sigma_low"]), np.asarray(fit["sigma_high"])]
    chol = [np.linalg.cholesky(0.5 * (s + s.T)) for s in sigma]
    if shock == "orthogonal":
        delta = [size * c[:, shock_var] for c in chol]
    else:
        delta = [size * s[:, shock_var] / math.sqrt(s[shock_var, shock_var]) for s in sigma]

    windows = []
    for t in range(L, T):
        w = y[t - L:t]
        r = tvar_regime(w, L, d, tv, gamma)
        if regime == "all" or (regime == "low" and r == 0) or (regime == "high" and r == 1):
            windows.append((t, r, w))
    if histories is not None and histories < len(windows):
        windows = [windows[i] for i in subsample(len(windows), histories, seed)]
    n_hist = len(windows)
    n_streams = n_draws // 2 if antithetic else n_draws
    cells = (H + 1) * k

    root = np.random.SeedSequence(seed)
    hist_seqs = root.spawn(n_hist)
    per_hist = np.zeros((n_hist, H + 1, k))
    var_eff = np.zeros((n_hist, H + 1, k))
    var_draw = np.zeros((n_hist, H + 1, k))
    q_lo = np.zeros((n_hist, H + 1, k))
    q_hi = np.zeros((n_hist, H + 1, k))
    regimes = []
    times = []
    for i, (t, r0, w) in enumerate(windows):
        draw_seqs = hist_seqs[i].spawn(n_streams)
        draws = np.zeros((n_draws, H + 1, k))
        eff = np.zeros((n_streams, H + 1, k))
        ridx = 0
        for s_idx, ds in enumerate(draw_seqs):
            gen = np.random.Generator(np.random.Philox(ds))
            z = normals(gen, cells).reshape(H + 1, k)
            for sign in ([1.0, -1.0] if antithetic else [1.0]):
                wa = w.copy()
                wb = w.copy()
                for h in range(H + 1):
                    sa = tvar_regime(wa, L, d, tv, gamma)
                    ua = sign * (chol[sa] @ z[h])
                    if h == 0:
                        ua = ua + delta[r0]
                    ya = tvar_step(coefs[sa], constant, wa, p, k, ua)
                    sb = tvar_regime(wb, L, d, tv, gamma)
                    ub = sign * (chol[sb] @ z[h])
                    yb = tvar_step(coefs[sb], constant, wb, p, k, ub)
                    draws[ridx, h] = ya - yb
                    wa = np.vstack([wa[1:], ya])
                    wb = np.vstack([wb[1:], yb])
                eff[s_idx] += draws[ridx] / (2.0 if antithetic else 1.0)
                ridx += 1
        per_hist[i] = draws.mean(axis=0)
        var_eff[i] = eff.var(axis=0, ddof=1) if n_streams >= 2 else np.nan
        var_draw[i] = draws.var(axis=0, ddof=1) if n_draws >= 2 else np.nan
        q_lo[i] = np.percentile(draws, 100 * bands[0], axis=0)
        q_hi[i] = np.percentile(draws, 100 * bands[1], axis=0)
        regimes.append(r0)
        times.append(t)

    regimes = np.asarray(regimes)
    girf = per_hist.mean(axis=0)
    out = {
        "shock": shock,
        "shock_var": shock_var,
        "size": size,
        "horizon": H,
        "n_draws": n_draws,
        "seed": seed,
        "antithetic": antithetic,
        "bands": list(bands),
        "regime": regime,
        "histories": histories,
        "girf": girf.tolist(),
        "lower": np.percentile(per_hist, 100 * bands[0], axis=0).tolist(),
        "upper": np.percentile(per_hist, 100 * bands[1], axis=0).tolist(),
        "per_history": per_hist.tolist(),
        "mc_se": (np.sqrt(var_eff.sum(axis=0) / n_streams) / n_hist).tolist(),
        "draw_sd": np.sqrt(var_draw.mean(axis=0)).tolist(),
        "draw_lower": q_lo.mean(axis=0).tolist(),
        "draw_upper": q_hi.mean(axis=0).tolist(),
        "girf_low_regime": (per_hist[regimes == 0].mean(axis=0).tolist()
                            if (regimes == 0).any() else None),
        "girf_high_regime": (per_hist[regimes == 1].mean(axis=0).tolist()
                             if (regimes == 1).any() else None),
        "history_regimes": regimes.tolist(),
        "history_times": times,
        "n_histories": int(n_hist),
        "n_low_histories": int((regimes == 0).sum()),
        "n_high_histories": int((regimes == 1).sum()),
        "shock_by_regime": [dl.tolist() for dl in delta],
    }
    return out


# ------------------------------------------------------------------- main

def main():
    # Linear VAR(2), k = 3, a full (non-diagonal) innovation covariance so
    # the generalized and orthogonal shocks differ.
    chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]]
    y_lin = sim_var(
        20260910, 300, c=[0.5, -0.2, 0.1],
        a_list=[[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]],
                [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]],
        chol=chol)
    linear_cases = []
    for p, trend, shock_var, size, H in [
        (2, "c", 0, 1.0, 12),
        (2, "c", 2, -1.5, 12),
        (1, "n", 1, 2.0, 8),
        (3, "c", 1, 1.0, 10),
    ]:
        linear_cases.append(linear_case(y_lin, p, trend, shock_var, size, H))

    # Threshold VAR: the strongly separated two-regime DGP of tvar.json,
    # short (T = 90) so the transcription stays cheap.
    y_tv = sim_tvar(
        20260911, 90, 1, 0.0,
        c_low=[1.0, 0.3], a_low=[[0.5, 0.1], [0.2, 0.4]],
        c_high=[-1.0, -0.3], a_high=[[0.1, 0.0], [-0.1, 0.5]])
    p, tv, delays, trim, constant = 1, 0, [1], 0.15, True
    fit = fit_tvar(y_tv, p, tv, delays, trim, constant)
    tvar_cases = []
    for shock, shock_var, size, H, n_draws, seed, anti, bands, regime, hist in [
        ("orthogonal", 0, 1.0, 6, 4, 7, True, (0.16, 0.84), "all", None),
        ("generalized", 1, -2.0, 5, 3, 11, False, (0.10, 0.90), "all", None),
        ("orthogonal", 0, 2.0, 6, 4, 3, True, (0.16, 0.84), "low", 8),
        ("generalized", 0, 1.0, 4, 4, 5, True, (0.25, 0.75), "high", None),
    ]:
        case = tvar_girf_transcribed(y_tv, fit, p, constant, shock, shock_var,
                                     size, H, n_draws, seed, anti, bands,
                                     regime, hist)
        tvar_cases.append(case)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "note": (
                "Koop-Pesaran-Potter GIRF engine: linear VAR orthogonal shock "
                "= statsmodels VARResults.irf(orth=True) (independent package), "
                "generalized shock = the Pesaran-Shin (1998) closed form "
                "transcribed in NumPy (documented formula); threshold-VAR GIRFs "
                "= an independent NumPy transcription of the documented engine "
                "reproducing its Philox streams (no third-party TVAR GIRF runs "
                "in this container: CRAN unreachable, no tsDyn)."
            ),
        },
        "linear": {
            "series": y_lin.tolist(),
            "cases": linear_cases,
        },
        "tvar": {
            "series": y_tv.tolist(),
            "p": p,
            "threshold_index": tv,
            "delays": delays,
            "trim": trim,
            "constant": constant,
            "fit": {kk: fit[kk] for kk in ("threshold", "delay", "coefs_low",
                                          "coefs_high", "sigma_low", "sigma_high",
                                          "n_low", "n_high", "nobs")},
            "cases": tvar_cases,
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); "
          f"{len(linear_cases)} linear cases, {len(tvar_cases)} TVAR cases")
    for c in linear_cases:
        print(f"  linear p={c['p']} trend={c['trend']} j={c['shock_var']} "
              f"size={c['size']} orth[1]={np.round(c['girf_orthogonal'][1], 5)} "
              f"gen[1]={np.round(c['girf_generalized'][1], 5)}")
    for c in tvar_cases:
        print(f"  tvar {c['shock']:11s} j={c['shock_var']} size={c['size']} "
              f"regime={c['regime']:4s} n_hist={c['n_histories']} "
              f"(low {c['n_low_histories']}, high {c['n_high_histories']}) "
              f"girf[1]={np.round(c['girf'][1], 5)}")


if __name__ == "__main__":
    main()
