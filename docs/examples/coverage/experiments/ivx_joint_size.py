"""What holds the joint IVX test's size in the number of predictors? (audit rounds 3-4, finding 4)

    .venv/bin/python docs/examples/coverage/experiments/ivx_joint_size.py            # full, ~20 min
    .venv/bin/python docs/examples/coverage/experiments/ivx_joint_size.py --quick    # smoke, ~1 min

THE PROBLEM BEING MEASURED
--------------------------
Rounds 3-4 measured `ivx_test`'s chi-square(k) joint Wald over-rejecting a
true null in the NUMBER of predictors at the shipped tuning (alpha = 0.95):
size ~0.05 / 0.10 / 0.17 / 0.26 at k = 1/3/5/8 (rho = 1, endogeneity -0.9,
n = 250), with the excess decaying like n^{-(1-alpha)/2} — i.e. never, at any
realistic sample. k = 1 is at nominal; the certified property tests only ever
ran k = 1.

This script measures the candidate repairs on shared seeded draws:

  raw a=0.95     the shipped statistic (M = s2u * Z'Z), default tuning
  raw a=0.70/0.50/0.30
                 the alpha ladder — a = 0.5 is the tuning the card recommends
                 for many predictors; the ladder shows what it buys and what
                 it does not
  dem            M = s2u * (Z'Z - N zbar zbar') — the demeaned variance the
                 rounds-3/4 finder tested and reported as rejecting MORE;
                 verified here so it can be discarded with evidence (it is
                 exactly right when every predictor is exogenous, and exactly
                 wrong under endogeneity, where the demeaning term carries a
                 mean shift, not just variance)
  fm             M = s2u * Z'Z - N zbar zbar' * Omega_FM with
                 Omega_FM = s2u - cov(u,e) Omega_ee^{-1} cov(e,u) — the
                 KMS-style FM-corrected normaliser (measured: no repair; the
                 endogenous coordinate's shift is conditionally deterministic
                 given the instrument path, so no variance matrix absorbs it)
  wild           a restricted SYSTEM wild bootstrap of the raw statistic
                 (null-imposed return residuals, AR(1) predictor residuals,
                 one shared Rademacher weight per date so the u-e pairing
                 survives, x* rebuilt recursively) — measured at the corner
                 cells with per-test runtime
  bonf           the union-intersection combination: the SCALAR IVX-Wald test
                 (whose measured size holds deep into its tail) run per
                 predictor, joint rejection when min_j p_j < level / k.
                 THE MEASURED WINNER -- shipped as
                 ivx_test(..., joint="bonferroni") via
                 tsecon_predreg::ivx_bonferroni; the full verdict table is in
                 docs/roadmap/21-long-horizon-and-joint-inference.md.

plus two DIAGNOSTIC arms that explain the mechanism (printed, not candidates):
the oracle-covariance statistic (even the true E[cc'] does not restore
chi-square — the failure is not one bad variance matrix) and the demeaned
variance at zero endogeneity (nominal at every k — which locates the whole
distortion in the endogeneity-x-demeaning interaction).

DESIGNS
-------
`first`: k independent random-walk/AR predictors, endogeneity delta carried
by predictor 0 only — reproduces the audit table almost exactly.
`factor`: every predictor's innovation loads on u with correlation delta
(cross-correlated predictors — closer to a panel of valuation ratios).
Both are run for the headline size table.

POWER
-----
Size-adjusted power (empirical null critical values from the same cells) for
the default and a=0.5, versus raw Bonferroni power (its size is at or below
nominal, so its raw power needs no adjustment): sparse alternative (one
genuine predictor) and diffuse alternative (equal slopes summing to the same
signal). MC standard error at 2000 reps is <= 0.011.

The NumPy transcription of `ivx_multi` is validated against the installed
`tsecon.ivx_test` (and the scalar path against `predictive_regression`) to
~1e-10 on a fixed dataset before anything is measured. Seeded end to end.
"""

import argparse
import sys
import time

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


def simulate_cell(rng, reps, n, k, rho, delta, design, beta=None):
    """reps datasets at once: r (reps, n), x (reps, n, k).

    r[t] carries u_t; the estimators regress r_{t+1} on x_t via their own
    alignment, so `r = u` is exactly the null of no predictability.
    """
    u = rng.standard_normal((reps, n))
    xi = rng.standard_normal((reps, n, k))
    if design == "factor":
        e = delta * u[:, :, None] + np.sqrt(1 - delta * delta) * xi
    elif design == "first":
        e = xi.copy()
        if abs(delta) > 0:
            e[:, :, 0] = delta * u + np.sqrt(1 - delta * delta) * xi[:, :, 0]
    else:
        raise ValueError(design)
    x = np.zeros((reps, n, k))
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + e[:, t]
    r = u.copy()
    if beta is not None:
        r[:, 1:] += np.einsum("stk,k->st", x[:, :-1], np.asarray(beta))
    return r, x


# ---------------------------------------------------------------------------
# Batched IVX machinery (transcribed from crates/tsecon-predreg/src/ivx.rs;
# cross-checked against the installed tsecon below)
# ---------------------------------------------------------------------------


def build_instrument(x, cz, alpha):
    """z (reps, N, k): z_0 = 0, z_t = Rz z_{t-1} + Dx_{t-1}."""
    reps, n, k = x.shape
    rz = 1.0 + cz / n ** alpha
    dx = x[:, 1:] - x[:, :-1]
    z = np.zeros((reps, n - 1, k))
    acc = np.zeros((reps, k))
    for t in range(n - 1):
        z[:, t] = acc
        acc = rz * acc + dx[:, t]
    return z


def ivx_joint_stats(r, x, cz=-1.0, alpha=0.95, variants=("raw",)):
    """The joint Wald under several normalisers, on the same data.

    Returns dict of (reps,) statistic arrays for the requested variants among
    "raw" (shipped: M = s2u Z'Z), "dem" (demeaned Szz), "fm" (KMS-style
    FM-corrected).
    """
    reps, n, k = x.shape
    big_n = n - 1
    z = build_instrument(x, cz, alpha)
    b = r[:, 1:]
    a = x[:, : n - 1]
    bd = b - b.mean(axis=1, keepdims=True)
    ad = a - a.mean(axis=1, keepdims=True)
    g = np.linalg.solve(
        np.einsum("stk,stl->skl", ad, ad),
        np.einsum("stk,st->sk", ad, bd)[..., None],
    )[..., 0]
    resid = bd - np.einsum("stk,sk->st", ad, g)
    s2u = np.einsum("st,st->s", resid, resid) / big_n
    c = np.einsum("stk,st->sk", z, bd)
    szz = np.einsum("stk,stl->skl", z, z)
    zbar = z.mean(axis=1)
    out = {}
    m_raw = s2u[:, None, None] * szz
    if "raw" in variants:
        out["raw"] = np.einsum("sk,sk->s", c, np.linalg.solve(m_raw, c[..., None])[..., 0])
    zz_outer = big_n * np.einsum("sk,sl->skl", zbar, zbar)
    if "dem" in variants:
        m_dem = s2u[:, None, None] * (szz - zz_outer)
        out["dem"] = np.einsum("sk,sk->s", c, np.linalg.solve(m_dem, c[..., None])[..., 0])
    if "fm" in variants:
        # AR(1)+intercept residuals per predictor; iid-innovation FM pieces.
        xm = a - a.mean(axis=1, keepdims=True)
        ym = x[:, 1:] - x[:, 1:].mean(axis=1, keepdims=True)
        rho_hat = np.einsum("stk,stk->sk", xm, ym) / np.einsum("stk,stk->sk", xm, xm)
        ehat = ym - rho_hat[:, None, :] * xm
        cov_ue = np.einsum("st,stk->sk", resid, ehat) / big_n
        cov_ee = np.einsum("stk,stl->skl", ehat, ehat) / big_n
        omega_fm = s2u - np.einsum(
            "sk,sk->s", cov_ue, np.linalg.solve(cov_ee, cov_ue[..., None])[..., 0]
        )
        omega_fm = np.maximum(omega_fm, 0.0)
        m_fm = m_raw - omega_fm[:, None, None] * zz_outer
        out["fm"] = np.einsum("sk,sk->s", c, np.linalg.solve(m_fm, c[..., None])[..., 0])
    return out


def ivx_scalar_stats(r, x, cz=-1.0, alpha=0.95):
    """Per-predictor SCALAR IVX Walds (reps, k) — each column exactly the
    shipped scalar `ivx` (own simple OLS residual variance)."""
    reps, n, k = x.shape
    big_n = n - 1
    z = build_instrument(x, cz, alpha)
    b = r[:, 1:]
    bd = b - b.mean(axis=1, keepdims=True)
    ws = np.empty((reps, k))
    for j in range(k):
        a = x[:, : n - 1, j]
        ad = a - a.mean(axis=1, keepdims=True)
        gj = np.einsum("st,st->s", ad, bd) / np.einsum("st,st->s", ad, ad)
        resid = bd - gj[:, None] * ad
        s2u = np.einsum("st,st->s", resid, resid) / big_n
        cj = np.einsum("st,st->s", z[:, :, j], bd)
        szz = np.einsum("st,st->s", z[:, :, j], z[:, :, j])
        ws[:, j] = cj * cj / (s2u * szz)
    return ws


def bonferroni_reject(ws, level=0.05):
    """Union-intersection rejection: min_j p_j < level / k."""
    k = ws.shape[1]
    pmin = stats.chi2.sf(ws.max(axis=1), 1)
    return pmin < level / k


def wild_pvalues(r, x, b_boot, rng, cz=-1.0, alpha=0.95):
    """Restricted system wild bootstrap p-values, one per dataset.

    Null-imposed return residuals u0 = b - mean(b); AR(1)+intercept residuals
    ehat per predictor; ONE Rademacher weight per date multiplies (u0, ehat)
    jointly so the endogeneity survives; x* rebuilt recursively at rho_hat.
    """
    reps, n, k = x.shape
    pvals = np.empty(reps)
    for s in range(reps):
        rs1, xs1 = r[s], x[s]
        w0 = ivx_joint_stats(rs1[None], xs1[None], cz, alpha)["raw"][0]
        b = rs1[1:]
        u0 = b - b.mean()
        xlag, xlead = xs1[:-1], xs1[1:]
        xm = xlag - xlag.mean(axis=0)
        ym = xlead - xlead.mean(axis=0)
        rho = np.einsum("tk,tk->k", xm, ym) / np.einsum("tk,tk->k", xm, xm)
        icept = xlead.mean(axis=0) - rho * xlag.mean(axis=0)
        ehat = xlead - icept - rho * xlag
        eta = np.where(rng.random((b_boot, n - 1)) < 0.5, -1.0, 1.0)
        us = u0[None, :] * eta
        es = ehat[None, :, :] * eta[:, :, None]
        xs = np.empty((b_boot, n, k))
        xs[:, 0] = xs1[0]
        for t in range(1, n):
            xs[:, t] = icept + rho * xs[:, t - 1] + es[:, t - 1]
        rb = np.empty((b_boot, n))
        rb[:, 0] = rs1[0]
        rb[:, 1:] = b.mean() + us
        wsb = ivx_joint_stats(rb, xs, cz, alpha)["raw"]
        pvals[s] = (1.0 + np.sum(wsb >= w0)) / (b_boot + 1.0)
    return pvals


# ---------------------------------------------------------------------------
# Validation against the installed tsecon (never used for the measurement)
# ---------------------------------------------------------------------------


def cross_check_against_tsecon(seed=20260818):
    try:
        import tsecon
    except ImportError:
        print("cross-check: tsecon not importable here; skipped (the predreg "
              "golden pins the same algebra)")
        return
    rng = np.random.default_rng(seed)
    r, x = simulate_cell(rng, 1, 300, 4, 0.98, -0.8, "first")
    mine = ivx_joint_stats(r, x)["raw"][0]
    ts = tsecon.ivx_test(r[0], x[0])
    assert abs(mine - ts["wald"]) < 1e-8 * max(1.0, abs(ts["wald"])), (mine, ts["wald"])
    ws = ivx_scalar_stats(r, x)[0]
    for j in range(4):
        sc = tsecon.predictive_regression(r[0], x[0, :, j])["ivx"]["wald"]
        assert abs(ws[j] - sc) < 1e-8 * max(1.0, abs(sc)), (j, ws[j], sc)
    bf = tsecon.ivx_test(r[0], x[0], joint="bonferroni")
    assert abs(bf["wald"] - ws.max()) < 1e-8
    print("cross-check vs tsecon: joint wald, per-column scalar walds, and "
          "joint='bonferroni' max all agree to ~1e-10")


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def size_tables(reps, seed):
    alphas = (0.95, 0.70, 0.50, 0.30)
    print("\n== SIZE (nominal 0.05; rejection rates of a TRUE null) ==")
    header = ("design  delta  rho   n     k | " +
              " ".join(f"a{a:<4}" for a in alphas) +
              " | dem    fm     bonf")
    print(header)
    store = {}
    for design in ("first", "factor"):
        for delta in (-0.9, 0.0):
            for rho in (1.0, 0.95):
                for n in (250, 1000):
                    for k in (1, 3, 5, 8):
                        rng = np.random.default_rng(
                            seed + hash((design, delta, rho, n, k)) % (2**32)
                        )
                        r, x = simulate_cell(rng, reps, n, k, rho, delta, design)
                        crit = stats.chi2.ppf(0.95, k)
                        cells = {}
                        for a in alphas:
                            w = ivx_joint_stats(r, x, alpha=a)["raw"]
                            cells[f"a{a}"] = np.mean(w > crit)
                            store[(design, delta, rho, n, k, f"raw{a}")] = w
                        extra = ivx_joint_stats(r, x, variants=("dem", "fm"))
                        cells["dem"] = np.mean(extra["dem"] > crit)
                        cells["fm"] = np.mean(extra["fm"] > crit)
                        ws = ivx_scalar_stats(r, x)
                        cells["bonf"] = np.mean(bonferroni_reject(ws))
                        store[(design, delta, rho, n, k, "scalar")] = ws
                        line = " ".join(f"{cells[f'a{a}']:.3f}" for a in alphas)
                        print(f"{design:6s}  {delta:+.1f}  {rho:.2f}  {n:<5d} {k} | "
                              f"{line} | {cells['dem']:.3f}  {cells['fm']:.3f}  "
                              f"{cells['bonf']:.3f}")
    return store


def wild_table(reps, b_boot, seed):
    print(f"\n== WILD BOOTSTRAP (restricted system; B={b_boot}; nominal 0.05) ==")
    for design, delta, rho, n, k in [
        ("first", -0.9, 1.0, 250, 1),
        ("first", -0.9, 1.0, 250, 8),
        ("first", 0.0, 1.0, 250, 1),
        ("first", 0.0, 1.0, 250, 8),
    ]:
        rng = np.random.default_rng(seed + 77 + k + (delta < 0) * 1000)
        r, x = simulate_cell(rng, reps, n, k, rho, delta, design)
        t0 = time.time()
        p = wild_pvalues(r, x, b_boot, rng)
        dt = (time.time() - t0) / reps
        print(f"  {design} delta={delta:+.1f} rho={rho} n={n} k={k}: "
              f"size={np.mean(p < 0.05):.3f}  ({dt*1000:.0f} ms/test)")


def power_tables(reps, seed, store):
    print("\n== POWER at n=250, delta=-0.9, design 'first' (nominal 0.05) ==")
    print("size-adjusted default / size-adjusted a=0.5 / bonferroni (raw — its size is <= nominal)")
    for rho in (1.0, 0.95):
        for k in (3, 8):
            null_def = store[("first", -0.9, rho, 250, k, "raw0.95")]
            null_a05 = store[("first", -0.9, rho, 250, k, "raw0.5")]
            q_def = np.quantile(null_def, 0.95)
            q_a05 = np.quantile(null_a05, 0.95)
            for kind in ("sparse", "diffuse"):
                for b0 in (0.02, 0.04):
                    beta = np.zeros(k)
                    if kind == "sparse":
                        beta[0] = b0
                    else:
                        beta[:] = b0 / np.sqrt(k)
                    rng = np.random.default_rng(
                        seed + 31 + hash((rho, k, kind, b0)) % (2**32)
                    )
                    r, x = simulate_cell(rng, reps, 250, k, rho, -0.9, "first",
                                         beta=beta)
                    w_def = ivx_joint_stats(r, x, alpha=0.95)["raw"]
                    w_a05 = ivx_joint_stats(r, x, alpha=0.5)["raw"]
                    ws = ivx_scalar_stats(r, x)
                    print(f"  rho={rho} k={k} {kind:7s} b={b0:.3f}: "
                          f"default(adj)={np.mean(w_def > q_def):.3f}  "
                          f"a0.5(adj)={np.mean(w_a05 > q_a05):.3f}  "
                          f"bonf={np.mean(bonferroni_reject(ws)):.3f}")


def diagnostics(reps, seed):
    """The two facts that locate the mechanism (printed for the record)."""
    print("\n== MECHANISM DIAGNOSTICS (design 'first', n=250, rho=1) ==")
    rng = np.random.default_rng(seed + 5000)
    # (a) scalar test is calibrated DEEP into its tail at delta=-0.9 --
    #     the k=1 'cancellation' is not a 5%-point coincidence.
    r, x = simulate_cell(rng, max(reps * 4, 4000), 250, 1, 1.0, -0.9, "first")
    w = ivx_joint_stats(r, x)["raw"]
    tail = {q: float(np.mean(w > stats.chi2.ppf(q, 1))) for q in (0.95, 0.99, 0.999)}
    print(f"  scalar tail calibration at delta=-0.9: "
          + " ".join(f"P>chi2_{q}={v:.4f}" for q, v in tail.items()))
    # (b) the demeaned variance is exactly right when every predictor is
    #     exogenous -- the k-distortion lives in endogeneity x demeaning.
    for k in (1, 8):
        r, x = simulate_cell(rng, reps, 250, k, 1.0, 0.0, "first")
        w = ivx_joint_stats(r, x, variants=("raw", "dem"))
        crit = stats.chi2.ppf(0.95, k)
        print(f"  delta=0 k={k}: raw={np.mean(w['raw'] > crit):.3f} (conservative)  "
              f"dem={np.mean(w['dem'] > crit):.3f} (nominal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    reps = args.reps or (100 if args.quick else 2000)
    wild_reps = max(reps // 4, 25)
    b_boot = 99 if args.quick else 199
    seed = 20260818

    print(f"ivx joint size/power candidates | reps={reps} wild_reps={wild_reps} B={b_boot}")
    cross_check_against_tsecon()
    t0 = time.time()
    store = size_tables(reps, seed)
    diagnostics(reps, seed)
    power_tables(reps, seed, store)
    wild_table(wild_reps, b_boot, seed)
    print(f"\ntotal wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
