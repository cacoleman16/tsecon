#!/usr/bin/env python3
"""LP versus VAR, head to head -- every number in
docs/guide/16-lp-vs-var-head-to-head.md is printed by this script.

Run with the project venv on a RELEASE build of the extension:

    .venv/bin/python docs/examples/lp_vs_var_head_to_head.py

Two parts, both seeded, so every printed number reproduces bit-for-bit:

  1. The Plagborg-Møller & Wolf (2021) finite-sample check. On one draw from
     a bivariate Gaussian VAR(2) it fits `var_irf` (Cholesky, shock ordered
     first, unit-normalised) and `lp` with the same lag length, prints the
     gap at every horizon, and reports the horizons at which the two point
     estimates agree to machine precision. It then rebuilds the LP with the
     matched control set the theorem needs (p lags of EVERY VAR variable,
     via `tsecon.ols`) and shows where the exact identity lives, and how
     each gap behaves as T grows.
  2. The Li, Plagborg-Møller & Wolf (2024) bias-variance Monte Carlo on a
     persistent ARMA-X DGP that no finite-order VAR nests: bias, SD and RMSE
     of `var_irf` (p = 1, 4, 12), `lp` (p = 1, 4, 12 lag-augmented, p = 4
     HAC) and `smooth_lp`, plus the pointwise 95% coverage of
     `var_irf_bands`, the lag-augmented LP bands, the HAC LP bands and the
     smooth-LP bands.
"""
from __future__ import annotations

import argparse
import os
import platform
import time

import numpy as np
import scipy
from scipy.stats import norm

import tsecon

Z95 = float(norm.ppf(0.975))
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def build_mode() -> str:
    """Which cargo profile the installed extension was copied from.

    The verdict is an exact size match against an on-disk cargo artifact
    ($CARGO_TARGET_DIR or the repo's target/); anything else is 'unknown'.
    """
    pkg = os.path.dirname(os.path.abspath(tsecon.__file__))
    sos = [f for f in os.listdir(pkg) if f.endswith((".so", ".dylib", ".pyd"))]
    if not sos:
        return "unknown (no compiled extension next to tsecon)"
    size = os.path.getsize(os.path.join(pkg, sos[0]))
    tdirs = [d for d in (os.environ.get("CARGO_TARGET_DIR"), os.path.join(REPO, "target")) if d]
    for profile in ("debug", "release"):
        for td in tdirs:
            for name in ("lib_core.so", "lib_core.dylib", "libtsecon.so", "libtsecon.dylib"):
                cand = os.path.join(td, profile, name)
                if os.path.exists(cand) and os.path.getsize(cand) == size:
                    return f"{profile} (== <target>/{profile}/{name}, {size / 1e6:.1f} MB)"
    return f"unknown ({sos[0]}, {size / 1e6:.1f} MB; no cargo artifact of that size found)"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def rule(width=78):
    print("-" * width)


def header(title):
    print()
    rule()
    print(title)
    rule()


# --------------------------------------------------------------------------
# Part 1 -- the finite-sample check of the PMW (2021) equivalence
# --------------------------------------------------------------------------
def simulate_var2(rng, T, persistent_shock: bool, burn=200):
    """One draw of a stable bivariate Gaussian VAR(2), shock variable first.

    persistent_shock=True: x has its own AR dynamics (a generic VAR column).
    persistent_shock=False: x is white noise, the innovation-like impulse
    `tsecon.lp` is built for. Either way x and y are contemporaneously
    correlated, so the Cholesky step is not a no-op.
    """
    A1 = np.array([[0.5, 0.0], [0.4, 0.6]])
    A2 = np.array([[0.2, 0.0], [0.1, -0.2]])
    if not persistent_shock:
        A1[0, :] = 0.0
        A2[0, :] = 0.0
    P = np.array([[1.0, 0.0], [0.5, 0.8]])  # chol(Sigma_u)
    z = np.zeros((T + burn, 2))
    eps = rng.standard_normal((T + burn, 2))
    for t in range(2, T + burn):
        z[t] = A1 @ z[t - 1] + A2 @ z[t - 2] + P @ eps[t]
    return z[burn:]


def var_unit_irf(data, p, H):
    """Cholesky IRF of column 1 to the column-0 shock, per UNIT impact."""
    v = tsecon.var_irf(data, lags=p, horizon=H, orth=True)
    return np.array([v[h][1][0] for h in range(H + 1)]) / v[0][0][0]


def matched_lp(y, x, p, H):
    """LP with the control set the PMW identity needs: a constant, x_t, and
    p lags of BOTH variables. Sample at horizon h: t = p, ..., T-1-h, the
    VAR's own effective sample minus the last h rows."""
    T = len(y)
    out = []
    for h in range(H + 1):
        cols = [np.ones(T - p - h), x[p:T - h]]
        for lag in range(1, p + 1):
            cols.append(x[p - lag:T - h - lag])
            cols.append(y[p - lag:T - h - lag])
        X = np.column_stack(cols)
        out.append(tsecon.ols(y[p + h:T], X)["params"][1])
    return np.array(out)


def part1(seed=20260910, T=400, H=12, p=2, tol=1e-12):
    header("PART 1 -- Plagborg-Moller & Wolf (2021): where is the finite-sample identity?")
    rng = np.random.default_rng(seed)
    print(f"DGP: bivariate Gaussian VAR({p}), shock variable x ordered first, T={T}, "
          f"horizons 0..{H}, seed={seed}")
    print("VAR(2) IRF = Cholesky response of y to the x shock, divided by the x impact")
    print("            (one-SD shock -> unit shock, the LP normalisation).")
    print("lp(y, x, n_lag_controls=2) controls: constant, 2 own-lags of y, and on the")
    print("            lag-augmented path h lags of x; on the HAC path no x lags.")
    print("matched LP: constant, x_t, 2 lags of x AND 2 lags of y (tsecon.ols).")

    for persistent in (True, False):
        data = simulate_var2(rng, T, persistent_shock=persistent)
        x, y = data[:, 0], data[:, 1]
        fit = tsecon.var_fit(data, lags=p)
        v = var_unit_irf(data, p, H)
        la = np.array(tsecon.lp(y, x, horizons=H, n_lag_controls=p)["irf"])
        hac = np.array(tsecon.lp(y, x, horizons=H, n_lag_controls=p, se="hac")["irf"])
        mt = matched_lp(y, x, p, H)
        label = "x has its own AR dynamics" if persistent else "x is white noise (innovation-like)"
        print()
        print(f"[{label}]  var_fit is_stable={fit['is_stable']}  min_root={fit['min_root']:.3f}")
        print(f"  {'h':>2} {'VAR(2)':>10} {'lp lag-aug':>11} {'lp hac':>10} {'matched':>10} "
              f"{'|VAR-lagaug|':>13} {'|VAR-hac|':>10} {'|VAR-matched|':>14}")
        for h in range(H + 1):
            print(f"  {h:>2} {v[h]:>10.6f} {la[h]:>11.6f} {hac[h]:>10.6f} {mt[h]:>10.6f} "
                  f"{abs(v[h] - la[h]):>13.2e} {abs(v[h] - hac[h]):>10.2e} {abs(v[h] - mt[h]):>14.2e}")
        for name, est in (("lp lag-augmented", la), ("lp hac", hac), ("matched-controls LP", mt)):
            agree = [h for h in range(H + 1) if abs(v[h] - est[h]) <= tol]
            print(f"  horizons where VAR(2) == {name} to {tol:.0e}: "
                  f"{agree if agree else 'NONE'}")

    print()
    print("How each gap scales with T (max over the stated horizons, one draw per T):")
    print(f"  {'T':>6} | {'persistent x: lagaug h=0':>24} {'matched h=0':>12} {'matched h>=1':>13} "
          f"| {'white-noise x: lagaug h=0':>25} {'matched h>=1':>13}")
    for T_ in (200, 800, 3200, 12800):
        row = []
        for persistent in (True, False):
            d = simulate_var2(rng, T_, persistent_shock=persistent)
            x_, y_ = d[:, 0], d[:, 1]
            v_ = var_unit_irf(d, p, H)
            la_ = np.array(tsecon.lp(y_, x_, horizons=H, n_lag_controls=p)["irf"])
            mt_ = matched_lp(y_, x_, p, H)
            row.append((abs(v_[0] - la_[0]), abs(v_[0] - mt_[0]), np.max(np.abs(v_[1:] - mt_[1:]))))
        (a0, m0, m1), (b0, _, n1) = row
        print(f"  {T_:>6} | {a0:>24.2e} {m0:>12.2e} {m1:>13.2e} | {b0:>25.2e} {n1:>13.2e}")


# --------------------------------------------------------------------------
# Part 2 -- the LPW (2024) bias-variance Monte Carlo
# --------------------------------------------------------------------------
A1, A2, B0, B1, THETA = 1.5, -0.54, 1.0, 0.5, 0.6


def simulate_armax(rng, T, burn=200):
    """y_t = 1.5 y_{t-1} - 0.54 y_{t-2} + x_t + 0.5 x_{t-1} + e_t + 0.6 e_{t-1},
    x_t ~ iid N(0,1) observed, e_t ~ iid N(0,1). AR roots 0.9 and 0.6; the
    MA(1) error means no finite-order VAR is correctly specified."""
    n = T + burn
    x = rng.standard_normal(n)
    e = rng.standard_normal(n)
    u = e.copy()
    u[1:] += THETA * e[:-1]
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = A1 * y[t - 1] + A2 * y[t - 2] + B0 * x[t] + B1 * x[t - 1] + u[t]
    return x[burn:], y[burn:]


def true_irf(H):
    r = np.zeros(H + 1)
    r[0] = B0
    r[1] = A1 * r[0] + B1
    for h in range(2, H + 1):
        r[h] = A1 * r[h - 1] + A2 * r[h - 2]
    return r


def part2(seed=20260910, T=240, H=20, reps=500):
    header("PART 2 -- Li, Plagborg-Moller & Wolf (2024): the bias-variance trade-off, measured")
    rng = np.random.default_rng(seed)
    truth = true_irf(H)
    print(f"DGP: y_t = {A1} y_(t-1) {A2:+} y_(t-2) + {B0} x_t + {B1} x_(t-1) + e_t + {THETA} e_(t-1),")
    print("     x_t ~ iid N(0,1) observed shock, e_t ~ iid N(0,1); AR roots 0.9 and 0.6.")
    print(f"T={T}, horizons 0..{H}, {reps} replications, seed={seed}, nominal 95% pointwise bands.")
    print("Estimand: response of y to a unit x shock. VAR IRFs are Cholesky (x first)")
    print("divided by the x impact for bias/SD/RMSE; var_irf_bands cover the one-SD")
    print("response, which equals the unit response here because sd(x) = 1.")
    print("True IRF:", " ".join(f"{t:.3f}" for t in truth))

    est_names = ["VAR(1)", "VAR(4)", "VAR(12)",
                 "LP(1) lag-aug", "LP(4) lag-aug", "LP(12) lag-aug",
                 "LP(4) HAC", "smooth LP(4) cv"]
    cov_names = ["VAR(1) asymptotic", "VAR(4) asymptotic", "VAR(12) asymptotic",
                 "LP(1) lag-aug HC1", "LP(4) lag-aug HC1", "LP(12) lag-aug HC1",
                 "LP(4) HAC", "smooth LP(4) se|lambda"]
    est = {k: np.zeros((reps, H + 1)) for k in est_names}
    cov = {k: np.zeros((reps, H + 1)) for k in cov_names}
    lam_used = np.zeros(reps)
    raw_gap = 0.0  # max |smooth_lp irf_raw - lp(se="hac") irf|: the lam=0 anchor, checked live
    t0 = time.perf_counter()
    for r in range(reps):
        x, y = simulate_armax(rng, T)
        data = np.column_stack([x, y])
        for p in (1, 4, 12):
            b = tsecon.var_irf_bands(data, lags=p, horizon=H, orth=True,
                                     method="asymptotic", alpha=0.05)
            pt = np.array([b["point"][h][1][0] for h in range(H + 1)])
            lo = np.array([b["lower"][h][1][0] for h in range(H + 1)])
            hi = np.array([b["upper"][h][1][0] for h in range(H + 1)])
            est[f"VAR({p})"][r] = pt / b["point"][0][0][0]
            cov[f"VAR({p}) asymptotic"][r] = (lo <= truth) & (truth <= hi)
            l = tsecon.lp(y, x, horizons=H, n_lag_controls=p)
            irf, se = np.array(l["irf"]), np.array(l["se"])
            est[f"LP({p}) lag-aug"][r] = irf
            cov[f"LP({p}) lag-aug HC1"][r] = np.abs(irf - truth) <= Z95 * se
        l = tsecon.lp(y, x, horizons=H, n_lag_controls=4, se="hac")
        irf, se = np.array(l["irf"]), np.array(l["se"])
        est["LP(4) HAC"][r] = irf
        cov["LP(4) HAC"][r] = np.abs(irf - truth) <= Z95 * se
        s = tsecon.smooth_lp(y, x, horizons=H, n_lag_controls=4, lam="cv")
        irf, se = np.array(s["irf"]), np.array(s["se"])
        est["smooth LP(4) cv"][r] = irf
        cov["smooth LP(4) se|lambda"][r] = np.abs(irf - truth) <= Z95 * se
        lam_used[r] = s["lambda_used"]
        raw_gap = max(raw_gap, float(np.max(np.abs(np.array(s["irf_raw"]) - est["LP(4) HAC"][r]))))
    elapsed = time.perf_counter() - t0
    print(f"Monte Carlo time: {elapsed:.1f} s ({elapsed / reps * 1e3:.0f} ms per replication)")

    show = [0, 1, 2, 4, 8, 12, 16, 20]

    def table(title, fn, absolute_mean):
        print()
        print(title)
        last = "mean|.|" if absolute_mean else "mean"
        print("  " + f"{'estimator':<18}" + "".join(f"{'h=' + str(h):>8}" for h in show) + f"{last:>9}")
        for k in est_names:
            v = fn(est[k])
            m = np.mean(np.abs(v)) if absolute_mean else np.mean(v)
            print("  " + f"{k:<18}" + "".join(f"{v[h]:>8.3f}" for h in show) + f"{m:>9.3f}")

    table("BIAS  (mean estimate - truth)", lambda a: a.mean(axis=0) - truth, True)
    table("SD    (standard deviation across replications)", lambda a: a.std(axis=0, ddof=1), False)
    table("RMSE  (sqrt of bias^2 + SD^2)",
          lambda a: np.sqrt(np.mean((a - truth) ** 2, axis=0)), False)

    print()
    print("RMSE winner by horizon (lowest RMSE among the eight estimators):")
    rmse = {k: np.sqrt(np.mean((est[k] - truth) ** 2, axis=0)) for k in est_names}
    for h in range(H + 1):
        best = min(est_names, key=lambda k: rmse[k][h])
        print(f"  h={h:>2}: {best:<18} RMSE={rmse[best][h]:.3f}"
              f"   (LP(4) lag-aug {rmse['LP(4) lag-aug'][h]:.3f}, VAR(4) {rmse['VAR(4)'][h]:.3f},"
              f" VAR(12) {rmse['VAR(12)'][h]:.3f})")

    print()
    print("COVERAGE of nominal 95% pointwise bands (share of replications containing the truth)")
    print("  " + f"{'band':<24}" + "".join(f"{'h=' + str(h):>8}" for h in show) + f"{'mean':>9}")
    for k in cov_names:
        c = cov[k].mean(axis=0)
        print("  " + f"{k:<24}" + "".join(f"{c[h]:>8.3f}" for h in show) + f"{c.mean():>9.3f}")
    se_mc = np.sqrt(0.95 * 0.05 / reps)
    print(f"  Monte Carlo SE of a 0.95 coverage estimate with {reps} replications: {se_mc:.3f}")
    print(f"  smooth_lp lambda_used: median {np.median(lam_used):.3g}, "
          f"IQR [{np.percentile(lam_used, 25):.3g}, {np.percentile(lam_used, 75):.3g}]; "
          f"max |irf_raw - lp(se='hac') irf| over all replications: {raw_gap:.1e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=500, help="Monte Carlo replications (default 500)")
    args = ap.parse_args()

    header("PROVENANCE")
    print(f"  date          : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  python        : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  platform      : {platform.platform()}  cpu_count={os.cpu_count()}")
    print(f"  cpu model     : {cpu_model()}")
    print(f"  tsecon        : {tsecon.__version__}   build: {build_mode()}")
    print(f"  numpy / scipy : {np.__version__} / {scipy.__version__}")

    t0 = time.perf_counter()
    part1()
    part2(reps=args.reps)
    print()
    print(f"Total wall-clock: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
