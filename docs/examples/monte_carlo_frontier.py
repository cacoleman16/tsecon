"""Frontier Monte Carlo experiments.

Where `monte_carlo.py` verifies that individual estimators have the properties
they claim (size, coverage, consistency), this file asks the harder, *comparative*
questions the applied literature actually argues about:

  F1. Local projections vs VAR for impulse responses — the bias/variance
      trade-off (Plagborg-Møller & Wolf 2021; Li, Plagborg-Møller & Wolf 2024).
  F2. LP-IV with a weak instrument — does the confidence interval still cover?

Both are self-contained: a known DGP, an analytically computable truth, and a
seeded simulation. Run:

    .venv/bin/python docs/examples/monte_carlo_frontier.py
"""
import sys
import time

import numpy as np

import tsecon

SEED = 20260718


def rule(width=72, ch="-"):
    print(ch * width)


def header(title):
    print()
    rule()
    print(title)
    rule()


# --------------------------------------------------------------------------- #
# F1. Local projections vs VAR: bias, variance, and lag misspecification
# --------------------------------------------------------------------------- #
# DGP — an observed exogenous shock z entering an AR(2) outcome:
#
#   z_t ~ N(0, 1)                                    (exogenous, observed)
#   y_t = a1 y_{t-1} + a2 y_{t-2} + b0 z_t + b1 z_{t-1} + u_t
#
# Both estimators target the same object (the response of y to a unit z shock),
# so the comparison is apples to apples — that equivalence is exactly
# Plagborg-Møller & Wolf's (2021) point.
A1, A2 = 0.6, -0.15
B0, B1 = 1.0, 0.5


def true_irf(horizons):
    """Analytic response of y to a unit z shock, by iterating the recursion."""
    h_max = horizons
    th = np.zeros(h_max + 1)
    for h in range(h_max + 1):
        val = B0 if h == 0 else 0.0
        if h == 1:
            val += B1
        if h >= 1:
            val += A1 * th[h - 1]
        if h >= 2:
            val += A2 * th[h - 2]
        th[h] = val
    return th


def simulate(rng, T, burn=200):
    n = T + burn
    z = rng.standard_normal(n)
    u = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = A1 * y[t - 1] + A2 * y[t - 2] + B0 * z[t] + B1 * z[t - 1] + u[t]
    return z[burn:], y[burn:]


def experiment_lp_vs_var(reps=400, T=240, horizons=12):
    rng = np.random.default_rng(SEED)
    truth = true_irf(horizons)

    # Two specifications: correct lag order, and a truncated (misspecified) one.
    specs = {"correct (2 lags)": 2, "truncated (1 lag)": 1}
    est = {
        (name, method): np.full((reps, horizons + 1), np.nan)
        for name in specs
        for method in ("LP", "VAR")
    }

    for r in range(reps):
        z, y = simulate(rng, T)
        data = np.column_stack([z, y])  # z ordered first => its own shock
        for name, p in specs.items():
            lp = tsecon.lp(y, z, horizons=horizons, n_lag_controls=p)
            est[(name, "LP")][r] = np.asarray(lp["irf"])
            irf = np.asarray(tsecon.var_irf(data, lags=p, horizon=horizons, orth=True))
            est[(name, "VAR")][r] = irf[:, 1, 0]  # response of y to a z shock

    header("F1. Local projections vs VAR — bias / variance / RMSE")
    print(f"reps={reps}, T={T}, DGP: y_t = {A1} y_(t-1) {A2:+} y_(t-2) "
          f"+ {B0} z_t + {B1} z_(t-1) + u_t")
    print("both estimators target the same unit-z-shock response (PM&W 2021)")

    for name in specs:
        print(f"\n  specification: {name}")
        print(f"    {'h':>3} | {'truth':>8} | {'LP bias':>9} {'LP sd':>8} {'LP rmse':>8}"
              f" | {'VAR bias':>9} {'VAR sd':>8} {'VAR rmse':>8}")
        print("    " + "-" * 76)
        for h in (0, 1, 2, 4, 8, 12):
            row = [f"    {h:>3} | {truth[h]:>8.4f} |"]
            for method in ("LP", "VAR"):
                e = est[(name, method)][:, h]
                bias = np.mean(e) - truth[h]
                sd = np.std(e, ddof=1)
                rmse = np.sqrt(np.mean((e - truth[h]) ** 2))
                row.append(f" {bias:>9.4f} {sd:>8.4f} {rmse:>8.4f} |")
            print("".join(row))

    # Headline comparison: average |bias| and RMSE over horizons 1..H.
    print("\n  averages over h = 1..%d:" % horizons)
    print(f"    {'specification':<20} {'|bias| LP':>10} {'|bias| VAR':>11}"
          f" {'rmse LP':>9} {'rmse VAR':>10}")
    for name in specs:
        stats = {}
        for method in ("LP", "VAR"):
            e = est[(name, method)][:, 1:]
            stats[method] = (
                np.mean(np.abs(np.mean(e, axis=0) - truth[1:])),
                np.mean(np.sqrt(np.mean((e - truth[1:]) ** 2, axis=0))),
            )
        print(f"    {name:<20} {stats['LP'][0]:>10.4f} {stats['VAR'][0]:>11.4f}"
              f" {stats['LP'][1]:>9.4f} {stats['VAR'][1]:>10.4f}")
    print("\n  expected: under CORRECT specification the VAR is more efficient")
    print("            (lower sd/rmse) while both are near-unbiased; under lag")
    print("            TRUNCATION the VAR's bias grows with the horizon while")
    print("            LP stays comparatively robust — the bias/variance trade-off.")


# --------------------------------------------------------------------------- #
# F2. LP-IV with a weak instrument: does the interval still cover?
# --------------------------------------------------------------------------- #
def experiment_weak_iv(reps=500, T=300, horizons=4):
    """Vary instrument strength; track first-stage F and CI coverage."""
    rng = np.random.default_rng(SEED + 1)
    theta = 1.0        # true impact effect of the impulse on y
    rho_uv = 0.7       # endogeneity: impulse innovation correlated with y's error

    header("F2. LP-IV with a weak instrument — coverage vs first-stage F")
    print(f"reps={reps}, T={T}, true impact effect = {theta}, "
          f"corr(impulse error, outcome error) = {rho_uv}")
    print(f"    {'pi':>6} | {'mean F':>8} | {'coverage h=0':>13} | {'median beta':>12}")
    print("    " + "-" * 52)

    for pi in (0.05, 0.10, 0.20, 0.50):
        covered, fstats, betas = 0, [], []
        for _ in range(reps):
            z = rng.standard_normal(T)                 # the instrument
            v = rng.standard_normal(T)
            x = pi * z + v                             # endogenous impulse
            u = rho_uv * v + np.sqrt(1 - rho_uv ** 2) * rng.standard_normal(T)
            y = theta * x + u                          # outcome

            try:
                fit = tsecon.lp_iv(y, x, z, horizons=horizons, n_lag_controls=1)
            except Exception:
                continue
            b = float(np.asarray(fit["irf"])[0])
            s = float(np.asarray(fit["se"])[0])
            f = fit.get("first_stage_f", fit.get("effective_f", np.nan))
            fstats.append(float(np.asarray(f).ravel()[0]) if f is not None else np.nan)
            betas.append(b)
            if s > 0 and abs(b - theta) <= 1.96 * s:
                covered += 1

        n = max(len(betas), 1)
        print(f"    {pi:>6.2f} | {np.nanmean(fstats):>8.2f} | {covered / n:>13.3f} |"
              f" {np.median(betas):>12.4f}")

    print("\n  expected: as the instrument weakens (small first-stage F) the point")
    print("            estimate is pulled toward the biased OLS value. Coverage")
    print("            of the nominal 95% interval degrades too, but far LESS than")
    print("            the point estimate does — the weak-instrument standard")
    print("            errors inflate, which partly self-corrects the interval.")
    print("            So a small F is primarily a POINT-ESTIMATE warning here:")
    print("            read the median beta column, not just the coverage column.")


def main():
    t0 = time.time()
    print("tsecon frontier Monte Carlo — comparative experiments")
    print("expected vs observed, every table reproducible from a fixed seed")
    experiment_lp_vs_var()
    experiment_weak_iv()
    print()
    rule()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    only = sys.argv[1:]
    if not only:
        main()
    else:
        if any("lp" in a or "var" in a for a in only):
            experiment_lp_vs_var()
        if any("iv" in a or "weak" in a for a in only):
            experiment_weak_iv()
