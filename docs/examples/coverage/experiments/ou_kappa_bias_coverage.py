"""ou_fit: finite-sample kappa bias, and log- vs level-scale half-life CI coverage.

    .venv/bin/python docs/examples/coverage/experiments/ou_kappa_bias_coverage.py           # full, ~2 min
    .venv/bin/python docs/examples/coverage/experiments/ou_kappa_bias_coverage.py --quick   # smoke, ~10 s

WHAT IS BEING MEASURED
----------------------
1. **The well-known upward bias of the OU mean-reversion MLE.** The AR(1)
   slope is biased down by ~ -(1+3 phi)/n (Kendall 1954); through
   kappa = -ln(phi)/dt that is an upward kappa bias of ~ (1+3 phi)/(n phi dt)
   — approximately 4 / (time span) for persistent spreads (Tang & Chen 2009;
   Yu 2012). ou_fit ships the unadjusted MLE (it is what the closed form and
   the statsmodels golden pin), so the bias must be *quantified*, not hidden:
   this script measures mean bias and RMSE of kappa_hat on a seeded grid and
   compares the measured bias with the first-order prediction.

2. **Which half-life CI construction to ship.** half_life = ln2/kappa is a
   monotone map, so the coverage of a mapped kappa interval equals the
   coverage of the kappa interval itself. Two candidates:

   level scale  kappa -+ z SE(kappa), upper half-life bound = +inf when the
                interval crosses zero                            (SHIPPED)
   log scale    kappa * exp(-+ z SE(kappa)/kappa)   (measured, not shipped)

   The a-priori argument favors the log scale (positive by construction,
   matches the right skew of ln2/kappa_hat) — and this measurement is why
   ou_fit ships the LEVEL scale instead: kappa_hat centers above the truth
   (the bias in part 1), a multiplicative interval around an upward-biased
   center never reaches down to a small true kappa, while the level interval
   — precisely by crossing zero and conceding "maybe no mean reversion"
   (its +inf branch) — does. Measured at 2000 reps the level scale covers
   closer to nominal in every cell (daily 5y at kappa 5/2/0.5/0.1:
   0.94/0.91/0.82/0.71 vs 0.89/0.80/0.53/0.21). The shipped coverage is
   measured from ou_fit's own half_life_ci surface, the log scale from
   kappa/kappa_se. Reps whose fit lands at phi_hat >= 1
   (mean_reverting=False: ou_fit honestly returns no CI) are counted as
   NON-covering for both constructions, so neither is flattered by dropping
   its failures; that fraction is reported too, as is the fraction of
   shipped intervals whose upper endpoint is +inf.

DGP: exact-discretization OU, mu = 0, sigma = 0.2, x0 = 0 (stationary mean),
seeded per cell. Cells cross kappa in {5, 2, 0.5, 0.1} with a daily grid
(dt = 1/252, T = 1260: a 5-year span) and a monthly grid (dt = 1/12,
T = 240: a 20-year span). kappa = 0.1 on a 5-year span is the deliberate
stress cell: the half-life (~6.9y) exceeds the span.

Results (2000 reps, seed base 20260826) are recorded in the cointegration
model card next to ou_fit.
"""
import sys
import time

import numpy as np
import tsecon

QUICK = "--quick" in sys.argv
REPS = 200 if QUICK else 2000
Z = 1.959963984540054  # Phi^{-1}(0.975)

CELLS = [
    # (kappa, dt, T, label)
    (5.0, 1 / 252, 1260, "daily 5y"),
    (2.0, 1 / 252, 1260, "daily 5y"),
    (0.5, 1 / 252, 1260, "daily 5y"),
    (0.1, 1 / 252, 1260, "daily 5y"),
    (5.0, 1 / 12, 240, "monthly 20y"),
    (2.0, 1 / 12, 240, "monthly 20y"),
    (0.5, 1 / 12, 240, "monthly 20y"),
    (0.1, 1 / 12, 240, "monthly 20y"),
]
MU, SIGMA = 0.0, 0.2


def simulate_paths(rng, kappa, dt, n, reps):
    """`reps` exact-discretization OU paths, vectorized across reps."""
    phi = np.exp(-kappa * dt)
    c = MU * (1.0 - phi)
    eta = np.sqrt(SIGMA**2 * (1.0 - phi**2) / (2.0 * kappa))
    x = np.empty((reps, n))
    x[:, 0] = MU
    shocks = rng.standard_normal((reps, n - 1))
    for t in range(1, n):
        x[:, t] = c + phi * x[:, t - 1] + eta * shocks[:, t - 1]
    return x


def run():
    print(f"reps = {REPS}, nominal level = 0.95, DGP mu={MU}, sigma={SIGMA}")
    print(
        f"{'cell':>12} {'kappa':>6} {'span':>6} | {'bias':>8} {'pred':>8} {'RMSE':>8} | "
        f"{'shipped':>8} {'cov log':>8} {'hi=inf':>6} {'no-CI':>6}"
    )
    for i, (kappa, dt, n, label) in enumerate(CELLS):
        rng = np.random.default_rng(20260826 + i)
        x = simulate_paths(rng, kappa, dt, n, REPS)
        hl_true = np.log(2.0) / kappa
        khat = np.full(REPS, np.nan)
        cover_shipped = np.zeros(REPS, dtype=bool)
        cover_log = np.zeros(REPS, dtype=bool)
        hi_inf = 0
        no_ci = 0
        for r in range(REPS):
            fit = tsecon.ou_fit(x[r], dt=dt)  # closed form; refusals impossible here
            khat[r] = fit["kappa"]
            if not fit["mean_reverting"]:
                no_ci += 1  # counted as non-covering for BOTH constructions
                continue
            # shipped: ou_fit's own half_life_ci (level-scale kappa interval
            # mapped through ln2/kappa; upper endpoint +inf when it crosses 0)
            lo, hi = fit["half_life_ci"]
            cover_shipped[r] = lo <= hl_true <= hi
            if np.isinf(hi):
                hi_inf += 1
            # log scale (the rejected alternative, measured honestly)
            k, se = fit["kappa"], fit["kappa_se"]
            cover_log[r] = k * np.exp(-Z * se / k) <= kappa <= k * np.exp(Z * se / k)
        phi_true = np.exp(-kappa * dt)
        pred = (1.0 + 3.0 * phi_true) / ((n - 1) * phi_true * dt)  # Kendall mapped
        bias = np.nanmean(khat) - kappa
        rmse = np.sqrt(np.nanmean((khat - kappa) ** 2))
        span = n * dt
        print(
            f"{label:>12} {kappa:>6.2f} {span:>5.1f}y | {bias:>8.4f} {pred:>8.4f} "
            f"{rmse:>8.4f} | {cover_shipped.mean():>8.3f} {cover_log.mean():>8.3f} "
            f"{hi_inf / REPS:>6.3f} {no_ci / REPS:>6.3f}"
        )


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"[{time.time() - t0:.1f}s]")
