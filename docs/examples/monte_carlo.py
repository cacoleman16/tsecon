"""Monte Carlo validation suite: does tsecon get the statistics right?

Run with the project venv (tsecon + numpy + scipy installed there):
    .venv/bin/python docs/examples/monte_carlo.py

Every table below is produced by a seeded simulation, so the numbers are
reproducible bit-for-bit. Each experiment states an *expected* value from
theory and reports the *observed* value from the simulation, so you can read
the two side by side and check that the estimator behaves the way its
asymptotics promise. Nothing here is a plot -- this is the correctness story
told in numbers, and it mirrors docs/examples/monte-carlo.md.

Three properties are checked:

  1. SIZE & POWER of the IVX predictive-regression test vs a naive OLS t-test,
     under a persistent, endogenous regressor (predictive_regression).
  2. COVERAGE of HAC / Newey-West confidence intervals for the mean of a
     serially-correlated series vs naive IID standard errors (ols).
  3. CONSISTENCY of the AR(1) slope estimator: bias and RMSE shrink at the
     rates theory predicts as the sample grows (var_fit).

Total runtime is well under two minutes on a laptop.
"""
import time

import numpy as np
from scipy.stats import norm

import tsecon

Z95 = float(norm.ppf(0.975))  # 1.959964..., the two-sided 5% normal cutoff


# --------------------------------------------------------------------------
# small table helpers -- keep every experiment's output aligned and readable
# --------------------------------------------------------------------------
def rule(width=64):
    print("-" * width)


def header(title):
    print()
    rule()
    print(title)
    rule()


def ar1(rng, n, phi, burn=100):
    """One draw of a mean-zero Gaussian AR(1): y_t = phi y_{t-1} + e_t.

    A burn-in of `burn` observations is discarded so the returned path is
    (to machine tolerance) drawn from the stationary distribution rather than
    from the y_0 = 0 start value.
    """
    e = rng.standard_normal(n + burn)
    y = np.zeros(n + burn)
    for t in range(1, n + burn):
        y[t] = phi * y[t - 1] + e[t]
    return y[burn:]


# --------------------------------------------------------------------------
# Experiment 1 -- IVX predictive regression: size at a true null, then power
# --------------------------------------------------------------------------
# DGP (Kostakis-Magdalinos-Stamatogiannis 2015 setup):
#     x_t = rho * x_{t-1} + u_t                (persistent predictor)
#     r_t = beta * x_{t-1} + e_t               (predictive regression)
#     corr(u_t, e_t) = delta  (strongly negative -> "Stambaugh" endogeneity)
#
# When beta = 0 the null of no predictability is TRUE, so a correctly sized
# 5% test should reject 5% of the time. The naive OLS t-test does not: as rho
# approaches a unit root the endogeneity inflates its rejection rate far above
# nominal. The IVX Wald test stays close to 5% for every rho -- that is its
# whole reason for existing. Then, at rho = 0.95, we switch the null off
# (beta > 0) to confirm IVX still has power to detect real predictability.
def _predictive_reject(rng, reps, T, rho, beta, delta):
    rej_ols = 0
    rej_ivx = 0
    root = 1.0 - delta * delta
    for _ in range(reps):
        z = rng.standard_normal((T, 2))
        u = z[:, 0]
        e = delta * z[:, 0] + np.sqrt(root) * z[:, 1]
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = rho * x[t - 1] + u[t]
        r = np.empty(T)
        r[0] = e[0]
        for t in range(1, T):
            r[t] = beta * x[t - 1] + e[t]
        res = tsecon.predictive_regression(r[1:], x[:-1])
        if abs(res["ols"]["tstat"]) > Z95:
            rej_ols += 1
        if res["ivx"]["pvalue"] < 0.05:
            rej_ivx += 1
    return rej_ols / reps, rej_ivx / reps


def experiment_ivx(reps=2000, T=250, delta=-0.95):
    header("1. IVX predictive regression -- size (true null) and power")
    print(f"reps={reps}, T={T}, endogeneity corr(u,e)={delta}, nominal level=0.05")
    print()

    # --- size: beta = 0, sweep persistence ---
    print("SIZE  (beta = 0, so 0.05 is the target rejection rate)")
    print(f"{'rho':>6} | {'OLS t-test':>12} | {'IVX Wald':>10}")
    rule(34)
    rng = np.random.default_rng(777)
    for rho in (0.90, 0.95, 0.99, 1.00):
        ols, ivx = _predictive_reject(rng, reps, T, rho, beta=0.0, delta=delta)
        print(f"{rho:>6.2f} | {ols:>12.3f} | {ivx:>10.3f}")
    print("expected: IVX ~ 0.05 for every rho; OLS blows up as rho -> 1")

    # --- power: rho fixed near unit root, sweep beta ---
    print()
    print("POWER (rho = 0.95; higher rejection is better once beta > 0)")
    print(f"{'beta':>6} | {'OLS t-test':>12} | {'IVX Wald':>10}")
    rule(34)
    rng = np.random.default_rng(999)
    for beta in (0.00, 0.05, 0.10):
        ols, ivx = _predictive_reject(rng, reps, T, 0.95, beta=beta, delta=delta)
        print(f"{beta:>6.2f} | {ols:>12.3f} | {ivx:>10.3f}")
    print("expected: rejection climbs toward 1.0 as beta grows -> IVX has power")


# --------------------------------------------------------------------------
# Experiment 2 -- HAC / Newey-West confidence-interval coverage
# --------------------------------------------------------------------------
# Estimate the mean of a serially-correlated AR(1) series by regressing it on a
# constant, and build a 95% CI as mu_hat +/- 1.96 * se. With IID (nonrobust)
# standard errors the CI ignores the autocorrelation and is far too narrow, so
# it covers the true mean (0) much less than 95% of the time. HAC / Newey-West
# standard errors widen the interval to account for the long-run variance and
# recover most of the nominal coverage. When phi = 0 there is no autocorrelation
# and HAC costs essentially nothing -- both methods sit at ~95%.
def experiment_hac(reps=2000, T=200):
    header("2. HAC vs IID confidence-interval coverage for a mean")
    print(f"reps={reps}, T={T}, true mean=0, nominal coverage=0.95")
    print()
    print(f"{'phi':>6} | {'IID cover':>10} | {'HAC cover':>10} | "
          f"{'IID width':>10} | {'HAC width':>10}")
    rule(60)
    X = np.ones((T, 1))
    rng = np.random.default_rng(4242)
    for phi in (0.00, 0.50, 0.80, 0.95):
        cov_iid = cov_hac = 0
        w_iid = w_hac = 0.0
        for _ in range(reps):
            y = ar1(rng, T, phi)
            ri = tsecon.ols(y, X, se_type="nonrobust")
            rh = tsecon.ols(y, X, se_type="hac")
            bi, si = ri["params"][0], ri["bse"][0]
            bh, sh = rh["params"][0], rh["bse"][0]
            if abs(bi) <= Z95 * si:
                cov_iid += 1
            if abs(bh) <= Z95 * sh:
                cov_hac += 1
            w_iid += 2 * Z95 * si
            w_hac += 2 * Z95 * sh
        print(f"{phi:>6.2f} | {cov_iid / reps:>10.3f} | {cov_hac / reps:>10.3f} | "
              f"{w_iid / reps:>10.3f} | {w_hac / reps:>10.3f}")
    print("expected: IID coverage collapses as phi grows; HAC stays near 0.95,")
    print("          and at phi=0 HAC matches IID (no cost when uncorrelated)")


# --------------------------------------------------------------------------
# Experiment 3 -- consistency of the AR(1) slope estimator
# --------------------------------------------------------------------------
# Fit an AR(1) by OLS (via var_fit) and track the estimated slope as T grows.
# The estimator is downward biased in finite samples -- Kendall's classic result
# gives bias ~ -(1 + 3*phi) / T -- but both the bias and the RMSE shrink toward
# zero as T increases. We report the theoretical bias alongside the observed
# one, and confirm RMSE falls at the sqrt(T) rate: quadrupling T roughly halves
# the RMSE.
def experiment_consistency(reps=1000, phi=0.7):
    header("3. AR(1) slope consistency -- bias and RMSE shrink with T")
    print(f"reps={reps}, true phi={phi}, estimator=OLS via var_fit")
    print(f"theoretical finite-sample bias ~ -(1 + 3*phi)/T = {-(1 + 3 * phi):.1f}/T")
    print()
    print(f"{'T':>6} | {'mean phi':>10} | {'bias':>9} | {'bias~':>9} | "
          f"{'RMSE':>9}")
    rule(56)
    prev_T = None
    prev_rmse = None
    for T in (100, 400, 1600, 6400):
        rng = np.random.default_rng(24680)
        est = np.empty(reps)
        for i in range(reps):
            y = ar1(rng, T, phi).reshape(-1, 1)
            res = tsecon.var_fit(y, lags=1, trend="c")
            est[i] = res["params"][1][0]
        bias = est.mean() - phi
        rmse = float(np.sqrt(np.mean((est - phi) ** 2)))
        bias_pred = -(1 + 3 * phi) / T
        ratio = "" if prev_rmse is None else f"  (x{prev_rmse / rmse:.2f} vs T={prev_T})"
        print(f"{T:>6} | {est.mean():>10.4f} | {bias:>+9.4f} | {bias_pred:>+9.4f} | "
              f"{rmse:>9.4f}{ratio}")
        prev_T, prev_rmse = T, rmse
    print("expected: bias -> 0 and tracks -(1+3phi)/T; RMSE ~ halves per 4x in T")


def main():
    print("tsecon Monte Carlo validation suite")
    print("expected vs observed, every table reproducible from a fixed seed")
    t0 = time.time()
    experiment_ivx()
    experiment_hac()
    experiment_consistency()
    print()
    rule()
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
