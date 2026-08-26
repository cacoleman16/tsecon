"""Do tsecon's regression standard errors deliver their nominal coverage?

    .venv/bin/python docs/examples/coverage/regression_se.py            # full run, ~1 min
    .venv/bin/python docs/examples/coverage/regression_se.py --quick    # smoke run, ~6 s

A 95% confidence interval is a promise about repeated samples: across
independent draws from the same data-generating process, the interval should
contain the *true* coefficient 95% of the time. `tsecon` proves its point
estimates and its standard-error *algebra* against independent references
elsewhere. This module tests the promise itself, for the standard-error
machinery attached to regression coefficients:

  * `ols`                  se_type = nonrobust / hc0 / hc1 / hc2 / hc3 / hac
  * `iv_gmm`               2sls / 2step / iterated, weight = robust / hac
  * `har_rv`               Bartlett-HAC SEs on the Corsi (2009) HAR
  * `recession_probit`     inverse-observed-information (Wald) SEs
  * `quantile_regression`  Powell kernel-sandwich SEs, per tau

Every number below is a measurement. Where coverage falls short it is printed
as-is with the diagnostics needed to say *why*. Three columns do that work:

  cover     share of replications whose interval contained the truth
  se/sd     reported SE divided by the Monte Carlo standard deviation of the
            point estimate. Below 1 means the reported SE is too small. Where
            the SE distribution is itself skewed, the MEDIAN ratio is reported
            next to the mean -- experiment 4a is a case where the mean ratio
            exceeds 1 while the typical interval is less than half wide enough.
  bias      mean (or median) point estimate minus truth. If |bias| is a large
            fraction of sd, the interval is centred in the wrong place and a
            correct SE cannot save it.

A third failure mode exists that neither column catches on its own: the
sampling distribution can simply not be normal, in which case no rescaling or
re-centring of a Wald interval works. Experiments 2 and 4a both test for that
directly by building the interval from the TRUE sampling variability (the
`oracle` column) and asking whether even that covers at nominal.

Reading those three together separates the two things this suite is careful
never to conflate:

  * the ESTIMATOR is wrong for the job -- a nonrobust SE under
    heteroskedasticity is *inconsistent*, so its coverage does not improve
    with T. Experiment 1 row 2 and Experiment 2's `nonrobust` column show a
    coverage that is stuck near 0.45 no matter how much data arrives.
  * the APPROXIMATION is what it is -- a Newey-West SE with a bandwidth of 4
    cannot capture a score autocorrelation of 0.90, and no bug fix changes
    that. Experiment 3 quantifies the shortfall instead of hiding it.

Experiment 2 pins that distinction down with an ORACLE column: the same
sandwich formula evaluated at the *true* per-observation error variances.
Because the errors there are Gaussian and the design is fixed within a
replication, the OLS coefficient is *exactly* normal with the oracle
variance, so the oracle interval covers exactly 95% at every sample size.
Any shortfall in the `hc0` / `hc1` columns is therefore attributable to the
variance ESTIMATE alone, not to the normal approximation. That is the
cleanest decomposition in this file.

Known-truth DGPs
----------------
Every design has a closed-form true coefficient vector.

`OLS`         y = 1 + 2x + e, with four error structures: iid Gaussian;
              heteroskedastic with sd(e|x) = |x|; AR(1) errors (phi = 0.7)
              alongside an AR(1) regressor; and iid t(3) errors, which have
              finite variance but INFINITE fourth moment -- the case where
              the sandwich estimator's own asymptotics wobble.
`LEVERAGE`    y = 1 + 2x + e with x ~ chi2(1) (right-skewed, so a handful of
              observations carry most of the leverage) and sd(e|x) = x. A
              deliberate stress test of the HC family in small samples.
`HAC-SLOPE`   y = 1 + x + e with x and e both AR(1) with the same phi, so the
              score x_t e_t has autocorrelation ~ phi^2. The existing
              `docs/examples/monte_carlo.py` experiment 2 covers HAC for a
              MEAN; this is the slope, and it is much harder.
`IV`          y = 1 + x + e, x = pi (z1 + z2) + v, corr(e, v) = 0.7. Sweeping
              pi walks the median first-stage F from ~90 (strong) through ~10
              (the conventional rule-of-thumb threshold) to ~1 (weak).
              Over-identified by one degree of freedom, so the Hansen J test
              has a true null to be sized against.
`HAR`         log RV follows the HAR recursion exactly:
              h_t = c + b_d h_{t-1} + b_w mean(h[t-5:t])
                      + b_m mean(h[t-22:t]) + e_t,
              with (c, b_d, b_w, b_m) = (-0.20, 0.35, 0.35, 0.25). Feeding
              exp(h) to `har_rv(variant="log")` makes those four numbers the
              exact truth. The window conventions are verified against a
              hand-built design in `structural_checks()`.
`PROBIT`      y_t = 1{b0 + b1 x_t + u_t > 0} with x_t AR(1) (phi = 0.9), in a
              common regime (event rate ~0.46) and a rare regime (rate
              ~0.06), the latter being the realistic recession-dating case.
`QUANTILE`    y = a + b x + (s0 + s1 x) u with u standard normal and
              x ~ U(0, 2). The conditional tau-quantile is exactly linear:
              intercept a + s0 z_tau, slope b + s1 z_tau. With s1 = 0 the
              slope is the same at every tau; with s1 > 0 it fans out.

Every table is a deterministic function of SEED. Monte Carlo standard errors
are printed next to every coverage number as `cover+-mcse`, with
mcse = sqrt(p (1 - p) / reps), so a reader can tell 0.93 from 0.95 honestly.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.stats import norm

import tsecon

# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
SEED = 20260729  # every table in this file is a deterministic function of this
NOMINAL = 0.95
Z = float(norm.ppf(0.5 + NOMINAL / 2.0))  # 1.959964..., two-sided 5% cutoff
REPS_FULL = 3000
REPS_QUICK = 250

# `ols` accepts exactly these, lower-case (see the ValueError it raises).
# hc2/hc3 were added after this suite first measured what their absence cost;
# experiment 2 now reports tsecon's own hc2/hc3 coverage rather than a
# reference computed here.
OLS_SE_TYPES = ("nonrobust", "hc0", "hc1", "hc2", "hc3", "hac")


# --------------------------------------------------------------------------
# small numerics helpers
# --------------------------------------------------------------------------
def mc_se(p_hat, reps):
    """Monte Carlo standard error of a coverage estimate."""
    return float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / reps))


def ar1_paths(rng, reps, n, phi, burn=200):
    """(reps, n) draws of a mean-zero Gaussian AR(1) at its stationary law.

    The first value is drawn from the stationary distribution and a further
    `burn` observations are discarded, so the returned block is stationary to
    machine tolerance rather than starting from y_0 = 0.
    """
    e = rng.standard_normal((reps, n + burn))
    if phi == 0.0:
        return e[:, burn:]
    y = np.empty_like(e)
    y[:, 0] = e[:, 0] / np.sqrt(1.0 - phi * phi)
    for t in range(1, n + burn):
        y[:, t] = phi * y[:, t - 1] + e[:, t]
    return y[:, burn:]


def covered(point, se, truth):
    """Does `point +- Z se` contain `truth`? NaN / non-finite SE never covers."""
    if not np.isfinite(se) or not np.isfinite(point):
        return False
    return bool(abs(point - truth) <= Z * se)


def sandwich_se(x, weights):
    """sqrt of the diagonal of (X'X)^-1 X' diag(w) X (X'X)^-1."""
    xtxi = np.linalg.inv(x.T @ x)
    mid = (x * weights[:, None]).T @ x
    return np.sqrt(np.diag(xtxi @ mid @ xtxi))


def leverage(x, xtxi):
    """Diagonal of the hat matrix, h_i = x_i' (X'X)^-1 x_i."""
    return np.einsum("ij,jk,ik->i", x, xtxi, x)


# --------------------------------------------------------------------------
# printing helpers
# --------------------------------------------------------------------------
def rule(width=104, char="-"):
    print(char * width)


def header(title):
    print()
    rule()
    print(title)
    rule()


def cov_cell(p, reps, width=13):
    """`0.943+-0.004` -- coverage with its own Monte Carlo standard error."""
    if not np.isfinite(p):
        return f"{'n/a':>{width}}"
    return f"{p:.3f}+-{mc_se(p, reps):.3f}".rjust(width)


def cov_pair(p, reps):
    """The (coverage, mcse) pair that every experiment stores in its result."""
    return {"cover": float(p), "mcse": mc_se(p, reps)}


# ==========================================================================
# Experiment 1 -- ols: se_type against four error structures, on the SLOPE
# ==========================================================================
# The right and wrong tool for four jobs. `nonrobust` is only consistent for
# the iid row; `hc0`/`hc1` add heteroskedasticity-robustness but do nothing
# about serial correlation; `hac` adds both but pays a bandwidth cost. The
# question is not "which is best" but "how much does the wrong choice cost,
# and does the right choice actually repair it".
OLS_DGPS = (
    ("iid Gaussian", "iid"),
    ("heteroskedastic sd=|x|", "het"),
    ("AR(1) errors+regressor .7", "ar1"),
    ("t(3) errors (inf kurtosis)", "t3"),
)

OLS_COLUMNS = (
    ("nonrobust", {"se_type": "nonrobust"}),
    ("hc0", {"se_type": "hc0"}),
    ("hc1", {"se_type": "hc1"}),
    ("hac auto", {"se_type": "hac"}),
    ("hac lag8", {"se_type": "hac", "maxlags": 8}),
)


def _ols_draw(rng, kind, reps, n):
    """Return (x block, e block) for one of the four OLS error structures."""
    if kind == "ar1":
        x = ar1_paths(rng, reps, n, 0.7)
        e = ar1_paths(rng, reps, n, 0.7)
        return x, e
    x = rng.standard_normal((reps, n))
    if kind == "iid":
        e = rng.standard_normal((reps, n))
    elif kind == "het":
        # sd(e | x) = |x|: the large-|x| observations, which dominate the
        # slope, are exactly the noisy ones. This is what breaks `nonrobust`.
        e = np.abs(x) * rng.standard_normal((reps, n))
    elif kind == "t3":
        # standardised to unit variance; t(3) has finite variance but its
        # fourth moment does not exist, which is the sandwich's problem.
        e = rng.standard_t(3, size=(reps, n)) / np.sqrt(3.0)
    else:  # pragma: no cover -- guarded by OLS_DGPS
        raise ValueError(kind)
    return x, e


def exp_ols_se_types(reps, n=200):
    beta = np.array([1.0, 2.0])
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for label, kind in OLS_DGPS:
        x, e = _ols_draw(rng, kind, reps, n)
        hits = {c: 0 for c, _ in OLS_COLUMNS}
        se_sum = {c: 0.0 for c, _ in OLS_COLUMNS}
        est = np.empty(reps)
        for i in range(reps):
            design = np.column_stack([np.ones(n), x[i]])
            y = design @ beta + e[i]
            for name, kw in OLS_COLUMNS:
                res = tsecon.ols(y, design, **kw)
                point, se = res["params"][1], res["bse"][1]
                hits[name] += covered(point, se, beta[1])
                se_sum[name] += se
            est[i] = point  # identical across se_types; the SE is what varies
        sd = float(est.std(ddof=1))
        row = {
            "dgp": label,
            "kind": kind,
            "mc_sd": sd,
            "bias": float(est.mean() - beta[1]),
            "cover": {},
            "se_over_sd": {},
            "mean_width": {},
        }
        for name, _ in OLS_COLUMNS:
            row["cover"][name] = cov_pair(hits[name] / reps, reps)
            row["se_over_sd"][name] = (se_sum[name] / reps) / sd
            row["mean_width"][name] = 2.0 * Z * se_sum[name] / reps
        rows.append(row)
    return {"name": "ols se_type", "reps": reps, "n": n, "truth": beta[1],
            "columns": [c for c, _ in OLS_COLUMNS], "rows": rows}


def report_ols_se_types(res):
    header(f"1. ols -- 95% CI coverage for the SLOPE, T={res['n']}, "
           f"reps={res['reps']}, truth={res['truth']:g}")
    print("Which se_type is CONSISTENT for which error structure decides each row.")
    print()
    print(f"{'error structure':<28}" + "".join(f"{c:>13}" for c in res["columns"]))
    rule(93)
    for row in res["rows"]:
        print(f"{row['dgp']:<28}"
              + "".join(cov_cell(row["cover"][c]["cover"], res["reps"])
                        for c in res["columns"]))
    print()
    print(f"{'se/sd ratio (mean SE / MC sd of the estimate; <1 = SE too small)':<28}")
    print(f"{'':<28}" + "".join(f"{c:>13}" for c in res["columns"]))
    rule(93)
    for row in res["rows"]:
        print(f"{row['dgp']:<28}"
              + "".join(f"{row['se_over_sd'][c]:>13.3f}" for c in res["columns"]))
    print()
    print("read: row 1 all five agree -- nothing is broken when nothing is wrong.")
    print("      row 2 `nonrobust` is INCONSISTENT; hc0/hc1 repair almost all of it.")
    print("      row 3 hc0/hc1 repair NOTHING -- they are not serial-correlation")
    print("            robust. Only `hac` moves, and it does not get all the way back.")
    print("      row 4 the fat tails cost every se_type a little, and the robust")
    print("            ones slightly MORE: the sandwich needs a fourth moment that")
    print("            t(3) does not have.")


# ==========================================================================
# Experiment 2 -- the HC family in small samples, with an ORACLE control
# ==========================================================================
# HC0 is the plain White sandwich; its residuals are systematically too small
# because they are fitted, and the shrinkage is worst at high-leverage points.
# HC1 multiplies the covariance by n/(n-k) -- with k=2 that is a rounding
# error. HC2 and HC3 divide by (1-h_i) and (1-h_i)^2, which is the fix that
# actually targets leverage. `ols` now EXPOSES both, so the hc2/hc3 columns
# below are tsecon's own output; the starred hc2*/hc3* columns are the same
# sandwich rebuilt with numpy on the same draws and kept as a cross-check.
# They agree to ~1e-15 and `assertions()` locks that down -- if the two ever
# diverge, the library moved. The oracle column uses the true sd(e_i|x_i), so
# it isolates the cost of ESTIMATING the variance from the cost of the normal
# approximation.
LEVERAGE_SIZES = (25, 50, 100, 400, 1600)
LEVERAGE_LIB_COLUMNS = ("nonrobust", "hc0", "hc1", "hc2", "hc3")
LEVERAGE_COLUMNS = LEVERAGE_LIB_COLUMNS + ("hc2*", "hc3*", "oracle")


def exp_hc_leverage(reps, sizes=LEVERAGE_SIZES):
    beta = np.array([1.0, 2.0])
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for n in sizes:
        x = rng.chisquare(1.0, size=(reps, n))  # right-skewed -> high leverage
        sd_e = x  # sd(e|x) = x > 0: the leverage points are also the noisy ones
        e = sd_e * rng.standard_normal((reps, n))
        hits = {c: 0 for c in LEVERAGE_COLUMNS}
        se_sum = {c: 0.0 for c in LEVERAGE_COLUMNS}
        est = np.empty(reps)
        # largest |tsecon hc2 - numpy hc2*| (and hc3) seen on any replication:
        # the library-vs-reference cross-check, asserted below.
        ref_gap = 0.0
        # largest violation of the exact ladder hc0 <= hc2 <= hc3, which holds
        # sample by sample because the weights are ordered pointwise.
        ladder_gap = 0.0
        for i in range(reps):
            design = np.column_stack([np.ones(n), x[i]])
            y = design @ beta + e[i]
            # library columns -- hc2/hc3 are now tsecon output, not a reference
            lib_se = {}
            for name in LEVERAGE_LIB_COLUMNS:
                res = tsecon.ols(y, design, se_type=name)
                point, se = res["params"][1], res["bse"][1]
                lib_se[name] = se
                hits[name] += covered(point, se, beta[1])
                se_sum[name] += se
            est[i] = point
            ladder_gap = max(ladder_gap,
                             lib_se["hc0"] - lib_se["hc2"],
                             lib_se["hc2"] - lib_se["hc3"])
            # numpy reference columns on the same draw
            xtxi = np.linalg.inv(design.T @ design)
            coef = xtxi @ (design.T @ y)
            resid = y - design @ coef
            h = leverage(design, xtxi)
            for name, w in (("hc2*", resid ** 2 / (1.0 - h)),
                            ("hc3*", resid ** 2 / (1.0 - h) ** 2),
                            ("oracle", sd_e[i] ** 2)):
                se = sandwich_se(design, w)[1]
                hits[name] += covered(coef[1], se, beta[1])
                se_sum[name] += se
                if name in ("hc2*", "hc3*"):
                    ref_gap = max(ref_gap, abs(se - lib_se[name[:-1]]))
        sd = float(est.std(ddof=1))
        row = {"n": n, "mc_sd": sd, "cover": {}, "se_over_sd": {},
               "max_ref_gap": float(ref_gap), "max_ladder_gap": float(ladder_gap)}
        for name in LEVERAGE_COLUMNS:
            row["cover"][name] = cov_pair(hits[name] / reps, reps)
            row["se_over_sd"][name] = (se_sum[name] / reps) / sd
        rows.append(row)
    return {"name": "hc family under leverage", "reps": reps,
            "columns": list(LEVERAGE_COLUMNS), "rows": rows,
            "max_ref_gap": max(r["max_ref_gap"] for r in rows),
            "max_ladder_gap": max(r["max_ladder_gap"] for r in rows)}


def report_hc_leverage(res):
    header(f"2. ols HC family -- x ~ chi2(1) (high leverage), sd(e|x) = x, "
           f"reps={res['reps']}")
    print("hc2/hc3 ARE tsecon outputs (`se_type=\"hc2\"` / `\"hc3\"`). The starred")
    print("hc2*/hc3* columns are the same sandwich rebuilt with numpy on the same")
    print("draws, kept as a cross-check: largest |tsecon - numpy| over every")
    print(f"replication and every T is {res['max_ref_gap']:.2e}. `oracle` is the")
    print("sandwich at the TRUE sd(e|x): with Gaussian errors the estimate is exactly")
    print("normal around it, so the oracle column is exactly 0.95 by construction and")
    print("every shortfall to its left is the variance ESTIMATE, not the normal")
    print("approximation.")
    print()
    print(f"{'T':>6}" + "".join(f"{c:>13}" for c in res["columns"]))
    rule(110)
    for row in res["rows"]:
        print(f"{row['n']:>6d}"
              + "".join(cov_cell(row["cover"][c]["cover"], res["reps"])
                        for c in res["columns"]))
    print()
    print(f"{'T':>6}" + "".join(f"{c:>13}" for c in res["columns"]) + "   (se/sd)")
    rule(110)
    for row in res["rows"]:
        print(f"{row['n']:>6d}"
              + "".join(f"{row['se_over_sd'][c]:>13.3f}" for c in res["columns"]))
    print()
    print("read: `nonrobust` never converges to 0.95 -- it is inconsistent here, so")
    print("      more data does not help. hc0 does converge, but slowly, and hc1's")
    print("      n/(n-k) correction is nearly worthless at k=2. hc2/hc3 target the")
    print("      leverage directly and recover most of the small-T gap -- that is")
    print("      the whole reason they were added. They are still not exact: read")
    print("      the T=25 row as `hc3 is the best available answer here`, not as")
    print("      `hc3 is nominal`. Note also what hc3 costs when leverage is NOT a")
    print("      problem -- at T=1600 it is indistinguishable from hc0/hc1, so the")
    print("      leverage correction is close to free asymptotically.")


# ==========================================================================
# Experiment 3 -- HAC bandwidth for a SLOPE with a persistent regressor
# ==========================================================================
# `docs/examples/monte_carlo.py` experiment 2 measures HAC coverage for a MEAN
# (regression on a constant). That is the easy case: the score is just the
# error. Here both the regressor and the error are AR(1) with the same phi, so
# the score x_t e_t inherits autocorrelation ~ phi^2 and the long-run variance
# is a much larger multiple of the short-run one. This is where Newey-West's
# automatic bandwidth -- floor(4 (T/100)^(2/9)) = 4 at T=200 -- is simply too
# short, and no amount of correctness in the SE algebra fixes it.
HAC_PHIS = (0.0, 0.5, 0.8, 0.95)
HAC_COLUMNS = (
    ("nonrobust", {"se_type": "nonrobust"}),
    ("hc1", {"se_type": "hc1"}),
    ("hac auto", {"se_type": "hac"}),
    ("hac lag12", {"se_type": "hac", "maxlags": 12}),
    ("hac lag24", {"se_type": "hac", "maxlags": 24}),
)


def exp_hac_slope(reps, n=200, phis=HAC_PHIS):
    beta = np.array([1.0, 1.0])
    rng = np.random.default_rng(SEED + 3)
    rows = []
    for phi in phis:
        x = ar1_paths(rng, reps, n, phi)
        e = ar1_paths(rng, reps, n, phi)
        hits = {c: 0 for c, _ in HAC_COLUMNS}
        se_sum = {c: 0.0 for c, _ in HAC_COLUMNS}
        est = np.empty(reps)
        for i in range(reps):
            design = np.column_stack([np.ones(n), x[i]])
            y = design @ beta + e[i]
            for name, kw in HAC_COLUMNS:
                res = tsecon.ols(y, design, **kw)
                point, se = res["params"][1], res["bse"][1]
                hits[name] += covered(point, se, beta[1])
                se_sum[name] += se
            est[i] = point
        sd = float(est.std(ddof=1))
        row = {"phi": phi, "mc_sd": sd, "score_ac": phi * phi,
               "cover": {}, "se_over_sd": {}}
        for name, _ in HAC_COLUMNS:
            row["cover"][name] = cov_pair(hits[name] / reps, reps)
            row["se_over_sd"][name] = (se_sum[name] / reps) / sd
        rows.append(row)
    return {"name": "hac bandwidth on a slope", "reps": reps, "n": n,
            "columns": [c for c, _ in HAC_COLUMNS], "rows": rows}


def report_hac_slope(res):
    header(f"3. ols HAC -- coverage for a SLOPE, x and e both AR(1) at phi, "
           f"T={res['n']}, reps={res['reps']}")
    print("Extends monte_carlo.py experiment 2 (which covers a MEAN) to a slope with")
    print("a persistent regressor. `hac auto` uses floor(4 (T/100)^(2/9)) = 4 lags.")
    print()
    print(f"{'phi':>6}{'score AC':>10}"
          + "".join(f"{c:>13}" for c in res["columns"]))
    rule(81)
    for row in res["rows"]:
        print(f"{row['phi']:>6.2f}{row['score_ac']:>10.2f}"
              + "".join(cov_cell(row["cover"][c]["cover"], res["reps"])
                        for c in res["columns"]))
    print()
    print(f"{'phi':>6}{'':>10}" + "".join(f"{c:>13}" for c in res["columns"])
          + "   (se/sd)")
    rule(81)
    for row in res["rows"]:
        print(f"{row['phi']:>6.2f}{'':>10}"
              + "".join(f"{row['se_over_sd'][c]:>13.3f}" for c in res["columns"]))
    print()
    print("read: this is the worst table in the file and it is NOT a bug. At")
    print("      phi=0.95 the reported HAC SE is under half the true sampling sd,")
    print("      so a nominal 95% interval covers under 60% of the time. Lengthening")
    print("      the bandwidth to 24 helps and is still nowhere near nominal: at")
    print("      T=200 there is not enough independent information in the sample to")
    print("      estimate a long-run variance this large. Report a slope like this")
    print("      with a bandwidth chosen for the persistence, and treat the interval")
    print("      as indicative. At phi=0 HAC costs essentially nothing.")


# ==========================================================================
# Experiment 4a -- iv_gmm across instrument strength
# ==========================================================================
# Weak instruments are the textbook case where the point estimate, not the
# standard error, is the problem: 2SLS is badly biased toward the OLS
# probability limit, the reported SE is if anything too WIDE, and the interval
# still misses because it is centred in the wrong place. The se/sd and bias
# columns are what make that diagnosis instead of a guess.
#
# `iv_gmm` now RETURNS this diagnostic itself, as `first_stage`: a list of
# dicts with keys regressor / fstat / dof_num / dof_den / pval, one per
# endogenous regressor with excluded instruments to explain it. Entries are
# OMITTED where the statistic is undefined, so the list can be shorter than
# the regressor count and must be indexed by `regressor`, never by position.
# The `lib med F` column below is tsecon's own number; `med F` remains the
# textbook homoskedastic F computed here. They are DIFFERENT statistics --
# tsecon's is heteroskedasticity-robust -- so they are compared as medians,
# not sample by sample.
#
# With ONE endogenous regressor this is the right object. With two or more it
# is NOT a weak-identification test: every regressor can clear 10 while the
# system is under-identified, because the instruments may predict only one
# common combination of them. Angrist-Pischke (per regressor) and
# Cragg-Donald / Kleibergen-Paap against Stock-Yogo (joint) are the right
# objects there and tsecon implements none of them. Even here F > 10 is not a
# safety threshold -- the pi=0.20 row below has a median F of ~10 and covers
# well under nominal.
IV_STRENGTHS = (0.60, 0.20, 0.05)
IV_METHODS = ("2sls", "2step", "iterated")


def _first_stage_f(x, z_excluded, n):
    """F on the excluded instruments in x = const + z_excluded' g + v."""
    zz = np.column_stack([np.ones(n), z_excluded])
    g = np.linalg.lstsq(zz, x, rcond=None)[0]
    r_full = x - zz @ g
    r_null = x - x.mean()
    q = z_excluded.shape[1]
    ssr_full = float(r_full @ r_full)
    return ((float(r_null @ r_null) - ssr_full) / q) / (ssr_full / (n - q - 1))


def exp_iv_strength(reps, n=250, strengths=IV_STRENGTHS, endog=0.7):
    b1 = 1.0
    rng = np.random.default_rng(SEED + 4)
    rows = []
    for pi in strengths:
        z = rng.standard_normal((reps, n, 2))
        v = rng.standard_normal((reps, n))
        w = rng.standard_normal((reps, n))
        e = endog * v + np.sqrt(1.0 - endog * endog) * w
        x = pi * (z[:, :, 0] + z[:, :, 1]) + v
        y = 1.0 + b1 * x + e
        acc = {m: {"hits": 0, "se": 0.0, "j": 0, "est": np.empty(reps),
                   "ses": np.empty(reps)}
               for m in IV_METHODS}
        f_stats = np.empty(reps)
        lib_f = np.full(reps, np.nan)   # tsecon's own first_stage, robust F
        fs_missing = 0                  # replications with no entry for col 1
        fs_keys = set()
        for i in range(reps):
            design = np.column_stack([np.ones(n), x[i]])
            inst = np.column_stack([np.ones(n), z[i]])
            f_stats[i] = _first_stage_f(x[i], z[i], n)
            for m in IV_METHODS:
                res = tsecon.iv_gmm(design, inst, y[i], method=m, weight="robust")
                if m == "2sls":
                    # index by `regressor`, NOT by position: undefined entries
                    # are omitted, so the list may be short.
                    by_col = {int(d["regressor"]): d
                              for d in res.get("first_stage", [])}
                    fs_keys.update(by_col)
                    if 1 in by_col:
                        lib_f[i] = float(by_col[1]["fstat"])
                    else:
                        fs_missing += 1
                point, se = res["params"][1], res["bse"][1]
                a = acc[m]
                a["est"][i] = point
                a["ses"][i] = se
                a["se"] += se
                a["hits"] += covered(point, se, b1)
                a["j"] += res["j_pval"] < 0.05
        for m in IV_METHODS:
            a = acc[m]
            sd = float(a["est"].std(ddof=1))
            # ORACLE: the same interval built from the TRUE sampling sd (known
            # only in simulation). If this still under-covers, the shortfall is
            # the SHAPE of the sampling distribution -- non-normal and
            # mis-centred -- and no rescaling of the reported SE can fix it.
            oracle = float(np.mean(np.abs(a["est"] - b1) <= Z * sd))
            centred = float(np.mean(
                np.abs(a["est"] - a["est"].mean()) <= Z * sd))
            iqr = float(np.percentile(a["est"], 75) - np.percentile(a["est"], 25))
            rows.append({
                "pi": pi, "method": m,
                "first_stage_f": float(np.median(f_stats)),
                # tsecon's own `first_stage` diagnostic on the same draws
                "lib_first_stage_f": float(np.nanmedian(lib_f)),
                "lib_first_stage_missing": fs_missing / reps,
                # the constant is exogenous, so it must NOT get an entry
                "lib_first_stage_cols": sorted(fs_keys),
                "cover": cov_pair(a["hits"] / reps, reps),
                "oracle_cover": cov_pair(oracle, reps),
                "centred_cover": cov_pair(centred, reps),
                "se_over_sd": (a["se"] / reps) / sd,
                # the MEDIAN reported SE is the honest scale summary when the SE
                # distribution is itself skewed, which is exactly the weak case.
                "median_se_over_sd": float(np.median(a["ses"])) / sd,
                # does the SE know when the estimate is off?
                "corr_err_se": float(np.corrcoef(np.abs(a["est"] - b1),
                                                 a["ses"])[0, 1]),
                # 1.349 sd is the IQR of a normal: below 1 means peaked + fat
                # tailed, i.e. the normal approximation is not describing this.
                "iqr_over_normal": iqr / (1.349 * sd),
                "skew": float(np.mean((a["est"] - a["est"].mean()) ** 3) / sd ** 3),
                "mc_sd": sd,
                "bias": float(a["est"].mean() - b1),
                "median_bias": float(np.median(a["est"]) - b1),
                "j_size": cov_pair(a["j"] / reps, reps),
            })
    return {"name": "iv_gmm strength", "reps": reps, "n": n, "truth": b1,
            "endog": endog, "rows": rows}


def report_iv_strength(res):
    header(f"4a. iv_gmm -- coverage vs instrument strength, T={res['n']}, "
           f"reps={res['reps']}, corr(e,v)={res['endog']}")
    print("Over-identified by 1 df, so `j_size` is the Hansen J rejection rate at a")
    print("TRUE null (all instruments valid) -- it should be 0.05.")
    print("`oracle` rebuilds the interval from the TRUE sampling sd instead of the")
    print("reported SE. `se/sd` uses the MEAN reported SE and `med se/sd` the MEDIAN;")
    print("when those two disagree the SE distribution is itself skewed and the mean")
    print("is not a usable summary. `IQR/N` is the estimate's interquartile range")
    print("divided by 1.349 sd, which is 1.00 for a normal -- below 1 means peaked")
    print("with fat tails, i.e. the Wald approximation is not describing this.")
    print("`med F` is the textbook homoskedastic first-stage F computed here; `lib F`")
    print("is the median of tsecon's own `first_stage` fstat for the endogenous")
    print("column, which is heteroskedasticity-ROBUST and so is a different statistic")
    print("on the same draws -- they track, they do not coincide sample by sample.")
    print()
    print(f"{'pi':>6}{'med F':>7}{'lib F':>7}{'method':>10}{'cover':>15}{'oracle':>15}"
          f"{'se/sd':>8}{'med se/sd':>10}{'IQR/N':>7}{'skew':>8}{'corr':>7}"
          f"{'bias':>9}{'J size':>15}")
    rule(125)
    for row in res["rows"]:
        print(f"{row['pi']:>6.2f}{row['first_stage_f']:>7.1f}"
              f"{row['lib_first_stage_f']:>7.1f}{row['method']:>10}"
              + cov_cell(row["cover"]["cover"], res["reps"], 15)
              + cov_cell(row["oracle_cover"]["cover"], res["reps"], 15)
              + f"{row['se_over_sd']:>8.3f}{row['median_se_over_sd']:>10.3f}"
              + f"{row['iqr_over_normal']:>7.3f}{row['skew']:>+8.2f}"
              + f"{row['corr_err_se']:>+7.2f}{row['bias']:>+9.4f}"
              + cov_cell(row["j_size"]["cover"], res["reps"], 15))
    print()
    pick = {(r["pi"], r["method"]): r for r in res["rows"]}
    weak = pick[(min(IV_STRENGTHS), "2sls")]
    mod = pick[(0.20, "2sls")]
    print("read: strong instruments -- all three methods sit at nominal, the mean and")
    print("      median SE agree, IQR/N is ~1, and the oracle adds nothing. Healthy.")
    print(f"      Weak instruments (median F {weak['first_stage_f']:.1f}) under-cover"
          f" at {weak['cover']['cover']:.3f}, and the mean")
    print(f"      se/sd of {weak['se_over_sd']:.2f} is a MIRAGE: the MEDIAN reported"
          " SE is only")
    print(f"      {weak['median_se_over_sd']:.2f} of the true sampling sd, with the"
          " mean dragged above 1 by a")
    print("      handful of replications carrying enormous SEs. The TYPICAL interval")
    print("      is far too narrow even though the AVERAGE one looks generous.")
    print(f"      IQR/N = {weak['iqr_over_normal']:.2f} says the sampling"
          " distribution is nothing like normal,")
    print("      which is why the mean SE and the sd are both such misleading")
    print(f"      summaries. A FIXED-width interval at the true sd covers "
          f"{weak['oracle_cover']['cover']:.3f} against")
    print(f"      the reported SE's {weak['cover']['cover']:.3f}, so the damage is"
          " done by how the SE VARIES")
    print("      from sample to sample -- it is smallest in exactly the samples")
    print(f"      where the estimate is worst. The bias ({weak['bias']:+.3f} against"
          f" a sampling sd")
    print(f"      of {weak['mc_sd']:.2f}) is real but is not the main story. The"
          " oracle is not")
    print("      available in practice, so the fix is not a better SE but a")
    print("      different interval: a weak-instrument-robust one (Anderson-Rubin),")
    print("      which tsecon does not expose. Read the `first_stage` diagnostic")
    print("      iv_gmm now returns before you read the SE -- and note that even at a")
    print(f"      median F of {mod['first_stage_f']:.0f} (tsecon's robust version:"
          f" {mod['lib_first_stage_f']:.1f}), the conventional")
    print("      rule-of-thumb threshold, the median")
    print(f"      SE is already short ({mod['median_se_over_sd']:.2f}) and coverage"
          f" is {mod['cover']['cover']:.3f}.")
    print("      `skew` is reported but is itself barely estimable here -- 2SLS has")
    print("      no moments as the concentration parameter goes to zero, so that")
    print("      column swings with the replication count. IQR/N is the robust read.")
    print("      `corr` is the correlation between |estimate - truth| and the")
    print("      reported SE: the SE does partly know when it is wrong, which is why")
    print("      coverage degrades gradually rather than collapsing.")
    print("      The J test over-rejects slightly under the efficient 2-step/iterated")
    print("      weight (Hansen-Heaton-Yaron 1996) and under-rejects when the")
    print("      first stage is weak, because it has nothing to detect with.")
    print(f"      `first_stage` reports on columns {weak['lib_first_stage_cols']}"
          " only -- the constant is")
    print("      exogenous, so it correctly gets no entry. Index the list by")
    print("      `regressor`; a missing entry means UNDEFINED, not a failed fit.")
    print("      With one endogenous regressor this F is the right object. With two")
    print("      or more it is NOT a weak-identification test: all of them can clear")
    print("      10 while the system is under-identified. Angrist-Pischke and")
    print("      Cragg-Donald / Kleibergen-Paap are the right objects there, and")
    print("      tsecon implements none of them.")


# ==========================================================================
# Experiment 4b -- iv_gmm weight="hac": what the default bandwidth now does
# ==========================================================================
# `iv_gmm(..., weight="hac")` used to be a SILENT NO-OP: `bandwidth` defaulted
# to 0.0, and a Bartlett kernel truncated at 0 lags IS the White estimator, so
# it returned results bit-identical to weight="robust" while the caller
# believed they had serial-correlation robustness. Earlier runs of this file
# asserted that identity to nail the bug down.
#
# It is fixed. `bandwidth=None` is now the default and selects the Newey-West
# rule of thumb floor(4 (n/100)^(2/9)); an EXPLICIT bandwidth=0.0 raises; the
# truncation actually used comes back as `hac_bandwidth`. The assertions below
# now check the OPPOSITE of what they used to: the default must NO LONGER
# equal weight="robust", it must equal the rule-of-thumb lag count, and
# bandwidth=0.0 must raise instead of silently returning White.
#
# What the fix does NOT do is restore coverage. The rule of thumb picks 4 lags
# at T=250, FEWER than the 10 that the original audit found still left 0.868
# against nominal 0.95. It is a sensible default, not a remedy, and the table
# below is what says so.
def nw_bandwidth(n):
    """The Newey-West rule of thumb `iv_gmm` now uses: floor(4 (n/100)^(2/9))."""
    return float(int(4.0 * (n / 100.0) ** (2.0 / 9.0)))


IV_HAC_COLUMNS = (
    ("robust", {"weight": "robust"}),
    ("hac auto (NW rule)", {"weight": "hac"}),
    ("hac bw=4", {"weight": "hac", "bandwidth": 4.0}),
    ("hac bw=10", {"weight": "hac", "bandwidth": 10.0}),
)
IV_HAC_DEFAULT = "hac auto (NW rule)"


def _iv_hac_guardrails(design, inst, y):
    """The two ValueErrors that replaced the silent no-op, checked once.

    Neither depends on the draw, so they are evaluated on a single replication
    rather than 3000 times.
    """
    out = {}
    try:
        tsecon.iv_gmm(design, inst, y, method="2step", weight="hac",
                      bandwidth=0.0)
        out["zero_bandwidth_raises"] = False
        out["zero_bandwidth_msg"] = "(no error raised)"
    except ValueError as exc:
        out["zero_bandwidth_raises"] = True
        out["zero_bandwidth_msg"] = str(exc).splitlines()[0]
    try:
        tsecon.iv_gmm(design, inst, y, method="2sls", weight="hac")
        out["twosls_hac_raises"] = False
        out["twosls_hac_msg"] = "(no error raised)"
    except ValueError as exc:
        out["twosls_hac_raises"] = True
        out["twosls_hac_msg"] = str(exc).splitlines()[0]
    # weight="robust" has no truncation to report, so hac_bandwidth is None
    out["robust_bandwidth_is_none"] = (
        tsecon.iv_gmm(design, inst, y, method="2step",
                      weight="robust").get("hac_bandwidth") is None)
    return out


def exp_iv_hac(reps, n=250, phi=0.8, endog=0.7, pi=0.60):
    b1 = 1.0
    rng = np.random.default_rng(SEED + 44)
    z = np.stack([ar1_paths(rng, reps, n, phi) for _ in range(2)], axis=2)
    v = ar1_paths(rng, reps, n, phi)
    w = ar1_paths(rng, reps, n, phi)
    e = endog * v + np.sqrt(1.0 - endog * endog) * w
    x = pi * (z[:, :, 0] + z[:, :, 1]) + v
    y = 1.0 + b1 * x + e
    hits = {c: 0 for c, _ in IV_HAC_COLUMNS}
    se_sum = {c: 0.0 for c, _ in IV_HAC_COLUMNS}
    # unlike `ols`, a GMM weight matrix changes the POINT estimate too, so each
    # column needs its own sampling sd for the se/sd column to mean anything.
    est = {c: np.empty(reps) for c, _ in IV_HAC_COLUMNS}
    # |se(weight="hac", default bw) - se(weight="robust")|. This used to be
    # exactly 0 in every replication -- that WAS the bug. The MINIMUM is now
    # the interesting statistic: it must be strictly positive everywhere.
    max_default_gap = 0.0
    min_default_gap = np.inf
    # |se(default) - se(explicit bandwidth=4)|: the default IS the rule, so at
    # T=250 (rule of thumb = 4) these two columns must agree bit for bit.
    max_auto_vs_bw4 = 0.0
    reported_bw = set()
    guards = _iv_hac_guardrails(np.column_stack([np.ones(n), x[0]]),
                                np.column_stack([np.ones(n), z[0]]), y[0])
    for i in range(reps):
        design = np.column_stack([np.ones(n), x[i]])
        inst = np.column_stack([np.ones(n), z[i]])
        ses = {}
        for name, kw in IV_HAC_COLUMNS:
            res = tsecon.iv_gmm(design, inst, y[i], method="2step", **kw)
            point, se = res["params"][1], res["bse"][1]
            ses[name] = se
            est[name][i] = point
            hits[name] += covered(point, se, b1)
            se_sum[name] += se
            if name == IV_HAC_DEFAULT:
                reported_bw.add(res.get("hac_bandwidth"))
        gap = abs(ses[IV_HAC_DEFAULT] - ses["robust"])
        max_default_gap = max(max_default_gap, gap)
        min_default_gap = min(min_default_gap, gap)
        max_auto_vs_bw4 = max(max_auto_vs_bw4,
                              abs(ses[IV_HAC_DEFAULT] - ses["hac bw=4"]))
    sd = {c: float(est[c].std(ddof=1)) for c, _ in IV_HAC_COLUMNS}
    return {
        "name": "iv_gmm hac bandwidth", "reps": reps, "n": n, "phi": phi,
        "mc_sd": sd,
        "max_default_gap": max_default_gap,
        "min_default_gap": float(min_default_gap),
        "max_auto_vs_bw4": max_auto_vs_bw4,
        "reported_bandwidth": sorted(b for b in reported_bw if b is not None),
        "n_reported_bandwidths": len(reported_bw),
        "nw_rule_bandwidth": nw_bandwidth(n),
        "columns": [c for c, _ in IV_HAC_COLUMNS],
        "cover": {c: cov_pair(hits[c] / reps, reps) for c, _ in IV_HAC_COLUMNS},
        "se_over_sd": {c: (se_sum[c] / reps) / sd[c] for c, _ in IV_HAC_COLUMNS},
        "bias": {c: float(est[c].mean() - b1) for c, _ in IV_HAC_COLUMNS},
        **guards,
    }


def report_iv_hac(res):
    header(f"4b. iv_gmm weight='hac' -- AR(1) errors phi={res['phi']}, "
           f"method='2step', T={res['n']}, reps={res['reps']}")
    print("FIXED (this experiment used to record the bug): `bandwidth` defaulted to")
    print("0.0, and Bartlett at 0 lags is White, so `weight=\"hac\"` alone changed")
    print("NOTHING. The default is now `bandwidth=None` -> the Newey-West rule of")
    print(f"thumb floor(4 (T/100)^(2/9)) = {res['nw_rule_bandwidth']:.0f} at T="
          f"{res['n']}, returned as `hac_bandwidth`")
    print(f"({res['reported_bandwidth']} on every one of {res['reps']} replications).")
    print("Smallest |se(hac,default) - se(robust)| over all replications: "
          f"{res['min_default_gap']:.3e}")
    print(f"(it used to be 0.000e+00 in every one). |se(default) - se(bandwidth=4)|"
          f" = {res['max_auto_vs_bw4']:.1e},")
    print("so the default really is the rule and not a coincidence.")
    print(f"  explicit bandwidth=0.0 raises ValueError: {res['zero_bandwidth_raises']}")
    print(f"  method='2sls' with weight='hac' raises ValueError: "
          f"{res['twosls_hac_raises']}")
    print(f"  weight='robust' reports hac_bandwidth=None: "
          f"{res['robust_bandwidth_is_none']}")
    print()
    print(f"{'weight':>22}{'cover':>15}{'se/sd':>9}{'bias':>10}{'mc sd':>9}")
    rule(65)
    for c in res["columns"]:
        print(f"{c:>22}" + cov_cell(res["cover"][c]["cover"], res["reps"], 15)
              + f"{res['se_over_sd'][c]:>9.3f}{res['bias'][c]:>+10.4f}"
              + f"{res['mc_sd'][c]:>9.4f}")
    print()
    print("read: a working default is NOT a fix for coverage, and this table is the")
    print("      reason to say so out loud. The rule of thumb picks "
          f"{res['nw_rule_bandwidth']:.0f} lags at T={res['n']},")
    print(f"      which buys "
          f"{res['cover'][IV_HAC_DEFAULT]['cover'] - res['cover']['robust']['cover']:+.3f}"
          f" of coverage over weight='robust' -- real, and")
    print("      nowhere near nominal. Passing a LONGER bandwidth than the rule does")
    print(f"      better ({res['cover']['hac bw=10']['cover']:.3f} at bw=10) and is"
          " still short of 0.95, exactly as in")
    print("      experiment 3: a Bartlett kernel cannot repair coverage under this")
    print("      much persistence at T=250. Choose the bandwidth for the persistence")
    print("      you actually have; the default only stops the silent no-op. The")
    print("      weight matrix moves the POINT estimate as well as the SE, so each")
    print("      row has its own sampling sd; the bias column confirms the")
    print("      differences are tiny.")


# ==========================================================================
# Experiment 5 -- har_rv Bartlett-HAC SEs on a correctly specified HAR
# ==========================================================================
# The HAR regressors are overlapping moving averages, which is why the model
# ships HAC SEs. But if the HAR *is* the DGP with iid innovations, the score is
# already serially uncorrelated and every extra HAC lag only adds noise to the
# variance estimate -- a downward-biased SE. The intercept is the interesting
# coefficient: with b_d + b_w + b_m = 0.95 the persistence is severely
# downward-biased in finite samples (Kendall), and because all four regressors
# have essentially the same sample mean as the target, that bias is pushed
# almost exactly into the constant, via
#     bias(c_hat) ~= -mu * sum(bias(b_hat)),   mu = c / (1 - b_d - b_w - b_m).
# `assertions()` checks that identity numerically rather than asserting the
# story is plausible.
HAR_TRUTH = np.array([-0.20, 0.35, 0.35, 0.25])  # [c, b_d, b_w, b_m]
HAR_SIGMA = 0.40
HAR_NAMES = ("const", "b_daily", "b_weekly", "b_monthly")
HAR_LAGS = (0, 5, 22)


def har_paths(rng, reps, n, truth=HAR_TRUTH, sigma=HAR_SIGMA, het=False, burn=250):
    """(reps, n) draws of log-RV from the HAR recursion `har_rv` actually fits.

    Windows match `crates/tsecon-realized/src/har.rs` exactly (the Corsi
    2009 aggregates, the 0.5.0 window fix): the daily term is h[t-1], the
    weekly term is mean(h[t-5:t]) and the monthly term is mean(h[t-22:t])
    (half-open, trailing means running through the daily lag).
    """
    c, b_d, b_w, b_m = truth
    total = n + burn
    mu = c / (1.0 - b_d - b_w - b_m)
    h = np.full((reps, total), mu)
    z = rng.standard_normal((reps, total))
    for t in range(23, total):
        daily = h[:, t - 1]
        weekly = h[:, t - 5:t].mean(axis=1)
        monthly = h[:, t - 22:t].mean(axis=1)
        # `het=True` scales the innovation by a function of the PAST only, so
        # the conditional mean -- and therefore the truth -- is unchanged.
        scale = np.sqrt(0.5 + 0.5 * (daily - mu) ** 2) if het else 1.0
        h[:, t] = c + b_d * daily + b_w * weekly + b_m * monthly + sigma * scale * z[:, t]
    return h[:, burn:]


def exp_har_hac(reps, n=1000, lags=HAR_LAGS):
    rows = []
    for het in (False, True):
        rng = np.random.default_rng(SEED + (5 if not het else 55))
        rv = np.exp(har_paths(rng, reps, n, het=het))
        for maxlags in lags:
            hits = np.zeros(4)
            se_sum = np.zeros(4)
            est = np.empty((reps, 4))
            for i in range(reps):
                # use_correction pinned to what the published tables measured:
                # this experiment ran (and its rows were harvested) while
                # har_rv defaulted the n/(n-k) HAC correction OFF. The default
                # flipped to True with the round-2 finding-4 fix; the delta on
                # the SEs is sqrt(n/(n-k)) ~ +0.2% at this design's n, an
                # order below the tables' +-0.004 MC error, so the published
                # rows remain valid for the pinned setting they name.
                res = tsecon.har_rv(rv[i], variant="log", hac_maxlags=maxlags,
                                    use_correction=False)
                point = np.asarray(res["params"], dtype=float)
                se = np.asarray(res["bse"], dtype=float)
                est[i] = point
                se_sum += se
                hits += np.abs(point - HAR_TRUTH) <= Z * se
            sd = est.std(axis=0, ddof=1)
            bias = est.mean(axis=0) - HAR_TRUTH
            rows.append({
                "het": het, "maxlags": maxlags,
                "cover": [cov_pair(v / reps, reps) for v in hits],
                "se_over_sd": list((se_sum / reps) / sd),
                "bias": list(bias),
                "mc_sd": list(sd),
                # the mechanical prediction described above
                "predicted_const_bias": float(
                    -(HAR_TRUTH[0] / (1.0 - HAR_TRUTH[1:].sum())) * bias[1:].sum()),
            })
    return {"name": "har_rv hac", "reps": reps, "n": n,
            "truth": list(HAR_TRUTH), "rows": rows}


def report_har_hac(res):
    header(f"5. har_rv -- HAC coverage on a correctly specified HAR, "
           f"T={res['n']}, reps={res['reps']}")
    print(f"truth [const, b_d, b_w, b_m] = {np.array(res['truth'])}, "
          f"persistence sum(b) = {sum(res['truth'][1:]):.2f}")
    print()
    print(f"{'errors':>8}{'maxlags':>9}" + "".join(f"{c:>13}" for c in HAR_NAMES))
    rule(69)
    for row in res["rows"]:
        tag = "het" if row["het"] else "iid"
        print(f"{tag:>8}{row['maxlags']:>9d}"
              + "".join(cov_cell(c["cover"], res["reps"]) for c in row["cover"]))
    print()
    print(f"{'errors':>8}{'maxlags':>9}" + "".join(f"{c:>13}" for c in HAR_NAMES)
          + "   (se/sd)")
    rule(69)
    for row in res["rows"]:
        tag = "het" if row["het"] else "iid"
        print(f"{tag:>8}{row['maxlags']:>9d}"
              + "".join(f"{v:>13.3f}" for v in row["se_over_sd"]))
    print()
    # bias is a property of the POINT estimate, so it is identical across the
    # maxlags rows -- printed once per error structure rather than three times.
    print(f"{'errors':>8}{'maxlags':>9}" + "".join(f"{c:>13}" for c in HAR_NAMES)
          + "   (bias -- does not depend on maxlags)")
    rule(69)
    for row in res["rows"]:
        if row["maxlags"] != 5:
            continue
        tag = "het" if row["het"] else "iid"
        print(f"{tag:>8}{'any':>9}"
              + "".join(f"{v:>+13.4f}" for v in row["bias"]))
    print()
    for row in res["rows"]:
        if row["maxlags"] == 5:
            tag = "het" if row["het"] else "iid"
            print(f"      const bias, {tag} errors: measured "
                  f"{row['bias'][0]:+.4f} vs -mu*sum(slope bias) = "
                  f"{row['predicted_const_bias']:+.4f}")
    print()
    print("read: the three SLOPES cover at nominal. The CONSTANT does not, and the")
    print("      line above says why: it is not the SE, it is the centring. With")
    print("      sum(b) = 0.95 the persistence is downward-biased, and since all")
    print("      four regressors share the target's sample mean, the constant")
    print("      absorbs the whole error -- the prediction matches to ~1%. This is")
    print("      the classic least-squares AR bias, not a defect in har_rv.")
    print("      Note also that MORE HAC lags monotonically SHRINK the SE here")
    print("      (see se/sd): the true score is serially uncorrelated, so every")
    print("      extra lag is pure estimation noise. Bandwidth is not free.")


# ==========================================================================
# Experiment 6 -- recession_probit Wald intervals, common vs rare events
# ==========================================================================
# The headline here is not coverage -- it is that in the rare-event regime a
# large share of replications have NO finite MLE at all (either the sample
# contains no recession months, or the predictors separate the outcome), and
# `recession_probit` correctly refuses with a diagnostic ValueError rather than
# returning a garbage interval. Any coverage number in that regime is therefore
# CONDITIONAL on the sample admitting an estimate, and that conditioning is not
# innocuous: the surviving samples are the informative ones.
PROBIT_CASES = (
    ("probit common phi=0.9", "probit", -0.25, 1.0, 0.9),
    ("probit rare   phi=0.9", "probit", -4.00, 1.0, 0.9),
    ("logit  rare   phi=0.9", "logit", -6.80, 1.7, 0.9),
)
PROBIT_SIZES = (100, 250, 1000)


def exp_probit_wald(reps, cases=PROBIT_CASES, sizes=PROBIT_SIZES):
    rows = []
    for idx, (label, link, b0, b1, phi) in enumerate(cases):
        for n in sizes:
            rng = np.random.default_rng(SEED + 600 + 10 * idx + sizes.index(n))
            x = ar1_paths(rng, reps, n, phi)
            index = b0 + b1 * x
            if link == "probit":
                y = (index + rng.standard_normal((reps, n)) > 0).astype(float)
            else:
                p = 1.0 / (1.0 + np.exp(-index))
                y = (rng.random((reps, n)) < p).astype(float)
            hits = 0
            se_sum = 0.0
            est = []
            ses = []
            fail_degenerate = 0
            fail_separation = 0
            for i in range(reps):
                design = np.column_stack([np.ones(n), x[i]])
                try:
                    res = tsecon.recession_probit(y[i], design, link=link)
                except ValueError as exc:
                    if "degenerate" in str(exc):
                        fail_degenerate += 1
                    else:
                        fail_separation += 1
                    continue
                point, se = res["params"][1], res["bse"][1]
                est.append(point)
                ses.append(se)
                se_sum += se
                hits += covered(point, se, b1)
            est = np.asarray(est)
            ses = np.asarray(ses)
            ok = est.size
            fails = fail_degenerate + fail_separation
            sd = float(est.std(ddof=1)) if ok > 1 else float("nan")
            rows.append({
                "case": label, "link": link, "n": n, "truth": b1,
                "event_rate": float(y.mean()),
                "n_ok": ok, "fail_share": fails / reps,
                "fail_degenerate": fail_degenerate / reps,
                "fail_separation": fail_separation / reps,
                "cover": cov_pair(hits / ok, ok) if ok else
                         {"cover": float("nan"), "mcse": float("nan")},
                "se_over_sd": (se_sum / ok) / sd if ok > 1 else float("nan"),
                "bias": float(est.mean() - b1) if ok else float("nan"),
                "median_bias": float(np.median(est) - b1) if ok else float("nan"),
                # share of survivors whose SE is 3x the median: near-separation
                "wide_share": float((ses > 3.0 * np.median(ses)).mean()) if ok else
                              float("nan"),
            })
    return {"name": "recession_probit wald", "reps": reps, "rows": rows}


def report_probit_wald(res):
    header(f"6. recession_probit -- Wald interval for the slope, reps={res['reps']}")
    print("`fail%` is the share of replications with NO finite MLE, split into a")
    print("degenerate response (no recession months at all) and (quasi-)complete")
    print("separation. `cover` is conditional on the survivors -- read it with")
    print("`fail%` in view, never alone.")
    print()
    print(f"{'case':>24}{'T':>6}{'rate':>7}{'fail%':>7}{'degen':>7}{'sep':>7}"
          f"{'cover':>15}{'se/sd':>8}{'bias':>9}{'med bias':>10}{'wide%':>7}")
    rule(107)
    for row in res["rows"]:
        print(f"{row['case']:>24}{row['n']:>6d}{row['event_rate']:>7.3f}"
              f"{row['fail_share']:>7.3f}{row['fail_degenerate']:>7.3f}"
              f"{row['fail_separation']:>7.3f}"
              + cov_cell(row["cover"]["cover"], max(row["n_ok"], 1), 15)
              + f"{row['se_over_sd']:>8.3f}{row['bias']:>+9.4f}"
              f"{row['median_bias']:>+10.4f}{row['wide_share']:>7.3f}")
    print()
    print("read: coverage among the survivors is at or ABOVE nominal everywhere --")
    print("      recession_probit's intervals are not too narrow. Two other things")
    print("      are true and matter more. (i) In the rare regime at T=100 a quarter")
    print("      of samples have no estimate at all; that is the right behaviour")
    print("      (there is no finite MLE) but it means the reported interval exists")
    print("      only for the luckier samples. (ii) The MLE is biased AWAY from zero")
    print("      in small samples -- see `med bias`, which shrinks like 1/T -- and")
    print("      a few near-separated survivors carry huge estimates AND huge SEs,")
    print("      which is why se/sd falls well below 1 while coverage stays high.")
    print("      se/sd < 1 with high coverage is an outlier signature, not a")
    print("      too-narrow interval: `wide%` shows those replications directly.")


# ==========================================================================
# Experiment 7 -- quantile_regression Powell sandwich SEs, per tau
# ==========================================================================
# The truth is the conditional quantile of a location-scale DGP, which is
# exactly linear in x, so every tau has a known coefficient pair. The Powell
# sandwich needs the conditional density at the fitted quantile, estimated
# with an Epanechnikov kernel and a Hall-Sheather bandwidth. In the tails
# there are few observations near the quantile, the sparsity estimate is
# noisy and biased, and coverage suffers -- which is exactly what shows up.
QR_TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
QR_A, QR_B, QR_S0 = 1.0, 1.0, 0.5
QR_DESIGNS = (
    ("homoskedastic  T=200", 0.0, 200),
    ("location-scale T=200", 0.5, 200),
    ("location-scale T=1000", 0.5, 1000),
)


def exp_quantile_powell(reps, taus=QR_TAUS, designs=QR_DESIGNS):
    taus = list(taus)
    z_tau = norm.ppf(taus)
    rows = []
    for label, s1, n in designs:
        # truth: Q_tau(y|x) = (a + s0 z_tau) + (b + s1 z_tau) x
        true_icpt = QR_A + QR_S0 * z_tau
        true_slope = QR_B + s1 * z_tau
        rng = np.random.default_rng(SEED + 700 + int(100 * s1) + n)
        x = rng.uniform(0.0, 2.0, size=(reps, n))
        u = rng.standard_normal((reps, n))
        y = QR_A + QR_B * x + (QR_S0 + s1 * x) * u
        hit_s = np.zeros(len(taus))
        hit_i = np.zeros(len(taus))
        se_sum = np.zeros(len(taus))
        est = np.empty((reps, len(taus)))
        # `converged` is a single flag over all taus and reflects the IRLS
        # iteration cap, not a failure. Track coverage on each side of it so
        # the headline numbers can be shown not to depend on the split.
        conv_flag = np.zeros(reps, dtype=bool)
        hit_split = {True: np.zeros(len(taus)), False: np.zeros(len(taus))}
        for i in range(reps):
            design = np.column_stack([np.ones(n), x[i]])
            res = tsecon.quantile_regression(y[i], design, taus=taus)
            ok = bool(res["converged"])
            conv_flag[i] = ok
            params = np.asarray(res["params"], dtype=float)
            bse = np.asarray(res["bse"], dtype=float)
            est[i] = params[:, 1]
            se_sum += bse[:, 1]
            hit = np.abs(params[:, 1] - true_slope) <= Z * bse[:, 1]
            hit_s += hit
            hit_split[ok] += hit
            hit_i += np.abs(params[:, 0] - true_icpt) <= Z * bse[:, 0]
        sd = est.std(axis=0, ddof=1)
        n_conv = int(conv_flag.sum())
        n_non = reps - n_conv
        rows.append({
            "design": label, "s1": s1, "n": n, "taus": taus,
            "true_slope": list(true_slope), "true_icpt": list(true_icpt),
            "nonconverged": n_non,
            "cover_slope": [cov_pair(v / reps, reps) for v in hit_s],
            "cover_icpt": [cov_pair(v / reps, reps) for v in hit_i],
            "cover_converged": [cov_pair(v / n_conv, n_conv) for v in hit_split[True]]
                               if n_conv else None,
            "cover_nonconverged": [cov_pair(v / n_non, n_non)
                                   for v in hit_split[False]] if n_non else None,
            "se_over_sd": list((se_sum / reps) / sd),
            "bias": list(est.mean(axis=0) - true_slope),
        })
    return {"name": "quantile_regression powell", "reps": reps,
            "taus": taus, "rows": rows}


def report_quantile_powell(res):
    header(f"7. quantile_regression -- Powell sandwich coverage per tau, "
           f"reps={res['reps']}")
    print("Truth is the conditional quantile of y = a + b x + (s0 + s1 x) u, which is")
    print("exactly linear, so the true slope at tau is b + s1 z_tau (it fans out with")
    print("tau when s1 > 0 and is flat when s1 = 0).")
    print("`converged` is one flag over all taus and trips when the IRLS hits")
    print("statsmodels' iteration cap; it is not an estimation failure. The two")
    print("`cover (conv/non)` rows show the headline numbers do not hinge on it.")
    for row in res["rows"]:
        print()
        print(f"{row['design']}   (non-converged replications: "
              f"{row['nonconverged']}/{res['reps']})")
        print(f"{'tau':>12}" + "".join(f"{t:>13.2f}" for t in row["taus"]))
        rule(103)
        print(f"{'true slope':>12}"
              + "".join(f"{v:>13.3f}" for v in row["true_slope"]))
        print(f"{'cover slope':>12}"
              + "".join(cov_cell(c["cover"], res["reps"]) for c in row["cover_slope"]))
        print(f"{'se/sd':>12}"
              + "".join(f"{v:>13.3f}" for v in row["se_over_sd"]))
        print(f"{'bias':>12}"
              + "".join(f"{v:>+13.4f}" for v in row["bias"]))
        print(f"{'cover icpt':>12}"
              + "".join(cov_cell(c["cover"], res["reps"]) for c in row["cover_icpt"]))
        if row["cover_converged"] and row["cover_nonconverged"]:
            n_conv = res["reps"] - row["nonconverged"]
            print(f"{'cover conv':>12}"
                  + "".join(f"{c['cover']:>13.3f}" for c in row["cover_converged"])
                  + f"   (n={n_conv})")
            print(f"{'cover non':>12}"
                  + "".join(f"{c['cover']:>13.3f}"
                            for c in row["cover_nonconverged"])
                  + f"   (n={row['nonconverged']})")
    print()
    print("read: the median is fine. The TAILS under-cover at T=200 -- ~0.87 at")
    print("      tau=0.05 against a nominal 0.95 -- and se/sd says the Powell SE is")
    print("      the culprit (~0.82-0.88 of the true sampling sd), not the point")
    print("      estimate, whose bias is negligible. The sparsity/density estimate")
    print("      that the sandwich needs is built from the few observations near an")
    print("      extreme conditional quantile, so it is badly determined. At T=1000")
    print("      the shortfall shrinks but has not closed. Treat an extreme-tau")
    print("      Powell interval at a few hundred observations as optimistic; a")
    print("      bootstrap over the whole quantile process is the honest alternative.")
    print("      The INTERCEPT over-covers in the location-scale designs (~0.98-0.99):")
    print("      x ~ U(0,2) puts x=0 at the edge of the support, so the intercept is")
    print("      an extrapolation whose sandwich SE is conservative.")


# ==========================================================================
# structural checks -- facts that must hold exactly, independent of any DGP
# ==========================================================================
def structural_checks():
    """Verify the HAR window convention and the se_type/ordering algebra.

    These are exact statements, so they are checked once at high precision
    rather than measured. Getting the HAR windows wrong would silently make
    experiment 5 measure the wrong truth.
    """
    facts = {}
    rng = np.random.default_rng(SEED + 90)
    n = 300
    rv = np.exp(0.3 * rng.standard_normal(n))
    h = np.log(rv)
    # rebuild `har_rv`'s design by hand from the documented windows (the
    # Corsi 2009 trailing means through the daily lag -- the 0.5.0 fix)
    rows = range(23, n)
    y = np.array([h[t] for t in rows])
    design = np.column_stack([
        np.ones(len(y)),
        np.array([h[t - 1] for t in rows]),
        np.array([h[t - 5:t].mean() for t in rows]),
        np.array([h[t - 22:t].mean() for t in rows]),
    ])
    hand = tsecon.ols(y, design, se_type="hac", maxlags=5, use_correction=False)
    lib = tsecon.har_rv(rv, variant="log", start=22, hac_maxlags=5,
                        use_correction=False)
    facts["har_design_params"] = float(
        np.max(np.abs(np.asarray(lib["params"]) - np.asarray(hand["params"]))))
    facts["har_design_bse"] = float(
        np.max(np.abs(np.asarray(lib["bse"]) - np.asarray(hand["bse"]))))
    facts["har_nobs"] = int(lib["nobs"]) - len(y)

    # hc1 = hc0 * sqrt(n / (n - k)) exactly, so the hc1 interval always
    # contains the hc0 one: hc1 coverage >= hc0 coverage in EVERY sample.
    m = 120
    x = np.column_stack([np.ones(m), rng.standard_normal(m)])
    yy = x @ np.array([1.0, 2.0]) + np.abs(x[:, 1]) * rng.standard_normal(m)
    hc0 = np.asarray(tsecon.ols(yy, x, se_type="hc0")["bse"])
    hc1 = np.asarray(tsecon.ols(yy, x, se_type="hc1")["bse"])
    facts["hc1_over_hc0"] = float(
        np.max(np.abs(hc1 / hc0 - np.sqrt(m / (m - x.shape[1])))))

    # hc2 / hc3 are the leverage-corrected sandwiches: weights r_i^2/(1-h_i)
    # and r_i^2/(1-h_i)^2. Reproduce both from the hat matrix by hand -- this
    # is the algebraic identity behind experiment 2's cross-check columns.
    xtxi = np.linalg.inv(x.T @ x)
    coef = xtxi @ (x.T @ yy)
    resid = yy - x @ coef
    h = leverage(x, xtxi)
    hc2 = np.asarray(tsecon.ols(yy, x, se_type="hc2")["bse"])
    hc3 = np.asarray(tsecon.ols(yy, x, se_type="hc3")["bse"])
    facts["hc2_vs_numpy"] = float(np.max(np.abs(
        hc2 - sandwich_se(x, resid ** 2 / (1.0 - h)))))
    facts["hc3_vs_numpy"] = float(np.max(np.abs(
        hc3 - sandwich_se(x, resid ** 2 / (1.0 - h) ** 2))))
    # 1 <= 1/(1-h) <= 1/(1-h)^2 pointwise, so the sandwich differences are
    # PSD and the ladder hc0 <= hc2 <= hc3 holds coefficient by coefficient
    # in EVERY sample. (hc1 is not in the ladder: n/(n-k) is unrelated to h.)
    facts["hc_ladder_violation"] = float(
        max(np.max(hc0 - hc2), np.max(hc2 - hc3)))

    # `se_type` must not touch the POINT estimate -- experiments 1-3 rely on
    # this to share one sampling sd across every se_type column, and it also
    # licenses comparing the numpy hc2*/hc3* references in experiment 2 to
    # tsecon's own hc2/hc3 columns on the same draw.
    params = [np.asarray(tsecon.ols(yy, x, se_type=s)["params"])
              for s in OLS_SE_TYPES]
    facts["params_invariant_to_se_type"] = float(
        max(np.max(np.abs(p - params[0])) for p in params))
    facts["params_match_numpy"] = float(np.max(np.abs(
        params[0] - np.linalg.lstsq(x, yy, rcond=None)[0])))

    # the se_type menu really is just these six, lower-case
    rejected = []
    for bad in ("HC0", "HC1", "HC2", "HC3", "HAC", "hc4"):
        try:
            tsecon.ols(yy, x, se_type=bad)
        except ValueError:
            rejected.append(bad)
    facts["rejected_se_types"] = rejected
    accepted = []
    for good in OLS_SE_TYPES:
        try:
            tsecon.ols(yy, x, se_type=good)
            accepted.append(good)
        except ValueError:
            pass
    facts["accepted_se_types"] = accepted

    # A leverage of exactly 1 makes the hc2/hc3 weight infinite AND forces the
    # residual to 0 by construction -- the fit runs through that point. `ols`
    # refuses instead of returning a near-infinite SE. A dummy that isolates a
    # single observation is the canonical way to produce h_i = 1.
    m2 = 10
    d = np.zeros(m2)
    d[3] = 1.0
    x_lev = np.column_stack([np.ones(m2), d])
    y_lev = x_lev @ np.array([1.0, 2.0]) + rng.standard_normal(m2)
    refused = []
    for name in ("hc2", "hc3"):
        try:
            tsecon.ols(y_lev, x_lev, se_type=name)
        except ValueError as exc:
            if "leverage" in str(exc):
                refused.append(name)
    facts["unit_leverage_refused"] = refused
    # hc0/hc1 do NOT refuse -- their weights stay finite there. Recording this
    # keeps the contrast honest rather than implying a blanket guard.
    facts["unit_leverage_hc0_ok"] = bool(
        np.all(np.isfinite(np.asarray(
            tsecon.ols(y_lev, x_lev, se_type="hc0")["bse"]))))
    return facts


def report_structural(facts):
    header("0. structural checks -- exact facts the experiments rely on")
    print(f"har_rv design matches a hand-built [const, h_(t-1), mean h[t-5:t],")
    print(f"  mean h[t-22:t]] regression: max |param diff| = "
          f"{facts['har_design_params']:.2e}, max |bse diff| = "
          f"{facts['har_design_bse']:.2e}, nobs diff = {facts['har_nobs']}")
    print(f"hc1 / hc0 == sqrt(n/(n-k)) exactly: max deviation = "
          f"{facts['hc1_over_hc0']:.2e}")
    print(f"hc2 / hc3 reproduce the hand-built leverage sandwiches r^2/(1-h) and")
    print(f"  r^2/(1-h)^2: max |diff| = {facts['hc2_vs_numpy']:.2e} and "
          f"{facts['hc3_vs_numpy']:.2e}. The exact ladder")
    print(f"  hc0 <= hc2 <= hc3 holds coefficient by coefficient (max violation "
          f"{facts['hc_ladder_violation']:.2e}).")
    print(f"se_type does not move the point estimate (max diff "
          f"{facts['params_invariant_to_se_type']:.2e}) and the estimate matches")
    print(f"  numpy lstsq (max diff {facts['params_match_numpy']:.2e}) -- so one")
    print("  sampling sd is shared across the se_type columns in experiments 1-3.")
    print(f"se_type accepts {facts['accepted_se_types']}")
    print(f"  and rejects {facts['rejected_se_types']} -- the menu is exactly")
    print(f"  {list(OLS_SE_TYPES)}, lower-case. hc2/hc3 were ADDED after this")
    print("  suite measured what their absence cost; experiment 2 now reports")
    print("  tsecon's own hc2/hc3 coverage rather than a numpy reference.")
    print(f"hc2/hc3 refuse a point with leverage exactly 1 "
          f"({facts['unit_leverage_refused']}) rather than")
    print("  return a near-infinite SE -- the weight 1/(1-h) is infinite there and")
    print("  the residual is 0 by construction. hc0 still returns a finite SE on the")
    print(f"  same design ({facts['unit_leverage_hc0_ok']}); the guard is specific to")
    print("  the leverage-corrected weights, not a blanket refusal.")


# ==========================================================================
# assertions -- only what is robustly true, with MC-aware slack
# ==========================================================================
def assertions(results, facts, reps):
    """Check claims that theory guarantees, with slack that scales with reps.

    Nothing here was tuned to pass. Every threshold is either an exact
    algebraic fact, or a gap so large that three Monte Carlo standard errors
    are irrelevant to it. Where the library genuinely under-covers, the
    assertion locks the SHORTFALL in rather than wishing it away -- those are
    properties of the asymptotics, not bugs, and a future reader should be
    told if they ever change.
    """
    checks = []

    def check(label, ok, detail):
        checks.append((label, bool(ok), detail))

    def at_least(p, floor, n):
        return p >= floor - 3.0 * mc_se(p, n)

    def at_most(p, ceil, n):
        return p <= ceil + 3.0 * mc_se(p, n)

    def gap_at_least(p_hi, p_lo, gap, n):
        slack = 3.0 * np.sqrt(mc_se(p_hi, n) ** 2 + mc_se(p_lo, n) ** 2)
        return (p_hi - p_lo) >= gap - slack

    # ---- structural (exact) ----
    check("har_rv design == hand-built HAR windows",
          facts["har_design_params"] < 1e-10 and facts["har_design_bse"] < 1e-10
          and facts["har_nobs"] == 0,
          f"params {facts['har_design_params']:.1e}, bse {facts['har_design_bse']:.1e}")
    check("hc1 == hc0 * sqrt(n/(n-k)) exactly",
          facts["hc1_over_hc0"] < 1e-12,
          f"max deviation {facts['hc1_over_hc0']:.1e}")
    check("ols exposes hc2/hc3 and they match the hand-built leverage sandwich",
          "hc2" in facts["accepted_se_types"] and "hc3" in facts["accepted_se_types"]
          and facts["hc2_vs_numpy"] < 1e-12 and facts["hc3_vs_numpy"] < 1e-12,
          f"accepted {facts['accepted_se_types']}, hc2 diff "
          f"{facts['hc2_vs_numpy']:.1e}, hc3 diff {facts['hc3_vs_numpy']:.1e}")
    check("ols: the exact ladder hc0 <= hc2 <= hc3 holds coefficient by coefficient",
          facts["hc_ladder_violation"] <= 1e-12,
          f"worst margin {facts['hc_ladder_violation']:.1e} (negative = strictly "
          "ordered)")
    check("ols: hc2/hc3 REFUSE a point with leverage exactly 1 (hc0 does not)",
          facts["unit_leverage_refused"] == ["hc2", "hc3"]
          and facts["unit_leverage_hc0_ok"],
          f"refused {facts['unit_leverage_refused']}, hc0 still finite "
          f"{facts['unit_leverage_hc0_ok']}")
    check("ols se_type menu is exactly the six lower-case names",
          list(facts["accepted_se_types"]) == list(OLS_SE_TYPES)
          and facts["rejected_se_types"] == ["HC0", "HC1", "HC2", "HC3", "HAC",
                                             "hc4"],
          f"accepted {facts['accepted_se_types']}, rejected "
          f"{facts['rejected_se_types']}")
    check("se_type leaves the point estimate untouched, and it matches numpy",
          facts["params_invariant_to_se_type"] == 0.0
          and facts["params_match_numpy"] < 1e-10,
          f"se_type diff {facts['params_invariant_to_se_type']:.1e}, "
          f"numpy diff {facts['params_match_numpy']:.1e}")

    # ---- experiment 1 ----
    ols_rows = {r["kind"]: r for r in results["ols"]["rows"]}
    iid = ols_rows["iid"]
    for name in results["ols"]["columns"]:
        p = iid["cover"][name]["cover"]
        # under iid Gaussian errors every one of the five is consistent, so a
        # nominal 95% interval must not be down at 0.92 by T=200.
        check(f"ols iid: {name} covers >= 0.92", at_least(p, 0.92, reps),
              f"{p:.3f}")
    het = ols_rows["het"]
    check("ols heteroskedastic: nonrobust is materially broken (< 0.85)",
          at_most(het["cover"]["nonrobust"]["cover"], 0.85, reps),
          f"{het['cover']['nonrobust']['cover']:.3f} "
          f"(se/sd {het['se_over_sd']['nonrobust']:.2f})")
    check("ols heteroskedastic: hc1 repairs >= 0.10 of coverage",
          gap_at_least(het["cover"]["hc1"]["cover"],
                       het["cover"]["nonrobust"]["cover"], 0.10, reps),
          f"hc1 {het['cover']['hc1']['cover']:.3f} vs "
          f"nonrobust {het['cover']['nonrobust']['cover']:.3f}")
    check("ols heteroskedastic: hc1 covers >= 0.90",
          at_least(het["cover"]["hc1"]["cover"], 0.90, reps),
          f"{het['cover']['hc1']['cover']:.3f}")
    ar = ols_rows["ar1"]
    check("ols AR(1) errors: hc1 does NOT repair (within 0.05 of nonrobust)",
          abs(ar["cover"]["hc1"]["cover"] - ar["cover"]["nonrobust"]["cover"]) < 0.05
          + 3.0 * mc_se(0.75, reps),
          f"hc1 {ar['cover']['hc1']['cover']:.3f} vs "
          f"nonrobust {ar['cover']['nonrobust']['cover']:.3f}")
    check("ols AR(1) errors: hac beats hc1 by >= 0.05",
          gap_at_least(ar["cover"]["hac auto"]["cover"],
                       ar["cover"]["hc1"]["cover"], 0.05, reps),
          f"hac {ar['cover']['hac auto']['cover']:.3f} vs "
          f"hc1 {ar['cover']['hc1']['cover']:.3f}")

    # ---- experiment 2 ----
    lev = {r["n"]: r for r in results["leverage"]["rows"]}
    # the library columns and the numpy cross-check must be the same numbers.
    # This is what licenses reading the hc2/hc3 coverage as tsecon's OWN.
    check("leverage: tsecon hc2/hc3 == the numpy hc2*/hc3* cross-check",
          results["leverage"]["max_ref_gap"] < 1e-10,
          f"max |tsecon - numpy| {results['leverage']['max_ref_gap']:.1e} over "
          f"{reps} reps x {len(lev)} sample sizes")
    check("leverage: hc0 <= hc2 <= hc3 in every replication (exact ladder)",
          results["leverage"]["max_ladder_gap"] <= 1e-12,
          f"max violation {results['leverage']['max_ladder_gap']:.1e}")
    for n, row in lev.items():
        p = row["cover"]["oracle"]["cover"]
        # exact: Gaussian errors + fixed design => the estimate is exactly
        # normal about the oracle variance, so this is 0.95 at EVERY T.
        check(f"leverage T={n}: oracle covers 0.95 (within 4 mcse)",
              abs(p - NOMINAL) <= 4.0 * mc_se(p, reps),
              f"{p:.3f} +- {mc_se(p, reps):.3f}")
        check(f"leverage T={n}: hc1 >= hc0 (algebraically nested)",
              row["cover"]["hc1"]["cover"] >= row["cover"]["hc0"]["cover"],
              f"hc1 {row['cover']['hc1']['cover']:.3f} >= "
              f"hc0 {row['cover']['hc0']['cover']:.3f}")
    small, large = min(lev), max(lev)
    check(f"leverage: hc0 improves >= 0.15 from T={small} to T={large}",
          gap_at_least(lev[large]["cover"]["hc0"]["cover"],
                       lev[small]["cover"]["hc0"]["cover"], 0.15, reps),
          f"{lev[small]['cover']['hc0']['cover']:.3f} -> "
          f"{lev[large]['cover']['hc0']['cover']:.3f}")
    check("leverage: nonrobust does NOT improve with T (inconsistent)",
          lev[large]["cover"]["nonrobust"]["cover"] < 0.75,
          f"T={small} {lev[small]['cover']['nonrobust']['cover']:.3f}, "
          f"T={large} {lev[large]['cover']['nonrobust']['cover']:.3f}")
    # This used to compare a numpy hc3* REFERENCE against tsecon's best
    # available HC, to argue hc2/hc3 were worth adding. They were added, so it
    # now measures tsecon's own hc3 against tsecon's own hc1 -- same gap, and
    # the library is on both sides of it.
    check(f"leverage T={small}: tsecon hc3 beats tsecon hc1 by >= 0.08",
          gap_at_least(lev[small]["cover"]["hc3"]["cover"],
                       lev[small]["cover"]["hc1"]["cover"], 0.08, reps),
          f"hc3 {lev[small]['cover']['hc3']['cover']:.3f} vs "
          f"hc1 {lev[small]['cover']['hc1']['cover']:.3f} "
          "-> the leverage correction, not n/(n-k), is what pays")
    check(f"leverage T={small}: hc2 sits between hc1 and hc3 in coverage",
          (lev[small]["cover"]["hc1"]["cover"]
           <= lev[small]["cover"]["hc2"]["cover"]
           <= lev[small]["cover"]["hc3"]["cover"]),
          f"hc1 {lev[small]['cover']['hc1']['cover']:.3f} <= "
          f"hc2 {lev[small]['cover']['hc2']['cover']:.3f} <= "
          f"hc3 {lev[small]['cover']['hc3']['cover']:.3f}")
    # honesty: hc3 is the best available answer at T=25, NOT a nominal one.
    check(f"leverage T={small}: even hc3 still under-covers (<= 0.92) -- recorded,"
          " not fixed",
          at_most(lev[small]["cover"]["hc3"]["cover"], 0.92, reps),
          f"{lev[small]['cover']['hc3']['cover']:.3f} against nominal 0.95, "
          f"oracle {lev[small]['cover']['oracle']['cover']:.3f}")
    check(f"leverage T={large}: hc3 costs nothing once leverage is diluted "
          "(within 0.02 of hc1)",
          abs(lev[large]["cover"]["hc3"]["cover"]
              - lev[large]["cover"]["hc1"]["cover"])
          <= 0.02 + 3.0 * mc_se(0.95, reps),
          f"hc1 {lev[large]['cover']['hc1']['cover']:.3f} vs "
          f"hc3 {lev[large]['cover']['hc3']['cover']:.3f}")

    # ---- experiment 3 ----
    hac = {r["phi"]: r for r in results["hac_slope"]["rows"]}
    check("hac slope phi=0: hac auto still covers >= 0.92 (bandwidth is cheap)",
          at_least(hac[0.0]["cover"]["hac auto"]["cover"], 0.92, reps),
          f"{hac[0.0]['cover']['hac auto']['cover']:.3f}")
    worst = hac[max(hac)]
    check(f"hac slope phi={max(hac)}: hac auto UNDER-covers (< 0.80) -- recorded,"
          " not fixed",
          at_most(worst["cover"]["hac auto"]["cover"], 0.80, reps),
          f"{worst['cover']['hac auto']['cover']:.3f}; "
          f"se/sd {worst['se_over_sd']['hac auto']:.2f}")
    check(f"hac slope phi={max(hac)}: nonrobust SE is < 0.50 of the true sd",
          worst["se_over_sd"]["nonrobust"] < 0.50,
          f"se/sd {worst['se_over_sd']['nonrobust']:.3f}")
    check(f"hac slope phi={max(hac)}: hac auto still beats nonrobust by >= 0.10",
          gap_at_least(worst["cover"]["hac auto"]["cover"],
                       worst["cover"]["nonrobust"]["cover"], 0.10, reps),
          f"{worst['cover']['hac auto']['cover']:.3f} vs "
          f"{worst['cover']['nonrobust']['cover']:.3f}")

    # ---- experiment 4 ----
    iv = results["iv"]
    strong = [r for r in iv["rows"] if r["pi"] == max(IV_STRENGTHS)]
    weak = [r for r in iv["rows"] if r["pi"] == min(IV_STRENGTHS)]
    for row in strong:
        check(f"iv strong ({row['method']}): covers >= 0.92",
              at_least(row["cover"]["cover"], 0.92, reps),
              f"{row['cover']['cover']:.3f}, median F "
              f"{row['first_stage_f']:.1f}")
        check(f"iv strong ({row['method']}): J size <= 0.12 at a true null",
              at_most(row["j_size"]["cover"], 0.12, reps),
              f"{row['j_size']['cover']:.3f}")
    for row in weak:
        check(f"iv weak ({row['method']}): under-covers (<= 0.90)",
              at_most(row["cover"]["cover"], 0.90, reps),
              f"{row['cover']['cover']:.3f}, median F "
              f"{row['first_stage_f']:.1f}")
        # the mean reported SE exceeds the true sd while the MEDIAN is far
        # below it -- the mean is a mirage from a few enormous SEs.
        check(f"iv weak ({row['method']}): mean se/sd > 1 but median se/sd < 0.75 "
              "(the mean SE is a mirage)",
              row["se_over_sd"] > 1.0 and row["median_se_over_sd"] < 0.75,
              f"mean {row['se_over_sd']:.2f}, median "
              f"{row['median_se_over_sd']:.2f}")
        check(f"iv weak ({row['method']}): sampling law is not normal "
              "(IQR < 0.7 of a normal's)",
              row["iqr_over_normal"] < 0.7,
              f"IQR/normal {row['iqr_over_normal']:.2f} "
              "(skew is not asserted -- it barely exists here)")
        # the decisive one: a FIXED-width interval at the true sd does much
        # better than the reported per-sample SEs, which localises the failure
        # in how the SE varies from sample to sample. (The oracle's own
        # distance from nominal is NOT asserted -- for a law with this little
        # tail mass under control it is not a stable statistic, which is
        # itself part of the finding.)
        check(f"iv weak ({row['method']}): a fixed-width interval at the TRUE sd "
              "covers >= 0.05 better than the reported SE",
              gap_at_least(row["oracle_cover"]["cover"], row["cover"]["cover"],
                           0.05, reps),
              f"oracle {row['oracle_cover']['cover']:.3f} vs reported "
              f"{row['cover']['cover']:.3f}")
    # iv_gmm now returns its own first-stage diagnostic. Assert the shape of
    # the contract, not just that a number came back: the ENDOGENOUS column
    # gets an entry, the exogenous constant does NOT, and the robust F tracks
    # the textbook one at the median (they are different statistics, so they
    # are not compared sample by sample).
    for row in iv["rows"]:
        if row["method"] != "2sls":
            continue
        tag = f"pi={row['pi']}"
        check(f"iv_gmm first_stage [{tag}]: reports the endogenous column only "
              "(the constant is exogenous)",
              row["lib_first_stage_cols"] == [1]
              and row["lib_first_stage_missing"] == 0.0,
              f"columns {row['lib_first_stage_cols']}, missing "
              f"{row['lib_first_stage_missing']:.3f}")
        check(f"iv_gmm first_stage [{tag}]: the robust F tracks the textbook F "
              "at the median (within 20%)",
              abs(row["lib_first_stage_f"] / row["first_stage_f"] - 1.0) < 0.20,
              f"tsecon {row['lib_first_stage_f']:.2f} vs textbook "
              f"{row['first_stage_f']:.2f}")
    mod2sls = [r for r in iv["rows"] if r["pi"] == 0.20 and r["method"] == "2sls"][0]
    # the honesty check: F > 10 is not a safety threshold. If this ever starts
    # failing because coverage rose to nominal at F ~ 10, delete it -- but do
    # not raise the threshold to make it pass.
    check("iv_gmm: a first-stage F at the rule-of-thumb 10 does NOT buy nominal "
          "coverage",
          9.0 <= mod2sls["lib_first_stage_f"] <= 13.0
          and at_most(mod2sls["cover"]["cover"], 0.93, reps),
          f"tsecon median F {mod2sls['lib_first_stage_f']:.1f}, coverage "
          f"{mod2sls['cover']['cover']:.3f}")

    ivhac = results["iv_hac"]
    # THIS ASSERTION USED TO BE ITS OWN OPPOSITE. It read "weight='hac' with
    # the DEFAULT bandwidth == weight='robust'", because `bandwidth` defaulted
    # to 0.0 and Bartlett at 0 lags IS White -- a silent no-op that the audit
    # verified bit-for-bit over 3000 replications. The default is now the
    # Newey-West rule of thumb, so what has to be locked in is the negation:
    # the default must NEVER coincide with weight='robust' again.
    check("iv_gmm weight='hac' with the DEFAULT bandwidth is NO LONGER "
          "weight='robust'",
          ivhac["min_default_gap"] > 1e-9,
          f"smallest |se diff| over {ivhac['reps']} reps "
          f"{ivhac['min_default_gap']:.1e} (largest "
          f"{ivhac['max_default_gap']:.1e}); it used to be 0.0e+00 in every one")
    check("iv_gmm: the default hac_bandwidth is the Newey-West rule "
          "floor(4 (T/100)^(2/9))",
          ivhac["reported_bandwidth"] == [ivhac["nw_rule_bandwidth"]]
          and ivhac["n_reported_bandwidths"] == 1,
          f"reported {ivhac['reported_bandwidth']}, rule "
          f"{ivhac['nw_rule_bandwidth']:.0f} at T={ivhac['n']}")
    check("iv_gmm: the default equals an EXPLICIT bandwidth=4 bit for bit "
          "(the rule, not a coincidence)",
          ivhac["max_auto_vs_bw4"] == 0.0,
          f"max |se diff| {ivhac['max_auto_vs_bw4']:.1e}")
    check("iv_gmm: an EXPLICIT bandwidth=0.0 raises ValueError instead of "
          "silently returning White",
          ivhac["zero_bandwidth_raises"],
          ivhac["zero_bandwidth_msg"][:88])
    check("iv_gmm: method='2sls' with weight='hac' raises (2SLS fixes its own "
          "weight)",
          ivhac["twosls_hac_raises"], ivhac["twosls_hac_msg"][:88])
    check("iv_gmm: weight='robust' reports hac_bandwidth=None (no truncation "
          "to report)",
          ivhac["robust_bandwidth_is_none"],
          f"{ivhac['robust_bandwidth_is_none']}")
    check("iv_gmm: passing bandwidth=10 buys >= 0.10 of coverage under AR(1)",
          gap_at_least(ivhac["cover"]["hac bw=10"]["cover"],
                       ivhac["cover"]["robust"]["cover"], 0.10, reps),
          f"{ivhac['cover']['robust']['cover']:.3f} -> "
          f"{ivhac['cover']['hac bw=10']['cover']:.3f}")
    # the fix is a default, NOT a remedy. The rule of thumb picks 4 lags at
    # T=250 -- fewer than the 10 that already left a large shortfall -- so the
    # default must still under-cover, and this assertion says so on the record.
    check("iv_gmm: the NEW default still UNDER-covers (<= 0.90) -- a working "
          "default is not a fix",
          at_most(ivhac["cover"][IV_HAC_DEFAULT]["cover"], 0.90, reps),
          f"{ivhac['cover'][IV_HAC_DEFAULT]['cover']:.3f} at bandwidth "
          f"{ivhac['nw_rule_bandwidth']:.0f}, vs "
          f"{ivhac['cover']['hac bw=10']['cover']:.3f} at bandwidth 10, "
          "nominal 0.95")
    check("iv_gmm: the rule-of-thumb bandwidth is SHORTER than 10, so it covers "
          "no better",
          ivhac["nw_rule_bandwidth"] < 10.0
          and ivhac["cover"][IV_HAC_DEFAULT]["cover"]
          <= ivhac["cover"]["hac bw=10"]["cover"] + 3.0 * mc_se(0.9, reps),
          f"rule {ivhac['nw_rule_bandwidth']:.0f} -> "
          f"{ivhac['cover'][IV_HAC_DEFAULT]['cover']:.3f}, bw=10 -> "
          f"{ivhac['cover']['hac bw=10']['cover']:.3f}")

    # ---- experiment 5 ----
    har = {(r["het"], r["maxlags"]): r for r in results["har"]["rows"]}
    for het in (False, True):
        row = har[(het, 5)]
        tag = "het" if het else "iid"
        for j, name in enumerate(HAR_NAMES[1:], start=1):
            check(f"har_rv {tag} maxlags=5: {name} covers >= 0.90",
                  at_least(row["cover"][j]["cover"], 0.90, reps),
                  f"{row['cover'][j]['cover']:.3f}")
        check(f"har_rv {tag} maxlags=5: the CONSTANT covers worse than b_daily",
              gap_at_least(row["cover"][1]["cover"], row["cover"][0]["cover"],
                           0.02, reps),
              f"b_daily {row['cover'][1]['cover']:.3f} vs "
              f"const {row['cover'][0]['cover']:.3f}")
        # prove the explanation instead of asserting it is plausible
        pred = row["predicted_const_bias"]
        meas = row["bias"][0]
        check(f"har_rv {tag}: const bias == -mu * sum(slope bias) within 30%",
              meas < 0 and abs(pred / meas - 1.0) < 0.30,
              f"measured {meas:+.4f}, predicted {pred:+.4f}")
        check(f"har_rv {tag}: more HAC lags => smaller SE (bandwidth is not free)",
              har[(het, 0)]["se_over_sd"][1] > har[(het, 22)]["se_over_sd"][1],
              f"se/sd b_daily: lag0 {har[(het, 0)]['se_over_sd'][1]:.3f} > "
              f"lag22 {har[(het, 22)]['se_over_sd'][1]:.3f}")

    # ---- experiment 6 ----
    pr = {(r["case"], r["n"]): r for r in results["probit"]["rows"]}
    common = [r for r in results["probit"]["rows"] if "common" in r["case"]]
    for row in common:
        check(f"recession_probit common T={row['n']}: covers >= 0.92",
              at_least(row["cover"]["cover"], 0.92, max(row["n_ok"], 1)),
              f"{row['cover']['cover']:.3f} on {row['n_ok']} survivors")
    rare = [r for r in results["probit"]["rows"]
            if "rare" in r["case"] and r["n"] == min(PROBIT_SIZES)]
    for row in rare:
        check(f"recession_probit {row['link']} rare T={row['n']}: >= 10% of "
              "replications have NO finite MLE",
              row["fail_share"] >= 0.10,
              f"fail {row['fail_share']:.3f} "
              f"(degenerate {row['fail_degenerate']:.3f}, "
              f"separation {row['fail_separation']:.3f})")
        check(f"recession_probit {row['link']} rare T={row['n']}: median bias is "
              "AWAY from zero",
              row["median_bias"] > 0.0, f"{row['median_bias']:+.4f}")
    # iterate PROBIT_CASES, not a set of the labels -- set order depends on
    # PYTHONHASHSEED and would reorder the printed report between runs.
    for case in [label for label, *_ in PROBIT_CASES]:
        b_small = pr[(case, min(PROBIT_SIZES))]["median_bias"]
        b_large = pr[(case, max(PROBIT_SIZES))]["median_bias"]
        check(f"recession_probit '{case.strip()}': median bias shrinks with T",
              abs(b_large) < abs(b_small),
              f"T={min(PROBIT_SIZES)} {b_small:+.4f} -> "
              f"T={max(PROBIT_SIZES)} {b_large:+.4f}")
        check(f"recession_probit '{case.strip()}': no failures at T="
              f"{max(PROBIT_SIZES)}",
              pr[(case, max(PROBIT_SIZES))]["fail_share"] == 0.0,
              f"{pr[(case, max(PROBIT_SIZES))]['fail_share']:.3f}")

    # ---- experiment 7 ----
    qr = {r["design"]: r for r in results["quantile"]["rows"]}
    median_idx = list(QR_TAUS).index(0.50)
    lo_idx = 0
    for label, row in qr.items():
        p_med = row["cover_slope"][median_idx]["cover"]
        check(f"quantile_regression [{label.strip()}]: tau=0.50 covers >= 0.90",
              at_least(p_med, 0.90, reps), f"{p_med:.3f}")
        p_lo = row["cover_slope"][lo_idx]["cover"]
        check(f"quantile_regression [{label.strip()}]: tau=0.05 covers no better "
              "than the median",
              p_lo <= p_med + 3.0 * np.sqrt(mc_se(p_lo, reps) ** 2
                                            + mc_se(p_med, reps) ** 2),
              f"tau=0.05 {p_lo:.3f} vs tau=0.50 {p_med:.3f}")
    short = qr["location-scale T=200"]
    long_ = qr["location-scale T=1000"]
    check("quantile_regression: tau=0.05 under-covers at T=200 (<= 0.92)",
          at_most(short["cover_slope"][lo_idx]["cover"], 0.92, reps),
          f"{short['cover_slope'][lo_idx]['cover']:.3f}; "
          f"se/sd {short['se_over_sd'][lo_idx]:.2f}")
    check("quantile_regression: the tau=0.05 Powell SE is < 0.95 of the true sd "
          "at T=200",
          short["se_over_sd"][lo_idx] < 0.95,
          f"se/sd {short['se_over_sd'][lo_idx]:.3f}")
    check("quantile_regression: tau=0.05 coverage improves from T=200 to T=1000",
          gap_at_least(long_["cover_slope"][lo_idx]["cover"],
                       short["cover_slope"][lo_idx]["cover"], 0.0, reps),
          f"{short['cover_slope'][lo_idx]['cover']:.3f} -> "
          f"{long_['cover_slope'][lo_idx]['cover']:.3f}")
    for label, row in qr.items():
        if not (row["cover_converged"] and row["cover_nonconverged"]):
            continue
        # the `converged` flag is an iteration cap, not a failure: dropping the
        # tripped replications must not move the tail coverage materially.
        c_conv = row["cover_converged"][lo_idx]["cover"]
        c_all = row["cover_slope"][lo_idx]["cover"]
        n_conv = reps - row["nonconverged"]
        check(f"quantile_regression [{label.strip()}]: dropping non-converged reps "
              "does not move tau=0.05 coverage",
              abs(c_conv - c_all) <= 0.03 + 3.0 * mc_se(c_all, n_conv),
              f"all {c_all:.3f} vs converged-only {c_conv:.3f} "
              f"({row['nonconverged']}/{reps} tripped the cap)")

    header("assertions")
    width = max(len(label) for label, _, _ in checks) + 2
    failed = 0
    for label, ok, detail in checks:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{flag}] {label:<{width}} {detail}")
    print()
    print(f"{len(checks) - failed}/{len(checks)} assertions passed "
          f"(slack: 3 Monte Carlo standard errors at reps={reps})")
    return failed


# ==========================================================================
# findings -- the honest summary, printed every run
# ==========================================================================
def findings(results, reps):
    header("FINDINGS -- where these intervals do not keep their promise")
    ols_rows = {r["kind"]: r for r in results["ols"]["rows"]}
    lev = {r["n"]: r for r in results["leverage"]["rows"]}
    hac = {r["phi"]: r for r in results["hac_slope"]["rows"]}
    har = {(r["het"], r["maxlags"]): r for r in results["har"]["rows"]}
    qr = {r["design"]: r for r in results["quantile"]["rows"]}
    weak = [r for r in results["iv"]["rows"] if r["pi"] == min(IV_STRENGTHS)
            and r["method"] == "2sls"][0]
    moderate = [r for r in results["iv"]["rows"] if r["pi"] == 0.20
                and r["method"] == "2sls"][0]
    rare = [r for r in results["probit"]["rows"]
            if "probit rare" in r["case"] and r["n"] == min(PROBIT_SIZES)][0]
    ivhac = results["iv_hac"]
    small = min(lev)
    worst_phi = max(hac)

    print("UNDER-COVERS -- the approximation, not the estimator. No fix exists;")
    print("these are properties of the asymptotics and should be documented.")
    print(f"  * ols hac on a SLOPE, x and e both AR(1) at phi={worst_phi}: "
          f"{hac[worst_phi]['cover']['hac auto']['cover']:.3f} at nominal 0.95")
    print(f"    (automatic bandwidth), {hac[worst_phi]['cover']['hac lag24']['cover']:.3f}"
          f" at 24 lags. The reported SE is "
          f"{hac[worst_phi]['se_over_sd']['hac auto']:.2f} of the true sampling sd:")
    print("    T=200 does not contain enough independent information to estimate a")
    print("    long-run variance this large. Worst number in the file.")
    print(f"  * quantile_regression tau=0.05, T=200: "
          f"{qr['location-scale T=200']['cover_slope'][0]['cover']:.3f}, with the")
    print(f"    Powell SE at {qr['location-scale T=200']['se_over_sd'][0]:.2f} of the")
    print("    true sd. The sandwich needs a conditional density estimated from the")
    print("    handful of points near an extreme quantile. Shrinks with T "
          f"({qr['location-scale T=1000']['cover_slope'][0]['cover']:.3f} at T=1000)")
    print("    but is not closed. Bootstrap the quantile process instead.")
    print(f"  * ols t(3) errors: every se_type loses a little "
          f"({ols_rows['t3']['cover']['hc0']['cover']:.3f} for hc0), and the robust")
    print("    ones lose slightly more. t(3) has no fourth moment, which is exactly")
    print("    what the sandwich's asymptotics assume.")
    print()
    print("UNDER-COVERS -- the ESTIMATOR is wrong for the job. Fixable by the caller.")
    print(f"  * nonrobust under heteroskedasticity: "
          f"{ols_rows['het']['cover']['nonrobust']['cover']:.3f} at T=200, and in the")
    print(f"    high-leverage design it is stuck at "
          f"{lev[max(lev)]['cover']['nonrobust']['cover']:.3f} even at T={max(lev)}.")
    print("    It is inconsistent; data does not help. Use hc0/hc1 -- or hc2/hc3")
    print("    when T is small and a few points carry the leverage.")
    print(f"  * hc0/hc1 under serial correlation: "
          f"{ols_rows['ar1']['cover']['hc1']['cover']:.3f}, indistinguishable from")
    print("    nonrobust. HC is not serial-correlation robust. Use hac.")
    print(f"  * iv_gmm with weak instruments (median F "
          f"{weak['first_stage_f']:.1f}): {weak['cover']['cover']:.3f}. The MEAN")
    print(f"    reported SE looks generous (se/sd {weak['se_over_sd']:.2f}) but that")
    print(f"    is a mirage -- the MEDIAN reported SE is only "
          f"{weak['median_se_over_sd']:.2f} of the true")
    print("    sampling sd, and the mean is dragged up by a few replications with")
    print(f"    enormous SEs (the estimate's IQR is only "
          f"{weak['iqr_over_normal']:.2f} of a normal's with")
    print("    the same sd). A FIXED-width interval at the true sd covers "
          f"{weak['oracle_cover']['cover']:.3f},")
    print("    so the damage is done by how the SE varies across samples -- it is")
    print("    smallest where the estimate is worst. That oracle is not available in")
    print("    practice, so the fix is a different interval, not a better SE:")
    print("    a weak-instrument-robust one (Anderson-Rubin), which tsecon")
    print(f"    does not expose. Even at median F {moderate['first_stage_f']:.0f} -- the")
    print(f"    conventional rule-of-thumb threshold -- coverage is already "
          f"{moderate['cover']['cover']:.3f}")
    print(f"    with a median se/sd of {moderate['median_se_over_sd']:.2f}.")
    print(f"  * har_rv CONSTANT: {har[(False, 5)]['cover'][0]['cover']:.3f} against")
    print(f"    nominal 0.95, bias {har[(False, 5)]['bias'][0]:+.4f}. Not the SE --")
    print("    the least-squares persistence bias at sum(b)=0.95 is pushed into the")
    print("    constant, and the prediction -mu*sum(slope bias) = "
          f"{har[(False, 5)]['predicted_const_bias']:+.4f} matches. The three slopes")
    print("    are fine; do not read the HAR intercept as if it were.")
    print()
    print("OVER-COVERS -- conservative, which is its own kind of wrong.")
    print(f"  * recession_probit, rare events (rate {rare['event_rate']:.3f}), T="
          f"{rare['n']}: coverage {rare['cover']['cover']:.3f} among survivors, but")
    print(f"    {rare['fail_share']:.1%} of replications have NO finite MLE at all")
    print(f"    ({rare['fail_degenerate']:.1%} no recession months, "
          f"{rare['fail_separation']:.1%} separation). The library correctly refuses")
    print("    instead of inventing an interval -- but the coverage you can measure")
    print("    is conditional on the sample being informative, and the surviving")
    print(f"    intervals are wide: {rare['wide_share']:.1%} have an SE over 3x the")
    print("    median. Read the fail column with the coverage column, always.")
    print(f"  * quantile_regression INTERCEPT in the location-scale designs: "
          f"{qr['location-scale T=200']['cover_icpt'][3]['cover']:.3f} at tau=0.50.")
    print("    x ~ U(0,2), so the intercept is an extrapolation to the edge of the")
    print("    support and its sandwich SE is conservative.")
    print()
    print("FIXED SINCE THE LAST AUDIT -- what the fix bought, and what it did NOT.")
    print("  * iv_gmm(weight=\"hac\") was a SILENT NO-OP. `bandwidth` defaulted to")
    print("    0.0 and a Bartlett kernel truncated at zero lags IS the White")
    print("    estimator, so it returned results bit-identical to weight=\"robust\"")
    print("    while the caller believed they had serial-correlation robustness.")
    print("    The default is now bandwidth=None -> the Newey-West rule of thumb")
    print(f"    floor(4 (T/100)^(2/9)) = {ivhac['nw_rule_bandwidth']:.0f} at "
          f"T={ivhac['n']}, returned as `hac_bandwidth`;")
    print("    an explicit bandwidth=0.0 raises; and method=\"2sls\" with")
    print("    weight=\"hac\" raises, because 2SLS fixes its weight at (Z'Z/n)^-1")
    print("    by construction, so accepting a weight there was the same no-op.")
    print("    Smallest |se(default) - se(robust)| over "
          f"{ivhac['reps']} replications is now")
    print(f"    {ivhac['min_default_gap']:.1e}; it used to be exactly 0 in every"
          " one.")
    print("    IT DOES NOT RESTORE COVERAGE, and that is the point worth carrying:")
    print(f"    the rule picks {ivhac['nw_rule_bandwidth']:.0f} lags, FEWER than the"
          " 10 that already left a")
    print(f"    shortfall. Default {ivhac['cover'][IV_HAC_DEFAULT]['cover']:.3f}, "
          f"bw=10 {ivhac['cover']['hac bw=10']['cover']:.3f}, robust "
          f"{ivhac['cover']['robust']['cover']:.3f}, nominal 0.95. A sensible")
    print("    default, not a remedy -- pick the bandwidth for the persistence you")
    print("    actually have.")
    print("  * ols gained se_type=\"hc2\" and \"hc3\", so the leverage correction is")
    print(f"    library output rather than a reference. At T={small} in the")
    print(f"    high-leverage design tsecon's own hc3 covers "
          f"{lev[small]['cover']['hc3']['cover']:.3f} and hc2 "
          f"{lev[small]['cover']['hc2']['cover']:.3f},")
    print(f"    against hc1's {lev[small]['cover']['hc1']['cover']:.3f} -- "
          f"{100 * (lev[small]['cover']['hc3']['cover'] - lev[small]['cover']['hc1']['cover']):.0f}"
          " points that n/(n-k) could never buy at")
    print("    k=2. Both reproduce a hand-built r^2/(1-h) sandwich to "
          f"{results['leverage']['max_ref_gap']:.0e}, and a")
    print("    point with leverage exactly 1 is REFUSED rather than returned as a")
    print(f"    near-infinite SE. hc3 is still NOT nominal at T={small} "
          f"({lev[small]['cover']['hc3']['cover']:.3f} against")
    print(f"    0.95, with the oracle at {lev[small]['cover']['oracle']['cover']:.3f}):"
          " it is the best available answer")
    print(f"    in that design, not a correct one. By T={max(lev)} it is"
          " indistinguishable")
    print(f"    from hc1 ({lev[max(lev)]['cover']['hc3']['cover']:.3f} vs "
          f"{lev[max(lev)]['cover']['hc1']['cover']:.3f}), so the correction is close"
          " to free asymptotically.")
    print("  * iv_gmm now returns `first_stage`: a robust per-regressor F as a list")
    print("    of dicts (regressor, fstat, dof_num, dof_den, pval). Index it by")
    print("    `regressor`, NOT by position -- entries are OMITTED where the")
    print("    statistic is undefined (exogenous regressor, no excluded")
    print("    instruments, rank-deficient Z), so a missing entry means UNDEFINED,")
    print(f"    not a failed fit. Here it reports on columns "
          f"{weak['lib_first_stage_cols']} only; the constant")
    print("    correctly gets none. It is NOT a weak-identification test with two")
    print("    or more endogenous regressors -- all of them can clear 10 while the")
    print("    system is under-identified -- and even with one, F > 10 is not a")
    print(f"    safety threshold: at a robust median F of "
          f"{moderate['lib_first_stage_f']:.1f} coverage is "
          f"{moderate['cover']['cover']:.3f}.")
    print("    Angrist-Pischke, Cragg-Donald / Kleibergen-Paap against Stock-Yogo,")
    print("    and Anderson-Rubin sets are all still missing.")
    print()
    print("API TRAPS -- correct code, misleading surface.")
    print(f"  * ols se_type is lower-case only: 'HC0' and 'HAC' raise ValueError.")
    print(f"  * quantile_regression's `converged` flag trips on "
          f"{qr['location-scale T=200']['nonconverged']}/{reps} replications at")
    print("    T=200 -- it is the IRLS iteration cap, one flag shared across all")
    print("    taus, not an estimation failure. Dropping those replications moves")
    print(f"    tau=0.05 coverage from "
          f"{qr['location-scale T=200']['cover_slope'][0]['cover']:.3f} to "
          f"{qr['location-scale T=200']['cover_converged'][0]['cover']:.3f}, so the")
    print("    tables above keep them. A per-tau flag would be more useful.")
    print()
    print("NOT BROKEN -- worth saying explicitly.")
    print(f"  * Under iid Gaussian errors all {len(results['ols']['columns'])} "
          "columns of experiment 1 sit at")
    print("    nominal at T=200, and every HC column of experiment 2 does by "
          f"T={max(lev)}")
    print(f"    (hc0 {lev[max(lev)]['cover']['hc0']['cover']:.3f}, "
          f"hc1 {lev[max(lev)]['cover']['hc1']['cover']:.3f}, "
          f"hc2 {lev[max(lev)]['cover']['hc2']['cover']:.3f}, "
          f"hc3 {lev[max(lev)]['cover']['hc3']['cover']:.3f}).")
    print("  * hc0/hc1 recover essentially all of the heteroskedasticity loss:")
    print(f"    {ols_rows['het']['cover']['nonrobust']['cover']:.3f} -> "
          f"{ols_rows['het']['cover']['hc1']['cover']:.3f}.")
    print("  * tsecon's hc2/hc3 reproduce a hand-built leverage sandwich to "
          f"{results['leverage']['max_ref_gap']:.0e}")
    print("    on every replication, and the exact ladder hc0 <= hc2 <= hc3 holds")
    print("    coefficient by coefficient in every sample.")
    print("  * har_rv's three slope coefficients cover at nominal under both iid")
    print("    and heteroskedastic innovations.")
    print("  * recession_probit's common-regime intervals cover at nominal from")
    print("    T=100 up.")
    print("  * Hansen J is close to correctly sized with strong instruments.")
    print()
    print(f"Monte Carlo standard error at reps={reps} and p=0.95: "
          f"{mc_se(0.95, reps):.4f}. Every coverage number above is printed as")
    print("cover+-mcse, so a 0.93 and a 0.95 can be told apart.")


# ==========================================================================
# driver
# ==========================================================================
def run(quick=False, reps=None):
    if reps is None:
        reps = REPS_QUICK if quick else REPS_FULL
    started = time.perf_counter()

    print(__doc__.splitlines()[0])
    print()
    print(f"seed = {SEED}   reps = {reps}   nominal = {NOMINAL}   "
          f"z = {Z:.6f}   mode = {'quick' if quick else 'full'}")
    print(f"Monte Carlo standard error at p=0.95: {mc_se(0.95, reps):.4f}")
    if quick:
        print("QUICK MODE: the MC standard error is large; read the full run for")
        print("any number you intend to quote.")

    facts = structural_checks()
    report_structural(facts)

    results = {}
    results["ols"] = exp_ols_se_types(reps)
    report_ols_se_types(results["ols"])

    results["leverage"] = exp_hc_leverage(reps)
    report_hc_leverage(results["leverage"])

    results["hac_slope"] = exp_hac_slope(reps)
    report_hac_slope(results["hac_slope"])

    results["iv"] = exp_iv_strength(reps)
    report_iv_strength(results["iv"])

    results["iv_hac"] = exp_iv_hac(reps)
    report_iv_hac(results["iv_hac"])

    results["har"] = exp_har_hac(reps)
    report_har_hac(results["har"])

    results["probit"] = exp_probit_wald(reps)
    report_probit_wald(results["probit"])

    results["quantile"] = exp_quantile_powell(reps)
    report_quantile_powell(results["quantile"])

    findings(results, reps)

    failed = assertions(results, facts, reps)
    elapsed = time.perf_counter() - started
    print()
    print(f"total runtime {elapsed:.1f}s (seed {SEED}, reps {reps})")
    if failed:
        raise AssertionError(f"{failed} coverage assertion(s) failed")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                       help=f"fast smoke run ({REPS_QUICK} replications)")
    parser.add_argument("--reps", type=int, default=None,
                       help="override the replication count")
    args = parser.parse_args()
    run(quick=args.quick, reps=args.reps)


if __name__ == "__main__":
    main()
