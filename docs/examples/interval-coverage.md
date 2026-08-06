# Interval coverage

A confidence interval is a **promise about repeated samples**: across
independent draws from the same data-generating process, a nominal 95% interval
should contain the truth 95% of the time. Every other validation tier in this
repository proves that `tsecon`'s *point estimates and standard-error algebra*
match an independent reference. None of them proves the promise itself.

This page is that proof, and it is an audit rather than an advertisement.
**40 interval-valued outputs across 21 functions** — a function paired with the
option and regime that change its answer — were re-estimated on thousands of
seeded draws from data-generating processes whose truth is known in closed form,
and the containment rate was counted. Where an interval covers at its nominal rate, the
number is here. Where it does not, the number is *also* here, with the
Monte Carlo standard error of the measurement next to it, an attribution of
*why*, and what to do instead.

!!! warning "The three headline results, before any table"

    1. **8 of the 32 frequentist intervals miss their nominal rate even in the
       design they are entitled to do well on.** That list is
       [below](#a-misses-even-in-the-favourable-design), and it is the most
       important output of this work.
    2. **The delta-method VAR IRF band loses coverage monotonically in the
       horizon** — 90% nominal, 89.7% at impact, **67.3% at h=12** with T=100 —
       and the standard error is *not* the problem. It is the normal
       approximation to a badly skewed statistic.
       [The table is here](#the-delta-method-irf-band-horizon-by-horizon).
    3. **A pointwise band is not a joint band, and the gap is enormous.** A
       nominal 90% pointwise IRF band contains the *whole* 13-horizon path in
       **72.2%** of samples at T=500; nominal 95% marginal VAR forecast bands
       contain every horizon and series at once in **40.9%** at T=100 — and
       still only 48.1% at T=800, because the joint rate does not converge to
       the marginal one. No function in this library reports a simultaneous
       band.

---

## Fixed in 0.2.0

This audit was written against `0.1.0`. Four of its findings were defects rather
than approximation error, and `0.2.0` closed all four. The rows they came from
are still in the tables below, annotated — an audit that deletes its findings
once they are fixed leaves nothing to check later, and the *sizes* of these gaps
are the argument for measuring coverage at all.

| finding | what shipped | measured effect |
|---|---|---|
| `ols` had no `hc2`/`hc3` | both added; match statsmodels to 2.96e-15 | T=25 leverage design: **0.682 → 0.863**. Still short of nominal |
| `iv_gmm(weight="hac")` was a silent no-op at its default `bandwidth=0.0` | default is now the Newey-West rule; explicit `0.0` raises; the truncation used is returned as `hac_bandwidth` | **0.632 → 0.842**. A working default, *not* a remedy — `bandwidth=10` still only reaches 0.868 |
| `iv_gmm` reported no first-stage F | `first_stage` returned per instrumented regressor | diagnostic only; coverage unchanged. F > 10 is still not a safe threshold (0.915 at median F = 10.5) |
| `arima_fit(d=1)` omitted the drift-uncertainty term | `drift_uncertainty=True` adds it via the delta method; default unchanged so the statsmodels golden survives | h=24, T=60: **0.902 → 0.945** |

**None of these repaired an approximation.** Each removed a case where the
library returned something other than what the caller asked for, or withheld a
number they needed to judge it. The approximation gaps on this page — the
delta-method IRF band decaying in the horizon, pointwise-is-not-joint, HAC under
persistence — are all still here, and are not bugs to be fixed.

Two of the four were breaking. Anyone who called `iv_gmm(weight="hac")` before
`0.2.0` received White standard errors; their published numbers will move.

---

## Why a golden fixture cannot prove this

A golden test loads one dataset, runs the estimator, and asserts the output
matches a reference implementation to 1e-8. That is a strong claim about
arithmetic and a *null* claim about statistics. Two calls can each reproduce
their reference to machine precision and still have completely different
coverage, because coverage is a property of the *sampling distribution* — it
depends on the sample size, the persistence, the horizon, and the leverage of
the design, none of which appear in a fixture.

Concretely: `iv_gmm(weight="hac")` with its default bandwidth is
**bit-identical** to `weight="robust"` — max absolute difference `0.000e+00`
across 3000 replications. Both are computed exactly as documented. One of them
covers 63.2% when the caller believes they asked for serial-correlation
robustness. No fixture can see that; only simulation can.

This is [Tier 6](../reference/testing.md#tier-6-interval-coverage) of the
[testing map](../reference/testing.md), and it sits next to
[Tier 5, Monte Carlo validation](monte-carlo.md), which does the same job for
*test size* and *estimator consistency*.

---

## Reproducing every number

```sh
# the whole audit: five families, one consolidated report
.venv/bin/python docs/examples/coverage/run_all.py            # 376 s here

# just the consolidated tables, without the five per-family reports
.venv/bin/python docs/examples/coverage/run_all.py --summary

# a smoke run; the MC standard errors are 2-4x larger, so do not quote it
.venv/bin/python docs/examples/coverage/run_all.py --quick     # ~40 s

# or one family at a time
.venv/bin/python docs/examples/coverage/regression_se.py       #  41 s
.venv/bin/python docs/examples/coverage/irf_bands.py           #  55 s
.venv/bin/python docs/examples/coverage/lp_family.py           #  67 s
.venv/bin/python docs/examples/coverage/forecast_intervals.py  # 113 s
.venv/bin/python docs/examples/coverage/bayes_and_sets.py      #  99 s
```

Every family draws from one master seed, `20260729`, and prints it. Every draw
is `default_rng([seed, experiment, replication])`, so the numbers do not depend
on call ordering, on `PYTHONHASHSEED`, or on whether you run one family or all
five — verified: every measured line from the five standalone runs appears
identically in the consolidated run. Each family asserts its own qualitative
findings and **exits non-zero if they stop holding**, so a statistical
regression fails a build rather than quietly rotting in this page.

```text
  module                   assertions   runtime   result
  ----------------------------------------------------------------------
  regression_se.py            87 pass     41.1s   OK
  irf_bands.py                27 pass     56.4s   OK
  lp_family.py                16 pass     68.1s   OK
  forecast_intervals.py        9 pass    111.2s   OK
  bayes_and_sets.py           23 pass     98.9s   OK

  39 probes harvested from 5 families in 376.2s
```

The coverage numbers are byte-reproducible; the runtimes are wall clock and will
not be.

---

## How to read the tables

**Monte Carlo standard error.** Every coverage number is printed as
`p ± se` with `se = sqrt(p(1−p)/reps)`. At `reps=3000` and `p=0.95` that is
`0.0040`, so 0.93 and 0.95 are five standard errors apart and can be told
apart honestly; at `reps=400` it is `0.015` and they cannot. The `dev` column
is `(coverage − nominal) / se`. A verdict of **UNDER** means `dev < −3`,
**OVER** means `dev > +3`. That is a statement about *statistical
distinguishability*, not about importance — which is why `gap pp` is printed
next to it. At `reps=3000`, both 0.934 and 0.588 are "UNDER"; they are 1.6pp
and 36pp, and they are not the same finding.

**`kind` — not everything shaped like an interval makes a coverage promise.**

| kind | what it is | is nominal coverage owed? |
|---|---|---|
| **CI** | frequentist confidence interval for a *parameter* | yes |
| **PRED** | predictive interval for a future *realisation* | yes |
| **CRED** | Bayesian credible band — a statement about the posterior | **no.** A shortfall measures the prior, not a defect |
| **SET** | set-identified bounds (sign restrictions) — not an interval about a point at all | **no.** The question is whether the identified *set* contains the truth |

Measuring frequentist coverage of a credible band is still informative — it
tells you how much work the prior is doing — but it answers a *different
question*, and this page labels it as such everywhere. Those rows are
segregated into [group C](#c-objects-that-make-no-frequentist-promise).

**`cause` — a miss is one of five things, and they need different responses.**

| cause | meaning | who can fix it |
|---|---|---|
| **APPROXIMATION** | the formula is right; its asymptotics have not arrived at this sample size, horizon, or persistence | nobody. Widen deliberately, or use a different kind of interval |
| **ESTIMATOR** | the estimator is wrong for the job, or is off-centre, so no standard error rescues it | the caller — or nobody, when it is inconsistency |
| **CONVENTION** | a deliberate library default (a bandwidth, a degrees-of-freedom choice, a discreteness padding) with a measured coverage cost | the caller, by overriding it |
| **API GAP** | the interval that would fix it is not exposed at all | the library. These are the actionable recommendations |
| **READING** | the interval is correct and the reader's question was different | the reader |

**Coverage is DGP-specific.** Every number below is conditional on the
data-generating process that produced it. The processes were chosen to be
*canonical* — the textbook cases a reader will recognise, plus the stress cases
applied macroeconomics actually lives in — and they are **not exhaustive**.
A number here is evidence about a mechanism, not a universal constant for the
function. The [DGPs are stated in full](#the-data-generating-processes) below,
and each family module's docstring derives its truth in closed form.

---

## The headline: 39 measured interval-valued outputs

Each row names **two** measured cells. `favourable` is a design the interval is
entitled to do well on; `stress` is a design that pushes the same interval
where applied work goes. Reading them together is what separates "this
asymptotic approximation degrades, as asymptotic approximations do" from "this
interval does not work".

### Table 1 — the favourable case

*A miss here is not the user's data.*

| surface | interval / option | kind | nom | design measured | coverage ± MC se | dev | gap pp | verdict |
|---|---|---|---|---|---|---|---|---|
| `tsecon.ols` | se_type="nonrobust" | CI | 0.95 | iid Gaussian, T=200 | 0.946 ± 0.004 | -1.0 | -0.4 | at nominal |
| `tsecon.ols` | se_type="hc1" | CI | 0.95 | heteroskedastic sd=\|x\|, T=200 | 0.942 ± 0.004 | -1.9 | -0.8 | at nominal |
| `tsecon.ols` | se_type="hc1"; small T, leverage | CI | 0.95 | x~chi2(1), sd(e\|x)=x, T=1600 | 0.940 ± 0.004 | -2.4 | -1.0 | at nominal |
| `tsecon.ols` | se_type="hac" | CI | 0.95 | x,e AR(1) phi=0.0, T=200 | 0.944 ± 0.004 | -1.5 | -0.6 | at nominal |
| `tsecon.iv_gmm` | 2sls / 2step / iterated | CI | 0.95 | median first-stage F = 90.1 | 0.946 ± 0.004 | -1.0 | -0.4 | at nominal |
| `tsecon.iv_gmm` | weight="hac" | CI | 0.95 | AR(1) errors phi=0.8, hac bw=10 | 0.868 ± 0.006 | -13.2 | -8.2 | **UNDER** |
| `tsecon.har_rv` | HAC SEs on the three slopes | CI | 0.95 | iid innovations, maxlags=5, b_daily | 0.945 ± 0.004 | -1.2 | -0.5 | at nominal |
| `tsecon.har_rv` | HAC SE on the CONSTANT | CI | 0.95 | het innovations, maxlags=0, const | 0.910 ± 0.005 | -7.7 | -4.0 | **UNDER** |
| `tsecon.recession_probit` | Wald interval | CI | 0.95 | probit common phi=0.9, T=250 | 0.952 ± 0.004 | +0.5 | +0.2 | at nominal |
| `tsecon.quantile_regression` | Powell sandwich, slope | CI | 0.95 | location-scale T=200, tau=0.5 | 0.940 ± 0.004 | -2.3 | -1.0 | at nominal |
| `tsecon.quantile_regression` | Powell sandwich, intercept | CI | 0.95 | homoskedastic  T=200, tau=0.5 | 0.958 ± 0.004 | +2.3 | +0.8 | at nominal |
| `tsecon.var_irf_bands` | method="asymptotic" | CI | 0.90 | n=500, h=0 | 0.909 ± 0.006 | +1.5 | +0.9 | at nominal |
| `tsecon.var_irf_bands` | method="bootstrap" | CI | 0.90 | n=100, h=0 | 0.848 ± 0.008 | -6.5 | -5.2 | **UNDER** |
| `tsecon.var_irf_bands` | method="bootstrap", bias_correct=True | CI | 0.90 | n=100, h=12 | 0.900 ± 0.007 | +0.0 | +0.0 | at nominal |
| `tsecon.var_irf_bands` | cumulative=True | CI | 0.90 | orth=True,cumulative=True, h=12 | 0.884 ± 0.007 | -2.2 | -1.6 | at nominal |
| `tsecon.var_irf_bands` | with the lag order wrong | CI | 0.90 | n=500, fit as VAR(4) [correct], h=4 | 0.905 ± 0.007 | +0.8 | +0.5 | at nominal |
| `tsecon.var_irf_bands` | pointwise band read as a JOINT band | CI | 0.90 | n=500, h=0 | 0.909 ± 0.006 | +1.5 | +0.9 | at nominal |
| `tsecon.lp` | se="lag_augmented" (the default) | CI | 0.95 | lag_augmented, h=0 | 0.945 ± 0.004 | -1.3 | -0.5 | at nominal |
| `tsecon.lp` | se="hac" | CI | 0.95 | T=800 hac, h=12 | 0.939 ± 0.005 | -2.1 | -1.1 | at nominal |
| `tsecon.lp_iv` | strong instrument | CI | 0.95 | strong iv, best h (1) | 0.930 ± 0.005 | -4.2 | -2.0 | **UNDER** |
| `tsecon.lp_iv` | weak instrument | CI | 0.95 | weak iv, closest h (0) | 0.969 ± 0.003 | +6.1 | +1.9 | **OVER** |
| `tsecon.lp_state` | per-regime response | CI | 0.95 | state0 lag_augmented, best h (1) | 0.953 ± 0.005 | +0.6 | +0.3 | at nominal |
| `tsecon.lp_multiplier` | integral multiplier | CI | 0.95 | multiplier, h=0 | 0.934 ± 0.005 | -3.5 | -1.6 | **UNDER** |
| `tsecon.smooth_lp` | lam="cv" (the default) | CI | 0.95 | lam=0, h=0 | 0.936 ± 0.009 | -1.5 | -1.4 | at nominal |
| `tsecon.arima_fit` | forecast_lower / forecast_upper | PRED | 0.95 | h=1 | 0.944 ± 0.009 | -0.7 | -0.6 | at nominal |
| `tsecon.arima_fit` | d=1 (random walk with drift) | PRED | 0.95 | h=1 | 0.939 ± 0.006 | -1.8 | -1.1 | at nominal |
| `tsecon.var_forecast` | lower / upper | PRED | 0.95 | h=1 | 0.948 ± 0.002 | -0.9 | -0.2 | at nominal |
| `tsecon.var_forecast` | marginal bands read as a JOINT band | PRED | 0.95 | h=1 | 0.944 ± 0.002 | -2.8 | -0.6 | at nominal |
| `tsecon.bvar_irf_draws` | 5th/95th posterior percentile band | CRED | 0.90 | prior 'oracle-tight', h=4 | 0.906 ± 0.011 | +0.5 | +0.6 | n/a |
| `tsecon.bvar_ssvs` | spike-and-slab credible band | CRED | 0.90 | prior 'SSVS spike-slab', h=0 | 0.899 ± 0.011 | -0.1 | -0.1 | n/a |
| `tsecon.bvar_irf_draws` | impact band vs an EXACT interval | CRED | 0.90 | exact chi-square interval for the same scalar | 0.904 ± 0.006 | +0.6 | +0.4 | n/a |
| `tsecon.sign_restricted_svar` | pointwise 5-95 band over rotations | CRED | 0.90 | lambda1=5.0, h=3 | 0.860 ± 0.017 | -2.3 | -4.0 | n/a |
| `tsecon.robust_svar_bounds` | Giacomini-Kitagawa robust region | CRED | 0.90 | lambda1=5.0, h=3 | 0.935 ± 0.012 | +2.8 | +3.5 | n/a |
| `tsecon.sign_restricted_svar` | set envelope (min/max over draws) | SET | 0.90 | lambda1=5.0, h=3 | 0.983 ± 0.007 | +12.6 | +8.3 | n/a |
| `tsecon.zero_sign_svar` | band at a TRUE point-identifying zero | CRED | 0.90 | lambda1=5.0, h=3 | 0.875 ± 0.017 | -1.5 | -2.5 | n/a |
| `tsecon.bai_perron` | break-date CI, conditional on detection | CI | 0.95 | break/sigma=1.0, T=800, cond. on detection | 0.970 ± 0.005 | +3.9 | +2.0 | **OVER** |
| `tsecon.bai_perron` | break-date CI, UNconditional | CI | 0.95 | break/sigma=3.0, T=200, detection 0.97 | 0.967 ± 0.004 | +4.1 | +1.7 | **OVER** |
| `tsecon.bai_perron` | break-date CI at a LARGE break | CI | 0.95 | break/sigma=1.0, T=200, cond. on detection | 0.957 ± 0.005 | +1.5 | +0.7 | at nominal |
| `tsecon.theta_forecast` / `tsecon.backtest` | no interval is returned | NONE | — | the library returns a point path only | — | — | — | no band |

### Table 2 — the stress case

*Sorted worst first. Every stress design was **chosen** to be stressful, so a
miss here is expected; the deliverable is its **size**, not its existence.*

| surface | interval / option | kind | nom | design measured | coverage ± MC se | dev | gap pp | verdict |
|---|---|---|---|---|---|---|---|---|
| `tsecon.var_irf_bands` | with the lag order wrong | CI | 0.90 | n=500, fit as VAR(1) [misspecified], h=4 | 0.061 ± 0.005 | -156.1 | -83.9 | **UNDER** |
| `tsecon.bai_perron` | break-date CI, UNconditional | CI | 0.95 | break/sigma=0.25, T=200, detection 0.29 | 0.233 ± 0.009 | -76.0 | -71.7 | **UNDER** |
| `tsecon.var_forecast` | marginal bands read as a JOINT band | PRED | 0.95 | every horizon and series inside simultaneously | 0.409 ± 0.006 | -85.2 | -54.1 | **UNDER** |
| `tsecon.var_irf_bands` | method="bootstrap" | CI | 0.90 | n=100, h=12 | 0.410 ± 0.011 | -44.6 | -49.0 | **UNDER** |
| `tsecon.ols` | se_type="hac" | CI | 0.95 | x,e AR(1) phi=0.95, T=200 | 0.588 ± 0.009 | -40.3 | -36.2 | **UNDER** |
| `tsecon.iv_gmm` | weight="hac" | CI | 0.95 | AR(1) errors phi=0.8, hac bw=0 (DEFAULT) | 0.632 ± 0.009 | -36.1 | -31.8 | **UNDER** |
| `tsecon.smooth_lp` | lam="cv" (the default) | CI | 0.95 | lam=cv, h=0 | 0.640 ± 0.018 | -17.1 | -31.0 | **UNDER** |
| `tsecon.bvar_ssvs` | spike-and-slab credible band | CRED | 0.90 | prior 'SSVS spike-slab', h=12 | 0.594 ± 0.019 | -16.5 | -30.6 | n/a |
| `tsecon.bvar_irf_draws` | 5th/95th posterior percentile band | CRED | 0.90 | prior 'default', h=4 | 0.610 ± 0.018 | -15.7 | -29.0 | n/a |
| `tsecon.ols` | se_type="hc1"; small T, leverage | CI | 0.95 | x~chi2(1), sd(e\|x)=x, T=25 | 0.682 ± 0.009 | -31.6 | -26.8 | **UNDER** |
| `tsecon.var_irf_bands` | method="asymptotic" | CI | 0.90 | n=100, h=12 | 0.673 ± 0.010 | -21.6 | -22.7 | **UNDER** |
| `tsecon.ols` | se_type="nonrobust" | CI | 0.95 | heteroskedastic sd=\|x\|, T=200 | 0.732 ± 0.008 | -27.0 | -21.8 | **UNDER** |
| `tsecon.ols` | se_type="hc1" | CI | 0.95 | AR(1) errors+regressor .7, T=200 | 0.735 ± 0.008 | -26.7 | -21.5 | **UNDER** |
| `tsecon.sign_restricted_svar` | pointwise 5-95 band over rotations | CRED | 0.90 | lambda1=0.2, h=3 | 0.698 ± 0.023 | -8.8 | -20.2 | n/a |
| `tsecon.var_irf_bands` | pointwise band read as a JOINT band | CI | 0.90 | n=500, all of h=0..12 at once | 0.722 ± 0.010 | -17.8 | -17.8 | **UNDER** |
| `tsecon.iv_gmm` | 2sls / 2step / iterated | CI | 0.95 | median first-stage F = 1.2 | 0.839 ± 0.007 | -16.5 | -11.1 | **UNDER** |
| `tsecon.var_irf_bands` | cumulative=True | CI | 0.90 | orth=True,cumulative=False, h=12 | 0.789 ± 0.009 | -12.2 | -11.1 | **UNDER** |
| `tsecon.zero_sign_svar` | band at a TRUE point-identifying zero | CRED | 0.90 | lambda1=0.2, h=3 | 0.797 ± 0.020 | -5.1 | -10.3 | n/a |
| `tsecon.robust_svar_bounds` | Giacomini-Kitagawa robust region | CRED | 0.90 | lambda1=0.2, h=3 | 0.800 ± 0.020 | -5.0 | -10.0 | n/a |
| `tsecon.quantile_regression` | Powell sandwich, slope | CI | 0.95 | location-scale T=200, tau=0.05 | 0.866 ± 0.006 | -13.5 | -8.4 | **UNDER** |
| `tsecon.lp` | se="hac" | CI | 0.95 | T=100 hac, h=12 | 0.870 ± 0.008 | -10.7 | -8.0 | **UNDER** |
| `tsecon.har_rv` | HAC SE on the CONSTANT | CI | 0.95 | het innovations, maxlags=22, const | 0.871 ± 0.006 | -12.9 | -7.9 | **UNDER** |
| `tsecon.bai_perron` | break-date CI, conditional on detection | CI | 0.95 | break/sigma=0.5, T=200, cond. on detection | 0.888 ± 0.008 | -8.0 | -6.2 | **UNDER** |
| `tsecon.var_irf_bands` | method="bootstrap", bias_correct=True | CI | 0.90 | n=100, h=0 | 0.842 ± 0.008 | -7.2 | -5.8 | **UNDER** |
| `tsecon.arima_fit` | forecast_lower / forecast_upper | PRED | 0.95 | worst h (10) | 0.899 ± 0.011 | -4.5 | -5.1 | **UNDER** |
| `tsecon.arima_fit` | d=1 (random walk with drift) | PRED | 0.95 | worst h (22) | 0.902 ± 0.008 | -6.3 | -4.8 | **UNDER** |
| `tsecon.lp_multiplier` | integral multiplier | CI | 0.95 | multiplier, worst h (8) | 0.903 ± 0.005 | -8.7 | -4.7 | **UNDER** |
| `tsecon.lp_state` | per-regime response | CI | 0.95 | state1 lag_augmented, worst h (7) | 0.907 ± 0.007 | -5.7 | -4.3 | **UNDER** |
| `tsecon.lp_iv` | strong instrument | CI | 0.95 | strong iv, worst h (5) | 0.909 ± 0.005 | -7.8 | -4.1 | **UNDER** |
| `tsecon.bvar_irf_draws` | impact band vs an EXACT interval | CRED | 0.90 | credible band, lambda1=5.0 | 0.874 ± 0.007 | -3.9 | -2.6 | n/a |
| `tsecon.har_rv` | HAC SEs on the three slopes | CI | 0.95 | het innovations, maxlags=22, b_monthly | 0.925 ± 0.005 | -5.2 | -2.5 | **UNDER** |
| `tsecon.var_forecast` | lower / upper | PRED | 0.95 | worst h (11) | 0.925 ± 0.003 | -9.2 | -2.5 | **UNDER** |
| `tsecon.lp` | se="lag_augmented" (the default) | CI | 0.95 | lag_augmented, worst h (6) | 0.934 ± 0.004 | -4.1 | -1.6 | **UNDER** |
| `tsecon.recession_probit` | Wald interval | CI | 0.95 | probit rare   phi=0.9, T=100, 25% no MLE | 0.968 ± 0.004 | +4.9 | +1.8 | **OVER** |
| `tsecon.lp_iv` | weak instrument | CI | 0.95 | weak iv, worst h (4) | 0.985 ± 0.002 | +15.5 | +3.5 | **OVER** |
| `tsecon.quantile_regression` | Powell sandwich, intercept | CI | 0.95 | location-scale T=200, tau=0.5 | 0.993 ± 0.002 | +28.2 | +4.3 | **OVER** |
| `tsecon.bai_perron` | break-date CI at a LARGE break | CI | 0.95 | break/sigma=3.0, T=200, cond. on detection | 0.998 ± 0.001 | +54.2 | +4.8 | **OVER** |
| `tsecon.sign_restricted_svar` | set envelope (min/max over draws) | SET | 0.90 | lambda1=0.2, h=3 | 0.953 ± 0.011 | +4.9 | +5.2 | n/a |
| `tsecon.theta_forecast` / `tsecon.backtest` | no interval is returned | NONE | — | the library returns a point path only | — | — | — | no band |

---

## Where the intervals miss

This is the section to act on. Each caveat links to the model card for the
function, so it is reachable from the function's own documentation as well as
from here.

### A. Misses even in the favourable design

**8 of 32 frequentist intervals.** These are off nominal in the design they are
entitled to do well on, so a caller cannot fix them by having better data.

| surface | favourable design | measured (nominal 0.95 / 0.90) | cause | what to do | documented in |
|---|---|---|---|---|---|
| `iv_gmm(weight="hac")` | AR(1) errors φ=0.8, `bandwidth=10` supplied | **0.868 ± 0.006** (automatic default: **0.842 ± 0.007**) | APPROXIMATION *(was CONVENTION — [fixed in 0.2.0](#fixed-in-020))* | the original finding: `bandwidth` defaulted to `0.0`, and a Bartlett kernel truncated at 0 lags **is** the White estimator, so `weight="hac"` alone changed nothing (verified bit-identical, max \|Δse\| = 0.000e+00 over 3000 reps) and covered **0.632 ± 0.009**. `0.2.0` made the default the Newey-West rule and made an explicit `0.0` an error. That lifts coverage to 0.842 and **no further** — at this persistence T=250 cannot estimate the long-run variance, and even `bandwidth=10` reaches only 0.868 | [GMM card](../reference/model-cards/gmm.md#iv_gmm-linear-iv-gmm) |
| `var_irf_bands(method="bootstrap")` | impact, persistent VAR, T=100 | **0.848 ± 0.008** (h=12: **0.410 ± 0.011**) | ESTIMATOR | the percentile band sits *below* an already downward-biased point estimate — a second dose of the same bias. Pass `bias_correct=True` on a persistent VAR: it lifts h=12 from 0.410 to **0.900** | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `har_rv` — the **constant** | heteroskedastic innovations, `maxlags=0` | **0.910 ± 0.005** (`maxlags=22`: **0.871 ± 0.006**) | ESTIMATOR | not the SE: the least-squares persistence bias at Σb = 0.95 is absorbed *entirely* by the intercept (measured bias −0.1000 against the mechanical prediction −0.1001). The three slopes are at nominal; do not read the HAR intercept as if it were | [Realized-vol card](../reference/model-cards/realized-vol.md#how-to-read-the-output) |
| `lp_iv` — strong instrument | median first-stage F ≈ 134, best horizon | **0.930 ± 0.005** (worst horizon: **0.909 ± 0.005**) | CONVENTION | the kernel covariance follows `linearmodels`' `debiased=False` convention (which is what makes the point estimates match the golden) and applies *p* Bartlett lags even at h=0, where the score has nothing to smooth. Subtract two to four points from the nominal level before quoting an LP-IV interval | [LP card](../reference/model-cards/local-projections.md#lp_iv-instrumented-local-projections-lp-iv) |
| `lp_multiplier` | impact, median F ≈ 159 | **0.934 ± 0.005** (widest window: **0.903 ± 0.005**) | CONVENTION | well centred (\|bias\|/sd ≤ 0.08) and strongly instrumented at every horizon, but se/sd ≈ 0.9: the honest critical value at T=240 is nearer 2.2 than 1.96 | [LP card](../reference/model-cards/local-projections.md#lp_multiplier-integral-multipliers) |
| `lp_iv` — weak instrument | kindest horizon of the weak arm itself | **0.969 ± 0.003** (worst: **0.985 ± 0.002**) | APPROXIMATION | it **over**-covers while the median interval width explodes about 5×. That is the correct symptom, not a lucky escape: Dufour (1997) — under weak identification no *bounded* confidence set can be honest, so a Wald set stays honest only by becoming uninformative. Report `first_stage_f` | [LP card](../reference/model-cards/local-projections.md#lp_iv-instrumented-local-projections-lp-iv) |
| `bai_perron` — unconditional | break/σ = 3, T=200 | **0.967 ± 0.004** (break/σ = 0.25: **0.233 ± 0.009**) | ESTIMATOR | at a small break, detection itself collapses to 0.29, so the rate a user actually faces is 0.233. A break-date CI is meaningful only once the break is detectable | [Structural-breaks card](../reference/model-cards/structural-breaks.md#bai_perron-multiple-breaks-how-many-when-how-sure) |
| `bai_perron` — conditional on detection | break/σ = 1, T=800 | **0.970 ± 0.005** (break/σ = 0.5, T=200: **0.888 ± 0.008**) | APPROXIMATION / CONVENTION | over-covers at a large break because the half-width is `ceil(c/scale)` **plus one index** on each side, and that discreteness padding dominates (0.998 at break/σ = 3). Under-covers at a small break because Bai's argmax limit distribution is a finite-sample approximation — it improves with T (0.877 → 0.914 → 0.944 at T = 200/400/800) while the interval *width* does not shrink at all (26.4 → 26.5 → 26.1), exactly as fixed-break asymptotics predict | [Structural-breaks card](../reference/model-cards/structural-breaks.md#bai_perron-multiple-breaks-how-many-when-how-sure) |

### B. At nominal when entitled, off under stress

**23 of 31.** These behave the way asymptotic approximations behave. Nothing
here is an alarm; the number to quote is the *size* of the loss in the regime
you are actually in.

| surface | stress design | measured | cause | what to do | documented in |
|---|---|---|---|---|---|
| `var_irf_bands` with the wrong lag order | VAR(4) truth fitted as VAR(1), h=4, T=500 | **0.061 ± 0.005** | ESTIMATOR | inconsistency, not a band problem: coverage gets *worse* as T grows — 17.8% at T=200, 6.2% at T=500. Choose the lag order on the data before reading any band; with the correct order the same cell covers 0.905 | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `var_forecast` marginal bands read as a joint band | 12 horizons × 2 series simultaneously, T=100 | **0.409 ± 0.006** | READING | see [pointwise is not joint](#pointwise-is-not-joint) | [VAR/SVAR card](../reference/model-cards/var-svar.md#reduced-form-var-var_fit-var_irf-var_fevd-var_granger-var_forecast) |
| `ols(se_type="hac")` on a **slope** | regressor *and* errors AR(1) at φ=0.95, T=200 | **0.588 ± 0.009** | APPROXIMATION | the reported HAC SE is 0.43 of the true sampling sd. Lengthening the bandwidth helps and does not close it (0.703 at 12 lags, 0.728 at 24): at T=200 the sample does not contain enough independent information to estimate a long-run variance this large. Report such a slope with a bandwidth chosen for the persistence and treat the interval as indicative | [HAC cookbook](../cookbook/hac-standard-errors.md#choosing-the-bandwidth) |
| `smooth_lp(lam="cv")` | impact response, T=200 | **0.640 ± 0.018** | ESTIMATOR | by design, and worst exactly where an applied reader looks first. At impact the cross-validated penalty pulls the estimate off the peak (\|bias\|/sd = 1.22 — the bias exceeds a whole sampling sd). A smooth-LP band is a band around the **penalized** estimand; the unpenalized `lam=0` anchor covers 0.936 at the same cell. Separately, `se` conditions on the selected λ, so mean se/sd falls from 0.907 to 0.812 | [LP card](../reference/model-cards/local-projections.md#smooth_lp-smooth-local-projections-barnichon-brownlees) |
| `ols(se_type="hc1")`, small T with leverage | x ~ chi2(1), sd(e\|x)=x, T=25 | **0.682 ± 0.009** | ESTIMATOR *(was API GAP — [fixed in 0.2.0](#fixed-in-020))* | `hc1`'s n/(n−k) factor buys 0.015 at k=2; the leverage correction 1/(1−h_i) is what matters. `0.2.0` added `hc2`/`hc3`, and tsecon's own `hc3` now covers **0.863 ± 0.006** on these draws — 18 points recovered, and still short of nominal, so prefer `hc3` at small n without treating it as a cure | [Inference guide](../guide/03-inference-toolkit.md#the-robust-standard-error-ladder) |
| `var_irf_bands(method="asymptotic")` | h=12, T=100 | **0.673 ± 0.010** | APPROXIMATION | [the horizon table](#the-delta-method-irf-band-horizon-by-horizon) | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `ols(se_type="nonrobust")` | heteroskedastic sd=\|x\|, T=200 | **0.732 ± 0.008** | ESTIMATOR | inconsistent, so data does not help: in the high-leverage design it is *stuck* at 0.428 even at T=1600 and slides **downward** with T. `hc0`/`hc1` repair almost all of it (0.732 → 0.942) | [Inference guide](../guide/03-inference-toolkit.md#the-robust-standard-error-ladder) |
| `ols(se_type="hc1")` under serial correlation | AR(1) errors, AR(1) regressor φ=0.7, T=200 | **0.735 ± 0.008** | ESTIMATOR | HC repairs **nothing** here — statistically indistinguishable from `nonrobust`'s 0.744, with se/sd 0.569 vs 0.579. HC is heteroskedasticity-robust, not serial-correlation robust. Only `hac` moves the number (0.876) | [HAC cookbook](../cookbook/hac-standard-errors.md#gotchas) |
| `var_irf_bands` pointwise read as joint | all of h=0..12 at once, T=500 | **0.722 ± 0.010** | READING | see [pointwise is not joint](#pointwise-is-not-joint) | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `iv_gmm` with weak instruments | median first-stage F = 1.2, T=250 | **0.839 ± 0.007** | ESTIMATOR + **API GAP** | the *mean* reported se/sd of 1.27 is a mirage: the **median** reported SE is only 0.456 of the true sampling sd, and the mean is dragged above 1 by a handful of replications with enormous SEs. IQR/(1.349 sd) = 0.42 says the sampling law is nothing like normal. A fixed-width interval at the true sd covers 0.968, so the damage is done by how the SE *varies* across samples (corr(\|error\|, SE) = +0.83 — it is smallest in exactly the samples where the estimate is worst). **And the rule-of-thumb F of 10 is not safe: at median F = 10.5 coverage is already 0.915 with a median se/sd of 0.841.** `0.2.0` added the `first_stage` diagnostic so the caller can at least see the strength; no Anderson-Rubin set is exposed, and that remains the real answer here | [GMM card](../reference/model-cards/gmm.md#iv_gmm-linear-iv-gmm) |
| `var_irf_bands` per-horizon vs `cumulative=True` | per-horizon band, h=12, T=200 | **0.789 ± 0.009** (cumulative: 0.884 ± 0.007) | APPROXIMATION | on this DGP the running sum is dominated by the early, well-estimated horizons, so it is a much more nearly linear function of the estimated slopes. Measured here, not a general theorem | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `quantile_regression`, extreme τ | τ=0.05, location-scale, T=200 | **0.866 ± 0.006** | APPROXIMATION | se/sd is 0.818 while the point-estimate bias is +0.008, so it is squarely the SE: the Powell sandwich needs a conditional density at the fitted quantile, estimated from the handful of observations near an extreme quantile. It shrinks with T (0.916 at T=1000) but does not close. τ=0.50 is fine (0.940). Bootstrap the quantile process for extreme τ at a few hundred observations | [Quantile card](../reference/model-cards/quantile.md#quantile_regression-linear-quantile-regression) |
| `lp(se="hac")` | h=12, T=100 | **0.870 ± 0.008** | APPROXIMATION | the default `se="lag_augmented"` covers better at **every** horizon on the same draws (paired gap +0.027 pooled over h≥6, se 0.0014). Newey-West at bandwidth h+p spends its degrees of freedom estimating autocovariances that lag augmentation has already removed. The gap closes in T (0.870 at T=100 → 0.939 at T=800) | [LP card](../reference/model-cards/local-projections.md#lp-local-projection-irfs) |
| `har_rv` slopes at a long bandwidth | heteroskedastic, `maxlags=22`, b_monthly | **0.925 ± 0.005** | APPROXIMATION | bandwidth is not free. On a correctly specified HAR with iid innovations, b_daily's se/sd falls 0.989 (maxlags=0) → 0.983 (5, the default) → 0.970 (22), and coverage 0.949 → 0.945 → 0.937. The default of 5 is a sensible compromise; 22 is a real cost | [Realized-vol card](../reference/model-cards/realized-vol.md#key-arguments-and-defaults) |
| `arima_fit` forecast band | AR(1) φ=0.9, worst horizon, T=100 | **0.899 ± 0.011** | APPROXIMATION | a **plug-in** band: the identical formula at the *true* parameters covers 0.946 on the same draws (paired plug-in cost +4.7pp ± 1.0). The formula is right; the gap is the price of not knowing φ and σ | [ARIMA card](../reference/model-cards/arima.md#arima_fit-exact-mle-arimapdq-fit-and-forecast) |
| `arima_fit(d=1)` forecast band | random walk with drift, h=24, T=60 | **0.902 ± 0.008** | APPROXIMATION | `forecast_se` is *exactly* σ̂·√h (to 2.7e-15), i.e. the h²/(T−1) drift-uncertainty term is omitted entirely. The shortfall is therefore predictable in closed form: 2Φ(z/√(1+h/(T−1)))−1 gives 90.2% at h=24, measured 90.3%. Restoring the term recovers 94.5% | [ARIMA card](../reference/model-cards/arima.md#arima_fit-exact-mle-arimapdq-fit-and-forecast) |
| `lp_state`, the persistent regime | state 1, worst horizon, T=300 | **0.907 ± 0.007** | ESTIMATOR | se/sd is close to 1, so this is centring, not scale: the interacted design identifies each regime off roughly half a persistent sample and \|bias\|/sd reaches 0.27. The quiet regime is at nominal (0.937–0.953). State-dependent LP needs more data than linear LP for the same interval to mean the same thing | [LP card](../reference/model-cards/local-projections.md#lp_state-state-dependent-local-projections) |
| `var_forecast` band | worst horizon, T=100 | **0.925 ± 0.003** | APPROXIMATION | the same plug-in story, and it is estimation error rather than bias: at T=800 the paired gap to the oracle band is +0.2pp and coverage is 0.948; at T=100 the gap is +2.4pp | [VAR/SVAR card](../reference/model-cards/var-svar.md#reduced-form-var-var_fit-var_irf-var_fevd-var_granger-var_forecast) |
| `lp(se="lag_augmented")` — the default | worst horizon, T=200 | **0.934 ± 0.004** | APPROXIMATION | the best-calibrated interval in the family, and the reason lag augmentation is the default. se/sd sits within a couple of percent of 1 at every horizon; the residual 1.6pp is ordinary finite-sample dynamic-regression bias, and it shrinks in T | [LP card](../reference/model-cards/local-projections.md#lp-local-projection-irfs) |
| `recession_probit` | rare events (rate 0.055), T=100 | **0.968 ± 0.004**, on survivors | ESTIMATOR | **over**-covers, and the number is *selected*: 25.0% of replications have no finite MLE and the library correctly raises (16.2% no recession months at all, 8.8% quasi-complete separation). Read the failure share with the coverage, always. Among survivors 12.4% carry an SE more than 3× the median, and the MLE is biased away from zero (median bias +0.117 at T=100, +0.012 at T=1000). Note `se/sd` = 0.496 *with* 0.968 coverage is an outlier signature, not a narrow interval | [Recession card](../reference/model-cards/recession.md#recession_probit) |
| `quantile_regression` intercept | x ~ U(0,2), so x=0 is at the edge, τ=0.50 | **0.993 ± 0.002** | APPROXIMATION | over-covers because the intercept is an extrapolation to the edge of the support and its sandwich SE is conservative. Per-coefficient coverage in a quantile regression is not uniform, and the intercept is usually not the quantity of interest | [Quantile card](../reference/model-cards/quantile.md#quantile_regression-linear-quantile-regression) |
| `var_irf_bands(bias_correct=True)` at impact | h=0, persistent VAR, T=100 | **0.842 ± 0.008** | APPROXIMATION | the cost side of a good trade: Kilian's correction buys ~49 coverage points at h=12 (0.410 → 0.900) and costs 5.8pp at impact. Take the trade on a persistent VAR | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `theta_forecast`, `backtest` | — | **no interval at all** | READING | both return point paths only; `backtest` returns no interval-bearing key. Any band you report around them is your own construction and its coverage is your claim, not the library's. (For reference, a DIY interval built from `backtest` errors on a random walk with drift covers 93.0% at h=1 falling to 90.6% at h=6 — our construction, not a library promise) | [Forecasting card](../reference/model-cards/forecasting.md#how-to-read-the-output) |

### C. Objects that make no frequentist promise

**7 rows.** Nothing here is a defect. A shortfall measures the **prior** or the
**identified set**, and the reason these are reported at all is that a reader
who expects 90% has misread the object.

| object | design | measured (nominal 0.90) | what the number means | documented in |
|---|---|---|---|---|
| `bvar_ssvs` credible band | true-but-small cross coefficient, h=12 | **0.594 ± 0.019** | the spike prior does exactly what it is for and zeroes a true 0.03 cross lag; the band then sits around zero. Compare the diffuse NIW band's 0.809 at the same cell | [Bayesian card](../reference/model-cards/bayesian.md#bvar_ssvs-spike-and-slab-stochastic-search-selection) |
| `bvar_irf_draws` credible band | **library-default** Minnesota prior (δ=0), h=4 | **0.610 ± 0.018** | the prior mean is white noise and the truth has own lags 0.85, so the band is in the wrong *place* (bias −0.182 against width 0.39). A well-centred, **tighter** prior (δ=0.85, λ₁=0.05) reaches 0.906 at a **narrower** width of 0.32 — higher coverage from a smaller band, which is the signature of a centring problem rather than a scale one. `bvar_hierarchical` loosens the badly centred prior (λ₁ 0.2 → 0.60) and tightens the well-centred one (0.2 → 0.16), exactly as marginal-likelihood logic says — but tuning tightness cannot fix a prior mean in the wrong place (0.784 at h=12) | [Bayesian card](../reference/model-cards/bayesian.md#bvar_irf_draws-posterior-impulse-response-draws) |
| `sign_restricted_svar` pointwise 5–95 band | λ₁=0.2, h=3 | **0.698 ± 0.023** | a Haar-rotation posterior summary that **mixes mutually inconsistent structural models** — neither a confidence interval nor the identified set. That is precisely what [`fry_pagan_svar`](../reference/model-cards/structural-identification.md#fry_pagan_svar-the-coherent-draw-the-median-band-is-not) exists to complain about. At λ₁=5.0 it is 0.860 | [Identification card](../reference/model-cards/structural-identification.md#robust_svar_bounds-the-identified-set-without-the-haar-artifact) |
| `zero_sign_svar` band at a **true** point-identifying zero | λ₁=0.2, h=3 | **0.797 ± 0.020** | here the zero pins the rotation, so the band *is* about a point — which makes this the cleanest available reading of what the Minnesota prior costs a frequentist reader. 0.875 at λ₁=5.0 | [VAR/SVAR card](../reference/model-cards/var-svar.md#zero_sign_svar-zero-and-sign-restrictions-together) |
| `robust_svar_bounds` (Giacomini-Kitagawa) | λ₁=0.2, h=3 | **0.800 ± 0.020** | the one set-identified object that *does* aim at 1−α containment, and it delivers under a diffuse reduced-form prior (0.935 at λ₁=5.0, and ≥0.930 across every cell). It is robust to the **rotation** prior; it inherits the Minnesota prior on the reduced form, and with δ=0 that prior pulls a persistent response down and takes the whole set with it | [Identification card](../reference/model-cards/structural-identification.md#robust_svar_bounds-the-identified-set-without-the-haar-artifact) |
| `bvar_irf_draws` impact band vs an **exact** interval | prior mean exactly right (white noise, δ=0) | **0.874 ± 0.007** vs the exact chi-square interval's **0.904 ± 0.006** on the same samples | even a perfect prior mean leaves ~3pp, and the mechanism is the conjugate convention: the inverse-Wishart posterior has df v₀+T = 104 while the residual sampling df is T_eff−k = 96. This is the cleanest statement on the page of why a credible band is not a confidence interval | [Bayesian card](../reference/model-cards/bayesian.md#bvar_fit-minnesota-niw-posterior) |
| `sign_restricted_svar` **set envelope** | λ₁=0.2, h=3 | **0.953 ± 0.011** | the union over the reduced-form posterior of the identified set — wider than any credible object, and near-total containment certifies very little. At impact, where a sign restriction leaves the set open down to zero, *every* object covers ≈1.000 and measures nothing | [Identification card](../reference/model-cards/structural-identification.md#robust_svar_bounds-the-identified-set-without-the-haar-artifact) |

---

## Pointwise is not joint

This is the largest reading error available on this page, and it is not a
defect in any function. A pointwise band promises that *this horizon's* true
response is inside *this horizon's* band with probability 1−α. It promises
nothing about the whole path, and the two rates are far apart:

| object | nominal, pointwise | measured pointwise (impact) | measured **jointly**, whole path |
|---|---|---|---|
| `var_irf_bands`, asymptotic, T=100, h=0..12 | 90% | 0.897 | **0.567 ± 0.011** |
| `var_irf_bands`, asymptotic, T=200, h=0..12 | 90% | 0.900 | **0.650 ± 0.011** |
| `var_irf_bands`, asymptotic, T=500, h=0..12 | 90% | 0.910 | **0.722 ± 0.010** |
| `var_irf_bands`, bootstrap, T=500, h=0..12 | 90% | 0.903 | **0.735 ± 0.010** |
| `var_forecast`, T=100, 12 horizons × 2 series | 95% | 0.944 | **0.409 ± 0.006** |
| `var_forecast`, T=800, 12 horizons × 2 series | 95% | 0.948 | **0.481 ± 0.006** |
| `arima_fit`, AR(1) φ=0.9, T=100, h=1..12 | 95% | 0.933 | **0.639 ± 0.018** |

Note the T=800 row: the joint rate is 0.481 even where the marginal rate is at
nominal (0.948 ± 0.002). **Joint coverage does not converge to the marginal level as
the sample grows** — it is a different quantity, and a band that contains the
whole path 95% of the time has to be materially wider. `tsecon` reports no
simultaneous (sup-t) band for any object; that is an
[API gap](#what-this-audit-recommends), and until it closes, a fan chart should
be read one horizon at a time.

---

## The delta-method IRF band, horizon by horizon

The single most actionable table in the audit. DGP: a stationary VAR(1) with
largest root 0.758 and Σ = [[1, 0.4], [0.4, 2]]; the population orthogonalised
IRF is exactly `A**h @ chol(Σ)`, so the truth is closed-form at every horizon.
Cell: the response of `y1` to orthogonalised shock 0. Nominal **90%**, 2000
replications, 399 bootstrap draws per replication.

```text
  n = 100
    h    truth  med bias   mean se    mc sd  |bias|/sd  cov asym  cov boot
    0   0.4000   -0.0094    0.1386   0.1424       0.07  89.7±0.7  88.8±0.7
    1   0.3500   -0.0237    0.1207   0.1270       0.19  87.8±0.7  86.7±0.8
    2   0.2860   -0.0340    0.1238   0.1281       0.27  87.2±0.7  84.4±0.8
    3   0.2260   -0.0367    0.1134   0.1155       0.32  85.5±0.8  81.7±0.9
    4   0.1753   -0.0374    0.0981   0.0987       0.38  83.5±0.8  80.0±0.9
    5   0.1347   -0.0330    0.0827   0.0826       0.40  81.2±0.9  78.7±0.9
    6   0.1029   -0.0285    0.0688   0.0685       0.42  79.0±0.9  77.5±0.9
    7   0.0784   -0.0243    0.0569   0.0567       0.43  76.6±0.9  76.5±0.9
    8   0.0596   -0.0200    0.0469   0.0469       0.43  74.4±1.0  75.8±1.0
    9   0.0452   -0.0166    0.0387   0.0389       0.43  72.5±1.0  75.3±1.0
   10   0.0343   -0.0136    0.0319   0.0324       0.42  70.7±1.0  74.7±1.0
   11   0.0260   -0.0111    0.0263   0.0270       0.41  68.7±1.0  74.6±1.0
   12   0.0197   -0.0091    0.0217   0.0226       0.40  67.3±1.0  74.3±1.0
  simultaneous coverage of the whole h=0..12 path (pointwise bands make NO such promise): asym 56.7±1.1  boot 62.5±1.1
    standardised statistic t = (point - truth)/se, asymptotic arm. The Wald band
    covers exactly when |t| <= 1.645, so these four rows ARE the coverage row above.
                         h=0     h=1     h=2     h=4     h=6     h=8    h=12
    skewness            0.00   -0.17   -0.29   -0.81   -2.00   -3.84   -9.48
    5th pct            -1.67   -1.96   -2.14   -2.68   -3.74   -5.50  -13.26
    median             -0.07   -0.20   -0.28   -0.38   -0.44   -0.48   -0.57
    95th pct            1.65    1.48    1.36    1.14    0.98    0.90    0.76
    se / mc sd          0.97    0.95    0.97    0.99    1.00    1.00    0.96
```

**Read the last row first.** `mean se / mc sd = 0.96` at h=12: the reported
standard error tracks the true sampling standard deviation to within 4%. The
standard error is *not* the problem. The problem is the row above it: the
standardised statistic has skewness **−9.48**, with 5th and 95th percentiles of
**−13.26 and +0.76** against the ±1.645 the Wald band assumes. The band is not
too narrow — it is **one-sidedly wrong**. Its lower edge is far too high and its
upper edge is never reached.

That distinction matters because it tells you what to do. Scaling the interval
up would not fix a skew; changing the *shape* would. The measurements confirm
it:

| n | h=0 | h=4 | h=8 | h=12 | joint, h=0..12 |
|---|---|---|---|---|---|
| 100, asymptotic | 89.7 ± 0.7 | 83.5 ± 0.8 | 74.4 ± 1.0 | **67.3 ± 1.0** | 56.7 ± 1.1 |
| 100, bootstrap | 88.8 ± 0.7 | 80.0 ± 0.9 | 75.8 ± 1.0 | **74.3 ± 1.0** | 62.5 ± 1.1 |
| 200, asymptotic | 90.0 ± 0.7 | 88.1 ± 0.7 | 82.5 ± 0.8 | **77.2 ± 0.9** | 65.0 ± 1.1 |
| 200, bootstrap | 90.0 ± 0.7 | 86.2 ± 0.8 | 84.0 ± 0.8 | **83.1 ± 0.8** | 70.0 ± 1.0 |
| 500, asymptotic | 91.0 ± 0.6 | 87.8 ± 0.7 | 86.6 ± 0.8 | **84.7 ± 0.8** | 72.2 ± 1.0 |
| 500, bootstrap | 90.3 ± 0.7 | 87.1 ± 0.8 | 86.7 ± 0.8 | **86.3 ± 0.8** | 73.5 ± 1.0 |

**Three practical conclusions.** (i) The bootstrap band is the better choice at
long horizons at every sample size measured here (74.3 vs 67.3 at T=100 h=12),
because it does not impose symmetry. (ii) The shortfall shrinks with T but is
still 5.3pp at T=500, h=12. (iii) `cumulative=True` holds up far better than the
per-horizon band on this DGP — 88.4% vs 78.9% at h=12, T=200 — because the
running sum is dominated by the early, well-estimated horizons.

And a fourth, from a different DGP: on a **persistent** VAR (largest root
0.950) the plain percentile bootstrap is *worse* than the Wald band, because the
band centre itself sits below an already downward-biased estimate. Kilian's bias
correction is the fix and it is dramatic:

```text
  n = 100                       h=0      h=1      h=2      h=4      h=6      h=8     h=10     h=12
  true response               1.000    0.950    0.902    0.815    0.735    0.663    0.599    0.540
  cov asymptotic           87.9±0.7 81.1±0.9 77.8±0.9 73.7±1.0 70.6±1.0 68.1±1.0 65.5±1.1 64.0±1.1
  cov bootstrap            84.8±0.8 67.2±1.0 55.1±1.1 45.6±1.1 41.6±1.1 41.4±1.1 41.1±1.1 41.0±1.1
  cov bootstrap+bc         84.2±0.8 85.0±0.8 86.7±0.8 88.6±0.7 89.4±0.7 89.5±0.7 89.8±0.7 90.0±0.7
  med bias asymptotic        -0.005   -0.044   -0.072   -0.117   -0.151   -0.177   -0.193   -0.202
  med bias bootstrap+bc      -0.005   -0.006    0.001    0.012    0.019    0.025    0.029    0.032
  centre-point bootstrap     -0.019   -0.060   -0.091   -0.122   -0.125   -0.113   -0.093   -0.072
  centre-point bootstrap+bc  -0.021   -0.029   -0.037   -0.045   -0.042   -0.029   -0.011    0.014
```

`centre-point` is the median of (lower+upper)/2 minus the point estimate. It is
0 for the symmetric Wald band by construction; the percentile bootstrap's
**−0.072** means the band sits *below* an estimate that is already −0.202 from
the truth. Coverage 0.410 → 0.900 from one keyword.

---

## Family detail

### Regression standard errors

`reps=3000`, nominal **95%**, `z = 1.959964`, MC se at p=0.95 is `0.0040`.
Designs: `y = 1 + 2x + e` with (row 1) iid Gaussian errors; (row 2) sd(e) = \|x\|;
(row 3) AR(1) errors with an AR(1) regressor at φ=0.7; (row 4) t(3) errors.

```text
error structure                 nonrobust          hc0          hc1     hac auto     hac lag8
---------------------------------------------------------------------------------------------
iid Gaussian                 0.946+-0.004 0.941+-0.004 0.941+-0.004 0.938+-0.004 0.933+-0.005
heteroskedastic sd=|x|       0.732+-0.008 0.940+-0.004 0.942+-0.004 0.939+-0.004 0.930+-0.005
AR(1) errors+regressor .7    0.744+-0.008 0.733+-0.008 0.735+-0.008 0.876+-0.006 0.891+-0.006
t(3) errors (inf kurtosis)   0.951+-0.004 0.944+-0.004 0.946+-0.004 0.943+-0.004 0.934+-0.005

se/sd ratio (mean SE / MC sd of the estimate; <1 = SE too small)
iid Gaussian                        0.999        0.988        0.993        0.981        0.968
heteroskedastic sd=|x|              0.572        0.960        0.965        0.954        0.941
AR(1) errors+regressor .7           0.579        0.566        0.569        0.804        0.839
t(3) errors (inf kurtosis)          0.987        0.960        0.965        0.953        0.941
```

Row 4 is a small but genuine reversal of the usual advice: under t(3) errors the
**robust** standard errors lose slightly *more* coverage than the naive one
(0.944 vs 0.951, se/sd 0.960 vs 0.987). t(3) has finite variance but no fourth
moment, which is exactly what the sandwich's asymptotics assume.

The next table is the cleanest decomposition in the audit. Design:
x ~ chi2(1) (high leverage), sd(e\|x) = x. `hc2`/`hc3` are `tsecon` output as of
`0.2.0`; the `hc2*`/`hc3*` columns are independent **NumPy references on the
same draws**, kept as a cross-check (the two agree to 1.04e-14 across every
replication and sample size).
`oracle` is the sandwich at the *true* per-observation error variances; with
Gaussian errors and a fixed design the estimate is exactly normal about it, so
the oracle column must be exactly 0.95 at every T. It is — which proves that
every shortfall to its left is the variance **estimate**, not the normal
approximation.

```text
     T    nonrobust          hc0          hc1         hc2*         hc3*       oracle
------------------------------------------------------------------------------------
    25 0.493+-0.009 0.667+-0.009 0.682+-0.009 0.773+-0.008 0.863+-0.006 0.951+-0.004
    50 0.482+-0.009 0.780+-0.008 0.789+-0.007 0.838+-0.007 0.893+-0.006 0.945+-0.004
   100 0.459+-0.009 0.849+-0.007 0.853+-0.006 0.885+-0.006 0.910+-0.005 0.950+-0.004
   400 0.440+-0.009 0.923+-0.005 0.924+-0.005 0.932+-0.005 0.942+-0.004 0.954+-0.004
  1600 0.428+-0.009 0.940+-0.004 0.940+-0.004 0.943+-0.004 0.945+-0.004 0.951+-0.004
```

`nonrobust` never converges — it is inconsistent here, so more data does not
help, and it slides *downward* with T. `hc0` converges but slowly. `hc1`'s
n/(n−k) factor is nearly worthless at k=2. **HC2/HC3 target the leverage
directly and recover most of the small-T gap** — which is why `0.2.0` added
them. Note the word *most*: `hc3` reaches 0.863 at T=25, not 0.95, and its mean
se/sd of 0.942 overstates the typical interval because the SE distribution is
skewed. The remaining gap is the same one the oracle column isolates.

And the worst single number in the audit — a slope with a persistent regressor
*and* persistent errors, `hac auto` = ⌊4(T/100)^(2/9)⌋ = 4 lags, T=200:

```text
   phi  score AC    nonrobust          hc1     hac auto    hac lag12    hac lag24
---------------------------------------------------------------------------------
  0.00      0.00 0.949+-0.004 0.947+-0.004 0.944+-0.004 0.930+-0.005 0.911+-0.005
  0.50      0.25 0.875+-0.006 0.871+-0.006 0.926+-0.005 0.922+-0.005 0.902+-0.005
  0.80      0.64 0.648+-0.009 0.634+-0.009 0.841+-0.007 0.873+-0.006 0.864+-0.006
  0.95      0.90 0.358+-0.009 0.340+-0.009 0.588+-0.009 0.703+-0.008 0.728+-0.008

   phi              nonrobust          hc1     hac auto    hac lag12    hac lag24   (se/sd)
---------------------------------------------------------------------------------
  0.00                  1.005        0.996        0.982        0.956        0.919
  0.50                  0.789        0.778        0.922        0.927        0.897
  0.80                  0.476        0.465        0.737        0.814        0.806
  0.95                  0.239        0.223        0.432        0.562        0.602
```

This extends [experiment 2 of the Monte Carlo suite](monte-carlo.md#2-hac-standard-errors-rescue-coverage-under-serial-correlation),
which covers a **mean**. A slope with a persistent regressor is materially
worse and was previously unmeasured. Note also the φ=0 column: a 24-lag
bandwidth costs 3.8 coverage points when there is no serial correlation to
soak up. Bandwidth is not free in either direction.

Full report, including `iv_gmm` × instrument strength (with the Hansen J size at
a true null), `har_rv`, `recession_probit` and `quantile_regression` per τ:
[`docs/examples/coverage/regression_se.py`](coverage/regression_se.py).

### VAR impulse-response bands

DGPs: `BASE` VAR(1) root 0.758; `PERSIST` VAR(1) root 0.950; `LAG4` VAR(4) root
0.900, all with Σ = [[1, 0.4], [0.4, 2]]. Nominal **90%**, `reps=2000`,
`n_boot=399`, horizons 0..12, n ∈ {100, 200, 500}.

Structural zeros are excluded from every claim and verified as exact facts
instead: under a Cholesky ordering the impact response of variable 0 to shock 1
is *identically* zero and `var_irf_bands` correctly reports `se = 0` with
`lower = upper = 0`. The truth is also zero, so that cell "covers" 100% of the
time by construction and measures nothing. Same for the whole `orth=False`
impact matrix, which is the identity with zero width.

The misspecification experiment deserves its own reading, because it is the one
place where more data makes coverage **worse**. `LAG4` truth, fitted as a
VAR(1), target held at the true VAR(4) path:

```text
  n = 200                               h=0      h=1      h=2      h=3      h=4      h=5      h=6      h=8     h=12
  true response                       0.400    0.220    0.109    0.052    0.188    0.175    0.122    0.104    0.067
  cov asymptotic, misspecified     82.7±0.8 68.8±1.0 69.1±1.0 72.5±1.0 17.8±0.9  7.0±0.6  7.1±0.6  1.2±0.2  0.1±0.0
  cov asymptotic, correct          88.1±0.7 88.8±0.7 88.5±0.7 90.5±0.7 89.1±0.7 87.2±0.7 87.5±0.7 87.0±0.8 81.7±0.9
  |bias|/mc_sd, misspecified           0.40     0.67     0.68     0.65     2.62     4.14     4.35     8.35    21.55
  |bias|/mc_sd, correct                0.02     0.08     0.08     0.06     0.16     0.21     0.28     0.26     0.38
  n = 500
  cov asymptotic, misspecified     77.5±0.9 48.9±1.1 47.1±1.1 47.5±1.1  6.2±0.5  0.5±0.2  0.7±0.2  0.0±0.0  0.0±0.0
  cov asymptotic, correct          90.4±0.7 90.2±0.7 90.1±0.7 90.0±0.7 90.5±0.7 89.1±0.7 88.8±0.7 89.0±0.7 86.2±0.8
  |bias|/mc_sd, misspecified           0.70     1.29     1.27     1.23     3.49     5.69     6.12    12.90    42.35
```

17.8% → 6.2% at h=4 as T goes 200 → 500. That is the signature of
inconsistency: the interval shrinks around the wrong number. `|bias|/mc_sd` of
42 at h=12 says the band is nowhere near the truth in units of its own width.
The correct-lag arm is at nominal throughout. No band can substitute for
choosing the lag order.

Full report: [`docs/examples/coverage/irf_bands.py`](coverage/irf_bands.py).

### Local projections

DGP: `y_t = Σ_{j<J} θ_j s_{t-j} + nuisance_t` with `θ_j = 0.7^j`, `J=25`, and
`s_t` iid standard normal. Because `s_t` is orthogonal in population to
everything else in the horizon-h projection, the population LP coefficient on
`s_t` is **exactly** `θ_h` — for any number of lag controls and whatever serial
correlation the nuisance term has. There is no approximation in the truth, so
any miss belongs to the interval. Nominal **95%**; T=200 unless stated.

The headline is that the library's default is the right default, and the
comparison is **paired** on the same draws so the difference carries its own
standard error:

```text
paired coverage difference, lag_augmented minus hac (same draws):
  h      diff   se_diff   diff/se
---------------------------------
  0   +0.0060    0.0019      3.15
  4   +0.0217    0.0029      7.54
  8   +0.0220    0.0034      6.49
 12   +0.0340    0.0037      9.14
pooled over h >= 6 (per-draw average, so cross-horizon correlation is handled): +0.0271 (se 0.0014)
```

`se="lag_augmented"` (Montiel Olea & Plagborg-Møller 2021) wins at **every**
horizon, the mechanism is visible in se/sd (1.000 vs 0.926 averaged over h≥6),
and the gap closes in T — HAC's h=12 coverage goes 0.870 (T=100) → 0.939
(T=800) as its se/sd rises 0.863 → 0.979.

`smooth_lp` is the largest frequentist miss in the family, and it is worst
exactly where an applied reader looks first:

```text
                  arm    h     truth       bias     sd_est    mean_se   se/sd   |b|/sd    cov95    mcse
                lam=0    0    1.0000  -0.0003552    0.03993    0.04024    1.01     0.01    0.936   0.009
               lam=cv    0    1.0000   -0.07852    0.06417    0.05081    0.79     1.22    0.640   0.018
              lam=100    0    1.0000   -0.03906    0.04482     0.0451    1.01     0.87    0.861   0.013
```

`|b|/sd = 1.22` at impact: the shrinkage bias exceeds a whole sampling standard
deviation, so the interval is centred in the wrong place and no standard error
saves it. The `lam=100` arm isolates the two effects — se/sd is 1.01 there (the
SE is correctly sized) while `|b|/sd` is already 0.87 (pure shrinkage). The
library's own model card says `se` "conditions on `lam` and does not account for
shrinkage bias"; **0.640 against 0.95 is the size of what that sentence is
hiding.** Read a smooth-LP band as a band around the penalized estimand, and
read the `lam=cv` column against the `lam=0` column rather than against 0.95.

Full report, including `lp_iv` strong vs weak, `lp_state` per regime, and
`lp_multiplier`, plus a decomposition of every arm's worst horizon into
`d_se` / `d_bias` / `d_other`:
[`docs/examples/coverage/lp_family.py`](coverage/lp_family.py).

### Predictive intervals

A predictive interval targets a future *realisation*, not a parameter, but the
promise has the same form. DGPs: AR(1) at φ ∈ {0.9, 0.5}, T=100; ARMA(1,1)
φ=0.6 θ=0.4; a stationary VAR(1) with A = [[.7,.15],[.1,.6]] and
Σ = [[1,.4],[.4,1]] at T ∈ {100, 800}; a random walk with drift 0.1 at
T ∈ {100, 60}. Nominal **95%**.

Two devices carry the argument, and both are stronger than a bare coverage
number.

**The oracle column.** Every library interval here is a *plug-in* interval: it
evaluates the textbook Gaussian formula at the estimated parameters and ignores
the sampling error in them. Run the identical formula at the **true** parameters
and it covers at nominal. The gap is therefore not a wrong standard error — it
is the price of not knowing the parameters, measured rather than asserted:

```text
  h |       library |    replicated |        oracle |  width(lib)
  1 |  93.3  (0.95) |  93.3  (0.95) |  94.6  (0.86) |       3.867
  4 |  91.3  (1.07) |  91.3  (1.07) |  94.4  (0.87) |       6.397
  8 |  90.3  (1.12) |  90.3  (1.12) |  94.3  (0.88) |       7.418
 12 |  90.7  (1.10) |  90.7  (1.10) |  95.9  (0.75) |       7.784
  plug-in cost (oracle minus library, PAIRED on the same reps, pp): h1:+1.3+/-0.6 h4:+3.1+/-0.8 h8:+4.0+/-0.9 h12:+5.1+/-1.0
```

(`replicated` is the shipped band rebuilt by hand as mean ± z·se; it agrees to
4.4e-15 over 700 fits, which is how we know the interval *is* the classical
conditional-on-parameters interval and nothing else.)

**A closed form.** For the I(1) case the omitted term is available exactly, so
the coverage the shipped band *must* attain is computable in advance:

```text
  h |       library |        oracle |     corrected |  width(lib)
  1 |  93.3  (0.64) |  94.3  (0.60) |  93.5  (0.64) |       3.868
 12 |  91.7  (0.71) |  94.9  (0.57) |  94.5  (0.59) |      13.401
 24 |  90.3  (0.76) |  95.9  (0.51) |  94.5  (0.59) |      18.951
  closed-form prediction for the shipped band: h1:94.8% h12:92.6% h24:90.2%
  measured minus predicted (pp): h1:-1.5 h12:-0.9 h24:+0.2
```

`arima_fit(0,1,0)` reports `forecast_se = σ̂·√h` **exactly** (to 2.7e-15) — the
h²/(T−1) drift-uncertainty term is omitted. The prediction
`2Φ(z/√(1+h/(T−1)))−1` gives 90.2% at h=24 and the measurement is 90.3%. When
measurement matches prediction, the shortfall is not merely observed, it is
*explained*. Restoring the term recovers 94.5%.

The nominal level is a real level, not a knob — `var_forecast` coverage is
strictly increasing in the requested level at every horizon:

```text
  h |   50% nom      80% nom      90% nom      95% nom      99% nom
  1 |   49.1 (-0.9)    79.6 (-0.4)    89.2 (-0.8)    94.2 (-0.8)    98.8 (-0.2)
  4 |   46.9 (-3.1)    76.6 (-3.4)    87.2 (-2.8)    92.9 (-2.1)    98.2 (-0.8)
  6 |   47.4 (-2.6)    77.1 (-2.9)    87.1 (-2.9)    92.9 (-2.1)    98.2 (-0.8)
```

Full report: [`docs/examples/coverage/forecast_intervals.py`](coverage/forecast_intervals.py).

### Bayesian bands and identified sets

This family exists mostly to get the *question* right. DGPs: a `PERSIST` VAR(1)
with own lags 0.85 (largest root 0.905) at T=100; white noise fitted as a VAR(1)
so the δ=0 prior mean is *exactly* correct; an SVAR at T=200 whose true impact
matrix satisfies every imposed sign; a truly recursive VAR at T=200; and
`y_t = e_t + δ·1{t ≥ T/2}` for the break experiments. Nominal **90%**.

The prior sweep is the whole story for `bvar_irf_draws` — and note that the
*best* row is both better-covering **and narrower** than the default, which is
the signature of a centring problem rather than a scale one:

```text
  coverage of the y0 <- shock0 orthogonalised IRF, nominal 0.90
  design                  h=0          h=1          h=2          h=4          h=8          h=12
  default                 0.790+-0.015 0.770+-0.016 0.686+-0.018 0.610+-0.018 0.600+-0.019 0.624+-0.018
  random walk             0.883+-0.012 0.844+-0.014 0.844+-0.014 0.857+-0.013 0.867+-0.013 0.873+-0.013
  oracle                  0.877+-0.012 0.794+-0.015 0.801+-0.015 0.813+-0.015 0.826+-0.014 0.827+-0.014
  oracle-tight            0.887+-0.012 0.881+-0.012 0.887+-0.012 0.906+-0.011 0.897+-0.011 0.883+-0.012
  over-tight              0.889+-0.012 0.874+-0.013 0.831+-0.014 0.730+-0.017 0.561+-0.019 0.403+-0.019
  diffuse                 0.869+-0.013 0.780+-0.016 0.777+-0.016 0.789+-0.015 0.806+-0.015 0.807+-0.015
  emp-Bayes d=0           0.886+-0.012 0.779+-0.016 0.770+-0.016 0.767+-0.016 0.779+-0.016 0.784+-0.016
  emp-Bayes d=1           0.883+-0.012 0.847+-0.014 0.849+-0.014 0.850+-0.013 0.874+-0.013 0.880+-0.012
  SSVS spike-slab         0.899+-0.011 0.877+-0.012 0.814+-0.015 0.711+-0.017 0.621+-0.018 0.594+-0.019

  WHY: median bias / mean band width / MC sd of the posterior median (y0 <- shock0)
  design                  h=0              h=2              h=8
  default                 +0.050/0.24/0.08 -0.117/0.33/0.13 -0.194/0.37/0.12
  oracle-tight            -0.010/0.23/0.07 -0.047/0.26/0.07 -0.094/0.34/0.08
  over-tight              -0.003/0.23/0.07 -0.048/0.20/0.06 -0.101/0.18/0.03
```

`over-tight` is the mirror-image warning: a *correctly centred* prior pushed to
λ₁=0.02 collapses to 0.403 at h=12, because being right about the own lags is
not enough when the true cross lags are crushed to zero.

And the three set-identified objects, which answer three different questions and
should never be read as one:

```text
  y0 <- shock0 structural IRF (truth: h0=+1.000 h1=+0.660 h2=+0.431 h3=+0.279 h4=+0.180 h6=+0.074)
    h=0 is sign-restricted: the set is open down to 0, so everything covers there.
  design                  h=0          h=1          h=2          h=3          h=4          h=6
  pointwise band, l1=0.2  0.993+-0.004 0.760+-0.021 0.688+-0.023 0.698+-0.023 0.710+-0.023 0.728+-0.022
  pointwise band, l1=5.0  0.988+-0.006 0.860+-0.017 0.848+-0.018 0.860+-0.017 0.873+-0.017 0.885+-0.016
  set envelope, l1=0.2    1.000+-0.000 0.970+-0.009 0.950+-0.011 0.953+-0.011 0.948+-0.011 0.945+-0.011
  robust CI, l1=0.2       1.000+-0.000 0.887+-0.016 0.815+-0.019 0.800+-0.020 0.792+-0.020 0.777+-0.021
  robust CI, l1=5.0       0.998+-0.002 0.948+-0.011 0.930+-0.013 0.935+-0.012 0.932+-0.013 0.930+-0.013
```

The ordering `pointwise band < robust CI < set envelope` **is** what the three
objects mean; only the middle one aims at 0.90. And the h=0 column certifies
nothing at all: a weak sign restriction leaves the identified set open down to
zero, so the truth is inside whatever the data say. Impact coverage of a
sign-restricted band is arithmetic, not calibration.

Full report, including `narrative_svar` (true statements tighten the h=1 band
from 0.486 to 0.368 while coverage moves 0.724 → 0.736 — and the ARW bands are
**not nested**, because importance reweighting can move an individual quantile
outward) and the `bai_perron` break-date experiments:
[`docs/examples/coverage/bayes_and_sets.py`](coverage/bayes_and_sets.py).

---

## Check one number yourself

Three short, seeded programs. Each one runs in under a second and reproduces a
finding from the tables above without the surrounding harness.

**1. The delta-method IRF band loses coverage in the horizon.** This is a
500-replication sketch of the 2000-replication measurement above, so expect
third-decimal differences (and note it starts the series at zero with a burn-in,
where the module draws an exactly-stationary initial condition):

```python
import numpy as np
import tsecon

# The BASE DGP of docs/examples/coverage/irf_bands.py: a stationary VAR(1)
# whose population orthogonalised IRF is exactly A**h @ chol(Sigma).
A = np.array([[0.70, 0.10], [0.15, 0.50]])
P = np.linalg.cholesky(np.array([[1.0, 0.4], [0.4, 2.0]]))
truth = np.array([(np.linalg.matrix_power(A, h) @ P)[1, 0] for h in range(13)])

reps, T, burn = 500, 200, 200
hits = np.zeros(13)
for r in range(reps):
    rng = np.random.default_rng([20260729, r])
    y = np.zeros((burn + T, 2))
    for t in range(1, burn + T):
        y[t] = A @ y[t - 1] + P @ rng.standard_normal(2)
    b = tsecon.var_irf_bands(y[burn:], lags=1, horizon=12, orth=True,
                             method="asymptotic", alpha=0.10)
    lo = np.asarray(b["lower"])[:, 1, 0]
    hi = np.asarray(b["upper"])[:, 1, 0]
    hits += (lo <= truth) & (truth <= hi)

cov = hits / reps
se = np.sqrt(cov * (1.0 - cov) / reps)
print(f"nominal 90% delta-method band, response of y1 to shock 0, T={T}, reps={reps}")
for h in (0, 4, 8, 12):
    print(f"  h={h:<3d} truth {truth[h]:.4f}   coverage {cov[h]:.3f} +- {se[h]:.3f}")
```

```text
nominal 90% delta-method band, response of y1 to shock 0, T=200, reps=500
  h=0   truth 0.4000   coverage 0.930 +- 0.011
  h=4   truth 0.1753   coverage 0.878 +- 0.015
  h=8   truth 0.0596   coverage 0.826 +- 0.017
  h=12  truth 0.0197   coverage 0.784 +- 0.018
```

**2. `weight="hac"` used to be the White estimator, and no longer is.** No
Monte Carlo needed — the old behaviour was an identity, and the fix is visible
in one draw. This audit found it; `0.2.0` closed it. The `"robust"` and
`bandwidth=10` numbers below are unchanged from the original run, which is the
point: only the default moved.

```python
import numpy as np
import tsecon

rng = np.random.default_rng(20260729)
T = 250
z = rng.standard_normal((T, 2))                      # two instruments
e = np.zeros(T)                                      # AR(1) errors, phi = 0.8
u = rng.standard_normal(T)
for t in range(1, T):
    e[t] = 0.8 * e[t - 1] + u[t]
x = (0.6 * z.sum(axis=1) + 0.7 * e + rng.standard_normal(T)).reshape(-1, 1)
y = 1.0 * x[:, 0] + e

robust = tsecon.iv_gmm(x, z, y, method="2step", weight="robust")
hac_auto = tsecon.iv_gmm(x, z, y, method="2step", weight="hac")
hac_bw10 = tsecon.iv_gmm(x, z, y, method="2step", weight="hac", bandwidth=10.0)

print(f'weight="robust"                  se = {robust["bse"][0]:.6f}')
print(f'weight="hac"  (auto: {hac_auto["hac_bandwidth"]:.0f} lags)     se = {hac_auto["bse"][0]:.6f}')
print(f'weight="hac", bandwidth=10.0     se = {hac_bw10["bse"][0]:.6f}')
print(f'|hac(auto) - robust|             = {abs(hac_auto["bse"][0] - robust["bse"][0]):.3e}')

try:
    tsecon.iv_gmm(x, z, y, method="2step", weight="hac", bandwidth=0.0)
except ValueError as exc:
    print(f'\nbandwidth=0.0 -> ValueError: {str(exc)[:72]}...')
print(f'\nfirst-stage F on the endogenous regressor: '
      f'{hac_auto["first_stage"][0]["fstat"]:.1f}')
```

```text
weight="robust"                  se = 0.204701
weight="hac"  (auto: 4 lags)     se = 0.198534
weight="hac", bandwidth=10.0     se = 0.193519
|hac(auto) - robust|             = 6.167e-03

bandwidth=0.0 -> ValueError: bandwidth=0.0 with weight="hac" is a no-op: a Bartlett kernel truncated ...

first-stage F on the endogenous regressor: 22.1
```

Before `0.2.0` the second line read `0.204701` and the fourth read
`0.000e+00` — bit-identical to `"robust"`, in every one of 3000 replications.
Now the default resolves to the Newey-West rule of thumb
`floor(4 (n/100)^(2/9))` = 4 lags at `T=250`, reports that choice back as
`hac_bandwidth`, and refuses an explicit `0.0` rather than honouring it.

**The fix does not repair the coverage**, and the numbers say so: 0.632 at the
old no-op default, **0.842** at the automatic rule, 0.868 at `bandwidth=10`
against a nominal 0.95. A working default is not a remedy — under moments this
persistent, `T=250` does not contain enough independent information to estimate
the long-run variance, and the automatic rule picks *fewer* lags than the
setting that reached 0.868. The last line is the other half of the fix: the
first-stage F is now reported, so the caller can see the instrument strength
their interval rests on.

**3. The omitted drift-uncertainty term in an I(1) forecast band.** Also an
identity:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(20260729)
T, H = 100, 12
y = np.cumsum(0.1 + rng.standard_normal(T))          # random walk with drift 0.1

fit = tsecon.arima_fit(y, p=0, d=1, q=0, constant=True,
                       forecast_steps=H, conf_alpha=0.05)
se = np.asarray(fit["forecast_se"])
sigma = se[0]                                        # forecast_se at h=1 IS sigma
h = np.arange(1, H + 1)

shipped = sigma * np.sqrt(h)                          # what the library reports
correct = sigma * np.sqrt(h + h**2 / (T - 1))         # + drift uncertainty
print(f"max |forecast_se - sigma*sqrt(h)| = {np.abs(se - shipped).max():.2e}")
print(" h   shipped se   with drift term   ratio")
for i in (0, 3, 7, 11):
    print(f"{h[i]:2d}   {shipped[i]:9.4f}   {correct[i]:15.4f}   "
          f"{shipped[i] / correct[i]:.4f}")
```

```text
max |forecast_se - sigma*sqrt(h)| = 4.44e-16
 h   shipped se   with drift term   ratio
 1      0.9666            0.9715   0.9950
 4      1.9332            1.9719   0.9804
 8      2.7340            2.8423   0.9619
12      3.3484            3.5456   0.9444
```

---

## The data-generating processes

Every number on this page is conditional on the process that produced it. These
are the canonical textbook cases plus the stress regimes applied work lives in;
they are **not** an exhaustive sweep, and a number here is evidence about a
mechanism rather than a constant for the function.

| family | processes | nominal | replications |
|---|---|---|---|
| `regression_se` | `y = 1 + 2x + e` with iid Gaussian / sd(e)=\|x\| / AR(1) errors with an AR(1) regressor at φ=0.7 / t(3) errors. A high-leverage design x ~ chi2(1), sd(e\|x)=x at T ∈ {25,…,1600}. A HAC slope design with x and e both AR(1) at φ ∈ {0, 0.5, 0.8, 0.95}. IV over-identified by 1 with corr(e,v)=0.7 and π ∈ {0.6, 0.2, 0.05}. A Corsi HAR with truth [−0.2, 0.35, 0.35, 0.25], Σb = 0.95. A probit/logit with an AR(1) index at φ=0.9 in a common (rate 0.46) and a rare (rate 0.055) regime. A quantile design `y = a + bx + (s₀ + s₁x)u`, x ~ U(0,2) | 95% | 3000 |
| `irf_bands` | `BASE` VAR(1) root 0.758; `PERSIST` VAR(1) root 0.950; `LAG4` VAR(4) root 0.900 fitted as a VAR(1); all Σ = [[1, 0.4], [0.4, 2]], n ∈ {100, 200, 500} | 90% | 2000 (399 bootstrap draws each) |
| `lp_family` | `y_t = Σ_j 0.7^j s_{t−j} + nuisance_t` truncated at J=25, T ∈ {100, 200, 400, 800}; a heteroskedastic variant; strong vs weak LP-IV; a two-state Markov design with P(stay)=0.9; a persistent-impulse multiplier design at ρ_x=0.8 | 95% | 700–4000 by experiment |
| `forecast_intervals` | AR(1) at φ ∈ {0.9, 0.5}, T=100; ARMA(1,1) φ=0.6 θ=0.4; VAR(1) A = [[.7,.15],[.1,.6]], Σ = [[1,.4],[.4,1]], T ∈ {100, 800}, fitted lags ∈ {1, 4}; random walk with drift 0.1 at (T=100, H=12) and (T=60, H=24) | 95% (plus a 50/80/90/95/99 sweep) | 600–6000 by experiment |
| `bayes_and_sets` | `PERSIST` VAR(1) own lags 0.85 at T=100 under nine priors; white noise fitted as a VAR(1) so δ=0 is exactly right; a sign-restricted SVAR at T=200 whose true A₀ satisfies every imposed sign; a truly recursive VAR at T=200; `y_t = e_t + δ·1{t ≥ T/2}` at T ∈ {200, 400, 800} and δ/σ ∈ {3, 2, 1, 0.5, 0.25} | 90% | 250–2500 by experiment |

---

## What is *not* measured

Stated so you do not have to discover it.

- **No simultaneous (sup-t) band exists to measure.** Every band on this page is
  pointwise or marginal. The joint rates in
  [pointwise is not joint](#pointwise-is-not-joint) are what a reader gets by
  mistake, not what a joint band would deliver.
- **No weak-instrument-robust set exists to measure *for these two rows*.**
  Anderson-Rubin is the right interval for the `iv_gmm` and `lp_iv`
  weak-instrument rows, and neither function exposes one. The library does now
  ship `proxy_ar_sets` — an Anderson-Rubin set for proxy-SVAR impulse
  responses — so the machinery exists and the gap is that it has not been
  extended to the IV regression estimators, not that it is unbuilt.
- **`lp(cumulative=...)` intervals are unmeasured.** `var_irf_bands`'
  cumulative bands are measured; LP's are not, and the two are different code
  paths.
- **`quantile_lp`, `growth_at_risk`, panel LP (Driscoll-Kraay), `favar`,
  `proxy_svar`, `nongaussian_svar`, GARCH forecast intervals, `dfm_nowcast`,
  MIDAS, and the term-structure fits are not in this audit.** They have golden
  and property coverage; they do not yet have measured interval coverage.
- **Only two nominal levels are swept.** `var_irf_bands` is measured at 68/90/95
  and `var_forecast` at 50/80/90/95/99. Everything else is measured at one
  level; coverage shortfalls are not guaranteed to scale linearly in α, and in
  fact the IRF sweep shows they do not — at h=12 the loss is 4.9pp at nominal
  68%, 13.7pp at 90% and 15.0pp at 95%.
- **One machine, one build.** These are statistical properties, not timings, so
  they are machine-independent — but they are measured against this working
  tree's release build of the extension.
- **Coverage is not the only thing that matters.** An interval can cover at
  nominal and be useless if it is enormous (the weak-instrument rows are exactly
  this). Where width is the story, the family reports print widths too.

---

## What this audit recommends

Ordered by how much coverage each would buy, and separated into what the
library should change and what a caller should change.

**For the library — shipped in `0.2.0`.** Three of the five recommendations
below were acted on. They are kept here, marked, rather than deleted: an audit
that quietly edits out its own findings once they are fixed cannot be checked
against later.

1. ~~**Add `hc2`/`hc3` to `ols`'s `se_type` menu.**~~ **Done.** Both match
   statsmodels HC2/HC3 to 2.96e-15. Measured on the same design that motivated
   it: tsecon's own `hc3` covers **0.863 ± 0.006** at T=25 where `hc1` covers
   0.682 — and is still short of nominal, so prefer it without treating it as a
   cure. The gap closes with n (0.910 at T=100, 0.945 at T=1600). (`se_type`
   remains lower-case only — `"HC0"` and `"HAC"` raise `ValueError`.)
2. ~~**Make `iv_gmm(weight="hac")` refuse to silently degrade.**~~ **Done, and
   it is a breaking change.** `bandwidth` now defaults to `None`, which selects
   the Newey-West rule; an explicit `0.0` raises; the resolved truncation comes
   back as `hac_bandwidth`. Coverage moves 0.632 → **0.842**, which is the
   honest number: it buys 21 points and **still misses nominal**, so this closed
   a silent-wrongness bug, not the coverage gap.
3. **Expose a simultaneous band.** *Still open, and now the largest gap on this
   page.* A sup-t or Bonferroni band for `var_irf_bands`, `lp`, and
   `var_forecast` would close the largest *reading* gap — 90% pointwise is 72%
   joint at T=500, and the joint rate does not improve toward the marginal one
   as T grows.
4. **Expose an Anderson-Rubin set for LP-IV and `iv_gmm`.** *Half done.*
   `iv_gmm` now reports `first_stage`, so the caller can at least see the
   instrument strength — but the audit also showed F > 10 is not a safe
   threshold (0.915 coverage at median F = 10.5), which is precisely why a
   weak-IV-robust set is still the real answer. Under weak identification no
   bounded Wald set can be honest.
5. **A per-τ convergence flag for `quantile_regression`.** *Still open.* The
   single shared `converged` bool trips on 232/3000 replications at T=200 and is
   the IRLS iteration cap, not an estimation failure — dropping those
   replications moves τ=0.05 coverage only from 0.866 to 0.870.

**For a caller, in order of how often it will bite.**

1. `se_type="hc1"` is not serial-correlation robust. Use `"hac"`, and choose the
   bandwidth for the persistence you actually have.
2. On a persistent VAR, pass `bias_correct=True` to `var_irf_bands`. 0.410 →
   0.900 at h=12.
3. Prefer the bootstrap band at long horizons and prefer `cumulative=True` where
   the cumulative response is what you mean.
4. Keep `lp`'s default `se="lag_augmented"`. It wins at every horizon measured.
5. Read a smooth-LP band as a band around the *penalized* estimand, and never as
   a confidence interval for the impact response.
6. Read `first_stage_f` before you read any IV standard error — and do not treat
   F = 10 as a green light: coverage there is already 0.915.
7. Read `recession_probit`'s failure share with its coverage. A quarter of rare-
   event samples at T=100 have no finite MLE.
8. Do not read the HAR intercept as if it were as reliable as the HAR slopes.
9. Read a fan chart one horizon at a time.

---

## See also

- **[Testing & validation](../reference/testing.md)** — where this tier sits
  among the nine, and what each of the others can and cannot prove.
- **[Validation matrix](../reference/validation-matrix.md)** — the per-family
  table of references, fixtures and tolerances that pins the *point* estimates.
- **[Monte Carlo validation](monte-carlo.md)** — test size, HAC coverage for a
  mean, and estimator consistency.
- **[Frontier Monte Carlo](monte-carlo-frontier.md)** — LP vs VAR bias/variance,
  and weak-instrument LP-IV.
- **[HAC standard errors](../cookbook/hac-standard-errors.md)** and
  **[Confidence bands on a VAR IRF](../cookbook/var-irf-bands.md)** — the
  recipes these caveats attach to.
- The five modules themselves, each with a derivation of its truth in its
  docstring:
  [`regression_se.py`](coverage/regression_se.py),
  [`irf_bands.py`](coverage/irf_bands.py),
  [`lp_family.py`](coverage/lp_family.py),
  [`forecast_intervals.py`](coverage/forecast_intervals.py),
  [`bayes_and_sets.py`](coverage/bayes_and_sets.py),
  and the runner [`run_all.py`](coverage/run_all.py).
