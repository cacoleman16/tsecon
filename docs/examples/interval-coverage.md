# Interval coverage

A confidence interval is a **promise about repeated samples**: across
independent draws from the same data-generating process, a nominal 95% interval
should contain the truth 95% of the time. Every other validation tier in this
repository proves that `tsecon`'s *point estimates and standard-error algebra*
match an independent reference. None of them proves the promise itself.

This page is that proof, and it is an audit rather than an advertisement.
**50 interval-valued outputs across 28 functions** — a function paired with the
option and regime that change its answer — were re-estimated on thousands of
seeded draws from data-generating processes whose truth is known in closed form,
and the containment rate was counted. Where an interval covers at its nominal rate, the
number is here. Where it does not, the number is *also* here, with the
Monte Carlo standard error of the measurement next to it, an attribution of
*why*, and what to do instead.

!!! warning "The three headline results, before any table"

    1. **12 of the 39 frequentist intervals miss their nominal rate even in the
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
       the marginal one. ~~No function in this library reports a simultaneous
       band.~~ **A simultaneous (sup-t) band now exists** — scored against a
       pointwise arm on the *same* replications it reaches **85.2% ± 1.1**
       (IRF, nominal 90%) and **90.5% ± 0.7** (forecast, nominal 95%). That is
       the largest repair on this page and **neither VAR arm reaches nominal**:
       [what it buys and what it cannot](#pointwise-is-not-joint).

---

## Fixed in 0.2.0

This audit was written against `0.1.0`. Four of its findings were defects rather
than approximation error, and `0.2.0` closed all four. The rows they came from
are still in the tables below, now measuring the shipped `0.2.0` behaviour —
the before-numbers live in this table, because an audit that deletes its
findings once they are fixed leaves nothing to check later, and the *sizes* of
these gaps are the argument for measuring coverage at all.

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
# the whole audit: seven families, one consolidated report
.venv/bin/python docs/examples/coverage/run_all.py            # 1904 s here

# just the consolidated tables, without the seven per-family reports
.venv/bin/python docs/examples/coverage/run_all.py --summary

# a smoke run; the MC standard errors are 2-4x larger, so do not quote it
.venv/bin/python docs/examples/coverage/run_all.py --quick     # ~40 s

# or one family at a time
.venv/bin/python docs/examples/coverage/regression_se.py       # 108 s
.venv/bin/python docs/examples/coverage/irf_bands.py           # 417 s
.venv/bin/python docs/examples/coverage/lp_family.py           # 192 s
.venv/bin/python docs/examples/coverage/forecast_intervals.py  # 236 s
.venv/bin/python docs/examples/coverage/bayes_and_sets.py      # 240 s
.venv/bin/python docs/examples/coverage/quantile_panel_lp.py   # 680 s
.venv/bin/python docs/examples/coverage/factor_midas.py        #  32 s

# the page-vs-registry row inventory (no Monte Carlo; runs in the test suite)
.venv/bin/python docs/examples/coverage/check_page.py          #  <1 s

# regenerate Table 1 / Table 2 below from the harvested results -- the rows
# on this page are pasted from this output, never typed
.venv/bin/python docs/examples/coverage/run_all.py --markdown tables.md
```

Every family draws from one master seed, `20260729`, and prints it. Every draw
is `default_rng([seed, experiment, replication])`, so the numbers do not depend
on call ordering, on `PYTHONHASHSEED`, or on whether you run one family or all
seven — verified: every measured line from the standalone runs appears
identically in the consolidated run. Each family asserts its own qualitative
findings and **exits non-zero if they stop holding**, so a statistical
regression fails a build rather than quietly rotting in this page.

```text
  module                   assertions   runtime   result
  ----------------------------------------------------------------------
  regression_se.py           110 pass    108.3s   OK
  irf_bands.py                67 pass    417.0s   OK
  lp_family.py                16 pass    191.6s   OK
  forecast_intervals.py       13 pass    235.8s   OK
  bayes_and_sets.py           23 pass    239.5s   OK
  quantile_panel_lp.py        17 pass    679.8s   OK
  factor_midas.py              9 pass     31.6s   OK

  50 probes harvested from 7 families in 1904.4s
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

## The headline: 50 measured interval-valued outputs

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
| `tsecon.ols` | se_type="hc3"; small T, leverage | CI | 0.95 | x~chi2(1), sd(e\|x)=x, T=1600 | 0.945 ± 0.004 | -1.3 | -0.5 | at nominal |
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
| `tsecon.quantile_lp` | Powell sandwich, identified iid shock | CI | 0.95 | iid T=400 tau=0.50, h=0 | 0.939 ± 0.008 | -1.5 | -1.1 | at nominal |
| `tsecon.quantile_lp` | persistent regressor (phi=0.8) | CI | 0.95 | p=4, worst h (6) | 0.909 ± 0.009 | -4.5 | -4.1 | **UNDER** |
| `tsecon.panel_lp` | se_type="driscoll_kraay" (the default) | CI | 0.95 | N=50 T=80 dk, h=0 | 0.916 ± 0.006 | -6.2 | -3.4 | **UNDER** |
| `tsecon.panel_lp` | bias_correction="spj" at short T | CI | 0.95 | SPJ, N=50, T=40, h=2 | 0.843 ± 0.007 | -14.7 | -10.7 | **UNDER** |
| `tsecon.lp` | cumulative="both" (default se resolves to "hac") | CI | 0.95 | T=400, h=12 (the pre-fix 0.507 cell) | 0.920 ± 0.006 | -4.9 | -3.0 | **UNDER** |
| `tsecon.favar` + `tsecon.var_irf_bands` | two-step bands conditioned on F-hat | CI | 0.90 | N=100 T=200 F-hat, h=0 | 0.887 ± 0.007 | -1.8 | -1.3 | at nominal |
| `tsecon.umidas` | se_type="hac" (the default) | CI | 0.95 | iid errors, T=300, k=1 (most recent lag) | 0.947 ± 0.004 | -0.8 | -0.3 | at nominal |
| `tsecon.theta_forecast` / `tsecon.backtest` | no interval is returned | NONE | — | the library returns a point path only | — | — | — | no band |
| `tsecon.weighted_midas` | no interval is returned | NONE | — | NLS point fit, weights and fit diagnostics only | — | — | — | no band |
| `tsecon.dfm_nowcast` | no interval is returned | NONE | — | point nowcast + smoothed factor path only | — | — | — | no band |
| `tsecon.nelson_siegel` | no interval is returned | NONE | — | factors, fitted lambda, residuals and R^2 only | — | — | — | no band |

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
| `tsecon.bvar_ssvs` | spike-and-slab credible band | CRED | 0.90 | prior 'SSVS spike-slab', h=12 | 0.594 ± 0.019 | -16.5 | -30.6 | n/a |
| `tsecon.smooth_lp` | lam="cv" (the default) | CI | 0.95 | lam=cv, h=0 | 0.646 ± 0.018 | -16.8 | -30.4 | **UNDER** |
| `tsecon.bvar_irf_draws` | 5th/95th posterior percentile band | CRED | 0.90 | prior 'default', h=4 | 0.610 ± 0.018 | -15.7 | -29.0 | n/a |
| `tsecon.ols` | se_type="hc1"; small T, leverage | CI | 0.95 | x~chi2(1), sd(e\|x)=x, T=25 | 0.682 ± 0.009 | -31.6 | -26.8 | **UNDER** |
| `tsecon.favar` + `tsecon.var_irf_bands` | two-step bands conditioned on F-hat | CI | 0.90 | N=20 T=800, worst h (7) | 0.673 ± 0.010 | -21.6 | -22.7 | **UNDER** |
| `tsecon.var_irf_bands` | method="asymptotic" | CI | 0.90 | n=100, h=12 | 0.673 ± 0.010 | -21.6 | -22.7 | **UNDER** |
| `tsecon.ols` | se_type="nonrobust" | CI | 0.95 | heteroskedastic sd=\|x\|, T=200 | 0.732 ± 0.008 | -27.0 | -21.8 | **UNDER** |
| `tsecon.ols` | se_type="hc1" | CI | 0.95 | AR(1) errors+regressor .7, T=200 | 0.735 ± 0.008 | -26.7 | -21.5 | **UNDER** |
| `tsecon.sign_restricted_svar` | pointwise 5-95 band over rotations | CRED | 0.90 | lambda1=0.2, h=3 | 0.698 ± 0.023 | -8.8 | -20.2 | n/a |
| `tsecon.quantile_lp` | persistent regressor (phi=0.8) | CI | 0.95 | p=0, worst h (6) | 0.758 ± 0.014 | -14.2 | -19.2 | **UNDER** |
| `tsecon.panel_lp` | bias_correction="spj" at short T | CI | 0.95 | SPJ, N=50, T=20, h=2 | 0.761 ± 0.009 | -22.1 | -18.9 | **UNDER** |
| `tsecon.var_irf_bands` | pointwise band read as a JOINT band | CI | 0.90 | n=500, all of h=0..12 at once | 0.722 ± 0.010 | -17.8 | -17.8 | **UNDER** |
| `tsecon.panel_lp` | se_type="driscoll_kraay" (the default) | CI | 0.95 | N=50 T=40, worst h (4) | 0.818 ± 0.008 | -17.1 | -13.2 | **UNDER** |
| `tsecon.umidas` | se_type="hac" (the default) | CI | 0.95 | AR(1) errors phi=0.7, T=150, intercept | 0.829 ± 0.007 | -17.6 | -12.1 | **UNDER** |
| `tsecon.iv_gmm` | 2sls / 2step / iterated | CI | 0.95 | median first-stage F = 1.2 | 0.839 ± 0.007 | -16.5 | -11.1 | **UNDER** |
| `tsecon.var_irf_bands` | cumulative=True | CI | 0.90 | orth=True,cumulative=False, h=12 | 0.789 ± 0.009 | -12.2 | -11.1 | **UNDER** |
| `tsecon.iv_gmm` | weight="hac" | CI | 0.95 | AR(1) errors phi=0.8, hac auto (NW rule) | 0.842 ± 0.007 | -16.3 | -10.8 | **UNDER** |
| `tsecon.zero_sign_svar` | band at a TRUE point-identifying zero | CRED | 0.90 | lambda1=0.2, h=3 | 0.797 ± 0.020 | -5.1 | -10.3 | n/a |
| `tsecon.robust_svar_bounds` | Giacomini-Kitagawa robust region | CRED | 0.90 | lambda1=0.2, h=3 | 0.800 ± 0.020 | -5.0 | -10.0 | n/a |
| `tsecon.ols` | se_type="hc3"; small T, leverage | CI | 0.95 | x~chi2(1), sd(e\|x)=x, T=25 | 0.863 ± 0.006 | -13.9 | -8.7 | **UNDER** |
| `tsecon.quantile_regression` | Powell sandwich, slope | CI | 0.95 | location-scale T=200, tau=0.05 | 0.866 ± 0.006 | -13.5 | -8.4 | **UNDER** |
| `tsecon.lp` | se="hac" | CI | 0.95 | T=100 hac, h=12 | 0.870 ± 0.008 | -10.7 | -8.0 | **UNDER** |
| `tsecon.har_rv` | HAC SE on the CONSTANT | CI | 0.95 | het innovations, maxlags=22, const | 0.871 ± 0.006 | -12.9 | -7.9 | **UNDER** |
| `tsecon.bai_perron` | break-date CI, conditional on detection | CI | 0.95 | break/sigma=0.5, T=200, cond. on detection | 0.888 ± 0.008 | -8.0 | -6.2 | **UNDER** |
| `tsecon.var_irf_bands` | method="bootstrap", bias_correct=True | CI | 0.90 | n=100, h=0 | 0.842 ± 0.008 | -7.2 | -5.8 | **UNDER** |
| `tsecon.quantile_lp` | Powell sandwich, identified iid shock | CI | 0.95 | iid T=200 tau=0.25, worst h (0) | 0.893 ± 0.010 | -5.8 | -5.7 | **UNDER** |
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
| `tsecon.lp` | cumulative="both" (default se resolves to "hac") | CI | 0.95 | T=1600, most extreme h (4) | 0.971 ± 0.004 | +5.4 | +2.1 | **OVER** |
| `tsecon.lp_iv` | weak instrument | CI | 0.95 | weak iv, worst h (4) | 0.985 ± 0.002 | +15.5 | +3.5 | **OVER** |
| `tsecon.quantile_regression` | Powell sandwich, intercept | CI | 0.95 | location-scale T=200, tau=0.5 | 0.993 ± 0.002 | +28.2 | +4.3 | **OVER** |
| `tsecon.bai_perron` | break-date CI at a LARGE break | CI | 0.95 | break/sigma=3.0, T=200, cond. on detection | 0.998 ± 0.001 | +54.2 | +4.8 | **OVER** |
| `tsecon.sign_restricted_svar` | set envelope (min/max over draws) | SET | 0.90 | lambda1=0.2, h=3 | 0.953 ± 0.011 | +4.9 | +5.2 | n/a |
| `tsecon.theta_forecast` / `tsecon.backtest` | no interval is returned | NONE | — | the library returns a point path only | — | — | — | no band |
| `tsecon.weighted_midas` | no interval is returned | NONE | — | NLS point fit, weights and fit diagnostics only | — | — | — | no band |
| `tsecon.dfm_nowcast` | no interval is returned | NONE | — | point nowcast + smoothed factor path only | — | — | — | no band |
| `tsecon.nelson_siegel` | no interval is returned | NONE | — | factors, fitted lambda, residuals and R^2 only | — | — | — | no band |

---

## Where the intervals miss

This is the section to act on. Each caveat links to the model card for the
function, so it is reachable from the function's own documentation as well as
from here.

### A. Misses even in the favourable design

**12 of 39**. These are off nominal in the design they are
entitled to do well on, so a caller cannot fix them by having better data.

| surface | favourable design | measured (nominal 0.95 / 0.90) | cause | what to do | documented in |
|---|---|---|---|---|---|
| `iv_gmm(weight="hac")` | AR(1) errors φ=0.8, `bandwidth=10` supplied | **0.868 ± 0.006** (automatic default: **0.842 ± 0.007**) | APPROXIMATION *(was CONVENTION — [fixed in 0.2.0](#fixed-in-020))* | the original finding: `bandwidth` defaulted to `0.0`, and a Bartlett kernel truncated at 0 lags **is** the White estimator, so `weight="hac"` alone changed nothing (verified bit-identical, max \|Δse\| = 0.000e+00 over 3000 reps) and covered **0.632 ± 0.009**. `0.2.0` made the default the Newey-West rule and made an explicit `0.0` an error. That lifts coverage to 0.842 and **no further** — at this persistence T=250 cannot estimate the long-run variance, and even `bandwidth=10` reaches only 0.868 | [GMM card](../reference/model-cards/gmm.md#iv_gmm-linear-iv-gmm) |
| `var_irf_bands(method="bootstrap")` | impact, persistent VAR, T=100 | **0.848 ± 0.008** (h=12: **0.410 ± 0.011**) | ESTIMATOR | the percentile band sits *below* an already downward-biased point estimate — a second dose of the same bias. Pass `method="bootstrap", bias_correct=True` on a persistent VAR: it lifts h=12 from 0.410 to **0.900**. (`bias_correct` is a property of the bootstrap resampling loop; setting it on the default asymptotic arm now raises rather than being silently ignored, which is how it shipped before 0.3.0) | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `har_rv` — the **constant** | heteroskedastic innovations, `maxlags=0` | **0.910 ± 0.005** (`maxlags=22`: **0.871 ± 0.006**) | ESTIMATOR | not the SE: the least-squares persistence bias at Σb = 0.95 is absorbed *entirely* by the intercept (measured bias −0.1000 against the mechanical prediction −0.1001). The three slopes are at nominal; do not read the HAR intercept as if it were | [Realized-vol card](../reference/model-cards/realized-vol.md#how-to-read-the-output) |
| `lp_iv` — strong instrument | median first-stage F ≈ 134, best horizon | **0.930 ± 0.005** (worst horizon: **0.909 ± 0.005**) | CONVENTION | the kernel covariance follows `linearmodels`' `debiased=False` convention (which is what makes the point estimates match the golden) and applies *p* Bartlett lags even at h=0, where the score has nothing to smooth. Subtract two to four points from the nominal level before quoting an LP-IV interval | [LP card](../reference/model-cards/local-projections.md#lp_iv-instrumented-local-projections-lp-iv) |
| `lp_multiplier` | impact, median F ≈ 159 | **0.934 ± 0.005** (widest window: **0.903 ± 0.005**) | CONVENTION | well centred (\|bias\|/sd ≤ 0.08) and strongly instrumented at every horizon, but se/sd ≈ 0.9: the honest critical value at T=240 is nearer 2.2 than 1.96 | [LP card](../reference/model-cards/local-projections.md#lp_multiplier-integral-multipliers) |
| `lp_iv` — weak instrument | kindest horizon of the weak arm itself | **0.969 ± 0.003** (worst: **0.985 ± 0.002**) | APPROXIMATION | it **over**-covers while the median interval width explodes about 5×. That is the correct symptom, not a lucky escape: Dufour (1997) — under weak identification no *bounded* confidence set can be honest, so a Wald set stays honest only by becoming uninformative. Report `first_stage_f` | [LP card](../reference/model-cards/local-projections.md#lp_iv-instrumented-local-projections-lp-iv) |
| `bai_perron` — unconditional | break/σ = 3, T=200 | **0.967 ± 0.004** (break/σ = 0.25: **0.233 ± 0.009**) | ESTIMATOR | at a small break, detection itself collapses to 0.29, so the rate a user actually faces is 0.233. A break-date CI is meaningful only once the break is detectable | [Structural-breaks card](../reference/model-cards/structural-breaks.md#bai_perron-multiple-breaks-how-many-when-how-sure) |
| `bai_perron` — conditional on detection | break/σ = 1, T=800 | **0.970 ± 0.005** (break/σ = 0.5, T=200: **0.888 ± 0.008**) | APPROXIMATION / CONVENTION | over-covers at a large break because the half-width is `ceil(c/scale)` **plus one index** on each side, and that discreteness padding dominates (0.998 at break/σ = 3). Under-covers at a small break because Bai's argmax limit distribution is a finite-sample approximation — it improves with T (0.877 → 0.914 → 0.944 at T = 200/400/800) while the interval *width* does not shrink at all (26.4 → 26.5 → 26.1), exactly as fixed-break asymptotics predict | [Structural-breaks card](../reference/model-cards/structural-breaks.md#bai_perron-multiple-breaks-how-many-when-how-sure) |
| `quantile_lp` — persistent regressor | default lag controls (p=4), worst (τ, h) cell | **0.909 ± 0.009** (no lag controls, worst cell: **0.758 ± 0.014**) | ESTIMATOR | the Powell sandwich is heteroskedasticity-robust, **not** HAC — exactly as the card says. The default lag controls whiten an AR(1) regressor and hold every cell at 0.91–0.95; with `n_lag_controls=0` nothing whitens the score and the growth_at_risk-shaped decay appears in full (se/sd 0.66 at the far horizon, \|bias\|/sd ≈ 0.4). Keep the lag controls at least as long as the regressor's AR order — and note the canonical iid-shock design is measured near nominal everywhere | [Quantile card](../reference/model-cards/quantile.md#quantile_lp-quantile-local-projections) |
| `panel_lp(se_type="driscoll_kraay")` — the default | N=50, T=80, impact | **0.916 ± 0.006** (T=40, worst horizon: **0.818 ± 0.008**) | APPROXIMATION | with a common shock the effective sample is T, not N·T: quintupling N moves pooled coverage by under a point while doubling T buys ~5pp. Driscoll-Kraay is a T-asymptotic estimator; at T=40 read the bands as indicative. And never swap in `se_type="cluster"` under a common factor — on the same draws it covers **0.20** (se/sd = 0.12) | [Panel-LP cookbook](../cookbook/panel-lp-standard-errors.md) |
| `panel_lp(bias_correction="spj")` | T=40, h=2 (a short panel — the correction's home turf) | **0.843 ± 0.007** (T=20: **0.761 ± 0.009**) | APPROXIMATION | the SPJ removes most of the Nickell bias (−0.141 → +0.015 at T=20, h=2 — corroborating its card almost exactly) and recomputes the SEs for the corrected estimator, but Driscoll-Kraay remains a short-T approximation, so neither SPJ nor uncorrected FE reaches nominal at T=20. The measured T=20 coverage *gain* over FE is +2.5pp paired (se 0.6pp) — real, and smaller than the card's 300-replication point numbers (0.743 → 0.823) suggest; at T=40 the paired gain is −1.5pp | [Panel card](../reference/model-cards/panel.md#nickell-bias-and-the-two-half-panel-corrections-panel_lp) |
| `lp(cumulative="both")` — the post-fix HAC default | T=400, h=12 (the repaired defect cell) | **0.920 ± 0.006** (T=1600, most extreme h: **0.971 ± 0.004**, over) | APPROXIMATION *(the 0.507 defect itself was fixed in 0.3.0)* | the official repair numbers: the pre-fix lag-augmented/HC1 default covered **0.507** at h=12 and was flat in T; the mode-dependent HAC default restores h=12 to 0.920 at T=400 and 0.956 at T=1600, with the residual deviation flipping mildly *conservative* at T=1600 mid-horizons — the Bartlett bandwidth `h + p` is generous once T is large relative to the MA(h) overlap. `se="lag_augmented"` with this mode now raises, and the suite asserts it | [LP card](../reference/model-cards/local-projections.md#lp-local-projection-irfs) |

### B. At nominal when entitled, off under stress

**27 of 39**. These behave the way asymptotic approximations behave. Nothing
here is an alarm; the number to quote is the *size* of the loss in the regime
you are actually in.

| surface | stress design | measured | cause | what to do | documented in |
|---|---|---|---|---|---|
| `var_irf_bands` with the wrong lag order | VAR(4) truth fitted as VAR(1), h=4, T=500 | **0.061 ± 0.005** | ESTIMATOR | inconsistency, not a band problem: coverage gets *worse* as T grows — 17.8% at T=200, 6.2% at T=500. Choose the lag order on the data before reading any band; with the correct order the same cell covers 0.905 | [VAR/SVAR card](../reference/model-cards/var-svar.md#confidence-bands-on-the-irf-var_irf_bands) |
| `var_forecast` marginal bands read as a joint band | 12 horizons × 2 series simultaneously, T=100 | **0.409 ± 0.006** | READING | see [pointwise is not joint](#pointwise-is-not-joint) | [VAR/SVAR card](../reference/model-cards/var-svar.md#reduced-form-var-var_fit-var_irf-var_fevd-var_granger-var_forecast) |
| `ols(se_type="hac")` on a **slope** | regressor *and* errors AR(1) at φ=0.95, T=200 | **0.588 ± 0.009** | APPROXIMATION | the reported HAC SE is 0.43 of the true sampling sd. Lengthening the bandwidth helps and does not close it (0.703 at 12 lags, 0.728 at 24): at T=200 the sample does not contain enough independent information to estimate a long-run variance this large. Report such a slope with a bandwidth chosen for the persistence and treat the interval as indicative | [HAC cookbook](../cookbook/hac-standard-errors.md#choosing-the-bandwidth) |
| `smooth_lp(lam="cv")` | impact response, T=200 | **0.646 ± 0.018** | ESTIMATOR | by design, and worst exactly where an applied reader looks first. At impact the cross-validated penalty pulls the estimate off the peak (\|bias\|/sd = 1.23 — the bias exceeds a whole sampling sd). A smooth-LP band is a band around the **penalized** estimand; the unpenalized `lam=0` anchor covers 0.936 at the same cell. Separately, `se` conditions on the selected λ, so mean se/sd falls from 0.907 to 0.813 | [LP card](../reference/model-cards/local-projections.md#smooth_lp-smooth-local-projections-barnichon-brownlees) |
| `ols(se_type="hc1")`, small T with leverage | x ~ chi2(1), sd(e\|x)=x, T=25 | **0.682 ± 0.009** | ESTIMATOR *(was API GAP — [fixed in 0.2.0](#fixed-in-020))* | `hc1`'s n/(n−k) factor buys 0.015 at k=2; the leverage correction 1/(1−h_i) is what matters. `0.2.0` added `hc2`/`hc3`, and tsecon's own `hc3` now covers **0.863 ± 0.006** on these draws — 18 points recovered, and still short of nominal, so prefer `hc3` at small n without treating it as a cure (its own row is next) | [Inference guide](../guide/03-inference-toolkit.md#the-robust-standard-error-ladder) |
| `ols(se_type="hc3")`, small T with leverage | x ~ chi2(1), sd(e\|x)=x, T=25 | **0.863 ± 0.006** | APPROXIMATION | the estimator `0.2.0` added because of this audit, measured on its own row: at nominal in the favourable design (0.945 ± 0.004 at T=1600) and still 8.7pp short at T=25 — the leverage correction recovers most of the small-T gap but not all of it. The SE distribution is skewed, so the mean se/sd of 0.942 overstates the typical interval; at a small, high-leverage n, widen deliberately or bootstrap | [Inference guide](../guide/03-inference-toolkit.md#the-robust-standard-error-ladder) |
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
| `quantile_lp` — identified iid shock | worst (τ, h, T) cell of the whole grid: τ=0.25, h=0, T=200 | **0.893 ± 0.010** | APPROXIMATION | the audit's cleanest good news: the quantile card's transferred growth_at_risk warning does **not** bind on an identified iid shock, because the check-loss score is serially uncorrelated by construction (the same mechanism that makes lag-augmented mean-LP inference work). The worst cell is the location-scale tail *impact* — a Powell kernel density-estimation cost, not an overlap one — and it shrinks in T (0.923 at T=400) | [Quantile card](../reference/model-cards/quantile.md#quantile_lp-quantile-local-projections) |
| `favar` + `var_irf_bands` — two-step bands | small noisy panel (N=20), T=800, worst horizon | **0.673 ± 0.010** | ESTIMATOR | `favar` ships no band; this is the guide's own construction and its warned generated-regressor hazard, priced against a true-factor oracle on the same draws. On a rich clean panel the two are within 0.3pp; on the small noisy one the F̂ band loses a further 6–14pp at long horizons — and more T makes it *worse* (0.747 → 0.673 at h=7) while the oracle improves, because the band shrinks around an O(1/N) factor-measurement distortion. Bootstrap the two-step procedure, or grow N before T | [Multivariate guide](../guide/07-multivariate.md) |
| `umidas` — the intercept | AR(1) errors φ=0.7, persistent HF regressor, T=150 | **0.829 ± 0.007** | APPROXIMATION | the HF-lag coefficients hold ~0.92+ even here; the intercept inherits the error's full serial correlation (se/sd = 0.71) — the har_rv-constant mechanism on a new surface. Quote the constant with care or lengthen `maxlags` | [Nowcasting/MIDAS card](../reference/model-cards/nowcasting-midas.md#how-to-read-the-output) |
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
whole path 95% of the time has to be materially wider.

### The remedy, and the two places it stops

*Added after `0.2.0`. The finding above is left exactly as it was measured — it
is still what a pointwise band delivers, and it is still the default.*

When this page was written, `tsecon` reported no simultaneous band for any
object, and that was [the largest open recommendation](#what-this-audit-recommends)
on it. A **simultaneous (sup-t) band** now exists: `var_irf_bands`,
`var_forecast`, `lp` and `smooth_lp` take `band="sup-t"` (or `"sidak"` /
`"bonferroni"`), and the two VAR functions take a `band_scope` as well. It
changes exactly one thing: the multiplier. Same point estimate, same standard
errors, and a constant `c` in `point ± c·se` chosen so that **every cell of a
declared family** is covered at once — the construction of Montiel Olea and
Plagborg-Møller. Scored against a pointwise arm **on the same replications**:

| object | nominal | design | pointwise, joint | **sup-t, joint** | Šidák | Bonferroni | measured by |
|---|---|---|---|---|---|---|---|
| `var_forecast` | 95% | T=100, 12 horizons × 2 series, K=24, 2000 reps | 42.0% ± 1.1 | **90.5% ± 0.7** | 91.3% ± 0.6 | 91.5% ± 0.6 | [`forecast_intervals.py`](coverage/forecast_intervals.py) |
| `var_irf_bands`, asymptotic | 90% | T=500, h=0..12, K=13, 1000 reps | 71.7% ± 1.4 | **85.2% ± 1.1** | 91.7% ± 0.9 | 92.0% ± 0.9 | [`irf_bands.py`](coverage/irf_bands.py) |
| `lp`, lag-augmented | 90% | T=240, 13 horizons, 400 reps | 36.5% | **81.8%** | — | — | the crate's own tests |
| `lp`, lag-augmented | 90% | T=720, 13 horizons, 400 reps | 42.7% | **89.5%** | — | — | the crate's own tests |

The two VAR rows are measured by this page's own modules, and both arms are read
off the **same call**: asking for a simultaneous band leaves `point`, `se`,
`lower` and `upper` bit-identical — those modules assert it — so the comparison
is paired rather than two independent runs, and every replication that gains the
path gains it because the multiplier grew. Their pointwise rates reproduce this
page's 0.409 and 0.722 on fresh seeds, and their sup-t rates reproduce what the
crates' own tests measure at higher replication counts (41.2% → 90.5% at 6000
reps; 70.4% → 84.8% at 3000). The LP rows are the crate tests' — `lp` has no
simultaneous-band arm in the seven modules below (its cumulative-mode
*pointwise* intervals are measured in `quantile_panel_lp.py`).

**Neither VAR simultaneous rate reaches nominal — 85.2% against 90%, 90.5%
against 95% — and that is the honest headline, not a footnote.** A sup-t band
fixes **multiplicity** and inherits every other defect on this page, because it
reuses the same standard errors:

- the IRF band's own **marginal** coverage on the same 1000 draws is 91.0% at
  h=0 and **85.3%** at h=12, so the residual belongs to
  [the delta-method problem](#the-delta-method-irf-band-horizon-by-horizon).
  What that cell needs is a better standard error, not a bigger multiplier;
- `var_forecast` is a **plug-in** band that ignores coefficient sampling error,
  so its pooled per-cell marginal rate is **93.4%**, not 95% — see
  [the oracle column](#predictive-intervals). A simultaneous band cannot hand
  back coverage the marginal band never had;
- **LP is the clean case, and it is the one that proves the mechanism.** At
  T=720, where the marginal rates sit on nominal, sup-t lands on nominal
  (89.5% against 90%). At T=240 it does not (81.8%) — for the same reason the
  marginals are off there.

The LP rows also settle what kind of problem this is. **Tripling T moved the
pointwise joint rate from 36.5% to 42.7%.** It is not converging: multiplicity
is not a small-sample caveat that data cures.

**Simultaneous over *what*? — this is a user-visible choice, not a detail.**
Every cell added to the family widens the band for every other cell in it, so
the same `alpha` over a different family is a different band, and a band whose
scope is ambiguous is worse than no band. The scopes are:

| object | scope | the family | K on the designs above (k=2 series; IRF h=0..12, forecast 12 steps) |
|---|---|---|---|
| `var_irf_bands` | `"horizon"` (default) | one response-shock pair's path — the object measured above | 13 |
| `var_irf_bands` | `"shock"` | every response to one shock, all horizons | 26 |
| `var_irf_bands` | `"all"` | the whole IRF grid | 52 |
| `var_forecast` | `"all"` (default) | every horizon of every series — the object measured above | 24 |
| `var_forecast` | `"horizon"` | one series' whole forecast path | 12 |
| `lp` / `smooth_lp` | — | the horizons of the one response | 13 |

Every result reports its scope and its `K`. Report them too.

**Which multiplier, and what the fallbacks cost.** The closed forms depend on
nothing but `K`: at K=13 and α=0.10 they are Šidák **2.6490** and Bonferroni
**2.6653**, against a pointwise **1.6449**. The sup-t multiplier is a property
of the *path*, because it is the only route that uses the dependence across
cells — adjacent horizons of an impulse response are strongly positively
correlated, while Šidák (exact under independence) and Bonferroni (valid under
any dependence) pay for a worst case a smooth response path does not present. On
this page's `BASE` VAR(1) it averages **2.0742** over 1000 replications: 22%
below Šidák, and only 26% above the pointwise `z`. On more persistent paths the
crates' tests see it run up to about 2.65 — no saving at all. **Read the critical
value your own fit reports; do not assume the saving.**

And read the Šidák and Bonferroni columns above before treating them as safe
fallbacks. On the IRF row they land at **91.7%** and **92.0%** against a nominal
90% — *over* the line, because over-widening happened to cancel a marginal
shortfall of an entirely different origin. That is a coincidence on one DGP, not
a calibration.

**Three things this does not do.** `lp_iv`, `lp_multiplier` and `lp_state` have
no cross-horizon covariance, so they get Šidák and Bonferroni **only** — sup-t
is refused with an error naming the reason, and their bands must never be
described as sup-t. The bootstrap simultaneous band is a **different shape**
from the bootstrap percentile band — symmetric `point ± c·se` against asymmetric
Efron percentiles — so it is not guaranteed to sit outside the percentile band
cell by cell, only outside the symmetric `point ± z·se`. And nothing that
already reads these functions changed meaning: `var_irf_bands` and `var_forecast` return
**pointwise** `lower`/`upper` whatever you pass — the simultaneous edges arrive
as extra `sim_lower`/`sim_upper` keys — and `lp` returns no band at all unless
asked. Every number in the tables above is a pointwise band and stays one, so
**a fan chart you did not ask a simultaneous band for is still read one horizon
at a time**.

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
               lam=cv    0    1.0000   -0.07812    0.06334    0.05076    0.80     1.23    0.646   0.018
              lam=100    0    1.0000   -0.03906    0.04482     0.0451    1.01     0.87    0.861   0.013
```

`|b|/sd = 1.23` at impact: the shrinkage bias exceeds a whole sampling standard
deviation, so the interval is centred in the wrong place and no standard error
saves it. The `lam=100` arm isolates the two effects — se/sd is 1.01 there (the
SE is correctly sized) while `|b|/sd` is already 0.87 (pure shrinkage). The
library's own model card says `se` "conditions on `lam` and does not account for
shrinkage bias"; **0.646 against 0.95 is the size of what that sentence is
hiding.** Read a smooth-LP band as a band around the penalized estimand, and
read the `lam=cv` column against the `lam=0` column rather than against 0.95.
(This cell moved 0.640 → 0.646 — a third of its own MC standard error — when
0.3.0 made the default CV λ grid scale-relative; every other pre-existing cell
on this page reproduces the earlier run byte-for-byte.)

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

### Quantile, panel and cumulative local projections

The three LP surfaces the original audit did not measure. DGPs: a
location-scale MA whose truncation at J = p + 1 makes the conditional
quantile *exactly* linear in `quantile_lp`'s design (so the per-tau truths
are closed-form: impact slope `θ₀ + 0.4·z_τ`, later slopes `θ_h`); a dynamic
panel with a common shock, a common factor, and truth `0.8·0.8^h`; and the
`lp_family` house MA for the cumulative modes. Nominal **95%** throughout;
1000–2500 replications by experiment.

**`quantile_lp` is the audit's cleanest good news, and the reason is
structural.** Its own model card warns that these Powell-sandwich standard
errors are "not HAC" and that `growth_at_risk`'s measured overlap
under-coverage (0.72 at h=8, τ=0.5, T=200) "is the right order of magnitude
to expect here too". Measured, that transfer does **not** bind on the
canonical design: with an identified i.i.d. shock the check-loss score is
serially uncorrelated at every horizon — the overlapping windows correlate
the ψ factors, but each is multiplied by an independent shock draw, the same
mechanism that makes lag-augmented inference work for mean LP. Coverage sits
at 0.89–0.97 across every (τ, h, T) cell, worst at the location-scale impact
tail (0.893 ± 0.010 at τ=0.25, T=200 — a density-estimation cost, not an
overlap one). The hazard is real where nothing whitens the regressor:

```text
  persistent regressor (phi_s = 0.8), tau = 0.50, T = 200, nominal 0.95
                        h=0     h=2     h=4     h=6
  p=4 (the default)   0.946   0.933   0.946   0.931     se/sd 1.00-1.01
  p=0 (no controls)   0.899   0.881   0.859   0.785     se/sd 0.88 -> 0.67
  paired difference, p=4 minus p=0, pooled h >= 3: +0.106 (se 0.008)
```

The default lag controls include the regressor's own lags, so the
residualised impulse is (nearly) the AR innovation and the sandwich survives;
strip them and the growth_at_risk-shaped decay appears in full, driven by the
SE (se/sd falls to 0.66 while |bias|/sd stays near 0.4). Keep
`n_lag_controls` at least the regressor's AR order.

**`panel_lp`'s Driscoll-Kraay default: the effective sample is T, and now
that is a number.** With a common shock and a common factor
(`y_it = α_i + 0.8·y_{i,t−1} + 0.8·s_t + 0.9·f_t + e_it`):

```text
  nominal 0.95, Driscoll-Kraay (the default), h = 0 / 2 / 4
  N=10 T=40   0.875   0.841   0.824        N=10 T=80   0.904   0.884   0.887
  N=50 T=40   0.889   0.839   0.818        N=50 T=80   0.916   0.887   0.894
```

Quintupling N moves pooled coverage by under a point; doubling T buys ~5pp.
The Nickell bias is visible and behaves exactly as Nickell says (−0.071 at
h=4, T=40 → −0.043 at T=80, untouched by N). And the cookbook's "wrong
reflex" is priced: `se_type="cluster"` on the *same draws* covers **0.20**
(se/sd = 0.12) — clustering by entity assumes independence across entities in
the presence of a common factor, and the paired coverage difference to
Driscoll-Kraay is +0.70 (se 0.004). The dramatic size is this DGP's strong
factor loading; the sign is general.

**The split-panel jackknife, re-measured on the panel card's own design at
~8× its replications** (γ_f = 0, N=50, bandwidth=2; the card's 300-rep table
claims FE 0.743 → SPJ 0.823 at T=20, h=2):

```text
  T=20  h=2   bias  fe -0.141 -> spj +0.015     cov95  fe 0.713 -> spj 0.761
  T=40  h=2   bias  fe -0.059 -> spj +0.011     cov95  fe 0.846 -> spj 0.843
  paired coverage difference, spj minus fe: T=20 +0.025 (se 0.006),
                                            T=40 -0.015 (se 0.005)
```

The bias claim is corroborated almost exactly (the card's −0.137 → +0.009).
The *coverage* gain at T=20 is real but smaller than the card's
point numbers suggest — +2.5pp paired, not +8pp — and at T=40 SPJ covers
very slightly *worse* than FE, consistent with the card's own T=40 cells.
Neither route approaches nominal at T=20; Driscoll-Kraay is itself a short-T
approximation, exactly as the card warns. Read that table with its
mcse (~0.025) in mind.

**`lp(cumulative="both")`: the official post-fix numbers.** The audit's most
serious open finding was that this mode's old lag-augmented/HC1 default
covered **0.507** at h=12, flat in T; 0.3.0 made `se=None` resolve to
`"hac"` here and made an explicit `se="lag_augmented"` raise (both asserted
every run):

```text
  cumulative="both", default se -> "hac", nominal 0.95
            h=0     h=4     h=8    h=12
  T=400   0.943   0.960   0.950   0.920      se/sd 0.94-1.05
  T=1600  0.936   0.971   0.965   0.956      se/sd 0.97-1.10
```

h=12 goes 0.507 → **0.920 ± 0.006** at T=400 and 0.956 at T=1600 — the
repair, measured on the audit's DGP (the fix's own probe reported 0.920 at
the matching cell). The residual deviation flips mildly *conservative* at
T=1600 mid-horizons (0.971 at h=4): the Bartlett bandwidth `h + p` is
generous once T is large relative to the MA(h) overlap. The
cumulated-outcome mode (`cumulative=True`) keeps its lag-augmented default
and stays at 0.93–0.96 everywhere, as the LP card states.

Full report:
[`docs/examples/coverage/quantile_panel_lp.py`](coverage/quantile_panel_lp.py).

### Factor models and mixed frequency

Split by an honest question asked first: does the function ship an interval
at all? `favar`, `weighted_midas`, `dfm_nowcast` and `nelson_siegel` do not —
verified against their live key sets on every run, so a future `se` key
breaks an assertion rather than silently outdating this page. `umidas` ships
HAC `bse`, and `favar`'s *guide* documents how bands get built in practice
(`var_irf_bands` on `[F̂, policy]`) while warning that "bands that condition
on F̂ as if it were data are too narrow". Both measurable claims are
measured.

**The FAVAR generated-regressor hazard, priced.** The guide's own
transmission DGP (two latent factors + a policy rate forming a VAR(1) with
diagonal Σ); the measured cell is the policy rate's response to the
recursive policy shock — the one FAVAR band cell invariant to factor
rotation, so its truth is closed-form. Each replication fits the two-step
band and, on the same draw, the infeasible band on the *true* factors:

```text
  nominal 0.90, h = 0 / 4 / 8, F-hat band vs oracle band (paired)
  N=100 T=200 (rich, clean)   F-hat 0.887 0.903 0.816   oracle 0.889 0.898 0.824
  N=20  T=200 (small, noisy)  F-hat 0.902 0.920 0.733   oracle 0.894 0.895 0.833
  N=20  T=800                 F-hat 0.897 0.877 0.728   oracle 0.893 0.905 0.892
  paired F-hat minus oracle, pooled h >= 4:
    N=100 T=200: -0.003 (se 0.001)   N=20 T=200: -0.058 (se 0.004)   N=20 T=800: -0.138 (se 0.006)
```

On the rich panel the two are within 0.3pp — Bai-Ng negligibility
(√T/N → 0) has effectively arrived, and what remains is the delta-method
horizon decay this page already documents. On the small noisy panel the
F̂ band loses a further 6–14pp at long horizons, and the T-growth column is
the diagnosis: from T=200 to T=800 the *oracle* improves (0.833 → 0.892 at
h=8) while the F̂ band gets **worse** (0.733 → 0.728, and 0.747 → 0.673 at
h=7, with |bias|/sd reaching 1.03) — the band shrinks like 1/√T around a
factor-measurement distortion that is O(1/N) and does not shrink in T at
all. No standard error that conditions on F̂ can see it; bootstrap the
two-step procedure, or grow N before T.

**`umidas`'s HAC intervals: the slopes hold, the intercept is the casualty.**
On a known-weights mixed-frequency DGP (truth `β·w_k` per lag, exactly):

```text
  nominal 0.95            intercept   worst HF-lag coefficient
  iid errors, T=300           0.943        0.940
  AR(1) errors phi_u=0.7,
  HF phi_h=0.9, T=150         0.829        0.918   (intercept se/sd 0.71)
```

The lag coefficients stay ~0.92+ even under persistent errors and
near-collinear lag columns; the intercept inherits the error's full serial
correlation — the same constant-under-persistence mechanism this page
documents for `har_rv`'s constant. Quote the constant with care or lengthen
`maxlags`.

Full report:
[`docs/examples/coverage/factor_midas.py`](coverage/factor_midas.py).

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
| `quantile_panel_lp` | a location-scale MA `y_t = Σ_j 0.7^j s_{t−j} + (1 + 0.4 s_t) e_t` truncated at J = p + 1 = 5 so the conditional quantile is *exactly* linear in the design, s ~ U(−1.5, 1.5) iid, T ∈ {200, 400}, τ ∈ {0.25, 0.5, 0.75}; the same MA driven by a Gaussian AR(1) regressor at φ_s=0.8 (pure location, closed-form slopes), fitted with p ∈ {4, 0}; a dynamic panel `y_it = α_i + 0.8 y_i,t−1 + 0.8 s_t + 0.9 f_t + e_it` with a common shock and a common factor, N ∈ {10, 50}, T ∈ {40, 80}, plus the panel card's own SPJ design (γ_f = 0, N=50, T ∈ {20, 40}); the lp_family house MA for `cumulative` at T ∈ {400, 1600} | 95% | 1000–2500 by experiment |
| `factor_midas` | the guide's FAVAR transmission DGP — 2 latent factors + a policy rate forming a VAR(1) with diagonal Σ, panel `X = ΛF + idio` with (N, idio sd, T) ∈ {(100, 0.5, 200), (20, 1.0, 200), (20, 1.0, 800)}, Λ redrawn per replication; a mixed-frequency DGP `y = 0.5 + 1.5 Σ_k w_k hf[t,k] + u` with exp-Almon weights, K=12, m=3, HF AR(1) φ_h ∈ {0.5, 0.9}, u iid or AR(1) φ_u=0.7, T ∈ {300, 150} | 90% (favar bands) / 95% (umidas) | 2000–3000 |

---

## What is *not* measured

Stated so you do not have to discover it.

- ~~**No simultaneous (sup-t) band exists to measure.**~~ **One exists now, and
  it is measured — but only for two of the four surfaces that offer it.**
  `irf_bands.py` and `forecast_intervals.py` each carry a simultaneous arm, and
  those numbers are in [pointwise is not joint](#pointwise-is-not-joint). They
  are **not** in Table 1 or Table 2: every row of those tables is a pointwise or
  marginal band, which is still what a caller gets by default and still what a
  reader gets by mistake. `lp` and `smooth_lp` also take a band selector and
  have **no arm in `lp_family.py`** — the LP rows quoted are the crate's own
  tests. Both VAR arms run at a reduced `band_n_sim` (20000 against the library
  default of 100000) to keep the harness runtime modest; that puts simulation
  noise into the multiplier, not bias into the coverage it delivers.
- **No weak-instrument-robust set exists to measure *for these two rows*.**
  Anderson-Rubin is the right interval for the `iv_gmm` and `lp_iv`
  weak-instrument rows, and neither function exposes one. The library does now
  ship `proxy_ar_sets` — an Anderson-Rubin set for proxy-SVAR impulse
  responses — so the machinery exists and the gap is that it has not been
  extended to the IV regression estimators, not that it is unbuilt.
- ~~**`lp(cumulative=...)` intervals are unmeasured.**~~ **Measured now** —
  [`quantile_panel_lp.py`](coverage/quantile_panel_lp.py) publishes the
  official post-fix numbers for the `cumulative="both"` HAC default (the
  audit's 0.507-at-h=12 defect) and re-measures the cumulated-outcome mode's
  lag-augmented default alongside.
- **`growth_at_risk`, `proxy_svar`, `nongaussian_svar`, GARCH forecast
  intervals, and `flp`/`flp_scenario` are not in this audit.** They have
  golden and property coverage; they do not yet have measured interval
  coverage *in this registry*. Two of them carry measured coverage
  elsewhere: `growth_at_risk`'s Powell → Newey-West coverage table lives on
  [its model card](../reference/model-cards/quantile.md#standard-errors-at-long-horizons-measured),
  and `proxy_ar_sets`' propagated coverage (declining through the default
  h=12) is on the
  [identification card](../reference/model-cards/structural-identification.md).
  For `flp` the gap is known to bite: its per-element `se` conditions on
  `functional_pca`'s estimated eigenfunctions and measures se/sd ≈ 0.66 on
  FPCA scores, flat in `T` (coverage ≈ 0.80 against a nominal 0.95) — the
  [functional-shocks model card](../reference/model-cards/functional-shocks.md#flp-functional-local-projections)
  carries the warning and the numbers; `flp_scenario`'s `w'beta` contrasts are
  algebraically immune to that term and measured at the ordinary LP-HAC level.
- **The `bvar_*` family is measured here as frequentist diagnostics only**
  (group C); its *Bayesian calibration* — draw from the model's own prior and
  check the credible sets — was measured by audit round 6, which found the
  ML-II `hyperprior="none"` collapse and the GLP plug-in's 0.82–0.85, and is
  recorded in `docs/roadmap/20-audit-round-6-findings.md` and on the
  [Bayesian card](../reference/model-cards/bayesian.md).
- **The term-structure fits ship no interval at all.** `nelson_siegel`'s key
  set is verified every run by
  [`factor_midas.py`](coverage/factor_midas.py) (its row is in the tables);
  `svensson` and `dynamic_ns` return the same interval-free shape (factors,
  residuals, R², forecasts) but are not under a per-run tripwire.
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
3. ~~**Expose a simultaneous band.**~~ **Done, after `0.2.0` — and it does not
   reach nominal on either VAR arm.** `var_irf_bands`, `var_forecast`, `lp` and
   `smooth_lp` take a band selector with four routes (sup-t, Šidák, Bonferroni,
   and the unchanged pointwise default), each reporting the cell family it is
   simultaneous over. Scored against a pointwise arm on the same replications:
   the forecast band goes **42.0% → 90.5%** at nominal 95%, the IRF band
   **71.7% → 85.2%** at nominal 90%, and LP **42.7% → 89.5%** at T=720. So this
   closed the *reading* gap and left two *coverage* gaps standing, both
   inherited: the IRF band's marginal coverage is itself 85.3% at h=12, and
   `var_forecast`'s pooled per-cell marginal rate is 93.4%. `lp_iv`,
   `lp_multiplier` and `lp_state` get the closed forms only.
   [The full comparison](#the-remedy-and-the-two-places-it-stops).
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
2. On a persistent VAR, pass `method="bootstrap", bias_correct=True` to
   `var_irf_bands` — the method matters, and passing `bias_correct` without it
   now raises. 0.410 →
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
9. Read a fan chart one horizon at a time — or ask for the simultaneous band
   and say which family it is simultaneous over. The pointwise default is
   unchanged, so nothing you already plotted became a joint statement.

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
- The seven modules themselves, each with a derivation of its truth in its
  docstring:
  [`regression_se.py`](coverage/regression_se.py),
  [`irf_bands.py`](coverage/irf_bands.py),
  [`lp_family.py`](coverage/lp_family.py),
  [`forecast_intervals.py`](coverage/forecast_intervals.py),
  [`bayes_and_sets.py`](coverage/bayes_and_sets.py),
  [`quantile_panel_lp.py`](coverage/quantile_panel_lp.py),
  [`factor_midas.py`](coverage/factor_midas.py),
  the runner [`run_all.py`](coverage/run_all.py), and the page-sync guard
  [`check_page.py`](coverage/check_page.py).
