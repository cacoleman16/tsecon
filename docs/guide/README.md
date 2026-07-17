# The tsecon Guide to Time Series Econometrics

**A free, full-length guide to time series — from your first autocorrelation
plot to research-grade structural identification — where every concept comes
paired with code that runs.**

The guide mirrors the structure of **tsecon**, a high-performance time series
econometrics library (Rust core, Python API) built in this repository. Each
chapter teaches the ideas on their own merits — the guide stands alone as a
course in time series econometrics — and then shows the exact library calls
that put them to work. Where a method is still on the library's
[roadmap](../../ROADMAP.md), the code is clearly labeled a preview and linked
to its module specification.

Every chapter follows the same ladder: **The idea** (plain-English intuition,
no equations) → the methods (why you care, the math, runnable code, the
classic mistakes) → **The frontier** (where research stands today) → **Which
method when** (a decision table) → further reading (the founding papers).

---

## The chapters

| # | Chapter | One line |
|---|---------|----------|
| 1 | [Thinking in Time Series](01-foundations.md) | What makes time-ordered data different, stationarity, random walks, and the transformations that tame them |
| 2 | [Exploring and Diagnosing a Series](02-exploration-and-diagnostics.md) | Reading ACF/PACF, testing for white noise and unit roots, and the confirmatory stationarity workflow |
| 3 | [Honest Inference with Dependent Data](03-inference-toolkit.md) | Why dependence breaks textbook standard errors — HAC, EWC, and the bootstrap family that fixes them |
| 4 | [Univariate Models: AR to State Space](04-univariate-models.md) | The model ladder: AR → ARMA → ARIMA → ETS → the Kalman filter → regime switching |
| 5 | [Forecasting: Practice and Evaluation](05-forecasting.md) | Backtesting discipline, benchmarks that are hard to beat, accuracy measures, and formal comparison tests |
| 6 | [Volatility: GARCH and Risk](06-volatility.md) | Why volatility clusters, the GARCH family, VaR/ES, and realized volatility |
| 7 | [Systems: VAR, Cointegration, and Factors](07-multivariate.md) | Modeling many series at once: VARs, impulse responses, common trends, and common factors |
| 8 | [Structural Identification](08-causal-identification.md) | From correlation to cause: Cholesky, long-run and sign restrictions, narrative shocks, and external instruments |
| 9 | [Local Projections](09-local-projections.md) | The modern impulse-response workhorse, its inference pitfalls, and LP-IV multipliers |
| 10 | [Bayesian Time Series](10-bayesian.md) | Priors as shrinkage, the Minnesota BVAR, samplers you can trust, and posterior impulse responses |
| 11 | [Nowcasting and Mixed Frequencies](11-nowcasting.md) | Reading the economy in real time: ragged edges, MIDAS, factor-model nowcasts, and news decomposition |
| 12 | [Machine Learning for Time Series](12-machine-learning.md) | Leakage-safe validation, shrinkage vs sparsity, trees and boosting, and an honest look at foundation models |
| 13 | [Nonlinear Dynamics: Regimes, Thresholds, and State-Dependent Responses](13-nonlinear-dynamics.md) | When linearity fails: threshold and Markov-switching systems, generalized impulse responses, and state-dependent local projections |
| 14 | [Panel Time Series](14-panel-time-series.md) | Heterogeneous panels: fixed effects, the mean-group and common-correlated-effects estimators, and panel local projections and VARs |
| 15 | [The Term Structure of Interest Rates](15-term-structure.md) | Fitting and forecasting the yield curve: Nelson-Siegel, Svensson, and the dynamic Nelson-Siegel |

Worked, figure-rich examples for many of these methods live in the
[gallery](../examples/README.md); the library's full technical plans live in
the [module specifications](../roadmap/).

---

## Learning paths

You don't have to read linearly. Four curated routes:

**The beginner path** — never touched time series before:
1 → 2 → 4 → 5. You'll finish able to diagnose a series, fit and select a
univariate model, and evaluate forecasts honestly. Add 3 when a p-value
matters to you.

**The forecaster's path** — you ship predictions:
1 → 2 → 4 → 5 → 12, with 6 if your target's *uncertainty* matters (finance,
risk) and 11 if your data arrive at mixed frequencies.

**The macro-structural path** — you ask "what does a shock do?":
1 → 2 → 3 → 7 → 8 → 9 → 10 → 13. This is the empirical-macro toolkit: from
VARs through identification to local projections and Bayesian estimation,
the sequence most PhD courses spread across two semesters — capped by 13,
where the linearity assumption everything else shares is finally relaxed.

**The risk path** — volatility and tails are your job:
1 → 2 → 3 → 6, then 5 for evaluating VaR forecasts like any other forecast.

---

## How code appears in the guide

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal(300))      # a random walk
tsecon.check_stationarity(y)["recommendation"]   # -> "Difference"
```

Blocks like this run today against the library. Blocks introduced with
"*Roadmap preview*" show the intended API for methods still being built —
each links to the module spec that defines it. Everything the library
computes is validated against reference implementations (statsmodels, SciPy,
NumPy, `arch`) down to tight numerical tolerances; the same discipline keeps
this guide's claims honest.

## Contributing and errata

The guide is versioned with the library. Corrections and clarity
improvements are welcome — a guide that teaches judgment has to earn trust
the same way the library does: by being checked.
