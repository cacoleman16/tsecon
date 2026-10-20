# Cookbook

Short, single-task pages. One question, one runnable answer, under thirty lines
of code, with the real output printed underneath. If you know what you want to
*do* and just need the call that does it, start here.

Every block on every page was executed against the installed library and the
output pasted verbatim. Each recipe stands alone — the data are simulated in
the snippet with a fixed seed, so you can paste any page into a fresh
interpreter with nothing but `pip install tsecon` and reproduce the numbers
exactly.

Not sure which model your problem calls for? Start at
[Which model when?](../which-model-when.md). Want the ideas rather than the
call? That is [the Guide](../guide/README.md). Want the assumptions, defaults,
and failure modes of a specific estimator? That is a
[model card](../reference/model-cards/diagnostics.md).

---

## Diagnostics — before you model anything

| Recipe | What it does |
|---|---|
| [Screen a new series before you model it](screen-a-series.md) | Run `check_series`, read the report, and act on its ordered recommendations |
| [Test whether a series has a unit root](unit-root-test.md) | The ADF + KPSS confirmatory quadrant via `check_stationarity`, and when to reach for `phillips_perron` |
| [Find structural breaks when you don't know the dates](structural-breaks.md) | `sup_f_test` for one break, `bai_perron` for several, with break-date confidence intervals |

## Inference — standard errors that survive review

| Recipe | What it does |
|---|---|
| [Get HAC (Newey-West) standard errors on a regression](hac-standard-errors.md) | The `se_type` ladder on `ols`, and why `hc1` does nothing for time series |
| [Driscoll-Kraay and clustered SEs for a panel local projection](panel-lp-standard-errors.md) | Why clustering by entity *shrinks* your SEs under a common shock, and what fixes it |

## Systems — VARs, forecasts, and identification

| Recipe | What it does |
|---|---|
| [Choose a VAR lag length when AIC, BIC, and HQIC disagree](var-lag-selection.md) | Compare lag orders on one effective sample, then use the analysis goal to break a disagreement |
| [Put confidence bands on a VAR impulse response](var-irf-bands.md) | `var_irf_bands` with bootstrap or asymptotic bands, sliced like `var_irf` |
| [Forecast a VAR with prediction intervals](var-forecast-intervals.md) | `var_forecast` with a stability check and a coverage knob |
| [Identify a monetary policy shock with sign restrictions](sign-restricted-svar.md) | A set-identified SVAR in a dozen lines, and how to read the acceptance rate |

## Causal effects

| Recipe | What it does |
|---|---|
| [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md) | `lp_multiplier` (Ramey-Zubairy), the first-stage F, and the ratio-of-peaks trap it avoids |

## Risk and volatility

| Recipe | What it does |
|---|---|
| [Model fat-tailed volatility with GARCH and forecast it](garch-fat-tails.md) | `garch_fit(dist="t")`, robust SEs, a variance path, and the post-fit residual check |
| [Compute growth-at-risk](growth-at-risk.md) | Quantiles of future GDP growth conditional on financial conditions |

## Reporting

| Recipe | What it does |
|---|---|
| [Export a results table to LaTeX or Markdown](results-table-export.md) | `tsecon.summarize` for the console, pandas for the paper |

---

## Conventions in every recipe

- **Arrays in, dicts out.** Every estimator takes plain arrays (or anything
  array-like — pandas Series and DataFrames are coerced) and returns a `dict`
  of documented keys. There is no fit/predict object to construct.
- **You supply the constant.** Functions taking a design matrix (`ols`,
  `sup_f_test`, `bai_perron`, `recession_probit`) use `x` exactly as given.
- **`alpha` is the non-coverage.** `alpha=0.10` is a 90% band, `alpha=0.05` a
  95% one.
- **Seeds are arguments, not globals.** Anything stochastic — bootstraps,
  rotation draws, posterior sampling — takes a `seed`. Report it.
- **No data loaders, no network calls.** `import tsecon` depends only on NumPy;
  you bring the data.

Missing a recipe you expected to find? Open an issue — the list is demand-driven.
