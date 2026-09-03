# Adversarial audit, round 11 — findings

> **Working document.** Continuation of
> [round 10](25-audit-round-10-findings.md), run under
> [the brief](16-adversarial-audit-brief.md). Excluded from the published
> site.

Round 11 ran four class sweeps no previous round had attempted, over the
**existing** 0.7.0 surface (162 public callables at `136160f`; the
machine-learning wave being built in parallel was out of scope). Every
sweep drove one shared registry of seeded, tiny canonical inputs
(`lab/audit/round11/registry.py`, one entry per callable, 162/162 reached),
so a finding in one sweep could be re-checked in another on the identical
call. The probe scripts and their logs are committed under
`lab/audit/round11/` — the round-9 tails died with their container; these
will not.

**Design.** Four finder/refuter sweeps, each candidate attacked (re-run,
second seed or size, the promise re-read on the surface that binds —
runtime `__doc__` first, then the stub and the card) before it could be
CONFIRMED:

- **E — result-object contract**: for every callable, `summarize()`
  renders; `json.dumps`/`pickle` round-trip the dict with arrays; every
  float is finite or its NaN/inf is documented; returned keys vs the keys
  the docstring, stub and card name (both directions); array shapes vs
  the shapes the docstrings state.
- **F — signature / stub / docstring drift**: `inspect.signature` of the
  compiled function vs the stub (names, order, has-default); defaults the
  prose states vs the runtime default; kwargs used in call snippets across
  docstrings, cards and `api.md` that the function does not accept; every
  string value a docstring lists as accepted, actually passed.
- **G — complexity cliffs**: every callable timed at T ∈ {200, 800, 3200}
  in a fresh subprocess, log-log slope fitted, flagged cells re-timed to
  refute noise; a second pass with **default arguments only** at T=3200.
- **H — seed contract**: for the 21 seed-taking parameters, same seed
  twice in-process, same seed across a process restart, different seed,
  and `seed=None`; plus every callable called twice (determinism) and the
  three parallel bootstrap tests at 1 vs 4 threads.

**Totals: 133 candidates raised across the four sweeps (plus 3 harness
bugs, not counted), 67 refuted, 14 findings confirmed (0 severe, 6 moderate, 8 low) — all 14 fixed in-branch
with regression tests — and 8 recorded OPEN below with their refutation
status.** The clean bills are the headline: no panic, no non-catchable
exception, no non-determinism, no seed that failed to reproduce across a
restart, no `summarize`/JSON/pickle failure, and no signature/stub drift
on any of the 162 callables. Every confirmed finding is a documentation-
contract defect: a surface a user reads (chiefly `help()`) saying less, or
something different, than what the function does.

| sweep | raised | refuted | confirmed (findings) | fixed | open |
|---|---|---|---|---|---|
| E — result contract | 83 | 34 | 47 items → 10 findings | 10 | 2 |
| F — signature/doc drift | 24 | 16 | 3 | 3 | 1 |
| G — complexity cliffs | 23 | 17 | 0 | 0 | 4 (measured, no promise violated) |
| H — seed contract | 3 | 0 | 1 | 1 | 0 |
| **total** | **133** (+3 harness bugs) | **67** | **14** | **14** | **8** |

---

## Severe

None. Sweep H is the one that could have produced a severe finding (an
accepted-and-ignored seed makes an unreproducible result look
reproducible, per the brief) and it came back clean on every axis:
21/21 seed parameters bit-identical for the same seed in-process **and**
across a subprocess restart, 21/21 different for a different seed on the
configuration where the seed is live, 162/162 callables bit-identical on
two consecutive calls, and `hansen_seo_test`/`setar_test`/
`threshold_var_test` bit-identical at `RAYON_NUM_THREADS=1` vs `4` as
their docstrings promise.

## Moderate

**M1 (E1). Five structural-identification functions' `help()` text was one
or two sentences while the stub carried the full return contract** —
`robust_svar_bounds` (runtime 120 chars vs stub 1189), `fry_pagan_svar`
(144 vs 1152), `hetero_svar` (347 vs 1383), `historical_decomposition`
(117 vs 1335), `narrative_svar` (156 vs 764). The brief names runtime
`__doc__` as the binding surface; on it, none of `robust_svar_bounds`'s
ten keys were named and its NaN convention ("unrestricted shocks are NaN
in every array" — sweep E(iii) fired *undocumented non-finite* on the
canonical call, 42 NaN cells per (7, 3, 3) array) existed only in the
stub. Root cause: the `///` doc comments in `lib.rs` were written as
one-line summaries for these functions and the stub docstrings were
extended later without a sync. **Fix**: the runtime doc comments now carry
the stub's contract verbatim (keeping the runtime's citation line); a
parametrized regression test asserts every returned key is backticked in
`fn.__doc__` for all 29 functions of this class.

**M2 (E2). `gpd_fit` / `gev_fit` `help()` lacked the stub's `Keys:` line**
— 7 and 6 returned keys unnamed at the binding surface (`beta`,
`converged`, `exceed_rate`, `loglik`, `se_beta`, `se_xi`,
`threshold_quantile`; `converged`, `loglik`, `n_maxima`, `se_mu`,
`se_sigma`, `se_xi`). Same root cause and fix as M1.

**M3 (F1). EGARCH multi-step variance forecasts are refused, and no surface
said so — and the refusal leaked an internal marker.** Sweep F(d) passed
every string value the `ccc_garch`/`dcc_garch` docstrings list for `vol`
and found `vol="egarch"` raising at the canonical `forecast_horizon=2`
with `EGARCH multi-step forecasts require simulation (TODO(phase0)); only
horizon = 1 is analytic`. `garch_fit` itself raises the same way at
`forecast_horizon >= 2` (`horizon=1` works). The stub for `ccc_garch`
promised "with forecast_horizon > 0 covariance_forecast ((horizon, k, k))"
unconditionally; `garch_fit`, `dcc_garch` and the volatility card were
silent. **Fix**: the crate message (`crates/tsecon-garch/src/results.rs`)
now states the limit and both remedies without the marker; the three
docstrings, the stub and the volatility card state that `vol="egarch"`
accepts `forecast_horizon` 0 or 1 only; a test pins horizon 1 working,
horizon 2 raising a `ValueError` naming `forecast_horizon=1`, and the
absence of `TODO` in the message.

**M4 (H1). `seed=None` is accepted by `conformal_forecast` and
`conformal_backtest` (`seed`) and by `proxy_ar_sets` (`rf_seed`) and
silently means seed 0.** Measured: `seed=None` bit-identical to `seed=0`
and different from `seed=1` on every one of the three, on the live
configuration (`method="enbpi"`; `rf_method="second_order"`). The
NumPy convention a user brings is `None` = fresh entropy, so two "unseeded"
runs agreeing exactly reads as robustness when it is one seed. Root cause:
`seed.unwrap_or(0)` / `rf_seed.unwrap_or(0)` with the sentinel documented
nowhere (the stub types it `int | None` without a word). **Fix**: all three
runtime docstrings and the stub state that `None` means seed 0 (and that
`n_boot=None` means 25 / `rf_draws=None` means 256); a test pins
`None ≡ 0 ≠ 1`.

**M5 (F2). The forecasting card's `backtest` table taught the 0.7.0
trap.** Its `period` row read "default `1`" for a kwarg that, since the
round-10 inert-argument fix, **raises** when passed explicitly with any
forecaster but `seasonal_naive`/`theta` (callables included); its
`forecaster` row read "default `"naive"`" while the signature default is
`None`. A reader copying the table's defaults into a call got a refusal.
**Fix**: both rows now show the signature defaults (`None (→ naive)`,
`None (→ 1)`) and the `period` row names the refusal and the separate
`insample_period` scale parameter; a test pins the rows against
`inspect.signature`.

**M6 (E3). Twenty-two further functions where the stub names returned keys
the runtime `help()` does not** — `adaptive_lasso`, `check_series`,
`connectedness`, `factor_model`, `flp`, `flp_scenario`, `functional_pca`,
`fvar_scenario`, `iv_gmm`, `ivx_test`, `jarque_bera`, `johansen`,
`nongaussian_svar`, `panel_lp`, `panel_pmg`, `panel_unit_root`,
`proxy_first_stage`, `quantile_regression`, `setar`, `smooth_lp`,
`sup_f_test`, `var_backtest` (3–14 keys each; `proxy_first_stage` also
returned `mop_cv_tau20`/`mop_cv_tau30`, named on no surface). Same class
as M1, each a smaller gap, so graded moderate as a group. **Fix**: a
`Keys:` sentence appended to each runtime doc comment (the stub's own
sentence where it had one, the measured key list otherwise;
`check_series`'s lives in `_inspect.py`); gated by the M1 test.

## Low

**L1 (E4). `max_rel_change` — returned by `lasso`, `elastic_net` and
`adaptive_lasso` — was documented on no surface** (the machine-learning
card lists `coef`, `n_iter`, `max_change`). It is the scale-free
`max_j |Δb_j|·‖x_j‖/‖y‖` the stopping rule actually compares with `tol`,
so it is the *more* useful of the two convergence diagnostics. Fixed in
the three docstrings and the stub; the card is owned by the ML wave (see
OPEN).

**L2 (E5). `local_level_smooth` returned six keys named nowhere**
(`filtered_state`, `filtered_state_var`, `smoothed_state`,
`smoothed_state_var`, `d_diffuse`, `loglik` named only in prose). Fixed
in docstring and stub.

**L3 (E6). The HP / Baxter-King / Christiano-Fitzgerald filters' one-line
docstrings named none of `trend`/`cycle`/`first_index`, and no model card
covers the trio** (only two guide chapters mention them). Docstrings and
stub fixed (`bk_filter` returns no trend and `first_index = k`; the other
two return the full sample); the missing card is OPEN.

**L4 (E7). Unnamed keys on four more functions**: `engle_granger`
(`coint_coefs`, `resid`, `used_lag`, `adf_nobs`, `n_vars`, `nobs` — prose
said "the step-1 coefficients and residuals"), `factor_model`
(`er_ratios`, and `er` on the runtime side), `gas_volatility`
(`converged`/`iterations` — "convergence info"), `zero_sign_svar`
(`arw_weighted`). Fixed in docstrings and stub.

**L5 (E8). `cg_regression`'s docstring named `se`/`t`/`p` as keys** —
"Returns `intercept`/`slope` with HAC `se`/`t`/`p`" — none of which
exists; the keys are `se_intercept`, `se_slope`, `t_slope`, `p_slope`
(there is no `t_intercept`/`p_intercept`). The one phantom-key candidate
of thirty that survived refutation. Fixed on both surfaces.

**L6 (E9). The panel card's "complete" key lists for `panel_lp` and
`lp_did` omitted the stamped settings keys** (`se_type`, `cumulative`,
`jackknife`, `bias_correction`; `absorbing`, `nonabsorbing_lag`,
`reweight`, `pooled`, `never_treated_only`, `se_type`) that the
docstrings document. Fixed; a test pins the lists.

**L7 (E10). Two imprecise "T" shape claims.** `dfm_nowcast.smoothed_factors`
was documented "(T, r)" but has one row per *balanced-panel* observation —
measured (200, 1) → (199, 1) → (198, 1) as 0/1/2 ragged-edge rows are
added; `proxy_svar.shock` was documented "(length T)" in the stub but has
T − lags rows (198 at T=200, lags=2; the runtime doc defined T as the
residual length, the stub did not). Both fixed and pinned.

**L8 (F5). `cv_splits(n)` with default arguments always raises** —
`train` defaults to 0, which the two walk-forward schemes refuse as an
empty first window ("purged_kfold" ignores it). The ML card's table said
"required > 0 for expanding/rolling"; the docstring and stub did not.
Surfaced by sweep G's default-arguments pass. Fixed on both.

## Sweep E — the rest of the ledger

- **(i)/(ii)** `summarize(res).summary()` rendered for 162/162; `json.dumps`
  (with an ndarray default) and `pickle` round-tripped 162/162 with every
  value bit-identical (NaN-aware compare). Nothing to report.
- **(iii)** Three non-finite results on canonical inputs: `dcs_local_level`
  (`nu_se` NaN under `density="t"` on this draw — documented per-parameter
  `*_se` semantics; refuted), `garch_fit` (one NaN in `se_mle`/`se_robust`
  with the boundary flag set — the round-7 contract; refuted),
  `robust_svar_bounds` (→ M1).
- **(iv)** The mechanical "returned key not named anywhere in `__doc__`"
  list had 60 entries, most on functions whose runtime docstring is a
  one-liner that names no keys at all (`adf`, `kpss`, `ljung_box`, …).
  Those are documented in the cards' `→ {"a", "b"}` lists (31 such lists
  checked, 29 exact), so they were **not** counted as findings; the
  confirmed items are the ones where a richer surface existed and
  disagreed (M1, M2, M6) or where *no* surface named the key (L1–L4). The
  30 phantom-key candidates (names in a "Returns"/"Keys" sentence not
  returned on the canonical call) refuted as conditional keys
  (`cov_error`, `nbar`, `nu`, `rho`, `pooled_*`, band keys, `inclusion_prob_cov`
  under `ssvs_cov=True` — verified present), nested keys (`ndiffs.steps[*]`,
  `dynamic_ns.forecast.*`), parameter names, or math tokens — all but
  `cg_regression` (L5).
- **(v)** 45 functions' array shapes diffed against their docstrings:
  `[horizon+1]`-first layouts, `(T, k, k)` covariance paths, `(N, r)`
  loadings, `(H+1) × K` FLP betas, `k × k` `q`/`impact` vectors, ACM's
  `T × M` panels — all as stated, except L7.

## Sweep F — the rest of the ledger

- **(a)** `inspect.signature` vs the stub: 162/162 agree on names, order
  and has-default. (The stub is regenerated into `api.md` by CI; the
  merge-corruption class round 10 found did not recur.)
- **(b)** 13 prose-default mismatches raised, 12 refuted as documented
  `None` sentinels (`panel_lp.bandwidth` "4.0 when omitted",
  `spread_zscore.dt`, `vecm.first_season`, `bn_filter.d0/dt`,
  `conformal_*.gamma`, …) or parser noise across a sentence boundary
  (`ols`/`har_rv` "pass use_correction=False to reproduce a default
  statsmodels call"). The thirteenth is F4 (OPEN).
- **(c)** Every keyword used in a `fn(...)` snippet across the docstrings,
  cards and `api.md` exists in the signature. The two card hits were
  `scipy.signal.welch(..., scaling=)` and R's `forecast::nsdiffs(test=)`
  (refuted); the ~160 `api.md` hits were the parser reading type
  annotations in signature blocks (tooling noise, excluded from the
  totals).
- **(d)** 43 listed string values probed, 39 accepted; the 4 refusals were
  `bootstrap_indices(scheme="moving"/"circular")` without `block_length`
  (legitimate) and the two EGARCH refusals (M3).
- **Cards**: 113 `| Argument | Default |` rows resolved against
  `inspect.signature`; 3 disagreed — the two `backtest` rows (M5) and
  `adf.autolag` (F4). Thirteen apparent hits on `check_stationarity` were
  the parser attributing the STL/MSTL tables to the wrong heading
  (refuted).
- **`kpss(nlags="auto")`** is bit-identical to `nlags=None` and `"legacy"`
  is accepted — the docstring's value list is exact (refuted candidate).

## Sweep G — complexity cliffs (the numbers are the deliverable)

Wall-clock seconds in a fresh subprocess per cell (4 cores, release
build, other sweeps running concurrently — so absolute values carry
noise; slopes were re-timed for every flagged cell and agreed within
0.2). "defaults only" passes nothing but the required arguments.
Canonical calls use small draw counts (`n_draws`/`n_boot` ≈ 40–50) where
a function samples; the defaults column shows the un-shrunk cost.

**Flags that survived re-timing**, all recorded OPEN because no
documented promise is violated:

| function | T=200 | T=800 | T=3200 | slope | defaults @3200 | note |
|---|---|---|---|---|---|---|
| `arima_fit` (AR(1)) | 0.27 | 1.13 | 5.39 | 1.08 | 6.10 | linear but ~2 ms per observation |
| `bn_decomposition` (ARIMA(2,1,2)) | 0.78 | 4.58 | 19.9 | 1.17 | 13.0 | same engine |
| `auto_arima` (max_p=max_q=2) | 4.96 | 17.3 | 70.0 | 0.95 | 62.4 (max 5/5) | same engine × candidates |
| `copula_select` | 0.005 | 0.026 | 0.18 | 1.27 | 23.1 | the default menu includes `"t"`: `copula_fit(family="t")` alone is 6.8 s at n=800 and 25.9 s at n=3200 (Gaussian/Clayton/Gumbel/Frank ≤ 0.17 s) |
| `historical_decomposition` | 0.003 | 0.014 | 0.42 | 1.96 | 0.43 | O(T²): the cumulated contributions are summed over all past shocks rather than recursed |
| `mcmc_diagnostics` | 0.001 | 0.007 | 0.09 | 1.68 | 0.09 | rank-normalized ESS autocorrelation without an FFT |

**Flags refuted by analysis**: `adf`/`check_stationarity` (slope 1.5–1.7:
the auto-lag search grows as T^0.25, so O(T^1.75) is the algorithm, and
0.08 s at T=3200), `cf_filter` (1.7–1.9: the exact asymmetric filter is
O(T²) by construction, statsmodels' too; 0.06 s), `cv_splits` (2.4–2.5:
the *output* is O(T²) index lists with `step=1`; 0.3 s). Twelve
"non-monotone" flags were all sub-31 ms cells (noise). `bai_perron`'s
O(n²) dynamic program (slope 1.94) is documented in `_inspect.py` and was
not on the linear-expected list. The five defaults-pass refusals were
legitimate (`bootstrap_indices` stationary needs `p`; `factor_model`'s
default `kmax=8` exceeds an 8-column panel's 7 eigenvalues;
`narrative_svar` needs a restriction; `var_backtest` read the returns as
hits) except `cv_splits` (L8).

<details><summary>Full table — 162 callables × {200, 800, 3200} + defaults-only at 3200</summary>

| function | T=200 | T=800 | T=3200 | log-log slope | T=3200, defaults only | flags |
|---|---|---|---|---|---|---|
| `accuracy` | 0.000 | 0.000 | 0.000 | 0.06 | 0.000 |  |
| `acf` | 0.000 | 0.000 | 0.000 | 0.19 | 0.005 |  |
| `acm_term_premium` | 0.002 | 0.011 | 0.038 | 1.02 | 0.053 |  |
| `adaptive_lasso` | 0.001 | 0.001 | 0.001 | 0.21 | 0.001 |  |
| `adf` | 0.001 | 0.009 | 0.084 | 1.49 | 0.073 | superlinear slope 1.67 (expected ~1) |
| `afns_adjustment` | 0.000 | 0.000 | 0.000 | 0.06 | 0.000 | non-monotone |
| `ar_loglik` | 0.001 | 0.006 | 0.010 | 0.90 | 0.011 |  |
| `arch_lm` | 0.000 | 0.000 | 0.001 | 0.36 | 0.000 |  |
| `arima_fit` | 0.274 | 1.128 | 5.393 | 1.08 | 6.100 | 5.4s at T=3200 (canonical kwargs); 6.1s at T=3200 with DEFAULT arguments |
| `auto_arima` | 4.960 | 17.346 | 70.021 | 0.95 | 62.403 | 70.0s at T=3200 (canonical kwargs); 62.4s at T=3200 with DEFAULT arguments |
| `backtest` | 0.000 | 0.001 | 0.012 | 1.25 | 0.019 |  |
| `bai_perron` | 0.001 | 0.006 | 0.136 | 1.94 | 0.182 |  |
| `bk_filter` | 0.000 | 0.000 | 0.000 | 0.14 | 0.005 |  |
| `bn_decomposition` | 0.783 | 4.584 | 19.865 | 1.17 | 13.025 | 19.9s at T=3200 (canonical kwargs); 13.0s at T=3200 with DEFAULT arguments |
| `bn_filter` | 0.030 | 0.083 | 0.279 | 0.80 | 0.333 |  |
| `bns_jump_test` | 0.000 | 0.000 | 0.001 | 0.21 | 0.001 |  |
| `bootstrap_indices` | 0.000 | 0.004 | 0.000 | 0.07 | — | non-monotone |
| `box_cox_lambda` | 0.001 | 0.001 | 0.004 | 0.74 | 0.004 |  |
| `bvar_fit` | 0.001 | 0.001 | 0.001 | 0.34 | 0.001 |  |
| `bvar_hierarchical` | 0.004 | 0.013 | 0.040 | 0.85 | 0.049 |  |
| `bvar_irf_draws` | 0.006 | 0.002 | 0.003 | -0.27 | 0.035 | non-monotone |
| `bvar_ssvs` | 0.010 | 0.014 | 0.028 | 0.37 | 1.527 |  |
| `ccc_garch` | 0.012 | 0.043 | 0.144 | 0.90 | 0.121 |  |
| `cf_filter` | 0.000 | 0.002 | 0.059 | 1.72 | 0.037 | superlinear slope 1.87 (expected ~1) |
| `cg_regression` | 0.000 | 0.000 | 0.001 | 0.23 | 0.001 |  |
| `check_series` | 0.015 | 0.017 | 0.093 | 0.65 | 0.065 |  |
| `check_stationarity` | 0.001 | 0.005 | 0.074 | 1.52 | 0.051 | superlinear slope 1.63 (expected ~1) |
| `chow_test` | 0.000 | 0.004 | 0.001 | 0.31 | 0.001 | non-monotone |
| `coherence` | 0.000 | 0.000 | 0.001 | 0.14 | 0.001 |  |
| `conformal_backtest` | 0.007 | 0.056 | 0.649 | 1.62 | 1.009 |  |
| `conformal_forecast` | 0.003 | 0.056 | 0.630 | 1.95 | 0.619 |  |
| `connectedness` | 0.001 | 0.001 | 0.002 | 0.30 | 0.002 |  |
| `copula_fit` | 0.003 | 0.012 | 0.075 | 1.23 | 0.075 |  |
| `copula_select` | 0.005 | 0.026 | 0.183 | 1.27 | 23.104 | 23.1s at T=3200 with DEFAULT arguments |
| `cusum_test` | 0.001 | 0.005 | 0.008 | 0.83 | 0.004 |  |
| `cv_splits` | 0.000 | 0.016 | 0.309 | 2.35 | — | superlinear slope 2.51 (expected ~1) |
| `cw_test` | 0.000 | 0.000 | 0.000 | 0.20 | 0.001 | non-monotone |
| `dcc_garch` | 0.051 | 0.200 | 0.798 | 0.99 | 0.776 |  |
| `dcc_test` | 0.007 | 0.038 | 0.103 | 0.95 | 0.110 |  |
| `dcs_local_level` | 0.025 | 0.042 | 0.189 | 0.72 | 0.173 |  |
| `dfgls` | 0.001 | 0.004 | 0.048 | 1.53 | 0.035 |  |
| `dfm_news` | 0.013 | 0.058 | 0.228 | 1.03 | 0.292 |  |
| `dfm_nowcast` | 0.005 | 0.018 | 0.076 | 0.99 | 0.114 |  |
| `dm_test` | 0.000 | 0.000 | 0.000 | 0.06 | 0.000 |  |
| `dsge_solve` | 0.000 | 0.000 | 0.000 | 0.03 | 0.000 |  |
| `dynamic_ns` | 0.001 | 0.002 | 0.016 | 0.95 | 0.015 |  |
| `elastic_net` | 0.000 | 0.000 | 0.001 | 0.22 | 0.001 |  |
| `engle_granger` | 0.001 | 0.009 | 0.040 | 1.32 | 0.056 |  |
| `factor_model` | 0.001 | 0.001 | 0.008 | 0.77 | — |  |
| `favar` | 0.001 | 0.002 | 0.007 | 0.69 | 0.009 |  |
| `flp` | 0.001 | 0.001 | 0.003 | 0.55 | 0.014 |  |
| `flp_scenario` | 0.001 | 0.001 | 0.004 | 0.63 | 0.013 |  |
| `forecast_disagreement` | 0.001 | 0.000 | 0.001 | 0.11 | 0.000 |  |
| `forecast_efficiency` | 0.001 | 0.001 | 0.001 | 0.16 | 0.001 |  |
| `frac_diff` | 0.000 | 0.001 | 0.007 | 1.01 | 0.012 |  |
| `frac_integrate` | 0.000 | 0.003 | 0.007 | 1.12 | 0.007 |  |
| `fry_pagan_svar` | 0.002 | 0.002 | 0.003 | 0.18 | 0.028 |  |
| `functional_pca` | 0.001 | 0.001 | 0.004 | 0.65 | 0.008 |  |
| `fvar_scenario` | 0.001 | 0.001 | 0.007 | 0.82 | 0.003 |  |
| `garch_fit` | 0.004 | 0.015 | 0.058 | 1.01 | 0.064 |  |
| `gas_volatility` | 0.003 | 0.010 | 0.051 | 1.05 | 0.041 |  |
| `gev_fit` | 0.006 | 0.002 | 0.002 | 0.07 | 0.128 | non-monotone |
| `gmm_nonlinear` | 0.036 | 0.159 | 0.663 | 1.06 | 0.625 |  |
| `gpd_fit` | 0.001 | 0.001 | 0.006 | 0.87 | 0.002 |  |
| `growth_at_risk` | 0.007 | 0.040 | 0.031 | 0.81 | 0.066 | non-monotone |
| `gw_test` | 0.000 | 0.000 | 0.000 | -0.09 | 0.000 |  |
| `hamilton_filter` | 0.000 | 0.000 | 0.001 | 0.18 | 0.001 |  |
| `hansen_seo_test` | 0.018 | 0.023 | 0.071 | 0.50 | 1.338 |  |
| `har_rv` | 0.000 | 0.001 | 0.005 | 0.93 | 0.001 |  |
| `hetero_svar` | 0.001 | 0.001 | 0.002 | 0.40 | 0.002 |  |
| `heteroskedasticity_test` | 0.000 | 0.000 | 0.003 | 0.84 | 0.001 |  |
| `historical_decomposition` | 0.003 | 0.014 | 0.423 | 1.96 | 0.434 | superlinear slope 1.85 (expected ~1) |
| `hp_filter` | 0.000 | 0.000 | 0.000 | 0.27 | 0.001 |  |
| `iv_gmm` | 0.001 | 0.001 | 0.001 | 0.33 | 0.006 |  |
| `ivx_test` | 0.001 | 0.001 | 0.001 | 0.24 | 0.001 |  |
| `jarque_bera` | 0.000 | 0.000 | 0.000 | 0.07 | 0.000 | non-monotone |
| `johansen` | 0.001 | 0.001 | 0.001 | 0.11 | 0.001 |  |
| `kpss` | 0.000 | 0.000 | 0.000 | 0.20 | 0.000 |  |
| `lasso` | 0.000 | 0.000 | 0.001 | 0.17 | 0.001 |  |
| `lasso_path` | 0.001 | 0.001 | 0.003 | 0.60 | 0.013 |  |
| `ljung_box` | 0.000 | 0.000 | 0.000 | 0.06 | 0.000 |  |
| `local_level_smooth` | 0.001 | 0.003 | 0.026 | 1.08 | 0.013 |  |
| `long_memory_d` | 0.001 | 0.001 | 0.001 | -0.07 | 0.001 |  |
| `long_run_svar` | 0.001 | 0.001 | 0.002 | 0.35 | 0.002 |  |
| `long_run_variance` | 0.000 | 0.000 | 0.000 | 0.07 | 0.000 |  |
| `lp` | 0.001 | 0.002 | 0.005 | 0.70 | 0.014 |  |
| `lp_did` | 0.001 | 0.003 | 0.011 | 1.00 | 0.022 |  |
| `lp_iv` | 0.004 | 0.014 | 0.047 | 0.88 | 0.068 |  |
| `lp_multiplier` | 0.009 | 0.033 | 0.126 | 0.93 | 0.682 |  |
| `lp_state` | 0.002 | 0.009 | 0.018 | 0.84 | 0.063 |  |
| `markov_switching_ar` | 0.011 | 0.019 | 0.060 | 0.62 | 0.053 |  |
| `max_share_svar` | 0.001 | 0.001 | 0.002 | 0.34 | 0.002 |  |
| `mcmc_diagnostics` | 0.001 | 0.007 | 0.090 | 1.68 | 0.091 | superlinear slope 1.66 (expected ~1) |
| `mean_group_var` | 0.001 | 0.001 | 0.003 | 0.44 | 0.003 |  |
| `midas_weights` | 0.000 | 0.000 | 0.000 | 0.11 | 0.000 |  |
| `mstl` | 0.002 | 0.006 | 0.021 | 0.92 | 0.020 |  |
| `narrative_svar` | 0.002 | 0.003 | 0.003 | 0.14 | — |  |
| `ndiffs` | 0.000 | 0.000 | 0.001 | 0.18 | 0.001 |  |
| `nelson_siegel` | 0.000 | 0.000 | 0.000 | 0.14 | 0.000 |  |
| `ng_perron` | 0.001 | 0.004 | 0.040 | 1.46 | 0.038 |  |
| `nongaussian_svar` | 0.001 | 0.002 | 0.005 | 0.57 | 0.005 |  |
| `nsdiffs` | 0.001 | 0.002 | 0.007 | 0.62 | 0.007 |  |
| `ols` | 0.000 | 0.001 | 0.001 | 0.43 | 0.001 |  |
| `optimal_block_length` | 0.000 | 0.001 | 0.001 | 0.19 | 0.001 |  |
| `ou_fit` | 0.000 | 0.000 | 0.000 | -0.02 | 0.000 | non-monotone |
| `pacf` | 0.000 | 0.000 | 0.000 | 0.12 | 0.001 |  |
| `panel_fe` | 0.001 | 0.001 | 0.004 | 0.56 | 0.004 |  |
| `panel_lp` | 0.003 | 0.010 | 0.052 | 1.03 | 0.061 |  |
| `panel_mean_group` | 0.000 | 0.001 | 0.001 | 0.40 | 0.001 |  |
| `panel_pmg` | 0.005 | 0.001 | 0.007 | 0.50 | 0.003 | non-monotone |
| `panel_unit_root` | 0.000 | 0.001 | 0.001 | 0.23 | 0.216 |  |
| `periodogram` | 0.001 | 0.001 | 0.001 | 0.12 | 0.001 |  |
| `phillips_ouliaris` | 0.000 | 0.000 | 0.001 | 0.21 | 0.001 |  |
| `phillips_perron` | 0.000 | 0.001 | 0.001 | 0.16 | 0.001 |  |
| `philox_uniforms` | 0.000 | 0.000 | 0.000 | 0.17 | 0.000 |  |
| `predictive_regression` | 0.000 | 0.000 | 0.001 | 0.41 | 0.001 |  |
| `proxy_ar_sets` | 0.001 | 0.001 | 0.002 | 0.37 | 0.002 |  |
| `proxy_first_stage` | 0.001 | 0.001 | 0.002 | 0.24 | 0.002 |  |
| `proxy_svar` | 0.001 | 0.001 | 0.002 | 0.23 | 0.002 |  |
| `proxy_svar_bands` | 0.006 | 0.010 | 0.045 | 0.74 | 1.272 |  |
| `pseudo_obs` | 0.000 | 0.000 | 0.001 | 0.31 | 0.005 |  |
| `quantile_lp` | 0.066 | 0.441 | 0.918 | 0.95 | 4.203 |  |
| `quantile_regression` | 0.003 | 0.006 | 0.062 | 1.06 | 0.081 |  |
| `realized_measures` | 0.000 | 0.000 | 0.000 | 0.14 | 0.000 |  |
| `realized_quarticity` | 0.000 | 0.000 | 0.000 | 0.12 | 0.000 |  |
| `realized_range` | 0.000 | 0.000 | 0.004 | 1.01 | 0.000 |  |
| `recession_probit` | 0.001 | 0.001 | 0.004 | 0.66 | 0.005 |  |
| `reset_test` | 0.000 | 0.000 | 0.001 | 0.41 | 0.001 |  |
| `ridge` | 0.001 | 0.001 | 0.001 | 0.16 | 0.001 |  |
| `robust_svar_bounds` | 0.003 | 0.003 | 0.004 | 0.09 | 0.056 |  |
| `seasonal_strength` | 0.001 | 0.001 | 0.004 | 0.66 | 0.004 |  |
| `setar` | 0.000 | 0.001 | 0.002 | 0.56 | 0.002 |  |
| `setar_test` | 0.002 | 0.008 | 0.028 | 0.89 | 0.105 |  |
| `sign_restricted_svar` | 0.001 | 0.002 | 0.003 | 0.19 | 0.020 |  |
| `smooth_lp` | 0.010 | 0.039 | 0.165 | 1.02 | 0.722 |  |
| `spread_zscore` | 0.000 | 0.000 | 0.000 | 0.16 | 0.000 |  |
| `star` | 0.003 | 0.011 | 0.044 | 0.98 | 0.136 |  |
| `star_eval` | 0.000 | 0.000 | 0.001 | 0.38 | 0.001 |  |
| `star_test` | 0.000 | 0.000 | 0.001 | 0.45 | 0.001 |  |
| `stl` | 0.001 | 0.001 | 0.004 | 0.60 | 0.004 |  |
| `structural_fevd` | 0.001 | 0.001 | 0.002 | 0.35 | 0.001 |  |
| `summarize` | 0.000 | 0.000 | 0.000 | -0.35 | 0.000 | non-monotone |
| `sup_f_test` | 0.000 | 0.000 | 0.000 | 0.12 | 0.000 |  |
| `svensson` | 0.000 | 0.000 | 0.000 | 0.06 | 0.000 | non-monotone |
| `theta_forecast` | 0.000 | 0.001 | 0.001 | 0.50 | 0.001 |  |
| `threshold_var` | 0.001 | 0.002 | 0.009 | 0.90 | 0.009 |  |
| `threshold_var_test` | 0.015 | 0.033 | 0.103 | 0.69 | 1.541 |  |
| `threshold_vecm` | 0.002 | 0.004 | 0.012 | 0.63 | 0.080 |  |
| `tripower_quarticity` | 0.000 | 0.000 | 0.001 | 0.21 | 0.005 |  |
| `umidas` | 0.000 | 0.001 | 0.003 | 0.70 | 0.007 |  |
| `var_backtest` | 0.000 | 0.001 | 0.001 | 0.27 | — |  |
| `var_fevd` | 0.001 | 0.001 | 0.002 | 0.29 | 0.001 |  |
| `var_fit` | 0.001 | 0.002 | 0.006 | 0.63 | 0.005 |  |
| `var_forecast` | 0.001 | 0.001 | 0.002 | 0.24 | 0.002 |  |
| `var_granger` | 0.001 | 0.001 | 0.002 | 0.36 | 0.002 |  |
| `var_irf` | 0.001 | 0.001 | 0.002 | 0.34 | 0.002 |  |
| `var_irf_bands` | 0.001 | 0.001 | 0.002 | 0.24 | 0.002 |  |
| `vecm` | 0.001 | 0.001 | 0.001 | 0.29 | 0.001 |  |
| `weighted_midas` | 0.002 | 0.009 | 0.029 | 0.91 | 0.033 |  |
| `welch` | 0.001 | 0.000 | 0.001 | 0.00 | 0.001 |  |
| `zero_sign_svar` | 0.002 | 0.002 | 0.003 | 0.17 | 0.029 |  |
| `zivot_andrews` | 0.002 | 0.027 | 0.351 | 1.87 | 0.305 |  |

</details>

## Sweep H — the seed ledger

| parameter | same seed, in-process | same seed, new process | different seed differs | `None` accepted |
|---|---|---|---|---|
| `bootstrap_indices.seed`, `philox_uniforms.seed` | ✓ | ✓ | ✓ | no |
| `var_irf_bands.seed` (bootstrap branch), `.band_seed` (sup-t) | ✓ | ✓ | ✓ | no |
| `var_forecast.band_seed`, `lp.band_seed`, `smooth_lp.band_seed` (sup-t) | ✓ | ✓ | ✓ | no |
| `bvar_irf_draws.seed`, `bvar_ssvs.seed` | ✓ | ✓ | ✓ | no |
| `sign_restricted_svar`, `zero_sign_svar`, `narrative_svar`, `fry_pagan_svar`, `robust_svar_bounds` `.seed` | ✓ | ✓ | ✓ | no |
| `historical_decomposition.seed` (`identification="sign"`) | ✓ | ✓ | ✓ | no |
| `proxy_svar_bands.seed` | ✓ | ✓ | ✓ | no |
| `hansen_seo_test`, `setar_test`, `threshold_var_test` `.seed` (also 1 vs 4 threads) | ✓ | ✓ | ✓ | no |
| `conformal_forecast.seed`, `conformal_backtest.seed` (enbpi) | ✓ | ✓ | ✓ | **yes → 0 (M4)** |
| `proxy_ar_sets.rf_seed` (second_order) | ✓ | ✓ | ✓ | **yes → 0 (M4)** |

`historical_decomposition` on its default `identification="cholesky"`
refuses an explicit `seed` (the round-2 sentinel), which is why the live
configuration is the sign mode. Three harness bugs (a positional/keyword
`seed` collision on `philox_uniforms`, EnbPI's `horizon=1` requirement in
`conformal_backtest`, JSON turning restriction tuples into lists) were
fixed and re-run; none was a library defect.

## OPEN — recorded, not fixed

1. **`inspect.signature` renders eight defaults as `...`** (`Ellipsis`):
   `adf`/`zivot_andrews`/`engle_granger` `autolag="aic"`,
   `box_cox_lambda.bounds=(-2.0, 2.0)`,
   `historical_decomposition.restrictions=[]`,
   `narrative_svar.sign_restrictions=[]`,
   `predictive_regression`/`ivx_test` `cz=-1.0`. PyO3's
   `__text_signature__` cannot express `Some(..)`, tuple, `vec![]` or
   negative-literal defaults, so IDE hovers show `autolag=...` while the
   docstrings and cards state the real default. Fixing it means changing
   `#[pyo3(signature)]` attributes, outside this round's edit scope.
   Status: confirmed, low, unfixed.
2. **`check_stationarity` returns `adf_statistic`/`adf_p_value`/
   `kpss_statistic`/`kpss_p_value`/`alpha` named on no surface** (the card
   says "the raw test statistics/p-values"). Confirmed, low; left for the
   diagnostics-card owner with the one-line fix noted.
3. **The machine-learning card's `adaptive_lasso` key list omits
   `max_rel_change`** (L1 fixed the docstrings and stub; the card belongs
   to the ML wave). Confirmed, low.
4. **No model card covers `hp_filter`/`bk_filter`/`cf_filter`**; the
   diagnostics card's trend-cycle section covers Hamilton and BN only.
   Confirmed, low.
5. **The ARIMA exact-MLE engine's constant** (G): ~2 ms per observation
   per fit, so `arima_fit` is 6 s and `auto_arima` 62 s at T=3200 with
   defaults. Linear, honest, and undocumented as a cost; a performance
   item, not a correctness one.
6. **The t-copula MLE's constant** (G): 26 s at n=3200 while the four
   other families take ≤ 0.2 s, so `copula_select`'s default menu is 130×
   the cost of its second-slowest member.
7. **`historical_decomposition` is O(T²)** (G): 0.4 s at T=3200, so ~40 s
   at 30 000 rows; a companion-form recursion would make it O(T).
8. **`mcmc_diagnostics` scales as T^1.7** (G): 0.09 s at 3200 draws; an
   FFT autocorrelation would make it T log T.

## Verification record

`cargo test -p tsecon-garch`: 30 passed over four binaries (9 unit, 4
golden, 15 property, 2 doc); `cargo clippy -p tsecon-garch` and
`-p tsecon-python` with `-D warnings` clean; `cargo fmt --all --check`
clean; `maturin develop --release` rebuilt; `pytest` on the new file plus
the 23 touched/adjacent files: **420 passed, 1 skipped** (the round's own
file: 41 parametrized key-gate cases, 3 seed/EGARCH/shape pins, 3 card
pins). `lib.rs` changed in doc comments only (187 lines added, 7
replaced, zero code lines); the one crate edit is the EGARCH message
text.

## Lessons

1. **The three surfaces drift in a predictable direction.** Round 4 found
   one runtime docstring staler than its stub; round 11 found 29, all the
   same shape — a stub extended with the return contract while the `///`
   comment kept its first-day one-liner. Measured before this round's sync, 143 of 162 runtime
   docstrings were already *longer* than their stub, so the direction is not "runtime is
   always worse": it is that key lists get added to whichever surface the
   author had open. The new test gates the binding surface for the 29;
   the durable fix is a generator that emits both from one source.
2. **A default-arguments pass finds what a canonical-arguments pass
   cannot.** Sweep G's "required arguments only" column was meant to
   catch cost; it caught `cv_splits(n)` always raising and `copula_select`'s
   130× default menu, neither visible with the sweep's own tuned kwargs.
3. **`None` is a promise in Python.** Three functions accepted `seed=None`
   and quietly chose 0. The sweep-H harness should carry the NumPy
   expectation (`None` = entropy) as its null and treat determinism under
   `None` as the candidate, which is how M4 surfaced.
4. **Parser-heavy sweeps need a reached/attempted line per lens** (the
   brief's rule). The kwargs-in-snippets lens produced ~160 false hits
   from type annotations before one true hit; reporting them as candidates
   would have buried the round. They were excluded from the totals and
   said so here.
