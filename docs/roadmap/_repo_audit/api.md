# Repository audit, sweep: API consistency across the whole surface

> **Working document.** One of the seven sweeps of the whole-repository
> audit run on tsecon 0.8.0 at `19d308e` (173 public callables), under
> [the brief](../16-adversarial-audit-brief.md). Excluded from the
> published site. Probe scripts, the master table and every raw output
> are committed under `lab/audit/repo/api/`.

## Scope and method

This sweep looks at the thing a user meets after the first hour and that
no round had looked at across all 173 functions at once: whether the
surface is *one* surface. Every callable was driven through the same
canonical, seeded, tiny call (`lab/audit/repo/api/registry.py` — the
round-11 registry of 162 extended with one builder per 0.8.0 callable;
173/173 reached) and six probes were run on top of it:

1. **Master table** (`probe_surface.py` → `surface.json`): the compiled
   signature (`inspect.signature` resolves through the coercion wrapper's
   `__wrapped__` to PyO3's text signature), each parameter's runtime default
   and stub annotation, the stub return annotation, the family from the
   model-card table in `docs/reference/README.md` (stub section heading
   where the table is silent), the runtime docstring's first line, and the
   returned key set with a value kind per key (scalar / 1-D / 2-D / list /
   dict / str / bool / None; one level of nested dicts). The compact table
   is the appendix of this document.
2. **Parameter-name clusters** (`probe_clusters.py`): every parameter
   spelling grouped by concept, with the functions using each.
3. **Return-key clusters** (same script): every returned key grouped by
   concept.
4. **Convention compliance, probed not read** (`probe_conventions.py`,
   `summarize_conventions.py`): for every callable, five malformed calls
   built from its canonical call — a NaN in the first data array; that
   array empty; that array with the wrong rank; `"nonsense_xyz"` for the
   first string-valued parameter; `-1` for the first integer-typed
   parameter — each in its own subprocess with a timeout, recording the
   exception class (`PanicException` is a `BaseException`, caught
   separately), whether the message names the argument, names the
   function, states a fix, or whether the call returned silently (and
   whether that silent return carried non-finite values). Plus: does every
   seed-taking function state the seed's default; how do fit-and-predict
   functions spell their prediction input; does every dict-returning
   function hand back NumPy arrays (`probe_shapes.py` covers the
   multivariate input orientation).
5. **Docstring structure** (`probe_docstrings.py`): on `help(tsecon.fn)`
   and on the stub — a one-line summary, every parameter named, every
   returned key backticked (the round-3/11 tripwire rule), a citation, a
   validation statement; the stub's return annotation against the returned
   kind and each parameter's stub type against its runtime default.
6. **Blast radius** (`probe_blast.py`): mentions of each spelling across
   the docs, the reference, the stub, the bindings, the tests and the
   examples, for the rename proposal.

Every finding below carries the probe output that produced it. Severity:
severe = a panic escape or a silent wrong-type return; moderate = a
convention collision that will bite users or an undocumented returned
key; low = spelling drift.

## Totals

| item | number |
|---|---|
| public callables / registry entries / canonical calls that succeed | 173 / 173 / 173 |
| return kinds | dict 154 · 1-D array 9 · float 5 · nested `list` 3 (`var_irf`, `var_fevd`, `bvar_irf_draws`) · `list[dict]` 1 (`cv_splits`) · 2-D array 1 (`pseudo_obs`) |
| distinct parameter names / parameter slots | 350 / 1019 |
| distinct top-level returned keys / nested keys | 733 / 134 |
| malformed calls made | 865 (173 × 5); 0 subprocess crashes, 0 timeouts |
| panic escapes (`PanicException`) | **0** |
| silent returns | 8 (4 NaN, 3 rank, 1 string) — 7 by documented design, 1 undocumented (`ar_loglik`), see F3 |
| functions fully compliant / partial / non-compliant on the five probes | **156 / 16 / 1** (the 1 is `summarize`, whose only string argument is a free-text title) |
| dict-returning functions handing back nested Python lists for matrix keys | 54 functions, 197 keys (+ 3 top-level list returns); 9 of the 54 docstrings say "list" |
| functions naming none of their returned keys on either surface / naming them without backticks | 33 / 11 (0 of the eleven 0.8.0 callables) — **fixed** |
| functions with option parameters never named in `help()` | 73 functions, 204 parameters (stub docstrings: 119 functions) — **fixed**, one generated line each |
| seed-taking parameters whose default `help()` did not state | 19 of 26 — 7 fixed by the generated line, 12 open (F6) |
| findings | 15 (0 severe, 5 moderate, 10 low); 5 fixed in-branch, 10 are proposals |

Post-fix docstring totals (re-run on the rebuilt module, `probe_docstrings.py`): functions with an unnamed option parameter 73 → 1, with an unbackticked returned key 44 → 1, with an unmentioned returned key 34 → 1 — the 1 is `summarize` in each case, whose "keys" are the probe's own input dict and whose only string argument is a free-text title. Citation 143/173 and grade 99/173 are unchanged (not targeted). `test_api_audit.py`: 33 parametrized tripwires + 2 tests, all passing alongside `test_stub_sync.py`, `test_docstring_keys.py`, `test_audit_round11.py`, `test_smoke.py`, `test_coerce.py` (165 passed).

## Findings

### Moderate

**F1 (moderate). 54 dict-returning functions hand back nested Python lists
for matrix-valued keys — and `var_irf`, `var_fevd`, `bvar_irf_draws` at the
top level — while the generated API reference promised "Every function
returns plain NumPy arrays and dictionaries".** Probe: `type(var_fit(d)["params"])`
is `list` (of lists); the same for `resid`, `fitted`, `sigma_u`; 197 such keys
across 54 functions (`surface.json`, kinds `list[list]`/`list[num]`) — the
VAR/SVAR family (`var_fit`, `var_forecast`, `var_irf_bands`, every
identification scheme), the Bayesian family, `ccc_garch`/`dcc_garch`,
`johansen`/`vecm`/`threshold_*`, the panel and term-structure families,
`quantile_regression`, `growth_at_risk`, `flp`, `dfm_nowcast`, `favar`,
`factor_model`, `functional_pca`. Root cause: 128 `mat_to_vec2` call sites in
`lib.rs` (`Vec<Vec<f64>>` → list of lists) against `into_pyarray` elsewhere
— the pattern is per-family, not per-key, so `ols` returns arrays and
`var_fit` returns lists for the same concept (`params`). Only 9 of the 54
runtime docstrings say "list" anywhere; the stub types `var_irf` correctly
as `list[list[list[float]]]`. Consequences: `r["params"].shape` fails,
`r["params"][:, 0]` fails, `np.asarray` is needed before any arithmetic,
and `summarize`/JSON work because they walk lists. **Fixed (docs)**: the
`api.md` preamble (generated by `docs/gen_api_reference.py`) now states the
split and names `np.asarray` as the conversion; a test pins the new wording.
**Proposal** (code, at 1.0 or behind a flag): convert at the boundary with
`into_pyarray` on all 128 sites — a type change, not a rename, so it needs
its own deprecation note; `np.asarray` users see no difference.

**F2 (moderate). `alpha` means three different things across 24 functions,
and the concept "confidence level" has four spellings in two orientations.**
From `surface.json` + each docstring: `alpha` is a **significance level /
miscoverage** (0.05 or 0.1) in 13 functions (`auto_arima`, `check_series`,
`check_stationarity`, `conformal_backtest`, `conformal_forecast`, `ndiffs`,
`nsdiffs` — validated but unused, by its own docstring —, `proxy_ar_sets`,
`proxy_svar_bands`, `robust_svar_bounds`, `var_backtest`, `var_forecast`,
`var_irf_bands`); a **penalty strength** in 9 (`ridge`, `lasso`,
`elastic_net`, `adaptive_lasso`, `group_lasso`, `post_lasso`, `pds_lasso`,
`kernel_ridge`, `mlp_regression` — scikit-learn's spelling, justified by the
literature); and the **IVX instrument-persistence exponent** (0.95) in 2
(`ivx_test`, `predictive_regression`), where a user reading `alpha=0.95`
as "95% confidence" is wrong and nothing in the signature says so
(`predictive_regression(alpha=0.05)` runs silently with a near-stationary
instrument). Around it, the same concept is spelled `band_alpha` (6, LP
family, 0.1), `conf_alpha` (2, ARIMA, `None`), and `level` (1, `ou_fit`,
0.95 — the *confidence* orientation, the opposite of `alpha`). The returned
key `alpha` (11 functions) is an echoed level in 9, the VECM adjustment
loadings in `vecm`, and the regression intercept nested in
`predictive_regression.ols`. **Fixed (docs)**: `ivx_test` now states what
its `alpha` is and is not (it was not mentioned by name at all). **Proposal**:
converge on `alpha` = miscoverage everywhere a level is meant; rename the
IVX exponent to `ivx_beta` (the paper's own symbol); see the rename table.

**F3 (moderate). `ar_loglik` silently treats NaN in `y` as a missing
observation, and no user-facing surface said so.** Probe: with `coeffs=[0.0]`
(separable likelihood) the full series gives `-67.116945`, the series with
`y[10] = NaN` gives `-66.003771`, and the series with `y[10]` *deleted*
gives `-66.003771` — bit-identical, so the NaN is skipped, not propagated;
with `coeffs=[0.5]` the value moves from `-65.5646` to `-64.7461`. The SSM
crate documents it (`filter.rs`: "NaN in `y` means missing; infinities are
rejected") and `local_level_smooth` says "NaN entries in `y` are treated as
missing", but `ar_loglik`'s docstring was a single sentence about
statsmodels conventions. The house convention is refusal-naming-the-argument
(the sentinel rule), so a user who expects the refusal gets a finite,
slightly different log-likelihood instead. **Fixed (docs + test)**: the
docstring and stub state the behaviour and the `inf` refusal; the test pins
both.

**F4 (moderate). 33 functions named none of their returned keys on either
surface** — the class audit 11 fixed 22 members of, with the same root cause
(a one-line `///` summary). Probe: `probe_docstrings.py`,
`keys_unmentioned_runtime ∧ keys_unmentioned_stub`: `accuracy` (7 keys),
`acf`, `adf` (5: `statistic`, `p_value`, `crit`, `used_lag`, `nobs`),
`arch_lm`, `backtest`, `bai_perron` (17), `bvar_hierarchical` (10),
`check_stationarity`, `chow_test`, `cusum_test`, `cw_test`,
`dcs_local_level`, `dm_test`, `growth_at_risk`, `gw_test`, `har_rv`,
`heteroskedasticity_test`, `kpss`, `ljung_box`, `lp_iv`, `lp_multiplier`,
`mcmc_diagnostics`, `ols`, `panel_fe`, `phillips_ouliaris`,
`phillips_perron`, `proxy_ar_sets`, `proxy_svar_bands` (14), `quantile_lp`,
`reset_test`, `svensson`, `umidas`, `var_granger`. **No recidivists**: all
eleven 0.8.0 callables name every key (most carry a `Keys:` line). **Fixed
(docs + test)**: a `Returned keys:` line naming every key of the canonical
call, on both surfaces, inserted by the committed script
`apply_doc_fixes.py`; `test_api_audit.py` parametrizes the round-3/11
tripwire over all 33. Eleven more functions mentioned their keys bare
(`Returns factors (T x K), factor_loadings, …`) and received the same
backticked line beside the prose (`acm_term_premium`, `auto_arima`,
`bn_filter`, `copula_fit`, `copula_select`, `lp`, `nelson_siegel`,
`optimal_block_length`, `realized_measures`, `setar_test`,
`sign_restricted_svar`); the stub keeps its own bare `Keys:` style where
it already had one (the backtick rule binds `__doc__`, not the stub).

**F5 (moderate). 73 functions never name 204 of their option parameters in
`help()` (the stub: 119 functions), so their defaults are invisible on the
surface a user reads.** Probe: `probe_docstrings.py`,
`params_missing_runtime` after excluding the leading data arguments. The
gap is family-shaped: the VAR family's `lags`/`trend` (`var_irf`,
`var_fevd`, `var_forecast`, `var_granger`, `var_irf_bands`,
`structural_fevd`, `max_share_svar`, `long_run_svar`), the Bayesian SVAR
family's `lags`/`n_draws`/`max_tries`/`seed`/`lambda1`
(`sign_restricted_svar`, `zero_sign_svar`, `narrative_svar`,
`fry_pagan_svar`, `robust_svar_bounds`, `historical_decomposition`), the
STL/MSTL loess options (16 parameters), the spectral `fs`/`nperseg`/
`noverlap`/`window`, `bvar_fit`'s hyperparameters, `auto_arima`'s nine
search bounds. Nine parameters are worse off: their compiled default
renders as `...` in `inspect.signature`/`help()` (PyO3 cannot print a
non-literal default: `autolag` in `adf`/`engle_granger`/`zivot_andrews`,
`box_cox_lambda.bounds`, `cz` in `ivx_test`/`predictive_regression`,
`historical_decomposition.restrictions`, `narrative_svar.sign_restrictions`,
`random_forest.max_features`), so prose is the *only* place the default can
live — and `ivx_test.cz`, `narrative_svar.sign_restrictions` and
`historical_decomposition.restrictions` were never named. **Fixed (docs)**:
a `Further arguments, with defaults:` line generated from the compiled
signature (correct by construction) on both surfaces for every unnamed
parameter with a printable default, plus hand-written sentences for the
three `...`-default parameters.

### Low

**F6 (low). Seed defaults and seed spellings.** 19 of the 26 seed-taking
parameters did not state their default in `help()`; two spellings coexist —
`seed` (21 functions, default 0) and `band_seed` (4: `lp`, `smooth_lp`,
`var_forecast`, `var_irf_bands`, default `20260807`, a date-shaped magic
number) — and `var_irf_bands` takes both (`seed` for the bootstrap,
`band_seed` for the sup-t simulation). `rf_seed` (`proxy_ar_sets`) and the
conformal pair default to `None`, documented since round 11 as seed 0.
**Partially fixed**: the F5 line states the default where `seed` was among
the unnamed parameters (7 parameters); 12 parameters on 11 functions still
mention the seed in prose without its value (open; list under Open).

**F7 (low). Three spellings for "maximum lag".** `maxlags` (8 functions),
`max_lags` (4), `maxlag` (2, `adf`/`engle_granger`), alongside `nlags` (5,
diagnostics), `hac_maxlags` (4) vs `hac_lags` (3) vs `lrv_lags` (2) for the
HAC bandwidth-in-lags. `lags` (38) and `p` (16) are the settled order
spellings and are justified (VAR/LP `lags`; AR order `p`). Blast radius of
the two outliers: `max_lags` 17 docs / 28 bindings / 16 tests mentions;
`maxlag` 7 / 17 / 4.

**F8 (low). Ten spellings for "horizon".** `horizon` (30 functions),
`horizons` (8: the LP family — a scalar maximum, plural because the return
is a vector), `forecast_horizon` (3: GARCH family), `forecast_steps` (2:
ARIMA), `steps` (2: `var_forecast`, `theta_forecast`), `h` (2: `dm_test`,
`hamilton_filter`), `h1` (`max_share_svar`, a *target* horizon — justified),
`post_window`/`pre_window` (`lp_did` — justified, they bound an event
window), `n_steps` (`boosting` — iterations, not a horizon). The 7
functions taking `x_test` (all 0.8.0 ML) are consistent with each other and
with scikit-learn; the older penalized family (`ridge`, `lasso`,
`elastic_net`, `adaptive_lasso`, `group_lasso`, `post_lasso`, `pds_lasso`)
returns `coef` and offers no prediction input, so "predict" means
`x_new @ coef` by hand — a family split rather than a spelling.

**F9 (low). Return-key spellings for the nine core concepts** (full tables
below): standard errors `se` (16) vs `bse` (12) — the statsmodels heritage
of the OLS/ARIMA/GARCH/panel/quantile families vs the newer `se`; p-values
`p_value` (16) vs `pvalue` (8) vs `p` (1, `ljung_box`'s per-lag) vs
`pvalues` (1); test statistics `stat` (9) vs `statistic` (8) vs `tvalues`
(5) vs `tstat`/`t_stat` (1 each); coefficients `params` (17) vs `coef` (8)
vs `beta` (7) vs `coefs` (2); log-likelihood `loglik` (19) vs `llf` (4);
residuals `residuals` (6) vs `resid` (5); iterations `iterations` (7) vs
`n_iter` (6); R² `rsquared` (6) vs `r_squared` (2); intervals
`lower`/`upper` (6 each) vs `conf_int` (1, `pds_lasso`) vs `ci_lower_90`
(`bai_perron`) vs `set_min`/`set_max` (identified sets — justified) vs
`half_life_ci`/`tail_lower`. Per-parameter standard errors flip between prefix and suffix: `se_beta`/`se_xi`/`se_mu`/`se_sigma` (EVT, copulas, GPD) vs `kappa_se`/`mu_se`/`sigma_se`/`phi_se`/`theta_se` (OU, DCS, PMG) — 12 prefix keys, 9 suffix keys. The dominant spelling is different in different
families because each family copied its reference package: `bse`/`tvalues`/
`pvalues`/`llf`/`rsquared` are statsmodels, `coef`/`n_iter`/`dual_coef` are
scikit-learn, `se`/`p_value`/`statistic`/`loglik` are tsecon's own.

**F10 (low). Error-message house style drifts where no shared formatter
exists.** The two probes that go through one code path are uniform: every
negative-count message reads `<param>=-1 is negative, but this parameter
counts something …` (131/131 name the parameter — the round-6/7 central
wrapper), and 86/87 unknown-string messages read `unknown <param> "…";
expected …` (all 87 name the parameter, 86 list the accepted values). The
two that do not: 59 distinct openings for a NaN in the data over 164
functions (`non-finite value (NaN or inf) in the data matrix at row N` ×30,
`input contains a non-finite value (NaN) at index N` ×16, `non-finite value
(NaN or infinity) in x` ×14, `the input series has a non-finite value` ×11,
…) — 103/164 name the argument, 123/164 state a fix; 81 distinct openings
for an empty input over 168 functions — 64/168 name the argument, 148/168
state a fix. Nothing is wrong in any single message; the drift is what a
user notices when the same mistake in two functions reads two ways.

**F11 (low). Four messages blame the wrong thing.** `gmm_nonlinear` with
`initial` passed as a column matrix raises the raw `TypeError: only
0-dimensional arrays can be converted to Python scalars`; with a NaN in
`initial` it says `input `x0` contains NaN` (the optimizer's internal name;
the parameter is `initial`); `hetero_svar` with empty `data` says
`regime_labels length 200 != number of observations 0` (the labels are
fine); `iv_gmm` with empty `x` says `dimension mismatch: regressor column
length vs y (expected 200, got 0)`. All four are ordinary `ValueError`/
`TypeError`s, not panics, so they are proposals.

**F12 (low). The rank-mismatch rebuild names arguments by position.** The
coercion wrapper's `_rank_error` (the message behind 164/165 rank probes)
writes `got arg0: array(200, 1)` although the parameter names are one
`inspect.signature` away — the sibling `_negative_int_error` already
resolves them (`lags=-1`). 26/165 rank messages name the argument, all 165
state the fix.

**F13 (low). Validation grade and citation on the runtime surface.** With a
deliberately loose regex, 143/173 runtime docstrings carry a citation and
99/173 a validation statement ("validated against", "pinned", "matches
statsmodels at 1e-8", "Monte-Carlo", "property"); the 74 without a grade
statement are mostly the pre-0.5 families whose grade lives on the model
card only (`kpss`, `arch_lm`, `dm_test`, `cw_test`, `gw_test`, `bvar_fit`,
`var_irf`, `var_fevd`, `vecm`, …). Every 0.8.0 callable states both. Table
only; no fix.

**F14 (low). Multivariate input orientation.** 89 callables take a 2-D or
list-of-arrays input. 81 take `(T, k)` — observations down, series across
— matching the coercion layer's rank message ("a 2-D array shaped
(observations, series)"). Five take `(N, T)` panels (`panel_fe` — plus a
`(k, N, T)` regressor cube —, `panel_lp`, `lp_did`, `panel_unit_root`,
`mcmc_diagnostics` as `(chains, draws)`), and three take a list of per-unit
arrays (`mean_group_var`, `panel_mean_group`, `panel_pmg`; plus
`forecast_disagreement`'s list of cross-sections). The flip is a family
convention (balanced panels as `(N, T)`) and each docstring states it, but
the rank message's `(observations, series)` advice is wrong for those
five. No fix; noted for the rename/convention pass.

**F15 (low). Stub types agree with what is returned — with one alias
quirk.** After treating `_F64` as an array alias, no stub return annotation
disagrees with the returned kind on any of the 173 (`summarize` is `Any`,
returning the input's type); no stub parameter type contradicts its
runtime default except the three `...`-rendered defaults (`bounds`, `cz`
×2), which are the PyO3 quirk of F5, not a stub error; parameter order is
identical on all 173. Clean.

## Parameter-cluster tables

Spelling → number of functions → functions (first twelve). Full lists in
`lab/audit/repo/api/out/clusters.md`; `param_defaults` in `clusters.json`
carries the default per spelling.

| concept | spelling | n functions | functions |
|---|---|---|---|
| randomness | `seed` | 21 | `bootstrap_indices`, `bvar_irf_draws`, `bvar_ssvs`, `conformal_backtest`, `conformal_forecast`, `echo_state_network`, `fry_pagan_svar`, `hansen_seo_test`, `historical_decomposition`, `kernel_ridge`, `mlp_regression`, `narrative_svar` … (+9) |
| randomness | `band_seed` | 4 | `lp`, `smooth_lp`, `var_forecast`, `var_irf_bands` |
| randomness | `rf_seed` | 1 | `proxy_ar_sets` |
| replication count | `n_draws` | 8 | `bvar_irf_draws`, `bvar_ssvs`, `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `robust_svar_bounds`, `sign_restricted_svar`, `zero_sign_svar` |
| replication count | `n_boot` | 7 | `conformal_backtest`, `conformal_forecast`, `hansen_seo_test`, `proxy_svar_bands`, `setar_test`, `threshold_var_test`, `var_irf_bands` |
| replication count | `max_tries` | 5 | `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| replication count | `band_n_sim` | 4 | `lp`, `smooth_lp`, `var_forecast`, `var_irf_bands` |
| replication count | `n_grid` | 3 | `bvar_hierarchical`, `hansen_seo_test`, `threshold_var_test` |
| replication count | `n_eval` | 2 | `conformal_backtest`, `conformal_forecast` |
| replication count | `n_weight_draws` | 2 | `historical_decomposition`, `narrative_svar` |
| replication count | `burn` | 1 | `bvar_ssvs` |
| replication count | `n_c` | 1 | `star` |
| replication count | `n_chains` | 1 | `bvar_ssvs` |
| replication count | `n_gamma` | 1 | `star` |
| replication count | `n_grid_beta` | 1 | `threshold_vecm` |
| replication count | `n_grid_gamma` | 1 | `threshold_vecm` |
| replication count | `n_lambdas` | 1 | `lasso_path` |
| replication count | `n_permutations` | 1 | `random_forest` |
| replication count | `n_seeds` | 1 | `mlp_regression` |
| replication count | `n_trees` | 1 | `random_forest` |
| replication count | `thin` | 1 | `bvar_ssvs` |
| confidence level | `alpha` | 24 | `adaptive_lasso`, `auto_arima`, `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `elastic_net`, `group_lasso`, `ivx_test`, `kernel_ridge`, `lasso`, `mlp_regression` … (+12) |
| confidence level | `band_alpha` | 6 | `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp`, `smooth_lp` |
| confidence level | `conf_alpha` | 2 | `arima_fit`, `auto_arima` |
| confidence level | `level` | 1 | `ou_fit` |
| penalty strength | `alpha` | 24 | `adaptive_lasso`, `auto_arima`, `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `elastic_net`, `group_lasso`, `ivx_test`, `kernel_ridge`, `lasso`, `mlp_regression` … (+12) |
| penalty strength | `lambda1` | 9 | `bvar_fit`, `bvar_irf_draws`, `fry_pagan_svar`, `historical_decomposition`, `narrative_svar`, `robust_svar_bounds`, `sign_restricted_svar`, `svensson`, `zero_sign_svar` |
| penalty strength | `l1_ratio` | 5 | `adaptive_lasso`, `elastic_net`, `group_lasso`, `lasso_path`, `post_lasso` |
| penalty strength | `lambda0` | 3 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws` |
| penalty strength | `lambda3` | 3 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws` |
| penalty strength | `lam` | 2 | `l1_trend_filter`, `smooth_lp` |
| penalty strength | `lamb` | 1 | `hp_filter` |
| penalty strength | `lambda1_hi` | 1 | `bvar_hierarchical` |
| penalty strength | `lambda1_init` | 1 | `bvar_hierarchical` |
| penalty strength | `lambda1_lo` | 1 | `bvar_hierarchical` |
| penalty strength | `mu` | 1 | `spread_zscore` |
| penalty strength | `penalty` | 1 | `l1_trend_filter` |
| penalty strength | `ridge_alpha` | 1 | `echo_state_network` |
| lag / order | `lags` | 38 | `bvar_fit`, `bvar_hierarchical`, `bvar_irf_draws`, `bvar_ssvs`, `check_series`, `conformal_backtest`, `conformal_forecast`, `connectedness`, `dcc_test`, `dfgls`, `favar`, `fry_pagan_svar` … (+26) |
| lag / order | `p` | 16 | `arima_fit`, `bn_decomposition`, `bn_filter`, `bootstrap_indices`, `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit`, `hamilton_filter`, `setar`, `setar_test`, `star` … (+4) |
| lag / order | `n_lag_controls` | 9 | `flp`, `flp_scenario`, `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `panel_lp`, `quantile_lp`, `smooth_lp` |
| lag / order | `maxlags` | 8 | `cg_regression`, `forecast_efficiency`, `hamilton_filter`, `lp`, `lp_multiplier`, `lp_state`, `ols`, `umidas` |
| lag / order | `delay` | 7 | `setar`, `setar_test`, `star`, `star_eval`, `star_test`, `threshold_var`, `threshold_var_test` |
| lag / order | `q` | 6 | `arima_fit`, `bn_decomposition`, `ccc_garch`, `dcc_garch`, `dcc_test`, `garch_fit` |
| lag / order | `d` | 5 | `arima_fit`, `auto_arima`, `frac_diff`, `frac_integrate`, `pds_lasso` |
| lag / order | `nlags` | 5 | `acf`, `arch_lm`, `kpss`, `ljung_box`, `pacf` |
| lag / order | `delays` | 4 | `setar`, `star`, `star_test`, `threshold_var` |
| lag / order | `hac_maxlags` | 4 | `flp`, `flp_scenario`, `har_rv`, `smooth_lp` |
| lag / order | `k_ar_diff` | 4 | `hansen_seo_test`, `johansen`, `threshold_vecm`, `vecm` |
| lag / order | `max_lags` | 4 | `dfgls`, `ng_perron`, `panel_unit_root`, `zivot_andrews` |
| lag / order | `order` | 4 | `conformal_backtest`, `conformal_forecast`, `l1_trend_filter`, `markov_switching_ar` |
| lag / order | `hac_lags` | 3 | `pds_lasso`, `proxy_ar_sets`, `proxy_first_stage` |
| lag / order | `max_d` | 3 | `auto_arima`, `ndiffs`, `nsdiffs` |
| lag / order | `factor_order` | 2 | `dfm_news`, `dfm_nowcast` |
| lag / order | `lrv_lags` | 2 | `cw_test`, `gw_test` |
| lag / order | `maxlag` | 2 | `adf`, `engle_granger` |
| lag / order | `D` | 1 | `auto_arima` |
| lag / order | `ar` | 1 | `bn_decomposition` |
| lag / order | `ma` | 1 | `bn_decomposition` |
| lag / order | `max_D` | 1 | `auto_arima` |
| lag / order | `max_P` | 1 | `auto_arima` |
| lag / order | `max_Q` | 1 | `auto_arima` |
| lag / order | `max_order` | 1 | `auto_arima` |
| lag / order | `max_p` | 1 | `auto_arima` |
| lag / order | `max_q` | 1 | `auto_arima` |
| horizon | `horizon` | 30 | `backtest`, `bvar_irf_draws`, `bvar_ssvs`, `conformal_backtest`, `conformal_forecast`, `connectedness`, `cv_splits`, `favar`, `fry_pagan_svar`, `fvar_scenario`, `gas_volatility`, `growth_at_risk` … (+18) |
| horizon | `horizons` | 8 | `flp`, `flp_scenario`, `lp`, `lp_iv`, `lp_multiplier`, `lp_state`, `quantile_lp`, `smooth_lp` |
| horizon | `forecast_horizon` | 3 | `ccc_garch`, `dcc_garch`, `garch_fit` |
| horizon | `forecast_steps` | 2 | `arima_fit`, `auto_arima` |
| horizon | `h` | 2 | `dm_test`, `hamilton_filter` |
| horizon | `steps` | 2 | `theta_forecast`, `var_forecast` |
| horizon | `h1` | 1 | `max_share_svar` |
| horizon | `n_steps` | 1 | `boosting` |
| horizon | `post_window` | 1 | `lp_did` |
| horizon | `pre_window` | 1 | `lp_did` |
| trend / deterministic | `trend` | 23 | `connectedness`, `engle_granger`, `favar`, `hetero_svar`, `long_run_svar`, `max_share_svar`, `mean_group_var`, `mstl`, `ng_perron`, `nongaussian_svar`, `phillips_ouliaris`, `proxy_ar_sets` … (+11) |
| trend / deterministic | `constant` | 6 | `arima_fit`, `setar`, `star`, `star_eval`, `threshold_var`, `threshold_var_test` |
| trend / deterministic | `regression` | 6 | `adf`, `dfgls`, `kpss`, `panel_unit_root`, `phillips_perron`, `zivot_andrews` |
| trend / deterministic | `drift` | 2 | `bn_decomposition`, `cf_filter` |
| trend / deterministic | `deterministic` | 1 | `vecm` |
| trend / deterministic | `first_season` | 1 | `vecm` |
| trend / deterministic | `intercept` | 1 | `ar_loglik` |
| trend / deterministic | `seasons` | 1 | `vecm` |
| standard-error type | `bandwidth` | 6 | `iv_gmm`, `kernel_regression`, `long_run_variance`, `panel_fe`, `panel_lp`, `phillips_ouliaris` |
| standard-error type | `use_correction` | 5 | `cg_regression`, `forecast_efficiency`, `hamilton_filter`, `har_rv`, `ols` |
| standard-error type | `se` | 4 | `hamilton_filter`, `lp`, `lp_state`, `quantile_regression` |
| standard-error type | `se_type` | 4 | `ols`, `panel_fe`, `panel_lp`, `umidas` |
| standard-error type | `kernel` | 3 | `kernel_regression`, `kernel_ridge`, `long_run_variance` |
| standard-error type | `robust` | 2 | `mstl`, `stl` |
| tolerance / iterations | `max_iter` | 13 | `adaptive_lasso`, `bvar_hierarchical`, `elastic_net`, `group_lasso`, `iv_gmm`, `l1_trend_filter`, `lasso`, `lasso_path`, `markov_switching_ar`, `nongaussian_svar`, `panel_pmg`, `pds_lasso` … (+1) |
| tolerance / iterations | `tol` | 13 | `adaptive_lasso`, `bvar_hierarchical`, `elastic_net`, `group_lasso`, `iv_gmm`, `l1_trend_filter`, `lasso`, `lasso_path`, `markov_switching_ar`, `nongaussian_svar`, `panel_pmg`, `pds_lasso` … (+1) |
| tolerance / iterations | `inner_iter` | 2 | `mstl`, `stl` |
| tolerance / iterations | `outer_iter` | 2 | `mstl`, `stl` |
| tolerance / iterations | `max_epochs` | 1 | `mlp_regression` |
| tolerance / iterations | `patience` | 1 | `mlp_regression` |
| prediction input | `x_test` | 7 | `boosting`, `echo_state_network`, `kernel_regression`, `kernel_ridge`, `mlp_regression`, `random_forest`, `regression_tree` |
| prediction input | `test` | 3 | `heteroskedasticity_test`, `ndiffs`, `panel_unit_root` |
| train/test split | `scheme` | 4 | `bootstrap_indices`, `cv_splits`, `midas_weights`, `weighted_midas` |
| train/test split | `window` | 4 | `backtest`, `coherence`, `periodogram`, `welch` |
| train/test split | `n_eval` | 2 | `conformal_backtest`, `conformal_forecast` |
| train/test split | `train` | 2 | `backtest`, `cv_splits` |
| train/test split | `validation_fraction` | 1 | `mlp_regression` |


**Reading the table.** Dominant spelling, outliers, and whether the outlier
is justified:

| concept | dominant | outliers | justified? |
|---|---|---|---|
| randomness | `seed` (21) | `band_seed` (4, default 20260807), `rf_seed` (1) | `band_seed` is a second stream in functions that also bootstrap — the concept is real, the magic default is not |
| replication count | `n_draws` (8, posterior/rotation draws), `n_boot` (7, bootstrap) | `band_n_sim` (4), `n_weight_draws` (2), `max_tries` (5), `n_permutations`, `n_seeds`, `n_trees` (1 each) | `n_draws` vs `n_boot` is the literature's own split (posterior draws vs bootstrap replications); `band_n_sim` is a third word for a draw count |
| confidence level | `alpha` = miscoverage (13) | `band_alpha` (6), `conf_alpha` (2), `level` (1, *confidence* orientation), `alpha` = IVX exponent (2), `alpha` = penalty (9) | penalty `alpha` is scikit-learn's and justified; the IVX exponent is not; `level` flips orientation |
| lag / order | `lags` (38), `p` (16) | `maxlags` (8) / `max_lags` (4) / `maxlag` (2); `nlags` (5); `hac_maxlags` (4) / `hac_lags` (3) / `lrv_lags` (2); `k_ar_diff` (4, statsmodels VECM) | `p`/`q`/`d` are the ARIMA literature; the three max-lag spellings are drift |
| horizon | `horizon` (30) | `horizons` (8), `forecast_horizon` (3), `forecast_steps` (2), `steps` (2), `h` (2) | `h1`, `pre_window`/`post_window` are different concepts; the rest are drift |
| trend / deterministic | `trend` (23) | `regression` (6, unit-root tests: statsmodels' word), `constant` (6), `deterministic` (1, VECM), `drift`, `intercept` | `regression` is the statsmodels/ADF heritage and justified; `constant` vs `trend` is drift |
| standard-error type | `se_type` (4), `se` (4) | `robust` (2), `bandwidth` (6) / `kernel` (3) / `use_correction` (5) for HAC | two spellings for the same selector (`se` in LP, `se_type` in OLS/panel/MIDAS) |
| tolerance / iterations | `tol` (13), `max_iter` (13) | `inner_iter`/`outer_iter` (STL), `max_epochs`/`patience` (MLP) | all justified (STL's and the MLP's own vocabulary) |
| method selector | `method` (14) | `kind`, `mode`, `model`, `test`, `test_type`, `variant`, `dist`, `scheme`, `ic`, `family`, `stop`, `forecaster`, `base`, `solver`, `activation`, `identification`, `bootstrap`, `importance`, `vol`, `mean` | each names a *different* selector; only `test`/`test_type` and `kind`/`method` overlap |
| prediction input | `x_test` (7) | — | consistent (scikit-learn's spelling) |
| multivariate input shape | `(T, k)` (81) | `(N, T)` (5 panel functions), list of per-unit arrays (3+1) | the panel flip is documented per function; see F14 |

## Return-key cluster tables

| concept | spelling | n functions | functions |
|---|---|---|---|
| standard errors | `se` | 16 | `copula_fit`, `flp`, `flp_scenario`, `long_memory_d`, `lp`, `lp_did`, `lp_iv`, `lp_multiplier`, `panel_lp`, `panel_mean_group`, `pds_lasso`, `proxy_first_stage` … (+4) |
| standard errors | `bse` | 12 | `arima_fit`, `auto_arima`, `bai_perron`, `forecast_efficiency`, `growth_at_risk`, `har_rv`, `iv_gmm`, `ols`, `panel_fe`, `quantile_regression`, `recession_probit`, `umidas` |
| standard errors | `se_valid` | 8 | `arima_fit`, `auto_arima`, `copula_fit`, `garch_fit`, `gev_fit`, `gpd_fit`, `star`, `star_eval` |
| standard errors | `se_type` | 4 | `lp_did`, `ols`, `panel_fe`, `panel_lp` |
| standard errors | `bse_high` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| standard errors | `bse_low` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| standard errors | `bse_linear` | 2 | `star`, `star_eval` |
| standard errors | `bse_nonlinear` | 2 | `star`, `star_eval` |
| standard errors | `kappa_se` | 2 | `dcs_local_level`, `ou_fit` |
| standard errors | `se_c` | 2 | `star`, `star_eval` |
| standard errors | `se_gamma` | 2 | `star`, `star_eval` |
| standard errors | `se_method` | 2 | `lp`, `lp_state` |
| standard errors | `se_xi` | 2 | `gev_fit`, `gpd_fit` |
| standard errors | `bartlett_se` | 1 | `acf` |
| standard errors | `bse_powell` | 1 | `growth_at_risk` |
| standard errors | `c_se` | 1 | `ou_fit` |
| standard errors | `coefs_se` | 1 | `mean_group_var` |
| standard errors | `cycle_se` | 1 | `bn_filter` |
| standard errors | `forecast_se` | 1 | `arima_fit` |
| standard errors | `intercept_se` | 1 | `mean_group_var` |
| standard errors | `irf_path_se` | 1 | `mean_group_var` |
| standard errors | `mu_se` | 1 | `ou_fit` |
| standard errors | `nu_se` | 1 | `dcs_local_level` |
| standard errors | `orth_irfs_se` | 1 | `mean_group_var` |
| standard errors | `phi_se` | 1 | `ou_fit` |
| standard errors | `scale_se` | 1 | `dcs_local_level` |
| standard errors | `se_asymptotic` | 1 | `long_memory_d` |
| standard errors | `se_beta` | 1 | `gpd_fit` |
| standard errors | `se_intercept` | 1 | `cg_regression` |
| standard errors | `se_mle` | 1 | `garch_fit` |
| standard errors | `se_mu` | 1 | `gev_fit` |
| standard errors | `se_raw` | 1 | `smooth_lp` |
| standard errors | `se_regression` | 1 | `long_memory_d` |
| standard errors | `se_rho` | 1 | `copula_fit` |
| standard errors | `se_robust` | 1 | `garch_fit` |
| standard errors | `se_sigma` | 1 | `gev_fit` |
| standard errors | `se_slope` | 1 | `cg_regression` |
| standard errors | `se_state0` | 1 | `lp_state` |
| standard errors | `se_state1` | 1 | `lp_state` |
| standard errors | `sigma_se` | 1 | `ou_fit` |
| standard errors | `theta_se` | 1 | `panel_pmg` |
| p-values | `p_value` | 16 | `adf`, `arch_lm`, `cw_test`, `dcc_test`, `dfgls`, `dm_test`, `gw_test`, `hansen_seo_test`, `jarque_bera`, `kpss`, `panel_unit_root`, `pds_lasso` … (+4) |
| p-values | `pvalue` | 8 | `chow_test`, `engle_granger`, `heteroskedasticity_test`, `ivx_test`, `phillips_ouliaris`, `phillips_perron`, `reset_test`, `zivot_andrews` |
| p-values | `adf_p_value` | 1 | `check_stationarity` |
| p-values | `bp_pvalue` | 1 | `ljung_box` |
| p-values | `f_pvalue` | 1 | `heteroskedasticity_test` |
| p-values | `h1_p_value` | 1 | `star_test` |
| p-values | `h2_p_value` | 1 | `star_test` |
| p-values | `h3_p_value` | 1 | `star_test` |
| p-values | `kpss_p_value` | 1 | `check_stationarity` |
| p-values | `lb_pvalue` | 1 | `ljung_box` |
| p-values | `lm3_f_p_value` | 1 | `star_test` |
| p-values | `lm3_p_value` | 1 | `star_test` |
| p-values | `p` | 1 | `dsge_solve` |
| p-values | `p_cc` | 1 | `var_backtest` |
| p-values | `p_dq` | 1 | `var_backtest` |
| p-values | `p_ind` | 1 | `var_backtest` |
| p-values | `p_slope` | 1 | `cg_regression` |
| p-values | `p_tail` | 1 | `gpd_fit` |
| p-values | `p_uc` | 1 | `var_backtest` |
| p-values | `per_unit_pvalue` | 1 | `panel_unit_root` |
| p-values | `pvalues` | 1 | `forecast_efficiency` |
| p-values | `wald_pvalue` | 1 | `forecast_efficiency` |
| coefficients | `params` | 17 | `arima_fit`, `auto_arima`, `bai_perron`, `copula_fit`, `favar`, `forecast_efficiency`, `garch_fit`, `gmm_nonlinear`, `growth_at_risk`, `har_rv`, `iv_gmm`, `ols` … (+5) |
| coefficients | `alpha` | 11 | `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `ndiffs`, `nsdiffs`, `proxy_svar_bands`, `robust_svar_bounds`, `var_backtest`, `var_irf_bands`, `vecm` |
| coefficients | `coef` | 8 | `adaptive_lasso`, `boosting`, `elastic_net`, `group_lasso`, `lasso`, `lp_did`, `panel_mean_group`, `pds_lasso` |
| coefficients | `weights` | 8 | `flp_scenario`, `fvar_scenario`, `mlp_regression`, `mstl`, `narrative_svar`, `stl`, `weighted_midas`, `zero_sign_svar` |
| coefficients | `beta` | 7 | `acm_term_premium`, `gpd_fit`, `hamilton_filter`, `hansen_seo_test`, `proxy_first_stage`, `threshold_vecm`, `vecm` |
| coefficients | `a` | 4 | `acm_term_premium`, `dcc_garch`, `gas_volatility`, `summarize` |
| coefficients | `param_names` | 4 | `arima_fit`, `auto_arima`, `copula_fit`, `garch_fit` |
| coefficients | `ar` | 3 | `bn_decomposition`, `bn_filter`, `markov_switching_ar` |
| coefficients | `b` | 3 | `dcc_garch`, `gas_volatility`, `summarize` |
| coefficients | `gamma` | 3 | `kernel_ridge`, `star`, `vecm` |
| coefficients | `params_high` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| coefficients | `params_low` | 3 | `setar`, `threshold_var`, `threshold_vecm` |
| coefficients | `phi` | 3 | `acm_term_premium`, `ou_fit`, `panel_pmg` |
| coefficients | `B` | 2 | `acm_term_premium`, `hetero_svar` |
| coefficients | `betas` | 2 | `flp`, `flp_scenario` |
| coefficients | `coefs` | 2 | `lasso_path`, `mean_group_var` |
| coefficients | `loadings` | 2 | `dfm_nowcast`, `factor_model` |
| coefficients | `params_linear` | 2 | `star`, `star_eval` |
| coefficients | `params_nonlinear` | 2 | `star`, `star_eval` |
| coefficients | `posterior_mean_coefs` | 2 | `bvar_fit`, `bvar_hierarchical` |
| coefficients | `theta` | 2 | `panel_pmg`, `smooth_lp` |
| coefficients | `beta_grid` | 1 | `threshold_vecm` |
| coefficients | `beta_ivx` | 1 | `ivx_test` |
| coefficients | `beta_linear` | 1 | `threshold_vecm` |
| coefficients | `coef_lasso` | 1 | `post_lasso` |
| coefficients | `coef_mean` | 1 | `bvar_ssvs` |
| coefficients | `coef_ols` | 1 | `post_lasso` |
| coefficients | `coef_path` | 1 | `boosting` |
| coefficients | `coef_per_unit` | 1 | `panel_mean_group` |
| coefficients | `coint_coefs` | 1 | `engle_granger` |
| coefficients | `delta` | 1 | `bn_filter` |
| coefficients | `det_coef` | 1 | `vecm` |
| coefficients | `dual_coef` | 1 | `kernel_ridge` |
| coefficients | `factor_loadings` | 1 | `acm_term_premium` |
| coefficients | `ma` | 1 | `bn_decomposition` |
| coefficients | `omega` | 1 | `gas_volatility` |
| coefficients | `params_named` | 1 | `garch_fit` |
| coefficients | `readout` | 1 | `echo_state_network` |
| fitted values | `fitted` | 12 | `acm_term_premium`, `boosting`, `echo_state_network`, `growth_at_risk`, `kernel_regression`, `kernel_ridge`, `mlp_regression`, `random_forest`, `regression_tree`, `spread_zscore`, `var_fit`, `weighted_midas` |
| fitted values | `trend` | 10 | `bn_decomposition`, `bn_filter`, `cf_filter`, `dfgls`, `hamilton_filter`, `hp_filter`, `l1_trend_filter`, `mstl`, `ng_perron`, `stl` |
| fitted values | `cycle` | 7 | `bk_filter`, `bn_decomposition`, `bn_filter`, `cf_filter`, `hamilton_filter`, `hp_filter`, `l1_trend_filter` |
| fitted values | `predicted` | 5 | `boosting`, `echo_state_network`, `mlp_regression`, `random_forest`, `regression_tree` |
| fitted values | `point` | 3 | `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| fitted values | `forecast` | 2 | `dynamic_ns`, `gas_volatility` |
| fitted values | `mean` | 2 | `conformal_backtest`, `conformal_forecast` |
| fitted values | `filtered_prob` | 1 | `markov_switching_ar` |
| fitted values | `filtered_state` | 1 | `local_level_smooth` |
| fitted values | `filtered_state_var` | 1 | `local_level_smooth` |
| fitted values | `forecasts` | 1 | `backtest` |
| fitted values | `nowcast` | 1 | `dfm_nowcast` |
| fitted values | `smoothed_factors` | 1 | `dfm_nowcast` |
| fitted values | `smoothed_prob` | 1 | `markov_switching_ar` |
| fitted values | `smoothed_prob_last_regime` | 1 | `markov_switching_ar` |
| fitted values | `smoothed_state` | 1 | `local_level_smooth` |
| fitted values | `smoothed_state_var` | 1 | `local_level_smooth` |
| residuals | `residuals` | 6 | `arima_fit`, `auto_arima`, `iv_gmm`, `nelson_siegel`, `svensson`, `weighted_midas` |
| residuals | `resid` | 5 | `dcs_local_level`, `engle_granger`, `mstl`, `stl`, `var_fit` |
| residuals | `innovations` | 1 | `bn_decomposition` |
| residuals | `std_resid` | 1 | `gas_volatility` |
| log-likelihood | `loglik` | 19 | `arima_fit`, `auto_arima`, `bn_decomposition`, `ccc_garch`, `copula_fit`, `dcc_garch`, `dcs_local_level`, `dfm_nowcast`, `garch_fit`, `gas_volatility`, `gev_fit`, `gpd_fit` … (+7) |
| log-likelihood | `llf` | 4 | `threshold_var`, `threshold_vecm`, `var_fit`, `vecm` |
| information criteria | `aic` | 13 | `arima_fit`, `auto_arima`, `bn_decomposition`, `copula_fit`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `lasso_path`, `setar`, `star`, `star_eval`, `threshold_var` … (+1) |
| information criteria | `bic` | 13 | `arima_fit`, `auto_arima`, `bn_decomposition`, `copula_fit`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `lasso_path`, `setar`, `star`, `star_eval`, `threshold_var` … (+1) |
| information criteria | `ic` | 2 | `auto_arima`, `setar` |
| information criteria | `aic_path` | 1 | `boosting` |
| information criteria | `aicc` | 1 | `auto_arima` |
| information criteria | `hqic` | 1 | `var_fit` |
| convergence | `converged` | 23 | `arima_fit`, `auto_arima`, `bn_decomposition`, `bvar_hierarchical`, `copula_fit`, `dcc_garch`, `dcs_local_level`, `garch_fit`, `gas_volatility`, `gev_fit`, `gmm_nonlinear`, `gpd_fit` … (+11) |
| convergence | `iterations` | 7 | `dcs_local_level`, `gas_volatility`, `gmm_nonlinear`, `markov_switching_ar`, `panel_pmg`, `quantile_regression`, `weighted_midas` |
| convergence | `n_iter` | 6 | `adaptive_lasso`, `elastic_net`, `group_lasso`, `l1_trend_filter`, `lasso`, `nongaussian_svar` |
| convergence | `boundary` | 3 | `arima_fit`, `auto_arima`, `garch_fit` |
| convergence | `boundary_note` | 3 | `arima_fit`, `auto_arima`, `garch_fit` |
| convergence | `cov_ok` | 2 | `arima_fit`, `auto_arima` |
| convergence | `fevals` | 2 | `gmm_nonlinear`, `star` |
| convergence | `at_bound` | 1 | `box_cox_lambda` |
| convergence | `bandwidth_at_boundary` | 1 | `kernel_regression` |
| convergence | `best_epoch` | 1 | `mlp_regression` |
| convergence | `budget_exhausted` | 1 | `auto_arima` |
| convergence | `n_accepted` | 1 | `fry_pagan_svar` |
| convergence | `n_criterion_evaluations` | 1 | `kernel_regression` |
| convergence | `n_evals` | 1 | `bvar_hierarchical` |
| confidence intervals | `alpha` | 11 | `check_series`, `check_stationarity`, `conformal_backtest`, `conformal_forecast`, `ndiffs`, `nsdiffs`, `proxy_svar_bands`, `robust_svar_bounds`, `var_backtest`, `var_irf_bands`, `vecm` |
| confidence intervals | `lower` | 6 | `box_cox_lambda`, `conformal_backtest`, `conformal_forecast`, `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| confidence intervals | `upper` | 6 | `box_cox_lambda`, `conformal_backtest`, `conformal_forecast`, `proxy_svar_bands`, `var_forecast`, `var_irf_bands` |
| confidence intervals | `quantiles` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `set_max` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `set_min` | 3 | `narrative_svar`, `sign_restricted_svar`, `zero_sign_svar` |
| confidence intervals | `band` | 2 | `var_forecast`, `var_irf_bands` |
| confidence intervals | `bound_lower` | 1 | `cusum_test` |
| confidence intervals | `bound_upper` | 1 | `cusum_test` |
| confidence intervals | `ci_lower_90` | 1 | `bai_perron` |
| confidence intervals | `ci_lower_95` | 1 | `bai_perron` |
| confidence intervals | `ci_scale` | 1 | `bai_perron` |
| confidence intervals | `ci_upper_90` | 1 | `bai_perron` |
| confidence intervals | `ci_upper_95` | 1 | `bai_perron` |
| confidence intervals | `conf_alpha` | 1 | `arima_fit` |
| confidence intervals | `conf_int` | 1 | `pds_lasso` |
| confidence intervals | `forecast_lower` | 1 | `arima_fit` |
| confidence intervals | `forecast_upper` | 1 | `arima_fit` |
| confidence intervals | `half_life_ci` | 1 | `ou_fit` |
| confidence intervals | `lower_efron` | 1 | `proxy_svar_bands` |
| confidence intervals | `lower_quantiles` | 1 | `robust_svar_bounds` |
| confidence intervals | `q_lower` | 1 | `conformal_forecast` |
| confidence intervals | `q_upper` | 1 | `conformal_forecast` |
| confidence intervals | `robust_ci_lower` | 1 | `robust_svar_bounds` |
| confidence intervals | `robust_ci_upper` | 1 | `robust_svar_bounds` |
| confidence intervals | `tail_lower` | 1 | `copula_fit` |
| confidence intervals | `tail_upper` | 1 | `copula_fit` |
| confidence intervals | `upper_efron` | 1 | `proxy_svar_bands` |
| confidence intervals | `upper_quantiles` | 1 | `robust_svar_bounds` |
| R-squared | `rsquared` | 6 | `dynamic_ns`, `har_rv`, `nelson_siegel`, `svensson`, `umidas`, `weighted_midas` |
| R-squared | `r_squared` | 2 | `cg_regression`, `forecast_efficiency` |
| R-squared | `rx_rsquared` | 1 | `acm_term_premium` |
| R-squared | `short_rate_rsquared` | 1 | `acm_term_premium` |
| R-squared | `var_rsquared` | 1 | `acm_term_premium` |
| R-squared | `yield_rsquared` | 1 | `acm_term_premium` |
| sample-size echo | `nobs` | 27 | `adf`, `arch_lm`, `cg_regression`, `dcc_test`, `dfgls`, `engle_granger`, `flp`, `hansen_seo_test`, `har_rv`, `iv_gmm`, `ivx_test`, `lp_did` … (+15) |
| sample-size echo | `n` | 6 | `box_cox_lambda`, `check_series`, `copula_fit`, `gpd_fit`, `jarque_bera`, `var_backtest` |
| sample-size echo | `neqs` | 5 | `hansen_seo_test`, `mean_group_var`, `threshold_var`, `threshold_var_test`, `threshold_vecm` |
| sample-size echo | `n_factors` | 4 | `acm_term_premium`, `dfm_nowcast`, `favar`, `flp` |
| sample-size echo | `n_proxy` | 4 | `proxy_ar_sets`, `proxy_first_stage`, `proxy_svar`, `proxy_svar_bands` |
| sample-size echo | `n_regressors` | 4 | `hansen_seo_test`, `threshold_var`, `threshold_var_test`, `threshold_vecm` |
| sample-size echo | `n_units` | 3 | `panel_mean_group`, `panel_pmg`, `panel_unit_root` |
| sample-size echo | `n_vars` | 3 | `engle_granger`, `hetero_svar`, `phillips_ouliaris` |
| sample-size echo | `n_obs` | 2 | `dcs_local_level`, `ou_fit` |
| sample-size echo | `n_train` | 2 | `echo_state_network`, `mlp_regression` |
| sample-size echo | `adf_nobs` | 1 | `engle_granger` |
| sample-size echo | `n_breaks` | 1 | `bai_perron` |
| sample-size echo | `n_calib` | 1 | `conformal_forecast` |
| sample-size echo | `n_controls_selected` | 1 | `pds_lasso` |
| sample-size echo | `n_endog` | 1 | `favar` |
| sample-size echo | `n_eval` | 1 | `conformal_backtest` |
| sample-size echo | `n_knots` | 1 | `l1_trend_filter` |
| sample-size echo | `n_maxima` | 1 | `gev_fit` |
| sample-size echo | `n_models` | 1 | `auto_arima` |
| sample-size echo | `n_origins` | 1 | `backtest` |
| sample-size echo | `n_parameters` | 1 | `mlp_regression` |
| sample-size echo | `n_stacked` | 1 | `dcc_test` |
| sample-size echo | `n_used` | 1 | `proxy_svar_bands` |
| sample-size echo | `n_validation` | 1 | `mlp_regression` |
| sample-size echo | `n_washout` | 1 | `echo_state_network` |
| sample-size echo | `nobs_per_h` | 1 | `lp_multiplier` |
| lag echo | `lags` | 8 | `dcc_test`, `hetero_svar`, `kpss`, `ljung_box`, `mean_group_var`, `phillips_ouliaris`, `phillips_perron`, `zivot_andrews` |
| lag echo | `delay` | 6 | `setar`, `setar_test`, `star`, `star_test`, `threshold_var`, `threshold_var_test` |
| lag echo | `used_lag` | 4 | `adf`, `dfgls`, `engle_granger`, `ng_perron` |
| lag echo | `hac_lags` | 2 | `growth_at_risk`, `proxy_first_stage` |
| lag echo | `k_ar_diff` | 2 | `hansen_seo_test`, `threshold_vecm` |
| lag echo | `max_d` | 2 | `ndiffs`, `nsdiffs` |
| lag echo | `order` | 2 | `auto_arima`, `nongaussian_svar` |
| lag echo | `factor_order` | 1 | `dfm_nowcast` |
| lag echo | `hac_lags_resolved` | 1 | `pds_lasso` |
| lag echo | `maxlags` | 1 | `cg_regression` |
| lag echo | `seasonal_order` | 1 | `auto_arima` |
| variance / covariance | `sigma2` | 8 | `acm_term_premium`, `bn_decomposition`, `ccc_garch`, `dcc_garch`, `panel_pmg`, `setar`, `star`, `star_eval` |
| variance / covariance | `sigma` | 7 | `acm_term_premium`, `cusum_test`, `gev_fit`, `ou_fit`, `spread_zscore`, `threshold_var`, `threshold_vecm` |
| variance / covariance | `sigma_u` | 3 | `favar`, `var_fit`, `vecm` |
| variance / covariance | `variance_forecast` | 3 | `ccc_garch`, `dcc_garch`, `garch_fit` |
| variance / covariance | `correlation` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `covariance` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `covariance_forecast` | 2 | `ccc_garch`, `dcc_garch` |
| variance / covariance | `param_cov` | 2 | `arima_fit`, `auto_arima` |
| variance / covariance | `scale` | 2 | `dcs_local_level`, `dfm_nowcast` |
| variance / covariance | `sigma_posterior_mean` | 2 | `bvar_fit`, `bvar_hierarchical` |
| variance / covariance | `correlation_forecast` | 1 | `dcc_garch` |
| variance / covariance | `correlation_last` | 1 | `dcc_garch` |
| variance / covariance | `cov` | 1 | `iv_gmm` |
| variance / covariance | `covs` | 1 | `flp` |
| variance / covariance | `factor_cov` | 1 | `dfm_nowcast` |
| variance / covariance | `omega_bar` | 1 | `bvar_fit` |
| variance / covariance | `qbar` | 1 | `dcc_garch` |
| variance / covariance | `s_bar` | 1 | `bvar_fit` |
| variance / covariance | `sigma2_high` | 1 | `setar` |
| variance / covariance | `sigma2_low` | 1 | `setar` |
| variance / covariance | `sigma_high` | 1 | `threshold_var` |
| variance / covariance | `sigma_low` | 1 | `threshold_var` |
| variance / covariance | `sigma_mean` | 1 | `bvar_ssvs` |
| variance / covariance | `sigma_regime1` | 1 | `hetero_svar` |
| variance / covariance | `sigma_regime2` | 1 | `hetero_svar` |
| variance / covariance | `sigma_se` | 1 | `ou_fit` |
| variance / covariance | `variance` | 1 | `gas_volatility` |
| variance / covariance | `variances` | 1 | `markov_switching_ar` |
| test statistic | `stat` | 9 | `dcc_test`, `engle_granger`, `hansen_seo_test`, `phillips_ouliaris`, `phillips_perron`, `setar_test`, `sup_f_test`, `threshold_var_test`, `zivot_andrews` |
| test statistic | `statistic` | 8 | `adf`, `arch_lm`, `dfgls`, `heteroskedasticity_test`, `jarque_bera`, `kpss`, `panel_unit_root`, `var_granger` |
| test statistic | `tvalues` | 5 | `forecast_efficiency`, `har_rv`, `ols`, `panel_fe`, `quantile_regression` |
| test statistic | `fstat` | 3 | `chow_test`, `heteroskedasticity_test`, `reset_test` |
| test statistic | `q` | 3 | `dsge_solve`, `max_share_svar`, `star_test` |
| test statistic | `wald` | 2 | `forecast_efficiency`, `ivx_test` |
| test statistic | `adf_statistic` | 1 | `check_stationarity` |
| test statistic | `ar_bound_stat` | 1 | `proxy_ar_sets` |
| test statistic | `bp_stat` | 1 | `ljung_box` |
| test statistic | `cw_stat` | 1 | `cw_test` |
| test statistic | `dm_stat` | 1 | `dm_test` |
| test statistic | `dq_stat` | 1 | `var_backtest` |
| test statistic | `gw_stat` | 1 | `gw_test` |
| test statistic | `h1_f_stat` | 1 | `star_test` |
| test statistic | `h2_f_stat` | 1 | `star_test` |
| test statistic | `h3_f_stat` | 1 | `star_test` |
| test statistic | `hln_stat` | 1 | `dm_test` |
| test statistic | `j_stat` | 1 | `iv_gmm` |
| test statistic | `kpss_statistic` | 1 | `check_stationarity` |
| test statistic | `lb_stat` | 1 | `ljung_box` |
| test statistic | `lm3_f_stat` | 1 | `star_test` |
| test statistic | `lm3_stat` | 1 | `star_test` |
| test statistic | `max_eig_stat` | 1 | `johansen` |
| test statistic | `mt_statistic` | 1 | `fry_pagan_svar` |
| test statistic | `t_stat` | 1 | `pds_lasso` |
| test statistic | `trace_stat` | 1 | `johansen` |
| test statistic | `tstat` | 1 | `panel_mean_group` |
| critical values | `crit` | 7 | `adf`, `dfgls`, `engle_granger`, `ng_perron`, `phillips_ouliaris`, `phillips_perron`, `zivot_andrews` |
| critical values | `sup_f_crit` | 1 | `bai_perron` |


(In the standard-error rows, `se_valid`, `se_method` and `se_type` are flags and selectors caught by the `se_*` pattern, not standard errors.)

## Compliance matrix summary

| probe | ValueError | TypeError | other Python exc | PanicException | silent return | n/a | names the argument | states a fix |
|---|---|---|---|---|---|---|---|---|
| nan | 164 | 0 | 0 | 0 | 4 | 5 | 103/164 | 123/164 |
| empty | 168 | 0 | 0 | 0 | 0 | 5 | 64/168 | 148/168 |
| ndim | 1 | 164 | 0 | 0 | 3 | 5 | 26/165 | 164/165 |
| string | 87 | 0 | 0 | 0 | 1 | 85 | 87/87 | 86/87 |
| negative | 131 | 0 | 0 | 0 | 0 | 42 | 131/131 | 131/131 |

Per-function verdict over 173 callables: full 156, partial 16, none 1, n/a 0.

Panic escapes: 0

Silent returns on **nan** (4): `ar_loglik` (finite out), `dfm_news` (finite out), `dfm_nowcast` (finite out), `local_level_smooth` (finite out)

Silent returns on **ndim** (3): `check_series` (finite out), `kernel_regression` (finite out), `kernel_ridge` (finite out)

Silent returns on **string** (1): `summarize` (finite out)

Per-function cells for the 17 partial/non-compliant callables: `lab/audit/repo/api/out/compliance.md`.

Interpretation of the eight silent returns: `dfm_nowcast`/`dfm_news` take
NaN by design (the ragged edge), `local_level_smooth` documents NaN as
missing, `ar_loglik` did not (F3, fixed); `kernel_ridge`/`kernel_regression`
document that `x` accepts 1-D for a single regressor and `check_series`
accepts an `(n, 1)` panel; `summarize`'s only string argument is a title.
The 16 "partial" verdicts are all messages that state the problem in
words without the parameter's name (F10) or blame the wrong argument
(F11); none is a raw PyO3 message except `gmm_nonlinear`'s rank case.

## Rename proposal

The pre-1.0 policy (ROADMAP §7, "minor = breaking allowed"; Module 11's
"two-release deprecation cycles … CI fails if a public symbol disappears
without a deprecation shim") allows: add the canonical name, accept the
old spelling as an alias that emits a `DeprecationWarning` naming the new
one, remove the alias at 1.0. For parameters this is a keyword alias in
the PyO3 signature (`#[pyo3(signature = (..., max_lags=None, maxlags=None))]`
with a one-line resolver that warns); for returned keys it is a second
key holding the same object (dicts, so aliasing is free) with the
deprecation stated in the docstring and a warning on `summarize`. Blast
radius = functions using the old spelling / textual mentions in
docs + tests + examples (from `probe_blast.py`; `docs` excludes `api.md`
and the roadmap).

| # | concept | canonical | deprecate → alias | functions | mentions (docs / tests / examples) | why this order |
|---|---|---|---|---|---|---|
| 1 | confidence level | `alpha` = miscoverage (0.05 / 0.1) | `band_alpha` (6), `conf_alpha` (2), `level` (1); rename the IVX exponent `alpha` → `ivx_beta` (2) | 11 | 8 + 10 + 375 (`level` is mostly prose) / 6 + 16 + 101 / 0 + 4 + 124 | the only cluster where the *same word* silently means opposite things (F2) |
| 2 | maximum lag | `max_lags` | `maxlags` (8), `maxlag` (2), `nlags` (5), `hac_maxlags` (4) → `hac_lags`, `lrv_lags` (2) → `hac_lags` | 21 | 51 + 7 + 64 + 5 + 4 / 29 + 4 + 11 + 1 + 3 / 30 + 0 + 11 + 2 + 0 | three spellings of one word; `hac_lags` already exists on 3 functions and 2 returned keys |
| 3 | horizon | `horizon` | `forecast_horizon` (3), `forecast_steps` (2), `steps` (2), `h` (2); keep `horizons` in the LP family (documented as the maximum) or alias it too | 9 (17 with `horizons`) | 13 + 17 + 66 + 1018† / 27 + 29 + 46 / 4 + 5 + 12 | † `h` is unsearchable in prose; the count is every `h`, not the parameter |
| 4 | standard errors (keys) | `se` | `bse` (12), `bse_high`/`bse_low`/`bse_linear`/`bse_nonlinear`/`bse_powell` (8) → `se_*` | 20 | 109 / 63 / 59 | one key rename covers the statsmodels-heritage families at once |
| 5 | p-values (keys) | `p_value` | `pvalue` (8), `pvalues` (1), `*_pvalue` (5) → `*_p_value` | 14 | 54 / 47 / 1 | second most-read key after the statistic |
| 6 | test statistic (keys) | `statistic` | `stat` (9), `*_stat` (17 keys) → `*_statistic`; `tvalues` (5), `tstat`, `t_stat` → `t_statistic` | 26 | 109 / 41 / 5 | pairs with #5 |
| 7 | coefficients (keys) | `params` (the settled majority, 17) | `coef` (8) / `coefs` (2) — *or* the reverse; pick one | 10 | 50 + 21 / 59 + 15 / 9 + 56 | scikit-learn users expect `coef`, statsmodels users `params`; either is defensible, the split is not |
| 8 | log-likelihood (keys) | `loglik` | `llf` (4) | 4 | 21 / 25 / 1 | cheap |
| 9 | residuals (keys) | `residuals` | `resid` (5) | 5 | 44 / 48 / 20 | cheap |
| 10 | iterations (keys) | `n_iter` | `iterations` (7) | 7 | 21 / 11 / 0 | cheap; matches `n_boot`/`n_draws` style |
| 11 | R² (keys) | `rsquared` | `r_squared` (2) | 2 | 2 / 0 / 0 | cheap |
| 12 | SE selector | `se_type` | `se` as a *parameter* (4, LP family) | 4 | — | frees `se` to mean only the returned key |
| 13 | seed | `seed` | `band_seed` (4) → `seed` where no second stream exists, `band_seed` documented as the sup-t stream where it does; replace `20260807` with a documented constant | 4 | 7 / 17 / 4 | F6 |
| 14 | trend | `trend` | `constant` (6) → `trend="c"`-style; keep `regression` on the unit-root tests (statsmodels) | 6 | — | low |
| 15 | matrix keys as arrays | `into_pyarray` | nested lists (54 functions, 197 keys) | 54 | — | a type change, not a rename; needs its own note (F1) |

Not proposed: `n_draws` vs `n_boot` (posterior draws vs bootstrap
replications are different objects and the literature keeps them apart);
penalty `alpha` (scikit-learn's word; the ML family is documented as
matching scikit-learn's objective); `p`/`q`/`d`; `regression` on the ADF
family; `h1`, `pre_window`/`post_window`; `set_min`/`set_max`.

## Clean bills

- **No panic escape on any of the 865 malformed calls**, no subprocess
  crash, no timeout — every failure is a Python exception a user can catch.
- **No silent wrong-type return**: the eight silent returns are all
  finite, and seven are documented design (the eighth, `ar_loglik`, is now
  documented).
- **Negative counts and unknown strings are uniform**: 131/131 negative
  messages name the parameter (the central wrapper); 87/87 unknown-string
  messages name the parameter and 86/87 list the accepted values.
- **Signature order**: the stub's parameter order matches the compiled
  signature on all 173; no stub return annotation disagrees with the
  returned kind.
- **The eleven 0.8.0 callables are the most consistent family on the
  surface**: every returned key backticked, every parameter named with its
  default, a citation and a grade in every docstring, `x_test` and `seed`
  spelled the same way in all of them, the `unknown … expected …` string
  refusal on every selector.
- **`check_series`, `summarize`** (the two pure-Python entry points) pass
  every applicable probe.

## Open

- **F1's code half** — converting the 128 `mat_to_vec2` sites to arrays —
  is a type change on 54 functions; not done here (outside the fix remit),
  proposed for the 1.0 freeze with a deprecation note.
- **F6**: 12 seed parameters on 11 functions still mention the seed
  without its default in prose (`bootstrap_indices`, `fry_pagan_svar`,
  `hansen_seo_test`, `kernel_ridge`, `random_forest`, `setar_test`,
  `var_irf_bands` (`seed`); `lp`, `smooth_lp`, `var_forecast`,
  `var_irf_bands` (`band_seed` = 20260807); `philox_uniforms` (`seed` =
  None)) — a hand edit each, not attempted.
- **F10–F12**: a shared NaN/empty-input formatter in the crates and a
  name-resolving `_rank_error` are code changes; proposals only.
- **F14**: the rank message's `(observations, series)` advice is wrong for
  the five `(N, T)` panel functions; a per-function hint would need the
  wrapper to know the family.
- The `Further arguments, with defaults:` lines (F5) state names and
  defaults, not semantics — the honest minimum; the semantics for the
  Bayesian-SVAR and STL/MSTL option sets deserve a prose pass.
- Not attempted for time: a probe of every *second* string-valued
  parameter (only the first per function was probed), a probe with a NaN in
  the *second* data array, and the docs-mention counts for keys that are
  also common English words (`h`, `level`, `lower`) are over-counts.

## Appendix — master table (173 callables)

Value kinds: `f` float · `i` int · `b` bool · `s` str · `∅` None · `{}` dict ·
`1D`/`2D`/`3D` float64 array · `1Du` uint64 array · `L` list of numbers ·
`LL` nested list · `Ls` list of str · `L{}` list of dicts · `La` list of
arrays · `Lb` list of bools · `L0` empty list. `…` marks a default PyO3
cannot print (F5).

| function | family | signature (compiled) | returned keys : kind | docstring first line |
|---|---|---|---|---|
| `accuracy` | Forecasting | `actual, forecast, insample=None, period=1` | `mae`:f `mape`:f `mase`:f `me`:f `rmse`:f `rmsse`:f `smape`:f | Forecast accuracy measures in one call. |
| `acf` | Diagnostics | `y, nlags=20, adjusted=False` | `acf`:1D `bartlett_se`:1D | Autocorrelation function with Bartlett standard errors. |
| `acm_term_premium` | Term structure | `yields, maturities, n_factors=5, periods_per_year=12.0` | `A`:1D `A_rn`:1D `B`:LL `B_rn`:LL `a`:1D `beta`:LL `c`:LL `delta0`:f `delta1`:1D `factor_loadings`:LL `factors`:LL `fitted`:LL `lambda0`:1D `lambda1`:LL `maturities`:L `mu`:1D `n_factors`:i `periods_per_year`:f `phi`:LL `risk_neutral`:LL `rx_maturities`:L `rx_rsquared`:1D `short_rate_rsquared`:f `sigma`:LL `sigma2`:f `term_premium`:LL `var_rsquared`:1D `yield_rsquared`:1D | ACM regression-based term premium (Adrian-Crump-Moench 2013): a Gaussian |
| `adaptive_lasso` | Machine learning | `x, y, alpha, l1_ratio=1.0, gamma=1.0, tol=1e-07, max_iter=100000` | `coef`:1D `max_change`:f `max_rel_change`:f `n_iter`:i | Adaptive LASSO of Zou (2006): a weighted-L1 penalty with data-driven |
| `adf` | Diagnostics | `y, regression="c", autolag=…, maxlag=None` | `crit`:{} `nobs`:i `p_value`:f `statistic`:f `used_lag`:i | Augmented Dickey-Fuller unit-root test with MacKinnon p-values. |
| `afns_adjustment` | Arbitrage-free Nelson-Siegel | `maturities, sigma, decay=0.0609` | → 1-D[f] | Arbitrage-free Nelson-Siegel yield adjustment (Christensen-Diebold- |
| `ar_loglik` | ARIMA | `y, coeffs, sigma2, intercept=0.0` | → float | Gaussian log-likelihood of an AR(p) model with intercept at fixed |
| `arch_lm` | Diagnostics | `resid, nlags=4` | `df`:i `nobs`:i `p_value`:f `statistic`:f | Engle's ARCH-LM test (statsmodels `het_arch` convention). |
| `arima_fit` | ARIMA | `y, p=1, d=0, q=0, seasonal=None, constant=True, forecast_steps=0, conf_alpha=None, drift_uncertainty=False` | `aic`:f `bic`:f `boundary`:1Db `boundary_note`:∅ `bse`:1D `conf_alpha`:f `converged`:b `cov_ok`:b `drift_uncertainty`:b `forecast_lower`:1D `forecast_mean`:1D `forecast_se`:1D `forecast_upper`:1D `loglik`:f `param_cov`:2D `param_names`:Ls `params`:1D `residuals`:1D `se_valid`:1Db | Fit an ARIMA(p,d,q) — optionally a seasonal SARIMA(p,d,q)(P,D,Q)_s |
| `auto_arima` | ARIMA | `y, seasonal_period=0, ic="aicc", stepwise=True, max_p=5, max_q=5, max_P=2, max_Q=2, max_order=5, max_d=2, max_D=1, d=None, D=None, alpha=0.05, forecast_steps=0, conf_alpha=None` | `D_test`:∅ `aic`:f `aicc`:f `bic`:f `boundary`:1Db `boundary_note`:∅ `bse`:1D `budget_exhausted`:b `constant`:b `converged`:b `cov_ok`:b `d_test`:{} `ic`:s `ic_value`:f `interpretation`:s `loglik`:f `n_models`:i `order`:tuple[int,int,int] `param_cov`:2D `param_names`:Ls `params`:1D `residuals`:1D `se_valid`:1Db `seasonal_order`:tuple[int,int,int,int] `stepwise`:b `trace`:L{} | Automatic ARIMA order selection — Hyndman-Khandakar (2008), the |
| `backtest` | Forecasting | `y, window="expanding", train=20, horizon=1, refit_every=1, forecaster=None, period=None, insample_period=1` | `accuracy`:L{} `forecasts`:LL `horizon`:i `n_origins`:i `origins`:L `targets`:LL | Pseudo-out-of-sample backtest over a rolling or expanding window. |
| `bai_perron` | Structural breaks | `y, x, max_breaks=5, trim=0.15` | `break_dates`:1Du `break_dates_by_m`:LL `bse`:LL `ci_lower_90`:1Du `ci_lower_95`:1Du `ci_scale`:1D `ci_upper_90`:1Du `ci_upper_95`:1Du `h`:i `n_breaks`:i `params`:LL `regime_ends`:1Du `regime_ssr`:1D `regime_starts`:1Du `ssr_path`:1D `sup_f_crit`:1D `sup_f_seq`:1D | Bai-Perron multiple structural breaks: global partitions by dynamic |
| `bk_filter` | filters | `y, low=6.0, high=32.0, k=12` | `cycle`:1D `first_index`:i | Baxter-King band-pass filter (loses `k` observations at each end — |
| `bn_decomposition` | filters | `y, p=2, q=2, ar=None, ma=None, drift=None` | `aic`:f `ar`:1D `bic`:f `converged`:b `cycle`:1D `drift`:f `first_index`:i `innovations`:1D `loglik`:f `long_run_multiplier`:f `ma`:1D `mode`:s `sigma2`:f `trend`:1D | Classic Beveridge-Nelson (1981) trend-cycle decomposition from an |
| `bn_filter` | filters | `y, p=12, delta=None, demean="sm", d0=None, dt=None` | `amplitude_to_noise`:f `ar`:1D `cycle`:1D `cycle_se`:f `delta`:f `drift`:f `first_index`:i `trend`:1D | Kamber-Morley-Wong (2018, REStat) Beveridge-Nelson filter: the BN |
| `bns_jump_test` | Realized volatility | `returns` | `ratio`:f | Barndorff-Nielsen-Shephard ratio jump test in the Huang & Tauchen |
| `boosting` | Trend filtering and boosting | `x, y, learning_rate=0.1, n_steps=500, stop="aic", x_test=None` | `aic_path`:1D `best_step`:i `coef`:1D `coef_path`:2D `df_path`:1D `fitted`:1D `predicted`:∅ `rss_path`:1D `selected`:1Di | Componentwise L2 boosting with single-column least-squares base |
| `bootstrap_indices` | bootstrap | `n, scheme="stationary", seed=0, block_length=None, p=None` | → 1-D[u] | Bootstrap resampling indices. |
| `box_cox_lambda` | unit roots / workflow | `y, method="mle", bounds=…, period=None` | `at_bound`:b `interpretation`:s `lambda`:f `loglik_at_one`:f `loglik_at_zero`:f `lower`:f `lr_vs_one`:f `lr_vs_zero`:f `method`:s `n`:i `objective`:f `period`:∅ `upper`:f | Variance-stabilizing Box-Cox lambda, with the objective at the optimum. |
| `bvar_fit` | Bayesian | `data, lags=2, lambda0=100.0, lambda1=0.2, lambda3=1.0, delta=0.0, scale_ar=4` | `log_marginal_likelihood`:f `omega_bar`:LL `posterior_mean_coefs`:LL `s_bar`:LL `sigma_posterior_mean`:LL `v_bar`:f | Fit a Bayesian VAR with the Minnesota / Normal-inverse-Wishart |
| `bvar_hierarchical` | Bayesian | `data, lags=2, delta=0.0, lambda0=100.0, lambda3=1.0, lambda1_init=0.2, lambda1_lo=0.0001, lambda1_hi=10.0, optimize="lambda1", hyperprior="glp", n_grid=25, max_iter=200, tol=1e-08, scale_ar=4` | `converged`:b `grid_lambda1`:L `grid_log_ml`:L `lambda1_fixed_log_ml`:f `lambda1_opt`:f `lambda3_opt`:f `log_marginal_likelihood`:f `log_posterior`:f `n_evals`:i `posterior_mean_coefs`:LL `sigma_posterior_mean`:LL | Hierarchical (empirical-Bayes / ML-II) Minnesota-BVAR: select the prior |
| `bvar_irf_draws` | Bayesian | `data, lags=2, horizon=16, n_draws=500, seed=0, lambda0=100.0, lambda1=0.2, lambda3=1.0, delta=0.0, cumulative=False, scale_ar=4` | → list[list] | Posterior draws of Cholesky-orthogonalized impulse responses from the |
| `bvar_ssvs` | Bayesian | `data, lags=2, n_draws=10000, burn=2000, seed=0, c0=0.1, c1=10.0, prior_inclusion=0.5, ssvs_cov=False, kappa0=None, kappa1=None, prior_inclusion_cov=0.5, gamma_a=0.01, gamma_b=None, horizon=16, thin=1, n_chains=1` | `coef_mean`:LL `diagnostics`:{} `inclusion_prob`:LL `irf_draws`:LL `sigma_mean`:LL | SSVS-BVAR (George, Sun & Ni 2008): spike-and-slab stochastic-search |
| `ccc_garch` | Volatility | `returns, forecast_horizon=0, vol="garch", mean="zero", univariate_dist="normal", p=1, o=None, q=1` | `correlation`:LL `covariance`:LL `covariance_forecast`:LL `loglik`:f `sigma2`:LL `variance_forecast`:LL | CCC-GARCH (Bollerslev 1990): a univariate GARCH per series with a |
| `cf_filter` | filters | `y, low=6.0, high=32.0, drift=True` | `cycle`:1D `first_index`:i `trend`:1D | Christiano-Fitzgerald asymmetric band-pass filter (full sample). Keys: `trend`, `cycle` an |
| `cg_regression` | Survey expectations | `errors, revisions, maxlags=None, use_correction=True` | `implied_rigidity`:f `intercept`:f `maxlags`:i `nobs`:i `p_slope`:f `r_squared`:f `se_intercept`:f `se_slope`:f `slope`:f `t_slope`:f | Coibion-Gorodnichenko (2015) information-rigidity regression. |
| `check_series` | Check series (one-call battery) | `data, seasonal_period=None, lags=None, alpha=0.05, max_breaks=5, trim=0.15` | `alpha`:f `analysis_scale`:{} `arch_effects`:{} `breaks`:{} `descriptives`:{} `kind`:s `long_memory`:{} `multiple_testing`:{} `n`:i `normality`:{} `outliers`:{} `recommendations`:L{} `seasonality`:{} `serial_correlation`:{} `stationarity`:{} `tests_run`:L{} | One-call diagnostic battery with model recommendations. |
| `check_stationarity` | Diagnostics | `y, alpha=0.05` | `adf_p_value`:f `adf_statistic`:f `alpha`:f `interpretation`:s `kpss_p_value`:f `kpss_statistic`:f `quadrant`:s `recommendation`:s | The stationarity decision workflow: ADF and KPSS run together and |
| `chow_test` | Specification & diagnostic tests | `y, x, split` | `df_den`:i `df_num`:i `fstat`:f `pvalue`:f `ssr1`:f `ssr2`:f `ssr_pooled`:f | Chow structural-break test at a known 0-indexed `split`: F-test that the |
| `coherence` | Spectral analysis | `x, y, nperseg=256, fs=1.0, noverlap=None, window="hann", detrend="constant"` | `coherence`:1D `freqs`:1D | Magnitude-squared coherence between two series via Welch cross-spectra. |
| `conformal_backtest` | conformal forecast intervals | `y, horizon=1, method="split", base=None, alpha=0.1, calib=None, mode="symmetric", period=1, gamma=None, n_eval=None, lags=None, n_boot=None, batch=None, seed=None, optimize_beta=None, order=None` | `alpha`:f `base`:s `err`:La `horizon`:i `level`:f `lower`:La `mean`:La `method`:s `n_eval`:i `origins`:L `realized_coverage`:1D `upper`:La | Online out-of-sample evaluation of conformal forecast intervals: form |
| `conformal_forecast` | conformal forecast intervals | `y, horizon=1, method="split", base=None, alpha=0.1, calib=None, mode="symmetric", period=1, gamma=None, n_eval=None, lags=None, n_boot=None, seed=None, optimize_beta=None, order=None` | `alpha`:f `base`:s `finite_sample_level`:f `horizon`:i `level`:f `lower`:1D `mean`:1D `method`:s `mode`:s `n_calib`:i `q_lower`:1D `q_upper`:1D `scores`:La `upper`:1D | Distribution-free conformal prediction intervals around a point |
| `connectedness` | VAR / SVAR | `data, lags=2, horizon=10, trend="c"` | `from_others`:1D `gfevd`:LL `net`:1D `pairwise_net`:LL `to_others`:1D `total`:f | Diebold-Yilmaz connectedness from a VAR's generalized FEVD. |
| `copula_fit` | static copulas | `u, family="gaussian", method="mle"` | `aic`:f `bic`:f `converged`:b `family`:s `loglik`:f `method`:s `n`:i `param_names`:Ls `params`:1D `rho`:f `se`:1D `se_rho`:f `se_valid`:b `tail_lower`:f `tail_upper`:f `tau`:f `tau_implied`:f | Fits a bivariate copula to (n, 2) probability-scale pseudo-observations. |
| `copula_select` | static copulas | `u, families=None, method="mle"` | `best_aic`:s `best_bic`:s `fits`:{} `ranking_aic`:Ls `ranking_bic`:Ls `skipped`:{} `verdict`:s | Fits several copula families to the same (n, 2) pseudo-observations |
| `cusum_test` | Specification & diagnostic tests | `y, x` | `bound_lower`:1D `bound_upper`:1D `path`:1D `sigma`:f | CUSUM parameter-stability test (Brown-Durbin-Evans 1975) on the recursive |
| `cv_splits` | Machine learning | `n, scheme="expanding", train=0, horizon=1, step=1, k=5, purge=0, embargo=0` | → list[dict] | Leakage-safe cross-validation splits for time-series / sequential data. |
| `cw_test` | Forecasting | `e_small, e_large, yhat_small, yhat_large, lrv_lags=0` | `cw_stat`:f `mean_adj_diff`:f `p_value`:f | Clark-West test for nested-model equal predictive accuracy (Clark-West |
| `dcc_garch` | Volatility | `returns, variant="dcc", dist="normal", forecast_horizon=0, vol="garch", mean="zero", univariate_dist="normal", p=1, o=None, q=1` | `a`:f `b`:f `converged`:b `correlation`:LL `correlation_forecast`:LL `correlation_last`:LL `covariance`:LL `covariance_forecast`:LL `dist`:s `g`:f `loglik`:f `qbar`:LL `sigma2`:LL `std_residuals`:LL `univariate`:L{} `variance_forecast`:LL `variant`:s | DCC-GARCH (Engle 2002): GARCH(1,1) per series with dynamic conditional |
| `dcc_test` | multivariate GARCH | `returns, lags=5, vol="garch", mean="zero", univariate_dist="normal", p=1, o=None, q=1` | `df`:i `lags`:i `n_stacked`:i `nobs`:i `p_value`:f `stat`:f | Engle-Sheppard (2001) test of constant conditional correlation — the |
| `dcs_local_level` | Volatility | `y, density="t"` | `aic`:f `bic`:f `converged`:b `density`:s `iterations`:i `kappa`:f `kappa_se`:f `level`:1D `loglik`:f `n_obs`:i `next_level`:f `nu`:f `nu_se`:f `resid`:1D `scale`:f `scale_se`:f | DCS robust local level (Harvey 2013; Harvey-Luati 2014): a score-driven |
| `dfgls` | unit roots / workflow | `y, regression="c", lags=None, max_lags=None, method="aic"` | `crit`:{} `nobs`:i `p_value`:f `statistic`:f `trend`:s `used_lag`:i | DF-GLS unit-root test (Elliott-Rothenberg-Stock 1996; null: unit root). |
| `dfm_news` | Nowcasting & MIDAS | `old_vintage, new_vintage, target_series=0, target_period=None, n_factors=1, factor_order=2` | `contributions`:L{} `new_nowcast`:f `old_nowcast`:f `target_period`:i `target_series`:i `total_revision`:f | News / update decomposition of a DFM nowcast revision (Bańbura-Modugno |
| `dfm_nowcast` | Nowcasting & MIDAS | `data, n_factors=1, factor_order=2, method="two_step"` | `center`:1D `edge_factor`:1D `factor_ar`:LL `factor_cov`:LL `factor_order`:i `fit_loglik`:f `idiosyncratic`:1D `loadings`:LL `loglik`:f `n_factors`:i `nowcast`:1D `scale`:1D `smoothed_factors`:LL | Dynamic-factor-model nowcast (Doz-Giannone-Reichlin 2011 two-step). |
| `dm_test` | Forecasting | `e1, e2, h=1, loss="squared"` | `dm_stat`:f `hln_stat`:f `mean_loss_diff`:f `p_value`:f | Diebold-Mariano test of equal predictive accuracy with the |
| `dsge_solve` | DSGE (linear RE solver) | `a, b, c, n_predetermined` | `eigenvalue_moduli`:1D `g`:LL `p`:LL `q`:LL `verdict`:s | Solve a linear rational-expectations (DSGE-lite) model by Blanchard-Kahn |
| `dynamic_ns` | Term structure | `panel, maturities, decay=0.0609` | `curvature`:1D `factors`:LL `forecast`:{} `lambda`:f `level`:1D `maturities`:1D `rsquared`:1D `slope`:1D | Dynamic Nelson-Siegel factors and one-step curve forecast |
| `echo_state_network` | Neural (MLP, echo state network) | `x, y, reservoir_size=200, spectral_radius=0.9, leak_rate=1.0, input_scaling=1.0, sparsity=0.1, washout=50, ridge_alpha=1e-06, seed=0, x_test=None` | `fitted`:1D `n_train`:i `n_washout`:i `predicted`:∅ `readout`:1D `reservoir_size`:i `spectral_radius_achieved`:f | Echo state network (reservoir computing; Jaeger 2001; Lukosevicius |
| `elastic_net` | Machine learning | `x, y, alpha, l1_ratio=0.5, tol=1e-08, max_iter=100000` | `coef`:1D `max_change`:f `max_rel_change`:f `n_iter`:i | Elastic-net regression via coordinate descent. Minimizes |
| `engle_granger` | cointegration | `data, trend="c", autolag=…, maxlag=None` | `adf_nobs`:i `coint_coefs`:1D `crit`:{} `n_vars`:i `nobs`:i `pvalue`:f `resid`:1D `stat`:f `used_lag`:i | Engle-Granger two-step cointegration test (null: no cointegration). |
| `factor_model` | factor model | `data, n_factors=2, kmax=8` | `eigenvalues`:1D `er`:i `er_ratios`:1D `factors`:LL `icp1`:i `icp2`:i `loadings`:LL `pcp1`:i `pcp2`:i | Static approximate factor model (PCA) with Bai-Ng factor selection. |
| `favar` | VAR / SVAR | `panel, policy, n_factors=2, lags=2, trend="c", slow_indices=None, horizon=20, orth=True` | `factors`:LL `irf_panel`:LL `irf_policy`:1D `n_endog`:i `n_factors`:i `params`:LL `policy_index`:i `sigma_u`:LL | Two-step factor-augmented VAR (Bernanke-Boivin-Eliasz 2005, QJE). |
| `flp` | Functional shocks (FVAR/FLP) | `y, scores, horizons=8, n_lag_controls=2, hac_maxlags=None` | `betas`:LL `covs`:LL `horizons`:1Du `n_factors`:i `nobs`:1Du `se`:LL | Functional local projection (Inoue-Rossi 2021): at each horizon h regress |
| `flp_scenario` | Functional shocks (FVAR/FLP) | `y, curves, delta, n_factors=3, horizons=8, n_lag_controls=2, hac_maxlags=None` | `betas`:LL `explained`:1D `horizons`:1Du `response`:1D `se`:1D `weights`:1D | One-call functional-shock IRF: functional PCA of the curves, joint FLP of |
| `forecast_disagreement` | Survey expectations | `panel, ddof=1` | `counts`:L `iqr`:1D `p25`:1D `p50`:1D `p75`:1D `std`:1D | Forecast-disagreement measures from a panel of forecasters. |
| `forecast_efficiency` | Survey expectations | `errors, regressors, maxlags=None, use_correction=True` | `bse`:1D `params`:1D `pvalues`:1D `r_squared`:f `tvalues`:1D `wald`:f `wald_df`:i `wald_pvalue`:f | Mincer-Zarnowitz forecast-efficiency (rationality) test. |
| `frac_diff` | Long memory | `x, d` | → 1-D[f] | Fractional differencing `(1 - L)^d x` via the binomial expansion. |
| `frac_integrate` | Long memory | `x, d` | → 1-D[f] | Fractional integration `(1 - L)^{-d} x` — the inverse of `frac_diff`. |
| `fry_pagan_svar` | structural identification | `data, restrictions, lags=2, horizon=12, n_draws=500, max_tries=400, seed=0, lambda1=0.2, target="restricted"` | `diagnostics`:{} `median_irf`:LL `median_target_irf`:LL `mt_index`:i `mt_statistic`:f `n_accepted`:i | Fry-Pagan (2011) median-target SVAR: the single accepted sign-restricted |
| `functional_pca` | Functional shocks (FVAR/FLP) | `curves, n_factors=3` | `eigenfunctions`:LL `eigenvalues`:1D `explained`:1D `mean_curve`:1D `scores`:LL `total_variance`:f | Functional PCA of a T x M panel of curve observations (e.g. daily |
| `fvar_scenario` | Functional shocks (FVAR/FLP) | `y, curves, delta, n_factors=3, lags=2, horizon=10` | `horizons`:1Du `implied_outcome_innovation`:f `response_outcome`:1D `responses`:LL `weights`:1D | FVAR whole-curve scenario (Inoue-Rossi 2021): fit a VAR to [scores, y] |
| `garch_fit` | Volatility | `y, vol="garch", mean="zero", dist="normal", p=1, o=None, q=1, forecast_horizon=0` | `aic`:f `bic`:f `boundary`:1Db `boundary_note`:s `conditional_volatility`:1D `converged`:b `loglik`:f `param_names`:Ls `params`:1D `params_named`:{} `se_mle`:1D `se_robust`:1D `se_valid`:1Db `std_residuals`:1D `variance_forecast`:1D | Fit a univariate volatility model by QMLE. |
| `gas_volatility` | Volatility | `y, density="gaussian", horizon=0` | `a`:f `aic`:f `b`:f `bic`:f `converged`:b `forecast`:1D `iterations`:i `loglik`:f `next_variance`:f `omega`:f `std_resid`:1D `variance`:1D | GAS(1,1) score-driven volatility (Creal-Koopman-Lucas 2013). |
| `gev_fit` | extreme value theory | `y, block_size=None, return_periods=None` | `block_size`:i `converged`:b `loglik`:f `mu`:f `n_maxima`:i `return_levels`:1D `return_periods`:1D `se_mu`:f `se_sigma`:f `se_valid`:b `se_xi`:f `sigma`:f `xi`:f | GEV block-maxima fit with return levels. |
| `gmm_nonlinear` | GMM | `moments_fn, initial, weight=None` | `converged`:b `fevals`:i `gbar`:1D `iterations`:i `nmoments`:i `nparams`:i `objective`:f `params`:1D | Nonlinear GMM driver (Hansen 1982) minimizing `gbar(theta)' W gbar(theta)` |
| `gpd_fit` | extreme value theory | `y, threshold=None, quantile=0.9, p_tail=None` | `beta`:f `converged`:b `es`:1D `exceed_rate`:f `loglik`:f `n`:i `n_exceed`:i `p_tail`:1D `se_beta`:f `se_valid`:b `se_xi`:f `threshold`:f `threshold_quantile`:f `var`:1D `xi`:f | Peaks-over-threshold GPD tail fit with McNeil-Frey (2000) VaR/ES. |
| `group_lasso` | structured penalties and post-selection | `x, y, groups, alpha, l1_ratio=0.0, group_weights=None, tol=1e-08, max_iter=10000` | `active_groups`:L `active_set`:L `alpha_max`:f `coef`:1D `converged`:b `kkt_violation`:f `max_rel_change`:f `n_iter`:i `objective`:f | Group LASSO (Yuan & Lin 2006) and sparse-group LASSO (Simon, Friedman, |
| `growth_at_risk` | Quantile regression & growth-at-risk | `y, conditions, horizon=1, taus=None, rearrange=True` | `bse`:LL `bse_powell`:LL `converged`:Lb `crossing`:b `current`:1D `fitted`:LL `fitted_raw`:LL `hac_lags`:i `horizon`:i `params`:LL `taus`:1D | Growth-at-risk (Adrian-Boyarchenko-Giannone 2019 AER): conditional |
| `gw_test` | Forecasting | `loss1, loss2, lrv_lags=0` | `df`:i `gw_stat`:f `p_value`:f | Giacomini-White unconditional test of equal predictive ability |
| `hamilton_filter` | filters | `y, h=8, p=4, method="regression", se=None, maxlags=None, use_correction=None` | `beta`:1D `cycle`:1D `first_index`:i `trend`:1D | Hamilton (2018) regression filter — the modern HP alternative. |
| `hansen_seo_test` | cointegration | `data, k_ar_diff=1, trim=0.05, n_grid=300, n_boot=499, seed=0, beta=None` | `beta`:1D `boot_stats`:1D `k_ar_diff`:i `lm_path`:1D `min_regime`:i `n_boot`:i `n_regressors`:i `neqs`:i `nobs`:i `p_value`:f `stat`:f `threshold`:f `thresholds`:1D | Hansen-Seo (2002) sup-LM test of LINEAR cointegration against |
| `har_rv` | Realized volatility | `rv, start=22, variant="level", hac_maxlags=5, use_correction=True` | `bse`:1D `nobs`:i `params`:1D `rsquared`:f `tvalues`:1D | HAR-RV heterogeneous autoregression of realized variance (Corsi 2009). |
| `hetero_svar` | Structural identification (advanced) | `data, regime_labels, lags=2, horizon=12, trend="c", base_regime=None, sign_normalization="max"` | `B`:LL `covariance_equality`:{} `horizon`:i `identified`:b `lags`:i `min_ratio_gap`:f `n_vars`:i `ratio_dist_from_unity`:L `regime1_label`:i `regime2_label`:i `regime_sizes`:L `sigma_regime1`:LL `sigma_regime2`:LL `sign_convention`:s `structural_irf`:LL `variance_ratios`:L | Identification through heteroskedasticity (Rigobon 2003; Lanne-Lutkepohl |
| `heteroskedasticity_test` | Specification & diagnostic tests | `y, x, test="white"` | `df`:i `f_pvalue`:f `fstat`:f `pvalue`:f `statistic`:f | Heteroskedasticity test on an OLS regression of `y` on `x` (a `T x k` |
| `historical_decomposition` | structural identification | `data, restrictions=…, lags=2, horizon=None, identification="cholesky", n_draws=500, max_tries=400, seed=0, lambda1=0.2, narrative_restrictions=None, n_weight_draws=200` | `baseline`:LL `hd`:LL `shocks`:LL `times`:L | Historical decomposition (Kilian & Lütkepohl 2017, ch.4): per-(time, variable, |
| `hp_filter` | filters | `y, lamb=1600.0, one_sided=False` | `cycle`:1D `first_index`:i `trend`:1D | Hodrick-Prescott filter (O(n) pentadiagonal solve). `one_sided=True` |
| `iv_gmm` | GMM | `x, z, y, method="2step", weight="robust", bandwidth=None, tol=1e-08, max_iter=100` | `bse`:1D `cov`:2D `first_stage`:L{} `hac_bandwidth`:∅ `j_dof`:i `j_pval`:f `j_stat`:f `nmoments`:i `nobs`:i `nparams`:i `params`:1D `residuals`:1D `steps`:i | Linear IV-GMM (Hansen 1982) with a robust or HAC weighting matrix. |
| `ivx_test` | Predictive regressions & IVX | `r, xs, cz=…, alpha=0.95, joint="bonferroni"` | `beta_ivx`:1D `joint`:s `nobs`:i `nregressors`:i `pvalue`:f `pvalue_scalar`:1D `rz`:f `wald`:f `wald_scalar`:1D | Joint IVX predictability test for several persistent predictors at once |
| `jarque_bera` | Diagnostics | `x` | `kurtosis`:f `n`:i `p_value`:f `skewness`:f `statistic`:f | Jarque-Bera normality test. Keys: `statistic`, `p_value`, `skewness`, `kurtosis` (raw, not |
| `johansen` | Cointegration & regimes | `data, k_ar_diff=1` | `eig`:1D `evec`:LL `max_eig_crit_90_95_99`:LL `max_eig_stat`:1D `rank_max_eig_5pct`:i `rank_trace_5pct`:i `trace_crit_90_95_99`:LL `trace_stat`:1D | Johansen cointegration test (Johansen 1991). `data` is T x k (rows are |
| `kernel_regression` | kernel methods | `x, y, bandwidth=None, kind="local_linear", kernel="gaussian", bandwidth_method="fixed", block=None, x_test=None` | `bandwidth`:1D `bandwidth_at_boundary`:b `bandwidth_method`:s `block`:∅ `cv_criterion`:f `effective_df`:f `fitted`:1D `kernel`:s `kind`:s `n_criterion_evaluations`:i | Nadaraya-Watson or local-linear kernel regression of `y` on `x` |
| `kernel_ridge` | kernel methods | `x, y, alpha=1.0, kernel="rbf", gamma=None, degree=3.0, coef0=1.0, x_test=None, rff_features=None, seed=0` | `dual_coef`:1D `fitted`:1D `gamma`:f `kernel`:s `n_rff_features`:∅ | Kernel ridge regression: exact dual solve, or the Rahimi-Recht |
| `kpss` | Diagnostics | `y, regression="c", nlags=None` | `lags`:i `p_value`:f `statistic`:f | KPSS stationarity test (null: stationary). |
| `l1_trend_filter` | Trend filtering and boosting | `y, lam, order=2, penalty="l1", tol=None, max_iter=None` | `converged`:b `cycle`:1D `duality_gap`:f `knots`:1Di `lam_max`:f `n_iter`:i `n_knots`:i `objective`:f `trend`:1D | L1 trend filtering (Kim, Koh & Boyd 2009) — a piecewise-linear trend |
| `lasso` | Machine learning | `x, y, alpha, tol=1e-08, max_iter=100000` | `coef`:1D `max_change`:f `max_rel_change`:f `n_iter`:i | Lasso regression (elastic net with l1_ratio = 1.0). Keys: `coef`, `n_iter`, `max_change` ( |
| `lasso_path` | Machine learning | `x, y, l1_ratio=1.0, n_lambdas=100, eps=0.001, tol=1e-07, max_iter=100000` | `aic`:1D `aic_best`:i `bic`:1D `bic_best`:i `coefs`:LL `df`:L `lambdas`:1D `rss`:1D | Elastic-net regularization path over an automatic lambda grid. |
| `ljung_box` | Diagnostics | `y, nlags=10` | `bp_pvalue`:1D `bp_stat`:1D `lags`:1Du `lb_pvalue`:1D `lb_stat`:1D | Ljung-Box and Box-Pierce portmanteau tests for lags 1..=nlags. |
| `local_level_smooth` | state space | `y, sigma2_eps, sigma2_eta` | `d_diffuse`:i `filtered_state`:1D `filtered_state_var`:1D `loglik`:f `smoothed_state`:1D `smoothed_state_var`:1D | Fit-free local-level pass: exact-diffuse Kalman filter + smoother at |
| `long_memory_d` | Long memory | `x, m=None, method="gph"` | `d`:f `m`:i `se`:f `se_asymptotic`:f `se_regression`:f | Estimate the fractional-integration (long-memory) parameter `d`. |
| `long_run_svar` | Structural identification (advanced) | `data, lags=2, horizon=12, trend="c", restrictions=None, normalize="long_run"` | `cumulative_irf`:LL `fevd`:LL `impact`:LL `irf`:LL `long_run`:LL `long_run_multiplier`:LL | Blanchard-Quah long-run SVAR: closed-form structural IRFs under the |
| `long_run_variance` | robust inference | `x, kernel="bartlett", bandwidth=None` | → float | Kernel long-run variance of a series (demeaned internally). |
| `lp` | Local projections | `y, shock, horizons=12, n_lag_controls=4, se=None, maxlags=None, cumulative=None, band=None, band_alpha=0.1, band_seed=20260807, band_n_sim=100000` | `horizons`:1Du `irf`:1D `se`:1D `se_method`:s | Local projection impulse responses (Jordà 2005). |
| `lp_did` | panel | `outcome, treatment, pre_window=4, post_window=8, absorbing=True, nonabsorbing_lag=0, reweight=False, pooled=False, never_treated_only=False` | `absorbing`:b `coef`:1D `horizons`:1Di `n_switchers`:1Du `never_treated_only`:b `nobs`:1Du `nonabsorbing_lag`:i `pooled`:b `reweight`:b `se`:1D `se_type`:s | LP-DiD: local-projections difference-in-differences (Dube, Girardi, |
| `lp_iv` | Local projections | `y, impulse, instrument, horizons=8, n_lag_controls=4, cumulative=None, band=None, band_alpha=0.1` | `first_stage_f`:1D `horizons`:1Du `irf`:1D `se`:1D | LP-IV: instrumental-variable local projections (Stock-Watson 2018, |
| `lp_multiplier` | Local projections | `y, impulse, instrument, horizons=20, n_lag_controls=4, maxlags=None, band=None, band_alpha=0.1` | `cumulative_impulse`:1D `cumulative_outcome`:1D `first_stage_f`:1D `horizons`:1Du `multiplier`:1D `nobs_per_h`:1Du `se`:1D | Ramey-Zubairy (2018) integral multiplier by one-step LP-IV. |
| `lp_state` | Local projections | `y, shock, state_indicator, horizons=12, n_lag_controls=4, se=None, maxlags=None, cumulative=None, band=None, band_alpha=0.1` | `horizons`:1Du `irf_state0`:1D `irf_state1`:1D `se_method`:s `se_state0`:1D `se_state1`:1D | State-dependent (interacted) local projections (Ramey & Zubairy 2018). |
| `markov_switching_ar` | Cointegration & regimes | `y, k_regimes=2, order=1, switching_variance=True, max_iter=500, tol=1e-06` | `ar`:1D `converged`:b `expected_durations`:L `filtered_prob`:2D `iterations`:i `loglik`:f `means`:L `regimes`:1Du `smoothed_prob`:2D `smoothed_prob_last_regime`:1D `transition`:LL `variances`:L | Markov-switching autoregression (Hamilton 1989), fitted by EM. |
| `max_share_svar` | Structural identification (advanced) | `data, lags=2, target=0, h0=0, h1=40, horizon=40, trend="c", exclude_impact=False, weighting="window", sign="cumsum"` | `eigenvalues`:L `fev_share`:L `impact`:L `irf`:LL `q`:L `share_window`:f | Max-share / maximum-FEV structural shock (Uhlig 2004; Francis, Owyang, |
| `mcmc_diagnostics` | Bayesian | `chains` | `ess_bulk`:f `ess_tail`:f `rhat`:f | MCMC convergence diagnostics (Vehtari et al. 2021, ArviZ-exact): |
| `mean_group_var` | Panel | `entities, lags=1, trend="c", horizon=10, response=0, impulse=0` | `coefs`:LL `coefs_se`:LL `intercept`:1D `intercept_se`:1D `irf_path`:1D `irf_path_se`:1D `lags`:i `n_entities`:i `neqs`:i `orth_irfs`:LL `orth_irfs_se`:LL | Pesaran-Smith (1995) mean-group panel VAR: fit a reduced-form VAR(p) |
| `midas_weights` | Nowcasting & MIDAS | `scheme, theta1, theta2, k` | → 1-D[f] | MIDAS weight function (normalized to sum 1). `scheme`: "exp_almon" |
| `mlp_regression` | Neural (MLP, echo state network) | `x, y, hidden=None, activation="tanh", alpha=0.0001, solver="adam", learning_rate=None, batch_size=None, max_epochs=500, validation_fraction=0.2, patience=None, n_seeds=5, seed=0, standardize=True, x_test=None` | `activation`:s `best_epoch`:L `converged`:Lb `fitted`:1D `member_predictions`:∅ `n_parameters`:i `n_train`:i `n_validation`:i `predicted`:∅ `solver`:s `train_loss_path`:La `validation_loss_path`:La `weights`:L{} `x_mean`:1D `x_scale`:1D `y_mean`:f `y_scale`:f | Feed-forward neural regressor with a seed ensemble, early stopping on a |
| `mstl` | filters | `y, periods, windows=None, iterate=2, trend=None, low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1, inner_iter=None, outer_iter=None` | `dropped_periods`:L0 `iterate`:i `periods`:L `resid`:1D `seasonal`:{} `seasonal_strength`:{} `trend`:1D `weights`:1D `windows`:L | MSTL — Multiple Seasonal-Trend decomposition using LOESS |
| `narrative_svar` | structural identification | `data, sign_restrictions=…, narrative_restrictions=None, lags=2, horizon=12, n_draws=500, max_tries=400, seed=0, lambda1=0.2, n_weight_draws=200` | `diagnostics`:{} `probs`:L `quantiles`:LL `set_max`:LL `set_min`:LL `weights`:L | Narrative sign-restricted Bayesian SVAR (Antolín-Díaz & Rubio-Ramírez 2018). |
| `ndiffs` | unit roots / workflow | `y, test="kpss", alpha=0.05, max_d=2` | `alpha`:f `d`:i `interpretation`:s `max_d`:i `steps`:L{} `stop`:s `test`:s | How many differences a series needs — with the evidence at every order. |
| `nelson_siegel` | Term structure | `maturities, yields, decay=0.0609, optimal_lambda=False` | `curvature`:f `factors`:1D `lambda`:f `level`:f `residuals`:1D `rsquared`:f `slope`:f | Nelson-Siegel yield-curve fit (Diebold-Li 2006). |
| `ng_perron` | unit roots / workflow | `y, trend="c", lags=None, max_lags=None` | `crit`:{} `mpt`:f `msb`:f `mza`:f `mzt`:f `nobs`:i `s2_ar`:f `trend`:s `used_lag`:i | Ng-Perron (2001) M unit-root tests (MZa, MZt, MSB, MPT; null: unit root). |
| `nongaussian_svar` | structural identification | `data, lags=2, horizon=12, trend="c", contrast="logcosh", max_iter=200, tol=1e-08, order_by="kurtosis"` | `converged`:b `impact`:LL `irf`:LL `n_iter`:i `order`:L `rotation`:LL `shock_kurtosis`:L | Non-Gaussian / independent-component SVAR identification (Lanne-Meitz- |
| `nsdiffs` | unit roots / workflow | `y, period, alpha=0.05, max_d=1` | `alpha`:f `d`:i `interpretation`:s `max_d`:i `period`:i `steps`:L{} `stop`:s `threshold`:f | How many SEASONAL differences a series needs — the Hyndman-Khandakar |
| `ols` | robust inference | `y, x, se_type="hac", maxlags=None, use_correction=True` | `bse`:1D `params`:1D `se_type`:s `tvalues`:1D | OLS with robust standard-error options. |
| `optimal_block_length` | bootstrap | `y` | `circular`:f `stationary`:f | Politis-White (2004) automatic block length with the Patton-Politis-White |
| `ou_fit` | cointegration | `x, dt=1.0, level=0.95` | `c`:f `c_se`:f `dt`:f `eta2`:f `half_life`:f `half_life_ci`:tuple[float,float] `kappa`:f `kappa_se`:f `level`:f `loglik`:f `mean_reverting`:b `mu`:f `mu_se`:f `n_obs`:i `phi`:f `phi_se`:f `sigma`:f `sigma_se`:f `stationary_sd`:f | Ornstein-Uhlenbeck mean-reversion fit for a spread, by the |
| `pacf` | Diagnostics | `y, nlags=20, method="yw"` | → 1-D[f] | Partial autocorrelation function. |
| `panel_fe` | Panel | `outcome, regressors, se_type="cluster", bandwidth=None` | `bse`:1D `params`:1D `se_type`:s `tvalues`:1D | Fixed-effects (within) panel OLS with panel-robust standard errors. |
| `panel_lp` | Panel | `outcome, shock, horizon=8, n_lag_controls=2, se_type="driscoll_kraay", bandwidth=None, cumulative=False, jackknife=False, bias_correction="none", band=None, band_alpha=0.1` | `bias_correction`:s `cumulative`:b `irf`:1D `jackknife`:b `nobs`:1Du `se`:1D `se_type`:s | Panel local projection of a common shock (Jordà 2005 for panels), with |
| `panel_mean_group` | Panel | `ys, xs, method="mg"` | `coef`:1D `coef_per_unit`:LL `k`:i `n_units`:i `se`:1D `tstat`:1D | Heterogeneous-panel mean-group estimator (Pesaran-Smith 1995) and its |
| `panel_pmg` | Panel | `ys, xs, tol=3e-13, max_iter=1000` | `iterations`:i `k`:i `loglik`:f `n_units`:i `phi`:1D `phi_bar`:f `sigma2`:1D `theta`:1D `theta_se`:1D | Pooled Mean Group (PMG) ARDL(1,1) panel estimator (Pesaran-Shin-Smith |
| `panel_unit_root` | Panel unit-root tests | `data, test="ips", lags=None, regression="c", max_lags=None, lrv_kernel="bartlett", lrv_bandwidth=None` | `n_units`:i `p_value`:f `per_unit_lags`:1Di `per_unit_nobs`:1Di `per_unit_pvalue`:1D `per_unit_tstat`:1D `regression`:s `statistic`:f `t_bar`:f `test`:s | First-generation panel unit-root tests: Levin-Lin-Chu, Im-Pesaran-Shin, |
| `pds_lasso` | structured penalties and post-selection | `y, d, x, alpha=None, hac_lags=None, tol=1e-08, max_iter=100000` | `alpha_d`:f `alpha_y`:f `coef`:f `conf_int`:tuple[float,float] `hac_lags_resolved`:i `n_controls_selected`:i `p_value`:f `se`:f `support_d`:L `support_y`:L `t_stat`:f `union_support`:L | Post-double-selection LASSO (Belloni, Chernozhukov & Hansen 2014) for |
| `periodogram` | Spectral analysis | `x, fs=1.0, window="boxcar", detrend="constant"` | `freqs`:1D `psd`:1D | Periodogram power spectral density (one FFT). Matches |
| `phillips_ouliaris` | Phillips-Perron & Ouliaris tests | `y, x, trend="c", test_type="Zt", bandwidth=None` | `crit`:{} `lags`:i `n_vars`:i `nobs`:i `pvalue`:f `stat`:f | Phillips-Ouliaris residual cointegration test (null: no cointegration). |
| `phillips_perron` | Phillips-Perron & Ouliaris tests | `y, regression="c", test_type="tau", lags=None` | `crit`:{} `lags`:i `nobs`:i `pvalue`:f `stat`:f `zalpha`:f `ztau`:f | Phillips-Perron unit-root test (semiparametric; null: unit root). |
| `philox_uniforms` | bootstrap | `seed, n` | → 1-D[f] | Uniform draws from the tsecon Philox stream. |
| `post_lasso` | structured penalties and post-selection | `x, y, alpha, l1_ratio=1.0, tol=1e-08, max_iter=100000` | `coef_lasso`:1D `coef_ols`:1D `n_selected`:i `rss`:f `support`:L | Post-LASSO OLS (Belloni & Chernozhukov 2013): fit the LASSO (or the |
| `predictive_regression` | Predictive regressions & IVX | `r, x, cz=…, alpha=0.95` | `ivx`:{} `nobs`:i `ols`:{} `stambaugh`:{} | Predictive regression of `r_{t+1}` on a persistent predictor `x_t`, with |
| `proxy_ar_sets` | structural identification | `data, proxy, lags=2, horizon=12, norm_var=0, unit=1.0, trend="c", alpha=0.05, variance="hc0", hac_lags=None, reduced_form_uncertainty=True, rf_method="delta", rf_draws=None, rf_seed=None` | `ar_bound_stat`:f `ar_bounded_all`:b `cells`:LL `critical_value`:f `first_stage_f`:f `impact`:1D `level`:f `n_proxy`:i `reduced_form_uncertainty`:b | Weak-instrument-robust (Anderson-Rubin) confidence SETS for a proxy-SVAR |
| `proxy_first_stage` | structural identification | `data, proxy, lags=2, norm_var=0, trend="c", variance="hc1", hac_lags=None` | `beta`:f `effective_f`:f `f_classical`:f `f_hc1`:f `hac_lags`:∅ `mop_cv_tau10`:f `mop_cv_tau20`:f `mop_cv_tau30`:f `mop_cv_tau5`:f `n_proxy`:i `reliability`:f `se`:f `tau_bound`:f `weak_folklore`:b `weak_mop_tau10`:b | First-stage instrument-strength diagnostics for a proxy SVAR: the |
| `proxy_svar` | Structural identification (advanced) | `data, proxy, lags=2, horizon=12, norm_var=0, unit=1.0, trend="c", robust_f=True` | `cov_um`:1D `first_stage`:{} `first_stage_f`:f `impact`:1D `irf`:LL `n_proxy`:i `relative_impact`:1D `reliability`:f `shock`:1D | Proxy SVAR / external-instrument identification (SVAR-IV): one structural |
| `proxy_svar_bands` | structural identification | `data, proxy, lags=2, horizon=12, norm_var=0, unit=1.0, trend="c", alpha=0.1, n_boot=2000, seed=0, bands="moving_block", block_length=None, robust_f=True` | `alpha`:f `asymptotically_valid`:b `block_length`:i `failure_warning`:∅ `failures`:{} `first_stage_f_draws`:1D `gamma_norm_draws`:1D `lower`:LL `lower_efron`:LL `method`:s `n_boot`:i `n_failed`:i `n_proxy`:i `n_used`:i `point`:LL `point_first_stage_f`:f `point_gamma_norm`:f `point_reliability`:f `reliability_draws`:1D `rho_draws`:LL `se`:LL `upper`:LL `upper_efron`:LL `validity_note`:s | Confidence bands for a proxy (external-instrument) SVAR impulse response. |
| `pseudo_obs` | static copulas | `x` | → 2-D[f] | Pseudo-observations: the average-rank probability-scale transform. |
| `quantile_lp` | Quantile regression & growth-at-risk | `y, shock, taus=None, horizons=12, n_lag_controls=4` | `converged`:LL `horizons`:1Du `irf`:LL `se`:LL `taus`:1D | Quantile local projections: per-horizon check-loss IRFs of `y` to `shock` |
| `quantile_regression` | Quantile regression & growth-at-risk | `y, x, taus=None, se="robust"` | `bandwidth`:1D `bse`:LL `converged`:b `iterations`:L `params`:LL `sparsity`:1D `taus`:1D `tvalues`:LL | Linear quantile regression at one or many quantile levels. |
| `random_forest` | Trees and forests | `x, y, n_trees=500, max_features=…, max_depth=None, min_samples_leaf=5, bootstrap="iid", block_length=None, seed=0, x_test=None, quantiles=None, importance="none", importance_groups=None, permutation_block=None, n_permutations=None` | `fitted`:1D `importance`:∅ `importance_groups_resolved`:∅ `max_features_resolved`:i `n_trees`:i `oob_mse`:f `oob_prediction`:1D `predicted`:∅ `quantile_predictions`:∅ | Random forest for regression (Breiman 2001) with time-series-aware |
| `realized_measures` | Realized volatility | `returns` | `bipower`:f `jump`:f `rv`:f | Realized volatility measures on a vector of high-frequency returns. |
| `realized_quarticity` | Realized volatility | `returns` | → float | Realized quarticity `RQ = (n/3) sum r_i^4` (Barndorff-Nielsen & |
| `realized_range` | Realized volatility | `high, low, method="parkinson", open=None, close=None` | → float | Range-based daily variance from OHLC bars, summed across the supplied |
| `recession_probit` | Recession probability | `y, x, link="probit", dynamic=False` | `bse`:1D `converged`:b `loglik`:f `params`:1D `probabilities`:1D `pseudo_r2`:f `zstats`:1D | Recession-probability model: probit or logit of a binary recession |
| `regression_tree` | Trees and forests | `x, y, max_depth=None, min_samples_leaf=1, min_samples_split=2, x_test=None` | `depth`:i `feature_importance`:1D `fitted`:1D `n_leaves`:i `n_nodes`:i `predicted`:∅ `splits`:LL | CART regression tree (Breiman et al. 1984) with scikit-learn's best-split |
| `reset_test` | Specification & diagnostic tests | `y, x, max_power=3` | `df_den`:i `df_num`:i `fstat`:f `pvalue`:f | Ramsey RESET functional-form test: F-test of powers of the fitted values |
| `ridge` | Machine learning | `x, y, alpha` | → 1-D[f] | Ridge regression (closed form). Minimizes \|\|y - Xb\|\|^2 + alpha*\|\|b\|\|^2, |
| `robust_svar_bounds` | structural identification | `data, restrictions, lags=2, horizon=12, n_draws=500, seed=0, lambda1=0.2, alpha=0.1` | `alpha`:f `diagnostics`:{} `lower_quantiles`:LL `probs`:L `restricted_shocks`:L `robust_ci_lower`:LL `robust_ci_upper`:LL `set_lower_mean`:LL `set_upper_mean`:LL `upper_quantiles`:LL | Giacomini-Kitagawa (2021) prior-robust identified-set bounds for a sign- |
| `seasonal_strength` | filters | `y, period` | `period`:i `seasonal_strength`:f `trend_strength`:f | Wang-Smith-Hyndman (2006) seasonal and trend strength from a |
| `setar` | regime switching | `y, p, delay=1, trim=0.15, delays=None, ic="aic", constant=True` | `aic`:f `bic`:f `bse_high`:1D `bse_low`:1D `delay`:i `ic`:f `ic_used`:s `k`:i `min_regime`:i `n_high`:i `n_low`:i `nobs`:i `params_high`:1D `params_low`:1D `sigma2`:f `sigma2_high`:f `sigma2_low`:f `ssr`:f `ssr_path`:1D `threshold`:f `thresholds`:1D | Two-regime self-exciting threshold autoregression (SETAR; Tong-Lim |
| `setar_test` | regime switching | `y, p, delay=1, trim=0.15, n_boot=499, seed=0` | `boot_stats`:1D `delay`:i `f_path`:1D `n_boot`:i `nobs`:i `p_value`:f `ssr_linear`:f `ssr_setar`:f `stat`:f `threshold`:f `thresholds`:1D | Hansen (1996) sup-F test of linearity against a two-regime SETAR(p) |
| `sign_restricted_svar` | VAR / SVAR | `data, restrictions, lags=2, horizon=12, n_draws=500, max_tries=400, seed=0, lambda1=0.2` | `diagnostics`:{} `probs`:L `quantiles`:LL `set_max`:LL `set_min`:LL | Sign-restricted Bayesian SVAR (Uhlig 2005; Rubio-Ramirez-Waggoner-Zha |
| `smooth_lp` | Local projections | `y, shock, horizons=12, n_lag_controls=4, lam=None, degree=3, n_basis=None, penalty_order=2, lambda_grid=None, n_folds=5, hac_maxlags=None, band=None, band_alpha=0.1, band_seed=20260807, band_n_sim=100000` | `cv_grid`:1D `cv_scores`:1D `horizons`:1Du `irf`:1D `irf_raw`:1D `lambda_used`:f `se`:1D `se_raw`:1D `theta`:1D | Smooth local projections (Barnichon-Brownlees 2019): the IRF path is |
| `spread_zscore` | cointegration | `x, kappa=None, mu=None, sigma=None, dt=None` | `fitted`:b `kappa`:f `mu`:f `sigma`:f `stationary_sd`:f `zscore`:1D | Z-score of a spread against the stationary Ornstein-Uhlenbeck law |
| `star` | regime switching | `y, p, model="lstar", delay=1, trim=0.15, delays=None, constant=True, n_gamma=25, n_c=25` | `aic`:f `best_cell`:tuple[int,int] `bic`:f `bse_linear`:1D `bse_nonlinear`:1D `c`:f `converged`:b `delay`:i `fevals`:i `gamma`:f `gamma_at_boundary`:b `gamma_standardized`:f `grid_c`:1D `grid_gamma`:1D `k`:i `loglik`:f `model`:s `nobs`:i `params_linear`:1D `params_nonlinear`:1D `s_sd`:f `se_c`:f `se_gamma`:f `se_valid`:b `sigma2`:f `ssr`:f `ssr_grid`:2D `transition`:1D | Smooth-transition autoregression (STAR; Terasvirta 1994), logistic or |
| `star_eval` | regime switching | `y, p, gamma, c, model="lstar", delay=1, constant=True` | `aic`:f `bic`:f `bse_linear`:1D `bse_nonlinear`:1D `k`:i `loglik`:f `nobs`:i `params_linear`:1D `params_nonlinear`:1D `se_c`:f `se_gamma`:f `se_valid`:b `sigma2`:f `ssr`:f `transition`:1D | The concentrated STAR fit at FIXED transition parameters `(gamma, c)` |
| `star_test` | regime switching | `y, p, delay=1, delays=None` | `best`:i `delay`:i `h1_f_stat`:f `h1_p_value`:f `h2_f_stat`:f `h2_p_value`:f `h3_f_stat`:f `h3_p_value`:f `k0`:i `lm3_f_p_value`:f `lm3_f_stat`:f `lm3_p_value`:f `lm3_stat`:f `nobs`:i `q`:i `ssr0`:f `ssr1`:f `ssr2`:f `ssr3`:f `suggested`:s `tests`:L{} | The Terasvirta STAR modeling-cycle battery: the LM3 linearity test |
| `stl` | filters | `y, period, seasonal=7, trend=None, low_pass=None, seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=False, seasonal_jump=1, trend_jump=1, low_pass_jump=1, inner_iter=None, outer_iter=None` | `config`:{} `period`:i `resid`:1D `seasonal`:1D `trend`:1D `weights`:1D | STL seasonal-trend decomposition using LOESS (Cleveland et al. 1990). |
| `structural_fevd` | structural identification | `data, lags=2, horizon=10, trend="c", impact=None, sigma="dfadj"` | `fevd`:LL `impact`:LL | Structural forecast-error variance decomposition for an arbitrary structural |
| `summarize` | unit roots / workflow | `obj, title=None, wrap="auto"` | `a`:f `b`:1D | Return a renderable results object for any tsecon output. |
| `sup_f_test` | Structural breaks | `y, x, trim=0.15` | `break_date`:i `dates`:1Du `f_path`:1D `h`:i `p_value`:f `stat`:f | Andrews (1993) sup-F (Quandt) test for a single structural break at an |
| `svensson` | Term structure | `maturities, yields, lambda1, lambda2` | `factors`:1D `lambda1`:f `lambda2`:f `residuals`:1D `rsquared`:f | Svensson (1994) four-factor yield-curve fit. |
| `theta_forecast` | Forecasting | `y, steps, period=1` | → 1-D[f] | The Theta method (Assimakopoulos-Nikolopoulos 2000) — a stubbornly hard |
| `threshold_var` | regime switching | `data, p, threshold_index=0, delay=1, trim=0.1, delays=None, constant=True` | `aic`:f `bic`:f `bse_high`:LL `bse_low`:LL `delay`:i `llf`:f `log_det_sigma`:f `logdet_path`:1D `min_regime`:i `n_high`:i `n_low`:i `n_regressors`:i `neqs`:i `nobs`:i `params_high`:LL `params_low`:LL `sigma`:LL `sigma_high`:LL `sigma_low`:LL `threshold`:f `threshold_index`:i `thresholds`:1D | Two-regime threshold VAR (the multivariate SETAR; Tong 1983; Tsay |
| `threshold_var_test` | regime switching | `data, p, threshold_index=0, delay=1, trim=0.1, n_grid=300, n_boot=499, seed=0, constant=True` | `boot_stats`:1D `delay`:i `min_regime`:i `n_boot`:i `n_regressors`:i `neqs`:i `nobs`:i `p_value`:f `stat`:f `threshold`:f `threshold_index`:i `thresholds`:1D `wald_path`:1D | Robust sup-Wald (score-form) test of a linear VAR(p) against the |
| `threshold_vecm` | cointegration | `data, k_ar_diff=1, trim=0.05, n_grid_gamma=300, n_grid_beta=None, beta_span=None, beta=None` | `beta`:1D `beta_grid`:1D `beta_linear`:1D `bse_high`:LL `bse_low`:LL `ect`:1D `frac_low`:f `k_ar_diff`:i `llf`:f `llf_linear`:f `log_det_sigma`:f `min_regime`:i `n_high`:i `n_low`:i `n_regressors`:i `neqs`:i `nobs`:i `params_high`:LL `params_low`:LL `sigma`:LL `threshold`:f | Hansen-Seo (2002) two-regime threshold VECM (threshold cointegration): |
| `tripower_quarticity` | Realized volatility | `returns` | → float | Tripower quarticity |
| `umidas` | Nowcasting & MIDAS | `y, hf_lags, se_type="hac", maxlags=None` | `bse`:1D `params`:1D `rsquared`:f | U-MIDAS: unrestricted mixed-frequency regression (= OLS of `y` on a |
| `var_backtest` | forecast comparison | `returns_or_hits, var_forecasts=None, alpha=0.05, dq_lags=4, input="auto"` | `alpha`:f `dq_df`:i `dq_includes_var`:b `dq_lags`:i `dq_stat`:f `dq_var_dropped`:b `expected_violations`:f `hit_rate`:f `lr_cc`:f `lr_ind`:f `lr_uc`:f `n`:i `n00`:i `n01`:i `n10`:i `n11`:i `n_violations`:i `p_cc`:f `p_dq`:f `p_ind`:f `p_uc`:f `pi01`:f `pi11`:f `verdict`:s | The VaR backtest battery: Kupiec (1995) unconditional coverage, |
| `var_fevd` | VAR / SVAR | `data, lags=2, horizon=10, trend="c"` | → list[list] | Forecast-error variance decomposition: `fevd[h][i][j]` is the share of |
| `var_fit` | VAR / SVAR | `data, lags=2, trend="c"` | `aic`:f `bic`:f `df_resid`:i `fitted`:LL `hqic`:f `is_stable`:b `llf`:f `max_root`:f `min_root`:f `nobs`:i `params`:LL `resid`:LL `sigma_u`:LL | Fit a VAR(p) by OLS and return estimates, fit statistics, residuals, |
| `var_forecast` | VAR / SVAR | `data, lags=2, steps=8, alpha=0.05, trend="c", band="pointwise", band_scope="all", band_seed=20260807, band_n_sim=100000` | `band`:s `lower`:LL `point`:LL `upper`:LL | Iterated VAR point forecasts with (innovation-uncertainty) intervals. |
| `var_granger` | VAR / SVAR | `data, caused, causing, lags=2, trend="c"` | `df_den`:i `df_num`:i `p_value`:f `statistic`:f | Granger-causality F test: do the `causing` variables help predict the |
| `var_irf` | VAR / SVAR | `data, lags=2, horizon=10, orth=True, trend="c", cumulative=False` | → list[list] | Impulse responses of a fitted VAR: `irfs[h][i][j]` is the response of |
| `var_irf_bands` | VAR / SVAR | `data, lags=2, horizon=10, orth=True, method="asymptotic", alpha=0.1, cumulative=False, n_boot=1000, seed=0, trend="c", bias_correct=False, band="pointwise", band_scope="horizon", band_seed=20260807, band_n_sim=100000` | `alpha`:f `band`:s `bias_correct`:b `lower`:LL `method`:s `n_boot`:∅ `point`:LL `se`:LL `upper`:LL | Frequentist confidence bands on VAR impulse responses — the banded |
| `vecm` | Cointegration & regimes | `data, k_ar_diff=1, coint_rank=1, deterministic="n", seasons=0, first_season=None` | `alpha`:LL `beta`:LL `det_coef`:LL `det_coef_coint`:L0 `gamma`:LL `llf`:f `sigma_u`:LL | VECM maximum-likelihood estimation at a given cointegrating rank |
| `weighted_midas` | Nowcasting & MIDAS | `y, hf_lags, scheme="exp_almon", weight_start=None` | `converged`:b `fitted`:1D `intercept`:f `iterations`:i `residuals`:1D `rsquared`:f `scheme`:s `slope`:f `ssr`:f `weight_params`:1D `weights`:1D | Weighted MIDAS regression fit by nonlinear least squares (Ghysels, |
| `welch` | Spectral analysis | `x, nperseg=256, fs=1.0, noverlap=None, window="hann", detrend="constant"` | `freqs`:1D `psd`:1D | Welch's averaged-periodogram PSD (periodic Hann, 50% overlap by |
| `zero_sign_svar` | VAR / SVAR | `data, sign_restrictions, zero_restrictions, lags=2, horizon=12, n_draws=500, max_tries=400, seed=0, lambda1=0.2, weighted=True` | `arw_weighted`:b `diagnostics`:{} `ess`:f `probs`:L `quantiles`:LL `set_max`:LL `set_min`:LL `weights`:L | Zero + sign restricted Bayesian SVAR (Rubio-Ramirez-Waggoner-Zha 2010 |
| `zivot_andrews` | unit roots / workflow | `y, regression="c", trim=0.15, max_lags=None, autolag=…, lags=None` | `break_index`:i `crit`:{} `lags`:i `nobs`:i `pvalue`:f `regression`:s `stat`:f `trim`:f | Zivot-Andrews unit-root test with one endogenous break (null: unit |
