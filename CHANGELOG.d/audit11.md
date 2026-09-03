### Fixed — documentation contract (audit round 11)

- **Five structural-identification functions' `help()` text carried none of
  their return contract** — `robust_svar_bounds`, `fry_pagan_svar`,
  `hetero_svar`, `historical_decomposition`, `narrative_svar` had one- or
  two-line runtime docstrings while the stub carried the full key list; on
  the binding surface `robust_svar_bounds` named none of its ten keys and
  its NaN-for-unrestricted-shocks convention existed only in the stub. The
  runtime docstrings now carry the contract; a regression test asserts
  every returned key is named in `fn.__doc__`.
- **`gpd_fit` / `gev_fit` runtime docstrings gained the stub's `Keys:`
  line** (7 and 6 returned keys were unnamed at `help()`).
- **Twenty-two further runtime docstrings now name every returned key the
  stub already named** (`adaptive_lasso`, `check_series`, `connectedness`,
  `factor_model`, `flp`, `flp_scenario`, `functional_pca`, `fvar_scenario`,
  `iv_gmm`, `ivx_test`, `jarque_bera`, `johansen`, `nongaussian_svar`,
  `panel_lp`, `panel_pmg`, `panel_unit_root`, `proxy_first_stage`,
  `quantile_regression`, `setar`, `smooth_lp`, `sup_f_test`,
  `var_backtest`); `proxy_first_stage`'s `mop_cv_tau20`/`mop_cv_tau30`
  were named on no surface before.
- **`max_rel_change` is documented** on `lasso`, `elastic_net` and
  `adaptive_lasso` (docstring + stub): the scale-free
  `max_j |Δb_j|·‖x_j‖/‖y‖` the stopping rule compares with `tol`, returned
  since the convergence fix but named nowhere.
- **`local_level_smooth`'s six keys, the HP/Baxter-King/Christiano-
  Fitzgerald filters' `trend`/`cycle`/`first_index`, and the unnamed keys
  of `engle_granger`, `factor_model` (`er_ratios`), `gas_volatility`
  (`converged`/`iterations`) and `zero_sign_svar` (`arw_weighted`)** are
  now named in docstring and stub.
- **`cg_regression`'s docstring named `se`/`t`/`p`, which are not keys**;
  it now names the real ones (`se_intercept`/`se_slope`, `t_slope`,
  `p_slope`).
- **`seed=None` is documented as seed 0, not fresh entropy**, on
  `conformal_forecast`/`conformal_backtest` (`seed`, EnbPI) and
  `proxy_ar_sets` (`rf_seed`); measured `None ≡ 0 ≠ 1` on all three and
  pinned. `n_boot=None` (25) and `rf_draws=None` (256) are stated too.
- **The EGARCH multi-step forecast refusal is documented and clean.**
  `vol="egarch"` accepts `forecast_horizon` 0 or 1 only in `garch_fit`,
  `ccc_garch` and `dcc_garch` (no closed-form multi-step EGARCH forecast
  exists; the simulation route is not shipped) — previously the stub
  promised `covariance_forecast` for any `forecast_horizon > 0` and the
  refusal read `... require simulation (TODO(phase0)) ...`. The message now
  states the limit and the remedies (`forecast_horizon=1`, or
  `vol="garch"`/`"gjr"`); the three docstrings, the stub and the
  volatility card state it; a test pins horizon 1 working, horizon 2
  raising, and no internal marker in the text.
- **The forecasting card's `backtest` table** showed `period` defaulting
  to `1` (an explicit `period` raises for non-seasonal forecasters since
  0.7.0) and `forecaster` to `"naive"` (the default is `None`); both rows
  now show the signature defaults and the `period` row names the refusal
  and `insample_period`.
- **The panel card's `panel_lp` / `lp_did` key lists** include the stamped
  settings keys the docstrings document.
- **Two imprecise shape claims**: `dfm_nowcast.smoothed_factors` has one
  row per balanced-panel observation (T minus the ragged-edge rows), not
  "(T, r)"; `proxy_svar.shock` has T − lags rows, not "length T" (stub).
- **`cv_splits`' docstring and stub say `train` defaults to 0, which the
  walk-forward schemes refuse** (the ML card already did) — a default call
  `cv_splits(n)` always raised without either surface saying why.
