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

Non-compliant or partial functions:
- `ar_loglik` [partial]: nan=SILENT; empty=ok; ndim=ok; string=n/a; negative=n/a
- `cg_regression` [partial]: nan=ValueError(no arg/fix: 'non-finite value (NaN or infinity) in regression response; the survey estimators'); empty=ok; ndim=ok; string=n/a; negative=ok
- `check_series` [partial]: nan=ok; empty=ok; ndim=SILENT; string=n/a; negative=ok
- `dfm_news` [partial]: nan=SILENT; empty=ValueError(no arg/fix: 'empty input: training panel is empty'); ndim=ok; string=n/a; negative=ok
- `dfm_nowcast` [partial]: nan=SILENT; empty=ValueError(no arg/fix: 'empty input: training panel is empty'); ndim=ok; string=ok; negative=ok
- `dsge_solve` [partial]: nan=ValueError(no arg/fix: 'non-finite value (NaN or infinity) in A; clean the model matrices before solving'); empty=ok; ndim=ok; string=n/a; negative=ok
- `factor_model` [partial]: nan=ValueError(no arg/fix: 'non-finite value (NaN or infinity) in panel x'); empty=ValueError(no arg/fix: 'empty input: panel x'); ndim=ok; string=n/a; negative=ok
- `forecast_efficiency` [partial]: nan=ValueError(no arg/fix: 'non-finite value (NaN or infinity) in regression response; the survey estimators'); empty=ok; ndim=ok; string=n/a; negative=ok
- `gmm_nonlinear` [partial]: nan=ValueError(no arg/fix: 'optimizer error: input `x0` contains NaN or infinity'); empty=ok; ndim=TypeError(no arg/fix: 'only 0-dimensional arrays can be converted to Python scalars'); string=n/a; negative=n/a
- `hetero_svar` [partial]: nan=ok; empty=ValueError(no arg/fix: 'regime_labels length 200 != number of observations 0'); ndim=ok; string=ok; negative=ok
- `iv_gmm` [partial]: nan=ValueError(no arg/fix: 'non-finite value (NaN or infinity) in regressor columns X; GMM estimators do not'); empty=ok; ndim=ok; string=ok; negative=ok
- `kernel_regression` [partial]: nan=ok; empty=ok; ndim=SILENT; string=ok; negative=ok
- `kernel_ridge` [partial]: nan=ok; empty=ok; ndim=SILENT; string=ok; negative=ok
- `local_level_smooth` [partial]: nan=SILENT; empty=ok; ndim=ok; string=n/a; negative=n/a
- `optimal_block_length` [partial]: nan=ValueError(no arg/fix: 'series contains NaN or infinite values'); empty=ValueError(no arg/fix: 'cannot resample an empty sample (n = 0)'); ndim=ok; string=n/a; negative=n/a
- `summarize` [none]: nan=n/a; empty=n/a; ndim=n/a; string=SILENT; negative=n/a
- `welch` [partial]: nan=ok; empty=ValueError(no arg/fix: 'the Welch segment length nperseg = 64 exceeds the sample size 0, so not even one'); ndim=ok; string=ok; negative=ok