# tsecon

**High-performance time series econometrics: a Rust core with a Python-first API.**

> Pre-1.0 software under active development. The name is settled — `tsecon` is
> what you install and import — but the API may still change before 1.0.

`tsecon` brings the scattered toolkit of modern time series econometrics —
diagnostics and specification tests, ARIMA, GARCH, VARs and structural
identification, local projections, Bayesian VARs, nowcasting, predictive
regressions, panels, the term structure — into one fast, validated, Python-first
library. The compute core is written from scratch in Rust (no BLAS, no heavy
dependencies) so bootstrap inference and Monte Carlo work that is painfully slow
elsewhere runs in seconds, with results bit-reproducible at any thread count.

Every estimator is **validation-gated**: nothing ships without a named golden
target — a reference implementation (statsmodels, SciPy, NumPy, `arch`,
`linearmodels`, scikit-learn, ArviZ), a documented closed form, or, where no
third-party implementation was reachable, an independent transcription of the
published algorithm backed by a seeded Monte-Carlo size/power check. The
[validation matrix](https://github.com/cacoleman16/tsecon/blob/main/docs/reference/validation-matrix.md)
names the target, fixture, and tolerance family by family and says plainly
which ones have no third-party reference; a cross-library parity gate
re-verifies the agreements that do exist in CI on every push. The
library ships **no data loaders and makes no network calls** — the only runtime
dependency is NumPy.

## Install

```sh
pip install tsecon
```

A single self-contained wheel whose only runtime dependency is NumPy — no Rust
toolchain, no system BLAS. Prebuilt wheels cover Linux (`x86_64`, `aarch64`),
macOS on Apple Silicon (`arm64`), and Windows (`x64`), for every Python >= 3.9.
Plotting is an optional extra:

```sh
pip install 'tsecon[plots]'    # adds matplotlib for the .plot_*() methods
```

Then check what you got:

```python
import tsecon
print(tsecon.__version__)                                       # 0.7.0
print(sum(callable(getattr(tsecon, n)) for n in dir(tsecon)     # 162
          if not n.startswith("_")))
```

There is no prebuilt wheel for Intel macOS (`x86_64`); on that platform, and for
building from a checkout, pip compiles from the source distribution, which needs
a [Rust toolchain](https://rustup.rs/) and Python >= 3.9. Contributors build
with [maturin](https://www.maturin.rs/):

```sh
pip install maturin
maturin develop --release -m bindings/python/Cargo.toml   # builds + installs into the active venv
```

## A taste

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)
y = np.cumsum(rng.standard_normal(300))          # a random walk

tsecon.check_stationarity(y)["recommendation"]   # -> "Difference"

# Fit a VAR, read off impulse responses, decompose the variance
data = rng.standard_normal((200, 3))
irf  = tsecon.var_irf(data, lags=2, horizon=16)          # [h][response][shock]
fevd = tsecon.var_fevd(data, lags=2, horizon=16)   # [h][variable][shock]

# Robust SEs, exact-MLE ARIMA with a forecast fan, GARCH with robust SEs,
# a Ramey-Zubairy fiscal multiplier, a Bayesian VAR — on stand-in inputs,
# so the whole block runs as written; swap in your own arrays.
X          = np.column_stack([np.ones(300), rng.standard_normal(300)])
returns    = rng.standard_normal(1500)
spending   = rng.standard_normal(300)
instrument = spending + rng.standard_normal(300)

tsecon.ols(y, X, se_type="hac")
tsecon.arima_fit(y, p=1, d=1, q=1, forecast_steps=12, conf_alpha=0.1)
tsecon.garch_fit(returns, vol="gjr", dist="t")
tsecon.lp_multiplier(y, spending, instrument, horizons=20)
tsecon.bvar_irf_draws(data, lags=2, horizon=12, n_draws=800)
```

Every function takes plain NumPy arrays and returns arrays or dicts of
documented keys. An opt-in `tsecon.results` layer wraps the same calls in
objects that render themselves (`.summary()`, `.plot_irf()`) without changing
the dict contract.

## What's here today

162 functions across the full applied workflow: the diagnostic battery and
unit-root workflow (ADF, KPSS, DF-GLS, Phillips-Perron, Ng-Perron,
Zivot-Andrews, the `ndiffs`/`nsdiffs` advisors, `check_stationarity`, and the
one-call `check_series` battery with model recommendations); specification and
stability tests (White, Breusch-Pagan, RESET, Chow, CUSUM) and Bai-Perron
multiple breaks; robust and HAC standard errors; the bootstrap family; an
exact-diffuse Kalman filter; ARIMA — seasonal via `seasonal=(P, D, Q, s)` and
automatically ordered by `auto_arima` — GARCH/GJR/EGARCH, and GAS score-driven
volatility; VAR/SVAR with sign-restricted identification, FAVAR, and
Diebold-Yilmaz connectedness; local projections (lag-augmented, LP-IV,
state-dependent, smooth, quantile, panel, LP-DiD, and the Ramey-Zubairy
integral multiplier); a Minnesota-NIW Bayesian VAR alongside hierarchical and
SSVS priors; GMM and IV-GMM; predictive regressions with IVX inference;
heterogeneous panels (mean group, CCE-MG, PMG) and panel unit-root tests;
cointegration (Johansen, VECM across every statsmodels deterministic case plus
centered seasonal dummies, Engle-Granger, Phillips-Ouliaris) and regime
dynamics (Markov switching, SETAR, STAR, threshold VAR, threshold VECM); MIDAS
and DFM nowcasting with a news decomposition; multivariate GARCH; realized
volatility; trend-cycle filters (HP, Baxter-King, Christiano-Fitzgerald,
Hamilton, Beveridge-Nelson); STL/MSTL seasonal decomposition; spectral
analysis; long memory; extreme-value tails and copulas; conformal forecast
intervals and a leakage-checked backtest engine; leakage-safe penalized
regression; growth-at-risk; recession-probability models;
survey-expectations tools; the Nelson-Siegel/Svensson term structure with the
arbitrage-free (AFNS) adjustment; and a linear rational-expectations
(DSGE-lite) solver.

The library ships with complete type stubs (`py.typed`), so autocomplete and
type checking work out of the box.

## Learn more

The full documentation — a 15-chapter guide to time series econometrics, model
cards for every estimator family, a worked figure gallery, eight replications
of published results, a Monte Carlo validation suite, and an honest benchmark
harness — lives in the [project repository](https://github.com/cacoleman16/tsecon).

## License

MIT OR Apache-2.0.
