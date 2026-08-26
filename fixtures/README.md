# Golden fixtures

Every estimator in `tsecon` is gated against a **golden fixture** in this
directory: a JSON file of reference values that the Rust (and Python) tests
must reproduce to a tight tolerance. This is how the library stays honest —
nothing lands without a target it has to hit.

## What the fixtures contain

The `*.json` files hold only **derived numeric values** — never a redistributed
dataset. Each is produced by a `generate_*.py` script (run with the project
venv) in one of two ways:

- **Simulated data**: seeded NumPy `default_rng` draws through a known
  data-generating process, plus the reference output computed either by an
  independent library (statsmodels, SciPy, `arch`, `linearmodels`,
  scikit-learn, ArviZ) or by a documented closed-form formula transcribed in
  the generator's docstring. One fixture (`lpdid.json`) additionally requires
  R with the `fixest` package on PATH: its generator shells out to the
  committed `generate_lpdid_fixtures.R`, which runs the LP-DiD reference
  conventions through fixest (the engine of the authors' own example code)
  and cross-checks them against the generator's independent NumPy
  reimplementation before anything is stored. `bn_filters.json` similarly
  requires R plus `$BNFILTER_R_DIR` pointing at a checkout of the
  Kamber-Morley-Wong replication code (bnfiltering.com lineage, packaged at
  `github.com/kletts/bnfilter` — sourced at generation time, **not vendored**:
  its DESCRIPTION carries no license grant); its generator reference-runs the
  authors' own `BN_Filter`/`select_delta`/`BN_Filter_stderr` and cross-checks
  an independent NumPy reimplementation against the R output at 1e-9 before
  anything is stored.
- **Transformations of two public-domain reference series** loaded from
  statsmodels' bundled datasets:
  - the **Nile** annual river-flow series (`sm.datasets.nile`), a classic
    public-domain series (1871–1970);
  - **US macrodata** (`sm.datasets.macrodata`), public-domain US-government
    (BEA/FRED) economic data.

  Only *statistics and transformations* of these (e.g. `100·log(realgdp)`,
  100× dlog growth rates, and fitted model outputs) are stored — no raw
  licensed dataset is redistributed.

One fixture is deliberately **not** a third-party golden:
`backtest_string_snapshot.json` (generator
`generate_backtest_string_snapshot.py`) is a *self-snapshot* of the
string-forecaster paths of `backtest`/`conformal_forecast`/
`conformal_backtest`, captured — as `float.hex()` values, so the comparison
is bitwise — from the build immediately before the Python-callable
forecaster plumbing landed in 0.6.0-dev. Its job is regression, not
validation: `test_backtest_callable.py` asserts the pre-existing string
surfaces stayed bit-identical. Regenerate it only to re-baseline after an
*intentional* behavioral change to those paths.

The `*.csv` files are the exception, and are data rather than derived values:
public datasets vendored **with attribution** for the replication pages
(`ramey_zubairy.csv` — the authors' public replication archive;
`gertler_karadi.csv` — the Gertler-Karadi (2015) AEJ replication dataset via
the Plagborg-Møller & Wolf `svma_iv` mirror, cross-checked against the
VAR-Toolbox mirror;
`yield_curve_recession.csv` — FRED series GS10/TB3MS/USREC;
`sunspots_tong.csv` — the public-domain annual Wolf sunspot numbers
1700–1988, via `statsmodels.datasets.sunspots`;
`glp_sw_panel.csv` — the Stock-Watson (2008) US quarterly panel exactly as
Giannone-Lenza-Primiceri (2015)'s own replication code consumes it, vendored
from the public FRBNY-DSGE/BrookingsPC2020 GitHub mirror of their web
replication files with the mirror's redistribution notice kept in the header;
`gsw_nss_params.csv` — the Gürkaynak-Sack-Wright NSS US Treasury curve
parameters, Federal Reserve Board public data, monthly 1961-2014;
`acm_published_10y.csv` — the NY Fed's published ACM 10-year term-premium
decomposition, quarterly, a level/shape validation target). Each carries its
source in its header comments.

Each fixture records the exact reference-library versions used, so the values
are reproducible. Regenerate any of them with, e.g.:

```sh
.venv/bin/python fixtures/generate_fixtures.py
```
