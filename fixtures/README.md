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
  the generator's docstring.
- **Transformations of two public-domain reference series** loaded from
  statsmodels' bundled datasets:
  - the **Nile** annual river-flow series (`sm.datasets.nile`), a classic
    public-domain series (1871–1970);
  - **US macrodata** (`sm.datasets.macrodata`), public-domain US-government
    (BEA/FRED) economic data.

  Only *statistics and transformations* of these (e.g. `100·log(realgdp)`,
  100× dlog growth rates, and fitted model outputs) are stored — no raw
  licensed dataset is redistributed.

The `*.csv` files are the exception, and are data rather than derived values:
public datasets vendored **with attribution** for the replication pages
(`ramey_zubairy.csv` — the authors' public replication archive;
`yield_curve_recession.csv` — FRED series GS10/TB3MS/USREC;
`sunspots_tong.csv` — the public-domain annual Wolf sunspot numbers
1700–1988, via `statsmodels.datasets.sunspots`;
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
