# Datasets

`tsecon.datasets` provides reference macro data on a **download-on-first-use**
policy. Nothing large is vendored into the repository and nothing is fetched at
import time: the first call downloads to a local cache, later calls read the
cache. Every loader records the source URL and a SHA-256 of the bytes it parsed,
so a dataset can be pinned and audited.

Only the standard library and NumPy are used — no `requests`, no `pandas`
requirement.

```python
from tsecon import datasets as ds

gs10 = ds.fred_series("GS10")     # 10-year Treasury, monthly
md   = ds.fred_md()               # the FRED-MD macro panel
```

---

## Cache

| | |
|---|---|
| Default location | `~/.cache/tsecon` (respects `XDG_CACHE_HOME`) |
| Override | set `TSECON_DATA_DIR` |
| Inspect | `ds.cache_dir()` |
| Clear | `ds.clear_cache()` → number of files removed |

Every loader accepts `refresh=True` to re-download, and `local_path=...` to
parse a file you already have — which is also how you work fully offline.

**Offline behaviour.** With no cache and no network the loader raises a
`RuntimeError` that names the URL it could not reach and tells you to use
`local_path=` or point `TSECON_DATA_DIR` at a directory that already holds the
file. It does not silently return empty data.

---

## `fred_series` — a single FRED series

**What it does.** Downloads one series from FRED's public CSV endpoint
(`fredgraph.csv`). **No API key is required.** If `FRED_API_KEY` is set in your
environment it is passed along — useful for institutional keys — but the loader
works without it, and the key is never written to disk or into the repository.

**Returns** a dict: `series_id`, `dates` (`datetime64[D]`), `values` (float,
missing observations as `nan` — FRED writes `.` for these), `nobs`, `source`,
`url`, `sha256`.

```python
from tsecon import datasets as ds

gs10 = ds.fred_series("GS10")
print(gs10["nobs"], gs10["dates"][0], gs10["values"][-1])
# 879 1953-04-01 4.47
```

Combine a few series into a panel for a VAR:

```python
import numpy as np
from tsecon import datasets as ds
import tsecon

ids = ["GS10", "GS2", "UNRATE"]
series = [ds.fred_series(i) for i in ids]
# align on the common date range, then stack into T x k
common = set(series[0]["dates"])
for s in series[1:]:
    common &= set(s["dates"])
dates = np.array(sorted(common))
panel = np.column_stack([
    s["values"][np.isin(s["dates"], dates)] for s in series
])
fit = tsecon.var_fit(panel, lags=2)
```

---

## `fred_md` — the FRED-MD monthly macro panel

**What it does.** Loads the FRED-MD panel of **McCracken & Ng (2016)** — the
standard large monthly macro dataset for factor models, FAVARs, and nowcasting.

!!! warning "The widely-cited FRED-MD URL is dead"
    The URL quoted in most tutorials and several packages,
    `files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv`, now returns
    **403 AccessDenied**. This loader uses the live `www.stlouisfed.org` media
    path, verified working. If the provider moves it again, the loader raises an
    error saying so rather than failing obscurely.

**Two file quirks the loader handles for you.**

1. **The `Transform:` row.** FRED-MD's second line is not data — it is
   McCracken-Ng's per-series transformation codes. It is parsed out into
   `transform_codes` rather than being read as an observation.
2. **The ragged edge.** The most recent months are only partly published, so
   trailing rows can be entirely missing. `drop_empty_rows=True` (the default)
   removes rows with no observations at all; genuine within-panel gaps stay as
   `nan`.

**Returns** a dict: `dates` (`datetime64[D]`), `names` (list of series ids),
`data` (`T × k` float array, missing as `nan`), `transform_codes` (int array
aligned to `names`), plus `source`, `url`, `sha256`.

```python
md = ds.fred_md()
md["data"].shape        # (800, 126)  — grows as vintages are published
md["names"][:3]         # ['RPI', 'W875RX1', 'DPCERA3M086SBEA']
md["transform_codes"][:3]
```

---

## `apply_fred_md_transforms` — the McCracken-Ng codes

FRED-MD is published in **levels**; the transformation codes tell you how to
render each series stationary before use. Applying them is not optional — factor
estimates on untransformed levels are meaningless.

| code | transformation |
|---|---|
| 1 | level (no change) |
| 2 | first difference `Δx` |
| 3 | second difference `Δ²x` |
| 4 | `log x` |
| 5 | `Δ log x` |
| 6 | `Δ² log x` |
| 7 | `Δ(x_t / x_{t-1} − 1)` |

```python
md = ds.fred_md()
X  = ds.apply_fred_md_transforms(md["data"], md["transform_codes"])
```

Differencing costs leading observations, so the transformed panel keeps its
original shape with leading `nan`s — column alignment with `names` and
`transform_codes` is preserved. Drop or impute those rows before estimating.

**Failure modes.** A code outside 1–7 raises rather than silently passing the
series through. A `codes` array whose length does not match the number of
columns raises — this is the common mistake after subsetting columns without
subsetting the codes alongside them.

---

## `ramey_zubairy` — the RZ (2018) historical macro dataset

**What it does.** Loads the quarterly US dataset behind Ramey & Zubairy's
government-spending multiplier estimates: 564 quarters (1875Q1–2015Q4)
including Ramey's **military-news shock** (`news`), nominal government spending,
GDP, the deflator, and CBO potential output. The core variables are jointly
available from 1890Q1.

The file lives inside the authors' replication zip (~750 kB). The loader
downloads it once, caches the archive, and extracts the CSV — nothing is
vendored into this repository.

**Returns** `quarter` (float, `2015.75` = 2015Q4), `names`, `data` (`T × k`,
missing as `nan`), a `series` dict mapping each name to its column, plus
`source`, `url`, `sha256`.

```python
from tsecon import datasets as ds

rz = ds.ramey_zubairy()
rz["series"]["news"].shape      # (564,)
rz["quarter"][-1]                # 2015.75
```

This dataset drives the
[Ramey-Zubairy replication](../examples/replication-ramey-zubairy.md), which
recovers integral multipliers of 0.66–0.71 — inside the published 0.6–0.8 range.

**Citation.** Ramey, V. A. & Zubairy, S. (2018), "Government Spending Multipliers
in Good Times and in Bad: Evidence from US Historical Data," *Journal of
Political Economy* 126(2):850-901.

---

## Licensing and citation

FRED series are US federal-government data redistributed by the Federal Reserve
Bank of St. Louis; consult their terms of use. Cite **McCracken, M. W. & Ng, S.
(2016), "FRED-MD: A Monthly Database for Macroeconomic Research", *Journal of
Business & Economic Statistics* 34(4):574-589** when you use FRED-MD.

Because nothing is vendored, the repository redistributes no third-party data —
each user downloads under the provider's own terms.
