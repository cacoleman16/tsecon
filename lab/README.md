# lab/ — tsecon's private research lab

Frontier forecasting methods **under evaluation**. Nothing in this
directory is part of the released tsecon library: it is excluded from the
wheel, the docs site, and the public API, carries no API-stability
promises, and is held to a *research* bar (seeded experiments, honest
writeups), not the library's golden-fixture validation bar. A method
leaves the lab only via the normal validation-gated path — a named,
runnable golden target and a Rust implementation behind the standard
`Spec -> fit() -> Results` grammar (see `ROADMAP.md`, section 0, "Scoped
next work" for the grading language).

## Directory map

```
lab/
├── prophet_lite/        # from-scratch Taylor-Letham (2018) decomposable
│   │                    # forecaster: piecewise-linear trend + L1 (Laplace-
│   │                    # prior) changepoints, Fourier seasonality,
│   │                    # changepoint-bootstrap intervals
│   ├── model.py         #   design matrices + exact MAP solver (FWL + CD lasso)
│   ├── uncertainty.py   #   Prophet interval scheme (future-changepoint bootstrap)
│   ├── api.py           #   fit() -> ProphetLiteResult (tsecon-style dict subclass)
│   ├── tests.py         #   8 seeded pytest tests
│   └── README.md
├── laplace/             # Laplace-family robust & quantile methods
│   ├── al_gas.py        #   AL score-driven dynamic quantile (adaptive CAViaR
│   │                    #   w/ mean reversion; Catania-Luati family)
│   ├── robust_filter.py #   DCS local level: Student-t (Harvey-Luati),
│   │                    #   Laplace/sign, Gaussian (= steady-state Kalman)
│   ├── al_arima.py      #   LAD/median ARMA by CSS (Davis-Dunsmuir / Ling)
│   ├── tests.py         #   7 seeded pytest tests
│   └── README.md
├── experiments/         # the end-to-end comparison study (this iteration)
│   ├── common.py        #   shared DGPs, losses, Kupiec, NW loss-diff test
│   ├── exp01_point_horse_race.py      # prophet_lite vs SARIMA/theta/snaive
│   ├── exp02_interval_calibration.py  # 80/95% coverage, 300 seeded reps
│   ├── exp03_robust_filtering.py      # DCS vs Kalman under outliers
│   ├── exp04_tail_quantiles.py        # AL-GAS vs GARCH-implied vs static 5% VaR
│   ├── exp05_lad_arima.py             # LAD vs Gaussian ARMA one-step forecasts
│   ├── run_all.py       #   runs exp01..exp05 in sequence (~10-13 min)
│   └── results/         #   generated: expNN.md tables + expNN.json payloads
├── audit/               # probe scripts + summaries behind the audit rounds
│   ├── round11/         #   audit round 11 sweeps (docs/roadmap/26-)
│   └── repo/            #   the whole-repository audit (docs/roadmap/27-)
├── REPORT.md            # findings memo: verdicts, tables, graduation candidates
└── README.md            # this file
```

## How to run everything

Everything uses the repo venv (numpy/scipy/statsmodels/pytest and the
locally built `tsecon` are already installed; no network access needed —
the only datasets are statsmodels' bundled CO2 and macrodata):

```bash
VENV=/home/user/tsecon/.venv/bin/python

# unit tests of the two method families (all seeded)
$VENV -m pytest /home/user/tsecon/lab/prophet_lite/tests.py -q   # 8 passed
cd /home/user/tsecon/lab/laplace && $VENV -m pytest tests.py -q  # 7 passed

# the full comparison study (writes results/expNN.{md,json}, ~10-13 min)
cd /home/user/tsecon/lab/experiments && $VENV run_all.py

# or any single experiment
cd /home/user/tsecon/lab/experiments && $VENV exp03_robust_filtering.py
```

Every experiment is seeded and deterministic; tables land on stdout and
in `experiments/results/`, and `lab/REPORT.md` embeds the same tables
with the verdicts.

## Provenance / licensing (rolled up from the module READMEs)

- **prophet_lite** — implemented from scratch from the published method:
  Taylor & Letham (2018), "Forecasting at Scale", *The American
  Statistician* 72(1). The reference implementation (facebook/prophet) is
  MIT-licensed and **no code was copied from it**. The lasso solver
  follows Friedman, Hastie & Tibshirani (2010, *JSS* 33(1)).
- **laplace** — implemented from the published literature: Creal-Koopman-
  Lucas (2013, JAE) and Harvey (2013, CUP) for GAS/DCS; Engle-Manganelli
  (2004, JBES) adaptive CAViaR; Koenker-Machado (1999) / Yu-Moyeed (2001)
  for the AL-quantile equivalence; Catania-Luati (J. Econometrics 2022)
  for the dynamic-quantile family; Chernozhukov-Fernandez-Val-Galichon
  (2010, Ecta) rearrangement; Harvey-Luati (2014, JASA) DCS-t filtering;
  Davis-Dunsmuir (1997) and Ling (2005) for LAD-ARMA. No proprietary
  code consulted anywhere.
- **Data**: seeded synthetic DGPs plus two datasets bundled with
  statsmodels (Mauna Loa CO2, US macrodata); no network calls, matching
  the library's no-data-loader boundary.
- Both modules ship under the repository's dual MIT/Apache-2.0 terms like
  the rest of the tree, but are **not** distributed in the wheel.

## Status

See `REPORT.md` for the current verdicts and the graduation-candidate
proposals from the latest study (2026-08-17).
