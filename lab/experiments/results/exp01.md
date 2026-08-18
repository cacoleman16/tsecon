## Experiment 1 — point-forecast horse race (rolling origin, expanding window, refit every origin)

### (a) Synthetic piecewise-trend + seasonal + outliers (T=240, 55 origins, expanding, refit each origin)

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite | 2.5647 | 1.8285 | 3.9017 | 2.4508 | 5.5393 | 3.3618 |
| sarima | 2.3668 | 1.7639 | 3.8471 | 2.9204 | 5.9995 | 4.7354 |
| theta | 2.0789 | 1.4604 | 3.6824 | 3.3680 | 6.3258 | 6.1080 |
| seasonal_naive | 4.9249 | 4.0950 | 3.9096 | 3.5979 | 4.0422 | 3.7813 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs sarima | 1 | 0.98 | 0.3293 | +0.976 |
| prophet_lite vs sarima | 12 | -0.96 | 0.3420 | -5.310 |
| prophet_lite vs theta | 1 | 1.64 | 0.1065 | +2.256 |
| prophet_lite vs theta | 12 | -0.35 | 0.7314 | -9.332 |

### (b) CO2 monthly means, interpolated (T=526, 20 origins; integer index + (12,5) Fourier seasonality since calendar-monthly spacing is irregular in days)

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite | 0.9651 | 0.8430 | 1.0502 | 0.8955 | 1.2226 | 1.0323 |
| sarima | 0.3465 | 0.2914 | 0.5851 | 0.4434 | 0.7566 | 0.5695 |
| theta | 0.3468 | 0.2941 | 0.7886 | 0.6518 | 1.1534 | 0.9660 |
| seasonal_naive | 1.6911 | 1.5382 | 1.6745 | 1.5090 | 1.6999 | 1.5437 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs sarima | 1 | 3.91 | 0.0009 | +0.811 |
| prophet_lite vs sarima | 12 | 2.55 | 0.0197 | +0.922 |
| prophet_lite vs theta | 1 | 3.99 | 0.0008 | +0.811 |
| prophet_lite vs theta (NW fallback) | 12 | 0.48 | 0.6279 | +0.164 |

Note: only 20 origins (SARIMA+prophet refits are expensive at T≈500), so the DM tests here are low-powered; read signs, not significance.

### (c) Real GDP growth, quarterly 400·dlog (T=202, 71 origins) — no seasonality; trend models should lose here

| model | RMSE h=1 | MAE h=1 | RMSE h=6 | MAE h=6 | RMSE h=12 | MAE h=12 |
|---|---|---|---|---|---|---|
| prophet_lite | 2.1385 | 1.6190 | 2.2165 | 1.7061 | 2.5853 | 1.8636 |
| ar1 | 2.0738 | 1.5766 | 2.2099 | 1.6526 | 2.6066 | 1.8055 |
| theta | 2.1045 | 1.6459 | 2.4771 | 1.9184 | 2.9529 | 2.2275 |
| mean | 2.1585 | 1.6168 | 2.2073 | 1.6500 | 2.6029 | 1.8022 |

| pair (squared loss) | h | DM (HLN) | p | mean d |
|---|---|---|---|---|
| prophet_lite vs ar1 | 1 | 0.79 | 0.4295 | +0.273 |
| prophet_lite vs ar1 | 12 | -0.09 | 0.9273 | -0.111 |
| prophet_lite vs mean | 1 | -0.26 | 0.7966 | -0.086 |
| prophet_lite vs mean | 12 | -0.08 | 0.9390 | -0.091 |

_Runtime: 194 s. Seeds fixed; rerun with `python exp01_point_horse_race.py`._
