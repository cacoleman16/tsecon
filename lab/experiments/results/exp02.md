## Experiment 2 — interval calibration, 300 seeded replications of the home-turf DGP

Training window T=120 with 3% outliers (6-10 sigma); the 12-step future is clean.  prophet_lite intervals: 500 predictive draws (future-changepoint bootstrap + Gaussian noise).  SARIMA (0,1,1)(0,1,1)_12 intervals: parametric Gaussian (innovation+filtering uncertainty, parameters treated as known — tsecon's documented statsmodels-matching default).  Binomial MC standard errors in parentheses (R=300).

| model | nominal | cov h=1 (se) | cov h=6 (se) | cov h=12 (se) | pooled h=1..12 | mean width |
|---|---|---|---|---|---|---|
| prophet_lite | 80% | 0.887 (0.018) | 0.893 (0.018) | 0.890 (0.018) | 0.897 | 4.26 |
| prophet_lite | 95% | 0.967 (0.010) | 0.980 (0.008) | 0.987 (0.007) | 0.981 | 6.49 |
| sarima | 80% | 0.900 (0.017) | 0.933 (0.014) | 0.957 (0.012) | 0.949 | 8.27 |
| sarima | 95% | 0.987 (0.007) | 0.993 (0.005) | 0.993 (0.005) | 0.994 | 12.65 |

prophet_lite fits converged: 300/300.

_Runtime: 467 s._
