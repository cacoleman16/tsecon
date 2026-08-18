## Experiment 3 — robust trend filtering under additive outliers

Local level, sigma_eta=0.1, sigma_eps=1.0, T=500, outliers at 8.0 sd, 30 replications per contamination level; RMSE of the one-step-predicted level against the clean truth (mean over reps, sd in parentheses), burn-in 20.

| method (one-step-predicted level unless noted) | RMSE 0% (sd) | RMSE 5% (sd) | RMSE 10% (sd) |
|---|---|---|---|
| DCS-t (robust) | 0.321 (0.037) | 0.342 (0.044) | 0.354 (0.052) |
| DCS-Laplace (robust) | 0.354 (0.043) | 0.360 (0.043) | 0.375 (0.049) |
| DCS-Gaussian (nested control) | 0.321 (0.036) | 0.441 (0.052) | 0.514 (0.078) |
| tsecon Kalman predicted @ UC-MLE | 0.321 (0.037) | 0.448 (0.051) | 0.516 (0.076) |
| tsecon Kalman SMOOTHED @ UC-MLE (look-ahead ref) | 0.225 (0.025) | 0.345 (0.056) | 0.365 (0.061) |

### Fitted gain kappa (the Gaussian gain-collapse failure mode)

| method | mean kappa 0% | mean kappa 5% | mean kappa 10% |
|---|---|---|---|
| DCS-Gaussian (nested control) | 0.0886 | 0.0406 | 0.0295 |
| DCS-t (robust) | 0.0908 | 0.1288 | 0.1322 |
| DCS-Laplace (robust) | 0.1161 | 0.0816 | 0.0617 |

### Nesting check on clean data (first 5 reps): DCS-Gaussian = steady-state Kalman

| rep | DCS-Gaussian kappa | steady-state Kalman gain | |diff| | path RMSE vs Kalman predicted |
|---|---|---|---|---|
| 0 | 0.1105 | 0.1111 | 5.8e-04 | 0.0019 |
| 1 | 0.0927 | 0.0948 | 2.1e-03 | 0.0059 |
| 2 | 0.0959 | 0.0971 | 1.3e-03 | 0.0037 |
| 3 | 0.0547 | 0.0489 | 5.8e-03 | 0.0249 |
| 4 | 0.0537 | 0.0561 | 2.4e-03 | 0.0082 |

_Runtime: 134 s._
