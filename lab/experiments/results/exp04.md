## Experiment 4 — 5% tail forecasting, GARCH(1,1)-t DGP

T=3000 (train 2000, test 1000), omega=0.05 alpha=0.10 beta=0.85 nu=5; parameters frozen at the training fit; 5 seeds, metrics averaged over seeds (Kupiec = count of seeds where the 5% unconditional-coverage test REJECTS).  Positive NW t means the row model has HIGHER pinball loss than AL-GAS.

| model | mean pinball (tau=.05) | mean hit rate | Kupiec rej. @5% | RMSE vs true quantile |
|---|---|---|---|---|
| AL-GAS dynamic quantile (lab) | 0.1125 | 0.052 | 0/5 | 0.350 |
| GARCH(1,1)-t implied (tsecon, correctly specified) | 0.1096 | 0.050 | 0/5 | 0.055 |
| GARCH(1,1)-normal implied (tsecon) | 0.1099 | 0.045 | 0/5 | 0.101 |
| static quantile_regression (tsecon) | 0.1184 | 0.058 | 2/5 | 0.483 |

| pinball loss differential (NW t, mean over seeds) | mean t | signif @5% |
|---|---|---|
| GARCH(1,1)-t implied (tsecon, correctly specified) - AL-GAS | -0.64 | 0/5 |
| GARCH(1,1)-normal implied (tsecon) - AL-GAS | -0.37 | 0/5 |
| static quantile_regression (tsecon) - AL-GAS | 1.72 | 2/5 |

_Runtime: 24 s._
