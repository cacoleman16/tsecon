## Experiment 5 (supplementary) — LAD/median ARMA one-step forecasts under heavy tails

ARMA(1,1) phi=0.6 theta=0.3 mean=0.5, train 300 / test 50 one-step forecasts with frozen parameters, 20 replications per innovation type; ratio < 1 favours LAD.

| DGP | LAD RMSE | Gauss RMSE | ratio | LAD MAE | Gauss MAE | ratio |
|---|---|---|---|---|---|---|
| t innovations | 1.8421 | 1.8542 | 0.993 | 1.1889 | 1.2023 | 0.989 |
| laplace innovations | 1.4672 | 1.4687 | 0.999 | 1.0414 | 1.0441 | 0.997 |
| gaussian innovations | 1.0158 | 1.0132 | 1.003 | 0.8094 | 0.8063 | 1.004 |

Gaussian-CSS twin phi=0.6223, theta=0.4264 vs tsecon.arima_fit exact MLE params=[0.0436, 0.6223, 0.4251, 1.0071] (['const', 'ar.L1', 'ma.L1', 'sigma2'])

_Runtime: 43 s._
