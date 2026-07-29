# Model fat-tailed volatility with GARCH and forecast it

Financial returns are not conditionally normal — even after GARCH soaks up the
volatility clustering, the standardized residuals stay fat-tailed. Fitting
Gaussian GARCH anyway is fine for the *conditional variance* (QMLE stays
consistent) but it misprices tail risk. `tsecon.garch_fit(dist="t")` fits the
Student-*t* likelihood and estimates the degrees of freedom alongside the
variance parameters.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(3)                        # 1500 daily returns, in percent
n, nu = 1500, 5.0
eps = rng.standard_t(nu, n) * np.sqrt((nu - 2) / nu)  # unit-variance fat-tailed shocks
r = np.zeros(n)
s2 = np.zeros(n)
s2[0] = 0.05 / (1 - 0.10 - 0.85)                      # start at the unconditional variance
for t in range(1, n):
    s2[t] = 0.05 + 0.10 * r[t - 1] ** 2 + 0.85 * s2[t - 1]
    r[t] = np.sqrt(s2[t]) * eps[t]

for dist in ("normal", "t"):
    f = tsecon.garch_fit(r, vol="garch", mean="constant", dist=dist)
    pars = "  ".join(f"{k} {v:+.4f}" for k, v in zip(f["param_names"], f["params"]))
    print(f"{dist:>6}   {pars}   bic {f['bic']:.1f}")
```

```
normal   mu -0.0294  omega +0.1490  alpha[1] +0.1331  beta[1] +0.7103   bic 4082.4
     t   mu -0.0295  omega +0.0945  alpha[1] +0.1418  beta[1] +0.7670  nu +5.5287   bic 3977.8
```

## Reading it

BIC drops by more than 100 points when the likelihood is allowed fat tails, and
`nu` comes back at 5.5 against the true 5. The conditional mean barely moves —
this is not a story about the mean.

Two parameters worth naming every time:

- **`alpha[1] + beta[1]` is the persistence.** Close to 1 means shocks to
  variance die slowly; at exactly 1 you have IGARCH and no unconditional
  variance.
- **`nu`** is the tail index. Below about 4 the kurtosis of the innovation
  distribution is infinite; below 2, the variance is. If `nu` estimates near the
  low end, that is a modelling signal, not a number to report and move past.

## Forecasting the variance path

```python
f = tsecon.garch_fit(r, vol="garch", mean="constant", dist="t", forecast_horizon=10)
print("robust SEs :", "  ".join(f"{k} {v:.4f}"
                                for k, v in zip(f["param_names"], f["se_robust"])))
print("persistence:", round(f["params"][2] + f["params"][3], 4))
print("today's vol:", round(f["conditional_volatility"][-1], 4))
print("vol forecast h=1..10:", np.round(np.sqrt(f["variance_forecast"]), 4))
```

```
robust SEs : mu 0.0210  omega 0.0359  alpha[1] 0.0341  beta[1] 0.0608  nu 0.7892
persistence: 0.9087
today's vol: 1.2444
vol forecast h=1..10: [1.133  1.123  1.1137 1.1053 1.0976 1.0905 1.084  1.0781 1.0727 1.0678]
```

`variance_forecast` is a **variance** path; take the square root for volatility
in return units, and multiply by `sqrt(252)` for an annualized figure on daily
data. The path decays from today's elevated level toward the unconditional
volatility at the rate set by the persistence — which is the entire economic
content of a GARCH forecast.

Quote `se_robust`, not `se_mle`: those are the Bollerslev-Wooldridge standard
errors, valid even when the Student-*t* likelihood is only an approximation.

## Did it work? Check the standardized residuals

```python
z = np.asarray(f["std_residuals"])
print("standardized-residual excess kurtosis:", round(float((z ** 4).mean() / (z ** 2).mean() ** 2 - 3), 3))
print("Ljung-Box on squared standardized residuals, p:",
      round(tsecon.ljung_box(z ** 2, nlags=10)["lb_pvalue"][-1], 4))
```

```
standardized-residual excess kurtosis: 4.024
Ljung-Box on squared standardized residuals, p: 0.9934
```

Ljung-Box on the **squared** standardized residuals is the post-fit test that
matters: p = 0.99 says no volatility clustering is left, so the GARCH part did
its job. The excess kurtosis of 4.0 says the residuals are still far from
normal — which is exactly why `dist="t"` earned its 100 BIC points. Under
`dist="normal"` this same number would have been a silent warning that your
value-at-risk is understated.

## Gotchas

- **Scale your returns.** Percent returns (roughly unit-scale) optimize far more
  reliably than raw decimals. Very persistent designs (`alpha + beta` above
  ~0.97) with a small `omega` can make the Student-*t* optimizer fail at the
  starting values with an "optimization failure" error — rescale, or fit
  `dist="normal"` first and use it as a sanity check.
- **`vol="gjr"`** adds the leverage effect (negative returns raise volatility
  more than positive ones), which is present in almost every equity series and
  absent from plain GARCH. `vol="egarch"` models log-variance instead.
- Fat tails *and* outliers you would rather down-weight than fit?
  `tsecon.gas_volatility(density="student_t")` is the score-driven alternative.
- Several series at once: `tsecon.ccc_garch` / `tsecon.dcc_garch`.

## See also

- Model card: [Volatility](../reference/model-cards/volatility.md)
- Guide: [6. Volatility](../guide/06-volatility.md)
- Recipes: [Screen a new series before you model it](screen-a-series.md) ·
  [Compute growth-at-risk](growth-at-risk.md)
