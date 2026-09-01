# Choose a VAR lag length when AIC, BIC, and HQIC disagree

Every extra VAR lag adds a full block of coefficients, so a lag that improves
fit can still make the system worse out of sample. Compare the information
criteria on the **same effective sample** first; then let the analysis goal,
not a majority vote, break a disagreement.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(6)
A1 = np.array([[0.45, 0.12], [0.08, 0.35]])
A2 = np.array([[0.18, 0.00], [0.10, -0.12]])
L = np.array([[0.80, 0.00], [0.25, 0.55]])
y = np.zeros((180, 2))
for t in range(2, len(y)):
    y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + L @ rng.standard_normal(2)
data = y[100:]

p_max, rows = 6, []
for p in range(1, p_max + 1):
    fit = tsecon.var_fit(data[p_max - p:], lags=p)
    rows.append((p, fit["aic"], fit["bic"], fit["hqic"]))
print("lag    AIC     BIC    HQIC")
for p, aic, bic, hqic in rows:
    print(f"{p:>3}  {aic:>6.3f}  {bic:>6.3f}  {hqic:>6.3f}")
for name, column in (("AIC", 1), ("BIC", 2), ("HQIC", 3)):
    print(f"{name} chooses p={min(rows, key=lambda row: row[column])[0]}")
```

```
lag    AIC     BIC    HQIC
  1  -1.541  -1.354  -1.466
  2  -1.624  -1.313  -1.500
  3  -1.552  -1.116  -1.379
  4  -1.532  -0.971  -1.308
  5  -1.489  -0.804  -1.216
  6  -1.383  -0.573  -1.060
AIC chooses p=2
BIC chooses p=1
HQIC chooses p=2
```

## Reading the disagreement

Lower is better. AIC charges the lightest complexity penalty and usually keeps
more dynamics; BIC charges the heaviest and favors a smaller system; HQIC sits
between them. Here the serious candidates are therefore VAR(1) and VAR(2), not
all six fitted orders.

Do not decide by counting two votes against one. Match the rule to the job:

- **Forecasting:** backtest both finalists. If their out-of-sample errors are
  effectively tied, prefer BIC's smaller VAR(1).
- **Impulse responses or Granger tests:** omitted dynamics can distort every
  horizon. Start from the AIC/HQIC choice, VAR(2), then check each equation's
  residuals with `tsecon.ljung_box`.

Write that decision rule down before inspecting the downstream result. The
criterion is a shortlist, not evidence that the chosen model is adequate.

## Why the sample is trimmed

`var_fit(data, lags=p)` consumes the first `p` rows. Without the slice
`data[p_max - p:]`, different candidates would receive different observations,
so a change in AIC or BIC could reflect the sample rather than the lag order.
The slice makes every fit end with the same data and use the same number of
effective observations.

## Gotchas

- Compare orders only when the variables, deterministic terms, and effective
  sample are identical.
- Check stationarity before treating the criteria or stability diagnostics as
  reliable; difference the series or use a VECM when appropriate.
- A minimum does not validate residual independence, stability, or forecast
  performance. Run those checks after selecting the lag.

## See also

- Model card: [VAR / SVAR](../reference/model-cards/var-svar.md)
- Guide: [Choosing the lag length when the criteria disagree](../guide/07-multivariate.md#choosing-the-lag-length-when-the-criteria-disagree)
- Recipe: [Forecast a VAR with prediction intervals](var-forecast-intervals.md)
