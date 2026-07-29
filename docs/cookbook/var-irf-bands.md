# Put confidence bands on a VAR impulse response

`tsecon.var_irf` gives you the point path and nothing else — deliberately, so
that the cheap call stays cheap. The moment you want to say whether a response
is *distinguishable from zero*, switch to `tsecon.var_irf_bands`, which takes
the same `(data, lags, horizon, orth)` arguments and returns a dict with
`point`, `se`, `lower`, and `upper`.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)                      # a toy 2-variable macro system
A = np.array([[0.55, 0.10], [0.35, 0.60]])          # VAR(1) coefficients
L = np.array([[1.00, 0.00], [0.50, 0.80]])          # correlated innovations
y = np.zeros((240, 2))
for t in range(1, 240):
    y[t] = A @ y[t - 1] + L @ rng.standard_normal(2)

b = tsecon.var_irf_bands(y, lags=2, horizon=10, method="bootstrap",
                         n_boot=2000, seed=0, alpha=0.10)   # 90% percentile bands
point, lower, upper = (np.array(b[k])[:, 1, 0] for k in ("point", "lower", "upper"))
for h in (0, 2, 4, 6, 8, 10):
    print(f"h={h:2d}   {point[h]:+.3f}   [{lower[h]:+.3f}, {upper[h]:+.3f}]")
```

```
h= 0   +0.511   [+0.406, +0.607]
h= 2   +0.680   [+0.533, +0.797]
h= 4   +0.471   [+0.293, +0.594]
h= 6   +0.253   [+0.095, +0.372]
h= 8   +0.118   [+0.003, +0.227]
h=10   +0.049   [-0.024, +0.139]
```

## Reading it

Every array is indexed `[horizon][response][shock]`, exactly like `var_irf`, so
`[:, 1, 0]` is "variable 1's response to a shock in variable 0". The response
is hump-shaped, peaks at `h=2`, still excludes zero at `h=8`, and no longer does
by `h=10` — that last line is the finding, not a footnote. Slicing the band
arrays the same way you slice `var_irf` means you can drop banded responses into
an existing plot without reshaping anything.

Two knobs decide what the band *means*:

- `alpha` is the **non**-coverage: `alpha=0.10` is a 90% band, `alpha=0.32` the
  68% band macro papers often plot.
- `seed` makes the bootstrap reproducible. Report it. A bootstrap band without
  a seed and an `n_boot` is not a replicable number.

## The asymptotic alternative

`method="asymptotic"` — the **default** — gives Lütkepohl delta-method standard
errors and Wald bands (`point ± z·se`) with no resampling. It is instant, and it
hands you a usable `se` array:

```python
a = tsecon.var_irf_bands(y, lags=2, horizon=10, method="asymptotic", alpha=0.10)
print("method:", a["method"], " n_boot:", a["n_boot"])
for h in (0, 4, 8):
    print(f"h={h:2d}   se {np.array(a['se'])[h, 1, 0]:.3f}   "
          f"[{np.array(a['lower'])[h, 1, 0]:+.3f}, {np.array(a['upper'])[h, 1, 0]:+.3f}]")
```

```
method: asymptotic  n_boot: None
h= 0   se 0.058   [+0.415, +0.606]
h= 4   se 0.091   [+0.322, +0.620]
h= 8   se 0.072   [-0.001, +0.236]
```

Note that the two methods disagree about `h=8`: the bootstrap band excludes
zero, the asymptotic one just barely includes it. That disagreement is
information — delta-method bands are symmetric by construction and are known to
undercover at longer horizons in short samples, which is why `method="bootstrap"`
is the one to reach for when the answer matters. `n_boot` comes back `None` on
the asymptotic branch so a logged result always records which method ran.

## Gotchas

- **`orth=True` is the default and it is an identification assumption.** The
  bands describe sampling uncertainty about a *Cholesky-ordered* response; they
  say nothing about whether the ordering is defensible. For a causal reading,
  go to [sign restrictions](sign-restricted-svar.md) or a local projection.
- **Check stability first.** `tsecon.var_fit(y, lags=2)["is_stable"]` must be
  `True`; near-unit-root systems make long-horizon bands unreliable under either
  method.
- `bias_correct=True` applies the Kilian (1998) bootstrap bias correction on the
  bootstrap branch — worth trying when the VAR is persistent.
- Want cumulative (level) responses? Pass `cumulative=True`; it means the same
  thing here as in `var_irf`.

## See also

- Model card: [VAR / SVAR](../reference/model-cards/var-svar.md)
- Guide: [7. Systems (VAR, Factors)](../guide/07-multivariate.md)
- Recipes: [Forecast a VAR with prediction intervals](var-forecast-intervals.md) ·
  [Identify a monetary policy shock with sign restrictions](sign-restricted-svar.md) ·
  [Export a results table to LaTeX or Markdown](results-table-export.md)
