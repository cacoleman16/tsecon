# Identify a monetary policy shock with sign restrictions

A Cholesky ordering is a point identification you usually cannot defend. Sign
restrictions ask for less: state only the signs you are confident about — "a
contractionary policy shock raises the policy rate and lowers prices for a
couple of quarters" — and let the data report the *set* of responses consistent
with them. The width of the resulting band is the finding.

## The recipe

```python
import numpy as np, tsecon

rng = np.random.default_rng(2)                       # columns: 0 output, 1 prices, 2 rate
A0 = np.array([[0.6, 0.2, -0.3],                     # impact of [demand, supply, policy]
               [0.4, -0.5, -0.2],
               [0.3, 0.1, 0.5]])
B = np.diag([0.6, 0.5, 0.7])
Y = np.zeros((300, 3))
for t in range(1, 300):
    Y[t] = B @ Y[t - 1] + A0 @ rng.standard_normal(3)

R = [(2, 2, 0, "+"), (2, 2, 1, "+"),                 # policy shock raises the rate, h=0,1
     (1, 2, 0, "-"), (1, 2, 1, "-")]                 # and lowers prices, h=0,1
out = tsecon.sign_restricted_svar(Y, restrictions=R, lags=2, horizon=8,
                                  n_draws=1000, seed=0)

q = np.array(out["quantiles"])                       # [h][variable][shock][prob]
for h in (0, 2, 4, 6, 8):                            # probs = 5, 16, 50, 84, 95%
    print(f"h={h}   output {q[h,0,2,2]:+.3f}   68% [{q[h,0,2,1]:+.3f}, {q[h,0,2,3]:+.3f}]")
print("acceptance rate:", round(out["diagnostics"]["acceptance_rate"], 4))
```

```
h=0   output -0.167   68% [-0.559, +0.264]
h=2   output -0.092   68% [-0.201, +0.056]
h=4   output -0.036   68% [-0.081, +0.010]
h=6   output -0.010   68% [-0.034, +0.006]
h=8   output -0.003   68% [-0.014, +0.004]
```
```
acceptance rate: 0.4237
```

## Reading it

A restriction is a `(variable, shock, horizon, sign)` tuple. Here shock 2 is the
policy shock, and we constrain only variables 2 (the rate) and 1 (prices).
**Output is deliberately left unrestricted** — its response is the question, and
its band is the answer.

That answer, honestly reported: output is centred negative at every horizon but
the 68% band straddles zero throughout. In this sample, sign restrictions on the
rate and prices are not enough to sign the output response. That is a real
result, and a point-identified scheme would have hidden it behind a single
confident line.

`quantiles` is `[horizon][variable][shock][prob]` with
`probs = [0.05, 0.16, 0.50, 0.84, 0.95]`, so index `2` is the median and
`1`/`3` bracket the 68% band. **The median is not "the" IRF** — it mixes
different rotations at different horizons, so no single structural model
produces that path.

## The acceptance rate is a diagnostic

```python
lo, hi = np.array(out["set_min"]), np.array(out["set_max"])
print(f"identified set for output at h=0: [{lo[0,0,2]:+.3f}, {hi[0,0,2]:+.3f}]")
print("rotations tried:", out["diagnostics"]["rotations_tried"],
      " accepted:", out["diagnostics"]["accepted"])
```

```
identified set for output at h=0: [-0.832, +0.649]
rotations tried: 2360  accepted: 1000
```

`set_min` / `set_max` are the raw identified-set envelope — every value a
surviving rotation produced, with no probability attached. Compare that
`[-0.83, +0.65]` to the 68% quantile band `[-0.56, +0.26]`: the gap between them
is entirely the Haar prior on rotations, not evidence.

An acceptance rate of 0.42 is healthy. Rates near `1e-5` mean your "posterior"
is a handful of draws and your restrictions are close to mutually inconsistent —
watch it every time you add a restriction, because acceptance decays roughly
exponentially in their number.

## Gotchas

- **The Haar prior is not uninformative** about the responses you care about
  (Baumeister-Hamilton 2015). Part of any sign-restricted band is prior. For
  prior-robust bounds use `tsecon.robust_svar_bounds`; for the single coherent
  draw closest to the median path, `tsecon.fry_pagan_svar`.
- **Do not stack restrictions to narrow the band** without watching acceptance.
  Narrowing by assumption is not evidence.
- **`seed` and `n_draws` belong in your write-up.** Both change the numbers.
- Need a zero restriction too ("this shock has no contemporaneous effect on
  output")? That is `tsecon.zero_sign_svar`. Have an external instrument? Use
  `tsecon.proxy_svar` instead — it point-identifies.

## See also

- Model card: [VAR / SVAR](../reference/model-cards/var-svar.md) ·
  [Structural identification (advanced)](../reference/model-cards/structural-identification.md)
- Guide: [8. Structural Identification](../guide/08-causal-identification.md)
- Recipes: [Put confidence bands on a VAR impulse response](var-irf-bands.md) ·
  [Estimate a fiscal multiplier from an instrumented shock](fiscal-multiplier.md)
