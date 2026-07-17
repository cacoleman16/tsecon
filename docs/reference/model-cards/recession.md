# Model card — Recession-probability models

`recession_probit`

A binary-choice model for a 0/1 recession indicator `y_t`: it maps a linear
index of leading variables — the term spread above all — into a recession
*probability* through a probit or logit link. The default is the textbook static
model (`Phi(x_t' beta)`); flipping `dynamic=True` fits the Kauppi-Saikkonen
(2008) dynamic probit, whose index carries its own autoregressive lag so that
last period's recession pressure raises this period's.

---

## `recession_probit`

**What it estimates.** The coefficients `beta` of a binary-choice model for the
probability that period `t` is a recession,

```text
P(y_t = 1 | x_t) = F(x_t' beta),
```

with `F = Phi` (standard normal CDF, `link="probit"`) or `F = Lambda` (logistic
CDF, `link="logit"`), fit by exact-likelihood maximum likelihood. With
`dynamic=True` the index is instead autoregressive,

```text
index_t = w + x_t' b + rho * index_{t-1},   P(y_t = 1) = Phi(index_t),
```

the Kauppi-Saikkonen (2008) dynamic probit: `rho` measures how much recession
pressure persists from one period to the next, and the estimator returns it
alongside the intercept `w` and the covariate slopes `b`.

**Assumptions.** The indicator is genuinely binary (`y_t in {0, 1}`), the
predictors are observed leading variables (the model does not lag them for you),
and the link is correctly specified. The static MLE requires that the classes be
*not perfectly separable* — a predictor that splits recessions from expansions
cleanly drives the coefficient to infinity and there is no finite maximum
(reported as a separation error). The dynamic index is assumed stationary
(`|rho| < 1`); it is initialized at its stationary mean `w / (1 - rho)`.

**When to use (and when not).** Use it to turn leading indicators into a
calibrated recession probability — the canonical Estrella-Mishkin term-spread
model is exactly `recession_probit(y, [const, spread])`. Reach for
`dynamic=True` when recessions are persistent (a downturn this quarter makes one
next quarter more likely) beyond what the covariates already capture; on such
data it fits strictly better because it nests the static probit at `rho = 0`. Do
**not** use it for a continuous target (this is classification, not regression),
and do not expect the dynamic model to help on data with no index persistence —
`rho` will simply estimate near zero.

**Key arguments and defaults (and why).** `link="probit"` is the default and the
convention in the recession-forecasting literature; `"logit"` fits the same data
with a logistic link (coefficients are on a different scale — roughly 1.6x the
probit's — but fitted probabilities are close). `dynamic=False` is the static
model; `dynamic=True` switches to the Kauppi-Saikkonen dynamic probit (probit
only — `link` is ignored). The one design rule that bites: **`x` must include a
constant column for the static model, and must NOT include one for the dynamic
model** — the dynamic estimator supplies its own intercept `w`, and a redundant
constant column makes the information matrix singular.

**How to read the output.** `params` are the coefficients (static: in the column
order of `x`; dynamic: laid out `[w, b_0, .., b_{m-1}, rho]`), with matching
`bse` (standard errors from the inverse observed information) and `zstats`
(`params / bse`, an asymptotic z-test). `probabilities` is the fitted
`P(y_t = 1)` path — the object you actually plot or threshold. `loglik` is the
maximized log-likelihood and `pseudo_r2` is McFadden's `1 - loglik/loglik_null`
(0 at the intercept-only fit, 1 at a perfect one; values around 0.2-0.4 are
"good" for binary choice, not the 0.9 you might expect from an R²). The dynamic
fit additionally exposes `w`, `beta`, and `rho` as named scalars/vectors, plus
`converged` (the optimizer flag). A `rho` near 1 flattens the likelihood surface
and can leave `converged=False` at a perfectly good optimum — read the estimates
and `loglik`, not the flag alone.

**Failure modes.** Complete or quasi-complete separation (a predictor that
perfectly classifies) has no finite MLE and raises a separation error — coarsen
or drop the offending predictor. Forgetting the constant column in the static
model biases every coefficient; *including* one in the dynamic model makes the
information matrix singular. A degenerate response (all-0 or all-1 — no
recessions in the sample) cannot be fit. Because the dynamic model is optimized
by a derivative-free simplex over a recursive index, treat a huge `|rho|` or a
`converged=False` flag as a prompt to inspect `loglik`, not an automatic error.

**Validated against.** `statsmodels`' `Probit` and `Logit` maximum-likelihood
estimators (`sm.Probit(y, X).fit()`, `sm.Logit(y, X).fit()`) on a fixed
simulated dataset — point estimates, analytic-Hessian standard errors,
log-likelihood, and McFadden pseudo-R² match to ~1e-5 (`fixtures/tsecon-
recession.json`), a genuine cross-implementation check by a separate code path.
The dynamic probit has no `statsmodels` counterpart and is validated
**property-only** in the crate's Rust suite: on data from a known
dynamic-probit DGP it recovers `rho` and `b` within Monte-Carlo bands, and its
log-likelihood exceeds the static probit's on persistent data.

**References.** Estrella & Mishkin (1998, term-spread probit); Kauppi &
Saikkonen (2008, dynamic probit); McFadden (1974, pseudo-R²).

Static model — the term-spread recession probit (`x` **includes** a constant):

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 600
spread = np.zeros(n)                           # term spread: a persistent (AR1) leading indicator
for t in range(1, n):
    spread[t] = 0.6 * spread[t - 1] + rng.standard_normal()

index = -0.7 - 0.9 * spread                    # inverted curve (low spread) -> higher recession risk
y = (index + rng.standard_normal(n) > 0).astype(float)   # latent-probit draw: P(y=1) = Phi(index)

X = np.column_stack([np.ones(n), spread])      # STATIC model: X MUST include the constant column
fit = tsecon.recession_probit(y, X, link="probit")
print("params [const, spread]:", np.round(fit["params"], 3))   # ~[-0.76, -0.82] (true -0.7, -0.9)
print("z-stats               :", np.round(fit["zstats"], 3))
print(f"pseudo-R2 = {fit['pseudo_r2']:.3f}   converged: {fit['converged']}")

logit = tsecon.recession_probit(y, X, link="logit")            # same data, logistic link
print("logit params          :", np.round(logit["params"], 3))  # rescaled ~1.6x the probit's
```

Dynamic model — Kauppi-Saikkonen probit (`x` **omits** the constant; the model
estimates its own intercept `w`):

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)
n = 1200
w, b, rho, px = -0.3, 1.0, 0.6, 0.5
x = np.zeros(n)
for t in range(1, n):
    x[t] = px * x[t - 1] + rng.standard_normal()   # a persistent predictor
y = np.zeros(n)
prev = w / (1 - rho)                               # stationary-mean initialization
for t in range(n):
    idx = w + b * x[t] + rho * prev                # the index carries its own lag
    y[t] = 1.0 if idx + rng.standard_normal() > 0 else 0.0
    prev = idx

Xd = x.reshape(-1, 1)                              # DYNAMIC model: NO constant column
dyn = tsecon.recession_probit(y, Xd, dynamic=True)
print("params [w, b, rho]:", np.round(dyn["params"], 3))   # ~[-0.25, 1.00, 0.61] (true -0.3, 1.0, 0.6)
print(f"rho = {dyn['rho']:.3f}   pseudo-R2 = {dyn['pseudo_r2']:.3f}")

stat = tsecon.recession_probit(y, np.column_stack([np.ones(n), x]))   # static comparison
print(f"loglik  dynamic {dyn['loglik']:.1f}  >=  static {stat['loglik']:.1f}")  # dynamic fits better
```
