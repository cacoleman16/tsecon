# Model card — Quantile regression & growth-at-risk

`quantile_regression` · `quantile_lp` · `growth_at_risk`

A conditional mean answers "what do we expect?"; a conditional quantile answers
"how bad (or good) can it plausibly get?". This family estimates the whole
conditional distribution one quantile at a time by minimizing the
Koenker-Bassett check loss, and its headline application is
**growth-at-risk**: the
Adrian-Boyarchenko-Giannone (2019) finding that financial conditions move the
*lower tail* of future GDP growth far more than they move its center or upper
tail. The center of a distribution can look calm while its downside quietly
deteriorates — quantile methods are how you see that happening.

The estimation problem at quantile level `tau` throughout is

```text
min_b  sum_t  rho_tau(y_t - x_t' b),      rho_tau(u) = u * (tau - 1{u < 0})
```

solved by iteratively reweighted least squares (IRLS), with Powell
kernel-sandwich standard errors (Epanechnikov kernel, Hall-Sheather
bandwidth) — the same estimator and covariance as statsmodels `QuantReg`
at its defaults. `growth_at_risk` adds one thing statsmodels has no
counterpart for: a Newey-West correction to that sandwich for the serial
correlation its *overlapping* `h`-step windows induce. Read
[the measured coverage table](#standard-errors-at-long-horizons-measured)
before you quote any multi-period interval from this family.

---

## `quantile_regression` — linear quantile regression

**What it estimates.** The coefficient vector `b(tau)` of the `tau`-th
conditional quantile `Q_tau(y | x) = x' b(tau)`, at one or many quantile levels
in a single call. Under a location-scale DGP `y = x'b + (x'g) * eps` the slope
genuinely *varies across tau* — `b(tau) = b + g * F_eps^{-1}(tau)` — and that
variation is the signal: a slope that is negative at `tau = 0.1` and positive
at `tau = 0.9` means the regressor widens the distribution, not (only) shifts
it.

**Assumptions.** Correct linear specification of each conditional quantile;
serially uncorrelated check-loss scores, because the Powell sandwich is a
heteroskedasticity-robust estimator and **not** a HAC one. If your design
has overlapping multi-step outcomes — anything of the `y_{t+h}` on `x_t`
shape — that assumption fails by construction and these `bse` are too
narrow; `growth_at_risk` carries the Newey-West correction for exactly that
case (with measured coverage below), `quantile_lp` does not yet. The
conditional density at the quantile must be positive (no flat spots of the
distribution at the quantile being estimated), and it is estimated by a
kernel: at tail `tau` in short samples that estimate is biased *upward*,
which biases every `bse` here *downward* (see the `fhat/f` note below).

**When to use (and when not).** Use it whenever the *tails* are the question —
risk measures, heterogeneous effects across the outcome distribution,
distributional shifts — or when outliers make the mean fragile (the median
regression at `tau = 0.5` is the robust-location workhorse). Do not use it to
answer a mean question: `b(0.5)` is the median effect, not the average effect.
Do not read fitted quantiles at wildly extrapolated `x`; and for extreme `tau`
(0.01, 0.99) remember that the effective sample is only the observations near
that tail.

**Key arguments and defaults (and why).** `x` is your full design matrix —
**include the constant column yourself** (statsmodels `exog` convention; the
function adds nothing). `taus` defaults to `[0.05, 0.25, 0.5, 0.75, 0.95]`, a
standard risk-profile grid. `se="robust"` (the Powell kernel sandwich) is the
only shipped flavor, matching the statsmodels default.

**How to read the output.** Per-tau lists, aligned with `taus`: `params[i]` is
the coefficient vector at `taus[i]` (column order of `x`), with `bse` and
`tvalues` alongside; `bandwidth` and `sparsity` are the Hall-Sheather bandwidth
and estimated sparsity (inverse density at the quantile) that feed the
sandwich; `iterations` counts IRLS steps per tau and `converged` is a single
bool over all of them. Plot `params[:][k]` against `taus` — a flat line says
"pure location shift", a sloped one says "the regressor changes the shape".

**Failure modes.** Forgetting the constant column (slopes then absorb the
intercept and every quantile is wrong). Quantile *crossing*: separate per-tau
fits can produce fitted quantiles that violate `Q_0.25 > Q_0.5` at some `x` —
harmless for coefficient interpretation, fatal for reading off a distribution;
`growth_at_risk` handles this with rearrangement, here you must check
yourself. Sparse tails: at extreme `tau` the sparsity estimate is noisy and
`bse` with it. `converged=False` (very rare at the shared 1e-6 tolerance)
means the IRLS hit its iteration cap — treat the affected tau with suspicion.

**Validated against.** statsmodels `QuantReg(y, x).fit(q=tau)` at all
defaults: `params` to 1e-6 (the shared IRLS stopping tolerance) and
`bse`/`bandwidth`/`sparsity` to 1e-6 relative, across multiple taus and
designs ([`fixtures/tsecon-quantile.json`](../../../fixtures/tsecon-quantile.json),
generated by
[`fixtures/generate_tsecon-quantile_fixtures.py`](../../../fixtures/generate_tsecon-quantile_fixtures.py)).

**References.** Koenker & Bassett (1978, *Econometrica* 46:33-50); Powell
(1991, kernel-sandwich covariance); Hall & Sheather (1988, bandwidth); Koenker
(2005, *Quantile Regression*).

The DGP below is location-scale by construction — the true slope is
`0.5 + 0.75 * z_tau` — so the estimated slopes should fan out across taus,
flipping sign in the lower tail:

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 1000
x1 = rng.uniform(0.0, 2.0, n)                       # a positive "risk factor"
eps = rng.standard_normal(n)
y = 1.0 + 0.5 * x1 + (0.5 + 0.75 * x1) * eps        # scale grows with x1
X = np.column_stack([np.ones(n), x1])               # YOUR explicit constant

fit = tsecon.quantile_regression(y, X, taus=[0.1, 0.5, 0.9])
z = {0.1: -1.2816, 0.5: 0.0, 0.9: 1.2816}           # standard-normal quantiles
for i, tau in enumerate(fit["taus"]):
    b, se = fit["params"][i], fit["bse"][i]
    print(f"tau={tau:.1f}: slope={b[1]:+.3f} (se {se[1]:.3f})   "
          f"true {0.5 + 0.75 * z[tau]:+.3f}")
print("converged:", fit["converged"])
# tau=0.1: slope=-0.504 (se 0.097)   true -0.461
# tau=0.5: slope=+0.526 (se 0.086)   true +0.500
# tau=0.9: slope=+1.353 (se 0.104)   true +1.461
# converged: True
```

The same regressor lowers the 10th percentile and raises the 90th — a pure
mean regression (slope 0.5 throughout) would have missed the entire story.

---

## `quantile_lp` — quantile local projections

**What it estimates.** Impulse responses of the conditional *quantiles* of `y`
to an identified shock: per horizon `h` and quantile `tau`, the coefficient on
`shock_t` in a check-loss regression of `y_{t+h}` on
`[shock_t, const, p lags of y and shock]` — the same design conventions as
[`lp`](local-projections.md), with the impulse coefficient in design column 0.
Where `lp` gives you one response path (the mean), this gives you a fan: how
the shock moves the bad tail, the center, and the good tail of the future
outcome separately.

**Assumptions.** An identified, exogenous-conditional-on-controls shock,
exactly as for `lp`; correct linear quantile specification per horizon. The
standard errors are the Powell kernel sandwich per (tau, horizon) —
heteroskedasticity-robust but **not** HAC. The `y_{t+h}`-on-`shock_t` design
overlaps exactly as `growth_at_risk` does, so the measured undercoverage in
[the table below](#standard-errors-at-long-horizons-measured) is the right
order of magnitude to expect here too, and `se[i][h]` does **not** yet carry
the Newey-West correction that `growth_at_risk`'s `bse` does. Treat
far-horizon bands as indicative, lean on the lag controls to soak up serial
dependence, and prefer `lp`'s HAC/lag-augmented inference when the mean
response is the question.

**When to use (and when not).** Use it when the question is distributional —
"does a financial shock raise *downside* risk to growth?", "do oil shocks fatten
the inflation tail?" — or when you suspect the mean IRF averages over
heterogeneous tail responses. If only the average response matters, use `lp`:
it is better-understood, has lag-augmented/HAC inference, and its bands are
tighter. For the specific "conditional quantiles of future growth given
current conditions" question, `growth_at_risk` below is the purpose-built
tool (conditions enter as regressors, not as an identified shock).

**Key arguments and defaults.** `taus` defaults to `[0.1, 0.5, 0.9]` (a
downside/center/upside fan); `horizons=12`, `n_lag_controls=4` as in `lp`. The
constant is added internally here (lp design conventions), unlike
`quantile_regression`.

**How to read the output.** `irf[i][h]` is the response of the `taus[i]`
quantile at horizon `h`, with `se[i][h]` alongside. Read *vertically*: the gap
between the `tau=0.1` and `tau=0.9` paths at each horizon is the shock's
effect on dispersion. Coinciding paths mean a pure location shift and `lp`
would have sufficed. `converged[i][h]` is the per-fit IRLS flag (the engine
tracked it all along; the binding used to drop it): a `False` entry hit the
1000-iteration cap before the shared 1e-6 coefficient tolerance, so that
point of the fan is the last IRLS iterate, not a verified check-loss
minimum — statsmodels `QuantReg` warns in the same situation. This is not a
corner case: an ordinary AR(1)-plus-shock design at T=200 exhausts the cap
at a median cell (the IRLS cycles between vertex solutions; pinned in the
test suite). Do not quote an unconverged point; refit with fewer lag
controls, a less extreme tau, or more data.

**Failure modes.** All of `lp`'s (non-identified shock, too few lag controls)
plus the quantile-specific ones: crossing between adjacent tau paths at some
horizons (fans are estimated independently per tau), and noisy extreme-tau
paths in short samples — every quantile LP at `tau = 0.05` on 150 observations
is an exercise in optimism.

**Validated against.** statsmodels `QuantReg` per (tau, horizon) on the
identical stacked design at 1e-6
([`fixtures/tsecon-quantile.json`](../../../fixtures/tsecon-quantile.json)).

**References.** Koenker & Bassett (1978); Jordà (2005); Adrian, Boyarchenko &
Giannone (2019) for the risk-fan reading; Powell (1991).

The DGP gives the shock a location effect that decays at 0.8 **and** a
contemporaneous scale effect (a positive shock compresses the noise), so at
`h = 0` the lower-tail response should exceed the upper-tail response, with
the fan closing at longer horizons:

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
n = 500
shock = rng.standard_normal(n)
eps = rng.standard_normal(n)
y = np.zeros(n)
for t in range(n):
    mean_t = sum(0.8 ** h * shock[t - h] for h in range(min(t, 20) + 1))
    y[t] = mean_t + np.exp(-0.4 * shock[t]) * eps[t]   # + shock -> less dispersion

q = tsecon.quantile_lp(y, shock, taus=[0.1, 0.5, 0.9], horizons=8, n_lag_controls=2)
irf = np.asarray(q["irf"])
print("       h=0     h=1     h=2     h=3")
for i, tau in enumerate(q["taus"]):
    print(f"tau={tau:.1f}: " + "  ".join(f"{irf[i, h]:+.3f}" for h in range(4)))
#        h=0     h=1     h=2     h=3
# tau=0.1: +1.384  +0.665  +0.710  +0.449
# tau=0.5: +0.996  +0.817  +0.540  +0.338
# tau=0.9: +0.632  +0.811  +0.733  +0.575
```

At impact the 10th-percentile response (1.384) is more than double the
90th-percentile response (0.632) around a median response of about 1 — the
shock shifts *and* tightens the distribution, and the fan closes toward the
common 0.8-decay path once the contemporaneous scale effect washes out.

---

## `growth_at_risk` — conditional quantiles of future growth (ABG 2019)

**What it estimates.** The Adrian-Boyarchenko-Giannone (2019 AER) object:
conditional quantiles of the `horizon`-ahead outcome given today's
conditioning variables,

```text
Q_tau( y_{t+h} | x_t )  =  b0(tau) + b1(tau)' conditions_t + b2(tau) * y_t
```

fitted by quantile regression at each `tau` and evaluated at **every**
observation `t` — canonically: quantiles of GDP growth `h` quarters ahead
given the NFCI and current growth. The signature finding this reproduces: the
lower quantiles of future growth are volatile and strongly driven by financial
conditions, while the upper quantiles stay roughly flat — downside risk moves,
upside doesn't.

**Assumptions.** Linear conditional quantiles in `[const, conditions, y_t]`
(the constant is added internally here); enough time-series stationarity for
the per-tau `QuantReg` fits and their Powell sandwiches to be meaningful. This
is a *predictive/descriptive* object — no causal identification is claimed for
the conditioning variables.

**When to use (and when not).** Use it for risk dashboards and financial-
stability monitoring: "what is the 5% worst-case for growth next year, given
today's financial conditions?" — and for testing whether some indicator moves
downside risk asymmetrically. Do not read the coefficients causally (tight
financial conditions may proxy for shocks already underway); do not use it for
`horizon = 0` nowcasts (`horizon >= 1` is required, by design); and if your
question is about the response to an *identified shock* rather than the
predictive content of observed conditions, use `quantile_lp`.

**Key arguments and defaults (and why).** `conditions` is `T × k` (one column
per indicator). `horizon = 1` and `taus = [0.05, 0.25, 0.5, 0.75, 0.95]`
mirror the ABG quarterly setup; `taus` must be strictly increasing.
`rearrange = True` applies the Chernozhukov-Fernandez-Val-Galichon (2010)
monotone rearrangement — a simple sort across `tau` at each `t` — which
repairs any quantile crossing without harming (weakly improving) estimation
quality; leave it on unless you specifically want to inspect the raw paths.

**How to read the output.** `params`/`bse` are per-tau coefficient rows in the
order `[const, conditions..., y_t]` — the asymmetry finding is read straight
off the conditions column: a large negative slope at `tau = 0.05` shrinking to
nothing at `tau = 0.95`. `bse` is the Powell sandwich with a Newey-West
(Bartlett) correction at `hac_lags = horizon - 1` lags — see the next
subsection for why, and for what it does and does not buy you.
`bse_powell` is the uncorrected sandwich, i.e. what statsmodels `QuantReg`
reports for this design; at `horizon = 1` the two are identical because
nothing overlaps. `fitted` is the `(n_taus × T)` matrix of conditional
quantiles after rearrangement (`fitted_raw` before); `crossing` flags whether
the raw paths crossed anywhere, i.e. whether rearrangement actually did
something. `current` is the risk read at the latest observation — the row of
`fitted` at `t = T-1`, the number a policymaker would quote today. Note the
timing: `fitted[:, t]` conditions on date-`t` information and describes
`y_{t+horizon}`. `converged` is the per-tau IRLS flag, aligned with
`params`: a `False` entry hit the 1000-iteration cap before the 1e-6
coefficient tolerance, and that tau's coefficients — with every fitted
quantile, the rearrangement, and the `current` read built on them — are the
last iterate, not a verified check-loss minimum. Check it before quoting
`current`.

**Failure modes.** Reading `current` without checking what the conditions were
at `T-1` (a benign read in calm conditions says nothing about the slope of
risk). Overfitting with many conditioning variables — ABG's power comes from
one financial-conditions index, not twenty indicators. **Quoting a long-horizon
tail interval as if it were a 95% interval** — read the next subsection first;
it is the single most important caveat on this function. And extreme taus on
short macro samples (200 quarterly observations put only ~10 of them below the
5% quantile) carry honest, wide uncertainty — report `bse`.

### Standard errors at long horizons (measured)

Growth-at-risk is a multi-quarter tool by construction, and the horizon it is
built for is the horizon where its standard error is worst. Two things go
wrong as `horizon` grows, and only one of them is fixed.

**1. Overlapping windows (fixed).** For `h > 1` consecutive regressions share
innovations — `y_{t+h}` and `y_{t+h-l}` have `h - l` of them in common — so
the check-loss score `x_t (tau - 1{u_t < 0})` is an MA(h-1) *even when the
conditional quantile is perfectly specified*. The plain Powell sandwich
assumes it is a martingale difference and is therefore optimistic by
construction. `bse` applies the standard remedy (Hansen-Hodrick 1980;
Newey-West 1987): a Bartlett-weighted long-run variance at `horizon - 1`
lags, the exact MA order the overlap induces. Bartlett rather than a
rectangular truncation because only Bartlett guarantees a positive
semi-definite covariance — the rectangular version really does return
negative variances on these designs.

**2. The density estimate (not fixed).** Both sandwiches divide by a kernel
estimate `f_hat(0)` of the residual density at the quantile. In the lower
tail that density is *convex*, so kernel smoothing biases the estimate
upward, and the bias grows with the horizon because the overlap shrinks the
effective sample. Too large an `f_hat` means too small a standard error.
This is inherited from the statsmodels-standard Hall-Sheather/Epanechnikov
recipe and is not specific to growth-at-risk.

Coverage of a nominal 95% interval for a slope coefficient, Monte Carlo on an
exact-truth design (correctly specified, Gaussian, homoskedastic, persistent
AR(1) conditioner, 4000 replications per cell), reported as
**Powell → Newey-West**:

| `horizon` | T=200, τ=0.05 | T=200, τ=0.50 | T=500, τ=0.05 | T=500, τ=0.50 |
|----------:|:--------------|:--------------|:--------------|:--------------|
| 1  | 0.899 → 0.899 | 0.958 → 0.958 | 0.916 → 0.916 | 0.951 → 0.951 |
| 2  | 0.850 → 0.858 | 0.916 → 0.936 | 0.882 → 0.894 | 0.912 → 0.936 |
| 4  | 0.763 → 0.798 | 0.827 → 0.914 | 0.791 → 0.850 | 0.812 → 0.922 |
| 8  | 0.642 → 0.714 | 0.716 → 0.897 | 0.682 → 0.822 | 0.719 → 0.916 |
| 12 | 0.614 → 0.686 | 0.671 → 0.884 | 0.631 → 0.794 | 0.650 → 0.906 |

How to read it. **At the median the correction is essentially the whole
story** — 0.671 → 0.884 at `T=200, h=12`, and 0.906 at `T=500`. **In the 5%
tail it is half the story.** The residual gap there is effect (2): the
measured `f_hat / f` ratio at `tau = 0.05` runs 1.23 (h=1) to 1.56 (h=12) at
`T=200`, and 1.12 to 1.28 at `T=500` — it shrinks with the sample but not
with anything you can pass to this function. At `tau = 0.5` that ratio stays
within 0.96-1.06, which is exactly why the median column behaves.

**What to do about it.** For a one-year (h=4) read on 50 years of quarterly
data, treat the `tau = 0.05` interval as roughly 80% rather than 95%. For
`h = 8` or beyond in the tail, do not quote a coefficient interval from `bse`
at all — quote the fitted quantile path and its economic movement, which is
what ABG's own figures do, or bootstrap the whole procedure. `bse_powell` is
strictly worse for this purpose and is kept only for statsmodels replication.
The one case where `bse_powell` is the better number is a serially
uncorrelated conditioner, where there is nothing to correct and the
Newey-West lag terms only add noise (measured cost there: about 5 points of
coverage at `h = 12, T = 250`). Macro conditioners are essentially never
that, and `y_t` is in the design regardless, so the correction is on by
default.

**Validated against.** statsmodels `QuantReg` per tau on the identical
`[const, conditions, y_t]` design, plus a numpy sort for the rearrangement, at
1e-6 — that pins `params`, `bse_powell`, and the fitted paths
([`fixtures/tsecon-quantile.json`](../../../fixtures/tsecon-quantile.json)).
The Newey-West `bse` has no statsmodels counterpart; it is pinned to an
independent numpy assembly of the same sandwich, and to an exact no-op at
`horizon = 1`, in the crate's golden tests, with the coverage improvement
above re-measured as a seeded property test
(`crates/tsecon-quantile/tests/properties.rs`).

**References.** Adrian, Boyarchenko & Giannone (2019, *American Economic
Review* 109:1263-1289); Chernozhukov, Fernandez-Val & Galichon (2010,
*Econometrica* 78:1093-1125, rearrangement); Koenker & Bassett (1978);
Hansen & Hodrick (1980, *JPE* 88:829-853) and Newey & West (1987,
*Econometrica* 55:703-708) for the overlapping-horizon correction.

The DGP builds in the ABG mechanism: tighter financial conditions `f` lower
the conditional *mean* of growth and raise its *variance*, so the lower tail
should move roughly `-0.5 - 0.35·1.645 ≈ -1.08` per unit of `f` while the
upper tail stays near flat:

```python
import numpy as np, tsecon

rng = np.random.default_rng(1)
n = 400
# f = financial-conditions index (higher = tighter), AR(1)
f = np.zeros(n)
e = rng.standard_normal(n)
for t in range(1, n):
    f[t] = 0.8 * f[t - 1] + 0.6 * e[t]
# growth: tighter conditions lower the mean AND raise the variance
y = np.zeros(n)
eps = rng.standard_normal(n)
for t in range(1, n):
    y[t] = 2.0 + 0.3 * y[t - 1] - 0.5 * f[t - 1] + np.exp(0.35 * f[t - 1]) * eps[t]

g = tsecon.growth_at_risk(y, f.reshape(-1, 1), horizon=1)   # taus default 5/25/50/75/95
params = np.asarray(g["params"])       # rows = taus, cols = [const, f, y_t]
bse = np.asarray(g["bse"])
print("tau    const     f-slope    y-slope")
for i, tau in enumerate(g["taus"]):
    print(f"{tau:.2f}  {params[i,0]:+.3f}   {params[i,1]:+.3f} ({bse[i,1]:.3f})  {params[i,2]:+.3f}")
print("crossing (raw paths):", g["crossing"])

fitted = np.asarray(g["fitted"])       # (n_taus, T) conditional quantiles, rearranged
tight = int(np.argmax(f))              # date with the tightest conditions in sample
loose = int(np.argmin(f))
print(f"\nnext-period growth quantiles, tightest (f={f[tight]:+.2f}) vs "
      f"loosest (f={f[loose]:+.2f}) conditions:")
for i, lbl in [(0, "Q05"), (2, "Q50"), (4, "Q95")]:
    print(f"  {lbl}: {fitted[i, tight]:+.2f} vs {fitted[i, loose]:+.2f}   "
          f"(moved {fitted[i, loose] - fitted[i, tight]:+.2f})")
cur = np.asarray(g["current"])         # the last-observation risk read
print(f"\ncurrent read (f={f[-1]:+.2f}):  "
      + "  ".join(f"Q{int(t*100)}={c:+.2f}" for t, c in zip(g["taus"], cur)))
# tau    const     f-slope    y-slope
# 0.05  -0.064   -0.771 (0.098)  +0.449
# 0.25  +0.879   -0.759 (0.090)  +0.415
# 0.50  +1.641   -0.491 (0.088)  +0.416
# 0.75  +2.526   -0.284 (0.106)  +0.365
# 0.95  +3.671   +0.113 (0.161)  +0.333
# crossing (raw paths): False
#
# next-period growth quantiles, tightest (f=+1.88) vs loosest (f=-2.01) conditions:
#   Q05: -0.19 vs +2.85   (moved +3.04)
#   Q50: +1.94 vs +3.89   (moved +1.95)
#   Q95: +4.86 vs +4.45   (moved -0.41)
#
# current read (f=-0.16):  Q5=+1.59  Q25=+2.42  Q50=+3.14  Q75=+3.81  Q95=+4.78
```

The f-slope column *is* the ABG picture: `-0.771` at the 5th percentile fading
monotonically to `+0.113` (insignificant) at the 95th. Moving from the
loosest to the tightest conditions in the sample drags the 5% quantile down by
3 points — from comfortable growth to outright contraction — while the 95%
quantile barely notices. The distribution of future growth doesn't shift under
financial stress; its left tail stretches.
