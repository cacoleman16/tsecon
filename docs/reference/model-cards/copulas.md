# Model card — Static copulas

**Family:** `pseudo_obs`, `copula_fit`, `copula_select`

Dependence, separated from the margins. Sklar's theorem factors any joint
distribution into its marginals and a *copula* — the joint distribution of
the ranks — so you can model "how do these two series move together,
especially in the tails" without committing to any marginal model at all.
This slice ships the five workhorse bivariate families: **Gaussian** (the
correlation benchmark — and provably *zero* tail dependence, the classic
reason correlation understates joint crashes), **Student-t** (elliptical
like the Gaussian, but with symmetric tail dependence controlled by `nu`),
**Clayton** (lower-tail dependent — joint crashes), **Gumbel** (upper-tail
dependent — joint booms), and **Frank** (radially symmetric, no tail
dependence, covers negative dependence).

| Function | Role |
|----------|------|
| `pseudo_obs` | Rank transform `u = rank/(n+1)` — raw margins to probability scale |
| `copula_fit` | One family: MLE or Kendall-tau inversion, SEs, AIC/BIC, tail dependence |
| `copula_select` | All requested families ranked by AIC/BIC, teaching verdict |

## What it estimates

- **`pseudo_obs(x)`** — columnwise average ranks scaled by `n + 1`
  (exactly scipy's `rankdata(method="average")/(n+1)`, ties included).
  Ranks see only order, so the output — and any copula fitted to it — is
  **invariant to strictly monotone transforms of each margin**: fitting on
  `x` or on `exp(x)` gives bit-identical results (property-tested). That
  invariance *is* the point of the copula decomposition.
- **`copula_fit(u, family, method)`** — the dependence parameter(s):
  `rho` for the elliptical families (`nu` too, for the t), `theta` for
  the Archimedeans, with observed-information standard errors (MLE),
  the log-likelihood, AIC/BIC, the empirical and fit-implied Kendall tau,
  and the family's **closed-form tail-dependence coefficients** — the
  probability that one margin is extreme given the other is:
  Gaussian/Frank `(0, 0)`; t `lambda = 2 t_{nu+1}(-sqrt((nu+1)(1-rho)/(1+rho)))`
  in both tails (Demarta-McNeil 2005); Clayton lower `2^(-1/theta)`;
  Gumbel upper `2 - 2^(1/theta)`.
- **`copula_select(u, families)`** — every family on the same data, ranked
  by AIC and BIC, with a verdict that says who wins, by how much, whether
  the two criteria agree, and what the winner implies for joint extremes.

## Assumptions

- **The pairs are i.i.d. draws from one joint distribution.** On raw time
  series the fit reads as *unconditional* dependence; serial dependence
  makes the effective sample smaller than `n` (SEs too tight). For
  conditional dependence, filter each margin first (e.g. GARCH
  standardized residuals) — the same two-step logic as EVT-POT.
- **Continuous margins.** Heavy ties (discretized data) push tau-b and the
  pseudo-observations away from the continuous theory; a genuinely
  constant column raises.
- **Bivariate, this slice.** `d > 2` for the Gaussian/t families (a
  correlation-matrix parameterization) is a natural extension, deferred.
  Clayton/Gumbel are fitted on their **positive-dependence branch only** —
  rotated/survival variants are deferred, and negative-tau data raises a
  teaching error pointing to Frank/Gaussian instead.

## When to use

- **Tail risk in pairs**: does a crash in one series make a crash in the
  other more likely *in the limit*? Fit t/Clayton/Gumbel and read the
  tail-dependence coefficients; compare against the Gaussian with
  `copula_select` — if the t wins by a wide AIC margin, correlation alone
  is understating joint extremes.
- **Dependence robust to marginal misspecification**: pseudo-observations
  make the answer identical for prices, returns, or log-returns.
- **Not** for time-varying dependence (dynamic copulas are out of scope),
  not for `d > 2` portfolios (vine-copula territory — see the non-goals),
  and not on short samples (fewer than 20 pairs raises; serious work wants
  hundreds).

## Key arguments and defaults

`copula_fit(u, family="gaussian", method="mle")`:

- `u` — an `(n, 2)` matrix strictly inside `(0, 1)`. **The caller
  rank/PIT-transforms first**; `pseudo_obs(x)` does it in one line, and
  its `n + 1` denominator is what keeps every value off the 0/1 boundary
  (where the elliptical quantile transforms blow up — passing raw data
  raises an error that says exactly this).
- `family` — `"gaussian"`, `"t"`, `"clayton"`, `"gumbel"`, `"frank"`.
- `method="mle"` — maximum likelihood on the copula density (BFGS + a
  tight Nelder-Mead polish, started from the tau inversion), with
  observed-information SEs. `method="tau"` — Kendall-tau inversion
  through each family's closed form (`rho = sin(pi tau/2)`,
  `theta = 2tau/(1-tau)` Clayton, `theta = 1/(1-tau)` Gumbel, Frank by
  root-finding the exact Debye-function tau); for the t family tau pins
  `rho` only and `nu` is profiled by MLE. Tau fits report **NaN SEs with
  `se_valid=False`** — the moment-based SE is deferred, not faked.

`copula_select(u, families=None, method="mle")` defaults to all five
families; Clayton/Gumbel are *skipped with a reason* (not failed) when
the empirical tau is non-positive, so the default menu works on any data.

## How to read the output

`rho`/`theta` (+`nu`) with `se_*` are the headline; `se_valid` is the
honesty flag — `False` means the observed information failed (e.g. the t
family's `nu` drifting to its Gaussian-limit barrier on near-Gaussian
data, where the likelihood is flat in `nu`) or `method="tau"`.
`tau` vs `tau_implied` is a quick specification check: a family whose
implied tau sits far from the data's tau is fighting the data.
`tail_lower`/`tail_upper` are the economically loaded numbers: the
Gaussian's are *identically zero at any* `rho < 1` — if joint extremes
matter and the t/Clayton/Gumbel fit is competitive, prefer it. In
`copula_select`, a `dAIC < 2` between the top families is statistically
near-indistinguishable — the verdict says so and tells you to choose on
tail behavior, and it flags when BIC (which charges more for the t's
second parameter) disagrees with AIC.

## Failure modes

- **Raw data passed as `u`** — anything at or outside `(0, 1)` raises
  with the `pseudo_obs` pointer. This is the most common copula-API
  mistake and it is caught loudly, never silently clipped.
- **Perfectly monotone pairs** (`u2` a deterministic transform of `u1`):
  every family's parameter sits at its boundary; the fit refuses
  (`|tau| = 1` is a functional relationship, not stochastic dependence).
- **Negative dependence with Clayton/Gumbel** — raises (this slice has no
  rotations); Frank and Gaussian cover negative dependence.
- **`nu` at the barrier**: on near-Gaussian data the t family's `nu` runs
  toward its upper bound (1000) with a flat likelihood — the fit returns,
  matches the Gaussian log-likelihood from above, and reports
  uncertified SEs rather than fabricated ones.
- **Serial dependence** — SEs are i.i.d.-based; filter the margins first.

## Validated against

statsmodels 0.14.6 `statsmodels.distributions.copula` and scipy 1.17.1
(`fixtures/tsecon-copula.json`): per-family pdf/logpdf/cdf grids at 1e-10
(the Gaussian CDF against the exact Owen's-T closed form — equal to
scipy's `multivariate_normal.cdf` at 5e-15; the t CDF, which statsmodels
does not implement, against scipy `quad` on the exact conditional 1-D
integral, cross-checked against `multivariate_t.cdf` at that reference's
~2e-4 QMC noise); tau-inversion fits against `fit_corr_param` (closed
maps at 1e-12); MLE fits against a Nelder-Mead-polished scipy optimum of
the statsmodels log-density (statsmodels exposes **no** copula MLE) —
parameters at 1e-6, log-likelihood at 1e-10, observed-information SEs at
1e-4; Kendall tau against `scipy.stats.kendalltau` at 1e-15 (ties
included) and `pseudo_obs` against `rankdata` exactly; tail dependence
against the closed forms, each verified in the generator by the numeric
copula limit — which caught a genuine defect in the reference:
statsmodels 0.14.6 `StudentTCopula.dependence_tail` mis-computes the t
formula through an operator-precedence slip (0.1438 where the true value
is 0.2532 at `rho=0.5, nu=4`); the correct Demarta-McNeil form is what
tsecon ships and pins. Recovery on simulated data from every family and
the monotone-invariance/exchangeability bit-exactness claims are
property-tested in Rust.

## References

Sklar (1959); Joe (2014), *Dependence Modeling with Copulas*; Genest
(1987) for Frank's tau; Genz (2004) for the bivariate-normal CDF;
Demarta & McNeil (2005) for the t copula; McNeil, Frey & Embrechts
(2015), ch. 7.

## Runnable example

```python
import numpy as np, tsecon

rng = np.random.default_rng(7)

# Joint-crash-prone data: a t copula (rho = 0.6, nu = 4) hiding behind two
# arbitrary margins — a "return"-looking one and a "price-level" one.
n = 1000
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
t_pair = z / np.sqrt(rng.chisquare(4, size=n) / 4)[:, None]
x = np.column_stack([np.exp(0.01 * t_pair[:, 0]) - 1.0,
                     100 + 5 * t_pair[:, 1]])

u = tsecon.pseudo_obs(x)                  # margins gone; only ranks remain
fit = tsecon.copula_fit(u, family="t")
print("rho:", round(fit["rho"], 3), "+/-", round(fit["se_rho"], 3),
      " nu:", round(fit["nu"], 2), "+/-", round(fit["se_nu"], 2))
# rho: 0.576 +/- 0.024  nu: 3.63 +/- 0.58        (truth: 0.6, 4)
print("tail dependence:", round(fit["tail_lower"], 3))
# tail dependence: 0.319

sel = tsecon.copula_select(u)             # all five families
print(sel["ranking_aic"])
# ['t', 'gaussian', 'gumbel', 'frank', 'clayton']
print(sel["verdict"])
# t minimizes AIC (-449.05), dAIC 63.46 over the runner-up gaussian; BIC
# agrees. The winner implies Kendall tau 0.391 and lower/upper tail
# dependence 0.319/0.319 — joint extremes stay dependent in the limit.
```

The Gaussian runner-up reports nearly the same `rho` (0.570) — correlation
alone cannot tell these models apart. The 63-point AIC gap and the 0.32
tail-dependence coefficient are the difference between "correlated" and
"they crash together": with 1,000 observations the data reject the
zero-tail-dependence Gaussian decisively.
