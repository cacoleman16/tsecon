# Monte Carlo validation

Golden fixtures prove tsecon reproduces a *reference implementation* on one
dataset. They cannot prove a **statistical property** — that a test holds its
size, that an interval covers at its nominal rate, that an estimator is
consistent. Those are claims about repeated sampling, and the only honest way to
check them is to simulate.

This page is the executable evidence. Every table below is real output from
[`monte_carlo.py`](monte_carlo.py), reproducible from a fixed seed:

```sh
.venv/bin/python docs/examples/monte_carlo.py     # ~23 s
```

Each experiment states what it *expects* before it reports what it *observed*,
so a regression shows up as a contradiction rather than a number you have to
squint at.

---

## 1 · IVX predictive regressions hold their size at a unit root

**The property.** Regressing a return on a *persistent, endogenous* predictor is
the Stambaugh setting, where the naive OLS t-test over-rejects — the closer the
predictor's root is to one, the worse it gets. The IVX Wald test
([Kostakis, Magdalinos & Stamatogiannis 2015](../reference/model-cards/predictive-regressions.md))
is designed to keep its size *uniformly* over that persistence. A test that
cannot hold its size will manufacture predictability that is not there.

`reps=2000, T=250, corr(u,e) = −0.95, nominal level 0.05`

**Size** — the true slope is zero, so 0.05 is the target rejection rate:

| ρ | OLS t-test | IVX Wald |
|---|---|---|
| 0.90 | 0.065 | **0.062** |
| 0.95 | 0.076 | **0.057** |
| 0.99 | 0.140 | **0.051** |
| 1.00 | 0.278 | **0.053** |

The OLS t-test rejects a true null **28% of the time at an exact unit root** —
five and a half times its nominal rate. IVX sits on 0.05 across the whole range,
including ρ = 1.

**Power** — at ρ = 0.95, rejection should climb as the true slope grows:

| β | OLS t-test | IVX Wald |
|---|---|---|
| 0.00 | 0.083 | 0.058 |
| 0.05 | 0.861 | 0.736 |
| 0.10 | 1.000 | 1.000 |

Size control is not bought with dead power: IVX reaches 0.74 at β = 0.05 and 1.0
at β = 0.10. (OLS's higher raw rejection at β = 0.05 is not a fair win — its
size is inflated to begin with, so some of those rejections are the same false
positives the size table just exposed.)

---

## 2 · HAC standard errors rescue coverage under serial correlation

**The property.** A 95% confidence interval for a mean should contain the truth
95% of the time. Under serial correlation, IID standard errors understate the
variance of the sample mean and the interval collapses. HAC (Newey-West)
standard errors are the fix.

`reps=2000, T=200, true mean = 0, nominal coverage 0.95`

| φ | IID coverage | HAC coverage | IID width | HAC width |
|---|---|---|---|---|
| 0.00 | 0.941 | 0.937 | 0.277 | 0.274 |
| 0.50 | 0.746 | **0.894** | 0.318 | 0.467 |
| 0.80 | 0.488 | **0.769** | 0.451 | 0.845 |
| 0.95 | 0.247 | **0.451** | 0.790 | 1.667 |

Two honest readings. First, the win: at φ = 0.8, IID intervals cover only 49% of
the time — you would call a null false twice as often as you should — while HAC
recovers to 77%. And at φ = 0, HAC costs nothing (0.937 vs 0.941, essentially
identical widths), so there is no penalty for using it defensively.

Second, the limitation we are *not* going to hide: **HAC improves coverage but
does not fully repair it near a unit root.** At φ = 0.95 it reaches 0.451, still
far from 0.95. That is expected — a fixed-bandwidth kernel estimator cannot
consistently estimate a long-run variance that is exploding — and it is exactly
why the persistent-regressor problem needs IVX (§1) rather than "just use HAC."

---

## 3 · The AR(1) slope estimator is consistent, with the textbook bias

**The property.** The OLS estimator of an AR(1) coefficient is biased downward
in finite samples by approximately `−(1 + 3φ)/T` (Kendall), and both that bias
and the RMSE should vanish as `T → ∞`.

`reps=1000, true φ = 0.7, estimator = OLS via var_fit, predicted bias = −3.1/T`

| T | mean φ̂ | bias | predicted −(1+3φ)/T | RMSE |
|---|---|---|---|---|
| 100 | 0.6661 | −0.0339 | −0.0310 | 0.0849 |
| 400 | 0.6914 | −0.0086 | −0.0077 | 0.0369 |
| 1600 | 0.6985 | −0.0015 | −0.0019 | 0.0180 |
| 6400 | 0.6994 | −0.0006 | −0.0005 | 0.0089 |

Two properties confirmed at once. The bias shrinks toward zero **and tracks the
closed-form prediction at every sample size** (−0.0339 vs −0.0310; −0.0086 vs
−0.0077; −0.0015 vs −0.0019) — it is the *known* finite-sample bias, not an
implementation error. And the RMSE falls by a factor of ~2 for every 4× increase
in `T` (0.0849 → 0.0369 → 0.0180 → 0.0089), the √T convergence rate.

---

## Why this page exists

A reference-implementation match tells you the arithmetic agrees on one sample.
It says nothing about whether a test is *valid*. These simulations are how the
library earns the claims its model cards make — and because they run in seconds
from a fixed seed, they are cheap enough to keep honest.

See also the [validation matrix](../reference/validation-matrix.md) for the
fixture-level, reference-implementation side of the same story.
