# Frontier Monte Carlo

The [Monte Carlo suite](monte-carlo.md) verifies that individual estimators have
the properties they advertise. This page asks the harder, *comparative* questions
that applied practice actually argues about — where the answer is a trade-off
rather than a verdict.

Every table is real output from
[`monte_carlo_frontier.py`](monte_carlo_frontier.py), reproducible from a fixed
seed:

```sh
.venv/bin/python docs/examples/monte_carlo_frontier.py     # ~1 s
```

---

## F1 · Local projections vs VAR: the bias/variance trade-off

**The question.** Plagborg-Møller & Wolf (2021) showed that local projections and
VARs *estimate the same impulse response* — they are not rival objects. So the
choice between them is not about identification; it is about **finite-sample
bias versus variance**, which is the question Li, Plagborg-Møller & Wolf (2024)
take up. LP is more robust to dynamic misspecification; a correctly specified VAR
is more efficient. This experiment measures both sides of that.

**Design.** An observed exogenous shock entering an AR(2) outcome, so the truth
is computable in closed form:

```text
z_t ~ N(0, 1)                                          (exogenous, observed)
y_t = 0.6·y_{t-1} − 0.15·y_{t-2} + 1.0·z_t + 0.5·z_{t-1} + u_t
```

Both estimators target the response of `y` to a unit `z` shock. Each is run
twice: once with the **correct** lag order, once **truncated** to one lag.
`reps=400, T=240`.

### Correctly specified — the VAR is more efficient

| h | truth | LP bias | LP sd | LP rmse | VAR bias | VAR sd | VAR rmse |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0002 | 0.0364 | 0.0364 | −0.0019 | 0.0553 | 0.0552 |
| 1 | 1.1000 | −0.0030 | 0.0713 | 0.0713 | −0.0047 | 0.0873 | 0.0873 |
| 2 | 0.5100 | −0.0108 | 0.1083 | 0.1087 | −0.0104 | 0.1094 | 0.1098 |
| 4 | 0.0081 | −0.0152 | 0.1173 | 0.1181 | −0.0106 | 0.0945 | 0.0950 |
| 8 | −0.0008 | −0.0049 | 0.1204 | 0.1204 | 0.0061 | 0.0094 | **0.0112** |
| 12 | 0.0000 | −0.0127 | 0.1184 | 0.1189 | −0.0001 | 0.0014 | **0.0014** |

Both are essentially unbiased, exactly as the equivalence result implies. But the
VAR's **variance is dramatically smaller at long horizons** — at h=12 its RMSE is
0.0014 against LP's 0.1189, roughly **eighty times** tighter. The reason is
structural, not incidental: the VAR extrapolates long-horizon responses from a
handful of estimated coefficients, while LP runs a fresh, increasingly noisy
regression at every horizon.

### Lag-truncated — the VAR acquires bias, LP does not

| h | truth | LP bias | LP sd | LP rmse | VAR bias | VAR sd | VAR rmse |
|---|---|---|---|---|---|---|---|
| 0 | 1.0000 | 0.0035 | 0.0479 | 0.0479 | −0.0012 | 0.0557 | 0.0557 |
| 1 | 1.1000 | −0.0014 | 0.0719 | 0.0718 | −0.0017 | 0.0883 | 0.0882 |
| 2 | 0.5100 | −0.0096 | 0.1086 | 0.1089 | −0.0335 | 0.0911 | 0.0970 |
| 4 | 0.0081 | −0.0159 | 0.1159 | 0.1168 | **+0.0789** | 0.0626 | 0.1007 |
| 8 | −0.0008 | −0.0043 | 0.1203 | 0.1203 | 0.0064 | 0.0078 | 0.0100 |
| 12 | 0.0000 | −0.0125 | 0.1182 | 0.1187 | 0.0005 | 0.0010 | 0.0011 |

Averaged over h = 1…12:

| specification | \|bias\| LP | \|bias\| VAR | rmse LP | rmse VAR |
|---|---|---|---|---|
| correct (2 lags) | 0.0090 | 0.0056 | 0.1115 | 0.0445 |
| truncated (1 lag) | 0.0089 | **0.0241** | 0.1112 | 0.0451 |

**The trade-off, made concrete.** Dropping a lag leaves LP's bias *completely
unmoved* (0.0090 → 0.0089) while the VAR's **quadruples** (0.0056 → 0.0241). At
h=4 the VAR's bias flips from −0.011 to **+0.079** and becomes larger than its
own standard deviation (0.063) — the error is now dominated by misspecification
rather than sampling noise, which is precisely the regime where confidence
intervals mislead.

**But we are not going to overclaim.** Even truncated, the VAR's *average RMSE
stays lower* (0.0451 vs 0.1112), because its variance advantage in this DGP is
large enough to outweigh the bias it picked up. The honest summary is:

> Lag truncation costs the VAR its unbiasedness but not, here, its overall
> accuracy. LP buys robustness; whether that robustness is worth ~2.5× the
> variance depends on how badly specified you fear your VAR is and which horizon
> you care about.

That conditional answer *is* the finding. A page that concluded "use LP" would be
selling you a result this simulation does not support.

---

## F2 · LP-IV with a weak instrument

**The question.** LP-IV identifies a dynamic causal effect by instrumenting the
impulse. Weak instruments are the classic failure mode — but *which* part of the
inference actually breaks first?

**Design.** An endogenous impulse `x = π·z + v` with the outcome error correlated
with `v` (`corr = 0.7`), so OLS is biased upward. True impact effect = 1.0.
Instrument strength `π` is varied. `reps=500, T=300`.

| π | mean first-stage F | coverage (h=0) | median β̂ |
|---|---|---|---|
| 0.05 | 1.68 | 0.940 | **1.2863** |
| 0.10 | 4.09 | 0.916 | 1.0947 |
| 0.20 | 13.33 | 0.930 | 0.9972 |
| 0.50 | 76.86 | 0.964 | 0.9883 |

**The result is not the one you might predict.** Coverage of the nominal 95%
interval barely moves — it sits between 0.92 and 0.96 across the whole range,
never collapsing. What *does* break is the **point estimate**: at `F = 1.68` the
median estimate is **1.29 against a truth of 1.0, a 29% bias** toward the
endogenous OLS value. By `F ≈ 13` the bias is essentially gone (0.997).

The explanation is that weak instruments inflate the LP-IV standard errors at the
same time as they bias the estimate, and the wider interval partly compensates —
so the interval keeps covering while the number at its centre is wrong.

**The practical lesson.** With a weak instrument, do not be reassured by an
interval that looks well-behaved. Read the **first-stage F**, and treat a small
one as a warning about the estimate itself. `tsecon.lp_iv` returns
`first_stage_f` per horizon precisely so this diagnostic is never out of sight:

```python
fit = tsecon.lp_iv(y, impulse, instrument, horizons=12)
fit["first_stage_f"][0]     # check this before reading fit["irf"]
```

---

## Why these live in the repository

Both experiments are cheap (≈1 second combined) and seeded, so they run in CI
alongside the unit tests. That matters: they are not marketing figures produced
once and pasted in, but live claims that would break loudly if an estimator
regressed. See the [validation matrix](../reference/validation-matrix.md) for the
fixture-level, reference-implementation half of the same story.

**References.** Plagborg-Møller, M. & Wolf, C. K. (2021), "Local Projections and
VARs Estimate the Same Impulse Responses," *Econometrica* 89(2):955-980. Li, D.,
Plagborg-Møller, M. & Wolf, C. K. (2024), "Local Projections vs. VARs: Lessons
from Thousands of DGPs," *Journal of Econometrics*. Stock, J. H. & Watson, M. W.
(2018), "Identification and Estimation of Dynamic Causal Effects in Macroeconomics
Using External Instruments," *Economic Journal* 128:917-948.
