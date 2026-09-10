# Chapter 16 — LP versus VAR, Head to Head

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** the VAR and its impulse responses ([Chapter 7](07-multivariate.md)), recursive identification ([Chapter 8](08-causal-identification.md)), and local projections with lag-augmented inference ([Chapter 9](09-local-projections.md)).

**You will learn:**

- What Plagborg-Møller and Wolf (2021) actually prove — that local projections and VARs share an *estimand* — and what that theorem does and does not say about the two *estimates* you compute from one sample
- Exactly which horizon the finite-sample LP and VAR point estimates coincide at, why, and what "same controls, same lag length" has to mean for the identity to hold — verified numerically, not asserted
- The Li, Plagborg-Møller and Wolf (2024) bias-variance trade-off reproduced on a seeded Monte Carlo you can re-run: bias, standard deviation, RMSE and band coverage for `var_irf`, `lp` and `smooth_lp` at short and long lag lengths
- The decision rule this library recommends, and where it lives in the [decision guide](../which-model-when.md#2c-i-want-a-single-equation-irf-no-full-var)

Every number in this chapter is printed by one committed script, [`docs/examples/lp_vs_var_head_to_head.py`](../examples/lp_vs_var_head_to_head.py), run once on a **release** build of the extension; its full output is reproduced verbatim at the [end of the chapter](#provenance-and-the-verbatim-output), and the command that regenerates it is shown there.

## The question

You have a shock series and an outcome, and you want the outcome's response at horizons 0 through 20. Chapter 7 gave you one way — fit a VAR, invert it, read the moving-average coefficients — and Chapter 9 gave you another — regress the outcome at each horizon directly on the shock. For two decades the choice was argued like a sports rivalry: VAR people called LPs noisy and wasteful, LP people called VARs misspecified and over-confident. Both were describing real properties of the estimators. Neither was describing a difference in what the estimators *estimate*.

This chapter settles the question the way this library settles questions: by measuring. Two results organise the modern literature. The first, Plagborg-Møller and Wolf (2021), is a theorem about the estimand — the object both methods are trying to recover. The second, Li, Plagborg-Møller and Wolf (2024), is a large simulation study about the estimators — how far, and in which direction, each lands from that object in samples of the size macroeconomists actually have. The chapter states the first precisely and checks its finite-sample content on tsecon's own functions, reproduces the second on a small seeded design, and then writes down a rule.

## What Plagborg-Møller and Wolf prove — and what they do not

Take the two-variable case that carries all the intuition. $x_t$ is the impulse variable — an observed shock, or the variable you would order first in a recursive VAR — and $y_t$ is the outcome. The recursive VAR($p$) regresses $(x_t, y_t)$ on a constant and $p$ lags of *both* variables, takes the Cholesky factor of the residual covariance with $x$ first, and reads the horizon-$h$ response of $y$ off the moving-average representation. The matching local projection regresses $y_{t+h}$ on $x_t$ and the *same* controls — a constant and $p$ lags of both variables — and reads the response off the coefficient on $x_t$.

**The theorem (population, $p \to \infty$).** With the lag length unrestricted, the LP coefficient at every horizon equals the recursive VAR's impulse response at that horizon, once the VAR response is scaled from a one-standard-deviation shock to a unit movement in $x$ (divide by the VAR's own impact response of $x$). Any identification scheme you can write as a linear projection in one framework you can write in the other: ordering $x$ first in the Cholesky factorisation is the same identifying assumption as putting $x_t$ first among the LP's contemporaneous controls. **LP versus VAR is not a disagreement about the estimand.** It never was.

**What "same controls, same lag length" buys.** Three things, and it is worth separating them because the popular summary — "LP and VAR give the same answer" — blurs all three:

1. *The same identification.* Matched controls make the two estimators target the same population impulse response. Different controls target different objects, and the gap between them is not estimation error; it does not shrink with $T$.
2. *One exact finite-sample identity, at horizon zero only.* With matched controls, the LP coefficient at $h = 0$ and the unit-normalised VAR impact response are the **same number to machine precision in every sample**. The reason is the Frisch-Waugh-Lovell theorem: the Cholesky impact response of $y$ to the $x$ shock, divided by the impact on $x$, is the OLS coefficient of the $y$ residual on the $x$ residual, and both residuals were formed by projecting on exactly the LP's control set. Nothing about $h \geq 1$ is exact: from horizon one on, the VAR estimate is a *product* of estimated one-step coefficient matrices, while the LP estimate is a *direct* projection. Under a correctly specified VAR($p$) both are consistent for the same number, and their gap is $O(T^{-1/2})$ — but it is a gap, in every finite sample.
3. *A diagnostic.* Because the two are consistent for the same object only when the VAR's lag structure is adequate, a divergence at long horizons that does not shrink as you add lags is telling you the VAR is extrapolating dynamics it did not estimate. That is the practical content of the theorem, and it is why the rule at the end of this chapter says *fit both*.

### Where the identity lives — checked, not asserted

The chapter's script simulates one draw of a stable bivariate Gaussian VAR(2) with $x$ first, fits `tsecon.var_irf` with `lags=2` and `tsecon.lp` with `n_lag_controls=2`, prints the gap at every horizon, and reports the horizons at which the two agree to $10^{-12}$. It also rebuilds the LP with the *matched* control set the theorem needs — a constant, $x_t$, and two lags of **both** variables — using `tsecon.ols`, because `tsecon.lp` does not build that regression. Here is what `lp` regresses on, from the [model card](../reference/model-cards/local-projections.md): $y_{t+h}$ on $x_t$, a constant, `n_lag_controls` own-lags of $y$, and — on the default lag-augmented path only — $h$ lags of $x$; on the HAC path, no lags of $x$ at all. That control set is designed for an innovation-like impulse (a measured shock, the use case of Chapter 9), and it is *not* the recursive VAR's control set unless $h = p$.

The measured result, on a VAR(2) whose $x$ has its own AR dynamics (a generic VAR column), $T = 400$:

```text
[x has its own AR dynamics]  var_fit is_stable=True  min_root=1.489
   h     VAR(2)  lp lag-aug     lp hac    matched  |VAR-lagaug|  |VAR-hac|  |VAR-matched|
   0   0.514865    0.558012   0.558012   0.514865      4.31e-02   4.31e-02       2.11e-15
   1   0.667378    0.665306   0.715191   0.669839      2.07e-03   4.78e-02       2.46e-03
   2   0.533931    0.543530   0.572043   0.543530      9.60e-03   3.81e-02       9.60e-03
   3   0.349863    0.307702   0.330219   0.304655      4.22e-02   1.96e-02       4.52e-02
   4   0.213444    0.232117   0.220656   0.226679      1.87e-02   7.21e-03       1.32e-02
   5   0.132457    0.152647   0.130262   0.153214      2.02e-02   2.20e-03       2.08e-02
   6   0.086593    0.052244   0.071994   0.066872      3.43e-02   1.46e-02       1.97e-02
   7   0.058813    0.057614   0.061325   0.057800      1.20e-03   2.51e-03       1.01e-03
   8   0.040322    0.056688   0.050385   0.027818      1.64e-02   1.01e-02       1.25e-02
   9   0.027438    0.080453   0.061500   0.056462      5.30e-02   3.41e-02       2.90e-02
  10   0.018480    0.065972   0.064210   0.054210      4.75e-02   4.57e-02       3.57e-02
  11   0.012375    0.057669   0.067258   0.053895      4.53e-02   5.49e-02       4.15e-02
  12   0.008279    0.108406   0.116483   0.112833      1.00e-01   1.08e-01       1.05e-01
  horizons where VAR(2) == lp lag-augmented to 1e-12: NONE
  horizons where VAR(2) == lp hac to 1e-12: NONE
  horizons where VAR(2) == matched-controls LP to 1e-12: [0]
```

And on the same VAR(2) with the $x$ equation switched off, so $x$ is white noise — the innovation-like impulse `lp` is built for:

```text
[x is white noise (innovation-like)]  var_fit is_stable=True  min_root=2.344
   h     VAR(2)  lp lag-aug     lp hac    matched  |VAR-lagaug|  |VAR-hac|  |VAR-matched|
   0   0.507213    0.492431   0.492431   0.507213      1.48e-02   1.48e-02       3.33e-16
   1   0.621153    0.621171   0.611203   0.623697      1.74e-05   9.95e-03       2.54e-03
   2   0.330462    0.332568   0.331594   0.332568      2.11e-03   1.13e-03       2.11e-03
   3   0.042365    0.039441   0.053867   0.047709      2.92e-03   1.15e-02       5.34e-03
   4  -0.043486   -0.077855  -0.069213  -0.078723      3.44e-02   2.57e-02       3.52e-02
   5  -0.029515   -0.097902  -0.098635  -0.103148      6.84e-02   6.91e-02       7.36e-02
   6  -0.007019   -0.099395  -0.106797  -0.100339      9.24e-02   9.98e-02       9.33e-02
   7   0.001739    0.057323   0.029002   0.040614      5.56e-02   2.73e-02       3.89e-02
   8   0.002160    0.060744   0.036557   0.045155      5.86e-02   3.44e-02       4.30e-02
   9   0.000796    0.055237   0.051545   0.050643      5.44e-02   5.07e-02       4.98e-02
  10   0.000017    0.000736  -0.014363  -0.017712      7.19e-04   1.44e-02       1.77e-02
  11  -0.000136   -0.086598  -0.105074  -0.101409      8.65e-02   1.05e-01       1.01e-01
  12  -0.000073    0.080388   0.063904   0.069248      8.05e-02   6.40e-02       6.93e-02
  horizons where VAR(2) == lp lag-augmented to 1e-12: NONE
  horizons where VAR(2) == lp hac to 1e-12: NONE
  horizons where VAR(2) == matched-controls LP to 1e-12: [0]
```

Read the last three lines of each block first. **`tsecon.lp` agrees with `tsecon.var_irf` at no horizon**, on either path, and the **matched-controls LP agrees at exactly one — $h = 0$** — with gaps of 2.11e-15 and 3.33e-16, which is machine precision for a 400-observation regression. That is the whole finite-sample content of the theorem: one exact number, at impact, and only with the control set the theorem specifies. Two more things are visible in the tables:

- At $h = 2 = p$ the lag-augmented `lp` estimate and the matched LP are the same number (0.543530 in the first block, 0.332568 in the second). This is not a coincidence: at $h = p$ the lag-augmented regression's $h$ lags of $x$ *are* the $p$ lags of $x$, so the two control sets coincide there — and still neither equals the VAR, because at $h \geq 1$ nothing does.
- The $h = 0$ gap between `lp` and the VAR is 4.31e-02 when $x$ is persistent and 1.48e-02 when $x$ is white noise. The next table shows those two gaps are different in kind.

How each gap behaves as the sample grows, one draw per $T$:

```text
How each gap scales with T (max over the stated horizons, one draw per T):
       T | persistent x: lagaug h=0  matched h=0  matched h>=1 | white-noise x: lagaug h=0  matched h>=1
     200 |                 8.47e-02     1.11e-16      1.88e-01 |                  3.07e-02      1.49e-01
     800 |                 3.68e-02     7.77e-16      5.29e-02 |                  1.21e-02      2.83e-02
    3200 |                 5.06e-02     2.11e-15      2.83e-02 |                  4.77e-03      4.46e-02
   12800 |                 5.01e-02     4.88e-15      1.34e-02 |                  4.20e-03      1.73e-02
```

The matched identity at $h = 0$ holds at every $T$ (1.11e-16 to 4.88e-15 — floating-point noise that grows slowly with the number of summed terms). The matched gap at $h \geq 1$ shrinks — 1.88e-01 at $T = 200$ to 1.34e-02 at $T = 12{,}800$ — as $O(T^{-1/2})$ says it should, and so does the `lp` gap when $x$ is white noise (3.07e-02 to 4.20e-03). But when $x$ has its own dynamics, the `lp` gap at $h = 0$ does **not** shrink: 8.47e-02, 3.68e-02, 5.06e-02, 5.01e-02. That is item 1 above, made visible. Omitting the impulse's own lags from the controls changes the estimand when the impulse is forecastable from its past, so `lp` and the VAR are then estimating different population objects, and no sample size closes the gap. When your impulse is a persistent observable rather than a measured innovation, the honest choices are the recursive VAR (which conditions on every lag), the matched regression built with `tsecon.ols` as above, or pre-whitening the impulse before it enters `lp` — the same warning [Chapter 9](09-local-projections.md#inference-done-right) attaches to lag augmentation's validity.

> **⚠ Common mistake — "LP and VAR give the same IRF, so I only need one."** The theorem says they estimate the same *object*. In a sample of 400 they produce the same *number* at one horizon, under matched controls, and differ by a few hundredths to a tenth at every other horizon in the tables above. A published claim that the two "coincide" at horizons beyond impact is a statement about $T \to \infty$, not about your data.

## The bias-variance trade-off, measured

Li, Plagborg-Møller and Wolf (2024) ran the comparison over thousands of DGPs calibrated to macroeconomic data. Their lessons: LP has the lower bias and much higher variance almost everywhere; VARs the reverse; at short horizons with matched lag lengths the two nearly coincide; in mean-squared-error terms, intermediate estimators — LPs shrunk toward a VAR, or VARs with more lags than an information criterion would pick — dominate both endpoints. A guide chapter cannot run thousands of DGPs, but it can run one that no finite-order VAR nests, with enough replications that the numbers are stable, and let you read the trade-off off a table instead of a paragraph.

**The design.** The outcome is a persistent ARMA-X process driven by an observed i.i.d. shock:

$$
y_t = 1.5\, y_{t-1} - 0.54\, y_{t-2} + x_t + 0.5\, x_{t-1} + e_t + 0.6\, e_{t-1}, \qquad x_t, e_t \sim \text{i.i.d. } N(0, 1).
$$

The autoregressive roots are 0.9 and 0.6, so the response is hump-shaped and still 0.57 at horizon 20; the MA(1) error means that no VAR($p$) with finite $p$ is correctly specified, though the misspecification fades as $p$ grows (the MA(1)'s autoregressive coefficients decay like $0.6^j$). $T = 240$ — sixty years of quarterly data — with 500 replications and horizons 0 to 20. The estimand is the response of $y$ to a unit $x$ shock; VAR responses are Cholesky with $x$ first, divided by the $x$ impact, exactly as in Part 1. Eight estimators are compared: `var_irf` at $p = 1, 4, 12$; `lp` on the default lag-augmented path at $p = 1, 4, 12$; `lp(se="hac")` at $p = 4$ (the un-augmented regression); and `smooth_lp(lam="cv")` at $p = 4$. For bands, `var_irf_bands(method="asymptotic", alpha=0.05)` for the VARs, `irf ± 1.96·se` for the LPs; the VAR band covers the one-standard-deviation response, which equals the unit response here because $\mathrm{sd}(x) = 1$.

```text
DGP: y_t = 1.5 y_(t-1) -0.54 y_(t-2) + 1.0 x_t + 0.5 x_(t-1) + e_t + 0.6 e_(t-1),
     x_t ~ iid N(0,1) observed shock, e_t ~ iid N(0,1); AR roots 0.9 and 0.6.
T=240, horizons 0..20, 500 replications, seed=20260910, nominal 95% pointwise bands.
Estimand: response of y to a unit x shock. VAR IRFs are Cholesky (x first)
divided by the x impact for bias/SD/RMSE; var_irf_bands cover the one-SD
response, which equals the unit response here because sd(x) = 1.
True IRF: 1.000 2.000 2.460 2.610 2.587 2.470 2.309 2.129 1.947 1.771 1.605 1.451 1.310 1.181 1.065 0.959 0.864 0.778 0.700 0.630 0.567
Monte Carlo time: 108.0 s (216 ms per replication)
```

**Bias**, the mean estimate minus the truth, at selected horizons and averaged in absolute value over all 21:

```text
BIAS  (mean estimate - truth)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20  mean|.|
  VAR(1)               0.007   0.010  -0.514  -0.768  -0.366   0.068   0.340   0.487    0.354
  VAR(4)               0.002  -0.007  -0.023  -0.058  -0.052  -0.188  -0.273  -0.265    0.140
  VAR(12)              0.004  -0.005  -0.022  -0.066  -0.098  -0.147  -0.183  -0.217    0.120
  LP(1) lag-aug        0.004   0.006   0.000  -0.033  -0.081  -0.159  -0.216  -0.235    0.122
  LP(4) lag-aug        0.002  -0.005  -0.020  -0.063  -0.104  -0.167  -0.218  -0.234    0.133
  LP(12) lag-aug       0.004  -0.004  -0.019  -0.069  -0.112  -0.176  -0.233  -0.244    0.141
  LP(4) HAC            0.002  -0.005  -0.021  -0.061  -0.108  -0.166  -0.191  -0.184    0.121
  smooth LP(4) cv      0.734   0.014  -0.249  -0.292  -0.133  -0.105  -0.148  -0.218    0.192
```

**Standard deviation** across replications:

```text
SD    (standard deviation across replications)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1)               0.116   0.233   0.259   0.242   0.227   0.230   0.238   0.245    0.231
  VAR(4)               0.067   0.170   0.268   0.429   0.518   0.524   0.490   0.407    0.438
  VAR(12)              0.071   0.180   0.288   0.468   0.691   0.726   0.637   0.519    0.558
  LP(1) lag-aug        0.132   0.222   0.314   0.467   0.695   0.761   0.829   0.825    0.643
  LP(4) lag-aug        0.067   0.170   0.269   0.437   0.684   0.745   0.821   0.816    0.623
  LP(12) lag-aug       0.069   0.175   0.281   0.464   0.704   0.768   0.835   0.821    0.639
  LP(4) HAC            0.067   0.168   0.265   0.429   0.650   0.709   0.749   0.722    0.584
  smooth LP(4) cv      0.510   0.268   0.302   0.489   0.598   0.665   0.716   0.759    0.598
```

**RMSE**, which is what a one-number ranking would use:

```text
RMSE  (sqrt of bias^2 + SD^2)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1)               0.116   0.233   0.575   0.805   0.431   0.240   0.415   0.545    0.445
  VAR(4)               0.067   0.169   0.269   0.433   0.520   0.556   0.561   0.485    0.468
  VAR(12)              0.071   0.180   0.288   0.472   0.697   0.740   0.662   0.562    0.573
  LP(1) lag-aug        0.132   0.222   0.314   0.468   0.700   0.776   0.856   0.857    0.656
  LP(4) lag-aug        0.067   0.170   0.269   0.441   0.692   0.763   0.848   0.848    0.637
  LP(12) lag-aug       0.069   0.175   0.281   0.469   0.712   0.787   0.866   0.856    0.655
  LP(4) HAC            0.067   0.168   0.266   0.433   0.658   0.727   0.772   0.745    0.596
  smooth LP(4) cv      0.894   0.268   0.391   0.569   0.612   0.673   0.731   0.789    0.643
```

**Coverage** of the nominal 95% pointwise bands (Monte Carlo standard error 0.010):

```text
COVERAGE of nominal 95% pointwise bands (share of replications containing the truth)
  band                         h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1) asymptotic          0.946   0.872   0.400   0.154   0.678   0.980   0.954   0.888    0.736
  VAR(4) asymptotic          0.948   0.942   0.950   0.936   0.926   0.866   0.830   0.768    0.884
  VAR(12) asymptotic         0.942   0.922   0.930   0.936   0.912   0.912   0.920   0.916    0.921
  LP(1) lag-aug HC1          0.958   0.946   0.954   0.936   0.916   0.904   0.908   0.922    0.919
  LP(4) lag-aug HC1          0.948   0.950   0.954   0.942   0.910   0.908   0.888   0.926    0.919
  LP(12) lag-aug HC1         0.940   0.932   0.946   0.938   0.914   0.896   0.882   0.928    0.914
  LP(4) HAC                  0.946   0.938   0.934   0.924   0.884   0.874   0.856   0.870    0.888
  smooth LP(4) se|lambda     0.144   0.884   0.808   0.714   0.810   0.842   0.844   0.856    0.783
  Monte Carlo SE of a 0.95 coverage estimate with 500 replications: 0.010
  smooth_lp lambda_used: median 3.53e+03, IQR [336, 1e+06]; max |irf_raw - lp(se='hac') irf| over all replications: 0.0e+00
```

### Reading the tables

**At short horizons with matched lag lengths, the two estimators are the same estimator.** At $p = 4$ the LP and the VAR have RMSE 0.067 and 0.067 at $h = 0$, 0.170 and 0.169 at $h = 1$, 0.269 and 0.269 at $h = 2$, 0.441 and 0.433 at $h = 4$. Their standard deviations agree to the third decimal through $h = 2$ (0.067/0.067, 0.170/0.170, 0.269/0.268) and their biases differ by at most 0.003. This is LPW's "nearly coincide" lesson, and Part 1 explains why: at impact they are literally the same number, and for the next few horizons the VAR's product of one-step coefficients has not yet compounded enough to differ from a direct projection.

**At long horizons the trade-off opens, and it is variance that opens it.** By $h = 12$ the LP(4) standard deviation is 0.745 against the VAR(4)'s 0.524; by $h = 20$, 0.816 against 0.407 — twice the noise, on a response whose true value is 0.567. The RMSE consequence: 0.763 vs 0.556 at $h = 12$ and 0.848 vs 0.485 at $h = 20$. The VAR's extrapolation, which is a liability under misspecification, is an asset for variance: its horizon-20 estimate is a function of a handful of one-step coefficients estimated from the whole sample, while the LP's is one coefficient from a regression that has lost twenty observations to the horizon and whose error contains twenty periods of intervening shocks.

**A too-short VAR is not a low-variance estimator; it is a wrong one.** VAR(1) has the smallest standard deviation of all eight estimators at every horizon from $h = 2$ on (0.23–0.26) and it is catastrophically biased where the hump is: −0.514 at $h = 2$ and −0.768 at $h = 4$ against true values of 2.46 and 2.61, then *positively* biased by 0.487 at $h = 20$ as its geometric decay crosses the true path. Its 95% band contains the truth in a fraction 0.400 of replications at $h = 2$ and 0.154 at $h = 4$. Look at the RMSE table and the winner list in the verbatim output with that in mind: VAR(1) has the *lowest* RMSE of all eight estimators at $h = 8$ through $18$, because its bias happens to cross zero around $h = 12$ (0.068) while its variance stays small. That is the right-for-the-wrong-reason trap of one-number rankings on one DGP, and it is why the decision rule below never ranks by RMSE alone.

**Lag length trades bias for variance on both sides — and LP is not unbiased in finite samples either.** Going from VAR(4) to VAR(12) cuts the long-horizon bias (−0.265 to −0.217 at $h = 20$) and raises the standard deviation (0.407 to 0.519); RMSE goes up. On the LP side the biases at $p = 1, 4, 12$ are −0.235, −0.234 and −0.244 at $h = 20$ — indistinguishable, and *not* zero. With an i.i.d. observed shock the LP coefficient is unbiased in population at every horizon, but in a sample of 240 the horizon-20 regression's error contains future values of $x$ that are regressors in other rows of the same design matrix, and the persistent lagged-$y$ controls carry a Nickell-type bias of their own; the result is the finite-sample LP bias that Herbst and Johannsen (2024) document, growing with $h/T$ and with persistence. Here it is the same size as the VAR(12)'s. The *bias* case for LP over VAR is about robustness to misspecification, not about exact unbiasedness, and at $h/T = 20/240$ it has largely been spent.

**Lag augmentation costs variance and buys valid bands.** The un-augmented `lp(se="hac")` at $p = 4$ has a smaller standard deviation than the lag-augmented default at every long horizon (0.722 vs 0.816 at $h = 20$): the $h$ extra regressors are not free. What they buy is inference. The lag-augmented HC1 band covers 0.926 at $h = 20$ and 0.919 on average over the 21 horizons; the HAC band 0.870 and 0.888. Both slip below 0.95 at long horizons — the LP bias just described is inside the band's width but not centred in it — and the lag-augmented band slips less. The VAR(4) band, meanwhile, falls to 0.768 at $h = 20$ (0.884 on average) because its bias is unaccounted for, and the VAR(12) band holds 0.912–0.942 at every printed horizon (0.921 on average) because its bias is smaller: the delta-method band is only as honest as the lag length.

**Smoothing is a dial, and cross-validation can turn it the wrong way.** `smooth_lp(lam="cv")` chose heavy smoothing on this hump-shaped response — a median $\lambda$ of 3.53e+03 with an interquartile range reaching 1e+06, the top of the grid — because the cross-validation score is dominated by the long-horizon blocks, where the outcome variance is enormous and a flat line fits as well as anything. The price is paid at impact: bias +0.734 and a standard deviation of 0.510 at $h = 0$, against 0.002 and 0.067 for the raw LP it is built on, and coverage of 0.144. Against its own raw path, `lp(se="hac")` at $p = 4$, the gains are confined to the middle horizons (RMSE 0.612 vs 0.658 at $h = 8$, 0.673 vs 0.727 at $h = 12$, 0.731 vs 0.772 at $h = 16$) and turn into losses at both ends (0.569 vs 0.433 at $h = 4$, 0.789 vs 0.745 at $h = 20$). The $\lambda = 0$ anchor held exactly in all 500 replications — the script checks `irf_raw` against `lp(se="hac")` and the largest gap was 0.0e+00 — so the raw LP is always in the returned dictionary next to the smoothed one, and this chapter's advice is to *plot both* and to treat a CV-chosen $\lambda$ at the top of its grid as a warning rather than an answer.

## The decision rule

The library's recommendation, written from the tables rather than from a preference, and mirrored in the [decision guide's single-equation IRF entry](../which-model-when.md#2c-i-want-a-single-equation-irf-no-full-var):

1. **Stop choosing an estimand.** Whichever you report, say which controls it conditions on; that — not "LP or VAR" — is what fixes the object being estimated. If the impulse is a measured innovation, `lp`'s control set is right. If it is a persistent observable, either put it first in a recursive `var_irf`, or build the matched regression with `tsecon.ols` as Part 1 does, or pre-whiten it; `lp` alone will not reach the VAR's estimand, at any $T$.
2. **For the point path and bands at horizons short relative to the sample, use the lag-augmented `lp`.** Through $h = 2$ at $T = 240$ it matched the best VAR's RMSE to within 0.001, and to within 0.01 at $h = 4$; its band covered 0.942–0.954 at the printed horizons through $h = 4$; and it did so without asking you to trust a lag structure it did not estimate.
3. **For long horizons, overlay a generously lagged VAR, and report both lines.** From $h = 8$ on, the LP's standard deviation is 1.3 to 2 times the VAR(4)'s and its RMSE 1.3 to 1.8 times higher. A VAR with the lag length an information criterion would call excessive (12 rather than 4 in this design) keeps much of the variance gain (SD 0.519 against the LP's 0.816 at $h = 20$), trims the long-horizon bias (−0.217 against −0.265), and holds its delta-method band at 0.912–0.942 where the shorter VAR's falls to 0.768 — at the price of a *higher* RMSE than VAR(4) at every horizon (0.562 against 0.485 at $h = 20$). What the extra lags buy is a band you can believe, not a lower error. Never report a short-lag VAR's long-horizon path on the strength of its RMSE: check its band coverage first, or read its divergence from the LP as the lag-length diagnostic it is.
4. **Read divergence as information, not as a verdict.** LP and VAR lines that separate at long horizons and *stay* separated as you add lags are telling you the VAR is extrapolating; lines that converge as you add lags are telling you the shorter VAR was too short. Either way the pattern is a specification test you get for free by fitting both.
5. **Smooth after you have looked, not instead of looking.** `smooth_lp` is the continuous version of this chapter's dial; use it with `irf_raw` beside it, and distrust a cross-validated $\lambda$ that lands at the top of its grid.

The two lag-length knobs (`lags` in `var_irf`, `n_lag_controls` in `lp`) are the same knob seen from two sides, and the numbers above are the argument for turning it up rather than down when an impulse response is the goal: the cost is variance you can see in a band, the benefit is bias you cannot.

## Provenance and the verbatim output

Build and versions, from the script's own banner: tsecon 0.9.0 on a **release** build of the extension (the banner matches the installed `_core` extension's exact file size against `<target>/release/lib_core.so`), Python 3.11.15, numpy 2.4.6, scipy 1.17.1, Linux x86_64 on an Intel Xeon @ 2.80 GHz with 4 cores. The whole script — both parts, 500 replications — ran in 108.5 s of wall-clock time, 216 ms per replication. The command:

```sh
.venv/bin/python docs/examples/lp_vs_var_head_to_head.py     # ~2 min on a release build
```

The seed is fixed inside the script (`20260910`), so re-running it reproduces every number below bit-for-bit on the same build; a different platform's BLAS may move the last digit of the Monte Carlo summaries and will not move a parity of the kind Part 1 reports.

<details markdown="1">
<summary>Full output of <code>docs/examples/lp_vs_var_head_to_head.py</code>, verbatim</summary>

```text

------------------------------------------------------------------------------
PROVENANCE
------------------------------------------------------------------------------
  date          : 2026-09-10 11:00:34 UTC
  python        : 3.11.15 (CPython)
  platform      : Linux-6.18.44-fc-v24-x86_64-with-glibc2.39  cpu_count=4
  cpu model     : Intel(R) Xeon(R) Processor @ 2.80GHz
  tsecon        : 0.9.0   build: release (== <target>/release/lib_core.so, 18.3 MB)
  numpy / scipy : 2.4.6 / 1.17.1

------------------------------------------------------------------------------
PART 1 -- Plagborg-Moller & Wolf (2021): where is the finite-sample identity?
------------------------------------------------------------------------------
DGP: bivariate Gaussian VAR(2), shock variable x ordered first, T=400, horizons 0..12, seed=20260910
VAR(2) IRF = Cholesky response of y to the x shock, divided by the x impact
            (one-SD shock -> unit shock, the LP normalisation).
lp(y, x, n_lag_controls=2) controls: constant, 2 own-lags of y, and on the
            lag-augmented path h lags of x; on the HAC path no x lags.
matched LP: constant, x_t, 2 lags of x AND 2 lags of y (tsecon.ols).

[x has its own AR dynamics]  var_fit is_stable=True  min_root=1.489
   h     VAR(2)  lp lag-aug     lp hac    matched  |VAR-lagaug|  |VAR-hac|  |VAR-matched|
   0   0.514865    0.558012   0.558012   0.514865      4.31e-02   4.31e-02       2.11e-15
   1   0.667378    0.665306   0.715191   0.669839      2.07e-03   4.78e-02       2.46e-03
   2   0.533931    0.543530   0.572043   0.543530      9.60e-03   3.81e-02       9.60e-03
   3   0.349863    0.307702   0.330219   0.304655      4.22e-02   1.96e-02       4.52e-02
   4   0.213444    0.232117   0.220656   0.226679      1.87e-02   7.21e-03       1.32e-02
   5   0.132457    0.152647   0.130262   0.153214      2.02e-02   2.20e-03       2.08e-02
   6   0.086593    0.052244   0.071994   0.066872      3.43e-02   1.46e-02       1.97e-02
   7   0.058813    0.057614   0.061325   0.057800      1.20e-03   2.51e-03       1.01e-03
   8   0.040322    0.056688   0.050385   0.027818      1.64e-02   1.01e-02       1.25e-02
   9   0.027438    0.080453   0.061500   0.056462      5.30e-02   3.41e-02       2.90e-02
  10   0.018480    0.065972   0.064210   0.054210      4.75e-02   4.57e-02       3.57e-02
  11   0.012375    0.057669   0.067258   0.053895      4.53e-02   5.49e-02       4.15e-02
  12   0.008279    0.108406   0.116483   0.112833      1.00e-01   1.08e-01       1.05e-01
  horizons where VAR(2) == lp lag-augmented to 1e-12: NONE
  horizons where VAR(2) == lp hac to 1e-12: NONE
  horizons where VAR(2) == matched-controls LP to 1e-12: [0]

[x is white noise (innovation-like)]  var_fit is_stable=True  min_root=2.344
   h     VAR(2)  lp lag-aug     lp hac    matched  |VAR-lagaug|  |VAR-hac|  |VAR-matched|
   0   0.507213    0.492431   0.492431   0.507213      1.48e-02   1.48e-02       3.33e-16
   1   0.621153    0.621171   0.611203   0.623697      1.74e-05   9.95e-03       2.54e-03
   2   0.330462    0.332568   0.331594   0.332568      2.11e-03   1.13e-03       2.11e-03
   3   0.042365    0.039441   0.053867   0.047709      2.92e-03   1.15e-02       5.34e-03
   4  -0.043486   -0.077855  -0.069213  -0.078723      3.44e-02   2.57e-02       3.52e-02
   5  -0.029515   -0.097902  -0.098635  -0.103148      6.84e-02   6.91e-02       7.36e-02
   6  -0.007019   -0.099395  -0.106797  -0.100339      9.24e-02   9.98e-02       9.33e-02
   7   0.001739    0.057323   0.029002   0.040614      5.56e-02   2.73e-02       3.89e-02
   8   0.002160    0.060744   0.036557   0.045155      5.86e-02   3.44e-02       4.30e-02
   9   0.000796    0.055237   0.051545   0.050643      5.44e-02   5.07e-02       4.98e-02
  10   0.000017    0.000736  -0.014363  -0.017712      7.19e-04   1.44e-02       1.77e-02
  11  -0.000136   -0.086598  -0.105074  -0.101409      8.65e-02   1.05e-01       1.01e-01
  12  -0.000073    0.080388   0.063904   0.069248      8.05e-02   6.40e-02       6.93e-02
  horizons where VAR(2) == lp lag-augmented to 1e-12: NONE
  horizons where VAR(2) == lp hac to 1e-12: NONE
  horizons where VAR(2) == matched-controls LP to 1e-12: [0]

How each gap scales with T (max over the stated horizons, one draw per T):
       T | persistent x: lagaug h=0  matched h=0  matched h>=1 | white-noise x: lagaug h=0  matched h>=1
     200 |                 8.47e-02     1.11e-16      1.88e-01 |                  3.07e-02      1.49e-01
     800 |                 3.68e-02     7.77e-16      5.29e-02 |                  1.21e-02      2.83e-02
    3200 |                 5.06e-02     2.11e-15      2.83e-02 |                  4.77e-03      4.46e-02
   12800 |                 5.01e-02     4.88e-15      1.34e-02 |                  4.20e-03      1.73e-02

------------------------------------------------------------------------------
PART 2 -- Li, Plagborg-Moller & Wolf (2024): the bias-variance trade-off, measured
------------------------------------------------------------------------------
DGP: y_t = 1.5 y_(t-1) -0.54 y_(t-2) + 1.0 x_t + 0.5 x_(t-1) + e_t + 0.6 e_(t-1),
     x_t ~ iid N(0,1) observed shock, e_t ~ iid N(0,1); AR roots 0.9 and 0.6.
T=240, horizons 0..20, 500 replications, seed=20260910, nominal 95% pointwise bands.
Estimand: response of y to a unit x shock. VAR IRFs are Cholesky (x first)
divided by the x impact for bias/SD/RMSE; var_irf_bands cover the one-SD
response, which equals the unit response here because sd(x) = 1.
True IRF: 1.000 2.000 2.460 2.610 2.587 2.470 2.309 2.129 1.947 1.771 1.605 1.451 1.310 1.181 1.065 0.959 0.864 0.778 0.700 0.630 0.567
Monte Carlo time: 108.0 s (216 ms per replication)

BIAS  (mean estimate - truth)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20  mean|.|
  VAR(1)               0.007   0.010  -0.514  -0.768  -0.366   0.068   0.340   0.487    0.354
  VAR(4)               0.002  -0.007  -0.023  -0.058  -0.052  -0.188  -0.273  -0.265    0.140
  VAR(12)              0.004  -0.005  -0.022  -0.066  -0.098  -0.147  -0.183  -0.217    0.120
  LP(1) lag-aug        0.004   0.006   0.000  -0.033  -0.081  -0.159  -0.216  -0.235    0.122
  LP(4) lag-aug        0.002  -0.005  -0.020  -0.063  -0.104  -0.167  -0.218  -0.234    0.133
  LP(12) lag-aug       0.004  -0.004  -0.019  -0.069  -0.112  -0.176  -0.233  -0.244    0.141
  LP(4) HAC            0.002  -0.005  -0.021  -0.061  -0.108  -0.166  -0.191  -0.184    0.121
  smooth LP(4) cv      0.734   0.014  -0.249  -0.292  -0.133  -0.105  -0.148  -0.218    0.192

SD    (standard deviation across replications)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1)               0.116   0.233   0.259   0.242   0.227   0.230   0.238   0.245    0.231
  VAR(4)               0.067   0.170   0.268   0.429   0.518   0.524   0.490   0.407    0.438
  VAR(12)              0.071   0.180   0.288   0.468   0.691   0.726   0.637   0.519    0.558
  LP(1) lag-aug        0.132   0.222   0.314   0.467   0.695   0.761   0.829   0.825    0.643
  LP(4) lag-aug        0.067   0.170   0.269   0.437   0.684   0.745   0.821   0.816    0.623
  LP(12) lag-aug       0.069   0.175   0.281   0.464   0.704   0.768   0.835   0.821    0.639
  LP(4) HAC            0.067   0.168   0.265   0.429   0.650   0.709   0.749   0.722    0.584
  smooth LP(4) cv      0.510   0.268   0.302   0.489   0.598   0.665   0.716   0.759    0.598

RMSE  (sqrt of bias^2 + SD^2)
  estimator              h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1)               0.116   0.233   0.575   0.805   0.431   0.240   0.415   0.545    0.445
  VAR(4)               0.067   0.169   0.269   0.433   0.520   0.556   0.561   0.485    0.468
  VAR(12)              0.071   0.180   0.288   0.472   0.697   0.740   0.662   0.562    0.573
  LP(1) lag-aug        0.132   0.222   0.314   0.468   0.700   0.776   0.856   0.857    0.656
  LP(4) lag-aug        0.067   0.170   0.269   0.441   0.692   0.763   0.848   0.848    0.637
  LP(12) lag-aug       0.069   0.175   0.281   0.469   0.712   0.787   0.866   0.856    0.655
  LP(4) HAC            0.067   0.168   0.266   0.433   0.658   0.727   0.772   0.745    0.596
  smooth LP(4) cv      0.894   0.268   0.391   0.569   0.612   0.673   0.731   0.789    0.643

RMSE winner by horizon (lowest RMSE among the eight estimators):
  h= 0: LP(4) lag-aug      RMSE=0.067   (LP(4) lag-aug 0.067, VAR(4) 0.067, VAR(12) 0.071)
  h= 1: LP(4) HAC          RMSE=0.168   (LP(4) lag-aug 0.170, VAR(4) 0.169, VAR(12) 0.180)
  h= 2: LP(4) HAC          RMSE=0.266   (LP(4) lag-aug 0.269, VAR(4) 0.269, VAR(12) 0.288)
  h= 3: LP(4) HAC          RMSE=0.354   (LP(4) lag-aug 0.358, VAR(4) 0.355, VAR(12) 0.386)
  h= 4: VAR(4)             RMSE=0.433   (LP(4) lag-aug 0.441, VAR(4) 0.433, VAR(12) 0.472)
  h= 5: VAR(4)             RMSE=0.485   (LP(4) lag-aug 0.517, VAR(4) 0.485, VAR(12) 0.545)
  h= 6: VAR(4)             RMSE=0.507   (LP(4) lag-aug 0.586, VAR(4) 0.507, VAR(12) 0.611)
  h= 7: VAR(4)             RMSE=0.514   (LP(4) lag-aug 0.651, VAR(4) 0.514, VAR(12) 0.663)
  h= 8: VAR(1)             RMSE=0.431   (LP(4) lag-aug 0.692, VAR(4) 0.520, VAR(12) 0.697)
  h= 9: VAR(1)             RMSE=0.333   (LP(4) lag-aug 0.719, VAR(4) 0.527, VAR(12) 0.715)
  h=10: VAR(1)             RMSE=0.261   (LP(4) lag-aug 0.738, VAR(4) 0.537, VAR(12) 0.725)
  h=11: VAR(1)             RMSE=0.230   (LP(4) lag-aug 0.749, VAR(4) 0.547, VAR(12) 0.730)
  h=12: VAR(1)             RMSE=0.240   (LP(4) lag-aug 0.763, VAR(4) 0.556, VAR(12) 0.740)
  h=13: VAR(1)             RMSE=0.277   (LP(4) lag-aug 0.791, VAR(4) 0.564, VAR(12) 0.743)
  h=14: VAR(1)             RMSE=0.323   (LP(4) lag-aug 0.816, VAR(4) 0.568, VAR(12) 0.725)
  h=15: VAR(1)             RMSE=0.371   (LP(4) lag-aug 0.828, VAR(4) 0.567, VAR(12) 0.696)
  h=16: VAR(1)             RMSE=0.415   (LP(4) lag-aug 0.848, VAR(4) 0.561, VAR(12) 0.662)
  h=17: VAR(1)             RMSE=0.455   (LP(4) lag-aug 0.849, VAR(4) 0.549, VAR(12) 0.630)
  h=18: VAR(1)             RMSE=0.490   (LP(4) lag-aug 0.840, VAR(4) 0.531, VAR(12) 0.603)
  h=19: VAR(4)             RMSE=0.510   (LP(4) lag-aug 0.839, VAR(4) 0.510, VAR(12) 0.581)
  h=20: VAR(4)             RMSE=0.485   (LP(4) lag-aug 0.848, VAR(4) 0.485, VAR(12) 0.562)

COVERAGE of nominal 95% pointwise bands (share of replications containing the truth)
  band                         h=0     h=1     h=2     h=4     h=8    h=12    h=16    h=20     mean
  VAR(1) asymptotic          0.946   0.872   0.400   0.154   0.678   0.980   0.954   0.888    0.736
  VAR(4) asymptotic          0.948   0.942   0.950   0.936   0.926   0.866   0.830   0.768    0.884
  VAR(12) asymptotic         0.942   0.922   0.930   0.936   0.912   0.912   0.920   0.916    0.921
  LP(1) lag-aug HC1          0.958   0.946   0.954   0.936   0.916   0.904   0.908   0.922    0.919
  LP(4) lag-aug HC1          0.948   0.950   0.954   0.942   0.910   0.908   0.888   0.926    0.919
  LP(12) lag-aug HC1         0.940   0.932   0.946   0.938   0.914   0.896   0.882   0.928    0.914
  LP(4) HAC                  0.946   0.938   0.934   0.924   0.884   0.874   0.856   0.870    0.888
  smooth LP(4) se|lambda     0.144   0.884   0.808   0.714   0.810   0.842   0.844   0.856    0.783
  Monte Carlo SE of a 0.95 coverage estimate with 500 replications: 0.010
  smooth_lp lambda_used: median 3.53e+03, IQR [336, 1e+06]; max |irf_raw - lp(se='hac') irf| over all replications: 0.0e+00

Total wall-clock: 108.5 s
```

</details>

## Further reading

- **Plagborg-Møller & Wolf (2021), "Local Projections and VARs Estimate the Same Impulse Responses", *Econometrica* 89(2)** — the population equivalence; the source of Part 1's statement and of item 1 in the decision rule.
- **Li, Plagborg-Møller & Wolf (2024), "Local projections vs. VARs: Lessons from thousands of DGPs", *Journal of Econometrics* 244** — the bias-variance trade-off at scale; Part 2 is a one-DGP reproduction of its design logic.
- **Montiel Olea & Plagborg-Møller (2021), "Local Projection Inference Is Simpler and More Robust Than You Think", *Econometrica* 89(4)** — why `lp` augments with the impulse's own lags by default, and what that costs in the SD table.
- **Herbst & Johannsen (2024), "Bias in local projections", *Journal of Econometrics* 240** — the finite-sample LP bias that shows up at $h = 20$ in the bias table.
- **Barnichon & Brownlees (2019), "Impulse Response Estimation by Smooth Local Projections", *Review of Economics and Statistics* 101(3)** — the estimator behind `smooth_lp`, and the dial this chapter warns you to watch.
- **Montiel Olea, Plagborg-Møller, Qian & Wolf (2024), "Double Robustness of Local Projections and Some Unpleasant VARithmetic"** — why VAR bias does not vanish as bands widen, the deeper reason behind item 3.
