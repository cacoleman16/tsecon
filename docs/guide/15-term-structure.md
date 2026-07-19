# Chapter 15 — The Term Structure of Interest Rates

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** the factor-model idea — a few latent variables driving many observed series — from chapters 7 and 12; the AR and state-space models of chapter 4; and the forecast-evaluation discipline of chapter 5, which is how any curve forecast earns its keep. No asset-pricing background is assumed.

**You will learn:**

- Why the yield curve — a dozen or more maturities that move together almost in lockstep — is really a three-number object, and what those three numbers mean
- How the Nelson-Siegel model turns a whole curve into a *level*, a *slope*, and a *curvature* by one cross-sectional regression on fixed basis functions
- What the decay parameter $\lambda$ does, when to pin it and when to estimate it, and why a single global value beats re-optimizing it curve by curve
- When one hump is not enough and the Svensson extension's second curvature factor earns its place
- How the dynamic Nelson-Siegel model turns a *panel* of curves into three forecastable time series and produces a one-step-ahead curve — and the two caveats (no-arbitrage and the two-step standard errors) that discipline the result

## The idea

Pull up the U.S. Treasury yield curve on any given day and you have eight, twelve, maybe fifteen numbers — the yield on a 3-month bill, a 1-year note, a 10-year bond, and everything between. Watch them over time and a striking fact appears: they almost never move independently. When the 2-year yield rises, the 5-year and the 10-year usually rise with it. When the curve steepens, it steepens all along its length. A principal-components analysis of any yield panel — exactly the machinery of chapter 7 — finds that **three factors explain well over 99% of the variance** of the entire curve. A fifteen-dimensional object lives, to a superb approximation, on a three-dimensional surface.

Those three factors have names, because they have shapes. Litterman and Scheinkman (1991) gave the field its vocabulary by looking at what each principal component *does* to the curve:

1. **Level** — a factor that shifts every yield up or down by roughly the same amount. Parallel shifts of the whole curve. This is far and away the biggest mover: the curve mostly goes up and down as a block.
2. **Slope** — a factor that tilts the curve, pushing short yields one way and long yields the other. The steepness of the curve, from steeply upward-sloping in early recoveries to inverted before recessions.
3. **Curvature** — a factor that bows the middle of the curve relative to its ends, lifting or dropping medium maturities while leaving the short and long ends roughly put. The "hump."

The question this chapter answers is the natural one: if the curve is really three numbers, can we *build a model that says so* — one that reads off a level, slope, and curvature from any curve, reconstructs the whole curve from just those three, and (the prize) forecasts next month's curve by forecasting three time series instead of fifteen? That is precisely the program Nelson and Siegel (1987) began and Diebold and Li (2006) completed. Their move was to stop estimating the three shapes from the data — as principal components would — and instead *impose* them as fixed, economically sensible basis functions, so the three factors become the coefficients of one small regression. It is the factor-model idea of chapter 12 with the loadings written down in advance rather than extracted, and that one change is what makes the curve forecastable.

Everything in this chapter is a variation on that theme: Nelson-Siegel fixes three basis functions and reads off three factors; Svensson adds a fourth for curves with two humps; dynamic Nelson-Siegel does the reading-off every period and then models the three (or four) factor *series* as they evolve. We use a real monthly Treasury panel throughout — eight maturities from 3 to 120 months, shipped as the golden fixture [`fixtures/termstructure.json`](../../fixtures/termstructure.json) that pins the library's numbers — so every number below is one you can reproduce.

## Nelson-Siegel: three numbers for the whole curve

A practitioner cares because this is the single most-used yield-curve model in central banks and on trading desks, and the reason is its economy: it compresses a curve to three interpretable numbers *without* throwing away the shape, so you can store decades of curves in a tiny table, interpolate to maturities you did not observe, and — the next section's payoff — forecast.

The model writes the yield at maturity $\tau$ as a weighted sum of three fixed shapes:

$$
y(\tau) \;=\; \beta_1 \;+\; \beta_2\,\underbrace{\frac{1 - e^{-\lambda \tau}}{\lambda \tau}}_{\text{slope loading}} \;+\; \beta_3\,\underbrace{\left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)}_{\text{curvature loading}} .
$$

Read the three loadings — the fixed functions of maturity that multiply each factor — and the Litterman-Scheinkman names fall out mechanically:

- The **level** loading is a constant $1$: $\beta_1$ shifts *every* maturity equally. As $\tau \to \infty$ the other two loadings vanish, so $\beta_1$ is the **long-rate limit** — the level of the curve.
- The **slope** loading $(1 - e^{-\lambda\tau})/(\lambda\tau)$ starts at $1$ for the shortest maturity and decays to $0$ for the longest, so $\beta_2$ tilts short yields relative to long ones. As $\tau \to 0$ both surviving loadings tend to $1$ and $0$ respectively, so the **instantaneous short rate is $\beta_1 + \beta_2$**: the slope factor is exactly the short-minus-long spread. (Note the sign: with Diebold and Li's parameterization a *positive* $\beta_2$ means short rates exceed long rates — an inverted curve — the opposite of the everyday "long minus short" slope convention.)
- The **curvature** loading starts at $0$, rises to a hump at an intermediate maturity, and returns toward $0$, so $\beta_3$ lifts or drops the belly of the curve without much touching either end.

The decay parameter $\lambda$ controls *where* the hump sits and how fast the slope loading dies off. Diebold and Li fix it at $\lambda = 0.0609$ for monthly data — the value that puts the curvature loading's peak at a maturity of 30 months — and with $\lambda$ fixed the model is *linear in the factors*, so estimation is a single cross-sectional ordinary least squares regression of the observed yields on the three loading columns. That is the whole trick: no optimizer, no starting values, one OLS per curve.

```python
import json
import numpy as np
import tsecon

d = json.load(open("fixtures/termstructure.json"))
maturities = np.array(d["maturities"])        # months: 3, 6, 12, 24, 36, 60, 84, 120
yields = np.array(d["yields_date100"])         # one month's curve, in percent

fit = tsecon.nelson_siegel(maturities, yields, decay=0.0609)
print(round(fit["level"], 3), round(fit["slope"], 3), round(fit["curvature"], 3))
# 4.639 -0.541 -1.046
print(round(fit["rsquared"], 4))               # 0.9537
print(np.round(fit["residuals"], 3))
# [ 0.044 -0.045 -0.033  0.048 -0.008  0.008 -0.025  0.011]
```

Three numbers now stand in for eight, and each one reads cleanly. The **level** of 4.64 is the long-rate anchor. The **slope** of $-0.54$ is negative, so short rates sit *below* long rates — an ordinary upward-sloping curve — and the implied instantaneous short rate is $\beta_1 + \beta_2 = 4.639 - 0.541 = 4.10$, essentially the observed 3-month yield of 4.09. The **curvature** of $-1.05$ says the belly of this curve is pulled *down* relative to its ends. The $R^2$ of 0.954 and residuals of a few basis points confirm that three factors reproduce all eight yields to within measurement noise — the whole reason the model is trusted.

To reconstruct or interpolate the curve, evaluate the same three loadings at any maturities you like and multiply by the fitted factors. Because the loadings are smooth closed-form functions, Nelson-Siegel gives you a yield at *every* maturity, not just the ones you observed — which is why it is the standard tool for building a smooth par curve from a handful of traded points.

> **⚠ Common mistake — reading the factors as free parameters.** The level, slope, and curvature are only interpretable *because* $\lambda$ is fixed and the loadings are the specific shapes above. If you compare Nelson-Siegel factors across datasets fitted with different $\lambda$, or against a principal-components "level/slope/curvature," you are comparing coefficients on different basis functions — the numbers are not on the same scale. Always report the $\lambda$ you used; factors are meaningful only relative to it.

## The decay parameter: fix it or fit it

The one genuine modeling choice in Nelson-Siegel is $\lambda$, and there are two schools. The first — Diebold and Li's — **fixes $\lambda$ globally** at a sensible value (0.0609 monthly) and never touches it again. The second **estimates $\lambda$** by nonlinear least squares, letting the data choose the decay that best fits each curve. tsecon supports both through one flag:

```python
fixed = tsecon.nelson_siegel(maturities, yields, decay=0.0609)
opt   = tsecon.nelson_siegel(maturities, yields, optimal_lambda=True)

print(round(fixed["rsquared"], 4), round(opt["rsquared"], 4))   # 0.9537 0.9581
print(round(opt["lambda"], 4))                                  # 0.0853
```

Estimating $\lambda$ buys a better fit — here the $R^2$ rises from 0.954 to 0.958, and the optimized $\lambda = 0.085$ moves the curvature hump inward to about 21 months. On any single curve, more freedom fits better; that is never in doubt.

The judgment is whether you *want* that freedom, and for a panel the answer is usually no. The reason is the whole point of the model: the factors are comparable across dates only if they load on the *same* basis functions. Re-optimize $\lambda$ each period and $\beta_2$ in January loads on a different slope shape than $\beta_2$ in February — you have made the "level, slope, curvature" time series that the next section forecasts incommensurable from one date to the next, and worse, the $\lambda$ estimate is notoriously ill-conditioned (the objective is nearly flat in it), so it jumps around chasing noise. Fixing $\lambda$ costs a sliver of cross-sectional fit and buys a clean, interpretable, *forecastable* set of factors. This is why Diebold and Li fix it, why the dynamic model of the next section does too, and why "fit $\lambda$" is best reserved for a one-off, best-possible fit to a *single* curve.

> **⚠ Common mistake — re-optimizing $\lambda$ every period, then modeling the factors.** It is the natural pipeline — "fit each curve as well as I can, then model the factor series" — and it quietly poisons the second step. Time-varying loadings mean the factor at date $t$ and the factor at date $t+1$ are coefficients on *different regressors*; their AR(1) dynamics mix genuine factor movement with basis drift, and the one-step curve forecast inherits the confusion. If you are going to model the factors as time series, fix $\lambda$ once for the whole panel.

## Svensson: room for a second hump

Nelson-Siegel's single curvature term can produce exactly one hump. Real curves — especially long ones spanning out to 30 years, and curves in unusual policy regimes — sometimes show *two* bends: one in the short belly and another further out. Svensson (1994) extended the model with a **second curvature factor**, carrying its own decay $\lambda_2$, so the curve can flex in two places at once:

$$
y(\tau) \;=\; \beta_1 \;+\; \beta_2\,\frac{1 - e^{-\lambda_1 \tau}}{\lambda_1 \tau} \;+\; \beta_3\!\left(\frac{1 - e^{-\lambda_1 \tau}}{\lambda_1 \tau} - e^{-\lambda_1 \tau}\right) \;+\; \beta_4\!\left(\frac{1 - e^{-\lambda_2 \tau}}{\lambda_2 \tau} - e^{-\lambda_2 \tau}\right).
$$

The first three terms are Nelson-Siegel verbatim; the fourth is a second curvature loading with its own hump location set by $\lambda_2$. Svensson **nests** Nelson-Siegel — send $\beta_4 \to 0$ (or the two decays together) and you are back to the three-factor model — so it can only fit at least as well. The European Central Bank and many other official curve-fitters use the Svensson (or the closely related Svensson-Söderlind) form as their production model precisely because the extra hump handles the long end that Nelson-Siegel can miss.

With both decays supplied, the model is again linear in the four factors — one OLS on four loading columns:

```python
ns = tsecon.nelson_siegel(maturities, yields, decay=0.0609)
sv = tsecon.svensson(maturities, yields, lambda1=0.0609, lambda2=0.03)

print(np.round(sv["factors"], 3))                          # [ 4.614 -0.514 -1.096  0.105]
print(round(ns["rsquared"], 4), round(sv["rsquared"], 4))  # 0.9537 0.9538
```

Here is the honest verdict this example is built to deliver: on *this* curve, the fourth factor is a tiny 0.105 and the $R^2$ improves from 0.9537 to only 0.9538 — a rounding error. This particular Treasury curve has one gentle hump, so Nelson-Siegel already captures its shape and Svensson's extra flexibility fits noise. That is the general rule: **Svensson earns its keep only when the curve genuinely has a second bend.** For a smooth, single-humped curve it adds a weakly identified parameter — the two decays $\lambda_1$ and $\lambda_2$ are prone to fighting each other for the same feature, and when they collide the loading columns become collinear and the factor estimates blow up. Reach for Svensson when you fit long maturities or see systematic Nelson-Siegel residuals at the far end; otherwise the parsimony of three factors is a feature.

> **⚠ Common mistake — using Svensson by default for the extra $R^2$.** More parameters always fit the in-sample curve better, so a higher $R^2$ is not evidence you needed the fourth factor. The costs are real: $\lambda_1$ and $\lambda_2$ are jointly hard to identify, and a near-zero $\beta_4$ with an arbitrary $\lambda_2$ is a red flag that the second hump is not in the data. Compare the *residual pattern*, not the $R^2$: if Nelson-Siegel's residuals are unstructured (as ours are, a few basis points with no shape), you do not need Svensson.

## Dynamic Nelson-Siegel: a curve that moves, and a forecast

Everything so far fits *one* curve. The payoff comes from a *panel* of curves — one row per date, one column per maturity — and Diebold and Li's (2006) insight: if a fixed-$\lambda$ Nelson-Siegel reads off a level, slope, and curvature every month, then the yield curve's evolution *is* the joint evolution of three time series. Forecast those three, map them back through the loadings, and you have forecast the entire curve. A fifteen-dimensional forecasting problem collapses to three univariate ones.

The **dynamic Nelson-Siegel** model is exactly this, in two steps:

1. **Cross-section, every period.** For each date $t$, run the fixed-$\lambda$ Nelson-Siegel regression to extract $\big(\beta_{1t},\,\beta_{2t},\,\beta_{3t}\big)$ — the level, slope, and curvature *at that date*. Stacking over $t$ gives three factor time series.
2. **Time series, factor by factor.** Model each factor's dynamics. Diebold and Li fit an independent **AR(1)** to each — $\beta_{i,t+1} = c_i + \phi_i \beta_{it} + \varepsilon_{i,t+1}$ — and forecast one step ahead. The forecast curve is then the three forecast factors run back through the same loadings.

tsecon's `dynamic_ns` does both steps and returns the factor paths, the per-date fit, the named level/slope/curvature series, and a one-step-ahead forecast bundle:

```python
panel = np.array(d["yields_panel"])            # 240 months x 8 maturities
dns = tsecon.dynamic_ns(panel, maturities, decay=0.0609)

factors = np.array(dns["factors"])             # (240, 3): level, slope, curvature paths
print(factors.shape)                           # (240, 3)
print(np.round(factors[-1], 3))                # last curve's factors: [ 5.445  0.51  -0.014]

fc = dns["forecast"]
print(np.round(fc["ar1_phi"], 3))              # [0.933 0.92  0.776]
print(np.round(np.array(fc["factors"]), 3))    # next-month factors: [5.434 0.474 0.01 ]
print(np.round(np.array(fc["yields"]), 3))     # next-month curve, 8 maturities:
# [5.869 5.833 5.773 5.687 5.63  5.563 5.529 5.501]
```

Read this from the inside out. The **factor paths** (240 × 3) are the level, slope, and curvature of every curve in the sample — the object you now model and plot instead of eight separate yield series. The last curve's factors are `[5.445, 0.51, -0.014]`: a positive slope of 0.51 says short rates are *above* long rates, an inverted curve, which the raw last row confirms (the 3-month yield of 5.90 tops the 120-month yield of 5.53).

That these factors really are level, slope, and curvature is not an article of faith — it is checkable against the data. In this panel the fitted level correlates 0.95 with the long (120-month) yield, the slope factor correlates 0.997 with the empirical short-minus-long spread, and the curvature factor correlates 0.98 with the textbook curvature proxy $2\,y(24) - y(3) - y(120)$. The imposed basis functions recovered exactly the movements they were designed to name.

The **AR(1) persistences** `ar1_phi = [0.933, 0.92, 0.776]` are the engine of the forecast and its most important diagnostic. All three are high — the level especially, at 0.93 — which says the factors are slow-moving and near-random-walk. The **one-step forecast** is then just each factor's AR(1) prediction: the level forecast of 5.434, for instance, is $c_1 + \phi_1 \beta_{1,T} = 0.356 + 0.933 \times 5.445$. Push the three forecast factors through the loadings and you get `forecast["yields"]` — next month's entire curve, all eight maturities, from three AR(1) forecasts. Here it barely moves from the last observed curve, which is exactly what persistence near one implies.

That near-unit persistence is also the model's sober lesson. Diebold and Li's headline was that DNS forecasts *beat the random walk* at longer horizons for some maturities — a genuine and much-replicated result — but at the one-month horizon a curve whose factors are this persistent is very hard to out-predict, and the AR(1) forecast lands close to "no change." As always in this guide, a curve forecast is only as good as the honest out-of-sample test behind it: run the DNS one-step (or $h$-step) forecast through the backtesting and Diebold-Mariano machinery of chapter 5 against a random-walk benchmark before claiming it adds value. The factor structure is what makes the forecast *possible* and *interpretable*; whether it *wins* is an empirical question you must settle with a proper backtest.

> **⚠ Common mistake — trusting the two-step standard errors as if the factors were observed.** The two-step estimator treats the first-stage fitted factors as if they were data in the second stage, so the AR(1) standard errors — and any forecast bands built naively from them — ignore the estimation error in the factors themselves. For point forecasts this two-step approach is fast and famously effective; for *inference* it understates uncertainty. The principled fix is to write DNS as a single state-space model — factors as latent states, the AR(1) as the transition, the loadings as the measurement matrix — and estimate it in one pass with the Kalman filter of chapter 4, which propagates factor uncertainty into everything downstream. Use the two-step for forecasting and exploration; reach for the one-step state-space form when you need honest error bands.

## Arbitrage-free Nelson-Siegel: keep the loadings, add one term

Everything to this point is *reduced-form* curve-fitting. Nelson-Siegel, Svensson, and dynamic Nelson-Siegel describe the curve and forecast it beautifully, but none of them is **arbitrage-free**: nothing in the fit forces the yields it quotes across maturities to be mutually consistent under no-arbitrage. That is fine for summarizing and forecasting — it is *not* fine the moment you price a derivative off the curve, extract risk-neutral rate expectations, or want a long yield that a trader could not, in principle, arbitrage against the shorter maturities. A no-arbitrage model ties the cross-section and the dynamics together with a single restriction; the price of admission has always seemed to be giving up the interpretable level-slope-curvature loadings for the machinery of an affine term-structure model.

Christensen, Diebold and Rudebusch (2011) showed you do not have to give them up. Their result is remarkably clean: **keep all three Nelson-Siegel factor loadings exactly as they are, and add one deterministic yield-adjustment term $-A(\tau)/\tau$.** With that single term the curve becomes arbitrage-free — a member of the affine class — while the part you like, the shapes that read off as level, slope, and curvature, is left completely untouched. The AFNS curve (independent-factor case, a diagonal factor-volatility matrix $\Sigma = \mathrm{diag}(\sigma_{11}, \sigma_{22}, \sigma_{33})$) is Nelson-Siegel plus that one correction:

$$
y(\tau) \;=\; \underbrace{\beta_1 \;+\; \beta_2\,\frac{1 - e^{-\lambda \tau}}{\lambda \tau} \;+\; \beta_3\!\left(\frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau}\right)}_{\text{ordinary Nelson-Siegel, unchanged}} \;-\; \frac{A(\tau)}{\tau} .
$$

The correction $A(\tau)/\tau$ is a function of the *factor volatilities* and the decay — not of the fitted factors — and it captures a convexity, or Jensen's-inequality, effect that the reduced-form model simply omits. Here is the intuition. A long yield is an average of expected future short rates, and expectations of a nonlinear (convex) function of a volatile state sit *below* the function of the expectation: $\mathbb{E}[f(X)] < f(\mathbb{E}[X])$ for the relevant curvature. The more volatile the factors and the longer the horizon over which that volatility compounds, the larger the wedge. So the adjustment is **non-positive** — it can only pull yields down, never up — and it **deepens with maturity**, because volatility has more time to compound out at the long end. At the short end there is almost nothing to correct; the gap opens up precisely where pricing cares about it. And it **vanishes as $\Sigma \to 0$**: with no factor volatility there is no convexity wedge, so $A(\tau)/\tau \to 0$ and AFNS collapses back onto plain Nelson-Siegel exactly. AFNS *nests* the reduced-form model, adding the no-arbitrage correction as a volatility-scaled perturbation on top of it.

tsecon's `afns_adjustment` computes exactly this term — the signed $-A(\tau)/\tau$, one value per maturity — from a maturity grid, the three diagonal factor volatilities $\sigma = [\sigma_{11}, \sigma_{22}, \sigma_{33}]$ (level, slope, curvature vols), and the decay. It does not fit anything: the volatilities are inputs you bring from a dynamic AFNS estimation, a calibration, or a scenario. You add the returned array to any Nelson-Siegel curve to obtain its arbitrage-free companion, $y_{\mathrm{AFNS}}(\tau) = y_{\mathrm{NS}}(\tau) + \text{adjustment}$. One keyword deserves a flag up front: the decay argument is spelled **`decay`, not `lambda`** — `lambda` is a reserved word in Python, so the library cannot use it.

Take the last curve of our Treasury panel, extract its Nelson-Siegel factors as before, reconstruct the reduced-form curve, and then make it arbitrage-free by adding the adjustment. We supply an illustrative set of monthly factor volatilities (level, slope, curvature) in the same month/decay units as the loadings:

```python
lam = 0.0609
yields = panel[-1]                                     # last curve in the panel
fit = tsecon.nelson_siegel(maturities, yields, decay=lam)
L, S, C = fit["level"], fit["slope"], fit["curvature"]

# Reduced-form NS curve on the observed maturity grid.
b_slope = (1 - np.exp(-lam*maturities)) / (lam*maturities)
b_curv  = b_slope - np.exp(-lam*maturities)
y_ns = L + S*b_slope + C*b_curv

# Illustrative independent-factor vols [level, slope, curvature], monthly units.
sigma = np.array([0.010, 0.020, 0.030])
adj = tsecon.afns_adjustment(maturities, sigma, decay=lam)   # signed -A(tau)/tau
y_afns = y_ns + adj

print("non-positive          :", bool(np.all(adj <= 0.0)))
print("deepens with maturity :", bool(np.all(np.diff(adj) <= 0.0)))
print("  tau    y_NS   y_AFNS   gap(bp)")
for t, a, b in zip(maturities, y_ns, y_afns):
    print(f"{t:5.0f}  {a:6.3f}  {b:6.3f}   {(b-a)*100:7.2f}")
# non-positive          : True
# deepens with maturity : True
#   tau    y_NS   y_AFNS   gap(bp)
#     3   5.910   5.909     -0.07
#     6   5.870   5.868     -0.25
#    12   5.804   5.795     -0.89
#    24   5.709   5.678     -3.05
#    36   5.648   5.587     -6.01
#    60   5.578   5.446    -13.20
#    84   5.541   5.328    -21.40
#   120   5.513   5.154    -35.87
```

Read the last column. Inside the first year the arbitrage-free curve is indistinguishable from the reduced-form one — the gap is under a basis point out to 12 months. Then it widens monotonically: 3 bp at 2 years, 13 bp at 5 years, and 36 bp by the 10-year point, the long end always pulled *down*. That widening, non-positive wedge is the entire content of the no-arbitrage restriction, and its magnitude scales with the *square* of the factor volatilities — dominated at the long end by the level vol $\sigma_{11}$, whose contribution grows like $\sigma_{11}^2\,\tau^2/6$. The specific numbers here are as large as they are because these illustrative vols are on the generous side; halve them and the long-end gap falls by a factor of four. The shape and the sign, though, are not calibration artifacts — they are what no-arbitrage requires.

The nesting is worth seeing directly, because it is the sanity check that the adjustment is a perturbation and nothing more:

```python
z = tsecon.afns_adjustment(maturities, np.array([0.0, 0.0, 0.0]), decay=lam)
print("sigma = 0 -> adjustment all zero:", bool(np.all(z == 0.0)))
# sigma = 0 -> adjustment all zero: True
```

With the factor volatilities switched off there is no convexity wedge, the adjustment is identically zero, and $y_{\mathrm{AFNS}} = y_{\mathrm{NS}}$: AFNS *is* Nelson-Siegel when volatility is zero. The full closed form of $A(\tau)/\tau$ — the slope- and curvature-vol terms on top of the level term shown above — and the exact argument contract live on the [AFNS model card](../reference/model-cards/afns.md). The teaching point stands on its own: no-arbitrage does not cost you the loadings you spent this chapter learning to read; it costs you one deterministic, volatility-scaled term that bends the long end down.

> **⚠ Common mistake — reading `afns_adjustment` as a fit.** It estimates nothing. It is a deterministic evaluation of the CDR closed form, and it is only as meaningful as the factor volatilities $\sigma$ you feed it — garbage vols give a garbage adjustment with no residual, no $R^2$, and no diagnostic to warn you. The volatilities must come from a dynamic AFNS estimation, a calibration, or an explicit scenario; and they, the maturities, and `decay` must all be in one consistent time unit (months with $\lambda = 0.0609$ here, or years with a years-scaled decay). A monthly vol against a yearly grid mis-scales the whole correction. And do not read the adjustment as a forecast or a risk premium — it is the convexity term, nothing more.

## The frontier

**Beyond the independent-factor AFNS.** The `afns_adjustment` above is the arbitrage-free bridge for the *diagonal*-volatility case; the general correlated-factor AFNS carries cross-terms, and estimating the factor volatilities jointly with the dynamics is a dynamic term-structure problem in its own right. Both sit on the road to the full affine term-structure models (Duffie-Kan 1996; Dai-Singleton 2000; Ang-Piazzesi 2003) that asset pricing is built on — models where the cross-section and the dynamics are locked together by no-arbitrage from the start rather than corrected after the fact.

**Yields and the macroeconomy, jointly.** The level, slope, and curvature are not just statistical conveniences — the slope famously predicts recessions, and the level tracks long-run inflation expectations. Diebold, Rudebusch and Aruoba (2006) put the DNS factors and macro variables (output, inflation, the policy rate) into *one* state-space VAR, letting the curve and the economy feed back on each other. This is the natural marriage of this chapter with the VARs of chapter 7 and the nowcasting of chapter 11, and it is where curve modeling meets monetary policy analysis.

**Term-premium decomposition.** A long yield is (roughly) the average expected short rate over the bond's life *plus* a term premium for bearing duration risk. Splitting the two is a central question for policy, and the regression-based decomposition of Adrian, Crump and Moench (2013) — three curve factors plus a pricing regression — has become the practitioner standard, precisely because it lives one step beyond the factor extraction this chapter teaches.

**The lower bound.** When policy rates sit at or below zero, Gaussian curve models happily predict deeply negative yields that markets will not deliver. Shadow-rate models (Black 1995; Krippner 2013; Wu and Xia 2016) let a latent "shadow" short rate go negative while the observed rate is floored at the bound — the term-structure analogue of the censoring problems elsewhere in econometrics, and essential for any curve work spanning the 2009–2015 or post-2020 episodes.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| Summarize or interpolate a *single* smooth curve | `nelson_siegel` with fixed `decay` | Three interpretable factors, one OLS, no optimizer; the field standard |
| Best-possible fit to *one* curve, comparability not needed | `nelson_siegel(optimal_lambda=True)` | NLS on $\lambda$ squeezes out the last basis points of cross-sectional fit |
| Curve with a genuine *second* hump (long maturities, odd regimes) | `svensson` | The fourth factor flexes a second bend; ECB-style production fitting |
| A *panel* of curves — extract factor series and forecast | `dynamic_ns` | Collapses a many-maturity forecast to three AR(1)s; the Diebold-Li workhorse |
| Comparable factors across dates in a panel | Fix `decay` globally, never re-optimize per date | Time-varying loadings make the factor series incommensurable |
| Honest forecast *bands*, not just point forecasts | State-space DNS + Kalman filter (roadmap) | Two-step SEs ignore factor estimation error; the one-step form propagates it |
| Make a Nelson-Siegel curve arbitrage-free (pricing, risk-neutral expectations) | `afns_adjustment` added to an NS fit | DNS is statistical, not arbitrage-free; AFNS keeps the loadings and adds the one no-arb yield-adjustment term |
| Curve *and* macro feedback (recession signals, policy) | DNS-macro state-space VAR (roadmap) | Ties the factors to output, inflation, and the policy rate in one system |
| Yields near or below zero | Shadow-rate models (roadmap) | Gaussian curve models predict impossible deeply-negative yields at the bound |

## What tsecon implements today

**Available now in Python** — everything this chapter's runnable code used, validated against the golden fixture [`fixtures/termstructure.json`](../../fixtures/termstructure.json):

- `tsecon.nelson_siegel(maturities, yields, decay=..., optimal_lambda=...)` — the three-factor Diebold-Li fit, with either a fixed decay or an NLS-optimal one; returns `level`, `slope`, `curvature`, `factors`, `lambda`, `residuals`, and `rsquared`
- `tsecon.svensson(maturities, yields, lambda1=..., lambda2=...)` — the four-factor extension that nests Nelson-Siegel; returns the four `factors`, both decays, `residuals`, and `rsquared`
- `tsecon.dynamic_ns(panel, maturities, decay=...)` — the two-step dynamic model over a $T \times$ (n maturities) panel; returns the `factors` path (T × 3), per-date `rsquared`, the named `level`/`slope`/`curvature` series, and a `forecast` dict with the one-step-ahead `factors` and `yields` plus the fitted `ar1_intercept` and `ar1_phi`
- `tsecon.afns_adjustment(maturities, sigma, decay=...)` — the Christensen-Diebold-Rudebusch (2011) independent-factor yield-adjustment term $-A(\tau)/\tau$; add it to a Nelson-Siegel curve to obtain its arbitrage-free companion. Takes the three-element factor-vol diagonal `sigma` (level, slope, curvature) and returns the signed, non-positive, maturity-deepening adjustment; note the keyword is `decay`, not `lambda`

All four take **NumPy arrays** — pass `np.array(...)`, not bare Python lists. The factor extraction is a cross-sectional OLS (chapter 2's regression machinery); the optimal-$\lambda$ and Svensson fits add a small nonlinear search over the decay(s); the dynamic model layers per-factor AR(1) estimation (chapter 4) on top; and `afns_adjustment` is a deterministic evaluation of the CDR closed form given volatilities you supply.

**Roadmap** — the further extensions the frontier section named are specified but not yet callable: the one-step **state-space / Kalman-filter** DNS (for honest forecast bands), the general **correlated-factor AFNS** and full affine term-structure models (for pricing), the **DNS-macro** joint state-space VAR (for curve-macro feedback), the **Adrian-Crump-Moench** term-premium decomposition, and **shadow-rate** models for the lower bound.

## Further reading

- **Nelson, C. R. and A. F. Siegel (1987), "Parsimonious Modeling of Yield Curves," *Journal of Business*.** The original three-factor curve; where the level/slope/curvature basis functions come from.
- **Diebold, F. X. and C. Li (2006), "Forecasting the Term Structure of Government Bond Yields," *Journal of Econometrics*.** The paper that made Nelson-Siegel *dynamic* and *forecastable* — fixed $\lambda$, factors as AR time series, the two-step estimator this chapter's `dynamic_ns` implements.
- **Litterman, R. and J. Scheinkman (1991), "Common Factors Affecting Bond Returns," *Journal of Fixed Income*.** The empirical discovery that three factors — level, slope, curvature — drive the whole curve; the names this chapter uses.
- **Svensson, L. E. O. (1994), "Estimating and Interpreting Forward Interest Rates: Sweden 1992–1994," *NBER Working Paper 4871*.** The four-factor extension with a second curvature term; the basis of many central banks' production curve-fitting.
- **Christensen, J. H. E., F. X. Diebold and G. D. Rudebusch (2011), "The Affine Arbitrage-Free Class of Nelson-Siegel Term Structure Models," *Journal of Econometrics*.** The arbitrage-free member of the family — keeps the loadings, adds the no-arbitrage yield-adjustment term.
- **Diebold, F. X., G. D. Rudebusch and S. B. Aruoba (2006), "The Macroeconomy and the Yield Curve: A Dynamic Latent Factor Approach," *Journal of Econometrics*.** The DNS factors and macro variables in one state-space system; curve meets economy.
- **Adrian, T., R. K. Crump and E. Moench (2013), "Pricing the Term Structure with Linear Regressions," *Journal of Financial Economics*.** The practitioner-standard regression-based term-premium decomposition, built on curve factors.
- **Diebold, F. X. and G. D. Rudebusch (2013), *Yield Curve Modeling and Forecasting: The Dynamic Nelson-Siegel Approach*, Princeton University Press.** The book-length treatment — fitting, dynamics, the state-space form, macro extensions, and arbitrage-free versions in one place.
