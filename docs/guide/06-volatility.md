# Chapter 6 — Volatility: GARCH and Risk

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** Chapters 1–3 — you should be comfortable with autocorrelation and the Ljung-Box test, maximum likelihood at the "write down the likelihood, maximize it" level, and the sandwich-variance idea behind robust standard errors.

**You will learn:**

- The three stylized facts of financial returns — volatility clustering, fat tails, the leverage effect — and *why* each one arises
- How ARCH and GARCH(1,1) turn "risk changes over time" into an estimable model, and what persistence, unconditional variance, and half-life mean
- How asymmetric models (GJR, EGARCH) and non-normal distributions fix GARCH's two biggest misspecifications, and why QMLE with Bollerslev-Wooldridge standard errors keeps inference honest anyway
- How to forecast variance over any horizon and convert it into Value-at-Risk and Expected Shortfall that survive a backtest
- Where the field is now: realized volatility from intraday data, HAR, DCC, and stochastic volatility

## The idea

Ask a simple question: can you predict tomorrow's stock return? Decades of evidence say essentially no — daily returns on a broad index are close to unpredictable in their *level*. If markets left easy money on the table, someone would have picked it up.

Now ask a different question: can you predict tomorrow's *risk*? Here the answer is emphatically yes. Look at a chart of daily S&P 500 returns over twenty years. The line hovers in a narrow band for months, then — around 2008, around March 2020 — it explodes into wild swings that persist for weeks before slowly calming down. Quiet days cluster with quiet days; violent days cluster with violent days. You cannot say whether tomorrow will be up or down, but you can say with real confidence whether tomorrow will be a *big* day or a *small* one, just by looking at the last few weeks.

That is the entire subject of this chapter: the mean of returns is nearly unforecastable, but the **variance** — the size of the typical move — is highly forecastable. Volatility modeling is the study of that second moment: how it evolves, how to estimate it, how to forecast it, and how to turn the forecast into a defensible risk number. A bank's regulatory capital, an option's price, and a portfolio manager's position size all rest on exactly this forecast.

The mechanics, in plain English: today's uncertainty is a weighted blend of yesterday's uncertainty and the size of yesterday's surprise. A big shock — in either direction — raises the market's temperature, and the temperature cools off only gradually. Everything else in this chapter is refinement: how fast the cooling is, whether bad news heats things more than good news, how fat the tails of the shocks are, and how to measure the temperature directly when you have intraday data.

## Three stylized facts of returns

Before writing any model, know what the data insist on. Three empirical regularities show up in virtually every liquid asset — equities, exchange rates, commodities, crypto — and any volatility model earns its keep by reproducing them.

**1. Volatility clustering.** Large changes tend to be followed by large changes, of either sign, and small changes by small changes — first documented by Mandelbrot (1963) for cotton prices. The signature in the data: returns themselves are nearly uncorrelated, but *squared* returns (or absolute returns) are strongly, persistently autocorrelated. Why it happens: information arrives in bursts (earnings seasons, crises, policy shocks), and markets take time to digest big news — trading volume and disagreement stay elevated for days after a shock.

**2. Fat tails.** Extreme returns occur far more often than a normal distribution allows — Fama (1965) already found daily stock returns wildly non-normal. A daily return five standard deviations from the mean should occur roughly once in 7,000 years under normality; real markets deliver one every few years. Why: part of it is clustering itself — a mixture of quiet regimes and turbulent regimes is fat-tailed even if each regime is normal — and part is genuinely jumpy news within a day.

**3. The leverage effect.** Negative returns raise future volatility more than positive returns of the same size — noted by Black (1976). Two stories compete: the original *leverage* story (a price drop mechanically raises a firm's debt-to-equity ratio, making the equity riskier) and the *volatility feedback* story (anticipated higher risk requires a higher expected return, which is achieved by an immediate price drop). Empirically the asymmetry is too large and too fast for the mechanical story alone, but the name stuck.

Two further regularities matter for model choice later in the chapter. Volatility has **long memory**: the autocorrelation of absolute or squared returns decays much more slowly than the geometric rate a short-memory model implies, staying visibly positive for hundreds of days — and absolute returns are actually *more* autocorrelated than squared ones, the "Taylor effect" documented by Ding, Granger, and Engle (1993). And volatility **comoves**: when one asset's volatility rises, others' tend to rise with it, which is why the multivariate models in the frontier section exist at all.

You can verify the three headline facts with tools you already have. The example simulates a GARCH process (defined formally in the next section) so you can see the facts emerge from the recursion itself:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(7)
n = 2500                                  # ~10 years of daily percent returns
omega, alpha, beta = 0.05, 0.10, 0.85     # persistence 0.95 — typical for equities
sigma2 = np.empty(n)
eps = np.empty(n)
sigma2[0] = omega / (1 - alpha - beta)    # start at the unconditional variance
z = rng.standard_normal(n)                # note: NORMAL shocks
for t in range(n):
    if t > 0:
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    eps[t] = np.sqrt(sigma2[t]) * z[t]
r = eps

tsecon.ljung_box(r, nlags=10)             # returns: no linear predictability
tsecon.ljung_box(r**2, nlags=10)          # squared returns: p-values ~ 0
tsecon.jarque_bera(r)                     # kurtosis > 3, normality rejected
tsecon.arch_lm(r, nlags=10)               # Engle's ARCH-LM test: rejects decisively
```

The Ljung-Box test on `r` finds nothing; on `r**2` it rejects overwhelmingly — that contrast *is* volatility clustering. And `jarque_bera` rejects normality even though every shock `z` was drawn from a normal distribution: time-varying variance alone manufactures fat unconditional tails.

> **⚠ Common mistake.** "My returns have fat tails, so I must use a Student-t model." Not necessarily. A GARCH model with normal innovations already implies a fat-tailed *unconditional* distribution — clustering does part of the job. The right diagnostic is the distribution of the **standardized residuals** $\hat\varepsilon_t/\hat\sigma_t$ *after* fitting the variance model. If those are still fat-tailed (they usually are, somewhat), then reach for a t distribution.

## ARCH: variance you can regress on

Engle (1982) made the second moment estimable with one move: let the conditional variance be a function of observable past shocks. Write returns as

$$r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t, \qquad z_t \sim \text{iid}(0, 1),$$

where $\mu$ is the (nearly unforecastable) mean, $\varepsilon_t$ is the surprise, $z_t$ is a standardized innovation, and $\sigma_t^2$ — the **conditional variance** — is known at time $t-1$. The ARCH($q$) model ("autoregressive conditional heteroskedasticity") says

$$\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \varepsilon_{t-i}^2, \qquad \omega > 0,\ \alpha_i \ge 0.$$

Yesterday's squared surprise feeds today's variance directly. The model is mostly pedagogical now — real returns need long lags, so $q$ balloons — but it contributes the field's standard specification test. The **ARCH-LM test** regresses $\hat\varepsilon_t^2$ on its own $q$ lags; under the null of no ARCH effects, $T \cdot R^2$ from that auxiliary regression is asymptotically $\chi^2_q$. That is exactly what `tsecon.arch_lm(resid, nlags=10)` computes (in the statsmodels `het_arch` convention). Run it on the residuals of *any* time series regression: a rejection means your homoskedastic standard errors are suspect and a volatility model has something to say.

## GARCH(1,1): the workhorse

A practitioner cares about GARCH(1,1) because it is the single most successful forecasting model in financial econometrics per parameter spent — Hansen and Lunde (2005) compared 330 volatility specifications and found none that reliably beat it for exchange-rate data. Bollerslev (1986) added one term to ARCH: today's variance also depends on *yesterday's variance*, not just yesterday's shock:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2,$$

with $\omega > 0$, $\alpha, \beta \ge 0$. Substituting the recursion into itself shows GARCH(1,1) is an ARCH($\infty$) with geometrically decaying weights $\alpha, \alpha\beta, \alpha\beta^2, \dots$ — an exponentially weighted memory of all past squared shocks, bought with two parameters. The interpretation of each:

- $\alpha$ (the **ARCH term**) is the reaction to news: how much a fresh squared shock moves the variance. Equity data typically give $\alpha \approx 0.05$–$0.10$.
- $\beta$ (the **GARCH term**) is the memory: how much of yesterday's variance carries into today. Typically $\beta \approx 0.85$–$0.93$.
- $\omega$ anchors the level.

Three derived quantities do most of the talking in applied work:

**Persistence.** $\alpha + \beta$ governs how long a volatility shock lasts. Provided $\alpha + \beta < 1$ the process is covariance-stationary and mean-reverting.

**Unconditional variance.** Taking expectations of both sides and solving:

$$\bar\sigma^2 = \frac{\omega}{1 - \alpha - \beta}.$$

This is the long-run average variance the process reverts to — the "climate" beneath the daily "weather."

**Half-life.** Deviations of the conditional variance from $\bar\sigma^2$ decay by the factor $\alpha+\beta$ each day, so the number of days for half the deviation to dissipate is

$$h_{1/2} = \frac{\ln 0.5}{\ln(\alpha + \beta)}.$$

Persistence 0.95 gives a half-life of about 13.5 trading days; 0.98 gives 34; 0.99 gives 69. Fitted equity persistence routinely lands in 0.95–0.99, which is why turbulence takes weeks to fade — exactly the clustering the eyeball sees.

One special case is everywhere in industry. Set $\omega = 0$ and $\beta = \lambda$, $\alpha = 1 - \lambda$, and GARCH(1,1) becomes the **RiskMetrics EWMA** (J.P. Morgan 1996):

$$\sigma_t^2 = (1 - \lambda)\, \varepsilon_{t-1}^2 + \lambda\, \sigma_{t-1}^2, \qquad \lambda = 0.94 \text{ for daily data}.$$

Nothing is estimated — $\lambda$ is fixed by convention — which is exactly why risk systems love it: no optimizer, no boundary problems, one line of code. But note what was given up. Persistence is $\alpha + \beta = 1$ exactly: this is an **integrated GARCH** (IGARCH), the unconditional variance $\omega/(1-\alpha-\beta)$ no longer exists, shocks to variance never decay, and the $h$-step forecast is flat at tomorrow's value for every horizon. EWMA is a serviceable filter for *tomorrow's* volatility and a poor model for any horizon beyond it — the mean reversion that estimated GARCH captures (and that the forecasting section below exploits) is precisely what the shortcut deletes.

Estimation of the full model is by maximum likelihood: given a distribution for $z_t$, the likelihood factors into one term per observation, each a normal (say) density with variance $\sigma_t^2$ computed by running the recursion forward. The recursion needs a starting value $\sigma_0^2$ — the **backcast**, typically an exponentially weighted average of the first squared residuals.

*Roadmap preview — this API lands with Module 03:*

```python
fit = tsecon.garch_fit(r, p=1, q=1, dist="t")    # QMLE, arch-style backcast init
fit["params"]            # mu, omega, alpha, beta, nu — the arch package ordering
fit["robust_bse"]        # Bollerslev-Wooldridge sandwich standard errors
fit["conditional_vol"]   # the sigma_t path; annualize with * np.sqrt(252)
fit["persistence"]       # alpha + beta, plus the implied half-life
```

The Rust engine behind this (`tsecon-garch`) is already built and validated against Kevin Sheppard's `arch` package: fixed-parameter log-likelihoods match to 1e-8 relative, conditional volatilities to 1e-6, robust standard errors to 5e-3.

> **⚠ Common mistake.** Fitting GARCH on returns in *decimal* units (0.01 for one percent). The log-likelihood surface becomes nearly flat in $\omega$ and optimizers quietly fail or stop early — this is the best-known gotcha in the `arch` package, which warns about it loudly. Fit on percent returns. Relatedly, two packages given the same data can report different estimates simply because they initialize $\sigma_0^2$ differently; when comparing results across software, match the variance initialization before suspecting a bug.

## Asymmetry: bad news is louder

Plain GARCH squares the shock, so a −2% day and a +2% day raise tomorrow's variance identically. The leverage effect says that is wrong — for equities, materially wrong. A practitioner who ignores it will underpredict volatility after crashes, which is precisely when the prediction matters. Two fixes dominate applied work.

**GJR-GARCH** (Glosten, Jagannathan, and Runkle 1993) adds a threshold term that activates only on bad news:

$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \gamma\, \varepsilon_{t-1}^2\, \mathbf{1}[\varepsilon_{t-1} < 0] + \beta \sigma_{t-1}^2,$$

where $\mathbf{1}[\cdot]$ is the indicator function. A positive shock contributes $\alpha \varepsilon^2$; a negative shock contributes $(\alpha + \gamma)\varepsilon^2$. Equity fits typically find $\gamma$ significantly positive and often larger than $\alpha$ itself — bad news can matter twice as much. Persistence generalizes to $\alpha + \beta + \gamma \cdot P(z_t < 0)$, which is $\alpha + \beta + \gamma/2$ when the innovations are symmetric.

**EGARCH** (Nelson 1991) instead models the *logarithm* of variance, in the parameterization tsecon shares with the `arch` package:

$$\ln \sigma_t^2 = \omega + \alpha \left( |z_{t-1}| - \sqrt{2/\pi} \right) + \gamma z_{t-1} + \beta \ln \sigma_{t-1}^2.$$

The $\alpha$ term reacts to the *magnitude* of the standardized shock (centered by $E|z| = \sqrt{2/\pi}$ for a standard normal), the $\gamma$ term to its *sign* — leverage shows up as $\gamma < 0$. Because the recursion lives in logs, $\sigma_t^2$ is positive by construction and no non-negativity constraints on parameters are needed, which is EGARCH's practical selling point when constraints bind in estimation.

The standard way to *see* an asymmetric model is the **news-impact curve** (Engle and Ng 1993): plot tomorrow's variance as a function of today's shock, holding lagged variance fixed at its unconditional level. Plain GARCH draws a symmetric parabola; GJR draws a parabola with a steeper left arm; EGARCH draws a curve that is asymmetric and, in logs, kinked at zero. Engle and Ng's companion **sign-bias tests** — regressions of squared standardized residuals on sign indicators of past shocks — are the formal check: if they reject after a symmetric GARCH fit, move to GJR or EGARCH.

> **⚠ Common mistake.** Multi-step forecasts from asymmetric models need $P(z < 0)$ and the partial moment $E[z^2 \mathbf{1}(z<0)]$ under the *actual* innovation distribution. Hardcoding $P(z<0) = 1/2$ is only correct for symmetric innovations — with a skew-t distribution it silently biases every GJR forecast. (This is one of the parity traps the tsecon forecast engine is tested against.)

## Fat tails and honest inference: distributions and QMLE

Even after GARCH soaks up clustering, standardized residuals $\hat z_t = \hat\varepsilon_t / \hat\sigma_t$ are usually still leptokurtic — too many outliers for a normal. Two responses, and you should understand both because they answer different questions.

**Response 1: use a better distribution.** Bollerslev (1987) proposed the standardized Student-t: with $\nu$ degrees of freedom (and $\nu > 2$ so the variance exists), the innovation is scaled by $\sqrt{(\nu-2)/\nu}$ so that it has unit variance whatever $\nu$ is. Daily equity fits typically estimate $\nu \approx 5$–$10$. The generalized error distribution (GED, used by Nelson 1991) is an alternative with both thinner- and fatter-than-normal options. When the residuals are also *skewed* — crashes bigger than rallies — skew-t distributions add an asymmetry parameter. One warning the literature has learned the hard way: Hansen (1994) and Fernandez and Steel (1998) define *different* distributions that are both called "skew-t," and mixing them up is a classic replication failure. A correct distribution matters most when the object of interest is a tail quantile — which, for risk management, it always is.

**Response 2: keep the normal likelihood but fix the standard errors.** Here is the deep result that makes GARCH practice defensible. Maximizing the *normal* likelihood when the true innovations are not normal is called **quasi-maximum likelihood estimation (QMLE)** — and for GARCH-type models it still gives consistent estimates of the variance parameters, because the normal score identifies the conditional variance correctly regardless of the true innovation law. What breaks is the *information-matrix equality*: the usual Hessian-based standard errors are no longer valid. Bollerslev and Wooldridge (1992) supply the fix — the sandwich covariance

$$\widehat{\mathrm{Avar}}(\hat\theta) = \hat A^{-1} \hat B \hat A^{-1},$$

where $\hat A$ is the average Hessian of the log-likelihood and $\hat B$ the average outer product of scores. If the normal likelihood were true, $A = B$ and the sandwich collapses to the usual MLE variance; when it is not, only the sandwich is right.

You have met this logic before: it is the same "the point estimate is fine, the naive variance is not" reasoning behind the HC and HAC corrections of Chapter 3. There the model for the mean was trusted and the error variance was misspecified; here the model for the variance is trusted and the innovation *density* is misspecified. Same sandwich, one level up.

> **⚠ Common mistake.** Reporting Hessian-only (or OPG-only) standard errors from a GARCH fit. Under non-normal innovations — i.e., always — they are simply wrong, often by a factor large enough to flip significance on $\alpha$. Robust (Bollerslev-Wooldridge) standard errors are the correct default, which is why `tsecon-garch` computes them as a first-class output rather than an option buried in a corner.

One more honesty note: when $\hat\alpha$ sits at zero or $\hat\alpha + \hat\beta$ sits at one, the parameter is on the boundary of its space and standard asymptotics fail entirely (Andrews 2001). A good library detects boundary solutions and warns rather than printing meaningless t-statistics.

## Forecasting variance: the term structure of volatility

Volatility forecasts are the product. An option desk needs the average variance over the option's life; a risk desk needs tomorrow's; a pension fund needs next quarter's. GARCH delivers the whole **term structure** in closed form.

The one-step forecast is just the recursion: $\sigma_{T+1}^2$ is known at time $T$. For $h \ge 2$, use $E_T[\varepsilon_{T+h-1}^2] = E_T[\sigma_{T+h-1}^2]$ and iterate:

$$E_T[\sigma_{T+h}^2] = \bar\sigma^2 + (\alpha + \beta)^{h-1}\left( \sigma_{T+1}^2 - \bar\sigma^2 \right).$$

Read it as a statement about mean reversion: the forecast starts at the current conditional variance and glides geometrically toward the unconditional variance $\bar\sigma^2$, at speed set by the persistence. After a crash, the term structure of forecast volatility slopes *down* (turbulence is expected to fade); in a calm market it slopes *up* (calm is also temporary). The half-life from the previous section is exactly the horizon at which half the gap has closed.

For the variance of a *multi-day* return — what a 10-day VaR actually needs — sum the daily forecasts (returns are serially uncorrelated, so variances add):

$$\mathrm{Var}_T\!\left[ \textstyle\sum_{h=1}^{H} r_{T+h} \right] = \sum_{h=1}^{H} E_T[\sigma_{T+h}^2].$$

Because the summands are not all equal to $\sigma_{T+1}^2$, this is **not** $H \sigma_{T+1}^2$.

A convention worth fixing here: practitioners quote **annualized volatility** — a daily standard deviation multiplied by $\sqrt{252}$ (trading days per year), so daily $\sigma = 1\%$ reads as "16 vol." That is a *units convention* for stating today's volatility, not a forecast: the honest one-year-ahead variance is the term-structure sum above, which sits between today's conditional variance and the unconditional one. Keep the two ideas separate and both are useful; conflate them and you have rediscovered the square-root-of-time rule.

Closed forms like the one above exist for GARCH and (with the partial moments mentioned earlier) for GJR. EGARCH has none: the recursion is in logs, and $E[\sigma^2] \ne \exp(E[\ln \sigma^2])$ by Jensen's inequality, so multi-step EGARCH forecasts must be simulated. This is where a parallel simulation engine stops being a luxury: honest EGARCH term structures, bootstrap prediction intervals (Pascual, Romo, and Ruiz 2006), and filtered historical simulation all reduce to simulating many paths through the fitted recursion.

> **⚠ Common mistake.** Scaling one-day volatility to $h$ days with $\sqrt{h}$ (the "square-root-of-time rule"). That rule assumes variance is *constant* — the one thing this whole chapter says it is not. Right after a shock, $\sqrt{h}$-scaling overstates long-horizon risk (it ignores mean reversion down); in a calm, it understates. The error compounds inside VaR: a 10-day VaR built by $\sqrt{10} \times$ one-day VaR is wrong in exactly the states where risk is mispriced most.

## From variance to risk: VaR and ES

Variance is a modeling object; a risk desk reports quantiles. Two risk measures dominate, and precision about their definitions prevents an entire genre of silent bugs.

Work with the **loss** $L_{t+1} = -r_{t+1}$, so that losses are positive numbers (fix this convention once and convert at the boundaries — tsecon's risk object stores its convention explicitly for exactly this reason). For a tail level $\alpha$ (say 1%):

**Value-at-Risk** is the loss quantile:

$$\mathrm{VaR}^{\alpha}_{t+1} = \inf\{ x : P(L_{t+1} \le x \mid \mathcal{F}_t) \ge 1 - \alpha \},$$

the loss exceeded with probability only $\alpha$. Under a location-scale model $r_{t+1} = \mu + \sigma_{t+1} z$, it has the parametric form $\mathrm{VaR}^{\alpha}_{t+1} = -\mu - \sigma_{t+1} q_\alpha$, where $q_\alpha = F_z^{-1}(\alpha)$ is the $\alpha$-quantile of the standardized innovation distribution — one more reason the distribution choice above was not cosmetic.

**Expected Shortfall** is the expected loss *given* a VaR exceedance:

$$\mathrm{ES}^{\alpha}_{t+1} = E\left[ L_{t+1} \mid L_{t+1} > \mathrm{VaR}^{\alpha}_{t+1},\ \mathcal{F}_t \right].$$

VaR answers "how bad is a bad day"; ES answers "how bad is a bad day *once it has arrived*." ES is subadditive (diversification cannot increase it) where VaR can fail to be, and it sees tail shape beyond the quantile — which is why the Basel Committee moved market-risk capital from 1% VaR to 2.5% ES. For a normal innovation the mean-zero formulas are $\mathrm{VaR}^\alpha = -\sigma q_\alpha$ and $\mathrm{ES}^\alpha = \sigma \phi(q_\alpha)/\alpha$ with $\phi$ the standard normal density; Student-t and skew-t versions need the distribution's partial expectations.

The parametric route is not the only one, and for the far tail it is often not the best. **Filtered historical simulation** (Barone-Adesi, Giannopoulos, and Vosper 1999) drops the distributional assumption: take the empirical distribution of the model's own standardized residuals $\hat z_t$, read the quantile off it, and scale by the forecast $\hat\sigma_{t+1}$ — the volatility model does the time-variation, the data do the tail shape. For quantiles deeper than the data can support directly (0.1% with a few years of history), **extreme value theory** fits a generalized Pareto distribution to the largest standardized residuals and extrapolates the tail parametrically — the McNeil and Frey (2000) two-step that remains the standard prescription for far-tail VaR and ES. Both pipelines sit on the roadmap's risk layer; FHS additionally needs only the bootstrap machinery tsecon already ships.

A risk model is validated by **backtesting**: compare the stream of VaR forecasts against realized losses and examine the **hit sequence** $I_t = \mathbf{1}[L_t > \mathrm{VaR}^\alpha_t]$. If the model is right, two things must hold:

- **Correct unconditional coverage** — hits occur at rate $\alpha$. Kupiec (1995) tests this with a likelihood-ratio statistic comparing the binomial likelihood at $\hat\pi = x/T$ (with $x$ observed hits in $T$ days) against the null rate $\alpha$; asymptotically $\chi^2_1$.
- **Independence** — hits do not cluster. A VaR that is violated four days in a row at the start of a crisis is failing exactly when it matters, even if its annual hit count looks fine. Christoffersen (1998) tests independence by fitting a first-order Markov chain to the hit sequence and testing whether the probability of a hit depends on yesterday's hit; the **conditional coverage** test combines both requirements into one $\chi^2_2$ statistic.

Independence failures are the fingerprint of a variance model that adapts too slowly — the fix is a better volatility model, not a bigger multiplier.

The whole pipeline fits in a few lines. Continuing with `r` and `sigma2` from the simulation in the stylized-facts section (where the true conditional variance is known, so this is the best case a model can hope for):

```python
q01 = -2.3263                        # 1% quantile of the standard normal
var01 = -np.sqrt(sigma2) * q01       # one-step 1% VaR path, as a positive loss
hits = (-r) > var01                  # the hit sequence I_t
hits.mean()                          # 0.0072 — 18 hits in 2500 days vs 25 expected
int(hits[:-1] @ hits[1:])            # 0 consecutive-hit pairs: no clustering
```

Eighteen hits against twenty-five expected looks like a shortfall, but the binomial standard deviation is $\sqrt{2500 \times 0.01 \times 0.99} \approx 5$, so the gap is well inside sampling noise — a Kupiec test would not reject, and that calibration judgment is exactly what the test formalizes. With a *fitted* model the same code runs on `fit["conditional_vol"]`, and a t or FHS quantile replaces the hardcoded normal one.

> **⚠ Common mistake.** Trusting asymptotic backtest p-values on short windows. A 1% VaR over 250 trading days — the regulatory standard — expects 2.5 violations; the $\chi^2$ approximation to the LR statistic is poor with counts that small, and the tests have little power regardless. Use exact binomial or Monte Carlo p-values (Dufour 2006), and treat a "pass" on one year of data as weak evidence, not validation. And check the sign convention *first*: backtesting the wrong tail produces beautiful-looking results that mean nothing.

## Measuring volatility: realized variance and HAR

Everything so far treats $\sigma_t^2$ as latent — inferred from daily returns through a model. High-frequency data changes the game: with intraday prices you can nearly *observe* daily variance. Sum squared 5-minute returns over the trading day and you get **realized variance**,

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2,$$

which converges (as sampling gets finer, in the absence of noise) to the day's true integrated variance — Andersen and Bollerslev (1998) used exactly this to show that GARCH forecasts were far better than the skeptics' $R^2$'s against daily squared returns suggested; the proxy was the problem, not the forecast. In practice microstructure noise — bid-ask bounce, discrete prices — contaminates very fine sampling, so practitioners stop at 5 minutes or use noise-robust estimators (realized kernels, pre-averaging; see the frontier section).

Once volatility is (approximately) observable, forecasting it becomes a regression problem. The benchmark is the **HAR** model — the heterogeneous autoregression of Corsi (2009) — which regresses tomorrow's RV on the average RV over the past day, week, and month:

$$RV_{t+1} = \beta_0 + \beta_d RV_t + \beta_w \overline{RV}_{t-4:t} + \beta_m \overline{RV}_{t-21:t} + u_{t+1}.$$

The three horizons proxy traders with daily, weekly, and monthly rebalancing frequencies — hence "heterogeneous" — and the overlapping averages give a parsimonious approximation to long memory. HAR is estimated by plain OLS, needs HAC standard errors (RV is persistent and its measurement error heteroskedastic), and remains the benchmark that fancier RV models must beat. It runs with tsecon's API today:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(1)
n = 1500
slow = np.zeros(n)                         # latent log-volatility: a slow
fast = np.zeros(n)                         # component plus a fast one
for t in range(1, n):
    slow[t] = 0.995 * slow[t - 1] + 0.05 * rng.standard_normal()
    fast[t] = 0.80 * fast[t - 1] + 0.25 * rng.standard_normal()
rv = np.exp(slow + fast) * rng.chisquare(48, n) / 48   # noisy daily RV

d = rv[21:-1]                                                  # daily lag
w = np.array([rv[t - 4:t + 1].mean() for t in range(21, n - 1)])   # weekly avg
m = np.array([rv[t - 21:t + 1].mean() for t in range(21, n - 1)])  # monthly avg
y = rv[22:]
X = np.column_stack([np.ones_like(d), d, w, m])

r = tsecon.ols(y, X, se_type="hac")        # persistent + heteroskedastic => HAC
r["params"]                                # [const, beta_d, beta_w, beta_m]
r["tvalues"]
```

Typical empirical estimates put meaningful weight on all three horizons — recent information matters most, but the monthly component is what gives RV forecasts their long memory. Extensions that dominate plain HAR out of sample (jump-separated HAR-J, semivariance-based SHAR, measurement-error-aware HARQ) are on the module roadmap, along with Realized GARCH (Hansen, Huang, and Shek 2012), which fuses the two worlds: a GARCH-type filter for returns *plus* a measurement equation linking RV to the latent variance.

> **⚠ Common mistake.** Comparing volatility forecasts with MAE, or with $R^2$ on standard deviations, against a noisy proxy like RV or daily squared returns. Patton (2011) showed that only certain loss functions — MSE and QLIKE among them — rank forecasts consistently when the evaluation target is a noisy proxy for the truth. With other losses, a *worse* forecast can systematically win. Use QLIKE or MSE, and pair the Diebold-Mariano comparison (Chapter on forecast evaluation; `tsecon.dm_test`) with HAC variance, since QLIKE differentials are heavy-tailed.

## The frontier

**Multivariate volatility.** Portfolios need conditional *covariance* matrices, and the workhorse is DCC — dynamic conditional correlation (Engle 2002): fit univariate GARCH to each asset, standardize, then drive a correlation matrix with a scalar GARCH-like recursion on the standardized residuals. It is the most-used multivariate volatility model in existence, and also a minefield the state of the art keeps repairing: the correlation-targeting estimator in original DCC is inconsistent (Aielli 2013 — his cDCC is the recommended fix), and the ubiquitous two-step standard errors that ignore first-stage estimation error are wrong (Engle and Sheppard 2001 give the correct stacked inference, essentially unavailable in shipped software). At the research edge, composite pairwise likelihoods (Pakel, Shephard, Sheppard, and Engle 2021) and nonlinear-shrinkage targeting (Engle, Ledoit, and Wolf 2019) push DCC to thousands of assets — the state of practice at quantitative funds, currently living only in author MATLAB code. All of this is the multivariate track of tsecon's Module 03.

**Stochastic volatility.** GARCH makes $\sigma_t^2$ a deterministic function of past data; **SV** models give volatility its own random innovations — a latent AR(1) in log-volatility. That one change makes the likelihood an intractable integral, estimated by MCMC via the mixture sampler of Kim, Shephard, and Chib (1998) with the Omori et al. (2007) refinement, or by particle filters for models with leverage and jumps. SV fits often beat GARCH in likelihood terms and are the natural building block inside macro models (time-varying-parameter VARs with SV are the modern standard in empirical macro — see the Bayesian chapter). The practical cost is computational, which is exactly the margin a parallel Rust MCMC core attacks.

**Volatility and the macroeconomy.** Daily GARCH dynamics say nothing about *why* the long-run level of volatility drifts across decades. Component models split the variance into a slow-moving trend and a mean-reverting cycle (Engle and Lee 1999), and GARCH-MIDAS (Engle, Ghysels, and Sohn 2013) goes further: the long-run component is driven directly by low-frequency macro variables — inflation, industrial-production growth — through MIDAS weights, making "does the business cycle drive market volatility?" an estimable question rather than a stylized claim. This branch is where volatility modeling meets the macro half of this guide.

**Score-driven models.** GAS models (Creal, Koopman, and Lucas 2013) update the variance by the scaled *score* of the conditional likelihood. With Student-t errors this yields Beta-t-EGARCH (Harvey 2013): a huge return is partially discounted as a fat-tail draw rather than fully fed into tomorrow's variance, giving robustness to outliers that plain GARCH lacks.

**Rough volatility.** Gatheral, Jaisson, and Rosenbaum (2018) argue log-RV behaves like fractional Brownian motion with Hurst exponent near 0.1 — far rougher than standard models imply — and that a simple forecasting rule exploiting this is strikingly accurate. It connects volatility econometrics to option-pricing models and is an active, contested research area (measurement noise in RV biases roughness estimates — how much of the roughness is real remains debated).

**Tail-risk evaluation.** ES is not *elicitable* on its own — no loss function is minimized in expectation by the true ES (Gneiting 2011) — but Fissler and Ziegel (2016) proved (VaR, ES) is *jointly* elicitable, enabling honest ES model comparison via joint scoring, and Patton, Ziegel, and Chen (2019) and Taylor (2019) built dynamic models that filter VaR and ES directly by minimizing those scores. This is the frontier of risk forecasting and is absent from all mainstream libraries; it anchors the roadmap's backtesting layer alongside the modern ES backtests (Acerbi and Szekely 2014; Nolde and Ziegel 2017; Bayer and Dimitriadis 2022).

Honest open problems: distinguishing genuine long memory from structural breaks in volatility (they mimic each other almost perfectly in-sample and imply different forecasts); persistence estimates biased toward one by unmodeled breaks; boundary inference when $\alpha \approx 0$; and the roughness-vs-noise identification problem in high-frequency data.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| Any regression residuals, before trusting SEs | `arch_lm`, `ljung_box` on squared residuals | Five minutes of testing tells you whether heteroskedasticity is even present |
| Baseline daily volatility forecast, one asset | GARCH(1,1) with Student-t, robust SEs | The benchmark 330 competitors could not reliably beat (Hansen-Lunde 2005) |
| Equity index or single stocks | GJR or EGARCH | The leverage effect is first-order for equities; sign-bias tests will confirm |
| Estimates keep hitting positivity constraints | EGARCH | Log-variance recursion needs no parameter constraints |
| Persistence estimate ≈ 1 and the sample spans crises | Component or Markov-switching GARCH (roadmap) | Unmodeled level breaks masquerade as unit-root persistence |
| Intraday data available | Realized variance + HAR; Realized GARCH (roadmap) | Measuring beats filtering; HAR is OLS-simple and hard to beat |
| Regulatory or desk-level VaR/ES | GARCH + t or FHS/EVT tails, then Kupiec + Christoffersen | Parametric normal tails underpredict 1% losses; unbacktested VaR is a number, not a model |
| Multi-day risk horizon | Sum the forecast variance term structure, or simulate | The square-root-of-time rule is wrong whenever volatility is mean-reverting |
| Portfolio covariances, a few to hundreds of assets | DCC/cDCC (roadmap) | Univariate GARCH margins + parsimonious correlation dynamics scale |
| Comparing two volatility forecasts | QLIKE or MSE loss + `dm_test` with HAC | Only proxy-robust losses rank correctly against noisy volatility proxies |

## What tsecon implements today

**Available now in Python** (`import tsecon`):

- `arch_lm(resid, nlags=4)` — Engle's ARCH-LM test (statsmodels `het_arch` convention); the pre-flight check for everything in this chapter
- `ljung_box(y, nlags=10)` — run it on returns *and* squared returns to see clustering
- `jarque_bera(x)` — skewness/kurtosis normality test for the fat-tails fact and for standardized-residual checks
- `acf(y, nlags=20)` — the autocorrelation function of squared returns is the clustering fingerprint
- `ols(y, X, se_type="hac")` — estimates HAR-RV models correctly today, as in the example above
- `dm_test(e1, e2, h=1, loss="squared")` and `accuracy(...)` — forecast-comparison machinery volatility horse races run on
- `bootstrap_indices`, `philox_uniforms` — the reproducible resampling/RNG substrate that filtered historical simulation will consume

**Built in Rust, awaiting Python bindings** (`crates/tsecon-garch`):

- GARCH(p, q) — including ARCH(p) as q = 0 — GJR-GARCH(p, o, q), and EGARCH(p, o, q), each with zero or constant mean and normal or standardized Student-t innovations
- QMLE via grid start + L-BFGS + Nelder-Mead polish, with `arch`-style backcast variance initialization
- Classical and Bollerslev-Wooldridge robust standard errors as first-class outputs
- Conditional-volatility paths, standardized residuals, information criteria, and analytic multi-step variance forecasts for GARCH/GJR (EGARCH multi-step awaits the simulation engine)
- Cross-package parity with the `arch` package pinned by golden fixtures (`fixtures/garch.json`): log-likelihoods to 1e-8 relative, conditional volatilities to 1e-6, robust SEs to 5e-3

**Roadmap** ([docs/roadmap/03-volatility.md](../roadmap/03-volatility.md)): the asymmetric and long-memory families (TGARCH, APARCH, FIGARCH), component and GARCH-MIDAS models, the skew-t/GED innovation zoo with exact partial moments, the VaR/ES layer with the full backtesting battery (Kupiec, Christoffersen, ES backtests, Fissler-Ziegel joint scoring), realized-measure construction and Realized GARCH/HEAVY, HAR extensions, DCC/cDCC/BEKK with correct two-step inference, stochastic volatility by MCMC and particle methods, and score-driven (GAS) models.

## Further reading

- **Engle (1982), Econometrica** — the ARCH paper: variance made conditional and estimable; the 2003 Nobel citation traces here.
- **Bollerslev (1986), Journal of Econometrics** — GARCH; three parameters that became the industry standard.
- **Bollerslev & Wooldridge (1992), Econometric Reviews** — QMLE theory and the sandwich standard errors every serious GARCH fit should report.
- **Nelson (1991), Econometrica** — EGARCH; asymmetry and constraint-free positivity via log-variance.
- **Glosten, Jagannathan & Runkle (1993), Journal of Finance** — the threshold asymmetry model most used in applied finance.
- **Christoffersen (1998), International Economic Review** — the conditional-coverage framework that made VaR backtesting a testable hypothesis.
- **Engle (2002), Journal of Business & Economic Statistics** — DCC; correlation dynamics at GARCH prices.
- **Corsi (2009), Journal of Financial Econometrics** — HAR: the pseudo-long-memory regression that is still the RV benchmark.
- **Francq & Zakoïan, *GARCH Models: Structure, Statistical Inference and Financial Applications* (2nd ed., Wiley)** — the rigorous graduate treatment of everything estimation-theoretic in this chapter.
- **McNeil, Frey & Embrechts, *Quantitative Risk Management* (rev. ed., Princeton)** — the standard reference for VaR, ES, EVT tails, and the risk-measure theory the frontier section touches.
