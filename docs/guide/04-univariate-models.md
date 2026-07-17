# Chapter 4 — Univariate Models: AR to State Space

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** the stationarity toolkit (ADF, KPSS, differencing) from the earlier chapters, plus OLS and maximum likelihood at first-econometrics-course level.

**You will learn:**

- how the AR → MA → ARMA → ARIMA → SARIMA ladder assembles one model family from two simple parts
- why conditional sum of squares (CSS) and exact maximum likelihood give different answers in small samples, and which to trust
- how AIC/BIC and auto-ARIMA choose model orders — and the classic ways automatic selection goes wrong
- what the state-space form and the Kalman filter actually do, and why missing data costs them nothing
- when to leave the linear world: regime switching, thresholds, and (at the frontier) long memory

## The idea

Suppose you need to forecast next quarter's GDP growth. You have exactly one thing: the history of GDP growth itself. No structural model, no other variables. Can you do anything useful?

Almost always yes, because economic time series have **momentum**. A quarter of strong growth tends to be followed by another decent quarter; a spike in inflation fades over several months rather than vanishing overnight. The whole univariate modeling enterprise is a machine for measuring that momentum and projecting it forward.

There are two natural ways to describe momentum, and they turn out to be the two atoms everything in this chapter is built from:

1. **The past of the series predicts its future.** "GDP growth was above average last quarter, so it will probably be above average this quarter, just less so." This is *autoregression* — literally, a regression of the series on itself.
2. **Past surprises linger.** "The oil shock three months ago is still working its way into prices." Here the series is a weighted memory of recent random shocks. This is a *moving average* of shocks.

Everything else is assembly. Combine the two and you get ARMA. Notice that some series (the *level* of GDP, prices, the money stock) drift without ever returning to a mean, so you model their *changes* instead — that is the "I" (integrated) in ARIMA. Notice that December looks like last December, and you get seasonal ARIMA. Notice that you'd rather think in terms of an unobserved "true" level being measured with noise — like trying to read the economy's underlying state through statistical fog — and you arrive at the state-space form and the Kalman filter, which quietly turns out to be the engine that estimates all the other models too.

Picture the simplest case as a thermostat-controlled room with a drafty window. The temperature (the series) keeps getting knocked around by gusts (shocks), but the thermostat pulls it back toward the set point (the mean). How hard it pulls is the autoregressive coefficient. If the thermostat is strong, disturbances die out quickly and the series hugs its mean; if it is weak, the room stays cold for a long time after each gust; if it is absent entirely, the temperature wanders wherever the gusts push it — a random walk, and the boundary of the stationary world. This chapter is about measuring the thermostat.

## Autoregression: momentum you can regress on

A practitioner cares about AR models because they are the cheapest credible answer to "how persistent is this series?" — the question behind inflation half-lives, output-gap dynamics, and nearly every benchmark forecast.

An **AR(p)** (autoregression of order p) says today's value is a linear function of the last $p$ values plus a fresh shock:

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t,
\qquad \varepsilon_t \sim \text{iid}(0, \sigma^2),
$$

where $c$ is an intercept, the $\phi_j$ are the autoregressive coefficients, and $\varepsilon_t$ (the **innovation**) is the part of $y_t$ that nothing in the past could have predicted.

**Stationarity** — the requirement that the series keeps returning to a fixed mean with stable variance — puts a condition on the coefficients. For an AR(1), $y_t = c + \phi y_{t-1} + \varepsilon_t$, the condition is simply $|\phi| < 1$: each period the deviation from the mean shrinks by the factor $\phi$. At $\phi = 1$ the thermostat is gone and you have a random walk. For general $p$, write the model with the lag polynomial $\phi(L) = 1 - \phi_1 L - \cdots - \phi_p L^p$ (where $L$ is the lag operator, $L y_t = y_{t-1}$); stationarity requires every root $z$ of $\phi(z) = 0$ to lie **outside the unit circle**. When it holds, the AR(1) mean is $\mu = c / (1 - \phi)$ — a formula worth memorizing, because the intercept of an AR model is *not* its mean.

The **Yule-Walker equations** are the moment-based route to estimation, and the intuition is worth having even if you always use maximum likelihood. Multiply the AR equation by $y_{t-k}$, take expectations, and divide by the variance: the autocorrelations $\rho_k$ must satisfy

$$
\rho_k = \phi_1 \rho_{k-1} + \phi_2 \rho_{k-2} + \cdots + \phi_p \rho_{k-p}, \qquad k \ge 1.
$$

The model's own recursion governs its correlation structure. Stack the first $p$ of these equations, plug in *sample* autocorrelations, and solve for the $\phi_j$: that is Yule-Walker estimation. It also explains the two identification fingerprints of an AR(p): the ACF decays geometrically (the recursion never lets it stop dead), while the partial autocorrelation function (PACF) — the correlation between $y_t$ and $y_{t-k}$ after controlling for everything in between — cuts off sharply after lag $p$.

You can see both fingerprints, and evaluate the exact likelihood, with today's API:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(42)
n = 400
phi1, phi2 = 1.3, -0.4          # a stationary AR(2)
y = np.zeros(n)
eps = rng.standard_normal(n)
for t in range(2, n):
    y[t] = phi1 * y[t-1] + phi2 * y[t-2] + eps[t]

r = tsecon.acf(y, nlags=24)      # dict: acf, bartlett_se
p = tsecon.pacf(y, nlags=24)     # Yule-Walker ("yw") or "ols"

# exact Gaussian log-likelihood at candidate parameters
tsecon.ar_loglik(y, [1.3, -0.4], sigma2=1.0)   # the truth
tsecon.ar_loglik(y, [0.9, 0.0], sigma2=1.0)    # a worse model: lower loglik

# a one-dimensional exact MLE by brute force, to see the machinery move.
# With phi2 = -0.4 fixed, stationarity requires phi1 + phi2 < 1, i.e.
# phi1 < 1.4 — the admissible region is a triangle, not a box, and
# ar_loglik refuses to evaluate outside it.
grid = np.linspace(1.0, 1.39, 79)
ll = [tsecon.ar_loglik(y, [g, -0.4], sigma2=1.0) for g in grid]
phi1_hat = grid[np.argmax(ll)]                 # lands near 1.3
```

![ACF and PACF of an AR(2)](../examples/img/01-acf-pacf.png)

The figure is this exact process: smooth ACF decay, two PACF spikes, then nothing — you read "AR(2)" straight off the plot. `ar_loglik` is not a toy: it evaluates the *exact* Gaussian likelihood through the state-space form with a stationary initial distribution, the same kernel a full ARIMA fit maximizes (more on this below).

> **⚠ Common mistake.** Fitting an AR model to a series you have not checked for stationarity. If the true process has a unit root, the fitted $\phi$ will hug 1, forecasts will be garbage at long horizons, and standard errors are invalid. Run `tsecon.check_stationarity(y)` first. A subtler trap: Yule-Walker estimates are mechanically biased *toward* stationarity near the unit circle (R's `ar()` default suffers from this), so near-unit-root persistence is systematically understated — one reason exact MLE is the default worth having.

## Moving averages and the ARMA compromise

Why would anyone model a series as a function of unobservable shocks? Because some economic quantities genuinely have **finite memory of surprises**. A one-off measurement quirk in this month's CPI affects the next release or two of the *inflation rate* and then is gone completely. An AR model cannot produce that pattern — its shocks decay geometrically but never fully vanish.

An **MA(q)** (moving average of order q) writes the series as a weighted sum of the last $q+1$ innovations:

$$
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}.
$$

An MA(q) is *always* stationary (it is a finite sum of white noise), and its ACF is exactly zero beyond lag $q$ — the mirror image of the AR fingerprint. The condition that matters instead is **invertibility**: the roots of $\theta(z) = 1 + \theta_1 z + \cdots + \theta_q z^q$ must lie outside the unit circle. The reason is identification, not stability: an MA(1) with coefficient $\theta$ and one with $1/\theta$ produce *identical* autocorrelations, so the data cannot tell them apart. The invertible representation is the one in which the shocks can be recovered from the observed history — the one where $\varepsilon_t$ means what you think it means.

Why combine the two? **Parsimony.** The Wold decomposition theorem (Wold, 1938) says every covariance-stationary process is an MA($\infty$) — an infinite weighted sum of past shocks. You cannot estimate infinitely many weights, but a ratio of two short polynomials can *approximate* a long one remarkably well, exactly as a rational function approximates a complicated curve better than a polynomial of the same total degree. That is the **ARMA(p,q)**:

$$
\phi(L)\, y_t = c + \theta(L)\, \varepsilon_t .
$$

An ARMA(1,1) with two dynamic parameters often fits what would take an AR(5) or an MA(8) — fewer parameters, tighter estimates, better forecasts. This trade is the heart of the Box-Jenkins methodology (Box and Jenkins, 1970): identify a *small* model from the ACF/PACF, estimate it, check the residuals, and only grow the model if the residuals demand it.

The residual check is available today, and it has a degrees-of-freedom subtlety worth internalizing now:

```python
# fit an ARMA(2,1) to the AR(2) series from the first block, then ask
# whether its one-step residuals still carry structure
fit   = tsecon.arima_fit(y, p=2, d=0, q=1, constant=True)
resid = fit["residuals"]
lb = tsecon.ljung_box(resid, nlags=10)   # dict: lags, lb_stat, lb_pvalue, ...
```

> **⚠ Common mistake.** Two, actually. First: the Ljung-Box test on *model residuals* must have its degrees of freedom reduced by the number of fitted ARMA parameters — comparing the statistic at lag 10 to a $\chi^2_{10}$ after fitting an ARMA(2,1) overstates the p-value; use $\chi^2_{10-3}$. Second: overfitting ARMA orders creates **near-canceling roots** — an ARMA(2,2) fit to ARMA(1,1) data puts an AR root nearly on top of an MA root, the likelihood goes flat along the cancellation direction, and the optimizer returns fragile nonsense with huge standard errors. When you see wildly offsetting AR and MA coefficients, shrink the model.

## ARIMA: integration is just differencing

Most macroeconomic *levels* — real GDP, the price level, the money stock — never return to any fixed mean. They are **integrated**: only after differencing do they become stationary. A series is I(d) (integrated of order $d$) if it must be differenced $d$ times; d = 1 covers nearly everything in practice, d = 2 occasionally fits price levels whose *inflation rate* also drifts.

**ARIMA(p,d,q)** is nothing more than "difference $d$ times, then fit an ARMA(p,q)":

$$
\phi(L)\, (1-L)^d\, y_t = c + \theta(L)\, \varepsilon_t ,
$$

where $(1-L) y_t = y_t - y_{t-1}$ is the first difference. Choosing $d$ is the stationarity workflow you already know — and it is a decision you make with unit-root tests *before* any model comparison, not one you let an information criterion make:

```python
rep = tsecon.check_stationarity(y)
rep["recommendation"]    # "Proceed" | "Difference" | "Detrend"
```

Seasonality gets the same treatment at the seasonal lag. **SARIMA(p,d,q)(P,D,Q)$_s$** multiplies a second set of polynomials in $L^s$ (with $s = 12$ for monthly data, 4 for quarterly) onto the first:

$$
\phi(L)\,\Phi(L^s)\,(1-L)^d\,(1-L^s)^D\, y_t = \theta(L)\,\Theta(L^s)\,\varepsilon_t .
$$

The story that made this famous is the **airline model**. Box and Jenkins took the monthly count of international airline passengers, 1949–1960 — a series with explosive trend growth and a seasonal swing that widens every year. Logs stabilize the widening; one regular difference removes the trend; one seasonal difference removes the stable seasonal pattern; and what remains is captured by just two MA parameters. The result, ARIMA(0,1,1)(0,1,1)$_{12}$ on the logged series, became the default model for seasonal economic data for decades, and its parameter estimates are the canonical cross-package validation target — tsecon's ARIMA crate validates against R's `arima()` on exactly this model.

The non-seasonal ARIMA engine — exact-MLE ARIMA(p,d,q) with correctly integrated-back forecast intervals — ships in Python today. Here it is on a synthetic monthly series with the airline model's shape (a trend plus a seasonal swing), fit in logs after one regular difference:

```python
rng = np.random.default_rng(11)
n = 144                                   # twelve years of monthly data
t = np.arange(n)
season = np.array([0, 6, 14, 8, 3, -2, -7, -5, 1, 4, 9, 15])[t % 12]
air = np.exp(4.6 + 0.010 * t + 0.02 * season + 0.03 * rng.standard_normal(n))

m = tsecon.arima_fit(np.log(air), p=0, d=1, q=1,
                     forecast_steps=24, conf_alpha=0.05)
m["forecast_mean"]                        # 24 log forecasts (integrate back by exp)
m["forecast_lower"], m["forecast_upper"]  # intervals on the log scale
```

The *seasonal* layer — a second set of polynomials at the seasonal lag, the airline model proper — is still on the roadmap:

> **Preview** — the seasonal `(P, D, Q, s)` order is on the [Module 02 roadmap](../roadmap/02-univariate.md); the call below shows the intended API, a seasonal argument added to the `arima_fit` that ships today.

```python
m = tsecon.arima_fit(np.log(air), p=0, d=1, q=1, seasonal=(0, 1, 1, 12),
                     forecast_steps=24, conf_alpha=0.05)
m["forecast_mean"]        # points + intervals that integrate back correctly
```

(The underlying engine — exact-MLE and CSS estimation of ARIMA(p,d,q) with forecast intervals — is implemented and tested in the `tsecon-arima` crate and now wired into Python as `arima_fit`; the seasonal layer is what remains.)

> **⚠ Common mistake.** Overdifferencing. If you difference a series that was already stationary, you *inject* an MA unit root: the differenced series has $\theta = -1$, the likelihood piles up on the invertibility boundary, and estimation becomes unstable. The symptom is a first-lag autocorrelation near $-0.5$ in the differenced series. Related trap: never difference through missing values — a difference across a gap is not a one-period change. Fit in levels via the state-space form instead (below), which handles gaps exactly.

## Estimation: CSS versus exact maximum likelihood

Every serious package offers two ways to estimate an ARMA model, and they do not agree. Knowing why saves you from the single most common "your package disagrees with R" confusion.

**Conditional sum of squares (CSS).** Fix the first $p$ observations as given, set the pre-sample innovations to zero, and run the model's recursion forward to compute one-step-ahead residuals $\hat\varepsilon_t$. Minimize

$$
S(\phi, \theta) = \sum_t \hat\varepsilon_t^2 .
$$

This is fast (no matrix algebra beyond the recursion) and asymptotically equivalent to MLE. But it *conditions on* the initial observations rather than modeling them — it throws away the information in how the series started.

**Exact maximum likelihood.** Treat the first observations as random draws from the process's own stationary distribution, and compute the full joint Gaussian likelihood of all $T$ observations. The practical route is the **prediction-error decomposition**: put the ARMA model in state-space form (next-to-last section), run the Kalman filter, and the likelihood factors into a product of one-step-ahead forecast densities,

$$
\log L = -\frac{1}{2} \sum_{t=1}^{T} \left( \log 2\pi F_t + \frac{v_t^2}{F_t} \right),
$$

where $v_t$ is the one-step forecast error and $F_t$ its variance — both produced by the filter. This is exactly what `tsecon.ar_loglik` computes, which is why the grid search in the AR section was a genuine (if brute-force) exact MLE.

When does the difference matter? **Small samples and near the unit circle.** With $T = 50$ quarterly observations, the $p$ initial values CSS discards are a nontrivial fraction of the data. And near $\phi = 1$ or $\theta = -1$, the stationary distribution of the initial conditions carries real information that CSS ignores; the two estimators can disagree in the second decimal place, which is enough to flip a model comparison. The industrial-strength recipe — the one tsecon's ARIMA engine implements — is: generate starting values with the Hannan-Rissanen (1982) long-AR regression, optionally refine with a cheap CSS pass, then maximize the exact likelihood, with the parameter space kept stationary-and-invertible throughout via the Monahan (1984) reparameterization (the admissible region for $p, q > 1$ is *not* a box, so naive coefficient bounds fail).

```python
# CSS-vs-exact intuition you can verify today: the exact likelihood
# "prices in" the initial conditions. Compare a persistent and a
# non-persistent model on a SHORT stretch of persistent data.
y_short = y[:60]
tsecon.ar_loglik(y_short, [0.95], sigma2=1.0)   # near-unit-root candidate
tsecon.ar_loglik(y_short, [0.50], sigma2=1.0)   # moderate candidate
# The gap between these two numbers includes the evidence in y[0]'s size —
# a large first observation is itself evidence of high persistence.
# CSS, conditioning on y[0], never sees that evidence.
```

> **⚠ Common mistake.** Comparing log-likelihoods (or AICs) across packages without matching conventions. Packages differ silently in whether the Gaussian constant is included, whether diffuse initial terms are excluded, and whether the variance is concentrated out. A likelihood that differs from R's by exactly $\frac{T}{2}\log 2\pi$ is not a bug — it is a convention. tsecon documents its conventions per model precisely so results reconcile to the digit.

## Choosing the order: information criteria and auto-ARIMA

With the ladder built, which rung do you stand on — ARMA(1,1)? AR(3)? The likelihood alone cannot decide: a bigger model always fits at least as well. **Information criteria** charge a complexity toll:

$$
\text{AIC} = -2\log L + 2k, \qquad
\text{BIC} = -2\log L + k \log T, \qquad
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{T-k-1},
$$

where $k$ counts estimated parameters. Lower is better. BIC's toll grows with $T$, so it selects smaller models and recovers the true order asymptotically if one exists; AIC aims instead at forecast accuracy and tolerates mild overfitting. AICc (Hurvich and Tsai, 1989) is AIC with a small-sample correction and is the right default for the sample sizes macroeconomists actually have.

**Auto-ARIMA** — the Hyndman-Khandakar (2008) algorithm behind R's `forecast::auto.arima`, and the single most used function in applied forecasting — automates the whole Box-Jenkins loop with one crucial piece of discipline: it chooses the differencing orders $d$ and $D$ *first*, using unit-root and seasonal-strength tests, and only then runs a stepwise AICc search over $(p, q, P, Q)$ within fixed differencing. The order of operations is not a detail. Differencing changes the data, and **information criteria are not comparable across different $d$** — an AIC computed on levels and one computed on differences are likelihoods of different datasets.

The dangers of the automatic philosophy deserve equal billing:

- **Automatic is not correct.** Auto-ARIMA finds the best *ARIMA* description. If the series has a structural break, the search compensates with spurious persistence (a break masquerades as a near-unit root); if it has outliers, it compensates with spurious MA terms. The algorithm never tells you the family was wrong.
- **Selection uncertainty vanishes from the output.** The standard errors of the chosen model pretend the specification was known in advance. After a data-driven search, they are too small.
- **The residual check is still your job.** A selected model whose residuals fail `ljung_box` (with the df correction) or show ARCH effects (`tsecon.arch_lm`) is a rejected model, whatever its AICc rank.

> **⚠ Common mistake.** Ranking models with different differencing orders by AIC. Select $d$ with `check_stationarity` (or KPSS sequences, as Hyndman-Khandakar do), then compare information criteria only among models sharing that $d$.

## Exponential smoothing: a family, not a hack

Exponential smoothing has a reputation problem: it looks like a rule of thumb. It is in fact a fully respectable model family — and in forecast competitions it has repeatedly embarrassed more elaborate rivals.

**Simple exponential smoothing (SES)** maintains a level estimate $\ell_t$ as an exponentially weighted average of the past:

$$
\ell_t = \alpha\, y_t + (1-\alpha)\, \ell_{t-1}, \qquad \hat y_{t+h|t} = \ell_t ,
$$

with smoothing weight $\alpha \in (0,1]$: high $\alpha$ chases the data, low $\alpha$ smooths hard. Holt's method adds a smoothed trend; Holt-Winters adds a smoothed seasonal, in additive or multiplicative form. For half a century these were "methods" without models — until Hyndman, Koehler, Snyder and Grose (2002) embedded all of them in an innovations **state-space taxonomy**, ETS(Error, Trend, Seasonal), with each component additive (A), multiplicative (M), damped (Ad), or absent (N). ETS(A,N,N) is SES; ETS(A,A,N) is Holt; ETS(M,A,M) is the multiplicative Holt-Winters most retail data want. The state-space grounding buys exactly what "methods" lacked: a likelihood (so AICc can select among all 30 taxonomy members automatically), proper prediction intervals, and simulation.

Two facts justify the family's competition record. SES is the *optimal* forecast for an ARIMA(0,1,1) process — so whenever a series is well described by "random walk plus transient noise," a two-parameter smoother is not a heuristic but the exact right answer. And the **Theta method** of Assimakopoulos and Nikolopoulos (2000), which won the 3,003-series M3 competition outright, was shown by Hyndman and Billah (2003) to be equivalent to SES with drift. That M3 winner is in tsecon today:

```python
rng = np.random.default_rng(3)
n, h = 140, 20                       # quarterly: trend + seasonal + AR noise
t = np.arange(n + h)
season = 4.0 * np.array([1.0, -0.4, 0.6, -1.2])[t % 4]
noise = np.zeros(n + h)
e = rng.standard_normal(n + h)
for i in range(1, n + h):
    noise[i] = 0.6 * noise[i-1] + 1.5 * e[i]
y = 50 + 0.3 * t + season + noise
train, test = y[:n], y[n:]

fc     = tsecon.theta_forecast(train, steps=h, period=4)   # the M3 benchmark
snaive = np.tile(train[-4:], h // 4 + 1)[:h]               # repeat last cycle

tsecon.accuracy(test, fc,     insample=train, period=4)["mase"]   # ~ 0.76
tsecon.accuracy(test, snaive, insample=train, period=4)["mase"]   # ~ 1.44
tsecon.dm_test(test - snaive, test - fc, h=1, loss="squared")["p_value"]  # 0.001
```

Theta halves the seasonal naive's error here, and the Diebold-Mariano test (covered in the forecast-evaluation chapter) confirms the gap is statistically real — never claim superiority without a benchmark and a test.

> **⚠ Common mistake.** Multiplicative-error or multiplicative-seasonal ETS models are undefined for zero or negative data — a series of net flows or growth rates cannot go through ETS(M,·,·). And the admissible parameter region of the ETS taxonomy is *larger* than the intuitive $[0,1]$ box (it is defined by eigenvalue stability conditions), so a fitted $\alpha = 1.3$ from a correct implementation is not necessarily an error — but a package that silently clips to the box is one.

## The state-space form and the Kalman filter

This is the chapter's load-bearing section. The state-space form is simultaneously (a) a model family in its own right, (b) the estimation engine behind exact-MLE ARIMA and all of ETS, and (c) the cleanest solution to missing data in existence.

The idea: separate **what is true** from **what you measure**. An unobserved state $\alpha_t$ evolves over time; you observe a noisy function of it:

$$
\begin{aligned}
y_t &= Z\, \alpha_t + \varepsilon_t, \qquad &\varepsilon_t &\sim N(0, H) \quad &\text{(measurement)}\\
\alpha_{t+1} &= T\, \alpha_t + \eta_t, \qquad &\eta_t &\sim N(0, Q) \quad &\text{(state transition)}
\end{aligned}
$$

The "hello world" is the **local level model**: the state is a single number $\mu_t$ — the economy's underlying level — following a random walk, observed with noise:

$$
y_t = \mu_t + \varepsilon_t, \qquad \mu_{t+1} = \mu_t + \eta_t .
$$

Everything is governed by the **signal-to-noise ratio** $q = \sigma^2_\eta / \sigma^2_\varepsilon$: how fast the truth moves relative to how badly you measure it. Add a second state for a slowly evolving slope and you have the **local linear trend** model — Harvey's (1989) structural-model building blocks, from which trend, cycle, and seasonal components are assembled like Lego.

The **Kalman filter** (Kalman, 1960) processes the data one observation at a time, alternating two steps:

- **Predict.** Push your current best guess of the state through the transition equation: "the level tomorrow is the level today," with uncertainty grown by $\sigma^2_\eta$.
- **Update.** See $y_t$, compute the surprise $v_t = y_t - \text{(predicted } y_t)$, and shift your state estimate toward the observation by a fraction $K_t$ of the surprise — the **Kalman gain**. For the local level model, $K_t = P_t / (P_t + \sigma^2_\varepsilon)$, where $P_t$ is your current state uncertainty. The formula *is* the intuition: when you are uncertain and the measurement is clean, $K_t \to 1$ and you trust the data; when you are confident and the measurement is noisy, $K_t \to 0$ and you trust yourself. The filter is a continuously self-adjusting weighted average — Bayes' rule applied sequentially.

Three consequences fall out for free:

1. **Exact likelihood.** The surprises $v_t$ and their variances $F_t$ plug into the prediction-error decomposition — this is how ARMA exact MLE actually gets computed once the ARMA model is written in companion (state-space) form.
2. **Missing data is trivial.** No observation at time $t$? *Skip the update step.* Predict as usual, learn nothing, let uncertainty grow, continue. The likelihood simply has fewer terms. No imputation, no interpolation, no differencing through gaps.
3. **Smoothing.** The filter uses data up to $t$; the **smoother** runs a backward pass to compute $E[\alpha_t \mid \text{all } T \text{ observations}]$ — the best retrospective estimate, which is what you want for historical trend extraction.

All of this is live in tsecon today, missing data included:

```python
rng = np.random.default_rng(7)
n = 200
level = np.cumsum(rng.normal(0.0, np.sqrt(0.5), n))    # random-walk truth
y = level + rng.normal(0.0, np.sqrt(4.8), n)           # noisy measurements
y[80:105] = np.nan                                     # a 25-period data gap

r = tsecon.local_level_smooth(y, sigma2_eps=4.8, sigma2_eta=0.5)
r["loglik"]               # exact likelihood, counting only observed points
r["smoothed_state"]       # E[level_t | all data] — the gap bridged automatically
band = 1.96 * np.sqrt(r["smoothed_state_var"])   # honest: wider inside the gap
```

![Kalman smoother bridging a 25-period gap](../examples/img/05-kalman.png)

The figure shows exactly what the theory promises: the smoother bridges the gap with a sensible path, the 95% band balloons precisely where information is missing, and the true latent path — which the model never saw inside the gap — stays within the band. Estimating the two variances (rather than fixing them, as here) is one numerical optimization of the exact likelihood away; that fitted local-level API, validated against the canonical Nile-river results of Durbin and Koopman (2012), is part of Module 02.

> **⚠ Common mistake.** Handling a missing observation by treating it as "observed with value 0" or by zero-weighting it. Both are wrong: the correct treatment is to *skip the measurement update entirely* and count only observed terms in the likelihood. A related implementation trap: initializing a nonstationary state (a random-walk level has no stationary distribution) with a "big number" variance approximation instead of the exact diffuse initialization — the likelihood constants come out wrong and cross-package comparisons silently break. tsecon uses exact diffuse initialization throughout.

## When one line is not enough: regimes and thresholds

Everything so far assumes one set of dynamics governs the whole sample. Macroeconomic reality objects: recessions are not expansions run backward. US GDP falls fast and recovers slowly; unemployment rises in spikes and declines in long glides. Linear ARMA models — whose impulse responses are symmetric by construction — cannot represent this. Two families relax the assumption, distinguished by *what triggers the switch*.

**Markov-switching AR** (Hamilton, 1989) makes the trigger an unobserved discrete state $s_t \in \{0, 1\}$ following a Markov chain:

$$
y_t = \mu_{s_t} + \phi_1 (y_{t-1} - \mu_{s_{t-1}}) + \cdots + \varepsilon_t,
\qquad P(s_t = j \mid s_{t-1} = i) = p_{ij}.
$$

Each regime has its own mean (and possibly variance and dynamics); the economy switches stochastically between them, and the model infers *from the data alone* the probability of being in each regime at each date. Hamilton's original application fit two regimes to postwar US GNP growth and found mean growth of roughly $1.2\%$ per quarter in one regime and $-0.4\%$ in the other, with both regimes persistent — and the smoothed probability of the low regime reproduced the NBER's recession dates almost exactly, without ever being shown them. That figure is one of the most famous in time series econometrics, and reproducing Hamilton's published estimates is the validation gate for tsecon's implementation.

**Threshold AR (SETAR)** (Tong and Lim, 1980) makes the trigger observable: the model switches when a lagged value of the series itself crosses a threshold $\tau$:

$$
y_t =
\begin{cases}
c_1 + \phi_1 y_{t-1} + \varepsilon_t & \text{if } y_{t-d} \le \tau \\
c_2 + \phi_2 y_{t-1} + \varepsilon_t & \text{if } y_{t-d} > \tau ,
\end{cases}
$$

("self-exciting" because the series triggers its own switches). **STAR** models (Teräsvirta, 1994) replace the hard switch with a smooth transition function, so the economy blends between regimes rather than jumping — natural for aggregates that sum many units crossing thresholds at different times, like real exchange rates under transaction costs.

When do you actually need these?

- **Reach for Markov-switching** when the regime is a latent classification you want the model to discover — business-cycle phases, high/low-volatility eras, policy regimes. The regime probabilities themselves are often the deliverable.
- **Reach for SETAR/STAR** when you can name the observable trigger — "dynamics change when the spread goes negative," "mean reversion kicks in when the real exchange rate deviates enough." The threshold estimate is interpretable in economic units.
- **Stay linear** when you have short samples or forecasting is the only goal. Nonlinear models need many visits to *each* regime to estimate it; with 150 quarterly observations and two recessions, the recession regime rests on a handful of data points, and the forecast gains over a good ARMA are often negligible.

These families are estimation minefields — multimodal likelihoods requiring multistart optimization, unbounded likelihoods as a regime variance heads to zero, off-by-one traps in the regime smoother — which is why the roadmap treats "reliability where statsmodels is fragile" as the feature.

> **⚠ Common mistake.** Testing "linear vs. threshold" (or "one regime vs. two") with a standard likelihood-ratio test and $\chi^2$ critical values. Under the null of linearity the threshold $\tau$ (or the transition matrix) is *unidentified* — the Davies problem — and the LR statistic is not $\chi^2$. Valid inference needs sup-type tests with simulated or bootstrapped critical values (Hansen, 1996). Every naive $\chi^2$ p-value in this territory overstates the evidence for nonlinearity.

## The frontier

**Long memory and ARFIMA.** The ladder's integer choice — $d = 0$ (shocks fade geometrically) or $d = 1$ (shocks last forever) — is coarser than some data. Realized volatility, inflation, and interest-rate spreads show autocorrelations that decay *hyperbolically*: far too slowly for any stationary ARMA, yet clearly reverting rather than random-walking. Fractional integration (Granger and Joyeux, 1980; Hosking, 1981) fills the gap by letting $d$ be fractional in $(1-L)^d$, defined through its binomial expansion; for $0 < d < 0.5$ the ARFIMA(p,d,q) process is stationary with autocorrelations decaying like $k^{2d-1}$ — **long memory**. Estimation splits into semiparametric approaches that use only low frequencies — the GPH log-periodogram regression (Geweke and Porter-Hudak, 1983), local Whittle (Robinson, 1995), and exact local Whittle, which is valid for nonstationary $d$ (Shimotsu and Phillips, 2005) — and full MLE via Sowell's (1992) exact autocovariances, numerically delicate as $d \to 0.5$. The genuinely open problem: **long memory and structural breaks are nearly observationally equivalent** (Diebold and Inoue, 2001) — a short-memory process with occasional mean shifts produces every classic long-memory signature — and distinguishing them on realistic sample sizes remains unsolved in general.

**Elsewhere on the research edge.** Score-driven (GAS/DCS) models (Creal, Koopman and Lucas, 2013) make parameters time-varying through the score of the likelihood, giving robust filters that automatically discount outliers. Bayesian TVP models with global-local shrinkage priors (Bitto and Frühwirth-Schnatter, 2019) let the data decide *which* coefficients drift — the current standard in empirical macro. Testing for the *number* of Markov regimes is finally practical via Carrasco, Hu and Ploberger (2014). Mixed causal-noncausal AR models (Lanne and Saikkonen, 2011; Gouriéroux and Zakoïan, 2017) use roots *inside* the unit circle, identified through non-Gaussianity, to capture bubble episodes that explode and collapse. And an ecosystem-wide embarrassment remains open: default prediction intervals nearly everywhere ignore parameter and selection uncertainty and are systematically too narrow — dramatically so near unit roots and for $T < 100$; bootstrap and conformal methods are the frontier fixes.

The [Module 02 roadmap](../roadmap/02-univariate.md) covers this terrain in tiers: the full ARMA/SARIMA/regARIMA stack, ETS taxonomy, and auto-ARIMA as core; SETAR/STAR, Markov-switching, structural breaks (Bai-Perron), and the GPH/HAR long-memory entry points as standard; Sowell ARFIMA, ELW, score-driven models, and shrinkage TVP as advanced and frontier tiers — each gated on reproducing published numbers (Hamilton's GNP estimates, the airline model, the Nile variances) rather than matching another package's defaults.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| Short-horizon forecast of a stationary series, no covariates | AR(p), order by AICc | Cheapest credible momentum model; hard to beat at h = 1–4 |
| ACF cuts off sharply; shocks visibly transient | MA(q) or ARMA(1,1) | Finite shock memory is what MA terms are for |
| Trending level series (GDP, prices) | ARIMA with d chosen by `check_stationarity` | Difference first; unit-root tests, not AIC, pick d |
| Monthly/quarterly data with stable seasonality | SARIMA — start at the airline model (0,1,1)(0,1,1)ₛ | Four parameters cover a remarkable share of seasonal economic series |
| Many series to forecast automatically | auto-ARIMA or AutoETS, then residual checks | Disciplined search beats hand-tuning at scale — but never skip diagnostics |
| Trend + seasonal forecasting, fast and robust | ETS / Theta (`theta_forecast`) | M3-competition-grade accuracy at trivial cost |
| Noisy measurements of an underlying level; gaps in the data | Local level/trend via `local_level_smooth` | Kalman filter handles missing data exactly, with honest uncertainty |
| Small sample, persistence near a unit root | Exact MLE, never CSS or Yule-Walker | Initial conditions carry real information; moment methods bias toward stationarity |
| Asymmetric dynamics with a latent phase (recessions) | Markov-switching AR | Infers regime probabilities from the data; the probabilities are the deliverable |
| Dynamics change at a nameable observable trigger | SETAR/STAR | Interpretable threshold in economic units; test linearity properly first |
| Autocorrelations decay too slowly for ARMA but the series reverts | ARFIMA / local Whittle estimate of d | Fractional d is the honest middle ground between I(0) and I(1) — but rule out breaks |

## What tsecon implements today

**Available now in Python** (`import tsecon`):

- `ar_loglik(y, coeffs, sigma2, intercept=0.0)` — exact Gaussian AR(p) log-likelihood via the state-space form with stationary initialization; the exact-MLE kernel this chapter's estimation section is built on
- `local_level_smooth(y, sigma2_eps, sigma2_eta)` — exact-diffuse Kalman filter and smoother for the local level model; NaNs handled natively as missing data
- `arima_fit(y, p, d, q, constant=False, forecast_steps=0, conf_alpha=None)` — exact-MLE ARIMA(p,d,q): params, log-likelihood, AIC/BIC, residuals, and multi-step forecasts with correctly integrated-back intervals
- Identification and diagnostics used throughout the chapter: `acf`, `pacf`, `ljung_box`, `jarque_bera`, `arch_lm`
- Differencing decisions: `adf`, `kpss`, `check_stationarity`
- The exponential-smoothing family's benchmark: `theta_forecast`, with `accuracy` and `dm_test` for honest evaluation

**Built in Rust, partly awaiting Python bindings** (`tsecon-arima` crate):

- The non-seasonal ARIMA(p,d,q) engine ships in Python as `arima_fit` above (exact MLE, log-likelihood, residuals, integrated-back forecast intervals). Still Rust-only and on the roadmap for Python: the CSS estimator (`fit_css`) and the seasonal SARIMA layer

**Roadmap** ([docs/roadmap/02-univariate.md](../roadmap/02-univariate.md)):

- SARIMA and regression with ARMA errors (regARIMA); Hannan-Rissanen starts and the Monahan reparameterization as public API
- The full ETS taxonomy with AutoETS selection; auto-ARIMA (Hyndman-Khandakar)
- Fitted unobserved-components models (local level/trend with estimated variances, cycles, stochastic seasonals), validated on the Nile and UK-seatbelt canon
- Markov-switching AR validated against Hamilton (1989); SETAR/STAR with proper sup-test linearity inference; Bai-Perron structural breaks
- ARFIMA (Sowell exact MLE), GPH and exact local Whittle estimators of d

## Further reading

- **Box, G. E. P. and G. M. Jenkins (1970), *Time Series Analysis: Forecasting and Control*, Holden-Day.** The book that created the ARIMA methodology, the airline model, and the identify-estimate-check loop this chapter walks.
- **Wold, H. (1938), *A Study in the Analysis of Stationary Time Series*, Almqvist & Wiksell.** The decomposition theorem that explains why ARMA models can approximate any stationary process — the license for the whole family.
- **Kalman, R. E. (1960), "A New Approach to Linear Filtering and Prediction Problems," *Journal of Basic Engineering*.** The filter, from outside economics entirely; sixty-five years later it is the estimation engine of this whole module.
- **Hamilton, J. D. (1989), "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle," *Econometrica*.** Markov-switching AR and the recession-probability plot that launched a literature.
- **Tong, H. and K. S. Lim (1980), "Threshold Autoregression, Limit Cycles and Cyclical Data," *Journal of the Royal Statistical Society B*.** The founding SETAR paper; nonlinear time series as a practical toolkit begins here.
- **Teräsvirta, T. (1994), "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models," *Journal of the American Statistical Association*.** The complete STAR modeling cycle — test, select, estimate — still the standard workflow.
- **Granger, C. W. J. and R. Joyeux (1980), "An Introduction to Long-Memory Time Series Models and Fractional Differencing," *Journal of Time Series Analysis*.** Fractional integration's founding paper (with Hosking, 1981, *Biometrika*, its independent twin).
- **Hyndman, R. J., A. B. Koehler, R. D. Snyder and S. Grose (2002), "A State Space Framework for Automatic Forecasting Using Exponential Smoothing Methods," *International Journal of Forecasting*.** The paper that turned exponential smoothing from methods into models.
- **Hyndman, R. J. and Y. Khandakar (2008), "Automatic Time Series Forecasting: The forecast Package for R," *Journal of Statistical Software*.** The auto-ARIMA and AutoETS algorithms every automatic-forecasting system now descends from.
- **Durbin, J. and S. J. Koopman (2012), *Time Series Analysis by State Space Methods*, 2nd ed., Oxford University Press.** The definitive state-space text: exact diffuse initialization, smoothing, missing data — the conventions tsecon implements.
