# Chapter 12 — Machine Learning for Time Series

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** OLS regression, the bias-variance idea, and the forecasting-evaluation toolkit (MASE, the Diebold-Mariano test) from the previous chapters.

**You will learn:**

- Why the standard machine-learning playbook — random K-fold cross-validation — produces fraudulent accuracy numbers on serial data, and how purging, embargoes, and rolling-origin evaluation fix it
- How ridge, LASSO, and elastic net trade a little bias for a lot of variance, and how they connect to the Bayesian shrinkage of Chapter 10
- Why factor models were machine learning before the name existed, and why they remain the benchmark to beat
- What the honest evidence says about trees, boosting, neural networks, and foundation models in macroeconomic forecasting
- Why you should almost never believe the story told by which variables LASSO selects

## The idea

Suppose you are asked to forecast US inflation twelve months out. You have the FRED-MD panel: about 130 monthly macroeconomic series — production, employment, housing starts, interest rates, money aggregates, exchange rates — going back to 1959. Classical practice says: pick a small model, maybe an AR(4) or a Phillips-curve regression, and ignore the other 125 series. That feels wasteful. Machine learning says: use everything, and let the algorithm decide what matters.

That instinct is often right, and this chapter is about making it work. But the first thing that breaks when you point ML at economic data is not the model — it is the **scorecard**. The standard ML recipe for measuring accuracy shuffles the data into random chunks, trains on most of them, and tests on the rest. With cross-sectional data (say, house prices) this is fine, because the observations do not know about each other. A time series is different: it *remembers*. This month's industrial production is nearly a copy of last month's. If March 2008 is in your test set while February and April 2008 sit in the training set, the model has effectively been shown the answer before the exam. It will score brilliantly and forecast terribly.

Picture two report cards for the same model. One is graded by a teacher who shuffles the exam questions in among the practice problems — the student has seen the neighbors of every question. The other is graded the only way the real world grades a forecaster: stand at a date, use only what was known then, predict forward, move the date, repeat. The first report card is flattering fiction. The second is the truth. Everything else in this chapter — penalized regressions, factor models, forests, neural networks — only means something once you insist on the second report card. So we start there, with the fraud, before we get to the models.

## Leakage comes first

**Leakage** is any path by which information from the evaluation period reaches the model during training. It is the cardinal sin of applied ML, and time series data makes it shockingly easy to commit. A practitioner cares because leakage does not make results look slightly better — it routinely turns a worthless model into an apparent star, and it has contaminated a nontrivial share of published "ML beats econometrics" claims.

The mechanics. **Cross-validation (CV)** estimates out-of-sample accuracy by splitting the data into K **folds** (chunks), training on K−1 of them, and testing on the held-out fold, rotating until every fold has been the test set. Its validity rests on the training and test sets being informationally separated. Serial dependence destroys that separation in two ways:

1. **Neighbor leakage.** With random folds, almost every test observation is temporally sandwiched between training observations that are highly correlated with it. The model can interpolate rather than forecast.
2. **Overlap leakage.** When you forecast h steps ahead, even a *correctly specified* model has serially correlated errors. Write the series in terms of its shocks (the Wold form of Chapter 2): the h-step-ahead forecast error is

$$
e_{t+h|t} \;=\; \sum_{j=0}^{h-1} \psi_j\, \varepsilon_{t+h-j},
$$

where $\varepsilon_t$ are the one-step shocks and $\psi_j$ the moving-average weights. Two forecast errors dated less than h periods apart share shocks — h-step errors follow an MA(h−1) process. A training row whose target window overlaps a test row's target window is *made of the same randomness* the test asks you to predict.

Three repairs, in increasing order of caution:

- **Rolling-origin evaluation** (also called pseudo-out-of-sample, POOS): stand at origin $t$, fit using only data observable at $t$, forecast $t+h$, advance the origin, repeat. This is the gold standard because it simulates real forecasting exactly, including re-estimation.
- **Purging**: if you must use fold-based CV (it is K times cheaper than rolling origin when tuning many hyperparameters), make folds *contiguous blocks*, never shuffled, and delete from the training set any row whose target window overlaps the test block — every row within h−1 of it.
- **Embargo**: additionally drop a buffer of rows immediately after the test block. This guards against slower leakage channels — features built from smoothed or slow-moving transforms that carry test-period information forward (López de Prado 2018 introduced the purge/embargo terminology in the finance-ML setting). The embargo should scale with the forecast horizon.

Here is the split logic, in plain numpy — this is the pattern every honest backtest in this chapter uses:

```python
import numpy as np

def rolling_origin_splits(n_rows, h, first_origin):
    """Row i's features end at time i; its target is at time i + h.
    Train only on rows whose targets are already observed at the origin."""
    for origin in range(first_origin, n_rows):
        yield np.arange(0, origin - h + 1), origin

def purged_kfold_splits(n_rows, n_folds, h, embargo=0):
    """Contiguous folds. Purge training rows within h-1 of the test block
    (their target windows overlap it); embargo a further buffer after it."""
    all_rows = np.arange(n_rows)
    for fold in np.array_split(all_rows, n_folds):
        lo, hi = fold[0], fold[-1]
        keep = (all_rows <= lo - h) | (all_rows >= hi + h + embargo)
        yield all_rows[keep], fold
```

One honest nuance, because blanket rules breed cargo cults: Bergmeir, Hyndman & Koo (2018) prove that ordinary K-fold CV *is* valid for purely autoregressive models with uncorrelated errors. The intuition: if the model's inputs are lags of the series and the errors are genuinely unpredictable noise, having temporal neighbors in the training set tells the model nothing about the test-period noise. The trouble is that the conditions are fragile — one-step horizon, correctly specified lag structure, no external features, no memorizing model. The moment you add a smoothed feature, a trend index, a multi-step horizon, or a flexible learner, the theorem no longer protects you. Rolling origin is never wrong; K-fold is sometimes right. Price that asymmetry accordingly.

You have already seen this movie in a different theater. In Chapter 1's bootstrap discussion, iid resampling of an autocorrelated series destroyed the dependence structure and produced confidence intervals roughly three times too narrow:

![The iid bootstrap versus the stationary block bootstrap on dependent data](../examples/img/04-bootstrap.png)

Random K-fold CV is the same pathology wearing a different hat: an iid assumption applied to dependent data, quietly understating uncertainty — here, understating forecast error.

> **⚠ Common mistake.** Standardizing (or differencing, detrending, imputing, or selecting features) on the *full sample* before splitting. The test data's mean and variance leak into every training fold through the scaler. Every transform must be fit on training data only, inside each split — the tsecon roadmap's pipeline protocol enforces this mechanically, and property-tests it: perturbing data after a fold boundary must not change any fitted transform.

## A teachable disaster: one model, two backtests

Nothing makes the point like watching the same model get two different grades. The demo below builds a series that is, by construction, unforecastable beyond a drift-adjusted "no-change" guess: a random walk with drift. The "ML model" is a 3-nearest-neighbor regression — for each new feature vector, find the 3 most similar training rows and average their targets. The feature set is four lags plus a **trend index** (the row number), the kind of harmless-looking calendar feature that ML pipelines add reflexively. Everything runs today:

```python
import numpy as np
import tsecon

# --- A series nothing should beat: a random walk with drift ---------------
rng = np.random.default_rng(42)
n, p, h = 500, 4, 8                       # sample size, lags, forecast horizon
y = 0.2 * np.arange(n) + np.cumsum(rng.standard_normal(n))

# Supervised design: row i holds lags (y_t, ..., y_{t-3}) plus a trend index;
# its target is y_{t+8}. The trend index looks harmless. It is not.
rows = np.arange(p - 1, n - h)
X = np.column_stack([y[rows - j] for j in range(p)] + [rows.astype(float)])
target = y[rows + h]

def knn(X_tr, y_tr, x_new, k=3):          # the "ML model": 3-nearest-neighbor
    d = np.abs(X_tr - x_new).sum(axis=1)
    return y_tr[np.argsort(d)[:k]].mean()

# --- Backtest 1: the LEAKY protocol ---------------------------------------
# Sin 1: standardize features on the FULL sample (test data leaks into scaling).
# Sin 2: random K-fold — test rows sit surrounded by their temporal neighbors.
Xz = (X - X.mean(0)) / X.std(0)
perm = rng.permutation(len(rows))
leaky_pred = np.empty(len(rows))
for fold in np.array_split(perm, 5):
    train = np.setdiff1d(perm, fold)
    for i in fold:
        leaky_pred[i] = knn(Xz[train], target[train], Xz[i])

# --- Backtest 2: the HONEST protocol (rolling origin + purge) -------------
start = 250                                # first forecast origin
oos = np.arange(start, len(rows))
honest_pred, naive_pred = np.empty(len(oos)), np.empty(len(oos))
for j, i in enumerate(oos):
    train = np.arange(0, i - h + 1)        # purge: keep rows whose target is
    mu, sd = X[train].mean(0), X[train].std(0)   # already observed at the origin
    honest_pred[j] = knn((X[train] - mu) / sd, target[train], (X[i] - mu) / sd)
    drift = np.diff(y[: rows[i] + 1]).mean()     # benchmark: drift-adjusted
    naive_pred[j] = y[rows[i]] + h * drift       # no-change forecast

# --- Scoring: same evaluation rows, same metrics --------------------------
actual, insample = target[oos], y[: rows[start] + 1]
for name, pred in [("leaky", leaky_pred[oos]), ("honest", honest_pred),
                   ("naive", naive_pred)]:
    a = tsecon.accuracy(actual, pred, insample=insample)
    print(f"{name:7s} RMSE {a['rmse']:5.2f}   MASE {a['mase']:5.2f}")

dm = tsecon.dm_test(actual - naive_pred, actual - honest_pred, h=h)
print(f"DM (naive vs model, honest errors): stat {dm['hln_stat']:.2f}, "
      f"p = {dm['p_value']:.3f}")
```

The output:

```text
leaky   RMSE  1.19   MASE  1.24
honest  RMSE  4.38   MASE  4.49
naive   RMSE  2.99   MASE  3.12
DM (naive vs model, honest errors): stat -2.34, p = 0.020
```

Read the disaster carefully, because both halves teach something:

- **The leaky backtest says the model is a triumph**: RMSE 1.19 against the no-change benchmark's 2.99 — "our ML model beats the random walk by 60%." The mechanism is memorization: after full-sample z-scoring, the trend index makes each row's nearest neighbors its immediate temporal neighbors, and random folds guarantee those neighbors are in the training set. Their targets share almost all their shocks with the test target (the MA(h−1) overlap made flesh). The model is not forecasting; it is looking up the answer key.
- **The honest backtest says the model is a liability**: RMSE 4.38 — 47% *worse* than no-change, and the Diebold-Mariano test (from the forecasting-evaluation chapter; here `e1` is the benchmark, so a negative statistic favors it) says the deficit is statistically real (p = 0.02). The same trend feature that powered the illusion now sabotages genuine forecasts: at a real out-of-sample origin, the current trend value lies *outside* everything in the training set, so nearest-neighbor lookup grabs stale rows and systematically under-predicts. Flexible learners interpolate beautifully and extrapolate badly.

The gap between the two report cards — MASE 1.24 versus 4.49 for the *same model on the same data* — is the entire argument of this section. The leaky number is the one that would have gone in the paper.

The scoring machinery (`tsecon.accuracy`, `tsecon.dm_test`) is the same discipline layer used everywhere in the library — never report a forecast without a benchmark, never claim superiority without a test:

![Forecast evaluation: Theta versus seasonal naive, with MASE and the DM test](../examples/img/09-forecast-eval.png)

> **⚠ Common mistake.** Fixing the leak in evaluation but not in *tuning*. If you select hyperparameters (a penalty weight, tree depth, k) by leaky CV and then report performance from an honest holdout, the damage is smaller but still real: you have optimized the model for interpolation, not forecasting, and you have used the holdout once per tuning decision. Tune and evaluate with dependence-aware splits, and keep a final untouched window if the model will face real money or real policy.

## Shrinkage with a frequentist face

Now the models. The workhorse of data-rich forecasting is not deep learning — it is penalized linear regression. A practitioner cares because the typical macro problem has dozens-to-hundreds of candidate predictors (and each contributes several lags) against a few hundred observations. OLS in that regime is unbiased and useless: the variance of the coefficient estimates overwhelms any signal. **Shrinkage** — deliberately biasing coefficients toward zero to slash their variance — is the classic bias-variance trade, and it usually wins.

The unified estimator is the **elastic net** (Zou & Hastie 2005), which contains ridge and LASSO as endpoints:

$$
\hat\beta \;=\; \arg\min_{\beta}\; \frac{1}{2n}\sum_{t}\left(y_t - x_t'\beta\right)^2
\;+\; \lambda\left[\frac{1-\alpha}{2}\,\lVert\beta\rVert_2^2 \;+\; \alpha\,\lVert\beta\rVert_1\right],
$$

where $\lambda \ge 0$ sets the overall penalty strength, and $\alpha \in [0,1]$ mixes the two penalty geometries. $\alpha = 0$ is **ridge** (Hoerl & Kennard 1970): the squared L2 penalty shrinks every coefficient smoothly toward zero but never exactly to zero — right for *dense* problems where many predictors carry a little signal. $\alpha = 1$ is the **LASSO** (Tibshirani 1996): the absolute-value L1 penalty has a kink at zero that sets many coefficients *exactly* to zero, performing variable selection and estimation in one step — right for *sparse* problems where a few predictors carry most of the signal. Intermediate $\alpha$ matters for time series because lags of the same variable are highly collinear, and pure LASSO arbitrarily keeps one of two correlated lags while dropping the other; the ridge component stabilizes that choice. The **adaptive LASSO** (Zou 2006) reweights the penalty by first-stage estimates and is the theoretically preferred variant under dependence (Medeiros & Mendes 2016).

This should feel familiar from Chapter 10. Ridge is exactly the posterior mean of a regression with a Gaussian prior centered at zero; the LASSO solution is the posterior *mode* under a Laplace prior. Shrinkage is one idea wearing two costumes — the Minnesota prior shrinking a VAR toward a random walk and a ridge penalty shrinking a forecasting regression toward zero are the same move, differing in what they shrink *toward* and how they choose the strength. The frequentist face tunes $\lambda$ by cross-validation or information criteria; the Bayesian face integrates over it.

Ridge is simple enough to run honestly today, in numpy, with rolling-origin tuning. Truth here is an AR(2); the design offers 24 lags against roughly 100 usable observations:

```python
import numpy as np

rng = np.random.default_rng(7)
n, p = 120, 24                        # 24 candidate lags, ~100 usable rows
eps = rng.standard_normal(n)
y = np.zeros(n)
for t in range(2, n):                 # truth: AR(2) — only two lags matter
    y[t] = 0.5 * y[t-1] - 0.3 * y[t-2] + eps[t]

rows = np.arange(p, n - 1)
X = np.column_stack([y[rows - j] for j in range(p)])
target = y[rows + 1]

def ridge(X_tr, z, lam):              # closed form: (X'X + lam*I)^{-1} X'z
    return np.linalg.solve(X_tr.T @ X_tr + lam * np.eye(X_tr.shape[1]),
                           X_tr.T @ z)

def honest_rmse(lam, start=60):       # rolling origin, one step, purged
    e = [target[i] - X[i] @ ridge(X[:i], target[:i], lam)
         for i in range(start, len(rows))]
    return np.sqrt(np.mean(np.square(e)))

grid = np.geomspace(0.1, 1000.0, 25)
lam_star = grid[np.argmin([honest_rmse(lam) for lam in grid])]
print(f"lambda* = {lam_star:.0f}")                       # heavy shrinkage: 464
print(f"OLS  {honest_rmse(1e-8):.3f}")                   # 1.107
print(f"ridge {honest_rmse(lam_star):.3f}")              # 0.960
```

Unpenalized OLS pays a 15% RMSE premium for estimating 22 coefficients that are truly zero; tuned ridge gets within a whisker of the noise floor (the innovation standard deviation is 1.0). That is the whole shrinkage story in three lines of output.

Two structured extensions matter for time series designs. The **group LASSO** (Yuan & Lin 2006) penalizes coefficients in pre-declared groups by $\lambda \sum_g \sqrt{p_g}\, \lVert \beta_g \rVert_2$, so a group — naturally, *all lags of one variable* — enters or leaves the model as a unit; selection happens at the economic level ("does oil matter?") rather than the lag level ("does the seventh lag of oil matter?"). The **sparse-group LASSO** (Simon, Friedman, Hastie & Tibshirani 2013) mixes in a within-group L1 term so a selected variable can still use only a few of its lags — it is also the engine under the nowcasting chapter's MIDAS regressions. For tuning, information criteria are often preferable to CV in time series (faster, no splitting subtleties): for the LASSO the degrees of freedom equal the number of nonzero coefficients *exactly* (Zou, Hastie & Tibshirani 2007), making BIC well-defined; when the predictor count outruns the sample, use the extended BIC (Chen & Chen 2008) or BIC will overselect catastrophically.

*Roadmap preview — this API lands with Module 10:*

```python
path = tsecon.lasso_path(target, X)                     # glmnet-convention path
fit  = tsecon.enet(target, X, alpha=0.5, tune="bic")    # IC-tuned elastic net
grp  = tsecon.group_lasso(target, X, groups="lag-block")
```

> **⚠ Common mistake.** Reporting ordinary OLS standard errors on coefficients after any selection step. Selection is a data-dependent event; conditioning on it invalidates the usual distribution theory, and the resulting "t-statistics" are fiction. Refitting OLS on the selected support (post-LASSO, Belloni & Chernozhukov 2013) de-biases the *point estimates* but does not rescue the standard errors — for honest inference you need the desparsified LASSO or post-double-selection, both below. A second, quieter trap: penalized objectives differ across software in loss scaling (the $1/2n$ above is the glmnet convention) and standardization defaults, so the same $\lambda$ means different things in different packages. tsecon matches glmnet's conventions exactly so published $\lambda$ values transfer.

## Factor models: ML before it was cool

Dimension reduction by principal components was solving the "wide data" problem in economics two decades before the phrase "machine learning" entered the field's vocabulary. A practitioner cares because the **diffusion index** approach of Stock & Watson (2002, JASA; 2002, JBES) remains, on the evidence, roughly tied with penalized regression as the best simple thing you can do with a large macro panel — and it is the benchmark every ML paper must beat.

The model: a large standardized panel $X_t$ (N series) is driven by a small number r of common **factors** $F_t$,

$$
X_t \;=\; \Lambda F_t + e_t,
$$

where $\Lambda$ (N × r) holds the **loadings** — how much each series responds to each factor — and $e_t$ is series-specific noise. Estimate $\hat F_t$ by principal components (the r directions capturing the most panel variance), then forecast with a small regression that any econometrician would recognize:

$$
\hat y_{t+h} \;=\; \hat\alpha + \hat\beta(L)'\hat F_t + \hat\gamma(L)\, y_t,
$$

with $\beta(L)$, $\gamma(L)$ short lag polynomials. The factors compress 130 series into a handful of estimated indexes — empirically, a "real activity" factor, a "prices" factor, an "interest-rate spread" factor — and the forecasting equation stays low-dimensional, stable, and interpretable. The number of factors is chosen by the Bai & Ng (2002) information criteria; McCracken & Ng (2016) maintain the FRED-MD panel precisely so this literature has a common testbed. Refinements push the compression toward the forecast target rather than the panel's own variance: **targeted predictors** (Bai & Ng 2008) screen the panel for series relevant to $y$ before extracting factors, and the **three-pass regression filter** (Kelly & Pruitt 2015) extracts factors supervised by the target directly. Factors also plug into structural work — the FAVAR of Bernanke, Boivin & Eliasz (2005) puts $\hat F_t$ inside a VAR from Chapter 4.

Where does the ML framing earn its keep? Factor models *are* unsupervised learning — PCA is the textbook first example — followed by a supervised readout. Thinking of them that way clarifies both their power (they exploit the panel's pervasive comovement, exactly the structure that defeats sparse methods, as the next section explains) and their obligations (the entire pipeline, standardization and factor extraction included, must sit inside the honest backtest loop).

*Roadmap preview — this API lands with Module 10:*

```python
f  = tsecon.diffusion_index(panel, n_factors="bai-ng", kmax=8)
fc = tsecon.factor_forecast(y, f["factors"], h=12, y_lags=4)
```

> **⚠ Common mistake.** Estimating factors once on the full sample and then "backtesting" the forecasting regression on subsamples. The factor estimates at date t now contain information from the entire future of the panel — leakage again, wearing a third hat. In a proper pseudo-out-of-sample exercise the factors are re-extracted at every origin from data observable at that origin. The difference is not academic: full-sample factors are visibly smoother and can flip the sign of apparent forecast gains.

## Trees and boosting: when nonlinearity pays

Everything so far is linear. **Regression trees** partition the feature space by recursive binary splits ("is unemployment above 6%? is the yield-curve slope negative?") and predict a constant in each cell; a **random forest** (Breiman 2001) averages hundreds of trees, each grown on a resampled version of the data with random feature subsets, trading the tree's interpretability for dramatically lower variance. **Gradient boosting** builds an additive model by fitting each new tree to the residuals of the ensemble so far. These are the strongest off-the-shelf nonlinear learners in tabular ML. Do they help in macro?

The honest answer: sometimes, measurably, and we roughly know when. Medeiros, Vasconcelos, Veiga & Zilberman (2021) ran the definitive US inflation horse race on FRED-MD data and found random forests delivering RMSE gains of the order of 10–30% over benchmarks, including through the Great Recession. Goulet Coulombe, Leroux, Stevanovic & Surprenant (2022) dissected *why* across a broad set of targets: the gains come specifically from **nonlinearity** — regime-dependent predictive relationships, interactions, thresholds — and concentrate at longer horizons and in turbulent periods; the resampling and the feature-subsetting are supporting cast. The flip side is equally well documented: for smooth, persistent aggregates in short samples — most quarterly GDP-type problems — trees rarely beat a well-tuned ridge or factor model, and their step-function forecasts extrapolate trends poorly (recall the disaster demo: flexible learners interpolate, they do not extrapolate). If your prior is "linear with modest noise," shrinkage wins; if your prior is "the Phillips curve has a kink somewhere," forests can find it without being told where.

For econometricians, **componentwise L2 boosting** (Bühlmann & Yu 2003; applied to recessions by Ng 2014) deserves a special mention: each boosting step fits a single predictor-lag by least squares and adds a small fraction of it to the model. Read sequentially, it is an automated ARDL-building procedure — a variable selector that econometricians can audit line by line — and it is deterministic and cheap. Simpler still, bagging a pretest rule (estimate, keep what is significant, average over bootstrap replicates — Inoue & Kilian 2008) captures much of the ensemble benefit with none of the machinery, and is a good pedagogical bridge from OLS to ensembles.

Two dependence-specific cautions govern all tree methods on time series. First, the resampling inside a forest assumes exchangeable observations; on serial data the library's implementation offers block and stationary bootstrap resampling (Chapter 1's machinery) instead of the iid draw. Second — see the mistake box. On tooling: per the scope policy, tsecon implements its own random forest (needed for time-series-aware resampling and for the macroeconomic random forest below) but *wraps* XGBoost and LightGBM behind optional adapters rather than reimplementing histogram boosting; the adapters exist so external learners can run inside the library's honest CV, backtesting, and interpretation machinery.

> **⚠ Common mistake.** Trusting a random forest's **out-of-bag (OOB) error** — the built-in accuracy estimate computed from trees that did not see each observation — as a forecast-accuracy measure. OOB is a *random K-fold in disguise*: the observations a tree did not see are temporal neighbors of the ones it did. Under autocorrelation OOB error is systematically optimistic. Always grade forests the same way as everything else: rolling-origin pseudo-out-of-sample metrics.

## The sparsity illusion

The LASSO's great seduction is the story it tells: "of your 130 predictors, these seven matter." Before you build a narrative on that, read Giannone, Lenza & Primiceri (2021) — this section is the guide's version of required reading.

Their question is whether economic prediction problems are actually sparse (few big coefficients, rest zero — LASSO's home turf) or dense (many small coefficients — ridge's home turf). Rather than assuming an answer, they estimate it: a spike-and-slab regression (Chapter 10 machinery) in which the *probability of inclusion* $q$ is itself an unknown parameter with a prior,

$$
\beta_j \;\sim\;
\begin{cases}
0 & \text{with probability } 1-q,\\[2pt]
\mathcal N\!\left(0, \gamma^2\right) & \text{with probability } q,
\end{cases}
\qquad q \sim \text{Beta},
$$

so the data can express a posterior belief about sparsity itself. Across six canonical economic prediction problems — macro forecasting with large panels among them — the posterior on $q$ piles up away from zero: the data favor *dense* models. Worse for the storyteller, even when a sparse model predicts well, the posterior over *which* predictors to include is spread across many near-equivalent subsets. The apparent sparsity of a LASSO fit is largely an artifact of the penalty, not a discovery about the economy: collinear predictors (and macro panels are pervasively collinear, precisely because of the factor structure of the previous section) offer many equally good sparse representations, and the L1 kink picks one arbitrarily. Change the sample by a year and the selected set reshuffles while the forecasts barely move.

Practical consequences, in order of importance. First, expect ridge-type dense shrinkage and factor models to forecast at least as well as the LASSO on typical macro panels — the horse-race evidence (Smeekes & Wijler 2018; Medeiros et al. 2021) generally agrees. Second, when you do want sparsity under a factor structure, remove the common factors first and run selection on the idiosyncratic remainders — FarmSelect (Fan, Ke & Wang 2020) — or the factors will masquerade as whichever individual series the penalty happens to like. Third, and always: selection is a modeling convenience, not a causal finding.

> **⚠ Common mistake.** Publishing the list of LASSO-selected variables with economic interpretation attached ("credit spreads drive inflation; money growth does not"). The selection event is unstable across samples, tuning choices, and collinear substitutes. If the *identity* of the predictors is the scientific question, you need the inference tools of the next section and the frontier — not a penalty's arbitrary tiebreak.

## Causal machine learning with dependent data

Everything above chases predictive accuracy. Much of econometrics instead wants one number with a confidence interval: the effect of a policy variable $d_t$ on an outcome $y_t$, with many controls $x_t$ whose functional form you would rather not specify. **Double/debiased machine learning (DML)** (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins 2018) is the framework that lets flexible ML estimate the nuisance pieces while preserving valid inference on the parameter you care about. In the partially linear model

$$
y_t \;=\; \theta\, d_t + g(x_t) + u_t, \qquad d_t \;=\; m(x_t) + v_t,
$$

$\theta$ is the target and $g$, $m$ are unknown nuisance functions. Two ideas make it work. **Neyman orthogonality**: estimate $\theta$ from residualized quantities — regress $y - \hat g(x)$ on $d - \hat m(x)$ — so that first-order errors in $\hat g$ and $\hat m$ do not transmit to $\hat\theta$ (this is Frisch-Waugh-Lovell wearing ML clothes). **Cross-fitting**: fit the nuisances on one part of the sample and form residuals on another, swapping roles, so the ML models' overfitting does not correlate with the errors they are cleaning. A simpler cousin for linear nuisances is **post-double-selection** (Belloni, Chernozhukov & Hansen 2014): run LASSO of $y$ on controls, LASSO of $d$ on controls, take the union of selected supports, and run OLS of $y$ on $d$ plus that union — never penalizing $d$ itself.

With time series data, both ideas need repair, and this is where the chapter's opening theme returns with force. Cross-fitting's sample splits are random partitions — exactly the leaky construction of the disaster demo. Under serial dependence the folds must be *contiguous blocks separated by embargo buffers* scaled to the dependence horizon, or the nuisance fits leak test-period shocks and $\hat\theta$'s distribution theory quietly fails. And the variance of the orthogonal score inherits the serial correlation of $u_t$ and $v_t$, so the standard error must come from a HAC/long-run variance estimator (Chapter 3's `tsecon.long_run_variance` machinery, with all its bandwidth caveats), not the iid sample variance. Both repairs are engineering-grade in the library's design; the honest caveat is that the formal theory for DML under general dependence is still an active research area, so treat time-series DML output as carefully as you would any frontier method. For PDS with time-series controls (lags, deterministic terms), the same HAC discipline applies to the final OLS step.

> **⚠ Common mistake.** Running an off-the-shelf DML package with its iid defaults — random cross-fitting folds and iid score variances — on time series or panel data with serial dependence. Nothing errors out; you simply get confidence intervals that are too narrow and folds that leak. Check what the software assumes before believing the second decimal of a standard error.

## Neural forecasters and foundation models

Neural networks enter macro forecasting in three tiers, and the evidence deserves to be read tier by tier.

**Small feedforward networks** — one or two hidden layers on lagged inputs — are the "NN" line in the serious horse races (Medeiros et al. 2021; Goulet Coulombe et al. 2022). Verdict: competitive but rarely dominant; they capture the same nonlinearity premium as forests, at the cost of fussier optimization. Because training is nondeterministic (random initialization, floating-point reduction order across threads), respectable practice averages an ensemble over ~10 seeds and promises statistical, not bitwise, reproducibility. This is the one neural model the library implements natively.

**Specialized deep forecasters** — N-BEATS, N-HiTS, DeepAR, the Temporal Fusion Transformer — won fame on large panels of related series (retail demand, electricity, the M4/M5 competitions). Their sweet spot is *cross-learning*: thousands of related series sharing one model. Macro's typical regime — one target, a few hundred quarterly observations — is close to their worst case, and they seldom beat well-tuned classical benchmarks there.

**Foundation models** — Chronos (Ansari et al. 2024), TimesFM (Das et al. 2024), Moirai (Woo et al. 2024), Lag-Llama (Rasul et al. 2024), TimeGPT (Garza et al. 2023) — are pretrained on enormous corpora of time series and forecast new series zero-shot, no training required. The benchmark numbers are genuinely impressive, and for a practitioner with thousands of heterogeneous series and no time to model each one, they are a real option. For econometric use, three honest concerns temper the enthusiasm. *Contamination*: the pretraining corpora plausibly contain the very series (FRED, M-competition data) on which the models are evaluated, so a zero-shot "win" may be a memory. *Small-T macro evidence*: on quarterly aggregates the zero-shot results are mixed and frequently fail to beat an AR(1) or a BVAR once compared properly. *Auditability*: releases ship with demo notebooks, not DM tests, real-time data vintages, or stability analysis; and API-only models like TimeGPT mean sending your data to someone else's server — a nonstarter for central-bank workflows.

tsecon's policy follows from Chapter 0's scope discipline: **wrap, don't own**. Model adapters live in an optional companion package, because the model zoo churns annually. What stays in core is the piece that does not churn: a *contamination-aware benchmark harness* — DM and related tests wired in, evaluation restricted to post-training-cutoff windows, real-time vintages respected, and flags raised when an evaluation series plausibly sits in a model's training corpus. When a classical method wins that audit, the harness will say so plainly; historically, it says so often.

> **⚠ Common mistake.** Citing a foundation model's leaderboard win on public datasets as evidence it will beat your AR benchmark on your series. Until the comparison is run on data the model provably never trained on, at your horizons, against tuned classical benchmarks, with a significance test, it is marketing, not econometrics.

## Interpretation that respects the calendar

Once a nonlinear model earns its accuracy, you will be asked *why* it works. The standard ML interpretation toolkit — permutation importance, partial dependence, Shapley values — silently assumes features can be varied independently. Lagged time series designs violate that grossly, in two ways worth separating.

**Permutation importance** measures a feature's value by shuffling it and watching accuracy fall. Shuffling one lag of one variable, alone, fabricates feature vectors that no dynamic process could generate — unemployment at 5% in the $t-1$ slot next to 9% in the $t-2$ slot — and the accuracy drop confounds the feature's contribution with the model's response to impossible inputs; persistent predictors get systematically inflated importance. The repairs: permute in *contiguous blocks* (preserving each feature's own autocorrelation), and group *all lags of one variable* into a single importance unit, so the question becomes the economically meaningful one — "how much does the oil-price history matter?" — rather than a lag-by-lag artifact. The same grouping logic applies to Shapley-value attributions (Lundberg & Lee 2017): compute coalition values over variables, not individual lag features, and prefer conditional over interventional variants under strong dependence. For effect shapes, prefer **accumulated local effects (ALE)** (Apley & Zhu 2020) over partial-dependence plots: PD averages model predictions over feature combinations far outside the data's support (the correlated-lags problem again), while ALE accumulates local differences where data actually live.

These defaults — block permutation, lag-grouped attribution, ALE first — are what the roadmap's interpretation toolkit ships, over any model registered through the library's forecaster protocol, native or wrapped.

> **⚠ Common mistake.** Reading a per-lag importance bar chart ("lag 3 of the spread is the fifth most important feature") as structure. Under collinearity between lags, importance smears arbitrarily across them; only the grouped, block-respecting quantity is stable enough to interpret.

## The frontier

Where the research edge sits in 2026, and how the library's roadmap tracks it:

- **Honest inference in high-dimensional time series regression.** The desparsified LASSO (van de Geer, Bühlmann, Ritov & Dezeure 2014) de-biases penalized estimates so that individual coefficients get valid confidence intervals; Adamek, Smeekes & Wilms (2023) extend it to serially dependent data with HAC-variance corrections. This is the missing inference layer for every "which predictors matter" question the sparsity-illusion section warned about, and a Tier 4 roadmap item gated on reproducing the authors' simulation coverage tables.
- **The macroeconomic random forest** (Goulet Coulombe 2024) inverts the usual design: instead of trees predicting $y$ directly, trees model the *time-varying coefficients of a linear macro equation*. The output is a generalized time-varying-parameter model with tree flexibility — interpretable by construction, and one of the few genuinely new econometric objects to come out of the ML wave.
- **DML under dependence** remains theoretically unsettled: block cross-fitting with embargoes works in simulation, but sharp conditions on mixing rates, embargo widths, and nuisance convergence under realistic macro dependence are open. The library ships the engineering with the caveat attached.
- **Foundation-model evaluation** is becoming its own methodological problem: contamination detection, real-time vintage discipline, and post-cutoff testing protocols are where econometrics can referee the ML claims. The contamination-aware harness is the roadmap's bet that the *audit* outlives any particular model.
- **Conformal prediction for time series** — distribution-free prediction intervals via EnbPI (Xu & Xie 2021) and adaptive conformal inference (Gibbs & Candès 2021) — loses its exchangeability foundation under dependence; the adaptive variants restore marginal coverage empirically. It lives in the forecasting-evaluation module and wraps any forecaster from this one.

Open problems worth a thesis: valid post-selection inference under general dependence without sample splitting; a workable theory of when cross-learning across series helps macro (transfer learning for economics); and evaluation protocols for pretrained forecasters that the pretraining corpus cannot game.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| Dozens-to-hundreds of correlated predictors, dense weak signals | Ridge, or a diffusion index | Macro panels are dense and factor-driven (Giannone-Lenza-Primiceri); smooth shrinkage and factors exploit exactly that structure |
| Genuinely sparse signal suspected, selection wanted | LASSO / adaptive LASSO / elastic net, BIC-tuned | Oracle-type selection under sparsity; elastic net stabilizes collinear lag choices |
| Whole variables should enter or leave (all their lags) | Group or sparse-group LASSO | Selection at the economic unit; per-lag selection is noise |
| Hundreds of series, need the standard benchmark | Stock-Watson diffusion index | The canonical data-rich baseline every paper must beat |
| Long monthly samples, nonlinearity plausible (inflation, recessions) | Random forest / boosting with block resampling, graded by POOS | The documented ML win in macro is nonlinearity (Medeiros et al.; Goulet Coulombe et al.) |
| One causal coefficient, many controls | PDS-LASSO or DML with block cross-fitting and HAC scores | Inference validity is the product; prediction is a byproduct |
| Short quarterly sample (T < ~150), smooth aggregate | AR / theta / BVAR from earlier chapters | ML's variance costs exceed its bias savings; classical wins the honest test |
| A zero-shot foundation-model claim on your desk | The contamination-aware harness: post-cutoff window, DM test vs AR(1) | Public-benchmark wins may be memorized; audit before adopting |
| Tuning *any* hyperparameter above | Rolling-origin CV, or purged-and-embargoed blocked CV | Random K-fold flatters every model under dependence |
| Explaining whatever won | Lag-grouped block-permutation importance, ALE | Per-lag permutation fabricates impossible histories and inflates persistent predictors |

## What tsecon implements today

**Available now in Python** (everything this chapter's runnable code used): the honest-evaluation spine — `accuracy` (RMSE/MAE/MASE and friends), `dm_test` (with the Harvey-Leybourne-Newbold correction), `theta_forecast` as a benchmark; the dependence-aware resampling stack — `bootstrap_indices`, `optimal_block_length`, `philox_uniforms`; and the inference machinery ML methods lean on — `ols(se_type="hac")` and `long_run_variance`. Leakage-safe splits are a dozen lines of numpy (this chapter's `rolling_origin_splits` / `purged_kfold_splits`) until the module ships them natively.

**Built in Rust awaiting bindings:** nothing from this module yet — the Module 10 solver stack has not started. Its foundations-layer dependencies are already live in the crates, though: the Philox RNG, the block-bootstrap engine, the HAC/long-run-variance module (including fixed-b machinery), and the optimizer suite the penalized solvers will call.

**Roadmap:** the full module design is in [docs/roadmap/10-machine-learning.md](../roadmap/10-machine-learning.md) — Tier 1: the leakage-safe pipeline protocol and TS-CV suite, the glmnet-convention elastic-net/LASSO/ridge path solvers with IC tuning, and the Stock-Watson diffusion-index facade; Tier 2: group and sparse-group LASSO, native random forests with block resampling, componentwise boosting, GBT adapters, PDS-LASSO, and dependence-aware interpretation; Tiers 3-4: DML with block cross-fitting, lag-grouped Shapley, the GLP sparsity diagnostic, the desparsified LASSO under dependence, the macroeconomic random forest, and the contamination-aware benchmark harness. Deep-learning and foundation-model adapters live in a companion package by scope ruling — wrap, don't own.

## Further reading

- **Tibshirani (1996), JRSS-B** — the LASSO paper; the L1 penalty and why it selects. The single most-cited object in this chapter.
- **Stock & Watson (2002), JASA** — principal-components forecasting with many predictors; the paper that made "data-rich" a benchmark rather than a dream.
- **Bergmeir, Hyndman & Koo (2018), Computational Statistics & Data Analysis** — the careful statement of when K-fold CV is valid for time series; antidote to both leakage and cargo-cult purging.
- **Medeiros, Vasconcelos, Veiga & Zilberman (2021), JBES** — the definitive US inflation horse race; the empirical case that ML gains in macro are real, and where they come from.
- **Goulet Coulombe, Leroux, Stevanovic & Surprenant (2022), Journal of Applied Econometrics** — dissects *which* ML ingredients matter for macro forecasting (answer: nonlinearity); the paper to read after the horse race.
- **Giannone, Lenza & Primiceri (2021), Econometrica** — the illusion of sparsity; required reading before interpreting any selected support.
- **Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), The Econometrics Journal** — double/debiased ML; the framework for valid causal inference with ML nuisances.
- **Mullainathan & Spiess (2017), Journal of Economic Perspectives** — "Machine learning: an applied econometric approach"; the gentlest serious orientation for economists.
- **Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical Learning*, 2nd ed., Springer** — the reference textbook for every method named here; free online.
- **López de Prado (2018), *Advances in Financial Machine Learning*, Wiley** — the source of purging and embargo; finance-flavored, but the leakage chapters transfer to macro intact.
