# Chapter 14 — Panel Time Series: Many Series, Shared Shocks, and Heterogeneous Dynamics

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** OLS and robust/HAC standard errors (chapter 3), the VAR and its impulse responses (chapter 7), and local projections (chapter 9) — chapter 9's "State dependence, panels, and other extensions" section previewed the panel LP this chapter delivers in full. The factor-model idea from chapter 7 (a few common factors driving a wide panel) is the conceptual key to this chapter's climax.

**You will learn:**

- Why stacking many time series into a panel buys statistical power that no single series can — and why that same stacking breaks the independence textbook standard errors assume, in *two* different directions at once
- How the fixed-effects panel regression removes time-invariant confounders, and how clustered versus Driscoll-Kraay standard errors handle serial correlation and cross-sectional dependence respectively
- How panel local projections estimate the impulse response to a *common* shock across many entities, and the two panel-specific pitfalls that grow exactly when you least want them to
- Why pooling a dynamic model across entities is *inconsistent*, not merely inefficient, when their dynamics differ — and how the Pesaran-Smith mean-group estimator sidesteps the problem by averaging per-entity fits
- The common-factor problem — the panel form of omitted-variable bias that more data cannot cure — and how Pesaran's common correlated effects augmentation purges it with nothing but cross-section averages

## The idea

Every chapter so far has watched *one* history unfold: one country's GDP, one asset's returns, one system of national accounts. A **panel** watches many at once — 17 countries over 150 years, 50 US states over 40 quarters, thousands of firms over a decade — every entity $i$ observed at every date $t$. The bargain is irresistible: where a single 70-quarter series gives you 70 observations to pin a slope, a panel of 24 such series gives you 1,680. Effects too faint to see in any one history light up when you stack them, which is why the questions modern macro cares about most — what follows a credit boom, how fiscal policy transmits, whether financial development causes growth — are panel questions.

The catch is that those 1,680 observations are nowhere near 1,680 independent draws, and the double-counting runs in two directions at once. **Down each entity's timeline**, this quarter's value is correlated with last quarter's — the serial dependence every earlier chapter has drilled into you. **Across entities at a single date**, a global recession pushes every country's output down together, an oil shock lifts every economy's inflation together — this is *cross-sectional dependence*, and it is new. A panel has both, and standard errors that ignore either will lie to you, usually by a factor of two or three in the direction of overconfidence. Half of panel econometrics is a careful accounting of which dependence your standard errors are allowed to respect.

Two more problems the single series never posed, both born of having *many* units:

**Do the units even share the same dynamics?** A pooled model forces one slope, one set of lag matrices, one impulse response on every entity. Sometimes that is exactly the discipline you want — more units, one number, tighter bands. But if Germany's and Greece's economies genuinely respond differently to a rate rise, forcing them to share coefficients does not just blur the two answers into an average; for *dynamic* models it produces an estimate that converges to the *wrong* number as your sample grows. Heterogeneity is not noise to be averaged away — it can be a bias to be reckoned with.

**And the deepest problem of all: common factors.** The world business cycle, the global financial cycle, the price of oil — forces that move *every* entity at once. Chapter 7 taught you to see these as factors: a handful of latent series $f_t$, loaded onto each observed series through unit-specific weights. In a panel they are not a nuisance to be swept into a time dummy, because they load *heterogeneously* — each country reads the global cycle through its own exposure. And here is the trap that gives this chapter its climax: if the same unobserved factor drives both your outcome and your regressor, it manufactures a correlation between them that has nothing to do with the causal slope you want. That is omitted-variable bias — but a peculiarly vicious form of it, because unlike ordinary noise it does **not** wash out as $N \to \infty$. Every new entity you add carries the *same* factor, so more data simply estimates the biased number more precisely. The fix, when you first see it, looks like a magic trick: you never observe the factor, yet you can purge it using only the cross-sectional averages of the data you already have.

This chapter climbs those problems in order. The **fixed-effects regression** with dependence-robust standard errors handles the two-directional dependence for a pooled slope. **Panel local projections** carry that machinery to impulse responses of a common shock. The **mean-group panel VAR** confronts dynamic heterogeneity by refusing to pool. And the **common correlated effects** estimator confronts the common-factor problem head-on — the payoff the whole chapter builds toward, demonstrated on a panel where naive pooling is visibly, stubbornly wrong and one extra idea sets it right.

A running dataset ties three of the four sections together. The fixture `fixtures/tsecon-panelts.json` holds a panel of $N = 24$ units observed over $T = 70$ periods, generated from

$$
y_{it} = a_i + b_i' x_{it} + \gamma_i f_t + e_{it}, \qquad x_{it} = \mu_i + \delta_i f_t + v_{it},
$$

with a single common factor $f_t$ loaded into *both* the outcome (through $\gamma_i$) and the two regressors (through $\delta_i$), and both loadings having positive means. The true average slope vector is $(1.5, -0.8)$. Keep that pair in mind: it is the number every estimator in this chapter is trying to hit, and watching which ones miss is how you will learn what each one assumes.

## Fixed effects with dependence-robust standard errors

Start with the workhorse. You have a panel and you want one slope — the average effect of $x$ on $y$, holding fixed everything about each entity that never changes. A practitioner reaches for this constantly: it is the panel version of the OLS regression from chapter 3, and 90% of applied panel work is some careful variation on it.

**The intuition: subtract each entity out of itself.** Suppose rich countries both grow faster *and* have more of whatever $x$ measures. A cross-sectional regression of growth on $x$ would confound the two. The fixed-effects trick removes the confound without ever measuring it: from every variable, subtract that *entity's own time-average*, then run OLS on the deviations. Anything about country $i$ that is constant over time — its institutions, its geography, its permanent growth level — is identical to its own mean and vanishes in the subtraction. What survives is purely *within-entity* variation: does $y$ move when $x$ moves *for the same unit over time*? That "within transformation" is the entire content of the fixed-effects estimator.

**The model.** With entity effects $a_i$ and a common slope vector $\beta$,

$$
y_{it} = a_i + \beta' x_{it} + \varepsilon_{it}, \qquad i = 1, \dots, N, \quad t = 1, \dots, T,
$$

estimated by OLS after within-demeaning (equivalently, with a full set of entity dummies). The point estimate is the easy part and rarely the interesting part. The hard part — the part that separates credible panel work from the rest — is the **standard error**, because $\varepsilon_{it}$ carries exactly the two-directional dependence the chapter opened with, and you must tell the estimator which kinds to respect:

- **`"nonrobust"`** assumes $\varepsilon_{it}$ is i.i.d. across both $i$ and $t$. In a genuine time-series panel this is essentially never true, and it is essentially always too small. Use it only as a deliberately naive baseline — a number whose job is to be beaten.
- **`"cluster"`** (by entity) allows *arbitrary* correlation and heteroskedasticity within each entity's timeline — the whole serial-correlation problem, handled without modeling it — while assuming entities are independent of one another. This is the default and the right first move when you have many entities and cross-sectional dependence is mild.
- **`"driscoll_kraay"`** goes further: it is a Newey-West HAC applied to the cross-sectional averages of the moment conditions, so it is robust to serial correlation *and* to cross-sectional dependence of arbitrary form (Driscoll and Kraay 1998). This is the choice for macro panels, where a global cycle correlates the entities' errors at every date and clustering-by-entity alone would understate the uncertainty.

`tsecon.panel_fe` takes the outcome as an $N \times T$ array and the regressors as a $k \times N \times T$ stack — exactly the layout the fixture stores — and switches standard-error regime with one argument:

```python
import json, numpy as np, tsecon

d = json.load(open("fixtures/tsecon-panelts.json"))
y = np.array(d["y"])          # N x T   outcome
x = np.array(d["x"])          # k x N x T   regressors
print(y.shape, x.shape)       # (24, 70) (2, 24, 70)

for se in ["nonrobust", "cluster", "driscoll_kraay"]:
    r = tsecon.panel_fe(y, x, se_type=se)
    print(f"{se:15s} params={np.round(r['params'],3)}  bse={np.round(r['bse'],3)}")
# nonrobust       params=[ 1.807 -0.465]  bse=[0.025 0.025]
# cluster         params=[ 1.807 -0.465]  bse=[0.072 0.07 ]
# driscoll_kraay  params=[ 1.807 -0.465]  bse=[0.041 0.027]
print("true slopes:", d["true_mean_slopes"])   # [1.5, -0.8]
```

Read the output two ways. First, the standard errors: the point estimates are identical across all three rows — the SE choice never touches $\hat\beta$, only its uncertainty — but the naive standard error (0.025) is *one-third* the clustered one (0.072). If you reported the nonrobust number you would claim three times the precision you have, the single most common way panel inference goes wrong. Driscoll-Kraay lands between them here because this fixture's cross-sectional dependence, while real, is moderate; in a genuine macro panel with a strong global cycle, Driscoll-Kraay routinely exceeds the clustered SE.

Second — and this is the seam the rest of the chapter pulls on — look at the *point estimate*. The pooled fixed-effects slope is $(1.81, -0.47)$, but the true average slope is $(1.5, -0.8)$. That is not sampling noise; no standard error is large enough to cover the gap. Fixed effects removed everything time-*invariant* about each entity, but the confounder in this data is the time-*varying* common factor $f_t$, and entity demeaning does nothing to it. Hold that failure in mind: it is the exact disease the mean-group and CCE sections diagnose and, eventually, cure.

> **⚠ Common mistake — clustering with too few clusters.** The cluster-robust standard error is a large-*number-of-clusters* approximation: its theory leans on $N$ (the count of entities) being large, not $T$. With a panel of 6 countries or 10 industries, the clustered SE is itself badly estimated and typically *too small* — the opposite of the reassurance you reached for it to get. Below roughly 30–40 clusters, use the wild cluster bootstrap (Cameron, Gelbach and Miller 2008) instead of trusting the asymptotic clustered SE, and never read a clustered $t$-statistic from a handful of clusters as if it were standard-normal. Driscoll-Kraay has the mirror-image requirement: it leans on large $T$, so it is the wrong tool for a short, wide panel.

## Panel local projections

Chapter 9 built the impulse response one regression per horizon, for a single time series. The panel version keeps that skeleton and pools it across entities — and it is the engine behind the Jordà-Schularick-Taylor macrohistory program, where "what happens after a credit boom?" is answered by projecting many countries' outcomes onto a common event, quarter by quarter across the horizon.

**The intuition.** You have a shock that hits every entity at the same dates — a global oil-price surprise felt by every oil-importing country, a US monetary shock transmitted to every emerging market — and you want its dynamic effect on some outcome. Run chapter 9's local projection, but stack all entities into one regression at each horizon, with entity fixed effects soaking up permanent differences in the outcome's level. Pooling across entities is what buys the tight bands: each country contributes its own realization of the same experiment.

**The model.** For each horizon $h = 0, 1, \dots, H$, one within-entity regression,

$$
y_{i,t+h} = a_i^{(h)} + \beta_h\, s_t + \sum_{\ell=1}^{p} \phi_{\ell}^{(h)}\, w_{i,t-\ell} + \xi_{i,t+h},
$$

where $s_t$ is the common shock (one value per date, shared by all entities), $a_i^{(h)}$ are entity fixed effects, and the lag controls $w$ purge pre-shock dynamics. The sequence $\{\beta_h\}_{h=0}^{H}$ *is* the impulse response. Because the errors carry both serial and cross-sectional dependence, Driscoll-Kraay standard errors are the sensible default — and `tsecon.panel_lp` makes them exactly that.

A synthetic panel makes the recovery visible. Build a dynamic panel in which a common shock $s_t$ propagates through a persistent outcome, $y_{i,t} = a_i + \rho\, y_{i,t-1} + \beta\, s_t + u_{i,t}$, so the true impulse response is the geometric decay $\beta \rho^h$:

```python
import numpy as np, tsecon

rng = np.random.default_rng(11)
N, T, rho, beta = 60, 250, 0.6, 1.0
s = rng.standard_normal(T)                 # ONE common shock per period
a = rng.normal(0, 1, N)                     # entity fixed effects
y = np.zeros((N, T))
for i in range(N):
    for t in range(1, T):
        y[i, t] = a[i] + rho * y[i, t-1] + beta * s[t] + 0.5 * rng.standard_normal()

r = tsecon.panel_lp(y, s, horizon=6, n_lag_controls=2, se_type="driscoll_kraay")
print("irf :", np.round(r["irf"], 3))   # [1.004 0.627 0.346 0.224 0.146 0.08  0.049]
print("se  :", np.round(r["se"], 3))    # [0.005 0.067 0.075 0.088 0.082 0.089 0.086]
print("true:", np.round(beta * rho**np.arange(7), 3))  # [1. 0.6 0.36 0.216 0.13 0.078 0.047]
```

The estimated IRF tracks the geometric truth horizon by horizon — impact $1.004$ against a true $1.0$, then $0.63, 0.35, 0.22, \dots$ shadowing $0.6, 0.36, 0.22, \dots$ — with Driscoll-Kraay bands that widen away from impact exactly as the overlapping-horizon dependence of chapter 9 predicts. Setting `cumulative=True` returns the *cumulative* response $\sum_{j\le h}\beta_j$ directly, the object you want for a multiplier; the `jackknife=True` flag turns on a bias correction discussed next.

Two panel-specific pathologies discipline this, and both were flagged in chapter 9's panel paragraph — here they are with the estimator in hand:

1. **Nickell bias that grows with the horizon.** Combining entity fixed effects with a lagged dependent variable produces the classic dynamic-panel (Nickell 1981) bias of order $O(1/T)$. In a local projection it is *worse*: because the horizon-$h$ regression effectively lags the outcome $h$ periods against the demeaned regressors, the bias scales like $O(h/T)$ — small at impact, but accumulating precisely at the long horizons where the interesting policy story usually lives, and dangerous exactly when your panel is short. The `jackknife=True` option applies a split-panel bias correction; on the panel above it barely moves the estimates (the panel is long, $T = 250$), which is the point — the correction matters when $T$ is small, and there it can shift long-horizon responses materially.
2. **The h-shifted sample changes shape.** Projecting the outcome $h$ periods ahead drops the last $h$ observations of every entity, and in an *unbalanced* panel it drops *different* entities' tails at different horizons. The effective sample quietly changes composition across the IRF, so a response that "moves" at long horizons may be partly telling you that the sample moved. Report the effective observation count per horizon (`panel_lp` returns `nobs` alongside `irf`) and be wary when it falls sharply.

> **⚠ Common mistake — a "shock" that is really an outcome.** Panel LP assumes $s_t$ is a genuine shock: unpredictable from the panel's own past. Feeding it a raw policy *variable* (the level of the interest rate, the level of government spending) rather than an identified *surprise* re-imports every endogeneity problem chapters 8 and 9 spent their length exorcising — the estimated "response" then confounds the policy's effect with whatever made policymakers act. Panel structure does not launder an unidentified shock; it just estimates the confounded object across more units. Identify the shock first (chapter 8), then panel-project it.

## Mean-group panel VAR: when the dynamics themselves differ

The fixed-effects and panel-LP estimators both *pool*: one slope, one set of dynamics, imposed on every entity. That is a modeling choice with a sharp downside when it is wrong. Pesaran and Smith (1995) proved the uncomfortable result at the heart of this section: in a *dynamic* panel — one with lagged dependent variables, i.e. a panel of VARs — if the true coefficients differ across entities, the pooled estimator does not merely lose efficiency, it converges to the **wrong** number even as $N$ and $T \to \infty$. Averaging heterogeneous *static* slopes is harmless; averaging heterogeneous *dynamics* by pooling is biased, because the pooled lag coefficient absorbs a piece of the cross-entity dispersion in a way that does not cancel.

**The mean-group answer is almost aggressively simple: never pool.** Fit a *separate* model to every entity, then average the results across entities. If each unit has enough time-series length to support its own VAR, you sidestep the pooling bias entirely — each entity's dynamics are estimated on its own data, and the cross-entity average is an honest estimate of the mean dynamics.

**The estimator.** For a panel of $k$-variable systems, fit a reduced-form VAR to each entity $i$ separately (this is chapter 7's `var_fit` under the hood), giving per-entity intercepts $\hat c_i$, lag matrices $\hat A_i$, and orthogonalized impulse responses $\widehat{\text{IRF}}_i(h)$. The mean-group estimates are the plain cross-entity averages,

$$
\hat\theta_{MG} = \frac{1}{N} \sum_{i=1}^{N} \hat\theta_i,
\qquad
\widehat{\mathrm{Var}}(\hat\theta_{MG}) = \frac{1}{N(N-1)} \sum_{i=1}^{N} (\hat\theta_i - \hat\theta_{MG})(\hat\theta_i - \hat\theta_{MG})',
$$

and the standard error is the *cross-entity dispersion* over $\sqrt{N}$ — $\mathrm{sd}_i(\hat\theta_i)/\sqrt{N}$ — not the within-entity sampling error. This is the same "average of per-unit estimates, spread of per-unit estimates as the SE" logic you will see again in the next section; it needs $N \ge 2$ to have any dispersion at all, and enough entities that the sample spread is a decent estimate of the population spread.

`tsecon.mean_group_var` takes the panel as a *list of per-entity matrices*, each $T_i \times k$ (oldest row first), and the $T_i$ may differ across entities:

```python
import numpy as np, tsecon

rng = np.random.default_rng(0)
base = np.array([[0.5, 0.1], [0.05, 0.4]])
entities = []
for _ in range(12):                          # 12 entities, each its own VAR(1)
    A = base + 0.06 * rng.standard_normal((2, 2))   # heterogeneous dynamics
    c = 0.2 * rng.standard_normal(2)
    Y = np.zeros((150, 2)); yv = np.zeros(2)
    for t in range(150):
        yv = c + A @ yv + 0.5 * rng.standard_normal(2); Y[t] = yv
    entities.append(Y)                       # a T_i x k matrix per entity

mg = tsecon.mean_group_var(entities, lags=1, trend="c", horizon=6,
                           response=1, impulse=0)
print("mean A1:\n", np.round(np.asarray(mg["coefs"])[0], 3))
# [[0.519 0.139]
#  [0.041 0.359]]
print("A1 dispersion se:\n", np.round(np.asarray(mg["coefs_se"])[0], 3))
# [[0.023 0.028]
#  [0.028 0.022]]
print("irf_path (var1 <- shock0):", np.round(mg["irf_path"], 3))
# [-0.016  0.015  0.017  0.013  0.01   0.007  0.005]
```

The mean lag matrix $(0.52, 0.14; 0.04, 0.36)$ recovers the generative center $(0.5, 0.1; 0.05, 0.4)$, with the dispersion-based standard errors measuring how much the 12 entities' dynamics actually spread around that center. The `response`/`impulse` arguments pull one orthogonalized IRF series (here variable 1's response to a shock in variable 0) out of the full `orth_irfs` array as `irf_path`, with its own dispersion band `irf_path_se` — the mean-group analogue of the chapter-7 impulse response, now with cross-country uncertainty attached.

When should you reach for it? When each entity has a real time-series length ($T_i$ large enough for a stable VAR — a handful of variables and a few lags need well over a hundred periods), when you have enough entities for a meaningful dispersion ($N$ in the double digits is comfortable), and when you have positive reason to think the dynamics genuinely differ. Its opposite number is the pooled panel VAR: cheaper and tighter when the homogeneity assumption holds, biased when it does not. A Hausman-style comparison of the pooled and mean-group estimates is the standard diagnostic — a large gap is evidence that pooling is imposing a homogeneity the data reject.

> **⚠ Common mistake — mean-group on entities too short to fit.** The estimator's independence from pooling bias is bought with a demand: each entity must carry its *own* stable VAR. Hand it entities with 40 quarters and a three-variable, two-lag system and every per-entity fit is a near-singular overfit, the individual $\hat\theta_i$ are wild, and their average — however unbiased in theory — is uselessly noisy in practice. Mean group trades the pooling bias for a hunger for per-entity length; when your entities are short, pooling (bias and all) or a shrinkage estimator that borrows strength across units may genuinely beat it. The bias you avoid is worthless if the variance you inherit is larger.

## The common-factor problem and the CCE cure

This is the chapter's summit, and the payoff for the foreshadowing. Recall the failure from the fixed-effects section: on the fixture, pooled FE returned $(1.81, -0.47)$ against a true $(1.5, -0.8)$, and the culprit was the time-varying common factor $f_t$ that entity demeaning cannot touch. The mean-group estimator of the previous section, for all its virtues against dynamic heterogeneity, is *equally* helpless here — averaging per-unit regressions that each omit the same factor just averages a common bias. To see the disease clearly and then watch it cured, meet the generative model of `fixtures/tsecon-panelts.json` in full:

$$
y_{it} = a_i + b_i' x_{it} + \gamma_i f_t + e_{it},
\qquad
x_{it} = \mu_i + \delta_i f_t + v_{it}.
$$

An unobserved common factor $f_t$ — read it as the global business cycle — drives the outcome through loadings $\gamma_i$ *and* drives the regressors through loadings $\delta_i$. Because $f_t$ appears on both sides, it manufactures a correlation between $x$ and $y$ that has nothing to do with the structural slope $b_i$. That is textbook omitted-variable bias, and the mean-group slope inherits it: averaging the per-unit OLS estimates leaves the term

$$
\text{bias} \;\approx\; \frac{\mathbb{E}[\gamma_i\, \delta_i]\, \mathrm{Var}(f)}{\mathrm{Var}(x)},
$$

which is **non-vanishing** precisely because the loadings $\gamma_i$ and $\delta_i$ have *nonzero means* — the factor is common, not idiosyncratic, so its contamination points the same way for (almost) every unit and cannot average to zero across units. This is what makes common-factor bias so much more dangerous than ordinary noise: enlarging $N$ estimates the wrong number ever more precisely. A tight confidence interval around a biased point is the worst of both worlds, and it is exactly what naive mean group delivers here.

**Pesaran's (2006) fix is the elegant heart of the chapter.** You cannot condition on $f_t$ because you never observe it. But look at what the *cross-sectional averages* of the data contain. Average the outcome equation over units at a fixed date $t$:

$$
\bar y_t = \bar a + \overline{b' x}_t + \bar\gamma\, f_t + \bar e_t,
$$

and as $N \to \infty$ the idiosyncratic error $\bar e_t \to 0$ while $\bar\gamma \to \mathbb{E}[\gamma_i] \neq 0$ — so $\bar y_t$ (together with the cross-sectional average $\bar x_t$) becomes an *observable combination that spans the space of the unobserved factor* $f_t$. The averages carry the common factor and average away everything idiosyncratic. So: augment each per-unit regression with the cross-section averages $\bar y_t$ and $\bar x_t$ as extra regressors, let them absorb the factor, and MG-average only the *own-$x$* slopes. The factor is netted out of every unit's slope before you average. No factor model is estimated, no number of factors is chosen up front, nothing latent is ever recovered — the averages you already have do all the work. This is **common correlated effects mean group** (CCE-MG).

`tsecon.panel_mean_group` implements both estimators behind one `method` switch. It takes the panel as per-unit response vectors `ys` and per-unit $T_i \times k$ regressor matrices `xs`; from the fixture's $N \times T$ and $k \times N \times T$ arrays that is one reshape:

```python
import json, numpy as np, tsecon

d = json.load(open("fixtures/tsecon-panelts.json"))
y = np.array(d["y"]); x = np.array(d["x"])
N, T = y.shape; K = x.shape[0]

ys = [y[i] for i in range(N)]                                        # per-unit y_i  (T,)
xs = [np.column_stack([x[k, i] for k in range(K)]) for i in range(N)]  # per-unit X_i  (T, k)

mg  = tsecon.panel_mean_group(ys, xs, method="mg")     # plain mean group
cce = tsecon.panel_mean_group(ys, xs, method="cce")    # CCE mean group

print("true slopes:", d["true_mean_slopes"])                     # [1.5, -0.8]
print("MG   coef  :", np.round(mg["coef"], 3),  "tstat", np.round(mg["tstat"], 2))
# MG   coef  : [ 1.778 -0.486] tstat [26.52 -7.37]
print("CCE  coef  :", np.round(cce["coef"], 3), "tstat", np.round(cce["tstat"], 2))
# CCE  coef  : [ 1.426 -0.782] tstat [ 29.6  -12.49]
print("MG   bias  :", np.round(np.array(mg["coef"])  - d["true_mean_slopes"], 3))  # [0.278 0.314]
print("CCE  bias  :", np.round(np.array(cce["coef"]) - d["true_mean_slopes"], 3))  # [-0.074 0.018]
```

There is the whole chapter in six numbers. Plain mean group returns $(1.78, -0.49)$ — biased by $+0.28$ and $+0.31$, badly enough that the *sign* of the second slope's magnitude is understated by 40% — and reports a $t$-statistic of 26 and $-7$, supremely confident in the wrong answer. CCE-MG returns $(1.43, -0.78)$, within a whisker of the true $(1.5, -0.8)$, with biases an order of magnitude smaller. The only difference between the two calls is `method="cce"`, and all that flag does is add the cross-section averages to each unit's regression. The common factor that defeated fixed effects *and* plain mean group is purged by the most elementary object in the panel — the average across units at each date.

The estimator returns `coef_per_unit` (the $N \times k$ matrix of individual slopes it averaged) alongside the summary, so you can inspect the cross-unit distribution the mean-group SE is built from — a wide spread is a signal that a single average slope may be hiding real heterogeneity worth modeling rather than summarizing.

The assumptions that make the trick work, and the conditions under which it frays:

- **CCE needs both $N$ and $T$ reasonably large.** The averages span the factor space only as $N \to \infty$; with a handful of units, $\bar y_t$ is a noisy proxy for the factor and the purge is partial. And each per-unit augmented regression needs enough $T$ to estimate its extra coefficients.
- **A rank condition on the factor loadings.** The cross-section averages can span the space of *up to* $k + 1$ factors (the number of averaged series). More common factors than that and CCE cannot fully absorb them; the loadings must also be non-degenerate (strong factors, nonzero-mean loadings) — the very condition that makes the bias non-vanishing is what makes the cure work.
- **Static CCE assumes the regressors are (weakly) exogenous given the factors.** With lagged dependent variables the basic estimator is biased in short $T$; the dynamic extension (Chudik and Pesaran 2015) adds lags of the cross-section averages to restore consistency — the frontier's business, below.

> **⚠ Common mistake — sweeping a common factor into a time dummy.** The instinct on meeting cross-sectional dependence is to add time fixed effects: one dummy per date to soak up "whatever hit everyone." That works if and only if the factor loads *homogeneously* — every entity with the same exposure $\gamma_i = \gamma$. A time dummy subtracts the same amount from every unit at date $t$, but a factor with heterogeneous loadings hits each unit by a *different* amount, and the dummy removes only the cross-sectional mean of that effect, leaving the loading-weighted remainder to keep biasing the slopes. CCE's cross-section averages, entered per unit, let each entity absorb the factor through its *own* estimated coefficient — which is precisely the heterogeneity a time dummy assumes away. When you suspect heterogeneous exposure to common shocks (you almost always should), test for residual cross-sectional dependence (Pesaran's CD test) after the time dummies; if it survives, you need CCE, not another dummy.

## The frontier

**Dynamic CCE and the small-$T$ bias.** The basic CCE estimator assumes the regressors are exogenous given the factors; add a lagged dependent variable and it inherits a Nickell-type bias in short panels. Chudik and Pesaran (2015) show that augmenting with a growing number of *lags* of the cross-section averages restores consistency, and their cross-sectionally-augmented distributed-lag (CS-DL) estimator targets long-run coefficients directly. This is the active edge of applied panel time series — the estimator most current cross-country growth and finance papers actually run — and it sits in the roadmap's heterogeneous-panel tier.

**Pooled mean group: homogeneous long run, heterogeneous short run.** Pesaran, Shin and Smith (1999) split the difference between pooling and mean group with the pooled mean group (PMG) estimator: it forces the *long-run* relationship to be common across entities (economic theory often predicts a shared equilibrium — purchasing power parity, a common capital-output ratio) while letting the *short-run* adjustment dynamics differ. A Hausman test of PMG against MG is the standard way to ask whether the long-run homogeneity restriction is admissible. The estimator itself ships today as `tsecon.panel_pmg` (below); the Hausman companion test is still on the roadmap.

**Interactive fixed effects — the other way to kill a factor.** CCE purges the factor with observed averages; Bai's (2009) interactive fixed effects instead *estimates* the factors and loadings jointly with the slopes, by iterating principal components on the residuals against least squares on the slopes. The two approaches target the same enemy from opposite directions — CCE never estimates the factor, Bai insists on it — and they have complementary strengths: CCE is robust and needs no count of factors, Bai's is efficient when the factor count is known and correct. A mature panel module offers both and a specification comparison between them.

**Global VARs.** When entities are *linked* — trade flows, financial exposures — rather than merely sharing common shocks, the global VAR (GVAR) of Pesaran, Schuermann and Weiner (2004) models each country's VAR with foreign variables built from trade-weighted averages of the others, then stacks them into a single global system whose spillovers can be traced. It is the tool for questions of *transmission between* units (a Chinese slowdown's path through the world economy), where this chapter's estimators treat the cross-sectional dependence as a nuisance to purge rather than an object to model.

**Cross-sectional dependence as a testable primitive.** Underneath all of this sits the question of whether cross-sectional dependence is present at all and how strong it is. Pesaran's CD test (2004; and the bias-corrected 2021 version) and the exponent-of-cross-sectional-dependence literature turn "is there a common factor?" into a hypothesis you test rather than assume — the diagnostic that should precede the choice between a time dummy, CCE, and a full factor model.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| One average slope, confounders are time-invariant per entity | `panel_fe`, cluster SE | Within transformation removes fixed confounders; clustering handles serial correlation |
| Same, but a global cycle correlates entities' errors | `panel_fe`, `se_type="driscoll_kraay"` | Robust to cross-sectional dependence *and* serial correlation; needs large $T$ |
| Few entities (< ~30) | `panel_fe` + wild cluster bootstrap | Asymptotic clustered SE is too small with few clusters |
| Impulse response to a shock felt by all entities | `panel_lp` with Driscoll-Kraay SE | Pools the same experiment across units for tight IRF bands |
| Panel LP over long horizons in a short panel | `panel_lp(..., jackknife=True)` | Nickell bias grows like $O(h/T)$; split-panel correction tames it |
| Each entity has its own dynamics, and enough $T$ to fit them | `mean_group_var` | Pooling heterogeneous dynamics is *inconsistent*, not just inefficient |
| Heterogeneous *static* slopes, no common factor | `panel_mean_group(method="mg")` | Averages per-unit slopes; dispersion SE across units |
| Common unobserved factor drives both $y$ and $x$ | `panel_mean_group(method="cce")` | Cross-section averages span the factor space and purge the bias |
| Heterogeneous exposure to common shocks | CCE, *not* a time dummy | A time dummy removes only homogeneous factor effects |
| Lagged dependent variable *and* common factors | Dynamic CCE / CS-DL (roadmap) | Static CCE is biased with dynamics in short $T$ |
| Theory predicts a shared long-run relation, heterogeneous adjustment | `panel_pmg` | Restricts the long run, frees the short run (the Hausman test against MG is roadmap) |
| Entities linked by trade/finance, not just common shocks | Global VAR (roadmap) | Models spillovers *between* units rather than purging them |
| Short entities, cannot fit a per-unit model | Pool (bias and all) or shrink | Mean group's variance explodes when $T_i$ is small |

## What tsecon implements today

**Available now in Python** — everything this chapter's four runnable examples call, plus the frontier's pooled mean group:

- `tsecon.panel_fe(outcome, regressors, se_type="cluster", bandwidth=4.0)` — fixed-effects panel OLS with `outcome` shaped $N \times T$ and `regressors` shaped $k \times N \times T$; `se_type` is `"nonrobust"`, `"cluster"` (by entity), or `"driscoll_kraay"`, and `bandwidth` is the Driscoll-Kraay HAC lag length. Returns `params`, `bse`, `tvalues`, `se_type`.
- `tsecon.panel_lp(outcome, shock, horizon=8, n_lag_controls=2, se_type="driscoll_kraay", bandwidth=4.0, cumulative=False, jackknife=False)` — panel local projection of a common `shock` (length $T$) on an $N \times T$ outcome with entity fixed effects; `cumulative` returns the summed multiplier, `jackknife` applies the split-panel bias correction. Returns `irf`, `se`, `nobs`.
- `tsecon.mean_group_var(entities, lags=1, trend="c", horizon=10, response=0, impulse=0)` — Pesaran-Smith mean-group panel VAR over a *list* of per-entity $T_i \times k$ matrices (the $T_i$ may differ). Returns averaged `intercept`, `coefs`, and orthogonalized `orth_irfs` with dispersion-based `*_se`, plus the selected `irf_path`/`irf_path_se` for one `(response, impulse)` pair.
- `tsecon.panel_mean_group(ys, xs, method="mg")` — mean-group (`"mg"`) and CCE-MG (`"cce"`) for a heterogeneous panel, taking per-unit response vectors `ys` and $T_i \times k$ regressor matrices `xs`. Returns `coef`, `se`, `tstat`, the per-unit slope matrix `coef_per_unit`, `n_units`, and `k`. Validated to $\sim$1e-10 against a statsmodels per-unit-OLS golden (see `fixtures/tsecon-panelts.json`).
- `tsecon.panel_pmg(ys, xs)` — the pooled mean group ARDL(1,1) estimator (Pesaran-Shin-Smith 1999), taking the same per-unit `ys`/`xs` as `panel_mean_group`. It pools the *long-run* coefficients across units by maximum likelihood while leaving the error-correction speed and short-run dynamics unit-specific, and returns the pooled long-run `theta` with `theta_se`, the average adjustment speed `phi_bar`, the per-unit speeds `phi` and innovation variances `sigma2`, and the `loglik`.

These lean on machinery from earlier chapters you can reach for directly: `tsecon.ols` with `se_type="hac"` (chapter 3) is the single-series engine underneath the panel regressions; `tsecon.var_fit` and `tsecon.var_irf` (chapter 7) are the per-entity fits `mean_group_var` averages; and the whole panel-LP design is chapter 9's local projection with an entity dimension bolted on.

**Roadmap** — the heterogeneous-panel frontier is specified but not yet callable:

- **Dynamic CCE / CS-DL** (Chudik-Pesaran 2015) for panels with lagged dependent variables and long-run coefficients; the **Hausman test** of `panel_pmg` against MG that decides whether the long-run homogeneity restriction is admissible.
- **Interactive fixed effects** (Bai 2009) as the estimate-the-factor alternative to CCE's purge-the-factor, with a specification comparison between the two.
- **Cross-sectional dependence diagnostics** — Pesaran's CD test and the CSD-exponent — to turn "is there a common factor?" into a pre-estimation hypothesis; **panel unit-root and cointegration** tests under cross-sectional dependence; and the **global VAR** for modeling spillovers between linked entities rather than purging common shocks.

## Further reading

- **Pesaran, M. H. and R. Smith (1995), "Estimating Long-Run Relationships from Dynamic Heterogeneous Panels," *Journal of Econometrics*.** The result that pooling heterogeneous dynamics is inconsistent, and the mean-group estimator that answers it — the foundation under both `mean_group_var` and `panel_mean_group`.
- **Pesaran, M. H. (2006), "Estimation and Inference in Large Heterogeneous Panels with a Multifactor Error Structure," *Econometrica*.** The common correlated effects estimator: the cross-section-average trick that purges an unobserved common factor without ever estimating it. The chapter's climax in one paper.
- **Driscoll, J. C. and A. C. Kraay (1998), "Consistent Covariance Matrix Estimation with Spatially Dependent Panel Data," *Review of Economics and Statistics*.** The standard error that is robust to both serial and cross-sectional dependence — `panel_fe`'s and `panel_lp`'s macro-panel default.
- **Nickell, S. (1981), "Biases in Dynamic Models with Fixed Effects," *Econometrica*.** Why fixed effects and a lagged dependent variable do not mix in short panels — the bias that grows with the horizon in panel LP.
- **Jordà, Ò., M. Schularick and A. M. Taylor (2013), "When Credit Bites Back," *Journal of Money, Credit and Banking*.** The panel-local-projection macrohistory program — what follows credit booms across 14 countries and 140 years — and the template for `panel_lp` in practice.
- **Cameron, A. C. and D. L. Miller (2015), "A Practitioner's Guide to Cluster-Robust Inference," *Journal of Human Resources*.** The definitive practical treatment of clustered standard errors, the few-clusters problem, and the wild cluster bootstrap.
- **Chudik, A. and M. H. Pesaran (2015), "Common Correlated Effects Estimation of Heterogeneous Dynamic Panel Data Models with Weakly Exogenous Regressors," *Journal of Econometrics*.** Dynamic CCE — the lagged-cross-section-average fix that carries CCE from static to dynamic panels.
- **Pesaran, M. H., Y. Shin and R. P. Smith (1999), "Pooled Mean Group Estimation of Dynamic Heterogeneous Panels," *Journal of the American Statistical Association*.** The middle path: common long run, heterogeneous short run, and the estimator that imposes it.
- **Bai, J. (2009), "Panel Data Models with Interactive Fixed Effects," *Econometrica*.** The estimate-the-factor alternative to CCE, and the reference point for what "controlling for common factors" can mean.
- **Pesaran, M. H. (2004; 2021), "General Diagnostic Tests for Cross-Sectional Dependence in Panels," *Empirical Economics*.** The CD test — how to decide whether cross-sectional dependence is present before choosing how to handle it.
