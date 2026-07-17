# Chapter 13 — Nonlinear Dynamics: Regimes, Thresholds, and State-Dependent Responses

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** VAR impulse responses (chapter 7), structural identification (chapter 8), and local projections (chapter 9). The regime-switching *univariate* models of chapter 4 are helpful context — this chapter takes their central idea to systems.

**You will learn:**

- Why every impulse response you have computed so far assumed linearity, and the three economic situations where that assumption fails hardest
- How threshold, smooth-transition, and Markov-switching VARs let a system's dynamics change with the state of the economy
- Why nonlinear models break the very definition of "the" impulse response — and how the generalized impulse response (GIRF) repairs it by simulation
- How to estimate state-dependent, smooth-transition, and sign/size-dependent impulse responses with local projections, including a version you can run today
- The two honest caveats that discipline every nonlinear-IRF claim: shock-responsive states change the estimand, and regime IRFs are averages over histories, not counterfactuals

## The idea

Every method in this guide so far — ARMA forecasts, VAR impulse responses, structural identification, local projections, Bayesian VARs — has shared one assumption so quietly universal you may not have noticed it: **linearity**. A shock of size two does exactly twice what a shock of size one does. A negative shock is the mirror image of a positive one. And the economy's response to a shock today is the same whether the economy is booming or collapsing. Linearity is what let a VAR summarize all dynamics in a few lag matrices, and what let one impulse response function stand for *every* shock at *every* moment.

Economics gives at least three reasons to doubt it:

1. **Recessions are not expansions run backward.** Chapter 4 made this point for a single series; it bites harder for shock transmission. In a deep recession, idle workers and machines mean a demand stimulus can raise output without crowding anything out; at full employment the same stimulus mostly raises prices. The *transmission mechanism itself* differs by phase.
2. **The zero lower bound.** When the policy rate is pinned at zero, the central bank's usual response to shocks is switched off — so every other shock propagates differently than it would in normal times. A constraint that binds only sometimes is the purest form of nonlinearity: the system follows different rules on different sides of it.
3. **Financial stress amplifies.** When banks are well capitalized, a credit-market disturbance is absorbed; when balance sheets are fragile, the same disturbance triggers fire sales, margin calls, and a credit crunch that feeds back on itself. Small shocks stay small in calm regimes and metastasize in stressed ones.

Each of these turns on the same picture: the economy as a forest, the shock as a spark. On a wet forest floor the spark fizzles; in a drought it becomes a wildfire. Nothing about the spark changed — the **state of the system** determined the response. A linear model, forced to report one answer, splits the difference and describes neither the wet forest nor the dry one.

The question that made this literature famous, and this chapter's running example: **is the fiscal multiplier bigger in recessions?** If a dollar of government spending buys more output when resources are idle, then stimulus timing matters enormously and the "one multiplier" reported by any linear model is an average of two numbers you actually needed separately. Answering that question needs three things this chapter builds in order: models whose dynamics depend on the state, a definition of "impulse response" that survives the loss of linearity, and regressions that estimate state-dependent responses directly.

One clarification about the word "nonlinear," because you have already met one nonlinear model without the label. GARCH (chapter 6) makes the conditional *variance* a nonlinear function of the shock history — that is exactly why volatility clusters and why risk responds asymmetrically to news. This chapter moves the nonlinearity from the variance to the **conditional mean**: not "uncertainty depends on the state" but "the expected effect of a shock depends on the state." The two are cousins, and the simulation tools at this chapter's core serve both.

One scoping note before the tour. Chapter 4 already introduced the univariate members of this family — Markov-switching AR, where a latent regime switches the mean and dynamics of one series, and SETAR/STAR, where an observable threshold does. Everything there carries over: the Davies problem in linearity testing, the multimodal likelihoods, the regime probabilities as deliverable — and one warning from that chapter's roadmap becomes a theme here: for nonlinear models, even *forecasting* requires simulation, because iterating the fitted conditional mean forward plug-in style is biased beyond one step. The impulse-response version of that fact is this chapter's centerpiece. What chapter 4 could not cover is what this chapter owns: **systems** — nonlinear VARs and their multi-variable interactions — and above all **impulse responses**, the object nonlinearity damages most.

## Nonlinear VARs: three ways to let the system switch

A practitioner cares because these are the models behind every "in times of financial stress, the effect is three times larger" headline — and because which one you reach for is determined by one question: *what triggers the switch?*

**Threshold VAR (TVAR): the trigger is observable.** Pick a threshold variable $z_t$ — a credit spread, inflation, capacity utilization — and let the entire VAR switch when its lagged value crosses a threshold $\tau$:

$$
Y_t =
\begin{cases}
c_1 + A_1(L) Y_{t-1} + u_t, & u_t \sim (0, \Sigma_1), & \text{if } z_{t-d} \le \tau \\
c_2 + A_2(L) Y_{t-1} + u_t, & u_t \sim (0, \Sigma_2), & \text{if } z_{t-d} > \tau ,
\end{cases}
$$

two complete VARs — coefficients *and* shock covariances — glued at the threshold. Estimation is conditionally linear: for any candidate $(\tau, d)$, each regime is just OLS on its subsample, so you grid-search the threshold over the order statistics of $z$ (trimming 10–15% at each end so both regimes keep enough observations) and pick the residual-covariance-minimizing value. The classic application is Balke (2000): a TVAR in output, prices, the federal funds rate, and a credit spread, with the spread itself as threshold variable — shocks hitting in the tight-credit regime propagate visibly more strongly, the financial-amplification story made quantitative.

Testing whether two regimes beat one deserves its own paragraph, because it is where casual work goes wrong first. Under the linear null the threshold $\tau$ does not exist — it is a nuisance parameter identified only under the alternative, exactly the Davies problem from chapter 4 — so the Wald statistic for "regime 1 equals regime 2" at any *single* candidate threshold has no $\chi^2$ distribution, and the honest statistic is the **sup-Wald**: the largest Wald statistic over the whole threshold grid. Its null distribution is nonstandard and case-specific, so you simulate it — Hansen's (1996) fixed-regressor bootstrap re-draws the errors, recomputes the full grid search on each draw, and reads the p-value off the simulated distribution of the sup; Tsay's (1998) arranged-autoregression test is the cheap first screen. The bootstrap-inside-a-grid-search structure is expensive and embarrassingly parallel — the same Rust argument as everywhere else in this chapter — and skipping it in favor of a $\chi^2$ table overstates the evidence for nonlinearity every single time.

**Smooth-transition VAR (STVAR): the trigger is observable but gradual.** Auerbach and Gorodnichenko (2012) replaced the hard switch with a logistic weight on a standardized state variable $z_t$ (for them, a moving average of GDP growth):

$$
Y_t = F(z_{t-1}) \left[ c_R + A_R(L) Y_{t-1} \right] + \big(1 - F(z_{t-1})\big)\left[ c_E + A_E(L) Y_{t-1} \right] + u_t,
\qquad
F(z) = \frac{e^{-\gamma z}}{1 + e^{-\gamma z}} ,
$$

so the economy is always a *blend* of a recession system (weight $F \to 1$ when $z$ is very negative) and an expansion system, with the transition speed $\gamma$ controlling how sharp the blend is. Aggregates built from many heterogeneous units crossing thresholds at different times are naturally smooth, which is the economic case for STVAR over TVAR. The honest fine print: $\gamma$ is weakly identified — the likelihood is nearly flat in it — so Auerbach and Gorodnichenko *calibrate* $\gamma = 1.5$ to match the observed fraction of time the US spends in recession, a convention the literature has largely inherited. A number that important should always appear in your table notes.

**Markov-switching VAR (MS-VAR): the trigger is latent.** Let an unobserved regime $s_t \in \{1, \dots, m\}$ follow a Markov chain, and let it switch the VAR's parameters. Krolzig's (1997) taxonomy names the variants by *which* parameters switch: the intercept (MSI), the mean (MSM — subtly different, because after a regime switch the process jumps immediately to the new mean rather than transitioning gradually), the autoregressive matrices (MSA), the error covariance (MSH), and combinations thereof. One honest paragraph is all the taxonomy needs: the labels matter because they encode real modeling choices with different dynamic implications, but every member is estimated the same way — the Hamilton filter and Kim smoother inside an EM loop — and every member inherits the same fragilities at system scale that chapter 4 catalogued for one series, now with many more parameters per regime: multimodal likelihoods demanding multistart, label switching, regimes that degenerate onto a handful of outliers, and delicate standard errors. The latent regime is MS-VAR's whole appeal — the model *discovers* the phases and hands you their probabilities — and estimation fragility is the price of that appeal. No maintained implementation exists in Python or R today, which tells you something about the price. One more distinction to carry into the next section: MS-VAR impulse responses come in two flavors — *regime-conditional* IRFs, computed as if the chain sat in regime $j$ forever, and *generalized* IRFs that integrate over the chain's own transitions. The first is a useful description of within-regime dynamics; only the second answers "what does a shock do," because the chain does not sit still.

**Interacted VAR (IVAR), briefly.** Instead of discrete regimes, let VAR coefficients vary continuously with an observed conditioning variable — coefficients interacted with the level of public debt, or the exchange-rate regime (Towbin and Weber 2013; Sá, Towbin and Wieladek 2014). It is the VAR cousin of the interaction regressions you already know from microeconometrics: cheap to estimate (OLS with interaction terms), and the natural tool when the "state" is a continuous policy-relevant variable rather than a phase.

Which trigger, then? The choice is less aesthetic than it looks:

- **Reach for TVAR or STVAR** when you can *name* the state variable in advance — a spread, slack, inflation. That naming is an identifying assumption, not a convenience: the model only ever compares dynamics across the regimes your variable defines. Threshold when the switch is institutional or mechanical (a bound, a rule, a covenant); smooth when the aggregate blends many units crossing thresholds at different times.
- **Reach for MS-VAR** when the phases themselves are what you want the data to discover, and the regime probabilities are part of the deliverable — at the cost of the heaviest estimation burden in this chapter.
- **Reach for IVAR** when the conditioning variable is continuous and the question is "how does transmission change *with* debt/openness/stress," not "which of two worlds are we in."
- **Stay linear, and say so,** when regime visits are scarce. The arithmetic is brutal: a two-regime VAR in four variables with four lags carries $2\left[n(np + 1) + n(n+1)/2\right] = 156$ parameters, and if the recession regime owns 40 of your 250 quarterly observations, 78 of those parameters rest on 40 data points. This is why credible TVARs are small and lag-short, why STVAR's smooth weighting (every observation informs both regimes, just unequally) is partly a statistical survival strategy, and why the nonlinear-LP designs of the next sections — which add a handful of interaction parameters rather than a second VAR — carry so much of the applied literature.

> **⚠ Common mistake — frozen-regime impulse responses.** Having estimated a TVAR or STVAR, it is tempting to compute two ordinary linear IRFs — one from $(c_1, A_1)$, one from $(c_2, A_2)$ — and present them as "the response in recessions" and "the response in expansions." Those are the responses of an economy *locked* in one regime forever. But the model you just estimated says regimes change — indeed, a large enough shock can itself push the economy across the threshold, and that endogenous switching is often the economically interesting part (a stimulus that ends the recession has a different total effect than one that doesn't). Frozen-regime IRFs answer a question the model was built to reject. What replaces them is the subject of the next section.

## The generalized impulse response: one definition that survives

This is the chapter's load-bearing section. In a linear VAR, "the impulse response" is well defined because three invariances hold automatically: the response to a shock of size $2\delta$ is twice the response to $\delta$ (**size invariance**), the response to $-\delta$ is the mirror image of the response to $+\delta$ (**sign invariance**), and the response is the same whatever the economy was doing when the shock hit (**history invariance**). One function of the horizon summarizes everything.

In a nonlinear model, all three fail — not as a pathology, but as the entire point. A spark in a drought is not half of two sparks in a drought, and it is nothing like a spark in the rain. So the object "the impulse response" has to be rebuilt from its definition. Koop, Pesaran and Potter (1996) did it by returning to what an IRF fundamentally is — a difference between two conditional expectations:

$$
GIRF(h, \delta, \omega_{t-1}) \;=\; E\!\left[\, y_{t+h} \mid \varepsilon_t = \delta,\; \omega_{t-1} \right] \;-\; E\!\left[\, y_{t+h} \mid \omega_{t-1} \right],
$$

the expected path of the economy given that a shock of size $\delta$ hit at time $t$, minus the expected path given no such intervention — both conditional on the **history** $\omega_{t-1}$, the state of the world when the shock arrived. Read the three arguments left to right and you can see the linear invariances being deliberately given up: the **generalized impulse response** depends on the horizon (as always), on the shock's size *and sign* $\delta$, and on the history $\omega_{t-1}$. In a linear model the definition collapses back to the familiar object: writing the VAR's moving-average representation with coefficient matrices $\Psi_h$,

$$
E\!\left[\, y_{t+h} \mid \varepsilon_t = \delta,\; \omega_{t-1} \right] - E\!\left[\, y_{t+h} \mid \omega_{t-1} \right] = \Psi_h\, \delta \qquad \text{for every } \omega_{t-1},
$$

the history cancels between the two conditional expectations and $\delta$ factors out — the sanity check that the GIRF is a generalization of the chapter-7 IRF, not a replacement for it.

Neither conditional expectation has a formula in a nonlinear model, so the GIRF is computed by **simulation**. The algorithm is worth internalizing because every nonlinear-VAR paper you read runs some version of it:

1. **Draw histories.** Take actual histories $\omega_{t-1}$ from the sample — for a regime-specific GIRF, the histories observed in that regime (all recession dates, say).
2. **Draw shock paths.** For each history, draw many future shock sequences $\varepsilon_{t+1}, \dots, \varepsilon_{t+H}$ from the model's residuals (or their distribution).
3. **Simulate in pairs.** For each (history, shock path) pair, simulate the model forward twice: once with $\varepsilon_t = \delta$ imposed at time $t$, once with $\varepsilon_t$ drawn like any other period — *using the same future shocks in both runs*, so the only difference between the two simulated paths is the intervention. Average the paired differences over shock paths: that is the history-conditional GIRF.
4. **Average over histories.** Averaging the history-conditional GIRFs over the recession histories gives "the response in recessions"; over all histories, the unconditional GIRF. Keeping the distribution *across* histories, rather than just its mean, is itself informative — a wide spread says the response depends heavily on initial conditions, and reporting only the average hides that.

The algorithm is small enough to read in full, so here it is end to end, on a two-regime AR(1) whose **persistence** switches — $\phi = 0.9$ below the threshold, $0.4$ above. (Later in the chapter, a shock's *impact* will switch with the regime; here it is the propagation that switches, because that is what makes histories, sizes, and signs matter beyond the first period.)

```python
import numpy as np

def step(y_prev, eps):                          # the model: all the engine needs
    phi = 0.9 if y_prev < 0.0 else 0.4          # persistent below, transient above
    return phi * y_prev + eps

def girf(history, delta, H=8, R=20_000, seed=0):
    rng = np.random.default_rng(seed)
    e0 = rng.standard_normal(R)                 # baseline's own time-t shock
    e = rng.standard_normal((R, H))             # future shocks, shared by both runs
    out = np.zeros(H + 1)
    for r in range(R):
        y_s = step(history, delta)              # shocked path: eps_t = delta imposed
        y_b = step(history, e0[r])              # baseline path: eps_t drawn
        out[0] += y_s - y_b
        for h in range(H):                      # same future shocks in both paths
            y_s, y_b = step(y_s, e[r, h]), step(y_b, e[r, h])
            out[h + 1] += y_s - y_b
    return out / R

print(np.round(girf(-1.5, 1.0), 2))       # [1. 0.92 0.75 0.61 0.49 0.4  0.33 0.27 0.22]
print(np.round(girf(+1.5, 1.0), 2))       # [1. 0.48 0.3  0.21 0.16 0.12 0.1  0.08 0.06]
print(np.round(girf(-1.5, 3.0) / 3, 2))   # [1. 0.63 0.45 0.34 0.27 0.21 0.17 0.14 0.12]
print(np.round(-girf(-1.5, -3.0) / 3, 2)) # [1. 0.89 0.78 0.68 0.58 0.49 0.42 0.35 0.29]
```

Four GIRFs, and all three linear invariances break in front of you:

- **History** (rows 1 vs 2): the *same unit shock* has decayed to 0.92 after one period when it lands in the persistent low regime, but to 0.48 when it lands in the high one. One period out, the response differs by a factor of two purely because of where the economy stood.
- **Size** (rows 1 vs 3): per unit of shock, a $3\sigma$ impulse from the same recession history dies much faster than a $1\sigma$ one — because the big shock ($0.9 \times (-1.5) + 3 = +1.65$) lifts the economy across the threshold *immediately*, and the rest of its life is spent decaying at the transient regime's rate. The shock changed the regime; the regime changed the shock's own propagation.
- **Sign** (rows 3 vs 4): mirror the $3\sigma$ shock and the mirrored response is nowhere near a mirror image — the negative shock digs the economy *deeper* into the persistent regime, so per unit it lingers roughly twice as long at medium horizons (0.68 vs 0.34 at $h = 3$). In a stressed economy, bad news lasts longer than good news of the same size: the model produced the asymmetry the linear world assumes away.

Step 3's trick — common random numbers across the paired runs — plus antithetic variates in step 2 are classic variance-reduction devices, and they matter because the raw computational bill is (histories) × (shock-path replications) × (horizons) × (cost per simulation step), *before* you add a bootstrap loop around the whole thing for confidence bands. In an interpreted loop this takes hours, which is why published GIRF bands are often thinner on replications than anyone would like and why the method has a reputation for pain. The structure, though, is embarrassingly parallel — thousands of independent simulations that never communicate — which is precisely the shape of problem a compiled core with counter-based parallel RNG dispatches in seconds. This is the same speed story the bootstrap chapters have been telling, and it is why tsecon's roadmap treats the KPP engine as shared infrastructure to be built once and reused by every nonlinear model in the library: given any model that can simulate forward from a state, the engine computes $GIRF(h, \delta, \text{history set})$ with variance reduction and reproducible parallelism.

If the toy above ran in a quarter of a second, where do the hours go? Scale each factor honestly: an estimated TVAR replaces the one-line `step` with a matrix recursion over several variables; one conditioning history becomes hundreds (every recession date in the sample); and — the multiplier that hurts — the GIRF you computed is a *point estimate*, conditional on the estimated parameters. Its uncertainty has two layers: **simulation noise**, which more draws kill (with $R = 20{,}000$ paired paths it is already negligible above), and **estimation uncertainty**, which no number of draws touches — the thresholds, coefficients, and covariances are themselves estimated. Honest GIRF bands therefore wrap a bootstrap around the entire engine: re-estimate the model on each bootstrap sample, rerun the full history-by-history simulation on each re-estimate, and read bands off the distribution. That is the algorithm above times a few hundred model refits, and it is the step published papers most often skimp on — thin replications, or bands that account for simulation noise only. It is also, being independent replications all the way down, exactly the workload a parallel compiled core is for.

> **⚠ Common mistake — computing one GIRF and scaling it.** In a linear world you compute the one-standard-deviation IRF and mentally rescale it to any shock you like; the habit dies hard. In a nonlinear model, $GIRF(h, 2\delta, \omega) \ne 2\,GIRF(h, \delta, \omega)$ and $GIRF(h, -\delta, \omega) \ne -GIRF(h, \delta, \omega)$ — indeed, *checking* those equalities is a useful diagnostic for how nonlinear your fitted model actually is. Size and sign are arguments of the function now: a serious nonlinear-IRF exercise simulates the sizes and signs it wants to talk about ($\pm 1\sigma$, $\pm 2\sigma$), and reports them separately. If they all turn out proportional, that is a finding (the nonlinearity is mild at business-cycle shock sizes), not a wasted computation.

## Nonlinear local projections: state dependence the regression way

Chapter 9 ended with a preview of this material; here is the full treatment. The LP framework's great trick is that state dependence costs almost nothing: interactions in a regression, machinery you have known since your first econometrics course.

**State-dependent LP.** Interact *everything* — impulse, controls, intercept — with a lagged regime indicator $I_{t-1}$:

$$
y_{t+h} = I_{t-1}\!\left[\alpha_{R,h} + \beta_{R,h}\, s_t + \gamma_{R,h}' w_t\right] + \left(1 - I_{t-1}\right)\!\left[\alpha_{E,h} + \beta_{E,h}\, s_t + \gamma_{E,h}' w_t\right] + \xi_{t+h},
$$

one regression per horizon as always, now yielding *two* impulse responses: $\{\beta_{R,h}\}$ for periods that began in the regime (recession) and $\{\beta_{E,h}\}$ for periods that did not. Everything extends by the same interaction move: for state-dependent *multipliers*, interact the one-step cumulative IV regression of chapter 9 with the state — instrument included — so the regime-specific multiplier comes out of a single regression with correct IV inference:

$$
\sum_{j=0}^{h} y_{t+j} = I_{t-1}\!\left[\mu_{R,h} + \mathcal{M}_{R,h} \sum_{j=0}^{h} g_{t+j} + \gamma_{R,h}' w_t\right] + \left(1 - I_{t-1}\right)\!\left[\,\cdot\,\right]_{E} + u_{t+h},
$$

with the cumulated spending in each block instrumented by the state-interacted shock. The choice of $I$ is itself a research decision with teeth: Ramey and Zubairy define slack as unemployment above a threshold (6.5% in their baseline) partly because it is observable in real time — no two-sided filtering, no revisions smuggling the future into the state — and they show the results care about that construction.

This is the Ramey-Zubairy (2018) design, and it produced the modern answer to this chapter's motivating question: across 125+ years of US data, **little evidence that fiscal multipliers exceed 1 even in high-slack states** — multipliers of roughly 0.6–0.7 in both regimes. That finding reversed Auerbach and Gorodnichenko's earlier smooth-transition result of recession multipliers well above 1, and the gap between the two papers is a case study in how much the methodological choices of chapters 9 and 13 — estimator, sample, state-variable construction — can move a headline number.

**Smooth-transition LP.** Replace the on/off indicator with the Auerbach-Gorodnichenko logistic weight from the STVAR section:

$$
y_{t+h} = F(z_{t-1})\!\left[\alpha_{R,h} + \beta_{R,h}\, s_t + \gamma_{R,h}' w_t\right] + \big(1 - F(z_{t-1})\big)\!\left[\alpha_{E,h} + \beta_{E,h}\, s_t + \gamma_{E,h}' w_t\right] + \xi_{t+h},
$$

so the estimated response varies continuously with the depth of the recession rather than snapping between two values. The STVAR fine print transfers wholesale: the same weak identification of $\gamma$ (calibrate it, report it), the same warning about state variables built from centered moving averages (a two-sided filter leaks *future* information into the regime classification — the standard critique of the original AG state variable), plus one new regression-level trap: when $F(z)$ rarely approaches 0 or 1 in your sample, the two weighted regressor blocks are nearly collinear and the regime-specific coefficients are estimated off very little independent variation — wide standard errors are the design telling you the sample never clearly visited one of the regimes.

**Sign and size dependence.** Nonlinearity in the *shock* rather than the state uses the same interaction logic. For asymmetry, split the shock into its positive and negative parts and let each have its own coefficient in one regression:

$$
y_{t+h} = \alpha_h + \beta^{+}_h \max(s_t, 0) + \beta^{-}_h \min(s_t, 0) + \gamma_h' w_t + \xi_{t+h},
$$

so $\beta^{+}_h \ne \beta^{-}_h$ is monetary policy "pushing on a string" (Tenreyro and Thwaites 2016: contractionary shocks bite, expansionary ones do much less, especially in recessions). For size dependence, add polynomial terms in the shock:

$$
y_{t+h} = \alpha_h + \beta_{1,h}\, s_t + \beta_{2,h}\, s_t^2 + \gamma_h' w_t + \xi_{t+h},
$$

so the marginal response per unit of shock, $\beta_{1,h} + 2\beta_{2,h}\, s_t$, varies with the shock's magnitude — Ben Zeev, Ramey and Zubairy (2023) use exactly this logic to ask whether large fiscal shocks work differently per dollar than small ones. Report the implied response at *named* shock sizes ($1\sigma$, $2\sigma$), never the raw coefficients alone: a quadratic coefficient is uninterpretable in isolation, and extrapolating the polynomial beyond the shock sizes the sample actually contains is fitting a curve where there is no data. Testing $\beta^{+} = \beta^{-}$, or $\beta_{2,h} = 0$, across a stretch of horizons is a joint hypothesis and needs the joint cross-horizon covariance from chapter 9 — pointwise comparisons of two bands do not answer it.

Two caveats discipline all of this, and they are the part of the chapter most worth remembering:

1. **The state must not respond to the shock.** Lagging the indicator ($I_{t-1}$, never $I_t$) is necessary but not sufficient. Gonçalves, Herrera, Kilian and Pesavento (2021; 2024) show that if the shock can move the economy across regimes *within the response horizon* — a stimulus large enough to end the recession it was launched in — then $\beta_{R,h}$ no longer estimates "the response conditional on being in a recession"; the estimand itself quietly changes into a blend of regime responses weighted by shock-induced transition probabilities. The problem is worst exactly when the nonlinearity is strong, which is when you cared most. Two checks you owe the reader: regress the future state indicator $I_{t+h}$ on today's shock — if the shock predicts regime switching at the horizons you report, the warning is live, not theoretical — and exploit the fact that the distortion is second-order for shocks too small to move the regime, so sensitivity of your regime IRFs to the shock's size is itself a diagnostic.
2. **A regime IRF is an average over histories, not a counterfactual for one economy.** $\beta_{R,h}$ averages the response over all the recession histories in your sample — mild recessions and deep ones, early exits and late ones — including all the regime switching that actually happened after each shock. It is the LP analogue of the KPP average-over-histories GIRF, and it inherits the same interpretation: it is *not* the response of an economy held in recession for $h$ periods (that was the frozen-regime mistake), and it is not the response of *your* economy in *this* recession. Nonlinear LPs estimate honest averages; the temptation is to read them as sharp counterfactuals.

Here is the whole design, runnable today against tsecon's primitives. The DGP has two regimes switched by whether *last* period's $y$ is below a threshold — so the state is predetermined, as caveat 1 demands — and the same shock hits **twice as hard** in the low ("recession") regime:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(42)
T, H, phi, tau = 600, 12, 0.6, 0.0

# DGP: two regimes, switched by whether LAST period's y is below tau.
# In the "recession" regime the same shock hits twice as hard.
eps = rng.standard_normal(T)          # the observed shock
eta = rng.standard_normal(T)          # everything else that moves y
y = np.zeros(T)
for t in range(1, T):
    rec = y[t - 1] < tau              # regime fixed BEFORE the shock arrives
    b = 2.0 if rec else 1.0           # true impact: 2 in recession, 1 outside
    y[t] = phi * y[t - 1] + b * eps[t] + 0.5 * eta[t]

I = (y < tau).astype(float)           # I[t-1] is the lagged, predetermined state

irf_rec = np.zeros(H + 1); se_rec = np.zeros(H + 1)
irf_exp = np.zeros(H + 1); se_exp = np.zeros(H + 1)
for h in range(H + 1):
    t = np.arange(1, T - h)
    yh, s = y[t + h], eps[t]
    Il, ylag = I[t - 1], y[t - 1]     # lagged indicator, lagged level
    X = np.column_stack([
        Il,       Il * s,       Il * ylag,        # recession block
        1 - Il, (1 - Il) * s, (1 - Il) * ylag,    # expansion block
    ])                                # the two indicators ARE the intercepts
    r = tsecon.ols(yh, X, se_type="hac", maxlags=h + 1)
    irf_rec[h], se_rec[h] = r["params"][1], r["bse"][1]
    irf_exp[h], se_exp[h] = r["params"][4], r["bse"][4]

print(np.round(irf_rec[:7], 2))   # [ 1.96  1.38  0.78  0.37  0.07  0.03 -0.1 ]
print(np.round(irf_exp[:7], 2))   # [ 0.95  0.74  0.5   0.34  0.14  0.01 -0.05]
gap = irf_rec[0] - irf_exp[0]     # 1.00: the state dependence, recovered
print(gap / np.sqrt(se_rec[0]**2 + se_exp[0]**2))   # t ~ 23 on the impact gap
```

Everything in the loop is chapter-9 machinery — shift, align, regress, collect — with one new move: the design matrix carries a full copy of every regressor for each regime, and the two indicator columns replace the single intercept. The impact responses land on the truth (1.96 vs 0.95 against a true 2 vs 1), and the gap is measured with a t-statistic of 23: with a predetermined state and an observed shock, state dependence really is almost free.

Now look at the *later* horizons, because the code just demonstrated caveat 2 live. The true impact multipliers are 2 and 1, but by horizon 4 the two estimated IRFs have nearly converged. No bug: a recession in this DGP is exited quickly (the autoregression pulls $y$ back toward zero, and roughly 43% of periods are below the threshold), so the horizon-4 response *averaged over recession histories* includes many paths that left the recession long before period 4 — exactly what the estimand promises, and exactly not "the response of an economy that stays in recession." If you want the response along frozen or specified regime paths, that is a different object, and it needs the model-based GIRF machinery of the previous section.

From here to a published-style result is bookkeeping you already know: cumulate each regime's IRF for state-dependent cumulative effects (or better, run the one-step cumulative regression interacted with the state, as Ramey and Zubairy do for multipliers), test the regime *gap* over a stretch of horizons with the joint covariance rather than eyeballing two bands, and apply chapter 9's inference upgrades — lag augmentation works unchanged on the interacted design, since each regime block just gains its own lag terms.

> **⚠ Common mistake — hunting asymmetry with split samples.** To test whether positive and negative shocks act differently, do *not* run one LP on the positive-shock episodes and another on the negative-shock episodes: the two regressions then condition on different histories, the samples are selected by the very variable under study, and the comparison confounds sign dependence with state dependence. Enter $\max(s_t, 0)$ and $\min(s_t, 0)$ together in one regression on the full sample, and test $\beta^+_h = -\beta^-_h$ (note the sign: symmetry means the responses are mirror images) with the joint covariance. The same logic applies to size: one regression with polynomial terms, not subsamples binned by shock magnitude.

## LP versus VAR when the world is nonlinear

Chapter 9's verdict — LPs and VARs estimate the same object, choose by bias-variance — needs an amendment here, because nonlinearity redistributes the costs.

**LPs make state dependence almost free.** An interaction term per regime, HAC or lag-augmented inference as before, done — the code above is thirty lines. The nonlinearity is *local to the regression*: you never specify how regimes evolve, so you cannot misspecify it. That agnosticism is also the limit: because the LP never models the transition process, it can only ever deliver the average-over-histories object, with no way to trace shock-induced regime switching or to ask what happens if the recession persists. The regression answers the question it answers.

**Nonlinear VARs make state dependence structural but simulation-heavy.** A TVAR or MS-VAR is a complete law of motion, so it can answer the questions LPs cannot: GIRFs at any shock size and sign, responses conditional on specific histories, the endogenous regime dynamics themselves. The price is everything this chapter has catalogued — threshold grid searches, sup-test bootstraps, EM fragility, and the full KPP simulation bill for every IRF you want, with a bootstrap around it for bands.

**Both must confront the GIRF issues.** This is the point most easily missed. Choosing LPs does not exempt you from Koop-Pesaran-Potter — it just picks, silently, one particular point in the GIRF's argument space: the average over the histories observed in each state, at the average shock size in the sample, with sign and size dependence assumed away unless you added those terms. The nonlinear VAR makes the same choices explicitly, as simulation settings.

The honest workflow, mirroring chapter 9's dual-reporting advice:

1. **Establish the linear baseline first** — a linear LP and a linear VAR on the same specification. Every nonlinear claim should be a measured departure from a baseline the reader has seen.
2. **Test for the nonlinearity before modeling it** — a sup-type threshold test, or simply the interaction terms' joint significance across horizons. "The two regime IRFs look different" is not a test.
3. **Estimate the state-dependent LP** with a lagged, defensible state variable; run the shock-predicts-state diagnostic; report which average over which histories the coefficients are.
4. **Escalate to a nonlinear VAR plus KPP GIRFs only when the question demands the structural objects** — specific-history responses, large-shock or asymmetric-shock counterfactuals, endogenous regime paths — and validate its regime-average GIRFs against the LP estimates from step 3, exactly the way chapter 9 overlaid LP and VAR lines.

## The frontier

**Time-varying responses without regimes.** Regimes discretize instability; sometimes transmission just *drifts*. Inoue, Rossi and Wang (2024) estimate kernel-weighted local projections that deliver the IRF as a smooth function of calendar time — monetary transmission before and after 1980, or across the ZLB decade, without asserting a switch date. The relationship to this chapter's models is worth stating plainly: a two-regime model asserts that all instability lives in one observable (or one latent chain), while the time-varying LP lets *anything* drift and asks the data to say when. The inference problem — joint bands over a time × horizon surface, boundary bias at the sample's ends — is the stacked-covariance machinery of chapter 9 taken up a dimension, and no maintained package in any language implements it yet.

**Functional shocks.** Some shocks are not numbers but *curves*: a QE announcement moves the whole yield curve — level, slope, curvature at once — and compressing that into "the" scalar shock size discards exactly the variation unconventional policy is made of. Inoue and Rossi's functional-shock approach parameterizes the announcement-day shift of the curve (a Nelson-Siegel or functional-principal-component basis) and runs LPs on the basis coefficients jointly, so the impulse response becomes a *surface*: response per point of the original curve, per horizon — with the honest caveat that the basis choice is itself an identification choice. This lives in the LP roadmap's frontier tier ([Module 07](../roadmap/07-local-projections.md)) alongside the time-varying LP.

**Nonlinear estimands and policy counterfactuals.** McKay and Wolf (2023) showed how to combine empirically identified IRFs to *multiple* policy shocks into counterfactual policy-rule outcomes that are robust to the Lucas critique under stated conditions — the beginning of a bridge from the reduced-form toolkit to the questions policy institutions actually ask ("what if the central bank had responded twice as aggressively?"). The conditions are the substance: you need enough distinct identified shocks to span the policy rule's degrees of freedom, which is a data-collection agenda as much as an econometric one. In parallel, Kolesár and Plagborg-Møller (2025) characterize exactly what weighted average of heterogeneous responses a misspecified nonlinear LP recovers — the theory that turns this chapter's caveat 2 from a warning into a computable diagnostic, scheduled in the roadmap as a diagnostic attached to every nonlinear LP result.

**Deep-learning IRFs, with skepticism.** Neural networks can in principle learn arbitrary nonlinear dynamics, and once fitted they are exactly the kind of simulable model the KPP engine consumes — so papers now report GIRFs from fitted networks. The honest assessment mirrors chapter 12's: macro samples are a few hundred observations; a function class that can represent any nonlinearity can overfit any sample; the regularization that prevents this is chosen by validation schemes whose time-series validity is itself delicate; and the resulting GIRFs come with essentially no inference theory — bands, when reported at all, ignore the model-selection step entirely. Where these methods have earned keep is as *flexibility audits*: if a heavily regularized network, honestly cross-validated in time, finds materially different responses than your TVAR, that is evidence the parametric regime structure is wrong — evidence worth having. As headline estimates, they are not yet close to displacing the interaction term and the threshold grid.

## Which method when

| Situation | Reach for | Because |
|---|---|---|
| First pass at "does the response differ by regime?" | State-dependent LP with a lagged indicator | Interactions are almost free; no transition model to misspecify |
| Response should vary *continuously* with the cycle | Smooth-transition LP (or STVAR), γ calibrated and reported | Aggregates blend regimes; hard switches are the wrong shape |
| Do positive and negative shocks act differently? | One LP with $\max(s,0)$ and $\min(s,0)$ entered together | Split samples confound sign with state; joint test across horizons |
| Do big shocks act differently per unit than small ones? | LP with polynomial shock terms; report response at named sizes | Size invariance is a linear-world assumption, now testable |
| Named observable trigger for system dynamics (spread, inflation) | TVAR with sup-test linearity inference, GIRFs by simulation | Interpretable threshold; complete law of motion for counterfactuals |
| Latent phases you want the system to discover | MS-VAR (Krolzig taxonomy) | Regime probabilities are the deliverable; budget for estimation fragility |
| Continuous conditioning variable (debt level, openness) | Interacted VAR | Regression-cheap; state-conditional IRFs at chosen values |
| Any IRF from any nonlinear model | KPP generalized IRFs — simulate sizes, signs, history sets | Size/sign/history invariance is gone; the GIRF arguments are the analysis |
| Response of an economy that *stays* in the regime | Model-based GIRF along conditioned regime paths | LPs only average over observed histories; this needs a law of motion |
| The state might respond to the shock | Lag the state, then test sensitivity; read Gonçalves et al. | If shocks move the regime within the horizon, the estimand changes |
| Transmission drifts without discrete regimes | Time-varying LP (Inoue-Rossi-Wang) | Regimes discretize what may be a slow drift |
| Nonlinearity lives in the variance, not the mean | GARCH family (`garch_fit`, chapter 6) | Volatility clustering and news asymmetry are nonlinear dynamics you can fit today |
| Short sample, few regime episodes | Stay linear, say so | Two recessions cannot identify a recession-specific system |

## What tsecon implements today

**Available now in Python** — everything this chapter's two runnable examples need:

- `tsecon.ols(y, X, se_type=...)` with `"hac"` and `"hc0"`/`"hc1"` — the engine of the hand-rolled state-dependent LP above, which runs today exactly as printed; chapter 9's lag-augmentation upgrade applies to the interacted design unchanged
- `tsecon.long_run_variance` — the HAC machinery underneath
- `tsecon.var_fit`, `tsecon.var_irf`, `tsecon.var_fevd`, `tsecon.var_forecast` — the *linear* VAR benchmark every nonlinear claim should be compared against
- `tsecon.philox_uniforms`, `tsecon.bootstrap_indices`, `tsecon.optimal_block_length` — counter-based parallel RNG and block-resampling primitives: the exact ingredients the KPP simulation engine is built from. The GIRF demo above deliberately used nothing but NumPy — the *concept* is thirty lines; what the roadmap engine adds is estimated-model integration, the bootstrap-over-refits bands, variance reduction, and the speed to make all of that routine
- `tsecon.garch_fit` — worth pausing on: GARCH (chapter 6) is itself a nonlinear time series model — the conditional *variance* is a nonlinear function of the shock history, which is why volatility clusters and why a 2σ shock changes future dynamics more than twice as much as a 1σ shock. If you have fit a GARCH model, you have already estimated nonlinear dynamics; this chapter moves the nonlinearity from the variance to the conditional mean

One disambiguation, since the name collision is unfortunate: today's `tsecon.hamilton_filter` is Hamilton's (2018) regression-based *detrending* filter (the modern HP-filter alternative, chapter 7's territory) — not the Hamilton (1989) *Markov-switching* filter this chapter discussed. The regime-switching filter arrives with the roadmap items below.

**Roadmap** — none of this chapter's named models is callable yet; the specs are written and tiered:

- **Univariate regime models** ([docs/roadmap/02-univariate.md](../roadmap/02-univariate.md)): Markov-switching AR validated against Hamilton's published GNP estimates (Tier 1); SETAR/TAR with Hansen threshold inference, Hansen/Tsay linearity tests, and the full Teräsvirta STAR modeling cycle (Tier 2)
- **Nonlinear VARs and the GIRF engine** ([docs/roadmap/04-multivariate.md](../roadmap/04-multivariate.md), Tier 3): TVAR with bootstrapped sup-tests (validated against Balke 2000), STVAR with the calibrated-γ convention (validated against the Auerbach-Gorodnichenko replication files), MS-VAR across the Krolzig taxonomy, and the shared Koop-Pesaran-Potter GIRF simulation engine — antithetic variates, common random numbers, reproducible parallelism — contributed to the library's foundations for reuse by every nonlinear model; interacted VARs sit in Tier 4
- **Nonlinear local projections** ([docs/roadmap/07-local-projections.md](../roadmap/07-local-projections.md)): state-dependent LP with a regime indicator, including the IV version for state-dependent fiscal multipliers and the Gonçalves-et-al. estimand warning, is Tier 1 core — validated against the Ramey-Zubairy state-dependent multipliers; smooth-transition LP and sign/size-dependent LP with joint asymmetry tests are Tier 2; time-varying LP and functional shocks are the frontier tier

## Further reading

- **Koop, G., M. H. Pesaran and S. M. Potter (1996), "Impulse Response Analysis in Nonlinear Multivariate Models," *Journal of Econometrics*.** The generalized impulse response: the definition, the simulation algorithm, and the demonstration that "the" IRF is a linear-world luxury.
- **Teräsvirta, T. (1994), "Specification, Estimation, and Evaluation of Smooth Transition Autoregressive Models," *Journal of the American Statistical Association*.** The smooth-transition modeling cycle; the univariate foundation under every STVAR and smooth-transition LP.
- **Hansen, B. E. (2011), "Threshold Autoregression in Economics," *Statistics and Its Interface*.** The threshold-model survey: estimation, the non-standard inference for thresholds, and the testing problem, by the author of most of its solutions.
- **Balke, N. S. (2000), "Credit and Economic Activity: Credit Regimes and Nonlinear Propagation of Shocks," *Review of Economics and Statistics*.** The classic threshold VAR application — financial-stress amplification made quantitative, and the standard GIRF validation target.
- **Auerbach, A. J. and Y. Gorodnichenko (2012), "Measuring the Output Responses to Fiscal Policy," *American Economic Journal: Economic Policy*.** Smooth-transition state dependence and the recession-multiplier result that set the agenda for a decade.
- **Ramey, V. A. and S. Zubairy (2018), "Government Spending Multipliers in Good Times and in Bad," *Journal of Political Economy*.** The state-dependent LP-IV benchmark and the revised answer to this chapter's motivating question.
- **Gonçalves, S., A. M. Herrera, L. Kilian and E. Pesavento (2021; 2024), *Journal of Econometrics*.** When the state responds to the shock, the estimand changes — the papers behind this chapter's first caveat, and required reading before any state-dependent claim.
- **Tenreyro, S. and G. Thwaites (2016), "Pushing on a String: US Monetary Policy Is Less Powerful in Recessions," *American Economic Journal: Macroeconomics*.** Sign and state asymmetry in monetary transmission; the template for the sign/size LP design.
- **Krolzig, H.-M. (1997), *Markov-Switching Vector Autoregressions*, Springer.** The MS-VAR taxonomy and the EM machinery; still the reference for what switches and what it implies.
- **Kilian, L. and H. Lütkepohl (2017), *Structural Vector Autoregressive Analysis*, Cambridge University Press, ch. 18.** The textbook treatment of nonlinear structural analysis — TVAR, STVAR, MS-VAR, and GIRFs in one place, with the inference caveats attached.
