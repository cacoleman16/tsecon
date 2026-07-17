# Chapter 8 — Structural Identification: From Correlation to Cause

> Part of [The tsecon Guide to Time Series Econometrics](README.md). Chapters mirror the library's modules; code runs against the current Python API unless marked otherwise.

**Prerequisites:** Chapter on VARs (fitting, impulse responses, forecast-error variance decompositions) and comfort with OLS and covariance matrices.

**You will learn:**

- Why the shocks a VAR estimates are correlated mixtures of the shocks economists care about, and why no amount of data fixes that on its own
- What "identification" means precisely: the rotation problem, and how many assumptions you owe before an impulse response is causal
- The major identification schemes — recursive, long-run, sign, heteroskedasticity, narrative, and instrument-based — each with its maintained assumptions and its classic failure mode
- How the modern instrument-based approaches connect VARs to local projections, and why the two estimate the same thing
- How to choose a scheme for your question, and what honest inference looks like when identification is a set rather than a point

## The idea

Here is the question that launched forty years of macroeconometrics: **what does a monetary tightening do to output?** The Federal Reserve raises its policy rate; a year later, unemployment is higher. Did the rate hike cause that? Or did the Fed raise rates *because* it saw inflation and an overheating economy coming — in which case the correlation between rates and future output tells you about the Fed's reaction, not about the Fed's power?

This is the simultaneity problem, and in macro data it is everywhere. The policy rate responds to the economy within the quarter; the economy responds to the policy rate within the quarter. Demand shifts move prices and quantities; so do supply shifts. When you fit a VAR — the workhorse multivariate model from the previous chapters — you get residuals, one per equation per period. It is tempting to call the residual in the interest-rate equation "the monetary policy shock." It is not. It is the part of the rate that the VAR's lags could not predict — a *mixture* of the Fed's genuine surprise decisions and the Fed's same-quarter reaction to demand surprises, supply surprises, financial surprises, and everything else that hit the economy that quarter. The residuals across equations are correlated with each other precisely because they share these underlying causes.

Think of it like listening to a recording of an orchestra made with one microphone per seat. Every microphone picks up every instrument — the violin mic mostly violin, but also the timpani two rows back. The VAR residuals are the microphone tracks: correlated blends. The structural shocks — the pure monetary shock, the pure demand shock — are the individual instruments. **Identification** is the problem of unmixing the tracks. And here is the uncomfortable truth this chapter is built around: the data alone cannot do the unmixing. The recording is consistent with many different seatings of the orchestra. You must bring outside information — economic theory, institutional knowledge, historical documents, or extra data such as a measured instrument — to pin down which unmixing is the true one. Every identification scheme in this chapter is a different kind of outside information, and each one buys you causality at the price of an assumption you must be prepared to defend.

This chapter is the guide's tour of that territory — and it maps directly onto the library module that tsecon considers its headline differentiator, because no maintained Python package covers modern SVAR identification today.

## Reduced form versus structure: the rotation problem

A practitioner cares about this section because it is the accounting identity of the whole subject: it tells you exactly how many assumptions you owe, and why "let the data decide" is not an available option.

Write the reduced-form VAR from the multivariate chapter as

$$
y_t = A_1 y_{t-1} + \cdots + A_p y_{t-p} + u_t, \qquad \mathbb{E}[u_t u_t'] = \Sigma_u,
$$

where $y_t$ is an $n \times 1$ vector of observables (say output growth, inflation, and a policy rate), the $A_j$ are estimated lag-coefficient matrices, and $u_t$ is the vector of one-step-ahead forecast errors — the *reduced-form innovations*. Everything here is estimable by OLS and, given enough data, known.

The **structural VAR (SVAR)** asserts that these innovations are linear combinations of underlying economic shocks:

$$
u_t = B \, \varepsilon_t, \qquad \mathbb{E}[\varepsilon_t \varepsilon_t'] = I_n,
$$

where $\varepsilon_t$ collects the structural shocks — mutually uncorrelated, unit variance by normalization — and the $n \times n$ **impact matrix** $B$ says how much each structural shock moves each variable within the period. Column $j$ of $B$ is the impact response of the whole system to shock $j$; combined with the estimated lag dynamics it generates the structural impulse response functions (IRFs) that answer causal questions.

Now count. The data reveal $\Sigma_u = BB'$. A symmetric $n \times n$ covariance matrix contains $n(n+1)/2$ distinct numbers. The matrix $B$ contains $n^2$ unknowns. The shortfall is

$$
n^2 - \frac{n(n+1)}{2} = \frac{n(n-1)}{2}
$$

numbers that the data cannot deliver. For a 3-variable VAR that is 3 missing restrictions; for a 7-variable monetary system, 21.

The geometric version of the same fact is the **rotation problem**. Suppose $B$ satisfies $BB' = \Sigma_u$. Take any orthogonal matrix $Q$ (a rotation: $QQ' = I$). Then $\tilde{B} = BQ$ satisfies

$$
\tilde{B}\tilde{B}' = B Q Q' B' = BB' = \Sigma_u
$$

equally well. Every rotation of a valid impact matrix is another valid impact matrix, producing *identical* reduced-form fit, identical forecasts, identical likelihood — and different causal stories. The set of $n \times n$ orthogonal matrices has dimension exactly $n(n-1)/2$: the rotation problem *is* the counting shortfall, seen geometrically. (For $n = 2$ that dimension is 1 — every rotation is a single angle $\theta$ — which is why bivariate examples are the standard way to *see* an identified set: as $\theta$ sweeps the circle, the implied impact matrices trace out every observationally equivalent model.) Two researchers with the same data and the same VAR can report opposite signs for the effect of money on output, and no statistical test can adjudicate between them, because their models are observationally equivalent.

The equivalence is easy to watch happen. Fit a VAR, take any admissible impact matrix, rotate it by any angle you like, and confirm that the implied residual covariance has not moved:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(0)
mix = np.array([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.3, 0.4, 1.0]])
y = rng.standard_normal((300, 3)) @ mix.T        # three correlated series

r = tsecon.var_fit(y, lags=1)
Sigma = np.array(r["sigma_u"])
B = np.linalg.cholesky(Sigma)                    # one admissible impact matrix

theta = 0.7                                      # any angle at all
Q = np.array([[ np.cos(theta), np.sin(theta), 0.0],
              [-np.sin(theta), np.cos(theta), 0.0],
              [ 0.0,           0.0,           1.0]])
B_tilde = B @ Q                                  # a different causal story...
print(np.abs(B_tilde @ B_tilde.T - Sigma).max()) # ...identical fit: ~1e-16
```

`B` and `B_tilde` imply different shocks, different impulse responses, different history — and the data cannot tell them apart.

Identification therefore means: impose at least $n(n-1)/2$ credible restrictions that select the true $B$ (point identification), or impose weaker restrictions that shrink the set of admissible $B$'s and be honest that you end with a set (set identification). Everything that follows is a catalog of where those restrictions can legitimately come from.

> ⚠ **Common mistake:** treating the reduced-form innovations $u_t$ themselves as shocks — "the residual in the interest-rate equation is the policy shock." Unless $B$ happens to be diagonal (it never is; the residuals are correlated), each $u_{it}$ is a blend of all $n$ structural shocks, and IRFs computed from unorthogonalized innovations answer no causal question at all.

## Recursive identification: what an ordering buys you

The oldest and still the most common answer, from Sims (1980): assume the impact matrix is **lower triangular**. Economically, you assert a *causal ordering within the period*: the first variable responds to no other shock on impact, the second responds contemporaneously only to the first, and so on. The last variable — often the policy rate — responds to everything on impact, but nothing responds to *it* until next period.

This is attractive because it is exactly enough: a lower-triangular $B$ has $n(n+1)/2$ free elements, matching what $\Sigma_u$ delivers, and there is a unique lower-triangular matrix with positive diagonal satisfying $BB' = \Sigma_u$ — the **Cholesky factor**:

$$
B = \operatorname{chol}(\Sigma_u).
$$

No optimization, no sampling; one matrix decomposition and you have point-identified shocks. In the classic monetary VAR of Christiano, Eichenbaum, and Evans (1999), slow-moving variables (output, prices) are ordered before the federal funds rate: the identifying claim is that production and pricing decisions cannot react to a policy surprise within the same quarter — plausible if firms set production plans and prices in advance — while the Fed *can* react to output and prices within the quarter, which is exactly what its information set allows. When the timing claim is institutionally grounded like this, Cholesky is defensible. When the ordering is chosen by habit, it is an unexamined assumption wearing a lab coat.

tsecon's `var_irf` computes orthogonalized (Cholesky) IRFs today. Here is the whole loop on synthetic data with a known recursive structure, so you can watch identification succeed when its assumption is true:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(42)
T, n = 400, 3
eps = rng.standard_normal((T, n))     # structural shocks: demand, output, policy

B0 = np.array([[1.0, 0.0, 0.0],       # demand reacts to nothing on impact
               [0.5, 1.0, 0.0],       # output absorbs demand within the period
               [0.3, 0.4, 1.0]])      # the policy rate leans against both
A1 = np.array([[0.5, 0.0, -0.2],
               [0.1, 0.4,  0.0],
               [0.0, 0.1,  0.5]])

y = np.zeros((T, n))
for t in range(1, T):
    y[t] = A1 @ y[t - 1] + B0 @ eps[t]

irf = tsecon.var_irf(y, lags=1, horizon=16)   # irf[h][i][j]: response of i to shock j
print(np.round(np.array(irf[0]), 2))          # impact matrix -> recovers B0
```

Because the true impact matrix is lower triangular and the ordering in the data matches it, the estimated impact step `irf[0]` reproduces `B0` (the Cholesky factor is unique). The gallery's VAR section runs exactly this kind of system — demand moves first, output responds with a lag, the policy rate leans against both — and the estimated IRF grid recovers the built-in story:

![Cholesky-identified impulse response grid](../examples/img/06-var-irf.png)

The companion decomposition question — *what share of each variable's forecast errors does each shock explain?* — is the forecast-error variance decomposition, also identified by the same Cholesky factor:

```python
fevd = np.array(tsecon.var_fevd(y, lags=1, horizon=16))  # [variable][horizon][shock]
print(np.round(fevd[:, 8, :], 2))   # shares at horizon 8; each row sums to 1
```

![Forecast-error variance decomposition](../examples/img/07-var-fevd.png)

Now the honesty check: the ordering *is* the identification. Permute the variables and you get a different causal story from the same data:

```python
perm = [2, 1, 0]                                    # policy ordered first
y_perm = np.ascontiguousarray(y[:, perm])
irf_perm = tsecon.var_irf(y_perm, lags=1, horizon=16)
print(np.round(np.array(irf_perm[0]), 2))           # a different impact matrix
```

Both runs fit the data identically. Only the first matches how the data were generated — and with real data, nothing inside the sample tells you which ordering that is. The module spec plans cheap ordering sweeps precisely so this sensitivity is a one-liner rather than an afterthought.

> ⚠ **Common mistake:** "I tried all orderings and the IRF barely changed, so identification doesn't matter here." Ordering-insensitivity only means the residual correlations are small; it is evidence about $\Sigma_u$, not about causality. Conversely, "robustness across orderings" cannot rescue a system where no recursive ordering is economically defensible — with financial variables like stock returns or spreads in the VAR, *every* slow-fast ordering is wrong, because asset prices react to everything within the day and the macro variables plausibly react to them. That is exactly the case that pushes you toward heteroskedasticity or external instruments below.

## Long-run restrictions: the Blanchard-Quah decomposition

Sometimes theory is silent about timing within a quarter but loud about the *long run*. The canonical example is Blanchard and Quah (1989): virtually any macro model implies that **demand shocks have no permanent effect on the level of output** — in the long run, output is pinned down by technology and supply — while **supply shocks do have permanent effects**. That is a restriction you can impose without taking any stand on who moves first within the quarter.

Formally, let $A(L) = I - A_1 L - \cdots - A_p L^p$, so $A(1) = I - A_1 - \cdots - A_p$. The cumulative (long-run) effect of the structural shocks on the levels of the variables is

$$
\Theta(1) = A(1)^{-1} B .
$$

Blanchard-Quah imposes that $\Theta(1)$ is lower triangular — in the bivariate (output growth, unemployment) system: the second ("demand") shock has zero long-run effect on output. That again supplies exactly $n(n-1)/2$ restrictions, and the solution is closed-form:

$$
\Theta(1) = \operatorname{chol}\!\left( A(1)^{-1} \Sigma_u A(1)^{-1\prime} \right), \qquad B = A(1)\, \Theta(1).
$$

It is a Cholesky decomposition applied at the infinite horizon instead of at impact — an elegant trick, and the origin of the whole supply/demand decomposition literature.

The scheme's weakness is equally famous. $A(1)^{-1}$ blows up as the VAR's largest roots approach one — precisely the region macro data live in — so small estimation errors in the lag coefficients become enormous errors in the long-run matrix. Faust and Leeper (1997) showed that in finite samples the long-run restriction can have essentially no bite: sizable short-run misidentification is consistent with the long-run constraint holding. The practical readings: check the estimated VAR's characteristic roots before trusting a long-run scheme (in the statsmodels convention `var_fit` follows, stability requires every root to lie *outside* the unit circle — a root modulus close to 1 is the warning sign), prefer the vector-error-correction formulation when cointegration is plausible, and treat Blanchard-Quah conclusions as fragile whenever persistence is high. tsecon's implementation (Module 06) will warn on near-unit roots by default, because the failure is silent otherwise.

*Roadmap preview — this API lands with Module 06:*

```python
svar = tsecon.svar(data, lags=8)
bq = svar.identify_long_run(permanent=["supply"], transitory=["demand"])
bq.irf(horizon=40)      # cumulated correctly for differenced variables
bq.diagnostics          # near-unit-root warning fires when a root modulus nears 1
```

> ⚠ **Common mistake:** forgetting that with differenced variables in the VAR, the interesting IRF is the *cumulated* one (the response of the level), and cumulating after orthogonalization is not the same as orthogonalizing cumulated responses if done carelessly. The library cumulates inside the IRF object so the level response of output to a demand shock visibly returns to zero — the restriction you imposed — as a built-in sanity check.

## Sign restrictions: honest bands, not points

Recursive and long-run schemes deliver a point — one $B$ — by imposing hard zeros that many economists find too strong. Uhlig (2005) proposed a humbler kind of outside information: *signs*. A contractionary monetary policy shock, whatever else it does, should raise the federal funds rate, lower prices, and lower nonborrowed reserves for a few quarters. Notice what is deliberately left out: the response of *output* is unrestricted, because that is the question. Any rotation whose IRFs violate the signs is rejected; every rotation that satisfies them is kept.

The algorithm is direct. Given the reduced form, the admissible impact matrices are

$$
\mathcal{B} = \left\{ \operatorname{chol}(\Sigma_u)\, Q \;:\; Q \in \mathcal{O}(n), \ \text{IRFs of } \operatorname{chol}(\Sigma_u) Q \text{ satisfy the sign restrictions} \right\}.
$$

Draw random rotations $Q$ uniformly (from the *Haar distribution* — the uniform distribution on the orthogonal group), keep the ones whose IRFs pass, and summarize the survivors. Because signs are inequalities, not equalities, they do not pin down a point: **you end with a set of models, not one**. This is called **set identification**, and it changes what honest reporting means. The output is a band of IRFs that are all fully consistent with both the data and your assumptions — the width of that band *is* a finding. If the output response to a sign-identified monetary shock spans zero, that is the paper's result, not a nuisance to be narrowed by prettier plotting. (Uhlig's own punchline was exactly this: under agnostic sign restrictions, the contractionary effect of money on output is far less certain than the Cholesky consensus suggested.)

Two honesty rules come with the method. First, **pointwise medians mix models**: the horizon-3 median and the horizon-8 median of the accepted draws generally come from *different* rotations, so the "median IRF" is not the IRF of any admissible model. Fry and Pagan (2011) proposed reporting the single accepted rotation closest to the pointwise medians (the median-target rotation) alongside the band; tsecon makes that the documented default companion output. Second — the caveat that a decade of applied work learned the hard way — **the uniform prior on rotations is not uninformative about the things you care about**. Baumeister and Hamilton (2015) showed that the Haar prior on $Q$ induces a definitely-not-flat prior on impulse responses and variance shares, and because the data cannot distinguish points *within* the identified set, that prior never washes out, no matter the sample size. Part of any "posterior band" from sign restrictions is Haar-prior artifact rather than evidence. The remedies are to plot prior against posterior (if they overlap heavily, the data barely spoke), to put priors on economically meaningful structural parameters instead (Baumeister-Hamilton's own program), or to report prior-robust bounds — the Giacomini-Kitagawa approach in the frontier section. tsecon's design treats these diagnostics as mandatory output, not options.

*Roadmap preview — this API lands with Module 06:*

```python
mon = svar.identify_signs(
    shock="monetary",
    restrictions={"ffr": "+", "prices": "-", "nbr": "-"},
    horizons=range(0, 6),
    draws=100_000,                  # embarrassingly parallel in the Rust core
)
mon.acceptance_rate                 # an identification diagnostic in itself
mon.irf_bands(alpha=0.32)           # pointwise bands, labeled honestly
mon.median_target()                 # Fry-Pagan single-model summary
mon.prior_posterior_overlay("gdp")  # how much is Haar artifact?
```

> ⚠ **Common mistake:** stacking on sign restrictions to narrow the band without watching the acceptance rate. Acceptance decays roughly exponentially in the number of restrictions; an acceptance rate of $10^{-5}$ means your "posterior" is a handful of surviving draws and the restrictions may be close to mutually inconsistent. The acceptance rate is itself an identification diagnostic — tsecon prints it with every fit. Also: combining *zero* restrictions with sign restrictions naively (impose the zeros, then sign-check) samples from the wrong distribution; the correct algorithm with importance weights is Arias, Rubio-Ramírez, and Waggoner (2018), and the library ships only the corrected version.

## Identification from variance shifts: heteroskedasticity as an instrument

All the schemes so far spend economic assumptions. Rigobon (2003) noticed that the *statistical* properties of the data can sometimes pay instead. The intuition is worth having in pictures. Simultaneity is a problem because a cloud of (price, quantity) points traced out by both supply and demand shocks lets you fit neither curve. But suppose you know that during a crisis window the *demand* shock variance triples while the supply curve and the supply shock variance stay put. In the crisis subsample, the data cloud stretches *along the supply curve* — demand shocks trace it out for you. Comparing the calm and crisis covariance matrices reveals the slope. The variance shift did the work an instrument usually does: it moved one curve while leaving the other fixed.

Formally, with two known regimes and a constant impact matrix,

$$
\Sigma_1 = B \Lambda_1 B', \qquad \Sigma_2 = B \Lambda_2 B',
$$

where $\Lambda_1, \Lambda_2$ are diagonal structural-shock variance matrices. Two covariance matrices give $n(n+1)$ equations for $n^2 + 2n$ unknowns minus normalizations — enough to identify $B$ (up to sign and column ordering) provided the *relative* variances $\lambda_{2i}/\lambda_{1i}$ are distinct across shocks: the columns of $B$ are the generalized eigenvectors solving $\Sigma_2 v = \lambda \Sigma_1 v$. No zeros, no signs, no instruments — identification bought purely from second moments shifting.

The event-study variant deserves its own mention because it is quietly everywhere in monetary economics. Rigobon and Sack (2003, 2004) compare the covariance of asset prices and policy rates on FOMC *announcement days* against neighboring control days: the policy-shock variance jumps on announcement days while everything else's variance stays roughly flat, so the announcement-day/control-day covariance *difference* isolates the policy response. This dominates a naive event study whenever announcement days also carry background news — the event study attributes all announcement-day movement to policy; the heteroskedasticity estimator attributes only the *extra variance*.

The price is a different set of maintained assumptions: the regime dates are known and correct, the impact coefficients are genuinely constant across regimes, and the relative variances genuinely differ. And the method delivers *statistically* identified shocks with no labels attached — shock 2 is "the one whose variance rose most," not "the monetary shock," until you attach economic meaning via sign patterns or correlation with external series. Labeling is a real step, easy to get silently wrong; tsecon's design makes unlabeled statistical shocks impossible to plot without a warning.

> ⚠ **Common mistake:** proceeding when the relative variances barely differ across regimes. Identification strength here is measured by the separation of the generalized eigenvalues; when two shocks' relative variances are similar, their columns of $B$ are near-unidentified and estimates are garbage with tight-looking bogus standard errors. The equality-of-relative-variances test must run automatically and gate the output — statistical identification fails quietly.

## Narrative identification: reading the record

The most labor-intensive outside information is also the most transparent: *read the documents*. Romer and Romer (2004) went through FOMC minutes and the Fed's internal Greenbook forecasts, meeting by meeting, and constructed a series of monetary policy shocks defined as the change in the intended funds rate *not* explained by the Fed's own forecasts of output and inflation — policy motion purged, by hand and by regression, of the systematic reaction to the economy. Ramey (2011) built a defense-news series by reading Business Week and other sources to date the moments when expectations of future military spending changed — capturing fiscal *news* when it arrives, rather than when spending shows up in the accounts, which matters because anticipated spending is already in agents' behavior long before it is in the data. Ramey and Zubairy (2018) extended the military-news series back to 1889 for state-dependent multiplier analysis, and Romer and Romer (2010) did the narrative exercise for tax changes, classifying each legislated change by motive so that only exogenously motivated changes count.

A narrative series is not itself an identification scheme — it is a measured proxy for a structural shock, and it enters the toolkit in three standard ways: as a direct regressor in a local projection (Chapter 7's method: regress $y_{t+h}$ on the shock, horizon by horizon), as an external instrument in a proxy SVAR (next section), or ordered first in a recursive VAR as an internal instrument (the section after). The identifying assumption has simply moved location: instead of a zero in a matrix, it is the claim that the narrative series is correlated with the true shock and uncorrelated with everything else hitting the economy — which you can now debate by reading the same documents the authors read. That transparency is the method's great virtue.

Its weaknesses are measurement error (hand-coded series are noisy proxies, which is precisely why the instrument machinery below exists), potential predictability (early narrative series turned out to be partially forecastable — a red flag for exogeneity), and instability: Hoesch, Rossi, and Sekhposyan (2023) document that the strength of the Romer-Romer and high-frequency instruments varies substantially over time, so a full-sample first-stage F can mask decades where the instrument is uninformative. There is also a sobering empirical fact to absorb before choosing a shock measure: Ramey's (2016) Handbook chapter runs the leading monetary shock series through identical specifications and shows that the estimated effects of "a monetary shock" differ materially across measures — the choice of identification is a first-order modeling decision, not a robustness footnote. tsecon's roadmap ships the canonical narrative series as documented, versioned loaders with vintage metadata and the standard usage recipes, plus rolling instrument-relevance diagnostics, and gates them against reproducing the Ramey handbook comparison figures.

> ⚠ **Common mistake:** regressing outcomes directly on a narrative series and reading the coefficient as "the effect of a one-unit structural shock." A narrative series is a *proxy* — correlated with the shock, contaminated by measurement error — so the raw coefficient is attenuated and its units are arbitrary. Use the series as an instrument (proxy SVAR or LP-IV) or apply a unit-effect normalization, and mind *anticipation*: for fiscal policy especially, legislated changes are known before they take effect, so dating the shock at implementation rather than at news arrival puts the shock in the wrong period and biases every horizon (the entire reason the Ramey news series exists).

## External instruments: the proxy SVAR

The proxy SVAR (also called SVAR-IV or external instruments) is the modern applied default for monetary and tax questions, developed by Stock and Watson and by Mertens and Ravn (2013). It formalizes how a measured shock series — narrative or otherwise — identifies a structural column without any restriction on the other columns.

Suppose you have an instrument $z_t$ (say, high-frequency futures surprises) for the structural shock of interest $\varepsilon_{1t}$. The two IV conditions are exactly the ones you know from cross-sectional econometrics:

$$
\mathbb{E}[z_t \varepsilon_{1t}] = \phi \neq 0 \quad \text{(relevance)}, \qquad \mathbb{E}[z_t \varepsilon_{jt}] = 0 \ \ \text{for } j \neq 1 \quad \text{(exogeneity)}.
$$

Then the covariance of the instrument with the reduced-form innovations reveals the shock's impact column up to scale:

$$
\mathbb{E}[u_t z_t] = B\, \mathbb{E}[\varepsilon_t z_t] = \phi \, b_1,
$$

so $b_1 \propto \mathbb{E}[u_t z_t]$ — regress each reduced-form residual on the instrument and read off the relative impact responses, fixing the scale with a unit-effect normalization (e.g., the shock raises the policy rate by 25 basis points on impact). One column of $B$ is identified; nothing need be assumed about the rest — which is all you need if one shock is the question.

The canonical example is Gertler and Karadi (2015): identify monetary policy shocks using **high-frequency surprises** — the change in federal funds futures prices in a 30-minute window around FOMC announcements — as the instrument in a monthly VAR with output, prices, the policy rate, and a credit spread. The logic is clean: within 30 minutes of the statement, essentially no other macro news arrives, so the futures move *is* the market's reading of the policy surprise (relevance), and it cannot be correlated with that month's other structural shocks (exogeneity). Their headline finding — modest rate moves produce outsized responses in credit spreads — is a result the recursive scheme could not deliver, because a VAR containing fast-moving financial variables admits no defensible Cholesky ordering, exactly the failure flagged earlier.

The dangers are the IV dangers, amplified by time series. **Weak instruments**: many published proxies have first-stage strength that is modest at best, and with a weak proxy the normalized IRFs have heavy-tailed distributions and conventional confidence bands are junk. The honest response is weak-instrument-robust inference — Montiel Olea, Stock, and Watson (2021) invert Anderson-Rubin statistics horizon by horizon, and the resulting confidence set can be an interval, a union of two rays, or the entire real line; when it is the whole line, the data are telling you the instrument cannot answer the question at that horizon, and software should render that honestly rather than clip it. **Invalid bootstraps**: the wild bootstrap that most published proxy-SVAR bands used is asymptotically invalid here (Jentsch and Lunsford 2019); the moving-block bootstrap is the valid replacement, and the corrected Mertens-Ravn tax-multiplier intervals widen substantially. No mainstream package implements the robust confidence sets today — it is a headline item of tsecon's identification module.

*Roadmap preview — this API lands with Module 06:*

```python
gk = svar.identify_proxy(instrument=hf_surprises, target="ffr",
                         normalize=("ffr", 0.25))
gk.report_card            # effective first-stage F, rolling relevance, alignment
gk.irf_bands(method="moving_block")            # Jentsch-Lunsford valid bands
gk.irf_bands(method="ar_robust")               # Montiel Olea-Stock-Watson sets;
                                               # may be rays or the whole line — shown as such
```

> ⚠ **Common mistake:** reporting delta-method or wild-bootstrap bands with a first-stage F of 4 and calling it identification. Check instrument strength *first*, use weak-IV-robust sets when it is questionable, and watch the unit-effect normalization: dividing by a near-zero impact coefficient makes IRF quantiles explode in some draws — a fragility tsecon detects and reports rather than averaging away. A subtler trap: proxies typically cover a shorter sample than the VAR and contain gaps; silently truncating to the overlap misaligns instrument and residuals, a classic proxy-SVAR bug the library catches at ingestion.

## Internal instruments and the local-projection connection

There is a strikingly simple alternative to the proxy machinery: put the instrument *inside* the VAR, ordered first, and use plain Cholesky. The orthogonalized shock to the instrument equation is then the identified structural shock, and the IRFs of the other variables to it are the causal responses. This "internal instrument" recipe, used informally since Ramey (2011), turns out to have a deep justification. Plagborg-Møller and Wolf (2021) proved that, in population, **a VAR with the instrument ordered first and a local projection on the instrument estimate the exact same impulse responses**. Local projections (Chapter 7) and VARs are not competing identification schemes — they are two estimators of the same object, differing only in finite-sample bias-variance trade-offs: the VAR extrapolates dynamics from its lag structure (lower variance, biased if the lag length is wrong at long horizons), while the LP estimates each horizon freshly (robust, noisier). The identification content lives entirely in the instrument; the estimator choice is a separate, second decision.

The internal-instrument route has one more quietly important property: it remains valid under **noninvertibility** — the situation where the structural shock cannot be recovered from current and past values of the VAR's variables (classic with fiscal foresight and news shocks: agents act on anticipated policy the econometrician's variables haven't registered yet). Ordering the measured shock series first sidesteps recovery entirely, because the shock is *in* the system. In tsecon's design, internal-versus-external instrument is a one-argument switch, and the docs teach the equivalence so users stop framing "LP versus VAR" as an identification debate.

You can run the internal-instrument pattern today with the existing API — here with the true shock as the "instrument" so the mechanics are transparent:

```python
import numpy as np
import tsecon

rng = np.random.default_rng(7)
T = 500
shock = rng.standard_normal(T)               # the measured shock series (e.g., narrative)
x = np.zeros(T); z = np.zeros(T)
for t in range(1, T):
    x[t] = 0.6 * x[t - 1] + 0.8 * shock[t] + 0.5 * rng.standard_normal()
    z[t] = 0.4 * z[t - 1] + 0.3 * x[t - 1] - 0.4 * shock[t] + 0.5 * rng.standard_normal()

data = np.column_stack([shock, x, z])        # instrument ordered FIRST
irf = tsecon.var_irf(data, lags=2, horizon=12)
resp_x = [irf[h][1][0] for h in range(13)]   # causal response of x to the shock
resp_z = [irf[h][2][0] for h in range(13)]
print(np.round(resp_x, 2))
```

Only the first column of the IRF array — the responses *to* the instrument's shock — carries a causal interpretation; the rest of the Cholesky structure is incidental. In real work the instrument is noisy, which attenuates the impact scale; the unit-effect normalization (divide the responses by the impact response of the policy variable) restores interpretability, exactly as in the proxy SVAR.

> ⚠ **Common mistake:** treating LP-versus-VAR disagreement at long horizons as evidence about identification. Same instrument, same identification — the divergence is the estimators' different bias-variance behavior at horizons the sample barely informs. The frontier fix is lag-augmented or bias-aware inference, not switching identification schemes.

## The frontier

The research frontier of this field is mostly about *honesty at the edges* — inference that admits what the data cannot say — and it is where tsecon's identification module stakes its claim, since almost none of it has a software home.

**Robust Bayes for set-identified models.** Giacomini and Kitagawa (2021) resolve the Haar-prior problem head-on: keep the standard prior on the reduced form (where data genuinely update beliefs), but replace the single prior on rotations with the *class of all priors* consistent with the identified set, and report the range of posterior means and robustified credible regions across the class. The output separates, draw by draw, what the data plus the restrictions imply from what the rotation prior was inventing. If the robust band is dramatically wider than the Haar-prior band, the discrepancy *is* the Haar artifact, made visible. Computationally it demands minimizing and maximizing IRFs over the admissible rotations for every reduced-form draw — a nonconvex optimization on the orthogonal group that the roadmap attacks with analytic active-set solutions where available and manifold optimization with many random starts elsewhere; this is precisely where a parallel Rust kernel changes what is feasible. The frequentist mirror image — confidence sets for the identified set itself — is Gafarov, Meier, and Montiel Olea (2018) and Granziera, Moon, and Schorfheide (2018), and Moon and Schorfheide (2012) is the classic warning that Bayesian and frequentist answers *diverge* under set identification even asymptotically.

**Correct zero-plus-sign sampling.** Arias, Rubio-Ramírez, and Waggoner (2018) proved that the widespread practice of imposing zero restrictions by construction and then checking signs samples from the wrong distribution, distorting inference in published papers, and supplied the corrected algorithm with volume-element importance weights. The corrected version is the only code path tsecon will ship, with importance-weight effective sample size monitored and reported.

**Sharper and stranger restrictions.** Narrative *sign* restrictions (Antolín-Díaz and Rubio-Ramírez 2018) constrain the model to agree with history — e.g., the monetary shock in October 1979 was contractionary, and it was the dominant driver of the funds rate move that month — and shrink sign-identified sets dramatically, at the cost of heavy-tailed importance weights that demand ESS discipline. Bounds on elasticities (Kilian and Murphy 2012) and on forecast-error-variance shares (Volpicella 2022) are further inequality families. The architectural bet of tsecon's module is that all of these are *composable*: zeros shape the null spaces the rotations are drawn from, signs, narratives, and FEVD bounds act as accept-reject or importance weights, proxies enter as moment conditions — any mix constraining the same rotation space, with the diagnostics (acceptance rates, ESS, prior-posterior overlays) emitted automatically because in set-identified settings *the diagnostics are the inference*. That composability exists in no package today, in any language; it is the module's centerpiece and the reason the roadmap calls identification the library's headline differentiator.

**Statistical identification, stress-tested.** Beyond two-regime Rigobon: Markov-switching variances, smooth-transition and GARCH covariances, stochastic-volatility identification (Bertsche and Braun 2022; Lewis 2021), and non-Gaussianity — mutually independent non-Gaussian shocks identify $B$ up to permutation and scale by the ICA theorem, failing if more than one shock is Gaussian. The honest open problem is that these methods rest on strong independence assumptions that are themselves economic claims (Montiel Olea, Plagborg-Møller, and Qian 2022); Drautzburg and Wright (2023) show how to relax independence into bounds. The library's documentation obligation, stated in the spec, is to teach when *not* to trust statistical identification.

**Open problems** worth knowing are open: inference on set-identified IRFs that is simultaneously sharp, uniformly valid, and computationally routine does not yet exist; weak-proxy-robust *Bayesian* inference is nascent (Giacomini, Kitagawa, and Read 2022); and the invertibility question — whether the shocks you seek are recoverable from the variables you observe — has diagnostics (Forni and Gambetti 2014) but no fully satisfying resolution inside pure SVARs.

## Which method when

The decision framework, compressed: start from the **question** (which shock?), then ask what **outside information you can defend** — a timing convention, a long-run neutrality, qualitative signs, documented variance regimes, an archival record, or a measured surprise series — and then match the **inference** to the identification (point schemes get standard bands; set schemes get set-valued honesty; instrument schemes get strength diagnostics first). If you have a credible instrument, use it — instrument-based schemes make the weakest assumptions about the rest of the system. If you have nothing but signs, accept that your answer is a set.

| Situation | Reach for | Because |
|---|---|---|
| A defensible within-period timing convention (slow macro variables, policy moves last) | Recursive/Cholesky (`var_irf` today) | Exactly identifying, transparent, one decomposition — and the assumption is auditable: either the timing story is institutionally true or it is not |
| Financial variables in the system (spreads, asset prices) | External or internal instruments | No recursive ordering is defensible when some variables react to everything within the period |
| Theory speaks about permanent versus transitory effects, not timing | Blanchard-Quah long-run restrictions | Imposes neutrality you believe anyway; but check the VAR's roots first — fragile near unit roots (Faust-Leeper) |
| Only weak qualitative beliefs ("contractionary policy doesn't raise prices") | Sign restrictions + Fry-Pagan median-target + prior-posterior overlay | Set identification matches the actual state of knowledge; the band width is the finding |
| Sign restrictions plus a few credible zeros | Arias-Rubio-Ramírez-Waggoner zero+sign | The only correct sampler for the combination; naive zeroing distorts inference |
| Documented variance regimes (crisis dates, announcement days) | Rigobon heteroskedasticity | Variance shifts substitute for economic restrictions; test that relative variances actually differ |
| An archival record isolating exogenous policy actions | Narrative series (Romer-Romer, Ramey news) as regressor, proxy, or internal instrument | Transparent, debatable identification; watch measurement error and time-varying strength |
| A high-frequency surprise series or other measured proxy | Proxy SVAR with weak-IV-robust bands (Montiel Olea-Stock-Watson) | Weakest assumptions on the rest of the system; the report card tells you if the instrument can carry the question |
| Fiscal foresight / news shocks / suspected noninvertibility | Instrument ordered first in the VAR, or LP-IV | Internal instruments stay valid under noninvertibility (Plagborg-Møller & Wolf) |
| Any set-identified result headed for publication | Giacomini-Kitagawa robust bounds alongside | Separates data information from rotation-prior artifact — the honest default |

## What tsecon implements today

**Available now in Python** (`import tsecon`): the recursive scheme end to end — `var_fit` (estimates, `sigma_u`, information criteria, and a characteristic-root stability summary for the long-run fragility check), `var_irf(data, lags, horizon, orth=True)` for Cholesky-orthogonalized impulse responses (set `orth=False` for reduced-form responses), `var_fevd` for the matching variance decompositions, `var_forecast`, and `var_granger`. The internal-instrument pattern from this chapter runs today by ordering the shock series first. Supporting machinery that identification inference leans on is also live: `ols(se_type="hac")` and `long_run_variance` for instrument first stages, `optimal_block_length` and `bootstrap_indices` for the moving-block bootstrap, and `philox_uniforms` for the reproducible parallel random streams that sign-restriction sampling requires.

**Built in Rust awaiting bindings:** no identification-specific kernels yet — but the foundations this module consumes are in the crates and exercised: companion-form IRF/FEVD recursions (bound via `var_irf`/`var_fevd`), the block-bootstrap engine, the Philox counter-based RNG (bitwise-reproducible accept-reject at any thread count), and the scaffolded Bayesian crate (`tsecon-bayes`, NIW-BVAR fixtures pinned) that will supply reduced-form posterior draws to every set-identification sampler.

**Roadmap:** everything else in this chapter — Haar rotation sampling with the QR sign-fix, sign and zero+sign restrictions with ARW importance weights, Blanchard-Quah and combined short/long-run restrictions, max-share, Rigobon and the statistical-identification family, narrative loaders and narrative sign restrictions, proxy SVARs with the mandatory identification report card and Montiel Olea-Stock-Watson robust sets, the composable restriction algebra, and Giacomini-Kitagawa robust Bayes — is specified in [docs/roadmap/06-identification.md](../roadmap/06-identification.md), the module the roadmap designates as the library's headline differentiator: no maintained Python home for modern SVAR identification exists today, and every fit is designed to print its identification diagnostics because, in set-identified and instrument-based settings, the diagnostics are the inference.

## Further reading

- **Sims (1980), "Macroeconomics and Reality," *Econometrica*** — the founding document: replaces incredible large-scale-model restrictions with VARs and poses the identification problem this chapter answers.
- **Blanchard & Quah (1989), *American Economic Review*** — long-run restrictions and the supply/demand decomposition; read alongside **Faust & Leeper (1997, *JBES*)** for why the scheme is fragile.
- **Uhlig (2005), *Journal of Monetary Economics*** — sign restrictions and agnostic identification of monetary shocks; the paper that made set identification mainstream.
- **Rigobon (2003), "Identification Through Heteroskedasticity," *Review of Economics and Statistics*** — variance shifts as instruments; the foundation of all statistical identification.
- **Romer & Romer (2004), *American Economic Review*** — the narrative monetary shock series; the template for identification by reading the record.
- **Mertens & Ravn (2013), *American Economic Review*** and **Gertler & Karadi (2015), *AEJ: Macroeconomics*** — the proxy SVAR in action: tax multipliers from narrative proxies, monetary transmission from high-frequency surprises.
- **Plagborg-Møller & Wolf (2021), *Econometrica*** — LPs and VARs estimate the same impulse responses; the result that reframed the estimator debate.
- **Arias, Rubio-Ramírez & Waggoner (2018), *Econometrica*** — the correct algorithm for zero-plus-sign restrictions; also a masterclass in how algorithmic details change inference.
- **Baumeister & Hamilton (2015), *Econometrica*** and **Giacomini & Kitagawa (2021), *Econometrica*** — the Haar-prior critique and the robust-Bayes answer; together, the modern standard for honest set-identified inference.
- **Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*, Cambridge University Press** — the comprehensive textbook treatment of everything in this chapter; **Ramey (2016), "Macroeconomic Shocks and Their Propagation," *Handbook of Macroeconomics*** — the definitive applied survey comparing shock measures scheme by scheme.
