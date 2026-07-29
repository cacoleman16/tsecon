# %% [markdown]
# # Replicating Blanchard & Quah (1989)
#
# The paper that made long-run restrictions standard practice. Blanchard and
# Quah identify two shocks in a bivariate system of output growth and
# unemployment by assuming that **demand shocks have no permanent effect on
# the level of output**, while supply shocks may.
#
# That single assumption is enough to identify the system, and it is a
# genuinely economic restriction rather than a timing convention — which is
# why the paper mattered.
#
# We reproduce the exercise end to end on the authors' sample window using data
# committed to the tsecon repository.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsecon

URL = ("https://raw.githubusercontent.com/cacoleman16/tsecon/main/"
       "fixtures/ramey_zubairy.csv")
try:
    raw = pd.read_csv("../fixtures/ramey_zubairy.csv", comment="#")   # local checkout
except FileNotFoundError:
    raw = pd.read_csv(URL, comment="#")                              # Colab

print(raw[["quarter", "rgdp", "unemp", "pop"]].dropna().shape, "usable quarterly observations")

# %% [markdown]
# ## Building the two series
#
# Blanchard and Quah use output **growth** (a series that is stationary) and
# **detrended unemployment**. The detrending matters: unemployment drifts over
# a long sample for reasons that have nothing to do with the business cycle,
# and the identification assumes both series are stationary.
#
# We use per-capita real GDP so that population growth does not masquerade as
# a supply shock, and restrict to 1948-1987 — the authors' own window.

# %%
d = raw[["quarter", "rgdp", "unemp", "pop"]].dropna()
d = d[(d.quarter >= 1948) & (d.quarter < 1988)]

growth = 100 * np.diff(np.log(d.rgdp.values / d["pop"].values))   # per-capita output growth
unemp = d.unemp.values[1:]
trend = np.polyval(np.polyfit(np.arange(len(unemp)), unemp, 1), np.arange(len(unemp)))
unemp_dt = unemp - trend                                          # linearly detrended

Y = np.column_stack([growth, unemp_dt])
print(f"sample {d.quarter.min():.0f}Q1 - {d.quarter.max():.0f}Q4,  n = {len(Y)}")

# %% [markdown]
# A quick sanity check before identifying anything — is the growth series
# actually stationary, and is detrended unemployment?

# %%
for name, series in [("output growth", growth), ("unemployment (detrended)", unemp_dt)]:
    r = tsecon.check_stationarity(series)
    print(f"{name:28} quadrant={r['quadrant']:<14} -> {r['recommendation']}")

# %% [markdown]
# ## The identification
#
# `long_run_svar` imposes the Blanchard-Quah restriction by construction: the
# long-run cumulative impact matrix is made lower-triangular, so the second
# shock (demand) has **zero** permanent effect on the first variable (output).

# %%
bq = tsecon.long_run_svar(Y, lags=8, horizon=40)

print("long-run impact matrix C(1)B:")
print(np.round(np.asarray(bq["long_run"]), 4))
print("\nThe (0,1) element is exactly zero — that IS the identifying assumption.")

# %% [markdown]
# ### Does the restriction actually bind?
#
# The restriction is on the effect **at infinity**. A finite-horizon cumulative
# response approaches zero rather than being zero at every horizon, and
# watching it converge is the best way to see what was assumed.

# %%
long = tsecon.long_run_svar(Y, lags=8, horizon=200)
cum = np.asarray(long["cumulative_irf"])

print("cumulative output response to the DEMAND shock:")
for h in (0, 4, 8, 20, 40, 100, 200):
    print(f"   h={h:<4} {cum[h, 0, 1]: .5f}")
print("\ncumulative output response to the SUPPLY shock (permanent):")
for h in (0, 4, 20, 40, 200):
    print(f"   h={h:<4} {cum[h, 0, 0]: .5f}")

# %% [markdown]
# The demand shock's effect on the *level* of output dies out completely by
# roughly ten years, while the supply shock settles at a permanent 0.53. The
# restriction is doing exactly what it claims.
#
# ## Sign normalisation — a convention, not a finding
#
# An SVAR identifies each shock only up to sign. As estimated, our "demand"
# shock happens to *lower* output and *raise* unemployment, i.e. it is a
# contractionary demand shock. To match the paper's presentation we flip it so
# that a positive shock is expansionary.
#
# This is worth pausing on: **the sign is chosen by you, not by the data.**
# Report which convention you used.

# %%
irf = np.asarray(bq["irf"]).copy()
cumf = np.asarray(bq["cumulative_irf"]).copy()

for s in range(2):                       # make each shock raise output on impact
    if cumf[4, 0, s] < 0:
        irf[:, :, s] *= -1
        cumf[:, :, s] *= -1

print("after normalisation, output response at h=4:  supply "
      f"{cumf[4,0,0]: .3f}   demand {cumf[4,0,1]: .3f}")
print("unemployment response at h=4:                 supply "
      f"{irf[4,1,0]: .3f}   demand {irf[4,1,1]: .3f}")

# %% [markdown]
# ## The figure the paper is known for
#
# Four panels: the level of output and unemployment, each responding to a
# supply and a demand shock. The signature result is the contrast in the top
# row — supply raises output permanently, demand raises it temporarily.

# %%
H = 40
h = np.arange(H + 1)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
panels = [
    (0, 0, cumf[:H + 1, 0, 0], "Output level  <-  supply shock"),
    (0, 1, cumf[:H + 1, 0, 1], "Output level  <-  demand shock"),
    (1, 0, irf[:H + 1, 1, 0], "Unemployment  <-  supply shock"),
    (1, 1, irf[:H + 1, 1, 1], "Unemployment  <-  demand shock"),
]
for r, c, series, title in panels:
    ax = axes[r][c]
    ax.plot(h, series, lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    if r == 1:
        ax.set_xlabel("quarters")
fig.suptitle("Blanchard-Quah (1989): supply and demand shocks, 1948-1987", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## How much of the business cycle is demand?
#
# The forecast-error variance decomposition answers the question the paper was
# really asking. `structural_fevd` takes the identified impact matrix, so the
# shares refer to the *structural* shocks rather than a Cholesky ordering.

# %%
fevd = np.asarray(tsecon.structural_fevd(Y, lags=8, horizon=40,
                                         impact=np.asarray(bq["impact"]))["fevd"])
print("share of forecast-error variance explained by each shock")
print("            output growth            unemployment")
print("  h      supply    demand        supply    demand")
for hh in (0, 1, 4, 12, 40):
    print(f"  {hh:<4}   {fevd[hh,0,0]:.2f}      {fevd[hh,0,1]:.2f}          "
          f"{fevd[hh,1,0]:.2f}      {fevd[hh,1,1]:.2f}")

# %% [markdown]
# ## What this replication does and does not establish
#
# **Reproduced:** the identification mechanics, the qualitative shape of the
# responses, and the central finding that demand disturbances move output
# temporarily while supply disturbances move it permanently.
#
# **Not identical to the published table.** Blanchard and Quah use a slightly
# different output series (they do not use the Ramey-Zubairy vintage), a
# different detrending of unemployment, and 1950s-vintage data that has since
# been revised many times. Point estimates will differ in the second digit.
# That is the honest situation for any replication of a 1989 paper, and it is
# worth stating rather than tuning until the numbers match.
#
# **The assumption is doing the work.** Nothing in the data says demand shocks
# are transitory — we assumed it. A different identification (sign
# restrictions, a proxy for monetary policy, heteroskedasticity) will give a
# different answer, and tsecon ships those too so you can check how much your
# conclusion depends on the assumption:

# %%
alt = tsecon.max_share_svar(Y, lags=8, target=0, h0=1, h1=40, horizon=40)
print("max-share shock explains "
      f"{alt['share_window']:.1%} of output's FEV over h=1..40")
print("(a different question, a different shock — compare, do not conflate)")

# %% [markdown]
# ## Further reading
#
# - [Model card: structural identification](https://cacoleman16.github.io/tsecon/reference/model-cards/structural-identification/)
# - [Guide chapter 8 — structural identification](https://cacoleman16.github.io/tsecon/guide/08-causal-identification/)
# - Blanchard, O. & Quah, D. (1989), "The Dynamic Effects of Aggregate Demand
#   and Supply Disturbances", *American Economic Review* 79(4), 655-673.
