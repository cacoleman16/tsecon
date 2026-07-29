# %% [markdown]
# # tsecon in five minutes
#
# This notebook takes you from nothing to an identified structural impulse
# response. Everything runs in the browser — no install, no Rust toolchain.
#
# What you will do:
#
# 1. screen a series before modelling it,
# 2. fit a VAR and read an impulse response off it,
# 3. put honest confidence bands on that response,
# 4. and see what the library refuses to guess for you.
#
# If you only read one thing, make it the last section.

# %%
import numpy as np
import pandas as pd
import tsecon

rng = np.random.default_rng(0)

# %% [markdown]
# ## 1 · Screen the series first
#
# The most expensive mistake in applied time series is modelling a series
# without asking what it is. `check_series` runs the diagnostic families in
# order and returns recommendations that point at concrete functions.
#
# We build a random walk — a series with no fixed mean to revert to.

# %%
y = np.cumsum(rng.standard_normal(300))

report = tsecon.check_series(y)
print("verdict:", report["stationarity"]["recommendation"])
print("analysis scale:", report["analysis_scale"]["scale"])
print()
for rec in report["recommendations"][:3]:
    print(f"- {rec['topic']}: {rec['finding']}")
    print(f"    -> {rec['suggestion'][:100]}...")

# %% [markdown]
# Note what happened: the battery decided the series needs differencing, and
# then ran every *downstream* test on the differences rather than on the level.
# Running an ARCH test on a trending level is a classic way to find volatility
# clustering that is not there.
#
# `tsecon.summarize` renders any result as a readable report:

# %%
print(tsecon.summarize(tsecon.adf(y), title="ADF on the level"))

# %% [markdown]
# ## 2 · Fit a VAR and read an impulse response
#
# We simulate a small three-variable system with known dynamics so we can
# check the answer against the truth.

# %%
A = np.array([[0.5, 0.1, 0.0],
              [0.0, 0.4, 0.1],
              [0.1, 0.0, 0.3]])
n, k = 400, 3
data = np.zeros((n, k))
for t in range(1, n):
    data[t] = A @ data[t - 1] + rng.standard_normal(k)

# pandas goes straight in — column names, integer columns, whatever you have
df = pd.DataFrame(data, columns=["output", "prices", "rate"])

fit = tsecon.var_fit(df, lags=2)
print("stable:", fit["is_stable"], " (stable iff min_root > 1:", round(fit["min_root"], 3), ")")
print("BIC:", round(fit["bic"], 4))

irf = np.asarray(tsecon.var_irf(df, lags=2, horizon=12, orth=True))
print("\nresponse of `prices` to a one-SD `output` shock, h=0..5:")
print(np.round(irf[:6, 1, 0], 4))

# %% [markdown]
# `irf[h][i][j]` is the response of variable *i* to a shock in variable *j* at
# horizon *h*. `orth=True` orthogonalises through the Cholesky factor, which
# means **the answer depends on the column ordering** of your data. That is a
# modelling assumption, not a detail — the next notebook is entirely about
# taking it seriously.
#
# ## 3 · Bands, not points
#
# A point estimate of an impulse response is not a finding. `var_irf_bands`
# gives you frequentist confidence bands two ways: the asymptotic delta method,
# and a residual bootstrap.

# %%
bands = tsecon.var_irf_bands(df, lags=2, horizon=12, orth=True, method="asymptotic", alpha=0.10)

point = np.asarray(bands["point"])[:, 1, 0]
lower = np.asarray(bands["lower"])[:, 1, 0]
upper = np.asarray(bands["upper"])[:, 1, 0]

print(" h   point     90% band")
for h in range(6):
    covers_zero = "  <- includes 0" if lower[h] <= 0 <= upper[h] else ""
    print(f"{h:2d}  {point[h]: .4f}   [{lower[h]: .4f}, {upper[h]: .4f}]{covers_zero}")

# %% [markdown]
# Plot it — the shaded band is what you should report, not the line.

# %%
import matplotlib.pyplot as plt

h = np.arange(len(point))
fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(h, lower, upper, alpha=0.25, label="90% band")
ax.plot(h, point, lw=2, label="point estimate")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("horizon (quarters)")
ax.set_ylabel("response of prices")
ax.set_title("Response to a one-SD output shock")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4 · What the library refuses to guess
#
# This is the part worth internalising, because it is where tsecon differs from
# a library that tries to be maximally convenient.
#
# **It will convert your data.** A pandas frame, an integer array, a list of
# numbers, a float32 slice — all fine, all converted at the boundary.

# %%
counts = (y * 10).astype(np.int64)          # e.g. a series read from CSV as int

print("list          ->", "ok" if "p_value" in tsecon.adf([float(v) for v in y]) else "no")
print("integer array ->", "ok" if "p_value" in tsecon.adf(counts) else "no")
print("DataFrame     ->", "ok" if "params" in tsecon.var_fit(df, lags=2) else "no")
print("float32 slice ->", "ok" if "p_value" in tsecon.adf(y.astype(np.float32)[::1]) else "no")

# %% [markdown]
# **It will not guess at anything ambiguous.** A nested Python list could be a
# data matrix *or* a restriction spec — nothing in the value distinguishes
# them — so it is left alone rather than silently reinterpreted:

# %%
try:
    tsecon.var_fit([[1.0, 2.0], [3.0, 4.0]] * 60, lags=1)
except TypeError as exc:
    print(str(exc)[:300])

# %% [markdown]
# And when you get a shape wrong, it tells you what it saw and what to do
# instead of raising a low-level type error:

# %%
try:
    tsecon.var_fit(df["output"], lags=2)      # 1-D where a system is wanted
except TypeError as exc:
    print(str(exc)[:300])

# %% [markdown]
# ## Where to go next
#
# - **`02_irf_bands_and_lp_vs_var.ipynb`** — how much your bands depend on the
#   method, and when local projections beat a VAR (and when they do not).
# - **`03_blanchard_quah.ipynb`** — a full replication of a published result.
# - [The guide](https://cacoleman16.github.io/tsecon/guide/) — sixteen chapters,
#   beginner to research-grade.
# - [Which model when](https://cacoleman16.github.io/tsecon/which-model-when/) —
#   start from your problem, not from a method.
