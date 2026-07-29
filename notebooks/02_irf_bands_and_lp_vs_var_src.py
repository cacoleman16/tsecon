# %% [markdown]
# # Impulse responses: bands, and the LP-vs-VAR choice
#
# Two questions that decide whether an impulse response means anything:
#
# 1. **How wide is the uncertainty, really?** Delta-method and bootstrap bands
#    can disagree, and *pointwise* bands are not what most readers think.
# 2. **VAR or local projection?** The honest answer is "it depends on the
#    horizon and on whether your lag order is right" — and we can measure it
#    rather than argue about it.
#
# Everything here is simulation, so we know the true answer and can check who
# gets closer.

# %%
import numpy as np
import matplotlib.pyplot as plt
import tsecon

rng = np.random.default_rng(20260729)

# %% [markdown]
# ## A DGP where we know the truth
#
# A stationary VAR(1). Because the process is a VAR(1), the population impulse
# response is available in closed form: for a shock through the Cholesky factor
# `P`, the response at horizon `h` is `A^h P`.

# %%
A = np.array([[0.6, 0.2],
              [0.0, 0.5]])
SIGMA = np.array([[1.0, 0.3],
                  [0.3, 1.0]])
P = np.linalg.cholesky(SIGMA)
H = 12


def true_irf(h_max=H):
    """Population orthogonalised IRF: Theta_h = A^h P."""
    out = np.zeros((h_max + 1, 2, 2))
    Ah = np.eye(2)
    for h in range(h_max + 1):
        out[h] = Ah @ P
        Ah = Ah @ A
    return out


def simulate(n, seed):
    r = np.random.default_rng(seed)
    e = r.multivariate_normal(np.zeros(2), SIGMA, size=n)
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = A @ y[t - 1] + e[t]
    return y


TRUE = true_irf()
print("true response of var1 to a var0 shock, h=0..5:")
print(np.round(TRUE[:6, 1, 0], 4))

# %% [markdown]
# ## 1 · Two ways to get a band, and when they differ
#
# `var_irf_bands` offers `method="asymptotic"` (Lütkepohl's delta method — a
# closed-form standard error) and `method="bootstrap"` (resample residuals,
# refit, take percentiles). They answer the same question with different
# assumptions about finite samples.

# %%
y = simulate(200, seed=1)

asy = tsecon.var_irf_bands(y, lags=1, horizon=H, orth=True, method="asymptotic", alpha=0.10)
boo = tsecon.var_irf_bands(y, lags=1, horizon=H, orth=True, method="bootstrap",
                           alpha=0.10, n_boot=2000, seed=7)

i, j = 0, 0   # response of variable 0 to a shock in variable 0
print(" h   truth    point    asymptotic band        bootstrap band")
for h in range(0, 9, 2):
    print(f"{h:2d}  {TRUE[h,i,j]: .3f}   {np.asarray(asy['point'])[h,i,j]: .3f}   "
          f"[{np.asarray(asy['lower'])[h,i,j]: .3f},{np.asarray(asy['upper'])[h,i,j]: .3f}]   "
          f"[{np.asarray(boo['lower'])[h,i,j]: .3f},{np.asarray(boo['upper'])[h,i,j]: .3f}]")

# %%
h = np.arange(H + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(h, np.asarray(asy["lower"])[:, i, j], np.asarray(asy["upper"])[:, i, j],
                alpha=0.20, label="90% asymptotic")
ax.plot(h, np.asarray(boo["lower"])[:, i, j], ls="--", lw=1.2, label="90% bootstrap")
ax.plot(h, np.asarray(boo["upper"])[:, i, j], ls="--", lw=1.2, color=ax.lines[-1].get_color())
ax.plot(h, np.asarray(asy["point"])[:, i, j], lw=2, color="black", label="estimate")
ax.plot(h, TRUE[:, i, j], lw=2, ls=":", color="crimson", label="truth")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("horizon"); ax.set_ylabel("response")
ax.set_title("Same data, two band methods, against the known truth")
ax.legend()
plt.tight_layout(); plt.show()

# %% [markdown]
# The bootstrap band is usually a little wider at longer horizons, because it
# does not lean on a large-sample approximation that gets thinner as the
# horizon compounds. With `n=200` the two mostly agree; shrink the sample and
# they separate.
#
# ### The caveat almost nobody states
#
# These are **pointwise** bands. "The 90% band excludes zero at h=6" is a
# statement about horizon 6 *alone*. It is not a statement about the response
# path as a whole — a path can wander outside a pointwise band far more than
# 10% of the time. Joint (simultaneous) bands are a different, wider object,
# and tsecon does not currently ship them; the model card says so plainly.
#
# Treat a pointwise band as "is this horizon distinguishable from zero", never
# as "the true path lies in here with 90% probability".

# %% [markdown]
# ## 2 · Does the band actually cover? A coverage check
#
# A confidence band is a promise about repeated samples. The only honest way to
# check it is to simulate many datasets and count.

# %%
def coverage(n, reps=300, method="asymptotic", lags=1, alpha=0.10, **kw):
    hits = np.zeros(H + 1)
    for r in range(reps):
        b = tsecon.var_irf_bands(simulate(n, seed=1000 + r), lags=lags, horizon=H,
                                 orth=True, method=method, alpha=alpha, **kw)
        lo, up = np.asarray(b["lower"])[:, i, j], np.asarray(b["upper"])[:, i, j]
        hits += (lo <= TRUE[:, i, j]) & (TRUE[:, i, j] <= up)
    return hits / reps


cov200 = coverage(200)
print("nominal 90% coverage of the asymptotic band, n=200")
for hh in (0, 1, 2, 4, 6, 8, 12):
    print(f"   h={hh:<3} {cov200[hh]:.0%}")

# %% [markdown]
# Impact (h=0) is close to nominal; coverage decays at longer horizons, which is
# the well-known finite-sample behaviour of delta-method bands. This is not a
# bug in the implementation — the asymptotic SEs match statsmodels to machine
# precision — it is what the approximation *is*. Knowing the shape of the
# failure is the point of running the check.

# %% [markdown]
# ## 3 · Local projections vs VAR
#
# The trade-off in one sentence: **a correctly specified VAR is more efficient;
# local projections are more robust when it is misspecified.**
#
# ### Case A — the VAR is correct
#
# We generated a VAR(1) and we fit a VAR(1). This is the VAR's best case.

# %%
def lp_path(y, horizon=H, lags=2):
    """LP response of variable 1 to a movement in variable 0.

    A *cross-variable* response, because the own-response LP (regressing a
    series on itself with its own lags as controls) is collinear by
    construction — tsecon says so rather than silently returning garbage.
    """
    out = tsecon.lp(y[:, 1], y[:, 0], horizons=horizon, n_lag_controls=lags)
    return np.asarray(out["irf"])


def rmse_against_truth(paths, truth):
    paths = np.asarray(paths)
    return np.sqrt(((paths - truth) ** 2).mean(axis=0))


REPS = 200
var_paths, lp_paths = [], []
for r in range(REPS):
    d = simulate(300, seed=5000 + r)
    var_paths.append(np.asarray(tsecon.var_irf(d, lags=1, horizon=H, orth=True))[:, 1, 0])
    lp_paths.append(lp_path(d))

# LP and the orthogonalised VAR IRF differ by a scale at impact; compare shapes
# by normalising both to their own impact value.
vt = np.asarray(var_paths) / np.asarray(var_paths)[:, [0]]
lt = np.asarray(lp_paths) / np.asarray(lp_paths)[:, [0]]
truth_n = TRUE[:, 1, 0] / TRUE[0, 1, 0]

print("RMSE against the true (normalised) path, correctly specified VAR(1):")
print("  h    VAR      LP")
for hh in (1, 2, 4, 8, 12):
    print(f"  {hh:<3}  {rmse_against_truth(vt, truth_n)[hh]:.4f}   {rmse_against_truth(lt, truth_n)[hh]:.4f}")

# %% [markdown]
# The VAR wins, and the gap widens with the horizon. That is the efficiency
# argument: the VAR extrapolates the whole path from one estimated companion
# matrix, so it borrows strength across horizons. LP estimates each horizon
# separately and pays for that in variance.
#
# ### Case B — the lag order is wrong
#
# Now the honest counterweight. We generate a VAR(**4**) and fit a VAR(**1**) —
# the everyday situation where the true dynamics are richer than the model.

# %%
A1 = np.array([[0.5, 0.1], [0.0, 0.4]])
A4 = np.array([[0.25, 0.0], [0.0, 0.25]])   # extra dynamics at lag 4


def simulate_var4(n, seed):
    r = np.random.default_rng(seed)
    e = r.multivariate_normal(np.zeros(2), SIGMA, size=n)
    y = np.zeros((n, 2))
    for t in range(4, n):
        y[t] = A1 @ y[t - 1] + A4 @ y[t - 4] + e[t]
    return y


def true_irf_var4(h_max=H):
    """Population IRF of the VAR(4) via its MA recursion."""
    psi = [np.eye(2)]
    for h in range(1, h_max + 1):
        m = psi[h - 1] @ A1 + (psi[h - 4] @ A4 if h >= 4 else 0)
        psi.append(m)
    return np.array([p @ P for p in psi])


TRUE4 = true_irf_var4()
v4, l4 = [], []
for r in range(REPS):
    d = simulate_var4(300, seed=9000 + r)
    v4.append(np.asarray(tsecon.var_irf(d, lags=1, horizon=H, orth=True))[:, 1, 0])   # WRONG lag order
    l4.append(lp_path(d, lags=1))                                                     # same wrong controls

v4n = np.asarray(v4) / np.asarray(v4)[:, [0]]
l4n = np.asarray(l4) / np.asarray(l4)[:, [0]]
truth4_n = TRUE4[:, 1, 0] / TRUE4[0, 1, 0]

print("Bias against the true path when the VAR(4) is fit as a VAR(1):")
print("  h    VAR bias    LP bias")
for hh in (1, 2, 4, 6, 8, 12):
    print(f"  {hh:<3}  {v4n.mean(axis=0)[hh]-truth4_n[hh]: .4f}    {l4n.mean(axis=0)[hh]-truth4_n[hh]: .4f}")

# %% [markdown]
# Now the picture flips at longer horizons. The misspecified VAR **compounds**
# its error: every horizon is generated by iterating the same wrong companion
# matrix, so a small one-step error becomes a large twelve-step one. LP
# estimates each horizon with its own regression, so a bad control set biases
# each horizon *once* rather than geometrically.
#
# That is the whole trade-off, and it is why the debate has no universal
# winner:
#
# | | VAR | Local projection |
# |---|---|---|
# | Correctly specified | **more efficient** (lower variance) | noisier, especially at long h |
# | Lag-truncated / misspecified | error compounds with horizon | error does **not** compound |
# | Long horizons | extrapolates | estimates directly, wide bands |
# | Nonlinearity, state dependence | awkward | natural (`lp_state`) |
# | Needs an identified system | yes | only the shock |
#
# ### Practical guidance
#
# - Report both when the answer matters. If they agree, you have learned
#   something; if they disagree, your lag order or specification is doing the
#   work and you should say so.
# - `smooth_lp` sits between the two: it penalises the LP path toward a
#   smooth function, buying back much of the VAR's efficiency without the
#   compounding. `lam=0` recovers plain LP exactly.
# - At short horizons the choice rarely matters. At h > 8 it usually does.

# %%
d1 = simulate_var4(300, seed=1)
sm = tsecon.smooth_lp(d1[:, 1], d1[:, 0], horizons=H, n_lag_controls=2, lam="cv")
print("smooth_lp cross-validated lambda:", round(sm["lambda_used"], 3))
print("(lam=0 would reproduce plain LP; larger lam shrinks toward a smooth path)")

# %% [markdown]
# ## What to take away
#
# 1. Always report a band, and know it is pointwise.
# 2. Check coverage by simulation when the result carries weight — the
#    asymptotic band's coverage decays with horizon, and you should know by how
#    much before you lean on h=12.
# 3. VAR vs LP is a bias-variance choice, not a correctness one. Efficiency
#    when you trust the specification; robustness when you do not.
#
# Next: **`03_blanchard_quah.ipynb`**, where these choices get made on real data
# to reproduce a published result.
