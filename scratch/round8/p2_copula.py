"""Round-8 probe: copula_fit / copula_select / pseudo_obs.

Reference: statsmodels 0.14.6 copula densities, scipy kendalltau/rankdata,
closed-form tau maps and tail dependence. Lenses 1,2,3,5.
"""
import numpy as np
import tsecon
from scipy import stats, optimize, integrate
from statsmodels.distributions.copula.api import (
    GaussianCopula, StudentTCopula, ClaytonCopula, GumbelCopula, FrankCopula)

attempted = 0
made = 0
fails = []


def check(name, cond, detail=""):
    global attempted, made
    attempted += 1
    made += 1
    if not cond:
        fails.append((name, detail))
    print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")


def expect_raise(name, fn, want=None):
    global attempted, made
    attempted += 1
    try:
        fn()
        made += 1
        fails.append((name, "no raise"))
        print(f"[FAIL] {name}: no raise")
    except Exception as e:
        made += 1
        ok = want is None or want.lower() in str(e).lower()
        if not ok:
            fails.append((name, f"raised but message lacks {want!r}: {e}"))
        print(f"[{'ok' if ok else 'FAIL'}] {name}: {type(e).__name__}: {str(e)[:110]}")


rng = np.random.default_rng(80802)
n = 600
# t-copula data (rho .6, nu 4) via the card's construction
z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=n)
tp = z / np.sqrt(rng.chisquare(4, size=n) / 4)[:, None]
x = np.column_stack([tp[:, 0], tp[:, 1]])
u = tsecon.pseudo_obs(x)

# ---------- pseudo_obs ----------
ref_u = np.column_stack([stats.rankdata(x[:, j], method="average") / (n + 1) for j in range(2)])
check("pseudo_obs == rankdata/(n+1) exactly", np.array_equal(u, ref_u))

xt = np.column_stack([x[:, 0].copy(), x[:, 1].copy()])
xt[5, 0] = xt[9, 0]  # inject a tie
ut = tsecon.pseudo_obs(xt)
rt = np.column_stack([stats.rankdata(xt[:, j], method="average") / (n + 1) for j in range(2)])
check("pseudo_obs ties == average-rank exactly", np.array_equal(ut, rt))

# strictly increasing transforms: bit-identical
x_inc = np.column_stack([np.exp(0.01 * x[:, 0]), 100 + 5 * x[:, 1]])
check("pseudo_obs invariant under increasing transforms", np.array_equal(tsecon.pseudo_obs(x_inc), u))

# strictly DECREASING transform: the docs say "any strictly monotone transform"
x_dec = np.column_stack([-x[:, 0], x[:, 1]])
u_dec = tsecon.pseudo_obs(x_dec)
same = np.array_equal(u_dec, u)
print(f"[note] decreasing transform leaves pseudo_obs unchanged? {same}"
      f" (docstring says 'any strictly monotone transform ... bit-identical')")
if not same:
    # and the fitted copula flips sign
    g0 = tsecon.copula_fit(u, family="gaussian")
    g1 = tsecon.copula_fit(u_dec, family="gaussian")
    print(f"        gaussian rho on x: {g0['rho']:.4f}, on (-x1, x2): {g1['rho']:.4f}")

# 3-column pseudo_obs accepted (documented); copula_fit bivariate only
u3 = tsecon.pseudo_obs(rng.standard_normal((50, 3)))
check("pseudo_obs accepts 3 columns", u3.shape == (50, 3))
expect_raise("copula_fit rejects 3 columns", lambda: tsecon.copula_fit(u3, family="gaussian"))

# ---------- copula_fit MLE vs scipy-optimized statsmodels log-density ----------
def sm_loglik(family, u, *params):
    if family == "gaussian":
        cop = GaussianCopula(corr=params[0])
    elif family == "t":
        cop = StudentTCopula(corr=params[0], df=params[1])
    elif family == "clayton":
        cop = ClaytonCopula(theta=params[0])
    elif family == "gumbel":
        cop = GumbelCopula(theta=params[0])
    elif family == "frank":
        cop = FrankCopula(theta=params[0])
    return np.sum(cop.logpdf(u))

fits = {}
for fam in ["gaussian", "t", "clayton", "gumbel", "frank"]:
    fits[fam] = tsecon.copula_fit(u, family=fam)

# loglik agreement at the fitted parameters
for fam in ["gaussian", "t", "clayton", "gumbel", "frank"]:
    f = fits[fam]
    if fam == "gaussian":
        ll = sm_loglik(fam, u, f["rho"])
    elif fam == "t":
        ll = sm_loglik(fam, u, f["rho"], f["nu"])
    else:
        ll = sm_loglik(fam, u, f["theta"])
    check(f"{fam}: loglik matches statsmodels density at fitted params",
          abs(f["loglik"] - ll) < 1e-8 * max(1, abs(ll)), f"{f['loglik']:.8f} vs {ll:.8f}")

# my own MLE (scipy) for the 1-param families
for fam, lo, hi in [("gaussian", -0.99, 0.99), ("clayton", 1e-4, 30.0),
                    ("gumbel", 1.0001, 30.0), ("frank", -30.0, 30.0)]:
    r = optimize.minimize_scalar(lambda p: -sm_loglik(fam, u, p), bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-10})
    mine = r.x
    key = "rho" if fam == "gaussian" else "theta"
    check(f"{fam}: MLE param matches my scipy optimum", abs(fits[fam][key] - mine) < 2e-5 * max(1, abs(mine)),
          f"{fits[fam][key]:.8f} vs {mine:.8f}")
    check(f"{fam}: tsecon loglik >= scipy optimum - 1e-7",
          fits[fam]["loglik"] >= -r.fun - 1e-7, f"{fits[fam]['loglik']:.10f} vs {-r.fun:.10f}")

# t: 2-D MLE
r = optimize.minimize(lambda p: -sm_loglik("t", u, np.tanh(p[0]), 2.0 + np.exp(p[1])),
                      x0=[np.arctanh(0.6), np.log(4.0 - 2.0)], method="Nelder-Mead",
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
rho_my, nu_my = np.tanh(r.x[0]), 2.0 + np.exp(r.x[1])
check("t: MLE (rho, nu) matches my scipy optimum",
      abs(fits["t"]["rho"] - rho_my) < 2e-4 and abs(fits["t"]["nu"] - nu_my) < 2e-2 * nu_my,
      f"({fits['t']['rho']:.6f},{fits['t']['nu']:.4f}) vs ({rho_my:.6f},{nu_my:.4f})")
check("t: tsecon loglik >= scipy optimum - 1e-6", fits["t"]["loglik"] >= -r.fun - 1e-6,
      f"{fits['t']['loglik']:.8f} vs {-r.fun:.8f}")

# ---------- AIC / BIC arithmetic ----------
for fam, k in [("gaussian", 1), ("t", 2), ("clayton", 1), ("gumbel", 1), ("frank", 1)]:
    f = fits[fam]
    check(f"{fam}: aic == 2k - 2loglik", abs(f["aic"] - (2 * k - 2 * f["loglik"])) < 1e-9,
          f"{f['aic']:.6f}")
    check(f"{fam}: bic == k ln n - 2loglik", abs(f["bic"] - (k * np.log(n) - 2 * f["loglik"])) < 1e-9,
          f"{f['bic']:.6f}")

# ---------- tau: empirical + implied maps ----------
tau_emp = stats.kendalltau(u[:, 0], u[:, 1]).statistic
check("tau == scipy kendalltau", abs(fits["gaussian"]["tau"] - tau_emp) < 1e-14,
      f"{fits['gaussian']['tau']:.12f} vs {tau_emp:.12f}")
check("gaussian tau_implied == 2/pi asin(rho)",
      abs(fits["gaussian"]["tau_implied"] - 2 / np.pi * np.arcsin(fits["gaussian"]["rho"])) < 1e-12)
check("t tau_implied == 2/pi asin(rho)",
      abs(fits["t"]["tau_implied"] - 2 / np.pi * np.arcsin(fits["t"]["rho"])) < 1e-12)
check("clayton tau_implied == th/(th+2)",
      abs(fits["clayton"]["tau_implied"] - fits["clayton"]["theta"] / (fits["clayton"]["theta"] + 2)) < 1e-12)
check("gumbel tau_implied == 1 - 1/th",
      abs(fits["gumbel"]["tau_implied"] - (1 - 1 / fits["gumbel"]["theta"])) < 1e-12)
# frank: tau = 1 - 4/th (1 - D1(th)); D1 via quad
th = fits["frank"]["theta"]
D1 = integrate.quad(lambda t: t / np.expm1(t), 0, th)[0] / th
check("frank tau_implied == exact Debye map",
      abs(fits["frank"]["tau_implied"] - (1 - 4 / th * (1 - D1))) < 1e-9)

# tau-inversion method vs statsmodels fit_corr_param + closed maps
ft = {}
for fam in ["gaussian", "t", "clayton", "gumbel", "frank"]:
    ft[fam] = tsecon.copula_fit(u, family=fam, method="tau")
check("tau method: gaussian rho == sin(pi tau/2)",
      abs(ft["gaussian"]["rho"] - np.sin(np.pi * tau_emp / 2)) < 1e-12)
check("tau method: clayton theta == 2tau/(1-tau)",
      abs(ft["clayton"]["theta"] - 2 * tau_emp / (1 - tau_emp)) < 1e-12)
check("tau method: gumbel theta == 1/(1-tau)",
      abs(ft["gumbel"]["theta"] - 1 / (1 - tau_emp)) < 1e-12)
sm_gauss = GaussianCopula().fit_corr_param(u)
check("tau method: gaussian rho == statsmodels fit_corr_param",
      abs(ft["gaussian"]["rho"] - sm_gauss) < 1e-12, f"{ft['gaussian']['rho']} vs {sm_gauss}")
for fam in ["gaussian", "clayton", "gumbel", "frank"]:
    check(f"tau method {fam}: NaN SEs + se_valid False",
          (not ft[fam]["se_valid"]) and all(np.isnan(s) for s in np.atleast_1d(ft[fam]["se"])))
check("tau method t: rho pinned by tau, nu profiled",
      abs(ft["t"]["rho"] - np.sin(np.pi * tau_emp / 2)) < 1e-12 and ft["t"]["nu"] > 2)

# ---------- tail dependence ----------
f = fits["t"]
lam = 2 * stats.t.cdf(-np.sqrt((f["nu"] + 1) * (1 - f["rho"]) / (1 + f["rho"])), f["nu"] + 1)
check("t tail == Demarta-McNeil closed form", abs(f["tail_lower"] - lam) < 1e-10,
      f"{f['tail_lower']:.8f} vs {lam:.8f}")
check("t tails symmetric", f["tail_lower"] == f["tail_upper"])
check("gaussian tails zero", fits["gaussian"]["tail_lower"] == 0.0 and fits["gaussian"]["tail_upper"] == 0.0)
check("frank tails zero", fits["frank"]["tail_lower"] == 0.0 and fits["frank"]["tail_upper"] == 0.0)
check("clayton lower == 2^(-1/th), upper 0",
      abs(fits["clayton"]["tail_lower"] - 2 ** (-1 / fits["clayton"]["theta"])) < 1e-12
      and fits["clayton"]["tail_upper"] == 0.0)
check("gumbel upper == 2 - 2^(1/th), lower 0",
      abs(fits["gumbel"]["tail_upper"] - (2 - 2 ** (1 / fits["gumbel"]["theta"]))) < 1e-12
      and fits["gumbel"]["tail_lower"] == 0.0)
# the statsmodels-bug reference number: rho=.5, nu=4 -> 0.2532
lam_ref = 2 * stats.t.cdf(-np.sqrt(5 * 0.5 / 1.5), 5)
check("Demarta-McNeil at (rho=.5,nu=4) == 0.2532 (card's claim)",
      abs(lam_ref - 0.2532) < 5e-4, f"{lam_ref:.6f}")

# ---------- copula_select ----------
sel = tsecon.copula_select(u)
check("select: fits for all five", set(sel["fits"].keys()) == {"gaussian", "t", "clayton", "gumbel", "frank"}
      if isinstance(sel["fits"], dict) else len(sel["fits"]) == 5, str(type(sel["fits"])))
# ranking consistent with aics
if isinstance(sel["fits"], dict):
    aics = {k: v["aic"] for k, v in sel["fits"].items()}
else:
    aics = {v["family"]: v["aic"] for v in sel["fits"]}
order = sorted(aics, key=lambda k: aics[k])
check("ranking_aic sorted by aic", list(sel["ranking_aic"]) == order,
      f"{sel['ranking_aic']} vs {order}")
check("best_aic consistent", sel["best_aic"] == order[0])
bics = {k: (sel["fits"][k]["bic"] if isinstance(sel["fits"], dict) else None) for k in aics}
if isinstance(sel["fits"], dict):
    order_b = sorted(bics, key=lambda k: bics[k])
    check("best_bic consistent", sel["best_bic"] == order_b[0])
check("t wins AIC on t-copula data", sel["best_aic"] == "t", sel["best_aic"])

# negative-dependence data: clayton/gumbel skipped with reason
z2 = rng.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]], size=200)
un = tsecon.pseudo_obs(z2)
seln = tsecon.copula_select(un)
skipped = seln["skipped"]
print("skipped:", skipped)
check("negative tau: clayton+gumbel skipped",
      ("clayton" in str(skipped)) and ("gumbel" in str(skipped)))
expect_raise("copula_fit clayton on negative tau raises teaching error",
             lambda: tsecon.copula_fit(un, family="clayton"), want="tau")

# ---------- degenerate input ----------
expect_raise("u outside (0,1) raises with pseudo_obs pointer",
             lambda: tsecon.copula_fit(x, family="gaussian"), want="pseudo_obs")
expect_raise("u == 0 boundary raises",
             lambda: tsecon.copula_fit(np.column_stack([np.linspace(0, 0.9, 30), np.linspace(0.05, 0.9, 30)]), family="gaussian"))
expect_raise("n < 20 raises", lambda: tsecon.copula_fit(u[:19], family="gaussian"))
expect_raise("NaN raises", lambda: tsecon.copula_fit(np.where(np.arange(n)[:, None] == 3, np.nan, u), family="gaussian"))
# perfectly monotone pair
um = tsecon.pseudo_obs(np.column_stack([np.arange(100.0), 2 * np.arange(100.0) + 1]))
expect_raise("perfectly monotone pair refuses", lambda: tsecon.copula_fit(um, family="gaussian"))
expect_raise("unknown family", lambda: tsecon.copula_fit(u, family="joe"))
expect_raise("unknown method", lambda: tsecon.copula_fit(u, family="gaussian", method="ifm"))
# constant column raw -> pseudo_obs puts it at 0.5..; copula_fit should refuse (|tau| undefined/0?)
uc = tsecon.pseudo_obs(np.column_stack([np.ones(50), np.arange(50.0)]))
print("constant-column pseudo_obs unique vals:", np.unique(uc[:, 0]))
try:
    r = tsecon.copula_fit(uc, family="gaussian")
    print("[note] constant column fit returned rho =", r["rho"], "loglik", r["loglik"])
except Exception as e:
    print("[note] constant column fit raises:", type(e).__name__, str(e)[:90])

# ---------- lens 2: monotone-invariance of the whole workflow (bitwise) ----------
fit_a = tsecon.copula_fit(tsecon.pseudo_obs(x), family="t")
fit_b = tsecon.copula_fit(tsecon.pseudo_obs(x_inc), family="t")
check("copula_fit bit-identical under increasing margin transforms",
      fit_a["rho"] == fit_b["rho"] and fit_a["nu"] == fit_b["nu"] and fit_a["loglik"] == fit_b["loglik"])

# near-Gaussian data: nu at its 1000 barrier, honest flags?
zg = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=400)
fg = tsecon.copula_fit(tsecon.pseudo_obs(zg), family="t")
print(f"[note] near-Gaussian t fit: nu={fg['nu']:.1f} se_valid={fg['se_valid']} "
      f"converged={fg['converged']} loglik={fg['loglik']:.4f}")
gau = tsecon.copula_fit(tsecon.pseudo_obs(zg), family="gaussian")
check("t loglik >= gaussian loglik (nesting, from above)", fg["loglik"] >= gau["loglik"] - 1e-6,
      f"{fg['loglik']:.6f} vs {gau['loglik']:.6f}")

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
