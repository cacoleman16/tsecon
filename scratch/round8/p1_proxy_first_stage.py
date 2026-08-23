"""Round-8 probe: proxy_first_stage vs statsmodels OLS + scipy ncx2.

Lenses: 1 (axes alive), 2 (scale), 3 (degenerate), 5 (doc promises),
reference cross-check (statsmodels HC1/HAC, scipy ncx2.ppf).
"""
import numpy as np
import tsecon
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from scipy import stats

attempted = 0
made = 0
fails = []


def check(name, cond, detail=""):
    global attempted, made
    attempted += 1
    made += 1
    status = "ok" if cond else "FAIL"
    if not cond:
        fails.append((name, detail))
    print(f"[{status}] {name} {detail}")


rng = np.random.default_rng(80801)
T = 300
n = 3
# VAR(2) DGP
A1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
A2 = 0.2 * np.eye(3)
eps = rng.standard_normal((T + 50, n))
y = np.zeros((T + 50, n))
for t in range(2, T + 50):
    y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + eps[t]
y = y[50:]
# proxy correlated with structural shock of var 0 (use reduced-form resid proxy)
proxy_full = 0.8 * eps[50:, 0] + 0.6 * rng.standard_normal(T)

lags = 2
res = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0)

# ---- reference: statsmodels VAR residuals + OLS HC1 ----
smvar = VAR(y).fit(lags, trend="c")
u = smvar.resid  # (T - lags, n)
m = proxy_full[lags:]
X = sm.add_constant(m)
ols = sm.OLS(u[:, 0], X).fit(cov_type="HC1")
beta_sm = ols.params[1]
se_sm = ols.bse[1]
f_sm = (beta_sm / se_sm) ** 2

check("beta vs statsmodels", abs(res["beta"] - beta_sm) < 1e-8 * max(1, abs(beta_sm)),
      f"{res['beta']:.12g} vs {beta_sm:.12g}")
check("effective_f (HC1) vs statsmodels t^2", abs(res["effective_f"] - f_sm) < 1e-6 * f_sm,
      f"{res['effective_f']:.10g} vs {f_sm:.10g}")
check("se vs statsmodels HC1", abs(res["se"] - se_sm) < 1e-8 * se_sm,
      f"{res['se']:.12g} vs {se_sm:.12g}")

# classical
res_cl = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="classical")
ols_cl = sm.OLS(u[:, 0], X).fit()
f_cl = (ols_cl.params[1] / ols_cl.bse[1]) ** 2
check("classical F vs statsmodels", abs(res_cl["effective_f"] - f_cl) < 1e-6 * f_cl,
      f"{res_cl['effective_f']:.10g} vs {f_cl:.10g}")

# HAC with explicit lags
L = 5
res_hac = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="hac", hac_lags=L)
ols_hac = sm.OLS(u[:, 0], X).fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": False})
# tsecon applies the HC1-style T/(T-2) dof correction (weakivtest convention)
To = len(m)
var_hac_sm = ols_hac.bse[1] ** 2 * (To / (To - 2))
f_hac_ref = ols_hac.params[1] ** 2 / var_hac_sm
check("HAC effective F vs statsmodels HAC*T/(T-2)",
      abs(res_hac["effective_f"] - f_hac_ref) < 1e-6 * f_hac_ref,
      f"{res_hac['effective_f']:.10g} vs {f_hac_ref:.10g}")
check("hac_lags echoed", res_hac["hac_lags"] == L, str(res_hac["hac_lags"]))

# default NW rule for hac_lags
res_hac_d = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="hac")
print("default hac_lags:", res_hac_d["hac_lags"], "NW rule floor(4*(T/100)^(2/9)) =",
      int(np.floor(4 * ((T - lags) / 100.0) ** (2.0 / 9.0))))
check("default hac_lags = NW rule on residual T",
      res_hac_d["hac_lags"] == int(np.floor(4 * ((T - lags) / 100.0) ** (2.0 / 9.0))),
      str(res_hac_d["hac_lags"]))

# ---- critical values vs scipy ----
for tau, key in [(0.05, "mop_cv_tau5"), (0.10, "mop_cv_tau10"), (0.20, "mop_cv_tau20"), (0.30, "mop_cv_tau30")]:
    ref = stats.ncx2.ppf(0.95, 1, 1.0 / tau)
    check(f"cv tau={tau} vs scipy ncx2.ppf", abs(res[key] - ref) < 1e-6 * ref,
          f"{res[key]:.10g} vs {ref:.10g}")

# tau_bound inversion vs scipy: ncx2.cdf(F, 1, 1/tau_bound) == 0.95
F = res["effective_f"]
tb = res["tau_bound"]
if np.isfinite(tb):
    p = stats.ncx2.cdf(F, 1, 1.0 / tb)
    check("tau_bound inverts by scipy cdf", abs(p - 0.95) < 1e-6, f"cdf={p:.10g}")
else:
    check("tau_bound finite for this strong DGP", False, "got inf")

# weak flags consistent
check("weak_mop_tau10 == (F <= cv10)", res["weak_mop_tau10"] == (F <= res["mop_cv_tau10"]))
check("weak_folklore == (F < 10)", res["weak_folklore"] == (F < 10.0))

# ---- lens 1: axes alive ----
check("variance axis alive (hc1 vs classical)", res["effective_f"] != res_cl["effective_f"])
check("variance axis alive (hc1 vs hac)", res["effective_f"] != res_hac["effective_f"])
r_t = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, trend="n")
check("trend axis alive", r_t["effective_f"] != res["effective_f"])
r_l = tsecon.proxy_first_stage(y, proxy_full, lags=4, norm_var=0)
check("lags axis alive", r_l["effective_f"] != res["effective_f"])
r_nv = tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=1)
check("norm_var axis alive", r_nv["effective_f"] != res["effective_f"])

# hac_lags under hc1 must raise (the hac_lags lesson)
try:
    tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="hc1", hac_lags=3)
    check("hac_lags under hc1 raises", False, "no raise")
except ValueError as e:
    check("hac_lags under hc1 raises", "hac" in str(e).lower(), str(e)[:80])

# ---- lens 2: scale invariance ----
r_s1 = tsecon.proxy_first_stage(y * 1e8, proxy_full, lags=lags, norm_var=0)
r_s2 = tsecon.proxy_first_stage(y, proxy_full * 1e-8, lags=lags, norm_var=0)
check("F invariant to data scale 1e8", abs(r_s1["effective_f"] - F) < 1e-6 * F,
      f"{r_s1['effective_f']:.10g}")
check("F invariant to proxy scale 1e-8", abs(r_s2["effective_f"] - F) < 1e-6 * F,
      f"{r_s2['effective_f']:.10g}")

# ---- proxy of length T (residual sample) accepted, same result ----
r_short = tsecon.proxy_first_stage(y, proxy_full[lags:], lags=lags, norm_var=0)
check("length-T proxy alias bit-identical", r_short["effective_f"] == F)

# ---- NaN proxy prefix (GK-style availability window) ----
proxy_nan = proxy_full.copy()
proxy_nan[:60] = np.nan
r_nan = tsecon.proxy_first_stage(y, proxy_nan, lags=lags, norm_var=0)
u_sub = u[58:, 0]  # residual rows 58.. correspond to obs rows 60..
m_sub = proxy_nan[60:]
ols_nan = sm.OLS(u_sub - 0, sm.add_constant(m_sub)).fit(cov_type="HC1")
f_nan = (ols_nan.params[1] / ols_nan.bse[1]) ** 2
check("NaN-prefix overlap count", r_nan["n_proxy"] == T - 60, str(r_nan["n_proxy"]))
check("NaN-prefix F vs statsmodels on overlap", abs(r_nan["effective_f"] - f_nan) < 1e-6 * f_nan,
      f"{r_nan['effective_f']:.10g} vs {f_nan:.10g}")

# ---- consistency with proxy_svar stamped dict ----
ps = tsecon.proxy_svar(y, proxy_full, lags=lags, norm_var=0)
check("proxy_svar first_stage stamped == standalone (bitwise F)",
      ps["first_stage"]["effective_f"] == F,
      f"{ps['first_stage']['effective_f']} vs {F}")
check("proxy_svar first_stage_f == effective_f", ps["first_stage_f"] == F)

# ---- lens 3: degenerate inputs ----
def expect_raise(name, fn):
    global attempted, made
    attempted += 1
    try:
        fn()
        made += 1
        fails.append((name, "no raise"))
        print(f"[FAIL] {name}: no raise")
    except Exception as e:
        made += 1
        print(f"[ok] {name}: {type(e).__name__}: {str(e)[:100]}")

expect_raise("constant proxy", lambda: tsecon.proxy_first_stage(y, np.ones(T), lags=lags, norm_var=0))
expect_raise("all-NaN proxy", lambda: tsecon.proxy_first_stage(y, np.full(T, np.nan), lags=lags, norm_var=0))
expect_raise("norm_var out of range", lambda: tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=5))
expect_raise("wrong proxy length", lambda: tsecon.proxy_first_stage(y, proxy_full[:100], lags=lags, norm_var=0))
expect_raise("hac bandwidth >= overlap", lambda: tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="hac", hac_lags=T))
expect_raise("unknown variance", lambda: tsecon.proxy_first_stage(y, proxy_full, lags=lags, norm_var=0, variance="hc3"))
# 3 finite obs is the minimum; 2 must fail
proxy_two = np.full(T, np.nan); proxy_two[:2] = [1.0, 2.0]
expect_raise("overlap of 2", lambda: tsecon.proxy_first_stage(y, proxy_two, lags=lags, norm_var=0))

# doc-key diff
expected_keys = {"beta", "se", "effective_f", "f_classical", "f_hc1", "reliability",
                 "n_proxy", "hac_lags", "mop_cv_tau5", "mop_cv_tau10", "mop_cv_tau20",
                 "mop_cv_tau30", "tau_bound", "weak_mop_tau10", "weak_folklore"}
check("returned keys == documented set", set(res.keys()) == expected_keys,
      str(sorted(set(res.keys()) ^ expected_keys)))

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
