"""Round-8 probe: acm_term_premium.

Lenses 1 (axes), 3 (degenerate), 5 (doc-key promises), internal identities:
fitted = risk_neutral + term_premium; short-rate equation; annualization.
"""
import numpy as np
import tsecon

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


# --- simulate a plausible yield panel (Nelson-Siegel-ish + noise), monthly ---
rng = np.random.default_rng(80803)
T = 360
mats = np.arange(1, 121)  # 1..120 months
M = len(mats)
# three latent factors with AR dynamics
L = np.zeros(T); S = np.zeros(T); C = np.zeros(T)
L[0], S[0], C[0] = 0.05, -0.01, 0.01
for t in range(1, T):
    L[t] = 0.0005 + 0.99 * L[t-1] + 0.0007 * rng.standard_normal()
    S[t] = 0.97 * S[t-1] + 0.0009 * rng.standard_normal()
    C[t] = 0.95 * C[t-1] + 0.0010 * rng.standard_normal()
lam = 0.06
yields = np.zeros((T, M))
for j, m in enumerate(mats):
    x = lam * m
    f1 = (1 - np.exp(-x)) / x
    f2 = f1 - np.exp(-x)
    yields[:, j] = L + S * f1 + C * f2 + 0.0002 * rng.standard_normal(T)
yields = np.clip(yields, 0.0001, None)

res = tsecon.acm_term_premium(yields, mats, n_factors=5, periods_per_year=12.0)

# --- doc-key check ---
doc_keys = {"factors", "factor_loadings", "mu", "phi", "sigma", "rx_maturities",
            "a", "beta", "c", "sigma2", "lambda0", "lambda1", "delta0", "delta1",
            "A", "B", "A_rn", "B_rn", "fitted", "risk_neutral", "term_premium",
            "var_rsquared", "rx_rsquared", "short_rate_rsquared", "yield_rsquared"}
got = set(res.keys())
check("returned keys cover documented set", doc_keys <= got, str(sorted(doc_keys - got)))
extra = got - doc_keys
print("extra keys (undocumented?):", sorted(extra))

fitted = np.asarray(res["fitted"]); rn = np.asarray(res["risk_neutral"]); tp = np.asarray(res["term_premium"])
check("shapes T x M", fitted.shape == (T, M) and rn.shape == (T, M) and tp.shape == (T, M),
      str(fitted.shape))
check("fitted = risk_neutral + term_premium (exact identity, 1e-12)",
      np.max(np.abs(fitted - (rn + tp))) < 1e-12, f"max {np.max(np.abs(fitted - (rn + tp))):.2e}")

# maturity-1: the model's one-period rate; premium there should be ~0
j1 = 0
print(f"[note] term premium at maturity 1: max |tp| = {np.max(np.abs(tp[:, j1])):.3e}")
check("term premium ~ 0 at maturity 1", np.max(np.abs(tp[:, j1])) < 1e-10)

# fit quality of the affine model on smooth data
yr = np.asarray(res["yield_rsquared"], dtype=float)
check("yield_rsquared high on smooth panel", np.min(yr) > 0.9, f"min {np.min(yr):.4f}")
print("[note] rsq shapes:", {k: np.shape(res[k]) for k in
      ["var_rsquared", "rx_rsquared", "short_rate_rsquared", "yield_rsquared"]})
print(f"[note] short_rate_rsquared = {float(np.atleast_1d(res['short_rate_rsquared'])[0]):.4f}")

# short-rate equation: delta0 + X delta1 should track the 1-period yield
X = np.asarray(res["factors"])
sr = res["delta0"] + X @ np.asarray(res["delta1"])
resid_sr = yields[:, 0] * 1.0 - sr * 12.0 / 12.0  # both annualized decimal?
# check units: fitted[:, 0] should approximate yields[:, 0]
check("fitted maturity-1 tracks observed 1m yield",
      np.corrcoef(fitted[:, 0], yields[:, 0])[0, 1] > 0.99,
      f"corr {np.corrcoef(fitted[:, 0], yields[:, 0])[0, 1]:.4f}")
check("fitted level is in decimal units (mean within 3x of input mean)",
      0.3 < np.mean(fitted) / np.mean(yields) < 3.0, f"{np.mean(fitted) / np.mean(yields):.3f}")

# --- lens 1: n_factors axis ---
r3 = tsecon.acm_term_premium(yields, mats, n_factors=3)
check("n_factors axis alive", not np.allclose(np.asarray(r3["term_premium"]), tp))
check("n_factors reflected in factors shape", np.asarray(r3["factors"]).shape[1] == 3)

# periods_per_year axis: affects annualization
r_q = tsecon.acm_term_premium(yields, mats, n_factors=5, periods_per_year=4.0)
check("periods_per_year axis alive", not np.allclose(np.asarray(r_q["fitted"]), fitted))

# --- rx_maturities: n >= 2 with n-1 in grid ---
rxm = np.asarray(res["rx_maturities"])
check("rx_maturities are the n>=2 with n-1 present", np.array_equal(rxm, mats[1:]), str(rxm[:5]))

# sparse grid: only pairs around return maturities get excess returns
sparse = np.array([1, 2, 12, 13, 60, 61, 120])
r_sp = tsecon.acm_term_premium(yields[:, np.searchsorted(mats, sparse)], sparse, n_factors=2)
rxm_sp = np.asarray(r_sp["rx_maturities"])
check("sparse grid rx_maturities = {2,13,61}", set(rxm_sp.tolist()) == {2, 13, 61}, str(rxm_sp))

# --- lens 3: degenerate inputs ---
expect_raise("maturities missing 1", lambda: tsecon.acm_term_premium(yields[:, 1:], mats[1:], n_factors=3))
expect_raise("non-ascending maturities", lambda: tsecon.acm_term_premium(yields, mats[::-1].copy(), n_factors=3))
expect_raise("n_factors > M", lambda: tsecon.acm_term_premium(yields[:, :4], mats[:4], n_factors=6))
expect_raise("NaN yield", lambda: tsecon.acm_term_premium(np.where(np.arange(T)[:, None] == 5, np.nan, yields), mats, n_factors=3))
expect_raise("too-short panel", lambda: tsecon.acm_term_premium(yields[:4], mats, n_factors=3))
expect_raise("n_factors = 0", lambda: tsecon.acm_term_premium(yields, mats, n_factors=0))
# grid with no adjacent pair -> no excess returns possible
try:
    iso = np.array([1, 12, 36, 60])
    tsecon.acm_term_premium(yields[:, np.searchsorted(mats, iso)], iso, n_factors=2)
    print("[note] no-adjacent-pair grid RAN (should it?)")
    fails.append(("no-adjacent-pair grid should raise (no excess returns can be built)", "ran"))
except Exception as e:
    print(f"[ok] no-adjacent-pair grid raises: {type(e).__name__}: {str(e)[:100]}")
attempted += 1; made += 1

# percent-units input (the doc's 100x warning): does anything catch it?
try:
    r_pct = tsecon.acm_term_premium(yields * 100.0, mats, n_factors=5)
    print(f"[note] percent input runs; yield_rsq={r_pct['yield_rsquared']:.4f}, "
          f"mean tp={np.mean(np.asarray(r_pct['term_premium'])):.4f} (decimal-input mean tp="
          f"{np.mean(tp):.6f})")
except Exception as e:
    print(f"[note] percent input raises: {type(e).__name__}: {str(e)[:80]}")

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
