"""tsecon.summarize labels ANY vector whose length equals a string-list sibling
with that list's names. New outputs put unrelated equal-length vectors next to
`param_names` / `names`, so the generic summary silently mislabels them:
ardl_bounds_test's bounds (crit_levels/crit_lower/crit_upper, always length 4)
get the UECM parameter names whenever the UECM has 4 parameters (the most
basic ARDL(1,1) with one regressor and the default case=3), and panel_re's
per-period `periods_used` flags in panel_fama_macbeth get coefficient names
whenever T == K."""
import numpy as np, tsecon
rng = np.random.default_rng(0)
x = np.cumsum(rng.standard_normal((200, 1)), axis=0)
y = 0.5 * x[:, 0] + rng.standard_normal(200)
r = tsecon.ardl_bounds_test(y, x, lags=1, order=1)          # case=3 default
print("param_names:", r["param_names"])
for line in tsecon.summarize(r).summary().splitlines():
    if line.startswith(("crit_levels", "crit_lower", "crit_upper")):
        print("  ", line)
# panel_fama_macbeth with T == K (3 periods, const + 2 regressors)
N, T = 20, 3
xs = rng.standard_normal((2, N, T)); yp = xs[0] + rng.standard_normal((N, T))
p = tsecon.panel_fama_macbeth(yp, xs)
print("names:", p["names"], " periods_used:", p["periods_used"])
for line in tsecon.summarize(p).summary().splitlines():
    if line.startswith("periods_used"):
        print("  ", line, "   <- one flag per PERIOD, not per coefficient")
