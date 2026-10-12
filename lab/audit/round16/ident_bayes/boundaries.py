"""Probe both sides of each size budget: the largest value a refusal quotes
must run (within the 4 GB cap and the deadline), the next must be refused by
name. Each call runs in its own capped subprocess.

usage: boundaries.py [case-substring ...]  (one case per invocation keeps each
run under the 5-minute deadline; out/boundaries.txt concatenates them)
"""
import os
import re
import subprocess
import sys

PY = sys.executable
FIX = r'''
import numpy as np, tsecon, time, resource, warnings
warnings.simplefilter("ignore")
def var(T, k, seed):
    r = np.random.default_rng(seed)
    A = 0.45 * np.eye(k) + 0.08
    X = np.zeros((T, k))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + r.standard_normal(k)
    return X
X3 = var(160, 3, 101)
X2 = var(160, 2, 202)
X4 = var(240, 4, 303)
SR3 = [(0, 0, 0, "+"), (1, 0, 0, "+")]
NR = [{"type": "shock_sign", "shock": 0, "period": 12, "sign": "+"}]
PROXY = X3[:, 0] + np.random.default_rng(7).standard_normal(160)
REG = np.where(np.arange(160) < 80, 0, 1)
XH = X2 * np.where(REG[:, None] == 1, 2.0, 1.0) * np.array([1.0, 0.5])
r = np.random.default_rng(8)
e = np.column_stack([r.laplace(size=160), r.standard_t(4, size=160)])
NG = e + 0.3 * e[:, ::-1]
f = np.cumsum(r.standard_normal(90)) * 0.1
PANEL = np.column_stack([f + r.standard_normal(90) for _ in range(5)])
POL = var(90, 1, 9)[:, 0]
'''

CHILD = FIX + r'''
V = {v}
t0 = time.time()
try:
    out = {call}
    print("RESULT ok %.2fs maxrss %.0f MB" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
except Exception as ex:
    print("RESULT raised %.2fs %s: %s" % (time.time() - t0, type(ex).__name__, str(ex)[:900]))
'''

CASES = {
    "structural_fevd:horizon": ("horizon", "tsecon.structural_fevd(X2, lags=1, horizon=V)"),
    "structural_fevd_impact:horizon": ("horizon", "tsecon.structural_fevd(X3, lags=2, horizon=V, impact=np.eye(3))"),
    "hetero_svar:horizon": ("horizon", "tsecon.hetero_svar(XH, REG, lags=1, horizon=V)"),
    "max_share_svar:horizon": ("horizon", "tsecon.max_share_svar(X3, lags=1, h1=8, horizon=V)"),
    "proxy_svar:horizon": ("horizon", "tsecon.proxy_svar(X3, PROXY, lags=2, horizon=V)"),
    "nongaussian_svar:horizon": ("horizon", "tsecon.nongaussian_svar(NG, lags=1, horizon=V)"),
    "favar:horizon": ("horizon", "tsecon.favar(PANEL, POL, n_factors=1, lags=1, horizon=V)"),
    "bvar_irf_draws:horizon": ("horizon", "tsecon.bvar_irf_draws(X3, lags=1, horizon=V, n_draws=20)"),
    "bvar_irf_draws:n_draws": ("n_draws", "tsecon.bvar_irf_draws(X3, lags=1, horizon=8, n_draws=V)"),
    "bvar_ssvs:horizon": ("horizon", "tsecon.bvar_ssvs(X2, lags=1, n_draws=120, burn=20, horizon=V)"),
    "bvar_ssvs:n_draws": ("n_draws", "tsecon.bvar_ssvs(X2, lags=1, n_draws=V, burn=20, horizon=4)"),
    "bvar_ssvs:n_chains": ("n_chains", "tsecon.bvar_ssvs(X2, lags=1, n_draws=120, burn=20, horizon=4, n_chains=V)"),
    "sign:horizon": ("horizon", "tsecon.sign_restricted_svar(X3, SR3, lags=1, horizon=V, n_draws=20, max_tries=100)"),
    "sign:n_draws": ("n_draws", "tsecon.sign_restricted_svar(X3, SR3, lags=1, horizon=4, n_draws=V, max_tries=100)"),
    "zero_sign:n_draws": ("n_draws", "tsecon.zero_sign_svar(X3, SR3, [(2, 0, 0)], lags=1, horizon=4, n_draws=V, max_tries=100)"),
    "fry_pagan:n_draws": ("n_draws", "tsecon.fry_pagan_svar(X3, SR3, lags=1, horizon=4, n_draws=V, max_tries=100)"),
    "narrative:n_draws": ("n_draws", "tsecon.narrative_svar(X3, SR3, lags=1, horizon=4, n_draws=V, max_tries=100)"),
    "robust:horizon": ("horizon", "tsecon.robust_svar_bounds(X3, SR3, lags=1, horizon=V, n_draws=20)"),
    "robust:n_draws": ("n_draws", "tsecon.robust_svar_bounds(X3, SR3, lags=1, horizon=4, n_draws=V)"),
    "hd_sign:n_draws": ("n_draws", "tsecon.historical_decomposition(X3, SR3, lags=1, identification='sign', n_draws=V, max_tries=100)"),
    "hd_sign_h:n_draws": ("n_draws", "tsecon.historical_decomposition(X3, [(0, 0, 0, '+'), (1, 0, 150, '+')], lags=1, horizon=158, identification='sign', n_draws=V, max_tries=100)"),
    "hd_sign:horizon": ("horizon", "tsecon.historical_decomposition(X3, SR3, lags=1, horizon=V, identification='sign', n_draws=20, max_tries=100)"),
    "proxy_bands:horizon": ("horizon", "tsecon.proxy_svar_bands(X3, PROXY, lags=2, horizon=V, n_boot=20, block_length=5)"),
    "proxy_bands:n_boot": ("n_boot", "tsecon.proxy_svar_bands(X3, PROXY, lags=2, horizon=6, n_boot=V, block_length=5)"),
    "proxy_ar_mc:rf_draws": ("rf_draws", "tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=6, rf_method='second_order', rf_draws=V, rf_seed=1)"),
    "proxy_ar_delta:horizon": ("horizon", "tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=V)"),
    "narrative_w:n_weight_draws": ("n_weight_draws", "tsecon.narrative_svar(X3, SR3, NR, lags=1, horizon=4, n_draws=10, max_tries=200, n_weight_draws=V)"),
    "narrative_w4:n_weight_draws": ("n_weight_draws", "tsecon.narrative_svar(X4, [(0, 0, 0, '+'), (1, 0, 0, '+')], NR, lags=1, horizon=4, n_draws=40, max_tries=400, n_weight_draws=V)"),
    "narrative_nr:n_draws": ("n_draws", "tsecon.narrative_svar(X3, SR3, NR, lags=1, horizon=4, n_draws=V, max_tries=100)"),
    "hd_sign_nr:n_draws": ("n_draws", "tsecon.historical_decomposition(X3, SR3, lags=1, identification='sign', narrative_restrictions=NR, n_draws=V, max_tries=100)"),
    "proxy_ar_mc:horizon": ("horizon", "tsecon.proxy_ar_sets(X3, PROXY, lags=2, horizon=V, rf_method='second_order', rf_draws=256, rf_seed=1)"),
    "bvar_hier:n_grid": ("n_grid", "tsecon.bvar_hierarchical(X3, lags=1, n_grid=V)"),
}


def child(call, v, timeout=200):
    code = CHILD.replace("{v}", str(v)).replace("{call}", call)
    p = subprocess.run(
        ["bash", "-c", f"ulimit -v 4000000; timeout {timeout} {PY} -c \"$CODE\""],
        env={**os.environ, "CODE": code}, capture_output=True, text=True,
    )
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT")]
    if lines:
        return lines[-1]
    return f"RESULT died rc={p.returncode} {p.stderr.strip()[-200:]}"


def main(filters):
    for name, (arg, call) in CASES.items():
        # An exact case name selects that case alone; anything else is a
        # substring filter.
        if filters and not any(f == name or (f not in CASES and f in name) for f in filters):
            continue
        r = child(call, 2**40, timeout=60)
        m = re.search(rf"pass {arg} <= (\d+)", r)
        if not m:
            print(f"{name}: 2**40 -> {r}", flush=True)
            continue
        n = int(m.group(1))
        print(f"{name}: quoted max {arg} <= {n}", flush=True)
        print(f"  {arg}={n}: {child(call, n)}", flush=True)
        r1 = child(call, n + 1, timeout=60)
        named = f"{arg} = {n + 1} is too large" in r1
        print(f"  {arg}={n + 1}: named={named} {r1[:160]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
