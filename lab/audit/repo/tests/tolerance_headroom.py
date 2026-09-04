"""Achieved error behind the Python re-checks that are looser than the matrix.

tolerance_vs_matrix.py flags seven Python golden re-checks whose literal is
looser than the tolerance the validation matrix documents for the family
(the Rust pin). This reproduces each comparison and prints the achieved
max error next to the asserted literal and the documented one, so the
proposal "tighten to the matrix value" is backed by measurement rather than
by the Rust pin alone. Nothing here changes a test.

Run:  .venv-wt/bin/python lab/audit/repo/tests/tolerance_headroom.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FIX = os.path.join(REPO, "fixtures")
sys.path.insert(0, os.path.join(REPO, "bindings", "python", "tests"))

import tsecon  # noqa: E402


def load(name):
    return json.load(open(os.path.join(FIX, name)))


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def absd(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


rows = []

# test_favar.py:44-45 / test_depth.py:89-90 — |PC1|,|PC2| vs numpy PCA (favar.json)
fav = load("favar.json")
xs = np.array(fav["X_standardized"]).T
res = tsecon.factor_model(xs, n_factors=fav["true_r"], kmax=8)
f = np.array(res["factors"])
rows.append(("test_depth.py:89  |PC1| atol", absd(np.abs(f[:, 0]), fav["pc1_abs"]), "atol=1e-5", "1e-6 rel (favar row)"))
rows.append(("test_depth.py:90  |PC2| atol", absd(np.abs(f[:, 1]), fav["pc2_abs"]), "atol=1e-5", "1e-6 rel (favar row)"))
n = xs.shape[0]
policy = np.random.default_rng(0).standard_normal(n)  # as test_favar.py builds it
r_fav = tsecon.favar(xs, policy, n_factors=fav["true_r"], lags=1, horizon=4)
if r_fav is not None:
    ff = np.array(r_fav["factors"])
    rows.append(("test_favar.py:44  |PC1| atol", absd(np.abs(ff[:, 0]), fav["pc1_abs"]), "atol=1e-4", "1e-6 rel (favar row)"))
    rows.append(("test_favar.py:45  |PC2| atol", absd(np.abs(ff[:, 1]), fav["pc2_abs"]), "atol=1e-4", "1e-6 rel (favar row)"))

# test_gmm.py:39-40 — Hansen J and p vs linearmodels (gmm.json)
g = load("gmm.json")
Y = np.array(g["y"]); ones = np.ones_like(Y)
X = np.column_stack([ones, np.array(g["w"]), np.array(g["x"])])
Z = np.column_stack([ones, np.array(g["w"]), np.array(g["z1"]), np.array(g["z2"])])
fit = tsecon.iv_gmm(X, Z, Y, method="2step", weight="robust")
rows.append(("test_gmm.py:39  j_stat abs", abs(fit["j_stat"] - g["ivgmm"]["j_stat"]), "< 1e-4", "1e-6 (gmm row)"))
rows.append(("test_gmm.py:40  j_pval abs", abs(fit["j_pval"] - g["ivgmm"]["j_pval"]), "< 1e-4", "1e-6 (gmm row)"))

# test_predreg.py:40-42 — IVX beta / Wald (predreg.json)
p = load("predreg.json")
sc = p["scalar"]
try:
    res = tsecon.predictive_regression(np.array(sc["r"]), np.array(sc["x"]))
    iv = sc["ivx"]
    rows.append(("test_predreg.py:40  beta_ivx abs", abs(res["ivx"]["beta_ivx"] - iv["beta_ivx"]), "< 1e-6", "1e-9 (predreg row)"))
    rows.append(("test_predreg.py:41  wald abs", abs(res["ivx"]["wald"] - iv["wald"]), "< 1e-5", "1e-9 (predreg row)"))
except Exception as e:  # noqa: BLE001
    rows.append(("test_predreg.py  (could not reproduce: %s)" % e, float("nan"), "", ""))

# test_roadmap_gaps.py:31-35 — recession probit vs statsmodels (tsecon-recession.json)
rec = load("tsecon-recession.json")
y = np.array(rec["y"], float)
x = np.column_stack([rec["const"], rec["spread"], rec["lead"]])
if x is not None:
    r = tsecon.recession_probit(y, x, link="probit")
    gg = rec["probit"]
    rows.append(("test_roadmap_gaps.py:31  params atol", absd(r["params"], gg["params"]), "atol=1e-5", "1e-6 (recession row)"))
    rows.append(("test_roadmap_gaps.py:32  bse atol", absd(r["bse"], gg["bse"]), "atol=1e-5", "1e-6 (recession row)"))
    rows.append(("test_roadmap_gaps.py:33  loglik abs", abs(r["loglik"] - gg["llf"]), "< 1e-4", "1e-6 (recession row)"))
    rows.append(("test_roadmap_gaps.py:35  probabilities atol", absd(r["probabilities"], gg["fitted"]), "atol=1e-5", "1e-6 (recession row)"))

# test_spectest_afns_dsge.py:133 — AFNS adjustment (afns.json)
af = load("afns.json")
worst = 0.0
for case in af["cases"]:
    out = tsecon.afns_adjustment(np.asarray(case["maturities"], float), np.asarray(case["sigma_diag"], float), decay=case["lambda"])
    worst = max(worst, rel(out, case["adjustment"]))
rows.append(("test_spectest_afns_dsge.py:133  adjustment rel", worst, "rtol=1e-9", "1e-10 (afns row)"))

# test_proxy_first_stage.py:90 — tau_bound (proxy_first_stage.json)
pf = load("proxy_first_stage.json")
worst = 0.0
for case in pf["cases"]:
    e = case["expected"]
    tb = e.get("tau_bound")
    if tb is None:
        continue
    try:
        proxy = np.array([np.nan if v is None else v for v in case["proxy"]], dtype=float)
        kw = {"lags": 0, "norm_var": case["norm_var"]}
        if case["hac_lags"] is not None:
            kw.update(variance="hac", hac_lags=case["hac_lags"])
        d = tsecon.proxy_first_stage(np.asarray(case["u"]), proxy, **kw)
        worst = max(worst, abs(d["tau_bound"] - tb) / abs(tb))
    except Exception:  # noqa: BLE001
        worst = float("nan")
        break
rows.append(("test_proxy_first_stage.py:90  tau_bound rel", worst, "rel=1e-5", "1e-6 (first-stage row)"))

print(f"{'check':46s} {'achieved':>12s}  {'asserted':10s} {'documented'}")
for name, err, asserted, documented in rows:
    print(f"{name:46s} {err:12.3e}  {asserted:10s} {documented}")
