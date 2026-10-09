"""Valid calls before and after the round-16 VAR-numerics fixes.

The balanced companion solve and the Jacobi symmetric root change numbers of
valid calls at the last bits. This shows they move TOWARD an independent
reference and that nothing else moves:

    python before_after_valid_calls.py dump before.json   # with the 0.11 batch-2 wheel
    python before_after_valid_calls.py dump after.json    # with the fixed wheel
    PYTHONPATH=<dir with mpmath> python before_after_valid_calls.py compare before.json after.json

Inputs: the 16 JMulTi residual matrices and the six VAR fits of
fixtures/var_diag.json, plus five seeded skewed panels (k = 1, 2, 3, 5, 8).
References: mpmath at 60 digits — the symmetric-root Jarque-Bera of each
matrix, and the companion roots of each fit's own coefficients.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]


def inputs():
    d = json.loads((REPO / "fixtures" / "var_diag.json").read_text())
    mats = [np.asarray(c["resid"]) for c in d["jmulti"]]
    rng = np.random.default_rng(5)
    for k in (1, 2, 3, 5, 8):
        mats.append(rng.standard_normal((120, k)) ** 3 + rng.standard_normal((120, k)))
    return d, mats


def plain(v):
    return v.tolist() if isinstance(v, np.ndarray) else v


def dump(path):
    import tsecon

    d, mats = inputs()
    out = {"mvn": [], "diag": [], "roots": [], "params": []}
    for m in mats:
        out["mvn"].append({k: plain(v) for k, v in tsecon.multivariate_normality(m).items()})
    for c in d["cases"]:
        y = np.asarray(d["series"][c["series"]])
        r = tsecon.var_diagnostics(y, lags=c["p"], trend=c["trend"])
        out["diag"].append({k: plain(v) for k, v in r.items()})
        f = tsecon.var_fit(y, lags=c["p"], trend=c["trend"])
        out["roots"].append([f["min_root"], f["max_root"], f["is_stable"]])
        out["params"].append(plain(f["params"]))
    Path(path).write_text(json.dumps(out))


def mp_sym_jb(x, mp):
    n, k = x.shape
    m = mp.matrix([[mp.mpf(float(v)) for v in row] for row in x])
    mean = [mp.fsum(m[i, j] for i in range(n)) / n for j in range(k)]
    u = mp.matrix(n, k)
    for i in range(n):
        for j in range(k):
            u[i, j] = m[i, j] - mean[j]
    e, q = mp.eigsy((u.T * u) / n)
    w = (q * mp.diag([1 / mp.sqrt(v) for v in e]) * q.T) * u.T
    b1 = [mp.fsum(w[j, i] ** 3 for i in range(n)) / n for j in range(k)]
    b2 = [mp.fsum(w[j, i] ** 4 for i in range(n)) / n - 3 for j in range(k)]
    return float(n * mp.fsum(b**2 for b in b1) / 6 + n * mp.fsum(b**2 for b in b2) / 24)


def compare(before, after):
    import mpmath as mp

    mp.mp.dps = 60
    b = json.loads(Path(before).read_text())
    a = json.loads(Path(after).read_text())
    changed = set()
    for key in ("mvn", "diag"):
        for x, y in zip(b[key], a[key]):
            changed |= {f"{key}:{k}" for k in x if x[k] != y[k]}
    print("keys that changed:", sorted(changed))
    print("var_fit params identical:", b["params"] == a["params"])
    d, mats = inputs()
    ref = [mp_sym_jb(m, mp) for m in mats]
    eo = [abs(x["jarque_bera_symmetric"] - r) / r for x, r in zip(b["mvn"], ref)]
    en = [abs(y["jarque_bera_symmetric"] - r) / r for y, r in zip(a["mvn"], ref)]
    print(f"symmetric JB vs mpmath, {len(ref)} matrices: before max {max(eo):.1e} mean "
          f"{np.mean(eo):.1e}; after max {max(en):.1e} mean {np.mean(en):.1e}")
    worst_b = worst_a = 0.0
    for c, rb, ra, params in zip(d["cases"], b["roots"], a["roots"], a["params"]):
        p, k = c["p"], len(params[0])
        nt = 1 if c["trend"] == "c" else 0
        pm = np.asarray(params)
        comp = np.zeros((k * p, k * p))
        comp[:k] = np.hstack([pm[nt + i * k: nt + (i + 1) * k].T for i in range(p)])
        comp[k:, : k * (p - 1)] = np.eye(k * (p - 1))
        mods = [abs(e) for e in mp.eig(mp.matrix(comp.tolist()), left=False, right=False)]
        mn, mx = float(1 / max(mods)), float(1 / min(mods))
        eb = max(abs(rb[0] - mn) / mn, abs(rb[1] - mx) / mx)
        ea = max(abs(ra[0] - mn) / mn, abs(ra[1] - mx) / mx)
        worst_b, worst_a = max(worst_b, eb), max(worst_a, ea)
        print(f"{c['name']:22s} min/max_root vs mpmath: before {eb:.1e} after {ea:.1e}")
    print(f"worst: before {worst_b:.1e} after {worst_a:.1e}")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
