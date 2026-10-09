"""Golden fixtures for equal-weighted cosine (EWC) HAR inference in
`tsecon-hac`: `ewc_lrv`, `ewc_default_b`, the matrix form `ewc_lrv_matrix`, and
`OlsFit::inference(SeType::Ewc { b })` / Python `ols(se_type="ewc")`.

GRADE: DOCUMENTED CLOSED FORM, NOT AN INDEPENDENT PACKAGE
---------------------------------------------------------
No reference library implements EWC. statsmodels, `arch` and `linearmodels`
were searched (the ABSENCE CANARY below greps every installed source file of
the three for "ewc" / "equal-weighted cosine" and stores the hit counts, which
must be zero), so the golden is the published closed form of
Lazarus, Lewis, Stock & Watson (2018, JBES, "HAR Inference: Recommendations
for Practice"), evaluated through an INDEPENDENT CODE PATH: SciPy's FFT-based
type-II discrete cosine transform rather than the direct cosine sum the crate
runs.

    Lambda_j  = sqrt(2/n) sum_{t=1..n} cos(pi j (t - 1/2) / n) z_t,  j = 1..B
              = scipy.fft.dct(z, type=2, norm="ortho", axis=0)[1:B+1]
    Omega_hat = (1/B) sum_{j=1..B} Lambda_j Lambda_j'      (k x k; scalar for k = 1)
    B_default = round(0.4 n^(2/3)), clamped to [1, n - 1]

(The ortho-normalised type-II DCT is sqrt(2/n) sum_t cos(pi j (2t-1) / (2n)) z_t
for j >= 1, which is the LLSW Lambda_j term for term; the generator asserts the
two agree on every stored case before writing anything.)

OLS-EWC (the `ols(se_type="ewc")` golden). With statsmodels 0.15.0
`OLS(y, X).fit()` supplying the coefficients and residuals u_t:

    z_t       = x_t u_t                      (the OLS scores, n x k)
    cov       = (X'X)^{-1} [n Omega_hat(z)] (X'X)^{-1}
                (no n/(n-k) factor, no prewhitening; the meat n Omega_hat is
                 the statsmodels HAC "sum" scaling)
    bse       = sqrt(diag cov),  tvalues = params / bse
    pvalues   = 2 * scipy.stats.t(df=B).sf(|tvalues|)     (Student t_B, not normal)

`ewc_default_b` is pinned on the n the spec lists plus the ten n in 2..10^6
whose 0.4 n^(2/3) lands CLOSEST to a half-integer, where a powf/rounding
difference between two implementations would show first. (Exact .5 ties are
impossible: n^(2/3) is rational only for n = m^3, and 0.4 m^2 never has
fractional part 1/2.) The reference B is computed EXACTLY, in integer
arithmetic, not with a float power (audit round 15: a double-precision
`0.4 * n**(2/3)` carries ~1e-15 relative error, enough to round the wrong way
when the true value sits within ~1e-6 of a half-integer, first at
n = 1,755,027,219):

    round(0.4 n^(2/3)) = r  <=>  125 (2r - 1)^3 <= 64 n^2 < 125 (2r + 1)^3

(cube 0.4 n^(2/3) >= r - 1/2 and 0.4 n^(2/3) < r + 1/2). `default_b_large_n`
adds the n in 10^6..3e10 whose 0.4 n^(2/3) lands closest to a half-integer
(found by an exact search over every tie r - 1/2 in that range), plus the
round-15 auditor's two cases, with the signed exact distance
0.4 n^(2/3) - (r - 1/2) in `default_b_large_n_gap` (60-digit Decimal).

Data: the Nile series (`sm.datasets.nile`, public domain — the tests load the
copy already stored in `diagnostics.json`, asserted identical here, so no new
copy of it is written) and seeded AR(1) draws (stored).

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr, which the Rust golden tests parse to identical bits
(serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_ewc_fixtures.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import re
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import fft, stats

HERE = Path(__file__).resolve().parent
OUT = HERE / "ewc.json"


# ------------------------------------------------------------------ EWC refs

def ewc_dct(z: np.ndarray, b: int) -> np.ndarray:
    """Omega_hat via SciPy's type-II DCT (ortho). `z` is n or n x k."""
    z2 = z.reshape(len(z), -1)
    lam = fft.dct(z2, type=2, norm="ortho", axis=0)[1:b + 1]  # b x k
    return lam.T @ lam / b


def ewc_direct(z: np.ndarray, b: int) -> np.ndarray:
    """The LLSW formula transcribed literally (generation-time cross-check)."""
    z2 = z.reshape(len(z), -1)
    n = z2.shape[0]
    t = np.arange(1, n + 1) - 0.5
    lam = np.array([np.sqrt(2.0 / n) * (np.cos(np.pi * j * t / n) @ z2) for j in range(1, b + 1)])
    return lam.T @ lam / b


def default_b(n: int) -> int:
    """round(0.4 n^(2/3)) clamped to [1, n - 1], EXACT: the largest r with
    125 (2r - 1)^3 <= 64 n^2 (Python integers; the float value is only the
    starting guess)."""
    m = 64 * n * n
    r = max(1, int(math.floor(0.4 * n ** (2.0 / 3.0) + 0.5)))
    while r > 1 and 125 * (2 * r - 1) ** 3 > m:
        r -= 1
    while 125 * (2 * r + 1) ** 3 <= m:
        r += 1
    return max(1, min(r, n - 1))


def check_close(a: np.ndarray, b: np.ndarray, rtol: float, what: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b), np.max(np.abs(b)) * 1e-12)
    rel = float(np.max(np.abs(a - b) / scale))
    assert rel <= rtol, (what, rel)
    return rel


# ---------------------------------------------------------- absence canary

CANARY_RE = re.compile(r"\bewc\b|equal[- _]weighted[- _]cosine", re.IGNORECASE)


def absence_canary() -> dict:
    """Grep every installed .py file of statsmodels/arch/linearmodels for EWC."""
    out = {}
    for pkg in ("statsmodels", "arch", "linearmodels"):
        spec = importlib.util.find_spec(pkg)
        if spec is None or spec.origin is None:
            out[pkg] = {"installed": False}
            continue
        root = Path(spec.origin).parent
        hits = []
        n_files = 0
        for path in sorted(root.rglob("*.py")):
            n_files += 1
            text = path.read_text(encoding="utf-8", errors="replace")
            if CANARY_RE.search(text):
                hits.append(str(path.relative_to(root)))
        mod = __import__(pkg)
        out[pkg] = {
            "installed": True,
            "version": getattr(mod, "__version__", "?"),
            "py_files_scanned": n_files,
            "files_mentioning_ewc": hits,
        }
        assert not hits, (
            f"{pkg} now mentions EWC in {hits}: a third-party reference may exist — "
            "re-grade the validation-matrix row before regenerating"
        )
    return out


# ------------------------------------------------------------------- series

def nile() -> np.ndarray:
    y = sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)
    stored = json.loads((HERE / "diagnostics.json").read_text(encoding="utf-8"))["nile"]
    assert np.array_equal(np.asarray(stored, dtype=float), y), "diagnostics.json nile drifted"
    return y


def ar1(rng: np.random.Generator, n: int, phi: float, burn: int = 100) -> np.ndarray:
    e = rng.standard_normal(n + burn)
    x = np.empty(n + burn)
    x[0] = e[0] / math.sqrt(1.0 - phi * phi)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


# ------------------------------------------------------------------- blocks

def lrv_block(name: str, x: np.ndarray, bs: list[int]) -> dict:
    """ewc_lrv on the raw series and on the demeaned series, every b in bs."""
    z = x - x.mean()
    raw, dem = {}, {}
    for b in bs:
        for series, store in ((x, raw), (z, dem)):
            v = ewc_dct(series, b)[0, 0]
            check_close(v, ewc_direct(series, b)[0, 0], 1e-11, f"{name} b={b}")
            store[str(b)] = float(v)
        # Mean invariance (no j = 0 term) — exact in exact arithmetic.
        check_close(raw[str(b)], dem[str(b)], 1e-10, f"{name} mean invariance b={b}")
    n = len(x)
    if (n - 1) in bs:
        # b = n - 1: the cosine basis is complete, so EWC = sum z^2 / (n - 1).
        check_close(dem[str(n - 1)], z @ z / (n - 1), 1e-12, f"{name} b=n-1 identity")
    return {
        "n": n,
        "b": bs,
        "default_b": default_b(n),
        "ewc_raw": raw,
        "ewc_demeaned": dem,
        "ewc_demeaned_at_default_b": float(ewc_dct(z, default_b(n))[0, 0]),
    }


def ols_ewc_block(name: str, y: np.ndarray, X: np.ndarray, bs: list[int]) -> dict:
    res = sm.OLS(y, X).fit()
    n, k = X.shape
    scores = X * res.resid[:, None]
    xtx_inv = np.linalg.inv(X.T @ X)
    cases = []
    for b in bs:
        omega = ewc_dct(scores, b)
        check_close(omega, ewc_direct(scores, b), 1e-10, f"{name} omega b={b}")
        cov = xtx_inv @ (n * omega) @ xtx_inv
        bse = np.sqrt(np.diag(cov))
        tv = res.params / bse
        pv = 2.0 * stats.t(df=b).sf(np.abs(tv))
        cases.append({
            "b": b,
            "b_is_default": b == default_b(n),
            "omega": omega.ravel().tolist(),
            "cov": cov.ravel().tolist(),
            "bse": bse.tolist(),
            "tvalues": tv.tolist(),
            "pvalues": pv.tolist(),
            "t_crit_975": float(stats.t(df=b).ppf(0.975)),
        })
    block = {
        "n": n,
        "k": k,
        "default_b": default_b(n),
        "y": y.tolist(),
        "x_cols": [X[:, j].tolist() for j in range(k)],
        "params": res.params.tolist(),
        "cases": cases,
    }
    return block


def near_half_n(limit: int = 1_000_000, count: int = 10) -> list[int]:
    n = np.arange(2, limit + 1, dtype=float)
    v = 0.4 * n ** (2.0 / 3.0)
    dist = np.abs(v - np.floor(v) - 0.5)
    idx = np.argsort(dist, kind="stable")[:count]
    return sorted(int(n[i]) for i in idx)


def half_gap(n: int) -> Decimal:
    """0.4 n^(2/3) - (r - 1/2) at r = default_b(n), to 60 digits (positive: the
    value sits just above the half-integer, so it rounds up to r)."""
    with localcontext() as ctx:
        ctx.prec = 60
        v = Decimal(4) / Decimal(10) * ((Decimal(n) ** 2).ln() / 3).exp()
        return v - (Decimal(default_b(n)) - Decimal("0.5"))


def near_half_n_large(lo: int = 1_000_000, hi: int = 30_000_000_000, count: int = 10) -> list[int]:
    """The n in [lo, hi] closest to a rounding tie of 0.4 n^(2/3).

    The tie r - 1/2 sits at n* = (2.5 r - 1.25)^(3/2); every integer n has its
    0.4 n^(2/3) nearest some tie, and the nearest integer to n* is the best
    candidate for that tie. A float pass ranks every tie in range (the float
    gap is accurate to ~1e-8 here, far below the ranking it feeds), the 400
    best are re-measured exactly (`half_gap`), and the `count` smallest |gap|
    are kept, each with its exact B from the integer rule."""
    r_lo = int(0.4 * lo ** (2.0 / 3.0))
    r_hi = int(0.4 * hi ** (2.0 / 3.0)) + 1
    r = np.arange(max(r_lo, 2), r_hi + 1, dtype=np.float64)
    n_star = (2.5 * r - 1.25) ** 1.5
    cand = np.concatenate([np.floor(n_star), np.ceil(n_star)])
    cand = cand[(cand >= lo) & (cand <= hi)]
    v = 0.4 * cand ** (2.0 / 3.0)
    gap = np.abs(v - np.floor(v) - 0.5)
    best = np.unique(cand[np.argsort(gap, kind="stable")[:400]].astype(np.int64))
    exact = sorted((abs(half_gap(int(n))), int(n)) for n in best)
    return sorted(n for _, n in exact[:count])


def main() -> None:
    canary = absence_canary()

    y_nile = nile()
    rng = np.random.default_rng(20261008)
    x_ar = ar1(rng, 250, 0.6)

    lrv = {
        "nile": lrv_block("nile", y_nile, [1, 4, 9, 16, len(y_nile) - 1]),
        "ar1_n250": lrv_block("ar1", x_ar, [1, 4, 9, 16, 249]),
    }

    spec_n = [2, 3, 50, 100, 200, 400, 1000, 12345]
    tricky = near_half_n()
    default_b_cases = {str(n): default_b(n) for n in sorted(set(spec_n + tricky))}
    gaps = {str(n): float(abs(0.4 * n ** (2 / 3) - math.floor(0.4 * n ** (2 / 3)) - 0.5)) for n in tricky}
    # Beyond 10^6 the float rule starts to misround; the exact rule is the golden.
    large = sorted(set(near_half_n_large() + [1_755_027_219, 100_292_764_931_117]))
    default_b_large = {str(n): default_b(n) for n in large}
    gaps_large = {str(n): float(half_gap(n)) for n in large}
    assert default_b_large["1755027219"] == 581991, "auditor case 1"
    assert default_b_large["100292764931117"] == 863455038, "auditor case 2"

    # OLS-EWC, k = 2: the spec's T = 250 design (x AR(1) 0.6, u AR(1) 0.5).
    rng2 = np.random.default_rng(20261008)
    T = 250
    x = np.empty(T)
    x[0] = rng2.standard_normal()
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng2.standard_normal()
    u = np.empty(T)
    u[0] = rng2.standard_normal()
    for t in range(1, T):
        u[t] = 0.5 * u[t - 1] + rng2.standard_normal()
    y2 = 1.0 + 0.5 * x + u
    X2 = sm.add_constant(x)

    # OLS-EWC, k = 3: correlated persistent regressors, persistent errors, T = 180.
    rng3 = np.random.default_rng(7310)
    T3 = 180
    x1 = ar1(rng3, T3, 0.7)
    x2 = 0.5 * x1 + ar1(rng3, T3, 0.3)
    e3 = ar1(rng3, T3, 0.6)
    y3 = -0.4 + 0.3 * x1 - 0.2 * x2 + e3
    X3 = sm.add_constant(np.column_stack([x1, x2]))

    ols_blocks = {
        "k2_t250": ols_ewc_block("k2", y2, X2, [default_b(T), 8]),
        "k3_t180": ols_ewc_block("k3", y3, X3, [default_b(T3), 8]),
    }

    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "grade": "documented closed form (LLSW 2018) via scipy.fft.dct; no package implements EWC",
            "absence_canary": canary,
            "note": (
                "EWC long-run variance Omega_hat = (1/B) sum_{j=1..B} Lambda_j Lambda_j', "
                "Lambda_j = sqrt(2/n) sum_t cos(pi j (t-1/2)/n) z_t, computed with "
                "scipy.fft.dct(type=2, norm='ortho') and cross-checked against the direct "
                "cosine sum at generation. OLS-EWC: statsmodels OLS params/resid, "
                "cov = (X'X)^-1 (n Omega_hat(X*u)) (X'X)^-1, p-values 2*scipy.stats.t(B).sf(|t|). "
                "The Nile series is read from diagnostics.json (asserted equal to "
                "sm.datasets.nile); the seeded AR(1) series is stored."
            ),
        },
        "series": {"ar1_n250": x_ar.tolist()},
        "lrv": lrv,
        "default_b": default_b_cases,
        "default_b_near_half_gap": gaps,
        "default_b_large_n": default_b_large,
        "default_b_large_n_gap": gaps_large,
        "ols": ols_blocks,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for name, blk in lrv.items():
        print(f"  lrv {name}: n={blk['n']} default_b={blk['default_b']} "
              f"demeaned={blk['ewc_demeaned']}")
    print("  default_b:", default_b_cases)
    print("  default_b_large_n:", default_b_large)
    print("  default_b_large_n_gap:", gaps_large)
    print("  near-half gaps:", gaps)
    for name, blk in ols_blocks.items():
        for c in blk["cases"]:
            print(f"  ols {name} b={c['b']}: bse={c['bse']} p={c['pvalues']}")
    print("  canary:", {k: (v.get("py_files_scanned"), len(v.get("files_mentioning_ewc", [])))
                        for k, v in canary.items()})


if __name__ == "__main__":
    main()
