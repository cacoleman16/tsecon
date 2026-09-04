"""Golden fixtures for the convex / greedy estimators of `tsecon-ml`:
`l1_trend_filter` (Kim-Koh-Boyd 2009 L1 trend filtering, plus its
squared-penalty / Hodrick-Prescott form) and `boosting` (componentwise
L2 boosting, Buhlmann-Yu 2003 / Buhlmann 2006).

Objective conventions (the crate matches these exactly):

    l1_trend_filter, penalty="l1":  (1/2)||y - x||^2 + lam * ||D_k x||_1
    l1_trend_filter, penalty="l2":  (1/2)||y - x||^2 + (lam/2) * ||D_k x||^2

with D_k = np.diff(np.eye(n), k, axis=0) the k-th order difference
operator (rows (-1, 1) for k = 1; (1, -2, 1) for k = 2). The "l2" form
with k = 2 is the Hodrick-Prescott filter under HP's own lam (the 1/2 on
both terms cancels), so lam = 1600 is quarterly HP.

L1 trend filtering — reference and grade (honest):

  * THIRD-PARTY leg: every "l1" case is solved by cvxpy with the
    Clarabel interior-point solver at the tightest tolerance Clarabel
    reaches (recorded per case as `ref_solver_tol` and `ref_status`), and
    the reference's OWN optimality certificate — the relative duality gap
    of its trend, evaluated by the independent NumPy `certificate` below
    — is stored as `ref_gap_rel`. That number is how accurate the
    reference is; the Rust golden test pins the crate's trend against it
    at a tolerance chosen from the measured agreement and reports both.
  * CERTIFICATE leg (the primary grade for a convex problem): the Rust and
    Python tests re-derive the KKT certificate for the crate's OWN
    solution from scratch — recover the dual variable v from the residual
    y - x by k negative cumulative sums (y - x = D' v at the optimum),
    clip it to [-lam, lam] so it is dual feasible, and evaluate
    P(x) - G(v) with P the primal and G the dual objective; weak duality
    makes that an upper bound on P(x) - P*. The tests assert the relative
    gap <= 1e-8 on every case. Nothing in that check depends on any
    solver, including the one in this file.
  * LIMITS: `lam_max` = ||(D D')^{-1} D y||_inf is the smallest penalty at
    which the L1 trend is the least-squares polynomial of degree k - 1;
    `poly_fit` (np.polyfit) is that polynomial. The tests assert the
    crate's `lam_max` and its trend at lam >= lam_max against these.
  * "l2" cases are pinned against the dense closed form
    np.linalg.solve(I + lam D'D, y) (documented-formula golden); the
    Python suite additionally asserts the k = 2 case equals
    `tsecon.hp_filter` at 1e-10 (a cross-surface identity).

Componentwise L2 boosting — reference and grade (honest):

  R mboost `glmboost` is the roadmap's validation target and is NOT
  runnable here (no R, CRAN egress denied). The reference is therefore a
  TRANSCRIPTION of the published algorithm into dense NumPy in this file
  (`boost_dense`): single-column least-squares base learners, greedy
  selection by residual sum of squares (ties to the smallest column
  index), coefficient update nu * b_j, F_0 = 0 (no intercept — pass a
  centered y and centered columns), and the boosting operator formed
  EXPLICITLY as an n x n matrix,

      B_m = B_{m-1} + nu * H_j (I - B_{m-1}),   H_j = x_j x_j' / x_j'x_j,

  so tr(B_m) — the degrees of freedom in Buhlmann's (2006) corrected AIC

      AIC_c(m) = log(RSS_m / n) + (1 + tr(B_m)/n) / (1 - (tr(B_m) + 2)/n)

  — is exact by construction. The crate keeps B_m in a factored form and
  never forms the n x n matrix; the pin at 1e-12 on `coef_path`,
  `df_path`, `aic_path` and the exact match on `selected` prove the two
  bookkeepings agree. `fitted` is stored as B_best @ y (the operator
  applied to y), which the crate's X @ coef must reproduce — a second,
  independent consistency check on the operator. What this fixture does
  NOT prove is agreement with mboost's numbers; that stays an open
  follow-up, stated on the model card.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`). All data are seeded simulations.

Run:  .venv-wt/bin/python fixtures/generate_convex_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "convex.json"

try:  # third-party leg — optional, recorded either way
    import cvxpy as cp
    import clarabel  # noqa: F401  (version stamp only)

    HAVE_CVXPY = True
except Exception:  # pragma: no cover - environment-dependent
    cp = None
    clarabel = None
    HAVE_CVXPY = False


# ------------------------------------------------------------------ series

def make_series():
    rng = np.random.default_rng(20260903)
    out = {}

    # Piecewise-linear trend with three kinks + Gaussian noise (order 2).
    n = 150
    t = np.arange(n, dtype=float)
    knots = [35, 80, 115]
    slopes = [0.15, -0.10, 0.20, -0.05]
    trend = np.zeros(n)
    level, slope = 0.0, slopes[0]
    seg = 0
    for i in range(n):
        if seg < len(knots) and i == knots[seg]:
            seg += 1
            slope = slopes[seg]
        if i > 0:
            level += slope
        trend[i] = level
    out["pwl"] = trend + 0.5 * rng.standard_normal(n)

    # Random walk (a stochastic trend — the case HP over-smooths).
    out["rw"] = np.cumsum(rng.standard_normal(200))

    # Piecewise-constant levels + noise (order 1: fused-lasso on levels).
    levels = np.repeat([0.0, 2.0, -1.0, 1.5], [30, 30, 30, 30]).astype(float)
    out["steps"] = levels + 0.3 * rng.standard_normal(120)

    # Quadratic trend plus an AR(1) cycle (a macro-like series).
    n = 100
    t = np.arange(n, dtype=float)
    cyc = np.zeros(n)
    e = rng.standard_normal(n)
    for i in range(1, n):
        cyc[i] = 0.7 * cyc[i - 1] + e[i]
    out["ar_trend"] = 0.002 * t**2 + 0.05 * t + cyc
    return out


# ------------------------------------------------ the transcriptions

def diff_matrix(n, k):
    return np.diff(np.eye(n), k, axis=0)


def certificate(y, x, lam, k):
    """KKT / duality-gap certificate for a candidate trend x.

    Recovers v with y - x = D' v (exact at the optimum) by k negative
    cumulative sums, clips it into the dual box so it is feasible, and
    returns (pobj, dobj, gap, v_raw)."""
    D = diff_matrix(y.size, k)
    r = y - x
    v = r.copy()
    for _ in range(k):
        v = -np.cumsum(v)[:-1]
    vc = np.clip(v, -lam, lam)
    dx = D @ x
    pobj = 0.5 * float(r @ r) + lam * float(np.abs(dx).sum())
    dobj = -0.5 * float(np.sum((D.T @ vc) ** 2)) + float(vc @ (D @ y))
    return pobj, dobj, pobj - dobj, v


def lam_max_of(y, k):
    D = diff_matrix(y.size, k)
    z = np.linalg.solve(D @ D.T, D @ y)
    return float(np.max(np.abs(z)))


def poly_fit(y, k):
    t = np.arange(y.size, dtype=float)
    return np.polyval(np.polyfit(t, y, k - 1), t)


def solve_l1tf_clarabel(y, lam, k):
    """cvxpy + Clarabel at the tightest tolerance that converges."""
    n = y.size
    D = diff_matrix(n, k)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(y - x) + lam * cp.norm1(D @ x)))
    last = None
    for tol in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8):
        try:
            prob.solve(
                solver=cp.CLARABEL,
                tol_gap_abs=tol,
                tol_gap_rel=tol,
                tol_feas=tol,
                tol_ktratio=1e-8,
                max_iter=1000,
            )
        except Exception as exc:  # pragma: no cover - solver-dependent
            last = (None, f"exception: {exc}", tol)
            continue
        if prob.status == cp.OPTIMAL and x.value is not None:
            return np.asarray(x.value, dtype=float), prob.status, tol
        last = (None if x.value is None else np.asarray(x.value, dtype=float), prob.status, tol)
    return last


def l1_cases(series):
    specs = [
        # (series, order, lam as a fraction of lam_max)
        ("pwl", 2, 0.02),
        ("pwl", 2, 0.2),
        ("rw", 2, 0.05),
        ("rw", 2, 0.5),
        ("rw", 1, 0.1),
        ("steps", 1, 0.1),
        ("steps", 1, 0.5),
        ("steps", 2, 0.3),
        ("ar_trend", 2, 0.01),
        ("ar_trend", 2, 0.25),
        # Beyond lam_max the trend is the least-squares polynomial. (Exactly
        # AT lam_max the problem is degenerate — one constraint active with
        # a zero multiplier — and neither solver's gap pins the trend
        # tightly there; the crate's own lam_max is exercised by property.)
        ("pwl", 2, 1.02),
        ("rw", 2, 2.0),
        ("steps", 1, 1.5),
        ("ar_trend", 2, 1.5),
    ]
    cases = []
    for name, k, frac in specs:
        y = series[name]
        lm = lam_max_of(y, k)
        lam = frac * lm
        case = {
            "name": f"{name}-k{k}-{frac:g}lammax",
            "series": name,
            "order": k,
            "lam": lam,
            "lam_frac": frac,
            "lam_max": lm,
            "poly_fit": poly_fit(y, k).tolist(),
        }
        if HAVE_CVXPY:
            xr, status, tol = solve_l1tf_clarabel(y, lam, k)
            if xr is not None:
                pobj, dobj, gap, _ = certificate(y, xr, lam, k)
                case.update(
                    trend_ref=xr.tolist(),
                    ref_status=str(status),
                    ref_solver_tol=tol,
                    ref_objective=pobj,
                    ref_gap_rel=(gap / pobj if pobj > 0 else 0.0),
                )
            else:  # pragma: no cover
                case.update(trend_ref=None, ref_status=str(status), ref_solver_tol=tol)
        cases.append(case)
    return cases


def l2_cases(series):
    specs = [
        ("pwl", 2, 1600.0),
        ("rw", 2, 100.0),
        ("ar_trend", 2, 6.25),
        ("steps", 1, 10.0),
        ("rw", 1, 0.5),
    ]
    cases = []
    for name, k, lam in specs:
        y = series[name]
        n = y.size
        D = diff_matrix(n, k)
        x = np.linalg.solve(np.eye(n) + lam * D.T @ D, y)
        dx = D @ x
        obj = 0.5 * float(np.sum((y - x) ** 2)) + 0.5 * lam * float(dx @ dx)
        cases.append({
            "name": f"{name}-k{k}-l2-{lam:g}",
            "series": name,
            "order": k,
            "lam": lam,
            "trend_ref": x.tolist(),
            "objective": obj,
        })
    return cases


def boost_dense(X, y, nu, M, X_test=None):
    """Componentwise L2 boosting with the boosting operator formed densely."""
    n, p = X.shape
    norm2 = (X**2).sum(axis=0)
    B = np.zeros((n, n))
    I = np.eye(n)
    coef = np.zeros(p)
    F = np.zeros(n)
    coef_path, selected, rss_path, df_path, aic_path = [], [], [], [], []
    for _ in range(M):
        U = y - F
        xr = X.T @ U
        scores = np.full(p, -np.inf)
        ok = norm2 > 0
        scores[ok] = xr[ok] ** 2 / norm2[ok]
        j = int(np.argmax(scores))  # first maximum: smallest index on ties
        b = xr[j] / norm2[j]
        coef[j] += nu * b
        F = F + nu * b * X[:, j]
        H = np.outer(X[:, j], X[:, j]) / norm2[j]
        B = B + nu * H @ (I - B)
        rss = float(np.sum((y - F) ** 2))
        df = float(np.trace(B))
        denom = 1.0 - (df + 2.0) / n
        aic = np.log(rss / n) + (1.0 + df / n) / denom if denom > 0 else np.inf
        coef_path.append(coef.copy())
        selected.append(j)
        rss_path.append(rss)
        df_path.append(df)
        aic_path.append(float(aic))
    aic_arr = np.array(aic_path)
    best = int(np.argmin(aic_arr))  # first minimum on ties
    # Re-run the operator to `best` to store B_best @ y independently of coef.
    Bb = np.zeros((n, n))
    for m in range(best + 1):
        j = selected[m]
        H = np.outer(X[:, j], X[:, j]) / norm2[j]
        Bb = Bb + nu * H @ (I - Bb)
    fitted = Bb @ y
    out = {
        "coef_path": [c.tolist() for c in coef_path],
        "selected": selected,
        "rss_path": rss_path,
        "df_path": df_path,
        "aic_path": aic_path,
        "best_step": best,
        "coef": coef_path[best].tolist(),
        "fitted_operator": fitted.tolist(),
        "df_best": df_path[best],
    }
    if X_test is not None:
        out["predicted"] = (X_test @ coef_path[best]).tolist()
    return out


def boosting_designs():
    designs = {}
    rng = np.random.default_rng(7)
    # Sparse truth, independent standardized columns.
    n, p = 60, 8
    X = rng.standard_normal((n, p))
    X = (X - X.mean(0)) / X.std(0)
    beta = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    y = X @ beta + 0.5 * rng.standard_normal(n)
    y = y - y.mean()
    Xt = rng.standard_normal((5, p))
    designs["sparse"] = {"X": X, "y": y, "X_test": Xt, "true_beta": beta}

    # Correlated columns (a block of three at rho ~ 0.8), sparse truth.
    n, p = 80, 6
    Z = rng.standard_normal((n, p))
    X = Z.copy()
    X[:, 1] = 0.8 * Z[:, 0] + 0.6 * Z[:, 1]
    X[:, 2] = 0.8 * Z[:, 0] + 0.6 * Z[:, 2]
    X = (X - X.mean(0)) / X.std(0)
    beta = np.array([2.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    y = X @ beta + 0.7 * rng.standard_normal(n)
    y = y - y.mean()
    designs["correlated"] = {"X": X, "y": y, "X_test": None, "true_beta": beta}
    return designs


def boosting_cases(designs):
    specs = [
        ("sparse", 0.1, 200),
        ("sparse", 0.5, 25),
        ("sparse", 1.0, 15),
        ("correlated", 0.1, 200),
        ("correlated", 0.3, 40),
    ]
    cases = []
    for name, nu, M in specs:
        d = designs[name]
        res = boost_dense(d["X"], d["y"], nu, M, d["X_test"])
        res.update(name=f"{name}-nu{nu:g}-M{M}", design=name, learning_rate=nu, n_steps=M)
        cases.append(res)
    return cases


# --------------------------------------------------------------- main

def main():
    series = make_series()
    l1 = l1_cases(series)
    l2 = l2_cases(series)
    designs = boosting_designs()
    boost = boosting_cases(designs)

    meta = {
        "numpy": np.__version__,
        "seed_series": 20260903,
        "seed_boosting": 7,
        "objective_note": (
            "l1_trend_filter: penalty='l1' minimizes (1/2)||y-x||^2 + lam*||D_k x||_1; "
            "penalty='l2' minimizes (1/2)||y-x||^2 + (lam/2)*||D_k x||^2 (= Hodrick-"
            "Prescott with HP's own lam for k=2). D_k = np.diff(np.eye(n), k, axis=0)."
        ),
        "certificate_note": (
            "Tests re-derive the KKT certificate for the crate's own trend: v from y-x by "
            "k negative cumsums, clipped to [-lam, lam]; gap = P(x) - G(v) <= 1e-8 * P(x)."
        ),
        "boosting_note": (
            "TRANSCRIPTION grade, not a third-party run: R mboost glmboost is not runnable "
            "here. boost_dense forms the n x n boosting operator explicitly, so tr(B_m) is "
            "exact; F_0 = 0 (no intercept), ties to the smallest column index, "
            "AIC_c(m) = log(RSS/n) + (1 + df/n)/(1 - (df+2)/n) (Buhlmann 2006)."
        ),
    }
    if HAVE_CVXPY:
        import clarabel as _cl

        meta.update(
            l1_reference="cvxpy + Clarabel (third-party interior-point solver)",
            cvxpy=cp.__version__,
            clarabel=getattr(_cl, "__version__", "unknown"),
        )
    else:  # pragma: no cover
        meta.update(
            l1_reference=(
                "cvxpy/Clarabel unavailable in the generating environment — no third-"
                "party trend stored; the certificate and the closed-form limits carry "
                "the L1 grade"
            ),
        )

    fixture = {
        "_meta": meta,
        "series": {k: v.tolist() for k, v in series.items()},
        "l1_cases": l1,
        "l2_cases": l2,
        "boost_designs": {
            k: {
                "X": v["X"].tolist(),
                "y": v["y"].tolist(),
                "X_test": None if v["X_test"] is None else v["X_test"].tolist(),
                "true_beta": v["true_beta"].tolist(),
            }
            for k, v in designs.items()
        },
        "boost_cases": boost,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")
    for c in l1:
        ref = c.get("ref_gap_rel")
        extra = (
            f" ref_status={c['ref_status']} ref_tol={c['ref_solver_tol']:g} "
            f"ref_gap_rel={ref:.2e}" if ref is not None else " (no third-party reference)"
        )
        print(f"  l1 {c['name']}: lam={c['lam']:.4g} lam_max={c['lam_max']:.4g}{extra}")
    for c in boost:
        print(f"  boost {c['name']}: best_step={c['best_step']} df={c['df_best']:.4f}")


if __name__ == "__main__":
    main()
