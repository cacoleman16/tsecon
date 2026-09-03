"""Golden fixtures for the structured-penalty and post-selection slice of
tsecon-ml: `group_lasso` (group LASSO / sparse-group LASSO), `post_lasso`
(post-LASSO OLS refit) and `pds_lasso` (post-double-selection with HAC
inference).

References and their honest grades
----------------------------------

* `group_lasso` — **independent package**: `skglm` (Bertrand, Klopfenstein,
  Bannier, Gidel & Massias 2022), whose `GroupLasso` estimator minimises
  exactly

      1/(2n) ||y - X w||^2 + alpha * sum_g weights_g ||w_g||_2

  and whose `WeightedL1GroupL2` penalty (with the `QuadraticGroup` datafit
  and the `GroupBCD` solver) minimises

      1/(2n) ||y - X w||^2 + alpha * [ sum_g wg_g ||w_g||_2 + sum_j wf_j |w_j| ] ,

  so tsecon's objective

      1/(2n) ||y - X b||^2 + alpha * [ (1 - l1_ratio) sum_g w_g ||b_g||_2
                                      + l1_ratio ||b||_1 ]

  is reproduced with wg_g = (1 - l1_ratio) w_g and wf_j = l1_ratio.  skglm
  is a working-set block coordinate-descent solver run here to
  `tol = 1e-12`; it is an approximate optimiser, so each case also records
  `reference_kkt`, the subgradient-KKT residual of *skglm's own* solution
  evaluated independently in NumPy (`kkt_residual` below, the same
  conditions the Rust test evaluates on tsecon's solution).  The pin
  tolerance the tests use is bounded below by that number: two solvers can
  only agree as well as the looser of them is converged.  The crate's own
  optimality certificate (KKT residual <= 1e-8, rigorous for this convex
  problem) is the tight grade; the skglm agreement is the cross-package
  grade.  The `l1_ratio = 1` case uses scikit-learn `Lasso` instead: it is
  the reduction that ties the group solver to the crate's existing
  scikit-learn-pinned coordinate descent.  `alpha_max` per case is a
  NumPy transcription of the closed form / bisection derived in the crate
  docs (a documented-formula value, checked by the all-zero property).
  The `group-lasso` package (Moe) was also installed and evaluated; it
  is a FISTA solver its own docstring calls "not as accurate", reaching
  only ~1e-4 agreement, so it is not used as a reference.

* `post_lasso` — **independent package**: scikit-learn `Lasso` /
  `ElasticNet` (tol 1e-12) for the selection stage, then
  `LinearRegression(fit_intercept=False)` on the selected columns for the
  refit.  No standard errors are produced or pinned, deliberately.

* `pds_lasso` — **independent package** for the exact leg: statsmodels
  `OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L,
  "use_correction": True}, use_t=False)` (and `cov_type="nonrobust",
  use_t=False` for `hac_lags = 0`) on `[d, X_union]`; the selection stage
  is scikit-learn `lasso_path` on the crate's grid (alpha_max =
  max|x_j'y|/n, 100 log-spaced points down to 1e-3 alpha_max) with the
  BIC `n ln(RSS/n) + ln(n) df` transcribed here.  The "forced" case uses a
  tiny alpha so both LASSOs select every control and the union is known.
  The statistical claim (PDS covers, single selection does not) is NOT a
  fixture: R `hdm` and Stata `pdslasso` are not runnable in this
  environment, and coverage is a repeated-sampling property, so it is
  carried by the seeded Monte Carlo in
  `crates/tsecon-ml/tests/structured_properties.rs`, whose numbers are on
  the model card.

This generator NEVER imports tsecon.  Doubles are written with json's
shortest round-trip repr; the Rust golden test parses them with
serde_json `float_roundtrip` to identical bits.  All data are seeded
NumPy `default_rng` draws.

Run:  .venv-wt/bin/python fixtures/generate_structured_fixtures.py
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "structured.json"


# ------------------------------------------------------------------ helpers

def standardize(X):
    return (X - X.mean(0)) / X.std(0)


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def group_members(groups):
    labels = sorted(set(int(g) for g in groups))
    return labels, [[j for j in range(len(groups)) if groups[j] == lab] for lab in labels]


def resolve_weights(mode, members):
    if mode == "sqrt_size":
        return np.array([math.sqrt(len(m)) for m in members])
    if mode == "none":
        return np.ones(len(members))
    return np.asarray(mode, dtype=float)


def kkt_residual(X, y, b, members, w, alpha, l1_ratio):
    """Largest subgradient-KKT residual of b for the tsecon objective.

    grad = -X'(y - Xb)/n.  Inactive group: ||S(-grad_g, lam1)||_2 <= lam2 w_g.
    Active group, b_j != 0: grad_j + lam2 w_g b_j/||b_g|| + lam1 sign(b_j) = 0;
    b_j == 0 inside an active group: |grad_j| <= lam1.
    """
    n = X.shape[0]
    lam1, lam2 = alpha * l1_ratio, alpha * (1.0 - l1_ratio)
    grad = -X.T @ (y - X @ b) / n
    worst = 0.0
    for g, m in enumerate(members):
        bg = b[m]
        gg = grad[m]
        nb = np.linalg.norm(bg)
        if nb == 0.0:
            worst = max(worst, np.linalg.norm(soft(-gg, lam1)) - lam2 * w[g])
        else:
            for k, j in enumerate(m):
                if bg[k] != 0.0:
                    worst = max(worst, abs(gg[k] + lam2 * w[g] * bg[k] / nb + lam1 * np.sign(bg[k])))
                else:
                    worst = max(worst, abs(gg[k]) - lam1)
    return float(max(worst, 0.0))


def objective(X, y, b, members, w, alpha, l1_ratio):
    n = X.shape[0]
    fit = float(np.sum((y - X @ b) ** 2)) / (2 * n)
    grp = sum(w[g] * np.linalg.norm(b[m]) for g, m in enumerate(members))
    return fit + alpha * ((1 - l1_ratio) * grp + l1_ratio * np.sum(np.abs(b)))


def alpha_max(X, y, members, w, l1_ratio):
    """Smallest alpha with the all-zero solution (crate docs' derivation)."""
    n = X.shape[0]
    z = X.T @ y / n
    worst = 0.0
    for g, m in enumerate(members):
        zg = z[m]
        if np.linalg.norm(zg) == 0.0:
            a = 0.0
        elif l1_ratio >= 1.0:
            a = float(np.max(np.abs(zg)))
        elif l1_ratio <= 0.0:
            a = float(np.linalg.norm(zg) / w[g])
        else:
            def phi(a):
                return np.linalg.norm(soft(zg, a * l1_ratio)) - a * (1 - l1_ratio) * w[g]
            lo, hi = 0.0, float(np.max(np.abs(zg)) / l1_ratio)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                if phi(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            a = hi
        worst = max(worst, a)
    return float(worst)


# --------------------------------------------------------------- group lasso

def skglm_group_lasso(X, y, members, w, alpha, l1_ratio):
    import skglm
    from skglm import GeneralizedLinearEstimator, GroupLasso
    from skglm.datafits import QuadraticGroup
    from skglm.penalties import WeightedL1GroupL2
    from skglm.solvers import GroupBCD

    if l1_ratio == 0.0:
        est = GroupLasso(
            groups=[list(m) for m in members], alpha=alpha, weights=w,
            fit_intercept=False, tol=1e-12, max_iter=10_000, max_epochs=100_000,
        )
        est.fit(X, y)
        return est.coef_.copy(), f"skglm {skglm.__version__} GroupLasso (celer working-set BCD, tol=1e-12)"

    grp_indices = np.concatenate([np.asarray(m, dtype=np.int32) for m in members])
    grp_ptr = np.zeros(len(members) + 1, dtype=np.int32)
    grp_ptr[1:] = np.cumsum([len(m) for m in members])
    p = X.shape[1]
    pen = WeightedL1GroupL2(
        alpha=alpha,
        weights_groups=(1.0 - l1_ratio) * np.asarray(w, dtype=float),
        weights_features=np.full(p, l1_ratio, dtype=float),
        grp_ptr=grp_ptr, grp_indices=grp_indices,
    )
    est = GeneralizedLinearEstimator(
        datafit=QuadraticGroup(grp_ptr, grp_indices),
        penalty=pen,
        solver=GroupBCD(tol=1e-12, max_iter=10_000, max_epochs=100_000, fit_intercept=False,
                        ws_strategy="fixpoint"),
    )
    est.fit(X, y)
    return est.coef_.copy(), (
        f"skglm {skglm.__version__} GeneralizedLinearEstimator(QuadraticGroup, "
        f"WeightedL1GroupL2, GroupBCD tol=1e-12, ws_strategy=fixpoint)"
    )


def gen_group_lasso():
    from sklearn.linear_model import Lasso

    designs = {}
    # (a) contiguous blocks, one within-group zero in an active group.
    rng = np.random.default_rng(9031)
    n, p = 160, 12
    X = rng.standard_normal((n, p))
    # Mild within-group correlation so the block Gram is not a multiple of
    # the identity (the case where the naive closed form is wrong).
    for g0 in range(0, p, 3):
        X[:, g0 + 1] += 0.5 * X[:, g0]
        X[:, g0 + 2] -= 0.3 * X[:, g0 + 1]
    beta = np.zeros(p)
    beta[0:3] = [1.5, -0.9, 0.6]
    beta[6:9] = [0.8, 0.0, -0.5]
    y = X @ beta + 1.2 * rng.standard_normal(n)
    designs["blocks"] = dict(
        X=standardize(X), y=y - y.mean(), groups=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        true_beta=beta,
    )
    # (b) scattered, non-contiguous labels with unequal sizes.
    rng = np.random.default_rng(9032)
    n, p = 120, 10
    X = rng.standard_normal((n, p))
    X[:, 4] += 0.6 * X[:, 1]
    X[:, 9] += 0.4 * X[:, 7]
    beta = np.zeros(p)
    groups_b = [2, 0, 2, 1, 0, 2, 1, 3, 0, 3]
    beta[[1, 4, 8]] = [1.0, -0.7, 0.4]      # label 0
    beta[[7, 9]] = [0.0, 0.9]               # label 3 (one within-group zero)
    y = X @ beta + 1.0 * rng.standard_normal(n)
    designs["scattered"] = dict(
        X=standardize(X), y=y - y.mean(), groups=groups_b, true_beta=beta,
    )

    specs = [
        ("blocks", 0.0, 0.05, "sqrt_size"),
        ("blocks", 0.0, 0.15, "sqrt_size"),
        ("blocks", 0.0, 0.10, "none"),
        ("blocks", 0.5, 0.08, "sqrt_size"),
        ("blocks", 0.3, 0.05, [1.0, 2.0, 0.5, 1.5]),
        ("blocks", 0.9, 0.06, "sqrt_size"),
        ("scattered", 0.0, 0.10, "sqrt_size"),
        ("scattered", 0.5, 0.05, "sqrt_size"),
        ("scattered", 0.2, 0.12, "none"),
        ("blocks", 1.0, 0.10, "sqrt_size"),        # reduction: sklearn Lasso
    ]
    cases = []
    for name, l1, alpha, wmode in specs:
        d = designs[name]
        X, y, groups = d["X"], d["y"], d["groups"]
        labels, members = group_members(groups)
        w = resolve_weights(wmode, members)
        if l1 == 1.0:
            est = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=1_000_000).fit(X, y)
            coef = est.coef_.copy()
            ref = "scikit-learn Lasso (tol=1e-12) — the l1_ratio=1 reduction"
        else:
            coef, ref = skglm_group_lasso(X, y, members, w, alpha, l1)
        cases.append({
            "name": f"{name}_l1{l1:g}_a{alpha:g}_{'custom' if isinstance(wmode, list) else wmode}",
            "design": name,
            "alpha": alpha,
            "l1_ratio": l1,
            "group_weights": wmode,
            "reference": ref,
            "coef": [float(v) for v in coef],
            "active_groups": [labels[g] for g, m in enumerate(members) if np.any(coef[m] != 0)],
            "objective": objective(X, y, coef, members, w, alpha, l1),
            "reference_kkt": kkt_residual(X, y, coef, members, w, alpha, l1),
            "alpha_max": alpha_max(X, y, members, w, l1),
        })
    return {
        k: {
            "X": [[float(v) for v in row] for row in d["X"]],
            "y": [float(v) for v in d["y"]],
            "groups": d["groups"],
            "true_beta": [float(v) for v in d["true_beta"]],
        }
        for k, d in designs.items()
    }, cases


# ---------------------------------------------------------------- post lasso

def gen_post_lasso(designs):
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression

    X = np.array(designs["blocks"]["X"])
    y = np.array(designs["blocks"]["y"])
    cases = []
    for alpha, l1 in [(0.10, 1.0), (0.20, 1.0), (0.10, 0.5)]:
        if l1 == 1.0:
            est = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=1_000_000)
        else:
            est = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, tol=1e-12,
                             max_iter=1_000_000)
        est.fit(X, y)
        support = [int(j) for j in np.flatnonzero(est.coef_ != 0)]
        refit = LinearRegression(fit_intercept=False).fit(X[:, support], y)
        coef_ols = np.zeros(X.shape[1])
        coef_ols[support] = refit.coef_
        rss = float(np.sum((y - X @ coef_ols) ** 2))
        cases.append({
            "name": f"post_a{alpha:g}_l1{l1:g}",
            "design": "blocks",
            "alpha": alpha,
            "l1_ratio": l1,
            "support": support,
            "coef_lasso": [float(v) for v in est.coef_],
            "coef_ols": [float(v) for v in coef_ols],
            "rss": rss,
        })
    return cases


# ----------------------------------------------------------------------- pds

def ar1(rng, n, rho, scale):
    e = np.empty(n)
    e[0] = rng.standard_normal() * scale / math.sqrt(1 - rho ** 2)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + scale * rng.standard_normal()
    return e


def bic_pick(X, target):
    """scikit-learn lasso_path on the crate's grid + the crate's BIC rule."""
    from sklearn.linear_model import lasso_path

    n = X.shape[0]
    alphas, coefs, _ = lasso_path(X, target, eps=1e-3, alphas=100, tol=1e-12,
                                  max_iter=1_000_000)
    bic = []
    for k in range(alphas.size):
        b = coefs[:, k]
        rss = float(np.sum((target - X @ b) ** 2))
        df = int(np.sum(b != 0))
        bic.append(n * math.log(max(rss, np.finfo(float).tiny) / n) + math.log(n) * df)
    i = int(np.argmin(bic))
    return float(alphas[i]), [int(j) for j in np.flatnonzero(coefs[:, i] != 0)], alphas, bic


def gen_pds():
    import statsmodels
    import statsmodels.api as sm

    rng = np.random.default_rng(9033)
    n, p = 200, 30
    X = rng.standard_normal((n, p))
    gamma = np.zeros(p)
    gamma[[0, 1, 5, 6]] = [1.0, 0.8, 0.9, -0.7]
    beta = np.zeros(p)
    beta[[0, 1, 2, 3, 4]] = [0.5, -0.4, 0.6, 0.3, -0.5]
    tau = 1.0
    v = ar1(rng, n, 0.5, 1.0)
    e = ar1(rng, n, 0.5, 1.0)
    d = X @ gamma + v
    y = tau * d + X @ beta + e
    Xs = standardize(X)
    dc = d - d.mean()
    yc = y - y.mean()

    def hac_fit(cols, L):
        Z = np.column_stack(cols)
        if L == 0:
            r = sm.OLS(yc, Z).fit(cov_type="nonrobust", use_t=False)
        else:
            r = sm.OLS(yc, Z).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": L, "use_correction": True},
                                  use_t=False)
        ci = np.asarray(r.conf_int(alpha=0.05))[0]
        return {
            "coef": float(r.params[0]), "se": float(r.bse[0]),
            "t_stat": float(r.tvalues[0]), "p_value": float(r.pvalues[0]),
            "conf_int": [float(ci[0]), float(ci[1])],
        }

    nw_rule = int(math.floor(4 * (n / 100) ** (2 / 9)))
    cases = []
    # Forced full support: alpha tiny selects every control in both equations.
    from sklearn.linear_model import Lasso
    tiny = 1e-6
    sy = [int(j) for j in np.flatnonzero(
        Lasso(alpha=tiny, fit_intercept=False, tol=1e-12, max_iter=1_000_000).fit(Xs, yc).coef_ != 0)]
    sd = [int(j) for j in np.flatnonzero(
        Lasso(alpha=tiny, fit_intercept=False, tol=1e-12, max_iter=1_000_000).fit(Xs, dc).coef_ != 0)]
    union = sorted(set(sy) | set(sd))
    assert union == list(range(p)), union
    for L in (nw_rule, 8, 0):
        cases.append({
            "name": f"forced_full_L{L}",
            "alpha": tiny,
            "hac_lags": L,
            "support_y": sy, "support_d": sd, "union_support": union,
            **hac_fit([dc] + [Xs[:, j] for j in union], L),
        })
    # BIC-selected supports on the same data.
    ay, sy, alphas_y, bic_y = bic_pick(Xs, yc)
    ad, sd, alphas_d, bic_d = bic_pick(Xs, dc)
    union = sorted(set(sy) | set(sd))
    for L in (nw_rule, 0):
        cases.append({
            "name": f"bic_L{L}",
            "alpha": "bic",
            "hac_lags": L,
            "alpha_y": ay, "alpha_d": ad,
            "support_y": sy, "support_d": sd, "union_support": union,
            **hac_fit([dc] + [Xs[:, j] for j in union], L),
        })
    return {
        "_note": (
            "y = tau d + X beta + e, d = X gamma + v, e and v AR(1) rho=0.5, "
            "tau = 1; X standardized (ddof=0), d and y centered. Controls 0,1 "
            "load on both equations, 2-4 on y only, 5-6 on d only."
        ),
        "statsmodels": statsmodels.__version__,
        "n": n, "p": p, "tau": tau,
        "true_beta": [float(v) for v in beta], "true_gamma": [float(v) for v in gamma],
        "newey_west_rule_maxlags": nw_rule,
        "X": [[float(v) for v in row] for row in Xs],
        "d": [float(v) for v in dc],
        "y": [float(v) for v in yc],
        "cases": cases,
    }


def main():
    import sklearn
    import skglm

    designs, gl_cases = gen_group_lasso()
    post_cases = gen_post_lasso(designs)
    pds = gen_pds()
    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "skglm": skglm.__version__,
            "python": platform.python_version(),
            "seeds": {"blocks": 9031, "scattered": 9032, "pds": 9033},
            "objective_note": (
                "group_lasso minimises (1/(2n))||y - Xb||^2 + alpha*[(1 - l1_ratio)"
                "*sum_g w_g||b_g||_2 + l1_ratio*||b||_1] with w_g = sqrt(|g|) "
                "('sqrt_size'), 1 ('none'), or the given per-group array in "
                "ascending label order. skglm's GroupLasso / WeightedL1GroupL2 use "
                "the same 1/(2n) data-fit scaling; wg = (1 - l1_ratio)*w_g, "
                "wf = l1_ratio reproduces the objective exactly. post_lasso refits "
                "sklearn LinearRegression(fit_intercept=False) on the LASSO support. "
                "pds_lasso's OLS is statsmodels OLS on [d, X_union] with "
                "cov_type='HAC' (Bartlett, maxlags=L, use_correction=True) or "
                "'nonrobust' for L=0, use_t=False (normal p-values and intervals) "
                "in both modes. No intercepts anywhere: data are centered here."
            ),
            "grade_note": (
                "group_lasso: independent-package golden (skglm, an approximate "
                "working-set solver at tol=1e-12; each case records reference_kkt, "
                "the NumPy KKT residual of skglm's own solution, which bounds how "
                "tightly the two can agree) plus the crate's convex KKT certificate. "
                "post_lasso: independent package (scikit-learn). pds_lasso exact leg: "
                "independent package (statsmodels HAC OLS on the selected union); "
                "its coverage claim is Monte-Carlo grade in structured_properties.rs, "
                "since R hdm / Stata pdslasso are not runnable here."
            ),
        },
        "group_lasso": {"designs": designs, "cases": gl_cases},
        "post_lasso": {"cases": post_cases},
        "pds": pds,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")
    for c in gl_cases:
        print(f"  group_lasso {c['name']}: active_groups={c['active_groups']} "
              f"reference_kkt={c['reference_kkt']:.2e} alpha_max={c['alpha_max']:.4f}")
    for c in post_cases:
        print(f"  post_lasso {c['name']}: support={c['support']}")
    for c in pds["cases"]:
        print(f"  pds {c['name']}: coef={c['coef']:.6f} se={c['se']:.6f} "
              f"union={len(c['union_support'])}")


if __name__ == "__main__":
    main()
