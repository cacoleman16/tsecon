"""Golden fixtures for the HC standard-error ladder in `tsecon-hac`
(HC0/HC1/HC2/HC3), the leverage-corrected members in particular.

Reference implementation (this venv): statsmodels
    sm.OLS(y, X).fit(cov_type="HC0" | "HC1" | "HC2" | "HC3").bse
plus the hat-matrix diagonal
    sm.OLS(y, X).fit().get_influence().hat_matrix_diag
which pins the leverage h_t = x_t' (X'X)^{-1} x_t that HC2/HC3 divide by.

This generator NEVER imports tsecon: every reference number here is produced
by statsmodels, independently of the crate under test.

WHY THESE THREE DESIGNS
-----------------------
`small_high_leverage` uses the same DGP as the LEVERAGE design in the library's
interval-coverage audit (`docs/examples/coverage/regression_se.py`), where
`ols(se_type="hc1")` covered only 0.682 at a nominal 0.95: y = 1 + 2x + e with
T = 25, a single chi2(1) regressor (so a handful of draws carry most of the
leverage), and conditional heteroskedasticity sd(e | x) = x. Same DGP, NOT the
same draws -- the audit has its own seed, so do not expect the numbers here to
reproduce any cell of the audit's coverage table. This is the regime where the
leverage correction 1/(1-h_t) matters and the HC1 factor n/(n-k) does not.

`multi_regressor` is a calmer k = 3 design (n = 120, two correlated
regressors, multiplicative heteroskedasticity) so the goldens also pin the
off-diagonal covariance path, not just the k = 2 diagonal.

`near_singleton` is an accuracy probe, not a realism probe: a dummy d with
d[7] = 1 and d[8] = EPS, so as EPS shrinks the fit interpolates observation 7
ever more exactly and 1 - h_7 falls like ~EPS^2 (measured: 9.6e-5, 9.6e-7,
9.6e-9, 9.6e-11, 9.6e-13 at EPS = 1e-2 ... 1e-6). Because h_7 is accumulated
near 1, forming 1 - h_7 is catastrophic cancellation: its ABSOLUTE error stays
a few ulp of 1, so its RELATIVE error is ~eps_mach/(1-h_7) and the HC3 weight
(1-h_t)^-2 inherits twice that. Two independent implementations therefore stop
agreeing to 1e-10 well before either one fails. Note cond(X'X) here is only
~47 -- this is NOT an ill-conditioning effect, and no conditioning-based
diagnostic would see it. Below EPS = 1e-7 statsmodels divides by a computed
zero and returns inf (bse recorded as null); the crate refuses instead.

DEFINITIONS BEING PINNED (MacKinnon & White 1985; statsmodels `_HCCM`)
----------------------------------------------------------------------
With bread B = (X'X)^{-1}, residuals u_t and leverage h_t:
    HC0: cov = B (sum_t u_t^2               x_t x_t') B
    HC1: cov = n/(n-k) * HC0
    HC2: cov = B (sum_t u_t^2/(1-h_t)       x_t x_t') B
    HC3: cov = B (sum_t u_t^2/(1-h_t)^2     x_t x_t') B
HC2 (Horn, Horn & Duncan 1975) is unbiased for cov under homoskedasticity;
HC3 (an approximation to the jackknife, Efron 1982) is the small-sample
default recommended by Long & Ervin (2000). Neither takes the n/(n-k)
inflation that HC1 applies.

Doubles are written with json's shortest round-trip repr, which the Rust
golden test parses to identical bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_hc_robust_fixtures.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels.api as sm

OUT = Path(__file__).parent

COV_TYPES = ["nonrobust", "HC0", "HC1", "HC2", "HC3"]


def flat(a) -> list[float]:
    return [float(v) for v in np.asarray(a).ravel()]


def fit_block(y: np.ndarray, X: np.ndarray) -> dict:
    """Params, hat diagonal, and bse/tvalues/cov under every cov_type."""
    plain = sm.OLS(y, X).fit()
    block = {
        "params": flat(plain.params),
        "hat_diag": flat(plain.get_influence().hat_matrix_diag),
        "bse": {},
        "tvalues": {},
        "cov": {},
    }
    for cov_type in COV_TYPES:
        res = sm.OLS(y, X).fit(cov_type=cov_type) if cov_type != "nonrobust" else plain
        key = cov_type.lower()
        block["bse"][key] = flat(res.bse)
        block["tvalues"][key] = flat(res.tvalues)
        # Row-major k x k, matching the crate's OlsInference::cov storage.
        block["cov"][key] = flat(np.asarray(res.cov_params()))
    return block


def gen_small_high_leverage() -> dict:
    """T = 25, x ~ chi2(1), sd(e | x) = x — the coverage audit's LEVERAGE DGP.

    Slope 2.0 to match `y = 1 + 2x + e` in
    docs/examples/coverage/regression_se.py. In exact arithmetic this moves
    `params` and `tvalues` only: x is in the design, so adding c*x to y shifts
    b_x by exactly c and leaves the residuals — hence every bse and cov —
    alone. In floating point the residuals are recomputed from a different y,
    so bse/cov did shift by <= 1.6e-15 relative when the slope was changed
    from 0.5; that is roundoff, not a different answer.
    """
    rng = np.random.default_rng(20260805)
    n = 25
    x = rng.chisquare(df=1, size=n)
    e = x * rng.standard_normal(n)
    y = 1.0 + 2.0 * x + e
    X = sm.add_constant(x)
    block = fit_block(y, X)
    block.update(n=n, y=flat(y), x1=flat(x))
    return block


def gen_multi_regressor() -> dict:
    """n = 120, k = 3, correlated regressors, multiplicative heteroskedasticity."""
    rng = np.random.default_rng(11235)
    n = 120
    x1 = rng.standard_normal(n)
    x2 = 0.6 * x1 + rng.standard_normal(n)
    e = np.exp(0.5 * x1) * rng.standard_normal(n)
    y = -0.4 + 1.2 * x1 - 0.7 * x2 + e
    X = sm.add_constant(np.column_stack([x1, x2]))
    block = fit_block(y, X)
    block.update(n=n, y=flat(y), x1=flat(x1), x2=flat(x2))
    return block


# EPS ladder for `near_singleton`; 1 - h_7 lands near 0.96 * EPS^2, so this
# walks the leverage complement from 1e-4 down through the crate's 1e-12
# LEVERAGE_FLOOR and out the other side.
NEAR_SINGLETON_EPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8]


def gen_near_singleton() -> dict:
    """n = 30, k = 3, a dummy that ALMOST isolates observation 7.

    See the module docstring: this exists to measure how far HC2/HC3 parity
    survives once `1 - h_t` is dominated by cancellation, not to be a
    plausible regression.
    """
    rng = np.random.default_rng(4041)
    n = 30
    x = rng.standard_normal(n)
    y = 1.0 + 2.0 * x + rng.standard_normal(n)

    cases = []
    for eps in NEAR_SINGLETON_EPS:
        d = np.zeros(n)
        d[7] = 1.0
        d[8] = eps
        X = np.column_stack([np.ones(n), x, d])
        plain = sm.OLS(y, X).fit()
        h = plain.get_influence().hat_matrix_diag
        case = {
            "eps": float(eps),
            "dummy": flat(d),
            "hat_diag": flat(h),
            # Recorded separately because a reader should not have to
            # subtract 17 significant digits by hand to see the point.
            "one_minus_h7": float(1.0 - h[7]),
            "cond_xtx": float(np.linalg.cond(X.T @ X)),
            "params": flat(plain.params),
            "bse": {},
        }
        for cov_type in ["HC0", "HC1", "HC2", "HC3"]:
            with np.errstate(divide="ignore", invalid="ignore"):
                bse = np.asarray(sm.OLS(y, X).fit(cov_type=cov_type).bse)
            # null, not inf: JSON has no infinity, and "statsmodels returned
            # a non-number here" is exactly what the Rust test asserts on.
            case["bse"][cov_type.lower()] = (
                flat(bse) if np.all(np.isfinite(bse)) else None
            )
        cases.append(case)
    return {"n": n, "y": flat(y), "x1": flat(x), "cases": cases}


def main() -> None:
    small = gen_small_high_leverage()
    multi = gen_multi_regressor()
    near = gen_near_singleton()
    out = {
        "_meta": {
            "statsmodels": sm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "note": "HC0/HC1/HC2/HC3 (and nonrobust) OLS standard errors, "
                    "t-values and parameter covariances from statsmodels "
                    "`fit(cov_type=...)`, plus the hat-matrix diagonal that "
                    "HC2/HC3 divide by. `near_singleton` walks 1 - h_7 down "
                    "to machine noise to measure where HC2/HC3 parity stops "
                    "holding. Formulas in the generator docstring.",
        },
        "small_high_leverage": small,
        "multi_regressor": multi,
        "near_singleton": near,
    }
    (OUT / "hc_robust.json").write_text(json.dumps(out, separators=(",", ":")))
    print(
        "wrote hc_robust.json  "
        f"(max leverage: small {max(small['hat_diag']):.4f}, "
        f"multi {max(multi['hat_diag']):.4f})"
    )
    for name, blk in [("small_high_leverage", small), ("multi_regressor", multi)]:
        ratios = np.asarray(blk["bse"]["hc3"]) / np.asarray(blk["bse"]["hc1"])
        print(f"  {name}: hc3/hc1 bse ratio = {np.round(ratios, 4).tolist()}")
    print(f"  near_singleton: cond(X'X) = {near['cases'][0]['cond_xtx']:.1f}")
    for case in near["cases"]:
        hc3 = case["bse"]["hc3"]
        shown = "statsmodels non-finite" if hc3 is None else f"hc3 bse[2] = {hc3[2]:.4g}"
        print(f"    eps={case['eps']:<8g} 1-h7={case['one_minus_h7']:.4e}  {shown}")


if __name__ == "__main__":
    main()
