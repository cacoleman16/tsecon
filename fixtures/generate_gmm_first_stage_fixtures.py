"""Golden fixtures for the tsecon-gmm first-stage diagnostics and the HAC
weighting path (reference: linearmodels 7.0).

Two things are pinned here, both findings of the interval-coverage audit:

1. **First-stage F.** `linearmodels` reports first-stage diagnostics through
   `IV2SLS(...).fit(cov_type="robust").first_stage.diagnostics`. Under a
   robust covariance its `f.stat` column is a **Wald chi2(q) statistic, not
   divided by q**:

       f.stat = b_excl' V_HC0[excl, excl]^{-1} b_excl,    dist chi2(q)

   where `b` is OLS of the endogenous regressor on `[exog, instruments]`,
   `V_HC0` is the White covariance with `debiased=False` (linearmodels'
   default), and `excl` selects the excluded instruments. See
   `linearmodels/iv/results.py::FirstStageResults.diagnostics`.

   tsecon reports the *Stata* convention instead — an F, HC1, referred to
   F(q, n - L) — because that is what the sibling first-stage diagnostics in
   this library already report (`proxy_svar`, `lp_iv`) and what applied users
   compare to the Staiger-Stock rule of thumb. The two are related by an
   exact, documented identity:

       F_HC1 = f.stat / q * (n - L) / n

   so `f_hc1` below is a closed-form transformation of the linearmodels
   number, not an independent computation. The generator additionally
   recomputes the HC0 Wald from scratch in numpy and asserts it reproduces
   linearmodels to 1e-10 relative, so the transcription itself is checked.

   `f_pval` is `scipy.stats.f.sf(f_hc1, q, n - L)`.

   `lm_chi2_pval` is linearmodels' own `f.pval` column, and the crate's golden
   test reads it: it checks `chi2.sf(f.stat, q)` reproduces it, which is what
   makes the `chi2(q)` premise above *auditable* rather than merely asserted.
   The whole conversion identity rests on that premise, so if linearmodels ever
   changed its reference distribution the failure surfaces there, with a
   legible cause, instead of silently shifting every `f_hc1`. Note that the
   strongly instrumented rows underflow to `f.pval = 0.0` exactly, so the test
   can only check those against a vanishing bound.

2. **HAC-weighted two-step GMM.** `IVGMM(..., weight_type="kernel",
   kernel="bartlett", bandwidth=m).fit(cov_type="kernel", kernel="bartlett",
   bandwidth=m, debiased=False)`. The audit found the crate's HAC path was a
   silent no-op at the default bandwidth, so it had never been pinned against
   a reference at a *nonzero* bandwidth. `m` is the Newey-West (1994)
   rule-of-thumb lag truncation `floor(4 * (n/100)^(2/9))`, which is the
   automatic rule the crate now offers.

Design A carries a serially correlated error (AR(1), rho = 0.6) so the HAC
weighting is materially different from the White weighting; design B has two
endogenous regressors so the per-regressor first-stage loop is exercised.

Run with the project venv:
    .venv/bin/python fixtures/generate_gmm_first_stage_fixtures.py
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import linearmodels
from linearmodels.iv import IV2SLS, IVGMM
from scipy import stats

OUT = Path(__file__).parent


def full(a):
    return [float(x) for x in np.asarray(a).ravel()]


def nw_maxlags(n: int) -> int:
    """Newey-West (1994) rule-of-thumb lag truncation."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hc0_wald_check(endog, zcols, excluded):
    """Independent numpy recomputation of the linearmodels robust `f.stat`.

    OLS of `endog` on all instrument columns `zcols`, White (HC0) covariance,
    Wald statistic on the `excluded` coefficient block.
    """
    z = np.column_stack(zcols)
    n, ell = z.shape
    b = np.linalg.lstsq(z, endog, rcond=None)[0]
    e = endog - z @ b
    ztz_inv = np.linalg.inv(z.T @ z)
    meat = z.T @ (z * (e**2)[:, None])
    v_hc0 = ztz_inv @ meat @ ztz_inv
    bb = b[excluded]
    return float(bb @ np.linalg.inv(v_hc0[np.ix_(excluded, excluded)]) @ bb)


def first_stage_block(res, endog_names, n, ell, q, zcols, excluded_idx):
    """Transcribe linearmodels' first-stage diagnostics and convert to the
    Stata-convention HC1 F that the crate reports."""
    diag = res.first_stage.diagnostics
    out = []
    for name in endog_names:
        wald = float(diag.loc[name, "f.stat"])
        # Cross-check the transcription against a from-scratch numpy HC0 Wald.
        check = hc0_wald_check(zcols["endog"][name], zcols["z"], excluded_idx)
        rel = abs(check - wald) / abs(wald)
        assert rel < 1e-10, f"{name}: numpy HC0 Wald {check} vs linearmodels {wald} (rel {rel:.3e})"
        f_hc1 = wald / q * (n - ell) / n
        out.append(
            {
                "name": name,
                "lm_wald_chi2_hc0": wald,
                "lm_chi2_pval": float(diag.loc[name, "f.pval"]),
                "f_hc1": f_hc1,
                "f_pval": float(stats.f.sf(f_hc1, q, n - ell)),
                "dof_num": q,
                "dof_den": n - ell,
            }
        )
    return out


def design_a():
    """One endogenous regressor, two excluded instruments, AR(1) error."""
    rng = np.random.default_rng(4242)
    n = 300
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    w = rng.standard_normal(n)
    u = np.zeros(n)
    e = rng.standard_normal(n)
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + e[t]
    x = 0.7 * z1 + 0.4 * z2 + 0.6 * u + 0.3 * rng.standard_normal(n)
    y = 1.0 + 0.5 * x - 0.4 * w + u
    df = pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2, "const": 1.0})

    # X = [const, w, x] (k = 3), Z = [const, w, z1, z2] (L = 4), q = 2.
    ell, q = 4, 2
    res2sls = IV2SLS(df["y"], df[["const", "w"]], df["x"], df[["z1", "z2"]]).fit(
        cov_type="robust"
    )
    zcols = {"z": [np.ones(n), w, z1, z2], "endog": {"x": x}}
    fs = first_stage_block(res2sls, ["x"], n, ell, q, zcols, [2, 3])

    m = nw_maxlags(n)
    hac = IVGMM(
        df["y"],
        df[["const", "w"]],
        df["x"],
        df[["z1", "z2"]],
        weight_type="kernel",
        kernel="bartlett",
        bandwidth=m,
    ).fit(cov_type="kernel", kernel="bartlett", bandwidth=m, debiased=False)

    return {
        "n": n,
        "note": "X = [const, w, x] with x endogenous; Z = [const, w, z1, z2]. "
        "AR(1) error, rho = 0.6.",
        "y": full(y),
        "x": full(x),
        "w": full(w),
        "z1": full(z1),
        "z2": full(z2),
        "endog_index": [2],
        "first_stage": fs,
        "hac_two_step": {
            "bandwidth": m,
            "param_order": ["const", "w", "x"],
            "params": {k: float(v) for k, v in hac.params.items()},
            "bse": {k: float(v) for k, v in hac.std_errors.items()},
            "j_stat": float(hac.j_stat.stat),
            "j_pval": float(hac.j_stat.pval),
        },
    }


def design_b():
    """Two endogenous regressors, three excluded instruments."""
    rng = np.random.default_rng(20260805)
    n = 400
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    z3 = rng.standard_normal(n)
    w = rng.standard_normal(n)
    u = rng.standard_normal(n)
    # x1 is strongly instrumented, x2 only weakly (small z3 loading) — the
    # per-regressor F must be able to say so.
    x1 = 0.9 * z1 + 0.5 * z2 + 0.7 * u + 0.4 * rng.standard_normal(n)
    x2 = 0.15 * z3 + 0.10 * z1 + 0.5 * u + 1.0 * rng.standard_normal(n)
    y = 0.5 + 0.4 * x1 - 0.3 * x2 + 0.2 * w + u
    df = pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "w": w, "z1": z1, "z2": z2, "z3": z3, "const": 1.0}
    )

    # X = [const, w, x1, x2] (k = 4), Z = [const, w, z1, z2, z3] (L = 5), q = 3.
    ell, q = 5, 3
    res2sls = IV2SLS(
        df["y"], df[["const", "w"]], df[["x1", "x2"]], df[["z1", "z2", "z3"]]
    ).fit(cov_type="robust")
    zcols = {"z": [np.ones(n), w, z1, z2, z3], "endog": {"x1": x1, "x2": x2}}
    fs = first_stage_block(res2sls, ["x1", "x2"], n, ell, q, zcols, [2, 3, 4])

    return {
        "n": n,
        "note": "X = [const, w, x1, x2] with x1, x2 endogenous; "
        "Z = [const, w, z1, z2, z3]. x2 is deliberately weakly instrumented.",
        "y": full(y),
        "x1": full(x1),
        "x2": full(x2),
        "w": full(w),
        "z1": full(z1),
        "z2": full(z2),
        "z3": full(z3),
        "endog_index": [2, 3],
        "first_stage": fs,
    }


def main():
    out = {
        "_meta": {
            "linearmodels": linearmodels.__version__,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": (
                "first_stage: linearmodels IV2SLS(...).fit(cov_type='robust')"
                ".first_stage.diagnostics. 'lm_wald_chi2_hc0' is its f.stat "
                "column (a chi2(q) Wald on the excluded instruments, HC0, "
                "debiased=False). 'f_hc1' = lm_wald_chi2_hc0 / q * (n - L) / n "
                "is the exact algebraic conversion to the Stata-convention "
                "HC1 first-stage F that tsecon reports; 'f_pval' is "
                "scipy.stats.f.sf(f_hc1, q, n - L). hac_two_step: IVGMM with "
                "weight_type='kernel', kernel='bartlett', bandwidth = "
                "floor(4*(n/100)^(2/9)), cov_type='kernel', debiased=False."
            ),
        },
        "design_a": design_a(),
        "design_b": design_b(),
    }
    (OUT / "gmm_first_stage.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote gmm_first_stage.json")
    for d in ("design_a", "design_b"):
        for fs in out[d]["first_stage"]:
            print(d, fs["name"], "wald", fs["lm_wald_chi2_hc0"], "F_hc1", fs["f_hc1"],
                  "pval", fs["f_pval"])


if __name__ == "__main__":
    main()
