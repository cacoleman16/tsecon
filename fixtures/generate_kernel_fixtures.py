"""Golden fixtures for the kernel-methods slice of tsecon-ml:
`kernel_ridge` (exact kernel ridge regression) and `kernel_regression`
(Nadaraya-Watson and local-linear nonparametric regression).

References (independent packages, both installed in the project venv):

  * `kernel_ridge` — scikit-learn `KernelRidge(alpha, kernel, gamma,
    degree, coef0)`: the dual solve `(K + alpha I) a = y` (sklearn's
    `_solve_cholesky_kernel`), fitted values `K a`, test predictions
    `K(X_test, X) a`, and `dual_coef_`. Kernels in sklearn's exact
    parameterization (`sklearn.metrics.pairwise`):
        rbf        K = exp(-gamma ||x - y||^2)
        laplacian  K = exp(-gamma ||x - y||_1)
        polynomial K = (gamma <x, y> + coef0)^degree
        linear     K = <x, y>
    with sklearn's default `gamma = None -> 1 / n_features` for the three
    gamma kernels. The random-Fourier-feature approximation is NOT pinned
    here (it is a seeded Monte-Carlo approximation; the crate checks it by
    property — determinism in the seed and convergence to the exact fit).

  * `kernel_regression` — statsmodels `KernelReg(endog, exog,
    var_type="c" * k, reg_type="lc" | "ll", bw=<fixed vector>)`:
    `fit()[0]` at the training rows and at `x_test`, with the product
    Gaussian kernel `prod_j phi((X_ij - x_j) / h_j) / prod_j h_j`
    (`statsmodels.nonparametric.kernels.gaussian` through `gpke`). Local
    constant is `sum_i K_i y_i / sum_i K_i`; local linear is the weighted
    least squares of y on `[1, X - x]` with weights `K_i`, solved through
    `np.linalg.pinv` (`_est_loc_linear`). The leave-one-out least-squares
    criterion is `KernelReg.cv_loo(bw, func)`:
        CV(h) = n^{-1} sum_i (y_i - g_{-i}(x_i))^2 .
    statsmodels' own `bw="cv_ls"` selection (Nelder-Mead `fmin` from the
    Scott reference start) is recorded for a PROPERTY comparison only —
    tsecon selects by a log grid plus golden-section refinement and does
    not chase the fmin path.

Documented-formula transcriptions (this file, NumPy), graded honestly as
transcriptions and cross-checked against statsmodels where the two
overlap (`l = 0` reproduces `cv_loo`; the same local fits reproduce
`fit()`; both asserted at 1e-12 before anything is written):

  * leave-block-out CV (Chu & Marron 1991 "modified cross-validation";
    Hart & Vieu 1990): when predicting at observation i drop the
    2l + 1 observations j with |i - j| <= l,
        CV_l(h) = n^{-1} sum_i (y_i - g_{-(i-l..i+l)}(x_i))^2 ;
  * the effective degrees of freedom `tr(S)` of the linear smoother:
    Nadaraya-Watson S_ii = K_ii / sum_j K_ij; local linear
    S_ii = [pinv(M_i)]_{00} K_ii with M_i = Z_i' W_i Z_i (Hastie &
    Tibshirani 1990, sec. 3.5).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`). Simulated, seeded data only.

Run:  .venv-wt/bin/python fixtures/generate_kernel_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "kernel.json"


# ----------------------------------------------------------- kernel ridge

def krr_cases():
    from sklearn.kernel_ridge import KernelRidge

    rng = np.random.default_rng(20260903)
    n, p, n_test = 60, 3, 15
    X = rng.standard_normal((n, p))
    X_test = 0.8 * rng.standard_normal((n_test, p))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] + 0.3 * rng.standard_normal(n)

    specs = [
        ("rbf_default", dict(kernel="rbf", alpha=1.0)),
        ("rbf_gamma", dict(kernel="rbf", alpha=0.1, gamma=0.5)),
        ("laplacian_default", dict(kernel="laplacian", alpha=1.0)),
        ("laplacian_gamma", dict(kernel="laplacian", alpha=0.3, gamma=0.2)),
        ("polynomial_default", dict(kernel="polynomial", alpha=1.0)),
        ("polynomial_custom", dict(kernel="polynomial", alpha=2.0, gamma=0.1,
                                   degree=2, coef0=0.5)),
        ("linear_default", dict(kernel="linear", alpha=1.0)),
        ("linear_small_alpha", dict(kernel="linear", alpha=0.05)),
    ]
    cases = []
    for name, kw in specs:
        m = KernelRidge(**kw).fit(X, y)
        params = dict(kernel=kw["kernel"], alpha=kw["alpha"],
                      gamma=kw.get("gamma"), degree=kw.get("degree", 3),
                      coef0=kw.get("coef0", 1.0))
        resolved_gamma = (None if kw["kernel"] == "linear"
                          else (kw.get("gamma") if kw.get("gamma") is not None
                                else 1.0 / p))
        cases.append(dict(
            name=name,
            params=params,
            gamma_resolved=resolved_gamma,
            dual_coef=m.dual_coef_.tolist(),
            fitted=m.predict(X).tolist(),
            predicted=m.predict(X_test).tolist(),
        ))
    return dict(X=X.tolist(), X_test=X_test.tolist(), y=y.tolist(), cases=cases)


# ------------------------------------------------------ kernel regression

def gauss_weights(x, x0, bw):
    """statsmodels' product Gaussian kernel, per training row."""
    k = x.shape[1]
    u = (x - x0) / bw
    return np.exp(-0.5 * (u ** 2).sum(axis=1)) / (math.sqrt(2.0 * math.pi) ** k * bw.prod())


def local_constant(x, y, x0, bw):
    w = gauss_weights(x, x0, bw)
    return (w * y).sum() / w.sum(), 1.0 / w.sum()


def local_linear(x, y, x0, bw):
    w = gauss_weights(x, x0, bw)
    z = np.column_stack([np.ones(x.shape[0]), x - x0])
    m = z.T @ (w[:, None] * z)
    v = z.T @ (w * y)
    pinv = np.linalg.pinv(m)
    return (pinv @ v)[0], pinv[0, 0]


EST = dict(lc=local_constant, ll=local_linear)


def transcribed_fit(x, y, x_pred, bw, reg_type):
    est = EST[reg_type]
    return np.array([est(x, y, x_pred[i], bw)[0] for i in range(x_pred.shape[0])])


def transcribed_block_cv(x, y, bw, l, reg_type):
    """Leave-(2l+1)-out least-squares CV; l = 0 is statsmodels' cv_loo."""
    est = EST[reg_type]
    n = x.shape[0]
    idx = np.arange(n)
    total = 0.0
    for i in range(n):
        keep = np.abs(idx - i) > l
        g = est(x[keep], y[keep], x[i], bw)[0]
        total += (y[i] - g) ** 2
    return total / n


def transcribed_effective_df(x, y, bw, reg_type):
    est = EST[reg_type]
    k = x.shape[1]
    k0 = 1.0 / (math.sqrt(2.0 * math.pi) ** k * bw.prod())  # K_ii
    return sum(est(x, y, x[i], bw)[1] * k0 for i in range(x.shape[0]))


def kernel_reg_series():
    rng1 = np.random.default_rng(7)
    n1 = 100
    x1 = rng1.uniform(-2.5, 2.5, n1)
    y1 = np.sin(1.5 * x1) + 0.3 * rng1.standard_normal(n1)
    x1_test = np.linspace(-2.0, 2.0, 11)

    rng2 = np.random.default_rng(11)
    n2 = 90
    x2 = rng2.standard_normal((n2, 2))
    y2 = np.sin(x2[:, 0]) + 0.5 * x2[:, 1] ** 2 + 0.3 * rng2.standard_normal(n2)
    x2_test = np.column_stack([np.linspace(-1.5, 1.5, 9), np.linspace(1.0, -1.0, 9)])

    return {
        "k1": dict(x=x1.reshape(-1, 1), y=y1, x_test=x1_test.reshape(-1, 1)),
        "k2": dict(x=x2, y=y2, x_test=x2_test),
    }


def kernel_reg_cases(series):
    from statsmodels.nonparametric.kernel_regression import KernelReg

    specs = [
        ("k1", "lc", [0.15]), ("k1", "lc", [0.3]), ("k1", "lc", [0.6]),
        ("k1", "ll", [0.15]), ("k1", "ll", [0.3]), ("k1", "ll", [0.6]),
        ("k2", "lc", [0.3, 0.4]), ("k2", "lc", [0.5, 0.5]), ("k2", "lc", [0.8, 0.35]),
        ("k2", "ll", [0.3, 0.4]), ("k2", "ll", [0.5, 0.5]), ("k2", "ll", [0.8, 0.35]),
    ]
    checks = {"fit_vs_transcription": 0.0, "cv_loo_vs_transcription_l0": 0.0}
    cases = []
    for sid, reg_type, bw in specs:
        s = series[sid]
        x, y, xt = s["x"], s["y"], s["x_test"]
        k = x.shape[1]
        bwv = np.asarray(bw, dtype=float)
        kr = KernelReg(y, x, var_type="c" * k, reg_type=reg_type, bw=bwv, rng=0)
        fitted = kr.fit()[0]
        predicted = kr.fit(xt)[0]
        func = kr.est[reg_type]
        cv_loo = float(np.squeeze(kr.cv_loo(bwv, func)))
        # Cross-check the transcription against statsmodels where they overlap.
        d_fit = float(np.max(np.abs(transcribed_fit(x, y, x, bwv, reg_type) - fitted)))
        d_cv = abs(transcribed_block_cv(x, y, bwv, 0, reg_type) - cv_loo)
        assert d_fit < 1e-12 and d_cv < 1e-12, (sid, reg_type, bw, d_fit, d_cv)
        checks["fit_vs_transcription"] = max(checks["fit_vs_transcription"], d_fit)
        checks["cv_loo_vs_transcription_l0"] = max(checks["cv_loo_vs_transcription_l0"], d_cv)
        cases.append(dict(
            series=sid,
            reg_type=reg_type,
            bw=bwv.tolist(),
            fitted=fitted.tolist(),
            predicted=predicted.tolist(),
            cv_loo=cv_loo,
            block_cv={str(l): float(transcribed_block_cv(x, y, bwv, l, reg_type)) for l in (2, 5)},
            effective_df=float(transcribed_effective_df(x, y, bwv, reg_type)),
        ))

    # statsmodels' own cv_ls optimum (fmin from the Scott start), for the
    # property test "our selected criterion is no worse than fmin's".
    selections = []
    for sid in ("k1", "k2"):
        for reg_type in ("lc", "ll"):
            s = series[sid]
            x, y = s["x"], s["y"]
            k = x.shape[1]
            kr = KernelReg(y, x, var_type="c" * k, reg_type=reg_type, bw="cv_ls", rng=0)
            bw_sel = np.asarray(kr.bw, dtype=float)
            func = kr.est[reg_type]
            selections.append(dict(
                series=sid, reg_type=reg_type,
                bw_cv_ls=bw_sel.tolist(),
                cv_loo_at_bw_cv_ls=float(np.squeeze(kr.cv_loo(bw_sel, func))),
                scott_reference=(1.06 * np.std(x, axis=0) * x.shape[0] ** (-1.0 / (4 + k))).tolist(),
            ))
    return cases, selections, checks


# ------------------------------------------------------------------- main

def main():
    import sklearn
    import statsmodels

    krr = krr_cases()
    series = kernel_reg_series()
    cases, selections, checks = kernel_reg_cases(series)

    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "seeds": {"kernel_ridge": 20260903, "kernel_regression_k1": 7,
                      "kernel_regression_k2": 11},
            "kernel_ridge": (
                "scikit-learn KernelRidge — independent package. Dual solve "
                "(K + alpha I) a = y; kernels rbf exp(-g||x-y||^2), laplacian "
                "exp(-g||x-y||_1), polynomial (g<x,y> + coef0)^degree, linear "
                "<x,y>; gamma=None -> 1/n_features. dual_coef_, predict(X), "
                "predict(X_test) pinned. Random Fourier features are NOT pinned "
                "(property-tested in the crate)."
            ),
            "kernel_regression": (
                "statsmodels KernelReg — independent package. reg_type lc "
                "(Nadaraya-Watson) and ll (local linear via pinv), product "
                "Gaussian kernel, var_type='c'*k, user-specified bw. fit() at "
                "the training rows and at x_test, and cv_loo(bw, func) = "
                "n^-1 sum_i (y_i - g_{-i}(x_i))^2, pinned. bw='cv_ls' (fmin) "
                "results are recorded for a property comparison only."
            ),
            "transcriptions": (
                "block_cv (leave-(2l+1)-out CV, Chu-Marron 1991 / Hart-Vieu "
                "1990) and effective_df (tr S) are documented-formula NumPy "
                "transcriptions in this file — no package computes them. The "
                "transcription reproduces statsmodels where they overlap: "
                "l = 0 equals cv_loo and the local fits equal fit(), both "
                "asserted at 1e-12 at generation (max diffs recorded in "
                "transcription_checks)."
            ),
            "transcription_checks": checks,
            "grade": (
                "kernel_ridge: independent package (sklearn). kernel_regression "
                "fitted/predicted/cv_loo: independent package (statsmodels). "
                "block_cv/effective_df: documented-formula transcription."
            ),
        },
        "kernel_ridge": krr,
        "kernel_regression": {
            "series": {
                sid: dict(x=s["x"].tolist(), y=s["y"].tolist(), x_test=s["x_test"].tolist())
                for sid, s in series.items()
            },
            "cases": cases,
            "cv_ls_selections": selections,
        },
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")
    for c in krr["cases"]:
        print(f"  krr {c['name']}: gamma={c['gamma_resolved']} dual_coef[0]={c['dual_coef'][0]:.6g}")
    for c in cases:
        print(f"  kreg {c['series']} {c['reg_type']} bw={c['bw']}: cv_loo={c['cv_loo']:.6g} "
              f"edf={c['effective_df']:.4f} block_cv={c['block_cv']}")
    for s in selections:
        print(f"  cv_ls {s['series']} {s['reg_type']}: bw={s['bw_cv_ls']} "
              f"cv={s['cv_loo_at_bw_cv_ls']:.6g} scott={s['scott_reference']}")


if __name__ == "__main__":
    main()
