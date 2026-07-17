"""Golden fixtures for the Extension crates: GMM/IV-GMM (linearmodels),
term structure / Nelson-Siegel (OLS at fixed lambda), and predictive
regressions / IVX (self-authored, documented formulas).

Run with the project venv: .venv/bin/python fixtures/generate_ext_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


# ------------------------------------------------------------------ GMM
def gen_gmm():
    import linearmodels
    from linearmodels.iv import IVGMM

    rng = np.random.default_rng(61)
    n = 500
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.standard_normal(n)
    # Endogenous x correlated with u through a shared shock.
    x = 0.8 * z1 + 0.5 * z2 + 0.7 * u + 0.3 * rng.standard_normal(n)
    w = rng.standard_normal(n)  # exogenous control
    y = 1.0 + 0.5 * x - 0.4 * w + u

    import pandas as pd
    df = pd.DataFrame({"y": y, "x": x, "w": w, "z1": z1, "z2": z2, "const": 1.0})
    # 2-step efficient GMM with a robust weighting matrix.
    res = IVGMM(df["y"], df[["const", "w"]], df["x"], df[["z1", "z2"]],
                weight_type="robust").fit(cov_type="robust")

    out = {
        "_meta": {"linearmodels": linearmodels.__version__, "numpy": np.__version__,
                  "python": platform.python_version(),
                  "note": "IVGMM(y ~ [const,w] + x endog, instruments [z1,z2]), 2-step "
                          "robust weighting, robust cov. Param order: const, w, x."},
        "y": full(y), "x": full(x), "w": full(w), "z1": full(z1), "z2": full(z2),
        "ivgmm": {"param_order": ["const", "w", "x"],
                  "params": {k: float(v) for k, v in res.params.items()},
                  "bse": {k: float(v) for k, v in res.std_errors.items()},
                  "j_stat": float(res.j_stat.stat), "j_pval": float(res.j_stat.pval)},
    }
    (OUT / "gmm.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote gmm.json")


# ---------------------------------------------------- term structure / NS
def gen_termstructure():
    import statsmodels.api as sm

    rng = np.random.default_rng(67)
    maturities = np.array([3, 6, 12, 24, 36, 60, 84, 120], dtype=float)  # months
    lam = 0.0609  # Diebold-Li (2006) fixed lambda (monthly)
    # Nelson-Siegel loadings for the three factors.
    t = maturities
    load1 = np.ones_like(t)
    load2 = (1 - np.exp(-lam * t)) / (lam * t)
    load3 = load2 - np.exp(-lam * t)
    B = np.column_stack([load1, load2, load3])

    # Simulate a panel of yield curves from time-varying factors.
    n = 240
    factors = np.zeros((n, 3))
    factors[0] = [5.0, -1.0, 0.5]
    for i in range(1, n):
        factors[i] = [0.99, 0.9, 0.8] * factors[i - 1] + rng.standard_normal(3) * [0.2, 0.3, 0.4]
        factors[i, 0] += 0.05
    yields = factors @ B.T + rng.standard_normal((n, len(t))) * 0.03

    # Cross-sectional OLS fit of one representative date (the golden): regress
    # that date's yields on the NS loadings -> the three factors.
    date = 100
    ols = sm.OLS(yields[date], B).fit()

    out = {
        "_meta": {"numpy": np.__version__,
                  "note": "Nelson-Siegel loadings at lambda=0.0609 (Diebold-Li 2006, "
                          "monthly): [1, (1-e^{-lt})/(lt), (1-e^{-lt})/(lt)-e^{-lt}]. "
                          "Cross-sectional OLS of a yield curve on the loadings gives "
                          "the [level, slope, curvature] factors."},
        "maturities": full(maturities), "lambda": lam,
        "ns_loadings": [full(B[:, j]) for j in range(3)],
        "yields_date100": full(yields[date]),
        "ns_fit_factors": full(ols.params),
        "ns_fit_rsquared": float(ols.rsquared),
        "yields_panel": [full(yields[i]) for i in range(n)],
    }
    (OUT / "termstructure.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote termstructure.json")


# ---------------------------------------- predictive regressions / IVX
def gen_predreg():
    # Self-authored, documented. Predictive regression r_{t+1}=a+b x_t+u_{t+1}
    # with a persistent regressor x_t=rho x_{t-1}+e_t and corr(u,e)!=0 (the
    # Stambaugh 1999 bias). Formulas the crate must reproduce:
    #   OLS beta: standard.
    #   Stambaugh bias-corrected: b_c = b_ols - (sigma_ue/sigma_ee)*(rho_ols_bias),
    #     with rho_ols_bias approx -(1+3 rho)/n (Kendall). We store the components.
    #   IVX (Kostakis-Magdalinos-Stamatogiannis 2015): instrument
    #     z_t = sum_{j=1}^{t} Rz^{t-j} Delta x_j, Rz = 1 + cz/n^alpha, cz=-1,
    #     alpha=0.95; beta_ivx = (sum z_t (r_{t+1}-rbar)) / (sum z_t (x_t-xbar)).
    rng = np.random.default_rng(71)
    n = 600
    rho = 0.98
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    u = -0.9 * e + np.sqrt(1 - 0.81) * rng.standard_normal(n)  # corr(u,e) = -0.9
    r = 0.0 + 0.05 * x + u  # r_{t+1} uses x_t; align below

    # Align: predictor x_t (t=0..n-2), target r_{t+1} (t=1..n-1).
    xt = x[:-1]
    rt1 = r[1:]
    xbar, rbar = xt.mean(), rt1.mean()
    b_ols = np.sum((xt - xbar) * (rt1 - rbar)) / np.sum((xt - xbar) ** 2)

    # IVX instrument.
    alpha, cz = 0.95, -1.0
    Rz = 1.0 + cz / n ** alpha
    dx = np.diff(x)  # Delta x_j, length n-1, aligned with xt index
    z = np.empty(len(xt))
    acc = 0.0
    for t in range(len(xt)):
        acc = Rz * acc + dx[t]
        z[t] = acc
    b_ivx = np.sum(z * (rt1 - rbar)) / np.sum(z * (xt - xbar))

    out = {
        "_meta": {"numpy": np.__version__,
                  "note": "Predictive regression with persistent regressor. Formulas in "
                          "the generator docstring; IVX per Kostakis-Magdalinos-"
                          "Stamatogiannis 2015 with cz=-1, alpha=0.95, Rz=1+cz/n^alpha."},
        "x": full(x), "r": full(r), "rho_true": rho,
        "ivx": {"cz": cz, "alpha": alpha, "Rz": float(Rz)},
        "beta_ols": float(b_ols), "beta_ivx": float(b_ivx),
    }
    (OUT / "predreg.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote predreg.json")


if __name__ == "__main__":
    gen_gmm()
    gen_termstructure()
    # gen_predreg() deferred: the self-authored IVX point estimate needs
    # validation against a real reference (R ivx / a published example) before
    # it ships as a golden. The estimator code is kept above for that follow-up.
