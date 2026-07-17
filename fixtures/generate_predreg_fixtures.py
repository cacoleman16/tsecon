"""Golden fixtures for tsecon-predreg: OLS / Stambaugh / IVX predictive
regressions with a persistent regressor.

These fixtures are DOCUMENTED-FORMULA goldens (validation target (a) of the
crate spec). On a FIXED simulated (r, x) series they compute every published
quantity by LITERALLY writing the closed-form formula in NumPy — no call to
the Rust crate, so the check is not circular. Passing them proves that the
Rust reproduces the published algebra to ~1e-10. It does NOT by itself prove
the formulas are the statistically right ones; that is what the crate's
Monte-Carlo size/power property tests establish (targets (b)-(d)).

Run with the project venv:
    .venv/bin/python fixtures/generate_predreg_fixtures.py

================================================================================
MODEL (the Stambaugh 1999 setting)
================================================================================
Predictive regression, one step ahead:
    r_{t+1} = alpha + beta * x_t + u_{t+1}
Persistent (near-unit-root) predictor, AR(1):
    x_t = rho * x_{t-1} + e_t
with corr(u, e) != 0 (contemporaneous innovation correlation) — the source of
the Stambaugh finite-sample bias.

ALIGNMENT. With a length-n series x[0..n-1], r[0..n-1] we form N = n-1 pairs
    predictor a_t := x[t]        for t = 0 .. N-1
    target    b_t := r[t+1]      for t = 0 .. N-1
so b_t is the return realised one period after the predictor a_t is observed.

================================================================================
(1) OLS predictive regression
================================================================================
    abar = mean(a), bbar = mean(b)
    beta_ols = sum_t (a_t-abar)(b_t-bbar) / sum_t (a_t-abar)^2
    alpha_ols = bbar - beta_ols*abar
    u_hat_t = b_t - alpha_ols - beta_ols*a_t
    s2_u = sum_t u_hat_t^2 / (N-2)                    (unbiased, OLS)
    se(beta_ols) = sqrt( s2_u / sum_t (a_t-abar)^2 )  (nonrobust)
    t(beta_ols) = beta_ols / se(beta_ols)

================================================================================
(2) Stambaugh (1999) bias correction
================================================================================
Because x is persistent and its innovation e is correlated with u, the OLS
predictor coefficient is biased in finite samples with
    E[beta_ols - beta] = (sigma_ue / sigma_ee) * E[rho_hat - rho].
Stambaugh (1999, J. Financial Economics 54:375-421, eqs. 4-6) removes it using
the AR(1) least-squares estimate rho_ols and the Kendall (1954) bias of the
AR(1) coefficient:
    rho_ols  = sum_t (x_{t-1}-xlbar)(x_t-xcbar) / sum_t (x_{t-1}-xlbar)^2
    e_hat_t  = x_t - (xcbar - rho_ols*xlbar) - rho_ols*x_{t-1}   (AR(1) resid)
    sigma_ee = sum e_hat_t^2 / (N)          (innovation variance of x)
    sigma_ue = sum u_hat_t * e_hat_t / (N)  (u_hat aligned to same target time)
    Kendall bias:  E[rho_hat - rho] ~ -(1 + 3*rho)/n      (rho_true or rho_ols)
    bias_rho = -(1 + 3*rho_ols)/n
    bias_term = (sigma_ue / sigma_ee) * bias_rho
    beta_corrected = beta_ols - bias_term
      = beta_ols - (sigma_ue/sigma_ee)*(rho_ols - rho_corrected),
        rho_corrected = rho_ols - bias_rho    (the Kendall-adjusted AR root)
SE. The correction shifts the point estimate by a data-dependent constant; to
first order its sampling variance is the OLS variance, so we report
    se(beta_corrected) = se(beta_ols).

================================================================================
(3) IVX estimator (Kostakis, Magdalinos & Stamatogiannis 2015, RFS 28:1506-1553)
================================================================================
Self-generated instrument, "mildly integrated" with persistence Rz just inside
the unit circle:
    Rz = 1 + cz / n^alpha          (defaults cz = -1, alpha = 0.95)
    Dx_k = x[k+1]-x[k]             (k = 0 .. n-2; carries innovation e_{k+1})
    z_t = sum_{k=0}^{t-1} Rz^{t-1-k} * Dx_k        (z_0 = 0)
        equivalently  z_0 = 0,  z_t = Rz*z_{t-1} + Dx_{t-1}
CRUCIAL: z_t uses x-innovations only up to time t, so it is PREDETERMINED with
respect to u_{t+1} (which is correlated with the *future* innovation e_{t+1}).
This predetermination is what makes IVX inference valid under endogeneity.

    beta_ivx = sum_t z_t*(b_t-bbar) / sum_t z_t*(a_t-abar)

================================================================================
(4) IVX-Wald predictability test for H0: beta = 0
================================================================================
KMS (2015, Sec. 3) self-normalised statistic. Because z_t is a mean-zero,
mildly-integrated process (z_0 = 0) that is asymptotically orthogonal to the
constant, the instrument is NOT demeaned in the variance normaliser:
    num = sum_t z_t*(b_t-bbar)         (= numerator of beta_ivx)
    Szz = sum_t z_t^2                  (RAW instrument second moment)
    s2u_ivx = sum_t u_hat_t^2 / N      (predictive-regression residual variance)
    W_ivx = num^2 / (s2u_ivx * Szz)
W_ivx is asymptotically chi-square(q) (q = number of predictors) UNIFORMLY over
the persistence of x — stationary, local-to-unity, or exact unit root — which
is the reason IVX exists. p-value = chi2_sf(W_ivx, q).

Multivariate (q predictors x_i, shared scalar Rz): with demeaned regressor
matrix A (columns a_i) and instruments z_i,
    A_mat[i][j] = sum_t z_{i,t}*(a_{j,t}-abar_j)      (q x q)
    c[i]        = sum_t z_{i,t}*(b_t-bbar)            (q)
    beta_ivx = A_mat^{-1} c
    M[i][j] = s2u_ivx * sum_t z_{i,t} z_{j,t}         (q x q, raw)
    W_ivx = c' M^{-1} c ,   ~ chi-square(q).
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


def ar1_series(n, rho, rng):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x, e


def ivx_instrument(x, cz, alpha):
    n = len(x)
    N = n - 1
    Rz = 1.0 + cz / n ** alpha
    dx = np.diff(x)  # dx[k] = x[k+1]-x[k]
    z = np.empty(N)
    acc = 0.0
    for t in range(N):
        z[t] = acc                 # z_t uses dx[0..t-1] only (predetermined)
        acc = Rz * acc + dx[t]
    return z, Rz


def ols_predictive(a, b):
    abar, bbar = a.mean(), b.mean()
    ad, bd = a - abar, b - bbar
    Saa = np.sum(ad * ad)
    beta_ols = np.sum(ad * bd) / Saa
    alpha_ols = bbar - beta_ols * abar
    u_hat = b - alpha_ols - beta_ols * a
    N = len(a)
    s2_u = np.sum(u_hat ** 2) / (N - 2)
    se = np.sqrt(s2_u / Saa)
    return dict(beta_ols=beta_ols, alpha_ols=alpha_ols, se=se,
                tstat=beta_ols / se, u_hat=u_hat, Saa=Saa)


def gen_scalar():
    rng = np.random.default_rng(20260717)
    n, rho, cue, beta_true = 500, 0.98, -0.9, 0.05
    x, e = ar1_series(n, rho, rng)
    u = cue * e + np.sqrt(1.0 - cue ** 2) * rng.standard_normal(n)
    r = 0.0 + beta_true * x + u

    a, b = x[:-1], r[1:]
    N = len(a)
    ols = ols_predictive(a, b)

    # --- Stambaugh ---
    xl, xc = x[:-1], x[1:]
    xlbar, xcbar = xl.mean(), xc.mean()
    rho_ols = np.sum((xl - xlbar) * (xc - xcbar)) / np.sum((xl - xlbar) ** 2)
    e_hat = xc - (xcbar - rho_ols * xlbar) - rho_ols * xl
    sigma_ee = np.sum(e_hat ** 2) / N
    sigma_ue = np.sum(ols["u_hat"] * e_hat) / N
    bias_rho = -(1.0 + 3.0 * rho_ols) / n
    bias_term = (sigma_ue / sigma_ee) * bias_rho
    beta_corrected = ols["beta_ols"] - bias_term

    # --- IVX + Wald ---
    cz, alpha = -1.0, 0.95
    z, Rz = ivx_instrument(x, cz, alpha)
    abar, bbar = a.mean(), b.mean()
    num = np.sum(z * (b - bbar))
    den = np.sum(z * (a - abar))
    beta_ivx = num / den
    Szz = np.sum(z * z)
    s2u_ivx = np.sum(ols["u_hat"] ** 2) / N
    wald = num * num / (s2u_ivx * Szz)
    # chi-square(1) survival = erfc(sqrt(W/2))
    from math import erfc, sqrt
    pval = erfc(sqrt(wald / 2.0))

    return dict(
        n=n, rho_true=rho, corr_ue=cue, beta_true=beta_true,
        x=full(x), r=full(r),
        ols=dict(beta_ols=float(ols["beta_ols"]), alpha_ols=float(ols["alpha_ols"]),
                 se=float(ols["se"]), tstat=float(ols["tstat"])),
        stambaugh=dict(rho_ols=float(rho_ols), sigma_ee=float(sigma_ee),
                       sigma_ue=float(sigma_ue), bias_rho=float(bias_rho),
                       bias_term=float(bias_term),
                       beta_corrected=float(beta_corrected),
                       se=float(ols["se"])),
        ivx=dict(cz=cz, alpha=alpha, Rz=float(Rz),
                 z=full(z), beta_ivx=float(beta_ivx),
                 num=float(num), den=float(den), Szz=float(Szz),
                 s2u=float(s2u_ivx), wald=float(wald), pvalue=float(pval)),
    )


def gen_multi():
    """Two-predictor IVX (matrix form), shared scalar Rz."""
    rng = np.random.default_rng(19990401)
    n = 600
    rho1, rho2 = 0.97, 0.90
    x1, e1 = ar1_series(n, rho1, rng)
    x2, e2 = ar1_series(n, rho2, rng)
    # u correlated with both innovations
    u = -0.7 * e1 - 0.4 * e2 + rng.standard_normal(n)
    r = 0.0 + 0.03 * x1 - 0.02 * x2 + u

    b = r[1:]
    a1, a2 = x1[:-1], x2[:-1]
    N = len(b)
    cols = [np.ones(N), a1, a2]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, b, rcond=None)
    u_hat = b - X @ coef
    s2u = np.sum(u_hat ** 2) / N

    cz, alpha = -1.0, 0.95
    z1, Rz = ivx_instrument(x1, cz, alpha)
    z2, _ = ivx_instrument(x2, cz, alpha)
    bbar = b.mean()
    a1d, a2d = a1 - a1.mean(), a2 - a2.mean()
    bd = b - bbar
    Z = [z1, z2]
    A = np.array([[np.sum(Z[i] * [a1d, a2d][j]) for j in range(2)] for i in range(2)])
    c = np.array([np.sum(Z[i] * bd) for i in range(2)])
    beta_ivx = np.linalg.solve(A, c)
    M = s2u * np.array([[np.sum(Z[i] * Z[j]) for j in range(2)] for i in range(2)])
    wald = float(c @ np.linalg.solve(M, c))
    from scipy.stats import chi2
    pval = float(chi2.sf(wald, 2))

    return dict(
        n=n, rho1=rho1, rho2=rho2,
        x1=full(x1), x2=full(x2), r=full(r),
        ivx=dict(cz=cz, alpha=alpha, Rz=float(Rz),
                 beta_ivx=full(beta_ivx), s2u=float(s2u),
                 wald=wald, pvalue=pval),
    )


def main():
    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": "Predictive regression (Stambaugh 1999 setting): OLS, "
                    "Stambaugh bias correction, IVX and IVX-Wald "
                    "(Kostakis-Magdalinos-Stamatogiannis 2015). All formulas "
                    "in the generator docstring; documented-formula golden.",
        },
        "scalar": gen_scalar(),
        "multi": gen_multi(),
    }
    (OUT / "predreg.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote predreg.json")


if __name__ == "__main__":
    main()
