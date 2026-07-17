"""Golden fixtures for tsecon-panelts: heterogeneous-panel MG and CCE-MG.

Run with the project venv:

    .venv/bin/python fixtures/generate_tsecon-panelts_fixtures.py

WHAT THIS VALIDATES
-------------------
Two estimators for a heterogeneous panel  y_it = a_i + b_i' x_it + e_it  with
unit-specific slope vectors b_i:

* Mean Group (Pesaran & Smith 1995).  Run a per-unit OLS of y_i on
  [const, x_i], collect the slope vectors b_i (the constant is dropped), and
  report the simple cross-unit average

      b_MG = (1/N) sum_i b_i .

  Its covariance is the sample covariance of the b_i divided by N,

      Var(b_MG) = (1 / (N (N-1))) sum_i (b_i - b_MG)(b_i - b_MG)' ,

  so SE_k = sd_i(b_ik) / sqrt(N) with sd the ddof=1 sample sd.  t = b/SE.

* Common Correlated Effects Mean Group (Pesaran 2006).  Augment each unit-i
  regression with the cross-section averages (over units, per time t) of y and
  of every x,  zbar_t = (1/N) sum_i (y_it, x_it),  as extra regressors, run the
  per-unit OLS on the augmented design [const, x_i, ybar, xbar], then MG-average
  ONLY the own-x slope coefficients.  The cross-section averages span the space
  of the unobserved common factors, so their inclusion purges the factor from
  the own-x slopes.

INDEPENDENCE OF THE GOLDEN
--------------------------
The per-unit OLS fits here are computed by statsmodels' `OLS(...).fit()`, whose
point estimates come from a Moore-Penrose pseudo-inverse (SVD) of the design.
The Rust crate solves the *same* least-squares problems through a completely
different numerical path -- Cholesky factorization of the normal equations in
`tsecon-hac::ols` -- and then applies the *same* deterministic averaging /
sample-covariance formulas written out above.  Because both sides evaluate the
identical closed-form MG and CCE-MG maps on bit-identical inputs (the panel is
stored at full float precision and parsed with serde_json's float_roundtrip),
agreement to ~1e-10 is expected and is a genuine cross-implementation check
(SVD least squares vs. Cholesky normal equations), not a tautology.

A statsmodels-vs-numpy.lstsq self-check is printed below to confirm the design
is well enough conditioned that the two OLS paths already agree to ~1e-12, so
1e-10 is a safe golden tolerance.

DGP
---
A common factor f_t is loaded heterogeneously into both y (via gamma_i) and
every regressor x (via delta_i), with gamma_i and delta_i having positive
means.  Omitting f -- as plain MG does -- therefore leaves an omitted-variable
bias  E[gamma_i delta_i] Var(f)/Var(x)  in the averaged slope that does NOT
wash out across units, so plain MG is biased for the true mean slope while
CCE-MG (which nets out f through the cross-section averages) is close.  The
stored `true_mean_slopes` let the reader see the bias directly.
"""
import json
import platform
from pathlib import Path

import numpy as np
import statsmodels.api as sm

OUT = Path(__file__).parent
META = {
    "numpy": np.__version__,
    "statsmodels": sm.__version__,
    "python": platform.python_version(),
}

rng = np.random.default_rng(20260717)

N, T, K = 24, 70, 2           # units, periods, regressors
b_mean = np.array([1.50, -0.80])   # true mean slope vector
b_sd = np.array([0.30, 0.25])      # cross-unit slope dispersion

# --- unit-specific parameters -------------------------------------------------
a = rng.normal(0.5, 1.0, N)                       # intercepts
b = b_mean[None, :] + rng.normal(0.0, 1.0, (N, K)) * b_sd[None, :]  # slopes b_i
gamma = rng.normal(1.0, 0.5, N)                   # factor loading in y (mean > 0)
delta = rng.normal(0.7, 0.3, (N, K))              # factor loading in x (mean > 0)

# --- common factor and idiosyncratic parts ------------------------------------
f = rng.normal(0.0, 1.0, T)                       # one common factor
mu_x = rng.normal(0.0, 1.0, (N, K))               # unit-specific x means

X = np.empty((N, T, K))
Y = np.empty((N, T))
for i in range(N):
    for k in range(K):
        X[i, :, k] = mu_x[i, k] + delta[i, k] * f + rng.normal(0.0, 1.0, T)
    e = rng.normal(0.0, 0.6, T)                    # idiosyncratic error
    Y[i] = a[i] + X[i] @ b[i] + gamma[i] * f + e


def per_unit_ols(y_i, x_i):
    """Slope vector (constant dropped) from statsmodels OLS (SVD/pinv)."""
    design = sm.add_constant(x_i, prepend=True)    # [const, x...]
    res = sm.OLS(y_i, design).fit()
    return np.asarray(res.params[1:])              # drop the constant


def mean_group(slopes):
    """MG point estimate, SE, t from an (N, k) array of per-unit slopes."""
    slopes = np.asarray(slopes)
    n = slopes.shape[0]
    coef = slopes.mean(axis=0)
    cov = np.cov(slopes, rowvar=False, ddof=1) / n   # sample cov of b_i, /N
    cov = np.atleast_2d(cov)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    return coef, se, t


# --- Mean Group ---------------------------------------------------------------
mg_slopes = np.array([per_unit_ols(Y[i], X[i]) for i in range(N)])
mg_coef, mg_se, mg_t = mean_group(mg_slopes)

# --- CCE Mean Group -----------------------------------------------------------
ybar = Y.mean(axis=0)                    # (T,)   cross-section average of y
xbar = X.mean(axis=0)                    # (T, K) cross-section averages of x
cs_aug = np.column_stack([ybar, xbar])   # (T, 1+K) common-average regressors

cce_slopes = np.empty((N, K))
for i in range(N):
    design = np.column_stack([X[i], cs_aug])   # [x_i, ybar, xbar]  (const added inside)
    slope = per_unit_ols(Y[i], design)         # slopes for [x_i, ybar, xbar]
    cce_slopes[i] = slope[:K]                   # keep only the own-x slopes
cce_coef, cce_se, cce_t = mean_group(cce_slopes)

# --- independence self-check: statsmodels(pinv) vs numpy.lstsq(SVD) -----------
def lstsq_slopes(y_i, x_i):
    design = np.column_stack([np.ones(len(y_i)), x_i])
    beta, *_ = np.linalg.lstsq(design, y_i, rcond=None)
    return beta[1:]


mg_lstsq = np.array([lstsq_slopes(Y[i], X[i]) for i in range(N)])
selfcheck = np.max(np.abs(mg_slopes - mg_lstsq))
print(f"self-check |statsmodels - numpy.lstsq| per-unit slopes: {selfcheck:.2e}")
print(f"true mean slopes : {b_mean}")
print(f"MG   coef        : {mg_coef}   (bias {mg_coef - b_mean})")
print(f"CCEMG coef       : {cce_coef}   (bias {cce_coef - b_mean})")


def m(a2d):
    return [[float(v) for v in row] for row in np.asarray(a2d)]


out = {
    "_meta": META,
    "design": {"N": N, "T": T, "K": K},
    "true_mean_slopes": [float(v) for v in b_mean],
    "y": [[float(v) for v in row] for row in Y],          # N x T
    "x": [m(X[:, :, k]) for k in range(K)],               # K x N x T
    "mg": {
        "coef": [float(v) for v in mg_coef],
        "se": [float(v) for v in mg_se],
        "tstat": [float(v) for v in mg_t],
        "coef_per_unit": m(mg_slopes),                    # N x K
    },
    "cce": {
        "coef": [float(v) for v in cce_coef],
        "se": [float(v) for v in cce_se],
        "tstat": [float(v) for v in cce_t],
        "coef_per_unit": m(cce_slopes),                   # N x K
    },
}

path = OUT / "tsecon-panelts.json"
path.write_text(json.dumps(out))
print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
