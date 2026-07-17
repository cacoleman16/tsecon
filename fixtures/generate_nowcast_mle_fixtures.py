#!/usr/bin/env python
"""Golden fixtures for the tsecon-nowcast ONE-STEP (full-information) Gaussian
maximum-likelihood estimator of the single-factor dynamic factor model.

WHAT THIS VALIDATES
===================
The crate ships two estimators of the single-factor DFM with an AR(p) factor and
white-noise (``error_order=0``) idiosyncratic errors:

  * the Doz-Giannone-Reichlin (2011) *two-step* estimator (validated elsewhere,
    ``fixtures/tsecon-nowcast.json``), and
  * the *one-step* Gaussian MLE added here, which maximises the exact Kalman
    log-likelihood over the DFM parameters.

The one-step MLE targets EXACTLY statsmodels'
``DynamicFactor(endog, k_factors=1, factor_order=p, error_order=0)`` model:

    y_t          = Lambda f_t + e_t,      e_t ~ N(0, diag(sigma2))          (N x 1)
    [f_t    ]    [phi_1 ... phi_p][f_{t-1}]     [1]
    [f_{t-1}] =  [  1   ...   0  ][f_{t-2}] + ..[0] eta_t,  eta_t ~ N(0, 1)  (p x p)
    [ ...   ]    [      ...      ][ ...   ]     [ ]

with the factor-innovation variance NORMALISED to 1 (statsmodels fixes Q = 1 and
lets the loadings carry the factor scale -- verified below: 'L*.f1.f1' AR terms
and 'sigma2.*' idiosyncratic terms are free parameters, but the state covariance
is NOT a free parameter and equals [[1]] at the optimum) and both intercepts
zero.  Because the data mean is not modelled, the panel is CENTRED (column means
removed) before fitting -- the crate's ``fit_mle`` centres internally and this
generator centres the same way, so both optimise the identical likelihood.

TWO KINDS OF CHECK (honest about what is exact and what is not)
==============================================================
1. TIGHT (reference-exact, ~1e-6):  given statsmodels' FITTED parameters, the
   crate's ``smooth_fixed`` (the already-validated Kalman filter/smoother) must
   reproduce statsmodels' maximised ``llf`` and its smoothed factor on the SAME
   centred panel.  This re-confirms the Kalman path at the MLE optimum, not just
   at the DGP parameters.                                         ==> MLE_FITTED

2. MLE OPTIMUM (optimiser-dependent, honest gap):  the crate's ``fit_mle`` and
   statsmodels optimise the SAME function, so the crate's maximised llf should
   land within a small tolerance of statsmodels' ``llf`` (report the gap).  The
   parameter VECTORS are NOT compared to tight tolerance: the factor is
   identified only up to sign (loadings and factor both flip), and optimiser
   differences make a tight parameter match fragile.  Instead the crate asserts
   (a) llf(MLE) >= llf(two-step) on the same centred panel (the MLE is the
   maximum, and is started FROM the two-step estimate), and (b) llf(MLE) is
   within tolerance of statsmodels' llf; as a property, on simulated data the
   MLE smoothed factor tracks the true factor (|corr| > 0.9).

DATA-GENERATING PROCESS
=======================
Balanced monthly panel, N series, T months, one common factor following a
stationary AR(p).  Drawn from a single seeded numpy Generator for bit-for-bit
reproducibility.
"""

import json
import os

import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nowcast_mle.json")

# --------------------------------------------------------------------------
# 1. Data-generating process (single seeded RNG for reproducibility).
# --------------------------------------------------------------------------
rng = np.random.default_rng(20260718)

N = 6            # cross-section size (series)
T = 200          # months
P = 2            # factor AR order
BURN = 200       # AR burn-in discarded

# True AR(p) factor coefficients (stationary: roots outside the unit circle).
PHI = np.array([0.5, 0.25])          # f_t = 0.5 f_{t-1} + 0.25 f_{t-2} + eta_t
assert np.all(np.abs(np.roots(np.r_[1.0, -PHI])) < 1.0), "AR not stationary"

# True loadings (mixed signs) and idiosyncratic variances.
LOADINGS = np.array([1.0, 0.8, -0.6, 0.7, -0.5, 0.9])
IDIO = np.array([0.30, 0.40, 0.25, 0.50, 0.35, 0.45])
assert LOADINGS.shape == (N,) and IDIO.shape == (N,)

# Simulate the factor with unit-variance innovations (matches the state-space
# normalisation Q = 1 used by DynamicFactor).
n_tot = T + BURN
eta = rng.standard_normal(n_tot)
f_full = np.zeros(n_tot)
for t in range(P, n_tot):
    f_full[t] = PHI[0] * f_full[t - 1] + PHI[1] * f_full[t - 2] + eta[t]
factor = f_full[BURN:]                          # length T, the "true" factor

# Idiosyncratic noise and the observed panel  y_t = Lambda f_t + e_t.
noise = rng.standard_normal((T, N)) * np.sqrt(IDIO)
Y = np.outer(factor, LOADINGS) + noise          # T x N balanced panel

# Centre by column means (DynamicFactor does not model the mean; both this
# generator and the crate's fit_mle remove it before fitting).
center = Y.mean(axis=0)
Yc = Y - center

# --------------------------------------------------------------------------
# 2. statsmodels DynamicFactor: one-step Gaussian MLE on the centred panel.
#    param order (verified via mod.param_names):
#        [ loadings (N), sigma2_idio (N), factor_AR (P) ]
# --------------------------------------------------------------------------
mod = DynamicFactor(Yc, k_factors=1, factor_order=P, error_order=0)
assert mod.k_states == P and mod.ssm.k_posdef == 1

expected_names = (
    [f"loading.f1.y{i+1}" for i in range(N)]
    + [f"sigma2.y{i+1}" for i in range(N)]
    + [f"L{j+1}.f1.f1" for j in range(P)]
)
assert list(mod.param_names) == expected_names, (list(mod.param_names), expected_names)

res = mod.fit(disp=False, maxiter=1000)
# A short EM/BFGS polish from the found optimum to sharpen llf a touch.
res = mod.fit(res.params, disp=False, maxiter=1000, method="lbfgs")

fitted = np.asarray(res.params)
loadings_mle = fitted[:N].tolist()
idio_mle = fitted[N:2 * N].tolist()
phi_mle = fitted[2 * N:2 * N + P].tolist()
llf_mle = float(res.llf)

# Q must be normalised to 1 at the optimum (it is not a free parameter).
qc = mod.ssm["state_cov"]
qc = qc[:, :, 0] if qc.ndim == 3 else qc
np.testing.assert_allclose(qc, [[1.0]])

# statsmodels' smoothed factor at the fitted params (state column 0).
smoothed_state = np.asarray(res.states.smoothed)   # T x P
assert smoothed_state.shape == (T, P)
smoothed_factor_mle = smoothed_state[:, 0].tolist()

# Confirm the state-space matrices are exactly what the crate assembles, so the
# TIGHT check is a like-for-like comparison.
def _m(name):
    a = mod.ssm[name]
    return a[:, :, 0] if a.ndim == 3 else a

res_check = mod.smooth(res.params, transformed=True)
np.testing.assert_allclose(_m("design")[:, 0], loadings_mle)
np.testing.assert_allclose(_m("design")[:, 1:], 0.0)
np.testing.assert_allclose(_m("transition")[0, :], phi_mle)       # AR in first row
np.testing.assert_allclose(_m("transition")[1, 0], 1.0)           # sub-diagonal one
np.testing.assert_allclose(_m("selection")[:, 0], [1.0, 0.0])
np.testing.assert_allclose(_m("state_cov"), [[1.0]])              # Q normalised
np.testing.assert_allclose(np.diag(_m("obs_cov")), idio_mle)
np.testing.assert_allclose(float(res_check.llf), llf_mle, rtol=0, atol=1e-9)

# Context: |corr| of the MLE smoothed factor with the simulated truth.
corr = float(np.corrcoef(smoothed_factor_mle, factor)[0, 1])

# --------------------------------------------------------------------------
# 3. Serialise.
# --------------------------------------------------------------------------
doc = {
    "_doc": (
        "Golden for the tsecon-nowcast ONE-STEP Gaussian MLE of the "
        "single-factor DFM (statsmodels DynamicFactor(k_factors=1, "
        "factor_order=p, error_order=0)). 'mle_fitted' is the reference: given "
        "statsmodels' fitted params, the crate's smooth_fixed must reproduce "
        "its llf and smoothed factor on the CENTRED panel to ~1e-6 (TIGHT). "
        "The crate's fit_mle optimises the SAME likelihood; it asserts "
        "llf(MLE) >= llf(two-step) and matches statsmodels' llf within a "
        "reported tolerance (OPTIMUM). Parameter vectors are NOT tolerance-"
        "matched (sign / optimiser ambiguity)."
    ),
    "dims": {"n_series": N, "n_obs": T, "factor_order": P, "k_factors": 1},
    "dgp": {
        "phi": PHI.tolist(),
        "loadings": LOADINGS.tolist(),
        "idiosyncratic": IDIO.tolist(),
        "factor_innovation_var": 1.0,
    },
    # Raw balanced panel (T x N, row-major). The crate centres it internally.
    "panel": Y.tolist(),
    # Column means removed before fitting (crate recomputes the same values;
    # stored so the TIGHT check builds the identical centred panel).
    "center": center.tolist(),
    # Simulated true factor (for the |corr| > 0.9 structural property).
    "true_factor": factor.tolist(),
    # MLE_FITTED reference-exact target (statsmodels one-step MLE optimum),
    # laid out to match the crate's DfmParams for a single factor.
    "mle_fitted": {
        "params_layout": (
            "loadings(N) then idiosyncratic(N) then factor_ar(P); factor_cov "
            "fixed to 1. Fitted on the centred panel (panel - center)."
        ),
        "loadings": loadings_mle,          # N, DfmParams.loadings (N x 1)
        "factor_ar": phi_mle,              # P, DfmParams.factor_ar (1 x P)
        "idiosyncratic": idio_mle,         # N, DfmParams.idiosyncratic
        "factor_cov": 1.0,                 # DfmParams.factor_cov (1 x 1)
        "loglik": llf_mle,                 # statsmodels maximised results.llf
        "smoothed_factor": smoothed_factor_mle,  # T, state column 0
        "corr_with_true_factor": corr,     # context only
    },
}

with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=2)

print(f"wrote {OUT}")
print(f"  N={N} T={T} P={P}")
print(f"  MLE llf = {llf_mle!r}")
print(f"  |corr(smoothed, true)| = {abs(corr):.4f}")
print(f"  fitted phi = {phi_mle}")
