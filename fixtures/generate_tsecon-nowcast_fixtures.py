#!/usr/bin/env python
"""Golden fixtures for the tsecon-nowcast crate (dynamic-factor nowcasting).

VALIDATION STRATEGY
===================
The tsecon-nowcast crate implements the Doz-Giannone-Reichlin (2011) *two-step*
dynamic-factor estimator and the Banbura-Modugno (2014) ragged-edge Kalman
nowcast.  Two distinct things must be validated, and they are validated
differently because they are different objects:

1.  THE KALMAN / STATE-SPACE STEP  ->  reference-EXACT (~1e-8).
    A single-factor dynamic factor model with an AR(p) factor and white-noise
    (error_order=0) idiosyncratic components is a linear-Gaussian state-space
    model:

        y_t          = Lambda f_t + e_t,       e_t ~ N(0, diag(sigma2))     (N x 1)
        [f_t ]        [phi_1 ... phi_p][f_{t-1}]        [1]
        [f_{t-1}] =   [  1   ...   0  ][f_{t-2}]  + ... [0] eta_t,  eta_t ~ N(0,1)
        [ ...  ]      [      ...      ][ ...   ]        [ ]
        (companion / statsmodels DynamicFactor layout: AR coeffs in the FIRST
         ROW, ones on the sub-diagonal; factor innovation variance NORMALISED
         to 1; stationary initialisation).

    statsmodels.tsa.statespace.dynamic_factor.DynamicFactor builds EXACTLY this
    representation.  We therefore fix the true DGP parameter vector, feed it to
    statsmodels via ``mod.smooth(params, transformed=True)`` (which runs the
    Kalman filter + smoother at those parameters WITHOUT re-estimating), and
    store the resulting Kalman log-likelihood ``res.llf`` and the smoothed
    states ``res.states.smoothed``.  The Rust crate, given the SAME loadings /
    factor-AR / idiosyncratic variances on the SAME raw panel, must reproduce
    both to ~1e-8.  This isolates and validates the crate's Kalman/state-space
    step against statsmodels as an independent reference.  ==> KALMAN_FIXED

2.  THE TWO-STEP DGR PARAMETER ESTIMATES  ->  NOT exact.
    The DGR two-step estimator (PCA factors -> factor VAR -> one Kalman pass) is
    a DIFFERENT estimator from statsmodels' one-step Gaussian MLE, so their
    parameter estimates and smoothed factors do NOT coincide and MUST NOT be
    tolerance-matched.  For context we also fit the statsmodels MLE and store
    its llf / smoothed factor, but the Rust tests only use these to sanity-check
    orders of magnitude and (via the simulated true factor) to check that the
    two-step smoothed factor tracks the truth (corr > 0.9) -- a structural
    property, not a golden equality.  ==> MLE_REF and the simulated truth.

So: the Kalman step is reference-exact; the two-step parameter estimates are the
DGR estimator (documented as such, not one-step MLE).

DATA-GENERATING PROCESS
=======================
Balanced monthly panel, N series, T months, one common factor following a
stationary AR(p).  Everything is drawn from a single seeded numpy Generator so
the fixture is reproducible bit-for-bit.
"""

import json
import os

import numpy as np
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tsecon-nowcast.json")

# --------------------------------------------------------------------------
# 1. Data-generating process (single seeded RNG for reproducibility).
# --------------------------------------------------------------------------
rng = np.random.default_rng(20260717)

N = 8            # cross-section size (series)
T = 180          # months (15 years)
P = 2            # factor AR order
BURN = 200       # AR burn-in discarded

# True AR(p) factor coefficients (stationary: roots outside unit circle).
PHI = np.array([0.6, 0.2])           # f_t = 0.6 f_{t-1} + 0.2 f_{t-2} + eta_t
assert np.all(np.abs(np.roots(np.r_[1.0, -PHI])) < 1.0), "AR not stationary"

# True loadings and idiosyncratic variances.
LOADINGS = np.array([1.0, 0.9, 0.8, -0.7, 0.6, -0.5, 1.1, 0.4])
IDIO = np.array([0.30, 0.45, 0.25, 0.50, 0.35, 0.40, 0.20, 0.55])
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

# --------------------------------------------------------------------------
# 2. statsmodels DynamicFactor: exact state-space representation.
#    param order (verified via mod.param_names):
#        [ loadings (N), sigma2_idio (N), factor_AR (P) ]
# --------------------------------------------------------------------------
mod = DynamicFactor(Y, k_factors=1, factor_order=P, error_order=0)
assert mod.k_states == P and mod.ssm.k_posdef == 1

# --- 2a. KALMAN_FIXED: filter/smoother at the TRUE parameters ------------
dgp_params = np.concatenate([LOADINGS, IDIO, PHI])
# sanity: param_names line up with our concatenation order.
expected_names = (
    [f"loading.f1.y{i+1}" for i in range(N)]
    + [f"sigma2.y{i+1}" for i in range(N)]
    + [f"L{j+1}.f1.f1" for j in range(P)]
)
assert list(mod.param_names) == expected_names, (mod.param_names, expected_names)

res_fixed = mod.smooth(dgp_params, transformed=True)
llf_fixed = float(res_fixed.llf)
# res.states.smoothed is a (T x k_states) DataFrame; column 0 is the factor.
smoothed_state_fixed = np.asarray(res_fixed.states.smoothed)   # T x P
assert smoothed_state_fixed.shape == (T, P)

# Confirm the state-space matrices are what the Rust crate will assemble.
def _m(name):
    a = mod.ssm[name]
    return a[:, :, 0] if a.ndim == 3 else a
np.testing.assert_allclose(_m("design")[:, 0], LOADINGS)
np.testing.assert_allclose(_m("design")[:, 1:], 0.0)
np.testing.assert_allclose(_m("transition")[0, :], PHI)          # AR in first row
np.testing.assert_allclose(_m("transition")[1, 0], 1.0)          # sub-diagonal one
np.testing.assert_allclose(_m("selection")[:, 0], [1.0, 0.0])
np.testing.assert_allclose(_m("state_cov"), [[1.0]])             # Q normalised
np.testing.assert_allclose(np.diag(_m("obs_cov")), IDIO)

# --- 2b. MLE_REF: one-step Gaussian MLE (context only, NOT a tight golden) --
res_mle = mod.fit(disp=False, maxiter=200)
llf_mle = float(res_mle.llf)
smoothed_factor_mle = np.asarray(res_mle.states.smoothed)[:, 0].tolist()
loadings_mle = np.asarray(res_mle.params[:N]).tolist()

# --------------------------------------------------------------------------
# 3. Serialise.
# --------------------------------------------------------------------------
doc = {
    "_doc": (
        "Golden for tsecon-nowcast. 'kalman_fixed' is the reference-EXACT "
        "target (statsmodels Kalman filter+smoother at the true DGP params, "
        "match to ~1e-8). 'mle_ref' and 'dgp' are context for the two-step "
        "DGR estimator, which is a DIFFERENT estimator and is validated "
        "structurally (corr with the true factor), not by tolerance."
    ),
    "dims": {"n_series": N, "n_obs": T, "factor_order": P, "k_factors": 1},
    "dgp": {
        "phi": PHI.tolist(),
        "loadings": LOADINGS.tolist(),
        "idiosyncratic": IDIO.tolist(),
        "factor_innovation_var": 1.0,
    },
    # The raw balanced panel (T x N, row-major) the crate consumes.
    "panel": Y.tolist(),
    # The simulated true factor (for the corr > 0.9 structural check).
    "true_factor": factor.tolist(),
    # KALMAN_FIXED reference-exact target.
    "kalman_fixed": {
        "params_order": "loadings(N), idiosyncratic(N), phi(P)",
        "loglik": llf_fixed,
        # T x P smoothed state; column 0 is the smoothed factor f_t.
        "smoothed_state": smoothed_state_fixed.tolist(),
    },
    # One-step MLE context (NOT a tight golden).
    "mle_ref": {
        "loglik": llf_mle,
        "loadings": loadings_mle,
        "smoothed_factor": smoothed_factor_mle,
    },
}

with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=2)

print(f"wrote {OUT}")
print(f"  N={N} T={T} P={P}")
print(f"  KALMAN_FIXED llf = {llf_fixed!r}")
print(f"  MLE_REF     llf = {llf_mle!r}")
