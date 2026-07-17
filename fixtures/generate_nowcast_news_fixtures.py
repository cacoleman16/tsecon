#!/usr/bin/env python
"""Golden fixture for the tsecon-nowcast NEWS / update decomposition
(Banbura-Modugno 2014), consumed by ``crates/tsecon-nowcast/tests/news.rs``.

WHAT IS BEING VALIDATED
=======================
When a newer data vintage reveals additional observations at the ragged edge,
the nowcast of a target series revises.  For a FIXED-parameter dynamic factor
model the Kalman smoother is a purely LINEAR operator on the observed data
(zero intercept: d = 0, c = 0, stationary initialisation has zero mean), so the
target nowcast is an AFFINE function of the observations and the revision
decomposes EXACTLY as a weighted sum of news:

    new_nowcast - old_nowcast = sum_j  weight_j * news_j                    (*)

with, for each newly-revealed cell j = (period t, series i),

    news_j     = actual_j - forecast_j
    forecast_j = E_old[y_j | old vintage]     (old-vintage Kalman forecast)
    weight_j   = d(nowcast) / d(actual_j)      (Kalman sensitivity)
    contribution_j = weight_j * news_j .

INDEPENDENT REFERENCE (this file)
=================================
The Rust crate computes the weights ANALYTICALLY, by exploiting the exact
linearity of its own Kalman smoother (a unit-impulse smoother pass).  To
validate that, this generator implements a completely INDEPENDENT reference: a
plain textbook multivariate Kalman filter + RTS (Rauch-Tung-Striebel) smoother
in NumPy, with missing-data handling (drop the missing rows of the measurement
update).  It is a different implementation, in a different language, using the
standard closed-form recursions:

    Predict:  a_pred = T a,                 P_pred = T P T' + R Q R'
    Update:   v = y_obs - Z_obs a_pred,     F = Z_obs P_pred Z_obs' + H_obs
              K = P_pred Z_obs' F^{-1}
              a = a_pred + K v,             P = P_pred - K Z_obs P_pred
    RTS:      C = P_filt T' P_pred^{-1}
              a_sm = a_filt + C (a_sm_next - a_pred_next)

with STATIONARY initialisation a_1 = 0, P_1 = solve_discrete_lyapunov(T, RQR').
This is the SAME state space the crate assembles (statsmodels DynamicFactor
layout, verified elsewhere in fixtures/generate_tsecon-nowcast_fixtures.py), so
the two smoothers agree to ~1e-9 -- but the golden here is generated with NO
reference to the Rust code, so it is a trustworthy independent check.

The reference computes, on this small design:
  * old_nowcast / new_nowcast  -- the common-component projection of the target
    onto the smoothed factor, de-standardised;
  * forecast_j                 -- old-vintage smoothed projection of each cell;
  * weight_j via FINITE DIFFERENCES -- perturb the raw new-vintage observation
    y_j by +/- eps, re-run the WHOLE smoother, and take the central difference
    [g(y_j+eps) - g(y_j-eps)] / (2 eps).  This is the "direct" definition of the
    Kalman weight, computed WITHOUT the analytic impulse-response trick, so
    matching it to ~1e-6 is a genuine cross-check of the Rust analytic weight.

As an internal sanity check the generator also asserts the exact identity (*)
holds for the reference itself to ~1e-10.
"""

import json
import os

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "nowcast_news.json")

# --------------------------------------------------------------------------
# 1. A small, fixed single-factor DFM on the STANDARDISED scale.
#    Parameters live on the standardised panel (as the crate's do); the raw
#    vintages are mapped back with per-series (center, scale).
# --------------------------------------------------------------------------
rng = np.random.default_rng(20260718)

N = 5            # series
T = 30           # periods
P = 2            # factor AR order
r = 1            # single factor

PHI = np.array([0.5, 0.2])                       # f_t = .5 f_{t-1}+.2 f_{t-2}+eta
assert np.all(np.abs(np.roots(np.r_[1.0, -PHI])) < 1.0), "AR not stationary"
LOADINGS = np.array([1.0, 0.8, -0.6, 0.5, 0.9])  # standardised-scale loadings
IDIO = np.array([0.40, 0.55, 0.30, 0.60, 0.45])  # idiosyncratic variances
Q = 1.0                                          # factor innovation variance

# De-standardisation moments (arbitrary but realistic; scale > 0).
CENTER = np.array([100.0, 50.0, -20.0, 10.0, 0.02])
SCALE = np.array([5.0, 3.0, 4.0, 2.0, 0.01])

# --------------------------------------------------------------------------
# 2. State-space matrices (companion / statsmodels DynamicFactor layout).
#      alpha = [f_t, f_{t-1}]',  m = r*P = 2
# --------------------------------------------------------------------------
m = r * P
Z_full = np.zeros((N, m))
Z_full[:, 0] = LOADINGS                            # y = Lambda f + e
H = np.diag(IDIO)
Tm = np.zeros((m, m))
Tm[0, :] = PHI                                     # AR coeffs in the first row
for i in range(1, m):
    Tm[i, i - 1] = 1.0                             # sub-diagonal shift
Rsel = np.zeros((m, r))
Rsel[0, 0] = 1.0
Qm = np.array([[Q]])
RQR = Rsel @ Qm @ Rsel.T
P1 = solve_discrete_lyapunov(Tm, RQR)              # stationary initial cov
a1 = np.zeros(m)


def kalman_smoother(z):
    """Multivariate Kalman filter + RTS smoother on standardised panel z (T x N).

    NaN entries are missing: the measurement update uses only the observed rows.
    Returns the smoothed states alpha_hat (T x m).  Standard closed-form
    recursions with stationary initialisation -- an independent reference, no
    reference to the Rust implementation.
    """
    T_ = z.shape[0]
    a_pred = np.empty((T_, m))
    P_pred = np.empty((T_, m, m))
    a_filt = np.empty((T_, m))
    P_filt = np.empty((T_, m, m))

    a, Pc = a1.copy(), P1.copy()
    for t in range(T_):
        if t == 0:
            ap, Pp = a1.copy(), P1.copy()
        else:
            ap = Tm @ a
            Pp = Tm @ Pc @ Tm.T + RQR
        a_pred[t], P_pred[t] = ap, Pp

        obs = np.where(np.isfinite(z[t]))[0]
        if obs.size == 0:
            a, Pc = ap, Pp
        else:
            Zt = Z_full[obs, :]
            Ht = H[np.ix_(obs, obs)]
            yv = z[t, obs] - Zt @ ap
            F = Zt @ Pp @ Zt.T + Ht
            K = Pp @ Zt.T @ np.linalg.inv(F)
            a = ap + K @ yv
            Pc = Pp - K @ Zt @ Pp
        a_filt[t], P_filt[t] = a, Pc

    # RTS backward smoother.
    a_sm = a_filt.copy()
    for t in range(T_ - 2, -1, -1):
        C = P_filt[t] @ Tm.T @ np.linalg.inv(P_pred[t + 1])
        a_sm[t] = a_filt[t] + C @ (a_sm[t + 1] - a_pred[t + 1])
    return a_sm


def standardize(raw):
    return (raw - CENTER) / SCALE


def target_nowcast(raw_vintage, series, period):
    """Common-component nowcast of `series` at `period`, de-standardised."""
    a_sm = kalman_smoother(standardize(raw_vintage))
    f = a_sm[period, :r]
    return CENTER[series] + SCALE[series] * (LOADINGS[series] * f[0])


# --------------------------------------------------------------------------
# 3. Simulate a raw panel, then build old/new vintages (ragged edge).
# --------------------------------------------------------------------------
n_tot = T + 200
eta = rng.standard_normal(n_tot) * np.sqrt(Q)
f = np.zeros(n_tot)
for t in range(P, n_tot):
    f[t] = PHI[0] * f[t - 1] + PHI[1] * f[t - 2] + eta[t]
factor = f[200:]
e = rng.standard_normal((T, N)) * np.sqrt(IDIO)
z_panel = np.outer(factor, LOADINGS) + e            # standardised panel
Y = CENTER + SCALE * z_panel                        # raw levels (T x N)

# New vintage: rows 0..T-3 fully observed; row T-2 fully observed;
# row T-1 observes series {0,1,2}, series {3,4} still missing.
new_vintage = Y.copy()
new_vintage[T - 1, 3] = np.nan
new_vintage[T - 1, 4] = np.nan

# Old vintage: like the new one but with MORE missing at the edge --
# row T-2 missing series {3,4}, row T-1 missing series {0,1,2} too.
old_vintage = new_vintage.copy()
old_vintage[T - 2, 3] = np.nan
old_vintage[T - 2, 4] = np.nan
old_vintage[T - 1, 0] = np.nan
old_vintage[T - 1, 1] = np.nan
old_vintage[T - 1, 2] = np.nan

# Newly-revealed set J = finite in new AND nan in old (row-major order).
J = []
for t in range(T):
    for i in range(N):
        if np.isfinite(new_vintage[t, i]) and not np.isfinite(old_vintage[t, i]):
            J.append((t, i))
# Sanity: every cell observed in old is unchanged in new.
for t in range(T):
    for i in range(N):
        if np.isfinite(old_vintage[t, i]):
            assert np.isfinite(new_vintage[t, i])
            assert old_vintage[t, i] == new_vintage[t, i]

TARGET_SERIES = 4         # missing at the edge in both vintages -> a true nowcast
TARGET_PERIOD = T - 1
assert not np.isfinite(new_vintage[TARGET_PERIOD, TARGET_SERIES])

# --------------------------------------------------------------------------
# 4. Reference computation: nowcasts, forecasts, finite-difference weights.
# --------------------------------------------------------------------------
a_sm_old = kalman_smoother(standardize(old_vintage))
a_sm_new = kalman_smoother(standardize(new_vintage))

old_nowcast = CENTER[TARGET_SERIES] + SCALE[TARGET_SERIES] * (
    LOADINGS[TARGET_SERIES] * a_sm_old[TARGET_PERIOD, 0]
)
new_nowcast = CENTER[TARGET_SERIES] + SCALE[TARGET_SERIES] * (
    LOADINGS[TARGET_SERIES] * a_sm_new[TARGET_PERIOD, 0]
)
total_revision = new_nowcast - old_nowcast

EPS = 1e-4
contributions = []
for (t, i) in J:
    # Old-vintage Kalman forecast of the cell (common-component projection).
    forecast = CENTER[i] + SCALE[i] * (LOADINGS[i] * a_sm_old[t, 0])
    actual = new_vintage[t, i]
    news = actual - forecast

    # Finite-difference weight: perturb the RAW new-vintage observation and
    # re-run the whole smoother (central difference).  Independent of any
    # analytic impulse-response argument.
    up = new_vintage.copy()
    up[t, i] += EPS
    dn = new_vintage.copy()
    dn[t, i] -= EPS
    g_up = target_nowcast(up, TARGET_SERIES, TARGET_PERIOD)
    g_dn = target_nowcast(dn, TARGET_SERIES, TARGET_PERIOD)
    weight = (g_up - g_dn) / (2.0 * EPS)

    contributions.append(
        {
            "series": i,
            "period": t,
            "actual": float(actual),
            "forecast": float(forecast),
            "news": float(news),
            "weight": float(weight),
            "contribution": float(weight * news),
        }
    )

# Internal check: the exact adding-up identity holds for the reference itself.
contrib_sum = sum(c["contribution"] for c in contributions)
assert abs(contrib_sum - total_revision) < 1e-9, (contrib_sum, total_revision)


# --------------------------------------------------------------------------
# 5. Serialise (NaN -> null; the Rust loader maps null back to NaN-for-missing).
# --------------------------------------------------------------------------
def with_nulls(a):
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in a]


doc = {
    "_doc": (
        "Golden for the tsecon-nowcast Banbura-Modugno NEWS decomposition. The "
        "reference is an INDEPENDENT NumPy multivariate Kalman filter + RTS "
        "smoother (see generator docstring). 'weight' is computed by FINITE "
        "DIFFERENCES (central, eps=1e-4) on that smoother -- the direct "
        "definition of d(nowcast)/d(obs_j) -- so matching the crate's ANALYTIC "
        "impulse-response weight to ~1e-6 is a genuine cross-check. The exact "
        "adding-up identity new_nowcast - old_nowcast == sum(contribution) is "
        "self-validating and asserted to ~1e-10."
    ),
    "dims": {"n_series": N, "n_obs": T, "factor_order": P, "k_factors": r},
    "params": {
        "loadings": LOADINGS.tolist(),
        "phi": PHI.tolist(),
        "idiosyncratic": IDIO.tolist(),
        "factor_innovation_var": Q,
    },
    "center": CENTER.tolist(),
    "scale": SCALE.tolist(),
    "old_vintage": with_nulls(old_vintage),
    "new_vintage": with_nulls(new_vintage),
    "target_series": TARGET_SERIES,
    "target_period": TARGET_PERIOD,
    "fd_epsilon": EPS,
    "golden": {
        "old_nowcast": float(old_nowcast),
        "new_nowcast": float(new_nowcast),
        "total_revision": float(total_revision),
        "contributions": contributions,
    },
}

with open(OUT, "w") as fh:
    json.dump(doc, fh, indent=2)

print(f"wrote {OUT}")
print(f"  N={N} T={T} P={P}  |J|={len(J)}  target=(series {TARGET_SERIES}, period {TARGET_PERIOD})")
print(f"  old_nowcast = {old_nowcast!r}")
print(f"  new_nowcast = {new_nowcast!r}")
print(f"  total_revision   = {total_revision!r}")
print(f"  sum contribution = {contrib_sum!r}")
