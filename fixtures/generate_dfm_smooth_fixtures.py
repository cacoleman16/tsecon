"""Golden fixtures for `dfm_smooth`, the fixed-parameter dynamic-factor Kalman
smoother (crate `tsecon-nowcast`: `smooth_fixed` and `smooth_standardized`).

WHAT IS PINNED
--------------
For r factors following a VAR(p) and white-noise idiosyncratic errors,

    y_t = Lambda F_t + e_t,                e_t ~ N(0, diag(idiosyncratic))
    F_t = A_1 F_{t-1} + ... + A_p F_{t-p} + eta_t,   eta_t ~ N(0, factor_cov)

in the stacked state alpha_t = [F_t', ..., F_{t-p+1}']' (m = r p): Z = [Lambda|0],
T = [A_1 ... A_p] over identity sub-blocks, R = [I_r; 0], Q = factor_cov,
H = diag(idiosyncratic), stationary (Lyapunov) initialization. At FIXED
parameters the Kalman filter/smoother output is a deterministic function of the
panel, so it is pinned against an independent package:

  * Case `dynamic_factor_r2_p2` — statsmodels `DynamicFactor(Y, k_factors=2,
    factor_order=2, error_order=0).smooth(params)` on a ragged panel (NaN at the
    edge and one interior gap): `llf`, `smoothed_state` (T x m) and
    `smoothed_state_cov` (T x m x m). statsmodels normalizes Q = I. Parameter
    order (asserted against `param_names`): the loadings row-major
    (loading.f1.y1, loading.f2.y1, ...), then sigma2.y1.., then the factor VAR
    per equation, row i of [A_1 | A_2].
  * Case `kalman_smoother_q` — Q != I, which DynamicFactor cannot express:
    statsmodels' generic `KalmanSmoother` with the same system matrices and
    `initialize_stationary()`, on the same ragged panel. A canary asserts the
    generic smoother with Q = I reproduces the DynamicFactor case, so both legs
    are one reference.
  * Case `standardized_r1_p3` — the standardization and the edge nowcast that
    `dfm_smooth(..., center=, scale=)` adds: a raw panel X, moments c and s, the
    standardized panel z = (X - c) / s computed elementwise in IEEE double (the
    same two operations the crate performs, so z is bit-identical), the generic
    KalmanSmoother on z (r = 1, p = 3, Q = 0.8), the smoothed factor at the last
    period (edge_factor) and nowcast = c + s * (Lambda @ edge_factor). The
    log-likelihood is that of z (no Jacobian term).

statsmodels' filter is the multivariate (or its own univariate) Kalman filter
and tsecon's is the univariate sequential filter; for diagonal H they agree to
roundoff, not bitwise (the agreement achieved is recorded by the Rust test).

SINGULAR OBSERVATION DENSITIES (`singular_cases`)
-------------------------------------------------
`idiosyncratic >= 0` admits exact zeros. When more than r series with zero
idiosyncratic variance are observed in the same period, one of them pins the
factor direction the others see and the others are deterministic functions of
it: the joint density of y_t is singular. The value pinned is the Gaussian
density on its support — the log-likelihood with the redundant cell set to NaN
(the element is skipped like a missing value; the Kalman filter's own rule for
a numerically singular element). Which series pins depends on the column
order, so two orders differ by the Jacobian n ln|lam_j / lam_i|. Each case
stores the full panel (what the smoother is called on), the redundant column
of each order, and two independent references for the reduced panel:

  * statsmodels' generic `KalmanSmoother` (initialize_stationary) on the panel
    with the redundant column set to NaN: llf, smoothed_state and
    smoothed_state_cov; with a canary that `DynamicFactor` (Q normalized to I,
    loadings scaled by sqrt(Q)) gives the same llf;
  * an exact reference: the same reduced-panel Kalman filter run in exact
    rational arithmetic (`fractions.Fraction`, every float64 input converted
    exactly, the stationary covariance from the exact Lyapunov solve) where it
    stays small (the two-period case), and in 80-digit `decimal` arithmetic
    over the T = 100 panel, with ln F evaluated at that precision.

statsmodels itself cannot serve as the oracle on the FULL panel (its filter
takes the dust update too: 1366.07 against -293.15 on the T = 100 panel), which
is why the full-panel values are not stored.

  * `zero_idio_two_periods` — the audit's repro: three series, the second and
    third with zero idiosyncratic variance, one observed period (the second
    all NaN), r = 1, p = 2, Q = 0.133.
  * `zero_idio_pair_t100` — T = 100, one VAR(2) factor, four series, series 1
    and 2 with zero idiosyncratic variance and observed every period, under
    the column orders listed in `orders`.

Data: seeded NumPy `default_rng` draws through the DGPs below (no dataset).
NaN (missing) cells are written as JSON null.

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr, which the Rust golden tests parse to identical bits
(serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_dfm_smooth_fixtures.py
"""

from __future__ import annotations

import decimal
import itertools
import json
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas
import scipy
import statsmodels
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother

HERE = Path(__file__).resolve().parent
OUT = HERE / "dfm_smooth.json"
SEED = 20261009


def nan_to_none(a: np.ndarray) -> list:
    """Nested list with NaN -> None (JSON null)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return [None if np.isnan(v) else float(v) for v in a]
    return [nan_to_none(row) for row in a]


def system(lam: np.ndarray, ar: np.ndarray, q: np.ndarray, idio: np.ndarray):
    n, r = lam.shape
    m = ar.shape[1]
    t_mat = np.zeros((m, m))
    t_mat[:r] = ar
    t_mat[r:, : m - r] = np.eye(m - r)
    z_mat = np.zeros((n, m))
    z_mat[:, :r] = lam
    r_mat = np.zeros((m, r))
    r_mat[:r] = np.eye(r)
    return z_mat, t_mat, r_mat, q, np.diag(idio)


def generic_smooth(y: np.ndarray, lam, ar, q, idio):
    n, r = lam.shape
    m = ar.shape[1]
    z_mat, t_mat, r_mat, q_mat, h_mat = system(lam, ar, q, idio)
    ks = KalmanSmoother(k_endog=n, k_states=m, k_posdef=r)
    ks.bind(np.ascontiguousarray(y))
    ks["design"] = z_mat
    ks["obs_cov"] = h_mat
    ks["transition"] = t_mat
    ks["selection"] = r_mat
    ks["state_cov"] = q_mat
    ks.initialize_stationary()
    out = ks.smooth()
    return (
        float(np.sum(out.llf_obs)),
        np.asarray(out.smoothed_state).T.copy(),
        np.moveaxis(np.asarray(out.smoothed_state_cov), 2, 0).copy(),
    )


def simulate(rng, n, t, lam, ar, q, idio, burn=200):
    r = lam.shape[1]
    p = ar.shape[1] // r
    chol = np.linalg.cholesky(q)
    f = np.zeros((t + burn, r))
    for s in range(p, t + burn):
        acc = chol @ rng.standard_normal(r)
        for lag in range(p):
            acc = acc + ar[:, lag * r:(lag + 1) * r] @ f[s - 1 - lag]
        f[s] = acc
    f = f[burn:]
    return f @ lam.T + rng.standard_normal((t, n)) * np.sqrt(idio)


def case_dynamic_factor(rng):
    n, t, r, p = 6, 150, 2, 2
    a1 = np.array([[0.5, 0.1], [0.0, 0.4]])
    a2 = np.array([[0.2, 0.0], [0.05, 0.1]])
    ar = np.c_[a1, a2]
    lam = rng.normal(size=(n, r))
    idio = rng.uniform(0.2, 0.6, n)
    y = simulate(rng, n, t, lam, ar, np.eye(r), idio)
    # Ragged edge: the last period half released, one series a period behind,
    # plus one interior gap (a skipped release).
    y[-1, :3] = np.nan
    y[-2, 0] = np.nan
    y[70, 4] = np.nan

    mod = DynamicFactor(y, k_factors=r, factor_order=p, error_order=0)
    names = list(mod.param_names)
    expected_names = (
        [f"loading.f{k + 1}.y{i + 1}" for i in range(n) for k in range(r)]
        + [f"sigma2.y{i + 1}" for i in range(n)]
        + [f"L{lag + 1}.f{j + 1}.f{i + 1}" for i in range(r) for lag in range(p) for j in range(r)]
    )
    assert names == expected_names, names
    params = np.r_[lam.ravel(), idio, ar.ravel()]
    res = mod.smooth(params, transformed=True)
    llf = float(res.llf)
    state = np.asarray(res.smoothed_state).T.copy()
    state_cov = np.moveaxis(np.asarray(res.smoothed_state_cov), 2, 0).copy()
    # The DynamicFactor transition is exactly the documented companion layout.
    assert np.array_equal(np.asarray(mod.ssm["transition"])[:r], ar)

    # Canary: the generic KalmanSmoother at Q = I is the same reference.
    llf_g, state_g, cov_g = generic_smooth(y, lam, ar, np.eye(r), idio)
    canary = {
        "llf_abs_diff": abs(llf_g - llf),
        "state_max_abs_diff": float(np.max(np.abs(state_g - state))),
        "state_cov_max_abs_diff": float(np.max(np.abs(cov_g - state_cov))),
    }
    assert canary["llf_abs_diff"] < 1e-9 and canary["state_max_abs_diff"] < 1e-10, canary
    return {
        "name": "dynamic_factor_r2_p2",
        "source": "statsmodels DynamicFactor(k_factors=2, factor_order=2, error_order=0).smooth(params)",
        "n_factors": r,
        "factor_order": p,
        "panel": nan_to_none(y),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": np.eye(r).tolist(),
        "idiosyncratic": idio.tolist(),
        "param_names": names,
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": state_cov.tolist(),
        "generic_vs_dynamic_factor": canary,
    }, y, lam, ar, idio


def case_generic_q(y, lam, ar, idio):
    q = np.array([[1.3, 0.4], [0.4, 0.7]])
    llf, state, cov = generic_smooth(y, lam, ar, q, idio)
    return {
        "name": "kalman_smoother_q",
        "source": "statsmodels KalmanSmoother (generic), initialize_stationary(), Q != I",
        "n_factors": lam.shape[1],
        "factor_order": ar.shape[1] // lam.shape[1],
        "panel": nan_to_none(y),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": cov.tolist(),
    }


def case_standardized(rng):
    n, t, r, p = 5, 120, 1, 3
    ar = np.array([[0.6, -0.2, 0.15]])
    q = np.array([[0.8]])
    lam = rng.uniform(0.4, 1.2, size=(n, r))
    idio = rng.uniform(0.3, 0.7, n)
    z_true = simulate(rng, n, t, lam, ar, q, idio)
    center = np.array([2.5, -1.0, 103.0, 0.04, 12.0])
    scale = np.array([0.7, 3.1, 12.5, 0.011, 1.9])
    x = center + scale * z_true
    # A staggered ragged edge: the series are released one to three periods late.
    x[-1, [0, 2, 3]] = np.nan
    x[-2, [0, 3]] = np.nan
    x[-3, 3] = np.nan
    z = (x - center) / scale  # the crate's exact operations, elementwise
    llf, state, cov = generic_smooth(z, lam, ar, q, idio)
    edge = state[-1, :r]
    nowcast = center + scale * (lam @ edge)
    return {
        "name": "standardized_r1_p3",
        "source": "statsmodels KalmanSmoother (generic) on z = (X - center) / scale; "
                  "nowcast = center + scale * (loadings @ smoothed_factor[T-1])",
        "n_factors": r,
        "factor_order": p,
        "panel": nan_to_none(x),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "loglik": llf,
        "smoothed_state": state.tolist(),
        "smoothed_state_cov": cov.tolist(),
        "edge_factor": edge.tolist(),
        "nowcast": nowcast.tolist(),
    }


# ---------------------------------------------------------------------------
# Singular observation densities: an exact reduced-panel Kalman filter.
# ---------------------------------------------------------------------------


def exact_stationary_cov(t_mat, rqr, num):
    """Solve P = T P T' + RQR' exactly in the number type `num` (Fraction or
    Decimal), by Gauss-Jordan elimination on the vec'd m^2 x m^2 system."""
    m = len(t_mat)
    t_q = [[num(v) for v in row] for row in t_mat]
    k = m * m
    # vec(P) row-major: index i*m + j. (I - T (x) T) vec(P) = vec(RQR').
    a = [[num(0)] * k for _ in range(k)]
    b = [num(rqr[i][j]) for i in range(m) for j in range(m)]
    for i in range(m):
        for j in range(m):
            r = i * m + j
            a[r][r] += num(1)
            for kk in range(m):
                for ll in range(m):
                    a[r][kk * m + ll] -= t_q[i][kk] * t_q[j][ll]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        for r in range(k):
            if r != col and a[r][col] != 0:
                f = a[r][col] / a[col][col]
                for c in range(col, k):
                    a[r][c] -= f * a[col][c]
                b[r] -= f * b[col]
    vec = [b[r] / a[r][r] for r in range(k)]
    return [[vec[i * m + j] for j in range(m)] for i in range(m)]


def exact_loglik(y, lam, ar, q, idio, num):
    """Univariate Kalman filter of the DFM in the number type `num` (Fraction:
    exact; Decimal: the context precision), NaN cells skipped. The rational
    part sum v^2 / F is accumulated in `num`; ln 2pi and ln F are evaluated in
    80-digit decimal arithmetic. Every element left in `y` must have F > 0
    (the redundant cells are NaN). Returns a Decimal."""
    n, r = lam.shape
    m = ar.shape[1]
    z_mat, t_mat, r_mat, q_mat, _ = system(lam, ar, q, idio)
    rqr = r_mat @ q_mat @ r_mat.T  # exact: R is a 0/1 selector
    t_q = [[num(v) for v in row] for row in t_mat]
    z_q = [[num(v) for v in row] for row in z_mat]
    h_q = [num(v) for v in idio]
    p = exact_stationary_cov(t_mat, rqr.tolist(), num)
    a = [num(0)] * m
    quad = num(0)
    log_f = []
    for t in range(y.shape[0]):
        for i in range(n):
            if np.isnan(y[t, i]):
                continue
            zi = z_q[i]
            v = num(y[t, i]) - sum(zi[j] * a[j] for j in range(m))
            mvec = [sum(p[j][k] * zi[k] for k in range(m)) for j in range(m)]
            f = sum(zi[j] * mvec[j] for j in range(m)) + h_q[i]
            assert f > 0, (t, i, f)
            quad += v * v / f
            log_f.append(f)
            a = [a[j] + mvec[j] * v / f for j in range(m)]
            p = [[p[j][k] - mvec[j] * mvec[k] / f for k in range(m)] for j in range(m)]
        a = [sum(t_q[j][k] * a[k] for k in range(m)) for j in range(m)]
        tp = [[sum(t_q[j][k] * p[k][l] for k in range(m)) for l in range(m)] for j in range(m)]
        p = [
            [sum(tp[j][k] * t_q[l][k] for k in range(m)) + num(rqr[j][l]) for l in range(m)]
            for j in range(m)
        ]
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        two_pi = 2 * decimal.Decimal(PI_80)
        total = len(log_f) * two_pi.ln() + sum(to_dec(f).ln() for f in log_f) + to_dec(quad)
        return -total / 2


PI_80 = (
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"
)


def to_dec(x):
    """A Fraction or Decimal as a Decimal at the current context precision."""
    if isinstance(x, Fraction):
        return decimal.Decimal(x.numerator) / decimal.Decimal(x.denominator)
    return +x


def singular_reference(y_full, redundant, lam, ar, q, idio):
    """statsmodels on the panel with column `redundant` set to NaN, plus the
    DynamicFactor canary."""
    yr = y_full.copy()
    yr[:, redundant] = np.nan
    llf, state, cov = generic_smooth(yr, lam, ar, q, idio)
    r = lam.shape[1]
    assert r == 1, "the DynamicFactor canary below is written for one factor"
    sq = np.sqrt(q[0, 0])
    dfm = DynamicFactor(yr, k_factors=1, factor_order=ar.shape[1], error_order=0)
    params = np.r_[(lam * sq).ravel(), idio, ar.ravel()]
    llf_dfm = float(dfm.smooth(params, transformed=True).llf)
    assert abs(llf_dfm - llf) < 1e-9 * max(1.0, abs(llf)), (llf_dfm, llf)
    return yr, llf, state, cov, abs(llf_dfm - llf)


def case_zero_idio_two_periods():
    z = np.array([
        [0.03666137558761732, 0.5527422219007221, -0.04042486581736605],
        [np.nan, np.nan, np.nan],
    ])
    lam = np.array([[-0.43272912833939525], [0.7196862105052038], [-0.05263433357820673]])
    ar = np.array([[-0.8642299294783886, -0.8484056219079968]])
    q = np.array([[0.13302995088165312]])
    idio = np.array([0.8355556816599208, 0.0, 0.0])
    redundant = 2  # series 1 is filtered first and pins the factor
    yr, llf, state, cov, canary = singular_reference(z, redundant, lam, ar, q, idio)
    exact_fraction = exact_loglik(yr, lam, ar, q, idio, Fraction)
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        exact_decimal = exact_loglik(yr, lam, ar, q, idio, decimal.Decimal)
        gap = abs(exact_fraction - exact_decimal)
    assert gap < decimal.Decimal("1e-60"), gap
    return {
        "name": "zero_idio_two_periods",
        "source": "statsmodels KalmanSmoother (generic) on the panel with the redundant "
                  "column NaN; exact reference: the same Kalman filter in rational "
                  "arithmetic (ln F at 80 digits)",
        "n_factors": 1,
        "factor_order": 2,
        "panel": nan_to_none(z),
        "loadings": lam.tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "orders": [
            {
                "order": [0, 1, 2],
                "redundant": redundant,
                "loglik": llf,
                "loglik_exact": float(exact_fraction),
                "loglik_exact_str": str(exact_fraction),
                "smoothed_state": state.tolist(),
                "smoothed_state_cov": cov.tolist(),
                "dynamic_factor_llf_abs_diff": canary,
            }
        ],
    }


def case_zero_idio_pair_t100():
    # The audit's design (refute_DF16-1/perm.py), regenerated from its seed.
    rng = np.random.default_rng(0)
    t_len = 100
    fac = np.zeros(t_len)
    a1, a2 = 0.5, 0.2
    for t in range(2, t_len):
        fac[t] = a1 * fac[t - 1] + a2 * fac[t - 2] + rng.standard_normal()
    lam = np.array([1.0, 0.7, -1.3, 0.4])
    idio = np.array([0.5, 0.0, 0.0, 0.3])
    y = np.outer(fac, lam) + rng.standard_normal((t_len, 4)) * np.sqrt(idio)
    ar = np.array([[a1, a2]])
    q = np.eye(1)
    orders = []
    # The four orders the unfixed filter got wrong, two it got right, and the
    # two in which the other zero-variance series pins the factor.
    for order in ([0, 1, 2, 3], [0, 1, 3, 2], [0, 3, 1, 2], [0, 3, 2, 1],
                  [1, 0, 2, 3], [2, 0, 1, 3], [3, 2, 1, 0], [1, 2, 3, 0]):
        assert list(order) in [list(o) for o in itertools.permutations(range(4))]
        yo = y[:, order]
        lam_o = lam[order][:, None]
        idio_o = idio[order]
        pos = [order.index(1), order.index(2)]
        redundant = max(pos)  # the zero-variance series filtered second
        yr, llf, state, cov, canary = singular_reference(yo, redundant, lam_o, ar, q, idio_o)
        with decimal.localcontext() as ctx:
            ctx.prec = 80
            exact = exact_loglik(yr, lam_o, ar, q, idio_o, decimal.Decimal)
        orders.append({
            "order": list(order),
            "redundant": redundant,
            "loglik": llf,
            "loglik_exact": float(exact),
            "loglik_exact_str": str(exact),
            "smoothed_state": state.tolist(),
            "smoothed_state_cov": cov.tolist(),
            "dynamic_factor_llf_abs_diff": canary,
        })
    return {
        "name": "zero_idio_pair_t100",
        "source": "statsmodels KalmanSmoother (generic) on the column-permuted panel with "
                  "the redundant column NaN; exact reference: the same Kalman filter in "
                  "80-digit decimal arithmetic",
        "n_factors": 1,
        "factor_order": 2,
        "panel": y.tolist(),
        "loadings": lam[:, None].tolist(),
        "factor_ar": ar.tolist(),
        "factor_cov": q.tolist(),
        "idiosyncratic": idio.tolist(),
        "jacobian_note": "pinned by series 1 minus pinned by series 2 = "
                         "n ln|lam_2 / lam_1| = 100 ln(1.3 / 0.7)",
        "orders": orders,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    a, y, lam, ar, idio = case_dynamic_factor(rng)
    b = case_generic_q(y, lam, ar, idio)
    c = case_standardized(rng)
    singular = [case_zero_idio_two_periods(), case_zero_idio_pair_t100()]
    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "pandas": pandas.__version__,
            "seed": SEED,
            "note": (
                "Fixed-parameter DFM Kalman smoother: statsmodels DynamicFactor.smooth "
                "(Q = I) and the generic KalmanSmoother (Q != I; standardized panel) at "
                "given parameters, stationary initialization. smoothed_state is T x m and "
                "smoothed_state_cov T x m x m (statsmodels' arrays moved to time-first). "
                "NaN cells of the panels are null. loglik is that of the panel the "
                "smoother saw (the standardized z for the standardized case). "
                "singular_cases: zero idiosyncratic variances observed together; the "
                "references are the density on the support (redundant column NaN), from "
                "statsmodels and from an exact (rational / 80-digit decimal) filter."
            ),
        },
        "cases": [a, b, c],
        "singular_cases": singular,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    for case in fixture["cases"]:
        print(f"  {case['name']}: loglik = {case['loglik']!r}")
    print("  generic vs DynamicFactor canary:", a["generic_vs_dynamic_factor"])
    print("  standardized nowcast:", c["nowcast"])
    for case in singular:
        for o in case["orders"]:
            print(
                f"  {case['name']} order {o['order']} (redundant {o['redundant']}): "
                f"loglik {o['loglik']!r}, exact {o['loglik_exact_str'][:24]}, "
                f"|sm - exact| = {abs(o['loglik'] - o['loglik_exact']):.1e}"
            )


if __name__ == "__main__":
    main()
