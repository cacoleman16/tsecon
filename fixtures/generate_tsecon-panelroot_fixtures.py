"""Golden fixtures for the panel unit-root tests (LLC, IPS, Fisher).

Reference implementations (this venv, NEVER importing tsecon):
  * per-unit ADF  -> statsmodels.tsa.stattools.adfuller (0.14.x), which the
      library's `tsecon_diag::adf` reproduces to ~1e-8. Every combined
      statistic is built from these per-unit tau/p-value/lag/nobs outputs.
  * Fisher        -> exact arithmetic on the per-unit p-values:
      P = -2 sum ln(p_i) ~ chi2(2N)          (scipy.stats.chi2.sf, right tail)
      Z = sum(Phi^{-1}(p_i))/sqrt(N) ~ N(0,1) (scipy.stats.norm, left tail)
      This is the STRONG sub-golden: every piece runs here and matches tsecon
      exactly, since tsecon's per-unit ADF == statsmodels.
  * IPS Wtbar     -> t_bar standardized with the transcribed Im-Pesaran-Shin
      (2003) Table 3 moments (IPS_E_*/IPS_V_*), keyed by (nobs, lag, case).
  * LLC t*_delta  -> a full NumPy reimplementation of the six-step
      Levin-Lin-Chu (2002) pipeline (auxiliary OLS via lstsq, Bartlett LRV
      with the 3.21 T^{1/3} bandwidth rule, pooled OLS, bias adjustment from
      the transcribed LLC (2002) Table 2, LLC_MU/LLC_SIGMA).

The transcribed tables below are byte-identical to the crate's `tables.rs`
and to the R package `plm`'s `adj.ips.wtbar` / `adj.levinlin` internal data.
The pipeline conventions (per-unit df-corrected sigma, pooled df-corrected
SE, mu*/sigma* keyed by the full T) match `plm::purtest(test = ...)`, whose
`Wtbar`, `levinlin`, `madwu`, and `invnormal` statistics reproduce the values
stored in `plm_anchor` below to floating-point precision (verified once, out
of band, against R; not a regeneration dependency).

Doubles are written with json's shortest round-trip repr, which the Rust
golden parses to identical bits (serde_json `float_roundtrip`).

Run:  python fixtures/generate_tsecon-panelroot_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import scipy.stats as st
from statsmodels.tsa.stattools import adfuller

OUT = Path(__file__).resolve().parent / "tsecon-panelroot.json"
CLAMP_EPS = 1e-16

# ---------------------------------------------------------------- tables
# Im-Pesaran-Shin (2003) Table 3; rows = lag 0..8, cols = T grid.
IPS_T = [10, 15, 20, 25, 30, 40, 50, 60, 70, 100]
NA = float("nan")
IPS_E_C = [
    [-1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532],
    [-1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530],
    [-1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514],
    [-1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512],
    [-1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495],
    [NA, NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494],
    [NA, NA, NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476],
    [NA, NA, NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474],
    [NA, NA, NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456],
]
IPS_V_C = [
    [1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735],
    [1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745],
    [1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754],
    [1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761],
    [2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771],
    [NA, NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781],
    [NA, NA, NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795],
    [NA, NA, NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806],
    [NA, NA, NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818],
]
IPS_E_CT = [
    [-2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177],
    [-2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179],
    [-1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158],
    [-1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158],
    [-1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135],
    [NA, NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135],
    [NA, NA, NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113],
    [NA, NA, NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112],
    [NA, NA, NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088],
]
IPS_V_CT = [
    [1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597],
    [1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605],
    [1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613],
    [2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625],
    [3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629],
    [NA, NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638],
    [NA, NA, NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650],
    [NA, NA, NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661],
    [NA, NA, NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670],
]
LLC_T = [25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250, 500]
LLC_MU = {
    "n": [0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "c": [-0.554, -0.546, -0.541, -0.537, -0.533, -0.531, -0.527, -0.524, -0.521, -0.520, -0.518, -0.509, -0.500],
    "ct": [-0.703, -0.674, -0.653, -0.637, -0.624, -0.614, -0.598, -0.587, -0.578, -0.571, -0.566, -0.533, -0.500],
}
LLC_SIGMA = {
    "n": [1.049, 1.035, 1.027, 1.021, 1.017, 1.014, 1.011, 1.008, 1.007, 1.006, 1.005, 1.001, 1.000],
    "c": [0.919, 0.889, 0.867, 0.850, 0.837, 0.826, 0.810, 0.798, 0.789, 0.782, 0.776, 0.742, 0.707],
    "ct": [1.003, 0.949, 0.906, 0.871, 0.842, 0.818, 0.780, 0.751, 0.728, 0.710, 0.695, 0.603, 0.500],
}


def _clamp_interp(x, xs, ys):
    xs = list(xs)
    ys = list(ys)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x < xs[i]:
            w = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + w * (ys[i] - ys[i - 1])
    return ys[-1]


def _interp_row(l, row):
    xs = [t for t, v in zip(IPS_T, row) if not math.isnan(v)]
    ys = [v for v in row if not math.isnan(v)]
    return _clamp_interp(l, xs, ys)


def ips_moments(l, p, trend):
    p = min(p, 8)
    e_tab, v_tab = (IPS_E_CT, IPS_V_CT) if trend else (IPS_E_C, IPS_V_C)
    return _interp_row(l, e_tab[p]), _interp_row(l, v_tab[p])


def llc_adj(t, case):
    return _clamp_interp(t, LLC_T, LLC_MU[case]), _clamp_interp(t, LLC_T, LLC_SIGMA[case])


# ------------------------------------------------------------- per unit
def adf_unit(y, regression, lag_mode, lag, max_lags):
    if lag_mode == "fixed":
        r = adfuller(y, regression=regression, maxlag=lag, autolag=None)
    else:
        r = adfuller(y, regression=regression, autolag=lag_mode.upper(), maxlag=max_lags)
    tau, pval, usedlag, nobs = r[0], r[1], r[2], r[3]
    return float(tau), float(pval), int(usedlag), int(nobs)


# ---------------------------------------------------------------- LLC
def _ntrend(case):
    return {"n": 0, "c": 1, "ct": 2}[case]


def _determ(case, rows):
    cols = []
    if _ntrend(case) >= 1:
        cols.append(np.ones(rows))
    if _ntrend(case) >= 2:
        cols.append(np.arange(1, rows + 1, dtype=float))
    return cols


def _rround(x):
    f = math.floor(x)
    d = x - f
    if abs(d - 0.5) < 1e-9:
        return float(f if f % 2 == 0 else f + 1)
    return float(round(x))


def _bartlett_lrv(dx, case, q):
    n = len(dx)
    if case == "c":
        dxx = dx - dx.mean()
    elif case == "ct":
        X = np.column_stack([np.ones(n), np.arange(1, n + 1, dtype=float)])
        b, *_ = np.linalg.lstsq(X, dx, rcond=None)
        dxx = dx - X @ b
    else:
        dxx = dx.copy()
    g0 = np.sum(dxx * dxx) / n
    s = g0
    for L in range(1, n):
        w = 1.0 - L / (q + 1.0)
        if w <= 0:
            break
        s += 2 * w * np.sum(dxx[L:] * dxx[:-L]) / n
    return s


def llc_stat(arr, case, lags_per_unit):
    N, T = arr.shape
    nt = _ntrend(case)
    etil_all, vtil_all, s_list, rows_list = [], [], [], []
    for i in range(N):
        y = arr[i]
        ell = lags_per_unit[i]
        rows = T - ell - 1
        t0 = ell + 1
        dy = y[t0:] - y[t0 - 1:T - 1]
        ly1 = y[t0 - 1:T - 1]
        dylags = [(y[t0 - j:T - j] - y[t0 - j - 1:T - j - 1]).astype(float) for j in range(1, ell + 1)]
        full = np.column_stack([ly1] + dylags + _determ(case, rows))
        bf, *_ = np.linalg.lstsq(full, dy, rcond=None)
        res = dy - full @ bf
        K = 1 + ell + nt
        sig = math.sqrt(np.sum(res * res) / (rows - K))
        aux_cols = _determ(case, rows) + dylags
        if aux_cols:
            D = np.column_stack(aux_cols)
            be, *_ = np.linalg.lstsq(D, dy, rcond=None)
            ehat = dy - D @ be
            bv, *_ = np.linalg.lstsq(D, ly1, rcond=None)
            vhat = ly1 - D @ bv
        else:
            ehat, vhat = dy.copy(), ly1.copy()
        etil_all.append(ehat / sig)
        vtil_all.append(vhat / sig)
        dyf = y[1:] - y[:-1]
        q = _rround(3.21 * T ** (1.0 / 3.0))
        s_list.append(math.sqrt(_bartlett_lrv(dyf, case, q)) / sig)
        rows_list.append(rows)
    etil = np.concatenate(etil_all)
    vtil = np.concatenate(vtil_all)
    sum_vv = float(np.sum(vtil * vtil))
    delta = float(np.sum(vtil * etil) / sum_vv)
    rss = float(np.sum((etil - delta * vtil) ** 2))
    npool = sum(rows_list)
    tildeT = npool / N
    sig_etil2 = rss / npool
    sd = math.sqrt(rss / (npool - 1)) / math.sqrt(sum_vv)
    tdelta = delta / sd
    sn = float(np.mean(s_list))
    mu, sg = llc_adj(T, case)
    tstar = (tdelta - npool * sn / sig_etil2 * sd * mu) / sg
    return {
        "t_star": tstar,
        "p_value": float(st.norm.cdf(tstar)),
        "delta_hat": delta,
        "t_delta": tdelta,
        "s_n": sn,
        "t_bar_periods": tildeT,
    }


# ---------------------------------------------------------------- panels
def rw_panel(seed, N, T):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((N, T)), axis=1)


def ar1_panel(seed, N, T, rho):
    rng = np.random.default_rng(seed)
    y = np.zeros((N, T))
    e = rng.standard_normal((N, T))
    for t in range(1, T):
        y[:, t] = rho * y[:, t - 1] + e[:, t]
    return y


def build_case(name, arr, regression, lag_mode, lag=None, max_lags=None):
    N, T = arr.shape
    per_t, per_p, per_l, per_n = [], [], [], []
    for i in range(N):
        tau, pval, usedlag, nobs = adf_unit(arr[i], regression, lag_mode, lag, max_lags)
        per_t.append(tau)
        per_p.append(pval)
        per_l.append(usedlag)
        per_n.append(nobs)
    per_t = np.array(per_t)
    clamped = np.clip(np.array(per_p), CLAMP_EPS, 1 - CLAMP_EPS)
    P = float(-2.0 * np.sum(np.log(clamped)))
    Z = float(np.sum(st.norm.ppf(clamped)) / math.sqrt(N))
    case = {
        "name": name,
        "N": N,
        "T": T,
        "regression": regression,
        "lag_mode": lag_mode,
        "lag": lag,
        "max_lags": max_lags,
        "data": [arr[i].tolist() for i in range(N)],
        "per_unit": {
            "tstat": per_t.tolist(),
            "pvalue": per_p,
            "lags": per_l,
            "nobs": per_n,
        },
        "fisher": {
            "maddala_wu": P,
            "mw_pvalue": float(st.chi2.sf(P, 2 * N)),
            "choi_z": Z,
            "choi_z_pvalue": float(st.norm.cdf(Z)),
            "clamped_pvalue": clamped.tolist(),
        },
    }
    if regression in ("c", "ct"):
        trend = regression == "ct"
        e_bar = np.mean([ips_moments(per_n[i], per_l[i], trend)[0] for i in range(N)])
        v_bar = np.mean([ips_moments(per_n[i], per_l[i], trend)[1] for i in range(N)])
        tbar = float(per_t.mean())
        wtbar = float(math.sqrt(N) * (tbar - e_bar) / math.sqrt(v_bar))
        case["ips"] = {"t_bar": tbar, "w_tbar": wtbar, "p_value": float(st.norm.cdf(wtbar))}
    # LLC only for balanced panels (all our N x T arrays are balanced).
    case["llc"] = llc_stat(arr, regression, per_l)
    return case


def build_unbalanced_case(name, series_list, regression, lag, max_lags):
    """A ragged panel for the IPS/Fisher path (LLC omitted)."""
    per_t, per_p, per_l, per_n = [], [], [], []
    for y in series_list:
        tau, pval, usedlag, nobs = adf_unit(np.asarray(y, float), regression, "fixed", lag, max_lags)
        per_t.append(tau)
        per_p.append(pval)
        per_l.append(usedlag)
        per_n.append(nobs)
    N = len(series_list)
    per_t = np.array(per_t)
    clamped = np.clip(np.array(per_p), CLAMP_EPS, 1 - CLAMP_EPS)
    P = float(-2.0 * np.sum(np.log(clamped)))
    Z = float(np.sum(st.norm.ppf(clamped)) / math.sqrt(N))
    trend = regression == "ct"
    e_bar = np.mean([ips_moments(per_n[i], per_l[i], trend)[0] for i in range(N)])
    v_bar = np.mean([ips_moments(per_n[i], per_l[i], trend)[1] for i in range(N)])
    tbar = float(per_t.mean())
    wtbar = float(math.sqrt(N) * (tbar - e_bar) / math.sqrt(v_bar))
    return {
        "name": name,
        "N": N,
        "regression": regression,
        "lag_mode": "fixed",
        "lag": lag,
        "max_lags": max_lags,
        "data": [list(map(float, y)) for y in series_list],
        "per_unit": {"tstat": per_t.tolist(), "pvalue": per_p, "lags": per_l, "nobs": per_n},
        "fisher": {
            "maddala_wu": P,
            "mw_pvalue": float(st.chi2.sf(P, 2 * N)),
            "choi_z": Z,
            "choi_z_pvalue": float(st.norm.cdf(Z)),
            "clamped_pvalue": clamped.tolist(),
        },
        "ips": {"t_bar": tbar, "w_tbar": wtbar, "p_value": float(st.norm.cdf(wtbar))},
    }


def main():
    cases = []
    # Random-walk panels (expect non-rejection) across N, T, regression, lag.
    rw = rw_panel(12345, 6, 50)
    for reg in ("n", "c", "ct"):
        cases.append(build_case(f"rw_N6_T50_{reg}_L1", rw, reg, "fixed", lag=1))
    cases.append(build_case("rw_N6_T50_c_L0", rw, "c", "fixed", lag=0))
    cases.append(build_case("rw_N6_T50_ct_L2", rw, "ct", "fixed", lag=2))
    cases.append(build_case("rw_N10_T100_c_aic", rw_panel(7, 10, 100), "c", "aic"))
    cases.append(build_case("rw_N10_T100_ct_aic", rw_panel(7, 10, 100), "ct", "aic"))
    # Stationary AR(1) panels (expect rejection).
    ar = ar1_panel(999, 8, 60, 0.5)
    for reg in ("c", "ct"):
        cases.append(build_case(f"ar1_N8_T60_{reg}_L1", ar, reg, "fixed", lag=1))
    cases.append(build_case("ar1_N20_T80_c_aic", ar1_panel(2024, 20, 80, 0.4), "c", "aic"))
    cases.append(build_case("ar1_N5_T50_n_L1", ar1_panel(55, 5, 50, 0.5), "n", "fixed", lag=1))

    # An unbalanced panel for the IPS/Fisher path.
    rng = np.random.default_rng(4242)
    lens = [45, 60, 55, 70, 50]
    ragged = [np.cumsum(rng.standard_normal(L)) for L in lens]
    unbalanced = [build_unbalanced_case("unbal_c_L1", ragged, "c", 1, None)]

    # A handful of plm::purtest reference values, transcribed from R
    # (plm 2.6-x) for the rw_N6_T50 panel with lags = 1, dfcor = TRUE. These
    # anchor the pipeline to an independent, published implementation; they
    # are constants, not recomputed here (R is not a regeneration dependency).
    plm_anchor = [
        {"case": "rw_N6_T50_c_L1", "ips_wtbar": 0.962606, "ips_p": 0.832127282,
         "madwu": 5.579775, "madwu_p": 0.935768111, "choi_z": 1.123082,
         "choi_p": 0.869298694, "llc_tstar": 0.566739, "tol": 1e-4},
        {"case": "rw_N6_T50_ct_L1", "ips_wtbar": 1.518904, "ips_p": 0.935606724,
         "madwu": 8.168919, "madwu_p": 0.771794462, "choi_z": 1.674405,
         "choi_p": 0.952974489, "llc_tstar": 0.812422, "tol": 1e-4},
        {"case": "rw_N6_T50_n_L1", "madwu": 14.248597, "madwu_p": 0.285115892,
         "choi_z": -0.923030, "choi_p": 0.17799571, "llc_tstar": -1.401989,
         "tol": 1e-4},
    ]

    fixture = {
        "meta": {
            "note": "Panel unit-root goldens. Per-unit ADF via statsmodels "
            "adfuller (== tsecon_diag::adf). Fisher is exact arithmetic on the "
            "p-values (strong sub-golden). IPS/LLC use the transcribed IPS 2003 "
            "Table 3 and LLC 2002 Table 2, matching plm::purtest conventions.",
            "clamp_eps": CLAMP_EPS,
        },
        "cases": cases + unbalanced,
        "plm_anchor": plm_anchor,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases) + len(unbalanced)} cases")


if __name__ == "__main__":
    main()
