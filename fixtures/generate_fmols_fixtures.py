"""Golden fixtures for the single-equation cointegrating regressions
`fmols` (Phillips-Hansen 1990 fully modified OLS), `dols` (Stock-Watson
1993 dynamic OLS) and `ccr` (Park 1992 canonical cointegrating regression).

Reference implementation (this venv): arch 8.0.0,
``arch.unitroot.cointegration.{FullyModifiedOLS, DynamicOLS,
CanonicalCointegratingReg}`` — an independent package — with the long-run
covariances of ``arch.covariance.kernel.{Bartlett, Parzen,
QuadraticSpectral}``. Every stored case is arch's own output (params, the
parameter covariance, residuals, R^2, the bandwidth and, for DOLS, the
selected leads/lags), except the two DOCUMENTED deviations below.

Conventions the Rust must reproduce (all arch's):

  * regressors ordered ``[x..., const, trend, quadratic_trend]``
    (``add_trend`` appends ``np.vander`` flipped to increasing powers,
    ``t = 1..n`` over the regression sample);
  * the residual system ``eta = [eta_1[1:], eta_2]`` with ``eta_1`` the
    static OLS residual and ``eta_2`` the first difference of the
    regressors detrended on ``x_trend`` (or, with ``diff=True`` and a
    trend, the differences detrended on the differenced trend);
  * kernel weights over lags ``0..int(bandwidth)`` for Bartlett/Parzen
    (``np.arange(int(bw + 1))``: the window stops at ``floor(bw)`` even
    when the weight there is positive) and all ``T - 1`` lags for
    quadratic spectral; biased ``1/T`` sums about zero
    (``center=False``);
  * arch's automatic bandwidth: ``n = ceil(4 (T/100)^rate)`` pilot lags
    on the unit-weighted column sum ``v = eta @ 1``,
    ``alpha = (sum 2 j^q sig_j / sum 2 sig_j)^2``,
    ``bw = c (alpha T)^(1/(2q+1))``, ceiled under ``force_int``
    (FM-OLS/CCR default ``True``, DOLS default ``False``) and capped at
    ``T - 1``; ``force_int`` also ceils an EXPLICIT bandwidth;
  * DOLS selects ``(lags, leads)`` by ``ln(RSS/nobs) + k c/nobs`` on the
    common sample of the largest candidate, refits on the chosen model's
    own sample; ``"unadjusted"`` covariance ``sigma2_HAC (Z'Z/n)^-1 / n``,
    ``"robust"`` the kernel-HAC sandwich on the scores.

Two blocks are NOT plain arch output and say so in the case record:

  1. ``bandwidth_rule = "andrews"``: arch has no Andrews (1991) AR(1)
     plug-in, so the bandwidth is the DOCUMENTED closed form transcribed
     here (Andrews 1991, eq. 6.4 with unit weights on the same column sum
     ``v`` arch's rule uses; ``tsecon-hac::andrews_bandwidth_ar1``):

        rho      = sum_t v_t v_{t-1} / sum_t v_{t-1}^2
        alpha(1) = 4 rho^2 / ((1 - rho)^2 (1 + rho)^2)      (Bartlett, q = 1)
        alpha(2) = 4 rho^2 / (1 - rho)^4                    (Parzen, QS, q = 2)
        S_T      = c (alpha(q) T)^(1/(2q+1)),  c = 1.1447 / 2.6614 / 1.3221,

     then ``ceil`` under ``force_int`` and ``min(S_T, T - 1)``; the
     estimates are arch's at that bandwidth passed EXPLICITLY (so the
     estimator itself is still arch-pinned; only the bandwidth value is a
     documented-formula golden).
  2. ``ccr`` with ``df_adjust=True``: arch 8.0 computes
     ``scale * omega_11 - omega_12 Omega_22^-1 omega_21`` where its own
     docstring (and its FM-OLS code) scale the whole conditional long-run
     variance ``scale * (omega_11 - ...)``, ``scale = T/(T - k)`` — an
     operator-precedence slip. The stored ``cov``/``se`` are
     ``scale * cov(df_adjust=False)`` (the documented estimator); arch's
     raw value is kept beside it as ``arch_cov_00`` so the deviation is
     measurable, and the generator asserts the two differ.

Every other option combination is pinned exactly as arch computes it.
Note arch's ``FullyModifiedOLS.fit`` raises ``KeyError`` for the spelling
``"quadratic-spectral"`` (it normalizes the name once and then looks the
raw string up again), so the generator passes ``"quadraticspectral"``.

Real-data illustration (DERIVED NUMBERS ONLY — the series are not stored):
``sm.datasets.get_rdataset("intdef", "wooldridge")`` (Wooldridge's annual
US 3-month T-bill rate ``i3`` on inflation ``inf``, 1948-2003, T = 56 —
his cointegration example) and ``get_rdataset("Tbrate", "Ecdat")``
(quarterly US T-bill rate ``r`` on inflation ``pi``, 1950Q1-1996Q4,
T = 188). The Python suite re-downloads them and skips when offline; the
numbers quoted in the model card come from this block.

Grade (honest): THIRD-PARTY golden (arch 8.0.0) for every estimator and
option except the two documented deviations, which are documented-formula
goldens; the kernel long-run covariances and arch's automatic bandwidth
are pinned separately against ``arch.covariance.kernel``. Whether the
corrected t-statistics are standard normal in finite samples is a
statistical claim the pin cannot make; the crate's seeded Monte Carlo
property tests (``fmols_properties.rs``) measure it.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_fmols_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import arch
import numpy as np
import statsmodels.api as sm
from arch.covariance.kernel import Bartlett, Parzen, QuadraticSpectral
from arch.unitroot.cointegration import (
    CanonicalCointegratingReg,
    DynamicOLS,
    FullyModifiedOLS,
)
from arch.utility.timeseries import add_trend

OUT = Path(__file__).resolve().parent / "fmols.json"

KERNELS = {
    "bartlett": ("bartlett", Bartlett, 1.1447, 1.0, 2 / 9),
    "parzen": ("parzen", Parzen, 2.6614, 2.0, 4 / 25),
    "quadratic-spectral": ("quadraticspectral", QuadraticSpectral, 1.3221, 2.0, 2 / 25),
}


# --------------------------------------------------------------- systems


def simulate(seed: int, t: int, kx: int, rho_e: float = 0.5, endog: float = 0.6,
             drift: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """A cointegrated system whose corrections have something to do: the
    regressors are random walks with AR(1) innovations, the equilibrium
    error is AR(1) AND correlated with the regressor innovations
    (`endog`), so plain OLS carries the Phillips-Hansen second-order bias."""
    rng = np.random.default_rng(seed)
    beta = np.linspace(1.0, 0.25, kx)
    u = np.zeros((t, kx))
    z = rng.standard_normal((t, kx))
    for i in range(1, t):
        u[i] = 0.3 * u[i - 1] + z[i]
    x = np.cumsum(u, axis=0) + drift * np.arange(1, t + 1)[:, None] * 0.05
    e = np.zeros(t)
    eps = rng.standard_normal(t)
    for i in range(1, t):
        e[i] = rho_e * e[i - 1] + eps[i] + endog * u[i].sum() / math.sqrt(kx)
    y = 0.5 + x @ beta + e + drift * np.arange(1, t + 1)
    return y, x, {"beta": beta.tolist(), "rho_e": rho_e, "endog": endog, "drift": drift}


def build_systems() -> dict[str, dict]:
    systems = {}
    for name, (seed, t, kx, drift) in {
        "sim_k1": (101, 200, 1, 0.0),
        "sim_k3": (103, 250, 3, 0.02),
        "sim_small": (105, 60, 2, 0.0),
    }.items():
        y, x, truth = simulate(seed, t, kx, drift=drift)
        systems[name] = {"y": y, "x": x, "truth": truth, "T": t, "k_x": kx}
    return systems


# ---------------------------------------------------- residual system


def residual_system(y, x, trend, x_trend=None, diff=False):
    """arch's `_common_fit` eta, transcribed: static OLS residual paired
    with the detrended regressor innovations."""
    z = add_trend(x, trend=trend)
    theta = np.linalg.lstsq(z, y, rcond=None)[0]
    eta_1 = y - z @ theta
    x_trend = trend if x_trend is None else x_trend
    tr = add_trend(nobs=x.shape[0], trend=x_trend)
    if tr.shape[1] > 1 and diff:
        delta_tr = np.diff(tr[:, 1:], axis=0)
        delta_x = np.diff(x, axis=0)
        gamma = np.linalg.lstsq(delta_tr, delta_x, rcond=None)[0]
        eta_2 = delta_x - delta_tr @ gamma
    else:
        if tr.shape[1]:
            gamma = np.linalg.lstsq(tr, x, rcond=None)[0]
            eps = x - tr @ gamma
        else:
            eps = x
        eta_2 = np.diff(eps, axis=0)
    return np.column_stack([eta_1[1:], eta_2]), theta


def andrews_bandwidth(eta: np.ndarray, kernel: str, force_int: bool) -> float:
    """Andrews (1991) AR(1) plug-in on the unit-weighted column sum (see
    the module docstring), finished as arch finishes its own rule."""
    _, _, c, q, _ = KERNELS[kernel]
    v = eta.sum(axis=1)
    rho = float(v[1:] @ v[:-1]) / float(v[:-1] @ v[:-1])
    if q == 1.0:
        alpha = 4 * rho**2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    else:
        alpha = 4 * rho**2 / (1 - rho) ** 4
    n = eta.shape[0]
    bw = c * (alpha * n) ** (1 / (2 * q + 1))
    if force_int:
        bw = math.ceil(bw)
    return float(min(bw, n - 1.0))


def newey_west_bandwidth(eta: np.ndarray, kernel: str, force_int: bool) -> float:
    """arch's `opt_bandwidth`, transcribed (asserted equal to arch's below)."""
    _, _, c, q, rate = KERNELS[kernel]
    v = eta.sum(axis=1)
    nobs = v.shape[0]
    n = int(np.ceil(4 * ((nobs / 100) ** rate)))
    f0 = f_q = 0.0
    for j in range(n + 1):
        sig = float(v[j:] @ v[: nobs - j]) / nobs
        scale = 1 + (j != 0)
        f0 += scale * sig
        f_q += scale * j**q * sig
    alpha = (f_q / f0) ** 2
    bw = c * (alpha * nobs) ** (1 / (2 * q + 1))
    if force_int:
        bw = np.ceil(bw)
    return float(min(bw, nobs - 1.0))


# ------------------------------------------------------------ one case


def _common(res, y, trend, kx, kernel, force_int, df_adjust):
    params = np.asarray(res.params, dtype=float)
    cov = np.asarray(res.cov, dtype=float)
    se = np.sqrt(np.diag(cov))
    return {
        "trend": trend,
        "k_x": kx,
        "kernel": kernel,
        "force_int": bool(force_int),
        "df_adjust": bool(df_adjust),
        "param_names": [str(c) for c in res.params.index],
        "params": params.tolist(),
        "se": se.tolist(),
        "tvalues": (params / se).tolist(),
        "pvalues": np.asarray(res.pvalues, dtype=float).tolist(),
        "cov": cov.tolist(),
        "resid": np.asarray(res.resid, dtype=float).tolist(),
        "bandwidth": float(res.bandwidth),
        "rsquared": float(res.rsquared),
        "rsquared_adj": float(res.rsquared_adj),
    }


def fm_or_ccr_case(system, y, x, estimator, trend, kernel="bartlett", bandwidth=None,
                   bandwidth_rule=None, force_int=True, diff=False, df_adjust=False,
                   x_trend=None):
    kx = x.shape[1]
    arch_kernel = KERNELS[kernel][0]
    cls = FullyModifiedOLS if estimator == "fmols" else CanonicalCointegratingReg
    eta, theta_ols = residual_system(y, x, trend, x_trend, diff)
    record = {
        "system": system,
        "estimator": estimator,
        "x_trend": x_trend,
        "diff": bool(diff),
        "bandwidth_arg": bandwidth,
        "bandwidth_rule": bandwidth_rule,
        "ols_params": theta_ols.tolist(),
        "source": "arch",
    }
    bw_arg = bandwidth
    fi = force_int
    if bandwidth_rule == "andrews":
        assert bandwidth is None
        bw_arg = andrews_bandwidth(eta, kernel, force_int)
        fi = False  # already finished; arch's ceil would be a no-op anyway
        record["source"] = "arch at the documented Andrews (1991) bandwidth"
    mod = cls(y, x, trend=trend, x_trend=x_trend)
    res = mod.fit(kernel=arch_kernel, bandwidth=bw_arg, force_int=fi, diff=diff,
                  df_adjust=df_adjust)
    record.update(_common(res, y, trend, kx, kernel, force_int, df_adjust))
    if bandwidth_rule == "andrews":
        record["bandwidth"] = bw_arg
        assert res.bandwidth == bw_arg
    elif bandwidth is None:
        # arch's rule, transcribed and asserted equal to what arch used.
        expect = newey_west_bandwidth(eta, kernel, force_int)
        # BLAS pairwise sums vs the sequential transcription: ~1e-13.
        assert math.isclose(res.bandwidth, expect, rel_tol=1e-12), (res.bandwidth, expect)
    # The residual system eta is arch's own (cross-check the transcription).
    est, eta_arch, _ = mod._common_fit(arch_kernel, bw_arg, fi, diff)
    assert np.allclose(eta, eta_arch, rtol=0, atol=1e-11), "eta transcription"
    record["long_run_variance"] = float(res.long_run_variance)
    if estimator == "ccr" and df_adjust:
        # Documented deviation (see the module docstring).
        base = cls(y, x, trend=trend, x_trend=x_trend).fit(
            kernel=arch_kernel, bandwidth=bw_arg, force_int=fi, diff=diff, df_adjust=False
        )
        m = y.shape[0] - 1
        nvar = len(record["params"])
        scale = m / (m - nvar)
        cov = scale * np.asarray(base.cov, dtype=float)
        record["arch_cov_00"] = float(np.asarray(res.cov)[0, 0])
        assert abs(record["arch_cov_00"] - cov[0, 0]) > 1e-12 * abs(cov[0, 0]), (
            "arch no longer deviates; drop the special case")
        record["cov"] = cov.tolist()
        record["se"] = np.sqrt(np.diag(cov)).tolist()
        record["tvalues"] = (np.asarray(record["params"]) / np.sqrt(np.diag(cov))).tolist()
        from scipy import stats
        record["pvalues"] = (2 * stats.norm.sf(np.abs(record["tvalues"]))).tolist()
        record["long_run_variance"] = float(scale * base.long_run_variance)
        record["source"] = "arch, df_adjust applied as documented (arch 8.0 precedence slip)"
    return record


def dols_case(system, y, x, trend, lags=None, leads=None, common=False, max_lag=None,
              max_lead=None, ic="bic", cov_type="unadjusted", kernel="bartlett",
              bandwidth=None, bandwidth_rule=None, force_int=False, df_adjust=False):
    kx = x.shape[1]
    arch_kernel = KERNELS[kernel][0]
    mod = DynamicOLS(y, x, trend=trend, lags=lags, leads=leads, common=common,
                     max_lag=max_lag, max_lead=max_lead, method=ic)
    record = {
        "system": system,
        "estimator": "dols",
        "lags_arg": lags,
        "leads_arg": leads,
        "common": bool(common),
        "max_lag_arg": max_lag,
        "max_lead_arg": max_lead,
        "ic": ic,
        "cov_type": cov_type,
        "bandwidth_arg": bandwidth,
        "bandwidth_rule": bandwidth_rule,
        "source": "arch",
    }
    bw_arg = bandwidth
    fi = force_int
    if bandwidth_rule == "andrews":
        assert bandwidth is None
        # The Andrews bandwidth is evaluated on the same series arch's rule
        # would see: the residuals (unadjusted) or the scores (robust) of
        # the regression at the selected leads/lags.
        pre = mod.fit(cov_type=cov_type, kernel=arch_kernel, force_int=force_int)
        lhs, rhs = mod._format_variables(pre.leads, pre.lags)
        eps = np.asarray(pre.resid)[:, None]
        series = eps if cov_type == "unadjusted" else np.asarray(rhs) * eps
        bw_arg = andrews_bandwidth(series, kernel, force_int)
        fi = False
        record["source"] = "arch at the documented Andrews (1991) bandwidth"
    res = mod.fit(cov_type=cov_type, kernel=arch_kernel, bandwidth=bw_arg, force_int=fi,
                  df_adjust=df_adjust)
    record.update(_common(res, y, trend, kx, kernel, force_int, df_adjust))
    full_params = np.asarray(res.full_params, dtype=float)
    full_cov = np.asarray(res.full_cov, dtype=float)
    record.update({
        "full_param_names": [str(c) for c in res.full_params.index],
        "full_params": full_params.tolist(),
        "full_se": np.sqrt(np.diag(full_cov)).tolist(),
        "full_cov": full_cov.tolist(),
        "lags": int(res.lags),
        "leads": int(res.leads),
        "nobs": int(np.asarray(res.resid).shape[0]),
    })
    if bandwidth_rule == "andrews":
        record["bandwidth"] = bw_arg
        assert res.bandwidth == bw_arg
    elif bandwidth is None:
        lhs, rhs = mod._format_variables(res.leads, res.lags)
        eps = np.asarray(res.resid)[:, None]
        series = eps if cov_type == "unadjusted" else np.asarray(rhs) * eps
        expect = newey_west_bandwidth(series, kernel, force_int)
        assert math.isclose(res.bandwidth, expect, rel_tol=1e-12), (res.bandwidth, expect)
    # Static OLS comparison (full sample).
    z = add_trend(x, trend=trend)
    record["ols_params"] = np.linalg.lstsq(z, y, rcond=None)[0].tolist()
    # The residual long-run variance the "unadjusted" covariance uses
    # (uncentered kernel LRV of the residuals at the reported bandwidth,
    # times the df scale).
    est = KERNELS[kernel][1](np.asarray(res.resid), bandwidth=res.bandwidth, center=False,
                             force_int=False)
    nobs, k = len(res.resid), full_params.shape[0]
    scale = nobs / (nobs - k) if df_adjust else 1.0
    record["long_run_variance"] = float(scale * np.asarray(est.cov.long_run)[0, 0])
    return record


# ---------------------------------------------------------- case grid


def gen_cases(systems):
    cases = []
    # Every system x trend x kernel at the defaults (automatic bandwidth,
    # force_int True for FM-OLS/CCR).
    for name in ("sim_k1", "sim_k3", "sim_small"):
        y, x = systems[name]["y"], systems[name]["x"]
        for trend in ("n", "c", "ct", "ctt"):
            for kernel in KERNELS:
                cases.append(fm_or_ccr_case(name, y, x, "fmols", trend, kernel=kernel))
                cases.append(fm_or_ccr_case(name, y, x, "ccr", trend, kernel=kernel))
    y, x = systems["sim_k1"]["y"], systems["sim_k1"]["x"]
    y3, x3 = systems["sim_k3"]["y"], systems["sim_k3"]["x"]
    for est in ("fmols", "ccr"):
        # Non-integer automatic bandwidths, every kernel.
        for kernel in KERNELS:
            cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "c", kernel=kernel,
                                        force_int=False))
        # Explicit bandwidths: integer, non-integer, non-integer ceiled.
        cases.append(fm_or_ccr_case("sim_k1", y, x, est, "c", bandwidth=4.0))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ct", kernel="parzen",
                                    bandwidth=3.7, force_int=False))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ct", kernel="parzen",
                                    bandwidth=3.7, force_int=True))
        cases.append(fm_or_ccr_case("sim_k1", y, x, est, "c", kernel="quadratic-spectral",
                                    bandwidth=0.0))
        # df_adjust (CCR: the documented deviation).
        cases.append(fm_or_ccr_case("sim_k1", y, x, est, "c", df_adjust=True))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ctt", kernel="parzen",
                                    df_adjust=True))
        # diff with a trend, and x_trend larger than trend.
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ct", diff=True))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ctt", kernel="parzen",
                                    diff=True))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "c", x_trend="ct"))
        cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "ct", x_trend="ctt", diff=True))
        cases.append(fm_or_ccr_case("sim_k1", y, x, est, "n", x_trend="c"))
        # The Andrews rule, every kernel, int and non-int.
        for kernel in KERNELS:
            cases.append(fm_or_ccr_case("sim_k3", y3, x3, est, "c", kernel=kernel,
                                        bandwidth_rule="andrews"))
            cases.append(fm_or_ccr_case("sim_k1", y, x, est, "ct", kernel=kernel,
                                        bandwidth_rule="andrews", force_int=False))
    # DOLS: defaults on the two systems large enough for the default
    # search cap. On `sim_small` (T = 60, k_x = 2) the default cap
    # ceil(12 (60/100)^(1/4)) = 11 leaves 60 - 1 - 22 = 37 rows for up to
    # 2 + 1 + 2 x 23 = 49 parameters: arch's `_check_inputs` only requires
    # obs_remaining > 0, so it silently runs an UNDERDETERMINED search
    # (rank-deficient lstsq, zero residuals, an information criterion of
    # -inf on the largest candidates) — tsecon refuses that specification
    # with a teaching error instead, and the `refusals` block below
    # records the case so the Rust golden pins the refusal.
    for name in ("sim_k1", "sim_k3"):
        yy, xx = systems[name]["y"], systems[name]["x"]
        for trend in ("n", "c", "ct", "ctt"):
            cases.append(dols_case(name, yy, xx, trend))
    for ic in ("aic", "hqic"):
        cases.append(dols_case("sim_k1", y, x, "c", ic=ic))
        cases.append(dols_case("sim_k3", y3, x3, "ct", ic=ic))
    cases.append(dols_case("sim_k1", y, x, "c", lags=2, leads=1))
    cases.append(dols_case("sim_k3", y3, x3, "ct", lags=1, leads=3, cov_type="robust"))
    cases.append(dols_case("sim_k1", y, x, "c", lags=2))          # leads searched
    cases.append(dols_case("sim_k3", y3, x3, "c", leads=0))        # lags searched
    cases.append(dols_case("sim_k1", y, x, "c", common=True))
    cases.append(dols_case("sim_k3", y3, x3, "ctt", common=True, ic="aic"))
    cases.append(dols_case("sim_k1", y, x, "c", max_lag=3, max_lead=2))
    cases.append(dols_case("sim_k3", y3, x3, "ct", max_lag=2, max_lead=2, common=True))
    cases.append(dols_case("sim_k1", y, x, "n", lags=0, leads=0))
    for kernel in KERNELS:
        cases.append(dols_case("sim_k1", y, x, "c", kernel=kernel, cov_type="robust"))
        cases.append(dols_case("sim_k3", y3, x3, "c", kernel=kernel, lags=1, leads=1,
                               force_int=True))
        cases.append(dols_case("sim_k1", y, x, "c", kernel=kernel, bandwidth_rule="andrews"))
        cases.append(dols_case("sim_k3", y3, x3, "ct", kernel=kernel, lags=1, leads=1,
                               cov_type="robust", bandwidth_rule="andrews"))
    cases.append(dols_case("sim_k1", y, x, "c", bandwidth=2.0))
    cases.append(dols_case("sim_k3", y3, x3, "ct", bandwidth=2.5, cov_type="robust",
                           kernel="parzen"))
    cases.append(dols_case("sim_k3", y3, x3, "ct", bandwidth=2.5, force_int=True))
    cases.append(dols_case("sim_k1", y, x, "c", df_adjust=True))
    cases.append(dols_case("sim_k3", y3, x3, "ctt", df_adjust=True, cov_type="robust"))
    cases.append(dols_case("sim_small", systems["sim_small"]["y"], systems["sim_small"]["x"],
                           "c", lags=1, leads=1, cov_type="robust", df_adjust=True))
    return cases


def refusals(systems):
    """Specifications arch accepts but which cannot be estimated, pinned
    as refusals (see the comment in `gen_cases`)."""
    t, kx = systems["sim_small"]["T"], systems["sim_small"]["k_x"]
    cap = math.ceil(12.0 * (t / 100) ** 0.25)
    rows = t - 1 - 2 * cap
    n_params = kx + 1 + kx * (2 * cap + 1)
    assert rows <= n_params
    return [{
        "system": "sim_small", "estimator": "dols", "trend": "c",
        "default_cap": cap, "rows": rows, "n_params": n_params,
        "must_name": ["max_lag", "max_lead"],
    }]


# -------------------------------------------------------- kernel block


def kernel_block(systems):
    """The long-run covariance pieces of a residual system against
    `arch.covariance.kernel`, plus the two automatic bandwidths."""
    y, x = systems["sim_k3"]["y"], systems["sim_k3"]["x"]
    eta, _ = residual_system(y, x, "c")
    block = {"eta": eta.tolist(), "cases": []}
    for kernel, (_, cls, _, _, _) in KERNELS.items():
        for force_int in (False, True):
            est = cls(eta, center=False, force_int=force_int)
            block["cases"].append({
                "kernel": kernel, "bandwidth_arg": None, "force_int": force_int,
                "bandwidth": float(est.bandwidth),
                "newey_west_bandwidth": newey_west_bandwidth(eta, kernel, force_int),
                "andrews_bandwidth": andrews_bandwidth(eta, kernel, force_int),
                "n_weights": int(est.kernel_weights.shape[0]),
                "short_run": np.asarray(est.cov.short_run).tolist(),
                "one_sided": np.asarray(est.cov.one_sided).tolist(),
                "long_run": np.asarray(est.cov.long_run).tolist(),
            })
            assert math.isclose(block["cases"][-1]["newey_west_bandwidth"], float(est.bandwidth),
                                rel_tol=1e-12)
        for bw in (0.0, 3.7, 6.0):
            est = cls(eta, bandwidth=bw, center=False)
            block["cases"].append({
                "kernel": kernel, "bandwidth_arg": bw, "force_int": False,
                "bandwidth": float(est.bandwidth),
                "n_weights": int(est.kernel_weights.shape[0]),
                "weights": np.asarray(est.kernel_weights)[:8].tolist(),
                "short_run": np.asarray(est.cov.short_run).tolist(),
                "one_sided": np.asarray(est.cov.one_sided).tolist(),
                "long_run": np.asarray(est.cov.long_run).tolist(),
            })
    return block


# ----------------------------------------------------------- real data


def real_block():
    out = {}
    specs = {
        "intdef": ("intdef", "wooldridge", "i3", ["inf"],
                   "Wooldridge intdef: 3-month T-bill rate i3 on inflation inf, annual 1948-2003"),
        "tbrate": ("Tbrate", "Ecdat", "r", ["pi"],
                   "Ecdat Tbrate: US T-bill rate r on inflation pi, quarterly 1950Q1-1996Q4"),
    }
    for key, (item, pkg, yname, xnames, desc) in specs.items():
        try:
            d = sm.datasets.get_rdataset(item, pkg).data
        except Exception as exc:  # pragma: no cover - offline
            print(f"  {item}/{pkg} unreachable ({exc!r}); real block skipped")
            continue
        y = d[yname].to_numpy(dtype=float)
        x = d[xnames].to_numpy(dtype=float)
        fm = FullyModifiedOLS(y, x, trend="c").fit()
        cc = CanonicalCointegratingReg(y, x, trend="c").fit()
        do = DynamicOLS(y, x, trend="c").fit()
        # The default search at T = 56 keeps a 31-row common sample for up
        # to 27 parameters and picks a (9, 11) model — arch's behaviour,
        # recorded as such; the fixed (1, 1) fit is the one a textbook
        # application uses on a sample this short.
        do11 = DynamicOLS(y, x, trend="c", lags=1, leads=1).fit()
        z = add_trend(x, trend="c")
        ols = np.linalg.lstsq(z, y, rcond=None)[0]
        rec = {"item": item, "package": pkg, "y": yname, "x": xnames, "T": int(y.shape[0]),
               "description": desc, "ols_params": ols.tolist()}
        for name, res in (("fmols", fm), ("ccr", cc), ("dols", do), ("dols_11", do11)):
            params = np.asarray(res.params, dtype=float)
            se = np.sqrt(np.diag(np.asarray(res.cov, dtype=float)))
            rec[name] = {"params": params.tolist(), "se": se.tolist(),
                         "tvalues": (params / se).tolist(), "bandwidth": float(res.bandwidth),
                         "rsquared": float(res.rsquared)}
        rec["dols"]["lags"] = int(do.lags)
        rec["dols"]["leads"] = int(do.leads)
        rec["dols_11"]["lags"] = 1
        rec["dols_11"]["leads"] = 1
        out[key] = rec
    return out


# ----------------------------------------------------------------- main


def main():
    systems = build_systems()
    cases = gen_cases(systems)
    fixture = {
        "_meta": {
            "reference": "arch.unitroot.cointegration.{FullyModifiedOLS, DynamicOLS, "
                         "CanonicalCointegratingReg} + arch.covariance.kernel",
            "arch_version": arch.__version__,
            "numpy_version": np.__version__,
            "statsmodels_version": sm.__version__,
            "generator": Path(__file__).name,
            "note": "see the module docstring for the two documented deviations "
                    "(the Andrews bandwidth rule; CCR df_adjust)",
        },
        "systems": {
            name: {"y": s["y"].tolist(), "x": s["x"].tolist(), "truth": s["truth"],
                   "T": s["T"], "k_x": s["k_x"]}
            for name, s in systems.items()
        },
        "cases": cases,
        "refusals": refusals(systems),
        "kernel": kernel_block(systems),
        "real": real_block(),
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    n_fm = sum(c["estimator"] == "fmols" for c in cases)
    n_cc = sum(c["estimator"] == "ccr" for c in cases)
    n_do = sum(c["estimator"] == "dols" for c in cases)
    print(f"wrote {OUT} ({n_fm} fmols, {n_cc} ccr, {n_do} dols cases; "
          f"{len(fixture['kernel']['cases'])} kernel cases; real: {sorted(fixture['real'])})")
    for key, rec in fixture["real"].items():
        print(f"  {key}: T={rec['T']} ols={np.round(rec['ols_params'], 4).tolist()}")
        for name in ("fmols", "dols", "dols_11", "ccr"):
            r = rec[name]
            print(f"    {name:7s} params={np.round(r['params'], 4).tolist()} "
                  f"se={np.round(r['se'], 4).tolist()} t={np.round(r['tvalues'], 3).tolist()} "
                  f"bw={r['bandwidth']:.4g}"
                  + (f" lags={r['lags']} leads={r['leads']}" if name.startswith("dols") else ""))


if __name__ == "__main__":
    main()
