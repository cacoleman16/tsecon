"""Golden fixture for the Beveridge-Nelson build-out and the Hamilton
filter's HAC standard errors -> fixtures/bn_filters.json.

Run with a NumPy/statsmodels venv (this script never imports tsecon); R
must be on PATH and $BNFILTER_R_DIR must point at the R/ directory of a
github.com/kletts/bnfilter checkout for the KMW reference leg (see
generate_bn_filter_fixtures.R):

    BNFILTER_R_DIR=/path/to/bnfilter/R .venv/bin/python \
        fixtures/generate_bn_filters_fixtures.py

============================================================================
WHAT KIND OF GOLDEN EACH BLOCK IS
============================================================================

* ``hamilton_hac`` — **statsmodels third-party golden.** statsmodels has
  no Hamilton (2018) filter (checked: nothing under
  statsmodels.tsa.filters provides it; the ``canary`` entry pins the
  absence), but the filter is literally OLS of y_t on [1, y_{t-h}, ...,
  y_{t-h-p+1}], so its coefficient inference IS statsmodels territory:
  ``OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L,
  "use_correction": ...})`` on the identical design. bse/tvalues are
  pinned for the nonrobust and several HAC settings, including the
  h-step-overlap default maxlags = h = 8.

* ``bn_arma`` — **documented-formula transcription golden with a partial
  statsmodels pin.** statsmodels has no Beveridge-Nelson decomposition
  either (same canary), so the trend/cycle values come from the
  independent NumPy transcription below of the companion-form BN
  computation (Morley 2002, J. Applied Econometrics 17:427-443):

      x_t   = Delta y_t - mu                          (demeaned growth)
      eps_t = conditional (zero-presample) ARMA innovation recursion
      X_t   = [x_t .. x_{t-p+1}, eps_t .. eps_{t-q+1}]'
      cycle_t = -e1' F (I-F)^{-1} X_t,   trend_t = y_t - cycle_t

  with F the ARMA companion matrix. Three exact algebraic identities are
  asserted before anything is stored: trend + cycle == y;
  Delta trend_t == mu + psi(1) * eps_t (the BN trend is exactly a random
  walk with drift); and the closed-form long-run multiplier
  psi(1) = theta(1)/phi(1) equals the *cumulative sum of statsmodels'*
  ``arma_impulse_response`` — the one leg statsmodels CAN check, and the
  genuine third-party pin of the number that defines the decomposition.

* ``kmw`` — **reference-run golden against the authors' own code.** The
  cycle/delta/AR/SE values come from actual R runs of the
  Kamber-Morley-Wong BN filter reference implementation
  (bnfiltering.com replication code, R conversion by Luke Hartigan of
  Ben Wong's MATLAB, as packaged by github.com/kletts/bnfilter, commit
  8af7924), executed by the committed generate_bn_filter_fixtures.R with
  the KMW-2018 baseline options: sample-mean demeaning, no iterative
  backcasting (ib = FALSE — the reference code's own "as in KMW2018"
  setting), delta_select = 1 (first local max of the amplitude-to-noise
  ratio, the 2018 criterion; grid d0 = 0.01, dt = 0.0005), fixed
  non-dynamic error bands. This generator ALSO reimplements the whole
  procedure independently in NumPy below and asserts agreement with the
  R run at 1e-9 before writing, so the stored numbers are simultaneously
  a reference run and a two-implementation transcription golden.
  Honest caveats: (a) the packaged code is the authors' current
  (2022-2025-refined) lineage run at its 2018-baseline settings, not a
  bit-frozen 2017 snapshot; it includes the Bayesian shrinkage prior
  (variance 0.5/j^2 on the Dickey-Fuller coefficients) that the refined
  code applies on all paths. (b) kletts/bnfilter is a re-packaging of
  the bnfiltering.com code, not the authors' own repository.

Series: the same 100*log US real GDP as fixtures/filters.json
(statsmodels macrodata, public-domain US-government data; only the
transformation is stored, and only in filters.json — the Rust tests read
it from there), plus a seeded simulated random-walk-with-drift + AR(2)
cycle series stored here.
"""
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima_process import arma_impulse_response

OUT = Path(__file__).parent

META = {
    "numpy": np.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
    "kmw_reference": "bnfiltering.com R code via github.com/kletts/bnfilter@8af7924",
}


# ---------------------------------------------------------------------------
# Independent NumPy transcriptions (the cross-check legs).
# ---------------------------------------------------------------------------

def bn_arma_numpy(y, ar, ma, drift):
    """Companion-form classic BN (Morley 2002), zero-presample innovations."""
    y = np.asarray(y, float)
    ar = np.asarray(ar, float)
    ma = np.asarray(ma, float)
    p, q = len(ar), len(ma)
    pp = max(p, 1)
    phi = np.zeros(pp)
    phi[:p] = ar

    x = np.diff(y) - drift
    T = len(x)
    eps = np.zeros(T)
    for t in range(T):
        acc = x[t]
        for i in range(1, pp + 1):
            if t - i >= 0:
                acc -= phi[i - 1] * x[t - i]
        for j in range(1, q + 1):
            if t - j >= 0:
                acc -= ma[j - 1] * eps[t - j]
        eps[t] = acc

    r = pp + q
    F = np.zeros((r, r))
    F[0, :pp] = phi
    F[0, pp:] = ma
    for i in range(1, pp):
        F[i, i - 1] = 1.0
    for j in range(1, q):
        F[pp + j, pp + j - 1] = 1.0
    w = np.linalg.solve((np.eye(r) - F).T, F[0, :])  # e1' F (I-F)^{-1}

    cycle = np.zeros(T)
    for t in range(T):
        st = np.zeros(r)
        for i in range(pp):
            st[i] = x[t - i] if t - i >= 0 else 0.0
        for j in range(q):
            st[pp + j] = eps[t - j] if t - j >= 0 else 0.0
        cycle[t] = -w @ st
    trend = y[1:] - cycle
    psi1 = (1.0 + ma.sum()) / (1.0 - ar.sum())
    return trend, cycle, eps, psi1


def kmw_numpy(y, p, delta, demean=True):
    """KMW-2018-baseline BN filter, transcribed independently."""
    y = np.asarray(y, float)
    dy = np.diff(y)
    drift = dy.mean() if demean else 0.0
    x = dy - drift
    T = len(x)
    rho = 1.0 - 1.0 / np.sqrt(delta)

    xp = np.concatenate([np.zeros(p + 2), x])  # unconditional-mean padding
    idx = np.arange(T) + p + 2  # padded positions of x_1..x_T

    # sigma^2 from the unrestricted zero-padded no-constant AR(p) OLS.
    Xu = np.column_stack([xp[idx - k] for k in range(1, p + 1)])
    beta_u, *_ = np.linalg.lstsq(Xu, xp[idx], rcond=None)
    sig2_ols = np.sum((xp[idx] - Xu @ beta_u) ** 2) / (T - p)

    # Dickey-Fuller design; Bayesian ridge with prior variance 0.5/j^2.
    dxp = xp - np.concatenate([[0.0], xp[:-1]])
    Xdf = np.column_stack([dxp[idx - k] for k in range(1, p)])
    ydf = xp[idx] - rho * xp[idx - 1]
    j = np.arange(1, p)
    v_post = np.linalg.inv(np.diag(j * j / 0.5) + (Xdf.T @ Xdf) / sig2_ols)
    psi = v_post @ (Xdf.T @ ydf) / sig2_ols

    phi = np.zeros(p)
    phi[p - 1] = -psi[p - 2]
    for i in range(p - 2, 0, -1):
        phi[i] = -psi[i - 1] - np.sum(phi[i + 1:])
    phi[0] = rho - np.sum(phi[1:])

    F = np.zeros((p, p))
    F[0, :] = phi
    F[1:, :-1] = np.eye(p - 1)
    w = np.linalg.solve((np.eye(p) - F).T, F[0, :])
    states = np.column_stack([xp[idx - k] for k in range(0, p)])
    cycle = -(states @ w)
    residuals = xp[idx] - Xu @ phi
    amp_to_noise = float(np.var(cycle, ddof=1) / np.mean(residuals ** 2))

    # Fixed KMW band: sigma_c^2 from the UNPADDED AR(p)-with-constant OLS,
    # state covariance from the vec'd discrete Lyapunov equation.
    rows = np.arange(p, T)
    Xc = np.column_stack([np.ones(T - p)] + [x[rows - k] for k in range(1, p + 1)])
    bc, *_ = np.linalg.lstsq(Xc, x[rows], rcond=None)
    sig2_c = np.sum((x[rows] - Xc @ bc) ** 2) / ((T - p) - (p + 1))
    vec_q = np.zeros(p * p)
    vec_q[0] = sig2_c
    sigma_x = (np.linalg.inv(np.eye(p * p) - np.kron(F, F)) @ vec_q).reshape(
        p, p, order="F"
    )
    cycle_se = float(np.sqrt(w @ sigma_x @ w))
    return {
        "delta": float(delta),
        "cycle": cycle,
        "ar": phi,
        "cycle_se": cycle_se,
        "sig2_ols_c": float(sig2_c),
        "amp_to_noise": amp_to_noise,
        "drift": float(drift),
    }


def kmw_numpy_select_delta(y, p, d0=0.01, dt=0.0005, demean=True):
    delta = d0
    old = kmw_numpy(y, p, delta, demean)["amp_to_noise"]
    while True:
        cand = delta + dt
        new = kmw_numpy(y, p, cand, demean)["amp_to_noise"]
        if new > old:
            delta, old = cand, new
        else:
            return delta


def kmw_reference_run(y, p, delta_arg, demean_arg):
    """The authors' code, via Rscript + generate_bn_filter_fixtures.R."""
    with tempfile.TemporaryDirectory() as td:
        series = Path(td) / "series.csv"
        out = Path(td) / "out.json"
        np.savetxt(series, np.asarray(y, float), fmt="%.17g")
        subprocess.run(
            [
                "Rscript",
                str(OUT / "generate_bn_filter_fixtures.R"),
                str(series),
                str(p),
                delta_arg,
                demean_arg,
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        return json.loads(out.read_text())


def kmw_case(y, p, delta_arg, demean_arg):
    """Reference-run + independent-transcription cross-check, then store."""
    ref = kmw_reference_run(y, p, delta_arg, demean_arg)
    demean = demean_arg == "sm"
    if delta_arg == "auto":
        mine_delta = kmw_numpy_select_delta(y, p, demean=demean)
        assert abs(mine_delta - ref["delta"]) < 1e-12, (mine_delta, ref["delta"])
    else:
        mine_delta = float(delta_arg)
    mine = kmw_numpy(y, p, mine_delta, demean)
    for key in ("cycle", "ar"):
        np.testing.assert_allclose(
            mine[key], np.asarray(ref[key]), rtol=1e-9, atol=1e-9, err_msg=key
        )
    for key in ("cycle_se", "sig2_ols_c", "amp_to_noise"):
        assert abs(mine[key] - ref[key]) <= 1e-9 * max(1.0, abs(ref[key])), key
    META.setdefault("r_version", ref["r_version"])
    return {
        "p": p,
        "demean": demean,
        "delta_mode": "auto" if delta_arg == "auto" else "fixed",
        "delta": ref["delta"],
        "cycle": ref["cycle"],
        "ar": ref["ar"],
        "cycle_se": ref["cycle_se"],
        "amp_to_noise": ref["amp_to_noise"],
        "drift": mine["drift"],
    }


def bn_arma_case(y, ar, ma, drift):
    trend, cycle, eps, psi1 = bn_arma_numpy(y, ar, ma, drift)
    y = np.asarray(y, float)
    # Exact identities (algebra, not estimation — fail loudly if broken).
    np.testing.assert_allclose(trend + cycle, y[1:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        np.diff(trend), drift + psi1 * eps[1:], rtol=0, atol=1e-9
    )
    # The genuine statsmodels pin: psi(1) == cumulative impulse response.
    irf = arma_impulse_response(np.r_[1, -np.asarray(ar)], np.r_[1, np.asarray(ma)],
                                leads=20000)
    assert abs(psi1 - irf.sum()) < 1e-8, (psi1, irf.sum())
    return {
        "ar": list(map(float, ar)),
        "ma": list(map(float, ma)),
        "drift": float(drift),
        "trend": trend.tolist(),
        "cycle": cycle.tolist(),
        "innovations": eps.tolist(),
        "long_run_multiplier": float(psi1),
        "long_run_multiplier_sm_cum_irf": float(irf.sum()),
    }


def main():
    mac = sm.datasets.macrodata.load_pandas().data
    gdp = 100.0 * np.log(mac["realgdp"].to_numpy())

    rng = np.random.default_rng(20260823)
    n = 240
    trend_shocks = rng.standard_normal(n) * 0.5
    level = np.cumsum(0.4 + trend_shocks) + 900.0
    c = np.zeros(n)
    e = rng.standard_normal(n) * 0.6
    for t in range(2, n):
        c[t] = 1.4 * c[t - 1] - 0.5 * c[t - 2] + e[t]
    sim = level + c

    # ---- canary: statsmodels ships neither estimator ----------------------
    import statsmodels.tsa.filters as smf

    absent = {
        "hamilton": not any("hamilton" in x.lower() for x in dir(smf)),
        "beveridge_nelson": not any(
            "beveridge" in x.lower() or x.lower() == "bn" for x in dir(sm.tsa)
        ),
    }
    assert all(absent.values()), absent

    # ---- hamilton_hac -----------------------------------------------------
    h, p = 8, 4
    rows = np.arange(h + p - 1, len(gdp))
    X = np.column_stack([np.ones(len(rows))] + [gdp[rows - h - k] for k in range(p)])
    fit = sm.OLS(gdp[rows], X).fit()

    def hac(maxlags, corr):
        f = sm.OLS(gdp[rows], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags, "use_correction": corr}
        )
        return {
            "maxlags": maxlags,
            "use_correction": corr,
            "bse": np.asarray(f.bse).tolist(),
            "tvalues": np.asarray(f.tvalues).tolist(),
        }

    hamilton_hac = {
        "h": h,
        "p": p,
        "beta": np.asarray(fit.params).tolist(),
        "nonrobust": {
            "bse": np.asarray(fit.bse).tolist(),
            "tvalues": np.asarray(fit.tvalues).tolist(),
        },
        "hac_h8_corr": hac(8, True),
        "hac_h8_nocorr": hac(8, False),
        "hac_l4_corr": hac(4, True),
    }

    # ---- bn_arma ----------------------------------------------------------
    # US GDP: the Morley-Nelson-Zivot (2003) ARIMA(2,1,2)-with-constant
    # spec, coefficients estimated by statsmodels SARIMAX in the same
    # simple-differencing convention tsecon's arima_fit uses; the BN is
    # then computed at those coefficients. mu = const / phi(1).
    mod = sm.tsa.SARIMAX(
        gdp, order=(2, 1, 2), trend="c", simple_differencing=True
    ).fit(disp=0)
    par = mod.params
    ar_hat = [float(par["ar.L1"]), float(par["ar.L2"])]
    ma_hat = [float(par["ma.L1"]), float(par["ma.L2"])]
    mu_hat = float(par["intercept"]) / (1.0 - sum(ar_hat))

    bn_arma = {
        "gdp_arima212": bn_arma_case(gdp, ar_hat, ma_hat, mu_hat),
        # Fixed textbook coefficients on the simulated series — a pure
        # formula pin with no estimation in the loop.
        "sim_arma11_fixed": bn_arma_case(sim, [0.5], [0.3], 0.4),
        "sim_ar2_fixed": bn_arma_case(sim, [0.3, 0.2], [], 0.4),
    }

    # ---- kmw --------------------------------------------------------------
    kmw = {
        "usgdp_p12_auto_sm": kmw_case(gdp, 12, "auto", "sm"),
        "usgdp_p12_fixed025_sm": kmw_case(gdp, 12, "0.25", "sm"),
        "sim_p12_auto_sm": kmw_case(sim, 12, "auto", "sm"),
        "sim_p8_fixed005_nd": kmw_case(sim, 8, "0.05", "nd"),
    }

    obj = {
        "_meta": META,
        "statsmodels_absence_canary": absent,
        "sim_series": sim.tolist(),
        "hamilton_hac": hamilton_hac,
        "bn_arma": bn_arma,
        "kmw": kmw,
    }
    path = OUT / "bn_filters.json"
    path.write_text(json.dumps(obj, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")
    print("kmw usgdp auto delta:", kmw["usgdp_p12_auto_sm"]["delta"])
    print("gdp arima212:", ar_hat, ma_hat, mu_hat)


if __name__ == "__main__":
    main()
