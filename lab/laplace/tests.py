"""Seeded pytest suite for lab/laplace (AL-GAS quantiles, DCS robust
local level, Laplace/LAD ARMA).

Run from the shared venv:

    cd /home/user/tsecon/lab/laplace
    /home/user/tsecon/.venv/bin/python -m pytest tests.py -v -s

Every DGP is seeded; asserted thresholds were set with a margin below
observed values (observed numbers quoted in comments and printed by
each test so regressions are visible).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from al_arima import fit_arma_css, simulate_arma  # noqa: E402
from al_gas import fit_al_gas, fit_al_gas_multi  # noqa: E402
from robust_filter import (  # noqa: E402
    fit_dcs_local_level,
    simulate_local_level,
    steady_state_gain,
)


# ----------------------------------------------------------------- AL-GAS

def _garch_dgp(T=3000, seed=7, om=0.02, al=0.10, be=0.88):
    """GARCH(1,1) with Gaussian shocks -> known conditional quantile path
    q_t(tau) = sigma_t * Phi^{-1}(tau)."""
    rng = np.random.default_rng(seed)
    sig2 = np.empty(T)
    y = np.empty(T)
    s2 = om / (1.0 - al - be)
    for t in range(T):
        sig2[t] = s2
        y[t] = np.sqrt(s2) * rng.standard_normal()
        s2 = om + al * y[t] ** 2 + be * s2
    return y, sig2


def test_al_gas_tracks_garch_quantile():
    from scipy.stats import norm

    y, sig2 = _garch_dgp(T=3000, seed=7)
    tau = 0.05
    q_true = np.sqrt(sig2) * norm.ppf(tau)

    r = fit_al_gas(y, tau)
    rmse_gas = float(np.sqrt(np.mean((r.q - q_true) ** 2)))
    q_static = np.quantile(y, tau)
    rmse_static = float(np.sqrt(np.mean((q_static - q_true) ** 2)))
    ratio = rmse_gas / rmse_static
    print(f"\n[al_gas] tau=0.05 GARCH tracking: RMSE gas={rmse_gas:.4f} "
          f"static={rmse_static:.4f} ratio={ratio:.3f} "
          f"hit={r.hit_rate:.4f} b={r.b:.4f} a={r.a:.4f}")
    # observed: ratio ~ 0.52, hit ~ 0.054
    assert ratio < 0.65, "AL-GAS must beat static quantile decisively"
    assert 0.03 < r.hit_rate < 0.07, "empirical coverage far from tau"
    assert abs(r.b) < 1.0
    assert r.a >= 0.0


def test_al_gas_multi_tau_non_crossing():
    y, _ = _garch_dgp(T=2000, seed=21)
    m = fit_al_gas_multi(y, (0.05, 0.25, 0.5, 0.75, 0.95))
    mono_cross = float(np.mean(np.any(np.diff(m.q_monotone, axis=1) < 0.0,
                                      axis=1)))
    print(f"\n[al_gas] multi-tau crossing: raw={m.crossing_frac:.4f} "
          f"after rearrangement={mono_cross:.4f}")
    # observed: raw crossing 0.0 on this DGP; keep a small allowance
    assert m.crossing_frac < 0.05
    assert mono_cross == 0.0
    # hit rates roughly ordered / near nominal at the extremes
    for tau in (0.05, 0.95):
        hr = m.results[tau].hit_rate
        assert abs(hr - tau) < 0.03, (tau, hr)


# ----------------------------------------------------- DCS robust filter

def test_robust_filter_beats_gaussian_under_outliers():
    y, mu_true, mask = simulate_local_level(
        800, sigma_eta=0.1, sigma_eps=1.0,
        outlier_frac=0.07, outlier_size=9.0, seed=11,
    )
    rmse = {}
    fits = {}
    for d in ("gaussian", "t", "laplace"):
        r = fit_dcs_local_level(y, d)
        fits[d] = r
        rmse[d] = float(np.sqrt(np.mean((r.mu - mu_true) ** 2)))
    print(f"\n[robust_filter] 7% additive outliers (9 sd): level RMSE "
          f"gauss={rmse['gaussian']:.4f} t={rmse['t']:.4f} "
          f"laplace={rmse['laplace']:.4f}  "
          f"(nu_hat={fits['t'].nu:.2f}, kappa_g={fits['gaussian'].kappa:.4f})")
    # observed: gauss 0.418, t 0.311 (ratio .74), laplace 0.324 (ratio .78)
    assert rmse["t"] < 0.85 * rmse["gaussian"]
    assert rmse["laplace"] < 0.85 * rmse["gaussian"]
    # t-likelihood should also dominate decisively on contaminated data
    assert fits["t"].loglik > fits["gaussian"].loglik + 100.0


def test_gaussian_nesting_on_clean_data():
    """DCS-Gaussian = steady-state Kalman; DCS-t -> Gaussian as nu -> inf."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    y, mu_true, _ = simulate_local_level(800, 0.1, 1.0, 0.0, seed=3)
    rg = fit_dcs_local_level(y, "gaussian")
    rt = fit_dcs_local_level(y, "t")

    uc = UnobservedComponents(y, "llevel").fit(disp=0)
    s2eps, s2eta = float(uc.params[0]), float(uc.params[1])
    k_ss = steady_state_gain(s2eta, s2eps)
    print(f"\n[robust_filter] clean-data nesting: kappa_gauss={rg.kappa:.4f} "
          f"steady-state Kalman gain={k_ss:.4f} |diff|={abs(rg.kappa - k_ss):.5f}; "
          f"DCS-t nu_hat={rt.nu:.1f} kappa_t={rt.kappa:.4f}")
    # observed |diff| ~ 6e-4
    assert abs(rg.kappa - k_ss) < 0.02
    # sigma of the DCS-Gaussian prediction error ~ sqrt(steady prediction
    # variance + obs variance) = sqrt(s2eps*(1+p)) with p = k/(1-k)
    sig_pred = np.sqrt(s2eps / (1.0 - k_ss))
    assert abs(rg.scale - sig_pred) / sig_pred < 0.05

    # paths: DCS-Gaussian vs exact Kalman predicted state (statsmodels)
    pred = np.asarray(uc.predicted_state[0])[: len(y)]
    d = rg.mu[100:] - pred[100:]
    print(f"[robust_filter] path vs exact Kalman (post burn-in) "
          f"rmse={np.sqrt(np.mean(d ** 2)):.5f}")
    assert np.sqrt(np.mean(d ** 2)) < 0.02   # observed ~0.0013

    # DCS-t on clean Gaussian data collapses to the Gaussian filter
    dt = rt.mu - rg.mu
    assert np.sqrt(np.mean(dt ** 2)) < 0.05  # observed ~0.002
    assert rt.nu > 30.0                      # observed: hits upper bound 200

    # cross-check against tsecon's fit-free exact-diffuse local level at
    # the UC-MLE variances: predictive DCS level ~ lagged filtered level
    import tsecon

    ts = tsecon.local_level_smooth(y, s2eps, s2eta)
    f = np.asarray(ts["filtered_state"])
    d2 = rg.mu[101:] - f[100:-1]
    print(f"[robust_filter] path vs tsecon.local_level_smooth filtered "
          f"rmse={np.sqrt(np.mean(d2 ** 2)):.5f}")
    assert np.sqrt(np.mean(d2 ** 2)) < 0.02  # observed ~0.0014


# ------------------------------------------------------------- AL-ARIMA

def _recovery(innov, reps, T=400, phi0=0.6, th0=0.3, seed0=123, df=2.5):
    out = {}
    for k in ("laplace", "gaussian"):
        est = np.empty((reps, 2))
        for r in range(reps):
            y = simulate_arma(T, phi0, th0, innov=innov, df=df,
                              seed=seed0 + r)
            f = fit_arma_css(y, 1, 1, innovations=k)
            est[r] = [f.phi[0], f.theta[0]]
        rmse = np.sqrt(np.mean((est - [phi0, th0]) ** 2, axis=0))
        out[k] = {"phi": rmse[0], "theta": rmse[1],
                  "joint": float(np.sqrt(np.mean((est - [phi0, th0]) ** 2)))}
    return out


def test_al_arima_recovery_heavy_tails():
    """Laplace-CSS (LAD) beats Gaussian-CSS under t(2.5) and Laplace
    innovations; Gaussian keeps its edge under Gaussian innovations."""
    res_t = _recovery("t", reps=30, seed0=123)
    res_l = _recovery("laplace", reps=30, seed0=555)
    res_g = _recovery("gaussian", reps=20, seed0=999)

    def line(tag, r):
        print(f"[al_arima] {tag}: "
              f"lap rmse(phi,theta)=({r['laplace']['phi']:.4f},"
              f"{r['laplace']['theta']:.4f})  "
              f"gauss=({r['gaussian']['phi']:.4f},"
              f"{r['gaussian']['theta']:.4f})  joint ratio "
              f"lap/gauss={r['laplace']['joint'] / r['gaussian']['joint']:.3f}")

    print()
    line("t(2.5) innov ", res_t)
    line("laplace innov", res_l)
    line("gauss innov  ", res_g)
    # observed joint ratios: 0.688 under t(2.5), 0.751 under laplace,
    # 1.304 under gaussian (ARE of LAD = pi/2 in variance -> ~1.25 in
    # RMSE asymptotically; 1.304 at T=400 with 20 reps is consistent)
    assert res_t["laplace"]["joint"] < 0.85 * res_t["gaussian"]["joint"]
    assert res_l["laplace"]["joint"] < res_l["gaussian"]["joint"]
    assert res_g["laplace"]["joint"] < 1.6 * res_g["gaussian"]["joint"]


def test_al_arima_gaussian_css_matches_statsmodels():
    from statsmodels.tsa.arima.model import ARIMA

    y = simulate_arma(600, 0.6, 0.3, innov="gaussian", seed=42)
    f = fit_arma_css(y, 1, 1, innovations="gaussian")
    sm_fit = ARIMA(y, order=(1, 0, 1)).fit()
    phi_sm = float(sm_fit.arparams[0])
    th_sm = float(sm_fit.maparams[0])
    print(f"\n[al_arima] gaussian CSS vs statsmodels exact MLE: "
          f"phi {f.phi[0]:.4f} vs {phi_sm:.4f}, "
          f"theta {f.theta[0]:.4f} vs {th_sm:.4f}")
    assert abs(f.phi[0] - phi_sm) < 0.08
    assert abs(f.theta[0] - th_sm) < 0.10


def test_al_arima_laplace_is_mle_under_laplace():
    """Under Laplace innovations the LAD fit's profile loglik must beat
    the Gaussian fit's Laplace-evaluated loglik on its own residuals."""
    y = simulate_arma(800, 0.6, 0.3, innov="laplace", seed=77)
    fl = fit_arma_css(y, 1, 1, innovations="laplace")
    fg = fit_arma_css(y, 1, 1, innovations="gaussian")
    eg = fg.resid[1:]
    b_g = np.mean(np.abs(eg))
    ll_g_as_laplace = -eg.size * (np.log(2.0 * b_g) + 1.0)
    print(f"\n[al_arima] laplace loglik: LAD fit={fl.loglik:.2f} "
          f"gaussian-fit residuals={ll_g_as_laplace:.2f}")
    assert fl.loglik >= ll_g_as_laplace - 1e-6
    assert abs(fl.phi[0] - 0.6) < 0.1 and abs(fl.theta[0] - 0.3) < 0.12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
