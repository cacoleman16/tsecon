"""Golden fixtures for conditional-sum-of-squares ARIMA estimation,
`arima_fit(method="css")` (crate `tsecon-arima`: `ArimaSpec::fit_css` and
`ArimaSpec::css_residuals`).

THE tsecon CSS CONVENTION (crates/tsecon-arima/src/estimate.rs)
----------------------------------------------------------------
x is the fully differenced series (seasonal difference first), phi* / theta*
the multiplied-out polynomials phi(L)Phi(L^s), theta(L)Theta(L^s), c an
INTERCEPT, and p* = p + s P the conditioning count:

    e_t = x_t - c - sum_i phi*_i x_{t-i} - sum_j theta*_j e_{t-j},  t = p*..n_x-1,
    e_s = 0 for s < p*,   sigma2 = SSR / n_c,   n_c = n_x - p* = n - d - D s - p*,
    loglik = -n_c/2 (ln 2 pi + 1 + ln sigma2),
    AIC = -2 loglik + 2 k,  BIC = -2 loglik + k ln n_c   (k counts sigma2).

THREE LEGS
----------
(A) Optimizer-free, third-party: statsforecast 2.1.1 (Nixtla, Apache-2.0)
    `statsforecast.arima.arima_css(x, arma, phi, theta)`, its C++ port of R's
    stats:::ARIMA_CSS, evaluated at FIXED parameters — at the leg-C optimum and
    at a perturbed interior point — returns (sigma2, residuals). R and
    statsforecast use the MEAN form: the series passed is x - mu with
    mu = c / (1 - sum phi*), and for d >= 1 the UNDIFFERENCED series with d in
    `arma` (R differences inside ARIMA_CSS; its ncond = d + D s + p + s P, its
    residuals are 0 at those positions). The expanded phi*/theta* are computed
    here with numpy.convolve and asserted equal to statsforecast's own
    `arima_transpar` expansion. Pins the recursion, SSR, sigma2 and n_c.
(B) AR(2)+c: statsmodels 0.15.0 `AutoReg(y, lags=2, trend="c")` — OLS, which IS
    the CSS optimum (params, sigma2 = ssr / nobs, llf).
(C) The optimum for ARMA(1,1)+c, SARIMA(1,0,0)(1,0,0)_4+c, MA(1) without a
    constant and ARIMA(1,1,1) without a constant is OPTIMIZER-DEFINED (no
    closed form): `scipy.optimize.least_squares(method="lm", xtol = ftol = gtol
    = 1e-15)` on scipy.signal.lfilter residuals, cross-checked by BFGS on the
    log-SSR (the spread is stored) and, for the constant cases, by
    statsforecast's own end-to-end `arima(method="CSS")` (its implied intercept
    stored, not gated). loglik/AIC/BIC follow from sigma2. bse: the inverse of
    `statsmodels.tools.numdiff.approx_hess3` of the conditional negative
    log-likelihood with sigma2 FREE, n_c/2 (ln 2 pi + ln s2) + SSR/(2 s2), at the
    LM optimum (tsecon differentiates the same objective).

THE R LOG-LIKELIHOOD CONVENTION DIFFERS (recorded, for the model card):
R / statsforecast report -2 loglik = n.used (ln 2 pi + 1 + ln sigma2) with
n.used = n - d - D s (NOT less ncond) while sigma2 = SSR / (n - ncond); tsecon
uses n_c in both places. R's CSS AIC is NA. `r_convention` stores both numbers.

Optimum DGPs are interior (stationary, invertible): tsecon optimizes inside the
constrained Monahan region while the references are unconstrained.

INSTALL (generator-only dependency, like lpdid's R/fixest): statsforecast
2.1.1 is not part of the reference set and its `__init__` imports fugue,
coreforecast and utilsforecast. Its wheel pins `pandas<3.0.0`; this fixture was
generated with statsforecast installed WITHOUT dependency resolution so the
reference venv keeps pandas 3.0.6 (only the C++ `arima_css`/`arima_transpar`
and the CSS `arima()` path are used; neither touches pandas):

    pip install --no-deps statsforecast==2.1.1 coreforecast==0.0.18 \\
        fugue==0.9.7 triad==1.0.2 adagio==0.2.6 utilsforecast==0.2.17 \\
        fsspec==2026.9.0 pyarrow==25.0.1 tqdm==4.70.1

The generator asserts the version it ran against.

Data: seeded NumPy `default_rng` draws (no dataset). This generator NEVER
imports tsecon. Doubles are written with json's shortest round-trip repr.

Run:  .venv/bin/python fixtures/generate_arima_css_fixtures.py
"""

from __future__ import annotations

import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas
import scipy
import statsmodels
from scipy import optimize, signal
from statsmodels.tools.numdiff import approx_hess3
from statsmodels.tsa.ar_model import AutoReg

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from statsforecast import arima as SF

HERE = Path(__file__).resolve().parent
OUT = HERE / "arima_css.json"
SEED = 20261009
STATSFORECAST = "2.1.1"
assert version("statsforecast") == STATSFORECAST, version("statsforecast")


# ---------------------------------------------------------------- the recursion
def expand_ar(a, big_a, s):
    """phi* with x_t = sum phi*_i x_{t-i} + ..., from (1 - a(L))(1 - A(L^s))."""
    lag = np.r_[1.0, -np.asarray(a, dtype=float)]
    slag = np.zeros(len(big_a) * s + 1)
    slag[0] = 1.0
    for i, v in enumerate(big_a):
        slag[(i + 1) * s] = -v
    return -np.convolve(lag, slag)[1:]


def expand_ma(b, big_b, s):
    lag = np.r_[1.0, np.asarray(b, dtype=float)]
    slag = np.zeros(len(big_b) * s + 1)
    slag[0] = 1.0
    for i, v in enumerate(big_b):
        slag[(i + 1) * s] = v
    return np.convolve(lag, slag)[1:]


def difference(y, d, big_d, s):
    x = np.asarray(y, dtype=float)
    for _ in range(big_d):
        x = x[s:] - x[:-s]
    for _ in range(d):
        x = np.diff(x)
    return x


def css_resid(x, c, ar, ma):
    p = len(ar)
    n = len(x)
    w = x[p:] - c
    for i in range(p):
        w = w - ar[i] * x[p - 1 - i:n - 1 - i]
    return signal.lfilter([1.0], np.r_[1.0, ma], w) if len(ma) else w


class Spec:
    def __init__(self, name, p, d, q, seasonal, constant):
        self.name = name
        self.p, self.d, self.q = p, d, q
        self.big_p, self.big_d, self.big_q, self.s = seasonal
        self.constant = constant

    @property
    def k_mean(self):
        return int(self.constant) + self.p + self.q + self.big_p + self.big_q

    def names(self):
        out = ["const"] if self.constant else []
        out += [f"ar.L{i}" for i in range(1, self.p + 1)]
        out += [f"ma.L{j}" for j in range(1, self.q + 1)]
        out += [f"ar.S.L{i * self.s}" for i in range(1, self.big_p + 1)]
        out += [f"ma.S.L{j * self.s}" for j in range(1, self.big_q + 1)]
        return out + ["sigma2"]

    def unpack(self, th):
        th = np.asarray(th, dtype=float)
        at = 0
        c = 0.0
        if self.constant:
            c = th[0]
            at = 1
        a = th[at:at + self.p]
        at += self.p
        b = th[at:at + self.q]
        at += self.q
        big_a = th[at:at + self.big_p]
        at += self.big_p
        big_b = th[at:at + self.big_q]
        s = max(self.s, 1)
        return c, expand_ar(a, big_a, s), expand_ma(b, big_b, s)

    def arma(self):
        # R / statsforecast: (p, q, P, Q, period, d, D)
        return np.array([self.p, self.q, self.big_p, self.big_q, max(self.s, 1), self.d, self.big_d],
                        dtype=np.int32)

    def residuals(self, y, th):
        x = difference(y, self.d, self.big_d, self.s)
        c, ar, ma = self.unpack(th)
        return css_resid(x, c, ar, ma)


# ---------------------------------------------------------------- DGPs
def sim_arma(rng, n, c, ar, ma, sd, burn=300):
    e = rng.standard_normal(n + burn) * sd
    x = np.zeros(n + burn)
    for t in range(n + burn):
        v = c + e[t]
        for i, a in enumerate(ar):
            if t - 1 - i >= 0:
                v += a * x[t - 1 - i]
        for j, b in enumerate(ma):
            if t - 1 - j >= 0:
                v += b * e[t - 1 - j]
        x[t] = v
    return x[burn:]


def datasets(rng):
    out = {}
    out["ar2_c"] = sim_arma(rng, 300, 0.4, [0.5, -0.3], [], 1.1)
    out["arma11_c"] = sim_arma(rng, 400, 0.5, [0.7], [0.4], 1.2)
    out["sarima100_100_4_c"] = sim_arma(rng, 400, 0.3, list(expand_ar([0.5], [0.4], 4)), [], 1.0)
    out["ma1"] = sim_arma(rng, 300, 0.0, [], [0.6], 1.0)
    w = sim_arma(rng, 300, 0.0, [0.5], [0.3], 1.0)
    out["arima111"] = np.cumsum(w)
    return out


SPECS = {
    "arma11_c": Spec("arma11_c", 1, 0, 1, (0, 0, 0, 0), True),
    "sarima100_100_4_c": Spec("sarima100_100_4_c", 1, 0, 0, (1, 0, 0, 4), True),
    "ma1": Spec("ma1", 0, 0, 1, (0, 0, 0, 0), False),
    "arima111": Spec("arima111", 1, 1, 1, (0, 0, 0, 0), False),
}


# ---------------------------------------------------------------- leg C
def fit_lm(spec, y):
    x0 = np.full(spec.k_mean, 0.1)
    ls = optimize.least_squares(lambda th: spec.residuals(y, th), x0, method="lm",
                                xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=200000)
    f = lambda th: np.log(np.sum(spec.residuals(y, th) ** 2))
    bf = optimize.minimize(f, x0, method="BFGS", options={"gtol": 1e-12, "maxiter": 20000})
    e = spec.residuals(y, ls.x)
    n_c = len(e)
    ssr = float(e @ e)
    s2 = ssr / n_c
    ll = -0.5 * n_c * (np.log(2 * np.pi) + 1 + np.log(s2))
    k = spec.k_mean + 1

    def negll(full):
        r = spec.residuals(y, full[:-1])
        return 0.5 * n_c * (np.log(2 * np.pi) + np.log(full[-1])) + r @ r / (2 * full[-1])

    full = np.r_[ls.x, s2]
    hess = approx_hess3(full, negll)
    cov = np.linalg.inv(hess)
    return {
        "params": full.tolist(),
        "sigma2": s2,
        "ssr": ssr,
        "nobs": n_c,
        "loglik": float(ll),
        "aic": float(-2 * ll + 2 * k),
        "bic": float(-2 * ll + k * np.log(n_c)),
        "bse": np.sqrt(np.diag(cov)).tolist(),
        "bfgs_minus_lm_max_abs": float(np.max(np.abs(bf.x - ls.x))),
        "lm_status": int(ls.status),
    }


# ---------------------------------------------------------------- leg A
def sf_eval(spec, y, mean_params):
    c, phi, theta = spec.unpack(mean_params)
    # statsforecast's own expansion agrees with ours (sign conventions too).
    coef = np.asarray(mean_params, dtype=float)[int(spec.constant):].copy()
    phi_sf, theta_sf = SF.arima_transpar(coef, spec.arma(), False)
    assert np.allclose(phi_sf, phi, rtol=0, atol=1e-15) and len(phi_sf) == len(phi), (phi_sf, phi)
    assert np.allclose(theta_sf, theta, rtol=0, atol=1e-15) and len(theta_sf) == len(theta)
    mu = c / (1.0 - phi.sum()) if spec.constant else 0.0
    yy = np.asarray(y, dtype=float) - mu
    s2_sf, resid_sf = SF.arima_css(yy, spec.arma(), phi, theta)
    resid_sf = np.asarray(resid_sf)
    ncond_r = spec.d + spec.big_d * spec.s + spec.p + spec.s * spec.big_p
    assert resid_sf.shape == (len(y),)
    assert np.all(resid_sf[:ncond_r] == 0.0), resid_sf[:ncond_r + 2]
    tail = resid_sf[ncond_r:]
    ours = spec.residuals(y, mean_params)
    return {
        "mean_params": list(map(float, mean_params)),
        "mu": float(mu),
        "arma": spec.arma().tolist(),
        "ncond_r": ncond_r,
        "nobs": len(tail),
        "sigma2": float(s2_sf),
        "ssr": float(s2_sf) * len(tail),
        "residuals": tail.tolist(),
        "numpy_transcription_max_abs_dev": float(np.max(np.abs(ours - tail))),
        "numpy_transcription_sigma2_rel_dev": float(abs(ours @ ours / len(ours) - s2_sf) / s2_sf),
    }


def sf_end_to_end(spec, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SF.arima(np.asarray(y, dtype=float), order=(spec.p, spec.d, spec.q),
                       seasonal={"order": (spec.big_p, spec.big_d, spec.big_q), "period": max(spec.s, 1)},
                       include_mean=spec.constant, method="CSS", tol=1e-12,
                       optim_control={"maxiter": 2000})
    coef = {k: float(v) for k, v in fit["coef"].items()}
    out = {"coef": coef, "sigma2": float(fit["sigma2"]), "loglik_r_convention": float(fit["loglik"]),
           "aic": None if fit["aic"] is None or not np.isfinite(fit["aic"]) else float(fit["aic"])}
    if spec.constant:
        _, phi, _ = spec.unpack(np.r_[0.0, [coef[k] for k in coef if k != "intercept"]])
        out["implied_intercept"] = coef["intercept"] * (1.0 - phi.sum())
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = datasets(rng)

    # Leg B: AR(2)+c, AutoReg OLS == CSS optimum.
    y_ar = data["ar2_c"]
    ar = AutoReg(y_ar, lags=2, trend="c").fit()
    nobs_ar = int(ar.nobs)
    k_ar = 4
    leg_b = {
        "name": "ar2_c",
        "order": [2, 0, 0],
        "seasonal": None,
        "constant": True,
        "y": y_ar.tolist(),
        "param_names": ["const", "ar.L1", "ar.L2", "sigma2"],
        "params": np.r_[ar.params, ar.sigma2].tolist(),
        "sigma2": float(ar.sigma2),
        "loglik": float(ar.llf),
        "nobs": nobs_ar,
        "aic": float(-2 * ar.llf + 2 * k_ar),
        "bic": float(-2 * ar.llf + k_ar * np.log(nobs_ar)),
        "source": "statsmodels AutoReg(lags=2, trend='c').fit(): OLS = the CSS optimum",
    }
    e = css_resid(y_ar, ar.params[0], ar.params[1:], [])
    assert len(e) == nobs_ar and abs(e @ e / nobs_ar - ar.sigma2) < 1e-12 * ar.sigma2
    assert abs(-0.5 * nobs_ar * (np.log(2 * np.pi) + 1 + np.log(ar.sigma2)) - ar.llf) < 1e-9

    cases = []
    for key, spec in SPECS.items():
        y = data[key]
        opt = fit_lm(spec, y)
        mean_opt = opt["params"][:-1]
        perturbed = [v + (0.05 if i % 2 == 0 else -0.04) for i, v in enumerate(mean_opt)]
        legs_a = [dict(sf_eval(spec, y, mean_opt), point="optimum"),
                  dict(sf_eval(spec, y, perturbed), point="perturbed")]
        n = len(y)
        n_used = n - spec.d - spec.big_d * spec.s
        s2 = opt["sigma2"]
        case = {
            "name": key,
            "order": [spec.p, spec.d, spec.q],
            "seasonal": [spec.big_p, spec.big_d, spec.big_q, spec.s] if spec.s else None,
            "constant": spec.constant,
            "y": np.asarray(y).tolist(),
            "param_names": spec.names(),
            "optimum": opt,
            "fixed_point_statsforecast": legs_a,
            "r_convention": {
                "n_used": n_used,
                "loglik_r": float(-0.5 * n_used * (np.log(2 * np.pi) + 1 + np.log(s2))),
                "loglik_tsecon": opt["loglik"],
            },
            "statsforecast_end_to_end": sf_end_to_end(spec, y),
        }
        cases.append(case)

    fixture = {
        "_meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "pandas": pandas.__version__,
            "statsforecast": STATSFORECAST,
            "statsforecast_install": "pip install --no-deps statsforecast==2.1.1 (plus its "
                                     "import-time deps coreforecast 0.0.18, fugue 0.9.7, triad 1.0.2, "
                                     "adagio 0.2.6, utilsforecast 0.2.17, fsspec, pyarrow, tqdm); "
                                     "pandas kept at the reference venv's version",
            "seed": SEED,
            "note": (
                "CSS ARIMA, tsecon intercept convention. Leg A: statsforecast arima_css (C++ "
                "port of R's ARIMA_CSS) at fixed parameters, mean form mu = c/(1 - sum phi*), "
                "residuals after R's ncond = d + D s + p + s P. Leg B: AutoReg OLS = CSS optimum. "
                "Leg C: optimizer-defined optimum (scipy LM, BFGS spread recorded), bse from "
                "approx_hess3 of the conditional loglik with sigma2 free."
            ),
        },
        "ar_autoreg": leg_b,
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(f"  ar2_c AutoReg params {leg_b['params']} llf {leg_b['loglik']!r}")
    for c in cases:
        o = c["optimum"]
        print(f"  {c['name']}: LM {np.round(o['params'], 8).tolist()} loglik {o['loglik']!r} "
              f"n_c {o['nobs']} BFGS-LM {o['bfgs_minus_lm_max_abs']:.1e}")
        for a in c["fixed_point_statsforecast"]:
            print(f"     leg A @{a['point']}: sigma2 {a['sigma2']!r} numpy-vs-sf resid "
                  f"{a['numpy_transcription_max_abs_dev']:.1e} sigma2 rel "
                  f"{a['numpy_transcription_sigma2_rel_dev']:.1e}")
        rc = c["r_convention"]
        print(f"     R loglik {rc['loglik_r']!r} vs tsecon {rc['loglik_tsecon']!r}; "
              f"statsforecast e2e {c['statsforecast_end_to_end']}")


if __name__ == "__main__":
    main()
