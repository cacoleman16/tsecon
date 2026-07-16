"""Generate golden-value fixtures from NumPy/SciPy/statsmodels.

Run:  python3 fixtures/generate_fixtures.py
Rust tests load these JSON files and must match to the stated tolerances.
Regenerating requires the pinned reference versions recorded in each file.
"""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import scipy.stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import acf, pacf, levinson_durbin

OUT = Path(__file__).parent
META = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
}


def dump(name: str, obj: dict) -> None:
    obj = {"_meta": META, **obj}
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1))
    print(f"wrote {path} ({path.stat().st_size} bytes)")


# ---------------------------------------------------------------- RNG
def gen_philox():
    cases = []
    for seed in [0, 42, 20260716]:
        bg = np.random.Philox(seed)
        raw = bg.random_raw(16).tolist()
        gen = np.random.Generator(np.random.Philox(seed))
        uniforms = gen.random(8).tolist()
        cases.append({"seed": seed, "raw_uint64": [str(x) for x in raw], "uniform_f64": uniforms})

    # Explicit key/counter: exercises the bare Philox4x32-10 engine without seeding logic.
    explicit = []
    for key, counter in [(0, 0), (0xDEADBEEF, 0), (123456789, 987654321)]:
        bg = np.random.Philox(key=key, counter=counter)
        explicit.append(
            {
                "key": str(key),
                "counter": str(counter),
                "raw_uint64": [str(x) for x in bg.random_raw(12)],
            }
        )

    # SeedSequence: entropy -> pool -> generated state words, plus spawning.
    seedseq = []
    for entropy in [0, 42, 20260716]:
        ss = np.random.SeedSequence(entropy)
        children = ss.spawn(3)
        seedseq.append(
            {
                "entropy": entropy,
                "state_uint32_8": [str(x) for x in ss.generate_state(8, np.uint32)],
                "state_uint64_4": [str(x) for x in ss.generate_state(4, np.uint64)],
                "children_state_uint32_4": [
                    [str(x) for x in c.generate_state(4, np.uint32)] for c in children
                ],
            }
        )

    dump("philox.json", {"seeded": cases, "explicit_key_counter": explicit, "seed_sequence": seedseq})


# ------------------------------------------------------- distributions
def gen_distributions():
    x = [-3.0, -1.5, -0.5, 0.0, 0.3, 1.0, 2.5]
    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    def pack(dist):
        return {
            "x": x,
            "pdf": dist.pdf(x).tolist(),
            "logpdf": dist.logpdf(x).tolist(),
            "cdf": dist.cdf(x).tolist(),
            "q": q,
            "ppf": dist.ppf(q).tolist(),
        }

    out = {
        "std_normal": pack(scipy.stats.norm()),
        "student_t": {"df": 5.0, **pack(scipy.stats.t(5.0))},
        "student_t_frac_df": {"df": 4.3, **pack(scipy.stats.t(4.3))},
        # scipy gennorm(beta): pdf = beta/(2*Gamma(1/beta)) * exp(-|x|^beta)
        "ged_gennorm": {"beta": 1.5, **pack(scipy.stats.gennorm(1.5))},
        "special_functions": {
            "erf_x": x,
            "erf": scipy.special.erf(x).tolist(),
            "lgamma_x": [0.5, 1.0, 2.5, 7.3, 21.0],
            "lgamma": scipy.special.gammaln([0.5, 1.0, 2.5, 7.3, 21.0]).tolist(),
            "betainc_args": [[2.0, 3.0, 0.4], [0.5, 0.5, 0.7], [5.0, 1.5, 0.2]],
            "betainc": [
                float(scipy.special.betainc(a, b, xx)) for a, b, xx in [[2.0, 3.0, 0.4], [0.5, 0.5, 0.7], [5.0, 1.5, 0.2]]
            ],
        },
    }
    dump("distributions.json", out)


# ---------------------------------------------------------- diagnostics
def nile_series() -> np.ndarray:
    return sm.datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)


def gen_diagnostics():
    y = nile_series()
    lb = acorr_ljungbox(y - y.mean(), lags=list(range(1, 11)), boxpierce=True)
    arch_stat, arch_p, _, _ = het_arch(y - y.mean(), nlags=4)
    jb_stat, jb_p, skew, kurt = sm.stats.stattools.jarque_bera(np.diff(y))
    acov = acf(y, nlags=10, fft=False, adjusted=False) * np.var(y)
    sigma_v, ar_coefs, pacf_ld, _, _ = levinson_durbin(y, nlags=10, isacov=False)
    out = {
        "nile": y.tolist(),
        "acf_20_unadjusted": acf(y, nlags=20, fft=False, adjusted=False).tolist(),
        "acf_20_adjusted": acf(y, nlags=20, fft=False, adjusted=True).tolist(),
        "pacf_20_ywm": pacf(y, nlags=20, method="ywm").tolist(),
        "pacf_20_ols": pacf(y, nlags=20, method="ols").tolist(),
        "ljung_box_lags_1_10": {
            "lb_stat": lb["lb_stat"].tolist(),
            "lb_pvalue": lb["lb_pvalue"].tolist(),
            "bp_stat": lb["bp_stat"].tolist(),
            "bp_pvalue": lb["bp_pvalue"].tolist(),
        },
        "arch_lm_4": {"lm_stat": float(arch_stat), "lm_pvalue": float(arch_p)},
        "jarque_bera_on_diff": {
            "stat": float(jb_stat),
            "pvalue": float(jb_p),
            "skew": float(skew),
            "kurtosis": float(kurt),
        },
        "levinson_durbin_10": {
            "ar_coefs": np.asarray(ar_coefs).tolist(),
            "pacf": np.asarray(pacf_ld).tolist(),
            "sigma2_final": float(sigma_v),
        },
    }
    dump("diagnostics.json", out)


# ------------------------------------------------------------- linalg
def gen_linalg():
    rng = np.random.default_rng(7)
    a2 = np.array([[0.7, 0.2], [-0.1, 0.5]])
    q2 = np.array([[1.0, 0.3], [0.3, 2.0]])
    m = rng.standard_normal((4, 4)) * 0.4
    a4 = m * 0.9 / max(abs(np.linalg.eigvals(m)))
    q4r = rng.standard_normal((4, 4))
    q4 = q4r @ q4r.T + np.eye(4)

    # Toeplitz solve: SPD autocovariance-like column.
    col = np.array([4.0, 2.4, 1.44, 0.864, 0.5184])
    b = np.array([1.0, -0.5, 2.0, 0.0, 1.5])
    out = {
        "discrete_lyapunov": [
            {
                "a": a2.tolist(),
                "q": q2.tolist(),
                "x": scipy.linalg.solve_discrete_lyapunov(a2, q2).tolist(),
            },
            {
                "a": a4.tolist(),
                "q": q4.tolist(),
                "x": scipy.linalg.solve_discrete_lyapunov(a4, q4).tolist(),
            },
        ],
        "toeplitz_solve": {
            "first_col": col.tolist(),
            "rhs": b.tolist(),
            "x": scipy.linalg.solve_toeplitz(col, b).tolist(),
        },
    }
    dump("linalg.json", out)


# ----------------------------------------------------------------- ssm
def gen_ssm():
    y = nile_series()
    sigma2_eps, sigma2_eta = 15099.0, 1469.1  # Durbin-Koopman (2012) MLEs

    def local_level(endog, exact_diffuse):
        mod = sm.tsa.UnobservedComponents(
            endog, level="llevel", use_exact_diffuse=exact_diffuse
        )
        params = np.array([sigma2_eps, sigma2_eta])
        res = mod.smooth(params)
        return mod, res

    _, res_exact = local_level(y, True)
    _, res_approx = local_level(y, False)

    y_missing = y.copy()
    y_missing[20:40] = np.nan
    _, res_missing = local_level(y_missing, True)

    # AR(2) with constant via SARIMAX at fixed parameters, stationary init.
    rng = np.random.default_rng(20260716)
    n = 250
    e = rng.standard_normal(n + 100)
    ar = np.empty(n + 100)
    ar[:2] = 0.0
    c, a1, a2, s2 = 1.5, 0.6, -0.2, 1.44
    for t in range(2, n + 100):
        ar[t] = c + a1 * ar[t - 1] + a2 * ar[t - 2] + np.sqrt(s2) * e[t]
    yar = ar[100:]
    smod = sm.tsa.SARIMAX(yar, order=(2, 0, 0), trend="c")
    ar_params = np.array([c, a1, a2, s2])

    out = {
        "nile": y.tolist(),
        "local_level_params": {"sigma2_eps": sigma2_eps, "sigma2_eta": sigma2_eta},
        "local_level_exact_diffuse": {
            "loglike": float(res_exact.llf),
            "filtered_state": res_exact.filtered_state[0].tolist(),
            "filtered_state_cov": res_exact.filtered_state_cov[0, 0].tolist(),
            "smoothed_state": res_exact.smoothed_state[0].tolist(),
            "smoothed_state_cov": res_exact.smoothed_state_cov[0, 0].tolist(),
        },
        "local_level_approx_diffuse_kappa1e6": {
            "loglike_burned1": float(res_approx.llf),
        },
        "local_level_missing_20_40_exact_diffuse": {
            "y": y_missing.tolist(),
            "loglike": float(res_missing.llf),
            "smoothed_state": res_missing.smoothed_state[0].tolist(),
        },
        "ar2_sarimax": {
            "y": yar.tolist(),
            "params_const_ar1_ar2_sigma2": ar_params.tolist(),
            "loglike": float(smod.loglike(ar_params)),
        },
    }
    dump("ssm.json", out)


# ------------------------------------------------------------ unit roots
def gen_unitroot():
    import warnings

    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.stattools import adfuller, kpss

    y = nile_series()
    rw = np.cumsum(np.random.default_rng(11).standard_normal(250)) + 5.0

    def adf_case(series, name, regression, autolag=None, maxlag=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = adfuller(series, maxlag=maxlag, regression=regression, autolag=autolag)
        stat, pval, usedlag, nobs, crit = res[0], res[1], res[2], res[3], res[4]
        return {
            "series": name,
            "regression": regression,
            "autolag": autolag,
            "maxlag": maxlag,
            "stat": float(stat),
            "pvalue": float(pval),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()},
        }

    adf_cases = [
        adf_case(y, "nile", "c", autolag="AIC"),
        adf_case(y, "nile", "ct", autolag="AIC"),
        adf_case(y, "nile", "n", autolag="AIC"),
        adf_case(y, "nile", "c", autolag="BIC"),
        adf_case(y, "nile", "c", autolag="t-stat"),
        adf_case(y, "nile", "c", autolag=None, maxlag=4),
        adf_case(rw, "rw", "c", autolag="AIC"),
        adf_case(rw, "rw", "ct", autolag="AIC"),
    ]

    kpss_cases = []
    for series, name in [(y, "nile"), (rw, "rw")]:
        for regression in ["c", "ct"]:
            for nlags in ["auto", "legacy"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, pval, lags, crit = kpss(series, regression=regression, nlags=nlags)
                kpss_cases.append(
                    {
                        "series": name,
                        "regression": regression,
                        "nlags": nlags,
                        "stat": float(stat),
                        "pvalue_interpolated_bounded": float(pval),
                        "lags": int(lags),
                        "crit": {k: float(v) for k, v in crit.items()},
                    }
                )

    # MacKinnon p-value surface: pin a grid so the Rust port is verifiable.
    grid = np.arange(-6.0, 3.01, 0.25)
    mackinnon = {
        reg: {
            "stat_grid": grid.tolist(),
            "pvalues": [float(mackinnonp(s, regression=reg, N=1)) for s in grid],
            "crit_1_5_10": [float(c) for c in mackinnoncrit(N=1, regression=reg, nobs=np.inf)],
        }
        for reg in ["n", "c", "ct"]
    }

    dump(
        "unitroot.json",
        {"nile": y.tolist(), "rw": rw.tolist(), "adf": adf_cases, "kpss": kpss_cases,
         "mackinnon_p_N1": mackinnon},
    )


# ------------------------------------------------------------------ HAC
def gen_hac():
    rng = np.random.default_rng(3)
    n = 200
    x1 = np.empty(n)
    x2 = np.empty(n)
    u = np.empty(n)
    x1[0], x2[0], u[0] = 0.0, 0.0, 0.0
    e = rng.standard_normal((3, n))
    for t in range(1, n):
        x1[t] = 0.7 * x1[t - 1] + e[0, t]
        x2[t] = 0.5 * x2[t - 1] + e[1, t]
        u[t] = 0.6 * u[t - 1] + e[2, t]
    yy = 1.0 + 0.5 * x1 - 0.3 * x2 + u
    X = sm.add_constant(np.column_stack([x1, x2]))
    ols = sm.OLS(yy, X).fit()

    cases = []
    for nlags in [4, 8, 12]:
        for correction in [True, False]:
            r = sm.OLS(yy, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": nlags, "use_correction": correction}
            )
            cases.append(
                {
                    "maxlags": nlags,
                    "use_correction": correction,
                    "bse": r.bse.tolist(),
                    "tvalues": r.tvalues.tolist(),
                }
            )

    # Long-run variance fixtures on the demeaned Nile (self-generated formulas,
    # documented: Bartlett LRV and the LLSW-2018 equal-weighted cosine (EWC) LRV).
    y = nile_series()
    z = y - y.mean()
    nn = len(z)

    def gamma(k):
        return float(z[: nn - k] @ z[k:] / nn)

    def bartlett_lrv(bw):
        return gamma(0) + 2 * sum((1 - j / (bw + 1)) * gamma(j) for j in range(1, bw + 1))

    tgrid = (np.arange(1, nn + 1) - 0.5) / nn
    def ewc_lrv(B):
        lam = [np.sqrt(2.0 / nn) * float(np.cos(np.pi * j * tgrid) @ z) for j in range(1, B + 1)]
        return float(np.mean(np.square(lam)))

    dump(
        "hac.json",
        {
            "regression": {
                "y": yy.tolist(),
                "x1": x1.tolist(),
                "x2": x2.tolist(),
                "ols_params": ols.params.tolist(),
                "ols_bse_nonrobust": ols.bse.tolist(),
                "hac_cases": cases,
            },
            "lrv_nile_demeaned": {
                "bartlett": {str(bw): bartlett_lrv(bw) for bw in [5, 10, 20]},
                "ewc": {str(B): ewc_lrv(B) for B in [4, 8, 16]},
                "newey_west_auto_maxlags_floor_4_n100_2_9": int(np.floor(4 * (nn / 100) ** (2 / 9))),
            },
        },
    )


# ---------------------------------------------------------------- ARIMA
def gen_arima():
    import warnings

    rng = np.random.default_rng(101)
    n = 300
    e = rng.standard_normal(n + 100) * np.sqrt(1.2)
    z = np.empty(n + 100)
    z[0] = 0.0
    for t in range(1, n + 100):
        z[t] = 0.7 * z[t - 1] + e[t] + 0.4 * e[t - 1]
    arma = z[100:] + 2.0  # ARMA(1,1) around a nonzero mean

    y = nile_series()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fixed-parameter log-likelihoods (deterministic golden values).
        m1 = sm.tsa.SARIMAX(arma - 2.0, order=(1, 0, 1), trend="n")
        ll_arma_fixed = float(m1.loglike(np.array([0.7, 0.4, 1.2])))

        m2 = sm.tsa.SARIMAX(y, order=(1, 1, 1), trend="n", simple_differencing=True)
        ll_arima_fixed = float(m2.loglike(np.array([0.3, -0.6, 20000.0])))

        # A full MLE fit as an optimizer target (match loglik, not the path).
        m3 = sm.tsa.SARIMAX(y, order=(1, 0, 1), trend="c")
        r3 = m3.fit(disp=False)

        # Fixed-parameter forecasting golden.
        fc = m3.smooth(r3.params).get_forecast(12)

    out = {
        "arma11": {
            "y": arma.tolist(),
            "note": "ARMA(1,1), phi=0.7, theta=0.4, sigma2=1.2, mean 2.0 added after simulation",
            "loglike_fixed_demeaned": ll_arma_fixed,
            "fixed_params_phi_theta_sigma2": [0.7, 0.4, 1.2],
        },
        "nile_arima111_simple_diff": {
            "fixed_params_phi_theta_sigma2": [0.3, -0.6, 20000.0],
            "loglike_fixed": ll_arima_fixed,
        },
        "nile_arma11c_fit": {
            "params_const_phi_theta_sigma2": r3.params.tolist(),
            "loglike": float(r3.llf),
            "aic": float(r3.aic),
            "bic": float(r3.bic),
            "forecast_mean_12": fc.predicted_mean.tolist(),
            "forecast_se_12": fc.se_mean.tolist(),
        },
    }
    dump("arima.json", out)


# ------------------------------------------------------------------ VAR
def gen_var():
    import warnings

    from statsmodels.tsa.api import VAR

    mac = sm.datasets.macrodata.load_pandas().data
    data = 100.0 * np.diff(np.log(mac[["realgdp", "realcons", "realinv"]].to_numpy()), axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = VAR(data)
        res = model.fit(2, trend="c")
        sel = model.select_order(8)
        irf = res.irf(10)
        fevd = res.fevd(10)
        gc = res.test_causality(0, [1], kind="f")  # does realcons Granger-cause realgdp
        point = res.forecast(data[-2:], 8)
        _, lower, upper = res.forecast_interval(data[-2:], 8, alpha=0.05)

    out = {
        "data_100dlog_gdp_cons_inv": data.tolist(),
        "var2c": {
            "params": res.params.tolist(),
            "param_names": list(res.params.index) if hasattr(res.params, "index") else None,
            "sigma_u": np.asarray(res.sigma_u).tolist(),
            "llf": float(res.llf),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "hqic": float(res.hqic),
            "fpe": float(res.fpe),
            "stable_max_root": float(max(abs(res.roots))),
        },
        "lag_selection_maxlags_8": {
            "aic": int(sel.aic),
            "bic": int(sel.bic),
            "hqic": int(sel.hqic),
            "fpe": int(sel.fpe),
        },
        "granger_cons_causes_gdp": {
            "stat": float(gc.test_statistic),
            "pvalue": float(gc.pvalue),
            "df": list(gc.df) if hasattr(gc, "df") and not np.isscalar(gc.df) else gc.df,
        },
        "irf_orth_h10": np.asarray(irf.orth_irfs).tolist(),
        "irf_nonorth_h10": np.asarray(irf.irfs).tolist(),
        "fevd_h10": np.asarray(fevd.decomp).tolist(),
        "forecast_8": {
            "point": np.asarray(point).tolist(),
            "lower95": np.asarray(lower).tolist(),
            "upper95": np.asarray(upper).tolist(),
        },
    }
    dump("var.json", out)


# -------------------------------------------------------------- filters
def gen_filters():
    mac = sm.datasets.macrodata.load_pandas().data
    y = 100.0 * np.log(mac["realgdp"].to_numpy())

    cycle_hp, trend_hp = sm.tsa.filters.hpfilter(y, lamb=1600.0)
    bk = sm.tsa.filters.bkfilter(y, low=6, high=32, K=12)
    cf_cycle, cf_trend = sm.tsa.filters.cffilter(y, low=6, high=32, drift=True)

    # Hamilton (2018) regression filter, h=8, p=4: regress y_{t} on
    # [1, y_{t-8}, y_{t-9}, y_{t-10}, y_{t-11}]; cycle = residual.
    h, p = 8, 4
    n = len(y)
    rows = range(h + p - 1, n)
    X = np.column_stack(
        [np.ones(len(rows))] + [y[[t - h - j for t in rows]] for j in range(p)]
    )
    yy = y[list(rows)]
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    hamilton_cycle = yy - X @ beta

    dump(
        "filters.json",
        {
            "y_100_log_realgdp": y.tolist(),
            "hp_1600": {"cycle": cycle_hp.tolist(), "trend": trend_hp.tolist()},
            "bk_6_32_K12": np.asarray(bk).ravel().tolist(),
            "cf_6_32_drift": {"cycle": np.asarray(cf_cycle).tolist(), "trend": np.asarray(cf_trend).tolist()},
            "hamilton_h8_p4": {
                "beta": beta.tolist(),
                "cycle": hamilton_cycle.tolist(),
                "first_cycle_index": h + p - 1,
            },
        },
    )


# ---------------------------------------------------------- forecasting
def gen_forecast():
    import warnings

    from statsmodels.tsa.forecasting.theta import ThetaModel

    mac = sm.datasets.macrodata.load_pandas().data
    y = mac["realgdp"].to_numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tm = ThetaModel(y, period=4, deseasonalize=True, use_test=False)
        tr = tm.fit()
        theta_fc = tr.forecast(8).to_numpy()

    # Diebold-Mariano with the Harvey-Leybourne-Newbold (1997) correction —
    # self-authored reference implementation, formulas documented here.
    rng = np.random.default_rng(55)
    n, hstep = 120, 3
    e1 = rng.standard_normal(n)
    e2 = 0.8 * e1 + 0.6 * rng.standard_normal(n) + 0.15
    d = e1**2 - e2**2  # squared-error loss differential
    dbar = d.mean()
    # HAC variance of dbar with rectangular (uniform) weights to h-1 lags
    gam = [np.mean((d[: n - k] - dbar) * (d[k:] - dbar)) for k in range(hstep)]
    var_dbar = (gam[0] + 2 * sum(gam[1:])) / n
    dm = dbar / np.sqrt(var_dbar)
    hln = dm * np.sqrt((n + 1 - 2 * hstep + hstep * (hstep - 1) / n) / n)
    from scipy import stats as sps

    dump(
        "forecast.json",
        {
            "theta_realgdp_p4": {"forecast_8": theta_fc.tolist(), "note": "statsmodels ThetaModel, deseasonalize=True, use_test=False"},
            "dm_test": {
                "e1": e1.tolist(),
                "e2": e2.tolist(),
                "h": hstep,
                "loss": "squared",
                "dm_stat": float(dm),
                "hln_stat": float(hln),
                "hln_pvalue_t_nminus1": float(2 * sps.t.sf(abs(hln), df=n - 1)),
            },
            "accuracy_small": {
                "actual": [10.0, 12.0, 9.0, 14.0, 11.0, 13.0],
                "forecast": [11.0, 11.5, 10.0, 12.5, 11.0, 14.0],
                "insample_for_mase": [8.0, 9.5, 9.0, 10.5, 10.0, 11.5, 11.0, 12.0],
                "mase_denominator": "mean absolute first difference of insample",
            },
        },
    )


if __name__ == "__main__":
    gen_philox()
    gen_distributions()
    gen_diagnostics()
    gen_linalg()
    gen_ssm()
    gen_unitroot()
    gen_hac()
    gen_arima()
    gen_var()
    gen_filters()
    gen_forecast()
