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


if __name__ == "__main__":
    gen_philox()
    gen_distributions()
    gen_diagnostics()
    gen_linalg()
    gen_ssm()
