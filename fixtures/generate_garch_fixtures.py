"""GARCH golden fixtures from Kevin Sheppard's `arch` package.

Run with the project venv (arch is installed there):
    .venv/bin/python fixtures/generate_garch_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np
import arch
from arch import arch_model

OUT = Path(__file__).parent
META = {"arch": arch.__version__, "numpy": np.__version__, "python": platform.python_version()}

# Simulated GARCH(1,1) returns: omega=0.05, alpha=0.08, beta=0.90 (near-IGARCH,
# realistic persistence), zero mean, normal innovations.
rng = np.random.default_rng(2024)
n = 2000
omega, alpha, beta = 0.05, 0.08, 0.90
sig2 = np.empty(n + 500)
r = np.empty(n + 500)
sig2[0] = omega / (1 - alpha - beta)
e = rng.standard_normal(n + 500)
for t in range(1, n + 500):
    sig2[t] = omega + alpha * r[t - 1] ** 2 + beta * sig2[t - 1] if t > 0 else sig2[0]
    r[t] = np.sqrt(sig2[t]) * e[t]
r[0] = np.sqrt(sig2[0]) * e[0]
ret = r[500:]


def fixed_and_fit(am, fixed_params, name):
    fr = am.fix(fixed_params)
    res = am.fit(disp="off")
    rob = am.fit(disp="off", cov_type="robust")
    return {
        "name": name,
        "fixed_params": list(map(float, fixed_params)),
        "loglike_fixed": float(fr.loglikelihood),
        "fit_params": {k: float(v) for k, v in res.params.items()},
        "fit_loglike": float(res.loglikelihood),
        "fit_bse_mle": {k: float(v) for k, v in res.std_err.items()},
        "fit_bse_robust": {k: float(v) for k, v in rob.std_err.items()},
        "conditional_volatility_first5": res.conditional_volatility[:5].tolist(),
        "conditional_volatility_last5": res.conditional_volatility[-5:].tolist(),
    }


cases = []
am = arch_model(ret, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
cases.append(fixed_and_fit(am, [0.05, 0.08, 0.90], "garch11_zero_normal"))

am = arch_model(ret, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
cases.append(fixed_and_fit(am, [0.01, 0.05, 0.08, 0.90], "garch11_const_normal"))

am = arch_model(ret, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="normal", rescale=False)
cases.append(fixed_and_fit(am, [0.05, 0.05, 0.06, 0.90], "gjr111_zero_normal"))

am = arch_model(ret, mean="Zero", vol="EGARCH", p=1, o=1, q=1, dist="normal", rescale=False)
cases.append(fixed_and_fit(am, [0.01, 0.10, -0.05, 0.98], "egarch111_zero_normal"))

am = arch_model(ret, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False)
cases.append(fixed_and_fit(am, [0.05, 0.08, 0.90, 8.0], "garch11_zero_t"))

out = {"_meta": META, "returns": ret.tolist(), "cases": cases}
path = OUT / "garch.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {path} ({path.stat().st_size} bytes)")
