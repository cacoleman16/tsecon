"""unobserved_components(mle_regression=False): the exact-diffuse filter misclassifies a
genuine diffuse observation late in a long diffuse period, so nobs_diffuse is wrong and the
log-likelihood (and states) are silently wrong -- or, when the unit-scaled re-check disagrees,
a well-scaled N(0,1) regressor is refused as "too far apart in scale".

Expected = statsmodels UnobservedComponents(use_exact_diffuse=True) at the same fixed
parameters, confirmed by an extended-precision de Jong augmented Kalman filter
(augkf.py, next to this file: no diffuse thresholds at all).

Run: auditvenv/bin/python repro_uc_state_regression_diffuse.py   (from this directory)
"""
import json, os, sys, warnings
import numpy as np
import tsecon
from statsmodels.tsa.statespace.structural import UnobservedComponents as UC

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augkf import augmented_loglike  # noqa: E402


def compare(label, y, X, spec, params):
    sm_kw = dict(spec)
    mod = UC(y, exog=X, use_exact_diffuse=True, **sm_kw)
    res = mod.smooth(np.asarray(params, float))
    aug = augmented_loglike(mod, np.asarray(params, float), y)[0]
    try:
        r = tsecon.unobserved_components(y, exog=X, fixed_params=list(params), **spec)
        got = f"tsecon loglik {r['loglik']:.6f} nobs_diffuse {r['nobs_diffuse']}"
    except ValueError as ex:
        got = "tsecon REFUSED: " + str(ex)[:260] + " ..."
    print(f"{label}\n  expected: statsmodels loglik {res.llf:.6f} nobs_diffuse {res.nobs_diffuse}; "
          f"augmented-KF (long double) {aug:.6f}\n  actual:   {got}\n")


# (1) BSM with two N(0,1) regressors in the state, round parameters (T = 120)
rng = np.random.default_rng(207)
T = 120
x = rng.normal(size=(T, 2))
y = np.cumsum(rng.normal(0, 0.3, T)) + np.tile(rng.normal(0, 1, 12), 10) + x @ [0.7, -0.4] + rng.normal(0, 1, T)
compare("(1) local linear trend + seasonal 12 + 2 regressors in the state, params [1, .1, .01, .05]",
        y, x, dict(level="local linear trend", seasonal=12, mle_regression=False), [1.0, 0.1, 0.01, 0.05])

# (2) smooth trend + fixed seasonal 12 + regressors in the state: the diffuse period never ends
cases = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "uc_state_regression_cases.json")))
for key in ("49", "382"):
    c = cases[key]
    yy = np.array([np.nan if v is None else v for v in c["y"]])
    compare(f"(2.{key}) {c['spec']} k={len(c['exog'][0])} T={len(yy)} params {np.round(c['params'], 4).tolist()}",
            yy, np.array(c["exog"]), c["spec"], c["params"])

# (3) false refusal of a unit-scale regressor (max |x| = 2.8): local level + seasonal 12
rng = np.random.default_rng(69)
T = 96
x = rng.normal(size=(T, 1))
y = np.cumsum(rng.normal(0, 0.3, T)) + np.tile([1.0, 0.5, -0.5, -1.0, 0.2, 0.0, 0.3, -0.2, 0.8, -0.6, 0.1, -0.6], 8) \
    + 0.7 * x[:, 0] + rng.normal(0, 1, T)
compare(f"(3) local level + seasonal 12 + 1 N(0,1) regressor in the state (max|x| = {np.abs(x).max():.2f}), params [1, .1, .05]",
        y, x, dict(level="local level", seasonal=12, mle_regression=False), [1.0, 0.1, 0.05])
