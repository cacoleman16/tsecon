"""ARIMA parameter standard errors from statsmodels SARIMAX.

An INDEPENDENT reference: nothing here imports tsecon. Every number is
produced by statsmodels' own state-space code.

    .venv/bin/python fixtures/generate_arima_bse_fixtures.py

Run it with the repo's own ``.venv`` so that the ``_meta`` block records
an environment someone here can reproduce. (It was once generated
elsewhere and recorded numpy 1.26.4 / scipy 1.17.1 against a repo venv on
numpy 2.5.1 / scipy 1.18.0. Regenerating left every simulated series
bit-identical -- the seeds below fix them -- and moved params and standard
errors by at most 3.3e-10 and 4.8e-8, purely from optimizer paths.)

What is pinned and why
----------------------
`res.bse` under ``cov_type='approx'`` is the square root of the diagonal
of the inverse *negative Hessian* of the log-likelihood at the reported
parameters -- statsmodels' ``cov_params_approx``, computed as
``pinv_extended(-H_total)``. That is the estimator tsecon-arima computes,
so the two are comparable at the same parameter point.

Two qualifications on "the same estimator", both of which matter only
away from the well-behaved cases pinned here:

* ``pinv_extended`` is a *pseudo*-inverse: it truncates small singular
  values, so it returns a number even when the information matrix does
  not identify the parameters. tsecon-arima inverts for real and refuses
  below an equilibrated reciprocal condition number of 1e-6. On these six
  cases the two agree (the tightest, ``nile_arma11c``, has rcond 5.1e-4);
  on a rank-deficient one they deliberately do not.
* ``cov_type='approx'`` defaults to ``approx_complex_step=True``, i.e.
  complex-step differentiation -- *not* ``numdiff.approx_hess3``. So the
  agreement below is not step-rule parity; both sides are estimating the
  same true Hessian, and that is the only invariant. tsecon-arima
  differentiates ``sigma2`` on a log scale for exactly that reason: it is
  a departure from the statsmodels step rule that buys accuracy the step
  rule was losing at small ``sigma2``.

Two properties of this fixture matter for the comparison to be clean:

1. Every model is built with ``simple_differencing=True``, matching the
   crate's differencing convention (the ARMA is fit to the d-th
   differences and d observations are lost).
2. Each case records the FITTED PARAMETERS as well as the standard
   errors, so the crate evaluates its Hessian at the same point rather
   than at its own optimizer's stopping point. Standard errors are a
   local curvature quantity: comparing them across two slightly
   different parameter vectors would confound the Hessian method with
   the optimizers' disagreement, and the Nile ARMA(1,1) case (where
   statsmodels' default fit stalls short of the maximizer -- see
   crates/tsecon-arima/tests/golden.rs) is a live example of that
   disagreement in this very repo.

`bse_opg` and `bse_oim` are recorded for context only. They are
different estimators (outer product of gradients; observed information
from the analytic score) and are asymptotically -- not numerically --
equal to `bse_approx`. The crate gates on `bse_approx`.

Statsmodels differentiates by complex step, which is exact to machine
precision. A real finite-difference Hessian cannot match that to 1e-10,
so the crate's tolerances are set to what is actually achieved; see the
`agreement_note` field.
"""

import json
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.simplefilter("ignore")

OUT = Path(__file__).parent
META = {
    "statsmodels": statsmodels.__version__,
    "scipy": scipy.__version__,
    "numpy": np.__version__,
    "python": platform.python_version(),
}

# One generator seed for every simulated series, so the fixture is
# reproducible from this file alone.
rng = np.random.default_rng(20260805)


def case(y, order, trend, name, note, store_series=True):
    """Fit with statsmodels and record params + the three bse flavours."""
    mod = SARIMAX(y, order=order, trend=trend, simple_differencing=True)
    res = mod.fit(disp=0, maxiter=500, cov_type="approx")
    out = {
        "name": name,
        "note": note,
        "order": list(order),
        "trend": trend,
        "nobs": int(res.nobs),
        "param_names": list(map(str, res.param_names)),
        "params": [float(v) for v in res.params],
        "loglike": float(res.llf),
        "bse_approx": [float(v) for v in res.bse],
        "bse_oim": [float(v) for v in mod.fit(disp=0, maxiter=500, cov_type="oim").bse],
        "bse_opg": [float(v) for v in mod.fit(disp=0, maxiter=500, cov_type="opg").bse],
    }
    if store_series:
        out["y"] = [float(v) for v in y]
    return out


cases = []

# --- The audit case: random walk with drift, T = 60. -------------------
# ARIMA(0,1,0) with a constant is the model whose forecast bands the
# interval-coverage audit measured at 90.2% against a 95% nominal level.
# Here the closed form is known exactly -- c_hat is the mean of the
# differences and Var(c_hat) = sigma2 / n -- so this case doubles as a
# check that statsmodels' own numbers agree with the analytic answer
# (asserted below), which is what makes it a trustworthy anchor.
T = 60
y_rw = np.cumsum(np.concatenate([[0.0], 0.4 + rng.standard_normal(T - 1)]))
cases.append(
    case(
        y_rw,
        (0, 1, 0),
        "c",
        "rw_drift_010c_T60",
        "random walk with drift, T=60: the interval-coverage audit case; "
        "closed form se(c) = sqrt(sigma2 / n), se(sigma2) = sqrt(2 sigma2^2 / n)",
    )
)

# --- Stationary ARMA and AR with a constant, T = 300. ------------------
n, burn = 300, 200
e = rng.standard_normal(n + burn)
x = np.zeros(n + burn)
for t in range(1, n + burn):
    x[t] = 0.5 + 0.6 * x[t - 1] + e[t] + 0.4 * e[t - 1]
y_arma = x[burn:]
cases.append(
    case(y_arma, (1, 0, 1), "c", "arma11c_T300", "ARMA(1,1) + constant, c=0.5 phi=0.6 theta=0.4")
)
cases.append(
    case(
        y_arma - y_arma.mean(),
        (1, 0, 1),
        "n",
        "arma11_noconst_T300",
        "same series demeaned, fit without a constant: k = 3, no leading const slot",
    )
)

x2 = np.zeros(n + burn)
for t in range(2, n + burn):
    x2[t] = 0.2 + 0.5 * x2[t - 1] - 0.3 * x2[t - 2] + e[t]
cases.append(
    case(x2[burn:], (2, 0, 0), "c", "ar2c_T300", "AR(2) + constant, phi=(0.5, -0.3)")
)

# --- Differenced case with a full ARMA block. --------------------------
cases.append(
    case(
        np.cumsum(y_arma),
        (1, 1, 1),
        "c",
        "arima111c_T300",
        "ARIMA(1,1,1) + constant: exercises the d > 0 path (one observation lost)",
    )
)

# --- Real data with wildly different parameter scales. -----------------
# The Nile discharge series already lives in diagnostics.json; it is read
# rather than duplicated. sigma2 is O(2e4) while phi is O(1), so this is
# the case that stresses the relative finite-difference step rule.
# statsmodels' default fit stalls short of the maximizer here (documented
# in crates/tsecon-arima/tests/golden.rs), which is precisely why the
# crate must evaluate its Hessian at the recorded `params`.
nile = np.asarray(json.loads((OUT / "diagnostics.json").read_text())["nile"], float)
cases.append(
    case(
        nile,
        (1, 0, 1),
        "c",
        "nile_arma11c",
        "Nile discharge, ARMA(1,1) + constant: sigma2 ~ 2e4 against phi ~ 1, and a "
        "statsmodels stopping point that is not a stationary point of the likelihood; "
        "the loosest agreement of the set",
        store_series=False,
    )
)

# --- Sanity: the T=60 case must match its closed form. -----------------
# If this ever fails the fixture is wrong, not the crate.
rw = cases[0]
dx = np.diff(y_rw)
nd = dx.size
closed_form = [float(np.sqrt(dx.var(ddof=0) / nd)), float(np.sqrt(2 * dx.var(ddof=0) ** 2 / nd))]
assert np.allclose(rw["params"], [dx.mean(), dx.var(ddof=0)], rtol=1e-6), rw["params"]
assert np.allclose(rw["bse_approx"], closed_form, rtol=1e-6), rw["bse_approx"]
rw["closed_form_bse"] = closed_form

blob = {
    "_meta": META,
    "_about": (
        "statsmodels SARIMAX parameter standard errors under cov_type='approx' "
        "(sqrt of the diagonal of the inverse negative Hessian of the log-likelihood), "
        "with simple_differencing=True. Compare at the recorded `params`, not at your "
        "own optimizer's stopping point."
    ),
    "agreement_note": (
        "statsmodels differentiates by complex step (machine precision); a real "
        "four-point finite-difference Hessian agrees to roughly 1e-7 relative on the "
        "well-conditioned cases and roughly 5e-6 on nile_arma11c, whose parameter "
        "scales span four orders of magnitude and whose evaluation point is not a "
        "stationary point. Do not gate tighter than that."
    ),
    "cases": {c["name"]: c for c in cases},
}

(OUT / "arima_bse.json").write_text(json.dumps(blob, indent=1) + "\n")
print(f"wrote arima_bse.json with {len(cases)} cases")
for c in cases:
    print(f"  {c['name']:22s} k={len(c['params'])} nobs={c['nobs']}")
