"""Golden fixtures for tsecon-recession: static probit / logit recession models.

VALIDATION TARGET (INDEPENDENT REFERENCE, target (a) of the crate spec).
========================================================================
The STATIC probit and the STATIC logit are validated against an INDEPENDENT
reference: statsmodels' own maximum-likelihood binary-choice estimators,

    sm.Probit(y, X).fit(disp=0)
    sm.Logit(y, X).fit(disp=0)

fit here on a FIXED simulated dataset. statsmodels computes the same population
quantity (the exact-likelihood MLE, its analytic-Hessian standard errors, the
log-likelihood, McFadden's pseudo-R^2, and the fitted probability path) by a
completely separate code path from the tsecon Rust crate, so reproducing its
numbers to ~1e-6 is a genuine cross-implementation check, not a circular one.

The DYNAMIC probit (Kauppi-Saikkonen 2008) has NO statsmodels reference and is
therefore NOT in this fixture. It is validated PROPERTY-ONLY inside the crate's
Rust test suite (`tests/properties.rs`): on data simulated from a known
dynamic-probit DGP the estimator recovers rho and b within Monte-Carlo bands,
and its log-likelihood exceeds the static model's on persistent data.

MODEL (static)
==============
Binary recession indicator y_t in {0, 1} regressed on X_t (a constant plus
leading predictors, e.g. the term spread):

    index_t = X_t' beta
    P(y_t = 1 | X_t) = F(index_t)

with F = Phi (standard normal CDF) for the probit and F = Lambda (logistic CDF)
for the logit. beta is the exact-likelihood MLE maximizing

    LL(beta) = sum_t [ y_t log F(index_t) + (1 - y_t) log(1 - F(index_t)) ].

Standard errors are the square roots of the diagonal of the inverse of the
negative analytic Hessian (observed information) at the MLE — statsmodels'
default 'nonrobust' covariance. z-statistics are beta / se. McFadden's
pseudo-R^2 is 1 - LL(beta_hat) / LL_null, where LL_null is the intercept-only
log-likelihood (its MLE probability is ybar = mean(y), so
LL_null = n [ ybar log ybar + (1 - ybar) log(1 - ybar) ]).

Run with the project venv:
    .venv/bin/python fixtures/generate_tsecon-recession_fixtures.py
"""

import json
import numpy as np
import statsmodels.api as sm

OUT = "fixtures/tsecon-recession.json"


def simulate(seed: int, n: int):
    """A fixed simulated recession dataset: constant + term spread + a second
    leading predictor, with y drawn from a true probit so the classes are
    reasonably balanced and NOT perfectly separable (a real MLE exists)."""
    rng = np.random.default_rng(seed)
    # Term spread: a persistent (AR(1)) leading indicator.
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.6 * spread[t - 1] + rng.standard_normal()
    # A second, less persistent leading predictor.
    lead = 0.3 * rng.standard_normal(n) + 0.2 * spread
    # True probit DGP. Negative spread coefficient: an inverted yield curve
    # (low/negative spread) raises recession probability.
    const = np.ones(n)
    X_true = np.column_stack([const, spread, lead])
    beta_true = np.array([-0.7, -0.9, 0.5])
    idx = X_true @ beta_true
    p = _phi_cdf(idx)  # true recession probability under the probit DGP
    y = (rng.uniform(size=n) < p).astype(float)
    return y, X_true


def _phi_cdf(z):
    from scipy.stats import norm

    return norm.cdf(z)


def fit_block(y, X, kind: str):
    if kind == "probit":
        res = sm.Probit(y, X).fit(disp=0)
    elif kind == "logit":
        res = sm.Logit(y, X).fit(disp=0)
    else:
        raise ValueError(kind)
    fitted = res.predict(X)  # fitted P(y=1) path
    return {
        "params": res.params.tolist(),
        "bse": res.bse.tolist(),
        "tvalues": res.tvalues.tolist(),  # z-statistics for an MLE
        "llf": float(res.llf),
        "llnull": float(res.llnull),
        "prsquared": float(res.prsquared),  # McFadden pseudo-R^2
        "fitted": fitted.tolist(),
    }


def main():
    y, X = simulate(seed=20260717, n=240)
    n1 = int(y.sum())
    assert 0 < n1 < len(y), "degenerate: all-0 or all-1 y"

    fixture = {
        "_doc": "Independent-reference golden: statsmodels Probit/Logit MLE on a "
        "fixed simulated recession dataset. Static models are reference-matched to "
        "~1e-6; the dynamic probit is property-only (no statsmodels reference).",
        "n": len(y),
        "k": X.shape[1],
        "n_recession": n1,
        "y": y.tolist(),
        # X columns as explicit vectors (statsmodels/linearmodels exog convention):
        "const": X[:, 0].tolist(),
        "spread": X[:, 1].tolist(),
        "lead": X[:, 2].tolist(),
        "probit": fit_block(y, X, "probit"),
        "logit": fit_block(y, X, "logit"),
    }

    with open(OUT, "w") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}: n={len(y)} k={X.shape[1]} recessions={n1}")
    print("probit params:", fixture["probit"]["params"])
    print("probit llf:", fixture["probit"]["llf"], "prsq:", fixture["probit"]["prsquared"])
    print("logit  params:", fixture["logit"]["params"])
    print("logit  llf:", fixture["logit"]["llf"], "prsq:", fixture["logit"]["prsquared"])


if __name__ == "__main__":
    main()
