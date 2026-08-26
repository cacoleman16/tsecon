"""Golden fixture for the Ornstein-Uhlenbeck spread utilities (field item 8).

Two legs, honest about their strength:

1. **Independent-package golden (statsmodels)** for the AR(1) discretization:
   the OU exact discretization is the Gaussian AR(1)
   ``X_{t+1} = c + phi X_t + eps,  eps ~ N(0, eta2)``, and statsmodels
   ``AutoReg(x, lags=1, trend='c').fit()`` computes exactly the estimator
   tsecon's ``ou_fit`` uses for that leg: OLS ``(const, ar.L1) = (c, phi)``,
   the MLE variance ``sigma2 = RSS / n`` (no dof correction), ``bse`` from
   ``sigma2 * (X'X)^{-1}``, the ``(c, phi)`` covariance, and the conditional
   log-likelihood. All are pinned per cell.

2. **Documented-formula golden (NumPy)** for the closed-form OU mapping on
   top (no package fits an OU by exact discretization with these outputs; the
   formulas below are the published inverse of the exact discretization and
   the delta method, transcribed directly):

       kappa  = -ln(phi) / dt
       mu     = c / (1 - phi)
       sigma2 = eta2 * (-2 ln phi) / (dt (1 - phi^2))
       SE(kappa)  = SE(phi) / (phi dt)
       Var(mu)    = g' V g,  g = [1/(1-phi), c/(1-phi)^2],  V = Cov[(c, phi)]
       a(phi)     = -2 ln(phi) / (dt (1 - phi^2))
       a'(phi)    = (-2/dt) [ (1-phi^2)/phi + 2 phi ln phi ] / (1-phi^2)^2
       Var(sigma2)= (eta2 a')^2 Var(phi) + a^2 (2 eta2^2 / n)
       SE(sigma)  = SE(sigma2) / (2 sigma)
       half_life  = ln 2 / kappa
       CI (LEVEL scale, level L — the construction the shipped Monte Carlo
       docs/examples/coverage/experiments/ou_kappa_bias_coverage.py measured
       as covering closer to nominal than the log-scale alternative in every
       cell): z = Phi^{-1}((1+L)/2), kappa_lo/hi = kappa -+ z SE(kappa),
              half_life_ci = (ln2/kappa_hi, ln2/kappa_lo)   if kappa_lo > 0
                           = (ln2/kappa_hi, +inf)           if kappa_lo <= 0
              (+inf is stored as JSON null in `half_life_ci[1]`)
       stationary_sd = sigma / sqrt(2 kappa)

Cells: a fast-reverting daily spread, a slow-reverting daily spread (the
half-life-CI stress case), a monthly spread, a weakly-identified short daily
spread whose fitted kappa interval crosses zero (asserted, pinning the
half-life CI's +inf upper branch), and a mildly explosive series whose
fitted AR(1) root lands at/over unity — pinning the honest
``mean_reverting = false`` branch (the generator asserts ``phi_hat >= 1`` so
the fixture cannot silently stop covering that branch).

This generator deliberately does NOT import tsecon.

Run with the worktree venv:
    .venv-wt/bin/python fixtures/generate_ou_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


def simulate_ou(rng, kappa, mu, sigma, dt, n, x0):
    """Exact-discretization OU path (the same AR(1) the estimator inverts)."""
    phi = np.exp(-kappa * dt)
    c = mu * (1.0 - phi)
    eta = np.sqrt(sigma**2 * (1.0 - phi**2) / (2.0 * kappa))
    x = np.empty(n)
    x[0] = x0
    shocks = rng.standard_normal(n - 1)
    for t in range(1, n):
        x[t] = c + phi * x[t - 1] + eta * shocks[t - 1]
    return x


def simulate_ar1(rng, c, phi, eta, n, x0):
    x = np.empty(n)
    x[0] = x0
    shocks = rng.standard_normal(n - 1)
    for t in range(1, n):
        x[t] = c + phi * x[t - 1] + eta * shocks[t - 1]
    return x


def ar1_leg(x):
    """The closed-form AR(1) OLS/MLE (identical to AutoReg, verified below)."""
    lag, lead = x[:-1], x[1:]
    n = len(lag)
    ml, md = lag.mean(), lead.mean()
    sxx = float(((lag - ml) ** 2).sum())
    sxy = float(((lag - ml) * (lead - md)).sum())
    phi = sxy / sxx
    c = md - phi * ml
    resid = lead - c - phi * lag
    rss = float((resid**2).sum())
    eta2 = rss / n
    llf = -0.5 * n * (np.log(2 * np.pi * eta2) + 1.0)
    var_phi = eta2 / sxx
    var_c = eta2 * (1.0 / n + ml**2 / sxx)
    cov_c_phi = -eta2 * ml / sxx
    return dict(
        n=n, phi=phi, c=c, eta2=eta2, llf=float(llf),
        phi_se=float(np.sqrt(var_phi)), c_se=float(np.sqrt(var_c)),
        cov_c_phi=float(cov_c_phi),
    )


def ou_map(leg, dt, level):
    """The documented closed-form OU mapping + delta method (docstring above)."""
    from scipy.stats import norm

    phi, c, eta2, n = leg["phi"], leg["c"], leg["eta2"], leg["n"]
    var_phi, var_c, cov_c_phi = leg["phi_se"] ** 2, leg["c_se"] ** 2, leg["cov_c_phi"]
    kappa = -np.log(phi) / dt
    mu = c / (1.0 - phi)
    om = 1.0 - phi**2
    a = -2.0 * np.log(phi) / (dt * om)
    sigma2 = eta2 * a
    sigma = np.sqrt(sigma2)
    kappa_se = np.sqrt(var_phi) / (phi * dt)
    g1 = 1.0 / (1.0 - phi)
    g2 = c / (1.0 - phi) ** 2
    mu_se = np.sqrt(g1**2 * var_c + g2**2 * var_phi + 2.0 * g1 * g2 * cov_c_phi)
    a_prime = (-2.0 / dt) * ((om / phi + 2.0 * phi * np.log(phi)) / om**2)
    var_eta2 = 2.0 * eta2**2 / n
    var_sigma2 = (eta2 * a_prime) ** 2 * var_phi + a**2 * var_eta2
    sigma_se = np.sqrt(var_sigma2) / (2.0 * sigma)
    out = dict(
        kappa=float(kappa), mu=float(mu), sigma=float(sigma),
        kappa_se=float(kappa_se), mu_se=float(mu_se), sigma_se=float(sigma_se),
    )
    if kappa > 0:
        z = norm.ppf(0.5 + level / 2.0)
        k_lo, k_hi = kappa - z * kappa_se, kappa + z * kappa_se
        hi = float(np.log(2.0) / k_lo) if k_lo > 0 else None  # None = +inf
        out["half_life"] = float(np.log(2.0) / kappa)
        out["half_life_ci"] = [float(np.log(2.0) / k_hi), hi]
        out["stationary_sd"] = float(np.sqrt(sigma2 / (2.0 * kappa)))
        out["mean_reverting"] = True
    else:
        out["half_life"] = None  # +inf in the crate; JSON has no inf
        out["half_life_ci"] = None
        out["stationary_sd"] = None
        out["mean_reverting"] = False
    return out


def gen():
    import statsmodels
    from statsmodels.tsa.ar_model import AutoReg

    cells = []

    specs = [
        # name,            kappa, mu,    sigma, dt,       T,    x0,  seed
        ("daily_fast",     5.0,   0.10,  0.40,  1 / 252,  1260, 0.6, 20260826),
        ("daily_slow",     0.5,  -0.25,  0.20,  1 / 252,  2520, 0.0, 8),
        ("monthly",        2.0,   1.50,  0.60,  1 / 12,   240,  1.0, 41),
    ]
    for name, kappa, mu, sigma, dt, n, x0, seed in specs:
        rng = np.random.default_rng(seed)
        x = simulate_ou(rng, kappa, mu, sigma, dt, n, x0)
        leg = ar1_leg(x)
        sm = AutoReg(x, lags=1, trend="c").fit()
        c_sm, phi_sm = (float(v) for v in sm.params)
        # The closed form and AutoReg are the same estimator; assert it here
        # so the fixture never ships with the two legs silently disagreeing.
        assert abs(leg["phi"] - phi_sm) <= 1e-12 * abs(phi_sm)
        assert abs(leg["c"] - c_sm) <= 1e-10 * max(1.0, abs(c_sm))
        assert abs(leg["eta2"] - float(sm.sigma2)) <= 1e-12 * float(sm.sigma2)
        assert abs(leg["llf"] - float(sm.llf)) <= 1e-10 * abs(float(sm.llf))
        bse = [float(v) for v in sm.bse]
        assert abs(leg["c_se"] - bse[0]) <= 1e-10 * bse[0]
        assert abs(leg["phi_se"] - bse[1]) <= 1e-10 * bse[1]
        cells.append(
            dict(
                name=name, dt=dt, level=0.95, x=full(x),
                true=dict(kappa=kappa, mu=mu, sigma=sigma),
                statsmodels=dict(c=c_sm, phi=phi_sm, sigma2=float(sm.sigma2),
                                 llf=float(sm.llf), c_se=bse[0], phi_se=bse[1],
                                 cov_c_phi=float(sm.cov_params()[0, 1])),
                ar1=leg, ou=ou_map(leg, dt, 0.95),
            )
        )

    # The weakly-identified branch: a short slow-reverting sample whose fitted
    # kappa is positive but whose level-scale kappa interval crosses zero, so
    # the half-life CI's upper endpoint is +inf (JSON null). Seed-searched so
    # the fixture is guaranteed to pin that branch.
    from scipy.stats import norm

    dt = 1 / 252
    z975 = norm.ppf(0.975)
    for seed in range(1000):
        rng = np.random.default_rng(1000 + seed)
        x = simulate_ou(rng, kappa=0.3, mu=0.0, sigma=0.2, dt=dt, n=252, x0=0.0)
        leg = ar1_leg(x)
        if not (0 < leg["phi"] < 1):
            continue
        kappa_hat = -np.log(leg["phi"]) / dt
        kappa_se = leg["phi_se"] / (leg["phi"] * dt)
        if kappa_hat - z975 * kappa_se <= 0:
            break
    else:  # pragma: no cover
        raise RuntimeError("no seed produced a zero-crossing kappa interval")
    sm = AutoReg(x, lags=1, trend="c").fit()
    assert abs(leg["phi"] - float(sm.params[1])) <= 1e-12 * abs(leg["phi"])
    ou = ou_map(leg, dt, 0.95)
    assert ou["mean_reverting"] and ou["half_life_ci"][1] is None
    cells.append(
        dict(
            name="daily_weak", dt=dt, level=0.95, x=full(x), seed_used=1000 + seed,
            true=dict(kappa=0.3, mu=0.0, sigma=0.2),
            statsmodels=dict(c=float(sm.params[0]), phi=float(sm.params[1]),
                             sigma2=float(sm.sigma2), llf=float(sm.llf),
                             c_se=float(sm.bse[0]), phi_se=float(sm.bse[1]),
                             cov_c_phi=float(sm.cov_params()[0, 1])),
            ar1=leg, ou=ou,
        )
    )

    # The non-mean-reverting branch: a mildly explosive AR(1) whose FITTED
    # root is >= 1 (searched over seeds so the property is guaranteed).
    dt = 1 / 252
    for seed in range(1000):
        rng = np.random.default_rng(seed)
        x = simulate_ar1(rng, c=0.0, phi=1.0015, eta=0.02, n=800, x0=0.5)
        leg = ar1_leg(x)
        if leg["phi"] > 1.0:  # strictly over unity: the kappa < 0 case
            break
    else:  # pragma: no cover - the search space makes this unreachable
        raise RuntimeError("no seed produced phi_hat > 1")
    sm = AutoReg(x, lags=1, trend="c").fit()
    assert abs(leg["phi"] - float(sm.params[1])) <= 1e-12 * abs(leg["phi"])
    cells.append(
        dict(
            name="explosive", dt=dt, level=0.95, x=full(x), seed_used=seed,
            true=dict(phi=1.0015, c=0.0, eta=0.02),
            statsmodels=dict(c=float(sm.params[0]), phi=float(sm.params[1]),
                             sigma2=float(sm.sigma2), llf=float(sm.llf),
                             c_se=float(sm.bse[0]), phi_se=float(sm.bse[1]),
                             cov_c_phi=float(sm.cov_params()[0, 1])),
            ar1=leg, ou=ou_map(leg, dt, 0.95),
        )
    )

    # spread_zscore reference: the documented formula on the daily_fast fit.
    zcell = cells[0]
    kappa, mu, sigma = (zcell["ou"][k] for k in ("kappa", "mu", "sigma"))
    xz = np.asarray(zcell["x"][:16])
    zs = (xz - mu) / (sigma / np.sqrt(2.0 * kappa))
    zscore = dict(cell="daily_fast", n_head=16, zscore_head=full(zs))

    return dict(
        meta=dict(
            generator="fixtures/generate_ou_fixtures.py",
            numpy=np.__version__,
            statsmodels=statsmodels.__version__,
            python=platform.python_version(),
        ),
        cells=cells,
        zscore=zscore,
    )


if __name__ == "__main__":
    fx = gen()
    path = OUT / "ou.json"
    path.write_text(json.dumps(fx) + "\n")
    for c in fx["cells"]:
        print(f"{c['name']:>10}: phi_hat={c['ar1']['phi']:.6f} "
              f"kappa_hat={c['ou']['kappa']:.4f} mr={c['ou']['mean_reverting']}")
    print(f"wrote {path} ({path.stat().st_size} bytes)")
