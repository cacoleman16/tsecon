"""Golden fixtures for the proxy SVAR / external-instrument identification.

VALIDATION STRATEGY
===================
Every number this file writes is produced by an INDEPENDENT reference —
statsmodels VAR for the reduced form and its MA representation, plus plain
NumPy for the method-of-moments identification — and NEVER by the tsecon Rust
crate, so reproducing these numbers in Rust is a genuine cross-implementation
check, not a circular one. All data are DERIVED from a seeded NumPy DGP; no
datasets are redistributed.

THE ESTIMAND
------------
Single-instrument, single-target-shock proxy SVAR (Stock & Watson 2018;
Mertens & Ravn 2013; Gertler & Karadi 2015; Montiel-Olea, Stock & Watson
2021). With reduced-form innovations u_t = H eps_t (E[eps eps'] = I, so
Sigma_u = H H') and an external instrument m_t that is relevant for the
target shock (E[m eps_0] = phi != 0) and exogenous to the others
(E[m eps_j] = 0, j != 0), the residual-instrument covariance
gamma = E[m_t u_t] = phi * H[:, 0] is EXACTLY proportional to the impact
column of the target shock. Normalizing on the norm_var entry gives the
scale-free relative impact rho = gamma / gamma[norm_var]; the unit-effect
normalization sets the impact vector b = unit * rho, and the reduced-form MA
matrices propagate it into a structural IRF irf_h = Psi_h b.

REFERENCE COMPUTATION (pure NumPy / statsmodels on the simulated data)
---------------------------------------------------------------------
1. Reduced form: statsmodels VAR(y).fit(lags, trend="c") gives residuals
   U (T x n), the df-adjusted Sigma_u (U'U / (T - m), m = 1 + n*lags), and
   the MA matrices Psi_0..Psi_H via res.ma_rep(H) (Psi_0 = I).
2. Overlap O = residual rows where the aligned proxy is finite (the first
   ~200 aligned entries are NaN to exercise the availability mask).
3. gamma_j = mean over O of (m - mbar)(U_j - ubar_j); rho = gamma / gamma[nv];
   b = unit * rho; irf_h = Psi_h @ b.
4. First stage: OLS of U[:, nv] on [1, m] over O. HC1-robust
   Var(beta) = (|O|/(|O|-2)) * sum((m-mbar)^2 e^2) / Smm^2, F = beta^2/Var;
   classical F = beta^2 Smm / (SSE/(|O|-2)); reliability = corrcoef(m, U_nv)^2.
5. Shock: eps1_hat = (U @ solve(Sigma_u, b)) / (b @ solve(Sigma_u, b)), over
   the FULL residual sample (independent of the proxy mask).

The crate golden test feeds the fixture's reduced-form quantities (U,
Sigma_u, Psi) and the aligned proxy straight into tsecon_ident::proxy_svar and
must reproduce (irf, relative_impact, impact, first_stage_f, reliability,
cov_um, n_proxy, shock) to rtol=1e-9, atol=1e-11 (only faer-vs-NumPy OLS /
Cholesky rounding differs).

A secondary parameter-recovery check (the estimated rho lands within ~0.05 of
the population truth H[:, 0] / H[nv, 0] at T=2000) proves the algebra is the
CORRECT estimator, not merely that two implementations of one formula agree.

Run with the project venv:
    .venv/bin/python fixtures/generate_proxy_svar_fixtures.py
"""

import json

import numpy as np
import scipy
import statsmodels
from statsmodels.tsa.api import VAR

OUT = "fixtures/proxy_svar.json"


def nan_to_null(arr):
    """A 1-D list with non-finite entries as None (JSON null).

    Python's json.dump emits the literal `NaN` for float nan, which is not
    valid JSON; serde_json (the Rust reader) rejects it. Encoding the proxy's
    unavailability mask as null keeps the fixture strict-JSON, and the crate
    test decodes null back to f64::NAN.
    """
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr)]


def simulate(seed):
    """Stable VAR(2) DGP with a known invertible H and a target-shock proxy."""
    rng = np.random.default_rng(seed)
    n = 3
    p = 2
    n_obs = 2000
    burn = 500
    total = n_obs + burn

    # Known invertible impact matrix; target shock = column 0.
    h = np.array(
        [
            [1.0, 0.4, 0.2],
            [0.5, 1.2, 0.3],
            [0.3, 0.5, 0.9],
        ]
    )
    assert abs(np.linalg.det(h)) > 1e-6

    # Stable lag matrices (companion spectral radius < 0.9).
    a1 = np.array(
        [
            [0.50, 0.10, 0.00],
            [0.00, 0.40, 0.10],
            [0.10, 0.00, 0.30],
        ]
    )
    a2 = np.array(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.00, 0.00, 0.10],
        ]
    )
    companion = np.zeros((2 * n, 2 * n))
    companion[:n, :n] = a1
    companion[:n, n:] = a2
    companion[n:, :n] = np.eye(n)
    sr = np.max(np.abs(np.linalg.eigvals(companion)))
    assert sr < 0.9, f"spectral radius {sr} not < 0.9"

    # Structural shocks and reduced-form innovations u_t = H eps_t.
    eps = rng.standard_normal((total, n))
    u = eps @ h.T

    y = np.zeros((total, n))
    for t in range(2, total):
        y[t] = a1 @ y[t - 1] + a2 @ y[t - 2] + u[t]

    y = y[burn:]  # (n_obs, n)
    eps = eps[burn:]  # (n_obs, n), aligned to y

    # Proxy: relevant only for eps_0, plus independent measurement noise.
    # Corr(m, eps_0)^2 = phi^2 / (phi^2 + sig_nu^2) ~= 0.3.
    phi = 1.0
    sig_nu = np.sqrt(phi * phi * 0.7 / 0.3)
    nu = sig_nu * rng.standard_normal(n_obs)
    proxy_raw = phi * eps[:, 0] + nu  # aligned to y rows

    return {
        "n": n,
        "p": p,
        "n_obs": n_obs,
        "H": h,
        "y": y,
        "proxy_raw": proxy_raw,
    }


def proxy_svar_reference(y, proxy_aligned, lags, horizon, norm_var, unit, robust_f):
    """Independent NumPy/statsmodels reference for one proxy-SVAR call.

    `proxy_aligned` is length T = n_obs - lags, aligned to the residual rows
    (NaN entries are dropped from the moments and first stage).
    """
    res = VAR(y).fit(lags, trend="c")
    uu = np.asarray(res.resid)  # (T, n)
    sigma_u = np.asarray(res.sigma_u)  # (n, n), df-adjusted
    psi = np.asarray(res.ma_rep(horizon))  # (H+1, n, n), Psi_0 = I
    t, n = uu.shape
    assert proxy_aligned.shape[0] == t

    overlap = np.where(np.isfinite(proxy_aligned))[0]
    n_proxy = int(overlap.size)
    m = proxy_aligned[overlap]
    uo = uu[overlap]  # (|O|, n)
    mbar = m.mean()
    ubar = uo.mean(axis=0)
    md = m - mbar

    # Identifying moment gamma_j = mean_O (m - mbar)(u_j - ubar_j).
    gamma = (md[:, None] * (uo - ubar)).mean(axis=0)  # (n,)
    rho = gamma / gamma[norm_var]
    b = unit * rho

    # Structural IRF irf_h = Psi_h @ b.
    irf = np.array([psi[h] @ b for h in range(horizon + 1)])  # (H+1, n)

    # First stage: OLS of u_norm on [1, m] over the overlap.
    y_ns = uo[:, norm_var]
    ybar = ubar[norm_var]
    yd = y_ns - ybar
    smm = float(np.sum(md * md))
    smy = float(np.sum(md * yd))
    syy = float(np.sum(yd * yd))
    beta = smy / smm
    e = yd - beta * md
    sse = float(np.sum(e * e))
    dof = n_proxy - 2
    if robust_f:
        var_hc1 = (n_proxy / dof) * float(np.sum(md * md * e * e)) / (smm * smm)
        first_stage_f = beta * beta / var_hc1
    else:
        first_stage_f = beta * beta * smm / (sse / dof)
    reliability = float(np.corrcoef(m, y_ns)[0, 1] ** 2)

    # Structural shock over the FULL residual sample.
    w = np.linalg.solve(sigma_u, b)
    denom = float(b @ w)
    shock = (uu @ w) / denom  # (T,)

    return {
        "resid": uu,
        "sigma_u": sigma_u,
        "psi": psi,
        "irf": irf,
        "relative_impact": rho,
        "impact": b,
        "first_stage_f": float(first_stage_f),
        "reliability": reliability,
        "cov_um": gamma,
        "n_proxy": n_proxy,
        "shock": shock,
    }


def build_case(
    name, seed, lags, horizon, norm_var, unit, robust_f, nan_prefix, recovery_tol
):
    sim = simulate(seed)
    y = sim["y"]
    n_obs = sim["n_obs"]
    proxy_full = sim["proxy_raw"].copy()  # length n_obs, aligned to y rows

    # Residual sample begins at row `lags`; the aligned proxy is proxy[lags:].
    # NaN the first `nan_prefix` residual-sample entries to exercise the mask.
    proxy_full[:lags] = np.nan  # presample rows (dropped by the [lags:] slice)
    proxy_full[lags : lags + nan_prefix] = np.nan
    proxy_aligned = proxy_full[lags:]  # length T

    ref = proxy_svar_reference(
        y, proxy_aligned, lags, horizon, norm_var, unit, robust_f
    )

    # Parameter-recovery sanity: estimated rho close to the population truth.
    # The target-shock impact column is H[:, 0]; the relative impact under
    # this call's normalization is H[:, 0] / H[norm_var, 0].
    h = sim["H"]
    rho_true = h[:, 0] / h[norm_var, 0]
    rho_err = float(np.max(np.abs(ref["relative_impact"] - rho_true)))
    assert rho_err < recovery_tol, (
        f"[{name}] rho recovery error {rho_err} >= {recovery_tol}"
    )
    assert ref["first_stage_f"] > 10.0, f"[{name}] weak instrument F={ref['first_stage_f']}"

    return {
        "name": name,
        "lags": lags,
        "horizon": horizon,
        "norm_var": norm_var,
        "unit": unit,
        "robust_f": robust_f,
        # Inputs for the end-to-end binding path (fit VAR from `data`).
        "data": y.tolist(),
        "proxy": nan_to_null(proxy_full),  # length n_obs (NaN prefix as null)
        # Reduced-form quantities the crate test feeds straight to proxy_svar.
        "resid": ref["resid"].tolist(),
        "sigma_u": ref["sigma_u"].tolist(),
        "psi": ref["psi"].tolist(),
        "proxy_aligned": nan_to_null(proxy_aligned),  # length T (NaN as null)
        "rho_true": rho_true.tolist(),
        # Expected outputs.
        "expected": {
            "irf": ref["irf"].tolist(),
            "relative_impact": ref["relative_impact"].tolist(),
            "impact": ref["impact"].tolist(),
            "first_stage_f": ref["first_stage_f"],
            "reliability": ref["reliability"],
            "cov_um": ref["cov_um"].tolist(),
            "n_proxy": ref["n_proxy"],
            "shock": ref["shock"].tolist(),
        },
    }


def main():
    seed = 20260722
    cases = [
        # Baseline: robust F, unit effect on variable 0, NaN prefix of 200.
        # norm_var=0 has the largest denominator gamma[0], so the tight
        # ~0.05 (3-4 MC std) parameter-recovery bound is the relevant check.
        build_case("baseline_robust", seed, 2, 12, 0, 1.0, True, 200, 0.05),
        # Classical F, normalize a different variable, negative unit, longer
        # horizon, no NaN prefix (full overlap). Normalizing on variable 1
        # (a smaller impact entry) legitimately amplifies the ratio's
        # sampling noise (MOSW), so the recovery bound is looser here; the
        # bit-comparable golden check still pins the normalization plumbing.
        build_case("classical_altnorm", seed + 1, 2, 16, 1, -2.0, False, 0, 0.6),
    ]

    fixture = {
        "_meta": {
            "description": "Golden fixtures for the single-instrument proxy SVAR "
            "(external-instrument SVAR-IV; Stock-Watson 2018, Mertens-Ravn 2013, "
            "Gertler-Karadi 2015, Montiel-Olea-Stock-Watson 2021).",
            "references": {
                "reduced_form": 'statsmodels VAR(y).fit(lags, trend="c"): resid, '
                "df-adjusted sigma_u, ma_rep(horizon) (Psi_0 = I)",
                "identification": "numpy method of moments: gamma = mean_O (m-mbar)(u-ubar); "
                "rho = gamma/gamma[nv]; b = unit*rho; irf_h = Psi_h @ b",
                "first_stage": "OLS u_nv ~ [1, m] over O; HC1-robust F = beta^2/Var_HC1(beta), "
                "classical F = beta^2 Smm/(SSE/(|O|-2)); reliability = corrcoef(m, u_nv)^2",
                "shock": "eps1_hat = (U @ solve(Sigma_u, b)) / (b @ solve(Sigma_u, b))",
            },
            "tolerance": {"rtol": 1e-9, "atol": 1e-11},
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "seed": seed,
        },
        "cases": cases,
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"wrote {OUT}")
    for c in cases:
        exp = c["expected"]
        print(
            f"  case {c['name']}: lags={c['lags']} H={c['horizon']} nv={c['norm_var']} "
            f"unit={c['unit']} robust_f={c['robust_f']} n_proxy={exp['n_proxy']} "
            f"F={exp['first_stage_f']:.2f} reliability={exp['reliability']:.4f} "
            f"rho={np.round(exp['relative_impact'], 4).tolist()}"
        )


if __name__ == "__main__":
    main()
