"""Golden fixtures for the Jentsch-Lunsford moving-block bootstrap bands
for proxy SVARs (`tsecon_var::proxy_bands`).

VALIDATION STRATEGY
===================
No external package implements Jentsch-Lunsford moving-block bands for the
proxy-SVAR estimand, so there is no third-party number to copy. This file is
therefore a **documented-formula golden**: the algorithm of
docs/roadmap/15-proxy-svar-bands.md section A is transcribed here into plain
NumPy (with statsmodels supplying the reduced-form VAR, an independent OLS
implementation), and the Rust crate must reproduce it cell for cell. This
file NEVER imports tsecon, so agreement is a genuine cross-implementation
check rather than a restatement of one code path.

The one thing a NumPy transcription cannot reproduce is the library's RNG.
So the fixture PINS THE BLOCK STARTS: a seeded integer matrix of shape
(B, N) is written into the JSON, this script consumes it, and the crate's
`proxy_svar_bands_from_starts` consumes the same matrix. The randomness is
then a shared input, not a thing to be matched, and everything downstream of
it -- centering, reconstruction, re-estimation, re-identification,
normalization, quantiles -- is compared exactly.

WHAT IS PINNED (and what it would catch)
----------------------------------------
* `point`            -- the unit-effect IRF, Psi_h @ b.
* `lower/upper`      -- Hall (basic / reverse-percentile) endpoints,
                        2*theta_hat - Q_{1-a/2} and 2*theta_hat - Q_{a/2}.
* `lower/upper_efron`-- Efron percentile endpoints, the Mertens-Ravn /
                        Gertler-Karadi convention.
* `se`               -- bootstrap SD over the surviving draws.
* `gamma_norm_draws` -- gamma*[norm_var] per draw. This is the joint-blocking
                        detector: under joint blocking these sit around the
                        sample value; under independent blocking they would
                        be centered at zero.
* `u_bar`, `m_bar`   -- the position-specific (Kuensch/BJT) centering terms.
                        Grand-mean centering is a silent no-op, so pinning
                        the position-wise terms is what makes that bug
                        visible.
* `rho_draws`        -- rho* = gamma*/gamma*[nv] per draw, the scale-free
                        diagnostic. rho*[nv] == 1 exactly in every surviving
                        draw, and theta*_0 == unit * rho*.
* `n_failed`         -- failed draws are counted, not dropped.

TWO CASES, AND WHY
------------------
`base`   -- a healthy proxy (6 leading NaNs out of 78). Every draw survives,
            so this case pins the arithmetic but says nothing about failure
            accounting: asserting 0 == 0 on five counters has no teeth.
`sparse` -- the SAME data and the SAME reduced form, with the instrument
            available on only four scattered dates. The availability pattern
            is itself resampled, so a substantial minority of draws retain
            fewer than three finite proxy entries and are counted rather than
            dropped. This is what gives the failure-accounting assertions
            something to fail on, and it pins the |O*| >= 3 classification
            boundary against an independent implementation.

WHAT THIS FILE CANNOT PIN
-------------------------
The crate splits `refit_failed` (the OLS refit or its MA recursion errored)
from `identification_failed` (the refit succeeded and `proxy_svar` rejected
the draw -- realistically a sigma_u* that is not positive definite). This
transcription has no Cholesky step, so it cannot reach the second counter and
always reports it as 0. That counter is the crate's own responsibility; the
`non_finite` counter likewise is exercised in `proxy_bands_props.rs` rather
than here.

THE ALGORITHM AS TRANSCRIBED HERE
---------------------------------
1. Fit y_t = nu + sum_i A_i y_{t-i} + u_t by OLS (statsmodels VAR, trend="c")
   on the full sample; keep Ahat, uhat (T x n), the df-adjusted Sigma_u, the
   MA matrices Psi_0..Psi_H, and the p ACTUAL presample rows.
2. Point estimate: gamma = mean over the finite-proxy overlap O of
   (m - mbar_O)(u - ubar_O); rho = gamma / gamma[nv]; b = unit * rho;
   theta_h = Psi_h @ b.
3. Position-wise centering over the T-ell+1 overlapping candidate starts:
   ubar_s = mean_i uhat[i+s], mbar_s = mean over the FINITE m[i+s].
4. Per replication r, with the pinned starts i_1..i_N: u*_t = uhat[i_j+s] -
   ubar_s and m*_t = m[i_j+s] - mbar_s, SAME i_j for both, blocks laid end to
   end and truncated to exactly T. NaN proxy entries propagate.
5. y* rebuilt recursively from Ahat with the actual presample rows and NO
   burn-in; refit by OLS; Psi*_h from the refit.
6. gamma* from the RE-ESTIMATED residuals paired with m*; rho* =
   gamma*/gamma*[nv] recomputed INSIDE the draw; b* = unit * rho*;
   theta*_h = Psi*_h @ b*.
7. Quantiles over the surviving draws; both interval types emitted.

Note step 6: b*[nv] == unit exactly in every draw, so the h=0 band for
norm_var is degenerate. The fixture records that degeneracy explicitly in
`degenerate_cell` so the crate test can assert it against a number that was
produced independently.

Also emitted: a WILD-bootstrap section proving the invalidity claim
numerically. With a common Rademacher draw on residuals and proxy, the
identifying moment sum_t m*_t u*_t' is bit-identical across draws
(`wild_moment_max_deviation` is exactly 0.0), which is why those bands are
labelled not asymptotically valid.

Run with the project venv:
    .venv/bin/python fixtures/generate_proxy_svar_bands_fixtures.py
"""

import json

import numpy as np
import statsmodels
from statsmodels.tsa.api import VAR

OUT = "fixtures/proxy_svar_bands.json"

# ---------------------------------------------------------------- the DGP
# A 3-variable proxy-SVAR with a KNOWN impact column, so the point estimate
# has something true to be near and the fixture is not just self-referential.
N = 3
LAGS = 2
HORIZON = 6
NORM_VAR = 0
UNIT = 1.0
ALPHA = 0.10
BLOCK_LENGTH = 5
N_BOOT = 150
N_OBS = 80
DGP_SEED = 20260805
STARTS_SEED = 71923

# The `sparse` case: the same data and the same reduced form, with the
# instrument available on only these four dates of the residual sample. With
# ell = 5 a block covers a kept date about a quarter of the time, so the
# number of finite proxy entries surviving into a draw is roughly
# Binomial(ceil(T/ell), 0.27) and a substantial minority of draws fall below
# the |O*| >= 3 threshold. That is the point: it makes the failure counters
# nonzero so the golden has something to be wrong about.
SPARSE_KEEP = (7, 26, 45, 64)
SPARSE_BLOCK_LENGTH = 5
SPARSE_N_BOOT = 120
SPARSE_STARTS_SEED = 5150

# True impact matrix H (u_t = H eps_t); column 0 is the target shock.
H_TRUE = np.array(
    [
        [1.00, 0.00, 0.00],
        [0.60, 0.90, 0.00],
        [-0.40, 0.30, 0.80],
    ]
)
A1_TRUE = np.array(
    [
        [0.50, 0.10, 0.00],
        [0.15, 0.45, 0.10],
        [0.00, 0.20, 0.40],
    ]
)
A2_TRUE = np.array(
    [
        [0.10, 0.00, 0.05],
        [0.00, 0.10, 0.00],
        [0.05, 0.00, 0.10],
    ]
)
PHI = 0.8  # instrument relevance: m_t = PHI * eps_{1t} + noise
PROXY_NOISE = 0.5
NAN_PREFIX = 6  # dates where the instrument is unavailable


def nan_to_null(arr):
    """A 1-D list with non-finite entries as None (JSON null), the repo
    convention: `json.dump` emits the literal `NaN`, which is not valid JSON.
    """
    return [None if not np.isfinite(x) else float(x) for x in np.asarray(arr)]


def simulate():
    """Simulate the VAR(2) and a relevant, exogenous proxy."""
    rng = np.random.default_rng(DGP_SEED)
    burn = 200
    total = N_OBS + burn
    eps = rng.standard_normal((total, N))
    u = eps @ H_TRUE.T
    y = np.zeros((total, N))
    for t in range(2, total):
        y[t] = 0.2 + A1_TRUE @ y[t - 1] + A2_TRUE @ y[t - 2] + u[t]
    y = y[burn:]
    # The proxy is relevant for eps_1 and orthogonal to the others by
    # construction; its noise makes the first stage nondegenerate.
    m_full = PHI * eps[burn:, 0] + PROXY_NOISE * rng.standard_normal(N_OBS)
    return y, m_full


# ------------------------------------------------- reduced form and moments


def ols_var(y, lags):
    """OLS VAR with a constant, returning (nu, [A_1..A_p], uhat, Sigma_u).

    Implemented directly (not via statsmodels) because the bootstrap refits
    thousands of times; `check_against_statsmodels` below verifies this
    routine reproduces statsmodels' fit on the full sample, so the fast path
    inherits the independent reference's conventions.
    """
    n_obs, n = y.shape
    t_eff = n_obs - lags
    z = np.ones((t_eff, 1 + n * lags))
    for i in range(1, lags + 1):
        z[:, 1 + (i - 1) * n : 1 + i * n] = y[lags - i : n_obs - i]
    yy = y[lags:]
    beta, *_ = np.linalg.lstsq(z, yy, rcond=None)  # (1 + n*p) x n
    resid = yy - z @ beta
    m_reg = 1 + n * lags
    sigma_u = resid.T @ resid / (t_eff - m_reg)
    nu = beta[0].copy()
    coefs = [beta[1 + (i - 1) * n : 1 + i * n].T.copy() for i in range(1, lags + 1)]
    return nu, coefs, resid, sigma_u


def ma_rep(coefs, horizon):
    """Psi_0 = I, Psi_h = sum_{i=1..min(h,p)} Psi_{h-i} A_i."""
    n = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc += psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def identify(resid, m, psi, norm_var, unit):
    """gamma -> rho -> b -> theta_h, plus the diagnostics.

    Returns None when the draw fails (fewer than 3 finite proxy entries, no
    proxy variance, or gamma[norm_var] at the floating-point floor). The
    caller COUNTS those; it never drops them silently.
    """
    o = np.isfinite(m)
    if o.sum() < 3:
        return None, "too_few_proxy_obs"
    md = m[o] - m[o].mean()
    if np.all(md == 0.0):
        return None, "zero_proxy_variance"
    ud = resid[o] - resid[o].mean(axis=0)
    gamma = (md[:, None] * ud).mean(axis=0)
    g_norm = gamma[norm_var]
    scale = np.max(np.abs(gamma))
    if not np.isfinite(g_norm) or g_norm == 0.0 or abs(g_norm) <= 1e-12 * scale:
        return None, "near_zero_gamma_norm"
    # Re-imposed INSIDE the draw: both the scale and the sign.
    rho = gamma / g_norm
    b = unit * rho
    theta = np.array([p @ b for p in psi])
    # The crate's `non_finite` guard, transcribed: an explosive Ahat* can
    # overflow the MA recursion, and the per-draw normalization can overflow
    # theta*. Such a draw is COUNTED, never admitted to the quantiles, where
    # a single inf would poison every cell of the band rather than its own.
    if not np.all(np.isfinite(theta)):
        return None, "non_finite"
    # First-stage diagnostics (HC1-robust F and the Stock-Watson reliability).
    smm = float(md @ md)
    yd = ud[:, norm_var]
    beta = float(md @ yd) / smm
    e = yd - beta * md
    no = float(o.sum())
    var_hc1 = (no / (no - 2.0)) * float((md**2) @ (e**2)) / smm**2
    f = beta * beta / var_hc1
    syy = float(yd @ yd)
    rel = float(md @ yd) ** 2 / (smm * syy) if syy > 0 else 0.0
    return {
        "theta": theta,
        "gamma_norm": float(g_norm),
        "rho": rho,
        "f": f,
        "rel": rel,
    }, None


def position_centering(uhat, m, ell):
    """ubar_s and mbar_s: means ACROSS CANDIDATE BLOCK STARTS at a fixed
    within-block position s -- not across time. Averaging across time gives
    the grand mean, which OLS with an intercept already forces to zero, so
    that version of the fix does nothing at all."""
    t = uhat.shape[0]
    n_starts = t - ell + 1
    u_bar = np.array([uhat[s : s + n_starts].mean(axis=0) for s in range(ell)])
    m_bar = np.zeros(ell)
    m_count = np.zeros(ell, dtype=int)
    for s in range(ell):
        window = m[s : s + n_starts]
        fin = np.isfinite(window)
        m_count[s] = int(fin.sum())
        m_bar[s] = float(window[fin].mean()) if fin.any() else 0.0
    return u_bar, m_bar, m_count


def block_indices(starts, ell, t):
    """Blocks laid end to end and truncated to EXACTLY t observations."""
    out = []
    for s in starts:
        if len(out) >= t:
            break
        take = min(ell, t - len(out))
        out.extend(range(s, s + take))
    return np.array(out, dtype=int)


def simulate_recursive(nu, coefs, init, ustar):
    """y*_t = nu + sum_i A_i y*_{t-i} + u*_t, initialized at the ACTUAL
    observed presample rows, with NO burn-in."""
    p = len(coefs)
    t_eff, n = ustar.shape
    y = np.zeros((p + t_eff, n))
    y[:p] = init
    for t in range(p, p + t_eff):
        acc = nu + ustar[t - p]
        for i in range(1, p + 1):
            acc = acc + coefs[i - 1] @ y[t - i]
        y[t] = acc
    return y


def run_case(y, nu, coefs, uhat, psi, m, ell, n_boot, starts_seed):
    """One band computation: the pinned starts, the draws, both intervals.

    Everything except the proxy mask, the block length and the RNG seed is
    shared with the other case, so the two differ only in how available the
    instrument is -- which is exactly the axis the failure counters live on.
    """
    t_eff = uhat.shape[0]
    point, why = identify(uhat, m, psi, NORM_VAR, UNIT)
    assert point is not None, f"full-sample identification failed: {why}"

    u_bar, m_bar, m_count = position_centering(uhat, m, ell)

    n_blocks = -(-t_eff // ell)  # ceil
    max_start = t_eff - ell
    rng = np.random.default_rng(starts_seed)
    starts = rng.integers(0, max_start + 1, size=(n_boot, n_blocks))

    init = y[:LAGS]
    thetas = []
    rhos = []
    gamma_norm_draws = []
    f_draws = []
    rel_draws = []
    failures = {
        "too_few_proxy_obs": 0,
        "zero_proxy_variance": 0,
        "near_zero_gamma_norm": 0,
        "refit_failed": 0,
        # The crate's counter for "the refit worked and proxy_svar rejected
        # the draw". Unreachable here: this transcription has no Cholesky, so
        # it cannot see a non-PD sigma_u*. Emitted as 0 so the shape matches.
        "identification_failed": 0,
        "non_finite": 0,
    }

    def record_failure(reason):
        failures[reason] += 1
        thetas.append(np.full((HORIZON + 1, N), np.nan))
        rhos.append(np.full(N, np.nan))
        gamma_norm_draws.append(np.nan)
        f_draws.append(np.nan)
        rel_draws.append(np.nan)

    for r in range(n_boot):
        idx = block_indices(starts[r], ell, t_eff)
        pos = np.arange(t_eff) % ell
        # THE SAME idx indexes the residual rows and the proxy entries.
        ustar = uhat[idx] - u_bar[pos]
        mstar = m[idx] - m_bar[pos]
        ysim = simulate_recursive(nu, coefs, init, ustar)
        try:
            _, coefs_b, resid_b, _ = ols_var(ysim, LAGS)
        except np.linalg.LinAlgError:
            record_failure("refit_failed")
            continue
        psi_b = ma_rep(coefs_b, HORIZON)
        if not all(np.all(np.isfinite(p)) for p in psi_b):
            record_failure("non_finite")
            continue
        draw, why = identify(resid_b, mstar, psi_b, NORM_VAR, UNIT)
        if draw is None:
            record_failure(why)
            continue
        thetas.append(draw["theta"])
        rhos.append(draw["rho"])
        gamma_norm_draws.append(draw["gamma_norm"])
        f_draws.append(draw["f"])
        rel_draws.append(draw["rel"])

    thetas = np.array(thetas)  # (B, H+1, N)
    ok = np.isfinite(thetas[:, 0, 0])
    n_used = int(ok.sum())
    n_failed = n_boot - n_used
    assert n_failed == sum(failures.values())
    assert n_used >= 2, "too few surviving draws to form a bootstrap distribution"
    kept = thetas[ok]

    q_lo = np.percentile(kept, 100 * ALPHA / 2, axis=0)
    q_hi = np.percentile(kept, 100 * (1 - ALPHA / 2), axis=0)
    theta_hat = point["theta"]
    hall_lo = 2 * theta_hat - q_hi
    hall_hi = 2 * theta_hat - q_lo
    se = kept.std(axis=0, ddof=1)

    # The free self-test of the per-draw normalization: b*[nv] == unit
    # exactly, so the h=0 cell of norm_var has zero width in BOTH intervals.
    degenerate_width = float(
        max(
            abs(hall_hi[0, NORM_VAR] - hall_lo[0, NORM_VAR]),
            abs(q_hi[0, NORM_VAR] - q_lo[0, NORM_VAR]),
        )
    )
    assert degenerate_width == 0.0, degenerate_width
    # rho*[nv] == 1 EXACTLY in every surviving draw, by construction.
    for r, rho in enumerate(rhos):
        if np.isfinite(rho[NORM_VAR]):
            assert rho[NORM_VAR] == 1.0, (r, rho[NORM_VAR])

    return {
        "proxy": nan_to_null(m),
        "block_length": ell,
        "n_boot": n_boot,
        "starts": starts.tolist(),
        "u_bar": u_bar.tolist(),
        "m_bar": m_bar.tolist(),
        "m_count": m_count.tolist(),
        "point": theta_hat.tolist(),
        "point_gamma_norm": point["gamma_norm"],
        "point_first_stage_f": point["f"],
        "point_reliability": point["rel"],
        "n_proxy": int(np.isfinite(m).sum()),
        "lower": hall_lo.tolist(),
        "upper": hall_hi.tolist(),
        "lower_efron": q_lo.tolist(),
        "upper_efron": q_hi.tolist(),
        "se": se.tolist(),
        "gamma_norm_draws": nan_to_null(gamma_norm_draws),
        "rho_draws": [nan_to_null(r) for r in rhos],
        "n_used": n_used,
        "n_failed": n_failed,
        "failures": failures,
        "degenerate_cell_width": degenerate_width,
    }


def main():
    y, m_full = simulate()
    n_obs = y.shape[0]
    t_eff = n_obs - LAGS

    # Align the proxy to the residual sample and mask the leading dates.
    m = m_full[LAGS:].copy()
    m[:NAN_PREFIX] = np.nan

    nu, coefs, uhat, sigma_u = ols_var(y, LAGS)
    check_against_statsmodels(y, nu, coefs, uhat, sigma_u)
    psi = ma_rep(coefs, HORIZON)

    base = run_case(y, nu, coefs, uhat, psi, m, BLOCK_LENGTH, N_BOOT, STARTS_SEED)

    # The same data and the same reduced form, with the instrument available
    # on four dates only, so the failure counters are nonzero.
    m_sparse = np.full(t_eff, np.nan)
    for k in SPARSE_KEEP:
        m_sparse[k] = m_full[LAGS + k]
    sparse = run_case(
        y,
        nu,
        coefs,
        uhat,
        psi,
        m_sparse,
        SPARSE_BLOCK_LENGTH,
        SPARSE_N_BOOT,
        SPARSE_STARTS_SEED,
    )
    assert sparse["n_failed"] > 0, "the sparse case must actually make draws fail"

    wild = wild_moment_check(uhat, m)

    payload = {
        "_meta": {
            "generator": "fixtures/generate_proxy_svar_bands_fixtures.py",
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "method": "Jentsch-Lunsford moving-block bootstrap for proxy SVARs",
            "reference": (
                "documented-formula golden: the algorithm of "
                "docs/roadmap/15-proxy-svar-bands.md section A transcribed into NumPy. "
                "No external package implements these bands, so there is no third-party "
                "number to copy; this file never imports tsecon."
            ),
        },
        "params": {
            "lags": LAGS,
            "trend": "c",
            "horizon": HORIZON,
            "norm_var": NORM_VAR,
            "unit": UNIT,
            "alpha": ALPHA,
            "block_length": BLOCK_LENGTH,
            "n_boot": N_BOOT,
            "nan_prefix": NAN_PREFIX,
        },
        "data": y.tolist(),
        # The `base` case is spread at the top level (its `proxy`, `starts`,
        # `point`, `lower`, ... keys), so the JSON shape the crate's golden
        # already reads is unchanged; `sparse` is the second case, nested.
        # Python's json.dump emits the literal `NaN` for a non-finite float,
        # which is not valid JSON and which serde_json refuses, so every
        # possibly-NaN array goes through `nan_to_null`.
        **base,
        "sparse": {
            **sparse,
            "keep": list(SPARSE_KEEP),
            "note": (
                "the same data and the same reduced form as the base case, with the "
                "instrument available on four dates only. The availability pattern is "
                "itself resampled, so a substantial minority of draws retain fewer than "
                "three finite proxy entries and are COUNTED rather than dropped. This is "
                "the case that gives the failure-accounting assertions teeth: the base "
                "case has n_failed = 0, where asserting 0 == 0 proves nothing."
            ),
        },
        "wild": wild,
        "truth": {
            "impact_column": (H_TRUE[:, 0] / H_TRUE[NORM_VAR, 0]).tolist(),
            "note": (
                "population relative impact rho = H[:, 0] / H[norm_var, 0]; the point "
                "estimate at T=78 is a noisy estimate of this, not equal to it."
            ),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {OUT}")
    print(f"  T={t_eff}")
    print(
        f"  base   ell={BLOCK_LENGTH} B={N_BOOT} n_proxy={base['n_proxy']} "
        f"n_failed={base['n_failed']} {base['failures']}"
    )
    print(
        f"  sparse ell={SPARSE_BLOCK_LENGTH} B={SPARSE_N_BOOT} "
        f"n_proxy={sparse['n_proxy']} n_failed={sparse['n_failed']} {sparse['failures']}"
    )
    print(f"  point rho  = {np.round(np.array(base['point'])[0], 4)}")
    print(f"  truth rho  = {np.round(H_TRUE[:, 0] / H_TRUE[NORM_VAR, 0], 4)}")
    print(
        f"  first-stage F = {base['point_first_stage_f']:.2f}   "
        f"gamma[nv] = {base['point_gamma_norm']:.4f}"
    )
    print(f"  wild max |moment deviation| = {wild['moment_max_deviation']:.3e}")


def wild_moment_check(uhat, m):
    """Claim 1 of the spec, measured: a common Rademacher draw applied to the
    residuals AND the proxy leaves sum_t m*_t u*_t' bit-identical."""
    rng = np.random.default_rng(4242)
    t = uhat.shape[0]
    m0 = np.nan_to_num(m, nan=0.0)
    base = m0 @ uhat
    worst = 0.0
    n_draws = 200
    for _ in range(n_draws):
        e = rng.choice([-1.0, 1.0], size=t)
        got = (e * m0) @ (e[:, None] * uhat)
        worst = max(worst, float(np.max(np.abs(got - base))))
    return {
        "n_draws": n_draws,
        "moment": base.tolist(),
        "moment_max_deviation": worst,
        "note": (
            "m*_t u*_t' = e_t^2 m_t uhat_t' = m_t uhat_t' pointwise, so the identifying "
            "moment carries no bootstrap variability under the wild bootstrap. This is "
            "why bands='wild' is labelled not asymptotically valid."
        ),
    }


def check_against_statsmodels(y, nu, coefs, uhat, sigma_u):
    """The fast OLS above must reproduce statsmodels' VAR fit, so the
    bootstrap loop inherits an independently implemented convention."""
    res = VAR(y).fit(LAGS, trend="c")
    assert np.allclose(res.intercept, nu, atol=1e-10), "intercept"
    for i, a in enumerate(coefs):
        assert np.allclose(res.coefs[i], a, atol=1e-10), f"A_{i + 1}"
    assert np.allclose(res.resid, uhat, atol=1e-10), "residuals"
    assert np.allclose(res.sigma_u, sigma_u, atol=1e-10), "sigma_u"
    psi_sm = res.ma_rep(HORIZON)
    psi_mine = ma_rep(coefs, HORIZON)
    for h in range(HORIZON + 1):
        assert np.allclose(psi_sm[h], psi_mine[h], atol=1e-10), f"Psi_{h}"


if __name__ == "__main__":
    main()
