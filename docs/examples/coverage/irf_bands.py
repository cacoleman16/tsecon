"""Do `tsecon`'s impulse-response bands cover at their nominal rate?

    .venv/bin/python docs/examples/coverage/irf_bands.py            # full run, ~1-2 min
    .venv/bin/python docs/examples/coverage/irf_bands.py --quick    # smoke run, ~10 s

Two functions are measured here: `var_irf_bands` (experiments 1-5, reduced-form
VAR bands under a Cholesky ordering) and `proxy_svar_bands` (experiment 6,
external-instrument SVAR bands).

A confidence band is a promise about repeated samples: a 90% band should
contain the *population* impulse response in 90% of samples. This module
keeps that promise honest by simulating from a VAR whose population IRF is
known in closed form, re-estimating the band on every draw, and counting.

Every number printed below is a measurement, not a target. Where coverage
falls short of nominal it is reported as-is, together with the diagnostics
needed to tell *why*: the median bias of the point estimate, the mean
reported standard error, and the actual Monte Carlo standard deviation of
the point estimate across replications. Those three columns separate the
three failure modes:

  * mean_se << mc_sd  -> the standard error itself is too small.
  * |bias| >~ mc_sd   -> the band is centred in the wrong place. Under
                         correct specification this is finite-sample bias in
                         the VAR slopes (it compounds with the horizon,
                         because the IRF is a polynomial in those slopes);
                         under lag misspecification it is inconsistency, and
                         no amount of data fixes it.
  * neither           -> the band is honest and the shortfall is the normal
                         approximation to a skewed sampling distribution.

Known-truth DGPs
----------------
`BASE`    stationary VAR(1), largest root 0.758. Population orthogonalised
          IRF is exactly `A^h P` with `P = chol(Sigma)` lower-triangular;
          non-orthogonalised is exactly `A^h`; cumulative is the running sum.
`PERSIST` stationary VAR(1), largest root 0.95. Same closed form. This is
          where least-squares shrinkage toward stationarity bites and where
          Kilian's (1998) bootstrap bias correction is supposed to help.
`LAG4`    stationary VAR(4) with a lag-4 hump, largest root 0.900. Truth is
          taken from the companion form. Fitting this as a VAR(1) is the
          misspecified case: the target stays the *true* path, so the band is
          being asked to cover something the estimator is not consistent for.
`PROXY`   stationary VAR(1), largest root 0.758, with a *named* structural
          impact matrix `B` (so `Sigma = B B'`) and an external instrument
          `m_t = rho * eps_{1t} + sqrt(1 - rho^2) * v_t`. The proxy is relevant
          for structural shock 0 with strength `rho` and exogenous with respect
          to every other shock by construction. A proxy SVAR identifies column
          0 of `B` up to scale, and the unit-effect normalisation fixes the
          scale, so the population target is exactly
          `Psi_h b_1 * unit / b_1[norm_var]` -- closed form, same companion
          algebra as above. Verified against an 80,000-observation fit.

Structural zeros -- read this before interpreting any table
-----------------------------------------------------------
Under a Cholesky ordering the impact response of variable 0 to a shock in
variable 1 is *identically* zero, and `var_irf_bands` correctly reports
`se = 0` and `lower = upper = 0` there. The truth is also zero, so that cell
"covers" in 100% of samples by construction and measures nothing. The same
holds for the whole `orth=False` impact matrix, which is the identity with
zero width. Those cells are verified as exact structural facts in
`assertions()` and excluded from every coverage claim. All coverage numbers
below track cells whose population impact response is nonzero.

The proxy-SVAR arm has one more such cell. The unit-effect normalisation pins
the `h = 0` response of `norm_var` at `unit` inside *every* bootstrap draw, so
that cell has `se = 0`, `lower = upper = point = unit`, and covers in 100% of
samples by construction. It is verified as an exact fact and then EXCLUDED from
every average reported for experiment 6. It is worth being blunt about why: at
`n = 240` the wild arm's real `h = 0` coverage is about 15%, and averaging it
with the degenerate 100% cell would print "57%" -- a number that describes
nothing. Where an average includes the degenerate cell it is labelled as such
and printed next to the excluded version.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from scipy.stats import norm

import tsecon

# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------
SEED = 20260729  # every table in this file is a deterministic function of this
NOMINAL = 0.90  # alpha = 0.10, which is `var_irf_bands`'s own default
ALPHA = 1.0 - NOMINAL
N_BOOT = 399
REPS_FULL = 2000
REPS_QUICK = 250

METHODS = ("asymptotic", "bootstrap")

# --- experiment 6 (proxy SVAR) ---------------------------------------------
PROXY_N_BOOT = 399  # matches N_BOOT so the two families cost the same per draw
PROXY_NORM_VAR = 0  # unit-effect normalisation is imposed on variable 0 ...
PROXY_UNIT = 1.0  # ... and pins its h=0 response at exactly this value
PROXY_RESP = 1  # the tracked cell: the OTHER variable, informative at h=0
# corr(m_t, eps_{1t}). Everything else in the proxy is independent noise, so
# this single number IS the instrument's strength.
PROXY_STRENGTHS = (("strong", 0.50), ("moderate", 0.25), ("weak", 0.12))
PROXY_BANDS = ("moving_block", "wild")
PROXY_INTERVALS = (("hall", "lower", "upper"), ("efron", "lower_efron", "upper_efron"))


# --------------------------------------------------------------------------
# DGP plumbing: exact stationary draws and closed-form population IRFs
# --------------------------------------------------------------------------
def make_dgp(coefs, sigma, name):
    """Bundle a VAR(p) DGP with everything needed to simulate and to know truth.

    `coefs` is a list `[A1, ..., Ap]`. The returned dict caches the companion
    matrix, the lower Cholesky factor of `sigma`, the Cholesky factor of the
    stationary companion covariance (so draws need no burn-in at all), and the
    list of nonzero lag blocks (so simulating a sparse VAR(4) is not 4x slower
    than it needs to be).
    """
    coefs = [np.asarray(a, dtype=float) for a in coefs]
    k = coefs[0].shape[0]
    p = len(coefs)
    m = k * p

    companion = np.zeros((m, m))
    companion[:k] = np.hstack(coefs)
    if p > 1:
        companion[k:, : k * (p - 1)] = np.eye(k * (p - 1))

    max_root = float(np.abs(np.linalg.eigvals(companion)).max())
    if max_root >= 1.0:
        raise ValueError(f"{name}: nonstationary, largest root {max_root:.4f}")

    chol_sigma = np.linalg.cholesky(np.asarray(sigma, dtype=float))

    # Stationary companion covariance G solves G = C G C' + Su, i.e.
    # (I - C (x) C) vec(G) = vec(Su). 64x64 for k=2, p=4 -- solved once.
    su = np.zeros((m, m))
    su[:k, :k] = np.asarray(sigma, dtype=float)
    gamma = np.linalg.solve(
        np.eye(m * m) - np.kron(companion, companion), su.reshape(-1, order="F")
    ).reshape(m, m, order="F")
    gamma = 0.5 * (gamma + gamma.T)
    chol_state = np.linalg.cholesky(gamma + 1e-12 * np.eye(m))

    return {
        "name": name,
        "coefs": coefs,
        "nz": [(lag, a) for lag, a in enumerate(coefs) if np.any(a)],
        "k": k,
        "p": p,
        "sigma": np.asarray(sigma, dtype=float),
        "P": chol_sigma,
        "companion": companion,
        "chol_state": chol_state,
        "max_root": max_root,
    }


def simulate(dgp, n, rng):
    """One exactly-stationary draw of length `n` from `dgp`. No burn-in needed."""
    k, p = dgp["k"], dgp["p"]
    buf = np.zeros((p + n, k))
    state = dgp["chol_state"] @ rng.standard_normal(k * p)
    for i in range(p):  # buf[p-1] = y_0, buf[p-2] = y_{-1}, ...
        buf[p - 1 - i] = state[i * k : (i + 1) * k]
    shocks = rng.standard_normal((n, k)) @ dgp["P"].T
    nz = dgp["nz"]
    for t in range(n):
        row = shocks[t].copy()
        for lag, a in nz:
            row += a @ buf[p + t - 1 - lag]
        buf[p + t] = row
    return buf[p:]


def true_irf(dgp, horizon, orth=True, cumulative=False):
    """Population IRF `[h][response][shock]`, exact, from the companion form.

    `orth=True` gives `Psi_h P` with `P = chol(Sigma)`; `orth=False` gives
    `Psi_h` itself (the identity at h=0). `cumulative` is the running sum over
    h. These match `var_irf`'s conventions, which were verified against a
    400k-observation fit while writing this file.
    """
    k, m = dgp["k"], dgp["companion"].shape[0]
    sel = np.zeros((k, m))
    sel[:, :k] = np.eye(k)
    power = np.eye(m)
    out = []
    for _ in range(horizon + 1):
        psi = sel @ power @ sel.T
        out.append(psi @ dgp["P"] if orth else psi)
        power = power @ dgp["companion"]
    out = np.asarray(out)
    return np.cumsum(out, axis=0) if cumulative else out


BASE = make_dgp(
    [[[0.70, 0.10], [0.15, 0.50]]],
    [[1.0, 0.4], [0.4, 2.0]],
    "BASE VAR(1), root 0.758",
)
PERSIST = make_dgp(
    [[[0.95, 0.00], [0.30, 0.55]]],
    [[1.0, 0.4], [0.4, 2.0]],
    "PERSIST VAR(1), root 0.950",
)
_Z = [[0.0, 0.0], [0.0, 0.0]]
LAG4 = make_dgp(
    [[[0.40, 0.08], [0.10, 0.30]], _Z, _Z, [[0.32, 0.00], [0.06, 0.26]]],
    [[1.0, 0.4], [0.4, 2.0]],
    "LAG4 VAR(4), root 0.900",
)


# --------------------------------------------------------------------------
# proxy-SVAR plumbing: a DGP whose STRUCTURAL IRF is known in closed form
# --------------------------------------------------------------------------
def make_proxy_dgp(coefs, impact, name):
    """A VAR(p) with a named structural impact matrix `B`, so `Sigma = B B'`.

    Column 0 of `B` is the shock the instrument is relevant for. No other
    column of `B` is identified by a proxy SVAR and none is used as truth
    here -- which is the point: the reduced form is the same object whatever
    `B` is, and the proxy is the only thing that picks out `b_1`. `B` is
    deliberately NOT triangular, so `b_1` is not a Cholesky column and a
    Cholesky-shaped bug could not pass.
    """
    impact = np.asarray(impact, dtype=float)
    dgp = make_dgp(coefs, impact @ impact.T, name)
    dgp["B"] = impact
    return dgp


def simulate_proxy(dgp, n, rng, rho):
    """One exactly-stationary draw `(y, m)` of length `n`. No burn-in needed.

    `m_t = rho * eps_{1t} + sqrt(1 - rho^2) * v_t` with `v_t` independent
    standard normal, so `E[m_t eps_t'] = (rho, 0, ..., 0)`: RELEVANT for
    structural shock 0 with strength `rho`, EXOGENOUS with respect to every
    other shock. Both proxy-SVAR conditions hold exactly, by construction, at
    every `rho` -- so anything experiment 6 measures is a property of the
    band, never of a violated assumption.
    """
    k, p = dgp["k"], dgp["p"]
    eps = rng.standard_normal((n, k))
    m = rho * eps[:, 0] + math.sqrt(1.0 - rho * rho) * rng.standard_normal(n)
    shocks = eps @ dgp["B"].T
    buf = np.zeros((p + n, k))
    state = dgp["chol_state"] @ rng.standard_normal(k * p)
    for i in range(p):
        buf[p - 1 - i] = state[i * k : (i + 1) * k]
    nz = dgp["nz"]
    for t in range(n):
        row = shocks[t].copy()
        for lag, a in nz:
            row += a @ buf[p + t - 1 - lag]
        buf[p + t] = row
    return buf[p:], m


def true_proxy_irf(dgp, horizon, norm_var=PROXY_NORM_VAR, unit=PROXY_UNIT):
    """Population proxy-SVAR IRF `[h][response]`, exact, from the companion form.

    The proxy identifies `b_1` only up to scale; the unit-effect normalisation
    fixes the scale so the impact response of `norm_var` is exactly `unit`.
    The target is therefore `Psi_h b_1 * unit / b_1[norm_var]`, with `Psi_h`
    read off the companion form exactly as in `true_irf`.
    """
    k, m = dgp["k"], dgp["companion"].shape[0]
    sel = np.zeros((k, m))
    sel[:, :k] = np.eye(k)
    b1 = dgp["B"][:, 0]
    impulse = b1 * (unit / b1[norm_var])
    power = np.eye(m)
    out = []
    for _ in range(horizon + 1):
        out.append((sel @ power @ sel.T) @ impulse)
        power = power @ dgp["companion"]
    return np.asarray(out)


PROXY = make_proxy_dgp(
    [[[0.70, 0.10], [0.15, 0.50]]],
    [[1.00, 0.30], [0.50, 1.40]],
    "PROXY VAR(1), root 0.758, structural impact b_1 = (1.0, 0.5)",
)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def mc_se(p_hat, reps):
    """Monte Carlo standard error of a coverage estimate, in probability units."""
    return math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / reps)


def boot_seed(tag, n, rep):
    """A deterministic bootstrap seed that does not depend on call ordering."""
    return 1 + (SEED + 100003 * n + 7919 * rep + 104729 * tag) % (2**31 - 2)


def bands(y, lags, horizon, orth, cumulative, method, tag, n, rep, bias_correct=False):
    kwargs = {}
    if method == "bootstrap":
        kwargs = {"n_boot": N_BOOT, "seed": boot_seed(tag, n, rep), "bias_correct": bias_correct}
    return tsecon.var_irf_bands(
        y,
        lags=lags,
        horizon=horizon,
        orth=orth,
        method=method,
        alpha=ALPHA,
        cumulative=cumulative,
        **kwargs,
    )


def cell(res, key, resp, shock):
    return np.asarray(res[key])[:, resp, shock]


def cov_cell(value, se_value=None):
    """Format `coverage +/- mc_se` as a fixed-width cell in percentage points."""
    if value is None:
        return f"{'--':>8}"
    if se_value is None:
        return f"{100.0 * value:8.1f}"
    return f"{100.0 * value:4.1f}±{100.0 * se_value:3.1f}"


def rule(width=104, char="-"):
    print(char * width)


def header(title):
    print()
    rule()
    print(title)
    rule()


# ==========================================================================
# Experiment 1 -- the horizon profile, correctly specified
# ==========================================================================
def exp_horizon_profile(reps, horizon=12, ns=(100, 200, 500)):
    """Pointwise coverage by horizon, asymptotic vs bootstrap, `orth=True`.

    Tracks the response of variable 1 to an orthogonalised shock in variable 0
    -- a cell whose population impact response is 0.4, not a structural zero.
    Also records simultaneous ("joint") coverage: the fraction of samples in
    which the *entire* h = 0..H true path lies inside the pointwise band. These
    are pointwise bands and make no joint promise, so joint coverage is
    expected to be far below nominal; it is reported because published IRF
    figures are routinely read as if it were not.
    """
    resp, shock = 1, 0
    truth = true_irf(BASE, horizon)[:, resp, shock]
    out = {
        "name": "exp1_horizon_profile",
        "dgp": BASE["name"],
        "cell": "response of y1 to orthogonalised shock 0",
        "truth": truth.tolist(),
        "nominal": NOMINAL,
        "reps": reps,
        "by_n": {},
        "kind": "BAND-H, frequentist, correctly specified (lags=1 = true p)",
    }
    for n in ns:
        rng = np.random.default_rng(SEED + n)
        cov = {m: np.zeros(horizon + 1) for m in METHODS}
        joint = {m: 0 for m in METHODS}
        se_sum = {m: np.zeros(horizon + 1) for m in METHODS}
        width_sum = {m: np.zeros(horizon + 1) for m in METHODS}
        points = np.empty((reps, horizon + 1))
        tstat = np.empty((reps, horizon + 1))
        for rep in range(reps):
            y = simulate(BASE, n, rng)
            for tag, method in enumerate(METHODS):
                res = bands(y, 1, horizon, True, False, method, tag, n, rep)
                lo = cell(res, "lower", resp, shock)
                hi = cell(res, "upper", resp, shock)
                inside = (lo <= truth) & (truth <= hi)
                cov[method] += inside
                joint[method] += bool(inside.all())
                se_sum[method] += cell(res, "se", resp, shock)
                width_sum[method] += hi - lo
                if method == "asymptotic":
                    point = cell(res, "point", resp, shock)
                    points[rep] = point
                    # The Wald band covers iff |t| <= z, so the distribution of
                    # t IS the coverage story, exactly and not approximately.
                    tstat[rep] = (point - truth) / cell(res, "se", resp, shock)
        qs = np.percentile(tstat, [5, 50, 95], axis=0)
        centred = tstat - tstat.mean(axis=0)
        scale = tstat.std(axis=0, ddof=1)
        out["by_n"][n] = {
            "coverage": {m: (cov[m] / reps).tolist() for m in METHODS},
            "joint_coverage": {m: joint[m] / reps for m in METHODS},
            "mean_se": {m: (se_sum[m] / reps).tolist() for m in METHODS},
            "mean_width": {m: (width_sum[m] / reps).tolist() for m in METHODS},
            "mc_sd_point": points.std(axis=0, ddof=1).tolist(),
            "median_bias": (np.median(points, axis=0) - truth).tolist(),
            "se_over_mc_sd": (
                se_sum["asymptotic"] / reps / points.std(axis=0, ddof=1)
            ).tolist(),
            "t_q05": qs[0].tolist(),
            "t_q50": qs[1].tolist(),
            "t_q95": qs[2].tolist(),
            "t_skew": ((centred**3).mean(axis=0) / scale**3).tolist(),
        }
    return out


def report_horizon_profile(res):
    header(
        f"EXP 1  Horizon profile, correctly specified.  {res['dgp']}\n"
        f"        cell: {res['cell']};  nominal {100 * res['nominal']:.0f}% band;  "
        f"R = {res['reps']} reps, n_boot = {N_BOOT}"
    )
    for n, block in res["by_n"].items():
        print(f"\n  n = {n}")
        print(
            f"  {'h':>3} {'truth':>8} {'med bias':>9} {'mean se':>9} {'mc sd':>8} "
            f"{'|bias|/sd':>10} {'cov asym':>9} {'cov boot':>9}"
        )
        for h, truth in enumerate(res["truth"]):
            bias = block["median_bias"][h]
            sd = block["mc_sd_point"][h]
            ca = block["coverage"]["asymptotic"][h]
            cb = block["coverage"]["bootstrap"][h]
            print(
                f"  {h:>3} {truth:8.4f} {bias:9.4f} {block['mean_se']['asymptotic'][h]:9.4f} "
                f"{sd:8.4f} {abs(bias) / sd if sd > 0 else float('nan'):10.2f} "
                f"{cov_cell(ca, mc_se(ca, res['reps'])):>9} "
                f"{cov_cell(cb, mc_se(cb, res['reps'])):>9}"
            )
        ja = block["joint_coverage"]["asymptotic"]
        jb = block["joint_coverage"]["bootstrap"]
        print(
            f"  simultaneous coverage of the whole h=0..{len(res['truth']) - 1} path "
            f"(pointwise bands make NO such promise): "
            f"asym {cov_cell(ja, mc_se(ja, res['reps'])).strip()}  "
            f"boot {cov_cell(jb, mc_se(jb, res['reps'])).strip()}"
        )
        show = tuple(h for h in (0, 1, 2, 4, 6, 8, 12) if h < len(res["truth"]))
        z = norm.ppf(1 - ALPHA / 2)
        print(
            f"    standardised statistic t = (point - truth)/se, asymptotic arm. The Wald band\n"
            f"    covers exactly when |t| <= {z:.3f}, so these four rows ARE the coverage row above."
        )
        print("    " + f"{'':<16}" + "".join(f"{'h=' + str(h):>8}" for h in show))
        for key, label in (
            ("t_skew", "skewness"),
            ("t_q05", "5th pct"),
            ("t_q50", "median"),
            ("t_q95", "95th pct"),
        ):
            print("    " + f"{label:<16}" + "".join(f"{block[key][h]:8.2f}" for h in show))
        print(
            "    " + f"{'se / mc sd':<16}"
            + "".join(f"{block['se_over_mc_sd'][h]:8.2f}" for h in show)
        )


# ==========================================================================
# Experiment 2 -- orth on/off crossed with cumulative on/off
# ==========================================================================
def exp_orth_cumulative(reps, horizon=12, n=200):
    """Coverage across the `orth` x `cumulative` grid at a single sample size.

    Paired design: every configuration sees the *same* simulated samples, so
    differences between rows are differences between bands and not Monte Carlo
    noise between draws. The tracked cell is again (response 1, shock 0). With
    `orth=False` its impact value is a structural zero inside a zero-width
    band, so the h = 0 column there is degenerate by construction and is
    flagged, not interpreted.
    """
    resp, shock = 1, 0
    grid = [(orth, cum) for orth in (True, False) for cum in (False, True)]
    truths = {
        key: true_irf(BASE, horizon, orth=key[0], cumulative=key[1])[:, resp, shock]
        for key in grid
    }
    cov = {(key, m): np.zeros(horizon + 1) for key in grid for m in METHODS}
    zero_width = {(key, m): np.zeros(horizon + 1) for key in grid for m in METHODS}
    rng = np.random.default_rng(SEED + 77)
    for rep in range(reps):
        y = simulate(BASE, n, rng)
        for i, key in enumerate(grid):
            for j, method in enumerate(METHODS):
                res = bands(y, 1, horizon, key[0], key[1], method, 2 * i + j, n, rep)
                lo = cell(res, "lower", resp, shock)
                hi = cell(res, "upper", resp, shock)
                cov[(key, method)] += (lo <= truths[key]) & (truths[key] <= hi)
                zero_width[(key, method)] += (hi - lo) <= 1e-12
    return {
        "name": "exp2_orth_cumulative",
        "dgp": BASE["name"],
        "cell": "response of y1 to shock 0",
        "n": n,
        "reps": reps,
        "nominal": NOMINAL,
        "kind": "BAND-H, frequentist, paired across configurations",
        "grid": {
            f"orth={key[0]},cumulative={key[1]}": {
                "truth": truths[key].tolist(),
                "coverage": {m: (cov[(key, m)] / reps).tolist() for m in METHODS},
                "frac_zero_width": {
                    m: (zero_width[(key, m)] / reps).tolist() for m in METHODS
                },
            }
            for key in grid
        },
    }


def report_orth_cumulative(res, show=(0, 1, 2, 3, 4, 6, 8, 10, 12)):
    header(
        f"EXP 2  orth x cumulative.  {res['dgp']}, n = {res['n']}, "
        f"nominal {100 * res['nominal']:.0f}%, R = {res['reps']}\n"
        f"        cell: {res['cell']}.  Cells are coverage% +/- MC se, in percentage points."
    )
    print("  " + f"{'configuration':<41}" + "".join(f"{'h=' + str(h):>9}" for h in show))
    for key, block in res["grid"].items():
        for method in METHODS:
            cells = []
            for h in show:
                c = block["coverage"][method][h]
                degenerate = block["frac_zero_width"][method][h] > 0.999
                cells.append(
                    f"{'[exact]':>9}"
                    if degenerate
                    else f"{cov_cell(c, mc_se(c, res['reps'])):>9}"
                )
            print(f"  {key + ' / ' + method:<41}" + "".join(cells))
    print(
        "  [exact] = zero-width band at a structurally determined value "
        "(orth=False impact is the identity); covers by construction."
    )


# ==========================================================================
# Experiment 3 -- persistence and the Kilian bias correction
# ==========================================================================
def exp_bias_correct(reps, horizon=12, ns=(100, 200)):
    """Near-unit-root VAR: does `bias_correct=True` buy back coverage?

    Tracks the *own* response of the persistent variable (response 0 to
    orthogonalised shock 0), which is where least-squares shrinkage of the
    0.95 root does its damage: the estimated response decays too fast, so the
    band drifts below the truth as the horizon grows.
    """
    resp, shock = 0, 0
    truth = true_irf(PERSIST, horizon)[:, resp, shock]
    configs = [
        ("asymptotic", False),
        ("bootstrap", False),
        ("bootstrap", True),
    ]
    out = {
        "name": "exp3_bias_correct",
        "dgp": PERSIST["name"],
        "cell": "own response of y0 to orthogonalised shock 0",
        "truth": truth.tolist(),
        "nominal": NOMINAL,
        "reps": reps,
        "by_n": {},
        "kind": "BAND-H, frequentist, correctly specified but persistent",
    }
    for n in ns:
        rng = np.random.default_rng(SEED + 31 + n)
        cov, bias, width, shift = {}, {}, {}, {}
        for method, bc in configs:
            cov[(method, bc)] = np.zeros(horizon + 1)
            bias[(method, bc)] = np.zeros((reps, horizon + 1))
            width[(method, bc)] = np.zeros(horizon + 1)
            # (lower+upper)/2 - point: how far the band's centre sits from the
            # point estimate it is supposed to be a band around. Zero for a
            # symmetric Wald band by construction; for a percentile bootstrap
            # it picks up MINUS the bootstrap bias, which is how a downward
            # slope bias gets counted twice.
            shift[(method, bc)] = np.zeros((reps, horizon + 1))
        for rep in range(reps):
            y = simulate(PERSIST, n, rng)
            for tag, (method, bc) in enumerate(configs):
                res = bands(y, 1, horizon, True, False, method, tag, n, rep, bias_correct=bc)
                lo = cell(res, "lower", resp, shock)
                hi = cell(res, "upper", resp, shock)
                point = cell(res, "point", resp, shock)
                cov[(method, bc)] += (lo <= truth) & (truth <= hi)
                bias[(method, bc)][rep] = point - truth
                width[(method, bc)] += hi - lo
                shift[(method, bc)][rep] = 0.5 * (lo + hi) - point
        out["by_n"][n] = {
            f"{method}{'+bc' if bc else ''}": {
                "coverage": (cov[(method, bc)] / reps).tolist(),
                "median_bias": np.median(bias[(method, bc)], axis=0).tolist(),
                "mean_width": (width[(method, bc)] / reps).tolist(),
                "median_centre_shift": np.median(shift[(method, bc)], axis=0).tolist(),
            }
            for method, bc in configs
        }
    return out


def report_bias_correct(res, show=(0, 1, 2, 4, 6, 8, 10, 12)):
    header(
        f"EXP 3  Persistence and Kilian bias correction.  {res['dgp']}\n"
        f"        cell: {res['cell']};  nominal {100 * res['nominal']:.0f}%, R = {res['reps']}"
    )
    for n, block in res["by_n"].items():
        print(f"\n  n = {n}")
        print("  " + f"{'':<24}" + "".join(f"{'h=' + str(h):>9}" for h in show))
        print("  " + f"{'true response':<24}" + "".join(f"{res['truth'][h]:9.3f}" for h in show))
        for label, stats in block.items():
            print(
                "  "
                + f"{'cov ' + label:<24}"
                + "".join(
                    f"{cov_cell(stats['coverage'][h], mc_se(stats['coverage'][h], res['reps'])):>9}"
                    for h in show
                )
            )
        for label, stats in block.items():
            print(
                "  "
                + f"{'med bias ' + label:<24}"
                + "".join(f"{stats['median_bias'][h]:9.3f}" for h in show)
            )
        for label, stats in block.items():
            print(
                "  "
                + f"{'mean width ' + label:<24}"
                + "".join(f"{stats['mean_width'][h]:9.3f}" for h in show)
            )
        for label, stats in block.items():
            print(
                "  "
                + f"{'centre-point ' + label:<24}"
                + "".join(f"{stats['median_centre_shift'][h]:9.3f}" for h in show)
            )
    print(
        "  centre-point = median of (lower+upper)/2 minus the point estimate. It is 0 for the\n"
        "  symmetric Wald band by construction; a negative value for the percentile bootstrap\n"
        "  means the band sits BELOW the estimate, adding a second dose of the same downward bias."
    )


# ==========================================================================
# Experiment 4 -- lag misspecification: the target does not move
# ==========================================================================
def exp_misspecified(reps, horizon=12, ns=(200, 500)):
    """VAR(4) truth, fitted as VAR(1) (wrong) and as VAR(4) (right).

    The target is the *true* VAR(4) impulse response in both cases. A VAR(1)
    fit is not consistent for it, so this is a pure test of what a correctly
    computed band does when it is centred on the wrong number: nothing. The
    `|bias|/mc_sd` column is the whole story, and it does not improve with n.
    """
    resp, shock = 1, 0
    truth = true_irf(LAG4, horizon)[:, resp, shock]
    fits = [(1, "fit as VAR(1) [misspecified]"), (4, "fit as VAR(4) [correct]")]
    out = {
        "name": "exp4_misspecified",
        "dgp": LAG4["name"],
        "cell": "response of y1 to orthogonalised shock 0",
        "truth": truth.tolist(),
        "nominal": NOMINAL,
        "reps": reps,
        "by_n": {},
        "kind": "BAND-H, frequentist; lags=1 arm is inconsistent for the target",
    }
    for n in ns:
        rng = np.random.default_rng(SEED + 97 + n)
        cov = {(lags, m): np.zeros(horizon + 1) for lags, _ in fits for m in METHODS}
        pts = {lags: np.empty((reps, horizon + 1)) for lags, _ in fits}
        for rep in range(reps):
            y = simulate(LAG4, n, rng)
            for i, (lags, _) in enumerate(fits):
                for j, method in enumerate(METHODS):
                    res = bands(y, lags, horizon, True, False, method, 2 * i + j, n, rep)
                    lo = cell(res, "lower", resp, shock)
                    hi = cell(res, "upper", resp, shock)
                    cov[(lags, method)] += (lo <= truth) & (truth <= hi)
                    if method == "asymptotic":
                        pts[lags][rep] = cell(res, "point", resp, shock)
        out["by_n"][n] = {
            label: {
                "coverage": {m: (cov[(lags, m)] / reps).tolist() for m in METHODS},
                "median_bias": (np.median(pts[lags], axis=0) - truth).tolist(),
                "mc_sd_point": pts[lags].std(axis=0, ddof=1).tolist(),
            }
            for lags, label in fits
        }
    return out


def report_misspecified(res, show=(0, 1, 2, 3, 4, 5, 6, 8, 12)):
    header(
        f"EXP 4  Lag misspecification.  {res['dgp']}; target is the TRUE VAR(4) path\n"
        f"        cell: {res['cell']};  nominal {100 * res['nominal']:.0f}%, R = {res['reps']}"
    )
    for n, block in res["by_n"].items():
        print(f"\n  n = {n}")
        print("  " + f"{'':<32}" + "".join(f"{'h=' + str(h):>9}" for h in show))
        print("  " + f"{'true response':<32}" + "".join(f"{res['truth'][h]:9.3f}" for h in show))
        for label, stats in block.items():
            for method in METHODS:
                print(
                    "  "
                    + f"{'cov ' + method + ', ' + label.split(' [')[1][:-1]:<32}"
                    + "".join(
                        f"{cov_cell(stats['coverage'][method][h], mc_se(stats['coverage'][method][h], res['reps'])):>9}"
                        for h in show
                    )
                )
        for label, stats in block.items():
            ratios = []
            for h in show:
                sd = stats["mc_sd_point"][h]
                ratios.append(abs(stats["median_bias"][h]) / sd if sd > 0 else float("nan"))
            print(
                "  "
                + f"{'|bias|/mc_sd, ' + label.split(' [')[1][:-1]:<32}"
                + "".join(f"{r:9.2f}" for r in ratios)
            )


# ==========================================================================
# Experiment 5 -- is the shortfall a level problem or a shape problem?
# ==========================================================================
def exp_nominal_levels(reps, horizon=12, n=200, levels=(0.68, 0.90, 0.95)):
    """Sweep the nominal level. Coverage should track it, offset by the same gap.

    If coverage sits a roughly constant number of points below every nominal
    level, the band's *shape* is right and only its calibration is off. If the
    gap widens at the tails, the normal/percentile approximation to the tails
    of the sampling distribution is the culprit.
    """
    resp, shock = 1, 0
    truth = true_irf(BASE, horizon)[:, resp, shock]
    out = {
        "name": "exp5_nominal_levels",
        "dgp": BASE["name"],
        "cell": "response of y1 to orthogonalised shock 0",
        "n": n,
        "reps": reps,
        "levels": list(levels),
        "by_level": {},
        "kind": "BAND-H, frequentist, level calibration",
    }
    rng = np.random.default_rng(SEED + 555)
    cov = {(lvl, m): np.zeros(horizon + 1) for lvl in levels for m in METHODS}
    for rep in range(reps):
        y = simulate(BASE, n, rng)
        for i, lvl in enumerate(levels):
            for j, method in enumerate(METHODS):
                kwargs = {}
                if method == "bootstrap":
                    kwargs = {"n_boot": N_BOOT, "seed": boot_seed(2 * i + j, n, rep)}
                res = tsecon.var_irf_bands(
                    y,
                    lags=1,
                    horizon=horizon,
                    orth=True,
                    method=method,
                    alpha=1.0 - lvl,
                    **kwargs,
                )
                lo = cell(res, "lower", resp, shock)
                hi = cell(res, "upper", resp, shock)
                cov[(lvl, method)] += (lo <= truth) & (truth <= hi)
    for lvl in levels:
        out["by_level"][lvl] = {
            m: (cov[(lvl, m)] / reps).tolist() for m in METHODS
        }
    return out


def report_nominal_levels(res, show=(0, 2, 4, 8, 12)):
    header(
        f"EXP 5  Nominal level sweep.  {res['dgp']}, n = {res['n']}, R = {res['reps']}\n"
        f"        cell: {res['cell']}.  Read each row against its own nominal level."
    )
    print("  " + f"{'nominal / method':<28}" + "".join(f"{'h=' + str(h):>9}" for h in show))
    for lvl in res["levels"]:
        for method in METHODS:
            row = res["by_level"][lvl][method]
            print(
                "  "
                + f"{f'{100 * lvl:.0f}% / {method}':<28}"
                + "".join(f"{cov_cell(row[h], mc_se(row[h], res['reps'])):>9}" for h in show)
            )


# ==========================================================================
# Experiment 6 -- proxy-SVAR bands: moving-block vs wild, Hall vs Efron
# ==========================================================================
def exp_proxy_svar(reps, horizon=12, ns=(240, 480)):
    """Coverage of `proxy_svar_bands` on a known-truth external-instrument SVAR.

    Four things are measured on one set of draws, so every comparison below is
    paired and none of the differences is Monte Carlo noise between samples:

    1. `bands="moving_block"` (Jentsch-Lunsford) with a STRONG instrument.
       This is the headline: the band the library recommends, on the design it
       is entitled to do well on.
    2. `bands="wild"` on the IDENTICAL draws. The wild scheme applies a common
       Rademacher draw to the residuals and to the proxy, which leaves the
       identifying moment untouched, so the impact vector carries no bootstrap
       variability at all. If that critique is right, the `h = 0` band should
       be far too narrow and should NOT improve with the sample size.
    3. Hall (`lower`/`upper`) against Efron (`lower_efron`/`upper_efron`) on
       the identical bootstrap distribution -- the two come out of the same
       call, so the comparison is exact. Also counted: how often each covers
       when the other does not, which is what "paired" actually buys.
    4. An instrument-strength sweep (`rho` = 0.50 / 0.25 / 0.12, i.e. median
       first-stage F of roughly 72 / 15 / 3 at n = 240). A Wald-type band is
       not entitled to weak instruments and this is where it should show it.

    The tracked cell is the response of variable 1, whose `h = 0` value (0.5)
    is informative. Variable 0 is `norm_var`: its `h = 0` cell is degenerate at
    `unit` and is excluded from every average, its `h >= 1` cells are ordinary.
    """
    truth = true_proxy_irf(PROXY, horizon)
    k = PROXY["k"]
    strengths = [s for s, _ in PROXY_STRENGTHS]
    arms = [
        (s, b, kind)
        for s in strengths
        for b in PROXY_BANDS
        for kind, _, _ in PROXY_INTERVALS
    ]
    # the one cell that covers by construction and therefore measures nothing
    keep = np.ones((horizon + 1, k), dtype=bool)
    keep[0, PROXY_NORM_VAR] = False

    out = {
        "name": "exp6_proxy_svar",
        "dgp": PROXY["name"],
        "cell": f"response of y{PROXY_RESP} to the proxy-identified shock",
        "truth": truth[:, PROXY_RESP].tolist(),
        "truth_norm_var": truth[:, PROXY_NORM_VAR].tolist(),
        "norm_var": PROXY_NORM_VAR,
        "unit": PROXY_UNIT,
        "nominal": NOMINAL,
        "reps": reps,
        "n_boot": PROXY_N_BOOT,
        "strengths": dict(PROXY_STRENGTHS),
        "n_cells_nondegenerate": int(keep.sum()),
        "by_n": {},
        "kind": "BAND-H, frequentist; the wild arm is documented NOT valid here",
    }
    for n in ns:
        rng = np.random.default_rng(SEED + 1301 + n)
        cov = {a: np.zeros((horizon + 1, k)) for a in arms}
        width = {a: np.zeros((horizon + 1, k)) for a in arms}
        fstat = {s: np.empty(reps) for s in strengths}
        n_failed = {s: 0 for s in strengths}
        failures = {s: {} for s in strengths}
        points = {s: np.empty((reps, horizon + 1)) for s in strengths}
        hall_only = {s: np.zeros(horizon + 1) for s in strengths}
        efron_only = {s: np.zeros(horizon + 1) for s in strengths}
        blocks, valid_flag = set(), {}
        for rep in range(reps):
            for si, (s, rho) in enumerate(PROXY_STRENGTHS):
                y, m = simulate_proxy(PROXY, n, rng, rho)
                for bi, b in enumerate(PROXY_BANDS):
                    res = tsecon.proxy_svar_bands(
                        y,
                        m,
                        lags=PROXY["p"],
                        horizon=horizon,
                        norm_var=PROXY_NORM_VAR,
                        unit=PROXY_UNIT,
                        alpha=ALPHA,
                        n_boot=PROXY_N_BOOT,
                        seed=boot_seed(10 + 2 * si + bi, n, rep),
                        bands=b,
                    )
                    inside = {}
                    for kind, lo_key, hi_key in PROXY_INTERVALS:
                        lo = np.asarray(res[lo_key])
                        hi = np.asarray(res[hi_key])
                        ok = (lo <= truth) & (truth <= hi)
                        inside[kind] = ok
                        cov[(s, b, kind)] += ok
                        width[(s, b, kind)] += hi - lo
                    valid_flag[b] = bool(res["asymptotically_valid"])
                    if b != "moving_block":
                        continue
                    # diagnostics are recorded once per draw, off the valid arm
                    fstat[s][rep] = float(res["point_first_stage_f"])
                    n_failed[s] += int(res["n_failed"])
                    for reason, count in res["failures"].items():
                        failures[s][reason] = failures[s].get(reason, 0) + int(count)
                    points[s][rep] = np.asarray(res["point"])[:, PROXY_RESP]
                    blocks.add(int(res["block_length"]))
                    h_ok = inside["hall"][:, PROXY_RESP]
                    e_ok = inside["efron"][:, PROXY_RESP]
                    hall_only[s] += h_ok & ~e_ok
                    efron_only[s] += e_ok & ~h_ok
        out["by_n"][n] = {
            "block_length": sorted(blocks),
            "asymptotically_valid": dict(valid_flag),
            "median_first_stage_f": {s: float(np.median(fstat[s])) for s in strengths},
            "n_failed": dict(n_failed),
            "failures": {s: dict(failures[s]) for s in strengths},
            "median_bias": {
                s: (np.median(points[s], axis=0) - truth[:, PROXY_RESP]).tolist()
                for s in strengths
            },
            "mc_sd_point": {
                s: points[s].std(axis=0, ddof=1).tolist() for s in strengths
            },
            "paired": {
                s: {
                    "hall_only": (hall_only[s] / reps).tolist(),
                    "efron_only": (efron_only[s] / reps).tolist(),
                }
                for s in strengths
            },
            "arms": {
                "/".join(a): {
                    "coverage": (cov[a][:, PROXY_RESP] / reps).tolist(),
                    "coverage_norm_var": (cov[a][:, PROXY_NORM_VAR] / reps).tolist(),
                    "mean_width": (width[a][:, PROXY_RESP] / reps).tolist(),
                    # the two averages the module refuses to conflate
                    "mean_coverage_excl_degenerate": float(
                        (cov[a] / reps)[keep].mean()
                    ),
                    "mean_coverage_incl_degenerate": float((cov[a] / reps).mean()),
                    "h0_avg_over_variables_incl_degenerate": float(
                        (cov[a][0] / reps).mean()
                    ),
                }
                for a in arms
            },
        }
    return out


def report_proxy_svar(res, show=(0, 1, 2, 3, 4, 6, 8, 12)):
    header(
        f"EXP 6  Proxy-SVAR bands.  {res['dgp']}\n"
        f"        cell: {res['cell']};  nominal {100 * res['nominal']:.0f}%, "
        f"R = {res['reps']}, n_boot = {res['n_boot']};  "
        f"normalisation: unit {res['unit']:.1f} effect on y{res['norm_var']}"
    )
    show = tuple(h for h in show if h < len(res["truth"]))
    for n, block in res["by_n"].items():
        fs = block["median_first_stage_f"]
        print(
            f"\n  n = {n}   median first-stage F: "
            + ", ".join(
                f"{s} (rho={res['strengths'][s]:.2f}) {fs[s]:.1f}" for s in fs
            )
            + f"   block length {block['block_length']}"
        )
        print(
            "  failed bootstrap draws, moving-block arm: "
            + ", ".join(f"{s} {block['n_failed'][s]}" for s in block["n_failed"])
            + f"  (out of {res['reps'] * res['n_boot']:,} per strength; "
            "failed draws are counted by reason, never dropped)"
        )
        print(
            "  " + f"{'true response':<34}" + "".join(f"{res['truth'][h]:9.3f}" for h in show)
        )
        print("  " + f"{'arm':<34}" + "".join(f"{'h=' + str(h):>9}" for h in show) + "   mean*")
        for arm, stats in block["arms"].items():
            row = "".join(
                f"{cov_cell(stats['coverage'][h], mc_se(stats['coverage'][h], res['reps'])):>9}"
                for h in show
            )
            print(f"  {arm:<34}{row}{100 * stats['mean_coverage_excl_degenerate']:8.1f}")
        print(
            f"  mean* = average over all {res['n_cells_nondegenerate']} non-degenerate "
            f"(h, variable) cells, EXCLUDING (y{res['norm_var']}, h=0). Its MC se is bounded "
            f"above by the\n  single-cell figure printed in each row, since the cells within a "
            "replication are positively correlated."
        )
        print(
            "  asymptotically_valid, as the library reports it: "
            + ", ".join(f"{b} {v}" for b, v in block["asymptotically_valid"].items())
            + ". The wild rows are NOT inference."
        )
        print("\n  mean band width, same cell (a band that covers by being useless is not a win)")
        for arm, stats in block["arms"].items():
            if arm.endswith("/efron"):
                # Hall and Efron are the SAME two bootstrap quantiles, reflected
                # about the point estimate, so their widths are identical to
                # floating point. Only the location differs. Verified below.
                continue
            print(
                f"  {arm.replace('/hall', ''):<34}"
                + "".join(f"{stats['mean_width'][h]:9.3f}" for h in show)
            )
        print(
            "\n  paired Hall vs Efron on the identical moving-block draws "
            "(% of samples where exactly one covers)"
        )
        for s, pair in block["paired"].items():
            print(
                f"  {s + ' Hall covers, Efron misses':<34}"
                + "".join(f"{100 * pair['hall_only'][h]:9.1f}" for h in show)
            )
            print(
                f"  {s + ' Efron covers, Hall misses':<34}"
                + "".join(f"{100 * pair['efron_only'][h]:9.1f}" for h in show)
            )
        wild = block["arms"]["strong/wild/hall"]
        print(
            f"\n  the excluded cell, made concrete: for strong/wild/hall the h=0 average over "
            f"BOTH variables is {100 * wild['h0_avg_over_variables_incl_degenerate']:.1f}%, "
            f"which is 100.0% (the\n  normalisation-pinned y{res['norm_var']} cell, degenerate) "
            f"averaged with {100 * wild['coverage'][0]:.1f}% (the real one). "
            "Only the second number means anything."
        )


# ==========================================================================
# structural facts and the assertions worth making
# ==========================================================================
def structural_checks():
    """Exact, non-statistical facts about the band layout. These are not
    coverage claims; they are the reason certain cells are excluded from
    coverage claims."""
    rng = np.random.default_rng(SEED + 4242)
    y = simulate(BASE, 300, rng)
    facts = {}
    for method in METHODS:
        kwargs = {"n_boot": N_BOOT, "seed": 11} if method == "bootstrap" else {}
        orth_res = tsecon.var_irf_bands(
            y, lags=1, horizon=4, orth=True, method=method, alpha=ALPHA, **kwargs
        )
        raw_res = tsecon.var_irf_bands(
            y, lags=1, horizon=4, orth=False, method=method, alpha=ALPHA, **kwargs
        )
        cum_res = tsecon.var_irf_bands(
            y,
            lags=1,
            horizon=4,
            orth=True,
            method=method,
            alpha=ALPHA,
            cumulative=True,
            **kwargs,
        )
        facts[method] = {
            # A cumulative response at h=0 IS the impact response, so its band
            # must be bit-identical. If it is not, `cumulative` is doing
            # something to the variance that it should not.
            "cumulative_impact_matches": bool(
                np.allclose(np.asarray(cum_res["lower"])[0], np.asarray(orth_res["lower"])[0], atol=0)
                and np.allclose(
                    np.asarray(cum_res["upper"])[0], np.asarray(orth_res["upper"])[0], atol=0
                )
            ),
            "cholesky_impact_zero_se": float(np.asarray(orth_res["se"])[0, 0, 1]),
            "cholesky_impact_width": float(
                np.asarray(orth_res["upper"])[0, 0, 1] - np.asarray(orth_res["lower"])[0, 0, 1]
            ),
            "raw_impact_max_width": float(
                np.abs(
                    np.asarray(raw_res["upper"])[0] - np.asarray(raw_res["lower"])[0]
                ).max()
            ),
            "raw_impact_is_identity": bool(
                np.allclose(np.asarray(raw_res["point"])[0], np.eye(2), atol=1e-12)
            ),
        }
    facts["asymptotic_halfwidth_over_se"] = _halfwidth_ratio(y)
    return facts


def proxy_structural_checks(big_t=80_000):
    """Exact facts about `proxy_svar_bands`, plus one convention check.

    The convention check is the only thing in this file that verifies the
    *target* rather than the band: it fits the estimator to an 80,000
    observation draw and compares against the closed form. It is not a
    coverage claim -- it is the guarantee that experiment 6 is aiming at the
    number the estimator is actually consistent for, so that a shortfall there
    can be read as a band problem instead of a bookkeeping error.
    """
    rng = np.random.default_rng(SEED + 8181)
    facts = {}

    y_big, m_big = simulate_proxy(PROXY, big_t, rng, 0.5)
    fit = tsecon.proxy_svar(y_big, m_big, lags=PROXY["p"], horizon=6)
    facts["big_t"] = big_t
    facts["big_t_max_abs_dev"] = float(
        np.abs(np.asarray(fit["irf"]) - true_proxy_irf(PROXY, 6)).max()
    )

    y, m = simulate_proxy(PROXY, 300, rng, 0.5)
    for b in PROXY_BANDS:
        res = tsecon.proxy_svar_bands(
            y,
            m,
            lags=PROXY["p"],
            horizon=6,
            norm_var=PROXY_NORM_VAR,
            unit=PROXY_UNIT,
            alpha=ALPHA,
            n_boot=PROXY_N_BOOT,
            seed=13,
            bands=b,
        )
        nv = PROXY_NORM_VAR
        pinned = [
            float(np.asarray(res[key])[0, nv])
            for key in ("point", "lower", "upper", "lower_efron", "upper_efron")
        ]
        note = res["validity_note"] or ""
        hall_w = np.asarray(res["upper"]) - np.asarray(res["lower"])
        efron_w = np.asarray(res["upper_efron"]) - np.asarray(res["lower_efron"])
        facts[b] = {
            # The normalisation is re-imposed INSIDE every draw, so this cell
            # is pinned at `unit` exactly -- in the point estimate and in both
            # interval types. A non-degenerate value here would mean the
            # normalisation had been hoisted out of the bootstrap loop.
            "pinned_cell": pinned,
            "pinned_exactly": all(v == PROXY_UNIT for v in pinned),
            "pinned_se": float(np.asarray(res["se"])[0, nv]),
            "asymptotically_valid": bool(res["asymptotically_valid"]),
            "validity_note_chars": len(note),
            "draws_accounted": int(res["n_used"]) + int(res["n_failed"]),
            "n_boot": int(res["n_boot"]),
            "failure_reasons": sorted(res["failures"]),
            # Hall and Efron are one bootstrap distribution read two ways.
            "hall_efron_width_gap": float(np.abs(hall_w - efron_w).max()),
        }
    return facts


def _halfwidth_ratio(y):
    """The asymptotic band is `point +/- z_{1-alpha/2} se`; confirm the z."""
    res = tsecon.var_irf_bands(y, lags=1, horizon=4, method="asymptotic", alpha=ALPHA)
    se = np.asarray(res["se"])
    lo = np.asarray(res["lower"])
    hi = np.asarray(res["upper"])
    mask = se > 1e-12
    return float(np.median(((hi - lo) / 2)[mask] / se[mask]))


def assertions(results, facts, reps):
    """Only claims that are robust by construction or by a wide statistical
    margin. Nothing here was tuned to pass; the numbers that do *not* hold up
    are reported in the tables above and in `findings()`, not asserted away."""
    checks = []

    def check(label, ok, detail):
        checks.append((label, bool(ok), detail))

    # --- exact structural facts (no statistics involved) -------------------
    for method in METHODS:
        f = facts[method]
        check(
            f"[{method}] Cholesky structural zero has se == 0 and zero-width band",
            f["cholesky_impact_zero_se"] == 0.0 and f["cholesky_impact_width"] == 0.0,
            f"se={f['cholesky_impact_zero_se']}, width={f['cholesky_impact_width']}",
        )
        check(
            f"[{method}] orth=False impact band is the identity with zero width",
            f["raw_impact_is_identity"] and f["raw_impact_max_width"] < 1e-12,
            f"max width={f['raw_impact_max_width']:.2e}",
        )
        check(
            f"[{method}] cumulative band at h=0 is bit-identical to the impact band",
            f["cumulative_impact_matches"],
            "cumsum over a single horizon must be a no-op",
        )
    check(
        "asymptotic band is point +/- z_{1-alpha/2} * se (z = 1.6449 at alpha=0.10)",
        abs(facts["asymptotic_halfwidth_over_se"] - norm.ppf(1 - ALPHA / 2)) < 1e-6,
        f"ratio={facts['asymptotic_halfwidth_over_se']:.6f}",
    )

    # --- impact coverage, correctly specified, largest n -------------------
    # The impact orthogonalised response is a smooth function of Sigma-hat
    # alone; its asymptotics are the least demanding thing in the file, so if
    # anything covers it is this. 3 MC se is a principled threshold, not a
    # fitted one.
    e1 = results["exp1"]
    biggest = max(e1["by_n"])
    for method in METHODS:
        c = e1["by_n"][biggest]["coverage"][method][0]
        tol = 3.0 * mc_se(c, reps)
        check(
            f"[{method}] impact (h=0) coverage within 3 MC se of nominal at n={biggest}",
            abs(c - NOMINAL) <= tol,
            f"coverage={100 * c:.1f}% vs {100 * NOMINAL:.0f}%, 3*mc_se={100 * tol:.1f}pp",
        )

    # --- coverage deteriorates with the horizon ---------------------------
    # A polynomial in the estimated slopes, plus a normal approximation, does
    # not get better as the polynomial degree rises. This is an ordering
    # claim, robust to the exact numbers.
    small = min(e1["by_n"])
    hmax = len(e1["truth"]) - 1
    for method in METHODS:
        prof = e1["by_n"][small]["coverage"][method]
        check(
            f"[{method}] long-horizon coverage below impact coverage at n={small}",
            prof[-1] < prof[0],
            f"h=0 {100 * prof[0]:.1f}% -> h={hmax} {100 * prof[-1]:.1f}%",
        )

    # --- the reported standard error is not the culprit --------------------
    # If mean(se) tracks the actual Monte Carlo sd of the estimate but coverage
    # still falls short, the variance is right and the *shape* of the sampling
    # distribution is what breaks. A +/-15% window is loose on purpose.
    for n, block in e1["by_n"].items():
        ratio = block["se_over_mc_sd"][-1]
        check(
            f"[n={n}] reported se tracks the true sampling sd at h={hmax} (within 15%)",
            0.85 < ratio < 1.15,
            f"mean se / mc sd = {ratio:.3f} while coverage is "
            f"{100 * block['coverage']['asymptotic'][-1]:.1f}% -- so the shortfall is the "
            f"normal approximation (t skew {block['t_skew'][-1]:+.2f}, "
            f"5th/95th pct {block['t_q05'][-1]:+.2f}/{block['t_q95'][-1]:+.2f} "
            f"against +/-{norm.ppf(1 - ALPHA / 2):.2f}), not an understated se",
        )

    # --- Kilian (1998) bias correction on a persistent VAR ----------------
    # Two claims with theory behind them and ~25-50pp of measured margin:
    # (i) a percentile bootstrap around a downward-biased estimate inherits the
    # bias twice and so does WORSE than the symmetric Wald band; (ii) turning
    # on bias_correct removes most of it.
    e3 = results["exp3"]
    for n, block in e3["by_n"].items():
        plain = block["bootstrap"]["coverage"][-1]
        corrected = block["bootstrap+bc"]["coverage"][-1]
        asym = block["asymptotic"]["coverage"][-1]
        check(
            f"[n={n}] persistent VAR: bias_correct=True lifts h={hmax} bootstrap coverage by >20pp",
            corrected - plain > 0.20,
            f"{100 * plain:.1f}% -> {100 * corrected:.1f}% (nominal {100 * NOMINAL:.0f}%)",
        )
        check(
            f"[n={n}] persistent VAR: UNcorrected percentile bootstrap is worse than the Wald band",
            plain < asym - 0.10,
            f"bootstrap {100 * plain:.1f}% vs asymptotic {100 * asym:.1f}%; the bootstrap band "
            f"centre sits {block['bootstrap']['median_centre_shift'][-1]:+.3f} from the estimate, "
            f"which is already {block['bootstrap']['median_bias'][-1]:+.3f} from the truth",
        )

    # --- pointwise bands do not deliver joint coverage --------------------
    for method in METHODS:
        joint = e1["by_n"][biggest]["joint_coverage"][method]
        check(
            f"[{method}] simultaneous coverage over {hmax + 1} horizons far below nominal",
            joint < NOMINAL - 3.0 * mc_se(joint, reps),
            f"joint={100 * joint:.1f}% vs pointwise nominal {100 * NOMINAL:.0f}%",
        )

    # --- misspecification destroys coverage; correct lags restore it ------
    e4 = results["exp4"]
    for n, block in e4["by_n"].items():
        wrong = block["fit as VAR(1) [misspecified]"]["coverage"]["asymptotic"]
        right = block["fit as VAR(4) [correct]"]["coverage"]["asymptotic"]
        check(
            f"[n={n}] VAR(1) fit to VAR(4) truth loses coverage at h=4 (bias, not variance)",
            wrong[4] < 0.5,
            f"coverage={100 * wrong[4]:.1f}%",
        )
        check(
            f"[n={n}] correct lag order recovers far more coverage at h=4",
            right[4] - wrong[4] > 0.25,
            f"VAR(4) {100 * right[4]:.1f}% vs VAR(1) {100 * wrong[4]:.1f}%",
        )
    # inconsistency, not slow convergence: more data makes coverage worse.
    ns4 = sorted(e4["by_n"])
    if len(ns4) > 1:
        lo_n, hi_n = ns4[0], ns4[-1]
        a = e4["by_n"][lo_n]["fit as VAR(1) [misspecified]"]["coverage"]["asymptotic"][4]
        b = e4["by_n"][hi_n]["fit as VAR(1) [misspecified]"]["coverage"]["asymptotic"][4]
        check(
            "misspecified coverage does NOT improve with n (it is inconsistency)",
            b <= a,
            f"n={lo_n}: {100 * a:.1f}%  ->  n={hi_n}: {100 * b:.1f}%",
        )

    # ======================================================================
    # proxy SVAR
    # ======================================================================
    pf = facts["proxy"]
    check(
        "proxy-SVAR target matches the estimator's normalisation convention "
        f"at T={pf['big_t']:,}",
        pf["big_t_max_abs_dev"] < 0.03,
        f"max |fitted - closed form| = {pf['big_t_max_abs_dev']:.4f} over h=0..6; "
        "a wrong impulse vector or a wrong scale would be off by an order of "
        "magnitude more, so experiment 6 is aiming at the right number",
    )
    for b in PROXY_BANDS:
        f = pf[b]
        check(
            f"[proxy/{b}] the (norm_var, h=0) cell is pinned at unit exactly, "
            "in point and BOTH interval types",
            f["pinned_exactly"] and f["pinned_se"] == 0.0,
            f"point/lower/upper/lower_efron/upper_efron = {f['pinned_cell']}, "
            f"se = {f['pinned_se']} -- the normalisation is re-imposed inside "
            "every draw, which is why this cell is excluded from every average",
        )
        check(
            f"[proxy/{b}] every bootstrap draw is accounted for (used + failed "
            "== n_boot), with all six failure reasons reported",
            f["draws_accounted"] == f["n_boot"] and len(f["failure_reasons"]) == 6,
            f"{f['draws_accounted']} == {f['n_boot']}; reasons: "
            + ", ".join(f["failure_reasons"]),
        )
        check(
            f"[proxy/{b}] Hall and Efron are the same two quantiles reflected "
            "about the point: identical width, different location",
            f["hall_efron_width_gap"] < 1e-9,
            f"max |width_hall - width_efron| = {f['hall_efron_width_gap']:.2e}",
        )
    check(
        "the library flags the wild bootstrap as NOT asymptotically valid here "
        "and the moving block as valid",
        pf["moving_block"]["asymptotically_valid"]
        and not pf["wild"]["asymptotically_valid"]
        and pf["wild"]["validity_note_chars"] > 100,
        f"moving_block valid={pf['moving_block']['asymptotically_valid']}, "
        f"wild valid={pf['wild']['asymptotically_valid']} with a "
        f"{pf['wild']['validity_note_chars']}-character validity_note",
    )

    e6 = results["exp6"]
    e6_ns = sorted(e6["by_n"])
    hmax6 = len(e6["truth"]) - 1
    for n in e6_ns:
        blk = e6["by_n"][n]
        mbb = blk["arms"]["strong/moving_block/hall"]
        wild = blk["arms"]["strong/wild/hall"]
        # -- the headline: the recommended band on the design it is entitled to
        check(
            f"[n={n}] strong instrument, moving-block Hall: impact coverage is "
            "in the right neighbourhood (>= 80% against a 90% promise)",
            mbb["coverage"][0] >= 0.80,
            f"coverage={100 * mbb['coverage'][0]:.1f}% +/- "
            f"{100 * mc_se(mbb['coverage'][0], reps):.1f}pp, median first-stage "
            f"F = {blk['median_first_stage_f']['strong']:.1f}",
        )
        # -- the Jentsch-Lunsford critique, reproduced on our own code ---------
        # A common Rademacher draw on residuals AND proxy cancels out of the
        # identifying moment, so the impact vector has no bootstrap variability
        # to speak of. The prediction is a far-too-narrow h=0 band. It is not a
        # close call: the measured gap is tens of percentage points.
        check(
            f"[n={n}] wild bootstrap catastrophically under-covers the IMPACT "
            "response (< 40% against a 90% promise)",
            wild["coverage"][0] < 0.40,
            f"wild {100 * wild['coverage'][0]:.1f}% vs moving-block "
            f"{100 * mbb['coverage'][0]:.1f}% on the SAME draws",
        )
        check(
            f"[n={n}] moving block beats wild at impact by more than 40pp",
            mbb["coverage"][0] - wild["coverage"][0] > 0.40,
            f"{100 * (mbb['coverage'][0] - wild['coverage'][0]):.1f}pp",
        )
        check(
            f"[n={n}] the mechanism: the wild impact band is less than a "
            "quarter the width of the moving-block one",
            wild["mean_width"][0] < 0.25 * mbb["mean_width"][0],
            f"mean width {wild['mean_width'][0]:.3f} vs {mbb['mean_width'][0]:.3f} "
            f"(ratio {wild['mean_width'][0] / mbb['mean_width'][0]:.3f}) -- the "
            "identifying moment is invariant to the Rademacher draw, so the "
            "impact vector barely moves across draws",
        )
        # -- weak instruments: a Wald-type band is not entitled to them --------
        weak = blk["arms"]["weak/moving_block/hall"]
        check(
            f"[n={n}] weak instrument (median F = "
            f"{blk['median_first_stage_f']['weak']:.1f}): the moving-block band "
            "goes uninformative -- more than 4x wider at impact",
            weak["mean_width"][0] > 4.0 * mbb["mean_width"][0],
            f"width {weak['mean_width'][0]:.3f} vs {mbb['mean_width'][0]:.3f} "
            f"({weak['mean_width'][0] / mbb['mean_width'][0]:.1f}x) around a true "
            f"response of {e6['truth'][0]:.2f}",
        )
        check(
            f"[n={n}] weak instrument: the moving-block band does NOT lose "
            "coverage at impact -- it loses width",
            weak["coverage"][0] >= NOMINAL - 0.02,
            f"coverage={100 * weak['coverage'][0]:.1f}% against a "
            f"{100 * NOMINAL:.0f}% promise "
            f"({100 * (weak['coverage'][0] - NOMINAL):+.1f}pp), while the band is "
            f"{weak['mean_width'][0] / mbb['mean_width'][0]:.1f}x wider than the "
            "strong-instrument one. The Wald-type band degrades by going "
            "uninformative, not by missing",
        )
        check(
            f"[n={n}] weak instrument AND the wild bootstrap is the worst "
            "combination in the file: h=1 coverage below 60%",
            blk["arms"]["weak/wild/hall"]["coverage"][1] < 0.60,
            f"coverage={100 * blk['arms']['weak/wild/hall']['coverage'][1]:.1f}%",
        )
        # -- ordering claims, robust to the exact numbers ----------------------
        check(
            f"[n={n}] moving-block coverage decays with the horizon, as every "
            "other band in this file does",
            mbb["coverage"][-1] < mbb["coverage"][0],
            f"h=0 {100 * mbb['coverage'][0]:.1f}% -> h={hmax6} "
            f"{100 * mbb['coverage'][-1]:.1f}%",
        )
        widths = [
            blk["arms"][f"{s}/moving_block/hall"]["mean_width"][0]
            for s, _ in PROXY_STRENGTHS
        ]
        check(
            f"[n={n}] impact band width is monotone in instrument strength",
            widths[0] < widths[1] < widths[2],
            " < ".join(f"{w:.3f}" for w in widths)
            + " for rho = "
            + "/".join(f"{r:.2f}" for _, r in PROXY_STRENGTHS),
        )
        # -- Hall vs Efron, measured on identical draws ------------------------
        efron = blk["arms"]["strong/moving_block/efron"]
        pair = blk["paired"]["strong"]
        # An ordering claim on identical draws, not a threshold: the paired
        # discordance is the evidence, and it runs one way at every n and
        # every horizon measured here.
        check(
            f"[n={n}] at the longest horizon Efron covers more than Hall, on "
            "the identical bootstrap draws",
            efron["coverage"][-1] - mbb["coverage"][-1] > 0.02
            and pair["efron_only"][-1] > pair["hall_only"][-1],
            f"h={hmax6}: Hall {100 * mbb['coverage'][-1]:.1f}% vs Efron "
            f"{100 * efron['coverage'][-1]:.1f}% "
            f"({100 * (efron['coverage'][-1] - mbb['coverage'][-1]):+.1f}pp); paired, "
            f"Efron-covers-Hall-misses {100 * pair['efron_only'][-1]:.1f}% of samples "
            f"against {100 * pair['hall_only'][-1]:.1f}% the other way. Same width "
            "either way, so this is entirely about where the band sits",
        )
        # -- the excluded cell is not a rounding detail ------------------------
        check(
            f"[n={n}] including the degenerate cell would overstate the wild "
            "arm's h=0 coverage by more than 20pp",
            wild["h0_avg_over_variables_incl_degenerate"] - wild["coverage"][0] > 0.20,
            f"h=0 average over both variables "
            f"{100 * wild['h0_avg_over_variables_incl_degenerate']:.1f}% vs the "
            f"informative cell alone {100 * wild['coverage'][0]:.1f}%",
        )
        check(
            f"[n={n}] no bootstrap draw failed, so no coverage number in "
            "experiment 6 is conditioned on a discarded tail",
            all(v == 0 for v in blk["n_failed"].values()),
            "n_failed = "
            + ", ".join(f"{s} {v}" for s, v in blk["n_failed"].items())
            + f" out of {reps * e6['n_boot']} draws each",
        )
    # The wild bootstrap's impact failure is INCONSISTENCY, not a small-sample
    # artefact: doubling the sample does not buy any of it back.
    if len(e6_ns) > 1:
        lo_n, hi_n = e6_ns[0], e6_ns[-1]
        a = e6["by_n"][lo_n]["arms"]["strong/wild/hall"]["coverage"][0]
        b = e6["by_n"][hi_n]["arms"]["strong/wild/hall"]["coverage"][0]
        check(
            "wild-bootstrap impact coverage does NOT improve with n (it is "
            "invalidity, not a finite-sample approximation)",
            b <= a + 0.02,
            f"n={lo_n}: {100 * a:.1f}%  ->  n={hi_n}: {100 * b:.1f}%",
        )

    # --- coverage is monotone in the nominal level ------------------------
    e5 = results["exp5"]
    lv = e5["levels"]
    for method in METHODS:
        seq = [e5["by_level"][lvl][method][0] for lvl in lv]
        check(
            f"[{method}] alpha is honoured: impact coverage rises materially with the level",
            all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))
            and seq[-1] - seq[0] > 0.15,
            " < ".join(f"{100 * s:.1f}%" for s in seq)
            + f" across nominal {'/'.join(f'{100 * x:.0f}%' for x in lv)}",
        )

    return checks


def findings(results, reps):
    """The honest list: where the bands miss, and which kind of miss it is."""
    lines = []
    e1, e2 = results["exp1"], results["exp2"]
    e3, e4, e5 = results["exp3"], results["exp4"], results["exp5"]
    z = norm.ppf(1 - ALPHA / 2)

    for n, block in e1["by_n"].items():
        for method in METHODS:
            prof = block["coverage"][method]
            worst_h = int(np.argmin(prof))
            worst = prof[worst_h]
            gap_se = (NOMINAL - worst) / max(mc_se(worst, reps), 1e-12)
            ratio = block["mean_se"][method][worst_h] / block["mc_sd_point"][worst_h]
            lines.append(
                f"EXP1 n={n:<4} {method:<11} worst pointwise coverage {100 * worst:.1f}% at h={worst_h}, "
                f"i.e. {gap_se:.1f} MC se BELOW the {100 * NOMINAL:.0f}% promise; "
                f"mean se {block['mean_se'][method][worst_h]:.4f} vs MC sd of the estimate "
                f"{block['mc_sd_point'][worst_h]:.4f} (ratio {ratio:.2f} -- "
                + (
                    "the se is NOT the problem"
                    if 0.85 < ratio < 1.15
                    else "the se itself is materially off"
                )
                + f"), median bias {block['median_bias'][worst_h]:+.4f}"
            )
        # The t-statistic diagnostic is defined for the Wald arm only.
        worst_h = int(np.argmin(block["coverage"]["asymptotic"]))
        lines.append(
            f"EXP1 n={n:<4} asymptotic  at h={worst_h} the standardised statistic has skewness "
            f"{block['t_skew'][worst_h]:+.2f} and 5th/95th percentiles "
            f"{block['t_q05'][worst_h]:+.2f}/{block['t_q95'][worst_h]:+.2f} against the "
            f"+/-{z:.2f} the Wald band assumes. The lower edge is badly too high and the "
            f"upper edge is never reached: the band is one-sidedly wrong, not too narrow."
        )
        joint = block["joint_coverage"]["asymptotic"]
        lines.append(
            f"EXP1 n={n:<4} simultaneous coverage of the full {len(e1['truth'])}-horizon path is "
            f"{100 * joint:.1f}% (asymptotic) -- pointwise bands, by design, make no "
            f"joint promise. NOT a defect; a reading hazard."
        )
    for key, block in e2["grid"].items():
        for method in METHODS:
            prof = block["coverage"][method]
            lines.append(
                f"EXP2 n={e2['n']} {key + '/' + method:<44} coverage h=1 {100 * prof[1]:.1f}%, "
                f"h=4 {100 * prof[4]:.1f}%, h={len(prof) - 1} {100 * prof[-1]:.1f}%"
            )
    lines.append(
        "EXP2 paired comparison: on this DGP the CUMULATIVE band holds up far better at long "
        "horizons than the per-horizon band -- the running sum is dominated by the early, "
        "well-estimated horizons, so it is a much more nearly linear function of A-hat. Not a "
        "general theorem; measured here."
    )
    hmax = len(e3["truth"]) - 1
    for n, block in e3["by_n"].items():
        for label, stats in block.items():
            prof = stats["coverage"]
            lines.append(
                f"EXP3 n={n:<4} {label:<14} persistent DGP: coverage h=0 {100 * prof[0]:.1f}% -> "
                f"h={hmax} {100 * prof[-1]:.1f}%; median bias at h={hmax} "
                f"{stats['median_bias'][-1]:+.3f} against a true response of {e3['truth'][-1]:.3f}; "
                f"band centre sits {stats['median_centre_shift'][-1]:+.3f} from the point estimate"
            )
    for n, block in e4["by_n"].items():
        for label, stats in block.items():
            prof = stats["coverage"]["asymptotic"]
            lines.append(
                f"EXP4 n={n:<4} {label:<30} coverage h=0 {100 * prof[0]:.1f}%, "
                f"h=4 {100 * prof[4]:.1f}%, h={hmax} {100 * prof[-1]:.1f}%; "
                f"|bias|/mc_sd at h=4 = "
                f"{abs(stats['median_bias'][4]) / max(stats['mc_sd_point'][4], 1e-12):.2f}"
            )
    for lvl in e5["levels"]:
        for method in METHODS:
            row = e5["by_level"][lvl][method]
            lines.append(
                f"EXP5 nominal {100 * lvl:.0f}% {method:<11} coverage h=0 {100 * row[0]:.1f}%, "
                f"h=4 {100 * row[4]:.1f}%, h={hmax} {100 * row[-1]:.1f}% "
                f"(under-covers by {100 * (lvl - row[-1]):.1f}pp at h={hmax})"
            )

    e6 = results["exp6"]
    hmax6 = len(e6["truth"]) - 1
    for n, blk in e6["by_n"].items():
        for arm, stats in blk["arms"].items():
            prof = stats["coverage"]
            strength = arm.split("/")[0]
            lines.append(
                f"EXP6 n={n:<4} {arm:<32} F~{blk['median_first_stage_f'][strength]:6.1f}  "
                f"coverage h=0 {100 * prof[0]:.1f}%, h=1 {100 * prof[1]:.1f}%, "
                f"h=4 {100 * prof[4]:.1f}%, h={hmax6} {100 * prof[-1]:.1f}%; "
                f"mean over the {e6['n_cells_nondegenerate']} non-degenerate cells "
                f"{100 * stats['mean_coverage_excl_degenerate']:.1f}%"
            )
    for n, blk in e6["by_n"].items():
        mbb = blk["arms"]["strong/moving_block/hall"]
        wild = blk["arms"]["strong/wild/hall"]
        lines.append(
            f"EXP6 n={n:<4} THE JENTSCH-LUNSFORD CRITIQUE, REPRODUCED ON THIS IMPLEMENTATION: "
            f"with a strong instrument the wild bootstrap covers the IMPACT response "
            f"{100 * wild['coverage'][0]:.1f}% of the time against a "
            f"{100 * NOMINAL:.0f}% promise, because its impact band is "
            f"{mbb['mean_width'][0] / max(wild['mean_width'][0], 1e-12):.1f}x too narrow "
            f"({wild['mean_width'][0]:.3f} against the moving block's "
            f"{mbb['mean_width'][0]:.3f}). The common Rademacher draw cancels out of "
            f"sum_t m_t u_t', so the impulse vector is nearly frozen across draws. "
            f"Use bands='wild' to REPRODUCE Mertens-Ravn / Gertler-Karadi figures, never "
            f"to make an inferential claim."
        )
        lines.append(
            f"EXP6 n={n:<4} the strong-instrument wild arm recovers by h>=2 "
            f"({100 * wild['coverage'][1]:.1f}% at h=1, {100 * wild['coverage'][2]:.1f}% at h=2) "
            f"because the reduced-form slopes DO vary across wild draws -- only the "
            f"identification step is frozen. So the damage is concentrated at impact and h=1, "
            f"which is exactly where proxy-SVAR papers put their headline number."
        )
        efron = blk["arms"]["strong/moving_block/efron"]
        pair = blk["paired"]["strong"]
        lines.append(
            f"EXP6 n={n:<4} Hall vs Efron, measured not asserted, on identical draws: at h=0 "
            f"Hall {100 * mbb['coverage'][0]:.1f}% / Efron {100 * efron['coverage'][0]:.1f}%; "
            f"at h={hmax6} Hall {100 * mbb['coverage'][-1]:.1f}% / Efron "
            f"{100 * efron['coverage'][-1]:.1f}%. Paired at h={hmax6}, Efron covers where Hall "
            f"misses in {100 * pair['efron_only'][-1]:.1f}% of samples and the reverse in "
            f"{100 * pair['hall_only'][-1]:.1f}%. Same width either way -- they are the same two "
            f"quantiles, reflected -- so this is purely about where the band sits. The library "
            f"recommends Hall; on THIS DGP Efron holds up better at long horizons and the two "
            f"are within a few points at impact. Neither dominates; report which you used."
        )
        weak = blk["arms"]["weak/moving_block/hall"]
        lines.append(
            f"EXP6 n={n:<4} weak instrument (median first-stage F "
            f"{blk['median_first_stage_f']['weak']:.1f}): the moving-block band does NOT lose "
            f"coverage at impact -- it covers {100 * weak['coverage'][0]:.1f}% against "
            f"{100 * NOMINAL:.0f}% -- it goes USELESS instead, "
            f"{weak['mean_width'][0] / mbb['mean_width'][0]:.1f}x wider "
            f"({weak['mean_width'][0]:.2f} around a true response of {e6['truth'][0]:.2f}). "
            f"That is the Wald-type band degrading in the only way it can. It still loses "
            f"coverage at long horizons ({100 * weak['coverage'][-1]:.1f}% at h={hmax6}). For "
            f"weak instruments prefer proxy_ar_sets, whose shape is allowed to say 'unbounded'."
        )
        lines.append(
            f"EXP6 n={n:<4} across the three moving-block arms, "
            f"{sum(blk['n_failed'].values())} bootstrap draws failed out of "
            f"{len(blk['n_failed']) * e6['reps'] * e6['n_boot']:,}. The six failure guards never "
            f"fired on this DGP -- the normalisation variable carries the LARGEST impact loading "
            f"here, so the near-zero-denominator tail the guards exist for is never entered. "
            f"This harness therefore measures nothing about how those guards behave when it is, "
            f"and no coverage number above is conditioned on a discarded draw."
        )
    lines.append(
        f"EXP6 bands are POINTWISE. No simultaneous band is measured for proxy_svar_bands "
        f"and none is offered by the library; the joint shortfall documented for "
        f"var_irf_bands above applies here for the same reason."
    )
    return lines


# ==========================================================================
def run(quick=False, reps=None):
    reps = reps if reps is not None else (REPS_QUICK if quick else REPS_FULL)
    horizon = 8 if quick else 12
    started = time.time()

    print("=" * 104)
    print("var_irf_bands -- COVERAGE OF VAR IMPULSE-RESPONSE BANDS")
    print("=" * 104)
    print(f"master seed        : {SEED}   (every number below is a function of it)")
    print(f"replications       : {reps}   (MC se at p=0.90 is {100 * mc_se(0.90, reps):.2f}pp)")
    print(f"bootstrap draws    : {N_BOOT} percentile draws per replication")
    print(f"nominal level      : {100 * NOMINAL:.0f}%  (alpha = {ALPHA:.2f}, the library default)")
    print(f"horizons           : 0..{horizon}")
    print(f"mode               : {'QUICK smoke run' if quick else 'full run'}")
    print("DGPs               : " + "; ".join(d["name"] for d in (BASE, PERSIST, LAG4, PROXY)))
    print(f"functions measured : tsecon.var_irf_bands (exp 1-5); "
          f"tsecon.proxy_svar_bands (exp 6, n_boot = {PROXY_N_BOOT})")

    facts = structural_checks()
    facts["proxy"] = proxy_structural_checks()
    results = {}
    ns_small = (100, 200) if quick else (100, 200, 500)
    results["exp1"] = exp_horizon_profile(reps, horizon, ns=ns_small)
    report_horizon_profile(results["exp1"])
    results["exp2"] = exp_orth_cumulative(reps, horizon)
    report_orth_cumulative(
        results["exp2"], show=tuple(h for h in (0, 1, 2, 3, 4, 6, 8, 10, 12) if h <= horizon)
    )
    results["exp3"] = exp_bias_correct(reps, horizon, ns=(100,) if quick else (100, 200))
    report_bias_correct(
        results["exp3"], show=tuple(h for h in (0, 1, 2, 4, 6, 8, 10, 12) if h <= horizon)
    )
    results["exp4"] = exp_misspecified(reps, horizon, ns=(200,) if quick else (200, 500))
    report_misspecified(
        results["exp4"], show=tuple(h for h in (0, 1, 2, 3, 4, 5, 6, 8, 12) if h <= horizon)
    )
    results["exp5"] = exp_nominal_levels(reps, horizon)
    report_nominal_levels(
        results["exp5"], show=tuple(h for h in (0, 2, 4, 8, 12) if h <= horizon)
    )
    results["exp6"] = exp_proxy_svar(reps, horizon, ns=(240,) if quick else (240, 480))
    report_proxy_svar(
        results["exp6"], show=tuple(h for h in (0, 1, 2, 3, 4, 6, 8, 12) if h <= horizon)
    )

    header("FINDINGS -- measured, not targeted")
    for line in findings(results, reps):
        print("  " + line)

    header("ASSERTIONS")
    checks = assertions(results, facts, reps)
    failed = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n         {detail}")

    elapsed = time.time() - started
    header(f"{len(checks) - len(failed)}/{len(checks)} assertions passed in {elapsed:.1f}s")
    if failed:
        raise AssertionError(
            "coverage assertions failed: " + "; ".join(f"{c[0]} ({c[2]})" for c in failed)
        )
    return {"results": results, "facts": facts, "checks": checks, "elapsed": elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--reps", type=int, default=None, help="override replication count")
    args = parser.parse_args()
    run(quick=args.quick, reps=args.reps)
