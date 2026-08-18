"""Interval COVERAGE for factor models and mixed-frequency regression.

    .venv/bin/python docs/examples/coverage/factor_midas.py [--quick]

Four surfaces from the nowcasting / factor side of the library, split by an
honest question asked first: DOES THIS FUNCTION SHIP AN INTERVAL AT ALL?

  favar           ships point IRFs only -- but the multivariate guide tells
                  the reader exactly how bands get built in practice (a VAR
                  on [factors, policy]), and warns that "bands that condition
                  on F-hat as if it were data are too narrow". Experiment 1
                  measures that warned construction: the generated-regressor
                  hazard, with a NUMBER on it.
  umidas          ships HAC standard errors (`bse`). Experiment 2 measures
                  their coverage on a known-weights mixed-frequency DGP.
  weighted_midas  ships NO interval-like output (point fit + weights only).
  dfm_nowcast     ships NO interval-like output (nowcast + factor path).
  nelson_siegel   ships NO interval-like output (factors + residuals + R^2).

The last three are recorded as measured facts, not assumptions: experiment 3
calls each function and asserts its returned key set carries nothing
interval-like, so if a later release adds a standard error the assertion
fails loudly and this family gains a measured row instead of a stale one.

--------------------------------------------------------------------------
The FAVAR DGP, and why its truth is closed-form
--------------------------------------------------------------------------
The guide's own transmission DGP (docs/guide/07-multivariate.md):

    F_t = Phi F_{t-1} + Psi R_{t-1} + eta_t          (2 latent factors)
    R_t = rho R_{t-1} + theta' F_{t-1} + sig_R nu_t  (policy rate)
    X_t = Lambda F_t + sig_x eps_t                   (N-series panel)

with Phi = [[.6,.05],[0,.5]], Psi = (-.5,-.3), rho = .7, theta = (.15,.10),
innovation sds (1, .8, .5), Lambda_ij ~ N(0,1) redrawn each replication. The
state (F1, F2, R) is a VAR(1) with companion A and DIAGONAL innovation
covariance, so under the recursive ordering (factors first, R last) the
population response of R to its own orthogonalised shock is exactly

    truth_h = [A**h]_{RR-column} * sig_R = (A**h diag(1,.8,.5))[2,2]

Crucially this cell is INVARIANT to any invertible transformation of the
factor block: the recursive shock to the last variable depends only on the
span of the variables ordered before it, and R itself is observed. So the
truth does not care that PCA recovers the factors only up to rotation, sign
and scale -- which is what makes this the one FAVAR band cell whose coverage
is well-defined without picking a normalisation. (favar's own `irf_policy`
and the var_irf_bands point path for this cell are asserted identical on the
same fit.)

Each replication fits `tsecon.favar` (plain two-step; the panel excludes R
and responds to it only with a lag, so no slow/fast split is needed), then
asks `var_irf_bands` (delta method, nominal 90%) for bands on
[F-hat, R] -- the construction the guide describes -- and, ON THE SAME DRAW,
for bands on the infeasible [F_true, R]. The paired difference is the
generated-regressor cost, separated from the delta-method horizon decay
that `irf_bands.py` already measures.

--------------------------------------------------------------------------
The MIDAS DGP
--------------------------------------------------------------------------
    hf_tau  = phi_h hf_{tau-1} + innov            (high-frequency AR(1))
    y_t     = 0.5 + 1.5 * sum_k w_k hf[t,k] + u_t (low frequency, K=12 lags,
                                                   m=3 HF periods per LF)

with w an exp-Almon shape normalised to sum to one. U-MIDAS regresses y on
all K lags plus a constant, so the population coefficient on lag k is
exactly beta * w_k and the intercept is exactly alpha -- no approximation.
The favourable arm has i.i.d. u and mild HF persistence; the stress arm has
AR(1) u (phi_u = .7), strong HF persistence (phi_h = .9, near-collinear lag
columns) and half the sample, which is HAC's home turf on paper and its
hard case in practice.
"""
import argparse
import re
import time

import numpy as np
from scipy.stats import norm

import tsecon

# --------------------------------------------------------------------------
# global configuration
# --------------------------------------------------------------------------
SEED = 20260729
Z95 = float(norm.ppf(0.975))
NOMINAL_FAVAR = 0.90          # var_irf_bands is measured at alpha = 0.10,
                              # matching irf_bands.py
NOMINAL = 0.95

# FAVAR DGP constants (the guide's transmission example)
PHI_F = np.array([[0.60, 0.05], [0.00, 0.50]])
PSI_F = np.array([-0.5, -0.3])
RHO_R, THETA_R = 0.7, np.array([0.15, 0.10])
SIG = np.array([1.0, 0.8, 0.5])
A_STATE = np.zeros((3, 3))
A_STATE[:2, :2] = PHI_F
A_STATE[:2, 2] = PSI_F
A_STATE[2, :2] = THETA_R
A_STATE[2, 2] = RHO_R
H_FAVAR = 8
BURN_F = 100
TRUTH_FAVAR = np.array([
    (np.linalg.matrix_power(A_STATE, h) @ np.diag(SIG))[2, 2]
    for h in range(H_FAVAR + 1)])

# MIDAS DGP constants
K_HF, M_RATIO = 12, 3
W_MIDAS = np.exp(0.2 * np.arange(K_HF) - 0.06 * np.arange(K_HF) ** 2)
W_MIDAS /= W_MIDAS.sum()
ALPHA_M, BETA_M, SIG_U = 0.5, 1.5, 0.4
TRUTH_MIDAS = np.concatenate([[ALPHA_M], BETA_M * W_MIDAS])

# matches se/bse/std/stderr/lower/upper/conf/ci/band/bound(s)/quantile(s)/
# interval as a whole underscore-delimited token, so `residuals`, `scheme`
# or a hypothetical `series` key cannot trip it by substring accident.
INTERVAL_KEY = re.compile(
    r"(?:^|_)(?:b?se|std(?:err)?|lower|upper|conf|ci|band|bounds?"
    r"|quantiles?|interval)(?:$|_)",
    re.IGNORECASE)


def _rng(experiment, rep):
    """A reproducible, independent stream per (experiment, replication)."""
    return np.random.default_rng([SEED, experiment, rep])


# --------------------------------------------------------------------------
# data-generating processes
# --------------------------------------------------------------------------
def dgp_favar(rng, T, N, idio_sd):
    """The guide's FAVAR transmission DGP, started past a burn-in."""
    eta = rng.standard_normal((T + BURN_F, 3)) * SIG
    X3 = np.zeros((T + BURN_F, 3))
    for t in range(1, T + BURN_F):
        X3[t] = A_STATE @ X3[t - 1] + eta[t]
    F = X3[BURN_F:, :2]
    R = X3[BURN_F:, 2]
    lam = rng.normal(size=(N, 2))
    Xp = F @ lam.T + idio_sd * rng.standard_normal((T, N))
    Xs = (Xp - Xp.mean(0)) / Xp.std(0)
    return Xs, R, F


def dgp_midas(rng, T_lf, phi_h, phi_u):
    """Known-weights mixed-frequency DGP; truth is TRUTH_MIDAS exactly."""
    n_hf = T_lf * M_RATIO + K_HF
    hf = np.zeros(n_hf)
    isd_h = np.sqrt(1.0 - phi_h ** 2)
    for t in range(1, n_hf):
        hf[t] = phi_h * hf[t - 1] + isd_h * rng.standard_normal()
    idx = ((np.arange(T_lf)[:, None] + 1) * M_RATIO - 1
           - np.arange(K_HF)[None, :] + K_HF)
    X = hf[idx]                       # nobs x K, most-recent-first
    u = np.zeros(T_lf)
    isd_u = np.sqrt(1.0 - phi_u ** 2) if phi_u else 1.0
    for t in range(1, T_lf):
        u[t] = phi_u * u[t - 1] + isd_u * rng.standard_normal()
    y = ALPHA_M + BETA_M * (X @ W_MIDAS) + SIG_U * u
    return y, X


# --------------------------------------------------------------------------
# coverage bookkeeping (house schema)
# --------------------------------------------------------------------------
def summarize(est, se, truth, arm, index_name="h"):
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    truth = np.asarray(truth, dtype=float)
    rows = []
    for h in range(est.shape[1]):
        ok = np.isfinite(est[:, h]) & np.isfinite(se[:, h]) & (se[:, h] > 0.0)
        e, s = est[ok, h], se[ok, h]
        n = int(ok.sum())
        if n < 2:
            continue
        covered = np.abs(e - truth[h]) <= Z95 * s
        p = float(covered.mean())
        sd = float(e.std(ddof=1))
        bias = float(e.mean() - truth[h])
        rows.append({
            "arm": arm,
            index_name: h,
            "truth": float(truth[h]),
            "bias": bias,
            "sd_est": sd,
            "mean_se": float(s.mean()),
            "med_se": float(np.median(s)),
            "se_over_sd": float(s.mean() / sd) if sd > 0 else float("nan"),
            "absbias_over_sd": abs(bias) / sd if sd > 0 else float("nan"),
            "cov95": p,
            "mcse": float(np.sqrt(p * (1.0 - p) / n)),
            "n_used": n,
        })
    return rows


SPEC = [
    ("arm", "arm", 26, "{:s}"),
    ("h", "h", 3, "{:d}"),
    ("truth", "truth", 8, "{:.4f}"),
    ("bias", "bias", 9, "{:+.4g}"),
    ("sd_est", "sd_est", 9, "{:.4g}"),
    ("mean_se", "mean_se", 9, "{:.4g}"),
    ("med_se", "med_se", 9, "{:.4g}"),
    ("se_over_sd", "se/sd", 6, "{:.2f}"),
    ("absbias_over_sd", "|b|/sd", 7, "{:.2f}"),
    ("cov95", "cov95", 7, "{:.3f}"),
    ("mcse", "mcse", 6, "{:.3f}"),
]


def print_table(rows, spec=SPEC):
    head = "  ".join(f"{hdr:>{w}}" for _, hdr, w, _ in spec)
    print(head)
    print("-" * len(head))
    last = None
    for r in rows:
        if last is not None and r["arm"] != last:
            print()
        last = r["arm"]
        print("  ".join(
            f"{(fmt.format(r[k]) if k in r else '.'):>{w}}"
            for k, _, w, fmt in spec))


def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ==========================================================================
# Experiment 1 -- FAVAR: bands that condition on F-hat, priced
# ==========================================================================
def exp_favar(reps, designs=((100, 0.5, 200), (20, 1.0, 200), (20, 1.0, 800)),
              experiment=1):
    """Coverage of the guide-described band construction, vs its oracle.

    Designs are (N, idio_sd, T): a rich clean panel where Bai-Ng asymptotics
    (sqrt(T)/N -> 0) have effectively arrived; a small noisy panel; and the
    same small panel with 4x the time series -- the cell that exposes the
    generated-regressor problem as INCONSISTENCY, because the band shrinks
    like 1/sqrt(T) around a factor-measurement bias that does not.
    """
    rows = []
    paired = {}
    point_identity_checked = False
    for i, (N, idio, T) in enumerate(designs):
        cov_f = np.full((reps, H_FAVAR + 1), np.nan)
        cov_o = np.full((reps, H_FAVAR + 1), np.nan)
        est_f = np.full((reps, H_FAVAR + 1), np.nan)
        se_f = np.full((reps, H_FAVAR + 1), np.nan)
        est_o = np.full((reps, H_FAVAR + 1), np.nan)
        se_o = np.full((reps, H_FAVAR + 1), np.nan)
        failed = 0
        for r in range(reps):
            rng = _rng(experiment * 10 + i, r)
            Xs, R, F = dgp_favar(rng, T, N, idio)
            try:
                fv = tsecon.favar(Xs, R, n_factors=2, lags=1,
                                  horizon=H_FAVAR)
                Fh = np.asarray(fv["factors"])
                b = tsecon.var_irf_bands(np.column_stack([Fh, R]), lags=1,
                                         horizon=H_FAVAR, orth=True,
                                         method="asymptotic", alpha=0.10)
                bo = tsecon.var_irf_bands(np.column_stack([F, R]), lags=1,
                                          horizon=H_FAVAR, orth=True,
                                          method="asymptotic", alpha=0.10)
            except Exception:                    # noqa: BLE001 - counted
                failed += 1
                continue
            if not point_identity_checked:
                # same-run invariant: the band's point path for this cell IS
                # favar's own irf_policy, bit-for-bit, so the coverage below
                # is a statement about the IRF favar reports.
                assert np.array_equal(
                    np.asarray(b["point"])[:, 2, 2],
                    np.asarray(fv["irf_policy"])), \
                    "var_irf_bands point path != favar irf_policy"
                point_identity_checked = True
            lo = np.asarray(b["lower"])[:, 2, 2]
            hi = np.asarray(b["upper"])[:, 2, 2]
            cov_f[r] = (lo <= TRUTH_FAVAR) & (TRUTH_FAVAR <= hi)
            est_f[r] = np.asarray(b["point"])[:, 2, 2]
            se_f[r] = np.asarray(b["se"])[:, 2, 2]
            lo = np.asarray(bo["lower"])[:, 2, 2]
            hi = np.asarray(bo["upper"])[:, 2, 2]
            cov_o[r] = (lo <= TRUTH_FAVAR) & (TRUTH_FAVAR <= hi)
            est_o[r] = np.asarray(bo["point"])[:, 2, 2]
            se_o[r] = np.asarray(bo["se"])[:, 2, 2]
        if failed:
            print(f"  note: favar/var_irf_bands raised on {failed}/{reps} "
                  f"draws (N={N}, T={T})")
        # nominal here is 0.90: containment was counted from the alpha=0.10
        # band edges, and summarize's cov95 column simply reports it.
        for est, se, cov, tag in ((est_f, se_f, cov_f, "F-hat"),
                                  (est_o, se_o, cov_o, "oracle F")):
            for row in summarize(est, se, TRUTH_FAVAR,
                                 f"N={N} T={T} {tag}"):
                row["cov95"] = float(np.nanmean(cov[:, row["h"]]))
                n = int(np.isfinite(cov[:, row["h"]]).sum())
                row["mcse"] = float(np.sqrt(
                    row["cov95"] * (1 - row["cov95"]) / n))
                rows.append(row)
        ok = np.isfinite(cov_f[:, 0]) & np.isfinite(cov_o[:, 0])
        d = cov_f[ok] - cov_o[ok]
        dbar = d[:, H_FAVAR // 2:].mean(axis=1)
        paired[f"N={N} T={T}"] = {
            "h_min": H_FAVAR // 2,
            "diff": float(dbar.mean()),
            "se_diff": float(dbar.std(ddof=1) / np.sqrt(dbar.size)),
        }
    return {
        "name": "favar: two-step bands conditioned on F-hat vs the oracle",
        "meta": {"designs": [list(d) for d in designs], "horizon": H_FAVAR,
                 "reps": reps, "nominal": NOMINAL_FAVAR,
                 "cell": "policy rate <- policy shock (rotation-invariant)"},
        "rows": rows,
        "paired_fhat_minus_oracle": paired,
    }


# ==========================================================================
# Experiment 2 -- U-MIDAS: HAC coefficient intervals
# ==========================================================================
def exp_umidas(reps, designs=((300, 0.5, 0.0, "fav"), (150, 0.9, 0.7,
                                                       "stress")),
               experiment=2):
    """Coverage of umidas's HAC `bse`, coefficient by coefficient.

    Rows are indexed by design column: k = 0 is the intercept, k = 1..12 the
    high-frequency lag coefficients (k = 1 the most recent lag). The truth is
    exact -- U-MIDAS is a correctly-specified OLS here.
    """
    rows = []
    for i, (T, phi_h, phi_u, tag) in enumerate(designs):
        est = np.full((reps, K_HF + 1), np.nan)
        se = np.full((reps, K_HF + 1), np.nan)
        for r in range(reps):
            y, X = dgp_midas(_rng(experiment * 10 + i, r), T, phi_h, phi_u)
            out = tsecon.umidas(y, X)
            est[r] = np.asarray(out["params"])
            se[r] = np.asarray(out["bse"])
        rows += summarize(est, se, TRUTH_MIDAS,
                          f"{tag} T={T} ph={phi_h} pu={phi_u}",
                          index_name="h")
    return {
        "name": "umidas: HAC coefficient intervals on a known-weights DGP",
        "meta": {"designs": [list(d) for d in designs], "K": K_HF,
                 "m": M_RATIO, "reps": reps, "nominal": NOMINAL,
                 "index": "k (0 = intercept, 1..12 = HF lag coefficients)"},
        "rows": rows,
    }


# ==========================================================================
# Experiment 3 -- the surfaces that ship no interval, verified not assumed
# ==========================================================================
def exp_no_interval(experiment=3):
    """Call each no-interval surface once and record its actual key set.

    The point of measuring "nothing to measure": if a later release adds a
    standard error or band to any of these, the interval-like-key assertion
    below fails, this experiment breaks loudly, and the family gains a
    measured row instead of silently keeping a stale "ships no interval"
    entry on the page.
    """
    rng = _rng(experiment, 0)
    calls = {}

    y, X = dgp_midas(rng, 120, 0.5, 0.0)
    calls["weighted_midas"] = tsecon.weighted_midas(y, X, scheme="exp_almon")

    T_dfm, N_dfm = 80, 8
    fac = np.zeros(T_dfm)
    for t in range(1, T_dfm):
        fac[t] = 0.7 * fac[t - 1] + rng.standard_normal()
    lam = rng.normal(size=N_dfm)
    panel = fac[:, None] * lam[None, :] + rng.standard_normal((T_dfm, N_dfm))
    calls["dfm_nowcast"] = tsecon.dfm_nowcast(panel, n_factors=1,
                                              factor_order=1)

    mats = np.array([3.0, 6.0, 12.0, 24.0, 36.0, 60.0, 120.0])
    yl = (5.0 + 0.5 * np.exp(-mats / 20.0)
          + 0.05 * rng.standard_normal(mats.size))
    calls["nelson_siegel"] = tsecon.nelson_siegel(mats, yl)

    out = {}
    for name, res in calls.items():
        keys = sorted(res.keys())
        interval_like = [k for k in keys if INTERVAL_KEY.search(k)]
        out[name] = {"keys": keys, "interval_like": interval_like}
    return {
        "name": "surfaces that ship no interval (key sets verified)",
        "surfaces": out,
    }


# ==========================================================================
# assertions
# ==========================================================================
def _rows_by(res, arm=None, h=None):
    out = res["rows"]
    if arm is not None:
        out = [r for r in out if r["arm"] == arm]
    if h is not None:
        out = [r for r in out if r["h"] == h]
    return out


def check(results, quick):
    checks = []

    def ok(label, passed, detail):
        checks.append((label, bool(passed), detail))

    # ---- FAVAR ----------------------------------------------------------
    e1 = results["favar"]
    rich = "N=100 T=200"
    small = "N=20 T=200"
    smallT = "N=20 T=800"
    d_rich = e1["paired_fhat_minus_oracle"][rich]
    d_long = e1["paired_fhat_minus_oracle"][smallT]
    ok("favar design check: on the rich clean panel the F-hat bands track "
       "the oracle (pooled paired gap within 2 points)",
       abs(d_rich["diff"]) < 0.02,
       f"pooled gap {d_rich['diff']:+.4f} (se {d_rich['se_diff']:.4f})")
    ok("favar: on the small noisy panel at T=800 the F-hat bands under-cover "
       "the oracle (paired, long horizons, > 3 se)",
       d_long["diff"] < -3.0 * d_long["se_diff"],
       f"pooled gap {d_long['diff']:+.4f} "
       f"(3*se = {3 * d_long['se_diff']:.4f})")
    h_long = H_FAVAR - 2                 # pooled over the last three horizons
    def _pool(arm):
        rows = [r for r in _rows_by(e1, arm) if r["h"] >= h_long]
        return float(np.mean([r["cov95"] for r in rows]))
    f_200 = _pool(f"{small} F-hat")
    f_800 = _pool(f"{smallT} F-hat")
    o_800 = _pool(f"{smallT} oracle F")
    ok("favar: the generated-regressor signature -- MORE data makes the "
       f"F-hat band WORSE at fixed N (pooled h >= {h_long}, T=200 -> 800), "
       "while the oracle improves",
       f_800 < f_200 - 0.02 and o_800 > f_800 + 0.05,
       f"F-hat pooled h>={h_long}: {f_200:.3f} (T=200) -> "
       f"{f_800:.3f} (T=800); oracle at T=800: {o_800:.3f}")

    # ---- U-MIDAS --------------------------------------------------------
    e2 = results["umidas"]
    fav = [r for r in e2["rows"] if r["arm"].startswith("fav") and r["h"] > 0]
    stress = [r for r in e2["rows"] if r["arm"].startswith("stress")]
    stress_slopes = [r for r in stress if r["h"] > 0]
    stress_const = [r for r in stress if r["h"] == 0][0]
    ok("umidas favourable: every HF-lag coefficient covers at >= 0.90",
       min(r["cov95"] for r in fav) >= 0.90,
       f"worst slope cov95={min(r['cov95'] for r in fav):.3f}")
    ok("umidas stress: the slopes hold up (worst >= 0.85)...",
       min(r["cov95"] for r in stress_slopes) >= 0.85,
       f"worst slope cov95={min(r['cov95'] for r in stress_slopes):.3f}")
    ok("umidas stress: ...and the INTERCEPT is where persistent errors "
       "bite (const covers at least 4 points below the worst slope)",
       stress_const["cov95"]
       < min(r["cov95"] for r in stress_slopes) - 0.04,
       f"const cov95={stress_const['cov95']:.3f} vs worst slope "
       f"{min(r['cov95'] for r in stress_slopes):.3f}")

    # ---- no-interval surfaces ------------------------------------------
    e3 = results["no_interval"]
    for name, info in e3["surfaces"].items():
        ok(f"{name} ships no interval-like key (so its page row is NONE, "
           f"verified)",
           not info["interval_like"],
           f"keys = {info['keys']}"
           + (f" -- INTERVAL-LIKE: {info['interval_like']}"
              if info["interval_like"] else ""))

    print()
    print("=" * 100)
    print("ASSERTIONS")
    print("=" * 100)
    width = max(len(lbl) for lbl, _, _ in checks)
    failures = []
    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<{width}}  {detail}")
        if not passed:
            failures.append(label)
    if quick:
        print()
        print("--quick: Monte Carlo standard errors are ~3x the default "
              "run's, so treat a near-miss as noise and re-run without "
              "--quick.")
    if failures:
        raise AssertionError("coverage assertions failed: "
                             + "; ".join(failures))
    return checks


# ==========================================================================
# driver
# ==========================================================================
NOTES = """
WHERE THE INTERVALS MISS -- the honest list, read with the tables above
----------------------------------------------------------------------
1. favar ships no band, and the band a careful reader will build -- the
   guide's own construction, var_irf_bands on [F-hat, policy] -- inherits
   var_irf_bands' delta-method horizon decay AND adds a generated-regressor
   cost on top. The grid separates the two: on a rich clean panel (N=100)
   the F-hat bands are statistically indistinguishable from the infeasible
   true-factor bands, i.e. Bai-Ng negligibility has arrived and the ONLY
   shortfall is the delta-method one already on the page. On a small noisy
   panel (N=20) the F-hat bands lose several extra points at long horizons,
   and the T-growth cell is the diagnosis: from T=200 to T=800 the ORACLE
   band improves while the F-hat band gets WORSE, because the interval
   shrinks like 1/sqrt(T) around a factor-measurement distortion that is
   O(1/N) and does not shrink in T at all. THE ESTIMATOR (a generated
   regressor is an errors-in-variables problem; no standard error that
   conditions on F-hat can see it) -- exactly the hazard the guide
   discloses, now with numbers. Bootstrap the two-step procedure, or get N
   up before T.

2. umidas's HAC intervals are close to nominal for the high-frequency lag
   COEFFICIENTS in both designs -- including the stress arm's persistent
   errors and near-collinear lag columns -- but the INTERCEPT under AR(1)
   errors under-covers materially, the same constant-under-persistence
   mechanism the page already documents for har_rv's constant: the
   intercept's score inherits the error's full serial correlation, and the
   Newey-West bandwidth that serves the slopes is too short for it. THE
   APPROXIMATION for the slopes; for the constant, quote it with care or
   lengthen maxlags.

3. weighted_midas, dfm_nowcast and nelson_siegel return NO interval-like
   output at all -- verified against their live key sets, not their docs.
   Any band you draw around a weighted-MIDAS slope, a nowcast, or a fitted
   yield curve is your own construction and its coverage is your claim, not
   the library's. (For weighted_midas the model card's key list is
   complete; note that a U-MIDAS regression of the same data DOES ship
   `bse` -- if you need an interval on a mixed-frequency slope, that is the
   supported route today.)
"""


def run(quick=False):
    reps_full = {"favar": 2000, "umidas": 3000}
    scale = 8 if quick else 1
    reps = {k: max(100, v // scale) for k, v in reps_full.items()}

    t0 = time.perf_counter()
    print("=" * 100)
    print("tsecon interval COVERAGE: factor models and mixed frequency")
    print("=" * 100)
    print(f"seed                = {SEED}   (every draw is default_rng("
          f"[{SEED}, experiment, replication]))")
    print(f"nominal levels      = favar bands {NOMINAL_FAVAR:.0%} "
          f"(alpha=0.10, as in irf_bands.py); umidas {NOMINAL:.0%}")
    print(f"mode                = {'QUICK SMOKE RUN' if quick else 'full'}")
    print("replications        = " + ", ".join(f"{k}:{v}" for k, v in
                                               reps.items()))

    results = {}

    header("EXPERIMENT 1 -- favar: the guide's band construction vs its "
           "oracle (nominal 0.90)")
    print("cell: the policy rate's response to the recursive policy shock --")
    print("the one FAVAR band cell that is invariant to factor rotation.\n")
    results["favar"] = exp_favar(reps["favar"])
    print_table(results["favar"]["rows"])
    print()
    for cell, d in results["favar"]["paired_fhat_minus_oracle"].items():
        print(f"paired coverage difference, F-hat minus oracle, {cell}, "
              f"pooled h >= {d['h_min']}: {d['diff']:+.4f} "
              f"(se {d['se_diff']:.4f})")

    header("EXPERIMENT 2 -- umidas: HAC coefficient intervals "
           "(nominal 0.95)")
    print("k = 0 is the intercept; k = 1..12 the HF lag coefficients.\n")
    results["umidas"] = exp_umidas(reps["umidas"])
    print_table(results["umidas"]["rows"])

    header("EXPERIMENT 3 -- the no-interval surfaces, key sets verified")
    results["no_interval"] = exp_no_interval()
    for name, info in results["no_interval"]["surfaces"].items():
        print(f"  {name:<16} keys: {', '.join(info['keys'])}")

    print(NOTES)
    results["_checks"] = check(results, quick)
    elapsed = time.perf_counter() - t0
    print()
    print(f"runtime: {elapsed:.1f} s")
    results["_runtime_s"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interval coverage for tsecon factor models and mixed "
                    "frequency")
    parser.add_argument("--quick", action="store_true",
                        help="cut every replication count by 8 for a smoke "
                             "run")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
