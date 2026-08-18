"""Interval COVERAGE for quantile, panel and cumulative local projections.

    .venv/bin/python docs/examples/coverage/quantile_panel_lp.py [--quick]

This module extends the local-projection coverage audit (`lp_family.py`) to
the three LP surfaces it did not measure: ``quantile_lp``, ``panel_lp`` (with
its Driscoll-Kraay default and the split-panel-jackknife correction), and
``lp(cumulative=...)`` -- the mode whose standard error was the most serious
defect of audit rounds 2-5, fixed in 0.3.0 by making the default SE
mode-dependent. Every number below is measured on a DGP whose truth is known
in closed form, with the Monte Carlo standard error of the measurement next
to it.

--------------------------------------------------------------------------
The known-truth DGPs, and the closed forms
--------------------------------------------------------------------------
QUANTILE LP (experiments 1-2). A location-scale MA:

    y_t = sum_{j=0}^{J-1} theta_j s_{t-j} + (1 + c*s_t) e_t,
    theta_j = 0.7**j, J = 5, c = 0.4, e_t iid N(0,1)

with s_t i.i.d. U(-1.5, 1.5) so the scale 1 + c*s_t is strictly positive.
The truncation J = p + 1 (p = 4 lag controls) is what makes the truth EXACT:
every s-lag that appears in y_{t+h} beyond s_t is inside the control set, so
the conditional tau-quantile of y_{t+h} given the design is exactly linear in
the design, and the population quantile-LP slope on s_t is

    h = 0 :  theta_0 + c * z_tau     (z_tau = Phi^{-1}(tau); the shock moves
                                      the scale, so the fan opens AT IMPACT)
    h >= 1:  theta_h                 (the scale channel is contemporaneous;
                                      future noise shifts the intercept only)

verified against a T = 200,000 fit to ~0.005 before any coverage was
counted. The PERSISTENT arm replaces s by a Gaussian AR(1) with parameter
phi_s = 0.8 (unit marginal variance) and drops the scale channel; by the
Markov property the population slope is

    slope_h = theta_h + sum_{j<h} theta_j * phi_s**(h-j)      (p >= 1)
    slope_h = sum_j theta_j * phi_s**|h-j|                    (p = 0)

also verified at T = 150,000. Why the persistent arm exists: the quantile
model card warns that these standard errors are "the Powell kernel sandwich
... heteroskedasticity-robust but NOT HAC", and predicts the growth_at_risk
card's overlap under-coverage "is the right order of magnitude to expect
here too". Whether that transfer is right is exactly what experiments 1-2
measure -- and the answer turns out to hinge on the lag controls, not on the
overlap. See the closing notes.

PANEL LP (experiments 3-4). A dynamic panel with a common shock:

    y_{i,t} = alpha_i + phi y_{i,t-1} + beta s_t + gamma f_t + e_{i,t}

with s_t (the observed common shock), f_t (an UNobserved common factor) and
e_{i,t} all i.i.d. N(0,1), alpha_i ~ N(0,1), phi = beta = 0.8, started at
stationarity via a 25-period burn-in. The horizon-h truth is beta * phi**h
exactly: s_t is i.i.d. and therefore orthogonal in population to every
control (y_{i,t-1}, s_{t-1}, the fixed effects). gamma = 0.9 in the
Driscoll-Kraay grid -- cross-sectional dependence beyond the shock is what
Driscoll-Kraay exists for -- and gamma = 0 in the SPJ experiment, which
reproduces the panel model card's own Monte Carlo design (N=50,
bandwidth=2, one lag of each control) at ~7x its replication count.

CUMULATIVE LP (experiment 5). The `lp_family.py` house DGP
(theta_j = 0.7**j truncated at J = 25, i.i.d. shock, independent nuisance).
Under cumulative="both" the regressor is sum_{j<=h} s_{t+j} and the
regressand sum_{k<=h} y_{t+k}; with an i.i.d. shock orthogonal to every
control the population coefficient is

    m_h = sum_{d=0}^{h} (h+1-d) theta_d / (h+1)

and under cumulative=True (cumulated outcome, contemporaneous impulse) it is
sum_{k<=h} theta_k. Both verified at T = 400,000 to ~0.002. This experiment
publishes the official post-fix number for the 0.3.0 repair: the audit
measured nominal 95% covering 0.507 at h=12 under the old HC1-on-overlap
default, and the fix makes `se=None` resolve to "hac" for this mode and
makes an explicit se="lag_augmented" RAISE. Both behaviours are asserted
here, not just described.

--------------------------------------------------------------------------
How to read the tables
--------------------------------------------------------------------------
Identical to lp_family.py: truth, bias, sd_est (Monte Carlo sd of the
estimate), mean/median reported SE, se/sd (the single most diagnostic
column: < 1 means the library understates its own sampling variability),
|b|/sd (off-centring), cov95 and its Monte Carlo standard error. Replications
where the library raised or returned non-finite output are dropped and
counted, never treated as covering.
"""
import argparse
import time

import numpy as np
from scipy.stats import norm

import tsecon

# --------------------------------------------------------------------------
# global configuration -- one seed for the whole suite, printed at the top
# --------------------------------------------------------------------------
SEED = 20260729
Z95 = float(norm.ppf(0.975))
NOMINAL = 0.95

# quantile-LP DGP constants
JQ = 5                       # = n_lag_controls + 1, which is what makes the
PQ = 4                       # conditional quantile EXACTLY linear in the design
THQ = 0.7 ** np.arange(JQ)
CSCALE = 0.4                 # scale channel: sd(e | s_t) = 1 + CSCALE * s_t
TAUS = (0.25, 0.5, 0.75)
HQ = 6
PHI_S = 0.8                  # AR(1) parameter of the persistent-regressor arm

# panel DGP constants
PHI_P, BETA_P, GAMMA_F = 0.8, 0.8, 0.9
BURN_P = 25

# cumulative-LP DGP constants (the lp_family house DGP)
JL = 25
THL = 0.7 ** np.arange(JL)
HL = 12


def _rng(experiment, rep):
    """A reproducible, independent stream per (experiment, replication)."""
    return np.random.default_rng([SEED, experiment, rep])


# --------------------------------------------------------------------------
# data-generating processes
# --------------------------------------------------------------------------
def dgp_quantile_iid(rng, T):
    """Location-scale MA with an i.i.d. uniform shock; truth in the docstring."""
    s = rng.uniform(-1.5, 1.5, T + JQ)
    e = rng.standard_normal(T + JQ)
    y = np.convolve(s, THQ)[:T + JQ] + (1.0 + CSCALE * s) * e
    return y[JQ:], s[JQ:]


def dgp_quantile_persistent(rng, T):
    """Pure-location MA driven by a Gaussian AR(1) regressor (phi_s = 0.8)."""
    n = T + JQ + 50                       # 50 extra periods of AR burn-in
    s = np.zeros(n)
    isd = np.sqrt(1.0 - PHI_S ** 2)       # unit marginal variance
    for t in range(1, n):
        s[t] = PHI_S * s[t - 1] + isd * rng.standard_normal()
    y = np.convolve(s, THQ)[:n] + rng.standard_normal(n)
    return y[JQ + 50:], s[JQ + 50:]


def truth_quantile_iid(tau, horizons):
    th = np.concatenate([THQ, np.zeros(max(0, horizons + 1 - JQ))])[:horizons + 1]
    tr = th.copy()
    tr[0] += CSCALE * float(norm.ppf(tau))
    return tr


def truth_quantile_persistent(horizons, p):
    th = np.concatenate([THQ, np.zeros(max(0, horizons + 1 - JQ))])
    if p >= 1:
        return np.array([th[h] + sum(th[j] * PHI_S ** (h - j) for j in range(h))
                         for h in range(horizons + 1)])
    return np.array([sum(th[j] * PHI_S ** abs(h - j) for j in range(JQ))
                     for h in range(horizons + 1)])


def dgp_panel(rng, N, T, gamma_f):
    """Dynamic panel with a common shock, started at stationarity."""
    total = T + BURN_P
    s = rng.standard_normal(total)
    f = rng.standard_normal(total)
    alpha = rng.normal(0.0, 1.0, N)
    y = np.zeros((N, total))
    y[:, 0] = alpha + BETA_P * s[0] + gamma_f * f[0] + rng.standard_normal(N)
    for t in range(1, total):
        y[:, t] = (alpha + PHI_P * y[:, t - 1] + BETA_P * s[t]
                   + gamma_f * f[t] + rng.standard_normal(N))
    return y[:, BURN_P:], s[BURN_P:]


def dgp_cumulative(rng, T, sig_eta=0.5):
    """The lp_family house DGP: y_t = sum_j THL_j s_{t-j} + eta_t."""
    s = rng.standard_normal(T + JL)
    y = np.convolve(s, THL)[JL:JL + T] + sig_eta * rng.standard_normal(T)
    return y, s[JL:JL + T]


def truth_cumulative_both(horizons):
    return np.array([sum((h + 1 - d) * THL[d] for d in range(h + 1)) / (h + 1)
                     for h in range(horizons + 1)])


def truth_cumulative_outcome(horizons):
    return np.cumsum(THL[:horizons + 1])


# --------------------------------------------------------------------------
# coverage bookkeeping (house schema, shared with lp_family.py)
# --------------------------------------------------------------------------
def summarize(est, se, truth, arm, extra=None):
    """Per-horizon coverage summary from (reps x H+1) estimate and SE arrays."""
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
        row = {
            "arm": arm,
            "h": h,
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
        }
        if extra is not None:
            row.update({k: float(v[h]) for k, v in extra.items()})
        rows.append(row)
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


def paired_diff(cov_a, cov_b, h_min=0):
    """Per-draw coverage difference a - b, pooled over horizons >= h_min.

    Both arrays are (reps x H+1) booleans measured on the SAME draws, so the
    pooled standard error comes from the per-draw average difference and the
    cross-horizon correlation is handled, exactly as in lp_family.py.
    """
    d = cov_a.astype(float) - cov_b.astype(float)
    dbar = d[:, h_min:].mean(axis=1)
    return {"h_min": h_min, "diff": float(dbar.mean()),
            "se_diff": float(dbar.std(ddof=1) / np.sqrt(dbar.size))}


# ==========================================================================
# Experiments 1-2 -- quantile_lp
# ==========================================================================
def exp_quantile_iid(reps, sizes=(200, 400), experiment=1):
    """quantile_lp on the canonical design: an identified i.i.d. shock.

    This is the use the model card describes ("an identified,
    exogenous-conditional-on-controls shock, exactly as for lp"), and it is
    the arm where the card's transferred growth_at_risk warning turns out NOT
    to apply: with an i.i.d. impulse the check-loss score s_t * psi_tau(u) is
    serially UNcorrelated at every horizon -- the overlap correlates the psi
    factors, but each is multiplied by an independent draw of s -- so the
    Powell sandwich's no-serial-correlation assumption holds by construction,
    for the same reason lag-augmented HC1 works for mean LP.
    """
    rows = []
    for i, T in enumerate(sizes):
        ests = {tau: np.full((reps, HQ + 1), np.nan) for tau in TAUS}
        ses = {tau: np.full((reps, HQ + 1), np.nan) for tau in TAUS}
        failed = 0
        for r in range(reps):
            y, s = dgp_quantile_iid(_rng(experiment * 10 + i, r), T)
            try:
                out = tsecon.quantile_lp(y, s, taus=list(TAUS), horizons=HQ,
                                         n_lag_controls=PQ)
            except Exception:                    # noqa: BLE001 - counted
                failed += 1
                continue
            irf, se = np.asarray(out["irf"]), np.asarray(out["se"])
            for k, tau in enumerate(TAUS):
                ests[tau][r] = irf[k]
                ses[tau][r] = se[k]
        for tau in TAUS:
            rows += summarize(ests[tau], ses[tau], truth_quantile_iid(tau, HQ),
                              f"iid T={T} tau={tau:.2f}")
        if failed:
            print(f"  note: quantile_lp raised on {failed}/{reps} draws "
                  f"(T={T})")
    return {
        "name": "quantile_lp: identified i.i.d. shock (the canonical design)",
        "meta": {"sizes": list(sizes), "horizons": HQ, "n_lag_controls": PQ,
                 "taus": list(TAUS), "reps": reps, "nominal": NOMINAL},
        "rows": rows,
    }


def exp_quantile_persistent(reps, T=200, experiment=2):
    """quantile_lp with a PERSISTENT regressor: where the card's warning lives.

    Two arms on the SAME draws:
      p=4 (the default lag controls)  the controls include the regressor's own
                                      lags, so the residualised impulse is the
                                      AR innovation -- nearly white -- and the
                                      Powell sandwich survives;
      p=0 (no controls)               nothing whitens the regressor, the score
                                      inherits its serial correlation, and the
                                      growth_at_risk-shaped under-coverage
                                      appears in full.

    The truths differ across arms (each is the population slope of ITS OWN
    projection; both closed forms are in the module docstring and were
    verified at T = 150,000), so the paired comparison below is about the
    INTERVALS, not about a shared estimand.
    """
    covered = {}
    rows = []
    for p in (PQ, 0):
        ests = {tau: np.full((reps, HQ + 1), np.nan) for tau in TAUS}
        ses = {tau: np.full((reps, HQ + 1), np.nan) for tau in TAUS}
        failed = 0
        truth = truth_quantile_persistent(HQ, p)
        for r in range(reps):
            y, s = dgp_quantile_persistent(_rng(experiment, r), T)
            try:
                out = tsecon.quantile_lp(y, s, taus=list(TAUS), horizons=HQ,
                                         n_lag_controls=p)
            except Exception:                    # noqa: BLE001 - counted
                failed += 1
                continue
            irf, se = np.asarray(out["irf"]), np.asarray(out["se"])
            for k, tau in enumerate(TAUS):
                ests[tau][r] = irf[k]
                ses[tau][r] = se[k]
        for tau in TAUS:
            rows += summarize(ests[tau], ses[tau], truth,
                              f"persistent p={p} tau={tau:.2f}")
        mid = 0.5
        covered[p] = np.abs(ests[mid] - truth) <= Z95 * ses[mid]
        if failed:
            print(f"  note: quantile_lp raised on {failed}/{reps} draws "
                  f"(p={p})")
    pooled = paired_diff(covered[PQ], covered[0], h_min=max(1, HQ // 2))
    return {
        "name": "quantile_lp: persistent regressor, with and without the "
                "whitening lag controls",
        "meta": {"T": T, "horizons": HQ, "phi_s": PHI_S, "taus": list(TAUS),
                 "reps": reps, "nominal": NOMINAL},
        "rows": rows,
        "paired_pooled_tau50": pooled,
    }


# ==========================================================================
# Experiments 3-4 -- panel_lp
# ==========================================================================
def exp_panel_dk(reps, grid=((10, 40), (50, 40), (10, 80), (50, 80)),
                 horizon=4, experiment=3):
    """panel_lp with its Driscoll-Kraay default, over the (N, T) grid.

    The cookbook's caveat is quantitative here: with a single common shock the
    effective sample is ~T, not N*T, so N ∈ {10, 50} should barely move the
    coverage while T ∈ {40, 80} moves it materially. A cluster-by-entity arm
    at the largest cell prices the cookbook's "wrong reflex" -- clustering
    assumes independence across entities exactly where a common factor makes
    them dependent -- as a paired coverage difference on the same draws.
    """
    truth = BETA_P * PHI_P ** np.arange(horizon + 1)
    rows = []
    covered = {}
    for i, (N, T) in enumerate(grid):
        est = np.full((reps, horizon + 1), np.nan)
        se = np.full((reps, horizon + 1), np.nan)
        for r in range(reps):
            y, s = dgp_panel(_rng(experiment * 10 + i, r), N, T, GAMMA_F)
            out = tsecon.panel_lp(y, s, horizon=horizon, n_lag_controls=1)
            est[r] = np.asarray(out["irf"])
            se[r] = np.asarray(out["se"])
        rows += summarize(est, se, truth, f"N={N} T={T} dk")
        covered[(N, T, "dk")] = np.abs(est - truth) <= Z95 * se
    # the cluster arm, on the SAME draws as the largest Driscoll-Kraay cell
    N, T = grid[-1]
    i = len(grid) - 1
    est = np.full((reps, horizon + 1), np.nan)
    se = np.full((reps, horizon + 1), np.nan)
    for r in range(reps):
        y, s = dgp_panel(_rng(experiment * 10 + i, r), N, T, GAMMA_F)
        out = tsecon.panel_lp(y, s, horizon=horizon, n_lag_controls=1,
                              se_type="cluster")
        est[r] = np.asarray(out["irf"])
        se[r] = np.asarray(out["se"])
    rows += summarize(est, se, truth, f"N={N} T={T} cluster")
    covered[(N, T, "cluster")] = np.abs(est - truth) <= Z95 * se
    pooled = paired_diff(covered[(N, T, "dk")], covered[(N, T, "cluster")])
    return {
        "name": "panel_lp: Driscoll-Kraay default over (N, T), common factor",
        "meta": {"grid": [list(c) for c in grid], "horizon": horizon,
                 "phi": PHI_P, "beta": BETA_P, "gamma_f": GAMMA_F,
                 "bandwidth": 4.0, "n_lag_controls": 1, "reps": reps,
                 "nominal": NOMINAL},
        "rows": rows,
        "paired_dk_minus_cluster": pooled,
    }


def exp_panel_spj(reps, sizes=(20, 40), N=50, horizon=2, experiment=4):
    """The split-panel jackknife, on the panel model card's own MC design.

    gamma_f = 0, N = 50, Driscoll-Kraay bandwidth = 2, one lag of each
    control: the exact design whose 300-replication table the card publishes
    (FE 0.74 -> SPJ 0.82 at T=20, h=2). Rerun here at ~7x the replications
    with this suite's seed policy, FE and SPJ paired on the same draws, so
    the published claim is corroborated (or not) with a standard error small
    enough to tell.
    """
    truth = BETA_P * PHI_P ** np.arange(horizon + 1)
    rows = []
    paired = {}
    bias = {}
    for i, T in enumerate(sizes):
        covs = {}
        for arm, kw in (("fe", {}), ("spj", {"bias_correction": "spj"})):
            est = np.full((reps, horizon + 1), np.nan)
            se = np.full((reps, horizon + 1), np.nan)
            for r in range(reps):
                y, s = dgp_panel(_rng(experiment * 10 + i, r), N, T, 0.0)
                out = tsecon.panel_lp(y, s, horizon=horizon, n_lag_controls=1,
                                      bandwidth=2.0, **kw)
                est[r] = np.asarray(out["irf"])
                se[r] = np.asarray(out["se"])
            rows += summarize(est, se, truth, f"T={T} {arm}")
            covs[arm] = np.abs(est - truth) <= Z95 * se
            bias[(T, arm)] = (est - truth).mean(axis=0)
        paired[T] = paired_diff(covs["spj"], covs["fe"], h_min=1)
    return {
        "name": "panel_lp: split-panel jackknife vs uncorrected FE "
                "(the model card's design)",
        "meta": {"sizes": list(sizes), "N": N, "horizon": horizon,
                 "bandwidth": 2.0, "n_lag_controls": 1, "reps": reps,
                 "card_claim": {"T": 20, "h": 2, "fe": 0.743, "spj": 0.823},
                 "nominal": NOMINAL},
        "rows": rows,
        "paired_spj_minus_fe": {str(k): v for k, v in paired.items()},
        "bias": {f"T={t} {a}": v.tolist() for (t, a), v in bias.items()},
    }


# ==========================================================================
# Experiment 5 -- lp(cumulative=...): the post-fix official numbers
# ==========================================================================
def exp_lp_cumulative(reps, sizes=(400, 1600), experiment=5):
    """lp(cumulative="both") under the 0.3.0 mode-dependent default.

    Two arms per T on the same draws: cumulative="both" (default se resolves
    to "hac" -- asserted, not assumed) and cumulative=True (cumulated
    outcome; default stays lag-augmented). The "both" numbers are the
    official post-fix measurement of the audit's most serious open finding:
    the pre-fix HC1 default covered 0.507 at h=12 on this DGP family. The
    refusal of se="lag_augmented" with cumulative="both" is asserted below.
    """
    rows = []
    for i, T in enumerate(sizes):
        ests = {a: np.full((reps, HL + 1), np.nan) for a in ("both", "outcome")}
        ses = {a: np.full((reps, HL + 1), np.nan) for a in ("both", "outcome")}
        se_methods = set()
        for r in range(reps):
            y, s = dgp_cumulative(_rng(experiment * 10 + i, r), T)
            ob = tsecon.lp(y, s, horizons=HL, n_lag_controls=4,
                           cumulative="both")
            oo = tsecon.lp(y, s, horizons=HL, n_lag_controls=4,
                           cumulative=True)
            se_methods.add((ob["se_method"], oo["se_method"]))
            ests["both"][r] = np.asarray(ob["irf"])
            ses["both"][r] = np.asarray(ob["se"])
            ests["outcome"][r] = np.asarray(oo["irf"])
            ses["outcome"][r] = np.asarray(oo["se"])
        rows += summarize(ests["both"], ses["both"], truth_cumulative_both(HL),
                          f"T={T} both (hac default)")
        rows += summarize(ests["outcome"], ses["outcome"],
                          truth_cumulative_outcome(HL),
                          f"T={T} outcome (lag-aug default)")
        assert se_methods == {("hac", "lag_augmented")}, se_methods
    # the refusal is part of the fix: lag augmentation cannot reach the
    # future-shock overlap, so asking for it with "both" must raise.
    y, s = dgp_cumulative(_rng(experiment, 0), 400)
    try:
        tsecon.lp(y, s, horizons=4, n_lag_controls=2, cumulative="both",
                  se="lag_augmented")
        refused = False
    except ValueError:
        refused = True
    return {
        "name": 'lp(cumulative="both"): the post-fix mode-dependent default',
        "meta": {"sizes": list(sizes), "horizons": HL, "n_lag_controls": 4,
                 "reps": reps, "nominal": NOMINAL,
                 "prefix_defect_cov_h12": 0.507},
        "rows": rows,
        "lag_augmented_refused": refused,
    }


# ==========================================================================
# assertions -- only things that are robustly true, stated as inequalities
# ==========================================================================
def _rows_by(res, arm=None, h=None):
    out = res["rows"]
    if arm is not None:
        out = [r for r in out if r["arm"] == arm]
    if h is not None:
        out = [r for r in out if r["h"] == h]
    return out


def check(results, quick):
    """Assert the robust qualitative facts; print each check and its numbers.

    Coverage LEVELS at stress cells are deliberately not pinned -- they are
    the measurements this module publishes. Floors are regression guards far
    from the measured values, not claims that the floor is good.
    """
    checks = []

    def ok(label, passed, detail):
        checks.append((label, bool(passed), detail))

    # ---- quantile_lp, iid arm ------------------------------------------
    q1 = results["quantile_iid"]
    big = q1["meta"]["sizes"][-1]
    lo0 = _rows_by(q1, f"iid T={big} tau=0.25", 0)[0]
    md0 = _rows_by(q1, f"iid T={big} tau=0.50", 0)[0]
    hi0 = _rows_by(q1, f"iid T={big} tau=0.75", 0)[0]
    # the location-scale design check: at impact the tau-slopes must fan out
    # around the median in the order the scale channel dictates.
    ok("quantile_lp design check: the impact fan opens (slope rises in tau)",
       (lo0["truth"] + lo0["bias"] < md0["truth"] + md0["bias"]
        < hi0["truth"] + hi0["bias"]),
       f"mean impact slopes {lo0['truth'] + lo0['bias']:.3f} < "
       f"{md0['truth'] + md0['bias']:.3f} < {hi0['truth'] + hi0['bias']:.3f}")
    ok(f"quantile_lp iid arm: median-tau impact is calibrated at T={big} "
       f"(within 4 points)",
       abs(md0["cov95"] - NOMINAL) <= 0.04,
       f"cov95={md0['cov95']:.3f} vs {NOMINAL:.2f}")
    worst_iid = min(q1["rows"], key=lambda r: r["cov95"])
    # 0.80 is a guard against a future regression producing the
    # growth_at_risk-style 0.6-0.7 collapse, far below the ~0.90+ measured;
    # the tail impact cell (tau=0.25, h=0) is the weakest and is a density-
    # estimation cost, not an overlap one.
    ok("quantile_lp iid arm: NO growth_at_risk-style collapse at any "
       "(tau, h, T) -- the card's transferred warning does not bind here",
       worst_iid["cov95"] >= 0.80,
       f"worst cell {worst_iid['arm']} h={worst_iid['h']}: "
       f"cov95={worst_iid['cov95']:.3f}")

    # ---- quantile_lp, persistent arm -----------------------------------
    q2 = results["quantile_persistent"]
    pool = q2["paired_pooled_tau50"]
    ok("quantile_lp persistent: the default lag controls rescue the sandwich "
       "(p=4 beats p=0, paired, long horizons)",
       pool["diff"] > 3.0 * pool["se_diff"],
       f"pooled paired gap {pool['diff']:+.4f}, "
       f"3*se_diff={3 * pool['se_diff']:.4f}")
    p0 = [r for r in q2["rows"] if r["arm"].startswith("persistent p=0")]
    worst_p0 = min(p0, key=lambda r: r["cov95"])
    ok("quantile_lp persistent, p=0: the overlap under-coverage is real "
       "(worst cell below 0.88)",
       worst_p0["cov95"] < 0.88,
       f"worst {worst_p0['arm']} h={worst_p0['h']}: "
       f"cov95={worst_p0['cov95']:.3f}, se/sd={worst_p0['se_over_sd']:.2f}")
    ok("quantile_lp persistent, p=0: the miss is mostly the SE "
       "(se/sd < 0.85 at the worst cell, off-centring secondary)",
       worst_p0["se_over_sd"] < 0.85 and worst_p0["absbias_over_sd"] < 0.5,
       f"se/sd={worst_p0['se_over_sd']:.2f}, "
       f"|b|/sd={worst_p0['absbias_over_sd']:.2f}")

    # ---- panel_lp, Driscoll-Kraay grid ---------------------------------
    p3 = results["panel_dk"]
    hmax = p3["meta"]["horizon"]

    def pooled_cov(arm):
        rows = _rows_by(p3, arm)
        return float(np.mean([r["cov95"] for r in rows]))

    ok("panel_lp DK: N is not the sample -- quintupling N moves pooled "
       "coverage by < 3 points at T=40",
       abs(pooled_cov("N=50 T=40 dk") - pooled_cov("N=10 T=40 dk")) < 0.03,
       f"pooled cov N=10: {pooled_cov('N=10 T=40 dk'):.3f}, "
       f"N=50: {pooled_cov('N=50 T=40 dk'):.3f}")
    ok("panel_lp DK: T is the sample -- doubling T raises pooled coverage "
       "at N=50",
       pooled_cov("N=50 T=80 dk") > pooled_cov("N=50 T=40 dk"),
       f"pooled cov T=40: {pooled_cov('N=50 T=40 dk'):.3f} -> "
       f"T=80: {pooled_cov('N=50 T=80 dk'):.3f}")
    ab40 = float(np.mean([abs(r["bias"]) for r in
                          _rows_by(p3, "N=50 T=40 dk") if r["h"] >= 1]))
    ab80 = float(np.mean([abs(r["bias"]) for r in
                          _rows_by(p3, "N=50 T=80 dk") if r["h"] >= 1]))
    b40 = _rows_by(p3, "N=50 T=40 dk", hmax)[0]["bias"]
    ok("panel_lp DK: Nickell bias is negative at long horizons and shrinks "
       "in T (pooled |bias| over h >= 1)",
       b40 < 0 and ab80 < ab40,
       f"bias at h={hmax}, T=40: {b40:+.3f}; pooled |bias| h>=1: "
       f"{ab40:.3f} (T=40) -> {ab80:.3f} (T=80)")
    pc = p3["paired_dk_minus_cluster"]
    ok("panel_lp: cluster-by-entity under a common factor covers WORSE than "
       "Driscoll-Kraay (paired, same draws)",
       pc["diff"] > 3.0 * pc["se_diff"],
       f"pooled paired gap dk-cluster {pc['diff']:+.4f}, "
       f"3*se_diff={3 * pc['se_diff']:.4f}")

    # ---- panel_lp, SPJ --------------------------------------------------
    p4 = results["panel_spj"]
    card = p4["meta"]["card_claim"]
    fe20 = _rows_by(p4, "T=20 fe", 2)[0]
    spj20 = _rows_by(p4, "T=20 spj", 2)[0]
    ok("panel_lp SPJ: the correction removes most of the T=20 bias "
       "(|bias| falls by > 1.5x at h=2)",
       abs(spj20["bias"]) < abs(fe20["bias"]) / 1.5,
       f"bias {fe20['bias']:+.3f} (fe) -> {spj20['bias']:+.3f} (spj)")
    ok("panel_lp FE: the uncorrected T=20 h=2 interval is invalid "
       "(coverage < 0.85, as the card and crate test state)",
       fe20["cov95"] < 0.85,
       f"cov95={fe20['cov95']:.3f}")
    ok("panel_lp SPJ: T=20 h=2 coverage inside the honest [0.72, 0.95] band "
       "-- better than FE's bias-broken interval, still short of nominal",
       0.72 <= spj20["cov95"] <= 0.95,
       f"cov95={spj20['cov95']:.3f} "
       f"(card published fe {card['fe']:.3f} -> spj {card['spj']:.3f} "
       f"at 300 reps)")
    ok("panel_lp SPJ: neither route reaches nominal 95% at T=20",
       fe20["cov95"] < 0.92 and spj20["cov95"] < 0.92,
       f"fe {fe20['cov95']:.3f}, spj {spj20['cov95']:.3f}")

    # ---- lp(cumulative=...) --------------------------------------------
    c5 = results["lp_cumulative"]
    ok('lp(cumulative="both", se="lag_augmented") raises rather than '
       "answering wrongly (the 0.3.0 guard)",
       c5["lag_augmented_refused"], "ValueError raised")
    sizes = c5["meta"]["sizes"]
    both_small = _rows_by(c5, f"T={sizes[0]} both (hac default)", HL)[0]
    defect = c5["meta"]["prefix_defect_cov_h12"]
    ok(f'lp(cumulative="both") post-fix h=12 coverage at T={sizes[0]} is '
       f"far above the pre-fix defect level ({defect})",
       both_small["cov95"] >= 0.85,
       f"cov95={both_small['cov95']:.3f} vs pre-fix {defect}")
    out_rows = [r for r in c5["rows"] if "outcome" in r["arm"]]
    ok("lp(cumulative=True) keeps its lag-augmented default sound "
       "(every horizon >= 0.85)",
       min(r["cov95"] for r in out_rows) >= 0.85,
       f"worst cov95={min(r['cov95'] for r in out_rows):.3f}")

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
1. quantile_lp on its CANONICAL design -- an identified i.i.d. shock with the
   default lag controls -- is measured close to nominal at every tau, horizon
   and sample size tried, including the location-scale impact cell where the
   fan opens. The model card's warning that growth_at_risk's overlap
   under-coverage "is the right order of magnitude to expect here too" does
   NOT bind on this design, and the reason is structural, not luck: an
   i.i.d. impulse makes the check-loss score serially uncorrelated (the
   overlap correlates the psi factors, but each is multiplied by an
   independent shock draw), which is the same mechanism that makes
   lag-augmented HC1 work for mean LP. SOUND, with the small residual gap
   (~1-2 points, worst in the tails) being the usual Powell kernel density
   estimate at a few hundred observations.

2. quantile_lp with a PERSISTENT regressor is where the card's warning
   lives, and the lag controls are the hinge. With the default p=4 the
   controls include the regressor's own lags, the residualised impulse is
   the AR innovation, and coverage stays within a few points of nominal.
   Strip the controls (p=0) and nothing whitens the score: coverage decays
   with the horizon exactly as the growth_at_risk table predicts, with
   se/sd falling in step -- the SE, not the centre. THE ESTIMATOR: the
   Powell sandwich is heteroskedasticity-robust, not HAC, precisely as the
   card says; what the card understates is that the DEFAULT lag controls
   already buy back most of the loss. Keep them.

3. panel_lp's Driscoll-Kraay default under-covers at T=40 by ~8-15 points
   depending on the horizon, and the (N, T) grid shows WHY in a way no
   single cell can: quintupling N moves nothing, doubling T moves
   everything. With a common shock the effective sample is T, and
   Driscoll-Kraay is a T-asymptotic estimator -- the cookbook's caveat,
   measured. THE APPROXIMATION, plus a visible Nickell bias component at
   longer horizons (negative, shrinking in T, untouched by N).

4. cluster-by-entity on the same draws is strictly worse wherever the
   common factor matters -- the cookbook's "wrong reflex" priced as a paired
   coverage difference. THE ESTIMATOR (wrong covariance for this dependence
   structure), fixable by the caller: use the default.

5. panel_lp's SPJ correction does what its card claims at the card's own
   design: it removes most of the Nickell bias at T=20 and recovers part --
   not all -- of the coverage gap, with both routes still short of nominal
   at T=20 because Driscoll-Kraay itself is a short-T approximation. The
   card's 300-replication table is corroborated within Monte Carlo error at
   ~7x the replications, with one nuance worth stating: at these seeds the
   SPJ coverage gain at T=20 is smaller than the card's point numbers
   suggest (the bias reduction is unambiguous; the coverage difference at
   h=1 is within noise). Read the card's table with its mcse (~0.025) in
   mind.

6. lp(cumulative="both") post-fix is the official repair number for the
   audit's most serious finding: the pre-fix HC1 default covered 0.507 at
   h=12 and quadrupling T did not help; the 0.3.0 mode-dependent HAC
   default restores h=12 to ~0.93-0.95 at T=400 and drifts mildly
   CONSERVATIVE at T=1600 (the Bartlett bandwidth h + p is generous when
   T is large relative to the overlap). The refusal of se="lag_augmented"
   with cumulative="both" is asserted. The cumulated-OUTCOME mode keeps its
   lag-augmented default and stays sound, exactly as the LP card states.
"""


def run(quick=False):
    reps_full = {
        "quantile_iid": 1000,
        "quantile_persistent": 1000,
        "panel_dk": 2500,
        "panel_spj": 2500,
        "lp_cumulative": 2000,
    }
    scale = 8 if quick else 1
    reps = {k: max(100, v // scale) for k, v in reps_full.items()}

    t0 = time.perf_counter()
    print("=" * 100)
    print("tsecon interval COVERAGE: quantile, panel and cumulative local "
          "projections")
    print("=" * 100)
    print(f"seed                = {SEED}   (every draw is default_rng("
          f"[{SEED}, experiment, replication]))")
    print(f"nominal level       = {NOMINAL:.0%} two-sided, z = {Z95:.6f}")
    print(f"mode                = {'QUICK SMOKE RUN' if quick else 'full'}")
    print("replications        = " + ", ".join(f"{k}:{v}" for k, v in
                                               reps.items()))

    results = {}

    header("EXPERIMENT 1 -- quantile_lp: identified i.i.d. shock "
           "(location-scale, exact truths)")
    print(f"y_t = sum theta_j s_(t-j) + (1 + {CSCALE}*s_t) e_t, "
          f"theta_j = 0.7**j truncated at J = {JQ} = p + 1;")
    print(f"impact truth per tau: theta_0 + {CSCALE}*z_tau; h >= 1 truth: "
          f"theta_h at every tau\n")
    results["quantile_iid"] = exp_quantile_iid(reps["quantile_iid"])
    print_table(results["quantile_iid"]["rows"])

    header("EXPERIMENT 2 -- quantile_lp: persistent regressor "
           "(phi_s = 0.8), p = 4 vs p = 0")
    results["quantile_persistent"] = exp_quantile_persistent(
        reps["quantile_persistent"])
    print_table(results["quantile_persistent"]["rows"])
    pool = results["quantile_persistent"]["paired_pooled_tau50"]
    print(f"\npaired coverage difference at tau=0.50, p=4 minus p=0, pooled "
          f"over h >= {pool['h_min']}: {pool['diff']:+.4f} "
          f"(se {pool['se_diff']:.4f})")

    header("EXPERIMENT 3 -- panel_lp: the Driscoll-Kraay default over "
           "(N, T), with a common factor")
    results["panel_dk"] = exp_panel_dk(reps["panel_dk"])
    m = results["panel_dk"]["meta"]
    print(f"y_it = a_i + {m['phi']}*y_i,t-1 + {m['beta']}*s_t + "
          f"{m['gamma_f']}*f_t + e_it; truth beta*phi**h; "
          f"reps = {m['reps']}\n")
    print_table(results["panel_dk"]["rows"])
    pc = results["panel_dk"]["paired_dk_minus_cluster"]
    print(f"\npaired coverage difference, driscoll_kraay minus cluster "
          f"(same draws, N=50 T=80): {pc['diff']:+.4f} "
          f"(se {pc['se_diff']:.4f})")

    header("EXPERIMENT 4 -- panel_lp: split-panel jackknife vs FE at short "
           "T (the model card's design)")
    results["panel_spj"] = exp_panel_spj(reps["panel_spj"])
    m = results["panel_spj"]["meta"]
    print(f"gamma_f = 0, N = {m['N']}, bandwidth = {m['bandwidth']}, "
          f"reps = {m['reps']} (the card's table used 300); card claims "
          f"fe {m['card_claim']['fe']} -> spj {m['card_claim']['spj']} at "
          f"T=20 h=2\n")
    print_table(results["panel_spj"]["rows"])
    for T, d in results["panel_spj"]["paired_spj_minus_fe"].items():
        print(f"paired coverage difference spj minus fe at T={T}, pooled "
              f"h >= {d['h_min']}: {d['diff']:+.4f} (se {d['se_diff']:.4f})")

    header('EXPERIMENT 5 -- lp(cumulative="both"): the post-fix '
           "mode-dependent default, official numbers")
    results["lp_cumulative"] = exp_lp_cumulative(reps["lp_cumulative"])
    m = results["lp_cumulative"]["meta"]
    print(f"house DGP (theta = 0.7**h, J = {JL}); pre-fix defect: nominal "
          f"95% covered {m['prefix_defect_cov_h12']} at h=12; "
          f"reps = {m['reps']}\n")
    print_table(results["lp_cumulative"]["rows"])

    print(NOTES)
    results["_checks"] = check(results, quick)
    elapsed = time.perf_counter() - t0
    print()
    print(f"runtime: {elapsed:.1f} s")
    results["_runtime_s"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interval coverage for quantile, panel and cumulative "
                    "local projections")
    parser.add_argument("--quick", action="store_true",
                        help="cut every replication count by 8 for a smoke "
                             "run")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
