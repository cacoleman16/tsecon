"""Do the library's Bayesian bands and set-identified bounds mean what a reader thinks?

    .venv/bin/python docs/examples/coverage/bayes_and_sets.py           # full run, ~2 min
    .venv/bin/python docs/examples/coverage/bayes_and_sets.py --quick   # smoke run, ~12 s

This module measures *frequentist* coverage for three families of objects that
do not all make the same promise. Getting the question right matters more here
than getting a number, so read this header before reading a table.

Three different questions
-------------------------
1. `bvar_irf_draws`, `bvar_ssvs` -- BAYESIAN CREDIBLE BANDS.
   A 90% credible band is a statement about the posterior, not about repeated
   samples. It makes NO frequentist coverage promise. Measuring frequentist
   coverage anyway is still informative -- it tells you how much the prior is
   doing -- but a shortfall is not a bug, and a reader who expects 90% has
   misread the object. We measure it as a DIAGNOSTIC and label it as such.
   The measurement is essentially a measurement of the Minnesota prior: how far
   `delta` (the prior mean on the own first lag) sits from the truth, and how
   hard `lambda1` pushes toward it.

2. `sign_restricted_svar`, `zero_sign_svar`, `narrative_svar`,
   `robust_svar_bounds` -- SET IDENTIFICATION.
   Sign restrictions do not identify a point; they identify a SET. The
   meaningful question is whether the identified set contains the true
   structural IRF, not whether a pointwise quantile band does. Three objects
   in the returned dict answer three different questions and are reported
   separately:
     * `quantiles[...,0]/[...,4]` -- the pointwise 5th/95th percentile band
       across accepted rotations. This is a credible band under the Haar
       rotation prior. It is NOT a confidence interval and NOT the identified
       set: it is a posterior summary that mixes mutually inconsistent
       structural models (which is exactly what `fry_pagan_svar` exists to
       complain about).
     * `set_min`/`set_max` -- the envelope of accepted draws, i.e. the union
       over the reduced-form posterior of the (numerically explored)
       identified set. Wider than any credible object; expect near-100%
       coverage that certifies very little.
     * `robust_ci_lower`/`robust_ci_upper` from `robust_svar_bounds` -- the
       Giacomini-Kitagawa prior-robust credible region. This one DOES aim at a
       coverage-style guarantee: it is built to contain every point of the
       identified set with posterior probability `1 - alpha`, so measuring
       whether it contains the truth at rate `1 - alpha` is a fair test.

3. `bai_perron` break-date confidence intervals -- a genuine FREQUENTIST
   interval for a discrete parameter (Bai 1997, homogeneous-moments case).
   Nominal 90%/95%. This is the only object here where "does it cover at the
   nominal rate?" is the question the method itself claims to answer, so it is
   the only place a shortfall is unambiguously a shortfall.

Every number below is a measurement. Where coverage falls short it is printed
as-is with its Monte Carlo standard error next to it, so a reader can tell
0.93 from 0.90 honestly. `se = sqrt(p(1-p)/reps)`. Nothing was tuned to make
an assertion pass; the assertions at the bottom are deliberately the robust
qualitative facts (a large gap between two designs, a monotone degradation),
never a number that happened to land.

Known-truth DGPs
----------------
`PERSIST` (experiments 1, 1b)  stationary VAR(1), own lags 0.85, largest root
    0.905, `Sigma = [[1,.3],[.3,1]]`. The population orthogonalised IRF is
    exactly `A1^h chol(Sigma)`. `T = 100` on purpose: the prior has to matter.
`WN` (experiment 1b)  white noise, `A1 = 0`. Fitted as a VAR(1) with
    `delta = 0`, so the Minnesota prior mean IS the truth. Any coverage gap
    left over cannot be prior mis-centring -- it isolates the posterior's
    scale convention for `Sigma`.
`SVAR` (experiments 2, 2b)  stationary VAR(1) with a known non-recursive
    structural impact matrix `A0 = [[1,-0.5],[0.6,0.9]]`. The true structural
    IRF `A1^h A0` satisfies the imposed sign pattern (shock 0 raises both
    variables on impact; shock 1 lowers y0 and raises y1), so the truth is a
    member of the identified set by construction -- which is what makes a set
    coverage check meaningful rather than a test of whether the restrictions
    were true.
`RECURSIVE` (experiment 2c)  same `A1`, `A0 = [[1,0],[0.6,0.9]]`. Truth is
    exactly recursive, so the zero restriction imposed by `zero_sign_svar` is
    TRUE and the object becomes point-identified -- the one set-identification
    function here whose band is a band about a point.
`BREAK` (experiments 3, 3b)  `y_t = mu + delta*1{t >= tau} + e_t`, `x` a
    constant, so the Bai (1997) homogeneous case holds exactly (this is the
    only case the model card claims).

Reading the BVAR tables: `cov` is coverage, `bias` the median of
(posterior-median IRF - truth) across replications, `width` the mean band
width. `|bias|` comparable to `width` means the band is in the wrong place
(prior mis-centring); small `|bias|` with small `width` means the posterior is
simply tighter than the sampling distribution.
"""

from __future__ import annotations

import argparse
import textwrap
import time

import numpy as np
from scipy import stats

import tsecon

# --------------------------------------------------------------------------
# Master seed. Every random number in this module descends from it, and every
# tsecon call that takes a `seed` gets a deterministic function of the
# replication index. Same seed => same table, printed at the top of the run.
SEED = 20260729

NOMINAL = 0.90  # the 5th/95th percentile band the library's `probs` gives us
LO_IDX, HI_IDX = 0, 4  # probs = [0.05, 0.16, 0.50, 0.84, 0.95]
ALPHA_ROBUST = 0.10  # robust_svar_bounds default => 90% robust credible region

REPS_BVAR_FULL, REPS_BVAR_QUICK = 700, 60
REPS_IMPACT_FULL, REPS_IMPACT_QUICK = 2500, 200
REPS_SET_FULL, REPS_SET_QUICK = 400, 40
REPS_NARR_FULL, REPS_NARR_QUICK = 250, 25
REPS_ZERO_FULL, REPS_ZERO_QUICK = 400, 40
REPS_BREAK_FULL, REPS_BREAK_QUICK = 2000, 250
REPS_BREAKT_FULL, REPS_BREAKT_QUICK = 1200, 150

# ---------------------------------------------------------------- DGPs
PERSIST = {
    "name": "PERSIST",
    "A1": np.array([[0.85, 0.10], [0.03, 0.85]]),
    "sigma": np.array([[1.0, 0.30], [0.30, 1.0]]),
    "T": 100,
}
WN = {"name": "WN", "A1": np.zeros((2, 2)), "sigma": PERSIST["sigma"], "T": 100}
SVAR = {
    "name": "SVAR",
    "A1": np.array([[0.60, 0.10], [0.05, 0.50]]),
    "A0": np.array([[1.0, -0.50], [0.60, 0.90]]),
    "T": 200,
}
RECURSIVE = {
    "name": "RECURSIVE",
    "A1": SVAR["A1"],
    "A0": np.array([[1.0, 0.0], [0.60, 0.90]]),
    "T": 200,
}

# The sign pattern of SVAR["A0"] at impact, imposed as restrictions. True by
# construction, so the truth is in the identified set.
SIGN_RESTRICTIONS = [(0, 0, 0, "+"), (1, 0, 0, "+"), (0, 1, 0, "-"), (1, 1, 0, "+")]
# Sign restrictions that are true of RECURSIVE["A0"] (its (0,1) cell is the zero).
RECURSIVE_SIGNS = [(0, 0, 0, "+"), (1, 1, 0, "+")]
RECURSIVE_ZEROS = [(0, 1, 0)]

CELLS = ((0, 0), (1, 0), (0, 1), (1, 1))
CELL_LABEL = {
    (0, 0): "y0 <- shock0",
    (1, 0): "y1 <- shock0",
    (0, 1): "y0 <- shock1",
    (1, 1): "y1 <- shock1",
}


# ==========================================================================
# helpers
# ==========================================================================
def mc_se(p, reps):
    """Monte Carlo standard error of a coverage estimate."""
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / max(reps, 1)))


def header(text):
    print()
    print("=" * 104)
    print(text)
    print("=" * 104)


def simulate_var1(A1, impact, T, reps, rng, burn=300):
    """`reps` independent VAR(1) paths, vectorised across replications.

    Returns `(Y, E)` with `Y[r]` a (T, n) sample and `E[r]` the (T, n) unit
    structural shocks that generated it (needed to build narrative
    restrictions that are TRUE in the DGP).
    """
    n = A1.shape[0]
    E = rng.standard_normal((reps, T + burn, n))
    Y = np.empty((reps, T + burn, n))
    prev = np.zeros((reps, n))
    for t in range(T + burn):
        prev = prev @ A1.T + E[:, t] @ impact.T
        Y[:, t] = prev
    return Y[:, burn:], E[:, burn:]


def true_var1_irf(A1, impact, horizon):
    """Population IRF `A1^h @ impact`, exact in closed form."""
    n = A1.shape[0]
    psi = np.eye(n)
    out = np.empty((horizon + 1, n, n))
    for h in range(horizon + 1):
        out[h] = psi @ impact
        psi = A1 @ psi
    return out


LABW = 24  # design-label column width, shared by every table below


def cov_table(title, row_labels, cov, reps, horizons, note=None, truth=None):
    """Print coverage +/- MC se; rows are designs, columns are horizons."""
    print()
    print(f"  {title}")
    if truth is not None:
        print("    truth: " + " ".join(f"h{h}={truth[h]:+.3f}" for h in horizons))
    if note:
        print(f"    {note}")
    print("  " + f"{'design':<{LABW}s}" + "".join(f"{'h=' + str(h):<13s}" for h in horizons))
    for label, row in zip(row_labels, cov):
        cells = "".join(f"{c:.3f}+-{mc_se(c, reps):.3f} " for c in row)
        print("  " + f"{label:<{LABW}s}" + cells)


def width_table(title, row_labels, width, horizons):
    """Print mean interval widths; rows are designs, columns are horizons."""
    print()
    print(f"  {title}")
    print("  " + f"{'design':<{LABW}s}" + "".join(f"{'h=' + str(h):<10s}" for h in horizons))
    for label, row in zip(row_labels, width):
        print("  " + f"{label:<{LABW}s}" + "".join(f"{w:<10.3f}" for w in row))


# ==========================================================================
# structural facts -- exact identities, verified once, not coverage claims
# ==========================================================================
def structural_checks():
    """Facts about what these functions return, checked at one draw.

    These are not coverage measurements; they exist so the coverage tables
    below can honestly exclude the cells that "cover" by construction, and so
    the inventory claim "bvar_fit exposes no interval" is verified rather than
    asserted from the type stub.
    """
    rng = np.random.default_rng(SEED + 99)
    Y, _ = simulate_var1(PERSIST["A1"], np.linalg.cholesky(PERSIST["sigma"]), 160, 1, rng)
    Yz, _ = simulate_var1(RECURSIVE["A1"], RECURSIVE["A0"], 200, 1, rng)
    d, dz = Y[0], Yz[0]

    fit = tsecon.bvar_fit(d, lags=1)
    hier = tsecon.bvar_hierarchical(d, lags=1)
    draws = np.array(tsecon.bvar_irf_draws(d, lags=1, horizon=4, n_draws=200, seed=1))
    draws_again = np.array(tsecon.bvar_irf_draws(d, lags=1, horizon=4, n_draws=200, seed=1))
    zs = tsecon.zero_sign_svar(
        dz, RECURSIVE_SIGNS, RECURSIVE_ZEROS, lags=1, horizon=4, n_draws=200, seed=1
    )
    zq = np.array(zs["quantiles"])
    bp = tsecon.bai_perron(
        np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 2.0]),
        np.ones((200, 1)),
        max_breaks=2,
    )

    interval_words = ("se", "bse", "lower", "upper", "ci_", "quantile", "pval", "tstat")
    fit_has_interval = any(w in k for k in fit for w in interval_words)
    hier_has_interval = any(w in k for k in hier for w in interval_words)

    return {
        # bvar_fit / bvar_hierarchical are point-summary functions: no band.
        "bvar_fit_keys": sorted(fit),
        "bvar_fit_has_interval": bool(fit_has_interval),
        "bvar_hierarchical_has_interval": bool(hier_has_interval),
        # Cholesky impact is exactly lower-triangular in every draw.
        "chol_impact_upper_zero": float(np.abs(draws[:, 0, 0, 1]).max()),
        # Same seed, same numbers.
        "bvar_irf_draws_reproducible": bool(np.array_equal(draws, draws_again)),
        # zero_sign_svar imposes the zero exactly: the band collapses there.
        "zero_cell_band_width": float(
            np.abs(zq[0, 0, 1, HI_IDX] - zq[0, 0, 1, LO_IDX])
        ),
        # Bai's 90% interval nests inside the 95% one.
        "bai_ci_nested": bool(
            bp["ci_lower_95"][0] <= bp["ci_lower_90"][0]
            and bp["ci_upper_90"][0] <= bp["ci_upper_95"][0]
        ),
        # break_dates is the LAST index of the earlier regime (off-by-one trap).
        "bai_break_date_is_regime_end": bool(bp["break_dates"][0] == bp["regime_ends"][0]),
    }


# ==========================================================================
# Experiment 1 -- BVAR credible bands: a measurement of the prior
# ==========================================================================
def exp_bvar_prior_centring(reps, horizon=12, n_draws=400):
    """Frequentist coverage of nominal-90% BVAR credible bands, by prior.

    NOT a coverage test of a confidence interval: a credible band makes no
    repeated-sampling promise. What this measures is how much of the band's
    position is the data and how much is `delta` (the prior mean on the own
    first lag) and `lambda1` (overall tightness).

    Designs, all on the same PERSIST samples:
      default        delta=0.0, lambda1=0.2   the library defaults; the prior
                                              mean is white noise while the
                                              truth has own lags 0.85
      random walk    delta=1.0, lambda1=0.2   over-persistent prior
      oracle         delta=0.85, lambda1=0.2  prior mean on the own lags IS
                                              the truth
      oracle-tight   delta=0.85, lambda1=0.05 well-centred AND aggressive
      over-tight     delta=0.85, lambda1=0.02 well-centred on the own lags but
                                              tight enough to crush the true
                                              cross lags to zero
      diffuse        delta=0.0, lambda1=5.0   prior barely binds; closest
                                              thing here to a flat-prior Bayes
                                              band
      emp-Bayes d=0  lambda1 from `bvar_hierarchical` (Giannone-Lenza-
      emp-Bayes d=1  Primiceri marginal-likelihood maximisation), plugged in.
                     Note this CONDITIONS on the selected lambda1 and so
                     ignores hyperparameter uncertainty.
      SSVS           `bvar_ssvs` spike-and-slab. The truth has small but
                     nonzero cross lags (0.10, 0.03), exactly the coefficients
                     a spike-and-slab is designed to zero out.
    """
    dgp = PERSIST
    A1, T = dgp["A1"], dgp["T"]
    chol = np.linalg.cholesky(dgp["sigma"])
    truth = true_var1_irf(A1, chol, horizon)

    rng = np.random.default_rng(SEED)
    Y, _ = simulate_var1(A1, chol, T, reps, rng)

    niw_designs = [
        ("default", dict(delta=0.0, lambda1=0.2)),
        ("random walk", dict(delta=1.0, lambda1=0.2)),
        ("oracle", dict(delta=0.85, lambda1=0.2)),
        ("oracle-tight", dict(delta=0.85, lambda1=0.05)),
        ("over-tight", dict(delta=0.85, lambda1=0.02)),
        ("diffuse", dict(delta=0.0, lambda1=5.0)),
    ]
    eb_designs = [("emp-Bayes d=0", 0.0), ("emp-Bayes d=1", 1.0)]
    labels = [lab for lab, _ in niw_designs] + [lab for lab, _ in eb_designs] + ["SSVS spike-slab"]
    design_params = {
        lab: f"delta={kw['delta']:.2f}, lambda1={kw['lambda1']:.2f}" for lab, kw in niw_designs
    }
    for lab, delta in eb_designs:
        design_params[lab] = f"delta={delta:.2f}, lambda1 from bvar_hierarchical"
    design_params["SSVS spike-slab"] = "bvar_ssvs spike-and-slab, library defaults"

    shape = (len(labels), horizon + 1, 2, 2)
    hits = np.zeros(shape)
    width = np.zeros(shape)
    med = np.empty((len(labels), reps, horizon + 1, 2, 2))
    lam_eb = {lab: [] for lab, _ in eb_designs}

    def score(k, draws):
        lo, mid, hi = np.percentile(draws, [5, 50, 95], axis=0)
        hits[k] += (lo <= truth) & (truth <= hi)
        width[k] += hi - lo
        med[k, r] = mid

    for r in range(reps):
        d = Y[r]
        seed_r = 1_000_000 + r
        for k, (_, kw) in enumerate(niw_designs):
            score(
                k,
                np.array(
                    tsecon.bvar_irf_draws(
                        d, lags=1, horizon=horizon, n_draws=n_draws, seed=seed_r, **kw
                    )
                ),
            )
        for j, (lab, delta) in enumerate(eb_designs):
            hb = tsecon.bvar_hierarchical(d, lags=1, delta=delta)
            lam = float(hb["lambda1_opt"])
            lam_eb[lab].append(lam)
            score(
                len(niw_designs) + j,
                np.array(
                    tsecon.bvar_irf_draws(
                        d,
                        lags=1,
                        horizon=horizon,
                        n_draws=n_draws,
                        seed=seed_r,
                        delta=delta,
                        lambda1=lam,
                    )
                ),
            )
        ss = tsecon.bvar_ssvs(
            d, lags=1, n_draws=2 * n_draws, burn=300, seed=seed_r, horizon=horizon
        )
        score(len(labels) - 1, np.array(ss["irf_draws"]))

    bias = np.median(med - truth[None, None], axis=1)
    mc_sd = med.std(axis=1, ddof=1)
    return {
        "dgp": dgp["name"],
        "T": T,
        "reps": reps,
        "horizon": horizon,
        "n_draws": n_draws,
        "nominal": NOMINAL,
        "labels": labels,
        "design_params": design_params,
        "cov": hits / reps,
        "width": width / reps,
        "bias": bias,
        "mc_sd": mc_sd,
        "truth": truth,
        "mean_lambda1_eb": {k: float(np.mean(v)) for k, v in lam_eb.items()},
        "kind": "CRED (Bayesian credible band -- no frequentist promise)",
    }


def report_bvar_prior_centring(res, show=(0, 1, 2, 4, 8, 12)):
    show = tuple(h for h in show if h <= res["horizon"])
    header(
        "EXPERIMENT 1  bvar_irf_draws / bvar_ssvs credible bands -- "
        "frequentist coverage as a DIAGNOSTIC"
    )
    print(f"  DGP {res['dgp']} (own lags 0.85, largest root 0.905), T={res['T']}, "
          f"lags=1, {res['reps']} reps, {res['n_draws']} posterior draws")
    print("  nominal band 90% (5th/95th posterior percentile). A credible band makes NO")
    print("  repeated-sampling promise -- these numbers measure the PRIOR, not a defect.")
    print("  designs (delta = prior mean on the own first lag, lambda1 = overall tightness):")
    for lab in res["labels"]:
        print(f"    {lab:<18s}{res['design_params'][lab]}")
    print("  mean lambda1 actually chosen by bvar_hierarchical: " + ", ".join(
        f"{k} -> {v:.3f}" for k, v in res["mean_lambda1_eb"].items()))

    for cell in ((0, 0), (1, 0)):
        i, j = cell
        cov_table(
            f"coverage of the {CELL_LABEL[cell]} orthogonalised IRF, nominal 0.90",
            res["labels"],
            res["cov"][:, :, i, j][:, list(show)],
            res["reps"],
            show,
            truth=res["truth"][:, i, j],
        )
    # the attribution triad: is the band the wrong width, or in the wrong place?
    bias_h = show[::2] if len(show) > 4 else show
    print()
    print("  WHY: median bias of the posterior median / mean band width / MC sd of the posterior")
    print("  median, for the y0 <- shock0 cell. |bias| ~ width means the band is in the wrong PLACE.")
    print("  " + f"{'design':<{LABW}s}" + "".join(f"{'h=' + str(h):<17s}" for h in bias_h))
    for k, lab in enumerate(res["labels"]):
        cells = "".join(
            f"{res['bias'][k, h, 0, 0]:+.3f}/{res['width'][k, h, 0, 0]:.2f}/"
            f"{res['mc_sd'][k, h, 0, 0]:.2f} "
            for h in bias_h
        )
        print("  " + f"{lab:<{LABW}s}" + cells)


# ==========================================================================
# Experiment 1b -- isolating the posterior scale convention for Sigma
# ==========================================================================
def exp_bvar_impact_vs_exact(reps, n_draws=600):
    """Impact band coverage when the prior mean IS the truth.

    On a white-noise DGP fitted as a VAR(1) with `delta=0`, the Minnesota
    prior mean is exactly correct, so prior mis-centring cannot explain any
    coverage gap. The impact response of y0 to shock 0 is `chol(Sigma)[0,0]`,
    the equation-0 innovation standard deviation -- a scalar with an EXACT
    frequentist interval from the residual chi-square. Running both on the
    same samples separates "the approximation" from "our harness".

    Mechanism, for reference: the conjugate NIW posterior uses
    `vbar = v0 + T` with `v0 = n + 2`, and does not subtract the `k = 1 + n*p`
    regressors, whereas the sampling distribution of the residual variance has
    `T_eff - k` degrees of freedom. Here that is `vbar = 104` against
    `df_resid = 96`, so the posterior for Sigma is tighter than the sampling
    distribution of its estimator. This is a standard conjugate-BVAR
    convention (Kadiyala-Karlsson, Banbura-Giannone-Reichlin), not an error --
    but it is why a 90% credible band is not a 90% confidence interval even
    with a perfectly centred prior.
    """
    dgp = WN
    T, chol = dgp["T"], np.linalg.cholesky(dgp["sigma"])
    target = chol[0, 0]
    rng = np.random.default_rng(SEED + 1)
    Y, _ = simulate_var1(dgp["A1"], chol, T, reps, rng, burn=10)

    hits_b = {0.2: 0, 5.0: 0}
    width_b = {0.2: 0.0, 5.0: 0.0}
    hits_exact = 0
    width_exact = 0.0
    for r in range(reps):
        d = Y[r]
        for lam in (0.2, 5.0):
            draws = np.array(
                tsecon.bvar_irf_draws(
                    d, lags=1, horizon=0, n_draws=n_draws, seed=2_000_000 + r,
                    delta=0.0, lambda1=lam,
                )
            )[:, 0, 0, 0]
            lo, hi = np.percentile(draws, [5, 95])
            hits_b[lam] += int(lo <= target <= hi)
            width_b[lam] += hi - lo
        yv = d[1:, 0]
        X = np.column_stack([np.ones(T - 1), d[:-1]])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        df = len(yv) - X.shape[1]
        s2 = float(resid @ resid) / df
        lo = np.sqrt(df * s2 / stats.chi2.ppf(0.95, df))
        hi = np.sqrt(df * s2 / stats.chi2.ppf(0.05, df))
        hits_exact += int(lo <= target <= hi)
        width_exact += hi - lo

    n, lags = 2, 1
    df_resid = (T - lags) - (1 + n * lags)
    vbar = (n + 2) + T
    naive = float(
        stats.chi2.cdf(stats.chi2.ppf(0.95, vbar), df_resid)
        - stats.chi2.cdf(stats.chi2.ppf(0.05, vbar), df_resid)
    )
    return {
        "dgp": dgp["name"],
        "T": T,
        "reps": reps,
        "target": float(target),
        "cov_bvar": {k: v / reps for k, v in hits_b.items()},
        "width_bvar": {k: v / reps for k, v in width_b.items()},
        "cov_exact": hits_exact / reps,
        "width_exact": width_exact / reps,
        "df_resid": df_resid,
        "iw_posterior_df": vbar,
        "df_only_prediction": naive,
        "kind": "CRED vs an exact frequentist reference",
    }


def report_bvar_impact_vs_exact(res):
    header("EXPERIMENT 1b  the impact band with a PERFECTLY CENTRED prior vs an exact interval")
    print(f"  DGP {res['dgp']} (white noise) fitted as a VAR(1) with delta=0, so the prior mean")
    print(f"  is exactly right. Target: chol(Sigma)[0,0] = {res['target']:.4f}. T={res['T']}, "
          f"{res['reps']} reps, nominal 90%.")
    print()
    print("  " + f"{'interval':<40s}{'coverage':<18s}{'mean width':<12s}")
    for lam in (0.2, 5.0):
        p = res["cov_bvar"][lam]
        print("  " + f"{'bvar_irf_draws band, lambda1=' + str(lam):<40s}"
              f"{p:.4f}+-{mc_se(p, res['reps']):.4f}    {res['width_bvar'][lam]:.4f}")
    p = res["cov_exact"]
    print("  " + f"{'exact chi-square interval (reference)':<40s}"
          f"{p:.4f}+-{mc_se(p, res['reps']):.4f}    {res['width_exact']:.4f}")
    print()
    print(f"  Mechanism: IW posterior df = v0 + T = {res['iw_posterior_df']}, but the residual")
    print(f"  sampling df is T_eff - k = {res['df_resid']}. A pure df mismatch of that size would")
    print(f"  predict {res['df_only_prediction']:.3f} coverage; the prior scale offsets part of it,")
    print(f"  which is why the measurement lands above the crude prediction. Direction and rough")
    print(f"  magnitude are the conjugate df convention -- the APPROXIMATION, not the estimator.")


# ==========================================================================
# Experiment 2 -- set identification: three objects, three questions
# ==========================================================================
def exp_set_coverage(reps, horizon=6, n_draws=300):
    """Does the identified SET contain the true structural IRF?

    The DGP's true `A0` satisfies the imposed sign pattern, so the truth is a
    member of the identified set by construction. Three objects are scored on
    identical samples, under the default Minnesota shrinkage and under a
    near-diffuse one, so that "the reduced-form prior" and "set
    identification" can be told apart:

      pointwise band  `sign_restricted_svar` quantiles[...,5%]..[...,95%].
                      A Haar-prior credible band. No coverage promise.
      set envelope    `set_min`/`set_max`. Union over posterior draws of the
                      explored identified set. Not an interval about a point.
      robust CI       `robust_svar_bounds` robust_ci_lower/upper at
                      alpha=0.10. This one DOES target 1-alpha containment of
                      the whole identified set, so 0.90 is a fair benchmark.
    """
    dgp = SVAR
    A1, A0, T = dgp["A1"], dgp["A0"], dgp["T"]
    truth = true_var1_irf(A1, A0, horizon)
    rng = np.random.default_rng(SEED + 2)
    Y, _ = simulate_var1(A1, A0, T, reps, rng, burn=200)

    lambdas = (0.2, 5.0)
    objects = ("pointwise band", "set envelope", "robust CI")
    hits = {(o, lam): np.zeros((horizon + 1, 2, 2)) for o in objects for lam in lambdas}
    width = {(o, lam): np.zeros((horizon + 1, 2, 2)) for o in objects for lam in lambdas}
    accept, empty = {lam: [] for lam in lambdas}, {lam: [] for lam in lambdas}

    for r in range(reps):
        d = Y[r]
        seed_r = 3_000_000 + r
        for lam in lambdas:
            s = tsecon.sign_restricted_svar(
                d, SIGN_RESTRICTIONS, lags=1, horizon=horizon,
                n_draws=n_draws, seed=seed_r, lambda1=lam,
            )
            q = np.array(s["quantiles"])
            lo, hi = q[..., LO_IDX], q[..., HI_IDX]
            hits[("pointwise band", lam)] += (lo <= truth) & (truth <= hi)
            width[("pointwise band", lam)] += hi - lo
            mn, mx = np.array(s["set_min"]), np.array(s["set_max"])
            hits[("set envelope", lam)] += (mn <= truth) & (truth <= mx)
            width[("set envelope", lam)] += mx - mn
            accept[lam].append(s["diagnostics"]["acceptance_rate"])

            b = tsecon.robust_svar_bounds(
                d, SIGN_RESTRICTIONS, lags=1, horizon=horizon, n_draws=n_draws,
                seed=seed_r, alpha=ALPHA_ROBUST, lambda1=lam,
            )
            lo, hi = np.array(b["robust_ci_lower"]), np.array(b["robust_ci_upper"])
            hits[("robust CI", lam)] += (lo <= truth) & (truth <= hi)
            width[("robust CI", lam)] += hi - lo
            empty[lam].append(b["diagnostics"]["empty_set_rate"])

    return {
        "dgp": dgp["name"],
        "T": T,
        "reps": reps,
        "horizon": horizon,
        "n_draws": n_draws,
        "objects": objects,
        "lambdas": lambdas,
        "cov": {k: v / reps for k, v in hits.items()},
        "width": {k: v / reps for k, v in width.items()},
        "acceptance_rate": {lam: float(np.mean(v)) for lam, v in accept.items()},
        "empty_set_rate": {lam: float(np.mean(v)) for lam, v in empty.items()},
        "truth": truth,
        "kind": "SET (bounds, not an interval about a point) + CRED",
    }


def report_set_coverage(res, show=(0, 1, 2, 3, 4, 6)):
    show = tuple(h for h in show if h <= res["horizon"])
    header("EXPERIMENT 2  sign-restricted SVAR -- does the identified SET contain the truth?")
    print(f"  DGP {res['dgp']}, T={res['T']}, lags=1, {res['reps']} reps, "
          f"{res['n_draws']} posterior draws")
    print("  restrictions (variable, shock, horizon, sign): "
          + " ".join(f"({v},{s},{h},{g})" for v, s, h, g in SIGN_RESTRICTIONS))
    print("  The true A0 satisfies every imposed sign, so the truth IS in the identified set.")
    print("  Only the robust CI (alpha=0.10) advertises 1-alpha containment; the pointwise band")
    print("  is a Haar-prior credible band and the set envelope is not an interval about a point.")
    print(f"  mean rotation acceptance rate: " + ", ".join(
        f"lambda1={lam} -> {v:.3f}" for lam, v in res["acceptance_rate"].items()))
    print(f"  robust_svar_bounds empty_set_rate: " + ", ".join(
        f"lambda1={lam} -> {v:.4f}" for lam, v in res["empty_set_rate"].items()))

    labels, rows = [], []
    for obj in res["objects"]:
        for lam in res["lambdas"]:
            labels.append(f"{obj}, l1={lam}")
            rows.append((obj, lam))
    note = ("h=0 is sign-restricted: the set is open down to 0, so everything covers there.")
    for cell in CELLS:
        i, j = cell
        cov = np.array([res["cov"][k][:, i, j][list(show)] for k in rows])
        cov_table(
            f"{CELL_LABEL[cell]} structural IRF",
            labels, cov, res["reps"], show, note=note, truth=res["truth"][:, i, j],
        )
    width_table(
        "mean widths, y0 <- shock0 (the persistent cell where the prior bites hardest)",
        labels,
        [[res["width"][k][h, 0, 0] for h in show] for k in rows],
        show,
    )


# ==========================================================================
# Experiment 2b -- narrative restrictions that are TRUE in the DGP
# ==========================================================================
def exp_narrative(reps, horizon=6, n_draws=300, n_episodes=20):
    """Does adding TRUE narrative information tighten the band without losing coverage?

    Narrative restrictions are only meaningful if they are true. Here they are
    true by construction: the DGP's own structural shocks are known, so for
    each replication we pick the `n_episodes` largest-magnitude periods of each
    shock and impose their ACTUAL signs. Antolin-Diaz & Rubio-Ramirez
    importance weights then reweight the accepted rotations.

    Note what the ARW device does and does not do: it REWEIGHTS the accepted
    rotations by 1/P(N|S), it does not delete them. So a weighted quantile is
    not nested inside the unweighted one, and adding true information can widen
    a particular cell even while it tightens the band on average. The number of
    widened cells is recorded (`n_widened`) rather than assumed away.
    """
    dgp = SVAR
    A1, A0, T = dgp["A1"], dgp["A0"], dgp["T"]
    truth = true_var1_irf(A1, A0, horizon)
    rng = np.random.default_rng(SEED + 3)
    Y, E = simulate_var1(A1, A0, T, reps, rng, burn=200)
    lags = 1

    hits = {"sign only": np.zeros((horizon + 1, 2, 2)), "narrative": np.zeros((horizon + 1, 2, 2))}
    width = {k: np.zeros((horizon + 1, 2, 2)) for k in hits}
    ess, nacc, minp = [], [], []
    for r in range(reps):
        d = Y[r]
        seed_r = 4_000_000 + r
        nr = []
        for sh in (0, 1):
            e = E[r, lags:, sh]
            for p in np.argsort(-np.abs(e))[:n_episodes]:
                nr.append({
                    "type": "shock_sign", "shock": sh, "period": int(p),
                    "sign": "+" if e[p] > 0 else "-",
                })
        s = tsecon.sign_restricted_svar(
            d, SIGN_RESTRICTIONS, lags=lags, horizon=horizon, n_draws=n_draws, seed=seed_r
        )
        nv = tsecon.narrative_svar(
            d, SIGN_RESTRICTIONS, narrative_restrictions=nr, lags=lags,
            horizon=horizon, n_draws=n_draws, seed=seed_r,
        )
        for key, out in (("sign only", s), ("narrative", nv)):
            q = np.array(out["quantiles"])
            lo, hi = q[..., LO_IDX], q[..., HI_IDX]
            hits[key] += (lo <= truth) & (truth <= hi)
            width[key] += hi - lo
        dg = nv["diagnostics"]
        ess.append(dg["ess"])
        nacc.append(dg["narrative_acceptance_rate"])
        minp.append(dg["min_ptilde"])

    cov = {k: v / reps for k, v in hits.items()}
    wid = {k: v / reps for k, v in width.items()}
    ratio = wid["narrative"] / wid["sign only"]
    return {
        "dgp": dgp["name"],
        "T": T,
        "reps": reps,
        "horizon": horizon,
        "n_draws": n_draws,
        "n_statements": 2 * n_episodes,
        "cov": cov,
        "width": wid,
        "width_ratio": ratio,
        "mean_width_ratio": float(ratio.mean()),
        "max_width_ratio": float(ratio.max()),
        "n_widened": int((ratio > 1.0).sum()),
        "n_cells": int(ratio.size),
        "mean_cov_sign": float(cov["sign only"].mean()),
        "mean_cov_narrative": float(cov["narrative"].mean()),
        "mean_ess": float(np.mean(ess)),
        "mean_narrative_acceptance": float(np.mean(nacc)),
        "mean_min_ptilde": float(np.mean(minp)),
        "truth": truth,
        "kind": "SET + CRED (importance-reweighted Haar band)",
    }


def report_narrative(res, show=(0, 1, 2, 3, 4, 6)):
    show = tuple(h for h in show if h <= res["horizon"])
    header("EXPERIMENT 2b  narrative_svar with narrative statements that are TRUE in the DGP")
    print(f"  DGP {res['dgp']}, T={res['T']}, {res['reps']} reps, {res['n_draws']} draws, "
          f"{res['n_statements']} true shock-sign statements per replication")
    print(f"  ARW importance weights: mean ESS {res['mean_ess']:.0f} of {res['n_draws']}, "
          f"mean narrative acceptance {res['mean_narrative_acceptance']:.3f}, "
          f"mean min_ptilde {res['mean_min_ptilde']:.3f}")
    for cell in ((0, 0), (1, 1)):
        i, j = cell
        cov = np.array([res["cov"][k][:, i, j][list(show)] for k in ("sign only", "narrative")])
        cov_table(f"{CELL_LABEL[cell]} pointwise-band coverage",
                  ["sign only", "narrative (true)"], cov, res["reps"], show)
    width_table(
        "mean band width, y0 <- shock0 (does true narrative information buy tightness?)",
        ["sign only", "narrative", "ratio narr/sign"],
        [[res["width"]["sign only"][h, 0, 0] for h in show],
         [res["width"]["narrative"][h, 0, 0] for h in show],
         [res["width_ratio"][h, 0, 0] for h in show]],
        show,
    )
    print()
    print(f"  Across all {res['n_cells']} cell-horizon pairs the mean width ratio is "
          f"{res['mean_width_ratio']:.4f}, but {res['n_widened']} of them")
    print(f"  WIDEN (max ratio {res['max_width_ratio']:.4f}).")
    print("  That is not a defect: ARW importance-reweights the accepted rotations rather than")
    print("  deleting them, so a weighted quantile need not nest inside the unweighted one.")
    print(f"  Mean coverage over all cells and horizons: sign only {res['mean_cov_sign']:.4f}, "
          f"narrative {res['mean_cov_narrative']:.4f}.")


# ==========================================================================
# Experiment 2c -- zeros that point-identify: the one band about a point
# ==========================================================================
def exp_zero_sign(reps, horizon=6, n_draws=300):
    """`zero_sign_svar` when the imposed zero is TRUE and point-identifies.

    With n=2 and a strict-upper-triangle impact zero, the rotation is pinned,
    so the object stops being set-identified and becomes an ordinary Bayesian
    band about a point (the recursive Cholesky posterior). That makes
    frequentist coverage a fair diagnostic and lets the reduced-form prior be
    read off cleanly: the same code, the same restrictions, two `lambda1`.
    """
    dgp = RECURSIVE
    A1, A0, T = dgp["A1"], dgp["A0"], dgp["T"]
    truth = true_var1_irf(A1, A0, horizon)
    rng = np.random.default_rng(SEED + 4)
    Y, _ = simulate_var1(A1, A0, T, reps, rng, burn=200)

    lambdas = (0.2, 5.0)
    hits = {lam: np.zeros((horizon + 1, 2, 2)) for lam in lambdas}
    width = {lam: np.zeros((horizon + 1, 2, 2)) for lam in lambdas}
    ess = {lam: [] for lam in lambdas}
    for r in range(reps):
        for lam in lambdas:
            z = tsecon.zero_sign_svar(
                Y[r], RECURSIVE_SIGNS, RECURSIVE_ZEROS, lags=1, horizon=horizon,
                n_draws=n_draws, seed=5_000_000 + r, lambda1=lam,
            )
            q = np.array(z["quantiles"])
            lo, hi = q[..., LO_IDX], q[..., HI_IDX]
            hits[lam] += (lo <= truth) & (truth <= hi)
            width[lam] += hi - lo
            ess[lam].append(z["ess"])
    return {
        "dgp": dgp["name"],
        "T": T,
        "reps": reps,
        "horizon": horizon,
        "n_draws": n_draws,
        "lambdas": lambdas,
        "cov": {lam: v / reps for lam, v in hits.items()},
        "width": {lam: v / reps for lam, v in width.items()},
        "mean_ess": {lam: float(np.mean(v)) for lam, v in ess.items()},
        "truth": truth,
        "kind": "CRED about a point-identified object",
    }


def report_zero_sign(res, show=(0, 1, 2, 3, 4, 6)):
    show = tuple(h for h in show if h <= res["horizon"])
    header("EXPERIMENT 2c  zero_sign_svar with a TRUE point-identifying zero")
    print(f"  DGP {res['dgp']} (truly recursive), T={res['T']}, {res['reps']} reps, "
          f"{res['n_draws']} draws")
    print(f"  zeros {RECURSIVE_ZEROS} (variable, shock, horizon); signs {RECURSIVE_SIGNS}")
    print("  The zero pins the rotation, so this is a credible band about a POINT (not a set).")
    print("  ARW weights are exactly 1 for impact-only zeros: mean ESS " + ", ".join(
        f"lambda1={lam} -> {v:.0f}" for lam, v in res["mean_ess"].items()))
    for cell in CELLS:
        i, j = cell
        if cell == (0, 1):
            print()
            print(f"  {CELL_LABEL[cell]}: the imposed zero. Band width is exactly 0 and the truth")
            print("  is exactly 0, so coverage is 1.000 by construction -- excluded, measures nothing.")
            continue
        cov = np.array([res["cov"][lam][:, i, j][list(show)] for lam in res["lambdas"]])
        cov_table(CELL_LABEL[cell], [f"lambda1={lam}" for lam in res["lambdas"]],
                  cov, res["reps"], show, truth=res["truth"][:, i, j])


# ==========================================================================
# Experiment 3 -- bai_perron break-date confidence intervals
# ==========================================================================
def _break_run(reps, T, delta, seed, trim=0.15, max_breaks=2):
    """One (T, delta) cell: detection rate, conditional and unconditional coverage."""
    tau = T // 2  # first index of the SECOND regime
    truth = tau - 1  # bai_perron reports the LAST index of the FIRST regime
    rng = np.random.default_rng(seed)
    x = np.ones((T, 1))
    det = h90 = h95 = 0
    widths, scales = [], []
    for _ in range(reps):
        y = rng.standard_normal(T)
        y[tau:] += delta
        res = tsecon.bai_perron(y, x, max_breaks=max_breaks, trim=trim)
        if int(res["n_breaks"]) != 1:
            continue
        det += 1
        lo95, hi95 = int(res["ci_lower_95"][0]), int(res["ci_upper_95"][0])
        lo90, hi90 = int(res["ci_lower_90"][0]), int(res["ci_upper_90"][0])
        h95 += int(lo95 <= truth <= hi95)
        h90 += int(lo90 <= truth <= hi90)
        widths.append(hi95 - lo95 + 1)
        scales.append(float(res["ci_scale"][0]))
    return {
        "T": T,
        "delta": delta,
        "reps": reps,
        "n_detected": det,
        "detect_rate": det / reps,
        "cond90": h90 / det if det else float("nan"),
        "cond95": h95 / det if det else float("nan"),
        "uncond95": h95 / reps,
        "mean_width95": float(np.mean(widths)) if widths else float("nan"),
        "mean_scale": float(np.mean(scales)) if scales else float("nan"),
    }


def exp_break_date_ci(reps, deltas=(3.0, 2.0, 1.0, 0.5, 0.25)):
    """Bai (1997) break-date CI coverage as the break shrinks. T fixed at 200.

    The only object in this module that promises frequentist coverage. Two
    coverage numbers are reported because two are defensible:
      cond    conditional on the sequential supF selecting exactly one break
              (the interval only exists then) -- but this conditions on a
              data-dependent event, so it is not a clean unconditional rate;
      uncond  counting a non-detection as a miss -- the rate a user who just
              runs the function and reads off the interval actually faces.
    """
    return {
        "T": 200,
        "trim": 0.15,
        "reps": reps,
        "cells": [
            _break_run(reps, 200, d, SEED + 10 + k) for k, d in enumerate(deltas)
        ],
        "kind": "CI/PRED for a discrete break date (frequentist, nominal 90/95%)",
    }


def exp_break_date_T(reps, Ts=(200, 400, 800), deltas=(1.0, 0.5)):
    """Same break magnitude, more data.

    Bai's interval is derived under FIXED break magnitude, where the break
    date is estimated with O(1) precision: the CI half-width is
    `ceil(c_alpha / (delta' Q delta / sigma^2))`, which does not involve T at
    all. So the interval should NOT shrink with T -- but its coverage should
    improve, because the sampling distribution of the argmax converges to the
    limit Bai's critical values come from. If coverage improves with T at a
    fixed break while the width stays put, the small-break shortfall is the
    finite-sample quality of the APPROXIMATION, not a wrong formula.
    """
    return {
        "reps": reps,
        "cells": [
            _break_run(reps, T, d, SEED + 40 + 7 * k + j)
            for k, T in enumerate(Ts)
            for j, d in enumerate(deltas)
        ],
        "kind": "CI for a discrete break date, T sweep at fixed break size",
    }


def _report_break_cells(cells, reps, title):
    print()
    print(f"  {title}")
    print("  " + f"{'T':>5s}{'break/sigma':>13s}{'detected':>11s}"
          f"{'cond 95%':>18s}{'cond 90%':>18s}{'uncond 95%':>18s}"
          f"{'width95':>10s}{'scale':>9s}")
    for c in cells:
        nd = max(c["n_detected"], 1)
        print("  " + f"{c['T']:>5d}{c['delta']:>13.2f}{c['detect_rate']:>11.3f}"
              f"{c['cond95']:>11.3f}+-{mc_se(c['cond95'], nd):.3f}"
              f"{c['cond90']:>11.3f}+-{mc_se(c['cond90'], nd):.3f}"
              f"{c['uncond95']:>11.3f}+-{mc_se(c['uncond95'], reps):.3f}"
              f"{c['mean_width95']:>10.1f}{c['mean_scale']:>9.2f}")


def report_break_date(res_delta, res_T):
    header("EXPERIMENT 3  bai_perron Bai (1997) break-date confidence intervals")
    print(f"  DGP BREAK: y_t = e_t + delta*1{{t >= T/2}}, x = constant (the homogeneous case the")
    print(f"  model card claims), trim={res_delta['trim']}, max_breaks=2, {res_delta['reps']} reps.")
    print("  TRAP: break_dates is the LAST index of the earlier regime, so the truth is T/2 - 1.")
    print("  scale = delta' Q delta / sigma^2; the CI half-width is ceil(c_alpha / scale) + 1.")
    _report_break_cells(res_delta["cells"], res_delta["reps"],
                        "coverage as the break shrinks (T = 200)")
    _report_break_cells(res_T["cells"], res_T["reps"],
                        "coverage as T grows at FIXED break magnitude "
                        f"({res_T['reps']} reps)")


# ==========================================================================
# findings and assertions
# ==========================================================================
def findings(res):
    e1, e1b, e2, e2b, e2c = res["exp1"], res["exp1b"], res["exp2"], res["exp2b"], res["exp2c"]
    e3, e3b = res["exp3"], res["exp3b"],
    out = []

    lab = e1["labels"]
    i_def, i_or, i_ot, i_dif, i_ss = (
        lab.index("default"),
        lab.index("oracle-tight"),
        lab.index("over-tight"),
        lab.index("diffuse"),
        lab.index("SSVS spike-slab"),
    )
    h = min(4, e1["horizon"])
    hmax = e1["horizon"]
    out.append(
        f"UNDER-COVERS (prior, not estimator): the DEFAULT BVAR prior (delta=0, lambda1=0.2) "
        f"gives a nominal-90% credible band with frequentist coverage "
        f"{e1['cov'][i_def, h, 0, 0]:.3f}+-{mc_se(e1['cov'][i_def, h, 0, 0], e1['reps']):.3f} "
        f"for the h={h} own IRF. Bias {e1['bias'][i_def, h, 0, 0]:+.3f} against width "
        f"{e1['width'][i_def, h, 0, 0]:.2f}: the band is in the wrong PLACE, because the prior "
        f"mean is white noise and the truth has own lags 0.85."
    )
    out.append(
        f"FIXED BY CENTRING, not by widening: the same 90% band with a well-centred, TIGHTER "
        f"prior (delta=0.85, lambda1=0.05) covers "
        f"{e1['cov'][i_or, h, 0, 0]:.3f}+-{mc_se(e1['cov'][i_or, h, 0, 0], e1['reps']):.3f} at "
        f"width {e1['width'][i_or, h, 0, 0]:.2f} -- higher coverage from a NARROWER band. This is "
        f"the sense in which a well-centred prior can reach nominal; it is not evidence about any "
        f"interval's calibration."
    )
    out.append(
        f"OVER-SHRINKAGE: push the same well-centred prior to lambda1=0.02 and coverage collapses "
        f"to {e1['cov'][i_ot, hmax, 0, 0]:.3f} at h={hmax} (own) and "
        f"{e1['cov'][i_ot, hmax, 1, 0]:.3f} (cross): correct on the own lags is not enough when the "
        f"true cross lags are crushed to zero."
    )
    out.append(
        f"BOTH DIRECTIONS FROM ONE PRIOR: the random-walk prior (delta=1) at h={hmax} covers "
        f"{e1['cov'][lab.index('random walk'), hmax, 0, 0]:.3f} for the own response "
        f"but {e1['cov'][lab.index('random walk'), hmax, 1, 0]:.3f} for the cross "
        f"response -- one cell near nominal, the other not, from a single hyperparameter."
    )
    eb = e1["mean_lambda1_eb"]
    i_eb0 = lab.index("emp-Bayes d=0")
    i_eb1 = lab.index("emp-Bayes d=1")
    out.append(
        "EMPIRICAL BAYES HELPS BUT CANNOT RESCUE A MIS-CENTRED PRIOR: bvar_hierarchical chooses "
        + ", ".join(f"mean lambda1 {v:.3f} at {k.split()[1]}" for k, v in eb.items())
        + f" -- it LOOSENS the badly centred delta=0 prior (0.2 -> "
        f"{eb['emp-Bayes d=0']:.2f}) and TIGHTENS the well-centred delta=1 prior "
        f"(0.2 -> {eb['emp-Bayes d=1']:.2f}), exactly as marginal-likelihood logic says "
        f"it should. Coverage at h={hmax} (own) goes to {e1['cov'][i_eb0, hmax, 0, 0]:.3f} and "
        f"{e1['cov'][i_eb1, hmax, 0, 0]:.3f} respectively: tuning the tightness cannot fix a prior "
        f"mean in the wrong place, and the plug-in band also ignores uncertainty in lambda1 itself."
    )
    out.append(
        f"SSVS UNDER-COVERS TRUE-BUT-SMALL COEFFICIENTS: bvar_ssvs bands cover "
        f"{e1['cov'][i_ss, hmax, 1, 0]:.3f}+-{mc_se(e1['cov'][i_ss, hmax, 1, 0], e1['reps']):.3f} "
        f"for the h={hmax} CROSS response against {e1['cov'][i_dif, hmax, 1, 0]:.3f} for the "
        f"diffuse NIW band. The spike prior does what it is for: it zeroes the true 0.10 and 0.03 "
        f"cross lags. Estimator behaving as designed, band therefore misplaced."
    )
    out.append(
        f"EVEN A PERFECT PRIOR LEAVES A GAP: on white noise fitted with delta=0 (prior mean exactly "
        f"right), the 90% impact band covers "
        f"{e1b['cov_bvar'][5.0]:.4f}+-{mc_se(e1b['cov_bvar'][5.0], e1b['reps']):.4f} while an exact "
        f"chi-square interval for the same scalar covers "
        f"{e1b['cov_exact']:.4f}+-{mc_se(e1b['cov_exact'], e1b['reps']):.4f} on the SAME samples. "
        f"The ~{100 * (e1b['cov_exact'] - e1b['cov_bvar'][5.0]):.1f}pp gap is the conjugate NIW df "
        f"convention (posterior df {e1b['iw_posterior_df']} vs residual df {e1b['df_resid']}) -- the "
        f"APPROXIMATION/convention, not a bug, and it is why a credible band is not a CI."
    )

    hh = min(3, e2["horizon"])
    band = e2["cov"][("pointwise band", 0.2)][hh, 0, 0]
    band_d = e2["cov"][("pointwise band", 5.0)][hh, 0, 0]
    rob = e2["cov"][("robust CI", 0.2)][hh, 0, 0]
    rob_d = e2["cov"][("robust CI", 5.0)][hh, 0, 0]
    env = e2["cov"][("set envelope", 0.2)][hh, 0, 0]
    out.append(
        f"SET IDENTIFICATION, IMPACT: at h=0 every restricted cell covers ~1.000 for all three "
        f"objects. That is not good calibration -- a weak sign restriction leaves the identified "
        f"set open down to 0, so the truth is inside whatever the data say. Impact coverage of a "
        f"sign-restricted band certifies nothing."
    )
    cell_min = {
        (i, j): min(e2["cov"][(o, lam)][hz, i, j]
                    for o in e2["objects"] for lam in e2["lambdas"]
                    for hz in range(e2["horizon"] + 1))
        for (i, j) in CELLS
    }
    triv = [c for c, v in cell_min.items() if v >= 0.95]
    if triv:
        out.append(
            "CELLS THAT MEASURE NOTHING: "
            + "; ".join(f"{CELL_LABEL[c]} never drops below {cell_min[c]:.3f}" for c in triv)
            + f" across every object, prior and horizon. Their true responses decay to about zero "
            f"fast (y0<-shock1 is {e2['truth'][2, 0, 1]:+.3f} at h=2, {e2['truth'][e2['horizon'], 0, 1]:+.3f} "
            f"at h={e2['horizon']}) while the identified set always contains zero. A coverage number "
            f"for a cell whose truth is nearly zero is arithmetic, not calibration. Reported, then "
            f"set aside: the informative cells in this design are y0<-shock0 and y1<-shock1."
        )
    out.append(
        f"THE POINTWISE HAAR BAND IS NOT A CONFIDENCE BAND: at h={hh} the y0<-shock0 pointwise "
        f"5-95 band covers {band:.3f}+-{mc_se(band, e2['reps']):.3f}, the prior-robust CI "
        f"{rob:.3f}+-{mc_se(rob, e2['reps']):.3f}, the set envelope "
        f"{env:.3f}+-{mc_se(env, e2['reps']):.3f}. The ordering band < robust < envelope is what "
        f"the three objects mean; only the middle one aims at 0.90."
    )
    out.append(
        f"UNDER-COVERS (reduced-form prior, not the set logic): the robust CI at h={hh} covers "
        f"{rob:.3f} under the DEFAULT lambda1=0.2 but {rob_d:.3f} under lambda1=5.0. "
        f"Giacomini-Kitagawa robustness is robustness to the ROTATION prior; it inherits the "
        f"Minnesota prior on the reduced form, and with delta=0 that prior pulls a persistent "
        f"response down and takes the whole set with it. Same story for the pointwise band "
        f"({band:.3f} -> {band_d:.3f})."
    )
    h1 = min(1, e2b["horizon"])
    out.append(
        f"TRUE NARRATIVE INFORMATION HELPS, MODESTLY: {e2b['n_statements']} true shock-sign "
        f"statements per sample take the mean ARW ESS to {e2b['mean_ess']:.0f}/{e2b['n_draws']} and "
        f"tighten the h={h1} y0<-shock0 band from "
        f"{e2b['width']['sign only'][h1, 0, 0]:.3f} to {e2b['width']['narrative'][h1, 0, 0]:.3f} "
        f"while coverage moves {e2b['cov']['sign only'][h1, 0, 0]:.3f} -> "
        f"{e2b['cov']['narrative'][h1, 0, 0]:.3f}. With n=2 the rotation space is one-dimensional "
        f"and the sign restrictions already pin most of it, so do not read this as the general "
        f"size of the narrative gain."
    )
    out.append(
        f"NARRATIVE BANDS ARE NOT NESTED (correcting an assumption this module originally made): "
        f"the mean width ratio narrative/sign-only is {e2b['mean_width_ratio']:.4f} over "
        f"{e2b['n_cells']} cell-horizon pairs, but {e2b['n_widened']} of them WIDEN (max ratio "
        f"{e2b['max_width_ratio']:.4f}). ARW imposes narrative information by importance-"
        f"reweighting the accepted rotations, not by discarding them, so the weighted quantile "
        f"band is not a subset of the unweighted one. Adding true information can move an "
        f"individual quantile outward. Correct behaviour, easy to mis-assume."
    )
    hz = min(3, e2c["horizon"])
    out.append(
        f"POINT-IDENTIFIED ZEROS, STILL SHORT: zero_sign_svar with a TRUE point-identifying zero "
        f"covers {e2c['cov'][0.2][hz, 0, 0]:.3f}+-{mc_se(e2c['cov'][0.2][hz, 0, 0], e2c['reps']):.3f} "
        f"at h={hz} (own) under the default lambda1=0.2 and "
        f"{e2c['cov'][5.0][hz, 0, 0]:.3f} under lambda1=5.0. Here the band IS about a point, so the "
        f"shortfall is the cleanest reading available of what the Minnesota prior costs a "
        f"frequentist reader."
    )

    big = e3["cells"][0]
    small = [c for c in e3["cells"] if abs(c["delta"] - 0.5) < 1e-9][0]
    tiny = e3["cells"][-1]
    out.append(
        f"OVER-COVERS: the bai_perron 95% break-date CI covers "
        f"{big['cond95']:.3f}+-{mc_se(big['cond95'], big['n_detected']):.3f} at break/sigma="
        f"{big['delta']:.1f} (nominal 0.95). The half-width is ceil(c/scale) PLUS ONE on each side, "
        f"so at a large break the discreteness padding dominates and the interval is conservative."
    )
    out.append(
        f"UNDER-COVERS: at break/sigma={small['delta']:.2f} the same 95% interval covers "
        f"{small['cond95']:.3f}+-{mc_se(small['cond95'], small['n_detected']):.3f} and the 90% one "
        f"{small['cond90']:.3f}+-{mc_se(small['cond90'], small['n_detected']):.3f}. At "
        f"break/sigma={tiny['delta']:.2f} it is {tiny['cond95']:.3f} conditional on detection, and "
        f"detection itself is only {tiny['detect_rate']:.3f} -- so the rate a user actually faces is "
        f"{tiny['uncond95']:.3f}+-{mc_se(tiny['uncond95'], tiny['reps']):.3f}."
    )
    tcells = {(c["T"], c["delta"]): c for c in e3b["cells"]}
    ws = [tcells[(T, 1.0)]["mean_width95"] for T in (200, 400, 800) if (T, 1.0) in tcells]
    cs = [tcells[(T, 0.5)]["cond95"] for T in (200, 400, 800) if (T, 0.5) in tcells]
    if len(ws) == 3 and len(cs) == 3:
        out.append(
            f"WHY, precisely: at break/sigma=1 the mean 95% CI width is "
            f"{ws[0]:.1f} / {ws[1]:.1f} / {ws[2]:.1f} at T=200/400/800 -- it does NOT shrink with T, "
            f"exactly as Bai's fixed-break asymptotics say (the break date is an O(1)-precision "
            f"parameter). Meanwhile coverage at break/sigma=0.5 goes "
            f"{cs[0]:.3f} -> {cs[1]:.3f} -> {cs[2]:.3f}. So the small-break shortfall is the "
            f"finite-sample quality of the argmax limit distribution -- the APPROXIMATION improving "
            f"in T -- not a mis-derived interval."
        )
    out.append(
        "NO INTERVAL AT ALL: bvar_fit and bvar_hierarchical return posterior MEANS and a marginal "
        "likelihood, no dispersion of any kind (verified in structural_checks). Any band from a "
        "Minnesota BVAR in this library has to come from bvar_irf_draws or bvar_ssvs. "
        "fry_pagan_svar likewise returns a single coherent draw, not an interval."
    )
    return out


def assertions(res, facts):
    """Only robust, qualitative facts. Nothing here was tuned to pass.

    Deliberately NOT asserted: any specific coverage number, and in particular
    anything that would have to hold at the third decimal. Where coverage
    genuinely fails it is printed by `findings`, not asserted away.
    """
    e1, e1b, e2, e2b, e2c = res["exp1"], res["exp1b"], res["exp2"], res["exp2b"], res["exp2c"]
    e3, e3b = res["exp3"], res["exp3b"]
    out = []
    lab = e1["labels"]
    reps1 = e1["reps"]
    h = min(4, e1["horizon"])
    hmax = e1["horizon"]

    # --- structural facts (exact identities, not coverage) -----------------
    out.append((
        "bvar_fit / bvar_hierarchical expose no interval-bearing key",
        not facts["bvar_fit_has_interval"] and not facts["bvar_hierarchical_has_interval"],
        f"bvar_fit keys = {facts['bvar_fit_keys']}",
    ))
    out.append((
        "bvar_irf_draws is reproducible at a fixed seed and Cholesky-exact at impact",
        facts["bvar_irf_draws_reproducible"] and facts["chol_impact_upper_zero"] == 0.0,
        f"repeat run identical; max |upper-triangle impact| = {facts['chol_impact_upper_zero']:.1e}",
    ))
    out.append((
        "zero_sign_svar imposes the zero exactly (band collapses at the restricted cell)",
        facts["zero_cell_band_width"] < 1e-10,
        f"band width at the zero cell = {facts['zero_cell_band_width']:.1e}",
    ))
    out.append((
        "bai_perron 90% interval nests in the 95% one; break_dates is the earlier regime's end",
        facts["bai_ci_nested"] and facts["bai_break_date_is_regime_end"],
        "nesting and the regime-end convention both hold",
    ))

    # --- experiment 1: prior centring dominates, in the right direction ----
    d_def = e1["cov"][lab.index("default"), h, 0, 0]
    d_or = e1["cov"][lab.index("oracle-tight"), h, 0, 0]
    out.append((
        "the default Minnesota prior materially under-covers the persistent own IRF",
        d_def < 0.80,
        f"h={h} own coverage {d_def:.3f}+-{mc_se(d_def, reps1):.3f} against nominal 0.90 "
        f"(threshold 0.80 is >8 MC se from nominal)",
    ))
    out.append((
        "a well-centred prior beats the default by a wide margin AND with a narrower band",
        (d_or - d_def) > 0.10
        and e1["width"][lab.index("oracle-tight"), h, 0, 0]
        <= e1["width"][lab.index("default"), h, 0, 0] * 1.10,
        f"oracle-tight {d_or:.3f} vs default {d_def:.3f} at h={h}; widths "
        f"{e1['width'][lab.index('oracle-tight'), h, 0, 0]:.2f} vs "
        f"{e1['width'][lab.index('default'), h, 0, 0]:.2f}",
    ))
    ot = e1["cov"][lab.index("over-tight"), hmax, 0, 0]
    out.append((
        "over-shrinking a correctly centred prior destroys coverage at long horizons",
        ot < 0.70,
        f"delta=0.85 lambda1=0.02 covers {ot:.3f} at h={hmax} (own)",
    ))
    ss = e1["cov"][lab.index("SSVS spike-slab"), hmax, 1, 0]
    out.append((
        "SSVS bands under-cover a true-but-small cross coefficient",
        ss < 0.80,
        f"h={hmax} cross coverage {ss:.3f}+-{mc_se(ss, reps1):.3f}; the spike zeroes the true 0.03",
    ))

    # --- experiment 1b: the exact reference validates the harness ----------
    out.append((
        "the exact chi-square reference interval is calibrated (validates this harness)",
        abs(e1b["cov_exact"] - 0.90) < 3 * mc_se(e1b["cov_exact"], e1b["reps"]),
        f"exact reference {e1b['cov_exact']:.4f}+-{mc_se(e1b['cov_exact'], e1b['reps']):.4f} "
        f"vs nominal 0.90",
    ))
    gap = e1b["cov_exact"] - e1b["cov_bvar"][5.0]
    out.append((
        "the BVAR impact band falls short of the exact interval even with a perfect prior mean",
        gap > 0.005,
        f"gap {100 * gap:.2f}pp (bvar {e1b['cov_bvar'][5.0]:.4f}, exact {e1b['cov_exact']:.4f}); "
        f"mechanism is the posterior df {e1b['iw_posterior_df']} vs residual df {e1b['df_resid']}",
    ))

    # --- experiment 2: the three set objects mean three things -------------
    hh = min(3, e2["horizon"])
    reps2 = e2["reps"]
    impact_min = min(
        e2["cov"][(o, lam)][0, i, j]
        for o in e2["objects"] for lam in e2["lambdas"] for (i, j) in CELLS
    )
    out.append((
        "at impact every sign-restricted object covers ~1 (the set is open to 0) -- "
        "so impact coverage certifies nothing",
        impact_min > 0.97,
        f"minimum impact coverage across all objects, priors and cells = {impact_min:.3f}",
    ))
    band_h = np.array([e2["cov"][("pointwise band", 0.2)][hz, 0, 0] for hz in range(1, e2["horizon"] + 1)])
    rob_h = np.array([e2["cov"][("robust CI", 0.2)][hz, 0, 0] for hz in range(1, e2["horizon"] + 1)])
    out.append((
        "the prior-robust bound covers materially better than the pointwise Haar band at h>=1",
        float(np.mean(rob_h - band_h)) > 0.03 and bool(np.all(rob_h >= band_h - 0.01)),
        f"mean gain {float(np.mean(rob_h - band_h)):+.3f} over h=1..{e2['horizon']} "
        f"(y0<-shock0); never worse by more than 0.01",
    ))
    diff_diffuse = float(
        np.mean([
            e2["cov"][(o, 5.0)][hz, 0, 0] - e2["cov"][(o, 0.2)][hz, 0, 0]
            for o in ("pointwise band", "robust CI")
            for hz in range(1, e2["horizon"] + 1)
        ])
    )
    out.append((
        "loosening the REDUCED-FORM prior fixes most of the set-identified shortfall",
        diff_diffuse > 0.03,
        f"mean coverage gain from lambda1 0.2 -> 5.0 across band and robust CI, h>=1: "
        f"{diff_diffuse:+.3f}",
    ))
    rob_diffuse_min = min(
        e2["cov"][("robust CI", 5.0)][hz, i, j]
        for hz in range(1, e2["horizon"] + 1) for (i, j) in CELLS
    )
    out.append((
        "with a near-diffuse reduced-form prior the robust CI is conservative, as GK predicts",
        rob_diffuse_min > 0.90 - 3 * mc_se(0.90, reps2),
        f"minimum robust-CI coverage over all cells and h>=1 = {rob_diffuse_min:.3f} "
        f"(nominal 0.90, 3 MC se = {3 * mc_se(0.90, reps2):.3f})",
    ))

    # --- experiment 2b: true narrative information tightens ON AVERAGE ------
    # NOTE: an earlier version of this file asserted that narrative restrictions
    # NEVER widen any cell. That assertion FAILED, and it deserved to: ARW
    # importance-reweights the accepted rotations instead of deleting them, so a
    # weighted quantile is not nested in the unweighted one. The claim was
    # wrong, not the measurement. What survives is the average statement.
    out.append((
        "true narrative restrictions tighten the band on average and the ARW weights bind",
        e2b["mean_width_ratio"] < 0.99 and e2b["mean_ess"] < e2b["n_draws"],
        f"mean width ratio narrative/sign = {e2b['mean_width_ratio']:.4f} over "
        f"{e2b['n_cells']} cell-horizon pairs ({e2b['n_widened']} of them widen, max ratio "
        f"{e2b['max_width_ratio']:.4f}); mean ARW ESS {e2b['mean_ess']:.0f}/{e2b['n_draws']}",
    ))
    out.append((
        "adding TRUE narrative information does not cost coverage on average",
        e2b["mean_cov_narrative"] > e2b["mean_cov_sign"] - 0.02,
        f"mean coverage over all cells/horizons: sign only {e2b['mean_cov_sign']:.4f} -> "
        f"narrative {e2b['mean_cov_narrative']:.4f}",
    ))

    # --- experiment 2c: point-identified, and the prior still shows -------
    hz = min(3, e2c["horizon"])
    out.append((
        "point-identifying zeros: the diffuse prior covers better than the default at h>=1",
        float(np.mean([
            e2c["cov"][5.0][k, 0, 0] - e2c["cov"][0.2][k, 0, 0]
            for k in range(1, e2c["horizon"] + 1)
        ])) > 0.02,
        f"mean own-response gain from lambda1 0.2 -> 5.0 over h=1..{e2c['horizon']}: "
        f"{float(np.mean([e2c['cov'][5.0][k, 0, 0] - e2c['cov'][0.2][k, 0, 0] for k in range(1, e2c['horizon'] + 1)])):+.3f}",
    ))

    # --- experiment 3: monotone degradation, T-invariant width -------------
    cells = sorted(e3["cells"], key=lambda c: -c["delta"])
    big, small = cells[0], [c for c in cells if abs(c["delta"] - 0.5) < 1e-9][0]
    out.append((
        "the break-date CI is conservative at a large break",
        big["cond95"] > 0.97,
        f"break/sigma={big['delta']:.1f}: {big['cond95']:.3f}"
        f"+-{mc_se(big['cond95'], big['n_detected']):.3f} vs nominal 0.95 "
        f"(ceil + 1 padding on each side)",
    ))
    out.append((
        "break-date CI coverage degrades as the break shrinks",
        (big["cond95"] - small["cond95"]) > 0.03,
        f"cond95 falls {big['cond95']:.3f} -> {small['cond95']:.3f} from break/sigma "
        f"{big['delta']:.1f} -> {small['delta']:.2f}",
    ))
    widths = [c["mean_width95"] for c in cells]
    out.append((
        "CI width is monotone decreasing in break magnitude (width ~ c/scale, scale ~ delta^2)",
        all(widths[k] <= widths[k + 1] + 1e-9 for k in range(len(widths) - 1)),
        "mean widths by decreasing break: " + ", ".join(f"{w:.1f}" for w in widths),
    ))
    det = [c["detect_rate"] for c in cells]
    out.append((
        "detection collapses at the smallest break, so unconditional coverage collapses with it",
        det[-1] < 0.6 and cells[-1]["uncond95"] < 0.5,
        f"break/sigma={cells[-1]['delta']:.2f}: detection {det[-1]:.3f}, unconditional 95% "
        f"coverage {cells[-1]['uncond95']:.3f}",
    ))
    tc = {(c["T"], c["delta"]): c for c in e3b["cells"]}
    Ts = sorted({c["T"] for c in e3b["cells"]})
    if len(Ts) >= 3 and all((T, 1.0) in tc for T in Ts) and all((T, 0.5) in tc for T in Ts):
        ws = [tc[(T, 1.0)]["mean_width95"] for T in Ts]
        out.append((
            "the CI width does not shrink with T at fixed break magnitude (Bai's fixed-break "
            "asymptotics: an O(1)-precision parameter)",
            (max(ws) - min(ws)) / np.mean(ws) < 0.10,
            f"mean 95% widths at T={Ts} and break/sigma=1: " + ", ".join(f"{w:.1f}" for w in ws),
        ))
        cs = [tc[(T, 0.5)]["cond95"] for T in Ts]
        out.append((
            "but coverage at a small break does improve with T -- a finite-sample approximation, "
            "not a wrong formula",
            (cs[-1] - cs[0]) > 0.02,
            f"cond95 at break/sigma=0.5: " + " -> ".join(f"{c:.3f}" for c in cs)
            + f" for T={Ts}",
        ))
    return out


# ==========================================================================
def run(quick=False, reps=None):
    started = time.time()
    r_bvar = reps or (REPS_BVAR_QUICK if quick else REPS_BVAR_FULL)
    r_imp = reps or (REPS_IMPACT_QUICK if quick else REPS_IMPACT_FULL)
    r_set = reps or (REPS_SET_QUICK if quick else REPS_SET_FULL)
    r_nar = reps or (REPS_NARR_QUICK if quick else REPS_NARR_FULL)
    r_zer = reps or (REPS_ZERO_QUICK if quick else REPS_ZERO_FULL)
    r_brk = reps or (REPS_BREAK_QUICK if quick else REPS_BREAK_FULL)
    r_brt = reps or (REPS_BREAKT_QUICK if quick else REPS_BREAKT_FULL)
    horizon = 8 if quick else 12
    hset = 4 if quick else 6
    n_draws = 200 if quick else 400
    n_set = 200 if quick else 300
    episodes = 8 if quick else 20
    deltas = (3.0, 1.0, 0.5, 0.25) if quick else (3.0, 2.0, 1.0, 0.5, 0.25)
    Ts = (200, 400, 800)

    print("=" * 104)
    print("BAYESIAN CREDIBLE BANDS AND SET-IDENTIFIED BOUNDS -- WHAT DO THEY ACTUALLY PROMISE?")
    print("=" * 104)
    print(f"master seed   : {SEED}   (every number below is a function of it)")
    print(f"mode          : {'QUICK smoke run' if quick else 'full run'}")
    print(f"replications  : bvar={r_bvar}  impact={r_imp}  set={r_set}  narrative={r_nar}  "
          f"zero={r_zer}  break={r_brk}  break-T={r_brt}")
    print(f"MC se at p=0.90: bvar {100 * mc_se(0.90, r_bvar):.2f}pp, set "
          f"{100 * mc_se(0.90, r_set):.2f}pp, break {100 * mc_se(0.90, r_brk):.2f}pp")
    print("nominal levels: 90% credible bands (5th/95th posterior percentile); "
          "robust_svar_bounds alpha=0.10;")
    print("                bai_perron 90% and 95% break-date CIs")
    print("KINDS         : CRED = Bayesian credible band, no frequentist promise; "
          "SET = identified-set")
    print("                bounds, not an interval about a point; CI = frequentist interval.")

    facts = structural_checks()
    res = {}
    res["exp1"] = exp_bvar_prior_centring(r_bvar, horizon=horizon, n_draws=n_draws)
    report_bvar_prior_centring(
        res["exp1"], show=tuple(h for h in (0, 1, 2, 4, 8, 12) if h <= horizon)
    )
    res["exp1b"] = exp_bvar_impact_vs_exact(r_imp)
    report_bvar_impact_vs_exact(res["exp1b"])
    res["exp2"] = exp_set_coverage(r_set, horizon=hset, n_draws=n_set)
    report_set_coverage(res["exp2"], show=tuple(h for h in (0, 1, 2, 3, 4, 6) if h <= hset))
    res["exp2b"] = exp_narrative(r_nar, horizon=hset, n_draws=n_set, n_episodes=episodes)
    report_narrative(res["exp2b"], show=tuple(h for h in (0, 1, 2, 3, 4, 6) if h <= hset))
    res["exp2c"] = exp_zero_sign(r_zer, horizon=hset, n_draws=n_set)
    report_zero_sign(res["exp2c"], show=tuple(h for h in (0, 1, 2, 3, 4, 6) if h <= hset))
    res["exp3"] = exp_break_date_ci(r_brk, deltas=deltas)
    res["exp3b"] = exp_break_date_T(r_brt, Ts=Ts)
    report_break_date(res["exp3"], res["exp3b"])

    header("FINDINGS -- measured, not targeted")
    for line in findings(res):
        print(textwrap.fill(line, width=102, initial_indent="  - ", subsequent_indent="    "))
        print()

    header("ASSERTIONS -- robust qualitative facts only, never a tuned number")
    checks = assertions(res, facts)
    failed = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(textwrap.fill(f"[{'PASS' if ok else 'FAIL'}] {label}", width=102,
                            initial_indent="  ", subsequent_indent="         "))
        print(textwrap.fill(detail, width=102, initial_indent="         ",
                            subsequent_indent="         "))

    elapsed = time.time() - started
    header(f"{len(checks) - len(failed)}/{len(checks)} assertions passed in {elapsed:.1f}s "
           f"(seed {SEED})")
    if failed:
        raise AssertionError(
            "coverage assertions failed: " + "; ".join(f"{c[0]} ({c[2]})" for c in failed)
        )
    return {"results": res, "facts": facts, "checks": checks, "elapsed": elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--reps", type=int, default=None, help="override every replication count")
    args = parser.parse_args()
    run(quick=args.quick, reps=args.reps)
