"""Interval COVERAGE for the local-projection family.

    .venv/bin/python docs/examples/coverage/lp_family.py [--quick]

A confidence interval is a promise about repeated samples: a nominal 95%
interval should contain the truth in 95% of independent draws from the same
data-generating process. Everywhere else in this repo we check that tsecon's
*point* estimates match an independent reference. This module checks the other
half of the contract -- whether the *intervals* keep their promise -- for
``lp``, ``lp_iv``, ``lp_state``, ``lp_multiplier`` and ``smooth_lp``.

Under-coverage is a finding, not a failure to be tuned away. Every number
printed below is what the simulation measured, with its Monte Carlo standard
error next to it, and the closing notes say plainly where the intervals miss.

--------------------------------------------------------------------------
The known-truth DGP
--------------------------------------------------------------------------
Every experiment draws from a linear model whose true impulse response is
known in closed form, so "coverage" is well defined at each horizon:

    y_t = sum_{j=0}^{J-1} theta_j * s_{t-j} + (nuisance)_t,   theta_j = rho^j

with `s_t` i.i.d. standard normal and the nuisance term built only from
variables dated t or earlier that are *independent of s*. Because `s_t` is
orthogonal in population to everything else in the horizon-h projection
(the constant and the lags y_{t-1},...,y_{t-p}), the population local
projection coefficient on `s_t` is exactly theta_h -- for ANY number of lag
controls, and whatever serial correlation the nuisance term has. There is no
approximation in the truth, so any miss is the interval's.

Note what this DGP does *not* do: y is not an AR(p), so p lag controls cannot
soak up its dynamics, and the horizon-h projection residual therefore contains
the past shocks s_{t-1}, s_{t-2}, ... . That is what makes the score
serially correlated in the plain-HAC specification and what lag augmentation
removes -- see the next block.

--------------------------------------------------------------------------
The two claims under test
--------------------------------------------------------------------------
(a) tsecon's DEFAULT for `lp` is se="lag_augmented": the horizon-h regression
    is augmented with the impulse's own lags s_{t-1},...,s_{t-h} and takes HC1
    (Eicker-Huber-White) standard errors, following Montiel Olea &
    Plagborg-Moller (2021). The argument for why this beats HAC is explicit
    in this DGP. Write the horizon-h score as x_t * u_{t,h} with x_t the
    residualised impulse; to leading order (treating x_t as s_t itself) its
    lag-k autocovariance contains the term
    theta_{h+k} * theta_{h-k} * E[s_t^2] E[s_{t-k}^2], which is nonzero for
    1 <= k <= min(h, J-1-h): the plain-HAC score really is serially
    correlated, and increasingly so as the horizon grows. Lag augmentation
    projects s_{t-1},...,s_{t-h} out of the regression, which kills exactly
    those overlap terms and leaves a serially uncorrelated score -- so HC1 is
    the right variance and no bandwidth has to be chosen. Whether that
    asymptotic argument buys real coverage at T = 200 is Experiment 1.

(b) `smooth_lp` reports standard errors CONDITIONAL on the smoothing parameter
    lambda, and its own docstring says so ("`se` conditions on `lam` and does
    not account for shrinkage bias"). Two separate costs hide in that
    sentence: the penalty biases the estimate toward a straight line, and
    cross-validating lambda adds sampling variability the reported SE never
    sees. Experiment 6 separates them by reporting bias, the Monte Carlo
    standard deviation of the estimate, and the mean reported SE side by side.

--------------------------------------------------------------------------
How to read the tables
--------------------------------------------------------------------------
    truth    the closed-form population value at that horizon
    bias     mean(estimate) - truth, over replications
    sd_est   Monte Carlo standard deviation of the estimate: the truth about
             how variable the estimator actually is
    mean_se  the average standard error the library reported
    med_se   its median. Watch for mean_se >> med_se: that is a standard error
             with no finite moment, which happens in the weak-instrument arm
             and means the mean is describing a handful of draws, not a
             typical interval. In every other arm the two agree.
    se/sd    mean_se / sd_est. This is the single most diagnostic column:
             ~1.0 means the reported SE is the right size, < 1.0 means the
             library is understating its own sampling variability.
    |b|/sd   |bias| / sd_est. Above ~0.3 a centred interval starts losing
             coverage from being off-centre, however good the SE is.
    cov95    fraction of replications whose 95% interval covered the truth
    mcse     Monte Carlo standard error of cov95, sqrt(p(1-p)/reps). Two
             coverage numbers less than ~2 mcse apart are not distinguishable.

Reading se/sd against |b|/sd is what separates "the standard error is wrong"
from "the estimator is off-centre and no standard error can fix it". The
closing table does that arithmetic explicitly: it converts each arm's se/sd
and bias/sd into the coverage a normal estimator with those two numbers would
have had, and reports what is left over -- the part explained by neither, i.e.
the randomness of the reported standard error itself and any non-normality of
the sampling distribution.

Total runtime is a little over a minute; --quick cuts the replication count by
8 for a smoke run (and its Monte Carlo standard errors are correspondingly ~3x
larger, so do not read --quick numbers as measurements).
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
Z95 = float(norm.ppf(0.975))  # 1.959964...
NOMINAL = 0.95

RHO = 0.7          # true IRF decay: theta_h = RHO**h
J = 25             # MA truncation; RHO**24 = 1.9e-4, and the truth is the
                   # truncated sum exactly, so nothing is approximated
THETA = RHO ** np.arange(J)


def _rng(experiment, rep):
    """A reproducible, independent stream per (experiment, replication).

    Seeding on the tuple rather than advancing one global generator means each
    replication's data depend only on its own coordinates: experiments can be
    run, skipped or reordered and every number stays the same.
    """
    return np.random.default_rng([SEED, experiment, rep])


# --------------------------------------------------------------------------
# data-generating processes
# --------------------------------------------------------------------------
def dgp_ma(rng, T, sig_eta=0.5, het=False):
    """The baseline: y_t = sum_j THETA_j s_{t-j} + eta_t, s i.i.d. N(0,1).

    `eta` is independent of `s`, so the population LP coefficient on s_t is
    exactly THETA[h] at every horizon h. With `het=True` the eta variance
    scales with |s_{t-1}|, i.e. heteroskedasticity that is predetermined with
    respect to the impulse -- it changes the efficient variance but not the
    truth.

    Returns (y, s) both of length T.
    """
    s = rng.standard_normal(T + J)
    # y_t = sum_j THETA_j s_{t-j}: a convolution, taken over the T base times
    # whose full history of J lags is available.
    ma = np.convolve(s, THETA)[J:J + T]
    scale = sig_eta * (1.0 + np.abs(s[J - 1:J + T - 1])) if het else sig_eta
    return ma + scale * rng.standard_normal(T), s[J:J + T]


def dgp_iv(rng, T, phi, sig_nu, delta=0.6, sig_eta=0.5):
    """An endogenous impulse with a valid instrument of tunable strength.

        s_t, om_t, nu_t, eta_t   i.i.d. N(0,1), mutually independent
        x_t = s_t + om_t                          (observed, endogenous)
        z_t = phi * s_t + sig_nu * nu_t           (instrument)
        y_t = sum_j THETA_j s_{t-j} + delta * om_t + sig_eta * eta_t

    `om_t` enters both x and y, so OLS on x is biased; z is correlated with
    the structural s and with nothing else. The population LP-IV coefficient
    is Cov(y_{t+h}, z_t) / Cov(x_t, z_t) = (THETA[h] * phi) / phi = THETA[h],
    exactly, and z_t is orthogonal to the lag controls. `phi` alone moves the
    instrument's strength without touching the truth -- which is the point.
    """
    s = rng.standard_normal(T + J)
    om = rng.standard_normal(T)
    sobs = s[J:J + T]
    y = (np.convolve(s, THETA)[J:J + T] + delta * om
         + sig_eta * rng.standard_normal(T))
    x = sobs + om
    z = phi * sobs + sig_nu * rng.standard_normal(T)
    return y, x, z


def dgp_multiplier(rng, T, rho_x=0.8, phi=1.0, sig_nu=0.5, delta=0.6,
                   sig_eta=0.5):
    """As `dgp_iv`, but the endogenous impulse is persistent:

        x_t = rho_x * x_{t-1} + s_t + om_t

    so accumulating x over a window is dominated by signal rather than by
    noise -- the realistic case for a spending or tax series, and the case in
    which the Ramey-Zubairy integral multiplier is worth estimating.

    The population one-step LP-IV multiplier at horizon h is
        m_h = Cov(sum_{j<=h} y_{t+j}, z_t) / Cov(sum_{j<=h} x_{t+j}, z_t)
            = (sum_{j<=h} THETA_j) / (sum_{j<=h} rho_x**j),
    since Cov(y_{t+j}, z_t) = THETA_j * phi and Cov(x_{t+j}, z_t)
    = rho_x**j * phi. phi cancels; see `true_multiplier`.
    """
    s = rng.standard_normal(T + J)
    om = rng.standard_normal(T + J)
    drv = s + om
    x = np.empty(T + J)
    x[0] = drv[0]
    for t in range(1, T + J):
        x[t] = rho_x * x[t - 1] + drv[t]
    sobs = s[J:J + T]
    y = (np.convolve(s, THETA)[J:J + T] + delta * om[J:J + T]
         + sig_eta * rng.standard_normal(T))
    z = phi * sobs + sig_nu * rng.standard_normal(T)
    return y, x[J:J + T], z


def true_multiplier(horizons, rho_x=0.8):
    """Closed-form integral multiplier for `dgp_multiplier`."""
    h = np.arange(horizons + 1)
    return np.cumsum(THETA[:horizons + 1]) / np.cumsum(rho_x ** h)


def dgp_state(rng, T, theta1, theta0, p_stay=0.9, sig_eta=0.5):
    """State-dependent propagation with a persistent, predetermined regime.

        I_t   two-state Markov chain, P(stay) = p_stay, marginal 1/2
        y_t = sum_j theta^{(I_{t-j-1})}_j s_{t-j} + eta_t

    The regime that governs the propagation of s_{t-j} is the one prevailing
    at t-j-1 -- lagged, so it is predetermined with respect to the shock,
    exactly as `lp_state` assumes. Writing a_t = I_{t-1} s_t and
    b_t = (1 - I_{t-1}) s_t turns the sum into two convolutions.

    Because I_{t-1} is in the time-(t-1) information set and s_t is i.i.d.,
    both interacted impulse columns are orthogonal in population to every
    control and to each other, so the population regime-1 and regime-0
    coefficients are exactly theta1[h] and theta0[h].
    """
    s = rng.standard_normal(T + J)
    flips = rng.random(T + J) >= p_stay
    start = int(rng.random() < 0.5)
    ind = (start + np.cumsum(flips)) % 2          # the Markov path, vectorised
    lag_ind = np.empty(T + J)
    lag_ind[0] = ind[0]                            # unused: t=0 is presample
    lag_ind[1:] = ind[:-1]
    a = lag_ind * s
    b = (1.0 - lag_ind) * s
    y = (np.convolve(a, theta1)[J:J + T] + np.convolve(b, theta0)[J:J + T]
         + sig_eta * rng.standard_normal(T))
    return y, s[J:J + T], ind[J:J + T].astype(float)


# --------------------------------------------------------------------------
# coverage bookkeeping
# --------------------------------------------------------------------------
def summarize(est, se, truth, arm, extra=None):
    """Per-horizon coverage summary from (reps x H+1) estimate and SE arrays.

    Replications in which the library raised, or returned a non-finite
    estimate or SE, are dropped from that horizon and counted in `n_used`;
    they are never silently treated as covering.
    """
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
            # The median SE matters when the mean is not summarising anything:
            # a just-identified 2SLS SE has no finite moments under weak
            # identification, so mean_se is driven by a handful of draws.
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
    ("arm", "arm", 21, "{:s}"),
    ("h", "h", 3, "{:d}"),
    ("truth", "truth", 8, "{:.4f}"),
    # .4g rather than .4f: the weak-instrument arm produces standard errors in
    # the thousands and a fixed-decimal format would silently break alignment.
    ("bias", "bias", 9, "{:+.4g}"),
    ("sd_est", "sd_est", 9, "{:.4g}"),
    ("mean_se", "mean_se", 9, "{:.4g}"),
    ("med_se", "med_se", 9, "{:.4g}"),
    ("se_over_sd", "se/sd", 6, "{:.2f}"),
    ("absbias_over_sd", "|b|/sd", 7, "{:.2f}"),
    ("cov95", "cov95", 7, "{:.3f}"),
    ("mcse", "mcse", 6, "{:.3f}"),
]


def print_table(rows, spec=SPEC, extra_cols=()):
    """Aligned fixed-width table; `extra_cols` are appended (key, hdr, w, fmt)."""
    spec = list(spec) + list(extra_cols)
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


def print_paired(res, a="lag_augmented", b="hac"):
    """The paired coverage difference, horizon by horizon.

    Both arms saw the same draws, so `diff` is a within-draw difference and
    `se_diff` is its Monte Carlo standard error -- much smaller than the two
    arms' individual mcse, which is what makes a 2-point gap readable.
    """
    print()
    print(f"paired coverage difference, {a} minus {b} (same draws):")
    head = f"{'h':>3}  {'diff':>8}  {'se_diff':>8}  {'diff/se':>8}"
    print(head)
    print("-" * len(head))
    for p in res["paired"]:
        ratio = p["diff"] / p["se_diff"] if p["se_diff"] > 0 else float("nan")
        print(f"{p['h']:>3}  {p['diff']:>+8.4f}  {p['se_diff']:>8.4f}  "
              f"{ratio:>8.2f}")
    pool = res["paired_pooled"]
    print(f"pooled over h >= {pool['h_min']} (per-draw average, so "
          f"cross-horizon correlation is handled): "
          f"{pool['diff']:+.4f} (se {pool['se_diff']:.4f})")


def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def collect(reps, draw, fit, nh, aux_keys=()):
    """Run `reps` replications of draw -> fit.

    `fit` returns (point_array, se_array) or (point, se, aux_dict). Returns
    (est, se, aux, n_failed) with every array indexed by the replication
    number, so a replication that raised leaves NaN in the same row of every
    array and nothing can drift out of alignment.

    Library exceptions are caught and counted rather than aborting the run: a
    weak instrument really can produce a singular projected design, and how
    often that happens is part of the answer.
    """
    est = np.full((reps, nh), np.nan)
    se = np.full((reps, nh), np.nan)
    aux = {k: np.full((reps, nh), np.nan) for k in aux_keys}
    failed = 0
    for r in range(reps):
        try:
            out = fit(*draw(r))
        except Exception:                        # noqa: BLE001 - counted below
            failed += 1
            continue
        if out is None:
            failed += 1
            continue
        est[r], se[r] = out[0], out[1]
        if len(out) > 2:
            for k, v in out[2].items():
                aux[k][r] = v
    return est, se, aux, failed


# ==========================================================================
# Experiment 1 -- the library's default vs the alternative it replaced
# ==========================================================================
def exp_lag_augmented_vs_hac(reps, T=200, horizons=12, n_lag_controls=4,
                             het=False, experiment=1):
    """Claim (a): does se="lag_augmented" cover better than se="hac"?

    Same draws for both arms, so the comparison is paired and the difference
    is not a lucky sample. `het=True` repeats it under heteroskedasticity
    that is predetermined w.r.t. the shock (both SEs claim robustness to it).
    """
    truth = THETA[:horizons + 1]
    rows = []
    covered = {}
    for arm, kw in (("lag_augmented", {"se": "lag_augmented"}),
                    ("hac", {"se": "hac"})):
        def draw(r, _het=het):
            return dgp_ma(_rng(experiment + (100 if _het else 0), r), T,
                          het=_het)

        def fit(y, s, _kw=kw):
            out = tsecon.lp(y, s, horizons=horizons,
                            n_lag_controls=n_lag_controls, **_kw)
            return out["irf"], out["se"]

        est, se, _, failed = collect(reps, draw, fit, horizons + 1)
        rows += summarize(est, se, truth, arm)
        assert failed == 0, f"lp raised on {failed} draws (arm={arm})"
        covered[arm] = (np.abs(est - truth) <= Z95 * se)

    # Both arms saw the SAME draws, so the coverage difference is a paired
    # comparison and its Monte Carlo standard error is the standard error of
    # the within-draw difference -- far smaller than sqrt(p(1-p)/reps) for each
    # arm separately, because the draws' luck cancels. This is what licenses a
    # statement about a 2-point gap from a few thousand replications.
    d = covered["lag_augmented"].astype(float) - covered["hac"].astype(float)
    paired = [{"h": h, "diff": float(d[:, h].mean()),
               "se_diff": float(d[:, h].std(ddof=1) / np.sqrt(d.shape[0]))}
              for h in range(horizons + 1)]
    # Pooled over the long horizons. Horizons within a draw are strongly
    # correlated, so the pooled standard error is computed from the per-draw
    # AVERAGE difference -- not by dividing a per-horizon standard error by
    # sqrt(number of horizons), which would assume independence and understate
    # it.
    h_long = max(1, horizons // 2)
    dbar = d[:, h_long:].mean(axis=1)
    pooled = {"h_min": h_long, "diff": float(dbar.mean()),
              "se_diff": float(dbar.std(ddof=1) / np.sqrt(dbar.size))}
    return {
        "name": "lp: lag-augmented (default) vs HAC"
                + (" [heteroskedastic]" if het else ""),
        "meta": {"T": T, "horizons": horizons, "n_lag_controls": n_lag_controls,
                 "reps": reps, "het": het, "nominal": NOMINAL},
        "rows": rows,
        "paired": paired,
        "paired_pooled": pooled,
    }


# ==========================================================================
# Experiment 2 -- is the gap finite-sample, or is it the estimator?
# ==========================================================================
def exp_sample_size(reps, sizes=(100, 200, 400, 800), horizons=12,
                    n_lag_controls=4, report=(0, 4, 8, 12), experiment=2):
    """Coverage at fixed horizons as T grows.

    A gap that closes as T grows is the ASYMPTOTIC APPROXIMATION being what it
    is. A gap that does not close points at the estimator or the SE formula.
    """
    truth = THETA[:horizons + 1]
    rows = []
    for i, T in enumerate(sizes):
        for arm, kw in (("lag_augmented", {"se": "lag_augmented"}),
                        ("hac", {"se": "hac"})):
            def draw(r, _T=T, _i=i):
                return dgp_ma(_rng(experiment * 10 + _i, r), _T)

            def fit(y, s, _kw=kw):
                out = tsecon.lp(y, s, horizons=horizons,
                                n_lag_controls=n_lag_controls, **_kw)
                return out["irf"], out["se"]

            est, se, _, failed = collect(reps, draw, fit, horizons + 1)
            assert failed == 0, f"lp raised on {failed} draws (T={T})"
            for row in summarize(est, se, truth, f"T={T} {arm}"):
                if row["h"] in report:
                    rows.append(row)
    return {
        "name": "lp: coverage as the sample grows (does the gap close?)",
        "meta": {"sizes": list(sizes), "horizons": horizons,
                 "n_lag_controls": n_lag_controls, "reps": reps,
                 "nominal": NOMINAL},
        "rows": rows,
    }


# ==========================================================================
# Experiment 3 -- LP-IV, strong instrument and weak instrument
# ==========================================================================
def exp_lp_iv(reps, T=200, horizons=8, n_lag_controls=4, experiment=3):
    """LP-IV coverage with a strong and then a deliberately weak instrument.

    The truth is THETA[h] in both arms -- only `phi` changes -- so the two
    rows are directly comparable, and a collapse in the weak arm is entirely
    an inference failure and not a change in the estimand. This is the honest
    warning: the weak-instrument 2SLS sampling distribution is not normal, so
    a normal interval around a point estimate cannot cover at its nominal
    rate no matter how good the standard error is.
    """
    truth = THETA[:horizons + 1]
    rows = []
    fstats = {}
    arms = (("strong iv", 1.0, 0.5), ("weak iv", 0.2, 1.0))
    for i, (arm, phi, sig_nu) in enumerate(arms):
        def draw(r, _i=i, _phi=phi, _snu=sig_nu):
            return dgp_iv(_rng(experiment * 10 + _i, r), T, _phi, _snu)

        def fit(y, x, z):
            out = tsecon.lp_iv(y, x, z, horizons=horizons,
                               n_lag_controls=n_lag_controls)
            return out["irf"], out["se"], {"f": out["first_stage_f"]}

        est, se, aux, failed = collect(reps, draw, fit, horizons + 1,
                                       aux_keys=("f",))
        med_f = np.nanmedian(aux["f"], axis=0)
        rows += summarize(est, se, truth, arm, extra={"median_f": med_f})
        fstats[arm] = med_f
        if failed:
            print(f"  note: lp_iv raised on {failed}/{reps} draws in "
                  f"the '{arm}' arm")
    return {
        "name": "lp_iv: strong vs weak instrument",
        "meta": {"T": T, "horizons": horizons,
                 "n_lag_controls": n_lag_controls, "reps": reps,
                 "arms": {a: {"phi": p, "sig_nu": s} for a, p, s in arms},
                 "nominal": NOMINAL},
        "rows": rows,
        "median_first_stage_f": {k: v.tolist() for k, v in fstats.items()},
        "extra_cols": (("median_f", "med_F", 8, "{:.1f}"),),
    }


# ==========================================================================
# Experiment 4 -- state-dependent LP, per regime
# ==========================================================================
def exp_lp_state(reps, T=300, horizons=8, n_lag_controls=2, p_stay=0.9,
                 experiment=4):
    """Per-regime coverage when the regime is persistent.

    A persistent regime is the empirically relevant case (recessions come in
    runs) and it is the hard case: each regime's coefficient is identified off
    roughly half the sample, and the interacted design doubles the parameter
    count, so the effective sample per regime is small.
    """
    theta1 = 1.2 * 0.85 ** np.arange(J)
    theta0 = 0.4 * 0.50 ** np.arange(J)
    rows = []
    for arm, kw in (("lag_augmented", {"se": "lag_augmented"}),
                    ("hac", {"se": "hac"})):
        def draw(r):
            return dgp_state(_rng(experiment, r), T, theta1, theta0,
                             p_stay=p_stay)

        for regime, key_i, key_s, tr in (
                ("state1", "irf_state1", "se_state1", theta1),
                ("state0", "irf_state0", "se_state0", theta0)):
            def fit(y, s, ind, _kw=kw, _ki=key_i, _ks=key_s):
                out = tsecon.lp_state(y, s, ind, horizons=horizons,
                                      n_lag_controls=n_lag_controls, **_kw)
                return out[_ki], out[_ks]

            est, se, _, failed = collect(reps, draw, fit, horizons + 1)
            rows += summarize(est, se, tr[:horizons + 1],
                              f"{regime} {arm}")
            if failed:
                print(f"  note: lp_state raised on {failed}/{reps} draws "
                      f"({regime} {arm})")
    return {
        "name": "lp_state: per-regime coverage, persistent regime",
        "meta": {"T": T, "horizons": horizons,
                 "n_lag_controls": n_lag_controls, "p_stay": p_stay,
                 "reps": reps, "nominal": NOMINAL},
        "rows": rows,
    }


# ==========================================================================
# Experiment 5 -- the Ramey-Zubairy integral multiplier
# ==========================================================================
def exp_lp_multiplier(reps, T=240, horizons=8, n_lag_controls=4,
                      experiment=5):
    """Coverage of the integral multiplier, horizon by horizon.

    The multiplier is a single 2SLS coefficient (not a delta-method ratio), so
    its SE is an honest SE of the reported parameter. But the identifying
    covariation is the CONTEMPORANEOUS instrument against a CUMULATED
    impulse, so how strong the first stage stays as the window widens is a
    property of the impulse process, not of the code. The median first-stage
    F is reported next to the coverage so the two can be read together.
    """
    truth = true_multiplier(horizons)

    def draw(r):
        return dgp_multiplier(_rng(experiment, r), T)

    def fit(y, x, z):
        out = tsecon.lp_multiplier(y, x, z, horizons=horizons,
                                   n_lag_controls=n_lag_controls)
        return out["multiplier"], out["se"], {"f": out["first_stage_f"]}

    est, se, aux, failed = collect(reps, draw, fit, horizons + 1,
                                   aux_keys=("f",))
    rows = summarize(est, se, truth, "multiplier",
                     extra={"median_f": np.nanmedian(aux["f"], axis=0)})
    if failed:
        print(f"  note: lp_multiplier raised on {failed}/{reps} draws")
    return {
        "name": "lp_multiplier: integral-multiplier coverage",
        "meta": {"T": T, "horizons": horizons,
                 "n_lag_controls": n_lag_controls, "reps": reps,
                 "rho_x": 0.8, "nominal": NOMINAL},
        "rows": rows,
        "extra_cols": (("median_f", "med_F", 8, "{:.1f}"),),
    }


# ==========================================================================
# Experiment 6 -- what conditioning on lambda costs smooth_lp
# ==========================================================================
def exp_smooth_lp(reps, T=200, horizons=12, n_lag_controls=4, experiment=6):
    """Claim (b): price the two costs the model card admits to.

    Three arms on the SAME draws:
      lam=0        no penalty: the stacked-LP HAC anchor. Any miss here is the
                   HAC approximation, not smoothing.
      lam=cv       the default: lambda chosen by leave-h-block-out CV, then
                   the SE computed as if that lambda had been given.
      lam=100      a deliberately heavy fixed penalty, to show the shrinkage
                   bias mechanism cleanly at a lambda nobody chose from data.

    The true IRF here is rho^h, which is curved, and the penalty
    (penalty_order=2) shrinks toward a straight line. So bias is expected;
    the question is how big it is relative to sd_est, and whether the reported
    SE at least gets the variance right. Read |b|/sd and se/sd together:
    |b|/sd large with se/sd near 1 is shrinkage bias; se/sd well below 1 is
    the SE understating variability, which is the specific cost of treating a
    cross-validated lambda as fixed.
    """
    truth = THETA[:horizons + 1]
    rows = []
    lambdas = []
    arms = (("lam=0", 0.0), ("lam=cv", "cv"), ("lam=100", 100.0))
    for arm, lam in arms:
        def draw(r):
            return dgp_ma(_rng(experiment, r), T)

        def fit(y, s, _lam=lam):
            out = tsecon.smooth_lp(y, s, horizons=horizons,
                                   n_lag_controls=n_lag_controls, lam=_lam)
            lam_col = np.full(horizons + 1, out["lambda_used"])
            return out["irf"], out["se"], {"lam": lam_col}

        est, se, aux, failed = collect(reps, draw, fit, horizons + 1,
                                       aux_keys=("lam",))
        rows += summarize(est, se, truth, arm)
        lam_used = aux["lam"][:, 0]
        lambdas.append((arm, float(np.nanmedian(lam_used)),
                        float(np.nanmin(lam_used)), float(np.nanmax(lam_used))))
        if failed:
            print(f"  note: smooth_lp raised on {failed}/{reps} draws ({arm})")
    return {
        "name": "smooth_lp: the cost of conditioning on lambda",
        "meta": {"T": T, "horizons": horizons,
                 "n_lag_controls": n_lag_controls, "reps": reps,
                 "nominal": NOMINAL},
        "rows": rows,
        "lambda_used": lambdas,
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


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def check(results, quick):
    """Assert the robust qualitative facts; print each check and its numbers.

    What is asserted here is deliberately limited to statements that are
    either guaranteed by the DGP's algebra or hold with a margin of many Monte
    Carlo standard errors. In particular the following are NOT asserted, even
    though the tables report them, because they are the measurements this
    module exists to publish and pinning them to a number would be tuning the
    experiment to its answer:

      * that lp(se="hac") ever reaches nominal coverage,
      * that lp_iv, lp_multiplier or smooth_lp reach nominal coverage,
      * any specific coverage LEVEL at a long horizon.

    Where a floor is asserted (e.g. "no arm falls below 0.85") it is a
    guard against a future regression turning a 4-point miss into a 40-point
    one, chosen far from the measured value, not a claim that 0.85 is good.
    """
    checks = []

    def ok(label, passed, detail):
        checks.append((label, bool(passed), detail))

    # ---- Experiment 1: lp's default ------------------------------------
    e1 = results["lag_augmented_vs_hac"]
    la0 = _rows_by(e1, "lag_augmented", 0)[0]
    # At impact there is no lag augmentation to do (h = 0 adds no impulse
    # lags), the score is a martingale difference by construction, and HC1 is
    # the textbook variance. Coverage must be calibrated -- asserted to within
    # 2 points, which allows for HC1's known mild downward bias and the use of
    # a normal rather than t critical value at ~190 residual degrees of
    # freedom, both of which push coverage slightly below nominal.
    ok("lp default is calibrated on impact (within 2 points of nominal)",
       abs(la0["cov95"] - NOMINAL) <= 0.02,
       f"cov95={la0['cov95']:.3f} vs {NOMINAL:.2f}")
    ok("lp default SE is correctly scaled on impact (se/sd in [0.85, 1.15])",
       0.85 <= la0["se_over_sd"] <= 1.15,
       f"se/sd={la0['se_over_sd']:.3f}")
    # Claim (a). The comparison is PAIRED (both arms see the same draws), so
    # the relevant standard error is that of the within-draw difference.
    paired = {p["h"]: p for p in e1["paired"]}
    pool = e1["paired_pooled"]
    hmin = pool["h_min"]
    hs = [h for h in paired if h >= hmin]
    ok(f"claim (a): lag-augmented covers better than HAC at horizons >= {hmin}",
       pool["diff"] > 3.0 * pool["se_diff"],
       f"pooled paired gap {pool['diff']:+.4f}, "
       f"3*se_diff={3 * pool['se_diff']:.4f}")
    ok(f"claim (a): the gap has the same sign at EVERY horizon >= {hmin}",
       all(paired[h]["diff"] > 0 for h in hs),
       "; ".join(f"h{h}:{paired[h]['diff']:+.3f}" for h in hs))
    # The mechanism, not just the outcome: HAC's SE is the part that is small.
    sd_hac = _mean([r for r in _rows_by(e1, "hac") if r["h"] >= hmin],
                   "se_over_sd")
    sd_la = _mean([r for r in _rows_by(e1, "lag_augmented")
                   if r["h"] >= hmin], "se_over_sd")
    ok("claim (a) mechanism: it is the SE, not the point estimate (se/sd)",
       sd_la > sd_hac,
       f"mean se/sd at h>={hmin}: lag_augmented {sd_la:.3f} vs "
       f"hac {sd_hac:.3f}")

    # ---- Experiment 2: is it finite-sample? -----------------------------
    e2 = results["sample_size"]
    sizes = e2["meta"]["sizes"]
    hmax = e2["meta"]["horizons"]
    hac_small = _rows_by(e2, f"T={sizes[0]} hac", hmax)[0]
    hac_big = _rows_by(e2, f"T={sizes[-1]} hac", hmax)[0]
    ok("HAC's long-horizon under-coverage shrinks as T grows",
       hac_big["cov95"] > hac_small["cov95"],
       f"h={hmax}: cov95 {hac_small['cov95']:.3f} (T={sizes[0]}) -> "
       f"{hac_big['cov95']:.3f} (T={sizes[-1]})")
    ok("... and so does the SE's shortfall (se/sd rises toward 1)",
       hac_big["se_over_sd"] > hac_small["se_over_sd"],
       f"h={hmax}: se/sd {hac_small['se_over_sd']:.3f} -> "
       f"{hac_big['se_over_sd']:.3f}")

    # ---- Experiment 3: LP-IV -------------------------------------------
    e3 = results["lp_iv"]
    s0 = _rows_by(e3, "strong iv", 0)[0]
    w0 = _rows_by(e3, "weak iv", 0)[0]
    strong = _rows_by(e3, "strong iv")
    weak = _rows_by(e3, "weak iv")
    # The design did what it was supposed to do: strength differs, truth
    # does not. Asserted so that a reader can trust the contrast.
    ok("LP-IV design check: the weak arm really is weak (median F < 10 < "
       "strong arm)",
       max(r["median_f"] for r in weak) < 10.0
       < min(r["median_f"] for r in strong),
       f"median F: weak max {max(r['median_f'] for r in weak):.1f}, "
       f"strong min {min(r['median_f'] for r in strong):.1f}")
    # The measured direction is OVER-coverage plus a length explosion, not the
    # collapse one might expect. Assert the length explosion (robust, it is a
    # first-stage identity) and that coverage is NOT at its nominal level.
    ok("weak-instrument LP-IV intervals explode in width (median SE ratio > 3)",
       w0["med_se"] / s0["med_se"] > 3.0,
       f"impact median SE {w0['med_se']:.3g} (weak) vs "
       f"{s0['med_se']:.3g} (strong), ratio "
       f"{w0['med_se'] / s0['med_se']:.1f}x")
    ok("weak-instrument LP-IV is not a 95% interval (off nominal by > 4 mcse)",
       abs(w0["cov95"] - NOMINAL) > 4.0 * w0["mcse"],
       f"impact cov95={w0['cov95']:.3f} "
       f"({'over' if w0['cov95'] > NOMINAL else 'under'}-covers), "
       f"4*mcse={4 * w0['mcse']:.3f}")
    ok("strong-instrument LP-IV stays above a 0.85 coverage floor",
       min(r["cov95"] for r in strong) >= 0.85,
       f"worst horizon cov95={min(r['cov95'] for r in strong):.3f}")

    # ---- Experiment 5: the multiplier ----------------------------------
    e5 = results["lp_multiplier"]
    mult = _rows_by(e5, "multiplier")
    ok("lp_multiplier is centred (|bias|/sd < 0.15 at every horizon)",
       max(r["absbias_over_sd"] for r in mult) < 0.15,
       f"worst |b|/sd={max(r['absbias_over_sd'] for r in mult):.3f}")
    ok("lp_multiplier's miss is therefore the SE, not weak instruments "
       "(median F > 10 throughout)",
       min(r["median_f"] for r in mult) > 10.0,
       f"min median F={min(r['median_f'] for r in mult):.1f}, "
       f"mean se/sd={_mean(mult, 'se_over_sd'):.3f}")

    # ---- Experiment 6: smooth_lp ---------------------------------------
    e6 = results["smooth_lp"]
    a0 = _rows_by(e6, "lam=0")
    acv = _rows_by(e6, "lam=cv")
    cv0 = _rows_by(e6, "lam=cv", 0)[0]
    an0 = _rows_by(e6, "lam=0", 0)[0]
    # Claim (b), part 1: shrinkage moves the centre. At impact the true IRF
    # peaks, so a smoothness penalty must pull the estimate down; that is the
    # penalty working as designed, and it is also what costs coverage.
    ok("claim (b): smooth_lp's default (lam=cv) is off-centre at impact",
       cv0["absbias_over_sd"] > 3.0 * an0["absbias_over_sd"]
       and cv0["absbias_over_sd"] > 0.5,
       f"|b|/sd at h=0: {an0['absbias_over_sd']:.2f} (lam=0) -> "
       f"{cv0['absbias_over_sd']:.2f} (lam=cv)")
    ok("claim (b): and it therefore covers far worse at impact",
       cv0["cov95"] < an0["cov95"] - 0.10,
       f"cov95 at h=0: {an0['cov95']:.3f} (lam=0) -> "
       f"{cv0['cov95']:.3f} (lam=cv)")
    # Claim (b), part 2: the SE conditions on lambda, so it cannot see the
    # variability that choosing lambda adds. se/sd must therefore be smaller
    # for the cross-validated arm than for the fixed-lambda anchor.
    ok("claim (b): conditioning on a CV-chosen lambda understates the SE",
       _mean(acv, "se_over_sd") < _mean(a0, "se_over_sd"),
       f"mean se/sd {_mean(a0, 'se_over_sd'):.3f} (lam=0) -> "
       f"{_mean(acv, 'se_over_sd'):.3f} (lam=cv)")

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
        print("--quick: Monte Carlo standard errors are ~3x the default run's,")
        print("so treat a near-miss as noise and re-run without --quick.")
    if failures:
        raise AssertionError("coverage assertions failed: "
                             + "; ".join(failures))
    return checks


# ==========================================================================
# a mechanical attribution pass over every row that was measured
# ==========================================================================
def _normal_coverage(r, b, z=Z95):
    """Coverage a normal estimator would have with SE ratio `r`, bias ratio `b`.

    If the estimate is theta + bias + sd*u with u standard normal and the
    reported standard error is (about) mean_se, then
        P(|estimate - theta| <= z*mean_se) = Phi(z*r - b) - Phi(-z*r - b)
    with r = mean_se/sd and b = bias/sd. Setting r = 1 isolates what the
    off-centring alone costs; setting b = 0 isolates what the SE's scaling
    alone costs. Whatever the two together do not explain is left over for the
    things this formula ignores: that the reported SE is itself random
    (which lowers coverage), and that the estimator's sampling distribution
    may not be normal at all (the weak-instrument case).
    """
    return float(norm.cdf(z * r - b) - norm.cdf(-z * r - b))


def decompose(row, nominal=NOMINAL):
    """Split a row's coverage miss into SE-scaling, off-centring and residual.

    Returns (predicted, d_se, d_bias, d_other, verdict). The three deltas are
    signed contributions in coverage points and are additive only to first
    order -- they are an attribution, not an identity.
    """
    r = row["se_over_sd"]
    b = row["bias"] / row["sd_est"] if row["sd_est"] > 0 else 0.0
    pred = _normal_coverage(r, b)
    d_se = _normal_coverage(r, 0.0) - nominal
    d_bias = _normal_coverage(1.0, b) - nominal
    d_other = row["cov95"] - pred
    parts = {"SE scaling": d_se, "off-centre": d_bias,
             "SE variability / non-normality": d_other}
    if abs(row["cov95"] - nominal) <= 3.0 * row["mcse"]:
        label = "at nominal"
    else:
        worst = min(parts, key=lambda k: parts[k])
        direction = "over" if row["cov95"] > nominal else "under"
        if parts[worst] > 0:                 # nothing is losing coverage
            best = max(parts, key=lambda k: parts[k])
            label = f"{direction}-covers, driven by {best}"
        else:
            label = f"{direction}-covers, mostly {worst}"
    return pred, d_se, d_bias, d_other, label


def print_misses(results, nominal=NOMINAL):
    """The worst-covering horizon of every arm, with the miss decomposed."""
    print()
    print("=" * 100)
    print("WORST HORIZON PER ARM, WITH THE MISS DECOMPOSED")
    print("=" * 100)
    print("pred  = coverage a normal estimator with this arm's se/sd and")
    print("        bias/sd would have had.")
    print("d_se  = coverage points lost (or gained) to the SE's scaling alone.")
    print("d_bias= coverage points lost to off-centring alone.")
    print("d_oth = cov95 - pred: what neither ratio explains, i.e. the")
    print("        randomness of the reported SE and any non-normality.")
    print()
    head = (f"{'experiment':<26}  {'arm':<21}  {'h':>3}  {'cov95':>7}  "
            f"{'pred':>6}  {'d_se':>7}  {'d_bias':>7}  {'d_oth':>7}  "
            f"attribution")
    print(head)
    print("-" * len(head))
    for key, res in results.items():
        if key.startswith("_") or "rows" not in res:
            continue
        arms = []
        for r in res["rows"]:
            if r["arm"] not in arms:
                arms.append(r["arm"])
        for arm in arms:
            rows = _rows_by(res, arm)
            worst = min(rows, key=lambda r: r["cov95"])
            pred, d_se, d_bias, d_oth, label = decompose(worst, nominal)
            print(f"{key:<26}  {arm:<21}  {worst['h']:>3}  "
                  f"{worst['cov95']:>7.3f}  {pred:>6.3f}  {d_se:>+7.3f}  "
                  f"{d_bias:>+7.3f}  {d_oth:>+7.3f}  {label}")


# ==========================================================================
# driver
# ==========================================================================
NOTES = """
WHERE THE INTERVALS MISS -- the honest list, read with the tables above
----------------------------------------------------------------------
Nothing in this section is a bug report unless it says so. Three different
things can make a nominal 95% interval miss, and they call for different
responses:  the SE formula can be wrong (a bug),  the SE can be right but its
finite-sample distribution not yet normal (the approximation),  or the
estimator can be off-centre so that no SE saves it (bias). Each item below
says which.

1. lp(se="hac") under-covers, and the shortfall grows with the horizon. The
   miss is in the STANDARD ERROR, not the point estimate: the two arms share
   draws and have identical bias, while HAC's se/sd sits below the
   lag-augmented arm's throughout. Newey-West at the default bandwidth
   h + p is spending its degrees of freedom estimating h + p autocovariances
   that, after the impulse is residualised, are close to zero in this DGP --
   and Bartlett-weighted sample autocovariances of a near-white score are
   biased toward shrinking the variance estimate. THE APPROXIMATION, not the
   formula: Experiment 2 shows both the coverage gap and the se/sd shortfall
   closing monotonically as T grows.

2. lp(se="lag_augmented"), the DEFAULT, is materially better at every horizon
   -- claim (a) holds, which is the whole reason it is the default -- but it is
   not exactly nominal either. Its se/sd sits within a couple of percent of 1
   throughout, and its coverage runs about a point below nominal in the middle
   horizons, drifting back up to nominal at the longest ones as se/sd crosses
   above 1 (h impulse lags on a sample of T - 2h leaves a small,
   heavily-parameterised regression whose HC1 variance is noisy and, on
   average, slightly generous). Both deviations are small and neither has a
   worrying sign. THE APPROXIMATION.

2b. Experiment 1b repeats the comparison with heteroskedastic errors. Both
   standard errors advertise heteroskedasticity-robustness and both keep it:
   every number moves, the RANKING does not. So HAC's deficit in item 1 is
   about smoothing serial correlation that is not there, not about
   heteroskedasticity.

3. Both lp arms are slightly off-centre at long horizons: bias is small in
   absolute terms but |bias|/sd climbs, because the true response has decayed
   toward zero while the sampling standard deviation has not. This is ordinary
   dynamic-regression finite-sample bias, it is the same in both arms, and it
   puts a ceiling on the coverage either SE can deliver. THE ESTIMATOR, in
   finite samples; it is consistent, and Experiment 2 shows it shrinking.

4. lp_iv with a WEAK instrument does not do what one might expect. It does not
   collapse: it OVER-covers (measure it in the table -- the level is well
   above 95%) while its median interval width explodes several-fold. That is
   the correct symptom, not a lucky escape. Just-identified 2SLS has no finite
   moments, and the standard error inflates in exactly the draws where the
   point estimate wanders, so the Wald interval is conservative on average and
   uninformative in length. Dufour (1997) is the reason to expect this: under
   weak identification no BOUNDED-length confidence set can have correct
   coverage, so a Wald set can only be honest by being enormous. Do not read
   the weak arm's `bias` and `sd_est` columns as population quantities --
   those moments do not exist; they describe this particular set of draws.
   THE APPROXIMATION, and unfixable by a better SE: the fix is an interval
   that is not centred on the point estimate at all (Anderson-Rubin), which
   the library does not currently offer for LP-IV. Report the first-stage F.

5. lp_iv with a STRONG instrument still under-covers by two to four points at
   EVERY horizon, impact included, with se/sd around 0.9 and negligible bias.
   Two known contributors, both in the SE: the kernel covariance follows
   linearmodels' `debiased=False` convention, so there is no
   degrees-of-freedom correction; and the default bandwidth h + p applies p
   lags of Bartlett smoothing even at h = 0, where the score has nothing to
   smooth. The convention is deliberate (it is what makes the numbers match
   the reference implementation to golden precision), but its effect on
   coverage is downward, and that is worth knowing before quoting an LP-IV
   interval at face value. Compare the impact row here with the
   lag-augmented impact row of Experiment 1, which is at nominal: the gap is
   not about instrumenting, it is about the covariance convention.

6. lp_multiplier is well centred (|bias|/sd stays small) and its instrument
   stays strong (median F comfortably above 10 at every horizon), yet coverage
   slides as the accumulation window widens, with se/sd around 0.9 throughout.
   So the multiplier's miss is squarely the SE: an se/sd of 0.9 means the
   honest critical value at T = 240 is nearer 2.2 than 1.96. Same convention
   issue as item 5, compounded by a bandwidth that grows with h. THE
   APPROXIMATION -- and note that this DGP is the FAVOURABLE case, because the
   impulse is persistent (rho_x = 0.8) so a cumulated impulse is mostly
   signal. With a transitory impulse the same code faces a mechanically
   decaying first stage, and then item 4 applies instead.

7. lp_state's regime-1 response -- the persistent regime -- is the most
   off-centre object in this whole module: |bias|/sd reaches roughly 0.4 at
   long horizons and coverage sits a few points low even though se/sd is close
   to 1. That is not the standard error's fault. The interacted design spends
   twice the parameters and identifies each regime off roughly half a
   persistent sample, so the lag controls cannot span the regime's dynamics.
   The quiet regime (state 0, faster decay) is close to nominal. THE
   ESTIMATOR, in finite samples: state-dependent LP needs more data than
   linear LP for the same interval to mean the same thing.

8. smooth_lp's DEFAULT (lam="cv") is the largest miss measured here, and it is
   worst exactly where an applied reader looks first: the impact response. The
   table gives the decomposition. At impact the cross-validated penalty pulls
   the estimate off the peak (|bias|/sd above 1 -- the bias exceeds a whole
   sampling standard deviation) so the interval is centred in the wrong place;
   coverage there is nowhere near 95%. THE ESTIMATOR, by design: a penalized
   IRF is a bias-variance trade and the bias is the thing being bought. A
   smooth-LP band is a band around the PENALIZED estimand, and reporting it as
   a confidence interval for the unpenalized impulse response is the error --
   the library's own model card says `se` does not account for shrinkage bias,
   and this is the size of what that sentence is hiding.

9. On top of the bias, cross-validating lambda costs a second, smaller thing:
   the reported SE conditions on the selected lambda, so it never sees the
   variability that selection adds. Compare se/sd in the lam=cv arm with the
   fixed-lambda arms -- it is lower, i.e. the library understates its own
   sampling variability by the amount lambda re-selection contributes. The
   fixed-lambda arm (lam=100) isolates the two effects: there, se/sd is close
   to 1 (SE correctly sized) while |bias|/sd at impact is already large (pure
   shrinkage). THE APPROXIMATION for the se/sd part -- a post-selection or
   bootstrap interval would fix it; the bias part cannot be fixed by any SE.

10. Even smooth_lp's UNPENALIZED anchor (lam=0) does not reach nominal in the
    middle horizons: se/sd drops to roughly 0.85-0.9 there, and coverage with
    it. lam=0 reproduces the per-horizon lp(se="hac") POINT estimates, but its
    variance comes from the stacked design's HAC sandwich over
    base-time-aggregated scores, which is not the same estimator as per-horizon
    HAC. So part of what looks like a cost of smoothing in the lam=cv arm is
    this anchor already being optimistic, and the lam=cv column should be read
    against the lam=0 column rather than against 0.95. Compare the lam=0 rows
    with the HAC rows of Experiment 1, which use a different variance for the
    same point estimates.

11. What is NOT measured here, and should not be assumed: lp(cumulative=...)
    intervals; any SIMULTANEOUS band. Every number in this module is a
    POINTWISE interval at a single horizon. A band that contains the WHOLE
    true IRF path 95% of the time must be wider than any of these -- with 13
    horizons and correlated estimates, materially wider -- and none of these
    functions reports one. Reading a pointwise 95% band as if it covered the
    path is a bigger error than every miss listed above.
"""


def run(quick=False):
    reps_full = {
        "lag_augmented_vs_hac": 4000,
        "lag_augmented_vs_hac_het": 2000,
        "sample_size": 2000,
        "lp_iv": 3000,
        "lp_state": 1500,
        "lp_multiplier": 3000,
        "smooth_lp": 700,
    }
    scale = 8 if quick else 1
    reps = {k: max(100, v // scale) for k, v in reps_full.items()}

    t0 = time.perf_counter()
    print("=" * 100)
    print("tsecon interval COVERAGE: the local-projection family")
    print("=" * 100)
    print(f"seed                = {SEED}   (every draw is default_rng("
          f"[{SEED}, experiment, replication]))")
    print(f"nominal level       = {NOMINAL:.0%} two-sided, z = {Z95:.6f}")
    print(f"true IRF            = theta_h = {RHO}**h, truncated at J = {J}")
    print(f"mode                = {'QUICK SMOKE RUN' if quick else 'full'}")
    print("replications        = " + ", ".join(f"{k}:{v}" for k, v in
                                               reps.items()))

    results = {}

    header("EXPERIMENT 1 -- lp: the default (lag-augmented HC1) vs HAC")
    results["lag_augmented_vs_hac"] = exp_lag_augmented_vs_hac(
        reps["lag_augmented_vs_hac"])
    print(f"T = {results['lag_augmented_vs_hac']['meta']['T']}, "
          f"p = {results['lag_augmented_vs_hac']['meta']['n_lag_controls']} "
          f"lag controls, reps = {reps['lag_augmented_vs_hac']}, "
          f"paired draws\n")
    print_table(results["lag_augmented_vs_hac"]["rows"])
    print_paired(results["lag_augmented_vs_hac"])

    header("EXPERIMENT 1b -- the same comparison under heteroskedasticity")
    print("eta_t variance scales with |s_{t-1}| (predetermined w.r.t. the")
    print("shock, so the truth is unchanged). Both SEs claim robustness.\n")
    results["lag_augmented_vs_hac_het"] = exp_lag_augmented_vs_hac(
        reps["lag_augmented_vs_hac_het"], het=True)
    print_table(results["lag_augmented_vs_hac_het"]["rows"])

    header("EXPERIMENT 2 -- lp: does the gap close as T grows?")
    results["sample_size"] = exp_sample_size(reps["sample_size"])
    print(f"horizons reported: 0, 4, 8, 12; reps = {reps['sample_size']}\n")
    print_table(results["sample_size"]["rows"])

    header("EXPERIMENT 3 -- lp_iv: strong instrument vs weak instrument")
    results["lp_iv"] = exp_lp_iv(reps["lp_iv"])
    print(f"strong: z = 1.0*s + 0.5*nu   weak: z = 0.2*s + 1.0*nu   "
          f"(truth unchanged)\n")
    print_table(results["lp_iv"]["rows"],
                extra_cols=results["lp_iv"]["extra_cols"])

    header("EXPERIMENT 4 -- lp_state: per-regime coverage")
    results["lp_state"] = exp_lp_state(reps["lp_state"])
    m = results["lp_state"]["meta"]
    print(f"T = {m['T']}, P(stay) = {m['p_stay']}, "
          f"theta1_h = 1.2*0.85**h, theta0_h = 0.4*0.5**h, "
          f"reps = {m['reps']}\n")
    print_table(results["lp_state"]["rows"])

    header("EXPERIMENT 5 -- lp_multiplier: the integral multiplier")
    results["lp_multiplier"] = exp_lp_multiplier(reps["lp_multiplier"])
    m = results["lp_multiplier"]["meta"]
    print(f"T = {m['T']}, persistent impulse rho_x = {m['rho_x']}, "
          f"reps = {m['reps']}; truth = cumsum(theta)/cumsum(rho_x**h)\n")
    print_table(results["lp_multiplier"]["rows"],
                extra_cols=results["lp_multiplier"]["extra_cols"])

    header("EXPERIMENT 6 -- smooth_lp: what conditioning on lambda costs")
    results["smooth_lp"] = exp_smooth_lp(reps["smooth_lp"])
    print(f"T = 200, reps = {reps['smooth_lp']}, penalty_order = 2 "
          f"(shrinks the IRF toward a straight line)")
    for arm, med, lo, hi in results["smooth_lp"]["lambda_used"]:
        print(f"  {arm:<8} lambda_used: median {med:g}, "
              f"range [{lo:g}, {hi:g}]")
    print()
    print_table(results["smooth_lp"]["rows"])

    print_misses(results)
    print(NOTES)
    results["_checks"] = check(results, quick)
    elapsed = time.perf_counter() - t0
    print()
    print(f"runtime: {elapsed:.1f} s")
    results["_runtime_s"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interval coverage for the tsecon local-projection family")
    parser.add_argument("--quick", action="store_true",
                        help="cut every replication count by 8 for a smoke run")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
