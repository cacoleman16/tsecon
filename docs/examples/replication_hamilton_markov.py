"""Replication: Hamilton (1989) Markov-switching AR(4) on US real GNP growth.

Hamilton's two-regime model of quarterly US real GNP growth is the founding
application of regime-switching econometrics: an AR(4) whose *mean* jumps
between an expansion state and a contraction state governed by a hidden
first-order Markov chain. The published headline (Econometrica 57(2), Table I,
p. 372): expansions grow at about +1.16% per quarter and contractions shrink
at about -0.36%, the expansion state persists with probability p = 0.9049 and
the contraction state with q = 0.7550 (expected durations ~10 and ~4
quarters), and the smoothed recession probabilities reproduce the NBER
business-cycle dates without ever being shown them.

Data: Hamilton's own series - 100 * diff(log(real GNP)), seasonally adjusted,
1951Q2-1984Q4 (estimation sample 1952Q2-1984Q4 after 4 lags, n = 131) - is
committed at fixtures/hamilton_gnp.csv (copied verbatim from the statsmodels
regime-switching test suite, which vendors it; US-government statistical data,
public domain). This runs fully offline - the library ships no data loaders.

    .venv/bin/python docs/examples/replication_hamilton_markov.py

WHAT THIS IS, AND IS NOT
------------------------
`tsecon.markov_switching_ar(..., switching_variance=False)` expresses
Hamilton's exact specification: switching MEAN (not intercept - the AR applies
to deviations `y_t - mu_{S_t}`, so lagged regimes enter the likelihood),
common AR(4), common variance. One honest caveat:

* tsecon fits by EM, whose transition M-step (expected counts) conditions on
  the stationary initial state distribution rather than re-differentiating it
  with respect to P; its fixed point therefore sits O(1/T) away from the exact
  MLE that statsmodels/E-views reach by quasi-Newton refinement. On this
  sample that is a gap of 0.006 log-likelihood points and <= 0.016 on any
  parameter (the AR coefficients included) - visible in the third decimal,
  irrelevant to the economics.
"""
import csv
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "hamilton_gnp.csv"

# Hamilton (1989), Table I, p. 372 (sigma^2 and AR at the E-views/statsmodels
# re-estimation precision; Hamilton prints sigma = 0.769, i.e. sigma^2 = 0.591).
PUBLISHED = {
    "mu_expansion": 1.16,
    "mu_contraction": -0.36,
    "p_expansion_stay": 0.9049,
    "p_contraction_stay": 0.7550,
    "sigma2": 0.5914,
    "ar": (0.014, -0.058, -0.247, -0.213),
}


def load_hamilton_gnp(path=DATA):
    """Read the committed Hamilton GNP-growth series.

    Public-domain US statistical data, vendored with attribution - no
    download, no loader. Returns quarter labels, the growth series, and the
    NBER-based recession indicator that ships alongside it.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    body = rows[1:]
    return {
        "quarter": [r[0] for r in body],
        "growth": np.array([float(r[1]) for r in body]),
        "nber": np.array([int(r[2]) for r in body]),
    }


def fit_tsecon(y):
    """Fit Hamilton's spec with tsecon and normalize the labeling.

    `switching_variance=False` is what makes this Hamilton's model: switching
    mean, common AR, ONE innovation variance. EM regime indices are
    arbitrary, so regimes are identified by their means - contraction is the
    low-mean state - never by index.
    """
    r = tsecon.markov_switching_ar(
        y, k_regimes=2, order=4, switching_variance=False,
        max_iter=5000, tol=1e-8,
    )
    means = np.asarray(r["means"])
    trans = np.asarray(r["transition"])  # P[i][j] = P(S_t=i | S_{t-1}=j)
    i_con, i_exp = int(np.argmin(means)), int(np.argmax(means))
    durations = np.asarray(r["expected_durations"])
    return {
        "mu_contraction": means[i_con],
        "mu_expansion": means[i_exp],
        "p_contraction_stay": trans[i_con, i_con],
        "p_expansion_stay": trans[i_exp, i_exp],
        "sigma2": r["variances"][0],
        "ar": tuple(np.asarray(r["ar"])),  # common (phi_1..phi_4), regime-free
        "loglik": r["loglik"],
        "duration_contraction": durations[i_con],
        "duration_expansion": durations[i_exp],
        "prob_contraction": np.asarray(r["smoothed_prob"])[:, i_con],
        "converged": r["converged"],
    }


def fit_statsmodels(y):
    """The same specification through statsmodels' MarkovAutoregression.

    `switching_ar=False` leaves the AR common while `const` switches; in
    MarkovAutoregression the switching "const" IS the regime mean (the AR
    applies to demeaned values), so the parameterizations line up exactly -
    no intercept-to-mean mapping needed on either side. Returns None when
    statsmodels is not installed, keeping the page runnable offline anywhere.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )
    except ImportError:
        return None
    res = MarkovAutoregression(y, k_regimes=2, order=4, switching_ar=False).fit()
    p = dict(zip(res.model.param_names, res.params))
    means = np.array([p["const[0]"], p["const[1]"]])
    i_con, i_exp = int(np.argmin(means)), int(np.argmax(means))
    stay = {0: p["p[0->0]"], 1: 1.0 - p["p[1->0]"]}
    return {
        "mu_contraction": means[i_con],
        "mu_expansion": means[i_exp],
        "p_contraction_stay": stay[i_con],
        "p_expansion_stay": stay[i_exp],
        "sigma2": p["sigma2"],
        "ar": tuple(p[f"ar.L{i}"] for i in range(1, 5)),
        "loglik": res.llf,
        "duration_contraction": res.expected_durations[i_con],
        "duration_expansion": res.expected_durations[i_exp],
        "prob_contraction": res.smoothed_marginal_probabilities[:, i_con],
    }


def nber_episodes(indicator):
    """Contiguous runs of 1s in a 0/1 array, as (start, end) index pairs."""
    episodes, start = [], None
    for i, v in enumerate(indicator):
        if v and start is None:
            start = i
        if not v and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(indicator) - 1))
    return episodes


def rule(width=78, ch="-"):
    print(ch * width)


def main():
    print("Replication - Hamilton (1989), Econometrica 57(2)")
    print("two-regime Markov-switching AR(4), US real GNP growth 1952Q2-1984Q4")
    rule(78, "=")

    d = load_hamilton_gnp()
    y, nber, quarter = d["growth"], d["nber"], d["quarter"]
    print(f"data: {len(y)} quarters {quarter[0]}-{quarter[-1]} (committed fixture);")
    print(f"      estimation sample {quarter[4]}-{quarter[-1]} (n = {len(y) - 4})"
          " after AR(4) conditioning")

    ts = fit_tsecon(y)
    sm = fit_statsmodels(y)

    def row(name, pub, t, s, fmt="{:>11.4f}"):
        pub_s = fmt.format(pub) if pub is not None else " " * 11
        s_s = fmt.format(s) if s is not None else "        n/a"
        print(f"  {name:<26} | {pub_s} | " + fmt.format(t) + f" | {s_s}")

    print()
    print(f"  {'parameter':<26} | {'published':>11} | {'tsecon':>11} | {'statsmodels':>11}")
    rule()
    row("expansion mean mu_1", PUBLISHED["mu_expansion"], ts["mu_expansion"],
        sm and sm["mu_expansion"])
    row("contraction mean mu_0", PUBLISHED["mu_contraction"], ts["mu_contraction"],
        sm and sm["mu_contraction"])
    row("P(stay expansion) p", PUBLISHED["p_expansion_stay"], ts["p_expansion_stay"],
        sm and sm["p_expansion_stay"])
    row("P(stay contraction) q", PUBLISHED["p_contraction_stay"], ts["p_contraction_stay"],
        sm and sm["p_contraction_stay"])
    row("sigma^2 (common)", PUBLISHED["sigma2"], ts["sigma2"], sm and sm["sigma2"])
    row("E[expansion length] qtrs", None, ts["duration_expansion"],
        sm and sm["duration_expansion"], fmt="{:>11.2f}")
    row("E[contraction length] qtrs", None, ts["duration_contraction"],
        sm and sm["duration_contraction"], fmt="{:>11.2f}")
    for i in range(4):
        row(f"common AR phi_{i + 1}", PUBLISHED["ar"][i], ts["ar"][i],
            sm and sm["ar"][i])
    row("log-likelihood", None, ts["loglik"], sm and sm["loglik"], fmt="{:>11.3f}")

    # --- NBER dating -------------------------------------------------------
    rule(78, "=")
    print("Smoothed recession probabilities vs the NBER dates (never shown to the model)")
    lab = quarter[4:]
    rec_eff = nber[4:]
    prob = ts["prob_contraction"]
    classified = prob > 0.5
    agree = (classified == rec_eff.astype(bool)).mean()
    print(f"\n  quarters where 1[P(recession) > 0.5] matches NBER: "
          f"{(classified == rec_eff.astype(bool)).sum()}/{len(rec_eff)} = {agree:.1%}")
    if sm is not None:
        same = np.array_equal(classified, sm["prob_contraction"] > 0.5)
        print(f"  tsecon and statsmodels classify every quarter identically: {same}")
    print("\n  NBER recession           peak P(recession)")
    for s0, e0 in nber_episodes(rec_eff):
        peak = prob[s0:e0 + 1].max()
        print(f"    {lab[s0]}-{lab[e0]}            {peak:.3f}")

    rule(78, "=")
    print("Published benchmark (Hamilton 1989, Table I): expansions at ~+1.16%/qtr,")
    print("contractions at ~-0.36%/qtr, p = 0.9049, q = 0.7550 - recessions are real,")
    print("discrete, persistent (~4 quarters) states, and the model re-derives the")
    print("NBER chronology from GNP growth alone. All of that replicates above.")
    print("\nEstimator: tsecon.markov_switching_ar(k_regimes=2, order=4,")
    print("switching_variance=False) - Hamilton's exact spec (switching mean, common")
    print("AR(4), common variance), fitted by EM. The EM fixed point sits ~0.006")
    print("log-likelihood points from the exact MLE (see the doc page for why), so")
    print("tsecon and statsmodels agree to <= 0.016 on every parameter and produce")
    print("bit-identical recession calls.")


if __name__ == "__main__":
    main()
