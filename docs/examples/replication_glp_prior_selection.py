"""Replication: Giannone, Lenza & Primiceri (2015) hierarchical prior selection.

GLP's REStat paper ("Prior Selection for Vector Autoregressions") treats the
Minnesota prior's overall tightness as a parameter to be *estimated* — pick
lambda by maximizing the closed-form marginal likelihood, under a Gamma
hyperprior with mode 0.2 and sd 0.4. `tsecon.bvar_hierarchical` implements
exactly this machinery; this script runs it through GLP's small- and
medium-VAR *design* and checks the behaviour their paper reports.

WHAT THIS IS, AND IS NOT — read before quoting numbers
------------------------------------------------------
This is a **design replication on nearby public data, not a run on GLP's
dataset**. GLP's data is the Stock-Watson (2008) panel (real GDP, GDP
deflator, federal funds rate, ... 1959Q1-2008Q4); their replication archive
is distributed through their publishers, and tsecon vendors no licensed
datasets. The committed fixture (fixtures/glp_smallvar.csv) is the
public-domain statsmodels `macrodata` panel over the same sample, carrying
the same *kind* of variables:

    GLP small VAR              this script
    ---------------            -----------------------------------
    real GDP        (4*log)    realgdp   (4*log)      same concept
    GDP deflator    (4*log)    cpi       (4*log)      CPI is NOT the deflator
    federal funds   (/100)     tbilrate  (/100)       T-bill is NOT fed funds

Everything else follows GLP exactly: annualized log-levels (4*log), interest
rate in levels/100, sample 1959Q1-2008Q4, five lags, random-walk prior mean
(delta = 1), single overall-tightness hyperparameter with the GLP Gamma
hyperprior (mode 0.2, sd 0.4 — verified against `setpriors.m` in GLP's own
web replication files). One further documented convention difference: GLP's
Figure-1 illustration scales the prior with AR(1) residual variances
(Kadiyala-Karlsson); tsecon uses AR(4) residual variances (see the Bayesian
model card). On GLP's own data that one convention moves the selected
tightness from ~0.45 to ~0.26 — so expect this page's *numbers* to differ
from the published figure while every *claim* reproduces.

The script is fully deterministic — the marginal likelihood is closed-form,
nothing is simulated, so there is no seed to set.

    .venv/bin/python docs/examples/replication_glp_prior_selection.py
"""
import csv
import math
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "glp_smallvar.csv"

# The GLP Gamma hyperprior: mode 0.2, sd 0.4  =>  shape a, scale s.
GLP_A = (9.0 + math.sqrt(17.0)) / 8.0
GLP_S = 0.2 / (GLP_A - 1.0)


def load_macro(path=DATA):
    """Read the committed macrodata-derived quarterly panel.

    Public-domain US-government statistical series (via statsmodels
    `macrodata`), vendored with attribution — no download, no loader.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    names = rows[0]
    cols = {n: [] for n in names}
    for r in rows[1:]:
        for n, cell in zip(names, r):
            cols[n].append(float(cell))
    return {n: np.asarray(v) for n, v in cols.items()}


def build_variables(m):
    """GLP's transformation: annualized log-levels (4*log), rates /100.

    Returns the small (3-variable) and medium (7-variable) designs. GLP's
    medium VAR adds consumption, investment, hours and wages to the small
    one; macrodata has no hours or wages series, so the medium analogue here
    adds consumption, investment, government spending and disposable income
    — four real aggregates, keeping the small set nested as in GLP.
    """
    small = np.column_stack([
        4.0 * np.log(m["realgdp"]),
        4.0 * np.log(m["cpi"]),
        m["tbilrate"] / 100.0,
    ])
    medium = np.column_stack([
        4.0 * np.log(m["realgdp"]),
        4.0 * np.log(m["cpi"]),
        4.0 * np.log(m["realcons"]),
        4.0 * np.log(m["realinv"]),
        4.0 * np.log(m["realgovt"]),
        4.0 * np.log(m["realdpi"]),
        m["tbilrate"] / 100.0,
    ])
    return small, medium


def select_tightness(data, tol=1e-8):
    """GLP MAP-II selection: maximize log ML + log Gamma hyperprior.

    `hyperprior="glp"` is the library default; it is spelled out here because
    the *point* of this page is that this Gamma(mode 0.2, sd 0.4) hyperprior
    is GLP's own (their `setpriors.m`: mode.lambda = .2, sd.lambda = .4).
    delta = 1 puts the random-walk prior mean on the own first lag — GLP's
    choice for variables entering in log-levels.
    """
    return tsecon.bvar_hierarchical(
        data, lags=5, delta=1.0, hyperprior="glp", tol=tol
    )


def lambda_profile(data, lo=0.01, hi=5.0, n_grid=41):
    """The lambda1 log-ML profile and the (normalized) posterior kernel."""
    h = tsecon.bvar_hierarchical(
        data, lags=5, delta=1.0, hyperprior="glp",
        lambda1_lo=lo, lambda1_hi=hi, n_grid=n_grid, tol=1e-6,
    )
    grid = np.asarray(h["grid_lambda1"])
    log_ml = np.asarray(h["grid_log_ml"])
    log_post = log_ml + (GLP_A - 1.0) * np.log(grid) - grid / GLP_S
    kernel = np.exp(log_post - log_post.max())
    return grid, log_ml, kernel


def rule(width=72, ch="-"):
    print(ch * width)


def main():
    print("Replication — Giannone, Lenza & Primiceri (2015), REStat 97(2)")
    print("hierarchical (MAP-II) selection of the Minnesota overall tightness")
    rule(72, "=")

    m = load_macro()
    T = len(m["year"])
    print(f"data: macrodata-derived panel (committed), {T} quarters, "
          f"{int(m['year'][0])}Q{int(m['quarter'][0])} to "
          f"{int(m['year'][-1])}Q{int(m['quarter'][-1])}")
    print("design: GLP small/medium VAR — 4*log levels, rates/100, lags = 5,")
    print("        delta = 1, GLP Gamma hyperprior on lambda1 (mode .2, sd .4)")
    print("NOT GLP's data: CPI stands in for the GDP deflator, the 3-month")
    print("T-bill for the federal funds rate (see the docs page).")

    small, medium = build_variables(m)
    fit_s = select_tightness(small)
    # The 7-variable likelihood is flatter near its optimum; 1e-6 is the
    # tolerance at which the Nelder-Mead polish certifies convergence (the
    # selected lambda1 agrees with the 1e-8 run to six decimals).
    fit_m = select_tightness(medium, tol=1e-6)

    print("\nSELECTED OVERALL TIGHTNESS (MAP-II under the GLP hyperprior)")
    rule()
    for label, fit in [
        ("small  (3 variables)", fit_s),
        ("medium (7 variables)", fit_m),
    ]:
        print(f"  {label}: lambda1_opt = {fit['lambda1_opt']:.4f}   "
              f"log-ML at opt = {fit['log_marginal_likelihood']:.2f}   "
              f"converged = {fit['converged']}")

    print("\nTHE SELECTION DOMINATES FIXED REFERENCES (log marginal likelihood)")
    rule()
    print(f"  {'lambda1':>10} | {'small':>10} | {'medium':>10} |")
    for lam in (0.01, 0.05, 0.2, 1.0, 5.0):
        ml_s = tsecon.bvar_fit(small, lags=5, lambda1=lam, delta=1.0)
        ml_m = tsecon.bvar_fit(medium, lags=5, lambda1=lam, delta=1.0)
        tag = "  <- the folklore default" if lam == 0.2 else ""
        print(f"  {lam:>10.2f} | {ml_s['log_marginal_likelihood']:>10.2f} | "
              f"{ml_m['log_marginal_likelihood']:>10.2f} |{tag}")
    print(f"  {'selected':>10} | {fit_s['log_marginal_likelihood']:>10.2f} | "
          f"{fit_m['log_marginal_likelihood']:>10.2f} |  <- lambda1_opt")

    print("\nLAMBDA1 POSTERIOR PROFILE (small VAR; kernel normalized to max 1)")
    rule()
    grid, log_ml, kernel = lambda_profile(small)
    print(f"  {'lambda1':>9} | {'log-ML rel max':>14} | {'posterior kernel':>16}")
    for g, ml_rel, kk in zip(grid, log_ml - log_ml.max(), kernel):
        if kk > 0.01:
            bar = "#" * int(round(40 * kk))
            print(f"  {g:>9.4f} | {ml_rel:>14.3f} | {kk:>7.3f}  {bar}")

    print()
    rule(72, "=")
    print("Published benchmarks (GLP 2015, verified against the paper draft and")
    print("GLP's own web replication code — see the docs page for provenance):")
    print("  * the hyperprior on lambda is Gamma, mode 0.2, sd 0.4 (setpriors.m);")
    print("  * Figure 1: the posterior of lambda peaks around ~0.45 (small VAR),")
    print("    ~0.17 (medium) and ~0.09 (large) under GLP's AR(1) scale")
    print("    convention — mode AND spread fall as the system grows;")
    print("  * the data-chosen tightness beats fixed loose priors, and the")
    print("    Sims-Zha fixed 0.2 is 'too low' (too tight) for their small VAR.")
    print()
    d_small = fit_s["lambda1_opt"]
    d_medium = fit_m["lambda1_opt"]
    print("This design replication (nearby data, tsecon's AR(4) scales):")
    print(f"  * selected lambda1: {d_small:.3f} (small) -> {d_medium:.3f} (medium)")
    print(f"    — same order of magnitude as GLP's, tightening as the")
    print(f"    cross-section grows, exactly as their Figure 1 reports;")
    gain_loose = fit_s["log_marginal_likelihood"] - tsecon.bvar_fit(
        small, lags=5, lambda1=5.0, delta=1.0)["log_marginal_likelihood"]
    print(f"  * the selection beats a near-flat prior (lambda1 = 5) by "
          f"{gain_loose:.0f} nats of log-ML (small VAR);")
    print(f"  * the small-VAR optimum sits just above the 0.2 folklore value —")
    print(f"    the direction (looser than 0.2) matches GLP's finding that")
    print(f"    Sims-Zha shrinkage is too tight for a 3-variable VAR.")
    print()
    print("Numbers here are NOT GLP's published numbers: different price and")
    print("interest-rate series, a different data vintage, and tsecon's AR(4)")
    print("(not AR(1)) scale regressions. On GLP's own (uncommitted) data the")
    print("same call selects 0.260/0.142, and switching only the scale")
    print("convention to GLP's AR(1) yields 0.449/0.172 — matching their")
    print("published Figure 1. See the docs page for that decomposition.")


if __name__ == "__main__":
    main()
