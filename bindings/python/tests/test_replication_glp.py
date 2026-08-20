"""Offline regression guard for the GLP (2015) prior-selection design replication.

Runs the replication's estimation against the committed macrodata-derived panel
(fixtures/glp_smallvar.csv) so the claims on the docs page cannot silently rot.
Fully offline and deterministic — the marginal likelihood is closed-form,
nothing is simulated.

What is pinned is the honest set of reproducible facts, NOT GLP's published
numbers (the data here is a nearby public panel, not theirs — see the docs
page): the selected tightness lands in a stated band of the right order of
magnitude, it tightens when the cross-section grows (GLP's Figure-1
direction), the selection dominates fixed loose/tight references in marginal
likelihood, and the returned optimum is internally exact (grid certificate,
refit consistency, hyperprior accounting).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

glp_repl = pytest.importorskip("replication_glp_prior_selection")
tsecon = pytest.importorskip("tsecon")


@pytest.fixture(scope="module")
def panel():
    return glp_repl.load_macro()


@pytest.fixture(scope="module")
def designs(panel):
    return glp_repl.build_variables(panel)


@pytest.fixture(scope="module")
def fit_small(designs):
    return glp_repl.select_tightness(designs[0])


@pytest.fixture(scope="module")
def fit_medium(designs):
    return glp_repl.select_tightness(designs[1], tol=1e-6)


def test_dataset_is_the_committed_glp_design_panel(panel, designs):
    small, medium = designs
    # 1959Q1-2008Q4, the GLP sample span (their vintage differs; this one is
    # the public-domain statsmodels macrodata panel).
    assert len(panel["year"]) == 200
    assert (panel["year"][0], panel["quarter"][0]) == (1959, 1)
    assert (panel["year"][-1], panel["quarter"][-1]) == (2008, 4)
    assert small.shape == (200, 3)
    assert medium.shape == (200, 7)
    # The GLP transformation: annualized log-levels; rates in levels/100.
    assert np.allclose(small[:, 0], 4 * np.log(panel["realgdp"]))
    assert np.all(small[:, 2] < 0.2)  # T-bill as a decimal, not percent
    # The small design is nested in the medium one (GLP's nesting).
    assert np.allclose(small[:, 0], medium[:, 0])
    assert np.allclose(small[:, 1], medium[:, 1])
    assert np.allclose(small[:, 2], medium[:, 6])


def test_selected_tightness_is_the_documented_order_of_magnitude(fit_small):
    """GLP's small VAR selects a tightness of a few tenths (their Figure 1
    posterior lives on roughly 0.2-0.7 under their AR(1) scale convention;
    tsecon's AR(4) convention lands lower — see the docs page). The honest
    pin: a stated band around this replication's 0.215, far from both the
    collapse-to-floor and the flat-prior corner."""
    assert fit_small["converged"]
    assert 0.15 < fit_small["lambda1_opt"] < 0.30
    # A 3-variable VAR wants a *looser* prior than the 0.2 folklore value —
    # the direction of GLP's "Sims-Zha is too low for the small VAR".
    assert fit_small["lambda1_opt"] > 0.2


def test_tightness_shrinks_as_the_cross_section_grows(fit_small, fit_medium):
    """GLP, on Figure 1: 'the posterior mode (and variance) of lambda
    decreases with the size of the model.' Small (3) -> medium (7)."""
    assert fit_medium["converged"]
    assert fit_medium["lambda1_opt"] < fit_small["lambda1_opt"]
    assert 0.10 < fit_medium["lambda1_opt"] < 0.20


def test_selection_dominates_fixed_references(designs, fit_small, fit_medium):
    """The optimum must beat fixed tight/conventional/loose lambda1 in
    marginal likelihood — the certificate implied by ML-II/MAP-II selection,
    and the in-sample face of GLP's dominance-over-fixed-priors finding."""
    small, medium = designs
    for dat, fit in [(small, fit_small), (medium, fit_medium)]:
        ml_opt = fit["log_marginal_likelihood"]
        ml_conv = tsecon.bvar_fit(dat, lags=5, lambda1=0.2, delta=1.0)
        ml_tight = tsecon.bvar_fit(dat, lags=5, lambda1=0.01, delta=1.0)
        ml_loose = tsecon.bvar_fit(dat, lags=5, lambda1=5.0, delta=1.0)
        assert ml_opt >= ml_conv["log_marginal_likelihood"] - 1e-6
        assert ml_opt > ml_tight["log_marginal_likelihood"] + 30.0
        assert ml_opt > ml_loose["log_marginal_likelihood"] + 50.0


def test_grid_certificate_and_refit_consistency(designs, fit_small):
    """Internal exactness: (i) under a flat hyperprior the returned optimum
    dominates every grid point; (ii) the refit at lambda1_opt is the same
    object bvar_fit produces at that lambda1."""
    small, _ = designs
    ml2 = tsecon.bvar_hierarchical(small, lags=5, delta=1.0, hyperprior="none")
    assert ml2["log_marginal_likelihood"] >= max(ml2["grid_log_ml"]) - 1e-6
    # MAP-II vs ML-II: 200 quarters dominate the diffuse hyperprior, so the
    # two selections differ only marginally.
    assert abs(ml2["lambda1_opt"] - fit_small["lambda1_opt"]) < 0.02

    refit = tsecon.bvar_fit(
        small, lags=5, lambda1=fit_small["lambda1_opt"], delta=1.0
    )
    assert np.isclose(
        refit["log_marginal_likelihood"],
        fit_small["log_marginal_likelihood"],
        rtol=1e-10,
        atol=0.0,
    )
    assert np.allclose(
        np.asarray(refit["posterior_mean_coefs"]),
        np.asarray(fit_small["posterior_mean_coefs"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_scale_ar1_moves_the_selection_toward_the_published_modes(designs, fit_small, fit_medium):
    """The 0.4.0 `scale_ar=1` option switches the prior's residual-scale
    regressions from tsecon's AR(4) default to GLP's own AR(1) convention
    (their setpriors.m, MNpsi=0). On this NEARBY panel the selection must
    move toward — but, different data, not onto — the published Figure-1
    modes (~0.45 small / ~0.17 medium): pinned bands around this
    replication's 0.269 (small) and 0.155 (medium)."""
    small, medium = designs
    s1 = glp_repl.select_tightness(small, scale_ar=1)
    m1 = glp_repl.select_tightness(medium, tol=1e-6, scale_ar=1)
    assert s1["converged"] and m1["converged"]
    # Toward the published small mode (0.215 -> 0.269, published ~0.45)...
    assert s1["lambda1_opt"] > fit_small["lambda1_opt"]
    assert 0.24 < s1["lambda1_opt"] < 0.30
    # ...and toward the published medium mode (0.145 -> 0.155, published ~0.17),
    assert m1["lambda1_opt"] > fit_medium["lambda1_opt"]
    assert 0.14 < m1["lambda1_opt"] < 0.17
    # still tightening as the cross-section grows (Figure 1's direction).
    assert m1["lambda1_opt"] < s1["lambda1_opt"]


def test_scale_ar_default_is_bit_identical_to_scale_ar4(designs):
    """The default path must not move: omitting `scale_ar` and passing
    `scale_ar=4` are the same computation, bit for bit, for bvar_fit,
    bvar_hierarchical, and bvar_irf_draws — and `scale_ar=1` genuinely
    changes the result (the option is not a no-op)."""
    small, _ = designs

    fit_default = tsecon.bvar_fit(small, lags=5, lambda1=0.2, delta=1.0)
    fit_ar4 = tsecon.bvar_fit(small, lags=5, lambda1=0.2, delta=1.0, scale_ar=4)
    assert fit_default == fit_ar4
    fit_ar1 = tsecon.bvar_fit(small, lags=5, lambda1=0.2, delta=1.0, scale_ar=1)
    assert fit_ar1["log_marginal_likelihood"] != fit_default["log_marginal_likelihood"]

    h_default = glp_repl.select_tightness(small)
    h_ar4 = glp_repl.select_tightness(small, scale_ar=4)
    assert h_default == h_ar4

    d_default = tsecon.bvar_irf_draws(small, lags=5, horizon=4, n_draws=3, seed=7, delta=1.0)
    d_ar4 = tsecon.bvar_irf_draws(
        small, lags=5, horizon=4, n_draws=3, seed=7, delta=1.0, scale_ar=4
    )
    assert d_default == d_ar4


def test_hyperprior_accounting_is_the_glp_gamma(fit_small):
    """log_posterior - log_ml must equal the log density of GLP's Gamma
    hyperprior (mode 0.2, sd 0.4) at the selected lambda1 — the exact prior
    their setpriors.m specifies."""
    a = glp_repl.GLP_A
    s = glp_repl.GLP_S
    # mode = (a-1)s = 0.2, var = a s^2 = 0.16
    assert math.isclose((a - 1) * s, 0.2, rel_tol=1e-12)
    assert math.isclose(a * s * s, 0.16, rel_tol=1e-12)
    lam = fit_small["lambda1_opt"]
    log_gamma_pdf = (a - 1) * math.log(lam) - lam / s - a * math.log(s) - math.lgamma(a)
    assert math.isclose(
        fit_small["log_posterior"] - fit_small["log_marginal_likelihood"],
        log_gamma_pdf,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
