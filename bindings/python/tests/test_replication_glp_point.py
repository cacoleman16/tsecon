"""POINT replication of GLP (2015) Figure 1 on GLP's own data.

The design replication (test_replication_glp.py) runs GLP's design on a
nearby public panel and honestly refuses to pin their published numbers.
This file runs it on GLP's data: fixtures/glp_sw_panel.csv is the
Stock-Watson panel exactly as their own web replication code consumes it
(the `y` matrix of their `DataSW.mat`, vendored from the public
FRBNY-DSGE/BrookingsPC2020 GitHub mirror with the mirror's redistribution
notice kept in the CSV header — see the docs page for provenance).

With `scale_ar=1` — the 0.4.0 option matching GLP's own residual-scale
convention (`setpriors.m`, MNpsi=0: prior scales from AR(1), not AR(4),
residual variances) — the selected tightness modes must land on the
published Figure-1 locations:

* small VAR (real GDP, GDP deflator, fed funds): published mode ~0.449,
  read from the figure's vector graphics at +/-0.03 (its resolution);
  obtained here 0.420 — pinned at |lambda - 0.449| <= 0.03, i.e. exactly
  the stated reading resolution (this is the loosest honest pin: the
  obtained mode sits at the lower edge of the 0.42-0.45 reading band, and
  the marginal likelihood is flat enough near its peak on this collinear
  levels data that the second decimal of the argmax is soft);
* medium VAR (all seven variables): published mode ~0.172; obtained
  0.1716 — pinned at |lambda - 0.172| <= 0.01.

Under the shipped AR(4) default the same data selects 0.260/0.142 — the
documented convention is the entire remaining gap, which is the claim the
0.3.0 design replication could only make at development time and this test
now makes in CI. Everything is closed-form and deterministic (no seed).
"""
import csv
from pathlib import Path

import numpy as np
import pytest

tsecon = pytest.importorskip("tsecon")

PANEL = Path(__file__).resolve().parents[3] / "fixtures" / "glp_sw_panel.csv"

# Published Figure-1 modes (GLP 2015) and the stated reading tolerances.
PUBLISHED_SMALL = 0.449
TOL_SMALL = 0.03  # the figure's vector-extraction resolution
PUBLISHED_MEDIUM = 0.172
TOL_MEDIUM = 0.01


@pytest.fixture(scope="module")
def panel():
    rows = [r for r in csv.reader(open(PANEL)) if r and not r[0].startswith("#")]
    names = rows[0]
    data = np.array([[float(v) for v in r] for r in rows[1:]])
    return names, data


@pytest.fixture(scope="module")
def designs(panel):
    names, data = panel
    cols = {n: data[:, i] for i, n in enumerate(names)}
    medium = np.column_stack([
        cols["rgdp_4log"], cols["pgdp_4log"], cols["cons_4log"],
        cols["gpdinv_4log"], cols["hours_4log"], cols["rcomp_4log"],
        cols["fedfunds_dec"],
    ])
    small = medium[:, [0, 1, 6]]
    return small, medium


def select(data, tol=1e-8, scale_ar=1):
    """GLP's Figure-1 estimation: 5 lags, random-walk prior mean, their
    Gamma(mode 0.2, sd 0.4) hyperprior (MAP-II), AR(scale_ar) prior scales."""
    return tsecon.bvar_hierarchical(
        data, lags=5, delta=1.0, hyperprior="glp", tol=tol, scale_ar=scale_ar
    )


def test_panel_is_glp_own_dataset(panel, designs):
    """The committed CSV is the 1959Q1-2008Q4 Stock-Watson panel in GLP's
    units (annualized log-levels; funds rate as a decimal)."""
    names, data = panel
    small, medium = designs
    assert data.shape == (200, 9)  # year, quarter, 7 series
    assert (data[0, 0], data[0, 1]) == (1959, 1)
    assert (data[-1, 0], data[-1, 1]) == (2008, 4)
    assert small.shape == (200, 3)
    assert medium.shape == (200, 7)
    # 4*log units: the quantity/price indexes live in the low tens...
    assert np.all((10.0 < medium[:, :6]) & (medium[:, :6] < 21.0))
    # ...and the funds rate is a decimal (1959Q1 = 2.57% -> 0.0257).
    assert np.all((0.0 < medium[:, 6]) & (medium[:, 6] < 0.20))
    assert np.isclose(medium[0, 6], 0.0257, atol=1e-6)


def test_small_var_mode_matches_published_figure1(designs):
    """Published small-VAR mode ~0.449 (Figure 1, read at +/-0.03): with
    GLP's own AR(1) scale convention the selection lands inside that
    reading band."""
    small, _ = designs
    fit = select(small)
    assert fit["converged"]
    lam = fit["lambda1_opt"]
    assert abs(lam - PUBLISHED_SMALL) <= TOL_SMALL, (
        f"small-VAR mode {lam:.4f} vs published {PUBLISHED_SMALL} "
        f"(tolerance {TOL_SMALL})"
    )
    # Self-regression pin on the value this build obtains (0.4205), so a
    # numerical drift inside the published band still fails loudly.
    assert 0.41 < lam < 0.43


def test_medium_var_mode_matches_published_figure1(designs):
    """Published medium-VAR mode ~0.172: obtained 0.1716 under AR(1)."""
    _, medium = designs
    fit = select(medium, tol=1e-6)
    assert fit["converged"]
    lam = fit["lambda1_opt"]
    assert abs(lam - PUBLISHED_MEDIUM) <= TOL_MEDIUM, (
        f"medium-VAR mode {lam:.4f} vs published {PUBLISHED_MEDIUM} "
        f"(tolerance {TOL_MEDIUM})"
    )
    assert 0.165 < lam < 0.18


def test_the_convention_is_the_gap(designs):
    """The headline: on GLP's own data the AR(4)-vs-AR(1) residual-scale
    convention accounts for the entire distance between tsecon's shipped
    default and the published modes. AR(4) selects ~0.260/~0.142 (outside
    the published bands); switching ONLY scale_ar closes it."""
    small, medium = designs
    s4 = select(small, scale_ar=4)
    m4 = select(medium, tol=1e-6, scale_ar=4)
    # The AR(4) default sits well below the published small mode...
    assert 0.24 < s4["lambda1_opt"] < 0.28
    assert abs(s4["lambda1_opt"] - PUBLISHED_SMALL) > 0.15
    # ...and below the published medium mode.
    assert 0.13 < m4["lambda1_opt"] < 0.16
    # Figure 1's cross-section direction holds under both conventions.
    s1 = select(small)
    m1 = select(medium, tol=1e-6)
    assert m4["lambda1_opt"] < s4["lambda1_opt"]
    assert m1["lambda1_opt"] < s1["lambda1_opt"]
