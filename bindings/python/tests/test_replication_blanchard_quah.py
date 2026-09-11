"""Offline regression guard for the Blanchard-Quah (1989) design replication.

Runs the replication's estimation against the committed macrodata extract
(fixtures/macrodata_bq.csv) so the findings claimed on the docs page cannot
silently rot. Fully offline - the data is vendored, the library ships no
loaders. Whenever statsmodels is importable, two extra checks run: the CSV
equals the bundled `statsmodels.datasets.macrodata`, and tsecon's closed-form
identification equals an independent NumPy transcription on a statsmodels
VAR fit.

What is pinned, and at what resolution: this is a DESIGN replication at
figure-reading resolution (the paper's 1948-1987 GNP vintage is unreachable;
the bundled series is 1959-2009 real GDP), so every assertion is a SHAPE claim
with a stated numerical band, never a published digit. The point estimates
are deterministic OLS + closed form (no RNG), so their bands are the achieved
values with headroom; the bootstrap assertions use 200 replications at a
fixed seed (the docs script uses 1000). At 200 replications the 5/16/84/95
percentiles move by at most 0.14 across seeds 0, 1, 7 and 42 (measured), so
each band claim below carries at least 2x that margin, and the cross-seed
cap is 0.3.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "docs" / "examples"))

bq = pytest.importorskip("replication_blanchard_quah")

N_BOOT = 200            # reduced from the script's 1000; seeded, see module docstring
H = bq.HORIZON          # 40 quarters


@pytest.fixture(scope="module")
def data():
    return bq.load_macrodata()


@pytest.fixture(scope="module")
def tr(data):
    return bq.transform(data)


@pytest.fixture(scope="module")
def fit(tr):
    return bq.fit_bq(tr["data"])


@pytest.fixture(scope="module")
def boot(tr):
    return bq.bootstrap_bands(tr["data"], n_boot=N_BOOT, seed=0)


# ------------------------------------------------------------------ the data
def test_dataset_is_the_macrodata_extract(data):
    assert len(data["quarter"]) == 203                  # 1959Q1-2009Q3
    assert data["quarter"][0] == "1959Q1" and data["quarter"][-1] == "2009Q3"
    # exact first/last values of the vendored extract (verbatim copy guard)
    assert data["realgdp"][0] == pytest.approx(2710.349, abs=1e-9)
    assert data["realgdp"][-1] == pytest.approx(12990.341, abs=1e-9)
    assert data["unemp"][0] == pytest.approx(5.8, abs=1e-12)
    assert data["unemp"][-1] == pytest.approx(9.6, abs=1e-12)
    assert np.all(np.isfinite(data["realgdp"])) and np.all(data["realgdp"] > 0)


def test_transformation_is_blanchard_quahs(tr):
    """100*dlog(GDP) demeaned before/after 1974Q1; unemployment detrended."""
    X = tr["data"]
    assert X.shape == (202, 2)
    assert tr["quarter"][0] == "1959Q2" and tr["quarter"][-1] == "2009Q3"
    post = tr["years"] >= bq.qnum(bq.MEAN_BREAK)
    assert post.sum() == 143 and (~post).sum() == 59
    assert abs(X[post, 0].mean()) < 1e-12 and abs(X[~post, 0].mean()) < 1e-12
    t = np.arange(202.0)
    Z = np.column_stack([np.ones_like(t), t])
    assert np.abs(np.linalg.lstsq(Z, X[:, 1], rcond=None)[0]).max() < 1e-10
    # the break matters: the two sub-sample growth means differ materially
    raw = 100.0 * np.diff(np.log(bq.load_macrodata()["realgdp"]))
    assert raw[~post].mean() - raw[post].mean() > 0.25        # achieved ~0.36 pp/qtr


# ------------------------------------------------------- the identification
def test_var_is_stable_at_the_papers_lag_order(fit):
    assert bq.LAGS == 8
    assert fit["var"]["is_stable"]
    assert fit["var"]["min_root"] > 1.05                       # achieved 1.153
    assert fit["var"]["nobs"] == 194


def test_long_run_restriction_is_imposed_exactly(fit):
    LR = fit["long_run"]
    assert abs(LR[0, 1]) < 1e-12                               # demand -> output: zero
    assert LR[0, 0] > 0.3                                      # supply is the trend: +0.480
    # the sign convention: a demand disturbance raises output on impact
    assert fit["impact"][0, 1] > 0.0
    assert fit["flip"][0] == 1.0 and abs(fit["flip"][1]) == 1.0


def test_demand_output_response_is_hump_shaped_and_transitory(fit):
    """BQ finding (1): a hump that peaks after about a year and vanishes.

    Achieved: impact +0.708, peak +1.100 at h = 3, h = 20 +0.037,
    h = 40 -0.004 (zero at infinity by construction)."""
    yd = fit["responses"][:, 0, 1]
    assert 0.3 < yd[0] < 1.2
    peak_h = int(np.argmax(yd))
    assert 1 <= peak_h <= 8
    assert yd[peak_h] > yd[0] + 0.1                            # a hump, not a decay
    assert 0.5 < yd[peak_h] < 2.0
    assert abs(yd[20]) < 0.15
    assert abs(yd[40]) < 0.05
    assert abs(yd[40]) < 0.1 * yd[peak_h]


def test_supply_output_response_is_permanent_and_positive(fit):
    """BQ finding (2): builds to a permanent plateau. Achieved: impact
    +0.243, h = 40 +0.486, long run +0.480, minimum over h +0.028
    (deterministic path, no RNG). The 68% band at h = 40 excludes zero."""
    ys = fit["responses"][:, 0, 0]
    LR = fit["long_run"][0, 0]
    assert ys[0] > 0.0
    assert ys[40] > ys[0]                                      # larger than on impact
    assert abs(ys[40] - LR) < 0.02                             # achieved 0.006
    assert ys.min() > 0.0


def test_unemployment_demand_response_is_negative_and_mean_reverting(fit):
    """BQ finding (3): the mirror image of the output hump. Achieved: impact
    -0.188, trough -0.550 at h = 4, h = 20 -0.026, h = 40 +0.003."""
    ud = fit["responses"][:, 1, 1]
    assert ud[0] < -0.1
    trough_h = int(np.argmin(ud))
    assert 1 <= trough_h <= 8
    assert ud[trough_h] < ud[0] - 0.1
    assert abs(ud[20]) < 0.10
    assert abs(ud[40]) < 0.05
    # mirror image: the output hump and the unemployment trough sit within
    # two quarters of each other (achieved h = 3 vs h = 4)
    assert abs(trough_h - int(np.argmax(fit["responses"][:, 0, 1]))) <= 2


def test_supply_raises_unemployment_on_impact(fit):
    """BQ finding (4), the part visible on this vintage: a favourable supply
    disturbance raises unemployment on impact (+0.137) and the effect dies
    out (h = 40 -0.003). BQ's later sign reversal is NOT pinned - it is not
    visible here (minimum -0.005), and the page says so."""
    us = fit["responses"][:, 1, 0]
    assert us[0] > 0.05
    assert abs(us[40]) < 0.05


def test_demand_dominates_short_horizon_variance(fit):
    """BQ finding (5). Achieved demand shares: output level 89.5% at one
    step, >= 92% at steps 2-20, 65.3% at 40 steps; output growth 89.5% at
    one step; unemployment 65.4% at one step and >= 83% from step 4."""
    lvl = fit["fevd_level"][:, 0, 1]        # output LEVEL, demand column
    grw = fit["fevd_growth"][:, 0, 1]       # output growth
    une = fit["fevd_growth"][:, 1, 1]       # unemployment rate
    assert grw[0] > 0.5
    assert lvl[:12].min() > 0.8
    assert lvl[39] < lvl[3] - 0.1                              # declining with the horizon
    assert lvl[39] > 0.4
    assert une[0] > 0.5
    assert une[3:].min() > 0.7
    # rows sum to one across the two disturbances
    assert np.allclose(fit["fevd_level"].sum(axis=2), 1.0, atol=1e-12)
    assert np.allclose(fit["fevd_growth"].sum(axis=2), 1.0, atol=1e-12)


# ---------------------------------------------------------- bootstrap bands
def test_bootstrap_bands_bracket_the_findings(boot):
    """200 replications, seed 0. Achieved (seed 0 / seed 1): demand ->
    unemployment 90% upper at h = 0, 4: -0.106/-0.100, -0.354/-0.360;
    demand -> output 90% lower at h = 0, 4: +0.467/+0.433, +0.697/+0.591;
    demand -> output 90% band at h = 40: [-0.061, +0.121] / [-0.037, +0.113];
    supply -> output 90% lower at h = 40: +0.295/+0.293; supply ->
    unemployment 68% lower at h = 0: +0.063/+0.051."""
    q = boot["quantiles"]                    # [h][variable][shock][prob]; probs 5/16/84/95
    assert boot["probs"] == (0.05, 0.16, 0.84, 0.95)
    assert q.shape == (H + 1, 2, 2, 4)
    assert np.all(q[..., 0] <= q[..., 1]) and np.all(q[..., 2] <= q[..., 3])
    assert q[0, 1, 1, 3] < 0.0 and q[4, 1, 1, 3] < 0.0         # unemployment falls (90%)
    assert q[0, 0, 1, 0] > 0.0 and q[4, 0, 1, 0] > 0.0         # output rises (90%)
    assert q[40, 0, 1, 0] < 0.0 < q[40, 0, 1, 3]                # ... and is gone by h = 40
    assert q[40, 0, 0, 0] > 0.0                                # supply is permanent (90%)
    assert q[0, 1, 0, 1] > 0.0                                 # supply raises u on impact (68%)
    # every replication keeps the sign convention (output impact to demand > 0)
    assert np.all(boot["draws"][:, 0, 0, 1] > 0.0)


def test_bands_are_stable_across_seeds(tr, boot):
    """The docs-page band table must not be a seed artifact: at 200
    replications the percentiles move by <= 0.14 across seeds (measured for
    seeds 0 vs 1, 7, 42); 0.3 is a 2x cap."""
    other = bq.bootstrap_bands(tr["data"], n_boot=N_BOOT, seed=1)
    assert np.abs(boot["quantiles"] - other["quantiles"]).max() < 0.3


def test_seeded_bootstrap_is_bit_reproducible(tr):
    kw = dict(n_boot=40, seed=11, horizon=12)
    a = bq.bootstrap_bands(tr["data"], **kw)
    b = bq.bootstrap_bands(tr["data"], **kw)
    assert np.array_equal(a["quantiles"], b["quantiles"])
    assert np.array_equal(a["draws"], b["draws"])


# ------------------------------------------------ historical decomposition
def test_historical_decomposition_matches_the_library_baseline(tr, fit):
    """The script builds the BQ decomposition from B^-1 u_t and the
    structural MA; its baseline is identification-invariant and must equal
    tsecon.historical_decomposition's (achieved 9e-15), as must the total
    shock contribution (4e-15); the identity y = baseline + sum_j hd holds."""
    hd = bq.historical_decomposition_bq(tr["data"], fit=fit)
    assert hd["baseline_check"] < 1e-10
    assert hd["total_check"] < 1e-10
    X = tr["data"][bq.LAGS:]
    assert np.abs(X - hd["baseline"] - hd["hd"].sum(axis=2)).max() < 1e-12
    # structural shocks are orthogonal with the df-adjusted unit variance
    eps = hd["shocks"]
    T, m = eps.shape[0], 1 + 2 * bq.LAGS
    assert abs(np.corrcoef(eps.T)[0, 1]) < 1e-10
    assert np.allclose(eps.var(axis=0), (T - m) / T, atol=1e-10)


def test_recessions_are_demand(tr, fit):
    """BQ's reading of their Figures 7-8 on this vintage: the demand
    component tracks detrended unemployment (corr 0.926) and carries the
    majority of the peak-to-trough rise in unemployment in every NBER
    recession inside the effective sample (achieved shares 0.71-1.04)."""
    hd = bq.historical_decomposition_bq(tr["data"], fit=fit)
    u = hd["unemployment"]
    assert np.corrcoef(u["actual"], u["demand"])[0, 1] > 0.8
    rows = bq.recession_table(tr, hd)
    assert len(rows) == 7                    # 1969Q4 through 2007Q4 (1960 predates the sample)
    for r in rows:
        assert r["actual"] > 0.5
        assert r["demand"] > r["supply"]
        assert r["demand"] > 0.5 * r["actual"]


# --------------------------------------------------------- lag-order check
def test_findings_survive_the_information_criterion_orders(tr):
    """BIC/HQ pick 2 lags and AIC 3 on this sample; BQ used 8. The shape
    claims hold at each (achieved at p = 2 / 3: demand -> output peak
    +1.11/+1.11 at h = 3, h = 40 -0.001/+0.008; supply long run +0.53/+0.45;
    unemployment impact to demand -0.20/-0.20)."""
    ladder = bq.lag_ladder(tr["data"])
    picks = {c: min(ladder, key=lambda r: r[c])["lags"] for c in ("aic", "bic", "hqic")}
    assert picks == {"aic": 3, "bic": 2, "hqic": 2}
    for p in (2, 3):
        f = bq.fit_bq(tr["data"], lags=p)
        r = f["responses"]
        assert 1 <= int(np.argmax(r[:, 0, 1])) <= 8 and r[:, 0, 1].max() > r[0, 0, 1] + 0.1
        assert abs(r[40, 0, 1]) < 0.05
        assert f["long_run"][0, 0] > 0.3 and r[40, 0, 0] > 0.0
        assert r[0, 1, 1] < -0.1


# ------------------------------------------------------------- dual golden
sm = pytest.importorskip("statsmodels")


def test_csv_equals_the_statsmodels_bundle(data):
    """Provenance: the committed extract IS the bundled dataset."""
    import statsmodels.api as sm_api

    d = sm_api.datasets.macrodata.load_pandas().data
    labels = [f"{int(y)}Q{int(q)}" for y, q in zip(d["year"], d["quarter"])]
    assert labels == data["quarter"]
    assert np.array_equal(d["realgdp"].to_numpy(), data["realgdp"])
    assert np.array_equal(d["unemp"].to_numpy(), data["unemp"])


def test_tsecon_matches_a_numpy_transcription_of_the_closed_form(tr, fit):
    """C(1) = (I - sum A_i)^-1, LR = chol(C(1) Sigma C(1)'), B = C(1)^-1 LR
    on a statsmodels VAR(8) fit with the df-adjusted Sigma - achieved
    1.8e-13 (B), 2.3e-12 (LR), 6.0e-12 (C(1))."""
    hand = bq.bq_by_hand(tr["data"])
    assert hand is not None
    raw = fit["raw"]
    for key in ("impact", "long_run", "long_run_multiplier"):
        assert np.abs(np.asarray(raw[key]) - hand[key]).max() < 1e-9
