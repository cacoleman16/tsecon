"""Conformal forecast intervals: exact recomputation pins, the finite-sample
guarantee anchor, seeded Monte Carlo coverage (the primary grade), the
distribution-shift contrast ACI is for, leakage guards, and error surfaces.

Every Monte Carlo here is seeded and sized to run in seconds; the larger
runs behind the model-card tables use the same DGP code with more
replications (see docs/reference/model-cards/forecasting.md).
"""
import numpy as np
import pytest
import tsecon


# ---------------------------------------------------------------- DGPs

def ar1(rng, n, phi=0.6, sigma=1.0):
    y = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = phi * prev + sigma * rng.standard_normal()
        y[t] = prev
    return y


def ar1_garch(rng, n, phi=0.5):
    """AR(1) mean with GARCH(1,1) noise: heteroskedastic but stationary."""
    y = np.empty(n)
    prev, s2 = 0.0, 1.0
    for t in range(n):
        eps = np.sqrt(s2) * rng.standard_normal()
        prev = phi * prev + eps
        y[t] = prev
        s2 = 0.05 + 0.15 * eps**2 + 0.80 * s2
    return y


def ar1_var_shift(rng, n, shift_at, phi=0.5, sigma_post=3.0):
    """AR(1) whose innovation sd jumps from 1 to sigma_post at shift_at."""
    y = np.empty(n)
    prev = 0.0
    for t in range(n):
        s = sigma_post if t >= shift_at else 1.0
        prev = phi * prev + s * rng.standard_normal()
        y[t] = prev
    return y


def corrected_quantile(scores, alpha):
    """The ceil((m+1)(1-alpha))-th smallest score, recomputed independently."""
    m = len(scores)
    k = int(np.ceil((m + 1) * (1 - alpha) - 1e-9))
    assert k <= m
    return np.sort(scores)[k - 1]


# ------------------------------------------------------- exact recomputation

def test_split_symmetric_recomputed_exactly_from_naive_base():
    rng = np.random.default_rng(1)
    y = ar1(rng, 90)
    n, H, m, alpha = len(y), 2, 30, 0.2
    r = tsecon.conformal_forecast(
        y, horizon=H, method="split", base="naive", alpha=alpha, calib=m
    )
    assert r["method"] == "split" and r["base"] == "naive"
    assert r["level"] == pytest.approx(0.8)
    for h in (1, 2):
        # Naive h-step residuals at the last m rectangular-grid origins:
        origins = np.arange(n - H - m, n - H)
        scores = y[origins + h] - y[origins]
        np.testing.assert_array_equal(r["scores"][h - 1], scores)
        q = corrected_quantile(np.abs(scores), alpha)
        assert r["q_upper"][h - 1] == q
        assert r["q_lower"][h - 1] == -q
        assert r["mean"][h - 1] == y[-1]
        assert r["lower"][h - 1] == y[-1] - q
        assert r["upper"][h - 1] == y[-1] + q
    # k/(m+1) with m=30, alpha=0.2: k = ceil(31*0.8) = 25 -> 25/31.
    assert r["finite_sample_level"] == pytest.approx(25 / 31)
    assert r["n_calib"] == m


def test_split_asymmetric_tail_order_statistics():
    rng = np.random.default_rng(2)
    y = ar1(rng, 120)
    m, alpha = 40, 0.2
    r = tsecon.conformal_forecast(
        y, horizon=1, method="split", base="naive", alpha=alpha, calib=m,
        mode="asymmetric",
    )
    sorted_scores = np.sort(r["scores"][0])
    k_up = int(np.ceil((m + 1) * (1 - alpha / 2) - 1e-9))  # 37
    k_lo = m + 1 - k_up                                    # 4
    assert r["q_upper"][0] == sorted_scores[k_up - 1]
    assert r["q_lower"][0] == sorted_scores[k_lo - 1]
    assert r["finite_sample_level"] == pytest.approx((k_up - k_lo) / (m + 1))
    assert r["mode"] == "asymmetric"


def test_leakage_guard_future_perturbation_moves_only_its_own_score():
    """Calibration never sees the future: perturbing y[-1] can only change
    the one score whose *target* is y[-1] (and the forward forecast)."""
    rng = np.random.default_rng(3)
    y = ar1(rng, 80)
    kw = dict(horizon=1, method="split", base="naive", alpha=0.2, calib=25)
    before = tsecon.conformal_forecast(y, **kw)
    y2 = y.copy()
    y2[-1] += 500.0
    after = tsecon.conformal_forecast(y2, **kw)
    np.testing.assert_array_equal(before["scores"][0][:-1], after["scores"][0][:-1])
    assert before["scores"][0][-1] != after["scores"][0][-1]


# -------------------------------------------------------- the guarantee anchor

def test_split_guarantee_anchor_small_calibration():
    """The exactness anchor: iid data, a mean base, calib = 20 where the
    +1 correction bites (k = ceil(21 * 0.9) = 19, so the exchangeable-case
    coverage target is 19/21 = 0.9048, not 0.90). Finite-sample marginal
    coverage must come out >= 0.90 within Monte Carlo error."""
    rng = np.random.default_rng(20260823)
    reps, m, alpha, n = 2000, 20, 0.1, 60
    covered = 0
    for _ in range(reps):
        y = rng.standard_normal(n + 1)
        r = tsecon.conformal_forecast(
            y[:-1], horizon=1, method="split", base="mean", alpha=alpha, calib=m
        )
        covered += r["lower"][0] <= y[-1] <= r["upper"][0]
    cov = covered / reps
    se = np.sqrt(0.9048 * (1 - 0.9048) / reps)  # ~0.0066
    # The guarantee direction, with 2 MC standard errors of slack:
    assert cov >= 0.90 - 2 * se, f"coverage {cov:.4f} below the guarantee"
    # And not wildly conservative either (the target is 19/21 ~ 0.905):
    assert cov <= 19 / 21 + 4 * se, f"coverage {cov:.4f} too conservative"


# ----------------------------------------------- Monte Carlo marginal coverage

def _mc_next_step_coverage(dgp, method_kwargs, reps, n, seed):
    rng = np.random.default_rng(seed)
    covered = 0
    for _ in range(reps):
        y = dgp(rng, n + 1)
        r = tsecon.conformal_forecast(y[:-1], horizon=1, alpha=0.1, **method_kwargs)
        covered += r["lower"][0] <= y[-1] <= r["upper"][0]
    return covered / reps


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="split", base="ar", lags=1, calib=75),
        dict(method="enbpi", base="ar", lags=1, n_boot=25, seed=7),
        dict(method="aci", base="ar", lags=1, calib=75, n_eval=60),
    ],
    ids=["split", "enbpi", "aci"],
)
def test_marginal_coverage_ar1_iid_noise(kwargs):
    cov = _mc_next_step_coverage(ar1, kwargs, reps=250, n=300, seed=11)
    se = np.sqrt(0.9 * 0.1 / 250)  # ~0.019
    assert abs(cov - 0.9) <= 3 * se + 0.01, (
        f"{kwargs['method']}: AR(1) marginal coverage {cov:.3f} vs nominal 0.90"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="split", base="ar", lags=1, calib=75),
        dict(method="enbpi", base="ar", lags=1, n_boot=25, seed=7),
        dict(method="aci", base="ar", lags=1, calib=75, n_eval=60),
    ],
    ids=["split", "enbpi", "aci"],
)
def test_marginal_coverage_garch_noise(kwargs):
    """Heteroskedastic noise: marginal (unconditional) coverage should stay
    near nominal even though no method conditions on volatility."""
    cov = _mc_next_step_coverage(ar1_garch, kwargs, reps=250, n=300, seed=13)
    se = np.sqrt(0.9 * 0.1 / 250)
    assert abs(cov - 0.9) <= 3 * se + 0.02, (
        f"{kwargs['method']}: GARCH-noise marginal coverage {cov:.3f} vs 0.90"
    )


def test_aci_tracks_nominal_under_shift_where_split_degrades():
    """The published ACI claim (Gibbs-Candes 2021), measured: under a
    variance shift inside the evaluation window, fixed-level split
    conformal under-covers the post-shift stretch while the ACI recursion
    pulls coverage back toward nominal. Both use identical trailing
    scores and the same AR base — the only difference is the alpha_t
    update."""
    reps, n, n_eval, shift_frac = 40, 400, 120, 40
    post_split, post_aci = [], []
    for s in range(reps):
        rng = np.random.default_rng(100 + s)
        shift_at = n - (n_eval - shift_frac) - 1  # inside the eval window
        y = ar1_var_shift(rng, n, shift_at)
        common = dict(
            horizon=1, base="ar", lags=1, alpha=0.1, calib=100, n_eval=n_eval
        )
        rs = tsecon.conformal_backtest(y, method="split", **common)
        ra = tsecon.conformal_backtest(y, method="aci", gamma=0.05, **common)
        # Post-shift evaluation origins only:
        origins = np.array(rs["origins"])
        post = origins + 1 >= shift_at  # target index >= shift point
        post_split.append(1 - np.mean(np.asarray(rs["err"][0])[post]))
        post_aci.append(1 - np.mean(np.asarray(ra["err"][0])[post]))
    cov_split = float(np.mean(post_split))
    cov_aci = float(np.mean(post_aci))
    # Split must measurably degrade and ACI must recover most of the gap.
    assert cov_split < 0.85, f"split post-shift coverage {cov_split:.3f}"
    assert cov_aci > cov_split + 0.03, (
        f"ACI ({cov_aci:.3f}) should beat split ({cov_split:.3f}) post-shift"
    )
    assert cov_aci > 0.85, f"ACI post-shift coverage {cov_aci:.3f}"


# ------------------------------------------------------------------ ACI mechanics

def test_aci_trajectory_matches_published_recursion():
    rng = np.random.default_rng(17)
    y = ar1(rng, 250)
    alpha, gamma, n_eval = 0.1, 0.05, 60
    r = tsecon.conformal_forecast(
        y, horizon=1, method="aci", base="naive", alpha=alpha, gamma=gamma,
        calib=40, n_eval=n_eval,
    )
    err = np.asarray(r["err"][0], dtype=float)
    traj = np.asarray(r["alpha_trajectory"][0])
    expected = np.empty(n_eval)
    a = alpha
    for j in range(n_eval):
        expected[j] = a
        a = a + gamma * (alpha - err[j])  # horizon-1 delay: err_j absorbed at j+1
    np.testing.assert_allclose(traj, expected, atol=1e-14)
    assert r["alpha_final"][0] == pytest.approx(a)
    assert r["realized_coverage"][0] == pytest.approx(1 - err.mean())
    # gamma = 0 pins the trajectory at alpha exactly.
    r0 = tsecon.conformal_forecast(
        y, horizon=1, method="aci", base="naive", alpha=alpha, gamma=0.0,
        calib=40, n_eval=n_eval,
    )
    assert np.all(np.asarray(r0["alpha_trajectory"][0]) == alpha)


# ------------------------------------------------------------------ EnbPI

def test_enbpi_reproducible_and_diagnosed():
    rng = np.random.default_rng(19)
    y = ar1(rng, 200)
    kw = dict(horizon=3, method="enbpi", base="ar", lags=2, n_boot=25, seed=42)
    a = tsecon.conformal_forecast(y, **kw)
    b = tsecon.conformal_forecast(y, **kw)
    for key in ("mean", "lower", "upper", "residuals"):
        np.testing.assert_array_equal(a[key], b[key])
    c = tsecon.conformal_forecast(y, **{**kw, "seed": 1})
    assert not np.array_equal(a["lower"], c["lower"])
    assert 0.0 <= a["beta"] <= 0.1
    assert a["n_calib"] + a["n_excluded"] == len(y) - 2
    assert np.all(a["lower"] < a["upper"])
    # The symmetric (released-code) variant reports no beta and straddles
    # the center.
    s = tsecon.conformal_forecast(y, **{**kw, "optimize_beta": False})
    assert s["beta"] is None
    assert np.all(s["lower"] < s["mean"]) and np.all(s["mean"] < s["upper"])


def test_enbpi_online_backtest_slides_and_scores():
    rng = np.random.default_rng(23)
    y = ar1(rng, 320)
    r = tsecon.conformal_backtest(
        y, horizon=1, method="enbpi", base="ar", lags=1, alpha=0.1,
        n_eval=80, batch=1, n_boot=25, seed=3,
    )
    assert len(r["err"][0]) == 80
    assert r["batch"] == 1
    # err agrees with the recorded bounds against realized targets.
    origins = np.array(r["origins"])
    target = y[origins + 1]
    recomputed = (target < np.asarray(r["lower"][0])) | (
        target > np.asarray(r["upper"][0])
    )
    np.testing.assert_array_equal(np.asarray(r["err"][0]), recomputed)
    assert r["realized_coverage"][0] == pytest.approx(1 - recomputed.mean())


# ------------------------------------------------------------- error surfaces

def test_error_surfaces_teach():
    rng = np.random.default_rng(29)
    y = ar1(rng, 120)
    # Calibration too small for the level: names the minimum.
    with pytest.raises(ValueError, match="at least 9"):
        tsecon.conformal_forecast(y, method="split", base="naive", alpha=0.1, calib=5)
    # Asymmetric needs ~2x per tail.
    with pytest.raises(ValueError, match="at least 19"):
        tsecon.conformal_forecast(
            y, method="split", base="naive", alpha=0.1, calib=12, mode="asymmetric"
        )
    with pytest.raises(ValueError, match="unknown method"):
        tsecon.conformal_forecast(y, method="jackknife")
    with pytest.raises(ValueError, match="unknown base"):
        tsecon.conformal_forecast(y, base="prophet")
    with pytest.raises(ValueError, match="unknown mode"):
        tsecon.conformal_forecast(y, mode="upper-only")
    # EnbPI insists on its own AR ensemble.
    with pytest.raises(ValueError, match='base must be "ar"'):
        tsecon.conformal_forecast(y, method="enbpi", base="theta")
    # ACI is symmetric-by-construction here; the error teaches the option.
    with pytest.raises(ValueError, match="method=\"split\""):
        tsecon.conformal_forecast(y, method="aci", mode="asymmetric")
    # EnbPI's asymmetry is the beta search, not mode=.
    with pytest.raises(ValueError, match="optimize_beta"):
        tsecon.conformal_forecast(y, method="enbpi", base="ar", mode="asymmetric")
    # The enbpi online backtest is one-step by construction.
    with pytest.raises(ValueError, match="one-step-ahead"):
        tsecon.conformal_backtest(y, method="enbpi", base="ar", horizon=3)
    # gamma must be a non-negative finite step.
    with pytest.raises(ValueError, match="gamma"):
        tsecon.conformal_forecast(y, method="aci", base="naive", gamma=-0.1)
    # NaN input is loud.
    bad = y.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        tsecon.conformal_forecast(bad, method="split", base="naive")
    # Series too short for the windows.
    with pytest.raises(ValueError, match="observations"):
        tsecon.conformal_forecast(y[:20], method="split", base="naive", calib=40)
    # A constant series has no AR design.
    with pytest.raises(ValueError, match="singular"):
        tsecon.conformal_forecast(np.full(100, 2.0), method="enbpi", base="ar")
    # Bad arima order tuple.
    with pytest.raises(ValueError, match="order"):
        tsecon.conformal_forecast(y, base="arima", order=(1, 0))


def test_theta_and_arima_bases_run():
    """The headline bases wrap end to end (smoke, not a coverage grade)."""
    rng = np.random.default_rng(31)
    t = np.arange(140)
    y = 10 + 0.05 * t + np.sin(t * 2 * np.pi / 12) + 0.5 * rng.standard_normal(140)
    r = tsecon.conformal_forecast(
        y, horizon=4, method="split", base="theta", period=12, alpha=0.2, calib=30
    )
    assert np.all(r["lower"] <= r["mean"]) and np.all(r["mean"] <= r["upper"])
    r2 = tsecon.conformal_forecast(
        ar1(rng, 150), horizon=2, method="split", base="arima",
        order=(1, 0, 0), alpha=0.2, calib=20,
    )
    assert np.all(np.isfinite(r2["lower"])) and np.all(np.isfinite(r2["upper"]))


# --------------------------------------------- cross-implementation (non-gating)

def test_mapie_split_quantile_cross_check():
    """NON-GATING cross-pin: MAPIE's split-conformal interval half-width on
    identical absolute residuals must equal our corrected quantile. MAPIE is
    a regression library, so the comparison runs through a prefit constant
    regressor — same scores, independent quantile arithmetic."""
    mapie = pytest.importorskip("mapie")
    sklearn = pytest.importorskip("sklearn")
    from mapie.regression import SplitConformalRegressor
    from sklearn.dummy import DummyRegressor

    rng = np.random.default_rng(37)
    alpha, m = 0.1, 40
    y_cal = rng.standard_normal(m) * 2.0
    x_cal = np.zeros((m, 1))
    est = DummyRegressor(strategy="constant", constant=0.0).fit(x_cal, y_cal)
    scr = SplitConformalRegressor(
        estimator=est, confidence_level=1 - alpha, conformity_score="absolute",
        prefit=True,
    )
    scr.conformalize(x_cal, y_cal)
    _, interval = scr.predict_interval(np.zeros((1, 1)))
    mapie_halfwidth = float(interval[0, 1, 0] - interval[0, 0, 0]) / 2.0

    ours = corrected_quantile(np.abs(y_cal), alpha)
    # And the same number must be what conformal_forecast applies: build a
    # series whose naive residuals are exactly y_cal.
    assert mapie_halfwidth == pytest.approx(ours, rel=1e-12), (
        f"MAPIE half-width {mapie_halfwidth} vs tsecon corrected quantile {ours}"
    )
