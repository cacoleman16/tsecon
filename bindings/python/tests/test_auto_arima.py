"""auto_arima: determinism, trace/refit consistency, error surfaces,
order recovery (CI-sane thresholds; the full MC study lives in
scripts/mc_auto_arima_recovery.py and its rates in the model card), and
a bundled-dataset seasonal smoke on monthly CO2.

The selection loop deliberately has no third-party parity gate — R's
auto.arima and pmdarima disagree with each other on real series — so
these tests pin the loop's *contract*: deterministic, internally
consistent (every reported number is reproducible by refitting the
reported orders through arima_fit), teaching errors, and
known-order recovery on simulated DGPs.
"""
import numpy as np
import pytest

import tsecon


def simulate_arma(rng, n, ar=(), ma=(), sigma=1.0, constant=0.0, burn=300):
    """y_t = c + sum phi_i y_(t-i) + e_t + sum theta_j e_(t-j)."""
    total = n + burn
    e = sigma * rng.standard_normal(total)
    y = np.zeros(total)
    for t in range(total):
        v = constant + e[t]
        for i, phi in enumerate(ar):
            if t > i:
                v += phi * y[t - 1 - i]
        for j, th in enumerate(ma):
            if t > j:
                v += th * e[t - 1 - j]
        y[t] = v
    return y[burn:]


def trace_tuple(r):
    return [
        (t["order"], t["seasonal_order"], t["constant"], t["ic"], t["status"])
        for t in r["trace"]
    ]


def test_deterministic_and_trace_refit_consistent():
    rng = np.random.default_rng(42)
    y = simulate_arma(rng, 300, ar=[0.7])

    r1 = tsecon.auto_arima(y)
    r2 = tsecon.auto_arima(y)

    # Determinism: identical selection, parameters, and full trace.
    assert r1["order"] == r2["order"]
    assert r1["seasonal_order"] == r2["seasonal_order"]
    assert r1["constant"] == r2["constant"]
    assert r1["ic_value"] == r2["ic_value"]
    np.testing.assert_array_equal(r1["params"], r2["params"])
    assert trace_tuple(r1) == trace_tuple(r2)

    # The reported best is the argmin over the eligible trace entries.
    ok_ics = [t["ic"] for t in r1["trace"] if t["status"] == "ok"]
    assert min(ok_ics) == r1["ic_value"]
    assert r1["n_models"] == len(r1["trace"])
    assert not r1["budget_exhausted"]

    # Refitting the selected orders through arima_fit reproduces the
    # reported fit exactly (same deterministic code path).
    p, d, q = r1["order"]
    P, D, Q, s = r1["seasonal_order"]
    refit = tsecon.arima_fit(
        y,
        p=p,
        d=d,
        q=q,
        seasonal=(P, D, Q, s) if s >= 2 else None,
        constant=r1["constant"],
    )
    assert refit["loglik"] == r1["loglik"]
    assert refit["aic"] == r1["aic"]
    np.testing.assert_array_equal(refit["params"], r1["params"])

    # The reported AICc is the aic-based formula at the shared (n, k).
    k = len(r1["params"])
    n_eff = len(y) - d - D * s
    aicc = r1["aic"] + 2 * k * (k + 1) / (n_eff - k - 1)
    assert r1["aicc"] == pytest.approx(aicc, rel=1e-12)
    assert r1["ic"] == "aicc"
    assert r1["ic_value"] == pytest.approx(r1["aicc"], rel=1e-12)


def test_result_keys_are_a_superset_of_arima_fit():
    rng = np.random.default_rng(1)
    y = simulate_arma(rng, 200, ar=[0.5])
    fit = tsecon.arima_fit(y, p=1, d=0, q=0, forecast_steps=4, conf_alpha=0.10)
    auto = tsecon.auto_arima(y, d=0, forecast_steps=4, conf_alpha=0.10)
    missing = (set(fit) - {"drift_uncertainty"}) - set(auto)
    assert not missing, f"auto_arima result lacks arima_fit keys: {missing}"
    for key in (
        "order",
        "seasonal_order",
        "constant",
        "converged",
        "ic",
        "ic_value",
        "aicc",
        "stepwise",
        "n_models",
        "budget_exhausted",
        "trace",
        "d_test",
        "D_test",
        "interpretation",
    ):
        assert key in auto, f"missing selection key {key}"
    assert len(auto["forecast_mean"]) == 4
    assert len(auto["forecast_lower"]) == 4
    # Fixed d skips the KPSS run and says so.
    assert auto["d_test"] is None
    assert auto["order"][1] == 0
    # Non-seasonal search: no nsdiffs run either.
    assert auto["D_test"] is None


def test_d_selection_on_a_random_walk_with_evidence():
    rng = np.random.default_rng(7)
    y = np.cumsum(rng.standard_normal(250))
    r = tsecon.auto_arima(y)
    assert r["order"][1] == 1, r["interpretation"]
    ev = r["d_test"]
    assert ev is not None and ev["d"] == 1 and ev["test"] == "kpss"
    assert ev["steps"][0]["needs_differencing"] is True
    assert ev["steps"][1]["needs_differencing"] is False


def test_ic_variants_and_bic_prefers_smaller():
    rng = np.random.default_rng(3)
    y = simulate_arma(rng, 240, ar=[0.6])
    r_aicc = tsecon.auto_arima(y, d=0)
    r_bic = tsecon.auto_arima(y, d=0, ic="bic")
    assert r_aicc["ic"] == "aicc" and r_bic["ic"] == "bic"
    # BIC penalizes harder, so its selected model is never larger.
    size = lambda r: r["order"][0] + r["order"][2]  # noqa: E731
    assert size(r_bic) <= size(r_aicc)


def test_error_surfaces_teach():
    rng = np.random.default_rng(0)
    y = simulate_arma(rng, 120, ar=[0.5])

    with pytest.raises(ValueError, match="seasonal_period"):
        tsecon.auto_arima(y, seasonal_period=1)
    with pytest.raises(ValueError, match="aicc"):
        tsecon.auto_arima(y, ic="aikaike")
    with pytest.raises(ValueError, match="forecast_steps"):
        tsecon.auto_arima(y, conf_alpha=0.05)
    with pytest.raises(ValueError, match="max_p"):
        tsecon.auto_arima(y, max_p=40)
    with pytest.raises(ValueError, match="stepwise"):
        tsecon.auto_arima(
            y,
            stepwise=False,
            seasonal_period=4,
            max_p=12,
            max_q=12,
            max_P=6,
            max_Q=6,
            max_order=36,
        )
    with pytest.raises(ValueError, match="seasonal_period"):
        tsecon.auto_arima(y, D=1)  # fixed D without a period

    bad = y.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError, match="index 5"):
        tsecon.auto_arima(bad)

    with pytest.raises(ValueError, match="auto_arima"):
        tsecon.auto_arima(y[:3])


def test_recovery_small_mc_nonseasonal():
    """CI-sane recovery gate: loose thresholds well below the measured
    MC rates (scripts/mc_auto_arima_recovery.py; model card quotes the
    full numbers), so this fails on regression, not on noise.

    Within-one = d exact and each of (p, q) within +-1 of truth.
    """
    cases = [
        ((1, 0, 0), dict(ar=[0.6])),
        ((0, 0, 1), dict(ma=[0.6])),
        ((1, 0, 1), dict(ar=[0.5], ma=[0.4])),
    ]
    reps = 12
    for truth, kw in cases:
        exact = within = 0
        for rep in range(reps):
            rng = np.random.default_rng(1000 + 17 * rep)
            y = simulate_arma(rng, 300, **kw)
            r = tsecon.auto_arima(y)
            p, d, q = r["order"]
            if (p, d, q) == truth:
                exact += 1
            if (
                d == truth[1]
                and abs(p - truth[0]) <= 1
                and abs(q - truth[2]) <= 1
            ):
                within += 1
        assert within >= 0.5 * reps, (
            f"truth {truth}: within-one {within}/{reps} — selection regressed"
        )


def test_seasonal_co2_monthly_selects_a_seasonal_model():
    """Bundled-dataset smoke: monthly CO2 (statsmodels' public-domain
    Mauna Loa series, resampled weekly -> monthly) with
    seasonal_period=12 must pick a model with a seasonal part, driven by
    the seasonal-strength evidence. Caps are tightened (max_p=max_q=2,
    max_P=max_Q=1) to keep the state dimension small — this is a smoke
    test of the seasonal plumbing, not a horse race.
    """
    sm = pytest.importorskip("statsmodels.api")
    data = sm.datasets.co2.load_pandas().data
    co2 = data["co2"].resample("MS").mean().ffill()
    y = co2.to_numpy()[-240:]  # the last 20 years

    r = tsecon.auto_arima(
        y, seasonal_period=12, max_p=2, max_q=2, max_P=1, max_Q=1
    )
    P, D, Q, s = r["seasonal_order"]
    assert s == 12
    assert D + P + Q >= 1, f"no seasonal part selected: {r['interpretation']}"
    ev = r["D_test"]
    assert ev is not None and ev["period"] == 12
    assert ev["d"] == D
    # CO2's seasonal cycle is unmistakable: the strength rule must call
    # for the seasonal difference (measured strength ~0.99 at D = 0).
    assert D == 1
    assert ev["steps"][0]["seasonal_strength"] > 0.9
    # After D = 1 the linear growth is a drift, not a trend: d = 0 with a
    # constant is the classic answer here, so only sanity-check d.
    assert r["order"][1] in (0, 1)
    # The winner's criterion is the trace minimum, seasonal case included.
    ok_ics = [t["ic"] for t in r["trace"] if t["status"] == "ok"]
    assert min(ok_ics) == r["ic_value"]


def test_summarize_wraps_the_result():
    rng = np.random.default_rng(5)
    y = simulate_arma(rng, 150, ar=[0.5])
    r = tsecon.auto_arima(y, d=0, max_p=2, max_q=1)
    text = str(tsecon.summarize(r).summary())
    assert "order" in text
