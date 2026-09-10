"""Golden and behavioral tests for the Koop-Pesaran-Potter GIRF bindings:
`var_girf` (linear VAR — the engine's exact reduction) and
`threshold_var_girf` (regime-dependent GIRFs of the threshold VAR).

Re-pins fixtures/girf.json through the Python surface (statsmodels'
Cholesky IRF and the Pesaran-Shin closed form at 1e-12; the TVAR
transcription at 1e-10 — see the generator header for the honest
grading), asserts exact equality with `tsecon.var_irf(orth=True)`, checks
seed determinism (in-process and across a fresh interpreter), the TVAR
result's shape / keys / regime split, and the teaching errors.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
GIRF = json.loads((FIX / "girf.json").read_text())
Y_LIN = np.array(GIRF["linear"]["series"])
Y_TV = np.array(GIRF["tvar"]["series"])
TV = GIRF["tvar"]


def _lin_id(c):
    return f"p{c['p']}-{c['trend']}-j{c['shock_var']}-s{c['size']}"


@pytest.mark.parametrize("case", GIRF["linear"]["cases"], ids=_lin_id)
def test_var_girf_orthogonal_matches_statsmodels_cholesky_irf(case):
    r = tsecon.var_girf(
        Y_LIN, p=case["p"], shock_var=case["shock_var"], size=case["size"],
        shock="orthogonal", horizon=case["horizon"], n_draws=1,
        antithetic=False, trend=case["trend"], seed=3,
    )
    np.testing.assert_allclose(r["girf"], case["girf_orthogonal"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["shock_vector"], case["shock_vector_orthogonal"],
                               rtol=0, atol=1e-12)
    assert r["n_histories"] == case["n_histories_all"]
    assert r["n_draws"] == 1 and r["n_effective_draws"] == 1
    assert r["shock"] == "orthogonal" and r["shock_var"] == case["shock_var"]
    # The whole across-history band collapses: the linear GIRF has no
    # history dependence.
    np.testing.assert_allclose(r["upper"], r["lower"], rtol=0, atol=1e-12)
    assert np.isnan(r["mc_se"]).all() and np.isnan(r["draw_sd"]).all()


@pytest.mark.parametrize("case", GIRF["linear"]["cases"], ids=_lin_id)
def test_var_girf_generalized_matches_pesaran_shin_closed_form(case):
    r = tsecon.var_girf(
        Y_LIN, p=case["p"], shock_var=case["shock_var"], size=case["size"],
        shock="generalized", horizon=case["horizon"], n_draws=1,
        antithetic=False, trend=case["trend"],
    )
    np.testing.assert_allclose(r["girf"], case["girf_generalized"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r["shock_vector"], case["shock_vector_generalized"],
                               rtol=0, atol=1e-12)
    assert r["shock_size_used"] == pytest.approx(
        case["shock_vector_generalized"][case["shock_var"]], abs=1e-12)


@pytest.mark.parametrize("case", GIRF["linear"]["cases"], ids=_lin_id)
def test_var_girf_orthogonal_equals_var_irf_exactly(case):
    """The documented identity: girf[h] == var_irf(orth=True)[h][:, j] * size."""
    irf = np.asarray(tsecon.var_irf(Y_LIN, lags=case["p"], horizon=case["horizon"],
                                    orth=True, trend=case["trend"]))
    expected = irf[:, :, case["shock_var"]] * case["size"]
    for n_draws, anti in [(1, False), (2, True), (6, True), (5, False)]:
        r = tsecon.var_girf(Y_LIN, p=case["p"], shock_var=case["shock_var"],
                            size=case["size"], horizon=case["horizon"],
                            n_draws=n_draws, antithetic=anti, trend=case["trend"],
                            seed=n_draws)
        np.testing.assert_allclose(np.asarray(r["girf"]), expected, rtol=0, atol=1e-12)
        if n_draws >= 2:
            np.testing.assert_allclose(r["draw_sd"], 0.0, rtol=0, atol=1e-12)
    assert r["shock_size_used"] == pytest.approx(
        irf[0, case["shock_var"], case["shock_var"]] * case["size"], abs=1e-12)


def test_var_girf_history_subsample_and_per_history_layout():
    r = tsecon.var_girf(Y_LIN, p=2, horizon=6, histories=12, seed=9)
    assert r["n_histories"] == 12
    per = np.asarray(r["per_history"])
    assert per.shape == (12, 7, 3)
    np.testing.assert_allclose(per, np.broadcast_to(np.asarray(r["girf"]), per.shape),
                               rtol=0, atol=1e-12)
    # A subsample larger than the sample is the whole sample, not an error.
    r_all = tsecon.var_girf(Y_LIN, p=2, horizon=6, histories=10_000)
    assert r_all["n_histories"] == Y_LIN.shape[0] - 2


# ------------------------------------------------------------ threshold VAR

def _tvar_call(case, **override):
    kw = dict(
        p=TV["p"], threshold_index=TV["threshold_index"], delays=TV["delays"],
        trim=TV["trim"], constant=TV["constant"], shock_var=case["shock_var"],
        size=case["size"], shock=case["shock"], horizon=case["horizon"],
        n_draws=case["n_draws"], seed=case["seed"], regime=case["regime"],
        histories=case["histories"], bands=tuple(case["bands"]),
        antithetic=case["antithetic"],
    )
    kw.update(override)
    return tsecon.threshold_var_girf(Y_TV, **kw)


def _tvar_id(c):
    return f"{c['shock']}-j{c['shock_var']}-s{c['size']}-{c['regime']}-h{c['histories']}"


@pytest.mark.parametrize("case", TV["cases"], ids=_tvar_id)
def test_threshold_var_girf_matches_numpy_engine_transcription(case):
    r = _tvar_call(case)
    assert r["threshold"] == pytest.approx(TV["fit"]["threshold"], rel=1e-12)
    assert r["delay"] == TV["fit"]["delay"]
    for key in ("girf", "lower", "upper", "per_history", "mc_se", "draw_sd",
                "draw_lower", "draw_upper", "shock_vector"):
        np.testing.assert_allclose(r[key], case["shock_by_regime"] if key == "shock_vector"
                                   else case[key], rtol=1e-10, atol=1e-10, err_msg=key)
    for key in ("girf_low_regime", "girf_high_regime"):
        if case[key] is None:
            assert r[key] is None
        else:
            np.testing.assert_allclose(r[key], case[key], rtol=1e-10, atol=1e-10, err_msg=key)
    assert r["history_regimes"] == case["history_regimes"]
    assert r["history_times"] == case["history_times"]
    for key in ("n_histories", "n_low_histories", "n_high_histories", "n_draws"):
        assert r[key] == case[key], key
    assert r["shock_size_used"] == pytest.approx(
        [case["shock_by_regime"][0][case["shock_var"]],
         case["shock_by_regime"][1][case["shock_var"]]], abs=1e-10)
    assert r["regime"] == case["regime"] and r["shock"] == case["shock"]


def test_threshold_var_girf_shape_keys_and_regime_split():
    r = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=8, n_draws=40, seed=1)
    k, hh = Y_TV.shape[1], 9
    assert np.asarray(r["girf"]).shape == (hh, k)
    assert np.asarray(r["per_history"]).shape == (r["n_histories"], hh, k)
    assert r["n_histories"] == Y_TV.shape[0] - 1
    assert r["n_low_histories"] + r["n_high_histories"] == r["n_histories"]
    assert sorted(set(r["history_regimes"])) == [0, 1]
    assert r["history_times"] == list(range(1, Y_TV.shape[0]))
    assert r["n_effective_draws"] == 20 and r["horizon"] == 8
    gl, gh = np.asarray(r["girf_low_regime"]), np.asarray(r["girf_high_regime"])
    mix = (gl * r["n_low_histories"] + gh * r["n_high_histories"]) / r["n_histories"]
    np.testing.assert_allclose(mix, r["girf"], rtol=0, atol=1e-12)
    assert np.all(np.asarray(r["lower"]) <= np.asarray(r["upper"]))
    assert np.isfinite(r["mc_se"]).all() and np.isfinite(r["draw_sd"]).all()
    # The regime-restricted calls reproduce the split of the "all" call
    # (same seed tree per history is NOT expected — histories are
    # re-indexed — but the regime labels and counts must agree).
    low = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=8, n_draws=40, seed=1,
                                    regime="low")
    high = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=8, n_draws=40, seed=1,
                                     regime="high")
    assert low["n_histories"] == r["n_low_histories"] and set(low["history_regimes"]) == {0}
    assert high["n_histories"] == r["n_high_histories"] and set(high["history_regimes"]) == {1}
    assert low["girf_high_regime"] is None and high["girf_low_regime"] is None
    np.testing.assert_allclose(low["girf_low_regime"], low["girf"], rtol=0, atol=0)
    # Regime dependence is visible: the two regime averages differ at
    # horizon 1 by far more than the Monte Carlo error.
    d = abs(gl[1, 0] - gh[1, 0])
    assert d > 5 * np.hypot(low["mc_se"][1][0], high["mc_se"][1][0])
    # The raw shock differs across regimes because the covariances do.
    assert len(r["shock_size_used"]) == 2 and np.asarray(r["shock_vector"]).shape == (2, k)


def test_threshold_var_girf_is_seed_deterministic_in_process_and_across_interpreters():
    a = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=6, n_draws=16, seed=7,
                                  histories=20)
    b = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=6, n_draws=16, seed=7,
                                  histories=20)
    assert a == b
    c = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=6, n_draws=16, seed=8,
                                  histories=20)
    assert c["girf"] != a["girf"]
    code = (
        "import json, numpy as np, tsecon, sys\n"
        f"y = np.array(json.load(open({str(FIX / 'girf.json')!r}))['tvar']['series'])\n"
        "r = tsecon.threshold_var_girf(y, p=1, trim=0.15, horizon=6, n_draws=16, seed=7,"
        " histories=20)\n"
        "print(json.dumps([r['girf'], r['history_times']]))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                         text=True, env={"RAYON_NUM_THREADS": "2", "PATH": ""})
    girf_child, times_child = json.loads(out.stdout)
    assert girf_child == a["girf"]
    assert times_child == a["history_times"]


def test_pandas_input_accepted():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(Y_TV, columns=["a", "b"])
    r = tsecon.threshold_var_girf(df, p=1, trim=0.15, horizon=4, n_draws=8, seed=2)
    r2 = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=4, n_draws=8, seed=2)
    assert r["girf"] == r2["girf"]
    v = tsecon.var_girf(pd.DataFrame(Y_LIN), p=1, horizon=3)
    assert np.asarray(v["girf"]).shape == (4, 3)


def test_teaching_errors():
    with pytest.raises(ValueError, match="unknown shock"):
        tsecon.var_girf(Y_LIN, p=1, shock="cholesky")
    with pytest.raises(ValueError, match="shock_var"):
        tsecon.var_girf(Y_LIN, p=1, shock_var=3)
    with pytest.raises(ValueError, match="n_draws"):
        tsecon.var_girf(Y_LIN, p=1, n_draws=3, antithetic=True)
    with pytest.raises(ValueError, match="n_draws"):
        tsecon.var_girf(Y_LIN, p=1, n_draws=0)
    with pytest.raises(ValueError, match="bands"):
        tsecon.var_girf(Y_LIN, p=1, bands=(0.9, 0.1))
    with pytest.raises(ValueError, match="horizon"):
        tsecon.var_girf(Y_LIN, p=1, horizon=2_000_000)
    with pytest.raises(ValueError, match="histories"):
        tsecon.var_girf(Y_LIN, p=1, histories=0)
    with pytest.raises(ValueError, match="size"):
        tsecon.var_girf(Y_LIN, p=1, size=float("nan"))
    with pytest.raises(ValueError, match="p >= 1"):
        tsecon.var_girf(Y_LIN, p=0)
    with pytest.raises(ValueError, match="unknown trend"):
        tsecon.var_girf(Y_LIN, p=1, trend="ct")
    with pytest.raises(ValueError, match="unknown regime"):
        tsecon.threshold_var_girf(Y_TV, p=1, regime="recession")
    with pytest.raises(ValueError, match="shock_var"):
        tsecon.threshold_var_girf(Y_TV, p=1, shock_var=2, n_draws=4)
    with pytest.raises(ValueError, match="histories"):
        tsecon.threshold_var_girf(Y_TV, p=1, histories=0, n_draws=4)
    with pytest.raises(ValueError, match="at least two series"):
        tsecon.threshold_var_girf(Y_TV[:, :1], p=1, n_draws=4)
    with pytest.raises(ValueError, match="non-finite"):
        bad = Y_TV.copy()
        bad[4, 0] = np.nan
        tsecon.threshold_var_girf(bad, p=1, n_draws=4)
    with pytest.raises(ValueError, match="n_draws"):
        tsecon.threshold_var_girf(Y_TV, p=1, n_draws=7, antithetic=True)
    # Absurd counts are refused before any allocation, as a ValueError.
    with pytest.raises(ValueError):
        tsecon.threshold_var_girf(Y_TV, p=1, n_draws=2**40)
    with pytest.raises(ValueError):
        tsecon.var_girf(Y_LIN, p=1, n_draws=2**63)


def test_docstrings_name_every_returned_key():
    import re

    def tokens(fn):
        return set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", fn.__doc__ or ""))

    r = tsecon.var_girf(Y_LIN, p=1, horizon=2)
    assert set(r) <= tokens(tsecon.var_girf), set(r) - tokens(tsecon.var_girf)
    r = tsecon.threshold_var_girf(Y_TV, p=1, trim=0.15, horizon=2, n_draws=4)
    assert set(r) <= tokens(tsecon.threshold_var_girf), set(r) - tokens(tsecon.threshold_var_girf)


def test_showcase_configuration_timing_is_reasonable(capsys):
    """The roadmap's speed-showcase configuration on the release wheel:
    T = 600, k = 3, 200 histories x 500 antithetic draws x horizon 20 (two
    paths each, 4.2M simulated periods). The printed number is the
    indicative single-machine timing quoted in the model card; run with
    `-s` to see it."""
    import time

    rng = np.random.default_rng(20260912)
    c_low = np.array([0.6, 0.2, 0.1]); a_low = np.array([[0.7, 0.1, 0.0], [0.2, 0.5, 0.1], [0.0, 0.1, 0.6]])
    c_high = np.array([-0.6, -0.2, -0.1]); a_high = np.array([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.1, 0.0, 0.3]])
    l_low = np.linalg.cholesky([[0.25, 0.05, 0.0], [0.05, 0.25, 0.05], [0.0, 0.05, 0.25]])
    l_high = np.linalg.cholesky([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    y = np.zeros((800, 3))
    for t in range(1, 800):
        if y[t - 1, 0] <= 0.0:
            y[t] = c_low + a_low @ y[t - 1] + l_low @ rng.standard_normal(3)
        else:
            y[t] = c_high + a_high @ y[t - 1] + l_high @ rng.standard_normal(3)
    y = y[200:]
    t0 = time.perf_counter()
    r = tsecon.threshold_var_girf(y, p=1, horizon=20, n_draws=500, seed=42, histories=200)
    elapsed = time.perf_counter() - t0
    assert r["n_histories"] == 200 and r["n_draws"] == 500
    with capsys.disabled():
        print(f"\nthreshold_var_girf showcase (T=600, k=3, 200 histories x 500 antithetic "
              f"draws x horizon 20, fit included): {elapsed:.3f} s")
    assert elapsed < 30.0


def test_guide_chapter_13_example_numbers():
    """The printed block in docs/guide/13-nonlinear-dynamics.md (GIRF
    section) is produced by exactly this code; the numbers there are
    reproduced here to the printed precision so the chapter cannot drift."""
    rng = np.random.default_rng(0)
    A_low, A_high = np.array([[0.8, 0.1], [0.1, 0.7]]), np.array([[0.2, 0.0], [0.0, 0.3]])
    y = np.zeros((600, 2))
    for t in range(1, 600):
        A = A_low if y[t - 1, 0] <= 0.0 else A_high
        y[t] = A @ y[t - 1] + 0.6 * rng.standard_normal(2)
    y = y[200:]

    kw = dict(p=1, shock_var=0, horizon=8, n_draws=500, seed=0)
    girf = tsecon.threshold_var_girf(y, size=1.0, **kw)
    lin = np.asarray(tsecon.var_girf(y, p=1, shock_var=0, horizon=8)["girf"])
    low, high = np.asarray(girf["girf_low_regime"]), np.asarray(girf["girf_high_regime"])
    assert f"{girf['threshold']:+.3f}" == "-0.114"
    assert (girf["n_histories"], girf["n_low_histories"], girf["n_high_histories"]) == (399, 288, 111)
    assert [f"{v:.3f}" for v in girf["shock_size_used"]] == ["0.597", "0.559"]
    printed = [
        ("+0.597", "+0.559", "+0.605"), ("+0.420", "+0.181", "+0.434"),
        ("+0.320", "+0.112", "+0.320"), ("+0.251", "+0.080", "+0.241"),
        ("+0.200", "+0.061", "+0.184"), ("+0.162", "+0.048", "+0.142"),
        ("+0.132", "+0.038", "+0.111"), ("+0.108", "+0.031", "+0.087"),
        ("+0.088", "+0.025", "+0.068"),
    ]
    for h, row in enumerate(printed):
        assert (f"{low[h, 0]:+.3f}", f"{high[h, 0]:+.3f}", f"{lin[h, 0]:+.3f}") == row, h
    neg = np.asarray(tsecon.threshold_var_girf(y, size=-1.0, **kw)["girf"])
    big = np.asarray(tsecon.threshold_var_girf(y, size=3.0, **kw)["girf"])
    g = np.asarray(girf["girf"])
    line = "h=2  GIRF(+1) %+.3f  -GIRF(-1) %+.3f  GIRF(+3)/3 %+.3f  MC se %.4f" % (
        g[2, 0], -neg[2, 0], big[2, 0] / 3, girf["mc_se"][2][0])
    assert line == "h=2  GIRF(+1) +0.262  -GIRF(-1) +0.336  GIRF(+3)/3 +0.179  MC se 0.0003"


def test_var_svar_card_example_prints_what_it_says():
    """The `var_girf` example in docs/reference/model-cards/var-svar.md."""
    rng = np.random.default_rng(0)
    k, n = 3, 400
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + 0.3 * rng.standard_normal(k)
    irf = np.asarray(tsecon.var_irf(Y, lags=2, horizon=8, orth=True))
    g = tsecon.var_girf(Y, p=2, shock_var=1, shock="orthogonal", horizon=8)
    assert np.abs(np.asarray(g["girf"]) - irf[:, :, 1]).max() < 1e-12
    assert (np.asarray(g["upper"]).shape, g["n_histories"]) == ((9, 3), 398)
