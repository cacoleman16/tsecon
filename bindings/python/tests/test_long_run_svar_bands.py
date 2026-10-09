"""Python contract tests for `long_run_svar_bands` (Blanchard-Quah long-run
SVAR confidence bands) and the binding side of `impact_signs`.

The numbers are pinned in the Rust golden (`long_run_bands_golden.rs`) against
`fixtures/long_run_svar_bands.json`, which `generate_long_run_svar_bands_fixtures.py`
builds without tsecon: an RNG replay of the documented bootstrap stream with
the Blanchard-Quah map re-run per draw (bootstrap arm, Kilian branch a
transcription of tsecon's own algorithm) and statsmodels' delta-method
building blocks with an analytic Jacobian (asymptotic arm). Here the same
fixture is reloaded through the binding at 1e-9, and the binding contract is
pinned: every returned key named on both doc surfaces, pandas input, the draw
shapes, and every refusal naming its parameter.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import tsecon

ROOT = Path(__file__).resolve().parents[1]
FX = json.loads((Path(__file__).resolve().parents[3] / "fixtures" / "long_run_svar_bands.json").read_text())
BQ = np.asarray(FX["bq_data"])
VAR3 = np.asarray(FX["var3_data"])
RTOL, ATOL = 1e-9, 1e-11


def _close(got, want, what):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, what
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL, err_msg=what)


def _kwargs(spec, method):
    kw = dict(
        lags=spec["lags"],
        horizon=spec["horizon"],
        alpha=spec["alpha"],
        normalize=spec["normalize"],
        cumulative=spec["cumulative"],
        method=method,
    )
    if spec.get("restrictions") is not None:
        kw["restrictions"] = [tuple(p) for p in spec["restrictions"]]
    if spec.get("impact_signs") is not None:
        kw["impact_signs"] = [tuple(p) for p in spec["impact_signs"]]
    if method == "bootstrap":
        kw.update(n_boot=spec["n_boot"], seed=spec["seed"], bias_correct=spec.get("bias_correct", False))
        if spec.get("band") == "sup-t":
            kw.update(band="sup-t", band_scope=spec["band_scope"])
    return kw


BOOT_CASES = [
    ("bq_var8", BQ),
    ("bq_var8_cum", BQ),
    ("bq_var8_bc", BQ),
    ("bq_var8_bc_cum", BQ),
    ("var3_perm", VAR3),
]
DELTA_CASES = [("bq_var8_delta", BQ), ("bq_var8_delta_cum", BQ), ("var3_perm_delta", VAR3)]


# --------------------------------------------------------------- goldens
@pytest.mark.parametrize("name, data", BOOT_CASES, ids=[c[0] for c in BOOT_CASES])
def test_bootstrap_arm_reloads_the_fixture(name, data):
    case = FX[name]
    r = tsecon.long_run_svar_bands(data, **_kwargs(case["spec"], "bootstrap"))
    for key in ("point", "se", "lower", "upper", "long_run_point", "long_run_se",
                "long_run_lower", "long_run_upper"):
        _close(r[key], case[key], f"{name}.{key}")
    assert r["n_unstable_draws"] == case["n_unstable_draws"]
    assert r["sign_anchor_mismatches"] == case.get("sign_anchor_mismatches")
    assert r["n_boot"] == case["spec"]["n_boot"]
    if "critical_value" in case:
        _close(r["critical_value"], case["critical_value"], f"{name}.critical_value")
        _close(r["sim_lower"], case["sim_lower"], f"{name}.sim_lower")
        _close(r["sim_upper"], case["sim_upper"], f"{name}.sim_upper")


@pytest.mark.parametrize("name, data", DELTA_CASES, ids=[c[0] for c in DELTA_CASES])
def test_asymptotic_arm_reloads_the_fixture(name, data):
    case = FX[name]
    r = tsecon.long_run_svar_bands(data, **_kwargs(case["spec"], "asymptotic"))
    for key in ("point", "se", "lower", "upper", "long_run_point", "long_run_se",
                "long_run_lower", "long_run_upper"):
        _close(r[key], case[key], f"{name}.{key}")
    assert r["n_boot"] is None and r["n_unstable_draws"] is None
    assert r["sign_anchor_mismatches"] is None


def test_var_irf_bands_kilian_case_reloads_the_fixture():
    case = FX["var_irf_bands_bc"]
    s = case["spec"]
    r = tsecon.var_irf_bands(
        np.asarray(FX["macro3_data"]), lags=s["lags"], horizon=s["horizon"], orth=True,
        method="bootstrap", alpha=s["alpha"], n_boot=s["n_boot"], seed=s["seed"],
        bias_correct=True,
    )
    for key in ("point", "se", "lower", "upper"):
        _close(r[key], case[key], f"var_irf_bands_bc.{key}")


# ------------------------------------------------------------- contract
def test_point_is_long_run_svar_and_the_imposed_zero_has_zero_width():
    kw = dict(lags=8, horizon=12, impact_signs=[(0, 1)])
    ref = tsecon.long_run_svar(BQ, **kw)
    for cumulative, key in ((False, "irf"), (True, "cumulative_irf")):
        for method in ("bootstrap", "asymptotic"):
            extra = {"n_boot": 50} if method == "bootstrap" else {}
            r = tsecon.long_run_svar_bands(BQ, method=method, cumulative=cumulative, **kw, **extra)
            assert np.array_equal(np.asarray(r["point"]), np.asarray(ref[key])), (method, key)
            assert np.array_equal(np.asarray(r["long_run_point"]), np.asarray(ref["long_run"]))
            assert r["long_run_se"][0][1] == 0.0
            assert r["long_run_lower"][0][1] == 0.0 == r["long_run_upper"][0][1]
            assert r["cumulative"] is cumulative and r["method"] == method
    assert ref["impact"][0][1] > 0.0      # the anchor: demand raises output on impact


def test_dataframe_input_and_draw_shapes():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(BQ, columns=["growth", "unemployment"])
    a = tsecon.long_run_svar_bands(df, lags=8, horizon=6, n_boot=30, impact_signs=[(0, 1)],
                                   return_draws=True)
    b = tsecon.long_run_svar_bands(BQ, lags=8, horizon=6, n_boot=30, impact_signs=[(0, 1)],
                                   return_draws=True)
    assert np.array_equal(a["irf_draws"], b["irf_draws"])
    assert isinstance(a["irf_draws"], np.ndarray) and a["irf_draws"].shape == (30, 7, 2, 2)
    assert a["long_run_draws"].shape == (30, 2, 2)
    # the anchor holds in every draw; the reported percentiles come from the draws
    assert np.all(a["irf_draws"][:, 0, 0, 1] >= 0.0)
    assert np.all(a["long_run_draws"][:, 0, 1] == 0.0)
    q = np.quantile(a["irf_draws"], [0.05, 0.95], axis=0)
    np.testing.assert_allclose(q[0], a["lower"], rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(q[1], a["upper"], rtol=1e-13, atol=1e-15)
    assert a["sign_anchor_mismatches"] == [0]
    assert a["min_root"] > 1.0 and a["n_unstable_draws"] == 0
    assert "irf_draws" not in tsecon.long_run_svar_bands(BQ, lags=8, horizon=2, n_boot=10)


def test_simultaneous_band_keys_on_both_arms():
    for method, extra in (("bootstrap", {"n_boot": 60}), ("asymptotic", {"band_n_sim": 5000})):
        r = tsecon.long_run_svar_bands(BQ, lags=8, horizon=8, method=method, band="sup-t",
                                       band_scope="shock", impact_signs=[(0, 1)], **extra)
        for key in ("sim_lower", "sim_upper", "critical_value", "pointwise_critical_value",
                    "n_cells", "n_cells_used"):
            assert key in r, (method, key)
        assert r["band"] == "sup-t" and r["band_scope"] == "shock" and r["n_cells"] == 18
        cv = np.asarray(r["critical_value"])
        assert np.all(cv >= r["pointwise_critical_value"])


def _doc_tokens(text):
    return set(re.findall(r"`+([A-Za-z_][A-Za-z_0-9]*)`+", text or ""))


def test_every_returned_key_is_named_on_both_doc_surfaces():
    keys = set()
    for kw in (
        dict(n_boot=20, return_draws=True, band="sup-t", impact_signs=[(0, 1)]),
        dict(method="asymptotic", band="bonferroni"),
        dict(n_boot=20, bias_correct=True, cumulative=True),
    ):
        keys |= set(tsecon.long_run_svar_bands(BQ, lags=8, horizon=4, **kw))
    runtime = _doc_tokens(tsecon.long_run_svar_bands.__doc__)
    stub = (ROOT / "python" / "tsecon" / "__init__.pyi").read_text(encoding="utf-8")
    body = re.search(r"def long_run_svar_bands\(.*?\"\"\"(.*?)\"\"\"", stub, re.S).group(1)
    for surface, tokens in (("runtime __doc__", runtime), ("stub", _doc_tokens(body))):
        missing = sorted(keys - tokens)
        assert not missing, f"long_run_svar_bands {surface} does not name {missing}"
    assert len(keys) == 26


# ------------------------------------------------------------- refusals
REFUSALS = [
    ({"alpha": 0.0}, "alpha"),
    ({"alpha": 1.5}, "alpha"),
    ({"alpha": float("nan")}, "alpha"),
    ({"method": "wild"}, "method"),
    ({"normalize": "unit"}, "normalize"),
    ({"trend": "ct"}, "trend"),
    ({"band": "supt-ish"}, "band"),
    ({"band_scope": "everything"}, "band_scope"),
    ({"n_boot": 1}, "n_boot"),
    ({"n_boot": 0}, "n_boot"),
    ({"lags": 0}, "lags"),
    ({"impact_signs": [(2, 0)]}, "impact_signs"),
    ({"impact_signs": [(0, 5)]}, "impact_signs"),
    ({"impact_signs": [(0, 1), (1, 1)]}, "impact_signs"),
    ({"restrictions": [(0, 0), (0, 1)]}, "restrictions"),
    ({"restrictions": [(1, 0)]}, "restrictions"),
    ({"restrictions": [(3, 0)]}, "restrictions"),
    ({"method": "asymptotic", "bias_correct": True}, "bias_correct"),
    ({"method": "asymptotic", "band": "sup-t", "band_n_sim": 1}, "band_n_sim"),
    # inert keywords
    ({"method": "asymptotic", "n_boot": 500}, "n_boot"),
    ({"method": "asymptotic", "seed": 3}, "seed"),
    ({"method": "asymptotic", "return_draws": True}, "return_draws"),
    ({"band_seed": 1}, "band_seed"),
    ({"band_n_sim": 5000}, "band_n_sim"),
    ({"band": "sup-t", "band_seed": 1}, "band_seed"),
    ({"method": "asymptotic", "band": "sidak", "band_n_sim": 5000}, "band_n_sim"),
    ({"band_scope": "all"}, "band_scope"),
    # budgets (checked before anything is allocated)
    ({"n_boot": 2**30, "horizon": 40}, "n_boot"),
    ({"n_boot": 2**30, "horizon": 40}, "horizon"),
    ({"n_boot": 100, "horizon": 10**8}, "horizon"),
    ({"method": "asymptotic", "band": "sidak", "band_scope": "all", "horizon": 10**5}, "band_scope"),
    ({"method": "asymptotic", "band": "sidak", "band_scope": "all", "horizon": 10**5}, "horizon"),
]


@pytest.mark.parametrize("kw, name", REFUSALS, ids=[f"{n}:{sorted(k)}" for k, n in REFUSALS])
def test_every_refusal_names_its_parameter(kw, name):
    import time

    base = dict(lags=8, horizon=4, n_boot=20)
    if kw.get("method") == "asymptotic":
        base.pop("n_boot")
    base.update(kw)
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as info:
        tsecon.long_run_svar_bands(BQ, **base)
    assert time.perf_counter() - t0 < 5.0
    assert re.search(rf"\b{name}\b", str(info.value)), str(info.value)


def test_malformed_data_is_refused_not_a_panic():
    for bad in (np.empty((0, 2)), BQ[:, :1], BQ[:5], np.where(np.arange(BQ.size).reshape(BQ.shape) == 50, np.nan, BQ),
                np.where(np.arange(BQ.size).reshape(BQ.shape) == 50, np.inf, BQ)):
        with pytest.raises((ValueError, TypeError)):
            tsecon.long_run_svar_bands(bad, lags=2, horizon=4, n_boot=10)
    with pytest.raises((ValueError, TypeError)):
        tsecon.long_run_svar_bands(BQ[:, 0], lags=2)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        tsecon.long_run_svar_bands(BQ, lags=2, n_boot=-1)


def test_near_unit_root_sample_reports_min_root_and_unstable_draws():
    """`min_root` is var_fit's (stable iff > 1, the inverse of a companion
    radius) and `n_unstable_draws` counts raw refits with radius >= 1. The
    exactly-singular I - A_1 - ... - A_p refusal (naming data and lags) is
    pinned in Rust, where the singular coefficients can be built directly."""
    rng = np.random.default_rng(5)
    y = np.zeros((80, 2))
    for t in range(1, 80):
        y[t] = np.array([[0.99, 0.0], [0.1, 0.5]]) @ y[t - 1] + rng.standard_normal(2)
    # Kilian's correction moves the bootstrap DGP towards the unit root, so
    # some raw refits come out explosive (they are counted, not dropped).
    r = tsecon.long_run_svar_bands(y, lags=1, horizon=4, n_boot=200, bias_correct=True)
    assert r["min_root"] == tsecon.var_fit(y, lags=1)["min_root"]
    assert 1.0 < r["min_root"] < 1.1
    assert 0 < r["n_unstable_draws"] < 200
    plain = tsecon.long_run_svar_bands(y, lags=1, horizon=4, n_boot=200)
    assert plain["n_unstable_draws"] < r["n_unstable_draws"]


# --------------------------------------------------- long_run_svar anchors
def test_impact_signs_flips_exactly_the_anchored_column():
    plain = tsecon.long_run_svar(BQ, lags=8, horizon=6)
    anch = tsecon.long_run_svar(BQ, lags=8, horizon=6, impact_signs=[(0, 1)])
    b0, b1 = np.asarray(plain["impact"]), np.asarray(anch["impact"])
    assert b0[0, 1] < 0.0 < b1[0, 1]                 # tsecon's LR normalization vs BQ's sign
    flip = np.array([1.0, -1.0])
    for key in ("impact", "long_run"):
        assert np.array_equal(np.asarray(anch[key]), np.asarray(plain[key]) * flip)
    assert np.array_equal(np.asarray(anch["irf"]), np.asarray(plain["irf"]) * flip)
    assert np.array_equal(np.asarray(anch["cumulative_irf"]), np.asarray(plain["cumulative_irf"]) * flip)
    assert np.array_equal(np.asarray(anch["fevd"]), np.asarray(plain["fevd"]))
    # an anchor that already holds is a no-op, bit for bit
    again = tsecon.long_run_svar(BQ, lags=8, horizon=6, impact_signs=[(0, 0)])
    assert np.array_equal(np.asarray(again["impact"]), b0)
    # B B' = Sigma is preserved by the relabelling
    sigma = np.asarray(tsecon.var_fit(BQ, lags=8)["sigma_u"])
    np.testing.assert_allclose(b1 @ b1.T, sigma, atol=1e-12)


@pytest.mark.parametrize("anchors", [[(2, 0)], [(0, 2)], [(0, 1), (1, 1)]])
def test_impact_signs_refusals_name_the_parameter(anchors):
    with pytest.raises(ValueError, match=r"\bimpact_signs\b"):
        tsecon.long_run_svar(BQ, lags=8, impact_signs=anchors)
