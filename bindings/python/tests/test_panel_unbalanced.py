"""The observation mask for unbalanced panels through the Python surface.

* linearmodels parity: re-pins fixtures/panel_unbalanced.json (PanelOLS 7.0
  on the Arellano-Bond EmplUK panel and on a seeded panel with entry, exit
  and internal gaps — see the generator header) through `panel_fe`,
  `panel_distributed_lag` and `panel_lp` with `mask=`;
* bit-identity: every balanced-panel call of `panel_fe`,
  `panel_distributed_lag`, `panel_lp`, `lp_did` and `mean_group_var` is
  float-hex identical to the 0.9.0 build captured in
  fixtures/panel_balanced_snapshot.json before the mask landed, and a mask
  that is 1 everywhere is bit-identical to no mask;
* the refusals name `mask` (dimensions, non-0/1 flags, NaN in an observed
  cell, LP-DiD and the half-panel jackknives on an unbalanced panel) and
  `entities` (an internal gap in the mean-group VAR).
"""
import json
import sys
import platform
import math
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
UNB = json.loads((FIX / "panel_unbalanced.json").read_text())
SNAP = json.loads((FIX / "panel_balanced_snapshot.json").read_text())
SE = {"nonrobust": ("nonrobust", None), "cluster_entity": ("cluster", None), "driscoll_kraay": ("driscoll_kraay", 4.0)}


def _arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows])


def _dataset(name):
    ds = UNB["datasets"][name]
    return ds, np.array(ds["mask"], dtype=float)


def _hexes(seq):
    return [float(v).hex() for v in np.asarray(seq, dtype=float).ravel()]


# The snapshot is bitwise only on the architecture that captured it. arm64
# contracts a multiply and an add into one FMA where x86-64 rounds twice, so
# the same least-squares chain lands a few bits away on an Apple-silicon
# runner — measured at 1 ULP on the plain within estimator and 11 ULP on the
# two-way quadratic distributed lag, which projects out time dummies before
# it fits. That is a property of the instruction set, not of the mask
# refactor, which is pinned bitwise on EVERY platform by
# test_all_ones_mask_is_bit_identical_to_no_mask below (two calls, one
# process).
#
# So off the capturing architecture this asserts a relative tolerance rather
# than a ULP count: a ULP budget is the wrong shape here, because the derived
# quantities (marginal effects, a turning point that divides by a quadratic
# coefficient) amplify a last-bit difference by however ill-conditioned the
# ratio is, while the tolerance that matters is the same for all of them.
# 1e-12 sits three orders above the 1e-15 reassociation noise and six below
# the 1e-6 a real change in the balanced path would move — skipping a cell,
# a different degrees-of-freedom count or a changed projection all move the
# answer far more than this. The `_probe` tests below pin both ends of that.
_SNAP_ARCH = SNAP.get("_platform", "x86_64-linux")
_THIS_ARCH = f"{platform.machine()}-{sys.platform}"
_SNAP_IS_BITWISE = _THIS_ARCH == _SNAP_ARCH
_CROSS_ARCH_RTOL = 1e-12


def _same_cross_arch(seq, want, what):
    """`_same`'s cross-architecture branch, exercised on every platform."""
    global _SNAP_IS_BITWISE
    was = _SNAP_IS_BITWISE
    _SNAP_IS_BITWISE = False
    try:
        _same(seq, want, what)
    finally:
        _SNAP_IS_BITWISE = was


def _same(seq, want, what):
    """Bitwise on the snapshot's architecture, else within `_CROSS_ARCH_RTOL`."""
    got = _hexes(seq)
    if _SNAP_IS_BITWISE:
        assert got == want, what
        return
    assert len(got) == len(want), f"{what}: {len(got)} values against {len(want)}"
    for i, (g, w) in enumerate(zip(got, want)):
        gv, wv = float.fromhex(g), float.fromhex(w)
        if gv == wv:
            continue
        scale = max(abs(gv), abs(wv))
        rel = abs(gv - wv) / scale if scale else abs(gv - wv)
        assert rel <= _CROSS_ARCH_RTOL, (
            f"{what}[{i}]: {gv!r} against the snapshot's {wv!r} — "
            f"{rel:.2e} relative on {_THIS_ARCH}, past the "
            f"{_CROSS_ARCH_RTOL:.0e} the snapshot's {_SNAP_ARCH} capture allows"
        )


# --------------------------------------------------------------- parity

@pytest.mark.parametrize("key", sorted(SE))
@pytest.mark.parametrize("case", UNB["fe_cases"], ids=lambda c: c["name"])
def test_panel_fe_and_dl_match_linearmodels_on_unbalanced_panels(case, key):
    ds, mask = _dataset(case["dataset"])
    y = _arr(ds[case["outcome"]])
    x = np.array([_arr(ds[n]) for n in case["regressors"]])
    se_type, bw = SE[key]
    want = case["fits"][key]
    if case["entity_effects"] and not case["time_effects"] and not case["entity_trends"]:
        kw = {"se_type": se_type, "mask": mask}
        if bw is not None:
            kw["bandwidth"] = bw
        r = tsecon.panel_fe(y, x, **kw)
        np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10)
        np.testing.assert_allclose(r["bse"], want["bse"], rtol=1e-10)
        np.testing.assert_allclose(r["tvalues"], want["tstats"], rtol=1e-10)
    # Every effects menu is reachable through the distributed-lag surface at lags=0.
    kw = dict(lags=0, powers=1, entity_effects=case["entity_effects"], time_effects=case["time_effects"],
              entity_trends=case["entity_trends"], se_type=se_type, mask=mask)
    if bw is not None:
        kw["bandwidth"] = bw
    r = tsecon.panel_distributed_lag(y, x, **kw)
    assert r["nobs"] == want["nobs"] == ds["n_obs"]
    assert r["df_resid"] == want["df_resid"]
    np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10)
    np.testing.assert_allclose(r["bse"], want["bse"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(r["cov"]), want["cov"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("key", sorted(SE))
@pytest.mark.parametrize("case", UNB["dl_cases"], ids=lambda c: c["name"])
def test_panel_distributed_lag_matches_linearmodels_on_unbalanced_panels(case, key):
    ds, mask = _dataset(case["dataset"])
    y = _arr(ds[case["outcome"]])
    x = np.array([_arr(ds[n]) for n in case["regressors"]])
    se_type, bw = SE[key]
    kw = dict(lags=case["lags"], powers=case["powers"], entity_effects=case["entity_effects"],
              time_effects=case["time_effects"], entity_trends=case["entity_trends"], se_type=se_type, mask=mask)
    if bw is not None:
        kw["bandwidth"] = bw
    if isinstance(case["eval_points"], list):
        kw["eval_points"] = np.array(case["eval_points"])
    r = tsecon.panel_distributed_lag(y, x, **kw)
    want = case["fits"][key]
    assert r["nobs"] == want["nobs"] == case["n_obs_lagged"]
    assert r["df_resid"] == want["df_resid"]
    np.testing.assert_allclose(r["params"], want["params"], rtol=1e-10)
    np.testing.assert_allclose(r["bse"], want["bse"], rtol=1e-10)
    np.testing.assert_allclose(np.asarray(r["cov"]), want["cov"], rtol=1e-10, atol=1e-14)
    delta = case["delta"][key]
    np.testing.assert_allclose(r["cumulative_effect"][0], delta["cumulative_effect"], rtol=1e-10)
    np.testing.assert_allclose(r["cumulative_se"][0], delta["cumulative_se"], rtol=1e-10)
    if case["powers"] == 2:
        pts = case.get("eval_points_used", case["eval_points"])
        np.testing.assert_allclose(r["eval_points"][0], pts, rtol=1e-12)
        np.testing.assert_allclose(r["marginal_effect"][0], delta["marginal_effect"], rtol=1e-10)
        np.testing.assert_allclose(r["turning_point"][0], delta["turning_point"], rtol=1e-10)


@pytest.mark.parametrize("key", sorted(SE))
@pytest.mark.parametrize("case", UNB["lp_cases"], ids=lambda c: c["name"])
def test_panel_lp_matches_linearmodels_per_horizon_on_unbalanced_panels(case, key):
    ds, mask = _dataset("lp_synthetic")
    y = _arr(ds["y"])
    shock = np.array(ds["shock"])
    se_type, bw = SE[key]
    kw = dict(horizon=case["max_horizon"], n_lag_controls=case["n_lag_controls"], se_type=se_type,
              cumulative=case["cumulative"], mask=mask)
    if bw is not None:
        kw["bandwidth"] = bw
    r = tsecon.panel_lp(y, shock, **kw)
    for h, want in enumerate(case["horizons"]):
        assert r["nobs"][h] == want["nobs"]
        assert r["irf"][h] == pytest.approx(want["irf"][key], rel=1e-10)
        assert r["se"][h] == pytest.approx(want["se"][key], rel=1e-10)


def test_empluk_panel_is_the_arellano_bond_one():
    ds, mask = _dataset("empluk")
    assert (ds["n_entities"], ds["n_periods"], ds["n_obs"]) == (140, 9, 1031)
    counts = mask.sum(axis=1)
    assert (int((counts == 7).sum()), int((counts == 8).sum()), int((counts == 9).sum())) == (103, 23, 14)
    r = tsecon.panel_fe(_arr(ds["log_emp"]), np.array([_arr(ds["log_wage"]), _arr(ds["log_capital"])]), mask=mask)
    assert r["params"].shape == (2,)


# ---------------------------------------------------------- bit-identity

def _snap_inputs():
    i = SNAP["inputs"]
    return (np.array(i["y"]), np.array(i["temp"]), np.array(i["x2"]), np.array(i["shock"]),
            np.array(i["treat"]), [np.array(e) for e in i["entities"]])


def test_balanced_calls_are_bit_identical_to_the_0_9_0_snapshot():
    y, temp, x2, shock, treat, entities = _snap_inputs()
    for key, rec in SNAP["panel_fe"].items():
        r = tsecon.panel_fe(y, np.array([temp, x2]), **rec["kwargs"])
        for k in ("params", "bse", "tvalues"):
            _same(r[k], rec[k], f"panel_fe/{key}/{k}")
    for key, rec in SNAP["panel_distributed_lag"].items():
        kw = dict(rec["kwargs"])
        if "eval_points" in kw:
            kw["eval_points"] = np.array(kw["eval_points"])
        r = tsecon.panel_distributed_lag(y, np.array([temp]), **kw)
        _same(r["params"], rec["params"], f"panel_distributed_lag/{key}/params")
        _same(r["bse"], rec["bse"], f"panel_distributed_lag/{key}/bse")
        _same(np.asarray(r["cov"]), rec["cov"], f"panel_distributed_lag/{key}/cov")
        _same(r["cumulative_effect"], rec["cumulative_effect"], f"panel_distributed_lag/{key}/cumulative_effect")
        _same(r["cumulative_se"], rec["cumulative_se"], f"panel_distributed_lag/{key}/cumulative_se")
        assert (r["nobs"], r["df_resid"]) == (rec["nobs"], rec["df_resid"]), key
        if "marginal_effect" in rec:
            for k in ("marginal_effect", "marginal_se", "turning_point", "turning_point_se"):
                _same(r[k], rec[k], f"panel_distributed_lag/{key}/{k}")
    for key, rec in SNAP["panel_lp"].items():
        r = tsecon.panel_lp(y, shock, **rec["kwargs"])
        _same(r["irf"], rec["irf"], f"panel_lp/{key}/irf")
        _same(r["se"], rec["se"], f"panel_lp/{key}/se")
        assert [int(v) for v in r["nobs"]] == rec["nobs"], key
    for key, rec in SNAP["lp_did"].items():
        r = tsecon.lp_did(y, treat, **rec["kwargs"])
        _same(r["coef"], rec["coef"], f"lp_did/{key}/coef")
        _same(r["se"], rec["se"], f"lp_did/{key}/se")
        assert [int(v) for v in r["nobs"]] == rec["nobs"], key
        assert [int(v) for v in r["n_switchers"]] == rec["n_switchers"], key
        if "pooled_post" in rec:
            _same([r["pooled_post_att"], r["pooled_post_se"]], rec["pooled_post"], f"lp_did/{key}/pooled_post")
            _same([r["pooled_pre_att"], r["pooled_pre_se"]], rec["pooled_pre"], f"lp_did/{key}/pooled_pre")
    rec = SNAP["mean_group_var"]["lag1"]
    r = tsecon.mean_group_var(entities, **rec["kwargs"])
    _same(r["intercept"], rec["intercept"], "mean_group_var/intercept")
    _same(np.asarray(r["coefs"]), rec["coefs"], "mean_group_var/coefs")
    _same(np.asarray(r["orth_irfs"]), rec["orth_irfs"], "mean_group_var/orth_irfs")
    _same(r["irf_path_se"], rec["irf_path_se"], "mean_group_var/irf_path_se")


def test_probe_the_cross_architecture_tolerance_accepts_reassociation_noise():
    """The tolerance must pass what an instruction-set difference produces.

    The worst measured on arm64 is 11 ULP on the two-way quadratic
    distributed lag (1.3e-15 relative); this probes an order beyond it.
    """
    want = SNAP["panel_fe"]["nonrobust"]["params"]
    vals = [float.fromhex(h) for h in want]
    drifted = [v + 16 * math.copysign(math.ulp(v), v) for v in vals]
    _same_cross_arch(drifted, want, "probe/reassociation")


def test_probe_the_cross_architecture_tolerance_rejects_a_real_change():
    """And it must fail anything a real change in the balanced path moves.

    The smallest such change — one observation entering or leaving a
    projection — moves a coefficient by parts per million, six orders past
    this tolerance. A 1e-9 nudge already has to fail.
    """
    want = SNAP["panel_fe"]["nonrobust"]["params"]
    nudged = [float.fromhex(h) * (1 + 1e-9) for h in want]
    with pytest.raises(AssertionError, match=r"relative on .*past the 1e-12"):
        _same_cross_arch(nudged, want, "probe/regression")


def test_all_ones_mask_is_bit_identical_to_no_mask():
    y, temp, x2, shock, treat, _ = _snap_inputs()
    ones = np.ones_like(y)
    a = tsecon.panel_fe(y, np.array([temp, x2]), se_type="driscoll_kraay", bandwidth=3.0)
    b = tsecon.panel_fe(y, np.array([temp, x2]), se_type="driscoll_kraay", bandwidth=3.0, mask=ones)
    c = tsecon.panel_fe(y, np.array([temp, x2]), se_type="driscoll_kraay", bandwidth=3.0, mask=ones.astype(bool))
    for k in ("params", "bse", "tvalues"):
        assert _hexes(a[k]) == _hexes(b[k]) == _hexes(c[k])
    a = tsecon.panel_distributed_lag(y, np.array([temp]), lags=2, powers=2, entity_trends=True)
    b = tsecon.panel_distributed_lag(y, np.array([temp]), lags=2, powers=2, entity_trends=True, mask=ones)
    assert _hexes(np.asarray(a["cov"])) == _hexes(np.asarray(b["cov"]))
    assert _hexes(a["turning_point"]) == _hexes(b["turning_point"])
    a = tsecon.panel_lp(y, shock, horizon=3, bias_correction="spj", se_type="cluster")
    b = tsecon.panel_lp(y, shock, horizon=3, bias_correction="spj", se_type="cluster", mask=ones)
    assert _hexes(a["irf"]) == _hexes(b["irf"]) and _hexes(a["se"]) == _hexes(b["se"])
    a = tsecon.lp_did(y, treat, pre_window=3, post_window=4)
    b = tsecon.lp_did(y, treat, pre_window=3, post_window=4, mask=ones)
    assert _hexes(a["coef"]) == _hexes(b["coef"]) and _hexes(a["se"]) == _hexes(b["se"])


# --------------------------------------------------------------- refusals

def test_masked_out_cells_are_never_read():
    ds, mask = _dataset("synthetic")
    y = _arr(ds["y"])
    x = np.array([_arr(ds["x"])])
    a = tsecon.panel_fe(y, x, mask=mask)
    y2, x2 = y.copy(), x.copy()
    y2[mask == 0] = np.inf
    x2[0][mask == 0] = 1e300
    b = tsecon.panel_fe(y2, x2, mask=mask)
    assert _hexes(a["params"]) == _hexes(b["params"]) and _hexes(a["bse"]) == _hexes(b["bse"])


def test_nan_without_a_mask_is_refused_pointing_at_the_mask():
    ds, mask = _dataset("synthetic")
    y = _arr(ds["y"])
    x = np.array([_arr(ds["x"])])
    with pytest.raises(ValueError, match="outcome: contains a non-finite value.*mask=") as info:
        tsecon.panel_fe(y, x)
    assert "observation mask" in str(info.value)
    # NaN in an OBSERVED cell is refused even with a mask, naming the input.
    y3 = np.where(mask == 1, y, 0.0)
    i, t = np.argwhere(mask == 1)[0]
    y3[i, t] = np.nan
    with pytest.raises(ValueError, match="outcome: contains a non-finite value"):
        tsecon.panel_fe(y3, np.where(mask == 1, x, 0.0), mask=mask)
    x3 = np.where(mask == 1, x, 0.0)
    x3[0, i, t] = np.inf
    with pytest.raises(ValueError, match="regressors: contains a non-finite value"):
        tsecon.panel_distributed_lag(np.where(mask == 1, y, 0.0), x3, lags=1, mask=mask)


def test_mask_refusals_name_the_mask_parameter():
    y, temp, x2, shock, treat, entities = _snap_inputs()
    x = np.array([temp])
    n, t = y.shape
    with pytest.raises(ValueError, match="mask must hold only 0/1"):
        tsecon.panel_fe(y, x, mask=np.full((n, t), 0.5))
    with pytest.raises(ValueError, match="mask must hold only 0/1"):
        tsecon.panel_fe(y, x, mask=np.where(np.ones((n, t)) > 0, np.nan, 1.0))
    with pytest.raises(ValueError, match="dimension mismatch: mask:"):
        tsecon.panel_fe(y, x, mask=np.ones((n - 1, t)))
    with pytest.raises(ValueError, match="mask: every row"):
        tsecon.panel_fe(y, x, mask=np.ones((n, t - 1)))
    with pytest.raises(ValueError, match="mask: the observation mask marks no cell"):
        tsecon.panel_fe(y, x, mask=np.zeros((n, t)))
    unb = np.ones((n, t))
    unb[0, :5] = 0.0
    unb[3, 20:23] = 0.0
    with pytest.raises(ValueError, match=r"lp_did needs a balanced panel \(mask=None\)"):
        tsecon.lp_did(y, treat, mask=unb)
    for kw in ({"jackknife": True}, {"bias_correction": "dj"}, {"bias_correction": "spj", "se_type": "cluster"}):
        with pytest.raises(ValueError, match="bias_correction.*mask"):
            tsecon.panel_lp(y, shock, horizon=3, mask=unb, **kw)
    # The plain masked LP runs, and its rows shrink with the gaps.
    r = tsecon.panel_lp(y, shock, horizon=3, mask=unb)
    full = tsecon.panel_lp(y, shock, horizon=3)
    assert all(a < b for a, b in zip(r["nobs"], full["nobs"]))
    # Too sparse for the lag design: the message names the mask and counts runs.
    sparse = (np.arange(t)[None, :] % 3 == 0).astype(float) * np.ones((n, 1))
    with pytest.raises(ValueError, match=r"\(mask\).*needs at least 3 usable observations, got 1"):
        tsecon.panel_distributed_lag(y, x, lags=2, mask=sparse)


def test_mean_group_var_names_the_entity_with_an_internal_gap():
    _, _, _, _, _, entities = _snap_inputs()
    bad = [e.copy() for e in entities]
    bad[2][7, 1] = np.nan
    with pytest.raises(ValueError, match=r"^entities\[2\]: contains a non-finite value") as info:
        tsecon.mean_group_var(bad, lags=1, horizon=3)
    assert "contiguous span" in str(info.value)
    # Ragged lengths (late entry / early exit) are the supported form.
    ragged = [e.copy() for e in entities]
    ragged[0] = ragged[0][5:]
    r = tsecon.mean_group_var(ragged, lags=1, horizon=3)
    assert r["n_entities"] == len(entities)
