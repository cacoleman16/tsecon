"""Python-boundary tests for lp_did (LP-DiD, Dube-Girardi-Jordà-Taylor
2025 J. Applied Econometrics).

The numeric authority is fixtures/lpdid.json — a reference-run golden:
the stored values come from an R/fixest run of the authors' own example
implementations (github.com/danielegirardi/lpdid), transcribed by
fixtures/generate_lpdid_fixtures.R and cross-checked against an
independent NumPy reimplementation at generation time (see the
generator's docstring for the grade and its one caveat: the SSC-only
Stata ado itself could not be fetched). The Rust suite pins all six
fixture cases; here we pin the Python boundary on all of them at 1e-10
and exercise argument plumbing, metadata stamps, the baseline row, and
the refusals — including the clean-control degeneracies the method's
docs warn about.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import tsecon

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[3] / "fixtures" / "lpdid.json").read_text()
)


def _panel(case):
    key = "panel_a" if case.startswith("A") else "panel_b"
    return np.array(FIXTURE[key]["y"]), np.array(FIXTURE[key]["d"])


def _run(case, **overrides):
    y, d = _panel(case)
    spec = FIXTURE["cases"][case]
    kwargs = dict(
        pre_window=spec["pre_window"],
        post_window=spec["post_window"],
        absorbing=spec["absorbing"],
        nonabsorbing_lag=spec["nonabsorbing_lag"],
        reweight=spec["reweight"],
        pooled=spec["pooled"],
        never_treated_only=spec["never_treated_only"],
    )
    kwargs.update(overrides)
    return tsecon.lp_did(y, d, **kwargs)


@pytest.mark.parametrize("case", sorted(FIXTURE["cases"]))
def test_lp_did_matches_the_fixest_reference_run(case):
    r = _run(case)
    spec = FIXTURE["cases"][case]
    horizons = list(r["horizons"])
    assert horizons == list(range(-spec["pre_window"], spec["post_window"] + 1))
    for key, want in spec["results"].items():
        if key == "pooled_post":
            got = (r["pooled_post_att"], r["pooled_post_se"], r["pooled_post_nobs"])
        elif key == "pooled_pre":
            got = (r["pooled_pre_att"], r["pooled_pre_se"], r["pooled_pre_nobs"])
        else:
            k = horizons.index(int(key))
            got = (r["coef"][k], r["se"][k], r["nobs"][k])
        assert got[0] == pytest.approx(want["coef"], rel=1e-10), key
        assert got[1] == pytest.approx(want["se"], rel=1e-10), key
        assert got[2] == want["nobs"], key


def test_baseline_row_is_exact_zero_and_samples_shrink():
    r = _run("A_vw")
    horizons = list(r["horizons"])
    k = horizons.index(-1)
    assert r["coef"][k] == 0.0 and r["se"][k] == 0.0 and r["nobs"][k] == 0
    assert r["nobs"][horizons.index(6)] < r["nobs"][horizons.index(0)]
    # 29 staggered switchers: the always-treated unit is not an event.
    assert r["n_switchers"][horizons.index(0)] == 29


def test_metadata_is_stamped():
    r = _run("B_rw")
    assert r["absorbing"] is False
    assert r["nonabsorbing_lag"] == 3
    assert r["reweight"] is True
    assert r["pooled"] is False
    assert r["never_treated_only"] is False
    assert r["se_type"] == "cluster_entity"
    assert "pooled_post_att" not in r  # pooled keys only when pooled=True


def test_pooled_keys_present_when_requested():
    r = _run("A_rw")
    for key in (
        "pooled_post_att",
        "pooled_post_se",
        "pooled_post_nobs",
        "pooled_post_n_switchers",
        "pooled_pre_att",
        "pooled_pre_se",
        "pooled_pre_nobs",
        "pooled_pre_n_switchers",
    ):
        assert key in r, key


def test_reversal_under_absorbing_raises_and_nonabsorbing_accepts():
    y, d = _panel("B_vw")  # panel B has entries AND exits
    with pytest.raises(ValueError, match="absorbing"):
        tsecon.lp_did(y, d, pre_window=3, post_window=4, absorbing=True)
    r = tsecon.lp_did(
        y, d, pre_window=3, post_window=4, absorbing=False, nonabsorbing_lag=3
    )
    assert len(r["coef"]) == 8


def test_never_treated_only_without_never_treated_units_raises():
    y, d = _panel("A_vw")
    d = d.copy()
    d[:, -1] = 1.0  # now every unit is treated at some point
    with pytest.raises(ValueError, match="never-treated"):
        tsecon.lp_did(y, d, pre_window=4, post_window=4, never_treated_only=True)


def test_window_exceeding_the_panel_raises():
    y, d = _panel("A_vw")
    with pytest.raises(ValueError, match="post window"):
        tsecon.lp_did(y, d, pre_window=2, post_window=y.shape[1])
    with pytest.raises(ValueError, match="pre window"):
        tsecon.lp_did(y, d, pre_window=y.shape[1], post_window=2)


def test_non_binary_treatment_raises():
    y, d = _panel("A_vw")
    d = d.copy()
    d[5, 10] = 0.7
    with pytest.raises(ValueError, match="binary"):
        tsecon.lp_did(y, d, pre_window=2, post_window=2)


def test_mode_lag_consistency_raises():
    y, d = _panel("A_vw")
    with pytest.raises(ValueError, match="nonabsorbing_lag"):
        tsecon.lp_did(y, d, pre_window=2, post_window=2, absorbing=False)
    with pytest.raises(ValueError, match="ambiguous"):
        tsecon.lp_did(y, d, pre_window=2, post_window=2, nonabsorbing_lag=3)


def test_no_switchers_raises():
    y, d = _panel("A_vw")
    with pytest.raises(ValueError, match="no treatment switch"):
        tsecon.lp_did(y, np.zeros_like(d), pre_window=2, post_window=2)


def test_integer_treatment_is_coerced():
    y, d = _panel("A_vw")
    r_int = tsecon.lp_did(y, d.astype(np.int64), pre_window=4, post_window=6)
    r_f64 = tsecon.lp_did(y, d, pre_window=4, post_window=6)
    np.testing.assert_array_equal(r_int["coef"], r_f64["coef"])


def test_docstring_names_every_returned_key():
    import re

    r = _run("A_rw")
    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.lp_did.__doc__ or ""))
    # pooled_pre_* keys are documented via the pooled_pre_ prefix wildcard
    # in the stub but the runtime doc must name every actual key.
    missing = set(r.keys()) - tokens
    assert not missing, f"lp_did.__doc__ does not name returned keys: {sorted(missing)}"
