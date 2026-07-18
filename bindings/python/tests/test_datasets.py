"""Tests for tsecon.datasets — the download-on-first-use reference loaders.

These run fully offline: every test either uses the committed FRED-MD sample
(`fixtures/fred_md_sample.csv`, a real 48-month slice of the live panel) or a
CSV written into tmp_path. Nothing here touches the network, so CI stays
hermetic while the loaders themselves remain live-capable.
"""

from pathlib import Path

import numpy as np
import pytest

from tsecon import datasets as ds

REPO = Path(__file__).resolve().parents[3]
FRED_MD_SAMPLE = REPO / "fixtures" / "fred_md_sample.csv"
RZ_SAMPLE = REPO / "fixtures" / "ramey_zubairy_sample.csv"


# --------------------------------------------------------------------------- #
# fred_md
# --------------------------------------------------------------------------- #
def test_fred_md_parses_the_real_sample():
    md = ds.fred_md(local_path=FRED_MD_SAMPLE)
    assert md["names"] == ["RPI", "W875RX1", "INDPRO", "CUMFNS", "ACOGNO"]
    # McCracken-Ng transform codes come from the 'Transform:' row, not the data.
    np.testing.assert_array_equal(md["transform_codes"], [5, 5, 5, 2, 5])
    assert md["data"].shape == (48, 5)
    assert md["dates"][0] == np.datetime64("1959-01-01")
    assert md["dates"][-1] == np.datetime64("1962-12-01")
    assert md["sha256"] and len(md["sha256"]) == 64


def test_fred_md_missing_values_become_nan():
    """ACOGNO has no observations this early — it must be nan, not 0 or dropped."""
    md = ds.fred_md(local_path=FRED_MD_SAMPLE)
    j = md["names"].index("ACOGNO")
    assert np.all(np.isnan(md["data"][:, j]))
    # ...while a fully-observed series has none.
    assert not np.any(np.isnan(md["data"][:, md["names"].index("RPI")]))


def test_fred_md_rejects_a_file_without_the_transform_row(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("sasdate,A,B\n1/1/1959,1,2\n2/1/1959,3,4\n")
    with pytest.raises(RuntimeError, match="(?i)transform"):
        ds.fred_md(local_path=bad)


def test_fred_md_rejects_a_truncated_file(tmp_path):
    bad = tmp_path / "short.csv"
    bad.write_text("sasdate,A\n")
    with pytest.raises(RuntimeError, match="(?i)too few rows"):
        ds.fred_md(local_path=bad)


# --------------------------------------------------------------------------- #
# fred_series
# --------------------------------------------------------------------------- #
def test_fred_series_parses_the_fredgraph_format(tmp_path):
    """The keyless fredgraph.csv shape: observation_date,<SERIES_ID>."""
    csv = tmp_path / "gs10.csv"
    csv.write_text(
        "observation_date,GS10\n"
        "1953-04-01,2.83\n"
        "1953-05-01,3.05\n"
        "1953-06-01,.\n"       # FRED writes '.' for a missing observation
        "1953-07-01,2.93\n"
    )
    s = ds.fred_series("GS10", local_path=csv)
    assert s["series_id"] == "GS10"
    assert s["nobs"] == 4
    assert s["dates"][0] == np.datetime64("1953-04-01")
    np.testing.assert_allclose(s["values"][[0, 1, 3]], [2.83, 3.05, 2.93])
    assert np.isnan(s["values"][2])


def test_fred_series_rejects_a_malformed_file(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("not_a_csv_header\n")
    with pytest.raises(RuntimeError, match="(?i)unexpected csv shape"):
        ds.fred_series("GS10", local_path=bad)


# --------------------------------------------------------------------------- #
# transformations
# --------------------------------------------------------------------------- #
def test_transform_codes_match_hand_computation():
    md = ds.fred_md(local_path=FRED_MD_SAMPLE)
    out = ds.apply_fred_md_transforms(md["data"], md["transform_codes"])
    assert out.shape == md["data"].shape

    rpi = md["data"][:, md["names"].index("RPI")]          # code 5 -> dlog
    np.testing.assert_allclose(out[1:, 0], np.diff(np.log(rpi)))
    assert np.isnan(out[0, 0])                              # differencing costs a row

    j = md["names"].index("CUMFNS")                         # code 2 -> first difference
    np.testing.assert_allclose(out[1:, j], np.diff(md["data"][:, j]))
    assert np.isnan(out[0, j])


@pytest.mark.parametrize("code", [1, 2, 3, 4, 5, 6, 7])
def test_every_documented_transform_code_runs(code):
    x = np.linspace(10.0, 20.0, 30).reshape(-1, 1)
    out = ds.apply_fred_md_transforms(x, [code])
    assert out.shape == x.shape
    # Codes 2/3/5/6/7 difference, so they must leave leading nans; 1 and 4 do not.
    leading_nan = int(np.isnan(out[:, 0]).sum())
    assert leading_nan == {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 1}[code]


def test_unknown_transform_code_is_rejected():
    x = np.ones((5, 1))
    with pytest.raises(ValueError, match="unknown FRED-MD transform code"):
        ds.apply_fred_md_transforms(x, [99])


def test_transform_codes_must_align_with_columns():
    x = np.ones((5, 3))
    with pytest.raises(ValueError, match="transform codes for"):
        ds.apply_fred_md_transforms(x, [5, 5])


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# ramey_zubairy
# --------------------------------------------------------------------------- #
def test_ramey_zubairy_parses_the_real_sample():
    rz = ds.ramey_zubairy(local_path=RZ_SAMPLE)
    assert rz["names"][:3] == ["news", "ngov", "ngdp"]
    assert "rgdp_potcbo" in rz["names"]
    assert rz["quarter"][0] == 1875.0
    # quarters are encoded as year + 0.25*(quarter-1)
    assert rz["quarter"][1] == pytest.approx(1875.25)
    assert rz["data"].shape == (len(rz["quarter"]), len(rz["names"]))
    assert len(rz["sha256"]) == 64


def test_ramey_zubairy_series_view_matches_data_columns():
    rz = ds.ramey_zubairy(local_path=RZ_SAMPLE)
    for j, name in enumerate(rz["names"]):
        np.testing.assert_array_equal(rz["series"][name], rz["data"][:, j])


def test_ramey_zubairy_early_rows_are_nan_not_zero():
    """1875 predates the macro series — those cells must be nan, never 0."""
    rz = ds.ramey_zubairy(local_path=RZ_SAMPLE)
    early = rz["quarter"] < 1880
    assert early.any()
    assert np.all(np.isnan(rz["series"]["ngdp"][early]))
    # ...while the WWII rows are fully populated.
    war = (rz["quarter"] >= 1941) & (rz["quarter"] <= 1945)
    assert war.any()
    assert not np.any(np.isnan(rz["series"]["ngdp"][war]))


def test_ramey_zubairy_rejects_an_empty_file(tmp_path):
    bad = tmp_path / "empty.csv"
    bad.write_text("quarter,news\n")
    with pytest.raises(RuntimeError, match="(?i)empty or malformed"):
        ds.ramey_zubairy(local_path=bad)


def test_ramey_zubairy_multiplier_pipeline_runs():
    """The published-replication path must stay wired: build RZ's variables
    from the loader output and get a finite cumulative response."""
    import tsecon

    rz = ds.ramey_zubairy(local_path=RZ_SAMPLE)
    s = rz["series"]
    g = (s["ngov"] / s["pgdp"]) / s["rgdp_potcbo"]
    y = s["rgdp"] / s["rgdp_potcbo"]
    ok = ~np.isnan(g + y)
    assert ok.sum() > 20
    out = tsecon.lp(y[ok], g[ok], horizons=4, n_lag_controls=1, cumulative=True)
    assert np.all(np.isfinite(np.asarray(out["irf"])))


def test_cache_dir_honours_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "somewhere"))
    assert ds.cache_dir() == tmp_path / "somewhere"


def test_offline_fetch_raises_a_teaching_error(monkeypatch, tmp_path):
    """With no cache and no network, the error must say what to do."""
    import urllib.error

    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "empty"))

    def boom(*a, **k):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(ds.urllib.request, "urlopen", boom)
    # (?s) so `.` spans the newlines in the multi-line teaching message.
    with pytest.raises(RuntimeError, match="(?is)could not reach.*local_path"):
        ds.fred_series("GS10")
