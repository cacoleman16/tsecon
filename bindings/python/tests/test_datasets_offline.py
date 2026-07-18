"""Loader machinery that the happy-path tests never reach — all of it offline.

`test_datasets.py` covers parsing a real file handed in via ``local_path=``.
What it does *not* touch is everything between the loader and the network: the
download-and-cache round trip, the cache-hit branch, the two teaching errors,
the FRED_API_KEY splice, the Ramey-Zubairy zip extraction, and the row-skipping
rules that keep dates aligned with data.

Those are exactly the paths where a bug returns a *plausible* answer instead of
raising — a stale cache, a mis-hashed payload, a row silently dropped out of one
column but not another. So they are tested here, and tested without a socket:
``urlopen`` is monkeypatched and every cache lives in ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tsecon import datasets as ds

REPO = Path(__file__).resolve().parents[3]
FRED_MD_SAMPLE = REPO / "fixtures" / "fred_md_sample.csv"
RZ_SAMPLE = REPO / "fixtures" / "ramey_zubairy_sample.csv"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, payload: bytes, seen: list | None = None):
    """Make ``urlopen`` return ``payload`` and optionally record the request."""

    def fake_urlopen(req, timeout=None):
        if seen is not None:
            seen.append(req.full_url)
        return _FakeResponse(payload)

    monkeypatch.setattr(ds.urllib.request, "urlopen", fake_urlopen)


def _raise(monkeypatch, exc: Exception):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(ds.urllib.request, "urlopen", boom)


FRED_CSV = b"observation_date,GS10\n1953-04-01,2.83\n1953-05-01,3.05\n"


# --------------------------------------------------------------------------- #
# cache_dir / clear_cache
# --------------------------------------------------------------------------- #
def test_cache_dir_falls_back_to_xdg_then_home(monkeypatch, tmp_path):
    """No TSECON_DATA_DIR: XDG_CACHE_HOME wins, else ~/.cache. Getting this
    wrong writes megabytes of downloads somewhere the user did not expect."""
    monkeypatch.delenv("TSECON_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert ds.cache_dir() == tmp_path / "xdg" / "tsecon"

    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert ds.cache_dir() == tmp_path / "home" / ".cache" / "tsecon"


def test_clear_cache_on_a_missing_directory_is_zero_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "never-created"))
    assert ds.clear_cache() == 0


def test_clear_cache_removes_files_counts_them_and_spares_subdirectories(
    monkeypatch, tmp_path
):
    """This function deletes things, so its blast radius is the test: files in
    the cache go, the directory itself and any subdirectory stay."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.csv").write_text("a")
    (cache / "b.zip").write_bytes(b"b")
    (cache / "sub").mkdir()
    (cache / "sub" / "keep.csv").write_text("keep")
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))

    assert ds.clear_cache() == 2
    assert cache.exists()
    assert not (cache / "a.csv").exists()
    assert not (cache / "b.zip").exists()
    assert (cache / "sub" / "keep.csv").exists()   # glob("*") is not recursive
    assert ds.clear_cache() == 0                    # idempotent


# --------------------------------------------------------------------------- #
# _fetch: download, cache, re-read, refresh
# --------------------------------------------------------------------------- #
def test_first_call_downloads_and_writes_the_cache(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))
    _serve(monkeypatch, FRED_CSV)

    s = ds.fred_series("GS10")
    assert s["nobs"] == 2
    assert s["sha256"] == hashlib.sha256(FRED_CSV).hexdigest()
    assert (cache / "fred_GS10.csv").read_bytes() == FRED_CSV


def test_second_call_reads_the_cache_and_reports_the_same_digest(monkeypatch, tmp_path):
    """The cache-hit branch must return the bytes *and* a hash of those bytes.
    A digest computed off the wrong buffer would make a dataset unpinnable."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "fred_GS10.csv").write_bytes(FRED_CSV)
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))
    _raise(monkeypatch, AssertionError("the cache hit must not touch the network"))

    s = ds.fred_series("GS10")
    assert s["nobs"] == 2
    assert s["sha256"] == hashlib.sha256(FRED_CSV).hexdigest()


def test_refresh_overwrites_a_stale_cache(monkeypatch, tmp_path):
    """Silently serving stale data is the failure mode this flag exists for."""
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "fred_GS10.csv").write_bytes(b"observation_date,GS10\n1953-04-01,1.00\n")
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))
    _serve(monkeypatch, FRED_CSV)

    s = ds.fred_series("GS10", refresh=True)
    assert s["nobs"] == 2
    np.testing.assert_allclose(s["values"], [2.83, 3.05])
    assert (cache / "fred_GS10.csv").read_bytes() == FRED_CSV


def test_http_error_names_the_403_history_and_the_way_out(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "empty"))
    _raise(
        monkeypatch,
        urllib.error.HTTPError("http://x", 403, "Forbidden", None, None),
    )
    with pytest.raises(RuntimeError, match="(?is)HTTP 403.*local_path"):
        ds.fred_md()


def test_local_path_never_consults_the_cache_or_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "empty"))
    _raise(monkeypatch, AssertionError("local_path must not download"))

    md = ds.fred_md(local_path=FRED_MD_SAMPLE)
    assert md["url"] == str(FRED_MD_SAMPLE)      # provenance points at the file
    assert md["data"].shape == (48, 5)


def test_fred_api_key_is_appended_when_present(monkeypatch, tmp_path):
    """The keyless endpoint is the default; a key, if set, must actually reach
    the request rather than being silently dropped."""
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("FRED_API_KEY", "abc123")
    seen: list[str] = []
    _serve(monkeypatch, FRED_CSV, seen)

    out = ds.fred_series("GS10")
    assert seen and seen[0].endswith("&api_key=abc123")
    assert out["url"] == seen[0]


def test_no_api_key_leaves_the_url_keyless(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    seen: list[str] = []
    _serve(monkeypatch, FRED_CSV, seen)

    ds.fred_series("GS10")
    assert seen and "api_key" not in seen[0]


# --------------------------------------------------------------------------- #
# row-skipping: the paths where dates could drift out of line with data
# --------------------------------------------------------------------------- #
def test_fred_series_skips_blank_and_short_rows_without_shifting_values(tmp_path):
    csv = tmp_path / "gs10.csv"
    csv.write_text(
        "observation_date,GS10\n"
        "1953-04-01,2.83\n"
        ",\n"                # blank date: dropped
        "1953-05-01\n"       # one cell only: dropped, not read as a value
        "1953-06-01,3.05\n"
    )
    s = ds.fred_series("GS10", local_path=csv)
    assert s["nobs"] == 2
    np.testing.assert_array_equal(
        s["dates"], np.array(["1953-04-01", "1953-06-01"], dtype="datetime64[D]")
    )
    np.testing.assert_allclose(s["values"], [2.83, 3.05])


def test_fred_md_skips_unlabelled_rows_and_pads_short_ones(tmp_path):
    """A ragged row must be padded with nan on the right, never left-shifted:
    a shifted row assigns one series' number to another series."""
    csv = tmp_path / "md.csv"
    csv.write_text(
        "sasdate,A,B,C\n"
        "Transform:,1,2,5\n"
        "1/1/1959,1,2,3\n"
        ",9,9,9\n"           # no date: dropped entirely
        "3/1/1959,4\n"       # short: B and C become nan, A stays 4
    )
    md = ds.fred_md(local_path=csv)
    assert md["names"] == ["A", "B", "C"]
    assert md["data"].shape == (2, 3)
    np.testing.assert_array_equal(
        md["dates"], np.array(["1959-01-01", "1959-03-01"], dtype="datetime64[D]")
    )
    np.testing.assert_allclose(md["data"][0], [1.0, 2.0, 3.0])
    assert md["data"][1, 0] == 4.0
    assert np.all(np.isnan(md["data"][1, 1:]))


def test_fred_md_drop_empty_rows_toggles_the_ragged_edge(tmp_path):
    """FRED-MD's last months are all-nan for many series; the flag decides
    whether such a row survives, and dates must follow the data either way."""
    csv = tmp_path / "md.csv"
    csv.write_text(
        "sasdate,A,B\n"
        "Transform:,1,1\n"
        "1/1/1959,1,2\n"
        "2/1/1959,,\n"       # entirely missing month
    )
    dropped = ds.fred_md(local_path=csv)
    assert dropped["data"].shape == (1, 2)
    assert dropped["dates"].shape == (1,)

    kept = ds.fred_md(local_path=csv, drop_empty_rows=False)
    assert kept["data"].shape == (2, 2)
    assert kept["dates"][-1] == np.datetime64("1959-02-01")
    assert np.all(np.isnan(kept["data"][1]))


def test_fred_md_accepts_iso_dates_unchanged(tmp_path):
    """_parse_md_date only rewrites M/D/YYYY; anything else is passed through
    rather than mangled, so an ISO file still parses."""
    csv = tmp_path / "md.csv"
    csv.write_text("sasdate,A\nTransform:,1\n1959-01-01,1\n1959-02-01,2\n")
    md = ds.fred_md(local_path=csv)
    np.testing.assert_array_equal(
        md["dates"], np.array(["1959-01-01", "1959-02-01"], dtype="datetime64[D]")
    )


def test_ramey_zubairy_skips_rows_with_no_quarter(tmp_path):
    csv = tmp_path / "rz.csv"
    csv.write_text("quarter,news,ngdp\n1875,1,2\n,7,7\n1875.25,3\n")
    rz = ds.ramey_zubairy(local_path=csv)
    np.testing.assert_allclose(rz["quarter"], [1875.0, 1875.25])
    np.testing.assert_allclose(rz["series"]["news"], [1.0, 3.0])
    assert rz["series"]["ngdp"][0] == 2.0
    assert np.isnan(rz["series"]["ngdp"][1])       # short row padded, not shifted


# --------------------------------------------------------------------------- #
# ramey_zubairy: the zip path, driven from a cached archive in tmp_path
# --------------------------------------------------------------------------- #
def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return buf.getvalue()


RZ_CSV = b"quarter,news,ngdp\n1875,1,2\n1875.25,3,4\n"


def test_rz_extracts_the_expected_member_from_a_cached_archive(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    blob = _zip_bytes({"Ramey_Zubairy_replication_codes/rzdatnew.csv": RZ_CSV})
    (cache / "ramey_zubairy_replication.zip").write_bytes(blob)
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))
    _raise(monkeypatch, AssertionError("a cached archive must not be re-downloaded"))

    rz = ds.ramey_zubairy()
    np.testing.assert_allclose(rz["quarter"], [1875.0, 1875.25])
    # The digest is of the extracted CSV, not of the enclosing zip.
    assert rz["sha256"] == hashlib.sha256(RZ_CSV).hexdigest()
    assert rz["url"].endswith("::Ramey_Zubairy_replication_codes/rzdatnew.csv")


def test_rz_finds_rzdatnew_under_a_renamed_top_level_directory(monkeypatch, tmp_path):
    """The authors have re-zipped with a different folder name before; the
    fallback search must find the file rather than claim the archive changed."""
    cache = tmp_path / "cache"
    cache.mkdir()
    blob = _zip_bytes({"RZ_codes_2019/rzdatnew.csv": RZ_CSV, "readme.txt": b"hi"})
    (cache / "ramey_zubairy_replication.zip").write_bytes(blob)
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))

    rz = ds.ramey_zubairy()
    assert rz["url"].endswith("::RZ_codes_2019/rzdatnew.csv")
    np.testing.assert_allclose(rz["series"]["news"], [1.0, 3.0])


def test_rz_raises_when_the_archive_has_no_rzdatnew(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ramey_zubairy_replication.zip").write_bytes(
        _zip_bytes({"readme.txt": b"hi", "other.csv": b"a,b\n1,2\n"})
    )
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))

    with pytest.raises(RuntimeError, match="no longer contains rzdatnew.csv"):
        ds.ramey_zubairy()


def test_rz_downloads_once_then_caches_the_archive(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("TSECON_DATA_DIR", str(cache))
    blob = _zip_bytes({"Ramey_Zubairy_replication_codes/rzdatnew.csv": RZ_CSV})
    calls: list[str] = []
    _serve(monkeypatch, blob, calls)

    first = ds.ramey_zubairy()
    assert (cache / "ramey_zubairy_replication.zip").read_bytes() == blob

    _raise(monkeypatch, AssertionError("the second call must hit the cache"))
    second = ds.ramey_zubairy()
    np.testing.assert_allclose(first["data"], second["data"])
    assert len(calls) == 1


def test_rz_offline_error_names_local_path_and_the_cache_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("TSECON_DATA_DIR", str(tmp_path / "empty"))
    _raise(monkeypatch, urllib.error.URLError("no route to host"))
    with pytest.raises(RuntimeError, match="(?is)could not download.*local_path.*TSECON_DATA_DIR"):
        ds.ramey_zubairy()


# --------------------------------------------------------------------------- #
# transform guards
# --------------------------------------------------------------------------- #
def test_transforms_reject_a_non_2d_panel():
    with pytest.raises(ValueError, match="must be 2-D"):
        ds.apply_fred_md_transforms(np.arange(5.0), [1])
    with pytest.raises(ValueError, match="must be 2-D"):
        ds.apply_fred_md_transforms(np.zeros((2, 2, 2)), [1, 1])
