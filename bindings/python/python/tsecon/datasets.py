"""Bundled and downloadable reference datasets.

Loaders follow a **download-on-first-use** policy: nothing large is vendored
into the repository, and nothing is fetched at import time. The first call
downloads to a local cache (``~/.cache/tsecon`` by default, or
``$TSECON_DATA_DIR``); later calls read the cache. Every loader records the
source URL and a SHA-256 of the bytes it parsed, so a dataset can be pinned and
audited.

Only the standard library and NumPy are used — no ``requests``, no ``pandas``
requirement.

Sources
-------
``fred_series``
    FRED's public keyless CSV endpoint (``fredgraph.csv``). No API key is
    required. If you have one, set ``FRED_API_KEY`` and it will be passed
    along, but the loader works without it.
``fred_md``
    The FRED-MD monthly macro panel of McCracken & Ng (2016).

    .. note::
       The URL widely cited for FRED-MD
       (``files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv``)
       now returns **403 AccessDenied**. This loader uses the live
       ``www.stlouisfed.org`` media path, verified working.

Licensing: FRED series are US federal government data redistributed by the
Federal Reserve Bank of St. Louis; see their terms of use. Cite McCracken & Ng
(2016) when using FRED-MD.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "cache_dir",
    "clear_cache",
    "fred_series",
    "fred_md",
    "apply_fred_md_transforms",
    "ramey_zubairy",
    "FRED_MD_URL",
    "FRED_SERIES_URL",
    "RAMEY_ZUBAIRY_URL",
]

FRED_SERIES_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_MD_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research"
    "/fred-md/monthly/current.csv"
)
RAMEY_ZUBAIRY_URL = (
    "https://econweb.ucsd.edu/~vramey/research/Ramey_Zubairy_replication_codes.zip"
)
_RZ_MEMBER = "Ramey_Zubairy_replication_codes/rzdatnew.csv"

_USER_AGENT = "tsecon dataset loader (https://github.com/cacoleman16/tsecon)"


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache_dir() -> Path:
    """Directory used for downloaded datasets (``$TSECON_DATA_DIR`` overrides)."""
    env = os.environ.get("TSECON_DATA_DIR")
    if env:
        return Path(env).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "tsecon"


def clear_cache() -> int:
    """Delete every cached download. Returns the number of files removed."""
    d = cache_dir()
    if not d.exists():
        return 0
    n = 0
    for f in d.glob("*"):
        if f.is_file():
            f.unlink()
            n += 1
    return n


def _fetch(url: str, cache_name: str, refresh: bool = False) -> tuple[str, str]:
    """Return ``(text, sha256)``, downloading to the cache on first use."""
    path = cache_dir() / cache_name
    if path.exists() and not refresh:
        raw = path.read_bytes()
        return raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest()

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https hosts
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"the data host returned HTTP {exc.code} for {url}.\n"
            "If this is FRED-MD, the provider may have moved the file again — the "
            "older files.stlouisfed.org path already went 403. Pass local_path=... "
            "to read a copy you already have."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"could not reach {url} ({exc.reason}).\n"
            "This loader downloads on first use and then caches, so it needs network "
            "access once. If you are offline or behind a proxy, download the file "
            "yourself and pass local_path=..., or set TSECON_DATA_DIR to a directory "
            f"already containing {cache_name!r}."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest()


def _read_text(url: str, cache_name: str, local_path, refresh: bool) -> tuple[str, str, str]:
    """Resolve text from ``local_path`` if given, else the cached download."""
    if local_path is not None:
        raw = Path(local_path).expanduser().read_bytes()
        return (
            raw.decode("utf-8", errors="replace"),
            hashlib.sha256(raw).hexdigest(),
            str(local_path),
        )
    text, digest = _fetch(url, cache_name, refresh=refresh)
    return text, digest, url


def _to_float(cell: str) -> float:
    cell = cell.strip()
    if not cell:
        return float("nan")
    try:
        return float(cell)
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# FRED single series
# --------------------------------------------------------------------------- #
def fred_series(series_id: str, *, local_path=None, refresh: bool = False) -> dict:
    """Download one FRED series as ``dates`` / ``values``.

    Uses FRED's public keyless CSV endpoint, so **no API key is required**. If
    ``FRED_API_KEY`` is set in the environment it is appended to the request,
    which is harmless and lets institutional keys apply.

    Returns a dict with ``series_id``, ``dates`` (``datetime64[D]``), ``values``
    (float, missing observations as ``nan``), ``nobs``, ``source``, ``url``, and
    ``sha256``.

    >>> gs10 = fred_series("GS10")           # doctest: +SKIP
    >>> gs10["values"][-1]                    # doctest: +SKIP
    4.47
    """
    url = FRED_SERIES_URL.format(series_id=series_id)
    key = os.environ.get("FRED_API_KEY")
    if key:
        url = f"{url}&api_key={key}"

    text, digest, src = _read_text(url, f"fred_{series_id}.csv", local_path, refresh)

    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows[0]) < 2:
        raise RuntimeError(
            f"unexpected CSV shape for FRED series {series_id!r}; got header {rows[:1]}. "
            "Check that the series id exists on fred.stlouisfed.org."
        )
    dates, values = [], []
    for row in rows[1:]:
        if len(row) < 2 or not row[0].strip():
            continue
        dates.append(row[0].strip())
        values.append(_to_float(row[1]))

    return {
        "series_id": series_id,
        "dates": np.array(dates, dtype="datetime64[D]"),
        "values": np.asarray(values, dtype=float),
        "nobs": len(values),
        "source": "Federal Reserve Bank of St. Louis (FRED)",
        "url": src,
        "sha256": digest,
    }


# --------------------------------------------------------------------------- #
# FRED-MD monthly panel
# --------------------------------------------------------------------------- #
def fred_md(*, local_path=None, refresh: bool = False, drop_empty_rows: bool = True) -> dict:
    """Load the FRED-MD monthly macro panel (McCracken & Ng 2016).

    The file's first data row is McCracken-Ng's ``Transform:`` row of
    transformation codes, one per series; it is parsed out into
    ``transform_codes`` rather than treated as data. The most recent months are
    a **ragged edge** — many series are still unpublished — so trailing rows can
    be entirely missing; ``drop_empty_rows`` removes rows with no observations
    at all.

    Returns ``dates`` (``datetime64[D]``), ``names`` (list of series ids),
    ``data`` (``T x k`` float array, missing as ``nan``), ``transform_codes``
    (int array aligned to ``names``), plus ``source``/``url``/``sha256``.

    >>> md = fred_md()                         # doctest: +SKIP
    >>> md["data"].shape                        # doctest: +SKIP
    (800, 126)
    """
    text, digest, src = _read_text(FRED_MD_URL, "fred_md_current.csv", local_path, refresh)

    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if len(rows) < 3:
        raise RuntimeError(
            "FRED-MD file has too few rows to be valid — expected a header, a "
            "'Transform:' row, and data. The provider may have changed the format."
        )

    names = [c.strip() for c in rows[0][1:]]
    if not rows[1] or "transform" not in rows[1][0].strip().lower():
        raise RuntimeError(
            "expected McCracken-Ng's 'Transform:' row as the second line of "
            f"FRED-MD; found {rows[1][:1]!r}. The file format may have changed."
        )
    codes = np.array([int(_to_float(c)) if c.strip() else 0 for c in rows[1][1:]], dtype=int)

    dates, data = [], []
    for row in rows[2:]:
        if not row[0].strip():
            continue
        vals = [_to_float(c) for c in row[1 : 1 + len(names)]]
        vals += [float("nan")] * (len(names) - len(vals))
        if drop_empty_rows and all(np.isnan(vals)):
            continue
        dates.append(_parse_md_date(row[0].strip()))
        data.append(vals)

    return {
        "dates": np.array(dates, dtype="datetime64[D]"),
        "names": names,
        "data": np.asarray(data, dtype=float),
        "transform_codes": codes,
        "source": "McCracken & Ng (2016) FRED-MD, Federal Reserve Bank of St. Louis",
        "url": src,
        "sha256": digest,
    }


def ramey_zubairy(*, local_path=None, refresh: bool = False) -> dict:
    """Load the Ramey & Zubairy (2018) quarterly US macro dataset.

    This is the dataset behind their government-spending multiplier estimates:
    564 quarters (1875Q1-2015Q4) including Ramey's **military-news shock**
    (``news``), nominal government spending, GDP, the deflator, and CBO
    potential output. The core variables are jointly available from 1890Q1.

    The file is distributed inside the authors' replication zip (~750 kB); the
    loader downloads it once, caches the archive, and extracts the CSV. Nothing
    is vendored into this repository.

    Returns ``quarter`` (float, e.g. ``2015.75`` for 2015Q4), ``names`` (column
    ids), ``data`` (``T x k`` float array, missing as ``nan``), plus a
    ``series`` dict mapping each name to its column for convenience, and
    ``source``/``url``/``sha256``.

    >>> rz = ramey_zubairy()                      # doctest: +SKIP
    >>> rz["series"]["news"].shape                 # doctest: +SKIP
    (564,)

    Cite: Ramey, V. A. & Zubairy, S. (2018), "Government Spending Multipliers
    in Good Times and in Bad: Evidence from US Historical Data", *Journal of
    Political Economy* 126(2):850-901.
    """
    if local_path is not None:
        raw = Path(local_path).expanduser().read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        src = str(local_path)
        text = raw.decode("utf-8", errors="replace")
    else:
        path = cache_dir() / "ramey_zubairy_replication.zip"
        if path.exists() and not refresh:
            blob = path.read_bytes()
        else:
            req = urllib.request.Request(
                RAMEY_ZUBAIRY_URL, headers={"User-Agent": _USER_AGENT}
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                    blob = resp.read()
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                raise RuntimeError(
                    f"could not download the Ramey-Zubairy replication archive from "
                    f"{RAMEY_ZUBAIRY_URL} ({exc}).\n"
                    "Download it yourself and pass local_path=<the extracted "
                    "rzdatnew.csv>, or set TSECON_DATA_DIR to a directory already "
                    "containing 'ramey_zubairy_replication.zip'."
                ) from exc
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)

        import zipfile

        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            member = _RZ_MEMBER
            if member not in zf.namelist():
                candidates = [n for n in zf.namelist() if n.endswith("rzdatnew.csv")]
                if not candidates:
                    raise RuntimeError(
                        "the Ramey-Zubairy archive no longer contains rzdatnew.csv; "
                        f"members look like: {zf.namelist()[:5]}"
                    )
                member = candidates[0]
            data_bytes = zf.read(member)
        digest = hashlib.sha256(data_bytes).hexdigest()
        src = f"{RAMEY_ZUBAIRY_URL}::{member}"
        text = data_bytes.decode("utf-8", errors="replace")

    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise RuntimeError("Ramey-Zubairy CSV appears empty or malformed")
    header = [c.strip() for c in rows[0]]
    names = header[1:]
    quarters, values = [], []
    for row in rows[1:]:
        if not row[0].strip():
            continue
        quarters.append(_to_float(row[0]))
        vals = [_to_float(c) for c in row[1 : 1 + len(names)]]
        vals += [float("nan")] * (len(names) - len(vals))
        values.append(vals)

    arr = np.asarray(values, dtype=float)
    return {
        "quarter": np.asarray(quarters, dtype=float),
        "names": names,
        "data": arr,
        "series": {n: arr[:, j] for j, n in enumerate(names)},
        "source": "Ramey & Zubairy (2018), JPE 126(2):850-901 — replication files",
        "url": src,
        "sha256": digest,
    }


def _parse_md_date(cell: str) -> str:
    """FRED-MD dates are ``M/D/YYYY``; return ISO ``YYYY-MM-DD``."""
    parts = cell.split("/")
    if len(parts) == 3:
        m, d, y = (p.strip() for p in parts)
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    return cell  # already ISO, or something we should not silently mangle


# --------------------------------------------------------------------------- #
# McCracken-Ng transformations
# --------------------------------------------------------------------------- #
def apply_fred_md_transforms(data, codes: Iterable[int]) -> np.ndarray:
    """Apply McCracken-Ng transformation codes column-wise to a FRED-MD panel.

    Codes (McCracken & Ng 2016, Appendix): 1 level, 2 first difference,
    3 second difference, 4 log, 5 first difference of log, 6 second difference
    of log, 7 first difference of the growth rate. Differencing introduces
    leading ``nan`` rows, which is intended — the transformed panel keeps its
    original shape so column alignment is preserved.
    """
    x = np.asarray(data, dtype=float)
    codes = np.asarray(list(codes), dtype=int)
    if x.ndim != 2:
        raise ValueError(f"data must be 2-D (T x k); got shape {x.shape}")
    if codes.shape[0] != x.shape[1]:
        raise ValueError(
            f"got {codes.shape[0]} transform codes for {x.shape[1]} columns — "
            "codes must align with the panel's series"
        )

    out = np.full_like(x, np.nan)
    for j, code in enumerate(codes):
        col = x[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            if code == 1:
                out[:, j] = col
            elif code == 2:
                out[1:, j] = np.diff(col)
            elif code == 3:
                out[2:, j] = np.diff(col, n=2)
            elif code == 4:
                out[:, j] = np.log(col)
            elif code == 5:
                out[1:, j] = np.diff(np.log(col))
            elif code == 6:
                out[2:, j] = np.diff(np.log(col), n=2)
            elif code == 7:
                out[1:, j] = np.diff(col / np.roll(col, 1) - 1.0)
            else:
                raise ValueError(
                    f"unknown FRED-MD transform code {code} for column {j} "
                    f"— expected 1-7 (McCracken & Ng 2016)"
                )
    return out
