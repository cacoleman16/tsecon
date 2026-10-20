"""check_series validates neither `lags` nor `max_breaks`.

check_series is pure Python (tsecon/_inspect.py) and calls the raw compiled
module, so neither the house negative-int nor the 2**48 pre-flight applies:
  * lags=-1, lags=0 and lags=2**62 are silently replaced (1, 1 and n-2 = 93)
    -- `_serial_correlation` clamps `max(1, min(lags, m - 2))`; nothing in the
    docstring says so and no warning is raised;
  * max_breaks=-1 / 0 / 2**31 are accepted silently on a series whose sup-F
    does not reject, and on one whose sup-F rejects the same call raises --
    for -1 a bare pyo3 `OverflowError: can't convert negative int to
    unsigned` that names nothing.
Expected: refuse invalid counts up front by name (as tsecon.ljung_box(nlags=-1)
and tsecon.bai_perron(max_breaks=-1) do).
"""
import warnings
import numpy as np
import tsecon

SEAS = 10.0 + np.sin(2 * np.pi * np.arange(96) / 12) + 0.3 * np.random.default_rng(4).standard_normal(96) + 0.02 * np.arange(96)
for lags in [4, -1, 0, 2**62]:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = tsecon.check_series(SEAS, seasonal_period=12, lags=lags)
    used = r["serial_correlation"]["ljung_box"]["lags"][-1]
    print(f"check_series(lags={lags}): EXPECTED refusal for lags<1 or absurd; ACTUAL returned, Ljung-Box run to lag {used}, warnings={len(w)}")
for lags in [-1, 2**62]:
    try:
        tsecon.ljung_box(SEAS, nlags=lags)
    except Exception as e:
        print(f"  (for contrast) ljung_box(nlags={lags}) -> {type(e).__name__}: {str(e)[:90]}")

t = np.arange(300)
quiet = np.random.default_rng(0).standard_normal(300)          # sup-F does not reject
broken = np.random.default_rng(1).standard_normal(300) + np.where((t >= 100) & (t < 200), 0.6, 0.0)  # sup-F rejects
for mb in [-1, 0, 2**31]:
    for name, y in [("no-break series", quiet), ("break series", broken)]:
        try:
            tsecon.check_series(y, max_breaks=mb)
            out = "returned (max_breaks silently accepted)"
        except BaseException as e:
            out = f"raised {type(e).__name__}: {str(e)[:110]}"
        print(f"check_series(max_breaks={mb}) on {name}: {out}")
try:
    tsecon.bai_perron(broken, np.ones((300, 1)), max_breaks=-1)
except Exception as e:
    print(f"  (for contrast) bai_perron(max_breaks=-1) -> {type(e).__name__}: {str(e)[:90]}")
