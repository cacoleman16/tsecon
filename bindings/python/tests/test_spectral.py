"""Golden tests for the spectral bindings against scipy.signal fixtures."""
import json
from pathlib import Path

import numpy as np
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
SP = json.loads((FIX / "spectral.json").read_text())
X = np.array(SP["x"]); Y = np.array(SP["y"])


def test_periodogram_matches_scipy():
    r = tsecon.periodogram(X, window="boxcar", detrend="none")
    np.testing.assert_allclose(r["freqs"], SP["periodogram"]["freqs"], atol=1e-10)
    np.testing.assert_allclose(r["psd"], SP["periodogram"]["psd"], rtol=1e-8, atol=1e-12)
    assert (np.asarray(r["psd"]) >= 0).all()


def test_welch_matches_scipy():
    r = tsecon.welch(X, nperseg=128, detrend="none")
    np.testing.assert_allclose(r["freqs"], SP["welch_nperseg128"]["freqs"], atol=1e-10)
    np.testing.assert_allclose(r["psd"], SP["welch_nperseg128"]["psd"], rtol=1e-8, atol=1e-12)


def test_coherence_matches_scipy():
    r = tsecon.coherence(X, Y, nperseg=128, detrend="none")
    np.testing.assert_allclose(r["coherence"], SP["coherence_nperseg128"]["coherence"],
                               rtol=1e-8, atol=1e-10)
    c = np.asarray(r["coherence"])
    assert (c >= -1e-12).all() and (c <= 1 + 1e-12).all()


# --------------------------------------------------------------------------- #
# Default-vs-default parity (the 0.6.0 fix). The docstrings claim these three
# match scipy.signal, whose default detrend is "constant"; tsecon's default
# used to be "none", so the two DEFAULT calls disagreed enormously on any
# series with a nonzero mean (measured welch gap 1678.1 at frequency 0 on a
# mean-5 series). The default is now "constant" too — pin it against a live
# scipy default call on a deliberately mean-shifted series, where "none"
# could never pass.
# --------------------------------------------------------------------------- #
_XM = X + 5.0  # nonzero mean: the case whose default output moved in 0.6.0
_YM = Y - 3.0


def test_periodogram_default_matches_scipy_default():
    from scipy import signal

    r = tsecon.periodogram(_XM)
    f, p = signal.periodogram(_XM)  # scipy defaults: boxcar, detrend="constant"
    np.testing.assert_allclose(r["freqs"], f, atol=1e-10)
    np.testing.assert_allclose(r["psd"], p, rtol=1e-8, atol=1e-12)
    # detrend="constant" actually acted: the DC ordinate is ~0, not ~n*mean^2
    assert np.asarray(r["psd"])[0] < 1e-15 * len(_XM) * 25.0


def test_welch_default_matches_scipy_default():
    from scipy import signal

    r = tsecon.welch(_XM, nperseg=128)
    f, p = signal.welch(_XM, nperseg=128)  # scipy defaults: hann, "constant"
    np.testing.assert_allclose(r["freqs"], f, atol=1e-10)
    np.testing.assert_allclose(r["psd"], p, rtol=1e-8, atol=1e-12)


def test_coherence_default_matches_scipy_default():
    from scipy import signal

    r = tsecon.coherence(_XM, _YM, nperseg=128)
    f, c = signal.coherence(_XM, _YM, nperseg=128)  # scipy default "constant"
    np.testing.assert_allclose(r["freqs"], f, atol=1e-10)
    np.testing.assert_allclose(r["coherence"], c, rtol=1e-8, atol=1e-10)
