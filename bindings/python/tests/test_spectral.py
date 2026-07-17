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
