"""Spectral-analysis golden fixtures from scipy.signal."""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy import signal

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]

rng = np.random.default_rng(101)
n = 512
# A signal with two sinusoids + noise, so the periodogram has clear peaks.
t = np.arange(n)
x = 2.0 * np.sin(2*np.pi*0.1*t) + 1.0 * np.sin(2*np.pi*0.25*t) + rng.standard_normal(n)
# Second series correlated with x for coherence.
y = 1.5 * np.sin(2*np.pi*0.1*t + 0.5) + rng.standard_normal(n)

f_p, Pxx = signal.periodogram(x, detrend=False, scaling="density", window="boxcar")
f_w, Pxx_w = signal.welch(x, nperseg=128, detrend=False, scaling="density")
f_c, Cxy = signal.coherence(x, y, nperseg=128, detrend=False)

out = {
    "_meta": {"scipy": scipy.__version__, "numpy": np.__version__,
              "python": platform.python_version(),
              "note": "scipy.signal periodogram(boxcar, density, detrend=False), "
                      "welch(nperseg=128, density, detrend=False), coherence(nperseg=128)."},
    "x": full(x), "y": full(y),
    "periodogram": {"freqs": full(f_p), "psd": full(Pxx)},
    "welch_nperseg128": {"freqs": full(f_w), "psd": full(Pxx_w)},
    "coherence_nperseg128": {"freqs": full(f_c), "coherence": full(Cxy)},
}
(OUT / "spectral.json").write_text(json.dumps(out, separators=(",", ":")))
print("wrote spectral.json", (OUT/"spectral.json").stat().st_size, "bytes")
