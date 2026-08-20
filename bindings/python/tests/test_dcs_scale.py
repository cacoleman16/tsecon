"""Scale robustness of dcs_local_level's fitted parameters (audit round 7).

Rescaling ``y -> c*y`` is a pure relabeling of the DCS level model
(``scale -> c*scale``; ``kappa``/``nu`` unit-free), so the estimator must
commute with it. Round 7's new-surface sweep found the Laplace sign filter —
whose likelihood is piecewise in ``kappa`` — landing in *different kink
basins depending on the units of y* (11/20 seeded series moved ``kappa`` by
up to 57% across eight decades, mapped log-likelihood gaps up to 4.6, all
with ``converged=True``; the smooth t/Gaussian fits moved 0/20). The fit now
optimizes on the internally standardized series and maps back exactly —
bit-exactly for power-of-two scales.
"""

import numpy as np

import tsecon


def _sim(seed=42, n=400):
    rng = np.random.default_rng(seed)
    level = np.cumsum(0.1 * rng.standard_normal(n))
    y = level + rng.standard_normal(n)
    idx = rng.choice(n, size=n // 20, replace=False)
    y[idx] += rng.choice([-8.0, 8.0], size=n // 20)
    return y


def test_power_of_two_rescaling_maps_the_fit_bit_exactly():
    y = _sim()
    for density in ("t", "laplace", "gaussian"):
        ref = tsecon.dcs_local_level(y, density=density)
        for k in (-16, 10):
            c = 2.0 ** k
            r = tsecon.dcs_local_level(np.asarray(c * y), density=density)
            assert np.float64(r["kappa"]).tobytes() == np.float64(ref["kappa"]).tobytes(), (
                f"{density} c=2^{k}: kappa {r['kappa']!r} vs {ref['kappa']!r}"
            )
            assert np.float64(r["scale"]).tobytes() == np.float64(ref["scale"] * c).tobytes()
            if density == "t":
                assert np.float64(r["nu"]).tobytes() == np.float64(ref["nu"]).tobytes()


def test_decade_rescaling_maps_the_smooth_fits_within_1e6():
    y = _sim(seed=7)
    for density in ("t", "gaussian"):
        ref = tsecon.dcs_local_level(y, density=density)
        for k in (-6, 6):
            c = 10.0 ** k
            r = tsecon.dcs_local_level(np.asarray(c * y), density=density)
            assert abs(r["kappa"] - ref["kappa"]) / abs(ref["kappa"]) < 1e-6
            assert abs(r["scale"] / c - ref["scale"]) / abs(ref["scale"]) < 1e-6
            # level path in mapped units
            dl = np.max(np.abs(np.asarray(r["level"]) / c - np.asarray(ref["level"])))
            assert dl / np.max(np.abs(ref["level"])) < 1e-6
