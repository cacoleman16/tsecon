"""Scale robustness of garch_fit's *fitted parameters* (audit round 7,
fixing round 1 finding b).

Rescaling the data ``y -> c*y`` is a pure relabeling of the GARCH model
(``omega -> c^2 omega``, ``mu -> c mu``, coefficients unchanged), so the
estimator must commute with it. Round 1 measured 52/330 cross-scale
comparisons converging to a different point; the round-2 fix repaired the
standard-error steps but the *fit* stayed unit-sensitive (the optimizer's
termination arithmetic saw the units through the shifted log-likelihood and
the stretched ``ln omega`` coordinate). Since 0.4.0 the optimizer runs on the
internally standardized series ``y / rms`` and the optimum is mapped back
exactly — bit-exactly for power-of-two scales, where standardization is a
pure exponent shift.
"""

import numpy as np

import tsecon


def _sim_garch(omega, alpha, beta, n, seed):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 200)
    s2 = omega / (1 - alpha - beta)
    y = np.empty(n + 200)
    for t in range(n + 200):
        y[t] = np.sqrt(s2) * z[t]
        s2 = omega + alpha * y[t] ** 2 + beta * s2
    return y[200:]


def test_power_of_two_rescaling_maps_the_fit_bit_exactly():
    y = _sim_garch(0.05, 0.08, 0.88, 900, 5)
    ref = tsecon.garch_fit(y, vol="garch", mean="constant", dist="normal")
    names = list(ref["param_names"])
    for k in (-20, 12):
        c = 2.0 ** k
        r = tsecon.garch_fit(np.asarray(c * y), vol="garch", mean="constant", dist="normal")
        for i, nm in enumerate(names):
            expected = ref["params"][i] * {"mu": c, "omega": c * c}.get(nm, 1.0)
            assert np.float64(r["params"][i]).tobytes() == np.float64(expected).tobytes(), (
                f"c=2^{k}: {nm} = {r['params'][i]!r} vs mapped reference {expected!r}"
            )


def test_decade_rescaling_maps_the_fit_within_1e6():
    """The audit's own probe design: decades round `c*y`, so bit-equality is
    impossible (genuinely different data), but the mapped optimum must agree
    far beyond statistical resolution."""
    y = _sim_garch(0.05, 0.08, 0.88, 1000, 7)
    ref = tsecon.garch_fit(y, vol="garch", mean="zero", dist="normal")
    rms = float(np.sqrt(np.mean(y**2)))
    names = list(ref["param_names"])
    for k in (-8, -4, 4, 8):
        c = 10.0 ** k
        r = tsecon.garch_fit(np.asarray(c * y), vol="garch", mean="zero", dist="normal")
        for i, nm in enumerate(names):
            mapped = r["params"][i] / {"mu": c, "omega": c * c}.get(nm, 1.0)
            denom = rms if nm == "mu" else abs(ref["params"][i])
            assert abs(mapped - ref["params"][i]) / denom < 1e-6, (
                f"c=1e{k}: {nm} mapped to {mapped} vs {ref['params'][i]}"
            )


def test_decimal_returns_fit_like_percent_returns():
    """The classic trap the round-1 audit hit: daily equity returns in
    decimals (omega ~ 1e-6). The fit and its standard errors must be the
    percent-scale ones, mapped."""
    y_pct = _sim_garch(0.05, 0.08, 0.88, 1200, 11)  # percent scale
    y_dec = np.asarray(y_pct / 100.0)  # decimal scale
    a = tsecon.garch_fit(y_pct, vol="garch", mean="zero", dist="normal")
    b = tsecon.garch_fit(y_dec, vol="garch", mean="zero", dist="normal")
    names = list(a["param_names"])
    assert all(b["se_valid"]), "decimal-scale SEs must be valid"
    for i, nm in enumerate(names):
        f = 1e-4 if nm == "omega" else 1.0
        assert abs(b["params"][i] - a["params"][i] * f) / (abs(a["params"][i]) * f) < 1e-6
        assert abs(b["se_robust"][i] - a["se_robust"][i] * f) / (a["se_robust"][i] * f) < 1e-3
