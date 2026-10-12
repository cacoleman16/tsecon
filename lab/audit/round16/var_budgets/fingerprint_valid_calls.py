"""Bit-level fingerprints of valid calls to the VAR-family callables whose
size budgets audit round 16 added or corrected.

Every number each call returns (dict values in sorted-key order, nested lists
and arrays flattened in order, ints and floats alike) is fed to FNV-1a over
its IEEE-754 bit pattern. Run it against the wheel before and after the
change on the same machine: identical lines mean identical outputs, bit for
bit. (The hashes are per platform, not portable: faer picks SIMD kernels by
CPU, so the last bits of a matrix product may differ across machines.)

Usage: python fingerprint_valid_calls.py > out.txt
"""

from __future__ import annotations

import struct

import numpy as np
import tsecon

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK = (1 << 64) - 1


def _feed(h: int, x: float) -> int:
    for b in struct.pack("<d", float(x)):
        h ^= b
        h = (h * FNV_PRIME) & MASK
    return h


def _walk(h: int, v: object) -> int:
    if isinstance(v, dict):
        for key in sorted(v):
            for ch in key.encode():
                h ^= ch
                h = (h * FNV_PRIME) & MASK
            h = _walk(h, v[key])
        return h
    if isinstance(v, np.ndarray):
        for x in v.ravel():
            h = _feed(h, x)
        return h
    if isinstance(v, (list, tuple)):
        for x in v:
            h = _walk(h, x)
        return h
    if v is None or isinstance(v, str):
        for ch in str(v).encode():
            h ^= ch
            h = (h * FNV_PRIME) & MASK
        return h
    return _feed(h, v)


def fingerprint(v: object) -> str:
    return f"{_walk(FNV_OFFSET, v):016x}"


def var_data(t: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((t, k))
    for i in range(1, t):
        y[i] = a @ y[i - 1] + rng.standard_normal(k)
    return y


X2 = var_data(200, 2, 1)
X3 = var_data(240, 3, 2)
rng = np.random.default_rng(7)
y_lp = rng.standard_normal(300).cumsum() * 0.1 + rng.standard_normal(300)
shock_lp = rng.standard_normal(300)
curves = rng.standard_normal((150, 12)).cumsum(axis=1)
y_fvar = rng.standard_normal(150)
delta = np.linspace(0.1, 0.3, 12)
entities = [var_data(120, 2, 10 + i) for i in range(4)]

CALLS = {
    "var_irf": lambda: tsecon.var_irf(X3, lags=2, horizon=12),
    "var_irf_noorth_cum": lambda: tsecon.var_irf(X3, lags=2, horizon=12, orth=False, cumulative=True),
    "var_fevd": lambda: tsecon.var_fevd(X3, lags=2, horizon=10),
    "connectedness": lambda: tsecon.connectedness(X3, lags=2, horizon=10),
    "fvar_scenario": lambda: tsecon.fvar_scenario(y_fvar, curves, delta, n_factors=2, lags=1, horizon=8),
    "mean_group_var": lambda: tsecon.mean_group_var(entities, lags=1, horizon=8, response=1, impulse=0),
    "long_run_svar": lambda: tsecon.long_run_svar(X2, lags=2, horizon=12),
    "long_run_svar_perm": lambda: tsecon.long_run_svar(
        X3, lags=1, restrictions=[(0, 0), (1, 0), (0, 2)], normalize="impact", impact_signs=[(1, 1)]
    ),
    "vib_asym": lambda: tsecon.var_irf_bands(X3, lags=2, horizon=8),
    "vib_asym_cum_noorth": lambda: tsecon.var_irf_bands(X3, lags=2, horizon=8, orth=False, cumulative=True),
    "vib_asym_supt": lambda: tsecon.var_irf_bands(X3, lags=2, horizon=8, band="sup-t", band_n_sim=20000),
    "vib_asym_supt_all": lambda: tsecon.var_irf_bands(
        X2, lags=1, horizon=6, band="sup-t", band_scope="all", band_n_sim=20000, band_seed=3
    ),
    "vib_asym_sidak_shock": lambda: tsecon.var_irf_bands(X3, lags=2, horizon=8, band="sidak", band_scope="shock"),
    "vib_asym_bonf": lambda: tsecon.var_irf_bands(X3, lags=2, horizon=8, band="bonferroni"),
    "vib_boot": lambda: tsecon.var_irf_bands(X2, lags=2, horizon=10, method="bootstrap", n_boot=200, seed=5),
    "vib_boot_cum_bias": lambda: tsecon.var_irf_bands(
        X2, lags=1, horizon=10, method="bootstrap", n_boot=150, seed=6, cumulative=True, bias_correct=True
    ),
    "vib_boot_noorth_supt": lambda: tsecon.var_irf_bands(
        X3, lags=1, horizon=6, method="bootstrap", n_boot=150, seed=8, orth=False, band="sup-t", band_scope="all"
    ),
    "lrsb_boot": lambda: tsecon.long_run_svar_bands(X2, lags=2, horizon=12, n_boot=200, seed=1),
    "lrsb_boot_cum_bias_signs": lambda: tsecon.long_run_svar_bands(
        X2, lags=2, horizon=12, n_boot=150, seed=2, cumulative=True, bias_correct=True, impact_signs=[(0, 1)]
    ),
    "lrsb_boot_perm_supt_draws": lambda: tsecon.long_run_svar_bands(
        X3,
        lags=1,
        horizon=6,
        restrictions=[(0, 0), (1, 0), (0, 2)],
        normalize="impact",
        n_boot=120,
        seed=4,
        band="sup-t",
        band_scope="shock",
        return_draws=True,
    ),
    "lrsb_boot_sidak_all": lambda: tsecon.long_run_svar_bands(
        X2, lags=1, horizon=5, n_boot=100, seed=9, band="sidak", band_scope="all"
    ),
    "lrsb_asym": lambda: tsecon.long_run_svar_bands(X2, lags=2, horizon=12, method="asymptotic"),
    "lrsb_asym_supt": lambda: tsecon.long_run_svar_bands(
        X2, lags=2, horizon=12, method="asymptotic", band="sup-t", band_n_sim=20000, cumulative=True
    ),
    "lrsb_asym_bonf_all": lambda: tsecon.long_run_svar_bands(
        X3, lags=1, horizon=4, method="asymptotic", band="bonferroni", band_scope="all"
    ),
    "lp_supt": lambda: tsecon.lp(y_lp, shock_lp, horizons=8, band="sup-t", band_n_sim=20000),
    "lp_supt_default": lambda: tsecon.lp(y_lp, shock_lp, horizons=6, band="sup-t"),
    "lp_sidak": lambda: tsecon.lp(y_lp, shock_lp, horizons=8, band="sidak"),
    "smooth_lp_supt": lambda: tsecon.smooth_lp(y_lp, shock_lp, horizons=8, band="sup-t", band_n_sim=20000),
    "var_forecast": lambda: tsecon.var_forecast(X3, lags=2, steps=8),
    "var_forecast_pu": lambda: tsecon.var_forecast(X3, lags=2, steps=8, param_uncertainty=True),
    "var_forecast_supt": lambda: tsecon.var_forecast(X3, lags=2, steps=8, band="sup-t", band_n_sim=20000),
    "var_forecast_supt_h": lambda: tsecon.var_forecast(
        X3, lags=2, steps=6, band="sup-t", band_scope="horizon", band_seed=11
    ),
    "var_forecast_bonf": lambda: tsecon.var_forecast(X3, lags=2, steps=8, band="bonferroni"),
}


def main() -> None:
    for name, call in CALLS.items():
        try:
            out = call()
            print(f"{name:28s} {fingerprint(out)}")
        except Exception as exc:  # noqa: BLE001 - a refusal is part of the record
            print(f"{name:28s} RAISED {type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    main()
