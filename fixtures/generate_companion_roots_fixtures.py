"""Golden fixtures for companion-matrix eigenvalues on BADLY SCALED VARs
(`tsecon_linalg::balanced_eigenvalues`, `spectral_radius`, `is_stable`, and
through them `VarResults::roots_moduli` / `is_stable`, the `min_root` /
`max_root` / `is_stable` keys of `var_fit`, and `min_root` /
`n_unstable_draws` / the Kilian shrink of `long_run_svar_bands` and
`var_irf_bands`).

    PYTHONPATH=<dir with mpmath> .venv/bin/python fixtures/generate_companion_roots_fixtures.py

Why: rescaling series i of a VAR by s_i is the diagonal similarity
A_l -> S A_l S^-1 of every lag matrix, so it leaves the roots unchanged. A
QR eigenvalue iteration run on the raw companion matrix is only accurate
relative to the matrix norm, though, and with series in units 1e14 apart
(a level in currency units next to a rate in decimals) the companion's
entries span about 1e-15 to 1e14, so its small and near-unit eigenvalues
come out wrong in the third decimal. LAPACK's dgeev balances first (dgebal:
a diagonal similarity by powers of two, exact in floating point); the
fixture pins the balanced solve against an arbitrary-precision reference.

Reference, independent of any balancing: mpmath (`mp.eig`, 60 significant
digits) on the EXACT float64 companion matrix, unbalanced. At 60 digits the
raw solve's error relative to the norm (about 1e-60 x 1e14) is far below
double precision, so every stored modulus is correct to the last bit of a
double. Each case is cross-checked against numpy.linalg.eigvals (LAPACK
dgeev, which balances) at 1e-13 relative before anything is written.

CASES
-----
* `companions`: lag-matrix sets with their companion eigenvalue moduli
  (descending) and spectral radius.
  - `seed21_var2_scaled`: the audit's near-unit-root VAR(2), k = 3,
    companion radius 0.9995 (numpy default_rng(21); random draws rescaled
    until the radius hits the target), under S = diag(1e7, 1, 1e-7).
  - `seed21_var2_unscaled`: the same lag matrices, S = I.
  - `var3_k4_complex`: a VAR(3), k = 4 (default_rng(7)) with complex
    roots, radius 0.97, under S = diag(1e6, 1e-3, 1, 1e-6).
  - `var2_k3_zero_column`: a VAR(2), k = 3 (default_rng(11)) whose second
    lag matrix has a zero column (a zero companion row/column pair that the
    balancing must skip), under S = diag(1e5, 1, 1e-5).
* `var_fit_scaled`: the seed-21 DGP simulated for 600 periods (the first
  200 discarded), its 400 rows multiplied column-wise by (1e7, 1, 1e-7). A
  VAR(2) with a constant is fitted here by column-equilibrated least squares
  (numpy.linalg.lstsq on the design divided by its column norms), and the
  reciprocal companion moduli of THOSE coefficients are stored as `roots`
  (statsmodels `VARResults.roots` convention, descending). tsecon's own OLS
  agrees with this one to about 1e-13 relative (the audit measured 2.5e-13),
  so the roots are pinned at 1e-9; the unbalanced solve was off by 2.3e-3
  in `min_root` here (0.997715 against 1.000887), the wrong side of 1.

The data rows are seeded simulations, not a dataset. Never imports tsecon.
Doubles are written with json's shortest round-trip repr.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import mpmath
import numpy as np

mpmath.mp.dps = 60

OUT = Path(__file__).with_name("companion_roots.json")


def companion(coefs: list[np.ndarray]) -> np.ndarray:
    k = coefs[0].shape[0]
    p = len(coefs)
    c = np.zeros((k * p, k * p))
    c[:k] = np.hstack(coefs)
    c[k:, : k * (p - 1)] = np.eye(k * (p - 1))
    return c


def np_radius(coefs: list[np.ndarray]) -> float:
    return float(np.abs(np.linalg.eigvals(companion(coefs))).max())


def mp_moduli(c: np.ndarray) -> list[float]:
    """Eigenvalue moduli of the exact float64 matrix `c` at 60 digits.

    A modulus below 1e-40 x the radius is the 60-digit iteration's residue of
    an exactly zero eigenvalue (a zero companion column) and is stored as 0.
    """
    a = mpmath.matrix([[mpmath.mpf(float(v)) for v in row] for row in c])
    ev = mpmath.eig(a, left=False, right=False)
    mods = sorted((abs(e) for e in ev), reverse=True)
    return [float(m) if m > mpmath.mpf("1e-40") * mods[0] else 0.0 for m in mods]


def scaled(coefs: list[np.ndarray], s: np.ndarray) -> list[np.ndarray]:
    return [np.diag(s) @ a @ np.diag(1.0 / s) for a in coefs]


def random_var(rng: np.random.Generator, k: int, p: int, target: float) -> list[np.ndarray]:
    """The audit's rescale-to-target-radius draw (svarfuzz/repro_minroot_flip.py)."""
    while True:
        a = [rng.standard_normal((k, k)) * (0.5 / (i + 1)) / np.sqrt(k) for i in range(p)]
        r = np_radius(a)
        if r < 1e-6:
            continue
        s = target / r
        a = [m * s ** (i + 1) for i, m in enumerate(a)]
        if np_radius(a) < 0.9999:
            return a


def case(name: str, coefs: list[np.ndarray]) -> dict:
    c = companion(coefs)
    mods = mp_moduli(c)
    ref = np.sort(np.abs(np.linalg.eigvals(c)))[::-1]
    m = np.array(mods)
    # Relative per modulus; an exactly zero eigenvalue (a zero companion
    # column) is compared absolutely, against the radius.
    rel = float(np.max(np.abs(ref - m) / np.where(m > 0.0, m, mods[0])))
    assert rel < 1e-13, (name, rel)
    entries = np.abs(c[c != 0.0])
    return {
        "name": name,
        "k": int(coefs[0].shape[0]),
        "lags": len(coefs),
        "coefs": [m.tolist() for m in coefs],
        "moduli": mods,
        "spectral_radius": mods[0],
        "entry_magnitude_range": [float(entries.min()), float(entries.max())],
        "numpy_geev_max_rel_diff": float(rel),
    }


def ols_var(y: np.ndarray, p: int) -> list[np.ndarray]:
    """VAR(p) with a constant by column-equilibrated least squares."""
    t, k = y.shape
    z = np.hstack([np.ones((t - p, 1))] + [y[p - i : t - i] for i in range(1, p + 1)])
    cn = np.linalg.norm(z, axis=0)
    b = np.linalg.lstsq(z / cn, y[p:], rcond=None)[0] / cn[:, None]
    return [b[1 + i * k : 1 + (i + 1) * k].T for i in range(p)]


def main() -> None:
    companions = []

    # The audit's seed-21 DGP (svarfuzz/repro_minroot_flip.py), k = 3, p = 2.
    k, p, t_keep, burn = 3, 2, 400, 200
    rng = np.random.default_rng(21)
    a21 = random_var(rng, k, p, 0.9995)
    s21 = np.array([1e7, 1.0, 1e-7])
    companions.append(case("seed21_var2_scaled", scaled(a21, s21)))
    companions.append(case("seed21_var2_unscaled", a21))

    # A VAR(3), k = 4, with complex roots.
    rng7 = np.random.default_rng(7)
    a7 = random_var(rng7, 4, 3, 0.97)
    companions.append(
        case("var3_k4_complex", scaled(a7, np.array([1e6, 1e-3, 1.0, 1e-6])))
    )

    # A VAR(2), k = 3, whose second lag matrix has a zero column.
    rng11 = np.random.default_rng(11)
    a11 = random_var(rng11, 3, 2, 0.9)
    a11[1][:, 1] = 0.0
    companions.append(
        case("var2_k3_zero_column", scaled(a11, np.array([1e5, 1.0, 1e-5])))
    )
    assert any(abs(e.imag) > 1e-3 for e in np.linalg.eigvals(companion(a7)))

    # The seed-21 DGP simulated and rescaled; roots of an independent OLS fit.
    n = t_keep + burn
    u = rng.standard_normal((n, k))
    y = np.zeros((n, k))
    for tt in range(p, n):
        y[tt] = u[tt] + sum(a21[i] @ y[tt - 1 - i] for i in range(p))
    y0 = y[burn:]
    ys = y0 * s21
    coefs = ols_var(ys, p)
    mods = mp_moduli(companion(coefs))
    roots = sorted((1.0 / m for m in mods), reverse=True)
    coefs0 = ols_var(y0, p)
    roots0 = sorted((1.0 / m for m in mp_moduli(companion(coefs0))), reverse=True)
    assert abs(roots[-1] - roots0[-1]) < 1e-9 * roots0[-1], (roots[-1], roots0[-1])
    assert roots[-1] > 1.0 + 1e-4, roots[-1]  # stable, by a clear margin
    var_fit_scaled = {
        "lags": p,
        "trend": "c",
        "scale": s21.tolist(),
        "data": ys.tolist(),
        "roots": roots,
        "min_root": roots[-1],
        "max_root": roots[0],
        "is_stable": bool(roots[-1] > 1.0),
        "unscaled_min_root": roots0[-1],
    }

    out = {
        "_meta": {
            "generator": "fixtures/generate_companion_roots_fixtures.py",
            "numpy": np.__version__,
            "mpmath": mpmath.__version__,
            "mp_dps": mpmath.mp.dps,
            "python": platform.python_version(),
        },
        "companions": companions,
        "var_fit_scaled": var_fit_scaled,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for c in companions:
        print(
            f"{c['name']:24s} radius {c['spectral_radius']:.15f} entries "
            f"{c['entry_magnitude_range'][0]:.2e}..{c['entry_magnitude_range'][1]:.2e} "
            f"geev rel {c['numpy_geev_max_rel_diff']:.1e}"
        )
    print(f"var_fit_scaled min_root {roots[-1]:.15f} (unscaled {roots0[-1]:.15f})")


if __name__ == "__main__":
    main()
