"""Golden fixtures for simultaneous (sup-t) critical values.

Reference implementations (this venv):
  * pointwise / Bonferroni / Sidak -> scipy.stats.norm.ppf on the documented
    per-cell levels:
        pointwise   z = Phi^-1(1 - alpha/2)
        Bonferroni  z = Phi^-1(1 - alpha/(2K))
        Sidak       z = Phi^-1(1 - a_K/2),  a_K = 1 - (1-alpha)^(1/K)
  * sup-t from draws -> numpy: max over cells of |draw - theta_hat|/se per
    draw, then numpy.quantile(., 1-alpha) with the default linear
    interpolation (Hyndman-Fan type 7).
  * sup-t from a covariance -> numpy.linalg.cholesky + scipy.stats.norm.ppf +
    numpy.quantile, on uniforms produced by the SplitMix64 recipe below. The
    Rust test reimplements that generator in u64 arithmetic, so both sides see
    the identical uniform stream with no megabyte of JSON; `uniform_head` and
    `uniform_mean` are stored so a mismatch in the generator is diagnosed
    before the critical values are compared.

    The Rust routine shifts each uniform by half a 2^-53 grid cell before
    inverting (so an exact 0.0 draw cannot invert to -inf); this generator does
    not. The shift shows up as a ~1e-11 absolute difference in the deepest tail
    draws and far less in the quantile, so the Rust golden compares at 1e-8
    absolute.

The sup-t construction is the one in Montiel Olea and Plagborg-Moller,
"Simultaneous confidence bands: Theory, implementation, and an application to
SVARs".

This generator NEVER imports tsecon. Doubles are written with json's shortest
round-trip repr.

Run:  python fixtures/generate_simultaneous_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import norm

MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
SPLIT_A = 0xBF58476D1CE4E5B9
SPLIT_B = 0x94D049BB133111EB


def splitmix64_uniforms(seed: int, n: int) -> np.ndarray:
    """Uniforms on [0, 1) from SplitMix64, as `(next_u64 >> 11) * 2**-53`.

    Reimplemented bit-for-bit in the Rust test; keep the two in step.
    """
    state = seed & MASK64
    out = np.empty(n, dtype=np.float64)
    scale = 2.0**-53
    for i in range(n):
        state = (state + GOLDEN_GAMMA) & MASK64
        z = state
        z = ((z ^ (z >> 30)) * SPLIT_A) & MASK64
        z = ((z ^ (z >> 27)) * SPLIT_B) & MASK64
        z = z ^ (z >> 31)
        out[i] = float(z >> 11) * scale
    return out


def ar1_corr(k: int, rho: float) -> np.ndarray:
    idx = np.arange(k)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def irf_shaped_se(k: int) -> np.ndarray:
    """Standard errors with the hump-then-decay shape of an IRF path."""
    h = np.arange(k, dtype=float)
    return 0.15 + 0.45 * np.exp(-0.5 * ((h - 2.0) / 4.0) ** 2)


def closed_form_block() -> list[dict]:
    rows = []
    for alpha in (0.32, 0.10, 0.05, 0.01):
        for k in (1, 2, 3, 5, 13, 20, 60, 200):
            per_cell_sidak = 1.0 - (1.0 - alpha) ** (1.0 / k)
            rows.append(
                {
                    "alpha": alpha,
                    "k": k,
                    "pointwise": float(norm.ppf(1.0 - alpha / 2.0)),
                    "bonferroni": float(norm.ppf(1.0 - alpha / (2.0 * k))),
                    "sidak": float(norm.ppf(1.0 - per_cell_sidak / 2.0)),
                }
            )
    return rows


def sup_t_from_draws(draws: np.ndarray, theta_hat: np.ndarray, se: np.ndarray,
                     alpha: float) -> float:
    """(1-alpha) quantile of the max over non-degenerate cells of |t|."""
    keep = se > 0.0
    t = np.abs(draws[:, keep] - theta_hat[keep]) / se[keep]
    return float(np.quantile(t.max(axis=1), 1.0 - alpha))


def draws_block(rng: np.random.Generator) -> dict:
    k, n_draws = 13, 999
    corr = ar1_corr(k, 0.85)
    se = irf_shaped_se(k)
    sigma = corr * np.outer(se, se)
    chol = np.linalg.cholesky(sigma)
    theta_hat = 0.8 * 0.85 ** np.arange(k)
    draws = theta_hat + rng.standard_normal((n_draws, k)) @ chol.T

    # A second standard-error vector with cell 6 pinned by construction
    # (proxy-SVAR normalization): zero se, must drop out of the maximum.
    se_pinned = se.copy()
    se_pinned[6] = 0.0
    draws_pinned = draws.copy()
    draws_pinned[:, 6] = theta_hat[6]

    alphas = [0.32, 0.10, 0.05, 0.01]
    return {
        "k": k,
        "n_draws": n_draws,
        "theta_hat": theta_hat.tolist(),
        "se": se.tolist(),
        "se_pinned": se_pinned.tolist(),
        "draws_row_major": draws.ravel(order="C").tolist(),
        "draws_pinned_row_major": draws_pinned.ravel(order="C").tolist(),
        "alphas": alphas,
        "critical_value": [sup_t_from_draws(draws, theta_hat, se, a) for a in alphas],
        "critical_value_pinned": [
            sup_t_from_draws(draws_pinned, theta_hat, se_pinned, a) for a in alphas
        ],
    }


def cov_case(name: str, sigma: np.ndarray, alphas: list[float], n_sim: int,
             seed: int) -> dict:
    k = sigma.shape[0]
    u = splitmix64_uniforms(seed, n_sim * k)
    assert u.min() > 0.0, "an exact-zero uniform would need the shift to match"
    z = norm.ppf(u).reshape(n_sim, k)
    chol = np.linalg.cholesky(sigma)
    x = z @ chol.T
    se = np.sqrt(np.diag(sigma))
    m = (np.abs(x) / se).max(axis=1)
    return {
        "name": name,
        "k": k,
        "n_sim": n_sim,
        "seed": seed,
        "sigma_row_major": sigma.ravel(order="C").tolist(),
        "uniform_head": u[:8].tolist(),
        "uniform_mean": float(u.mean()),
        "alphas": alphas,
        "critical_value": [float(np.quantile(m, 1.0 - a)) for a in alphas],
    }


def cov_block() -> list[dict]:
    alphas = [0.32, 0.10, 0.05, 0.01]
    n_sim = 100_000
    cases = []

    k = 13
    se = irf_shaped_se(k)

    # Persistent IRF-like path: adjacent horizons correlated ~0.85.
    sigma = ar1_corr(k, 0.85) * np.outer(se, se)
    cases.append(cov_case("irf_ar1_rho085", sigma, alphas, n_sim, 0x5EED_0001))

    # Near-independent cells: sup-t should land close to Sidak here.
    sigma = np.diag(se**2)
    cases.append(cov_case("independent", sigma, alphas, n_sim, 0x5EED_0002))

    # Almost perfectly dependent: sup-t should land close to pointwise.
    sigma = ar1_corr(k, 0.999) * np.outer(se, se)
    cases.append(cov_case("nearly_collinear", sigma, alphas, n_sim, 0x5EED_0003))

    # A single cell, every route's collapse case.
    cases.append(cov_case("single_cell", np.array([[0.37**2]]), alphas,
                          n_sim, 0x5EED_0004))

    return cases


def main() -> None:
    rng = np.random.default_rng(20260807)
    payload = {
        "meta": {
            "generator": "fixtures/generate_simultaneous_fixtures.py",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "method": (
                "sup-t simultaneous confidence bands, Montiel Olea and "
                "Plagborg-Moller"
            ),
            "quantile_convention": "numpy.quantile default (Hyndman-Fan type 7)",
            "uniform_generator": (
                "SplitMix64 seeded per case; u = (next_u64 >> 11) * 2**-53"
            ),
        },
        "closed_form": closed_form_block(),
        "sup_t_draws": draws_block(rng),
        "sup_t_cov": cov_block(),
    }
    out = Path(__file__).with_name("simultaneous.json")
    out.write_text(json.dumps(payload))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
