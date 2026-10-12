"""Monte-Carlo recovery of a stationary damped cycle by unobserved_components.

The seeded design of crates/tsecon-ssm/tests/uc_properties.rs
(``stationary_cycle_mle_recovers_damping_and_period``), run over many more
seeds than that 8-seed smoke test: a local level (variance 0.01) plus a
damped stochastic cycle (period 12, damping 0.9, variance 0.5) started from
its stationary distribution, irregular variance 0.2, T = 200; the fit is
``level="llevel"``, a damped stochastic cycle with ``cycle_init="stationary"``
and ``cycle_period_bounds=(4, 40)``, default ``n_starts``. The series are the
test's own: the same Knuth MMIX LCG (``Lcg::new(31000 + seed)``) and
Box-Muller ``normal()``, transcribed, so seeds 0..7 reproduce the test.
Its output is quoted in docs/reference/model-cards/unobserved-components.md.

    .venv/bin/python scripts/mc_uc_stationary_cycle_recovery.py [n_seeds]

Reports the median absolute period and damping errors with bootstrap
standard errors (2000 resamples, ``default_rng(0)``), and how many fits do
not converge or land outside the test's band (period within 3, damping
within 0.15).
"""
import math
import sys

import numpy as np

import tsecon

MASK = (1 << 64) - 1


class Lcg:
    def __init__(self, seed):
        self.s = (seed * 0x9E3779B97F4A7C15 + 1) & MASK

    def uniform(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & MASK
        return (self.s >> 11) / float(1 << 53)


def normal(rng):
    u1 = max(rng.uniform(), 1e-300)
    u2 = rng.uniform()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def series(seed, n=200, period=12.0, rho=0.9, s2c=0.5, s2l=0.01, s2e=0.2):
    lam = 2.0 * math.pi / period
    sd0 = math.sqrt(s2c / (1.0 - rho * rho))
    rng = Lcg(31_000 + seed)
    c, cs = sd0 * normal(rng), sd0 * normal(rng)
    level = 0.0
    y = []
    for _ in range(n):
        y.append(level + c + math.sqrt(s2e) * normal(rng))
        c, cs = (rho * (math.cos(lam) * c + math.sin(lam) * cs) + math.sqrt(s2c) * normal(rng),
                 rho * (-math.sin(lam) * c + math.cos(lam) * cs) + math.sqrt(s2c) * normal(rng))
        level += math.sqrt(s2l) * normal(rng)
    return np.array(y)


def boot_se(v, b=2000):
    rng = np.random.default_rng(0)
    v = np.asarray(v)
    return float(np.std([np.median(rng.choice(v, v.size)) for _ in range(b)], ddof=1))


def main(n_seeds):
    ep, ed, bad = [], [], []
    for seed in range(n_seeds):
        f = tsecon.unobserved_components(series(seed), level="llevel", cycle=True, damped_cycle=True,
                                         stochastic_cycle=True, cycle_period_bounds=(4.0, 40.0),
                                         cycle_init="stationary")
        names = list(f["param_names"])
        p_hat = 2.0 * math.pi / f["params"][names.index("frequency.cycle")]
        d_hat = f["params"][names.index("damping.cycle")]
        ep.append(abs(p_hat - 12.0))
        ed.append(abs(d_hat - 0.9))
        if not f["converged"] or ep[-1] >= 3.0 or ed[-1] >= 0.15:
            bad.append((seed, bool(f["converged"]), round(p_hat, 2), round(d_hat, 6)))
    print(f"tsecon {tsecon.__version__}, {n_seeds} seeds, T = 200")
    print(f"  seeds 0..7: median |period error| {np.median(ep[:8]):.4f}, median |damping error| {np.median(ed[:8]):.5f}")
    print(f"  all: median |period error| {np.median(ep):.3f} (bootstrap se {boot_se(ep):.3f}), "
          f"median |damping error| {np.median(ed):.4f} (bootstrap se {boot_se(ed):.4f})")
    print(f"  not converged or outside the band: {len(bad)} of {n_seeds}: {bad}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
