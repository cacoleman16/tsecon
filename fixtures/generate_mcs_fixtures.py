"""Golden fixtures for the Hansen-Lunde-Nason (2011) Model Confidence Set
(`model_confidence_set`).

Reference: arch 8.0.0 `arch.bootstrap.MCS(losses, size, reps, block_size,
method, bootstrap, seed).compute()` — an INDEPENDENT package — read through
`included`, `excluded`, `pvalues` (the MCS p-values, a running maximum along
the elimination path, indexed by model in elimination order) and, for the
range method, `_variances`.

What is stored, and the honest grade
------------------------------------
Every case holds the T x m loss panel (as m columns), the `reps` resample
index arrays arch drew (`MCS._bootstrap_indices`, kept by arch "for
testing"), and arch's results. The Rust core cannot reproduce NumPy's
generator, so the EXACT leg feeds those indices through
`model_confidence_set_with_indices`:

* INDEPENDENT-PACKAGE golden, EXACT: `mean_losses`, `included`,
  `excluded`, `elimination_order` (the index of arch's `pvalues` frame),
  `mcs_p_values` (per model), and for method "R" the pairwise bootstrap
  variance matrix. The generator recomputes the whole elimination from the
  stored indices with a transcription of arch's `_compute_r` /
  `_compute_max` whose summation orders are the ones the Rust uses
  (sequential over periods and replications, NumPy's pairwise summation
  over the models for the cross-sectional means of the max method), and
  asserts bit-for-bit agreement of the variances, the running-max p-values
  and the elimination order before storing a case. The minimum gap between
  the bootstrap maxima and the observed statistic at every step, and the
  margin between the eliminated model's statistic and the runner-up, are
  stored (`min_gap`, `min_margin`) and asserted > 1e-9, so the p-values
  and the elimination path are exact by construction.

* CROSS-IMPLEMENTATION transcription: the per-step raw p-values
  (`step_p_values`) and the observed statistics T_R / T_max per step
  (`statistics`), which arch does not expose — computed by the same
  transcription whose running maximum IS arch's `pvalues`.

* MONTE CARLO leg (`mc`): arch's included set and MCS p-values at 4000
  replications with another seed; tsecon's own seeded bootstrap at 4000
  replications must reproduce the set and land within 0.05 of each
  p-value (two independent 4000-draw bootstraps differ by ~0.011 sd at
  p = 0.5). The designs are separated so that every MCS p-value at the MC
  leg is at least 5 bootstrap-to-bootstrap standard deviations
  (sqrt(2 p (1 - p) / 4000)) plus 0.01 away from `size` (asserted), which
  is what makes the set comparison meaningful rather than a coin flip.

arch's default block size is int(sqrt(T)); tsecon's is the Politis-White
optimal length (averaged over the loss columns). Every case here passes
`block_size` explicitly so that difference never enters the pin.

This generator NEVER imports tsecon. It imports the shared NumPy-order
helpers from generate_spa_fixtures.py (imported, not re-typed).

Run:  .venv/bin/python fixtures/generate_mcs_fixtures.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

import arch
from arch.bootstrap import MCS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_spa_fixtures import (  # noqa: E402
    assert_bits,
    col_mean_seq,
    col_means_seq,
    loss_panel,
    pairwise_sum,
)

ARCH_NAME = {"stationary": "stationary", "circular": "circular", "moving_block": "moving block"}


# --------------------------------------------------------------------------
# The MCS transcription (mirrors crates/tsecon-forecast/src/mcs.rs)
# --------------------------------------------------------------------------


def mcs_transcribe(cols, resamples, size, method):
    n = len(cols[0])
    m = len(cols)
    reps = len(resamples)
    mean = [col_mean_seq(c) for c in cols]
    order, pvals, stats, gaps, margins = [], [], [], [], []
    included = [True] * m
    extra = {}
    if method == "R":
        ms = [col_means_seq(cols, idx) for idx in resamples]
        var = [[0.0] * m for _ in range(m)]
        for b in range(reps):
            for i in range(m):
                for j in range(m):
                    d = (ms[b][i] - ms[b][j]) - (mean[i] - mean[j])
                    var[i][j] += d * d
        var = [[v / reps for v in row] for row in var]
        for i in range(m):
            var[i][i] += 1.0
        sd = [[math.sqrt(v) for v in row] for row in var]
        std_ld = [[(mean[i] - mean[j]) / sd[i][j] for j in range(m)] for i in range(m)]
        extra["variances"] = var
        n_incl = m
        while n_incl > 1:
            incl = [k for k in range(m) if included[k]]
            test_stat = -math.inf
            loc = 0
            vals = []
            for a, i in enumerate(incl):
                for j in incl:
                    v = std_ld[i][j]
                    vals.append(v)
                    if v > test_stat:
                        test_stat = v
                        loc = a
            runner = max([v for v in vals if v != test_stat], default=-math.inf)
            margins.append((test_stat - runner) / (1.0 + abs(test_stat)))
            count = 0
            gap = math.inf
            for b in range(reps):
                mx = -math.inf
                for i in incl:
                    for j in incl:
                        z = ((ms[b][i] - ms[b][j]) - (mean[i] - mean[j])) / sd[i][j]
                        if z > mx:
                            mx = z
                if test_stat < mx:
                    count += 1
                gap = min(gap, abs(mx - test_stat) / (1.0 + abs(test_stat)))
            gaps.append(gap)
            pval = count / reps
            out = incl[loc]
            order.append(out)
            pvals.append(pval)
            stats.append(test_stat)
            included[out] = False
            n_incl -= 1
    elif method == "max":
        errs = [[v - mu for v in c] for c, mu in zip(cols, mean)]
        rows = []
        for idx in resamples:
            row = col_means_seq(errs, idx)
            grand = pairwise_sum(row) / m
            rows.append([v - grand for v in row])
        n_incl = m
        while n_incl > 1:
            incl = [k for k in range(m) if included[k]]
            kk = len(incl)
            ss = [0.0] * kk
            centred = []
            for b in range(reps):
                tmp = [rows[b][k] for k in incl]
                grand = pairwise_sum(tmp) / kk
                c = [v - grand for v in tmp]
                centred.append(c)
                for a in range(kk):
                    ss[a] += c[a] * c[a]
            sd = [math.sqrt(s / reps) for s in ss]
            assert all(s > 0 for s in sd), "zero std in max method"
            ld = [mean[k] for k in incl]
            gm = pairwise_sum(ld) / kk
            ld = [v - gm for v in ld]
            std_ld = [v / s for v, s in zip(ld, sd)]
            test_stat = max(std_ld)
            runner = max([v for v in std_ld if v != test_stat], default=-math.inf)
            margins.append((test_stat - runner) / (1.0 + abs(test_stat)))
            count = 0
            gap = math.inf
            for b in range(reps):
                mx = max(c / s for c, s in zip(centred[b], sd))
                if test_stat < mx:
                    count += 1
                gap = min(gap, abs(mx - test_stat) / (1.0 + abs(test_stat)))
            gaps.append(gap)
            pval = count / reps
            for a, v in enumerate(std_ld):
                if v == test_stat:
                    order.append(incl[a])
                    pvals.append(pval)
                    included[incl[a]] = False
                    n_incl -= 1
            stats.append(test_stat)
    else:
        raise ValueError(method)
    for k in range(m):
        if included[k]:
            order.append(k)
            pvals.append(1.0)
    mcs_p = [0.0] * m
    running = -math.inf
    for k, p in zip(order, pvals):
        if p > running:
            running = p
        mcs_p[k] = running
    return {
        "mean_losses": mean,
        "elimination_order": order,
        "step_p_values": pvals,
        "statistics": stats,
        "mcs_p_values": mcs_p,
        "included": [k for k in range(m) if mcs_p[k] > size],
        "excluded": [k for k in range(m) if mcs_p[k] <= size],
        "n_steps": len(stats),
        "min_gap": min(gaps),
        "min_margin": min(margins),
        **extra,
    }


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

CASES = [
    # name, n, method, bootstrap, block_size, size, seed, scales, bias
    ("range_stationary_m5", 100, "R", "stationary", 5, 0.10, 7,
     [0.8, 0.8, 1.0, 1.2, 1.5], [0.0, 0.0, 0.9, 1.2, 1.5]),
    ("max_stationary_m5", 100, "max", "stationary", 5, 0.10, 7,
     [0.8, 0.8, 1.0, 1.2, 1.5], [0.0, 0.0, 0.9, 1.2, 1.5]),
    ("range_circular_m4", 100, "R", "circular", 4, 0.05, 11,
     [0.9, 0.9, 0.9, 1.4], [0.0, 0.0, 0.0, 0.8]),
    ("max_moving_block_m3", 80, "max", "moving_block", 3, 0.10, 3,
     [0.8, 1.0, 1.5], [0.0, 0.4, 1.0]),
    ("range_two_models", 90, "R", "stationary", 4, 0.10, 5,
     [0.9, 1.3], [0.0, 0.7]),
    # Equal predictive ability: every model should survive.
    ("range_all_equal", 100, "R", "circular", 5, 0.10, 19,
     [1.0, 1.0, 1.0, 1.0], None),
    ("max_all_equal", 100, "max", "circular", 5, 0.10, 19,
     [1.0, 1.0, 1.0, 1.0], None),
]
REPS = 200
MC_REPS = 4000


def run_arch(losses_arr, size, reps, block_size, method, kind, seed):
    mcs = MCS(losses_arr, size, reps=reps, block_size=block_size, method=method,
              bootstrap=ARCH_NAME[kind], seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mcs.compute()
    return mcs


def run_case(name, n, method, kind, block_size, size, seed, scales, bias):
    rng = np.random.default_rng(2000 + seed)
    cols = loss_panel(rng, n, scales, bias=bias)
    m = len(cols)
    losses_arr = np.column_stack([np.asarray(c) for c in cols])

    mcs = run_arch(losses_arr, size, REPS, block_size, method, kind, seed)
    resamples = [np.asarray(r, dtype=int).tolist() for r in mcs._bootstrap_indices]
    assert len(resamples) == REPS
    tr = mcs_transcribe(cols, resamples, size, method)

    arch_order = [int(v) for v in mcs.pvalues.index]
    arch_p_in_order = mcs.pvalues["Pvalue"].to_numpy(dtype=float)
    assert tr["elimination_order"] == arch_order, f"{name}: elimination order {tr['elimination_order']} vs arch {arch_order}"
    mine_in_order = [tr["mcs_p_values"][k] for k in arch_order]
    assert_bits(mine_in_order, arch_p_in_order, f"{name}: MCS p-values")
    assert tr["included"] == [int(v) for v in mcs.included], f"{name}: included set"
    assert tr["excluded"] == [int(v) for v in mcs.excluded], f"{name}: excluded set"
    assert_bits(tr["mean_losses"], losses_arr.mean(0), f"{name}: mean losses")
    if method == "R":
        assert_bits(np.asarray(tr["variances"]), mcs._variances, f"{name}: pairwise variances")
    assert tr["min_gap"] > 1e-9, f"{name}: bootstrap maximum within 1e-9 of a step statistic ({tr['min_gap']})"
    assert tr["min_margin"] > 1e-9, f"{name}: elimination argmax within 1e-9 of the runner-up ({tr['min_margin']})"

    # Monte Carlo leg.
    mc_seed = seed + 500
    mc = run_arch(losses_arr, size, MC_REPS, block_size, method, kind, mc_seed)
    mc_p = [float(mc.pvalues.loc[k, "Pvalue"]) for k in range(m)]
    # Separation in units of the bootstrap-to-bootstrap standard deviation
    # sqrt(2 p (1 - p) / B) (two independent B-draw bootstraps), with a 0.01
    # floor: an independent bootstrap must reproduce the set, not flip it.
    for p_k in mc_p:
        sd = math.sqrt(2 * p_k * (1 - p_k) / MC_REPS)
        assert abs(p_k - size) >= 5 * sd + 0.01, \
            f"{name}: MC-leg p-value {p_k} sits within 5 sd + 0.01 of size {size} ({mc_p})"
    mc_block = {
        "reps": MC_REPS,
        "arch_seed": mc_seed,
        "included": [int(v) for v in mc.included],
        "excluded": [int(v) for v in mc.excluded],
        "mcs_p_values": mc_p,
    }

    print(f"{name:22s} n={n:3d} m={m} {method:3s} {kind:12s} b={block_size} size={size} "
          f"included={tr['included']} order={tr['elimination_order']} "
          f"p={[round(p, 3) for p in tr['mcs_p_values']]} steps={tr['n_steps']} "
          f"min_gap={tr['min_gap']:.1e} margin={tr['min_margin']:.1e} mc_incl={mc_block['included']}")
    case = {
        "name": name,
        "n": n,
        "m": m,
        "method": method,
        "bootstrap": kind,
        "block_size": block_size,
        "size": size,
        "arch_seed": seed,
        "reps": REPS,
        "scales": scales,
        "losses": cols,
        "resamples": resamples,
        "mc": mc_block,
    }
    case.update({k: v for k, v in tr.items()})
    return case


def main():
    cases = [run_case(*c) for c in CASES]
    out = {
        "_meta": {
            "generator": "fixtures/generate_mcs_fixtures.py",
            "reference": "arch.bootstrap.MCS (independent package); per-step statistics from the asserted-bit-identical transcription",
            "arch": arch.__version__,
            "numpy": np.__version__,
            "notes": [
                "resamples[b] = MCS._bootstrap_indices[b], the indices arch's default_rng(arch_seed) bootstrap drew for replication b.",
                "mcs_p_values[k] is arch's `pvalues` (running maximum of the step p-values along the elimination path); step_p_values are the raw step p-values in elimination order (survivors 1.0); statistics are the observed T_R / T_max per step.",
                "The transcription's variances, elimination order and running-max p-values were asserted bit-identical to arch for every case; min_gap / min_margin > 1e-9 certify the decisions cannot flip under sub-1e-9 arithmetic differences.",
                "arch warns and continues on a zero T_max standard deviation; tsecon refuses (no such case is stored).",
            ],
            "mc_tolerance": 0.05,
        },
        "cases": cases,
    }
    path = HERE / "mcs.json"
    path.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
