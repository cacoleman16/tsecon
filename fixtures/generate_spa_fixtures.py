"""Golden fixtures for White's (2000) Reality Check / Hansen's (2005) test
for Superior Predictive Ability (`spa_test`) and the Romano-Wolf (2005)
StepM procedure (`stepm_test`).

Reference: arch 8.0.0 — `arch.bootstrap.SPA`, `arch.bootstrap.RealityCheck`
and `arch.bootstrap.StepM` (an INDEPENDENT package), plus a NumPy
transcription of Hansen's studentized statistic on the same resamples.

What is stored, and the honest grade of each block
--------------------------------------------------
Every case holds the benchmark losses, the model loss columns, and the
`reps` resample index arrays arch drew from its NumPy `default_rng(seed)`
(200 replications of T = 80..100 rows, integers). The Rust core cannot
reproduce NumPy's generator, so the exact leg feeds those indices through
`spa_test_with_indices`; the public seeded `spa_test` is pinned separately
at Monte Carlo tolerance (the `mc` block, 4000 replications).

* "arch" (INDEPENDENT-PACKAGE golden, EXACT): `mean_loss_diff`
  (`_loss_diff.mean(0)`), `loss_diff_var` (`_loss_diff_var`, Hansen's 2005
  eq. 9 stationary-bootstrap kernel with q = 1/block_size, or the nested
  bootstrap variance), the consistent-recentring flags, the three replicate
  statistics per replication (`_simulated_vals.max(0)`, the max over models
  of the re-centred resampled means, on the MEAN scale — arch does not
  scale by sqrt(T)), the three p-values, and the critical values at
  90/95/99% (`critical_values`). The generator ALSO recomputes every one of
  these from the stored indices with explicit sequential loops (the
  summation order the Rust uses) and asserts bit-for-bit agreement with
  arch before storing the case (`bit_exact: true`, every case with m >= 2
  models); the minimum gap between any replicate statistic and the observed
  statistic is stored (`min_gap`) and asserted > 1e-9, so the p-values are
  exact by construction, not by luck. MEASURED EXCEPTION: NumPy reduces a
  C-ordered T x m panel sequentially over t for m >= 2, but coalesces a
  T x 1 panel into a 1-D reduction and sums it pairwise (asserted below),
  so the single-model case (`bit_exact: false`) is pinned at 1e-13 with
  its p-values still exact.

  THE FINDING THIS GENERATOR RECORDS: arch's `studentize` flag is INERT in
  8.0.0 — `_loss_diff_var` feeds only the consistent-recentring threshold,
  never a division — and the generator ASSERTS identical p-values and
  critical values with the flag on and off (`_meta.studentize_inert`).
  `arch.bootstrap.SPA` is therefore White's un-studentized Reality Check
  statistic with Hansen's three re-centrings, and `RealityCheck` is
  literally `class RealityCheck(SPA): pass`. tsecon's `studentize=False`
  path is what this block pins.

* "studentized" (DOCUMENTED-FORMULA golden, Hansen 2005): the same
  quantities with each loss differential divided by its omega_k =
  sqrt(loss_diff_var_k), i.e. T = sqrt(n) max_k dbar_k / omega_k and
  T*_b = sqrt(n) max_k (dbar*_{b,k} - g_k) / omega_k, computed by this
  file's transcription on arch's resamples. No package studentizes, so this
  block is a cross-implementation pin of the published formula, not a
  third-party number. Every statistic is reported on the sqrt(T) scale;
  the "arch" block's mean-scale numbers times sqrt(T) are what the Rust
  reports for `studentize=False`.

* "stepm" (INDEPENDENT-PACKAGE golden for the un-studentized rule):
  `arch.bootstrap.StepM(...).superior_models` at the stated size, asserted
  equal to the transcription's stepwise rule on the same resamples; the
  studentized StepM set comes from the transcription only. SECOND FINDING:
  arch's StepM raises `ValueError: zero-size array to reduction operation
  maximum` whenever every model is declared superior over two or more
  steps (its loop re-runs the SPA on an empty selector); such cases store
  `arch_raises: true` and the transcription's set (all models), which is
  what tsecon returns.

* "mc" (MONTE CARLO tolerance): arch's three p-values at 4000 replications
  with a different seed (un-studentized), and the transcription's
  studentized p-values on those 4000 resamples. tsecon's own seeded
  bootstrap at 4000 replications must land within 0.05 of each (two
  independent bootstraps of 4000 draws differ by at most ~0.011 in
  standard deviation at p = 0.5, so 0.05 is ~4.5 sd).

Resampling conventions — MEASURED, not assumed (`_meta.conventions`,
`check_scheme_conventions`). For each of the three schemes and three
(T, block_size) settings, arch's own raw draws are replayed through the rule
`tsecon_bootstrap::indices` documents and must reproduce arch's index array
element for element: the stationary chain (uniform start, restart at a fresh
uniform index on the coin, otherwise +1 wrapping at T), the circular block
layout (ceil(T/b) uniform starts in 0..T, b consecutive indices modulo T) and
the moving block (starts in 0..T-b+1, no wrap), all truncated to T. THE ONE
DIFFERENCE, read off both sources and recorded in the block: arch continues
a stationary block when `u > p` (restarts on `u <= p`), tsecon restarts on
`u < p` — they disagree only on the null event `u == p` (2^-53 per step; the
exact ties are counted, and were 0 in every draw taken). So the two libraries
differ only in the random-number generator. The realized restart frequency,
mean run length and wrap counts are stored alongside.

Size under the null (`_meta.size_study`, `arch_size_study`): arch's OWN
rejection frequencies of the consistent p-value on the crate property test's
least-favourable design (six exchangeable squared-error loss columns,
T = 200, m = 5, B = 300, 400 Monte Carlo replications), at four block
lengths. The crate test `spa_size_unstudentized_is_near_nominal` compares
tsecon's own seeded draws against these, so a size distortion the two share
is reported as the METHOD's and one they do not share is a bug. These are
un-studentized rates (arch computes nothing else); the studentized path has
no third-party rate and is measured, not validated, in the crate test.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_spa_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import arch
from arch.bootstrap import SPA, RealityCheck, StepM
from arch.bootstrap.base import (
    CircularBlockBootstrap,
    MovingBlockBootstrap,
    StationaryBootstrap,
)

HERE = Path(__file__).resolve().parent
CRIT_ALPHAS = [0.10, 0.05, 0.01]

# --------------------------------------------------------------------------
# NumPy-order transcription helpers (shared with generate_mcs_fixtures.py)
# --------------------------------------------------------------------------


def pairwise_sum(a):
    """NumPy's pairwise summation of a contiguous 1-D float64 array
    (numpy/core/src/umath/loops_utils.h.src), on Python floats."""
    n = len(a)
    if n < 8:
        res = 0.0
        for v in a:
            res += v
        return res
    if n <= 128:
        r = list(a[:8])
        i = 8
        lim = n - (n % 8)
        while i < lim:
            for j in range(8):
                r[j] += a[i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        while i < n:
            res += a[i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return pairwise_sum(a[:n2]) + pairwise_sum(a[n2:])


def percentile_linear(sorted_vals, q):
    """numpy.percentile(x, 100 q) with the default 'linear' method: virtual
    index (n - 1) q and the `_lerp` branch at gamma >= 0.5."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    virt = (n - 1) * q  # the `linear` entry of numpy's _QuantileMethods
    if virt >= n - 1:
        return sorted_vals[-1]
    if virt < 0:
        return sorted_vals[0]
    prev = math.floor(virt)
    lo = int(prev)
    hi = min(lo + 1, n - 1)
    gamma = virt - prev
    a, b = sorted_vals[lo], sorted_vals[hi]
    diff = b - a
    if gamma >= 0.5:
        return b - diff * (1.0 - gamma)
    return a + diff * gamma


def col_mean_seq(col):
    acc = 0.0
    for v in col:
        acc += v
    return acc / len(col)


def col_means_seq(cols, idx, shift=None):
    """Column means of the resampled panel, accumulated sequentially in t
    (NumPy's axis-0 reduction order for a C-ordered T x m panel)."""
    n = len(idx)
    out = []
    for k, col in enumerate(cols):
        s = shift[k] if shift is not None else 0.0
        acc = 0.0
        for t in idx:
            acc += col[t] - s
        out.append(acc / n)
    return out


def hansen_kernel_variance(e_cols, n, block_size):
    """Hansen (2005, eq. 9) as arch's `_compute_variance` evaluates it, with
    the sums made explicit."""
    t = n
    p = 1.0 / block_size
    var = []
    for col in e_cols:
        s = 0.0
        for x in col:
            s += x * x
        var.append(s / t)
    for i in range(1, t):
        kappa = ((1.0 - (i / t)) * ((1 - p) ** i)) + ((i / t) * ((1 - p) ** (t - i)))
        for k, col in enumerate(e_cols):
            s = 0.0
            for j in range(t - i):
                s += col[j] * col[j + i]
            var[k] += 2 * kappa * s / t
    return var


def arch_bootstrap(kind, block_size, data, seed):
    if kind == "stationary":
        return StationaryBootstrap(block_size, data, seed=seed)
    if kind == "circular":
        return CircularBlockBootstrap(block_size, data, seed=seed)
    if kind == "moving_block":
        return MovingBlockBootstrap(block_size, data, seed=seed)
    raise ValueError(kind)


ARCH_NAME = {"stationary": "stationary", "circular": "circular", "moving_block": "moving block"}


def arch_indices(kind, block_size, data, seed, reps):
    """The index arrays arch's SPA/MCS loop draws: a fresh bootstrap with the
    same seed, `update_indices()` once per replication (that is all
    `bootstrap(reps)` consumes)."""
    bs = arch_bootstrap(kind, block_size, data, seed)
    return [np.asarray(bs.update_indices(), dtype=int).copy() for _ in range(reps)]


# --------------------------------------------------------------------------
# Do arch's resampling conventions agree with tsecon-bootstrap's?
# --------------------------------------------------------------------------


def tsecon_stationary(first, restart_targets, u, n, p):
    """`tsecon_bootstrap::indices(BlockScheme::Stationary { p }, n, ..)`, as
    crates/tsecon-bootstrap/src/schemes.rs documents it: start at a uniform
    index; at each later step RESTART at a fresh uniform index when the
    step's uniform is `< p`, otherwise CONTINUE to the next observation,
    wrapping at `n`. Fed another library's raw draws, it must reproduce that
    library's index array — which is what makes this a convention check and
    not a restatement."""
    out = [int(first)]
    for i in range(1, n):
        out.append(int(restart_targets[i]) if u[i] < p else (out[-1] + 1) % n)
    return out


def tsecon_blocks(kind, starts, n, b):
    """`BlockScheme::CircularBlock` / `MovingBlock`: blocks of `b` consecutive
    rows from the given starts — taken modulo `n` (circular) or plain
    (moving block) — concatenated, the last block truncated so the resample
    has length `n`."""
    out = []
    for start in starts:
        take = min(b, n - len(out))
        if take <= 0:
            break
        if kind == "circular":
            out.extend((int(start) + t) % n for t in range(take))
        else:
            out.extend(int(start) + t for t in range(take))
    return out


def check_scheme_conventions(draws=200):
    """Measure, rather than assume, that `arch`'s stationary / circular /
    moving-block resamplers lay out the SAME blocks as `tsecon-bootstrap`'s.

    The check replays arch's OWN raw random draws through tsecon's documented
    rule and asserts the resulting index array is arch's, element for element:

    * stationary — arch draws `n` candidate restart indices and `n` uniforms
      from `default_rng(seed)` and then walks the chain; feeding those same
      two arrays to `tsecon_stationary` must return arch's array. This pins
      the restart coin, the fresh-uniform restart target, the +1 continuation
      and the wrap at `n` all at once;
    * circular / moving block — arch draws `ceil(n/b)` block starts; feeding
      those to `tsecon_blocks` must return arch's array. This pins the block
      length, the consecutive layout, the modulo-`n` wrap (circular) or its
      absence (moving block), the start range, and the truncation to `n`.

    THE ONE DIFFERENCE, read off the two sources: arch CONTINUES the block
    when `u > p` (`arch/bootstrap/_samplers_python.py`), i.e. restarts on
    `u <= p`; tsecon restarts on `u < p` (`schemes.rs`). The two disagree only
    on the null event `u == p` exactly (probability 2^-53 per step, counted
    below and 0 in every draw taken here), so the schemes are identical as
    distributions. The measured restart frequency and the realized block
    statistics are reported alongside.
    """
    out = {"difference": (
        "arch restarts a stationary block when u <= p, tsecon-bootstrap when "
        "u < p; the two differ only on the null event u == p (exact ties "
        "counted below). Every other convention — restart target, +1 "
        "continuation, wrap at n, block length, start range, truncation to n "
        "— is identical, verified by replaying arch's own draws through "
        "tsecon's rule."
    )}
    data_rng = np.random.default_rng(4242)
    for kind in ("stationary", "circular", "moving_block"):
        rows = []
        for n, b in ((80, 6), (100, 5), (150, 12)):
            data = data_rng.standard_normal((n, 2))
            arrs = arch_indices(kind, b, data, 99, draws)
            # arch's own generator, replayed in its documented draw order.
            gen = np.random.default_rng(99)
            ties = restarts = steps = wraps = 0
            run_lengths = []
            for arr in arrs:
                assert arr.shape == (n,), f"{kind}: arch returned {arr.shape}, expected ({n},)"
                assert 0 <= int(arr.min()) and int(arr.max()) < n, f"{kind}: index outside 0..{n}"
                if kind == "stationary":
                    p = 1.0 / b
                    cand = gen.integers(n, size=n, dtype=np.int64)
                    u = gen.random(n)
                    mine = tsecon_stationary(cand[0], cand, u, n, p)
                    assert mine == arr.tolist(), \
                        f"stationary (n={n}, b={b}): tsecon's rule on arch's draws is not arch's array"
                    ties += int(np.sum(u[1:] == p))
                    run = 1
                    for i in range(1, n):
                        steps += 1
                        if u[i] > p:  # arch's continuation test; restart otherwise
                            run += 1
                            if int(arr[i - 1]) == n - 1:
                                wraps += 1
                        else:
                            restarts += 1
                            run_lengths.append(run)
                            run = 1
                    run_lengths.append(run)
                else:
                    n_blocks = -(-n // b)
                    hi = n if kind == "circular" else n - b + 1
                    starts = gen.integers(hi, size=n_blocks, dtype=np.int64)
                    mine = tsecon_blocks(kind, starts, n, b)
                    assert mine == arr.tolist(), \
                        f"{kind} (n={n}, b={b}): tsecon's block layout on arch's starts is not arch's array"
                    if kind == "circular":
                        wraps += sum(1 for j in range(n_blocks)
                                     for t in range(max(0, min(b, n - j * b)))
                                     if int(starts[j]) + t >= n)
            row = {"n": n, "block_size": b, "draws": draws, "replayed_exactly": True}
            if kind == "stationary":
                row["restart_freq"] = restarts / steps
                row["restart_freq_nominal"] = 1.0 / b
                row["mean_run_length"] = float(np.mean(run_lengths))
                row["wrap_transitions"] = wraps
                row["u_equals_p_ties"] = ties
                assert abs(row["restart_freq"] - 1.0 / b) < 0.02, \
                    f"stationary: measured restart frequency {row['restart_freq']} far from 1/b"
                assert wraps > 0, "stationary: the wrap at n never happened — check the convention"
            else:
                row["wrapped_indices"] = wraps
                if kind == "circular":
                    assert wraps > 0, "circular: no block ever wrapped — check the convention"
            rows.append(row)
        out[kind] = rows
    return out


# --------------------------------------------------------------------------
# The SPA transcription (mirrors crates/tsecon-forecast/src/spa.rs)
# --------------------------------------------------------------------------


def spa_transcribe(bench, model_cols, resamples, block_size, studentize, nested):
    n = len(bench)
    m = len(model_cols)
    d = [[bench[t] - col[t] for t in range(n)] for col in model_cols]
    dbar = [col_mean_seq(col) for col in d]
    e = [[v - mu for v in col] for col, mu in zip(d, dbar)]
    ms = [col_means_seq(d, idx) for idx in resamples]
    reps = len(resamples)
    if nested:
        mdem = [col_means_seq(d, idx, shift=dbar) for idx in resamples]
        var = []
        for k in range(m):
            s = 0.0
            for b in range(reps):
                s += mdem[b][k]
            mean = s / reps
            v = 0.0
            for b in range(reps):
                x = mdem[b][k] - mean
                v += x * x
            var.append(n * (v / reps))
    else:
        var = hansen_kernel_variance(e, n, block_size)
    omega = [math.sqrt(v) for v in var]
    loglog = math.log(math.log(n))
    threshold = [-math.sqrt(((var[k] / n) * 2) * loglog) for k in range(m)]
    recentered = [dbar[k] >= threshold[k] for k in range(m)]
    g = [
        [0.0 if dbar[k] < 0 else dbar[k] for k in range(m)],
        [dbar[k] if recentered[k] else 0.0 for k in range(m)],
        list(dbar),
    ]
    boot = [[], [], []]
    zc = []
    for b in range(reps):
        row_c = []
        for j in range(3):
            mx = -math.inf
            for k in range(m):
                z = ms[b][k] - g[j][k]
                if studentize:
                    z /= omega[k]
                if z > mx:
                    mx = z
                if j == 1:
                    row_c.append(z)
            boot[j].append(mx)
        zc.append(row_c)
    obs = [dbar[k] / omega[k] if studentize else dbar[k] for k in range(m)]
    best = 0
    mobs = obs[0]
    for k in range(1, m):
        if obs[k] > mobs:
            mobs = obs[k]
            best = k
    pvals = [sum(1 for x in boot[j] if x > mobs) / reps for j in range(3)]
    sq = math.sqrt(n)
    scaled = [[sq * x for x in boot[j]] for j in range(3)]
    crit = [[percentile_linear(sorted(scaled[j]), 1 - a) for a in CRIT_ALPHAS] for j in range(3)]
    gaps = [min(abs(x - mobs) for x in boot[j]) / (1.0 + abs(mobs)) for j in range(3)]
    # Margin of the consistent-recentring decision: a flip needs dbar to sit
    # within this relative distance of the threshold.
    margin = min(abs(dbar[k] - threshold[k]) / (1.0 + abs(threshold[k])) for k in range(m))
    return {
        "mean_loss_diff": dbar,
        "loss_diff_var": var,
        "recentered": recentered,
        "statistic": sq * mobs,
        "best_model": best,
        "statistic_mean_scale": mobs,
        "p_lower": pvals[0],
        "p_consistent": pvals[1],
        "p_upper": pvals[2],
        "crit_lower": crit[0],
        "crit_consistent": crit[1],
        "crit_upper": crit[2],
        "boot_lower": scaled[0],
        "boot_consistent": scaled[1],
        "boot_upper": scaled[2],
        "boot_mean_scale": boot,
        "min_gap": min(gaps),
        "recentre_margin": margin,
        "_zc": zc,
        "_obs": obs,
    }


def stepm_transcribe(zc, obs, n, size):
    m = len(obs)
    reps = len(zc)
    remaining = [True] * m
    superior = []
    steps = []
    crits = []
    sq = math.sqrt(n)
    while True:
        mx = []
        for b in range(reps):
            best = -math.inf
            for k in range(m):
                if remaining[k] and zc[b][k] > best:
                    best = zc[b][k]
            mx.append(best)
        crit = percentile_linear(sorted(mx), 1 - size)
        better = [k for k in range(m) if remaining[k] and obs[k] > crit]
        crits.append(sq * crit)
        steps.append(better)
        for k in better:
            remaining[k] = False
        superior.extend(better)
        if not better or len(superior) == m:
            break
    return sorted(superior), steps, crits


# --------------------------------------------------------------------------
# Data-generating process
# --------------------------------------------------------------------------


def ar1(rng, n, rho):
    e = rng.standard_normal(n)
    x = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = rho * prev + math.sqrt(1 - rho * rho) * e[t]
        x[t] = prev
    return x


def loss_panel(rng, n, scales, rho=0.5, common=0.7, bias=None):
    """Squared-error losses of forecasts whose errors share an AR(1) common
    component (cross-model dependence) plus an idiosyncratic AR(1) part
    scaled per model; `bias` shifts a model's errors (a dominated model)."""
    u = ar1(rng, n, rho)
    cols = []
    for k, s in enumerate(scales):
        v = ar1(rng, n, rho)
        err = common * u + s * v
        if bias is not None:
            err = err + bias[k]
        cols.append((err**2).tolist())
    return cols


def assert_close(a, b, rtol, what):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = np.maximum(np.abs(b), 1.0)
    if a.shape != b.shape or np.any(np.abs(a - b) > rtol * scale):
        raise AssertionError(f"{what}: transcription differs from arch beyond {rtol}: max |diff| = {np.max(np.abs(a - b))}")


def assert_bits(a, b, what):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(a.view(np.int64), b.view(np.int64)):
        raise AssertionError(f"{what}: transcription is not bit-identical to arch: max |diff| = {np.max(np.abs(a - b))}")


# --------------------------------------------------------------------------
# arch's OWN size under the null, so the crate's size study has a reference
# --------------------------------------------------------------------------

SIZE_LEVELS = [0.05, 0.10, 0.25, 0.50]
SIZE_MC = 400
SIZE_N = 200
SIZE_M = 6  # a benchmark plus five models, all exchangeable
SIZE_REPS = 300
# (label, rho of the AR(1) forecast errors, block_size)
SIZE_CONFIGS = [
    ("iid_b2", 0.0, 2),
    ("ar05_b3", 0.5, 3),
    ("ar05_b8", 0.5, 8),
    ("ar05_b14", 0.5, 14),
]


def arch_size_study():
    """How often does `arch.bootstrap.SPA`'s consistent p-value fall below
    alpha when EVERY model is exactly as good as the benchmark?

    The design is the crate property test's (`spa_mcs_properties.rs`): six
    exchangeable squared-error loss columns whose forecast errors share an
    AR(1) common component (`loss_panel` with equal scales and no bias), the
    least favourable configuration `mu = 0`. The rates are a property of the
    METHOD, not of this library, and they are what
    `spa_size_unstudentized_is_near_nominal` compares tsecon's own seeded
    draws against — so that an over-size the test shares with its reference
    is reported as the method's, while one it does not share is a bug.

    arch's `studentize` flag is inert, so these are the UN-studentized
    (White Reality Check) rates; no package computes the studentized ones.
    """
    out = {"n": SIZE_N, "m": SIZE_M - 1, "reps": SIZE_REPS, "mc": SIZE_MC,
           "levels": SIZE_LEVELS, "studentize": False, "rates": {}}
    levels = np.asarray(SIZE_LEVELS)
    for label, rho, block in SIZE_CONFIGS:
        hits = np.zeros(len(SIZE_LEVELS))
        for r in range(SIZE_MC):
            rng = np.random.default_rng(770000 + r)
            cols = loss_panel(rng, SIZE_N, [1.0] * SIZE_M, rho=rho)
            bench = np.asarray(cols[0])
            models = np.column_stack([np.asarray(c) for c in cols[1:]])
            s = SPA(bench, models, block_size=block, reps=SIZE_REPS,
                    bootstrap="stationary", studentize=False, seed=9000 + r)
            s.compute()
            hits += float(s.pvalues["consistent"]) <= levels
        rates = (hits / SIZE_MC).tolist()
        out["rates"][label] = {"rho": rho, "block_size": block, "rates": rates}
        print(f"arch size study {label:10s} rho={rho} b={block:2d}: "
              f"P(p_consistent <= alpha) at {SIZE_LEVELS} -> "
              f"{[round(v, 4) for v in rates]}", flush=True)
    return out


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

CASES = [
    # name, n, bootstrap, block_size, nested, seed, scales (benchmark first), bias
    ("stationary_m4", 100, "stationary", 5, False, 7, [1.0, 0.8, 0.9, 1.1, 1.3], None),
    ("circular_m3", 100, "circular", 4, False, 11, [1.0, 0.85, 1.0, 1.2], None),
    ("moving_block_m5", 80, "moving_block", 6, False, 3, [1.0, 0.9, 0.95, 1.05, 1.1, 1.4], None),
    ("stationary_m4_nested", 100, "stationary", 5, True, 21, [1.0, 0.8, 0.9, 1.1, 1.3], None),
    ("single_model", 90, "stationary", 3, False, 5, [1.0, 0.75], None),
    # A clearly dominated benchmark (its errors carry a 1.5 shift): every
    # model better, p-values at zero, every model superior under StepM at
    # 200 and at 4000 replications alike (probed before it was chosen).
    ("dominated_benchmark", 100, "stationary", 5, False, 13, [1.0, 0.7, 0.8, 0.9], [1.5, 0.0, 0.0, 0.0]),
    # Every model worse than the benchmark (some significantly): the
    # consistent re-centring leaves the bad ones un-centred.
    ("no_model_better", 100, "circular", 5, False, 17, [1.0, 1.4, 1.6, 2.0], [0.0, 0.3, 0.6, 1.0]),
]
REPS = 200
MC_REPS = 4000
STEPM_SIZES = [0.05, 0.10]


def run_case(name, n, kind, block_size, nested, seed, scales, bias):
    rng = np.random.default_rng(1000 + seed)
    cols = loss_panel(rng, n, scales, bias=bias)
    bench = cols[0]
    models = cols[1:]
    m = len(models)
    bench_arr = np.asarray(bench)
    models_arr = np.column_stack([np.asarray(c) for c in models])

    # --- arch, both flag values: the inertness finding, asserted.
    res = {}
    for st in (True, False):
        spa = SPA(bench_arr, models_arr, block_size=block_size, reps=REPS,
                  bootstrap=ARCH_NAME[kind], studentize=st, nested=nested, seed=seed)
        spa.compute()
        res[st] = spa
    assert_bits(res[True].pvalues.values, res[False].pvalues.values, f"{name}: studentize inert (p)")
    for a in CRIT_ALPHAS:
        assert_bits(res[True].critical_values(a).values, res[False].critical_values(a).values,
                    f"{name}: studentize inert (crit {a})")
    spa = res[False]
    rc = RealityCheck(bench_arr, models_arr, block_size=block_size, reps=REPS,
                      bootstrap=ARCH_NAME[kind], nested=nested, seed=seed)
    rc.compute()
    assert_bits(rc.pvalues.values, spa.pvalues.values, f"{name}: RealityCheck is SPA")

    # --- arch's resample indices, reconstructed and verified against the
    # last index array the SPA loop left behind.
    loss_diff = bench_arr[:, None] - models_arr
    resamples = arch_indices(kind, block_size, loss_diff, seed, REPS)
    assert np.array_equal(np.asarray(spa.bootstrap._index, dtype=int), resamples[-1]), \
        f"{name}: reconstructed indices do not match arch's last draw"
    idx_lists = [r.tolist() for r in resamples]

    # --- transcription, un-studentized: bit-identical to arch for m >= 2.
    # NumPy reduces a C-ordered T x m panel over axis 0 sequentially in t
    # when m >= 2, but coalesces a T x 1 panel into a contiguous 1-D reduction
    # and sums it PAIRWISE; the Rust always sums sequentially, so the
    # single-model case is pinned at 1e-13 instead (its p-values are still
    # exact: min_gap certifies no comparison can flip at that distance).
    bit_exact = m >= 2
    pin = assert_bits if bit_exact else (lambda a, b, what: assert_close(a, b, 1e-13, what))
    tr = spa_transcribe(bench, models, idx_lists, block_size, studentize=False, nested=nested)
    pin(tr["mean_loss_diff"], spa._loss_diff.mean(0), f"{name}: dbar")
    pin(tr["loss_diff_var"], spa._loss_diff_var, f"{name}: variances")
    assert list(tr["recentered"]) == [bool(v) for v in spa._valid_columns], f"{name}: recentring flags"
    max_sim = np.max(spa._simulated_vals, 0)  # (reps, 3): lower, consistent, upper
    for j, key in enumerate(("boot_lower", "boot_consistent", "boot_upper")):
        pin(tr["boot_mean_scale"][j], max_sim[:, j], f"{name}: {key}")
    assert_bits([tr["p_lower"], tr["p_consistent"], tr["p_upper"]], spa.pvalues.values, f"{name}: p-values")
    pin(tr["statistic_mean_scale"], np.max(loss_diff.mean(0)), f"{name}: statistic")
    for i, a in enumerate(CRIT_ALPHAS):
        cv = spa.critical_values(a).values
        mine = [percentile_linear(sorted(tr["boot_mean_scale"][j]), 1 - a) for j in range(3)]
        pin(mine, cv, f"{name}: percentile port at alpha {a}")
    if not bit_exact:
        # Record the measured NumPy order for the single column: pairwise.
        for idx in idx_lists[:20]:
            col = loss_diff[idx, 0].tolist()
            assert pairwise_sum(col) / n == float(loss_diff[idx].mean(0)[0]), \
                f"{name}: NumPy's T x 1 reduction is not the pairwise order either"
    assert tr["min_gap"] > 1e-9, f"{name}: a replicate sits within 1e-9 of the statistic ({tr['min_gap']})"
    assert tr["recentre_margin"] > 1e-9, f"{name}: recentring decision within 1e-9 of the threshold"
    arch_block = {
        "bit_exact": bit_exact,
        "mean_loss_diff": tr["mean_loss_diff"],
        "loss_diff_var": tr["loss_diff_var"],
        "recentered": tr["recentered"],
        "statistic_mean_scale": tr["statistic_mean_scale"],
        "statistic": tr["statistic"],
        "best_model": tr["best_model"],
        "p_lower": float(spa.pvalues["lower"]),
        "p_consistent": float(spa.pvalues["consistent"]),
        "p_upper": float(spa.pvalues["upper"]),
        "crit_mean_scale": {str(a): spa.critical_values(a).values.tolist() for a in CRIT_ALPHAS},
        "crit_lower": tr["crit_lower"],
        "crit_consistent": tr["crit_consistent"],
        "crit_upper": tr["crit_upper"],
        "boot_mean_scale_lower": max_sim[:, 0].tolist(),
        "boot_mean_scale_consistent": max_sim[:, 1].tolist(),
        "boot_mean_scale_upper": max_sim[:, 2].tolist(),
        "boot_lower": tr["boot_lower"],
        "boot_consistent": tr["boot_consistent"],
        "boot_upper": tr["boot_upper"],
        "min_gap": tr["min_gap"],
        "recentre_margin": tr["recentre_margin"],
    }

    # --- StepM: arch vs the transcription's stepwise rule.
    stepm_block = {}
    for size in STEPM_SIZES:
        sm = StepM(bench_arr, models_arr, size=size, block_size=block_size, reps=REPS,
                   bootstrap=ARCH_NAME[kind], studentize=False, nested=nested, seed=seed)
        sup, steps, crits = stepm_transcribe(tr["_zc"], tr["_obs"], n, size)
        try:
            sm.compute()
            arch_raises = False
        except ValueError as exc:
            # arch's StepM loop re-runs the SPA on an empty selector once every
            # model has been declared superior over two or more steps, and
            # np.max of a zero-size array raises. tsecon stops the loop instead;
            # the transcription must then hold every model, reached in >= 2 steps.
            assert "zero-size array" in str(exc), f"{name}: unexpected arch StepM error: {exc}"
            assert len(sup) == m and len(steps) >= 2, f"{name}: arch raised but the set is not all-models-in->=2-steps"
            arch_raises = True
        if not arch_raises:
            assert sup == [int(v) for v in sm.superior_models], \
                f"{name}: StepM superior set differs at size {size}: {sup} vs {list(sm.superior_models)}"
        stepm_block[str(size)] = {"superior_models": sup, "steps": steps, "step_crit_values": crits,
                                  "arch_raises": arch_raises}

    # --- studentized transcription (Hansen's formulas), same resamples.
    ts = spa_transcribe(bench, models, idx_lists, block_size, studentize=True, nested=nested)
    assert ts["min_gap"] > 1e-9, f"{name}: studentized replicate within 1e-9 of the statistic"
    stud_stepm = {}
    for size in STEPM_SIZES:
        sup, steps, crits = stepm_transcribe(ts["_zc"], ts["_obs"], n, size)
        stud_stepm[str(size)] = {"superior_models": sup, "steps": steps, "step_crit_values": crits}
    stud_block = {k: v for k, v in ts.items() if not k.startswith("_") and k != "boot_mean_scale"}
    stud_block["stepm"] = stud_stepm

    # --- Monte Carlo leg: 4000 arch replications with another seed.
    mc_seed = seed + 500
    mc = SPA(bench_arr, models_arr, block_size=block_size, reps=MC_REPS,
             bootstrap=ARCH_NAME[kind], studentize=False, nested=nested, seed=mc_seed)
    mc.compute()
    mc_res = arch_indices(kind, block_size, loss_diff, mc_seed, MC_REPS)
    assert np.array_equal(np.asarray(mc.bootstrap._index, dtype=int), mc_res[-1])
    mc_stud = spa_transcribe(bench, models, [r.tolist() for r in mc_res], block_size,
                             studentize=True, nested=nested)
    mc_unstud = spa_transcribe(bench, models, [r.tolist() for r in mc_res], block_size,
                               studentize=False, nested=nested)
    assert_bits([mc_unstud["p_lower"], mc_unstud["p_consistent"], mc_unstud["p_upper"]],
                mc.pvalues.values, f"{name}: MC-leg transcription vs arch")
    mc_block = {
        "reps": MC_REPS,
        "arch_seed": mc_seed,
        "unstudentized": {"p_lower": float(mc.pvalues["lower"]),
                          "p_consistent": float(mc.pvalues["consistent"]),
                          "p_upper": float(mc.pvalues["upper"])},
        "studentized": {"p_lower": mc_stud["p_lower"], "p_consistent": mc_stud["p_consistent"],
                        "p_upper": mc_stud["p_upper"]},
    }

    print(f"{name:24s} n={n:3d} m={m} {kind:12s} b={block_size} nested={nested!s:5s} "
          f"arch p=({arch_block['p_lower']:.3f},{arch_block['p_consistent']:.3f},{arch_block['p_upper']:.3f}) "
          f"stud p=({ts['p_lower']:.3f},{ts['p_consistent']:.3f},{ts['p_upper']:.3f}) "
          f"stepm05={stepm_block['0.05']['superior_models']} min_gap={tr['min_gap']:.2e}")
    return {
        "name": name,
        "n": n,
        "m": m,
        "bootstrap": kind,
        "block_size": block_size,
        "nested": nested,
        "arch_seed": seed,
        "reps": REPS,
        "scales": scales,
        "benchmark": bench,
        "models": models,
        "resamples": idx_lists,
        "arch": arch_block,
        "stepm": stepm_block,
        "studentized": stud_block,
        "mc": mc_block,
    }


def main():
    conventions = check_scheme_conventions()
    print("resampling conventions: arch's index arrays replayed exactly by "
          "tsecon-bootstrap's rule for all three schemes")
    cases = [run_case(*c) for c in CASES]
    size_study = arch_size_study()
    out = {
        "_meta": {
            "generator": "fixtures/generate_spa_fixtures.py",
            "reference": "arch.bootstrap.SPA / RealityCheck / StepM (independent package) + NumPy transcription of Hansen (2005) for the studentized path",
            "arch": arch.__version__,
            "numpy": np.__version__,
            "studentize_inert": True,
            "notes": [
                "arch 8.0.0: SPA.studentize is inert (asserted: identical p-values and critical values on/off); RealityCheck is `class RealityCheck(SPA): pass`.",
                "arch statistics are on the mean scale; tsecon reports sqrt(n) x (statistic, replicates, critical values). The p-values are invariant.",
                "resamples[b] are the indices arch's default_rng(arch_seed) bootstrap drew for replication b. The `conventions` block MEASURES that arch's stationary / circular / moving-block layouts are tsecon-bootstrap's: arch's own raw draws replayed through tsecon's documented rule reproduce arch's index arrays element for element at three (n, block_size) settings. The single difference is the tie convention of the stationary restart coin (arch restarts on u <= p, tsecon on u < p), which differs only on the null event u == p.",
                "The `size_study` block is arch's OWN rejection frequency under the least favourable null on the crate property test's design, so that an over-size tsecon shares with its reference is reported as the method's and one it does not share is a bug.",
                "nested=True: arch clones the bootstrap with a deepcopy of the integer seed, so the nested variance uses the same resamples as the test; the fixture's `loss_diff_var` is asserted bit-identical to that construction.",
                "Every stored 'arch' number was recomputed from the stored indices with explicit sequential sums and asserted bit-identical; min_gap > 1e-9 certifies the p-values cannot flip under sub-1e-9 arithmetic differences.",
            ],
            "crit_levels": [1 - a for a in CRIT_ALPHAS],
            "mc_tolerance": 0.05,
            "conventions": conventions,
            "size_study": size_study,
        },
        "cases": cases,
    }
    path = HERE / "spa.json"
    path.write_text(json.dumps(out, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
