"""Every achieved value quoted in the docstrings of
bindings/python/tests/test_replication_blanchard_quah.py for the native
bootstrap bands, and the cross-seed drift its cap is set against.

    python lab/bq_bands/guard_numbers.py > lab/bq_bands/out/guard_numbers.txt

Runs the replication script's own `bootstrap_bands` (tsecon.long_run_svar_bands,
200 replications as in the guard) at seeds 0, 1, 7 and 42.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "examples"))
import replication_blanchard_quah as bq  # noqa: E402

import tsecon  # noqa: E402

X = bq.transform(bq.load_macrodata())["data"]
N_BOOT = 200
boots = {s: bq.bootstrap_bands(X, n_boot=N_BOOT, seed=s) for s in (0, 1, 7, 42)}
print(f"tsecon {tsecon.__version__}; {N_BOOT} replications; quantiles [h][variable][shock][5/16/84/95]")
for s in (0, 1):
    q = boots[s]["quantiles"]
    print(f"seed {s}:")
    print(f"  demand -> unemployment 90% upper at h = 0, 4: {q[0, 1, 1, 3]:+.3f}, {q[4, 1, 1, 3]:+.3f}")
    print(f"  demand -> output 90% lower at h = 0, 4: {q[0, 0, 1, 0]:+.3f}, {q[4, 0, 1, 0]:+.3f}")
    print(f"  demand -> output 90% band at h = 40: [{q[40, 0, 1, 0]:+.3f}, {q[40, 0, 1, 3]:+.3f}]")
    print(f"  supply -> output 90% lower at h = 40: {q[40, 0, 0, 0]:+.3f}")
    print(f"  supply -> unemployment 68% lower at h = 0: {q[0, 1, 0, 1]:+.3f}")
    print(f"  min over draws of output's impact response to demand: {boots[s]['draws'][:, 0, 0, 1].min():+.3f}")
    nat = boots[s]["native"]
    print(f"  sign_anchor_mismatches {nat['sign_anchor_mismatches']}, n_unstable_draws {nat['n_unstable_draws']}")
for s in (1, 7, 42):
    d = np.abs(boots[0]["quantiles"] - boots[s]["quantiles"]).max()
    print(f"max |quantile(seed 0) - quantile(seed {s})| = {d:.3f}")
hand = bq.bootstrap_draws_by_hand(X, n_boot=40, seed=0, horizon=12)
nat = tsecon.long_run_svar_bands(X, lags=8, horizon=12, impact_signs=bq.ANCHOR, n_boot=40, seed=0,
                                 return_draws=True)["irf_draws"]
print(f"native irf_draws vs the hand-built construction (40 draws, seed 0, h <= 12): "
      f"max |diff| = {np.abs(hand - nat).max():.1e}")
fit = bq.fit_bq(X)
hd = bq.historical_decomposition_bq(X, fit=fit)
print(f"native impact HD vs hand-built: max |diff| = {hd['hand_check']:.1e}; baseline vs Cholesky "
      f"{hd['baseline_check']:.1e}; total contribution vs Cholesky {hd['total_check']:.1e}")
