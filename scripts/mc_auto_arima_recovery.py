"""Monte-Carlo order-recovery study for auto_arima — the primary grade.

The selection loop has no gating third-party reference (R's auto.arima
and pmdarima disagree with each other on real series, so "parity" would
pin an implementation accident); its published grade is how often it
recovers KNOWN orders from simulated DGPs. This script measures that and
its output is quoted verbatim in docs/reference/model-cards/arima.md.

    .venv-wt/bin/python scripts/mc_auto_arima_recovery.py

Definitions
-----------
exact       selected (p, d, q) — and (P, D, Q) for seasonal DGPs —
            equals the truth exactly (the constant is not scored: the
            DGPs are zero-mean and a fitted mean near zero is not an
            order error).
within-one  d (and D) selected exactly, and each of p, q (and P, Q)
            within +-1 of the truth — the "adjacent model" band in
            which forecasts are typically indistinguishable.

Every rep is a fresh seeded default_rng draw; the 95% CI is the
normal-approximation binomial interval.
"""
import sys
import time

import numpy as np

import tsecon


def simulate_arma(rng, n, ar=(), ma=(), sigma=1.0, constant=0.0, burn=400):
    total = n + burn
    e = sigma * rng.standard_normal(total)
    y = np.zeros(total)
    for t in range(total):
        v = constant + e[t]
        for i, phi in enumerate(ar):
            if t > i:
                v += phi * y[t - 1 - i]
        for j, th in enumerate(ma):
            if t > j:
                v += th * e[t - 1 - j]
        y[t] = v
    return y[burn:]


DGPS = [
    # name, truth (p,d,q,P,D,Q), seasonal_period, n, reps, simulator
    (
        "AR(1) phi=0.6, n=300",
        (1, 0, 0, 0, 0, 0),
        0,
        300,
        200,
        lambda rng, n: simulate_arma(rng, n, ar=[0.6]),
    ),
    (
        "AR(2) phi=(0.5,0.25), n=300",
        (2, 0, 0, 0, 0, 0),
        0,
        300,
        200,
        lambda rng, n: simulate_arma(rng, n, ar=[0.5, 0.25]),
    ),
    (
        "MA(1) theta=0.6, n=300",
        (0, 0, 1, 0, 0, 0),
        0,
        300,
        200,
        lambda rng, n: simulate_arma(rng, n, ma=[0.6]),
    ),
    (
        "ARMA(1,1) (0.5, 0.4), n=300",
        (1, 0, 1, 0, 0, 0),
        0,
        300,
        200,
        lambda rng, n: simulate_arma(rng, n, ar=[0.5], ma=[0.4]),
    ),
    (
        "ARIMA(1,1,1) (0.5, 0.4), n=300",
        (1, 1, 1, 0, 0, 0),
        0,
        300,
        200,
        lambda rng, n: np.cumsum(simulate_arma(rng, n, ar=[0.5], ma=[0.4])),
    ),
    (
        "SARIMA (1,0,0)(1,0,0)[4] (0.5, 0.6), n=300",
        (1, 0, 0, 1, 0, 0),
        4,
        300,
        100,
        # (1 - 0.5L)(1 - 0.6L^4) multiplied out.
        lambda rng, n: simulate_arma(rng, n, ar=[0.5, 0.0, 0.0, 0.6, -0.30]),
    ),
    (
        "airline (0,1,1)(0,1,1)[12] (-0.4, -0.6), n=144",
        (0, 1, 1, 0, 1, 1),
        12,
        144,
        60,
        # Integrate an MA with theta(L) = (1 - 0.4L)(1 - 0.6L^12) at
        # both lags 1 and 12: cumsum then seasonal cumsum.
        lambda rng, n: _airline(rng, n),
    ),
]


def _airline(rng, n):
    ma_l = {1: -0.4, 12: -0.6, 13: 0.24}  # (1 - .4L)(1 - .6L^12)
    w = simulate_arma(
        rng, n + 13, ma=[ma_l.get(k, 0.0) for k in range(1, 14)]
    )
    # w = diff(seasonal_diff(y)): undo the seasonal difference (the two
    # difference operators commute), then the regular one.
    x = w.copy()
    for t in range(12, len(x)):
        x[t] += x[t - 12]
    return np.cumsum(x)[-n:]


def ci95(k, n):
    p = k / n
    half = 1.96 * np.sqrt(p * (1 - p) / n)
    return f"{p:.2f} [{max(0.0, p - half):.2f}, {min(1.0, p + half):.2f}]"


def main():
    print("auto_arima MC order recovery (defaults: stepwise AICc)")
    print(f"tsecon {tsecon.__version__}, numpy {np.__version__}")
    rows = []
    for name, truth, s, n, reps, sim in DGPS:
        tp, td, tq, tP, tD, tQ = truth
        exact = within = 0
        t0 = time.time()
        for rep in range(reps):
            rng = np.random.default_rng(97 + 1013 * rep)
            y = sim(rng, n)
            r = tsecon.auto_arima(y, seasonal_period=s)
            p, d, q = r["order"]
            P, D, Q, _ = r["seasonal_order"]
            if (p, d, q, P, D, Q) == truth:
                exact += 1
            if (
                d == td
                and D == tD
                and abs(p - tp) <= 1
                and abs(q - tq) <= 1
                and abs(P - tP) <= 1
                and abs(Q - tQ) <= 1
            ):
                within += 1
        dt = time.time() - t0
        row = (
            f"{name:45s} reps={reps:3d}  exact {ci95(exact, reps):18s} "
            f"within-one {ci95(within, reps):18s} ({dt:6.1f}s)"
        )
        rows.append(row)
        print(row, flush=True)
    print("\nsummary (paste into the model card):")
    for row in rows:
        print("  " + row)


if __name__ == "__main__":
    sys.exit(main())
