"""NON-GATING cross-run: tsecon.auto_arima vs pmdarima.auto_arima.

Informative only — nothing in the test suite gates on this, and the
model card reports it as context, not validation. The two implement the
same published algorithm with different engines and different seasonal-
differencing tests (pmdarima: OCSB; tsecon, like R's default: seasonal
strength), so disagreement on near-ties is expected and documented, not
a defect. R's forecast::auto.arima and pmdarima themselves disagree on
real series; that is exactly why the gating grade is MC recovery.

    .venv-wt/bin/python scripts/cross_check_auto_arima_pmdarima.py

Datasets: seeded simulated DGPs plus two bundled public-domain
statsmodels series (monthly-resampled Mauna Loa CO2, annual sunspots) —
no network data.
"""
import time
import warnings

import numpy as np

import tsecon

warnings.filterwarnings("ignore")

try:
    import pmdarima as pm

    PMD = pm.__version__
except Exception as e:  # pragma: no cover - environment-dependent
    PMD = None
    PMD_ERR = repr(e)


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


def run_pair(y, seasonal_period=0):
    ours = tsecon.auto_arima(y, seasonal_period=seasonal_period)
    kw = dict(
        information_criterion="aicc",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    if seasonal_period >= 2:
        theirs = pm.auto_arima(y, seasonal=True, m=seasonal_period, **kw)
    else:
        theirs = pm.auto_arima(y, seasonal=False, **kw)
    t_order = tuple(theirs.order)
    t_seas = tuple(theirs.seasonal_order) if seasonal_period >= 2 else (0, 0, 0, 0)
    return ours, t_order, t_seas


def main():
    print("tsecon.auto_arima vs pmdarima.auto_arima — NON-GATING cross-run")
    if PMD is None:
        print(f"pmdarima unavailable in this venv: {PMD_ERR}")
        print("(reported honestly; nothing gates on this)")
        return
    print(f"tsecon {tsecon.__version__}, pmdarima {PMD}, numpy {np.__version__}")

    dgps = [
        ("AR(1) 0.6 n=300", 0, lambda r: simulate_arma(r, 300, ar=[0.6]), 25),
        ("MA(1) 0.6 n=300", 0, lambda r: simulate_arma(r, 300, ma=[0.6]), 25),
        (
            "ARMA(1,1) n=300",
            0,
            lambda r: simulate_arma(r, 300, ar=[0.5], ma=[0.4]),
            25,
        ),
        (
            "ARIMA(1,1,1) n=300",
            0,
            lambda r: np.cumsum(simulate_arma(r, 300, ar=[0.5], ma=[0.4])),
            25,
        ),
        (
            "SARIMA(1,0,0)(1,0,0)[4] n=300",
            4,
            lambda r: simulate_arma(r, 300, ar=[0.5, 0.0, 0.0, 0.6, -0.30]),
            10,
        ),
    ]

    for name, s, sim, reps in dgps:
        same_full = same_d = same_pq1 = 0
        t0 = time.time()
        for rep in range(reps):
            rng = np.random.default_rng(555 + 271 * rep)
            y = sim(rng)
            ours, t_order, t_seas = run_pair(y, s)
            o_order = tuple(ours["order"])
            o_seas = tuple(ours["seasonal_order"])
            if o_order == t_order and (s < 2 or o_seas[:3] == t_seas[:3]):
                same_full += 1
            if o_order[1] == t_order[1]:
                same_d += 1
            if (
                o_order[1] == t_order[1]
                and abs(o_order[0] - t_order[0]) <= 1
                and abs(o_order[2] - t_order[2]) <= 1
            ):
                same_pq1 += 1
        print(
            f"{name:32s} reps={reps:2d}  identical orders {same_full / reps:.2f}  "
            f"same d {same_d / reps:.2f}  same d & (p,q) within one "
            f"{same_pq1 / reps:.2f}  ({time.time() - t0:6.1f}s)",
            flush=True,
        )

    # --- Bundled statsmodels datasets (public domain, no network). ---
    import statsmodels.api as sm

    co2 = (
        sm.datasets.co2.load_pandas()
        .data["co2"]
        .resample("MS")
        .mean()
        .ffill()
        .to_numpy()[-240:]
    )
    ours, t_order, t_seas = run_pair(co2, seasonal_period=12)
    print(
        f"co2 monthly (last 240):  tsecon {tuple(ours['order'])}"
        f"x{tuple(ours['seasonal_order'])}  pmdarima {t_order}x{t_seas}",
        flush=True,
    )

    sunspots = sm.datasets.sunspots.load_pandas().data["SUNACTIVITY"].to_numpy()
    ours, t_order, _ = run_pair(sunspots)
    print(
        f"sunspots annual (n={len(sunspots)}): tsecon {tuple(ours['order'])}  "
        f"pmdarima {t_order}"
    )


if __name__ == "__main__":
    main()
