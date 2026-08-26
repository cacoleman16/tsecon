"""Round-8 probe: theta_forecast vs statsmodels ThetaModel.

The __doc__ promise is bare: "Matches statsmodels ThetaModel." The card adds
"(deseasonalize=True)". The Rust module doc alone adds use_test=False.
statsmodels' own default is deseasonalize=True, use_test=True (a seasonality
pre-test). Measure where the promise holds and where it silently doesn't.
"""
import numpy as np
import pandas as pd
import tsecon
from statsmodels.tsa.forecasting.theta import ThetaModel

attempted = 0
made = 0
fails = []


def check(name, cond, detail=""):
    global attempted, made
    attempted += 1
    made += 1
    if not cond:
        fails.append((name, detail))
    print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")


def sm_theta(y, period, steps, use_test):
    s = pd.Series(y)
    tm = ThetaModel(s, period=period, deseasonalize=True, use_test=use_test)
    return tm.fit().forecast(steps).to_numpy()


rng = np.random.default_rng(80809)

# 1. strongly seasonal, multiplicative-looking (positive) series
t = np.arange(144)
y_seas = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 144)
fc = tsecon.theta_forecast(y_seas, steps=12, period=12)
fc_sm_test = sm_theta(y_seas, 12, 12, use_test=True)
fc_sm_notest = sm_theta(y_seas, 12, 12, use_test=False)
d_test = np.max(np.abs(fc - fc_sm_test) / np.abs(fc_sm_test))
d_notest = np.max(np.abs(fc - fc_sm_notest) / np.abs(fc_sm_notest))
print(f"strong seasonality: rel diff vs use_test=True {d_test:.2e}, vs use_test=False {d_notest:.2e}")
check("strong seasonality matches statsmodels (either flavor) at 1e-4",
      min(d_test, d_notest) < 1e-4, f"{min(d_test, d_notest):.2e}")

# 2. NON-seasonal data declared with period=12: statsmodels' default pre-test
#    will decline to deseasonalize; what does tsecon do?
y_ns = 50 + np.cumsum(rng.normal(0, 1, 144))
fc2 = tsecon.theta_forecast(y_ns, steps=8, period=12)
fc2_sm_test = sm_theta(y_ns, 12, 8, use_test=True)
fc2_sm_notest = sm_theta(y_ns, 12, 8, use_test=False)
d2_test = np.max(np.abs(fc2 - fc2_sm_test) / np.abs(fc2_sm_test))
d2_notest = np.max(np.abs(fc2 - fc2_sm_notest) / np.abs(fc2_sm_notest))
print(f"non-seasonal y, period=12: rel diff vs default statsmodels (use_test=True) {d2_test:.2e}, "
      f"vs use_test=False {d2_notest:.2e}")
print("  -> tsecon matches the use_test=False flavor only:", d2_notest < 1e-6 and d2_test > 1e-4)

# 3. period=1
y3 = 20 + 0.1 * np.arange(100) + rng.normal(0, 1, 100)
fc3 = tsecon.theta_forecast(y3, steps=6, period=1)
s3 = pd.Series(y3)
tm3 = ThetaModel(s3, period=1, deseasonalize=False)
fc3_sm = tm3.fit().forecast(6).to_numpy()
d3 = np.max(np.abs(fc3 - fc3_sm) / np.abs(fc3_sm))
check("period=1 matches statsmodels deseasonalize=False", d3 < 1e-6, f"{d3:.2e}")

# 4. quarterly realgdp-style: the golden's own configuration
import statsmodels.api as smapi
gdp = smapi.datasets.macrodata.load_pandas().data["realgdp"].to_numpy()
fc4 = tsecon.theta_forecast(gdp, steps=8, period=4)
fc4_test = sm_theta(gdp, 4, 8, use_test=True)
fc4_notest = sm_theta(gdp, 4, 8, use_test=False)
d4t = np.max(np.abs(fc4 - fc4_test) / np.abs(fc4_test))
d4n = np.max(np.abs(fc4 - fc4_notest) / np.abs(fc4_notest))
print(f"realgdp period=4: vs use_test=True {d4t:.2e}, vs use_test=False {d4n:.2e}")
check("realgdp matches the use_test=False golden at 1e-6", d4n < 1e-6, f"{d4n:.2e}")
print(f"  -> statsmodels default (use_test=True) on realgdp deseasonalizes? diff {d4t:.2e} "
      f"(0 means the pre-test also chose to deseasonalize)")

# 5. Monte Carlo: how often does the default-vs-tsecon divergence bite on
#    weakly seasonal positive data?
div = 0
reps = 20
for k in range(reps):
    r2 = np.random.default_rng(9000 + k)
    amp = 1.0  # weak seasonality vs noise sd 2
    ys = 100 + 0.1 * t + amp * np.sin(2 * np.pi * t / 12) + r2.normal(0, 2, 144)
    a = tsecon.theta_forecast(ys, steps=4, period=12)
    b = sm_theta(ys, 12, 4, use_test=True)
    if np.max(np.abs(a - b) / np.abs(b)) > 1e-6:
        div += 1
attempted += 1
made += 1
print(f"weak seasonality: tsecon differs from DEFAULT statsmodels ThetaModel in {div}/{reps} draws")

# 6. degenerate inputs
def expect_raise(name, fn):
    global attempted, made
    attempted += 1
    try:
        fn()
        made += 1
        fails.append((name, "no raise"))
        print(f"[FAIL] {name}: no raise")
    except Exception as e:
        made += 1
        print(f"[ok] {name}: {type(e).__name__}: {str(e)[:90]}")

expect_raise("steps=0", lambda: tsecon.theta_forecast(y3, steps=0))
expect_raise("period=0", lambda: tsecon.theta_forecast(y3, steps=4, period=0))
expect_raise("too short (n<4)", lambda: tsecon.theta_forecast(np.array([1.0, 2.0, 3.0]), steps=2))
expect_raise("n < 2*period", lambda: tsecon.theta_forecast(y3[:20], steps=2, period=12))
expect_raise("NaN", lambda: tsecon.theta_forecast(np.array([1.0, np.nan, 2.0, 3.0, 4.0]), steps=2))
# constant series: SES alpha undefined-ish; what happens?
try:
    fcc = tsecon.theta_forecast(np.full(50, 7.0), steps=3)
    print("[note] constant series ->", fcc)
except Exception as e:
    print("[note] constant series raises:", type(e).__name__, str(e)[:80])

# 7. scale equivariance
fc_s = tsecon.theta_forecast(y_seas * 1e6, steps=12, period=12)
check("forecast equivariant in scale", np.allclose(fc_s, fc * 1e6, rtol=1e-9),
      f"max rel {np.max(np.abs(fc_s / 1e6 - fc) / np.abs(fc)):.2e}")

print(f"\ncomparisons attempted: {attempted}, made: {made}, failures: {len(fails)}")
for f in fails:
    print("  FAIL:", f)
