"""B1_insufficient_data_unnamed

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_B1_insufficient_data_unnamed.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('nlags', 'tsecon.arch_lm(resid=G, nlags=2**31)'),  # arch_lm:nlags=2**31 -> unnamed: 'arch_lm: series has 500 observations but needs at least 4294967298; supply more data or re'
    ('lags', 'tsecon.dfgls(y=RW, lags=2**31)'),  # dfgls:lags=2**31 -> unnamed: 'dfgls: series has 200 observations but needs at least 4294967299; supply more data or redu'
    ('max_lags', 'tsecon.dfgls(y=RW, max_lags=2**31)'),  # dfgls#auto:max_lags=2**31 -> unnamed: 'dfgls: series has 200 observations but needs at least 4294967299; supply more data or redu'
    ('lags', 'tsecon.ng_perron(y=RW, lags=2**31)'),  # ng_perron:lags=2**31 -> unnamed: 'ng_perron: series has 200 observations but needs at least 4294967299; supply more data or '
    ('max_lags', 'tsecon.ng_perron(y=RW, max_lags=2**31)'),  # ng_perron#auto:max_lags=2**31 -> unnamed: 'ng_perron: series has 200 observations but needs at least 4294967299; supply more data or '
    ('calib', 'tsecon.conformal_backtest(y=POS, horizon=1, calib=2**31, n_eval=20)'),  # conformal_backtest:calib=2**31 -> unnamed: 'split conformal (online): series has 200 observations but needs at least 2147483670; suppl'
    ('n_eval', 'tsecon.conformal_backtest(y=POS, horizon=1, calib=30, n_eval=2**31)'),  # conformal_backtest:n_eval=2**31 -> unnamed: 'split conformal (online): series has 200 observations but needs at least 2147483680; suppl'
    ('lags', "tsecon.conformal_backtest(y=POS, method='enbpi', base='ar', lags=2**31, n_boot=10, seed=1, n_eval=20)"),  # conformal_backtest#enbpi:lags=2**31 -> unnamed: 'EnbPI: series has 200 observations but needs at least 4294967318; supply more data or redu'
    ('calib', 'tsecon.conformal_forecast(y=POS, horizon=2, calib=2**31)'),  # conformal_forecast:calib=2**31 -> unnamed: 'split conformal: series has 200 observations but needs at least 2147483651; supply more da'
    ('n_eval', "tsecon.conformal_forecast(y=POS, method='aci', n_eval=2**31, calib=30)"),  # conformal_forecast#aci:n_eval=2**31 -> unnamed: 'adaptive conformal inference: series has 200 observations but needs at least 2147483680; s'
    ('lags', "tsecon.conformal_forecast(y=POS, base='ar', lags=2**31, calib=30)"),  # conformal_forecast#ar:lags=2**31 -> unnamed: 'AR least-squares forecast: series has 170 observations but needs at least 4294967298; supp'
    ('lags', "tsecon.conformal_forecast(y=POS, method='enbpi', base='ar', lags=2**31, n_boot=10, seed=1)"),  # conformal_forecast#enbpi:lags=2**31 -> unnamed: 'EnbPI: series has 200 observations but needs at least 4294967298; supply more data or redu'
    ('insample_period', 'tsecon.backtest(y=POS, train=100, horizon=2, refit_every=5, insample_period=2**31)'),  # backtest:insample_period=2**31 -> unnamed: 'MASE: series has 100 observations but needs at least 2147483649; supply more data or reduc'
    ('insample_period', 'tsecon.backtest(y=POS, train=100, horizon=2, refit_every=5, insample_period=0)'),  # backtest:insample_period=0 -> unnamed: 'MASE: period = 0 is invalid for a series of length 100: requires period >= 1 (use 1 for no'
    ('lags', 'tsecon.dcc_test(returns=G2, lags=2**31)'),  # dcc_test:lags=2**31 -> unnamed: 'insufficient data: 500 observations, at least 2147483650 required'
    ('block', "tsecon.kernel_regression(x=XNC[:, 0], y=YNC, bandwidth_method='block_cv', block=2**31)"),  # kernel_regression:block=2**31 -> unnamed: 'insufficient data: 200 observations, at least 4294967299 required (kernel_regression)'
    ('min_samples_leaf', 'tsecon.random_forest(x=XNC, y=YNC, n_trees=20, max_features=2, max_depth=4, min_samples_leaf=2**31)'),  # random_forest:min_samples_leaf=2**31 -> unnamed: 'insufficient data: 200 observations, at least 4294967296 required (tree fit)'
    ('min_samples_leaf', 'tsecon.regression_tree(x=XNC, y=YNC, max_depth=3, min_samples_leaf=2**31)'),  # regression_tree:min_samples_leaf=2**31 -> unnamed: 'insufficient data: 200 observations, at least 4294967296 required (tree fit)'
    ('min_samples_split', 'tsecon.regression_tree(x=XNC, y=YNC, max_depth=3, min_samples_split=2**31)'),  # regression_tree:min_samples_split=2**31 -> unnamed: 'insufficient data: 200 observations, at least 2147483648 required (tree fit)'
    ('period', 'tsecon.stl(y=SEAS, period=2**31, seasonal=7, trend=23, low_pass=13, inner_iter=2, outer_iter=0)'),  # stl:period=2**31 -> unnamed: 'stl needs at least 4294967296 observations but got 96: STL smooths each cycle-subseries (a'
    ('period', 'tsecon.seasonal_strength(y=SEAS, period=2**31)'),  # seasonal_strength:period=2**31 -> unnamed: 'stl needs at least 4294967296 observations but got 96: STL smooths each cycle-subseries (a'
    ('period', 'tsecon.nsdiffs(y=SEAS, period=2**31, max_d=1)'),  # nsdiffs:period=2**31 -> unnamed: 'stl needs at least 4294967296 observations but got 96: STL smooths each cycle-subseries (a'
    ('seasonal_period', 'tsecon.auto_arima(y=Y, max_p=2, max_q=2, forecast_steps=2, seasonal_period=2**31)'),  # auto_arima:seasonal_period=2**31 -> unnamed: 'auto_arima: choosing the seasonal differencing order D via the seasonal-strength rule fail'
    ('D', 'tsecon.auto_arima(y=SEAS, seasonal_period=12, max_p=1, max_q=1, max_P=1, max_Q=1, D=2**31)'),  # auto_arima#seas:D=2**31 -> unnamed: 'ARIMA needs at least 13 usable observations for seasonal differencing before the KPSS d-se'
]

if __name__ == "__main__":
    run(CASES, timeout=60)
