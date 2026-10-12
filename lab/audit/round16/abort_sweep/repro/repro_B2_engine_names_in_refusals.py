"""B2_engine_names_in_refusals

Generated from the house-wide abort sweep (results.jsonl). Each call runs in
its own subprocess under `ulimit -v 4000000; timeout 60` with the sweep's
small seeded fixtures (../prelude.py). Expected: a refusal naming the
parameter; actual: see ACTUAL lines (rc=-6 is SIGABRT from the Rust
allocator's `memory allocation of N bytes failed`).
Run: python lab/audit/round16/abort_sweep/repro/repro_B2_engine_names_in_refusals.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repro_common import run

CASES = [
    ('n_draws', 'tsecon.sign_restricted_svar(data=X3, restrictions=SR3, lags=1, horizon=4, n_draws=0, max_tries=100)'),  # sign_restricted_svar:n_draws=0 -> unnamed: 'invalid argument: n_posterior_draws must be at least 1'
    ('n_draws', 'tsecon.narrative_svar(data=X3, sign_restrictions=SR3, lags=1, horizon=4, n_draws=0, max_tries=100, n_weight_draws=20)'),  # narrative_svar:n_draws=0 -> unnamed: 'invalid argument: n_posterior_draws must be at least 1'
    ('n_draws', 'tsecon.zero_sign_svar(data=X3, sign_restrictions=SR3, zero_restrictions=[(2, 0, 0)], lags=1, horizon=4, n_draws=0, max_tries=100)'),  # zero_sign_svar:n_draws=0 -> unnamed: 'invalid argument: n_posterior_draws must be at least 1'
    ('n_draws', 'tsecon.fry_pagan_svar(data=X3, restrictions=SR3, lags=1, horizon=4, n_draws=0, max_tries=100)'),  # fry_pagan_svar:n_draws=0 -> unnamed: 'invalid argument: n_posterior_draws must be at least 1'
    ('n_draws', "tsecon.historical_decomposition(data=X3, restrictions=SR3, identification='sign', lags=1, n_draws=0, max_tries=100)"),  # historical_decomposition#sign:n_draws=0 -> unnamed: 'invalid argument: n_posterior_draws must be at least 1'
    ('lags', 'tsecon.long_run_svar(data=X2, lags=0, horizon=8)'),  # long_run_svar:lags=0 -> unnamed: 'invalid argument: coefs must contain at least one VAR lag matrix'
    ('lags', 'tsecon.bvar_ssvs(data=X2, lags=0, n_draws=200, burn=50, horizon=4)'),  # bvar_ssvs:lags=0 -> unnamed: 'lag length p must be at least 1'
    ('lags', 'tsecon.nongaussian_svar(data=np.column_stack([G[:200], G2[:200, 1]]) + 0.3 * np.column_stack([G2[:200,1], G[:200]]), lags=0, horizon=6, max_iter=100)'),  # nongaussian_svar:lags=0 -> unnamed: 'invalid argument: lag length p must be at least 1'
    ('band_n_sim', "tsecon.lp(y=Y, shock=Y2, horizons=6, n_lag_controls=2, band='sup-t', band_n_sim=0)"),  # lp#supt:band_n_sim=0 -> unnamed: 'simultaneous band: sup-t needs n_sim >= 2 Gaussian simulations, and at least ~50,000 in pr'
    ('band_n_sim', "tsecon.smooth_lp(y=Y, shock=Y2, horizons=6, n_lag_controls=2, lam=1.0, band='sup-t', band_n_sim=0)"),  # smooth_lp#supt:band_n_sim=0 -> unnamed: 'simultaneous band: sup-t needs n_sim >= 2 Gaussian simulations, and at least ~50,000 in pr'
    ('band_n_sim', "tsecon.var_forecast(data=X2, lags=2, steps=4, band='sup-t', band_n_sim=0)"),  # var_forecast#supt:band_n_sim=0 -> unnamed: 'n_sim must be at least 2 to simulate a sup-t critical value; 100000 is the recommended def'
    ('band_n_sim', "tsecon.var_irf_bands(data=X2, lags=1, horizon=6, band='sup-t', band_n_sim=0)"),  # var_irf_bands#supt:band_n_sim=0 -> unnamed: 'n_sim must be at least 2 to simulate a sup-t critical value; 100000 is the recommended def'
    ('max_P', 'tsecon.auto_arima(y=SEAS, seasonal_period=12, max_p=1, max_q=1, max_P=2**31, max_Q=1)'),  # auto_arima#seas:max_P=2**31 -> unnamed: 'max_seasonal_p and max_seasonal_q must be at most 6: each seasonal order multiplies the st'
    ('max_Q', 'tsecon.auto_arima(y=SEAS, seasonal_period=12, max_p=1, max_q=1, max_P=1, max_Q=2**31)'),  # auto_arima#seas:max_Q=2**31 -> unnamed: 'max_seasonal_p and max_seasonal_q must be at most 6: each seasonal order multiplies the st'
    ('rf_draws', "tsecon.proxy_ar_sets(data=X3, proxy=PROXY, lags=2, horizon=6, rf_method='second_order', rf_draws=0, rf_seed=1)"),  # proxy_ar_sets#boot:rf_draws=0 -> unnamed: 'invalid argument: draws must be an even number of at least 32 (the sampler works in antith'
    ('max_lags', 'tsecon.zivot_andrews(y=RW, max_lags=2**31)'),  # zivot_andrews:max_lags=2**31 -> unnamed: 'zivot_andrews: nlags = 2147483648 is invalid for a series of length 200: requires maxlag <'
    ('max_lags', 'tsecon.panel_unit_root(data=POUT, max_lags=2**31)'),  # panel_unit_root#auto:max_lags=2**31 -> unnamed: 'per-unit ADF failed for unit 0: adf: nlags = 2147483648 is invalid for a series of length '
    ('lrv_bandwidth', "tsecon.panel_unit_root(data=POUT, test='llc', lags=1, lrv_bandwidth=-1)"),  # panel_unit_root#llc:lrv_bandwidth=-1 -> unnamed: 'the Levin-Lin-Chu auxiliary regression / long-run variance failed for unit 0: bandwidth = '
    ('maxlag', 'tsecon.adf(y=Y, maxlag=2**31, autolag=None)'),  # adf#fixed:maxlag=2**31 -> named: 'adf: nlags = 2147483648 is invalid for a series of length 200: requires maxlag <= n/2 - nt'
    ('factor_order', 'tsecon.dfm_nowcast(data=DFMR, factor_order=2**31)'),  # dfm_nowcast:factor_order=2**31 -> unnamed: 'factor VAR error: data has 79 rows but a VAR with lags = p = 2147483648 on its k=1 series '
    ('factor_order', 'tsecon.dfm_news(old_vintage=DFMOLD, new_vintage=DFMNEW, factor_order=2**31)'),  # dfm_news:factor_order=2**31 -> unnamed: 'factor VAR error: data has 79 rows but a VAR with lags = p = 2147483648 on its k=1 series '
    ('target_series', 'tsecon.dfm_news(old_vintage=DFMOLD, new_vintage=DFMNEW, target_series=2**31)'),  # dfm_news:target_series=2**31 -> unnamed: 'series index 2147483648 out of range (panel has 5 series)'
    ('n_factors', 'tsecon.factor_model(data=DFMP, n_factors=0, kmax=3)'),  # factor_model:n_factors=0 -> unnamed: 'invalid factor count: number of factors must satisfy 1 <= r <= min(n, N) (requested 0, max'
    ('n_factors', 'tsecon.favar(panel=DFMP, policy=_ar1(80, 3), n_factors=0, lags=1, horizon=8, slow_indices=[0, 1])'),  # favar:n_factors=0 -> unnamed: 'invalid factor count: number of factors must satisfy 1 <= r <= min(n, N) (requested 0, max'
    ('hac_lags', "tsecon.proxy_ar_sets(data=X3, proxy=PROXY, lags=2, horizon=6, variance='hac', hac_lags=2**31)"),  # proxy_ar_sets#hac:hac_lags=2**31 -> unnamed: 'invalid argument: HAC bandwidth must be smaller than the proxy overlap count'
    ('hac_lags', "tsecon.proxy_first_stage(data=X3, proxy=PROXY, lags=2, variance='hac', hac_lags=2**31)"),  # proxy_first_stage#hac:hac_lags=2**31 -> unnamed: 'invalid argument: HAC bandwidth must be smaller than the proxy overlap count'
    ('k_regimes', 'tsecon.markov_switching_ar(y=SETY, k_regimes=0, order=1, max_iter=50)'),  # markov_switching_ar:k_regimes=0 -> unnamed: 'invalid model specification: a Markov-switching model needs at least two regimes'
    ('k_regimes', 'tsecon.markov_switching_ar(y=SETY, k_regimes=2**31, order=1, max_iter=50)'),  # markov_switching_ar:k_regimes=2**31 -> unnamed: 'invalid model specification: expanded state space k^(order+1) exceeds the internal cap of '
    ('lrv_lags', 'tsecon.cw_test(e_small=Y, e_large=Y2, yhat_small=Y3, yhat_large=RW, lrv_lags=2**31)'),  # cw_test:lrv_lags=2**31 -> unnamed: 'Clark-West: long-run-variance lag truncation 2147483648 is invalid for 200 observations: r'
    ('lrv_lags', 'tsecon.gw_test(loss1=Y ** 2, loss2=Y2 ** 2, lrv_lags=2**31)'),  # gw_test:lrv_lags=2**31 -> unnamed: 'Giacomini-White: long-run-variance lag truncation 2147483648 is invalid for 200 observatio'
]

if __name__ == "__main__":
    run(CASES, timeout=60)
