"""dfm_smooth (loadings, factor_ar, factor_cov) and var_conditional_forecast
(conditions, condition_sd) accept complex arrays and silently use the real
part (NumPy prints one ComplexWarning per call site at most; under
warnings.simplefilter('ignore') or in a second call there is nothing). This
batch refuses exactly this for `impact` on historical_decomposition and
structural_fevd ('a complex matrix is not truncated to its real part'), and
the other batch-2 data arguments (adl_midas y/hf_lags, multivariate_normality
resid, long_run_svar_bands data) refuse complex too."""
import warnings
import numpy as np, tsecon
warnings.simplefilter("ignore")
rng = np.random.default_rng(3)
T, N = 60, 5
L = rng.standard_normal((N, 1)); A = np.array([[0.7]]); ID = np.full(N, 0.5)
P = rng.standard_normal((T, N))
base = tsecon.dfm_smooth(P, L, A, ID)
for name, kw in [("loadings", dict(loadings=L + 5j)), ("factor_ar", dict(factor_ar=A + 0.9j)),
                 ("factor_cov", dict(factor_cov=np.eye(1) + 3j))]:
    args = dict(loadings=L, factor_ar=A, idiosyncratic=ID); args.update(kw)
    r = tsecon.dfm_smooth(P, **args)
    print(f"dfm_smooth({name}=<complex>): accepted, loglik identical to the real-part call: {r['loglik'] == base['loglik']}")
X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
c0 = tsecon.var_conditional_forecast(X, np.array([[1.0, np.nan]]), condition_sd=np.array([[0.3, np.nan]]), soft="signal")
c1 = tsecon.var_conditional_forecast(X, np.array([[1.0 + 7j, np.nan]]), condition_sd=np.array([[0.3 + 9j, np.nan]]), soft="signal")
print("var_conditional_forecast(complex conditions & condition_sd): accepted, point identical:", c1["point"] == c0["point"])
for name, f in [("historical_decomposition(impact=<complex>)", lambda: tsecon.historical_decomposition(X, identification="impact", impact=np.eye(2) + 1j)),
                ("adl_midas(y=<complex>)", lambda: tsecon.adl_midas(X[:, 0] + 1j, X[:, 1:])),
                ("multivariate_normality(<complex>)", lambda: tsecon.multivariate_normality(X + 1j))]:
    try:
        f(); print(name, "accepted")
    except (TypeError, ValueError) as e:
        print(f"{name}: refused ({type(e).__name__})")
