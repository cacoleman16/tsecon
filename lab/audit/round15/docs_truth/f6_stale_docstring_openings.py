"""Two CHANGED callables kept an opening paragraph the wave made false.
(a) help(tsecon.var_forecast) still opens 'Iterated VAR point forecasts with
(innovation-uncertainty) intervals ... Intervals reflect innovation uncertainty
only (coefficients treated as known)', yet param_uncertainty=True (documented
further down the same docstring) widens lower/upper by Omega(h)/T. The stub /
api.md opening was rewritten ('with (1-alpha) intervals'), so help() and the
stub now disagree on this point.
(b) help(tsecon.ols) still opens '`se_type`: "nonrobust", "hc0", "hc1", "hc2",
"hc3", or "hac"', omitting "ewc", while the stub opens 'nonrobust / HC0 / HC1 /
HC2 / HC3 / HAC / EWC'."""
import ast, json, numpy as np, tsecon
W = str(__import__("pathlib").Path(__file__).resolve().parents[4])
stub = {n.name: ast.get_docstring(n) for n in ast.parse(open(f"{W}/bindings/python/python/tsecon/__init__.pyi").read()).body
        if isinstance(n, ast.FunctionDef)}
rt = tsecon.var_forecast.__doc__
print("runtime var_forecast says 'innovation uncertainty only':", "innovation uncertainty only" in " ".join(rt.split()))
print("stub var_forecast says it:", "innovation uncertainty only" in " ".join(stub["var_forecast"].split()))
data = np.asarray(json.load(open(f"{W}/fixtures/var.json"))["data_100dlog_gdp_cons_inv"])
a = tsecon.var_forecast(data, lags=2, steps=8)
b = tsecon.var_forecast(data, lags=2, steps=8, param_uncertainty=True)
print("half-width ratio with param_uncertainty=True at h=8:",
      np.round((np.asarray(b["upper"]) - np.asarray(b["point"]))[7] / (np.asarray(a["upper"]) - np.asarray(a["point"]))[7], 4))
first = tsecon.ols.__doc__.split("\n\n")[1]
print("runtime ols first se_type list:", " ".join(first.split())[:140])
print("stub ols first line:", stub["ols"].splitlines()[0])
