# The batch-2 docstrings/CHANGELOG say `impact=` (structural_fevd, historical_decomposition) takes
# "an ndarray, a pandas object or a nested list", and condition_sd/conditions take a DataFrame; the
# runtime accepts them, but the stub annotations (widened in batch 2 to add Sequence[Sequence[float]])
# exclude pandas, so `mypy --strict` with pandas-stubs rejects the documented calls.
# mypy leg: see mypy/typed_pandas_documented.py and mypy/mypy_pandas.out, produced by
#   MYPYPATH=<audit16/api>/mypy/stubs <audit16/api>/mypyvenv/bin/mypy --strict typed_pandas_documented.py
import ast, inspect, os, numpy as np, pandas as pd, tsecon
stub = os.path.join(os.path.dirname(tsecon.__file__), '__init__.pyi')
tree = ast.parse(open(stub).read())
fn = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
def ann(f, arg):
    a = fn[f].args
    for x in a.args + a.kwonlyargs:
        if x.arg == arg: return ast.unparse(x.annotation)
for f, arg in (('structural_fevd', 'impact'), ('historical_decomposition', 'impact'), ('var_conditional_forecast', 'conditions'), ('var_conditional_forecast', 'condition_sd')):
    doc = inspect.getdoc(getattr(tsecon, f))
    print(f'{f}({arg}): stub type = {ann(f, arg)}; docstring mentions pandas: {"pandas" in doc}')
X = np.random.default_rng(0).standard_normal((200, 2))
A0 = pd.DataFrame([[1.0, 0.0], [0.5, 1.0]])
print('runtime accepts DataFrame impact:', tsecon.structural_fevd(X, impact=A0)['impact'], tsecon.historical_decomposition(X, identification='impact', impact=A0)['impact'])
print('runtime accepts DataFrame conditions/condition_sd:', tsecon.var_conditional_forecast(X, pd.DataFrame([[1.0, np.nan]]), condition_sd=pd.DataFrame([[0.5, np.nan]]), soft='signal')['mode'])
here = os.path.dirname(os.path.abspath(__file__))
print(open(os.path.join(here, 'mypy', 'mypy_pandas.out')).read())
