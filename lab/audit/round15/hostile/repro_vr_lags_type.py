"""variance_ratio: the stub docstring (__init__.pyi) says the per-horizon keys, `lags` included,
are "1-D arrays aligned with the given order when it is a sequence"; `lags` comes back as a
Python list (the runtime docstring says so: "(`lags` then echoes a list)"). Stub and runtime
docstring disagree.
"""
import numpy as np, tsecon, re, pathlib
rng = np.random.default_rng(0)
r = tsecon.variance_ratio(np.cumsum(rng.standard_normal(200)), lags=np.array([2, 4]))
print("type(lags) =", type(r["lags"]).__name__, "| type(vr) =", type(r["vr"]).__name__)
stub = pathlib.Path(tsecon.__file__).with_name("__init__.pyi").read_text()
seg = stub[stub.index("def variance_ratio("):][:3500]
print("stub mentions 'echoes a list':", "echoes a list" in seg, "| runtime doc mentions it:", "echoes a list" in tsecon.variance_ratio.__doc__)
