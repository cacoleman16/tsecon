"""auto_ets accepts n_sim=0 and n_sim=2**31 silently (echoing n_sim back)
when the selected model is class 1, while ets_fit refuses the same values by
name ("n_sim = 0 is invalid ...", "n_sim = 2147483648 is invalid: n_sim x
horizon ... 2^28") and refuses an inert n_sim on a class-1 model outright."""
import numpy as np, tsecon
r = np.random.default_rng(9); e = r.standard_normal(200); y = np.empty(200); p = 0.0
for t in range(200):
    p = 0.5 * p + e[t]; y[t] = p
y = 10.0 + np.abs(y) + np.arange(200) * 0.05
for n_sim in [0, 2**31]:
    out = tsecon.auto_ets(y, horizon=3, level=0.8, n_sim=n_sim, seed=1, allow_multiplicative_trend=True)
    print(f"auto_ets(n_sim={n_sim}): EXPECTED refusal naming n_sim; ACTUAL returned spec={out['spec']} class1={out['class1']} n_sim echo={out['n_sim']}")
    for err in ["add", "mul"]:
        try:
            tsecon.ets_fit(y, error=err, horizon=3, level=0.8, n_sim=n_sim, seed=1)
            print(f"   ets_fit(error={err!r}, n_sim={n_sim}): returned")
        except Exception as ex:
            print(f"   ets_fit(error={err!r}, n_sim={n_sim}): {type(ex).__name__}: {str(ex)[:110]}")
