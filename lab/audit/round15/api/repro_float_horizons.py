"""Integer-valued float horizons/orders: variance_ratio accepts them inside a
sequence but refuses the same value as a scalar; ardl_bounds_test refuses both
forms. Same concept (an integer lag count given as 4.0), three behaviours."""
import numpy as np, tsecon
rng = np.random.default_rng(0)
lvl = np.cumsum(rng.standard_normal(300))
y = np.cumsum(rng.standard_normal(300)); x = np.cumsum(rng.standard_normal((300, 2)), axis=0)
def t(label, f):
    try:
        r = f(); print(f"{label:42s} ACCEPTED  lags/order echo = {r.get('lags') if 'order' not in r else r['order']}")
    except Exception as e:
        print(f"{label:42s} REFUSED   {type(e).__name__}: {str(e)[:90]}")
t("variance_ratio(lags=4.0)", lambda: tsecon.variance_ratio(lvl, lags=4.0))
t("variance_ratio(lags=[4.0])", lambda: tsecon.variance_ratio(lvl, lags=[4.0]))
t("variance_ratio(lags=[2.0, 4.0])", lambda: tsecon.variance_ratio(lvl, lags=[2.0, 4.0]))
t("variance_ratio(lags=np.array([2., 4.]))", lambda: tsecon.variance_ratio(lvl, lags=np.array([2.0, 4.0])))
t("ardl_bounds_test(order=2.0)", lambda: tsecon.ardl_bounds_test(y, x, lags=2, order=2.0))
t("ardl_bounds_test(order=[2.0, 2.0])", lambda: tsecon.ardl_bounds_test(y, x, lags=2, order=[2.0, 2.0]))
t("ardl_bounds_test(order=np.array([2., 2.]))", lambda: tsecon.ardl_bounds_test(y, x, lags=2, order=np.array([2.0, 2.0])))
