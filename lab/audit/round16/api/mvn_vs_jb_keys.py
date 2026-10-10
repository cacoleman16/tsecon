# multivariate_normality(x) on one series is advertised as the univariate companion of jarque_bera(x)
# ("At k = 1 the three Jarque-Bera forms equal jarque_bera(x)['statistic']"), but the two callables
# use the SAME key names `skewness` / `kurtosis` for DIFFERENT quantities: jarque_bera returns the sample
# skewness and the raw (non-excess) kurtosis; multivariate_normality returns the chi-square component
# statistics T*b1^2/6 and T*b2^2/24, with the moments under `*_components` (and kurtosis there is EXCESS).
import numpy as np, tsecon, scipy.stats as ss
x = np.random.default_rng(5).standard_t(5, 80)
jb = tsecon.jarque_bera(x); mv = tsecon.multivariate_normality(x)
print("statistic agrees:", jb['statistic'], mv['jarque_bera'])
print("jarque_bera['skewness']           =", jb['skewness'], "(scipy.stats.skew =", ss.skew(x), ")")
print("multivariate_normality['skewness'] =", mv['skewness'], "(= T*b1^2/6 =", len(x) * ss.skew(x) ** 2 / 6, ")")
print("jarque_bera['kurtosis']           =", jb['kurtosis'], "(raw kurtosis, scipy fisher=False =", ss.kurtosis(x, fisher=False), ")")
print("multivariate_normality['kurtosis'] =", mv['kurtosis'], "| kurtosis_components =", mv['kurtosis_components'], "(EXCESS)")
print("count key: jarque_bera 'n' vs multivariate_normality 'nobs'")
