"""Golden fixtures for tsecon-longmemory: fractional differencing / integration,
the Geweke-Porter-Hudak (1983) log-periodogram estimator of the memory
parameter d, and the Robinson (1995) local-Whittle estimator.

These are ALL DOCUMENTED-FORMULA goldens. On fixed inputs NumPy computes every
published quantity by literally writing out the closed form below — it NEVER
calls the Rust crate, so the checks are non-circular. Matching them proves the
Rust reproduces the documented algebra. It does NOT by itself prove the
estimators recover a true d; that statistical claim is what the crate's seeded
Monte-Carlo property tests establish (ARFIMA(0,d,0), d in {0.2, 0.4}).

Run with the project venv:
    .venv/bin/python fixtures/generate_longmemory_fixtures.py

================================================================================
(1) FRACTIONAL DIFFERENCING  (1 - L)^d  — documented-formula, ~1e-12
================================================================================
Binomial expansion of (1 - L)^d = sum_{k>=0} pi_k L^k with the exact recursion
    pi_0 = 1,   pi_k = pi_{k-1} * (k - 1 - d) / k,     k = 1, 2, ...
Applied to a finite sample x_0..x_{n-1}, truncated at the start of the sample:
    y_t = sum_{k=0}^{t} pi_k * x_{t-k},     t = 0 .. n-1.
Fractional integration is (1 - L)^{-d}, i.e. the same convolution with weights
pi_k(-d). It is the EXACT inverse of fractional differencing by d (the filter is
lower-triangular Toeplitz with unit diagonal), so
    frac_integrate(frac_diff(x, d), d) == x   to round-off.

================================================================================
(2) GPH log-periodogram regression  — documented-formula, ~1e-8
================================================================================
Raw periodogram at the lowest m Fourier frequencies lambda_j = 2*pi*j/n,
j = 1..m:  I_j = |X_j|^2 where X = rfft(x)  (any overall scaling cancels below).
Regressor and response:
    R_j = -2 * log( 2 * sin(lambda_j / 2) )
    Y_j = log I_j
OLS of Y on [1, R] (slope = d_hat):  with Rbar = mean(R),
    Sxx      = sum_j (R_j - Rbar)^2
    d_hat    = sum_j (R_j - Rbar) * (Y_j - Ybar) / Sxx
    resid_j  = Y_j - (intercept + d_hat * R_j)
    s2       = sum_j resid_j^2 / (m - 2)
    se_reg   = sqrt( s2 / Sxx )              (OLS nonrobust SE of the slope)
Documented GPH asymptotic SE (uses sum(R-Rbar)^2 -> 4m):
    se       = pi / sqrt(24 * m)
d_hat, se, and se_reg are all invariant to the overall periodogram scaling
(a constant shift of Y is absorbed by the intercept), so they are reproduced
regardless of the FFT's normalization convention.

================================================================================
(3) Robinson (1995) local-Whittle  — documented-formula, ~1e-6
================================================================================
Concentrated Gaussian-semiparametric objective over d in (-1/2, 1):
    R(d) = log( (1/m) sum_{j=1}^m lambda_j^{2d} I_j ) - (2d/m) sum_{j=1}^m log lambda_j
    d_hat = argmin_{d} R(d)
The minimizer is invariant to the overall scaling of I_j (rescaling adds a
d-independent constant to R). We locate it two independent ways — a dense grid
and scipy's bounded scalar minimizer — and assert they agree before storing.
Asymptotic SE:
    se = 1 / (2 * sqrt(m)).
"""
import json
import platform
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a).ravel()]


def frac_weights(d, n_weights):
    """pi_0..pi_{n_weights-1} of (1 - L)^d by the exact binomial recursion."""
    w = np.zeros(n_weights)
    w[0] = 1.0
    for k in range(1, n_weights):
        w[k] = w[k - 1] * ((k - 1 - d) / k)
    return w


def frac_diff(x, d):
    """(1 - L)^d x, start-of-sample-truncated: y_t = sum_{k=0}^t pi_k x_{t-k}."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = frac_weights(d, n)
    y = np.zeros(n)
    for t in range(n):
        y[t] = sum(w[k] * x[t - k] for k in range(t + 1))
    return y


def frac_integrate(x, d):
    """(1 - L)^{-d} x — the exact inverse of frac_diff by d."""
    return frac_diff(x, -d)


def gen_fracdiff():
    cases = []
    # A fixed, distinct-valued input so every weight is exercised.
    x = np.array([1.0, -2.0, 3.5, 0.0, 4.2, -1.1, 2.7, -3.3,
                  0.9, 1.6, -0.4, 5.0, -2.5, 3.1, 0.2, -1.8])
    for d in (0.3, -0.4, 0.75, 1.0):
        nw = len(x)
        w = frac_weights(d, nw)
        fd = frac_diff(x, d)
        fi = frac_integrate(x, d)
        rt = frac_integrate(frac_diff(x, d), d)  # round-trip == x
        cases.append(dict(
            d=float(d), n_weights=nw,
            x=full(x), weights=full(w),
            frac_diff=full(fd), frac_integrate=full(fi), roundtrip=full(rt),
        ))
    return dict(cases=cases)


def arfima_0d0(n, d, rng):
    """Simulate ARFIMA(0, d, 0): x = (1 - L)^{-d} e, e ~ N(0,1)."""
    e = rng.standard_normal(n)
    return frac_integrate(e, d)


def periodogram_lowfreq(x, m):
    """(lambda_j, I_j) for j = 1..m; I_j = |rfft(x)[j]|^2 (raw)."""
    n = len(x)
    X = np.fft.rfft(x)
    j = np.arange(1, m + 1)
    lam = 2.0 * np.pi * j / n
    I = np.abs(X[1:m + 1]) ** 2
    return lam, I


def gph_estimate(x, m):
    lam, I = periodogram_lowfreq(x, m)
    R = -2.0 * np.log(2.0 * np.sin(lam / 2.0))
    Y = np.log(I)
    Rbar, Ybar = R.mean(), Y.mean()
    Sxx = np.sum((R - Rbar) ** 2)
    d_hat = np.sum((R - Rbar) * (Y - Ybar)) / Sxx
    intercept = Ybar - d_hat * Rbar
    resid = Y - (intercept + d_hat * R)
    s2 = np.sum(resid ** 2) / (m - 2)
    se_reg = np.sqrt(s2 / Sxx)
    se = np.pi / np.sqrt(24.0 * m)
    return dict(d=float(d_hat), se=float(se), se_regression=float(se_reg),
                intercept=float(intercept))


def local_whittle_estimate(x, m):
    lam, I = periodogram_lowfreq(x, m)
    log_lam = np.log(lam)
    sum_log_lam = log_lam.sum()

    def R(d):
        weighted = np.mean(lam ** (2.0 * d) * I)
        return np.log(weighted) - (2.0 * d / m) * sum_log_lam

    # Two independent minimizations must agree.
    grid = np.linspace(-0.49, 0.99, 148001)  # 1e-5 spacing
    d_grid = grid[np.argmin([R(d) for d in grid])]
    res = minimize_scalar(R, bounds=(-0.4999, 0.9999), method="bounded",
                          options={"xatol": 1e-11})
    d_opt = float(res.x)
    assert abs(d_opt - d_grid) < 1e-4, (d_opt, d_grid)
    se = 1.0 / (2.0 * np.sqrt(m))
    return dict(d=d_opt, se=float(se), objective=float(R(d_opt)))


def gen_semiparametric():
    rng = np.random.default_rng(20260717)
    n = 1024
    d_true = 0.3
    x = arfima_0d0(n, d_true, rng)
    m = int(np.floor(np.sqrt(n)))  # = 32
    gph = gph_estimate(x, m)
    lw = local_whittle_estimate(x, m)
    return dict(n=n, d_true=d_true, m=m, x=full(x), gph=gph, whittle=lw)


def main():
    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": "Long memory (tsecon-longmemory): fractional differencing / "
                    "integration (~1e-12), GPH log-periodogram regression "
                    "(~1e-8), and Robinson (1995) local Whittle (~1e-6). All "
                    "documented-formula goldens; formulas in the generator "
                    "docstring. NumPy never calls the Rust crate.",
        },
        "fracdiff": gen_fracdiff(),
        "semiparametric": gen_semiparametric(),
    }
    (OUT / "longmemory.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote longmemory.json")


if __name__ == "__main__":
    main()
