"""Golden fixture for tsecon-panel::panel_lp with bias_correction = "spj" —
the Mei, Sheng & Shi (2026) split-panel jackknife for panel local
projections.

Run with a NumPy-only venv (this script never imports tsecon):

    .venv/bin/python fixtures/generate_panel_spj_fixtures.py

============================================================================
WHAT KIND OF GOLDEN IS THIS
============================================================================
This is a **TRANSCRIPTION golden**, NOT an external-package golden. The
method's reference implementation is R-only: the `pLP` package
(github.com/zhentaoshi/panel-local-projection, function `panelLP`,
`R/panelLP.R` fetched 2026-08-18), companion code to

    Mei, Z., Sheng, L. & Shi, Z. (2026). "Nickell bias in panel local
    projection: Financial crises are worse than you think."
    Journal of International Economics (arXiv:2302.13455).

That repository commits real datasets (Romer-Romer 2017 etc.) but NO stored
numeric outputs, and no R interpreter is available in this environment, so a
run-the-reference golden is not obtainable here. Instead this generator
reimplements the `panelLP.R` algebra independently in NumPy — the split
convention, the half demeaning, the 2F - (A+B)/2 combination, and both of
the reference SE constructions — and the Rust crate must reproduce it
through a different numerical path (faer QR / Cholesky vs. NumPy lstsq /
inv) to 1e-10 relative. Validation grade, stated honestly: transcription of
the authors' reference algebra + seeded Monte Carlo property tests
(bias-reduction and coverage in crates/tsecon-panel/tests/spj_properties.rs
and on the panel model card); not an output-level match against a run of
their code.

============================================================================
THE ESTIMATOR (panelLP.R, method = "SPJ"; balanced-panel case)
============================================================================
Per horizon h with L = max(shock_lags, outcome_lags) lags, the usable
regression rows are t in [L, T - h) (0-based): the regressand y_{i,t+h} (or
the cumulated sum_{j<=h} y_{i,t+j}) on [shock_t, shock lags, outcome lags],
entity-demeaned (within). The reference then:

1. fits the FULL usable sample -> beta_full, demeaned design Xd, demeaned
   outcome yd;
2. splits the usable ROWS at the floor of the median usable period
   (R: cut = floor(median(complete-case indices))); in 0-based terms with
   a0 = L, b0 = T - h - 1 the first half is t in [a0, c0] and the second
   t in [c0 + 1, b0] with c0 = (a0 + b0) // 2 — an odd row count gives the
   extra row to the FIRST half; the halves never overlap. Leads and lags
   keep indexing the FULL panel (they cross the split), only the rows are
   split; each half is re-demeaned within itself -> beta_a, beta_b;
3. combines: beta_spj = 2 beta_full - (beta_a + beta_b) / 2;
4. computes SEs at the CORRECTED coefficients with jackknife-adjusted
   scores: residual e = yd - Xd beta_spj, adjusted regressor
   dd_it = 2 x~_it - x~half_it (x~half demeaned within the half containing
   t), bread = (Xd'Xd)^{-1} from the full sample, and either
   * cluster by entity: W = sum_i (sum_t dd_it e_it)(...)', scaled by the
     Stata-style factor (N/(N-1)) * ((n-1)/(n-k)) (absorbed effects NOT
     counted; group debias ON) — the pLP default, or
   * Driscoll-Kraay: Bartlett HAC on the per-period sums
     a_t = sum_i dd_it e_it with weights w_j = 1 - j/(bw+1) and NO
     small-sample factor (pLP's dk_var). pLP hardcodes the truncation
     bw = floor((T-h)^(1/4)); tsecon honours the user bandwidth, and this
     fixture sets bw = 2 = floor((T-h)^(1/4)) for every stored horizon so
     the stored numbers ARE the pLP auto-bandwidth ones.

These SPJ covariance conventions intentionally differ from the linearmodels
conventions of tsecon's uncorrected panel_lp route; the divergence is
documented in crates/tsecon-panel/src/lp.rs and on the panel model card.

============================================================================
DGP — dynamic panel hit by a common shock (Nickell territory)
============================================================================
y_{i,t} = alpha_i + rho y_{i,t-1} + beta s_t + eps_{i,t}, a balanced panel
with N = 12 entities and T = 41 periods (odd, to pin the odd-split
convention), rho = 0.5, beta = 0.8, alpha_i ~ N(0,1), s_t iid N(0,1),
eps ~ N(0, 0.7^2), 30 burn-in periods. With controls (shock lag + outcome
lag) the horizon-h projection coefficient on s_t is beta * rho^h.
"""

import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
META = {
    "numpy": np.__version__,
    "python": platform.python_version(),
    "reference": "pLP::panelLP (github.com/zhentaoshi/panel-local-projection, "
    "R/panelLP.R, fetched 2026-08-18); Mei-Sheng-Shi 2026 JIE / "
    "arXiv:2302.13455",
}

rng = np.random.default_rng(20260818)

N, T = 12, 41
RHO, BETA = 0.5, 0.8
SIGMA_E = 0.7
HMAX = 4
SHOCK_LAGS, OUTCOME_LAGS = 1, 1
K = 1 + SHOCK_LAGS + OUTCOME_LAGS
L = max(SHOCK_LAGS, OUTCOME_LAGS)

BURN = 30
shock_all = rng.standard_normal(T + BURN)
alpha = rng.standard_normal(N)
y_all = np.zeros((N, T + BURN))
y_all[:, 0] = alpha + BETA * shock_all[0] + SIGMA_E * rng.standard_normal(N)
for t in range(1, T + BURN):
    y_all[:, t] = (
        alpha
        + RHO * y_all[:, t - 1]
        + BETA * shock_all[t]
        + SIGMA_E * rng.standard_normal(N)
    )
y = y_all[:, BURN:]
shock = shock_all[BURN:]


def build_rows(rows, h, cumulative):
    """Regression rows for horizon h at the given 0-based periods `rows`;
    leads and lags index the FULL panel (the MSS bookkeeping)."""
    s = len(rows)
    yv = np.empty((N, s))
    x = np.empty((N, s, K))
    for idx, t in enumerate(rows):
        if cumulative:
            yv[:, idx] = y[:, t : t + h + 1].sum(axis=1)
        else:
            yv[:, idx] = y[:, t + h]
        x[:, idx, 0] = shock[t]
        for lag in range(1, SHOCK_LAGS + 1):
            x[:, idx, lag] = shock[t - lag]
        for lag in range(1, OUTCOME_LAGS + 1):
            x[:, idx, SHOCK_LAGS + lag] = y[:, t - lag]
    return yv, x


def within_fit(yv, x):
    """Entity-demean and stack entity-major; OLS via lstsq (SVD)."""
    yd = yv - yv.mean(axis=1, keepdims=True)
    xd = x - x.mean(axis=1, keepdims=True)
    s = yv.shape[1]
    a = xd.reshape(N * s, K)
    b = yd.reshape(N * s)
    beta, *_ = np.linalg.lstsq(a, b, rcond=None)
    return beta, a, b


def bartlett_weight(lag, bw):
    w = 1.0 - lag / (bw + 1.0)
    return w if w > 0.0 else 0.0


def spj_horizon(h, cumulative, se_kind, bw):
    a0, b0 = L, T - h - 1
    c0 = (a0 + b0) // 2  # floor(median) split of the usable rows (pLP)
    rows_full = np.arange(a0, b0 + 1)
    rows_a = np.arange(a0, c0 + 1)
    rows_b = np.arange(c0 + 1, b0 + 1)
    s_full, s_a, s_b = len(rows_full), len(rows_a), len(rows_b)

    beta_full, xd_full, yd_full = within_fit(*build_rows(rows_full, h, cumulative))
    beta_a, xd_a, _ = within_fit(*build_rows(rows_a, h, cumulative))
    beta_b, xd_b, _ = within_fit(*build_rows(rows_b, h, cumulative))
    beta_spj = 2.0 * beta_full - 0.5 * (beta_a + beta_b)

    # Residuals at the corrected coefficients on the full demeaned data.
    e = yd_full - xd_full @ beta_spj

    # Adjusted scores dd = 2 x~ - x~half, rows aligned entity-major.
    dd = np.empty_like(xd_full)
    for i in range(N):
        r0 = i * s_full
        dd[r0 : r0 + s_a] = (
            2.0 * xd_full[r0 : r0 + s_a] - xd_a[i * s_a : (i + 1) * s_a]
        )
        dd[r0 + s_a : r0 + s_full] = (
            2.0 * xd_full[r0 + s_a : r0 + s_full] - xd_b[i * s_b : (i + 1) * s_b]
        )

    bread = np.linalg.inv(xd_full.T @ xd_full)
    n = N * s_full
    if se_kind == "cluster":
        w = np.zeros((K, K))
        for i in range(N):
            gi = dd[i * s_full : (i + 1) * s_full].T @ e[i * s_full : (i + 1) * s_full]
            w += np.outer(gi, gi)
        scale = (N / (N - 1)) * ((n - 1) / (n - K))
        cov = bread @ (scale * w) @ bread
    elif se_kind == "driscoll_kraay":
        agg = np.zeros((s_full, K))
        for i in range(N):
            agg += dd[i * s_full : (i + 1) * s_full] * e[
                i * s_full : (i + 1) * s_full, None
            ]
        meat = agg.T @ agg
        for lag in range(1, s_full):
            wt = bartlett_weight(lag, bw)
            if wt == 0.0:
                break
            gamma = agg[lag:].T @ agg[:-lag]
            meat += wt * (gamma + gamma.T)
        cov = bread @ meat @ bread  # no small-sample factor (pLP dk_var)
    else:
        raise ValueError(se_kind)

    return {
        "cut0": int(c0),
        "s_a": int(s_a),
        "s_b": int(s_b),
        "nobs": int(n),
        "beta_full": [float(v) for v in beta_full],
        "beta_a": [float(v) for v in beta_a],
        "beta_b": [float(v) for v in beta_b],
        "beta_spj": [float(v) for v in beta_spj],
        "se_spj": [float(v) for v in np.sqrt(np.diag(cov))],
    }


DK_BW = 2.0
for h in range(HMAX + 1):
    auto = int(np.floor((T - h) ** 0.25))
    assert auto == int(DK_BW), (h, auto)  # stored DK numbers = pLP auto bw

cases = {
    "spj_cluster": {"cumulative": False, "se": "cluster", "bandwidth": None},
    "spj_dk_bw2": {"cumulative": False, "se": "driscoll_kraay", "bandwidth": DK_BW},
    "spj_cluster_cumulative": {
        "cumulative": True,
        "se": "cluster",
        "bandwidth": None,
    },
}

result = {}
for name, spec in cases.items():
    result[name] = {
        "cumulative": spec["cumulative"],
        "se_type": spec["se"],
        "bandwidth": spec["bandwidth"],
        "horizons": [
            spj_horizon(h, spec["cumulative"], spec["se"], spec["bandwidth"])
            for h in range(HMAX + 1)
        ],
    }
    irf = [hh["beta_spj"][0] for hh in result[name]["horizons"]]
    se0 = [hh["se_spj"][0] for hh in result[name]["horizons"]]
    print(f"{name:24s} irf {np.round(irf, 4)}  se {np.round(se0, 4)}")

true_irf = [BETA * RHO**h for h in range(HMAX + 1)]
print(f"{'true beta*rho^h':24s}     {np.round(true_irf, 4)}")

out = {
    "_meta": META,
    "_doc": "transcription golden: NumPy reimplementation of the pLP::panelLP "
    "SPJ algebra (split, combination, adjusted-score cluster and "
    "Driscoll-Kraay sandwiches); NOT a stored-output match of the R "
    "package (no numeric outputs are committed there and R is "
    "unavailable here). See the generator docstring.",
    "design": {
        "N": N,
        "T": T,
        "rho": RHO,
        "beta": BETA,
        "sigma_e": SIGMA_E,
        "max_horizon": HMAX,
        "shock_lags": SHOCK_LAGS,
        "outcome_lags": OUTCOME_LAGS,
    },
    "true_irf": true_irf,
    "y": [[float(v) for v in row] for row in y],
    "shock": [float(v) for v in shock],
    "cases": result,
}

path = OUT / "panel_spj.json"
path.write_text(json.dumps(out))
print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
