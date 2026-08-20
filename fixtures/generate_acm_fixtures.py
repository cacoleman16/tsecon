"""Golden fixtures for the ACM (Adrian-Crump-Moench 2013) regression-based
term premium (roadmap E2 / build-next "ACM").

VALIDATION-FIRST / NON-CIRCULAR: this generator does NOT call the tsecon Rust
crate. It transcribes the DOCUMENTED three-step linear-regression estimator of

    Adrian, T., Crump, R. K., & Moench, E. (2013). "Pricing the Term
    Structure with Linear Regressions." Journal of Financial Economics,
    110(1), 110-138 (also FRBNY Staff Report 340).

directly into NumPy and runs the ENTIRE pipeline — PCA factor extraction,
VAR(1), excess-return regressions, price-of-risk recovery, and the affine
bond-price recursions — independently. The Rust crate is expected to
reproduce these numbers to ~1e-8.

The pipeline (all conventions the Rust implementation must match)
------------------------------------------------------------------
Inputs: a T x M panel of annualized, continuously-compounded zero-coupon
log-yields in DECIMAL (0.05 = 5%), observed at integer maturities `n_i`
measured in periods (months for monthly data), ascending, containing 1.

0.  Factors: demean the yield panel column-by-column, take the first K right
    singular vectors W (SVD), factors X = Ydemeaned @ W.  Scale each factor
    to unit SAMPLE standard deviation (ddof=1), folding the scale into W, and
    flip signs so each loading column sums positive.  (ACM use the PCs of
    zero-coupon yields; scaling/sign are innocuous normalizations — every
    model output below is exactly invariant to invertible linear maps of the
    factors.)

1.  VAR(1) with intercept, by OLS equation-by-equation:
        X_{t+1} = mu + Phi X_t + v_{t+1},        t = 1..T-1,
    innovations v-hat (T-1 x K),  Sigma = v'v / (T-2)  (v has exact zero mean
    because the VAR includes a constant, so this is the ddof=1 covariance the
    NY Fed's own code uses via cov()).

2.  Excess returns.  Per-period log price p_t(n) = -n y_t(n)/ppy; short rate
    r_t = y_t(1)/ppy.  For every maturity n >= 2 with n-1 in the grid,
        rx_{t+1}(n) = p_{t+1}(n-1) - p_t(n) - r_t
    (buy an n-period bond at t, sell it at t+1 as an (n-1)-period bond).
    Stacked regression on a constant, the LAGGED factors, and the
    CONTEMPORANEOUS innovations:
        rx_{t+1} = a + c' X_t + beta' v_{t+1} + e_{t+1},
    with a (N), c (N x K), beta (N x K); sigma^2 = sum(e^2)/(N (T-1))
    (the paper's tr(EE')/NT).

3.  Prices of risk.  B* (N x K^2) has rows vec(beta_i beta_i')' =
    kron(beta_i, beta_i);  the convexity-adjusted cross-sectional regressions
        lambda0 = (beta'beta)^{-1} beta' (a + 1/2 (B* vec(Sigma) + sigma^2 1_N)),
        lambda1 = (beta'beta)^{-1} beta' c.

Short rate: r_t = delta0 + delta1' X_t by OLS over all T dates.

Affine recursions (exactly the NY Fed production code's form; note the
one-period bond is priced without error, so the seed is A_1/B_1, and the
convexity term enters only from n = 2):
    A_1 = -delta0,  B_1 = -delta1,
    A_n = A_{n-1} + B_{n-1}'(mu - lambda0)
                  + 1/2 (B_{n-1}' Sigma B_{n-1} + sigma^2) - delta0,
    B_n = (Phi - lambda1)' B_{n-1} - delta1.
Risk-neutral: the same recursion with lambda0 = 0, lambda1 = 0 (convexity
terms kept).  Fitted / risk-neutral yields (annualized decimal):
    y-hat_t(n) = -(A_n + B_n' X_t) * ppy / n,
term premium = fitted - risk-neutral.

Two legs
--------
- "sim": a simulated affine DGP with KNOWN prices of risk — factors follow
  the VAR(1), yields are the exact affine prices plus 5bp iid measurement
  noise. Golden values pin every pipeline output; the stored true term
  premium documents recovery (an MC block reports corr/MAE over 30 seeds).
- "gsw": a REAL yield panel — zero-coupon yields computed from the vendored
  monthly Gurkaynak-Sack-Wright NSS parameters (fixtures/gsw_nss_params.csv,
  1961-06..2014-04, Federal Reserve Board data; see that file's header for
  provenance).  Golden values pin the pipeline; the NY Fed's PUBLISHED ACM
  10-year decomposition (fixtures/acm_published_10y.csv, quarterly) is
  stored alongside for a level/shape comparison — a validation target, not a
  bit-exact golden (the published series is estimated on the Fed's own
  FFR-spliced curve inputs and re-estimated as the sample grows).

Run with the project venv:
    .venv/bin/python fixtures/generate_acm_fixtures.py
"""
import csv
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a, dtype=float).ravel()]
mat2 = lambda a: [[float(x) for x in row] for row in np.asarray(a, dtype=float)]


def acm_pipeline(yields, maturities, k, ppy):
    """The ACM (2013) three-step pipeline, transcribed as documented above.

    Never calls tsecon. Returns every quantity the Rust crate exposes.
    """
    Y = np.asarray(yields, dtype=float)
    mats = np.asarray(maturities, dtype=int)
    T, M = Y.shape

    # Factors: PCs of the demeaned panel, unit sample std, mean-positive loadings.
    Yd = Y - Y.mean(axis=0)
    _, s, vt = np.linalg.svd(Yd, full_matrices=False)
    W = vt[:k].T
    F = Yd @ W
    sd = F.std(axis=0, ddof=1)
    F, W = F / sd, W / sd
    sign = np.where(W.sum(axis=0) < 0.0, -1.0, 1.0)
    F, W = F * sign, W * sign

    # Step 1: VAR(1) with intercept.
    X_lhs = F[1:]
    Z_var = np.column_stack([np.ones(T - 1), F[:-1]])
    var_coef, *_ = np.linalg.lstsq(Z_var, X_lhs, rcond=None)
    mu = var_coef[0]
    phi = var_coef[1:].T
    V = X_lhs - Z_var @ var_coef
    Tv = T - 1
    Sigma = V.T @ V / (Tv - 1)
    var_r2 = 1.0 - (V ** 2).sum(axis=0) / ((X_lhs - X_lhs.mean(axis=0)) ** 2).sum(axis=0)

    # Excess returns.
    per_period = Y / ppy
    P = -per_period * mats
    col = {int(n): j for j, n in enumerate(mats)}
    r = per_period[:, col[1]]
    rx_mats = [int(n) for n in mats if n >= 2 and (n - 1) in col]
    N = len(rx_mats)
    RX = np.empty((Tv, N))
    for j, n in enumerate(rx_mats):
        RX[:, j] = P[1:, col[n - 1]] - P[:-1, col[n]] - r[:-1]

    # Step 2: rx on [1, X_{t-1}, v_t].
    Z = np.column_stack([np.ones(Tv), F[:-1], V])
    abc, *_ = np.linalg.lstsq(Z, RX, rcond=None)
    E = RX - Z @ abc
    a = abc[0]
    c = abc[1:1 + k].T
    beta = abc[1 + k:].T
    sigma2 = (E ** 2).sum() / (N * Tv)
    rx_r2 = 1.0 - (E ** 2).sum(axis=0) / ((RX - RX.mean(axis=0)) ** 2).sum(axis=0)

    # Step 3: prices of risk.
    bstar = np.stack([np.kron(beta[i], beta[i]) for i in range(N)])
    a_adj = a + 0.5 * (bstar @ Sigma.reshape(-1) + sigma2)
    lam, *_ = np.linalg.lstsq(beta, np.column_stack([a_adj, c]), rcond=None)
    lambda0, lambda1 = lam[:, 0], lam[:, 1:]

    # Short-rate equation.
    Zs = np.column_stack([np.ones(T), F])
    dl, *_ = np.linalg.lstsq(Zs, r, rcond=None)
    delta0, delta1 = dl[0], dl[1:]
    es = r - Zs @ dl
    sr_r2 = 1.0 - (es ** 2).sum() / ((r - r.mean()) ** 2).sum()

    # Affine recursions.
    nmax = int(mats[-1])

    def recur(l0, l1):
        A = np.zeros(nmax + 1)
        B = np.zeros((nmax + 1, k))
        A[1], B[1] = -delta0, -delta1
        for n in range(2, nmax + 1):
            A[n] = (A[n - 1] + B[n - 1] @ (mu - l0)
                    + 0.5 * (B[n - 1] @ Sigma @ B[n - 1] + sigma2) - delta0)
            B[n] = (phi - l1).T @ B[n - 1] - delta1
        return A[1:], B[1:]

    A, B = recur(lambda0, lambda1)
    Arn, Brn = recur(np.zeros(k), np.zeros((k, k)))

    fitted = np.empty((T, M))
    rn = np.empty((T, M))
    for j, n in enumerate(mats):
        fitted[:, j] = -(A[n - 1] + F @ B[n - 1]) * ppy / n
        rn[:, j] = -(Arn[n - 1] + F @ Brn[n - 1]) * ppy / n
    tp = fitted - rn
    yield_r2 = 1.0 - ((Y - fitted) ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)

    return dict(factors=F, loadings=W, mu=mu, phi=phi, sigma=Sigma,
                rx_maturities=rx_mats, a=a, beta=beta, c=c, sigma2=sigma2,
                lambda0=lambda0, lambda1=lambda1, delta0=delta0, delta1=delta1,
                A=A, B=B, A_rn=Arn, B_rn=Brn,
                fitted=fitted, risk_neutral=rn, term_premium=tp,
                var_rsquared=var_r2, rx_rsquared=rx_r2,
                short_rate_rsquared=sr_r2, yield_rsquared=yield_r2)


# ---------------------------------------------------------------------------
# Simulated affine DGP with known prices of risk (the recovery leg).
# ---------------------------------------------------------------------------
SIM_K, SIM_T, SIM_NMAX, PPY = 3, 240, 60, 12.0
SIM_MU = np.array([0.02, -0.01, 0.005])
SIM_PHI = np.array([[0.97, 0.02, 0.00],
                    [0.00, 0.90, 0.03],
                    [0.01, 0.00, 0.80]])
SIM_CHOL = np.array([[0.16, 0.00, 0.00],
                     [0.02, 0.11, 0.00],
                     [-0.01, 0.01, 0.09]])
SIM_SIGMA = SIM_CHOL @ SIM_CHOL.T
SIM_DELTA0 = 0.003
SIM_DELTA1 = np.array([0.0011, 0.0006, 0.0004])
SIM_LAMBDA0 = np.array([-0.12, 0.08, -0.05])
SIM_LAMBDA1 = np.array([[-0.020, 0.015, -0.010],
                        [0.012, -0.018, 0.008],
                        [-0.006, 0.010, -0.015]])
SIM_NOISE_SD = 0.0005  # 5bp iid measurement noise on annualized yields


def true_recursions(l0, l1):
    """The DGP's exact recursions (no sigma^2: the DGP prices without error)."""
    A = np.zeros(SIM_NMAX + 1)
    B = np.zeros((SIM_NMAX + 1, SIM_K))
    A[1], B[1] = -SIM_DELTA0, -SIM_DELTA1
    for n in range(2, SIM_NMAX + 1):
        A[n] = (A[n - 1] + B[n - 1] @ (SIM_MU - l0)
                + 0.5 * B[n - 1] @ SIM_SIGMA @ B[n - 1] - SIM_DELTA0)
        B[n] = (SIM_PHI - l1).T @ B[n - 1] - SIM_DELTA1
    return A[1:], B[1:]


def simulate(seed):
    """One draw of the affine DGP: (observed yields, true yields, true TP)."""
    At, Bt = true_recursions(SIM_LAMBDA0, SIM_LAMBDA1)
    Arn, Brn = true_recursions(np.zeros(SIM_K), np.zeros((SIM_K, SIM_K)))
    rng = np.random.default_rng(seed)
    X = np.empty((SIM_T, SIM_K))
    x = np.linalg.solve(np.eye(SIM_K) - SIM_PHI, SIM_MU)
    for _ in range(200):
        x = SIM_MU + SIM_PHI @ x + SIM_CHOL @ rng.standard_normal(SIM_K)
    for t in range(SIM_T):
        x = SIM_MU + SIM_PHI @ x + SIM_CHOL @ rng.standard_normal(SIM_K)
        X[t] = x
    mats = np.arange(1, SIM_NMAX + 1)
    yt = np.empty((SIM_T, SIM_NMAX))
    yr = np.empty((SIM_T, SIM_NMAX))
    for j, n in enumerate(mats):
        yt[:, j] = -(At[n - 1] + X @ Bt[n - 1]) * PPY / n
        yr[:, j] = -(Arn[n - 1] + X @ Brn[n - 1]) * PPY / n
    y_obs = yt + SIM_NOISE_SD * rng.standard_normal(yt.shape)
    return y_obs, yt, yt - yr


def golden_block(res, mats, report_mats):
    """The pipeline outputs stored for the Rust/Python golden tests."""
    idx = {int(n): j for j, n in enumerate(mats)}
    g = {
        "factors": mat2(res["factors"]),
        "loadings": mat2(res["loadings"]),
        "mu": full(res["mu"]),
        "phi": mat2(res["phi"]),
        "sigma": mat2(res["sigma"]),
        "rx_maturities": [int(n) for n in res["rx_maturities"]],
        "a": full(res["a"]),
        "beta": mat2(res["beta"]),
        "c": mat2(res["c"]),
        "sigma2": float(res["sigma2"]),
        "lambda0": full(res["lambda0"]),
        "lambda1": mat2(res["lambda1"]),
        "delta0": float(res["delta0"]),
        "delta1": full(res["delta1"]),
        "A": full(res["A"]),
        "B": mat2(res["B"]),
        "A_rn": full(res["A_rn"]),
        "B_rn": mat2(res["B_rn"]),
        "var_rsquared": full(res["var_rsquared"]),
        "rx_rsquared": full(res["rx_rsquared"]),
        "short_rate_rsquared": float(res["short_rate_rsquared"]),
        "yield_rsquared": full(res["yield_rsquared"]),
        "fitted_row0": full(res["fitted"][0]),
        "fitted_row_last": full(res["fitted"][-1]),
        "term_premium_row0": full(res["term_premium"][0]),
        "term_premium_row_last": full(res["term_premium"][-1]),
    }
    for n in report_mats:
        j = idx[n]
        g[f"fitted_{n}"] = full(res["fitted"][:, j])
        g[f"risk_neutral_{n}"] = full(res["risk_neutral"][:, j])
        g[f"term_premium_{n}"] = full(res["term_premium"][:, j])
    return g


def gen_sim():
    mats = np.arange(1, SIM_NMAX + 1)
    y_obs, _, tp_true = simulate(20260819)
    res = acm_pipeline(y_obs, mats, SIM_K, PPY)

    # Monte-Carlo recovery evidence: the pipeline's term premium tracks the
    # DGP's true premium across independent samples.
    corr60, mae60, corr36 = [], [], []
    for rep in range(30):
        yo, _, tpt = simulate(1000 + rep)
        rr = acm_pipeline(yo, mats, SIM_K, PPY)
        e60, t60 = rr["term_premium"][:, 59], tpt[:, 59]
        e36, t36 = rr["term_premium"][:, 35], tpt[:, 35]
        corr60.append(np.corrcoef(e60, t60)[0, 1])
        corr36.append(np.corrcoef(e36, t36)[0, 1])
        mae60.append(np.abs(e60 - t60).mean() * 1e4)
    recovery = {
        "n_rep": 30,
        "tp60_corr_mean": float(np.mean(corr60)),
        "tp60_corr_min": float(np.min(corr60)),
        "tp36_corr_mean": float(np.mean(corr36)),
        "tp60_mae_bp_mean": float(np.mean(mae60)),
        "tp60_mae_bp_max": float(np.max(mae60)),
        "tp60_true_mean_bp": float(tp_true[:, 59].mean() * 1e4),
    }

    return {
        "periods_per_year": PPY,
        "n_factors": SIM_K,
        "maturities": [int(n) for n in mats],
        "yields": mat2(y_obs),
        "true": {
            "mu": full(SIM_MU), "phi": mat2(SIM_PHI), "sigma": mat2(SIM_SIGMA),
            "delta0": SIM_DELTA0, "delta1": full(SIM_DELTA1),
            "lambda0": full(SIM_LAMBDA0), "lambda1": mat2(SIM_LAMBDA1),
            "noise_sd": SIM_NOISE_SD,
            "term_premium_12": full(tp_true[:, 11]),
            "term_premium_36": full(tp_true[:, 35]),
            "term_premium_60": full(tp_true[:, 59]),
        },
        "golden": golden_block(res, mats, (12, 36, 60)),
        "recovery": recovery,
    }


# ---------------------------------------------------------------------------
# The real-data leg: GSW zero-coupon yields, 1961-06..2014-04.
# ---------------------------------------------------------------------------
GSW_MATS = np.array([1, 2, 5, 6, 11, 12, 23, 24, 35, 36, 47, 48, 59, 60,
                     71, 72, 83, 84, 95, 96, 107, 108, 119, 120])
# The pairs (n-1, n) around ACM's own excess-return maturities {6,12,...,120}
# plus maturities 1 and 2, so rx_maturities = {2, 6, 12, 24, ..., 120}.


def read_csv_comments(path):
    with open(path) as f:
        rows = [row for row in csv.reader(f) if not row[0].startswith("#")]
    header, data = rows[0], rows[1:]
    return header, data


def gen_gsw():
    header, data = read_csv_comments(OUT / "gsw_nss_params.csv")
    cols = {name: i for i, name in enumerate(header)}
    dates = [row[cols["DATE"]] for row in data]
    par = {name: np.array([float(row[cols[name]]) for row in data])
           for name in ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")}

    # NSS zero-coupon yields (percent -> decimal). Rows with TAU2 = 0 carry
    # BETA3 = 0 (plain Nelson-Siegel dates): the Svensson term is zero there.
    n_years = GSW_MATS / 12.0
    t1 = n_years[None, :] / par["TAU1"][:, None]
    g1 = (1.0 - np.exp(-t1)) / t1
    tau2 = par["TAU2"][:, None]
    safe_tau2 = np.where(tau2 > 0.0, tau2, 1.0)
    t2 = n_years[None, :] / safe_tau2
    sv = np.where(tau2 > 0.0, (1.0 - np.exp(-t2)) / t2 - np.exp(-t2), 0.0)
    Y = (par["BETA0"][:, None] + par["BETA1"][:, None] * g1
         + par["BETA2"][:, None] * (g1 - np.exp(-t1))
         + par["BETA3"][:, None] * sv) / 100.0

    res = acm_pipeline(Y, GSW_MATS, 5, PPY)

    # Published-series comparison: the NY Fed's ACM 10-year decomposition
    # (quarterly). Align quarter-end months to our monthly rows.
    phdr, pdata = read_csv_comments(OUT / "acm_published_10y.csv")
    pcols = {name: i for i, name in enumerate(phdr)}
    month_of = {d[:7]: i for i, d in enumerate(dates)}
    rows_idx, pub_tp10, pub_y10 = [], [], []
    for row in pdata:
        m = row[pcols["DATE"]][:7]
        if m in month_of:
            rows_idx.append(month_of[m])
            pub_tp10.append(float(row[pcols["ACMTP10"]]))
            pub_y10.append(float(row[pcols["ACMY10"]]))
    rows_idx = np.array(rows_idx)
    pub_tp10 = np.array(pub_tp10)
    pub_y10 = np.array(pub_y10)
    j10 = list(GSW_MATS).index(120)
    ours_tp = res["term_premium"][rows_idx, j10] * 100.0
    ours_y = res["fitted"][rows_idx, j10] * 100.0
    stats = {
        "n_quarters": int(len(rows_idx)),
        "tp10_corr": float(np.corrcoef(ours_tp, pub_tp10)[0, 1]),
        "tp10_mean_gap_pp": float((ours_tp - pub_tp10).mean()),
        "tp10_rmse_pp": float(np.sqrt(((ours_tp - pub_tp10) ** 2).mean())),
        "y10_corr": float(np.corrcoef(ours_y, pub_y10)[0, 1]),
        "y10_rmse_pp": float(np.sqrt(((ours_y - pub_y10) ** 2).mean())),
    }
    print("  gsw vs published ACM 10y:", json.dumps(stats))

    return {
        "periods_per_year": PPY,
        "n_factors": 5,
        "maturities": [int(n) for n in GSW_MATS],
        "dates": dates,
        "yields": mat2(Y),
        "golden": golden_block(res, GSW_MATS, (12, 60, 120)),
        "published": {
            "source": "FRBNY ACM term premia (2021 vintage via the Brookings "
                      "mirror); see fixtures/acm_published_10y.csv.",
            "quarter_row_idx": [int(i) for i in rows_idx],
            "acmtp10": full(pub_tp10),
            "acmy10": full(pub_y10),
            "stats": stats,
        },
    }


def main():
    sim = gen_sim()
    print("  sim recovery:", json.dumps(sim["recovery"]))
    gsw = gen_gsw()
    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "reference": "Adrian, Crump & Moench (2013), J. Financial Economics "
                         "110(1), 110-138 (FRBNY Staff Report 340): the three-step "
                         "regression-based Gaussian affine term-structure estimator.",
            "note": "DOCUMENTED-FORMULA golden (non-circular; no tsecon call): the "
                    "ENTIRE pipeline — PCA factors, VAR(1), excess-return "
                    "regressions, lambda0/lambda1 recovery, affine recursions, "
                    "fitted/risk-neutral yields, term premium — is built "
                    "independently in NumPy. 'sim' is an affine DGP with known "
                    "prices of risk (recovery evidence in sim.recovery); 'gsw' is "
                    "the real GSW yield panel with the NY Fed's published ACM 10y "
                    "series as a level/shape comparison (gsw.published.stats). "
                    "Yields are annualized continuously-compounded DECIMALS; "
                    "maturities are integer periods (months).",
        },
        "sim": sim,
        "gsw": gsw,
    }
    (OUT / "acm.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote acm.json")


if __name__ == "__main__":
    main()
