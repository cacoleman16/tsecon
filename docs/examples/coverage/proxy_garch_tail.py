"""Interval COVERAGE for proxy-SVAR inference, GARCH, growth-at-risk and
functional local projections -- the families the original audit left
unmeasured.

    .venv/bin/python docs/examples/coverage/proxy_garch_tail.py [--quick]

Seven surfaces, split by the same honest question the factor_midas module
asks first: DOES THIS FUNCTION SHIP AN INTERVAL AT ALL?

  proxy_svar_bands  ships the Jentsch-Lunsford moving-block bootstrap bands
                    (Hall and Efron endpoints) plus the deliberately invalid
                    wild reproduction arm. Experiment 1 measures all three.
  proxy_ar_sets     ships weak-IV-robust Anderson-Rubin SETS with the
                    reduced-form uncertainty propagated by rf_method="delta"
                    (default), the opt-in "second_order", or the opt-in
                    "second_order_bc" (the same simulation centred at Pope-
                    bias-corrected coefficients -- the residual-gap follow-up
                    of roadmap note 21). Experiment 2 measures all three,
                    paired, on the card's VAR(2) DGP and on the routine
                    VAR(1) that note 21 used.
  growth_at_risk    ships two Powell sandwiches: `bse` (Newey-West corrected
                    at horizon-1 lags, the default story) and `bse_powell`
                    (uncorrected, kept for statsmodels replication).
                    Experiment 3 measures both on an exact-truth design.
  garch_fit         ships MLE and Bollerslev-Wooldridge robust parameter
                    standard errors. Experiment 4 measures both under
                    Gaussian and Student-t(5) innovations. Its
                    `variance_forecast` is a documented POINT path -- "it
                    carries no interval or coverage level, and none is
                    implied" -- and experiment 6 verifies that no interval
                    key exists next to it.
  flp               ships per-horizon HAC `se` that CONDITIONS ON THE SCORES
                    -- the card warns the per-element se is inconsistent for
                    functional_pca scores. Experiment 5 prices that warning
                    against the exempt cases (external scores;
                    flp_scenario's w'beta contrasts).
  flp_scenario      the documented reporting route; measured in the same
                    experiment.
  nongaussian_svar  ships NO interval-like output at all (point B, IRF,
                    kurtosis diagnostics). Experiment 6 verifies the key set
                    every run, so a future `se` breaks an assertion instead
                    of silently outdating the page.

--------------------------------------------------------------------------
The proxy DGPs, and why their truth is closed-form
--------------------------------------------------------------------------
The card VAR(2) is the 3-variable system of
fixtures/generate_proxy_ar_fixtures.py (the configuration behind
proxy_ar_sets' published coverage table), T = 300, spectral radius 0.68;
the routine VAR(1) is roadmap note 21's second DGP, T = 250, radius ~0.70.
Both draw structural shocks eps ~ N(0, I), u = H eps, and a proxy
m = phi*eps_0 + sig_nu*nu for shock 0 (phi = 1.0, sig_nu = 1.5 -- strong).
The unit-normalized impulse response of variable i at horizon h is exactly

    lambda(h, i) = (Psi_h H[:, 0])_i / H[norm_var, 0]

with Psi_h the MA matrices of the true coefficients -- no approximation, so
any miss belongs to the interval. The (norm_var, h=0) cell is degenerate at
`unit` by construction and excluded from every claim (it is asserted
degenerate instead).

--------------------------------------------------------------------------
The growth-at-risk DGP
--------------------------------------------------------------------------
    x_t = phi x_{t-1} + v_t                (the conditioner, phi = 0.85)
    y_t = rho y_{t-1} + beta x_{t-1} + e_t (the outcome, rho = 0.5)

(y_t, x_t) is a Gaussian VAR(1) state with companion A = [[rho, beta],
[0, phi]], so y_{t+h} | (x_t, y_t) is exactly Gaussian-linear and the
tau-quantile regression of y_{t+h} on [const, x_t, y_t] is CORRECTLY
SPECIFIED with closed-form truth:

    slope on x_t  = [A^h]_{0,1}      slope on y_t = [A^h]_{0,0}
    intercept     = z_tau * sigma_h,  sigma_h^2 = sum_{j<h} ([A^j]_{0,0}^2
                                                  + [A^j]_{0,1}^2)

The overlapping windows make the check-loss score an MA(h-1) even though
the model is exact -- precisely the structure `bse`'s Newey-West correction
targets and `bse_powell` ignores. The measured cell is the slope on x (the
"conditions" column ABG's asymmetry finding is read from).

--------------------------------------------------------------------------
The GARCH DGP
--------------------------------------------------------------------------
GARCH(1,1) with (omega, alpha, beta) = (0.05, 0.10, 0.85) -- unconditional
variance 1, persistence 0.95 -- and z_t either standard normal or
Student-t(5) scaled to unit variance. The t(5) arm fitted with
dist="normal" is exactly the QMLE case the Bollerslev-Wooldridge `se_robust`
exists for; `se_mle` is the inverse Hessian that is only valid when the
innovation distribution is correct. Replications where any parameter sits
on a boundary or `se_valid` is False are excluded and counted (the library
itself marks them NaN rather than inventing a standard error).

--------------------------------------------------------------------------
The functional-shocks DGPs
--------------------------------------------------------------------------
Curves on an M = 8 grid spanned by two exactly orthonormal shapes,
X_t = s1_t phi1 + s2_t phi2 (+ 0.05 grid noise in the persistent design).

  iid design         s1, s2 i.i.d. Gaussian; y_t = sum_j theta1_j s1_{t-j}
                     + theta2_j s2_{t-j} + noise. Because the scores are
                     i.i.d. impulses, the population LP coefficient at
                     horizon h is EXACTLY theta_h for any lag controls (the
                     lp_family identification trick).
  persistent design  s1, s2 AR(1) at 0.9 / 0.7; y_t = t1 s1_t + t2 s2_t +
                     noise, fitted with n_lag_controls = 0 so the
                     population coefficient at horizon h is exactly
                     tk * rho_k^h. This is the shape of the card's own
                     yield-curve example, where the warned se collapse is
                     dramatic.

Each replication runs flp on the functional_pca scores AND on the true
scores (the documented exempt case) on the same draw, plus flp_scenario for
a fixed in-span scenario delta = 1.2 phi1 - 0.5 phi2, whose truth is
1.2 * beta1_h - 0.5 * beta2_h by the eigenfunction-projection identity.
Estimated scores are sign-aligned to the truth per replication (the sign is
a documented convention, not an estimate).
"""
import argparse
import re
import time

import numpy as np
from scipy.stats import chi2, norm

import tsecon

# --------------------------------------------------------------------------
# global configuration
# --------------------------------------------------------------------------
SEED = 20260729
Z95 = float(norm.ppf(0.975))
NOMINAL = 0.95                 # proxy_ar_sets, growth_at_risk, garch, flp
NOMINAL_BANDS = 0.90           # proxy_svar_bands' alpha=0.10 convention

# ---- proxy DGPs (the card VAR(2) and roadmap note 21's routine VAR(1)) ----
CARD_H = np.array([[1.0, 0.4, 0.2],
                   [0.5, 1.2, 0.3],
                   [0.3, 0.5, 0.9]])
CARD_A = [np.array([[0.50, 0.10, 0.00],
                    [0.00, 0.40, 0.10],
                    [0.10, 0.00, 0.30]]),
          np.diag([0.10, 0.10, 0.10])]
ROUTINE_H = np.array([[1.0, 0.3, 0.1],
                      [0.4, 1.1, 0.2],
                      [0.2, 0.4, 0.8]])
ROUTINE_A = [np.array([[0.65, 0.15, 0.00],
                       [0.00, 0.55, 0.15],
                       [0.10, 0.00, 0.45]])]
PROXY_DGPS = {
    "card_var2": {"H": CARD_H, "A": CARD_A, "T": 300, "lags": 2},
    "routine_var1": {"H": ROUTINE_H, "A": ROUTINE_A, "T": 250, "lags": 1},
}
PHI_PROXY, SIG_NU, NORM_VAR, UNIT = 1.0, 1.5, 0, 1.0
H_PROXY = 12

# ---- growth-at-risk DGP ---------------------------------------------------
GAR_RHO, GAR_BETA, GAR_PHI = 0.5, 0.5, 0.85
GAR_A = np.array([[GAR_RHO, GAR_BETA], [0.0, GAR_PHI]])
GAR_T = 240
GAR_TAUS = [0.05, 0.5]
GAR_HS = [1, 4, 8, 12]

# ---- GARCH DGP ------------------------------------------------------------
G_OMEGA, G_ALPHA, G_BETA = 0.05, 0.10, 0.85
GARCH_TRUTH = np.array([G_OMEGA, G_ALPHA, G_BETA])
GARCH_NAMES = ["omega", "alpha[1]", "beta[1]"]

# ---- functional-shocks DGPs -----------------------------------------------
FLP_M, FLP_H = 8, 8
FLP_PHI1 = np.ones(FLP_M) / np.sqrt(FLP_M)
FLP_PHI2 = np.linspace(-1.0, 1.0, FLP_M)
FLP_PHI2 -= FLP_PHI2.mean()
FLP_PHI2 /= np.linalg.norm(FLP_PHI2)
# iid design
FLP_J = 25
FLP_TH1 = 1.0 * 0.75 ** np.arange(FLP_J)
FLP_TH2 = -0.6 * 0.55 ** np.arange(FLP_J)
# persistent design
FLP_R1, FLP_R2 = 0.9, 0.7
FLP_SD1, FLP_SD2 = 1.0, 0.6
FLP_T1, FLP_T2 = -0.4, 0.5
FLP_NOISE = 0.05
FLP_W1, FLP_W2 = 1.2, -0.5
FLP_DELTA = FLP_W1 * FLP_PHI1 + FLP_W2 * FLP_PHI2

# same interval-key tripwire as factor_midas.py
INTERVAL_KEY = re.compile(
    r"(?:^|_)(?:b?se|std(?:err)?|lower|upper|conf|ci|band|bounds?"
    r"|quantiles?|interval)(?:$|_)",
    re.IGNORECASE)


def _rng(experiment, rep):
    """A reproducible, independent stream per (experiment, replication)."""
    return np.random.default_rng([SEED, experiment, rep])


# --------------------------------------------------------------------------
# data-generating processes
# --------------------------------------------------------------------------
def dgp_proxy(rng, spec, phi=PHI_PROXY, sig_nu=SIG_NU, burn=500):
    """VAR(p) y_t = sum A_i y_{t-i} + H eps_t with a proxy for shock 0."""
    coefs, h_mat, T = spec["A"], spec["H"], spec["T"]
    p, n = len(coefs), h_mat.shape[0]
    total = T + burn
    eps = rng.standard_normal((total, n))
    u = eps @ h_mat.T
    y = np.zeros((total, n))
    for t in range(p, total):
        acc = u[t].copy()
        for i, a in enumerate(coefs):
            acc += a @ y[t - 1 - i]
        y[t] = acc
    m = phi * eps[burn:, 0] + sig_nu * rng.standard_normal(T)
    return y[burn:], m


def ma_matrices(coefs, horizon):
    """Psi_0 = I, Psi_h = sum_{i<=min(h,p)} Psi_{h-i} A_i."""
    p, n = len(coefs), coefs[0].shape[0]
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc += psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return np.array(psi)


def proxy_truth(spec, horizon=H_PROXY):
    """lambda(h, i): the unit-normalized response, exact at every cell."""
    psi = ma_matrices(spec["A"], horizon)
    hcol = spec["H"][:, 0]
    return np.array([UNIT * (psi[h] @ hcol) / hcol[NORM_VAR]
                     for h in range(horizon + 1)])


def dgp_gar(rng, T=GAR_T):
    v = rng.standard_normal(T)
    e = rng.standard_normal(T)
    x = np.zeros(T)
    y = np.zeros(T)
    for t in range(1, T):
        x[t] = GAR_PHI * x[t - 1] + v[t]
        y[t] = GAR_RHO * y[t - 1] + GAR_BETA * x[t - 1] + e[t]
    return y, x


def gar_truth(h, taus=GAR_TAUS):
    """Per-tau [const, x-slope, y-slope]; exact (see module docstring)."""
    ah = np.linalg.matrix_power(GAR_A, h)
    var = 0.0
    for j in range(h):
        aj = np.linalg.matrix_power(GAR_A, j)
        var += aj[0, 0] ** 2 + aj[0, 1] ** 2
    sig = np.sqrt(var)
    return np.array([[sig * norm.ppf(tau), ah[0, 1], ah[0, 0]]
                     for tau in taus])


def dgp_garch(rng, T, dist, burn=500):
    if dist == "normal":
        z = rng.standard_normal(T + burn)
    else:                              # standardized Student-t(5)
        z = rng.standard_t(5, size=T + burn) / np.sqrt(5.0 / 3.0)
    y = np.empty(T + burn)
    s2 = 1.0
    for t in range(T + burn):
        if t > 0:
            s2 = G_OMEGA + G_ALPHA * y[t - 1] ** 2 + G_BETA * s2
        y[t] = np.sqrt(s2) * z[t]
    return y[burn:]


def dgp_flp_iid(rng, T):
    s1 = 2.0 * rng.standard_normal(T + FLP_J)
    s2 = 1.0 * rng.standard_normal(T + FLP_J)
    y = np.zeros(T)
    for j in range(FLP_J):
        y += (FLP_TH1[j] * s1[FLP_J - j:T + FLP_J - j]
              + FLP_TH2[j] * s2[FLP_J - j:T + FLP_J - j])
    y += 0.8 * rng.standard_normal(T)
    s1, s2 = s1[FLP_J:], s2[FLP_J:]
    curves = np.outer(s1, FLP_PHI1) + np.outer(s2, FLP_PHI2)
    return y, curves, s1, s2


def dgp_flp_persistent(rng, T):
    e1 = rng.standard_normal(T)
    e2 = rng.standard_normal(T)
    s1 = np.zeros(T)
    s2 = np.zeros(T)
    for t in range(1, T):
        s1[t] = FLP_R1 * s1[t - 1] + FLP_SD1 * np.sqrt(1 - FLP_R1 ** 2) * e1[t]
        s2[t] = FLP_R2 * s2[t - 1] + FLP_SD2 * np.sqrt(1 - FLP_R2 ** 2) * e2[t]
    y = FLP_T1 * s1 + FLP_T2 * s2 + 0.3 * rng.standard_normal(T)
    curves = (np.outer(s1, FLP_PHI1) + np.outer(s2, FLP_PHI2)
              + FLP_NOISE * rng.standard_normal((T, FLP_M)))
    return y, curves, s1, s2


# --------------------------------------------------------------------------
# coverage bookkeeping (house schema, as in factor_midas.py)
# --------------------------------------------------------------------------
def summarize(est, se, truth, arm, index_name="h"):
    est = np.asarray(est, dtype=float)
    se = np.asarray(se, dtype=float)
    truth = np.asarray(truth, dtype=float)
    rows = []
    for h in range(est.shape[1]):
        ok = np.isfinite(est[:, h]) & np.isfinite(se[:, h]) & (se[:, h] > 0.0)
        e, s = est[ok, h], se[ok, h]
        n = int(ok.sum())
        if n < 2:
            continue
        covered = np.abs(e - truth[h]) <= Z95 * s
        p = float(covered.mean())
        sd = float(e.std(ddof=1))
        bias = float(e.mean() - truth[h])
        rows.append({
            "arm": arm,
            index_name: h,
            "truth": float(truth[h]),
            "bias": bias,
            "sd_est": sd,
            "mean_se": float(s.mean()),
            "med_se": float(np.median(s)),
            "se_over_sd": float(s.mean() / sd) if sd > 0 else float("nan"),
            "absbias_over_sd": abs(bias) / sd if sd > 0 else float("nan"),
            "cov95": p,
            "mcse": float(np.sqrt(p * (1.0 - p) / n)),
            "n_used": n,
        })
    return rows


def cov_row(arm, h, hits, n, extra=None):
    """A bare containment-rate row for band/set experiments."""
    p = float(hits / n)
    row = {"arm": arm, "h": int(h), "cov95": p,
           "mcse": float(np.sqrt(p * (1.0 - p) / n)), "n_used": int(n)}
    if extra:
        row.update(extra)
    return row


SPEC = [
    ("arm", "arm", 30, "{:s}"),
    ("h", "h", 3, "{:d}"),
    ("truth", "truth", 8, "{:.4f}"),
    ("bias", "bias", 9, "{:+.4g}"),
    ("sd_est", "sd_est", 9, "{:.4g}"),
    ("mean_se", "mean_se", 9, "{:.4g}"),
    ("se_over_sd", "se/sd", 6, "{:.2f}"),
    ("absbias_over_sd", "|b|/sd", 7, "{:.2f}"),
    ("cov95", "cov", 7, "{:.3f}"),
    ("mcse", "mcse", 6, "{:.3f}"),
]


def print_table(rows, spec=SPEC):
    head = "  ".join(f"{hdr:>{w}}" for _, hdr, w, _ in spec)
    print(head)
    print("-" * len(head))
    last = None
    for r in rows:
        if last is not None and r["arm"] != last:
            print()
        last = r["arm"]
        print("  ".join(
            f"{(fmt.format(r[k]) if k in r else '.'):>{w}}"
            for k, _, w, fmt in spec))


def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def set_contains(cell, lam):
    """Containment under proxy_ar_sets' documented kind semantics."""
    kind = cell["kind"]
    if kind == "interval":
        return cell["lower"] <= lam <= cell["upper"]
    if kind == "point":
        return lam == cell["lower"]
    if kind == "ray_below":
        return lam <= cell["upper"]
    if kind == "ray_above":
        return lam >= cell["lower"]
    if kind == "exterior":
        return (lam <= cell["excluded_lower"]
                or lam >= cell["excluded_upper"])
    return kind == "whole"


# ==========================================================================
# Experiment 1 -- proxy_svar_bands: moving-block Hall / Efron, and the wild
# reproduction arm
# ==========================================================================
def exp_proxy_bands(reps_mbb, reps_wild, n_boot, experiment=1):
    spec = PROXY_DGPS["card_var2"]
    truth = proxy_truth(spec)
    n = spec["H"].shape[0]
    hz = H_PROXY

    def run_arm(reps, bands, exp_id):
        hits_hall = np.zeros((hz + 1, n))
        hits_efron = np.zeros((hz + 1, n))
        n_failed_total = 0
        valid_flags = set()
        degenerate_ok = True
        for r in range(reps):
            rng = _rng(exp_id, r)
            y, m = dgp_proxy(rng, spec)
            b = tsecon.proxy_svar_bands(
                y, m, lags=spec["lags"], horizon=hz, norm_var=NORM_VAR,
                unit=UNIT, alpha=1.0 - NOMINAL_BANDS, n_boot=n_boot,
                seed=r, bands=bands)
            lo, hi = np.asarray(b["lower"]), np.asarray(b["upper"])
            loe = np.asarray(b["lower_efron"])
            hie = np.asarray(b["upper_efron"])
            hits_hall += (lo <= truth) & (truth <= hi)
            hits_efron += (loe <= truth) & (truth <= hie)
            n_failed_total += int(b["n_failed"])
            valid_flags.add(bool(b["asymptotically_valid"]))
            degenerate_ok &= (lo[0, NORM_VAR] == UNIT == hi[0, NORM_VAR])
        return {"hall": hits_hall / reps, "efron": hits_efron / reps,
                "reps": reps, "n_failed": n_failed_total,
                "valid_flags": sorted(valid_flags),
                "degenerate_ok": bool(degenerate_ok)}

    mbb = run_arm(reps_mbb, "moving_block", experiment * 10)
    wild = run_arm(reps_wild, "wild", experiment * 10 + 1)

    rows = []
    for tag, res, endpoints in (("mbb", mbb, ("hall", "efron")),
                                ("wild", wild, ("hall",))):
        for ep in endpoints:
            cov = res[ep]
            for i in range(n):
                for h in range(hz + 1):
                    if h == 0 and i == NORM_VAR:
                        continue          # degenerate by construction
                    rows.append(cov_row(f"{tag} {ep} var{i}", h,
                                        cov[h, i] * res["reps"],
                                        res["reps"]))
    return {
        "name": "proxy_svar_bands: moving-block vs wild, Hall vs Efron",
        "meta": {"dgp": "card_var2", "T": spec["T"], "lags": spec["lags"],
                 "horizon": hz, "n_boot": n_boot, "nominal": NOMINAL_BANDS,
                 "reps_mbb": reps_mbb, "reps_wild": reps_wild,
                 "phi": PHI_PROXY, "sig_nu": SIG_NU},
        "rows": rows,
        "mbb": {"n_failed": mbb["n_failed"], "valid_flags": mbb["valid_flags"],
                "degenerate_ok": mbb["degenerate_ok"]},
        "wild": {"n_failed": wild["n_failed"],
                 "valid_flags": wild["valid_flags"],
                 "degenerate_ok": wild["degenerate_ok"]},
    }


# ==========================================================================
# Experiment 2 -- proxy_ar_sets: rf_method="delta" vs "second_order", paired
# ==========================================================================
AR_METHODS = ("delta", "second_order", "second_order_bc")


def exp_proxy_ar(reps, experiment=2):
    hz = H_PROXY
    out_rows = []
    detail = {}
    for d_idx, (dname, spec) in enumerate(PROXY_DGPS.items()):
        truth = proxy_truth(spec)
        n = spec["H"].shape[0]
        keep0 = [i for i in range(n) if i != NORM_VAR]
        cell_hits = {m: np.zeros((hz + 1, n)) for m in AR_METHODS}
        miss_above = {m: 0 for m in AR_METHODS}
        miss_below = {m: 0 for m in AR_METHODS}
        wratio = {m: {8: [], 12: []} for m in AR_METHODS[1:]}
        bounded = {m: 0 for m in AR_METHODS}
        total = {m: 0 for m in AR_METHODS}
        for r in range(reps):
            rng = _rng(experiment * 10 + d_idx, r)
            y, m_proxy = dgp_proxy(rng, spec)
            res = {}
            for method in AR_METHODS:
                kw = {} if method == "delta" else {"rf_method": method}
                res[method] = tsecon.proxy_ar_sets(
                    y, m_proxy, lags=spec["lags"], horizon=hz,
                    norm_var=NORM_VAR, unit=UNIT, alpha=1.0 - NOMINAL, **kw)
            for method, s in res.items():
                for h in range(hz + 1):
                    for i in range(n):
                        if h == 0 and i == NORM_VAR:
                            continue
                        cell = s["cells"][h][i]
                        total[method] += 1
                        if cell["kind"] in ("interval", "point"):
                            bounded[method] += 1
                        got = set_contains(cell, truth[h, i])
                        cell_hits[method][h, i] += got
                        if not got and h == hz:
                            if (cell["kind"] == "interval"
                                    and truth[h, i] > cell["upper"]):
                                miss_above[method] += 1
                            else:
                                miss_below[method] += 1
            for method in wratio:
                for h in wratio[method]:
                    for i in range(n):
                        cd = res["delta"]["cells"][h][i]
                        cs = res[method]["cells"][h][i]
                        if (cd["kind"] == "interval"
                                and cs["kind"] == "interval"):
                            wratio[method][h].append(
                                (cs["upper"] - cs["lower"])
                                / (cd["upper"] - cd["lower"]))
        for method in AR_METHODS:
            for h in range(hz + 1):
                cells = keep0 if h == 0 else list(range(n))
                p_hits = cell_hits[method][h, cells].sum()
                out_rows.append(cov_row(f"{dname} {method}", h, p_hits,
                                        reps * len(cells)))
        worst = {}
        for method in AR_METHODS:
            cells = cell_hits[method][hz] / reps
            worst[method] = float(min(cells))
        detail[dname] = {
            "miss_above_h12": {m: int(miss_above[m]) for m in AR_METHODS},
            "miss_below_h12": {m: int(miss_below[m]) for m in AR_METHODS},
            "worst_cell_h12": worst,
            "bounded_count": {m: bounded[m] for m in AR_METHODS},
            "bounded_share": {m: bounded[m] / max(total[m], 1)
                              for m in AR_METHODS},
            "median_width_ratio": {
                m: {h: float(np.median(w)) if w else float("nan")
                    for h, w in wratio[m].items()}
                for m in wratio},
        }
    return {
        "name": ('proxy_ar_sets: rf_method="delta" vs "second_order" vs '
                 '"second_order_bc" (paired)'),
        "meta": {"horizon": hz, "nominal": NOMINAL, "reps": reps,
                 "alpha": 1.0 - NOMINAL,
                 "critical_value": float(chi2.ppf(NOMINAL, 1))},
        "rows": out_rows,
        "detail": detail,
    }


# ==========================================================================
# Experiment 3 -- growth_at_risk: bse (Newey-West) vs bse_powell
# ==========================================================================
def exp_gar(reps, experiment=3):
    n_taus = len(GAR_TAUS)
    est = {(k, tau): np.full((reps, len(GAR_HS)), np.nan)
           for k in ("bse", "powell") for tau in GAR_TAUS}
    ses = {(k, tau): np.full((reps, len(GAR_HS)), np.nan)
           for k in ("bse", "powell") for tau in GAR_TAUS}
    max_h1_diff = 0.0
    for r in range(reps):
        rng = _rng(experiment * 10, r)
        y, x = dgp_gar(rng)
        for hi, h in enumerate(GAR_HS):
            g = tsecon.growth_at_risk(y, x.reshape(-1, 1), horizon=h,
                                      taus=GAR_TAUS)
            params = np.asarray(g["params"])[:, 1]      # the x slope
            bse = np.asarray(g["bse"])[:, 1]
            bpw = np.asarray(g["bse_powell"])[:, 1]
            if h == 1:
                max_h1_diff = max(max_h1_diff,
                                  float(np.abs(np.asarray(g["bse"])
                                               - np.asarray(g["bse_powell"])
                                               ).max()))
            for ti in range(n_taus):
                tau = GAR_TAUS[ti]
                est[("bse", tau)][r, hi] = params[ti]
                ses[("bse", tau)][r, hi] = bse[ti]
                est[("powell", tau)][r, hi] = params[ti]
                ses[("powell", tau)][r, hi] = bpw[ti]
    truth_by_h = np.array([gar_truth(h)[0, 1] for h in GAR_HS])
    rows = []
    for (k, tau) in est:
        for row in summarize(est[(k, tau)], ses[(k, tau)], truth_by_h,
                             f"{k} tau={tau:.2f}"):
            row["h"] = GAR_HS[row["h"]]   # index -> real horizon
            rows.append(row)
    rows.sort(key=lambda r: (r["arm"], r["h"]))
    return {
        "name": "growth_at_risk: bse (Newey-West) vs bse_powell, exact truth",
        "meta": {"T": GAR_T, "taus": GAR_TAUS, "horizons": GAR_HS,
                 "nominal": NOMINAL, "reps": reps,
                 "cell": "the slope on the conditioning variable"},
        "rows": rows,
        "max_abs_bse_diff_h1": max_h1_diff,
    }


# ==========================================================================
# Experiment 4 -- garch_fit: se_mle vs se_robust, Gaussian vs t(5) QMLE
# ==========================================================================
def exp_garch(reps, designs=((2000, "normal"), (500, "normal"), (2000, "t5")),
              experiment=4):
    rows = []
    excluded = {}
    for d_idx, (T, dist) in enumerate(designs):
        est = np.full((reps, 3), np.nan)
        se_m = np.full((reps, 3), np.nan)
        se_r = np.full((reps, 3), np.nan)
        n_bad = 0
        for r in range(reps):
            rng = _rng(experiment * 10 + d_idx, r)
            f = tsecon.garch_fit(dgp_garch(rng, T, dist))
            if not all(f["se_valid"]) or any(f["boundary"]):
                n_bad += 1        # the library marks these NaN; count them
                continue
            est[r] = np.asarray(f["params"])
            se_m[r] = np.asarray(f["se_mle"])
            se_r[r] = np.asarray(f["se_robust"])
        tag = f"{dist} T={T}"
        excluded[tag] = {"n_boundary_or_invalid": n_bad, "reps": reps}
        rows += summarize(est, se_m, GARCH_TRUTH, f"{tag} se_mle",
                          index_name="h")
        rows += summarize(est, se_r, GARCH_TRUTH, f"{tag} se_robust",
                          index_name="h")
    return {
        "name": "garch_fit: se_mle vs se_robust (h = parameter index: "
                "0 omega, 1 alpha, 2 beta)",
        "meta": {"designs": [list(d) for d in designs],
                 "truth": GARCH_TRUTH.tolist(), "params": GARCH_NAMES,
                 "nominal": NOMINAL, "reps": reps},
        "rows": rows,
        "excluded": excluded,
    }


# ==========================================================================
# Experiment 5 -- flp / flp_scenario: the generated-regressor warning, priced
# ==========================================================================
def exp_flp(reps, T_persist=400, T_iid=300, experiment=5):
    hz = FLP_H
    rows = []

    # ---- persistent design (the card's dramatic case) --------------------
    truth1 = FLP_T1 * FLP_R1 ** np.arange(hz + 1)
    truth2 = FLP_T2 * FLP_R2 ** np.arange(hz + 1)
    truth_scn = FLP_W1 * truth1 + FLP_W2 * truth2
    arms = {k: (np.full((reps, hz + 1), np.nan),
                np.full((reps, hz + 1), np.nan))
            for k in ("est k1", "est k2", "true k1", "true k2", "scenario")}
    for r in range(reps):
        rng = _rng(experiment * 10, r)
        y, curves, s1, s2 = dgp_flp_persistent(rng, T_persist)
        fp = tsecon.functional_pca(curves, n_factors=2)
        sc = np.asarray(fp["scores"])
        g1 = np.sign(np.sum(sc[:, 0] * s1))
        g2 = np.sign(np.sum(sc[:, 1] * s2))
        fl = tsecon.flp(y, sc, horizons=hz, n_lag_controls=0)
        flt = tsecon.flp(y, np.column_stack([s1, s2]), horizons=hz,
                         n_lag_controls=0)
        fs = tsecon.flp_scenario(y, curves, FLP_DELTA, n_factors=2,
                                 horizons=hz, n_lag_controls=0)
        be, se = np.asarray(fl["betas"]), np.asarray(fl["se"])
        bt, st = np.asarray(flt["betas"]), np.asarray(flt["se"])
        arms["est k1"][0][r], arms["est k1"][1][r] = g1 * be[:, 0], se[:, 0]
        arms["est k2"][0][r], arms["est k2"][1][r] = g2 * be[:, 1], se[:, 1]
        arms["true k1"][0][r], arms["true k1"][1][r] = bt[:, 0], st[:, 0]
        arms["true k2"][0][r], arms["true k2"][1][r] = bt[:, 1], st[:, 1]
        arms["scenario"][0][r] = np.asarray(fs["response"])
        arms["scenario"][1][r] = np.asarray(fs["se"])
    truths = {"est k1": truth1, "true k1": truth1,
              "est k2": truth2, "true k2": truth2, "scenario": truth_scn}
    label = {"est k1": "persistent est-scores k1",
             "est k2": "persistent est-scores k2",
             "true k1": "persistent true-scores k1",
             "true k2": "persistent true-scores k2",
             "scenario": "persistent scenario w'beta"}
    for k, (e, s) in arms.items():
        rows += summarize(e, s, truths[k], label[k])

    # ---- iid design (the canonical identified-impulse case) --------------
    arms_iid = {k: (np.full((reps, hz + 1), np.nan),
                    np.full((reps, hz + 1), np.nan))
                for k in ("est k1", "true k1")}
    for r in range(reps):
        rng = _rng(experiment * 10 + 1, r)
        y, curves, s1, s2 = dgp_flp_iid(rng, T_iid)
        fp = tsecon.functional_pca(curves, n_factors=2)
        sc = np.asarray(fp["scores"])
        g1 = np.sign(np.sum(sc[:, 0] * s1))
        fl = tsecon.flp(y, sc, horizons=hz)
        flt = tsecon.flp(y, np.column_stack([s1, s2]), horizons=hz)
        be, se = np.asarray(fl["betas"]), np.asarray(fl["se"])
        bt, st = np.asarray(flt["betas"]), np.asarray(flt["se"])
        arms_iid["est k1"][0][r] = g1 * be[:, 0]
        arms_iid["est k1"][1][r] = se[:, 0]
        arms_iid["true k1"][0][r] = bt[:, 0]
        arms_iid["true k1"][1][r] = st[:, 0]
    for k, (e, s) in arms_iid.items():
        rows += summarize(e, s, FLP_TH1[:hz + 1], f"iid {k.replace(' ', '-scores ')}")
    return {
        "name": "flp / flp_scenario: estimated vs external scores vs w'beta",
        "meta": {"T_persistent": T_persist, "T_iid": T_iid, "horizon": hz,
                 "nominal": NOMINAL, "reps": reps,
                 "scenario_weights": [FLP_W1, FLP_W2],
                 "n_lag_controls": {"persistent": 0, "iid": 2}},
        "rows": rows,
    }


# ==========================================================================
# Experiment 6 -- the surfaces that ship no interval, verified not assumed
# ==========================================================================
def exp_no_interval(experiment=6):
    """nongaussian_svar's whole key set, and garch_fit's variance_forecast.

    garch_fit legitimately ships `se_mle`/`se_robust` for its PARAMETERS
    (measured in experiment 4), so the whole-key-set tripwire cannot apply;
    the claim under test is narrower and exactly what the docstring states:
    `variance_forecast` is a bare point path with NO companion interval --
    no `forecast_se`, no `forecast_lower`/`upper`, no quantiles.
    """
    rng = _rng(experiment, 0)
    spec = PROXY_DGPS["card_var2"]
    y, _ = dgp_proxy(rng, spec)
    ng = tsecon.nongaussian_svar(y, lags=spec["lags"], horizon=4)
    ng_keys = sorted(ng.keys())
    ng_interval_like = [k for k in ng_keys if INTERVAL_KEY.search(k)]

    gf = tsecon.garch_fit(dgp_garch(rng, 500, "normal"), forecast_horizon=12)
    gf_keys = sorted(gf.keys())
    forecast_keys = [k for k in gf_keys if "forecast" in k.lower()]
    forecast_interval_like = [k for k in forecast_keys
                              if k != "variance_forecast"]
    return {
        "name": "no-interval surfaces (key sets verified)",
        "nongaussian_svar": {"keys": ng_keys,
                             "interval_like": ng_interval_like},
        "garch_forecast": {"keys": gf_keys,
                           "forecast_keys": forecast_keys,
                           "interval_like_forecast_keys":
                               forecast_interval_like,
                           "horizon_returned":
                               len(np.asarray(gf["variance_forecast"]))},
    }


# ==========================================================================
# assertions
# ==========================================================================
def _row(res, arm, h):
    hits = [r for r in res["rows"] if r["arm"] == arm and r["h"] == h]
    assert len(hits) == 1, (arm, h, len(hits))
    return hits[0]


def _worst(res, prefix):
    rows = [r for r in res["rows"] if r["arm"].startswith(prefix)]
    assert rows, prefix
    return min(rows, key=lambda r: r["cov95"])


def check(results, quick):
    checks = []

    def ok(label, passed, detail):
        checks.append((label, bool(passed), detail))

    # ---- proxy_svar_bands ----------------------------------------------
    e1 = results["proxy_bands"]
    ok("proxy_svar_bands: the (norm_var, h=0) cell is degenerate at unit in "
       "EVERY draw (the normalization is inside the loop)",
       e1["mbb"]["degenerate_ok"] and e1["wild"]["degenerate_ok"],
       f"mbb {e1['mbb']['degenerate_ok']}, wild {e1['wild']['degenerate_ok']}")
    ok("proxy_svar_bands: the wild arm labels itself asymptotically invalid "
       "and the moving block valid",
       e1["wild"]["valid_flags"] == [False]
       and e1["mbb"]["valid_flags"] == [True],
       f"wild flags {e1['wild']['valid_flags']}, "
       f"mbb flags {e1['mbb']['valid_flags']}")
    mbb_imp = _row(e1, "mbb hall var1", 0)
    wild_imp = min(_row(e1, "wild hall var1", 0),
                   _row(e1, "wild hall var2", 0), key=lambda r: r["cov95"])
    ok("proxy_svar_bands: the wild arm COLLAPSES at impact (< 0.5) while "
       "the moving block stays a working band (> 0.8) -- the frozen-moment "
       "defect, measured",
       wild_imp["cov95"] < 0.5 < 0.8 < mbb_imp["cov95"],
       f"wild impact {wild_imp['cov95']:.3f}, mbb impact "
       f"{mbb_imp['cov95']:.3f}")
    mbb_worst = _worst(e1, "mbb hall")
    ok("proxy_svar_bands: the moving-block Hall band loses ground at long "
       "horizons (worst cell at least 5pp below impact -- the inherited "
       "reduced-form bootstrap decay the card documents)",
       mbb_worst["cov95"] < mbb_imp["cov95"] - 0.05,
       f"worst {mbb_worst['arm']} h={mbb_worst['h']}: "
       f"{mbb_worst['cov95']:.3f} vs impact {mbb_imp['cov95']:.3f}")
    hall12 = np.mean([_row(e1, f"mbb hall var{i}", H_PROXY)["cov95"]
                      for i in range(3)])
    efron12 = np.mean([_row(e1, f"mbb efron var{i}", H_PROXY)["cov95"]
                       for i in range(3)])
    ok("proxy_svar_bands: at h=12 the Efron band covers better than Hall "
       "on this DGP (the skewed-tail direction; the card recommends Hall "
       "on theory grounds -- read both)",
       efron12 > hall12 + 0.03,
       f"h=12 pooled: Efron {efron12:.3f} vs Hall {hall12:.3f}")

    # ---- proxy_ar_sets --------------------------------------------------
    e2 = results["proxy_ar"]
    d_card_h1 = _row(e2, "card_var2 delta", 1)
    d_card_h12 = _row(e2, "card_var2 delta", 12)
    s_card_h12 = _row(e2, "card_var2 second_order", 12)
    d_rout_h12 = _row(e2, "routine_var1 delta", 12)
    s_rout_h12 = _row(e2, "routine_var1 second_order", 12)
    ok("proxy_ar_sets delta: at nominal at short horizons (h=1 within 3 "
       "MC se of 0.95)",
       abs(d_card_h1["cov95"] - NOMINAL) <= 3 * d_card_h1["mcse"],
       f"h=1: {d_card_h1['cov95']:.3f} +- {d_card_h1['mcse']:.3f}")
    ok("proxy_ar_sets delta: the long-horizon decline reproduces the card "
       "(card VAR(2) h=12 below 0.92; routine VAR(1) h=12 below 0.88)",
       d_card_h12["cov95"] < 0.92 and d_rout_h12["cov95"] < 0.88,
       f"card {d_card_h12['cov95']:.3f}, routine {d_rout_h12['cov95']:.3f}")
    ok("proxy_ar_sets second_order: repairs the long horizon on BOTH DGPs "
       "(h=12 gains >= 4pp over delta, paired draws)",
       (s_card_h12["cov95"] > d_card_h12["cov95"] + 0.04
        and s_rout_h12["cov95"] > d_rout_h12["cov95"] + 0.04),
       f"card {d_card_h12['cov95']:.3f} -> {s_card_h12['cov95']:.3f}; "
       f"routine {d_rout_h12['cov95']:.3f} -> {s_rout_h12['cov95']:.3f}")
    det = e2["detail"]
    one_sided = all(det[d]["miss_below_h12"]["delta"]
                    <= 0.05 * max(det[d]["miss_above_h12"]["delta"], 1)
                    for d in det)
    ok("proxy_ar_sets delta: the h=12 misses are one-sided (truth above "
       "the set), the documented direction",
       one_sided,
       "; ".join(f"{d}: above {det[d]['miss_above_h12']['delta']} / below "
                 f"{det[d]['miss_below_h12']['delta']}" for d in det))
    wr = det["card_var2"]["median_width_ratio"]["second_order"][12]
    ok("proxy_ar_sets second_order: the h=12 width price is bounded "
       "(median ratio to delta in [1.2, 2.0], card says ~1.45x)",
       1.2 <= wr <= 2.0, f"median width ratio h=12: {wr:.3f}")
    bc_card_h12 = _row(e2, "card_var2 second_order_bc", 12)
    bc_rout_h12 = _row(e2, "routine_var1 second_order_bc", 12)
    bc_min = min(r["cov95"] for r in e2["rows"]
                 if r["arm"].endswith("second_order_bc"))
    ok("proxy_ar_sets second_order_bc: at or above nominal at EVERY "
       "horizon on both DGPs (worst cell >= 0.935) -- the conservative "
       "floor it exists to buy",
       bc_min >= 0.935, f"worst by-horizon coverage: {bc_min:.3f}")
    ok("proxy_ar_sets second_order_bc: closes the residual gap on the "
       "routine VAR(1) at h=12 (above second_order, and >= 0.94) at the "
       "price of overshooting where second_order already reached nominal",
       (bc_rout_h12["cov95"] > s_rout_h12["cov95"]
        and bc_rout_h12["cov95"] >= 0.94
        and bc_card_h12["cov95"] >= s_card_h12["cov95"]),
       f"routine h=12: {s_rout_h12['cov95']:.3f} -> "
       f"{bc_rout_h12['cov95']:.3f}; card h=12: {s_card_h12['cov95']:.3f} "
       f"-> {bc_card_h12['cov95']:.3f} (conservative)")
    bcounts = [det[d]["bounded_count"] for d in det]
    ok("proxy_ar_sets: the boundedness decision is IDENTICAL across all "
       "three rf_methods on every draw (the correction enters v0 only)",
       all(len(set(bc.values())) == 1 for bc in bcounts),
       "; ".join(f"{d}: {det[d]['bounded_count']}" for d in det))
    wr_bc = det["card_var2"]["median_width_ratio"]["second_order_bc"][12]
    ok("proxy_ar_sets second_order_bc: the h=12 width price is bounded "
       "(median ratio to delta in [1.4, 2.3], measured ~1.8x)",
       1.4 <= wr_bc <= 2.3, f"median width ratio h=12: {wr_bc:.3f}")

    # ---- growth_at_risk -------------------------------------------------
    e3 = results["gar"]
    ok("growth_at_risk: bse == bse_powell at horizon=1 exactly (nothing "
       "overlaps, the documented identity)",
       e3["max_abs_bse_diff_h1"] == 0.0,
       f"max |bse - bse_powell| at h=1: {e3['max_abs_bse_diff_h1']:.2e}")
    b50_h1 = _row(e3, "bse tau=0.50", 1)
    b50_h12 = _row(e3, "bse tau=0.50", 12)
    p50_h12 = _row(e3, "powell tau=0.50", 12)
    b05_h12 = _row(e3, "bse tau=0.05", 12)
    ok("growth_at_risk: at h=1 the median-slope interval is at nominal "
       "(within 3 MC se)",
       abs(b50_h1["cov95"] - NOMINAL) <= 3 * b50_h1["mcse"],
       f"h=1 tau=0.5: {b50_h1['cov95']:.3f} +- {b50_h1['mcse']:.3f}")
    ok("growth_at_risk: at the median the Newey-West correction is the "
       "story -- bse beats bse_powell by >= 5pp at h=12",
       b50_h12["cov95"] > p50_h12["cov95"] + 0.05,
       f"h=12 tau=0.5: bse {b50_h12['cov95']:.3f} vs powell "
       f"{p50_h12['cov95']:.3f}")
    ok("growth_at_risk: in the tail the correction is only half the story "
       "-- tau=0.05 h=12 still under-covers (< 0.90, the card's "
       "density-estimate residual)",
       b05_h12["cov95"] < 0.90,
       f"h=12 tau=0.05 bse: {b05_h12['cov95']:.3f}")

    # ---- garch_fit ------------------------------------------------------
    e4 = results["garch"]
    fav_m = _worst(e4, "normal T=2000 se_mle")
    fav_r = _worst(e4, "normal T=2000 se_robust")
    t5_m = _worst(e4, "t5 T=2000 se_mle")
    t5_r = _worst(e4, "t5 T=2000 se_robust")
    ok("garch_fit: under Gaussian innovations at T=2000 both SE routes "
       "hold (worst parameter >= 0.92)",
       min(fav_m["cov95"], fav_r["cov95"]) >= 0.92,
       f"worst mle {fav_m['cov95']:.3f} (param {fav_m['h']}), worst robust "
       f"{fav_r['cov95']:.3f} (param {fav_r['h']})")
    ok("garch_fit: under t(5) innovations the MLE (inverse-Hessian) SEs "
       "collapse (worst < 0.85) while Bollerslev-Wooldridge holds most of "
       "it (worst > mle worst + 8pp) -- the QMLE story, measured",
       t5_m["cov95"] < 0.85 and t5_r["cov95"] > t5_m["cov95"] + 0.08,
       f"t5 worst: mle {t5_m['cov95']:.3f}, robust {t5_r['cov95']:.3f}")
    bad_share = max(v["n_boundary_or_invalid"] / v["reps"]
                    for v in e4["excluded"].values())
    ok("garch_fit: boundary / se_valid=False replications are rare on "
       "these designs (< 3% -- they are counted, never imputed)",
       bad_share < 0.03,
       "; ".join(f"{k}: {v['n_boundary_or_invalid']}/{v['reps']}"
                 for k, v in e4["excluded"].items()))

    # ---- flp / flp_scenario --------------------------------------------
    e5 = results["flp"]
    tru_imp = _row(e5, "persistent true-scores k1", 0)
    est_imp = _row(e5, "persistent est-scores k1", 0)
    scn_imp = _row(e5, "persistent scenario w'beta", 0)
    ok("flp: with EXTERNAL (true) scores the impact interval is at "
       "nominal (>= 0.93) -- the documented exempt case",
       tru_imp["cov95"] >= 0.93, f"impact: {tru_imp['cov95']:.3f}")
    ok("flp: with functional_pca scores the per-element impact interval "
       "collapses (>= 25pp below the external-score arm on the same "
       "draws) -- the card's generated-regressor warning, priced",
       est_imp["cov95"] < tru_imp["cov95"] - 0.25,
       f"est {est_imp['cov95']:.3f} vs true {tru_imp['cov95']:.3f}, "
       f"se/sd {est_imp['se_over_sd']:.2f}")
    ok("flp_scenario: the w'beta contrast is immune -- impact coverage "
       ">= 0.93 on the same draws where the per-element se collapses",
       scn_imp["cov95"] >= 0.93, f"scenario impact: {scn_imp['cov95']:.3f}")
    iid_est = _worst(e5, "iid est-scores k1")
    ok("flp: on the canonical iid-impulse design the estimated-score "
       "hazard is mild away from impact (worst cell >= 0.84)",
       iid_est["cov95"] >= 0.84,
       f"iid est-scores worst h={iid_est['h']}: {iid_est['cov95']:.3f}")

    # ---- no-interval surfaces ------------------------------------------
    e6 = results["no_interval"]
    ok("nongaussian_svar ships no interval-like key (so its page row is "
       "NONE, verified)",
       not e6["nongaussian_svar"]["interval_like"],
       f"keys = {e6['nongaussian_svar']['keys']}")
    ok("garch_fit variance_forecast is a bare point path: no forecast "
       "interval / se / quantile key exists beside it (the docstring's "
       "'none is implied', verified)",
       e6["garch_forecast"]["forecast_keys"] == ["variance_forecast"]
       and not e6["garch_forecast"]["interval_like_forecast_keys"],
       f"forecast keys = {e6['garch_forecast']['forecast_keys']}")

    print()
    print("=" * 100)
    print("ASSERTIONS")
    print("=" * 100)
    width = max(len(lbl) for lbl, _, _ in checks)
    failures = []
    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<{width}}  {detail}")
        if not passed:
            failures.append(label)
    if quick:
        print()
        print("--quick: Monte Carlo standard errors are ~3x the default "
              "run's, so treat a near-miss as noise and re-run without "
              "--quick.")
    if failures:
        raise AssertionError("coverage assertions failed: "
                             + "; ".join(failures))
    return checks


# ==========================================================================
# driver
# ==========================================================================
NOTES = """
WHERE THE INTERVALS MISS -- the honest list, read with the tables above
----------------------------------------------------------------------
1. proxy_svar_bands' moving-block band starts ~3-4pp short at impact and
   loses further ground through the horizon -- the decay the card already
   attributes to the reduced-form VAR bootstrap (no Kilian correction on
   the proxy path), now measured in this registry. On this DGP the Efron
   percentile band beats the recommended Hall band at h=12, because the
   bootstrap distribution is right-skewed exactly where the Hall reflection
   hurts; the card's Hall recommendation is a theory default, and both
   endpoints ship, so read both at long horizons. The wild arm is not an
   interval at all in the identifying step (the moment is bit-identical in
   every draw) and collapses at impact -- it exists to reproduce published
   bands and says so via asymptotically_valid=False.
2. proxy_ar_sets' default delta propagation reproduces the audit numbers:
   at nominal through h~4, then a one-sided decline (the truth exits above)
   to ~0.87-0.89 on the card DGP and ~0.83-0.85 on the routine VAR(1) at
   h=12. rf_method="second_order" (the shipped opt-in repair) recovers most
   of it on both DGPs at a ~1.45x h=12 width price -- and still sits ~1-2pp
   short on the routine VAR(1), the honest residual roadmap note 21
   records. rf_method="second_order_bc" (the note-21 follow-up: the same
   simulation centred at Pope-bias-corrected coefficients) closes that
   residual and is at-or-above nominal at every horizon on both DGPs --
   a conservative floor, not a calibration: it overshoots where
   second_order already reached nominal, at a ~1.8x h=12 width price.
   The default stays "delta"; pass "second_order" for the best point
   calibration, "second_order_bc" when long-horizon under-coverage is the
   error you most need to rule out.
3. growth_at_risk: bse's Newey-West correction is the whole story at the
   median (h=12 coverage ~0.91 vs ~0.82 uncorrected) and HALF the story in
   the 5% tail, where the Powell density estimate keeps the corrected
   interval under 0.90 at long horizons -- the card's own table, reproduced
   on an exact-truth design in this registry. bse_powell is kept for
   statsmodels replication, and at h=1 the two are identical (asserted
   exact every run).
4. garch_fit: at T=2000 under Gaussian innovations both SE routes are at
   nominal. Under t(5) innovations fitted with dist="normal" -- the QMLE
   case every fat-tailed financial series is in -- the inverse-Hessian
   se_mle collapses to ~0.74-0.76 while Bollerslev-Wooldridge se_robust
   holds ~0.89-0.93. Quote se_robust unless you have a reason to believe
   the innovation distribution. variance_forecast ships NO interval, and
   the key set is verified every run.
5. flp's per-element se conditions on the scores. With external scores it
   is at nominal (measured). With functional_pca scores on a persistent
   yield-curve-like design the impact interval collapses to ~0.45 with
   se/sd ~0.3 -- the card's warning, priced -- while flp_scenario's w'beta
   contrast on the SAME draws stays ~0.95 at impact (the documented
   reporting route, and the algebraic immunity is real). On the canonical
   iid-impulse design the hazard is confined to impact and mild. Report
   scenarios, not raw score coefficients.
"""


def run(quick=False):
    reps_full = {"bands_mbb": 1000, "bands_wild": 400, "proxy_ar": 1000,
                 "gar": 1500, "garch": 1000, "flp": 1500}
    scale = 8 if quick else 1
    reps = {k: max(50, v // scale) for k, v in reps_full.items()}
    n_boot = 500 if quick else 2000

    t0 = time.perf_counter()
    print("=" * 100)
    print("tsecon interval COVERAGE: proxy-SVAR inference, GARCH, "
          "growth-at-risk and functional LP")
    print("=" * 100)
    print(f"seed                = {SEED}   (every draw is default_rng("
          f"[{SEED}, experiment, replication]))")
    print(f"nominal levels      = proxy_svar_bands {NOMINAL_BANDS:.0%} "
          f"(alpha=0.10, its convention); everything else {NOMINAL:.0%}")
    print(f"mode                = {'QUICK SMOKE RUN' if quick else 'full'}")
    print("replications        = " + ", ".join(f"{k}:{v}" for k, v in
                                               reps.items())
          + f", n_boot:{n_boot}")

    results = {}

    header("EXPERIMENT 1 -- proxy_svar_bands: moving-block (Hall + Efron) "
           "and the wild reproduction arm (nominal 0.90)")
    print("card VAR(2) DGP, T=300, strong instrument; the degenerate "
          "(norm_var, h=0)\ncell is asserted exact and excluded.\n")
    results["proxy_bands"] = exp_proxy_bands(reps["bands_mbb"],
                                             reps["bands_wild"], n_boot)
    print_table(results["proxy_bands"]["rows"])
    print()
    print(f"mbb draws failed inside the bootstrap: "
          f"{results['proxy_bands']['mbb']['n_failed']} (counted by the "
          f"library, never dropped)")

    header('EXPERIMENT 2 -- proxy_ar_sets: rf_method="delta" vs '
           '"second_order" vs "second_order_bc", paired (nominal 0.95)')
    print("coverage is the mean over non-degenerate cells at each h; "
          "misses/widths at h=12 below.\n")
    results["proxy_ar"] = exp_proxy_ar(reps["proxy_ar"])
    print_table(results["proxy_ar"]["rows"])
    print()
    for dname, d in results["proxy_ar"]["detail"].items():
        misses = ", ".join(
            f"{m} {d['miss_above_h12'][m]}/{d['miss_below_h12'][m]}"
            for m in AR_METHODS)
        worst = ", ".join(f"{m} {d['worst_cell_h12'][m]:.3f}"
                          for m in AR_METHODS)
        print(f"  {dname}: h=12 misses above/below -- {misses}")
        print(f"  {dname}: worst h=12 cell -- {worst}; bounded counts "
              f"{d['bounded_count']} (identical = boundedness untouched)")
        wrs = d["median_width_ratio"]
        print(f"  {dname}: median width ratio to delta -- second_order "
              f"h=8 {wrs['second_order'][8]:.3f} / h=12 "
              f"{wrs['second_order'][12]:.3f}; second_order_bc "
              f"h=8 {wrs['second_order_bc'][8]:.3f} / h=12 "
              f"{wrs['second_order_bc'][12]:.3f}")

    header("EXPERIMENT 3 -- growth_at_risk: bse (Newey-West) vs bse_powell "
           "(nominal 0.95)")
    print("cell: the slope on the conditioning variable; h is the "
          "forecast horizon.\n")
    results["gar"] = exp_gar(reps["gar"])
    print_table(results["gar"]["rows"])

    header("EXPERIMENT 4 -- garch_fit: se_mle vs se_robust "
           "(nominal 0.95; h = parameter index 0 omega / 1 alpha / 2 beta)")
    results["garch"] = exp_garch(reps["garch"])
    print_table(results["garch"]["rows"])
    print()
    for k, v in results["garch"]["excluded"].items():
        print(f"  {k}: {v['n_boundary_or_invalid']}/{v['reps']} replications "
              f"boundary or se_valid=False (excluded, counted)")

    header("EXPERIMENT 5 -- flp / flp_scenario: the generated-regressor "
           "warning, priced (nominal 0.95)")
    results["flp"] = exp_flp(reps["flp"])
    print_table(results["flp"]["rows"])

    header("EXPERIMENT 6 -- the no-interval surfaces, key sets verified")
    results["no_interval"] = exp_no_interval()
    e6 = results["no_interval"]
    print(f"  nongaussian_svar  keys: "
          f"{', '.join(e6['nongaussian_svar']['keys'])}")
    print(f"  garch_fit         forecast keys: "
          f"{', '.join(e6['garch_forecast']['forecast_keys'])} "
          f"(horizon {e6['garch_forecast']['horizon_returned']}, point path "
          f"only)")

    print(NOTES)
    results["_checks"] = check(results, quick)
    elapsed = time.perf_counter() - t0
    print()
    print(f"runtime: {elapsed:.1f} s")
    results["_runtime_s"] = elapsed
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Interval coverage for tsecon proxy-SVAR inference, "
                    "GARCH, growth-at-risk and functional LP")
    parser.add_argument("--quick", action="store_true",
                        help="cut every replication count by 8 for a smoke "
                             "run")
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
