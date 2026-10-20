"""Golden fixtures for `long_run_svar_bands` (Blanchard-Quah long-run SVAR
confidence bands) and one `var_irf_bands` bootstrap case with Kilian's bias
correction.

    .venv/bin/python fixtures/generate_long_run_svar_bands_fixtures.py

Never imports tsecon. Two kinds of reference live here.

1. BOOTSTRAP ARM - a documented-formula cross-implementation golden (the GIRF
   precedent: an RNG replay, not stored indices). Replication b uses
   `numpy.random.Philox(numpy.random.SeedSequence(seed).spawn(n_boot)[b])`;
   each of the T_eff iid row indices is drawn by bitmask rejection on
   `Philox.random_raw()` (mask = 2**bitlen(n-1) - 1, ONE raw u64 per attempt,
   accept if < n). That is tsecon_rng::Stream::substreams plus
   tsecon_bootstrap::indices(Iid); crates/tsecon-rng/src/stream.rs documents
   the NumPy equivalence and fixtures/philox.json pins it. Per replication:
   y*_t = c + u*_{t-p} + sum_i A_i y*_{t-i} from the observed first p rows (no
   burn-in), an OLS refit (numpy.linalg.lstsq), and the Blanchard-Quah map
   re-run inside the draw (same restrictions / normalization / impact_signs),
   cumulated per draw when asked. Summaries: SD with ddof=1, percentiles at
   q(n-1) with linear interpolation. The full-sample fit is statsmodels'
   `VAR(data).fit(p, trend="c")`.

   The Kilian (1998) branch is a TRANSCRIPTION OF TSECON'S OWN ALGORITHM
   (crates/tsecon-var/src/irf_bootstrap.rs estimate_bias / correct_coefs):
   inner seed = seed + 0x9E3779B97F4A7C15 mod 2**64, n_inner = n_boot, inner
   trend None iff every fitted intercept is exactly 0.0, bias_raw =
   mean(A*) - A_hat, delta from 1.0 decreased by repeated `delta -= 0.01`
   until the companion radius of A_hat - delta*bias_raw is < 1 (zero bias if
   delta <= 0); corrected coefficients by the same per-call shrink, falling
   back to the uncorrected ones. Its only EXTERNAL anchor is Monte Carlo
   coverage (crates/tsecon-var/tests/long_run_bands_props.rs), and the
   validation matrix grades it that way.

   Margins are asserted so no pinned case is knife-edge: |radius - 1| > 1e-6
   at every Kilian/correct_coefs/n_unstable decision, |impact[v, s]| > 1e-8 at
   every anchor decision, |B[j, j]| > 1e-8 under normalize="impact". A case
   that failed a margin would be re-seeded, never pinned.

2. DELTA-METHOD ARM - Lutkepohl (2005) with statsmodels 0.15's own building
   blocks: IRAnalysis.G (dvec Phi_h / dalpha'), cov_a ((Z'Z)^-1 kron Sigma_u
   on the lag block, df-adjusted Sigma_u), cov_sig (2 D+ (Sigma kron Sigma)
   D+'), divided by T = nobs as in IRAnalysis._orth_cov. The Blanchard-Quah
   Jacobian dvec B / d(alpha, vech Sigma) is analytic (forward mode through
   D, C(1), the Cholesky of C(1) Sigma C(1)', the column permutation and the
   point estimate's column signs) and is checked here two ways: against
   central finite differences of the whole map (< 1e-6 relative), and - in
   the B = chol(Sigma_u) special case - against statsmodels
   IRAnalysis.stderr(orth=True) (< 1e-12 relative). IRAnalysis.lr_effect_cov
   (orth=True) is NOT used: in statsmodels 0.15 its Sigma term omits the /T
   that _orth_cov and cum_effect_cov apply.

CASES
-----
* `bq_var8` / `bq_var8_cum`: the Blanchard-Quah replication data
  (fixtures/macrodata_bq.csv, the page's transform), VAR(8), horizon 40,
  n_boot 200, seed 20260911, impact_signs [(0, 1)], plain / cumulative; plus
  the delta-method arm on the same fit.
* `bq_var8_bc` / `bq_var8_bc_cum`: as above with bias_correct=True.
* `var3_perm`: a seeded 3-variable VAR(2), restrictions [(0, 0), (1, 0),
  (0, 2)] (a column permutation of the recursive pattern), normalize="impact",
  alpha 0.32, n_boot 150, horizon 8, band="sup-t" (band_scope "horizon",
  critical value from the draws); plus the delta-method arm.
* `var_irf_bands_bc`: `var_irf_bands(method="bootstrap", orth=True,
  bias_correct=True)` on 100*dlog(real GDP, consumption, investment) from
  statsmodels' macrodata, VAR(2), horizon 6, n_boot 100 - the shared Kilian
  path pinned to this independent transcription.

Each bootstrap case stores the first 10 indices of replication 0
(`first_indices`) so a stream mismatch is diagnosable at a glance.
"""

import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.api import VAR

HERE = Path(__file__).resolve().parent
OUT = HERE / "long_run_svar_bands.json"
CSV = HERE / "macrodata_bq.csv"
BIAS_SEED_OFFSET = 0x9E3779B97F4A7C15
MARGINS = {"radius": math.inf, "anchor": math.inf, "diag": math.inf}

META = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
}


# ------------------------------------------------------------------- data
def qnum(label):
    y, q = label.split("Q")
    return int(y) + (int(q) - 1) / 4.0


def bq_data():
    rows = [r for r in csv.reader(open(CSV)) if r and not r[0].startswith("#")]
    assert rows[0] == ["quarter", "realgdp", "unemp"]
    body = rows[1:]
    quarter = [r[0] for r in body][1:]
    gdp = np.array([float(r[1]) for r in body])
    unemp = np.array([float(r[2]) for r in body])[1:]
    growth = 100.0 * np.diff(np.log(gdp))
    years = np.array([qnum(q) for q in quarter])
    post = years >= qnum("1974Q1")
    growth = growth - np.where(post, growth[post].mean(), growth[~post].mean())
    t = np.arange(len(unemp), dtype=float)
    Z = np.column_stack([np.ones_like(t), t])
    unemp = unemp - Z @ np.linalg.lstsq(Z, unemp, rcond=None)[0]
    return np.column_stack([growth, unemp])


def var3_data():
    rng = np.random.default_rng(20260914)
    A1 = np.array([[0.50, 0.10, 0.00], [0.15, 0.40, -0.10], [0.05, 0.10, 0.35]])
    A2 = np.array([[0.10, 0.00, 0.05], [0.00, 0.15, 0.00], [-0.05, 0.00, 0.10]])
    L = np.array([[1.0, 0.0, 0.0], [0.4, 0.9, 0.0], [-0.3, 0.2, 0.7]])
    y = np.zeros((200 + 300 + 2, 3))
    for t in range(2, len(y)):
        y[t] = 0.3 + A1 @ y[t - 1] + A2 @ y[t - 2] + L @ rng.standard_normal(3)
    return y[300:]


def macro3_data():
    d = sm.datasets.macrodata.load_pandas().data
    return 100.0 * np.diff(np.log(d[["realgdp", "realcons", "realinv"]].to_numpy()), axis=0)


# ------------------------------------------------------------ primitives
def uniform_index(bg, n):
    """tsecon_bootstrap uniform_index: bitmask rejection, one raw per attempt."""
    if n == 1:
        return 0
    mask = (1 << (n - 1).bit_length()) - 1
    while True:
        v = int(bg.random_raw()) & mask
        if v < n:
            return v


def iid_indices(child, n):
    bg = np.random.Philox(child)
    return [uniform_index(bg, n) for _ in range(n)]


def var_ols(y, p, trend_c=True):
    T, k = y.shape
    te = T - p
    cols = ([np.ones((te, 1))] if trend_c else []) + [y[p - l:T - l] for l in range(1, p + 1)]
    Z = np.hstack(cols)
    Y = y[p:]
    Bm = np.linalg.lstsq(Z, Y, rcond=None)[0]
    U = Y - Z @ Bm
    m = Z.shape[1]
    ntr = 1 if trend_c else 0
    coefs = [Bm[ntr + (l - 1) * k: ntr + l * k].T.copy() for l in range(1, p + 1)]
    intercept = Bm[0].copy() if trend_c else np.zeros(k)
    return {"coefs": coefs, "intercept": intercept, "resid": U, "sigma": U.T @ U / (te - m)}


def sm_fit(y, p):
    res = VAR(y).fit(p, trend="c")
    return res, {
        "coefs": [np.asarray(a).copy() for a in res.coefs],
        "intercept": np.asarray(res.intercept).copy(),
        "resid": np.asarray(res.resid).copy(),
        "sigma": np.asarray(res.sigma_u).copy(),
    }


def radius(coefs, track=True):
    k = coefs[0].shape[0]
    p = len(coefs)
    C = np.zeros((k * p, k * p))
    C[:k, :] = np.hstack(coefs)
    if p > 1:
        C[k:, :-k] = np.eye(k * (p - 1))
    r = float(np.abs(np.linalg.eigvals(C)).max())
    if track:
        MARGINS["radius"] = min(MARGINS["radius"], abs(r - 1.0))
    return r


def simulate(coefs, intercept, init, ustar):
    """Rust simulate_recursive, operation for operation."""
    p = len(coefs)
    k = init.shape[1]
    te = ustar.shape[0]
    A = [a.tolist() for a in coefs]
    c = intercept.tolist()
    u = ustar.tolist()
    y = init.tolist() + [[0.0] * k for _ in range(te)]
    for t in range(p, p + te):
        for r in range(k):
            v = c[r] + u[t - p][r]
            for i in range(1, p + 1):
                a = A[i - 1][r]
                yt = y[t - i]
                for cc in range(k):
                    v += a[cc] * yt[cc]
            y[t][r] = v
    return np.array(y)


def scale_subtract(coefs, bias, delta):
    return [a - delta * b for a, b in zip(coefs, bias)]


def estimate_bias(coefs_hat, intercept, init, uc, seed, n_inner, p):
    trend_c = not all(c == 0.0 for c in intercept)
    te = uc.shape[0]
    children = np.random.SeedSequence((seed + BIAS_SEED_OFFSET) % 2**64).spawn(n_inner)
    acc = [np.zeros_like(a) for a in coefs_hat]
    for b in range(n_inner):
        idx = iid_indices(children[b], te)
        ys = simulate(coefs_hat, intercept, init, uc[idx])
        est = var_ols(ys, p, trend_c)["coefs"]
        acc = [a + e for a, e in zip(acc, est)]
    nf = float(n_inner)
    bias_raw = [a / nf - c for a, c in zip(acc, coefs_hat)]
    delta = 1.0
    while True:
        if radius(scale_subtract(coefs_hat, bias_raw, delta)) < 1.0:
            return [delta * b for b in bias_raw]
        delta -= 0.01
        if delta <= 0.0:
            return [np.zeros_like(b) for b in bias_raw]


def correct_coefs(coefs, bias):
    delta = 1.0
    while True:
        cand = scale_subtract(coefs, bias, delta)
        if radius(cand) < 1.0:
            return cand
        delta -= 0.01
        if delta <= 0.0:
            return [a.copy() for a in coefs]


def ordering(zeros, k):
    """tsecon_ident long_run_ordering: col_of_pos[c] = shock with c zeros."""
    if zeros is None:
        return list(range(k))
    rows = [[] for _ in range(k)]
    for v, s in zeros:
        rows[s].append(v)
    count_to_shock = [None] * k
    for s, r in enumerate(rows):
        z = sorted(set(r))
        assert z == list(range(len(z)))
        count_to_shock[len(z)] = s
    return count_to_shock


def ma_psi(coefs, H):
    k = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(k)]
    for h in range(1, H + 1):
        acc = np.zeros((k, k))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def bq_map(coefs, sigma, H, zeros, normalize_impact, anchors, cumulative, track=True):
    k = sigma.shape[0]
    D = np.eye(k)
    for a in coefs:
        D = D - a
    C1 = np.linalg.inv(D)
    M = C1 @ sigma @ C1.T
    M = 0.5 * (M + M.T)
    L = np.linalg.cholesky(M)
    LR = np.zeros((k, k))
    for c, s in enumerate(ordering(zeros, k)):
        LR[:, s] = L[:, c]
    B = D @ LR
    if normalize_impact:
        for s in range(k):
            if track:
                MARGINS["diag"] = min(MARGINS["diag"], abs(B[s, s]))
            if B[s, s] < 0:
                B[:, s] = -B[:, s]
                LR[:, s] = -LR[:, s]
    pre = []
    for v, s in anchors or []:
        pre.append(float(B[v, s]))
        if track:
            MARGINS["anchor"] = min(MARGINS["anchor"], abs(B[v, s]))
        if B[v, s] < 0:
            B[:, s] = -B[:, s]
            LR[:, s] = -LR[:, s]
    theta = np.array([ps @ B for ps in ma_psi(coefs, H)])
    if cumulative:
        theta = np.cumsum(theta, axis=0)
    return {"B": B, "LR": LR, "C1": C1, "D": D, "L": L, "theta": theta, "pre": pre}


def percentile_sorted(s, q):
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1.0)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def summarize(draws, alpha):
    """draws: (n, ...) -> se (ddof 1), lower, upper elementwise (Rust order)."""
    n = draws.shape[0]
    flat = draws.reshape(n, -1)
    se = np.empty(flat.shape[1])
    lo = np.empty(flat.shape[1])
    up = np.empty(flat.shape[1])
    for c in range(flat.shape[1]):
        vals = flat[:, c].tolist()
        mean = sum(vals) / n
        ss = sum((v - mean) * (v - mean) for v in vals)
        se[c] = math.sqrt(ss / (n - 1.0))
        s = sorted(vals)
        lo[c] = percentile_sorted(s, alpha / 2.0)
        up[c] = percentile_sorted(s, 1.0 - alpha / 2.0)
    shp = draws.shape[1:]
    return se.reshape(shp), lo.reshape(shp), up.reshape(shp)


def sup_t_from_draws(fam_draws, theta, sd, alpha):
    active = [j for j in range(len(sd)) if sd[j] > 0]
    maxima = []
    for row in fam_draws:
        m = 0.0
        for j in active:
            t = abs((row[j] - theta[j]) / sd[j])
            if t > m:
                m = t
        maxima.append(m)
    c = float(np.quantile(np.array(maxima), 1.0 - alpha))
    return max(c, float(norm.ppf(1.0 - alpha / 2.0)))


# ---------------------------------------------------------------- bootstrap
def residual_bootstrap(X, p, n_boot, seed, bias_correct, per_draw):
    _, fit = sm_fit(X, p)
    te = fit["resid"].shape[0]
    uc = fit["resid"] - fit["resid"].sum(axis=0) / te
    init = X[:p].copy()
    bias = None
    dgp = [a.copy() for a in fit["coefs"]]
    if bias_correct:
        bias = estimate_bias(fit["coefs"], fit["intercept"], init, uc, seed, n_boot, p)
        dgp = correct_coefs(fit["coefs"], bias)
    children = np.random.SeedSequence(seed).spawn(n_boot)
    out = []
    first = None
    for b in range(n_boot):
        idx = iid_indices(children[b], te)
        if b == 0:
            first = idx[:10]
        ys = simulate(dgp, fit["intercept"], init, uc[idx])
        fb = var_ols(ys, p, True)
        cb = correct_coefs(fb["coefs"], bias) if bias is not None else fb["coefs"]
        out.append(per_draw(fb, cb))
    return fit, dgp, out, first


def bq_bootstrap_case(X, p, H, n_boot, seed, alpha, zeros, normalize_impact, anchors,
                      bias_correct, band_supt=False):
    """Plain and cumulative results from ONE set of draws."""
    def per_draw(fb, cb):
        r = radius(fb["coefs"])
        m = bq_map(cb, fb["sigma"], H, zeros, normalize_impact, anchors, False)
        return {"theta": m["theta"], "LR": m["LR"], "pre": m["pre"], "radius": r}

    fit, dgp, draws, first = residual_bootstrap(X, p, n_boot, seed, bias_correct, per_draw)
    pt = bq_map(dgp, fit["sigma"], H, zeros, normalize_impact, anchors, False)
    out = {}
    for cum in (False, True):
        point = np.cumsum(pt["theta"], axis=0) if cum else pt["theta"]
        th = np.array([np.cumsum(d["theta"], axis=0) if cum else d["theta"] for d in draws])
        se, lo, up = summarize(th, alpha)
        lr = np.array([d["LR"] for d in draws])
        lse, llo, lup = summarize(lr, alpha)
        case = {
            "point": point.tolist(), "se": se.tolist(), "lower": lo.tolist(),
            "upper": up.tolist(),
            "long_run_point": pt["LR"].tolist(), "long_run_se": lse.tolist(),
            "long_run_lower": llo.tolist(), "long_run_upper": lup.tolist(),
            "impact_point": pt["B"].tolist(),
            "n_unstable_draws": int(sum(d["radius"] >= 1.0 for d in draws)),
            "first_indices": first,
        }
        if anchors:
            case["sign_anchor_mismatches"] = [
                int(sum((d["pre"][a] < 0.0) != (pt["pre"][a] < 0.0) for d in draws))
                for a in range(len(anchors))
            ]
        if band_supt:
            k = point.shape[1]
            cv = [[None] * k for _ in range(k)]
            for i in range(k):
                for j in range(k):
                    cv[i][j] = sup_t_from_draws(th[:, :, i, j].tolist(), point[:, i, j].tolist(),
                                                se[:, i, j].tolist(), alpha)
            case["critical_value"] = cv
            case["sim_lower"] = (point - np.array(cv)[None] * se).tolist()
            case["sim_upper"] = (point + np.array(cv)[None] * se).tolist()
        out[cum] = case
    return out


# ------------------------------------------------------------- delta method
def chol_dir(L, dM):
    k = L.shape[0]
    Linv = np.linalg.inv(L)
    Xm = Linv @ dM @ Linv.T
    Phi = np.tril(Xm, -1) + 0.5 * np.diag(np.diag(Xm))
    return L @ Phi


def vech_pairs(k):
    return [(i, j) for i in range(k) for j in range(i, k)]


def bq_jacobian(coefs, sigma, pt, zeros):
    """Analytic forward-mode dvec B, dvec LR w.r.t. (vec alpha, vech Sigma)."""
    k = sigma.shape[0]
    p = len(coefs)
    dim, pairs = p * k * k, vech_pairs(k)
    C1, D, L, LR = pt["C1"], pt["D"], pt["L"], pt["LR"]
    col = ordering(zeros, k)
    signs = np.ones(k)
    for c, s in enumerate(col):
        signs[s] = 1.0 if LR[c, s] >= 0 else -1.0
        assert np.allclose(signs[s] * LR[:, s], L[:, c], atol=1e-12)
    JB = np.zeros((k * k, dim + len(pairs)))
    JLR = np.zeros_like(JB)
    for d in range(dim + len(pairs)):
        dD = np.zeros((k, k))
        dS = np.zeros((k, k))
        if d < dim:
            a, e = divmod(d, k)
            dD[e, a % k] = -1.0
        else:
            i, j = pairs[d - dim]
            dS[i, j] = dS[j, i] = 1.0
        dC1 = -C1 @ dD @ C1
        dM = dC1 @ sigma @ C1.T + C1 @ dS @ C1.T + C1 @ sigma @ dC1.T
        dL = chol_dir(L, dM)
        dLR = np.zeros((k, k))
        for c, s in enumerate(col):
            dLR[:, s] = signs[s] * dL[:, c]
        dB = dD @ LR + D @ dLR
        JB[:, d] = dB.reshape(-1, order="F")
        JLR[:, d] = dLR.reshape(-1, order="F")
    return JB, JLR


def delta_rows(B, JB, G, Phi, H, cumulative):
    """Rows (B' kron I) G_h + (I kron Phi_h) J_B per horizon, cell order h, j, i."""
    k = B.shape[0]
    k2 = k * k
    dim = G[0].shape[1] if len(G) else JB.shape[1]
    rows = []
    F = np.zeros((k2, dim))
    Xi = np.zeros((k, k))
    for h in range(H + 1):
        if h == 0:
            Gh = np.zeros((k2, dim))
        elif cumulative:
            F = F + G[h - 1]
            Gh = F
        else:
            Gh = G[h - 1]
        if cumulative:
            Xi = Xi + Phi[h]
            Ph = Xi
        else:
            Ph = Phi[h]
        Ja = np.kron(B.T, np.eye(k)) @ Gh + np.kron(np.eye(k), Ph) @ JB[:, :dim]
        Js = np.kron(np.eye(k), Ph) @ JB[:, dim:]
        rows.append((Ja, Js))
    return rows


def delta_se(rows, cov_a, cov_sig, T):
    out = []
    for Ja, Js in rows:
        v = np.einsum("ra,ab,rb->r", Ja, cov_a, Ja) + np.einsum("ra,ab,rb->r", Js, cov_sig, Js) / T
        out.append(np.sqrt(np.maximum(v, 0.0)))
    return out


def bq_delta_case(X, p, H, alpha, zeros, normalize_impact, anchors, cumulative):
    res, fit = sm_fit(X, p)
    ira = res.irf(H)
    G, cov_a, cov_sig, T = list(ira.G), np.asarray(ira.cov_a), np.asarray(ira.cov_sig), float(ira.T)
    assert T == res.nobs
    Phi = np.asarray(ira.irfs)
    k = X.shape[1]
    pt = bq_map(fit["coefs"], fit["sigma"], H, zeros, normalize_impact, anchors, cumulative)
    JB, JLR = bq_jacobian(fit["coefs"], fit["sigma"], pt, zeros)
    rows = delta_rows(pt["B"], JB, G, Phi, H, cumulative)
    se_vec = delta_se(rows, cov_a, cov_sig, T)
    se = np.array([s.reshape(k, k, order="F") for s in se_vec])
    dim = cov_a.shape[0]
    lr_se = np.sqrt(np.maximum(
        np.einsum("ra,ab,rb->r", JLR[:, :dim], cov_a, JLR[:, :dim])
        + np.einsum("ra,ab,rb->r", JLR[:, dim:], cov_sig, JLR[:, dim:]) / T, 0.0
    )).reshape(k, k, order="F")
    # Finite-difference check of the analytic Jacobian of the whole map.
    fd_check(fit, H, zeros, normalize_impact, anchors, cumulative, pt, JB, JLR, rows)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    theta = pt["theta"]
    return {
        "point": theta.tolist(), "se": se.tolist(),
        "lower": (theta - z * se).tolist(), "upper": (theta + z * se).tolist(),
        "long_run_point": pt["LR"].tolist(), "long_run_se": lr_se.tolist(),
        "long_run_lower": (pt["LR"] - z * lr_se).tolist(),
        "long_run_upper": (pt["LR"] + z * lr_se).tolist(),
    }


def fd_check(fit, H, zeros, normalize_impact, anchors, cumulative, pt, JB, JLR, rows):
    coefs, sigma = fit["coefs"], fit["sigma"]
    k = sigma.shape[0]
    p = len(coefs)
    dim = p * k * k
    pairs = vech_pairs(k)
    eps = 1e-6

    def F(cf, sg):
        m = bq_map(cf, sg, H, zeros, normalize_impact, anchors, cumulative, track=False)
        assert np.allclose(np.sign(m["B"][np.abs(m["B"]) > 1e-6]),
                           np.sign(pt["B"][np.abs(m["B"]) > 1e-6]))
        th = np.concatenate([t.reshape(-1, order="F") for t in m["theta"]])
        return th, m["LR"].reshape(-1, order="F"), m["B"].reshape(-1, order="F")

    Jth = np.vstack([np.hstack([Ja, Js]) for Ja, Js in rows])
    worst = 0.0
    for d in range(dim + len(pairs)):
        cp = [a.copy() for a in coefs]
        cm = [a.copy() for a in coefs]
        sp, smn = sigma.copy(), sigma.copy()
        if d < dim:
            a, e = divmod(d, k)
            lag, v = divmod(a, k)
            cp[lag][e, v] += eps
            cm[lag][e, v] -= eps
        else:
            i, j = pairs[d - dim]
            for (r, c) in {(i, j), (j, i)}:
                sp[r, c] += eps
                smn[r, c] -= eps
        tp, lp, bp = F(cp, sp)
        tm, lm, bm = F(cm, smn)
        for fd, an in (((tp - tm) / (2 * eps), Jth[:, d]), ((lp - lm) / (2 * eps), JLR[:, d]),
                       ((bp - bm) / (2 * eps), JB[:, d])):
            scale = max(np.abs(an).max(), 1e-3)
            worst = max(worst, np.abs(fd - an).max() / scale)
    assert worst < 1e-6, f"analytic BQ Jacobian vs central differences: {worst:.2e}"
    FD_WORST.append(worst)


FD_WORST = []


def chol_reduction_check(X, p, H):
    """B = chol(Sigma_u): the same delta code must reproduce statsmodels."""
    res, fit = sm_fit(X, p)
    ira = res.irf(H)
    G, cov_a, cov_sig, T = list(ira.G), np.asarray(ira.cov_a), np.asarray(ira.cov_sig), float(ira.T)
    Phi = np.asarray(ira.irfs)
    sigma = fit["sigma"]
    k = sigma.shape[0]
    P = np.linalg.cholesky(sigma)
    dim = cov_a.shape[0]
    pairs = vech_pairs(k)
    JB = np.zeros((k * k, dim + len(pairs)))
    for n, (i, j) in enumerate(pairs):
        dS = np.zeros((k, k))
        dS[i, j] = dS[j, i] = 1.0
        JB[:, dim + n] = chol_dir(P, dS).reshape(-1, order="F")
    rows = delta_rows(P, JB, G, Phi, H, False)
    se = np.array([s.reshape(k, k, order="F") for s in delta_se(rows, cov_a, cov_sig, T)])
    ref = np.asarray(ira.stderr(orth=True))
    rel = np.abs(se - ref).max() / np.abs(ref).max()
    assert rel < 1e-12, f"chol reduction vs statsmodels stderr(orth=True): {rel:.2e}"
    return rel


# ---------------------------------------------------------- var_irf_bands bc
def var_irf_bc_case(X, p, H, n_boot, seed, alpha):
    def chol_irf(cf, sg):
        P = np.linalg.cholesky(sg)
        return np.array([ps @ P for ps in ma_psi(cf, H)])

    fit, dgp, draws, first = residual_bootstrap(
        X, p, n_boot, seed, True, lambda fb, cb: chol_irf(cb, fb["sigma"]))
    point = chol_irf(dgp, fit["sigma"])
    se, lo, up = summarize(np.array(draws), alpha)
    return {
        "point": point.tolist(), "se": se.tolist(), "lower": lo.tolist(),
        "upper": up.tolist(), "first_indices": first,
    }


# --------------------------------------------------------------------- main
def main():
    X = bq_data()
    bq_args = dict(p=8, H=40, n_boot=200, seed=20260911, alpha=0.1, zeros=None,
                   normalize_impact=False, anchors=[(0, 1)])
    plain = bq_bootstrap_case(X, bias_correct=False, **bq_args)
    bc = bq_bootstrap_case(X, bias_correct=True, **bq_args)
    spec_bq = {"lags": 8, "horizon": 40, "n_boot": 200, "seed": 20260911, "alpha": 0.1,
               "restrictions": None, "normalize": "long_run", "impact_signs": [[0, 1]]}
    delta_bq = {cum: bq_delta_case(X, 8, 40, 0.1, None, False, [(0, 1)], cum)
                for cum in (False, True)}

    X3 = var3_data()
    zeros3 = [(0, 0), (1, 0), (0, 2)]
    v3 = bq_bootstrap_case(X3, p=2, H=8, n_boot=150, seed=20260915, alpha=0.32, zeros=zeros3,
                           normalize_impact=True, anchors=None, bias_correct=False,
                           band_supt=True)
    delta_v3 = bq_delta_case(X3, 2, 8, 0.32, zeros3, True, None, False)
    spec_v3 = {"lags": 2, "horizon": 8, "n_boot": 150, "seed": 20260915, "alpha": 0.32,
               "restrictions": [list(z) for z in zeros3], "normalize": "impact",
               "impact_signs": None, "band": "sup-t", "band_scope": "horizon"}

    rel_bq = chol_reduction_check(X, 8, 40)
    rel_v3 = chol_reduction_check(X3, 2, 8)

    M3 = macro3_data()
    vib = var_irf_bc_case(M3, 2, 6, 100, 20260913, 0.1)

    assert MARGINS["radius"] > 1e-6, MARGINS
    assert MARGINS["anchor"] > 1e-8, MARGINS
    assert MARGINS["diag"] > 1e-8, MARGINS

    fx = {
        "_meta": META,
        "_doc": (
            "long_run_svar_bands goldens: bootstrap arm by an RNG replay of "
            "SeedSequence(seed).spawn(n_boot) Philox streams (bitmask rejection, one raw u64 per "
            "attempt) with the Blanchard-Quah map re-run per draw; Kilian branch transcribed from "
            "tsecon's irf_bootstrap.rs; delta arm from statsmodels IRAnalysis G/cov_a/cov_sig/T "
            "with an analytic BQ Jacobian checked by central differences and, at B = chol(Sigma), "
            "against IRAnalysis.stderr(orth=True). Generated by "
            "fixtures/generate_long_run_svar_bands_fixtures.py; never imports tsecon."
        ),
        "margins": {k: float(v) for k, v in MARGINS.items()},
        "fd_jacobian_worst_rel": max(FD_WORST),
        "chol_reduction_rel": {"bq_var8": rel_bq, "var3_perm": rel_v3},
        "bq_data": X.tolist(),
        "var3_data": X3.tolist(),
        "macro3_data": M3.tolist(),
        "bq_var8": {"spec": {**spec_bq, "cumulative": False, "bias_correct": False}, **plain[False]},
        "bq_var8_cum": {"spec": {**spec_bq, "cumulative": True, "bias_correct": False}, **plain[True]},
        "bq_var8_bc": {"spec": {**spec_bq, "cumulative": False, "bias_correct": True}, **bc[False]},
        "bq_var8_bc_cum": {"spec": {**spec_bq, "cumulative": True, "bias_correct": True}, **bc[True]},
        "bq_var8_delta": {"spec": {**spec_bq, "cumulative": False}, **delta_bq[False]},
        "bq_var8_delta_cum": {"spec": {**spec_bq, "cumulative": True}, **delta_bq[True]},
        "var3_perm": {"spec": {**spec_v3, "cumulative": False, "bias_correct": False}, **v3[False]},
        "var3_perm_delta": {"spec": {**spec_v3, "cumulative": False}, **delta_v3},
        "var_irf_bands_bc": {
            "spec": {"lags": 2, "horizon": 6, "n_boot": 100, "seed": 20260913, "alpha": 0.1,
                     "orth": True, "cumulative": False, "bias_correct": True},
            **vib,
        },
    }
    OUT.write_text(json.dumps(fx, indent=1) + "\n")
    print("wrote", OUT)
    print("  margins", fx["margins"], "fd worst", fx["fd_jacobian_worst_rel"],
          "chol rel", fx["chol_reduction_rel"])
    c = fx["bq_var8_cum"]
    print("  cumulative output<-supply h=40 lower:", c["lower"][40][0][0])
    print("  unemployment<-demand h=4 se:", fx["bq_var8"]["se"][4][1][1])
    print("  delta se[0][0][0]:", fx["bq_var8_delta"]["se"][0][0][0],
          " se[4][1][1]:", fx["bq_var8_delta"]["se"][4][1][1])
    print("  n_unstable plain/bc:", fx["bq_var8"]["n_unstable_draws"],
          fx["bq_var8_bc"]["n_unstable_draws"], " mismatches:",
          fx["bq_var8"]["sign_anchor_mismatches"], fx["bq_var8_bc"]["sign_anchor_mismatches"])
    print("  var3 cv:", fx["var3_perm"]["critical_value"])


if __name__ == "__main__":
    main()
