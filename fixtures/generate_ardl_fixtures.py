"""Golden fixtures for `ardl_bounds_test`: the ARDL / unrestricted
error-correction model (UECM) and the Pesaran-Shin-Smith (2001) bounds test
for a level relationship.

Reference implementation (this venv; never imports tsecon):
statsmodels 0.15.0 ``statsmodels.tsa.ardl.{UECM, ARDL, ardl_select_order}``,
``statsmodels.tsa.ardl.pss_critical_values`` (the 32-million-draw PSS
response surface) and ``statsmodels.regression.linear_model.OLS``, with
SciPy for the normal / Student-t tails.

The model (statsmodels' UECM, numpy input):

    Delta y_t = d_t' delta + pi y_{t-1} + sum_j theta_j x_{j,t-1}
                + sum_{i=1}^{p-1} phi_i Delta y_{t-i}
                + sum_j sum_{l=0}^{q_j-1} beta_{j,l} Delta x_{j,t-l} + e_t,

rows t = h .. T-1 with h = max(p, max_j q_j); the deterministics d_t are
``const`` and ``trend`` = t + 1 over the FULL sample (statsmodels'
``DeterministicProcess.in_sample``), chosen by the PSS case: case 1 none,
cases 2/3 a constant, cases 4/5 constant and trend. Columns are ordered
``[const, trend, y.L1, x1.L1..xk.L1, D.y.L1..D.y.L{p-1}, D.x1.L0..``
``D.x1.L{q1-1}, ...]`` (statsmodels' names with 1-based x indices).

Every stored number is statsmodels' own output or a documented formula
evaluated on it, and the generator asserts the two routes agree:

  * estimates: ``UECM(...).fit()`` params / bse (stored as ``se``) /
    tvalues / cov, and the residual Delta y - Z b computed here and
    asserted equal to ``res.resid`` (statsmodels issue #9793 changed the
    UECM resid convention before 0.15; this pins >= 0.15);
  * fit statistics (``nobs``, ``df_resid``, ``llf``, ``pvalues``) come
    from statsmodels ``OLS`` on the UECM's own design (``model._y``,
    ``model._x``) and ``aic``/``bic``/``hqic`` = -2 llf + c (m + 1),
    c = 2 / ln n / 2 ln ln n (statsmodels' Lutkepohl form). This is
    DELIBERATE, see DEVIATION 2 below;
  * the bounds F through TWO independent statsmodels routes:
    (a) ``UECMResults.bounds_test(case).statistic`` and (b) the levels
    ``ARDL(...).fit().f_test((R, r))`` with the affine restriction
    sum phi = 1, sum_l beta_{j,l} = 0 (plus const = 0 for case 2,
    trend = 0 for case 4); the t statistic on y.L1 through (a) the UECM
    tvalue and (b) (sum phi - 1) / se from the levels covariance. The two
    agree to 1.5e-9 relative on F at worst (asserted at 1e-8): the levels
    design is far worse conditioned (cond up to ~6e5 on macrodata), so
    route (a) is the pinned value and route (b) its independent check;
  * long run: ``-ci_params`` / ``ci_bse`` without the y entry, normal
    p-values evaluated here with ``norm.sf`` and asserted against
    statsmodels' ``ci_pvalues`` (1 - cdf) at atol 1e-15;
  * selection: ``ardl_select_order`` (common sample h* = max(P, Q),
    deterministics partialled out by FWL) filtered to the UECM-admissible
    subgrid p >= 1 and every q_j >= 1 (and to the fixed lags / order when
    one of them is fixed); the winner's margin over the runner-up is
    asserted above 1e-6, and the winner is refit on its OWN sample;
  * critical values / p-values: ``pss_critical_values.crit_vals`` and
    the response surface at the CORRECTED key k = n_x (DEVIATION 1),
    p = ``norm.sf(sum a_i (ln F)^i)``, small_p iff F > stat_star, asserted
    against statsmodels' ``_pss_pvalue`` (1 - cdf) at atol 1e-15.

DEVIATION 1 (statsmodels table-key defect, documented, not inherited).
``UECMResults.bounds_test`` looks the PSS table up at
k = len(ardl_order) = 1 + n_x, but the table generator
(``_pss_critical_values/pss.py``) keys k by the number of REGRESSORS
(``u`` has k + 1 columns: y plus k x's). statsmodels' own simulator
``_pss_simulate`` keys k as the total variable count (y plus k - 1 x's),
so it is right at 1 + n_x — and it reproduces the table row at n_x, not at
n_x + 1. Evidence stored here: ``canary_key`` (bounds_test's crit / p
equal the k = n_x + 1 table row for every n_x <= 9 case), ``canary_sim``
(``_pss_simulate`` at 1 + n_x within 0.10 of the n_x row and more than
0.3 from the n_x + 1 row) and ``published`` (PSS 2001's own Table CI(iii)
/ CI(v) 5% bounds within 0.05 of the n_x row).

DEVIATION 2 (statsmodels ARDL/UECM nobs defect, documented, not
inherited). ``ARDL.__init__`` rebuilds the regressors on hold_back
h = max(p, max q_j) but never resets ``nobs`` from ``AutoReg.__init__``,
which used hold_back p. So whenever max q_j > p, ``res.nobs`` is
T - p instead of the T - h rows the regression has, and ``llf``,
``aic``/``bic``/``hqic``, ``df_resid`` and the Student-t ``pvalues`` (on
that df) are off, while ``bse``/``cov`` (OLS's s^2 = ssr/(T - h - m)) are
right — the reported p-values and standard errors use different degrees
of freedom. tsecon uses the regression's actual row count; the fixture
stores statsmodels' raw values in ``sm_reported`` and asserts that they
equal the OLS values exactly when p >= max q_j and differ otherwise.

Data (DERIVED NUMBERS, plus one public-domain dataset; see
fixtures/README.md):
  * ``macrodata``: y = 100 log realcons, x = [100 log realgdp, realint]
    (statsmodels' bundled US macrodata, public domain), T = 203;
  * ``danish``: the Johansen-Juselius (1990) Danish money-demand series
    lrm on [lry, ibo, ide] (statsmodels ``danish_data``, marked public
    domain in statsmodels' COPYRIGHT), T = 55, stored verbatim because the
    Rust golden needs its input series;
  * seeded systems (``numpy.random.default_rng``): ``sim_k2`` (cointegrated,
    an I(1) and an I(0) regressor, T = 160), ``sim_null`` (independent
    random walks, T = 200), ``sim_small`` (cointegrated, T = 40),
    ``sim_k10`` / ``sim_k11`` (10 and 11 regressors, T = 150).

Grade: third-party golden (statsmodels 0.15.0) on the estimates, F and t
(two routes), long run, fit statistics (OLS route) and selection (IC table);
the critical values and p-values are statsmodels' tabulated asymptotic
surface at the corrected key (a transcription golden of a simulated
surface, not an exact distribution); no t-bound critical values,
finite-sample surfaces or q_j = 0 orders ship.
"""
from __future__ import annotations

import json
import re
from fractions import Fraction
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.tsa.ardl import pss_critical_values as pss
from statsmodels.tsa.ardl.model import _pss_pvalue, _pss_simulate

warnings.simplefilter("ignore")

OUT = Path(__file__).with_name("ardl.json")
ROUTE_B_DEV = []
EXACT_DEV = []
TREND = {1: "n", 2: "c", 3: "c", 4: "ct", 5: "ct"}
N_DET = {"n": 0, "c": 1, "ct": 2}
LEVELS = [0.90, 0.95, 0.99, 0.999]

# a numeric compare: the string "0.9" >= "0.15" would pass
assert tuple(int(v) for v in statsmodels.__version__.split(".")[:2]) >= (0, 15), statsmodels.__version__


# --------------------------------------------------------------------- data

def systems():
    out = {}
    md = sm.datasets.macrodata.load_pandas().data
    out["macrodata"] = dict(
        y=100.0 * np.log(md["realcons"].to_numpy(dtype=float)),
        x=np.column_stack([100.0 * np.log(md["realgdp"].to_numpy(dtype=float)),
                           md["realint"].to_numpy(dtype=float)]),
        source="statsmodels macrodata: y = 100 log realcons, x = [100 log realgdp, realint]",
    )
    d = sm.datasets.danish_data.load_pandas().data
    out["danish"] = dict(
        y=d["lrm"].to_numpy(dtype=float),
        x=d[["lry", "ibo", "ide"]].to_numpy(dtype=float),
        source="statsmodels danish_data (Johansen-Juselius 1990, public domain): lrm on [lry, ibo, ide]",
    )
    # seeded cointegrated system, k_x = 2: an I(1) and an I(0) regressor.
    rng = np.random.default_rng(20261008)
    T = 160
    x1 = np.cumsum(rng.standard_normal(T))
    x2 = np.zeros(T)
    e2 = rng.standard_normal(T)
    for t in range(1, T):
        x2[t] = 0.5 * x2[t - 1] + e2[t]
    y = np.zeros(T)
    u = rng.standard_normal(T)
    for t in range(1, T):
        y[t] = (y[t - 1] - 0.3 * (y[t - 1] - 1.0 - 0.8 * x1[t - 1] + 0.5 * x2[t - 1])
                + 0.4 * (x1[t] - x1[t - 1]) + 0.5 * u[t])
    out["sim_k2"] = dict(y=y, x=np.column_stack([x1, x2]),
                         source="seeded: y ECM on an I(1) and an AR(0.5) regressor, alpha = -0.3, "
                                "theta = [0.8, -0.5], const 1.0 (rng 20261008)")
    # seeded null: independent random walks.
    rng = np.random.default_rng(20261009)
    lv = np.cumsum(rng.standard_normal((200, 2)), axis=0)
    out["sim_null"] = dict(y=lv[:, 0].copy(), x=lv[:, 1:].copy(),
                           source="seeded: independent random walks (rng 20261009)")
    # small sample, cointegrated, k_x = 1.
    rng = np.random.default_rng(20261010)
    T = 40
    x = np.cumsum(rng.standard_normal(T))
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = y[t - 1] - 0.5 * (y[t - 1] - 2.0 * x[t - 1]) + 0.3 * (x[t] - x[t - 1]) + rng.standard_normal()
    out["sim_small"] = dict(y=y, x=x[:, None], source="seeded: cointegrated, T = 40 (rng 20261010)")
    # 10 and 11 regressors.
    for k, seed in ((10, 20261011), (11, 20261012)):
        rng = np.random.default_rng(seed)
        T = 150
        X = np.cumsum(rng.standard_normal((T, k)), axis=0)
        w = np.linspace(0.5, -0.5, k)
        y = np.zeros(T)
        for t in range(1, T):
            y[t] = y[t - 1] - 0.4 * (y[t - 1] - X[t - 1] @ w) + rng.standard_normal()
        out[f"sim_k{k}"] = dict(y=y, x=X, source=f"seeded: {k} random-walk regressors (rng {seed})")
    for s in out.values():
        s["T"] = int(len(s["y"]))
        s["k_x"] = int(s["x"].shape[1])
    return out


# ------------------------------------------------------------ PSS surface

def surface_p(stat, k, case, i1):
    """sf-based PSS response-surface p-value at key (k, case, i1)."""
    key = (k, case, i1)
    if stat <= 0.0:
        return 1.0
    a = pss.small_p[key] if stat > pss.stat_star[key] else pss.large_p[key]
    ls = np.log(stat)
    z = sum(c * ls ** i for i, c in enumerate(a))
    return float(stats.norm.sf(z))


def crit_and_p(stat, k, case):
    if k > 10:
        nan4 = [float("nan")] * 4
        return nan4, nan4, float("nan"), float("nan")
    lo = [float(v) for v in pss.crit_vals[(k, case, False)]]
    up = [float(v) for v in pss.crit_vals[(k, case, True)]]
    pl = surface_p(stat, k, case, False)
    pu = surface_p(stat, k, case, True)
    assert abs(pl - float(_pss_pvalue(stat, k, case, False))) <= 1e-15
    assert abs(pu - float(_pss_pvalue(stat, k, case, True))) <= 1e-15
    return lo, up, pl, pu


def verdict(pl, pu, alpha):
    if np.isnan(pu) or np.isnan(pl):
        return "unavailable"
    if pu < alpha:
        return "level_relationship"
    if pl >= alpha:
        return "no_level_relationship"
    return "inconclusive"


# ------------------------------------------------------------- one fit

def _solve_exact(A, rhs):
    """Gauss-Jordan elimination in exact rationals (A is square, full rank)."""
    n = len(A)
    M = [A[i][:] + [rhs[i]] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                fct = M[r][c]
                M[r] = [a - fct * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def exact_f_and_t(Z, dy, rows):
    """The PSS F and the t on y.L1 in EXACT rational arithmetic.

    The design and the regressand are binary floats, so ``Fraction`` holds
    them exactly; the normal equations are then solved without rounding and
    F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - m)) (restricted model: the
    restricted columns dropped), t^2 = b_pi^2 / (s^2 [(Z'Z)^-1]_pi,pi),
    rounded to double only at the end. This is the documented-formula
    referee for the cases where the two floating-point statsmodels routes
    disagree beyond 1e-10 (an ill-conditioned restriction block, e.g. the
    constant beside levels near 800 in case 2): it shows which one rounds.
    """
    n, m = Z.shape
    Zf = [[Fraction(float(v)) for v in r] for r in Z]
    yf = [Fraction(float(v)) for v in dy]
    A = [[sum(Zf[i][a] * Zf[i][b] for i in range(n)) for b in range(m)] for a in range(m)]
    c = [sum(Zf[i][a] * yf[i] for i in range(n)) for a in range(m)]
    yy = sum(v * v for v in yf)
    b = _solve_exact(A, c)
    ssr_u = yy - sum(bi * ci for bi, ci in zip(b, c))
    keep = [j for j in range(m) if j not in rows]
    Ar = [[A[a][bb] for bb in keep] for a in keep]
    cr = [c[a] for a in keep]
    br = _solve_exact(Ar, cr) if keep else []
    ssr_r = yy - sum(bi * ci for bi, ci in zip(br, cr))
    s2 = ssr_u / (n - m)
    F = (ssr_r - ssr_u) / len(rows) / s2
    return F, s2, A, b



def ic_values(llf, nobs, m):
    return dict(aic=-2 * llf + 2.0 * (m + 1),
                bic=-2 * llf + np.log(nobs) * (m + 1),
                hqic=-2 * llf + 2.0 * np.log(np.log(nobs)) * (m + 1))


def rename(name):
    return re.sub(r"x(\d+)", lambda mo: f"x{int(mo.group(1)) + 1}", name)


def restriction_rows(case, n_det, k):
    """Indices (UECM column order) of the bounds-test restriction."""
    lv = list(range(n_det, n_det + 1 + k))
    if case == 2:
        return [0] + lv
    if case == 4:
        return [1] + lv
    return lv


def fit_case(y, X, p, q, case, alpha):
    T, k = X.shape
    trend = TREND[case]
    n_det = N_DET[trend]
    order = {j: int(q[j]) for j in range(k)}
    mod = UECM(y, p, X, order, trend=trend)
    res = mod.fit()
    Z = np.asarray(mod._x)
    dy = np.asarray(mod._y)
    h = int(mod.hold_back)
    assert h == max(p, max(q))
    nobs, m = Z.shape
    assert nobs == T - h and m == n_det + p + sum(qq + 1 for qq in q)
    b = np.asarray(res.params)
    resid = dy - Z @ b
    assert np.max(np.abs(resid - np.asarray(res.resid))) < 1e-12
    if n_det == 2:
        assert np.allclose(Z[:, 1], np.arange(h, T) + 1.0)  # trend = t + 1, full sample
    ols = OLS(dy, Z).fit()
    assert np.allclose(ols.params, b, rtol=1e-11, atol=1e-13)
    cov = np.asarray(res.cov_params())
    assert np.allclose(np.asarray(ols.cov_params()), cov, rtol=1e-10, atol=1e-16)
    se = np.asarray(res.bse)
    tv = np.asarray(res.tvalues)
    ssr = float(resid @ resid)
    llf = float(ols.llf)
    assert abs(llf - (-nobs / 2.0 * (np.log(2 * np.pi * ssr / nobs) + 1.0))) < 1e-10 * abs(llf) + 1e-12
    df_resid = int(round(ols.df_resid))
    assert df_resid == nobs - m
    pvals = np.asarray(ols.pvalues)
    assert np.allclose(pvals, 2 * stats.t.sf(np.abs(tv), df_resid), rtol=1e-12, atol=1e-300)
    ics = ic_values(llf, nobs, m)

    # DEVIATION 2 canary: statsmodels' reported nobs is T - p.
    sm_nobs = int(res.nobs)
    assert sm_nobs == T - p
    if p >= max(q):
        assert sm_nobs == nobs
        assert abs(float(res.llf) - llf) < 1e-9 * abs(llf)
        assert abs(float(res.bic) - ics["bic"]) < 1e-9 * abs(ics["bic"])
        assert np.allclose(np.asarray(res.pvalues), pvals, rtol=1e-12, atol=1e-300)
    else:
        assert sm_nobs == nobs + (h - p)
        assert abs(float(res.llf) - llf) > 1e-6

    # route (a): UECM bounds_test; t on y.L1
    if k + 1 <= 10:
        bt = res.bounds_test(case=case)
    else:  # statsmodels switches to simulation; only the statistic is used
        bt = res.bounds_test(case=case, asymptotic=False, nsim=500, rng=0)
    F_a = float(bt.statistic)
    rows = restriction_rows(case, n_det, k)
    R = np.zeros((len(rows), m))
    for i, c in enumerate(rows):
        R[i, c] = 1.0
    rb = R @ b
    F_own = float(rb @ np.linalg.solve(R @ cov @ R.T, rb) / len(rows))
    assert abs(F_own - F_a) <= 1e-10 * F_a
    t_a = float(tv[n_det])
    F_x, s2_x, A_x, b_x = exact_f_and_t(Z, dy, rows)
    e_pi = [Fraction(int(j == n_det)) for j in range(m)]
    inv_pp = _solve_exact(A_x, e_pi)[n_det]
    t2 = b_x[n_det] ** 2 / (s2_x * inv_pp)
    t_x = float(np.sign(float(b_x[n_det]))) * float(np.sqrt(float(t2)))
    F_x = float(F_x)
    EXACT_DEV.append((abs(F_a - F_x) / F_x, abs(t_a - t_x) / abs(t_x)))
    assert abs(F_a - F_x) <= 1e-9 * F_x, (F_a, F_x)
    assert abs(t_a - t_x) <= 1e-9 * abs(t_x), (t_a, t_x)

    # route (b): levels ARDL with the affine restriction
    a = ARDL(y, p, X, order, trend=trend).fit()
    names = list(a.model.exog_names)
    K = len(names)
    bl = np.asarray(a.params)
    covl = np.asarray(a.cov_params())
    Rl, rl = [], []
    phi = np.zeros(K)
    phi[[names.index(f"y.L{i}") for i in range(1, p + 1)]] = 1.0
    Rl.append(phi)
    rl.append(1.0)
    for j in range(k):
        r = np.zeros(K)
        r[[names.index(f"x{j}.L{l}") for l in range(q[j] + 1)]] = 1.0
        Rl.append(r)
        rl.append(0.0)
    if case == 2:
        r = np.zeros(K)
        r[names.index("const")] = 1.0
        Rl.append(r)
        rl.append(0.0)
    if case == 4:
        r = np.zeros(K)
        r[names.index("trend")] = 1.0
        Rl.append(r)
        rl.append(0.0)
    Rl = np.array(Rl)
    ft = a.f_test((Rl, np.array(rl)))
    F_b = float(np.squeeze(ft.fvalue))
    t_b = float((phi @ bl - 1.0) / np.sqrt(phi @ covl @ phi))
    ROUTE_B_DEV.append((abs(F_b - F_a) / F_a, abs(t_b - t_a) / abs(t_a), np.linalg.cond(np.asarray(a.model._x))))
    # the levels design is far worse conditioned than the UECM's (cond up
    # to ~6e5 here): the two routes agree to 1.5e-9 relative on F at worst.
    assert abs(F_b - F_a) <= 1e-8 * F_a, (F_a, F_b)
    assert abs(t_b - t_a) <= 1e-8 * abs(t_a), (t_a, t_b)
    assert abs(float(a.ssr) - ssr) <= 1e-9 * ssr

    # long run
    ci = np.asarray(res.ci_params)
    ci_bse = np.asarray(res.ci_bse)
    ci_p = np.asarray(res.ci_pvalues)
    keep = [i for i in range(n_det + 1 + k) if i != n_det]
    lr = -ci[keep]
    lr_se = ci_bse[keep]
    lr_t = lr / lr_se
    lr_p = 2 * stats.norm.sf(np.abs(lr_t))
    assert np.all(np.abs(lr_p - ci_p[keep]) <= 1e-15)
    det_names = ["const", "trend"][:n_det]
    lr_names = det_names + [f"x{j + 1}" for j in range(k)]

    lo, up, pl, pu = crit_and_p(F_a, k, case)
    rec = dict(
        lags=int(p), order=[int(v) for v in q], case=case, trend=trend, n_x=k, n_det=n_det,
        n_params=m, n_restrictions=len(rows), nobs=nobs, n_total=T, hold_back=h, df_resid=df_resid,
        param_names=[rename(nm) for nm in mod.exog_names],
        params=b.tolist(), se=se.tolist(), tvalues=tv.tolist(), pvalues=pvals.tolist(),
        cov=cov.tolist(), resid=resid.tolist(), ssr=ssr, llf=llf, **ics,
        statistic=F_a, statistic_levels=F_b, statistic_exact=F_x,
        t_statistic=t_a, t_statistic_levels=t_b, t_statistic_exact=t_x,
        adjustment=float(b[n_det]),
        long_run=lr.tolist(), long_run_se=lr_se.tolist(), long_run_tvalues=lr_t.tolist(),
        long_run_pvalues=lr_p.tolist(), long_run_names=lr_names,
        crit_lower=lo, crit_upper=up, p_value_lower=pl, p_value_upper=pu,
        alpha=alpha, verdict=verdict(pl, pu, alpha),
        sm_reported=dict(nobs=sm_nobs, llf=float(res.llf), df_resid=float(res.df_resid),
                         bic=float(res.bic), pvalues=np.asarray(res.pvalues).tolist()),
    )
    if k + 1 <= 10:
        # DEVIATION 1 canary: statsmodels keys the table at n_x + 1.
        cv = bt.critical_values
        sm_lo = cv["lower"].to_numpy(dtype=float)
        sm_up = cv["upper"].to_numpy(dtype=float)
        assert np.array_equal(sm_lo, np.asarray(pss.crit_vals[(k + 1, case, False)]))
        assert np.array_equal(sm_up, np.asarray(pss.crit_vals[(k + 1, case, True)]))
        sm_pl = float(bt.pvalue["lower"])
        sm_pu = float(bt.pvalue["upper"])
        assert abs(sm_pl - surface_p(F_a, k + 1, case, False)) <= 1e-15
        assert abs(sm_pu - surface_p(F_a, k + 1, case, True)) <= 1e-15
        rec["canary_key"] = dict(sm_crit_lower=sm_lo.tolist(), sm_crit_upper=sm_up.tolist(),
                                 sm_p_value_lower=sm_pl, sm_p_value_upper=sm_pu, sm_key_k=k + 1)
    return rec


# ------------------------------------------------------------ selection

def select_case(y, X, case, ic, lags=None, order=None, max_lag=None, max_order=None, alpha=0.05):
    T, k = X.shape
    trend = TREND[case]
    P = lags if lags is not None else (max_lag if max_lag is not None else 4)
    Q = max(order) if order is not None else (max_order if max_order is not None else 4)
    maxorder = {j: int(order[j]) for j in range(k)} if order is not None else Q
    sel = ardl_select_order(y, P, X, maxorder, trend=trend, ic=ic)
    table = getattr(sel, ic)
    qs = order if order is not None else [Q] * k
    full = (P + 1) * int(np.prod([qq + 2 for qq in qs]))
    assert len(table) == full, (len(table), full)  # no IC collisions collapsed the table
    adm = []
    for val, key in table.items():
        ylag, xo = key[0], key[1]
        if ylag is None or ylag < 1:
            continue
        if lags is not None and ylag != lags:
            continue
        xv = [xo[j] for j in range(k)]
        if any(v is None or v < 1 for v in xv):
            continue
        if order is not None and xv != list(order):
            continue
        adm.append((float(val), int(ylag), [int(v) for v in xv]))
    n_lag = 1 if lags is not None else P
    n_ord = 1 if order is not None else Q ** k
    assert len(adm) == n_lag * n_ord, (len(adm), n_lag * n_ord)
    adm.sort(key=lambda r: r[0])
    best, runner = adm[0], adm[1]
    margin = runner[0] - best[0]
    assert margin > 1e-6, margin
    p, q = best[1], best[2]
    rec = fit_case(y, X, p, q, case, alpha)
    # the winner's IC on the common sample, recomputed with the deterministics
    # in the regression instead of partialled out (FWL): same up to ~1e-13
    hstar = max(P, Q)
    refit = ARDL(y, p, X, {j: q[j] for j in range(k)}, trend=trend, hold_back=hstar)
    rz, ry = np.asarray(refit._x), np.asarray(refit._y)
    e = ry - rz @ np.linalg.lstsq(rz, ry, rcond=None)[0]
    n_c = len(ry)
    assert n_c == T - hstar
    llf_c = -n_c / 2.0 * (np.log(2 * np.pi * (e @ e) / n_c) + 1.0)
    ic_c = ic_values(llf_c, n_c, rz.shape[1])[ic]
    assert abs(ic_c - best[0]) < 1e-9 * abs(best[0]) + 1e-9, (ic_c, best[0])
    rec.update(selected=True, ic=ic, ic_value=best[0], runner_up=dict(lags=runner[1], order=runner[2], ic_value=runner[0]),
               ic_margin=margin, max_lag=int(P), max_order=int(Q), n_models=len(adm),
               common_nobs=int(T - hstar))
    unrestricted = sel.model.ardl_order
    rec["sm_unrestricted_pick"] = [int(v) for v in unrestricted]
    try:
        UECM.from_ardl(sel.model)
        rec["sm_from_ardl"] = "ok"
    except ValueError as exc:
        rec["sm_from_ardl"] = f"raises ValueError: {exc}"
    return rec


# --------------------------------------------------------------- cases

def clean(v):
    """NaN -> null (JSON has no NaN; the Rust and Python loaders read null as NaN)."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def main():
    syst = systems()
    cases = []

    def pattern(k):
        if k == 1:
            return [2]
        return [(1, 3, 2)[j % 3] for j in range(k)]

    # fixed orders: every case x three specifications
    for name in ("macrodata", "danish", "sim_k2", "sim_null", "sim_small"):
        s = syst[name]
        k = s["k_x"]
        for case in (1, 2, 3, 4, 5):
            for p, q in ((1, [1] * k), (3, [3] * k), (2, pattern(k))):
                alpha = {1: 0.05, 2: 0.10, 3: 0.05, 4: 0.01, 5: 0.05}[case]
                rec = fit_case(s["y"], s["x"], p, q, case, alpha)
                rec.update(system=name, lags_arg=p, order_arg=q, ic_arg=None, max_lag_arg=None,
                           max_order_arg=None, selected=False, ic="bic", ic_value=float("nan"),
                           max_lag=p, max_order=max(q), n_models=1)
                cases.append(rec)

    # searched specifications
    for name in ("macrodata", "danish", "sim_k2", "sim_small", "sim_null"):
        s = syst[name]
        y, X, k = s["y"], s["x"], s["k_x"]
        specs = [
            dict(case=3, ic="bic"),
            dict(case=3, ic="aic"),
            dict(case=3, ic="hqic"),
            dict(case=3, ic="bic", max_lag=2, max_order=2),
            dict(case=5, ic="aic", max_lag=3, max_order=2),
            dict(case=1, ic="bic"),
            dict(case=3, ic="bic", order=pattern(k)),                 # lags-only search
            dict(case=2, ic="aic", order=[1] * k, max_lag=2),        # lags-only, capped
            dict(case=3, ic="bic", lags=2),                           # order-only search
            dict(case=4, ic="hqic", lags=1, max_order=2),             # order-only, capped
        ]
        if name == "sim_null":
            specs = specs[:1] + specs[4:6]
        for sp in specs:
            sp = dict(sp)
            case = sp.pop("case")
            ic = sp.pop("ic")
            rec = select_case(y, X, case, ic, **sp)
            rec.update(system=name, lags_arg=sp.get("lags"), order_arg=sp.get("order"), ic_arg=ic,
                       max_lag_arg=sp.get("max_lag"), max_order_arg=sp.get("max_order"))
            cases.append(rec)

    # 10 regressors (table edge) and 11 (no table: NaN + crit_note)
    for name, case in (("sim_k10", 3), ("sim_k10", 5), ("sim_k11", 3)):
        s = syst[name]
        k = s["k_x"]
        rec = fit_case(s["y"], s["x"], 1, [1] * k, case, 0.05)
        rec.update(system=name, lags_arg=1, order_arg=[1] * k, ic_arg=None, max_lag_arg=None,
                   max_order_arg=None, selected=False, ic="bic", ic_value=float("nan"),
                   max_lag=1, max_order=1, n_models=1)
        cases.append(rec)

    # ---- the macrodata headline (degenerate case #1) and the Danish story
    head = next(c for c in cases if c["system"] == "macrodata" and c["selected"] and c["case"] == 3
                and c["ic_arg"] == "bic" and c["lags_arg"] is None and c["order_arg"] is None
                and c["max_lag_arg"] is None)
    assert head["lags"] == 1 and head["order"] == [1, 1]
    dan = next(c for c in cases if c["system"] == "danish" and c["selected"] and c["case"] == 3
               and c["ic_arg"] == "bic" and c["lags_arg"] is None and c["order_arg"] is None
               and c["max_lag_arg"] is None)
    assert dan["sm_unrestricted_pick"] == [3, 1, 0], dan["sm_unrestricted_pick"]
    assert dan["sm_from_ardl"].startswith("raises"), dan["sm_from_ardl"]
    assert dan["lags"] == 3 and dan["order"] == [1, 1, 1]
    stories = dict(
        macrodata_bic_case3=dict(lags=head["lags"], order=head["order"], statistic=head["statistic"],
                                 t_statistic=head["t_statistic"], p_value_upper=head["p_value_upper"],
                                 p_value_lower=head["p_value_lower"], crit95=[head["crit_lower"][1], head["crit_upper"][1]],
                                 sm_p_value_upper=head["canary_key"]["sm_p_value_upper"],
                                 sm_crit95=[head["canary_key"]["sm_crit_lower"][1], head["canary_key"]["sm_crit_upper"][1]]),
        danish_bic_case3=dict(unrestricted=dan["sm_unrestricted_pick"], from_ardl=dan["sm_from_ardl"],
                              admissible=[dan["lags"], dan["order"]], ic_value=dan["ic_value"],
                              margin=dan["ic_margin"]),
    )

    # ---- statsmodels quirks recorded as data
    d = syst["danish"]
    u0 = UECM(d["y"], 0, d["x"], 1, trend="c")
    u1 = UECM(d["y"], 1, d["x"], 1, trend="c")
    lags0_same = bool(np.array_equal(np.asarray(u0._x), np.asarray(u1._x)))
    assert lags0_same and list(u0.exog_names) == list(u1.exog_names)
    try:
        UECM(d["y"], 3, d["x"], {0: 1, 1: 0, 2: 2}, trend="c")
        order0 = "accepted"
    except ValueError as exc:
        order0 = f"raises ValueError: {exc}"
    assert order0.startswith("raises")
    quirks = dict(uecm_lags0_builds_lags1_design=lags0_same, uecm_order0=order0)

    # ---- PSS table (every constant) and a p-value grid on every key
    table = []
    grid = []
    for key in sorted(pss.crit_vals):
        kk, case, i1 = key
        table.append(dict(k=kk, case=case, i1=i1, crit=[float(v) for v in pss.crit_vals[key]],
                          large_p=[float(v) for v in pss.large_p[key]],
                          small_p=[float(v) for v in pss.small_p[key]],
                          stat_star=float(pss.stat_star[key])))
        ss = float(pss.stat_star[key])
        Fs = [0.05, 0.3, 1.0, ss * (1 - 1e-9), ss, ss * (1 + 1e-9), 2.5, 5.0, 10.0, 25.0, 80.0]
        ps = []
        for F in Fs:
            pv = surface_p(F, kk, case, i1)
            assert abs(pv - float(_pss_pvalue(F, kk, case, i1))) <= 1e-15
            ps.append(pv)
        grid.append(dict(k=kk, case=case, i1=i1, stat=Fs, p=ps))
    assert len(table) == 100

    # ---- canary_sim: statsmodels' own simulator at its key 1 + n_x
    sim = []
    for kx in (1, 3):
        cv, _ = _pss_simulate(5.0, 1 + kx, 3, nobs=1000, nsim=40000, rng=12345)
        lo = cv["lower"].to_numpy(dtype=float)
        up = cv["upper"].to_numpy(dtype=float)
        t_k = (pss.crit_vals[(kx, 3, False)][1], pss.crit_vals[(kx, 3, True)][1])
        t_k1 = (pss.crit_vals[(kx + 1, 3, False)][1], pss.crit_vals[(kx + 1, 3, True)][1])
        # 0.10, not the 0.08 the design spec proposed: the k_x = 1 lower bound
        # sits 0.085 from the table at this seed (40 000 draws at nobs = 1000);
        # the n_x + 1 row is more than 0.3 away in every entry.
        assert abs(lo[1] - t_k[0]) < 0.10 and abs(up[1] - t_k[1]) < 0.10, (lo, up, t_k)
        assert abs(lo[1] - t_k1[0]) > 0.3 and abs(up[1] - t_k1[1]) > 0.3, (lo, up, t_k1)
        sim.append(dict(n_x=kx, sim_key_k=1 + kx, case=3, nobs=1000, nsim=40000, rng=12345,
                        sim_lower=lo.tolist(), sim_upper=up.tolist(),
                        table_k_lower95=float(t_k[0]), table_k_upper95=float(t_k[1]),
                        table_k1_lower95=float(t_k1[0]), table_k1_upper95=float(t_k1[1])))

    # ---- published: PSS (2001) Table CI(iii) (case III) and CI(v) (case V),
    # 5% asymptotic F bounds, k = 1..4 regressors. Transcribed by hand; NOT
    # machine-verified in-container (the PDF cannot be fetched) — a reviewer
    # verifies it against the paper. The 0.05 tolerance against the surface
    # catches a gross slip, not a 0.01 one.
    published = dict(
        citation="Pesaran, Shin & Smith (2001), Journal of Applied Econometrics 16(3), 289-326, "
                 "Table CI(iii) (case III) and Table CI(v) (case V), 95% columns",
        verified_against_pdf=False,
        rows=[
            dict(case=3, k=1, lower=4.94, upper=5.73), dict(case=3, k=2, lower=3.79, upper=4.85),
            dict(case=3, k=3, lower=3.23, upper=4.35), dict(case=3, k=4, lower=2.86, upper=4.01),
            dict(case=5, k=1, lower=6.56, upper=7.30), dict(case=5, k=2, lower=4.87, upper=5.85),
            dict(case=5, k=3, lower=4.01, upper=5.07), dict(case=5, k=4, lower=3.47, upper=4.57),
        ],
    )
    for r in published["rows"]:
        lo = pss.crit_vals[(r["k"], r["case"], False)][1]
        up = pss.crit_vals[(r["k"], r["case"], True)][1]
        assert abs(lo - r["lower"]) < 0.05 and abs(up - r["upper"]) < 0.05, (r, lo, up)

    fixture = {
        "_meta": {
            "reference": "statsmodels.tsa.ardl.{UECM, ARDL, ardl_select_order, pss_critical_values} "
                         "+ statsmodels OLS + scipy.stats",
            "statsmodels_version": statsmodels.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "python_version": sys.version.split()[0],
            "generator": Path(__file__).name,
            "note": "see the module docstring: DEVIATION 1 (PSS table key k = n_x, not 1 + n_x) and "
                    "DEVIATION 2 (nobs = T - max(p, q), not T - p) are documented, not inherited",
            "levels": LEVELS,
        },
        "systems": {n: dict(y=s["y"].tolist(), x=s["x"].tolist(), T=s["T"], k_x=s["k_x"], source=s["source"])
                    for n, s in syst.items()},
        "cases": cases,
        "stories": stories,
        "quirks": quirks,
        "pss_table": table,
        "pss_pvalue_grid": grid,
        "canary_sim": sim,
        "published": published,
    }
    OUT.write_text(json.dumps(clean(fixture), indent=1, allow_nan=False) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")
    ex = np.array(EXACT_DEV)
    print("statsmodels route (a) vs exact rationals: max rel |dF| =", ex[:, 0].max(), " max rel |dt| =", ex[:, 1].max())
    dev = np.array(ROUTE_B_DEV)
    print("route (b) vs (a): max rel |dF| =", dev[:, 0].max(), " max rel |dt| =", dev[:, 1].max(),
          " max cond(levels design) =", dev[:, 2].max())
    print("macrodata headline:", stories["macrodata_bic_case3"])
    print("danish story:", stories["danish_bic_case3"])
    for s in sim:
        print("canary_sim:", s)


if __name__ == "__main__":
    main()
