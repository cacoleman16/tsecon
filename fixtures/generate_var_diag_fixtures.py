"""Golden fixtures for the VAR residual diagnostics `var_diagnostics` and
the lag-order selection binding `var_select_order` (crate `tsecon-var`,
module `diagnostics.rs` / `select.rs`).

Reference: statsmodels `VARResults` (independent package) —

  * `test_whiteness(nlags, adjusted=False)` and `adjusted=True`: the
    multivariate Portmanteau statistics Q_h / adjusted Q_h, their
    k^2 (nlags - p) degrees of freedom and chi-squared p-values
    (Lütkepohl 2005, section 4.4.3);
  * `test_normality()`: the multivariate Jarque-Bera omnibus statistic,
    its 2k degrees of freedom and p-value (statsmodels orthogonalises the
    centred residuals with the LOWER CHOLESKY factor of resid'resid / T);
  * `roots` (moduli, sorted descending — the reciprocal characteristic
    roots) and `is_stable()`;
  * `VAR.select_order(maxlags)`: the AIC / BIC / HQIC / FPE table on the
    common sample and the selected orders.

The skewness and kurtosis COMPONENTS of the normality test (`lam_skew`,
`lam_kurt`, and the per-component `b1`, `b2`) are not returned by
statsmodels; they are recomputed here by transcribing the lines of
statsmodels' own `test_normality` (documented formula, same
orthogonalisation), and `lam_skew + lam_kurt` is asserted equal to the
returned omnibus statistic before anything is stored.

Data: the seeded simulated VAR(2) of var_cf.json (k = 3; the same
generator, the same seed, stored again so each fixture is self-contained),
a second simulated VAR(2) driven by Student-t(4) innovations so the
normality test has something to reject, and the macro system of
var_cf.json (100 dlog realgdp, 400 dlog cpi, diff tbilrate —
transformations of statsmodels' bundled macrodata, not the dataset).

Residual NORMALITY beyond statsmodels (0.11.0) — three statistics on the same
column-centred residuals and ML covariance S = u'u / T, graded separately:

  * `normality` — the Cholesky form above (statsmodels; it is also JMulTi's
    block labelled "Lütkepohl (1993)").
  * `normality_symmetric` — the multivariate Jarque-Bera with the SYMMETRIC
    inverse root S^{-1/2} = H Lambda^{-1/2} H' in place of the Cholesky
    factor, plain b1 and excess b2, NO further transforms. This is what JMulTi
    prints under "Reference: Doornik & Hansen (1994)". INDEPENDENT-PROGRAM
    golden at printed precision: JMulTi's own output files, bundled in the
    statsmodels 0.15.0 wheel (tsa/vector_ar/tests/JMulTi_results/*_diag.txt),
    on all 16 specifications statsmodels ships — the macrodata VAR(4) in
    levels with trend nc/ncs/c/cs/ct/cst and the e6 VECM(k_ar_diff=3,
    coint_rank=1) with deterministic nc/ncs/co/cos/colo/coslo/ci/cis/cili/
    cisli (file tags from statsmodels' own `dt_s_tup_to_string`, which puts
    the seasonal "s" after the first two characters: coslo and cisli, not
    colos and cilis). The residuals are statsmodels' fits of those
    specifications (calibrated: the Cholesky transcription on them must hit
    the "Lütkepohl (1993)" block printed in the same file, and the per-
    variable "JARQUE-BERA TEST" rows — the univariate JB at k = 1). Printed
    to 4 decimals, so the pin is |diff| <= 5e-5 on statistics AND p-values;
    each leg's margin (5e-5 minus its worst gap) is stored separately. The
    1e-10 pins come from the transcriptions stored next to them.
  * `doornik_hansen` — the Doornik-Hansen (2008) omnibus test, transcribed
    from the paper (Oxford Bulletin of Economics and Statistics 70(s1),
    927-939): correlation-matrix symmetric root, D'Agostino's skewness
    transform z1 and the Shenton-Bowman / Wilson-Hilferty kurtosis transform
    z2. DOCUMENTED-FORMULA golden — no installable package implements it
    (statsmodels, scipy, arch, linearmodels and scikit-learn were searched).
    Two legs make it more than a transcription: z1 equals
    `scipy.stats.skewtest` applied to the transformed columns (asserted at
    1e-10 relative), and the kurtosis coefficients (a, c, kappa) satisfy the
    exact mean-matching identity 2 kappa (E b2 - 1 - E b1) = 2 (a + c E b1)
    with E b1 = 6(n-2)/((n+1)(n+3)), E b2 = 3(n-1)/(n+1), asserted in exact
    rational arithmetic (fractions.Fraction) for n in {8, 9, 10, 20, 50, 200,
    1000}. The common denominator d cancels from that identity; the crate's
    null Monte Carlo of z2 guards it.

The JMulTi residual matrices are derived values (fitted residuals), not the
datasets. NOTE on regeneration: the six original cases move by up to ~2e-12
relative across NumPy/SciPy/Python versions (BLAS and libm rounding), well
inside their 1e-10 pins; the series themselves are bit-stable.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden tests parse to identical
bits (serde_json `float_roundtrip`).

Run:  .venv/bin/python fixtures/generate_var_diag_fixtures.py
"""

from __future__ import annotations

import json
import platform
import re
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.datasets.interest_inflation.data as e6_data
import statsmodels.datasets.macrodata.data as macro_data
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_var_output import (
    dt_s_tup_to_string as var_tag,
)
from statsmodels.tsa.vector_ar.tests.JMulTi_results.parse_jmulti_vecm_output import (
    dt_s_tup_to_string as vecm_tag,
)
from statsmodels.tsa.vector_ar.vecm import VECM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_var_cf_fixtures import macro_system, sim_var  # noqa: E402

OUT = Path(__file__).resolve().parent / "var_diag.json"
JMULTI_DIR = Path(statsmodels.__file__).resolve().parent / "tsa" / "vector_ar" / "tests" / "JMulTi_results"
PRINTED_TOL = 5e-5  # JMulTi prints 4 decimals


def sim_var_t(seed, T, c, a_list, chol, df, burn=100):
    """VAR(p) with Student-t(df) innovations (scaled to unit variance)."""
    rng = np.random.default_rng(seed)
    k = len(c)
    p = len(a_list)
    n = T + burn + p
    y = np.zeros((n, k))
    c = np.asarray(c)
    a_list = [np.asarray(a) for a in a_list]
    chol = np.asarray(chol)
    scale = np.sqrt((df - 2.0) / df)
    for t in range(p, n):
        v = c.copy()
        for i, a in enumerate(a_list, start=1):
            v = v + a @ y[t - i]
        y[t] = v + chol @ (scale * rng.standard_t(df, size=k))
    return y[burn + p:]


def normality_components(res):
    """Transcription of statsmodels.tsa.vector_ar.var_model.test_normality."""
    resid_c = np.asarray(res.resid) - np.asarray(res.resid).mean(0)
    sig = resid_c.T @ resid_c / res.nobs
    Pinv = np.linalg.inv(np.linalg.cholesky(sig))
    w = Pinv @ resid_c.T
    b1 = (w ** 3).sum(1) / res.nobs
    b2 = (w ** 4).sum(1) / res.nobs - 3
    lam_skew = float(res.nobs * (b1 @ b1) / 6)
    lam_kurt = float(res.nobs * (b2 @ b2) / 24)
    return b1, b2, lam_skew, lam_kurt


def jb_block(w, n, k):
    """Multivariate Jarque-Bera of orthogonalised rows w (n x k): plain b1,
    excess b2, lambda_s = n b1'b1/6, lambda_k = n b2'b2/24 (Lütkepohl 2005,
    section 4.5), with chi-squared p-values."""
    b1 = (w ** 3).sum(0) / n
    b2 = (w ** 4).sum(0) / n - 3
    lam_s = float(n * (b1 @ b1) / 6)
    lam_k = float(n * (b2 @ b2) / 24)
    stat = lam_s + lam_k
    return {
        "statistic": stat,
        "pvalue": float(stats.chi2.sf(stat, 2 * k)),
        "df": 2 * k,
        "skewness": lam_s,
        "skewness_pvalue": float(stats.chi2.sf(lam_s, k)),
        "kurtosis": lam_k,
        "kurtosis_pvalue": float(stats.chi2.sf(lam_k, k)),
        "skewness_components": b1.tolist(),
        "kurtosis_components": b2.tolist(),
    }


def centred_ml_cov(u):
    uc = u - u.mean(0)
    return uc, uc.T @ uc / u.shape[0]


def normality_cholesky(u):
    """statsmodels' `test_normality` orthogonalisation: w_t = P^{-1} u~_t with
    P the lower Cholesky factor of the ML covariance of the centred residuals."""
    n, k = u.shape
    uc, s = centred_ml_cov(u)
    w = np.linalg.solve(np.linalg.cholesky(s), uc.T).T
    return jb_block(w, n, k)


def normality_symmetric(u):
    """JMulTi's "Doornik & Hansen (1994)" block: w_t = S^{-1/2} u~_t with the
    symmetric inverse root S^{-1/2} = H Lambda^{-1/2} H' of the ML covariance
    of the CENTRED residuals; no further transforms."""
    n, k = u.shape
    uc, s = centred_ml_cov(u)
    lam, h = np.linalg.eigh(s)
    w = uc @ (h @ np.diag(lam ** -0.5) @ h.T)
    return jb_block(w, n, k)


def dh2008(u):
    """Doornik & Hansen (2008), transcribed from the paper.

    r_t = H Lambda^{-1/2} H' V^{-1/2} (u_t - ubar), with V = diag(S) and
    C = V^{-1/2} S V^{-1/2} = H Lambda H' (correlation-matrix symmetric root);
    per column the plain moments m_q = mean(r^q), sqrt(b1) = m3 / m2^{3/2},
    b2 = m4 / m2^2, then (n = T)

      beta  = 3(n^2+27n-70)(n+1)(n+3) / ((n-2)(n+5)(n+7)(n+9))
      w2    = -1 + sqrt(2(beta-1)),  delta = 1/sqrt(log(sqrt(w2)))
      y     = sqrt(b1) sqrt((w2-1)(n+1)(n+3) / (12(n-2))),  z1 = delta asinh(y)
      d     = (n-3)(n+1)(n^2+15n-4)
      a     = (n-2)(n+5)(n+7)(n^2+27n-70) / (6d)
      c     = (n-7)(n+5)(n+7)(n^2+2n-5) / (6d)
      kappa = (n+5)(n+7)(n^3+37n^2+11n-313) / (12d)
      alpha = a + c b1,  chi = 2 kappa (b2 - 1 - b1)
      z2    = (cbrt(chi / (2 alpha)) - 1 + 1/(9 alpha)) sqrt(9 alpha)
      E     = z1'z1 + z2'z2 ~ chi2(2k)
    """
    n, k = u.shape
    uc, s = centred_ml_cov(u)
    vinv = np.diag(1.0 / np.sqrt(np.diag(s)))
    c_mat = vinv @ s @ vinv
    lam, h = np.linalg.eigh(c_mat)
    r = uc @ vinv @ (h @ np.diag(lam ** -0.5) @ h.T)
    m2 = (r ** 2).mean(0)
    m3 = (r ** 3).mean(0)
    m4 = (r ** 4).mean(0)
    rb1 = m3 / m2 ** 1.5
    b2 = m4 / m2 ** 2
    b1 = rb1 ** 2
    nf = float(n)
    beta = 3 * (nf**2 + 27 * nf - 70) * (nf + 1) * (nf + 3) / ((nf - 2) * (nf + 5) * (nf + 7) * (nf + 9))
    w2 = -1 + np.sqrt(2 * (beta - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
    y = rb1 * np.sqrt((w2 - 1) * (nf + 1) * (nf + 3) / (12 * (nf - 2)))
    z1 = delta * np.arcsinh(y)
    d = (nf - 3) * (nf + 1) * (nf**2 + 15 * nf - 4)
    a = (nf - 2) * (nf + 5) * (nf + 7) * (nf**2 + 27 * nf - 70) / (6 * d)
    c = (nf - 7) * (nf + 5) * (nf + 7) * (nf**2 + 2 * nf - 5) / (6 * d)
    kappa = (nf + 5) * (nf + 7) * (nf**3 + 37 * nf**2 + 11 * nf - 313) / (12 * d)
    alpha = a + c * b1
    chi = 2 * kappa * (b2 - 1 - b1)
    z2 = (np.cbrt(chi / (2 * alpha)) - 1 + 1 / (9 * alpha)) * np.sqrt(9 * alpha)
    # Leg 1: z1 is D'Agostino's skewness test of each transformed column.
    sk = stats.skewtest(r, axis=0).statistic
    assert np.all(np.abs(z1 - sk) <= 1e-10 * np.maximum(np.abs(sk), 1.0)), (z1, sk)
    e_skew = float(z1 @ z1)
    e_kurt = float(z2 @ z2)
    e = e_skew + e_kurt
    return {
        "statistic": e,
        "pvalue": float(stats.chi2.sf(e, 2 * k)),
        "df": 2 * k,
        "skewness": e_skew,
        "skewness_pvalue": float(stats.chi2.sf(e_skew, k)),
        "kurtosis": e_kurt,
        "kurtosis_pvalue": float(stats.chi2.sf(e_kurt, k)),
        "z_skewness": z1.tolist(),
        "z_kurtosis": z2.tolist(),
        "sqrt_b1": rb1.tolist(),
        "b2": b2.tolist(),
    }


def assert_mean_matching_identity():
    """Leg 2 of the DH 2008 kurtosis transform, in EXACT rationals: the gamma
    approximation's mean matching E[chi | b1] = 2 alpha, taken at the null
    moments E b1 = 6(n-2)/((n+1)(n+3)) and E b2 = 3(n-1)/(n+1), reads
    2 kappa (E b2 - 1 - E b1) == 2 (a + c E b1). It pins a, c and kappa
    against transcription typos (d cancels)."""
    ns = [8, 9, 10, 20, 50, 200, 1000]
    for n0 in ns:
        n = Fraction(n0)
        eb1 = Fraction(6) * (n - 2) / ((n + 1) * (n + 3))
        eb2 = Fraction(3) * (n - 1) / (n + 1)
        d = (n - 3) * (n + 1) * (n**2 + 15 * n - 4)
        a = (n - 2) * (n + 5) * (n + 7) * (n**2 + 27 * n - 70) / (6 * d)
        c = (n - 7) * (n + 5) * (n + 7) * (n**2 + 2 * n - 5) / (6 * d)
        kappa = (n + 5) * (n + 7) * (n**3 + 37 * n**2 + 11 * n - 313) / (12 * d)
        assert 2 * kappa * (eb2 - 1 - eb1) == 2 * (a + c * eb1), n0
    # D'Agostino's beta is exactly 3 at n = 7 (omega^2 = 1: the transform
    # degenerates), which is why the statistic needs n >= 8.
    n = Fraction(7)
    assert 3 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9)) == 3
    return ns


def univariate_jb_rows(u):
    """JMulTi's per-variable JARQUE-BERA rows: centred biased moments,
    s = m3 / m2^1.5, kurtosis = m4 / m2^2 (raw), JB = n (s^2/6 + (kurt-3)^2/24)."""
    n = u.shape[0]
    uc = u - u.mean(0)
    m2 = (uc ** 2).mean(0)
    s = (uc ** 3).mean(0) / m2 ** 1.5
    kt = (uc ** 4).mean(0) / m2 ** 2
    jb = n * (s ** 2 / 6 + (kt - 3) ** 2 / 24)
    return [{"statistic": float(jb[i]), "pvalue": float(stats.chi2.sf(jb[i], 2)),
             "skewness": float(s[i]), "kurtosis": float(kt[i])} for i in range(u.shape[1])]


def parse_normality(txt, which):
    """The "TESTS FOR NONNORMALITY" section of a JMulTi *_diag.txt: `which` is
    "dh" (the "Doornik & Hansen (1994)" block) or "lutkepohl"."""
    sec = txt.split("TESTS FOR NONNORMALITY")[1].split("***")[0]
    parts = sec.split("Reference:")
    blk = parts[1] if which == "dh" else parts[2]
    assert ("Doornik & Hansen (1994)" in blk) == (which == "dh"), blk[:60]

    def num(pat):
        return float(re.search(pat, blk).group(1))

    pv = [float(v) for v in re.findall(r"p-value:\s+([\d.]+)", blk)]
    assert len(pv) == 3, blk
    return {
        "joint": num(r"joint test statistic:\s+([\d.]+)"),
        "p_joint": pv[0],
        "df": num(r"degrees of freedom:\s+([\d.]+)"),
        "skew": num(r"skewness only:\s+([\d.]+)"),
        "p_skew": pv[1],
        "kurt": num(r"kurtosis only:\s+([\d.]+)"),
        "p_kurt": pv[2],
    }


def parse_jb_rows(txt):
    sec = txt.split("JARQUE-BERA TEST")[1].split("***")[0]
    rows = []
    for line in sec.strip().splitlines()[1:]:
        f = line.split()
        if not f:
            continue
        rows.append({"variable": f[0], "statistic": float(f[1]), "pvalue": float(f[2]),
                     "skewness": float(f[3]), "kurtosis": float(f[4])})
    return rows


def printed_gap(full, printed):
    """Worst |full - printed| over the statistics and over the p-values."""
    st = max(abs(full[a] - printed[b]) for a, b in
             (("statistic", "joint"), ("skewness", "skew"), ("kurtosis", "kurt")))
    pv = max(abs(full[a] - printed[b]) for a, b in
             (("pvalue", "p_joint"), ("skewness_pvalue", "p_skew"), ("kurtosis_pvalue", "p_kurt")))
    return st, pv


def jmulti_specs():
    """The 16 specifications statsmodels' JMulTi tests fit, with their file
    tags from statsmodels' own `dt_s_tup_to_string` (VAR and VECM rules)."""
    y = macro_data.load_pandas().data[macro_data.variable_names].astype(float).to_numpy()
    out = []
    for det, trend in (("nc", "n"), ("c", "c"), ("ct", "ct")):
        for seasons in (0, 4):
            exog = None
            if seasons:
                ex = np.zeros((seasons - 1, len(y)))
                for i in range(seasons - 1):
                    ex[i, i::seasons] = 1
                exog = ex.T
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = VAR(y, exog).fit(maxlags=4, trend=trend, method="ols")
            tag = var_tag((det, seasons))
            out.append((f"macrodata_jmulti_{tag}", "VAR",
                        {"data": "macrodata [realcons, realgdp, realinv] in levels",
                         "lags": 4, "trend": trend, "seasons": seasons},
                        np.asarray(res.resid), res))
    z = e6_data.load_pandas().data[["Dp", "R"]].astype(float).to_numpy()
    for det in ("nc", "co", "colo", "ci", "cili"):
        for seasons, first in ((0, 0), (4, 1)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = VECM(z, k_ar_diff=3, coint_rank=1, deterministic=det, seasons=seasons,
                           first_season=first).fit(method="ml")
            tag = vecm_tag((det, seasons))
            out.append((f"vecm_e6_jmulti_{tag}", "VECM",
                        {"data": "e6 [Dp, R]", "k_ar_diff": 3, "coint_rank": 1,
                         "deterministic": det, "seasons": seasons, "first_season": first},
                        np.asarray(res.resid), res))
    return out


def jmulti_cases():
    cases = []
    for name, model, spec, u, res in jmulti_specs():
        path = JMULTI_DIR / f"{name}_diag.txt"
        assert path.exists(), f"{path} missing — the tag rule is wrong"
        txt = path.read_text(encoding="latin_1")
        dh_printed = parse_normality(txt, "dh")
        lut_printed = parse_normality(txt, "lutkepohl")
        jb_printed = parse_jb_rows(txt)
        n, k = u.shape
        sym = normality_symmetric(u)
        chol = normality_cholesky(u)
        dh = dh2008(u)
        jb_rows = univariate_jb_rows(u)
        tn = res.test_normality()
        assert abs(chol["statistic"] - float(tn.test_statistic)) <= 1e-10 * max(1.0, chol["statistic"])
        assert int(dh_printed["df"]) == int(lut_printed["df"]) == 2 * k == int(tn.df)
        assert len(jb_printed) == k
        g_sym, g_sym_p = printed_gap(sym, dh_printed)
        g_lut, g_lut_p = printed_gap(chol, lut_printed)
        g_jb = max(max(abs(r[key] - p[key]) for key in ("statistic", "pvalue", "skewness", "kurtosis"))
                   for r, p in zip(jb_rows, jb_printed))
        for what, g in (("symmetric statistics", g_sym), ("symmetric p-values", g_sym_p),
                        ("Cholesky vs Lütkepohl statistics", g_lut),
                        ("Cholesky vs Lütkepohl p-values", g_lut_p),
                        ("univariate Jarque-Bera rows", g_jb)):
            assert g <= PRINTED_TOL + 1e-12, f"{name}: {what} miss JMulTi's print by {g:.3e}"
        # The symmetric root is NOT the Cholesky factor: the two blocks differ.
        assert abs(sym["statistic"] - chol["statistic"]) > 1e-3, name
        cases.append({
            "name": name,
            "source": f"statsmodels {statsmodels.__version__} tsa/vector_ar/tests/JMulTi_results/{name}_diag.txt",
            "model": model,
            "spec": spec,
            "nobs": n,
            "k": k,
            "resid": u.tolist(),
            "printed_symmetric": dh_printed,
            "printed_lutkepohl": lut_printed,
            "printed_jarque_bera": jb_printed,
            "symmetric": sym,
            "cholesky": chol,
            "doornik_hansen": dh,
            "jarque_bera_rows": jb_rows,
            "statsmodels_test_normality": {"statistic": float(tn.test_statistic),
                                           "pvalue": float(tn.pvalue), "df": int(tn.df)},
            "margins": {
                "symmetric_statistic": PRINTED_TOL - g_sym,
                "symmetric_pvalue": PRINTED_TOL - g_sym_p,
                "cholesky_vs_lutkepohl_statistic": PRINTED_TOL - g_lut,
                "cholesky_vs_lutkepohl_pvalue": PRINTED_TOL - g_lut_p,
                "jarque_bera_rows": PRINTED_TOL - g_jb,
            },
        })
    assert len(cases) == 16
    return cases


def diag_case(name, series_name, y, p, trend, nlags_list):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(y).fit(p, trend=trend)
    k = res.neqs
    port = []
    for nlags in nlags_list:
        un = res.test_whiteness(nlags=nlags, adjusted=False)
        ad = res.test_whiteness(nlags=nlags, adjusted=True)
        assert un.df == ad.df == k * k * (nlags - p)
        port.append({
            "nlags": nlags,
            "statistic": float(un.test_statistic),
            "pvalue": float(un.pvalue),
            "adjusted": float(ad.test_statistic),
            "adjusted_pvalue": float(ad.pvalue),
            "df": int(un.df),
        })
    norm = res.test_normality()
    b1, b2, lam_skew, lam_kurt = normality_components(res)
    assert abs(lam_skew + lam_kurt - float(norm.test_statistic)) < 1e-10 * max(1.0, float(norm.test_statistic))
    roots = np.abs(np.asarray(res.roots))
    assert np.all(np.diff(roots) <= 1e-12)  # statsmodels sorts descending
    return {
        "name": name,
        "series": series_name,
        "p": p,
        "trend": trend,
        "nobs": int(res.nobs),
        "portmanteau": port,
        "normality": {
            "statistic": float(norm.test_statistic),
            "pvalue": float(norm.pvalue),
            "df": int(norm.df),
            "skewness": lam_skew,
            "skewness_pvalue": float(stats.chi2.sf(lam_skew, k)),
            "kurtosis": lam_kurt,
            "kurtosis_pvalue": float(stats.chi2.sf(lam_kurt, k)),
            "skewness_components": b1.tolist(),
            "kurtosis_components": b2.tolist(),
        },
        "normality_symmetric": normality_symmetric(np.asarray(res.resid)),
        "doornik_hansen": dh2008(np.asarray(res.resid)),
        "roots": roots.tolist(),
        "eigenvalue_moduli": np.sort(1.0 / roots)[::-1].tolist(),
        "is_stable": bool(res.is_stable()),
    }


def normality_cholesky_of(case, y, y_t, macro):
    """The Cholesky transcription on a stored case's refitted residuals (it
    must equal statsmodels' test_normality, which `normality` stores)."""
    data = {"sim_var2_k3": y, "sim_var2_k3_t4": y_t, "macro_growth_infl_dtbill": macro}[case["series"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(data).fit(case["p"], trend=case["trend"])
    return normality_cholesky(np.asarray(res.resid))


def select_case(series_name, y, maxlags, trend):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = VAR(y).select_order(maxlags=maxlags, trend=trend)
    p_min = 1 if trend == "n" else 0
    return {
        "series": series_name,
        "max_lags": maxlags,
        "trend": trend,
        "candidates": list(range(p_min, maxlags + 1)),
        "aic_values": [float(v) for v in sel.ics["aic"]],
        "bic_values": [float(v) for v in sel.ics["bic"]],
        "hqic_values": [float(v) for v in sel.ics["hqic"]],
        "fpe_values": [float(v) for v in sel.ics["fpe"]],
        "selected": {kk: int(v) for kk, v in sel.selected_orders.items()},
    }


def main():
    chol = [[0.8, 0.0, 0.0], [0.3, 0.6, 0.0], [-0.2, 0.25, 0.5]]
    a_list = [[[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]],
              [[0.1, 0.0, 0.05], [0.0, 0.1, 0.0], [0.05, 0.0, 0.1]]]
    y = sim_var(20260910, 300, c=[0.5, -0.2, 0.1], a_list=a_list, chol=chol)
    y_t = sim_var_t(20260912, 300, c=[0.5, -0.2, 0.1], a_list=a_list, chol=chol, df=4)
    macro = macro_system()

    cases = [
        diag_case("var2c_sim", "sim_var2_k3", y, 2, "c", [10, 6, 3]),
        diag_case("var3n_sim", "sim_var2_k3", y, 3, "n", [10, 5]),
        diag_case("var1c_sim_underfit", "sim_var2_k3", y, 1, "c", [10, 4]),
        diag_case("var2c_sim_t4", "sim_var2_k3_t4", y_t, 2, "c", [10]),
        diag_case("macro_var2c", "macro_growth_infl_dtbill", macro, 2, "c", [12, 8]),
        diag_case("macro_var4c", "macro_growth_infl_dtbill", macro, 4, "c", [12]),
    ]
    for c in cases:
        assert abs(c["normality"]["statistic"]
                   - normality_cholesky_of(c, y, y_t, macro)["statistic"]) <= 1e-10 * max(1.0, c["normality"]["statistic"])
    identity_ns = assert_mean_matching_identity()
    jm = jmulti_cases()
    selects = [
        select_case("sim_var2_k3", y, 8, "c"),
        select_case("sim_var2_k3", y, 6, "n"),
        select_case("macro_growth_infl_dtbill", macro, 8, "c"),
    ]
    fixture = {
        "_meta": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "python": platform.python_version(),
            "note": (
                "VAR residual diagnostics from statsmodels VARResults: "
                "test_whiteness (adjusted False/True), test_normality (omnibus; "
                "the skewness/kurtosis components transcribed from the same "
                "code), roots moduli, is_stable; VAR.select_order tables. "
                "Macro series are transformations of statsmodels' macrodata."
            ),
            "normality_note": (
                "normality = Cholesky (statsmodels test_normality; JMulTi 'Lütkepohl (1993)'). "
                "normality_symmetric = symmetric-root JB with no transforms (JMulTi 'Doornik & "
                "Hansen (1994)'; independent-program golden at printed precision, the `jmulti` "
                "list). doornik_hansen = Doornik-Hansen (2008) transcribed from the paper "
                "(documented formula; z1 == scipy.stats.skewtest of the transformed columns, "
                "the kurtosis coefficients satisfy the exact mean-matching identity)."
            ),
            "jmulti_printed_tolerance": PRINTED_TOL,
            "dh2008_skewtest_rtol": 1e-10,
            "dh2008_identity_checked_n": identity_ns,
            "jmulti_tag_rule": "statsmodels dt_s_tup_to_string (VAR and VECM parsers)",
        },
        "series": {
            "sim_var2_k3": y.tolist(),
            "sim_var2_k3_t4": y_t.tolist(),
            "macro_growth_infl_dtbill": macro.tolist(),
        },
        "cases": cases,
        "select_order": selects,
        "jmulti": jm,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {len(cases)} diagnostic cases, "
          f"{len(selects)} select_order cases")
    for c in cases:
        q = c["portmanteau"][0]
        n = c["normality"]
        print(f"  {c['name']:22s} p={c['p']} {c['trend']} T={c['nobs']} "
              f"Q({q['nlags']})={q['statistic']:.3f} p={q['pvalue']:.4f} "
              f"adj={q['adjusted']:.3f} p={q['adjusted_pvalue']:.4f} | "
              f"JB={n['statistic']:.3f} p={n['pvalue']:.4f} "
              f"(skew {n['skewness']:.3f}, kurt {n['kurtosis']:.3f}) | "
              f"min root={c['roots'][-1]:.4f} stable={c['is_stable']}")
    for s in selects:
        print(f"  select {s['series']:24s} maxlags={s['max_lags']} {s['trend']} -> {s['selected']}")
    for c in cases:
        sy, dh = c["normality_symmetric"], c["doornik_hansen"]
        print(f"  {c['name']:22s} symmetric JB={sy['statistic']:.4f} p={sy['pvalue']:.4f} | "
              f"DH2008 E={dh['statistic']:.4f} p={dh['pvalue']:.4f}")
    worst = {key: min(c["margins"][key] for c in jm) for key in jm[0]["margins"]}
    for c in jm:
        sy, pr = c["symmetric"], c["printed_symmetric"]
        print(f"  {c['name']:26s} T={c['nobs']} k={c['k']} symmetric {sy['statistic']:.5f}/"
              f"{sy['skewness']:.5f}/{sy['kurtosis']:.5f} printed {pr['joint']}/{pr['skew']}/"
              f"{pr['kurt']} | Cholesky {c['cholesky']['statistic']:.5f} printed Lütkepohl "
              f"{c['printed_lutkepohl']['joint']} | DH2008 {c['doornik_hansen']['statistic']:.4f}")
    print(f"  JMulTi minimum margins (5e-5 minus worst gap): {worst}")


if __name__ == "__main__":
    main()
