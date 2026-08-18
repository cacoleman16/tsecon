"""Golden fixtures for the VaR backtest battery: Kupiec (1995),
Christoffersen (1998), and the Engle-Manganelli (2004) DQ test.

Honest grading per component:

  * Kupiec LR_uc and Christoffersen LR_ind / LR_cc are CLOSED FORM — this
    generator computes them independently in NumPy/SciPy from first
    principles (the Bernoulli/Markov likelihoods written out below), on
    seeded hit sequences and on hand-computable tiny cases whose full
    arithmetic is in the comments.
  * The DQ statistic pins against a statsmodels-OLS construction on the
    identical hit sequences (strong third-party for the regression
    algebra): DQ = fitted'fitted / (alpha (1-alpha)) with fitted from
    sm.OLS(Hit, X), which equals Hit'X (X'X)^{-1} X'Hit because the
    fitted values are the projection P Hit. A direct linear-algebra
    cross-check (normal equations via numpy.linalg.solve) must agree to
    1e-10 relative or the generator refuses to write.
  * Published worked example: J.P. Morgan disclosed 20 downside 95%-VaR
    breaches over the 252 trading days of 1998 — the worked backtest
    example in Jorion, "Value at Risk" (3rd ed.), ch. 6, reproduced
    throughout the FRM curriculum. LR_uc(T=252, N=20, alpha=0.05) =
    3.9126 (Jorion prints 3.91), a borderline rejection against the 3.84
    chi-squared(1) 5% critical value. Verified via web search of
    FRM/Jorion study material (the book text itself was not fetchable
    from this environment); the statistic here is recomputed exactly
    from the formula, not transcribed at two decimals.

Conventions pinned by these fixtures (must match the Rust docs):

  * Sign convention: returns and VaR forecasts on the SAME (return)
    scale; the VaR forecast is the alpha-quantile of the conditional
    return distribution (negative for small alpha); a violation is
    return < VaR (strict).
  * Christoffersen transitions run over the n-1 pairs (t-1, t); empty
    cells use the 0*ln(0) = 0 continuity convention; LR_cc = LR_uc +
    LR_ind (chi-squared(2)).
  * DQ regression: Hit_t = hit_t - alpha on [1, Hit_{t-1..t-L}, VaR_t]
    over t = L..n-1; k = L + 2 with the VaR regressor, L + 1 without;
    DQ ~ chi-squared(k).

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  /home/user/tsecon/.venv/bin/python fixtures/generate_var_backtest_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

OUT = Path(__file__).resolve().parent / "var_backtest.json"


# ------------------------------------------------------- first principles

def xlogy(count: int, p: float) -> float:
    """count * ln(p) with the 0 * ln(0) = 0 continuity convention."""
    return 0.0 if count == 0 else count * np.log(p)


def kupiec(hits: np.ndarray, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures LR, written out.

    LR_uc = -2 ln[ (1-a)^n0 a^n1 / ((1-p)^n0 p^n1) ],  p = n1/n.
    """
    n = hits.size
    n1 = int(hits.sum())
    n0 = n - n1
    p = n1 / n
    ll_null = xlogy(n0, 1.0 - alpha) + xlogy(n1, alpha)
    ll_alt = xlogy(n0, 1.0 - p) + xlogy(n1, p)
    lr = -2.0 * (ll_null - ll_alt)
    lr = lr if lr > 0.0 else 0.0     # clamp rounding noise AND -0.0 to +0.0
    return {
        "n": n,
        "n_violations": n1,
        "hit_rate": p,
        "lr_uc": float(lr),
        "p_uc": float(chi2.sf(lr, 1)),
    }


def christoffersen(hits: np.ndarray, alpha: float) -> dict:
    """Christoffersen (1998) LR_ind and LR_cc = LR_uc + LR_ind."""
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi2 = (n01 + n11) / (h.size - 1)
    ll0 = xlogy(n00 + n10, 1.0 - pi2) + xlogy(n01 + n11, pi2)
    ll1 = (xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01)
           + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11))
    lr_ind = -2.0 * (ll0 - ll1)
    lr_ind = lr_ind if lr_ind > 0.0 else 0.0
    lr_uc = kupiec(hits, alpha)["lr_uc"]
    lr_cc = lr_uc + lr_ind
    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi01": float(pi01), "pi11": float(pi11),
        "lr_ind": float(lr_ind), "p_ind": float(chi2.sf(lr_ind, 1)),
        "lr_cc": float(lr_cc), "p_cc": float(chi2.sf(lr_cc, 2)),
    }


def dq_statsmodels(hits: np.ndarray, alpha: float, lags: int,
                   var: np.ndarray | None) -> dict:
    """Engle-Manganelli (2004) DQ via statsmodels OLS (the third-party
    golden for the regression algebra), cross-checked against the normal
    equations.

    Hit_t = hit_t - alpha regressed on [1, Hit_{t-1..t-L}, VaR_t] over
    t = L..n-1; DQ = Hit'X (X'X)^{-1} X'Hit / (alpha (1-alpha)) = the sum
    of squared OLS fitted values over alpha(1-alpha), ~ chi2(k).

    Rank rule (documented in the Rust module and mirrored here): the
    chi-squared df is the RANK of the design. A VaR column inside the
    span of [1, lagged hits] — a constant VaR path, i.e. an unconditional
    VaR model — is dropped and df shrinks by one; the projection (and so
    the statistic) is unchanged by the drop. (statsmodels would pinv its
    way through the singular design but report the nominal column count
    as df, which is wrong — hence the explicit rule here.)
    """
    H = hits.astype(float) - alpha
    n = H.size
    y = H[lags:]
    cols = [np.ones(n - lags)]
    for j in range(1, lags + 1):
        cols.append(H[lags - j:n - j])
    var_dropped = False
    if var is not None:
        vcol = np.asarray(var, float)[lags:]
        base = np.column_stack(cols)
        resid = vcol - base @ np.linalg.lstsq(base, vcol, rcond=None)[0]
        if np.linalg.norm(resid) <= 1e-8 * np.linalg.norm(vcol):
            var_dropped = True          # collinear: drop, don't mis-count df
        else:
            cols.append(vcol)
    X = np.column_stack(cols)
    k = X.shape[1]
    assert np.linalg.matrix_rank(X) == k, "lagged-hit columns must be full rank"

    fit = sm.OLS(y, X).fit()
    fitted = np.asarray(fit.fittedvalues, float)
    dq_sm = float(fitted @ fitted / (alpha * (1.0 - alpha)))

    # Independent cross-check: the same quadratic form from the normal
    # equations. Refuse to write a fixture the two routes disagree on.
    xtx = X.T @ X
    xth = X.T @ y
    dq_direct = float(xth @ np.linalg.solve(xtx, xth) / (alpha * (1.0 - alpha)))
    rel = abs(dq_sm - dq_direct) / max(abs(dq_direct), 1e-300)
    assert rel < 1e-10, f"statsmodels vs normal-equations DQ mismatch: {rel}"

    return {
        "dq_lags": lags,
        "dq_df": k,
        "includes_var": var is not None and not var_dropped,
        "var_dropped": var_dropped,
        "dq_stat": dq_sm,
        "p_dq": float(chi2.sf(dq_sm, k)),
    }


def battery(hits: np.ndarray, alpha: float, lags: int,
            var: np.ndarray | None) -> dict:
    out = kupiec(hits, alpha)
    out.update(christoffersen(hits, alpha))
    out.update(dq_statsmodels(hits, alpha, lags, var))
    out["alpha"] = alpha
    return out


# ------------------------------------------------------------- hit DGPs

def bernoulli_hits(n: int, alpha: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(n) < alpha).astype(float)


def markov_hits(n: int, pi01: float, pi11: float, seed: int) -> np.ndarray:
    """First-order Markov hits: clustered violations. The stationary
    unconditional rate is pi01 / (pi01 + 1 - pi11)."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    state = 0
    for t in range(n):
        p = pi11 if state == 1 else pi01
        state = 1 if rng.random() < p else 0
        h[t] = state
    return h


# ----------------------------------------------------------------- cases

def gen_hit_cases() -> list[dict]:
    """Seeded hit sequences, backtested WITHOUT VaR forecasts (the
    hits-only entry point; DQ df = lags + 1)."""
    cases = []
    specs = [
        # (name, hits, alpha, dq_lags)
        ("bern_n500_a05_s1", bernoulli_hits(500, 0.05, 1), 0.05, 4),
        ("bern_n1000_a05_s2", bernoulli_hits(1000, 0.05, 2), 0.05, 4),
        ("bern_n1000_a01_s3", bernoulli_hits(1000, 0.01, 3), 0.01, 4),
        ("bern_n500_a10_s4", bernoulli_hits(500, 0.10, 4), 0.10, 2),
        # Clustered: pi11 = 0.4 >> unconditional rate 0.05 (pi01 chosen so
        # the stationary rate is exactly 0.05) — LR_ind/DQ should light up.
        ("markov_n1000_s5",
         markov_hits(1000, 0.05 * (1 - 0.4) / (1 - 0.05), 0.4, 5), 0.05, 4),
        # Wrong rate, independent: LR_uc should light up, LR_ind not.
        ("bern_n1000_a05_rate10_s6", bernoulli_hits(1000, 0.10, 6), 0.05, 4),
    ]
    for name, hits, alpha, lags in specs:
        case = {"name": name, "alpha": alpha,
                "hits": [int(v) for v in hits]}
        case.update(battery(hits, alpha, lags, None))
        cases.append(case)
    return cases


def gen_return_cases() -> list[dict]:
    """Return + VaR-forecast cases exercising the sign convention
    (violation = return < VaR quantile, VaR negative) and the DQ VaR
    regressor (df = lags + 2)."""
    cases = []
    z975 = 1.959963984540054   # Phi^{-1}(0.975); N(0,1) alpha=0.025 quantile
    z95 = 1.6448536269514722   # Phi^{-1}(0.95)
    specs = []

    # (a) Correct model: AR(1)-in-volatility ("GARCH-ish") returns with the
    # TRUE conditional-normal VaR — all three tests should be quiet.
    rng = np.random.default_rng(10)
    n = 750
    sig = np.empty(n)
    r = np.empty(n)
    s2 = 1.0
    for t in range(n):
        sig[t] = np.sqrt(s2)
        r[t] = sig[t] * rng.standard_normal()
        s2 = 0.05 + 0.10 * r[t] ** 2 + 0.85 * s2
    var_true = -z95 * sig
    specs.append(("garch_true_var_a05", r, var_true, 0.05, 4))

    # (b) Mis-specified: an UNCONDITIONAL VaR against the same
    # heteroskedastic returns — clustering, LR_ind/DQ should reject.
    var_flat = np.full(n, -z95 * np.std(r))
    specs.append(("garch_flat_var_a05", r, var_flat, 0.05, 4))

    # (c) iid returns with the true iid VaR at alpha = 0.025.
    rng = np.random.default_rng(11)
    r2 = rng.standard_normal(600)
    var2 = np.full(600, -z975)
    specs.append(("iid_true_var_a025", r2, var2, 0.025, 4))

    for name, ret, var, alpha, lags in specs:
        hits = (ret < var).astype(float)
        case = {
            "name": name,
            "alpha": alpha,
            "returns": [float(v) for v in ret],
            "var_forecasts": [float(v) for v in var],
        }
        case.update(battery(hits, alpha, lags, var))
        cases.append(case)
    return cases


def gen_hand_cases() -> list[dict]:
    """Tiny cases whose LR arithmetic is verifiable by hand.

    HAND CALCULATION — kupiec_hand_n250_x5 (n = 250, 5 violations,
    alpha = 0.05, so 12.5 expected and pi_hat = 0.02):

        ll_null = 245 ln(0.95) + 5 ln(0.05)
                = 245(-0.0512932944) + 5(-2.9957322736)
                = -12.5668571 - 14.9786614 = -27.5455185
        ll_alt  = 245 ln(0.98) + 5 ln(0.02)
                = 245(-0.0202027073) + 5(-3.9120230054)
                = -4.9496633 - 19.5601150 = -24.5097783
        LR_uc   = -2(ll_null - ll_alt) = -2(-3.0357402) = 6.0714803
        p       = P(chi2_1 > 6.0714803) = 0.0137        -> reject: the
        VaR is too conservative (5 violations where 12.5 were expected).

    The violations are placed at t = 40, 90, 140, 190, 240 (never
    consecutive), so the Christoffersen cells are n11 = 0, n01 = n10 = 5,
    n00 = 239, exercising the 0 ln 0 continuity path:

        pi01 = 5/244 = 0.0204918, pi11 = 0, pi2 = 5/249 = 0.0200803
        ll0 = 244 ln(1 - pi2) + 5 ln(pi2)   [pooled]
        ll1 = 239 ln(1 - pi01) + 5 ln(pi01) + 5 ln(1) + 0
        LR_ind = -2(ll0 - ll1) = 0.2059     (tiny: too few violations to
        see clustering either way), LR_cc = 6.2774 ~ chi2(2), p = 0.0433.

    JORION / J.P. MORGAN 1998 — jorion_jpm_1998 (n = 252, 20 violations,
    alpha = 0.05; Jorion "Value at Risk" ch. 6): LR_uc = 3.9126 (the book
    prints 3.91) vs the 3.84 critical value — borderline rejection. The
    hit *placement* is not published, only the count; violations here are
    spread regularly (every 12th day), so only lr_uc/p_uc in this case
    are the published-example pin, and the Markov/DQ numbers are ordinary
    first-principles goldens on that sequence.
    """
    cases = []

    hits = np.zeros(250)
    hits[[40, 90, 140, 190, 240]] = 1.0
    case = {"name": "kupiec_hand_n250_x5", "alpha": 0.05,
            "hits": [int(v) for v in hits]}
    case.update(battery(hits, 0.05, 4, None))
    # The hand values above, asserted so a formula slip in THIS generator
    # cannot silently ship: LR_uc to the 7 hand digits.
    assert abs(case["lr_uc"] - 6.0714803) < 5e-7, case["lr_uc"]
    assert abs(case["p_uc"] - 0.0137) < 5e-5, case["p_uc"]
    assert case["n11"] == 0 and case["n01"] == 5 and case["n10"] == 5
    cases.append(case)

    hits = np.array([1.0 if (t % 12 == 6 and t < 240) else 0.0
                     for t in range(252)])
    assert int(hits.sum()) == 20
    case = {"name": "jorion_jpm_1998", "alpha": 0.05,
            "hits": [int(v) for v in hits]}
    case.update(battery(hits, 0.05, 4, None))
    assert abs(case["lr_uc"] - 3.9126) < 5e-5, case["lr_uc"]   # Jorion: 3.91
    cases.append(case)

    return cases


# ------------------------------------------------------------------ main

def main() -> None:
    fixture = {
        "hit_cases": gen_hit_cases(),
        "return_cases": gen_return_cases(),
        "hand_cases": gen_hand_cases(),
        "_meta": {
            "note": (
                "LR_uc/LR_ind/LR_cc from first-principles NumPy/SciPy "
                "(closed-form likelihoods); DQ from statsmodels OLS fitted "
                "values, cross-checked against the normal equations at "
                "1e-10; chi-squared p-values from scipy.stats.chi2.sf. "
                "Sign convention: violation = return < VaR quantile, both "
                "on the return scale. Published pin: Jorion VaR ch.6 / "
                "J.P. Morgan 1998, LR_uc(252, 20, 0.05) = 3.91."
            ),
            "statsmodels": sm.version.version if hasattr(sm, "version") else "",
            "numpy": np.__version__,
        },
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, indent=1)
    n_cases = sum(len(fixture[k]) for k in ("hit_cases", "return_cases", "hand_cases"))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes); {n_cases} cases")


if __name__ == "__main__":
    main()
