"""Every number the 0.11.0 residual-normality and StepM family-wise-error
documentation quotes from the Python side, recomputed through the installed
wheel (the Rust-side Monte Carlo numbers are printed by the crate tests
themselves: `cargo test -p tsecon-var --test var_cf_properties -- --nocapture`
and `cargo test -p tsecon-forecast --test spa_mcs_properties -- --nocapture`).

    python lab/stepm_dh/card_numbers.py            # everything (~1 min)
    python lab/stepm_dh/card_numbers.py jmulti     # one section
    python lab/stepm_dh/card_numbers.py example
    python lab/stepm_dh/card_numbers.py stepm

Sections:

* ``jmulti`` — `tsecon.multivariate_normality` on the 16 JMulTi residual
  matrices stored in fixtures/var_diag.json, against JMulTi's printed
  "Doornik & Hansen (1994)" block (the symmetric-root Jarque-Bera), its
  "Lütkepohl (1993)" block (the Cholesky form) and its per-variable
  Jarque-Bera rows (k = 1): the worst absolute gap per block, and one
  example row.
* ``example`` — the runnable example of the var-svar model card's residual
  normality subsection (its printed output is pasted under the code).
* ``stepm`` — tsecon's un-studentized `stepm_test` PAIRED with arch's own
  StepM: the generator stored arch's per-replication superior sets on four
  seeded designs (fixtures/spa.json, `_meta.stepm_fwer_study`); the panels
  are rebuilt here from the same NumPy seeds (checked against the stored
  SHA-256) and tsecon runs on exactly those panels with its own bootstrap
  seed 9000 + r. Reported per design and size: tsecon's FWER on those 1000
  panels, arch's, the share of panels where the two superior sets are
  identical, arch's own agreement with itself under a second seed, and the
  discordant FWER pairs (McNemar's b, c).

Nothing here is a golden: the goldens are the fixtures and the tests. This
script only reproduces the numbers the prose quotes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import tsecon

ROOT = Path(__file__).resolve().parents[2]


def section_jmulti():
    dg = json.loads((ROOT / "fixtures" / "var_diag.json").read_text())
    worst = {"symmetric (DH 1994 block)": 0.0, "Cholesky (Lutkepohl block)": 0.0,
             "univariate JB rows": 0.0}
    for c in dg["jmulti"]:
        u = np.array(c["resid"])
        r = tsecon.multivariate_normality(u)
        p, lp = c["printed_symmetric"], c["printed_lutkepohl"]
        for key, pk in (("jarque_bera_symmetric", "joint"), ("skewness_symmetric", "skew"),
                        ("kurtosis_symmetric", "kurt"), ("jarque_bera_symmetric_pvalue", "p_joint"),
                        ("skewness_symmetric_pvalue", "p_skew"),
                        ("kurtosis_symmetric_pvalue", "p_kurt")):
            worst["symmetric (DH 1994 block)"] = max(worst["symmetric (DH 1994 block)"],
                                                     abs(r[key] - p[pk]))
        for key, pk in (("jarque_bera", "joint"), ("skewness", "skew"), ("kurtosis", "kurt"),
                        ("jarque_bera_pvalue", "p_joint"), ("skewness_pvalue", "p_skew"),
                        ("kurtosis_pvalue", "p_kurt")):
            worst["Cholesky (Lutkepohl block)"] = max(worst["Cholesky (Lutkepohl block)"],
                                                      abs(r[key] - lp[pk]))
        for j, row in enumerate(c["printed_jarque_bera"]):
            one = tsecon.multivariate_normality(u[:, j])
            for got, want in ((one["jarque_bera"], row["statistic"]),
                              (one["jarque_bera_pvalue"], row["pvalue"]),
                              (one["skewness_components"][0], row["skewness"]),
                              (one["kurtosis_components"][0] + 3.0, row["kurtosis"])):
                worst["univariate JB rows"] = max(worst["univariate JB rows"], abs(got - want))
        if c["name"] in ("macrodata_jmulti_nc", "vecm_e6_jmulti_coslo"):
            print(f"{c['name']} (T = {c['nobs']}, k = {c['k']}): symmetric JB "
                  f"{r['jarque_bera_symmetric']:.5f} / skew {r['skewness_symmetric']:.5f} / "
                  f"kurt {r['kurtosis_symmetric']:.5f}  [JMulTi prints {p['joint']} / {p['skew']} / "
                  f"{p['kurt']}];  Cholesky JB {r['jarque_bera']:.5f} [JMulTi Lutkepohl {lp['joint']}];  "
                  f"Doornik-Hansen (2008) {r['doornik_hansen']:.4f} (p {r['doornik_hansen_pvalue']:.3g})")
    print(f"16 JMulTi outputs, worst |tsecon - printed| per block (print precision 5e-5): "
          + ", ".join(f"{k} {v:.3e}" for k, v in worst.items()))
    # Two conventions the match pins, shown by breaking them: (a) WITHOUT the
    # column centring (NumPy transcription of the symmetric-root JB on the raw
    # residuals), and (b) the transformed Doornik-Hansen (2008) statistic in
    # place of JMulTi's joint statistic.
    gap_raw, gap_dh = 0.0, math.inf
    for c in dg["jmulti"]:
        u = np.array(c["resid"])
        t = len(u)
        s = u.T @ u / t
        lam, h = np.linalg.eigh(s)
        w = u @ (h @ np.diag(lam ** -0.5) @ h.T)
        b1, b2 = (w ** 3).mean(0), (w ** 4).mean(0) - 3
        raw = t * (b1 @ b1) / 6 + t * (b2 @ b2) / 24
        gap_raw = max(gap_raw, abs(raw - c["printed_symmetric"]["joint"]))
        dh = tsecon.multivariate_normality(u)["doornik_hansen"]
        gap_dh = min(gap_dh, abs(dh - c["printed_symmetric"]["joint"]))
    print(f"uncentred symmetric-root JB vs JMulTi's joint statistic: worst gap {gap_raw:.3f}; "
          f"Doornik-Hansen (2008) vs JMulTi's joint statistic: SMALLEST gap {gap_dh:.3f}")


def section_example():
    # --- the var-svar card's residual-normality example, verbatim ---
    rng = np.random.default_rng(0)
    k, n = 3, 200
    A = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.5]])
    L = np.array([[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.5, 0.3, 0.8]])
    Y = np.zeros((n, k))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + L @ rng.standard_t(6, k)   # correlated, fat-tailed shocks

    d = tsecon.var_diagnostics(Y, lags=1, nlags=10)
    for key in ("jarque_bera", "jarque_bera_symmetric", "doornik_hansen"):
        print(f"{key:22s} {d[key]:7.2f}   p = {d[key + '_pvalue']:.2g}")
    print("DH z1 (skewness):", np.round(d["doornik_hansen_z_skewness"], 2),
          " z2 (kurtosis):", np.round(d["doornik_hansen_z_kurtosis"], 2))

    # Change the units of two series: only the JMulTi symmetric root moves.
    d2 = tsecon.var_diagnostics(Y * [1.0, 100.0, 0.01], lags=1, nlags=10)
    print("rescaled units:", [round(d2[key], 2) for key in
                              ("jarque_bera", "jarque_bera_symmetric", "doornik_hansen")])

    # The same tests on any residual matrix, or on one series (k = 1).
    u = np.array(tsecon.var_fit(Y, lags=1)["resid"])
    assert tsecon.multivariate_normality(u)["doornik_hansen"] == d["doornik_hansen"]
    one = tsecon.multivariate_normality(u[:, 0])
    print(f"series 0 alone: DH {one['doornik_hansen']:.2f} (p = {one['doornik_hansen_pvalue']:.2g}),"
          f" JB {one['jarque_bera']:.2f} = jarque_bera {tsecon.jarque_bera(u[:, 0])['statistic']:.2f}")
    print("T = 7:", tsecon.var_diagnostics(Y[:8], lags=1, nlags=2)["doornik_hansen"])


def _ar1(rng, n, rho):
    """fixtures/generate_spa_fixtures.py's ar1 (copied; the lab never
    imports the generators)."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    prev = 0.0
    for t in range(n):
        prev = rho * prev + math.sqrt(1 - rho * rho) * e[t]
        x[t] = prev
    return x


def _loss_panel(rng, n, scales, rho=0.5, common=0.7):
    u = _ar1(rng, n, rho)
    return [((common * u + s * _ar1(rng, n, rho)) ** 2).tolist() for s in scales]


def section_stepm():
    spa = json.loads((ROOT / "fixtures" / "spa.json").read_text())
    st = spa["_meta"]["stepm_fwer_study"]
    mc, n, m = st["mc"], st["n"], st["m"]
    print(f"StepM, studentize=False, n = {n}, m = {m}, B = {st['reps']}, block {st['block_size']}, "
          f"{mc} panels per design (arch's own panels, paired)")
    print(f"{'design':13s} {'size':>5s} {'tsecon FWER':>12s} {'arch FWER':>10s} {'same set':>9s} "
          f"{'arch vs arch':>13s} {'FWER discordant (t only, a only)':>34s}")
    for label, d in st["designs"].items():
        true_null = np.zeros(m, dtype=bool)
        true_null[d["true_nulls"]] = True
        h = hashlib.sha256()
        mine = {"0.05": [], "0.1": []}
        for r in range(mc):
            panel = np.column_stack(_loss_panel(np.random.default_rng(880000 + r), n, d["scales"]))
            h.update(np.ascontiguousarray(panel, dtype=np.float64).tobytes())
            for key, size in (("0.05", 0.05), ("0.1", 0.10)):
                out = tsecon.stepm_test(panel[:, 0], panel[:, 1:], size=size, block_size=8, reps=300,
                                        bootstrap="stationary", studentize=False, seed=9000 + r)
                mine[key].append(sorted(out["superior_models"]))
        if h.hexdigest() != d["panel_sha256"]:
            print(f"{label}: this NumPy does not reproduce the fixture's panels; pairing invalid")
            continue
        for key in ("0.05", "0.1"):
            a = d["rates"][key]
            fw_t = np.array([any(true_null[s]) for s in mine[key]])
            fw_a = np.array([any(true_null[s]) for s in a["superior_sets"]])
            same = np.mean([x == y for x, y in zip(mine[key], a["superior_sets"])])
            b, c = int(np.sum(fw_t & ~fw_a)), int(np.sum(fw_a & ~fw_t))
            print(f"{label:13s} {key:>5s} {fw_t.mean():12.3f} {a['fwer']:10.3f} {same:9.3f} "
                  f"{a['arch_agreement']:13.3f} {f'({b}, {c})':>34s}")


SECTIONS = {"jmulti": section_jmulti, "example": section_example, "stepm": section_stepm}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(SECTIONS):
        print(f"==== {name}")
        SECTIONS[name]()
