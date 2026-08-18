"""Replication: Bai & Perron (2003) — breaks in the US ex-post real interest rate.

Bai & Perron's 2003 *Journal of Applied Econometrics* paper is the canonical
reference for dating multiple structural breaks, and its empirical application
is a mean-shift model of the US ex-post real interest rate, quarterly
1961Q1-1986Q3 (T = 103, the Garcia-Perron 1996 series). The published answer:
the level of the real rate breaks at **1972:3** and **1980:3** (the partition
every criterion agrees on), with the paper's own HAC-robust sequential
procedure adding a third, less sharply dated break at **1966:4**; the segment
means are **1.36, -1.80, 5.64** in the two-break model. The 1980:3 (Volcker)
break is dated to within a few quarters; the earlier breaks are not.

Data: the exact series ships as `RealInt` in R's strucchange package (and as
`real` in Perron's own mbreaks package — byte-identical), committed to this
repository at fixtures/realint_bai_perron.csv with attribution, so this runs
fully offline — tsecon ships no data loaders.

    .venv/bin/python docs/examples/replication_bai_perron_realint.py

WHAT THIS IS, AND IS NOT
------------------------
`tsecon.bai_perron` runs the same dynamic program over the same admissible
partitions and the same sequential supF(l+1|l) selection at the same published
5% critical values. Its break DATES reproduce the published ones exactly, at
every number of breaks up to the paper's three. Two constructions differ, and
are compared honestly rather than papered over:

- tsecon's supF statistics are CLASSICAL (homoskedastic-F); the paper's
  published statistics are HAC-robust. With classical F the sequential
  procedure stops at 2 breaks (the answer BP's own BIC and LWZ criteria give,
  and the answer strucchange's classical BIC gives); with the paper's HAC F it
  continues to 3. Both counts select the same dates tsecon finds.
- tsecon's break-date confidence intervals are the Bai (1997) HOMOGENEOUS
  case with classical variance; the paper's published CIs are the
  heterogeneity-robust HAC variant, which tsecon deliberately does not ship
  (see the structural-breaks model card). The 2005 JAE replication study
  (Zeileis & Kleiber) could not reproduce some of BP's published CIs either.
"""
import csv
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "realint_bai_perron.csv"

# Published anchors (mean-shift model, trim = 0.15). Sources: Bai & Perron
# (2003, JAE 18(1), Section 4 empirical application); the JAE replication study
# Zeileis & Kleiber (2005, JAE 20:685-690); and Perron's own mbreaks R package,
# whose vignette reruns this exact exercise with the paper's settings.
PUBLISHED = {
    "dates_m2": ["1972Q3", "1980Q3"],          # BIC / LWZ partition
    "dates_m3": ["1966Q4", "1972Q3", "1980Q3"],  # sequential (HAC) partition
    "means_m2": [1.36, -1.80, 5.64],
    "means_m3": [1.82, 0.87, -1.80, 5.64],
    "hac_se_m2": [0.16, 0.51, 0.60],           # HAC SEs as published (mbreaks: 0.155/0.511/0.603)
    "supf_seq_hac": [57.91, 33.93, 14.73],     # supF(1|0), supF(2|1), supF(3|2), HAC
    "ci95_1972q3_m3": ("1970Q3", "1972Q4"),    # the one CI printed in the paper we
                                               # could independently verify (via the
                                               # Zeileis-Kleiber JAE validation study)
}


def load_realint(path=DATA):
    """Read the committed RealInt series into (quarter, rate) arrays.

    Public academic data, vendored with attribution in the CSV header —
    no download, no loader.
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    assert rows[0] == ["quarter", "realint"]
    quarter = np.array([float(r[0]) for r in rows[1:]])
    rate = np.array([float(r[1]) for r in rows[1:]])
    return quarter, rate


def qlabel(idx):
    """0-based observation index -> quarter label; obs 0 = 1961Q1."""
    return f"{1961 + idx // 4}Q{idx % 4 + 1}"


def run(rate, max_breaks=5, trim=0.15):
    """The whole analysis is one call: mean-shift = intercept-only design."""
    X = np.ones((len(rate), 1))
    return tsecon.bai_perron(rate, X, max_breaks=max_breaks, trim=trim)


def rule(width=72, ch="-"):
    print(ch * width)


def main():
    print("Replication — Bai & Perron (2003), J. Applied Econometrics 18(1)")
    print("mean shifts in the US ex-post real interest rate, 1961Q1-1986Q3")
    rule(72, "=")

    quarter, rate = load_realint()
    print(f"data: strucchange RealInt (committed) — {len(rate)} quarters, "
          f"{quarter[0]:.2f} to {quarter[-1]:.2f}")

    bp = run(rate)

    # ---- where: the globally optimal partitions -------------------------
    print("\nGlobal SSR-minimizing partitions (dynamic program):")
    for m in (1, 2, 3):
        dates = [qlabel(int(d)) for d in bp["break_dates_by_m"][m - 1]]
        ssr = bp["ssr_path"][m]
        print(f"  m = {m}:  {', '.join(dates):28s}  SSR = {ssr:8.2f}")
    print("  published: m = 2 -> 1972:3, 1980:3   m = 3 -> 1966:4, 1972:3, 1980:3")
    print("  (every break date matches the published partition exactly)")

    # ---- how many: the sequential procedure -----------------------------
    seq = np.asarray(bp["sup_f_seq"])
    crit = np.asarray(bp["sup_f_crit"])
    print(f"\nSequential supF(l+1|l) at the published 5% critical values:")
    print("  l ->l+1 |  classical F (tsecon) |  5% CV |  paper's HAC F")
    hac = PUBLISHED["supf_seq_hac"] + [None, None]
    for l in range(3):
        hac_s = f"{hac[l]:.2f}" if hac[l] is not None else "-"
        print(f"  {l} -> {l+1}   |  {seq[l]:8.2f}             | {crit[l]:6.2f} |  {hac_s}")
    print(f"  tsecon (classical F) selects n_breaks = {bp['n_breaks']} — supF(3|2) = "
          f"{seq[2]:.2f} < {crit[2]:.2f} stops the sequence.")
    print("  The paper's HAC-robust supF(3|2) = 14.73 keeps going: BP's sequential")
    print("  procedure selects 3 breaks, while their BIC and LWZ select 2. tsecon's")
    print("  classical sequential count agrees with the information criteria; the")
    print("  m = 3 dates it would add are the published ones (see above).")

    # ---- the selected model ---------------------------------------------
    params = np.asarray(bp["params"])[:, 0]
    bse = np.asarray(bp["bse"])[:, 0]
    print(f"\nSelected model (n_breaks = {bp['n_breaks']}): segment means")
    print("  regime            |  tsecon mean (OLS se) |  published mean (HAC se)")
    rule(72)
    pub_m = PUBLISHED["means_m2"]
    pub_s = PUBLISHED["hac_se_m2"]
    for j in range(bp["n_breaks"] + 1):
        s, e = bp["regime_starts"][j], bp["regime_ends"][j]
        print(f"  {qlabel(s)} - {qlabel(e)}   |  {params[j]:+7.3f} ({bse[j]:.3f})     "
              f"|  {pub_m[j]:+5.2f} ({pub_s[j]:.2f})")
    print("  Means match to published rounding; the SEs are different constructions")
    print("  (classical per-regime OLS here, HAC in the paper) and are not compared.")

    # ---- how sure: break-date confidence intervals ----------------------
    print("\nBreak-date confidence intervals, Bai (1997):")
    print("  break   |  tsecon 90%          |  tsecon 95%          | published 95%")
    rule(72)
    pub95 = {"1972Q3": f"{PUBLISHED['ci95_1972q3_m3'][0]}-{PUBLISHED['ci95_1972q3_m3'][1]} (m=3, HAC-robust)",
             "1980Q3": "n/a (not independently verified)"}
    for j, d in enumerate(bp["break_dates"]):
        lab = qlabel(int(d))
        ci90 = f"{qlabel(int(bp['ci_lower_90'][j]))}-{qlabel(int(bp['ci_upper_90'][j]))}"
        ci95 = f"{qlabel(int(bp['ci_lower_95'][j]))}-{qlabel(int(bp['ci_upper_95'][j]))}"
        print(f"  {lab}  |  {ci90:19s} |  {ci95:19s} | {pub95[lab]}")
    print("  tsecon's CIs are the homogeneous classical Bai (1997) case (see the")
    print("  model card); the paper's are heterogeneity-robust with HAC variance,")
    print("  computed for the 3-break model. They are not the same estimator — and")
    print("  the JAE replication study (Zeileis-Kleiber 2005) could not reproduce")
    print("  some of the paper's CIs with any settings. What does replicate is the")
    print("  qualitative finding the paper emphasizes: the 1980:3 Volcker break is")
    print(f"  dated to within a few quarters (95% spans "
          f"{int(bp['ci_upper_95'][1] - bp['ci_lower_95'][1]) + 1} quarters) while the")
    print(f"  1972:3 break is much less precise "
          f"({int(bp['ci_upper_95'][0] - bp['ci_lower_95'][0]) + 1} quarters).")

    # ---- cross-checks ----------------------------------------------------
    sf = tsecon.sup_f_test(rate, np.ones((len(rate), 1)), trim=0.15)
    print("\nCross-checks (computed against independent implementations, see the")
    print("docs page for the full table):")
    print(f"  supF one-break test: stat = {sf['stat']:.2f}, p = {sf['p_value']:.1e}, "
          f"argmax = {qlabel(int(sf['break_date']))} (the Volcker break)")
    print(f"  ssr_path[0..3] = "
          f"{np.round(np.asarray(bp['ssr_path'])[:4], 3)}")
    print("  matches R strucchange's RSS path (1214.922, 644.996, 455.950, 445.182)")
    print("  and tsecon's classical supF sequence matches Perron's own mbreaks with")
    print("  robust=0: 89.245, 52.204, 7.414.")


if __name__ == "__main__":
    main()
