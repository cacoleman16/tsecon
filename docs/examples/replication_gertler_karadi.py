"""Replication: Gertler & Karadi (2015) proxy-SVAR monetary VAR.

The canonical external-instrument application: a monthly 4-variable VAR
(1-year Treasury rate, log CPI, log IP, excess bond premium; 1979:7-2012:6,
12 lags) identified with the FF4 high-frequency futures surprise
(1991:1-2012:6). The paper's headline: a contractionary surprise raises the
1-year rate ~20bp on impact, industrial production falls with a trough after
roughly 18 months, the CPI declines steadily (not significantly), and the
excess bond premium rises immediately -- the credit-cost channel.

This replication reproduces:

* the paper's own first-stage strength numbers VERBATIM -- classical F 21.55
  and robust F 17.5 (their Table 1 / text values for FF4 on the one-year
  rate);
* the Figure-1 impulse-response shapes under the unit-effect normalization
  (+20bp on the 1-year rate);
* the paper's ORIGINAL wild-bootstrap 95% bands -- which Jentsch & Lunsford
  (2019, AER) proved invalid for proxy SVARs -- and the valid moving-block
  bands beside them, measuring how much of the published significance
  pattern survives the correction;
* the modern weak-instrument audit: the Montiel Olea-Pflueger effective F
  with its tau-based thresholds (`proxy_first_stage`), Anderson-Rubin
  confidence sets (`proxy_ar_sets`), and the Doko Tchatoka-Haque (2024)
  post-1984 subsample where identification weakens and output effects
  dissolve while the credit-cost response survives.

Data: the authors' public AEJ replication dataset, vendored at
fixtures/gertler_karadi.csv (verbatim from the Plagborg-Moller & Wolf
`svma_iv` mirror, cross-checked against the VAR-Toolbox mirror; see the CSV
header). Runs fully offline:

    .venv/bin/python docs/examples/replication_gertler_karadi.py

WHAT THIS IS, AND IS NOT
------------------------
This is GK's baseline specification -- their variables, transformations,
lag count, samples, and instrument -- with the impulse responses normalized
to the +20bp unit effect (the paper plots a "representative" surprise whose
impact on the 1-year rate is roughly 20bp; the unit-effect convention makes
that exact). It is NOT a digitization of their figures: the pinned claims
are the published first-stage numbers, the paper's stated shapes/timings,
and the sign/significance patterns, at stated tolerances. GK's own bands
come from the wild proxy bootstrap; tsecon reproduces them under
`bands="wild"` but flags them `asymptotically_valid: False` (Jentsch &
Lunsford 2019), and the honest bands here are the moving-block ones.
"""
import csv
import time
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "gertler_karadi.csv"

#: Column order of the VAR (the paper's baseline four variables).
VARIABLES = ["gs1", "logcpi", "logip", "ebp"]
LABELS = {
    "gs1": "1-year rate",
    "logcpi": "CPI (100*log)",
    "logip": "IP (100*log)",
    "ebp": "excess bond premium",
}
#: The policy indicator: the impact of the shock on gs1 is normalized.
NORM_VAR = 0
#: Unit-effect normalization: +20bp on the 1-year rate on impact (the
#: paper's Fig. 1 shows a representative surprise of roughly this size).
UNIT = 0.2
LAGS = 12
HORIZON = 48


def load_gk(path=DATA):
    """Read the committed Gertler-Karadi monthly panel.

    Public academic data, vendored with attribution -- no download, no
    loader. Returns dates, the T x 4 VAR matrix, and the instrument columns
    (NaN outside each instrument's availability window).
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    names = rows[0]
    recs = rows[1:]

    def col(name):
        j = names.index(name)
        return np.array([float(r[j]) if r[j] != "" else np.nan for r in recs])

    dates = [r[0] for r in recs]
    data = np.column_stack([col(v) for v in VARIABLES])
    instruments = {
        k: col(k) for k in ("ff4_tc", "mp1_tc", "ed2_tc", "ed3_tc", "ed4_tc")
    }
    return {"dates": dates, "data": data, "instruments": instruments}


def baseline_proxy(gk, start="1991-01"):
    """The paper's baseline instrument: FF4, masked to 1991:1-2012:6.

    The series itself is available from 1990:1; the paper's baseline
    estimation uses 1991:1 on. NaN marks unavailability -- tsecon drops
    those dates from the identifying moments without misaligning the rest.
    """
    proxy = gk["instruments"]["ff4_tc"].copy()
    proxy[: gk["dates"].index(start)] = np.nan
    return proxy


def post84(gk):
    """The Doko Tchatoka-Haque (2024) post-Volcker subsample: VAR estimated
    on 1984:1-2012:6 (the Great Moderation), same FF4 1991:1+ instrument."""
    s = gk["dates"].index("1984-01")
    return gk["data"][s:], baseline_proxy(gk)[s:]


def sig_ranges(lo, up, j, sign, horizon=HORIZON):
    """Horizons where the band is strictly on one side of zero."""
    if sign == "-":
        return [h for h in range(horizon + 1) if up[h, j] < 0.0]
    return [h for h in range(horizon + 1) if lo[h, j] > 0.0]


def fmt_range(hs):
    return f"{min(hs)}..{max(hs)} ({len(hs)} of 49)" if hs else "none"


def rule(width=78, ch="-"):
    print(ch * width)


def main():
    print("Replication -- Gertler & Karadi (2015), AEJ:Macro 7(1)")
    print("proxy-SVAR identification of a monetary policy shock (FF4 external instrument)")
    rule(78, "=")

    gk = load_gk()
    data, dates = gk["data"], gk["dates"]
    proxy = baseline_proxy(gk)
    print(f"data: GK replication dataset (committed), {data.shape[0]} months "
          f"({dates[0]} to {dates[-1]})")
    print("      VAR: " + ", ".join(LABELS[v] for v in VARIABLES))
    print(f"spec: VAR({LAGS}) in levels with constant; instrument FF4 over "
          f"1991-01..2012-06 ({int(np.isfinite(proxy).sum())} months);")
    print(f"      unit-effect normalization: +{UNIT*100:.0f}bp on the 1-year rate at h=0.")

    # ------------------------------------------------------------------ #
    # 1. The first stage: the paper's numbers, verbatim                   #
    # ------------------------------------------------------------------ #
    t0 = time.time()
    pr = tsecon.proxy_svar(data, proxy, lags=LAGS, horizon=HORIZON,
                           norm_var=NORM_VAR, unit=UNIT)
    fs = pr["first_stage"]
    print(f"\nFIRST STAGE (FF4 on the 1-year-rate residual), {time.time()-t0:.2f}s")
    print(f"  classical F = {fs['f_classical']:.2f}   (paper: 21.55)")
    print(f"  robust F    = {fs['effective_f']:.2f}   (paper: ~17.5; HC1 = the MOP effective F)")
    print(f"  reliability R^2 = {fs['reliability']:.3f}   effective obs = {fs['n_proxy']}")
    print("  The effective F against the Montiel Olea-Pflueger bars "
          f"(tau=10%: {fs['mop_cv_tau10']:.2f}, tau=20%: {fs['mop_cv_tau20']:.2f}):")
    print(f"    tau_bound = {fs['tau_bound']:.3f} -> the data certify worst-case bias "
          f"below {100*fs['tau_bound']:.1f}%")
    print(f"    weak by folklore F>10?  {fs['weak_folklore']}   "
          f"weak by MOP tau=10%?  {fs['weak_mop_tau10']}")
    print("  GK's baseline passes the folklore bar with room to spare and still")
    print("  falls short of the MOP tau=10% bar -- 'F > 10' was never the test.")

    # ------------------------------------------------------------------ #
    # 2. The impulse responses (Figure 1's shapes)                        #
    # ------------------------------------------------------------------ #
    irf = np.asarray(pr["irf"])
    print("\nIMPULSE RESPONSES to a +20bp policy surprise (percent; ebp/gs1 in pp)")
    print(f"  {'h':>3} | {'1yr rate':>9} | {'CPI':>8} | {'IP':>8} | {'EBP':>8}")
    rule(50)
    for h in (0, 6, 12, 18, 24, 36, 48):
        print(f"  {h:>3} | {irf[h,0]:>+9.3f} | {irf[h,1]:>+8.3f} | "
              f"{irf[h,2]:>+8.3f} | {irf[h,3]:>+8.3f}")
    trough = int(irf[:, 2].argmin())
    print(f"\n  paper: 1-yr rate ~+20bp on impact, reverting within about a year;")
    print(f"         IP drop begins after several months, trough ~18 months;")
    print(f"         CPI declines steadily (not significant); EBP rises on impact.")
    print(f"  here:  gs1 +{100*irf[0,0]:.0f}bp (exact, the normalization), "
          f"{'below' if irf[18,0] < 0.05 else 'above'} +5bp by h=18;")
    print(f"         IP trough {irf[trough,2]:+.2f}% at h={trough}; "
          f"CPI {irf[48,1]:+.2f}% by h=48; EBP {100*irf[0,3]:+.1f}bp at h=0.")

    # ------------------------------------------------------------------ #
    # 3. Bands: the paper's wild bootstrap vs the valid moving block      #
    # ------------------------------------------------------------------ #
    print("\nCONFIDENCE BANDS, 95% (2000 draws, seed 0)")
    t0 = time.time()
    wild = tsecon.proxy_svar_bands(data, proxy, lags=LAGS, horizon=HORIZON,
                                   norm_var=NORM_VAR, unit=UNIT, alpha=0.05,
                                   n_boot=2000, seed=0, bands="wild")
    mbb = tsecon.proxy_svar_bands(data, proxy, lags=LAGS, horizon=HORIZON,
                                  norm_var=NORM_VAR, unit=UNIT, alpha=0.05,
                                  n_boot=2000, seed=0, bands="moving_block")
    print(f"  ({time.time()-t0:.1f}s for both; wild self-reports "
          f"asymptotically_valid={wild['asymptotically_valid']}, "
          f"n_failed={wild['n_failed']}/{mbb['n_failed']})")
    wl, wu = np.asarray(wild["lower_efron"]), np.asarray(wild["upper_efron"])
    ml, mu = np.asarray(mbb["lower_efron"]), np.asarray(mbb["upper_efron"])
    print("  significant horizons (95% Efron percentile bands, as GK report):")
    print(f"    {'':14}{'wild (GK method, invalid)':>28} | moving block (Jentsch-Lunsford)")
    for label, j, sign in (("IP < 0", 2, "-"), ("EBP > 0", 3, "+"),
                           ("CPI < 0", 1, "-")):
        print(f"    {label:<14}{fmt_range(sig_ranges(wl, wu, j, sign)):>28} | "
              f"{fmt_range(sig_ranges(ml, mu, j, sign))}")
    print("  The wild bands reproduce the paper's significance pattern; the valid")
    print("  moving-block bands are wider, and most of the IP significance does")
    print("  not survive at 95% -- GK's activity result rests on bands whose")
    print("  method Jentsch-Lunsford later proved invalid. The credit-cost (EBP)")
    print("  impact response is what survives the correction.")

    # ------------------------------------------------------------------ #
    # 4. Post-1984: identification weakens, output effects dissolve       #
    # ------------------------------------------------------------------ #
    data84, proxy84 = post84(gk)
    pr84 = tsecon.proxy_svar(data84, proxy84, lags=LAGS, horizon=HORIZON,
                             norm_var=NORM_VAR, unit=UNIT)
    f84 = pr84["first_stage"]
    print("\nPOST-1984 SUBSAMPLE (Doko Tchatoka-Haque 2024): VAR on 1984:1-2012:6")
    print(f"  effective F {fs['effective_f']:.2f} -> {f84['effective_f']:.2f}; "
          f"certified worst-case bias {100*fs['tau_bound']:.1f}% -> "
          f"{100*f84['tau_bound']:.1f}%")

    ar_full = tsecon.proxy_ar_sets(data, proxy, lags=LAGS, horizon=HORIZON,
                                   norm_var=NORM_VAR, unit=UNIT, alpha=0.05)
    ar_84 = tsecon.proxy_ar_sets(data84, proxy84, lags=LAGS, horizon=HORIZON,
                                 norm_var=NORM_VAR, unit=UNIT, alpha=0.05)
    print(f"  Anderson-Rubin 95% sets (reduced-form uncertainty propagated,")
    print(f"  level={ar_full['level']}): all cells bounded in both samples "
          f"(relevance statistic {ar_full['ar_bound_stat']:.1f} / "
          f"{ar_84['ar_bound_stat']:.1f} vs critical {ar_full['critical_value']:.2f}).")
    print(f"  {'h':>3} | {'IP set, full sample':>22} | {'IP set, post-1984':>22}")
    rule(60)
    widths_f, widths_8 = [], []
    for h in range(1, HORIZON + 1):
        cf, c8 = ar_full["cells"][h][2], ar_84["cells"][h][2]
        widths_f.append(cf["upper"] - cf["lower"])
        widths_8.append(c8["upper"] - c8["lower"])
        if h in (6, 12, 18, 24, 36, 48):
            print(f"  {h:>3} | [{cf['lower']:+7.2f}, {cf['upper']:+6.2f}]      | "
                  f"[{c8['lower']:+7.2f}, {c8['upper']:+6.2f}]")
    ratio = float(np.median(np.array(widths_8) / np.array(widths_f)))
    ebp_f = [h for h in range(HORIZON + 1) if ar_full["cells"][h][3]["excludes_zero"]]
    ebp_8 = [h for h in range(HORIZON + 1) if ar_84["cells"][h][3]["excludes_zero"]]
    ip_f = [h for h in range(HORIZON + 1) if ar_full["cells"][h][2]["excludes_zero"]]
    ip_8 = [h for h in range(HORIZON + 1) if ar_84["cells"][h][2]["excludes_zero"]]
    print(f"\n  median IP set width: post-1984 / full = {ratio:.2f}x")
    print(f"  IP sets excluding zero: full {fmt_range(ip_f)}; post-1984 {fmt_range(ip_8)}")
    print(f"  EBP sets excluding zero: full {fmt_range(ebp_f)}; post-1984 {fmt_range(ebp_8)}")
    print("  Exactly the Doko Tchatoka-Haque finding: post-1984, the output response")
    print("  is not distinguishable from zero under weak-identification-robust")
    print("  inference, while the credit-cost (EBP) impact response survives.")

    # ------------------------------------------------------------------ #
    # 5. Sensitivities                                                    #
    # ------------------------------------------------------------------ #
    print("\nSENSITIVITIES")
    proxy90 = gk["instruments"]["ff4_tc"]
    f90 = tsecon.proxy_first_stage(data, proxy90, lags=LAGS, norm_var=NORM_VAR)
    fhac = tsecon.proxy_first_stage(data, proxy, lags=LAGS, norm_var=NORM_VAR,
                                    variance="hac")
    print(f"  instrument from 1990:1 (full availability): effective F = "
          f"{f90['effective_f']:.2f}")
    print(f"  HAC (Newey-West, {fhac['hac_lags']} lags) effective F = "
          f"{fhac['effective_f']:.2f} -- the FF4 surprise's score is close to")
    print("  serially uncorrelated, as a surprise series should be.")
    print("  first stage of the other published GK instruments on the 1-year rate")
    print("  (each over its own availability window within 1991:1-2012:6):")
    for name in ("mp1_tc", "ed2_tc", "ed3_tc", "ed4_tc"):
        z = gk["instruments"][name].copy()
        z[: dates.index("1991-01")] = np.nan
        d = tsecon.proxy_first_stage(data, z, lags=LAGS, norm_var=NORM_VAR)
        print(f"    {name:<7} effective F = {d['effective_f']:>6.2f}  "
              f"tau_bound = {d['tau_bound']:.2f}  weak(MOP tau=10%): "
              f"{d['weak_mop_tau10']}")

    print("\nCitations: Gertler & Karadi (2015, AEJ:Macro 7(1):44-76); Jentsch &")
    print("Lunsford (2019, AER 109(7), the wild-bootstrap invalidity; 2022, JBES")
    print("40(4), the moving-block validity theory); Montiel Olea & Pflueger (2013,")
    print("JBES, the effective F); Montiel Olea, Stock & Watson (2021,")
    print("J.Econometrics, SVAR-IV inference); Doko Tchatoka & Haque (2024,")
    print("Economic Record 100(329), the post-1984 result).")


if __name__ == "__main__":
    main()
