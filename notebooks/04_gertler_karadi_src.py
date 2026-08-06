# %% [markdown]
# # Replicating Gertler & Karadi (2015)
#
# A proxy SVAR — external-instrument identification — on the paper that made
# high-frequency monetary surprises the standard instrument in macro.
#
# Gertler and Karadi identify a monetary policy shock in a four-variable
# monthly VAR (log industrial production, log CPI, the one-year government
# bond rate, the Gilchrist-Zakrajšek excess bond premium) using the surprise
# in **three-month-ahead fed funds futures around FOMC announcements** (FF4)
# as an *external* instrument. The VAR is estimated 1979:7-2012:6; the
# instrument only exists from 1991:1.
#
# The pedagogical payoff is the contrast with a Cholesky ordering, which the
# paper puts side by side in its Figure 1. We reproduce both and show where
# they part company — and in point estimates they part company badly. Whether
# the data can *resolve* that parting turns out to be a different question,
# with a different answer, and it takes the second half of this notebook to
# get there.
#
# **And then a second contrast, about inference rather than identification.**
# The bands in the published figure come from a wild bootstrap, which Jentsch
# and Lunsford (2019) showed is not asymptotically valid for this estimand.
# `tsecon.proxy_svar_bands` computes both — their moving-block bootstrap, and
# the wild bootstrap it replaces — so we do not have to take the critique on
# trust. We reproduce it here, on Gertler and Karadi's own data, in numbers:
# the mechanical reason the wild bootstrap fails is a two-line calculation on
# the residuals that we run below and that comes out to *exactly* zero, and the
# consequence on this data is a valid band half again wider than the invalid
# one and roughly a quarter as many responses called significant.

# %% [markdown]
# ## The data, and where it comes from
#
# This is the hard part of any external-instrument exercise. The FF4 surprise
# series is constructed by Gertler and Karadi from intraday fed funds futures
# quotes; the underlying quotes are proprietary and the constructed series
# lives in their replication package, which is deposited at openICPSR behind a
# login. We could not read its licence, so **nothing is redistributed here**:
# this notebook downloads the data at runtime from Valerie Ramey's replication
# archive for her 2016 *Handbook of Macroeconomics* chapter, which bundles
# Gertler and Karadi's own instrument series alongside the macro series in one
# spreadsheet.
#
# > **Attribution.** `Monetarydat.xlsx`, from
# > *Ramey, V. A. (2016), "Macroeconomic Shocks and Their Propagation",
# > Handbook of Macroeconomics vol. 2* — replication archive
# > `Ramey_HOM_monetary.zip`, `https://econweb.ucsd.edu/~vramey/research.html`.
# > The `FF4_TC` column is the Gertler-Karadi (2015) instrument; `EBP` is the
# > Gilchrist-Zakrajšek (2012) excess bond premium; `LIP`, `LCPI` and `GS1`
# > are logs/levels of Federal Reserve Board and BLS series.
#
# Two things worth being explicit about:
#
# 1. **`tsecon` itself never touches the network.** The library has no HTTP
#    code and no bundled datasets. A notebook is documentation, so it is
#    allowed to fetch; an estimator is not.
# 2. We read the spreadsheet with the **standard library only** (an `.xlsx` is
#    a zip of XML), so this notebook needs no dependency beyond numpy, pandas
#    and matplotlib.

# %%
import io
import os
import tempfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tsecon

ZIP_URL = "https://econweb.ucsd.edu/~vramey/research/Ramey_HOM_monetary.zip"
MEMBER = "Monetarydat.xlsx"
SHEET = "Monthly"
XL = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def read_xlsx_sheet(blob, sheet_name):
    """One worksheet out of an .xlsx, using only the standard library.

    Returns {row_number: {column_index: value}}. An .xlsx is a zip archive of
    XML parts: the workbook names the sheets, a relationship file maps each
    name to its XML part, and strings are pooled in sharedStrings.xml.
    """
    zf = zipfile.ZipFile(io.BytesIO(blob))
    book = ET.fromstring(zf.read("xl/workbook.xml"))
    targets = {r.get("Id"): r.get("Target")
               for r in ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))}
    part = None
    for sh in book.iter(XL + "sheet"):
        if sh.get("name") == sheet_name:
            t = targets[sh.get(REL + "id")]
            part = t if t.startswith("xl/") else "xl/" + t.lstrip("/")
    if part is None:
        raise KeyError(f"no sheet named {sheet_name!r} in the workbook")
    pool = []
    if "xl/sharedStrings.xml" in zf.namelist():
        pool = ["".join(t.text or "" for t in si.iter(XL + "t"))
                for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(XL + "si")]

    def col(ref):                                   # "AB12" -> 27
        n = 0
        for ch in ref:
            if not ch.isalpha():
                break
            n = n * 26 + ord(ch.upper()) - 64
        return n - 1

    out = {}
    for row in ET.fromstring(zf.read(part)).iter(XL + "row"):
        cells = {}
        for c in row.iter(XL + "c"):
            v = c.find(XL + "v")
            if v is None or v.text is None:
                continue
            cells[col(c.get("r"))] = (pool[int(v.text)] if c.get("t") == "s"
                                      else v.text if c.get("t") == "str"
                                      else float(v.text))
        out[int(row.get("r"))] = cells
    return out


def fetch_workbook():
    """The spreadsheet as bytes: a local copy if there is one, else download."""
    override = os.environ.get("TSECON_GK_XLSX")
    if override:
        with open(override, "rb") as fh:
            print(f"using local workbook from TSECON_GK_XLSX={override}")
            return fh.read()
    cache = os.path.join(tempfile.gettempdir(), "ramey_hom_monetary.zip")
    if not os.path.exists(cache):
        try:
            with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
                blob = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"could not download {ZIP_URL} ({exc}).\n"
                "This notebook deliberately redistributes no third-party data, so it\n"
                "needs the archive at runtime. If the URL has moved, find\n"
                "'Macroeconomic Shocks and Their Propagation -> Data and Programs:\n"
                "Monetary Shocks' on Valerie Ramey's research page, download the zip,\n"
                "extract Monetarydat.xlsx, and re-run with\n"
                "  TSECON_GK_XLSX=/path/to/Monetarydat.xlsx"
            ) from exc
        with open(cache, "wb") as fh:
            fh.write(blob)
        print(f"downloaded {len(blob)/1e6:.1f} MB -> cached at {cache}")
    else:
        print(f"using cached archive at {cache}")
    with zipfile.ZipFile(cache) as zf:
        return zf.read(MEMBER)


rows = read_xlsx_sheet(fetch_workbook(), SHEET)
header = {name: j for j, name in rows[1].items()}
cols = ["DATES", "LIP", "LCPI", "GS1", "EBP", "FF4_TC"]
raw = pd.DataFrame({c: [rows.get(r, {}).get(header[c], np.nan)
                        for r in range(2, max(rows) + 1)] for c in cols})
raw = raw[raw.DATES.notna()].reset_index(drop=True)
raw["t"] = np.round((raw.DATES - 1959.0) * 12).astype(int)     # months since 1959:1


def stamp(t):
    """Month index since 1959:1 -> '1979:07'."""
    return f"{1959 + t // 12}:{1 + t % 12:02d}"


print(f"\nworkbook sheet {SHEET!r}: {len(raw)} monthly rows, "
      f"{stamp(raw.t.iloc[0])} to {stamp(raw.t.iloc[-1])}")
for c in cols[1:]:
    ok = raw[c].notna()
    print(f"  {c:8} {ok.sum():4d} obs   "
          f"{stamp(raw.t[ok].iloc[0])} - {stamp(raw.t[ok].iloc[-1])}")

# %% [markdown]
# ## The system
#
# Gertler and Karadi's "simple VAR" is four variables, monthly, 12 lags, a
# constant, 1979:7-2012:6 — the sample starts with Volcker's appointment. The
# instrument runs 1991:1-2012:6, which is *shorter*. That asymmetry is not a
# nuisance, it is the design: the lag coefficients and reduced-form residuals
# are estimated on the full sample, and only the identification step — the
# covariance between the instrument and those residuals — uses the shorter
# window. `proxy_svar` handles this by dropping `NaN` proxy entries from the
# moments and the first stage, so we simply blank the instrument before 1991.

# %%
def month(year, m):
    return (year - 1959) * 12 + (m - 1)


s = raw[(raw.t >= month(1979, 7)) & (raw.t <= month(2012, 6))].reset_index(drop=True)

NAMES = ["log IP", "log CPI", "1-year rate", "EBP"]
POLICY = 2                                   # index of the policy indicator
Y = np.column_stack([100 * s.LIP.values,     # x100 so IRFs read in percent
                     100 * s.LCPI.values,
                     s.GS1.values,           # percent per annum
                     s.EBP.values])          # percentage points
proxy = s.FF4_TC.values.copy()
proxy[s.t.values < month(1991, 1)] = np.nan  # instrument window only

print(f"VAR sample     1979:07 - 2012:06,  T = {len(Y)} months, no missing values: "
      f"{np.isfinite(Y).all()}")
print(f"instrument     1991:01 - 2012:06,  {int(np.isfinite(proxy).sum())} months "
      f"({100 * np.isfinite(proxy).mean():.0f}% of the VAR sample)")

# %% [markdown]
# ### Screening the instrument before using it
#
# An external instrument is supposed to be *news*. If this month's surprise
# were forecastable from last month's, it would not be much of a surprise.
# Let us look, rather than assume.

# %%
z = proxy[np.isfinite(proxy)]
lb = tsecon.ljung_box(z, nlags=12)
print(f"FF4 surprise: mean {z.mean(): .4f}  sd {z.std(ddof=1):.4f}  "
      f"min {z.min(): .3f}  max {z.max(): .3f}  (percentage points)")
print("first four autocorrelations:",
      np.round(np.asarray(tsecon.acf(z, nlags=4)["acf"])[1:], 3))
print(f"Ljung-Box Q(12) = {lb['lb_stat'][-1]:.2f},  p = {lb['lb_pvalue'][-1]:.4f}")

# %% [markdown]
# **The instrument is serially correlated, and clearly so** — first-order
# autocorrelation near 0.3, Ljung-Box rejecting at any conventional level. This
# is a finding, not a defect we should hide, and it has a mechanical
# explanation. The underlying surprises are per-announcement, measured in a
# tight window around each FOMC statement. Gertler and Karadi turn them into a
# monthly series by cumulating them into a running daily series, taking monthly
# *averages* of that, and then first-differencing (their footnote 11). Averaging
# within the month before differencing is a moving-average filter, and it
# induces exactly this kind of positive dependence. A surprise on the 28th of
# the month lands partly in the following month's value.
#
# So the monthly instrument is *not* white noise, and it is not supposed to be.
# What the identification actually requires is orthogonality to non-monetary
# structural shocks in the same month, which is a different and untestable
# condition. Reporting the autocorrelation is worth doing anyway, because it is
# also the reason valid inference here needs a *block* bootstrap: the
# instrument carries serial dependence that an i.i.d. resampling scheme would
# destroy. That is not a rhetorical point in this notebook — we run the block
# bootstrap below and show what the alternative costs.

# %% [markdown]
# ## What the external instrument buys you
#
# A Cholesky ordering identifies the policy shock by *timing assumptions*:
# some variables are assumed not to respond to policy within the month, and
# the central bank is assumed not to respond to others within the month. Those
# are assumptions about a calendar, not about economics, and there are 4! = 24
# of them to choose from.
#
# An external instrument replaces all of that with one moment condition. Let
# $u_t$ be the reduced-form VAR residuals and $m_t$ the instrument. If
#
# - **relevance**: $\mathbb{E}[m_t \varepsilon^{\mathrm{mp}}_t] \neq 0$, and
# - **exogeneity/exclusion**: $\mathbb{E}[m_t \varepsilon^{j}_t] = 0$ for every
#   other structural shock $j$,
#
# then $\mathbb{E}[m_t u_t]$ is proportional to the monetary shock's impact
# column, which pins that column down up to scale. No variable is forced to be
# unresponsive on impact, and no zero is imposed anywhere: the instrument does
# all of the identifying work, which is the point Gertler and Karadi press.
#
# The first condition is testable and we test it. The second is **not
# testable** — it is the assumption you are asking your reader to grant.

# %%
gk = tsecon.proxy_svar(Y, proxy, lags=12, horizon=48,
                       norm_var=POLICY, unit=1.0, trend="c")
gk_classical = tsecon.proxy_svar(Y, proxy, lags=12, horizon=48,
                                 norm_var=POLICY, unit=1.0, trend="c", robust_f=False)

# Published values: the note to Figure 1 of Gertler & Karadi (2015), AEJ:
# Macroeconomics 7(1), plus column (2) of their Table 3 for the first-stage
# slope and its robust t.
PUBLISHED = {"F": 21.55, "robust F": 17.64, "R2 (%)": 7.76, "n": 258,
             "slope": 1.151, "robust |t|": 4.184}

# The first-stage slope is a regression of the policy residual on the
# instrument, so cov(m, u) / var(m) recovers it from the returned moments.
slope = gk["cov_um"][POLICY] / np.var(z, ddof=1)

print("first stage: the instrument against the policy variable's VAR residual")
print(f"{'':22}{'measured':>10}{'published':>11}")
print(f"{'observations':22}{gk['n_proxy']:>10}{PUBLISHED['n']:>11}")
print(f"{'slope on FF4':22}{slope:>10.3f}{PUBLISHED['slope']:>11.3f}")
print(f"{'robust |t|':22}{np.sqrt(gk['first_stage_f']):>10.3f}"
      f"{PUBLISHED['robust |t|']:>11.3f}")
print(f"{'F':22}{gk_classical['first_stage_f']:>10.2f}{PUBLISHED['F']:>11.2f}")
print(f"{'robust F':22}{gk['first_stage_f']:>10.2f}{PUBLISHED['robust F']:>11.2f}")
print(f"{'reliability, R^2 (%)':22}{100 * gk['reliability']:>10.2f}"
      f"{PUBLISHED['R2 (%)']:>11.2f}")

# %% [markdown]
# That is the replication landing. Five independently-reported first-stage
# quantities reproduce to within about a percent, on the same 258 observations.
#
# **The $F$ and the reliability say different things.** The robust $F$ of about 17.6 clears the
# rule-of-thumb threshold of 10 that Stock, Wright and Yogo (2002) — and
# Gertler and Karadi, citing them — use to argue a weak-instrument problem is
# unlikely. The **reliability** statistic, $\mathrm{Corr}(m_t, u^{\mathrm{mp}}_t)^2$,
# says something different and less comfortable: the surprise explains under
# 8% of the monthly innovation in the one-year rate. Both are true. A strong
# instrument in the weak-IV sense can still be a small-$R^2$ instrument, and
# the small $R^2$ is why the *scale* of the identified impact vector is the
# shakiest part of the whole exercise.
#
# ### Is the first stage a free parameter?
#
# The residual being instrumented depends on the lag order, so the $F$ is not
# invariant to it. Worth knowing how much freedom that gives you.

# %%
print(f"{'lags':>6}{'robust F':>11}{'F':>9}{'reliability %':>15}{'obs':>7}")
for p in (3, 6, 12, 18, 24):
    a = tsecon.proxy_svar(Y, proxy, lags=p, horizon=1, norm_var=POLICY)
    b = tsecon.proxy_svar(Y, proxy, lags=p, horizon=1, norm_var=POLICY, robust_f=False)
    print(f"{p:>6}{a['first_stage_f']:>11.2f}{b['first_stage_f']:>9.2f}"
          f"{100 * a['reliability']:>15.2f}{a['n_proxy']:>7}")
print("\n12 lags — Gertler-Karadi's choice — is the specification that reproduces")
print("both published F statistics most closely, and the instrument weakens")
print("sharply beyond it. Shorter lag orders look *better* on the F, which is a")
print("useful reminder that a first-stage F is a property of a specification.")

# %% [markdown]
# ## The unit-effect normalisation
#
# The moment condition identifies the impact column only up to scale, so scale
# is a *choice*. `proxy_svar` uses the unit-effect normalisation: `norm_var`
# rises by `unit` on impact. With `norm_var=2, unit=1.0` a positive shock
# raises the one-year rate by exactly 1 percentage point on impact, and every
# other response is "per 100 basis points".
#
# Gertler and Karadi instead plot a **one-standard-deviation** shock. Those are
# the same object with a different ruler, and the ruler is measurable: the
# standard deviation of the identified shock series tells us how big a typical
# shock is in the unit-effect metric.

# %%
irf = np.asarray(gk["irf"])
shock = np.asarray(gk["shock"])
sd = shock.std(ddof=1)

print("impact column (per 100bp on the one-year rate):")
for nm, v in zip(NAMES, np.asarray(gk["impact"])):
    print(f"   {nm:14}{v: .4f}")
print(f"\nidentified shock series: {len(shock)} obs (the VAR residual sample), "
      f"sd = {sd:.4f}")
print(f"-> a one-standard-deviation shock raises the one-year rate {100 * sd:.1f} bp "
      "on impact")
print("   Gertler-Karadi report 'roughly 25 basis points' for their one-sd shock.")
print(f"\nOver the instrument window alone the sd is "
      f"{shock[-int(gk['n_proxy']):].std(ddof=1):.4f} "
      f"({100 * shock[-int(gk['n_proxy']):].std(ddof=1):.1f} bp) — the 'one sd' ruler")
print("depends on which sample you measure it over. We use the full residual")
print("sample below and say so; nothing about the estimate itself changes.")

# %% [markdown]
# ## The impulse responses
#
# Two rulers, one estimate. The left block is per 100 bp; the right block is
# the same path scaled to a one-standard-deviation shock, which is what the
# published figure plots.

# %%
SHORT = ["IP", "CPI", "1y", "EBP"]
print("            per 100bp" + " " * 30 + "|  scaled to one sd")
print("         " + "".join(f"{nm:>13}" for nm in NAMES)
      + "   |" + "".join(f"{nm:>9}" for nm in SHORT))
for h in (0, 1, 3, 6, 12, 18, 24, 30, 36, 48):
    print(f"  h={h:<5}" + "".join(f"{irf[h, i]:13.3f}" for i in range(4))
          + "   |" + "".join(f"{sd * irf[h, i]:9.3f}" for i in range(4)))

trough = int(np.argmin(irf[:, 0]))
print(f"\nIP trough: {irf[trough, 0]:.2f}% per 100bp at h = {trough}, "
      f"i.e. {sd * irf[trough, 0]:.2f}% for a one-sd shock")

# %% [markdown]
# ### Against the published figure
#
# Gertler and Karadi describe Figure 1 in the text rather than tabulating it,
# so the honest comparison is against the magnitudes they state.

# %%
print(f"{'':34}{'measured':>11}   published")
print(f"{'one-year rate, impact (bp)':34}{100 * sd * irf[0, 2]:>11.1f}   ~25")
print(f"{'IP trough (%)':34}{sd * irf[trough, 0]:>11.2f}   ~-0.5")
print(f"{'IP trough horizon (months)':34}{trough:>11d}   ~18")
print(f"{'EBP on impact (bp)':34}{100 * sd * irf[0, 3]:>11.1f}   ~10")
print(f"{'EBP at h=6 (bp)':34}{100 * sd * irf[6, 3]:>11.1f}   ~7 for another half year")
print(f"{'CPI at h=24 (%)':34}{sd * irf[24, 1]:>11.2f}   small, insignificant decline")

# %% [markdown]
# Magnitudes land: the one-standard-deviation shock is a 23 bp tightening
# rather than 25, industrial production troughs at about half a percent, the
# excess bond premium jumps 13 bp on impact (a third more than the ~10 the
# paper reports) and is still elevated by about 8 bp half a year out, and the
# CPI decline is small. The one clear discrepancy
# is **timing**: our IP trough is at 25 months rather than the "roughly a year
# and a half" the paper describes. Reading a trough off a published
# figure is imprecise, but the difference is real and we do not tune it away —
# the candidate explanations are the data vintage (this is Ramey's 2016
# extract, not Gertler and Karadi's 2013 vintage of industrial production and
# the excess bond premium) and the exact treatment of the crisis months.
#
# ## The Cholesky contrast
#
# This is the comparison the paper leads with. Under the recursive scheme
# Gertler and Karadi consider, the one-year rate is ordered **second to last**
# and the excess bond premium last: the Fed may respond within the month to
# output and prices but not to the excess bond premium, and the premium may
# respond to policy on impact. Our column order is already exactly that.

# %%
chol = np.asarray(tsecon.var_irf(Y, lags=12, horizon=48, orth=True, trend="c"))
raw_impact = chol[0, POLICY, POLICY]
chol_irf = chol[:, :, POLICY] / raw_impact          # rescale to 100bp on impact

print(f"one-sd Cholesky shock moves the one-year rate {100 * raw_impact:.1f} bp on "
      "impact; rescaled to 100bp below\n")
print("           " + "".join(f"{nm:>13}" for nm in NAMES))
for h in (0, 1, 3, 6, 12, 18, 24, 36, 48):
    print(f"  h={h:<7}" + "".join(f"{chol_irf[h, i]:13.3f}" for i in range(4)))

ip_pk = int(np.argmax(chol_irf[:25, 0]))
cpi_pk = int(np.argmax(chol_irf[:25, 1]))
print(f"\nunder Cholesky, a *tightening* raises IP to {chol_irf[ip_pk, 0]:+.2f}% at "
      f"h={ip_pk} (output puzzle)")
print(f"                            and the CPI to {chol_irf[cpi_pk, 1]:+.2f}% at "
      f"h={cpi_pk} (price puzzle)")
print(f"and the excess bond premium *falls* on impact, {chol_irf[0, 3]:+.3f} pp, "
      f"against {irf[0, 3]:+.3f} pp with the instrument")

# %% [markdown]
# Both classic puzzles, in the sign the paper reports, plus a third: credit
# spreads *narrow* after a monetary tightening, which no theory wants.
#
# Gertler and Karadi's reading is that the recursive restriction is the
# problem. The Fed does look at credit spreads; a low excess bond premium
# signals a strong economy and invites tightening. Forbid the Fed from
# reacting to the premium within the month and that reverse causation is
# loaded onto the "shock", which then arrives with a strong economy attached —
# so output and prices rise and spreads fall. The external instrument does not
# have to take a stand on the timing, and the puzzles go away.
#
# ### The ordering is doing the work — and only for Cholesky
#
# Worth making concrete. Permute the variables and re-identify: the proxy SVAR
# is invariant (the moment condition does not know about column order), while
# the Cholesky answer changes.

# %%
perm = [POLICY, 3, 0, 1]                      # [1-year rate, EBP, log IP, log CPI]
gk_perm = tsecon.proxy_svar(Y[:, perm], proxy, lags=12, horizon=48,
                            norm_var=0, unit=1.0, trend="c")
gap = np.abs(np.asarray(gk_perm["irf"]) - irf[:, perm]).max()
print(f"proxy SVAR, IRFs after permuting columns: max abs difference = {gap:.2e}")
print(f"           first-stage robust F: {gk['first_stage_f']:.4f} vs "
      f"{gk_perm['first_stage_f']:.4f}")

chol_perm = np.asarray(tsecon.var_irf(Y[:, perm], lags=12, horizon=48, trend="c"))
alt = chol_perm[:, 2, 0] / chol_perm[0, 0, 0]       # IP response, policy ordered first
print("\nCholesky IP response per 100bp, two orderings:")
print(f"{'h':>4}{'[IP,CPI,1y,EBP]':>18}{'[1y,EBP,IP,CPI]':>18}")
for h in (0, 6, 12, 24, 36):
    print(f"{h:>4}{chol_irf[h, 0]:>18.3f}{alt[h]:>18.3f}")
print("\nSame data, same lag order, different story. That is the cost of")
print("identifying a shock with a calendar convention.")

# %% [markdown]
# ## Inference: the bands the paper drew, and the bands that are valid
#
# Gertler and Karadi's Figure 1 has shaded regions. They come from a wild
# bootstrap — a common Rademacher sign flip applied to the reduced-form
# residuals and to the instrument — which is what Mertens and Ravn (2013) used
# before them and what most of the proxy-SVAR literature used after. Jentsch
# and Lunsford (2019) showed that this is *not* an asymptotically valid
# bootstrap for a proxy SVAR, and proposed a **moving-block** bootstrap that
# is: the joint pair $(u_t, m_t)$ is resampled in overlapping blocks under one
# set of block starts, the VAR is re-estimated inside every draw, and the
# unit-effect normalisation is re-imposed per draw.
#
# `proxy_svar_bands` computes both, so the critique is checkable rather than
# citable. Same data, same lag order, same number of draws, same seed.

# %%
H, ALPHA, N_BOOT, SEED = 48, 0.10, 2000, 0
BAND_KW = dict(lags=12, horizon=H, norm_var=POLICY, unit=1.0, trend="c",
               alpha=ALPHA, n_boot=N_BOOT, seed=SEED)
mbb = tsecon.proxy_svar_bands(Y, proxy, bands="moving_block", **BAND_KW)
wild = tsecon.proxy_svar_bands(Y, proxy, bands="wild", **BAND_KW)

for label, r in (("moving block", mbb), ("wild", wild)):
    print(f"{label:14} asymptotically_valid={str(r['asymptotically_valid']):5}  "
          f"block_length={r['block_length']:3}  n_used={r['n_used']}  "
          f"n_failed={r['n_failed']}")
print(f"\nfailed draws by reason (moving block): {dict(mbb['failures'])}")
print(f"failure_warning: {mbb['failure_warning']}")
print(f"\nthe h=0 response of the policy variable, moving-block 90% band: "
      f"[{np.asarray(mbb['lower'])[0, POLICY]:.6f}, "
      f"{np.asarray(mbb['upper'])[0, POLICY]:.6f}]")
print("that cell is degenerate by construction — the unit-effect normalisation")
print("pins it in every draw — and a non-degenerate value there would mean the")
print("normalisation had been hoisted out of the bootstrap loop.")
print("\nwhat the library says about the wild bootstrap it just computed:")
print(f"  {wild['validity_note']}")

# %% [markdown]
# ### Why the wild bootstrap cannot help here, in two lines of arithmetic
#
# The identifying moment of a proxy SVAR is
# $\hat\gamma \propto \sum_t m_t \hat u_t'$. The wild bootstrap draws
# $e_t \in \{-1, +1\}$ and sets $u^*_t = e_t \hat u_t$ and $m^*_t = e_t m_t$ —
# *the same* $e_t$ on both, because flipping the residual without flipping the
# instrument would destroy the correlation that identifies the shock. But then
#
# $$\sum_t m^*_t u^{*\prime}_t = \sum_t e_t^2\, m_t \hat u_t' = \sum_t m_t \hat u_t'$$
#
# because $e_t^2 = 1$. The identifying moment is not merely similar across
# draws; it is the *same floating-point number*. Whatever the wild bootstrap is
# resampling, it is not the uncertainty in the object that does the
# identification. That is checkable on the real data, so let us check it — and
# first confirm that the residuals we compute by hand are the ones the library
# is using.

# %%
P = 12
Xlag = np.column_stack([np.ones(len(Y) - P)]
                       + [Y[P - lag:len(Y) - lag] for lag in range(1, P + 1)])
U = Y[P:] - Xlag @ np.linalg.lstsq(Xlag, Y[P:], rcond=None)[0]  # reduced-form residuals
inst = np.isfinite(proxy[P:])                                  # the instrument window
mk, Uk = proxy[P:][inst], U[inst]

gamma = ((mk - mk.mean())[:, None] * Uk).sum(0) / inst.sum()
print("our residual moment vs the library's cov_um: max abs difference "
      f"{np.abs(gamma - np.asarray(gk['cov_um'])).max():.2e}  (same residuals)")

base = (mk[:, None] * Uk).sum(0)                  # sum_t m_t u_t', as in the theory
base_c = ((mk - mk.mean())[:, None] * Uk).sum(0)  # as the estimator computes it
rng = np.random.default_rng(0)
worst = worst_c = 0.0
for _ in range(200):
    e = rng.choice([-1.0, 1.0], size=len(mk))          # one Rademacher draw...
    m_star, U_star = e * mk, e[:, None] * Uk           # ...applied to BOTH
    worst = max(worst, float(np.abs((m_star[:, None] * U_star).sum(0) - base).max()))
    worst_c = max(worst_c, float(np.abs(
        ((m_star - m_star.mean())[:, None] * U_star).sum(0) - base_c).max()))
print(f"\nidentifying moment, sum_t m_t u_t':  {np.round(base, 6)}")
print(f"max deviation over 200 common-Rademacher draws: {worst:.3e}")
print("Not 'small'. Zero, exactly, in every draw, because e_t^2 = 1.")
print(f"\nsame check on the demeaned moment the estimator actually forms: "
      f"{worst_c:.3e}")
print("nonzero, but that is the sample-mean correction wobbling, not the")
print("covariance: the identifying content itself does not move at all.")

# %% [markdown]
# So the wild scheme injects **no** variability into the moment that identifies
# the shock. Two caveats keep that from being a claim of a literally zero-width
# band, and both are worth stating. First, the estimator subtracts the proxy's
# *sample* mean before forming the moment, and a sign-flipped sample has a
# different sample mean, so the centred version above does move a little —
# that is the centering constant wobbling, not the covariance. Second, the
# pseudo-data are regenerated and the VAR is re-estimated inside every draw, so
# the reduced-form part of the uncertainty is represented. What is missing is
# the part that is specific to external-instrument identification — the part
# Jentsch and Lunsford's paper is about — and the library's per-draw
# diagnostics show the size of the gap.

# %%
for label, r in (("moving block", mbb), ("wild", wild)):
    g = np.asarray(r["gamma_norm_draws"])
    f = np.asarray(r["first_stage_f_draws"])
    print(f"{label:14} sd of gamma[norm_var] across draws {g.std(ddof=1):.3e}   "
          f"sd of the first-stage F {f.std(ddof=1):6.2f}")
ratio_g = (np.asarray(mbb["gamma_norm_draws"]).std(ddof=1)
           / np.asarray(wild["gamma_norm_draws"]).std(ddof=1))
print(f"\nthe valid bootstrap finds {ratio_g:.1f}x more dispersion in the identifying")
print("moment than the wild bootstrap does, on identical data and draw count.")

# %% [markdown]
# ### What that costs, in band width and in conclusions
#
# The Hall (basic) band is the recommended one and is what we report; the
# `lower_efron`/`upper_efron` percentile band is what the original papers plot,
# and the two differ when the bootstrap distribution is skewed. The
# $(h{=}0,\ \text{policy})$ cell is dropped from every count below because it is
# degenerate under both schemes.

# %%
lo_m, up_m = np.asarray(mbb["lower"]), np.asarray(mbb["upper"])
lo_w, up_w = np.asarray(wild["lower"]), np.asarray(wild["upper"])
free = np.ones((H + 1, 4), bool)
free[0, POLICY] = False                       # the degenerate normalisation cell

print("90% bands per 100bp, moving block vs wild")
print(f"{'':22}{'point':>8}{'moving block':>22}{'wild':>22}{'width ratio':>13}")
for hz in (0, 6, 12, trough, 36, 48):
    for i in range(4):
        row = f"  h={hz:<3} {NAMES[i]:14}{irf[hz, i]:>8.3f}"
        if free[hz, i]:
            ratio = (up_m[hz, i] - lo_m[hz, i]) / (up_w[hz, i] - lo_w[hz, i])
            print(row
                  + f"{f'[{lo_m[hz, i]:+.3f}, {up_m[hz, i]:+.3f}]':>22}"
                  + f"{f'[{lo_w[hz, i]:+.3f}, {up_w[hz, i]:+.3f}]':>22}"
                  + f"{ratio:>12.2f}x")
        else:
            print(row + f"{'[degenerate at 1]':>22}{'[degenerate at 1]':>22}"
                        f"{'':>13}")
    print()

w_m, w_w = (up_m - lo_m)[free], (up_w - lo_w)[free]
r_w = w_m / w_w
print(f"width ratio, moving block / wild, over {free.sum()} non-degenerate cells:")
print(f"   median {np.median(r_w):.2f}x   min {r_w.min():.2f}x   max {r_w.max():.2f}x")
print(f"   moving block is the wider band in {(r_w > 1).sum()} of {r_w.size} cells")
grid = np.full(free.shape, -np.inf)          # the degenerate cell is 0/0; keep it out
grid[free] = r_w
worst_cell = np.unravel_index(np.argmax(grid), grid.shape)
print(f"   widest gap at h={worst_cell[0]}, {NAMES[worst_cell[1]]}")


def excludes_zero(lo, up):
    return int(((lo > 0) | (up < 0))[free].sum())


print(f"\ncells whose 90% band excludes zero, of {free.sum()}:")
print(f"   moving block {excludes_zero(lo_m, up_m):4d}")
print(f"   wild         {excludes_zero(lo_w, up_w):4d}")
print(f"\nIP at the trough, h={trough}: point {irf[trough, 0]:.2f}% per 100bp")
print(f"   moving block  [{lo_m[trough, 0]:+.2f}, {up_m[trough, 0]:+.2f}]")
print(f"   wild          [{lo_w[trough, 0]:+.2f}, {up_w[trough, 0]:+.2f}]")
print(f"   Hall vs Efron (moving block): [{lo_m[trough, 0]:+.2f}, {up_m[trough, 0]:+.2f}]"
      f" vs [{np.asarray(mbb['lower_efron'])[trough, 0]:+.2f}, "
      f"{np.asarray(mbb['upper_efron'])[trough, 0]:+.2f}]")

# %% [markdown]
# **That is the Jentsch-Lunsford critique, reproduced on a real replication.**
# The valid band is wider than the invalid one in 194 of the 195 non-degenerate
# cells, by a median factor of 1.47 and by as much as 3.3 at the impact
# horizon, where the identifying moment is doing the most work and the wild
# scheme's blind spot is therefore largest. The exception — one cell where the
# wild band is 7% wider — is the kind of noise you should expect from two
# bootstrap distributions with 2000 draws each; the direction of the effect is
# systematic, and it is one-sided.
#
# The consequence for what you would *write down* is larger than the width
# numbers suggest, because significance is a threshold and widths near the
# threshold matter most. At a nominal 90%, the wild bootstrap declares 110 of
# the 195 cells significantly different from zero. The valid bootstrap declares
# 28 — a factor of four fewer claims about the world, from the same estimate on
# the same data, differing only in how the resampling was done.
#
# The headline result survives, but not comfortably. The industrial-production
# trough of $-2.09\%$ per 100bp has a moving-block band of $[-3.87, -0.05]$ —
# still signed, and only just: the upper end is five hundredths of a percent
# from zero, where the wild band puts it at $-0.94$. Note also that Hall and
# Efron disagree materially here ($[-3.87, -0.05]$ against $[-4.13, -0.30]$),
# which is the signature of a skewed bootstrap distribution; the original
# papers report the percentile (Efron) band, `tsecon` recommends and defaults
# to reporting Hall.
#
# ## The figure
#
# Gertler and Karadi's Figure 1, both identifications on one set of axes, per
# 100 basis points, with the moving-block band on the instrument-identified
# response. The band is **pointwise**: it covers each (horizon, variable) cell
# at 90%, not the whole path simultaneously. `tsecon` has no simultaneous
# (sup-$t$) band for this estimator, so we do not claim one.

# %%
h = np.arange(H + 1)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for ax, i, title in zip(axes.ravel(), [2, 0, 1, 3],
                        ["One-year rate", "Industrial production",
                         "CPI", "Excess bond premium"]):
    ax.fill_between(h, lo_m[:, i], up_m[:, i], color="C0", alpha=0.18, lw=0,
                    label="moving-block 90% (pointwise)")
    ax.plot(h, irf[:H + 1, i], lw=2, color="C0", label="external instrument (FF4)")
    ax.plot(h, chol_irf[:H + 1, i], lw=1.6, color="C3", ls="--", label="Cholesky")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("percent" if i in (0, 1) else "percentage points", fontsize=8)
axes[0][0].legend(fontsize=8, frameon=False)
for ax in axes[1]:
    ax.set_xlabel("months")
fig.suptitle("Gertler-Karadi (2015): one-year rate shock, per 100bp, 1979:7-2012:6",
             fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### The choice of bootstrap changes what the figure says
#
# Look at the red dashed Cholesky line against the shaded region. Now that
# there *is* a region, we can ask how often the recursive path escapes it — and
# ask it of both bootstraps.

# %%
for label, lo, up in (("moving block (valid)", lo_m, up_m),
                      ("wild (not valid)   ", lo_w, up_w)):
    esc = (chol_irf[:H + 1] < lo) | (chol_irf[:H + 1] > up)
    print(f"Cholesky response outside the 90% band, {label}: {esc[free].sum():3d} of "
          f"{free.sum()} cells")
    for i in range(4):
        hs = [hz for hz in range(H + 1) if free[hz, i] and esc[hz, i]]
        if hs:
            print(f"      {NAMES[i]:14}{len(hs):3d} cells, horizons {min(hs)}-{max(hs)}")

# %% [markdown]
# **63 of 195, against zero of 195.** With the published procedure's bands you
# would report that the recursive identification is significantly rejected at
# 63 horizon-variable cells — 38 of the 49 CPI cells, the excess bond premium
# over its first fourteen months, industrial production between months 12 and
# 20. With the valid bands you cannot reject it anywhere. The choice of
# resampling scheme is the difference between a decisive empirical rejection
# and no rejection at all, on identical point estimates.
#
# Be careful about what the second answer does and does not say. It is *not* a
# test that the two identifications agree, and it is not a joint statement
# about paths: it says the proxy SVAR on its own is not precise enough to
# reject the recursive answer cell by cell, on 258 months of instrument. The
# argument for the external instrument was never that it beats Cholesky in a
# horse race of confidence bands — it is that its identifying assumption is one
# you can state and argue about, where the recursive assumption is a claim
# about a calendar that changes the answer when you permute the columns. That
# argument is untouched. But anyone reading the two point-estimate paths as a
# decisive rejection is reading more than this sample supports, and an invalid
# bootstrap is what made that reading look safe.
#
# ### Both bootstraps, on one set of axes
#
# The same responses with the two bands on top of each other. The
# shaded region is the valid one; the dashed red pair is what the published
# procedure draws.

# %%
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for ax, i, title in zip(axes.ravel(), [2, 0, 1, 3],
                        ["One-year rate", "Industrial production",
                         "CPI", "Excess bond premium"]):
    ax.fill_between(h, lo_m[:, i], up_m[:, i], color="C0", alpha=0.18, lw=0,
                    label="moving block (valid)")
    ax.plot(h, lo_w[:, i], lw=1.3, color="C3", ls="--", label="wild (not valid)")
    ax.plot(h, up_w[:, i], lw=1.3, color="C3", ls="--")
    ax.plot(h, irf[:H + 1, i], lw=2, color="C0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("percent" if i in (0, 1) else "percentage points", fontsize=8)
axes[0][0].legend(fontsize=8, frameon=False)
for ax in axes[1]:
    ax.set_xlabel("months")
fig.suptitle("Two 90% bands for the same estimate, 2000 draws, same seed",
             fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The other kind of honesty: weak-instrument-robust sets
#
# The moving-block bootstrap fixes the *bootstrap*. It does not fix weak
# identification — its asymptotics are strong-instrument asymptotics, and a
# Wald-type band around a ratio whose denominator might be near zero is exactly
# the object that fails when the instrument is not strong.
#
# This data set is the interesting case for that, because its two strength
# diagnostics disagree, as we saw above: a robust first-stage $F$ near 18, well
# clear of the rule-of-thumb 10, alongside a reliability $R^2$ under 8%. The
# moving-block draws let us say something sharper than either number — the
# bootstrap distribution of the first-stage $F$ itself.

# %%
fdraw = np.asarray(mbb["first_stage_f_draws"])
rdraw = np.asarray(mbb["reliability_draws"])
print(f"first-stage F : point {mbb['point_first_stage_f']:.2f},  "
      f"bootstrap 5th pct {np.quantile(fdraw, 0.05):.2f},  "
      f"median {np.median(fdraw):.2f},  95th pct {np.quantile(fdraw, 0.95):.2f}")
print(f"                {100 * (fdraw < 10).mean():.0f}% of moving-block draws fall "
      "below the rule-of-thumb F = 10")
print(f"reliability   : point {mbb['point_reliability']:.4f},  "
      f"5th pct {np.quantile(rdraw, 0.05):.4f},  "
      f"95th pct {np.quantile(rdraw, 0.95):.4f}")

# %% [markdown]
# The point estimate clears 10 comfortably; a substantial minority of resamples
# of the same data do not. That is precisely the configuration in which a
# weak-instrument-robust set is worth computing alongside the Wald band rather
# than instead of it.
#
# `proxy_ar_sets` inverts an Anderson-Rubin statistic in closed form. What
# comes back need not be an interval: it can be the *complement* of an interval
# (two rays), the whole line, or a single point. Dufour (1997) is the reason —
# no bounded confidence set can be valid when a parameter may be unidentified,
# so an honest procedure has to be allowed to return an unbounded answer. The
# shape is the finding, and we print whatever shape we get.

# %%
ar = tsecon.proxy_ar_sets(Y, proxy, lags=12, horizon=H, norm_var=POLICY, unit=1.0,
                          trend="c", alpha=ALPHA, variance="hc0")
cells = ar["cells"]


def describe(c):
    """Render one AR set honestly, whatever shape it turned out to be."""
    if c["kind"] in ("interval", "point"):
        return f"[{c['lower']:+.3f}, {c['upper']:+.3f}]"
    if c["kind"] == "exterior":                       # two rays, NOT an interval
        return f"everything outside ({c['excluded_lower']:+.3f}, {c['excluded_upper']:+.3f})"
    if c["kind"] == "ray_below":
        return f"(-inf, {c['upper']:+.3f}]"
    if c["kind"] == "ray_above":
        return f"[{c['lower']:+.3f}, +inf)"
    return {"whole": "the whole real line", "empty": "empty"}[c["kind"]]


shapes = {}
for hz in range(H + 1):
    for i in range(4):
        shapes[cells[hz][i]["kind"]] = shapes.get(cells[hz][i]["kind"], 0) + 1
print(f"level {ar['level']}, AR critical value {ar['critical_value']:.3f}, "
      f"n_proxy {ar['n_proxy']}, reduced-form uncertainty "
      f"{ar['reduced_form_uncertainty']}")
print(f"boundedness statistic {ar['ar_bound_stat']:.2f} vs critical value "
      f"{ar['critical_value']:.2f}  ->  every set bounded: {ar['ar_bounded_all']}")
print(f"set shapes over all {4 * (H + 1)} cells: {shapes}")

print(f"\n{'':22}{'AR set':>24}{'moving-block band':>24}")
for hz in (0, 6, 12, trough, 36, 48):
    for i in range(4):
        print(f"  h={hz:<3} {NAMES[i]:14}{describe(cells[hz][i]):>24}"
              f"{f'[{lo_m[hz, i]:+.3f}, {up_m[hz, i]:+.3f}]':>24}")
    print()

ar_lo = np.array([[cells[hz][i]["lower"] for i in range(4)] for hz in range(H + 1)],
                 dtype=float)
ar_up = np.array([[cells[hz][i]["upper"] for i in range(4)] for hz in range(H + 1)],
                 dtype=float)
n_ex0 = sum(cells[hz][i]["excludes_zero"] for hz in range(H + 1) for i in range(4)
            if free[hz, i])
print(f"cells whose AR set excludes zero, of {free.sum()}: {n_ex0}"
      f"   (moving block: {excludes_zero(lo_m, up_m)}, wild: {excludes_zero(lo_w, up_w)})")
if ar["ar_bounded_all"]:
    ra = (ar_up - ar_lo)[free] / w_m
    print(f"AR set width / moving-block width: median {np.median(ra):.2f}x, "
          f"min {ra.min():.2f}x, max {ra.max():.2f}x")

# %% [markdown]
# **The shapes we actually got: 195 bounded intervals and one point.** The
# point is the $(h{=}0,\ \text{one-year rate})$ cell, which the normalisation
# fixes at exactly 1 — the same degeneracy the bootstrap band shows, arrived at
# by completely different algebra. Nothing here is an exterior set, a ray or
# the whole line, and that is a *result*: the statistic that governs
# boundedness comes out at 10.11 against a critical value of 2.71, far enough
# from the non-identified boundary that every cell closes. Had we been handed a
# weaker
# instrument, `describe` above would be printing "everything outside
# $(a, b)$" — two rays, which must never be reported as though it were the
# interval $[a, b]$ — or "the whole real line". We wrote the branches; this
# data set did not need them.
#
# So what did the robust set buy? Not much width — a median of 1.09 times the
# moving-block band, ranging from 0.86 to 1.35, so it is not uniformly the
# wider object either. That is the *good* case, and it is worth seeing: when
# the instrument really is strong enough, the weak-IV-robust set and the Wald
# band land in much the same place, and the agreement is evidence rather than
# assumption. It buys two things all the same. First, its validity does not
# rest on the strong-instrument asymptotics that the moving-block bootstrap
# needs, which matters here precisely because 36% of the bootstrap draws put
# the first-stage $F$ below 10 even though the point estimate is 17.58. Second,
# it disagrees in informative places: at the trough ($h = 25$) it gives
# $[-4.74, -0.13]$ for industrial production against the band's
# $[-3.87, -0.05]$, and at the same horizon it signs the one-year rate's own
# reversal ($[-1.02, -0.03]$) where the band does not ($[-0.79, +0.08]$).
# Counting cells, the AR sets exclude zero 38 times against the moving-block
# band's 28 — the robust object is not a widened version of the Wald band, it
# is a different statistic, and it is not uniformly less decisive.

# %% [markdown]
# ### What reduced-form uncertainty costs
#
# An AR set for a proxy SVAR has two sources of sampling error: the identifying
# moment, and the VAR coefficients that map it into a response at horizon $h$.
# Conditioning on the estimated VAR — treating $\hat\Psi_h$ as if it were the
# truth — makes the algebra easier and the sets much narrower. It also destroys
# the coverage: the library's own Monte Carlo puts nominal-95% coverage at
# 0.119 by $h = 8$ without propagation and 0.913 with it. `proxy_ar_sets`
# propagates by default, and returns `level = None` if you switch it off,
# because a set conditional on the reduced form has no honest $1-\alpha$ label.
# On this data the difference is not subtle.

# %%
ar_cond = tsecon.proxy_ar_sets(Y, proxy, lags=12, horizon=H, norm_var=POLICY,
                               unit=1.0, trend="c", alpha=ALPHA, variance="hc0",
                               reduced_form_uncertainty=False)
cc = ar_cond["cells"]
n_ex0_cond = sum(cc[hz][i]["excludes_zero"] for hz in range(H + 1) for i in range(4)
                 if free[hz, i])
print(f"reduced-form uncertainty propagated: level={ar['level']}, "
      f"{n_ex0} of {free.sum()} cells exclude zero")
print(f"reduced-form uncertainty omitted:    level={ar_cond['level']}, "
      f"{n_ex0_cond} of {free.sum()} cells exclude zero")
print(f"\n{'':22}{'propagated':>24}{'conditional on the VAR':>26}")
for hz in (6, 12, trough, 48):
    for i in (0,):
        print(f"  h={hz:<3} {NAMES[i]:14}{describe(cells[hz][i]):>24}"
              f"{describe(cc[hz][i]):>26}")
flips = [(hz, i) for hz in range(H + 1) for i in range(4)
         if free[hz, i] and cc[hz][i]["excludes_zero"]
         and not cells[hz][i]["excludes_zero"]]
print(f"\nDropping the reduced-form uncertainty flips {len(flips)} of the "
      f"{free.sum()} cells from\n'contains zero' to 'excludes zero'. Those "
      "narrower sets are not a 90% anything,\nand the library refuses to label "
      "them as one — which is why `level` came back None.")

# %% [markdown]
# ### Putting the three inference objects on one picture
#
# Point estimate, moving-block band, wild band, AR set — industrial production,
# the response the paper's headline is about.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.fill_between(h, ar_lo[:, 0], ar_up[:, 0], color="C2", alpha=0.13, lw=0,
                label="Anderson-Rubin 90% set (weak-IV robust)")
ax.fill_between(h, lo_m[:, 0], up_m[:, 0], color="C0", alpha=0.22, lw=0,
                label="moving-block 90% band")
ax.plot(h, lo_w[:, 0], lw=1.3, color="C3", ls="--", label="wild 90% band (not valid)")
ax.plot(h, up_w[:, 0], lw=1.3, color="C3", ls="--")
ax.plot(h, irf[:H + 1, 0], lw=2.2, color="C0", label="point estimate")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("months")
ax.set_ylabel("percent per 100bp")
ax.set_title("Industrial production: three answers to 'how sure are we?'", fontsize=11)
ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Why not just run LP-IV instead?
#
# Local projections instrumented by the same surprise are the natural
# alternative (Stock-Watson 2018), and `tsecon.lp_iv` reports standard errors
# without any of the bootstrap machinery above. So why not sidestep the whole
# question and read the inference off that? Because the two estimators do not
# instrument the same thing, and the diagnostic says so loudly.
#
# The proxy SVAR instruments the *reduced-form residual* of the one-year rate —
# the rate with 12 lags of all four variables projected out — and there the
# surprise explains 8% with a robust $F$ of 17.6. `lp_iv` instruments the
# **level** of the impulse, and its control set is (by the documented
# convention) a constant plus `n_lag_controls` lags of the **outcome**. When the
# outcome is industrial production, the policy rate's own lags are therefore
# *not* controlled for, and the first stage is asked to explain the level of a
# near-random-walk interest rate with a monthly surprise of standard deviation
# 0.05. Watch it fail, and then watch why:

# %%
keep = np.isfinite(proxy)
lp_ip = tsecon.lp_iv(Y[keep, 0], Y[keep, POLICY], proxy[keep],
                     horizons=36, n_lag_controls=2)
lp_rate = tsecon.lp_iv(Y[keep, POLICY], Y[keep, POLICY], proxy[keep],
                       horizons=36, n_lag_controls=2)
f_ip = np.asarray(lp_ip["first_stage_f"])
f_rate = np.asarray(lp_rate["first_stage_f"])
b, se = np.asarray(lp_ip["irf"]), np.asarray(lp_ip["se"])

print(f"LP-IV on the same instrument, 1991:01-2012:06 ({int(keep.sum())} obs)")
print(f"  outcome = log IP   (controls: lags of IP)    effective F "
      f"{f_ip.min():6.3f} - {f_ip.max():6.3f}")
print(f"  outcome = 1y rate  (controls: lags of 1y)    effective F "
      f"{f_rate.min():6.3f} - {f_rate.max():6.3f}")
print(f"  proxy SVAR (controls: 12 lags of all four)   robust F    "
      f"{gk['first_stage_f']:.3f}")
print(f"\nresponse of log IP per 1pp of the one-year rate, from the first row above:")
print(f"{'h':>4}{'coefficient':>14}{'se':>12}")
for hh in (0, 12, 24, 36):
    print(f"{hh:>4}{b[hh]:>14.2f}{se[hh]:>12.2f}")
print("\nThe diagnosis is not 'the instrument is weak' — it is strong (F ~ 15) the")
print("moment the policy rate's own lags are in the control set. It is that this")
print("particular LP-IV design leaves them out, so the coefficients above (an")
print("11-25 percent IP response to 100bp!) and their standard errors describe a")
print("regression nobody should run. Having standard errors is not the same as")
print("having inference. Read the F first, every time.")

# %% [markdown]
# ## What replicates, what does not, and what you must assume
#
# **Replicated, to about a percent:** the first-stage slope (1.154 vs 1.151),
# its robust $|t|$ (4.19 vs 4.184), the first-stage $F$ (21.8 vs 21.55), the
# robust $F$ (17.6 vs 17.64), the reliability $R^2$ (7.9% vs 7.76%) and the
# observation count (258). Then, from the figure's text: the size of a
# one-standard-deviation shock (23 bp vs "roughly 25"), the depth of the
# industrial-production trough (about -0.5%), the impact and persistence of the
# excess bond premium (13 bp then ~8 bp, against ~10 then ~7), and — the
# qualitative headline — the output puzzle, price puzzle and *narrowing* credit
# spread that a Cholesky ordering produces on the same data.
#
# **Not replicated:** the timing of the IP trough (25 months here, about 18 in
# the paper). We do not tune that away; the candidates remain the data vintage
# and the treatment of the crisis months.
#
# **Deliberately not matched: the published bands.** We can now draw them —
# `bands="wild"` is exactly the published procedure — and we can show on this
# data that they are too narrow, by a median factor of 1.47 in width and by a
# factor of about four in the number of responses they call significant. The
# band this notebook plots is therefore *not* the band in the paper, on
# purpose. The bands are also **pointwise**, at both stages: neither the
# moving-block band nor the AR set covers the whole response path
# simultaneously, and `tsecon` has no sup-$t$ band for this estimator.
#
# **A finding the point estimates hid.** With valid bands, the Cholesky
# response lies inside the proxy-SVAR band at all 195 non-degenerate cells; with
# the wild bands it lies outside at 63. The identification contrast is stark in
# point estimates and unresolved at pointwise 90% — which does not weaken the
# case for the instrument (that case is about the assumption you have to
# defend, not about band width) but does change what a reader is entitled to
# conclude from the figure.
#
# **Assumed, not shown:**
#
# - *Exclusion.* The FF4 surprise is uncorrelated with every non-monetary
#   structural shock in the month of the announcement. Nothing above tests
#   this. Gertler and Karadi's own Table 4 finds the surprises are partly
#   predictable from the Fed's internal forecasts, which is a direct challenge
#   to it; Miranda-Agrippino and Ricco (2021) and Bauer and Swanson (2023)
#   build on that.
# - *No anticipation within the month.* Monthly averaging of daily surprises
#   is a modelling choice; a surprise late in the month is treated the same as
#   one early in it. We measured the cost of that choice above — a first-order
#   autocorrelation near 0.3 in the instrument itself.
# - *Invertibility.* The proxy SVAR needs the structural shocks to be
#   recoverable from the four-variable VAR residuals. Add a fifth variable and
#   the identified shock changes.
# - *One shock, not four.* An external instrument identifies **one column** of
#   the impact matrix. There is no full structural decomposition here, and no
#   FEVD for the other three shocks.
# - *Strong-instrument asymptotics, for the bootstrap.* The moving-block band
#   fixes the resampling scheme, not weak identification. The AR sets are the
#   part of this notebook that does not need that assumption, and on this data
#   they broadly agree with the band — which is a check that passed, not a
#   guarantee it will pass on your instrument.
#
# ## Provenance, stated plainly
#
# We committed no third-party data. The Gertler-Karadi instrument's canonical
# home is their openICPSR replication package, which is behind a login; we
# could not read its licence terms and therefore did not redistribute the
# series, not even the single column we use. Ramey's archive is fetched at
# runtime, attributed above, and cached in your temp directory. Four of the
# five series we use (industrial production, the CPI, the one-year rate, and
# the Fed's published excess bond premium) are public-domain US government
# output and could be assembled from FRED and federalreserve.gov directly; the
# FF4 surprise is the one series with no clean public substitute, because the
# intraday futures quotes behind it are proprietary. That is the real
# constraint in this literature, and it is worth knowing about before planning
# work that depends on it.
#
# **Reproducibility.** The point estimates, the Cholesky contrast and the AR
# sets draw no random numbers at all — they are deterministic functions of the
# workbook, the lag order, the horizon and the sample window written above. The
# two bootstraps do, and they are pinned: `n_boot=2000, seed=0`, printed in the
# code rather than buried in a default. The one place we call a random number
# generator ourselves — the Rademacher demonstration — is seeded too, and its
# answer is exactly zero for every seed, which is the whole point. Re-run this
# notebook and you get these numbers. The only thing that could move them is
# Ramey's archive being revised, which is also why the download is attributed
# rather than silently vendored.
#
# ## Further reading
#
# - [Model card: proxy SVAR](https://cacoleman16.github.io/tsecon/reference/model-cards/structural-identification/)
# - [Guide chapter 8 — structural identification](https://cacoleman16.github.io/tsecon/guide/08-causal-identification/)
# - Gertler, M. & Karadi, P. (2015), "Monetary Policy Surprises, Credit Costs,
#   and Economic Activity", *AEJ: Macroeconomics* 7(1), 44-76.
# - Stock, J. H. & Watson, M. W. (2018), "Identification and Estimation of
#   Dynamic Causal Effects in Macroeconomics Using External Instruments",
#   *Economic Journal* 128, 917-948.
# - Jentsch, C. & Lunsford, K. G. (2019), "The Dynamic Effects of Personal
#   Income Tax Changes on Macroeconomic Aggregates: A Reassessment",
#   *American Economic Review* — the wild-bootstrap critique and the
#   moving-block replacement, reproduced above on this data. Their
#   asymptotic theory also appears as a Federal Reserve Bank of Cleveland
#   working paper, "Asymptotically Valid Bootstrap Inference for Proxy SVARs".
# - Montiel Olea, J. L., Stock, J. H. & Watson, M. W. (2021), "Inference in
#   Structural Vector Autoregressions Identified with an External Instrument",
#   *Journal of Econometrics* — the weak-instrument-robust sets.
# - Dufour, J.-M. (1997), "Some Impossibility Theorems in Econometrics with
#   Applications to Structural and Dynamic Models", *Econometrica* — why an
#   honest confidence set has to be allowed to be unbounded.
# - Ramey, V. A. (2016), "Macroeconomic Shocks and Their Propagation",
#   *Handbook of Macroeconomics* vol. 2 — the source of the data used here.
