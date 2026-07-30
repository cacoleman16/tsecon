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
# they part company — and they part company badly.
#
# **What this notebook does not do:** invent bands. `tsecon.proxy_svar` returns
# a point estimate. Valid inference for a proxy SVAR needs the Jentsch-Lunsford
# (2019) moving-block bootstrap, which is a documented roadmap item and is not
# in v1. We say so where it matters instead of drawing a shaded region the
# library did not compute.

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
# destroy.

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
# ## The figure
#
# Gertler and Karadi's Figure 1, both identifications on one set of axes, per
# 100 basis points. **No bands** — see the next section for why not.

# %%
H = 48
h = np.arange(H + 1)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
for ax, i, title in zip(axes.ravel(), [2, 0, 1, 3],
                        ["One-year rate", "Industrial production",
                         "CPI", "Excess bond premium"]):
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
# ## Inference: what is missing, and how much it matters
#
# `proxy_svar` returns a point estimate. There is no `lower`, no `upper`, no
# `se`, and this notebook will not manufacture one. Asymptotically valid
# bootstrap inference for a proxy SVAR is the Jentsch-Lunsford (2019)
# moving-block bootstrap — a v2 item on the roadmap. The naive residual
# bootstrap that works for a plain VAR is *not* valid here; Jentsch and
# Lunsford's point is precisely that it gets the coverage wrong, because the
# instrument and the residuals must be resampled jointly and in blocks.
#
# That said, "no bands" is not the same as "no information about precision".
# For a single instrument the robust first-stage $F$ is the square of a
# $t$-statistic, and that $t$ prices the scale of the impact vector.

# %%
t_first = np.sqrt(gk["first_stage_f"])
print(f"robust first-stage F = {gk['first_stage_f']:.2f}  ->  |t| = {t_first:.2f}")
print(f"relative standard error on the first-stage coefficient ~ 1/|t| = "
      f"{100 / t_first:.0f}%")
print()
print(f"The whole IRF path is proportional to that coefficient, so a ~{100/t_first:.0f}% "
      "relative\nstandard error is a LOWER BOUND on the uncertainty in every number in\n"
      "the table above: it ignores sampling error in the 12 lags of VAR\n"
      "coefficients and in the residual covariance. Bands would be wide.")
print()
print(f"Concretely, the IP trough of {irf[trough, 0]:.2f}% per 100bp carries at least "
      f"+/-{abs(1.96 * irf[trough, 0] / t_first):.2f}%\nfrom the first stage alone: "
      f"a range of roughly [{irf[trough,0]*(1+1.96/t_first):.2f}, "
      f"{irf[trough,0]*(1-1.96/t_first):.2f}]. That range is NOT a\n"
      "proxy-SVAR confidence interval, makes no coverage promise, and must not be\n"
      "reported as one — it is a floor on how much room the point estimate has.")

# %% [markdown]
# ### Why not just run LP-IV instead?
#
# `tsecon.lp_iv` *does* report standard errors, and local projections
# instrumented by the same surprise are the natural alternative (Stock-Watson
# 2018). So why not read inference off that instead? Because the two
# estimators do not instrument the same thing, and the diagnostic says so
# loudly.
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
# the paper), and every confidence band in the published figure, because
# `proxy_svar` v1 does not compute one. The paper's bands come from a wild
# bootstrap of both stages; we draw nothing.
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
# **Reproducibility.** Nothing here draws a random number — no bootstrap, no
# simulation, so no seed to set. Every figure in this notebook is a
# deterministic function of the workbook it downloads and the lag order,
# horizon and sample window written above. Re-run it and you get these numbers.
# The only thing that could move them is Ramey's archive being revised, which is
# also why the download is attributed rather than silently vendored.
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
# - Jentsch, C. & Lunsford, K. G. (2019), "Asymptotically Valid Bootstrap
#   Inference for Proxy SVARs", FRB Cleveland working paper — the bands this
#   notebook does not draw.
# - Ramey, V. A. (2016), "Macroeconomic Shocks and Their Propagation",
#   *Handbook of Macroeconomics* vol. 2 — the source of the data used here.
