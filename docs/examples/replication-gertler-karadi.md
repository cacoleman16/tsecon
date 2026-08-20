# Replication — Gertler & Karadi (2015)

The canonical proxy-SVAR application: Gertler and Karadi identified a
monetary policy shock in a monthly four-variable VAR from **high-frequency
fed funds futures surprises**, and found that modest policy-rate movements
produce large credit-cost responses and a real-activity decline. This page
replicates the paper on its own data with
[`proxy_svar`](../reference/model-cards/structural-identification.md#proxy_svar-external-instrument-identification-svar-iv) —
reproducing its first-stage strength numbers **verbatim** — and then runs
the modern audit the subsequent literature demanded: valid moving-block
bands ([Jentsch-Lunsford](../reference/model-cards/structural-identification.md#proxy_svar_bands-moving-block-bootstrap-bands-for-the-proxy-svar)),
the Montiel Olea-Pflueger effective F with its honest thresholds
([`proxy_first_stage`](../reference/model-cards/structural-identification.md#proxy_first_stage-the-effective-f-and-the-thresholds-it-must-clear)),
weak-instrument-robust Anderson-Rubin sets
([`proxy_ar_sets`](../reference/model-cards/structural-identification.md#proxy_ar_sets-weak-instrument-robust-anderson-rubin-sets)),
and the Doko Tchatoka-Haque post-1984 subsample where identification
measurably weakens.

```sh
.venv/bin/python docs/examples/replication_gertler_karadi.py
```

The data is the paper's own AEJ replication dataset: 396 months
(1979:7–2012:6) of the 1-year Treasury rate, 100·log CPI, 100·log
industrial production, the Gilchrist-Zakrajšek excess bond premium, and the
published FOMC-surprise instrument series. It is committed at
[`fixtures/gertler_karadi.csv`](../../fixtures/gertler_karadi.csv)
(vendored verbatim from the Plagborg-Møller & Wolf
[`svma_iv`](https://github.com/mikkelpm/svma_iv) mirror and cross-checked
column-by-column against the independent
[VAR-Toolbox](https://github.com/ambropo/VAR-Toolbox) mirror — agreement to
float-print precision), so the replication runs fully offline.

---

## The specification

GK's baseline, exactly: **VAR(12)** in levels with a constant on
(1-year rate, log CPI, log IP, EBP) over the **full 1979:7–2012:6 sample**;
the external instrument is the **three-month-ahead fed funds futures
surprise (FF4)** in a 30-minute window around FOMC announcements, used over
**1991:1–2012:6** (258 months). The VAR's lag coefficients are estimated on
the full sample and the instrument enters only the identification moments —
tsecon's NaN-masked proxy handles the two samples natively:

```python
pr = tsecon.proxy_svar(data, proxy,          # proxy is NaN before 1991:1
                       lags=12, horizon=48, norm_var=0, unit=0.2)
```

`unit=0.2` is the unit-effect normalization: the shock raises the 1-year
rate by exactly **+20bp on impact**, the size of the "representative"
surprise in the paper's Figure 1.

---

## The first stage: the paper's numbers, verbatim

| statistic (FF4 on the 1-year-rate residual) | published | tsecon |
|---|---|---|
| classical first-stage F | 21.55 | **21.55** |
| heteroskedasticity-robust F | ~17.5 | **17.50** |
| effective observations | 258 | 258 |

The robust F **is** the Montiel Olea-Pflueger effective F in this
just-identified case, and the new
[`proxy_first_stage`](../reference/model-cards/structural-identification.md#proxy_first_stage-the-effective-f-and-the-thresholds-it-must-clear)
diagnostic supplies the thresholds the number should be judged against —
which is where the replication stops being a victory lap:

| | effective F | MOP τ=10% bar | certified worst-case bias | folklore F>10 |
|---|---|---|---|---|
| baseline 1979–2012 | 17.50 | 23.11 | **15.5%** | passes |
| post-1984 subsample | 13.82 | 23.11 | **23.3%** | passes |

GK's instrument clears the Stock-Wright-Yogo folklore bar with room to
spare — as the paper says — and still falls short of the MOP τ=10% bar. The
data certify worst-case relative bias below 15.5%, not below 10%. That is
not a refutation of the paper (its F is honestly reported and its result
was corroborated by Montiel Olea-Stock-Watson's weak-IV-robust re-analysis);
it is the demonstration that "F > 10" was never the test, and it is why the
Anderson-Rubin sets below are reported alongside every band.

---

## The impulse responses

To a +20bp policy surprise (responses in percent; rate/EBP in percentage
points; point estimates, full sample):

| h (months) | 1-yr rate | CPI | IP | EBP |
|---|---|---|---|---|
| 0 | +0.200 | −0.034 | +0.030 | +0.116 |
| 6 | +0.132 | −0.020 | −0.139 | +0.068 |
| 12 | +0.066 | −0.030 | −0.302 | +0.020 |
| 18 | −0.037 | −0.070 | −0.400 | +0.022 |
| 24 | −0.086 | −0.095 | −0.425 | +0.013 |
| 36 | −0.069 | −0.135 | −0.337 | −0.006 |
| 48 | −0.007 | −0.134 | −0.190 | −0.013 |

Against the paper's stated shapes (Fig. 1 and text):

* **the 1-year rate** "increases roughly 20 basis points on impact and then
  reverts back to trend after roughly a year" — here +20bp exact (the
  normalization) and below +7bp by month 12, negative by month 18;
* **industrial production** shows "a significant and fairly rapid drop …
  that begins after several months and reaches a trough after roughly 18
  months" — here the response turns negative in month 4 and troughs at
  **−0.43% in month 25**, with month 18 already within 0.03% of the trough
  (the trough is a plateau across months 18–30, not a spike);
* **the CPI** "declines steadily, though this decline is not significant" —
  here a slow drift to −0.13% by month 48, never significant below;
* **the excess bond premium** "increases eight basis points on impact and
  remains at that level for roughly eight months" — here **+12bp** on
  impact, then a ~6bp plateau through month 8, positive through month 31.
  The impact size differs because the normalizations differ (the paper's
  representative surprise is its one-standard-deviation shock, not an exact
  +20bp); the sign, timing and the credit-channel *excess* (a 20bp rate
  move producing a disproportionate spread response) reproduce.

---

## The bands: GK's wild bootstrap vs the valid moving block

GK computed 95% bands with the **wild proxy bootstrap** — the method
Jentsch & Lunsford (2019, AER) later proved is *not* asymptotically valid
for proxy SVARs (a common Rademacher draw leaves the identifying moment
bit-identical across replications). tsecon ships both: `bands="wild"`
reproduces the paper's method (and self-reports
`asymptotically_valid: False`), `bands="moving_block"` is the
Jentsch-Lunsford bootstrap whose validity is proven. Significant horizons
at 95% (Efron percentile bands, 2000 draws, seed 0):

| response | wild (GK's method, invalid) | moving block (valid) |
|---|---|---|
| IP < 0 | h = 7…40 (34 of 49) | h = 25…29 (3 of 49) |
| EBP > 0 | h = 0…8 (9 of 49) | h = 0…3 (2 of 49) |
| CPI < 0 | h = 30…48 (19 of 49) | none |

The wild bands reproduce the paper's significance pattern; the valid bands
are materially wider, and **most of the published IP significance does not
survive the correction at 95%** (at 90%, IP remains significant over
h = 16…40). The credit-cost impact response is what survives at every
level. This is the measured, on-this-data form of the Jentsch-Lunsford
warning, and the reason `proxy_svar_bands` defaults to the moving block.

Both bootstraps report `n_failed: 0` — no draw was lost to a near-zero
denominator, consistent with a first stage of this strength.

---

## Post-1984: identification weakens, output effects dissolve

Doko Tchatoka & Haque (2024, *Economic Record*) re-ran GK excluding the
Volcker disinflation and found that **post-1984, monetary policy shocks
show no significant output effects under weak-identification-robust
inference, despite large credit-cost movements**. Reproduced here: the VAR
re-estimated on 1984:1–2012:6 (same FF4 1991:1+ instrument), with 95%
Anderson-Rubin sets (reduced-form uncertainty propagated) from
`proxy_ar_sets`:

| h | IP set, full sample | IP set, post-1984 |
|---|---|---|
| 6 | [−0.68, +0.34] | [−0.88, +0.57] |
| 12 | [−1.05, +0.21] | [−1.39, +0.51] |
| 18 | [−1.18, +0.10] | [−1.62, +0.50] |
| 24 | [−1.15, +0.06] | [−1.73, +0.44] |
| 36 | [−0.91, +0.09] | [−1.63, +0.38] |
| 48 | [−0.60, +0.15] | [−1.19, +0.45] |

* the effective F falls from 17.50 to **13.82**, the certified worst-case
  bias bound rises from 15.5% to **23.3%** — the new diagnostic flags the
  weakening before any band is drawn;
* every AR set stays bounded in both samples (the relevance statistic
  clears its χ²₁ critical value: 10.0 and 8.7 vs 3.84), but the post-1984
  IP sets are a **median 1.80× wider**;
* the **EBP impact response excludes zero in both samples** (post-1984 at
  h = 0, 2, 4, 6) while **no IP set excludes zero in either** — under
  fully robust inference the credit-cost channel is the finding that
  survives, and post-1984 the output response is unambiguously
  indistinguishable from zero. Exactly the Doko Tchatoka-Haque conclusion.

(That the *full-sample* IP sets also straddle zero pointwise is not a
contradiction of the table above: the AR sets add weak-instrument
robustness *and* reduced-form-coefficient uncertainty on top of a 95%
level — they are the most conservative object on this page, by design.)

---

## Sensitivities, and what matches what

* **Instrument window.** Starting the instrument at its 1990:1 availability
  instead of the paper's 1991:1 moves the effective F from 17.50 to 17.58 —
  nothing changes.
* **HAC first stage.** A Newey-West (5-lag) effective F is 20.36 vs the
  HC1 17.50 — the FF4 surprise's score is close to serially uncorrelated,
  as a surprise series should be.
* **The other published GK instruments** (first stage on the 1-year rate,
  same 1991:1–2012:6 window): `mp1_tc` F = 16.36; the eurodollar surprises
  `ed2/ed3/ed4_tc` F = 6.97 / 4.66 / 4.16 — every one of them below the MOP
  τ=10% bar, the eurodollars below the folklore bar too. The paper's choice
  of FF4 as the baseline is visibly the strongest available instrument.
* **What is deliberately not claimed:** agreement with figure pixels (the
  paper's figures were not digitized — every pinned claim comes from the
  paper's text and tables); the paper's Cholesky comparison column; and its
  GSS-decomposition/forward-guidance extensions (Section IV), which need
  the full futures term structure, not the four instruments vendored here.

**Known deviations, stated plainly.** (i) The IRF normalization is
unit-effect (+20bp exact) where the paper plots a one-standard-deviation
surprise (≈20bp) — shapes are invariant, impact magnitudes of the
non-normalized variables scale accordingly. (ii) tsecon's bands re-estimate
the VAR inside every bootstrap draw with no Kilian bias correction; GK's
wild implementation differs in that detail as well as in validity. (iii)
GK report first-stage statistics for several instrument sets; the 21.55 /
17.5 comparison is their FF4-on-the-1-year-rate entry, the baseline
identification. (iv) The published robust F is quoted at the paper's
printed precision ("17.5"); tsecon's 17.50 matches it at that precision.

The regression test
([`test_replication_gk.py`](../../bindings/python/tests/test_replication_gk.py))
pins the dataset vintage, the verbatim first-stage numbers, the IRF shape
facts, the wild-vs-moving-block significance contrast, and the post-1984
weakening, all offline.

**Citations.** Gertler, M. & Karadi, P. (2015), "Monetary Policy Surprises,
Credit Costs, and Economic Activity," *AEJ: Macroeconomics* 7(1):44–76 —
please cite the paper if you use the data. Jentsch, C. & Lunsford, K.G.
(2019), *AER* 109(7):2655–2678 (wild-bootstrap invalidity) and (2022),
*JBES* 40(4):1876–1891 (moving-block validity). Montiel Olea, J.L. &
Pflueger, C. (2013), *JBES* 31(3):358–369 (effective F). Montiel Olea,
J.L., Stock, J.H. & Watson, M.W. (2021), *J. Econometrics* 225(1):74–87
(SVAR-IV inference). Doko Tchatoka, F. & Haque, Q. (2024), *Economic
Record* 100(329):234–259 (post-1984). Gilchrist, S. & Zakrajšek, E. (2012),
*AER* 102(4):1692–1720 (the excess bond premium). Data mirrors:
Plagborg-Møller & Wolf's `svma_iv` repository; Cesa-Bianchi's VAR-Toolbox.

**See also.**
[`proxy_svar` model card](../reference/model-cards/structural-identification.md#proxy_svar-external-instrument-identification-svar-iv) ·
[`proxy_first_stage`](../reference/model-cards/structural-identification.md#proxy_first_stage-the-effective-f-and-the-thresholds-it-must-clear)
for why 23.11 (not 10) is the bar ·
[`proxy_ar_sets`](../reference/model-cards/structural-identification.md#proxy_ar_sets-weak-instrument-robust-anderson-rubin-sets)
for what an honest confidence statement looks like when instruments weaken ·
the [interval-coverage audit](interval-coverage.md), which measured on
synthetic data the same folklore-threshold failure this page demonstrates
on published data.
