# Replication — Uhlig (2005)

The Ramey-Zubairy page targets a published *number*. This one targets a
published *shape of uncertainty*: Uhlig (2005) identified a contractionary
monetary policy shock purely from sign restrictions and reported that the
data then **refuse to say** what the shock does to output — the 68% band on
the real-GDP response straddles zero at essentially every horizon, while the
price puzzle of recursive VARs disappears. The width of the band is the
finding, which makes this the natural flagship for
[`sign_restricted_svar`](../reference/model-cards/var-svar.md#sign_restricted_svar-set-identification-by-sign-restrictions):
a replication that must reproduce an *ambiguity*, quantitatively.

```sh
.venv/bin/python docs/examples/replication_uhlig_monetary.py
```

The data is the paper's own: 468 months (1965:1–2003:12) of six US series —
real GDP (interpolated to monthly), the GDP deflator, a commodity price
index, the federal funds rate, nonborrowed reserves, total reserves — all in
100·log except the funds rate, exactly as distributed in the RATS replication
archive and re-distributed by the R package
[VARsignR](https://cran.r-project.org/package=VARsignR) as `uhligdata`. It is
committed at [`fixtures/uhlig2005.csv`](../../fixtures/uhlig2005.csv), so the
replication runs fully offline.

---

## The identification

A **contractionary monetary policy shock** is *defined* as any structural
shock under which, for months $k = 0, \dots, 5$ after impact (Uhlig's
$K = 5$),

* the **GDP deflator** is not positive,
* the **commodity price index** is not positive,
* **nonborrowed reserves** are not positive,
* the **federal funds rate** is not negative,

and — the agnostic part — **real GDP and total reserves are left
unrestricted**, because the output response is the question. In tsecon's
`(variable, shock, horizon, sign)` convention that is 24 tuples:

```python
restr = []
for h in range(6):                      # horizons 0..5 = Uhlig's K = 5
    restr += [(1, 3, h, "-"),           # GDP deflator       not positive
              (2, 3, h, "-"),           # commodity prices   not positive
              (4, 3, h, "-"),           # nonborrowed res.   not positive
              (3, 3, h, "+")]           # federal funds rate not negative

r = tsecon.sign_restricted_svar(data, restr, lags=12, horizon=60,
                                n_draws=2000, max_tries=1000,
                                seed=0, lambda1=10.0)
```

VAR(12) on the monthly levels, 60-month impulse responses, 2000 reduced-form
posterior draws. The run accepts **2000 of 2000** posterior draws in 34,106
Haar rotations (**acceptance rate 5.9%**) and takes about **3 seconds**
through the Rust core.

---

## The result

Pointwise 16/50/84% posterior quantiles (Uhlig's Fig. 6 plots exactly these
three), seed 0. Responses in percent (the data are 100·log), funds rate in
percentage points:

| h (months) | real GDP 16% | 50% | 84% | deflator 16% | 50% | 84% | ffr 50% | commod. 84% | NBR 84% |
|---|---|---|---|---|---|---|---|---|---|
| 0 | −0.007 | +0.097 | +0.190 | −0.063 | −0.036 | −0.014 | +0.197 | −0.375 | −0.394 |
| 3 | +0.001 | +0.145 | +0.277 | −0.100 | −0.059 | −0.027 | +0.221 | −0.447 | −0.339 |
| 6 | −0.072 | +0.073 | +0.209 | −0.118 | −0.067 | −0.029 | +0.114 | −0.690 | −0.181 |
| 12 | −0.041 | +0.091 | +0.212 | −0.207 | −0.136 | −0.071 | +0.033 | −0.905 | −0.184 |
| 24 | −0.110 | +0.036 | +0.165 | −0.366 | −0.232 | −0.111 | −0.051 | −1.052 | −0.028 |
| 36 | −0.119 | +0.039 | +0.190 | −0.522 | −0.325 | −0.162 | −0.085 | −0.778 | +0.202 |
| 48 | −0.128 | +0.041 | +0.212 | −0.633 | −0.392 | −0.184 | −0.089 | −0.465 | +0.372 |
| 60 | −0.142 | +0.035 | +0.214 | −0.707 | −0.427 | −0.191 | −0.072 | −0.197 | +0.448 |

Against the paper's three headline claims:

**(a) No price puzzle.** Uhlig's abstract: the GDP price deflator "falls only
slowly" after a contractionary shock. Here the deflator's **84% quantile is
negative at every one of the 61 horizons** (its maximum is −0.014, at
impact) — the restrictions only impose this for months 0–5; that it *stays*
negative for five years is the replicated finding. The median declines
gradually to −0.43% by month 60: falling, and only slowly. The commodity
price index shows the same pattern (84% quantile negative through month 60).

**(b) The GDP response is ambiguous.** The paper's quantitative statement:
"with a two-thirds probability, a typical shock will move real GDP by up to
0.2 percent" — in *either* direction, "consistent with the conventional view,
but also consistent with monetary neutrality". Here the 16–84% band on real
GDP **straddles zero at all 55 horizons from month 6 to month 60**, and over
those horizons stays within **[−0.14, +0.22]%** — the ~0.2-percent magnitude
of the text. (In months 0–5 the median is *positive*, up to +0.15 — the
short-lived "output puzzle" tilt visible in Uhlig's Fig. 6 — and the 16%
quantile grazes zero at h = 3; the ambiguity claim lives at h ≥ 6, and is
pinned there.)

**(c) The shock looks like monetary policy.** The funds-rate median rises
about +0.2 percentage points on impact and decays within roughly two years;
nonborrowed reserves fall on impact and recover. Those magnitudes are our
run's, not digitized from the paper's figures — the claims pinned against the
paper are the sign/straddle facts and the ~0.2% band magnitude its text
states.

The regression test
([`test_replication_uhlig.py`](../../bindings/python/tests/test_replication_uhlig.py))
pins all of (a)–(c) at 300 draws with a fixed seed — across seeds the
16/50/84 quantiles at these horizons move by only ~0.02–0.04, so the pins
hold with generous margins.

---

## What matches Uhlig's procedure, and what does not

This is Uhlig's **pure-sign-restriction (rejection) approach** — the paper's
benchmark, Fig. 6 — not its penalty-function variant (which trades the hard
accept/reject for a sign-violation penalty and is not implemented here;
tsecon's [`fry_pagan_svar`](../reference/model-cards/structural-identification.md#fry_pagan_svar-the-coherent-draw-the-median-band-is-not)
addresses the "median mixes models" concern the penalty function also
targets). The correspondence and the honest deviations:

* **Impulse-vector distribution — matches.** Uhlig draws an impulse vector
  uniformly on the unit sphere per posterior draw; tsecon draws a full Haar
  rotation and reads off the restricted column. A column of a Haar-distributed
  orthogonal matrix *is* uniform on the sphere, so the candidate distribution
  is identical. The per-shock sign flip (try $-\alpha$ when $\alpha$ fails)
  matches the reference implementation's accept logic (VARsignR
  `uhlig.reject`).
* **Weighting across posterior draws — differs, second order.** tsecon keeps
  the *first* accepted rotation per posterior draw (up to `max_tries`), so
  every reduced-form draw that yields any acceptance is weighted equally.
  Uhlig kept *every* accepted (draw, sub-draw) pair, which weights
  reduced-form draws in proportion to the Haar volume of their accepted set.
  With acceptance rates in the 5–6% range and stable across draws, the
  difference is second order here.
* **Reduced-form prior — differs, made negligible.** Uhlig used a flat
  Normal-inverse-Wishart prior on a VAR with *no constant*. tsecon's sampler
  draws from a Minnesota-NIW posterior with an intercept; the replication
  sets `lambda1 = 10.0`, which loosens the Minnesota shrinkage to the point
  where the likelihood dominates (`lambda1 = 10` and `lambda1 = 2` give bands
  identical to ~0.01). The script re-runs at the library default
  `lambda1 = 0.2` and prints the comparison: band edges move by a few
  hundredths of a percent and no conclusion changes.
* **Strict vs weak inequalities — measure zero.** Uhlig's restrictions are
  weak ("not positive"); tsecon's checker requires strict signs. For
  continuous posterior draws the difference has probability zero.
* **Draw counts.** Uhlig's published bands use all accepted draws from a
  large joint sample; ours use 2000 accepted draws, which is past the point
  where the 16/50/84 quantiles are stable (seed-to-seed movement ~0.02).

What is deliberately **not** claimed: agreement with figure pixels. The
paper's figures were not digitized; every number pinned here comes from the
paper's text and the procedure's own output.

**Citation.** Uhlig, H. (2005), "What are the effects of monetary policy on
output? Results from an agnostic identification procedure," *Journal of
Monetary Economics* 52(2):381–419. Data redistributed from the VARsignR
package (Danne, 2015, CRAN, GPL ≥ 3), originally the paper's RATS replication
archive; the underlying series are US-government statistics (FRED/BEA) plus a
Global Financial Data commodity index. Please cite the paper if you use the
data.

**See also.** [`sign_restricted_svar` model card](../reference/model-cards/var-svar.md#sign_restricted_svar-set-identification-by-sign-restrictions) ·
[guide: causal identification](../guide/08-causal-identification.md), which walks
the same punchline on synthetic data ·
[`robust_svar_bounds`](../reference/model-cards/structural-identification.md#robust_svar_bounds-the-identified-set-without-the-haar-artifact)
for the Baumeister-Hamilton caveat that part of any sign-restriction band is
prior, not evidence — a caveat Uhlig's *ambiguity* finding is robust to, since
removing the Haar prior only widens the set around zero.
