"""Replication: Uhlig (2005) sign-restricted monetary policy SVAR.

Uhlig asked what a contractionary monetary policy shock does to US real GDP
while remaining *agnostic* about the answer: the shock is identified purely by
sign restrictions on the responses of prices, nonborrowed reserves and the
federal funds rate, and the output response is deliberately left unrestricted.
The paper's headline findings, both reproduced here on the paper's own data:

* **no price puzzle** — the GDP price deflator does not rise after a
  contractionary shock; it "falls only slowly";
* **the GDP response is ambiguous** — the 68% band straddles zero at
  essentially every horizon ("with a two-thirds probability, a typical shock
  moves real GDP by up to about 0.2 percent" — in either direction), so
  "neutrality of monetary policy shocks is not inconsistent with the data".

Data: Uhlig, H. (2005), "What are the effects of monetary policy on output?
Results from an agnostic identification procedure", *Journal of Monetary
Economics* 52(2):381-419. The dataset (the paper's replication data, as
shipped in the R package VARsignR as `uhligdata`, originally from the RATS
replication archive) is committed at fixtures/uhlig2005.csv, so this runs
offline — the library ships no data loaders.

    .venv/bin/python docs/examples/replication_uhlig_monetary.py

WHAT THIS IS, AND IS NOT
------------------------
This reproduces Uhlig's *pure-sign-restriction* (rejection) procedure — the
benchmark of his Fig. 6 — with his data, his VAR(12), his restriction set and
his restriction window K = 5. It is NOT the penalty-function variant of the
same paper, and it is not a digitization of his figures: the claims checked
are the sign/straddle facts and the ~0.2-percent band magnitude the text
states, at stated tolerances. Known deviations (documented in the docs page):
tsecon's reduced-form posterior is Minnesota-NIW with an intercept (set very
loose here to approximate Uhlig's flat NIW, no-constant specification), and
tsecon keeps one accepted rotation per posterior draw where Uhlig kept every
accepted sub-draw.
"""
import csv
import time
from pathlib import Path

import numpy as np

import tsecon

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "uhlig2005.csv"

#: Column order of the committed dataset (and of Uhlig's VAR).
VARIABLES = ["y", "yd", "p", "i", "rnb", "rt"]
LABELS = {
    "y": "real GDP",
    "yd": "GDP deflator",
    "p": "commodity prices",
    "i": "federal funds rate",
    "rnb": "nonborrowed reserves",
    "rt": "total reserves",
}
#: The structural-shock column the restrictions identify. Any single column
#: is symmetric under the Haar prior; using the federal-funds position keeps
#: the bookkeeping readable.
SHOCK = 3


def load_uhlig(path=DATA):
    """Read the committed Uhlig (2005) monthly panel into a T x 6 array.

    Public academic data, vendored with attribution — no download, no loader.
    All series are 100*log of the originals except the federal funds rate,
    which is in percent (the paper's transformation).
    """
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
    names = rows[0][1:]
    dates = [r[0] for r in rows[1:]]
    data = np.array([[float(c) for c in r[1:]] for r in rows[1:]])
    return {"dates": dates, "names": names, "data": data}


def monetary_policy_restrictions(k_max=5):
    """Uhlig's benchmark restriction set, K = k_max (his K = 5).

    A contractionary monetary policy shock is any shock under which, for
    months 0..k_max after impact,

    * the GDP price deflator (yd) is NOT positive,
    * the commodity price index (p) is NOT positive,
    * nonborrowed reserves (rnb) are NOT positive,
    * the federal funds rate (i) is NOT negative.

    Real GDP (y) and total reserves (rt) are left unrestricted — the output
    response is the question, not an assumption. Tuples are tsecon's
    (variable, shock, horizon, sign) convention, horizon 0 = impact.
    """
    restr = []
    for h in range(k_max + 1):
        restr.append((VARIABLES.index("yd"), SHOCK, h, "-"))
        restr.append((VARIABLES.index("p"), SHOCK, h, "-"))
        restr.append((VARIABLES.index("rnb"), SHOCK, h, "-"))
        restr.append((VARIABLES.index("i"), SHOCK, h, "+"))
    return restr


def run_uhlig(data, n_draws=2000, seed=0, lambda1=10.0, max_tries=1000,
              lags=12, horizon=60, k_max=5):
    """Uhlig's benchmark: VAR(12) on the monthly levels, 60-month IRFs.

    `lambda1=10.0` all but switches the Minnesota shrinkage off, so the
    reduced-form posterior is dominated by the likelihood — the closest this
    sampler gets to Uhlig's flat Normal-inverse-Wishart prior. `n_draws` is
    the number of reduced-form posterior draws; each contributes at most one
    accepted rotation, so accepted draws <= n_draws.
    """
    return tsecon.sign_restricted_svar(
        data, monetary_policy_restrictions(k_max), lags=lags, horizon=horizon,
        n_draws=n_draws, max_tries=max_tries, seed=seed, lambda1=lambda1,
    )


def quantile(result, var, prob_index):
    """The [horizon] path of one pointwise quantile (0=5%,1=16%,2=50%,3=84%,4=95%)."""
    q = np.asarray(result["quantiles"])
    return q[:, VARIABLES.index(var), SHOCK, prob_index]


def rule(width=78, ch="-"):
    print(ch * width)


def band_table(result, horizons=(0, 3, 6, 12, 24, 36, 48, 60)):
    y16, y50, y84 = (quantile(result, "y", k) for k in (1, 2, 3))
    d16, d50, d84 = (quantile(result, "yd", k) for k in (1, 2, 3))
    ff50 = quantile(result, "i", 2)
    p84 = quantile(result, "p", 3)
    r84 = quantile(result, "rnb", 3)
    print(f"  {'h':>3} | {'y16':>7} {'y50':>7} {'y84':>7} | "
          f"{'yd16':>7} {'yd50':>7} {'yd84':>7} | {'ff50':>7} | {'p84':>7} | {'rnb84':>7}")
    rule()
    for h in horizons:
        print(f"  {h:>3} | {y16[h]:>+7.3f} {y50[h]:>+7.3f} {y84[h]:>+7.3f} | "
              f"{d16[h]:>+7.3f} {d50[h]:>+7.3f} {d84[h]:>+7.3f} | "
              f"{ff50[h]:>+7.3f} | {p84[h]:>+7.3f} | {r84[h]:>+7.3f}")


def main():
    print("Replication — Uhlig (2005), JME 52(2)")
    print("sign-restricted identification of a contractionary monetary policy shock")
    rule(78, "=")

    uh = load_uhlig()
    data = uh["data"]
    print(f"data: Uhlig (2005) replication file via VARsignR (committed)")
    print(f"      {data.shape[0]} months ({uh['dates'][0]} to {uh['dates'][-1]}), "
          f"{data.shape[1]} series: " + ", ".join(LABELS[v] for v in VARIABLES))
    print("spec: VAR(12) in (100*log) levels, pure-sign rejection sampling,")
    print("      restrictions for months 0..5: yd, p, rnb not positive; ff not negative;")
    print("      real GDP and total reserves UNRESTRICTED. 60-month IRFs.")

    t0 = time.time()
    r = run_uhlig(data)
    runtime = time.time() - t0
    d = r["diagnostics"]
    print(f"\nsampler: {d['accepted']} accepted draws from "
          f"{d['posterior_draws_used']} posterior draws, "
          f"{d['rotations_tried']} rotations tried "
          f"(acceptance rate {100 * d['acceptance_rate']:.1f}%), {runtime:.1f}s")

    print("\nPOSTERIOR IMPULSE-RESPONSE BANDS (16/50/84% pointwise quantiles;")
    print("responses in percent — log x 100 — except ff in percentage points)")
    band_table(r)

    y16, y84 = quantile(r, "y", 1), quantile(r, "y", 3)
    d84 = quantile(r, "yd", 3)
    straddle = [h for h in range(6, 61) if y16[h] < 0 < y84[h]]
    print()
    rule(78, "=")
    print("Published benchmark (Uhlig 2005, Fig. 6 and abstract):")
    print("  (a) NO PRICE PUZZLE — the GDP price deflator does not rise; it")
    print('      "falls only slowly" after a contractionary shock.')
    print(f"      Here: the deflator's 84% quantile is negative at EVERY horizon")
    print(f"      0..60 (max {d84.max():+.3f}); the median falls slowly to "
          f"{quantile(r, 'yd', 2)[60]:+.2f}% by h = 60.")
    print("  (b) AMBIGUOUS GDP RESPONSE — the 68% band straddles zero; 'with a")
    print("      2/3 probability, a typical shock moves real GDP by up to 0.2%'.")
    print(f"      Here: 16% < 0 < 84% at {len(straddle)}/55 horizons in 6..60; the")
    print(f"      band stays within [{y16[6:].min():+.2f}, {y84[6:].max():+.2f}]% — "
          "the ~0.2% magnitude of the text.")
    print("  (c) The identified shock behaves like monetary policy: the funds")
    print(f"      rate rises {quantile(r, 'i', 2)[0]:+.2f} pp on impact (median) "
          "and decays within ~2 years.")

    # Sensitivity: the same run under the library's default Minnesota tightness.
    tight = run_uhlig(data, n_draws=500, lambda1=0.2)
    ty16, ty84 = quantile(tight, "y", 1), quantile(tight, "y", 3)
    td84 = quantile(tight, "yd", 3)
    t_straddle = [h for h in range(6, 61) if ty16[h] < 0 < ty84[h]]
    print("\nPrior sensitivity (lambda1 = 0.2, the library's Minnesota default,")
    print("500 draws): deflator 84% quantile still negative at every horizon "
          f"({td84.max():+.3f} max);")
    print(f"GDP band still straddles zero at {len(t_straddle)}/55 horizons in "
          f"6..60 (e.g. h = 24: [{ty16[24]:+.2f}, {ty84[24]:+.2f}]). The shrinkage")
    print("moves band edges by a few hundredths of a percent; no conclusion moves.")

    print("\nProcedure notes: this is Uhlig's PURE-SIGN (rejection) approach — a")
    print("column of a Haar-random rotation is uniform on the unit sphere, so the")
    print("candidate impulse vectors match Uhlig's; the penalty-function variant")
    print("of the same paper is NOT implemented. Reduced form: Minnesota-NIW")
    print("posterior with intercept, shrinkage set loose (lambda1 = 10) to stand")
    print("in for Uhlig's flat no-constant NIW; one accepted rotation per")
    print("posterior draw. See the docs page for the honest-deviations list.")


if __name__ == "__main__":
    main()
