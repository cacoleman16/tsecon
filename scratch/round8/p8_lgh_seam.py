"""Round-8 probe: ln_gamma_half_ratio seam accuracy, against a 50-digit
Decimal Stirling reference.

Transcribes the Rust arithmetic (Lanczos g=7/9-term ln_gamma; the asymptotic
series branch) into Python doubles — deterministic IEEE arithmetic, so this
mirrors the shipped code up to libm ulp differences — and measures:
  1. literal-branch error just below the seam (doc claims ~1e-10 abs there);
  2. series-branch truncation just above (doc claims O(1e-18), sub-ulp);
  3. the seam jump (value discontinuity crossing x = 1e3);
  4. monotonicity across the seam in doubles;
  5. the doc's headline: literal difference at x = 5e15 carries O(1e3) error.
"""
from decimal import Decimal, getcontext
import math

getcontext().prec = 60

# ---- Rust transcription (doubles) ----
LANCZOS_G = 7.0
LANCZOS_COEF = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
]
LN_2PI = 1.8378770664093454835606594728112353


def ln_gamma(x):
    if x == 1.0 or x == 2.0:
        return 0.0
    w = x - 1.0
    a = LANCZOS_COEF[0]
    for k in range(1, 9):
        a += LANCZOS_COEF[k] / (w + k)
    t = w + LANCZOS_G + 0.5
    return 0.5 * LN_2PI + (w + 0.5) * math.log(t) - t + math.log(a)


def half_ratio(x):
    if x < 1e3:
        return ln_gamma(x + 0.5) - ln_gamma(x)
    inv = 1.0 / x
    return 0.5 * math.log(x) - 0.125 * inv * (1.0 - inv * inv / 24.0)


# ---- high-precision reference: Stirling with Bernoulli terms ----
# B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66, B12=-691/2730
BERN = [Decimal(1) / 6, Decimal(-1) / 30, Decimal(1) / 42, Decimal(-1) / 30,
        Decimal(5) / 66, Decimal(-691) / 2730, Decimal(7) / 6]
PI_D = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


def ln_gamma_hp(xd):
    """ln Gamma via Stirling for large x (x >= 100 is plenty at prec 60)."""
    x = Decimal(xd)
    half = Decimal("0.5")
    ln2pi = (2 * PI_D).ln()
    s = (x - half) * x.ln() - x + half * ln2pi
    for n, b in enumerate(BERN, start=1):
        s += b / (2 * n * (2 * n - 1) * x ** (2 * n - 1))
    return s


def true_ratio(x):
    return ln_gamma_hp(Decimal(x) + Decimal("0.5")) - ln_gamma_hp(Decimal(x))


print("x, branch, double_value, true, abs_err")
rows = []
for x in [990.0, 999.0, 999.5, math.nextafter(1e3, 0.0), 1e3, 1000.5, 1001.0, 1010.0,
          1e4, 1e6, 1e9, 1e12, 5e15]:
    v = half_ratio(x)
    tr = true_ratio(x)
    err = abs(Decimal(v) - tr)
    branch = "literal" if x < 1e3 else "series "
    rows.append((x, branch, v, float(err)))
    print(f"{x:>22.6f} {branch} {v:.15f} err={float(err):.3e}")

# 1. literal error just below the seam
err_below = max(e for x, b, v, e in rows if b == "literal" and x >= 990)
print(f"\nmax literal-branch abs error near seam: {err_below:.3e} (doc claims ~1e-10)")
assert_ok1 = err_below < 5e-10

# 2. series truncation just above
err_above = max(e for x, b, v, e in rows if b == "series " and x <= 1010)
print(f"max series-branch abs error near seam: {err_above:.3e} (doc claims sub-ulp ~1e-18; "
      f"double rounding floor ~1e-16)")
assert_ok2 = err_above < 5e-15

# 3. seam jump
just_below = half_ratio(math.nextafter(1e3, 0.0))
at_seam = half_ratio(1e3)
jump = at_seam - just_below
true_step = float(true_ratio(1e3) - true_ratio(math.nextafter(1e3, 0.0)))
print(f"\nseam jump (double): {jump:.3e}; true infinitesimal step: {true_step:.3e}; "
      f"spurious part ~{abs(jump - true_step):.3e}")

# 4. monotonicity across the seam on a fine grid of doubles
xs = [990 + 0.25 * k for k in range(81)]  # 990..1010
vals = [half_ratio(x) for x in xs]
mono = all(b > a for a, b in zip(vals, vals[1:]))
print(f"monotone increasing across seam on 0.25-grid: {mono}")

# check a denser grid right at the seam (spacing ~1 ulp of 1e3)
import itertools
x = 999.9999999
fine = []
for _ in range(50):
    fine.append(x)
    x = math.nextafter(x, 2000.0) + 1e-8
fine += [1000.0 + 1e-8 * k for k in range(50)]
fv = [half_ratio(t) for t in fine]
viol = sum(1 for a, b in zip(fv, fv[1:]) if b < a)
print(f"local monotonicity violations in the 1e-8-step corridor: {viol} "
      f"(the literal branch's own noise makes tiny non-monotonicity possible below the seam)")

# 5. the headline: literal difference at 5e15 is garbage, helper is exact
x = 5e15
lit = ln_gamma(x + 0.5) - ln_gamma(x)
hp = float(true_ratio(x))
print(f"\nat x=5e15: literal={lit:.6f}, helper={half_ratio(x):.15f}, true={hp:.15f}, "
      f"literal error={abs(lit-hp):.3e}, helper error={abs(half_ratio(x)-hp):.3e}")

print("\nVERDICTS:")
print(f"  literal-branch near-seam error < 5e-10: {assert_ok1}")
print(f"  series-branch near-seam error < 5e-15: {assert_ok2}")
print(f"  helper at 5e15 error < 1e-12 relative: {abs(half_ratio(x)-hp)/hp < 1e-12}")
