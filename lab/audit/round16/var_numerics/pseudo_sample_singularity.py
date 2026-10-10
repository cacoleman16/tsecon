"""How often a residual-bootstrap pseudo-sample has a singular refit covariance.

Source of the numbers in the structural-identification card ("A refit that
cannot be identified ... or a pseudo-sample that cannot be refitted at all").

A VAR(1) with an intercept on k series and T rows has T_eff = T - 1 residual
rows. Replication b resamples T_eff rows iid from them. If the resample
contains d distinct rows, the centred pseudo-residuals have rank at most
d - 1, and the refit residuals M_Z u* (Z holds the intercept) have rank at
most d - 1 too, so the refit's k x k residual covariance is singular whenever
d <= k. The number of length-n sequences over n symbols with exactly j
distinct values is C(n, j) * j! * S(n, j) (S: Stirling numbers of the second
kind), so

    P(d <= k) = sum_{j=1..k} C(n, j) j! S(n, j) / n^n,   n = T_eff,

exactly, and at least one of B draws is such a draw with probability
1 - (1 - P)^B. Pure Python, exact rationals; no tsecon, no randomness.

    python lab/audit/round16/var_numerics/pseudo_sample_singularity.py
"""

from fractions import Fraction
from math import comb, factorial


def stirling2(n: int, j: int) -> int:
    return sum((-1) ** i * comb(j, i) * (j - i) ** n for i in range(j + 1)) // factorial(j)


def p_at_most_k_distinct(n: int, k: int) -> Fraction:
    hits = sum(comb(n, j) * factorial(j) * stirling2(n, j) for j in range(1, k + 1))
    return Fraction(hits, n**n)


def main() -> None:
    k, b = 2, 1000
    print(f"k = {k}, lags = 1, intercept, n_boot = {b}")
    print(" T  T_eff  P(draw has <= k distinct rows)  P(>= 1 of the draws)")
    for t in (6, 7, 8, 9, 10, 12):
        p = p_at_most_k_distinct(t - 1, k)
        at_least_one = 1.0 - (1.0 - float(p)) ** b
        print(f"{t:2d}  {t - 1:5d}  {float(p):30.3e}  {at_least_one:20.3f}")


if __name__ == "__main__":
    main()
