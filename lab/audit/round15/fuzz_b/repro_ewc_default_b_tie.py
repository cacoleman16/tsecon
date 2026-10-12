"""ewc_default_b(n) is documented as round(0.4 * n^(2/3)) (clamped to [1, n-1]).
Near a half-integer the float evaluation n**(2/3) misrounds for large n, so the
returned B is off by one. Exact check in integer arithmetic:
    round(0.4 n^(2/3)) >= r  <=>  0.4 n^(2/3) >= r - 1/2  <=>  64 n^2 >= 125 (2r - 1)^3
"""
from decimal import Decimal, getcontext
import tsecon

getcontext().prec = 60
for n in (1755027219, 100292764931117):
    got = tsecon.ewc_default_b(n)
    v = Decimal(4) / Decimal(10) * (Decimal(n) ** 2).ln().__truediv__(3).exp()   # 0.4 n^(2/3), 60 digits
    exact = int((v + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR"))
    ok_int = 64 * n * n >= 125 * (2 * exact - 1) ** 3      # exact integer certificate that B >= exact
    print(f"n={n}: 0.4 n^(2/3) = {v:.12f}; expected round = {exact} (integer certificate {ok_int}); "
          f"tsecon.ewc_default_b = {got}")
