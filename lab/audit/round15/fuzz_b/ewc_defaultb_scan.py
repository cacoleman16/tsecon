import math, tsecon, sys, time
def exact(n):
    r = max(0, int(round(0.4 * n ** (2 / 3))) - 3)
    while 125 * (2 * (r + 1) - 1) ** 3 <= 64 * n * n:
        r += 1
    return min(max(r, 1), n - 1)
# every half-integer crossing r - 1/2 for r = 2..R: n* = ceil(sqrt(125 (2r-1)^3 / 64))
R = int(sys.argv[1])
t0 = time.time(); first = None; cnt = 0; checked = 0
for r in range(2, R):
    num = 125 * (2 * r - 1) ** 3
    # smallest n with 64 n^2 >= num
    n = math.isqrt(num // 64)
    while 64 * n * n < num: n += 1
    for nn in (n - 1, n):
        if nn < 2 or nn >= 2**48: continue
        checked += 1
        a = tsecon.ewc_default_b(nn); b = exact(nn)
        if a != b:
            cnt += 1
            if first is None: first = (nn, a, b)
print("r up to", R, "n up to", n, "checked", checked, "mismatches", cnt, "first", first, time.time() - t0)
