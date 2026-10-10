import os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
exec(open(os.path.join(HERE, "prelude.py")).read())
from cases import B

only = sys.argv[1:]
names = sorted(n for n in dir(tsecon) if not n.startswith("_") and callable(getattr(tsecon, n)))
covered = {k.split("#")[0] for k in B}
print("callables:", len(names), "covered:", len(covered & set(names)), "missing:", sorted(set(names) - covered), "unknown:", sorted(covered - set(names)))
bad = 0
for key, spec in sorted(B.items()):
    if only and key.split("#")[0] not in only and key not in only:
        continue
    fn = getattr(tsecon, key.split("#")[0])
    code = f"tsecon.{key.split('#')[0]}(" + ", ".join(f"{k}={v}" for k, v in spec["args"].items()) + ")"
    t0 = time.time()
    try:
        r = eval(code)
        st = "OK"
        msg = ""
    except BaseException as e:
        st = "FAIL"
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        bad += 1
    dt = time.time() - t0
    if st != "OK" or dt > 2.0:
        print(f"{st} {key} {dt:.2f}s {msg}")
print("bad", bad)
