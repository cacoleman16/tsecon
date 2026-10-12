"""House-wide abort/hang sweep: every int/size-like parameter of every public
callable, values 2**31, 2**40, 2**62, -1, 0, each call in its own subprocess
under ulimit -v 4000000 and timeout 60."""
import ast, json, os, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cases import B

VALUES = [("2**31", 2**31), ("2**40", 2**40), ("2**62", 2**62), ("-1", -1), ("0", 0)]
PRE = open(os.path.join(HERE, "prelude.py")).read()

TEMPLATE = PRE + r'''
import resource
_t0 = time.time()
try:
    _r = {CALL}
    _out = {"status": "ok", "type": type(_r).__name__}
except BaseException as _ex:
    _out = {"status": "exc", "type": type(_ex).__name__, "msg": str(_ex)[:2000]}
_out["t"] = round(time.time() - _t0, 3)
_out["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
print("@@RESULT@@" + json.dumps(_out), flush=True)
'''


def setp(obj, path, val):
    if not path:
        return val
    k = path[0]
    if isinstance(obj, tuple):
        l = list(obj); l[k] = setp(l[k], path[1:], val); return tuple(l)
    if isinstance(obj, list):
        l = list(obj); l[k] = setp(l[k], path[1:], val); return l
    if isinstance(obj, dict):
        d = dict(obj); d[k] = setp(d[k], path[1:], val); return d
    raise TypeError(type(obj))


def gen_cases():
    cases = []
    for key, spec in sorted(B.items()):
        fn = key.split("#")[0]
        for par in spec["sweep"]:
            m = re.match(r"^([A-Za-z_0-9]+)(.*)$", par)
            base, sub = m.group(1), m.group(2)
            path = [ast.literal_eval(s) for s in re.findall(r"\[([^\]]+)\]", sub)]
            for vname, v in VALUES:
                args = dict(spec["args"])
                if path:
                    lit = ast.literal_eval(args[base])
                    args[base] = repr(setp(lit, path, v))
                else:
                    args[base] = vname
                call = f"tsecon.{fn}(" + ", ".join(f"{k}={a}" for k, a in args.items()) + ")"
                cases.append({"id": f"{key}:{par}={vname}", "fn": fn, "variant": key, "param": par,
                              "base": base, "value": vname, "call": call})
    return cases


def run_case(case, timeout=60, mem_kb=4000000):
    code = TEMPLATE.replace("{CALL}", case["call"])
    os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.path.join(HERE, "tmp")) as f:
        f.write(code)
        path = f.name
    t0 = time.time()
    p = subprocess.run(["bash", "-c", f"ulimit -v {mem_kb}; timeout {timeout} {PY} {path}"],
                       capture_output=True, text=True, cwd=os.path.join(HERE, "tmp"))
    wall = time.time() - t0
    os.unlink(path)
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            res = json.loads(line[len("@@RESULT@@"):])
    if res is None:
        res = {"status": "TIMEOUT" if p.returncode == 124 else "CRASH", "rc": p.returncode,
               "stderr": p.stderr[-1500:]}
    res["wall"] = round(wall, 2)
    out = dict(case)
    out.update(res)
    return out


def classify(r):
    st = r["status"]
    if st == "TIMEOUT":
        return "timeout"
    if st == "CRASH":
        return "abort"
    if st == "ok":
        return "ok"
    if r["type"] == "MemoryError":
        return "memoryerror"
    if r["type"] in ("PanicException", "SystemError"):
        return "panic"
    msg = r.get("msg", "")
    names = {r["base"]}
    for k in re.findall(r"\['([A-Za-z_]+)'\]", r["param"]):
        names.add(k)
    if any(n in msg for n in names):
        return "named"
    return "unnamed"


if __name__ == "__main__":
    out_path = sys.argv[1]
    sel = sys.argv[2:]  # optional variant filters
    cases = gen_cases()
    if sel:
        cases = [c for c in cases if c["variant"] in sel or c["fn"] in sel]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(json.loads(line)["id"])
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} cases, {len(todo)} to run", flush=True)
    with open(out_path, "a") as f, ThreadPoolExecutor(int(os.environ.get("W", "3"))) as ex:
        for r in ex.map(run_case, todo):
            r["class"] = classify(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if r["class"] not in ("ok", "named"):
                print(r["class"], r["id"], r.get("wall"), (r.get("msg") or r.get("stderr") or "")[:200].replace("\n", " | "), flush=True)
