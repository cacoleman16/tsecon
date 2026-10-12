"""Fork-server variant of harness.py: the parent imports tsecon and builds the
fixtures once (no tsecon function is called in the parent), then every case
runs in a freshly forked child with RLIMIT_AS = 4000000 KiB (== ulimit -v
4000000) and a 60 s wall-clock kill. Same case list, same classification."""
import json, os, resource, select, signal, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import gen_cases, classify

exec(open(os.path.join(HERE, "prelude.py")).read())

MEM = 4000000 * 1024
TIMEOUT = float(os.environ.get("TO", "60"))
W = int(os.environ.get("W", "3"))


def child(case, wfd):
    resource.setrlimit(resource.RLIMIT_AS, (MEM, MEM))
    t0 = time.time()
    try:
        r = eval(case["call"])
        out = {"status": "ok", "type": type(r).__name__}
    except BaseException as ex:  # noqa
        out = {"status": "exc", "type": type(ex).__name__, "msg": str(ex)[:2000]}
    out["t"] = round(time.time() - t0, 3)
    out["maxrss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    data = json.dumps(out).encode()
    os.write(wfd, data)
    os.close(wfd)
    os._exit(0)


def main():
    out_path = sys.argv[1]
    sel = sys.argv[2:]
    cases = gen_cases()
    if sel:
        cases = [c for c in cases if c["variant"] in sel or c["fn"] in sel or c["id"] in sel]
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(json.loads(line)["id"])
    todo = [c for c in cases if c["id"] not in done]
    print(f"{len(cases)} cases, {len(todo)} to run", flush=True)
    running = {}  # pid -> (case, rfd, t0, buf)
    f = open(out_path, "a")
    idx = 0
    while idx < len(todo) or running:
        while idx < len(todo) and len(running) < W:
            case = todo[idx]; idx += 1
            rfd, wfd = os.pipe()
            sys.stdout.flush(); sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                os.close(rfd)
                # child stderr -> /dev/null except keep last bytes via a file
                errf = os.open(os.path.join(HERE, "tmp", f"err_{os.getpid()}.txt"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
                os.dup2(errf, 2)
                try:
                    child(case, wfd)
                finally:
                    os._exit(3)
            os.close(wfd)
            running[pid] = [case, rfd, time.time(), b""]
        # drain pipes
        fds = {v[1]: pid for pid, v in running.items()}
        if fds:
            r, _, _ = select.select(list(fds), [], [], 0.05)
            for fd in r:
                pid = fds[fd]
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    chunk = b""
                running[pid][3] += chunk
        for pid in list(running):
            case, rfd, t0, buf = running[pid]
            wp, status = os.waitpid(pid, os.WNOHANG)
            if wp == 0:
                if time.time() - t0 > TIMEOUT:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    res = {"status": "TIMEOUT", "rc": 124, "stderr": ""}
                    finish(f, case, res, t0, rfd, pid)
                    del running[pid]
                continue
            # exited: read remaining
            while True:
                try:
                    chunk = os.read(rfd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    break
                buf += chunk
            if buf:
                res = json.loads(buf.decode())
            else:
                errtxt = ""
                try:
                    errtxt = open(os.path.join(HERE, "tmp", f"err_{pid}.txt")).read()[-1500:]
                except OSError:
                    pass
                if os.WIFSIGNALED(status):
                    rc = -os.WTERMSIG(status)
                else:
                    rc = os.WEXITSTATUS(status)
                res = {"status": "CRASH", "rc": rc, "stderr": errtxt}
            finish(f, case, res, t0, rfd, pid)
            del running[pid]
    f.close()


def finish(f, case, res, t0, rfd, pid):
    try:
        os.close(rfd)
    except OSError:
        pass
    try:
        os.unlink(os.path.join(HERE, "tmp", f"err_{pid}.txt"))
    except OSError:
        pass
    res["wall"] = round(time.time() - t0, 2)
    out = dict(case); out.update(res)
    out["class"] = classify(out)
    f.write(json.dumps(out) + "\n"); f.flush()
    if out["class"] not in ("ok", "named"):
        print(out["class"], out["id"], out.get("wall"), (out.get("msg") or out.get("stderr") or "")[:200].replace("\n", " | "), flush=True)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "tmp"), exist_ok=True)
    main()
