"""Parent side of the cell protocol: run a list of mutations for one callable
through `cell_runner.py`, restarting the child after a death or a hang."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))


def _reader(proc, q):
    for line in proc.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)


def run_cells(name, muts, rlimit_gb=4.0, deadline=30.0, env_extra=None):
    """muts: list of dicts with an "id" key. Returns one record per mutation."""
    pending = {m["id"]: m for m in muts}
    order = [m["id"] for m in muts]
    records = {}
    env = dict(os.environ, RAYON_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    if env_extra:
        env.update(env_extra)
    while pending:
        err_path = os.path.join(HERE, "out", f".{name}.stderr")
        with open(err_path, "w") as err_fh:
            proc = subprocess.Popen(
                [sys.executable, os.path.join(HERE, "cell_runner.py"), name, str(rlimit_gb)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_fh, text=True, env=env, cwd=HERE,
            )
            q = queue.Queue()
            threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
            todo = [i for i in order if i in pending]
            for mid in todo:
                proc.stdin.write(json.dumps(pending[mid]) + "\n")
                proc.stdin.flush()
                dl = pending[mid].get("deadline", deadline)
                try:
                    line = q.get(timeout=dl)
                except queue.Empty:
                    proc.kill(); proc.wait()
                    records[mid] = {"id": mid, "outcome": "HANG", "deadline": dl}
                    del pending[mid]
                    break
                if line is None:
                    proc.wait()
                    tail = open(err_path).read()[-800:]
                    outcome = "CRASH"
                    detail = f"rc={proc.returncode}"
                    if "memory allocation of" in tail:
                        outcome = "ALLOC-ABORT"
                        nums = [w for w in tail.replace("\n", " ").split() if w.isdigit()]
                        detail += f" bytes={nums[-1] if nums else '?'}"
                    elif "capacity overflow" in tail:
                        outcome = "CRASH-CAPACITY-OVERFLOW"
                    records[mid] = {"id": mid, "outcome": outcome, "detail": detail, "stderr_tail": tail[-300:]}
                    del pending[mid]
                    break
                assert line.startswith("DONE "), line
                rec = json.loads(line[5:])
                records[mid] = rec
                del pending[mid]
            else:
                try:
                    proc.stdin.close()
                    proc.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    proc.kill()
        try:
            os.remove(err_path)
        except OSError:
            pass
    return [records[i] for i in order]
