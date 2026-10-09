"""How far the Blanchard-Quah page's bootstrap band table moved when it switched
from the hand-built bootstrap (0.10.0: replication b seeded from
SeedSequence(seed).generate_state(n)[b]) to the native long_run_svar_bands
(SeedSequence(seed).spawn(n)[b]). Parses both versions of the table.

    python lab/bq_bands/page_table_move.py > lab/bq_bands/out/page_table_move.txt
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGE = "docs/examples/replication-blanchard-quah.md"
OLD_REV = "574ceaf"  # the 0.10.0 release commit


def table(text):
    i = text.index("| response | h | point | 68% band | 90% band |")
    rows = text[i:].split("\n\n")[0].split("\n")[2:]
    out = {}
    for r in rows:
        c = [x.strip() for x in r.strip("|").split("|")]
        out[(c[0], c[1])] = [float(x.replace("−", "-"))
                             for x in re.findall(r"[−+\-]?\d+\.\d+", " ".join(c[3:]))]
    return out


old = table(subprocess.run(["git", "show", f"{OLD_REV}:{PAGE}"], cwd=REPO, capture_output=True,
                           text=True, check=True).stdout)
new = table((REPO / PAGE).read_text(encoding="utf-8"))
assert old.keys() == new.keys()
moves = {k: max(abs(a - b) for a, b in zip(old[k], new[k])) for k in old}
worst = max(moves, key=moves.get)
print(f"{len(old)} rows, 4 bounds each; largest move {moves[worst]:.3f} at {worst}")
