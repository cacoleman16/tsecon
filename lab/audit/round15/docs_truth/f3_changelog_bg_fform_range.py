"""CHANGELOG [Unreleased] BG bullet: 'the LM form ... rejected 12.0-12.7% ... with
nlags=1, iid errors and a regressor of mean 2, while the F-form held 4.5-5.1% in
the same cells.' In the committed output (and on re-running the committed lab
script) the same cells (x mean 2, intercept 0, nlags=1) give F = 0.046 / 0.051,
i.e. 4.6-5.1%; 4.5% is the omitted-intercept (0.5) cell, where LM is 37.6%."""
import pathlib, re, runpy, sys, types
W = pathlib.Path(__file__).resolve().parents[4]
cl = (W / "CHANGELOG.md").read_text()
m = re.search(r"rejected 12\.0.12\.7%[^.]*?F-form held ([0-9.]+).([0-9.]+)%", cl)
print("CHANGELOG quotes F-form range:", m.group(1), "-", m.group(2), "%")
# re-run the committed lab script with its output redirected out of the tree
src = (W / "lab/serial_corr/no_constant_size.py").read_text()
out = pathlib.Path(__file__).with_name("no_constant_size_rerun.txt")
g = {"__name__": "__main__", "__file__": str(W / "lab/serial_corr/no_constant_size.py")}
code = compile(src, "no_constant_size.py", "exec")
import builtins
orig_write = pathlib.Path.write_text
def redirect(self, text, *a, **k):
    return orig_write(out, text, *a, **k)
pathlib.Path.write_text = redirect
try:
    exec(code, g)
finally:
    pathlib.Path.write_text = orig_write
rows = [l.split() for l in out.read_text().splitlines() if re.match(r"\s+\d+\s+x mean", l)]
same = [(r[0], r[5]) for r in rows if r[3] == "2" and r[4] == "0.0"]
print("re-run, x mean 2, intercept 0.0: (T, LM p=1, F p=1) =",
      [(r[0], r[5], r[6]) for r in rows if r[3] == "2" and r[4] == "0.0"])
