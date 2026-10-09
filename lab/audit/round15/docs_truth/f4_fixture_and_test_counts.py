"""testing.md (Tier 4) and validation-matrix.md (intro + 'Python layer' paragraph)
were bumped by this wave to '2379 tests in 116 files' / '68 of the 101 fixture
JSONs'. Recounted with the page's own stated method on the integrated tree:
84 of 117 fixture JSONs are reloaded from Python (the 33-name 'never opened'
list on the matrix is exactly right, so 101 - 68 = 33 only matches by
coincidence), 122 test files, and pytest collects ~2922 tests. The matrix also
says 'The counts above describe the suite before the 0.11 wave', but the
pre-wave tree (e38965e) counts 77 of 110."""
import glob, os, re, subprocess, sys, pathlib, tempfile
W = str(__import__("pathlib").Path(__file__).resolve().parents[4])
PY = sys.executable

def count(root):
    fx = {os.path.basename(p) for p in glob.glob(f"{root}/fixtures/*.json")}
    names = set()
    for f in glob.glob(f"{root}/bindings/python/tests/*.py"):
        names |= {os.path.basename(m) for m in re.findall(r"[\"']([A-Za-z0-9_./-]+\.json)[\"']", open(f).read())}
    files = len(glob.glob(f"{root}/bindings/python/tests/test_*.py"))
    return len(names & fx), len(fx), sorted(fx - names), files

r, t, never, files = count(W)
print(f"integrated tree: {r} of {t} fixture JSONs reloaded; {len(never)} never opened; {files} test files")
for doc in ("docs/reference/testing.md", "docs/reference/validation-matrix.md"):
    for i, l in enumerate(open(f"{W}/{doc}"), 1):
        if re.search(r"of the 10[01]|tests in 11[56] files", l):
            print(f"  {doc}:{i}: ...{re.search(r'.{0,40}(of the 10[01]|tests in 11[56] files).{0,30}', l).group(0)}...")
with tempfile.TemporaryDirectory() as d:
    subprocess.run(f"git -C {W} archive e38965e fixtures bindings/python/tests | tar -x -C {d}", shell=True, check=True)
    r0, t0, _, f0 = count(d)
    print(f"pre-wave tree e38965e: {r0} of {t0} reloaded, {f0} test files (docs said 67 of 100, 115 files)")
out = subprocess.run([PY, "-m", "pytest", "-p", "no:cacheprovider", "--collect-only", "-q", f"{W}/bindings/python/tests"],
                     capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}).stdout
print("pytest:", out.strip().splitlines()[-1])
