"""validation-matrix.md line ~292-294 (added by this wave): 'ewc.json and
var_forecast_cov.json come from a fourth venv generation ... and are the first
fixtures to record statsmodels 0.15.0.' Eight fixtures committed BEFORE the
wave (e38965e) already record statsmodels 0.15.0 in their _meta, and the same
page (line ~237) separately says variance_ratio.json 'opens' the fourth
generation; serial_corr, panel_static and uc_extensions record the same
NumPy 2.5.3 / SciPy 1.18.1 / statsmodels 0.15.0 / Python 3.13.16 triple too."""
import json, subprocess
W = str(__import__("pathlib").Path(__file__).resolve().parents[4])
def meta(rev, path):
    try:
        d = json.loads(subprocess.run(["git", "-C", W, "show", f"{rev}:{path}"], capture_output=True, text=True, check=True).stdout)
    except Exception:
        return None
    m = d.get("_meta") or d.get("meta") or {}
    return m if isinstance(m, dict) else {}
files = subprocess.run(["git", "-C", W, "ls-tree", "--name-only", "e38965e", "fixtures/"], capture_output=True, text=True).stdout.split()
pre = [f for f in files if f.endswith(".json") and (meta("e38965e", f) or {}).get("statsmodels") == "0.15.0"]
print("pre-wave fixtures already recording statsmodels 0.15.0:", pre)
for f in ("variance_ratio", "serial_corr", "panel_static", "uc_extensions", "ewc", "var_forecast_cov"):
    m = meta("HEAD", f"fixtures/{f}.json")
    print(f, {k: m.get(k) for k in ("python", "numpy", "scipy", "statsmodels")})
for i, l in enumerate(open(f"{W}/docs/reference/validation-matrix.md"), 1):
    if "first fixtures to record" in l or "opens a fourth generation" in l:
        print(f"matrix:{i}: {l.strip()}")
