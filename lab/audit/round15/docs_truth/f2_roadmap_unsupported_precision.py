"""ROADMAP.md line 35 quotes two precision figures that no committed artifact
supports: the exact DW p-value 'agreeing with a 30-digit reference to 5.3e-16'
(there is no 30-digit/mpmath reference anywhere; the golden is LAPACK+QUADPACK
and the card/CHANGELOG/matrix quote 3.1e-15), and 'an independent Lutkepohl
transcription agrees with it to 2.8e-15' (the fixture records <= 1.04e-15,
which the validation matrix quotes as <= 1.0e-15)."""
import json, pathlib, subprocess, numpy as np, tsecon
W = pathlib.Path(__file__).resolve().parents[4]
road = (W / "ROADMAP.md").read_text()
for frag in ("30-digit reference to 5.3e-16", "agrees with it to 2.8e-15"):
    print("ROADMAP claims:", frag, "->", frag in road)
hits = subprocess.run(["git", "-C", str(W), "grep", "-l", "-i", "-E", "mpmath|30-digit|5\\.3e-16|2\\.8e-15"],
                      capture_output=True, text=True).stdout.split()
print("files mentioning a 30-digit reference / those numbers:", hits)
fx = json.loads((W / "fixtures/serial_corr.json").read_text())
worst = 0.0
for c in fx["cases"]:
    y = np.array(c["y"]); X = np.array(c["columns"]).T
    for alt in ("positive", "negative", "two_sided"):
        p = tsecon.durbin_watson_test(y, X, alternative=alt)["pvalue"]
        worst = max(worst, abs(p - c["durbin_watson"]["pvalue_" + alt]))
print(f"DW exact p vs its (LAPACK+QUADPACK) golden: max abs dev {worst:.2e} (card/CHANGELOG: 3.1e-15)")
vfc = json.loads((W / "fixtures/var_forecast_cov.json").read_text())
print("Omega NumPy transcription vs statsmodels, per case:",
      [f"{c['max_rel_dev']['omega_numpy_vs_statsmodels']:.2e}" for c in vfc["cases"]])
