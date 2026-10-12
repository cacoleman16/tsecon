"""REPRO (finding EWC-precision): robust-inference card / CHANGELOG coverage table (T=400: hac 0.905,
ewc 0.940; 'MC se 0.002') and the ewc_lrv example ('size at 5% EWC + t_B: 0.066'). The quoted MC se is
sqrt(.95*.05/R) at the NOMINAL rate (lab/audit/wave11/ewc_varcov_measurements.py), not at the measured
coverage. Fresh seeds; runtime ~40 s."""
import math, numpy as np, tsecon
from scipy import stats, signal
def ar1(rng, n, rho, burn=200): return signal.lfilter([1.0], [1.0, -rho], rng.standard_normal(n + burn))[burn:]
z = stats.norm.ppf(0.975)
for T, R, cl_h, cl_e in ((100, 10000, 0.855, 0.915), (400, 60000, 0.905, 0.940)):
    rng = np.random.default_rng([97531, T]); hh = he = 0
    for _ in range(R):
        x = ar1(rng, T, .7); y = 1.0 + ar1(rng, T, .7); X = np.column_stack([np.ones(T), x])
        h = tsecon.ols(y, X, se_type="hac"); e = tsecon.ols(y, X, se_type="ewc"); tb = stats.t.ppf(.975, e["ewc_b"])
        hh += abs(h["params"][1]) <= z * h["bse"][1]; he += abs(e["params"][1]) <= tb * e["bse"][1]
    ch, ce = hh / R, he / R; sh, s_e = math.sqrt(ch * (1 - ch) / R), math.sqrt(ce * (1 - ce) / R)
    print(f"T={T} R={R}: hac claimed {cl_h} measured {ch:.4f} (se {sh:.4f}, z {(cl_h-ch)/sh:+.1f});  ewc claimed {cl_e} measured {ce:.4f} (se {s_e:.4f}, z {(cl_e-ce)/s_e:+.1f})")
    print(f"   binomial se at the CLAIMED coverages with the card's 10000 reps: hac {math.sqrt(cl_h*(1-cl_h)/1e4):.4f}, ewc {math.sqrt(cl_e*(1-cl_e)/1e4):.4f}  (card column says 0.002)")
R, n = 40000, 200; rng = np.random.default_rng(2468)
e = rng.standard_normal((R, n)); e[:, 0] /= math.sqrt(0.75); y = signal.lfilter([1.0], [1.0, -0.5], e, axis=1)
b = tsecon.ewc_default_b(n); crit = stats.t.ppf(.975, b)
te = y.mean(1) / np.sqrt(np.array([tsecon.ewc_lrv(r, b=b) for r in y]) / n)
pe = np.mean(np.abs(te) > crit)
print(f"ewc_lrv example (AR(.5), n=200, B={b}): claimed size 0.066 (1000 reps)  measured {pe:.4f} (se {math.sqrt(pe*(1-pe)/R):.4f}, R={R})")
