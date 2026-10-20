"""REPRO (finding UC-cycle-recovery): unobserved-components card quotes the stationary-cycle recovery
'8 seeds, T = 200: median |period error| 0.637, median |damping error| 0.0117' (uc_properties.rs).
Same DGP (local level + damped stochastic cycle, period 12, damping 0.9, sigma2 cycle .5, level .01,
irregular .2, cycle drawn from its stationary distribution), cycle_init="stationary", bounds (4, 40),
fresh seeds, 120 replications. Prints the medians with bootstrap SEs and the convergence/escape counts.
Runtime ~4-5 min on a loaded 4-core machine."""
import sys, math, time, numpy as np, tsecon
R = int(sys.argv[1]) if len(sys.argv) > 1 else 120
rng = np.random.default_rng(24680); n = 200
period, rho, s2c, s2l, s2e = 12.0, 0.9, 0.5, 0.01, 0.2
lam = 2 * math.pi / period; sd0 = math.sqrt(s2c / (1 - rho * rho))
ep, ed, conv = [], [], []
t0 = time.time()
for r in range(R):
    c, cs = sd0 * rng.standard_normal(), sd0 * rng.standard_normal(); lvl = 0.0; y = np.empty(n)
    for t in range(n):
        y[t] = lvl + c + math.sqrt(s2e) * rng.standard_normal()
        c, cs = (rho * (math.cos(lam) * c + math.sin(lam) * cs) + math.sqrt(s2c) * rng.standard_normal(),
                 rho * (-math.sin(lam) * c + math.cos(lam) * cs) + math.sqrt(s2c) * rng.standard_normal())
        lvl += math.sqrt(s2l) * rng.standard_normal()
    f = tsecon.unobserved_components(y, level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True,
                                     cycle_period_bounds=(4.0, 40.0), cycle_init="stationary")
    nm = f["param_names"]
    ep.append(2 * math.pi / f["params"][nm.index("frequency.cycle")] - period); ed.append(f["params"][nm.index("damping.cycle")] - rho); conv.append(f["converged"])
ep, ed = np.abs(ep), np.abs(ed)
bs = lambda a: np.std([np.median(rng.choice(a, a.size)) for _ in range(2000)])
print(f"{R} fresh seeds: median |period-12| {np.median(ep):.3f} (boot se {bs(ep):.3f}; claimed 0.637 on 8 seeds)")
print(f"            median |damping-0.9| {np.median(ed):.4f} (boot se {bs(ed):.4f}; claimed 0.0117 on 8 seeds)")
print(f"  P(8-seed median <= 0.0117) by resampling: {np.mean([np.median(rng.choice(ed, 8)) <= 0.0117 for _ in range(20000)]):.3f}")
print(f"  non-converged {int(np.sum(~np.array(conv)))}; |period err|>=3: {int(np.sum(ep>=3))}; |damping err|>=0.15: {int(np.sum(ed>=0.15))}; elapsed {time.time()-t0:.0f}s")
