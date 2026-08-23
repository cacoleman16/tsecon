"""Refute the BNS 'low power' candidate: compute the predicted z for the
probe's jump size, then re-run with a jump that should be detected."""
import numpy as np
import tsecon

n = 390
sig = 0.01
iv = n * sig ** 2
for jump in [0.06, 0.12, 0.2]:
    rj_frac = jump ** 2 / (iv + jump ** 2)
    theta = np.pi ** 2 / 4 + np.pi - 5
    z_pred = np.sqrt(n) * rj_frac / np.sqrt(theta)
    hits = 0
    reps = 300
    for k in range(reps):
        rr = np.random.default_rng(70_000 + k)
        r = sig * rr.standard_normal(n)
        r[200] += jump
        if tsecon.bns_jump_test(r)["ratio"] > 1.645:
            hits += 1
    print(f"jump {jump}: relative-jump {rj_frac:.3f}, predicted z ~ {z_pred:.2f}, "
          f"measured power {hits/reps:.3f}")
