"""REPRO (finding BG-omitted-intercept): specification-tests card quotes the LM form at nlags=1 rejecting
'37.6% and 80.5% with an omitted intercept of 0.5' (T=50, 200; x = 2 + N(0,1) drawn once and held fixed,
lab/serial_corr/no_constant_size.py). That number belongs to one x draw (seed 20264006) whose
||M_x 1||^2/T sits at the 2.6th percentile of designs; across fresh designs the rate is ~0.875.
Runtime ~10 s."""
import numpy as np, tsecon
def cell(seed, n, R):
    rng = np.random.default_rng(seed); x = 2.0 + rng.standard_normal(n); X = x[:, None]; lm = 0
    for _ in range(R): lm += tsecon.breusch_godfrey_test(0.5 + .5 * x + rng.standard_normal(n), X, nlags=1)["pvalue"] < .05
    return lm / R, 1 - x.sum() ** 2 / (n * (x ** 2).sum())
print("repo design (seed 20264006, T=200): rate %.4f  ||M_x 1||^2/T %.4f" % cell(20264006, 200, 2000))
res = np.array([cell(3100 + s, 200, 3000) for s in range(20)])
print(f"20 fresh T=200 designs x 3000 reps: rate mean {res[:,0].mean():.4f} min {res[:,0].min():.4f} max {res[:,0].max():.4f};"
      f" ||M_x 1||^2/T mean {res[:,1].mean():.4f}   (claimed 0.805)")
