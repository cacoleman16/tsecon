"""panel_re refuses an outcome that is constant within every entity, but the message blames
`regressors` ("the random-effects (quasi-demeaned) design is rank-deficient — duplicated or
linearly dependent regressors") although the regressors are fine. The within-fit guard
(re.rs: ssr_w <= WITHIN_EXACT_FIT^2 * tss_w, whose own comment names the "constant within every
entity" case) compares rounding noise with rounding noise: y - ybar_i is ~1e-17, not 0, so the
ratio is ~1 and the guard passes; sigma2_eps is then ~1e-33, theta_i = 1, and the quasi-demeaned
constant column vanishes. With y = 1.0 exactly (ybar exact) the right message appears.

Expected: "outcome: the within residuals vanish ..." in both cases.
"""
import numpy as np, tsecon
rng = np.random.default_rng(7)
N, T = 50, 30
X = rng.standard_normal((2, N, T))
for lab, Y in [("y = 1.0 everywhere", np.ones((N, T))),
               ("y = a random constant per entity", np.repeat(rng.standard_normal(N)[:, None], T, axis=1))]:
    try:
        tsecon.panel_re(Y, X)
        print(lab, "-> accepted")
    except ValueError as e:
        print(f"{lab:35s} -> {str(e)[:150]}")
