"""historical_decomposition(identification="impact") on an invertible impact
matrix at an extreme scale: A0*1e300 is refused as 'singular by the
numpy.linalg.matrix_rank convention' although numpy.linalg.matrix_rank gives
full rank (the refusal's own stated rule does not hold); A0*1e-200 is accepted
and shock_covariance comes back inf. structural_fevd(impact=A0*1e300) returns
NaN shares silently, and A0*1e-200 is refused as a 'degenerate impact matrix'
(underflow, not degeneracy)."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
A0 = np.array(tsecon.long_run_svar(X)["impact"])
for s in [1e300, 1e-200]:
    A = A0 * s
    print(f"scale {s:.0e}: numpy.linalg.matrix_rank = {np.linalg.matrix_rank(A)}")
    try:
        r = tsecon.historical_decomposition(X, identification="impact", impact=A)
        print("   historical_decomposition: accepted; shock_covariance =", r["shock_covariance"])
    except ValueError as e:
        print("   historical_decomposition: refused:", str(e)[:150])
    try:
        r = tsecon.structural_fevd(X, impact=A)
        print("   structural_fevd: accepted; fevd[1] =", r["fevd"][1])
    except ValueError as e:
        print("   structural_fevd: refused:", str(e)[:150])
