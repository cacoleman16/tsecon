"""Golden fixtures for tsecon-dsge: linear rational-expectations (Blanchard-Kahn).

VALIDATION TARGET (DOCUMENTED CLOSED-FORM FORMULA, target (b) of the crate spec).
================================================================================
The Blanchard-Kahn (1980) linear RE solver is validated against a DOCUMENTED
CLOSED-FORM SOLUTION: two small textbook forward-looking models whose policy
matrices G, P, Q and eigenvalues are known analytically, written out in full
below. The analytic values in this fixture are typed straight from those
formulas -- this generator NEVER calls the tsecon Rust solver, and never calls
any external RE solver either, so the match is non-circular. numpy is used ONLY
to (a) assemble the model matrices and (b) INDEPENDENTLY cross-check the
eigenvalues via `numpy.linalg.eigvals`, which is a wholly separate code path
from faer's real-Schur eigensolver in the Rust crate.

THE SOLVER'S MODEL FORM
=======================
A linearized model is written in the first-order expectational form

    A . E_t[y_{t+1}] = B . y_t + C . z_{t+1}

where y_t = [k_t ; x_t] stacks the n_k PREDETERMINED (backward-looking) block
k_t on top of the n_x NON-PREDETERMINED (jump / forward-looking) block x_t, and
z_{t+1} is a mean-zero exogenous innovation with E_t[z_{t+1}] = 0. Writing
M = A^{-1} B and N = A^{-1} C, the reduced form is E_t[y_{t+1}] = M y_t + N z.

Because E_t[z_{t+1}] = 0 the innovation drops out of the forward-looking solve,
which therefore works on the homogeneous system E_t[y_{t+1}] = M y_t. Eigen-
decompose M = V L V^{-1} with L = diag(eigenvalues) ordered so the STABLE
eigenvalues (|lambda| < 1) come first and the UNSTABLE ones (|lambda| > 1)
last. Let W = V^{-1} and partition its unstable rows W_2 = [W21 | W22] into the
predetermined columns W21 (n_u x n_k) and the jump columns W22 (n_u x n_x).

BLANCHARD-KAHN CONDITION: a unique non-explosive solution exists iff the number
of unstable eigenvalues n_u equals the number of jump variables n_x. Then:

    policy rule   x_t     = G k_t,        G = -W22^{-1} W21          (n_x x n_k)
    law of motion k_{t+1} = P k_t + Q z,  P = M11 + M12 G  (n_k x n_k)
                                          Q = N_k  (predetermined rows of N)

where M11, M12 are the predetermined-row blocks of M. If n_u < n_x the solution
is INDETERMINATE (a continuum of stable solutions); if n_u > n_x there is NO
stable solution. The stable eigenvalues of M are exactly the eigenvalues of P,
so P is always stable when Blanchard-Kahn holds.

--------------------------------------------------------------------------------
MODEL 1 -- Cagan / asset-price (scalar jump, scalar predetermined shock)
--------------------------------------------------------------------------------
The canonical forward-looking asset-price / Cagan money-demand equation

    p_t = a . E_t[p_{t+1}] + u_t,     0 < a < 1,

driven by an exogenous AR(1) fundamental u_t = rho . u_{t-1} + eps_t, |rho| < 1.
The textbook FUNDAMENTAL (no-bubble) solution solves p forward:

    p_t = sum_{j>=0} a^j E_t[u_{t+j}] = sum_{j>=0} a^j rho^j u_t = u_t / (1 - a rho).

Cast into the solver's form with y = [u ; p] (u predetermined, p jump). Taking
E_t of the AR(1) gives E_t u_{t+1} = rho u_t; rearranging the pricing equation
gives E_t p_{t+1} = (1/a) p_t - (1/a) u_t. Hence

    M = [[ rho ,   0  ],        C = [[ sigma ],
         [ -1/a ,  1/a ]]             [   0   ]]

(the innovation eps loads on the predetermined u row only; A = I so N = C).
Eigenvalues of M are rho (stable) and 1/a > 1 (unstable): n_u = 1 = n_x, so
Blanchard-Kahn holds. The unstable left-eigen row gives, analytically,

    G = 1 / (1 - a rho)  (so p_t = u_t / (1 - a rho), matching the sum above),
    P = rho,             (u_{t+1} = rho u_t + eps_{t+1}),
    Q = sigma.

--------------------------------------------------------------------------------
MODEL 2 -- two exogenous shocks, one jump (matrix P and multi-column G)
--------------------------------------------------------------------------------
A jump p driven by two independent AR(1) fundamentals u1 (rho1), u2 (rho2):

    p_t = a . E_t[p_{t+1}] + u1_t + u2_t.

By linearity of the same forward sum, p_t = u1_t/(1 - a rho1) + u2_t/(1 - a rho2).
With y = [u1 ; u2 ; p] (u1, u2 predetermined; p jump),

    M = [[ rho1,   0  ,  0  ],     C = [[ sigma1,   0   ],
         [  0  ,  rho2,  0  ],           [   0   , sigma2],
         [ -1/a, -1/a ,  1/a]]           [   0   ,   0   ]]

Eigenvalues rho1, rho2 (stable) and 1/a > 1 (unstable): n_u = 1 = n_x. Then

    G = [ 1/(1 - a rho1) ,  1/(1 - a rho2) ]   (1 x 2),
    P = diag(rho1, rho2),
    Q = diag(sigma1, sigma2).

--------------------------------------------------------------------------------
MODEL 3 -- Blanchard-Kahn FAILURE cases (property fixture, no G/P/Q)
--------------------------------------------------------------------------------
Two mis-specified variants of Model 1, used to check the solver's verdict:
  * too FEW jumps  (both variables declared predetermined, n_x = 0 < n_u = 1)
    -> NO stable solution;
  * too MANY jumps (both variables declared jump, n_x = 2 > n_u = 1)
    -> INDETERMINATE.
Only the matrices and n_predetermined are stored; the verdict is asserted in
the Rust property tests.
"""

import json
import numpy as np


def cagan(a, rho, sigma):
    M = np.array([[rho, 0.0], [-1.0 / a, 1.0 / a]])
    A = np.eye(2)
    B = M.copy()
    C = np.array([[sigma], [0.0]])
    g = np.array([[1.0 / (1.0 - a * rho)]])          # p_t = u_t / (1 - a rho)
    P = np.array([[rho]])
    Q = np.array([[sigma]])
    eig = sorted([rho, 1.0 / a])
    return A, B, C, g, P, Q, eig


def two_shock(a, rho1, rho2, sigma1, sigma2):
    M = np.array(
        [
            [rho1, 0.0, 0.0],
            [0.0, rho2, 0.0],
            [-1.0 / a, -1.0 / a, 1.0 / a],
        ]
    )
    A = np.eye(3)
    B = M.copy()
    C = np.array([[sigma1, 0.0], [0.0, sigma2], [0.0, 0.0]])
    g = np.array([[1.0 / (1.0 - a * rho1), 1.0 / (1.0 - a * rho2)]])
    P = np.diag([rho1, rho2])
    Q = np.diag([sigma1, sigma2])
    eig = sorted([rho1, rho2, 1.0 / a])
    return A, B, C, g, P, Q, eig


def cross_check_eigs(B, analytic):
    """INDEPENDENT check: numpy's eigvals of M = A^{-1}B (A = I here) must match
    the analytic eigenvalues used above. Separate code path from faer."""
    got = sorted(np.real(np.linalg.eigvals(B)))
    assert np.allclose(got, analytic, atol=1e-12), (got, analytic)


def block(A, B, C, g, P, Q, eig, n_pre, name, desc):
    cross_check_eigs(B, eig)
    return {
        "name": name,
        "description": desc,
        "n_predetermined": n_pre,
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
        "G": g.tolist(),
        "P": P.tolist(),
        "Q": Q.tolist(),
        "eigenvalues_sorted_abs": [abs(e) for e in sorted(eig, key=abs)],
        "eigenvalues_real_sorted": sorted(eig),
    }


def main():
    out = {
        "_doc": "Closed-form Blanchard-Kahn golden; see generator docstring.",
        "cagan": block(
            *cagan(a=0.5, rho=0.6, sigma=1.0),
            n_pre=1,
            name="cagan_asset_price",
            desc="p_t = a E_t p_{t+1} + u_t, u AR(1); G = 1/(1 - a rho).",
        ),
        "two_shock": block(
            *two_shock(a=0.4, rho1=0.5, rho2=0.8, sigma1=1.0, sigma2=0.7),
            n_pre=2,
            name="two_shock_jump",
            desc="p driven by two AR(1) fundamentals; diagonal P and Q.",
        ),
    }

    # Model 3: mis-specification cases (matrices only, verdict asserted in Rust).
    A, B, C, _, _, _, _ = cagan(a=0.5, rho=0.6, sigma=1.0)
    out["misspec_too_few_jumps"] = {
        "description": "Cagan matrices with n_predetermined = 2 (n_jump = 0 < "
        "n_unstable = 1): NO stable solution.",
        "n_predetermined": 2,
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
    }
    out["misspec_too_many_jumps"] = {
        "description": "Cagan matrices with n_predetermined = 0 (n_jump = 2 > "
        "n_unstable = 1): INDETERMINATE.",
        "n_predetermined": 0,
        "A": A.tolist(),
        "B": B.tolist(),
        "C": C.tolist(),
    }

    with open("fixtures/tsecon-dsge.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote fixtures/tsecon-dsge.json")


if __name__ == "__main__":
    main()
