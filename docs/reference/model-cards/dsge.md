# Model card — Linear rational-expectations (DSGE-lite) solver

`dsge_solve`

A linearized rational-expectations model writes today's variables in terms of
what agents *expect* tomorrow's variables to be. The forward-looking pieces — a
price level, an asset price, a shadow value — carry no natural initial
condition: nothing in the past pins them down. What pins them is the refusal to
explode. Of all the paths consistent with the model's equations, only the ones
that do not blow up are admissible, and it is that no-bubble requirement, not an
initial value, that selects a solution. Whether it selects *exactly one* is the
existence-and-uniqueness question **Blanchard & Kahn (1980)** answered.

Their answer is a counting rule. Write the model in reduced form and
eigen-decompose its transition matrix; each eigenvalue with modulus above one is
an *unstable* direction the solution must steer away from, and each
non-predetermined **jump** variable is a free coordinate the model gets to choose
so that it does. A unique non-explosive solution exists precisely when the number
of unstable eigenvalues equals the number of jump variables — the jumps have
exactly enough freedom to zero out every explosive direction. Too few unstable
roots and the model is **indeterminate**: a continuum of stable solutions, the
door through which sunspots and self-fulfilling beliefs enter. Too many and there
is **no stable solution** at all; everything explodes. When the counts match, the
stable eigenvectors deliver a **policy rule** that ties each jump to the
predetermined state, and a **law of motion** that propagates the state forward —
stably, by construction.

This is the roadmap's deliberately minimal layer: the linear RE *solver*, not a
full DSGE estimation suite. You hand it the three matrices of an already
linearized model and it returns the policy rule `G`, the state transition `P`,
the shock impact `Q`, the eigenvalue moduli that drove the verdict, and a
plain-language classification into the three Blanchard-Kahn regimes. The model is
supplied in first-order expectational form

```text
A · E_t[y_{t+1}] = B · y_t + C · z_{t+1},     y_t = [ predetermined ; jump ]
```

with the `n_predetermined` backward-looking variables stacked on top of the
forward-looking ones and `z_{t+1}` a mean-zero innovation (`E_t[z_{t+1}] = 0`).
Writing `M = A⁻¹B` and `N = A⁻¹C` gives the reduced form `E_t[y_{t+1}] = M·y_t + N·z`
that the eigen-machinery works on.

---

## `dsge_solve` — Blanchard-Kahn policy rule and law of motion

**What it estimates.** Given the linear model matrices `A`, `B`, `C` and the
number of predetermined variables, it computes the rational-expectations
solution in state-space form: the **policy matrix** `G` mapping the predetermined
state to the jumps (`jump_t = G·predetermined_t`), the **law of motion** `P` and
**impact matrix** `Q` governing the predetermined block
(`predetermined_{t+1} = P·predetermined_t + Q·z`), the sorted **eigenvalue
moduli** of the reduced form `M = A⁻¹B`, and a **verdict** classifying the model
as having a unique stable solution, being indeterminate, or having no stable
solution. Nothing here is estimated from data — the "solve" is a deterministic
linear-algebra map from a calibrated/estimated model to its decision rule. The
policy rule comes from the stable eigenvectors, `G = V_xs·V_ks⁻¹`, and the state
dynamics from `P = M_kk + M_kx·G`, `Q = N_k`; the stable eigenvalues of `M` are
exactly the eigenvalues of `P`, so a returned `P` is always stable.

**Assumptions.** The model is already linear (or log-linear) around a steady
state and cast in the expectational form above. The lead matrix `A` is
**invertible** — the solver forms `M = A⁻¹B` explicitly, the "regular" case. The
predetermined variables occupy the **top** `n_predetermined` rows of the stacked
state `y_t = [predetermined ; jump]`; that ordering is a promise the caller makes,
not something the solver infers. Innovations load **only on predetermined
(exogenous-state) equations**: because `E_t[z_{t+1}] = 0`, the shock drops out of
the forward-looking solve and re-enters only the law of motion, so a shock
written onto a jump row is not representable as `jump_t = G·predetermined_t` and
is rejected. The stable subspace of `M` must be diagonalizable (no defective
repeated roots), and no eigenvalue may sit on the unit circle, where the
stable/unstable split is undefined.

**When to use (and when not).** Use it to solve any small linearized
forward-looking model to its decision rule and to *check determinacy* — the
verdict is the headline: does this calibration deliver a unique non-explosive
equilibrium? It is the right tool for textbook and teaching-scale models —
Cagan/asset-price pricing, a Fisherian or exogenous-block New-Keynesian core, an
`(I − aR)⁻¹` present-value multiplier — and for scanning a parameter grid to map
the determinacy region (where the count of unstable roots crosses the count of
jumps). Do **not** reach for it as a DSGE *estimation* package: there is no
likelihood, no prior, no data step here. Do not hand it a **singular pencil** — a
model with static/definitional equations that make `A` singular; substitute those
out first or use a QZ (generalized-Schur / gensys) solver. Do not route a shock
directly onto a forward-looking equation; move it onto an exogenous AR state, as
the Cagan example does. And do not read a determinacy verdict as economic truth
if you mis-declared `n_predetermined` — the count of jumps is *your* input, and
getting it wrong flips the verdict.

**Key arguments and defaults (and why).** There are no soft tuning knobs — every
argument is a structural declaration. `a`, `b`, `c` are the `n×n`, `n×n`, and
`n×m` matrices `A`, `B`, `C`; `A` and `B` describe the `n` endogenous equations
and `C` maps the `m` innovations onto them. `n_predetermined` is the integer
count of backward-looking variables and is the one input that carries economic
content beyond the algebra: it is *how many* of the top rows are predetermined,
and therefore — since the total `n` is fixed — how many jump variables the
Blanchard-Kahn count is compared against. It must satisfy
`0 ≤ n_predetermined ≤ n`. The stacking convention (predetermined on top, jumps
below) is fixed, not a default you can flip: order your state vector to match it.
The shock-routing convention (innovations on predetermined rows only) is likewise
structural — it falls out of `E_t[z_{t+1}] = 0`, not out of a parameter choice.

**How to read the output.** A dict with five keys.

- **`verdict`** — a plain-language string; read it first. For a well-posed model
  it reads `"unique stable solution (N unstable eigenvalue(s) = N jump
  variable(s))"`, the two counts being equal by definition of the unique case.
  When they disagree the verdict names the regime (indeterminate, or no stable
  solution) and the solve raises rather than returning matrices.
- **`eigenvalue_moduli`** — a NumPy array of the moduli `|λ|` of the reduced-form
  `M`, sorted ascending. Entries below `1` are the stable roots (they become the
  eigenvalues of `P`); entries above `1` are the unstable roots the jumps must
  neutralize. Eyeball where the array crosses `1`: the count above the line is
  `n_unstable`, and it should equal your jump count.
- **`g`** — nested Python lists, the policy matrix `G` (`n_jump × n_predetermined`).
  Row `i` gives how jump `i` loads on each predetermined variable:
  `jump_t = G·predetermined_t`.
- **`p`** — nested lists, the transition matrix `P` (`n_predetermined ×
  n_predetermined`) of `predetermined_{t+1} = P·predetermined_t + Q·z`. It is
  always stable.
- **`q`** — nested lists, the impact matrix `Q` (`n_predetermined × m`) of the
  innovation on the predetermined block.

The Python binding exposes exactly these five; the impulse responses and
simulations that the underlying Rust crate offers are not surfaced, but you do
not need them — given `G`, `P`, `Q` you can trace any response yourself by
iterating `predetermined_{t+1} = P·predetermined_t` from an impact `Q·z` and
reading `jump_t = G·predetermined_t` off each step, exactly the loop shown below.

**Failure modes.** Each is raised, never silently swallowed. **Singular `A`** —
the lead matrix is not invertible, so `M = A⁻¹B` does not exist; this is the
"error that teaches" (message: the solver handles only an invertible `A` —
substitute out static equations or use a QZ-based solver for a singular pencil).
**Shock on a jump row** — an innovation loads on a forward-looking equation, which
the `E_t[z_{t+1}] = 0` convention cannot represent; route it through an exogenous
AR state. **Mis-declared `n_predetermined`** — declare too few jumps and a
genuinely unique model is flagged *no stable solution*; too many and it is flagged
*indeterminate*. The verdict is only as trustworthy as the count you supply.
**Unit root** — an eigenvalue on (within tolerance of) the unit circle leaves the
stable/unstable split undefined; re-specify away from the boundary. **Defective
stable block** — a repeated eigenvalue with too few eigenvectors makes
`G = V_xs·V_ks⁻¹` undefined (the model must be diagonalizable on its stable
subspace). **Non-negligible imaginary policy rule** — for a real model complex
eigenvalues cancel in conjugate pairs; a residual imaginary part signals a
near-defective eigenspace or numerical breakdown. Dimension, emptiness, and
non-finite entries are checked at model construction.

**Validated against.** The crate's own **documented closed-form golden**
(`fixtures/tsecon-dsge.json`, generated by
`fixtures/generate_tsecon-dsge_fixtures.py`) — the generator types the analytic
`G`, `P`, `Q` straight from the textbook derivation and *never calls the solver*,
so the match is non-circular. It pins two models to ~1e-8: the Cagan/asset-price
model `p_t = a·E_t[p_{t+1}] + u_t` with an AR(1) fundamental, whose fundamental
(no-bubble) solution `p_t = u_t/(1 − a·rho)` gives `G = 1/(1 − a·rho)`, `P = rho`,
`Q = sigma`; and a two-shock variant with a diagonal `P` and a multi-column `G`.
The generator independently re-derives the eigenvalues via `numpy.linalg.eigvals`
— a separate code path from the crate's real-Schur eigensolver — so the
eigenvalue check is genuine. Crate **property tests** (`tests/properties.rs`)
additionally confirm the solved `P` is stable, that Blanchard-Kahn flags a
too-few-jumps model as *no stable solution* and a too-many-jumps model as
*indeterminate*, that impulse responses revert to zero, that a complex-conjugate
(oscillatory) exogenous block yields a *real* policy rule matching the closed-form
`(I − aR)⁻¹` forward sum, and that the singular-`A`, shock-on-jump, and unit-root
cases each raise their specific error. The **Python binding** is structurally
tested in `bindings/python/tests/test_spectest_afns_dsge.py`: the Cagan saddle
path (verdict, `G = 1/(1 − a·rho)`, and the stable/unstable eigenvalue moduli
`{rho, 1/a}`) and the singular-`A` teaching error. There is no statsmodels or
Dynare cross-check — the reference is the analytic closed form, checked
independently of the solver.

**References.** Blanchard, O. J. & Kahn, C. M. (1980), "The solution of linear
difference models under rational expectations," *Econometrica* 48(5):1305-1311.
Cagan, P. (1956), "The monetary dynamics of hyperinflation," in *Studies in the
Quantity Theory of Money* (M. Friedman, ed.), University of Chicago Press. Sims,
C. A. (2002), "Solving linear rational expectations models," *Computational
Economics* 20:1-20 (the QZ / gensys generalization for the singular-pencil case
this solver leaves out of scope).

```python
import numpy as np, tsecon

# Cagan money demand:  m_t = a·E_t[m_{t+1}] + rho·x_t,   x_t = rho·x_{t-1} + eps_t
# with a = 0.7 (semi-elasticity) and an AR(1) forcing variable, rho = 0.6.
# Stack y = (x, m):  x is predetermined (the exogenous state), m is the jump.
a, rho = 0.7, 0.6
A = np.array([[1.0, 0.0],
              [0.0, a]])          # lead matrix (invertible)
B = np.array([[rho, 0.0],
              [-1.0, 1.0]])
C = np.array([[1.0],
              [0.0]])             # eps loads on the predetermined (x) row only

sol = tsecon.dsge_solve(A, B, C, n_predetermined=1)

print("verdict :", sol["verdict"])
print("g       :", sol["g"], "   (m_t = x_t / (1 - a*rho))")
print("p       :", sol["p"], "   (x reverts at its own AR root rho)")
print("q       :", sol["q"])
print("|eig|   :", np.round(sol["eigenvalue_moduli"], 6),
      "  (stable rho, unstable 1/a)")
print("G check :", 1.0 / (1.0 - a * rho))
# verdict : unique stable solution (1 unstable eigenvalue(s) = 1 jump variable(s))
# g       : [[1.7241379310344827]]    (m_t = x_t / (1 - a*rho))
# p       : [[0.6]]    (x reverts at its own AR root rho)
# q       : [[1.0]]
# |eig|   : [0.6      1.428571]   (stable rho, unstable 1/a)
# G check : 1.7241379310344827
```

The economics is the whole point of the counting rule. There is one unstable
root (`1/a = 1.4286`, above the circle) and one jump variable (`m`), so
Blanchard-Kahn is satisfied with equality: the price level `m` is *pinned* to the
fundamental `x` by the unique forward-looking solution, with no bubble term free
to roam. The loading `G = 1/(1 − a·rho) = 1/0.58 = 1.7241` is exactly the Cagan
present-value multiplier — the discounted sum `Σ (a·rho)^j` of the fundamental's
own persistence. The stable root is just `rho = 0.6`, the fundamental's AR root,
which reappears as `P`: the state reverts at its own pace and the jump rides along.

**Tracing the saddle path.** The binding returns matrices, not trajectories, but
`G`, `P`, `Q` are all you need. A one-time unit innovation `eps = 1` lands on the
state as `Q·eps`, then propagates by `x_{t+1} = P·x_t` while the jump reads off as
`m_t = G·x_t` at each date — the impulse response, by hand:

```python
import numpy as np, tsecon

a, rho = 0.7, 0.6
A = np.array([[1.0, 0.0], [0.0, a]])
B = np.array([[rho, 0.0], [-1.0, 1.0]])
C = np.array([[1.0], [0.0]])
sol = tsecon.dsge_solve(A, B, C, n_predetermined=1)

P = np.asarray(sol["p"], float)          # state transition
Q = np.asarray(sol["q"], float)          # innovation impact
G = np.asarray(sol["g"], float)          # jump loading

k = Q @ np.array([1.0])                   # impact of a unit eps on the state x
for t in range(6):
    x, m = k[0], (G @ k)[0]
    print(f"t={t}:  x={x:6.4f}   m={m:6.4f}")
    k = P @ k                             # x_{t+1} = P x_t
# t=0:  x=1.0000   m=1.7241
# t=1:  x=0.6000   m=1.0345
# t=2:  x=0.3600   m=0.6207
# t=3:  x=0.2160   m=0.3724
# t=4:  x=0.1296   m=0.2234
# t=5:  x=0.0778   m=0.1341
```

Both series decay at the stable root `rho = 0.6`, and at every date the price is
`1.7241 ×` the fundamental — the saddle path, drawn one step at a time.

**The error that teaches.** Give the solver a **singular lead matrix** `A` and it
does not produce garbage — it tells you the reduced form `M = A⁻¹B` does not exist
and names the QZ generalization it would take to handle a singular pencil. (Here
`A` is all-zeros, the degenerate limit; the same error fires for any rank-deficient
`A`, e.g. a model with an un-substituted static/definitional equation.)

```python
import numpy as np, tsecon

A = np.zeros((2, 2))                       # singular lead matrix
B = np.array([[0.6, 0.0], [-1.0, 1.0]])
C = np.array([[1.0], [0.0]])
try:
    tsecon.dsge_solve(A, B, C, n_predetermined=1)
except Exception as exc:
    print(type(exc).__name__, ":", exc)
# ValueError : the lead matrix A is singular, so M = A^{-1} B does not exist; this
# solver handles only an invertible A — substitute out static equations, or use a
# QZ-based solver for a singular pencil
```
