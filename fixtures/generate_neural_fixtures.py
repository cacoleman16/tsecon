"""Golden fixtures for the neural slice of tsecon-ml: `mlp_regression`
(feed-forward regressor) and `echo_state_network` (reservoir computing).

Two legs, graded separately and honestly.

MLP — INDEPENDENT-PACKAGE golden against scikit-learn 1.9.0 `MLPRegressor`,
designed so the Rust never has to reproduce an optimizer trajectory (which
no two Adam/L-BFGS implementations share bit for bit). For each
(architecture, activation) case the fixture stores sklearn's *fitted*
`coefs_` / `intercepts_` (solver="lbfgs", run to its own stopping rule),
and the pins are on the mechanics evaluated AT THOSE WEIGHTS:

  (a) `predict_test` — sklearn `predict` on held-out rows (the Rust forward
      pass must match at 1e-12);
  (b) `loss_fitted` — the scikit-learn objective at the fitted weights,

          L(W, b) = (1/(2n)) sum_i (y_i - yhat_i)^2
                    + (alpha/(2n)) sum_layers ||W_l||_F^2,

      intercepts unpenalized (`_multilayer_perceptron._backprop`:
      `squared_loss` = 0.5 * mean((y - yhat)^2), plus
      `0.5 * alpha * sum(coef.ravel() @ coef.ravel()) / n_samples`), recomputed
      here from that formula in NumPy AND read back from sklearn's own
      `_backprop` and from `est.loss_` (which for lbfgs is `opt_res.fun`, the
      objective at the returned weights) — all three asserted identical
      before anything is stored;
  (c) `random_weights` — Glorot-scale random weights with sklearn's
      `_backprop` loss and analytic gradient at those weights (the Rust
      analytic gradient must match sklearn's backprop at 1e-10, and its own
      central finite difference at 1e-6 relative on the smooth activations);
  (d) `grad_fitted` / `grad_norm_inf_fitted` — sklearn's backprop gradient
      at its converged weights, with the generator asserting the solution is
      stationary (inf-norm below STATIONARY_BAR) before storing; the Rust
      gradient at those weights must reproduce the norm at 1e-8.

  What this does NOT pin: the trajectory of either optimizer, early
  stopping, or the seed ensemble. Those are covered by seeded
  Monte-Carlo / property tests on the Rust side (recovery of a nonlinear
  AR(1) map out of sample, ensemble-beats-median-member, early stopping
  fires / does not, seed contract) whose numbers the model card quotes.

ESN — the reservoir mechanics are a TRANSCRIPTION golden (Jaeger 2001;
Lukosevicius 2012, "A practical guide to applying echo state networks",
eqs. 2-3 and 9): an explicit small reservoir W and input matrix W_in are
drawn HERE in NumPy, the leaky-integrator state path

        s_t = (1 - a) s_{t-1} + a tanh(W s_{t-1} + W_in u_t),   s_0 = 0,

is transcribed in a NumPy loop, and the readout is the ridge solution on
Z_t = [1, u_t, s_t] for t >= washout, minimizing ||y - Z b||^2 + alpha
||b||^2 (the intercept column is penalized like every other coefficient,
Lukosevicius eq. 9 — the same scikit-learn `Ridge(fit_intercept=False)`
convention the crate's `ridge` already carries). The readout is
cross-checked against scikit-learn `Ridge` (an independent-package leg for
the regression step). If `reservoirpy` is importable, its `Reservoir` is
run with the SAME explicit W / W_in / lr and its state path is pinned
against the transcription; `_meta.esn.reservoirpy` records the outcome so
the grade of the state-path leg is stated honestly (third-party when the
pin held, transcription otherwise). `spectral` stores a sparse random
matrix whose spectral radius NumPy computed with `numpy.linalg.eigvals`,
so the Rust radius estimate — and the rescaling to a target radius — can
be pinned at 1e-6 (no third-party ESN library is needed for that: the
eigenvalue is the reference).

Statistical correctness of the public ESN estimator (NARMA-10 NRMSE, seed
contract) is established by property tests, whose numbers are quoted in
the model card.

This generator NEVER imports tsecon. Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`). All data are seeded simulations.

Run:  .venv-wt/bin/python fixtures/generate_neural_fixtures.py
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

OUT = Path(__file__).resolve().parent / "neural.json"

SEED = 20260903
# The generator asserts sklearn's own L-BFGS solution is stationary to this
# inf-norm before storing it as a "converged weights" case. scipy's L-BFGS-B
# stops on its relative-decrease rule (ftol ~ 2.2e-9) long before gtol=1e-12,
# so the achieved norm is small but not zero; the value is stored per case.
# Measured: the smooth activations (tanh, logistic) reach ~5e-5; the relu
# objective is only piecewise smooth and its L-BFGS-B run stops at ~2e-3,
# which sets the bar. What the Rust test pins is the *stored* norm at 1e-8.
STATIONARY_BAR = 1e-2

ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
}


# ------------------------------------------------------------------- data


def make_mlp_data(seed):
    """Seeded nonlinear regression: n=200 training rows, p=3 inputs."""
    rng = np.random.default_rng(seed)
    n, p, n_test = 200, 3, 20
    x = rng.standard_normal((n + n_test, p))
    f = np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.3 * x[:, 2] * x[:, 0]
    y = f + 0.1 * rng.standard_normal(n + n_test)
    return x[:n], y[:n], x[n:]


# -------------------------------------------------- the MLP transcription


def forward(coefs, intercepts, act, x):
    """sklearn `_forward_pass`: hidden layers `act`, identity output."""
    a = x
    n_layers = len(coefs)
    for i in range(n_layers):
        z = a @ coefs[i] + intercepts[i]
        a = act(z) if i < n_layers - 1 else z
    return a[:, 0]


def objective(coefs, intercepts, act, x, y, alpha):
    """The scikit-learn MLPRegressor objective (module docstring, eq. (b))."""
    n = x.shape[0]
    yhat = forward(coefs, intercepts, act, x)
    data_fit = 0.5 * np.mean((y - yhat) ** 2)
    penalty = 0.5 * alpha * sum(float(c.ravel() @ c.ravel()) for c in coefs) / n
    return float(data_fit + penalty)


def sklearn_backprop(est, coefs, intercepts, x, y):
    """Loss and gradients from sklearn's OWN `_backprop` at arbitrary weights.

    The estimator's fitted arrays are swapped for `coefs`/`intercepts`, the
    private routine is called exactly as `_loss_grad_lbfgs` calls it, and
    the arrays are restored afterwards.
    """
    saved = (est.coefs_, est.intercepts_)
    est.coefs_ = [c.copy() for c in coefs]
    est.intercepts_ = [b.copy() for b in intercepts]
    try:
        n_samples = x.shape[0]
        layer_units = [x.shape[1]] + list(est.hidden_layer_sizes) + [1]
        activations = [x] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)
        coef_grads = [
            np.empty((n_in, n_out)) for n_in, n_out in zip(layer_units[:-1], layer_units[1:])
        ]
        intercept_grads = [np.empty(n_out) for n_out in layer_units[1:]]
        loss, cg, ig = est._backprop(
            x, y.reshape(-1, 1), None, activations, deltas, coef_grads, intercept_grads
        )
        assert n_samples == x.shape[0]
        return float(loss), [g.copy() for g in cg], [g.copy() for g in ig]
    finally:
        est.coefs_, est.intercepts_ = saved


def glorot_random(rng, layer_units, activation):
    """Random weights on sklearn's Glorot-uniform scale (its `_init_coef`)."""
    factor = 2.0 if activation == "logistic" else 6.0
    coefs, intercepts = [], []
    for fan_in, fan_out in zip(layer_units[:-1], layer_units[1:]):
        bound = np.sqrt(factor / (fan_in + fan_out))
        coefs.append(rng.uniform(-bound, bound, size=(fan_in, fan_out)))
        intercepts.append(rng.uniform(-bound, bound, size=fan_out))
    return coefs, intercepts


def mlp_cases(x, y, x_test):
    specs = [
        ("tanh_16", (16,), "tanh", 1e-4, 1),
        ("relu_8_4", (8, 4), "relu", 1e-3, 2),
        ("logistic_10", (10,), "logistic", 1e-4, 3),
        ("tanh_12_6", (12, 6), "tanh", 1e-4, 4),
    ]
    cases = []
    for name, hidden, activation, alpha, rs in specs:
        est = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=activation,
            solver="lbfgs",
            alpha=alpha,
            max_iter=20000,
            max_fun=200000,
            tol=1e-12,
            random_state=rs,
        )
        est.fit(x, y)
        coefs = [c.copy() for c in est.coefs_]
        intercepts = [b.copy() for b in est.intercepts_]
        act = ACTIVATIONS[activation]

        # (a) forward pass = sklearn predict
        pred = est.predict(x_test)
        assert np.max(np.abs(forward(coefs, intercepts, act, x_test) - pred)) < 1e-12

        # (b) objective: formula == sklearn _backprop == est.loss_
        loss_formula = objective(coefs, intercepts, act, x, y, alpha)
        loss_bp, cg, ig = sklearn_backprop(est, coefs, intercepts, x, y)
        assert abs(loss_formula - loss_bp) < 1e-12, (loss_formula, loss_bp)
        assert abs(loss_formula - est.loss_) < 1e-12, (loss_formula, est.loss_)

        # (d) stationarity of sklearn's own solution
        grad_inf = max(float(np.max(np.abs(g))) for g in cg + ig)
        grad_2 = float(np.sqrt(sum(float(g.ravel() @ g.ravel()) for g in cg + ig)))
        assert grad_inf < STATIONARY_BAR, (name, grad_inf)

        # (c) sklearn backprop at Glorot-scale random weights
        rng = np.random.default_rng(SEED + rs)
        layer_units = [x.shape[1]] + list(hidden) + [1]
        rc, ri = glorot_random(rng, layer_units, activation)
        loss_r, cg_r, ig_r = sklearn_backprop(est, rc, ri, x, y)
        assert abs(loss_r - objective(rc, ri, act, x, y, alpha)) < 1e-12

        cases.append(
            {
                "name": name,
                "hidden": list(hidden),
                "activation": activation,
                "alpha": alpha,
                "random_state": rs,
                "n_iter": int(est.n_iter_),
                "coefs": [c.tolist() for c in coefs],
                "intercepts": [b.tolist() for b in intercepts],
                "predict_test": pred.tolist(),
                "loss_fitted": loss_formula,
                "loss_attr": float(est.loss_),
                "grad_fitted": {
                    "coefs": [g.tolist() for g in cg],
                    "intercepts": [g.tolist() for g in ig],
                },
                "grad_norm_inf_fitted": grad_inf,
                "grad_norm_2_fitted": grad_2,
                "random_weights": {
                    "coefs": [c.tolist() for c in rc],
                    "intercepts": [b.tolist() for b in ri],
                    "loss": loss_r,
                    "grad_coefs": [g.tolist() for g in cg_r],
                    "grad_intercepts": [g.tolist() for g in ig_r],
                },
            }
        )
        print(
            f"  mlp {name}: n_iter={est.n_iter_} loss={loss_formula:.6e} "
            f"|grad|_inf={grad_inf:.3e}"
        )
    return cases


# ------------------------------------------------- the ESN transcription


def esn_states(w, w_in, u, leak):
    """Lukosevicius (2012) eqs. 2-3 with zero initial state, no bias."""
    n_units = w.shape[0]
    s = np.zeros(n_units)
    out = np.empty((u.shape[0], n_units))
    for t in range(u.shape[0]):
        s = (1.0 - leak) * s + leak * np.tanh(w @ s + w_in @ u[t])
        out[t] = s
    return out


def esn_readout(states, u, y, washout, alpha):
    """Ridge on Z = [1, u, s], rows t >= washout: (Z'Z + alpha I)^{-1} Z'y."""
    z = np.column_stack([np.ones(u.shape[0]), u, states])[washout:]
    yy = y[washout:]
    b = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ yy)
    return z, b


def esn_transcription():
    rng = np.random.default_rng(SEED + 100)
    n_units, p, t_len = 6, 2, 40
    leak, washout, alpha, target_sr = 0.7, 5, 1e-3, 0.8
    mask = rng.uniform(size=(n_units, n_units)) < 0.5
    w = rng.standard_normal((n_units, n_units)) * mask
    w *= target_sr / np.max(np.abs(np.linalg.eigvals(w)))
    w_in = rng.uniform(-1.0, 1.0, size=(n_units, p))
    u = rng.uniform(-0.5, 0.5, size=(t_len, p))
    y = np.sin(u[:, 0] * 3.0) + 0.5 * u[:, 1] + 0.05 * rng.standard_normal(t_len)

    states = esn_states(w, w_in, u, leak)
    z, b = esn_readout(states, u, y, washout, alpha)
    # Independent-package cross-check of the readout: scikit-learn Ridge
    # (fit_intercept=False, the same objective) on the same design.
    ridge = Ridge(alpha=alpha, fit_intercept=False, solver="svd").fit(z, y[washout:])
    ridge_gap = float(np.max(np.abs(ridge.coef_ - b)))
    assert ridge_gap < 1e-10, ridge_gap
    fitted = z @ b

    rp_meta = {"installed": False, "version": None, "max_abs_state_diff": None}
    try:
        import reservoirpy
        from reservoirpy.nodes import Reservoir

        res = Reservoir(units=n_units, lr=leak, W=w.copy(), Win=w_in.copy(), bias=0.0,
                        activation="tanh")
        rp_states = np.asarray(res.run(u.copy()))
        gap = float(np.max(np.abs(rp_states - states)))
        rp_meta = {
            "installed": True,
            "version": reservoirpy.__version__,
            "max_abs_state_diff": gap,
        }
        assert gap < 1e-12, gap
        print(f"  esn: reservoirpy {reservoirpy.__version__} state path pinned, "
              f"max abs diff {gap:.3e}")
    except ImportError:
        print("  esn: reservoirpy not importable — state path is transcription-graded")

    return {
        "w": w.tolist(),
        "w_in": w_in.tolist(),
        "u": u.tolist(),
        "y": y.tolist(),
        "leak_rate": leak,
        "washout": washout,
        "ridge_alpha": alpha,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(w)))),
        "states": states.tolist(),
        "readout": b.tolist(),
        "readout_sklearn_ridge_max_abs_diff": ridge_gap,
        "fitted": fitted.tolist(),
    }, rp_meta


def esn_spectral():
    """A sparse random matrix and its exact spectral radius (numpy eigvals)."""
    rng = np.random.default_rng(SEED + 200)
    n = 30
    mask = rng.uniform(size=(n, n)) < 0.2
    w = rng.standard_normal((n, n)) * mask
    eig = np.linalg.eigvals(w)
    return {
        "w": w.tolist(),
        "radius_numpy": float(np.max(np.abs(eig))),
        "target": 0.9,
    }


def main():
    x, y, x_test = make_mlp_data(SEED)
    mlp = {
        "x_train": x.tolist(),
        "y_train": y.tolist(),
        "x_test": x_test.tolist(),
        "cases": mlp_cases(x, y, x_test),
    }
    transcription, rp_meta = esn_transcription()
    spectral = esn_spectral()
    print(f"  esn spectral: numpy radius {spectral['radius_numpy']:.12f}")

    fixture = {
        "_meta": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "python": platform.python_version(),
            "seed": SEED,
            "mlp": {
                "grade": (
                    "independent package: scikit-learn 1.9.0 MLPRegressor "
                    "(solver=lbfgs) — forward pass, objective, and analytic "
                    "gradient pinned AT sklearn's fitted and at random weights; "
                    "the optimizer trajectory is deliberately NOT pinned"
                ),
                "objective_note": (
                    "L = (1/(2n)) sum (y - yhat)^2 + (alpha/(2n)) sum_l ||W_l||_F^2; "
                    "intercepts unpenalized; hidden activation tanh/relu/logistic, "
                    "identity output — sklearn _backprop / squared_loss convention."
                ),
                "stationary_bar_inf_norm": STATIONARY_BAR,
                "weights_layout": (
                    "coefs[l] is (fan_in, fan_out) — rows index inputs, columns "
                    "units — and intercepts[l] is (fan_out,), sklearn's layout."
                ),
            },
            "esn": {
                "grade": (
                    "state path: "
                    + (
                        "third-party (reservoirpy Reservoir with the same explicit "
                        "W / Win / lr pinned the NumPy transcription at "
                        f"{rp_meta['max_abs_state_diff']:.3e})"
                        if rp_meta["installed"]
                        else "documented-algorithm transcription (reservoirpy not "
                        "importable at generation time)"
                    )
                    + "; readout: closed-form ridge cross-checked against "
                    "scikit-learn Ridge(fit_intercept=False); spectral radius: "
                    "numpy.linalg.eigvals."
                ),
                "state_equation": (
                    "s_t = (1 - a) s_{t-1} + a tanh(W s_{t-1} + W_in u_t), s_0 = 0, "
                    "no reservoir bias (Lukosevicius 2012 eqs. 2-3; reservoirpy "
                    "Reservoir._step with bias=0)."
                ),
                "readout_note": (
                    "b = argmin ||y - Z b||^2 + alpha ||b||^2 on Z_t = [1, u_t, s_t], "
                    "t >= washout (the constant column IS penalized: Lukosevicius "
                    "eq. 9 / scikit-learn Ridge(fit_intercept=False))."
                ),
                "reservoirpy": rp_meta,
            },
        },
        "mlp": mlp,
        "esn": {"transcription": transcription, "spectral": spectral},
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
