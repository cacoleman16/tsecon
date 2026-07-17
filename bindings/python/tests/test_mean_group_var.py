"""Property tests for the Pesaran-Smith mean-group panel VAR binding.

The mean-group (MG) estimator fits a separate reduced-form VAR to every
entity and averages, with dispersion-based cross-entity standard errors
`sd(theta_i) / sqrt(N)`. There is no external golden here; instead we pin
the MG output against the *already-bound* per-entity primitives — `var_fit`
(coefficients / intercept) and `var_irf` (orthogonalized IRFs) — averaged
by hand. Since `mean_group_var` fits each entity through the exact same
`VarSpec::fit(...).irf(...)` code path, the average must agree to machine
precision.
"""
import numpy as np
import pytest
import tsecon


def _simulate_panel(n_units=4, T=200, k=2, seed=0):
    """A small heterogeneous panel of stable VAR(1) processes.

    Each entity gets its own (distinct) intercept and lag matrix, so the
    per-entity fits genuinely differ and the cross-entity dispersion is
    non-degenerate. Returned as a list of (T x k) matrices, oldest row
    first — the list-of-2D-arrays shape the binding accepts.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[0.5, 0.1], [0.05, 0.4]])
    entities = []
    for _ in range(n_units):
        A = base + 0.05 * rng.standard_normal((k, k))
        c = 0.2 * rng.standard_normal(k)
        Y = np.zeros((T, k))
        y = np.zeros(k)
        for t in range(T):
            y = c + A @ y + 0.5 * rng.standard_normal(k)
            Y[t] = y
        entities.append(Y)
    return entities


def _unit_var1(entity):
    """Per-entity VAR(1) intercept and A_1 via the bound `var_fit`.

    `var_fit` returns statsmodels-layout `params` of shape (1 + k) x k:
    row 0 is the intercept per equation; rows 1.. are the lag block. The
    crate builds `coefs[0][(r, c)] = params[1 + c, r]` (estimate.rs), so
    A_1 = params[1:, :].T and intercept = params[0, :].
    """
    fit = tsecon.var_fit(entity, lags=1, trend="c")
    params = np.asarray(fit["params"])
    intercept = params[0, :].copy()
    A1 = params[1:, :].T.copy()
    return intercept, A1


def test_mean_group_equals_average_of_unit_vars():
    entities = _simulate_panel()
    N = len(entities)
    H = 6
    mg = tsecon.mean_group_var(entities, lags=1, trend="c", horizon=H)

    unit_intercepts, unit_A1, unit_irfs = [], [], []
    for e in entities:
        ic, A1 = _unit_var1(e)
        unit_intercepts.append(ic)
        unit_A1.append(A1)
        unit_irfs.append(np.asarray(tsecon.var_irf(e, lags=1, horizon=H,
                                                    orth=True, trend="c")))
    unit_intercepts = np.array(unit_intercepts)   # N x k
    unit_A1 = np.array(unit_A1)                    # N x k x k
    unit_irfs = np.array(unit_irfs)                # N x (H+1) x k x k

    # Point estimates: MG = simple cross-entity average.
    np.testing.assert_allclose(mg["intercept"], unit_intercepts.mean(0),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mg["coefs"])[0], unit_A1.mean(0),
                               rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mg["orth_irfs"]), unit_irfs.mean(0),
                               rtol=1e-9, atol=1e-12)

    # Dispersion SEs: sd across entities / sqrt(N), (N-1) divisor.
    def disp_se(x):
        return x.std(0, ddof=1) / np.sqrt(N)

    np.testing.assert_allclose(mg["intercept_se"], disp_se(unit_intercepts),
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mg["coefs_se"])[0], disp_se(unit_A1),
                               rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(mg["orth_irfs_se"]), disp_se(unit_irfs),
                               rtol=1e-8, atol=1e-12)


def test_shapes_and_metadata():
    entities = _simulate_panel()
    k = entities[0].shape[1]
    H = 5
    mg = tsecon.mean_group_var(entities, lags=1, trend="c", horizon=H)
    assert mg["n_entities"] == len(entities)
    assert mg["neqs"] == k
    assert mg["lags"] == 1
    assert len(mg["coefs"]) == 1                       # one lag matrix
    assert np.asarray(mg["coefs"][0]).shape == (k, k)
    assert np.asarray(mg["coefs_se"][0]).shape == (k, k)
    assert len(mg["orth_irfs"]) == H + 1               # horizons 0..H
    assert np.asarray(mg["orth_irfs"][0]).shape == (k, k)
    assert len(mg["intercept"]) == k


def test_irf_path_matches_selected_pair():
    # `mg_irf_path` pulls one (response, impulse) series out of the IRF array.
    entities = _simulate_panel()
    H = 5
    mg = tsecon.mean_group_var(entities, lags=1, trend="c", horizon=H,
                               response=1, impulse=0)
    orth = np.asarray(mg["orth_irfs"])
    orth_se = np.asarray(mg["orth_irfs_se"])
    assert len(mg["irf_path"]) == H + 1
    np.testing.assert_allclose(mg["irf_path"], orth[:, 1, 0],
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(mg["irf_path_se"], orth_se[:, 1, 0],
                               rtol=1e-12, atol=1e-14)


def test_accepts_unequal_time_dimensions():
    # Entities may differ in T_i; only the k variables must match.
    rng = np.random.default_rng(3)
    entities = [rng.standard_normal((150, 2)),
                rng.standard_normal((90, 2)),
                rng.standard_normal((120, 2))]
    mg = tsecon.mean_group_var(entities, lags=1, trend="c", horizon=3)
    assert mg["n_entities"] == 3
    assert mg["neqs"] == 2


def test_requires_at_least_two_entities():
    # The dispersion SE needs N >= 2; a singleton panel must raise.
    rng = np.random.default_rng(1)
    with pytest.raises(Exception):
        tsecon.mean_group_var([rng.standard_normal((100, 2))],
                              lags=1, trend="c", horizon=2)
