"""Regression pins for audit round 16: the size budgets of the VAR family.

Every horizon, step, replication and simulation count of the VAR-family
callables is checked against one 2^28-value (2 GiB) working-set budget,
counted as real storage, before anything it sizes is allocated, and refused
naming the Python keyword with the largest value that fits. Each refusal
pinned here used to abort the interpreter (a single allocation of 120 GB or
more), panic in a worker thread and hang, or name the wrong parameter; they
now return in milliseconds, so they are tested in-process. Also pinned:
`var_irf_bands` refuses the inert keywords `long_run_svar_bands` refuses,
`long_run_svar` names `lags` and `restrictions`, and `normalize="long_run"`
signs each shock's long-run pivot (not the diagonal) under a permuted
pattern. The measurement scripts are under `lab/audit/round16/var_budgets/`.
"""

import ast
import re
import time
from pathlib import Path

import numpy as np
import pytest

import tsecon

REPO = Path(__file__).resolve().parents[3]
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"

HUGE = [2**31, 2**40]


def _var(n=200, k=2, seed=1):
    rng = np.random.default_rng(seed)
    a = 0.5 * np.eye(k) + 0.1 * np.ones((k, k)) / k
    y = np.zeros((n, k))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + rng.standard_normal(k)
    return y


X2 = _var()
X3 = _var(240, 3, 2)


def _refused(call, name):
    """The call is refused, fast, naming `name` with the largest value that
    fits; returns that value (None when the message says none fits)."""
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as info:
        call()
    assert time.perf_counter() - t0 < 5.0
    msg = str(info.value)
    assert re.search(rf"\b{name}\b", msg), msg
    m = re.search(rf"the largest {name} that fits (?:at the other arguments )?is (\d+)", msg)
    if m:
        return int(m.group(1))
    assert f"no {name} fits" in msg, msg
    return None


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


# ------------------------------------------------- A1 / H16-01: horizon budget
HORIZON_CALLS = {
    "var_irf": lambda h: tsecon.var_irf(X2, horizon=h),
    "var_fevd": lambda h: tsecon.var_fevd(X2, horizon=h),
    "var_irf_bands": lambda h: tsecon.var_irf_bands(X2, horizon=h),
    "connectedness": lambda h: tsecon.connectedness(X3, horizon=h),
    "long_run_svar": lambda h: tsecon.long_run_svar(X2, horizon=h),
    "long_run_svar_signs": lambda h: tsecon.long_run_svar(X2, horizon=h, impact_signs=[(0, 1)]),
    "mean_group_var": lambda h: tsecon.mean_group_var(
        [_var(120, 2, 10 + i) for i in range(4)], horizon=h
    ),
    "fvar_scenario": lambda h: tsecon.fvar_scenario(
        np.random.default_rng(7).standard_normal(150),
        np.random.default_rng(8).standard_normal((150, 12)).cumsum(axis=1),
        np.linspace(0.1, 0.3, 12),
        n_factors=2,
        lags=1,
        horizon=h,
    ),
}


@pytest.mark.parametrize("h", HUGE)
@pytest.mark.parametrize("name", sorted(HORIZON_CALLS))
def test_an_absurd_horizon_is_refused_by_name_before_allocating(name, h):
    largest = _refused(lambda: HORIZON_CALLS[name](h), "horizon")
    assert largest is not None and 0 < largest < h


def test_the_horizon_budget_reports_its_boundary():
    # Two variables: 8 copies of a 2 x 2 matrix in faer and Python-list form
    # (84 doubles) per horizon against 2^28 doubles.
    assert _refused(lambda: tsecon.var_irf(X2, horizon=2**31), "horizon") == 399456
    assert _refused(lambda: tsecon.var_irf(X2, horizon=399457), "horizon") == 399456
    # var_fevd counts `horizon` matrices, var_irf `horizon + 1`.
    assert _refused(lambda: tsecon.var_fevd(X2, horizon=2**31), "horizon") == 399457
    # A valid horizon is untouched.
    assert len(tsecon.var_irf(X2, horizon=12)) == 13


def test_the_forecast_paths_name_steps():
    assert _refused(lambda: tsecon.var_forecast(X2, steps=2**31), "steps") == 399457
    with pytest.raises(ValueError) as info:
        tsecon.var_forecast(X2, steps=2**40)
    assert str(info.value).startswith("steps:") and "got horizon" not in str(info.value)


# ------------------------------------------- H16-02: bootstrap working set
def test_var_irf_bands_bootstrap_budgets_n_boot_and_horizon():
    n = _refused(lambda: tsecon.var_irf_bands(X2, method="bootstrap", n_boot=10**7), "n_boot")
    assert n == 2886323
    for h in HUGE:
        _refused(
            lambda: tsecon.var_irf_bands(X2, method="bootstrap", n_boot=50, horizon=h), "horizon"
        )
    msg = ""
    try:
        tsecon.var_irf_bands(X2, method="bootstrap", n_boot=10**7)
    except ValueError as e:
        msg = str(e)
    assert msg.startswith("n_boot and horizon:"), msg


# --------------------------------- H16-03: long_run_svar_bands' true count
def test_long_run_svar_bands_counts_the_store_it_really_uses():
    # The count per replication is the store slot (responses, long-run matrix,
    # radius) plus a fixed 48-value overhead: at horizon 0 on two variables,
    # 2 * 4 + 1 + 48 + 1 = 58 doubles, so n_boot = 4e6 (2.32e8 doubles) is
    # admitted and 4.7e6 is refused naming n_boot.
    assert _refused(
        lambda: tsecon.long_run_svar_bands(X2, n_boot=10**7, horizon=0), "n_boot"
    ) == 4628185
    assert _refused(lambda: tsecon.long_run_svar_bands(X2, n_boot=10**7), "n_boot") == 2532327


def test_kept_draws_are_the_reported_object():
    # The draws now come from one flat store; they are still the replications
    # whose percentiles the bands are, cumulated iff cumulative.
    r = tsecon.long_run_svar_bands(X2, horizon=6, n_boot=99, seed=3, return_draws=True)
    draws = np.asarray(r["irf_draws"])
    assert draws.shape == (99, 7, 2, 2)
    assert np.asarray(r["long_run_draws"]).shape == (99, 2, 2)
    np.testing.assert_allclose(
        np.percentile(draws, 5.0, axis=0), np.asarray(r["lower"]), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.percentile(np.asarray(r["long_run_draws"]), 95.0, axis=0),
        np.asarray(r["long_run_upper"]),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.std(draws, axis=0, ddof=1), np.asarray(r["se"]), rtol=1e-10, atol=1e-14
    )


# --------------------- H16-13 / B6 / DOC16-3: the term that is over is named
def test_an_oversized_band_n_sim_leads_its_own_refusal():
    for kw in ({}, {"horizon": 6, "lags": 1}):
        msg = ""
        try:
            tsecon.long_run_svar_bands(
                X2, method="asymptotic", band="sup-t", band_n_sim=2**31, **kw
            )
        except ValueError as e:
            msg = str(e)
        assert msg.startswith("band_n_sim:"), msg
        assert "horizon and band_scope" not in msg, msg
    largest = _refused(
        lambda: tsecon.long_run_svar_bands(X2, method="asymptotic", band="sup-t", band_n_sim=2**31),
        "band_n_sim",
    )
    assert largest == 19173084
    # The largest count it names is admitted by the budget; a slightly smaller
    # one runs (a full run at the boundary would take minutes).
    r = tsecon.long_run_svar_bands(
        X2, method="asymptotic", band="sup-t", band_n_sim=20_000, horizon=4
    )
    assert "critical_value" in r


def test_a_pointwise_delta_band_is_not_told_to_change_what_it_does_not_read():
    msg = ""
    try:
        tsecon.long_run_svar_bands(X2, lags=2, horizon=10**6, method="asymptotic")
    except ValueError as e:
        msg = str(e)
    assert msg.startswith("horizon and lags:"), msg
    assert "band_scope" not in msg and "band_n_sim" not in msg, msg
    assert _refused(
        lambda: tsecon.long_run_svar_bands(X2, lags=2, horizon=10**6, method="asymptotic"),
        "horizon",
    ) == 4095
    # A family's covariance names band_scope.
    msg = ""
    try:
        tsecon.long_run_svar_bands(
            X2, lags=2, horizon=10**5, method="asymptotic", band="sidak", band_scope="all"
        )
    except ValueError as e:
        msg = str(e)
    assert msg.startswith("horizon and band_scope:"), msg


# --------------- pre-existing: var_irf_bands' quadratic delta-method work
def test_var_irf_bands_caps_the_quadratic_jacobian_work():
    y8 = _var(300, 2, 3)
    assert _refused(lambda: tsecon.var_irf_bands(y8, lags=8, horizon=10**4), "horizon") == 2047
    msg = ""
    try:
        tsecon.var_irf_bands(y8, lags=8, horizon=10**4)
    except ValueError as e:
        msg = str(e)
    assert msg.startswith("horizon:") and "QUADRATIC" in msg, msg
    assert "band_scope" not in msg and "band_n_sim" not in msg, msg


# ------------------------------------------------- A5: band_n_sim budget
BAND_N_SIM_CALLS = {
    "lp": lambda n: tsecon.lp(
        np.random.default_rng(1).standard_normal(300),
        np.random.default_rng(2).standard_normal(300),
        band="sup-t",
        band_n_sim=n,
    ),
    "smooth_lp": lambda n: tsecon.smooth_lp(
        np.random.default_rng(1).standard_normal(300),
        np.random.default_rng(2).standard_normal(300),
        band="sup-t",
        band_n_sim=n,
    ),
    "var_forecast": lambda n: tsecon.var_forecast(X2, band="sup-t", band_n_sim=n),
    "var_irf_bands": lambda n: tsecon.var_irf_bands(X2, lags=1, horizon=6, band="sup-t", band_n_sim=n),
}


@pytest.mark.parametrize("n", HUGE)
@pytest.mark.parametrize("name", sorted(BAND_N_SIM_CALLS))
def test_an_absurd_band_n_sim_is_refused_by_name_before_allocating(name, n):
    largest = _refused(lambda: BAND_N_SIM_CALLS[name](n), "band_n_sim")
    assert largest is not None and 2 <= largest < n


@pytest.mark.parametrize("name", sorted(BAND_N_SIM_CALLS))
def test_band_n_sim_below_two_names_band_n_sim(name):
    with pytest.raises(ValueError, match=r"band_n_sim must be at least 2"):
        BAND_N_SIM_CALLS[name](0)


def test_lp_band_n_sim_budget_is_the_shared_sup_t_budget():
    # One family of K = 13 cells: (K + 1) * band_n_sim <= 2^28 binds first.
    assert _refused(lambda: BAND_N_SIM_CALLS["lp"](2**31), "band_n_sim") == 2**28 // 14


def test_var_forecast_budgets_its_joint_covariance_naming_steps():
    largest = _refused(lambda: tsecon.var_forecast(X2, lags=2, steps=10**5, band="sidak"), "steps")
    assert largest == 685


# ----------------------------------------- API16-1: inert keywords refused
INERT = [
    ({"method": "asymptotic", "n_boot": 5000}, "n_boot"),
    ({"method": "asymptotic", "seed": 7}, "seed"),
    ({"method": "asymptotic", "band_scope": "all"}, "band_scope"),
    ({"method": "asymptotic", "band": "sidak", "band_seed": 1}, "band_seed"),
    ({"method": "asymptotic", "band": "bonferroni", "band_n_sim": 5000}, "band_n_sim"),
    ({"method": "bootstrap", "n_boot": 40, "band_scope": "shock"}, "band_scope"),
    ({"method": "bootstrap", "n_boot": 40, "band_seed": 1}, "band_seed"),
    ({"method": "bootstrap", "n_boot": 40, "band": "sup-t", "band_n_sim": 5000}, "band_n_sim"),
]


@pytest.mark.parametrize("kw, name", INERT, ids=[n for _, n in INERT])
def test_var_irf_bands_refuses_inert_keywords_like_its_sibling(kw, name):
    with pytest.raises(ValueError, match=rf"^{name} is not read here and would be silently ignored"):
        tsecon.var_irf_bands(X2, lags=1, horizon=4, **kw)
    lrsb = dict(kw)
    with pytest.raises(ValueError, match=rf"^{name} is not read here"):
        tsecon.long_run_svar_bands(X2, lags=1, horizon=4, **lrsb)


def test_defaults_and_the_results_facade_are_untouched():
    base = tsecon.var_irf_bands(X2, lags=1, horizon=4)
    again = tsecon.var_irf_bands(
        X2, lags=1, horizon=4, n_boot=1000, seed=0, band_seed=20260807, band_n_sim=100000
    )
    assert np.array_equal(np.asarray(base["se"]), np.asarray(again["se"]))
    from tsecon.results import VARResults

    fit = VARResults.fit(X2, lags=1)
    bands = fit.irf_bands(horizon=4, method="asymptotic")
    assert np.array_equal(np.asarray(bands["se"]), np.asarray(base["se"]))
    boot = fit.irf_bands(horizon=4, method="bootstrap", n_boot=50, seed=3, band="sup-t")
    assert "sim_lower" in boot


# ------------------------------------- H16-08 / API16-3 / known: long_run_svar
def test_long_run_svar_names_lags_and_restrictions():
    for call in (
        lambda: tsecon.long_run_svar(X2, lags=0),
        lambda: tsecon.long_run_svar(X2, lags=0, impact_signs=[(0, 1)]),
        lambda: tsecon.long_run_svar_bands(X2, lags=0),
    ):
        with pytest.raises(ValueError, match=r"^lags must be at least 1") as info:
            call()
        assert "coefs" not in str(info.value)
    for bad in ([(0,)], [(0, 1, 1)], [(-1, 0)], [(2**64, 0)]):
        with pytest.raises(ValueError, match=r"\brestrictions\b"):
            tsecon.long_run_svar(X2, restrictions=bad)


def test_long_run_normalization_signs_each_shocks_pivot():
    # A valid column permutation of the recursive pattern: shock 0 has two
    # long-run zeros, shock 1 none, shock 2 one. The positive entries are the
    # pivots LR[m_s, s], m_s = the number of zeros in shock s's column (the
    # Cholesky diagonal), not the diagonal, which includes an imposed zero.
    y = np.random.default_rng(3).standard_normal((300, 3))
    zeros = [(0, 0), (1, 0), (0, 2)]
    lr = np.asarray(tsecon.long_run_svar(y, lags=1, restrictions=zeros)["long_run"])
    m = {s: sum(1 for (_, c) in zeros if c == s) for s in range(3)}
    assert all(lr[m[s], s] > 0 for s in range(3)), lr
    assert lr[0, 0] == 0.0 and lr[1, 0] == 0.0 and lr[0, 2] == 0.0
    assert not (np.diag(lr) > 0).all()
    # Under the default recursive pattern the pivots are the diagonal.
    lr_rec = np.asarray(tsecon.long_run_svar(y, lags=1)["long_run"])
    assert (np.diag(lr_rec) > 0).all()
    for fn in (tsecon.long_run_svar, tsecon.long_run_svar_bands):
        with pytest.raises(ValueError, match=r"pivot") as info:
            fn(y, normalize="bogus")
        assert "positive long-run diagonal" not in str(info.value)


def test_help_and_stub_agree_on_the_impact_signs_range():
    doc = tsecon.long_run_svar.__doc__ or ""
    assert "0..k-1" in doc and "outside `0..k`" not in doc
    assert "0..k-1" in _stub_doc("long_run_svar")
    for name in ("long_run_svar", "long_run_svar_bands"):
        assert "positive LR diagonal" not in _stub_doc(name)
        assert "pivot" in _stub_doc(name) + (getattr(tsecon, name).__doc__ or "")


# ------------------------------------------------------ docs name the budgets
@pytest.mark.parametrize(
    "name",
    [
        "var_irf",
        "var_fevd",
        "var_irf_bands",
        "connectedness",
        "fvar_scenario",
        "mean_group_var",
        "long_run_svar",
        "long_run_svar_bands",
        "var_forecast",
        "lp",
        "smooth_lp",
    ],
)
def test_the_budget_is_stated_on_both_doc_surfaces(name):
    runtime = getattr(tsecon, name).__doc__ or ""
    stub = _stub_doc(name)
    for doc in (runtime, stub):
        assert "2^28" in doc, name
        assert "largest" in doc, name
