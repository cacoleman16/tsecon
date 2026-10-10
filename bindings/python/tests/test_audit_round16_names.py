"""Audit round 16, the house sweep re-run on the fixed wheel: the refusals that
still named no Python keyword, or the wrong one.

Each case is one the sweep (lab/audit/round16/abort_sweep) classified as an
unnamed refusal after the round's fixes: the engine's own wording ("block
length", "r", "slow index", "invalid cointegration rank", "restriction out
of range"), a conformal grid parameter blamed for what the base forecaster's
own setting asks, and a modest horizon blamed for an absurd lag order.
"""

import numpy as np
import pytest

import tsecon


def _var3(t=200, seed=7):
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    y = np.zeros((t, 3))
    e = rng.standard_normal((t, 3))
    for s in range(1, t):
        y[s] = a @ y[s - 1] + e[s]
    return y


X3 = _var3()
SR = [(0, 0, 0, "+"), (1, 0, 0, "+")]


@pytest.mark.parametrize("scheme", ["moving", "circular"])
@pytest.mark.parametrize("value", [0, 101, 2**31])
def test_bootstrap_indices_names_block_length(scheme, value):
    with pytest.raises(ValueError, match=rf"block_length = {value} is outside the valid range 1\.\.=100"):
        tsecon.bootstrap_indices(100, scheme=scheme, seed=1, block_length=value)


@pytest.mark.parametrize("value", [0, 200, 2**31])
def test_proxy_svar_bands_names_block_length(value):
    proxy = X3[:, 0] - 0.5 * np.roll(X3[:, 0], 1) + 0.5 * np.random.default_rng(93).standard_normal(200)
    with pytest.raises(ValueError, match=r"block_length \(the moving-block length\) must satisfy"):
        tsecon.proxy_svar_bands(X3, proxy, lags=2, horizon=6, n_boot=30, block_length=value)


@pytest.mark.parametrize("value", [3, 2**31])
def test_vecm_names_coint_rank(value):
    rng = np.random.default_rng(3)
    common = np.cumsum(rng.standard_normal(200))
    data = np.column_stack([common + rng.standard_normal(200), 0.5 * common + rng.standard_normal(200)])
    with pytest.raises(ValueError, match=rf"coint_rank = {value} is invalid: .* 0 <= coint_rank <= 2"):
        tsecon.vecm(data, k_ar_diff=1, coint_rank=value)


def _panel():
    rng = np.random.default_rng(11)
    f = np.cumsum(rng.standard_normal(80)) * 0.1
    panel = np.column_stack([f * (0.5 + 0.1 * j) + rng.standard_normal(80) for j in range(6)])
    policy = np.zeros(80)
    for s in range(1, 80):
        policy[s] = 0.6 * policy[s - 1] + rng.standard_normal()
    return panel, policy


@pytest.mark.parametrize("value", [0, 7, 2**31])
def test_favar_names_n_factors(value):
    panel, policy = _panel()
    with pytest.raises(ValueError, match=rf"n_factors = {value} is invalid: .* min\(T, N\) = 6"):
        tsecon.favar(panel, policy, n_factors=value, lags=1, horizon=8)


def test_favar_names_the_slow_index_and_the_count():
    panel, policy = _panel()
    with pytest.raises(ValueError, match=r"slow_indices\[1\] = 6 is out of range: .* below N = 6"):
        tsecon.favar(panel, policy, n_factors=1, lags=1, horizon=8, slow_indices=[0, 6])
    with pytest.raises(ValueError, match=r"slow_indices holds 1 series but n_factors = 2"):
        tsecon.favar(panel, policy, n_factors=2, lags=1, horizon=8, slow_indices=[0])
    # The valid slow-fast call is unchanged.
    out = tsecon.favar(panel, policy, n_factors=1, lags=1, horizon=8, slow_indices=[0, 1])
    assert np.all(np.isfinite(np.asarray(out["irf_panel"], dtype=float)))


@pytest.mark.parametrize("value", [0, 1, -1])
def test_gev_fit_names_return_periods(value):
    y = np.abs(np.random.default_rng(5).standard_normal(400))
    with pytest.raises(ValueError, match=rf"invalid return period {value} in return_periods"):
        tsecon.gev_fit(y, block_size=10, return_periods=[10, value])


def test_dsge_solve_names_n_predetermined_and_the_right_remedies():
    a = np.eye(2)
    b = np.array([[0.9, 0.1], [0.2, 1.5]])
    c = np.array([[1.0], [0.0]])
    # One unstable root and two jumps: too many jumps.
    with pytest.raises(ValueError, match=r"indeterminate: 1 unstable eigenvalue\(s\) < 2 jump") as exc:
        tsecon.dsge_solve(a, b, c, n_predetermined=0)
    msg = str(exc.value)
    assert "n_predetermined" in msg and "add a jump" not in msg, msg
    # One unstable root and no jump: too few jumps.
    with pytest.raises(ValueError, match=r"no stable solution: 1 unstable eigenvalue\(s\) > 0 jump") as exc:
        tsecon.dsge_solve(a, b, c, n_predetermined=2)
    msg = str(exc.value)
    assert "n_predetermined" in msg and "remove a jump" not in msg, msg
    assert tsecon.dsge_solve(a, b, c, n_predetermined=1)["verdict"].startswith("unique")


@pytest.mark.parametrize(
    ("restriction", "path"),
    [
        ((5, 0, 0, "+"), r"restrictions\[1\]\[0\] = 5"),
        ((1, 3, 0, "+"), r"restrictions\[1\]\[1\] = 3"),
        ((1, 0, 2**31, "+"), rf"restrictions\[1\]\[2\] = {2**31}"),
    ],
)
@pytest.mark.parametrize("fn", ["sign_restricted_svar", "fry_pagan_svar", "robust_svar_bounds"])
def test_restriction_entries_past_their_bound_are_named(fn, restriction, path):
    spec = [SR[0], restriction]
    kw = {"n_draws": 30} if fn == "robust_svar_bounds" else {"n_draws": 30, "max_tries": 100}
    with pytest.raises(ValueError, match=path + r" is out of range: it must be below"):
        getattr(tsecon, fn)(X3, restrictions=spec, lags=1, horizon=4, **kw)


def test_zero_and_sign_specs_are_named_by_their_own_keyword():
    with pytest.raises(ValueError, match=r"zero_restrictions\[0\]\[0\] = 3 is out of range"):
        tsecon.zero_sign_svar(
            X3, sign_restrictions=SR, zero_restrictions=[(3, 0, 0)], lags=1, horizon=4,
            n_draws=20, max_tries=100,
        )
    with pytest.raises(ValueError, match=r"sign_restrictions\[0\]\[2\] = 9 is out of range"):
        tsecon.zero_sign_svar(
            X3, sign_restrictions=[(0, 0, 9, "+")], zero_restrictions=[(2, 0, 0)], lags=1,
            horizon=4, n_draws=20, max_tries=100,
        )


def test_historical_decomposition_sign_mode_names_the_restriction():
    with pytest.raises(ValueError, match=r"restrictions\[0\]\[0\] = 4 is out of range"):
        tsecon.historical_decomposition(
            X3, restrictions=[(4, 0, 0, "+")], identification="sign", lags=1, n_draws=20,
            max_tries=100,
        )


@pytest.mark.parametrize("method", ["split", "aci"])
def test_conformal_names_the_base_setting_when_no_window_fits(method):
    y = 10.0 + np.cumsum(np.abs(np.random.default_rng(2).standard_normal(200)))
    kw = {"n_eval": 20} if method == "aci" else {}
    with pytest.raises(ValueError, match=r"base=\"theta\" needs 2 \* period") as exc:
        tsecon.conformal_forecast(y, horizon=2, calib=30, period=2**31, method=method, **kw)
    assert "calib = 30 is too large" not in str(exc.value)
    with pytest.raises(ValueError, match=r"needs 2 \* period"):
        tsecon.conformal_backtest(y, horizon=2, calib=30, n_eval=20, period=2**31)


def test_conformal_still_names_calib_when_calib_starves_the_window():
    y = 10.0 + np.cumsum(np.abs(np.random.default_rng(2).standard_normal(200)))
    with pytest.raises(ValueError, match=r"calib = 197 is too large"):
        tsecon.conformal_forecast(y, horizon=2, calib=197)


def test_proxy_ar_sets_blames_an_absurd_lag_order_not_the_horizon():
    proxy = X3[:, 0] - 0.5 * np.roll(X3[:, 0], 1) + 0.5 * np.random.default_rng(93).standard_normal(200)
    with pytest.raises(ValueError, match=r"^lags = 2147483648") as exc:
        tsecon.proxy_ar_sets(X3, proxy, lags=2**31, horizon=6)
    assert not str(exc.value).startswith("horizon"), str(exc.value)


def test_narrative_svar_names_sign_and_narrative_entries():
    with pytest.raises(ValueError, match=r"sign_restrictions\[0\]\[2\] = 9 is out of range"):
        tsecon.narrative_svar(
            X3, sign_restrictions=[(0, 0, 9, "+")], lags=1, horizon=4, n_draws=20, max_tries=100
        )
    narr = [{"type": "shock_sign", "shock": 5, "period": 10, "sign": "+"}]
    with pytest.raises(ValueError, match=r"narrative_restrictions\[0\]\['shock'\] = 5 is out of range"):
        tsecon.narrative_svar(
            X3, sign_restrictions=SR, narrative_restrictions=narr, lags=1, horizon=4, n_draws=20,
            max_tries=200,
        )
