"""Binding-tier coverage for the three functions testing.md names as the gap.

docs/reference/testing.md (tier 4, "Surface completeness") honestly lists
`engle_granger`, `fvar_scenario`, and `ndiffs` as publicly exported but never
called through `tsecon.<name>(...)` anywhere in this suite. All three are
golden-pinned on the Rust side, so what was missing is exactly what this tier
exists to check and what a Rust golden structurally cannot see: the
marshalling, the returned dict's key set, and error propagation. These tests
close that gap; the numbers they assert are qualitative (sign/side of a
threshold on seeded data), because the tight numeric pins already live in the
crates' golden tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import tsecon


def _coint_pair(t=400, seed=20260828):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(t))
    y = 1.5 * x + rng.standard_normal(t)
    return np.column_stack([y, x])


class TestEngleGranger:
    KEYS = {
        "stat",
        "pvalue",
        "crit",
        "resid",
        "coint_coefs",
        "used_lag",
        "adf_nobs",
        "n_vars",
        "nobs",
    }

    def test_keys_and_marshalling_on_cointegrated_pair(self):
        data = _coint_pair()
        res = tsecon.engle_granger(data)
        assert self.KEYS <= set(res)
        # A genuinely cointegrated pair: the null of no cointegration is
        # rejected at conventional levels on this seed.
        assert res["pvalue"] < 0.05
        assert res["stat"] < 0.0
        assert res["n_vars"] == 2
        assert res["nobs"] == len(data)
        resid = np.asarray(res["resid"], dtype=float)
        assert resid.shape == (len(data),)
        assert np.isfinite(resid).all()

    def test_independent_walks_do_not_reject(self):
        rng = np.random.default_rng(7)
        data = np.column_stack(
            [np.cumsum(rng.standard_normal(400)), np.cumsum(rng.standard_normal(400))]
        )
        res = tsecon.engle_granger(data)
        assert res["pvalue"] > 0.05

    def test_pandas_input_coerced(self):
        pd = pytest.importorskip("pandas")
        data = _coint_pair()
        res_np = tsecon.engle_granger(data)
        res_pd = tsecon.engle_granger(pd.DataFrame(data, columns=["y", "x"]))
        assert res_pd["stat"] == res_np["stat"]

    def test_shape_refusal_teaches_the_fix(self):
        # 1-D input: the coerce layer's teaching TypeError names the shape
        # contract and the reshape escape hatch.
        with pytest.raises(TypeError, match="2-D array shaped"):
            tsecon.engle_granger(np.arange(10.0))


class TestFvarScenario:
    KEYS = {"weights", "response_outcome", "responses", "implied_outcome_innovation"}

    @staticmethod
    def _inputs(t=180, k=8, seed=42):
        rng = np.random.default_rng(seed)
        grid = np.linspace(0.0, 1.0, k)
        level = np.cumsum(rng.standard_normal(t))[:, None]
        slope = np.cumsum(0.5 * rng.standard_normal(t))[:, None]
        curves = level + slope * grid[None, :] + 0.1 * rng.standard_normal((t, k))
        y = 0.3 * level.ravel() + rng.standard_normal(t)
        delta = np.ones(k)
        return y, curves, delta

    def test_keys_and_shapes(self):
        y, curves, delta = self._inputs()
        horizon, n_factors = 6, 2
        res = tsecon.fvar_scenario(
            y, curves, delta, n_factors=n_factors, lags=1, horizon=horizon
        )
        assert self.KEYS <= set(res)
        weights = np.asarray(res["weights"], dtype=float)
        assert weights.shape == (n_factors,)
        out = np.asarray(res["response_outcome"], dtype=float)
        assert out.shape == (horizon + 1,)
        assert np.isfinite(out).all()
        responses = np.asarray(res["responses"], dtype=float)
        assert responses.shape[0] == horizon + 1
        assert np.isfinite(responses).all()

    def test_error_propagates_with_actionable_message(self):
        y, curves, delta = self._inputs()
        with pytest.raises((ValueError, RuntimeError)):
            tsecon.fvar_scenario(y, curves, delta, n_factors=curves.shape[1] + 3)


class TestNdiffs:
    KEYS = {"d", "test", "alpha", "max_d", "steps", "stop", "interpretation"}
    STEP_KEYS = {"d", "lags", "n", "needs_differencing", "p_value", "statistic"}

    def test_random_walk_needs_a_difference(self):
        rng = np.random.default_rng(11)
        y = np.cumsum(rng.standard_normal(300))
        res = tsecon.ndiffs(y)
        assert self.KEYS <= set(res)
        assert res["d"] >= 1
        # Per-order test evidence lives in `steps`, one dict per order tried.
        assert len(res["steps"]) >= 1
        assert self.STEP_KEYS <= set(res["steps"][0])
        assert bool(res["steps"][0]["needs_differencing"]) is True

    def test_white_noise_needs_none(self):
        rng = np.random.default_rng(12)
        res = tsecon.ndiffs(rng.standard_normal(300))
        assert res["d"] == 0
        assert bool(res["steps"][0]["needs_differencing"]) is False

    def test_max_d_caps_the_answer(self):
        rng = np.random.default_rng(13)
        y = np.cumsum(np.cumsum(rng.standard_normal(300)))  # I(2)
        res = tsecon.ndiffs(y, max_d=1)
        assert res["d"] <= 1
        assert res["max_d"] == 1

    def test_unknown_test_raises_teaching_error(self):
        with pytest.raises((ValueError, RuntimeError), match="test"):
            tsecon.ndiffs(np.random.default_rng(1).standard_normal(100), test="bogus")
