"""The input-coercion layer: pandas / array-likes / off-dtype arrays are
accepted everywhere, while non-data arguments (integer label arrays,
restriction specs, ragged panels, callables, scalars) are left untouched.

The safety-critical property is the second half: a blanket "convert to float64"
would corrupt the danger-zone arguments, so these tests pin that they pass
through byte-for-byte.
"""
import numpy as np
import pytest
import tsecon
from tsecon._coerce import _coerce


def _var_data(n=200, k=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, k))


# --------------------------------------------------------------------------- #
# _coerce unit behaviour
# --------------------------------------------------------------------------- #
def test_coerce_copies_only_when_needed():
    x = np.ascontiguousarray(_var_data(), dtype=np.float64)
    assert _coerce(x) is x  # zero-copy hot path: already float64 + C-contiguous

    f32 = x.astype(np.float32)
    out = _coerce(f32)
    assert out.dtype == np.float64 and out.flags["C_CONTIGUOUS"]
    assert np.allclose(out, f32)

    fortran = np.asfortranarray(x)
    assert _coerce(fortran).flags["C_CONTIGUOUS"]
    sliced = np.arange(20.0)[::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    assert _coerce(sliced).flags["C_CONTIGUOUS"]


def test_coerce_leaves_non_float_arrays_untouched():
    for dt in (np.int64, np.uint32, np.bool_):
        a = np.ones(5, dtype=dt)
        assert _coerce(a) is a, dt  # integer/bool arrays are never coerced


def test_coerce_leaves_specs_ragged_callables_scalars_untouched():
    spec = [(0, 1, 1, "+"), (0, 2, 0, "-")]
    assert _coerce(spec) is spec
    ragged = [np.zeros(3), np.zeros(4)]
    assert _coerce(ragged) is ragged
    fn = lambda p, x: x  # noqa: E731
    assert _coerce(fn) is fn
    for scalar in (5, 0.05, "c", True, None):
        assert _coerce(scalar) is scalar


def test_coerce_handles_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_var_data(), columns=["a", "b", "c"])
    out = _coerce(df)
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    assert out.flags["C_CONTIGUOUS"] and np.allclose(out, df.to_numpy())
    s = pd.Series(np.arange(10.0))
    assert _coerce(s).shape == (10,)


# --------------------------------------------------------------------------- #
# end-to-end: previously-rejected inputs now work
# --------------------------------------------------------------------------- #
def test_dataframe_accepted_end_to_end():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_var_data(), columns=["gdp", "cons", "inv"])
    assert "params" in tsecon.var_fit(df, lags=2)
    assert "p_value" in tsecon.adf(pd.Series(_var_data()[:, 0]))


def test_offdtype_and_noncontiguous_accepted():
    data = _var_data()
    assert "params" in tsecon.var_fit(data.astype(np.float32), lags=2)
    assert "params" in tsecon.var_fit(np.asfortranarray(data), lags=2)
    y = np.random.default_rng(1).standard_normal(400)[::2]  # non-contiguous
    assert "p_value" in tsecon.adf(y)


# --------------------------------------------------------------------------- #
# end-to-end: danger zones remain intact
# --------------------------------------------------------------------------- #
def test_integer_label_arguments_still_work():
    data = _var_data()
    reg = (np.arange(200) >= 100).astype(np.int64)  # integer regime labels
    assert tsecon.hetero_svar(data, reg, lags=1)["identified"] in (True, False)
    g = tsecon.var_granger(data, caused=[0], causing=[1], lags=2)  # integer indices
    assert "statistic" in g


def test_restriction_spec_lists_reach_the_estimator():
    data = _var_data()
    r = tsecon.long_run_svar(data, lags=2, restrictions=[(0, 1), (0, 2), (1, 2)])
    assert "impact" in r


def test_ragged_panel_lists_pass_through():
    rng = np.random.default_rng(3)
    ys = [rng.standard_normal(80), rng.standard_normal(90)]
    xs = [rng.standard_normal((80, 1)), rng.standard_normal((90, 1))]
    assert tsecon.panel_pmg(ys, xs) is not None


def test_integer_data_array_still_raises_documented_gap():
    # type alone can't tell an int DATA array from an int LABEL array, so int
    # arrays are never coerced; an int-typed data matrix still errors (pass
    # .astype(float) or a DataFrame). This pins the honest, safe gap.
    with pytest.raises(TypeError):
        tsecon.var_fit(_var_data().astype(np.int64), lags=2)


# --------------------------------------------------------------------------- #
# the wrapper preserves introspection (IDEs, help(), the stub-sync guard)
# --------------------------------------------------------------------------- #
def test_wrapper_preserves_signature_and_doc():
    import inspect

    assert hasattr(tsecon.var_fit, "__wrapped__")
    assert str(inspect.signature(tsecon.var_fit)) == "(data, lags=2, trend='c')"
    assert tsecon.var_fit.__name__ == "var_fit"
    assert (tsecon.var_fit.__doc__ or "").strip()


def test_outputs_are_unchanged_by_wrapping():
    data = _var_data()
    # var_irf returns a nested list, not a dict — the wrapper touches inputs only
    irf = tsecon.var_irf(data, lags=2, horizon=5)
    assert isinstance(irf, list) and isinstance(irf[0], list)
