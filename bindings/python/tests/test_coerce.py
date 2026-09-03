"""The input-coercion layer: pandas / array-likes / off-dtype arrays / integer
arrays are accepted everywhere, while non-data arguments (integer label and
index parameters, restriction specs, ragged panels, callables, scalars) are
left untouched.

Coercion is *parameter-aware*: an integer array is data everywhere except in
the four audited label/index parameters (``hetero_svar.regime_labels``,
``var_granger.caused``/``causing``, ``favar.slow_indices``). The
safety-critical property is the second half — a blanket "convert to float64"
would corrupt those danger-zone arguments, so these tests pin that they pass
through byte-for-byte, positionally *and* by keyword, as lists *and* as numpy
integer arrays.
"""
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tsecon
from tsecon import _core
from tsecon._coerce import _EXEMPT, _coerce, _coerce_data, _exempt_positions, _wrap

FIXTURES = Path(__file__).parents[3] / "fixtures"


def _var_data(n=200, k=3, seed=0):
    return np.random.default_rng(seed).standard_normal((n, k))


# --------------------------------------------------------------------------- #
# _coerce / _coerce_data unit behaviour
# --------------------------------------------------------------------------- #
def test_coerce_copies_only_when_needed():
    x = np.ascontiguousarray(_var_data(), dtype=np.float64)
    assert _coerce(x) is x  # zero-copy hot path: already float64 + C-contiguous
    assert _coerce_data(x) is x

    f32 = x.astype(np.float32)
    out = _coerce(f32)
    assert out.dtype == np.float64 and out.flags["C_CONTIGUOUS"]
    assert np.allclose(out, f32)

    fortran = np.asfortranarray(x)
    assert _coerce(fortran).flags["C_CONTIGUOUS"]
    sliced = np.arange(20.0)[::2]
    assert not sliced.flags["C_CONTIGUOUS"]
    assert _coerce(sliced).flags["C_CONTIGUOUS"]
    assert _coerce_data(sliced).flags["C_CONTIGUOUS"]


def test_coerce_conservative_leaves_non_float_arrays_untouched():
    # _coerce is the fallback rule used when a signature cannot be resolved.
    for dt in (np.int64, np.uint32, np.bool_):
        a = np.ones(5, dtype=dt)
        assert _coerce(a) is a, dt


def test_coerce_data_converts_integer_and_bool_arrays():
    for dt in (np.int8, np.int32, np.int64, np.uint32, np.uint64, np.bool_):
        out = _coerce_data(np.ones(5, dtype=dt))
        assert out.dtype == np.float64 and out.flags["C_CONTIGUOUS"], dt
        assert np.allclose(out, 1.0)
    # values survive the conversion, not just the dtype
    src = np.array([[1, -2], [3, -4]], dtype=np.int64)
    np.testing.assert_array_equal(_coerce_data(src), src.astype(np.float64))
    assert src.dtype == np.int64  # the caller's array is not mutated in place


def test_coerce_data_leaves_zero_d_integer_arrays_alone():
    # `lags=np.array(2)` is a scalar in array clothing: the boundary accepts it
    # through __index__, and float64-ifying it would break that.
    scalar_arr = np.array(2)
    assert _coerce_data(scalar_arr) is scalar_arr
    assert "params" in tsecon.var_fit(_var_data(), lags=np.array(2))
    assert "params" in tsecon.var_fit(_var_data(), lags=np.int64(2))  # numpy scalar


def test_coerce_data_leaves_complex_object_datetime_alone():
    # converting these would drop information or raise something less clear
    # than the boundary error they already produce.
    for a in (
        np.ones(3, dtype=np.complex128),
        np.array([object(), object()], dtype=object),
        np.arange("2020-01", "2020-04", dtype="datetime64[M]"),
    ):
        assert _coerce_data(a) is a, a.dtype


def test_coerce_leaves_specs_callables_scalars_untouched():
    for fn in (_coerce, _coerce_data):
        spec = [(0, 1, 1, "+"), (0, 2, 0, "-")]
        assert fn(spec) is spec
        callable_ = lambda p, x: x  # noqa: E731
        assert fn(callable_) is callable_
        for scalar in (5, 0.05, "c", True, None):
            assert fn(scalar) is scalar


def test_ragged_lists_conservative_vs_data_rule():
    ragged = [np.zeros(3), np.zeros(4)]
    # the conservative rule (used when a signature can't be resolved) is
    # identity; the data rule keeps the list but coerces its elements.
    assert _coerce(ragged) is ragged
    out = _coerce_data(ragged)
    assert isinstance(out, list) and len(out) == 2
    assert all(e.dtype == np.float64 for e in out)


def test_coerce_handles_pandas():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_var_data(), columns=["a", "b", "c"])
    for fn in (_coerce, _coerce_data):
        out = fn(df)
        assert isinstance(out, np.ndarray) and out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"] and np.allclose(out, df.to_numpy())
        assert fn(pd.Series(np.arange(10.0))).shape == (10,)


# --------------------------------------------------------------------------- #
# the parameter-aware mechanism
# --------------------------------------------------------------------------- #
def test_exempt_positions_resolve_to_the_documented_slots():
    # hetero_svar(data, regime_labels, ...) -> position 1
    assert _exempt_positions(_core.hetero_svar, _EXEMPT["hetero_svar"]) == {1}
    # var_granger(data, caused, causing, ...) -> positions 1 and 2
    assert _exempt_positions(_core.var_granger, _EXEMPT["var_granger"]) == {1, 2}
    # favar(panel, policy, n_factors, lags, trend, slow_indices, ...) -> 5
    assert _exempt_positions(_core.favar, _EXEMPT["favar"]) == {5}


def test_every_exempt_name_exists_in_its_compiled_signature():
    for fn_name, params in _EXEMPT.items():
        fn = getattr(_core, fn_name)
        sig = set(inspect.signature(fn).parameters)
        assert params <= sig, (fn_name, params - sig)


def test_every_compiled_function_exposes_a_signature():
    # the mechanism degrades safely without one, but silent degradation would
    # re-open the papercut, so pin that PyO3 keeps emitting __text_signature__.
    builtin = type(_core.var_fit)
    checked = 0
    for name in dir(_core):
        if name.startswith("_"):
            continue
        fn = getattr(_core, name)
        if isinstance(fn, builtin):
            inspect.signature(fn)  # raises if __text_signature__ is missing
            checked += 1
    assert checked > 100, f"only {checked} compiled functions found"


def test_signature_drift_falls_back_to_conservative_behaviour():
    # a stale exempt set (parameter renamed away) must NOT be guessed around
    assert _exempt_positions(_core.var_granger, frozenset({"renamed_param"})) is None
    assert _exempt_positions(_core.var_granger, frozenset({"caused", "gone"})) is None

    # and a callable with no resolvable signature falls back too
    assert _exempt_positions(min, frozenset({"anything"})) is None

    # the fallback wrapper leaves integer arrays untouched (the old rule)
    seen = {}

    def fake_var_granger(data, caused, causing, lags=2, trend="c"):
        seen["data"] = data
        return "ok"

    fake_var_granger.__name__ = "var_granger"
    # force the fallback by hiding the signature behind a *args wrapper
    def varargs(*args, **kwargs):
        return fake_var_granger(*args, **kwargs)

    varargs.__name__ = "var_granger"
    assert _exempt_positions(varargs, _EXEMPT["var_granger"]) is None
    wrapped = _wrap(varargs)
    ints = np.ones((5, 2), dtype=np.int64)
    wrapped(ints, [0], [1])
    assert seen["data"] is ints  # conservative: not converted


def test_exempt_set_covers_every_integer_rust_parameter():
    """Re-derive the exempt set from the Rust source so it cannot go stale.

    Any new ``Vec<usize>`` / ``Vec<i64>`` parameter added to a binding — in
    ``lib.rs`` or in any per-slice ``src/*.rs`` module — must be added to
    ``_EXEMPT`` as well, otherwise the wrapper would float64-ify a label
    array. Skipped when running against an installed wheel.
    """
    src_dir = Path(__file__).parents[1] / "src"
    rs_files = sorted(src_dir.glob("*.rs")) if src_dir.exists() else []
    if not rs_files:  # pragma: no cover - source-tree only
        pytest.skip("Rust source not available")
    # Every binding file, not just lib.rs: the per-slice modules
    # (`ml_structured.rs`, ...) register their own pyfunctions.
    src = []
    for rs in rs_files:
        src.extend(rs.read_text(encoding="utf-8").splitlines())

    fn_re = re.compile(r"^\s*(?:pub\s+)?fn\s+([a-z_0-9]+)\s*[<(]")
    int_param_re = re.compile(
        r"^\s+([a-z_0-9]+)\s*:\s*(?:Option<)?Vec<\s*"
        r"(?:usize|isize|i8|i16|i32|i64|u8|u16|u32|u64)\s*>"
    )
    found: dict[str, set[str]] = {}
    current = None
    for line in src:
        m = fn_re.match(line)
        if m:
            current = m.group(1)
            continue
        m = int_param_re.match(line)
        if m and current is not None:
            found.setdefault(current, set()).add(m.group(1))

    # the audit found something (guards against the regex silently matching 0)
    assert found, "integer-parameter scan matched nothing - regex is stale"
    expected = {k: set(v) for k, v in _EXEMPT.items()}
    assert found == expected, (
        "integer parameters in src/*.rs disagree with _coerce._EXEMPT: "
        f"rust={found} exempt={expected}"
    )


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


def test_integer_data_matrix_accepted_end_to_end():
    """The headline first-run papercut: data read as int64 now just works."""
    data = (_var_data() * 100).astype(np.int64)
    res = tsecon.var_fit(data, lags=2)
    assert "params" in res
    # identical to explicitly converting, i.e. the values were not mangled
    ref = tsecon.var_fit(data.astype(np.float64), lags=2)
    np.testing.assert_allclose(np.array(res["params"]), np.array(ref["params"]))


def test_integer_series_accepted_end_to_end():
    rng = np.random.default_rng(2)
    y = (rng.standard_normal(300).cumsum() * 50).astype(np.int64)  # int64 levels
    assert "p_value" in tsecon.adf(y)
    # a narrow integer dtype (e.g. counts read from CSV as int32) too
    assert "p_value" in tsecon.adf(rng.integers(0, 500, size=300).astype(np.int32))


def test_zero_one_integer_dummy_accepted_by_recession_probit():
    rng = np.random.default_rng(4)
    n = 400
    x = rng.standard_normal((n, 1))
    y = (x[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)  # 0/1 int
    res = tsecon.recession_probit(y, x)
    assert "coefficients" in res or "params" in res
    ref = tsecon.recession_probit(y.astype(np.float64), x)
    assert res.keys() == ref.keys()
    # a boolean mask is the other common spelling of the same dummy
    assert tsecon.recession_probit(y.astype(bool), x).keys() == ref.keys()


def test_integer_arrays_convert_for_keyword_arguments_too():
    # the coercion is applied to **kwargs as well as *args
    data = (_var_data() * 100).astype(np.int64)
    assert "params" in tsecon.var_fit(data=data, lags=2)


# --------------------------------------------------------------------------- #
# end-to-end: danger zones remain intact
# --------------------------------------------------------------------------- #
def test_hetero_svar_regime_labels_stay_integer():
    data = _var_data()
    reg = (np.arange(200) >= 100).astype(np.int64)  # numpy integer label array
    assert tsecon.hetero_svar(data, reg, lags=1)["identified"] in (True, False)
    # by keyword, and as a plain Python list
    assert tsecon.hetero_svar(data, regime_labels=reg, lags=1)["identified"] in (True, False)
    assert tsecon.hetero_svar(data, reg.tolist(), lags=1)["identified"] in (True, False)
    # other integer dtypes are equally untouched
    assert tsecon.hetero_svar(data, reg.astype(np.int32), lags=1)["identified"] in (True, False)
    # and the caller's array is handed through unmodified
    assert reg.dtype == np.int64


def test_exempt_parameters_pass_pandas_through_untouched():
    # Exempt parameters are handed to the boundary verbatim — including pandas
    # objects, which PyO3 extracts as integer sequences directly. (The old
    # blanket rule float64-ified them and broke the call.)
    pd = pytest.importorskip("pandas")
    data = _var_data()
    reg = pd.Series((np.arange(200) >= 100).astype(np.int64))
    assert tsecon.hetero_svar(data, reg, lags=1)["identified"] in (True, False)
    g = tsecon.var_granger(data, caused=pd.Series([0]), causing=pd.Series([1]), lags=2)
    assert "statistic" in g


def test_var_granger_index_arrays_stay_integer():
    data = _var_data()
    for caused, causing in (
        ([0], [1]),  # lists
        (np.array([0]), np.array([1])),  # numpy integer index arrays
        (np.array([0], dtype=np.uint32), np.array([1], dtype=np.uint32)),
        (np.array([0, 2]), np.array([1])),
    ):
        g = tsecon.var_granger(data, caused=caused, causing=causing, lags=2)
        assert "statistic" in g and np.isfinite(g["statistic"])
    # positionally, too: var_granger(data, caused, causing, ...)
    g = tsecon.var_granger(data, np.array([0]), np.array([1]), 2)
    assert "statistic" in g


def test_favar_slow_indices_stay_integer():
    fixture = json.loads((FIXTURES / "favar.json").read_text(encoding="utf-8"))
    panel = np.array(fixture["X_standardized"]).T
    policy = np.random.default_rng(0).standard_normal(panel.shape[0])
    slow = np.arange(0, panel.shape[1] // 2)  # numpy integer index array
    res = tsecon.favar(panel, policy, n_factors=2, lags=2, slow_indices=slow)
    assert res["n_factors"] == 2
    assert tsecon.favar(panel, policy, 2, 2, "c", slow.tolist())["n_factors"] == 2
    # positional slot 5 is the exempt one
    assert tsecon.favar(panel, policy, 2, 2, "c", slow)["n_factors"] == 2


def test_integer_data_still_converts_in_the_same_call_as_integer_labels():
    """The per-parameter split really is per-parameter, not per-call.

    Each of the three functions gets an integer DATA argument (must convert)
    and an integer LABEL/INDEX argument (must not) in the *same* call.
    """
    data = (_var_data() * 100).astype(np.int64)
    ref_data = data.astype(np.float64)

    reg = (np.arange(200) >= 100).astype(np.int64)
    res = tsecon.hetero_svar(data, reg, lags=1)
    ref = tsecon.hetero_svar(ref_data, reg, lags=1)
    assert res["identified"] in (True, False)
    np.testing.assert_allclose(np.array(res["B"]), np.array(ref["B"]))
    np.testing.assert_array_equal(res["regime_sizes"], ref["regime_sizes"])

    res = tsecon.var_granger(data, np.array([0]), np.array([1]), lags=2)
    ref = tsecon.var_granger(ref_data, np.array([0]), np.array([1]), lags=2)
    assert res["statistic"] == pytest.approx(ref["statistic"])

    fixture = json.loads((FIXTURES / "favar.json").read_text(encoding="utf-8"))
    panel = (np.array(fixture["X_standardized"]).T * 100).astype(np.int64)
    policy = (np.random.default_rng(0).standard_normal(panel.shape[0]) * 100).astype(np.int64)
    slow = np.arange(0, panel.shape[1] // 2)
    res = tsecon.favar(panel, policy, n_factors=2, lags=2, slow_indices=slow)
    ref = tsecon.favar(
        panel.astype(np.float64), policy.astype(np.float64),
        n_factors=2, lags=2, slow_indices=slow,
    )
    np.testing.assert_allclose(np.array(res["factors"]), np.array(ref["factors"]))


def test_restriction_spec_lists_reach_the_estimator():
    data = _var_data()
    r = tsecon.long_run_svar(data, lags=2, restrictions=[(0, 1), (0, 2), (1, 2)])
    assert "impact" in r


def test_ragged_panel_lists_pass_through():
    rng = np.random.default_rng(3)
    ys = [rng.standard_normal(80), rng.standard_normal(90)]
    xs = [rng.standard_normal((80, 1)), rng.standard_normal((90, 1))]
    assert tsecon.panel_pmg(ys, xs) is not None


# --------------------------------------------------------------------------- #
# the wrapper preserves introspection (IDEs, help(), the stub-sync guard)
# --------------------------------------------------------------------------- #
def test_wrapper_preserves_signature_and_doc():
    assert hasattr(tsecon.var_fit, "__wrapped__")
    assert str(inspect.signature(tsecon.var_fit)) == "(data, lags=2, trend='c')"
    assert tsecon.var_fit.__name__ == "var_fit"
    assert (tsecon.var_fit.__doc__ or "").strip()
    # the exempt-parameter functions keep their introspection as well
    assert tsecon.var_granger.__name__ == "var_granger"
    assert "caused" in str(inspect.signature(tsecon.var_granger))
    assert (tsecon.hetero_svar.__doc__ or "").strip()


def test_outputs_are_unchanged_by_wrapping():
    data = _var_data()
    # var_irf returns a nested list, not a dict — the wrapper touches inputs only
    irf = tsecon.var_irf(data, lags=2, horizon=5)
    assert isinstance(irf, list) and isinstance(irf[0], list)


# --------------------------------------------------------------------------- #
# sequences: flat numeric data converts, specs and nested literals do not
# --------------------------------------------------------------------------- #
def test_flat_numeric_sequences_are_accepted_as_data():
    y = np.random.default_rng(0).standard_normal(200)
    assert "p_value" in tsecon.adf([float(v) for v in y])    # list of floats
    assert "p_value" in tsecon.adf([int(v * 10) for v in y])  # list of ints


def test_tuple_options_are_not_treated_as_data():
    # a tuple is how this API spells a fixed-arity option; converting one to an
    # array breaks the binding (regression: weighted_midas weight_start).
    opt = (2.0, 3.0)
    assert _coerce_data(opt) is opt
    y = np.random.default_rng(0).standard_normal(300)
    assert "lambda" in tsecon.box_cox_lambda(np.abs(y) + 1.0, bounds=(-2.0, 2.0))


def test_restriction_specs_are_never_converted():
    # `[(0, 1), (0, 2)]` is all-numeric and rectangular, exactly like a matrix.
    # Converting it would corrupt the restriction silently, so nested sequences
    # are left alone -- this is the regression guard for that.
    spec = [(0, 1), (0, 2), (1, 2)]
    assert _coerce_data(spec) is spec
    data = _var_data()
    assert "impact" in tsecon.long_run_svar(data, lags=2, restrictions=spec)
    sign_spec = [(0, 0, 0, "+")]
    assert _coerce_data(sign_spec) is sign_spec


def test_ragged_panel_elements_are_coerced_but_the_list_is_kept():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(3)
    ys = [pd.Series(rng.standard_normal(80)), rng.standard_normal(90).astype(np.float32)]
    xs = [rng.standard_normal((80, 1)).astype(np.float32), rng.standard_normal((90, 1))]
    out = _coerce_data(ys)
    assert isinstance(out, list)                       # container preserved
    assert all(isinstance(e, np.ndarray) and e.dtype == np.float64 for e in out)
    assert tsecon.panel_pmg(ys, xs) is not None        # end to end


def test_wrong_rank_raises_a_teaching_error():
    pd = pytest.importorskip("pandas")
    y = np.random.default_rng(0).standard_normal(200)
    with pytest.raises(TypeError) as excinfo:
        tsecon.var_fit(pd.DataFrame({"gdp": y})["gdp"], lags=2)
    msg = str(excinfo.value)
    assert "wrong shape or type" in msg
    assert "(200,)" in msg          # reports the actual shape
    assert "df[['x']]" in msg       # tells the user what to do


# --------------------------------------------------------------------------- #
# negative integer arguments (audit round 6 residue, fixed round 7)
# --------------------------------------------------------------------------- #
# PyO3 used to surface `lags=-1` and friends as a raw, unattributed
# ``OverflowError: can't convert negative int to unsigned`` library-wide. The
# coercion layer now rebuilds exactly that conversion error into the library's
# teaching ValueError, naming the function and the offending parameter. A
# representative sample across crates; the mechanism is central (`_call`), so
# one wrapper covers every compiled function.


def _negative_cases():
    y = np.random.default_rng(0).standard_normal(220)
    data = np.column_stack([y, np.roll(y, 1)])
    return [
        (lambda: tsecon.var_fit(data, lags=-2), "lags=-2"),
        (lambda: tsecon.stl(y, 12, outer_iter=-1), "outer_iter=-1"),
        (lambda: tsecon.garch_fit(y, p=-1), "p=-1"),
        (lambda: tsecon.lp(y, y, horizons=-4), "horizons=-4"),
        # numpy integers count as integers
        (lambda: tsecon.var_fit(data, lags=np.int64(-3)), "lags=-3"),
    ]


def test_negative_integer_arguments_raise_teaching_valueerrors():
    for call, expected_fragment in _negative_cases():
        with pytest.raises(ValueError) as excinfo:
            call()
        msg = str(excinfo.value)
        assert expected_fragment in msg, msg          # names the parameter
        assert "nonnegative integer" in msg           # states the contract
        # the original PyO3 error is chained, not swallowed
        assert isinstance(excinfo.value.__cause__, OverflowError)


def test_negative_integer_error_names_the_function():
    with pytest.raises(ValueError, match=r"^garch_fit: "):
        tsecon.garch_fit(np.random.default_rng(1).standard_normal(300), q=-1)


def test_negative_seasonal_order_keeps_its_bespoke_teaching_error():
    # arima_fit validates the (P, D, Q, s) tuple itself with a specific
    # message; the central wrapper must not shadow a better error that the
    # boundary already raises.
    y = np.random.default_rng(4).standard_normal(120)
    with pytest.raises(ValueError, match="non-negative"):
        tsecon.arima_fit(y, seasonal=(0, -1, 0, 4))


def test_negative_float_parameters_are_untouched():
    # A parameter that legitimately accepts negative values (f64) must not
    # trip the negative-int guard, even when passed as a Python int.
    y = np.cumsum(np.random.default_rng(2).standard_normal(180))
    data = np.column_stack([y, 0.5 * y + np.random.default_rng(3).standard_normal(180)])
    out = tsecon.bvar_fit(data, lags=1, delta=-1)
    assert "log_marginal_likelihood" in out
