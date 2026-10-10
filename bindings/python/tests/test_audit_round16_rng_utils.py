"""Regression pins for audit round 16: the RNG-spawn and utility share.

* A4 (central part): the replication driver used to materialize one RNG
  stream per replication before any check, so ``n_boot = 2**31`` aborted the
  interpreter on a 154 GB allocation. Streams are now derived lazily (every
  seeded output unchanged — pinned below against the pre-fix build), and the
  three fixed-regressor bootstrap tests refuse ``n_boot`` above 10**6 by name.
* A6: ``star`` ``n_gamma``/``n_c`` and ``threshold_vecm`` ``n_grid_beta`` grid
  budgets.
* A7: ``bootstrap_indices(n)``, ``philox_uniforms(n)``, ``midas_weights(k)``
  and ``acm_term_premium(maturities)`` output-length budgets.
* B2: ``markov_switching_ar`` refusals name ``k_regimes`` / ``order``.
* Seeds: every seed-like keyword refuses a value outside ``[0, 2**64 - 1]``
  by name with the range (it used to raise a bare ``OverflowError``).

Every refusal here fires before anything is allocated, so the cells run
in-process. No test depends on a platform-specific bit pattern: the pins use
NumPy ``default_rng`` data and a relative tolerance.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import tsecon
from tsecon import _coerce

REPO = Path(__file__).resolve().parents[3]
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"
SEED_MAX = 2**64 - 1


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _setar_y(n=150, seed=20261016):
    e = np.random.default_rng(seed).standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = (0.6 if y[t - 1] <= 0 else -0.3) * y[t - 1] + e[t]
    return y


def _var2(n=150, seed=20261017):
    e = np.random.default_rng(seed).standard_normal((n, 2))
    a = np.array([[0.5, 0.1], [0.2, 0.3]])
    y = np.zeros((n, 2))
    for t in range(1, n):
        y[t] = a @ y[t - 1] + e[t]
    return y


def _coint2(n=150, seed=20261018):
    r = np.random.default_rng(seed)
    x = np.cumsum(r.standard_normal(n))
    return np.column_stack([1.5 * x + r.standard_normal(n), x])


def _yields(n=60, m=8, seed=20261019):
    r = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    mats = np.arange(1, m + 1)[None, :]
    return 0.03 + 0.001 * mats + 1e-4 * np.sin(t * m + mats) + 1e-5 * r.standard_normal((n, m))


# ----------------------------------------------------------------- A4: n_boot
_BOOT_TESTS = {
    "setar_test": lambda n_boot: tsecon.setar_test(_setar_y(), 1, n_boot=n_boot, seed=20261016),
    "threshold_var_test": lambda n_boot: tsecon.threshold_var_test(
        _var2(), 1, n_grid=40, n_boot=n_boot, seed=20261016
    ),
    "hansen_seo_test": lambda n_boot: tsecon.hansen_seo_test(
        _coint2(), n_grid=40, n_boot=n_boot, seed=20261016
    ),
}

# Printed by exactly these calls (n_boot = 7, seed = 20261016) against the
# PRE-fix build — PR #30 head fdd90d3, whose driver materialized one stream
# per replication; the lazily derived streams must reproduce them.
_PINS = {
    "setar_test": (
        6.388702133761198,
        0.5,
        [1.5563343890078825, 4.947327766459517, 7.476492048617228, 4.45221466174056,
         11.63190210507755, 5.943870720018226, 7.506072973359188],
    ),
    "threshold_var_test": (
        10.750049137911748,
        0.75,
        [15.431593789967813, 16.46418509528628, 14.487922686702666, 13.560729951134382,
         8.215662771679137, 8.638149766825732, 16.07819104598124],
    ),
    "hansen_seo_test": (
        13.808923152038822,
        0.25,
        [11.691460142880812, 9.760155577308128, 16.918906668167594, 11.609104406404771,
         12.788787398330708, 11.712973871001182, 11.356634647981062],
    ),
}


@pytest.mark.parametrize("name", sorted(_BOOT_TESTS))
def test_lazily_derived_replication_streams_keep_every_seeded_output(name):
    stat, p_value, boot = _PINS[name]
    r = _BOOT_TESTS[name](7)
    np.testing.assert_allclose(r["stat"], stat, rtol=1e-9)
    assert r["p_value"] == p_value
    np.testing.assert_allclose(r["boot_stats"], boot, rtol=1e-9)
    # Replication i depends on (seed, i) only: a longer run extends the
    # shorter one instead of reshuffling it.
    longer = _BOOT_TESTS[name](23)
    np.testing.assert_array_equal(np.asarray(longer["boot_stats"])[:7], r["boot_stats"])


@pytest.mark.parametrize("name", sorted(_BOOT_TESTS))
@pytest.mark.parametrize("n_boot", [10**6 + 1, 2**31, 2**40])
def test_bootstrap_tests_refuse_n_boot_beyond_the_cap_by_name(name, n_boot):
    # 2**31 used to abort the interpreter allocating 154 GB of RNG streams;
    # 2**40 raised a message about the "RNG substream spawn limit".
    with pytest.raises(ValueError, match=r"n_boot") as exc:
        _BOOT_TESTS[name](n_boot)
    msg = str(exc.value)
    assert "1000000" in msg and "spawn" not in msg, msg
    assert "n_boot" in _stub_doc(name) and "1000000" in _stub_doc(name)


# ------------------------------------------------------------- A6: grid sizes
@pytest.mark.parametrize(
    "n_gamma, n_c, needle",
    [
        (2**31, 5, "at n_c = 5 the largest admissible n_gamma is 209715"),
        (5, 2**31, "at n_gamma = 5 the largest admissible n_c is 209715"),
        (2**40, 5, "at n_c = 5 the largest admissible n_gamma is 209715"),
        (1025, 1024, "at n_c = 1024 the largest admissible n_gamma is 1024"),
    ],
)
def test_star_refuses_a_grid_beyond_the_cell_budget(n_gamma, n_c, needle):
    with pytest.raises(ValueError) as exc:
        tsecon.star(_setar_y(300), 1, n_gamma=n_gamma, n_c=n_c)
    msg = str(exc.value)
    assert f"n_gamma = {n_gamma} and n_c = {n_c}" in msg and "1048576" in msg, msg
    assert needle in msg, msg


def test_star_default_grid_is_unchanged():
    r = tsecon.star(_setar_y(300), 1)
    assert np.asarray(r["ssr_grid"]).shape == (25, 25)


@pytest.mark.parametrize("n_grid_beta", [52429, 2**31, 2**40])
def test_threshold_vecm_refuses_a_beta_grid_beyond_the_cell_budget(n_grid_beta):
    data = _coint2(200)
    with pytest.raises(ValueError) as exc:
        tsecon.threshold_vecm(data, n_grid_gamma=20, n_grid_beta=n_grid_beta)
    msg = str(exc.value)
    assert f"n_grid_beta = {n_grid_beta}" in msg, msg
    assert "at n_grid_gamma = 20 the largest admissible n_grid_beta is 52428" in msg, msg


# ------------------------------------------------------- A7: output lengths
@pytest.mark.parametrize("n", [2**28 + 1, 2**31, 2**40])
@pytest.mark.parametrize(
    "kw",
    [
        {"scheme": "iid"},
        {"p": 0.1},
        {"scheme": "moving", "block_length": 5},
        {"scheme": "circular", "block_length": 5},
    ],
)
def test_bootstrap_indices_refuses_n_beyond_the_budget(n, kw):
    with pytest.raises(ValueError, match=rf"n = {n} exceeds .*n = 268435456"):
        tsecon.bootstrap_indices(n, seed=0, **kw)


@pytest.mark.parametrize("n", [2**28 + 1, 2**31, 2**40])
def test_philox_uniforms_refuses_n_beyond_the_budget(n):
    with pytest.raises(ValueError, match=rf"n = {n} exceeds the largest admissible n = 268435456"):
        tsecon.philox_uniforms(1, n)


def test_utility_outputs_are_unchanged_inside_the_budget():
    # Same NumPy references as the existing goldens.
    u = tsecon.philox_uniforms(20261016, 5)
    np.testing.assert_array_equal(u, np.random.Generator(np.random.Philox(20261016)).random(5))
    assert tsecon.bootstrap_indices(50, scheme="iid", seed=3).shape == (50,)


@pytest.mark.parametrize("scheme, t1, t2", [("exp_almon", 0.1, -0.01), ("beta", 1.0, 5.0)])
@pytest.mark.parametrize("k", [2**27 + 1, 2**31, 2**40])
def test_midas_weights_refuses_k_beyond_the_budget(scheme, t1, t2, k):
    with pytest.raises(ValueError, match=rf"k = {k} high-frequency lags exceeds the largest admissible k = 134217728"):
        tsecon.midas_weights(scheme, t1, t2, k)


@pytest.mark.parametrize("huge", [120_001, 2**31, 2**40])
def test_acm_term_premium_applies_the_jsz_maturity_cap(huge):
    mats = list(range(1, 8)) + [huge]
    with pytest.raises(ValueError, match=rf"maturities\[7\] = {huge} periods exceeds the supported maximum of 120000"):
        tsecon.acm_term_premium(_yields(), mats, n_factors=2)


def test_acm_term_premium_cap_itself_is_admissible():
    mats = list(range(1, 8)) + [120_000]
    r = tsecon.acm_term_premium(_yields(), mats, n_factors=2)
    assert list(r["maturities"]) == mats


# ------------------------------------------------- B2: markov k_regimes / order
@pytest.mark.parametrize(
    "k_regimes, order, needles",
    [
        (0, 1, ["k_regimes = 0", "k_regimes >= 2"]),
        (1, 1, ["k_regimes = 1", "k_regimes >= 2"]),
        (65, 1, ["k_regimes = 65 and order = 1", "largest admissible k_regimes is 64"]),
        (3, 20, ["k_regimes = 3 and order = 20", "at k_regimes = 3 the largest admissible order is 6"]),
        (2**31, 1, ["k_regimes = 2147483648 and order = 1", "k_regimes itself must be at most 4096"]),
    ],
)
def test_markov_switching_ar_refusals_name_k_regimes_and_order(k_regimes, order, needles):
    with pytest.raises(ValueError) as exc:
        tsecon.markov_switching_ar(_setar_y(), k_regimes=k_regimes, order=order, max_iter=5)
    msg = str(exc.value)
    assert all(n in msg for n in needles), msg
    assert "k^(order+1)" not in msg and "invalid model specification" not in msg, msg


# --------------------------------------------------------------------- seeds
def _seed_keywords():
    """(function, keyword) for every seed-like keyword in the stub."""
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                if _coerce._is_seed_name(arg.arg):
                    out.append((node.name, arg.arg))
    return out


SEED_KEYWORDS = _seed_keywords()


def test_the_seed_rule_covers_every_seed_keyword_and_no_count():
    names = {kw for _, kw in SEED_KEYWORDS}
    assert {"seed", "band_seed", "rf_seed"} <= names
    assert len(SEED_KEYWORDS) >= 30
    assert not _coerce._is_seed_name("n_seeds")  # mlp_regression's count


@pytest.mark.parametrize("fn, kw", SEED_KEYWORDS)
@pytest.mark.parametrize("bad", [2**64, 2**70, -1])
def test_every_seed_keyword_refuses_out_of_range_values_by_name(fn, kw, bad):
    # Refused by the coercion layer before the call reaches the compiled
    # function, so no data is needed; the error used to be a bare
    # OverflowError ("int too big to convert") naming nothing.
    with pytest.raises(ValueError) as exc:
        getattr(tsecon, fn)(**{kw: bad})
    msg = str(exc.value)
    assert msg.startswith(f"{fn}: {kw}={bad} is outside the valid range"), msg
    assert f"0 <= {kw} <= 2**64 - 1 = {SEED_MAX}" in msg, msg


def test_seed_range_refusal_names_positional_and_numpy_seeds():
    with pytest.raises(ValueError, match=r"philox_uniforms: seed=18446744073709551616 is outside"):
        tsecon.philox_uniforms(2**64, 3)
    with pytest.raises(ValueError, match=r"philox_uniforms: seed=-1 is outside"):
        tsecon.philox_uniforms(np.int64(-1), 3)
    with pytest.raises(ValueError, match=r"seed=-1 and band_seed=18446744073709551616 are outside"):
        tsecon.var_irf_bands(_var2(), seed=-1, band_seed=2**64)


def test_every_in_range_seed_is_still_accepted():
    for s in (0, 2**63, SEED_MAX, np.uint64(SEED_MAX)):
        assert tsecon.philox_uniforms(s, 3).shape == (3,)
    r = tsecon.setar_test(_setar_y(), 1, n_boot=5, seed=SEED_MAX)
    assert r["n_boot"] == 5
    doc = _stub_doc("bootstrap_indices")
    assert "0 <= seed <= 2**64 - 1" in doc and "268435456" in doc
