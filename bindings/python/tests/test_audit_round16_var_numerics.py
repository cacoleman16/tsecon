"""Regression pins for adversarial audit round 16: VAR numerics.

* SVF-1 — companion-matrix eigenvalues were solved without balancing, so
  `min_root` / `max_root` / `is_stable`, `n_unstable_draws` and the Kilian
  (`bias_correct=True`) bands depended on the units of the series.
* SVF-2 (with the round-15 B15-VAR-1) — a singular residual covariance was
  blamed on a duplicated column: for a sample with fewer residual degrees of
  freedom than series, and for a bootstrap pseudo-sample (no replication
  index either).
* DF16-4 — the symmetric-root Jarque-Bera lost up to five digits on
  covariances graded by very different column scales.
* H16-04 — `multivariate_normality` validated its input three times and had
  no size or work budget (110 s at T = 1500, k = 1000).
* DOC16-5 — its non-finite refusal explained itself with the VAR estimator's
  coefficients.

Goldens: `fixtures/companion_roots.json` and
`fixtures/var_normality_graded.json` (60-digit mpmath). Only numpy is needed.
"""
import ast
import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

import tsecon

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "fixtures"
PYI = REPO / "bindings" / "python" / "python" / "tsecon" / "__init__.pyi"


def _fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _stub_doc(name):
    tree = ast.parse(PYI.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not in the stub")


def _scaled_sample():
    block = _fixture("companion_roots.json")["var_fit_scaled"]
    return np.asarray(block["data"]), np.asarray(block["scale"]), block


def _worst_rel(scaled, unscaled, s):
    """max |a_i / s_i - b| / max |b| for cubes indexed [h][response][shock]."""
    a = np.asarray(scaled, dtype=float) / s[None, :, None]
    b = np.asarray(unscaled, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ------------------------------------------------------------------ SVF-1


def test_var_roots_on_badly_scaled_data_hit_the_mpmath_golden():
    data, _, block = _scaled_sample()
    r = tsecon.var_fit(data, lags=block["lags"])
    assert r["min_root"] == pytest.approx(block["min_root"], rel=1e-9)
    assert r["max_root"] == pytest.approx(block["max_root"], rel=1e-9)
    # The unbalanced solve read 0.997715 here: a stable VAR called unstable.
    assert r["is_stable"] is True and r["min_root"] > 1.0
    lr = tsecon.long_run_svar_bands(data, lags=2, horizon=4, method="asymptotic")
    assert lr["min_root"] == r["min_root"]


def test_roots_do_not_depend_on_the_units_of_the_series():
    data, s, _ = _scaled_sample()
    a = tsecon.var_fit(data, lags=2)
    b = tsecon.var_fit(data / s, lags=2)
    for key in ("min_root", "max_root"):
        assert a[key] == pytest.approx(b[key], rel=1e-12), key
    assert a["is_stable"] == b["is_stable"]


def test_kilian_bands_and_unstable_draws_do_not_depend_on_the_units():
    data, s, _ = _scaled_sample()
    kw = dict(lags=2, horizon=4, n_boot=200, seed=3, bias_correct=True)
    a = tsecon.long_run_svar_bands(data, **kw)
    b = tsecon.long_run_svar_bands(data / s, **kw)
    assert a["n_unstable_draws"] == b["n_unstable_draws"]
    assert a["min_root"] == pytest.approx(b["min_root"], rel=1e-12)
    for key in ("point", "se", "lower", "upper"):
        assert _worst_rel(a[key], b[key], s) <= 1e-9, key

    kw = dict(lags=2, horizon=4, method="bootstrap", n_boot=200, seed=3, bias_correct=True)
    a = tsecon.var_irf_bands(data, **kw)
    b = tsecon.var_irf_bands(data / s, **kw)
    for key in ("point", "se", "lower", "upper"):
        assert _worst_rel(a[key], b[key], s) <= 1e-9, key


# ------------------------------------------------------- SVF-2 / B15-VAR-1


def test_too_few_residual_df_is_refused_by_sample_size():
    # The round-15 repro: 4 independent series, T = 7, lags = 1, no trend:
    # T - m = 6 - 4 = 2 < k = 4, so Sigma_u has rank 2 whatever the data.
    y = np.random.default_rng(0).normal(size=(7, 4))
    for call in (
        lambda: tsecon.var_fit(y, lags=1, trend="n"),
        lambda: tsecon.var_forecast(y, lags=1, steps=2, trend="n"),
    ):
        with pytest.raises(ValueError) as exc:
            call()
        text = str(exc.value)
        assert "Sigma_u, the residual covariance, is singular" in text
        assert "df_resid = 2" in text and "fewer than the k=4 series" in text
        assert "Even lags=1 would need 9 rows" in text
        assert "redundant column" not in text


# The Kilian inner bias bootstrap's stream: seed + BIAS_SEED_OFFSET (mod 2^64).
_BIAS_SEED_OFFSET = 0x9E3779B97F4A7C15


def _singular_replications(seed, n_boot, te, k):
    """Independent replay of the iid bootstrap index stream (replication b:
    Philox seeded by child b of SeedSequence(seed).spawn(n_boot), masked
    rejection on its raw 64-bit output) listing the replications whose
    resample of te residual rows has at most k distinct rows — with an
    intercept the refit residuals then have rank <= k - 1, so the refit's
    Sigma_u is singular by construction."""
    mask = (1 << (te - 1).bit_length()) - 1
    out = []
    for b, child in enumerate(np.random.SeedSequence(seed).spawn(n_boot)):
        bg = np.random.Philox(child)
        idx = set()
        drawn = 0
        while drawn < te:
            v = int(bg.random_raw()) & mask
            if v < te:
                idx.add(v)
                drawn += 1
        if len(idx) <= k:
            out.append(b)
    return out


def test_a_singular_pseudo_sample_is_reported_by_replication_index():
    # T = 6, k = 2, lags = 1: T_eff - m = 2 = k, so the full sample fits, but
    # about one resample in ten draws at most 2 distinct residual rows. Such a
    # draw is refused by counting the rows, so the replication named is the
    # first one of the index stream, whatever rounding does to a Cholesky
    # factorization (which lets most of these singular covariances through).
    y = np.random.default_rng(1).standard_normal((6, 2))
    tsecon.var_fit(y, lags=1)
    tsecon.long_run_svar(y, lags=1)
    tsecon.long_run_svar_bands(y, lags=1, method="asymptotic")
    for seed in range(3):
        first = _singular_replications(seed, 50, 5, 2)[0]
        for call in (
            lambda: tsecon.long_run_svar_bands(y, lags=1, horizon=3, n_boot=50, seed=seed),
            lambda: tsecon.var_irf_bands(
                y, lags=1, horizon=3, method="bootstrap", n_boot=50, seed=seed
            ),
        ):
            with pytest.raises(ValueError) as exc:
                call()
            text = str(exc.value)
            assert f"bootstrap replication {first} (0-based" in text, (seed, text)
            assert "too few distinct residual rows" in text, text
            assert "Sigma_u is singular" in text and "redundant column" not in text
        # Reduced-form responses never read Sigma_u: every draw counts.
        r = tsecon.var_irf_bands(
            y, lags=1, horizon=3, orth=False, method="bootstrap", n_boot=50, seed=seed
        )
        assert np.all(np.isfinite(r["se"]))


def test_the_kilian_inner_bootstrap_no_longer_dies_on_a_singular_pseudo_sample():
    # T = 7, k = 2, lags = 1 at seed 14: no outer resample is singular, so
    # plain Efron passes, but three inner bias-bootstrap resamples are — and
    # bias_correct=True died there, though the inner bootstrap only needs the
    # slopes.
    rng = np.random.default_rng(1)
    rng.standard_normal((6, 2))
    y = rng.standard_normal((7, 2))
    seed = 14
    assert _singular_replications(seed, 50, 6, 2) == []
    assert _singular_replications(seed + _BIAS_SEED_OFFSET, 50, 6, 2) == [34, 44, 45]
    kw = dict(lags=1, horizon=3, n_boot=50, seed=seed)
    tsecon.long_run_svar_bands(y, **kw)
    r = tsecon.long_run_svar_bands(y, bias_correct=True, **kw)
    assert np.all(np.isfinite(r["se"]))


def test_the_docs_no_longer_call_a_failed_refit_measure_zero():
    doc = _stub_doc("long_run_svar_bands")
    assert "pseudo-sample cannot be refitted" in doc
    card = (REPO / "docs/reference/model-cards/structural-identification.md").read_text(
        encoding="utf-8"
    )
    assert "This is a measure-zero event" not in card


# ------------------------------------------------------------------ DF16-4


def test_symmetric_root_jarque_bera_on_graded_covariances_hits_the_mpmath_golden():
    fx = _fixture("var_normality_graded.json")
    keys = (
        "jarque_bera_symmetric",
        "jarque_bera_symmetric_pvalue",
        "skewness_symmetric",
        "skewness_symmetric_pvalue",
        "kurtosis_symmetric",
        "kurtosis_symmetric_pvalue",
    )
    assert len(fx["cases"]) == 44
    for i, case in enumerate(fx["cases"]):
        x = np.asarray(fx["panels"][str(case["k"])]) * np.asarray(case["scales"])
        r = tsecon.multivariate_normality(x)
        for key in keys:
            assert r[key] == pytest.approx(case[key], rel=1e-10), (i, key)
        for key in ("skewness_symmetric_components", "kurtosis_symmetric_components"):
            e = np.asarray(case[key])
            assert np.max(np.abs(np.asarray(r[key]) - e)) <= 1e-10 * np.max(np.abs(e)), (i, key)


# ------------------------------------------------------------------ H16-04


def test_multivariate_normality_refuses_past_its_work_budget_before_any_work():
    x = np.zeros((1500, 1000))  # took 98 s before the budget
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as exc:
        tsecon.multivariate_normality(x)
    assert time.perf_counter() - t0 < 1.0
    text = str(exc.value)
    assert text.startswith("resid is T = 1500 rows by k = 1000 columns"), text
    assert re.search(r"at T = 1500 rows pass at most k = \d+ columns", text), text
    assert "2^34 work units" in text


def test_multivariate_normality_refuses_past_its_memory_budget_before_copying():
    # A zero-stride view, handed to the compiled function directly (the Python
    # input layer scans arrays): the refusal must come from the shape alone,
    # before the binding copies the 100 million rows.
    x = np.broadcast_to(np.float64(0.0), (100_000_000,))
    t0 = time.perf_counter()
    with pytest.raises(ValueError) as exc:
        tsecon._core.multivariate_normality(x)
    assert time.perf_counter() - t0 < 1.0
    text = str(exc.value)
    assert "at k = 1 columns at most T = 89478483 rows" in text, text
    assert "2^28 values" in text


def test_the_budget_refusal_names_only_admissible_shapes():
    # Every admissible shape also needs T >= 8 and T > k: past the work budget
    # at k = 900 the rows the budget alone would allow (641) are fewer than k.
    core = tsecon._core.multivariate_normality
    with pytest.raises(ValueError) as exc:
        core(np.broadcast_to(np.float64(0.0), (901, 900)))
    text = str(exc.value)
    assert "at T = 901 rows pass at most k = 881 columns" in text, text
    assert "at k = 900 columns no row count fits" in text, text
    with pytest.raises(ValueError) as exc:
        core(np.broadcast_to(np.float64(0.0), (7, 10**6)))
    assert "T = 7 rows is below the 8 the test needs" in str(exc.value), str(exc.value)


def test_the_normality_cost_and_budget_are_documented():
    doc = _stub_doc("multivariate_normality")
    assert "5 T k^2 + 20 k^3" in doc and "2^34" in doc
    assert "5 T k^2 + 20 k^3" in tsecon.multivariate_normality.__doc__


# ------------------------------------------------------------------ DOC16-5


def test_the_non_finite_refusal_explains_the_normality_test_not_the_var():
    x = np.random.default_rng(0).standard_normal((30, 2))
    x[3, 1] = np.nan
    with pytest.raises(ValueError) as exc:
        tsecon.multivariate_normality(x)
    text = str(exc.value)
    assert "resid" in text and "row 3, column 1" in text and "normality statistic" in text
    assert "VAR estimator" not in text and "coefficient" not in text
    with pytest.raises(ValueError, match="VAR estimator has no missing-value handling"):
        tsecon.var_fit(x, lags=1)
