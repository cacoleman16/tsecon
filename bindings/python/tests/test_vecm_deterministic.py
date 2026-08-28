"""Field item 12 (correctness trap): vecm's deterministic cases vs johansen —
and the restricted-cases follow-up that finishes the item.

The original defect: ``vecm`` silently fit the no-deterministic case
(statsmodels ``deterministic="n"``) while ``johansen`` documents and assumes
the unrestricted constant (``coint_johansen`` det_order=0) — a caller reading
the two against each other on drifting log levels got cointegrating vectors a
cosine of ~0.57 apart with no warning. The 0.6.0 fix: ``vecm`` accepts
``deterministic="n"|"co"`` (default "n", unchanged), both docstrings name
their case and cross-reference each other, and ``deterministic="co"``
reconciles ``vecm`` with ``johansen`` exactly. 0.6.0 refused the restricted
statsmodels cases with a teaching error naming a follow-up; this slice ships
them: ``"ci"``/``"li"`` (constant/trend INSIDE the cointegration relation —
the reduced-rank step widens the cointegrating matrix, and the extra rows
come back as ``det_coef_coint``, statsmodels' own split), ``"lo"``, the four
combinations, and centered seasonal dummies (``seasons=``/``first_season=``).

Goldens: ``fixtures/vecm_deterministic.json`` — statsmodels VECM under "n"
and "co" + coint_johansen(det_order=0) on seeded drifting data (dataset 1),
every deterministic case + coint_johansen(det_order=1) on seeded trending
data (the ``trending`` block), and two seasons=4 fits on a seeded quarterly
pair (the ``seasonal`` block).
"""
import json
from pathlib import Path

import numpy as np
import pytest
import tsecon

FIX = Path(__file__).parents[3] / "fixtures"
VD = json.loads((FIX / "vecm_deterministic.json").read_text())
# data is stored series-major (k lists of length T); transpose to T x k.
DATA = np.array(VD["data"]).T
K_AR_DIFF = VD["k_ar_diff"]
RANK = VD["coint_rank"]


def cosine(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_vecm_default_is_deterministic_n():
    """The default (no argument) is exactly deterministic="n" — the
    documented statsmodels case — so existing callers see no change."""
    r_default = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK)
    r_n = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="n")
    np.testing.assert_array_equal(r_default["beta"], r_n["beta"])
    np.testing.assert_array_equal(r_default["alpha"], r_n["alpha"])
    assert r_default["llf"] == r_n["llf"]
    # And it matches the statsmodels deterministic="n" golden.
    fx = VD["vecm_n"]
    np.testing.assert_allclose(r_default["alpha"], fx["alpha"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["beta"], fx["beta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["gamma"], fx["gamma"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r_default["sigma_u"], fx["sigma_u"], rtol=1e-6, atol=1e-10)
    assert r_default["llf"] == pytest.approx(fx["llf"], rel=1e-6)
    # No deterministic terms -> det_coef has zero columns.
    assert np.asarray(r_default["det_coef"]).size == 0


def test_vecm_co_matches_statsmodels():
    """deterministic="co" (unrestricted constant) matches statsmodels
    VECM(..., deterministic="co") — alpha, beta, gamma, det_coef, sigma_u,
    llf — at the tolerance the existing vecm goldens use."""
    r = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="co")
    fx = VD["vecm_co"]
    np.testing.assert_allclose(r["alpha"], fx["alpha"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["beta"], fx["beta"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["gamma"], fx["gamma"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["det_coef"], fx["det_coef"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(r["sigma_u"], fx["sigma_u"], rtol=1e-6, atol=1e-10)
    assert r["llf"] == pytest.approx(fx["llf"], rel=1e-6)
    assert np.asarray(r["det_coef"]).shape == (3, 1)


def test_vecm_co_reconciles_with_johansen_and_n_diverges():
    """The reporter's scenario as a regression artifact: on seeded drifting
    log levels, vecm(deterministic="co") spans exactly the cointegrating
    space johansen (det_order=0, unrestricted constant) tests — cosine ~1 —
    while the deterministic="n" default diverges (cosine ~0.63 on this
    draw; the reporter measured ~0.57 on theirs). The divergence itself is
    pinned against the statsmodels-computed fixture value, so this is
    documented behavior, not an accident."""
    joh = tsecon.johansen(DATA, k_ar_diff=K_AR_DIFF)
    r_co = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK, deterministic="co")
    r_n = tsecon.vecm(DATA, k_ar_diff=K_AR_DIFF, coint_rank=RANK)

    # johansen's first eigenvector, normalized as the VECM normalizes beta.
    evec = np.asarray(joh["evec"])
    beta_joh = evec[:, 0] / evec[0, 0]
    beta_co = np.asarray(r_co["beta"])[:, 0]
    beta_n = np.asarray(r_n["beta"])[:, 0]

    # Matching cases reconcile: cosine ~1.
    assert abs(cosine(beta_co, beta_joh)) > 1 - 1e-10
    # Mismatched cases diverge, exactly as the fixture pins.
    cos_n_co = cosine(beta_n, beta_co)
    assert cos_n_co == pytest.approx(VD["beta_cosine_n_co"], rel=1e-6)
    assert cos_n_co < 0.8  # visibly different cointegrating vectors

    # And johansen's eigenvalues are the ones the "co" reduced-rank step
    # maximizes over (same eigenproblem), pinned against statsmodels.
    np.testing.assert_allclose(joh["eig"], VD["johansen"]["eig"], atol=1e-8)


def test_johansen_evec_matches_statsmodels_up_to_sign():
    """The newly exposed evec matches statsmodels coint_johansen's (both
    S_11-orthonormal; eigensolvers pick column signs arbitrarily)."""
    joh = tsecon.johansen(DATA, k_ar_diff=K_AR_DIFF)
    evec = np.asarray(joh["evec"])
    evec_fx = np.array(VD["johansen"]["evec"])
    assert evec.shape == evec_fx.shape
    for j in range(evec.shape[1]):
        sign = 1.0 if evec[:, j] @ evec_fx[:, j] >= 0 else -1.0
        np.testing.assert_allclose(sign * evec[:, j], evec_fx[:, j], rtol=1e-6, atol=1e-8)


def test_vecm_unknown_deterministic_rejected():
    """Invalid deterministic strings are refused with a teaching error naming
    the nine statsmodels cases. (0.6.0 refused "ci"/"colo" here as
    not-yet-implemented; they are now supported — see the trending goldens —
    so the refusal surface is the genuinely invalid strings, including the
    statsmodels conflicts of the same term on both sides.)"""
    for bad in ("coci", "lico", "nc", "", "c", "seasonal", "cocolo"):
        with pytest.raises(ValueError, match="unknown deterministic"):
            tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, deterministic=bad)
    # The refusal teaches: names every case family and the johansen pairing.
    with pytest.raises(ValueError, match=r'"cili"'):
        tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, deterministic="bogus")
    with pytest.raises(ValueError, match="johansen"):
        tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, deterministic="bogus")


def test_vecm_seasons_one_rejected():
    """seasons=1 (a one-period "cycle" with zero dummy columns) is refused
    with a teaching error; seasons=0 means none."""
    with pytest.raises(ValueError, match="seasons"):
        tsecon.vecm(DATA, k_ar_diff=1, coint_rank=1, seasons=1)


def test_docstrings_name_the_deterministic_cases():
    """The docstring floor: vecm names its default case and johansen's
    convention; johansen names its constant and points at "co"."""
    vdoc = tsecon.vecm.__doc__
    jdoc = tsecon.johansen.__doc__
    assert '"n"' in vdoc and '"co"' in vdoc
    assert "deterministic" in vdoc
    assert "johansen" in vdoc  # cross-reference
    assert "det_order=0" in jdoc or "det_order = 0" in jdoc
    assert "unrestricted constant" in jdoc.lower()
    assert 'deterministic="co"' in jdoc  # cross-reference back to vecm
    # The restricted-cases follow-up: every case named, the restricted
    # split documented, and the Johansen det_order correspondence taught.
    for case in ('"ci"', '"lo"', '"li"', '"colo"', '"coli"', '"cilo"', '"cili"'):
        assert case in vdoc, f"vecm docstring must name {case}"
    for key in ("det_coef_coint", "det_coef", "seasons", "first_season"):
        assert key in vdoc, f"vecm docstring must name {key}"
    assert "det_order" in vdoc


# ---------------------------------------------------------------------------
# The restricted cases (0.7.0 follow-up): trending-data goldens for every
# statsmodels deterministic string, seasonal goldens, and the dict surface.

TR = VD["trending"]
TDATA = np.array(TR["data"]).T
ALL_CASES = ["n", "co", "ci", "lo", "li", "colo", "coli", "cilo", "cili"]


def _assert_block_close(r, fx, label):
    """Every pinned estimate at the 1e-6 golden tolerance (empty blocks
    must be empty on both sides)."""
    for key in ("alpha", "beta", "det_coef_coint", "gamma", "det_coef", "sigma_u"):
        got = np.asarray(r[key], float)
        want = np.asarray(fx[key], float)
        if want.size == 0:
            assert got.size == 0, f"{label}:{key} expected empty, got {got!r}"
            continue
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8, err_msg=f"{label}:{key}")
    assert r["llf"] == pytest.approx(fx["llf"], rel=1e-6), f"{label}:llf"


@pytest.mark.parametrize("case", ALL_CASES)
def test_vecm_every_deterministic_case_matches_statsmodels(case):
    """Each of the nine statsmodels deterministic cases matches
    VECM(k_ar_diff=2, coint_rank=1, deterministic=case) on the trending
    draw: alpha, beta, det_coef_coint (the widened-beta rows, constant row
    first then trend row), gamma, det_coef, sigma_u, llf at 1e-6."""
    r = tsecon.vecm(TDATA, k_ar_diff=TR["k_ar_diff"], coint_rank=TR["coint_rank"],
                    deterministic=case)
    _assert_block_close(r, TR["cases"][case], case)


def test_vecm_restricted_split_shapes():
    """The statsmodels VECMResults split is reproduced key for key: beta
    keeps the k variable rows, det_coef_coint carries the restricted rows
    (constant first, then trend), det_coef the short-run columns."""
    shapes = {
        "n": (0, 0), "co": (0, 1), "ci": (1, 0), "lo": (0, 1), "li": (1, 0),
        "colo": (0, 2), "coli": (1, 1), "cilo": (1, 1), "cili": (2, 0),
    }
    for case, (n_coint, n_det) in shapes.items():
        r = tsecon.vecm(TDATA, k_ar_diff=2, coint_rank=1, deterministic=case)
        beta = np.asarray(r["beta"], float)
        if n_coint:
            dcc = np.asarray(r["det_coef_coint"], float)
            assert dcc.shape == (n_coint, 1), case
        else:
            assert np.asarray(r["det_coef_coint"]).size == 0, case
        dc = np.asarray(r["det_coef"], float)
        assert beta.shape == (3, 1), case
        # The normalization beta[:r,:r] = I, to float round-off (the top
        # block is multiplied by its own inverse, as in statsmodels).
        assert abs(beta[0, 0] - 1.0) < 1e-12, (
            f"{case}: widened-beta normalization beta[:r,:r]=I"
        )
        if n_det:
            assert dc.shape == (3, n_det), case
        else:
            assert dc.size == 0, case


def test_vecm_seasonal_matches_statsmodels():
    """seasons=4 centered seasonal dummies match statsmodels — both with
    the unrestricted constant at first_season=2 (the phase is pinned) and
    with the restricted constant ("ci" + seasons)."""
    se = VD["seasonal"]
    sdata = np.array(se["data"]).T
    for key, det in [("co_s4_fs2", "co"), ("ci_s4_fs0", "ci")]:
        fx = se[key]
        r = tsecon.vecm(sdata, k_ar_diff=se["k_ar_diff"], coint_rank=se["coint_rank"],
                        deterministic=det, seasons=se["seasons"],
                        first_season=fx["first_season"])
        _assert_block_close(r, fx, key)
        # det_coef column order: constant first, then the 3 centered dummies.
        n_det = 1 + (se["seasons"] - 1) if det == "co" else se["seasons"] - 1
        assert np.asarray(r["det_coef"]).shape == (2, n_det)


def test_vecm_first_season_changes_the_answer():
    """first_season shifts the dummy phase — a wrong phase is a different
    (worse) model, so the two fits must not coincide."""
    se = VD["seasonal"]
    sdata = np.array(se["data"]).T
    r0 = tsecon.vecm(sdata, k_ar_diff=1, coint_rank=1, deterministic="co",
                     seasons=4, first_season=0)
    r2 = tsecon.vecm(sdata, k_ar_diff=1, coint_rank=1, deterministic="co",
                     seasons=4, first_season=2)
    assert r0["llf"] != r2["llf"]
    assert not np.allclose(r0["det_coef"], r2["det_coef"])


def test_vecm_trending_case_choice_moves_beta():
    """The fixture-measured cross-case divergences reproduce live: the
    restricted trend visibly rotates beta on the trending draw, and
    "colo" agrees with coint_johansen(det_order=1) only asymptotically
    (pinned at ~1 - 6e-9, not exact — unlike the exact "co" <->
    det_order=0 identity above)."""
    pins = TR["beta_cosines"]
    b = {c: np.asarray(tsecon.vecm(TDATA, k_ar_diff=2, coint_rank=1,
                                   deterministic=c)["beta"])[:, 0]
         for c in ("co", "coli", "ci", "cili", "colo")}
    assert cosine(b["co"], b["coli"]) == pytest.approx(pins["co_coli"], rel=1e-6)
    assert cosine(b["ci"], b["cili"]) == pytest.approx(pins["ci_cili"], rel=1e-6)
    assert cosine(b["co"], b["coli"]) < 0.999  # the case choice matters
    evec1 = np.asarray(TR["johansen_det1"]["evec"])
    joh1_beta = evec1[:, 0] / evec1[0, 0]
    cos1 = cosine(b["colo"], joh1_beta)
    assert cos1 == pytest.approx(pins["colo_joh1"], rel=1e-6)
    assert 0.9999 < abs(cos1) < 1.0  # close, and honestly not exact


def test_vecm_dict_round_trip_and_summarize():
    """The result stays a plain JSON-serializable dict (the library's dict
    grammar) and renders through the generic results wrapper with every
    key visible — matching how the other vecm outputs are consumed."""
    import json as _json

    r = tsecon.vecm(TDATA, k_ar_diff=2, coint_rank=1, deterministic="cili")
    assert _json.loads(_json.dumps(r)) == {k: v for k, v in r.items()}
    assert set(r) == {"alpha", "beta", "det_coef_coint", "gamma", "det_coef",
                      "sigma_u", "llf"}
    from tsecon.results import summarize
    text = summarize(r, title="vecm").summary()
    for key in ("alpha", "beta", "det_coef_coint", "gamma", "sigma_u", "llf"):
        assert key in text


def test_vecm_docstring_names_every_returned_key():
    """The docstring-keys tripwire, applied to vecm: every returned key is
    named in __doc__ (the audit rounds 3-4 drift class)."""
    import re

    tokens = set(re.findall(r"`([A-Za-z_][A-Za-z_0-9]*)`", tsecon.vecm.__doc__ or ""))
    keys = set(tsecon.vecm(TDATA, k_ar_diff=2, coint_rank=1, deterministic="cili").keys())
    missing = keys - tokens
    assert not missing, f"vecm.__doc__ does not name returned keys: {sorted(missing)}"
