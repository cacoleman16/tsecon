"""Tests for the leakage-safe CV split binding.

The split geometry is checked analytically: no split may put a test index
at or before any training index (that would leak the future into the
past), and the purged K-fold must honor its purge/embargo gaps.
"""
import pytest
import tsecon


def test_expanding_origin_no_leakage():
    splits = tsecon.cv_splits(50, scheme="expanding", train=20, horizon=5, step=5)
    assert len(splits) > 0
    for s in splits:
        assert s["train"] == list(range(0, max(s["train"]) + 1))  # contiguous from 0
        assert len(s["test"]) == 5
        # Every test index is strictly after the whole training set.
        assert min(s["test"]) > max(s["train"])
    # Expanding: successive training sets grow.
    sizes = [len(s["train"]) for s in splits]
    assert sizes == sorted(sizes)
    assert sizes[0] == 20


def test_rolling_origin_fixed_width():
    splits = tsecon.cv_splits(50, scheme="rolling", train=15, horizon=3, step=3)
    assert len(splits) > 0
    for s in splits:
        assert len(s["train"]) == 15  # fixed window
        assert min(s["test"]) > max(s["train"])


def test_purged_kfold_gaps():
    purge, embargo = 2, 3
    splits = tsecon.cv_splits(100, scheme="purged_kfold", k=5, purge=purge, embargo=embargo)
    assert len(splits) == 5
    for s in splits:
        train = set(s["train"])
        test = set(s["test"])
        assert not (train & test)  # disjoint
        lo, hi = min(s["test"]), max(s["test"])
        # No training index within the purge gap before / embargo gap after.
        for t in range(lo - purge, lo):
            assert t not in train
        for t in range(hi + 1, hi + 1 + embargo):
            assert t not in train


def test_unknown_scheme_errors():
    with pytest.raises(ValueError):
        tsecon.cv_splits(50, scheme="bogus")


# --------------------------------------------------------------------------
# Audit fix (round 2, cv_splits): purge/embargo were silently ignored on
# scheme="expanding"/"rolling". Purge now opens a real train/test gap on the
# walk-forward schemes; embargo — meaningless there by construction — raises.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walk_forward_purge_opens_the_documented_gap(scheme):
    purge = 3
    base = tsecon.cv_splits(60, scheme=scheme, train=20, horizon=5, step=5)
    purged = tsecon.cv_splits(60, scheme=scheme, train=20, horizon=5, step=5, purge=purge)
    assert purged != base  # the argument is no longer inert
    assert len(purged) == len(base)
    for b, p in zip(base, purged):
        # Purge must not move the test geometry, only truncate training.
        assert p["test"] == b["test"]
        assert p["train"] == b["train"][:-purge]
        # The documented gap: exactly `purge` indices between the end of
        # training and the start of testing.
        assert min(p["test"]) - max(p["train"]) - 1 == purge


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walk_forward_purge_leakage_property(scheme):
    # No train index within `purge` of its fold's test start.
    purge = 4
    for s in tsecon.cv_splits(80, scheme=scheme, train=25, horizon=6, step=6, purge=purge):
        test_start = min(s["test"])
        assert all(t < test_start - purge for t in s["train"])


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walk_forward_embargo_raises_instead_of_ignoring(scheme):
    # Walk-forward training windows end before their test block, so an
    # embargo (an exclusion AFTER the test block) has nothing to act on;
    # accepting it silently is the audited defect.
    with pytest.raises(ValueError, match="embargo"):
        tsecon.cv_splits(60, scheme=scheme, train=20, horizon=5, step=5, embargo=4)
    # The error teaches the working alternatives.
    with pytest.raises(ValueError, match="purged_kfold"):
        tsecon.cv_splits(60, scheme=scheme, train=20, horizon=5, step=5, embargo=1)


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_walk_forward_purge_consuming_the_window_errors(scheme):
    with pytest.raises(ValueError):
        tsecon.cv_splits(60, scheme=scheme, train=20, horizon=5, step=5, purge=20)


# --------------------------------------------------------------------------
# Field fix (0.5.0 report, item 11): under scheme="purged_kfold" the embargo
# was ABSORBED by the purge — the right-hand exclusion was max(purge,
# embargo), so (purge=21, embargo=10) was bit-identical to (21, 0). AFML
# ch. 7 (Lopez de Prado 2018; mlfinlab likewise) measures the embargo from
# the END of the purged window, so the exclusions add: the right-hand gap
# is purge + embargo.
# --------------------------------------------------------------------------

def _right_gaps(splits):
    """Measured gap after each test block: first training index past the
    block minus the block's exclusive end (None when no index follows)."""
    gaps = []
    for s in splits:
        te = max(s["test"]) + 1
        after = [t for t in s["train"] if t >= te]
        gaps.append(min(after) - te if after else None)
    return gaps


@pytest.mark.parametrize(
    "purge,embargo,gap", [(21, 0, 21), (0, 10, 10), (21, 10, 31), (21, 30, 51)]
)
def test_purged_kfold_right_gap_is_purge_plus_embargo(purge, embargo, gap):
    splits = tsecon.cv_splits(300, scheme="purged_kfold", k=4, purge=purge, embargo=embargo)
    # Every fold with a right-hand training block shows the exact additive gap.
    assert _right_gaps(splits)[:-1] == [gap] * 3
    # The left gap stays purge-only: there is no left embargo.
    for s in splits[1:]:
        lo = min(s["test"])
        before = [t for t in s["train"] if t < lo]
        assert lo - max(before) - 1 == purge


def test_purged_kfold_embargo_not_absorbed_by_purge():
    # The pre-fix defect made these two calls bit-identical.
    absorbed = tsecon.cv_splits(300, scheme="purged_kfold", k=4, purge=21, embargo=0)
    additive = tsecon.cv_splits(300, scheme="purged_kfold", k=4, purge=21, embargo=10)
    assert absorbed != additive


def test_purged_kfold_embargo_reaching_past_the_sample_end():
    # A right band past n erases the right-hand training block, nothing more.
    splits = tsecon.cv_splits(50, scheme="purged_kfold", k=5, purge=0, embargo=100)
    assert splits[0]["train"] == []
    assert splits[-1]["train"] == list(range(40))
