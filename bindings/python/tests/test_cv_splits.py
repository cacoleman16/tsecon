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
