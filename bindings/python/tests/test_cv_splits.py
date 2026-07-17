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
