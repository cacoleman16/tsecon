"""Golden fixtures for the CART regression tree (`regression_tree`) and, through
the exact single-tree bridge, for the deterministic core of `random_forest`.

Reference: scikit-learn 1.9.0 `DecisionTreeRegressor(criterion="squared_error",
splitter="best", max_features=None, random_state=0)` — an INDEPENDENT
package golden.  Every stored case records the fitted training values, the
test-set predictions, `n_nodes`/`n_leaves`/`depth`, the impurity-based
`feature_importances_`, and the multiset of (feature, threshold) split pairs
sorted by (feature, threshold).

Conventions the Rust implementation reproduces (all documented in
crates/tsecon-ml/src/tree.rs):

  * best-split search over every feature; MSE criterion with the proxy
    improvement sum_left^2/n_left + sum_right^2/n_right; a split must leave
    at least `min_samples_leaf` rows on each side; a node is a leaf when it
    has fewer than `min_samples_split` rows, fewer than 2*min_samples_leaf
    rows, sits at `max_depth`, or is pure (impurity <= machine epsilon);
  * the threshold is the midpoint of the two adjacent sorted distinct
    values (`v[p-1]/2 + v[p]/2`), and two values within scikit-learn's
    FEATURE_THRESHOLD = 1e-7 of each other are treated as one value;
  * leaves predict the training mean; feature importance is the
    weighted-impurity decrease summed per feature and normalized to 1.

Why exact matching is possible — the tie-break.  scikit-learn visits the
features in an order drawn from its private RNG (`random_state`) and keeps
the FIRST split whose proxy improvement is strictly best, so whenever two
features yield the same partition of a node the winner depends on that
RNG.  Such ties are measure-zero for continuous data in nodes of a
reasonable size, but they are CERTAIN in two-row nodes (every feature
yields the same partition), so unbounded depth with `min_samples_leaf=1`
is deliberately NOT stored.  For every stored case this generator refits
the tree under five different `random_state` values and asserts the
fitted tree (features, thresholds, leaf values) is identical, which
proves no RNG-dependent tie-break was exercised: the stored tree is the
unique best-split tree, reproducible by any implementation of the
conventions above.

Float32.  scikit-learn casts the design to float32 before growing and
before predicting, so thresholds are midpoints of float32 values.  The
fixture's X and X_test are drawn in float64 and ROUNDED TO FLOAT32
(stored as the exactly representable float64 values), so the Rust side,
which works in float64, sees bit-identical feature values and forms
bit-identical midpoints.

Grade (honest): independent-package golden (scikit-learn 1.9.0) for the
deterministic tree.  The full random forest (bootstrap resampling, feature
subsampling, OOB, quantile forests, importance) is NOT pinned here — its
randomness is tsecon's own Philox stream — and is validated by seeded
property / Monte-Carlo tests in crates/tsecon-ml/tests/trees_properties.rs
whose measured numbers are quoted on the model card.

This generator NEVER imports tsecon.  Doubles are written with json's
shortest round-trip repr, which the Rust golden test parses to identical
bits (serde_json `float_roundtrip`).

Run:  .venv-wt/bin/python fixtures/generate_trees_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor

OUT = Path(__file__).resolve().parent / "trees.json"

SEED = 20260903
N_TRAIN = 300
N_TEST = 120
P = 8  # Friedman #1 uses the first five columns; three are pure noise.
NOISE_SD = 1.0

# (max_depth, min_samples_leaf, min_samples_split); None = unbounded depth.
# Candidate settings. A candidate is STORED only if its tree is invariant to
# scikit-learn's feature-visit order (see the tie-break note above); the
# ones that exercise an RNG tie-break are listed in `_meta.excluded_settings`
# so the reader can see which regimes are not golden-reproducible.
CANDIDATES = [
    (2, 1, 2),
    (3, 1, 2),
    (4, 2, 2),
    (5, 3, 2),
    (6, 3, 10),
    (3, 1, 30),
    (None, 5, 2),
    (None, 8, 2),
    (None, 10, 2),
    (None, 20, 2),
]
MIN_STORED = 7
TIE_STATES = (1, 7, 42, 123, 2026)


def friedman1(x):
    return (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )


def as_float32_grid(a):
    """Round to float32 and return the exactly representable float64 values."""
    return a.astype(np.float32).astype(np.float64)


def tree_signature(t):
    tr = t.tree_
    return (
        tr.feature.tolist(),
        tr.children_left.tolist(),
        tr.children_right.tolist(),
        np.asarray(tr.threshold),
        np.asarray(tr.value).ravel(),
    )


def same_tree(a, b):
    """Structurally identical trees with thresholds/values within 1e-12.

    Node VALUES are not compared bitwise: scikit-learn accumulates a node's
    sum in whatever order its last feature sort left the samples, so the
    same tree grown under a different feature-visit order differs in the
    leaf means by ~1e-14 (measured), which is summation order, not a
    different tree. A genuine tie-break changes `feature`/children and
    moves thresholds by O(1).
    """
    if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
        return False
    return (
        np.max(np.abs(a[3] - b[3])) <= 1e-12
        and np.max(np.abs(a[4] - b[4])) <= 1e-12
    )


def fit(x, y, max_depth, msl, mss, random_state=0):
    return DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_features=None,
        max_depth=max_depth,
        min_samples_leaf=msl,
        min_samples_split=mss,
        random_state=random_state,
    ).fit(x, y)


def main():
    rng = np.random.default_rng(SEED)
    x = as_float32_grid(rng.uniform(size=(N_TRAIN, P)))
    x_test = as_float32_grid(rng.uniform(size=(N_TEST, P)))
    y = friedman1(x) + NOISE_SD * rng.standard_normal(N_TRAIN)

    cases = []
    excluded = []
    for max_depth, msl, mss in CANDIDATES:
        t = fit(x, y, max_depth, msl, mss)
        sig = tree_signature(t)
        # Tie-freeness proof: five other feature-visit orders, same tree.
        tied = [
            rs
            for rs in TIE_STATES
            if not same_tree(sig, tree_signature(fit(x, y, max_depth, msl, mss, rs)))
        ]
        if tied:
            print(
                f"  excluded {(max_depth, msl, mss)}: RNG tie-break exercised "
                f"(random_state in {tied} grows a different tree)"
            )
            excluded.append(
                {
                    "max_depth": max_depth,
                    "min_samples_leaf": msl,
                    "min_samples_split": mss,
                    "differing_random_states": tied,
                }
            )
            continue
        tr = t.tree_
        internal = tr.children_left != -1
        pairs = sorted(
            zip(tr.feature[internal].tolist(), tr.threshold[internal].tolist())
        )
        if not pairs:
            raise SystemExit(f"setting {(max_depth, msl, mss)} grew no split")
        cases.append(
            {
                "name": f"depth{max_depth}_leaf{msl}_split{mss}",
                "params": {
                    "max_depth": max_depth,
                    "min_samples_leaf": msl,
                    "min_samples_split": mss,
                },
                "fitted": t.predict(x).tolist(),
                "predicted": t.predict(x_test).tolist(),
                "n_nodes": int(tr.node_count),
                "n_leaves": int(tr.n_leaves),
                "depth": int(tr.max_depth),
                "feature_importances": t.feature_importances_.tolist(),
                "splits": [[int(f), float(th)] for f, th in pairs],
            }
        )

    if len(cases) < MIN_STORED:
        raise SystemExit(f"only {len(cases)} tie-free cases; need {MIN_STORED}")

    fixture = {
        "_meta": {
            "reference": (
                "scikit-learn DecisionTreeRegressor(criterion='squared_error', "
                "splitter='best', max_features=None, random_state=0)"
            ),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "seed": SEED,
            "dgp": (
                "Friedman #1: y = 10 sin(pi x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 "
                "+ 5 x5 + N(0, 1); x ~ U(0,1)^8 rounded to float32 (three "
                "noise columns)"
            ),
            "conventions": (
                "midpoint thresholds v[p-1]/2 + v[p]/2 over float32-rounded "
                "features; FEATURE_THRESHOLD 1e-7 tie window; min_samples_leaf "
                "on both sides; leaves predict the training mean; "
                "feature_importances_ = normalized weighted impurity decrease"
            ),
            "tie_break": (
                "scikit-learn keeps the first strictly-best split in its "
                "RNG-ordered feature visit; every stored case was refit under "
                "random_state in {1, 7, 42, 123, 2026} and the tree was "
                "identical, so no RNG-dependent tie-break was exercised and "
                "the stored tree is the unique best-split tree. Unbounded "
                "depth with min_samples_leaf=1 is excluded on purpose: "
                "two-row nodes tie on every feature."
            ),
            "grade": (
                "independent-package golden for the deterministic tree "
                "(and for random_forest through its exact single-tree bridge: "
                "bootstrap='none', max_features='all', n_trees=1). The full "
                "forest is property / Monte-Carlo graded in "
                "crates/tsecon-ml/tests/trees_properties.rs"
            ),
            "excluded_settings": excluded,
        },
        "X": x.tolist(),
        "X_test": x_test.tolist(),
        "y": y.tolist(),
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {OUT}")
    for c in cases:
        print(
            f"  {c['name']}: n_nodes={c['n_nodes']} n_leaves={c['n_leaves']} "
            f"depth={c['depth']} importances={np.round(c['feature_importances'], 3)}"
        )


if __name__ == "__main__":
    main()
