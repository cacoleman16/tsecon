"""Golden fixtures for the Fry-Pagan (2011) median-target SVAR selection
(`fry_pagan_svar` / `tsecon_ident::median_target`).

VALIDATION STRATEGY
===================
Nothing here imports tsecon. The fixture STORES, as INPUT, a fixed set of `D`
candidate sign-normalized structural IRFs (random orthogonal rotations of a
fixed Cholesky IRF, filtered/normalized to a sign pattern), and INDEPENDENTLY
computes — in pure NumPy — the pointwise median, the pointwise (population)
standard deviation, every draw's median-target criterion `MT(d)`, the winning
index `argmin_d MT(d)`, and the pointwise median band. The Rust `median_target`
run on the identical stored draws must reproduce the stored `mt_index` (exactly)
and `mt_statistic` / `median_irf` (to 1e-10). Because the draws are the input,
the only thing under test is the SELECTION RULE — a pure-arithmetic cross-check.

The Fry-Pagan criterion, per target cell (variable i, shock j, horizon h):
    med  = median_d  Theta^(d)[h, i, j]
    sd   = std_d     Theta^(d)[h, i, j]            (population, ddof=0)
    z^(d)= (Theta^(d)[h, i, j] - med) / sd         (cells with sd == 0 dropped)
    MT(d)= sum_{target cells} (z^(d))^2
    d*   = argmin_d MT(d)                           (ties -> lowest index)

HONEST NOTE
-----------
The selection rule is validated exactly; the object it selects (one interior
point of the identified set) inherits the set-identification caveat — which
point depends on the informative Haar sampling prior. This is a descriptive
summary, not a point-identified estimate.
"""

import json
import os

import numpy as np


def ma_weights(coefs, horizon):
    """Psi_0..Psi_horizon from lag matrices coefs = [A_1, ..., A_p]."""
    n = coefs[0].shape[0]
    p = len(coefs)
    psi = [np.eye(n)]
    for h in range(1, horizon + 1):
        acc = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            acc = acc + psi[h - i] @ coefs[i - 1]
        psi.append(acc)
    return psi


def cholesky_irf(coefs, sigma, horizon):
    """Theta_h = Psi_h @ chol_lower(Sigma), the recursive structural IRF."""
    lower = np.linalg.cholesky(sigma)
    psi = ma_weights(coefs, horizon)
    return np.array([ps @ lower for ps in psi])  # [H+1, n, n]


def haar_orthogonal(n, rng):
    """A (roughly Haar) orthogonal matrix via QR of a Gaussian matrix with the
    Mezzadri (2007) R-diagonal sign fix. Only used to spread the candidate
    draws; the fixture stores the resulting IRFs, so the exact law is immaterial.
    """
    z = rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diag(r))
    d[d == 0] = 1.0
    return q * d  # column-scale by the R-diagonal signs


def orient(candidate, restrictions):
    """Choose per-shock column signs so every sign restriction holds; return the
    sign-normalized IRF or None if some restricted shock admits no orientation.

    `candidate` is [H+1, n, n]; `restrictions` a list of (var, shock, h, sign)
    with sign in {+1, -1}. Mirrors SignRestrictionSet::accept_orientations: an
    unrestricted shock keeps sign +1.
    """
    n = candidate.shape[1]
    shocks = sorted({r[1] for r in restrictions})
    orient_vec = np.ones(n)
    for s in shocks:
        chosen = None
        for sign_try in (1.0, -1.0):
            ok = True
            for (v, sh, h, sg) in restrictions:
                if sh != s:
                    continue
                val = sign_try * candidate[h, v, sh]
                if sg > 0 and not (val > 0.0):
                    ok = False
                    break
                if sg < 0 and not (val < 0.0):
                    ok = False
                    break
            if ok:
                chosen = sign_try
                break
        if chosen is None:
            return None
        orient_vec[s] = chosen
    out = candidate.copy()
    for s in range(n):
        if orient_vec[s] != 1.0:
            out[:, :, s] *= orient_vec[s]
    return out


def accepted_draws(base, restrictions, n_draws, rng):
    """Draw rotations, form candidate = base @ Q, sign-normalize, keep n_draws
    accepted draws. Returns array [D, H+1, n, n]."""
    n = base.shape[1]
    out = []
    tries = 0
    while len(out) < n_draws and tries < 100 * n_draws:
        tries += 1
        q = haar_orthogonal(n, rng)
        candidate = np.array([theta @ q for theta in base])  # [H+1, n, n]
        normalized = orient(candidate, restrictions)
        if normalized is not None:
            out.append(normalized)
    if len(out) < n_draws:
        raise RuntimeError(f"only accepted {len(out)}/{n_draws} draws")
    return np.array(out)


def median_target_numpy(draws, target_cells):
    """Independent NumPy median-target: returns (mt_index, mt_statistic, mt_per_draw,
    median_irf). `draws` is [D, H+1, n, n]; target_cells a list of (i, j, h)."""
    med = np.median(draws, axis=0)          # [H+1, n, n]
    sd = np.std(draws, axis=0)              # population std (ddof=0)
    d_count = draws.shape[0]
    mt = np.zeros(d_count)
    for (i, j, h) in target_cells:
        s = sd[h, i, j]
        if s > 0.0:
            z = (draws[:, h, i, j] - med[h, i, j]) / s
            mt += z * z
    mt_index = int(np.argmin(mt))           # np.argmin returns the FIRST min -> lowest index
    return mt_index, float(mt[mt_index]), mt, med


def as_rows(mat):
    return [[float(mat[i, j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]


def theta_to_rows(theta):
    """[H+1, n, n] ndarray -> list of row-major matrices."""
    return [as_rows(theta[h]) for h in range(theta.shape[0])]


def pack_scenario(name, base, restrictions, target, n_draws, seed):
    rng = np.random.default_rng(seed)
    draws = accepted_draws(base, restrictions, n_draws, rng)  # [D, H+1, n, n]
    n = draws.shape[2]
    n_h = draws.shape[1]
    if target == "restricted":
        shocks = sorted({r[1] for r in restrictions})
    elif target == "all":
        shocks = list(range(n))
    else:
        raise ValueError(target)
    target_cells = [(i, j, h) for h in range(n_h) for i in range(n) for j in shocks]

    mt_index, mt_stat, mt_all, med = median_target_numpy(draws, target_cells)

    # Guarantee a decisive winner so the stored integer index is stable across
    # independent double-precision arithmetic.
    sorted_mt = np.sort(mt_all)
    gap = sorted_mt[1] - sorted_mt[0]
    assert gap > 1e-6, f"{name}: winner gap {gap} too small for a stable index"

    # Sanity: the winner is a member of the accepted set and attains the min.
    assert mt_index == int(np.argmin(mt_all))
    assert abs(mt_all[mt_index] - sorted_mt[0]) < 1e-15

    return {
        "name": name,
        "n": n,
        "horizon": n_h - 1,
        "n_draws": int(n_draws),
        "target": target,
        "restricted_shocks": sorted({r[1] for r in restrictions}),
        "restrictions": [[int(v), int(s), int(h), ("+" if sg > 0 else "-")]
                         for (v, s, h, sg) in restrictions],
        "target_cells": [[int(i), int(j), int(h)] for (i, j, h) in target_cells],
        "draws": [theta_to_rows(draws[d]) for d in range(draws.shape[0])],
        "expected": {
            "mt_index": int(mt_index),
            "mt_statistic": float(mt_stat),
            "median_irf": theta_to_rows(med),
            "winner_gap": float(gap),
        },
    }


def main():
    # --- fixed VAR(1), n = 3 -------------------------------------------------
    a1 = np.array([[0.5, 0.1, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])
    sig3 = np.array([[1.0, 0.4, 0.2], [0.4, 0.97, 0.33], [0.2, 0.33, 0.62]])
    horizon = 8
    base = cholesky_irf([a1], sig3, horizon)  # [H+1, 3, 3]

    # Sign pattern on shock 0: variable 0 up, variable 1 down on impact.
    restrictions = [(0, 0, 0, +1), (1, 0, 0, -1)]

    scenarios = [
        pack_scenario("restricted_shock0_3var", base, restrictions,
                      target="restricted", n_draws=50, seed=20260722),
        pack_scenario("all_cells_3var", base, restrictions,
                      target="all", n_draws=50, seed=20260731),
    ]

    # -- independent invariants (no tsecon) ----------------------------------
    for sc in scenarios:
        draws = np.array(sc["draws"])                    # [D, H+1, n, n]
        cells = [tuple(c) for c in sc["target_cells"]]
        idx, stat, mt_all, med = median_target_numpy(draws, cells)
        assert idx == sc["expected"]["mt_index"]
        assert abs(stat - sc["expected"]["mt_statistic"]) < 1e-12
        # Winner really is the argmin over ALL draws.
        for d in range(draws.shape[0]):
            assert mt_all[d] >= mt_all[idx] - 1e-12
        # Sign restrictions hold on every stored draw (they are the accepted set).
        for d in range(draws.shape[0]):
            for (v, s, h, sg) in sc["restrictions"]:
                val = draws[d, h, v, s]
                assert (val > 0) if sg == "+" else (val < 0), "stored draw violates its own sign pattern"

    fixtures = {"scenarios": scenarios}
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "fry_pagan_svar.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fixtures, fh, indent=2)
        fh.write("\n")
    print("wrote", path)
    for sc in scenarios:
        print(f"  {sc['name']}: D={sc['n_draws']} target={sc['target']} "
              f"mt_index={sc['expected']['mt_index']} "
              f"mt_stat={sc['expected']['mt_statistic']:.6f} "
              f"gap={sc['expected']['winner_gap']:.4f}")


if __name__ == "__main__":
    main()
