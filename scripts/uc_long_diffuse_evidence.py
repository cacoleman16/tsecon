"""Evidence for the exact-diffuse filter's diffuse-element test on long
diffuse periods (0.11.0, audit round 15: B15-UC-1 and B15-UC-2). Its output
is what the CHANGELOG, the unobserved-components model card and the
comments in crates/tsecon-ssm/src/filter.rs quote.

    .venv/bin/python scripts/uc_long_diffuse_evidence.py

Three parts:

1. A transcription of the filter's diffuse-element test, in NumPy on the
   system matrices statsmodels builds for the ``long_diffuse`` case
   ``llevel_seasonal12_fixed_nan2_14`` of ``fixtures/uc_extensions.json``
   (local level + fixed dummy seasonal 12, the same calendar month missing
   at t = 2 and 14). It prints the old test (``F_inf`` against ``1e-14``
   times ``|z|' P_mag |z|``, ``P_mag`` carried as ``|T| P_mag |T|'`` plus
   ``|M_inf| |K0|'``) next to the new one (the square-root factor's
   ``F_root`` against ``1e-14`` times ``|z|' |U| |z|``, ``U = T U T'``),
   following the old test's own decisions, at every element from t = 12
   that either test calls diffuse.
2. Every ``long_diffuse`` / ``long_diffuse_mle`` case of the fixture through
   the INSTALLED tsecon wheel against its statsmodels golden. Run it against
   a wheel built from before the fix to see the old numbers.
3. The B15-UC-2 design over 400 seeds: a local level + dummy seasonal 12 +
   one N(0, 1) regressor in the state (``mle_regression=False``, T = 96,
   ``default_rng(seed)``, fixed parameters [1, 0.1, 0.05]) through the
   installed tsecon and statsmodels: how many fits are refused, how many
   end the diffuse period somewhere else than statsmodels, and the largest
   log-likelihood gap.
4. The ``lambda -> 0`` divergence of a diffuse cycle on the series of
   ``the_cycle_log_likelihood_diverges_as_the_frequency_goes_to_zero`` in
   crates/tsecon-ssm/tests/uc_properties.rs (its LCG(808) and Box-Muller
   draws, transcribed): the fixed-parameter sweep, and the MLE with the
   period bounds opened to (2, 1e9) against the default (2, nobs).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

import tsecon

warnings.simplefilter("ignore")
FX = json.loads((Path(__file__).parents[1] / "fixtures" / "uc_extensions.json").read_text())
D = FX["data"]
PAT = np.array([1.0, 0.5, -0.5, -1.0, 0.2, 0.0, 0.3, -0.2, 0.8, -0.6, 0.1, -0.6])


def replay():
    case = next(c for c in FX["long_diffuse"] if c["name"] == "llevel_seasonal12_fixed_nan2_14")
    y = np.asarray(D[case["data"]], float)
    mod = UnobservedComponents(y, "llevel", seasonal=12, stochastic_seasonal=False, use_exact_diffuse=True)
    mod.update(np.asarray(case["params"], float))
    m = mod.k_states
    T = np.asarray(mod.ssm["transition"]).reshape(m, m)
    z = np.asarray(mod.ssm["design"]).reshape(-1)
    print(f"[1] {case['name']}: rho(|T|) = {max(abs(np.linalg.eigvals(np.abs(T)))):.4f}, "
          f"max |T^12 - I| = {np.abs(np.linalg.matrix_power(T, 12) - np.eye(m)).max():.1e}")
    p_inf = np.eye(m)
    p_mag = np.eye(m)
    unres = np.eye(m)
    root = np.eye(m)
    for t in range(len(y)):
        if np.sum(p_inf * p_inf) <= 1e-10:
            print(f"    P_inf washed out at t = {t} under the old test")
            break
        if not np.isnan(y[t]):
            mi = p_inf @ z
            f_inf = max(z @ mi, 0.0)
            rank_ok = f_inf > 1e-10 * (np.abs(z) @ np.abs(p_inf) @ np.abs(z))
            old_ref = np.abs(z) @ p_mag @ np.abs(z)
            u = root.T @ z
            f_root = u @ u
            new_ref = np.abs(z) @ np.abs(unres) @ np.abs(z)
            old = rank_ok and f_inf > 1e-14 * old_ref
            new = rank_ok and f_root > 1e-14 * new_ref
            if t >= 12 and (old or new):
                print(f"    t = {t:3d}: F_inf = {f_inf:.4g}; old threshold 1e-14 * {old_ref:.4g} = {1e-14 * old_ref:.4g}"
                      f" -> {'diffuse' if old else 'refused'}; new: F_root = {f_root:.4g} against 1e-14 * {new_ref:.4g}"
                      f" -> {'diffuse' if new else 'refused'}")
            if old:
                k0 = mi / f_inf
                p_inf = p_inf - np.outer(mi, k0)
                p_mag = p_mag + np.outer(np.abs(mi), np.abs(k0))
                root = root - np.outer(root @ u, u / f_root)
            p_inf = (p_inf + p_inf.T) / 2
        p_inf = T @ p_inf @ T.T
        p_mag = np.abs(T) @ p_mag @ np.abs(T).T
        unres = T @ unres @ T.T
        root = T @ root
        if t == 40:
            print("    (old test still diffuse at t = 40; stopped)")
            break


def kwargs(spec):
    return {k: (np.asarray(D[v], float) if k == "exog" else v) for k, v in spec.items()}


def cases():
    print(f"[2] tsecon {tsecon.__version__} against the long_diffuse goldens (statsmodels, use_exact_diffuse=True)")
    for c in FX["long_diffuse"]:
        try:
            r = tsecon.unobserved_components(np.asarray(D[c["data"]], float), fixed_params=c["params"],
                                             **kwargs(c["spec"]))
            got = (f"loglik {r['loglik']:.6f}, nobs_diffuse {r['nobs_diffuse']:3d} "
                   f"(gap {r['loglik'] - c['loglike']:+.2e})")
        except ValueError as e:
            got = "refused: " + str(e)[:100] + " ..."
        print(f"    {c['name']:44s} golden {c['loglike']:.6f}, nobs_diffuse {c['nobs_diffuse']:2d} | {got}")
    for c in FX["long_diffuse_mle"]:
        r = tsecon.unobserved_components(np.asarray(D[c["data"]], float), **kwargs(c["spec"]))
        print(f"    MLE {c['name']}: better reference {c['best']['llf']:.6f} (nobs_diffuse {c['best']['nobs_diffuse']}) | "
              f"tsecon {r['loglik']:.6f} (nobs_diffuse {r['nobs_diffuse']}) at {np.round(r['params'], 5).tolist()}")


def regressor_design(n_seeds=400):
    refused, nd_off, gap = [], [], 0.0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        T = 96
        x = rng.normal(size=(T, 1))
        y = np.cumsum(rng.normal(0, 0.3, T)) + np.tile(PAT, 8) + 0.7 * x[:, 0] + rng.normal(0, 1, T)
        p = [1.0, 0.1, 0.05]
        ref = UnobservedComponents(y, "llevel", seasonal=12, exog=x, mle_regression=False,
                                   use_exact_diffuse=True).smooth(np.array(p))
        try:
            r = tsecon.unobserved_components(y, level="llevel", seasonal=12, exog=x, mle_regression=False,
                                             fixed_params=p)
        except ValueError:
            refused.append(seed)
            continue
        if r["nobs_diffuse"] != ref.nobs_diffuse:
            nd_off.append((seed, r["nobs_diffuse"], int(ref.nobs_diffuse)))
        gap = max(gap, abs(r["loglik"] - ref.llf) / abs(ref.llf))
    print(f"[3] local level + seasonal 12 + one N(0, 1) regressor in the state, {n_seeds} seeds: "
          f"refused {len(refused)} {refused}; nobs_diffuse unlike statsmodels {len(nd_off)} {nd_off[:5]}; "
          f"largest relative loglik gap among the rest {gap:.1e}")


def cycle_divergence():
    mask = (1 << 64) - 1
    state = [(808 * 0x9E3779B97F4A7C15 + 1) & mask]

    def uniform():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & mask
        return (state[0] >> 11) / float(1 << 53)

    def normal():
        u1 = max(uniform(), 1e-300)
        u2 = uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    y, level = [], 0.0
    for _ in range(120):
        y.append(level + normal())
        level += 0.3 * normal()
    y = np.array(y)
    spec = dict(level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True)
    sweep = []
    for f in (1e-2, 1e-4, 1e-6, 1e-8):
        r = tsecon.unobserved_components(y, cycle_period_bounds=(2.0, 1e9), fixed_params=[0.8, 0.3, 0.2, f, 0.9],
                                         **spec)
        sweep.append(f"{f:g}: {r['loglik']:.3f} (nobs_diffuse {r['nobs_diffuse']})")
    print("[4] diffuse cycle, frequency -> 0 at fixed parameters: " + ", ".join(sweep))
    for bounds in ((2.0, 1e9), None):
        kw = dict(spec) if bounds is None else dict(spec, cycle_period_bounds=bounds)
        r = tsecon.unobserved_components(y, **kw)
        names = list(r["param_names"])
        print(f"    MLE, cycle_period_bounds {bounds or '(2, nobs) default'}: loglik {r['loglik']:.2f}, period "
              f"{2 * np.pi / r['params'][names.index('frequency.cycle')]:.3g}, converged {r['converged']}")


if __name__ == "__main__":
    replay()
    cases()
    regressor_design(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    cycle_divergence()
