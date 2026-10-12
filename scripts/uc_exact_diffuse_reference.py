"""An exact-arithmetic reference for the exact-diffuse filter of
``unobserved_components``, and the numbers quoted from it in
crates/tsecon-ssm/src/filter.rs (``TOLERANCE_CANCEL``),
crates/tsecon-ssm/tests/uc_properties.rs and
crates/tsecon-ssm/tests/uc_extensions_golden.rs (audit round 15).

    .venv/bin/python scripts/uc_exact_diffuse_reference.py

statsmodels supplies only the system matrices of the model (design,
transition, selection, covariances and the initialization at the given
parameters). The recursion is the Koopman-Durbin univariate exact-diffuse
filter, written out here in 90-significant-digit decimal arithmetic: every
double-precision input converts to ``Decimal`` exactly, so this is the
exact-diffuse log-likelihood of the double inputs to about 90 digits, with
no tolerance tuned to double precision. Roundoff is near 1e-90 and every
genuine ``F_inf`` below is above 1e-20, so ``F_inf > 1e-60`` decides "zero
or not" with decades to spare on both sides. Where a diffuse direction is
resolved only weakly (``F_inf`` many orders below the diffuse variance the
observation could see), double precision cannot follow the exact value,
and this shows by how much.

Three parts:

1. Every ``long_diffuse`` golden of fixtures/uc_extensions.json: the exact
   log-likelihood against statsmodels' (the golden) and the installed
   tsecon's.
2. The diffuse cycle as its frequency goes to zero, on the series of
   ``the_cycle_log_likelihood_diverges_as_the_frequency_goes_to_zero`` in
   uc_properties.rs (its LCG(808) and Box-Muller draws, transcribed): the
   exact log-likelihood and the first ``F_inf`` of the weakly observed
   cycle state, against tsecon; plus, from a float transcription of
   tsecon's diffuse-element test, which sightings of that state take the
   diffuse branch (``F_root / |z|'|U||z|`` against ``TOLERANCE_CANCEL =
   1e-14``).
3. The same divergence on the GDP sweep of uc_extensions.json
   (``stationary_cycle_lambda_sweep``, a smooth trend + diffuse cycle):
   exact against tsecon and statsmodels.
"""
import json
import warnings
from decimal import Decimal as D
from decimal import getcontext
from pathlib import Path

import numpy as np
from statsmodels.tsa.statespace.structural import UnobservedComponents

import tsecon

warnings.simplefilter("ignore")
getcontext().prec = 90
ZERO = D(0)
TINY = D("1e-60")
LN2PI = (2 * D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628")).ln()
FX = json.loads((Path(__file__).parents[1] / "fixtures" / "uc_extensions.json").read_text())


def system(mod, params):
    """The double-precision system matrices statsmodels builds at params."""
    mod.update(np.asarray(params, float))
    ssm = mod.ssm

    def tv(name, nd):
        x = np.asarray(ssm[name], float)
        return x[..., None] if x.ndim == nd - 1 else x

    a1, p_inf, p_star = ssm.initialization(model=ssm)
    sel, q = tv("selection", 3)[:, :, 0], tv("state_cov", 3)[:, :, 0]
    assert tv("transition", 3).shape[2] == 1 and tv("state_intercept", 2).shape[1] == 1
    return dict(y=np.asarray(ssm.endog, float)[0], Z=tv("design", 3)[0], d=tv("obs_intercept", 2)[0],
                H=tv("obs_cov", 3)[0, 0], T=tv("transition", 3)[:, :, 0], c=tv("state_intercept", 2)[:, 0],
                R=sel, Q=q, a1=np.asarray(a1, float), Ps=np.asarray(p_star, float), Pi=np.asarray(p_inf, float))


def at(x, t):
    return x[..., t if x.shape[-1] > 1 else 0]


def exact_loglik(s):
    """(log-likelihood, nobs_diffuse, [(t, F_inf)]) in decimal arithmetic."""
    dm = lambda a: [[D(float(v)) for v in row] for row in np.atleast_2d(a)]  # noqa: E731
    dv = lambda a: [D(float(v)) for v in np.ravel(a)]  # noqa: E731
    m = s["T"].shape[0]
    tnz = [[(j, v) for j, v in enumerate(row) if v != 0] for row in dm(s["T"])]
    rd, qd = dm(s["R"]), dm(s["Q"])
    rq = [[sum((rd[i][k] * qd[k][l] for k in range(len(qd))), ZERO) for l in range(len(qd))] for i in range(m)]
    rqr = [[sum((rq[i][l] * rd[j][l] for l in range(len(qd))), ZERO) for j in range(m)] for i in range(m)]
    c = dv(s["c"])

    def sandwich(p):
        tp = [[sum((v * p[j][k] for j, v in tnz[i]), ZERO) for k in range(m)] for i in range(m)]
        return [[sum((tp[i][j] * v for j, v in tnz[k]), ZERO) for k in range(m)] for i in range(m)]

    def mv(p, z):
        return [sum((pij * zj for pij, zj in zip(row, z) if zj != 0), ZERO) for row in p]

    def dot(u, v):
        return sum((x * w for x, w in zip(u, v)), ZERO)

    a, ps, pi = dv(s["a1"]), dm(s["Ps"]), dm(s["Pi"])
    ll, nd, in_diffuse, f_infs = ZERO, 0, True, []
    for t, yt in enumerate(s["y"]):
        in_diffuse = in_diffuse and sum((x * x for row in pi for x in row), ZERO) > TINY
        nd += in_diffuse
        if not np.isnan(yt):
            z, h = dv(at(s["Z"], t)), D(float(at(s["H"], t)))
            v = D(float(yt)) - D(float(at(s["d"], t))) - dot(z, a)
            ms = mv(ps, z)
            fs = dot(z, ms) + h
            mi = mv(pi, z) if in_diffuse else [ZERO] * m
            fi = dot(z, mi)
            if fi > TINY:
                f_infs.append((t, fi))
                k0 = [x / fi for x in mi]
                k1 = [x / fi - k * fs / fi for x, k in zip(ms, k0)]
                a = [ai + v * k for ai, k in zip(a, k0)]
                ps = [[ps[i][j] - ms[i] * k0[j] - mi[i] * k1[j] for j in range(m)] for i in range(m)]
                pi = [[pi[i][j] - mi[i] * k0[j] for j in range(m)] for i in range(m)]
                ll -= (LN2PI + fi.ln()) / 2
            elif fs > TINY:
                k = [x / fs for x in ms]
                a = [ai + v * kk for ai, kk in zip(a, k)]
                ps = [[ps[i][j] - ms[i] * k[j] for j in range(m)] for i in range(m)]
                ll -= (LN2PI + fs.ln() + v * v / fs) / 2
        a = [sum((v * a[j] for j, v in tnz[i]), ZERO) + c[i] for i in range(m)]
        ps = sandwich(ps)
        ps = [[ps[i][j] + rqr[i][j] for j in range(m)] for i in range(m)]
        if in_diffuse:
            pi = sandwich(pi)
    return float(ll), nd, f_infs


def sightings(s):
    """tsecon's diffuse-element test, transcribed in float (filter.rs): for
    every diffuse-period observation, F_root / (|z|' |U| |z|) and whether it
    took the diffuse branch. Only the decision is transcribed; the update is
    the plain Koopman-Durbin one."""
    y, T = s["y"], s["T"]
    a, ps, pi = s["a1"].copy(), s["Ps"].copy(), s["Pi"].copy()
    cols = [i for i in range(len(a)) if pi[i, i] > 0]
    root = np.zeros((len(a), len(cols)))
    for j, i in enumerate(cols):
        root[i, j] = np.sqrt(pi[i, i])
    unres = pi.copy()
    rqr = s["R"] @ s["Q"] @ s["R"].T
    out, in_diffuse = [], True
    for t in range(len(y)):
        in_diffuse = in_diffuse and np.sum(pi * pi) > 1e-10
        if not in_diffuse:
            break
        if not np.isnan(y[t]):
            z, h = at(s["Z"], t), at(s["H"], t)
            v = y[t] - at(s["d"], t) - z @ a
            ms, mi = ps @ z, pi @ z
            fs, fi = max(z @ ms + h, 0.0), max(z @ mi, 0.0)
            u = root.T @ z
            ratio = (u @ u) / max(np.abs(z) @ np.abs(unres) @ np.abs(z), 1e-300)
            take = fi > 1e-10 * (np.abs(z) @ np.abs(pi) @ np.abs(z)) and ratio > 1e-14
            out.append((t, ratio, bool(take)))
            if take:
                k0 = mi / fi
                a = a + v * k0
                ps = ps - np.outer(ms, k0) - np.outer(mi, ms / fi - k0 * fs / fi)
                pi = pi - np.outer(mi, k0)
                root = root - np.outer(root @ u, u / (u @ u))
            elif fs > 0:
                a = a + v * ms / fs
                ps = ps - np.outer(ms, ms / fs)
        a = T @ a + s["c"]
        ps = T @ ((ps + ps.T) / 2) @ T.T + rqr
        pi = T @ ((pi + pi.T) / 2) @ T.T
        root, unres = T @ root, T @ unres @ T.T
    return out


def long_diffuse():
    print("[1] long_diffuse goldens: exact log-likelihood, and statsmodels' and tsecon's gaps to it")
    data = FX["data"]
    for c in FX["long_diffuse"]:
        kw = dict(c["spec"])
        x = np.asarray(data[kw.pop("exog")], float) if "exog" in kw else None
        y = np.asarray(data[c["data"]], float)
        ll, nd, _ = exact_loglik(system(UnobservedComponents(y, exog=x, use_exact_diffuse=True, **kw), c["params"]))
        r = tsecon.unobserved_components(y, fixed_params=c["params"], **kw, **({} if x is None else dict(exog=x)))
        print(f"    {c['name']:44s} exact {ll:.10f} (nobs_diffuse {nd:2d}) | statsmodels {c['loglike'] - ll:+.1e}"
              f" | tsecon {r['loglik'] - ll:+.1e}")


def lcg_series():
    mask = (1 << 64) - 1
    state = [(808 * 0x9E3779B97F4A7C15 + 1) & mask]

    def uniform():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & mask
        return (state[0] >> 11) / float(1 << 53)

    def normal():
        u1 = max(uniform(), 1e-300)
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * uniform())

    y, level = [], 0.0
    for _ in range(120):
        y.append(level + normal())
        level += 0.3 * normal()
    return np.array(y)


def cycle_sweep():
    print("[2] diffuse cycle, frequency -> 0 (uc_properties.rs series, parameters [0.8, 0.3, 0.2, f, 0.9])")
    y = lcg_series()
    spec = dict(level="llevel", cycle=True, damped_cycle=True, stochastic_cycle=True)
    for f in (1e-2, 1e-4, 1e-6, 1e-7, 1e-8):
        p = [0.8, 0.3, 0.2, f, 0.9]
        s = system(UnobservedComponents(y, use_exact_diffuse=True, **spec), p)
        ll, nd, f_infs = exact_loglik(s)
        r = tsecon.unobserved_components(y, cycle_period_bounds=(2.0, 1e9), fixed_params=p, **spec)
        # The first two observations resolve the level and the cycle's first
        # state; what follows is the weakly observed second cycle state.
        later = sightings(s)[2:]
        took = next((i for i, (_, _, take) in enumerate(later) if take), len(later))
        refused = [q for _, q, _ in later[:took]]
        where = f"t = {later[took][0]} at {later[took][1]:.1e}" if took < len(later) else "no t"
        print(f"    {f:g}: exact {ll:.6f} (nobs_diffuse {nd}, last F_inf {float(f_infs[-1][1]):.2e} at t = "
              f"{f_infs[-1][0]}) | tsecon {r['loglik']:.6f} (nobs_diffuse {r['nobs_diffuse']}, gap "
              f"{r['loglik'] - ll:+.3f}); the transcribed test takes the cycle state at {where}, after "
              f"refusing {len(refused)} sightings (largest {max(refused, default=0.0):.1e})")


def gdp_sweep():
    print("[3] GDP smooth trend + diffuse cycle, frequency -> 0 (stationary_cycle_lambda_sweep)")
    y = np.asarray(FX["data"]["lgdp"], float)
    spec = dict(level="strend", cycle=True, damped_cycle=True, stochastic_cycle=True)
    for sw in FX["stationary_cycle_lambda_sweep"]:
        s = system(UnobservedComponents(y, use_exact_diffuse=True, **spec), sw["params"])
        ll, nd, f_infs = exact_loglik(s)
        r = tsecon.unobserved_components(y, cycle_period_bounds=(2.0, 1e9), fixed_params=sw["params"], **spec)
        print(f"    {sw['frequency']:g}: exact {ll:.6f} (smallest F_inf {min(float(v) for _, v in f_infs):.1e}) | "
              f"tsecon {r['loglik']:.6f} | statsmodels {sw['diffuse']:.6f}")


if __name__ == "__main__":
    long_diffuse()
    cycle_sweep()
    gdp_sweep()
