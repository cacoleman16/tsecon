"""Golden fixture for tsecon-panel::lp_did — LP-DiD, the local-projections
difference-in-differences of Dube, Girardi, Jordà & Taylor (2025,
J. Applied Econometrics 40(7), doi:10.1002/jae.70000; NBER WP 31184).

Run with a NumPy-only venv (this script never imports tsecon); R with the
fixest package must be on PATH for the reference leg:

    .venv/bin/python fixtures/generate_lpdid_fixtures.py

============================================================================
WHAT KIND OF GOLDEN IS THIS
============================================================================
A **reference-run golden with an independent transcription cross-check**.
The stored coefficient/SE/nobs values come from an actual R run of the
authors' estimator conventions through fixest — the engine of the authors'
own reference code — executed by the committed companion script
`generate_lpdid_fixtures.R`, which transcribes line-by-line from the
authors' repository (github.com/danielegirardi/lpdid, fetched 2026-08-19):

  * LP_DiD_R_example_VW.R lines 136-178 — the baseline variance-weighted
    event study, its clean-control filters, and the pooled estimates;
  * LP_DiD_R_example_EW.R lines 138-166 (get_reweights) and 177-198 /
    267-284 — the equally-weighted (reweighted) estimator, with pre
    horizons and the pooled pre using the h=0 weights and the pooled post
    the h=H weights; the same construction ships in the R port
    (github.com/alexCardazzi/lpdid, R/func.R get_weights, lines 45-66);
  * LPDiD_nonabsorbing_example.do lines 187-212 (variant 5b) — the
    non-absorbing clean-control window CCS_h / CCS_mh with stabilization
    lag L, including Stata's missing-value semantics (changes outside the
    observed panel count as clean);
  * the Stata lpdid package's `nevertreated` option — control pool
    restricted to units never treated in the observed sample.

The caveat that keeps this honest: the R leg is a faithful transcription
of the authors' example code re-targeted at these simulated panels, not a
run of the (SSC-only) Stata `lpdid` ado itself. The Stata ado could not
be fetched in this environment (SSC/RePEc egress-blocked); the authors'
GitHub example scripts are their own reference implementations of the
same estimator and are what this fixture pins.

This generator ALSO reimplements the whole estimator independently in
NumPy below and asserts agreement with the R run at 1e-9 relative before
writing the fixture, so the stored numbers are simultaneously a
transcription golden (two independent implementations, two numerical
paths: fixest demeaned WLS vs NumPy explicit algebra).

============================================================================
THE ESTIMATOR (per stored case)
============================================================================
Per post horizon h = 0..H:  regress  y_{i,t+h} - y_{i,t-1}  on the switch
indicator ΔD_it with period fixed effects, on the clean sample; per pre
horizon -j (j = 2..Q):  y_{i,t-j} - y_{i,t-1}  likewise. h = -1 is the
omitted baseline. Clean samples:

  absorbing:      post: ΔD=1 | D_{t+h}=0;   pre: ΔD=1 | D_t=0
  non-absorbing:  ΔD ∈ {0,1} and no status change in [t-L, t-1] (post
                  additionally none in [t+1, t+h]; pre horizon -j widens
                  the lag window to [t-L-(j-1), t-1]); changes outside
                  the panel count as clean (Stata missing semantics)
  never-treated:  control side replaced by "unit never treated in sample"

Weights: OLS = variance-weighted ATT; reweighted = each period cell's
rows weighted by (n1_t + n0_t)/n0_t (drop cells with n1_t = 0), which
equals the reference's inverse switcher-residual-share weights up to an
overall scale that WLS is invariant to; pre-side regressions join the
h=0 weights map. Pooled post: regressand mean(y_{t..t+H}) - y_{t-1} on
the horizon-H clean sample (weights from h=H); pooled pre:
mean(y_{t-Q..t-2}) - y_{t-1} on the pre clean sample (weights from h=0).

Cluster-by-entity SEs in the fixest/reghdfe small-sample convention the
authors fix via setFixest_ssc(ssc(adj=TRUE, cluster.adj=TRUE)):
(n-1)/(n-K) * G/(G-1) with K = 1 + #period cells (all absorbed period
effects counted — verified against fixest at machine precision).

============================================================================
DGPs
============================================================================
Panel A (absorbing, staggered): N = 40, T = 30. Units 0..9 never treated;
unit 10 always treated (treated at t = 0 — no switch event; the estimator
must exclude it from both groups); units 11..39 adopt at dates cycling
{6, 10, 14, 18, 22}. y = unit FE + period FE + N(0, 0.5²) noise + effect;
effect after adoption ramps min(e+1, 4)/4 toward θ_i = 2 + 0.1·(22 - date)
+ N(0, 0.2²) — earlier cohorts have larger effects, so the VW and EW
estimands differ visibly.

Panel B (non-absorbing): N = 50, T = 36, stabilization lag L = 3. 15
never-treated units; the rest enter at a random date in [5, 22], stay a
random 4..9 periods, half exit and some re-enter ≥ 5 periods later.
Effects ramp in over 3 periods while treated and decay to zero within 3
periods of exit (consistent with L = 3).
"""

import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# DGPs
# ---------------------------------------------------------------------------

rng = np.random.default_rng(20260819)


def panel_a():
    n, t_len = 40, 30
    alpha = rng.standard_normal(n)
    delta = rng.standard_normal(t_len)
    y = alpha[:, None] + delta[None, :] + 0.5 * rng.standard_normal((n, t_len))
    d = np.zeros((n, t_len))
    dates = {10: 0}
    cohort = [6, 10, 14, 18, 22]
    for k, i in enumerate(range(11, 40)):
        dates[i] = cohort[k % 5]
    for i, date in dates.items():
        d[i, date:] = 1.0
        theta = 2.0 + 0.1 * (22 - date) + 0.2 * rng.standard_normal()
        for t in range(date, t_len):
            e = t - date
            y[i, t] += theta * min(e + 1, 4) / 4.0
    return y, d


def panel_b():
    n, t_len = 50, 36
    alpha = rng.standard_normal(n)
    delta = rng.standard_normal(t_len)
    y = alpha[:, None] + delta[None, :] + 0.5 * rng.standard_normal((n, t_len))
    d = np.zeros((n, t_len))
    for i in range(15, n):
        entry = int(rng.integers(5, 23))
        dur = int(rng.integers(4, 10))
        exit_ = entry + dur
        spells = [(entry, min(exit_, t_len))]
        if exit_ < t_len - 6 and rng.uniform() < 0.6:
            re_entry = exit_ + int(rng.integers(5, 9))
            if re_entry < t_len:
                spells.append((re_entry, t_len))
        elif exit_ >= t_len:
            spells = [(entry, t_len)]
        theta = 1.5 + rng.uniform(0.0, 1.0)
        for a, b in spells:
            d[i, a:b] = 1.0
        # effects: ramp in over 3 periods while treated; decay within 3 after exit
        eff = np.zeros(t_len)
        since_entry, since_exit = None, None
        for t in range(t_len):
            if d[i, t] == 1.0:
                since_entry = 0 if (t == 0 or d[i, t - 1] == 0.0) else since_entry + 1
                since_exit = None
                eff[t] = theta * min(since_entry + 1, 3) / 3.0
            else:
                if t > 0 and d[i, t - 1] == 1.0:
                    since_exit = 0
                elif since_exit is not None:
                    since_exit += 1
                if since_exit is not None:
                    eff[t] = theta * max(0.0, 1.0 - (since_exit + 1) / 3.0)
        y[i] += eff
    return y, d


YA, DA = panel_a()
YB, DB = panel_b()

CASES = {
    "A_vw": dict(panel="A", q=4, h=6, absorbing=True, lag=0, rw=False, nt=False, pooled=True),
    "A_rw": dict(panel="A", q=4, h=6, absorbing=True, lag=0, rw=True, nt=False, pooled=True),
    "A_nt": dict(panel="A", q=4, h=6, absorbing=True, lag=0, rw=False, nt=True, pooled=True),
    "B_vw": dict(panel="B", q=3, h=4, absorbing=False, lag=3, rw=False, nt=False, pooled=True),
    "B_rw": dict(panel="B", q=3, h=4, absorbing=False, lag=3, rw=True, nt=False, pooled=False),
    "B_nt": dict(panel="B", q=3, h=4, absorbing=False, lag=3, rw=False, nt=True, pooled=False),
}


# ---------------------------------------------------------------------------
# Independent NumPy reimplementation (the transcription leg)
# ---------------------------------------------------------------------------


def lpdid_numpy(y, d, q, h_max, absorbing, lag, rw, nt, pooled):
    n, t_len = y.shape
    dd = np.full((n, t_len), np.nan)
    dd[:, 1:] = d[:, 1:] - d[:, :-1]
    ever = d.max(axis=1) == 1.0
    chg = np.zeros((n, t_len))
    chg[:, 1:] = (np.abs(dd[:, 1:]) == 1.0).astype(float)
    csum = np.concatenate([np.zeros((n, 1)), np.cumsum(chg, axis=1)], axis=1)

    def no_change(i, a, b):
        lo, hi = max(a, 1), min(b, t_len - 1)
        if lo > hi:
            return True
        return csum[i, hi + 1] - csum[i, lo] == 0

    def rows_for(kind, hj):
        rows = []  # (i, t, x, y)
        for i in range(n):
            for t in range(1, t_len):
                if np.isnan(dd[i, t]):
                    continue
                x = dd[i, t]
                if kind in ("post", "pooled_post"):
                    hh = h_max if kind == "pooled_post" else hj
                    if t + hh > t_len - 1:
                        continue
                    if absorbing:
                        ok = x == 1.0 or (
                            x == 0.0 and (not ever[i] if nt else d[i, t + hh] == 0.0)
                        )
                    else:
                        ok = (
                            x >= 0.0
                            and no_change(i, t - lag, t - 1)
                            and no_change(i, t + 1, t + hh)
                            and (x == 1.0 or not nt or not ever[i])
                        )
                    if kind == "post":
                        yv = y[i, t + hh] - y[i, t - 1]
                    else:
                        yv = y[i, t : t + h_max + 1].mean() - y[i, t - 1]
                else:  # pre / pooled_pre
                    jj = q if kind == "pooled_pre" else hj
                    if t < jj:
                        continue
                    if absorbing:
                        ok = x == 1.0 or (
                            x == 0.0 and (not ever[i] if nt else d[i, t] == 0.0)
                        )
                    else:
                        ok = (
                            x >= 0.0
                            and no_change(i, t - lag - (jj - 1), t - 1)
                            and (x == 1.0 or not nt or not ever[i])
                        )
                    if kind == "pre":
                        yv = y[i, t - jj] - y[i, t - 1]
                    else:
                        yv = y[i, t - q : t - 1].mean() - y[i, t - 1]
                if ok:
                    rows.append((i, t, max(x, 0.0), yv))
        return rows

    def cell_weights(rows):
        n1, n0 = np.zeros(t_len), np.zeros(t_len)
        for _, t, x, _ in rows:
            if x == 1.0:
                n1[t] += 1
            else:
                n0[t] += 1
        w = np.full(t_len, np.nan)
        for t in range(t_len):
            if n1[t] > 0:
                assert n0[t] > 0, f"switcher cell without controls at t={t}"
                w[t] = (n1[t] + n0[t]) / n0[t]
        return w

    w0 = cell_weights(rows_for("post", 0)) if rw else None

    def regress(rows, wtable):
        if wtable is not None:
            rows = [r for r in rows if np.isfinite(wtable[r[1]])]
        arr = np.array(rows)
        i_, t_, x_, yv = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2], arr[:, 3]
        w = wtable[t_] if wtable is not None else np.ones(len(rows))
        sw = np.bincount(t_, weights=w, minlength=t_len)
        swx = np.bincount(t_, weights=w * x_, minlength=t_len)
        swy = np.bincount(t_, weights=w * yv, minlength=t_len)
        xt = x_ - swx[t_] / sw[t_]
        yt = yv - swy[t_] / sw[t_]
        sxx = np.sum(w * xt * xt)
        beta = np.sum(w * xt * yt) / sxx
        e = yt - beta * xt
        g = np.bincount(i_, weights=w * xt * e, minlength=n)
        meat = np.sum(g * g)
        nn = len(rows)
        k = 1 + len(np.unique(t_))
        g_cl = len(np.unique(i_))
        var = (nn - 1) / (nn - k) * g_cl / (g_cl - 1) * meat / sxx**2
        return float(beta), float(np.sqrt(var)), int(nn), int(np.sum(x_ == 1.0))

    out = {}
    for hh in range(h_max + 1):
        rows = rows_for("post", hh)
        out[str(hh)] = regress(rows, cell_weights(rows) if rw else None)
    for jj in range(2, q + 1):
        out[str(-jj)] = regress(rows_for("pre", jj), w0)
    if pooled:
        rows = rows_for("pooled_post", None)
        out["pooled_post"] = regress(rows, cell_weights(rows) if rw else None)
        if q >= 2:
            out["pooled_pre"] = regress(rows_for("pooled_pre", None), w0)
    return out


# ---------------------------------------------------------------------------
# Reference leg: run the fixest transcription in R
# ---------------------------------------------------------------------------


def write_csv(path, y, d):
    n, t_len = y.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("unit,time,y,d\n")
        for i in range(n):
            for t in range(t_len):
                f.write(f"{i + 1},{t + 1},{float(y[i, t])!r},{int(d[i, t])}\n")


def run_r(tmp):
    a_csv, b_csv = tmp / "panelA.csv", tmp / "panelB.csv"
    out_csv = tmp / "lpdid_R_out.csv"
    write_csv(a_csv, YA, DA)
    write_csv(b_csv, YB, DB)
    res = subprocess.run(
        ["Rscript", str(OUT / "generate_lpdid_fixtures.R"), str(a_csv), str(b_csv), str(out_csv)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise RuntimeError("R reference run failed")
    print(res.stdout.strip())
    r_out = {}
    with open(out_csv, encoding="utf-8") as f:
        header = f.readline().strip().replace('"', "").split(",")
        idx = {name: k for k, name in enumerate(header)}
        for line in f:
            parts = line.strip().split(",")
            case = parts[idx["case"]].strip('"')
            horizon = parts[idx["horizon"]].strip('"')
            r_out.setdefault(case, {})[horizon] = (
                float(parts[idx["coef"]]),
                float(parts[idx["se"]]),
                int(parts[idx["nobs"]]),
                int(parts[idx["nsw"]]),
            )
    r_version = subprocess.run(
        ["Rscript", "-e", 'cat(R.version.string, "| fixest", as.character(packageVersion("fixest")))'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return r_out, r_version


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        r_out, r_version = run_r(Path(tmpdir))

    max_dev = 0.0
    cases_out = {}
    for name, spec in CASES.items():
        y, d = (YA, DA) if spec["panel"] == "A" else (YB, DB)
        py = lpdid_numpy(
            y, d, spec["q"], spec["h"], spec["absorbing"], spec["lag"], spec["rw"],
            spec["nt"], spec["pooled"],
        )
        ref = r_out[name]
        assert set(py) == set(ref), (name, sorted(py), sorted(ref))
        horizons = {}
        for key in py:
            (b_p, s_p, n_p, w_p), (b_r, s_r, n_r, w_r) = py[key], ref[key]
            for got, want in ((b_p, b_r), (s_p, s_r)):
                dev = abs(got - want) / max(abs(want), 1e-12)
                max_dev = max(max_dev, dev)
                assert dev < 1e-9, (name, key, got, want, dev)
            assert n_p == n_r, (name, key, "nobs", n_p, n_r)
            assert w_p == w_r, (name, key, "n_switchers", w_p, w_r)
            horizons[key] = {"coef": b_r, "se": s_r, "nobs": n_r, "n_switchers": w_r}
        cases_out[name] = {
            "pre_window": spec["q"],
            "post_window": spec["h"],
            "absorbing": spec["absorbing"],
            "nonabsorbing_lag": spec["lag"],
            "reweight": spec["rw"],
            "never_treated_only": spec["nt"],
            "pooled": spec["pooled"],
            "results": horizons,
        }
        est = {k: v["coef"] for k, v in horizons.items() if not k.startswith("pooled")}
        keys = sorted(est, key=int)
        print(f"{name:6s} " + " ".join(f"{k}:{est[k]:+.2f}" for k in keys))
    print(f"max |NumPy - R| relative deviation: {max_dev:.2e}")

    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "r": r_version,
            "reference": "Dube-Girardi-Jordà-Taylor 2025 JAE (doi:10.1002/jae.70000); "
            "authors' example implementations at github.com/danielegirardi/lpdid "
            "(LP_DiD_R_example_VW.R, LP_DiD_R_example_EW.R, "
            "LPDiD_nonabsorbing_example.do, fetched 2026-08-19) run through "
            "fixest by generate_lpdid_fixtures.R; R port consulted: "
            "github.com/alexCardazzi/lpdid R/func.R",
            "numpy_vs_r_max_rel_dev": max_dev,
        },
        "_doc": "reference-run golden: stored values are from an R/fixest run of "
        "the authors' LP-DiD conventions (transcribed line-by-line from their "
        "example code — see generate_lpdid_fixtures.R for the file:line map), "
        "cross-checked here against an independent NumPy reimplementation at "
        "1e-9 before writing. Not a run of the SSC Stata ado (egress-blocked); "
        "the caveat is stated in the generator docstring.",
        "panel_a": {"y": YA.tolist(), "d": DA.tolist()},
        "panel_b": {"y": YB.tolist(), "d": DB.tolist()},
        "cases": cases_out,
    }
    path = OUT / "lpdid.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
