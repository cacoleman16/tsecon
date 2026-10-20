"""The `var_forecast(param_uncertainty=True)` work budget on WIDE VARs, where
the `2 steps m^3` term (m = n_trend + k*lags regressors per equation) and
not the cubic-in-`steps` term trips it (audit round 15, SWEEP-8).

For each (k, lags, T) below, on seeded iid N(0, 1) data (the cost does not
depend on the values):

* the work count of `crates/tsecon-var/src/forecast.rs` per step, split into
  the steps-driven terms `steps^3 (k^2 + 8) / 3 + steps^2 k^3 / 2` and the
  m-driven terms `2 steps m^3 + steps^2 m^2`;
* the largest `steps` the 2^34 budget admits by that count, the call timed
  there (best of 3), and the refusal one step past it — which parameter it
  names and which branch of the message it takes;
* for a VAR where not even one step fits, the refusal naming `lags`.

Run from the repository root with the release extension importable:

    python lab/audit/wave11/var_param_uncertainty_wide.py \
        | tee lab/audit/wave11/out/var_param_uncertainty_wide.txt
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import tsecon

BUDGET = 1 << 34


def terms(steps: int, k: int, m: int) -> tuple[int, int]:
    horizon = steps**3 * (k * k + 8) // 3 + steps * steps * k**3 // 2
    regressors = 2 * steps * m**3 + steps * steps * m * m
    return horizon, regressors


def largest_steps(k: int, m: int) -> int:
    s = 0
    while sum(terms(s + 1, k, m)) <= BUDGET:
        s += 1
    return s


def refusal(data: np.ndarray, lags: int, steps: int) -> str:
    try:
        tsecon.var_forecast(data, lags=lags, steps=steps, param_uncertainty=True)
    except ValueError as exc:
        return str(exc)
    return "NOT REFUSED"


def timed(data: np.ndarray, lags: int, steps: int) -> float:
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        tsecon.var_forecast(data, lags=lags, steps=steps, param_uncertainty=True)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    rng = np.random.default_rng(20261009)
    out: dict = {"tsecon": tsecon.__version__, "budget": BUDGET, "cases": []}
    for k, lags, t_obs in ((60, 20, 3000), (30, 10, 400), (3, 2, 300)):
        m = 1 + k * lags
        data = rng.standard_normal((t_obs, k))
        cap = largest_steps(k, m)
        h, r = terms(cap + 1, k, m)
        msg = refusal(data, lags, cap + 1)
        case = {
            "k": k, "lags": lags, "T": t_obs, "m": m,
            "largest_admitted_steps": cap,
            "work_at_cap_over_budget": sum(terms(cap, k, m)) / BUDGET,
            "seconds_at_cap": timed(data, lags, cap),
            "refused_steps": cap + 1,
            "steps_terms_over_budget": h / BUDGET,
            "m_terms_over_budget": r / BUDGET,
            "refusal_names": msg.split(" = ")[0].removeprefix("invalid "),
            "refusal_branch": ("m terms" if "the m terms dominate" in msg
                               else "cubic in steps" if "CUBIC in steps" in msg else "?"),
            "refusal": msg,
        }
        out["cases"].append(case)
        print(f"k={k} lags={lags} T={t_obs} m={m}: largest admitted steps {cap} "
              f"(work {case['work_at_cap_over_budget']:.3f} x budget, "
              f"{case['seconds_at_cap']:.2f} s); at steps={cap + 1} the steps terms are "
              f"{case['steps_terms_over_budget']:.4f} x budget and the m terms "
              f"{case['m_terms_over_budget']:.4f} x -> refused naming "
              f"{case['refusal_names']} ({case['refusal_branch']})")
        print(f"    {msg}")
    # m = 1 + 64*32 = 2049: 2 m^3 alone exceeds 2^34, so no horizon fits.
    k, lags, t_obs = 64, 32, 2200
    data = rng.standard_normal((t_obs, k))
    msg = refusal(data, lags, 1)
    out["no_horizon_fits"] = {"k": k, "lags": lags, "T": t_obs, "m": 1 + k * lags,
                              "work_at_one_step_over_budget": sum(terms(1, k, 1 + k * lags)) / BUDGET,
                              "refusal": msg}
    print(f"k={k} lags={lags} T={t_obs} m={1 + k * lags}: one step is "
          f"{out['no_horizon_fits']['work_at_one_step_over_budget']:.4f} x budget ->")
    print(f"    {msg}")
    path = Path(__file__).resolve().parent / "out" / "var_param_uncertainty_wide.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {path.relative_to(Path(__file__).resolve().parents[3])}")


if __name__ == "__main__":
    main()
