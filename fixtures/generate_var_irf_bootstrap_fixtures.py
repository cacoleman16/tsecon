"""Golden/reference fixtures for the bootstrap VAR-IRF confidence bands.

Run:  python3 fixtures/generate_var_irf_bootstrap_fixtures.py

This generator imports **statsmodels only** (never tsecon): it emits the
shared DGP plus the statsmodels *asymptotic* impulse-response standard
errors (`irf.stderr`, `irf.cum_effect_stderr`) that the Rust bootstrap test
compares its bootstrap SEs against for a magnitude sanity check, and a loose
Monte-Carlo error band (`irf.errband_mc`, whose RNG differs from tsecon's so
it is *not* bit-matchable — sanity only).

The reproducibility pin — tsecon's own seeded point + bands under a fixed
`(seed, n_boot, ...)` config — is produced by the Rust crate and stored
under the top-level `tsecon_snapshot` key. This script PRESERVES any
existing `tsecon_snapshot` when it re-dumps the file, so regenerating the
statsmodels references never clobbers the Rust snapshot (Python's json and
serde_json both round-trip f64 via the shortest exact decimal, so the bits
survive the merge).

Regenerating requires the pinned reference versions recorded in `_meta`.
"""
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

OUT = Path(__file__).parent
FIXTURE = OUT / "var_irf_bootstrap.json"

META = {
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "statsmodels": statsmodels.__version__,
    "python": platform.python_version(),
}

# The shared DGP: 100 * dlog of US real GDP / consumption / investment,
# a VAR(2) with a constant — identical to fixtures/var.json so the point
# IRFs already agree with tsecon.var_irf.
HORIZON = 10

# The configuration the Rust reproducibility snapshot is pinned at (the Rust
# test reads these back so the two sides never drift).
SNAPSHOT_PARAMS = {
    "lags": 2,
    "trend": "c",
    "horizon": 4,
    "orth": True,
    "cumulative": False,
    "alpha": 0.10,
    "n_boot": 300,
    "seed": 20260722,
    "bias_correct": False,
}


def main() -> None:
    mac = sm.datasets.macrodata.load_pandas().data
    data = 100.0 * np.diff(
        np.log(mac[["realgdp", "realcons", "realinv"]].to_numpy()), axis=0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VAR(data).fit(2, trend="c")
        irf = res.irf(HORIZON)
        se_orth = np.asarray(irf.stderr(orth=True))
        se_nonorth = np.asarray(irf.stderr(orth=False))
        cum_se_orth = np.asarray(irf.cum_effect_stderr(orth=True))
        cum_se_nonorth = np.asarray(irf.cum_effect_stderr(orth=False))
        # Loose MC band (statsmodels' own bootstrap; different RNG from
        # tsecon, so this is a sanity reference only).
        mc_lo, mc_hi = irf.errband_mc(orth=True, repl=1000, signif=0.10, seed=987654321)

    out = {
        "_meta": META,
        "_doc": (
            "Bootstrap VAR-IRF band references. statsmodels asymptotic SEs "
            "(delta method) for magnitude sanity; errband_mc for a loose, "
            "non-bit-matchable MC band. tsecon_snapshot (Rust-produced) pins "
            "reproducibility."
        ),
        "data_100dlog_gdp_cons_inv": data.tolist(),
        "horizon": HORIZON,
        "asymptotic_se_orth": se_orth.tolist(),
        "asymptotic_se_nonorth": se_nonorth.tolist(),
        "cum_asymptotic_se_orth": cum_se_orth.tolist(),
        "cum_asymptotic_se_nonorth": cum_se_nonorth.tolist(),
        "point_orth_h10": np.asarray(irf.orth_irfs).tolist(),
        "point_nonorth_h10": np.asarray(irf.irfs).tolist(),
        "errband_mc_orth_signif10": {
            "repl": 1000,
            "seed": 987654321,
            "lower": np.asarray(mc_lo).tolist(),
            "upper": np.asarray(mc_hi).tolist(),
        },
        "snapshot_params": SNAPSHOT_PARAMS,
    }

    # Preserve the Rust-produced reproducibility snapshot across regenerations.
    if FIXTURE.exists():
        try:
            prev = json.loads(FIXTURE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
        if "tsecon_snapshot" in prev:
            out["tsecon_snapshot"] = prev["tsecon_snapshot"]

    FIXTURE.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)")
    if "tsecon_snapshot" not in out:
        print(
            "NOTE: tsecon_snapshot not yet present -- run the Rust snapshot "
            "emitter (cargo test -p tsecon-var emit_snapshot -- --nocapture "
            "--ignored) and merge its JSON under `tsecon_snapshot`."
        )


if __name__ == "__main__":
    main()
