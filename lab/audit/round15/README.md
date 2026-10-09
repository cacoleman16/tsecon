# Audit round 15 — repro scripts

The minimal reproduction behind each finding in
[`docs/roadmap/30-audit-round-15-findings.md`](../../../docs/roadmap/30-audit-round-15-findings.md),
one directory per finder lens. Each script is the version that showed the
defect on the wheel built from `3431249`, which was the PR #30 head when the
round ran.

After the fixes, a script prints the corrected behaviour (or the named
refusal) instead of the defect. The refuted findings' scripts still
reproduce what was observed. The findings document explains why each of
those is not a defect.

Run them from the repository root with the reference environment the
fixtures pin (numpy 2.5.3, scipy 1.18.1, statsmodels 0.15.0, arch 8.0.0,
linearmodels 7.0, pandas 3.0.6) and an installed `tsecon` wheel:

```sh
python lab/audit/round15/fuzz_a/repro_ardl_case2_F.py
```

| Directory | Lens | Findings |
|---|---|---|
| `fuzz_a/` | Differential fuzzing against arch / statsmodels | F1, F2, F3 |
| `fuzz_b/` | Differential fuzzing against linearmodels / statsmodels / closed forms | B15-UC-1, B15-UC-2, B15-EWC-1, B15-VAR-1 (refuted). `augkf.py` is the auditor's extended-precision augmented Kalman filter reference. `uc_pmag*.py` replay the old filter's cancellation test. |
| `hostile/` | Malformed-input sweep | SWEEP-1 … SWEEP-10 |
| `docs_truth/` | Documentation truth | DOC15-1 … DOC15-7 |
| `api/` | API consistency | API15-1 … API15-6. `repro_vr_lags_stub_type.sh` runs `mypy --strict` on `stubcheck/`. |
| `mc_claims/` | Statistical-claim re-measurement | EWC-PRECISION-1, UC-CYCLE-1, and the refuted VR-SIZE-1, PANEL-PETERSEN-1, ARDL-SIZE-1, BG-NOCONST-1 |

The 90-digit exact-diffuse reference used to verify the Kalman filter fix
is committed separately at `scripts/uc_exact_diffuse_reference.py`.
