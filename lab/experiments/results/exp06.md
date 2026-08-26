# exp06 — conformal interval wrappers (split / EnbPI / ACI)

One-step-ahead intervals at nominal 90%; all methods share the AR(1) least-squares base.

## Setting A — GARCH(1,1)-t returns (T=500, eval=150, R=100)

| method | coverage | se(mean) | median width | Kupiec rej @5% |
|---|---|---|---|---|
| split | 0.8999 | 0.0039 | 2.8834 | 0.2600 |
| aci (gamma=.005) | 0.9054 | 0.0027 | 2.8926 | 0.0700 |
| enbpi (B=25) | 0.8945 | 0.0049 | 2.8979 | 0.3300 |

## Setting B — variance shift sd 1→3 inside the eval window (T=400, eval=120, post-shift 80, R=100)

| method | post-shift coverage | se(mean) | full-window coverage |
|---|---|---|---|
| split | 0.7052 | 0.0038 | 0.7696 |
| aci (gamma=.005) | 0.7938 | 0.0026 | 0.8309 |
| aci (gamma=.05) | 0.8919 | 0.0016 | 0.8949 |
| enbpi (B=25) | 0.5095 | 0.0040 | 0.6318 |

_runtime 10 s_
