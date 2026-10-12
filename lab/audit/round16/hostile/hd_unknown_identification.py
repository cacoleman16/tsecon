"""historical_decomposition: an unknown identification string combined with
impact= is refused with a FALSE explanation -- it says the (nonexistent) mode
'identifies its own shocks and would ignore' impact -- instead of 'unknown
identification'. The typo case (identification="Impact") is exactly the one
where impact is passed, so the user is told to drop impact rather than fix the
spelling. Without impact the same string gets the right message."""
import numpy as np, tsecon
rng = np.random.default_rng(1); X = np.zeros((200, 2))
for t in range(1, 200): X[t] = 0.5 * X[t-1] + rng.standard_normal(2)
A0 = tsecon.long_run_svar(X)["impact"]
for ident in ["Impact", "impact ", "impcat"]:
    for kw in ({"impact": A0}, {}):
        try:
            tsecon.historical_decomposition(X, identification=ident, **kw)
            print(repr(ident), bool(kw), "ACCEPTED")
        except ValueError as e:
            print(f"identification={ident!r} impact={'given' if kw else 'absent'}:\n   {str(e)[:170]}")
print("expected (both cases): 'unknown identification ...; expected \"cholesky\"|\"impact\"|\"sign\"'")
