"""Golden fixtures for the arbitrage-free Nelson-Siegel (AFNS) yield-adjustment
term (roadmap E2, Christensen, Diebold & Rudebusch 2011).

VALIDATION-FIRST / NON-CIRCULAR: this generator does NOT call the tsecon Rust
crate. It transcribes the DOCUMENTED CLOSED-FORM independent-factor
yield-adjustment term from

    Christensen, J. H. E., Diebold, F. X., & Rudebusch, G. D. (2011).
    "The affine arbitrage-free class of Nelson-Siegel term structure models."
    Journal of Econometrics, 164(1), 4-20.

directly into NumPy and evaluates it on a grid of maturities / lambda / sigma.
The Rust crate is expected to reproduce these numbers to ~1e-10.

The independent-factor AFNS keeps the three Nelson-Siegel factor loadings

    y(tau) = L + S*(1 - e^{-lam*tau})/(lam*tau)
               + C*[(1 - e^{-lam*tau})/(lam*tau) - e^{-lam*tau}]  -  A(tau)/tau

and ADDS a deterministic, maturity-dependent yield-adjustment term
`-A(tau)/tau` (denoted `-C(tau)/tau` in the crate) that makes the curve
arbitrage-free. With a DIAGONAL volatility matrix Sigma = diag(s11, s22, s33)
(the "independent-factor" AFNS), CDR (2011) give the closed form

  A(tau)/tau =
      s11^2 * ( tau^2 / 6 )

    + s22^2 * [ 1/(2*lam^2)
                - (1 - e^{-lam*tau}) / (lam^3 * tau)
                + (1 - e^{-2*lam*tau}) / (4 * lam^3 * tau) ]

    + s33^2 * [ 1/(2*lam^2)
                + e^{-lam*tau} / lam^2
                - (tau * e^{-2*lam*tau}) / (4*lam)
                - 3 * e^{-2*lam*tau} / (4*lam^2)
                - 2 * (1 - e^{-lam*tau}) / (lam^3 * tau)
                + 5 * (1 - e^{-2*lam*tau}) / (8 * lam^3 * tau) ]

`A(tau)/tau` is non-negative and (via the s11 tau^2/6 term) grows without bound
in maturity, so the signed adjustment `-A(tau)/tau` added to the yields is
negative and its magnitude grows with maturity -- the arbitrage-free
concavity/convexity effect. As Sigma -> 0 the adjustment vanishes and AFNS
nests plain Nelson-Siegel exactly.

Run with the project venv:
    .venv/bin/python fixtures/generate_afns_fixtures.py
"""
import json
import platform
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
full = lambda a: [float(x) for x in np.asarray(a, dtype=float).ravel()]


def afns_c_over_tau(maturities, lam, sigma_diag):
    """A(tau)/tau, the CDR (2011) independent-factor yield-adjustment term.

    Transcribed verbatim from the closed form above; never calls tsecon.
    Returns the POSITIVE term A(tau)/tau; the signed yield adjustment added to
    the curve is its negation, -A(tau)/tau.
    """
    tau = np.asarray(maturities, dtype=float)
    s11, s22, s33 = (float(s) for s in sigma_diag)
    e1 = np.exp(-lam * tau)
    e2 = np.exp(-2.0 * lam * tau)

    term11 = tau ** 2 / 6.0

    term22 = (
        1.0 / (2.0 * lam ** 2)
        - (1.0 - e1) / (lam ** 3 * tau)
        + (1.0 - e2) / (4.0 * lam ** 3 * tau)
    )

    term33 = (
        1.0 / (2.0 * lam ** 2)
        + e1 / (lam ** 2)
        - (tau * e2) / (4.0 * lam)
        - 3.0 * e2 / (4.0 * lam ** 2)
        - 2.0 * (1.0 - e1) / (lam ** 3 * tau)
        + 5.0 * (1.0 - e2) / (8.0 * lam ** 3 * tau)
    )

    return s11 ** 2 * term11 + s22 ** 2 * term22 + s33 ** 2 * term33


def gen_afns():
    # A grid of (maturities, lambda, sigma) cases. Maturities in years.
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

    cases_spec = [
        # Diebold-Li-ish decay with modest independent factor vols.
        (0.5, [0.010, 0.008, 0.012]),
        # Faster decay, only the level factor carries volatility.
        (0.7, [0.015, 0.0, 0.0]),
        # Only slope volatility.
        (0.6, [0.0, 0.020, 0.0]),
        # Only curvature volatility.
        (0.6, [0.0, 0.0, 0.020]),
        # A larger-vol, slower-decay case to exercise all three terms.
        (0.3, [0.007, 0.011, 0.009]),
        # Sigma -> 0 exactly: the adjustment must be all zeros (NS nesting).
        (0.5, [0.0, 0.0, 0.0]),
    ]

    cases = []
    for lam, sigma in cases_spec:
        c = afns_c_over_tau(maturities, lam, sigma)
        cases.append({
            "maturities": full(maturities),
            "lambda": float(lam),
            "sigma_diag": [float(s) for s in sigma],
            # Positive documented term A(tau)/tau.
            "c_over_tau": full(c),
            # Signed adjustment added to the curve, -A(tau)/tau (<= 0).
            "adjustment": full(-c),
        })

    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "reference": "Christensen, Diebold & Rudebusch (2011), J. Econometrics "
                         "164(1), 4-20; independent-factor AFNS yield-adjustment "
                         "term A(tau)/tau.",
            "note": "DOCUMENTED-FORMULA golden (non-circular; no tsecon call). "
                    "c_over_tau is the positive CDR term A(tau)/tau; adjustment is "
                    "the signed yield adjustment -A(tau)/tau added to the curve. "
                    "sigma_diag=[s11,s22,s33] are the diagonal factor volatilities.",
        },
        "cases": cases,
    }
    (OUT / "afns.json").write_text(json.dumps(out, separators=(",", ":")))
    print("wrote afns.json")


if __name__ == "__main__":
    gen_afns()
