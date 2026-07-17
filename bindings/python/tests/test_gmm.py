"""Golden test for the IV-GMM binding against linearmodels.

fixtures/gmm.json holds an IVGMM(y ~ [const, w] + x endog, instruments
[z1, z2]) two-step robust fit from linearmodels; the crate reproduces it
to machine precision (see the crate's own golden test).
"""
import json
from pathlib import Path

import numpy as np
import tsecon

FIXTURES = Path(__file__).parents[3] / "fixtures"
GMM = json.loads((FIXTURES / "gmm.json").read_text())

Y = np.array(GMM["y"])
XV = np.array(GMM["x"])
W = np.array(GMM["w"])
Z1 = np.array(GMM["z1"])
Z2 = np.array(GMM["z2"])
ONES = np.ones_like(Y)

# Regressors X = [const, w, x] (x is endogenous); instruments
# Z = [const, w, z1, z2] (exogenous regressors instrument themselves).
X = np.column_stack([ONES, W, XV])
Z = np.column_stack([ONES, W, Z1, Z2])

ORDER = GMM["ivgmm"]["param_order"]  # ["const", "w", "x"]
PARAMS = [GMM["ivgmm"]["params"][k] for k in ORDER]
BSE = [GMM["ivgmm"]["bse"][k] for k in ORDER]


def test_iv_gmm_two_step_robust_matches_linearmodels():
    fit = tsecon.iv_gmm(X, Z, Y, method="2step", weight="robust")
    np.testing.assert_allclose(fit["params"], PARAMS, atol=1e-6)
    np.testing.assert_allclose(fit["bse"], BSE, atol=1e-6)
    # Over-identified (4 instruments, 3 params) -> Hansen J with 1 dof.
    assert fit["j_dof"] == 1
    assert abs(fit["j_stat"] - GMM["ivgmm"]["j_stat"]) < 1e-4
    assert abs(fit["j_pval"] - GMM["ivgmm"]["j_pval"]) < 1e-4
    assert fit["nparams"] == 3
    assert fit["nmoments"] == 4
    assert fit["steps"] == 2


def test_exactly_identified_gmm_equals_2sls():
    # Drop z2 so the model is exactly identified (3 instruments, 3 params);
    # GMM then coincides with 2SLS regardless of the weighting matrix.
    z_exact = np.column_stack([ONES, W, Z1])
    two_step = tsecon.iv_gmm(X, z_exact, Y, method="2step", weight="robust")
    tsls = tsecon.iv_gmm(X, z_exact, Y, method="2sls")
    np.testing.assert_allclose(two_step["params"], tsls["params"], atol=1e-9)
    # Exactly identified -> no over-identifying restrictions to test.
    assert "j_stat" not in tsls
