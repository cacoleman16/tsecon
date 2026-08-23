#!/bin/bash
set -u
cd "$(dirname "$0")/../.." || exit 9
PY=".venv-wt/bin/python"
$PY - <<'EOF'
import tsecon
d = tsecon.theta_forecast.__doc__
assert "use_test=False" in d, "theta doc missing qualifier"
for f in (tsecon.pseudo_obs, tsecon.copula_fit):
    doc = f.__doc__ or ""
    assert "strictly monotone transform" not in doc, f.__name__
    assert "increasing" in doc, f.__name__
d = tsecon.acm_term_premium.__doc__
for k in ("maturities", "n_factors", "periods_per_year"):
    assert k in d, k
print("rebuilt docstrings carry the fixes")
EOF
s1=$?
$PY -m pytest bindings/python/tests/test_copula.py bindings/python/tests/test_acm.py \
    bindings/python/tests/test_docstring_keys.py -q 2>&1 | tail -6
s2=${PIPESTATUS[0]}
echo "DOC_CHECK_EXIT=$s1 PYTEST_EXIT=$s2"
exit $(( s1 || s2 ))
