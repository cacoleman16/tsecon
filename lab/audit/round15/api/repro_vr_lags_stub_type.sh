#!/bin/sh
# Runs mypy --strict (as CI's typecheck job does) on a documented variance_ratio call.
cd "$(dirname "$0")"
python -m mypy --strict --ignore-missing-imports stubcheck/check_vr_lags_typing.py
# runtime accepts it:
python -c "import numpy as np, tsecon; print('runtime:', tsecon.variance_ratio(np.cumsum(np.random.default_rng(0).standard_normal(300)), lags=np.array([2, 4, 8]))['vr'])"
