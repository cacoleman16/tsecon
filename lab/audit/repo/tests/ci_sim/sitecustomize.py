"""Import blocker that simulates CI's python job environment.

.github/workflows/ci.yml installs only `pip pytest numpy scipy` before running
`pytest bindings/python/tests` against the wheel, so every importorskip-gated
test (statsmodels, arch, scikit-learn, linearmodels, mapie, pandas, matplotlib)
skips there. Put this directory on PYTHONPATH and the same venv reports what CI
actually runs, without rebuilding a wheel:

    PYTHONPATH=lab/audit/repo/tests/ci_sim .venv-wt/bin/python -m pytest bindings/python/tests -q -rs

(polars is not in CI either; it is blocked too.)
"""
import sys

BLOCKED = {"statsmodels", "arch", "sklearn", "linearmodels", "mapie", "pandas", "matplotlib", "polars"}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked to simulate CI: {name}")
        return None


sys.meta_path.insert(0, _Blocker())
