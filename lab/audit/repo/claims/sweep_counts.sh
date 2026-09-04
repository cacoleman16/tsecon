#!/usr/bin/env bash
# Recompute every count the repository states about itself, with the repo's own
# rules (docs/reference/testing.md names them). Prints name=value lines; the
# ledger in docs/roadmap/_repo_audit/claims.md quotes them.
#
# Run from the repository root:  bash lab/audit/repo/claims/sweep_counts.sh
set -u
cd "$(dirname "$0")/../../../.." || exit 1
PY=${CLAIMS_PYTHON:-.venv-wt/bin/python}

echo "version_workspace=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')"
echo "version_pyproject=$(grep -m1 '^version' bindings/python/pyproject.toml | sed 's/.*"\(.*\)"/\1/')"
echo "version_citation_cff=$(grep -m1 '^version:' CITATION.cff | awk '{print $2}')"
echo "version_wheel=$($PY -c 'import tsecon; print(tsecon.__version__)')"
echo "rust_toolchain=$(grep channel rust-toolchain.toml | sed 's/.*"\(.*\)"/\1/')"
echo "rust_version_min=$(grep -m1 'rust-version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')"
echo "requires_python=$(grep -m1 'requires-python' bindings/python/pyproject.toml | sed 's/.*"\(.*\)"/\1/')"
echo "maturin_req=$(grep -m1 'requires = ' bindings/python/pyproject.toml | sed 's/.*\["\(.*\)"\].*/\1/')"
echo "numpy_dep=$(grep -m1 'dependencies = ' bindings/python/pyproject.toml | sed 's/.*\["\(.*\)"\].*/\1/')"

echo "public_callables=$($PY -c 'import tsecon; print(sum(callable(getattr(tsecon, n)) for n in dir(tsecon) if not n.startswith("_")))')"
echo "crates_dirs=$(ls -d crates/*/ | wc -l)"
echo "crates_with_tests_dir=$(ls -d crates/*/tests 2>/dev/null | wc -l)"
echo "workspace_members=$(sed -n '/^members/,/^\]/p' Cargo.toml | grep -c '"')"
echo "rust_src_lines=$(find crates -path '*/src/*' -name '*.rs' | xargs cat | wc -l)"
echo "rust_all_lines=$(find crates -name '*.rs' | xargs cat | wc -l)"
echo "fixtures_json=$(ls fixtures/*.json | wc -l)"
echo "fixture_generators=$(ls fixtures/generate_*.py | wc -l)"
echo "fixture_generators_R=$(ls fixtures/generate_*.R 2>/dev/null | wc -l)"
echo "fixture_csv=$(ls fixtures/*.csv | wc -l)"
echo "python_test_files=$(ls bindings/python/tests/test_*.py | wc -l)"
echo "python_tests_collected=$($PY -m pytest bindings/python/tests --collect-only -q 2>/dev/null | tail -1)"
echo "python_results_test_files=$(ls bindings/python/tests/test_results_*.py | wc -l)"
echo "python_results_tests_collected=$($PY -m pytest bindings/python/tests/test_results_*.py --collect-only -q 2>/dev/null | tail -1)"
echo "python_replication_test_files=$(ls bindings/python/tests/test_replication_*.py | wc -l)"
echo "python_replication_tests_collected=$($PY -m pytest bindings/python/tests/test_replication_*.py --collect-only -q 2>/dev/null | tail -1)"
echo "pytest_raises_calls=$(grep -o 'pytest.raises(' bindings/python/tests/*.py | wc -l)"
echo "fixtures_named_in_python_tests=$(for j in $(grep -oh '[A-Za-z0-9_-]*\.json' bindings/python/tests/*.py | sort -u); do [ -f fixtures/$j ] && echo $j; done | wc -l)"
echo "rust_static_tests_integration=$(grep -c '#\[test\]' crates/*/tests/*.rs | awk -F: '{s+=$NF} END {print s}')"
echo "rust_static_tests_unit=$(grep -rc '#\[test\]' crates/*/src | awk -F: '{s+=$NF} END {print s}')"
echo "rust_static_ignored=$(grep -rh '#\[ignore' crates/*/tests/*.rs crates/*/src | grep -v '^\s*//' | wc -l)"
for kind in golden propert validation; do
  echo "rust_${kind}_tests=$(grep -c '#\[test\]' crates/*/tests/*${kind}*.rs | awk -F: '{s+=$NF} END {print s}')"
  echo "rust_${kind}_files=$(ls crates/*/tests/*${kind}*.rs | wc -l)"
  echo "rust_${kind}_crates=$(ls crates/*/tests/*${kind}*.rs | awk -F/ '{print $2}' | sort -u | wc -l)"
done
echo "crates_depending_on_hac=$(grep -l '^tsecon-hac' crates/*/Cargo.toml | wc -l)"
echo "guide_chapters=$(ls docs/guide/[0-9]*.md | wc -l)"
echo "cookbook_recipes=$(ls docs/cookbook/*.md | grep -v README | wc -l)"
echo "replication_pages=$(ls docs/examples/replication-*.md | wc -l)"
echo "replication_scripts=$(ls docs/examples/replication_*.py | wc -l)"
echo "model_cards=$(ls docs/reference/model-cards/*.md | wc -l)"
echo "model_cards_in_reference_readme=$(grep -c 'model-cards/' docs/reference/README.md)"
echo "model_cards_in_nav=$(grep -c 'model-cards/' mkdocs.yml)"
echo "coverage_family_modules=$(ls docs/examples/coverage/*.py | grep -v -e run_all -e check_page | wc -l)"
echo "validation_matrix_rows=$(grep -c '^| ' docs/reference/validation-matrix.md)"
echo "roadmap_module_specs=$(ls docs/roadmap/[0-9][0-9]-*.md | grep -v -i audit | wc -l)"
echo "benchmark_ops=$(grep -c '^| .* | `tsecon\.' benchmarks/README.md)"
echo "generators_importing_tsecon=$(grep -l 'import tsecon' fixtures/*.py | tr '\n' ' ')"
echo "testing_md_tier_headings=$(grep -c '^### Tier' docs/reference/testing.md)"
echo "testing_md_file_table_rows=$(grep -c '^| `test_' docs/reference/testing.md)"
