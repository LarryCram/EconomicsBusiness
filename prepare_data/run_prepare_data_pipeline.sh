#!/usr/bin/env bash
# run_pipeline.sh — Run the prepare_data pipeline stages in order.
# Must be run from the project root: bash prepare_data/run_pipeline.sh

set -euo pipefail

PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Stage 1: Assemble journal registry ==="
"$PYTHON" prepare_data/journal_assembler_era_harzing_wos_scopus.py

echo "=== Stage 2: Filter and match to OpenAlex ==="
"$PYTHON" prepare_data/journal_filter_match_oa.py

echo "=== Stage 3: Load corpus entities ==="
"$PYTHON" prepare_data/load_corpus_entities.py

echo "=== Stage 4: Build edge lists ==="
"$PYTHON" prepare_data/build_edge_lists.py

echo "=== Stage 5: Verify edge lists ==="
"$PYTHON" prepare_data/verify_edge_lists.py

echo "=== Stage 6: Build institution field classification ==="
"$PYTHON" prepare_data/build_institution_field_eb.py

echo "=== Stage 7: Compute mode-specific SCC unit tables ==="
"$PYTHON" prepare_data/filter_mode_units.py

echo "=== Pipeline complete ==="
