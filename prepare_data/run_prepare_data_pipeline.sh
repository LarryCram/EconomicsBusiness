#!/usr/bin/env bash
# run_pipeline.sh — Run the three prepare_data stages in order.
# Must be run from the project root: bash prepare_data/run_pipeline.sh

set -euo pipefail

PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Stage 1: Assemble journal registry ==="
"$PYTHON" prepare_data/journal_assembler_era_harzing_wos.py

echo "=== Stage 2: Filter and match to OpenAlex ==="
"$PYTHON" prepare_data/journal_filter_match_oa.py

echo "=== Stage 3: Build tables ==="
"$PYTHON" prepare_data/table_maker.py

echo "=== Pipeline complete ==="
