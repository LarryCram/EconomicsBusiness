#!/usr/bin/env bash
# run_results_pipeline.sh — Run all figure and analysis scripts in order.
# Must be run from the project root: bash spectral_results_analysis/run_results_pipeline.sh

set -euo pipefail

PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Communities: source citation networks ==="
"$PYTHON" spectral_results_analysis/source_communities.py

echo "=== Fig 2: field scope sensitivity ==="
"$PYTHON" spectral_results_analysis/fig_2.py

echo "=== Fig 3: vSS vs vSB / vII vs vIB ==="
"$PYTHON" spectral_results_analysis/fig_3.py

echo "=== Fig 4: community identity phi2/sqrt(v) ==="
"$PYTHON" spectral_results_analysis/fig_4.py

echo "=== Fig 5 ==="
"$PYTHON" spectral_results_analysis/fig_5.py

echo "=== Fig 6 ==="
"$PYTHON" spectral_results_analysis/fig_6.py

echo "=== Fig 7: bootstrap stability ==="
"$PYTHON" spectral_results_analysis/fig_7.py

echo "=== Fig 8 ==="
"$PYTHON" spectral_results_analysis/fig_8.py

echo "=== Fig stability ==="
"$PYTHON" spectral_results_analysis/fig_stability.py

echo "=== Fig log-ratio stability ==="
"$PYTHON" spectral_results_analysis/fig_log_ratio_stability.py

echo "=== Results pipeline complete ==="
