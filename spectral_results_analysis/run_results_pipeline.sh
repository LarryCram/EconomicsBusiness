#!/usr/bin/env bash
# run_results_pipeline.sh — Run all figure and analysis scripts in order.
# Must be run from the project root: bash spectral_results_analysis/run_results_pipeline.sh

set -euo pipefail

PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(dirname "$0")/../.venv/bin/python}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Export baseline rankings (all field filters) ==="
"$PYTHON" spectral_results_analysis/export_baseline_rankings.py

echo "=== Communities: source citation networks ==="
"$PYTHON" spectral_results_analysis/source_communities.py

echo "=== Fig 2: field scope sensitivity ==="
"$PYTHON" spectral_results_analysis/fig_2.py

echo "=== Fig 3: rank curves, cross-mask scatter, fig_3b ==="
"$PYTHON" spectral_results_analysis/fig_3.py

echo "=== Fig 4: community identity phi2/sqrt(v) ==="
"$PYTHON" spectral_results_analysis/fig_4.py

echo "=== Fig 5 ==="
"$PYTHON" spectral_results_analysis/fig_5.py

echo "=== Fig 6 ==="
"$PYTHON" spectral_results_analysis/fig_6.py

# echo "=== Fig 7: bootstrap stability ==="
# "$PYTHON" spectral_results_analysis/fig_7.py

echo "=== Fig 8 ==="
"$PYTHON" spectral_results_analysis/fig_8.py

echo "=== Fig stability ==="
"$PYTHON" spectral_results_analysis/fig_stability.py

echo "=== Fig log-ratio stability ==="
"$PYTHON" spectral_results_analysis/fig_log_ratio_stability.py

echo "=== Network entropy diagnostics ==="
"$PYTHON" spectral_results_analysis/network_entropy.py

echo "=== Column structure diagnostics ==="
"$PYTHON" spectral_results_analysis/column_structure.py

echo "=== Cross-mask influence gain ==="
"$PYTHON" spectral_results_analysis/cross_mask_gain.py

echo "=== Compare field modes ==="
"$PYTHON" spectral_results_analysis/compare_field_modes.py

echo "=== Table: kernel structure ==="
"$PYTHON" spectral_results_analysis/table_kernel_structure.py

echo "=== Fig: J_bar histogram ==="
"$PYTHON" spectral_results_analysis/fig_jbar_hist.py

echo "=== Diag: epsilon=1 pi comparison ==="
"$PYTHON" spectral_results_analysis/diag_epsilon_pi.py

echo "=== Analyse: enclave / anti-ranking sources ==="
"$PYTHON" spectral_results_analysis/analyse_enclaves.py

echo "=== Analyse: Meta A+ outliers ==="
"$PYTHON" spectral_results_analysis/analyse_meta_outliers.py

echo "=== Results pipeline complete ==="
