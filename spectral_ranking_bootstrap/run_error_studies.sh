#!/usr/bin/env bash
# run_error_studies.sh — re-run the bootstrap and metadata-error studies used in the paper.
#
#   stages 1,2,3 (sequential), then in parallel:
#     stage 4 --filters resample | source | institution | reference, and year_v2
#
# Usage (from the project root; detached, survives logout):
#   ECONBUS_CONFIG=config.rev.yaml setsid nohup spectral_ranking_bootstrap/run_error_studies.sh \
#       > /dev/null 2>&1 &
#
# Progress: $WORKING/logs/error_studies_<stamp>/STATUS (refreshed every REPORT_EVERY s);
# one log per job alongside it; DONE is written at the end with each job's exit code.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
: "${ECONBUS_CONFIG:?set ECONBUS_CONFIG (e.g. config.rev.yaml)}"
export ECONBUS_CONFIG
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
REPORT_EVERY=${REPORT_EVERY:-600}
OMEGA=${OMEGA:-0}
N=${N:-1000}

WORKING=$($PY -c 'from util import load_config; print(load_config().working)')
case "$WORKING" in
  */econ_bus|*/econ_bus_qss_submitted) echo "refusing to write to locked $WORKING" >&2; exit 2;;
esac
STAMP=$(date +%Y%m%d_%H%M)
LOG=$WORKING/logs/error_studies_$STAMP
mkdir -p "$LOG"
echo "$LOG" > "$WORKING/logs/error_studies_LATEST"

status() {
  {
    echo "error studies — $(date '+%F %T')   WORKING=$WORKING  omega=$OMEGA  n=$N"
    for f in "$LOG"/*.log; do
      [ -e "$f" ] || continue
      last=$(grep -E 'rep +[0-9]+/|Stage [0-9]|Done\.|Saved|Error|Traceback' "$f" | tail -1)
      printf '  %-14s %s\n' "$(basename "$f" .log)" "${last:-(starting)}"
    done
  } > "$LOG/STATUS.tmp" && mv "$LOG/STATUS.tmp" "$LOG/STATUS"
}

reporter() { while true; do status; sleep "$REPORT_EVERY"; done; }
reporter & REP=$!
trap 'kill $REP 2>/dev/null; status' EXIT

echo "start $(date '+%F %T')" > "$LOG/TIMES"
$PY spectral_ranking_bootstrap/bootstrap_oa_errors.py --stage 1,2,3 --n "$N" --omega "$OMEGA" \
    > "$LOG/stages123.log" 2>&1
rc=$?
echo "stages123 rc=$rc $(date '+%F %T')" >> "$LOG/TIMES"
if [ $rc -ne 0 ]; then echo "stages123 rc=$rc" > "$LOG/DONE"; exit $rc; fi

declare -A PIDS
for f in resample source institution reference; do
  $PY spectral_ranking_bootstrap/bootstrap_oa_errors.py --stage 4 --filters "$f" --n "$N" \
      > "$LOG/stage4_$f.log" 2>&1 &
  PIDS[stage4_$f]=$!
done
$PY spectral_ranking_bootstrap/bootstrap_oa_errors.py --stage year_v2 --n "$N" --omega "$OMEGA" \
    > "$LOG/year_v2.log" 2>&1 &
PIDS[year_v2]=$!

: > "$LOG/DONE.tmp"
fail=0
for job in "${!PIDS[@]}"; do
  wait "${PIDS[$job]}"; rc=$?
  echo "$job rc=$rc $(date '+%F %T')" | tee -a "$LOG/TIMES" >> "$LOG/DONE.tmp"
  [ $rc -ne 0 ] && fail=1
done
mv "$LOG/DONE.tmp" "$LOG/DONE"
exit $fail
