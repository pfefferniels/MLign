#!/bin/zsh
# Overnight comparison suite. Waits for mpmify's v31 trainer (caffeinate) to
# exit, then runs the heavy parangonar/model comparisons sequentially.
# Self-limiting: aborts any NEW step after the cutoff hour (07:00) so mpmify's
# v4 training starts on a quiet machine. Detached-safe; log = eval/results/gap-run.log.
cd /Users/nielspfeffer/Projects/MLign
PY=.venv/bin/python
LOG=eval/results/gap-run.log
log() { echo "$(date +%H:%M:%S) $*" >> $LOG; }

log "gap-runner armed; waiting for caffeinate to exit"
while pgrep -f caffeinate > /dev/null; do sleep 120; done
log "GAP OPEN"

run_step() {
  local name=$1; shift
  local hour=$(date +%H)
  if [ "$hour" -ge 7 ] && [ "$hour" -lt 12 ]; then
    log "SKIP $name (past 07:00 cutoff)"
    return
  fi
  log "START $name"
  "$@" >> $LOG 2>&1
  log "DONE $name (exit $?)"
}

mkdir -p eval/results
run_step dualdtw-robust-test $PY -W ignore eval/run_eval.py --aligner dualdtw --robust-only --split test --out eval/results/dualdtw-robust-test.json
run_step model-robust-test   $PY -W ignore eval/run_eval.py --aligner model:runs/v2/best.pt --robust-only --split test --out eval/results/v2-robust-test.json
run_step dualdtw-4x22mm      $PY -W ignore eval/run_4x22_mismatch.py --aligner dualdtw --out eval/results/dualdtw-4x22mm.json
run_step gluenote-robust-test $PY -W ignore eval/run_eval.py --aligner gluenote --robust-only --split test --out eval/results/gluenote-robust-test.json
run_step model-batik         $PY -W ignore eval/run_batik.py --ckpt runs/v2/best.pt --out eval/results/v2-batik.json
run_step dualdtw-batik       $PY -W ignore eval/run_batik.py --aligner dualdtw --out eval/results/dualdtw-batik.json
log "gap suite complete"
