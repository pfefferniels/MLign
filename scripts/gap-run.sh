#!/bin/zsh
# Overnight comparison suite. Waits for mpmify's trainer (caffeinate) to exit,
# then runs heavy comparisons sequentially for AT MOST 4 hours after the gap
# opens (the agreed window length — relative, battery-sleep-proof), then stops
# starting new steps. Log: eval/results/gap-run.log.
cd /Users/nielspfeffer/Projects/MLign
PY=.venv/bin/python
LOG=eval/results/gap-run.log
log() { echo "$(date +%H:%M:%S) $*" >> $LOG; }

log "gap-runner armed; waiting for caffeinate to exit"
while pgrep -f "^caffeinate -is python3" > /dev/null; do sleep 120; done
GAP_OPEN=$(date +%s)
log "GAP OPEN"

run_step() {
  local name=$1; shift
  local now=$(date +%s)
  if [ $(( now - GAP_OPEN )) -gt 14400 ]; then
    log "SKIP $name (past gap-open+4h)"
    return
  fi
  log "START $name"
  "$@" >> $LOG 2>&1
  log "DONE $name (exit $?)"
}

mkdir -p eval/results
run_step dualdtw-robust-test $PY -W ignore eval/run_eval.py --aligner dualdtw --robust-only --split test --out eval/results/dualdtw-robust-test.json
run_step model-robust-test   $PY -W ignore eval/run_eval.py --aligner model:runs/v1c/best.pt --robust-only --split test --out eval/results/mlign-robust-test.json
run_step dualdtw-4x22mm      $PY -W ignore eval/run_4x22_mismatch.py --aligner dualdtw --out eval/results/dualdtw-4x22mm.json
run_step gluenote-robust-test $PY -W ignore eval/run_eval.py --aligner gluenote --robust-only --split test --out eval/results/gluenote-robust-test.json
run_step model-batik         $PY -W ignore eval/run_batik.py --ckpt runs/v1c/best.pt --out eval/results/mlign-batik.json
run_step dualdtw-batik       $PY -W ignore eval/run_batik.py --aligner dualdtw --out eval/results/dualdtw-batik.json
log "gap suite complete"
