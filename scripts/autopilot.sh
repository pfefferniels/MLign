#!/bin/zsh
# MLign autopilot: survives harness/session kills. Idempotent — run any time;
# it only starts what's missing. Journals to runs/autopilot.log.
cd /Users/nielspfeffer/Projects/MLign
log() { echo "$(date +%H:%M:%S) $*" >> runs/autopilot.log; }

# 1. Ensure v1 corpus generators (4 shards) run to completion.
for p in none light medium heavy; do
  case $p in none) s=1000;; light) s=2000;; medium) s=3000;; heavy) s=4000;; esac
  f=data/corpus/v1-$p.jsonl
  n=$( [ -f $f ] && wc -l < $f || echo 0 )
  if [ "$n" -lt 4000 ] && ! pgrep -f "generate.mjs data/corpus/v1-$p" > /dev/null; then
    log "starting generator $p (have $n/4000)"
    nohup nice -n 12 node scripts/corpus/generate.mjs $f 4000 $s --robustness $p \
      > data/corpus/v1-$p.log 2>&1 &
  fi
done

# 2. When all shards are complete and no generator runs, ensure training runs.
total=$(cat data/corpus/v1-*.jsonl 2>/dev/null | wc -l)
if [ "$total" -ge 15900 ] && ! pgrep -f "generate.mjs" > /dev/null; then
  if ! pgrep -f "train.py --corpus" > /dev/null; then
    log "starting v1 training (corpus rows: $total)"
    mkdir -p runs
    nohup nice -n 10 .venv/bin/python -W ignore scripts/train.py \
      --corpus 'data/corpus/v1-*.jsonl,data/corpus/selfsup-v1.jsonl' --epochs 24 --device cpu --threads 4 \
      --matchability --run runs/v1b > runs/v1b.out 2>&1 &
  fi
fi
