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

# 2. When synthetic shards + leakage-free selfsup are complete, ensure v1c training.
syn=$(cat data/corpus/v1-*.jsonl 2>/dev/null | wc -l)
ss=$( [ -f data/corpus/selfsup-v2.jsonl ] && wc -l < data/corpus/selfsup-v2.jsonl || echo 0 )
gen_running=""
pgrep -f "generate.mjs" > /dev/null && gen_running=1
pgrep -f "corpus/selfsup.py" > /dev/null && gen_running=1
if [ "$syn" -ge 15900 ] && [ "$ss" -ge 6000 ] && [ -z "$gen_running" ]; then
  if ! pgrep -f "train.py --corpus" > /dev/null; then
    log "starting v1c training (syn $syn + selfsup $ss rows)"
    mkdir -p runs
    nohup nice -n 10 .venv/bin/python -W ignore scripts/train.py \
      --corpus 'data/corpus/v1-*.jsonl,data/corpus/selfsup-v2.jsonl' \
      --epochs 24 --device cpu --threads 4 \
      --matchability --run runs/v1c > runs/v1c.out 2>&1 &
  fi
fi
