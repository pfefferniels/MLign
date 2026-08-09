#!/bin/zsh
# MLign autopilot: survives harness/session kills. Idempotent — run any time;
# it only starts what's missing. Journals to runs/autopilot.log.
cd /Users/nielspfeffer/Projects/MLign
log() { echo "$(date +%H:%M:%S) $*" >> runs/autopilot.log; }

# Ensure v2 training (warm-started from v1c at epoch 3) keeps running.
# Corpus: 16k v1 synthetic + 8k v2orn ornament-rich + 7.4k selfsup-v2.
syn=$(cat data/corpus/v1-*.jsonl 2>/dev/null | wc -l)
orn=$(cat data/corpus/v2orn-*.jsonl 2>/dev/null | wc -l)
ss=$( [ -f data/corpus/selfsup-v2.jsonl ] && wc -l < data/corpus/selfsup-v2.jsonl || echo 0 )
if [ "$syn" -ge 15900 ] && [ "$orn" -ge 7900 ] && [ "$ss" -ge 6000 ]; then
  if ! pgrep -f "train.py --corpus" > /dev/null; then
    log "restarting v2 training (resumes runs/v2/last.pt)"
    mkdir -p runs
    nohup nice -n 10 .venv/bin/python -W ignore scripts/train.py \
      --corpus 'data/corpus/v1-*.jsonl,data/corpus/v2orn-*.jsonl,data/corpus/selfsup-v2.jsonl' \
      --epochs 24 --device cpu --threads 4 \
      --matchability --run runs/v2 > runs/v2.out 2>&1 &
  fi
fi
