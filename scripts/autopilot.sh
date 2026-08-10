#!/bin/zsh
# MLign autopilot: idempotent supervisor. v3 = rebalanced mix, warm-started
# from v1c-e3 (best real-transfer checkpoint). Real-music share ~34% via
# selfsup oversampling (listed twice); ornament share halved (light shard only).
cd /Users/nielspfeffer/Projects/MLign
log() { echo "$(date +%H:%M:%S) $*" >> runs/autopilot.log; }
if ! pgrep -f "train.py --corpus" > /dev/null; then
  log "restarting v3 training (resumes runs/v3/last.pt)"
  mkdir -p runs
  nohup nice -n 10 .venv/bin/python -W ignore scripts/train.py \
    --corpus 'data/corpus/v1-*.jsonl,data/corpus/v2orn-light.jsonl,data/corpus/v3exag-*.jsonl,data/corpus/selfsup-v2.jsonl,data/corpus/selfsup-v2b.jsonl' \
    --epochs 24 --device cpu --threads 4 \
    --matchability --run runs/v3 > runs/v3.out 2>&1 &
fi
