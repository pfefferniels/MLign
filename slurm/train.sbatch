#!/bin/bash
# MLign aligner training on bwUniCluster 3.0.
#
#   sbatch slurm/train.sbatch                                  # defaults below
#   sbatch --export=ALL,RUN=runs/c1,EPOCHS=24 slurm/train.sbatch
#   sbatch -p dev_gpu_h100 -t 00:30:00 --export=ALL,RUN=runs/smoke,EPOCHS=1 slurm/train.sbatch
#
# CORPUS_GLOBS MUST NOT be passed inside --export=... : sbatch splits that list
# on commas, so a comma-separated glob list is silently truncated at the first
# comma and the job trains on a fraction of the corpus while reporting success.
# Export it into the environment instead and let --export=ALL carry it:
#
#   export CORPUS_GLOBS='v1-*.jsonl,v2orn-light.jsonl,selfsup-v2.jsonl'
#   sbatch --export=ALL,RUN=runs/v4h100,EPOCHS=96 slurm/train.sbatch
#
# The "staged N corpus files" line in the log is the check: verify it matches
# what you expect before trusting a run.
#
# Corpus is staged to node-local NVMe ($TMPDIR) because the batcher re-reads the
# raw JSONL every epoch; Lustre hates that access pattern.
#
#SBATCH --job-name=mlign
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=niels.pfeffer@gmail.com

set -euo pipefail

WS=/pfs/work9/workspace/scratch/tu_amipf01-ml
REPO="$WS/MLign"

RUN="${RUN:-runs/c1}"
EPOCHS="${EPOCHS:-24}"
D_MODEL="${D_MODEL:-192}"
N_LAYERS="${N_LAYERS:-4}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
LR="${LR:-3e-4}"
# Same globs as the local run, rebased onto $TMPDIR.
CORPUS_GLOBS="${CORPUS_GLOBS:-v1-*.jsonl,v2orn-light.jsonl,v3exag-*.jsonl,selfsup-v2.jsonl,selfsup-v2b.jsonl}"
# Dedicated validation corpus (real-music proxy). Empty = train.py's internal
# val_frac split off the training mix, which cannot detect overfitting TO that
# mix — see the v4h100 post-mortem. Staged alongside the training files.
VAL_CORPUS="${VAL_CORPUS:-}"
# 0 = off. N = write runs/<name>/snap-eNNN.pt every N epochs, for offline
# checkpoint selection against a real benchmark.
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"

module load devel/python/3.13.3-gnu-14.2
PY="$HOME/venvs/ml-cu128/bin/python"

cd "$REPO"
mkdir -p slurm-logs "$(dirname "$RUN")"

# --- stage corpus to node-local NVMe -----------------------------------------
STAGE="$TMPDIR/corpus"
mkdir -p "$STAGE"
IFS=',' read -ra GLOBS <<< "$CORPUS_GLOBS"
for g in "${GLOBS[@]}"; do
    cp -v "$REPO"/data/corpus/$g "$STAGE"/ 2>/dev/null || true
done
echo "staged $(ls -1 "$STAGE" | wc -l) train files, $(du -sh "$STAGE" | cut -f1)"

CORPUS=""
for g in "${GLOBS[@]}"; do
    CORPUS="${CORPUS:+$CORPUS,}$STAGE/$g"
done

# Validation corpus is staged into its OWN directory: staging it beside the
# training files would let the train globs swallow it and silently train on
# the validation set.
VAL_ARGS=()
if [ -n "$VAL_CORPUS" ]; then
    VSTAGE="$TMPDIR/valcorpus"
    mkdir -p "$VSTAGE"
    IFS=',' read -ra VGLOBS <<< "$VAL_CORPUS"
    VCORPUS=""
    for g in "${VGLOBS[@]}"; do
        cp "$REPO"/data/corpus/$g "$VSTAGE"/ 2>/dev/null || true
        VCORPUS="${VCORPUS:+$VCORPUS,}$VSTAGE/$g"
    done
    echo "staged $(ls -1 "$VSTAGE" | wc -l) val files: $(ls -1 "$VSTAGE" | tr '\n' ' ')"
    VAL_ARGS=(--val-corpus "$VCORPUS")
fi

SNAP_ARGS=()
[ "$SNAPSHOT_EVERY" != "0" ] && SNAP_ARGS=(--snapshot-every "$SNAPSHOT_EVERY")

# Ornament attribution head. Its loss is supervised only on espressivo-rendered
# rows (meta.gen = mlign-*); real-GT and self-supervised rows are ignored by it,
# so enabling this is safe for any mix but only useful when the mix contains
# ornament shards.
#
# ATTR_CONDITIONED lets the head read the match head's (detached) insertion
# decision instead of re-deriving it: "bias" or "factored", empty = the v2 head.
ATTR_ARGS=()
[ "${ATTRIBUTION:-0}" != "0" ] && ATTR_ARGS=(--attribution --attr-weight "${ATTR_WEIGHT:-0.2}")
[ -n "${ATTR_CONDITIONED:-}" ] && ATTR_ARGS+=(--attr-conditioned "$ATTR_CONDITIONED")
# SEED shifts weight init and batch order together; 0 is every run so far.
SEED_ARGS=()
[ -n "${SEED:-}" ] && SEED_ARGS=(--seed "$SEED")

# --- run ---------------------------------------------------------------------
echo "=== $(date -Is) job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "run=$RUN epochs=$EPOCHS d_model=$D_MODEL n_layers=$N_LAYERS max_tokens=$MAX_TOKENS"

srun "$PY" -W ignore scripts/train.py \
    --corpus "$CORPUS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --max-tokens "$MAX_TOKENS" \
    --d-model "$D_MODEL" \
    --n-layers "$N_LAYERS" \
    --matchability \
    --device cuda \
    "${VAL_ARGS[@]}" \
    "${SNAP_ARGS[@]}" \
    "${ATTR_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    --run "$RUN"

echo "=== $(date -Is) done ==="
