# MLign — symbolic score→performance alignment

MEI (or MusicXML) score + performed MIDI → note-level alignment (matches,
insertions, deletions, each with a confidence), learned end-to-end and trained
on synthetic expressive performances with exact ground truth.

**Status (2026-08-16):** on the untouched nASAP holdout (84 performances, MAESTRO-v2
test pieces) MLign v1 scores **0.9878** match-F vs **0.9852** for DualDTW
(parangonar's hand-tuned SOTA) — 65 wins / 17 losses per performance,
p < 1e-7 — and is best on insertions and deletions too. Full tables, protocols
and caveats: [`docs/RESULTS.md`](docs/RESULTS.md) (generated from
`eval/results/*.json`).

## Quickstart

```bash
# python 3.13 venv with torch/numpy/partitura (see requirements-train.txt for the training subset)
PYTHONPATH=src .venv/bin/python -m mlign.cli align score.musicxml performance.mid --format json
PYTHONPATH=src .venv/bin/python -m mlign.cli align score.mei performance.mid --format match -o out.match
```

Formats: `json` (internal records with confidence), `match` (partitura/parangonar
match file v1.0.0), `jsonl` (mpmify row-schema mirror). MEI input is parsed with
espressivo (xml:ids preserved); MusicXML with partitura. The released model
`models/mlign-v1.pt` (6 MB, 1.5M params) is the default; `--engine baseline`
runs the classical DTW fallback.

## What's here

| path | what |
|---|---|
| `src/mlign/` | model (`model.py`), inference + decode (`infer.py`), tables, CLI, MEI export, repeat inference |
| `src/robustness/` | performer-error / restart / skip layer with typed edit log (= alignment GT); shared with mpmify |
| `scripts/corpus/` | synthetic corpus generator (espressivo renders + ornaments + exaggeration + robustness), self-supervised real-MIDI corruption, real-GT windows |
| `scripts/train.py` | trainer (resumable, atomic checkpoints, dedicated real-music validation, snapshots) |
| `eval/` | nASAP / Vienna 4x22 / Batik harnesses, mismatch + folded-score benchmarks, dev-long tier, metrics (TISMIR-compatible, unit-tested) |
| `docs/DESIGN.md` | architecture + decisions; `docs/RESULTS.md` all numbers; `LOG.md` the full journal |
| `research/` | literature, local-codebase and espressivo studies; peer-agent coordination contracts |
| `slurm/` | bwUniCluster H100 job script |

## Method in one paragraph

A single-stream transformer over `[score notes][performance notes]` (T5-style
relative positions, per-note feature embeddings, dustbin/matchability heads)
produces a match matrix; decoding builds a monotone time map from a cluster-DTW
over blended pitch-set + model-confidence cost fused with mutual anchors, then
assigns notes per pitch (rarest first, map rebuilt after round 1), with a
same-pitch residual rescue. Repeats are handled as preprocessing (candidate
unfoldings ranked by pitch-set local alignment + note-count prior). Training data
is ~47% real music (self-supervised corruptions of nASAP train performances +
real ground-truth windows) and ~53% synthetic performances rendered by espressivo
with exact provenance (ornaments, exaggeration curriculum, error/restart/skip
layer). Checkpoints are selected on a real-music validation loss — the single
most important choice; see RESULTS §5.

## Reproducing

Datasets (gitignored): `data/benchmarks/{asap-dataset,vienna4x22,batik_plays_mozart}`
plus TheGlueNote's 4x22 `.npz` files. Evaluate the released model:

```bash
.venv/bin/python -W ignore eval/run_eval.py --aligner model:models/mlign-v1.pt --robust-only --split test
.venv/bin/python -W ignore eval/run_4x22.py --ckpt models/mlign-v1.pt
.venv/bin/python -W ignore eval/run_4x22_repeats.py --ckpt models/mlign-v1.pt
```
