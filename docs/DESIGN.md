# MLign — method

## Problem and representation

Input: a score as a table of notes (id, onset and duration in quarters, pitch,
voice) and a performance as a table of notes (id, onset and duration in
seconds, pitch, velocity). Output: a flat list of records in parangonar's
convention, extended:

```
{"label": "match",     "score_id", "perf_id", "confidence", "sub": {from, to} | null}
{"label": "insertion", "perf_id",  "confidence"}
{"label": "deletion",  "score_id", "confidence"}
```

Ids are the score's `xml:id`s (MEI via espressivo, MusicXML via partitura) and
partitura's performed-note ids, so the output is directly comparable with
match files and directly attachable to an edition. Exports: match file v1.0.0,
JSONL rows, and MEI's performance module (`src/mlign/mei_export.py`).

## Model (`src/mlign/model.py`)

One transformer over the concatenation `[S] s₁…sₙ [P] p₁…pₘ`. Each note is one
token: a pitch embedding plus a small MLP over continuous features (log
inter-onset interval, log duration, pitch class, voice or velocity), plus a
segment embedding. Positions are T5-style bucketed relative biases within each
segment, so there is no fixed sequence ceiling. Four pre-LN blocks, d = 192,
four heads, ≈ 1.5 M parameters.

Two projected note matrices give a bilinear similarity S·Pᵀ. Every score note
also gets a "not played" logit and every performed note an "extra" logit —
LightGlue-style per-note matchability heads rather than one shared dustbin
vector. Training loss is symmetric cross-entropy: each score note classifies
over {performed notes ∪ deleted}, each performed note over {score notes ∪
inserted}.

## Decoding (`src/mlign/infer.py`)

1. Confidence = dual softmax of the similarity matrix (with the null columns
   appended), i.e. the mutual-nearest-neighbour score.
2. Time map. Onset clusters on both sides; DTW over clusters with cost
   ½·(1 − Jaccard of pitch sets) + ½·(1 − mean model confidence); the DTW path
   is unioned with high-confidence mutual pitch-equal anchors into one monotone
   score-time → performance-time map. Neither alone suffices: sparse anchors
   leave interpolation holes, the DTW alone ignores what the model knows.
3. Assignment. Per pitch, rarest first, a small monotone DP matches score
   onsets projected through the map against performed onsets within a
   tolerance, cost = time deviation minus a confidence bonus. Then the map is
   rebuilt from the round-1 matches and assignment is repeated (this pins
   repeated-note runs under heavy rubato). A residual pass pairs remaining
   same-pitch score/performance notes within a tight window.
4. Leftovers become deletions/insertions; confidences are the dual-softmax
   value for matches and the null-share for the rest.

Long pieces are processed in score windows along the band of a coarse
cluster-DTW map, logits accumulated with overlap counts.

Repeats: if the score is folded, candidate unfoldings are enumerated from the
repeat structure and ranked by a banded local alignment of onset pitch sets
against the performance plus a note-count prior (`src/mlign/repeats.py`); the
winner is aligned as usual.

## Training data

| source | rows | what |
|---|---|---|
| espressivo renders | 64 k | random 1–2-part scores (mpmify samplers) performed with sampled MPM maps (tempo, rubato, dynamics, articulation, asynchrony, movement), generated ornaments on 30–50 % of pieces, an exaggeration curriculum on part of it, and the performer-error layer (`src/robustness/`: deletions, slips, wrong notes, timing shifts, restart-with-correction, skipped passages, timing jitter). Ground truth from espressivo's per-note provenance and the layer's edit log. |
| self-supervised real MIDI | 44 k | 512-note windows of nASAP training performances corrupted with a port of TheGlueNote's `reorder()` (deletions, insertions, skips, repeats, trills, tempo curves, jitter). |
| real ground truth | 6 k | windows of nASAP training alignments (match files). |
| real validation | 1.6 k | disjoint nASAP training pieces; the checkpoint-selection criterion. |

The released model (`v5real`, epoch 21) trained 32 epochs on one H100 in about
two hours. All test-split pieces (nASAP robust ∩ MAESTRO-v2 test) are excluded
from every source above.

## What did not work, briefly

- Selecting checkpoints on a mixed synthetic+real validation loss: it keeps
  improving after real-music transfer has peaked (measured in-run: optimum at
  epoch 22 on real validation, epoch 29 on the mix).
- Selecting on the Vienna 4x22 dev benchmark: its four short pieces do not
  contain long, deletion-heavy sonata movements; a checkpoint that set records
  there lost on the test split.
- A 4× larger model (d 320, 8 layers) on a 2× larger corpus: it overfits the
  synthetic distribution after ~10 epochs and transfers worse; the same corpus
  at the small size is a wash. Real-music share and valid selection matter
  more than capacity here.
- Biasing the null logits at decode time: no effect — insertion/deletion
  decisions are made structurally by the assignment step, not by a threshold.
