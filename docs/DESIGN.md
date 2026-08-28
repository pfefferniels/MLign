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
| espressivo renders | 64 k | random 1–2-part scores (mpmify samplers) performed with sampled MPM maps (tempo, rubato, dynamics, articulation, asynchrony, movement), generated ornaments (trills, mordents, turns) at a 0.3–0.5 probability per eligible note, which puts at least one ornament in ~99 % of pieces, an exaggeration curriculum on part of it, and the performer-error layer (`src/robustness/`: deletions, slips, wrong notes, timing shifts, restart-with-correction, skipped passages, timing jitter). Ground truth from espressivo's per-note provenance and the layer's edit log. |
| self-supervised real MIDI | 44 k | 512-note windows of nASAP training performances corrupted with a port of TheGlueNote's `reorder()` (deletions, insertions, skips, repeats, trills, tempo curves, jitter). |
| real ground truth | 6 k | windows of nASAP training alignments (match files). |
| real validation | 1.6 k | disjoint nASAP training pieces; the checkpoint-selection criterion. |

The released model (`v5real`, epoch 21) trained 32 epochs on one H100 in about
two hours. All test-split pieces (nASAP robust ∩ MAESTRO-v2 test) are excluded
from every source above.

## Ornament attribution

Every aligner in this space, ours included, answers "which played note is this
score note?". None answers **"which played notes realize this ornament?"** —
a trill's 11 notes become 1 match and 10 unattributed insertions.

We surveyed the three reference systems and the benchmark ground truth before
building anything:

| system | ornament → principal attribution |
|---|---|
| Nakamura HMM (`AlignmentTool`) | none; no ornament handling in the source |
| TheGlueNote | none; no ornament handling in the source |
| parangonar, as benchmarked | none — `process_ornaments=False` is the default |
| parangonar, `process_ornaments=True` | picks **one** performed note per ornament-marked score note (`match/matchers.py:1181-1220`); the rest of the figure stays unattributed insertions |

So the gap is real, but two things about it are easy to overstate, and both
bound what this feature can claim:

**The representation is not new.** partitura's match format v1.0.0 already has
`ornament({Anchor},{OrnamentType})-{NoteLine}` (`io/matchlines_v1.py:1090`),
read into `label="ornament"` records. The anchor slot exists; nobody fills it.
The claim is "first system to predict it", not "new representation".

**There is no ground truth anywhere.** Counted over the corpora in `data/`:
ASAP's 1063 match files contain **zero** `ornament(` lines (they are format
v5.0; the 164 files matching "trill" carry `trill-mark` as a *score* attribute,
and the trill's remaining notes are plain insertions). Vienna 4x22 (88 files)
and Batik (36 files) are format **1.0.0** — the format that supports anchors —
and also contain zero. One-to-many matches occur 11 times in 1063 ASAP files,
none of them ornament-marked. Our espressivo corpus is therefore the only
source of attribution labels that exists, which means the capability can be
trained and demonstrated but **not** benchmarked against prior work.

Consequences that shape the implementation:

- Attribution is a **separate bilinear head** (`ModelConfig.attribution`), not
  a reuse of the match similarity: the match head is trained to send ornament
  notes to the null column, so one score cannot rank both. Additive by
  construction — alignment metrics cannot regress from enabling it.
- Its loss is supervised **only on espressivo-rendered rows**
  (`meta.gen` = `mlign-*`). Real-GT and self-supervised rows contain real
  trills that are simply unlabelled; an empty `orn` there means "unknown", not
  "none", and supervising it would teach the head that real trills are not
  ornaments.
- Checkpoint **selection stays on the alignment loss alone**
  (`scripts/train.py:run_val`). Selection is the one choice already shown to
  decide this benchmark, and attribution quality must not be allowed to move
  it.
- The benchmark output convention is untouched: ornament notes are still
  emitted as insertions, since nASAP scores them that way. Attribution is an
  extra channel, landing on the MEI `xml:id` of the principal.

## Target repertoire: early recordings

The benchmarks above (nASAP, 4x22, Batik) are all post-war playing, while the
intended application is **early recordings** — pre-WWII rolls and 78s, where
deviation is both larger and different in kind: broad and dense arpeggiation,
free tempo, heavier ornamentation, and notes added beyond the text. Benchmark
rank and fitness for that repertoire can therefore diverge, and where they do,
the repertoire wins. The generator carries four adaptations:

- **Sampled `<temporalSpread>`** (`--breadth f`). Each ornament gets its own
  `ornamentDef` with a drawn frame length and placement, instead of the three
  fixed defs the corpus used to share (trill 100 %, mordent 30 %, turn 50 %,
  no offset). A share of figures is anticipated — begun before the notated
  onset — as early pianists routinely do. Measured over 30 pieces, breadth
  1 → 3 moves the median realized ornament span from 312 ms to 476 ms and the
  longest figure from 8 to 16 notes.
- **Exaggeration profiles** (`--exaggerate early`). The original curriculum
  ranges were calibrated on post-war playing; `early` widens them, rubato
  (0.5–3.5) and tempo (0.4–3.0) above all.
- **Consonant added notes.** Distinct from the neighbour-slip error model
  (±1–2 semitones, velocity ×0.4–0.75), which describes a mistake, not the
  intentional octave doublings, filled chord tones and unwritten ornaments of
  early practice. Added notes carry an anchor, so they feed the attribution
  head like any other ornament.
- **`imprecisionMap` humanisation**, so synthetic evaluation is not measured
  on unrealistically clean renders.

The last point is what makes the attribution head evaluable at all: with no
ornament ground truth in any real corpus, synthetic data is the only test set
there is, and it is only worth trusting to the extent that its performances
are not mechanically exact.

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
