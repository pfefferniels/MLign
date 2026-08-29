# MLign

Symbolic score→performance alignment: an MEI or MusicXML score plus a MIDI
performance in, a note-level alignment out — every score note matched to a
performed note or marked as not played, every performed note matched or marked
as extra, each decision with a confidence. Learned end-to-end, trained largely
on synthetic expressive performances with exact ground truth.

On the held-out nASAP test split (84 performances the model never saw in any
form) MLign v3 reaches **0.9896 ± 0.0165** match-F against **0.9852** for
DualDTW (parangonar 3.1.0), the strongest published system we could run; paired
per performance, 68 wins / 3 ties / 13 losses. The margin is small and does not
carry over automatically — on Batik-plays-Mozart the two are at parity. On
alignment v3 ties its own predecessor v2 (35W / 18T / 31L, p = 0.36); what it
adds is an ornament-attribution head that works on real recordings rather than
only on our own renders. Numbers, protocols and negative results:
[`docs/RESULTS.md`](docs/RESULTS.md), [`docs/DESIGN.md`](docs/DESIGN.md).

## What this builds on

The problem, the datasets, the metric and the baselines are other people's work.

- **Peter, Cancino-Chacón, Foscarin, McLeod, Henkel, Karystinaios & Widmer**,
  *Automatic Note-Level Score-to-Performance Alignments in the ASAP Dataset*
  (TISMIR 2023): the alignments and the evaluation protocol, adopted unchanged,
  and the reference systems in
  [parangonar](https://github.com/sildater/parangonar). DualDTW (Peter, ISMIR
  2023) is what we measure against, and its two-phase design — monotone time map
  first, per-pitch assignment second — is one we kept.
- **Peter & Widmer**, [TheGlueNote](https://github.com/sildater/thegluenote)
  (ISMIR 2024): transformer note matching learned from self-supervised
  corruptions of raw MIDI, and the dustbin formulation for insertions and
  deletions. Our self-supervised data source is a port of their `reorder()`.
- **Nakamura, Yoshii & Katayose** (ISMIR 2017): the standing classical baseline,
  and the heavy-tailed timing model and skip transitions decoding still needs.
- **Peter, Hu & Widmer**, *How to Infer Repeat Structures in MIDI Performances*
  (2025): repeat structure as preprocessing — rank candidate unfoldings, then
  align — reimplemented here.
- Datasets: **ASAP** (Foscarin, McLeod, Rigaux, Jacquemard, Sakai, ISMIR 2020),
  the **Vienna 4x22 Piano Corpus** (Goebl, 1999), **Batik-plays-Mozart** (Hu &
  Widmer, ISMIR 2023). The per-note matchability head is **LightGlue**'s
  (Lindenberger, Sarlin, Pollefeys, ICCV 2023).

## What is different here

- **Synthetic supervision with note provenance.** Performances are rendered
  from scores by [espressivo](https://github.com/pfefferniels/espressivo) with
  MPM instructions — rubato, asynchrony, articulation, dynamics, arpeggios,
  generated ornaments — and every rendered note keeps the score note's
  `xml:id`; a performer-error layer (wrong notes, slips, restarts, skipped
  passages) logs its edits as further ground truth. Labels no corruption of
  MIDI can produce, at the cost of a distribution we choose rather than observe.
- **Ornament attribution** (v3). A second head asks, per performed note, which
  written note it elaborates; otherwise a trill is one match and ten loose
  insertions. Training supervision is synthetic — espressivo renders, where
  provenance is exhaustive — but the evaluation is not. partitura's
  `ornament(Anchor,Type)` slot is indeed empty in every corpus we have (zero
  such lines across ASAP's 1063 match files, 4x22's 88 and Batik's 36), which
  we long read as "no real ground truth exists". That was wrong: ASAP and Batik
  put the ornament sign in the **score note's own attribute list**, with the
  played notes following as `insertion-note` lines, and 1527 real ornament
  groups can be recovered from them (`scripts/corpus/real_orn_gt.py`). Measured
  there, v3 gets 19.2 % / 51.3 % of figures exactly right against v2's 2.0 % /
  3.3 % — a 9.5× and 15.6× gain that no synthetic holdout could see, since
  those rank the models the other way round (`docs/RESULTS.md` §6). The
  alignment output is unchanged: ornament notes are still emitted as
  insertions, as the benchmark scores them.
- **Real music decides which checkpoint ships.** Training mixes real material
  (self-supervised and real-ground-truth windows from the nASAP training pieces)
  with the synthetic corpus, and checkpoints are selected on a real-music
  validation loss. The mixed synthetic criterion picks one seven epochs too
  late, a popular dev benchmark one that loses on the test set
  (`docs/RESULTS.md` §5).
- **MEI-native.** Alignments land on the edition's `xml:id`s; export to MEI's
  performance module besides match files and JSONL.
- **Folded scores.** Given a score folded to one pass, MLign infers the played
  repeat structure and reaches 0.996 match-F on the Vienna 4x22 repeat pieces,
  where published symbolic systems score below 0.37 (RUMAA, WASPAA 2025,
  Table 2; RUMAA itself reaches 98.4 pooled F, but from audio).

The model is small: 1.6 M parameters, four transformer layers over one stream of
score and performance tokens with relative positions, a bilinear match head,
per-note matchability, and the attribution head. Decoding fuses a cluster-level
DTW over pitch-set and confidence cost with the model's mutual anchors into a
monotone time map, then assigns notes per pitch.

## Use

```bash
# align (json | match | jsonl); models/mlign-v3.pt is the default checkpoint
PYTHONPATH=src python -m mlign.cli align score.mei performance.mid --format match -o out.match

# interactive test page at http://127.0.0.1:8765/
PYTHONPATH=src python -m mlign.serve
```

MusicXML input and the released model need only `torch`, `numpy` and
`partitura`; MEI input needs a built espressivo (path in `src/mlign/cli.py`).
Benchmarks (`eval/`) expect ASAP, Vienna 4x22 and Batik under
`data/benchmarks/`. For inference outside Python, `models/mlign-v3.onnx` (plus a
3.3 MB fp16 build) exports the encoder and all three heads with a JSON sidecar
describing every tensor; [`docs/DECODE-CONTRACT.md`](docs/DECODE-CONTRACT.md)
specifies the decoding around it for a browser port.

## Layout

`src/mlign/` model, inference, CLI, server, MEI export, repeat inference ·
`src/robustness/` performer-error layer · `scripts/` corpus generators, trainer,
ONNX export · `eval/` benchmark harnesses and metrics · `slurm/` cluster job ·
`models/mlign-v3.pt` the released model (run v9fact, epoch 20, conditioned
attribution), `models/mlign-v2.pt` (v8early, epoch 30) and `models/mlign-v1.pt`
(v5real, epoch 21) the previous ones.

MIT. Built with the espressivo, mpmify and bwUniCluster teams.
