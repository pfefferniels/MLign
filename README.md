# MLign

Symbolic score→performance alignment: an MEI or MusicXML score plus a MIDI
performance in, a note-level alignment out — every score note matched to a
performed note or marked as not played, every performed note matched or marked
as extra, each decision with a confidence. Learned end-to-end, trained largely
on synthetic expressive performances with exact ground truth.

On the held-out nASAP test split (84 performances the model never saw in any
form) MLign v2 reaches **0.9895 ± 0.0164** match-F against **0.9852** for
DualDTW (parangonar 3.1.0), the strongest published system we could run; paired
per performance, 69 wins / 3 ties / 12 losses. The margin is small and does not
carry over automatically — on Batik-plays-Mozart the two are at parity. Numbers,
protocols and negative results: [`docs/RESULTS.md`](docs/RESULTS.md),
[`docs/DESIGN.md`](docs/DESIGN.md).

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
- **Ornament attribution** (v2). A second head asks, per performed note, which
  written note it elaborates; otherwise a trill is one match and ten loose
  insertions. partitura's match format has had the slot since v1.0.0
  (`ornament(Anchor,Type)`), but no corpus we know of fills it — zero such lines
  across ASAP's 1063 match files, 4x22's 88 and Batik's 36. The espressivo
  renders are therefore the only ground truth available, and the head is trained
  and evaluated on synthetic data alone, never against prior work
  (`eval/run_attribution.py`). The alignment output is unchanged: ornament notes
  are still emitted as insertions, as the benchmark scores them.
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
# align (json | match | jsonl); models/mlign-v2.pt is the default checkpoint
PYTHONPATH=src python -m mlign.cli align score.mei performance.mid --format match -o out.match

# interactive test page at http://127.0.0.1:8765/
PYTHONPATH=src python -m mlign.serve
```

![test page](docs/testpage.png)

MusicXML input and the released model need only `torch`, `numpy` and
`partitura`; MEI input needs a built espressivo (path in `src/mlign/cli.py`).
Benchmarks (`eval/`) expect ASAP, Vienna 4x22 and Batik under
`data/benchmarks/`. For inference outside Python, `models/mlign-v2.onnx` (plus a
3.3 MB fp16 build) exports the encoder and all three heads with a JSON sidecar
describing every tensor; [`docs/DECODE-CONTRACT.md`](docs/DECODE-CONTRACT.md)
specifies the decoding around it for a browser port.

## Layout

`src/mlign/` model, inference, CLI, server, MEI export, repeat inference ·
`src/robustness/` performer-error layer · `scripts/` corpus generators, trainer,
ONNX export · `eval/` benchmark harnesses and metrics · `slurm/` cluster job ·
`models/mlign-v2.pt` the released model (run v8early, epoch 30, with
attribution), `models/mlign-v1.pt` the previous one (v5real, epoch 21).

MIT. Built with the espressivo, mpmify and bwUniCluster teams.
