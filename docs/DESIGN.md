# MLign design (v1) — consolidated after research reports

Supersedes docs/architecture-sketch.md. Sources: research/00 (peer contracts),
research/02 (local codebases), research/03 (espressivo). Status: binding until
revised; revisions must be journaled in LOG.md.

## 1. What we exploit that nobody else has

1. **Exact synthetic supervision**: espressivo renders MEI+MPM with per-note
   provenance (two agreeing channels: MIDI FF01 text events + facade data).
   Realistic rubato/asynchrony/ornaments with perfect GT — vs TheGlueNote's
   purely mechanical `reorder()` corruptions.
2. **Score features everyone discards**: every existing system is functionally
   (pitch, onset)-only. MEI/partitura carry voice, staff, grace/ornament type,
   metric position, duration — we feed them to the model.
3. **Error/restart/skip robustness** trained in via our robustness layer.
4. **Richer output**: parangonar dicts + `substitution` label (only Nakamura's
   format can express wrong notes today) + confidence + ornament grouping
   (anchor/slot/pass, D10/D15 contract) + MEI `<performance>/<when>` export.

## 2. Alignment representation (internal)

Flat list of dicts, parangonar-compatible, extended:

```python
{"label": "match",     "score_id", "performance_id", "confidence", "sub": {from,to}|None}
{"label": "insertion", "performance_id", "confidence",
                        "ornament": {anchor_score_id, ref, slot, pass}|None}
{"label": "deletion",  "score_id", "confidence"}
```

Exports: (a) partitura match file (v1.0.0) — normalize ornament→insertion for
scoring parity; (b) mpmify JSONL row schema; (c) MEI performance module
(`<recording>/<when>` à la aligned-mei, plus deletions/insertions/confidence).

## 3. Model (v1)

Single-stream encoder over `[S] score-notes [P] perf-notes` (one token per
note — sum of feature embeddings, not TheGlueNote's 4 tokens/note):

- **Features** score: pitch emb + Δonset(quarter, log) + duration(quarter, log)
  + voice + grace flag + ornament-type + metric position (beat fraction);
  perf: pitch emb + IOI(sec, log) + duration(sec, log) + velocity.
- **Positions**: T5-bucket relative bias (no absolute table → no 512 ceiling).
  Segment embeddings distinguish sides.
- **Heads**: bilinear similarity matrix + dustbin null row/col (learned vectors)
  = insertion/deletion as first-class predictions (TheGlueNote's best idea);
  substitution head (match despite pitch mismatch); ornament-role head
  (perf note → anchor score note) once W7 corpus data exists.
- **Loss**: symmetric CE (each score note over perf∪null, each perf note over
  score∪null) — already implemented; add substitution + ornament losses later.
- **Inference scoring**: dual-softmax product (mutual confidence) before
  decoding.
- Size: start d=192/L=4 (~1.5M), scale to d=256/L=6 (~6M) if MPS timing allows.

## 4. Decoding (the two-phase structure all serious systems converge on)

1. **Confident monotone map**: from the model's confidence matrix take
   high-confidence mutual pairs, dual forward/backward agreement filter,
   median-agreement gate → monotone score→perf time map. Piecewise for
   repeats/restarts: detect jump discontinuities (band-break heuristic or
   RepeatIdentifier on onset pitch-sets) → piecewise-monotone segments.
2. **Note assignment**: per pitch (rarest first, map rebuilt after each pitch —
   parangonar's updating-map trick), assign score↔perf within map tolerance;
   leftovers → null heads decide insertion vs substitution vs deletion.

Windowing at inference: coarse anchor pass first (cheap: onset-cluster DTW or
strided low-res model pass) → run full model only on windows along the
(piecewise) band. Never TheGlueNote's full 2-D cross product.

## 5. Training data (three-source hybrid)

- **A. espressivo renders** (scripts/corpus/generate.mjs): mpmify samplers +
  robustness layer (errors/restarts/skips/jitter). v0 = 16k pieces in flight;
  v1 after espressivo E1/E2 fixes land (articulation identity + dynamics
  curvature — GT valid either way). Longer/realer scores later (PDMX→MEI via
  Verovio pip). Velocity trap avoided (sampler stays in [0,127]).
- **B. reorder()-style self-supervision** on real performance MIDI (TheGlueNote
  datasets/reorder — ~180 lines to port): covers extreme tempo scaling,
  segment skip/repeat *on every sample*, trills. Use ONLY train-split nASAP
  performances (no test leakage).
- **C. nASAP train-split GT fine-tune** (1063 alignments exist; TheGlueNote
  never used them!) — real annotation conventions (ornament labels, ties).

## 6. Evaluation

- Datasets: nASAP (piece-level split — replicate parangonar's papers' split if
  published; else disjoint-piece split, journal it), Vienna 4x22 (.npz local,
  43,450 matches), Batik-plays-Mozart. All three on disk.
- Metrics: parangonar `fscore_alignments` semantics (my eval/metrics.py is
  compatible) on match/insertion/deletion + asynchrony percentiles
  (<25/50/100ms) via score-following eval. Normalize ornament labels first.
- Baselines to beat, run locally: DualDTWNoteMatcher (SOTA), AutomaticNoteMatcher,
  TheGlueNoteMatcher (all parangonar 3.1.0, verified working), my DTW floor.
  Nakamura upstream = clone + clang++ build (budget an MEI→fmt3x writer; the
  local WASM fork is crippled — numbers from it would be unfair to Nakamura).
- My floor so far: baseline DTW ≈0.99 match-F on easy Bach; full-1063 run in
  progress.

## 7. Product

CLI: `mlign align score.mei performance.mid -o out.{match,json,mei}` —
MEI-native (espressivo conversion for score parsing, ids = xml:ids), MIDI
parsed with symusic/mido (NOT partitura in the product path — its CC-dropping
/ reordering traps; partitura only for benchmark-path parity). Confidence in
every record. Optional: WASM build later (embind template exists in the C++
port).

## 8. Non-goals (for now)

Audio alignment; online/real-time following; hand separation. Revisit post-v1.
