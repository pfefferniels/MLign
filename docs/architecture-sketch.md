# MLign architecture sketch (v0, pre-research-report draft)

Hypothesis to be validated/refined by research/01–03 reports. Written 2026-08-09.

## The pipeline

```
MEI ──convertMeiToMsmMpm──▶ score notes  ─┐
     (espressivo; ids, ticks, pitch,      ├─▶ [1] neural matcher ─▶ [2] structured ─▶ [3] refinement ─▶ alignment
      voice, ornament/grace marks)        │      (transformer)        decoding           passes           artifacts
MIDI ──parse──▶ perf notes (ms, pitch,   ─┘
      velocity, pedal)
```

## Stage 1 — neural note matcher (the ML core)

Dual-sequence transformer producing a match-probability matrix over
(score note, perf note) pairs plus per-note unmatched logits
(TheGlueNote-shaped, but):

- **Trained on unlimited synthetic espressivo data** with exact GT including
  ornament expansions, performer errors, correction-restarts, skips —
  supervision nobody else has. Mixed with real nASAP training splits.
- **Tempo-invariant input features**: IOIs / local relative time, not absolute
  time; onset-rank; pitch + pitch-class; voice/part/staff; velocity (perf side);
  score-side flags (grace, trill-marked, arpeggio-marked, chord-member).
- **Long pieces**: windowed inference with overlap-stitching (relative/rotary
  attention; window in the hundreds of notes, stride half).
- **Heads**: match matrix (contrastive/cross-entropy over pairs), score-note
  deletion head, perf-note insertion head, ornament-role head (perf note is
  ornament-member of matched principal: which score note is its anchor).

## Stage 2 — structured decoding

Probability matrix → valid alignment. Candidates to evaluate:

- Typed DP over (score idx, perf idx) with states {match, del, ins,
  ornament-expansion} — voice-aware monotonicity, chord asynchrony tolerance.
- Piecewise-monotonic path finding for repeats/jumps/restarts (Nakamura-style
  skip states; or segment-break detection + per-segment DP).
- Simple: mutual-argmax + constraint cleanup (what TheGlueNote does — beatable).

One-to-many (ornament) alignment is a first-class output: principal match +
member insertions grouped under the principal's score id — richer than
parangonar's one-to-one + ins/del.

## Stage 3 — refinement

- Pitch-consistency verification (a 'match' with pitch mismatch only allowed
  where the substitution head is confident).
- Ornament clustering of residual insertions around matched principals.
- Per-pair confidence in the output.

## Output artifacts

1. Internal JSON (docs/alignment-format.md, to be written): keyed by MEI
   xml:id, ornament sub-roles per D10/D15 contract.
2. Parangonar/partitura match-file export (comparability).
3. JSONL mirror of the mpmify row schema (their consumption).

## Training under M1/8GB constraints

- Compact model first (≈4–6 layers, d≈192–256, ~5–10M params), MPS, mixed
  precision, gradient accumulation; streaming JSONL dataset (no RAM corpus).
- Scale data, not params: the synthetic generator is the advantage.
- Multi-day training is acceptable; checkpoint every epoch; resumable.

## Claimed novelties (why this beats parangonar)

1. Exact ornament-provenance supervision from synthetic rendering (espressivo).
2. Robustness ops (errors/restarts/skips) trained-in, matching real recital data.
3. Typed structured decoding → structurally valid, one-to-many-capable output.
4. MEI-native: alignment lands directly on xml:ids (usable by digital editions;
   no MusicXML detour).

## Open questions for the research reports

- TheGlueNote exact architecture/tokenization/numbers (beat it, don't re-invent
  blindly); parangonar pipeline details + eval protocol.
- Match-file format details for ornament/repeat representation (partitura spec).
- How nASAP annotates trills/errors/repeats — our GT convention must match eval.
- Whether sustain pedal features help (Batik has pedal-rich Mozart).
