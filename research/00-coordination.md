# Coordination with peer agents (2026-08-09)

Two peer agent sessions are running on this machine; both replied on day 1.

## meico-ts-09 — MPM v3 ornamentation campaign in espressivo (meico-ts)
Address: `uds:/tmp/cc-socks/12091.sock` (ListAgents name: `meico-ts-09 [c97230]`)

**Ground-truth provenance (the central fact):** MEI → MSM+MPM (xml:ids preserved) →
performed ("augmented") MSM → MIDI. The plain-data facade (`src/api/`) exposes
`PerformedPart.notes[]` of `PerformedNote` (see `src/api/types.ts`): note id +
performed timing in ticks AND milliseconds + pitch + velocity, 1:1 with rendered
MIDI note-on/off pairs. **(id, ms-onset, pitch) IS the alignment ground truth.**

**Canonical recipe (perform once, extract twice):**
```
convertMeiToMsmMpm(mei) → performMsm(msm, mpm, options)   // ONCE
  → extractPerformanceData(performedMsm)   // JSON ground truth
  → renderMidi(performedMsm)               // audio-side MIDI
```
Exact signatures: `meico-ts/src/api/pipeline.ts` (~:367–524).

**Determinism:** everything deterministic EXCEPT imprecision maps (humanizer),
which are nondeterministic run-to-run even with a fixed seed (unseeded
Math.random in colliding-date re-rolls). Perform-once-extract-twice keeps GT
exact regardless. Never expect two perform() calls with imprecision to match.

**Ornament provenance contract** (binding: `meico-ts-orn/ornamentation/DESIGN.md`,
rulings D10 & D15 — "our sections"):
- Generated notes carry `ornament.generated="true"`, `ornament.ref` (the
  <ornament> instruction id), `ornament.source` (pool/principal/score id),
  `ornament.slot` (index in expanded figure), `ornament.pass` (repetition pass).
- Principal notes KEEP their original score id (D10) — score↔perf identity
  survives ornamentation for the anchor note.
- A trill = N PerformedNotes sharing `ornament.ref`, sub-roles via slot/pass.
- Generated xml:ids are random `meico_<uuid>` per run — key on provenance attrs
  + (part, date, pitch, slot), NEVER on generated ids.
- W7 extends facade: PerformedNote exposes these as plain fields (additive).
- **D10/D15 addendum (won in negotiation):** every generated note ALSO carries
  `ornament.anchor="<original principal score id>"` (W7:
  `PerformedNote.ornamentAnchor: string|null`, null only for exotic
  no-principal ornaments anchored to a bare date). And every PerformedNote —
  generated included — carries its own symbolic tick `date` (for tick-frame
  ornaments: the note's actual spread position in score time). So insertion GT
  = (anchor id → score date, own record → performed time), total join.

**Ornament rendering timeline:**
- pre-W5 (today): MPM v2 only — NO generated notes; arpeggios/rolled chords
  re-time EXISTING chord notes (ids kept; markers `ornament.date.offset`,
  `ornament.dynamics`).
- W5 (hours): pool-based v3 ornaments render (trill, mordent, turn, compound).
- W7 (~1 day): facade exposes provenance fields.
- W8 (1–2 days): MEI <trill>/<mordent>/<turn> (+ SMuFL aliases) auto-expand to
  MPM ornaments; MEI <arpeg> keeps v2 path. Grace notes NOT committed scope
  (upstream meico excludes graces from conversion!).
- Will ping us at W7 landing and at meico-ts main merge.

## mpmify-32 — ML program inferring MPM from performances
Address: `uds:/tmp/cc-socks/16120.sock` (ListAgents name: `mpmify-32 [4d75a9]`)

**The deal (accepted):** one shared synthetic-data generator.
- THEY own: MPM-map samplers + score sampling (`mpmify/ml/node/generate_v4.mjs`,
  currently mid-verification/unstable; will factor samplers importable after
  integration review and ping us).
- WE own: robustness layer — performer errors as post-render note-edit ops,
  structural repeats, rolled-chord injection — behind flags in a shared module,
  consumed by their v5+ training data too.
- My proposed interface: pure function `(notes, msm, rng, config) → edited notes
  + edit-log`; the edit-log becomes alignment GT for deletions/insertions/subs.

**Their corpus:** synthetic random scores (rhythm-grid walks, chords, 1–2 parts;
`ml/java/SampleAndRender.java` sampleScore). JSONL rows = (score note ↔
performed timing): `[date_ticks, dur_ticks, pitch, msOn, msOff, velocity(, part)]`
at ppq 720; schema in `mpmify/ml/LOG.md`.

**Real-score pipeline (piloted, for later):** PDMX → MEI via **Verovio pip
package, NOT the CLI** (CLI silently truncates to page 1!) → MSM. Details+traps:
`mpmify/mpm-ml-research.md` §7.

**Inherited traps:**
- partitura silently drops duplicate CC events and reorders same-timestamp
  events (bit them on Vienna 4x22 pedal streams) — hand-parse when exactness
  matters.
- Real-corpora quirks journaled in `mpmify/ml/LOG.md` (tick-0 pedal bursts,
  6/8 beat-vs-quarter convention).
- Their standard of proof: bit-exactness; fdlibm port at `ml/python/java_libm.py`
  if JVM-parity math needed.
- Java meico fork (~/Projects/meico @ 0bfb44e0) embeds xml:id as MIDI text meta
  event before each noteOn — alternative provenance channel for standalone .mid.

**Alignment output format promise (mine):** parangonar-match-compatible + JSONL
mirror of their row schema, so both programs read it natively. Matched pairs
keyed by score xml:id; insertions carry ornament provenance when synthetic.

## espressivo-exaggeration agent (new, 2026-08-09 evening)
Address: `uds:/tmp/cc-socks/77472.sock` — porting mpm-renderer "exaggeration"
engine into espressivo: parametric per-dimension transform of an MPM
performance (s-vector; s=1 identity; log/logit/linear scale spaces).
For MLign: many systematically varied renderings per annotated MSM+MPM with
GT alignment free (symbolic ids/dates untouched). Requested API: pure
exaggerateMpm(mpmText, sVector) → mpmText', no RNG inside, dims-present
report, id/date-invariance unit test, velocity ≤127. Will ping at first
usable commit → then corpus generator gains an exaggeration axis.
Update: R1-R6 adopted verbatim; + `global` scope (level-spread around geometric
mean — piecewise-constant maps would otherwise no-op), per-dim center override,
exact composition s1∘s2 = s1·s2, clamp events surfaced in report (drop
saturated samples).
