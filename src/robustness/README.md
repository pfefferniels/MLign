# Robustness layer

Performer-error and structural perturbations over espressivo `PerformanceData`,
with a typed edit log that doubles as alignment ground truth. An
interface shared with the mpmify project (pure function, explicit seeded rng,
typed edit log).

- `robustness.mjs` — `applyRobustness(data, config, seed)` → `{data, edits}`
- `gt.mjs` — `editsToAlignment(data, edits)` → `{alignment, perfNotes, unattributed}`;
  `shiftToMatchedZero(perfNotes, alignment)` → perfNotes in the mpmify clock
  convention (first matched onset = 0.0, earlier insertions negative).
  `editsToAlignment` itself emits absolute facade ms (never negative).
- `rng.mjs` — seeded sfc32 + distributions. No `Math.random` anywhere.

Two config sections produce extra notes, and they model opposite things.
`insert` is the ERROR model: a brushed neighbour key, quiet and chromatic.
`add` is the INTENT model of early-recording piano style — octave doublings,
filled-in chord tones and unwritten ornaments, consonant with what sounds and
played at the anchor's weight. Both are attributable insertions: an added note
carries `origin = { type: 'addition', near, anchor, flavour, … }` back to the
score note it was generated from.

Run tests with the glob form (Node 23 resolves the bare directory oddly and
reports a phantom failure):

    node --test 'src/robustness/test/*.mjs'
