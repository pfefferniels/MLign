# Robustness layer

Performer-error and structural perturbations over espressivo `PerformanceData`,
with a typed edit log that doubles as alignment ground truth. Shared-generator
contract with the mpmify ML program — see `research/00-coordination.md`.

- `robustness.mjs` — `applyRobustness(data, config, seed)` → `{data, edits}`
- `gt.mjs` — `editsToAlignment(data, edits)` → `{alignment, perfNotes, unattributed}`;
  `shiftToMatchedZero(perfNotes, alignment)` → perfNotes in the mpmify clock
  convention (first matched onset = 0.0, earlier insertions negative).
  `editsToAlignment` itself emits absolute facade ms (never negative).
- `rng.mjs` — seeded sfc32 + distributions. No `Math.random` anywhere.

Run tests with the glob form (Node 23 resolves the bare directory oddly and
reports a phantom failure):

    node --test 'src/robustness/test/*.mjs'
