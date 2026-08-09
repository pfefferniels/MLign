# Training corpus format (v0)

One gzipped JSONL file per corpus shard; one line per rendered performance.
Index-based (compact); string ids only in sidecar arrays for traceability.

```jsonc
{
  "meta": {
    "gen": "mlign-v0",            // generator version
    "seed": "shard3:17",           // full determinism handle
    "score": "fixture:tempo.mei",  // score source ref
    "mpm": { /* sampler config snapshot */ },
    "robustness": { /* config snapshot */ }
  },
  // Score notes, sorted (onset, pitch). Implicit index si = row number.
  // [onset_ticks, dur_ticks, pitch, voice]  @ ppq 720 (mpmify convention)
  "score": [[0, 720, 60, 0], ...],
  "scoreIds": ["n0", ...],         // MEI/MSM xml:ids, parallel to score
  // Performed notes, sorted (onset, pitch). Implicit index pi = row number.
  // [onset_ms, dur_ms, pitch, velocity]  — mpmify clock (first matched = 0.0,
  // earlier insertions negative)
  "perf": [[0.0, 210.5, 60, 64], ...],
  "align": [[si, pi], ...],        // matches (includes substituted pitches)
  "subs": [[si, fromPitch, toPitch], ...],   // subset of align with wrong notes
  "ins": [[pi, kind], ...],        // kind: 0 slip, 1 restart-first-pass,
                                   //       2 ornament (post-W7), 3 other
  "orn": [[pi, anchor_si, slot, pass]], // ornament provenance (post-W7 only)
  "del": [si, ...]                 // score notes never sounded
}
```

Invariants (checked by the writer):
- every pi ∈ exactly one of {align, ins}; every si ∈ exactly one of {align, del};
- perf sorted by onset; score sorted by (onset, pitch);
- `unattributed == 0` from editsToAlignment, else the sample is discarded.

Sources of truth: espressivo facade `PerformanceData` (score-side ids + exact
performed ms), robustness layer edit log, ornament provenance fields (W7).
