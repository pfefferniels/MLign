# MLign journal

Working journal of the autonomous MLign build. Newest entries at the bottom.
Durable state lives here + research/*.md; session task list mirrors phase status.

## 2026-08-09 — Day 1

**~10:45 Kickoff.** Mission: MEI score + performed MIDI → (near-)perfect
note-level alignment; beat parangonar. Repo initialized.

**Recon results:**
- espressivo = ~/Projects/meico-ts (npm "espressivo"): MEI/MSM+MPM → expressive
  MIDI, TS port of meico. NOT at ../espressivo.
- Local baselines available: original-parangonar (Python), parangonar (C++/WASM
  port!), thegluenote-main (incl. release checkpoints 103M, preprocessed nASAP
  pairs 29M, Vienna 4x22 testing data 227M), AlignmentTool (Nakamura, C++).
- Hardware: Apple M1, 8 GB RAM, 7-core GPU, torch 2.11.0 + MPS available.
  Consequence: compact model + strong inductive bias + unlimited synthetic data
  + structured decoding; streaming datasets; multi-day training is fine.

**Peer coordination (see research/00-coordination.md):** both peers replied
day 1. Ground truth = espressivo facade perform-once-extract-twice; ornament
provenance contract D10/D15 + ornament.anchor addendum (negotiated by us).
Deal with mpmify-32: shared generator, we own robustness/error/repeat layer.
meico-ts-09 pings us at W7 + merge (~1-2 days) for generated-note ornaments.

**Smoke test (scratchpad/smoke.mjs): PASSED.** MEI fixture → convertMeiToMsmMpm
→ performMsm → extractPerformanceData: PerformedNote{id:"n1", pitch, date,
duration, velocity, milliseconds{date,end}} per part + renderExpressiveMidi
bytes; deterministic twice-run. Facade path for imprecision runs:
performMsm ONCE → extractPerformanceData(aug) + renderExpressiveMidi({msm:aug},
NO mpm — that path renders performed attributes verbatim).

**Research agents launched (background):** lit-research (papers/SOTA numbers),
code-study (parangonar/TheGlueNote/AlignmentTool internals + baseline
runnability), espressivo-study (facade/MPM coverage details). Reports →
research/01..03-*.md.
