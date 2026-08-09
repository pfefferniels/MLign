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

**~12:00 Robustness layer v1 shipped** (src/robustness/: rng.mjs, robustness.mjs,
gt.mjs + 8 invariant tests green). Pure fn (data, config, seed) → {data, edits};
ops delete/insert/substitute/shift/restart/skip; provenance travels on inserted
notes (origin field), inherited through copy-of-copy chains; GT flattener
editsToAlignment → parangonar-style match/insertion/deletion + perfNotes list
(p0..pN in global onset order). E2E against real espressivo render: clean.
GT convention: replay pass keeps score ids (matches), botched first attempt =
insertions — mirrors nASAP annotation practice.

**~12:30 Clock convention pinned with mpmify** (their review passed all tests):
editsToAlignment emits absolute facade ms (≥0); shiftToMatchedZero(perfNotes,
alignment) converts to their convention (first matched onset = 0.0, earlier
insertions negative). Test invocation pinned: node --test 'src/robustness/test/*.mjs'
(bare dir form phantom-fails on Node 23). 9/9 tests.

**Datasets landed** (data/benchmarks/, gitignored): asap-dataset 1.9G sparse
(no audio), batik_plays_mozart 120M, vienna4x22 23M (match/ dir present).
Disk after: ~10G free — corpus budget must stay small; stream + gzip.

**~14:15 Full eval loop closed.** eval/: nasap.py (match+tsv GT loaders,
cross-validated 25/25), metrics.py (parangonar-compatible per-class P/R/F,
macro-avg), run_eval.py (runner). src/mlign/: tables.py (ScoreTable via
partitura merge+unfold_maximal — reproduces nASAP "-1" id space exactly;
PerfTable via load_performance_midi — reproduces n0.. perf ids, verified),
baseline.py (onset-cluster DTW + Jaccard + within-cluster pitch pairing).
Baseline on 5 Bach fugue perfs: match F≈0.991, ins F≈0.72, del F≈0.74,
~5.6 s/perf. This is the floor; full-1063 run deferred (≈1.7 h).
1063 GT performances indexed. nASAP GT semantics: tsv rows are perf-notes
(xml_id="insertion" spurious; midi_id="deletion" silent score note).
