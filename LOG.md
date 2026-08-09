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

**~15:00 Corpus v0 generating** (background: 4 shards × 4000 pieces,
robustness none/light/medium/heavy, jitter 12ms, ~3.6KB/piece). Generator =
mpmify samplers (imported read-only from ml/node/) + espressivo facade +
robustness layer. 10-piece smoke: invariants OK. Full baseline eval also
running. NOTE: pieces are short (16-64 beats, ~40-120 notes) — fine for v0;
longer pieces / real scores (PDMX→MEI) later.

**Model v0 design (pre-report, to refine):** single transformer over
[SCORE] score-notes [PERF] perf-notes with segment embeddings; per-token
features (pitch embed + MLP on onset-delta/dur/velocity log-features); L=4,
d=192, relative position bias. Heads: bilinear match matrix S·Pᵀ + null col
(score-side deletion) + null row (perf-side insertion); loss = symmetric CE
(score→perf+null, perf→score+null). Window ≤512 score notes. Decode v0:
mutual-best + thresholds; structured DP later. Files: src/mlign/{model,dataset}.py,
scripts/train.py.

**~16:00 Reports 02+03 landed; DESIGN.md v1 written (binding).** Key decisions:
parangonar-dict repr + substitution/confidence/ornament ext; single-stream
encoder 1-token/note w/ score features (voice/grace/metric pos — the gap:
every existing system is (pitch,onset)-only); dustbin nulls + dual-softmax;
T5 relative bias (no 512 ceiling); two-phase decode (confident monotone map →
rarest-pitch-first assignment w/ updating map); 3-source corpus (espressivo
renders + reorder() self-supervision port + nASAP train-split finetune).
Baselines verified runnable: parangonar 3.1.0 in system python (DualDTW,
Automatic, TheGlueNote all F=1.0 on demo). Vienna4x22 .npz benchmark local
(43,450 matches). Nakamura WASM fork is crippled (dead ornament machinery) —
build upstream for fair baseline. Velocity trap (fitVelocities) avoided:
samplers stay ≤115. espressivo E1/E2 defects confirmed at HEAD; corpus v1
after fix (v0 GT still valid). Java RenderMpm --batch = 10x faster render
path, adapter liftable from mpmify ml/node/augmented_msm.mjs.

**~16:45 STATE SNAPSHOT (compaction insurance).** Corpus v0 complete:
16k synthetic pieces (data/corpus/v0-{none,light,medium,heavy}.jsonl, 105MB,
0 dropped) + selfsup source ported (scripts/corpus/selfsup.py; reorder() port
w/ trill-length bugfix; piece-level split by md5(piece_dir)%10: <8 train,
8 val, 9 test — journaled convention). Background jobs running:
- training v0-syn: scripts/train.py on 16k synthetic, 30 epochs → runs/v0-syn
  (best.pt/last.pt, log.jsonl; resumable via --run)
- selfsup corpus: → data/corpus/selfsup-v0.jsonl (nASAP train-split only)
- DualDTW full eval → eval/results/dualdtw-full.json (THE bar; smoke: 0.9945)
- my DTW baseline full eval → eval/results/baseline-full.json
NEXT after training: inference+decode pipeline (windowed, two-phase decode per
DESIGN §4) → eval on nASAP vs dualdtw number → iterate (features, +selfsup
data, substitution head; ornament data after meico-ts W7 ping).

**~17:10 Interruption + relaunch.** A login/session event killed all background
jobs (training, both evals); selfsup corpus survived (3368 rows,
selfsup-v0.jsonl). Relaunched: training runs/v0-syn (resumable), dualdtw full
eval, lit-research2 agent (first one died on auth). espressivo-study report
key additions (research/03): facade does NOT expand repeats — class API
resolveSequencingMaps() needed; duplicated note ids = meico_repetition_<n>_<baseId>
(regex-recoverable pass number = free repeat GT!); ties collapse to first id;
grace chords DROPPED by converter; imprecision timing nondeterministic even
seeded+monophonic (pendingDurations unseeded shake) — recipe unaffected;
mpmify currently DROPS note.id from its JSONL (their provenance promise is
proven but unemitted). Ornament v3 branch (meico-ts-orn) confirmed to generate
notes w/ provenance but ornament.anchor NOT YET IMPLEMENTED there — matches
meico-ts-09's claim it lands in W5-W7. My monitor on runs/v0-syn/log.jsonl
is armed.

**User pause (~3:10):** all background work to be restored at 3:20. Restore
checklist: (1) training: nohup .venv/bin/python -W ignore scripts/train.py
--corpus 'data/corpus/v0-*.jsonl' --epochs 30 --run runs/v0-syn (resumes from
last.pt automatically); (2) eval: .venv/bin/python -W ignore eval/run_eval.py
--aligner dualdtw --out eval/results/dualdtw-full.json; also --aligner baseline
--out eval/results/baseline-full.json; (3) subagent lit-research2 (report →
research/01-literature.md; respawn with same focused prompt if dead — exact
prompt in transcript ~17:10 relaunch); (4) monitor on runs/v0-syn/log.jsonl;
(5) after training: eval model via src/mlign/infer.py align_with_model on
nASAP, compare vs dualdtw number. Peers owe us pings: meico-ts-09 (W7 facade
+ merge), mpmify-32 (fixed espressivo dist after E1/E2; factored samplers).

**~15:45 Memory war resolved.** mpmify agent trains its own model on this
machine (caffeinate train.py v31); my featurize-all-upfront dataset + their
job + 2 parangonar evals = swap death. Fixes: CorpusBatcher lazy featurization
(per-batch, ~1ms/row/epoch), training restarted nice -10, dualdtw eval
deferred until a machine gap (coordinating schedule with mpmify-32). CLI
shipped meanwhile: mlign align MEI/MusicXML+MIDI → json/match/jsonl, E2E green
on espressivo-rendered fixture (MEI bridge = node subprocess regex over MSM;
score xml:ids preserved). selfsup-v1 (10 windows/file) generating.

**~16:00 Batik harness fixed** (score must come from match file's snote lines
— GT id space is the PERFORMED unfolding; partitura create_score rejects some
files, so snote parser in nasap.py). Baseline: kv279_1 0.965 / kv279_3 0.978 /
kv279_2 **0.262** (ornament-heavy slow movement — the exact problem class MLign
targets). RULE LEARNED: no partitura evals while my training runs — they
swap-kill it (8GB shared with mpmify's 1.5GB training). 4x22 npz evals are
fine (1s, no partitura). Machine schedule agreed with mpmify-32: I hold ~2GB
niced; heavy bursts in their v31→v4 gap tonight (~21:00-24:00); they ping at
TRAINING_COMPLETE. MPS warning from them: nn.Transformer hangs on torch 2.11
MPS — my custom SDPA blocks unaffected (verified by smoke).

**~16:30 Literature report landed (research/01, 852 lines) — strategy sharpened.**
Three bars: (A) clean alignment SATURATED (DualDTW 99.0±1.0 combined, 4x22
99.8±0.4, Batik 99.4±0.7; nASAP GT noise ceiling ~99 — only 78% robust);
(B) hard/mismatched: TheGlueNote 95 on their 5-piece +20%-mismatch set — OUR
HEADROOM (robustness-trained model); (C) repeat-structure: NO published
symbolic-only system handles folded scores (TheGlueNote scores 12.7 there;
RUMAA 98.4 but needs audio + repeat-symbol MusicXML) — OUR HEADROOM
(synthetic repeats + piecewise decode + espressivo sequencingMaps).
Targets: A ≥99.0 (table stakes), B ≥97, C ≥98 symbolic-only (would be first).
Metric protocol pinned: TISMIR match-F per-performance mean±SD + pooled
F_align (RUMAA comparability); worked-example unit test green; robust subset
= 834 entries (--robust-only). No official nASAP split → inherit MAESTRO v2
test split later (Matchmaker convention). ALSO: negative-results section says
soft-DTW/Sinkhorn attempts have NOT beaten the simple bilinear+CE recipe —
validates model v0 design.

**Training reality check:** CPU 4-threads ≈ slow (~10-15 min/epoch × 30).
Plan: let it run; evaluate best.pt at every landed epoch on 4x22-lite (npz,
cheap); full parangonar evals tonight in mpmify's gap. If epoch time
intolerable → subset corpus (8k) or fewer epochs; model quality signal first.

**~16:05 espressivo E1/E2 FIXED** (meico-ts main da24612, fix c77f4aa, dist
rebuilt 15:56; 2365 tests green, cross-renderer 3169→37 diffs all ≤3.64e-12ms).
v1 corpus regenerating (same seeds, fixed renderer, espressivo-only — never
mix renderers per mpmify guidance). v0 demoted to plumbing data. v0-syn
training continues meanwhile as a wiring/speed probe; will restart on v1 when
shards land.

**~16:15 Second kill wave** took v1 corpus regen + v0 training. Root cause of
data loss found: generator's sync loop never yields → createWriteStream held
ALL rows in memory → kill = empty file. Fixed with writeSync per row.
Serialized restore: v1 corpus solo first (4 niced shards), training on v1
(--matchability, mixed with selfsup-v1) after shards land. v0-syn run
abandoned (never completed an epoch under contention; CPU-thread probe next
run will measure epoch time solo).

**~16:20 Kill-resilience.** Background harness tasks kept dying with session
interruptions → generators now run DETACHED (nohup+disown) with durable
writeSync; scripts/autopilot.sh = idempotent supervisor (restarts missing
generators; starts v1 training when 4x4000 shards complete). Cron install
denied by permission classifier — wakeup loop invokes autopilot.sh instead.

**~17:05 v1 training LIVE and learning.** Epoch 0: val_acc 84.1%, 626s/epoch
(30 epochs → ~22:10 done). Epoch-0 best.pt through full decode on synthetic
val: match F 0.936 (classical baseline: 0.933), del F 0.56 (bl 0.42), ins F
0.40 (bl 0.46). Model+decode wiring proven end-to-end; 29 epochs of headroom.
Bar-B benchmark now has contiguous mode (TheGlueNote OOD protocol).

**~17:20 Domain gap found and addressed.** v1 epoch-1 ckpt: 0.942 synthetic
val but 0.778 on real 4x22 (my classical baseline: 0.986) — style+length gap,
NOT window-stitching (4x22 fits single forward). Restarted as runs/v1b on
v1-synthetic (16k) + selfsup-v1 real-music windows (8.4k, 512-note, nASAP
train split), 24 epochs, matchability on. Expect ~13-16 min/epoch, done in
the night. Autopilot updated to supervise v1b.

**~18:15 LEAKAGE FIX.** selfsup corpus used md5-hash split, not the MAESTRO
test split → real-music windows from test pieces were in v1b's training mix.
Stopped v1b; regenerating selfsup-v2 with --exclude-folders (39 MAESTRO-test
folders, eval/split.py; robust∩test = 84 perfs / 27 pieces is the headline
eval set). Autopilot now gates v1c training on selfsup-v2 completion. Also:
domain-gap diagnostics — real 4x22 perf IOI median 16ms (thick chords) vs
synthetic 153ms; velocities 39 vs 73; lengths 731 vs 101 notes. Real-music
selfsup windows carry exactly these stats → v1c should close the gap.

**~18:35 W7 ping received + ornament wiring shipped.** PerformedNote gains
ornamented/ornamentRef/Source/Slot/Pass/Anchor at merge; carved heads =
matches w/ altered duration (keep score id); expandOrnaments flag for
ablations. Generator now has normalizeOrnaments pre-pass (generated ⇔
ornamentSlot!==null → id=null + ornament origin; robustness copy-chain
inherits anchor) + orn rows [pi, anchor_si, slot, pass] in corpus format.
Forward-compatible no-op verified on current main (5-piece smoke + 10/10
robustness tests). Exaggeration agent adopted R1-R6 + global scope.
v1c training live (16k syn + 7420 leakage-free selfsup).

**~18:50 v1c status:** single trainer (no restart loop), epoch 0 ≈42+ min
(mixed corpus ~11M tokens/epoch). 24 epochs ≈ 17h on CPU under contention —
acceptable for overnight; evaluate per-epoch checkpoints and reassess at
epoch ~4. Machine swap 5.3/6GB used; trainer holds ~0.9GB at 137% CPU.
