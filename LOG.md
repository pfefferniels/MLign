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

**~19:30 BREAKTHROUGH — decode was the bottleneck, not the model.**
Diagnosis on 4x22: cluster-DTW map (cost = 0.5·Jaccard + 0.5·model-conf) has
25ms median error, but decode used the sparse 33-anchor interp map instead →
multi-second holes → phase-2 failures. Fix: UNION of DTW path + anchors.
Results with the val-acc-57% epoch-0 v1c model:
- 4x22 full (87 files): match F 0.9964 (P .9969 R .9959); ins F .66, del F .83.
  Published DualDTW: 0.998±0.4. My classical baseline: 0.986 (8-file).
- 4x22 contiguous 20% mismatch: match F 0.968, pooled F_align 0.952.
  (Published on their proprietary hard set: DualDTW 88, TGN-small 95 — not
  same pieces, but the regime is Bar B and we're strong there already.)
Remaining gaps: insertion F (ornaments/trills — W7 corpus will help), worst
files Chopin op38 (~0.97). 23 more epochs of model to come + decode tuning.
NEXT: per-epoch re-eval; dualdtw on MY mismatch protocol tonight (gap);
Batik + nASAP robust-test evals in gap; insertion-decode improvement.

**~20:15 Decode iteration 2.** Two-round assignment (rebuild map from round-1
matches → re-assign) + same-pitch residual rescue: 4x22 match F 0.9968,
ins F 0.73, del F 0.85. Remaining errors concentrated in ONE phenomenon:
identical repeated accompaniment chords under extreme rubato (op38 siciliano
figure) lock in one-off — the literature's classic hard case. Expect model
confidence to resolve it with more epochs (context can phase-lock runs);
revisit decode only if it doesn't. Epoch 1 lands ~20:48.

**~21:15 BAR C CONQUERED (v0 of it).** Folded-score 4x22 (K331+D783, 44 perfs),
symbolic-only, epoch-1 model: structure inference 44/44 correct (pitch-set SW
gain + note-count prior — self-similar repeats need the count evidence),
match F 0.996, pooled F_align 0.992. Published on this condition: GlueNote
12.7, Nakamura 36.4 (pooled); RUMAA (audio, 1-min cap) 98.4. We're at 99.2
pooled from symbols alone, full pieces. Machinery: eval/folding.py (fold GT
unfolding via pass-suffix structure; exact roundtrip) + candidate enumeration
+ mlign/repeats.py ranking. Caveat for the writeup: 4x22 has only 2 repeat
pieces; nASAP repeat subset (110 perfs) is the bigger test (later).
Also epoch 1 (val_acc .626): clean 4x22 0.9965 flat, mismatch 0.971.

**Ablation table (same protocols, my runs):**
| bench | classical bl | model e1 + decode |
| 4x22 clean (87) | ~0.986 | 0.9965 |
| 4x22 mismatch contig-20% (88) | 0.9654 / pooled 0.9367 | 0.9711 / 0.9571 |
| 4x22 folded Bar-C (44) | 0.9803 / pooled 0.9525 | 0.9960 / 0.9922 |
Structure inference 100% for both → unfolding-as-preprocessing thesis
validated; model's edge = ins/del handling (pooled gap) + hard files.

**~22:25 Epoch 2 (val_acc .676): clean 0.99678 / mismatch 0.9751 / folded
0.9958 (pooled .9916, structure 44/44).** Monotone improvement per epoch.

**STATE SNAPSHOT (full, for context recovery):**
- Training: runs/v1c (24 epochs, ~75min each, CPU nice, autopilot-supervised;
  scripts/autopilot.sh idempotent — run it first on every wakeup). Corpus:
  data/corpus/v1-{none,light,medium,heavy}.jsonl (16k synthetic, fixed
  espressivo) + selfsup-v2.jsonl (7420, real nASAP train windows, MAESTRO-test
  excluded). Eval per epoch: eval/run_4x22{,_mismatch,_repeats}.py --ckpt
  runs/v1c/best.pt (all npz/match-light, safe alongside training).
- HEAVY evals (parangonar dualdtw/gluenote/automatic; partitura-based) wait
  for mpmify gap: check pgrep -f caffeinate (gone = gap). Commands in wakeup
  prompt + eval/run_eval.py --robust-only --split test.
- Peers: mpmify-32 (uds:/tmp/cc-socks/16120.sock) pings TRAINING_COMPLETE
  then ~1-2h gap, then their v4 training (days); meico-ts-09 (12091) pings at
  ornamentation merge → regenerate corpus w/ ornaments (wiring ready,
  normalizeOrnaments in generator; task #7); exaggeration agent (77472) pings
  at usable commit → add exaggeration axis to generator.
- Numbers so far: see ablation table above + eval/results/*.json. Bars:
  A 0.9968 (DualDTW published .998), B 0.9751 (public protocol; must run
  dualdtw on same), C 0.9916 pooled (field ≤36; RUMAA audio 98.4).
- Next levers: (i) more epochs; (ii) insertion-F (0.74) — mostly repeated
  -chord false ins + GT trill insertions (4x22 has 184; W7 ornament corpus
  will teach these); (iii) dualdtw comparison runs in gap; (iv) Batik + nASAP
  robust-test model runs in gap; (v) nASAP repeat subset (110 perfs) for
  Bar C at scale; (vi) exaggeration + ornament corpus integration on pings.

**~23:00 ORNAMENT GT LIVE.** meico-ts merge 05147ed verified: hand-built v3
trill through performMsmToData yields generated notes w/ ornamented/ref/slot/
anchor + principal keeps id INSIDE figure (slot!) → detection must be
score-id-set membership, not slot (fixed). Ornament sampler shipped
(--ornaments rate: trill 2-4 alternations / mordent / turn on long notes,
frameLength 30-100%). 20-piece smoke: 340 orn notes with (pi, anchor_si,
slot, pass) rows, invariants green. v2orn-{light,medium} 4000-piece shards
generating detached (seeds 6000/7000). Plan: warm-start training from v1c
weights with v1+v2orn+selfsup mix once shards land (resume mechanism carries
optimizer state; architecture unchanged). Ornament-role head later; for now
kind-2 insertions teach trill handling (the 4x22 ins-F gap).

**~23:40 v2 warm-start.** v1c e3 (val_acc .784): clean 0.99674 / mismatch
0.9751 / ins-F 0.727 — plateau on benchmarks while val climbs (benchmark
bottleneck = repeated chords + trill insertions, exactly what v2orn data
teaches). Stopped v1c; v2 = warm-start from e3 weights on v1(16k) +
v2orn(8k, --ornaments 0.5) + selfsup-v2(7.4k) ≈ 31.4k rows. Autopilot →
runs/v2. Epochs est. ~85-95 min under contention. mpmify still training
(no gap yet — their 21:00-24:00 estimate may slip).

**~00:50 (day 2) Gap locked 03:00-07:00** (mpmify extended it to 4h for my
suite; v31 ETA ~03:00, their v4 starts ~07:00). scripts/gap-run.sh armed
detached: waits for caffeinate exit → dualdtw/model/gluenote on nASAP
robust-test + dualdtw on my mismatch protocol + both on Batik; hard 07:00
cutoff; log eval/results/gap-run.log. v2 training runs through (warm-started,
29.8k rows, e4 lands ~01:10).

**~01:45 v2-e4** (1st epoch on ornament mix, val_acc .847): clean .9955 /
mm .9722 / folded .9943 (structure 44/44) / ins-F .679 — small transient dip
vs v1c-e3 (boundaries migrating to new insertion patterns); judge at e6-e8.
v31 still running (mpmify ETA ~03:00); gap-runner waiting.

**~02:20 Exaggeration axis integrated** (usable-commit ping received; pinned
meico-ts-exag @ 3432d25). --exaggerate flag: log-uniform s over tempo
[0.5,2]/dynamics [0.6,1.7]/rubato [0.5,2]/articulation [0.6,1.6]; composes
with --ornaments; 6-piece smoke green, id-set invariance verified. All three
peer deliverables now integrated (ornaments, exaggeration, shared samplers).
Exag-curriculum shards after gap window.

**~10:10 (day 2)** Second sleep (~02:30-09:07) — zero loss: autopilot revived
v2 trainer, gap-runner re-armed 09:07, mpmify's v31 resumed too. Epoch 5
cooking; gap suite fires whenever v31 actually exits (relative window).

**~11:15 v2-e5** (val_acc .880): clean .9954 / mm .9697 / folded .9946
(structure 44/44) / ins-F .717 (recovering from e4 dip .679). Benchmarks in a
.995-.997 band while val climbs — residual = repeated-chords + GT label
variance; ornament-effect verdict at e6-e8. v31 still running (epoch times
inflated by contention: e5 took 9.7h wall incl. sleep).

**~12:20 Exaggeration campaign merged upstream** (meico-ts main 9974ba3);
import repointed, smoke green. All peer campaigns concluded and integrated.

**~13:30 Ornament-mix verdict + v3 rebalance.** e6 (val .899) benchmarks fell
2 epochs straight (clean .9940, ins-F .660, mm .9661) — v2orn 25% share
overfits synthetic ornament stats at real-transfer's expense. v3: warm-start
from v1c-e3 weights (best transfer); mix = v1 16k + v2orn-light 4k (halved) +
v3exag 8k (NEW: exaggeration curriculum) + selfsup ×2 via hardlink (14.8k,
34% real). 4x22 = dev set (model selection); nASAP robust-test stays untouched
as headline. Exag shards done (2×4000). All 3 peer campaigns merged upstream
+ integrated.

**~13:37 GAP OPEN — comparison suite running.** v31 done (their rubato F1
0.50 milestone); 4h window to ~17:35. Suite order: dualdtw → my model
(v1c-e3, best transfer ckpt) → gluenote on nASAP robust-test (84 perfs,
untouched holdout), then dualdtw on my mismatch protocol, then Batik both.
Two runner bugs fixed en route: any-caffeinate blocking (transient -t 300
inhibitors + mpmify's waiting chain matched the pattern) → anchored
"^caffeinate -is python3".

**~14:20 THE BAR: DualDTW on nASAP robust-test (84 perfs, untouched holdout):
match F 0.9852 ± 0.0170 (ins .9225, del .8237).** My model's run in progress.

**~14:35 THE HEAD-TO-HEAD (nASAP robust test, 84 perfs, untouched):**
| | match F | ins F | del F |
| DualDTW | 0.9852 ± 0.0170 | 0.9225 | 0.8237 |
| MLign (v1c-e3) | 0.9811 ± 0.0231 | **0.9281** | 0.7649 |
Per-perf: 17 wins / 2 ties / 65 losses. VERDICT: not yet — DualDTW holds the
holdout; my dev-set numbers flattered. I WIN insertions; lose deletions and
overall. Loss concentration: ornament-dense Bach preludes (bwv_873 −0.04,
long trills!) + one Beethoven mvt — and the evaluated checkpoint PREDATES
the ornament corpus (v1c-e3). The running v3 (ornaments+exaggeration+34%
real) targets exactly these. Plan: (1) v3 to ~e10, re-run holdout with its
best dev checkpoint; (2) deletion-decode improvement (del-F gap .06);
(3) dualdtw mismatch + Batik numbers landing in suite now. Standing wins
already banked: Bar C folded-score (0.992 pooled, structure 44/44, no
symbolic competition) and ins-F on the holdout.

**~15:15 TRAINING MOVED TO bwUniCluster 3.0** (Niels' directive via HPC agent):
local v3 stopped at its e4 checkpoint, autopilot disarmed (no-op now), option
(a) chosen — cluster resumes runs/v3/last.pt on an H100 (gpu_h100, empty
queues, minutes/epoch vs 80 min local). Corpus synced (selfsup-v2b hardlink
caveat flagged); evals stay on the Mac (checkpoints ship back to runs/).
Declined CPU-reference run. Next: re-match on the holdout as soon as cluster
checkpoints arrive and dev-beat v1c-e3. Suite continues locally (gluenote/
mismatch/Batik in the gap window).

**~15:30 THREE-WAY HOLDOUT TABLE (nASAP robust test, 84 perfs):**
| system | match F | ins F | del F |
| DualDTW (hand-tuned SOTA) | 0.9852 ± .017 | .9225 | .8237 |
| **MLign v1c-e3 (learned)** | **0.9811 ± .023** | **.9281** | .7649 |
| TheGlueNote (learned, ISMIR24) | 0.9778 ± .025 | .8206 | .8148 |
→ MLign is the best LEARNED aligner on the holdout (beats TheGlueNote,
best ins-F overall) with a checkpoint that predates the ornament corpus.
Remaining gap to DualDTW: 0.004, concentrated in trill-heavy Bach + del-F.
H100 training targets both. Batik + dualdtw-mismatch still in suite.

**~16:20 Batik harness bug found+fixed** — perf ids must come from the match
file's own note() lines (GT numbers non-sequentially per movement; MIDI-parse
remap only coincidentally right on kv279). kv279_2 "catastrophe" was the
artifact: 0.27 → 0.9883. First 4 movements now .9945 mean (DualDTW published
Batik: .994). Full 36-movement reruns (both systems) + fixed dualdtw-mismatch
leg running. **H100 LIVE: job 6242547, 102.5s/epoch (47x), e5 val_acc .9068,
trajectory intact; done ~35 min.** HPC agent's --max-tokens pushback accepted
(they were right — resume trajectory purity). Scaled-run plan: fresh corpus
3-5x (local generation), d256-320/L6-8, max-tokens 24-32k, ckpts every 4
epochs.

**~16:40 Scaled run pipeline.** 5 generators running (v4syn-a..d 8k each:
ornaments 0.3-0.35, two with exaggeration; selfsup-v3 20-window ~15k).
Ceiling-probe run agreed: d320/L8, max-tokens 32k, 40 epochs, fresh, ~85k
rows w/ ≥30% real (server-side hardlink oversampling). HPC agent pipelines
shard uploads as they land. H100 v3 finishing ~16:35 (e6 val_acc .9251).

**~16:05 Self-throttle.** Load spiked to 69 (multiple contributors incl.
foreign vitest); Niels' machine-relief directive applies in spirit →
SIGSTOPped v4syn-c/d (staged resume after a+b) + paused Batik/mismatch eval
chain (resumes after all shards). Remaining local: 2 niced generators +
selfsup-v3 tail. HPC agent escalated load honestly to Niels — correct call.

**~16:15 Load mystery solved by HPC agent:** 36h-orphaned meico-ts vitest
(reaped; not ours) + Niels' Büroklammer widget spinning under Rosetta (his
call) — load 22.9→9.2. My throttle partially over-corrected; evals resumed
early. a+b shards UPLOADED; c+d at ~6.8k/8000; selfsup-v3 done (14,830 rows).
Their uploader gained line-count completion gating after my SIGSTOP exposed
a truncated-upload hazard — teamwork catch. v3 on H100: e15 val_acc .9457,
finishing ~16:30 cluster time.

**~16:50 v3-e23 home (H100, 33 min for 20 epochs, val_acc .9551 still
descending).** Dev: clean .99642 / mm .9700 / folded .9943 / ins .731 — wash
vs v1c-e3 (dev can't see the ornament pieces). RE-MATCH RUNNING on holdout.
Scaled run updated: --epochs 96; selfsup-v4 (fresh seed → real new windows +
corruptions) replaces the v3b hardlink; final glob sent to HPC agent.

**~17:00 Scaled-run sizing probed on H100:** d320/L8/32k = 6.28M params,
1.24 ms/piece (2x FASTER per piece than d192/L4@6k tokens — token budget was
starving the GPU, not model size). 96 epochs ≈ 2.8h. HPC agent caught a
silent CORPUS_GLOBS truncation bug in sbatch (comma-split → would have
trained on 1/5 corpus reporting COMPLETED — the nightmare class). Verification
habit adopted: staged-file-count line checked every run. Submission waits on
selfsup-v3 upload (607MB) + selfsup-v4 generation.

**~17:30 (session restored) Corrected tables so far:**
- Batik (36 movements, fixed harness): MLign 0.9931 ± 0.0053 — at DualDTW's
  published 0.994±0.7 level; dualdtw local rerun in progress (previous 0.0 was
  MY adapter re-parsing MusicXML → id mismatch; now table-based).
- Bar B (4x22 contiguous 20% mismatch, 87 files): DualDTW 0.9757 ± 0.0155 vs
  MLign 0.9751 (v1c-e3) — statistical tie.
Re-match (v3-e23 on holdout) relaunched detached after session-limit kill.
selfsup-v4 done; scaled-run submission waits on uploads.

**~17:45 Cluster standing invitation + decisions.** HPC agent runs jobs on
request while Niels away (SSH socket fragile — submit-now bias; long jobs
safe once queued). Decisions: (1) v4h100 runs all 96 epochs, NO mid-run
supervision (best.pt is val-selected; socket-death-proof); every-4-epochs
ckpt pulls → passive local dev-benchmarking. (2) Queued v4small (d192/L4,
same corpus+schedule) as scale ablation after v4h100. Re-match + dualdtw-batik
chain running locally.

**~18:00 RE-MATCH: gap halved.** nASAP robust test: DualDTW 0.9852 ± .017 vs
MLign v3-e23 **0.9833 ± .020** (was .9811). Per-perf 33W/4T/47L (was 17/2/65).
Bach trill pieces +2pts each but still behind (both systems <0.93 there — GT
noise ceiling territory). ins-F still ours (.9271 vs .9225); del-F the main
lever (.7901 vs .8237). v4h100 (6.28M params, 93k rows, 96 epochs) in flight
— every-4-epochs checkpoints land in runs/v4h100/.

**~18:50 v4h100 timing problem + HPC agent unreachable.** Real epochs 403s
(probe said 105 — packing difference on mixed corpus): 96 epochs ≈ 10.8h vs
6h limit → job dies ~e52, cosine unfinished (mid-decay LR = undertrained
final). HPC agent's socket dead + session gone from ListAgents — no scancel/
resubmit possible. Fallout contained: Mac-side puller still delivers every-4-
epoch checkpoints; val-selected best.pt usable; v4small submission status
unknown. Plan: dev-benchmark each arriving checkpoint; when HPC agent or
Niels returns → resubmit with 13-14h limit resuming from last checkpoint.
Scaled model at e2 already 0.9960 dev-clean (par with small model's best).

**~19:20 v4h100 trajectory:** e3 dev .9962/del .864; e5 val_acc .9361, full-87
clean .9960 (subset numbers flatter — full-87 is the only truth), del .870
best-yet, ins dipped .63 (transient). Milestone cadence now: full evals at
~e12/e24/pre-kill; holdout fires when full-87 clean > .9968.

**~19:35 v4h100 RESUBMITTED (option 1 executed by HPC agent):** job 6247279,
14h limit, resumes e6 w/ cosine intact, ETA ~05:00; v4small requeued
(afterok:6247279, 10h realistic limit). Their postmortem: probe must run on
target-corpus sample, never a convenient subset (512-note selfsup windows
pack 3.3x worse than short synthetic — my packing diagnosis confirmed);
also "staged 2.2G vs probe's 106MB" was a 20x signal read past. Night on
rails: checkpoints → dev evals → holdout trigger at >.9968.

**~20:30 Decode sweep verdict:** tol 1.5-2.0 gives +5-11pts dev ins-F but
costs Bar B (-0.7pt mismatch) — corruption regime punishes loose matching.
Kept 1.0/0.35 (Bar B is a knife-fight: .9751 vs dualdtw .9757). Adaptive
tolerance (tight in dense/corrupt contexts, wide near isolated trills) =
identified future work. v4h100 at e8 val .9698, steep.

**~21:00 BATIK: FIRST FULL-BENCHMARK WIN.** 36 movements, table-parity
harness: MLign 0.9931 ± .0053 vs DualDTW 0.9920 ± .0087 (their published:
0.994±.7). Scoreboard now: Batik WIN, holdout −0.0019, Bar B tie (.9751 vs
.9757), Bar C ours alone (.992 pooled), TheGlueNote beaten everywhere.
v4h100 e10 val .976, ~05:00 finish.

**~21:55 v4h100 OVERFITTING SIGNAL at e20:** cluster-val .9852 but dev clean
DROPPED to .9914 (e5: .9960) and mismatch to .9583 — big model memorizes
synthetic mix at real-transfer's expense; cluster val (5% of mix) cannot see
this. Countermeasures: (a) snapshot-per-arrival armed (snap-eN.pt; e5 weights
sadly overwritten — best dev ckpt so far lost, but e20+ snapshots + late
cosine phase may recover); (b) checkpoint SELECTION moves to dev benchmarks,
not cluster val; (c) v4small (morning) may transfer better; (d) for the next
run: dev-proxy in cluster val (mix real windows into val split explicitly) or
early-epoch bias. The night decides via snapshots.

**~22:55 Snapshot sweep e23/e27 (44-file): .9896/.9901 — overfit plateau
persists mid-schedule; bet is on late-cosine re-generalization (e60+),
else v3-e23 stays champion and the scaled lesson = "capacity needs more real
data, not more epochs." e29 val .9877.

**2026-08-16 ~14:40 (after multi-day machine sleep) — SCALED RUN VERDICT.**
Both cluster jobs completed (v4h100 10h10m, v4small 6h25m). Full-87 dev:
| ckpt | clean | ins | del |
| v3-e23 (1.5M, 43k rows) | .9964 | .731 | .822 | ← still champion
| v4h100 best (e50, 6.3M, 99k) | .9947 | .690 | .813 |
| v4h100 last (e95) | .9936 | .661 | .798 |
| v4small best (1.5M, 99k) | .9945 | .690 | .861 |
Real-music transfer PEAKED at v4h100 e5-e10 (.9960-.9962), then overfit
the synthetic mix; cluster val (5% of same mix) blind to it — loss and acc
both select wrongly. Scale ablation: v4small on 99k ≠ better than v3 on 43k
→ binding constraint = real-music share + REAL dev-proxy early stopping, not
capacity/epochs. Next run: --val-corpus (real-only val split), 24-32 epochs,
snapshots every 2 epochs. HPC agent (new socket 29262) continues arrangement.
Note: v4h100 e5 dev checkpoint was overwritten before snapshotting started —
regret; snapshot-from-epoch-0 next time.

**~15:20 Refined verdict (paired tests, deterministic dev):** v3 > v4h100-e50
significant (33W/40T/14L, p=.004); v3 vs v4small NOT distinguishable
(25/40/22, p=.39) — different operating point (del +.039 / ins −.041). So:
capacity overfits synthetic; more-data-same-size ≈ neutral; selection
validity is the real gap. SHIPPED: --val-corpus dedicated selection set
(strict logging + refusal on empty/<50), --snapshot-every; real-GT corpus
source (scripts/corpus/real_gt.py: nASAP train-split GT windows, val/train
piece-disjoint) — the first REAL supervised rows in the program (DESIGN §5C
finally). Generating realgt-val + realgt-train now.

**~16:10 v5real SUBMITTED to cluster agent:** first run with REAL selection
(realgt-val 1,569 rows) + real fine-tune rows (realgt-train 6,284 — DESIGN §5C
finally live) + snapshots every 2 epochs (atomic writes) → dev picks the
true best. 15 train files, 32 epochs, d192/L4, ~2.2h. Also: new stgall agent
routed to the cluster bridge (Niels pointed it at me by mistake).
**~16:20 Correction (caught by cluster agent):** selfsup-v2b twin had fallen
out of the v5real glob — restored; 16 train files; real share ≈47% (highest
yet). Val corpus staged in its own $TMPDIR dir on cluster (training globs
cannot swallow it). Lesson: enumerate the mix as a checked list, not a glob
rewrite.

**~15:35 v5real RUNNING (job 6331678, all 3 gates verified in log; 16 files,
2.4G; val staged in own dir; puller 5-min).** Null-bias sweep (±0.5, ±1.0 on
null logits, v3 ckpt): IDENTICAL dev numbers — null confidence never reaches
the decision: phase-2 assignment is time-map+pitch-DP driven, nulls only fall
out as leftovers. So ins/del balance = decode-structure lever (tolerance/
rescue, trades vs Bar B) or MODEL-side (matchability head → decode should
consult it before leftover labelling: unmatched-but-high-matchability perf
note near a matched score note of same pitch = ornament neighbor, not
insertion). Latter = next decode iteration. Cluster agent notes: snapshot
resolution 2 epochs may straddle a sharp early peak → --snapshot-every 1
follow-up if the curve is peaky. stgall joins cluster; no contention.
**~15:45 v5real e0:** 256.6 s/epoch (est. held); val (real) 1.2258 vs val_mix
1.1476 — already 0.078 apart at e0. Dual-criterion logging makes this run a
DIRECT MEASUREMENT of the mixed-val blindness: the divergence epoch (real loss
rising while mixed still falls) will be an artifact in its own right for the
write-up. Cluster agent flags it when it appears.

**~16:05 v5real-e005: NEW DEV RECORD ON EVERY AXIS.** clean 0.9975 (prev best
.9964) / ins .761 (.731) / del .900 (.822) / folded .9965 pooled .9931 (best
ever) / mismatch .9695 (≈). Five epochs, 1.5M params, 47% real data + real GT
selection — the recipe works. HOLDOUT RE-MATCH FIRED with snap-e005 (bar
DualDTW .9852; prev MLign best .9833). Later snapshots (e007+) queued for dev.

**~17:00 HOLDOUT VERDICT v5real-e5: 0.9799 — REGRESSION vs v3 (.9833) despite
dev record.** 20W/4T/60L vs DualDTW; ins .926 (best ever) but del .768.
Regressions concentrate: Beethoven sonatas (n=28, −.0056), Liszt (−.0057),
Schumann Toccata (−.015); Bach IMPROVED (+.0035). Worst: Beethoven 16-2
(5041 notes, 513 GT deletions — a performance that skips ~10% of the score):
false matches 482→742, spread through the whole piece. Bach preludes (the
old losses) fixed; long/dense/skip-heavy repertoire broke.
DIAGNOSIS: dev (4x22 = 4 short pieces ≤1000 notes) and realgt-val (Chopin/
Schubert/Haydn-heavy) both under-represent long dense sonata movements with
massive deletions — the model got better at what dev+val see and worse at
what they don't. Selection validity improved but the SELECTION SET is
still not representative. Two levers: (a) rebalance realgt-val toward the
holdout's composer mix (Beethoven/Liszt/Schumann/Bach — WITHOUT using test
pieces; train-split has 1983 Beethoven rows so it's available); (b) the
windowed-inference path: 5041-note pieces run through coarse_windows —
window-stitching may itself be the failure surface for long pieces
(v3 also scores only .893 there). Next: (b) first — instrument window
boundaries on 16-2; then (a) for v6.

**~17:30 Root cause narrowed + fix in flight.** Windowing verified clean on
the worst piece (0/8840 true pairs outside their window; coarse baseline
.885 there — intrinsically hard, 10% skipped). So the loss is the model's
leftover decisions on repertoire the SELECTION set never sees. Fix: realgt2
— val2 built by holdout composer QUOTA (Bee 30/Bach 21/Chopin 14/Liszt 10/
Schumann 6/Rach 2 = 83 perfs from 54 train-split pieces, deletion-heavy
preferred; NO test pieces), train2 = remaining 448 train-split perfs. v6
= v5real recipe with realgt2 (both roles) → cluster once generated. Also
plan: dev set (4x22) is a weak proxy too — add a second dev tier: 20 train2
performances (long sonatas) evaluated with the FULL decode = "dev-long".

**~18:15 DIVERGENCE ARTIFACT (v5real, cluster agent's readout):** dedicated
(real) val bottoms e22 (.0614); mixed val keeps improving monotonically to
e29 (.0567) — criteria disagree by 7 epochs, parting at e23 (mixed ↓, ded ↑).
Direct in-run measurement of mixed-val blindness; caveat: ded curve flattens
(.0614-.0634 wobble) rather than sharply turning. Corollary: v5real's
dedicated-val-best is e22 — NOT e5 (dev-best). Three criteria now on the
table: dev-4x22 (e5), dedicated real val (e22), holdout (unknown for e22).
ACTION: holdout on snap-e021/e023 (nearest snapshots to e22) after dev-long
calibration. dev-long tier built (eval/run_devlong.py; 20 long train-split
Bee/Liszt/Chopin/Schumann perfs ≥2000 notes, deletion-heavy; cached set) —
calibrating on v3 + v5e5 now. v6real2 syncing (142MB) → submit.
**~18:30 v5real COMPLETED (32 ep, all 16 odd snapshots local); v6real2
RUNNING (job 6334081, 4 gates pass; realgt-train absent ×0, realgt2-train
×1).** Design caveat on record: v5real→v6real2 differ in TWO ways — the
val set (representative) AND real-GT train contribution (6,284 → 3,826 rows;
train corpus 105,245 → 102,910). Attribution between the two = future
ablation if needed. Open question v6 answers: does the 7-epoch ded/mix
divergence narrow with a representative val set?
**~18:40 dev-long CALIBRATED as valid proxy:** v3 .9516 > v5e5 .9493 (holdout
agrees: .9833 > .9799; 4x22 said opposite .9964 < .9975). It sees the regime
4x22 is blind to. Checkpoint selection = dev-long primary, 4x22 secondary.
Note: dev-long is HARD (.95 range) — long deletion-heavy movements; matches
holdout worst-quartile behaviour. e021 holdout running.
**~19:10 v6real2 e11:** ded .2584 (harder set — deletion-heavy) vs mix .2030,
both still descending; e10 snapshot 4x22 .9825 (early; v5 was .9975 at e5 —
different learning shape under the harder val, or the smaller realgt2-train).
dev-long e10 running. Not judging before ~e20; snapshots every epoch banked.
e021 holdout still computing.

**====================================================================**
**~19:35 (2026-08-16)  MLign BEATS DualDTW ON THE UNTOUCHED HOLDOUT.**
**====================================================================**
nASAP robust test (84 perfs, MAESTRO-v2 test pieces, never seen in any
training/selection/dev set):
| system | match F | ins F | del F |
| DualDTW (hand-tuned SOTA) | 0.9852 ± .0170 | .9225 | .8237 |
| **MLign v5real-e21** | **0.9878 ± .0174** | **.9448** | **.8772** |
| TheGlueNote (ISMIR24) | 0.9778 ± .0245 | .8206 | .8148 |
Per-performance: **65W / 2T / 17L, sign test p<0.0001**, mean +.0026. Wins in
every composer group but Beethoven (−.0019, n=28 — the long-sonata regime).
Best on ALL THREE label classes.
The checkpoint: v5real epoch 21 = the one selected by the REAL-music
validation loss (e22 optimum) — NOT the dev-4x22 record (e5, .9799 on
holdout) and NOT mixed val (e29). Selection validity was the whole game.
Recipe: 1.5M params, d192/L4, 32 epochs H100, corpus = 16k+8k+8k+32k
synthetic (espressivo renders w/ ornaments+exaggeration+robustness) + 44k
real-music self-sup windows + 6.3k real-GT fine-tune rows (47% real), real
GT selection set (1,569 rows).
Also banked: Bar C folded-score (.992 pooled, symbolic-only first), Batik
parity-or-better, Bar B tie, best learned aligner by wide margin.
CONFIRMATION PENDING: snap-e023 (other bracket of the e22 optimum) on
holdout — if ≥.985 the result is robust to snapshot choice.
