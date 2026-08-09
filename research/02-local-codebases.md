# 02 — Local codebases: what we already have on disk

Study of six local repositories relevant to MLign (MEI score + performed MIDI → note-level
alignment). Read-only survey; nothing was modified. All paths absolute.

| Repo | What it is | Runnable now? | Value to MLign |
|---|---|---|---|
| `/Users/nielspfeffer/Projects/original-parangonar` | Upstream parangonar 3.1.0 (Python, CPJKU/Vienna) | **Yes, verified F=1.0** | **Primary baseline + alignment representation + eval harness** |
| `/Users/nielspfeffer/Projects/parangonar` | Copilot-written C++/WASM port of *one* parangonar matcher | Builds; algorithm is broken | WASM/embind build template only |
| `/Users/nielspfeffer/Projects/thegluenote-main` | TheGlueNote (ISMIR 2024) transformer note matcher | Needs 2 one-line fixes | **Augmentation engine + checkpoints + dual-softmax formulation** |
| `/Users/nielspfeffer/Projects/AlignmentTool` | Pfeffer's WASM fork of Nakamura's HMM aligner | Prebuilt wasm works; source doesn't compile | Second baseline (weakened); HMM design ideas |
| `/Users/nielspfeffer/Projects/scorewarp` | Goebl/Weigl SVG score-warping viewer | Yes (static web) | MAPS JSON format; 87 GT alignment files |
| `/Users/nielspfeffer/Projects/aligned-mei` | "As Played By" — MEI alignment editor | Yes (`npm run dev`) | **The MEI alignment representation** |

---

## 1. `original-parangonar` — the reference implementation

Upstream `sildater/parangonar` **v3.1.0**, commit `96431b8`, Apache 2.0. Authors Silvan Peter,
Carlos Cancino-Chacón, Florian Henkel (JKU Linz / OFAI, ERC "Whither Music?"). 8,710 LOC.
File I/O is delegated to **partitura**.

### 1.1 How parangonar represents an alignment

This is the central question and the answer is refreshingly simple: **a flat Python list of
dicts**, one per note-level decision. There is no graph, no matrix, no interval structure.

```python
{"label": "match",     "score_id": "n42", "performance_id": "n57"}
{"label": "deletion",  "score_id": "n43"}                            # score note not played
{"label": "insertion",                    "performance_id": "n58"}   # played note not in score
{"label": "ornament",                     "performance_id": "n59", "type": "trill"}
```

Properties that matter for MLign:

- **IDs, not indices.** `score_id` is a partitura note id (which for MEI input is the MEI
  `xml:id`); `performance_id` is a MIDI note id. This makes the representation
  order-independent and directly serializable.
- **Strictly one-to-one for matches.** No m:n. A trill in the score matched against 12 played
  notes is represented as 1 match + 11 insertions, unless ornament processing is on.
- **`ornament` is read but never written.** partitura's match reader emits `label="ornament"`
  for `ornament(Anchor)-note(...)` lines in v1.0.0 match files
  (`partitura/io/importmatch.py`), and the writer handles it. But **no parangonar matcher ever
  produces an `ornament` label** — `CleanOrnamentMatcher`
  (`parangonar/match/matchers.py:1307`) emits ornament decisions as plain `match`. There is a
  helper `convert_grace_to_insertions(alignment)` (`parangonar/match/utils.py:364`) that
  rewrites `ornament` → `insertion` so that predictions and ground truth become comparable.
  **This asymmetry is a real evaluation trap**: on a match file containing ornaments, a
  perfect predictor scores < 1.0 on `match` unless you normalize first.
- **No confidence, no scores, no alternatives.** Every decision is hard.
- **Grace notes** are ordinary score notes flagged `is_grace` in the note array; they are
  pulled out, given synthetic negative onset offsets (−0.1 beat per grace note,
  `matchers.py:1354-1359`), matched, and mixed back in.

`alignment_dicts_to_array()` (`utils.py:20`) converts to a structured array with
`matchtype` ∈ {`"0"` match, `"1"` deletion, `"2"` insertion} for the Parangonada web viewer.

### 1.2 Input features — what the algorithms actually see

partitura note arrays, verified on the demo file:

```
score:       onset_beat duration_beat onset_quarter duration_quarter onset_div
             duration_div pitch voice id is_grace grace_type divs_pq
performance: onset_sec duration_sec onset_tick duration_tick pitch velocity
             track channel id
```

**But the matchers use almost none of it.** Across all offline matchers the features actually
consumed are:

| Feature | Used by |
|---|---|
| `pitch` (score + perf) | everything |
| `onset_beat` (score) / `onset_sec` (perf) | everything |
| `duration_beat` / `duration_sec` | only for piano-roll rasterization in `AutomaticNoteMatcher`, and for the ornament time window |
| `is_grace` | `DualDTWNoteMatcher` only |
| `voice`, `staff`, velocity, key/time signature, beat position, articulation, dynamics | **never** |

So the state of the art in this family is, functionally, **(pitch, onset) matching**. That is
the single clearest gap MLign can attack: everything MEI knows — voice, staff, beam/tuplet
grouping, metric position, ties, slurs, ornament type, `<app>`/`<rdg>` variants — is discarded.

### 1.3 The four offline matchers

#### `AutomaticNoteMatcher` = `PianoRollNoNodeMatcher` (`matchers.py:1820`)

The 2023 TISMIR method used to build the (n)ASAP note alignments. Four stages:

1. **Coarse DTW** on binarized piano rolls. `alignment_times_from_dtw()`
   (`preprocessors.py:19`) rasterizes both sides at `time_div=16` steps per beat/second,
   binarizes the performance roll, runs vanilla DTW on the **time axis**
   (`matcher(s_pianoroll.T, p_pianoroll_ones.T)` — the transpose is load-bearing), then
   resamples the path onto a regular score-time grid of spacing `SCORE_FINE_NODE_LENGTH=4.0`
   beats via `interp1d`, and re-adds the minimum onsets. Output: a list of
   (score_beat, perf_sec) anchors.
2. **Windowing.** `cut_note_arrays()` (`preprocessors.py:177`) cuts both note arrays into
   overlapping windows around the anchors, with `sfuzziness=4.0` beats /
   `pfuzziness=4.0` seconds of slack, the performance slack scaled by the local tempo ratio.
3. **Per-window fine DTW + symbolic matching.** Same DTW at
   `SCORE_FINE_NODE_LENGTH=0.25`, then `SequenceAugmentedGreedyMatcher`
   (`matchers.py:88`): for each pitch class independently, map the score onsets into
   performance time and, when the two same-pitch sequences differ in length by `k`, search
   `C(n,k)` combinations (capped at `cap_combinations=100`, randomly sampled above that) for
   the omission set minimizing squared onset distance. Leftovers → insertions/deletions.
4. **Mending.** `mend_note_alignments()` (`preprocessors.py:248`) resolves cross-window
   conflicts by graph traversal (`traverse_the_alignment_graph`, depth 150) and re-runs the
   symbolic matcher on the seams.

#### `DualDTWNoteMatcher` (`matchers.py:1963`) — the default and SOTA

Peter, ISMIR 2023. Not piano-roll based; operates on **onset-level pitch sets**.

1. `OnsetMatcherDTW` (`matchers.py:1666`) builds, for each unique score onset, the *set* of
   pitches sounding there, then DTWs the performance pitch *sequence* against that sequence of
   sets using `DTWSL` with `element_of_set_metric` (cost 0 if the played pitch is in the set,
   1 otherwise). Grace notes excluded.
2. **The same thing again, reversed** (`np.flipud` both sequences). This is the "dual" part.
3. `get_score_to_perf_map()` (`matchers.py:1100`) reconciles the two passes: a score onset gets
   a performance time only if forward and backward medians agree within **0.1 s**; otherwise it
   is dropped. Repeated-pitch blocks across adjacent onsets are handled explicitly
   (`block_by_pitch_by_onset`) to avoid the classic repeated-note collapse. Outliers > 0.1 s
   from the median at an onset are discarded. Result: a monotone `interp1d` score→performance
   time map built only from *confident* anchors.
4. `CleanOrnamentMatcher` (`matchers.py:1307`) does the final note assignment: grace notes are
   mixed back in at −0.1 beat offsets; then **per pitch class**, score onsets are projected
   through the map and `unique_alignments()` (`matchers.py:374`) runs a small 1-D DTW on the
   two onset sequences with an `onset_threshold=1.5 s` gate, keeping the closest unique pairs.
   Unmatched → deletion/insertion.
5. Optionally (`process_ornaments=True`, needs the `score_part`, not just the note array),
   score notes carrying partitura `ornaments` are re-processed: the previously assigned match
   is demoted to an insertion, and the ornament is re-matched against any insertion within
   ±2 semitones inside `[onset−0.25, onset+duration]` in performance time.

The design insight worth stealing: **decouple the time-warping estimate from the note
assignment.** Stage 1–3 produce a robust monotone timing map from high-confidence evidence
only; stage 4 does per-pitch assignment against that map. Both TheGlueNote and Nakamura's tool
converge on the same two-phase structure.

#### `AnchorPointNoteMatcher` = `PianoRollSequentialMatcher` (`matchers.py:1704`)

Identical to `AutomaticNoteMatcher` minus stage 1 — you supply the anchors. `node_array()`
(`utils.py:210`) generates them from an existing alignment at a beat or measure interval, with
optional Gaussian "tapping noise" to simulate human annotation. This is how the (n)ASAP
alignments were bootstrapped from beat annotations, and it is the obvious semi-automatic mode
for MLign.

#### `TheGlueNoteMatcher` (`matchers.py:2043`)

Wraps the pretrained transformer (see §3). Works on any two MIDIs, no score needed. Tiles a
512×512-note window over the full N₁×N₂ grid at stride 256, accumulates the dual-softmax
confidence matrix, runs weighted DTW on its inverse, then hands the resulting onset map to
`get_note_matches_with_updating_map()` (`matchers.py:1537`) — the same per-pitch
`unique_alignments` assignment as `DualDTW`, but iterating **rarest pitch first** and rebuilding
the interpolation map after each pitch, so confident rare-pitch matches progressively refine
the warp used for common pitches. Nice trick, model-agnostic.

### 1.4 Online matchers (`match/online_matchers.py`, 765 LOC)

`OnlineTransformerMatcher` / `OnlinePureTransformerMatcher` use `AlignmentTransformer`
(`pretrained_models.py:41`): a small encoder-only transformer, `token_number=91`,
`dim_model=128`, `num_heads=4`, `num_decoder_layers=6`, sinusoidal PE up to 50,000 positions,
2-class output head — it makes *local* alignment decisions and the pure variant skips the tempo
model. `TOLTWMatcher` / `OLTWMatcher` are Dixon-style on-line time warping over symbolic
sequences with `tempo_and_pitch_metric` (`dp/metrics.py:116`). All four expose `.offline()`
which just streams the whole performance through. Not directly relevant to MLign's offline
goal, but `TempoModel` (`online_matchers.py:21`) is a compact, reusable local-tempo estimator.

### 1.5 The DP library (`parangonar/dp/`, 3,015 LOC)

Genuinely task-agnostic and the most immediately liftable code in the repo:
`DTW`, `WDTW` (arbitrary step patterns + directional weights + penalties), `DTWSL`
(single-loop, for non-vector elements like pitch *sets*), `FDTW` (FlexDTW, Bükey et al. —
flexible start/end points), `NW`, `NW_DTW` (Grachten's Needleman-Wunsch time warping), `WNWTW`,
`ONW`, `BSW` (bounded Smith-Waterman, used for repeat identification),
`SubPartDynamicProgramming`, plus `OLTW`/`T_OLTW`. numpy + numba. Tests in
`tests/test_align.py` pin exact expected paths.

### 1.6 Mismatch module

`RepeatIdentifier` (`mismatch/repeat_identification.py`) infers which score sections a
performance actually traverses, by bounded Smith-Waterman over onset pitch-sets against
partitura's segment graph, enumerating musically valid section sequences (starts at start, ends
at end, valid repeat counts) up to `max_number_of_paths=100000`. `SubPartMatcher`
(`mismatch/subpart.py`) aligns a single monophonic score voice (bass line) against a full
performance. **Both matter for MLign**: MEI scores carry `<expansion>`/repeats, and a real
performance may take different repeats.

### 1.7 Evaluation

`fscore_alignments(prediction, ground_truth, types)` (`evaluate/eval.py:14`) is exact
set-intersection over the alignment dicts, per label type, returning P/R/F. Note the
**semantics**: `filtered_correct = [pred for pred in pred_filtered if pred in gt_filtered]` —
dict equality, so a match is correct only if *both* ids agree. Empty-on-both-sides scores 1.0.

`evaluate_score_following()` (`eval.py:88`) is the timing metric: build score→performance time
maps from GT and from the prediction (via partitura's performance codec, `remove_ornaments=True`),
evaluate both at the score onsets both agree on, and report median absolute asynchrony plus the
fraction under 25/50/100 ms. **MLign should report both families** — F-score on
match/insertion/deletion, and asynchrony percentiles.

`evaluate/plot.py` gives `plot_alignment`, `plot_alignment_comparison`, `plot_alignment_mappings`.

### 1.8 I/O formats parangonar can read/write

- **Vienna match files** (via partitura) — v5.0 and v1.0.0. The demo file
  `parangonar/assets/mozart_k265_var1.match` is v5.0:
  ```
  info(matchFileVersion,5.0).
  info(midiClockUnits,480).
  snote(n9,[C,n],3,1:1,0,1/4,0.0,1.0,[])-note(n0,[C,n],3,683,747,747,70).
  ```
  `snote(id, [step, alter], octave, measure:beat, offset, duration, onset_beat, offset_beat, [attrs])`
  paired with `note(id, [step, alter], octave, onset_tick, offset_tick, sound_off_tick, velocity)`.
  Deletions are bare `snote(...)`, insertions are `insertion-note(...)`, ornaments are
  `ornament(anchor)-note(...)`.
- **Parangonada CSV** — `save_parangonada_csv()` (`utils.py:57`), for the web viewer.
- **(n)ASAP** — via partitura.
- **Piano Precision / Sonic Visualiser CSV** — `evaluate/io.py`.
- **MAPS JSON** — `save_maps()` (`evaluate/io.py:406`). This is the format `scorewarp`
  consumes, but **the schemas differ**: parangonar writes `{"xml_id": str, "obs_mean_onset":
  float, "velocity": int, "obs_num": int}` while scorewarp expects `xml_id` and `velocity` to
  be **arrays** (one entry per chord member). A converter is a five-line fix but it is a real
  incompatibility today.

### 1.9 Running it as a baseline — verified working

The system Python 3.13 already has everything: **parangonar 3.1.0, partitura 1.6.0,
torch 2.11.0, numpy 2.2.6**. I ran all three offline matchers on the bundled Mozart file
(218 score notes, 219 performance notes, GT = 218 matches + 1 insertion):

| Matcher | match F | insertion F | deletion F | wall time |
|---|---|---|---|---|
| `AutomaticNoteMatcher` | 1.000 | 1.000 | 1.000 | 3.1 s |
| `DualDTWNoteMatcher` (with ornaments) | 1.000 | 1.000 | 1.000 | 0.8 s |
| `TheGlueNoteMatcher` (CPU) | 1.000 | 1.000 | 1.000 | 3.6 s |

```python
import partitura as pt, parangonar as pa
perf, gt, score = pt.load_match(filename=MATCH, create_score=True)
pred = pa.DualDTWNoteMatcher()(score.note_array(include_grace_notes=True),
                               perf.note_array(),
                               process_ornaments=True, score_part=score[0])
pa.print_fscore_alignments(pred, gt)
```

Caveat: this file is trivially easy (no deletions, no ornaments, one insertion). Real
benchmarking needs Vienna 4x22 (present as `.npz` in the TheGlueNote repo, see §3.6), (n)ASAP,
and Batik. `TheGlueNoteMatcher` prints DTW timing to stdout — silence it in a harness.

### 1.10 Reusable vs rebuild

**Reuse as-is:** the alignment dict representation; `fscore_alignments` +
`evaluate_score_following`; the whole `parangonar/dp/` library; `node_array()` for
semi-automatic anchors; partitura for all file I/O; the plotting functions;
`RepeatIdentifier` for repeat structure.

**Steal the idea, rebuild the code:** the dual forward/backward agreement filter; the
"confident-anchors-only monotone map, then per-pitch assignment" two-phase structure;
the rarest-pitch-first map refinement.

**Rebuild:** anything that touches features. The matchers are pitch+onset only, the ornament
path is bolted on and label-inconsistent, and the combinatorial omission search in
`SequenceAugmentedGreedyMatcher` is exponential-with-a-cap (and **non-deterministic** above the
cap — it random-samples).

---

## 2. `parangonar` (C++/WASM port) — take the build, not the algorithm

Copilot-generated C++17 port, 3,011 LOC vs 8,710 Python. README is explicit: *"This port only
implements the AutomaticNoteMatcher algorithm"*, *"created for the most part using GitHub's
Copilot Agent Mode."* `README_old.md` is a doctored copy of the upstream README and still
advertises a dozen algorithms that do not exist here — ignore it.

**What is ported.** `Note` struct (no `is_grace`, `voice` present but never read), `Alignment`
{MATCH, INSERTION, DELETION} — no ornament label. Three matchers: `SimplestGreedyMatcher`,
`SequenceAugmentedGreedyMatcher`, `AutomaticNoteMatcher`. Two DTWs (vanilla — the only one used;
weighted — dead code). Euclidean + cosine metrics. A **read-only** parser for v5.0 match files
(no writer, no ornament/trill/`meta` lines, no tempo map).

**Not ported:** `DualDTWNoteMatcher`, `TheGlueNoteMatcher`, `AnchorPointNoteMatcher`,
`CleanOrnamentMatcher`, all online matchers, the mismatch module, NWTW/FlexDTW/OLTW, grace and
ornament handling.

**The algorithm does not work.** A subagent re-implemented the C++ pipeline faithfully in
Python and traced it: `note_array::compute_pianoroll` returns `[time][pitch]` whereas
partitura's returns `[pitch][time]`, but the port copies partitura's `.T` verbatim — so DTW
walks the *pitch* axis with feature vectors of mismatched length, and
`euclidean_distance` returns `+inf` for every cell (it returns infinity on length mismatch).
The coarse alignment for a 47-beat piece yields 41 anchors spanning 0–2.5 "beats", i.e. exactly
`pitch_index/16`. `SCORE_FINE_NODE_LENGTH` is accepted and never read. The mender was later
given a global `SimplestGreedyMatcher` fallback, and *that* fallback alone reproduces the
0.9633 F-score in the commit message `0e5b694 "achieve target F-score of 0.963"` to four
decimals. The test suite's only accuracy assertion is `f_score > 0.5` on a synthetic scale; the
real-data test prints a warning below 0.8 and asserts nothing.

**What is worth taking:** the emscripten/embind setup. `CMakeLists.txt:97-111` plus
`cpp/src/wasm_bindings.cpp` is a clean, working template — `MODULARIZE=1`,
`EXPORT_NAME='ParangonarModule'`, `ALLOW_MEMORY_GROWTH=1`, `register_vector<T>` for passing
struct arrays across the JS boundary, with `example.html` demonstrating handle lifetime. The
JS surface is `Module.align(NoteArray, NoteArray, Config) → AlignmentVector` and
`Module.match(NoteArray, NoteArray)`. A native (non-WASM) build also works —
`parangonar_cpp` is a dependency-free static lib. Three inconsistent copies of the wasm exist
(committed / working tree / `build_emscripten/`); the root one shipped for `example.html`
predates the mender rewrite.

The vendored `original/parangonar/` is **not** a duplicate of `original-parangonar` — it is a
dirty working copy of the same commit carrying Niels' own **`SubsequenceMatcher`** work
(1,212 LOC, TheGlueNote-based fragment location inside a longer reference) plus three scripts:
`find_subsequence.py` (visualization), `implant.py` (a FastAPI service that splices a performed
phrase into a reference), `parrot.py` (live MIDI phrase matcher/playback). Relevant if MLign
ever needs "where in the score is this fragment".

---

## 3. `thegluenote-main` — TheGlueNote (Peter & Widmer, ISMIR 2024)

The premise that `src/` is empty is wrong — this project puts all module code *inside*
`__init__.py`. 2,174 LOC total. `src.zip` is a redundant identical backup.

### 3.1 Architecture

**Encoder-only, single-stream "concatenate-and-self-attend"** — not Siamese, no cross-attention
module. Both note sequences become one token stream and ordinary self-attention does double
duty as intra- and cross-sequence attention.

`src/models/__init__.py:11-97`. Per note, **4 miditok tokens are summed** (not concatenated)
into one note vector, so 512+512 notes → **1026 transformer positions**. A learned absolute
positional table `nn.Embedding(1026, D)` is the *only* signal telling the model which half a
token belongs to — no segment embedding, no CLS/SEP.

| | small ckpt | mid ckpt | large (config only) |
|---|---|---|---|
| dim_model | 128 | 256 | 512 |
| layers | 4 | 6 | 8 |
| heads | 8 | 8 | 8 |
| notes/window | 512 (+1 null) | 512 | 512 |
| encoder params | 1,113,344 | 5,673,984 | ≈28 M |
| + `GlueHead` | 660,353 | 2,106,625 | — |

Pre-LN, GELU. Similarity = **dot product of the two halves after one shared linear projection**
`embed_out`, giving a 513×513 logit matrix, then

```python
confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)   # dual softmax
```

the SuperGlue/LoFTR mutual-nearest-neighbour score. **Index 0 of each axis is a reserved
"unmatched" dustbin** — insertion/deletion is a first-class prediction, not a threshold. That
is the single most important design idea here.

`GlueHead` is a second 2-layer transformer that consumes the confidence matrix and emits
per-note class logits — a learned replacement for DTW post-processing. It only tiles the
diagonal at a hardcoded stride of 512, so it cannot recover from structural offset; parangonar
ships only the DTW path.

**Cross-check with parangonar's copy**: `pretrained_models.py:106` is a verbatim copy with
defaults baked to the small model and the dual softmax folded into `forward()`. I verified all
62 tensors of `parangonar/assets/thegluenote_small_checkpoint.pt` are **bit-identical** to
`gluenote.*` in `data/checkpoints/release/small.ckpt`.

### 3.2 Tokenization

`miditok.Structured()` with **library defaults** (the `TOKENIZER_PARAMS` block at
`datasets/__init__.py:9-21` is dead code — never passed to a `TokenizerConfig`).
**4 tokens/note: TimeShift → Pitch → Velocity → Duration.** Vocab = 314, of which 61
(`PitchDrum_*`) are never emitted. TimeShift is the **inter-onset interval to the previous
note**, so time is delta-encoded and shift-invariant.

At inference (`eval/__init__.py:580-619`) both sequences are made
**velocity- and duration-agnostic** — every note becomes `Velocity_63`, duration 100 ticks —
and the score is linearly stretched onto the performance's total span. So **two of the four
tokens per note carry zero information at test time**; the model effectively runs on
(pitch, IOI). This is 2× wasted sequence length and a clear thing to redesign.

### 3.3 Training — the part worth stealing

**Fully self-supervised. No aligned corpus is used or needed.** `process_file`
(`datasets/__init__.py:90-161`) loads one MIDI, copies it, and corrupts the copy with
`reorder()`; the ground-truth matrix is the bookkeeping index array that `reorder` returns for
free. Row/col 0 is set wherever a note has no partner inside the crop window — that is how
insertion/deletion is taught. 50% random swap of which side is s1 enforces symmetry.

The **13 augmentations** in `reorder()` (`datasets/__init__.py:229-412`):

1. global pitch transposition (both sides — an invariance prior)
2. random note deletions, `p=0.2` of notes
3. **segment skip** — a passage cut and onsets spliced, `randint(8,200)` notes, **fires on every sample**
4. random note insertions, `p=0.2`, random pitch/vel/dur from the piece's own range
5. **trill/ornament insertion** — `randint(20,100)` alternating notes, ±1–3 semitones, **unconditional**
6. **segment repeat** — a passage duplicated in place, tagged as *not* a valid match, **fires on every sample**
7. global tempo scaling, `clip(N(1, 0.5), 0.2, 5.0)` — up to 5× either way
8. **per-IOI tempo curve** — `2^N(0,0.5)` on every inter-onset interval independently
9. onset jitter ±50 ticks, then re-sorted (so also produces intra-chord order swaps)
10. duration noise ±250 ticks
11. velocity noise ±10
12. random side swap, p=0.5
13. random 512-note crop

Loss: `CrossEntropyLoss` applied **bidirectionally** over the raw logits (each s2 note is a
513-way classification over s1 indices, and vice versa), plus a third term on the head output.
Unweighted sum. No contrastive/InfoNCE, no Sinkhorn. Adam, `betas=(0.9,0.98)`, `eps=1e-9`,
`CosineAnnealingWarmRestarts(T_0=2000)` per step, no warmup, no clipping, no weight decay.
lr 5e-4, batch 24 (small) / 16 (mid). Actually trained to epoch 2004 / step 82,205 (small).

Data: `data/nasap/` = **1,145 flattened single-track 480-ppq MIDIs** (157 quantized scores +
988 performances), 3.99 M notes, median 3,069 notes/file. **No ground-truth alignment
anywhere in the training data** — the nASAP score/performance pairing is not exploited at all.
`check_files` drops files with <1024 notes, leaving 1,034 usable.

### 3.4 Inference

Windowing is a **full 2-D cross product**: every 512-note score window against *every* 512-note
performance window, stride 256, `⌈N₁/256⌉ × ⌈N₂/256⌉` forward passes, unnormalized overlap-add
(interior cells get up to 4× the mass of border cells). This is the scaling wall — two
20,000-note pieces would need ~6,100 forwards and a 20,000² matrix before DTW starts.

Post-processing (`"dtw"` mode, the one parangonar ships): invert the confidence matrix → WDTW
with `directional_weights=[1,2,1]` → trim degenerate endpoints → onset anchor pairs (merging
within 5 ticks by median) → `get_note_matches_with_updating_map` (rarest-pitch-first, map
rebuilt after each pitch). The `"matrix"` mode is pure greedy argmax with no monotonicity or
one-to-one constraint.

### 3.5 Running it

Blockers, all small: `configs/__init__.py:20` has a **SyntaxError** (missing comma) so nothing
imports; `train.py` hardcodes a nonexistent checkpoint path and the from-scratch branch is
commented out; `torch==2.2.1+cu118` doesn't exist for macOS; **`numpy<2` is a hard pin** (miditok
3.0.3 breaks on numpy 2.x at `structured.py:72`); `parangonar==1.1.0` is a hard pin (three
imports moved in 3.x). Inference is comfortably CPU-feasible — a subagent measured **75 ms per
512+512 window (small), 171 ms (mid)** on this machine. The practical path is what parangonar
does: skip Lightning, `torch.load`, strip the `gluenote.` prefix.

### 3.6 Data assets

- `data/checkpoints/release/small.ckpt` 18.9 MB, `mid.ckpt` 84.2 MB (both ~11× their weights
  because they carry Adam state). Large model is a download link only.
- **`data/testing/4x22/` — 88 `.npz` = the full Vienna 4x22 benchmark with ground truth**
  (4 pieces × 22 pianists, 464 MB). Keys include `gt_alignment`, `score_note_array`,
  `performance_note_array`, `score_note_array_grace`, `score_note_array_ornament`,
  `onset_alignment_path` (+ reverse). **43,450 GT matches, 440 deletions, 184 insertions.**
  One file (`Chopin_op10_no3_p01.npz`) has an older schema.
  Also present but unused by the model: rich partitura `score_features` — voice, staff,
  downbeat, articulation, ornaments, dynamics, metrical strength. **Sitting right there,
  ignored.** That is MLign's opening.

### 3.7 Reusable vs rebuild

**Take:** the checkpoints (verified working, CPU); `reorder()` — ~180 self-contained numpy
lines that turn any raw MIDI corpus into supervised alignment data; the dual-softmax + dustbin
formulation (3 lines); `get_note_matches_with_updating_map`; the 4x22 test set; the
velocity/duration-agnostic + shift + stretch normalization.

**Rebuild:** the tokenizer (a per-note feature MLP over pitch + log-IOI + score features would
halve sequence length and drop the miditok/numpy pins); all training scaffolding; the data
loader (re-tokenizes whole files every epoch); the `GlueHead` path.

**Architectural limits to design around:** the 512-note window is a **structural ceiling**, not
a hyperparameter — learned absolute positions mean you cannot go longer without retraining
(RoPE/ALiBi would fix it); quadratic tiling with no band restriction or coarse-to-fine pruning;
**no score features at all**; monotonicity and one-to-one are delegated entirely to
post-processing; solo-piano-only training data.

---

## 4. `AlignmentTool` — Nakamura's HMM, via a WASM fork

**Not** the upstream `eita-nakamura/AlignmentTool` shell toolchain. It is
`pfefferniels/AlignmentTool` (75 commits, last 2024-02-19), MIT, dual copyright *Eita Nakamura
2019 / Niels Pfeffer 2024*, which vendored Nakamura's C++ at commit `293814b` and then stripped
it to the score-following + error-detection + realignment core, exposing **two functions** to JS
via embind. All CLI binaries, all file I/O, all MusicXML/MIDI parsing and all shell pipelines
were deleted. There is no `MIDIToMIDIAlign.sh`.

Deleted upstream binaries: `midi2pianoroll`, `SprToFmt3x`, `MusicXMLToFmt3x`, `MusicXMLToHMM`,
`Fmt3xToHmm`, `RealignmentMOHMM`, **`MatchToCorresp`**. `ErrorDetection` and `Realignment`
survive as library functions.

### 4.1 The JS API

```ts
align(midiNotes: MidiNote[], scoreNotes: NoteEvent[],
      secondsPerQuarterNote: number, ticksPerQuarterNote: number): MatchResult
alignMidiToMidi(midiNotes1, midiNotes2, secondsPerQuarterNote): MatchResult

MidiNote  = { onset, offset, id, pitch, channel }
NoteEvent = { scoreTime, staff, voice, suborder, type, duration,
              sitches: string[], notetypes: string[], ids: string[] }
Match     = { scoreId, midiId, matchStatus, errorIndex, skipIndex }
MatchResult = { matches: Match[], missingNotes: string[] }
```

`align()` runs the **full three-stage pipeline** — Fmt3x → HMM → Viterbi score-following →
`detectErrors` → `realign(widthSec=0.3)` — so the JS call does replicate the upstream shell
chain, minus file formats. `secondsPerQuarterNote` is the *initial tempo prior* for the Kalman
filter, not a fixed tempo. Note `lib/index.js:78` **discards `result.missingNotes`** for
`align()` (deletions are silently lost) and mutates the caller's objects.

### 4.2 Formats

**fmt3x** (`_fmt3x.txt`), tab-separated, `9 + 3·numNotes` fields:

```
stime  barnum  staff  voice  subvoice  subOrder  eventtype  dur  numNotes
       sitch_1…n   notetype_1…n   fmt1ID_1…n
```

- `subOrder`: after-notes are **negative** (−4, −3, …), short appoggiaturas 0, the main chord 1.
- `eventtype` ∈ `chord | rest | short-app | after-note | tremolo`.
- `sitches` are **spelled** (`C#5`, `Bb3`, `G##2`). **For an ornamented note it is a comma
  triple `principal,upper,lower`**, e.g. `C5,D5,B4`.
- `notetypes` = `<base>.<arpInfo>.<ferInfo>`, base ∈ `N` normal, `Tr` trill, `Mr` mordent,
  `Im` inverted mordent, `Tn` turn, `It` inverted turn, `Dt`/`DIt` delayed (inverted) turn;
  `Arp<n>` arpeggio, `Fer` fermata. So `N..` = plain note.
- **Ties** have no field — tied notes are merged into one fmt3x note whose `fmt1ID` is the
  comma-joined list of constituents.
- **Repeats are not represented.** fmt3x is flat and unexpanded; jumps are handled at decode
  time by the HMM's backward and large-skip transitions.

**match** (`_match.txt`), 12 tab-separated fields:

```
//Version: ScorePerfmMatch_v170503
ID  ontime  offtime  sitch  onvel  offvel  channel  matchStatus  stime  fmt1ID  errorInd  skipInd
...
//Missing <stime>	<fmt1ID>
```

- **Insertions** = a normal note line with `errorInd=3`, `fmt1ID="*"` (cluster-wise extra) or
  `errorInd=2`, `fmt1ID="&"` (note-wise extra — declared but never produced by this code),
  `stime=-1` after realignment.
- **Deletions** = the trailing `//Missing` lines, *not* note lines.
- **Substitutions** = `errorInd=1`: a matched pair with disagreeing pitches; `fmt1ID` names the
  *intended* score note, field 4 holds the *played* pitch. **Nakamura's format expresses
  something parangonar's cannot: a wrong note.**
- `skipInd`: `0` first note, `1` resumption after a skip, `-`/`+` otherwise — but this build
  only ever writes `0` and `-`.

**corresp** (`_corresp.txt`) — recovered from `git show 293814b:Code/MatchToCorresp_v170918.cpp`.
This is what `scorewarp`'s pipeline consumed:

```
// alignID alignOntime alignSitch alignPitch alignOnvel refID refOntime refSitch refPitch refOnvel
```

Ten tab-separated fields, times at 6 decimals. Unmatched on either side is written as
`*  -1  *  -1  -1`. `refID` is the fmt1ID's trailing segment after the last `-`. Missing notes
are appended after the matched ones.

### 4.3 The model

**HMM topology** (`src/Hmm.hpp`, `src/ScoreFollower.hpp`, 846 LOC). States are score positions,
two-level: `TopId` = chord group, `BotId` = internal sub-position (chord / arpeggio /
appoggiatura / trill step). Self-transition `p = (d−1+0.1)/(d+0.1)` where `d` = notes in the
state — the standard "a d-note chord emits d observations" construction.

Top-level transition distribution over Δ ∈ [−3, +2], hardcoded:
`{−2: 0.00516, −1: 0.00886, 0: 0.01342, +1: 0.94531, +2: 0.00610, +3: 0.00073}`,
**plus a global "large skip" from the current MAP state to any state at logprob −40**. That,
with the backward transitions, is what handles **repeats, da capo, restarts and arbitrary
jumps** — since fmt3x never expands repeats.

Observation model:
- **pitch**: mixture around the score pitch — 0.95 exact, 0.015 at ±1 semitone, 0.022 at ±2,
  0.0047 at ±12, 0.0083/9 spread over the rest.
- **IOI**: four structural densities — chordal (exponential, σ=0.01 s), arpeggio (Gaussian
  μ=0.05, σ=0.05), appoggiatura/inter-cluster (μ=0.13, σ=0.07), trill (μ=0.082, σ=0.015) —
  each mixed 0.95/0.05 with a Cauchy insertion density at 0.5 s. Chord-to-chord transitions use
  a **Cauchy** (half-width 0.3 s) centred at the tempo-predicted gap minus `stolenTime =
  0.13 s × max(numArp, numInterCluster)`. Heavy tails = robustness to gross timing errors.

**Tempo**: a **switching Kalman filter** (`src/SwitchingKalmanFilter.hpp`) over seconds-per-tick
with two observation-noise regimes, σ=0.014 (normal) and σ=0.16 (outlier/expressive), so a
single hesitation is absorbed without corrupting the estimate. Updated only on clean forward
chord-to-chord steps with IOI > 35 ms.

**Realignment = MOHMM = merged-output HMM** (`src/Realignment.hpp`, 1,699 LOC). Piano
performance is modelled as the *merge* of two quasi-independent hand streams. Error regions are
identified (pitch errors, extras, reordered notes, missing notes), each widened by ±0.3 s, and a
region is re-decoded **only if it contains at least two different error kinds**. Re-decoding:
render the score region as a synthetic piano roll → HMM hand separation → one part-HMM per hand
→ Viterbi over the merged state space
`i = i_s + 2(i_h + Nh(iR + N_R·iL))` where `i_s` = emitting hand and `i_h` = notes since that
hand last played → per-state error detection. **Hand separation** (`HandSeparationForPR.hpp`)
uses Nakamura's trained interval log-prob tables plus a hand-crossing penalty and a
"stretch beyond a tenth" prior.

**Input features**: pitch (spelled and MIDI), onset/offset seconds, score time in ticks, and —
uniquely among the systems here — **event type (chord/grace/after-note), ornament type,
staff and voice**. Velocity is carried but not used in the likelihoods.

### 4.4 Critical caveat: the ornament machinery is dead in this build

`Hmm::ConvertFromFmt3x` (`src/Hmm.hpp:59-105`) is upstream's **trivial** converter: one `CH`
state per fmt3x event, `numClusters=1`, `numArp=0`, all voices 0. Upstream's *rich* converter
(`ConvertFromHom`, which actually produces the SA/AN/TR states) is fed by the
`Fmt1x → Fmt2 → Hom` chain — **and those three headers were deleted from this fork**.
Therefore, in the WASM `align()`:

- every state is `CH`; `notetypes` (`Tr..`, `Mr..`) and `suborder` are accepted at the API
  boundary and **silently ignored**;
- `staff` and `voice` are stored and then **discarded**;
- the trill-overlap block, the TR pitch branch, `stolenTime`, the arpeggio/appoggiatura IOI
  weights and the TR realignment expansion are all **dead code**.

So this build is a chord-HMM with skip transitions and a Kalman tempo model — the ornament and
hand-structure sophistication that makes Nakamura's tool interesting is unreachable. Two further
regressions: `SwitchingKalmanFilter.hpp:27` declares `const int M_ = pow(0.2/tickPerSec_, 2.)`
where upstream had `double` — it truncates to **0**, zeroing the initial tempo variance; and
`src/PianoRoll.hpp` was mid-refactor into `using PianoRoll = std::vector<PianoRollEvt>` with a
**missing semicolon**, so the working tree does not compile while every consumer still uses the
old `pr.evts[...]` API.

### 4.5 Running it as a baseline

**Present and usable right now:** `lib/Matcher.wasm` (324 KB, 2024-02-19), `lib/Matcher.js`
(111 KB), `lib/index.js`, `lib/Matcher.d.ts`. Node v23.8.0 is installed. So
`node -e "import('./lib/index.esm.mjs')..."` with hand-built note arrays works today — that is
the fastest path to a baseline number, and it is what `example/example.html` demonstrates
(`await align(midiNotes, scoreNotes, 0.1, 4)`).

**Rebuilding is harder:** needs **emsdk** (`emcc` is *not* on PATH), plus `external/gcem`, which
is **untracked and has no `.gitmodules`** — a fresh clone cannot build. `node_modules/` contains
only `@types/emscripten`. And HEAD compiles but the working tree does not.

**Recommendation for MLign:** to get a *fair* Nakamura baseline, do **not** use this fork. Clone
upstream `eita-nakamura/AlignmentTool`, build the real binaries with clang++ (pure C++, no
dependencies), and use `MIDIToMIDIAlign.sh` / `MusicXMLToMIDIAlign.sh`. Use this fork only if you
want an in-browser aligner. Either way you need a fmt3x writer from MEI — and note that
`Fmt3x::ConvertFromPianoRoll` has an inherited upstream bug where `duplicateOnsets` is
constructed but never `push_back`ed, so it is always empty.

---

## 5. `scorewarp` — Goebl & Weigl, mdw Vienna

Two-line README: *"Horizontal warping of Verovio SVGs to match performance timeline."*
Vanilla JS, no build step. Demo version 13 April 2025.

**Algorithm** (`scoreWarper.js`): render MEI with Verovio at `breaks:'none'` into one long
system; for each aligned note collect where it *is* engraved (`.notehead use` bbox x) and where
it *should* be (`time2svg(obs_mean_onset)`, a global linear time→x map); build a **per-pixel
displacement field** `warpArr[x] = Δx` by piecewise-linear interpolation between note anchors;
then translate point-like elements and translate+x-scale extended ones (beams, hairpins, slurs,
barlines). Geometry is preserved — nothing is re-rastered. `warpIndividualNotes()` optionally
breaks chords apart, though all chord members land on the same x since MAPS has no per-note
onsets.

**MAPS JSON schema** (`eval/`, 87 files):

```json
{ "obs_mean_onset": 129.788,
  "xml_id":   ["note-0000000238705390", "note-0000000198440388", "note-0000000361809620"],
  "velocity": [29, 31, 30],
  "confidence": 0, "obs_num": 291 }
```

- `obs_mean_onset` — performed onset in seconds, **mean over the chord**. `-1` = the score
  notes in this entry were never performed (**deletion**).
- `xml_id` — MEI `xml:id`s. `trompa-align_inserted_<Pitch><Oct>` = a performed note with no
  score counterpart (**insertion**).
- `velocity` — index-parallel to `xml_id`; the only per-performed-note datum. There is **no
  performed-note identifier** at all — a performed note is identified only by (obs_num, array
  position).
- `duo*` files add `beat` and `qstamp` (score onset in quarters).

Naming: `Op53_2_P08-A.boe.mid.maps.json` = Beethoven op. 53 mvt 2, pianist P08, take A,
Bösendorfer MIDI grand. `.expansion-default.json` = ids from the repeat-expanded rendering
(duplicates carry `-rend2`). `D2-Beethoven-Op31-2-2.boe_corresp.txt.maps.json` is the smoking
gun that the upstream pipeline was **Nakamura corresp → maps.json** via David Weigl's
`trompa-align`.

**For MLign:** 87 files of real MEI-id ↔ performance-time alignment on Beethoven sonatas and
the Togetherness duo corpus — usable as evaluation data *if* the corresponding MEIs are fetched
(they are pulled from GitHub at runtime, not vendored). The warping-field rendering approach is
the right way to display an alignment. The format itself is weaker than what we need: no
performed-note ids, chord-mean onsets only, no durations.

---

## 6. `aligned-mei` — "As Played By" / Alignment Desk

`github.com/pfefferniels/as-played-by`, deployed at **`https://align.encoded-ghosts.org`**
(the CNAME survives only on the `gh-pages` branch — the one in `main` is 0 bytes, so the next
`npm run deploy` would drop the custom domain). Vite 7 + React 19 + TS, Verovio 6, midifile-ts,
CodeMirror for in-app MEI editing.

### 6.1 The MEI alignment representation — the most directly adoptable thing in any of these repos

It stays inside **MEI 5.1's performance module** rather than inventing a sidecar format.
Written by `src/When.ts`, read by `src/parseRecordings.ts`. From
`/Users/nielspfeffer/Projects/aligned-mei/public/transcription.mei`:

```xml
<performance>
   <recording source="c9050e75-97a8-4862-9533-0f4b1439802b">
      <when absolute="28618ms" abstype="smil"
            corresp="symbol_acfc087a-e169-4fac-be23-b2d2ea72cefa" data="#npk4lw6">
         <extData type="velocity">41</extData>
         <extData type="duration">979ms</extData>
         <extData type="onsetTicks">28618</extData>
         <extData type="durationTicks">979</extData>
      </when>
      ...
   </recording>
</performance>
```

- `@data="#npk4lw6"` → the **score** side, an MEI `<note xml:id>`.
- `@corresp="symbol_..."` → the **performance** side; taken from a MIDI text meta-event
  preceding the note-on, else a synthetic `${track}-${tick}-note-${ch}-${pitch}`.
- `@absolute` + `@abstype="smil"` → performed onset.
- Pedals use the same container with `@type="sustain"` and two ids in `@corresp` (down and up).
- `<recording @source>` → a `<manifestation xml:id>` in the header carrying capture provenance
  (`<physDesc><captureMode>`), so **multiple recordings of one score coexist**.
- Score notes *also* carry `@corresp` to the same performance symbol id — a three-way id
  triangle.

`public/transcription.mei` is a fully worked example: 1,033 `<when>` (926 notes + 107 pedals),
8,050 `<extData>`, Schumann *Träumerei* from a Welte roll. `test/traumerei.mei` is the plain
unaligned score.

It **round-trips**: `parseRecordings.ts` → `buildMidiFile.ts` (tempo 60 BPM, `ticksPerBeat:1000`
so 1 tick = 1 ms) regenerates playable MIDI with MEI ids embedded as text events for
score-following highlight. A representation you can play back is a representation that has been
debugged.

### 6.2 Gaps in that representation

- **"MEI customization" is aspirational** — there is no ODD, no RNG, no schema anywhere.
- **`<extData>` is an escape hatch**: velocity, duration and ticks all live outside MEI's own
  vocabulary, with units baked into strings (`"979ms"`, parsed by `parseInt` stopping at `m`).
- **Pitch is not stored** — it is recomputed from the score note, so you cannot record "the
  pianist played a different pitch here" without editing the score. (`CreateReading.tsx` gestures
  at this with `<rdg source="performance">` but it is half-wired.)
- **No insertions, deletions, or confidence.** The current `Match` type is just
  `{score_id, performance_id}`; the earlier parangonar-backed code had
  `label: 'match'|'insertion'|'deletion'` and that was lost. Unmatched performed notes live only
  in React state. **This is the single biggest gap for MLign.**
- `@source` is a bare ID, not `#id` — likely schema-invalid.
- No `<avFile>` pointing at the media.

### 6.3 The aligner and the parangonar remnant

`src/NaiveAligner.ts` is a greedy sequential exact-pitch chord matcher with **hard bail-out** —
group score notes by symbolic onset, consume the performance queue in time order, and `return`
at the first mismatch. So alignment always covers only a prefix, and the workflow is
operator-in-the-loop: fix the MEI, re-align, repeat.

Score extraction (`getNotesFromMEI`) goes through Verovio's `renderToTimemap()`: onsets in
**quarter notes** (`qstamp`), pitch from `getMIDIValuesForElement`, tie-ends dropped so a tied
chain is one event, duplicate `(onset,pitch)` removed, and `appXPathQuery:
["./rdg[contains(@source,'performance')]"]` selects the performed reading of any `<app>`.

**`public/parangonar.wasm` (228 KB) is an orphan** — `grep -rn parangonar src/` returns nothing.
Commit `ce9e018` (knip cleanup) deleted `src/loadParangonar.ts` and `public/parangonar.js` but
left the `.wasm`. The historical API is recoverable via
`git show ce9e018^:src/loadParangonar.ts` and used `config.alignment_type = "greedy"`,
`sfuzziness=0.1`, `pfuzziness=0.5`. `package-lock.json` also still references a local
`"../AlignmentTool"` package, so a Nakamura binding was wired up at some point too.

`src/Aligner.tsx` (690 lines) is the ScoreWarp counterpart: absolute per-note placement
(`newX = span.onsetMs * stretchX`) followed by ~500 lines repairing what that breaks —
`multiplyStems`, `multiplyLedgerLines`, `redoTies` (recomputed Béziers), `redoBeams`,
`redoBarLines`, `redoAnchored`, plus velocity → opacity mapping. Unmatched notes are tinted
`darkred`. ScoreWarp's displacement-field approach is the better technique; aligned-mei's
representation is the better data model.

---

## 7. Direct answers to the framing questions

**How does parangonar represent an alignment?** A flat list of dicts with `label` ∈
{`match`, `deletion`, `insertion`, `ornament`}, carrying partitura note IDs (= MEI `xml:id` for
MEI input). Matches are strictly 1:1. `ornament` is readable and writable via partitura but is
**never emitted by any matcher** — normalize with `convert_grace_to_insertions()` before
scoring. No confidence values anywhere.

**What input features does each system use?**

| System | Features |
|---|---|
| parangonar `AutomaticNoteMatcher` | pitch, onset, duration (piano-roll raster) |
| parangonar `DualDTWNoteMatcher` | pitch, onset, `is_grace`; duration only for the ornament window |
| TheGlueNote | pitch + IOI (velocity and duration are constanted out at inference) |
| Nakamura (upstream) | spelled pitch, onset/offset sec, score ticks, event type, **ornament type, staff, voice** |
| Nakamura (this WASM fork) | spelled pitch, onsets, score ticks — ornament type/staff/voice accepted then discarded |
| aligned-mei `NaiveAligner` | pitch + symbolic onset, exact match only |

**Nobody uses voice, staff, metric position, beaming, ties, slurs, dynamics or articulation.**
partitura extracts all of it; the Vienna 4x22 `.npz` files literally ship it alongside the
alignments; every system throws it away.

**Can we run parangonar and AlignmentTool locally as baselines?**
- parangonar: **yes, verified.** Nothing to install — parangonar 3.1.0, partitura 1.6.0,
  torch 2.11.0 are already in the system Python 3.13. All three offline matchers scored F=1.0
  on the bundled Mozart file in under 4 s each.
- AlignmentTool: **partially.** The prebuilt `lib/Matcher.wasm` + node v23.8.0 give a working
  `align()` today, but that build has the ornament/voice/staff machinery dead and two numeric
  regressions. For an honest Nakamura baseline, build upstream from source with clang++ (no
  dependencies) — but budget time for an MEI → fmt3x writer.
- TheGlueNote standalone: needs a one-character syntax fix, a numpy<2 environment, and
  retargeted imports. Easier to call it through parangonar, which is what `TheGlueNoteMatcher`
  does with a bit-identical copy of the small checkpoint.

**What's reusable vs what must be rebuilt?**

*Reuse directly:* parangonar's alignment dict format and evaluation functions; the entire
`parangonar/dp/` DP library; partitura for all I/O; TheGlueNote's `reorder()` augmentation
engine and its two checkpoints; the Vienna 4x22 `.npz` benchmark; aligned-mei's
`<performance>/<recording>/<when>` MEI markup plus `parseRecordings.ts` and `buildMidiFile.ts`;
the C++ port's emscripten/embind CMake template if we ever want browser deployment.

*Steal the design, rebuild the code:* the two-phase structure (confident monotone timing map →
per-pitch note assignment) that all three serious systems converge on; the dual forward/backward
agreement filter; dual-softmax with an explicit dustbin row/column; the rarest-pitch-first
iterative map refinement; Nakamura's heavy-tailed (Cauchy) timing likelihoods and his
switching-Kalman tempo model with an outlier regime.

*Rebuild from scratch:* the feature front-end (this is where the win is — MEI's voice, staff,
metric position, ties, ornament type, grace status, `<app>` variants); a positional scheme that
is not a hard 512-note ceiling; windowing that is banded/coarse-to-fine rather than a full 2-D
cross product; an alignment representation with confidence and with wrong-note (substitution)
support, which only Nakamura's format currently expresses.

---

## 8. Concrete implications for MLign

1. **Adopt parangonar's alignment dicts as the internal representation, extended.** Add
   `confidence: float` and a `substitution` label (score note played at the wrong pitch —
   Nakamura's `errorInd=1`, which parangonar cannot express). Keep partitura note IDs so MEI
   `xml:id`s flow through unchanged.
2. **Serialize to aligned-mei's `<when>` markup**, but fix its three gaps: emit deletions and
   insertions, carry confidence, and write an actual ODD. That gives us round-trippable,
   playable, standards-shaped output and reuses `parseRecordings.ts` / `buildMidiFile.ts` for
   verification.
3. **Bootstrap training data with TheGlueNote's `reorder()`** — self-supervised corruption of
   raw MIDI needs no aligned corpus and already covers the right failure modes (skips, repeats,
   trills, insertions, deletions, global + local tempo, jitter). Then layer the espressivo-rendered
   MEI+MPM synthetic pipeline on top, which gives something `reorder()` cannot: *musically
   plausible* rubato and ornament realization with exact ground truth.
4. **Feed the model what everyone else discards.** Voice, staff, metric position, tie/grace
   status, ornament type, beam grouping. The Vienna 4x22 `.npz` files already carry these as
   `score_features`, so the ablation is cheap to run.
5. **Evaluation harness:** parangonar's `fscore_alignments` on match/insertion/deletion plus
   `evaluate_score_following`'s asynchrony percentiles, against `DualDTWNoteMatcher` (the real
   SOTA), `TheGlueNoteMatcher`, and upstream Nakamura. The Vienna 4x22 `.npz` set is the
   ready-made first benchmark: 88 pairs, 43,450 GT matches, already on disk at
   `/Users/nielspfeffer/Projects/thegluenote-main/data/testing/4x22/`.
6. **Normalize ornament labels before scoring anything.** Otherwise a perfect predictor loses
   points on every file that contains a trill.
7. **Ignore the C++ parangonar port's algorithm** (its headline F-score is its greedy fallback),
   and don't use the AlignmentTool fork for baseline *numbers* — its ornament and hand-structure
   modelling is unreachable dead code.
