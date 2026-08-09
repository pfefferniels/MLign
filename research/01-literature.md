# 01 — Literature: SOTA numbers for symbolic score→performance note alignment

Compiled 2026-08-09 for MLign. Scope: published evaluation numbers we must beat, newer
work (2024–2026), directly applicable techniques, and the benchmark datasets.

**Verification convention used throughout:**

- `[PDF]` — I read the actual paper PDF and transcribed the table myself. Highest confidence.
- `[HTML]` — extracted from publisher/arXiv HTML via automated fetch. High confidence, but
  column-merging errors are possible (one such error was caught and corrected, see §2.3).
- `[2nd]` — relayed from a secondary source or a research subagent, not verified against the
  primary PDF. Treat as a lead, not a fact.

---

## 0. Executive summary — the numbers to beat

Three separate bars, on three separate protocols. They are **not** interchangeable.

| Bar | Benchmark | Metric | Best published | Source |
|---|---|---|---|---|
| **A. Clean symbolic alignment** | Vienna 4x22 (88 perfs) | match F, per-perf avg | **99.8 ± 0.4** DualDTW | ISMIR 2023 T1 `[PDF]` |
| | Magaloff (~150 pieces) | " | **98.4 ± 0.9** DualDTW | " |
| | Zeilinger (29 perfs) | " | **99.3 ± 0.9** DualDTW | " |
| | Batik (36 movements) | " | **99.4 ± 0.7** DualDTW | " |
| | Combined | " | **99.0 ± 1.0** DualDTW | " |
| **B. Hard/mismatched symbolic** | 5 proprietary pieces, +20% mismatch | match F, mean of 5 | **95** TGN-small+DTW | GlueNote T4 `[PDF]` |
| **C. Repeat-structure robustness** | Vienna 4x22, folded scores | F_align (pooled) | **98.4** RUMAA | RUMAA T2 `[HTML]` |

**The strategic read.** Bar A is saturated — DualDTW is at 99.0 combined, and the residual
1% is substantially dataset-annotation noise rather than model error. Chasing it is a poor
use of effort and is nearly unmeasurable given that three of the four datasets are
proprietary. **Bars B and C are where the headroom is.** TheGlueNote scores **12.7** on
repeat-containing scores where RUMAA holds 98.4 — a in the field. But note the
protocol asymmetry that makes this comparison less damning than it looks (§3.1): RUMAA
takes audio + MusicXML *with repeat symbols* and is *designed* to emit repeat/skip tokens,
while GlueNote is handed a folded score it has no mechanism to unfold. The honest framing
is that GlueNote never attempted the task, not that it attempted and failed.

**Recommended targets for MLign:**
1. Match DualDTW on Bar A within noise (≥99.0 combined) — table stakes, not a contribution.
2. Beat TheGlueNote on Bar B by a clear margin (≥97 mean, vs. 95).
3. Handle repeats natively so that Bar C is ≥98 *from symbolic input alone*, which no
   published symbolic-only system currently does.

---

## 1. The parangonar line: exact published numbers

`parangonar` (https://github.com/sildater/parangonar) is the reference implementation.
Its README designates **`DualDTWNoteMatcher`** as "Default and SOTA for standard score to
performance matching" `[HTML]`. Matcher classes exposed: offline —
`AutomaticNoteMatcher`, `DualDTWNoteMatcher`, `TheGlueNoteMatcher`, `AnchorPointNoteMatcher`;
online — `OnlineTransformerMatcher`, `OnlinePureTransformerMatcher`, `TOLTWMatcher`,
`OLTWMatcher`; mismatch handling — `RepeatIdentifier`, `SubPartMatcher`; audio —
`AudioToScoreMatcher`, `AudioToScoreMatcherLimited`.

### 1.1 Name mapping (important — the papers and the code use different names)

| Paper name | parangonar class |
|---|---|
| `hDTW+sym` (TISMIR 2023) | `AutomaticNoteMatcher` |
| `DTW Offline` (ISMIR 2023) | `DualDTWNoteMatcher` |
| `Linear` + beat anchors (TISMIR 2023) | `AnchorPointNoteMatcher` |
| `hNWTW+sym` (TISMIR 2023) | not exposed as a top-level class |

RUMAA's table uses the TISMIR naming (`hDTW+sym`), which is why it appears to omit
AutomaticNoteMatcher — it does not; it is the same thing.

### 1.2 AutomaticNoteMatcher — TISMIR 2023

> Silvan David Peter, Carlos Eduardo Cancino-Chacón, Francesco Foscarin, Andrew Philip
> McLeod, Florian Henkel, Emmanouil Karystinaios, Gerhard Widmer.
> **"Automatic Note-Level Score-to-Performance Alignments in the ASAP Dataset."**
> *TISMIR* 6(1), 2023, pp. 27–42. DOI [10.5334/tismir.149](https://doi.org/10.5334/tismir.149)
> · https://transactions.ismir.net/articles/10.5334/tismir.149

**Table 2 — fully automatic models, no anchor points** `[PDF]`
*Caption: "Dataset-wise averaged F-Scores of each model. \* Superscripts are not
statistically different from Nakamura's (α = 0.01)."*

| Method | 4×22 | Zeilinger | Magaloff |
|---|---|---|---|
| **hDTW+sym** (= AutomaticNoteMatcher) | **98.53 %** | **97.98 %\*** | **94.57 %\*** |
| hNWTW+sym | 97.38 % | 95.07 %\* | 90.91 % |
| Nakamura | 98.97 % | 97.61 %\* | 95.18 %\* |

Note that in the *fully automatic* setting AutomaticNoteMatcher does **not** beat Nakamura's
HMM on 4×22 or Magaloff; the differences are not statistically significant. Statistical
testing was Friedman test followed by Wilcoxon signed-rank with Bonferroni correction.

**Table 5 — anchor-point models (the paper's actual contribution)** `[PDF]`
*Caption: "Values with superscripts are statistically better (\*) or worse (†) than
Nakamura's automatic alignment (α = 0.01)."*

| Granularity | Method | 4×22 | Zeilinger | Magaloff |
|---|---|---|---|---|
| — | NAKAMURA | 98.97 | 97.61 | 95.18 |
| Beats | Greedy | 99.28 | 98.09 | 95.68 |
| Beats | **Linear** | **99.87\*** | **99.67\*** | **98.87\*** |
| Beats | DTW | 99.81\* | 99.48\* | 98.67\* |
| Beats | NWTW | **99.91\*** | 99.61\* | 98.78\* |
| Measures | Greedy | 97.59† | 96.01 | 90.33† |
| Measures | Linear | 99.28 | **99.30\*** | 97.82\* |
| Measures | DTW | 99.31\* | 98.88 | 97.66\* |
| Measures | NWTW | **99.63\*** | 99.25\* | 97.88\* |

**These anchor-point numbers are not a fair comparison target.** They consume
ground-truth beat or measure annotations as input. They are the right baseline only if
MLign is also given beat annotations.

**Table 4 — tuned hyperparameters (beat-wise anchors, 6-piece tuning set)** `[PDF]`

| Method | Hyperparameters | F-measure |
|---|---|---|
| Greedy | Window size: 3 | 95.43 % |
| Linear | Fuzziness: 0.95 | 98.71 % |
| DTW | Fuzziness: 0.65, L₄-norm | 98.74 % |
| NWTW | Fuzziness: 0.8, γ: 0.5, Cosine | 98.75 % |

Tuning pieces (excluded from Tables 2/5): Chopin Nocturnes Op. 9 Nos. 1 and 2, Etude
Op. 10 No. 11, Nocturne Op. 15 No. 2, Barcarolle Op. 60, and Beethoven Op. 53 mvt. 3.
**Five of these six are exactly TheGlueNote's test pieces** (§2.4) — a leakage concern
worth flagging when citing GlueNote's Table 4 as an independent comparison.

**Robustness** `[PDF]`: all anchor-point models are flat for tapping noise below ±100 ms
and degrade roughly linearly above that (Fig. 5, Magaloff). For context the paper cites
professional-musician synchronization error <20 ms and beat-annotation SD of 27–68 ms.

**Architecture in brief** `[PDF]`: piano rolls at 16 samples/beat (score) and 16 samples/s
(performance); coarse DTW → cut score into 4-beat segments → fine-grained DTW per segment →
onset time mapping → combinatorial symbolic note matching per pitch → segment mending with a
conflict-resolution graph. The symbolic matcher solves, per pitch and window,
`i* = argmin_{i∈I} Σ_k |S_k − P_k^i|` over all `|S|`-combinations of performance onsets —
`|P|C|S|`, which is the combinatorial bottleneck that forces the windowing. It **cannot
detect both insertions and deletions of the same pitch in the same window**, and implicitly
favors matches over insertions/deletions.

### 1.3 DualDTWNoteMatcher — ISMIR 2023

> Silvan David Peter. **"Online Symbolic Music Alignment with Offline Reinforcement
> Learning."** *ISMIR 2023*. arXiv:[2401.00466](https://arxiv.org/abs/2401.00466)

The paper is primarily about the *online* RL matcher; the offline DualDTW model is
introduced in §3 and is the one that became parangonar's default.

**Table 1 — the headline SOTA table** `[PDF]`
*Caption: "Dataset-wise averaged F-scores and standard deviations of each model."*

| Dataset | DTW Offline (= DualDTW) | Nakamura |
|---|---|---|
| Magaloff | **98.4 ± 0.9 %** | 97.8 ± 1.4 % |
| Zeilinger | **99.3 ± 0.9 %** | 98.8 ± 1.2 % |
| Batik | **99.4 ± 0.7 %** | 98.5 ± 2.1 % |
| Vienna 4x22 | **99.8 ± 0.4 %** | 99.5 ± 0.5 % |
| Combined | **99.0 ± 1.0 %** | 98.5 ± 1.5 % |

Significance `[PDF]`: two-sided sign test on performance-wise rankings, significantly
(α = 0.01) higher for DualDTW on **all datasets except Vienna 4x22**. On Vienna 4x22 both
models reach F = 1.0 on 38 of 88 performances; DualDTW is higher on the remaining 50.

> ⚠️ **Discrepancy worth knowing.** Nakamura's Magaloff score is **95.18** in TISMIR 2023
> but **97.8 ± 1.4** in ISMIR 2023. The ISMIR paper states Nakamura's output was
> "post-processed to produce the same output format as ours," which is the likely
> explanation. Do not mix numbers across the two papers.

**Table 3 — piece-wise F-scores on the five hard pieces** `[PDF]`
(Same five pieces GlueNote uses. `OAM` = the online alignment model.)

| Piece | OAM | DTW Offline | Nakamura |
|---|---|---|---|
| B. Op. 53 3rd m. | 99.0 % | 99.4 % | 98.2 % |
| C. Op. 9 No. 1 | 97.6 % | 98.4 % | 98.8 % |
| C. Op. 9 No. 2 | 97.4 % | 99.1 % | 97.6 % |
| C. Op. 10 No. 11 | 90.3 % | 96.3 % | 94.3 % |
| C. Op. 60 | 95.1 % | 97.9 % | 94.7 % |

**Online results** (relevant only if MLign ever needs real-time) `[PDF]`:
Table 2 — greedy agent score-onset hit rate Top0 94.5 ± 0.8 %, Top1 96.6 ± 0.5 %,
Top2 97.6 ± 0.4 %. Table 4 — median asynchrony / % within 25/50/100 ms:
OLTW 60.6 ms / 38.0 / 63.3 / 86.7; GAM 36.0 ms / 89.0 / 91.4 / 94.6;
OAM 15.7 ms / 91.4 / 93.8 / 96.6.

**DualDTW architecture, precisely** `[PDF]` — worth reproducing because it is the baseline
we must beat and it is remarkably simple:

1. **Pitch sequence warping.** Performance notes are integers `p_t ∈ {1…88}`; score onsets
   are *sets* of pitches `s_t ∈ P(I)\{∅}`. Non-symmetric inclusion metric:
   `m(p_t, s_t) = 0 if p_t ∈ s_t else 1`. Two DTW paths are computed — forward and
   backward (on inverted sequences).
2. **Cleanup.** Wherever forward and backward paths disagree they "bracket" an ambiguous
   region; all bracketed notes are excluded from the path. Bracketed segments are then
   separated by pitch; if two pitch-wise subsequences with *matching note counts* are found
   they are aligned directly and added back, otherwise the path is linearly interpolated.
   This yields a score-time → performance-time mapping.
3. **Onset sequence warping.** Split both sides into pitch-wise sequences over the union of
   score and performance pitches. Project all score onsets (beats) into performance time via
   the step-2 mapping. Align onset sequences with a plain **L₁** metric; for non-unique
   alignments keep the lowest-distance tuple. A **maximum distance threshold of 5 seconds**
   suppresses spurious matches.

No learning anywhere. This is why it is fast and why it is brittle to structural mismatch.

### 1.4 TheGlueNote — ISMIR 2024

> Silvan David Peter, Gerhard Widmer. **"TheGlueNote: Learned Representations for Robust
> and Flexible Note Alignment."** *ISMIR 2024*, pp. 603–610.
> arXiv:[2408.04309](https://arxiv.org/abs/2408.04309) · code
> https://github.com/sildater/thegluenote

**Table 2 — Vienna 4x22, ablation of model size × match extractor** `[PDF]`

| Model | Sim Matrix P/R/F | Decoder Head P/R/F | DTW P/R/F | TL | VL | VA |
|---|---|---|---|---|---|---|
| Pitch-Onset Similarity Matrix | 7 / 7 / **7** | — | 85 / 89 / **82** | — | — | — |
| TGN-large | 97 / 97 / **97** | 96 / 97 / **96** | 99 / 100 / **99** | 0.183 | 0.126 | 0.958 |
| TGN-mid | 81 / 81 / **81** | 87 / 88 / **87** | 99 / 99 / **98** | 0.171 | 0.145 | 0.996 |
| TGN-small | 75 / 75 / **75** | 83 / 87 / **81** | 99 / 99 / **99** | 0.374 | 0.280 | 0.902 |

> ⚠️ An automated HTML scrape of this table **silently merged the baseline's DTW columns
> into its Sim Matrix columns**, producing a bogus "baseline F = 82 raw." The correct
> reading is F = 7 raw, F = 82 after DTW. Verified against the PDF. Mentioning this
> because it changes the interpretation completely: the hand-crafted similarity matrix is
> *useless* on its own, and DTW post-processing is doing nearly all the work for it.

The most consequential fact in this table: **DTW post-processing collapses the difference
between a 1.1M and a 28M parameter model** (99 vs 99). The learned representation matters
far less than the alignment decoder. That is either an indictment of the representation or
evidence that Vienna 4x22 is too easy — Table 4 suggests the latter, partly.

**Table 3 — model sizes** `[PDF]`

| Model | #params | Residual dim | Blocks | Heads | Batch size | Decoder head params |
|---|---|---|---|---|---|---|
| TGN-large | 28M | 512 | 8 | 8 | 8 | 27M |
| TGN-mid | 5.7M | 256 | 6 | 8 | 16 | 2M |
| TGN-small | 1.1M | 128 | 4 | 8 | 24 | 0.6M |

**Table 4 — five hard pieces vs. SOTA references** `[PDF]`
*All values are note-match F-scores in %, except the runtime column in seconds.*

*Default data:*

| Model | B. Op.53 3rd | C. Op.9 №1 | C. Op.9 №2 | C. Op.10 №11 | C. Op.60 | Mean | Runtime |
|---|---|---|---|---|---|---|---|
| Nakamura HMM | 98 | 99 | 98 | 94 | 95 | 98 | 152 |
| Peter AutomaticNoteMatcher | 99 | 84 | 94 | 96 | 89 | 92 | 588 |
| Peter DualDTWMatcher | 99 | 98 | 99 | 96 | 98 | **98** | 96 |
| TGN-large + DTW | 99 | 99 | 98 | 96 | 97 | **98** | 33 |
| TGN-mid + DTW | 96 | 98 | 98 | 96 | 98 | 97 | 27 |
| TGN-small + DTW | 99 | 98 | 98 | 96 | 97 | **98** | **21** |

*20 % mismatch data:*

| Model | B. Op.53 3rd | C. Op.9 №1 | C. Op.9 №2 | C. Op.10 №11 | C. Op.60 | Mean | Runtime |
|---|---|---|---|---|---|---|---|
| Nakamura HMM | 39 | 65 | 35 | 20 | 63 | 44 | 6458 |
| Peter AutomaticNoteMatcher | 82 | 74 | 89 | 71 | 75 | 78 | 808 |
| Peter DualDTWMatcher | 85 | 96 | 94 | 80 | 83 | 88 | 208 |
| TGN-large + DTW | 94 | 96 | 95 | 93 | 94 | 94 | 42 |
| TGN-mid + DTW | 92 | 95 | 96 | 92 | 95 | 94 | 38 |
| TGN-small + DTW | 94 | 97 | 95 | 93 | 94 | **95** | 31 |

> ⚠️ The printed "Mean" column does not always equal the arithmetic mean of the five
> listed values (e.g. Nakamura default: 98, 99, 98, 94, 95 → arithmetic 96.8, printed 98).
> Transcribed as printed. Likely a note-count-weighted mean, but the paper does not say.
> **If we reproduce this table we should report the unweighted mean explicitly** and note
> the divergence.

**Reproducibility problem.** The five test pieces are from the **proprietary** Magaloff
(Flossmann et al. 2010) and Zeilinger (Cancino-Chacón et al. 2017) datasets. We cannot
reproduce Table 4 without access to those. **We will need to construct a public equivalent**
— see §5 for the recommendation. The Nakamura baseline is the C++ `AlignmentTool_v190813`
from https://midialignment.github.io/.

**Training setup** `[PDF]`: 200k steps, single GTX 1080 Ti (12 GB), LR 5×10⁻⁴, cosine
annealing with warm restarts every 2k steps, FFN inverted bottleneck 4× residual dim.
Trained on **1,032 valid MIDI files from (n)ASAP** — using only the score and performance
MIDI, **not** the alignments. Ground truth is entirely synthetic: each MIDI file is copied
and both copies are augmented. Augmentation is recomputed **every batch**. 480 ticks/beat
at 120 bpm (1 tick ≈ 1 ms). Transposition applied to the maximal extent of the keyboard,
affecting both subsequences.

**Table 1 — augmentation parameters** `[PDF]` (this recipe is the paper's real contribution)

| Feature | Noise / mismatch |
|---|---|
| Tempo `T_t` | `g·T_t·2^{n_t}`, `g ~ N(1, 0.5)`, `n_t ~ N(0, 0.5)` |
| Onset `O_t` | `O_t + n_t`, `n_t ~ U(−50, 50)` ticks |
| Velocity `V_t` | `V_t + n_t`, `n_t ~ U(−10, 10)` |
| Duration `D_t` | `D_t + n_t`, `n_t ~ U(−250, 250)` ticks |
| Repeats | `P = 1`, `#notes ~ U(8, 200)` |
| Skips | `P = 1`, `#notes ~ U(8, 200)` |
| Insertions | `P = 0.2`, random location |
| Deletions | `P = 0.2`, random location |
| Trills | `P = 1`, `#notes ~ U(20, 100)` |

Mismatches are inserted contiguously, at most one augmentation of each type per 512-note
sequence. The paper is explicit that these values are "given for reproducibility and
transparency, although various other variations were tested, we do not claim that these are
optimal values" — i.e. **the augmentation distribution is unoptimized and is a legitimate
avenue for us to improve on.**

**The 20 % mismatch test manipulation** `[PDF]`: extended (100+ note) mismatches covering
~20 % of notes; each 512-note subsequence pair contains exactly **two** mismatching segments,
one in `s1` and one in `s2`, each contiguous with randomly sampled notes. Critically, the
paper notes "such randomized contiguous mismatches are different from the synthetic mismatch
segments seen during training" — so this is a genuine out-of-distribution test, not a
train/test-matched one.

**Architecture** `[PDF]`: two 512-note subsequences, each prepended with a "default note"
(the dustbin) → 513 each. Structured tokenization: relative onset, pitch, duration, velocity
= 4 tokens/note; the four embeddings are **summed** per note plus a learned positional
embedding → one residual stream of length 1026 for the concatenated pair. Self-attention
over the concatenation therefore covers s1–s1, s2–s2, s1–s2 and s2–s1 simultaneously. Final
LayerNorm + one dimension-preserving linear, split, **dot product** → 513×513 similarity.
Loss = dual cross-entropy (softmax across rows + across columns), unmatched notes targeting
the default note; a third CE from a decoder head that reads the similarity matrix and
predicts the match index directly.

**DTW match extraction** `[PDF]`: on the **reciprocal** of learned pairwise distance, with
step set `[[0,1],[1,1],[1,0]]` and weights `[1,2,1]` — chosen so the directions are
normalized under Manhattan distance and the diagonal is *not* favored. The path is
explicitly **not** used as direct note prediction; it defines a coarse mapping `m: ℝ→ℝ` by
linear interpolation between path onset times. Then, separately per pitch `p`, DTW on onset
sequences finds pairs minimizing distance between `s2^p` and `m(s1^p)`; newly matched notes
**overwrite** the original path and `m` is updated. For files longer than 512 notes:
similarity matrices are computed on 512-note windows with **stride 256** and aggregated into
a global matrix.

**The authors' own stated next step** `[PDF]`, verbatim from §6: SoftDTW "appears promising
to bridge this gap while keeping sensible alignment constraints in an end-to-end model.
However, we want to stress again that the monotonicity condition of (soft)DTW does not
strictly hold in symbolic music even though it has proven an effective heuristic."

They also flag: "an open question is whether this type of token-based match representation
learning can be used in audio or multimodal domains," and that the training data is specific
to solo piano common-practice repertoire.

---

## 2. The evaluation protocol — read this before building the harness

This is where most reproduction attempts go wrong. There are **two incompatible F-score
definitions** in the literature.

### 2.1 The parangonar / TISMIR definition (match-only F) `[PDF]`

An alignment is a list of labelled tuples: `match(score_id, perf_id)`, `deletion(score_id)`,
`insertion(perf_id)`. Verbatim from TISMIR §3.5:

- "A predicted match is counted as a true positive only if the notes are matched in the
  ground truth alignment."
- "A predicted insertion or deletion note is counted as true positive if the note is marked
  as an insertion or a deletion in the ground truth, respectively."
- "A false positive is a predicted note label that isn't in the ground truth, a false
  negative is a ground truth note label that isn't predicted. All notes have a predicted
  label as well as a ground truth label, so false negatives always correspond to false
  positives, and vice versa, albeit not necessarily the same number."
- "this measure does not discriminate the types of errors: mismatches, false matches, and
  false insertions or deletions."

**Worked example, TISMIR Table 1** `[PDF]` — memorize this, it is the unit test:

| | |
|---|---|
| Prediction | `m(sn1,pn1)`, `m(sn2,pn2)` |
| Ground truth | `d(sn1)`, `i(pn1)`, `m(sn2,pn2)` |
| True positive | `m(sn2,pn2)` |
| False positive | `m(sn1,pn1)` |
| False negative | `d(sn1)`, `i(pn1)` |
| Precision | 1/2 |
| Recall | 1/3 |

Key properties:
- **Exact note-ID pair equality. No timing tolerance whatsoever.** There is no ±50 ms window.
- **No true negatives exist.**
- P/R/F are computed **per label class separately**; published headline numbers are the
  **match** F-score only.
- **Aggregation is per-performance, then averaged** — "F-measures, averaged across each
  dataset," harmonic mean "of the predicted performance-wise alignment." **Not** pooled over
  all notes. This matters: a short piece counts as much as a long one.

Implementation: `parangonar.evaluate.fscore_alignments(prediction, ground_truth, types)` in
`parangonar/evaluate/eval.py` — it filters both lists by label then does
`TP = [p for p in pred if p in gt]`, plain dict equality `[2nd]`. Edge case: both filtered
lists empty returns `(1.0, 1.0, 1.0)` `[2nd]`. `partitura` does **not** ship an F-score
function — it supplies I/O (`load_match`, `load_alignment_from_ASAP`) and
`performance_codec.get_time_maps_from_alignment` for the temporal metrics `[2nd]`.

### 2.2 The RUMAA definition (pooled F_align) `[HTML]`

RUMAA's `F_align` counts "matched note pairs **and** inserted/deleted notes as True
Positives, unmatched predicted notes as False Positives, and missing ground-truth notes as
False Negatives."

**This pools all three label classes into one number.** It is systematically more generous
than the match-only F, because correctly identifying an insertion now earns credit in the
same pool. **RUMAA's 98.4 and DualDTW's 99.8 are not on the same scale.** Any table we
publish must state which definition is in use.

### 2.3 Practical guidance for our harness

1. Implement the TISMIR match-only F as the primary metric; assert against the Table 1
   worked example as a unit test.
2. Also implement pooled `F_align` so we can be listed on RUMAA's table.
3. Report insertion-F and deletion-F separately — the headline match-F hides exactly the
   failure mode (ornaments, trills) that we care most about.
4. Aggregate per-performance and report mean ± SD, matching ISMIR 2023's presentation.
5. Use the **robust subset** of (n)ASAP for evaluation (§4.2).
6. Significance testing convention in this literature: Friedman + Wilcoxon signed-rank with
   Bonferroni (TISMIR), or two-sided sign test on performance-wise rankings (ISMIR 2023).

---

## 3. Newer work, 2024–2026

### 3.1 RUMAA — the repeat-robustness bar

> Sungkyun Chang, Simon Dixon, Emmanouil Benetos. **"RUMAA: Repeat-Aware Unified Music
> Audio Analysis for Score-Performance Alignment, Transcription, and Mistake Detection."**
> **WASPAA 2025**. arXiv:[2507.12175](https://arxiv.org/abs/2507.12175) ·
> DOI 10.1109/WASPAA66052.2025.11230990

**Table 2 — note-level `F_align` on the Vienna piano dataset** `[HTML]`

| Model | w/o repeat | w/ repeat |
|---|---|---|
| Nakamura HMM | 99.0 | 36.4 |
| hDTW+sym (= AutomaticNoteMatcher) | 98.5 | 28.2 |
| GlueNote Transformer | 98.5 | **12.7** |
| AMT + Nakamura | 97.4 | 31.8 |
| AMT + hDTW+sym | 96.9 | 26.5 |
| AMT + GlueNote | 96.9 | 26.3 |
| **RUMAA** | 98.4 | **98.4** |

Protocol `[HTML]`: "w/o repeat" uses **unfolded** scores across all songs; "w/ repeat" uses
**original** scores for Mozart K331 and Schubert D783 — the two of the four Vienna pieces
that carry repeat symbols. `AMT + X` rows are the audio pipeline: transcribe first, then
align symbolically. The top three rows are fed symbolic MIDI directly.

**Architecture** `[HTML]`: score encoder takes MusicXML → ABC notation → pre-trained M3
encoder (CLaMP2), 768-dim projected to 1024. Audio encoder is a 12-layer Transformer on
spectrograms (16 kHz, 12 frames/s), 1024-dim. Decoder is a 6-block Transformer with
hierarchical cross-attention (audio first, then score) and a **tri-stream output**: T1
score-aligned performance transcription, T2 performance-aligned score conversion, T3 edit-op
tagging (`Insert`/`Delete`/`Match`/`Repeat`).

Other results `[HTML]`: score-informed transcription on (n)ASAP F_on 99.1 / F_off-vel 93.6 /
MAE_vel 4.0; without score F_on 96.1 (Maestro), 95.9 ((n)ASAP).

**Stated limitation** `[HTML]`: "struggles with long audio sequences (over one minute) due
to cross-chunk memory limits, restricting its use on extended real-world recordings."
Evaluation limited to clean single-instrument data; online processing unexplored.

**How to read this for MLign.** RUMAA solves repeats by *tokenizing the edit operations* and
letting an autoregressive decoder emit them — it never needs the score pre-unfolded. That is
the right idea. But it is audio-input, it is capped at ~1 minute, and its `F_align` is the
generous pooled metric. **A symbolic-only system that gets ≥98 pooled F_align on folded
scores at full piece length would be a clear, defensible contribution.**

### 3.2 Repeat-structure inference — the cheap solution

> Silvan Peter, Patricia Hu, Gerhard Widmer. **"How to Infer Repeat Structures in MIDI
> Performances."** arXiv:[2505.05055](https://arxiv.org/abs/2505.05055) (May 2025), 3 pages.

Smith–Waterman-style **local** alignment on pitch sets: match gain +1 / mismatch −1, gain
clipped to [0, 10], accumulation `ag(i,j) = max(ag(i−1,j), ag(i−1,j−1)) + m(i,j)`. Then
**score-informed backtracking** enumerates all musically valid structural versions (repeat at
most twice, coda once), scores each by total gain minus a per-segment penalty, and picks the
best `[2nd]`.

**Result** `[2nd]`: ~85 % correct on the 110 (n)ASAP performances with notated repeats/skips
— and on inspection **all 17 "errors" were faulty dataset annotations**, not method
failures. Shipped in parangonar as `RepeatIdentifier`.

**This is the single highest-value, lowest-cost thing we can adopt.** It says the repeat
problem is largely solvable as *preprocessing*, decoupled from the matcher. Combining
`RepeatIdentifier`-style unfolding with a strong note matcher plausibly reaches RUMAA-level
repeat robustness without an autoregressive decoder.

### 3.3 Other 2024–2026 work

| Paper | Venue | Relevance |
|---|---|---|
| **"Just Label the Repeats for In-The-Wild Audio-to-Score Alignment"** — Bukey, Feffer, Donahue. arXiv:[2411.07428](https://arxiv.org/abs/2411.07428) | ISMIR 2024 | Human-in-the-loop: click repeat signs to unroll, then vanilla DTW. Measure accuracy on repeat subset M13_R: 0.17 baseline → 0.20 automatic → **0.83–0.95 with labels**, <6 s annotation/page `[2nd]`. Evidence that *fully automatic* jump detection from the alignment matrix alone is unreliable, and that **unfolding beats jump-aware DP**. |
| **PianoCoRe: Combined and Refined Piano MIDI Dataset** — Ilya Borovik. DOI 10.5334/tismir.333, arXiv:2605.06627 | TISMIR 2026 | 250,046 performances / 5,625 pieces / 483 composers / 21,763 h. Note-aligned subset **PianoCoRe-A = 157,207 performances aligned to 1,591 scores** — by a wide margin the **largest open note-aligned corpus**, built with DualDTW + a refinement pipeline (RAScoP). **Obvious training set for MLign** — but note the alignments are DualDTW output, so training on them naively caps us at DualDTW quality. `[2nd]` |
| **"A Flexible Encoding Model for Non-Unique Note Alignments"** — Chiruthapudi, Štefunko, Peter, Hu, Hajič jr., Cancino-Chacón. arXiv:2606.28032 | MEC 2026 | Extends the Match file format for **non-one-to-one** alignments via "virtual pointer notes," plus richer `section` lines. Directly relevant if MLign must emit many-to-many links (ornaments, arpeggios, continuo, rehearsal repetitions). `[2nd]` |
| **"Score-Agnostic Structure Analysis in Large-Scale Performance Datasets"** — Hu, Peter, Widmer. arXiv:2605.25951 | MEC 2026 | Groups transcriptions of the same piece by structural realization via pairwise alignment + hierarchical clustering; no score or audio needed. ~1,500 transcriptions / 88 works. No accuracy figures found. `[2nd]` |
| **"Precise and Simple Audio-to-Score Alignment"** — Peter, Hu, Widmer. arXiv:2605.20014 | MEC 2026 | DP matching of audio onset/spectral features directly to score positions, no transcription model. **86 ms mean / 21 ms median** error vs. 135/49 ms audio-to-audio baseline; pure symbolic MIDI-to-score reference is **6 ms mean / 0 ms median**. That last figure is a useful sanity ceiling for symbolic alignment. `[2nd]` |
| **"A Study of Parallelizable Alternatives to DTW for Aligning Long Sequences"** — Yang, Shaw, Tsai. arXiv:2607.15478 | 2026 | **ParDTW** computes *exact* DTW diagonally, **1.5–2 orders of magnitude** faster on long sequences, with GPU implementations. Directly useful if we keep a DTW backend. `[2nd]` |
| **"Estimating the Reliability of DTW Alignments Using Circumstantial Evidence"** — Pratapneni, Yuan, Tsai. arXiv:2607.15443 | ISMIR 2026 | Unsupervised reliability metric: re-estimate a path segment with FlexDTW, measure agreement. **Aggregate AUROC 0.97**. Good template for a confidence head. `[2nd]` |
| **Matchmaker** — Park, Cancino-Chacón, Chiruthapudi, Nam. arXiv:[2510.10087](https://arxiv.org/abs/2510.10087) | ISMIR 2025 | Open-source real-time score-following framework with standardized metrics. Uses "only the pieces in the **MAESTRO v2 test split**" for (n)ASAP → 43 pieces / 59 performances / 100,958 notes / 2.65 h. **This is the closest thing to a community-standard (n)ASAP split.** `[2nd]` |
| **LadderSym** — Chou et al. arXiv:2510.08580 | ICLR 2026 | Two-stream interleaved transformer with inter-stream alignment for practice-error detection. MAESTRO-E missed-note F1 **26.8 → 56.3**, extra-note F1 **72.0 → 86.4**. `[2nd]` |
| **"Pairing Real-Time Piano Transcription with Symbol-level Tracking"** — Peter, Hu, Widmer. arXiv:[2505.05078](https://arxiv.org/abs/2505.05078) | SMC 2025 | Online, audio front-end. Peripheral. `[2nd]` |
| **CODA** — Yang, Chen, Han. arXiv:2607.21899 | ISMIR 2026 | Cascaded online *discontinuity-aware* alignment for image-based score following. Numbers not retrieved. `[2nd]` |
| **FuSiLi** — Bukey, Novack, Jung, Jeong, Donahue. arXiv:2607.10023 | ISMIR 2026 | **Sinkhorn-based soft alignment** over local image-patch/audio-frame features, learning local correspondence from *global* supervision only. Architecturally the closest recent work to our matching head. Numbers not in abstract. `[2nd]` |

### 3.4 Explicit negative results (valuable — don't re-search these)

- **No successor to TheGlueNote exists.** Peter & Widmer's 2025–26 output moved to repeat
  inference, audio-to-score, and structure clustering — not a new note-matching network.
  Verified via arXiv listings, dblp, and citation graphs `[2nd]`.
- **Semantic Scholar's citation index for TheGlueNote is stale** — returns only 2 citations
  (RenCon 2025 and RUMAA) `[HTML]`. Use the TISMIR 10.5334/tismir.149 citation graph instead
  (~30 citing works) `[2nd]`.
- **ICASSP 2025/2026: nothing on symbolic note alignment** `[2nd]`.
- **TASLP 2024–2026: nothing directly on score-performance note alignment** beyond Tsai's
  DTW-parallelization work `[2nd]`.
- **ISMIR 2026 has no public program yet** (conference 8–12 Nov 2026, Abu Dhabi;
  notifications 10 Jul 2026). Only ~22 accepted papers have reached arXiv `[2nd]`. **Re-check
  the program in September/October** — this is the most likely source of a surprise competitor.
- **Eita Nakamura has published no new alignment work 2024–2026** `[2nd]`. His 2017 HMM
  remains the standing baseline.

---

## 4. Datasets

### 4.1 Batik-plays-Mozart

> Patricia Hu, Gerhard Widmer. **"The Batik-plays-Mozart Corpus: Linking Performance to
> Score to Musicological Annotations."** *ISMIR 2023*.
> arXiv:[2309.02399](https://arxiv.org/abs/2309.02399) ·
> data https://github.com/huispaty/batik_plays_mozart

`[2nd]` throughout this subsection unless noted.

- **12 Mozart sonatas / 36 movements** (KV 279–284, 330–333, 457, 533). **102,421 performed
  notes**, 223.28 min.
- Alignment composition: **98,318 matches (95.36 %), 4,103 insertions (4.44 %), 207
  deletions (0.20 %).** The high insertion rate is mostly unnotated ornament/trill notes —
  which makes Batik the **best public testbed for ornament handling**, and directly relevant
  to our espressivo ornament-provenance work.
- Pianist Roland Batik on a **Bösendorfer SE290**, 1.25 ms onset/offset resolution. Audio is
  commercially sold (Gramola) and **not** redistributed.
- **Score source is NOT KernScores.** Scores are the **Neue Mozart-Ausgabe**, taken from the
  DCML *Annotated Mozart Sonatas* MuseScore 3 files, converted to **MusicXML** with unique
  note IDs. The original 1990s Batik alignments used an in-house encoding; the corpus is a
  score2score remapping onto NMA note IDs.
- **Alignment format: match file v1.0.0** (Foscarin et al., MEC 2022, arXiv:2206.01104; spec
  at https://cpjku.github.io/matchfile/). Line form:
  `snote(n1-1,[E,n],4,1:1,0,1/4,0.0000,1.0000,[v1,staff1])-note(n1,64,761,1351,60,1,0)`.
  Load with `partitura.load_match()`. A `curate_data` branch also emits
  `perf2score/<mvt>/alignment.csv` note-ID pairs.
- **Musicological annotations**: Hentschel, Neuwirth & Rohrmeier, *The Annotated Mozart
  Sonatas*, TISMIR 2021, DOI 10.5334/tismir.63, https://github.com/DCMLab/mozart_piano_sonatas
  — ~15,000 DCML harmony labels, ~1,100 cadence labels, phrase boundaries, wired in as a git
  submodule. Batik covers 36 of those 54 movements.
- **Gotcha**: `scores_edited/` contains scores pruned to only the repeats the pianist
  actually played. Use `pt.score.unfold_part_maximal(score[0], ignore_leaps=False)`.
- Branches: `main` (curated), `curate_data` (pipeline), `audio_aligned_midi` (MIDI/match
  adjusted for MIDI↔audio clock drift).
- **Evaluated on by**: ISMIR 2023 Table 1 only (DualDTW 99.4 ± 0.7 vs Nakamura 98.5 ± 2.1).
  TheGlueNote does **not** evaluate on Batik. TISMIR 2023 does **not** either.

### 4.2 (n)ASAP

> ASAP: Foscarin, McLeod, Rigaux, Jacquemard, Sakai. *ISMIR 2020*, pp. 534–541.
> https://archives.ismir.net/ismir2020/paper/000127.pdf
> (n)ASAP note alignments: Peter et al., TISMIR 2023. https://github.com/CPJKU/asap-dataset
> CC BY-NC-SA 4.0.

**Table 6 from TISMIR — exact dataset statistics** `[PDF]`
(S = scores, P = performances, S-Notes / P-Notes = note counts, Mins = total minutes)

| Composer | S | P | S-Notes | P-Notes | Mins |
|---|---|---|---|---|---|
| Bach | 59 | 169 | 117,218 | 321,688 | 387 |
| Balakirev | 1 | 10 | 16,490 | 139,608 | 87 |
| Beethoven | 63 | 271 | 431,704 | 1,668,873 | 1,761 |
| Brahms | 1 | 1 | 3,514 | 1,667 | 6 |
| Chopin | 36 | 289 | 236,186 | 1,410,369 | 1,257 |
| Debussy | 2 | 3 | 10,800 | 14,470 | 13 |
| Glinka | 1 | 2 | 4,246 | 9,074 | 10 |
| Haydn | 12 | 44 | 56,230 | 190,942 | 215 |
| Liszt | 17 | 121 | 181,274 | 1,192,297 | 900 |
| Mozart | 6 | 16 | 33,796 | 73,927 | 78 |
| Prokofiev | 1 | 8 | 9,438 | 38,231 | 33 |
| Rachmaninoff | 4 | 8 | 13,552 | 20,941 | 30 |
| Ravel | 4 | 22 | 32,248 | 108,519 | 140 |
| Schubert | 15 | 62 | 134,576 | 453,464 | 499 |
| Schumann | 11 | 28 | 63,593 | 122,356 | 129 |
| Scriabin | 2 | 13 | 18,342 | 145,441 | 125 |
| **All** | **235** | **1067** | **1,363,207** | **5,911,867** | **5670** |

**How the ground truth was actually made** `[PDF]` — critical, and widely misdescribed:

- Scores are **unfolded** to played length first; repeated notes get suffixed IDs
  (`n112-1` = first play, `n112-2` = second).
- Final alignment used **linear interpolation with 0.95 window fuzziness** on existing ASAP
  beat annotations as anchors, then the combinatorial pitch-wise symbolic matcher. Verbatim:
  "Notably, no dynamic time warping of any kind was used for the final note alignment."
- The best segment-constrained algorithm "produces reliable note alignments in more than
  97 % of cases."
- **Nothing was manually corrected.** Verbatim: "No alignment mistakes are corrected during
  this process." The authors visually inspected all 1,067 alignments in Parangonada (~3 min
  each) and labelled each robust / non-robust.
- **832 of 1,067 (~78 %) assessed robust.** The check was deliberately conservative — a piece
  is non-robust if *any* perceivably misaligned note is present. Verbatim recommendation:
  "We recommend using only the robust alignments for critical tasks like model evaluation."
  Exposed as `robust_note_alignment` in `metadata.csv` / `asap_annotations.json`.

**Consequence for MLign**: (n)ASAP ground truth is itself anchor-point-interpolation output
at ~97 % reliability, with a known ~22 % questionable fraction. **We cannot meaningfully
measure above ~99 % on it.** Use the robust subset, and treat any headline above ~99 as
noise, not progress.

**Train/test split**: there is **no official split** `[2nd]`. The README warns to deduplicate
on `(title, composer)` because two folders can hold the same piece with different repeat
structures. Community practice: Matchmaker uses the **MAESTRO v2 test split** (43 pieces / 59
performances / 100,958 notes) `[2nd]`. TheGlueNote trains on 1,032 (n)ASAP MIDI files and
tests elsewhere `[PDF]`. **Recommendation: inherit MAESTRO v2's split** — closest to a
convention, and it makes us comparable to Matchmaker.

Formats: `note_alignment.tsv` (`xml_id`, `midi_id`, `track`, `channel`, `pitch`, `onset`),
`.match` files, and Parangonada CSVs. Loader:
`pt.io.importparangonada.load_alignment_from_ASAP` `[2nd]`. v2.1 (May 2025) merged ASAP v1.2
and fixed 7 scores + 18 alignments `[2nd]`.

### 4.3 Vienna 4x22

> Werner Goebl, 1999. DOI 10.21939/4X22.
> https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/index.html
> Symbolic mirror with match v1.0.0 + MusicXML + MIDI: https://github.com/CPJKU/vienna4x22

- **4 pieces/excerpts (two by Chopin, one by Mozart, one by Schubert) × 22 pianists = 88
  performances, >40k performed notes, ~2 hours** `[PDF, TISMIR §3.4]`.
- Pieces `[2nd]`: Chopin Etude Op. 10 No. 3; Chopin Ballade Op. 38; Schubert Deutscher Tanz
  D 783 No. 15; Mozart K 331 mvt. 1. Measure ranges differ between the mdw page and the
  CPJKU README (the latter's larger spans presumably count written repeats) — **unresolved**.
- **Only two of the four pieces (K331, D783) carry repeat symbols** — these are the ones
  RUMAA's "w/ repeat" condition uses `[HTML]`.
- Recorded Jan–Feb 1999 on a Bösendorfer SE290 `[2nd]`.
- **This is the only fully public dataset of the classic four.** Magaloff, Zeilinger and
  (audio for) Batik are proprietary.
- **It is saturated**: DualDTW 99.8 ± 0.4, Nakamura 99.5 ± 0.5, with both at F = 1.0 on 38 of
  88 performances `[PDF]`. Reporting a win here is not evidence of anything.

### 4.4 Availability summary — the reproducibility problem

| Dataset | Public? | Used by |
|---|---|---|
| Vienna 4x22 | ✅ yes | TISMIR, ISMIR23, GlueNote, RUMAA |
| (n)ASAP | ✅ yes | TISMIR (produced), GlueNote (training) |
| Batik-plays-Mozart | ✅ symbolic yes, audio no | ISMIR23 only |
| Magaloff | ❌ proprietary | TISMIR, ISMIR23, GlueNote (test) |
| Zeilinger | ❌ proprietary | TISMIR, ISMIR23, GlueNote (test) |
| PianoCoRe-A | ✅ yes | — (new) |

**The two hardest benchmarks in the literature are both proprietary.** We cannot reproduce
GlueNote's Table 4 or ISMIR 2023's Table 1 in full. See §5.

---

## 5. Directly applicable techniques

### 5.1 Sinkhorn / optimal transport matching heads (the SuperGlue lineage)

**SuperGlue** (Sarlin, DeTone, Malisiewicz, Rabinovich, CVPR 2020,
arXiv:[1911.11763](https://arxiv.org/abs/1911.11763)) is the origin of the dustbin
formulation `[2nd]`. Score matrix is a plain inner product of *unnormalized* descriptors
(magnitude therefore encodes confidence); the reference implementation additionally divides
by `d^0.5`. The **dustbin is a single learned scalar** `z` filling the entire appended row
*and* column, initialized to 1.0. Marginals `a = [1_M, N]`, `b = [1_N, M]` let the dustbin
absorb up to N (resp. M) units of mass. Sinkhorn runs **100 iterations in log space** via
`u ← log_mu − logsumexp(Z + v)`. Loss is NLL over ground-truth matches plus unmatched
entries, with **no term on the dustbin-dustbin cell**.

**LightGlue** (Lindenberger, Sarlin, Pollefeys, ICCV 2023,
arXiv:[2306.13643](https://arxiv.org/abs/2306.13643)) **abandons Sinkhorn** for
dual-softmax × per-point matchability `[2nd]`:

```
sim       = mdesc0 @ mdesc1.T          # mdesc = Linear(x) / d**0.25
σ_i       = sigmoid(Linear(x_i))       # matchability, per keypoint, both sides
log P_ij  = logsoftmax_row(sim) + logsoftmax_col(sim) + log σ_i + log σ_j
log P_i,⊥ = log(1 − σ_i)
```

**This is the single most important architectural recommendation in this report.** The
dustbin becomes a **learned per-element unary head** rather than one global scalar. For note
alignment that is exactly right: "this ornament note is unmatchable" is a property of the
*note*, not a global constant. TheGlueNote's "default note" token is closer to SuperGlue's
global scalar and is, in our reading, its weakest design choice.

Dual-softmax vs. Sinkhorn `[2nd]`: LoFTR (arXiv:2104.00680) implemented both and found them
comparable. Efficient LoFTR (CVPR 2024, arXiv:2403.04765) drops dual-softmax at inference
entirely. RoMa v2 (arXiv:2511.15706) uses a learned certainty head. **Verdict: if our
supervision already contains one-to-one ground truth, dual-softmax + per-note matchability
gets the same effect as Sinkhorn for free and with cleaner gradients.** Sinkhorn's real cost
is 100 sequential logsumexp passes over an M×N matrix, which is prohibitive at note-sequence
scale. Note also that FuSiLi (§3.3) is applying Sinkhorn soft alignment to music right now,
so this lineage is live in our field.

### 5.2 Differentiable DTW / soft-DTW losses

**soft-DTW** (Cuturi & Blondel, ICML 2017, arXiv:[1703.01541](https://arxiv.org/abs/1703.01541))
`[2nd]`: soft-min `min^γ(a) = −γ log Σ exp(−a_i/γ)`, recursion
`R_{i,j} = D_{i,j} + min^γ(R_{i−1,j}, R_{i,j−1}, R_{i−1,j−1})`. Forward **and** backward are
O(nm) time and space; the backward pass is a second DP producing the **expected alignment
matrix** E — which is precisely the object we would want as soft supervision for a similarity
matrix. Paper uses γ ∈ {0.1, 1.0}.

**Known failure modes** `[2nd]`: soft-DTW is **non-convex in its inputs** and
`sDTW(x,x) ≠ 0` — it can even be negative — so it is *not* a divergence and minimizing it can
push outputs *away* from the target. **Use the debiased soft-DTW divergence instead**
(Blondel, Mensch, Vert, AISTATS 2021, arXiv:[2010.08354](https://arxiv.org/abs/2010.08354)):
`D(x,y) = sDTW(x,y) − ½sDTW(x,x) − ½sDTW(y,y)`, which is nonnegative and zero iff x = y.

**Memory** `[2nd]`: standard CUDA soft-DTW materializes `D ∈ ℝ^{B×N×M}` and is hard-capped
around N = 1024 by the 1024-thread anti-diagonal launch. A 2026 tiled implementation
(arXiv:2602.17206) reaches N = M = 2048, and a fused recompute mode drops memory from
O(BNM) to O(B(N+M)) — 8,256 MB → 161 MB at B=32, L=512, D=64 — at a **10–15× runtime
penalty**. The original backward also overflows for γ < 0.1.

**The caveat that matters most**, from TheGlueNote's own discussion `[PDF]`: "the monotonicity
condition of (soft)DTW does not strictly hold in symbolic music." Applying soft-DTW naively
across notes will fight the data at every chord and ornament. See §5.3 for the fix.

### 5.3 Monotonic attention and the two-level monotonicity prior

Monotonic attention (Raffel et al., ICML 2017) and MoChA (Chiu & Raffel, ICLR 2018,
arXiv:[1712.05382](https://arxiv.org/abs/1712.05382)) `[2nd]`: selection probability
`p_ij = σ(e_ij)`, expected alignment
`α_ij = p_ij((1−p_{i,j−1})α_{i,j−1}/p_{i,j−1} + α_{i−1,j})`, parallelizable via
cumprod/cumsum. Zero-mean Gaussian noise is added pre-sigmoid during training so `p` becomes
effectively binary. **The cumprod underflows on long sequences** — this is the known hazard.
Monotonic Alignment Search (Glow-TTS, NeurIPS 2020,
arXiv:[2005.11129](https://arxiv.org/abs/2005.11129)) is a Viterbi over a strictly monotonic
lattice, O(T·S), reused in VITS; CTC forced alignment is the same object with a blank symbol.

**The design recommendation for MLign.** Score→performance alignment is monotonic at
**onset** granularity but *not* at **note** granularity — chords are unordered sets,
ornaments expand one score note into many performance notes, and the hands desynchronize.
So: **enforce monotonicity on a coarse onset-index axis, and leave within-onset matching to a
free, non-monotonic assignment head.** This is exactly what DualDTW and TheGlueNote both do
empirically — global DTW for the coarse map, then *pitch-separated onset DTW* to recover
within-chord and trill cases that no globally monotonic path can express. The difference is
that they do it as hand-written post-processing; **we should make it a differentiable
two-level structure trained end-to-end.** That is the clearest architectural gap in the
literature and, combined with §5.1's per-note matchability, is our strongest thesis.

### 5.4 Long-context handling

The hard constraint: **FlashAttention does not help us**, because our output *is* the M×N
matrix. A 100k×100k fp32 score matrix is ~40 TB. Any design that must materialize it
globally is dead on arrival.

What actually works:

- **Windowing** — TheGlueNote's 512-note windows with stride 256, aggregating into a global
  sparse matrix `[PDF]`. Simple, proven, and the obvious starting point.
- **Anchor cascade — MrMsDTW** (Prätzlich, Driedger, Müller, ICASSP 2016) `[2nd]`: coarse DTW
  → project path to fine level → derive anchors whose maximum rectangular extent
  `R(A) ≤ τ` → local DTWs inside those rectangles → refinement pass around anchors →
  concatenate. **Memory is constant in τ, not in sequence length.** Verified figures: Wagner
  *Das Rheingold* (8752 s vs 8930 s) needs **1455.8 GB** for full DTW, 204.4 MB for MsDTW,
  and **0.763 MB** for MrMsDTW at τ = 10⁵, "basically yielding the same alignments." Note
  TISMIR 2023 explicitly says its two-step approach is "conceptually similar to multi-scale
  DTW as presented by Prätzlich et al." `[PDF]`.
- **Do NOT use FastDTW** — Wu & Keogh, "FastDTW is approximate and Generally Slower than the
  Algorithm it Approximates," arXiv:[2003.11246](https://arxiv.org/abs/2003.11246) `[2nd]`.
- **ParDTW** (§3.3) for an exact, GPU-parallel DTW backend `[2nd]`.
- **Coarse-to-fine** as in LoFTR/Efficient LoFTR — the vision analogue of the anchor cascade.

### 5.5 Repeats and structural jumps

Covered in §3.1–3.2. The consolidated recommendation: **treat structure inference as a
separate preprocessing stage** (Smith–Waterman local alignment + enumeration of musically
valid structural versions, per arXiv:2505.05055) so the matcher never has to represent a jump
inside its attention window. "Just Label the Repeats" (§3.3) independently supports this:
unfolding beats jump-aware DP, and fully automatic jump detection from the alignment matrix
alone is unreliable. JumpDTW (Fremerey, Müller, Clausen, ISMIR 2010) achieved >99 % bar-level
accuracy on Beethoven sonatas but **requires block boundaries as input**, and later work
reports it degrades badly when jump locations are uncertain `[2nd]`.

---

## 6. Implications for MLign

**On what to target.** Bar A is saturated and half-unmeasurable. Build to Bars B and C.
Specifically, the defensible contribution is: *a symbolic-only note matcher that handles
repeats and structural mismatch natively at full piece length, at DualDTW-level accuracy on
clean data.* Nothing published does all three.

**On architecture.** Four concrete changes from TheGlueNote, in descending expected value:

1. **Per-note matchability head (LightGlue-style) instead of a single default-note token.**
   Unmatchability is a per-note property; ornaments are exactly the case where this matters.
2. **Two-level alignment structure**: differentiable monotonic alignment on the onset axis,
   free assignment within onsets. Makes the pitch-separated onset DTW that everyone bolts on
   as post-processing into a trained component.
3. **Structure/repeat inference as preprocessing** (Smith–Waterman + valid-version
   enumeration), not as something the network must learn inside a 512-note window.
4. **Anchor cascade (MrMsDTW-style) for the 10k–100k note regime**, replacing fixed-stride
   windowing.

**On training data.** TheGlueNote's augmentation table (§1.4) is the recipe to start from,
and the authors explicitly disclaim it as unoptimized — tuning it is legitimate low-hanging
fruit. Two extensions worth making: (a) generate repeats/skips that are **structurally
valid** rather than random contiguous blocks, and (b) exploit our MEI+MPM synthetic pipeline
to produce ground truth that is genuinely exact rather than DualDTW output. That second point
is a real advantage over PianoCoRe-A: **training on DualDTW-generated alignments caps us at
DualDTW quality.**

**On evaluation — the thing most likely to bite us.** Three traps:

1. **Two incompatible F definitions** (§2.1 vs §2.2). State which one every number uses.
2. **The proprietary-benchmark problem** (§4.4). GlueNote's headline hard-case table uses
   Magaloff and Zeilinger, which we cannot obtain. **Recommendation: define a public
   "MLign-hard" benchmark** — Batik + the (n)ASAP robust subset + Vienna 4x22 with folded
   scores — apply GlueNote's exact 20 % mismatch manipulation, and run parangonar's matchers
   and TheGlueNote checkpoints ourselves. That gives a like-for-like comparison on data
   anyone can download, which is worth more than matching an unreproducible number.
3. **The (n)ASAP ceiling** (§4.2). Ground truth is ~97 % reliable with 22 % flagged
   non-robust and nothing hand-corrected. Above ~99 % we are measuring annotation noise.

**Also worth noting**: five of TheGlueNote's six-piece test set were TISMIR 2023's
*hyperparameter tuning* pieces (§1.2). When citing GlueNote Table 4 as an independent
comparison against parangonar, that overlap should be disclosed.

---

## 7. Source URLs

**Primary (numbers to beat)**
- TISMIR 2023 — https://transactions.ismir.net/articles/10.5334/tismir.149 · https://apmcleod.github.io/pdf/ASAP-align.pdf
- ISMIR 2023 (DualDTW) — https://arxiv.org/abs/2401.00466
- TheGlueNote ISMIR 2024 — https://arxiv.org/abs/2408.04309 · https://github.com/sildater/thegluenote
- RUMAA WASPAA 2025 — https://arxiv.org/abs/2507.12175

**Code / data**
- parangonar — https://github.com/sildater/parangonar
- partitura — https://cpjku.github.io/partitura_tutorial/notebooks/02_alignment/Symbolic_Music_Alignment.html
- (n)ASAP — https://github.com/CPJKU/asap-dataset
- Batik-plays-Mozart — https://github.com/huispaty/batik_plays_mozart
- Vienna 4x22 — https://github.com/CPJKU/vienna4x22 · https://repo.mdw.ac.at/projects/IWK/the_vienna_4x22_piano_corpus/index.html
- Match file spec — https://cpjku.github.io/matchfile/
- Nakamura AlignmentTool — https://midialignment.github.io/

**Techniques**
- SuperGlue https://arxiv.org/abs/1911.11763 · LightGlue https://arxiv.org/abs/2306.13643
- LoFTR https://arxiv.org/abs/2104.00680 · Efficient LoFTR https://arxiv.org/abs/2403.04765
- soft-DTW https://arxiv.org/abs/1703.01541 · soft-DTW divergence https://arxiv.org/abs/2010.08354
- MoChA https://arxiv.org/abs/1712.05382 · Glow-TTS/MAS https://arxiv.org/abs/2005.11129
- FastDTW critique https://arxiv.org/abs/2003.11246

**2024–2026**
- Repeat structures https://arxiv.org/abs/2505.05055 · Just Label the Repeats https://arxiv.org/abs/2411.07428
- Matchmaker https://arxiv.org/abs/2510.10087 · Real-time transcription+tracking https://arxiv.org/abs/2505.05078
