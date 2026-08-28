# MLign decode — porting contract

Normative specification of everything `src/mlign/infer.py` does around the
model, for the TypeScript/browser port. Where this document and the Python
disagree, the Python wins and this document is a bug — but the golden fixtures
in `test/golden/` are generated from the Python, so a disagreement shows up as
a failing fixture rather than as silent drift.

The reference implementation is NumPy. Most of the porting risk is not in the
algorithms — they are short — but in NumPy semantics that a naive JavaScript
transcription gets wrong: unstable sorts, first-vs-last argmax ties, pairwise
summation, and scalar type promotion inside float32 arithmetic. §8 collects
every one of those. Three of them were found by writing this document's rules
out in JavaScript and diffing against the fixtures; each one changed the
output on at least one real piece.

Scope: from the two note tables to the alignment triples. The model forward
itself is out of scope (see `scripts/export_onnx.py`); §5 defines exactly which
tensors cross that boundary in each direction, so the ONNX parity check and
this contract meet at a specified interface.

---

## 1. Fixtures

`test/golden/<slug>/` — one directory per case:

| file | contents |
| --- | --- |
| `manifest.json` | tables, row, windows, featurized token arrays, decode stage snapshots, triples, metadata |
| `sim.f32.bin` | `(n, m)` accumulated similarity logits |
| `null_s.f32.bin` | `(n,)` per-score-note null logit |
| `null_p.f32.bin` | `(m,)` per-perf-note null logit |
| `conf.f32.bin` | `(n, m)` dual-softmax confidence — the decode's working matrix |
| `feat_w<NN>_cont.f32.bin` | `(T, 6)` `featurize` continuous features, one file per window |

Every `.bin` is **raw, little-endian, C-order (row-major) IEEE-754 float32 with
no header**, so in Node:

```js
const b = readFileSync(path);
// a Buffer's byteOffset is rarely 4-aligned — slice into a fresh ArrayBuffer
const a = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
```

`manifest.arrays[name]` documents each one: `{file, dtype, endianness, order,
shape, offset, bytes, sha256, note}`. Nothing is gzipped and nothing is
truncated. Total across all seven fixtures is 34.7 MB, of which 25 MB is the
Berceuse alone; a consumer that only needs the algorithm covered can take the
other six for 9.0 MB and leave the Berceuse referenced in place.

### Manifest keys

| key | meaning |
| --- | --- |
| `schema` | `"mlign-golden/1"` |
| `meta` | `n`, `m`, `tokens`, `windowed`, checkpoint path + sha256, model config, git commit, timestamps, library versions, `constants`, `overrides`, `coverage`, `counts` |
| `meta.constants` | every constant in §7, at the values **that fixture's run actually used** |
| `meta.overrides` | non-empty only where `mlign.infer`'s module constants were patched for that run; the patched names and values (§4) |
| `meta.coverage` | window census: `sim_cells_by_window_count`, `score_notes_by_window_count`, `perf_notes_by_window_count`, `sim_at_minus_1e9`, `null_s_at_plus_1e9`, `null_p_at_plus_1e9`, `covered_sim_min/max` (§5) |
| `score` | `{id, onset (quarters), duration (quarters), pitch, voice}` in `ScoreTable` order |
| `perf` | `{id, onset (sec), duration (sec), pitch, velocity}` in `PerfTable` order |
| `row` | `{score: [[onset×720, duration×720, pitch, voice%5], …], perf: [[onset×1000, duration×1000, pitch, velocity], …]}` — the exact featurize/decode input (§2) |
| `windows` | `[[s0, s1, p0, p1], …]`; `[[0, n, 0, m]]` when unwindowed |
| `baseline_pairs` | windowed pieces only: the baseline aligner's `(score_idx, perf_idx)` matches that `coarse_windows` builds on (§4.2) |
| `featurized` | per window: `{window, n, m, T, pitch[], segment[], position[], cont}` where `cont` names an entry in `arrays` |
| `stages` | `anchors_raw`, `anchors`, `dtw_ax`, `dtw_ay`, `map1_ax`, `map1_ay`, `map2_ax`, `map2_ay`, `rounds_run`, `round1_matched_s`, `round2_matched_s`, `rescued` |
| `triples` | the `decode()` output |

Index `i` into `score`/`row.score` is `score_idx`; index `j` into
`perf`/`row.perf` is `perf_idx`. JSON floats are Python `repr` output, i.e.
shortest-round-trip, so `JSON.parse` recovers the exact float64.

### Cases

| slug | n | m | windows | m/d/i | size | exercises |
| --- | --- | --- | --- | --- | --- | --- |
| `schubert-d783-15` | 328 | 316 | 1 | 314/14/2 | 1.0 MB | baseline single-window path; negative score onsets (anacrusis) |
| `schubert-d783-15-win128` | 328 | 316 | 5 | 314/14/2 | 1.1 MB | the whole windowing path at 1/25 the Berceuse's size — see below |
| `mozart-k331-1st-mov` | 482 | 479 | 1 | 478/4/1 | 2.1 MB | — |
| `chopin-op38-p19` | 731 | 729 | 1 | 727/4/2 | 4.6 MB | a DTW backtrack tie that only breaks correctly under exact float32 (§8.3) |
| `chopin-berceuse-op57` | 1756 | 1728 | 9 | 1626/130/102 | 25.1 MB | windowing at realistic scale; anchors dropped by the monotone filter (1609 → 1590) |
| `synth-rescue` | 10 | 9 | 1 | 4/6/5 | 24 KB | the residual rescue (§6.7); single assignment round |
| `synth-flat-map` | 8 | 11 | 1 | 1/7/10 | 24 KB | the `len(ax) < 2` zero map (§6.5); single assignment round |

`schubert-d783-15-win128` is the same piece and performance as
`schubert-d783-15`, generated with `mlign.infer`'s module constants patched at
runtime to `WIN_SCORE = 128` and `MAX_SINGLE_TOKENS = 0` (`src/mlign/infer.py`
is not modified; `meta.overrides` records the patch and `meta.constants` shows
the effective values). Lowering `WIN_SCORE` alone would not have windowed it —
Schubert is 646 tokens, under the 2000 default — hence the second override,
where `0` means "always window". It gives five windows over 328 score notes and
covers everything the Berceuse covers of the windowing path: the accumulation
bookkeeping, the `first`/count logic, the `-1e9` sentinel (38.8 % of `sim`
here), and `coarse_windows`' dependence on the baseline cluster-DTW (306
baseline pairs). It is not redundant with the unwindowed Schubert: the two
agree on 313 of 314 match pairs but differ on one, and every confidence
differs, so a port cannot pass it by accident.

**A port reading these fixtures must take `WIN_SCORE`, `WIN_STRIDE` and
`MAX_SINGLE_TOKENS` from `meta.constants`, not from its own defaults.** That is
the one place where hardcoding §7 will fail a fixture.

The two `synth-*` cases are decode-only: fabricated `sim`/`null` arrays from a
seeded generator (`scripts/make_golden.py:synth_case`), no model and no
featurize. They exist because no real piece reaches those branches — see §6.7
for why the rescue is unreachable on well-aligned input. Their recorded output
is the real Python `decode()`'s, which is all the port must match.

### What must match

* **Exactly**: every triple's `label`, `score_idx`, `perf_idx`, and their order;
  the window tuples; the featurize token arrays; `baseline_pairs`; the anchor
  list; the DTW path.
* **To 1e-6 relative**: `confidence` values, `sim`, `conf`, `cont`.

Bit-exactness on `conf` is *not* required and not achievable: `np.exp` on a
float32 array and `Math.fround(Math.exp(x))` differ by up to one float32 ulp.
Measured on the fixtures, following this document, 97–99.9 % of `conf` entries
come out bit-identical and the worst relative error is 2.5e-7. That is small
enough that every discrete decision downstream — argmax, thresholds, DTW ties —
came out identical on all seven fixtures.

---

## 2. Tables and the row

`ScoreTable` and `PerfTable` are both sorted by `np.lexsort((pitch, onset))`:
**primary key onset ascending, secondary key pitch ascending, stable**. The
sorted position is the index used everywhere downstream. The browser builds
these tables from MEI rather than MusicXML; the sort is still part of the
contract.

Score onsets are in quarters and **may be negative** (an anacrusis puts the
first note at −1.0 in `schubert-d783-15`). Performance onsets are in seconds.

`tables_to_row` converts to the model's units:

```
row.score[i] = [ score.onset[i] * 720.0, score.duration[i] * 720.0,
                 score.pitch[i], score.voice[i] % 5 ]
row.perf[j]  = [ perf.onset[j] * 1000.0, perf.duration[j] * 1000.0,
                 perf.pitch[j], perf.velocity[j] ]
```

`% 5` is Python's floor-mod. Voices are non-negative in practice, but use
`((v % 5) + 5) % 5` in TypeScript if the source can produce negatives.

**Everything downstream reads `row`, never the table.** `decode` recovers times
as `row.score[i][0] / 720` and `row.perf[j][0] / 1000`, and `x * 720 / 720` is
**not** `x`: for uniformly random float64 it differs by one ulp about 17 % of
the time (2 % for `x * 1000 / 1000`). Carry the row values; do not shortcut back
to the table's onsets. Fixtures store both so this is checkable.

---

## 3. featurize

`dataset.featurize` turns a row (or a window's slice of one) into the model's
token arrays. `T = 2 + n + m`; the layout is

```
index:   0          1 … n            n+1        n+2 … n+1+m
token:   [S-marker] score notes      [P-marker] perf notes
```

Times first, in float64:

```
s_onset[i] = row.score[i][0] / 720          s_dur[i] = row.score[i][1] / 720
p_onset[j] = row.perf[j][0]  / 1000         p_dur[j] = row.perf[j][1]  / 1000
```

Deltas are `np.diff(..., prepend=first)`, i.e. computed **after** the division:

```
s_delta[0] = 0                              s_delta[i] = s_onset[i] - s_onset[i-1]
p_delta[0] = 0                              p_delta[j] = p_onset[j] - p_onset[j-1]
```

### 3.1 `cont` — (T, 6) float32

All six columns are computed in float64 and rounded to float32 once, at the
end. Both marker rows are all-zero. For a note with `delta`, `dur`, `pitch`,
and a segment-specific `extra`:

| col | value |
| --- | --- |
| 0 | `log1p(max(delta, 0) * 2)` |
| 1 | `log1p(max(dur, 0) * 2)` |
| 2 | `pitch / 64 - 1` |
| 3 | `(pitch % 12) / 11 * 2 - 1` |
| 4 | `extra` |
| 5 | segment flag: `0` for score notes, `1` for perf notes |

`extra` is `voice / 4` for score notes (`voice` is the already-`% 5` value from
the row, so `extra ∈ {0, 0.25, 0.5, 0.75, 1}`) and `velocity / 64 - 1` for perf
notes.

Column 3 associates left to right: `((pitch % 12) / 11) * 2 - 1`. Pitches are
0…127 so JavaScript's truncating `%` agrees with NumPy's floor-mod.

`Math.log1p` in float64, stored into a `Float32Array`, reproduces
`np.log1p(float64).astype(float32)` bit-for-bit — verified across all 12
windows of the five real fixtures, 69048 values, zero mismatches.

### 3.2 `pitch`, `segment`, `position` — (T,) int

```
pitch    = [128, ...score pitches, 128, ...perf pitches]      // 128 = MARKER_PITCH
segment  = [0 × (1 + n), 1 × (1 + m)]
position = [0, 1, …, n, 0, 1, …, m]                            // restarts per segment
```

The S-marker is position 0 and score note `i` is position `i + 1`; the P-marker
is position 0 again and perf note `j` is position `j + 1`.

### 3.3 collate

With one sample there is no padding: `T = 2 + n + m`, `n_max = n`, `m_max = m`,
`pad` all false. The port never batches, so collate is the identity plus a
leading batch dimension. `target_s`/`target_p` are training-only; the port does
not need them.

---

## 4. Windowing

```
MAX_SINGLE_TOKENS = 2000     WIN_SCORE = 384
stride = WIN_SCORE // 2 = 192            MARGIN_SEC = 3.0
```

If `2 + n + m <= MAX_SINGLE_TOKENS`, there is exactly one window `(0, n, 0, m)`
and §4.1–4.2 are skipped entirely. Otherwise the windows come from
`coarse_windows`, which depends on the classical baseline aligner. Two fixtures
take that branch: `chopin-berceuse-op57` at the shipped constants (3486 tokens
→ 9 windows) and `schubert-d783-15-win128` at `WIN_SCORE = 128`,
`MAX_SINGLE_TOKENS = 0` (646 tokens → 5 windows). Read the three constants from
`meta.constants` when checking against a fixture.

### 4.1 Onset clusters

Used by both the baseline (§4.2) and the decode's DTW map (§6.4), with
different epsilons:

```
score clusters: split between i-1 and i where s_onset[i] - s_onset[i-1] >  1e-9
perf  clusters: split between j-1 and j where p_onset[j] - p_onset[j-1] >  0.05
```

This is a **consecutive-difference** split, not greedy clustering: a run of
notes each 40 ms apart forms one perf cluster of unbounded span. (The
`baseline.py` docstring calls it "50 ms greedy clusters"; the code is the
above, and the code is the contract.) Both onset arrays are sorted, so clusters
are contiguous index ranges. There is always at least one cluster per side.

### 4.2 `align_baseline` (`src/mlign/baseline.py`)

Only its *match* pairs are used, and only as `(score_idx, perf_idx)`.

Cluster cost is Jaccard distance over pitch **sets** (deduplicated):

```
cost[i][j] = float32( 1 - |Si ∩ Pj| / |Si ∪ Pj| )        ( = 1 when the union is empty )
```

DTW with `GAP = 0.75`, accumulator `acc` a `(ns+1, mp+1)` **float32** array:

```
acc[0][0] = 0
acc[0][j] = float32( running float64 sum of j × GAP )       for j = 1 … mp
acc[i][0] = float32( running float64 sum of i × GAP )       for i = 1 … ns
acc[i][j] = min( acc[i-1][j-1] + cost[i-1][j-1],
                 acc[i-1][j]   + GAP,
                 acc[i][j-1]   + GAP )                       all adds in float32
```

`np.inf` never participates: the whole first row and column are finite, so
every cell is reachable. Backtrack from `(ns, mp)` while `i > 0 and j > 0`,
recomputing the three candidates in float32:

```
d = acc[i-1][j-1] + cost[i-1][j-1]
v = acc[i-1][j]   + GAP
h = acc[i][j-1]   + GAP
best = min(d, v, h)
if   best == d:  record cluster pair (i-1, j-1); i -= 1; j -= 1
elif best == v:  i -= 1
else:            j -= 1
```

**The `d` → `v` → `h` order is load-bearing** and so is the float32 arithmetic;
see §8.3. Reverse the recorded pairs.

Within each paired cluster `(ci, cj)`, walk the score notes in ascending index
order and give each the lowest-index not-yet-used perf note of the same pitch
in `cj`; unpaired notes on either side are deletions/insertions the caller
ignores. (Diagonal DTW steps strictly advance both indices, so no cluster
appears twice and the "already used" bookkeeping across clusters never fires.)

Verified exactly: 1526 pairs on `chopin-berceuse-op57`, 306 on
`schubert-d783-15-win128`.

### 4.3 `coarse_windows`

Let `pairs` be the baseline matches sorted ascending by `(score_idx,
perf_idx)`, `s_idx` their score indices and `p_time[k] = p_onset[pairs[k][1]]`.
If `pairs` is empty, return `[(0, n, 0, m)]`.

```
for s0 = 0, 192, 384, …  while s0 < n:
    s1  = min(n, s0 + 384)
    sel = { k : s0 <= s_idx[k] < s1 }
    if |sel| < 2:  t_lo, t_hi = p_onset[0], p_onset[m-1]
    else:          t_lo = min(p_time[sel]) - 3.0
                   t_hi = max(p_time[sel]) + 3.0
    p0 = searchsorted_left(p_onset, t_lo)      // first index with p_onset >= t_lo
    p1 = searchsorted_right(p_onset, t_hi)     // first index with p_onset >  t_hi
    p0 = max(0, p0)
    p1 = min(m, max(p1, p0 + 1))
    emit (s0, s1, p0, p1)
    if s1 >= n: break
```

The `break` (not the loop condition) ends it. Its only effect is to suppress
the trailing short window that `range(0, n, stride)` would otherwise emit after
one has already reached the end of the score — `[1728, 1756)` for the Berceuse,
`[320, 328)` for `win128`. What that leaves is worth stating exactly, because
it is easy to assume otherwise:

* every window **except the last** is exactly `WIN_SCORE` score notes;
* the **last** window is `n - s0` notes, normally *shorter* than `WIN_SCORE`
  (220 for the Berceuse, 72 for `win128`) — the break fires precisely when a
  window would run past the end, so a full final window only happens when `n`
  is an exact multiple. It is always longer than the stride, though.
* consecutive windows overlap by exactly `WIN_SCORE - stride` score notes —
  never more, never less. For even `WIN_SCORE` that equals the stride (192 and
  64 in the two fixtures); for odd it is one greater.

Verified exhaustively over `WIN_SCORE ∈ {2, 3, 5, 8, 16, 128, 129, 384, 385}`
crossed with `n ∈ [1, 1200)`.

Windows are **not** token-budgeted: `2 + (s1-s0) + (p1-p0)` may exceed
`MAX_SINGLE_TOKENS`, and does in practice (the Berceuse's largest is 854).

Verified exactly: all 9 windows on `chopin-berceuse-op57`, all 5 on
`schubert-d783-15-win128`.

---

## 5. Logit accumulation

Model interface, per window, batch size 1:

* in: `pitch (1, T) int64`, `cont (1, T, 6) float32`, `segment (1, T) int64`,
  `position (1, T) int64`, `pad (1, T) bool` (all false), `n_score = [ns]`,
  `n_perf = [mp]`
* out: `logits_s2p (1, ns, mp+1) float32`, `logits_p2s (1, mp, ns+1) float32`

Column `mp` of `logits_s2p` is the score note's null ("deleted") logit; column
`ns` of `logits_p2s` is the perf note's null ("inserted") logit.

Accumulate over windows in **float32**:

```
sim   = full((n, m), -1e9)      cnt      = zeros((n, m))
null_s = zeros(n)               null_s_cnt = zeros(n)
null_p = zeros(m)               null_p_cnt = zeros(m)

for (s0, s1, p0, p1) in windows:
    ns, mp = s1 - s0, p1 - p0
    block[a][b] = logits_s2p[a][b] + logits_p2s[b][a]        for a<ns, b<mp
    for a, b in the window:
        if cnt[s0+a][p0+b] == 0:  sim[s0+a][p0+b] = 0        // clear the sentinel first
        sim[s0+a][p0+b] += block[a][b]
        cnt[s0+a][p0+b] += 1
    null_s[s0+a] += logits_s2p[a][mp];  null_s_cnt[s0+a] += 1
    null_p[p0+b] += logits_p2s[b][ns];  null_p_cnt[p0+b] += 1

sim    = sim / max(cnt, 1)                  // uncovered cells stay at -1e9
null_s = null_s / max(null_s_cnt, 1)
null_p = null_p / max(null_p_cnt, 1)
null_s[null_s_cnt == 0] = 1e9               // never covered ⇒ certainly unmatched
null_p[null_p_cnt == 0] = 1e9
```

Both accumulations are float32-in, float32-out. In JavaScript, read from a
`Float32Array`, add or divide in float64, write back to the `Float32Array`: for
`+ - * /` with float32 operands the double rounding is provably innocuous
(float64's 53 bits exceed the 2 × 24 + 2 needed), so this is bit-identical to
NumPy. That guarantee does **not** extend to adding a float64 constant — see
§8.3.

### 5.1 What the stored `sim` actually holds

Two things about `sim.f32.bin` that are easy to get backwards, both verified
bit-exactly against a per-window re-run of the model on
`schubert-d783-15-win128`:

**It is the mean over covering windows, not a running sum.** The `/ cnt`
divide above has already been applied. `sim == raw accumulated sum` is false
wherever a cell is covered twice; `sim == sum / cnt` is exactly true. Same for
`null_s` and `null_p`. A port that re-derives `sim` from ONNX output must
divide before comparing.

**A singly-covered cell is exactly 2 × the model's bilinear similarity.** The
model computes one `sim` tensor and slices it both ways, so
`logits_s2p[:ns, :mp]` and `logits_p2s[:mp, :ns]ᵀ` are *the same numbers* —
verified as exact array equality, not approximate. `block` therefore doubles
it. This factor 2 is not cosmetic: it halves the effective softmax temperature
in §6.1, so dropping it (or adding a third direction) changes every confidence
and shifts the `>= 0.35` anchor cut. Keep the sum of the two directional
blocks as written rather than "optimising" it to `2 * sim` or to one direction.

### 5.2 Which sentinel values actually appear

Both `-1e9` and `1e9` are exactly representable in float32, so a reader can and
should test them with `===` rather than a tolerance. They are real values in
the stored arrays, not padding to be skipped, and nothing else in the arrays
comes near them: across the fixtures, covered `sim` spans −142 to +41 and the
null logits span +4.5 to +13.7. `meta.coverage` records each fixture's
`covered_sim_min`/`covered_sim_max` so this is checkable rather than assumed.

| fixture | `sim == -1e9` | `null_s == 1e9` | `null_p == 1e9` |
| --- | --- | --- | --- |
| `schubert-d783-15-win128` | 40176 of 103648 cells (38.8 %) | 0 | 0 |
| `chopin-berceuse-op57` | 2016824 of 3034368 cells (66.5 %) | 0 | 0 |
| the five single-window fixtures | 0 | 0 | 0 |

The `-1e9` set coincides *exactly* with the cells no window covered — the
generator asserts this on every run. Window coverage per fixture is in
`meta.coverage`; for the Berceuse, 624520 cells are covered once and 393024
twice, and for `win128`, 35184 once and 28288 twice.

**No fixture produces a `+1e9` null.** For score notes that is structural, not
luck: `coarse_windows` steps `s0` by `stride = WIN_SCORE / 2` and each window
spans `WIN_SCORE`, so consecutive windows overlap and their union is always the
whole of `[0, n)` — a score note cannot be uncovered.

For **performed** notes there is no such guarantee, and this is a live case
rather than a hypothetical. A window's perf range is its anchors' span widened
by `MARGIN_SEC = 3.0` and clipped to the note array, so a performed note more
than 3 s from the nearest anchor of every window is covered by nothing —
lead-in noise, tuning, applause, a stray MIDI tail. It does not occur in either
windowed fixture (perf notes are covered once, twice or three times: 194/39,
1021/112, 513/165) nor in any of the 140 benchmark performances (§9), but it is
reachable on real input, and an uncovered performed note that is not handled
decodes as a **match at confidence 0.0**. §6.6.1 specifies the guard that makes
it an insertion instead; implementing it is not optional.

`-1e9` survives the softmax as an exact zero probability; `+1e9` makes the null
win outright, so an uncovered note becomes a deletion or insertion. Neither
needs special-casing in the decode — the arithmetic handles them — but a reader
that filters "implausible" values out of the arrays will corrupt both.

---

## 6. decode

```
anchor_conf = 0.35    tol_sec = 1.0    RESCUE_SEC = 0.35
```

Recover, in float64: `s_pitch`, `p_pitch` from column 2 of the row;
`s_onset[i] = row.score[i][0] / 720`; `p_onset[j] = row.perf[j][0] / 1000`.

### 6.1 Dual softmax

```
sm_s[i][:] = softmax_float32( [ sim[i][0…m-1], null_s[i] ] )[0…m-1]     // per row
sm_p[j][:] = softmax_float32( [ sim[0…n-1][j], null_p[j] ] )[0…n-1]     // per column
conf[i][j] = float32( sm_s[i][j] * sm_p[j][i] )
```

`softmax_float32(v)` is, in float32 storage throughout:

```
mx = max(v)                        // value only; no index, so ties are irrelevant
e[k] = exp(v[k] - mx)
s = sum(e)                         // NumPy pairwise summation — see §8.2
out[k] = e[k] / s
```

`conf` is the decode's only working matrix; `sim`/`null_*` are used again only
in §6.8.

### 6.2 Anchors

```
best_p[i] = argmax_j conf[i][j]          // FIRST maximum on ties
best_s[j] = argmax_i conf[i][j]          // FIRST maximum on ties
anchors_raw = [ (i, best_p[i]) for i = 0 … n-1
                where best_s[best_p[i]] == i
                  and conf[i][best_p[i]] >= float32(0.35)
                  and s_pitch[i] == p_pitch[best_p[i]] ]
```

Ascending `i` order. The threshold is compared **against `float32(0.35) =
0.3499999940395355`**, not against the float64 `0.35` — see §8.4.

Then `anchors = monotone_subset(anchors_raw)`.

### 6.3 `monotone_subset`

Longest chain that is increasing in score onset and **non-decreasing** in perf
onset.

```
sort the pairs by (s_onset[i], p_onset[j]), STABLY
t[k] = p_onset[ pairs[k][1] ]
tails = [], tail_idx = [], links = []
for k, tk in enumerate(t):
    pos = bisect_right(tails, tk)          // insertion point AFTER equal entries
    if pos == len(tails):  tails.push(tk);  tail_idx.push(k)
    else:                  tails[pos] = tk; tail_idx[pos] = k
    links.push( pos > 0 ? tail_idx[pos-1] : -1 )     // tail_idx[pos-1] as of this iteration
out = []
k = tail_idx[len(tails) - 1]
while k >= 0:  out.push(pairs[k]);  k = links[k]
return reverse(out)
```

Empty input returns empty. `bisect_right` (not `bisect_left`) is what makes the
chain non-decreasing rather than strictly increasing: equal perf onsets — a
chord — stay in the chain.

### 6.4 `cluster_dtw_map`

Clusters exactly as §4.1. For score cluster `Si` (index set `sc`) and perf
cluster `Pj` (index set `pc`):

```
jac      = |pitchset(Si) ∩ pitchset(Pj)| / |pitchset(Si) ∪ pitchset(Pj)|     (0 if the union is empty)
c_conf   = mean_float32( conf[a][b]  for a in sc, b in pc )     // row-major over the submatrix
cost[i][j] = float32( 0.5 * (1 - jac) + 0.5 * (1 - min(1, c_conf * 20)) )
```

`jac` and the final expression are float64; `c_conf` is `np.mean` on a float32
array, i.e. a **float32 pairwise sum divided by float32(count)** (§8.2). Using
a float64 accumulator here changes the DTW path on `chopin-op38-p19`.

Then the same DTW as §4.2 but with `GAP = 0.6`, and the backtrack records
onsets instead of cluster indices:

```
if best == d:  ax.push(s_onset[first index of score cluster i-1])
               ay.push(p_onset[first index of perf  cluster j-1])
```

Reverse both. Same `d`/`v`/`h` tie order, same float32 requirement — and here
`GAP = 0.6` is not representable in float32, which makes §8.3 bite.

### 6.5 The score-time → perf-time map

```
ax = concat( dtw_ax, [ s_onset[i] for (i, _) in anchors ] )      // DTW entries FIRST
ay = concat( dtw_ay, [ p_onset[j] for (_, j) in anchors ] )
order = stable_argsort(ax)                                       // ties keep input order
ax, ay = ax[order], ay[order]

if len(ax) >= 2:
    keep unique ax values, taking the FIRST index of each run
    s2p_time(x) = interp(x, ax_unique, ay_kept)
else:
    s2p_time(x) = 0                                              // constant zero
```

Two consequences of the concatenation order plus the stable sort plus
first-occurrence dedup: when a DTW knot and an anchor sit at the same score
onset, **the DTW knot wins**; when two anchors do, the earlier one in ascending
`i` wins.

`ay` need not be monotone — the union of two monotone sequences sorted by `x`
can decrease in `y` — so `s2p_time` is not necessarily monotone. That is what
makes §6.7 reachable.

The `len(ax) < 2` test is made **before** deduplication. Deduplication can
still leave a single knot, in which case `interp` returns that one value
everywhere (see below). `synth-flat-map` covers the `len(ax) < 2` branch.

#### `interp` — `np.interp` semantics

TypeScript has no `np.interp`. Reproduce it exactly, in float64:

```
interp(x, xp, fp):                     // xp non-decreasing, len(xp) >= 1
    if len(xp) == 1:      return fp[0]
    if x <  xp[0]:        return fp[0]                  // clamp, no extrapolation
    if x >  xp[-1]:       return fp[-1]                 // clamp, no extrapolation
    j = largest index with xp[j] <= x
    if j == len(xp) - 1:  return fp[j]                  // x == xp[-1]
    if xp[j] == x:        return fp[j]                  // exact hit short-circuits
    slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
    return slope * (x - xp[j]) + fp[j]                  // this association, not a lerp
```

Clamping matters: score onsets outside the map's range are common (anacrusis
before the first knot, final chord after the last), and extrapolating instead
of clamping puts them seconds away from any perf note.

### 6.6 Per-pitch assignment, two rounds

```
covered_p[j] = null_p[j] < 1e9        // §6.6.1 — computed once, loop-invariant

matched_s = [-1] × n        matched_p = [-1] × m

for round in 0, 1:
    reset matched_s, matched_p to -1
    for each distinct score pitch (iteration order is free — see §8.5):
        si = [ i : s_pitch[i] == pitch and matched_s[i] == -1 ]      // ascending
        pj = [ j : p_pitch[j] == pitch and matched_p[j] == -1
                   and covered_p[j] ]                                // ascending
        if si or pj is empty: continue
        expected = [ s2p_time(s_onset[i]) for i in si ]
        for (a, b) in assign_monotone(expected, [p_onset[j] for j in pj],
                                      tol_sec, conf[si][pj]):
            matched_s[si[a]] = pj[b];  matched_p[pj[b]] = si[a]

    if round == 0:
        got = [ (i, matched_s[i]) : matched_s[i] >= 0 ]              // ascending i
        if len(got) < 8:              break                          // keep round 0's matches
        pairs2 = monotone_subset(got)
        if len(pairs2) < 8:           break                          // keep round 0's matches
        rx = [ s_onset[i] for (i, _) in pairs2 ]
        ry = [ p_onset[j] for (_, j) in pairs2 ]
        keep unique rx values, first index of each run; index ry the same way
        s2p_time(x) = interp(x, rx, ry)                              // used by round 1 AND §6.7
```

Both `break`s leave round 0's `matched_s`/`matched_p` in place — they are not
discarded. There is no `len(rx) >= 2` guard here; a single surviving knot is
handled by `interp`'s `len(xp) == 1` case.

#### 6.6.1 The uncovered-performed-note guard

**A performed note that no window covered must never be matched.** It is an
insertion, unconditionally.

§5.2 explains that `coarse_windows` covers every *score* note structurally but
not every *performed* one — a window's perf range is its anchors' span widened
by `MARGIN_SEC`, so a performed note more than 3 s from the nearest anchor of
every window falls outside all of them. `accumulate_logits` marks these with
`null_p[j] = 1e9` and leaves the whole `sim` column at `-1e9`.

Without the guard, such a note decodes as a **match at confidence 0.0**, which
is a silent wrong answer rather than a crash. The mechanism: its `conf` column
is all zeros, so `assign_monotone`'s cost `delta - 0.5·tol·conf` collapses to
`delta` alone, and matching still beats skipping both sides whenever
`delta ≤ tol` (cost ≤ 1.0 against 2 × SKIP = 1.2). All it takes is a score note
of the same pitch whose real partner was never played, with the stray within
1 s of the mapped time — the DP then has nothing else to pair that pitch with
and takes the stray. Real shapes that produce this: lead-in noise, tuning,
applause, a stray MIDI tail.

So `covered_p` gates **both** places a `matched_p` entry can be set:

* the per-pitch candidate list `pj` in §6.6, and
* the `res_p` leftovers feeding the rescue in §6.7.

Nothing else needs guarding. Uncovered notes cannot become anchors (§6.2)
because their confidence is 0, which is below `anchor_conf` under any
threshold; and the final loop in §6.8 already emits an insertion for every `j`
with `matched_p[j] < 0`, so a gated note falls through to exactly the right
label, at confidence ≈ 1.0 (the `1e9` null dominates its softmax).

The score side is deliberately **not** guarded: `coarse_windows` steps by
`WIN_SCORE/2` with span `WIN_SCORE`, so consecutive windows always overlap and
their union is the whole of `[0, n)` (§5.2). `null_s` can therefore never carry
the sentinel, and adding a symmetric `covered_s` term would be dead code.

No fixture exercises this path — no benchmark performance in nASAP, dev-long,
Batik or Vienna 4x22 has a single uncovered performed note (see §9). It is
specified because it is reachable on real input the benchmarks do not contain,
and because two independent TypeScript ports found it before any fixture did.

#### `assign_monotone`

Monotone matching between two sorted time lists, in **float64**.

```
SKIP = tol * 0.6 = 0.6            INF = 1e18
dp[0][j] = j * SKIP                        // a multiplication, NOT a running sum
dp[i][0] = i * SKIP
for i = 1 … a, j = 1 … b:
    delta      = |expected[i-1] - actual[j-1]|
    match_cost = dp[i-1][j-1] + (delta <= tol ? delta - 0.5 * tol * conf_block[i-1][j-1] : INF)
    del_cost   = dp[i-1][j]   + SKIP
    ins_cost   = dp[i][j-1]   + SKIP
    best       = min(match_cost, del_cost, ins_cost)
    dp[i][j]   = best
    back[i][j] = (best == match_cost) ? 0 : (best == del_cost) ? 1 : 2
```

Tie order is **match, then del, then ins** — the mirror of §4.2's `d`/`v`/`h`.
`0.5 * tol * conf` is exact in both languages (halving a float32 is exact), so
the mixed float32/float64 arithmetic here needs no special handling. `dp` is
bounded by `(i + j) * SKIP` because every cell takes a min against a skip, so
`INF` only ever acts as a hard reject and never accumulates.

Backtrack from `(a, b)` while `i > 0 and j > 0`: `0` records `(i-1, j-1)` and
decrements both, `1` decrements `i`, `2` decrements `j`. Reverse the result.

Note that `dp`'s boundary is a *multiplication* while §4.2's DTW boundary is a
*running sum*, and in float64 those are not the same number (§8.6). The Python
multiplies here, so multiply.

### 6.7 Residual rescue

```
by_pitch_p = { pitch : [ j : matched_p[j] < 0, covered_p[j],             // §6.6.1
                             p_pitch[j] == pitch ] }                     // ascending j
cands = []
for i where matched_s[i] < 0:                                            // ascending i
    e = s2p_time(s_onset[i])
    for j in by_pitch_p[s_pitch[i]] (may be absent):
        d = |p_onset[j] - e|
        if d <= 0.35:  cands.push((d, i, j))
sort cands by (d, i, j)                                 // tuple order: d, then i, then j
for (d, i, j) in cands:
    if matched_s[i] < 0 and matched_p[j] < 0:
        matched_s[i] = j;  matched_p[j] = i
```

**This never fires on well-aligned real input**, and that is not an accident:
while `s2p_time` is monotone, `assign_monotone`'s optimum can never leave a
same-pitch pair that could be added without crossing, because adding one costs
at most `tol = 1.0` while removing the two `SKIP`s it replaces saves `1.2`. It
becomes reachable only when `s2p_time` is locally non-monotone (§6.5), which
the five real fixtures never produce. `synth-rescue` is the coverage: it
rescues one pair. Implement it anyway — the port must match the Python on
inputs the Python was written for, not only on inputs it currently sees.

### 6.8 Triples

Order: all `n` score notes in ascending index, then the unmatched perf notes in
ascending index.

```
for i = 0 … n-1:
    if matched_s[i] >= 0:
        { label: "match", score_idx: i, perf_idx: matched_s[i],
          confidence: conf[i][matched_s[i]] }
    else:
        { label: "deletion", score_idx: i,
          confidence: softmax_float32([ sim[i][0…m-1], null_s[i] ])[m] }
for j = 0 … m-1 where matched_p[j] < 0:
    { label: "insertion", perf_idx: j,
      confidence: softmax_float32([ sim[0…n-1][j], null_p[j] ])[n] }
```

The deletion/insertion confidence is the null's share of that note's softmax
mass — mathematically the same quantity `sm_s`/`sm_p` already hold, recomputed
from scratch. `align_with_model` then replaces `score_idx`/`perf_idx` with the
tables' string ids; the fixtures record the index form.

---

## 7. Constants

| name | value | where |
| --- | --- | --- |
| `MARKER_PITCH` | 128 | §3.2 |
| PPQ | 720.0 | §2, §3 |
| `MAX_SINGLE_TOKENS` | 2000 *(overridable — read `meta.constants`)* | §4 |
| `WIN_SCORE` | 384 *(overridable — read `meta.constants`)* | §4.3 |
| stride = `WIN_SCORE / 2` | 192 *(overridable)* | §4.3 |
| `MARGIN_SEC` | 3.0 | §4.3 |
| uncovered `sim` | −1e9 | §5 |
| uncovered null | +1e9 | §5 |
| `anchor_conf` | 0.35 | §6.2 |
| `tol_sec` | 1.0 | §6.6 |
| `SKIP` | `tol * 0.6` = 0.6 | §6.6 |
| assign `INF` | 1e18 | §6.6 |
| conf bonus factor | 0.5 | §6.6 |
| `RESCUE_SEC` | 0.35 | §6.7 |
| DTW gap (decode) | 0.6 | §6.4 |
| DTW gap (baseline) | 0.75 | §4.2 |
| DTW conf gain | 20.0 | §6.4 |
| score cluster eps | 1e-9 | §4.1 |
| perf cluster eps | 0.05 | §4.1 |

Every manifest carries these under `meta.constants`, at the values that
fixture's run used. Only the three windowing constants are ever overridden, and
only in `schubert-d783-15-win128`; everything else is identical across all
seven. When a fixture's `meta.overrides` is non-empty, `meta.constants` is
authoritative and this table is not.

---

## 8. NumPy semantics that JavaScript gets wrong

### 8.1 argmax ties → first occurrence

`np.argmax` returns the **lowest** index among equal maxima. A loop using `>`
matches; a loop using `>=` returns the last and is wrong. (`Math.max(...arr)`
plus `indexOf` gives the right answer but spreads the whole row onto the
argument stack, which throws outright once rows get long enough.) Used in §6.2
for both `best_p` and `best_s`.

`x.max()` in the softmax needs no tie rule — only the value is used.

### 8.2 `np.sum` / `np.mean` on float32 use pairwise summation

Not a naive left-to-right loop, and not a float64 accumulator. NumPy splits the
array recursively:

```
pairwise(a, off, n):
    n < 8    : naive left-to-right, float32
    n <= 128 : eight float32 accumulators seeded with a[off … off+7], strided by 8;
               combined as ((r0+r1) + (r2+r3)) + ((r4+r5) + (r6+r7)), then the
               fewer-than-8 remainder added left to right
    else     : n2 = floor(n/2) rounded DOWN to a multiple of 8;
               pairwise(a, off, n2) + pairwise(a, off+n2, n-n2)
```

with every add in float32. `np.mean` is `float32(pairwise_sum / float32(count))`.
Both verified against NumPy 2.2.6 on 300 random arrays — exact.

A naive float32 loop is badly wrong (200.003 vs 200.00003 summing 2000 copies
of 0.1). A float64 accumulator is close — within about 2e-7 relative — but not
equal, and in `cluster_dtw_map`'s `c_conf` that 2e-7 flips a DTW backtrack tie
on `chopin-op38-p19` and moves one path knot by 0.22 s. Use the pairwise form
wherever a float32 array is reduced: §6.1's softmax denominator and §6.4's
`c_conf`.

### 8.3 NEP-50: a Python float added to a float32 is cast to float32 *first*

`acc[i-1][j] + 0.6` with `acc` float32 computes `float32(acc) + float32(0.6)`,
where `float32(0.6) = 0.6000000238418579`. The naive JavaScript
`Math.fround(a + 0.6)` adds the **float64** `0.6` and rounds afterwards; the two
differ for about 0.5 % of operands. Define `const GAP32 = Math.fround(0.6)` and
add that.

This was the last divergence in `chopin-op38-p19`: with bit-identical `conf`
and the exact pairwise mean, the DTW path was still wrong until the gap was
frounded. It applies to every `float32 ± python_float` in §4.2 and §6.4. It
does **not** apply to float32-only arithmetic (§5), where double rounding
through float64 is provably harmless.

`GAP = 0.75` in the baseline is exactly representable, so the bug is invisible
there — which is exactly why it must be written down rather than discovered.

### 8.4 Threshold comparisons are also float32

`conf[i][j] >= 0.35` compares against `float32(0.35) = 0.3499999940395355`. A
`conf` value of exactly `float32(0.35)` is an anchor in NumPy and is not one in
naive JavaScript. Compare against `Math.fround(0.35)`.

By contrast `d <= 0.35` in §6.7 and `delta <= tol` in §6.6 are float64 on both
sides — no cast.

### 8.5 `np.argsort`'s default kind is **not** stable

`np.argsort(counts)` in §6.6 uses `kind='quicksort'` (introsort). On 40 equal
values it does not even return `arange` — it returns a scrambled permutation.
No JavaScript sort reproduces that, and none needs to: the per-pitch
subproblems are **disjoint**, because a pair is only ever formed between equal
pitches, so no pitch's assignment can touch another's `matched_s`/`matched_p`
entries. The "rarest pitch first" ordering is inert. Iterate pitches in any
order — ascending is the obvious choice.

`scripts/make_golden.py` asserts this on every fixture, on every run, by
replacing `np.argsort` with a random permutation and re-decoding three times.

Where stability *is* required the Python asks for it explicitly:
`np.argsort(ax, kind="stable")` in §6.5, and Python's `sorted` (Timsort, stable)
in §6.3. `Array.prototype.sort` has been stable since ES2019, but return `0`
for equal keys rather than falling through to an accidental comparison, or add
an explicit index tiebreak.

### 8.6 `np.unique`, and cumsum vs multiply

`np.unique(v, return_index=True)` returns the sorted distinct values and the
index of the **first** occurrence of each. In §6.5 `v` is already
non-decreasing, so this reduces to "take the first index of each equal run" —
which, with the concatenation order of §6.5, is what makes the DTW knot beat a
coincident anchor.

`np.unique` also *sorts*, but both call sites feed it sorted input, so the port
does not need a sort there.

Separately: §4.2/§6.4 build their DTW boundary with `np.cumsum` (a running
float64 sum) while §6.6 builds `dp`'s boundary with `np.arange * SKIP` (a
multiplication). For `0.6` the two differ in float64 from `k = 6` onward
(`3.5999999999999996` vs `3.6`). The DTW boundary is then rounded to float32,
which erases the difference at any realistic cluster count — but `dp` stays
float64, where it does not. Reproduce each as written: running sum in the DTW,
multiplication in `assign_monotone`.

### 8.7 float32 vs float64 boundaries

| quantity | dtype |
| --- | --- |
| `sim`, `null_s`, `null_p`, `cnt`, model logits | float32 |
| `cont` | float64 computed, stored float32 |
| `sm_s`, `sm_p`, `conf`, DTW `cost` and `acc` | float32 |
| `s_onset`, `p_onset`, `dtw_ax`/`ay`, map knots, `expected`, `dp` | float64 |
| triple `confidence` | float32 value widened to float64 for output |

The DTW `cost` matrices are float32 but their *inputs* (`jac`, the `c_conf`
expression) are float64, rounded once on store. `dp` in `assign_monotone` is
float64 throughout even though `conf_block` is float32.

### 8.8 `x * 720 / 720 ≠ x`

Covered in §2. The row is the source of truth for onsets, not the table.

### 8.9 `%` on negative values

Python's `%` floors, JavaScript's truncates. Only `voice % 5` (§2) and
`pitch % 12` (§3.1) use it, and both operands are non-negative in practice.
Guard `voice` if the MEI path can yield negatives.

### 8.10 `exp` cannot be made bit-exact

`np.exp` on float32 and `Math.fround(Math.exp(x))` differ by up to one float32
ulp. This is the only irreducible difference; everything else in this document
is reproducible bit-for-bit. Budget 1e-6 relative on `conf` and on triple
confidences, and exact agreement on every discrete decision. Measured on the
fixtures, the drift stayed at 2.5e-7 relative and changed no decision.

---

## 9. Coverage

Everything in this document was re-implemented in JavaScript from these rules
alone and diffed against all seven fixtures. The harness lives outside the repo
at `~/.claude/missions/mlign-wasm/artifacts/` — `contract-check.mjs` for §5–§6,
`windowing-check.mjs` for §4, `featurize-check.mjs` for §3. It is not the port;
it is the proof that the spec is complete and unambiguous.

Result: all 3767 triples across the seven fixtures identical in label, index
and order; anchor lists identical; DTW paths identical; 14 windows and 1832
baseline pairs across the two windowed fixtures identical; 69048 `cont` values
and 34524 `pitch`/`segment`/`position` entries across 17 windows bit-identical;
`conf` within 2.5e-7 relative.

### How often the uncovered-performed-note case actually arises

`coarse_windows` needs no model, so window coverage can be scanned over a whole
benchmark tier cheaply. Across **every performance of all four tiers** — 84
nASAP robust-holdout, 20 dev-long, 36 Batik (128 of the 140 windowed), plus
Vienna 4x22, which is single-window throughout — there is **not one uncovered
performed note, and not one uncovered score note**:

| tier | performances | windowed | uncovered perf notes | uncovered score notes |
| --- | --- | --- | --- | --- |
| nASAP robust holdout | 84 | 72 | 0 | 0 |
| dev-long | 20 | 20 | 0 | 0 |
| Batik | 36 | 36 | 0 | 0 |
| Vienna 4x22 (folded) | 44 | 0 | 0 | 0 |

So §6.6.1's guard is a no-op on every published benchmark: `covered_p` is
all-true, `pj` and `res_p` are unchanged, and `decode` is the identical
function. That is an identity, not a statistical result — which is why adding
the guard moved no benchmark number by any amount (§9 of
`artifacts/golden-gen.md` carries the measured A/B). The case is real
nonetheless; it is simply absent from curated concert recordings, which start
at the first note and stop at the last.

### Branches not covered by any fixture

Specified but unverified:

* the §6.6.1 guard itself — no fixture and no benchmark performance has an
  uncovered performed note. The reproduction is in the report; the shape is a
  score note whose partner was never played plus a stray performed note of that
  pitch within `tol_sec` of the mapped time.
* `coarse_windows` returning `[(0, n, 0, m)]` because the baseline found no
  matches at all.
* `assign_monotone` with an empty `si` or `pj` (the `continue`) — note that the
  guard makes an empty `pj` reachable where it previously was not, when every
  performed note of a pitch is uncovered.
* A window whose perf range collapses to `p0 == p1 == m`.
* The `pitchset` union being empty in a Jaccard cost (only possible with an
  empty cluster, which §4.1 cannot produce).
