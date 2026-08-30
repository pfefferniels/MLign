# Corpora — how each shard was generated

`data/` is not in the repository: the shards are hundreds of megabytes and are
regenerated rather than stored. So this file is the provenance. Every command
below reproduces its shard exactly — the samplers are seeded and the generators
are deterministic given `(seed, flags, generator commit)`.

Since 2026-08-29 each generator also writes `<out>.jsonl.recipe.json` beside its
output, recording argv, the resolved options and the generator's commit. The
shards predating that carry a backfilled sidecar marked `reconstructed`, whose
command came out of the session transcript that ran it — for a while that
transcript was the only surviving record, which is what the sidecar exists to
prevent. **`meta.seed` on a row is not provenance**: it pins the sampler and
says nothing about the flags, and the flags are what make two shards different.

## Ornament corpus, second pass (v4 / `models/mlign-v4.pt`)

`orn4` is what v4 trained on, beside `asap-orn2-train` and the usual synthetic
and real mix (read from the checkpoint's own `config.corpus`, not from a job
log). It is not a redesign of the section below: three faults in the generator
were found and fixed, and the shard was rebuilt.

- `breadth` multiplied the trill **rate** as well as its width, and
  `pairsPerQuarter` is calibrated for breadth 1. Every ornament shard before
  this one used breadth 2 or 3, so every ornament run so far trained on trills
  roughly 50 % too fast. `--trill-rate real` separates the two.
- `figureCfg` was built in `main()` and never passed to `sampleOrnaments`, so
  `--turn-lead` and `--upper-start` had never done anything.
- The written-out alternation was spaced [28, 65] ms against a real 72–92;
  `--orn-step` sets it.

```bash
node scripts/corpus/generate.mjs data/corpus/orn4-a.jsonl 40000 101 \
    --ornaments 1 --breadth 2 --add-rate 0.4 --imprecision subtle --jitter 18 \
    --orn-jitter 3 --min-dur 25 --exaggerate early --sign-share 0.8 \
    --add-orn-weight 0.8 --restrike 1.0 --turn-lead 0.8 --trill-rate real \
    --orn-step 65,100
node scripts/corpus/generate.mjs data/corpus/orn4-b.jsonl 40000 202 \
    --ornaments 1 --breadth 2 --add-rate 0.4 --imprecision subtle --jitter 18 \
    --orn-jitter 3 --min-dur 25 --exaggerate early --sign-share 0.8 \
    --add-orn-weight 0.8 --restrike 1.0 --turn-lead 0.8 --trill-rate real \
    --orn-step 65,100
node scripts/corpus/generate_real.mjs data/corpus/asap-orn2-train.jsonl \
    data/corpus/asap-spec-train.jsonl 301 --role train --takes 12 \
    --breadth 2 --add-rate 0.4 --imprecision subtle --jitter 18 \
    --orn-jitter 3 --min-dur 25 --exaggerate early
```

Unlike `orn-a`/`orn-b` below, the two `orn4` shards differ in **seed alone**
(101 and 202). The variation that used to come from differing breadth and
imprecision now comes from `--sign-share` and the figure samplers.

Measured on the shard rather than assumed from the flags, which is the whole
lesson of the three faults above:

```bash
.venv/bin/python scripts/corpus/validate.py data/corpus/orn4-a.jsonl --limit 4000
```

reports 8.26 ornament groups per 1000 score notes, 2.68 % of played notes, and
figure sizes to 18.

The figure-shape statistics that drove the recalibration — unison rate, whether
a figure falls after its principal, alternation step in ms — are **not** produced
by any script in this repository; they were measured ad hoc and live in
`notes/ornament/FINDINGS.md`, which is not published. With that caveat: orn4
reaches a median step of 82.9 ms against real Batik's 81.2, and a unison rate of
.46 against .52, both close. "Figure entirely after its principal" does not
close, at about .66 against Batik's .949.

That last gap is deliberate. `upperStartProb` is the knob that would shut it, and
Batik (.949) and ASAP (.525) disagree flatly on the statistic, because it records
how each corpus's ground truth was *derived* rather than how anyone played.
Fitting it would fit an annotation convention, so it stays at its default.

## Ornament corpus (v3 / `models/mlign-v3.pt`)

Realistic ornament rates and figures — 8.53 events per 1000 *sounding* notes,
grace runs and arpeggios three quarters of them. It replaced `early-a`/`early-b`
below, which put 141.8 ornament groups per 1000 notes into every piece and no
grace notes anywhere.

```bash
node scripts/corpus/generate.mjs data/corpus/orn-a.jsonl 40000 101 \
    --ornaments 1 --breadth 2 --add-rate 0.4 --imprecision natural --exaggerate early
node scripts/corpus/generate.mjs data/corpus/orn-b.jsonl 40000 202 \
    --ornaments 1 --breadth 3 --add-rate 0.4 --imprecision early --exaggerate early
```

`orn-a` and `orn-b` are **not** two draws of one distribution: the breadth of
the sampled `<temporalSpread>` and the imprecision level both differ, and that
is the point of having two.

`--add-rate 0.4` is load-bearing. The robustness layer's consonant additions land
in the *same attribution channel* as espressivo's ornaments — an added octave
elaborates its anchor exactly as a trill note does — and `presetMedium` puts 25
of them per 1000 notes, three times the real ornament rate.

### Holdouts

```bash
node scripts/corpus/generate.mjs data/corpus/orn-holdout.jsonl 4000 909 \
    --ornaments 1 --breadth 2 --add-rate 0.4 --imprecision natural --exaggerate early
node scripts/corpus/generate.mjs data/corpus/orn-shift-holdout.jsonl 4000 919 \
    --ornaments 1 --breadth 4 --add-rate 0.4 --imprecision early --exaggerate early
```

`orn-shift-holdout` is deliberately outside the training settings (`--breadth 4`)
to measure behaviour under distribution shift.

### Real ASAP scores, performed with their own ornaments

```bash
node scripts/corpus/generate_real.mjs data/corpus/asap-orn-train.jsonl \
    data/corpus/asap-spec-train.jsonl 301 --role train --takes 12 \
    --breadth 2 --add-rate 0.4 --imprecision natural --exaggerate early
node scripts/corpus/generate_real.mjs data/corpus/asap-orn-holdout.jsonl \
    data/corpus/asap-spec-train.jsonl 302 --role holdout --takes 4 \
    --breadth 2 --add-rate 0.4 --imprecision natural --exaggerate early
```

The spec file comes from `scripts/corpus/asap_spec.py`. 241 of 242 ASAP scores
carry `@id` on every `<note>` and partitura preserves it, so the raw-XML ornament
signs join the parsed score exactly.

## Early-recording corpus (v2 / `models/mlign-v2.pt`)

```bash
node scripts/corpus/generate.mjs data/corpus/early-a.jsonl 8000 21000
node scripts/corpus/generate.mjs data/corpus/early-b.jsonl 8000 22000
node scripts/corpus/generate.mjs data/corpus/early-holdout.jsonl 2000 99001
node scripts/corpus/generate.mjs data/corpus/shift-holdout.jsonl 2000 77003
```

Superseded for ornament work. Kept because `early-holdout` and `shift-holdout`
are the distribution v2 was built for, and a model that wins on the realistic
holdouts by forgetting these entirely is a different result from one that wins
on both — so both are reported (`docs/RESULTS.md` §6).

## Ground truth from real recordings

Not generated: derived. `scripts/corpus/real_orn_gt.py` reads Nakamura match
files and recovers 1527 real ornament groups (740 Batik, 787 ASAP) into
`data/corpus/realorn-{batik,asap}.jsonl`. ASAP and Batik put the ornament sign in
the **score note's attribute list**, with the played notes following as
`insertion-note` lines; the `ornament(Anchor,Type)` slot partitura provides is
empty in every corpus we have. Restrict ASAP to `robust_note_alignment` — 682 of
1063 match files are v5.0 with empty attribute lists — and rank candidates
time-first, then pitch, or a descending trill chain hands each figure's upper
neighbour the previous principal's pitch.

These labels are **partial**: an unattributed insertion means the derivation could
not resolve it, not that the note is ordinary, so such notes are ignored rather
than scored. The rows are deliberately not `mlign-*`, so training can never pick
them up (`src/mlign/dataset.py` gates the attribution loss on that prefix).

## Validation

```bash
.venv/bin/python scripts/corpus/validate.py data/corpus/<shard>.jsonl
```

Checkpoint selection uses `realgt-val.jsonl` and the alignment loss alone. A
synthetic validation set picks the wrong checkpoint — for alignment and, as it
turned out independently, for ornament attribution too (`docs/RESULTS.md` §5–6).
