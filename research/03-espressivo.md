# espressivo (meico-ts) — rendering API, note-ID provenance, and MPM feature coverage

**Subject:** `espressivo` v0.8.8, an idiomatic TypeScript port of the Java library
[meico](https://github.com/cemfi/meico) by Axel Berndt.
**Checkout studied:** `/Users/nielspfeffer/Projects/meico-ts`, branch `ts-idiomatic`, HEAD
`760d6ed` ("chore(infra): program complete").
**Sibling checkout:** `/Users/nielspfeffer/Projects/meico-ts-orn`, branch `ornamentation-v3`,
HEAD `6ac7aa6`, working tree dirty.
**Purpose of this study:** determine whether MLign can synthesise expressive MIDI performances
from MEI scores *with note-level ground-truth alignment* — every emitted MIDI note traced back to
its source MEI note `xml:id` — without forking the library.
**Date:** 2026-08-09. Everything below was read from source and, where marked **[verified]**,
confirmed by running the built `dist/` against the repository's own fixtures.

---

## 0. Executive answer

**Yes. Provenance is available without forking, through two independent channels, and they agree
exactly.**

1. **The MIDI file itself carries the ids.** `Msm.processScore` emits a MIDI **text meta event
   (`FF 01`) containing the note's `xml:id` immediately before every note-on, at the same tick**
   (`src/msm/Msm.ts:1320-1327`). This is not incidental — `src/midi/EventMaker.ts:670-674`
   documents it as the feature's purpose: *"`Msm.exportMidi` writes one of these before every
   noteOn, carrying the note's `xml:id`. That is what makes a rendered MIDI file traceable back
   to the MEI."*
2. **The facade returns the ids as data.** `PerformedNote.id` (`src/api/types.ts:97-112`) pairs
   the `xml:id` with `pitch`, symbolic `date`/`duration`, `velocity`, and performed
   `milliseconds.date` / `milliseconds.end`.

**[verified]** Across five MEI fixtures (`multi_part`, `repeats_endings`, `tuplets`,
`comprehensive`, `articulations`): **100 % of MIDI note-ons carried a text-event id (75/75), and
100 % matched a facade data note on `(id, pitch)` with `Math.round(milliseconds.date) == noteOn
tick` and `Math.round(milliseconds.end) == noteOff tick`.**

The one thing to get right is *how* you invoke the pipeline. See §3 for the canonical recipe;
the short version is **perform once, then derive both the labels and the MIDI from that single
augmented MSM** — because a second `perform` call re-rolls the imprecision randomness (§5).

---

## 1. Public API — the programmatic path from Node

### 1.1 Package and entry points

`package.json` declares `"name": "espressivo"`, `"type": "module"`, `"main": "dist/index.js"`,
`"types": "dist/index.d.ts"`, `engines: node >= 18.18`. ESM only. Dependencies:
`@xmldom/xmldom`, `xpath`, `uuid`. **No file I/O and no `process` access anywhere in the
library** — documents enter and leave as strings, so it runs in a browser too.

The `exports` map exposes only `"."`. `dist/` is present and current in this checkout. Note that
`mpmify` deep-imports `dist/api/index.js` past the exports map via an absolute path — that works
but is not a supported subpath; MLign should import the package root instead.

### 1.2 The seven facade functions

All in `src/api/pipeline.ts`, re-exported from `src/index.ts`. Everything crossing the boundary
is plain data (strings, numbers, `Uint8Array`, object literals) — `structuredClone`-safe,
JSON-round-trippable except the MIDI payloads, fresh value on every call.

| Function | In | Out |
| --- | --- | --- |
| `convertMeiToMsmMpm(mei, options?)` | MEI text | `readonly MovementDocuments[]` — one `{ index, title, msm, mpm }` per `mdiv` |
| `listPerformances(mpm)` | MPM text | `readonly PerformanceInfo[]` — `{ index, name, ppq }` |
| `performMsm({ msm, mpm }, options?)` | MSM + MPM text | augmented MSM text |
| `extractPerformanceData(augmentedMsm)` | augmented MSM text | `PerformanceData` |
| `performMsmToData({ msm, mpm }, options?)` | MSM + MPM text | `PerformanceData` |
| `renderMidi({ msm }, options?)` | MSM text | `Uint8Array` — score as written |
| `renderExpressiveMidi({ msm, mpm? }, options?)` | MSM (+ MPM) text | `Uint8Array` — as performed |

Exact signatures:

```ts
function convertMeiToMsmMpm(mei: XmlText, options?: ConvertOptions): readonly MovementDocuments[];
function performMsm(input: { readonly msm: XmlText; readonly mpm: XmlText },
                    options?: PerformOptions): XmlText;
function extractPerformanceData(augmentedMsm: XmlText): PerformanceData;
function performMsmToData(input: { readonly msm: XmlText; readonly mpm: XmlText },
                          options?: PerformOptions): PerformanceData;
function renderExpressiveMidi(input: { readonly msm: XmlText; readonly mpm?: XmlText },
                              options?: PerformOptions & MidiOptions): Uint8Array;
```

Options (`src/api/types.ts:33-95`):

- **`ConvertOptions`** — `ppq` (default 720, raised automatically if the source needs a finer
  grid), `dontUseChannel10` (default `true`), `ignoreExpansions` (default `false`), `cleanup`
  (default `true`), `sourceName` (drives the MPM `RelatedResource` URI *and* the generated
  `<comment>` text; omit for the file-less variant).
- **`PerformOptions`** — `performance` (name or 0-based index, default 0), `seed`,
  `movementSampleMaxStep` (default 0.1).
- **`MidiOptions`** — `generateProgramChanges` (default `true`); `renderMidi` also takes `bpm`
  (default 120).

### 1.3 Output shape

```ts
interface PerformedNote {
  readonly id: string | null;          // the note's xml:id, or null
  readonly pitch: Midi7Bit;
  readonly date: Ticks;                // SYMBOLIC MSM time
  readonly duration: Ticks;            // SYMBOLIC MSM duration
  readonly velocity: Midi7Bit;         // continuous float, NOT clamped — see §3.3
  readonly milliseconds: { readonly date: Milliseconds; readonly end: Milliseconds };
}
interface PerformedPart {
  readonly index: number; readonly name: string | null;
  readonly midiChannel: number | null; readonly midiPort: number | null;
  readonly notes: readonly PerformedNote[];
  readonly controlChanges: readonly ControlChangeStream[];
}
interface PerformanceData { readonly title: string; readonly ppq: Ticks;
                            readonly parts: readonly PerformedPart[]; }
```

`ControlChangeStream` is `{ kind: 'channelVolume' | 'position', controller, ccNumber, points }`
where `points` are `{ date, milliseconds, value }`. `channelVolume` is CC 7 (sub-note dynamics);
`position` is the pedalling stream, `sustain` → CC 64, `soft` → CC 67, anything else → 0.

### 1.4 Errors

The facade validates and throws typed errors, never returns `null`. `MeicoError` is the root;
subclasses `ParseError`, `EmptyDocumentError`, `PerformanceNotFoundError`, `InvalidOptionError`,
`MissingNodeError`. The *class* API underneath (`Mei`, `Msm`, `Mpm`, `Performance`, `Midi`)
behaves the Java way instead: it **logs to the console and returns `null`**.

### 1.5 Console noise

The interior logs progress unconditionally (`Rendering performance …`, `Processing global data.`,
`Performing part …`, `Performance rendering finished.`, plus `Converting … to MIDI.`). For batch
generation you will want to capture rather than discard `console.log`/`console.error` — a parse
warning or a defaulted-attribute notice arrives on the same channel, and mpmify learned to
whitelist the four known progress lines and report anything else
(`mpmify/ml/node/generate_v4.mjs:266-271`).

---

## 2. Note-ID provenance — the central question

### 2.1 The chain, end to end

```
MEI <note xml:id="n1">
  └─ Mei2MsmMpmConverter            copies xml:id onto the MSM <note>
MSM <note xml:id="n1" date=… duration=… midi.pitch=…>
  └─ Performance.perform            adds velocity, milliseconds.date, milliseconds.date.end
                                    IN PLACE on the same element — the id is never touched
augmented MSM <note xml:id="n1" … milliseconds.date="600" milliseconds.date.end="1455">
  ├─ extractPerformanceData         → PerformedNote.id = "n1"
  └─ Msm.processScore               → FF 01 text event "n1"  +  note-on  +  note-off
```

Nothing in `Performance.perform` rewrites or drops `xml:id`. The render passes only *add*
attributes (`date.perf`, `duration.perf`, `date.end.perf`, `velocity`, `milliseconds.date`,
`milliseconds.date.end`, `modified`, and the transient `ornament.*` markers).

### 2.2 The MIDI text-event channel, precisely

`src/msm/Msm.ts:1313-1341`, the expressive branch of `processScore`:

```ts
const xmlId = n.getAttribute('id', 'http://www.w3.org/XML/1998/namespace');
const textEvent = EventMaker.createTextEvent(date, xmlId === null ? 'unknown' : xmlId.getValue());
if (textEvent !== null) track.add(textEvent);
const noteOn = EventMaker.createNoteOn(chan, date, pitch, velocity);
if (noteOn !== null) track.add(noteOn);
```

Four properties make this a reliable extraction seam:

1. **Adjacency is guaranteed.** `Track.add` keeps the track sorted by tick and
   `Array.prototype.sort` is stable (ES2019+), so events added at the same tick keep insertion
   order (`src/midi/MidiTypes.ts:352-377`). The text event and its note-on are added
   consecutively with nothing between, so **no event can ever be interposed**. Note-offs from
   earlier notes and the control-change streams (added before `processScore`) sort *ahead* of
   the pair at an equal tick, never between it.
2. **One text event per note-on, always.** The only way to lose it is `createTextEvent`
   returning `null`, which happens only on an exception from the `MetaMessage` constructor.
3. **A note with no `xml:id` yields the literal string `'unknown'`** in the expressive branch
   (the symbolic branch yields `''` instead — an asymmetry inherited from `Msm.java:1000`).
   Treat `'unknown'` as a sentinel, not an id.
4. **The text is UTF-8 encoded** with the encoded byte length, not the character count
   (`EventMaker.ts:637-642`).

**Extraction rule for MLign:** parse the SMF; within each track, for each note-on, the
immediately preceding event at the same tick is an `FF 01` meta event whose payload is the
source MEI `xml:id`. Pair the note-on with the first following note-off of the same
`(channel, pitch)`.

**[verified]** on 75 notes across five fixtures: 75/75 note-ons had an id, 0 duplicates, 0
`'unknown'`.

### 2.3 The data channel, and the mapping between them

`performMsmToData` / `extractPerformanceData` give the same ids alongside richer values.
Differences from the MIDI, both of which matter:

- **Rounding.** MIDI ticks are `Math.round(milliseconds.date)`
  (`Msm.readMillisecondsDateFromElement`, `src/msm/Msm.ts:1804`) and
  `Math.round(milliseconds.date.end)`. The data path does **not** round — it hands you the raw
  double (`src/api/pipeline.ts:236-239` says so explicitly). So the data is *higher precision*
  than the MIDI, and MLign should train labels against the rounded values if the model consumes
  MIDI ticks.
- **Velocity clamping.** See §3.3 — the data velocity is pre-compression.

Because the tempo trick sets **1 MIDI tick = 1 millisecond** in expressive export
(`makeMillisecondTickTempo`, `Msm.ts:1191-1194`: tempo = `60000 / ppq` quarter-BPM), MIDI ticks
*are* milliseconds. No conversion needed.

### 2.4 Provenance survives repeats too — see §6.

### 2.5 Conclusion

**No fork is needed.** Both channels are first-class, documented, and load-bearing for the
library's own fixture comparisons (the reference `.mid` files contain these text events, so a
change to them would break the equivalence suite). The `PerformedNote.id` field is part of the
frozen facade contract.

---

## 3. The canonical recipe for MLign

### 3.1 Perform once; derive everything from one augmented MSM

```ts
import { convertMeiToMsmMpm, performMsm, extractPerformanceData,
         renderExpressiveMidi } from 'espressivo';

const [movement] = convertMeiToMsmMpm(meiText, { sourceName: 'piece.mei' });

// ONE performance realisation:
const augmented = performMsm({ msm: movement.msm, mpm: mpmText }, { seed, performance: 0 });

// Both artefacts derived from THAT SINGLE realisation:
const labels = extractPerformanceData(augmented);       // ids + ms + velocity
const midi   = renderExpressiveMidi({ msm: augmented }); // no mpm ⇒ renders as it stands
```

**Do not** call `performMsmToData(input, opts)` and `renderExpressiveMidi(input, opts)`
separately with the same seed and assume they agree. Those are two independent `perform` calls,
and imprecision maps re-roll unseeded randomness on each (§5). With no imprecision map in the
MPM the two paths do agree, but the single-perform recipe is correct unconditionally and costs
nothing.

`renderExpressiveMidi` with `mpm` omitted takes the no-performance path
(`Msm.exportExpressiveMidi()` → `renderMidi(83.33, true, true)`). It requires the MSM to already
carry `milliseconds.date` — which the augmented MSM does — and throws `EmptyDocumentError`
otherwise. One caveat: on that path **`generateProgramChanges` is hard-coded `true`** (Java
`Msm.java:667`, reproduced deliberately), and passing any `PerformOptions` field is an
`InvalidOptionError` rather than a silent no-op.

**[verified]** Rendering the same augmented MSM twice is **byte-identical**. On the `all_maps`
fixture (which includes two imprecision distributions), all 8 notes matched between
`extractPerformanceData` and the MIDI on rounded `ms.date` and `ms.end`, with zero velocity
mismatches.

### 3.2 The augmented MSM is a durable checkpoint

It is XML text and can be stored. It is the single source of truth for one performance
realisation: labels, MIDI, and any re-render all reproduce from it exactly. For a training
corpus this is the artefact worth persisting alongside the `.mid`.

### 3.3 Velocity: a real trap

`Msm.renderMidi` calls **`fitVelocities(0, 127)` before building any note event**
(`Msm.ts:1084`, implementation at `1613-1749`). It **rewrites the `velocity` attributes in
place** using a piecewise-linear compression with a 0.66 roll-off factor whenever any velocity
falls outside `[0, 127]`. `extractPerformanceData` reads velocities **before** that compression.

**[verified]** With an MPM forced to `volume="200"`, the data path reported velocities
`{211, 208, 201, 197, 203, 195}` while the MIDI carried `{127, 126, 125}`.

Mitigations, in order of preference: (a) keep MPM dynamics inside `[0, 127]` so the branch never
fires — `fitVelocities` returns early when nothing is out of range, so in-range data matches the
MIDI exactly after `Math.round`; (b) read velocity from the MIDI note-on rather than the data;
(c) re-parse the augmented MSM after rendering (the class API mutates it in place, but the
facade re-parses from text each call, so the caller's string is untouched — this does *not* work
through the facade).

Note also that MSM velocity is a **continuous double** (e.g. `32.111028088559394`), so
sub-integer dynamics differences are observable in the data but quantised away in the MIDI.

### 3.4 MIDI file structure

Track 0 is the global track (tempo, then global marker / time-signature / key-signature maps).
Each MSM part **with a `midi.channel`** gets its own track, in document order — so part order in
the MSM is part of the MIDI byte output. Within a part the event order is: port + channel-prefix
meta at tick 0, program change, track name, key/time-signature/marker maps, channel-volume CCs,
position (pedal) CCs, then the note events. Parts without `midi.channel` are silently skipped.

---

## 4. MPM feature coverage — what is implemented and what actually moves notes

All maps live in `src/mpm/elements/maps/`. The render order is fixed in
`Performance.renderPartSymbolic` (`src/mpm/elements/Performance.ts:668-718`) and
`renderPartMilliseconds` (`:744-778`), and the order is load-bearing.

**Symbolic (tick) domain, in this order:**

| Map | File | Effect |
| --- | --- | --- |
| `dynamicsMap` | `DynamicsMap.ts` | Writes `velocity` on every note; emits a new `channelVolumeMap` for sub-note dynamics (CC 7). Runs first because it reads symbolic dates. **With no dynamicsMap anywhere, every note gets `velocity="100.0"`.** |
| `movementMap` | `MovementMap.ts` | Does **not** touch the score. Builds and returns a new `positionMap` of sampled curve points → pedal CCs. |
| `metricalAccentuationMap` | `MetricalAccentuationMap.ts` | Adds `accentuation * scale` to each note's existing `velocity`. Must run after dynamics, and before rubato (which moves the dates the pattern is measured against). |
| `articulationMap` (pass 1) | `ArticulationMap.ts` | Tick-domain articulation: `absoluteDuration`, `relativeDuration`, `absoluteVelocity*`, `relativeVelocity`, `absoluteDelay`, detune. Can move onsets, hence the `map.sort()` afterwards. |
| `rubatoMap` | `RubatoMap.ts` | Warps `date.perf` / `date.end.perf` within a frame of `frameLength` ticks. **`loop` defaults to `false`, under which only the first frame is warped** and the rest of the span is silently unaffected. |
| `ornamentationMap` (pass 1+2) | `OrnamentationMap.ts` | Writes `ornament.*` markers, then folds the tick-domain ones into `velocity` / `date.perf` / `duration.perf` / `date.end.perf`. |

**The pivot:** `tempoMap` (`TempoMap.ts`) converts ticks → `milliseconds.date` /
`milliseconds.date.end` for every registered map. A transition from `bpm` to `transition.to`
bent by `meanTempoAt` has no closed form (the duration is the integral of 1/tempo), so it is
integrated with **Simpson's rule** — the only place besides `RubatoMap:336` that calls
`Math.pow`/`Math.log` on the render path. With no tempoMap the fallback is **1 tick = 1 ms**.

**Millisecond domain, in this order:**

| Map | File | Effect |
| --- | --- | --- |
| `asynchronyMap` | `AsynchronyMap.ts` | Shifts `milliseconds.date` (and end) of a whole part by a millisecond offset — the "plays ahead of / behind the beat" map. Applied to the score, the pedalMap, the channelVolumeMap and the positionMap. |
| `articulationMap` (pass 2) | `ArticulationMap.ts` | The `*Ms` modifiers: `absoluteDurationMs`, `absoluteDurationChangeMs`, `absoluteDelayMs`. |
| `ornamentationMap` (pass 3) | `Performance.ts:876-918` | Folds `ornament.milliseconds.date.offset`, `ornament.milliseconds.duration`, `ornament.noteoff.shift` into `milliseconds.date` / `.end`. **This copy, not `OrnamentationMap`'s own, is the one the pipeline runs.** |
| `imprecisionMap.timing` | `ImprecisionMap.ts` | Offsets `milliseconds.date`; separately offsets `milliseconds.date.end` via a deferred `pendingDurations` pass. Floored at 0. |
| `imprecisionMap.dynamics` | " | Offsets `velocity`. Unclamped here; clamped later by `fitVelocities`. |
| `imprecisionMap.toneduration` | " | Offsets `milliseconds.date.end` only. |
| `imprecisionMap.tuning` | " | Writes a `tuning.offset` attribute — **and nothing ever reads it.** |

### 4.1 `imprecisionMap.tuning` is inert for MIDI output

`grep -rn "tuning.offset" src/` returns only the two lines in `ImprecisionMap.ts` that *write*
it. The MIDI renderer emits no pitch-bend events; `processScore` reads only `midi.pitch`,
`velocity` and the two millisecond attributes. `PerformedNote` has no tuning field either. So
**tuning imprecision is invisible in both output channels.** If MLign wants microtonal
detuning it must add it downstream.

### 4.2 Two live defects inherited from mpmify's audit — **still present at HEAD `760d6ed`**

Both were reported by the mpmify program and I re-confirmed them against current source. Both
are **silent** — nothing warns.

- **E1 — literal articulations render as the identity.**
  `ArticulationMap.getArticulationDataOf` (`src/mpm/elements/maps/ArticulationMap.ts:93-121`)
  reads only `xml:id`, `noteid`, the style, and `name.ref`. It does **not** read the twelve
  numeric modifier attributes that `ArticulationMap.java` reads. Its own doc-comment states the
  design: *"Only the identifying fields are read here; the numeric modifiers are not. They live
  on the referenced `articulationDef`."* Consequence: an `<articulation>` carrying
  `relativeDuration` / `absoluteVelocityChange` **directly, without a `name.ref`**, does
  nothing. `ArticulationData`'s own XML constructor *does* parse them — it is simply not the
  path the map uses. (`addArticulation`, the serializer, also mentions them — a decoy.)
- **E2 — dynamics transitions ignore `curvature` and `protraction`.**
  `DynamicsMap.getDynamicsDataOf` (`src/mpm/elements/maps/DynamicsMap.ts:100-131`) reads
  `volume`, `transition.to` and `subNoteDynamics`, but never `curvature` or `protraction`. The
  Java version reads both. `DynamicsData.curvature/protraction` therefore stay `null` and get
  defaulted to `0.0` at first use (`DynamicsData.ts:121`). **Every dynamics transition renders
  on the default Bézier.**

**Why the library's byte-parity claim is not contradicted:** parity is proven against the
fixture corpus, and the MEI→MSM/MPM converter always emits `name.ref` articulations and never
emits `curvature`/`protraction`. Both defects live off the fixture-covered path. mpmify's
generator writes literal articulations and curved transitions directly into MPM, which is how it
found them: on a 60-piece pilot (7251 notes) it measured **velocity wrong on 4053 notes and
`milliseconds.date.end` wrong on 1590**; `milliseconds.date` was unaffected (tempo + rubato
exact).

**[verified] MEI-derived articulations do work.** On `articulations.mei` the first note renders
`velocity 95, ms 0→160` against a symbolic duration of 720 ticks — a real staccato. So MLign is
unaffected *as long as its MPMs use `name.ref` articulations and uncurved dynamics
transitions*. If MLign's MPM sampler writes literal articulation modifiers or curved dynamics,
it must either avoid them or render through the Java fork.

### 4.3 Movement-map quirks worth knowing (from mpmify's CANONICAL.md, and consistent with source)

- `MovementData.getTForDate` terminates when the Bézier's x-error is `< 1.0` **tick**, so on an
  `L`-tick segment the returned position carries systematic error up to `127/L` CC units. Below
  ~180 ticks `curvature`/`protraction` stop meaning anything.
- The CC stream is a zero-order hold at `movementSampleMaxStep`-spaced samples. Going 0.1 → 0.02
  buys at most 0.8 CC for 4.5× the events and 4.5× the (quadratic) sampling time. Keep 0.1.
- `getMovementSegment` emits **duplicate points**; a constant element emits three identical
  points at its own date and nothing after, so the held value between elements is only
  recoverable under **last-wins** semantics.
- A global `movementMap` is rendered **once per MSM part**, each copy shifted by that part's
  asynchrony offset.

---

## 5. Randomness and determinism

### 5.1 Where randomness lives

`RandomNumberProvider` (`src/supplementary/RandomNumberProvider.ts`) is a **Mulberry32** PRNG,
seeded at construction from `Math.floor(Math.random() * 2147483647) || 1` (line 52) unless
`setSeed()` pins it. It is a *deterministic sequence, not independent samples*: correlated
distributions (brownian noise, compensating triangle) derive each value from the previous one,
and `getValue(index)` advances the state. The class doc is emphatic that **the number and order
of `getValue` calls is part of the output**.

Seed resolution (`ImprecisionMap.ts:349-354`):

```ts
if (dd.seed !== null) random.setSeed(dd.seed);                        // MPM seed attribute wins
else if (ctx?.options.seed !== undefined)
  random.setSeed(deriveSeed(ctx.options.seed, ordinal, impIndex));    // PerformOptions.seed
// else: keep the constructor's Math.random() seed
```

`deriveSeed` (`src/mpm/RenderOptions.ts:66-73`) folds `(base, streamOrdinal, distributionIndex)`
with `Math.imul(h ^ p, 0x27d4eb2d)`, never returning 0. `streamOrdinal` is a per-render
monotonic counter on `RenderContext`, incremented once per `renderImprecisionToMap` call — so
the pair `(ordinal, impIndex)` is unique per provider, and reproducible because the call order
is fixed for identical input.

### 5.2 Three unseeded `Math.random()` sites defeat it

1. **`doHandover`** (`ImprecisionMap.ts:526-534`) — when a correlated distribution has no
   predecessor, its starting value is drawn with a bare `Math.random()`.
2. **`shakeOffsets` / `shakeTimingOffsets`** (`:536-595`) — when two or more offsets land on the
   *same* `milliseconds.date`, one is chosen to keep its value by
   `Math.floor(Math.random() * entries.length)` and the rest are re-rolled.
3. **`shake`** (`:611-624`) — the re-roll itself builds a **fresh `RandomNumberProvider` with no
   `setSeed()`**, so it starts from a `Math.random()` seed.

This is faithful to `ImprecisionMap.java:845,894` and is deliberate. `PARITY.md §4` states it as
a charter rule: *"Never add byte comparison for imprecision output."*

### 5.3 How bad it is in practice — worse than the docs suggest

The README frames this as "reproducible only while no two offsets share a date, which for
polyphonic input is often false." **For the timing domain it is worse: monophonic input
collides too.**

For `imprecisionMap.timing`, `offsets` is keyed by millisecond date and receives **two kinds of
entry**: each note's onset (keyed by its `milliseconds.date`) *and*, from the deferred
`pendingDurations` pass, each note's end (keyed by its `milliseconds.date.end`,
`ImprecisionMap.ts:454-466`). In any legato or contiguous score, note *k*'s end equals note
*k+1*'s onset — **a collision on every note boundary**, which fires the shake.

**[verified]** The `imprecision_timing` fixture is **monophonic, 1 part, 8 notes, with an
explicit `seed="42"` on its `distribution.uniform`, and zero symbolic date collisions.** Across
six runs at `{ seed: 42 }`: **7 of 8 notes varied, spread up to 7.97 ms.**

Contrast: `imprecision_dynamics` and `all_maps` **were** fully reproducible at a fixed seed
(the dynamics domain keys only on onsets, so no boundary collisions arise).

### 5.4 What this means for MLign

- **Never rely on `seed` to reproduce a performance.** Treat each `perform` call as drawing a
  fresh sample.
- **The single-perform recipe (§3.1) makes this a non-issue** — you keep the realisation you
  drew, as an augmented MSM, and everything downstream is deterministic from it. Two renders of
  the same augmented MSM are byte-identical **[verified]**.
- If you need bit-reproducible *sampling* as well, persist the augmented MSM, not the seed.
- `meico_<uuid>` identifiers (from `Mei.addIds()` / `Msm.addIds()` / ornament generation) are
  random per run by construction. **Stamp ids into the MEI once, persist that MEI, and never
  regenerate** — otherwise your labels key on ids that change between runs. `src/xml/ids.ts`
  additionally warns that `addUUID` is order-sensitive: changing how many are drawn, or in what
  order, changes canonicalised output.

---

## 6. Repetitions

### 6.1 The facade does not expand repeats

**[verified]** `convertMeiToMsmMpm` on `repeats_endings.mei` produces an MSM with **4 `<goto>`
elements in the global `sequencingMap` and 12 `<note>` elements** — the score as written, with
the repeat structure left symbolic. Nothing in `src/api/` or `Mei2MsmMpmConverter` calls
`resolveSequencingMaps`; `grep -rn "resolveSequencingMaps" src/` finds only the doc comment and
the `index.ts` export. `Performance.perform` registers the `sequencingMap` for timing processing
but does not expand it, and the MIDI renderer plays the map straight through.

**So by default, repeats are NOT played.** If MLign wants repeats in the audio, it must expand
them explicitly through the class API before performing:

```ts
import { Msm } from 'espressivo';
const msmObj = new Msm(movement.msm);
const idChain = msmObj.resolveSequencingMaps();   // Map<string,string>
const expandedMsm = msmObj.getRootElement().toXML();
```

**[verified]** This took the fixture from **12 to 22 notes**, returned a 10-entry id chain, and
`performMsmToData` on the expanded MSM produced 22 performed notes.

### 6.2 How duplicated notes' ids are formed — and why this is good news

`Msm.applySequencingMapToMap` (`src/msm/Msm.ts:821-947`) tags the *original* element with a
temporary `repetitionCounter` and gives copy *n* the id:

```
meico_repetition_<n>_<baseId>
```

The base id is read from the copy, and the original element's id is never rewritten — so
**the prefix is flat, never nested**.

**[verified]** ids produced: `meico_repetition_1_n5`, `meico_repetition_1_n6`,
`meico_repetition_2_n5`, `meico_repetition_2_n6`, …; no double prefix anywhere; base ids
recovered perfectly by `id.replace(/^meico_repetition_(\d+)_/, '')`, yielding `n5, n6, n7, n8,
n9, n10`.

**This is the ideal shape for alignment ground truth:** one regex recovers the source MEI id,
and the capture group gives the repetition pass number for free.

`resolveSequencingMaps` also returns a `Map<string,string>` which is a **chain, not a lookup
table**: `base → rep1 → rep2 → …`. To reach the *n*-th copy you follow it *n−1* steps. That
chain exists for `updateMpmNoteidsAfterResolvingRepetitions`
(`src/mei/mpmNoteIds.ts`), which repairs MPM `noteid` references after expansion — it iterates
each map element with a matching `noteid` from index 1 and steps the chain once per element.
**If you expand repeats, call it on any MPM map that carries `noteid` references** (articulation
and ornamentation maps do), or those references will all point at the first occurrence.

For MLign's purposes the regex is simpler and sufficient; the chain matters only if you also
need to rewrite MPM.

---

## 7. MEI coverage

The converter is `src/mei/Mei2MsmMpmConverter.ts` (4710 lines). Dispatch is a table
`ELEMENT_HANDLERS` (`:290-600`) mapping each MEI element name to a handler and a traversal
verdict (`'done'` / `'descend'` / ignore). Unknown elements throw rather than being skipped
(`getLocalName()` strips namespaces, so foreign-namespace content is reachable).

Preprocessing (`:237-241`): snapshot the document if `cleanup`, resolve `copyof`/`sameas`,
drop `rend`, resolve `expansion`s unless `ignoreExpansions`.

**[verified] behaviour on hand-built minimal MEI:**

| Construct | Result |
| --- | --- |
| `<chord>` | Every member note emitted separately, all with their own `xml:id`, same date/duration. |
| Two `<layer>`s in one `<staff>` | **Merged into a single MSM part.** Layers do *not* become separate parts or channels. 3 notes, 1 part. |
| Two `<staff>`s | **Two MSM parts**, channels 0 and 1. Staves are the part unit. |
| `<tuplet>` | Handled; durations scaled correctly (20 notes, 240-tick triplets at ppq 720). `tupletSpan` handled separately by date+layer match. A `tuplet` missing `num`/`numbase` makes the whole duration 0 — Java does the same. |
| Grace `<note>` (standalone) | **Emitted with `duration = 0`** (`computeDuration`, `:3908`: `if (ofThis.getAttribute('grace') !== null) return 0.0`). Renders as a **zero-length MIDI note** (note-on and note-off at the same tick). |
| Grace `<chord>` | **Dropped entirely** — `ELEMENT_HANDLERS.chord` returns `'done'` without descending when `grace` is present (`:379-381`). Its notes never reach the MSM. |
| `<tie>` across bars | **Merged into one MSM note** with the summed duration, keeping the **first** note's `xml:id`. The terminal note's id disappears from the output. |
| `<arpeg>` | Converted to an MPM ornament. Notes keep their ids; onsets are spread (measured −22 / 0 / +22 ms). Arpeggio note order is deferred to `arpeggiosToSort` and resolved by pitch once all pitches are known. |
| `<trill>`, `<mordent>`, `<turn>` | **`IGNORE`.** The converter's own comment (`:321-323`): *"meico does not read ornaments from MEI (Java carries a TODO saying so). MPM ornamentation reaches the output through `arpeg` and through MPM styles, not through these."* No extra notes, no timing change. |
| Note with no `xml:id` | `PerformedNote.id === null`; MIDI text event says `'unknown'`. |

Also handled: `slur`, `tie`, `artic`, `dynam`, `hairpin`, `tempo`, `pedal`, `octave`, `bTrem`,
`fTrem` (routed to `processChord`), `mRest`, `multiRest`, `space`, `mSpace`, `halfmRpt`,
`staffDef`/`layerDef` defaults, `startid`/`endid`/`tstamp2`/`plist` reference resolution,
transposition and `oct` write-back.

### 7.1 Stamping ids on an id-less MEI

`Mei.addIds()` adds `meico_<uuid>` `xml:id`s to elements lacking them. **[verified]** it added 5
ids to a 2-note fragment and those ids flowed through to `PerformedNote.id`. But they are
**random per run** — so run it once, persist the resulting MEI, and use that as your corpus
input. (`Msm.addIds()` also exists but is not on the pipeline path.)

### 7.2 Implication for MLign's alignment representation

Three MEI constructs break a naive 1:1 score-note ↔ MIDI-note bijection, and the representation
must accommodate them:

- **Ties** collapse *n* MEI notes into 1 MIDI note (only the first id survives) — a
  many-score-notes-to-one-performance-note mapping.
- **Grace chords** delete notes entirely — score notes with no performance counterpart.
- **Grace notes** produce zero-duration MIDI notes, which many MIDI parsers and most models will
  either drop or choke on.
- **Repeats** (if expanded) map one score note to *n* performance notes — the inverse direction.

---

## 8. Ornamentation — current state and the `-orn` branch

### 8.1 In `meico-ts` (the branch MLign would use today): ornaments never create notes

`OrnamentData.apply()` returns an empty array unconditionally, and
`src/mpm/elements/maps/data/OrnamentData.ts:76` says so: *"The return value is **always an empty
array**, and the Java reference is the same (`OrnamentData.java`, where a TODO marks the spot).
It is the seam for a feature that does not exist yet … the `for (const chord of od.apply(...))`
loop in `OrnamentationMap.apply` is dead by construction."*

What v2 ornamentation *does* is write `ornament.*` markers onto notes that already exist, which
later passes fold into timing and dynamics: `ornament.dynamics`, `ornament.date.offset`,
`ornament.duration`, `ornament.milliseconds.date.offset`, `ornament.milliseconds.duration`,
`ornament.noteoff.shift`.

**[verified]** The `ornamentation` fixture goes **12 notes → 12 notes** through `performMsm`;
the augmented MSM contains exactly `ornament.dynamics`, `ornament.date.offset`,
`ornament.milliseconds.date.offset`, `ornament.noteoff.shift`; all 12 ids remain unique.

**So today: no ornament renders into extra MIDI notes.** Arpeggios spread onsets; trills,
mordents and turns do nothing at all (and are not even read from MEI — §7).

This is *good* for MLign v1: the score↔performance mapping stays a clean bijection modulo the
tie/grace/repeat cases in §7.2.

### 8.2 In `meico-ts-orn` (`ornamentation-v3`): ornaments DO create notes

The branch is a second worktree of the same repo, 17 commits past the merge base `a09f82c`, with
its two largest files (`ornamentInstantiation.ts`, 1117 lines, and its 1030-line test)
**untracked** and four more files modified-not-committed. It is work in progress.

New modules: `ornamentInstantiation.ts` (the note-generating renderer),
`ornamentExpansion.ts` (581 lines, the pure expansion engine), `noteOrder.ts` (298 lines, the v3
`note.order` grammar with chords `[ … ]` and repeat groups `|: … :|`), `OrnamentNote.ts`,
`TemporalValue.ts` (unit-suffixed values: `12ticks`, `50%`, `300ms`). `OrnamentationMap.ts` grows
466 → 714 lines.

`src/mei/Mei2MsmMpmConverter.ts`, `src/msm/Msm.ts` and **all of `src/api/`** are byte-identical
between the trees — the MEI importer emits no v3 ornaments and the facade has no ornament fields
yet.

**Generated notes carry partial provenance.** `createNote`
(`meico-ts-orn/src/mpm/elements/maps/ornamentInstantiation.ts:877-905`) clones the principal,
strips a `NOT_INHERITED` list (only `velocity` and `modified` survive), then:

```ts
addUUID(note);                                                    // its own random meico_<uuid>
note.addAttribute(new Attribute('ornament.generated', 'true'));
if (ornament.ornamentId !== null)
  note.addAttribute(new Attribute('ornament.ref', ornament.ornamentId));  // the MPM <ornament>, not the note
```

Crucially, `assignPrincipalId` (`:925`) takes the principal's original `xml:id` and **overwrites
it onto exactly one generated note** (preferring the one whose source is the principal, then a
same-pitch note, then the first). The principal element itself is then **carved out of the map
entirely** (`carve`, `:960`, ending in `owner.removeElement(principal)`) unless a tick-domain
`alignment="at end"` frame leaves an audible head.

**The gap:** there is no `ornament.parent` or `sourceId` attribute. If the expansion never
sources a note from the principal and the principal is carved away, **the original principal id
lands on an arbitrary heir or vanishes**. The branch's own journal records this
(`meico-ts-orn/ornamentation/LOG.md:1115`, "D10/D15 addendum — `ornament.anchor` (MLign join
guarantee)") and rules that every generated note should additionally carry
`ornament.anchor="<original principal note id>"`, exposed by a future W7 as
`PerformedNote.ornamentAnchor`. **That is a decision on paper — it does not exist in code.**
Grep for `ornament.anchor`, `ornament.source`, `ornament.slot`, `ornament.pass` in
`meico-ts-orn/src/` returns nothing.

Also journaled there and directly relevant to MLign:

- Expansion is **RNG-free and bit-deterministic**, but **generated `xml:id`s are random per
  run** — so supervision must key on provenance attributes plus `(part, date, pitch, slot)`,
  **never on generated ids**.
- v3 ornament ground truth would be **espressivo-only**: upstream Java is unimplemented, and
  meico PR #31 is described as defective — *do not validate against it*.
- MPM carries **no version marker**, so an exporter must pick a generation per document.
- Generated notes break the 1:1 bijection, so the schema must carry performance-only notes with
  a `generated` flag — the same machinery real-data insertions need.

Neither tree's `package.json` version differs (both `0.8.8`), and neither `README.md` nor
`PARITY.md` documents v3 ornamentation. `-orn` also *lacks* several `ts-idiomatic` fixes (the
TD2 namespace-typo acceptance, the TD4 `Attribute.detach` fix, `parseJavaDouble`), so it is not
simply "main plus ornaments".

**Recommendation:** build MLign v1 on `meico-ts` (`ts-idiomatic`). Treat v3 ornamentation as a
later corpus generation, and if you adopt it, get `ornament.anchor` implemented first — without
it the score-side join is not total.

---

## 9. How mpmify drives espressivo, and what it learned

`/Users/nielspfeffer/Projects/mpmify/ml/node/` contains six dependency-free ESM scripts.

**API surface used: exactly one function, `performMsmToData`.** No `convertMeiToMsmMpm`, no
`performMsm`, no `renderMidi`/`renderExpressiveMidi`, no class API.

```js
// generate_v4.mjs:510-513
const { performMsmToData } = await import(ESPRESSIVO);
const captured = captureConsole(() =>
  pieces.map((p) => performMsmToData(p.docs, { movementSampleMaxStep: opt.movementMaxStep })));

// verify_v4.mjs:281
const ts = performMsmToData({ msm, mpm });          // no options at all
```

`paths.mjs:4` hard-codes `ESPRESSIVO =
'/Users/nielspfeffer/Projects/meico-ts/dist/api/index.js'` — an absolute path to a built file,
deep-imported past the `exports` map. **The dependency is declared nowhere**: mpmify's
`package.json` does not list espressivo, and `ml/node/package.json` says so explicitly (*"No npm
dependencies: espressivo is imported from an absolute dist path"*). Provenance is recorded as a
*state*, not a commit ("meico-ts HEAD 415bbd2 + uncommitted changes, dist/ built 2026-08-09
11:29").

**No MEI anywhere.** MSM and MPM are synthesised directly as XML strings by `xml.mjs`. **No MIDI
is ever rendered** by either the TS or the Java path.

**Provenance handling:** `xml.mjs:81` mints positional ids `p<partNumber>n<index>`;
`performMsmToData` surfaces them as `PerformedNote.id`; `verify_v4.mjs:201` compares them
bit-exactly as a cross-renderer alignment key — **but `generate_v4.mjs:354-366` drops the id from
the emitted JSONL**, which carries only `[date, duration, pitch, msOn, msOff, velocity, part]`.
So mpmify proves the id channel works and then discards it. `augmented_msm.mjs` is a hand-rolled
regex reader that parses meico-Java's augmented MSM into the exact shape `performMsmToData`
returns — a deliberate independent reimplementation of `extractPerformanceData`, used to compare
the two renderers. It reads `xml:id` into `note.id` but builds **no index and no alignment
structure**.

**Determinism strategy:** avoid the problem. `KNOWN_MAPS` is `['tempo', 'dynamics',
'articulation', 'rubato', 'asynchrony', 'movement']` — **no imprecision map is ever sampled**, so
`PerformOptions.seed` never comes into play and the render is deterministic by construction.
Sampling-side randomness uses `java_random.mjs`, a bit-exact port of `java.util.Random`'s 48-bit
LCG, so the TS sampler can be proven faithful to the Java one by diffing the JSONL.

**The tolerance model in `verify_v4.mjs` is the most transferable lesson.** macOS libm differs
from Java's fdlibm by 1 ULP on ~10 % of `pow`/`log` arguments. The ULP budget is *derived per
piece* (`4 * (1 + #tempoInstructions + rubato?)`) because a millisecond date is an accumulated
sum of Simpson integrals, each with its own `Math.pow`; a fixed 2-ULP envelope was "seed-lucky"
and failed. Only four field classes get any envelope (`note.ms.date`, `note.ms.end`,
`cc.position.ms`, `cc.channelVolume.ms`); everything else must be bit-exact. The magnitude floor
`ABS_TOLERANCE_MS = 1e-6` sits in a *measured* gap: libm divergences top out at 3.64e-12 ms while
the smallest genuine logic divergence is 0.0167 ms. A `--probe-no-pow` control strips every
transcendental call from the render path to earn the attribution.

**Known-quirks list carried by CANONICAL.md**, all inherited by espressivo:

- **R8 / NaN hazard.** `TempoMap.renderTempoToMap` picks a note's tempo segment from the note's
  **unwarped** key but evaluates the formula on `date.perf`, which `RubatoMap` has already
  warped without touching the key. With `intensity > 1` the warped date can fall *before* the
  segment start and `Math.pow(negative, non-integral)` yields **NaN milliseconds**. Measured:
  24/2485 notes shifted by up to 316 ms, 4 notes NaN.
- `rubato loop` defaults to `false` → only the first frame warps.
- A dangling final tempo transition renders inert; `transition.to == bpm` is rewritten to
  constant; `meanTempoAt <= 0` → constant at target, `>= 1` → constant at start.
- A missing date-0 instruction falls back to silent defaults: 100 bpm, velocity 100.
- Sim2real: purely-synthetic v1 did not transfer to real Vienna 4x22 performances (Chopin
  op. 10/3 at ~31 qBPM, *below* the sampler's [40, 200] range, gave 8990 ms median render RMSE
  against a ~404 ms constant baseline). Maps stayed well-formed; the values were wrong. Domain
  randomisation is the fix they adopted.

**A coordination note already exists for MLign** (`mpmify/ml/LOG.md:198-206`): the agreed
division is that MLign owns the performer-error / repeat / rolled-chord injection layer and
mpmify owns the MPM map samplers, with one shared generator; the robustness layer is to be a
pure function `(notes, msm, rng, config) → edited notes + typed edit-log
(delete/insert/substitute/shift)`, the edit-log doubling as alignment ground truth and as
mpmify's unmatched-note training signal.

---

## 10. Risks and recommendations for MLign

**Do this:**

1. **Stamp `xml:id`s into the MEI corpus once** (`Mei.addIds()` or your own), persist, never
   regenerate — generated ids are random per run.
2. **Use the single-perform recipe** (§3.1): `performMsm` → augmented MSM → both
   `extractPerformanceData` and `renderExpressiveMidi({ msm: augmented })`. Persist the
   augmented MSM as the realisation record.
3. **Read provenance from the MIDI text events** as the primary channel (it is what a consumer
   of the `.mid` alone can do, and it validates the data channel), cross-checking against
   `PerformedNote.id`.
4. **Keep MPM dynamics within `[0, 127]`** so `fitVelocities` never fires, or take velocity from
   the MIDI.
5. **Decide repeats explicitly.** They are not expanded by default. If you expand, recover base
   ids with `/^meico_repetition_(\d+)_/` and call
   `updateMpmNoteidsAfterResolvingRepetitions` if your MPM carries `noteid` references.
6. **Model the non-bijective cases** from the start: ties (n→1, first id survives), grace chords
   (dropped), grace notes (zero-duration), repeats (1→n). This is the same machinery the
   performer-error injection layer needs.

**Watch out for:**

- **`imprecisionMap.timing` is not reproducible even at a fixed seed, even monophonically**
  (§5.3) — the note-boundary collision fires the unseeded shake on essentially every note.
- **E1/E2** (§4.2): literal articulation modifiers and dynamics `curvature`/`protraction` are
  silently ignored. Use `name.ref` articulations and uncurved transitions, or render through the
  Java fork.
- **`imprecisionMap.tuning` is inert** — written, never read, no pitch bend (§4.1).
- **Trills/mordents/turns in MEI are ignored** (§7) and **v2 ornaments never generate notes**
  (§8.1). Ornament-rich ground truth needs the `-orn` branch, which needs `ornament.anchor`
  implemented first (§8.2).
- **Zero-duration grace notes** will produce degenerate MIDI notes; decide whether to filter,
  lengthen, or model them.
- **Parts come from staves, not layers** — a two-voice piano staff is one MIDI channel, so
  voice separation is not available from the part structure.
- **The library logs unconditionally**; capture stdout/stderr in batch runs rather than
  discarding it.
- **`espressivo` is not published to npm** and its licence field is pending (it is GPL v3
  derivative work). Consume it via a local `file:` dependency and resolve licensing before any
  distribution.

---

## Appendix — key file map

| Concern | File |
| --- | --- |
| Facade | `src/api/pipeline.ts`, `src/api/types.ts`, `src/api/errors.ts` |
| Public exports | `src/index.ts` |
| MEI → MSM/MPM | `src/mei/Mei2MsmMpmConverter.ts` (4710 lines), `src/mei/Mei.ts` |
| MPM noteid repair after repeats | `src/mei/mpmNoteIds.ts` |
| MSM model, repeats, MIDI render | `src/msm/Msm.ts` (`processScore` :1293, `applySequencingMapToMap` :821, `fitVelocities` :1613) |
| Render orchestration | `src/mpm/elements/Performance.ts` (`perform` :404, `renderPartSymbolic` :668, `renderPartMilliseconds` :744) |
| Render knobs / seed derivation | `src/mpm/RenderOptions.ts` |
| MPM maps | `src/mpm/elements/maps/*.ts` |
| Randomness | `src/mpm/elements/maps/ImprecisionMap.ts`, `src/supplementary/RandomNumberProvider.ts` |
| MIDI event construction | `src/midi/EventMaker.ts` (`createTextEvent` :676), `src/midi/MidiTypes.ts` (`Track.add` :373) |
| id generation | `src/xml/ids.ts` |
| Deliberate Java divergences | `PARITY.md` (§4 = nondeterminism) |
