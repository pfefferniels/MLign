/**
 * Synthetic training-corpus generator v0.
 *
 * Composition (the score/MPM samplers live in the fenby repo):
 *   score + MPM-map sampling — fenby's modules (imported read-only);
 *   rendering               — espressivo facade (performMsmToData, once);
 *   robustness + GT         — our layer.
 *
 * Usage:
 *   node scripts/corpus/generate.mjs <out.jsonl> <numPieces> <seed>
 *        [--robustness none|light|medium|heavy] [--jitter <stdMs>]
 *        [--ornaments <rate>]        per-eligible-note ornament probability
 *        [--breadth <f>]             ≥1, widens sampled <temporalSpread>
 *        [--exaggerate [modern|early]]
 *        [--imprecision subtle|natural|early]   MPM imprecisionMap humanisation
 *
 * The `early` settings target pre-WWII recorded style — broad arpeggiation,
 * free tempo, heavy ornamentation — which is the repertoire MLign is for, and
 * which the modern-calibrated defaults under-represent.
 *
 * Output: notes/corpus-format.md rows, one per line. Pieces whose GT fails an
 * invariant are dropped (counted in the final summary line on stderr).
 */

import { appendFileSync, openSync, closeSync, writeSync } from 'node:fs';
// The score/MPM samplers moved out of mpmify into their own repo (mpmify
// f79697e "Remove ml/: fenby is its own repository now"); mpmify/ml/node/ is
// gone, and these paths followed it there.
import { JavaRandom } from '/Users/nielspfeffer/Projects/fenby/node/java_random.mjs';
import {
  samplePieceV4,
  documentsFor,
  captureConsole,
} from '/Users/nielspfeffer/Projects/fenby/node/generate_v4.mjs';
import { performMsmToData } from '/Users/nielspfeffer/Projects/meico-ts/dist/api/index.js';
import {
  applyRobustness,
  presetLight,
  presetMedium,
  presetHeavy,
  mergeConfig,
} from '../../src/robustness/robustness.mjs';
import { editsToAlignment, shiftToMatchedZero } from '../../src/robustness/gt.mjs';

const PRESETS = { none: {}, light: presetLight, medium: presetMedium, heavy: presetHeavy };

function parseArgs(argv) {
  const pos = [];
  const opt = {
    robustness: 'medium', jitter: 12, ornaments: 0,
    exaggerate: false, profile: 'modern', breadth: 1, imprecision: '',
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--robustness') opt.robustness = argv[++i];
    else if (a === '--jitter') opt.jitter = Number(argv[++i]);
    else if (a === '--ornaments') opt.ornaments = Number(argv[++i]);
    // --exaggerate takes an optional profile name: `--exaggerate early`.
    else if (a === '--exaggerate') {
      opt.exaggerate = true;
      if (argv[i + 1] && !argv[i + 1].startsWith('--')) opt.profile = argv[++i];
    } else if (a === '--breadth') opt.breadth = Number(argv[++i]);
    else if (a === '--imprecision') opt.imprecision = argv[++i];
    else pos.push(a);
  }
  if (pos.length !== 3) {
    throw new Error(
      'usage: generate.mjs <out.jsonl> <numPieces> <seed> [--robustness p] [--jitter ms]\n' +
        '       [--ornaments rate] [--exaggerate [modern|early]] [--breadth f]\n' +
        '       [--imprecision subtle|natural|early]',
    );
  }
  if (opt.exaggerate && !EXAG_PROFILES[opt.profile]) {
    throw new Error(`unknown exaggeration profile: ${opt.profile}`);
  }
  if (opt.imprecision && !IMPRECISION_LEVELS[opt.imprecision]) {
    throw new Error(`unknown imprecision level: ${opt.imprecision}`);
  }
  return { out: pos[0], n: Number(pos[1]), seed: BigInt(pos[2]), ...opt };
}

const WANT = { dynamics: true, articulation: true, rubato: true, asynchrony: true, movement: true, accentuation: false };
const OPT = { twoPartProb: 0.5, asynchronyProb: 1.0, movementProb: 0.5 };

// ---------------------------------------------------------------------------
// Ornament sampling (MPM v3, meico-ts 05147ed). Injected into the built MPM
// text: a styleDef trio (trill/mordent/turn) in global/header +
// an ornamentationMap in global/dated. Pool-note figures per the v3 fixture
// syntax (note.order referencing pool notes + the principal by noteid).
// ---------------------------------------------------------------------------

// frameLength ranges per ornament kind, as a percentage of the principal's
// duration. `breadth` biases the draw toward the wide end without ever leaving
// the range (pow(u, 1/breadth)), so a wider setting cannot produce a figure
// that overruns what espressivo can measure against the principal.
const SPREAD_PCT = { trill: [55, 100], mordent: [12, 45], turn: [25, 75] };

/** One sampled <temporalSpread> — the axis that was previously a constant. */
function sampleSpread(kind, rng, breadth) {
  const [lo, hi] = SPREAD_PCT[kind];
  const length = lo + (hi - lo) * Math.pow(rng.nextDouble(), 1 / breadth);
  // Placement. Early pianists routinely begin the figure before the notated
  // onset, so a share of ornaments is anticipated. The offset is in ticks
  // against a percentage frameLength, which is the MPM spec's own fig.-3
  // combination (meico-ts ornamentInstantiation.ts:473).
  const anticipated = rng.nextDouble() < 0.15 + 0.2 * (breadth - 1);
  const offset = anticipated ? -Math.round(rng.nextDouble() * 180 * breadth) : 0;
  const shift = rng.nextDouble() < 0.8 ? ' noteoff.shift="monophonic"' : '';
  return (
    `<temporalSpread frame.offset="${offset.toFixed(1)}ticks"` +
    ` frameLength="${length.toFixed(0)}%"${shift} />`
  );
}

/**
 * Sample ornaments for a piece: candidates = part-1 notes with duration ≥ one
 * quarter, spaced ≥ one quarter apart; each gets trill/mordent/turn.
 * rateP = probability per candidate; `breadth` ≥ 1 widens the temporal spread
 * toward early-recording style.
 *
 * Every ornament gets its OWN ornamentDef, so `<temporalSpread>` can vary per
 * instance. Sharing three fixed defs (the previous shape) left arpeggiation
 * breadth and ornament placement as constants across the whole corpus —
 * precisely the axis this repertoire exercises hardest.
 *
 * Returns `{ header, map }`, both '' when nothing was sampled.
 */
export function sampleOrnaments(piece, rng, rateP, breadth = 1) {
  const part = piece.parts[0];
  const entries = [];
  const defs = [];
  let lastDate = -1e9;
  part.notes.forEach((n, i) => {
    if (n.dur < 720 || n.date - lastDate < 720) return;
    if (rng.nextDouble() >= rateP) return;
    lastDate = n.date;
    const id = `p${part.number}n${i}`;
    const kind = ['trill', 'mordent', 'turn'][rng.nextInt(3)];
    const upper = 1 + rng.nextInt(2); // chromatic 1..2 ≈ diatonic neighbor
    let poolNotes;
    let order;
    if (kind === 'trill') {
      // 2..6 alternation pairs: a broad early-recording trill runs longer
      // than the 2..4 the corpus had.
      const reps = 2 + rng.nextInt(Math.round(3 + 2 * (breadth - 1)));
      poolNotes = `<note xml:id="u" interval.chromatic="${upper}.0" />`;
      order = Array.from({ length: reps }, () => '#u ' + `#${id}`).join(' ');
    } else if (kind === 'mordent') {
      poolNotes = `<note xml:id="u" interval.chromatic="${upper}.0" />`;
      order = `#${id} #u #${id}`;
    } else {
      poolNotes =
        `<note xml:id="u" interval.chromatic="${upper}.0" />` +
        `<note xml:id="l" interval.chromatic="-${upper}.0" />`;
      order = `#u #${id} #l #${id}`;
    }
    const defName = `orn${i}`;
    const alignment = kind === 'turn' || rng.nextDouble() < 0.2 ? 'at end' : 'at start';
    defs.push(
      `<ornamentDef name="${defName}" alignment="${alignment}">` +
        `${sampleSpread(kind, rng, breadth)}</ornamentDef>`,
    );
    entries.push(
      `<ornament date="${n.date.toFixed(1)}" name.ref="${defName}" noteid="#${id}"` +
        ` note.order="${order}" xml:id="mlorn${i}">${poolNotes}</ornament>`,
    );
  });
  if (entries.length === 0) return { header: '', map: '' };
  return {
    header:
      '<ornamentationStyles><styleDef name="mlignOrns">' +
      defs.join('') +
      '</styleDef></ornamentationStyles>',
    map: `<ornamentationMap><style date="0.0" name.ref="mlignOrns" />${entries.join('')}</ornamentationMap>`,
  };
}

/** Splice ornament header + map, and any imprecision maps, into a buildMpm document. */
export function injectOrnaments(mpmXml, orn, imprecisionXml = '') {
  const map = (orn && orn.map) || '';
  if (!map && !imprecisionXml) return mpmXml;
  let out = mpmXml;
  if (map) out = out.replace('<global><header />', `<global><header>${orn.header}</header>`);
  return out.replace('<dated><tempoMap>', `<dated>${map}${imprecisionXml}<tempoMap>`);
}

// ---------------------------------------------------------------------------
// Humanisation (MPM imprecisionMap). Attribute names taken from meico-ts's own
// parser (mpm/elements/maps/data/distribution.ts): the DOMAIN is the element's
// local name — `imprecisionMap.timing`, not a `domain=` attribute — and each
// distribution family has its own element name and attribute set.
//
// Safe for ground truth: the renderer perturbs `milliseconds.date`,
// `milliseconds.date.end` and `velocity` IN PLACE on existing elements. It
// creates no notes, removes none and touches no xml:id, so per-note provenance
// survives it (verified in ImprecisionMap.renderImprecisionToMap).
//
// σ in ms for timing, in velocity units for dynamics. `early` is the loose
// timing of a pre-WWII recording, not a modern studio take.
// ---------------------------------------------------------------------------
export const IMPRECISION_LEVELS = {
  subtle: { timing: 8, dynamics: 3 },
  natural: { timing: 18, dynamics: 6 },
  early: { timing: 35, dynamics: 10 },
};

/** imprecisionMap XML for the given level, or '' when disabled. */
export function sampleImprecision(rng, level) {
  const p = IMPRECISION_LEVELS[level];
  if (!p) return '';
  // Draw the actual σ per piece around the level's centre, so a corpus is not
  // one single humanisation setting repeated.
  const sd = (c) => (c * (0.7 + 0.6 * rng.nextDouble())).toFixed(2);
  // A seed per map keeps generation reproducible from the piece seed alone.
  const seed = () => rng.nextInt(2 ** 30);
  const t = sd(p.timing);
  const d = sd(p.dynamics);
  return (
    `<imprecisionMap.timing><distribution.gaussian date="0.0" seed="${seed()}"` +
    ` deviation.standard="${t}" limit.lower="${(-3 * t).toFixed(2)}"` +
    ` limit.upper="${(3 * t).toFixed(2)}" milliseconds.timingBasis="1.0" />` +
    `</imprecisionMap.timing>` +
    `<imprecisionMap.dynamics><distribution.gaussian date="0.0" seed="${seed()}"` +
    ` deviation.standard="${d}" limit.lower="${(-3 * d).toFixed(2)}"` +
    ` limit.upper="${(3 * d).toFixed(2)}" />` +
    `</imprecisionMap.dynamics>`
  );
}

const r3 = (v) => Math.round(v * 1000) / 1000;

/**
 * Ornament pre-pass over the facade's PerformanceData (meico-ts ornamentation
 * merge, 05147ed; on older dists the fields are undefined → identity).
 *
 * A note is GENERATED iff its id is not a known score id (generated notes get
 * random meico_<uuid> ids; slot membership is NOT sufficient — the principal
 * itself appears inside the figure with a slot and keeps its score id, and
 * stays a match per D10). Generated notes get id=null + an ornament origin so
 * the robustness layer and editsToAlignment treat them as provenanced
 * insertions. Carved heads keep score ids (match with altered duration).
 */
export function normalizeOrnaments(data, scoreIdSet) {
  let touched = false;
  const parts = data.parts.map((part) => ({
    ...part,
    notes: part.notes.map((n) => {
      if (n.id !== null && scoreIdSet.has(n.id)) return n;
      if (!n.ornamented && n.id !== null) return n; // unknown non-ornament id: leave as-is
      touched = true;
      return {
        ...n,
        id: null,
        origin: {
          type: 'ornament',
          anchor: n.ornamentAnchor ?? null,
          ref: n.ornamentRef ?? null,
          slot: n.ornamentSlot ?? -1,
          pass: n.ornamentPass ?? 0,
        },
      };
    }),
  }));
  return touched ? { ...data, parts } : data;
}

// Exaggeration axis (meico-ts-exag branch, pinned 3432d25; dynamic import so
// the generator still runs where the worktree is absent).
const EXAG_DIST = '/Users/nielspfeffer/Projects/meico-ts/dist/index.js'; // main @ 9974ba3
let exagMod = null;
export async function loadExaggeration() {
  if (exagMod === null) exagMod = await import(EXAG_DIST);
  return exagMod;
}

// Exaggeration curriculum ranges. `modern` is the original, calibrated on
// post-war playing. `early` widens the deviation axes — rubato and tempo above
// all — toward pre-WWII practice (free tempo, heavy rubato), where the
// original caps sit well inside the everyday range rather than at its edge.
export const EXAG_PROFILES = {
  modern: { tempo: [0.5, 2.0], dynamics: [0.6, 1.7], rubato: [0.5, 2.0], articulation: [0.6, 1.6] },
  early: { tempo: [0.4, 3.0], dynamics: [0.5, 2.0], rubato: [0.5, 3.5], articulation: [0.5, 1.8] },
};

/** Log-uniform s-vector sampler over one profile's ranges. */
export function sampleExagFactors(rng, profile = 'modern') {
  const p = EXAG_PROFILES[profile] ?? EXAG_PROFILES.modern;
  const lu = ([lo, hi]) => Math.exp(Math.log(lo) + rng.nextDouble() * (Math.log(hi) - Math.log(lo)));
  return {
    tempo: lu(p.tempo),
    dynamics: lu(p.dynamics),
    rubato: lu(p.rubato),
    articulation: lu(p.articulation),
  };
}

/** One corpus row, or null when an invariant fails. */
export function buildSample(piece, robustnessCfg, seedStr, ornDefs = null, exagFactors = null, imprecisionXml = '') {
  const { msm, mpm: mpmBase } = documentsFor(piece);
  let mpm = injectOrnaments(mpmBase, ornDefs, imprecisionXml);
  if (exagFactors !== null) {
    if (exagMod === null) throw new Error('call loadExaggeration() first');
    mpm = exagMod.exaggerateMpm(mpm, { factors: exagFactors, msm }).mpm;
  }
  const rendered = captureConsole(() => performMsmToData({ msm, mpm })).value;
  const scoreIdSet = new Set();
  for (const part of piece.parts) {
    for (let i = 0; i < part.notes.length; i++) scoreIdSet.add(`p${part.number}n${i}`);
  }
  const clean = normalizeOrnaments(rendered, scoreIdSet);

  const { data, edits } = applyRobustness(clean, robustnessCfg, seedStr);
  const { alignment, perfNotes, unattributed } = editsToAlignment(data, edits);
  if (unattributed > 0) return null;
  const shifted = shiftToMatchedZero(perfNotes, alignment);

  // Score side straight from the sampled piece (ids match buildMsm's p<part>n<i>).
  const scoreRows = [];
  piece.parts.forEach((part, voice) => {
    part.notes.forEach((n, i) => {
      scoreRows.push({ id: `p${part.number}n${i}`, onset: n.date, dur: n.dur, pitch: n.pitch, voice });
    });
  });
  scoreRows.sort((a, b) => a.onset - b.onset || a.pitch - b.pitch);
  const si = new Map(scoreRows.map((row, i) => [row.id, i]));
  const pi = new Map(shifted.map((row, i) => [row.perfId, i]));

  const align = [];
  const subs = [];
  const ins = [];
  const orn = [];
  const del = [];
  const INS_KIND = { slip: 0, 'restart-first-pass': 1, ornament: 2, addition: 3 };
  for (const rec of alignment) {
    if (rec.label === 'match') {
      const s = si.get(rec.scoreId);
      const p = pi.get(rec.perfId);
      if (s === undefined || p === undefined) return null;
      align.push([s, p]);
      if (rec.sub) subs.push([s, rec.sub.from, rec.sub.to]);
    } else if (rec.label === 'insertion') {
      const p = pi.get(rec.perfId);
      if (p === undefined) return null;
      const kind = rec.provenance.type;
      ins.push([p, INS_KIND[kind] ?? 4]);
      // Attribution channel. Both espressivo's generated ornaments and the
      // robustness layer's consonant additions elaborate a specific written
      // note, so both are attributable and both belong here — an added octave
      // or filled chord tone answers "which note does this belong to" exactly
      // as a trill note does. `ins` keeps them distinguishable by kind.
      if (kind === 'ornament' || kind === 'addition') {
        const anchor = rec.provenance.anchor ?? null;
        const anchorSi = anchor !== null ? si.get(anchor) : undefined;
        orn.push([p, anchorSi ?? -1, rec.provenance.slot ?? 0, rec.provenance.pass ?? 0]);
      }
    } else {
      const s = si.get(rec.scoreId);
      if (s === undefined) return null;
      del.push([s]);
    }
  }

  // Invariants: total coverage of both sides.
  if (align.length + ins.length !== shifted.length) return null;
  if (align.length + del.length !== scoreRows.length) return null;

  return {
    meta: { gen: 'mlign-v0', seed: seedStr },
    score: scoreRows.map((row) => [row.onset, row.dur, row.pitch, row.voice]),
    scoreIds: scoreRows.map((row) => row.id),
    perf: shifted.map((n) => [r3(n.onsetMs), r3(n.offsetMs - n.onsetMs), n.pitch, n.velocity]),
    align,
    subs,
    ins,
    orn,
    del: del.map(([s]) => s),
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.exaggerate) await loadExaggeration();
  const preset = PRESETS[args.robustness];
  if (!preset) throw new Error(`unknown robustness preset: ${args.robustness}`);
  const cfg = mergeConfig({ ...preset, jitter: { stdMs: args.jitter } });

  // Synchronous writes: the generation loop is pure sync compute and never
  // yields to the event loop, so a stream would buffer EVERYTHING in memory
  // until end() — one kill loses the whole shard (it did). writeSync persists
  // each row immediately.
  const fd = openSync(args.out, 'w');
  let written = 0;
  let dropped = 0;
  for (let i = 0; i < args.n; i++) {
    const rng = new JavaRandom(args.seed * 1000003n + BigInt(i));
    const piece = samplePieceV4(rng, i, WANT, OPT);
    const ornDefs = args.ornaments > 0 ? sampleOrnaments(piece, rng, args.ornaments, args.breadth) : null;
    const exagFactors = args.exaggerate ? sampleExagFactors(rng, args.profile) : null;
    const imprecision = sampleImprecision(rng, args.imprecision);
    let row;
    try {
      row = buildSample(piece, cfg, `${args.seed}:${i}`, ornDefs, exagFactors, imprecision);
    } catch (err) {
      process.stderr.write(`piece ${i} render failed: ${err.message}\n`);
      row = null;
    }
    if (row === null) {
      dropped++;
      continue;
    }
    writeSync(fd, JSON.stringify(row) + '\n');
    written++;
    if ((i + 1) % 50 === 0) process.stderr.write(`...${i + 1}/${args.n}\n`);
  }
  closeSync(fd);
  process.stderr.write(`wrote ${written} samples to ${args.out} (${dropped} dropped)\n`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
