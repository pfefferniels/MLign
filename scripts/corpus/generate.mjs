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
 *        [--ornaments <scale>]       ornament density, 1 = the real ASAP rate
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

import { appendFileSync, openSync, closeSync, writeSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
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
import {
  DEFAULTS,
  buildOrnamentation,
  injectOrnaments,
  normalizeOrnaments,
} from '../../src/corpus/ornaments.mjs';
import { editsToAlignment, shiftToMatchedZero } from '../../src/robustness/gt.mjs';

const PRESETS = { none: {}, light: presetLight, medium: presetMedium, heavy: presetHeavy };

function parseArgs(argv) {
  const pos = [];
  const opt = {
    robustness: 'medium', jitter: 12, ornaments: 0, addRate: null, ornJitter: null, minDur: null,
    exaggerate: false, profile: 'modern', breadth: 1, imprecision: '',
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--robustness') opt.robustness = argv[++i];
    else if (a === '--jitter') opt.jitter = Number(argv[++i]);
    else if (a === '--ornaments') opt.ornaments = Number(argv[++i]);
    else if (a === '--add-rate') opt.addRate = Number(argv[++i]);
    else if (a === '--orn-jitter') opt.ornJitter = Number(argv[++i]);
    else if (a === '--min-dur') opt.minDur = Number(argv[++i]);
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
// Ornament placement. The FIGURES live in src/corpus/ornaments.mjs, shared with
// the real-score generator; what stays here is which notes get one, and how
// many — the axis that was most badly wrong.
// ---------------------------------------------------------------------------

/**
 * How many ornament signs a piece carries, per 1000 score notes.
 *
 * Measured over the 203 train-eligible ASAP scores: **4838 ornament EVENTS
 * over 535 736 sounding notes = 9.03 per 1000**. 14.3 % of scores carry none;
 * the rest fit a lognormal with median 6.33 and ln-sd 1.05.
 *
 * Both halves of that sentence name a place an earlier version of this
 * constant went wrong, in opposite directions:
 *
 * - The DENOMINATOR is sounding notes, not `<note>` elements. 49 011 of the
 *   latter are rests, which cannot carry an ornament.
 * - The NUMERATOR is events, not elements. `<arpeggiate>` and `<grace>` are
 *   written per note — a rolled four-note chord emits four, a grace run one
 *   per grace — and `<wavy-line>` is not an ornament at all but the trill's
 *   extension line, 265 of whose 549 occurrences sit in the same
 *   `<ornaments>` element as the `<trill-mark>` they extend. Counting
 *   elements rather than events inflates the rate by 42 %.
 *
 * The corpus this replaces put 141.8 ornament groups per 1000 score notes into
 * every single piece, and a third of all played notes belonged to one. A head
 * trained on that has no way to learn what "not an ornament" looks like, which
 * is exactly the half that failed to transfer to real recordings.
 */
export const ORN_RATE = { zeroProb: 0.143, lnMedian: 1.846, lnSd: 1.047 };

/**
 * Relative frequency of each event kind, from the same census. The ordering is
 * the finding: grace notes and arpeggios are three quarters of all notated
 * ornaments in this repertoire, and the corpus had neither.
 *
 * `inverted-turn`, `delayed-turn` and `schleifer` do not occur in ASAP at all,
 * so they stay realizable — a real score may still ask for one — but are never
 * sampled here.
 */
export const ORN_KINDS = [
  ['grace', 0.470],
  ['arpeggio', 0.253],
  ['trill', 0.145],
  ['tremolo', 0.055],
  ['inverted-mordent', 0.037],
  ['turn', 0.032],
  ['mordent', 0.007],
];

/**
 * Notes per grace run: 1452 singles, 500 doubles, 215 triples, 78 fours in
 * ASAP (mean 1.63). A slide of two or three leaning notes is common enough
 * that a corpus of only single appoggiaturas would be a different repertoire.
 */
export const GRACE_RUN = [[1, 0.638], [2, 0.220], [3, 0.094], [4, 0.048]];

/** Box-Muller — JavaRandom is a port of java.util.Random minus nextGaussian. */
function gaussian(rng) {
  const u = Math.max(rng.nextDouble(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng.nextDouble());
}

/** Signs per 1000 notes for one piece, drawn from the fitted real distribution. */
export function sampleOrnRate(rng, scale = 1) {
  if (rng.nextDouble() < ORN_RATE.zeroProb) return 0;
  return scale * Math.exp(ORN_RATE.lnMedian + ORN_RATE.lnSd * gaussian(rng));
}

/** Weighted choice over an `[[value, probability], …]` table. */
function pick(rng, table) {
  let u = rng.nextDouble();
  for (const [value, p] of table) {
    u -= p;
    if (u < 0) return value;
  }
  return table[table.length - 1][0];
}

/**
 * The pitches of one grace run, leaning stepwise into the principal — which is
 * what a slide or a double appoggiatura does. A run that just sat on one
 * neighbour would make the multi-note case indistinguishable from a repeated
 * single one.
 */
function gracePitches(rng, pitch) {
  const k = pick(rng, GRACE_RUN);
  const dir = rng.nextDouble() < 0.6 ? -1 : 1; // more often from below
  const out = [];
  for (let i = k; i >= 1; i--) out.push(pitch + dir * (i === 1 ? 1 + rng.nextInt(2) : i + rng.nextInt(2)));
  return out;
}

/**
 * Sample ornaments for a piece and realize them.
 *
 * `rateScale` multiplies the drawn per-piece rate — 1 reproduces the ASAP
 * distribution, and a shard can be deliberately ornament-rich without going
 * back to "every piece, every long note". A note carries at most one sign, and
 * the figures that need room (trill, turn) are not placed on notes too short
 * to hold them.
 *
 * Returns `{ header, map }`, both '' when nothing was sampled.
 */
export function sampleOrnaments(piece, rng, rateScale, breadth = 1) {
  const part = piece.parts[0];
  const total = piece.parts.reduce((a, p) => a + p.notes.length, 0);
  // Stochastic rounding, not Math.round. A sampled piece is ~95 notes, so the
  // expected count is well under 1 and rounding to nearest turns "0.4 signs"
  // into either 0 or 1 with a threshold — which put a floor under the rate and
  // left the corpus 7× too dense even after the distribution was right.
  const expected = (sampleOrnRate(rng, rateScale) * total) / 1000;
  const want = Math.floor(expected) + (rng.nextDouble() < expected % 1 ? 1 : 0);
  if (want <= 0) return { header: '', map: '' };

  // Chords, by onset — an arpeggio needs one, and a sampled piece has them at
  // about 17 % of its onsets.
  const chords = new Map();
  part.notes.forEach((n, i) => {
    const at = chords.get(n.date);
    if (at) at.push(i);
    else chords.set(n.date, [i]);
  });
  const chordDates = [...chords.entries()].filter(([, v]) => v.length >= 2).map(([d]) => d);

  const requests = [];
  const taken = new Set();
  // Rejection sampling over part 1 — cheaper than shuffling, and the retry
  // budget keeps a piece with few eligible notes from spinning.
  for (let tries = 0; requests.length < want && tries < want * 20; tries++) {
    const kind = pick(rng, ORN_KINDS);
    const id = (i) => `p${part.number}n${i}`;

    if (kind === 'arpeggio') {
      if (chordDates.length === 0) continue;
      const date = chordDates[rng.nextInt(chordDates.length)];
      const members = chords.get(date);
      if (members.some((i) => taken.has(i))) continue;
      for (const i of members) taken.add(i);
      requests.push({
        msmId: id(members[0]),
        date,
        kind,
        // Low to high: the order the chord is rolled in.
        chordIds: [...members].sort((a, b) => part.notes[a].pitch - part.notes[b].pitch).map(id),
        index: requests.length,
      });
      continue;
    }

    const i = rng.nextInt(part.notes.length);
    if (taken.has(i)) continue;
    const n = part.notes[i];
    const durQuarters = n.dur / 720;
    if ((kind === 'trill' || kind === 'turn' || kind === 'tremolo') && durQuarters < 0.5) continue;
    if (durQuarters < 0.125) continue;
    taken.add(i);
    requests.push({
      msmId: id(i),
      date: n.date,
      durQuarters,
      pitch: n.pitch,
      kind,
      // A sampled grace has no notated pitch of its own, so the run is drawn
      // from the real distribution of run lengths and leans into the principal.
      gracePitches: gracePitches(rng, n.pitch),
      beams: 1 + rng.nextInt(3),
      slashed: rng.nextDouble() < 0.5,
      index: requests.length,
    });
  }
  return buildOrnamentation(requests, rng, { ...DEFAULTS, breadth });
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
  const { data: clean } = normalizeOrnaments(rendered, scoreIdSet);

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

/**
 * Write `<out>.recipe.json`: the argv that produced this shard, the resolved
 * options, and the commit the generator was at.
 *
 * A sidecar rather than a field on every row, because the row-level cost is not
 * free — 40 000 rows x ~150 bytes is ~6 MB of the same sentence — and because
 * the question it answers ("how do I regenerate this?") is asked of the file,
 * not of a row. Written BEFORE the loop, so an interrupted shard still says
 * what it was trying to be.
 */
function writeRecipe(args) {
  let commit = null;
  try {
    commit = execFileSync('git', ['rev-parse', 'HEAD'],
                          { cwd: new URL('../..', import.meta.url).pathname,
                            encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch { /* not a checkout, or no git: the argv below is still the point */ }

  const recipe = {
    schema: 'mlign-corpus-recipe/1',
    generator: 'scripts/corpus/generate.mjs',
    // The reproduction command, verbatim and runnable.
    argv: process.argv.slice(1).join(' '),
    options: args,
    commit,
    written: new Date().toISOString(),
  };
  const path = `${args.out}.recipe.json`;
  // `seed` is a BigInt (the sampler needs the width); JSON.stringify throws on
  // one rather than coercing, so it is rendered as the digits it is.
  const json = JSON.stringify(recipe, (_k, v) => (typeof v === 'bigint' ? v.toString() : v), 2);
  writeFileSync(path, json + '\n');
  process.stderr.write(`recipe: ${path}\n`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.exaggerate) await loadExaggeration();
  const preset = PRESETS[args.robustness];
  if (!preset) throw new Error(`unknown robustness preset: ${args.robustness}`);
  // The robustness layer's consonant additions land in the SAME attribution
  // channel as espressivo's ornaments — an added octave elaborates its anchor
  // exactly as a trill note does — so their rate is part of the ornament base
  // rate, not just an alignment-robustness setting. `medium` puts 25 of them
  // per 1000 notes, 3× the real ornament rate, which is how the attribution
  // head came to be trained on a world where added notes are the common case.
  // Overridable rather than changed in the preset: the presets define what the
  // existing corpora mean.
  const cfg = mergeConfig({
    ...preset,
    jitter: {
      stdMs: args.jitter,
      // A trill's notes are ~40 ms apart; jittering each independently at the
      // ordinary σ reorders them and erases the figure. See robustness.mjs.
      ornamentStdMs: args.ornJitter,
      minDurMs: args.minDur ?? undefined,
    },
    ...(args.addRate === null ? {} : { add: { ...preset.add, rate: args.addRate } }),
  });

  // How this shard was made, beside the shard itself.
  //
  // `data/` is gitignored, so a corpus carries no provenance a repo can hold:
  // `meta.seed` on every row pins the sampler but says nothing about the flags,
  // and those flags ARE the corpus — `--breadth 2 --imprecision natural` and
  // `--breadth 3 --imprecision early` are what make orn-a and orn-b different
  // shards rather than two draws of one. Regenerating "the same corpus with one
  // thing changed" is impossible without them, and reconstructing them from
  // memory silently flattens exactly the axis they vary.
  //
  // Once cost me an afternoon: the only surviving record of how orn-a/orn-b and
  // their holdouts were generated was a chat transcript. Not again.
  writeRecipe(args);

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
