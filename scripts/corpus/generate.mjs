/**
 * Synthetic training-corpus generator v0.
 *
 * Composition (the score/MPM samplers are shared with the mpmify project):
 *   score + MPM-map sampling — mpmify's modules (imported read-only);
 *   rendering               — espressivo facade (performMsmToData, once);
 *   robustness + GT         — our layer.
 *
 * Usage:
 *   node scripts/corpus/generate.mjs <out.jsonl> <numPieces> <seed>
 *        [--robustness none|light|medium|heavy] [--jitter <stdMs>]
 *
 * Output: docs/corpus-format.md rows, one per line. Pieces whose GT fails an
 * invariant are dropped (counted in the final summary line on stderr).
 */

import { appendFileSync, openSync, closeSync, writeSync } from 'node:fs';
import { JavaRandom } from '/Users/nielspfeffer/Projects/mpmify/ml/node/java_random.mjs';
import {
  samplePieceV4,
  documentsFor,
  captureConsole,
} from '/Users/nielspfeffer/Projects/mpmify/ml/node/generate_v4.mjs';
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
  const opt = { robustness: 'medium', jitter: 12, ornaments: 0, exaggerate: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--robustness') opt.robustness = argv[++i];
    else if (a === '--jitter') opt.jitter = Number(argv[++i]);
    else if (a === '--ornaments') opt.ornaments = Number(argv[++i]);
    else if (a === '--exaggerate') opt.exaggerate = true;
    else pos.push(a);
  }
  if (pos.length !== 3) {
    throw new Error('usage: generate.mjs <out.jsonl> <numPieces> <seed> [--robustness p] [--jitter ms]');
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

const ORN_HEADER =
  '<ornamentationStyles><styleDef name="mlignOrns">' +
  '<ornamentDef name="trill" alignment="at start">' +
  '<temporalSpread frame.offset="0.0ticks" frameLength="100%" noteoff.shift="monophonic" /></ornamentDef>' +
  '<ornamentDef name="mordent" alignment="at start">' +
  '<temporalSpread frame.offset="0.0ticks" frameLength="30%" noteoff.shift="monophonic" /></ornamentDef>' +
  '<ornamentDef name="turn" alignment="at end">' +
  '<temporalSpread frame.offset="0.0ticks" frameLength="50%" noteoff.shift="monophonic" /></ornamentDef>' +
  '</styleDef></ornamentationStyles>';

/**
 * Sample ornaments for a piece: candidates = part-1 notes with duration ≥ one
 * quarter, spaced ≥ one quarter apart; each gets trill/mordent/turn.
 * rateP = probability per candidate. Returns the ornamentationMap XML or ''.
 */
export function sampleOrnaments(piece, rng, rateP) {
  const part = piece.parts[0];
  const entries = [];
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
      const reps = 2 + rng.nextInt(3); // 2..4 alternation pairs
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
    entries.push(
      `<ornament date="${n.date.toFixed(1)}" name.ref="${kind}" noteid="#${id}"` +
        ` note.order="${order}" xml:id="mlorn${i}">${poolNotes}</ornament>`,
    );
  });
  if (entries.length === 0) return '';
  return `<ornamentationMap><style date="0.0" name.ref="mlignOrns" />${entries.join('')}</ornamentationMap>`;
}

/** Splice ornament header + map into a buildMpm document. */
export function injectOrnaments(mpmXml, ornMapXml) {
  if (!ornMapXml) return mpmXml;
  return mpmXml
    .replace('<global><header />', `<global><header>${ORN_HEADER}</header>`)
    .replace('<dated><tempoMap>', `<dated>${ornMapXml}<tempoMap>`);
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

/** Log-uniform s-vector sampler over the safe curriculum ranges. */
export function sampleExagFactors(rng) {
  const lu = (lo, hi) => Math.exp(Math.log(lo) + rng.nextDouble() * (Math.log(hi) - Math.log(lo)));
  return {
    tempo: lu(0.5, 2.0),
    dynamics: lu(0.6, 1.7),
    rubato: lu(0.5, 2.0),
    articulation: lu(0.6, 1.6),
  };
}

/** One corpus row, or null when an invariant fails. */
export function buildSample(piece, robustnessCfg, seedStr, ornMapXml = '', exagFactors = null) {
  const { msm, mpm: mpmBase } = documentsFor(piece);
  let mpm = injectOrnaments(mpmBase, ornMapXml);
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
  const INS_KIND = { slip: 0, 'restart-first-pass': 1, ornament: 2 };
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
      ins.push([p, INS_KIND[rec.provenance.type] ?? 3]);
      if (rec.provenance.type === 'ornament') {
        const anchorSi = rec.provenance.anchor !== null ? si.get(rec.provenance.anchor) : undefined;
        orn.push([p, anchorSi ?? -1, rec.provenance.slot, rec.provenance.pass]);
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
    const ornMap = args.ornaments > 0 ? sampleOrnaments(piece, rng, args.ornaments) : '';
    const exagFactors = args.exaggerate ? sampleExagFactors(rng) : null;
    let row;
    try {
      row = buildSample(piece, cfg, `${args.seed}:${i}`, ornMap, exagFactors);
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
