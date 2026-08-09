/**
 * Synthetic training-corpus generator v0.
 *
 * Composition per the shared-generator deal (research/00-coordination.md):
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

import { createWriteStream } from 'node:fs';
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
  const opt = { robustness: 'medium', jitter: 12 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--robustness') opt.robustness = argv[++i];
    else if (a === '--jitter') opt.jitter = Number(argv[++i]);
    else pos.push(a);
  }
  if (pos.length !== 3) {
    throw new Error('usage: generate.mjs <out.jsonl> <numPieces> <seed> [--robustness p] [--jitter ms]');
  }
  return { out: pos[0], n: Number(pos[1]), seed: BigInt(pos[2]), ...opt };
}

const WANT = { dynamics: true, articulation: true, rubato: true, asynchrony: true, movement: true, accentuation: false };
const OPT = { twoPartProb: 0.5, asynchronyProb: 1.0, movementProb: 0.5 };

const r3 = (v) => Math.round(v * 1000) / 1000;

/** One corpus row, or null when an invariant fails. */
export function buildSample(piece, robustnessCfg, seedStr) {
  const { msm, mpm } = documentsFor(piece);
  const clean = captureConsole(() => performMsmToData({ msm, mpm })).value;

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
    del: del.map(([s]) => s),
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const preset = PRESETS[args.robustness];
  if (!preset) throw new Error(`unknown robustness preset: ${args.robustness}`);
  const cfg = mergeConfig({ ...preset, jitter: { stdMs: args.jitter } });

  const out = createWriteStream(args.out);
  let written = 0;
  let dropped = 0;
  for (let i = 0; i < args.n; i++) {
    const rng = new JavaRandom(args.seed * 1000003n + BigInt(i));
    const piece = samplePieceV4(rng, i, WANT, OPT);
    let row;
    try {
      row = buildSample(piece, cfg, `${args.seed}:${i}`);
    } catch (err) {
      process.stderr.write(`piece ${i} render failed: ${err.message}\n`);
      row = null;
    }
    if (row === null) {
      dropped++;
      continue;
    }
    out.write(JSON.stringify(row) + '\n');
    written++;
    if ((i + 1) % 50 === 0) process.stderr.write(`...${i + 1}/${args.n}\n`);
  }
  out.end(() => {
    process.stderr.write(`wrote ${written} samples to ${args.out} (${dropped} dropped)\n`);
  });
}

if (import.meta.url === `file://${process.argv[1]}`) main();
