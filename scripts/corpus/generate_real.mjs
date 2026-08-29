/**
 * Real-score corpus generator: ASAP scores, performed by espressivo, with their
 * OWN notated ornaments realized.
 *
 *   node scripts/corpus/generate_real.mjs <out.jsonl> <specs.jsonl> <seed> [options]
 *
 * Why this exists. The sampled-piece generator gives correct ornament
 * provenance in music nobody wrote: random rhythm-grid walks with ornaments
 * scattered over them at a rate drawn from a distribution. Real trills sit in
 * idiomatic figuration and real voice-leading, on the notes a composer chose,
 * at the rate that composer wrote them — and the head's failure on real
 * recordings is a failure to recognize exactly that context.
 *
 * The score side comes from `scripts/corpus/asap_spec.py`, which joins
 * partitura's note table to the raw MusicXML's ornament signs through the
 * `@id` every ASAP `<note>` carries. The performance side is the same
 * espressivo path the sampled generator uses, so the ground truth is the same
 * ground truth: per-note ids through a single perform() call.
 *
 * Espressivo used to derive MPM ornament instructions from an MEI's own
 * `<trill>`/`<mordent>`/`<turn>` elements; that capability is gone from the
 * library, and we need it from MusicXML rather than MEI anyway, so the
 * sign → instruction mapping lives here (src/corpus/ornaments.mjs) instead.
 *
 * Options
 *   --takes <n>        renderings per score, each with its own sampled MPM (4)
 *   --window <n>       score notes per emitted row (128)
 *   --stride <n>       window step (64)
 *   --breadth <f>      ≥1, widens ornament figures toward early-recording style
 *   --robustness p     none|light|medium|heavy   --jitter <ms>   --add-rate <r>
 *   --imprecision l    subtle|natural|early      --exaggerate [modern|early]
 *   --limit <n>        only the first n scores (smoke runs)
 *
 * Output: notes/corpus-format.md rows, `meta.gen = "mlign-real-v1"`.
 */

import { openSync, closeSync, writeSync, readFileSync } from 'node:fs';
import { JavaRandom } from '/Users/nielspfeffer/Projects/fenby/node/java_random.mjs';
import {
  PPQ,
  sampleTempoMap,
  sampleDynamicsMap,
  sampleArticulationMap,
  sampleRubatoMap,
  sampleAsynchronyMap,
  sampleMovementMap,
} from '/Users/nielspfeffer/Projects/fenby/node/sampler.mjs';
import { buildMsm, buildMpm } from '/Users/nielspfeffer/Projects/fenby/node/xml.mjs';
import { performMsmToData } from '/Users/nielspfeffer/Projects/meico-ts/dist/api/index.js';
import {
  applyRobustness,
  presetLight,
  presetMedium,
  presetHeavy,
  mergeConfig,
} from '../../src/robustness/robustness.mjs';
import { editsToAlignment, shiftToMatchedZero } from '../../src/robustness/gt.mjs';
import {
  DEFAULTS,
  buildOrnamentation,
  injectOrnaments,
  normalizeOrnaments,
} from '../../src/corpus/ornaments.mjs';
import {
  EXAG_PROFILES,
  IMPRECISION_LEVELS,
  loadExaggeration,
  sampleExagFactors,
  sampleImprecision,
} from './generate.mjs';

const PRESETS = { none: {}, light: presetLight, medium: presetMedium, heavy: presetHeavy };
const TEMPO_SEG_MAX = [6, 8, 12, 16]; // fenby generate_v4's own domain

function parseArgs(argv) {
  const pos = [];
  const opt = {
    takes: 4, window: 128, stride: 64, breadth: 1, robustness: 'medium', jitter: 12,
    addRate: null, imprecision: '', exaggerate: false, profile: 'modern', limit: 0,
    role: 'all',
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--takes') opt.takes = Number(argv[++i]);
    else if (a === '--window') opt.window = Number(argv[++i]);
    else if (a === '--stride') opt.stride = Number(argv[++i]);
    else if (a === '--breadth') opt.breadth = Number(argv[++i]);
    else if (a === '--robustness') opt.robustness = argv[++i];
    else if (a === '--jitter') opt.jitter = Number(argv[++i]);
    else if (a === '--add-rate') opt.addRate = Number(argv[++i]);
    else if (a === '--imprecision') opt.imprecision = argv[++i];
    else if (a === '--limit') opt.limit = Number(argv[++i]);
    else if (a === '--role') opt.role = argv[++i];
    else if (a === '--exaggerate') {
      opt.exaggerate = true;
      if (argv[i + 1] && !argv[i + 1].startsWith('--')) opt.profile = argv[++i];
    } else pos.push(a);
  }
  if (pos.length !== 3)
    throw new Error('usage: generate_real.mjs <out.jsonl> <specs.jsonl> <seed> [options]');
  if (opt.exaggerate && !EXAG_PROFILES[opt.profile])
    throw new Error(`unknown exaggeration profile: ${opt.profile}`);
  if (opt.imprecision && !IMPRECISION_LEVELS[opt.imprecision])
    throw new Error(`unknown imprecision level: ${opt.imprecision}`);
  return { out: pos[0], specs: pos[1], seed: BigInt(pos[2]), ...opt };
}

/** Ascending distinct onset dates of one part — the articulation map's domain. */
function distinctDates(notes) {
  return [...new Set(notes.map((n) => n.date))].sort((a, b) => a - b);
}

/**
 * A fenby-shaped `piece` from a score spec, plus the spec-id → MSM-id map.
 *
 * `buildMsm` names notes `p<part.number>n<index>` by position, so the map has
 * to be built from the same ordering the piece hands it — which is why this
 * returns both rather than letting a caller guess.
 */
function pieceFor(spec, rng, want) {
  const totalTicks = Math.max(spec.totalTicks, PPQ);
  const idOf = new Map();
  const parts = spec.parts.map((p, pi) => {
    const number = pi + 1;
    const notes = [...p.notes].sort((a, b) => a.date - b.date || a.pitch - b.pitch);
    notes.forEach((n, i) => idOf.set(n.id, `p${number}n${i}`));
    return { name: pi === 0 ? 'Piano' : `Staff${number}`, number, midiChannel: pi, midiPort: 0, notes };
  });

  const tempi = sampleTempoMap(rng, totalTicks, {
    bpmLo: 25, bpmHi: 240, segMin: 4, segSpan: TEMPO_SEG_MAX[rng.nextInt(4)] - 3,
  });
  const dyns = want.dynamics
    ? sampleDynamicsMap(rng, totalTicks, {
        volLo: 30, volSpan: 85, segMin: 4, segSpan: TEMPO_SEG_MAX[rng.nextInt(4)] - 3,
      })
    : [];
  // Articulation is per-part (fenby's CANONICAL A6): a global map addresses
  // dates, and meico resolves a date carrying no note in *this* part onto the
  // part's next note — which on two independent hands mislabels most of them.
  for (const part of parts) {
    part.articulation = want.articulation ? sampleArticulationMap(rng, distinctDates(part.notes)) : [];
    part.asynchrony = [];
  }
  // Asynchrony is by definition per-part, and belongs on the hand that lags.
  if (want.asynchrony && parts.length > 1)
    parts[1].asynchrony = sampleAsynchronyMap(rng, totalTicks);

  return {
    piece: {
      index: 0,
      totalTicks,
      parts,
      maps: {
        tempo: tempi,
        dynamics: dyns,
        articulation: [],
        rubato: want.rubato ? sampleRubatoMap(rng, totalTicks, tempi) : [],
        movement: want.movement ? sampleMovementMap(rng, totalTicks) : [],
        accentuation: null,
      },
    },
    idOf,
  };
}

/**
 * Ornament requests from the score's OWN signs.
 *
 * Nothing is sampled here except the realization: which notes are ornamented,
 * with what, and how often is what the composer wrote.
 *
 * `wavy-line` is DROPPED, not folded into `trill`. It is the trill's extension
 * line rather than an ornament: 265 of its 549 ASAP occurrences sit in the very
 * same `<ornaments>` element as the `<trill-mark>` they extend (so they are
 * already covered), and most of the rest are the `stop` end, which sits on the
 * note where the line finishes — usually not the trilled note at all. Folding
 * them in would put a few hundred trills on the wrong notes.
 *
 * `tremolo` is not modelled and is skipped rather than guessed at.
 */
function requestsFor(spec, idOf, byId) {
  const reqs = [];
  const seen = new Set();
  const push = (r) => {
    if (seen.has(r.msmId)) return; // one figure per note
    seen.add(r.msmId);
    reqs.push({ ...r, index: reqs.length });
  };
  // An arpeggio sign names one note; the figure is its whole chord, which the
  // spec lists separately.
  const chordOf = new Map();
  for (const members of spec.chords || []) for (const id of members) chordOf.set(id, members);

  for (const s of spec.signs) {
    const kind = s.kind;
    if (kind === 'tremolo' || kind === 'wavy-line') continue;
    if (kind === 'arpeggio') {
      const members = (s.chordIds || chordOf.get(s.noteId) || [])
        .map((i) => [byId.get(i), idOf.get(i)])
        .filter(([n, m]) => n && m);
      if (members.length < 2) continue;
      const order = members.sort((a, b) => a[0].pitch - b[0].pitch).map(([, m]) => m);
      push({ msmId: order[0], date: members[0][0].date, kind: 'arpeggio', chordIds: order });
      continue;
    }
    const n = byId.get(s.noteId);
    const msmId = idOf.get(s.noteId);
    if (!n || !msmId) continue;
    push({ msmId, date: n.date, durQuarters: n.dur / PPQ, pitch: n.pitch, kind });
  }

  // Grace notes are not score notes — asap_spec keeps them out of the note
  // table — so each becomes a generated note anchored to its principal, which
  // is exactly what a grace note is.
  //
  // Grouped by principal first: a slide of two or three graces is ONE figure
  // leaning into one note (ASAP mean 1.63 notes per run), and pushing them
  // separately would let the per-note dedup below keep only the first.
  const runs = new Map();
  for (const g of spec.graces) {
    if (!runs.has(g.principal)) runs.set(g.principal, []);
    runs.get(g.principal).push(g);
  }
  for (const [principal, run] of runs) {
    const n = byId.get(principal);
    const msmId = idOf.get(principal);
    if (!n || !msmId) continue;
    push({
      msmId, date: n.date, durQuarters: n.dur / PPQ, pitch: n.pitch, kind: 'grace',
      gracePitches: run.map((g) => g.pitch),
      // A run counts as crushed if any of its notes is slashed.
      slashed: run.some((g) => g.slashed),
    });
  }
  return reqs;
}

const r3 = (v) => Math.round(v * 1000) / 1000;

/** One rendered take: the whole score, as a single un-windowed row. */
function renderTake(spec, rng, args, cfg, seedStr) {
  const { piece, idOf } = pieceFor(spec, rng, {
    dynamics: true, articulation: true, rubato: true, asynchrony: true, movement: true,
  });
  const byId = new Map();
  for (const p of spec.parts) for (const n of p.notes) byId.set(n.id, n);

  const orn = buildOrnamentation(requestsFor(spec, idOf, byId), rng, { ...DEFAULTS, breadth: args.breadth });
  const msm = buildMsm(spec.id, `s${spec.id.replace(/[^A-Za-z0-9]/g, '_')}`, PPQ, piece.parts);
  let mpm = injectOrnaments(buildMpm('perf', PPQ, piece.maps, piece.parts), orn,
                            sampleImprecision(rng, args.imprecision));
  if (args.exaggerate) mpm = exagMod.exaggerateMpm(mpm, { factors: sampleExagFactors(rng, args.profile), msm }).mpm;

  const rendered = performMsmToData({ msm, mpm });
  const scoreIdSet = new Set();
  for (const part of piece.parts)
    for (let i = 0; i < part.notes.length; i++) scoreIdSet.add(`p${part.number}n${i}`);
  const { data: clean } = normalizeOrnaments(rendered, scoreIdSet);

  const { data, edits } = applyRobustness(clean, cfg, seedStr);
  const { alignment, perfNotes, unattributed } = editsToAlignment(data, edits);
  if (unattributed > 0) return null;
  const shifted = shiftToMatchedZero(perfNotes, alignment);

  const scoreRows = [];
  piece.parts.forEach((part, voice) => {
    part.notes.forEach((n, i) => {
      scoreRows.push({ id: `p${part.number}n${i}`, onset: n.date, dur: n.dur, pitch: n.pitch,
                       voice: (n.voice ?? voice) % 5 });
    });
  });
  scoreRows.sort((a, b) => a.onset - b.onset || a.pitch - b.pitch);
  const si = new Map(scoreRows.map((row, i) => [row.id, i]));
  const pi = new Map(shifted.map((row, i) => [row.perfId, i]));
  return { scoreRows, shifted, alignment, si, pi };
}

const INS_KIND = { slip: 0, 'restart-first-pass': 1, ornament: 2, addition: 3 };

/**
 * Cut one take into rows of `win` score notes.
 *
 * A window owns a contiguous run of score notes, the performed notes matched to
 * them, and every insertion whose performed onset falls inside that span — the
 * same rule `scripts/corpus/real_gt.py` uses for real performances, so the two
 * real-score corpora window alike. Indices are rebased per window; both
 * coverage invariants are re-checked afterwards, because a rebasing bug looks
 * exactly like good data.
 */
function* windows(take, win, stride, spec, seedStr) {
  const { scoreRows, shifted, alignment, si, pi } = take;
  const s2p = new Map();
  const insertions = [];
  const deletions = new Set();
  const subs = new Map();
  const ornOf = new Map();
  for (const rec of alignment) {
    if (rec.label === 'match') {
      const s = si.get(rec.scoreId);
      const p = pi.get(rec.perfId);
      if (s === undefined || p === undefined) return;
      s2p.set(s, p);
      if (rec.sub) subs.set(s, rec.sub);
    } else if (rec.label === 'insertion') {
      const p = pi.get(rec.perfId);
      if (p === undefined) return;
      insertions.push({ p, kind: rec.provenance.type, prov: rec.provenance });
      if (rec.provenance.type === 'ornament' || rec.provenance.type === 'addition')
        ornOf.set(p, rec.provenance);
    } else {
      const s = si.get(rec.scoreId);
      if (s === undefined) return;
      deletions.add(s);
    }
  }
  insertions.sort((a, b) => a.p - b.p);

  for (let start = 0; start < scoreRows.length; start += stride) {
    const end = Math.min(start + win, scoreRows.length);
    if (end - start < win / 2) break; // a stub tail teaches nothing

    const matched = [];
    for (let s = start; s < end; s++) if (s2p.has(s)) matched.push(s2p.get(s));
    if (matched.length < 8) continue;
    // Keep the performed window contiguous: every played note between the first
    // and last matched one belongs to it, whatever its label. Picking only the
    // labelled ones would leave holes a model reads as silence.
    const lo = Math.min(...matched);
    const hi = Math.max(...matched);
    const keep = [];
    for (let p = lo; p <= hi; p++) keep.push(p);
    const pRebase = new Map(keep.map((p, i) => [p, i]));

    const sRebase = new Map();
    for (let s = start; s < end; s++) sRebase.set(s, s - start);

    const align = [];
    const del = [];
    for (let s = start; s < end; s++) {
      const p = s2p.get(s);
      if (p === undefined || !pRebase.has(p)) del.push(sRebase.get(s));
      else align.push([sRebase.get(s), pRebase.get(p)]);
    }
    const ins = [];
    const orn = [];
    for (const x of insertions) {
      if (!pRebase.has(x.p)) continue;
      const p = pRebase.get(x.p);
      ins.push([p, INS_KIND[x.kind] ?? 4]);
      const prov = ornOf.get(x.p);
      if (!prov) continue;
      const a = prov.anchor != null ? si.get(prov.anchor) : undefined;
      const rebased = a !== undefined && sRebase.has(a) ? sRebase.get(a) : -1;
      orn.push([p, rebased, prov.slot ?? 0, prov.pass ?? 0]);
    }
    const subsOut = [];
    for (let s = start; s < end; s++) {
      const sub = subs.get(s);
      if (sub && s2p.has(s) && pRebase.has(s2p.get(s)))
        subsOut.push([sRebase.get(s), sub.from, sub.to]);
    }

    const scoreSlice = scoreRows.slice(start, end);
    const perfSlice = keep.map((p) => shifted[p]);
    if (align.length + ins.length !== perfSlice.length) continue;
    if (align.length + del.length !== scoreSlice.length) continue;

    const t0 = perfSlice.length ? perfSlice[0].onsetMs : 0;
    const d0 = scoreSlice.length ? scoreSlice[0].onset : 0;
    yield {
      meta: { gen: 'mlign-real-v1', seed: seedStr, score: spec.id, window: [start, end] },
      score: scoreSlice.map((row) => [r3(row.onset - d0), r3(row.dur), row.pitch, row.voice]),
      scoreIds: scoreSlice.map((row) => row.id),
      perf: perfSlice.map((n) => [r3(n.onsetMs - t0), r3(n.offsetMs - n.onsetMs), n.pitch, n.velocity]),
      align,
      subs: subsOut,
      ins,
      orn,
      del,
    };
  }
}

let exagMod = null;

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.exaggerate) exagMod = await loadExaggeration();
  const preset = PRESETS[args.robustness];
  if (!preset) throw new Error(`unknown robustness preset: ${args.robustness}`);
  const cfg = mergeConfig({
    ...preset,
    jitter: { stdMs: args.jitter },
    ...(args.addRate === null ? {} : { add: { ...preset.add, rate: args.addRate } }),
  });

  let specs = readFileSync(args.specs, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l));
  // The holdout is a set of SCORES, not of windows: windows of the same piece
  // share its figuration, so splitting them would leak the very thing the
  // holdout is meant to test. Every eighth score by id, so the split is a
  // property of the corpus rather than of a seed.
  specs.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  if (args.role === 'holdout') specs = specs.filter((_, i) => i % 8 === 0);
  else if (args.role === 'train') specs = specs.filter((_, i) => i % 8 !== 0);
  else if (args.role !== 'all') throw new Error(`unknown role: ${args.role}`);
  if (args.limit) specs = specs.slice(0, args.limit);

  // fenby's `jd` writes a double in Java's `Double.toString` SHAPE, which turns
  // scientific above 1e7 and below 1e-3 — and an MSM with `1.0244922188E7` in a
  // date is not an MSM. One ASAP score reaches there (Beethoven op. 57/1, whose
  // repeat handling runs the clock to 10.7M ticks); dropping it loudly beats
  // rendering it into silent nonsense.
  const inContract = (v) => v === 0 || (Math.abs(v) < 1e7 && Math.abs(v) >= 1e-3);
  const before = specs.length;
  specs = specs.filter((s) => {
    const bad = s.parts.some((p) => p.notes.some((n) => !inContract(n.date) || !inContract(n.dur)));
    if (bad) process.stderr.write(`skipping ${s.id}: tick values outside the MSM number contract\n`);
    return !bad;
  });
  if (specs.length !== before)
    process.stderr.write(`${before - specs.length} of ${before} scores skipped\n`);

  const fd = openSync(args.out, 'w');
  let written = 0;
  let droppedTakes = 0;
  let failed = 0;
  for (const [k, spec] of specs.entries()) {
    for (let t = 0; t < args.takes; t++) {
      const seedStr = `${args.seed}:${k}:${t}`;
      const rng = new JavaRandom(args.seed * 1000003n + BigInt(k * 97 + t));
      let take;
      try {
        take = renderTake(spec, rng, args, cfg, seedStr);
      } catch (err) {
        process.stderr.write(`${spec.id} take ${t} failed: ${err.message}\n`);
        failed++;
        continue;
      }
      if (take === null) {
        droppedTakes++;
        continue;
      }
      for (const row of windows(take, args.window, args.stride, spec, seedStr)) {
        writeSync(fd, JSON.stringify(row) + '\n');
        written++;
      }
    }
    if ((k + 1) % 10 === 0) process.stderr.write(`...${k + 1}/${specs.length} scores, ${written} rows\n`);
  }
  closeSync(fd);
  process.stderr.write(
    `wrote ${written} rows from ${specs.length} scores to ${args.out} ` +
      `(${droppedTakes} takes dropped on GT invariants, ${failed} render failures)\n`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) main();
