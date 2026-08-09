/**
 * The robustness layer: performer-error and structural perturbations over an
 * espressivo PerformanceData value, with a typed edit log that doubles as
 * alignment ground truth.
 *
 * Shared-generator contract with the mpmify ML program (2026-08-09):
 *   applyRobustness(data, config, seed) → { data, edits }
 * — a pure function; all randomness through the explicit seeded rng; every op
 * class behind a config flag (all off by default); edit ops typed
 * delete/insert/substitute/shift/restart/skip, each carrying the source note
 * reference so the log is lossless.
 *
 * Ground-truth semantics of an edited PerformanceData:
 *   - a note with id !== null is a MATCH to the score note of that xml:id
 *     (substituted pitches and shifted onsets remain matches);
 *   - a note with id === null is an INSERTION (spurious hit, or the botched
 *     first pass of a correction-restart — the replay keeps the score ids,
 *     mirroring the nASAP annotation convention that the successful pass is
 *     the aligned one);
 *   - score ids appearing in delete/skip edits are DELETIONS.
 * `editsToAlignment` in gt.mjs flattens exactly this.
 *
 * Times are the facade's milliseconds fields; symbolic date/duration are kept
 * verbatim on copies so an inserted first-pass note still tells the model what
 * score position the performer intended (never used as identity — the log is).
 */

import { makeRng, uniform, normal, chance, pick, poisson, randint } from './rng.mjs';

/** All ops off; enable per class. Rates are expected events per 100 notes. */
export const defaultConfig = Object.freeze({
  delete: { rate: 0 },
  insert: { rate: 0 },
  substitute: { rate: 0, octaveWeight: 0.15 },
  shift: { rate: 0, stdMs: 35, hesitationP: 0.15, hesitationMs: [90, 300] },
  restart: { lambda: 0, spanMs: [800, 4000], gapMs: [250, 1500], dropLastP: 0.5 },
  skip: { lambda: 0, spanMs: [500, 3000], hesitationMs: [60, 250] },
});

export const presetLight = mergeConfig({
  delete: { rate: 0.4 }, insert: { rate: 0.4 }, substitute: { rate: 0.5 }, shift: { rate: 1.5 },
});
export const presetMedium = mergeConfig({
  delete: { rate: 1.2 }, insert: { rate: 1.2 }, substitute: { rate: 1.5 }, shift: { rate: 4 },
  restart: { lambda: 0.4 }, skip: { lambda: 0.25 },
});
export const presetHeavy = mergeConfig({
  delete: { rate: 3 }, insert: { rate: 3 }, substitute: { rate: 4 }, shift: { rate: 8 },
  restart: { lambda: 1.2 }, skip: { lambda: 0.7 },
});

export function mergeConfig(partial) {
  const out = {};
  for (const key of Object.keys(defaultConfig)) {
    out[key] = { ...defaultConfig[key], ...(partial?.[key] ?? {}) };
  }
  return Object.freeze(out);
}

/**
 * @param data espressivo PerformanceData ({ ppq, parts: [{ notes: [...] }] })
 * @param config see defaultConfig; partial configs are merged over it
 * @param seed number | string
 * @returns {{ data: object, edits: object[] }} fresh data (input untouched)
 */
export function applyRobustness(data, config, seed) {
  const cfg = mergeConfig(config);
  const rng = makeRng(seed);
  const edits = [];

  // Working copy: flat per-part arrays of mutable note records.
  let parts = data.parts.map((part) => ({
    ...part,
    notes: part.notes.map((n) => cloneNote(n)),
  }));
  const totalNotes = parts.reduce((s, p) => s + p.notes.length, 0);
  if (totalNotes === 0) return { data: { ...data, parts }, edits };

  const span = timeSpan(parts);

  // ---- structural ops first (they move global time, all parts together) ----

  const restarts = poisson(rng, cfg.restart.lambda);
  for (let i = 0; i < restarts; i++) applyRestart(parts, cfg.restart, rng, edits);

  const skips = poisson(rng, cfg.skip.lambda);
  for (let i = 0; i < skips; i++) applySkip(parts, cfg.skip, rng, edits);

  // ---- local ops, deterministic scan order: part index, then note index ----

  const pDel = cfg.delete.rate / 100;
  const pIns = cfg.insert.rate / 100;
  const pSub = cfg.substitute.rate / 100;
  const pShift = cfg.shift.rate / 100;

  for (let pi = 0; pi < parts.length; pi++) {
    const part = parts[pi];
    const kept = [];
    const inserted = [];
    for (const note of part.notes) {
      // Structural copies (id null) are exempt from further identity-bearing
      // ops but still get played "again" plausibly — leave them as they are.
      const isCopy = note.id === null;

      if (!isCopy && chance(rng, pDel)) {
        edits.push({ op: 'delete', part: pi, note: cloneNote(note) });
        continue; // dropped
      }

      if (!isCopy && chance(rng, pSub)) {
        const octave = chance(rng, cfg.substitute.octaveWeight);
        const delta = octave ? pick(rng, [-12, 12]) : pick(rng, [-2, -1, 1, 2]);
        const to = clampPitch(note.pitch + delta);
        edits.push({ op: 'substitute', part: pi, id: note.id, from: note.pitch, to, kind: octave ? 'octave' : 'neighbor' });
        note.pitch = to;
      }

      if (chance(rng, pShift)) {
        const hesitate = chance(rng, cfg.shift.hesitationP);
        const delta = hesitate
          ? uniform(rng, cfg.shift.hesitationMs[0], cfg.shift.hesitationMs[1])
          : normal(rng, 0, cfg.shift.stdMs);
        edits.push({ op: 'shift', part: pi, id: note.id, deltaMs: delta });
        note.milliseconds.date += delta;
        note.milliseconds.end += delta;
      }

      if (chance(rng, pIns)) {
        // Neighbor slip: a brushed adjacent key next to this note.
        const slip = cloneNote(note);
        slip.id = null;
        slip.origin = { type: 'slip', near: note.id };
        slip.pitch = clampPitch(note.pitch + pick(rng, [-2, -1, 1, 2]));
        slip.milliseconds.date = note.milliseconds.date + uniform(rng, -30, 30);
        slip.milliseconds.end = slip.milliseconds.date + uniform(rng, 25, 90);
        slip.velocity = clampVelocity(Math.round(note.velocity * uniform(rng, 0.4, 0.75)));
        edits.push({ op: 'insert', part: pi, near: note.id, note: cloneNote(slip) });
        inserted.push(slip);
      }

      kept.push(note);
    }
    part.notes = kept.concat(inserted);
  }

  // Clamp into non-negative time (a shift can push the first onset below 0)
  // and restore the reading order.
  let minOnset = Infinity;
  for (const part of parts) for (const n of part.notes) minOnset = Math.min(minOnset, n.milliseconds.date);
  if (minOnset < 0) {
    for (const part of parts) for (const n of part.notes) {
      n.milliseconds.date -= minOnset;
      n.milliseconds.end -= minOnset;
    }
  }
  for (const part of parts) {
    part.notes.sort((x, y) => x.milliseconds.date - y.milliseconds.date || x.pitch - y.pitch);
  }

  return { data: { ...data, parts }, edits };
}

// ---------------------------------------------------------------------------

function applyRestart(parts, cfg, rng, edits) {
  const { min, max } = timeSpan(parts);
  if (!(max > min)) return;
  const len = uniform(rng, cfg.spanMs[0], cfg.spanMs[1]);
  if (max - min <= len) return;
  const t0 = uniform(rng, min, max - len);
  const t1 = t0 + len;
  const gap = uniform(rng, cfg.gapMs[0], cfg.gapMs[1]);
  const delta = len + gap;

  const copies = [];
  for (let pi = 0; pi < parts.length; pi++) {
    const part = parts[pi];
    const firstPass = [];
    for (const note of part.notes) {
      if (note.milliseconds.date >= t1) {
        // Everything after the stumble point happens later.
        note.milliseconds.date += delta;
        note.milliseconds.end += delta;
      } else if (note.milliseconds.date >= t0) {
        // Segment note: the original becomes the successful replay…
        const copy = cloneNote(note);
        // Provenance travels ON the note (not only in the log): a later
        // structural op may shift this copy in time, so position is not a
        // stable key. When this restart re-copies an earlier insertion
        // (copy-of-copy, or a slip), the score reference is inherited through
        // its origin so provenance always bottoms out at a real score id.
        const sourceId = note.id ?? note.origin?.sourceId ?? note.origin?.near ?? null;
        copy.id = null;
        copy.origin = { type: 'restart-first-pass', sourceId, t0 };
        copy.velocity = clampVelocity(Math.round(copy.velocity * uniform(rng, 0.8, 1.0)));
        firstPass.push({ sourceId, copy });
        note.milliseconds.date += delta;
        note.milliseconds.end += delta;
      } else if (note.milliseconds.end > t0) {
        // Held into the stumble: released at the break.
        note.milliseconds.end = Math.min(note.milliseconds.end, t1);
      }
    }
    // …and the copy, staying at the original time, is the botched first pass.
    // The stumble reason: the last notes before the break often go wrong —
    // drop or truncate the tail of the attempt.
    firstPass.sort((a, b) => a.copy.milliseconds.date - b.copy.milliseconds.date);
    if (firstPass.length > 0 && chance(rng, cfg.dropLastP)) {
      const dropped = firstPass.splice(firstPass.length - randint(rng, 1, Math.min(2, firstPass.length)));
      void dropped; // performer broke off before these — they never sound
    }
    for (const { sourceId, copy } of firstPass) {
      copy.milliseconds.end = Math.min(copy.milliseconds.end, t1); // released at the break
      part.notes.push(copy);
      copies.push({ part: pi, sourceId, note: cloneNote(copy) });
    }
  }
  edits.push({ op: 'restart', t0, lenMs: t1 - t0, gapMs: gap, copies });
}

function applySkip(parts, cfg, rng, edits) {
  const { min, max } = timeSpan(parts);
  if (!(max > min)) return;
  const len = uniform(rng, cfg.spanMs[0], cfg.spanMs[1]);
  if (max - min <= len * 2) return; // don't hollow out tiny pieces
  const t0 = uniform(rng, min + len / 2, max - len);
  const t1 = t0 + len;
  const hesitation = uniform(rng, cfg.hesitationMs[0], cfg.hesitationMs[1]);
  const delta = len - hesitation;

  const removed = [];
  for (let pi = 0; pi < parts.length; pi++) {
    const part = parts[pi];
    const kept = [];
    for (const note of part.notes) {
      if (note.milliseconds.date >= t0 && note.milliseconds.date < t1) {
        removed.push({ part: pi, note: cloneNote(note) });
        continue;
      }
      if (note.milliseconds.date >= t1) {
        note.milliseconds.date -= delta;
        note.milliseconds.end -= delta;
      } else if (note.milliseconds.end > t0) {
        note.milliseconds.end = t0; // released where the lapse begins
      }
      kept.push(note);
    }
    part.notes = kept;
  }
  edits.push({ op: 'skip', t0, t1, hesitationMs: hesitation, removed });
}

// ---------------------------------------------------------------------------

function cloneNote(n) {
  return { ...n, milliseconds: { ...n.milliseconds } };
}

function timeSpan(parts) {
  let min = Infinity;
  let max = -Infinity;
  for (const part of parts) {
    for (const n of part.notes) {
      min = Math.min(min, n.milliseconds.date);
      max = Math.max(max, n.milliseconds.date);
    }
  }
  return { min, max };
}

const clampPitch = (p) => Math.max(0, Math.min(127, p));
const clampVelocity = (v) => Math.max(1, Math.min(127, v));
