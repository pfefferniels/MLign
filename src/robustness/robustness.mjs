/**
 * The robustness layer: performer-error and structural perturbations over an
 * espressivo PerformanceData value, with a typed edit log that doubles as
 * alignment ground truth.
 *
 * Interface shared with the mpmify project:
 *   applyRobustness(data, config, seed) → { data, edits }
 * — a pure function; all randomness through the explicit seeded rng; every op
 * class behind a config flag (all off by default); edit ops typed
 * delete/insert/add/substitute/shift/restart/skip, each carrying the source
 * note reference so the log is lossless.
 *
 * Two unrelated models produce extra notes, and the distinction matters for
 * what the aligner learns: `insert` is the ERROR model (a brushed neighbour
 * key, quiet and chromatic) while `add` is the INTENT model of early-recording
 * piano style — octave doublings, filled-in chord tones and unwritten
 * ornaments, consonant and played at full weight.
 *
 * Ground-truth semantics of an edited PerformanceData:
 *   - a note with id !== null is a MATCH to the score note of that xml:id
 *     (substituted pitches and shifted onsets remain matches);
 *   - a note with id === null is an INSERTION (spurious hit, deliberate
 *     addition, or the botched first pass of a correction-restart — the replay
 *     keeps the score ids, mirroring the nASAP annotation convention that the
 *     successful pass is the aligned one);
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
  // Consonant added notes (see the header): rate counts EVENTS per 100 notes;
  // an octave/chord-tone event is one note, an ornament event a short figure.
  add: {
    rate: 0,
    octaveWeight: 0.45, chordToneWeight: 0.4, ornamentWeight: 0.15,
    spreadMs: 18, velScale: [0.8, 1.1],
    ornamentNotes: [2, 4], ornamentStepMs: [28, 65],
  },
  substitute: { rate: 0, octaveWeight: 0.15 },
  shift: { rate: 0, stdMs: 35, hesitationP: 0.15, hesitationMs: [90, 300] },
  restart: { lambda: 0, spanMs: [800, 4000], gapMs: [250, 1500], dropLastP: 0.5 },
  skip: { lambda: 0, spanMs: [500, 3000], hesitationMs: [60, 250] },
  // GT-neutral humanizer: Gaussian onset/offset noise on EVERY note, unlogged
  // (alignment is unchanged by it). Deterministic replacement for espressivo's
  // unseedable imprecision maps. offsetStdMs defaults to onset's std.
  //
  // `ornamentStdMs` is the WITHIN-FIGURE spread for ornament notes, which is a
  // different quantity from the note-to-note noise around it. An ornament is a
  // motor pattern: the whole figure arrives early or late as a unit, but its
  // internal timing is far tighter than the playing around it. Jittering each
  // of a trill's notes independently at the ordinary σ is not humanisation —
  // at 40 ms between notes it reorders them and erases the figure's shape. So
  // a figure draws ONE shared offset at `stdMs` and each of its notes a small
  // one at `ornamentStdMs`. null = the old behaviour (same σ as everything
  // else), kept so existing corpora still mean what they meant.
  //
  // `minDurMs` is the floor a jittered note is not allowed to fall below; the
  // offset draw is independent of the onset's, so a short note can otherwise
  // invert. 8 ms is the historical value and is not a real piano note — it is
  // where the ~11 % of sub-15 ms ornament notes in the older corpora came from.
  jitter: { stdMs: 0, offsetStdMs: null, ornamentStdMs: null, minDurMs: 8 },
});

export const presetLight = mergeConfig({
  delete: { rate: 0.4 }, insert: { rate: 0.4 }, add: { rate: 0.8 }, substitute: { rate: 0.5 },
  shift: { rate: 1.5 },
});
export const presetMedium = mergeConfig({
  delete: { rate: 1.2 }, insert: { rate: 1.2 }, add: { rate: 2.5 }, substitute: { rate: 1.5 },
  shift: { rate: 4 }, restart: { lambda: 0.4 }, skip: { lambda: 0.25 },
});
export const presetHeavy = mergeConfig({
  delete: { rate: 3 }, insert: { rate: 3 }, add: { rate: 6 }, substitute: { rate: 4 },
  shift: { rate: 8 }, restart: { lambda: 1.2 }, skip: { lambda: 0.7 },
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
  const pAdd = cfg.add.rate / 100;
  const pSub = cfg.substitute.rate / 100;
  const pShift = cfg.shift.rate / 100;

  // Harmony an addition may draw a chord tone from: the notes as they stand
  // after the structural ops, onset-sorted across parts. Built once — the
  // local ops that follow move onsets by tens of ms at most, well inside the
  // window `soundingPitchClasses` queries.
  const sounding = pAdd > 0 ? soundingIndex(parts) : null;

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

      // Consonant addition. Guarded on the rate so a disabled `add` draws no
      // random numbers at all and leaves every other op's stream untouched.
      if (!isCopy && pAdd > 0 && chance(rng, pAdd)) {
        for (const added of addition(note, cfg.add, rng, sounding)) {
          edits.push({
            op: 'add', part: pi, near: note.id, flavour: added.origin.flavour,
            slot: added.origin.slot ?? 0, note: cloneNote(added),
          });
          inserted.push(added);
        }
      }

      kept.push(note);
    }
    part.notes = kept.concat(inserted);
  }

  // GT-neutral timing jitter, after all identity-bearing ops.
  if (cfg.jitter.stdMs > 0) {
    const offStd = cfg.jitter.offsetStdMs ?? cfg.jitter.stdMs;
    const ornStd = cfg.jitter.ornamentStdMs ?? cfg.jitter.stdMs;
    const minDur = cfg.jitter.minDurMs ?? 8;
    // One shared offset per figure, drawn once and reused: the figure moves as
    // a unit. Keyed on the ornament instruction id, so a repeated ornament's
    // separate passes are separate figures.
    const figureShift = new Map();
    for (const part of parts) {
      for (const n of part.notes) {
        const ref = n.origin?.type === 'ornament' ? `${n.origin.ref}:${n.origin.pass ?? 0}` : null;
        if (ref !== null && !figureShift.has(ref)) figureShift.set(ref, normal(rng, 0, cfg.jitter.stdMs));
        const shift = ref === null ? normal(rng, 0, cfg.jitter.stdMs)
                                   : figureShift.get(ref) + normal(rng, 0, ornStd);
        n.milliseconds.date += shift;
        n.milliseconds.end += normal(rng, 0, ref === null ? offStd : ornStd);
        if (n.milliseconds.end < n.milliseconds.date + minDur) {
          n.milliseconds.end = n.milliseconds.date + minDur; // keep a sounding note
        }
      }
    }
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
        // (copy-of-copy, a slip, or a consonant addition — all three key their
        // anchor as `near`), the score reference is inherited through its
        // origin so provenance always bottoms out at a real score id.
        const sourceId = note.id ?? note.origin?.sourceId ?? note.origin?.near ?? note.origin?.anchor ?? null;
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
// Consonant added notes. The anchor is always a real score note, and every
// added note carries `near`/`anchor` back to it, so an addition is an
// attributable insertion exactly like an espressivo-generated ornament note.
// ---------------------------------------------------------------------------

const PIANO_LO = 21;
const PIANO_HI = 108;

/** The added notes for one addition event on `note` (possibly empty). */
function addition(note, cfg, rng, sounding) {
  let flavour = pick(rng, ['octave', 'chordtone', 'ornament'],
    [cfg.octaveWeight, cfg.chordToneWeight, cfg.ornamentWeight]);
  // "Unwritten" only means anything on a note carrying no ornament sign; the
  // facade flags the principals it ornamented and the figures it generated.
  if (flavour === 'ornament' && (note.ornamented || note.ornamentSlot != null)) flavour = 'octave';
  if (flavour === 'ornament') return ornamentFigure(note, cfg, rng);

  let pitch = null;
  if (flavour === 'chordtone') {
    pitch = chordTone(note, rng, sounding);
    if (pitch === null) flavour = 'octave'; // harmony offered nothing — double instead
  }
  if (flavour === 'octave') pitch = octaveDouble(note, rng);
  if (pitch === null) return [];

  // Doublings and fills are struck with the anchor and held with it.
  const added = cloneNote(note);
  added.id = null;
  added.origin = { type: 'addition', near: note.id, anchor: note.id, flavour };
  added.pitch = pitch;
  const dt = uniform(rng, -cfg.spreadMs, cfg.spreadMs);
  added.milliseconds.date += dt;
  added.milliseconds.end += dt;
  added.velocity = scaleVelocity(note.velocity, cfg, rng);
  return [added];
}

/** Octave doubling: down for the bass, up for the treble, always in range. */
function octaveDouble(note, rng) {
  const down = note.pitch - 12 >= PIANO_LO;
  const up = note.pitch + 12 <= PIANO_HI;
  if (!down && !up) return null;
  if (!down) return note.pitch + 12;
  if (!up) return note.pitch - 12;
  // The bass octave is the idiom; the treble doubling is the rarer one.
  return chance(rng, note.pitch < 60 ? 0.8 : 0.35) ? note.pitch - 12 : note.pitch + 12;
}

/** Thirds, sixths and octaves, from the anchor, in either direction. */
const CHORD_STEPS = [3, 4, 8, 9, 12];

/**
 * A chord tone filled in above or below the anchor, its pitch class taken from
 * what actually sounds at the anchor's onset — never from a guessed key.
 * null when the sounding harmony offers nothing.
 */
function chordTone(note, rng, sounding) {
  const pcs = soundingPitchClasses(sounding, note.milliseconds.date);
  const cands = [];
  for (const step of CHORD_STEPS) {
    for (const p of [note.pitch - step, note.pitch + step]) {
      if (p < PIANO_LO || p > PIANO_HI) continue;
      if (pcs.has(p % 12)) cands.push(p);
    }
  }
  if (cands.length === 0) return null;
  // The octave is trivially available (it is the anchor's own class) — take a
  // real third or sixth whenever the harmony has one.
  const inner = cands.filter((p) => Math.abs(p - note.pitch) !== 12);
  return pick(rng, inner.length > 0 ? inner : cands);
}

/**
 * A written-out ornament on a note that carries no ornament sign: an
 * upper-neighbour alternation (trill/mordent), struck after the principal, or
 * a turn figure led into the beat. The anchor keeps sounding — these are only
 * the extra notes, so slot 0 is the first EXTRA, not the principal.
 */
function ornamentFigure(note, cfg, rng) {
  const upper = randint(rng, 1, 2); // chromatic 1–2 ≈ the diatonic neighbour
  const turn = chance(rng, 0.35);
  const offsets = turn
    ? [upper, 0, -upper] // …resolving onto the anchor
    : Array.from({ length: randint(rng, cfg.ornamentNotes[0], cfg.ornamentNotes[1]) },
      (_, k) => (k % 2 === 0 ? upper : 0));
  const step = uniform(rng, cfg.ornamentStepMs[0], cfg.ornamentStepMs[1]);

  const out = [];
  offsets.forEach((semitones, slot) => {
    const added = cloneNote(note);
    added.id = null;
    added.origin = {
      type: 'addition', near: note.id, anchor: note.id, flavour: 'ornament',
      ref: turn ? 'turn' : 'alternation', slot, pass: 0,
    };
    added.pitch = clampPitch(note.pitch + semitones);
    // Turns lead into the anchor, alternations follow its attack.
    added.milliseconds.date = turn
      ? note.milliseconds.date - (offsets.length - slot) * step
      : note.milliseconds.date + (slot + 1) * step;
    added.milliseconds.end = added.milliseconds.date + Math.max(20, step * uniform(rng, 0.55, 0.95));
    added.velocity = scaleVelocity(note.velocity, cfg, rng);
    out.push(added);
  });
  return out;
}

/** Onset-sorted flat view of the notes, for the harmony query. */
function soundingIndex(parts) {
  const all = [];
  for (const part of parts) {
    for (const n of part.notes) {
      all.push({ date: n.milliseconds.date, end: n.milliseconds.end, pitch: n.pitch });
    }
  }
  all.sort((x, y) => x.date - y.date);
  return all;
}

/** Struck within `SOUND_TOL` of t (asynchrony) or still held over it. */
const SOUND_TOL = 25;
function soundingPitchClasses(sounding, t) {
  const pcs = new Set();
  for (const n of sounding) {
    if (n.date > t + SOUND_TOL) break; // onset-sorted
    if (n.end > t || t - n.date <= SOUND_TOL) pcs.add(n.pitch % 12);
  }
  return pcs;
}

const scaleVelocity = (velocity, cfg, rng) =>
  clampVelocity(Math.round(velocity * uniform(rng, cfg.velScale[0], cfg.velScale[1])));

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
