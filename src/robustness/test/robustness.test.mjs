import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { applyRobustness, presetMedium, presetHeavy, mergeConfig } from '../robustness.mjs';
import { editsToAlignment } from '../gt.mjs';

/** Two-part grid piece: `n` notes per part, 250 ms apart, 720 ppq quarters. */
function makePiece(n = 40) {
  const part = (prefix, basePitch) => ({
    name: prefix,
    notes: Array.from({ length: n }, (_, i) => ({
      id: `${prefix}${i}`,
      pitch: basePitch + (i % 12),
      date: i * 720,
      duration: 720,
      velocity: 80,
      milliseconds: { date: i * 250, end: i * 250 + 230 },
    })),
  });
  return { ppq: 720, parts: [part('a', 60), part('b', 40)] };
}

const scoreIds = (data) => data.parts.flatMap((p) => p.notes.map((n) => n.id));

test('no-op config returns identical notes and no edits', () => {
  const piece = makePiece();
  const { data, edits } = applyRobustness(piece, {}, 42);
  assert.equal(edits.length, 0);
  assert.deepEqual(
    data.parts.map((p) => p.notes),
    piece.parts.map((p) => p.notes),
  );
});

test('same seed → identical output; different seed → different output', () => {
  const piece = makePiece();
  const a = applyRobustness(piece, presetHeavy, 'seed-1');
  const b = applyRobustness(piece, presetHeavy, 'seed-1');
  const c = applyRobustness(piece, presetHeavy, 'seed-2');
  assert.deepEqual(a, b);
  assert.notDeepEqual(JSON.stringify(a), JSON.stringify(c));
});

test('input data is never mutated', () => {
  const piece = makePiece();
  const snapshot = JSON.stringify(piece);
  applyRobustness(piece, presetHeavy, 7);
  assert.equal(JSON.stringify(piece), snapshot);
});

test('identity conservation: every score id is matched XOR deleted, every perf note matched XOR inserted', () => {
  for (const seed of [1, 2, 3, 4, 5]) {
    const piece = makePiece(60);
    const { data, edits } = applyRobustness(piece, presetHeavy, seed);
    const { alignment, perfNotes, unattributed } = editsToAlignment(data, edits);

    assert.equal(unattributed, 0, `seed ${seed}: unattributed insertions`);

    const matches = alignment.filter((r) => r.label === 'match');
    const insertions = alignment.filter((r) => r.label === 'insertion');
    const deletions = alignment.filter((r) => r.label === 'deletion');

    // Perf side: every performed note appears exactly once.
    assert.equal(matches.length + insertions.length, perfNotes.length, `seed ${seed}`);
    const perfIds = new Set([...matches, ...insertions].map((r) => r.perfId));
    assert.equal(perfIds.size, perfNotes.length, `seed ${seed}: duplicate perfIds`);

    // Score side: matched ∪ deleted = original ids, disjoint, no duplicates.
    const matchedIds = matches.map((r) => r.scoreId);
    const deletedIds = deletions.map((r) => r.scoreId);
    const union = [...matchedIds, ...deletedIds].sort();
    const original = scoreIds(piece).sort();
    assert.deepEqual(union, original, `seed ${seed}: score ids not conserved`);
  }
});

test('substitution keeps the match and records from/to', () => {
  const piece = makePiece(80);
  const cfg = mergeConfig({ substitute: { rate: 30 } });
  const { data, edits } = applyRobustness(piece, cfg, 11);
  const subs = edits.filter((e) => e.op === 'substitute');
  assert.ok(subs.length > 5, `expected many substitutions, got ${subs.length}`);
  const { alignment } = editsToAlignment(data, edits);
  for (const sub of subs) {
    const rec = alignment.find((r) => r.label === 'match' && r.scoreId === sub.id);
    assert.ok(rec, `substituted ${sub.id} must stay matched`);
    assert.deepEqual(rec.sub, { from: sub.from, to: sub.to, kind: sub.kind });
    assert.notEqual(sub.from, sub.to);
  }
});

test('restart extends the timeline and preserves all score ids as matches', () => {
  const piece = makePiece(60);
  const cfg = mergeConfig({ restart: { lambda: 5 } }); // force ≥1 with high prob.
  const { data, edits } = applyRobustness(piece, cfg, 3);
  const restarts = edits.filter((e) => e.op === 'restart');
  assert.ok(restarts.length >= 1, 'no restart sampled');

  const { alignment } = editsToAlignment(data, edits);
  const matchedIds = alignment.filter((r) => r.label === 'match').map((r) => r.scoreId).sort();
  assert.deepEqual(matchedIds, scoreIds(piece).sort());

  const copies = restarts.flatMap((r) => r.copies);
  const insertions = alignment.filter((r) => r.label === 'insertion');
  assert.equal(insertions.length, copies.length);
  for (const ins of insertions) assert.equal(ins.provenance.type, 'restart-first-pass');
  for (const ins of insertions) assert.ok(ins.provenance.sourceId, 'copy must reference its source');

  const lastOriginal = 59 * 250;
  const maxOnset = Math.max(...data.parts.flatMap((p) => p.notes.map((n) => n.milliseconds.date)));
  assert.ok(maxOnset > lastOriginal, 'timeline must extend past original end');
});

test('skip removes a span, closes the gap, and reports deletions', () => {
  const piece = makePiece(60);
  const cfg = mergeConfig({ skip: { lambda: 5 } });
  const { data, edits } = applyRobustness(piece, cfg, 9);
  const skips = edits.filter((e) => e.op === 'skip');
  assert.ok(skips.length >= 1, 'no skip sampled');

  const removedIds = skips.flatMap((s) => s.removed.map((r) => r.note.id)).filter((id) => id !== null);
  assert.ok(removedIds.length > 0, 'skip removed nothing');

  const { alignment } = editsToAlignment(data, edits);
  const deletedIds = alignment.filter((r) => r.label === 'deletion').map((r) => r.scoreId);
  for (const id of removedIds) assert.ok(deletedIds.includes(id), `${id} must be a deletion`);

  const outIds = new Set(data.parts.flatMap((p) => p.notes.map((n) => n.id)));
  for (const id of removedIds) assert.ok(!outIds.has(id), `${id} must be gone from the data`);

  // Gap closed: total span shrinks by roughly the skipped length minus hesitation.
  const maxOnset = Math.max(...data.parts.flatMap((p) => p.notes.map((n) => n.milliseconds.date)));
  assert.ok(maxOnset < 59 * 250, 'timeline must contract');
});

test('onsets never go negative and parts stay onset-sorted', () => {
  for (const seed of [21, 22, 23]) {
    const { data } = applyRobustness(makePiece(50), presetMedium, seed);
    for (const part of data.parts) {
      let prev = -Infinity;
      for (const n of part.notes) {
        assert.ok(n.milliseconds.date >= 0, 'negative onset');
        assert.ok(n.milliseconds.date >= prev, 'unsorted part');
        prev = n.milliseconds.date;
      }
    }
  }
});

test('shiftToMatchedZero puts the first matched onset at 0 with earlier insertions negative', async () => {
  const { shiftToMatchedZero } = await import('../gt.mjs');
  const piece = makePiece(30);
  const cfg = mergeConfig({ restart: { lambda: 6 }, insert: { rate: 8 } });
  const { data, edits } = applyRobustness(piece, cfg, 17);
  const { alignment, perfNotes } = editsToAlignment(data, edits);
  const shifted = shiftToMatchedZero(perfNotes, alignment);
  const matched = new Set(alignment.filter((r) => r.label === 'match').map((r) => r.perfId));
  const firstMatched = shifted.find((n) => matched.has(n.perfId));
  assert.equal(firstMatched.onsetMs, 0);
  for (let i = 1; i < shifted.length; i++) {
    assert.ok(shifted[i].onsetMs >= shifted[i - 1].onsetMs, 'order preserved');
  }
  assert.ok(perfNotes.every((n) => n.onsetMs >= 0), 'absolute emission stays non-negative');
});

test('jitter moves timings but leaves the alignment GT untouched', () => {
  const piece = makePiece(40);
  const base = applyRobustness(piece, { delete: { rate: 2 }, insert: { rate: 2 } }, 31);
  const jit = applyRobustness(piece, { delete: { rate: 2 }, insert: { rate: 2 }, jitter: { stdMs: 25 } }, 31);
  const a = editsToAlignment(base.data, base.edits);
  const b = editsToAlignment(jit.data, jit.edits);
  // perfIds are onset-ordinal and may legitimately reorder under jitter;
  // compare the identity-stable views instead.
  const view = (x) => ({
    matches: x.alignment.filter((r) => r.label === 'match').map((r) => r.scoreId).sort(),
    deletions: x.alignment.filter((r) => r.label === 'deletion').map((r) => r.scoreId).sort(),
    insertions: x.alignment
      .filter((r) => r.label === 'insertion')
      .map((r) => `${r.provenance.type}:${r.provenance.near ?? r.provenance.sourceId ?? ''}`)
      .sort(),
  });
  assert.deepEqual(view(a), view(b));
  const onsetsA = a.perfNotes.map((n) => n.onsetMs);
  const onsetsB = b.perfNotes.map((n) => n.onsetMs);
  assert.notDeepEqual(onsetsA, onsetsB, 'jitter must move onsets');
  for (const n of b.perfNotes) assert.ok(n.offsetMs >= n.onsetMs + 8);
});

// --- consonant added notes -------------------------------------------------

/**
 * Block-triad piece: root/third/fifth per beat in part a over a bass in part
 * b, 250 ms apart and released before the next beat — so the notes sounding at
 * a beat are exactly the ones struck on it, and the expected pitch-class set
 * is computable here without duplicating the layer's window logic.
 */
function makeChordPiece(n = 40) {
  const ROOTS = [60, 65, 67, 62];
  const a = [];
  const b = [];
  for (let i = 0; i < n; i++) {
    const root = ROOTS[i % ROOTS.length];
    const ms = { date: i * 250, end: i * 250 + 230 };
    [0, 4, 7].forEach((iv, k) => {
      a.push({
        id: `a${i}_${k}`, pitch: root + iv, date: i * 720, duration: 720, velocity: 78,
        milliseconds: { ...ms },
      });
    });
    b.push({
      id: `b${i}`, pitch: root - 24, date: i * 720, duration: 720, velocity: 70,
      milliseconds: { ...ms },
    });
  }
  return { ppq: 720, parts: [{ name: 'a', notes: a }, { name: 'b', notes: b }] };
}

/** Additions only, on every note, with nothing else moving the timeline. */
const addOnly = (over = {}) => mergeConfig({ add: { rate: 100, ...over } });

const noteById = (piece) =>
  new Map(piece.parts.flatMap((p) => p.notes.map((n) => [n.id, n])));

/** Pitch classes struck at t — the fixture has no notes held across a beat. */
function pcsAt(piece, t) {
  const pcs = new Set();
  for (const part of piece.parts) {
    for (const n of part.notes) if (n.milliseconds.date === t) pcs.add(n.pitch % 12);
  }
  return pcs;
}

test('additions are consonant: octaves are exactly ±12, chord tones sound in the harmony', () => {
  const piece = makeChordPiece(40);
  const byId = noteById(piece);
  const seen = new Set();
  for (const seed of [1, 2, 3]) {
    const { edits } = applyRobustness(piece, addOnly(), seed);
    const adds = edits.filter((e) => e.op === 'add');
    assert.ok(adds.length > 50, `expected many additions, got ${adds.length}`);
    for (const add of adds) {
      const anchor = byId.get(add.near);
      assert.ok(anchor, 'addition must name a real score note');
      seen.add(add.flavour);
      const interval = add.note.pitch - anchor.pitch;
      if (add.flavour === 'octave') {
        assert.equal(Math.abs(interval), 12, `octave doubling at ${interval} semitones`);
      } else if (add.flavour === 'chordtone') {
        assert.ok([3, 4, 8, 9, 12].includes(Math.abs(interval)), `chord tone at ${interval}`);
        const pcs = pcsAt(piece, anchor.milliseconds.date);
        assert.ok(pcs.has(add.note.pitch % 12), `chord tone ${add.note.pitch} not in the harmony`);
      } else {
        assert.equal(add.flavour, 'ornament');
        assert.ok(Math.abs(interval) <= 2, `ornament note at ${interval} semitones`);
        const dt = add.note.milliseconds.date - anchor.milliseconds.date;
        assert.ok(Math.abs(dt) < 300, `ornament note ${dt} ms from its anchor`);
        assert.ok(add.note.milliseconds.end > add.note.milliseconds.date, 'ornament note must sound');
      }
      // Played at the anchor's weight, never at the slip's ×0.4–0.75.
      assert.ok(add.note.velocity >= Math.floor(anchor.velocity * 0.8), 'addition too quiet');
      assert.ok(add.note.velocity <= Math.ceil(anchor.velocity * 1.1), 'addition too loud');
    }
  }
  assert.deepEqual([...seen].sort(), ['chordtone', 'octave', 'ornament']);
});

test('octave and chord-tone additions land essentially with their anchor', () => {
  const piece = makeChordPiece(40);
  const byId = noteById(piece);
  const { edits } = applyRobustness(piece, addOnly(), 4);
  for (const add of edits.filter((e) => e.op === 'add' && e.flavour !== 'ornament')) {
    const anchor = byId.get(add.near);
    const dt = add.note.milliseconds.date - anchor.milliseconds.date;
    assert.ok(Math.abs(dt) <= 18, `doubling ${dt} ms off its anchor`);
    const held = add.note.milliseconds.end - add.note.milliseconds.date;
    assert.equal(held, anchor.milliseconds.end - anchor.milliseconds.date, 'doubling must be held with the anchor');
  }
});

test('every added note is attributable and stays out of unattributed', () => {
  const piece = makeChordPiece(30);
  const scoreIdSet = new Set(scoreIds(piece));
  for (const seed of [11, 12, 13]) {
    const { data, edits } = applyRobustness(piece, addOnly(), seed);
    const { alignment, unattributed } = editsToAlignment(data, edits);
    assert.equal(unattributed, 0, `seed ${seed}: unattributed additions`);

    const provenances = alignment
      .filter((r) => r.label === 'insertion' && r.provenance.type === 'addition')
      .map((r) => r.provenance);
    assert.equal(provenances.length, edits.filter((e) => e.op === 'add').length);
    for (const prov of provenances) {
      assert.ok(scoreIdSet.has(prov.near), `near ${prov.near} is not a score id`);
      assert.equal(prov.anchor, prov.near, 'anchor mirrors near for the orn channel');
      assert.ok(['octave', 'chordtone', 'ornament'].includes(prov.flavour));
      if (prov.flavour === 'ornament') {
        assert.equal(typeof prov.slot, 'number');
        assert.equal(prov.pass, 0);
      }
    }
    // Ornament figures number their slots 0..k-1 within one anchor.
    const figures = new Map();
    for (const prov of provenances.filter((p) => p.flavour === 'ornament')) {
      if (!figures.has(prov.near)) figures.set(prov.near, []);
      figures.get(prov.near).push(prov.slot);
    }
    for (const [near, slots] of figures) {
      assert.deepEqual(slots.sort((x, y) => x - y), slots.map((_, i) => i), `figure on ${near}`);
    }
  }
});

test('additions under the presets keep the alignment total', () => {
  for (const seed of [31, 32, 33]) {
    const piece = makeChordPiece(30);
    const { data, edits } = applyRobustness(piece, presetHeavy, seed);
    const { alignment, perfNotes, unattributed } = editsToAlignment(data, edits);
    assert.equal(unattributed, 0);
    const matches = alignment.filter((r) => r.label === 'match');
    const insertions = alignment.filter((r) => r.label === 'insertion');
    assert.equal(matches.length + insertions.length, perfNotes.length);
    const added = insertions.filter((r) => r.provenance.type === 'addition');
    assert.ok(added.length > 0, `seed ${seed}: preset heavy produced no additions`);
  }
});

test('a restart re-copying an addition inherits the anchor score id', () => {
  // Additions already present in the input (a second robustness pass, or a
  // corpus builder that pre-added notes) key their anchor as `near`, the same
  // field the restart copier reads for slips.
  const withAddition = () => {
    const piece = makeChordPiece(40);
    piece.parts[0].notes.push({
      id: null, pitch: piece.parts[0].notes[30].pitch + 12, date: 10 * 720, duration: 720,
      velocity: 78, milliseconds: { date: 10 * 250 + 5, end: 10 * 250 + 235 },
      origin: { type: 'addition', near: 'a10_0', anchor: 'a10_0', flavour: 'octave' },
    });
    return piece;
  };
  let copiedTheAddition = false;
  for (let seed = 1; seed <= 12; seed++) {
    const { data, edits } = applyRobustness(withAddition(), mergeConfig({ restart: { lambda: 6 } }), seed);
    for (const copy of edits.filter((e) => e.op === 'restart').flatMap((e) => e.copies)) {
      assert.ok(copy.sourceId !== null, 'a first-pass copy must reference a score id');
      if (copy.sourceId === 'a10_0' && copy.note.origin.type === 'restart-first-pass') {
        copiedTheAddition = true;
      }
    }
    assert.equal(editsToAlignment(data, edits).unattributed, 0, `seed ${seed}`);
  }
  assert.ok(copiedTheAddition, 'no restart ever re-copied the pre-existing addition');
});

test('additions are deterministic under a fixed seed', () => {
  const piece = makeChordPiece(40);
  const a = applyRobustness(piece, addOnly(), 'add-seed');
  const b = applyRobustness(piece, addOnly(), 'add-seed');
  const c = applyRobustness(piece, addOnly(), 'add-seed-2');
  assert.deepEqual(a, b);
  assert.notEqual(JSON.stringify(a), JSON.stringify(c));
  // …including the presets, whose add rates are part of the seeded stream.
  assert.deepEqual(
    applyRobustness(piece, presetMedium, 77),
    applyRobustness(piece, presetMedium, 77),
  );
});

test('add.rate 0 draws no random numbers: output is byte-identical to the pre-add layer', () => {
  const cfg = mergeConfig({
    delete: { rate: 2 }, insert: { rate: 2 }, substitute: { rate: 2 }, shift: { rate: 5 },
    restart: { lambda: 1 }, skip: { lambda: 0.5 }, jitter: { stdMs: 10 },
  });
  const out = applyRobustness(makePiece(60), cfg, 'golden-1');
  assert.equal(out.edits.filter((e) => e.op === 'add').length, 0);
  // Recorded from the implementation as it stood before `add` existed.
  assert.equal(
    createHash('sha256').update(JSON.stringify(out)).digest('hex'),
    'dafac54070cff06e4b509f2bd8b4686ba17026af76575c75aad49af71f4238d5',
  );
  // The weights and spreads are inert while the rate is 0.
  const loud = mergeConfig({ ...cfg, add: { rate: 0, octaveWeight: 1, spreadMs: 400 } });
  assert.deepEqual(applyRobustness(makePiece(60), loud, 'golden-1'), out);
});
