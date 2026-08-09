import { test } from 'node:test';
import assert from 'node:assert/strict';
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
