/**
 * Ornament realization, checked by actually rendering it.
 *
 *   node --test test/*.test.mjs
 *
 * (the glob, not `node --test test/` — this node resolves a bare directory as a
 * module path and dies before it discovers anything)
 *
 * These are the invariants the attribution ground truth rests on. A figure that
 * renders but loses its anchor, or a principal that stops being a match, is not
 * a worse training example — it is a wrong label, and it looks exactly like a
 * right one in the corpus file.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import { JavaRandom } from '/Users/nielspfeffer/Projects/fenby/node/java_random.mjs';
import { buildMsm, buildMpm } from '/Users/nielspfeffer/Projects/fenby/node/xml.mjs';
import { performMsmToData } from '/Users/nielspfeffer/Projects/meico-ts/dist/api/index.js';
import {
  DEFAULTS,
  buildOrnamentation,
  injectOrnaments,
  normalizeOrnaments,
} from '../src/corpus/ornaments.mjs';

const PPQ = 720;
// Four half notes and, at the end, a three-note chord to arpeggiate.
const NOTES = [
  { date: 0, dur: 1440, pitch: 60 },
  { date: 1440, dur: 1440, pitch: 64 },
  { date: 2880, dur: 1440, pitch: 67 },
  { date: 4320, dur: 1440, pitch: 55 },
  { date: 4320, dur: 1440, pitch: 59 },
  { date: 4320, dur: 1440, pitch: 62 },
];
const PARTS = [{ name: 'Piano', number: 1, midiChannel: 0, midiPort: 0, notes: NOTES, asynchrony: [], articulation: [] }];
const MSM = buildMsm('t', 't', PPQ, PARTS);
const MAPS = {
  tempo: [{ date: 0, bpm: 100, meterNumerator: 4, transitionTo: null }],
  dynamics: [], articulation: [], rubato: [], movement: [], accentuation: null,
};
const SCORE_IDS = new Set(NOTES.map((_, i) => `p1n${i}`));

/** Render one request list and hand back the normalized performance. */
function render(requests, seed = 7, breadth = 1.5) {
  const orn = buildOrnamentation(requests, new JavaRandom(BigInt(seed)), { ...DEFAULTS, breadth });
  const mpm = injectOrnaments(buildMpm('perf', PPQ, MAPS, PARTS), orn);
  const { data, dropped } = normalizeOrnaments(performMsmToData({ msm: MSM, mpm }), SCORE_IDS);
  return { notes: data.parts[0].notes, dropped, orn };
}

const GENERATING = ['trill', 'mordent', 'inverted-mordent', 'turn', 'inverted-turn', 'delayed-turn', 'grace', 'tremolo'];

function requestFor(kind, index = 0) {
  return {
    msmId: 'p1n1', date: 1440, durQuarters: 2, pitch: 64, kind, index,
    gracePitches: [62], slashed: false, beams: 3,
  };
}

for (const kind of GENERATING) {
  test(`${kind}: every generated note names its principal`, () => {
    const { notes } = render([requestFor(kind)]);
    const generated = notes.filter((n) => n.id === null);
    assert.ok(generated.length >= 1, `${kind} generated nothing`);
    for (const n of generated) {
      assert.equal(n.origin.type, 'ornament');
      assert.equal(n.origin.anchor, 'p1n1', `${kind} slot ${n.origin.slot} lost its anchor`);
    }
  });

  test(`${kind}: the principal stays a match (D10)`, () => {
    const { notes } = render([requestFor(kind)]);
    // Every score note must still be present under its own id — an ornament may
    // move and shorten its principal, never consume it.
    for (const id of SCORE_IDS)
      assert.ok(notes.some((n) => n.id === id), `${kind} swallowed ${id}`);
  });

  test(`${kind}: no note of zero length survives`, () => {
    for (let seed = 1; seed <= 40; seed++) {
      const { notes } = render([requestFor(kind)], seed, 2.5);
      for (const n of notes)
        assert.ok(Number(n.milliseconds.end) - Number(n.milliseconds.date) > 0,
          `${kind} seed ${seed}: zero-length note ${n.id}`);
    }
  });
}

test('trill: length follows the principal, and the figure accelerates', () => {
  const short = render([{ ...requestFor('trill'), durQuarters: 0.5 }], 3);
  const long = render([{ ...requestFor('trill'), durQuarters: 4 }], 3);
  const count = (r) => r.notes.filter((n) => n.id === null).length;
  assert.ok(count(long) > count(short),
    `a 4-quarter trill (${count(long)}) must have more notes than a half-quarter one (${count(short)})`);

  // Gaps shrink: the mean of the second half is shorter than of the first.
  const figure = long.notes
    .filter((n) => n.origin?.anchor === 'p1n1' || n.id === 'p1n1')
    .map((n) => Number(n.milliseconds.date))
    .sort((a, b) => a - b);
  const gaps = figure.slice(1).map((d, i) => d - figure[i]);
  const half = Math.floor(gaps.length / 2);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  assert.ok(mean(gaps.slice(half)) < mean(gaps.slice(0, half)),
    `trill should speed up: first half ${mean(gaps.slice(0, half)).toFixed(1)} ms, second ${mean(gaps.slice(half)).toFixed(1)} ms`);
});

test('grace: the notated pitch is the pitch that sounds', () => {
  const { notes } = render([{ ...requestFor('grace'), gracePitches: [61] }]);
  const generated = notes.filter((n) => n.id === null);
  assert.equal(generated.length, 1);
  assert.equal(generated[0].pitch, 61);
});

test('arpeggio: spreads the chord and generates nothing', () => {
  const req = { msmId: 'p1n3', date: 4320, kind: 'arpeggio', index: 0,
                chordIds: ['p1n3', 'p1n4', 'p1n5'] };
  const { notes, orn } = render([req]);
  assert.ok(!orn.map.includes('noteid='), 'an arpeggio must stay a v2 ornament');
  assert.equal(notes.filter((n) => n.id === null).length, 0, 'an arpeggio generates no notes');
  assert.equal(notes.length, NOTES.length, 'the chord must not be doubled');
  const at = (id) => Number(notes.find((n) => n.id === id).milliseconds.date);
  const [a, b, c] = ['p1n3', 'p1n4', 'p1n5'].map(at);
  assert.ok(a !== b && b !== c, `the chord did not spread: ${a} ${b} ${c}`);
});

test('a piece with no requests renders unchanged', () => {
  const { notes, orn } = render([]);
  assert.equal(orn.map, '');
  assert.equal(notes.length, NOTES.length);
  assert.equal(notes.filter((n) => n.id === null).length, 0);
});

test('tremolo: repeats the written note, faster with more beams', () => {
  const count = (b) => {
    const { notes } = render([{ ...requestFor('tremolo'), beams: b }], 5);
    return notes.filter((n) => n.id === null).length;
  };
  const [one, three] = [count(1), count(3)];
  assert.ok(three > one, `3 beams (${three}) must be denser than 1 beam (${one})`);
  // Every generated note is the principal's own pitch — that is what a
  // single-note tremolo is, and espressivo's dedup rule spares it only because
  // every slot shares one pitch.
  const { notes } = render([requestFor('tremolo')], 5);
  for (const n of notes.filter((x) => x.id === null)) assert.equal(n.pitch, 64);
});
