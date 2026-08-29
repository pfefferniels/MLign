/**
 * Ornament realization: notated sign → MPM v3 `<ornamentDef>` + `<ornament>`.
 *
 * Shared by the sampled-piece generator (scripts/corpus/generate.mjs) and the
 * real-score one, so both corpora contain the same *kind* of figure and only
 * the placement differs.
 *
 * What changed against the first version of this code, and why:
 *
 * - **Length follows the principal.** A flat 2–6 alternation pairs regardless
 *   of the note being ornamented is the one thing no real trill does: a trill
 *   on a half note has many more alternations than one on an eighth. Pairs are
 *   now `dur_quarters × pairsPerQuarter`, clamped.
 * - **Figures accelerate.** `<temporalSpread intensity>` is a real MPM
 *   attribute (espressivo TemporalSpread.ts:267) and the layout it drives is
 *   `pow(i / (n-1), intensity) * length + start` (ornamentInstantiation.ts:746),
 *   so `intensity < 1` crowds the end — a figure that starts slow and speeds
 *   up, which is what players actually do. Uniform spacing (`intensity = 1`)
 *   was previously the only shape in the corpus.
 * - **Trills may start on the upper note** and may end with a termination
 *   (Nachschlag), both of which are the norm in the repertoire rather than
 *   exotic variants.
 * - **Grace notes exist.** They are 6.3 per 1000 notes in ASAP — a third of
 *   all notated ornament signs — and the corpus had none at all.
 *
 * MusicXML naming trap, preserved deliberately: `<mordent>` is the LOWER
 * (main-auxiliary-main below) figure and `<inverted-mordent>` is the upper one,
 * the Pralltriller — the opposite of what the words suggest to most readers.
 */

/** Realization knobs. `breadth` ≥ 1 widens everything toward early-recording style. */
export const DEFAULTS = {
  breadth: 1,
  // Alternation pairs per quarter of the principal's notated value. A trill at
  // ~120 bpm with 4 pairs/quarter runs at 16 notes/s, the fast end of real
  // playing; the sampled range spans the plausible band rather than a point.
  pairsPerQuarter: [1.6, 4.2],
  maxPairs: 14,
  // < 1 accelerates. Real trills rarely decelerate, so the range sits below 1
  // with only a little room above.
  intensity: [0.55, 1.1],
  upperStartProb: 0.55,
  terminationProb: 0.45,
  anticipationProb: 0.15,
  gracePrincipalShare: [0.25, 0.55], // appoggiatura: how much of the principal it takes
  acciaccaturaShare: [0.06, 0.16],
};

const lerp = (rng, [lo, hi]) => lo + (hi - lo) * rng.nextDouble();

/** Bias a draw toward the wide end without ever leaving the range. */
const wide = (rng, [lo, hi], breadth) => lo + (hi - lo) * Math.pow(rng.nextDouble(), 1 / breadth);

const f1 = (v) => v.toFixed(1);

/**
 * One `<temporalSpread>`.
 *
 * `frameLength` is a percentage of the principal's duration, so a figure can
 * never outrun the note it decorates however wide `breadth` gets. The offset is
 * in ticks against a percentage length — the spec's own figure-3 combination,
 * which espressivo reads (ornamentInstantiation.ts:473).
 */
function spread(rng, cfg, { lengthPct, intensity, anticipate = null, monophonic = 0.8, alignment = 'at start' }) {
  // An `at end` figure already sits in the tail of its note; pulling it earlier
  // as well pushes its last slot past the note's end, where `noteoff.shift`
  // clips it to zero length. That is the carved-head cliff, and it is simply a
  // contradiction in terms — an anticipated *delayed* ornament.
  const p = alignment === 'at end' ? 0 : anticipate ?? cfg.anticipationProb + 0.2 * (cfg.breadth - 1);
  const anticipated = p > rng.nextDouble();
  const offset = anticipated ? -Math.round(rng.nextDouble() * 180 * cfg.breadth) : 0;
  const shift = rng.nextDouble() < monophonic ? ' noteoff.shift="monophonic"' : '';
  const int = intensity === 1 ? '' : ` intensity="${intensity.toFixed(3)}"`;
  return (
    `<temporalSpread frame.offset="${f1(offset)}ticks"` +
    ` frameLength="${lengthPct.toFixed(0)}%"${int}${shift} />`
  );
}

/**
 * The pitches of a figure, as chromatic intervals from the principal.
 *
 * A diatonic neighbour would be the musically right thing and MPM can express
 * it (`interval.diatonic`, resolved against the MSM key signature), but only
 * where the MSM carries a key signature — fenby's `buildMsm` writes an empty
 * `<keySignatureMap />`. Until the real-score path writes one, the neighbour is
 * drawn as 1 or 2 semitones, which is the interval a diatonic neighbour has
 * anyway; it is simply not guaranteed to be the *right* one of the two.
 */
function neighbour(rng) {
  return 1 + rng.nextInt(2);
}

/**
 * A trill: alternations between the principal and its upper neighbour, as many
 * as the principal's value affords, accelerating, optionally starting on the
 * upper note and optionally with a termination.
 */
function trill(rng, cfg, { id, durQuarters }) {
  const up = neighbour(rng);
  const perQuarter = lerp(rng, cfg.pairsPerQuarter) * cfg.breadth;
  const pairs = Math.max(2, Math.min(cfg.maxPairs, Math.round(durQuarters * perQuarter)));
  const upperFirst = rng.nextDouble() < cfg.upperStartProb;

  const order = [];
  for (let i = 0; i < pairs; i++) order.push(upperFirst ? `#u #${id}` : `#${id} #u`);
  const pool = [`<note xml:id="u" interval.chromatic="${up}.0" />`];
  // Termination: the figure turns through the LOWER neighbour before landing
  // back on the principal. Written out rather than left to the renderer's
  // landing rule, which only repeats a group's first slot.
  if (rng.nextDouble() < cfg.terminationProb) {
    const down = neighbour(rng);
    pool.push(`<note xml:id="l" interval.chromatic="-${down}.0" />`);
    order.push(`#l #${id}`);
  }
  const alignment = rng.nextDouble() < 0.15 ? 'at end' : 'at start';
  return {
    pool: pool.join(''),
    order: order.join(' '),
    spread: spread(rng, cfg, {
      // A trill fills its note; only a very short one leaves room to spare.
      lengthPct: wide(rng, [70, 100], cfg.breadth),
      intensity: lerp(rng, cfg.intensity),
      alignment,
    }),
    alignment,
  };
}

/** Mordent — three slots, fast, on the beat. `upper` picks Pralltriller vs. lower mordent. */
function mordent(rng, cfg, { id }, upper) {
  const step = neighbour(rng);
  const sign = upper ? '' : '-';
  return {
    pool: `<note xml:id="u" interval.chromatic="${sign}${step}.0" />`,
    order: `#${id} #u #${id}`,
    spread: spread(rng, cfg, {
      lengthPct: wide(rng, [12, 40], cfg.breadth),
      // A mordent is a snap: fast at the start, then the principal holds.
      intensity: lerp(rng, [1.0, 1.6]),
      monophonic: 0.9,
    }),
    alignment: 'at start',
  };
}

/** Turn — four slots around the principal. `inverted` starts from below. */
function turn(rng, cfg, { id }, inverted, delayed) {
  const up = neighbour(rng);
  const down = neighbour(rng);
  const pool =
    `<note xml:id="u" interval.chromatic="${up}.0" />` +
    `<note xml:id="l" interval.chromatic="-${down}.0" />`;
  const order = inverted ? `#l #${id} #u #${id}` : `#u #${id} #l #${id}`;
  // A delayed turn sits in the second half of the note; MPM's way of saying
  // that is to hang the frame off the note's end.
  const alignment = delayed || rng.nextDouble() < 0.2 ? 'at end' : 'at start';
  return {
    pool,
    order,
    spread: spread(rng, cfg, {
      lengthPct: wide(rng, [25, 75], cfg.breadth),
      intensity: lerp(rng, cfg.intensity),
      alignment,
    }),
    alignment,
  };
}

/**
 * A grace note, from its ACTUAL notated pitch — the one figure whose pitch is
 * known rather than sampled, so the interval is exact.
 *
 * Acciaccatura (`slashed`) is short and usually crushed in before the beat;
 * appoggiatura is long and takes its value out of the principal, on the beat.
 */
function grace(rng, cfg, { id, pitch }, gracePitches, slashed) {
  const pool = gracePitches
    .map((p, i) => `<note xml:id="g${i}" interval.chromatic="${f1(p - pitch)}" />`)
    .join('');
  const order = gracePitches.map((_, i) => `#g${i}`).join(' ') + ` #${id}`;
  const share = slashed ? cfg.acciaccaturaShare : cfg.gracePrincipalShare;
  return {
    pool,
    order,
    spread: spread(rng, cfg, {
      lengthPct: lerp(rng, share) * 100,
      intensity: 1,
      // The crushed grace is the one figure that is anticipated by default.
      anticipate: slashed ? 0.7 : 0.2,
      monophonic: 0.95,
    }),
    alignment: 'at start',
  };
}

/**
 * An arpeggiated chord — the commonest notated ornament sign in the repertoire
 * (3739 of 9288 in train-eligible ASAP) and the one the corpus never had.
 *
 * Unlike every other figure here this one generates NO notes: it re-times the
 * chord's own notes, which keep their score ids and stay matches. That is
 * precisely why it matters. A wide roll puts five notes 400 ms apart around one
 * written moment — the exact surface shape of an ornament figure — while every
 * one of them is a written note. Without these the head only ever sees dense
 * clusters that ARE insertions, and has no reason to learn the difference.
 *
 * It has to be spelled in MPM **v2** (bare `frame.start`/`frameLength` plus
 * `time.unit`), not v3: espressivo warns outright that a v3 temporalSpread
 * "carries no v2 frame, so it will spread nothing", and a v3 ornament naming
 * the chord's ids in `note.order` does not re-time them at all — it generates
 * pitch-copies beside the originals and doubles the chord.
 *
 * The frame is in milliseconds because a roll takes about the same wall-clock
 * time whatever the tempo.
 */
function arpeggio(rng, cfg, { chordIds }) {
  const ids = rng.nextDouble() < 0.12 ? [...chordIds].reverse() : chordIds; // occasionally top-down
  const ms = wide(rng, [70, 340], cfg.breadth) * cfg.breadth;
  // Where the roll sits against the beat. Pianists commonly place the top note
  // on the beat and start the roll before it, so most frames are anticipated.
  const start = -ms * (rng.nextDouble() < 0.65 ? rng.nextDouble() : 0);
  const shift = rng.nextDouble() < 0.5 ? ' noteoff.shift="true"' : '';
  const int = lerp(rng, [0.75, 1.35]);
  return {
    def:
      `<ornamentDef name="%NAME%">` +
      `<temporalSpread frame.start="${start.toFixed(1)}" frameLength="${ms.toFixed(1)}"` +
      ` time.unit="milliseconds" intensity="${int.toFixed(3)}"${shift} /></ornamentDef>`,
    order: ids.map((i) => `#${i}`).join(' '),
  };
}

/**
 * Realize one request into `{ def, entry }` XML, or null when the kind is not
 * modelled.
 *
 * A request names the principal by its MSM `xml:id` and carries the notated
 * facts the figure's shape depends on:
 *   { msmId, date, durQuarters, pitch, kind, gracePitches?, index }
 */
export function realizeSign(req, rng, cfg = DEFAULTS) {
  const id = req.msmId;
  const defName = `orn${req.index}`;
  if (req.kind === 'arpeggio') {
    if (!req.chordIds || req.chordIds.length < 2) return null;
    const a = arpeggio(rng, cfg, req);
    return {
      def: a.def.replace('%NAME%', defName),
      // No `noteid` and no note pool: those are the v3 markers, and one of them
      // would turn this back into a note-generating ornament.
      entry:
        `<ornament date="${f1(req.date)}" name.ref="${defName}"` +
        ` note.order="${a.order}" xml:id="mlorn${req.index}" />`,
    };
  }
  let fig;
  switch (req.kind) {
    case 'trill':
      fig = trill(rng, cfg, { id, durQuarters: req.durQuarters });
      break;
    case 'inverted-mordent': // MusicXML's upper mordent = Pralltriller
      fig = mordent(rng, cfg, { id }, true);
      break;
    case 'mordent':
      fig = mordent(rng, cfg, { id }, false);
      break;
    case 'turn':
      fig = turn(rng, cfg, { id }, false, false);
      break;
    case 'inverted-turn':
      fig = turn(rng, cfg, { id }, true, false);
      break;
    case 'delayed-turn':
      fig = turn(rng, cfg, { id }, false, true);
      break;
    case 'grace':
      fig = grace(rng, cfg, { id, pitch: req.pitch }, req.gracePitches, req.slashed);
      break;
    default:
      return null;
  }
  return {
    def: `<ornamentDef name="${defName}" alignment="${fig.alignment}">${fig.spread}</ornamentDef>`,
    entry:
      `<ornament date="${f1(req.date)}" name.ref="${defName}" noteid="#${id}"` +
      ` note.order="${fig.order}" xml:id="mlorn${req.index}">${fig.pool}</ornament>`,
  };
}

/** `{ header, map }` for a list of requests; both '' when nothing was realized. */
export function buildOrnamentation(requests, rng, cfg = DEFAULTS) {
  const defs = [];
  const entries = [];
  for (const req of requests) {
    const r = realizeSign(req, rng, cfg);
    if (r === null) continue;
    defs.push(r.def);
    entries.push(r.entry);
  }
  if (entries.length === 0) return { header: '', map: '' };
  return {
    header:
      '<ornamentationStyles><styleDef name="mlignOrns">' + defs.join('') + '</styleDef></ornamentationStyles>',
    map: `<ornamentationMap><style date="0.0" name.ref="mlignOrns" />${entries.join('')}</ornamentationMap>`,
  };
}

/**
 * Ornament pre-pass over the facade's PerformanceData.
 *
 * A note is GENERATED iff its id is not a known score id (generated notes get
 * random `meico_<uuid>` ids; slot membership is NOT sufficient — the principal
 * itself appears inside the figure with a slot and keeps its score id, and
 * stays a match per D10). Generated notes get id=null + an ornament origin so
 * the robustness layer and editsToAlignment treat them as provenanced
 * insertions. Carved heads keep score ids (match with altered duration).
 *
 * Generated notes of zero length are dropped rather than emitted. They happen
 * where a frame's last slot lands on the next onset and `noteoff.shift` clips
 * it — the carved-head cliff — and a note nobody could ever hear is not
 * something to ask a model to detect. Returns the count so a caller can watch
 * it rather than discover it in the data later.
 */
export function normalizeOrnaments(data, scoreIdSet) {
  let touched = false;
  let dropped = 0;
  const parts = data.parts.map((part) => ({
    ...part,
    notes: part.notes.flatMap((n) => {
      if (n.id !== null && scoreIdSet.has(n.id)) return [n];
      if (!n.ornamented && n.id !== null) return [n]; // unknown non-ornament id: leave as-is
      touched = true;
      if (Number(n.milliseconds?.end) - Number(n.milliseconds?.date) <= 0) {
        dropped++;
        return [];
      }
      return [{
        ...n,
        id: null,
        origin: {
          type: 'ornament',
          anchor: n.ornamentAnchor ?? null,
          ref: n.ornamentRef ?? null,
          slot: n.ornamentSlot ?? -1,
          pass: n.ornamentPass ?? 0,
        },
      }];
    }),
  }));
  return { data: touched ? { ...data, parts } : data, dropped };
}

/** Splice ornament header + map, and any imprecision maps, into a buildMpm document. */
export function injectOrnaments(mpmXml, orn, imprecisionXml = '') {
  const map = (orn && orn.map) || '';
  if (!map && !imprecisionXml) return mpmXml;
  let out = mpmXml;
  if (map) out = out.replace('<global><header />', `<global><header>${orn.header}</header>`);
  return out.replace('<dated><tempoMap>', `<dated>${map}${imprecisionXml}<tempoMap>`);
}
