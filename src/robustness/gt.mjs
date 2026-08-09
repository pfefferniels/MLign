/**
 * Flatten an edited PerformanceData + edit log into alignment ground truth.
 *
 * Output record kinds (parangonar-compatible triple semantics):
 *   { label: 'match',     scoreId, perfId, sub? }  — perf note realizes score note
 *   { label: 'insertion', perfId, provenance }     — spurious / first-pass note
 *   { label: 'deletion',  scoreId }                — score note never sounded
 *
 * perfId: performed notes get stable sequential ids p0..pN in global onset
 * order at emission time (the facade's own ids are score identities and vanish
 * on inserted notes, so performance identity must be minted here — same policy
 * as partitura's performed-note ids that parangonar consumes).
 *
 * Insertion provenance is read from the note's own `origin` field (stamped by
 * applyRobustness), never from positional fingerprints — structural ops may
 * move inserted notes after their log entry was written.
 *
 * A null-id note WITHOUT an origin existed in the input (a score note that
 * never had an xml:id). That breaks match identity, so it is reported as
 * `unattributed` — corpus builders must guarantee id totality upstream and
 * treat unattributed > 0 as a generator bug.
 */

/**
 * @param editedData PerformanceData returned by applyRobustness
 * @param edits      the edit log returned with it
 * @returns {{ alignment: object[], perfNotes: object[], unattributed: number }}
 *   perfNotes: [{ perfId, part, id, pitch, date, duration, velocity, onsetMs, offsetMs }]
 *   in global onset order — the JSONL-ready performance-side note list.
 */
export function editsToAlignment(editedData, edits) {
  const substituted = new Map();
  for (const edit of edits) {
    if (edit.op === 'substitute' && edit.id !== null) substituted.set(edit.id, edit);
  }

  // Global onset order across parts → perf ids.
  const all = [];
  editedData.parts.forEach((part, pi) => {
    for (const n of part.notes) all.push({ part: pi, n });
  });
  all.sort((x, y) =>
    x.n.milliseconds.date - y.n.milliseconds.date || x.n.pitch - y.n.pitch || x.part - y.part,
  );

  const alignment = [];
  const perfNotes = [];
  let unattributed = 0;
  all.forEach(({ part, n }, i) => {
    const perfId = `p${i}`;
    perfNotes.push({
      perfId, part, id: n.id, pitch: n.pitch, date: n.date, duration: n.duration,
      velocity: n.velocity, onsetMs: n.milliseconds.date, offsetMs: n.milliseconds.end,
    });
    if (n.id !== null) {
      const rec = { label: 'match', scoreId: n.id, perfId };
      const sub = substituted.get(n.id);
      if (sub) rec.sub = { from: sub.from, to: sub.to, kind: sub.kind };
      alignment.push(rec);
    } else if (n.origin) {
      alignment.push({ label: 'insertion', perfId, provenance: n.origin });
    } else {
      unattributed++;
      alignment.push({ label: 'insertion', perfId, provenance: { type: 'unattributed' } });
    }
  });

  for (const edit of edits) {
    if (edit.op === 'delete' && edit.note.id !== null) {
      alignment.push({ label: 'deletion', scoreId: edit.note.id });
    } else if (edit.op === 'skip') {
      for (const { note } of edit.removed) {
        if (note.id !== null) alignment.push({ label: 'deletion', scoreId: note.id });
      }
    }
  }

  return { alignment, perfNotes, unattributed };
}
