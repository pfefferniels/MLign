"""Corpus source D: REAL ornament→principal ground truth, mined from match files.

Until now the attribution head ("which written note does this played note
elaborate") could only be scored on espressivo renders, i.e. against the
generator's own ornament model. Real match files DO carry the ornament sign —
not as `ornament(...)` lines (there are none anywhere in ASAP, Batik or 4x22)
but in the snote's attribute list:

    snote(n140-1,[G,n],5,7:1,1/8,1/8,24.5,25.0,[v1,staff1,trill-mark])-note(n133,81,...).
    insertion-note(n135,79,13068,13111,59,1,0).

The sign says a written note is ornamented; the surrounding insertions are the
notes the performer added to realise it. Joining the two is the whole job here.

Where the signs are, and are not. Only matchfile v1.0.0 carries an attribute
list at all: ASAP's other 682 files are v5.0 and every snote there has `[]`, so
they are silently unusable. That leaves Batik's 36 and 381 of ASAP's 1063
(276 of which also have robust_note_alignment). Vienna 4x22's 88 files are
v1.0.0 but their attribute vocabulary is only staff/voice/accent/staccato/grace
— no ornament sign anywhere — so 4x22 is not a source.

NOT TRAINABLE, BY CONSTRUCTION. `meta.gen` is "realorn-v1", deliberately not
"mlign-*", because src/mlign/dataset.py gates attribution SUPERVISION on that
prefix. These labels are incomplete: an empty `orn` means "no ornament sign we
could resolve", never "not an ornament". A performer's unmarked arpeggiation,
a trill whose principal was itself deleted, and a plain wrong-note slip all
look identical here. Supervising on that would teach the head that real
ornaments are not ornaments — the one label error that ruins it on exactly the
material we care about. Rows are evaluable and never trainable.

For the same reason, an evaluator must read the ins kinds as:
  kind 2 + an `orn` entry → known ornament note, known anchor: score it;
  kind 4                  → unknown, IGNORE it (not "none").
Scoring kind-4 insertions as negatives would make detection precision
meaningless. eval/run_attribution.py does not do this today — it requires
meta.gen == "mlign-*" and would refuse this file outright.

Attribution rule (see ANCHOR_* constants): pitch-and-time, never "most recent
matched note". The recency rule is wrong and the counterexample is in the
corpus — kv279_1 bar 7, where the trill's added notes are interleaved with
left-hand matched notes and recency anchors them to a bass note in the other
hand. Ambiguous groups are dropped rather than guessed.

Usage:
  .venv/bin/python scripts/corpus/real_orn_gt.py out.jsonl --corpus batik
  .venv/bin/python scripts/corpus/real_orn_gt.py out.jsonl --corpus asap [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

BATIK = ROOT / "data/benchmarks/batik_plays_mozart/match"
ASAP = ROOT / "data/benchmarks/asap-dataset"

# Windowing: identical to scripts/corpus/real_gt.py, the existing real-performance
# windowing, so the two real corpora are the same shape of sample.
WIN = 384
STRIDE = 256

# --- attribution rule parameters -------------------------------------------
# How far before the principal's own played onset an added note may still
# belong to it. Non-zero because the matcher often pairs the principal snote
# with a LATER performed note (a trill starting on the upper auxiliary gets its
# second note matched), and because grace-note figures are played ahead of the
# beat.
ANCHOR_PRE_MS = 250.0
# How far after: the principal's own NOTATED length, mapped into performed time
# through the file's own alignment (a whole-note trill at 40bpm runs 6 s, and any
# fixed ms bound would truncate it), plus this much slack for a figure that
# spills past the written value. 400 ms and not the 1500 ms flat floor we
# started from: 98.7% of Batik's and 96.2% of ASAP's attributed notes fall
# INSIDE the notated span, and hand-checking what a flat floor buys past it
# found spurious notes every time — a `mordent` collecting a note 1.3 s later,
# a finished trill collecting the next bar's melody note.
ANCHOR_POST_SLACK_MS = 400.0
# Trills, mordents and turns are close-neighbour figures: the added notes sit a
# step or two from the principal. Wide enough for a turn's lower neighbour on a
# whole-tone side (±2) plus a Nachschlag leading tone, tight enough to exclude
# the other hand.
PITCH_TOL = 4
# A realised trill, mordent or turn re-strikes its principal or brushes a
# neighbour, so at least one note of the group must land within a step of it.
# A group that never does is the known failure mode of the SIGN itself: a
# MusicXML exporter attaches a trill over a chord to one chord tone, not
# necessarily the trilled one (Batik kv283_3 has the sign on A4 of an A4+C5
# chord while the performer trills C5-D5). We cannot tell which tone is meant,
# so we drop the group rather than name the wrong principal — 2.3% of Batik
# groups and 0.8% of ASAP's, and in 17 of those 19 a simultaneous chord partner
# is visibly the better fit.
GROUP_MIN_NEIGHBOUR = 2

# Signs that mean "the performer adds notes here". `grace` is excluded on
# purpose: a grace note is itself a written snote, so it gets matched, not
# inserted — treating it as an anchor only adds spurious candidates next to
# real trills. `tremolo` is excluded because a two-note tremolo spans intervals
# the pitch rule cannot bound. `scoop` (2 in the whole corpus) is a slide, not
# a note-generating ornament.
ORNAMENT_SIGNS = frozenset(
    {"trill-mark", "wavy-line", "mordent", "inverted-mordent", "turn", "inverted-turn"}
)

_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ALTER = {"n": 0, "": 0, "#": 1, "s": 1, "x": 2, "ss": 2, "b": -1, "f": -1, "bb": -2, "ff": -2}

# snote(Id,[Step,Alter],Octave,Bar:Beat,Offset,Duration,OnsetBeat,OffsetBeat,[attrs])
# then either -note(...) or -deletion.
_SNOTE = re.compile(
    r"^snote\((?P<id>[^,]+),\[(?P<step>[A-Ga-g]),(?P<alter>[^\]]*)\],(?P<oct>-?\d+),"
    r"[^,]*,[^,]*,(?P<dur>[^,]*),(?P<on>-?[\d.]+),(?P<off>-?[\d.]+),"
    r"\[(?P<attrs>[^\]]*)\]\)(?P<tail>.*)$"
)
# matchfile v1.0.0: note(Id,MidiPitch,Onset,Offset,Velocity,Channel,Track).
# Verified against partitura 1.6.0 io/matchlines_v1.py MatchNote.field_names.
# v5.0 spells the pitch instead — those files also carry EMPTY attribute lists,
# so they hold no ornament sign and are skipped upstream.
_NOTE = re.compile(
    r"^note\((?P<id>[^,]+),(?P<pitch>-?\d+),(?P<on>-?\d+),(?P<off>-?\d+),(?P<vel>-?\d+),"
)
_TS = re.compile(r"^scoreprop\(timeSignature,(\d+)/(\d+),[^,]*,[^,]*,(-?[\d.]+)\)")
_VOICE = re.compile(r"\bv(\d+)\b")


@dataclass
class SNote:
    id: str
    pitch: int
    onset_q: float
    dur_q: float
    voice: int
    signs: frozenset
    perf_id: str | None  # None = deletion


@dataclass
class PNote:
    id: str
    pitch: int
    onset: float  # seconds
    duration: float  # seconds
    velocity: int


class BeatToQuarter:
    """Beat→quarter map from the file's own timeSignature changes.

    The match file's OnsetBeat counts the time signature's DENOMINATOR unit, not
    quarters: in 12/16 a sixteenth spans 1.0 beat. Emitting those numbers as
    ppq-720 ticks would tell the model a sixteenth is a quarter. 51 of the 417
    usable files change denominator mid-piece, so the map has to be piecewise.
    """

    def __init__(self, changes: list[tuple[float, float]]):
        changes = sorted(changes) or [(0.0, 1.0)]
        self.b = np.array([c[0] for c in changes], dtype=float)
        self.qpb = np.array([c[1] for c in changes], dtype=float)
        cum = np.zeros(len(changes))
        for i in range(1, len(changes)):
            cum[i] = cum[i - 1] + (self.b[i] - self.b[i - 1]) * self.qpb[i - 1]
        self.cum = cum

    def __call__(self, beat: float) -> float:
        i = max(0, int(np.searchsorted(self.b, beat, side="right")) - 1)
        return float(self.cum[i] + (beat - self.b[i]) * self.qpb[i])


def parse_match(path: Path) -> tuple[list[SNote], list[PNote], dict]:
    """Score and performance notes in the GT's own id space, from the match file.

    Both sides come from the match file rather than the MusicXML/MIDI, because
    only the match file carries the ornament attributes, Batik numbers its MIDI
    notes non-sequentially (a MIDI parse will not reproduce the note ids), and
    partitura's create_score rejects some Batik scores outright.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    units = float(re.search(r"info\(midiClockUnits,(\d+)\)", text).group(1))
    rate = float(re.search(r"info\(midiClockRate,(\d+)\)", text).group(1))
    sec_per_tick = rate / 1e6 / units  # partitura utils.music.midi_ticks_to_seconds

    changes = []
    for raw in text.splitlines():
        m = _TS.match(raw.strip())
        if m:
            changes.append((float(m.group(3)), 4.0 / int(m.group(2))))
    b2q = BeatToQuarter(changes)

    snotes: list[SNote] = []
    pnotes: dict[str, PNote] = {}
    stats = {"snote_unparsed": 0, "dur_mismatch": 0, "dur_checked": 0, "perf_reused": 0}

    def take_note(body: str) -> str | None:
        m = _NOTE.match(body)
        if not m:
            return None
        nid = m.group("id")
        if nid in pnotes:
            stats["perf_reused"] += 1
            return nid
        on, off = int(m.group("on")), int(m.group("off"))
        pnotes[nid] = PNote(
            id=nid,
            pitch=int(m.group("pitch")),
            onset=on * sec_per_tick,
            duration=max(0.001, (off - on) * sec_per_tick),
            velocity=max(1, min(127, int(m.group("vel")))),
        )
        return nid

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("insertion-note("):
            take_note(line[len("insertion-") :])
            continue
        if not line.startswith("snote("):
            continue
        m = _SNOTE.match(line)
        if not m:
            stats["snote_unparsed"] += 1
            continue
        on_q, off_q = b2q(float(m.group("on"))), b2q(float(m.group("off")))
        # The written Duration is an independent statement of the same span, so
        # it checks the beat→quarter map on every single note rather than on our
        # reading of the header. Counted, never rejected: inside a tuplet the two
        # legitimately differ (Duration is the notation symbol, the beat span is
        # the sounding value), which is all 86 of ASAP's 574911 disagreements.
        try:
            written_q = float(Fraction(m.group("dur"))) * 4.0
        except (ValueError, ZeroDivisionError):
            written_q = None
        if written_q is not None and off_q > on_q:
            stats["dur_checked"] += 1
            if abs((off_q - on_q) - written_q) > 1e-2:
                stats["dur_mismatch"] += 1
        tail = m.group("tail")
        perf_id = take_note(tail[1:]) if tail.startswith("-note(") else None
        vm = _VOICE.search(m.group("attrs"))
        snotes.append(
            SNote(
                id=m.group("id"),
                pitch=12 * (int(m.group("oct")) + 1)
                + _STEP[m.group("step").upper()]
                + _ALTER.get(m.group("alter").strip(), 0),
                onset_q=on_q,
                dur_q=max(0.0, off_q - on_q),
                voice=int(vm.group(1)) if vm else 0,
                signs=frozenset(m.group("attrs").split(",")) & ORNAMENT_SIGNS,
                perf_id=perf_id,
            )
        )
    return snotes, sorted(pnotes.values(), key=lambda p: (p.onset, p.pitch)), stats


def score_time_map(snotes: list[SNote], perf: dict[str, PNote]):
    """score quarters → performed seconds, from the file's own matches.

    Only used to turn a principal's notated length into a performed window
    length. Chord onsets collapse to their median played onset and the curve is
    forced monotone: expressive asynchrony inverts local order often enough that
    a raw interpolation would run backwards.
    """
    xs, ys = [], []
    for sn in snotes:
        if sn.perf_id is not None and sn.perf_id in perf:
            xs.append(sn.onset_q)
            ys.append(perf[sn.perf_id].onset)
    if len(xs) < 2:
        return None
    order = np.argsort(np.asarray(xs), kind="stable")
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    ux, first = np.unique(xs, return_index=True)
    bounds = list(first[1:]) + [len(ys)]
    uy = np.maximum.accumulate([float(np.median(ys[a:b])) for a, b in zip(first, bounds)])
    return lambda q: float(np.interp(q, ux, uy))


def attribute(
    snotes: list[SNote],
    perf_list: list[PNote],
    pitch_tol: int = PITCH_TOL,
    pre_ms: float = ANCHOR_PRE_MS,
    post_slack_ms: float = ANCHOR_POST_SLACK_MS,
) -> tuple[dict[str, str], dict]:
    """Insertion perf-id → anchor snote id, by pitch and time.

    Deliberately NOT "the most recent matched note": in kv279_1 bar 7 the
    trill's added notes are interleaved with left-hand matched notes, so recency
    anchors them to a bass note in the other hand.
    """
    perf = {p.id: p for p in perf_list}
    matched = {sn.perf_id for sn in snotes if sn.perf_id is not None}
    tmap = score_time_map(snotes, perf)
    stats = {"anchors": 0, "anchors_deleted": 0, "multi_cand": 0, "dropped_tie": 0,
             "dropped_not_neighbour": 0}
    if tmap is None:
        return {}, stats

    windows = []  # (lo_ms, hi_ms, pitch, anchor_id, onset_ms, end_ms)
    for sn in snotes:
        if not sn.signs:
            continue
        if sn.perf_id is None or sn.perf_id not in perf:
            # A principal that never sounded cannot be time-anchored, and
            # guessing its window from score time would be exactly the kind of
            # invented label this file exists to avoid.
            stats["anchors_deleted"] += 1
            continue
        stats["anchors"] += 1
        t = perf[sn.perf_id].onset * 1000.0
        span = max(0.0, tmap(sn.onset_q + sn.dur_q) - tmap(sn.onset_q)) * 1000.0
        windows.append((t - pre_ms, t + span + post_slack_ms, sn.pitch, sn.id, t, t + span))

    out: dict[str, str] = {}
    for p in perf_list:
        if p.id in matched:
            continue
        t = p.onset * 1000.0
        # Rank by TIME first, and by distance to the principal's whole notated
        # span rather than to its onset. Pitch-first is wrong and the corpus is
        # full of the counterexample: Mozart's stepwise-descending trill chains
        # (kv279_1 bars 22-23, C-B-A a beat apart) realise each trill as
        # principal + upper neighbour, so each trill's SECOND note has the exact
        # pitch of the PREVIOUS trill's principal and pitch-first hands it back
        # a beat too late. Using the span, not the onset, keeps a long trill's
        # late notes from defecting to whatever ornament starts inside it.
        cand = []
        for lo, hi, pitch, aid, t_on, t_end in windows:
            if not (lo <= t <= hi) or abs(p.pitch - pitch) > pitch_tol:
                continue
            dt = 0.0 if t_on <= t <= t_end else min(abs(t - t_on), abs(t - t_end))
            cand.append((dt, abs(p.pitch - pitch), aid))
        if not cand:
            continue  # not an ornament note as far as this file can tell
        cand.sort()
        if len(cand) > 1:
            stats["multi_cand"] += 1
            if cand[0][:2] == cand[1][:2]:
                stats["dropped_tie"] += 1  # genuinely ambiguous: never guess
                continue
        out[p.id] = cand[0][2]

    by_anchor: dict[str, list[str]] = {}
    for pid, aid in out.items():
        by_anchor.setdefault(aid, []).append(pid)
    spitch = {sn.id: sn.pitch for sn in snotes}
    for aid, pids in by_anchor.items():
        if min(abs(perf[p].pitch - spitch[aid]) for p in pids) > GROUP_MIN_NEIGHBOUR:
            stats["dropped_not_neighbour"] += len(pids)
            for p in pids:
                del out[p]
    return out, stats


def rows_for(path: Path, meta_extra: dict) -> tuple[list[dict], dict]:
    snotes, perf_list, stats = parse_match(path)
    if not snotes or not perf_list:
        return [], stats
    if stats["perf_reused"]:
        # ASAP's Mozart 12-2 MunA04/WuuE03 list 14 played notes each as BOTH a
        # match and an insertion, every one of them at a trill-mark. A file that
        # contradicts itself exactly where we are reading is not a file to mine
        # ornament truth from.
        stats["skipped_inconsistent"] = 1
        return [], stats
    ins2anchor, astats = attribute(snotes, perf_list)
    stats |= astats

    snotes = sorted(snotes, key=lambda s: (s.onset_q, s.pitch))
    s_index = {sn.id: i for i, sn in enumerate(snotes)}
    p_index = {p.id: j for j, p in enumerate(perf_list)}
    s2p = {sn.id: sn.perf_id for sn in snotes if sn.perf_id in p_index}
    # Insertion = played note no snote claims ANYWHERE, not merely one this
    # window failed to match; the window edge must not invent insertions.
    claimed = set(s2p.values())
    ins_j = {j for j, p in enumerate(perf_list) if p.id not in claimed}

    out = []
    n = len(snotes)
    for start in range(0, max(1, n - WIN + 1), STRIDE):
        s_slice = list(range(start, min(n, start + WIN)))
        if len(s_slice) < 64:
            continue
        p_matched = {p_index[s2p[snotes[i].id]]: i for i in s_slice if snotes[i].id in s2p}
        if len(p_matched) < 32:
            continue
        p_sorted = sorted(p_matched)
        t_lo, t_hi = perf_list[p_sorted[0]].onset, perf_list[p_sorted[-1]].onset
        p_slice = set(p_matched)
        for j in ins_j:
            if t_lo <= perf_list[j].onset <= t_hi:
                p_slice.add(j)
        p_slice = sorted(p_slice)
        p_local = {j: k for k, j in enumerate(p_slice)}
        s_local = {i: k for k, i in enumerate(s_slice)}

        # An ornament straddling the window edge keeps only the part that is
        # actually in the row: an anchor outside the window has no index to name,
        # so its notes stay kind 4 (unknown) rather than becoming a dangling label.
        orn_by_anchor: dict[int, list[int]] = {}
        for j in p_slice:
            aid = ins2anchor.get(perf_list[j].id)
            if aid is None or j in p_matched:
                continue
            si = s_index.get(aid)
            if si is None or si not in s_local:
                continue
            orn_by_anchor.setdefault(si, []).append(j)
        if not orn_by_anchor:
            continue  # the row would carry no ornament ground truth at all

        attributed = {j for js in orn_by_anchor.values() for j in js}
        t0 = perf_list[p_slice[0]].onset
        score_rows = [
            [round(snotes[i].onset_q * 720.0, 3), round(snotes[i].dur_q * 720.0, 3),
             snotes[i].pitch, snotes[i].voice % 5]
            for i in s_slice
        ]
        perf_rows = [
            [round((perf_list[j].onset - t0) * 1000.0, 3), round(perf_list[j].duration * 1000.0, 3),
             perf_list[j].pitch, perf_list[j].velocity]
            for j in p_slice
        ]
        align = [[s_local[i], p_local[j]] for j, i in p_matched.items()]
        matched_s = set(p_matched.values())
        dele = [s_local[i] for i in s_slice if i not in matched_s]
        # kind 4 "other", not 0 "slip": nothing here distinguishes a slip from an
        # unmarked addition from an ornament we failed to resolve.
        ins = [[p_local[j], 2 if j in attributed else 4] for j in p_slice if j not in p_matched]
        orn = []
        for si, js in orn_by_anchor.items():
            for slot, j in enumerate(sorted(js, key=lambda j: perf_list[j].onset)):
                orn.append([p_local[j], s_local[si], slot, 0])
        out.append({
            "meta": {"gen": "realorn-v1", "source": str(path.relative_to(ROOT)), "w": start,
                     **meta_extra},
            "score": score_rows,
            "scoreIds": [snotes[i].id for i in s_slice],
            "perf": perf_rows,
            "align": align, "subs": [], "ins": ins, "orn": orn, "del": dele,
        })
    return out, stats


def is_v1(path: Path) -> bool:
    """v5.0 match files spell the pitch differently AND carry empty attribute
    lists, so they hold no ornament sign at all — 682 of ASAP's 1063."""
    return "matchFileVersion,1.0.0" in path.read_text(encoding="utf-8", errors="replace")[:80]


def sources(corpus: str) -> list[tuple[Path, dict]]:
    if corpus == "batik":
        return [(p, {"piece": p.stem}) for p in sorted(BATIK.glob("*.match")) if is_v1(p)]

    from split import test_split  # noqa: E402  (eval/ is on sys.path)

    robust = set()
    with open(ASAP / "metadata.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            flag = row.get("robust_note_alignment", "").strip().lower()
            if flag in ("true", "yes") or (flag and flag not in ("false", "no") and float(flag) > 0):
                robust.add(row["midi_performance"])
    folders, _ = test_split()
    out = []
    for p in sorted(ASAP.rglob("*.match")):
        piece = str(p.parent.relative_to(ASAP))
        if str(p.with_suffix(".mid").relative_to(ASAP)) not in robust or not is_v1(p):
            continue
        # Split is recorded, not filtered: the head was never supervised on
        # ornaments anywhere, but the aligner did train on realgt rows from
        # train-split pieces, so an evaluator may want test-only.
        out.append((p, {"piece": f"{piece}/{p.stem}", "split": "test" if piece in folders else "train"}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--corpus", choices=["batik", "asap"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sources(args.corpus)
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} match files for corpus={args.corpus}", file=sys.stderr)

    written = 0
    total = {}
    with open(args.out, "w") as fh:
        for k, (path, meta) in enumerate(files):
            try:
                rows, stats = rows_for(path, meta)
            except Exception as err:  # a malformed header, an unreadable file
                print(f"skip {path}: {err!r}", file=sys.stderr)
                continue
            for key, v in stats.items():
                total[key] = total.get(key, 0) + v
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                written += 1
            if (k + 1) % 50 == 0:
                print(f"...{k + 1}/{len(files)} ({written} rows)", file=sys.stderr)
    print(f"wrote {written} rows | {total}", file=sys.stderr)


if __name__ == "__main__":
    main()
