"""ASAP score specs: one JSON "score spec" per MusicXML score, for the Node
generator that renders real ASAP scores through espressivo with their notated
ornaments realized.

Two passes over each score, because neither alone is sufficient:

  partitura  gives the note table (onsets/durations in quarters, voice, staff)
             and is the ONLY source for grace-note principals (GraceNote.main_note).
             It models no ornament signs at all -- no Trill/Mordent/Turn/Arpeggio
             objects exist -- so signs cannot come from here.
  raw XML    gives the ornament signs, the chord structure, the key signatures
             and the tie structure. It cannot give onsets without re-implementing
             the divisions/backup/forward cursor -- which we do anyway, and then
             CHECK against partitura WITHIN each measure: the two must differ by
             one constant per measure, or the walk is wrong and the score is
             flagged. (Across measures they may legitimately differ: a few
             MuseScore exports have bars whose voices do not add up.)

The join key is the MusicXML `@id` on `<note>`, which partitura preserves
verbatim in note_array()["id"]. 241 of 242 ASAP scores carry one on every note;
the one that does not is skipped (see NO_ID_SCORES).

One line of JSON per score:

  id            piece folder relative to the asap root
  path          the MusicXML, relative to the repo root
  ppq 720, totalTicks, anacrusisTicks (see below)
  key           {fifths, mode} in force at the start; mode is null when the
                source omits <mode>, which it does for 402 of 531 <key>s
  keyChanges    [{date, fifths, mode}], deduplicated, sorted
  parts         [{number: staff, notes: [{id, date, dur, pitch, voice}]}],
                notes sorted by (date, pitch); voice is partitura's mod 5
  signs         [{noteId, kind, placement, accidental, value, number, date}],
                plus `chord` (ids low->high) on kind "arpeggio". `value` holds
                the tremolo's beam count, the wavy-line's start|stop and the
                arpeggio's direction; `number` pairs the two ends of a line.
  graces        [{id, principal, principalDate, tiedToPrincipal, pitch,
                slashed, staff, seq}]; `seq` orders a run of graces on one
                principal, `principalDate` is the onset the grace leans on
  chords        [[id, ...]] -- ids struck together, size >= 2
  warnings      what was odd about this score; empty for 190 of 201

Two id-space facts that bite if ignored:
  - partitura FOLDS tied notes into one entry keyed by the tie's HEAD id, so the
    ids of tie continuations vanish from the note table. Signs landing on them
    (a `<wavy-line type="stop">` typically does) are remapped to the head;
    chord members that are continuations are dropped, since they are held, not
    struck, at that onset.
  - grace notes are not score notes. They are excluded from parts[].notes and
    listed in `graces` against the id of the note they lean on -- an id which is
    itself sometimes folded away, so `principalDate` carries the onset.

Dates are ticks at ppq 720 (quarters * 720, rounded to 3 dp). partitura puts an
anacrusis at NEGATIVE quarters (origin = the first full measure's downbeat); the
whole score is shifted so nothing is < 0 and the shift is recorded as
`anacrusisTicks`, because the consumer renders this spec as a standalone score.

Parts are split by STAFF (the renderer wants a treble part and a bass part). The
6 scores with >2 staves get >2 parts; the 2 scores with two `<part>` elements get
part 2's staff numbers offset so they do not collide with part 1's.

Usage:
  .venv/bin/python scripts/corpus/asap_spec.py out.jsonl [--split train|test|all]
                                               [--limit N] [--asap-root PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from split import test_split  # noqa: E402

PPQ = 720.0

# Beethoven 17-2 has 2 of its 2030 `<note>` elements without an @id, which breaks
# the raw-XML <-> partitura join for the whole score. Skipped rather than
# half-joined. (The check in spec_for finds any others, should this go stale.)
NO_ID_SCORES = {"Beethoven/Piano_Sonatas/17-2"}

# `<ornaments>` children we normalize; anything else in there is counted as
# unknown rather than silently dropped.
ORNAMENT_KINDS = {
    "trill-mark": "trill",
    "mordent": "mordent",
    "inverted-mordent": "inverted-mordent",
    "turn": "turn",
    "inverted-turn": "inverted-turn",
    "delayed-turn": "delayed-turn",
    "delayed-inverted-turn": "delayed-inverted-turn",
    "vertical-turn": "vertical-turn",
    "schleifer": "schleifer",
    "shake": "shake",
    "haydn": "haydn",
    "tremolo": "tremolo",
    "wavy-line": "wavy-line",
}
# `<notations>` children that are signs in their own right.
NOTATION_KINDS = {"arpeggiate": "arpeggio", "non-arpeggiate": "non-arpeggio"}

STEP_SEMIS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# The Node consumer asserts these on every numeric field.
TICK_MAX = 1e7
TICK_EPS = 1e-3


def bad_tick(v: float) -> bool:
    return not (abs(v) < TICK_MAX) or (0 < abs(v) < TICK_EPS)


def tie_key(note: ET.Element) -> tuple:
    """Sounding pitch + voice + staff: what a tie start/stop pair shares."""
    p = note.find("pitch")
    sounding = None if p is None else (
        p.findtext("step"), p.findtext("alter") or "0", p.findtext("octave"))
    return (note.findtext("voice"), note.findtext("staff"), sounding)


class RawScore:
    """One raw-XML pass: cursor walk + signs + chords + ties + key changes.

    Keyed by MusicXML part id ('P1', ...) and note @id. Positions are quarters
    from that part's first measure start -- an origin that differs from
    partitura's by a constant per part, which spec_for measures and checks.
    """

    def __init__(self, path: Path):
        self.root = ET.parse(path).getroot()
        self.pos: dict[str, float] = {}  # note id -> quarters (raw origin)
        self.part_of: dict[str, str] = {}
        self.measure_of: dict[str, tuple[str, int]] = {}
        self.staff: dict[str, int] = {}  # raw <staff>, 1 when absent
        self.pitch_of: dict[str, int] = {}
        self.is_grace: set[str] = set()
        self.slashed: set[str] = set()
        self.principal_of: dict[str, str] = {}  # grace id -> next real note, same voice
        self.chords: list[list[str]] = []  # <chord/> groups, len >= 2
        self.tie_head: dict[str, str] = {}  # continuation id -> head id
        self.keys: list[tuple[str, int, float, int, str | None]] = []
        self.signs: list[dict] = []
        self.unknown: Counter = Counter()
        self.raw_kinds: Counter = Counter()  # pre-collapse element census
        self.missing_id = 0
        self._walk()

    def _walk(self) -> None:
        for part_el in self.root.findall("part"):
            pid = part_el.get("id") or "P1"
            divisions = 1.0
            pos = 0.0  # cursor, quarters from this part's first measure
            chord_pos = 0.0  # onset of the current chord's first note
            group: list[str] = []
            open_ties: dict[tuple, str] = {}
            pending_graces: dict[str | None, list[str]] = defaultdict(list)
            for midx, measure in enumerate(part_el.findall("measure")):
                # A measure ends at the FURTHEST point any of its voices reached,
                # not wherever the last <backup>-ed voice happened to stop: voices
                # need not all fill the bar, and the short one must not shorten it.
                mmax = pos
                for el in measure:
                    if el.tag == "attributes":
                        d = el.findtext("divisions")
                        if d:
                            divisions = float(d)
                        for k in el.findall("key"):
                            f = k.findtext("fifths")
                            if f is not None:
                                self.keys.append((pid, midx, pos, int(f), k.findtext("mode")))
                    elif el.tag == "backup":
                        pos -= float(el.findtext("duration") or 0) / divisions
                    elif el.tag == "forward":
                        pos += float(el.findtext("duration") or 0) / divisions
                    elif el.tag == "note":
                        nid = el.get("id")
                        if nid is None:
                            self.missing_id += 1
                        is_chord = el.find("chord") is not None
                        grace = el.find("grace") is not None
                        dur = el.findtext("duration")
                        # A <chord/> note sounds with the previous note and does
                        # not move the cursor; a grace note carries no <duration>.
                        if is_chord:
                            onset = chord_pos
                        else:
                            onset = chord_pos = pos
                            pos += float(dur) / divisions if dur else 0.0
                        if not is_chord:
                            self._flush(group)
                            group = []
                        if nid is not None:
                            self.measure_of[nid] = (pid, midx)
                            self._note(el, pid, nid, onset, grace)
                            if el.find("rest") is None:
                                group.append(nid)
                                self._tie(el, nid, open_ties)
                                # A grace leans on the next real note of its
                                # voice; this backs up GraceNote.main_note,
                                # which is None when the chain is broken.
                                voice = el.findtext("voice")
                                if grace:
                                    pending_graces[voice].append(nid)
                                else:
                                    for g in pending_graces.pop(voice, ()):
                                        self.principal_of[g] = nid
                    mmax = max(mmax, pos)
                pos = mmax
            self._flush(group)

    def _note(self, el: ET.Element, pid: str, nid: str, onset: float, grace: bool) -> None:
        self.pos[nid] = onset
        self.part_of[nid] = pid
        st = el.findtext("staff")
        self.staff[nid] = int(st) if st else 1
        p = el.find("pitch")
        if p is not None:
            self.pitch_of[nid] = (
                12 * (int(p.findtext("octave") or 4) + 1)
                + STEP_SEMIS.get(p.findtext("step") or "C", 0)
                + int(float(p.findtext("alter") or 0))
            )
        if grace:
            self.is_grace.add(nid)
            if (el.find("grace").get("slash") or "").lower() == "yes":
                self.slashed.add(nid)
        self._signs(el, nid)

    def _flush(self, group: list[str]) -> None:
        # Grace chords are not score chords; they leave via `graces`.
        if len(group) >= 2 and not any(g in self.is_grace for g in group):
            self.chords.append(group)

    def _tie(self, el: ET.Element, nid: str, open_ties: dict) -> None:
        key = tie_key(el)
        head = nid
        types = {t.get("type") for t in el.findall("tie")}
        if "stop" in types:
            prev = open_ties.pop(key, None)
            if prev is None:
                # MuseScore lets a tie change voice mid-span (Rachmaninoff
                # op32/5 m24-25). partitura pairs those anyway, so relax the
                # key instead of losing the tie and orphaning its signs.
                alt = (next((k for k in open_ties if k[1:] == key[1:]), None)
                       or next((k for k in open_ties if k[2] == key[2]), None))
                prev = open_ties.pop(alt) if alt else None
            if prev is not None:
                head = prev
                self.tie_head[nid] = prev
        if "start" in types:
            open_ties[key] = head

    def _signs(self, el: ET.Element, nid: str) -> None:
        nots = el.find("notations")
        if nots is None:
            return
        for child in nots:
            if child.tag in NOTATION_KINDS:
                self.raw_kinds[child.tag] += 1
                num = child.get("number")
                self.signs.append({
                    "noteId": nid, "kind": NOTATION_KINDS[child.tag],
                    "placement": child.get("placement"), "accidental": None,
                    "value": child.get("direction"),
                    "number": int(num) if num else None,
                })
        orn = nots.find("ornaments")
        if orn is None:
            return
        # An <accidental-mark> sitting loose in <ornaments> qualifies that note's
        # ornament(s) -- which accidental the auxiliary note takes.
        loose_acc = orn.findtext("accidental-mark")
        for child in orn:
            if child.tag == "accidental-mark":
                continue
            self.raw_kinds[child.tag] += 1
            if child.tag not in ORNAMENT_KINDS:
                self.unknown[child.tag] += 1
                continue
            value = None
            if child.tag == "tremolo":
                value = (child.text or "").strip() or None  # beam count
            elif child.tag == "wavy-line":
                value = child.get("type")  # start|stop: a trill's continuation
            num = child.get("number")
            self.signs.append({
                "noteId": nid, "kind": ORNAMENT_KINDS[child.tag],
                "placement": child.get("placement"),
                "accidental": child.findtext("accidental-mark") or loose_acc,
                "value": value,
                "number": int(num) if num else None,
            })


def collapse_arpeggios(raw: RawScore, stats: Counter) -> list[dict]:
    """`<arpeggiate>` sits on EVERY note of the rolled chord. Collapse each set to
    ONE sign on the lowest-pitched note, carrying the chord's ids low->high.

    Grouped by (part, notated onset, number) using the RAW walk's onsets, so a
    member that is a tie continuation still groups with the chord it is drawn in.
    MuseScore omits @number for a single roll -- which legitimately spans both
    staves of a grand-staff chord -- and uses it only to separate simultaneous
    independent rolls, so onset alone is the right key when it is absent."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    out: list[dict] = []
    for s in raw.signs:
        nid = s["noteId"]
        if s["kind"] != "arpeggio" or nid not in raw.pos:
            out.append(s)
            continue
        groups[(raw.part_of[nid], raw.pos[nid], s["number"])].append(s)
    for members in groups.values():
        ids = sorted({m["noteId"] for m in members},
                     key=lambda i: (raw.pitch_of.get(i, 0), i))
        stats["arpeggio-cross-staff"] += len({raw.staff.get(i, 1) for i in ids}) > 1
        # The source sometimes marks only one note of the rolled simultaneity;
        # reported rather than widened, which would be inventing the chord.
        stats["arpeggio-single-note"] += len(ids) < 2
        head = dict(members[0])
        head["noteId"] = ids[0]
        head["chord"] = ids
        out.append(head)
    return out


def spec_for(path: Path, piece: str, warn: list[str], stats: Counter) -> dict | None:
    import partitura
    from partitura.score import GraceNote

    raw = RawScore(path)
    if raw.missing_id:
        warn.append(f"{raw.missing_id} <note> without @id")
        return None
    for tag, n in raw.unknown.items():
        warn.append(f"{n} unhandled <ornaments> child <{tag}>")
    # Pre-collapse element census, so the arpeggio collapse can be audited
    # against a plain grep of the corpus.
    for tag, n in raw.raw_kinds.items():
        stats["raw:" + tag] += n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = partitura.load_score(str(path))

    # Per-PART note arrays: the Score-level array renames ids to "P01_n540",
    # destroying the join key. Staff numbers restart per part, so they are
    # offset to stay unique across the score.
    rows: list[dict] = []
    graces: list[dict] = []
    offsets: dict[tuple, float] = {}  # (part id, measure index) -> raw->partitura
    staff_base, dropped_graces = 0, 0
    for part in score.parts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            na = part.note_array(include_staff=True, include_grace_notes=True)
        if len(na) == 0:
            continue
        pid = part.id or "P1"

        # Check the raw walk against partitura WITHIN each measure: the two must
        # differ by one constant per measure, which validates the chord /
        # backup / forward / divisions handling. They can differ ACROSS measures
        # -- a handful of MuseScore exports (Islamey) have voices that do not
        # add up to the bar, and partitura takes the bar's notated length where
        # this walk takes the furthest voice -- so the offset is measured per
        # measure and used to date key changes. Tolerance covers partitura's
        # float32 note_array; a real walk bug is off by >= 1/64 quarter.
        per_measure: dict[tuple, list[float]] = defaultdict(list)
        for r in na:
            nid = str(r["id"])
            if nid in raw.pos:
                per_measure[raw.measure_of[nid]].append(
                    (float(r["onset_quarter"]) - raw.pos[nid], float(r["onset_quarter"])))
        off_bad = 0
        for mkey, ds in per_measure.items():
            vals = sorted(d for d, _ in ds)
            offsets[mkey] = vals[len(vals) // 2]
            tol = 1e-3 + 1e-6 * max(abs(o) for _, o in ds)
            off_bad += vals[-1] - vals[0] > tol
        if off_bad:
            warn.append(f"part {pid}: {off_bad}/{len(per_measure)} measures whose "
                        f"voices do not add up to the bar (raw walk and partitura "
                        f"place them differently); dates there are partitura's, "
                        f"key changes in those bars approximate")

        # Graces come from the object model, not from note_array: a grace with a
        # <tie type="stop"> is folded away there like any other tie continuation,
        # which loses 6 of the train split's 3700. `seq` is its index in the run
        # of graces leading to the principal -- the order they must be played in.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for gn in part.iter_all(GraceNote):
                mn = gn.main_note
                principal = str(mn.id) if mn is not None else raw.principal_of.get(str(gn.id))
                if principal is None:
                    dropped_graces += 1
                    continue
                graces.append({
                    "id": str(gn.id), "principal": principal, "pitch": int(gn.midi_pitch),
                    "slashed": str(gn.id) in raw.slashed,
                    "staff": staff_base + int(gn.staff or 1),
                    "seq": sum(1 for _ in gn.iter_grace_seq(backwards=True)) - 1,
                })

        for r in na:
            nid = str(r["id"])
            staff = staff_base + int(r["staff"] or 1)
            if r["is_grace"]:
                continue
            rows.append({
                "id": nid, "q": float(r["onset_quarter"]),
                "dq": float(r["duration_quarter"]), "pitch": int(r["pitch"]),
                "voice": int(r["voice"]) % 5, "staff": staff,
            })
        staff_base += int(na["staff"].max()) or 1

    if not rows:
        warn.append("no notes")
        return None

    # partitura puts an anacrusis at negative quarters; shift so nothing is < 0.
    shift = max(0.0, -min(r["q"] for r in rows))
    by_staff: dict[int, list[dict]] = defaultdict(list)
    bad = 0
    for r in rows:
        date = round((r["q"] + shift) * PPQ, 3)
        dur = round(r["dq"] * PPQ, 3)
        bad += bad_tick(date) or bad_tick(dur)
        by_staff[r["staff"]].append({"id": r["id"], "date": date, "dur": dur,
                                     "pitch": r["pitch"], "voice": r["voice"]})
    if bad:
        # A spec the consumer would assert on is worse than no spec.
        hi = max(abs(n["date"]) for v in by_staff.values() for n in v)
        warn.append(f"{bad}/{len(rows)} notes outside the tick contract "
                    f"(|date| up to {hi:.0f}); score not written")
        return None
    parts = [{"number": n, "notes": sorted(by_staff[n], key=lambda x: (x["date"], x["pitch"]))}
             for n in sorted(by_staff)]

    date_of = {r["id"]: round((r["q"] + shift) * PPQ, 3) for r in rows}
    note_ids = set(date_of)
    grace_ids = {g["id"] for g in graces}

    # Ticks for ANY raw position, via its measure's raw->partitura offset.
    # Measures holding no sounding note (partitura drops rests) borrow the
    # nearest earlier measure's offset.
    known: dict[str, list[int]] = defaultdict(list)
    for p, mi in sorted(offsets):
        known[p].append(mi)

    def raw_date(pid: str, midx: int, q: float) -> float:
        near = bisect_right(known[pid], midx) - 1
        off = offsets[(pid, known[pid][near])] if near >= 0 else 0.0
        return round((q + off + shift) * PPQ, 3)

    def date_of_raw(nid: str) -> float | None:
        if nid in date_of:
            return date_of[nid]
        if nid in raw.pos:
            return raw_date(*raw.measure_of[nid], raw.pos[nid])
        return None

    # Signs land on tie continuations too (a wavy-line stop typically does);
    # partitura folded those ids away, so walk them back to the tie head.
    def head(nid: str) -> str:
        seen = 0
        while nid not in note_ids and nid in raw.tie_head and seen < 64:
            nid = raw.tie_head[nid]
            seen += 1
        return nid

    signs = collapse_arpeggios(raw, stats)
    remapped = unresolved = 0
    for s in signs:
        if "chord" in s:
            s["chord"] = list(dict.fromkeys(head(i) for i in s["chord"]))
        h = head(s["noteId"])
        remapped += h != s["noteId"]
        s["noteId"] = h
        if h not in note_ids and h not in grace_ids:
            unresolved += 1
        s["date"] = date_of_raw(h)
    signs.sort(key=lambda s: (s["date"] if s["date"] is not None else -1.0,
                              s["kind"], s["noteId"]))

    if remapped:
        warn.append(f"{remapped} sign(s) remapped from a tie continuation to its head")
    if unresolved:
        warn.append(f"{unresolved} sign(s) whose noteId resolves to no note and no grace")
    if dropped_graces:
        warn.append(f"{dropped_graces} grace(s) dropped: no principal")
    stats["sign-remapped"] += remapped
    stats["sign-unresolved"] += unresolved
    stats["grace-dropped"] += dropped_graces

    # Chord members that are tie continuations are HELD at this onset, not
    # struck, so they leave the group; a group of one is not a chord.
    chords = []
    for c in raw.chords:
        keep = [i for i in c if i in note_ids]
        if len(keep) >= 2:
            chords.append(keep)
        elif len(c) >= 2:
            stats["chord-shrunk"] += 1

    keys = []
    dated = sorted((raw_date(pid, midx, q), f, m) for pid, midx, q, f, m in raw.keys)
    for date, f, m in dated:
        if not keys or (f, m) != (keys[-1]["fifths"], keys[-1]["mode"]):
            keys.append({"date": date, "fifths": f, "mode": m})
    if not keys:
        warn.append("no <key> in score; assuming C major")

    # Sanity: unique ids, sorted, non-negative, graces pointing at real notes.
    seen: set[str] = set()
    for p in parts:
        prev = None
        for n in p["notes"]:
            if n["id"] in seen:
                warn.append(f"duplicate note id {n['id']}")
            seen.add(n["id"])
            if n["date"] < 0:
                warn.append(f"negative date on {n['id']}")
            if prev is not None and (n["date"], n["pitch"]) < prev:
                warn.append(f"notes out of order at {n['id']}")
            prev = (n["date"], n["pitch"])
    # `principal` can name a note that is NOT in parts[].notes: MuseScore writes
    # a rolled chord as graces tied into the chord tones, and any tied-into
    # principal is folded into its tie head by partitura. The pointer is still
    # the true MusicXML one, so `principalDate` carries the notated onset the
    # grace leans on -- which is what a renderer actually needs -- and
    # `tiedToPrincipal` says the principal is held rather than struck there.
    orphan = tied = 0
    for g in graces:
        g["principalDate"] = date_of_raw(g["principal"])
        g["tiedToPrincipal"] = g["principal"] not in note_ids
        tied += g["tiedToPrincipal"]
        if g["principalDate"] is None:
            orphan += 1
    graces.sort(key=lambda g: (g["principalDate"] if g["principalDate"] is not None
                               else -1.0, g["seq"], g["pitch"]))
    if orphan:
        warn.append(f"{orphan} grace(s) with no datable principal")
    stats["grace-orphan"] += orphan
    stats["grace-principal-not-struck"] += tied

    total = max(n["date"] + n["dur"] for p in parts for n in p["notes"])
    if bad_tick(total):
        warn.append(f"totalTicks out of contract: {total}")
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:
        rel = str(path)
    return {
        "id": piece,
        "path": rel,
        "ppq": int(PPQ),
        "totalTicks": round(total, 3),
        "anacrusisTicks": round(shift * PPQ, 3),
        "key": {"fifths": keys[0]["fifths"], "mode": keys[0]["mode"]} if keys
               else {"fifths": 0, "mode": None},
        "keyChanges": keys,
        "parts": parts,
        "signs": signs,
        "graces": graces,
        "chords": chords,
        "warnings": warn,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--asap-root", default=str(ROOT / "data/benchmarks/asap-dataset"))
    args = ap.parse_args()

    asap = Path(args.asap_root)
    test_folders, _ = test_split()
    todo = []
    for p in sorted(asap.glob("**/xml_score.musicxml")):
        piece = str(p.parent.relative_to(asap))
        if args.split == "train" and piece in test_folders:
            continue
        if args.split == "test" and piece not in test_folders:
            continue
        todo.append((piece, p))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} scores for split={args.split}", file=sys.stderr)

    t0 = time.time()
    written = notes = graces = chords = 0
    kinds: Counter = Counter()
    stats: Counter = Counter()
    skipped: list[str] = []
    flagged: list[str] = []
    with open(args.out, "w") as fh:
        for k, (piece, p) in enumerate(todo):
            if piece in NO_ID_SCORES:
                skipped.append(f"{piece}: <note> elements without @id, "
                               f"raw-XML <-> partitura join impossible")
                continue
            warn: list[str] = []
            try:
                spec = spec_for(p, piece, warn, stats)
            except Exception as err:
                skipped.append(f"{piece}: {type(err).__name__}: {err}")
                continue
            if spec is None:
                skipped.append(f"{piece}: " + "; ".join(warn))
                continue
            fh.write(json.dumps(spec) + "\n")
            written += 1
            notes += sum(len(pt["notes"]) for pt in spec["parts"])
            graces += len(spec["graces"])
            chords += len(spec["chords"])
            for s in spec["signs"]:
                kinds[s["kind"]] += 1
            if warn:
                flagged.append(f"{piece}: " + "; ".join(warn))
            if (k + 1) % 25 == 0:
                print(f"...{k + 1}/{len(todo)} ({written} written, {notes} notes, "
                      f"{time.time() - t0:.0f}s)", file=sys.stderr)

    dt = time.time() - t0
    size = Path(args.out).stat().st_size
    print(f"\nwrote {written} scores in {dt:.1f}s -> {args.out} ({size / 1e6:.1f} MB)",
          file=sys.stderr)
    print(f"skipped {len(skipped)}:", file=sys.stderr)
    for s in skipped:
        print(f"  {s}", file=sys.stderr)
    print(f"notes {notes} | graces {graces} | chords {chords} | "
          f"signs {sum(kinds.values())}", file=sys.stderr)
    for kk, v in kinds.most_common():
        print(f"  {kk:22s} {v}", file=sys.stderr)
    print(f"signs per 1000 notes: {1000.0 * sum(kinds.values()) / max(1, notes):.2f}",
          file=sys.stderr)
    for kk, v in sorted(stats.items()):
        print(f"  {kk:22s} {v}", file=sys.stderr)
    if flagged:
        print(f"\n{len(flagged)} score(s) with warnings:", file=sys.stderr)
        for f in flagged:
            print(f"  {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
