"""Loading nASAP note-alignment ground truth.

The (n)ASAP dataset (data/benchmarks/asap-dataset) provides per performance:
  <perf>_note_alignments/note_alignment.tsv  — columns xml_id, midi_id, track,
      channel, pitch, onset. One row per PERFORMANCE note in the aligned pass:
      xml_id is the unfolded-score note id ("n2-1", repeat-pass suffix) for a
      match, or the literal string "insertion". Score notes that never sound
      have no row — they are the deletions, recoverable only against the score.
  <perf>.match — the same alignment in matchfile v1.0.0 syntax, which also
      lists deletions explicitly (snote lines without a paired note).

We read the .match file for the complete triple set (matches, insertions,
deletions) and the tsv only as a cross-check, because deletions matter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Alignment:
    """Ground-truth or predicted alignment as id-pair sets."""

    matches: frozenset[tuple[str, str]]  # (score_id, perf_id)
    insertions: frozenset[str]  # perf_id
    deletions: frozenset[str]  # score_id


_SNOTE_NOTE = re.compile(r"^snote\(([^,]+),.*?\)-note\(([^,]+),")
_SNOTE_ONLY = re.compile(r"^snote\(([^,]+),")
_NOTE_ONLY = re.compile(r"^(?:insertion-)?note\(([^,]+),")


def load_match(path: str | Path) -> Alignment:
    """Parse a matchfile into id-pair sets.

    Handles the three line kinds that carry alignment information:
      snote(...)-note(...)      match
      snote(...)-deletion.      deletion (also: bare snote without -note)
      insertion-note(...)       insertion
    Ornament/trill lines (`trill(...)-note(...)` etc.) count their perf note as
    an insertion for triple purposes, mirroring parangonar's evaluation.
    """
    matches: set[tuple[str, str]] = set()
    insertions: set[str] = set()
    deletions: set[str] = set()

    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("info(", "scoreprop(", "%")):
            continue
        if line.startswith("snote("):
            pair = _SNOTE_NOTE.match(line)
            if pair:
                matches.add((pair.group(1), pair.group(2)))
                continue
            solo = _SNOTE_ONLY.match(line)
            if solo:  # snote(...)-deletion. or unpaired snote
                deletions.add(solo.group(1))
                continue
        elif line.startswith("insertion-note("):
            note = _NOTE_ONLY.match(line)
            if note:
                insertions.add(note.group(1))
                continue
        elif line.startswith(("trill(", "ornament(", "trailing_played_note-", "hammer_bounce-")):
            note = re.search(r"-note\(([^,]+),", line)
            if note:
                insertions.add(note.group(1))
                continue
        # remaining kinds (sustain(), pedal lines, ...) carry no note identity

    return Alignment(frozenset(matches), frozenset(insertions), frozenset(deletions))


def load_tsv(path: str | Path) -> Alignment:
    """Read note_alignment.tsv into the same triple sets as `load_match`.

    Row semantics: xml_id == "insertion" → spurious performance note;
    midi_id == "deletion" → score note that never sounded; otherwise a match.
    """
    matches: set[tuple[str, str]] = set()
    insertions: set[str] = set()
    deletions: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        assert header.startswith("xml_id"), f"unexpected tsv header in {path}"
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            xml_id, midi_id = cols[0], cols[1]
            if xml_id == "insertion":
                insertions.add(midi_id)
            elif midi_id == "deletion":
                deletions.add(xml_id)
            else:
                matches.add((xml_id, midi_id))
    return Alignment(frozenset(matches), frozenset(insertions), frozenset(deletions))


@dataclass
class NasapIndex:
    """All performances of the dataset with their score and GT paths."""

    root: Path
    entries: list[dict] = field(default_factory=list)

    @classmethod
    def build(cls, root: str | Path) -> "NasapIndex":
        root = Path(root)
        entries = []
        for match in sorted(root.rglob("*.match")):
            perf = match.with_suffix("")  # …/Shi05M
            midi = perf.with_suffix(".mid")
            tsv = perf.parent / f"{perf.name}_note_alignments" / "note_alignment.tsv"
            score = perf.parent / "xml_score.musicxml"
            if midi.exists() and score.exists():
                entries.append(
                    {
                        "piece": str(perf.parent.relative_to(root)),
                        "performer": perf.name,
                        "match": match,
                        "midi": midi,
                        "tsv": tsv if tsv.exists() else None,
                        "score": score,
                    }
                )
        return cls(root=root, entries=entries)
