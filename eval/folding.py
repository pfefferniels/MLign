"""Constructing the folded-score condition from a played-unfolding score.

The 4x22 GT score (match-file id space) is the played unfolding: pass-2 notes
carry ids `<base>-2`. That labelling IS the repeat structure, so we can:

  fold(score)        → pass-1 notes only, later onsets shifted left over the
                       removed pass-2 spans (the score as printed, one pass);
  candidates(score)  → every combination of "section played once/twice",
                       reconstructed in the GT id space (clones get -2 ids and
                       the span's onset shift), so a correctly-chosen variant
                       is id-identical to the GT unfolding.

Sections are the maximal contiguous onset spans covered by -2 notes.
"""

from __future__ import annotations

import numpy as np


def _base_pass(note_id: str) -> tuple[str, int]:
    if "-" in note_id:
        base, _, suffix = note_id.rpartition("-")
        if suffix.isdigit():
            return base, int(suffix)
    return note_id, 1


def analyze(records: list[dict]) -> dict:
    """records: score notes {id, onset, duration, pitch, voice} in the PLAYED
    unfolding. Returns {sections: [(start, end, shift)], max_pass}."""
    twos = sorted(
        (r["onset"], r["onset"] + r["duration"]) for r in records if _base_pass(r["id"])[1] >= 2
    )
    sections = []
    for on, off in twos:
        if sections and on <= sections[-1][1] + 2.0:  # merge within 2 quarters
            sections[-1] = (sections[-1][0], max(sections[-1][1], off))
        else:
            sections.append((on, off))
    return {"sections": sections}


def fold(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (folded records = pass-1 only with compacted onsets, section
    descriptors [{start, length, notes}] in FOLDED timeline)."""
    info = analyze(records)
    # spans of pass-2 notes in played timeline, sorted
    spans = info["sections"]

    def shift_of(onset: float) -> float:
        s = 0.0
        for a, b in spans:
            if onset >= b - 1e-9:
                s += b - a
        return s

    folded = []
    for r in records:
        base, p = _base_pass(r["id"])
        if p >= 2:
            continue
        folded.append({**r, "onset": r["onset"] - shift_of(r["onset"])})

    # section descriptors in folded timeline: each played span (a,b) of pass-2
    # notes repeats the material of length (b-a) that ENDS at folded time
    # fold(a) — i.e. source material [fold(a)-(b-a), fold(a)).
    sections = []
    for a, b in spans:
        length = b - a
        fa = a - shift_of(a)
        sections.append({"src_start": fa - length, "src_end": fa, "length": length})
    return folded, sections


def unfold_candidate(folded: list[dict], sections: list[dict], take: tuple[bool, ...]) -> list[dict]:
    """Reconstruct an unfolding in the GT id convention.

    Folded timeline: section k = [src_start, src_end), length L. Taken
    sections repeat immediately after their first pass. With
    shift(x) = Σ L_k over taken sections with src_end ≤ x:
      first-pass note at folded x → played x + shift(x);
      clone of a section-k source note at folded x → played x + shift(src_start_k) + L_k
    (its first pass ends at src_end + shift(src_start_k); the clone block
    follows contiguously; the note keeps its offset x − src_start within it).
    """
    secs = sorted(zip(sections, take), key=lambda st: st[0]["src_start"])

    def shift(x: float) -> float:
        s = 0.0
        for sec, taken in secs:
            if taken and sec["src_end"] <= x + 1e-9:
                s += sec["length"]
        return s

    out = []
    for r in folded:
        out.append({**r, "onset": r["onset"] + shift(r["onset"])})
    for sec, taken in secs:
        if not taken:
            continue
        base_shift = shift(sec["src_start"]) + sec["length"]
        for r in folded:
            if sec["src_start"] - 1e-9 <= r["onset"] < sec["src_end"] - 1e-9:
                base, _p = _base_pass(r["id"])
                out.append({**r, "id": f"{base}-2", "onset": r["onset"] + base_shift})
    out.sort(key=lambda r: (r["onset"], r["pitch"]))
    return out
