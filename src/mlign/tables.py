"""The aligner's two input tables.

The aligner core is format-agnostic: it consumes a ScoreTable and a PerfTable
and returns alignment triples over their ids. Everything format-specific lives
in the constructors:

  ScoreTable.from_musicxml   benchmark path — partitura, merge_parts +
                             unfold_part_maximal, reproducing the nASAP id
                             space ("n2-1" repeat-pass ids) exactly.
  ScoreTable.from_records    synthetic path — espressivo score notes (already
                             unfolded, ids chained by the converter).
  PerfTable.from_midi        benchmark path — partitura.load_performance_midi,
                             reproducing nASAP's performance ids ("n0"...)
                             exactly (verified against match files).
  PerfTable.from_records     synthetic path — robustness-layer perfNotes rows.

Times: score in quarters (tempo-free), performance in seconds. Both tables are
sorted by (onset, pitch) and keep a stable integer index; ids are per-format
strings only used at the boundaries.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

_SCORE_DTYPE = np.dtype(
    [
        ("onset", "f8"),  # quarters, unfolded timeline
        ("duration", "f8"),  # quarters
        ("pitch", "i4"),
        ("voice", "i4"),
        ("id", "U32"),
    ]
)

_PERF_DTYPE = np.dtype(
    [
        ("onset", "f8"),  # seconds
        ("duration", "f8"),  # seconds
        ("pitch", "i4"),
        ("velocity", "i4"),
        ("id", "U32"),
    ]
)


def _sorted(arr: np.ndarray) -> np.ndarray:
    return arr[np.lexsort((arr["pitch"], arr["onset"]))]


@dataclass(frozen=True)
class ScoreTable:
    notes: np.ndarray  # _SCORE_DTYPE, sorted by (onset, pitch)

    @classmethod
    def from_musicxml(cls, path) -> "ScoreTable":
        import partitura

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = partitura.load_score(str(path))
            part = partitura.score.merge_parts(score.parts)
            unfolded = partitura.score.unfold_part_maximal(part)
            na = unfolded.note_array()
        out = np.empty(len(na), dtype=_SCORE_DTYPE)
        out["onset"] = na["onset_quarter"]
        out["duration"] = na["duration_quarter"]
        out["pitch"] = na["pitch"]
        out["voice"] = na["voice"]
        out["id"] = na["id"]
        return cls(_sorted(out))

    @classmethod
    def from_records(cls, records, ppq: int = 720) -> "ScoreTable":
        """espressivo-side score notes: [{id, pitch, date, duration, part}, ...]."""
        out = np.empty(len(records), dtype=_SCORE_DTYPE)
        for i, r in enumerate(records):
            out[i] = (r["date"] / ppq, r["duration"] / ppq, r["pitch"], r.get("part", 0), r["id"])
        return cls(_sorted(out))

    def __len__(self) -> int:
        return len(self.notes)


@dataclass(frozen=True)
class PerfTable:
    notes: np.ndarray  # _PERF_DTYPE, sorted by (onset, pitch)

    @classmethod
    def from_midi(cls, path) -> "PerfTable":
        import partitura

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            perf = partitura.load_performance_midi(str(path))
            na = perf.note_array()
        out = np.empty(len(na), dtype=_PERF_DTYPE)
        out["onset"] = na["onset_sec"]
        out["duration"] = na["duration_sec"]
        out["pitch"] = na["pitch"]
        out["velocity"] = na["velocity"]
        out["id"] = na["id"]
        return cls(_sorted(out))

    @classmethod
    def from_records(cls, records) -> "PerfTable":
        """Robustness-layer perfNotes rows: [{perfId, pitch, onsetMs, offsetMs, velocity}, ...]."""
        out = np.empty(len(records), dtype=_PERF_DTYPE)
        for i, r in enumerate(records):
            out[i] = (
                r["onsetMs"] / 1000.0,
                (r["offsetMs"] - r["onsetMs"]) / 1000.0,
                r["pitch"],
                r["velocity"],
                r["perfId"],
            )
        return cls(_sorted(out))

    def __len__(self) -> int:
        return len(self.notes)
