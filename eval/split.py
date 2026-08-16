"""The nASAP evaluation split (Matchmaker convention: MAESTRO v2 test ∩ nASAP).

Computed locally: 74 performances / 39 piece folders whose
maestro_midi_performance is in MAESTRO v2's test split. (Matchmaker reports
43 pieces / 59 performances — likely a different nASAP version or extra
filtering); the numbers here are deterministic from the two CSVs.

Everything NOT in the test-piece folders is fair training material.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAESTRO_CSV = ROOT / "data/benchmarks/maestro-v2.0.0.csv"
ASAP_META = ROOT / "data/benchmarks/asap-dataset/metadata.csv"


def maestro_test_files() -> set[str]:
    out = set()
    with open(MAESTRO_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test":
                out.add(row["midi_filename"])
    return out


def test_split() -> tuple[set[str], set[str]]:
    """Returns (test piece folders, test performance midi paths rel. to asap root)."""
    mtest = maestro_test_files()
    folders: set[str] = set()
    perfs: set[str] = set()
    with open(ASAP_META, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mm = row.get("maestro_midi_performance", "")
            if mm and mm.replace("{maestro}/", "") in mtest:
                folders.add(row["folder"])
                perfs.add(row["midi_performance"])
    return folders, perfs


def is_test_piece(piece_folder: str) -> bool:
    folders, _ = _cached()
    return piece_folder in folders


_cache: tuple[set[str], set[str]] | None = None


def _cached():
    global _cache
    if _cache is None:
        _cache = test_split()
    return _cache
