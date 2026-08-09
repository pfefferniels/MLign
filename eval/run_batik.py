"""Batik-plays-Mozart benchmark: 36 Mozart sonata movements, matchfile v1.0.0 GT.

Same protocol as run_eval (nASAP): load_match GT, aligner fed the MusicXML
score (merged+maximally unfolded) and the performance MIDI.

Usage:
  .venv/bin/python eval/run_batik.py --aligner baseline
  .venv/bin/python eval/run_batik.py --ckpt runs/v0-syn/best.pt --limit 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import AlignmentScore, macro_average  # noqa: E402
from nasap import load_match  # noqa: E402
from run_eval import get_aligner, triples_to_alignment  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402

BATIK = ROOT / "data/benchmarks/batik_plays_mozart"


def score_from_match(match_path: Path) -> ScoreTable:
    import numpy as np

    from nasap import score_notes_from_match

    records = score_notes_from_match(match_path)
    out = np.empty(
        len(records),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
    )
    for i, r in enumerate(records):
        out[i] = (r["onset"], r["duration"], r["pitch"], r["voice"], r["id"])
    return ScoreTable(out[np.lexsort((out["pitch"], out["onset"]))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="baseline")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    name = f"model:{args.ckpt}" if args.ckpt else args.aligner
    aligner = get_aligner(name)

    matches = sorted((BATIK / "match").glob("*.match"))
    if args.limit:
        matches = matches[: args.limit]

    scores, rows = [], []
    t0 = time.time()
    for mf in matches:
        stem = mf.stem
        entry = {
            "score": BATIK / "scores" / f"{stem}.musicxml",
            "midi": BATIK / "midi" / f"{stem}.mid",
        }
        if not entry["score"].exists() or not entry["midi"].exists():
            print(f"SKIP {stem}: missing files", file=sys.stderr)
            continue
        try:
            # Score from the match file (create_score=True): Batik's GT id space
            # is the PERFORMED unfolding (some repeats taken), not the maximal
            # one — parangonar's papers evaluate the same way. MusicXML in
            # scores/ is only the notation source.
            score = score_from_match(mf)
            perf = PerfTable.from_midi(entry["midi"])
            triples = aligner(entry, score, perf)
            # Batik matchfiles number performance notes n1.. (1-based); partitura
            # yields P01_n0.. for these single-track MIDIs (verified on kv279_1:
            # P01_n{k} == GT n{k+1} by pitch+tick). Remap before scoring.
            for t in triples:
                pid = t.get("perf_id")
                if pid and pid.startswith("P01_n"):
                    t["perf_id"] = f"n{int(pid[5:]) + 1}"
            pred = triples_to_alignment(triples)
            truth = load_match(mf)
            s = AlignmentScore.compare(truth, pred)
            scores.append(s)
            rows.append({"piece": stem, "match_f": s.match.fscore})
            print(f"{stem}: {s.row()}", flush=True)
        except Exception as err:
            print(f"FAIL {stem}: {err}", file=sys.stderr)

    agg = macro_average(scores)
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
