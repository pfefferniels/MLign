"""Run an aligner over nASAP and score it against ground truth.

Usage:
  .venv/bin/python eval/run_eval.py --aligner baseline --limit 5
  .venv/bin/python eval/run_eval.py --aligner baseline --out eval/results/baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import AlignmentScore, macro_average  # noqa: E402
from nasap import Alignment, NasapIndex, load_match  # noqa: E402

from mlign import PerfTable, ScoreTable  # noqa: E402


def triples_to_alignment(triples: list[dict]) -> Alignment:
    return Alignment(
        matches=frozenset(
            (t["score_id"], t["perf_id"]) for t in triples if t["label"] == "match"
        ),
        insertions=frozenset(t["perf_id"] for t in triples if t["label"] == "insertion"),
        deletions=frozenset(t["score_id"] for t in triples if t["label"] == "deletion"),
    )


def get_aligner(name: str):
    if name == "baseline":
        from mlign.baseline import align_baseline

        return align_baseline
    raise SystemExit(f"unknown aligner: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=str(ROOT / "data/benchmarks/asap-dataset"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    aligner = get_aligner(args.aligner)
    idx = NasapIndex.build(args.dataset)
    entries = idx.entries[: args.limit] if args.limit else idx.entries

    scores: list[AlignmentScore] = []
    rows = []
    score_cache: dict[str, ScoreTable] = {}
    t0 = time.time()
    failures = 0
    for k, e in enumerate(entries):
        try:
            key = str(e["score"])
            if key not in score_cache:
                score_cache[key] = ScoreTable.from_musicxml(e["score"])
            score = score_cache[key]
            perf = PerfTable.from_midi(e["midi"])
            pred = triples_to_alignment(aligner(score, perf))
            truth = load_match(e["match"])
            s = AlignmentScore.compare(truth, pred)
            scores.append(s)
            rows.append(
                {
                    "piece": e["piece"],
                    "performer": e["performer"],
                    "match_f": s.match.fscore,
                    "insertion_f": s.insertion.fscore,
                    "deletion_f": s.deletion.fscore,
                }
            )
            if args.limit:
                print(f"{e['piece']}/{e['performer']}: {s.row()}")
        except Exception:
            failures += 1
            print(f"FAIL {e['piece']}/{e['performer']}", file=sys.stderr)
            traceback.print_exc(limit=1)
        if (k + 1) % 50 == 0:
            print(f"...{k + 1}/{len(entries)} ({time.time() - t0:.0f}s)", flush=True)

    agg = macro_average(scores)
    agg["failures"] = failures
    agg["aligner"] = args.aligner
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
