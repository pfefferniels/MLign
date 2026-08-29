"""Evaluate an aligner on held-out CORPUS rows (synthetic/selfsup val split).

Fast feedback between benchmark runs: same triple metrics as run_eval, but the
GT comes from the corpus row itself. Uses the last 5% of each file as val
(matching train.py's seed-0 permutation would be nicer, but a fixed tail is
deterministic across runs and the files were written in generation order —
fine for relative comparisons).

Usage:
  .venv/bin/python eval/eval_corpus.py --ckpt runs/v0-syn/best.pt \
      --corpus data/corpus/v0-medium.jsonl --limit 50
  .venv/bin/python eval/eval_corpus.py --aligner baseline --corpus ... --limit 50
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
from nasap import Alignment  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402


def row_tables(r: dict):
    score = ScoreTable.from_records(
        [
            {
                "id": r["scoreIds"][i],
                "pitch": int(r["score"][i][2]),
                "date": r["score"][i][0],
                "duration": r["score"][i][1],
                "part": int(r["score"][i][3]),
            }
            for i in range(len(r["score"]))
        ]
    )
    perf = PerfTable.from_records(
        [
            {
                "perfId": f"p{i}",
                "pitch": int(p[2]),
                "onsetMs": p[0],
                "offsetMs": p[0] + p[1],
                "velocity": int(p[3]),
            }
            for i, p in enumerate(r["perf"])
        ]
    )
    return score, perf


def row_truth(r: dict) -> Alignment:
    return Alignment(
        matches=frozenset((r["scoreIds"][si], f"p{pi}") for si, pi in r["align"]),
        insertions=frozenset(f"p{pi}" for pi, _kind in r["ins"]),
        deletions=frozenset(r["scoreIds"][si] for si in r["del"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--aligner", default="")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--tail-frac", type=float, default=0.05)
    args = ap.parse_args()

    if args.ckpt:
        import torch

        from mlign.infer import align_with_model
        from mlign.model import NoteAligner, config_from_ckpt

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = NoteAligner(config_from_ckpt(cfg, ckpt["model"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        def aligner(score, perf):
            return align_with_model(model, score, perf, device)

        name = f"model:{args.ckpt}"
    elif args.aligner == "baseline":
        from mlign.baseline import align_baseline

        aligner = align_baseline
        name = "baseline"
    else:
        raise SystemExit("pass --ckpt or --aligner baseline")

    rows = []
    for path in args.corpus:
        lines = Path(path).read_text().splitlines()
        tail = max(1, int(len(lines) * args.tail_frac))
        rows.extend(json.loads(line) for line in lines[-tail:] if line.strip())
    rows = rows[: args.limit]

    scores = []
    t0 = time.time()
    for r in rows:
        score, perf = row_tables(r)
        pred_triples = aligner(score, perf)
        pred = Alignment(
            matches=frozenset(
                (t["score_id"], t["perf_id"]) for t in pred_triples if t["label"] == "match"
            ),
            insertions=frozenset(
                t["perf_id"] for t in pred_triples if t["label"] == "insertion"
            ),
            deletions=frozenset(
                t["score_id"] for t in pred_triples if t["label"] == "deletion"
            ),
        )
        scores.append(AlignmentScore.compare(row_truth(r), pred))

    agg = macro_average(scores)
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
