"""Vienna 4x22 benchmark from TheGlueNote's preprocessed .npz files.

Cheap (no MusicXML/MIDI parsing — note arrays precomputed) and exactly the
data TheGlueNote's paper evaluated on: 88 performances, 4 pieces × 22 pianists.

Usage:
  .venv/bin/python eval/run_4x22.py --aligner baseline
  .venv/bin/python eval/run_4x22.py --ckpt runs/v0-syn/best.pt --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from metrics import AlignmentScore, macro_average  # noqa: E402
from nasap import Alignment  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402

NPZ_DIR = Path("/Users/nielspfeffer/Projects/thegluenote-main/data/testing/4x22")


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    sna = d["score_note_array"]
    pna = d["performance_note_array"]

    score_arr = np.empty(
        len(sna),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
    )
    score_arr["onset"] = sna["onset_quarter"]
    score_arr["duration"] = sna["duration_quarter"]
    score_arr["pitch"] = sna["pitch"]
    score_arr["voice"] = sna["voice"]
    score_arr["id"] = sna["id"]

    perf_arr = np.empty(
        len(pna),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U32")],
    )
    perf_arr["onset"] = pna["onset_sec"]
    perf_arr["duration"] = pna["duration_sec"]
    perf_arr["pitch"] = pna["pitch"]
    perf_arr["velocity"] = pna["velocity"]
    perf_arr["id"] = pna["id"]

    order_s = np.lexsort((score_arr["pitch"], score_arr["onset"]))
    order_p = np.lexsort((perf_arr["pitch"], perf_arr["onset"]))
    score = ScoreTable(score_arr[order_s])
    perf = PerfTable(perf_arr[order_p])

    matches, insertions, deletions = set(), set(), set()
    for rec in d["gt_alignment"]:
        if rec["label"] == "match":
            matches.add((rec["score_id"], rec["performance_id"]))
        elif rec["label"] == "insertion":
            insertions.add(rec["performance_id"])
        elif rec["label"] == "deletion":
            deletions.add(rec["score_id"])
    truth = Alignment(frozenset(matches), frozenset(insertions), frozenset(deletions))
    return score, perf, truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.ckpt:
        import torch

        from mlign.infer import align_with_model
        from mlign.model import ModelConfig, NoteAligner

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = NoteAligner(
            ModelConfig(
                d_model=cfg.get("d_model", 192),
                n_layers=cfg.get("n_layers", 4),
                matchability=cfg.get("matchability", False),
            )
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        def aligner(score, perf):
            return align_with_model(model, score, perf, device)

        name = f"model:{args.ckpt}"
    else:
        from mlign.baseline import align_baseline

        aligner = align_baseline
        name = "baseline"

    files = sorted(NPZ_DIR.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]

    scores, rows = [], []
    t0 = time.time()
    for f in files:
        try:
            score, perf, truth = load_npz(f)
        except Exception as err:
            print(f"SKIP {f.name}: {err}", file=sys.stderr)
            continue
        triples = aligner(score, perf)
        pred = Alignment(
            matches=frozenset((t["score_id"], t["perf_id"]) for t in triples if t["label"] == "match"),
            insertions=frozenset(t["perf_id"] for t in triples if t["label"] == "insertion"),
            deletions=frozenset(t["score_id"] for t in triples if t["label"] == "deletion"),
        )
        s = AlignmentScore.compare(truth, pred)
        scores.append(s)
        rows.append({"file": f.name, "match_f": s.match.fscore})

    agg = macro_average(scores)
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    worst = sorted(rows, key=lambda r: r["match_f"])[:5]
    print("worst:", json.dumps(worst))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
