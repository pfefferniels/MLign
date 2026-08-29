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
    """Returns f(entry, score, perf) → triples."""
    if name == "baseline":
        from mlign.baseline import align_baseline

        return lambda entry, score, perf: align_baseline(score, perf)
    if name in ("dualdtw", "automatic", "gluenote"):
        return make_parangonar_aligner(name)
    if name.startswith("model:"):
        return make_model_aligner(name.split(":", 1)[1])
    raise SystemExit(f"unknown aligner: {name}")


def make_model_aligner(ckpt_path: str):
    import torch

    from mlign.infer import align_with_model
    from mlign.model import NoteAligner, config_from_ckpt

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    # The attribution head is inferred from the weights, not the config: a
    # checkpoint may predate `attribution` being recorded. It plays no part in
    # alignment decoding — it is built only so the state dict loads strictly,
    # which keeps a genuine architecture mismatch an error rather than a
    # silently half-loaded model.
    model = NoteAligner(config_from_ckpt(cfg, ckpt["model"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def align(entry, score, perf):
        return align_with_model(model, score, perf, device)

    return align


def make_parangonar_aligner(name: str):
    """parangonar baselines, fed exactly like MLign (same unfolding, same ids)."""
    import contextlib
    import io
    import warnings

    import parangonar as pa
    import partitura

    matcher = {
        "dualdtw": pa.DualDTWNoteMatcher,
        "automatic": pa.AutomaticNoteMatcher,
        "gluenote": pa.TheGlueNoteMatcher,
    }[name]()

    def align(entry, score, perf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = partitura.load_score(str(entry["score"]))
            part = partitura.score.merge_parts(s.parts)
            unfolded = partitura.score.unfold_part_maximal(part)
            sna = unfolded.note_array(include_grace_notes=True)
            pna = partitura.load_performance_midi(str(entry["midi"])).note_array()
            kwargs = {}
            if name == "dualdtw":
                kwargs = {"process_ornaments": False, "score_part": unfolded}
            with contextlib.redirect_stdout(io.StringIO()):
                pred = matcher(sna, pna, **kwargs)
        out = []
        for d in pred:
            if d["label"] == "match":
                out.append({"label": "match", "score_id": d["score_id"], "perf_id": d["performance_id"]})
            elif d["label"] == "deletion":
                out.append({"label": "deletion", "score_id": d["score_id"]})
            elif d["label"] in ("insertion", "ornament"):
                out.append({"label": "insertion", "perf_id": d["performance_id"]})
        return out

    return align


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default=str(ROOT / "data/benchmarks/asap-dataset"))
    ap.add_argument("--out", default="")
    ap.add_argument("--robust-only", action="store_true",
                    help="only the ~78%% of alignments flagged robust (TISMIR rec.)")
    ap.add_argument("--split", choices=["all", "test", "train"], default="all",
                    help="MAESTRO-v2-derived piece split (eval/split.py)")
    args = ap.parse_args()

    aligner = get_aligner(args.aligner)
    idx = NasapIndex.build(args.dataset, robust_only=args.robust_only)
    if args.split != "all":
        from split import test_split

        folders, _ = test_split()
        keep = (lambda p: p in folders) if args.split == "test" else (lambda p: p not in folders)
        idx.entries = [e for e in idx.entries if keep(e["piece"])]
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
            pred = triples_to_alignment(aligner(e, score, perf))
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
