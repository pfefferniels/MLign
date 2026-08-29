"""Bar-C benchmark: Vienna 4x22 repeat pieces from FOLDED scores.

RUMAA's "w/ repeat" condition, symbolic-only: the aligner receives the score
folded to one pass (no repeat realization) and must recover the played
structure. Published symbolic systems collapse here (GlueNote 12.7, Nakamura
36.4 pooled F_align; RUMAA itself 98.4 but from audio).

Construction (eval/folding.py): the GT score IS the played unfolding with
pass-labelled ids, so folding and candidate unfoldings are derived from the GT
itself and reconstruct its exact id space; structure inference picks the
candidate (mlign.repeats pitch-set gain), then the aligner runs on it.
Scored against the untouched GT triples: a wrong structure choice loses all
pass-2 matches — exactly the RUMAA-style penalty.

Usage:
  .venv/bin/python eval/run_4x22_repeats.py --aligner baseline
  .venv/bin/python eval/run_4x22_repeats.py --ckpt runs/v1c/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from folding import fold, unfold_candidate  # noqa: E402
from metrics import AlignmentScore, macro_average, pooled_prf  # noqa: E402
from nasap import Alignment, load_match, score_notes_from_match  # noqa: E402

from mlign.repeats import candidate_score, onset_pitch_sets  # noqa: E402
from mlign.tables import PerfTable, ScoreTable  # noqa: E402

V4X22 = ROOT / "data/benchmarks/vienna4x22"
REPEAT_PIECES = ["Mozart_K331_1st-mov", "Schubert_D783_no15"]


def to_table(records: list[dict]) -> ScoreTable:
    arr = np.empty(
        len(records),
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
    )
    for i, r in enumerate(records):
        arr[i] = (r["onset"], r["duration"], r["pitch"], r["voice"], r["id"])
    return ScoreTable(arr[np.lexsort((arr["pitch"], arr["onset"]))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--oracle-structure", action="store_true",
                    help="skip inference; use the true unfolding (ablation ceiling)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.ckpt:
        import torch

        from mlign.infer import align_with_model
        from mlign.model import NoteAligner, config_from_ckpt

        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        model = NoteAligner(config_from_ckpt(cfg, ckpt["model"]))
        model.load_state_dict(ckpt["model"])
        model.eval()

        def aligner(score, perf):
            return align_with_model(model, score, perf, "cpu")

        name = f"model:{args.ckpt}"
    else:
        from mlign.baseline import align_baseline

        aligner = align_baseline
        name = "baseline"

    scores, pooled, rows = [], [], []
    t0 = time.time()
    for piece in REPEAT_PIECES:
        for mf in sorted((V4X22 / "match").glob(f"{piece}_p*.match")):
            perf_midi = V4X22 / "midi" / f"{mf.stem}.mid"
            if not perf_midi.exists():
                print(f"SKIP {mf.stem}: no midi", file=sys.stderr)
                continue
            truth = load_match(mf)
            records = score_notes_from_match(mf)
            folded, sections = fold(records)
            perf = PerfTable.from_midi(perf_midi)

            # structure inference: rank all 2^k candidates
            perf_sets = onset_pitch_sets(perf.notes["onset"], perf.notes["pitch"], eps=0.05)
            best_take, best_gain = None, -1e18
            for take in itertools.product([False, True], repeat=len(sections)):
                cand = unfold_candidate(folded, sections, take)
                cand_sorted = sorted(cand, key=lambda r: (r["onset"], r["pitch"]))
                onsets = np.array([r["onset"] for r in cand_sorted])
                pitches = np.array([r["pitch"] for r in cand_sorted])
                sets = onset_pitch_sets(onsets, pitches, eps=1e-9)
                # Count prior: self-similar repeats make local gain nearly
                # blind between 1x and 2x; the unexplained-notes fraction is
                # the decisive evidence (razor-thin 0.001 margins observed).
                count_penalty = 0.5 * abs(len(cand_sorted) - len(perf)) / max(1, len(perf))
                gain = candidate_score(sets, perf_sets) - 0.02 * sum(take) - count_penalty
                if gain > best_gain:
                    best_gain, best_take = gain, take
            true_take = tuple([True] * len(sections))
            if args.oracle_structure:
                best_take = true_take
            chosen = unfold_candidate(folded, sections, best_take)

            triples = aligner(to_table(chosen), perf)
            pred = Alignment(
                matches=frozenset((t["score_id"], t["perf_id"]) for t in triples if t["label"] == "match"),
                insertions=frozenset(t["perf_id"] for t in triples if t["label"] == "insertion"),
                deletions=frozenset(t["score_id"] for t in triples if t["label"] == "deletion"),
            )
            s = AlignmentScore.compare(truth, pred)
            scores.append(s)
            pooled.append(pooled_prf(truth, pred)[2])
            rows.append({
                "file": mf.stem, "sections": len(sections),
                "structure_correct": best_take == true_take,
                "match_f": s.match.fscore,
            })
            print(
                f"{mf.stem}: take={best_take} correct={best_take == true_take} F={s.match.fscore:.4f}",
                flush=True,
            )

    agg = macro_average(scores)
    agg["pooled_f_align"] = sum(pooled) / max(len(pooled), 1)
    agg["structure_acc"] = sum(1 for r in rows if r["structure_correct"]) / max(1, len(rows))
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
