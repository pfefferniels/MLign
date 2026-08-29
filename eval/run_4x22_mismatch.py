"""Public Bar-B benchmark: Vienna 4x22 with controlled performance mismatch.

TheGlueNote's hard-set protocol (ISMIR 2024 Table 4, "20% mismatch data") runs
on proprietary Magaloff/Zeilinger pieces. This reproduces the *condition* on
public data: each 4x22 performance gets 20% note deletions + 20% insertions
(uniform random pitch/onset like their augmentation), GT adjusted accordingly.
Deterministic per (file, seed) → comparable across aligners.

The mismatch is applied to the PERFORMANCE side only; the score stays intact.
GT transformation: deleted perf note → its match becomes a deletion; inserted
note → labelled insertion. (Their protocol corrupts one side the same way.)

Usage:
  .venv/bin/python eval/run_4x22_mismatch.py --aligner baseline
  .venv/bin/python eval/run_4x22_mismatch.py --ckpt runs/v1/best.pt
  .venv/bin/python eval/run_4x22_mismatch.py --aligner dualdtw   # heavy: parangonar
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

from metrics import AlignmentScore, macro_average, pooled_prf  # noqa: E402
from nasap import Alignment  # noqa: E402
from run_4x22 import NPZ_DIR, load_npz  # noqa: E402

from mlign.tables import PerfTable  # noqa: E402


def apply_mismatch(perf: PerfTable, truth: Alignment, rate: float, seed: int,
                   mode: str = "contiguous"):
    """mode='contiguous' reproduces TheGlueNote's OOD condition (ISMIR 2024
    §5): extended contiguous mismatch segments — one deleted span (the
    performance omits a passage) + one inserted span (a passage not in the
    score), each ~rate/2 of the notes. mode='uniform' scatters instead."""
    rng = np.random.default_rng(seed)
    notes = perf.notes
    m = len(notes)

    if mode == "contiguous":
        k_del = max(1, int(rate / 2 * m))
        start = int(rng.integers(0, max(1, m - k_del)))
        del_idx = set(range(start, start + k_del))
    else:
        k_del = int(rate * m)
        del_idx = set(rng.choice(m, size=k_del, replace=False).tolist())
    kept = [i for i in range(m) if i not in del_idx]
    deleted_ids = {str(notes["id"][i]) for i in del_idx}

    # insertions: random-note passage(s) not in the score
    p_lo, p_hi = int(notes["pitch"].min()), int(notes["pitch"].max()) + 1
    t_lo, t_hi = float(notes["onset"].min()), float(notes["onset"].max())
    if mode == "contiguous":
        k_ins = max(1, int(rate / 2 * len(kept)))
        anchor = float(rng.uniform(t_lo, t_hi * 0.8))
        ioi = rng.uniform(0.04, 0.25, k_ins)
        onsets_ins = anchor + np.cumsum(ioi)
    else:
        k_ins = int(rate * len(kept))
        onsets_ins = rng.uniform(t_lo, t_hi, k_ins)
    ins = np.empty(k_ins, dtype=notes.dtype)
    ins["onset"] = onsets_ins
    ins["duration"] = rng.uniform(0.05, 0.6, k_ins)
    ins["pitch"] = rng.integers(p_lo, p_hi, k_ins)
    ins["velocity"] = rng.integers(30, 100, k_ins)
    ins_ids = [f"mm{k}" for k in range(k_ins)]
    ins["id"] = ins_ids

    out = np.concatenate([notes[kept], ins])
    out = out[np.lexsort((out["pitch"], out["onset"]))]

    new_matches = {(s, p) for (s, p) in truth.matches if p not in deleted_ids}
    newly_deleted = {s for (s, p) in truth.matches if p in deleted_ids}
    new_truth = Alignment(
        matches=frozenset(new_matches),
        insertions=frozenset(
            {p for p in truth.insertions if p not in deleted_ids} | set(ins_ids)
        ),
        deletions=frozenset(truth.deletions | newly_deleted),
    )
    return PerfTable(out), new_truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--rate", type=float, default=0.2)
    ap.add_argument("--mode", choices=["contiguous", "uniform"], default="contiguous")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.ckpt:
        import torch

        from mlign.infer import align_with_model
        from mlign.model import NoteAligner, config_from_ckpt

        device = "cpu"
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
    elif args.aligner in ("dualdtw", "automatic", "gluenote"):
        import contextlib
        import io
        import warnings

        import parangonar as pa

        matcher = {
            "dualdtw": pa.DualDTWNoteMatcher,
            "automatic": pa.AutomaticNoteMatcher,
            "gluenote": pa.TheGlueNoteMatcher,
        }[args.aligner]()

        def aligner(score, perf, _m=matcher):
            # full field set parangonar's matchers touch (incl. is_grace)
            sna = np.empty(len(score.notes), dtype=[
                ("onset_beat", "f8"), ("duration_beat", "f8"), ("pitch", "i4"),
                ("voice", "i4"), ("id", "U32"), ("is_grace", "b"),
            ])
            for f_dst, f_src in [("onset_beat", "onset"), ("duration_beat", "duration"),
                                 ("pitch", "pitch"), ("voice", "voice"), ("id", "id")]:
                sna[f_dst] = score.notes[f_src]
            sna["is_grace"] = 0
            pna = np.empty(len(perf.notes), dtype=[
                ("onset_sec", "f8"), ("duration_sec", "f8"), ("pitch", "i4"),
                ("velocity", "i4"), ("id", "U32"),
            ])
            for f_dst, f_src in [("onset_sec", "onset"), ("duration_sec", "duration"),
                                 ("pitch", "pitch"), ("velocity", "velocity"), ("id", "id")]:
                pna[f_dst] = perf.notes[f_src]
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                pred = _m(sna, pna)
            out = []
            for d in pred:
                if d["label"] == "match":
                    out.append({"label": "match", "score_id": d["score_id"], "perf_id": d["performance_id"]})
                elif d["label"] == "deletion":
                    out.append({"label": "deletion", "score_id": d["score_id"]})
                elif d["label"] in ("insertion", "ornament"):
                    out.append({"label": "insertion", "perf_id": d["performance_id"]})
            return out

        name = args.aligner
    else:
        raise SystemExit("pass --ckpt or --aligner")

    files = sorted(NPZ_DIR.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]

    scores, pooled, rows = [], [], []
    t0 = time.time()
    for k, f in enumerate(files):
        try:
            score, perf, truth = load_npz(f)
        except Exception as err:
            print(f"SKIP {f.name}: {err}", file=sys.stderr)
            continue
        perf2, truth2 = apply_mismatch(perf, truth, args.rate, args.seed + k, args.mode)
        triples = aligner(score, perf2)
        pred = Alignment(
            matches=frozenset((t["score_id"], t["perf_id"]) for t in triples if t["label"] == "match"),
            insertions=frozenset(t["perf_id"] for t in triples if t["label"] == "insertion"),
            deletions=frozenset(t["score_id"] for t in triples if t["label"] == "deletion"),
        )
        s = AlignmentScore.compare(truth2, pred)
        scores.append(s)
        pooled.append(pooled_prf(truth2, pred)[2])
        rows.append({"file": f.name, "match_f": s.match.fscore})

    agg = macro_average(scores)
    agg["pooled_f_align"] = sum(pooled) / max(len(pooled), 1)
    agg["aligner"] = name
    agg["rate"] = args.rate
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
