"""dev-long: second local dev tier — long, dense, deletion-heavy repertoire.

Vienna 4x22 (dev tier 1) is 4 short pieces ≤~1000 notes; it rewarded a model
(v5real-e5) that then regressed on the holdout's long sonata movements. This
tier picks ~20 train-split (never test) performances that look like the
holdout's failure class — Beethoven/Liszt/Schumann/Chopin movements ≥2000
score notes, preferring higher deletion rates — and runs the FULL pipeline
(partitura score/perf, windowed inference, decode) exactly as the holdout
does. Deterministic; ~2-4 min per performance.

Usage:
  .venv/bin/python eval/run_devlong.py --ckpt runs/v6real2/snap-e005.pt
  .venv/bin/python eval/run_devlong.py --aligner dualdtw   (reference)
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
from nasap import NasapIndex, load_match  # noqa: E402
from run_eval import get_aligner, triples_to_alignment  # noqa: E402
from split import test_split  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402

TARGET_COMPOSERS = ("Beethoven", "Liszt", "Schumann", "Chopin")
MIN_SCORE_NOTES = 2000
N_PERFS = 20
CACHE = ROOT / "eval" / "devlong-set.json"


def select_set() -> list[dict]:
    """Deterministic selection; cached so the tier never drifts."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    folders, _ = test_split()
    idx = NasapIndex.build(ROOT / "data/benchmarks/asap-dataset", robust_only=True)
    cands = []
    for e in idx.entries:
        comp = e["piece"].split("/")[0]
        if e["piece"] in folders or comp not in TARGET_COMPOSERS:
            continue
        t = load_match(e["match"])
        n_score = len(t.matches) + len(t.deletions)
        if n_score < MIN_SCORE_NOTES:
            continue
        cands.append({
            "piece": e["piece"], "performer": e["performer"],
            "n_score": n_score, "del_rate": len(t.deletions) / max(1, n_score),
        })
    # per composer proportional to holdout, deletion-heavy first, ≤2 per piece
    quota = {"Beethoven": 9, "Liszt": 5, "Chopin": 4, "Schumann": 2}
    chosen, per_piece = [], {}
    for comp, want in quota.items():
        pool = sorted((c for c in cands if c["piece"].startswith(comp + "/")), key=lambda c: -c["del_rate"])
        got = 0
        for c in pool:
            if got >= want:
                break
            if per_piece.get(c["piece"], 0) >= 2:
                continue
            per_piece[c["piece"]] = per_piece.get(c["piece"], 0) + 1
            chosen.append(c)
            got += 1
    CACHE.write_text(json.dumps(chosen, indent=1))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligner", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    name = f"model:{args.ckpt}" if args.ckpt else (args.aligner or "baseline")
    aligner = get_aligner(name)

    idx = NasapIndex.build(ROOT / "data/benchmarks/asap-dataset", robust_only=True)
    by_key = {(e["piece"], e["performer"]): e for e in idx.entries}
    chosen = select_set()
    scores, rows = [], []
    t0 = time.time()
    for c in chosen:
        e = by_key.get((c["piece"], c["performer"]))
        if e is None:
            continue
        score = ScoreTable.from_musicxml(e["score"])
        perf = PerfTable.from_midi(e["midi"])
        pred = triples_to_alignment(aligner(e, score, perf))
        s = AlignmentScore.compare(load_match(e["match"]), pred)
        scores.append(s)
        rows.append({"piece": c["piece"], "performer": c["performer"], "match_f": s.match.fscore,
                     "n_score": c["n_score"], "del_rate": round(c["del_rate"], 3)})
        print(f"{c['piece']}/{c['performer']} (n={c['n_score']}, del {c['del_rate']:.2f}): F {s.match.fscore:.4f}", flush=True)
    agg = macro_average(scores)
    agg["aligner"] = name
    agg["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(agg, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"aggregate": agg, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
