"""Does averaging several seeds of one configuration beat any single one of them?

Three seeds of orn4+factored score .4784, .4162 and .3892 decoded group-exact on
clean Batik — a between-run sd of .0457 against an effect of about .10. That
spread is the condition under which averaging helps: the runs disagree, and if
their errors are even partly independent the average has less of them.

`infer.accumulate` already averages logits over the windows covering a played
note. Averaging over MODELS is the same operation one level up, on the same
tensors, so nothing about the decode changes — it is handed one `Evidence` built
from several forward passes instead of one.

Selecting a checkpoint on realorn-batik would be selecting on the benchmark, so
this reports the ensemble beside the members rather than picking a winner.

  .venv/bin/python eval/ensemble_attribution.py --ckpts runs/v15fact/best.pt \
      runs/v16seed/best.pt runs/v16seed2/best.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import parse_row  # noqa: E402
from mlign.infer import Evidence, Ornaments, accumulate, decode  # noqa: E402
from run_attribution_decoded import NOT_ATTRIBUTED, truth  # noqa: E402
from run_attribution import load_model  # noqa: E402


def mean_evidence(evs: list[Evidence]) -> Evidence:
    """Average the members' logits, the way windows are already averaged.

    `rank` is renormalised after averaging for the same reason a windowed row is:
    a mean of normalised rows is not itself normalised.
    """
    sim = np.mean([e.sim for e in evs], axis=0)
    null_s = np.mean([e.null_s for e in evs], axis=0)
    null_p = np.mean([e.null_p for e in evs], axis=0)
    orn = None
    if all(e.ornaments is not None for e in evs):
        rank = np.mean([e.ornaments.rank for e in evs], axis=0)
        peak = rank.max(axis=1, keepdims=True)
        rank = rank - (peak + np.log(np.exp(rank - peak).sum(axis=1, keepdims=True)))
        orn = Ornaments(rank=rank,
                        log_gate=np.mean([e.ornaments.log_gate for e in evs], axis=0))
    return Evidence(sim=sim, null_s=null_s, null_p=null_p, ornaments=orn)


def outcomes(evidence_of, rows: list[dict]) -> tuple[np.ndarray, int, int]:
    """Per-figure exactness, plus notes called ornaments and how many are wrong."""
    exact, called, false = [], 0, 0
    for row in rows:
        gt = truth(row)
        if not gt:
            continue
        ev = evidence_of(row)
        triples = decode(row, ev.sim, ev.null_s, ev.null_p, ornaments=ev.ornaments)
        pred = {t["perf_idx"]: (t.get("ornament") or {}).get("anchor_score_idx", NOT_ATTRIBUTED)
                for t in triples if t["label"] == "insertion"}
        matched = {int(pi) for _, pi in row["align"]}
        for pi, anchor in pred.items():
            if anchor != NOT_ATTRIBUTED:
                called += 1
                false += pi in matched
        figures: dict[int, list[int]] = collections.defaultdict(list)
        for pi, anchor in gt.items():
            figures[anchor].append(pi)
        exact += [all(pred.get(pi, NOT_ATTRIBUTED) == anchor for pi in pis)
                  for anchor, pis in sorted(figures.items())]
    return np.array(exact, dtype=bool), called, false


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--corpus", default="batik")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [parse_row(l) for l in open(ROOT / f"data/corpus/realorn-{args.corpus}.jsonl", "rb")
            if l.strip()]
    rows = [r for r in rows if str(r.get("meta", {}).get("gen", "")).startswith("realorn-")]
    if args.limit:
        rows = rows[: args.limit]

    models = [load_model(c, args.device) for c in args.ckpts]
    # One forward pass per model per row, reused by the members and the ensemble.
    cache: dict[int, list[Evidence]] = {}

    def evidence_for(i: int, row: dict) -> list[Evidence]:
        if i not in cache:
            cache[i] = [accumulate(m, row, args.device) for m in models]
        return cache[i]

    print(f"{'system':34s} {'group-exact':>12} {'called orn':>11} {'false':>8}")
    per: list[np.ndarray] = []
    for k, ckpt in enumerate(args.ckpts):
        ex, called, false = outcomes(
            lambda row, k=k: evidence_for(id(row), row)[k], rows)
        per.append(ex)
        print(f"  {ckpt:32s} {ex.mean():12.4f} {called:11d} {false / max(called,1):8.4f}")

    ex, called, false = outcomes(lambda row: mean_evidence(evidence_for(id(row), row)), rows)
    print(f"  {'ENSEMBLE of ' + str(len(models)):32s} {ex.mean():12.4f} {called:11d} "
          f"{false / max(called,1):8.4f}")

    # Is beating the best member real, or a resample away from it? Paired over
    # the same figures, so between-figure variance drops out.
    rng = np.random.default_rng(0)
    top = int(np.argmax([m.mean() for m in per]))
    d = ex.astype(float) - per[top].astype(float)
    draws = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    pv = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    print(f"\n  ensemble vs its best member ({args.ckpts[top]}):")
    print(f"    diff {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  p={max(pv, 5e-5):.5f}"
          f"  gained {int((ex & ~per[top]).sum())} lost {int((per[top] & ~ex).sum())}")

    best = max(m.mean() for m in per)
    print(f"\n  best member {best:.4f}   mean member {np.mean([m.mean() for m in per]):.4f}"
          f"   ensemble {ex.mean():.4f}   ensemble - best {ex.mean() - best:+.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps({
            "ckpts": args.ckpts, "corpus": args.corpus,
            "members": [round(float(m.mean()), 4) for m in per],
            "ensemble": round(float(ex.mean()), 4),
            "false_on_matched": round(false / max(called, 1), 4),
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
