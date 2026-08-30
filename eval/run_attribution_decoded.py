"""Ornament attribution scored through the DECODE, not through the head's argmax.

`run_attribution.py` reads `logits_attr.argmax(-1)` directly. That is the head's
own answer and it is dominated by the match head: the conditioned none column
carries `log_matched`, so a played note the match head believes it has matched
is silenced whatever the attribution head thinks. On real Batik that silences
48.8 % of all ornament figures, and vetoed accuracy is .0000 by construction.

The deployed pipeline does not work that way. It decides matches with a
per-pitch monotone assignment, and every played note left over is an insertion.
For those notes `log_matched` is known to be zero, so the head's remaining two
factors — `gate` = P(elaborates a written note | insertion) and the ranking over
score notes — are exactly the right posterior. This script scores that pipeline.

Both numbers are worth having. The head number says what the head learned; this
one says what a user gets.

Usage:
  .venv/bin/python eval/run_attribution_decoded.py --ckpt runs/v12both/best.pt \
      --corpus data/corpus/realorn-batik.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import parse_row  # noqa: E402
from mlign.infer import ORNAMENT_MIN_PROB, accumulate, decode  # noqa: E402
from run_attribution import load_model  # noqa: E402

NOT_ATTRIBUTED = -1


def truth(row: dict) -> dict[int, int]:
    """Played-note index → anchor score index, for the notes this row can score.

    The same rule `dataset.featurize` applies to `realorn-*`: an insertion the
    derivation could not resolve is ignored, never counted as "not an ornament".
    """
    return {int(pi): int(anchor)
            for pi, anchor, *_ in row.get("orn", ())
            if int(anchor) >= 0}


def known_plain(row: dict) -> set[int]:
    """Played notes known NOT to be inserted ornament notes: the matched ones.

    Recall alone would reward calling every insertion an ornament, and the
    figures are scored only where ground truth exists. These are the notes a
    false positive can be counted against without inventing a label — an
    unresolved insertion stays out of both sets.
    """
    return {int(pi) for _, pi in row["align"]}


def predict(model, row: dict, device: str, min_prob: float) -> dict[int, int]:
    """Played-note index → anchor score index, as the pipeline would report it."""
    ev = accumulate(model, row, device)
    triples = decode(row, ev.sim, ev.null_s, ev.null_p, ornaments=ev.ornaments,
                     ornament_min_prob=min_prob)
    out = {}
    for t in triples:
        if t["label"] != "insertion":
            continue
        orn = t.get("ornament")
        out[t["perf_idx"]] = orn["anchor_score_idx"] if orn else NOT_ATTRIBUTED
    return out


def evaluate(model, rows: list[dict], device: str, min_prob: float) -> dict:
    hit = tot = 0
    exact = groups = 0
    lost_to_match = 0  # a true ornament note the decode called a match
    by_size: collections.Counter = collections.Counter()
    by_size_exact: collections.Counter = collections.Counter()

    called_orn = false_orn = 0
    for row in rows:
        gt = truth(row)
        if not gt:
            continue
        pred = predict(model, row, device, min_prob)
        plain = known_plain(row)
        for pi, anchor in pred.items():
            if anchor == NOT_ATTRIBUTED:
                continue
            called_orn += 1
            false_orn += pi in plain
        for pi, anchor in gt.items():
            tot += 1
            got = pred.get(pi, NOT_ATTRIBUTED)
            hit += got == anchor
            lost_to_match += pi not in pred
        figures: dict[int, list[int]] = collections.defaultdict(list)
        for pi, anchor in gt.items():
            figures[anchor].append(pi)
        for anchor, pis in figures.items():
            groups += 1
            ok = all(pred.get(pi, NOT_ATTRIBUTED) == anchor for pi in pis)
            exact += ok
            size = min(len(pis), 9)
            by_size[size] += 1
            by_size_exact[size] += ok

    return {
        "notes": tot,
        "note_acc": round(hit / max(tot, 1), 4),
        "groups": groups,
        "group_exact": round(exact / max(groups, 1), 4),
        "called_match_by_decode": round(lost_to_match / max(tot, 1), 4),
        "called_ornament": called_orn,
        # of everything the pipeline called an ornament, the share that is a
        # note the match file says was matched, so certainly not one
        "false_on_matched": round(false_orn / max(called_orn, 1), 4),
        "by_size": {("9+" if s == 9 else str(s)): {
            "groups": by_size[s],
            "group_exact": round(by_size_exact[s] / by_size[s], 4),
        } for s in sorted(by_size)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-prob", type=float, default=ORNAMENT_MIN_PROB,
                    help="gate threshold: P(elaborates a written note | insertion)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [parse_row(l) for l in open(args.corpus, "rb") if l.strip()]
    rows = [r for r in rows if str(r.get("meta", {}).get("gen", "")).startswith("realorn-")]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"{args.corpus} carries no realorn-* rows to score.")

    res = evaluate(load_model(args.ckpt, args.device), rows, args.device, args.min_prob)
    res |= {"ckpt": args.ckpt, "corpus": args.corpus, "rows": len(rows),
            "min_prob": args.min_prob}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
