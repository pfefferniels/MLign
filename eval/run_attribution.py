"""Ornament attribution: how well does the model say WHICH score note a played
ornament note belongs to.

Two tiers, and the distinction is the whole point of reading these numbers.

**Synthetic (default).** Held-out espressivo renders, `meta.gen = mlign-*`,
with exhaustive provenance. They say "the head learned the generator's ornament
model", NOT "the head works on real recordings" — a holdout drawn from the same
generator measures the generator. Keep the generation seed disjoint from
training.

**Real recordings (`--real-orn`).** `meta.gen = realorn-*`, derived from
Nakamura match files by `scripts/corpus/real_orn_gt.py`. The long-standing
claim that no real corpus annotates ornament→principal attribution was wrong:
it is true that there are zero `ornament(` lines anywhere, but both ASAP and
Batik put the sign in the **snote's attribute list** and the played ornament
notes follow as `insertion-note` lines. Those labels are PARTIAL — an
unattributed insertion means the derivation could not resolve it, not that the
note is ordinary — so such notes are ignored rather than scored, and the rows
are deliberately not `mlign-*` so training can never pick them up.

Reported:
  detect P/R/F  is this played note part of an ornament at all
  attr acc      of the notes that ARE ornaments, share given the RIGHT anchor
  group exact   share of ornament figures whose notes ALL land on the anchor
                — the number that matters for "these 11 notes are that trill",
                since a figure with one stray note is not a usable group

Usage:
  .venv/bin/python eval/run_attribution.py --ckpt runs/v7orn/best.pt \
      --corpus data/corpus/orn-holdout.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import collate, featurize, parse_row  # noqa: E402
from mlign.model import NoteAligner, config_from_ckpt  # noqa: E402


def load_model(ckpt_path: str, device: str) -> NoteAligner:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    model = NoteAligner(config_from_ckpt(ckpt.get("config"), attribution=True))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    attr_missing = [k for k in missing if k.startswith("attr_")]
    if attr_missing:
        raise SystemExit(
            f"{ckpt_path} has no attribution head ({attr_missing}) — it was trained "
            "without --attribution, so there is nothing to evaluate here."
        )
    return model.to(device).eval()


def evaluate(model: NoteAligner, rows: list, device: str, batch: int, real_orn: bool = False) -> dict:
    det_tp = det_fp = det_fn = 0
    attr_hit = attr_tot = 0
    groups_exact = groups_tot = 0
    rows_seen = 0

    for start in range(0, len(rows), batch):
        chunk = [featurize(parse_row(r), real_orn=real_orn) for r in rows[start : start + batch]]
        b = collate(chunk, device)
        if "target_attr" not in b:
            continue
        with torch.no_grad():
            logits = model(b)["logits_attr"]
            # width BEFORE argmax collapses it: (B, m_max, n_max + 1), and the
            # "not an ornament" class is the last column, at n_max
            none_idx = logits.shape[-1] - 1
            pred = logits.argmax(-1).cpu()
        target = b["target_attr"].cpu()

        for i in range(len(chunk)):
            m = chunk[i]["m"]
            t = target[i, :m]
            p = pred[i, :m]
            sup = t != -100
            if not bool(sup.any()):
                continue
            rows_seen += 1
            # "is an ornament" = target/prediction names a score note rather
            # than the none column
            t_orn = sup & (t != none_idx) & (t >= 0)
            p_orn = sup & (p != none_idx)
            det_tp += int((t_orn & p_orn).sum())
            det_fp += int((~t_orn & p_orn & sup).sum())
            det_fn += int((t_orn & ~p_orn).sum())

            hit = (p == t) & t_orn
            attr_hit += int(hit.sum())
            attr_tot += int(t_orn.sum())

            # figure-level: group the true ornament notes by their anchor
            by_anchor: dict[int, list[int]] = collections.defaultdict(list)
            for j in torch.nonzero(t_orn).flatten().tolist():
                by_anchor[int(t[j])].append(j)
            for anchor, idxs in by_anchor.items():
                groups_tot += 1
                if all(int(p[j]) == anchor for j in idxs):
                    groups_exact += 1

    prec = det_tp / max(det_tp + det_fp, 1)
    rec = det_tp / max(det_tp + det_fn, 1)
    return {
        "rows": rows_seen,
        "detect_p": round(prec, 4),
        "detect_r": round(rec, 4),
        "detect_f": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "attr_acc": round(attr_hit / max(attr_tot, 1), 4),
        "attr_n": attr_tot,
        "group_exact": round(groups_exact / max(groups_tot, 1), 4),
        "group_n": groups_tot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True, help="held-out synthetic rows (jsonl)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    ap.add_argument("--real-orn", action="store_true",
                    help="also score realorn-* rows: ornament ground truth derived from "
                         "Nakamura match files on REAL recordings. Their labels are partial "
                         "(an unattributed insertion means unresolved, not 'not an ornament'), "
                         "so those notes are ignored rather than counted against the model")
    args = ap.parse_args()

    rows = [l for l in open(args.corpus, "rb") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    prefixes = ("mlign-", "realorn-") if args.real_orn else ("mlign-",)
    provenanced = [
        r for r in rows
        if str(parse_row(r).get("meta", {}).get("gen", "")).startswith(prefixes)
    ]
    if not provenanced:
        raise SystemExit(
            f"{args.corpus} has no rows with ornament ground truth (meta.gen = "
            f"{' or '.join(p + '*' for p in prefixes)}). Attribution cannot be scored on it."
        )
    if len(provenanced) < len(rows):
        print(
            f"note: {len(rows) - len(provenanced)} of {len(rows)} rows carry no ornament "
            "provenance and are excluded",
            file=sys.stderr,
        )

    res = evaluate(load_model(args.ckpt, args.device), provenanced, args.device, args.batch,
                   real_orn=args.real_orn)
    res |= {"ckpt": args.ckpt, "corpus": args.corpus,
            "tier": "real-recording" if args.real_orn else "synthetic-only"}
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
