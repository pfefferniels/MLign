"""Ornament attribution broken out by figure size AND notated ornament type.

`run_attribution.py` reports group-exact pooled. Stratified by figure size it is
not a smooth length decay: on Batik v12both scores 6.1 % on 3-note figures
against ~42 % and ~45 % on the sizes either side, and on ASAP 27.5 % on 4-note
figures against ~66 % and ~62 %. Note count correlates with ornament TYPE (a
mordent is three notes, a trill is many), so a whole type may be systematically
mis-anchored rather than a length being hard.

The corpus rows carry no type label, but they carry `scoreIds` and
`meta.source`, so each figure's anchor can be joined back to its match file and
the snote's ornament sign read off directly. Nothing is regenerated.

Usage:
  .venv/bin/python eval/attr_by_type.py --ckpt runs/v12both/best.pt \
      --corpus data/corpus/realorn-batik.jsonl --out notes/ornament/bytype-batik.json
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "corpus"))

from mlign.dataset import collate, featurize, parse_row  # noqa: E402
from mlign.infer import accumulate, decode  # noqa: E402
from real_orn_gt import parse_match  # noqa: E402
from run_attribution import load_model  # noqa: E402

# A trill sign and its continuation line mean the same figure; the rest stand
# on their own. Anything unexpected keeps its raw spelling rather than being
# folded into a bucket that would hide it.
_CANON = {"trill-mark": "trill", "wavy-line": "trill"}


def canonical_sign(signs: frozenset) -> str:
    named = sorted({_CANON.get(s, s) for s in signs})
    return "+".join(named) if named else "unsigned"


@functools.lru_cache(maxsize=None)
def signs_of(source: str) -> dict[str, frozenset]:
    """snote id → its ornament signs, for one match file."""
    snotes, _, _ = parse_match(ROOT / source)
    return {sn.id: sn.signs for sn in snotes}


@dataclass(frozen=True)
class Figure:
    piece: str
    anchor: str
    sign: str
    size: int
    hits: int
    vetoed: int  # notes the match head claimed for a written note
    unison: int  # notes struck at the anchor's own pitch
    unison_hits: int
    veto_hits: int

    @property
    def exact(self) -> bool:
        return self.hits == self.size


def figures(model, rows: list, device: str, batch: int):
    """Every scored ornament figure, one record each."""
    for start in range(0, len(rows), batch):
        parsed = [parse_row(r) for r in rows[start : start + batch]]
        chunk = [featurize(p, real_orn=True) for p in parsed]
        b = collate(chunk, device)
        if "target_attr" not in b:
            continue
        with torch.no_grad():
            out = model(b)
            logits = out["logits_attr"]
            none_idx = logits.shape[-1] - 1
            said_ins = (out["logits_p2s"].argmax(-1) == none_idx).cpu()
        pred = logits.argmax(-1).cpu()
        target = b["target_attr"].cpu()

        for i, row in enumerate(parsed):
            m = chunk[i]["m"]
            t, p, ins_i = target[i, :m], pred[i, :m], said_ins[i, :m]
            t_orn = (t != -100) & (t != none_idx) & (t >= 0)
            if not bool(t_orn.any()):
                continue
            ids, score, perf = row["scoreIds"], row["score"], row["perf"]
            sign_map = signs_of(row["meta"]["source"])
            by_anchor: dict[int, list[int]] = collections.defaultdict(list)
            for j in torch.nonzero(t_orn).flatten().tolist():
                by_anchor[int(t[j])].append(j)
            for anchor, idxs in by_anchor.items():
                if anchor >= len(ids):
                    continue  # a padded column, never a real anchor
                aid = ids[anchor]
                a_pitch = score[anchor][2]
                hit = [int(p[j]) == anchor for j in idxs]
                veto = [not bool(ins_i[j]) for j in idxs]
                uni = [perf[j][2] == a_pitch for j in idxs]
                yield Figure(
                    piece=row["meta"].get("piece", ""),
                    anchor=f"{row['meta']['source']}#{aid}",
                    sign=canonical_sign(sign_map.get(aid, frozenset())),
                    size=len(idxs),
                    hits=sum(hit),
                    vetoed=sum(veto),
                    unison=sum(uni),
                    unison_hits=sum(h for h, u in zip(hit, uni) if u),
                    veto_hits=sum(h for h, v in zip(hit, veto) if v),
                )


def decoded_figures(model, rows: list, device: str):
    """The same records, but attributed as the PIPELINE does it.

    `figures` reads the head's argmax, which the match head dominates. This
    reads what a user gets: the decode names the insertions, and the head is
    asked only about those.
    """
    for row in rows:
        gt = {int(p): int(a) for p, a, *_ in row.get("orn", ()) if int(a) >= 0}
        if not gt:
            continue
        ev = accumulate(model, row, device)
        tri = decode(row, ev.sim, ev.null_s, ev.null_p, ornaments=ev.ornaments)
        pred = {t["perf_idx"]: (t.get("ornament") or {}).get("anchor_score_idx", -1)
                for t in tri if t["label"] == "insertion"}
        ids, score, perf = row["scoreIds"], row["score"], row["perf"]
        sign_map = signs_of(row["meta"]["source"])
        by_anchor: dict[int, list[int]] = collections.defaultdict(list)
        for pi, anchor in gt.items():
            by_anchor[anchor].append(pi)
        for anchor, idxs in by_anchor.items():
            if anchor >= len(ids):
                continue
            aid = ids[anchor]
            a_pitch = score[anchor][2]
            hit = [pred.get(j, -1) == anchor for j in idxs]
            veto = [j not in pred for j in idxs]   # the decode called it a match
            uni = [perf[j][2] == a_pitch for j in idxs]
            yield Figure(
                piece=row["meta"].get("piece", ""),
                anchor=f"{row['meta']['source']}#{aid}",
                sign=canonical_sign(sign_map.get(aid, frozenset())),
                size=len(idxs), hits=sum(hit), vetoed=sum(veto), unison=sum(uni),
                unison_hits=sum(h for h, u in zip(hit, uni) if u),
                veto_hits=sum(h for h, v in zip(hit, veto) if v),
            )


def size_bucket(size: int) -> str:
    return str(size) if size <= 8 else "9+"


def crosstab(figs: list[Figure], key) -> dict:
    tot: collections.Counter = collections.Counter()
    exact: collections.Counter = collections.Counter()
    notes: collections.Counter = collections.Counter()
    hits: collections.Counter = collections.Counter()
    for f in figs:
        k = key(f)
        tot[k] += 1
        exact[k] += int(f.exact)
        notes[k] += f.size
        hits[k] += f.hits
    return {
        str(k): {
            "groups": tot[k],
            "group_exact": round(exact[k] / tot[k], 4),
            "note_acc": round(hits[k] / max(notes[k], 1), 4),
            "notes": notes[k],
        }
        for k in tot
    }


def render(title: str, table: dict, order=None) -> str:
    keys = order(table) if order else sorted(table)
    width = max((len(k) for k in keys), default=1)
    lines = [f"  {title}", f"    {'bucket'.ljust(width)}  groups  grp-exact  note-acc  notes"]
    lines += [
        f"    {k.ljust(width)}  {t['groups']:6d}  {t['group_exact']:9.4f}  "
        f"{t['note_acc']:8.4f}  {t['notes']:5d}"
        for k in keys
        if (t := table[k])
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decoded", action="store_true",
                    help="attribute as the pipeline does, not by the head's argmax")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [l for l in open(args.corpus, "rb") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    rows = [r for r in rows
            if str(parse_row(r).get("meta", {}).get("gen", "")).startswith("realorn-")]
    if not rows:
        raise SystemExit(f"{args.corpus} carries no realorn-* rows to score.")

    model = load_model(args.ckpt, args.device)
    figs = list(decoded_figures(model, [parse_row(r) for r in rows], args.device)
                if args.decoded else figures(model, rows, args.device, args.batch))
    clean = [f for f in figs if f.vetoed == 0]
    by_size = crosstab(figs, lambda f: size_bucket(f.size))
    by_sign = crosstab(figs, lambda f: f.sign)
    by_both = crosstab(figs, lambda f: f"{f.sign}/{size_bucket(f.size)}")
    by_size_clean = crosstab(clean, lambda f: size_bucket(f.size))

    def size_order(t):
        return sorted(t, key=lambda k: (k == "9+", int(k) if k != "9+" else 0))

    notes = sum(f.size for f in figs)
    uni, uni_hit = sum(f.unison for f in figs), sum(f.unison_hits for f in figs)
    vet, vet_hit = sum(f.vetoed for f in figs), sum(f.veto_hits for f in figs)
    print(f"{args.ckpt}  {args.corpus}  {len(figs)} figures, {notes} notes")
    print(render("by figure size", by_size, size_order))
    print(render("by figure size, figures with NO vetoed note", by_size_clean, size_order))
    print(render("by ornament type", by_sign))
    print(render("by type x size (>=5 groups)",
                 {k: v for k, v in by_both.items() if v["groups"] >= 5}))
    print(f"  unison notes (anchor's own pitch) {uni}/{notes} = {uni / max(notes, 1):.4f}, "
          f"acc {uni_hit / max(uni, 1):.4f} vs {(sum(f.hits for f in figs) - uni_hit) / max(notes - uni, 1):.4f} "
          f"on the rest")
    print(f"  vetoed notes {vet}/{notes} = {vet / max(notes, 1):.4f}, acc {vet_hit / max(vet, 1):.4f}; "
          f"of the vetoed, {sum(min(f.vetoed, f.unison) for f in figs)} are also unison")
    print(f"  figures with >=1 vetoed note: {len(figs) - len(clean)}/{len(figs)}")

    res = {"ckpt": args.ckpt, "corpus": args.corpus, "figures": len(figs),
           "by_size": by_size, "by_size_no_veto": by_size_clean,
           "by_sign": by_sign, "by_sign_size": by_both,
           "detail": [asdict(f) | {"exact": f.exact} for f in figs]}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
