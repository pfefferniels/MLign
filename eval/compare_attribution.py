"""Paired bootstrap between two checkpoints' ornament attribution.

A difference in group-exact between two runs is a difference between two
proportions over the same figures, and every cell in the ornament 2x2 is n=1.
Quoting the point estimate alone has repeatedly made a marginal result look
settled: v12both beats v3 by .0459 on clean Batik, which sounds decisive until
the interval turns out to be [+.0108, +.0811] with 106 figures gained and 72
lost. Pairing over figures removes between-figure variance, which is most of it.

The gained/lost counts matter as much as the interval. Group-exact is monotone
under attributing more notes, so it cannot by itself tell a better predictor
from a louder one — but a merely louder one loses figures in proportion to what
it gains, and a real improvement does not.

Batik is the corpus to read. 209 of realorn-asap's 225 rows are performances the
match head trained on, so `--corpus` defaults to batik and ASAP is reported
clean-only when asked for.

  .venv/bin/python eval/compare_attribution.py --base runs/v12both/best.pt \
      --cand runs/v14evid/best.pt
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
from mlign.infer import accumulate, decode  # noqa: E402
from run_attribution_decoded import NOT_ATTRIBUTED, is_clean, truth  # noqa: E402
from run_attribution import load_model  # noqa: E402


def figure_outcomes(ckpt: str, rows: list[dict], device: str) -> np.ndarray:
    """Whether each figure came out whole, in a fixed order across checkpoints."""
    model = load_model(ckpt, device)
    out = []
    for row in rows:
        gt = truth(row)
        if not gt:
            continue
        ev = accumulate(model, row, device)
        triples = decode(row, ev.sim, ev.null_s, ev.null_p, ornaments=ev.ornaments)
        pred = {t["perf_idx"]: (t.get("ornament") or {}).get("anchor_score_idx", NOT_ATTRIBUTED)
                for t in triples if t["label"] == "insertion"}
        figures: dict[int, list[int]] = collections.defaultdict(list)
        for pi, anchor in gt.items():
            figures[anchor].append(pi)
        out += [all(pred.get(pi, NOT_ATTRIBUTED) == anchor for pi in pis)
                for anchor, pis in sorted(figures.items())]
    return np.array(out, dtype=bool)


def paired_bootstrap(base: np.ndarray, cand: np.ndarray, resamples: int, seed: int) -> dict:
    diff = cand.astype(float) - base.astype(float)
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return {
        "figures": len(diff),
        "base": round(float(base.mean()), 4),
        "cand": round(float(cand.mean()), 4),
        "diff": round(float(diff.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p": round(max(float(p), 1 / resamples), 5),
        "gained": int((cand & ~base).sum()),
        "lost": int((base & ~cand).sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the checkpoint to beat")
    ap.add_argument("--cand", required=True)
    ap.add_argument("--corpus", default="batik", choices=["batik", "asap"])
    ap.add_argument("--clean-only", action="store_true",
                    help="drop performances any run trained or selected on")
    ap.add_argument("--resamples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    path = ROOT / f"data/corpus/realorn-{args.corpus}.jsonl"
    rows = [parse_row(l) for l in open(path, "rb") if l.strip()]
    rows = [r for r in rows if str(r.get("meta", {}).get("gen", "")).startswith("realorn-")]
    if args.clean_only:
        rows = [r for r in rows if is_clean(r)]
    if args.corpus == "asap" and not args.clean_only:
        print("note: 92.9 % of realorn-asap is performances the match head trained on; "
              "pass --clean-only for the 36-figure holdout", file=sys.stderr)

    res = paired_bootstrap(figure_outcomes(args.base, rows, args.device),
                           figure_outcomes(args.cand, rows, args.device),
                           args.resamples, args.seed)
    res |= {"base_ckpt": args.base, "cand_ckpt": args.cand,
            "corpus": args.corpus, "clean_only": args.clean_only}
    verdict = ("candidate better" if res["ci95"][0] > 0 else
               "candidate worse" if res["ci95"][1] < 0 else "indistinguishable")
    print(f"{args.base} -> {args.cand}   realorn-{args.corpus}"
          f"{' (clean only)' if args.clean_only else ''}")
    print(f"  group-exact {res['base']:.4f} -> {res['cand']:.4f}   diff {res['diff']:+.4f}"
          f"   95% CI [{res['ci95'][0]:+.4f}, {res['ci95'][1]:+.4f}]   p={res['p']:.5f}")
    print(f"  figures gained {res['gained']}, lost {res['lost']}, of {res['figures']}"
          f"   -> {verdict}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
