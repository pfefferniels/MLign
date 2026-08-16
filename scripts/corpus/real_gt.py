"""Corpus source C: REAL nASAP ground-truth alignments as corpus rows.

Windows of real (score, performance, GT alignment) from nASAP performances
that are NOT in the MAESTRO-derived test split. Two uses:
  --role val   → a dedicated real-music validation set for checkpoint
                 SELECTION (the fix for synthetic-overfit blindness);
  --role train → real-GT fine-tuning rows (DESIGN §5 source C).
Disjoint piece folders between roles are enforced by --split-seed hashing so
val pieces never appear in train rows.

Row format = docs/corpus-format.md (score ticks at 720 ppq from quarters;
perf ms; align/ins/del from the match file). Windows: WIN score notes with
the perf notes whose GT match falls inside + unmatched perf notes whose onset
lies within the window's performed time span.

Usage:
  .venv/bin/python scripts/corpus/real_gt.py data/corpus/realgt-val.jsonl --role val
  .venv/bin/python scripts/corpus/real_gt.py data/corpus/realgt-train.jsonl --role train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from nasap import NasapIndex, load_match  # noqa: E402
from split import test_split  # noqa: E402

from mlign.tables import PerfTable, ScoreTable  # noqa: E402

WIN = 384
STRIDE = 256


def role_of(piece: str, val_frac: float = 0.25) -> str:
    h = int(hashlib.md5(("realgt:" + piece).encode()).hexdigest(), 16) % 1000
    return "val" if h < val_frac * 1000 else "train"


# Composer quotas for a HOLDOUT-REPRESENTATIVE selection set (v2 role "val2"):
# proportions of the nASAP robust test split (Beethoven 28 / Bach 19 / Liszt
# 17 / Chopin 13 / Schumann 6 / Rachmaninoff 1 of 84), applied to train-split
# pieces only. The v1 val set was Chopin/Schubert/Haydn-heavy and blind to
# long dense sonata movements with heavy deletions — v5real regressed exactly
# there while setting dev records. Ordering within composer prefers
# deletion-heavy performances (the failure class) so the proxy sees them.
VAL2_QUOTA = {"Beethoven": 0.33, "Bach": 0.23, "Liszt": 0.20, "Chopin": 0.15,
              "Schumann": 0.07, "Rachmaninoff": 0.02}


def select_val2(entries, target_perfs: int = 90):
    """Pick ~target_perfs performances from train-split entries by composer
    quota, preferring deletion-heavy performances within each composer.
    Pieces chosen here are the val2 role; everything else stays train."""
    from collections import defaultdict

    by_comp = defaultdict(list)
    for e in entries:
        c = e["piece"].split("/")[0]
        if c in VAL2_QUOTA:
            t = load_match(e["match"])
            n = len(t.matches) + len(t.deletions)
            del_rate = len(t.deletions) / n if n else 0.0
            by_comp[c].append((del_rate, e))
    chosen, chosen_pieces = [], set()
    for c, frac in VAL2_QUOTA.items():
        want = max(1, round(target_perfs * frac))
        # deletion-heavy first, but cap 2 perfs per piece for diversity
        per_piece = defaultdict(int)
        for del_rate, e in sorted(by_comp[c], key=lambda t: -t[0]):
            if len([x for x in chosen if x["piece"].split("/")[0] == c]) >= want:
                break
            if per_piece[e["piece"]] >= 2:
                continue
            per_piece[e["piece"]] += 1
            chosen.append(e)
            chosen_pieces.add(e["piece"])
    return chosen, chosen_pieces


def rows_for(entry, want_role: str) -> list[dict]:
    truth = load_match(entry["match"])
    score = ScoreTable.from_musicxml(entry["score"])
    perf = PerfTable.from_midi(entry["midi"])
    s_ids = [str(x) for x in score.notes["id"]]
    p_ids = [str(x) for x in perf.notes["id"]]
    s_index = {sid: i for i, sid in enumerate(s_ids)}
    p_index = {pid: j for j, pid in enumerate(p_ids)}
    s2p = {s: p for s, p in truth.matches if s in s_index and p in p_index}

    out = []
    n = len(s_ids)
    for start in range(0, max(1, n - WIN + 1), STRIDE):
        s_slice = list(range(start, min(n, start + WIN)))
        if len(s_slice) < 64:
            continue
        # perf notes matched to window score notes
        p_matched = {p_index[s2p[s_ids[i]]]: i for i in s_slice if s_ids[i] in s2p}
        if len(p_matched) < 32:
            continue
        p_sorted_matched = sorted(p_matched)
        t_lo = perf.notes["onset"][p_sorted_matched[0]]
        t_hi = perf.notes["onset"][p_sorted_matched[-1]]
        # unmatched perf notes inside the span → insertions
        p_slice = set(p_matched)
        for j in range(len(p_ids)):
            if p_ids[j] in truth.insertions and t_lo <= perf.notes["onset"][j] <= t_hi:
                p_slice.add(j)
        p_slice = sorted(p_slice)
        p_local = {j: k for k, j in enumerate(p_slice)}
        s_local = {i: k for k, i in enumerate(s_slice)}

        t0 = float(perf.notes["onset"][p_slice[0]])
        score_rows = [
            [round(float(score.notes["onset"][i]) * 720.0, 3), round(float(score.notes["duration"][i]) * 720.0, 3),
             int(score.notes["pitch"][i]), int(score.notes["voice"][i]) % 5]
            for i in s_slice
        ]
        perf_rows = [
            [round((float(perf.notes["onset"][j]) - t0) * 1000.0, 3), round(float(perf.notes["duration"][j]) * 1000.0, 3),
             int(perf.notes["pitch"][j]), int(perf.notes["velocity"][j])]
            for j in p_slice
        ]
        align = [[s_local[i], p_local[j]] for j, i in p_matched.items()]
        matched_s = {i for i in p_matched.values()}
        dele = [s_local[i] for i in s_slice if i not in matched_s]
        ins = [[p_local[j], 3] for j in p_slice if j not in p_matched]
        out.append({
            "meta": {"gen": "realgt-v0", "src": f"{entry['piece']}/{entry['performer']}", "w": start},
            "score": score_rows,
            "scoreIds": [s_ids[i] for i in s_slice],
            "perf": perf_rows,
            "align": align, "subs": [], "ins": ins, "orn": [], "del": dele,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--role", choices=["val", "train", "val2", "train2"], required=True,
                    help="val/train: hash split (v1). val2/train2: holdout-representative "
                         "composer-quota val2 (deletion-heavy preferred), train2 = the rest.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    folders, _ = test_split()
    idx = NasapIndex.build(ROOT / "data/benchmarks/asap-dataset", robust_only=True)
    train_split = [e for e in idx.entries if e["piece"] not in folders]
    if args.role in ("val2", "train2"):
        val2, val2_pieces = select_val2(train_split)
        if args.role == "val2":
            entries = val2
        else:
            entries = [e for e in train_split if e["piece"] not in val2_pieces]
        from collections import Counter
        print("val2 composers:", Counter(e["piece"].split("/")[0] for e in val2).most_common(), file=sys.stderr)
        print("val2 pieces:", len(val2_pieces), "| train2 perfs:", len(train_split) - sum(1 for e in train_split if e["piece"] in val2_pieces), file=sys.stderr)
    else:
        entries = [e for e in train_split if role_of(e["piece"]) == args.role]
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} performances for role={args.role}", file=sys.stderr)

    written = 0
    score_cache = {}
    with open(args.out, "w") as fh:
        for k, e in enumerate(entries):
            try:
                for row in rows_for(e, args.role):
                    fh.write(json.dumps(row) + "\n")
                    written += 1
            except Exception as err:
                print(f"skip {e['piece']}/{e['performer']}: {err}", file=sys.stderr)
            if (k + 1) % 25 == 0:
                print(f"...{k + 1}/{len(entries)} ({written} rows)", file=sys.stderr)
    print(f"wrote {written} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
