"""Check a corpus file against notes/corpus-format.md, and describe it.

The format doc says its invariants are "checked by the writer". They are — by
each writer, separately, in the writer's own terms. This checks them from
outside, on the file as it lies on disk, which is the only place a rebasing bug
in a windowing pass or a mis-joined anchor actually shows up.

    .venv/bin/python scripts/corpus/validate.py data/corpus/orn-a.jsonl [--limit N]

Exits non-zero if any row violates an invariant. The statistics it prints are
the ones the ornament work is calibrated against, so they are part of the point
rather than decoration.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys

INS_KIND = {0: "slip", 1: "restart-first-pass", 2: "ornament", 3: "addition", 4: "other"}


def check_row(row: dict, i: int) -> list[str]:
    """Every invariant the format states, plus the ones it implies."""
    bad = []
    n, m = len(row["score"]), len(row["perf"])
    align = row["align"]
    ins = row["ins"]
    dele = row["del"]

    s_seen = [s for s, _ in align]
    p_seen = [p for _, p in align] + [p for p, _ in ins]
    if len(set(s_seen)) != len(s_seen):
        bad.append("a score note is matched twice")
    if len(set(p_seen)) != len(p_seen):
        bad.append("a performed note is matched and/or inserted twice")
    if sorted(set(s_seen) | set(dele)) != list(range(n)):
        bad.append(f"score coverage: {len(set(s_seen))} matched + {len(set(dele))} deleted != {n}")
    if sorted(set(p_seen)) != list(range(m)):
        bad.append(f"perf coverage: {len(set(p_seen))} of {m} played notes accounted for")

    if any(a < 0 or a >= n for a in s_seen) or any(a < 0 or a >= n for a in dele):
        bad.append("score index out of range")
    if any(p < 0 or p >= m for p in p_seen):
        bad.append("perf index out of range")

    onsets = [r[0] for r in row["perf"]]
    if onsets != sorted(onsets):
        bad.append("perf not sorted by onset")
    keys = [(r[0], r[2]) for r in row["score"]]
    if keys != sorted(keys):
        bad.append("score not sorted by (onset, pitch)")

    if any(r[1] <= 0 for r in row["perf"]):
        bad.append("a played note has non-positive duration")

    ins_at = {p: k for p, k in ins}
    for rec in row.get("orn", ()):
        p, anchor = rec[0], rec[1]
        if p not in ins_at:
            bad.append(f"orn names played note {p}, which is not an insertion")
        elif ins_at[p] not in (2, 3):
            bad.append(f"orn names played note {p}, an insertion of kind {ins_at[p]}")
        if anchor >= n:
            bad.append(f"orn anchor {anchor} out of range (n={n})")
    return [f"row {i}: {b}" for b in bad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-errors", type=int, default=20)
    args = ap.parse_args()

    rows = n_score = n_perf = 0
    errors: list[str] = []
    kinds = collections.Counter()
    groups = collections.Counter()
    sizes = collections.Counter()
    gens = collections.Counter()
    orn_free = 0
    with open(args.corpus) as fh:
        for i, line in enumerate(itertools.islice(fh, args.limit or None)):
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            n_score += len(row["score"])
            n_perf += len(row["perf"])
            gens[row.get("meta", {}).get("gen", "?")] += 1
            if len(errors) < args.max_errors:
                errors.extend(check_row(row, i))

            at = {p: k for p, k in row["ins"]}
            per_anchor: dict[tuple[int, int], int] = collections.defaultdict(int)
            for rec in row.get("orn", ()):
                k = at.get(rec[0], 4)
                kinds[k] += 1
                per_anchor[(k, rec[1])] += 1
            for (k, _), c in per_anchor.items():
                groups[k] += 1
                if k == 2:
                    sizes[c] += 1
            if not any(k == 2 for k, _ in per_anchor):
                orn_free += 1

    print(f"{args.corpus}: {rows} rows, {n_score} score notes, {n_perf} played notes")
    print(f"  generators: {dict(gens)}")
    for k in sorted(kinds):
        print(f"  {INS_KIND.get(k, k):20s}: {groups[k]:6d} groups, {kinds[k]:7d} notes"
              f"  -> {1000 * groups[k] / max(n_score, 1):6.2f} groups/1000 score notes,"
              f" {100 * kinds[k] / max(n_perf, 1):5.2f}% of played notes")
    attributable = sum(kinds.values())
    print(f"  attributable insertions: {100 * attributable / max(n_perf, 1):.2f}% of played notes")
    print(f"  rows with no ornament: {orn_free}/{rows} = {100 * orn_free / max(rows, 1):.0f}%")
    if sizes:
        top = sorted(sizes.items())
        print(f"  ornament figure sizes: {top[:10]}{' …' if len(top) > 10 else ''}"
              f"  (max {max(sizes)})")

    if errors:
        print(f"\nFAILED — {len(errors)} violation(s):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    print("\nOK — every row satisfies notes/corpus-format.md")


if __name__ == "__main__":
    main()
