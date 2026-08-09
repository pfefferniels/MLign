"""Classical baseline aligner: onset-cluster DTW + within-cluster pitch matching.

Roughly the shape of parangonar's AutomaticNoteMatcher: both note tables are
grouped into onset clusters (score: exact onsets; performance: 50 ms greedy
clusters), a DTW over the cluster sequences with pitch-set Jaccard cost finds
the temporal correspondence, and within each correspondence equal pitches are
paired. Unpaired notes become insertions/deletions. No learning; exists to
(a) exercise the evaluation loop end-to-end and (b) provide the floor.
"""

from __future__ import annotations

import numpy as np

from .tables import PerfTable, ScoreTable


def _score_clusters(table: ScoreTable) -> list[np.ndarray]:
    onsets = table.notes["onset"]
    boundaries = np.flatnonzero(np.diff(onsets) > 1e-9) + 1
    return np.split(np.arange(len(onsets)), boundaries)


def _perf_clusters(table: PerfTable, eps: float = 0.05) -> list[np.ndarray]:
    onsets = table.notes["onset"]
    boundaries = np.flatnonzero(np.diff(onsets) > eps) + 1
    return np.split(np.arange(len(onsets)), boundaries)


def _pitch_sets(table, clusters) -> list[frozenset[int]]:
    return [frozenset(int(p) for p in table.notes["pitch"][c]) for c in clusters]


def align_baseline(score: ScoreTable, perf: PerfTable) -> list[dict]:
    """Returns alignment triples over table ids (same shape as eval GT)."""
    sc = _score_clusters(score)
    pc = _perf_clusters(perf)
    s_sets = _pitch_sets(score, sc)
    p_sets = _pitch_sets(perf, pc)

    n, m = len(sc), len(pc)
    # Cost matrix: Jaccard distance between cluster pitch sets.
    cost = np.empty((n, m), dtype=np.float32)
    for i, ss in enumerate(s_sets):
        for j, ps in enumerate(p_sets):
            inter = len(ss & ps)
            union = len(ss | ps)
            cost[i, j] = 1.0 - (inter / union if union else 0.0)

    # DTW with diagonal/vertical/horizontal steps, gap penalty for skips.
    GAP = 0.75
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    acc[0, 1:] = np.cumsum(np.full(m, GAP))
    acc[1:, 0] = np.cumsum(np.full(n, GAP))
    for i in range(1, n + 1):
        row_cost = cost[i - 1]
        prev = acc[i - 1]
        cur = acc[i]
        for j in range(1, m + 1):
            cur[j] = min(
                prev[j - 1] + row_cost[j - 1],
                prev[j] + GAP,
                cur[j - 1] + GAP,
            )

    # Backtrack.
    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        d = acc[i - 1, j - 1] + cost[i - 1, j - 1]
        v = acc[i - 1, j] + GAP
        h = acc[i, j - 1] + GAP
        best = min(d, v, h)
        if best == d:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif best == v:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    # Within matched clusters: pair equal pitches in order.
    triples: list[dict] = []
    matched_s: set[int] = set()
    matched_p: set[int] = set()
    perf_used_by_cluster: dict[int, set[int]] = {}
    for ci, cj in pairs:
        s_idx = sc[ci]
        p_idx = pc[cj]
        used = perf_used_by_cluster.setdefault(cj, set())
        by_pitch: dict[int, list[int]] = {}
        for k in p_idx:
            if k not in matched_p:
                by_pitch.setdefault(int(perf.notes["pitch"][k]), []).append(int(k))
        for k in s_idx:
            if k in matched_s:
                continue
            cand = by_pitch.get(int(score.notes["pitch"][k]))
            if cand:
                pk = cand.pop(0)
                triples.append(
                    {
                        "label": "match",
                        "score_id": str(score.notes["id"][k]),
                        "perf_id": str(perf.notes["id"][pk]),
                    }
                )
                matched_s.add(int(k))
                matched_p.add(pk)
                used.add(pk)

    for k in range(len(score)):
        if k not in matched_s:
            triples.append({"label": "deletion", "score_id": str(score.notes["id"][k])})
    for k in range(len(perf)):
        if k not in matched_p:
            triples.append({"label": "insertion", "perf_id": str(perf.notes["id"][k])})
    return triples
