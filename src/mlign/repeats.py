"""Repeat-structure inference: which unfolding did the performance play?

Approach of Peter/Hu/Widmer, "How to Infer Repeat Structures in MIDI
Performances" (arXiv:2505.05055), reimplemented: local (Smith-Waterman-style)
alignment between the performance's onset pitch-set sequence and the FOLDED
score's onset pitch-set sequence, then enumeration of musically valid
structural versions, each scored by its accumulated gain along the diagonal
band it implies; the best version wins and the score is unfolded accordingly.

v0 scope: segment sequence enumeration is delegated to the caller (espressivo /
partitura both expose the segment graph); this module scores a candidate
unfolding against the performance and ranks candidates. That sidesteps
reimplementing the repeat grammar and keeps the module format-agnostic.

score_segments: list of segments, each a list of onset pitch-sets (the folded
score in reading order, split at repeat barlines / endings).
candidates: each a list of segment indices in playing order (e.g. [0,0,1] =
first segment twice, then the second once).
"""

from __future__ import annotations

import numpy as np


def onset_pitch_sets(onsets: np.ndarray, pitches: np.ndarray, eps: float = 1e-9) -> list[frozenset[int]]:
    """Group a sorted note table into per-onset pitch sets. For performances
    use a clustering eps of ~0.03-0.05 s."""
    sets: list[frozenset[int]] = []
    current: set[int] = set()
    last = None
    for onset, pitch in zip(onsets, pitches):
        if last is not None and onset - last > eps:
            sets.append(frozenset(current))
            current = set()
        current.add(int(pitch))
        last = onset
    if current:
        sets.append(frozenset(current))
    return sets


def _gain(a: frozenset[int], b: frozenset[int]) -> float:
    if not a or not b:
        return -1.0
    inter = len(a & b)
    union = len(a | b)
    return 2.0 * inter / union - 1.0  # +1 identical … −1 disjoint


def candidate_score(candidate_sets: list[frozenset[int]], perf_sets: list[frozenset[int]],
                    band: int = 64) -> float:
    """Banded accumulated-gain alignment (the 2505.05055 recurrence:
    ag(i,j) = max(ag(i-1,j), ag(i-1,j-1)) + m(i,j), gain clipped to [0,10]),
    normalized by performance length. The band keeps it O(n·band)."""
    n, m = len(candidate_sets), len(perf_sets)
    if n == 0 or m == 0:
        return -1e9
    slope = n / m
    prev = np.full(m + 1, -1e9)
    prev[0] = 0.0
    NEG = -1e9
    for i in range(1, n + 1):
        cur = np.full(m + 1, NEG)
        center = int(round(i / slope))
        lo = max(1, center - band)
        hi = min(m, center + band)
        row_set = candidate_sets[i - 1]
        for j in range(lo, hi + 1):
            best_prev = max(prev[j], prev[j - 1])
            if best_prev <= NEG / 2:
                continue
            g = _gain(row_set, perf_sets[j - 1])
            cur[j] = float(np.clip(best_prev + g, best_prev - 1.0, best_prev + 10.0))
        prev = cur
    tail = prev[max(1, m - band) :]
    return float(tail.max() / m)


def rank_candidates(score_segments: list[list[frozenset[int]]],
                    candidates: list[list[int]],
                    perf_sets: list[frozenset[int]],
                    segment_penalty: float = 0.02) -> list[tuple[float, list[int]]]:
    """Score every candidate unfolding; returns (score, candidate) sorted best
    first. segment_penalty discourages gratuitous extra segments (per 2505.05055)."""
    ranked = []
    for cand in candidates:
        sets: list[frozenset[int]] = []
        for seg in cand:
            sets.extend(score_segments[seg])
        s = candidate_score(sets, perf_sets) - segment_penalty * len(cand)
        ranked.append((s, cand))
    ranked.sort(key=lambda t: -t[0])
    return ranked
