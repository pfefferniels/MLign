"""Inference: (ScoreTable, PerfTable) + trained model → alignment triples.

v0 pipeline (DESIGN §4, two-phase-lite):
  1. featurize both tables like a corpus row;
  2. forward the model — whole piece if it fits, else windows along a coarse
     onset-cluster-DTW band, logits accumulated with overlap counts;
  3. decode: dual-softmax confidence → mutual high-confidence, pitch-equal
     anchor pairs → monotone filter (longest increasing subsequence by onset)
     → interpolated score→perf time map → per-pitch assignment (rarest pitch
     first, map-guided, tolerance-gated) → null logits decide leftovers.
"""

from __future__ import annotations

import numpy as np
import torch

from .dataset import collate, featurize
from .tables import PerfTable, ScoreTable

MAX_SINGLE_TOKENS = 2000
WIN_SCORE = 384  # score notes per window
MARGIN_SEC = 3.0


def tables_to_row(score: ScoreTable, perf: PerfTable) -> dict:
    return {
        "score": [
            [float(n["onset"]) * 720.0, float(n["duration"]) * 720.0, int(n["pitch"]), int(n["voice"]) % 5]
            for n in score.notes
        ],
        "perf": [
            [float(n["onset"]) * 1000.0, float(n["duration"]) * 1000.0, int(n["pitch"]), int(n["velocity"])]
            for n in perf.notes
        ],
        "align": [],
        "subs": [],
        "ins": [],
        "del": [],
    }


@torch.no_grad()
def accumulate_logits(model, row: dict, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (sim (n,m), null_s (n,), null_p (m,)) accumulated over windows."""
    n = len(row["score"])
    m = len(row["perf"])

    if 2 + n + m <= MAX_SINGLE_TOKENS:
        pairs = [(0, n, 0, m)]
    else:
        pairs = coarse_windows(row, n, m)

    sim = np.full((n, m), -1e9, dtype=np.float32)
    cnt = np.zeros((n, m), dtype=np.float32)
    null_s = np.zeros(n, dtype=np.float32)
    null_s_cnt = np.zeros(n, dtype=np.float32)
    null_p = np.zeros(m, dtype=np.float32)
    null_p_cnt = np.zeros(m, dtype=np.float32)

    for s0, s1, p0, p1 in pairs:
        sub = {
            "score": row["score"][s0:s1],
            "perf": row["perf"][p0:p1],
            "align": [], "subs": [], "ins": [], "del": [],
        }
        batch = collate([featurize(sub)], device)
        out = model(batch)
        ls2p = out["logits_s2p"][0].float().cpu().numpy()  # (ns, mp+1)
        lp2s = out["logits_p2s"][0].float().cpu().numpy()

        ns, mp = s1 - s0, p1 - p0
        block = ls2p[:ns, :mp] + lp2s[:mp, :ns].T
        region = sim[s0:s1, p0:p1]
        first = cnt[s0:s1, p0:p1] == 0
        region[first] = 0.0
        region += block
        sim[s0:s1, p0:p1] = region
        cnt[s0:s1, p0:p1] += 1.0

        null_s[s0:s1] += ls2p[:ns, mp]
        null_s_cnt[s0:s1] += 1.0
        null_p[p0:p1] += lp2s[:mp, ns]
        null_p_cnt[p0:p1] += 1.0

    cnt[cnt == 0] = 1.0
    sim = sim / cnt
    null_s = null_s / np.maximum(null_s_cnt, 1.0)
    null_p = null_p / np.maximum(null_p_cnt, 1.0)
    # notes never covered by any window: strongly "unmatched"
    null_s[null_s_cnt == 0] = 1e9
    null_p[null_p_cnt == 0] = 1e9
    return sim, null_s, null_p


def coarse_windows(row: dict, n: int, m: int) -> list[tuple[int, int, int, int]]:
    """Score windows + perf ranges via a cheap monotone map from cluster DTW."""
    from .baseline import _perf_clusters, _pitch_sets, _score_clusters

    score = ScoreTable(np.array(
        [(r[0] / 720.0, r[1] / 720.0, int(r[2]), int(r[3]), str(i)) for i, r in enumerate(row["score"])],
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
    ))
    perf = PerfTable(np.array(
        [(r[0] / 1000.0, r[1] / 1000.0, int(r[2]), int(r[3]), str(i)) for i, r in enumerate(row["perf"])],
        dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U32")],
    ))
    # anchor map from the DTW baseline's cluster path
    from .baseline import align_baseline

    triples = align_baseline(score, perf)
    s_on = {int(t["score_id"]): score.notes["onset"][int(t["score_id"])] for t in triples if t["label"] == "match"}
    pairs = sorted(
        (int(t["score_id"]), int(t["perf_id"])) for t in triples if t["label"] == "match"
    )
    if not pairs:
        return [(0, n, 0, m)]
    s_idx = np.array([p[0] for p in pairs])
    p_idx = np.array([p[1] for p in pairs])
    p_time = perf.notes["onset"][p_idx]

    out = []
    stride = WIN_SCORE // 2
    for s0 in range(0, n, stride):
        s1 = min(n, s0 + WIN_SCORE)
        sel = (s_idx >= s0) & (s_idx < s1)
        if sel.sum() < 2:
            t_lo, t_hi = float(perf.notes["onset"][0]), float(perf.notes["onset"][-1])
        else:
            t_lo = float(p_time[sel].min()) - MARGIN_SEC
            t_hi = float(p_time[sel].max()) + MARGIN_SEC
        p_on = perf.notes["onset"]
        p0 = int(np.searchsorted(p_on, t_lo, "left"))
        p1 = int(np.searchsorted(p_on, t_hi, "right"))
        p0, p1 = max(0, p0), min(m, max(p1, p0 + 1))
        out.append((s0, s1, p0, p1))
        if s1 >= n:
            break
    return out


def decode(row: dict, sim: np.ndarray, null_s: np.ndarray, null_p: np.ndarray,
           anchor_conf: float = 0.35, tol_sec: float = 1.0) -> list[dict]:
    n, m = sim.shape
    s_pitch = np.array([r[2] for r in row["score"]], dtype=int)
    p_pitch = np.array([r[2] for r in row["perf"]], dtype=int)
    s_onset = np.array([r[0] for r in row["score"]]) / 720.0
    p_onset = np.array([r[0] for r in row["perf"]]) / 1000.0

    # dual softmax over sim with null columns appended
    a = np.concatenate([sim, null_s[:, None]], axis=1)
    b = np.concatenate([sim.T, null_p[:, None]], axis=1)
    sm_s = _softmax(a, axis=1)[:, :m]
    sm_p = _softmax(b, axis=1)[:, :n]
    conf = sm_s * sm_p.T  # (n, m)

    # --- phase 1: monotone time map.
    # Primary evidence: mutual high-confidence pitch-equal anchors. Fallback /
    # reinforcement: cluster-level DTW over blended cost (model confidence +
    # pitch-set Jaccard) — TheGlueNote's ablation shows the DTW path over the
    # confidence matrix carries weak models; anchors alone collapse there.
    best_p = conf.argmax(axis=1)
    best_s = conf.argmax(axis=0)
    anchors = []
    for i in range(n):
        j = best_p[i]
        if best_s[j] == i and conf[i, j] >= anchor_conf and s_pitch[i] == p_pitch[j]:
            anchors.append((i, j))
    anchors = _monotone_subset(anchors, s_onset, p_onset)

    dtw_ax, dtw_ay = _cluster_dtw_map(s_onset, s_pitch, p_onset, p_pitch, conf)

    # Union, not choice: the DTW path densifies the gaps between sparse
    # anchors (33 anchors over 450 notes leave multi-second interpolation
    # holes); anchors sharpen the path where the model is confident.
    ax = np.concatenate([dtw_ax, [s_onset[i] for i, _ in anchors]])
    ay = np.concatenate([dtw_ay, [p_onset[j] for _, j in anchors]])
    order = np.argsort(ax, kind="stable")
    ax, ay = ax[order], ay[order]
    if len(ax) >= 2:
        ax, keep = np.unique(ax, return_index=True)
        ay = np.asarray(ay)[keep]
        def s2p_time(x):
            return np.interp(x, ax, ay)
    else:
        def s2p_time(x):
            return np.zeros_like(np.asarray(x, dtype=float))

    # --- phase 2: per-pitch assignment, rarest pitch first; then rebuild the
    # map from the matches and re-assign. Round 1's map comes from cluster DTW
    # (median ~25ms but locally skewed under heavy rubato → off-by-one shifts
    # inside repeated-pitch runs); round 2's map interpolates the dense round-1
    # matches, which pins those runs.
    matched_s = np.full(n, -1, dtype=int)
    matched_p = np.full(m, -1, dtype=int)
    for _round in range(2):
        matched_s.fill(-1)
        matched_p.fill(-1)
        pitches, counts = np.unique(s_pitch, return_counts=True)
        for pitch in pitches[np.argsort(counts)]:
            si = np.flatnonzero((s_pitch == pitch) & (matched_s == -1))
            pj = np.flatnonzero((p_pitch == pitch) & (matched_p == -1))
            if len(si) == 0 or len(pj) == 0:
                continue
            exp = s2p_time(s_onset[si])
            # small DP: monotone assignment minimizing |perf_time - expected|
            pairs = _assign_monotone(exp, p_onset[pj], tol_sec, conf[np.ix_(si, pj)])
            for a_i, b_j in pairs:
                matched_s[si[a_i]] = pj[b_j]
                matched_p[pj[b_j]] = si[a_i]
        if _round == 0:
            got = np.flatnonzero(matched_s >= 0)
            if len(got) < 8:
                break
            pairs2 = _monotone_subset(
                [(int(i), int(matched_s[i])) for i in got], s_onset, p_onset
            )
            if len(pairs2) < 8:
                break
            rx = np.array([s_onset[i] for i, _ in pairs2])
            ry = np.array([p_onset[j] for _, j in pairs2])
            rx, keep2 = np.unique(rx, return_index=True)
            ry = ry[keep2]
            def s2p_time(x, _rx=rx, _ry=ry):  # noqa: F811
                return np.interp(x, _rx, _ry)

    # Residual rescue: an unmatched score note + unmatched perf note of the
    # SAME pitch within a tight map window is almost always a pair the
    # per-pitch DP dropped on a local order violation — labelling them
    # insertion+deletion would double-count one mistake. Greedy nearest-first.
    RESCUE_SEC = 0.35
    res_s = [i for i in range(n) if matched_s[i] < 0]
    res_p = [j for j in range(m) if matched_p[j] < 0]
    by_pitch_p: dict[int, list[int]] = {}
    for j in res_p:
        by_pitch_p.setdefault(int(p_pitch[j]), []).append(j)
    cands = []
    for i in res_s:
        exp = float(s2p_time(np.array([s_onset[i]]))[0])
        for j in by_pitch_p.get(int(s_pitch[i]), ()):
            d = abs(p_onset[j] - exp)
            if d <= RESCUE_SEC:
                cands.append((d, i, j))
    cands.sort()
    for d, i, j in cands:
        if matched_s[i] < 0 and matched_p[j] < 0:
            matched_s[i] = j
            matched_p[j] = i

    triples: list[dict] = []
    for i in range(n):
        if matched_s[i] >= 0:
            j = int(matched_s[i])
            triples.append({
                "label": "match", "score_idx": i, "perf_idx": j,
                "confidence": float(conf[i, j]),
            })
        else:
            # confidence that the note is truly deleted = null share of the
            # score note's softmax mass
            null_share = float(_softmax(np.concatenate([sim[i], [null_s[i]]]), axis=0)[-1])
            triples.append({"label": "deletion", "score_idx": i, "confidence": null_share})
    for j in range(m):
        if matched_p[j] < 0:
            null_share = float(_softmax(np.concatenate([sim[:, j], [null_p[j]]]), axis=0)[-1])
            triples.append({"label": "insertion", "perf_idx": j, "confidence": null_share})
    return triples


def _cluster_dtw_map(s_onset, s_pitch, p_onset, p_pitch, conf):
    """Onset-cluster DTW with cost = 0.5·(1−Jaccard) + 0.5·(1−mean conf).
    Returns (score onsets, perf onsets) along the path — a dense monotone map
    that works even when the model is weak (pitch evidence) and sharpens as
    the model improves (confidence evidence)."""
    s_bounds = np.flatnonzero(np.diff(s_onset) > 1e-9) + 1
    s_clusters = np.split(np.arange(len(s_onset)), s_bounds)
    p_bounds = np.flatnonzero(np.diff(p_onset) > 0.05) + 1
    p_clusters = np.split(np.arange(len(p_onset)), p_bounds)

    ns, mp = len(s_clusters), len(p_clusters)
    s_sets = [frozenset(int(x) for x in s_pitch[c]) for c in s_clusters]
    p_sets = [frozenset(int(x) for x in p_pitch[c]) for c in p_clusters]

    cost = np.empty((ns, mp), dtype=np.float32)
    for i, sc in enumerate(s_clusters):
        ss = s_sets[i]
        conf_rows = conf[sc]
        for j, pc in enumerate(p_clusters):
            ps = p_sets[j]
            union = len(ss | ps)
            jac = len(ss & ps) / union if union else 0.0
            c_conf = float(conf_rows[:, pc].mean())
            cost[i, j] = 0.5 * (1.0 - jac) + 0.5 * (1.0 - min(1.0, c_conf * 20.0))

    GAP = 0.6
    acc = np.full((ns + 1, mp + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    acc[0, 1:] = np.cumsum(np.full(mp, GAP))
    acc[1:, 0] = np.cumsum(np.full(ns, GAP))
    for i in range(1, ns + 1):
        row = cost[i - 1]
        prev = acc[i - 1]
        cur = acc[i]
        for j in range(1, mp + 1):
            cur[j] = min(prev[j - 1] + row[j - 1], prev[j] + GAP, cur[j - 1] + GAP)

    ax, ay = [], []
    i, j = ns, mp
    while i > 0 and j > 0:
        d = acc[i - 1, j - 1] + cost[i - 1, j - 1]
        v = acc[i - 1, j] + GAP
        h = acc[i, j - 1] + GAP
        best = min(d, v, h)
        if best == d:
            ax.append(float(s_onset[s_clusters[i - 1][0]]))
            ay.append(float(p_onset[p_clusters[j - 1][0]]))
            i, j = i - 1, j - 1
        elif best == v:
            i -= 1
        else:
            j -= 1
    return np.array(ax[::-1]), np.array(ay[::-1])


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _monotone_subset(anchors, s_onset, p_onset):
    """Longest chain increasing in both coordinates (patience-style, O(k log k))."""
    if not anchors:
        return anchors
    anchors = sorted(anchors, key=lambda ij: (s_onset[ij[0]], p_onset[ij[1]]))
    times = [p_onset[j] for _, j in anchors]
    import bisect

    tails: list[float] = []
    links: list[int] = []
    tail_idx: list[int] = []
    for k, t in enumerate(times):
        pos = bisect.bisect_right(tails, t)
        if pos == len(tails):
            tails.append(t)
            tail_idx.append(k)
        else:
            tails[pos] = t
            tail_idx[pos] = k
        links.append(tail_idx[pos - 1] if pos > 0 else -1)
    # reconstruct
    out = []
    k = tail_idx[len(tails) - 1]
    while k >= 0:
        out.append(anchors[k])
        k = links[k]
    return out[::-1]


def _assign_monotone(expected, actual, tol, conf_block):
    """DP over two short sorted lists: match monotonically, cost = time delta
    minus a confidence bonus; skip allowed both sides."""
    a, b = len(expected), len(actual)
    SKIP = tol * 0.6
    INF = 1e18
    dp = np.full((a + 1, b + 1), INF)
    dp[0, :] = np.arange(b + 1) * SKIP
    dp[:, 0] = np.arange(a + 1) * SKIP
    back = np.zeros((a + 1, b + 1), dtype=np.int8)
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            delta = abs(expected[i - 1] - actual[j - 1])
            match_cost = dp[i - 1, j - 1] + (delta - 0.5 * tol * conf_block[i - 1, j - 1] if delta <= tol else INF)
            del_cost = dp[i - 1, j] + SKIP
            ins_cost = dp[i, j - 1] + SKIP
            best = min(match_cost, del_cost, ins_cost)
            dp[i, j] = best
            back[i, j] = 0 if best == match_cost else (1 if best == del_cost else 2)
    pairs = []
    i, j = a, b
    while i > 0 and j > 0:
        step = back[i, j]
        if step == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def align_with_model(model, score: ScoreTable, perf: PerfTable, device: str = "cpu") -> list[dict]:
    row = tables_to_row(score, perf)
    sim, null_s, null_p = accumulate_logits(model, row, device)
    triples = decode(row, sim, null_s, null_p)
    out = []
    for t in triples:
        conf = t.get("confidence")
        if t["label"] == "match":
            out.append({
                "label": "match",
                "score_id": str(score.notes["id"][t["score_idx"]]),
                "perf_id": str(perf.notes["id"][t["perf_idx"]]),
                "confidence": conf,
            })
        elif t["label"] == "deletion":
            out.append({
                "label": "deletion",
                "score_id": str(score.notes["id"][t["score_idx"]]),
                "confidence": conf,
            })
        else:
            out.append({
                "label": "insertion",
                "perf_id": str(perf.notes["id"][t["perf_idx"]]),
                "confidence": conf,
            })
    return out
