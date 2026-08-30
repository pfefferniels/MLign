"""Golden fixtures for the TypeScript port of the MLign decode.

The browser build re-implements src/mlign/infer.py in TypeScript. Nothing in
that port is covered by the Python side, so a divergence — a DTW tie broken the
other way, a float32 accumulation done in float64, np.interp's clamping outside
the knot range, np.unique keeping the wrong duplicate — surfaces only as a
slightly worse F1 on a benchmark nobody re-runs. This script freezes what the
Python actually computes so the port can be diffed against it.

Per configured (score, performance) pair it runs the real pipeline

    tables -> row -> featurize/collate -> model -> sim/null_s/null_p -> decode

and writes every intermediate the port has to reproduce. Large float arrays go
out as raw little-endian float32 .bin (sim alone is 12 MB for the Berceuse);
everything else lives in manifest.json, whose "arrays" block documents the byte
layout. docs/DECODE-CONTRACT.md is the prose companion — this script is the
machine-checkable half.

Stage snapshots (anchors, DTW map, per-round matches, rescues) come from a
traced copy of decode() kept in this file. The copy is not trusted: every run
asserts that it reproduces the real decode()'s triples exactly AND that the
real decode makes the identical sequence of _monotone_subset /
_cluster_dtw_map / _assign_monotone calls (recorded by wrapping them).

Usage:
  PYTHONPATH=src .venv/bin/python scripts/make_golden.py           # generate + verify
  PYTHONPATH=src .venv/bin/python scripts/make_golden.py --verify  # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign import infer  # noqa: E402
from mlign.dataset import collate, featurize  # noqa: E402
from mlign.infer import (  # noqa: E402
    _softmax,
    accumulate_logits,
    coarse_windows,
    decode,
    tables_to_row,
)
from mlign.model import ModelConfig, NoteAligner  # noqa: E402
from mlign.tables import PerfTable, ScoreTable  # noqa: E402

SCHEMA = "mlign-golden/1"

# (slug, score path, performance path, module-constant overrides). The fourth
# element patches mlign.infer's module globals for that fixture's run only.
#
# The Berceuse is the realistic multi-window case but costs 25 MB, which is a
# bad trade for a repo that only wants to run the port's tests. So the windowed
# path is ALSO covered at 1/25th the size by re-running the Schubert with a
# small WIN_SCORE. MAX_SINGLE_TOKENS has to come down too: Schubert is 646
# tokens, so without that it takes the single-window branch whatever WIN_SCORE
# says. 0 means "always window".
PIECES: list[tuple[str, str, str, dict]] = [
    ("schubert-d783-15", "web/demo/schubert_d783_15.musicxml", "web/demo/schubert_d783_15_p01.mid", {}),
    (
        "schubert-d783-15-win128",
        "web/demo/schubert_d783_15.musicxml",
        "web/demo/schubert_d783_15_p01.mid",
        {"WIN_SCORE": 128, "MAX_SINGLE_TOKENS": 0},
    ),
    (
        "mozart-k331-1st-mov",
        "data/benchmarks/vienna4x22/musicxml/Mozart_K331_1st-mov.musicxml",
        "data/benchmarks/vienna4x22/midi/Mozart_K331_1st-mov_p01.mid",
        {},
    ),
    (
        "chopin-op38-p19",
        "data/benchmarks/vienna4x22/musicxml/Chopin_op38.musicxml",
        "data/benchmarks/vienna4x22/midi/Chopin_op38_p19.mid",
        {},
    ),
    (
        "chopin-berceuse-op57",
        "data/benchmarks/asap-dataset/Chopin/Berceuse_op_57/xml_score.musicxml",
        "data/benchmarks/asap-dataset/Chopin/Berceuse_op_57/Tario07M.mid",
        {},
    ),
]


@contextmanager
def overridden(overrides: dict):
    """Patch mlign.infer module constants for one fixture, then restore. Keeps
    the generator honest about not editing src/mlign/infer.py.

    The window constants are validated rather than trusted: coarse_windows feeds
    WIN_SCORE // 2 to range(), which rejects a float stride (TypeError) and a
    zero one (ValueError) — but only after a window's worth of work, and a port
    that indexes with the same value may not reject either. Fail here instead."""
    for k in overrides:
        if not hasattr(infer, k):
            raise SystemExit(f"unknown mlign.infer constant {k!r}")
    for k in ("WIN_SCORE", "MAX_SINGLE_TOKENS"):
        if k in overrides and not isinstance(overrides[k], int):
            raise SystemExit(f"{k} override must be an int, got {overrides[k]!r}")
    if overrides.get("WIN_SCORE", infer.WIN_SCORE) < 2:
        raise SystemExit("WIN_SCORE override must be >= 2 (stride = WIN_SCORE // 2 must be nonzero)")
    saved = {k: getattr(infer, k) for k in overrides}
    for k, v in overrides.items():
        setattr(infer, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(infer, k, v)

# Two decode branches survive all four real pieces untouched, and a port could
# get them wrong (or omit them) and still pass:
#
#   * the residual rescue never fires. It cannot, while s2p_time is monotone:
#     the DP's optimum can never leave a same-pitch pair that is addable
#     without crossing, because adding one costs at most tol=1.0 and removes
#     two SKIPs worth 1.2. The rescue only becomes reachable when the
#     interpolation map is locally NON-monotone — the anchor/DTW union is
#     sorted by score onset but its perf onsets need not be — which real
#     alignments of this quality never produce.
#   * `len(got) < 8` (single round) and `len(ax) < 2` (s2p_time ≡ 0).
#
# So both are covered by synthetic decode-only fixtures: a seeded random
# (row, sim, null_s, null_p) fed to the real decode. The inputs are fabricated;
# the recorded output is the real Python's, which is all the port must match.
# Seeds found by scanning synth_case(seed) for the wanted branch.
SYNTHETIC: list[tuple[str, int, str]] = [
    ("synth-rescue", 185, "residual rescue fires (2 notes); single assignment round"),
    ("synth-flat-map", 60, "s2p_time is the len(ax)<2 zeros fallback; single assignment round"),
]


def synth_case(seed: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Self-contained pseudo-random decode input. Fully determined by `seed`,
    so a fixture only has to record the integer."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 22))
    m = int(rng.integers(6, 22))
    flat = bool(rng.random() < 0.15)  # one onset cluster per side
    lo = int(rng.integers(55, 70))
    s_pitch = rng.integers(lo, lo + int(rng.integers(2, 8)), n)
    p_pitch = rng.integers(lo, lo + int(rng.integers(2, 8)), m)
    if flat:
        s_on, p_on = np.zeros(n), np.sort(rng.random(m) * 0.04)
    else:
        s_on = np.sort(rng.integers(0, 10, n)).astype(float)
        p_on = np.sort(rng.random(m) * 8.0)
    row = {
        "score": [[float(s_on[i] * 720.0), 720.0, int(s_pitch[i]), int(i % 5)] for i in range(n)],
        "perf": [[float(p_on[j] * 1000.0), 300.0, int(p_pitch[j]), int(40 + j % 60)] for j in range(m)],
        "align": [], "subs": [], "ins": [], "del": [],
    }
    sim = (rng.random((n, m)) * 14.0 - 5.0).astype(np.float32)
    null_s = (rng.random(n) * 5.0 - 2.5).astype(np.float32)
    null_p = (rng.random(m) * 5.0 - 2.5).astype(np.float32)
    return row, sim, null_s, null_p


# Everything the contract pins, emitted into every manifest so a fixture is
# readable without the source tree. Read live from the module so an overridden
# fixture records the values its run actually used, not the defaults.
def constants() -> dict:
    return {
        "MARKER_PITCH": 128,
        "PPQ": 720.0,
        "MAX_SINGLE_TOKENS": infer.MAX_SINGLE_TOKENS,
        "WIN_SCORE": infer.WIN_SCORE,
        "WIN_STRIDE": infer.WIN_SCORE // 2,
        "MARGIN_SEC": infer.MARGIN_SEC,
        "UNCOVERED_SIM": -1e9,
        "UNCOVERED_NULL": 1e9,
        "ANCHOR_CONF": 0.35,
        "TOL_SEC": 1.0,
        "SKIP_FACTOR": 0.6,
        "ASSIGN_INF": 1e18,
        "CONF_BONUS_FACTOR": 0.5,
        "RESCUE_SEC": 0.35,
        "DTW_GAP_DECODE": 0.6,
        "DTW_GAP_BASELINE": 0.75,
        "DTW_CONF_GAIN": 20.0,
        "SCORE_CLUSTER_EPS": 1e-9,
        "PERF_CLUSTER_EPS": 0.05,
    }


# --------------------------------------------------------------------------
# traced decode


def _decode_traced(row: dict, sim: np.ndarray, null_s: np.ndarray, null_p: np.ndarray,
                   anchor_conf: float = 0.35, tol_sec: float = 1.0) -> tuple[list[dict], dict]:
    """Line-for-line copy of infer.decode that also returns stage snapshots.

    Any edit to infer.decode breaks the equality assertions in `build()`, which
    is the point: the copy exists to observe, never to define."""
    st: dict = {}
    n, m = sim.shape
    s_pitch = np.array([r[2] for r in row["score"]], dtype=int)
    p_pitch = np.array([r[2] for r in row["perf"]], dtype=int)
    s_onset = np.array([r[0] for r in row["score"]]) / 720.0
    p_onset = np.array([r[0] for r in row["perf"]]) / 1000.0

    a = np.concatenate([sim, null_s[:, None]], axis=1)
    b = np.concatenate([sim.T, null_p[:, None]], axis=1)
    sm_s = infer._softmax(a, axis=1)[:, :m]
    sm_p = infer._softmax(b, axis=1)[:, :n]
    conf = sm_s * sm_p.T

    best_p = conf.argmax(axis=1)
    best_s = conf.argmax(axis=0)
    anchors = []
    for i in range(n):
        j = best_p[i]
        if best_s[j] == i and conf[i, j] >= anchor_conf and s_pitch[i] == p_pitch[j]:
            anchors.append((i, j))
    st["anchors_raw"] = [[int(i), int(j)] for i, j in anchors]
    anchors = infer._monotone_subset(anchors, s_onset, p_onset)
    st["anchors"] = [[int(i), int(j)] for i, j in anchors]

    dtw_ax, dtw_ay = infer._cluster_dtw_map(s_onset, s_pitch, p_onset, p_pitch, conf)
    st["dtw_ax"], st["dtw_ay"] = dtw_ax, dtw_ay

    ax = np.concatenate([dtw_ax, [s_onset[i] for i, _ in anchors]])
    ay = np.concatenate([dtw_ay, [p_onset[j] for _, j in anchors]])
    order = np.argsort(ax, kind="stable")
    ax, ay = ax[order], ay[order]
    if len(ax) >= 2:
        ax, keep = np.unique(ax, return_index=True)
        ay = np.asarray(ay)[keep]

        def s2p_time(x):
            return np.interp(x, ax, ay)

        st["map1_ax"], st["map1_ay"] = ax, ay
    else:
        def s2p_time(x):
            return np.zeros_like(np.asarray(x, dtype=float))

        st["map1_ax"] = st["map1_ay"] = np.zeros(0)

    covered_p = null_p < infer.UNCOVERED_NULL
    st["uncovered_perf"] = [int(j) for j in np.flatnonzero(~covered_p)]

    matched_s = np.full(n, -1, dtype=int)
    matched_p = np.full(m, -1, dtype=int)
    st["map2_ax"] = st["map2_ay"] = np.zeros(0)
    st["rounds_run"] = 0
    for _round in range(2):
        st["rounds_run"] = _round + 1
        matched_s.fill(-1)
        matched_p.fill(-1)
        pitches, counts = np.unique(s_pitch, return_counts=True)
        for pitch in pitches[np.argsort(counts)]:
            si = np.flatnonzero((s_pitch == pitch) & (matched_s == -1))
            pj = np.flatnonzero((p_pitch == pitch) & (matched_p == -1) & covered_p)
            if len(si) == 0 or len(pj) == 0:
                continue
            exp = s2p_time(s_onset[si])
            pairs = infer._assign_monotone(exp, p_onset[pj], tol_sec, conf[np.ix_(si, pj)])
            for a_i, b_j in pairs:
                matched_s[si[a_i]] = pj[b_j]
                matched_p[pj[b_j]] = si[a_i]
        st[f"round{_round + 1}_matched_s"] = matched_s.copy()
        if _round == 0:
            got = np.flatnonzero(matched_s >= 0)
            if len(got) < 8:
                break
            pairs2 = infer._monotone_subset(
                [(int(i), int(matched_s[i])) for i in got], s_onset, p_onset
            )
            if len(pairs2) < 8:
                break
            rx = np.array([s_onset[i] for i, _ in pairs2])
            ry = np.array([p_onset[j] for _, j in pairs2])
            rx, keep2 = np.unique(rx, return_index=True)
            ry = ry[keep2]
            st["map2_ax"], st["map2_ay"] = rx, ry

            def s2p_time(x, _rx=rx, _ry=ry):  # noqa: F811
                return np.interp(x, _rx, _ry)

    RESCUE_SEC = 0.35
    res_s = [i for i in range(n) if matched_s[i] < 0]
    res_p = [j for j in range(m) if matched_p[j] < 0 and covered_p[j]]
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
    st["rescued"] = []
    for d, i, j in cands:
        if matched_s[i] < 0 and matched_p[j] < 0:
            matched_s[i] = j
            matched_p[j] = i
            st["rescued"].append([int(i), int(j)])

    triples: list[dict] = []
    for i in range(n):
        if matched_s[i] >= 0:
            j = int(matched_s[i])
            triples.append({
                "label": "match", "score_idx": i, "perf_idx": j,
                "confidence": float(conf[i, j]),
            })
        else:
            null_share = float(infer._softmax(np.concatenate([sim[i], [null_s[i]]]), axis=0)[-1])
            triples.append({"label": "deletion", "score_idx": i, "confidence": null_share})
    for j in range(m):
        if matched_p[j] < 0:
            null_share = float(infer._softmax(np.concatenate([sim[:, j], [null_p[j]]]), axis=0)[-1])
            triples.append({"label": "insertion", "perf_idx": j, "confidence": null_share})
    st["conf"] = conf
    return triples, st


class _CallLog:
    """Wraps infer's decode helpers so the real decode's internal call sequence
    can be compared against the traced copy's, argument for argument."""

    NAMES = ("_monotone_subset", "_cluster_dtw_map", "_assign_monotone")

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._orig: dict = {}

    def __enter__(self) -> "_CallLog":
        for name in self.NAMES:
            fn = self._orig[name] = getattr(infer, name)

            def wrapper(*args, _fn=fn, _name=name):
                out = _fn(*args)
                self.calls.append((_name, _digest(args), _digest(out)))
                return out

            setattr(infer, name, wrapper)
        return self

    def __exit__(self, *exc) -> None:
        for name, fn in self._orig.items():
            setattr(infer, name, fn)


def _digest(obj) -> str:
    """Order-sensitive hash of arbitrarily nested arrays/lists/scalars."""
    h = hashlib.sha256()

    def walk(o):
        if isinstance(o, np.ndarray):
            h.update(b"A")
            h.update(str(o.dtype).encode())
            h.update(np.ascontiguousarray(o).tobytes())
        elif isinstance(o, (list, tuple)):
            h.update(b"L")
            for x in o:
                walk(x)
            h.update(b"l")
        else:
            h.update(repr(float(o) if isinstance(o, (int, float, np.number)) else o).encode())

    walk(obj)
    return h.hexdigest()


# --------------------------------------------------------------------------
# fixture IO


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_f32(out_dir: Path, name: str, arr: np.ndarray, arrays: dict, note: str = "") -> None:
    """Raw little-endian float32, C order — `new Float32Array(buf)` in Node."""
    a = np.ascontiguousarray(arr, dtype="<f4")
    path = out_dir / f"{name}.f32.bin"
    path.write_bytes(a.tobytes())
    arrays[name] = {
        "file": path.name,
        "dtype": "float32",
        "endianness": "little",
        "order": "C",
        "shape": list(a.shape),
        "offset": 0,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "note": note,
    }


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return [_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def _git_head() -> tuple[str, bool]:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                capture_output=True, text=True).stdout.strip())
    return head, dirty


def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mcfg = ModelConfig(
        d_model=cfg.get("d_model", 192),
        n_layers=cfg.get("n_layers", 4),
        matchability=cfg.get("matchability", False),
    )
    model = NoteAligner(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, mcfg


# --------------------------------------------------------------------------
# generation


def build(slug: str, score_path: Path, perf_path: Path, out_dir: Path,
          model, mcfg: ModelConfig, ckpt_path: Path, device: str,
          overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    score = ScoreTable.from_musicxml(score_path)
    perf = PerfTable.from_midi(perf_path)
    row = tables_to_row(score, perf)
    n, m = len(score), len(perf)
    tokens = 2 + n + m
    windowed = tokens > infer.MAX_SINGLE_TOKENS

    sim, null_s, null_p = accumulate_logits(model, row, device)

    with _CallLog() as log:
        triples = decode(row, sim, null_s, null_p)
    real_calls = list(log.calls)
    with _CallLog() as log2:
        triples_t, stages = _decode_traced(row, sim, null_s, null_p)
    assert log2.calls == real_calls, f"{slug}: traced decode diverges in helper calls"
    assert triples_t == triples, f"{slug}: traced decode diverges in triples"

    windows = coarse_windows(row, n, m) if windowed else [(0, n, 0, m)]

    # Window coverage census: how many windows touch each cell/note, and where
    # the sentinels actually landed. Both are exactly representable in float32,
    # so a reader can test them with ==.
    cell_cnt = np.zeros((n, m), dtype=np.int32)
    s_cnt = np.zeros(n, dtype=np.int32)
    p_cnt = np.zeros(m, dtype=np.int32)
    for s0, s1, p0, p1 in windows:
        cell_cnt[s0:s1, p0:p1] += 1
        s_cnt[s0:s1] += 1
        p_cnt[p0:p1] += 1
    census = lambda a: {str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))}  # noqa: E731
    coverage = {
        "sim_cells_by_window_count": census(cell_cnt),
        "score_notes_by_window_count": census(s_cnt),
        "perf_notes_by_window_count": census(p_cnt),
        "sim_at_minus_1e9": int((sim == np.float32(-1e9)).sum()),
        "null_s_at_plus_1e9": int((null_s == np.float32(1e9)).sum()),
        "null_p_at_plus_1e9": int((null_p == np.float32(1e9)).sum()),
        "covered_sim_min": float(sim[cell_cnt > 0].min()),
        "covered_sim_max": float(sim[cell_cnt > 0].max()),
    }
    assert np.array_equal(sim == np.float32(-1e9), cell_cnt == 0), \
        f"{slug}: the -1e9 sentinel does not coincide with the uncovered cells"

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    _write_f32(out_dir, "sim", sim, arrays,
               "accumulate_logits output: MEAN over covering windows of "
               "(logits_s2p + logits_p2s^T), i.e. 2x the model's bilinear sim "
               "for a singly-covered cell; exactly -1e9 where no window covered")
    _write_f32(out_dir, "null_s", null_s, arrays,
               "mean over covering windows; exactly +1e9 where no window covered")
    _write_f32(out_dir, "null_p", null_p, arrays,
               "mean over covering windows; exactly +1e9 where no window covered")
    _write_f32(out_dir, "conf", stages.pop("conf"), arrays,
               "dual-softmax confidence sm_s * sm_p.T, the decode's working matrix")

    featurized = []
    for w, (s0, s1, p0, p1) in enumerate(windows):
        sub = {
            "score": row["score"][s0:s1], "perf": row["perf"][p0:p1],
            "align": [], "subs": [], "ins": [], "del": [],
        }
        f = featurize(sub)
        _write_f32(out_dir, f"feat_w{w:02d}_cont", f["cont"], arrays,
                   f"featurize.cont for window {w} = [{s0},{s1},{p0},{p1}]")
        featurized.append({
            "window": [s0, s1, p0, p1],
            "n": int(f["n"]), "m": int(f["m"]), "T": int(2 + f["n"] + f["m"]),
            "pitch": f["pitch"].tolist(),
            "segment": f["segment"].tolist(),
            "position": f["position"].tolist(),
            "cont": f"feat_w{w:02d}_cont",
        })

    baseline_pairs = None
    if windowed:
        # coarse_windows' hidden dependency: the baseline aligner's matches over
        # index-ids, which set every window's performance range.
        from mlign.baseline import align_baseline

        bscore = ScoreTable(np.array(
            [(r[0] / 720.0, r[1] / 720.0, int(r[2]), int(r[3]), str(i)) for i, r in enumerate(row["score"])],
            dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("voice", "i4"), ("id", "U32")],
        ))
        bperf = PerfTable(np.array(
            [(r[0] / 1000.0, r[1] / 1000.0, int(r[2]), int(r[3]), str(i)) for i, r in enumerate(row["perf"])],
            dtype=[("onset", "f8"), ("duration", "f8"), ("pitch", "i4"), ("velocity", "i4"), ("id", "U32")],
        ))
        bt = align_baseline(bscore, bperf)
        baseline_pairs = sorted(
            (int(t["score_id"]), int(t["perf_id"])) for t in bt if t["label"] == "match"
        )

    head, dirty = _git_head()
    labels = [t["label"] for t in triples]
    manifest = {
        "schema": SCHEMA,
        "piece": slug,
        "meta": {
            "score_path": str(score_path.relative_to(ROOT)),
            "perf_path": str(perf_path.relative_to(ROOT)),
            "n": n, "m": m, "tokens": tokens, "windowed": windowed,
            # NOT the shipped defaults where non-empty: mlign.infer's module
            # constants were patched at runtime for this fixture only, to force
            # the windowed path on a small piece. See scripts/make_golden.py.
            "overrides": overrides,
            "coverage": coverage,
            "checkpoint": str(ckpt_path.relative_to(ROOT)),
            "checkpoint_sha256": _sha256(ckpt_path),
            "model_config": {
                "d_model": mcfg.d_model, "n_layers": mcfg.n_layers, "n_heads": mcfg.n_heads,
                "d_ff": mcfg.d_ff, "max_rel": mcfg.max_rel, "matchability": mcfg.matchability,
            },
            "git_commit": head, "git_dirty": dirty,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": {
                "python": sys.version.split()[0], "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "constants": constants(),
            "counts": {lab: labels.count(lab) for lab in ("match", "deletion", "insertion")},
        },
        "arrays": arrays,
        # ScoreTable/PerfTable order: np.lexsort((pitch, onset)) — onset primary,
        # pitch secondary, stable. Index into these == score_idx/perf_idx.
        "score": [
            {"id": str(x["id"]), "onset": float(x["onset"]), "duration": float(x["duration"]),
             "pitch": int(x["pitch"]), "voice": int(x["voice"])}
            for x in score.notes
        ],
        "perf": [
            {"id": str(x["id"]), "onset": float(x["onset"]), "duration": float(x["duration"]),
             "pitch": int(x["pitch"]), "velocity": int(x["velocity"])}
            for x in perf.notes
        ],
        # tables_to_row output — the exact featurize/decode input. Kept verbatim
        # because decode's s_onset is row[0]/720, which is NOT bit-identical to
        # score.onset (the *720 then /720 round trip loses a ulp ~17% of the time).
        "row": {"score": row["score"], "perf": row["perf"]},
        "windows": [list(w) for w in windows],
        "baseline_pairs": baseline_pairs,
        "featurized": featurized,
        "stages": _jsonable(stages),
        "triples": triples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1) + "\n")
    return manifest


def build_synthetic(slug: str, seed: int, note: str, out_dir: Path) -> dict:
    """Decode-only fixture: no score file, no model, no featurize, no windows.
    Same manifest shape as a real piece, minus the keys that mean nothing here."""
    row, sim, null_s, null_p = synth_case(seed)
    n, m = len(row["score"]), len(row["perf"])

    with _CallLog() as log:
        triples = decode(row, sim, null_s, null_p)
    real_calls = list(log.calls)
    with _CallLog() as log2:
        triples_t, stages = _decode_traced(row, sim, null_s, null_p)
    assert log2.calls == real_calls, f"{slug}: traced decode diverges in helper calls"
    assert triples_t == triples, f"{slug}: traced decode diverges in triples"

    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict = {}
    _write_f32(out_dir, "sim", sim, arrays, "fabricated logits, not model output")
    _write_f32(out_dir, "null_s", null_s, arrays, "fabricated")
    _write_f32(out_dir, "null_p", null_p, arrays, "fabricated")
    _write_f32(out_dir, "conf", stages.pop("conf"), arrays, "dual-softmax sm_s * sm_p.T")

    head, dirty = _git_head()
    labels = [t["label"] for t in triples]
    manifest = {
        "schema": SCHEMA,
        "piece": slug,
        "meta": {
            "synthetic": True, "seed": seed, "covers": note,
            "generator": "scripts/make_golden.py:synth_case",
            "n": n, "m": m, "windowed": False, "overrides": {}, "coverage": None,
            "git_commit": head, "git_dirty": dirty,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
            "constants": constants(),
            "counts": {lab: labels.count(lab) for lab in ("match", "deletion", "insertion")},
        },
        "arrays": arrays,
        "score": None,
        "perf": None,
        "row": {"score": row["score"], "perf": row["perf"]},
        "windows": [[0, n, 0, m]],
        "baseline_pairs": None,
        "featurized": None,
        "stages": _jsonable(stages),
        "triples": triples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(_jsonable(manifest), indent=1) + "\n")
    return manifest


# --------------------------------------------------------------------------
# verification


def _load_f32(out_dir: Path, spec: dict) -> np.ndarray:
    raw = (out_dir / spec["file"]).read_bytes()
    assert len(raw) == spec["bytes"], f"{spec['file']}: size {len(raw)} != {spec['bytes']}"
    return np.frombuffer(raw, dtype="<f4").reshape(spec["shape"]).copy()


def verify(out_dir: Path) -> str:
    """Reload the fixture and re-run decode from the stored arrays only."""
    man = json.loads((out_dir / "manifest.json").read_text())
    assert man["schema"] == SCHEMA
    sim = _load_f32(out_dir, man["arrays"]["sim"])
    null_s = _load_f32(out_dir, man["arrays"]["null_s"])
    null_p = _load_f32(out_dir, man["arrays"]["null_p"])
    row = {"score": man["row"]["score"], "perf": man["row"]["perf"], "align": [],
           "subs": [], "ins": [], "del": []}

    triples = decode(row, sim, null_s, null_p)
    stored = man["triples"]
    assert len(triples) == len(stored), f"{out_dir.name}: {len(triples)} vs {len(stored)} triples"
    for a, b in zip(triples, stored):
        assert a["label"] == b["label"], f"{out_dir.name}: label {a} vs {b}"
        assert a.get("score_idx") == b.get("score_idx"), f"{out_dir.name}: {a} vs {b}"
        assert a.get("perf_idx") == b.get("perf_idx"), f"{out_dir.name}: {a} vs {b}"
        assert a["confidence"] == b["confidence"], f"{out_dir.name}: conf {a} vs {b}"

    conf = _load_f32(out_dir, man["arrays"]["conf"])
    assert np.array_equal(conf, _dual_softmax(sim, null_s, null_p)), f"{out_dir.name}: conf"

    _check_pitch_order_irrelevant(row, sim, null_s, null_p, triples)
    return f"{out_dir.name}: {len(stored)} triples re-derived from stored arrays, identical"


def _dual_softmax(sim: np.ndarray, null_s: np.ndarray, null_p: np.ndarray) -> np.ndarray:
    n, m = sim.shape
    sm_s = _softmax(np.concatenate([sim, null_s[:, None]], axis=1), axis=1)[:, :m]
    sm_p = _softmax(np.concatenate([sim.T, null_p[:, None]], axis=1), axis=1)[:, :n]
    return sm_s * sm_p.T


def _check_pitch_order_irrelevant(row, sim, null_s, null_p, triples, trials: int = 3) -> None:
    """decode iterates pitches in `pitches[np.argsort(counts)]` order, and
    np.argsort's default kind is an UNSTABLE introsort — no TS sort reproduces
    its tie order (with all-equal counts it does not even return arange). It
    does not have to: the per-pitch subproblems are disjoint, because a pair is
    only ever formed between equal pitches, so no pitch's assignment can touch
    another's matched_s/matched_p entries. The order is therefore inert; assert
    that empirically instead of only by argument, by feeding decode a random
    pitch permutation. The contract lets the port iterate pitches ascending."""
    real = np.argsort
    rng = np.random.default_rng(0)

    def shuffling_argsort(x, *a, **k):
        # only the bare np.argsort(counts) call is hijacked; the map's
        # np.argsort(ax, kind="stable") passes a kind and is left alone.
        if not a and not k:
            return rng.permutation(len(x))
        return real(x, *a, **k)

    np.argsort = shuffling_argsort
    try:
        for _ in range(trials):
            assert decode(row, sim, null_s, null_p) == triples, \
                "pitch iteration order changed the decode output"
    finally:
        np.argsort = real


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="test/golden", help="fixture root (default: test/golden)")
    # Any checkpoint does. A fixture pins the DECODE, and `verify()` re-runs it
    # from the stored sim/null_s/null_p alone, never loading a model — so the
    # checkpoint only chooses which numbers the port is diffed against, not what
    # is being tested. It does not track the released model and need not: a
    # release bump would rewrite every fixture and move the port's diffs for
    # nothing. The fixtures on disk predate this default and were built from
    # `models/mlign-v1.pt`, which their manifests record.
    ap.add_argument("--ckpt", default="models/mlign-v2.pt")
    ap.add_argument("--pieces", default="", help="comma-separated slugs (default: all)")
    ap.add_argument("--verify", action="store_true", help="verify existing fixtures, generate nothing")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    root = ROOT / args.out
    wanted = {s.strip() for s in args.pieces.split(",") if s.strip()}
    pieces = [p for p in PIECES if not wanted or p[0] in wanted]
    synth = [s for s in SYNTHETIC if not wanted or s[0] in wanted]

    def report(slug: str, man: dict, t0: float) -> None:
        size = sum(f.stat().st_size for f in (root / slug).iterdir())
        c = man["meta"]["counts"]
        ov = man["meta"].get("overrides") or {}
        print(
            f"{slug:<26} n={man['meta']['n']:>5} m={man['meta']['m']:>5} "
            f"windows={len(man['windows'])} "
            f"{c['match']}/{c['deletion']}/{c['insertion']} m/d/i "
            f"{size / 1e6:.2f} MB {time.time() - t0:.1f}s"
            + (f"  [overrides: {', '.join(f'{k}={v}' for k, v in ov.items())}]" if ov else ""),
            flush=True,
        )

    if not args.verify:
        if pieces:
            ckpt = ROOT / args.ckpt
            model, mcfg = load_model(ckpt, args.device)
            for slug, sp, pp, ov in pieces:
                t0 = time.time()
                with overridden(ov):
                    man = build(slug, ROOT / sp, ROOT / pp, root / slug,
                                model, mcfg, ckpt, args.device, ov)
                report(slug, man, t0)
        for slug, seed, note in synth:
            t0 = time.time()
            report(slug, build_synthetic(slug, seed, note, root / slug), t0)

    for slug, *_ in pieces + synth:
        print(verify(root / slug), flush=True)
    total = sum(f.stat().st_size for d in root.iterdir() if d.is_dir() for f in d.iterdir())
    print(f"total {total / 1e6:.2f} MB in {root}")


if __name__ == "__main__":
    main()
