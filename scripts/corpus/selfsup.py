"""Self-supervised corpus source B: corrupt real performance MIDI.

Port of TheGlueNote's `reorder()` augmentation (thegluenote-main
src/datasets/__init__.py:229-412) with fixes:
  - seeded rng (np.random.default_rng), fully deterministic;
  - trill block used `no_of_insertions` for its index-array length (crashes or
    silently mislabels when it differs from the trill length) — fixed;
  - trill pitch alternation kept, but velocities follow the anchor note
    instead of uniform-random over the piece range;
  - segment-skip guarded for short files.

Time units: everything runs in milliseconds (symusic seconds × 1000). Score
rows are emitted at pseudo-ticks = ms × 0.72 (≙ 720 ppq at 120 bpm), voice 0.

Usage:
  .venv/bin/python scripts/corpus/selfsup.py out.jsonl --glob 'data/benchmarks/asap-dataset/**/*[!e].mid' \
      --windows 4 --window-size 512 --seed 7 --exclude-test
"""

from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

INS = -1  # insertion marker in origin array (original used 1e7 / 2e7)


def piece_split(piece_dir: str) -> str:
    """Deterministic piece-level split, journaled in DESIGN/LOG: md5 mod 10."""
    h = int(hashlib.md5(piece_dir.encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def reorder(onsets, velocities, pitches, durations, rng,
            prob_insertions=0.2, prob_deletions=0.2,
            prob_repeats=1.0, range_repeats=(4, 40),
            prob_skips=1.0, range_skips=(4, 40),
            range_tempo_curve=(0.5, 0.5), range_timing=(0, 50),
            range_velocities=(0, 10), range_durations=(0, 250),
            trill=True):
    """Returns corrupted (onsets, velocities, durations, pitches, origin).
    origin[i] = index into the ORIGINAL arrays, or -1 for inserted notes."""
    onsets = onsets.astype(np.float64).copy()
    pitches = pitches.copy()
    durations = durations.astype(np.float64).copy()
    velocities = velocities.copy()
    o_min, o_max = onsets[0], onsets[-1]
    p_min, p_max = int(pitches.min()), int(pitches.max()) + 1
    d_min, d_max = float(durations.min()), float(durations.max()) + 1
    v_min, v_max = int(velocities.min()), int(velocities.max()) + 1

    n0 = len(onsets)
    origin = np.arange(n0)

    # deletions
    k_del = int(prob_deletions * n0)
    delete_idx = rng.choice(n0, size=k_del, replace=False)
    keep = np.setdiff1d(np.arange(n0), delete_idx)
    onsets, pitches, velocities, durations, origin = (
        onsets[keep], pitches[keep], velocities[keep], durations[keep], origin[keep]
    )

    # segment skip
    n = len(onsets)
    if rng.random() < prob_skips and n > range_skips[1] + 2:
        length = rng.integers(range_skips[0], range_skips[1])
        start = rng.integers(0, n - length - 1)
        shift = onsets[start + length] - onsets[start] - 23
        sel = np.r_[0:start, start + length : n]
        onsets, pitches, velocities, durations, origin = (
            onsets[sel], pitches[sel], velocities[sel], durations[sel], origin[sel]
        )
        onsets[start:] -= shift

    # random insertions
    n = len(onsets)
    k_ins = int(prob_insertions * n)
    if k_ins:
        onsets = np.concatenate([onsets, rng.uniform(o_min, o_max, k_ins)])
        velocities = np.concatenate([velocities, rng.integers(v_min, v_max, k_ins)])
        durations = np.concatenate([durations, rng.uniform(d_min, d_max, k_ins)])
        pitches = np.concatenate([pitches, rng.integers(p_min, p_max, k_ins)])
        origin = np.concatenate([origin, np.full(k_ins, INS)])
        order = np.argsort(onsets, kind="stable")
        onsets, pitches, velocities, durations, origin = (
            onsets[order], pitches[order], velocities[order], durations[order], origin[order]
        )

    # trill burst (fires per `trill` flag; original fired always)
    n = len(onsets)
    if trill and n:
        at = rng.integers(n)
        count = int(rng.integers(20, 100))
        step_ms = float(rng.integers(10, 100))
        base_pitch = int(pitches[at])
        alt_pitch = min(127, base_pitch + int(rng.integers(1, 3)))
        t_onsets = onsets[at] + np.arange(count) * step_ms
        t_pitches = np.where(np.arange(count) % 2 == 0, alt_pitch, base_pitch)
        t_durs = rng.uniform(max(5.0, step_ms - 5), step_ms + 30, count)
        t_vels = np.clip(velocities[at] + rng.integers(-8, 8, count), 1, 127)
        onsets = np.concatenate([onsets, t_onsets])
        velocities = np.concatenate([velocities, t_vels])
        durations = np.concatenate([durations, t_durs])
        pitches = np.concatenate([pitches, t_pitches])
        origin = np.concatenate([origin, np.full(count, INS)])  # fixed length bug
        order = np.argsort(onsets, kind="stable")
        onsets, pitches, velocities, durations, origin = (
            onsets[order], pitches[order], velocities[order], durations[order], origin[order]
        )

    # segment repeat (copies are non-matches, like the original's 2e7)
    n = len(onsets)
    if rng.random() < prob_repeats and n > range_repeats[1] + 2:
        length = int(rng.integers(range_repeats[0], range_repeats[1]))
        start = int(rng.integers(0, n - length - 1))
        ins_at = start + length
        shift = onsets[ins_at] - onsets[start] + 23
        onsets = np.insert(onsets, ins_at, onsets[start:ins_at])
        onsets[ins_at:] += shift
        durations = np.insert(durations, ins_at, durations[start:ins_at])
        velocities = np.insert(velocities, ins_at, velocities[start:ins_at])
        pitches = np.insert(pitches, ins_at, pitches[start:ins_at])
        origin = np.insert(origin, ins_at, np.full(length, INS))

    # tempo curve
    n = len(onsets)
    ioi = np.diff(onsets)
    global_tempo = float(np.clip(rng.normal(1, range_tempo_curve[0]), 0.2, 5.0))
    curve = 2.0 ** rng.normal(0, range_tempo_curve[1], size=n - 1)
    onsets = np.cumsum(np.concatenate([onsets[:1], global_tempo * ioi * curve]))

    # onset jitter (re-sorts, so chord-order swaps emerge)
    onsets = onsets + rng.uniform(range_timing[0], range_timing[1], n)
    order = np.argsort(onsets, kind="stable")
    onsets, pitches, velocities, durations, origin = (
        onsets[order], pitches[order], velocities[order], durations[order], origin[order]
    )

    durations = np.clip(durations + rng.uniform(range_durations[0], range_durations[1], n), 30, 500000)
    velocities = np.clip(velocities + rng.integers(range_velocities[0], range_velocities[1], n), 1, 127)

    return onsets, velocities, durations, pitches, origin


def midi_notes_ms(path: str):
    import symusic

    score = symusic.Score(path, ttype="second")
    rows = []
    for track in score.tracks:
        for note in track.notes:
            rows.append((note.time * 1000.0, note.duration * 1000.0, note.pitch, note.velocity))
    rows.sort(key=lambda r: (r[0], r[2]))
    return np.array(rows, dtype=np.float64)


def window_rows(notes: np.ndarray, size: int, count: int, rng) -> list[np.ndarray]:
    if len(notes) <= size:
        return [notes] * (1 if len(notes) >= 32 else 0)
    starts = rng.integers(0, len(notes) - size, size=count)
    return [notes[s : s + size] for s in starts]


def build_row(win: np.ndarray, rng, meta: dict) -> dict | None:
    onsets, durs, pitches, vels = win[:, 0], win[:, 1], win[:, 2].astype(int), win[:, 3].astype(int)
    onsets = onsets - onsets[0]
    try:
        c_on, c_vel, c_dur, c_pit, origin = reorder(onsets, vels, pitches, durs, rng)
    except Exception:
        return None
    c_on = c_on - (c_on[origin >= 0].min() if (origin >= 0).any() else c_on.min())

    score = [[round(o * 0.72, 3), round(d * 0.72, 3), int(p), 0] for o, d, p in zip(onsets, durs, pitches)]
    perf = [[round(o, 3), round(d, 3), int(p), int(v)] for o, d, p, v in zip(c_on, c_dur, c_pit, c_vel)]
    align = [[int(origin[i]), i] for i in range(len(origin)) if origin[i] >= 0]
    ins = [[i, 3] for i in range(len(origin)) if origin[i] < 0]
    matched = {int(origin[i]) for i in range(len(origin)) if origin[i] >= 0}
    dele = [i for i in range(len(score)) if i not in matched]
    return {
        "meta": meta,
        "score": score,
        "scoreIds": [f"s{i}" for i in range(len(score))],
        "perf": perf,
        "align": align,
        "subs": [],
        "ins": ins,
        "del": dele,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--glob", required=True)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-test", action="store_true",
                    help="drop files whose piece dir hashes to val/test")
    ap.add_argument("--exclude-folders", default="",
                    help="file with piece folders (relative to asap root) to exclude, one per line")
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob, recursive=True))
    files = [f for f in files if not f.endswith("midi_score.mid")]
    if args.exclude_test:
        files = [f for f in files if piece_split(str(Path(f).parent)) == "train"]
    if args.exclude_folders:
        excl = {l.strip() for l in open(args.exclude_folders) if l.strip()}
        before = len(files)
        files = [
            f for f in files
            if not any(folder in str(Path(f).parent).replace("\\", "/") for folder in excl)
        ]
        print(f"excluded {before - len(files)} files via folder list", file=sys.stderr)
    print(f"{len(files)} files", file=sys.stderr)

    written = 0
    with open(args.out, "w") as out:
        for fi, path in enumerate(files):
            rng = np.random.default_rng([args.seed, fi])
            try:
                notes = midi_notes_ms(path)
            except Exception as err:
                print(f"skip {path}: {err}", file=sys.stderr)
                continue
            if len(notes) < 32:
                continue
            for w, win in enumerate(window_rows(notes, args.window_size, args.windows, rng)):
                row = build_row(win, rng, {"gen": "selfsup-v0", "src": str(Path(path).name), "w": w})
                if row is not None:
                    out.write(json.dumps(row) + "\n")
                    written += 1
            if (fi + 1) % 100 == 0:
                print(f"...{fi + 1}/{len(files)} ({written} rows)", file=sys.stderr)
    print(f"wrote {written} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
