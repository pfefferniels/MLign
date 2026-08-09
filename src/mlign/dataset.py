"""Corpus JSONL → training batches.

v0 keeps whole pieces (corpus pieces are ≤ ~250 notes); windowing arrives with
real-score corpora. Sample layout fed to the model:

    [S-marker] s_1 … s_n [P-marker] p_1 … p_m

Targets: target_s[i] = perf index of the match, or the null index (= m_max,
the shared last column after batch padding), pads -100; target_p mirrored.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

MARKER_PITCH = 128


def load_corpus(paths: list[str | Path]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def featurize(row: dict) -> dict:
    """One corpus row → numpy arrays (unpadded)."""
    score = np.asarray(row["score"], dtype=np.float64)  # [onset_t, dur_t, pitch, voice]
    perf = np.asarray(row["perf"], dtype=np.float64)  # [onset_ms, dur_ms, pitch, vel]
    n, m = len(score), len(perf)

    ppq = 720.0
    s_onset = score[:, 0] / ppq  # quarters
    s_delta = np.diff(s_onset, prepend=s_onset[0])
    s_dur = score[:, 1] / ppq
    p_onset = perf[:, 0] / 1000.0  # seconds
    p_delta = np.diff(p_onset, prepend=p_onset[0])
    p_dur = perf[:, 1] / 1000.0

    def cont_block(delta, dur, pitch, extra, seg_flag):
        return np.stack(
            [
                np.log1p(np.maximum(delta, 0.0) * 2.0),
                np.log1p(np.maximum(dur, 0.0) * 2.0),
                pitch / 64.0 - 1.0,
                (pitch % 12) / 11.0 * 2.0 - 1.0,
                extra,
                np.full_like(delta, seg_flag),
            ],
            axis=1,
        )

    s_cont = cont_block(s_delta, s_dur, score[:, 2], score[:, 3] / 4.0, 0.0)
    p_cont = cont_block(p_delta, p_dur, perf[:, 2], perf[:, 3] / 64.0 - 1.0, 1.0)

    pitch = np.concatenate(
        [[MARKER_PITCH], score[:, 2].astype(np.int64), [MARKER_PITCH], perf[:, 2].astype(np.int64)]
    )
    cont = np.concatenate(
        [np.zeros((1, 6)), s_cont, np.zeros((1, 6)), p_cont], axis=0
    ).astype(np.float32)
    segment = np.concatenate([np.zeros(1 + n, dtype=np.int64), np.ones(1 + m, dtype=np.int64)])
    position = np.concatenate([np.arange(1 + n, dtype=np.int64), np.arange(1 + m, dtype=np.int64)])

    target_s = np.full(n, -1, dtype=np.int64)  # -1 → null, resolved at collate
    for si, pi in row["align"]:
        target_s[si] = pi
    target_p = np.full(m, -1, dtype=np.int64)
    for si, pi in row["align"]:
        target_p[pi] = si
    # ins/del stay -1 (null); everything is covered by corpus invariants.

    return {
        "pitch": pitch,
        "cont": cont,
        "segment": segment,
        "position": position,
        "target_s": target_s,
        "target_p": target_p,
        "n": n,
        "m": m,
    }


def collate(samples: list[dict], device: str = "cpu") -> dict:
    B = len(samples)
    T = max(2 + s["n"] + s["m"] for s in samples)
    n_max = max(s["n"] for s in samples)
    m_max = max(s["m"] for s in samples)

    pitch = torch.zeros((B, T), dtype=torch.long)
    cont = torch.zeros((B, T, 6), dtype=torch.float32)
    segment = torch.zeros((B, T), dtype=torch.long)
    position = torch.zeros((B, T), dtype=torch.long)
    pad = torch.ones((B, T), dtype=torch.bool)
    target_s = torch.full((B, n_max), -100, dtype=torch.long)
    target_p = torch.full((B, m_max), -100, dtype=torch.long)
    n_score = torch.zeros(B, dtype=torch.long)
    n_perf = torch.zeros(B, dtype=torch.long)

    for b, s in enumerate(samples):
        t = 2 + s["n"] + s["m"]
        pitch[b, :t] = torch.from_numpy(np.ascontiguousarray(s["pitch"]))
        cont[b, :t] = torch.from_numpy(s["cont"])
        segment[b, :t] = torch.from_numpy(s["segment"])
        position[b, :t] = torch.from_numpy(s["position"])
        pad[b, :t] = False
        ts = torch.from_numpy(s["target_s"]).clone()
        ts[ts < 0] = m_max  # null column
        target_s[b, : s["n"]] = ts
        tp = torch.from_numpy(s["target_p"]).clone()
        tp[tp < 0] = n_max
        target_p[b, : s["m"]] = tp
        n_score[b] = s["n"]
        n_perf[b] = s["m"]

    batch = {
        "pitch": pitch,
        "cont": cont,
        "segment": segment,
        "position": position,
        "pad": pad,
        "target_s": target_s,
        "target_p": target_p,
        "n_score": n_score,
        "n_perf": n_perf,
    }
    return {k: v.to(device) for k, v in batch.items()}


class CorpusBatcher:
    """Token-budgeted batches over featurized rows, shuffled each epoch."""

    def __init__(self, rows: list[dict], max_tokens: int = 6000, seed: int = 0):
        self.samples = [featurize(r) for r in rows]
        self.max_tokens = max_tokens
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(len(self.samples))
        batch: list[dict] = []
        tokens = 0
        for i in order:
            s = self.samples[i]
            t = 2 + s["n"] + s["m"]
            if batch and tokens + t > self.max_tokens:
                yield batch
                batch, tokens = [], 0
            batch.append(s)
            tokens += t
        if batch:
            yield batch

    def __len__(self):
        total = sum(2 + s["n"] + s["m"] for s in self.samples)
        return max(1, math.ceil(total / self.max_tokens))
