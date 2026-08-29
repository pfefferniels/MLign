"""Prove the exported ONNX encoder is interchangeable with the PyTorch model.

Three checks on a real score and a real performance, run through
`mlign.infer` twice — once with the eager model, once with the ONNX session
plus the host-side head math the TypeScript port will implement:

  1. tensor sweep — the four graph outputs against the eager wrapper across
     sequence lengths, so a shape bug in the dynamic T axis cannot hide;
  2. accumulated logits — (sim, null_s, null_p) against
     `infer.accumulate_logits`. This is the sensitive check and the one an
     ONNX port must be accepted on;
  3. decoded triples — identical labels and ids.

(3) is what a user would notice, but it is a weak signal: an injected bug that
takes the matchability nulls from the projections instead of the encoder
output moves the logits by 138% and still decodes to byte-identical triples on
this piece, because the decode leans on pitch equality and onset-cluster DTW.
Same for dropping the symmetric term (50%) and for an off-by-one in the score
slice (62%). All three are caught by (2) at a relative error of 1e-4. Do not
accept an ONNX or TypeScript port on the triples alone.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/test_onnx_parity.py \
      --onnx models/mlign-v2.onnx --ckpt models/mlign-v2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from export_onnx import INPUT_NAMES, Enc, EncDustbin, load_model  # noqa: E402
from mlign.dataset import MARKER_PITCH, featurize  # noqa: E402
from mlign import infer  # noqa: E402
from mlign.model import NoteAligner  # noqa: E402
from mlign.tables import PerfTable, ScoreTable  # noqa: E402

# fp32 weights round-trip to ~1e-5; --fp16 weight storage costs ~2e-3 on the
# raw vectors (and, so far, nothing at all on the decoded alignment).
TOL = {"float32": 1e-4, "float16": 5e-3}


def make_session(onnx_path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # match the browser's single-threaded default
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def fake_batch(T: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random but structurally valid: markers, split segments, per-segment
    positions. A batch of zeros would not exercise the relative-position
    bucketing at all."""
    n = (T - 2) // 2
    m = T - 2 - n
    return {
        "pitch": np.concatenate([[MARKER_PITCH], rng.integers(0, 128, n),
                                 [MARKER_PITCH], rng.integers(0, 128, m)])[None].astype(np.int64),
        "cont": rng.standard_normal((1, T, 6)).astype(np.float32),
        "segment": np.concatenate([np.zeros(1 + n), np.ones(1 + m)])[None].astype(np.int64),
        "position": np.concatenate([np.arange(1 + n), np.arange(1 + m)])[None].astype(np.int64),
    }


def check_tensor_sweep(sess: ort.InferenceSession, wrapper: torch.nn.Module,
                       lengths: list[int], tol: float) -> float:
    rng = np.random.default_rng(0)
    worst = 0.0
    for T in lengths:
        feed = fake_batch(T, rng)
        with torch.no_grad():
            ref = wrapper(*[torch.from_numpy(feed[k]) for k in INPUT_NAMES])
        got = sess.run(None, feed)
        errs = [float(np.abs(got[i] - ref[i].numpy()).max()) for i in range(4)]
        err = max(errs)
        worst = max(worst, err)
        print(f"  T={T:5d}  max|onnx-torch| = {err:.3e}  "
              f"(s {errs[0]:.1e}, p {errs[1]:.1e}, match_s {errs[2]:.1e}, match_p {errs[3]:.1e})"
              f"  {'OK' if err < tol else 'FAIL'}")
    scale_err = abs(float(sess.run(None, fake_batch(64, rng))[4]) - float(wrapper.m.scale.detach()))
    print(f"  scale output: |onnx-torch| = {scale_err:.3e}")
    return max(worst, scale_err)


def _logsumexp(x: np.ndarray) -> np.ndarray:
    mx = x.max(axis=-1, keepdims=True)
    return mx + np.log(np.exp(x - mx).sum(axis=-1, keepdims=True))


def _log_softmax(x: np.ndarray) -> np.ndarray:
    return x - _logsumexp(x)


def _logsigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def rebuild_logits_attr(got: dict[str, np.ndarray], n: int, m: int) -> np.ndarray:
    """The sidecar's `head.attribution`, written out in numpy — (m, n + 1).

    Deliberately built from the graph outputs and nothing else, the way the
    TypeScript host will have to: if this needed anything from `NoteAligner`,
    the sidecar would be incomplete. Which mode the checkpoint was trained with
    is read off the outputs the graph carries, again as a host would.

    The conditioned modes fold in the match head's own verdict, taken from the
    p->s logits the host already holds — here a single window, so those rows are
    the full ones and the conditioning applies directly.
    """
    attr_s = got["attr_s"][0][1:1 + n]              # (n, d)
    attr_p = got["attr_p"][0][2 + n:2 + n + m]      # (m, d)
    row = np.concatenate(
        [attr_p @ attr_s.T, (attr_p @ got["attr_none"])[:, None]], axis=1
    ) * float(got["attr_scale"])
    if "attr_cond_w" not in got and "attr_gate" not in got:
        return row

    s_vec = got["s"][0][1:1 + n]
    p_vec = got["p"][0][2 + n:2 + n + m]
    null_row = got["match_p"][0, 2 + n:2 + n + m]   # (m, 1), already scaled
    lp = _log_softmax(np.concatenate([p_vec @ s_vec.T * float(got["scale"]), null_row], axis=1))
    log_ins = np.maximum(lp[:, -1:], NoteAligner.LOG_FLOOR)
    log_matched = np.maximum(_logsumexp(lp[:, :-1]), NoteAligner.LOG_FLOOR)

    if "attr_cond_w" in got:
        return np.concatenate(
            [row[:, :-1], row[:, -1:] + float(got["attr_cond_w"]) * log_matched], axis=1)
    # "factored": the learned none column is unused, and the result is already
    # a log-distribution rather than logits.
    rank = row[:, :-1]
    gate = got["attr_gate"][0, 2 + n:2 + n + m]     # (m, 1)
    return np.concatenate(
        [log_ins + _logsigmoid(gate) + rank - _logsumexp(rank),
         np.logaddexp(log_ins + _logsigmoid(-gate), log_matched)], axis=1)


def check_attribution(sess: ort.InferenceSession, model, lengths: list[int], tol: float) -> float:
    """The sidecar's attribution formula against `NoteAligner.forward` itself.

    The tensor sweep above proves the graph emits the right vectors; this proves
    that the dot products the sidecar tells a host to take of them reconstruct
    the head. It is the same distinction as (1) versus (2) for the match head,
    and for the same reason: a host can hold correct vectors and still build the
    wrong matrix out of them.

    With a conditioned head that is the larger half of the claim: the row is no
    longer the attribution head alone but a combination of both heads, and the
    combination lives entirely in the sidecar prose.
    """
    names = [o.name for o in sess.get_outputs()]
    if "attr_s" not in names:
        print("  (no attribution head in this graph)")
        return 0.0
    mode = "factored" if "attr_gate" in names else ("bias" if "attr_cond_w" in names else "")
    print(f"  attr_conditioned={mode!r}"
          + (" — logits_attr rebuilt from BOTH heads" if mode else ""))

    rng = np.random.default_rng(1)
    worst = 0.0
    for T in lengths:
        feed = fake_batch(T, rng)
        n = (T - 2) // 2
        m = T - 2 - n

        with torch.no_grad():
            ref = model({
                **{k: torch.from_numpy(v) for k, v in feed.items()},
                "pad": torch.zeros((1, T), dtype=torch.bool),
                "n_score": torch.tensor([n]),
                "n_perf": torch.tensor([m]),
            })["logits_attr"][0].numpy()

        rebuilt = rebuild_logits_attr(dict(zip(names, sess.run(None, feed))), n, m)

        # Relative, like the accumulated-logit check and for the same reason:
        # this is a matrix of 192-term dot products, so its entries are two
        # orders of magnitude larger than the vectors they come from, and an
        # absolute tolerance calibrated for the vectors would reject a graph
        # whose every logit is right to four figures.
        scale = max(1.0, float(np.abs(ref).max()))
        adiff = float(np.abs(rebuilt - ref).max())
        rel = adiff / scale
        worst = max(worst, rel)
        print(f"  T={T:5d}  |ref|max {scale:7.2f}  max|host-torch| {adiff:.3e}  "
              f"relative {rel:.3e}  {'OK' if rel < tol else 'FAIL'}")

    return worst


def accumulate_logits_onnx(sess: ort.InferenceSession, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`mlign.infer.accumulate_logits` with the model replaced by the session.

    Deliberately a transcription, not a refactor: this is the reference the
    TypeScript host has to match, so it repeats the windowing and averaging
    verbatim instead of importing them.
    """
    n, m = len(row["score"]), len(row["perf"])
    pairs = ([(0, n, 0, m)] if 2 + n + m <= infer.MAX_SINGLE_TOKENS
             else infer.coarse_windows(row, n, m))

    sim = np.full((n, m), -1e9, dtype=np.float32)
    cnt = np.zeros((n, m), dtype=np.float32)
    null_s = np.zeros(n, dtype=np.float32)
    null_s_cnt = np.zeros(n, dtype=np.float32)
    null_p = np.zeros(m, dtype=np.float32)
    null_p_cnt = np.zeros(m, dtype=np.float32)

    for s0, s1, p0, p1 in pairs:
        sub = {"score": row["score"][s0:s1], "perf": row["perf"][p0:p1],
               "align": [], "subs": [], "ins": [], "del": []}
        f = featurize(sub)
        ns, mp = s1 - s0, p1 - p0
        feed = {"pitch": f["pitch"][None].astype(np.int64),
                "cont": f["cont"][None],
                "segment": f["segment"][None].astype(np.int64),
                "position": f["position"][None].astype(np.int64)}
        # Named, not positional: a graph carrying the attribution head returns
        # four more tensors, and this is the path that does not want them. It is
        # also how a host skips paying for them.
        s_tok, p_tok, match_s, match_p, scale = sess.run(
            ["s", "p", "match_s", "match_p", "scale"], feed
        )

        # NoteAligner.forward, B=1, no padding: score tokens live at [1, 1+n),
        # perf tokens at [2+n, 2+n+m). The null logits come straight from the
        # graph — matchability_s/p were applied to the encoder output, not to
        # the projections, and the graph preserves that.
        s_vec = s_tok[0, 1:1 + ns]
        p_vec = p_tok[0, 2 + ns:2 + ns + mp]
        block_sim = np.einsum("nd,md->nm", s_vec, p_vec) * scale
        null_col = match_s[0, 1:1 + ns, 0]
        null_row = match_p[0, 2 + ns:2 + ns + mp, 0]

        block = block_sim + block_sim  # ls2p[:ns,:mp] + lp2s[:mp,:ns].T
        region = sim[s0:s1, p0:p1]
        first = cnt[s0:s1, p0:p1] == 0
        region[first] = 0.0
        region += block
        sim[s0:s1, p0:p1] = region
        cnt[s0:s1, p0:p1] += 1.0

        null_s[s0:s1] += null_col
        null_s_cnt[s0:s1] += 1.0
        null_p[p0:p1] += null_row
        null_p_cnt[p0:p1] += 1.0

    cnt[cnt == 0] = 1.0
    sim = sim / cnt
    null_s = null_s / np.maximum(null_s_cnt, 1.0)
    null_p = null_p / np.maximum(null_p_cnt, 1.0)
    null_s[null_s_cnt == 0] = 1e9
    null_p[null_p_cnt == 0] = 1e9
    return sim, null_s, null_p


def align_with_onnx(sess: ort.InferenceSession, score: ScoreTable, perf: PerfTable) -> list[dict]:
    row = infer.tables_to_row(score, perf)
    sim, null_s, null_p = accumulate_logits_onnx(sess, row)
    out = []
    for t in infer.decode(row, sim, null_s, null_p):
        rec = {"label": t["label"], "confidence": t.get("confidence")}
        if t["label"] in ("match", "deletion"):
            rec["score_id"] = str(score.notes["id"][t["score_idx"]])
        if t["label"] in ("match", "insertion"):
            rec["perf_id"] = str(perf.notes["id"][t["perf_idx"]])
        out.append(rec)
    return out


def key(t: dict) -> tuple:
    return (t["label"], t.get("score_id"), t.get("perf_id"))


def check_end_to_end(sess: ort.InferenceSession, model, score: ScoreTable, perf: PerfTable,
                     tol: float) -> tuple[float, bool]:
    """Both halves of the real check, on one piece.

    (a) the accumulated (sim, null_s, null_p) — the host head math against
        `infer.accumulate_logits`. This is the sensitive half;
    (b) the decoded triples. Identity here is what a user would notice, but it
        is a *weak* signal: the decode leans hard on pitch equality and the
        onset-cluster DTW, so on an easy piece it survives grossly wrong
        logits. Never accept an ONNX port on (b) alone.
    """
    row = infer.tables_to_row(score, perf)
    n, m = len(row["score"]), len(row["perf"])
    n_windows = 1 if 2 + n + m <= infer.MAX_SINGLE_TOKENS else len(infer.coarse_windows(row, n, m))
    print(f"  {n} score notes, {m} perf notes, T = {2 + n + m}, {n_windows} window(s)")

    ref_logits = infer.accumulate_logits(model, row, "cpu")
    got_logits = accumulate_logits_onnx(sess, row)
    worst = 0.0
    for name, a, b in zip(("sim", "null_s", "null_p"), ref_logits, got_logits):
        # Cells no window covered carry the ±1e9 sentinels; they must land in
        # the same places on both sides, and they must not set the yardstick.
        covered = np.abs(a) < 1e8
        assert (covered == (np.abs(b) < 1e8)).all(), f"{name}: window coverage differs"
        scale = max(1.0, float(np.abs(a[covered]).max()))
        adiff = float(np.abs(a[covered] - b[covered]).max())
        rel = adiff / scale
        worst = max(worst, rel)
        print(f"  {name:<6} |ref|max {scale:7.2f} over {covered.sum()}/{covered.size} covered  "
              f"max|onnx-torch| {adiff:.3e}  relative {rel:.3e}  {'OK' if rel < tol else 'FAIL'}")

    ref = infer.align_with_model(model, score, perf)
    got = align_with_onnx(sess, score, perf)
    same = len(ref) == len(got) and all(key(a) == key(b) for a, b in zip(ref, got))
    counts = {lab: sum(1 for t in ref if t["label"] == lab) for lab in ("match", "deletion", "insertion")}
    print(f"  triples: {len(ref)} torch / {len(got)} onnx  {counts}")
    if not same:
        diff = [(a, b) for a, b in zip(ref, got) if key(a) != key(b)]
        print(f"  MISMATCH in {len(diff)} triple(s); first: torch={diff[0][0]} onnx={diff[0][1]}")
    else:
        conf_err = max(abs((a["confidence"] or 0.0) - (b["confidence"] or 0.0)) for a, b in zip(ref, got))
        print(f"  identical labels + ids; max |confidence| delta = {conf_err:.3e}")
    return worst, same


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--onnx", default="models/mlign-v2.onnx")
    ap.add_argument("--ckpt", default="models/mlign-v2.pt")
    ap.add_argument("--lengths", type=int, nargs="+", default=[300, 600, 1200, 2000])
    ap.add_argument("--score", default="web/demo/schubert_d783_15.musicxml")
    ap.add_argument("--perf", default="web/demo/schubert_d783_15_p01.mid")
    ap.add_argument("--force-win-score", type=int, default=128,
                    help="second end-to-end pass with the windowing thresholds lowered to this "
                         "many score notes per window, so overlap accumulation is exercised too")
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, _ = load_model(Path(args.ckpt))
    wrapper = (Enc if model.cfg.matchability else EncDustbin)(model).eval()
    sess = make_session(Path(args.onnx))

    sidecar = Path(args.onnx).with_suffix(Path(args.onnx).suffix + ".json")
    meta = json.loads(sidecar.read_text())
    assert meta["model"]["d_model"] == model.cfg.d_model, "sidecar d_model disagrees with the checkpoint"
    assert meta["model"]["matchability"] == model.cfg.matchability, "sidecar matchability disagrees"
    assert meta["model"].get("attribution", False) == model.cfg.attribution, "sidecar attribution disagrees"
    assert meta["model"].get("attr_conditioned", "") == model.cfg.attr_conditioned, \
        "sidecar attr_conditioned disagrees"
    assert abs(meta["head"]["scale"] - float(model.scale.detach())) < 1e-9, "sidecar scale disagrees"
    storage = meta["model"]["weight_storage"]
    tol = TOL[storage]
    print(f"sidecar {sidecar.name}: agrees with the checkpoint ({storage} weight storage)")

    print(f"\n[1] tensor sweep (tolerance {tol:g})")
    worst = check_tensor_sweep(sess, wrapper, args.lengths, tol)

    print(f"\n[1b] attribution head, as the sidecar tells a host to rebuild it")
    attr_worst = check_attribution(sess, model, args.lengths, tol)

    score = ScoreTable.from_musicxml(ROOT / args.score)
    perf = PerfTable.from_midi(ROOT / args.perf)
    print(f"\n[2] end-to-end — {Path(args.score).name} / {Path(args.perf).name}, single pass")
    logit_err, same = check_end_to_end(sess, model, score, perf, tol)

    # Same piece, windowing forced on: the overlap-averaged path has its own
    # slicing (perf sub-ranges, tokens counted twice) and is where a host port
    # is most likely to drift from the reference.
    print(f"\n[3] end-to-end — same piece, windowed (WIN_SCORE={args.force_win_score})")
    infer.MAX_SINGLE_TOKENS, infer.WIN_SCORE = 0, args.force_win_score
    logit_err_win, same_win = check_end_to_end(sess, model, score, perf, tol)

    logit_worst = max(logit_err, logit_err_win)
    worst = max(worst, attr_worst)
    ok = worst < tol and logit_worst < tol and same and same_win
    print(f"\n{'PASS' if ok else 'FAIL'}: worst tensor diff {worst:.3e}, worst logit diff "
          f"{logit_worst:.3e} (both < {tol:g}); triples {'identical' if same else 'DIFFER'} "
          f"single-pass / {'identical' if same_win else 'DIFFER'} windowed")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
