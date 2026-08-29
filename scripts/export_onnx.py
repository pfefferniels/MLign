"""Export the MLign encoder to ONNX for onnxruntime-web.

The browser port runs the transformer in ONNX Runtime Web and everything around
it in TypeScript. `NoteAligner.forward` cannot be exported as it stands: it
reads n/m off the batch with `.item()` and loops over the batch to slice the
score and perf spans out of the token sequence. Both bake data-dependent shapes
into the graph — the one thing WASM runtimes handle worst.

So the graph stops at the encoder plus the per-token heads and returns one row
per token. The (n, m) slicing, the score x perf matmul and the whole decode stay
in host code, where they cost nothing. What is left has exactly one dynamic axis
(sequence length T), no shape arithmetic and no control flow.

The sidecar JSON written next to the .onnx is the Python/TypeScript contract:
the featurization constants, the head math the host has to reproduce, and the
learned `scale`. `scripts/test_onnx_parity.py` is what proves it.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/export_onnx.py \
      --ckpt models/mlign-v3.pt --out models/mlign-v3.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign import infer  # noqa: E402
from mlign.dataset import MARKER_PITCH  # noqa: E402
from mlign.model import ModelConfig, NoteAligner, config_from_ckpt  # noqa: E402

INPUT_NAMES = ["pitch", "cont", "segment", "position"]
# The per-token ones come first, because they are the ones with a dynamic axis.
PER_TOKEN_OUTPUTS = ["s", "p", "match_s", "match_p"]
OUTPUT_NAMES = [*PER_TOKEN_OUTPUTS, "scale"]
# Appended, never inserted: a host written against the four-output graph keeps
# working, and one that wants attribution asks for the extra names.
ATTR_PER_TOKEN_OUTPUTS = ["attr_s", "attr_p"]
ATTR_OUTPUT_NAMES = [*ATTR_PER_TOKEN_OUTPUTS, "attr_none", "attr_scale"]
# The same rule one level down: a conditioned checkpoint adds its extra tensors
# after the whole attribution block, so a host written against the v2
# attribution contract reads the same names at the same positions. Which is
# also why `attr_gate` trails the non-dynamic `attr_none`/`attr_scale` instead
# of joining the per-token group at the front.
#
# `residual` extends `factored` the same way `factored` extended v2: it keeps
# `attr_gate` where it was and appends its own tensor after it, so a host that
# only knows `factored` reads a residual graph's gate correctly — which is
# exactly why a host must test for `attr_override` FIRST. `residual` is a
# superset, so asking about the gate first answers `factored` for both; both
# `conditioning_from_state` below and `test_onnx_parity.check_attribution`
# order their checks that way.
COND_OUTPUT_NAMES = {
    "bias": ["attr_cond_w"],
    "factored": ["attr_gate"],
    "residual": ["attr_gate", "attr_override"],
}
COND_PER_TOKEN_OUTPUTS = {
    "factored": ["attr_gate"],
    "residual": ["attr_gate", "attr_override"],
}


def _check_mode(cfg: ModelConfig) -> str:
    """Refuse a conditioning mode this exporter has no contract for.

    Silence here is the dangerous failure. Every lookup below is a `.get(mode,
    [])`, so an unrecognised mode exports a graph that is missing the tensors
    its own row is built from while the sidecar still announces the mode by
    name — a host would detect the absent output, fall back to the mode one
    step down, and read a row that means something else. Loudly refusing to
    export beats shipping a graph that is wrong only in its numbers.
    """
    mode = cfg.attr_conditioned
    if cfg.attribution and mode and mode not in COND_OUTPUT_NAMES:
        raise SystemExit(
            f"export_onnx: attr_conditioned={mode!r} has no export contract. "
            f"Known: {sorted(COND_OUTPUT_NAMES)}. Add its outputs to "
            "COND_OUTPUT_NAMES/COND_PER_TOKEN_OUTPUTS, its tensors to "
            "attribution_outputs(), its row to the sidecar's "
            "head.attribution.conditioned, and its branch to "
            "test_onnx_parity.rebuild_logits_attr — then re-run the parity check."
        )
    return mode


def output_names(cfg: ModelConfig) -> list[str]:
    if not cfg.attribution:
        return list(OUTPUT_NAMES)
    return OUTPUT_NAMES + ATTR_OUTPUT_NAMES + COND_OUTPUT_NAMES.get(_check_mode(cfg), [])


def per_token_outputs(cfg: ModelConfig) -> list[str]:
    if not cfg.attribution:
        return list(PER_TOKEN_OUTPUTS)
    return (PER_TOKEN_OUTPUTS + ATTR_PER_TOKEN_OUTPUTS
            + COND_PER_TOKEN_OUTPUTS.get(_check_mode(cfg), []))


def attribution_outputs(model: NoteAligner, x: torch.Tensor) -> tuple:
    """The attribution head, exported the way the match head is: the projected
    vectors, not the (m, n) matrix they multiply out to.

    Which is the whole reason it fits. The host already accumulates `sim` window
    by window from `s` and `p`; giving it `attr_s` and `attr_p` lets it do the
    identical thing for attribution with the identical bookkeeping. Emitting
    `logits_attr` instead would mean a graph that knows where the score half
    ends, and an (m, n) tensor per window on the wire.

    A conditioned head adds whatever it alone owns — the gate projection or the
    bias weight. The rest of the conditioning is a function of the match logits,
    which the host has already computed and the graph must not recompute: inside
    a window the p->s softmax runs over that window's score notes, not the whole
    score, so a graph-side log P(insertion) would be the wrong quantity.
    """
    if not model.cfg.attribution:
        return ()
    out = (model.attr_s(x), model.attr_p(x), model.attr_none, model.attr_scale)
    mode = _check_mode(model.cfg)
    if mode == "bias":
        return (*out, model.attr_cond_w)
    if mode == "factored":
        return (*out, model.attr_gate(x))
    if mode == "residual":
        # Both projections, raw. The override is a second per-token logit and
        # nothing else: like the gate, what it multiplies is the host's own
        # match evidence, so the graph can no more compute its effect than it
        # can compute `log_ins`.
        return (*out, model.attr_gate(x), model.attr_override(x))
    return out


class Enc(nn.Module):
    """B=1, unpadded. Per-token projected vectors + per-token null logits.

    Both projections are run over every token, score markers and perf tokens
    included; the host keeps the slice it needs. Running out_s over perf tokens
    is ~0.4% of the total flops and buys a graph with no slicing in it.
    """

    def __init__(self, model: NoteAligner):
        super().__init__()
        self.m = model

    def forward(self, pitch, cont, segment, position):
        pad = torch.zeros_like(pitch, dtype=torch.bool)
        x = self.m.encode(pitch, cont, segment, position, pad)
        return (
            self.m.out_s(x),
            self.m.out_p(x),
            self.m.matchability_s(x),
            self.m.matchability_p(x),
            self.m.scale,
            *attribution_outputs(self.m, x),
        )


class EncDustbin(nn.Module):
    """matchability=False variant: the null logits come from the dustbin dot
    products instead of the unary heads, so they need the *projected* vectors
    and the learned null vectors, which the host gets from the sidecar."""

    def __init__(self, model: NoteAligner):
        super().__init__()
        self.m = model

    def forward(self, pitch, cont, segment, position):
        pad = torch.zeros_like(pitch, dtype=torch.bool)
        x = self.m.encode(pitch, cont, segment, position, pad)
        s, p = self.m.out_s(x), self.m.out_p(x)
        null_col = (s @ self.m.null_p)[..., None] * self.m.scale
        null_row = (p @ self.m.null_s)[..., None] * self.m.scale
        return s, p, null_col, null_row, self.m.scale, *attribution_outputs(self.m, x)


def conditioning_from_state(state: dict) -> str:
    """Which `attr_conditioned` mode a state dict was trained with.

    For checkpoints whose saved config predates the flag. Each mode owns a
    parameter that is not optional within it, so the weights answer this
    outright — but `residual` is a superset of `factored` and carries a gate
    too, so it has to be asked about first or it would answer `factored` and
    export a graph missing the override."""
    if any(k.startswith("attr_override.") for k in state):
        return "residual"
    if any(k.startswith("attr_gate.") for k in state):
        return "factored"
    return "bias" if "attr_cond_w" in state else ""


def load_model(ckpt_path: Path) -> tuple[NoteAligner, dict]:  # (model, checkpoint)
    """Rebuild the model from the checkpoint's own config — never from CLI
    defaults, or a mismatched d_model loads silently into a wrong-shaped graph."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["config"]
    state = ckpt["model"]
    defaults = ModelConfig()
    cfg = config_from_ckpt(
        cfg_dict,
        # Four fields `config_from_ckpt` does not carry because no training flag
        # sets them; they still have to survive, or the graph is reshaped.
        n_heads=int(cfg_dict.get("n_heads", defaults.n_heads)),
        d_ff=int(cfg_dict.get("d_ff", defaults.d_ff)),
        dropout=float(cfg_dict.get("dropout", defaults.dropout)),
        max_rel=int(cfg_dict.get("max_rel", defaults.max_rel)),
        # Both head flags fall back to the weights when the saved config predates
        # them. `load_state_dict` below is strict, so a wrong guess is a loud
        # error rather than a silently half-exported head.
        attribution=bool(cfg_dict.get(
            "attribution", any(k.startswith("attr_") for k in state))),
        attr_conditioned=cfg_dict.get("attr_conditioned") or conditioning_from_state(state),
    )
    model = NoteAligner(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def compress_initializers_fp16(model: onnx.ModelProto, min_elems: int = 1024) -> tuple[int, int]:
    """Store the large float initializers as float16 plus a Cast back to float32.

    Halves the download without changing the compute graph's precision: ORT
    constant-folds the Cast at session load, so every op still runs in fp32.
    Small tensors — biases, LayerNorm gains, the attention scale — stay fp32;
    they are a rounding error of the file size and the place where rounding
    would actually cost accuracy.
    """
    graph = model.graph
    casts, n_elems = [], 0
    for init in list(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        arr = numpy_helper.to_array(init)
        if arr.size < min_elems:
            continue
        name16 = init.name + "_fp16"
        graph.initializer.remove(init)
        graph.initializer.append(numpy_helper.from_array(arr.astype(np.float16), name16))
        casts.append(
            helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name=f"Cast_{name16}")
        )
        n_elems += arr.size
    # Casts only consume initializers, so prepending keeps the graph topological.
    body = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + body)
    return len(casts), n_elems


def build_sidecar(model: NoteAligner, ckpt_path: Path, out_path: Path, opset: int, fp16: bool) -> dict:
    """The Python/TypeScript contract.

    Everything a host needs to (a) turn two note tables into the four input
    tensors and (b) turn the five outputs back into the logits that
    `NoteAligner.forward` would have produced. Nothing here is derivable from
    the .onnx file, and every constant is read from the source of truth
    (`mlign.dataset`, `mlign.infer`, the checkpoint) rather than retyped.
    """
    cfg = model.cfg
    return {
        "format": "mlign-onnx-sidecar",
        "version": 1,
        "model": {
            "file": out_path.name,
            "checkpoint": ckpt_path.name,
            "opset": opset,
            # fp16 here means *storage only*: initializers are float16 with a
            # Cast to float32 in front of every consumer. Compute is fp32
            # either way, and the runtime I/O dtypes never change.
            "weight_storage": "float16" if fp16 else "float32",
            "params": int(sum(p.numel() for p in model.parameters())),
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "max_rel": cfg.max_rel,
            "matchability": cfg.matchability,
            "attribution": cfg.attribution,
            "attr_conditioned": cfg.attr_conditioned,
        },
        "graph": {
            # Batch is pinned to 1 and padding is not modelled: the host runs
            # one window at a time and `pad` is all-false inside the graph.
            "batch": 1,
            "dynamic_axis": {"name": "T", "index": 1},
            "inputs": [
                {"name": "pitch", "dtype": "int64", "shape": [1, "T"],
                 "doc": "MIDI pitch per token; MARKER_PITCH for the two segment markers"},
                {"name": "cont", "dtype": "float32", "shape": [1, "T", NoteAligner.N_CONT],
                 "doc": "continuous features, see featurize.cont_channels"},
                {"name": "segment", "dtype": "int64", "shape": [1, "T"],
                 "doc": "0 for the score half (marker included), 1 for the perf half"},
                {"name": "position", "dtype": "int64", "shape": [1, "T"],
                 "doc": "index within the segment; restarts at 0 at the perf marker"},
            ],
            "outputs": [
                {"name": "s", "dtype": "float32", "shape": [1, "T", cfg.d_model],
                 "doc": "out_s(encoder output) per token"},
                {"name": "p", "dtype": "float32", "shape": [1, "T", cfg.d_model],
                 "doc": "out_p(encoder output) per token"},
                {"name": "match_s", "dtype": "float32", "shape": [1, "T", 1],
                 "doc": ("null (deletion) logit per token"
                         + (" — matchability_s applied to the PRE-projection encoder output"
                            if cfg.matchability
                            else " — dustbin dot product, already scaled"))},
                {"name": "match_p", "dtype": "float32", "shape": [1, "T", 1],
                 "doc": ("null (insertion) logit per token"
                         + (" — matchability_p applied to the PRE-projection encoder output"
                            if cfg.matchability
                            else " — dustbin dot product, already scaled"))},
                {"name": "scale", "dtype": "float32", "shape": [],
                 "doc": "the learned logit scale, identical to head.scale below"},
                *([
                    {"name": "attr_s", "dtype": "float32", "shape": [1, "T", cfg.d_model],
                     "doc": "attr_s(encoder output) per token — the attribution head's score side"},
                    {"name": "attr_p", "dtype": "float32", "shape": [1, "T", cfg.d_model],
                     "doc": "attr_p(encoder output) per token — its perf side"},
                    {"name": "attr_none", "dtype": "float32", "shape": [cfg.d_model],
                     "doc": "the learned 'not an ornament' vector, constant per model"},
                    {"name": "attr_scale", "dtype": "float32", "shape": [],
                     "doc": "the attribution head's own logit scale, separate from `scale`"},
                ] if cfg.attribution else []),
                *([
                    {"name": "attr_cond_w", "dtype": "float32", "shape": [],
                     "doc": "the learned weight on log P(matched) in the none column; "
                            "see head.attribution.conditioned"},
                ] if cfg.attribution and cfg.attr_conditioned == "bias" else []),
                *([
                    {"name": "attr_gate", "dtype": "float32", "shape": [1, "T", 1],
                     "doc": "attr_gate(encoder output) per token — the logit of "
                            "P(this insertion elaborates a written note); "
                            "see head.attribution.conditioned"},
                ] if cfg.attribution and cfg.attr_conditioned in ("factored", "residual") else []),
                *([
                    {"name": "attr_override", "dtype": "float32", "shape": [1, "T", 1],
                     "doc": "attr_override(encoder output) per token — the logit of the "
                            "share of P(matched) the attribution head is allowed to claim "
                            "back from the match head; see head.attribution.conditioned"},
                ] if cfg.attribution and cfg.attr_conditioned == "residual" else []),
            ],
        },
        "featurize": {
            # Mirrors mlign.dataset.featurize + mlign.infer.tables_to_row.
            "token_layout": "[MARKER] s_1 .. s_n [MARKER] p_1 .. p_m   (T = 2 + n + m)",
            "marker_pitch": MARKER_PITCH,
            "marker_cont": "all zeros",
            "n_cont": NoteAligner.N_CONT,
            "score_ppq": 720.0,      # score onsets/durations: seconds*ppq, then /ppq -> quarters
            "perf_ms_per_sec": 1000.0,  # perf onsets/durations: seconds*1000, then /1000 -> seconds
            "voice_mod": 5,          # tables_to_row: voice % 5 before the /4 below
            "segment": {"score_marker": 0, "score_notes": 0, "perf_marker": 1, "perf_notes": 1},
            "position": "arange(1 + n) for the score half, then arange(1 + m) for the perf half",
            "cont_channels": [
                {"index": 0, "name": "delta",
                 "expr": "log1p(max(onset - prev_onset, 0) * 2)",
                 "units": "score: quarters; perf: seconds",
                 "note": "the first note of each half has delta 0 (numpy diff prepends its own onset)"},
                {"index": 1, "name": "duration",
                 "expr": "log1p(max(duration, 0) * 2)",
                 "units": "score: quarters; perf: seconds"},
                {"index": 2, "name": "pitch_abs", "expr": "pitch / 64 - 1"},
                {"index": 3, "name": "pitch_class", "expr": "(pitch % 12) / 11 * 2 - 1"},
                {"index": 4, "name": "extra",
                 "expr": "score: (voice % 5) / 4   |   perf: velocity / 64 - 1"},
                {"index": 5, "name": "segment_flag", "expr": "0.0 for score notes, 1.0 for perf notes"},
            ],
        },
        "head": {
            # Reproduces NoteAligner.forward for B=1 with no padding. Indices
            # are into the token axis of the graph outputs.
            "scale": float(model.scale.detach()),
            "score_token_slice": "[1, 1 + n)",
            "perf_token_slice": "[2 + n, 2 + n + m)",
            "sim": "sim[i][j] = dot(s[1 + i], p[2 + n + j]) * scale        # (n, m)",
            "null_col": ("null_col[i] = match_s[1 + i]                       # (n,) deletion logit"
                         if cfg.matchability else
                         "null_col[i] = match_s[1 + i]   # already dot(s, null_p) * scale"),
            "null_row": ("null_row[j] = match_p[2 + n + j]                   # (m,) insertion logit"
                         if cfg.matchability else
                         "null_row[j] = match_p[2 + n + j]   # already dot(p, null_s) * scale"),
            "logits_s2p": "concat([sim, null_col[:, None]], axis=1)          # (n, m + 1)",
            "logits_p2s": "concat([sim.T, null_row[:, None]], axis=1)        # (m, n + 1)",
            "matchability_applies_to": (
                "the encoder output BEFORE out_s/out_p — the graph already does this, "
                "so the host must NOT re-derive the null logits from s/p"
                if cfg.matchability else "n/a"
            ),
            # Only needed when the checkpoint predates the matchability head;
            # with matchability=True the graph emits the null logits directly.
            "null_s": None if cfg.matchability else [float(v) for v in model.null_s.detach()],
            "null_p": None if cfg.matchability else [float(v) for v in model.null_p.detach()],
            # A second p->s distribution, answering a different question from
            # `sim`: not "which written note is this played note" but "which
            # written note does it ornament". Deliberately its own bilinear map -
            # the match head is trained to send an ornament note to the null
            # column, so the same score cannot rank both.
            "attribution": None if not cfg.attribution else {
                "attr_scale": float(model.attr_scale.detach()),
                "attr": "attr[j][i] = dot(attr_p[2 + n + j], attr_s[1 + i]) * attr_scale   # (m, n)",
                "attr_none_col": "none[j] = dot(attr_p[2 + n + j], attr_none) * attr_scale  # (m,)",
                "logits_attr": ("concat([attr, none[:, None]], axis=1)             # (m, n + 1)"
                                if not cfg.attr_conditioned else
                                "concat([attr, none[:, None]], axis=1)  — but NOT the final row: "
                                "`conditioned` below replaces it"),
                "read_as": ("argmax over each row: the written note this played note ornaments, "
                            "or the last column for 'not an ornament'. Softmax over the row gives "
                            "a usable confidence."
                            + ("" if cfg.attr_conditioned not in ("factored", "residual") else
                               f" — except in {cfg.attr_conditioned!r} mode, where the row is "
                               "already normalized; see conditioned.already_normalized")),
                "accumulate": ("exactly as `sim` is accumulated, and with the same window "
                               "bookkeeping: attr += the window's block, then divide by the "
                               "window count. The none column averages the same way."
                               + ("" if not cfg.attr_conditioned else
                                  " The conditioning is applied once, after that averaging; "
                                  "see conditioned.when.")),
                "supervision": ("trained only on espressivo-rendered rows, where ornament "
                                "provenance is exhaustive. It has never been asked about a real "
                                "trill with a known answer, because no corpus annotates one."),
                "optional": ("the two per-token attribution outputs double the bytes a window "
                             "returns. A host that does not want them can name only the outputs "
                             "it needs when it runs the session."),
                # Where the head stops answering "is this an ornament at all" for
                # itself and takes the match head's answer instead. That half was
                # only ever supervised on synthetic rows; the match head learned it
                # from real insertion labels.
                "conditioned": None if not cfg.attr_conditioned else {
                    "mode": cfg.attr_conditioned,
                    "log_floor": NoteAligner.LOG_FLOOR,
                    "match_evidence": (
                        "from the host's own p->s logits, but NOT the accumulated matrix "
                        "it feeds the decode — that one holds 2x the raw sim (see "
                        "host.accumulate), while training saw the UNDOUBLED logits_p2s "
                        "this file's `head` block defines. Halve the sim half and leave "
                        "null_row alone, which is already single because it appears in "
                        "only one of the two accumulated directions:\n"
                        "  logits_p2s[j] = concat([accumulated_sim.T[j] / 2, null_row[j]])\n"
                        "Getting this wrong is not a rounding error: doubling sharpens the "
                        "log_softmax, and on a real flourish it moved per-note attribution "
                        "confidence from .987 to .112 — under any sane acceptance "
                        "threshold, a dropped ornament.\n"
                        "The graph emits nothing extra for this, and must not: inside a "
                        "window the p->s softmax runs over that window's score notes only.\n"
                        "  lp             = log_softmax(logits_p2s, axis=1)   # (m, n + 1)\n"
                        "  log_ins[j]     = max(lp[j][n], log_floor)\n"
                        "  log_matched[j] = max(logsumexp(lp[j][0:n]), log_floor)\n"
                        "The floor is NoteAligner.LOG_FLOOR — an unclamped certain match "
                        "head puts an unbounded term into the row."),
                    "when": (
                        "ONCE, after window accumulation, on the full rows: accumulate "
                        "attr / none"
                        + (" / gate" if cfg.attr_conditioned == "factored" else "")
                        + (" / gate / override" if cfg.attr_conditioned == "residual" else "")
                        + " exactly as above, then apply this to the averaged result. It is a "
                        "nonlinear function of the whole p->s row, so averaging conditioned "
                        "windows is not the same thing."),
                    "detached": ("the match term carries no gradient in training either — the "
                                 "attribution loss must stay unable to move the alignment. It "
                                 "changes nothing for a host, but it is why the two heads can "
                                 "be reproduced independently."),
                    **({
                        "attr_cond_w": float(model.attr_cond_w.detach()),
                        "logits_attr": (
                            "concat([attr, (none + attr_cond_w * log_matched)[:, None]], axis=1)"
                            "   # (m, n + 1)\n"
                            "i.e. the unconditioned row with one term added to its last column. "
                            "Still unnormalized logits: softmax the row for a confidence."),
                    } if cfg.attr_conditioned == "bias" else {
                        "gate": "gate[j] = attr_gate[2 + n + j][0]   # (m,), read at the PERF tokens",
                        **({} if cfg.attr_conditioned != "residual" else {
                            "override": ("over[j] = attr_override[2 + n + j][0]   # (m,), read at "
                                         "the PERF tokens"),
                        }),
                        "logits_attr": (
                            "the whole row is rebuilt, and `none` (the attr_none dot product) is "
                            "NOT part of it — attr_none stays in the output list only so the "
                            "attribution block keeps its shape:\n"
                            "  log_rank[j][i] = attr[j][i] - logsumexp(attr[j][0:n])   # (m, n)\n"
                            + ("  ins[j], rest[j] = log_ins[j], log_matched[j]\n"
                               if cfg.attr_conditioned == "factored" else
                               "  ins[j]  = logaddexp(log_ins[j], log_matched[j] + "
                               "logsigmoid(over[j]))\n"
                               "  rest[j] = log_matched[j] + logsigmoid(-over[j])\n")
                            + "  row[j][i] = ins[j] + logsigmoid(gate[j]) + log_rank[j][i]\n"
                              "  row[j][n] = logaddexp(ins[j] + logsigmoid(-gate[j]), rest[j])\n"
                              "with logsigmoid(x) = -log1p(exp(-x)) and "
                              "logaddexp(a, b) = max(a, b) + log1p(exp(-|a - b|)).\n"
                              "Reads as P(anchor = i) = P(insertion) * P(attributable | insertion) "
                              "* P(i | attributable); the none column collects both ways of not "
                              "being an ornament — matched, or an unattributable insertion."
                            + ("" if cfg.attr_conditioned == "factored" else
                               "\n`residual` differs in `ins` alone: a learned share of the "
                               "MATCHED mass moves into the insertion branch, so attribution can "
                               "still name an ornament where the match head believes the note "
                               "matched something — the one case `factored` structurally cannot "
                               "reach. What is left of that mass stays in the none column, so the "
                               "row remains a distribution: ins + rest = P(ins) + P(matched) = 1. "
                               "attr_override is initialised at -4, which makes an untrained "
                               "residual head very nearly `factored`.")),
                        "already_normalized": (
                            "the row IS a log-distribution over the n + 1 options: exp() it for a "
                            "confidence, do NOT softmax it again. Softmaxing would discard exactly "
                            "the match-head calibration this mode buys."),
                    }),
                },
            },
        },
        "host": {
            # mlign.infer constants the TypeScript decode has to match.
            "max_single_tokens": infer.MAX_SINGLE_TOKENS,
            "win_score": infer.WIN_SCORE,
            "margin_sec": infer.MARGIN_SEC,
            "window_stride": infer.WIN_SCORE // 2,
            "accumulate": ("per window: sim += logits_s2p[:n, :m] + logits_p2s[:m, :n].T "
                           "(so every covered cell holds 2x the raw sim), then divide by the "
                           "window count; the null logits are averaged the same way"),
            "uncovered_sim": -1e9,   # (i, j) cells no window reached keep the init value
            "uncovered_null": 1e9,   # notes no window reached are forced "unmatched"
            "decode_constants": ("not duplicated here on purpose — they belong to the decode "
                                 "(mlign.infer.decode), not to the model contract"),
        },
    }


def summarize(onnx_model: onnx.ModelProto, path: Path) -> None:
    ops = Counter(n.op_type for n in onnx_model.graph.node)
    init_bytes = sum(len(i.raw_data) for i in onnx_model.graph.initializer)
    print(f"file: {path}  {path.stat().st_size / 1e6:.2f} MB "
          f"({len(onnx_model.graph.initializer)} initializers, {init_bytes / 1e6:.2f} MB weights)")
    print(f"ops ({len(ops)} distinct, {sum(ops.values())} nodes): "
          + ", ".join(f"{k}x{v}" for k, v in sorted(ops.items())))
    for kind, vals in (("in", onnx_model.graph.input), ("out", onnx_model.graph.output)):
        for v in vals:
            dims = [d.dim_param or str(d.dim_value) for d in v.type.tensor_type.shape.dim]
            print(f"  {kind:>3} {v.name:<9} {onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type):<7} "
                  f"[{', '.join(dims) or 'scalar'}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default="models/mlign-v3.pt")
    ap.add_argument("--out", default="models/mlign-v3.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--fp16", action="store_true",
                    help="store large weights as float16 + Cast (halves the download; compute stays fp32)")
    ap.add_argument("--example-t", type=int, default=600, help="trace length; the T axis stays dynamic")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(ckpt_path)
    wrapper = (Enc if model.cfg.matchability else EncDustbin)(model).eval()
    print(f"checkpoint: {ckpt_path}  epoch={ckpt.get('epoch', '?')} "
          f"d_model={model.cfg.d_model} n_layers={model.cfg.n_layers} "
          f"matchability={model.cfg.matchability} "
          f"attribution={model.cfg.attribution}"
          f"{f'/{model.cfg.attr_conditioned}' if model.cfg.attr_conditioned else ''} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    # A trace input with the real token layout: markers, split segments and
    # per-segment positions. Only T is dynamic, so the values do not matter for
    # correctness — but a realistic one keeps the traced shapes honest.
    T = args.example_t
    n = (T - 2) // 2
    m = T - 2 - n
    example = (
        torch.tensor([[MARKER_PITCH, *np.random.randint(0, 128, n), MARKER_PITCH,
                       *np.random.randint(0, 128, m)]], dtype=torch.long),
        torch.randn(1, T, NoteAligner.N_CONT),
        torch.tensor([[0] * (1 + n) + [1] * (1 + m)], dtype=torch.long),
        torch.tensor([[*range(1 + n), *range(1 + m)]], dtype=torch.long),
    )

    # dynamo=False on purpose: the TorchScript exporter is what produces the
    # 22-op, all-standard graph below. The dynamo path emits ops (and a shape
    # story) that onnxruntime-web is fussier about.
    t0 = time.time()
    torch.onnx.export(
        wrapper, example, str(out_path), opset_version=args.opset, dynamo=False,
        input_names=INPUT_NAMES, output_names=output_names(model.cfg),
        dynamic_axes={k: {1: "T"} for k in INPUT_NAMES + per_token_outputs(model.cfg)},
    )
    print(f"exported in {time.time() - t0:.1f}s")

    onnx_model = onnx.load(str(out_path))
    if args.fp16:
        n_casts, n_elems = compress_initializers_fp16(onnx_model)
        onnx.save(onnx_model, str(out_path))
        print(f"fp16 weight storage: {n_casts} initializers, {n_elems:,} values")
    onnx.checker.check_model(onnx_model, full_check=True)

    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    sidecar_path.write_text(json.dumps(build_sidecar(model, ckpt_path, out_path, args.opset, args.fp16), indent=2) + "\n")

    summarize(onnx_model, out_path)
    print(f"sidecar: {sidecar_path}  {sidecar_path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
