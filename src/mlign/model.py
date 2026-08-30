"""MLign note-matching model v0.

One bidirectional transformer over the concatenation
    [SCORE] s_1 … s_n [PERF] p_1 … p_m
with segment embeddings, producing contextual note vectors; a bilinear match
head scores every (score, perf) pair, and learned null vectors give every
score note a "deleted" option and every perf note an "inserted" option.

Loss (symmetric cross-entropy): each score note classifies over
{perf notes ∪ null}, each perf note over {score notes ∪ null}. Insertions'
targets are the null column; deletions' the null row.

Sized for M1/8GB training: default d=192, 4 layers, 4 heads ≈ 4M params.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 192
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_rel: int = 64  # relative-position bucket clip (per segment)
    # LightGlue-style per-note matchability (sigmoid unary head) replacing the
    # dustbin-vector nulls: null logit = logit(1 - σ_i) directly per note.
    # (research/01 §5.1: "unmatchable" is a property of the note, not a global.)
    matchability: bool = False
    # Ornament attribution: a SECOND p→s distribution answering "which score
    # note does this played note ornament, if any". Deliberately a separate
    # bilinear map rather than a reuse of `sim`: the match head is trained to
    # send ornament notes to the null column, so the same score cannot also
    # rank their principal highly. Keeping it separate means the head is
    # strictly additive — the alignment metrics cannot regress from adding it.
    attribution: bool = False
    attr_weight: float = 0.2
    # Conditioning the attribution head on the match head's insertion decision.
    # The head splits into two questions that behave completely differently:
    # WHICH written note a played note elaborates transfers to real recordings;
    # WHETHER it is an ornament at all does not. But "whether" is largely a
    # question the match head has already answered — and answered from REAL
    # data, since realgt rows carry true insertion labels while attribution is
    # supervised on synthetic rows alone. So hand that half over instead of
    # making attribution re-derive it.
    #   ""         — off (v2 behaviour: an independent, self-taught none column)
    #   "bias"     — none column += w · log P(matched); minimal, strictly additive
    #   "factored" — P(anchor) = P(ins) · P(attributable | ins) · P(anchor | ins)
    #   "residual" — factored, plus a learned per-note override that lets
    #                attribution disagree with the match head. Measured on real
    #                recordings, plain "factored" recovers 0 of the 525 ornament
    #                notes the match head misjudges as matched — a structural
    #                veto worth ~12.7 % of all ornament notes — while the
    #                unconditioned head recovers 4.9 % of them. This keeps
    #                factored's win where the match head is right (84.5 % vs
    #                72.2 %) and reopens the door where it is wrong.
    #                MEASURED (v10res): it does not work. The override trains
    #                but converges to sigmoid ~.0008 on the vetoed notes, BELOW
    #                its value where it is inert, because raising it costs the
    #                ~100x more numerous matched notes. Kept for reproducibility.
    #   "calibrated" — factored, with the GATE priced by the head's own ranking
    #                margin, and NO override at all. The ablation of "evidenced":
    #                v12both showed the override fires LESS than in v11evid
    #                (crossover 11.24 vs 8.89) while attribution improves, and
    #                vetoed recovery stayed .0000 for a fifth model — so the
    #                margin is doing CALIBRATION, not overriding. If that is the
    #                whole effect this reproduces it with one parameter fewer,
    #                one ONNX output fewer, and no third mode in the host.
    #   "evidenced" — residual, with the override additionally priced by the
    #                head's OWN ranking margin (top1 - top2 over score notes,
    #                detached). Overriding is then cheap only where the head is
    #                sure WHICH note is ornamented and stays at zero where the
    #                ranking is flat, which is what makes the majority class
    #                stop paying for the parameter.
    # The match-head term is always DETACHED: alignment must stay unable to
    # feel the attribution loss, which is what kept the head strictly additive.
    attr_conditioned: str = ""


def config_from_ckpt(cfg: dict | None, state: dict | None = None, **overrides) -> ModelConfig:
    """ModelConfig as a checkpoint describes itself.

    `cfg` is the checkpoint's training args (`config`, i.e. `vars(args)`);
    `state` is its `model` state dict. Heads are read from the args when they
    are recorded there and inferred from the WEIGHTS when they are not — an
    older checkpoint predates the flag, and the weights cannot be wrong about
    which heads exist.

    Every loader used to spell this out for itself, and every one of them was
    missing at least one field: `mlign align`, `mlign serve` and four eval
    scripts could not load `models/mlign-v2.pt` at all, because they built a
    model with no attribution head and then loaded strictly into it. That is
    the failure this exists to make impossible — pass the state dict and the
    call site cannot fall behind a new head again.
    """
    cfg = cfg or {}
    keys = tuple(state or ())
    has = lambda p: any(k.startswith(p) for k in keys)
    attribution = cfg.get("attribution")
    if attribution is None:
        attribution = has("attr_")
    conditioned = cfg.get("attr_conditioned")
    if not conditioned:
        # Most specific first: `evidenced` is `residual` plus one scalar, and
        # `residual` is `factored` plus the override, so asking in the other
        # order answers with the mode one step down and loads strictly into a
        # model missing a parameter.
        conditioned = ("calibrated" if "attr_gate_margin" in keys
                       else "evidenced" if "attr_override_margin" in keys
                       else "residual" if has("attr_override")
                       else "factored" if has("attr_gate")
                       else "bias" if "attr_cond_w" in keys else "")
    fields = {
        "d_model": int(cfg.get("d_model", 192)),
        "n_layers": int(cfg.get("n_layers", 4)),
        "matchability": bool(cfg.get("matchability", has("matchability_"))),
        "attribution": bool(attribution),
        "attr_weight": float(cfg.get("attr_weight", 0.2)),
        "attr_conditioned": conditioned,
    }
    return ModelConfig(**(fields | overrides))


def _logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """log(e^a + e^b), spelled out rather than torch.logaddexp so the ONNX
    exporter has only ops it knows. Both arguments are clamped above -inf by
    their callers, so `a - b` is never nan."""
    m = torch.maximum(a, b)
    return m + torch.log1p(torch.exp(-(a - b).abs()))


# How far the top-ranked score note leads the runner-up, in log space: the
# attribution head's own confidence about WHICH note is ornamented, as distinct
# from whether anything is. Bounded on both sides so it can be a model input.
RANK_MARGIN_MAX = 20.0


def _rank_margin(log_rank: torch.Tensor) -> torch.Tensor:
    """(B, m, n) log-distribution -> (B, m, 1) top1 - top2, detached.

    Detached on purpose: `evidenced` lets the override READ the ranking as
    evidence, and a gradient path back would let it sharpen the ranking to buy
    itself room — the same reason every match-head term in this file is
    detached.

    Padded score columns arrive as -inf. With fewer than two real candidates the
    runner-up is -inf and the margin would be +inf, so it is clamped; a row with
    one candidate is "maximally confident" by construction and the clamp says
    exactly that without producing a non-finite input.
    """
    lr = log_rank.detach()
    if lr.shape[-1] < 2:
        return torch.full_like(lr[..., :1], RANK_MARGIN_MAX)
    top2 = lr.topk(2, dim=-1).values
    margin = top2[..., 0:1] - top2[..., 1:2]
    # nan_to_num before clamp: (-inf) - (-inf) is nan, which clamp propagates.
    return torch.nan_to_num(margin, nan=RANK_MARGIN_MAX, posinf=RANK_MARGIN_MAX).clamp(
        min=0.0, max=RANK_MARGIN_MAX
    )


class RelPosBias(nn.Module):
    """T5-style bucketed relative position bias, shared across layers."""

    def __init__(self, n_heads: int, max_rel: int):
        super().__init__()
        self.max_rel = max_rel
        self.bias = nn.Embedding(2 * max_rel + 1, n_heads)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (B, T) integer positions WITHIN segment; cross-segment
        # pairs get the clipped extreme, which the model learns to treat as
        # "no positional relation".
        rel = positions[:, None, :] - positions[:, :, None]
        rel = rel.clamp(-self.max_rel, self.max_rel) + self.max_rel
        return self.bias(rel).permute(0, 3, 1, 2)  # (B, H, T, T)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.n_heads = cfg.n_heads

    def forward(self, x: torch.Tensor, bias: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q = q.view(B, T, H, D // H).transpose(1, 2)
        k = k.view(B, T, H, D // H).transpose(1, 2)
        v = v.view(B, T, H, D // H).transpose(1, 2)
        mask = bias.masked_fill(pad[:, None, None, :], float("-inf"))
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.proj(att.transpose(1, 2).reshape(B, T, D)))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class NoteAligner(nn.Module):
    # Continuous feature layout (see dataset.featurize): score and perf notes
    # share the vector shape; unused slots are zero for the other segment.
    N_CONT = 6

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ModelConfig()
        self.pitch_emb = nn.Embedding(129, cfg.d_model)  # 128 = segment marker token
        self.segment_emb = nn.Embedding(2, cfg.d_model)
        self.cont_proj = nn.Sequential(
            nn.Linear(self.N_CONT, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.rel_bias = RelPosBias(cfg.n_heads, cfg.max_rel)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_s = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_p = nn.Linear(cfg.d_model, cfg.d_model)
        self.null_s = nn.Parameter(torch.randn(cfg.d_model) * 0.02)  # "deleted"
        self.null_p = nn.Parameter(torch.randn(cfg.d_model) * 0.02)  # "inserted"
        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(cfg.d_model)))
        if cfg.matchability:
            self.matchability_s = nn.Linear(cfg.d_model, 1)
            self.matchability_p = nn.Linear(cfg.d_model, 1)
        if cfg.attribution:
            self.attr_s = nn.Linear(cfg.d_model, cfg.d_model)
            self.attr_p = nn.Linear(cfg.d_model, cfg.d_model)
            self.attr_none = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
            # own temperature, so attribution gradients never retune the
            # match head's scale
            self.attr_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(cfg.d_model)))
            if cfg.attr_conditioned == "bias":
                self.attr_cond_w = nn.Parameter(torch.tensor(1.0))
            elif cfg.attr_conditioned in ("factored", "residual", "evidenced", "calibrated"):
                # P(attributable | insertion): not every insertion elaborates a
                # written note — slips and repeat restarts do not — so the
                # factorization needs this third factor to stay honest.
                self.attr_gate = nn.Linear(cfg.d_model, 1)
                if cfg.attr_conditioned in ("residual", "evidenced"):
                    # P(this is an ornament anyway | the match head says matched).
                    # Initialised strongly negative so the run starts as plain
                    # `factored` and has to earn every override.
                    self.attr_override = nn.Linear(cfg.d_model, 1)
                    nn.init.zeros_(self.attr_override.weight)
                    nn.init.constant_(self.attr_override.bias, -4.0)
                if cfg.attr_conditioned == "calibrated":
                    # Same scalar, same detached margin — but it prices the GATE,
                    # P(attributable | insertion), instead of an override. Zero
                    # init, so the run starts as bit-exact `factored`.
                    self.attr_gate_margin = nn.Parameter(torch.tensor(0.0))
                if cfg.attr_conditioned == "evidenced":
                    # One scalar: how much the head's OWN ranking margin is
                    # allowed to buy an override. Kept as a separate parameter
                    # rather than an extra input column so the whole term stays
                    # exportable — the host cannot recompute `attr_override`
                    # from a graph output if the margin is baked into it, but
                    # it can add `w · margin` itself. See export_onnx.
                    self.attr_override_margin = nn.Parameter(torch.tensor(0.0))
            elif cfg.attr_conditioned:
                raise ValueError(f"unknown attr_conditioned mode: {cfg.attr_conditioned!r}")

    def encode(self, pitch, cont, segment, position, pad):
        x = self.pitch_emb(pitch) + self.segment_emb(segment) + self.cont_proj(cont)
        bias = self.rel_bias(position)
        for block in self.blocks:
            x = block(x, bias, pad)
        return self.ln_f(x)

    def forward(self, batch: dict) -> dict:
        """batch tensors — see dataset.collate. Returns logits.

        logits_s2p: (B, n_max, m_max+1)  last column = null (deletion)
        logits_p2s: (B, m_max, n_max+1)  last column = null (insertion)
        """
        x = self.encode(batch["pitch"], batch["cont"], batch["segment"], batch["position"], batch["pad"])

        B = x.shape[0]
        n_max = batch["n_score"].max().item()
        m_max = batch["n_perf"].max().item()
        d = x.shape[-1]

        s = x.new_zeros((B, n_max, d))
        p = x.new_zeros((B, m_max, d))
        s_pad = torch.ones((B, n_max), dtype=torch.bool, device=x.device)
        p_pad = torch.ones((B, m_max), dtype=torch.bool, device=x.device)
        for b in range(B):
            n = int(batch["n_score"][b])
            m = int(batch["n_perf"][b])
            # layout per sample: [S-marker] n score notes [P-marker] m perf notes
            s[b, :n] = x[b, 1 : 1 + n]
            p[b, :m] = x[b, 2 + n : 2 + n + m]
            s_pad[b, :n] = False
            p_pad[b, :m] = False

        s_enc, p_enc = s, p
        s = self.out_s(s)
        p = self.out_p(p)
        sim = torch.einsum("bnd,bmd->bnm", s, p) * self.scale

        if self.cfg.matchability:
            # null logit = raw unary; the softmax then weighs it against the
            # pairwise sims — equivalent in effect to LightGlue's factorized
            # log(1-σ) without needing a separate normalization.
            null_col = self.matchability_s(s_enc)
            null_row = self.matchability_p(p_enc)
        else:
            null_col = torch.einsum("bnd,d->bn", s, self.null_p)[:, :, None] * self.scale
            null_row = torch.einsum("bmd,d->bm", p, self.null_s)[:, :, None] * self.scale

        logits_s2p = torch.cat([sim, null_col], dim=2)
        logits_s2p = logits_s2p.masked_fill(
            torch.cat([p_pad, torch.zeros_like(p_pad[:, :1])], dim=1)[:, None, :], float("-inf")
        )
        logits_p2s = torch.cat([sim.transpose(1, 2), null_row], dim=2)
        s_pad_col = torch.cat([s_pad, torch.zeros_like(s_pad[:, :1])], dim=1)[:, None, :]
        logits_p2s = logits_p2s.masked_fill(s_pad_col, float("-inf"))

        out = {"logits_s2p": logits_s2p, "logits_p2s": logits_p2s, "s_pad": s_pad, "p_pad": p_pad}

        if self.cfg.attribution:
            # (B, m, n+1); last column = "not an ornament".
            a_p = self.attr_p(p_enc)
            attr = torch.einsum("bmd,bnd->bmn", a_p, self.attr_s(s_enc))
            none_col = torch.einsum("bmd,d->bm", a_p, self.attr_none)[:, :, None]
            logits_attr = torch.cat([attr, none_col], dim=2) * self.attr_scale
            logits_attr = logits_attr.masked_fill(s_pad_col, float("-inf"))
            if self.cfg.attr_conditioned:
                logits_attr = self._condition_attr(logits_attr, logits_p2s, p_enc)
                logits_attr = logits_attr.masked_fill(s_pad_col, float("-inf"))
            out["logits_attr"] = logits_attr
            gate = self._attr_gate(logits_attr, p_enc)
            if gate is not None:
                out["attr_gate_logit"] = gate

        return out

    # The match head's own verdict on this played note, as two log-probabilities
    # that sum to one. Clamped: a match head that is *certain* would otherwise
    # put an infinite loss on a mislabelled row and blow the run up.
    LOG_FLOOR = -12.0

    def _match_evidence(self, logits_p2s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lp = torch.log_softmax(logits_p2s.detach(), dim=-1)
        log_ins = lp[..., -1:].clamp(min=self.LOG_FLOOR)
        log_matched = torch.logsumexp(lp[..., :-1], dim=-1, keepdim=True).clamp(min=self.LOG_FLOOR)
        return log_ins, log_matched

    def _attr_gate(self, logits_attr, p_enc) -> torch.Tensor | None:
        """The gate LOGIT: P(elaborates a written note | insertion) before sigmoid.

        Emitted un-squashed on purpose. A decoder averages it over the windows
        covering a played note, and averaging logits then squashing is both what
        every other logit here gets and what the browser host does. Averaging
        `logsigmoid` instead takes a geometric mean of the probabilities, which
        is systematically smaller by concavity and lands straight on a 0.5
        threshold.

        Exported because a decoder needs it and cannot reliably rebuild it.
        `logsumexp(score columns) - log_ins` recovers it exactly under
        `factored`, but `residual` and `evidenced` feed the gate an
        override-augmented `log_ins`, so that identity over-reads them — on
        v12both by +0.32 nats on average and up to +3.3 on exactly the vetoed
        notes, flipping 13.8 % of threshold decisions. Emitting it removes the
        guesswork, costs no parameter, and needs no retraining: every trained
        checkpoint already computes this tensor.
        """
        if self.cfg.attr_conditioned in ("factored", "residual", "evidenced", "calibrated"):
            gate = self.attr_gate(p_enc)
            if self.cfg.attr_conditioned == "calibrated":
                rank = logits_attr[..., :-1]
                gate = gate + self.attr_gate_margin * _rank_margin(
                    rank - torch.logsumexp(rank, dim=-1, keepdim=True))
            return gate.squeeze(-1)
        return None

    def _condition_attr(self, logits_attr, logits_p2s, p_enc) -> torch.Tensor:
        log_ins, log_matched = self._match_evidence(logits_p2s)
        if self.cfg.attr_conditioned == "bias":
            return torch.cat(
                [logits_attr[..., :-1], logits_attr[..., -1:] + self.attr_cond_w * log_matched],
                dim=-1,
            )
        # "factored": the head keeps only the question it is good at — the
        # ranking over score notes — while "is this an ornament at all" becomes
        # P(insertion) (match head, real-data-calibrated) times a learned
        # P(attributable | insertion). With log_ins detached, a row whose target
        # is "none" contributes no gradient to the ranking at all.
        rank = logits_attr[..., :-1]
        log_rank = rank - torch.logsumexp(rank, dim=-1, keepdim=True)
        gate = self.attr_gate(p_enc)
        if self.cfg.attr_conditioned == "calibrated":
            gate = gate + self.attr_gate_margin * _rank_margin(log_rank)
        log_rest = log_matched
        if self.cfg.attr_conditioned in ("residual", "evidenced"):
            # Move a learned share of the MATCHED mass back into play, so a
            # played note can be an ornament even where the match head thinks it
            # matched something. Stays a proper distribution: the two branches
            # below still sum to P(ins) + P(matched) = 1.
            over_logit = self.attr_override(p_enc)
            if self.cfg.attr_conditioned == "evidenced":
                # `residual` priced itself out. Measured on v10res, its override
                # reached sigmoid .0008 on the very notes it exists for, an
                # order of magnitude BELOW its value on notes where it is inert
                # — because raising it costs the majority class. For a genuinely
                # matched note the target IS the none column, and lifting the
                # override moves mass out of `log_rest` (all of which lands
                # there) into `log_ins` (only `sigmoid(-gate)` of which returns).
                # Matched notes outnumber vetoed ornament notes ~100:1, so that
                # penalty wins everywhere and the override never fires.
                #
                # So make the bid conditional on the head's OWN evidence, which
                # is what a residual path was supposed to be: `margin` is how
                # far the top-ranked score note leads the runner-up, in log
                # space. It is large only where the head is sure WHICH note is
                # being ornamented and near zero where the ranking is flat — so
                # an override is cheap exactly on the recoverable notes and
                # stays free at zero on the matched ones that used to veto it.
                #
                # DETACHED, like every other match-head term here: the override
                # reads the ranking as evidence and must not be able to sharpen
                # it to buy itself room.
                over_logit = over_logit + self.attr_override_margin * _rank_margin(log_rank)
            over = F.logsigmoid(over_logit)
            log_ins = _logaddexp(log_ins, log_matched + over)
            log_rest = log_matched + F.logsigmoid(-over_logit)
        return torch.cat(
            [log_ins + F.logsigmoid(gate) + log_rank,
             _logaddexp(log_ins + F.logsigmoid(-gate), log_rest)],
            dim=-1,
        )


def alignment_loss(out: dict, batch: dict, weight_attr: float = 0.2) -> tuple[torch.Tensor, dict]:
    """Symmetric CE. Targets: batch['target_s'] (B, n_max) with perf index or
    m (null) or -100 (pad); batch['target_p'] (B, m_max) likewise.

    When the model carries an attribution head, a third CE over
    batch['target_attr'] (B, m_max: anchor score index, n_max = "not an
    ornament", -100 = unsupervised) is added at `weight_attr`."""
    B, n_max, _ = out["logits_s2p"].shape
    m_max = out["logits_p2s"].shape[1]

    # Padded columns were masked to -inf; renormalize target index space:
    # each sample's null lives at its own m (< m_max+1). Targets are built by
    # the collate with the sample's own null index, so CE works directly.
    loss_s = F.cross_entropy(
        out["logits_s2p"].reshape(B * n_max, -1),
        batch["target_s"].reshape(B * n_max),
        ignore_index=-100,
    )
    loss_p = F.cross_entropy(
        out["logits_p2s"].reshape(B * m_max, -1),
        batch["target_p"].reshape(B * m_max),
        ignore_index=-100,
    )
    loss = 0.5 * (loss_s + loss_p)

    with torch.no_grad():
        pred_s = out["logits_s2p"].argmax(-1)
        valid = batch["target_s"] != -100
        acc_s = ((pred_s == batch["target_s"]) & valid).sum() / valid.sum().clamp(min=1)
        pred_p = out["logits_p2s"].argmax(-1)
        valid_p = batch["target_p"] != -100
        acc_p = ((pred_p == batch["target_p"]) & valid_p).sum() / valid_p.sum().clamp(min=1)

    stats = {"loss_s": loss_s.item(), "loss_p": loss_p.item(),
             "acc_s": acc_s.item(), "acc_p": acc_p.item()}

    if "logits_attr" in out and "target_attr" in batch:
        target_attr = batch["target_attr"]
        n_max = out["logits_attr"].shape[2] - 1
        sup = target_attr != -100
        if bool(sup.any()):
            # A batch can legitimately contain no supervised rows at all (a
            # size bucket of only real-GT/self-supervised windows). Masked-out
            # cross_entropy would then be 0/0 = nan and poison the weights, so
            # the term is skipped rather than computed.
            loss_attr = F.cross_entropy(
                out["logits_attr"].reshape(B * m_max, -1),
                target_attr.reshape(B * m_max),
                ignore_index=-100,
            )
            loss = loss + weight_attr * loss_attr
            with torch.no_grad():
                pred_a = out["logits_attr"].argmax(-1)
                hit = pred_a == target_attr
                acc_attr = (hit & sup).sum() / sup.sum().clamp(min=1)
                # The number that actually matters: of the played notes that
                # ARE ornaments, how many got their principal right. Overall
                # accuracy is dominated by the "none" class and looks great
                # even for a head that never attributes anything.
                is_orn = sup & (target_attr < n_max)
                acc_orn = (hit & is_orn).sum() / is_orn.sum().clamp(min=1)
            stats |= {"loss_attr": loss_attr.item(), "acc_attr": acc_attr.item(),
                      "acc_attr_orn": acc_orn.item(), "n_attr_orn": int(is_orn.sum())}
        else:
            stats |= {"loss_attr": 0.0, "acc_attr": 0.0, "acc_attr_orn": 0.0, "n_attr_orn": 0}

    return loss, stats
