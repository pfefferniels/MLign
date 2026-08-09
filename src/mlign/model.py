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

        s = self.out_s(s)
        p = self.out_p(p)
        sim = torch.einsum("bnd,bmd->bnm", s, p) * self.scale

        null_col = torch.einsum("bnd,d->bn", s, self.null_p)[:, :, None] * self.scale
        null_row = torch.einsum("bmd,d->bm", p, self.null_s)[:, :, None] * self.scale

        logits_s2p = torch.cat([sim, null_col], dim=2)
        logits_s2p = logits_s2p.masked_fill(
            torch.cat([p_pad, torch.zeros_like(p_pad[:, :1])], dim=1)[:, None, :], float("-inf")
        )
        logits_p2s = torch.cat([sim.transpose(1, 2), null_row], dim=2)
        logits_p2s = logits_p2s.masked_fill(
            torch.cat([s_pad, torch.zeros_like(s_pad[:, :1])], dim=1)[:, None, :], float("-inf")
        )
        return {"logits_s2p": logits_s2p, "logits_p2s": logits_p2s, "s_pad": s_pad, "p_pad": p_pad}


def alignment_loss(out: dict, batch: dict) -> tuple[torch.Tensor, dict]:
    """Symmetric CE. Targets: batch['target_s'] (B, n_max) with perf index or
    m (null) or -100 (pad); batch['target_p'] (B, m_max) likewise."""
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

    return loss, {"loss_s": loss_s.item(), "loss_p": loss_p.item(), "acc_s": acc_s.item(), "acc_p": acc_p.item()}
