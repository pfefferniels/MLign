"""The attribution head's conditioning on the match head.

The one property worth pinning: reading the match head's insertion decision
must not give the attribution loss a route back into the match head. That is
what kept the head strictly additive (v2 vs v7attr: n.s. on all three alignment
metrics), and it is easy to lose by dropping a `.detach()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import collate, featurize  # noqa: E402
from mlign.model import ModelConfig, NoteAligner, alignment_loss, config_from_ckpt  # noqa: E402

MODES = ["", "bias", "factored", "residual"]

# Match-head-only parameters: the encoder is shared with attribution by design,
# these are not.
MATCH_ONLY = ("out_s.", "out_p.", "null_s", "null_p", "scale", "matchability_")


def a_row() -> dict:
    """Four score notes; five played notes, of which the last two are a trill
    on score note 1."""
    return {
        "meta": {"gen": "mlign-v0", "seed": "t"},
        "score": [[0, 720, 60, 0], [720, 720, 64, 0], [1440, 720, 67, 0], [2160, 720, 72, 0]],
        "perf": [[0.0, 480.0, 60, 64], [500.0, 200.0, 64, 70], [700.0, 120.0, 66, 66],
                 [820.0, 120.0, 64, 66], [1000.0, 480.0, 67, 64]],
        "align": [[0, 0], [1, 1], [2, 4]],
        "subs": [],
        "ins": [[2, 2], [3, 2]],
        "orn": [[2, 1, 0, 0], [3, 1, 1, 0]],
        "del": [3],
    }


def a_batch(device: str = "cpu") -> dict:
    return collate([featurize(a_row()), featurize(a_row())], device)


def a_model(mode: str) -> NoteAligner:
    torch.manual_seed(0)
    return NoteAligner(ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64,
                                   dropout=0.0, matchability=True,
                                   attribution=True, attr_conditioned=mode))


@pytest.mark.parametrize("mode", MODES)
def test_shape_and_finiteness(mode: str) -> None:
    out = a_model(mode).eval()(a_batch())
    logits = out["logits_attr"]
    assert logits.shape == (2, 5, 4 + 1)  # (B, m, n + none)
    assert torch.isfinite(logits).all(), "no -inf/nan outside padded columns"


@pytest.mark.parametrize("mode", MODES)
def test_attribution_loss_never_reaches_the_match_head(mode: str) -> None:
    model = a_model(mode)
    batch = a_batch()
    out = model(batch)
    # The attribution term alone — as if attr_weight were the whole loss.
    m_max = out["logits_attr"].shape[1]
    loss = torch.nn.functional.cross_entropy(
        out["logits_attr"].reshape(2 * m_max, -1),
        batch["target_attr"].reshape(2 * m_max),
        ignore_index=-100,
    )
    loss.backward()
    for name, p in model.named_parameters():
        if name.startswith(MATCH_ONLY) and p.grad is not None:
            assert torch.count_nonzero(p.grad) == 0, f"attribution loss moved {name}"


def test_off_by_default_is_the_v2_head() -> None:
    """The unconditioned path must stay bit-identical, so a v2 checkpoint keeps
    scoring exactly what it scored."""
    assert ModelConfig().attr_conditioned == ""
    plain = a_model("").eval()
    with torch.no_grad():
        logits = plain(a_batch())["logits_attr"]
        p_enc, s_enc = None, None
        x = plain.encode(*[a_batch()[k] for k in ("pitch", "cont", "segment", "position", "pad")])
        s_enc, p_enc = x[:, 1:5], x[:, 6:11]
        a_p = plain.attr_p(p_enc)
        expect = torch.cat(
            [a_p @ plain.attr_s(s_enc).transpose(1, 2),
             (a_p @ plain.attr_none)[:, :, None]], dim=2) * plain.attr_scale
    assert torch.allclose(logits, expect, atol=1e-6)


def test_factored_is_a_normalized_distribution() -> None:
    """`factored` builds log-probabilities directly rather than logits, so the
    row must already sum to one — that is what makes the none column mean
    "the match head thinks this note matched something"."""
    out = a_model("factored").eval()(a_batch())
    total = out["logits_attr"].logsumexp(-1)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-4)


def test_factored_none_tracks_the_match_head() -> None:
    """A played note the match head is sure it matched cannot be an ornament,
    whatever the ranking says."""
    model = a_model("factored").eval()
    batch = a_batch()
    with torch.no_grad():
        out = model(batch)
        _, log_matched = model._match_evidence(out["logits_p2s"])
        none = out["logits_attr"][..., -1:]
    # P(none) ≥ P(matched): the remaining mass can only come from insertions
    # judged unattributable, which adds to it.
    assert (none >= log_matched - 1e-4).all()


@pytest.mark.parametrize("mode", MODES)
def test_alignment_loss_runs_with_the_head(mode: str) -> None:
    out = a_model(mode)(a_batch())
    loss, stats = alignment_loss(out, a_batch(), weight_attr=0.2)
    assert torch.isfinite(loss)
    assert stats["n_attr_orn"] == 4  # two trill notes per row, two rows


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("attribution", [True, False])
def test_a_checkpoint_describes_itself(mode: str, attribution: bool) -> None:
    """Every loader in the repo rebuilds the model from `config_from_ckpt`, so
    a checkpoint has to survive the round trip with its config THROWN AWAY.

    This is not hypothetical: `mlign align`, `mlign serve` and four eval scripts
    each hand-rolled this and each omitted a field, so none of them could load
    the shipped `models/mlign-v2.pt` at all — they built a model with no
    attribution head and then loaded strictly into it.
    """
    if mode and not attribution:
        pytest.skip("conditioning only exists with the head")
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0,
                      matchability=True, attribution=attribution, attr_conditioned=mode)
    state = NoteAligner(cfg).state_dict()
    # Nothing but d_model/n_layers survives — the heads must come from the weights.
    rebuilt = config_from_ckpt({"d_model": 32, "n_layers": 2}, state,
                               n_heads=2, d_ff=64, dropout=0.0)
    assert rebuilt.attribution is attribution
    assert rebuilt.attr_conditioned == mode
    assert rebuilt.matchability is True
    NoteAligner(rebuilt).load_state_dict(state)  # strict: raises if a head is missing
