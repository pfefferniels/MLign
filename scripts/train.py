"""Train the MLign note aligner on the synthetic corpus.

Usage:
  .venv/bin/python scripts/train.py --corpus 'data/corpus/v0-*.jsonl' \
      --epochs 20 --run runs/v0
Resumable: picks up runs/<name>/last.pt if present.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import os

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import CorpusBatcher, collate, load_corpus  # noqa: E402
from mlign.model import ModelConfig, NoteAligner, alignment_loss  # noqa: E402


def atomic_save(obj, path: Path) -> None:
    """torch.save to a temp file then os.replace — rename is atomic on one
    filesystem, so a concurrent rsync/reader can only ever see complete files."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--val-corpus", default="",
                    help="comma-separated globs for a DEDICATED validation set "
                         "(e.g. real-music windows). When set, checkpoint selection "
                         "uses ONLY these rows (the mixed-corpus val is still logged as "
                         "val_mix_*). Rows in --val-corpus are excluded from training if "
                         "they also appear in --corpus.")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also save runs/<name>/snap-e<N>.pt every N epochs (0 = off)")
    ap.add_argument("--run", default="runs/v0")
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--matchability", action="store_true")
    ap.add_argument("--attribution", action="store_true",
                    help="train the ornament-attribution head (needs espressivo-rendered "
                         "rows; other sources are ignored by the head's loss)")
    ap.add_argument("--attr-weight", type=float, default=0.2)
    ap.add_argument("--attr-conditioned", default="",
                    choices=["", "bias", "factored", "residual", "evidenced", "calibrated"],
                    help="let the attribution head read the match head's (detached) "
                         "insertion decision instead of re-deriving it: 'bias' adds "
                         "log P(matched) to the none column, 'factored' rebuilds the "
                         "whole distribution as P(ins)·P(attributable|ins)·P(anchor), "
                         "'residual' adds a learned override so it can still disagree "
                         "(measured: it does not — the override is priced out by the "
                         "~100x more numerous matched notes), 'evidenced' prices that "
                         "override by the head's own ranking margin so overriding is "
                         "cheap only where it is confident WHICH note is ornamented, "
                         "'calibrated' drops the override entirely and prices the GATE "
                         "by that margin instead")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    if device == "cpu":
        torch.set_num_threads(args.threads)
    run_dir = Path(args.run)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    paths = sorted({p for g in args.corpus.split(",") for p in glob.glob(g.strip())})
    dedicated_paths = sorted({p for g in args.val_corpus.split(",") if g.strip() for p in glob.glob(g.strip())}) if args.val_corpus else []
    train_paths = [p for p in paths if p not in set(dedicated_paths)]
    rows = load_corpus(train_paths)
    rng = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(rows), generator=rng).tolist()
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows = [rows[i] for i in perm[:n_val]]
    train_rows = [rows[i] for i in perm[n_val:]]
    sel_rows = load_corpus(dedicated_paths) if dedicated_paths else None
    print(
        f"corpus: {len(train_rows)} train / {len(val_rows)} val-mix"
        + (f" / {len(sel_rows)} val-dedicated (SELECTION)" if sel_rows else "")
        + f" pieces; device={device}; staged {len(train_paths)} train files",
        flush=True,
    )
    for p in train_paths:
        print(f"  train file: {p}", flush=True)
    if dedicated_paths:
        for p in dedicated_paths:
            print(f"  VAL-DEDICATED file: {p}", flush=True)
        print(f"  selection criterion: dedicated val ({len(sel_rows)} rows) — NOT the mixed split", flush=True)
        if len(sel_rows) < 50:
            raise SystemExit("--val-corpus resolved to <50 rows — refusing to run with a degenerate selection set")
    elif args.val_corpus:
        raise SystemExit(f"--val-corpus {args.val_corpus!r} matched NO files — refusing silent fallback to mixed val")

    train_b = CorpusBatcher(train_rows, max_tokens=args.max_tokens, seed=1)
    val_b = CorpusBatcher(val_rows, max_tokens=args.max_tokens, seed=2)
    sel_b = CorpusBatcher(sel_rows, max_tokens=args.max_tokens, seed=3) if sel_rows else None

    model = NoteAligner(ModelConfig(d_model=args.d_model, n_layers=args.n_layers,
                                    matchability=args.matchability,
                                    attribution=args.attribution,
                                    attr_weight=args.attr_weight,
                                    attr_conditioned=args.attr_conditioned)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_b))

    start_epoch = 0
    best_val = float("inf")
    last = run_dir / "last.pt"
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"resumed from epoch {ckpt['epoch']}")

    def run_val(batcher=None) -> tuple[float, float, dict]:
        """Returns (selection_loss, acc, extra).

        selection_loss is the ALIGNMENT loss only, never the combined one:
        adding the attribution term to the number that picks checkpoints would
        silently change the selection criterion that produced mlign-v1, and
        checkpoint selection is the one thing in this project that has already
        been shown to decide the benchmark (STATE: mixed-synthetic and 4x22
        both pick wrong). Attribution is reported alongside, not folded in.
        """
        batcher = batcher or val_b
        model.eval()
        tot, count = 0.0, 0
        accs = []
        orn_hit, orn_n = 0.0, 0
        with torch.no_grad():
            for batch_samples in batcher:
                batch = collate(batch_samples, device)
                out = model(batch)
                _, m = alignment_loss(out, batch, weight_attr=args.attr_weight)
                tot += 0.5 * (m["loss_s"] + m["loss_p"])
                accs.append((m["acc_s"] + m["acc_p"]) / 2)
                # weight by ornament count: most batches have none, so a plain
                # mean over batches would be dominated by empty ones
                orn_hit += m.get("acc_attr_orn", 0.0) * m.get("n_attr_orn", 0)
                orn_n += m.get("n_attr_orn", 0)
                count += 1
        model.train()
        extra = {"attr_orn_acc": round(orn_hit / orn_n, 4) if orn_n else None,
                 "attr_orn_n": orn_n}
        return tot / max(count, 1), sum(accs) / max(len(accs), 1), extra

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tot, count = 0.0, 0
        for batch_samples in train_b:
            batch = collate(batch_samples, device)
            out = model(batch)
            loss, metrics = alignment_loss(out, batch, weight_attr=args.attr_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            count += 1
            if device == "mps" and count % 50 == 0:
                torch.mps.empty_cache()
        val_loss, val_acc, val_extra = run_val()
        rec = {
            "epoch": epoch,
            "train_loss": round(tot / max(count, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        if val_extra["attr_orn_n"]:
            rec["attr_orn_acc"] = val_extra["attr_orn_acc"]
            rec["attr_orn_n"] = val_extra["attr_orn_n"]
        if sel_b is not None:
            sel_loss, sel_acc, sel_extra = run_val(sel_b)
            rec["val_mix_loss"], rec["val_mix_acc"] = rec["val_loss"], rec["val_acc"]
            rec["val_loss"], rec["val_acc"] = round(sel_loss, 4), round(sel_acc, 4)
            val_loss = sel_loss  # selection criterion = dedicated set
            # The real-music selection set carries no ornament provenance, so
            # attribution is always reported from the synthetic mix.
            if sel_extra["attr_orn_n"]:
                rec["attr_orn_acc_sel"] = sel_extra["attr_orn_acc"]
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        state = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "config": vars(args),
        }
        atomic_save(state, last)
        if val_loss < best_val:
            best_val = val_loss
            state["best_val"] = best_val
            atomic_save(state, run_dir / "best.pt")
        if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
            atomic_save({"model": model.state_dict(), "epoch": epoch, "config": vars(args)},
                        run_dir / f"snap-e{epoch:03d}.pt")


if __name__ == "__main__":
    main()
