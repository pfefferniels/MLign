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

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mlign.dataset import CorpusBatcher, collate, load_corpus  # noqa: E402
from mlign.model import ModelConfig, NoteAligner, alignment_loss  # noqa: E402


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

    model = NoteAligner(ModelConfig(d_model=args.d_model, n_layers=args.n_layers, matchability=args.matchability)).to(device)
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

    def run_val(batcher=None) -> tuple[float, float]:
        batcher = batcher or val_b
        model.eval()
        tot, count = 0.0, 0
        accs = []
        with torch.no_grad():
            for batch_samples in batcher:
                batch = collate(batch_samples, device)
                out = model(batch)
                loss, m = alignment_loss(out, batch)
                tot += loss.item()
                accs.append((m["acc_s"] + m["acc_p"]) / 2)
                count += 1
        model.train()
        return tot / max(count, 1), sum(accs) / max(len(accs), 1)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tot, count = 0.0, 0
        for batch_samples in train_b:
            batch = collate(batch_samples, device)
            out = model(batch)
            loss, metrics = alignment_loss(out, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            count += 1
            if device == "mps" and count % 50 == 0:
                torch.mps.empty_cache()
        val_loss, val_acc = run_val()
        rec = {
            "epoch": epoch,
            "train_loss": round(tot / max(count, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": sched.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        if sel_b is not None:
            sel_loss, sel_acc = run_val(sel_b)
            rec["val_mix_loss"], rec["val_mix_acc"] = rec["val_loss"], rec["val_acc"]
            rec["val_loss"], rec["val_acc"] = round(sel_loss, 4), round(sel_acc, 4)
            val_loss = sel_loss  # selection criterion = dedicated set
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
        torch.save(state, last)
        if val_loss < best_val:
            best_val = val_loss
            state["best_val"] = best_val
            torch.save(state, run_dir / "best.pt")
        if args.snapshot_every and (epoch + 1) % args.snapshot_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": vars(args)},
                       run_dir / f"snap-e{epoch}.pt")


if __name__ == "__main__":
    main()
