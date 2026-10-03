"""Train a spoofing countermeasure on ASVspoof 2019 LA, select on dev EER, score dev + eval.

  python train/train_cm.py --model aasist --root /kaggle/input/asvpoof-2019-dataset/LA/LA --out out/aasist
Recipe (AASIST paper): Adam lr 1e-4, weight decay 1e-4, cosine schedule, weighted cross-entropy
[spoof 0.1, bona fide 0.9], 64,600-sample random crops, batch 24. Mixed precision (fp16).
Writes: <out>/best.pt, <out>/log.jsonl, <out>/scores_{dev,eval}.txt (utt attack label score).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voiceshield.data import ASVspoofLA  # noqa: E402
from voiceshield.metrics import eer  # noqa: E402
from voiceshield.models import build, n_params  # noqa: E402


@torch.no_grad()
def score(model, ds, device, bs=64, workers=4):
    model.eval()
    out = np.zeros(len(ds), np.float32)
    for x, _, idx in DataLoader(ds, batch_size=bs, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logit = model(x.to(device, non_blocking=True)).float()
        out[idx.numpy()] = (logit[:, 1] - logit[:, 0]).cpu().numpy()   # higher = more bona fide
    return out


def write_scores(path, ds, s):
    with open(path, "w") as f:
        for u, v in zip(ds.items, s):
            f.write(f"{u.utt} {u.attack} {'bonafide' if u.label else 'spoof'} {v:.6f}\n")


def eer_of(ds, s):
    lab = np.array([u.label for u in ds.items])
    return eer(s[lab == 1], s[lab == 0])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-budget-h", type=float, default=11.0, help="stop training after this many hours (Kaggle 12 h limit)")
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N utterances per split")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    tr = ASVspoofLA("train", train=True, limit=args.limit, seed=args.seed, root=root)
    dv = ASVspoofLA("dev", limit=args.limit, root=root)
    model = build(args.model).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    w = torch.tensor([0.1, 0.9], device=dev)   # [spoof, bona fide]: the AASIST weighting
    loader = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True,
                        drop_last=True, persistent_workers=args.workers > 0)
    meta = {"model": args.model, "params": n_params(model), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "train_utts": len(tr), "dev_utts": len(dv), "bs": args.bs, "epochs_planned": args.epochs}
    print(json.dumps(meta), flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    best, t0 = 1.0, time.time()
    for ep in range(args.epochs):
        tr.set_epoch(ep); model.train(); te = time.time(); tot = n = 0
        for x, y, _ in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logit = model(x).float()
            loss = F.cross_entropy(logit, y, weight=w)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        s = score(model, dv, dev, workers=args.workers)
        d_eer = eer_of(dv, s)
        row = {"epoch": ep + 1, "loss": tot / n, "dev_eer": d_eer, "epoch_min": (time.time() - te) / 60,
               "elapsed_h": (time.time() - t0) / 3600, "lr": sched.get_last_lr()[0]}
        if d_eer < best:
            best = d_eer
            torch.save({"model": args.model, "state": model.state_dict(), "epoch": ep + 1, "dev_eer": d_eer}, out / "best.pt")
            row["saved"] = True
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if (time.time() - t0) / 3600 > args.time_budget_h:
            print(f"time budget reached after epoch {ep + 1}", flush=True)
            break

    ck = torch.load(out / "best.pt", map_location=dev)
    model.load_state_dict(ck["state"])
    write_scores(out / "scores_dev.txt", dv, score(model, dv, dev, workers=args.workers))
    if not args.no_eval:
        ev = ASVspoofLA("eval", limit=args.limit, root=root)
        s = score(model, ev, dev, workers=args.workers)
        write_scores(out / "scores_eval.txt", ev, s)
        print(json.dumps({"best_epoch": ck["epoch"], "dev_eer": ck["dev_eer"], "eval_eer": eer_of(ev, s)}), flush=True)


if __name__ == "__main__":
    main()
