"""Pathfinder (regenerated with the official generator, explore3d/data/gen_pathfinder.py): are the two markers joined by one
dashed curve? One network per mixer: 3x3 conv stem with stride s, L residual mixer blocks at that resolution (the
thin2d block), average and max pool, layer norm, linear classifier. Batches 0-39 train (160k), 40-44 validation, 45-49 test
(20k each), as in LRA.

  python explore3d/pathfinder.py --res 128 --model m2d --L 4 --out runs/pathfinder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "explore3d")]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from thin2d import Block, make_mixer


def load(root, res, batches):
    base = Path(root) / f"pf{res}" / "curv_contour_length_14"
    xs, ys = [], []
    for b in batches:
        meta = np.load(base / "metadata" / f"{b}.npy")
        for row in meta:
            xs.append(np.asarray(Image.open(base / row[0] / row[1]), np.uint8))
            ys.append(int(row[3]))
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


class PathNet(nn.Module):
    def __init__(self, kind, res, C=64, L=4, stride=2):
        super().__init__()
        n = res // stride
        self.stem = nn.Sequential(nn.Conv2d(1, C, 3, stride=stride, padding=1), nn.GELU())
        self.blocks = nn.Sequential(*[Block(C, make_mixer(kind, C, n, i)) for i in range(L)])
        self.norm = nn.LayerNorm(2 * C)
        self.head = nn.Linear(2 * C, 2)

    def forward(self, x):
        # mostly-background images: a per-pixel norm followed by an average pool leaves almost no per-image signal,
        # so pool first (average and max) and normalise the pooled vector
        h = self.blocks(self.stem(x))
        return self.head(self.norm(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)))


def run_eval(net, X, Y, dev, bs=256):
    net.eval()
    hit = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(dev).float().div(255).sub(0.5).div(0.5)[:, None]
            hit += (net(x).argmax(1).cpu() == Y[i:i + bs]).sum().item()
    net.train()
    return hit / len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--model", required=True)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default="/data/datasets/pathfinder")
    ap.add_argument("--out", default="runs/pathfinder")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"pf{args.res}_{args.model}_L{args.L}_C{args.C}_s{args.stride}" + (f"_seed{args.seed}" if args.seed else "")
    torch.manual_seed(args.seed)
    Xtr, Ytr = load(args.root, args.res, range(40))
    Xva, Yva = load(args.root, args.res, range(40, 45))
    Xte, Yte = load(args.root, args.res, range(45, 50))
    net = PathNet(args.model, args.res, args.C, args.L, args.stride).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    steps = int(args.epochs * len(Xtr) / args.batch)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(Xtr)} / {len(Xva)} / {len(Xte)}, {steps} steps", file=log, flush=True)
    t0, best, best_test, skipped = time.time(), 0.0, 0.0, 0
    g = torch.Generator().manual_seed(args.seed)
    perm, pos = torch.randperm(len(Xtr), generator=g), 0
    for it in range(1, steps + 1):
        if pos + args.batch > len(perm):
            perm, pos = torch.randperm(len(Xtr), generator=g), 0
        idx = perm[pos:pos + args.batch]; pos += args.batch
        x = Xtr[idx].to(dev).float().div(255).sub(0.5).div(0.5)[:, None]
        y = Ytr[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        else:
            skipped += 1
        sched.step()
        if it % 1000 == 0 or it == steps:
            va = run_eval(net, Xva, Yva, dev)
            if va >= best:
                best, best_test = va, run_eval(net, Xte, Yte, dev)
            print(f"it {it} loss {loss.item():.4f} val {va:.4f} best_val {best:.4f} test@best {best_test:.4f} "
                  f"{time.time() - t0:.0f}s", file=log, flush=True)
    r = dict(name=name, res=args.res, model=args.model, L=args.L, params_k=nparam / 1e3, best_val=best,
             test_at_best_val=best_test, final_test=run_eval(net, Xte, Yte, dev), seconds=time.time() - t0,
             skipped_steps=skipped, args=vars(args))
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
