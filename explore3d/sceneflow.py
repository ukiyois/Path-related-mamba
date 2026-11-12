"""Exploration E6: aggregators inside one stereo network on SceneFlow FlyingThings3D (finalpass).

Data: /data/sceneflow/FlyingThings3D/{frames_finalpass,disparity}/{TRAIN,TEST}/{A,B,C}/NNNN/left|right.
Training: random 256x512 crops, photometric jitter, max disparity 192, smooth-L1 on 0 < d < 192.
Evaluation: the first --n_test TEST pairs at full 540x960 (padded to 544x960), EPE / bad-1 / bad-3 / D1
on 0 < d < 192. The network is flying_planes.StereoNet (CNN features, group-wise correlation, aggregator,
soft-argmin); only the aggregator changes.

  python explore3d/sceneflow.py --model routed --steps 20000 --out runs/e6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "explore3d")]

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SF = Path("/data/sceneflow/FlyingThings3D")


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().rstrip()
        channels = 3 if header == "PF" else 1
        w, h = map(int, re.findall(r"\d+", f.readline().decode()))
        scale = float(f.readline().decode().rstrip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4")
    data = data.reshape(h, w, channels) if channels == 3 else data.reshape(h, w)
    return np.ascontiguousarray(np.flipud(data))


def list_pairs(split):
    pairs = []
    for left in sorted((SF / "frames_finalpass" / split).glob("*/*/left/*.png")):
        right = left.parent.parent / "right" / left.name
        disp = Path(str(left.parent).replace("frames_finalpass", "disparity")) / (left.stem + ".pfm")
        if right.exists() and disp.exists():
            pairs.append((left, right, disp))
    return pairs


class SceneFlow(Dataset):
    def __init__(self, split, crop=(256, 512), train=True):
        self.pairs = list_pairs(split)
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from PIL import Image
        lp, rp, dp = self.pairs[i]
        L = np.asarray(Image.open(lp).convert("RGB"), np.float32) / 255
        R = np.asarray(Image.open(rp).convert("RGB"), np.float32) / 255
        D = read_pfm(dp).astype(np.float32)
        if self.train:
            ch, cw = self.crop
            y = np.random.randint(0, L.shape[0] - ch + 1)
            x = np.random.randint(0, L.shape[1] - cw + 1)
            L, R, D = L[y:y + ch, x:x + cw], R[y:y + ch, x:x + cw], D[y:y + ch, x:x + cw]
            for img in (L, R):                                   # independent brightness / contrast
                img *= np.random.uniform(0.8, 1.2)
                img += np.random.uniform(-0.05, 0.05)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return t(np.clip(L, 0, 1)).permute(2, 0, 1), t(np.clip(R, 0, 1)).permute(2, 0, 1), t(D)


def evaluate(net, ds, n, max_disp, dev):
    net.eval()
    tot = dict(epe=0.0, bad1=0.0, bad3=0.0, d1=0.0)
    cnt = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(min(n, len(ds))):
            L, R, D = ds[i]
            L, R, D = L[None].to(dev), R[None].to(dev), D[None].to(dev)
            H, W = L.shape[-2:]
            ph = (32 - H % 32) % 32
            pw = (32 - W % 32) % 32
            Lp = F.pad(L, (0, pw, ph, 0), mode="replicate")
            Rp = F.pad(R, (0, pw, ph, 0), mode="replicate")
            pred = net(Lp, Rp).float()[:, ph:, :W]
            valid = (D > 0) & (D < max_disp)
            err = (pred - D).abs()[valid]
            d = D[valid]
            tot["epe"] += err.sum().item()
            tot["bad1"] += (err > 1).sum().item()
            tot["bad3"] += (err > 3).sum().item()
            tot["d1"] += ((err > 3) & (err > 0.05 * d)).sum().item()
            cnt += valid.sum().item()
    net.train()
    return {k: v / max(cnt, 1) for k, v in tot.items()}


def main():
    from flying_planes import StereoNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_disp", type=int, default=192)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0, help="initialisation and data-order seed (evaluation is fixed)")
    ap.add_argument("--out", default="runs/e6")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_s{args.steps}{args.tag}" + (f"_seed{args.seed}" if args.seed else "")
    train = SceneFlow("TRAIN")
    test = SceneFlow("TEST", train=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    net = StereoNet(args.model, args.max_disp, args.C, args.L, 256, 512).to(dev)
    if args.ckpt and hasattr(net, "body"):
        from torch.utils.checkpoint import checkpoint
        body = net.body

        class Ck(torch.nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

            def forward(self, x):
                for blk in self.seq:
                    x = checkpoint(blk, x, use_reentrant=False) if self.training else blk(x)
                return x
        net.body = Ck(body)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.02)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, persistent_workers=True, pin_memory=True)
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(train)} train / {len(test)} test pairs", file=log, flush=True)
    t0 = time.time()
    it = 0
    ck_path = out / f"{name}.pt"                          # resume after a crash (saved every 1000 steps)
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        it = ck["it"]
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        else:                                             # older checkpoints: fast-forward the schedule
            for _ in range(it):
                sched.step()
        print(f"resumed at it {it}", file=log, flush=True)
    while it < args.steps:
        for L, R, D in loader:
            L, R, D = L.to(dev, non_blocking=True), R.to(dev, non_blocking=True), D.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = net(L, R)
            valid = (D > 0) & (D < args.max_disp)
            loss = F.smooth_l1_loss(pred.float()[valid], D[valid])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            it += 1
            if it % 500 == 0:
                print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", file=log, flush=True)
            if it % 5000 == 0 or it == args.steps:
                m = evaluate(net, test, args.n_test if it == args.steps else 100, args.max_disp, dev)
                print(json.dumps(dict(it=it, **m)), file=log, flush=True)
            if it % 1000 == 0 or it == args.steps:
                torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it),
                           ck_path)
            if it >= args.steps:
                break
    if "m" not in locals():                               # resumed after the last step: evaluate once more
        m = evaluate(net, test, args.n_test, args.max_disp, dev)
    r = dict(name=name, model=args.model, params_k=nparam / 1e3, seconds=time.time() - t0, **m, args=vars(args))
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
