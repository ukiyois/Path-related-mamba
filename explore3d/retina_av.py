"""Retinal artery / vein classification: a real 2-D image task in which a label has to travel along the structure.
Arteries and veins leave the optic disc as two trees, cross each other and look alike away from the disc; whether a
vessel is an artery or a vein is a property of the tree it belongs to, so the label of a peripheral vessel has to be
carried along the vessel from the disc. Data: the multi-disease fundus segmentation dataset (figshare 32646378, CC BY
4.0), the released images minus the authors' AV_Hard cases, 512 x 512, 80 / 20 per disease archive
(explore3d/data/prep_fundus_av.py).

Network: the 2-D U-Net of thin2d.py (stages at 1/2, 1/4, 1/8) on whole images with two output channels (artery, vein);
only the token mixer changes. Training: batches of whole images, dihedral transforms and brightness / contrast jitter,
BCE + soft Dice per channel, AdamW + OneCycle, bf16.
Metrics (as in the artery / vein literature): on ground-truth vessel pixels the larger of the two probabilities decides;
accuracy, sensitivity (arteries) and specificity (veins); Dice of the artery and of the vein masks (threshold 0.5). The
accuracy is also reported by the route of each pixel from the optic disc along its own tree: axis turns (Thm 1), orthant
switches and length; and by the quadrant around the disc centre (vessels leave the disc in all four directions, so a
one-orthant model is blind to two of the quadrants: Thm 5).

  python explore3d/retina_av.py --model m2d_oc4
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
import torch.nn.functional as F

from thin2d import ThinNet

SIZE = 512
BINS = dict(turns=[0, 1, 2, 3, 4, 5, 6], switches=[0, 1, 2, 3], length=[0, 64, 128, 256])   # last bin open-ended


def load(root, names, geometry=False):
    items = []
    for n in names:
        z = np.load(Path(root) / f"{n}.npz")
        it = dict(name=n, img=z["img"], art=z["art"].astype(bool), vein=z["vein"].astype(bool))
        if geometry:
            for k in BINS:
                it[k] = (z[f"{k}_art"], z[f"{k}_vein"])
            it["quadrant"] = quadrants(z["disc"] > 0)
        items.append(it)
    return items


def quadrants(disc):
    """Quadrant of every pixel around the optic disc centre (0 up-left, 1 up-right, 2 down-left, 3 down-right; -1 without
    a disc). A path leaving the disc into quadrant q is monotone in that orthant: a top-left sweep carries it into 3,
    a bottom-right sweep into 0, and 1 and 2 need the anti-diagonal orthants (Thm 5)."""
    if not disc.any():
        return np.full(disc.shape, -1, np.int8)
    cy, cx = np.argwhere(disc).mean(0)
    yy, xx = np.mgrid[:disc.shape[0], :disc.shape[1]]
    return ((yy >= cy) * 2 + (xx >= cx)).astype(np.int8)


def batch(items, idx, rng, dev):
    xs, ys = [], []
    for i in idx:
        x = items[i]["img"].astype(np.float32) / 255
        y = np.stack([items[i]["art"], items[i]["vein"]], -1).astype(np.float32)
        k = rng.integers(8)                                   # dihedral group: rotations and transposes
        x, y = np.rot90(x, k % 4), np.rot90(y, k % 4)
        if k >= 4:
            x, y = x.transpose(1, 0, 2), y.transpose(1, 0, 2)
        x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev)
    return (x - 0.5) / 0.25, torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2).to(dev)


def bin_of(v, edges):
    """Bin index for route values v (-1 = disc not reached -> last index)."""
    b = np.digitize(v, edges) - 1 if edges[1] > 1 else np.minimum(v, edges[-1])
    return np.where(v < 0, len(edges), b)


def evaluate(net, items, dev, bs=8):
    tp, fp, fn = np.zeros(2), np.zeros(2), np.zeros(2)
    right, total = np.zeros(2), np.zeros(2)                   # A/V decision on true artery / vein pixels
    by = {k: (np.zeros(len(e) + 1), np.zeros(len(e) + 1)) for k, e in BINS.items()}
    quad = (np.zeros(5), np.zeros(5))                         # four quadrants around the disc, then no disc
    net.eval()
    with torch.no_grad():
        for s in range(0, len(items), bs):
            chunk = items[s:s + bs]
            x = torch.from_numpy(np.stack([it["img"] for it in chunk]).astype(np.float32) / 255).permute(0, 3, 1, 2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(net(((x - 0.5) / 0.25).to(dev)).float()).cpu().numpy()
            for it, pi in zip(chunk, p):
                for c, lab in enumerate((it["art"], it["vein"])):
                    pred = pi[c] > 0.5
                    tp[c] += (pred & lab).sum(); fp[c] += (pred & ~lab).sum(); fn[c] += (~pred & lab).sum()
                    correct = (pi[c] > pi[1 - c])[lab]
                    right[c] += correct.sum(); total[c] += lab.sum()
                    for k, edges in BINS.items():
                        b = bin_of(it[k][c][lab].astype(np.int64), edges)
                        np.add.at(by[k][0], b, correct); np.add.at(by[k][1], b, 1)
                    q = it["quadrant"][lab].astype(np.int64)
                    q = np.where(q < 0, 4, q)
                    np.add.at(quad[0], q, correct); np.add.at(quad[1], q, 1)
    net.train()
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    names = lambda e: [str(v) for v in e[:-1]] + [f"{e[-1]}+", "unreached"]
    return dict(acc=float(right.sum() / total.sum()), sens=float(right[0] / total[0]), spec=float(right[1] / total[1]),
                dice_artery=float(dice[0]), dice_vein=float(dice[1]),
                **{f"acc_by_{k}": {n: dict(n=int(t), acc=float(h / max(t, 1))) for n, h, t in zip(names(e), *by[k])}
                   for k, e in BINS.items()},
                acc_by_quadrant={n: dict(n=int(t), acc=float(h / max(t, 1))) for n, h, t in
                                 zip(("up_left", "up_right", "down_left", "down_right", "no_disc"), *quad)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="/data/datasets/fundus_av")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/retina_av")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"retina_av_{args.model}_C{args.C}_L{args.L}" + (f"_seed{args.seed}" if args.seed else "")
    split = json.loads((Path(args.root) / "split.json").read_text())
    train, test = load(args.root, split["train"]), load(args.root, split["test"], geometry=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    net = ThinNet(args.model, args.C, args.L, SIZE, out=2).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(train)} train / {len(test)} test images, {args.steps} steps",
          file=log, flush=True)
    t0, skipped = time.time(), 0
    for it in range(1, args.steps + 1):
        x, y = batch(train, rng.integers(0, len(train), args.batch), rng, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x).float()
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum((2, 3)) + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
        loss = F.binary_cross_entropy_with_logits(logit, y) + dice.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        else:
            skipped += 1
        sched.step()
        if it % 500 == 0:
            print(f"it {it} loss {loss.item():.4f} {time.time() - t0:.0f}s", file=log, flush=True)
    train_s = time.time() - t0
    r = dict(name=name, model=args.model, params_k=nparam / 1e3, train_seconds=train_s, skipped_steps=skipped,
             **evaluate(net, test, dev), args=vars(args))
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
