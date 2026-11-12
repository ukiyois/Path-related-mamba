"""Pulmonary artery / vein separation on non-contrast CT: a real 3-D task in which a label has to travel along the
structure. HiPaS (Chu et al., Nat. Commun. 2025; 250 scans with artery and vein masks), isotropic 1.6 mm, 200 / 50
train / test (explore3d/data/prep_hipas.py). Without contrast, arteries and veins have the same intensity and run side
by side; which tree a vessel belongs to is decided by where it connects (pulmonary trunk or left atrium), so the label
of a peripheral vessel has to be carried along the vessel through its turns in 3-D.

Network: the 3-D U-Net of vessel3d.py (stages at 1/2 and 1/4) with two output channels (artery, vein); only the token
mixer changes (routed, iso, b0, mamba_tri, conv). Training as in vessel3d.py: 96^3 patches, two thirds centred on vessel
voxels, random axis flips, BCE + soft Dice per channel, AdamW + OneCycle, bf16.
Metrics (as in HiPaS): Dice of the artery and of the vein masks (threshold 0.5); and the artery / vein accuracy on the
ground-truth vessel voxels (the larger of the two probabilities), overall and by geodesic distance along the vessel from
the root of its tree (its thickest part: distance transform >= 0.8 of the maximum).

  python explore3d/pulmav.py --model routed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "explore3d")]

import numpy as np
import torch
import torch.nn.functional as F

from vessel3d import VesselNet, pad_to

SPACING = 1.6                                   # mm per voxel (prep_hipas.py)
GEO_BINS = (0, 32, 64, 96, 128)                 # mm along the vessel from the root of its tree
GEO_NAMES = ("0-32", "32-64", "64-96", "96-128", "128+", "unreached")


def load(root, names):
    vols = []
    for n in names:
        z = np.load(Path(root) / f"{n}.npz")
        vols.append((n, z["img"], z["art"].astype(bool), z["vein"].astype(bool)))
    return vols


def sample(vols, coords, B, P, rng, dev):
    xs, ys = [], []
    for _ in range(B):
        i = rng.integers(len(vols))
        _, img, art, vein = vols[i]
        img, art, vein = pad_to(img, P), pad_to(art, P), pad_to(vein, P)
        if rng.random() < 2 / 3 and len(coords[i]):                   # centre on a vessel voxel
            c = coords[i][rng.integers(len(coords[i]))]
            lo = [int(np.clip(int(ci) - P // 2, 0, s - P)) for ci, s in zip(c, art.shape)]
        else:
            lo = [int(rng.integers(0, s - P + 1)) for s in art.shape]
        sl = tuple(slice(a, a + P) for a in lo)
        x = img[sl].astype(np.float32)
        y = np.stack([art[sl], vein[sl]]).astype(np.float32)
        for ax in range(3):
            if rng.random() < 0.5:
                x, y = np.flip(x, ax), np.flip(y, ax + 1)
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
    x = torch.from_numpy(np.stack(xs))[:, None].to(dev)
    return (x - 0.5) / 0.25, torch.from_numpy(np.stack(ys)).to(dev)


def predict(net, img, P, dev):
    """Sliding windows (P^3, stride P/2), probabilities averaged: (2, ...) artery and vein."""
    shape = img.shape
    a = pad_to(img, P).astype(np.float32)
    S = P // 2
    full = [max(P, -(-(s - P) // S) * S + P) for s in a.shape]
    x = np.zeros(full, np.float32)
    x[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    xt = (torch.from_numpy(x)[None, None].to(dev) - 0.5) / 0.25
    prob, cnt = torch.zeros([2] + full, device=dev), torch.zeros(full, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, full[0] - P + 1, S):
            for j in range(0, full[1] - P + 1, S):
                for k in range(0, full[2] - P + 1, S):
                    p = torch.sigmoid(net(xt[:, :, i:i + P, j:j + P, k:k + P]).float())[0]
                    prob[:, i:i + P, j:j + P, k:k + P] += p
                    cnt[i:i + P, j:j + P, k:k + P] += 1
    return (prob / cnt)[:, :shape[0], :shape[1], :shape[2]].cpu().numpy()


def tree_distance(mask):
    """Geodesic distance (mm) inside the mask from the root of the tree; -1 outside it or where the root is not
    reached (a piece not connected to the root at this resolution)."""
    from scipy import ndimage
    from skimage.graph import MCP_Geometric
    edt = ndimage.distance_transform_edt(mask)
    root = np.argwhere(edt >= 0.8 * edt.max())
    d, _ = MCP_Geometric(np.where(mask, 1.0, np.inf), fully_connected=True).find_costs(root)
    return np.where(mask & np.isfinite(d), d * SPACING, -1).astype(np.float32)


def geodesic(root, name, art, vein):
    """Cached (2, ...) distances along the artery and the vein trees of one volume."""
    path = Path(root) / "geo_v1" / f"{name}.npz"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, art=tree_distance(art).astype(np.float16), vein=tree_distance(vein).astype(np.float16))
        os.replace(tmp, path)
    z = np.load(path)
    return z["art"].astype(np.float32), z["vein"].astype(np.float32)


def evaluate(net, vols, P, dev, root):
    tp, fp, fn = np.zeros(2), np.zeros(2), np.zeros(2)
    hit, tot = np.zeros(len(GEO_NAMES)), np.zeros(len(GEO_NAMES))
    net.eval()
    for name, img, art, vein in vols:
        p = predict(net, img, P, dev)
        geo = geodesic(root, name, art, vein)
        for c, lab in enumerate((art, vein)):
            pred = p[c] > 0.5
            tp[c] += (pred & lab).sum(); fp[c] += (pred & ~lab).sum(); fn[c] += (~pred & lab).sum()
            correct = (p[c] > p[1 - c])[lab]                         # artery / vein decision on this tree's voxels
            g = geo[c][lab]
            b = np.digitize(g, GEO_BINS) - 1
            b[g < 0] = len(GEO_NAMES) - 1
            np.add.at(hit, b, correct)
            np.add.at(tot, b, 1)
    net.train()
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return dict(dice_artery=float(dice[0]), dice_vein=float(dice[1]), av_accuracy=float(hit.sum() / tot.sum()),
                av_accuracy_by_geodesic_mm={n: dict(n=int(t), acc=float(h / max(t, 1)))
                                            for n, h, t in zip(GEO_NAMES, hit, tot)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="/data/datasets/hipas")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/pulmav")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"pulmav_{args.model}_C{args.C}_L{args.L}" + (f"_seed{args.seed}" if args.seed else "")
    split = json.loads((Path(args.root) / "split.json").read_text())
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, test = load(args.root, split["train"]), load(args.root, split["test"])
    coords = [np.argwhere(pad_to(a, args.patch) | pad_to(v, args.patch)).astype(np.int16) for _, _, a, v in train]
    net = VesselNet(args.model, args.C, args.L, out=2).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    ck_path = out / f"{name}.pt"
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(train)} train / {len(test)} test volumes", file=log, flush=True)
    it0 = 0
    if ck_path.exists():                                              # resume after a crash
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        it0, rng = ck["it"], ck["rng"]
        print(f"resumed at it {it0}", file=log, flush=True)
    t0, skipped = time.time(), 0
    for it in range(it0 + 1, args.steps + 1):
        x, y = sample(train, coords, args.batch, args.patch, rng, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x).float()
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum((2, 3, 4)) + 1) / (p.sum((2, 3, 4)) + y.sum((2, 3, 4)) + 1)
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
        if it % 1000 == 0 or it == args.steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it, rng=rng),
                       ck_path)
    m = evaluate(net, test, args.patch, dev, args.root)
    r = dict(name=name, model=args.model, params_k=nparam / 1e3, train_seconds=time.time() - t0, skipped_steps=skipped,
             **m, args=vars(args))
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
