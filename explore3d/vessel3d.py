"""3-D thin-structure segmentation: hepatic vessels (MSD Task08, isotropic 1.6 mm, explore3d/data/prep_vessel3d.py) with a
3-D U-Net in which only the token mixer changes.

Network: stem conv at full resolution (16 ch) -> stages at 1/2 and 1/4 (C and 2C channels, L mixer blocks each; block =
RMS norm -> mixer -> residual -> pointwise FFN) -> decoder (trilinear up, skip, two 3x3x3 convs) -> logits.
Mixers: routed (learned routing pi), iso (pi = 1/3 fixed: isotropic coupling), b0 (axis-aligned scans, no coupling),
mamba_tri (three serialisations, SegMamba-style), conv (depthwise 3x3x3).
Training: 96^3 patches, two thirds centred on vessel voxels, random axis flips, BCE + soft Dice, AdamW + OneCycle, bf16.
Evaluation on the test split: sliding windows (96^3, stride 48, probabilities averaged), threshold 0.5. Dice (pooled),
clDice (3-D skeletons, per volume), and recall of the ground-truth centre lines by local direction (second moments of
the skeleton in a 7^3 window): axis-aligned (largest direction cosine > cos 22.5 deg), space-diagonal (smallest
cosine > 0.4), face-diagonal (the rest).

  python explore3d/vessel3d.py --model routed --out runs/vessel3d
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


class Norm3d(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, C, 1, 1, 1))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5) * self.w


class Block3d(nn.Module):
    def __init__(self, C, mixer):
        super().__init__()
        self.n1, self.mix, self.n2 = Norm3d(C), mixer, Norm3d(C)
        self.ffn = nn.Sequential(nn.Conv3d(C, 2 * C, 1), nn.GELU(), nn.Conv3d(2 * C, C, 1))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


def make_mixer(kind, C):
    if kind in ("routed", "iso"):
        from routed3d import RoutedRoesser3D
        return RoutedRoesser3D(C, routing="learned" if kind == "routed" else "uniform")
    from syn_stereo import ConvMixer, MambaMixer, RoesserMixer
    if kind == "b0":
        return RoesserMixer(C, False)
    if kind == "mamba_tri":
        return MambaMixer(C, "tri")
    if kind == "conv":
        return ConvMixer(C)
    raise ValueError(kind)


def conv_block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.GELU(), nn.Conv3d(co, co, 3, padding=1), nn.GELU())


class VesselNet(nn.Module):
    def __init__(self, kind, C=32, L=2, out=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.GELU())
        self.d1, self.d2 = nn.Conv3d(16, C, 2, stride=2), nn.Conv3d(C, 2 * C, 2, stride=2)
        self.s1 = nn.Sequential(*[Block3d(C, make_mixer(kind, C)) for _ in range(L)])
        self.s2 = nn.Sequential(*[Block3d(2 * C, make_mixer(kind, 2 * C)) for _ in range(L)])
        self.u1, self.u0 = conv_block(3 * C, C), conv_block(C + 16, 16)
        self.head = nn.Conv3d(16, out, 1)
        self.out = out

    def forward(self, x):
        f0 = self.stem(x)
        f1 = self.s1(self.d1(f0))
        f2 = self.s2(self.d2(f1))
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-3:], mode="trilinear", align_corners=False)
        h = self.u1(torch.cat([up(f2, f1), f1], 1))
        logits = self.head(self.u0(torch.cat([up(h, f0), f0], 1)))
        return logits[:, 0] if self.out == 1 else logits


def load(root, names):
    vols = []
    for n in names:
        z = np.load(Path(root) / f"{n}.npz")
        vols.append((n, z["img"], z["lab"].astype(bool)))
    return vols


def pad_to(a, P):
    pads = [(0, max(0, P - s)) for s in a.shape]
    return np.pad(a, pads) if any(p[1] for p in pads) else a



def occlude3d(img, lab, centres, R, rng):
    """Hide the vessel inside balls of radius R around centre-line voxels: vessel voxels (label dilated by 1) in a ball
    are replaced by the mean intensity of the ball's other voxels plus noise of their standard deviation; the label
    is unchanged."""
    from scipy import ndimage
    img = img.astype(np.float32).copy()
    struct = ndimage.binary_dilation(lab, iterations=1)
    for c in centres:
        lo = [max(0, int(ci) - R) for ci in c]
        hi = [min(s, int(ci) + R + 1) for ci, s in zip(c, lab.shape)]
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        g = np.ogrid[tuple(slice(a, b) for a, b in zip(lo, hi))]
        ball = sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= R * R
        bg, hole = ball & ~struct[sl], ball & struct[sl]
        if bg.sum() < 8 or not hole.any():
            continue
        sub = img[sl]
        mu, sd = sub[bg].mean(), sub[bg].std()
        sub[hole] = mu + rng.standard_normal(hole.sum()) * sd
    return img

def sample(vols, B, P, rng, dev, occ=0, R=4, skels=None):
    xs, ys = [], []
    for _ in range(B):
        i = rng.integers(len(vols))
        _, img, lab = vols[i]
        img, lab = pad_to(img, P), pad_to(lab, P)
        if rng.random() < 2 / 3 and lab.any():                       # centre on a vessel voxel
            c = np.argwhere(lab)[rng.integers(lab.sum())]
            lo = [int(np.clip(ci - P // 2, 0, s - P)) for ci, s in zip(c, lab.shape)]
        else:
            lo = [int(rng.integers(0, s - P + 1)) for s in lab.shape]
        sl = tuple(slice(a, a + P) for a in lo)
        x, y = img[sl].astype(np.float32), lab[sl].astype(np.float32)
        if occ:                                                       # hide `occ` pieces of the vessel tree
            sk = np.argwhere(pad_to(skels[i], P)[sl])
            if len(sk):
                x = occlude3d(x, y > 0, sk[rng.integers(len(sk), size=occ)], R, rng)
        for ax in range(3):
            if rng.random() < 0.5:
                x, y = np.flip(x, ax), np.flip(y, ax)
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
    x = torch.from_numpy(np.stack(xs))[:, None].to(dev)
    return (x - 0.5) / 0.25, torch.from_numpy(np.stack(ys)).to(dev)


def predict(net, img, P, dev):
    shape = img.shape
    a = pad_to(img, P).astype(np.float32)
    S = P // 2
    full = [max(P, -(-(s - P) // S) * S + P) for s in a.shape]
    x = np.zeros(full, np.float32)
    x[:a.shape[0], :a.shape[1], :a.shape[2]] = a
    xt = (torch.from_numpy(x)[None, None].to(dev) - 0.5) / 0.25
    prob, cnt = torch.zeros(full, device=dev), torch.zeros(full, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, full[0] - P + 1, S):
            for j in range(0, full[1] - P + 1, S):
                for k in range(0, full[2] - P + 1, S):
                    p = torch.sigmoid(net(xt[:, :, i:i + P, j:j + P, k:k + P]).float())[0]
                    prob[i:i + P, j:j + P, k:k + P] += p
                    cnt[i:i + P, j:j + P, k:k + P] += 1
    return (prob / cnt)[:shape[0], :shape[1], :shape[2]].cpu().numpy()


def direction_class(skel):
    """0 axis-aligned, 1 face-diagonal, 2 space-diagonal for every skeleton voxel (second moments in a 7^3 window)."""
    from scipy import ndimage
    r = 3
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1].astype(np.float64)
    s = skel.astype(np.float64)
    M = np.empty(skel.shape + (3, 3))
    for a in range(3):
        for b in range(a, 3):
            M[..., a, b] = M[..., b, a] = ndimage.convolve(s, g[a] * g[b], mode="constant")
    idx = np.nonzero(skel)
    _, vec = np.linalg.eigh(M[idx])
    v = np.sort(np.abs(vec[:, :, -1]), axis=1)                        # principal direction cosines, ascending
    cls = np.ones(len(v), np.int64)
    cls[v[:, 2] > np.cos(np.radians(22.5))] = 0
    cls[v[:, 0] > 0.4] = 2
    out = -np.ones(skel.shape, np.int64)
    out[idx] = cls
    return out


def evaluate(net, vols, P, dev):
    from skimage.morphology import skeletonize
    tp = fp = fn = 0
    cl, hit, tot = [], np.zeros(3), np.zeros(3)
    net.eval()
    for _, img, lab in vols:
        pred = predict(net, img, P, dev) > 0.5
        tp += (pred & lab).sum(); fp += (pred & ~lab).sum(); fn += (~pred & lab).sum()
        sp, sg = skeletonize(pred), skeletonize(lab)
        sp, sg = sp > 0, sg > 0
        tprec, tsens = (sp & lab).sum() / max(sp.sum(), 1), (sg & pred).sum() / max(sg.sum(), 1)
        cl.append(2 * tprec * tsens / max(tprec + tsens, 1e-9))
        d = direction_class(sg)
        for c in range(3):
            m = d == c
            tot[c] += m.sum(); hit[c] += (m & pred).sum()
    net.train()
    names = ("axis", "face_diagonal", "space_diagonal")
    return dict(dice=2 * tp / max(2 * tp + fp + fn, 1), cldice=float(np.mean(cl)),
                recall_by_direction={n: float(hit[c] / max(tot[c], 1)) for c, n in enumerate(names)},
                centreline_voxels_by_direction={n: int(tot[c]) for c, n in enumerate(names)})



def evaluate_occluded(net, vols, P, dev, R, per_volume=40, seed=123):
    """Bridging in 3-D: hide the vessel in up to per_volume balls (radius R, centres on the centre line, at least 2R
    apart, fixed seed) and measure the recall of the hidden centre-line voxels by local direction."""
    from skimage.morphology import skeletonize
    rng = np.random.default_rng(seed)
    hit, tot = np.zeros(3), np.zeros(3)
    net.eval()
    for _, img, lab in vols:
        sg = skeletonize(lab) > 0
        cand = np.argwhere(sg)
        if not len(cand):
            continue
        d = direction_class(sg)
        rng.shuffle(cand)
        centres = []
        for c in cand:
            if all(((c - e) ** 2).sum() >= (2 * R) ** 2 for e in centres):
                centres.append(c)
            if len(centres) == per_volume:
                break
        pred = predict(net, occlude3d(img, lab, centres, R, rng), P, dev) > 0.5
        for c in centres:
            lo = [max(0, int(ci) - R) for ci in c]
            hi = [min(s, int(ci) + R + 1) for ci, s in zip(c, lab.shape)]
            sl = tuple(slice(a, b) for a, b in zip(lo, hi))
            g = np.ogrid[tuple(slice(a, b) for a, b in zip(lo, hi))]
            ball = sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= R * R
            k = d[tuple(c)]
            hidden = ball & sg[sl]
            tot[k] += hidden.sum(); hit[k] += (hidden & pred[sl]).sum()
    net.train()
    names = ("axis", "face_diagonal", "space_diagonal")
    return dict(bridging_recall=float(hit.sum() / max(tot.sum(), 1)),
                bridging_recall_by_direction={n: float(hit[c] / max(tot[c], 1)) for c, n in enumerate(names)},
                bridging_voxels_by_direction={n: int(tot[c]) for c, n in enumerate(names)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="/data/datasets/vessel3d")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occlude", type=int, default=0, help="vessel pieces hidden per training patch (0 = off)")
    ap.add_argument("--occ_r", type=int, default=4, help="radius of a hidden piece (voxels of 1.6 mm)")
    ap.add_argument("--out", default="runs/vessel3d")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"vessel_{args.model}_C{args.C}_L{args.L}" + (f"_occ{args.occlude}r{args.occ_r}" if args.occlude else "") \
        + (f"_seed{args.seed}" if args.seed else "")
    split = json.loads((Path(args.root) / "split.json").read_text())
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, test = load(args.root, split["train"]), load(args.root, split["test"])
    skels = None
    if args.occlude:
        from skimage.morphology import skeletonize
        skels = [skeletonize(lab) > 0 for _, _, lab in train]
    net = VesselNet(args.model, args.C, args.L).to(dev)
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
        x, y = sample(train, args.batch, args.patch, rng, dev, args.occlude, args.occ_r, skels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x).float()
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
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
    m = evaluate(net, test, args.patch, dev)
    m.update(evaluate_occluded(net, test, args.patch, dev, args.occ_r))
    r = dict(name=name, model=args.model, params_k=nparam / 1e3, train_seconds=time.time() - t0, skipped_steps=skipped,
             **m, args=vars(args))
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
