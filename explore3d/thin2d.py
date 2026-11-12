"""Thin-structure segmentation testbed: one U-Net-style network in which only the token mixer changes.

Data: <root>/<name>/{train,test}/{img,lab,fov}/*.png (explore3d/data/prep_thin2d.py): drive, chase, deepcrack, roads.
Network: stem conv at full resolution (16 ch) -> three stages at 1/2, 1/4, 1/8 resolution (C, 2C, 4C channels, L blocks
each; block = RMS norm -> mixer -> residual -> pointwise FFN) -> U-Net decoder (bilinear up, skip, two 3x3 convs) -> logits.
Training: random crops, the 8 dihedral transforms (flips and 90-degree rotations), brightness / contrast jitter,
BCE + soft Dice, AdamW + OneCycle, bf16 autocast.
Evaluation on the test split: sliding windows of the training crop (stride crop / 2, probabilities averaged), threshold
0.5, inside the field of view when one is given. Pixel F1 and IoU (pooled), clDice (per image, averaged), and recall of
the ground-truth centre lines by local orientation (0 / 45 / 90 / 135 degrees with x right and y down, so 45 runs from
top-left to bottom-right; orientation from the second moments of the skeleton within a 7x7 window). Bridging: the
structure is hidden in discs on the centre line (explore3d/thin2d.occlude) and the hidden centre line must be recovered;
--occlude k hides k pieces per training crop so that the network learns to bridge.

  python explore3d/thin2d.py --data drive --model m2d --out runs/thin2d
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

DEFAULTS = {"drive": dict(crop=256, steps=4000), "chase": dict(crop=256, steps=4000),
            "deepcrack": dict(crop=256, steps=6000), "roads": dict(crop=256, steps=10000)}


# ----------------------------------------------------------------------------- data
def load_split(root, name, split):
    d = Path(root) / name / split
    items = []
    for p in sorted((d / "img").glob("*.png")):
        img = np.asarray(Image.open(p).convert("RGB"), np.uint8)
        lab = np.asarray(Image.open(d / "lab" / p.name)) > 127
        fov = (np.asarray(Image.open(d / "fov" / p.name)) > 127) if (d / "fov" / p.name).exists() else None
        items.append((p.stem, img, lab, fov))
    return items



def occlude(img, lab, centres, R, rng):
    """Erase the structure inside discs of radius R around the given centre-line pixels: structure pixels (label
    dilated by 2) in a disc are replaced by the mean colour of the disc's background pixels plus noise of their
    standard deviation. The label is unchanged, so the hidden piece must be inferred from the visible structure.
    Returns the image and the mask of erased pixels."""
    from scipy import ndimage
    H, W = lab.shape
    img = img.astype(np.float32).copy()
    struct = ndimage.binary_dilation(lab, iterations=2)
    erased = np.zeros_like(lab)
    yy, xx = np.ogrid[:H, :W]
    for cy, cx in centres:
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= R * R
        bg = disc & ~struct
        hole = disc & struct
        if bg.sum() < 4 or not hole.any():
            continue
        mu, sd = img[bg].mean(0), img[bg].std(0)
        img[hole] = mu + rng.standard_normal((hole.sum(), img.shape[2])) * sd
        erased |= hole
    return np.clip(img, 0, 255), erased

def sample_batch(items, B, crop, rng, dev, occ=0, R=10, skels=None):
    xs, ys = [], []
    for _ in range(B):
        i = rng.integers(len(items))
        _, img, lab, fov = items[i]
        H, W = lab.shape
        for _ in range(20):                                   # crops mostly inside the field of view
            y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            if fov is None or fov[y0 + crop // 2, x0 + crop // 2]:
                break
        x = img[y0:y0 + crop, x0:x0 + crop]
        y = lab[y0:y0 + crop, x0:x0 + crop]
        if occ:                                               # hide `occ` pieces of the structure
            sk = np.argwhere(skels[i][y0:y0 + crop, x0:x0 + crop])
            if len(sk):
                x, _ = occlude(x, y, sk[rng.integers(len(sk), size=occ)], R, rng)
        x = x.astype(np.float32) / 255
        y = y.astype(np.float32)
        k = rng.integers(8)                                   # dihedral group: rotations and transposes
        x, y = np.rot90(x, k % 4), np.rot90(y, k % 4)
        if k >= 4:
            x, y = x.transpose(1, 0, 2), y.T
        x = np.clip((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1), 0, 1)
        xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev)
    return (x - 0.5) / 0.25, torch.from_numpy(np.stack(ys)).to(dev)


# ----------------------------------------------------------------------------- network
class Norm2d(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, C, 1, 1))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-5) * self.w


class Block(nn.Module):
    def __init__(self, C, mixer):
        super().__init__()
        self.n1, self.mix, self.n2 = Norm2d(C), mixer, Norm2d(C)
        self.ffn = nn.Sequential(nn.Conv2d(C, 2 * C, 1), nn.GELU(), nn.Conv2d(2 * C, C, 1))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        return x + self.ffn(self.n2(x))


def make_mixer(kind, C, n, layer):
    import zoo2d_ext
    if kind in zoo2d_ext.KINDS:
        return zoo2d_ext.build_mixer(kind, C, n, layer)
    import zoo2d
    if kind == "vmamba":
        return zoo2d.VMambaSS2D(C)
    if kind == "rowcol":
        return zoo2d.RowColMamba(C)
    if kind == "vssd":
        return zoo2d.NonCausalSSD(C)
    if kind == "conv":
        return nn.Sequential(nn.Conv2d(C, C, 3, padding=1, groups=C), nn.GELU(), nn.Conv2d(C, C, 1))
    raise ValueError(kind)


def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GELU(), nn.Conv2d(co, co, 3, padding=1), nn.GELU())


class ThinNet(nn.Module):
    def __init__(self, kind, C=32, L=2, crop=256, out=1):
        super().__init__()
        self.out = out
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU())
        widths, chans, layer = [C, 2 * C, 4 * C], 16, 0
        self.down, self.stages = nn.ModuleList(), nn.ModuleList()
        for s, w in enumerate(widths):
            self.down.append(nn.Conv2d(chans, w, 2, stride=2))
            n = crop // 2 ** (s + 1)
            self.stages.append(nn.Sequential(*[Block(w, make_mixer(kind, w, n, i)) for i in range(L)]))
            chans = w
        self.up = nn.ModuleList([conv_block(widths[2] + widths[1], widths[1]), conv_block(widths[1] + widths[0], widths[0]),
                                 conv_block(widths[0] + 16, 16)])
        self.head = nn.Conv2d(16, out, 1)

    def forward(self, x):
        f0 = self.stem(x)
        feats, h = [], f0
        for d, st in zip(self.down, self.stages):
            h = st(d(h))
            feats.append(h)
        h = feats[2]
        for up, skip in zip(self.up, (feats[1], feats[0], f0)):
            h = up(torch.cat([F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        logits = self.head(h)
        return logits[:, 0] if self.out == 1 else logits



class PlainNet(nn.Module):
    """Single resolution: stride-2 conv stem, L mixer blocks at 1/2 resolution (no further downsampling, as in plain
    vision SSMs such as Vim / PlainMamba and in cost-volume aggregators), bilinear upsampling with a full-resolution
    skip, two 3x3 convs, logits."""

    def __init__(self, kind, C=64, L=6, crop=256):
        super().__init__()
        self.full = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU())
        self.stem = nn.Sequential(nn.Conv2d(3, C, 3, stride=2, padding=1), nn.GELU())
        self.blocks = nn.Sequential(*[Block(C, make_mixer(kind, C, crop // 2, i)) for i in range(L)])
        self.up = conv_block(C + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        f = self.full(x)
        h = F.interpolate(h, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up(torch.cat([h, f], 1)))[:, 0]

# ----------------------------------------------------------------------------- evaluation
def predict(net, img, crop, dev):
    """sliding windows of size crop, stride crop // 2, probabilities averaged"""
    H, W = img.shape[:2]
    s = crop // 2
    Hp, Wp = max(crop, -(-(H - crop) // s) * s + crop), max(crop, -(-(W - crop) // s) * s + crop)
    x = np.zeros((Hp, Wp, 3), np.float32)
    x[:H, :W] = img.astype(np.float32) / 255
    x = (torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) - 0.5) / 0.25
    prob = torch.zeros(Hp, Wp, device=dev)
    cnt = torch.zeros(Hp, Wp, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for y0 in range(0, Hp - crop + 1, s):
            for x0 in range(0, Wp - crop + 1, s):
                p = torch.sigmoid(net(x[:, :, y0:y0 + crop, x0:x0 + crop]).float())[0]
                prob[y0:y0 + crop, x0:x0 + crop] += p
                cnt[y0:y0 + crop, x0:x0 + crop] += 1
    return (prob / cnt)[:H, :W].cpu().numpy()


def orientation_bins(skel):
    """Bin (0..3 for 0, 45, 90, 135 degrees; x right, y down) of every skeleton pixel from the second moments of the
    skeleton pixels in its 7x7 window."""
    from scipy import ndimage
    r = 3
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    s = skel.astype(np.float64)
    sxx = ndimage.convolve(s, dx ** 2, mode="constant")
    syy = ndimage.convolve(s, dy ** 2, mode="constant")
    sxy = ndimage.convolve(s, dx * dy, mode="constant")
    ang = np.degrees(0.5 * np.arctan2(2 * sxy, sxx - syy)) % 180        # principal direction of the local point cloud
    return (((ang + 22.5) // 45) % 4).astype(np.int64)


def evaluate(net, items, crop, dev):
    from skimage.morphology import skeletonize
    tp = fp = fn = 0
    cl, rec_hit, rec_tot = [], np.zeros(4), np.zeros(4)
    net.eval()
    for _, img, lab, fov in items:
        pred = predict(net, img, crop, dev) > 0.5
        m = fov if fov is not None else np.ones_like(lab)
        p, g = pred & m, lab & m
        tp += (p & g).sum(); fp += (p & ~g).sum(); fn += (~p & g).sum()
        sp, sg = skeletonize(p), skeletonize(g)
        tprec = (sp & g).sum() / max(sp.sum(), 1)
        tsens = (sg & p).sum() / max(sg.sum(), 1)
        cl.append(2 * tprec * tsens / max(tprec + tsens, 1e-9))
        b = orientation_bins(sg)
        for k in range(4):
            sk = sg & (b == k)
            rec_tot[k] += sk.sum(); rec_hit[k] += (sk & p).sum()
    net.train()
    return dict(f1=2 * tp / max(2 * tp + fp + fn, 1), iou=tp / max(tp + fp + fn, 1), cldice=float(np.mean(cl)),
                recall_by_orientation={a: float(rec_hit[k] / max(rec_tot[k], 1)) for k, a in enumerate((0, 45, 90, 135))},
                centreline_pixels_by_orientation={a: int(rec_tot[k]) for k, a in enumerate((0, 45, 90, 135))})



def evaluate_occluded(net, items, crop, dev, R, per_image=30, seed=123):
    """Bridging: in every test image hide the structure in up to per_image discs (radius R, centres on the centre
    line, at least 2R apart, fixed seed) and measure the recall of the hidden centre-line pixels, by the orientation
    of the centre line at the disc centre."""
    from skimage.morphology import skeletonize
    rng = np.random.default_rng(seed)
    hit, tot, pos, pos_ok = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    net.eval()
    for _, img, lab, fov in items:
        sg = skeletonize(lab & (fov if fov is not None else True))
        cand = np.argwhere(sg)
        if not len(cand):
            continue
        b = orientation_bins(sg)
        rng.shuffle(cand)
        centres = []
        for c in cand:
            if all((c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2 >= (2 * R) ** 2 for d in centres):
                centres.append(c)
            if len(centres) == per_image:
                break
        occ_img, _ = occlude(img, lab, centres, R, rng)
        pred = predict(net, occ_img.astype(np.uint8), crop, dev) > 0.5
        yy, xx = np.ogrid[:lab.shape[0], :lab.shape[1]]
        for cy, cx in centres:
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= R * R
            k = b[cy, cx]
            hidden = disc & sg
            tot[k] += hidden.sum(); hit[k] += (hidden & pred).sum()
            pos[k] += (disc & pred).sum(); pos_ok[k] += (disc & pred & lab).sum()
    net.train()
    angles = (0, 45, 90, 135)
    return dict(bridging_recall_by_orientation={a: float(hit[k] / max(tot[k], 1)) for k, a in enumerate(angles)},
                bridging_precision_by_orientation={a: float(pos_ok[k] / max(pos[k], 1)) for k, a in enumerate(angles)},
                bridging_pixels_by_orientation={a: int(tot[k]) for k, a in enumerate(angles)},
                bridging_recall=float(hit.sum() / max(tot.sum(), 1)),
                bridging_precision=float(pos_ok.sum() / max(pos.sum(), 1)))

# ----------------------------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, choices=list(DEFAULTS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default="/data/datasets/thin2d")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="unet", choices=["unet", "plain"])
    ap.add_argument("--occlude", type=int, default=0, help="pieces of structure hidden per training crop (0 = off)")
    ap.add_argument("--occ_r", type=int, default=10, help="radius of a hidden piece (pixels)")
    ap.add_argument("--out", default="runs/thin2d")
    args = ap.parse_args()
    crop = args.crop or DEFAULTS[args.data]["crop"]
    steps = args.steps or DEFAULTS[args.data]["steps"]
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.data}_{args.model}_C{args.C}_L{args.L}" + ("_plain" if args.arch == "plain" else "") \
        + (f"_occ{args.occlude}r{args.occ_r}" if args.occlude else "") \
        + (f"_seed{args.seed}" if args.seed else "")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, test = load_split(args.root, args.data, "train"), load_split(args.root, args.data, "test")
    skels = None
    if args.occlude:
        from skimage.morphology import skeletonize
        skels = [skeletonize(lab) for _, _, lab, _ in train]
    net = (PlainNet(args.model, args.C, args.L, crop) if args.arch == "plain" else ThinNet(args.model, args.C, args.L, crop)).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(train)} train / {len(test)} test images, crop {crop}, {steps} steps",
          file=log, flush=True)
    t0, skipped = time.time(), 0
    for it in range(1, steps + 1):
        x, y = sample_batch(train, args.batch, crop, rng, dev, args.occlude, args.occ_r, skels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(x).float()
        p = torch.sigmoid(logit)
        dice = 1 - (2 * (p * y).sum((1, 2)) + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1)
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
    m = evaluate(net, test, crop, dev)
    m.update(evaluate_occluded(net, test, crop, dev, args.occ_r))
    r = dict(name=name, data=args.data, model=args.model, params_k=nparam / 1e3, train_seconds=train_s,
             skipped_steps=skipped, **m, args=vars(args))
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
