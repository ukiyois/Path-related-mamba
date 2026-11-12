"""Zero-shot evaluation of SceneFlow-trained stereo networks (explore3d/sceneflow.py checkpoints).

ETH3D two-view training (27 scenes): EPE and bad-1.0 on non-occluded pixels.
Middlebury 2014 training, half resolution (15 scenes): EPE and bad-2.0 on non-occluded pixels, the usual
zero-shot protocol (predictions capped at max_disp 192, as in training).
Data: realdata/eth3d/<scene>/{im0,im1}.png, disp0GT.pfm, mask0nocc.png; realdata/mb/MiddEval3/trainingH/...

  python explore3d/zeroshot.py --ckpt runs/e6/routed_s20000_C64.pt --model routed --C 64
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "explore3d")]

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sceneflow import read_pfm

REAL = ROOT / "realdata"


def load_pair(d):
    def rgb(p):
        a = np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
        return torch.from_numpy(a).permute(2, 0, 1)
    gt = read_pfm(d / "disp0GT.pfm").astype(np.float32)
    nocc = np.asarray(Image.open(d / "mask0nocc.png")) == 255
    return rgb(d / "im0.png"), rgb(d / "im1.png"), torch.from_numpy(gt), torch.from_numpy(nocc)


def predict(net, L, R, dev):
    H, W = L.shape[-2:]
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    Lp = F.pad(L[None].to(dev), (0, pw, ph, 0), mode="replicate")
    Rp = F.pad(R[None].to(dev), (0, pw, ph, 0), mode="replicate")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return net(Lp, Rp).float()[0, ph:, :W].cpu()


def run(net, scenes, thr, dev):
    epe = bad = n = 0.0
    per = {}
    for d in scenes:
        L, R, gt, nocc = load_pair(d)
        pred = predict(net, L, R, dev)
        valid = nocc & torch.isfinite(gt) & (gt > 0)
        err = (pred - gt).abs()[valid]
        epe += err.sum().item(); bad += (err > thr).sum().item(); n += valid.sum().item()
        per[d.name] = round((err > thr).float().mean().item() * 100, 2)
    return dict(epe=epe / n, bad=bad / n * 100, per_scene=per)


def main():
    from flying_planes import StereoNet
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--max_disp", type=int, default=192)
    ap.add_argument("--out", default="runs/zeroshot")
    args = ap.parse_args()
    dev = "cuda"
    net = StereoNet(args.model, args.max_disp, args.C, 4, 256, 512).to(dev)
    sd = torch.load(args.ckpt, map_location=dev, weights_only=False)["net"]
    sd = {k.replace("body.seq.", "body."): v for k, v in sd.items()}  # checkpointed-body wrapper
    net.load_state_dict(sd)
    net.eval()
    eth = sorted(p for p in (REAL / "eth3d").iterdir() if (p / "disp0GT.pfm").exists())
    mbh = sorted(p for p in (REAL / "mb" / "MiddEval3" / "trainingH").iterdir() if (p / "disp0GT.pfm").exists())
    r = dict(ckpt=args.ckpt, model=args.model, C=args.C,
             eth3d=run(net, eth, 1.0, dev), middlebury_half=run(net, mbh, 2.0, dev))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / (Path(args.ckpt).stem + ".json"), "w") as f:
        json.dump(r, f, indent=1)
    print(f"{Path(args.ckpt).stem}: ETH3D EPE {r['eth3d']['epe']:.3f} bad1 {r['eth3d']['bad']:.2f}% | "
          f"Middlebury-H EPE {r['middlebury_half']['epe']:.3f} bad2 {r['middlebury_half']['bad']:.2f}%")


if __name__ == "__main__":
    main()
