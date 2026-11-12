"""Exploration E2: thin oblique corridors (a test of the depth-separation hypothesis).

Each sample holds several disjoint 4-connected staircase corridors on an n x n grid
(translates of one random monotone path with runs of 1-2 cells, so every corridor
turns about once per two cells). The first cell of each corridor carries a random
value; the target is that value on the whole corridor and zero elsewhere. A model
must carry each value along its own staircase, around ~n turns, without leaking
into the neighbouring corridors.

Prediction: one coupled Roesser layer follows staircases of any length; without the
coupling (b = 0) or with axis-aligned scans, one layer moves information along rows
or columns only, so L layers reach about L turns. A single 1-D order (raster or
Hilbert) must keep every corridor's value in its state.

  python explore3d/corridor.py --model roesser --n 64 --L 2
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



def corridors(B, n, gen, spacing=6, max_run=2, orient="main"):
    """mask (B,n,n), source (B,n,n), target (B,n,n), far (B,n,n): far = cells in the second
    half of their corridor. orient: "main" (top-left to bottom-right), "anti" (bottom-left to
    top-right: every sample flipped upside down) or "mixed" (each sample either way)."""
    mask = np.zeros((B, n, n), np.float32)
    src = np.zeros((B, n, n), np.float32)
    tgt = np.zeros((B, n, n), np.float32)
    far = np.zeros((B, n, n), bool)
    for b in range(B):
        for _ in range(20):                         # resample until corridors do not touch
            path = [(0, 0)]
            i = j = 0
            horiz = bool(gen.integers(2))
            while i < 2 * n and j < 2 * n:
                run = int(gen.integers(1, max_run + 1))
                for _ in range(run):
                    j, i = (j + 1, i) if horiz else (j, i + 1)
                    path.append((i, j))
                horiz = not horiz
            P = np.array(path)
            lab = -np.ones((n, n), np.int64)
            ok = True
            k = 0
            for c in range(-2 * n, n, spacing):
                cells = P + np.array([0, c])
                cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < n) & (cells[:, 1] >= 0) & (cells[:, 1] < n)]
                if len(cells) < 4:
                    continue
                lab_k = np.zeros((n, n), bool)
                lab_k[cells[:, 0], cells[:, 1]] = True
                grown = lab_k.copy()
                grown[1:] |= lab_k[:-1]; grown[:-1] |= lab_k[1:]
                grown[:, 1:] |= lab_k[:, :-1]; grown[:, :-1] |= lab_k[:, 1:]
                if (lab[grown] >= 0).any():
                    ok = False
                    break
                lab[lab_k] = k
                v = float(gen.uniform(-1, 1))
                order = np.argsort(cells[:, 0] + cells[:, 1], kind="stable")
                cells = cells[order]
                src[b, cells[0, 0], cells[0, 1]] = v
                tgt[b, cells[:, 0], cells[:, 1]] = v
                half = cells[len(cells) // 2:]
                far[b, half[:, 0], half[:, 1]] = True
                k += 1
            if ok:
                mask[b] = (lab >= 0)
                break
            src[b] = 0; tgt[b] = 0; far[b] = False
        flip = orient == "anti" or (orient == "mixed" and gen.integers(2) == 1)
        if flip:
            for t in (mask, src, tgt, far):
                t[b] = t[b][::-1].copy()
    return [torch.from_numpy(t) for t in (mask, src, tgt, far)]


def kturn_corridors(B, n, gen, turns, orient="main", lmin=4, lmax=10, tries=300):
    """Corridors with exactly ``turns`` turns: turns + 1 axis-parallel segments of lmin..lmax cells,
    alternating axes and heading down/right, placed at random without touching (8-neighbourhood).
    Returns mask, src, tgt, far (the last segment) and seg (segment index of every corridor cell,
    counted from the source; -1 elsewhere). Segment s lies at rectilinear link distance s + 1 from
    the source, so an axis-aligned stack of L layers can reach segments 0..L-1 only (Thm 1)."""
    mask = np.zeros((B, n, n), np.float32)
    src = np.zeros((B, n, n), np.float32)
    tgt = np.zeros((B, n, n), np.float32)
    far = np.zeros((B, n, n), bool)
    seg = -np.ones((B, n, n), np.int64)
    for b in range(B):
        occ = np.zeros((n + 2, n + 2), bool)                     # padded occupancy incl. the 1-cell halo
        for _ in range(tries):
            lens = gen.integers(lmin, lmax + 1, turns + 1)
            horiz = bool(gen.integers(2))
            cells, sid, i, j = [(0, 0)], [0], 0, 0
            for k, run in enumerate(lens):
                for _ in range(int(run)):
                    i, j = (i, j + 1) if horiz else (i + 1, j)
                    cells.append((i, j)); sid.append(k)
                horiz = not horiz
            P, S = np.array(cells), np.array(sid)
            if P[:, 0].max() >= n or P[:, 1].max() >= n:
                continue
            Q = P + np.array([gen.integers(0, n - P[:, 0].max()), gen.integers(0, n - P[:, 1].max())])
            if occ[Q[:, 0] + 1, Q[:, 1] + 1].any():
                continue
            v = float(gen.uniform(-1, 1))
            mask[b, Q[:, 0], Q[:, 1]] = 1
            tgt[b, Q[:, 0], Q[:, 1]] = v
            src[b, Q[0, 0], Q[0, 1]] = v
            seg[b, Q[:, 0], Q[:, 1]] = S
            far[b, Q[S == turns, 0], Q[S == turns, 1]] = True
            for di in (0, 1, 2):
                for dj in (0, 1, 2):
                    occ[Q[:, 0] + di, Q[:, 1] + dj] = True
        flip = orient == "anti" or (orient == "mixed" and gen.integers(2) == 1)
        if flip:
            for t in (mask, src, tgt, far, seg):
                t[b] = t[b][::-1].copy()
    return [torch.from_numpy(t) for t in (mask, src, tgt, far, seg)]


class Mamba2D(nn.Module):
    def __init__(self, C, kind, n):
        super().__init__()
        from mamba_ssm import Mamba
        flat = torch.arange(n * n).view(n, n)
        if kind == "raster":
            orders = [flat.reshape(-1)]
        elif kind == "cross":                       # rows and columns, as in VMamba
            orders = [flat.reshape(-1), flat.t().reshape(-1)]
        elif kind == "hilbert":
            from syn_stereo import hilbert_index_3d as hilbert_index   # works for any number of axes
            bits = int(np.ceil(np.log2(n)))
            yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            h = hilbert_index(np.stack([yy.ravel(), xx.ravel()], 1), bits)
            orders = [torch.from_numpy(np.argsort(h, kind="stable"))]
        else:
            raise ValueError(kind)
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.f = nn.ModuleList(Mamba(d_model=C, d_state=16, d_conv=4, expand=2) for _ in orders)
        self.b = nn.ModuleList(Mamba(d_model=C, d_state=16, d_conv=4, expand=2) for _ in orders)
        for i, o in enumerate(orders):
            self.register_buffer(f"perm{i}", o, persistent=False)
        self.k = len(orders)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = self.dw(x).reshape(B, C, -1)
        out = torch.zeros_like(flat)
        for i in range(self.k):
            perm = getattr(self, f"perm{i}")
            seq = flat[:, :, perm].transpose(1, 2).contiguous()
            y = self.f[i](seq) + self.b[i](seq.flip(1)).flip(1)
            out[:, :, perm] = out[:, :, perm] + y.transpose(1, 2)
        return (out / self.k).reshape(B, C, H, W)


class ConvMix(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.pw = nn.Conv2d(C, C, 1)

    def forward(self, x):
        return self.pw(F.gelu(self.dw(x)))


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


def build(kind, C, L, n):
    def mixer(layer):
        import zoo2d_ext
        if kind in zoo2d_ext.KINDS:                # published mixers, official code (one per scan paradigm)
            return zoo2d_ext.build_mixer(kind, C, n, layer)
        if kind in ("roesser", "roesser_b0", "sum4", "roesser_o2"):   # not released yet (local only)
            from roesser_mixers import Roesser2D, RoesserOrthants2D
            return {"roesser": lambda: Roesser2D(C), "roesser_b0": lambda: Roesser2D(C, coupling=False),
                    "sum4": lambda: Roesser2D(C, variant="sum4"), "roesser_o2": lambda: RoesserOrthants2D(C)}[kind]()
        if kind in ("vmamba", "mamba2d", "rowcol", "vssd", "attn"):
            import zoo2d
            return {"vmamba": lambda: zoo2d.VMambaSS2D(C), "mamba2d": lambda: zoo2d.Mamba2DMixer(C),
                    "rowcol": lambda: zoo2d.RowColMamba(C), "vssd": lambda: zoo2d.NonCausalSSD(C),
                    "attn": lambda: zoo2d.Attention2D(C, n)}[kind]()
        if kind.startswith("mamba_"):
            return Mamba2D(C, kind[6:], n)
        if kind == "conv":
            return ConvMix(C)
        raise ValueError(kind)
    return nn.Sequential(nn.Conv2d(2, C, 1), *[Block(C, mixer(i)) for i in range(L)], nn.Conv2d(C, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--orient", default="main", choices=["main", "anti", "mixed"])
    ap.add_argument("--turns", type=int, default=-1, help="k-turn corridors (exactly this many turns)")
    ap.add_argument("--seed", type=int, default=0, help="initialisation and training-data seed (validation is fixed)")
    ap.add_argument("--aug", default="", choices=["", "flip"],
                    help="flip: random horizontal and vertical flip of every training sample (flip augmentation)")
    ap.add_argument("--eval_orients", default="", help="comma list, e.g. main,anti: report each test orientation")
    ap.add_argument("--out", default="runs/e2")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.model}_n{args.n}_L{args.L}" + ("" if args.orient == "main" else f"_{args.orient}")
    if args.turns >= 0:
        name += f"_K{args.turns}"
        sample = lambda B, g: kturn_corridors(B, args.n, g, args.turns, orient=args.orient)
    else:
        sample = lambda B, g: corridors(B, args.n, g, orient=args.orient) + [None]
    if args.aug:
        name += f"_aug{args.aug}"
    if args.seed:
        name += f"_seed{args.seed}"
    torch.manual_seed(args.seed)
    net = build(args.model, args.C, args.L, args.n).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    gen = np.random.default_rng(args.seed)
    val = [sample(args.batch, np.random.default_rng(1000 + i)) for i in range(4)]
    orients = [o for o in args.eval_orients.split(",") if o]
    val_by = {}
    for o in orients:                               # same seeds as val, one orientation each
        if args.turns >= 0:
            val_by[o] = [kturn_corridors(args.batch, args.n, np.random.default_rng(1000 + i), args.turns, orient=o)
                         for i in range(4)]
        else:
            val_by[o] = [corridors(args.batch, args.n, np.random.default_rng(1000 + i), orient=o) + [None]
                         for i in range(4)]
    t0 = time.time()
    skipped = 0
    log = open(out / f"{name}.log", "a")
    for it in range(1, args.steps + 1):
        mask, src, tgt, far = (t.to(dev) for t in sample(args.batch, gen)[:4])
        if args.aug == "flip":                     # independent random flips along each axis, per sample
            fy = torch.from_numpy(gen.integers(0, 2, args.batch).astype(bool)).to(dev).view(-1, 1, 1)
            fx = torch.from_numpy(gen.integers(0, 2, args.batch).astype(bool)).to(dev).view(-1, 1, 1)
            mask, src, tgt = (torch.where(fy, t.flip(1), t) for t in (mask, src, tgt))
            mask, src, tgt = (torch.where(fx, t.flip(2), t) for t in (mask, src, tgt))
        x = torch.stack([mask, src], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(x)[:, 0].float()
        w = 1.0 + 4.0 * mask                       # weight corridor cells
        loss = (w * (pred - tgt) ** 2).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):                 # skip a non-finite step (bf16 Mamba blew up once)
            skipped += 1
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        sched.step()
        if it % 500 == 0:
            print(f"it {it} loss {loss.item():.5f} {time.time() - t0:.0f}s", file=log, flush=True)
    net.eval()
    K = max(args.turns, 0) + 1

    def evaluate(vset):
        se_all, se_far, acc_far, n_all, n_far = 0.0, 0.0, 0.0, 0, 0
        se_seg, n_seg = np.zeros(K), np.zeros(K)
        with torch.no_grad():
            for mask, src, tgt, far, seg in vset:
                mask, src, tgt, far = (t.to(dev) for t in (mask, src, tgt, far))
                pred = net(torch.stack([mask, src], 1))[:, 0].float()
                m = mask > 0
                se_all += ((pred - tgt)[m] ** 2).sum().item(); n_all += m.sum().item()
                se_far += ((pred - tgt)[far] ** 2).sum().item(); n_far += far.sum().item()
                acc_far += ((pred - tgt)[far].abs() < 0.1).sum().item()
                if seg is not None:                   # squared error by segment index (link distance - 1)
                    seg = seg.to(dev)
                    for k in range(K):
                        sk = seg == k
                        se_seg[k] += ((pred - tgt)[sk] ** 2).sum().item(); n_seg[k] += sk.sum().item()
        return se_all, se_far, acc_far, n_all, n_far, se_seg, n_seg

    se_all, se_far, acc_far, n_all, n_far, se_seg, n_seg = evaluate(val)
    by_orient = {}
    for o, vset in val_by.items():
        a, f, c, na, nf, ss, ns = evaluate(vset)
        by_orient[o] = dict(mse_corridor=a / na, mse_far=f / nf, acc_far=c / nf)
        if args.turns >= 0:
            by_orient[o]["mse_seg"] = [round(x / max(y, 1), 5) for x, y in zip(ss, ns)]
    r = dict(name=name, model=args.model, n=args.n, L=args.L, params_k=nparam / 1e3,
             mse_corridor=se_all / n_all, mse_far=se_far / n_far, acc_far=acc_far / n_far,
             var_target=1 / 3, seconds=time.time() - t0, skipped_steps=skipped, seed=args.seed, aug=args.aug)
    if by_orient:
        r["by_orient"] = by_orient
    if args.turns >= 0:
        assert n_seg.min() > 0, "k-turn run without segment labels"
        r.update(turns=args.turns, mse_seg=[round(a / max(c, 1), 5) for a, c in zip(se_seg, n_seg)])
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
