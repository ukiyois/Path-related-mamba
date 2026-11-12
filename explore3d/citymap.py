"""Shortest paths on real city maps: a real task in which the confinement assumed by the depth bounds holds.

Data: the CSM benchmark of Neural A* (Yonetani et al., ICML 2021), the authors' own file
(planning-datasets, data/street/mixed_064_moore_c16.npz): MovingAI street maps of real cities (Sturtevant 2012)
cut into 128 x 128 crops and resized to 64 x 64; 3200 / 400 / 400 maps for training / validation / test (the
test crops come from maps not used for training); one goal per map in a 16-cell corner region; Dijkstra distance
to the goal over the eight neighbours with unit step cost.

Walls block every route, so the distance at a start depends on the goal only through free cells. The network sees
free space and the goal and outputs the distance field in one pass; it is the single-resolution stack of the
corridor experiments (corridor.build), so the same mixers run on synthetic corridors and on real streets.

Read-out and metrics as in Neural A* and GPPN: from start cells drawn as in Neural A* (distance bands 55-70 /
70-85 / 85-100 %; 6 per validation map, 15 per test map) the path goes downhill on the predicted field over the
eight neighbours without revisiting a cell; Suc = the goal is reached, Opt = it is reached in the optimal number of
steps. Both are also reported by the geometry of the best 4-connected route from the goal to the start: axis turns
(rectilinear link distance, Thm 1) and orthant switches (monotone pieces - 1, Thm 5); so is the error of the predicted
distance at every reachable cell, by that cell's route geometry (the read-out above can recover from a wrong start
value once the walk gets close to the goal; the field itself cannot).

  python explore3d/citymap.py --model roesser_b0 --L 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "explore3d")]

import numpy as np
import torch

from corridor import build

FAR = 4096                                       # distance stored for walls and unreachable cells
MOORE = [(-1, 0), (0, 1), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)]
NEWS = [(-1, 0), (0, 1), (0, -1), (1, 0)]
ORTHANTS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
BANDS = np.array([0.55, 0.70, 0.85, 1.0])        # Neural A* start bands: 55-70 / 70-85 / 85-100 % of the distances
STARTS = {1: 6, 2: 15}                           # start cells per validation / test map
TURN_BINS = [0, 1, 2, 3, 4, 5, 6]                # the last bin holds 6 or more
SWITCH_BINS = [0, 1, 2, 3]                       # the last bin holds 3 or more


def load(path):
    """[(free, goal, dist)] for train, validation, test; free 1 = street, goal one-hot, dist in steps (FAR = none)."""
    z = np.load(path)
    splits = []
    for k in range(3):
        free = z[f"arr_{4 * k}"].astype(np.float32)
        goal = z[f"arr_{4 * k + 1}"][:, 0].astype(np.float32)
        dist = -z[f"arr_{4 * k + 3}"][:, 0].astype(np.float32)
        splits.append((free, goal, dist))
    return splits


def draw_starts(dist, k, rng):
    """k start cells as in Neural A* (get_random_start_map): one of three bands of the distance distribution over
    the reachable cells, then a cell in that band."""
    v = -dist.ravel()
    th = np.percentile(v[v > v.min()], 100 * (1 - BANDS))
    out = []
    for _ in range(k):
        r = rng.integers(0, len(th) - 1)
        cand = np.flatnonzero((v >= th[r + 1]) & (v <= th[r]))
        if not len(cand):                            # band between two integer distances: widen to both
            cand = np.flatnonzero((v >= np.floor(th[r + 1])) & (v <= np.ceil(th[r])))
        out.append(divmod(int(rng.choice(cand)), dist.shape[1]))
    return out


AXIS_MOVES = [(0, d, 0) for d in (-1, 1)] + [(d, 0, 1) for d in (-1, 1)]   # states: 0 = along x, 1 = along y
ORTHANT_MOVES = [(dy, dx, t) for dy, dx in NEWS for t, (sy, sx) in enumerate(ORTHANTS)
                 if dy in (0, sy) and dx in (0, sx)]                       # states: orthants matching the move's signs


def _zero_one_bfs(free, goal, n_states, moves):
    """Fewest state changes on 4-connected routes from the goal; a move (dy, dx, t) enters state t and costs 1 when
    t differs from the current state. Returns the minimum over states (-1 where no route exists)."""
    H, W = free.shape
    inf = np.iinfo(np.int32).max
    cost = np.full((n_states, H, W), inf, np.int32)
    dq = deque()
    for s in range(n_states):
        cost[s][goal] = 0
        dq.append((0, s) + goal)
    while dq:
        c, s, y, x = dq.popleft()
        if c > cost[s, y, x]:
            continue
        for dy, dx, t in moves:
            yy, xx = y + dy, x + dx
            nc = c + (t != s)
            if 0 <= yy < H and 0 <= xx < W and free[yy, xx] and nc < cost[t, yy, xx]:
                cost[t, yy, xx] = nc
                (dq.appendleft if nc == c else dq.append)((nc, t, yy, xx))
    best = cost.min(0)
    return np.where(best == inf, -1, best).astype(np.int16)


def route_geometry(free, goal):
    """(axis turns, orthant switches) of the best 4-connected route from the goal to every cell."""
    return _zero_one_bfs(free, goal, 2, AXIS_MOVES), _zero_one_bfs(free, goal, 4, ORTHANT_MOVES)


def geometry_cache(path, splits):
    """Route geometry and start cells for validation and test, computed once next to the data file."""
    cache = Path(path).with_name("citymap_eval_v1.npz")
    if not cache.exists():
        arrs = {}
        for k in (1, 2):
            free, goal, dist = splits[k]
            starts, turns, switches = [], [], []
            for i in range(len(free)):
                g = tuple(int(v) for v in np.argwhere(goal[i] > 0)[0])
                starts.append(draw_starts(dist[i], STARTS[k], np.random.default_rng(1372 + i)))
                a, o = route_geometry(free[i] > 0, g)
                turns.append(a); switches.append(o)
            arrs[f"starts{k}"] = np.array(starts, np.int16)
            arrs[f"turns{k}"] = np.stack(turns)
            arrs[f"switches{k}"] = np.stack(switches)
        tmp = cache.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, cache)
    z = np.load(cache)
    return {k: (z[f"starts{k}"], z[f"turns{k}"], z[f"switches{k}"]) for k in (1, 2)}


def rollout(pred, free, start, goal, max_steps=256):
    """Greedy descent on the predicted distance over the eight neighbours, never revisiting a cell."""
    H, W = free.shape
    y, x = start
    seen = {start}
    for n in range(max_steps):
        if (y, x) == goal:
            return True, n
        best, bv = None, None
        for dy, dx in MOORE:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and free[yy, xx] and (yy, xx) not in seen \
                    and (bv is None or pred[yy, xx] < bv):
                best, bv = (yy, xx), pred[yy, xx]
        if best is None:
            return False, n
        y, x = best
        seen.add(best)
    return (y, x) == goal, max_steps


def summarise(rows, col, bins):
    out = {}
    for b in bins + [-1]:
        if b == -1:
            sel = rows[:, col] == -1                # no 4-connected route (only diagonal squeezes)
        elif b == bins[-1]:
            sel = rows[:, col] >= b
        else:
            sel = rows[:, col] == b
        if sel.any():
            out[str(b) + ("+" if b == bins[-1] else "")] = dict(n=int(sel.sum()), suc=float(rows[sel, 0].mean()),
                                                               opt=float(rows[sel, 1].mean()))
    return out


def mae_by(err, geo, valid, bins):
    """Distance error over every reachable cell, by the cell's route geometry (the theorems' variable)."""
    out = {}
    for b in bins + [-1]:
        sel = valid & ((geo >= b) if b == bins[-1] else (geo == b))
        if sel.any():
            out[str(b) + ("+" if b == bins[-1] else "")] = dict(n=int(sel.sum()), mae=float(err[sel].mean()))
    return out


def evaluate(net, split, ev, dev, batch=50):
    """Metrics, and the per-query rows (suc, opt, turns, switches, optimal length) for later analysis."""
    free, goal, dist = split
    starts, turns, switches = ev
    preds = []
    with torch.no_grad():
        for i in range(0, len(free), batch):
            x = torch.from_numpy(np.stack([free[i:i + batch], goal[i:i + batch]], 1)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds.append(net(x)[:, 0].float().cpu().numpy() * 64)
    pred = np.concatenate(preds)
    valid = dist < FAR
    err = np.abs(pred - dist)
    rows = []
    for i in range(len(free)):
        g = tuple(int(v) for v in np.argwhere(goal[i] > 0)[0])
        for s in map(tuple, starts[i].tolist()):
            suc, n = rollout(pred[i], free[i] > 0, s, g)
            rows.append((suc, suc and n == int(dist[i][s]), turns[i][s], switches[i][s], int(dist[i][s])))
    rows = np.array(rows, np.int64)
    return dict(suc=float(rows[:, 0].mean()), opt=float(rows[:, 1].mean()), n=len(rows),
                mae=float(err[valid].mean()),
                by_turns=summarise(rows, 2, TURN_BINS), by_switches=summarise(rows, 3, SWITCH_BINS),
                mae_by_turns=mae_by(err, turns, valid, TURN_BINS),
                mae_by_switches=mae_by(err, switches, valid, SWITCH_BINS)), rows


def dihedral(t, k):
    t = torch.rot90(t, k % 4, (-2, -1))
    return t.transpose(-2, -1) if k >= 4 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--C", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/data/datasets/citymap/mixed_064_moore_c16.npz")
    ap.add_argument("--out", default="runs/citymap")
    args = ap.parse_args()
    dev = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"citymap_{args.model}_L{args.L}_C{args.C}" + (f"_seed{args.seed}" if args.seed else "")
    splits = load(args.data)
    ev = geometry_cache(args.data, splits)
    torch.manual_seed(args.seed)
    net = build(args.model, args.C, args.L, 64).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    free, goal, dist = (torch.from_numpy(a).to(dev) for a in splits[0])
    gen = np.random.default_rng(args.seed)
    log = open(out / f"{name}.log", "a")
    print(f"{name}: {nparam / 1e3:.1f}k params, {len(free)} training maps, {args.steps} steps", file=log, flush=True)
    t0 = time.time()
    skipped = 0
    for it in range(1, args.steps + 1):
        idx = gen.integers(0, len(free), args.batch)
        ks = gen.integers(0, 8, args.batch)          # dihedral group: rotations and transposes
        f = torch.stack([dihedral(free[i], k) for i, k in zip(idx, ks)])
        g = torch.stack([dihedral(goal[i], k) for i, k in zip(idx, ks)])
        d = torch.stack([dihedral(dist[i], k) for i, k in zip(idx, ks)])
        valid = d < FAR
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = net(torch.stack([f, g], 1))[:, 0].float()
        loss = (pred - d / 64)[valid].abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):                   # skip a non-finite step
            skipped += 1
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        sched.step()
        if it % 500 == 0:
            print(f"it {it} loss {loss.item():.5f} {time.time() - t0:.0f}s", file=log, flush=True)
    net.eval()
    (val, _), (test, rows) = evaluate(net, splits[1], ev[1], dev), evaluate(net, splits[2], ev[2], dev)
    np.savez_compressed(out / f"{name}_queries.npz", rows=rows,
                        columns=np.array(["suc", "opt", "turns", "switches", "length"]))
    r = dict(name=name, model=args.model, L=args.L, C=args.C, params_k=nparam / 1e3, steps=args.steps,
             seconds=time.time() - t0, skipped_steps=skipped, seed=args.seed, val=val, test=test)
    print(json.dumps(r), file=log, flush=True)
    with open(out / f"{name}.jsonl", "a") as fh:
        fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
