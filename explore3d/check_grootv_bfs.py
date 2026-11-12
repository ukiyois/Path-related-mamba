"""GrootV's BFS op on this GPU: (sorted_index, sorted_parent) must be a breadth-first order of the minimum spanning tree
returned by its MST op (root = vertex 0, every vertex once, each parent before its children, parent-child pairs = the
tree edges). Checked on random features and on corridor-like features with many tied edge weights.

  python explore3d/check_grootv_bfs.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "explore3d")]

import numpy as np
import torch

import zoo2d_ext

zoo2d_ext.grootv(8)
ts = sys.modules["grootv_models.tree_scanning"]
core = sys.modules["grootv_models.tree_scan_utils.tree_scan_core"]


def check(fea, tag):
    B, _, H, W = fea.shape
    tree = core.MinimumSpanningTree("Cosine", torch.exp)(fea)            # (B, L-1, 2) vertex pairs
    idx, par, _ = ts._BFS.apply(tree, 4)
    bad = 0
    for b in range(B):
        edges = {tuple(sorted(e)) for e in tree[b].long().tolist()}
        order, parent = idx[b].long().tolist(), par[b].long().tolist()
        ok = (order[0] == 0 and sorted(order) == list(range(H * W)) and len(edges) == H * W - 1)
        pos = {v: i for i, v in enumerate(order)}
        got = set()
        for i in range(1, H * W):
            pv = order[parent[i]] if 0 <= parent[i] < H * W else -1
            if pv < 0 or pos[pv] >= i:
                ok = False
                break
            got.add(tuple(sorted((order[i], pv))))
        ok = ok and got == edges
        bad += not ok
    print(f"{tag}: {B - bad}/{B} samples give a valid BFS of the MST", flush=True)


torch.manual_seed(0)
check(torch.randn(4, 8, 16, 16, device="cuda"), "random features 16x16")
check(torch.randn(4, 8, 64, 64, device="cuda"), "random features 64x64")
import corridor
for orient in ("main", "anti"):
    mask, src, _, _ = corridor.corridors(4, 64, np.random.default_rng(5), orient=orient)
    fea = torch.stack([mask, src, torch.ones_like(mask)], 1).cuda() + 1e-3 * torch.randn(4, 3, 64, 64, device="cuda")
    check(fea, f"corridors ({orient}), near-tied weights")
    check(torch.stack([mask, src, torch.ones_like(mask)], 1).cuda(), f"corridors ({orient}), exact ties")
