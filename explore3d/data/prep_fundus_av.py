"""Retinal arteries and veins: the multi-disease fundus segmentation dataset (figshare 10.6084/m9.figshare.32646378,
CC BY 4.0; 7,720 image IDs with vessel, artery-vein and optic disc/cup masks, 26 disease archives). Only images released
with the record are used (5,772; the others would have to be fetched from their sources), minus those flagged AV_Hard
(severe pathology, uncorrected AI masks). Every image and its masks are resized to 512 x 512 (they are already square,
zero-padded) and stored as npz: img uint8 RGB, art / vein / disc uint8, and the route geometry of each vessel pixel from
the optic disc along its own tree (4-connected, mask dilated by one pixel so that diagonal one-pixel vessels connect):
axis turns, orthant switches and length in pixels (int16, -1 where the disc is not reached). Split 80 / 20 per archive.

  python explore3d/data/prep_fundus_av.py /data/datasets/fundus_av/raw /data/datasets/fundus_av
(raw holds the 26 archives and av_hard.txt, the IDs flagged AV_Hard in the record's hard_cases.xlsx)
"""
import io
import json
import sys
import zipfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SIZE = 512
NEWS = [(-1, 0), (0, 1), (0, -1), (1, 0)]
ORTHANTS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
AXIS_MOVES = [(0, d, 0) for d in (-1, 1)] + [(d, 0, 1) for d in (-1, 1)]
ORTHANT_MOVES = [(dy, dx, t) for dy, dx in NEWS for t, (sy, sx) in enumerate(ORTHANTS)
                 if dy in (0, sy) and dx in (0, sx)]


def zero_one_bfs(domain, sources, n_states, moves):
    """Fewest state changes on 4-connected routes inside domain from any source (as in citymap.py)."""
    H, W = domain.shape
    inf = np.iinfo(np.int32).max
    cost = np.full((n_states, H, W), inf, np.int32)
    dq = deque()
    for y, x in sources:
        for s in range(n_states):
            cost[s, y, x] = 0
            dq.append((0, s, y, x))
    while dq:
        c, s, y, x = dq.popleft()
        if c > cost[s, y, x]:
            continue
        for dy, dx, t in moves:
            yy, xx = y + dy, x + dx
            nc = c + (t != s)
            if 0 <= yy < H and 0 <= xx < W and domain[yy, xx] and nc < cost[t, yy, xx]:
                cost[t, yy, xx] = nc
                (dq.appendleft if nc == c else dq.append)((nc, t, yy, xx))
    best = cost.min(0)
    return np.where(best == inf, -1, best).astype(np.int16)


def path_length(domain, sources):
    H, W = domain.shape
    dist = np.full((H, W), -1, np.int16)
    dq = deque()
    for y, x in sources:
        dist[y, x] = 0
        dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for dy, dx in NEWS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and domain[yy, xx] and dist[yy, xx] < 0:
                dist[yy, xx] = dist[y, x] + 1
                dq.append((yy, xx))
    return dist


def geometry(mask, disc):
    """(turns, switches, length) from the optic disc along this tree; roots = tree pixels within 8 px of the disc."""
    from scipy import ndimage
    domain = ndimage.binary_dilation(mask)
    roots = [tuple(p) for p in np.argwhere(domain & ndimage.binary_dilation(disc, iterations=8))]
    if not roots:
        neg = np.full(mask.shape, -1, np.int16)
        return neg, neg, neg
    return (zero_one_bfs(domain, roots, 2, AXIS_MOVES), zero_one_bfs(domain, roots, 4, ORTHANT_MOVES),
            path_length(domain, roots))


def one(args):
    zpath, arch, image_id, out = args
    from PIL import Image
    with zipfile.ZipFile(zpath) as z:
        read = lambda folder: Image.open(io.BytesIO(z.read(f"{arch}/{folder}/{image_id}.png")))
        img = np.asarray(read("image").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
        av = np.asarray(read("artery_vein_mask").convert("RGB").resize((SIZE, SIZE), Image.NEAREST))
        dc = np.asarray(read("disc_cup_mask").convert("RGB").resize((SIZE, SIZE), Image.NEAREST))
    art, vein, disc = av[..., 0] > 127, av[..., 2] > 127, dc.max(-1) > 127
    ta, sa, la = geometry(art, disc)
    tv, sv, lv = geometry(vein, disc)
    np.savez_compressed(out / f"{image_id}.npz", img=img, art=art.astype(np.uint8), vein=vein.astype(np.uint8),
                        disc=disc.astype(np.uint8), turns_art=ta, switches_art=sa, length_art=la,
                        turns_vein=tv, switches_vein=sv, length_vein=lv)
    return image_id, int(art.sum()), int(vein.sum()), int(disc.sum())


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    hard = set((raw / "av_hard.txt").read_text().split())
    jobs = []
    for zpath in sorted(raw.glob("*.zip")):
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                p = Path(n)
                if p.parent.name == "image" and p.suffix == ".png" and p.stem not in hard:
                    jobs.append((zpath, p.parent.parent.name, p.stem, out))
    print(len(jobs), "images", flush=True)
    done = []
    with ProcessPoolExecutor(32) as ex:
        for image_id, na, nv, nd in ex.map(one, jobs, chunksize=4):
            done.append(image_id)
            if nd == 0 or na == 0 or nv == 0:
                print("check", image_id, na, nv, nd, flush=True)
    rng = np.random.default_rng(0)
    split = dict(train=[], test=[])
    for arch in sorted({j[1] for j in jobs}):
        ids = sorted(j[2] for j in jobs if j[1] == arch)
        perm = rng.permutation(len(ids))
        k = int(round(0.8 * len(ids)))
        split["train"] += [ids[i] for i in perm[:k]]
        split["test"] += [ids[i] for i in perm[k:]]
    split = {k: sorted(v) for k, v in split.items()}
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(len(split["train"]), "train /", len(split["test"]), "test")
