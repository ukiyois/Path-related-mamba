"""HiPaS pulmonary arteries and veins on non-contrast CT (Chu et al., Nat. Commun. 2025; Zenodo record 14879605: 250 scans,
artery and vein masks) -> isotropic 1.6 mm volumes cropped around the vessels (npz: img float16 in [0, 1] (lung window),
art and vein uint8), and a fixed 200 / 50 train / test split.

  python explore3d/data/prep_hipas.py /data/datasets/hipas/raw /data/datasets/hipas
(raw holds ct_scan.zip, annotation.zip and spacing.csv with case,sx,sy,sz in mm, taken from the record's metadata.xlsx)
"""
import csv
import io
import json
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SPACING = 1.6               # mm, isotropic
HU = (-1000.0, 400.0)       # lung window


def read(zpath, member):
    with zipfile.ZipFile(zpath) as z:
        return np.load(io.BytesIO(z.read(member)))["data"]


def one(args):
    case, sp, ct_member, art_member, vein_member, raw, out = args
    from scipy import ndimage
    zoom = np.asarray(sp, np.float64) / SPACING
    img = ndimage.zoom(np.clip(read(raw / "ct_scan.zip", ct_member).astype(np.float32), *HU), zoom, order=1)
    art = ndimage.zoom(read(raw / "annotation.zip", art_member).astype(np.float32), zoom, order=1) > 0.5
    vein = ndimage.zoom(read(raw / "annotation.zip", vein_member).astype(np.float32), zoom, order=1) > 0.5
    nz = np.argwhere(ndimage.binary_dilation(art | vein, iterations=8))
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img = ((img[sl] - HU[0]) / (HU[1] - HU[0])).astype(np.float16)
    np.savez_compressed(out / f"{case}.npz", img=img, art=art[sl].astype(np.uint8), vein=vein[sl].astype(np.uint8))
    return case, img.shape, int(art[sl].sum()), int(vein[sl].sum())


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw / "ct_scan.zip") as z:
        ct = {Path(n).stem: n for n in z.namelist() if n.endswith(".npz")}
    with zipfile.ZipFile(raw / "annotation.zip") as z:
        ann = {(Path(n).parent.name, Path(n).stem): n for n in z.namelist() if n.endswith(".npz")}
    spacing = {r["case"]: (float(r["sx"]), float(r["sy"]), float(r["sz"]))
               for r in csv.DictReader(open(raw / "spacing.csv"))}
    cases = sorted(c for c in spacing if c in ct and ("artery", c) in ann and ("vein", c) in ann)
    jobs = [(c, spacing[c], ct[c], ann[("artery", c)], ann[("vein", c)], raw, out) for c in cases]
    with ProcessPoolExecutor(16) as ex:
        for case, shape, na, nv in ex.map(one, jobs):
            print(case, shape, na, nv, flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(cases))
    split = dict(train=sorted(cases[i] for i in perm[:200]), test=sorted(cases[i] for i in perm[200:]))
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(len(split["train"]), "train /", len(split["test"]), "test")
