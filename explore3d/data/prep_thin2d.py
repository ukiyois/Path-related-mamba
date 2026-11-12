"""Thin-structure datasets in one layout: <out>/<name>/<split>/{img,lab,fov}/<id>.png (lab, fov in {0, 255}).

  drive      DRIVE (HF Zomba copy): train = official training 21-40, test = official test 01-20; FOV from the image
  chase      CHASE_DB1 (HF Zomba copy, 14 / 14 split), first observer labels; FOV from the image
  deepcrack  DeepCrack (official repository zip): 300 train / 237 test
  roads      Massachusetts Roads (HF asolodin parquet): official 1108 train / 49 test (+14 validation kept as val)

  python explore3d/data/prep_thin2d.py /data/datasets /data/datasets/thin2d
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def save(out, split, name, img, lab, fov=None):
    for kind, arr in (("img", img), ("lab", lab), ("fov", fov)):
        if arr is None:
            continue
        d = out / split / kind
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(d / f"{name}.png")


def fov_from_image(img, thr=20):
    """Circular field of view of a fundus image: red channel above thr, largest component, eroded by 2 px."""
    m = img[..., 0] > thr
    lab, n = ndimage.label(m)
    if n > 1:
        m = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    m = ndimage.binary_fill_holes(m)
    return (ndimage.binary_erosion(m, iterations=2) * 255).astype(np.uint8)


def binary(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    return ((a > (0 if a.dtype == bool else 127)) * 255).astype(np.uint8)


def drive(src, out):
    for split, sub in (("train", "train"), ("test", "val")):
        for p in sorted((src / "drive" / sub / "input").iterdir()):
            img = np.asarray(Image.open(p).convert("RGB"))
            ld = src / "drive" / sub / "label"
            lp = ld / (p.stem + ".png") if (ld / (p.stem + ".png")).exists() else ld / (p.stem + "_manual1.png")
            lab = binary(Image.open(lp))
            save(out / "drive", split, p.stem, img, lab, fov_from_image(img))


def chase(src, out):
    for split, sub in (("train", "train"), ("test", "val")):
        for p in sorted((src / "chase" / sub / "input").iterdir()):
            img = np.asarray(Image.open(p).convert("RGB"))
            lab = binary(Image.open(src / "chase" / sub / "label" / (p.stem + "_1stHO.png")))
            save(out / "chase", split, p.stem, img, lab, fov_from_image(img))


def deepcrack(src, out):
    z = zipfile.ZipFile(src / "DeepCrack.zip")
    names = set(z.namelist())
    for split in ("train", "test"):
        for n in sorted(x for x in names if x.startswith(f"{split}_img/") and not x.endswith("/")):
            stem = Path(n).stem
            lab_name = next(x for x in names if x.startswith(f"{split}_lab/") and Path(x).stem == stem)
            img = np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"))
            lab = binary(Image.open(io.BytesIO(z.read(lab_name))))
            save(out / "deepcrack", split, stem, img, lab)


def roads(src, out):
    import pyarrow.parquet as pq
    for f in sorted((src / "massroads" / "data").glob("*.parquet")):
        split = f.name.split("-")[0]
        split = {"validation": "val"}.get(split, split)
        shard = f.name.split("-")[1]
        t = pq.read_table(f).to_pylist()
        for i, row in enumerate(t):
            stem = f"{shard}_{i:04d}"                    # the parquet rows carry no file names
            img = np.asarray(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB"))
            lab = binary(Image.open(io.BytesIO(row["label"]["bytes"])))
            save(out / "roads", split, stem, img, lab)


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    only = sys.argv[3:]                                     # optional subset, e.g. roads
    for name, fn in (("drive", drive), ("chase", chase), ("deepcrack", deepcrack), ("roads", roads)):
        if only and name not in only:
            continue
        fn(src, out)
        for split in ("train", "val", "test"):
            d = out / name / split / "img"
            if d.exists():
                print(name, split, len(list(d.iterdir())), flush=True)
