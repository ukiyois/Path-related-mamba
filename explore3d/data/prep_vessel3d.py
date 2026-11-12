"""MSD Task08 hepatic vessels -> isotropic 1.6 mm volumes cropped around the labels (npz: img float16 in [0, 1], lab uint8
with 1 = vessel; tumour voxels are background), and a fixed 243 / 60 train / test split.

  python explore3d/data/prep_vessel3d.py /data/datasets/msd/Task08_HepaticVessel /data/datasets/vessel3d
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SPACING = 1.6          # mm, isotropic
HU = (-100.0, 250.0)   # abdominal window


def one(args):
    img_path, lab_path, out = args
    import nibabel as nib
    from scipy import ndimage
    im = nib.load(str(img_path))
    sp = np.abs(np.asarray(im.header.get_zooms()[:3], np.float64))
    img = np.asarray(im.dataobj, np.float32)
    raw = np.asarray(nib.load(str(lab_path)).dataobj).astype(np.uint8)
    zoom = sp / SPACING
    img = ndimage.zoom(np.clip(img, *HU), zoom, order=1)
    lab = ndimage.zoom((raw == 1).astype(np.float32), zoom, order=1) > 0.5
    fg = ndimage.zoom((raw > 0).astype(np.float32), zoom, order=0) > 0.5          # vessels and tumours: liver region
    nz = np.argwhere(ndimage.binary_dilation(fg, iterations=16))
    lo, hi = nz.min(0), nz.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img = ((img[sl] - HU[0]) / (HU[1] - HU[0])).astype(np.float16)
    np.savez_compressed(out / (img_path.name.replace(".nii.gz", ".npz")), img=img, lab=lab[sl].astype(np.uint8))
    return img_path.name, img.shape, int(lab[sl].sum())


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cases = sorted((src / "imagesTr").glob("*.nii.gz"))
    jobs = [(c, src / "labelsTr" / c.name, out) for c in cases]
    with ProcessPoolExecutor(24) as ex:
        for name, shape, nv in ex.map(one, jobs):
            print(name, shape, nv, flush=True)
    rng = np.random.default_rng(0)
    names = [c.name.replace(".nii.gz", "") for c in cases]
    perm = rng.permutation(len(names))
    split = dict(train=sorted(names[i] for i in perm[:243]), test=sorted(names[i] for i in perm[243:]))
    (out / "split.json").write_text(json.dumps(split, indent=1))
    print(len(split["train"]), "train /", len(split["test"]), "test")
