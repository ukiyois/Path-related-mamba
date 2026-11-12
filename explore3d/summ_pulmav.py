"""Table of the pulmonary artery / vein runs (runs/pulmav): Dice of artery and vein, artery / vein accuracy on the vessel
voxels, and that accuracy by geodesic distance from the root of each tree.

  python explore3d/summ_pulmav.py [runs/pulmav]
"""
import json
import sys
from pathlib import Path

BINS = ["0-32", "32-64", "64-96", "96-128", "128+", "unreached"]
print(f"{'run':28s} {'params':>7s} {'Dice A':>7s} {'Dice V':>7s} {'A/V acc':>8s} | acc by mm from the root " + " ".join(f"{b:>9s}" for b in BINS))
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/pulmav").glob("*.jsonl")):
    r = json.loads(f.read_text().strip().splitlines()[-1])
    g = r["av_accuracy_by_geodesic_mm"]
    print(f"{r['name']:28s} {r['params_k']:6.1f}k {r['dice_artery']:7.4f} {r['dice_vein']:7.4f} {r['av_accuracy']:8.4f} | "
          f"{' ' * 23} " + " ".join(f"{g[b]['acc']:9.3f}" for b in BINS))
