"""Table of the retinal artery / vein runs (runs/retina_av): accuracy, sensitivity (arteries), specificity (veins) and
Dice on the test images, then the accuracy by quadrant around the optic disc, by orthant switches and by axis turns of
each pixel's route from the disc.

  python explore3d/summ_retina.py [runs/retina_av]
"""
import json
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/retina_av").glob("*.jsonl")):
    r = json.loads(f.read_text().strip().splitlines()[-1])
    print(f"{r['model']:11s} {r['params_k']:6.1f}k  acc {r['acc']:.4f} sens {r['sens']:.4f} spec {r['spec']:.4f}  "
          f"Dice A {r['dice_artery']:.4f} V {r['dice_vein']:.4f}  ({r['train_seconds'] / 60:.0f} min)")
    for key in ("acc_by_quadrant", "acc_by_switches", "acc_by_turns"):
        if key in r:
            print(f"{'':13s}{key[7:]:9s} " + "  ".join(f"{k} {v['acc']:.3f}" for k, v in r[key].items()))
