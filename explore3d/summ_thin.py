"""Table of the thin-structure testbed runs (runs/thin2d): F1, clDice, recall by centre-line orientation and, when
present, bridging recall (hidden centre line recovered) by orientation.

  python explore3d/summ_thin.py [runs/thin2d]
"""
import json
import sys
from pathlib import Path

for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/thin2d").glob("*.jsonl")):
    r = json.loads(f.read_text().strip().splitlines()[-1])
    ro = r["recall_by_orientation"]
    s = f"{r['name']:34s} F1 {r['f1']:.4f} clDice {r['cldice']:.4f} | rec 0/45/90/135 " + \
        " ".join(f"{ro[k]:.3f}" for k in ("0", "45", "90", "135"))
    br = r.get("bridging_recall_by_orientation")
    if br:
        s += f" | bridge {r['bridging_recall']:.3f}: " + " ".join(f"{br[k]:.3f}" for k in ("0", "45", "90", "135"))
        bp = r.get("bridging_precision_by_orientation")
        if bp:
            s += f" prec {r['bridging_precision']:.3f}: " + " ".join(f"{bp[k]:.3f}" for k in ("0", "45", "90", "135"))
    print(s + f" | {r['params_k']:.0f}k {r['train_seconds'] / 60:.0f}m")
