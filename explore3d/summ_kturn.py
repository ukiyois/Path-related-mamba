"""Table of the k-turn corridor runs (runs/e2k): far-segment MSE and MSE per segment index (chance 0.333).

  python explore3d/summ_kturn.py [runs/e2k ...]
"""
import json
import sys
from pathlib import Path

rows = []
for d in sys.argv[1:] or ["runs/e2k"]:
    for f in sorted(Path(d).glob("*.jsonl")):
        r = json.loads(f.read_text().strip().splitlines()[-1])
        rows.append(r)
rows.sort(key=lambda r: (r["model"], r["L"]))
print(f"{'model':12s} {'L':>2s} {'params':>7s}  far    " + " ".join(f"seg{k}" for k in range(len(rows[0]["mse_seg"])) if rows))
for r in rows:
    print(f"{r['model']:12s} {r['L']:>2d} {r['params_k']:6.1f}k {r['mse_far']:.3f}  " + " ".join(f"{v:.3f}" for v in r["mse_seg"])
          + f"   ({r['seconds'] / 60:.0f} min)")
