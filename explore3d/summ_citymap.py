"""Table of the city-map runs (runs/citymap): test Opt / Suc overall and Suc by axis turns and by orthant switches of the
best 4-connected route from the goal to the start.

  python explore3d/summ_citymap.py [runs/citymap]
"""
import json
import sys
from pathlib import Path

rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/citymap").glob("*.jsonl")):
    rows.append(json.loads(f.read_text().strip().splitlines()[-1]))
turns = ["0", "1", "2", "3", "4", "5", "6+"]
switches = ["0", "1", "2", "3+"]
print(f"{'model':12s} {'L':>2s} {'Opt':>6s} {'Suc':>6s} {'MAE':>5s} | Suc by axis turns " + " ".join(f"{t:>5s}" for t in turns)
      + " | by orthant switches " + " ".join(f"{s:>5s}" for s in switches))
for r in sorted(rows, key=lambda r: (r["L"], r["model"], r["C"])):
    t = r["test"]
    label = r["model"] + ("" if r["C"] == 64 else f" C{r['C']}")
    bt = " ".join(f"{t['by_turns'][k]['suc']:5.2f}" if k in t["by_turns"] else "    -" for k in turns)
    bs = " ".join(f"{t['by_switches'][k]['suc']:5.2f}" if k in t["by_switches"] else "    -" for k in switches)
    print(f"{label:12s} {r['L']:2d} {t['opt']:6.3f} {t['suc']:6.3f} {t['mae']:5.1f} | {' ' * 17} {bt} | {' ' * 19} {bs}")
    if "mae_by_turns" in t:                          # distance error at every reachable cell, by its geometry
        mt = " ".join(f"{t['mae_by_turns'][k]['mae']:5.1f}" if k in t["mae_by_turns"] else "    -" for k in turns)
        ms = " ".join(f"{t['mae_by_switches'][k]['mae']:5.1f}" if k in t["mae_by_switches"] else "    -"
                      for k in switches)
        print(f"{'':12s} {'':2s} {'':6s} {'':6s} {'':5s} | {'MAE by axis turns':17s} {mt} | {'MAE by switches':19s} {ms}")
