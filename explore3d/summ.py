"""Print one line per result in runs/<dir>/*.jsonl:  python explore3d/summ.py runs/e2 [key ...]"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
keys = sys.argv[2:]
rows = []
for f in sorted(d.glob("*.jsonl")):
    for line in open(f):
        r = json.loads(line)
        ks = keys or [k for k, v in r.items() if isinstance(v, float) and k not in ("seconds",)]
        vals = "  ".join(f"{k} {r[k]:.4g}" for k in ks if k in r and isinstance(r[k], (int, float)))
        rows.append(f"{r.get('name', f.stem):34s} {vals}  [{r.get('seconds', 0) / 60:.0f} min]")
print("\n".join(rows))
