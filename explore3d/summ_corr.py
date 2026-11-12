"""One-line summaries of corridor runs: e2zoo (dense corridors), e2k (k-turn), e2aug (flip augmentation).

  python explore3d/summ_corr.py
"""
import json
from pathlib import Path

for d in ("runs/e2zoo", "runs/e2k", "runs/e2aug"):
    for f in sorted(Path(d).glob("*.jsonl")):
        r = json.loads(f.read_text().strip().splitlines()[-1])
        extra = ""
        if "mse_seg" in r:
            extra = " seg " + " ".join(f"{v:.3f}" for v in r["mse_seg"])
        if "by_orient" in r:
            extra += " | " + " ".join(f"{o} {v['mse_far']:.3f}" for o, v in r["by_orient"].items())
        print(f"{d[5:]:6s} {f.stem:34s} far {r['mse_far']:.3f}{extra}  ({r['params_k']:.0f}k, {r['seconds'] / 60:.0f} min"
              f"{', skipped ' + str(r['skipped_steps']) if r.get('skipped_steps') else ''})")
