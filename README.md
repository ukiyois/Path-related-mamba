# Path-related-mamba

Experiments on how far information can travel through multi-dimensional state-space-model (SSM) scans: how many layers
an axis-aligned, orthant or serialized scan needs to carry a signal along a path with a given number of turns, and
what changes when the scan is coupled, covers more orthants, or routes its coupling in 3-D.

## Layout

- `explore3d/`: experiment drivers and evaluation.
  - synthetic: `corridor.py` (2-D corridors with exactly K turns, orientation and flip-augmentation controls);
  - real data where the signal has to follow a structure: `citymap.py` (shortest paths on real city maps, the CSM
    benchmark of Neural A*), `pulmav.py` (pulmonary artery / vein separation on non-contrast CT, HiPaS),
    `retina_av.py` (retinal artery / vein classification), `sceneflow.py` (stereo cost aggregation);
  - scope: `thin2d.py`, `vessel3d.py` (ordinary thin-structure segmentation);
  - published mixers wrapped from their official code: `zoo2d_ext.py`, `zoo2d.py`;
  - data preparation: `explore3d/data/`; result tables: `explore3d/summ_*.py`.
- `third_party/`: `baselines.txt` pins the official repositories; `fetch_baselines.sh` clones them.

Some mixers and kernels used in the experiments are not released yet; the drivers that need them stop with an
import error until they are.

## Data

Nothing is stored in the repository. The preparation scripts document the sources: Neural A* planning-datasets
(CSM), HiPaS (Zenodo 14879605), the multi-disease fundus segmentation dataset (figshare 32646378), MSD Task08,
DeepCrack, Massachusetts Roads, SceneFlow.
