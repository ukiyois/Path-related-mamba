"""Pathfinder images with the official generator (third_party/baselines/pathfinder, Linsley et al. / Kim et al.): the LRA
release bucket is no longer public, so the task is regenerated. One process per batch id.

  python explore3d/data/gen_pathfinder.py <out_dir> <window> <contour_length> <n_per_batch> <n_batches> [n_procs]

Settings follow the generator's wrapper for the hard condition (contour length 14, distractors of a third of it,
30 / distractor_length distractor snakes, continuity 1.8, paddle margin 3), with marker, paddle and padding sizes
scaled from its 300 px window to the target window.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

GEN = Path(__file__).resolve().parents[2] / "third_party" / "baselines" / "pathfinder"
sys.path.insert(0, str(GEN))


def run(job):
    out, window, cl, n, bid = job
    import numpy as np
    from PIL import Image
    import snakes2
    import snakes2_wrapper_args as A

    def imsave(path, arr):                                      # scipy.misc.imsave semantics: bytescale floats
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            lo, hi = float(a.min()), float(a.max())
            a = ((a - lo) / (hi - lo if hi > lo else 1) * 255).round().astype(np.uint8)
        Image.fromarray(a).save(path)
    snakes2._imsave = imsave
    circle = snakes2.draw_circle                                # scipy.misc.imresize returned 0..255 uint8; the
    snakes2.draw_circle = lambda *a: circle(*a) * 255.0          # repository's float replacement returns 0..1, and the
                                                                # caller divides markers by 255 -> invisible markers
    np.random.seed(1000 + bid)
    import random
    random.seed(1000 + bid)
    args = A.Args(contour_path=os.path.join(out, f"curv_contour_length_{cl}"), batch_id=bid, n_images=n)
    s = window / 300.0
    args.window_size = [window, window]
    args.padding = max(1, int(round(22 * s)))
    args.antialias_scale = 4
    args.paddle_margin_list = [max(1, int(round(3 * s)))]
    args.marker_radius = 3 if window <= 64 else 4                # markers must stay visible (LRA-style circles)
    args.paddle_thickness = max(1, round(2 * s * 4)) / 4          # thickness x antialias must be an integer
    args.paddle_length = max(2, int(round(5 * s)))
    args.seed_distance = max(4, int(round(27 * s)))
    args.continuity = 1.8
    args.contour_length = cl
    args.distractor_length = cl // 3                          # Python 2 integer division in the original wrapper
    args.num_distractor_snakes = 30 // args.distractor_length
    args.snake_contrast_list = [1.0]
    args.use_single_paddles = False
    args.segmentation_task = False
    args.segmentation_task_double_circle = False
    snakes2.from_wrapper(args)
    return bid


if __name__ == "__main__":
    out, window, cl, n, nb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    procs = int(sys.argv[6]) if len(sys.argv) > 6 else 32
    with Pool(procs) as p:
        for bid in p.imap_unordered(run, [(out, window, cl, n, b) for b in range(nb)]):
            print("batch", bid, "done", flush=True)
