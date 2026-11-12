"""GrootV's compiled tree scan on this GPU against a path-product reference: the two-pass refine computes
F_v = sum_u (prod of the edge weights on the tree path u - v) x_u, the weight of edge (v, parent v) being the
child's A-bar. Checks forward and the feature / edge-weight gradients on small random grids.

  python explore3d/check_grootv.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "explore3d")]

import torch

import zoo2d_ext

zoo2d_ext.grootv(8)                                        # imports grootv_models.tree_scanning
ts = sys.modules["grootv_models.tree_scanning"]
core = sys.modules["grootv_models.tree_scan_utils.tree_scan_core"]
torch.manual_seed(0)
H = W = 6
L, B, D = H * W, 2, 3
fea = torch.randn(B, D, H, W, device="cuda")
tree = core.MinimumSpanningTree("Cosine", torch.exp)(fea)
idx, par, chi = ts._BFS.apply(tree, 4)
x = torch.randn(B, D, L, device="cuda", dtype=torch.float32, requires_grad=True)
a = (torch.rand(B, D, L, device="cuda") * 0.9 + 0.05).requires_grad_()
ew = ts.batch_index_opr(a, idx)                             # weights in BFS order, as tree_scanning_core does
out = ts._Refine.apply(x, ew, idx, par, chi, torch.ones_like(idx, dtype=ew.dtype))
print("shapes", tuple(idx.shape), tuple(par.shape), tuple(out.shape))


def reference(x, a, idx, par):
    """dense path-product kernel on the BFS tree; parent of BFS position i is BFS position par[i]."""
    Bn = x.shape[0]
    outs = []
    for b in range(Bn):
        order = idx[b].long().tolist()
        parent_pos = par[b].long().tolist()
        vert_parent = {order[i]: (order[parent_pos[i]] if i > 0 else -1) for i in range(L)}
        # path product between every pair: walk up to the root, products of child A-bar on each edge
        anc = {}
        for v in range(L):
            chain, w, u = [v], [torch.ones(D, device=x.device)], v
            while vert_parent[u] >= 0:
                w.append(w[-1] * a[b, :, u])
                u = vert_parent[u]; chain.append(u)
            anc[v] = dict(zip(chain, w))
        K = torch.zeros(D, L, L, device=x.device)
        for v in range(L):
            for u in range(L):
                common = next(c for c in anc[u] if c in anc[v])        # lowest common ancestor
                K[:, v, u] = anc[v][common] * anc[u][common]
        outs.append(torch.einsum("dvu,du->dv", K, x[b]))
    return torch.stack(outs)


for name, perm in [("feature in vertex order", lambda t: t),
                   ("feature in BFS order", lambda t: ts.batch_index_opr(t, idx))]:
    ref = reference(x, a, idx, par)
    err = (perm(ref) - out).abs().max().item() if name.endswith("BFS order") else (ref - out).abs().max().item()
    print(f"{name}: max |op - reference| {err:.2e}  (scale {ref.abs().max().item():.2e})")
g = torch.randn_like(out)
gx_op, ga_op = torch.autograd.grad((out * g).sum(), (x, a))
ref = reference(x, a, idx, par)
gx_ref, ga_ref = torch.autograd.grad((ref * g).sum(), (x, a))
print(f"grad x err {(gx_op - gx_ref).abs().max().item():.2e}, grad a err {(ga_op - ga_ref).abs().max().item():.2e} "
      f"(scales {gx_ref.abs().max().item():.1e}, {ga_ref.abs().max().item():.1e})")
