"""Mamba2D's JIT-compiled wavefront kernel on this GPU against a loop over its recurrence (wf_cuda.cu):
h(0,0) = dL B_L x;  h(0,j) = e^{dL A_L} h(0,j-1) + dL B_L x;  h(i,0) = e^{dT A_T} h(i-1,0) + dT B_T x;
h(i,j) = (e^{dL A_L} h(i,j-1) + dL B_L x + e^{dT A_T} h(i-1,j) + dT B_T x) / 2;  y = sum_n C h + D x.
Forward and gradients (x and all parameters) on a small grid.

  python explore3d/check_mamba2d.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "explore3d")]

import torch

import zoo2d_ext

zoo2d_ext.mamba2d(8)
wf = sys.modules["m2d_models.kernels.wavefront_cuda"]
torch.manual_seed(0)
Bn, H, W, E, N, R = 2, 7, 9, 6, 3, 2
x = torch.randn(Bn, H, W, E, device="cuda", requires_grad=True)
AT_log = torch.randn(E, N, device="cuda").mul(0.3).requires_grad_()
AL_log = torch.randn(E, N, device="cuda").mul(0.3).requires_grad_()
x_proj_w = torch.randn(2 * R + 3 * N, E, device="cuda").mul(0.3).requires_grad_()
dTw, dLw = (torch.randn(E, R, device="cuda").mul(0.5).requires_grad_() for _ in range(2))
dTb, dLb = (torch.randn(E, device="cuda").mul(0.5).requires_grad_() for _ in range(2))
D = torch.randn(E, device="cuda").requires_grad_()
args = (x, AT_log, AL_log, x_proj_w, dTw, dTb, dLw, dLb, D)


def reference(x, AT_log, AL_log, x_proj_w, dTw, dTb, dLw, dLb, D):
    dT, dL, AT, AL, BT, BL, C = wf.proj_params(x, AT_log, AL_log, x_proj_w, dTw, dTb, dLw, dLb)
    eT, eL = torch.exp(dT[..., None] * AT), torch.exp(dL[..., None] * AL)          # (B,H,W,E,N)
    bT, bL = dT[..., None] * BT[..., None, :] * x[..., None], dL[..., None] * BL[..., None, :] * x[..., None]
    h = [[None] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                h[i][j] = bL[:, i, j]
            elif i == 0:
                h[i][j] = eL[:, i, j] * h[i][j - 1] + bL[:, i, j]
            elif j == 0:
                h[i][j] = eT[:, i, j] * h[i - 1][j] + bT[:, i, j]
            else:
                h[i][j] = 0.5 * (eL[:, i, j] * h[i][j - 1] + bL[:, i, j] + eT[:, i, j] * h[i - 1][j] + bT[:, i, j])
    hs = torch.stack([torch.stack(r, 1) for r in h], 1)                                # (B,H,W,E,N)
    return (hs * C[..., None, :]).sum(-1) + x * D


for recomp in ("partial", "full"):
    y_op = wf.wavefront_scan_cuda(*args, recomp=recomp)
    y_ref = reference(*args)
    g = torch.randn_like(y_ref)
    go = torch.autograd.grad((y_op * g).sum(), args)
    gr = torch.autograd.grad((y_ref * g).sum(), args)
    rel = max(((a - b).abs().max() / (b.abs().max() + 1e-12)).item() for a, b in zip(go, gr))
    print(f"recomp={recomp}: forward max err {(y_op - y_ref).abs().max().item():.2e} (scale {y_ref.abs().max().item():.1e}), "
          f"max relative grad err over x and 8 parameters {rel:.2e}", flush=True)
