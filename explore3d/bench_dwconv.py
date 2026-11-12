"""Depthwise 3x3x3 conv on the cost-volume lattice: equivalent implementations, fwd + bwd time.

  python explore3d/bench_dwconv.py
"""
import time

import torch
import torch.nn.functional as F

B, C, D, H, W = 4, 64, 48, 64, 128
x = torch.randn(B, C, D, H, W, device="cuda", dtype=torch.bfloat16, requires_grad=True)
w = (torch.randn(C, 1, 3, 3, 3, device="cuda") * 0.2).requires_grad_()
g = torch.randn(B, C, D, H, W, device="cuda", dtype=torch.bfloat16)


def native():
    return F.conv3d(x, w.to(x.dtype), padding=1, groups=C)


def channels_last():
    return F.conv3d(x.contiguous(memory_format=torch.channels_last_3d), w.to(x.dtype), padding=1, groups=C)


def three_conv2d():
    """y[:, :, d] = sum_dz conv2d_dw(x[:, :, d + dz - 1], w[:, :, dz]); D folded into the batch."""
    xs = x.transpose(1, 2).reshape(B * D, C, H, W)
    wk = w.to(x.dtype)
    ys = [F.conv2d(xs, wk[:, :, dz], padding=1, groups=C).view(B, D, C, H, W) for dz in range(3)]
    y = ys[1].clone()
    y[:, 1:] += ys[0][:, :-1]
    y[:, :-1] += ys[2][:, 1:]
    return y.transpose(1, 2)


def cudnn_direct():
    return torch.cudnn_convolution(x, w.to(x.dtype), (1, 1, 1), (1, 1, 1), (1, 1, 1), C, False, False, True)


def timeit(fn, n=5):
    for _ in range(2):
        torch.autograd.grad(fn(), (x, w), g)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        torch.autograd.grad(fn(), (x, w), g)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


ref = native().float()
for name, fn in [("native depthwise3d", native), ("channels_last_3d", channels_last), ("3 x conv2d", three_conv2d),
                 ("cudnn_convolution", cudnn_direct)]:
    try:
        err = (fn().float() - ref).abs().max().item()
        print(f"{name:20s} fwd+bwd {timeit(fn):7.1f} ms   max diff vs native {err:.1e}", flush=True)
    except Exception as e:
        print(f"{name:20s} failed: {type(e).__name__}: {str(e)[:100]}", flush=True)
