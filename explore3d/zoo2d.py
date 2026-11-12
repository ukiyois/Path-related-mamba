"""Published 2-D mixers for the corridor diagnostics (explore3d/corridor.py).

  vmamba     VMamba SS2D (third_party/baselines/VMamba), forward_type v01: four cross scans, mamba_ssm
             selective scan, torch cross-scan
  mamba2d    Mamba2D's recurrence re-implemented from its CUDA kernel (third_party/baselines/Mamba2D,
             wf_cuda.cu): interior h = 1/2 (dA_L h_left + dt_L B_L x + dA_T h_up + dt_T B_T x), first row /
             column without the 1/2; scans from TL and BR (its double-scan setting) summed
  rowcol     2DMamba-style composition inside one layer: a row scan, then a column scan of its output
  vssd       non-causal SSD (VSSD-style): every token reads one global state sum_j B_j x_j
  attn       multi-head self-attention with a learned 2-D position embedding
All take and return (B, C, H, W).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
VMAMBA = ROOT / "third_party" / "baselines" / "VMamba" / "classification" / "models"


class VMambaSS2D(nn.Module):
    def __init__(self, C):
        super().__init__()
        if str(VMAMBA) not in sys.path:
            sys.path.insert(0, str(VMAMBA))
        from vmamba import SS2D
        self.m = SS2D(d_model=C, d_state=16, ssm_ratio=2.0, dt_rank="auto", d_conv=3, conv_bias=True,
                      forward_type="v01", channel_first=False)

    def forward(self, x):
        return self.m(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _fm_scan(dAL, dAT, bxL, bxT):
    """Mamba2D recurrence from the top-left corner, one anti-diagonal at a time.
    All inputs (B, H, W, E, N); returns h (B, H, W, E, N)."""
    B, H, W, E, N = dAL.shape
    diag_prev = None
    i_prev = None
    cells = []
    for k in range(H + W - 1):
        i0, i1 = max(0, k - W + 1), min(H - 1, k)
        ii = torch.arange(i0, i1 + 1, device=dAL.device)
        jj = k - ii
        aL, aT = dAL[:, ii, jj], dAT[:, ii, jj]
        bL, bT = bxL[:, ii, jj], bxT[:, ii, jj]
        if diag_prev is None:
            h = bL
        else:
            left = torch.zeros_like(bL)
            up = torch.zeros_like(bL)
            hasL = jj >= 1                                  # (i, j-1) sits on the previous diagonal
            hasT = ii >= 1                                  # (i-1, j) as well
            idxL = (ii - i_prev)[hasL]
            idxT = (ii - 1 - i_prev)[hasT]
            left[:, hasL] = diag_prev[:, idxL]
            up[:, hasT] = diag_prev[:, idxT]
            interior = (hasL & hasT).view(1, -1, 1, 1)
            both = 0.5 * (aL * left + bL + aT * up + bT)
            only_l = aL * left + bL
            only_t = aT * up + bT
            h = torch.where(interior, both, torch.where(hasL.view(1, -1, 1, 1), only_l, only_t))
        cells.append((ii, jj, h))
        diag_prev, i_prev = h, i0
    out = dAL.new_zeros(B, H, W, E, N)
    for ii, jj, h in cells:
        out = out.index_put((torch.arange(B, device=out.device)[:, None], ii[None, :], jj[None, :]), h)
    return out


class Mamba2DMixer(nn.Module):
    def __init__(self, C, d_state=16, expand=2):
        super().__init__()
        E = C * expand
        self.E, self.N = E, d_state
        self.dt_rank = max(1, math.ceil(C / 16))
        self.in_proj = nn.Linear(C, 2 * E, bias=False)
        self.conv = nn.Conv2d(E, E, 3, padding=1, groups=E)
        self.x_proj = nn.ModuleList(nn.Linear(E, 2 * self.dt_rank + 3 * d_state, bias=False) for _ in range(2))
        self.dt_T = nn.ModuleList(nn.Linear(self.dt_rank, E) for _ in range(2))
        self.dt_L = nn.ModuleList(nn.Linear(self.dt_rank, E) for _ in range(2))
        A = torch.log(torch.arange(1, d_state + 1).float()).repeat(E, 1)
        self.AT_log = nn.Parameter(A.clone().unsqueeze(0).repeat(2, 1, 1))
        self.AL_log = nn.Parameter(A.clone().unsqueeze(0).repeat(2, 1, 1))
        self.D = nn.Parameter(torch.ones(2, E))
        self.out_proj = nn.Linear(E, C, bias=False)

    def _scan(self, x, s):
        """x (B, H, W, E) -> (B, H, W, E) for direction s (0: from top-left)."""
        dbc = self.x_proj[s](x)
        dT, dL, BT, BL, Cc = torch.split(dbc, [self.dt_rank, self.dt_rank, self.N, self.N, self.N], -1)
        dT = F.softplus(self.dt_T[s](dT)).unsqueeze(-1)              # (B,H,W,E,1)
        dL = F.softplus(self.dt_L[s](dL)).unsqueeze(-1)
        AT, AL = -torch.exp(self.AT_log[s]), -torch.exp(self.AL_log[s])  # (E,N)
        xe = x.unsqueeze(-1)
        h = _fm_scan(torch.exp(dL * AL), torch.exp(dT * AT),
                     dL * BL.unsqueeze(-2) * xe, dT * BT.unsqueeze(-2) * xe)
        return (h * Cc.unsqueeze(-2)).sum(-1) + self.D[s] * x

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).float()
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, -1)
        xi = F.silu(self.conv(xi.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        y = self._scan(xi, 0) + self._scan(xi.flip(1, 2), 1).flip(1, 2)   # TL and BR scans
        return self.out_proj(y * F.silu(z)).permute(0, 3, 1, 2)


class RowColMamba(nn.Module):
    """A row scan, then a column scan of its output (bidirectional each), within one layer."""

    def __init__(self, C):
        super().__init__()
        from mamba_ssm import Mamba
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.rf, self.rb = Mamba(C, d_state=16, expand=2), Mamba(C, d_state=16, expand=2)
        self.cf, self.cb = Mamba(C, d_state=16, expand=2), Mamba(C, d_state=16, expand=2)

    @staticmethod
    def _bi(f, b, seq):
        return f(seq) + b(seq.flip(1)).flip(1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.dw(x)
        rows = x.permute(0, 2, 3, 1).reshape(B * H, W, C).contiguous()
        r = self._bi(self.rf, self.rb, rows).view(B, H, W, C)
        cols = r.permute(0, 2, 1, 3).reshape(B * W, H, C).contiguous()
        c = self._bi(self.cf, self.cb, cols).view(B, W, H, C).permute(0, 3, 2, 1)
        return c


class NonCausalSSD(nn.Module):
    """VSSD-style: y_i = C_i^T sum_j (dt_j B_j x_j) / L + D x_i per head (no causal mask, no decay)."""

    def __init__(self, C, heads=4, d_state=16):
        super().__init__()
        self.h, self.N = heads, d_state
        self.dw = nn.Conv2d(C, C, 3, padding=1, groups=C)
        self.proj = nn.Linear(C, C + 2 * heads * d_state + heads, bias=False)
        self.D = nn.Parameter(torch.ones(C))
        self.out = nn.Linear(C, C)

    def forward(self, x):
        B, C, H, W = x.shape
        t = F.silu(self.dw(x)).permute(0, 2, 3, 1).reshape(B, H * W, C)
        p = self.proj(t)
        v, Bm, Cm, dt = torch.split(p, [C, self.h * self.N, self.h * self.N, self.h], -1)
        v = v.view(B, -1, self.h, C // self.h)
        Bm, Cm = Bm.view(B, -1, self.h, self.N), Cm.view(B, -1, self.h, self.N)
        dt = F.softplus(dt).unsqueeze(-1)
        S = torch.einsum("blhn,blhd->bhnd", Bm * dt, v) / (H * W)
        y = torch.einsum("blhn,bhnd->blhd", Cm, S).reshape(B, -1, C) + self.D * t
        return self.out(y).view(B, H, W, C).permute(0, 3, 1, 2)


class Attention2D(nn.Module):
    def __init__(self, C, n, heads=4):
        super().__init__()
        self.h = heads
        self.pos = nn.Parameter(torch.randn(1, n * n, C) * 0.02)
        self.qkv = nn.Linear(C, 3 * C)
        self.out = nn.Linear(C, C)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.permute(0, 2, 3, 1).reshape(B, H * W, C) + self.pos
        q, k, v = self.qkv(t).view(B, H * W, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, H * W, C)
        return self.out(y).view(B, H, W, C).permute(0, 3, 1, 2)
