"""Published 2-D token mixers, official code from third_party/baselines, each wrapped as (B, C, H, W) -> (B, C, H, W).

One representative per scan paradigm (the path-family zoo):
  zigzag      PlainMamba2D            continuous zig-zag scans (4 orders, direction-aware B)     PlainMamba
  localmamba  LocalMamba SS2D         line scans + local-window scans (searched directions)     LocalMamba
  effvmamba   EfficientVMamba SS2D    atrous scans on the four 2x2 sub-grids                     EfficientVMamba
  msvmamba    MSVMamba SS2D           one full-resolution scan + three at half resolution        MSVMamba
  quadmamba   QuadMamba SS2D          learnable quadtree windows (stage-0 path)                  QuadMamba
  defmamba    DefMamba DSSM           raster, reversed and deformable (learned) scan orders      DefMamba
  grootv      GrootV Tree_SSM         propagation on a minimum spanning tree of the feature grid GrootV
  swin        Swin window attention   8x8 windows, shifted by 4 on odd layers (timm)             Swin
  swin16      Swin window attention   16x16 windows, shifted by 8 (window-size control)             Swin
  mambaout    MambaOut gated CNN      GatedCNNBlock without its norm and residual                MambaOut
  m2d         Mamba2D M2DBlock        released "2D_local" mixer, TL / BR sweeps on even / odd layers  Mamba2D
  m2d_g/_oc   two half-width M2DBlocks; _oc sweeps the second group on the W-flipped map
  m2d_g4/_oc4 four quarter-width M2DBlocks; _oc4 flips them by none / W / H / H+W: all four orthants in every layer
The selective scan runs on mamba_ssm's CUDA kernel (selective_scan_cuda) through each repository's own fallback
path; GrootV needs its tree_scan extension (third-party/TreeScan, python setup.py build_ext --inplace).
Each process builds one kind: LocalMamba and EfficientVMamba both ship a top-level package called `lib`.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "third_party" / "baselines"


def _pkg(name, pkg_dir, module):
    """Import pkg_dir/module.py as name.module without running the package __init__."""
    if name not in sys.modules:
        p = types.ModuleType(name)
        p.__path__ = [str(pkg_dir)]
        sys.modules[name] = p
    return importlib.import_module(f"{name}.{module}")


def _file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ChannelsLast(nn.Module):
    """(B, C, H, W) -> op on (B, H, W, C) -> (B, C, H, W)."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, x):
        return self.op(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


# ----------------------------------------------------------------------------- continuous zig-zag
def zigzag(C, n=None):
    for m in ("mmcv", "mmcv.cnn", "mmcv.cnn.bricks", "mmcv.cnn.bricks.transformer", "mmcv.cnn.utils",
              "mmcv.cnn.utils.weight_init"):                          # the mixer only needs trunc_normal_
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["mmcv.cnn.bricks.transformer"].build_dropout = lambda cfg: nn.Identity()
    sys.modules["mmcv.cnn.utils.weight_init"].trunc_normal_ = nn.init.trunc_normal_
    try:
        importlib.import_module("mamba_ssm.ops.triton.layernorm")
    except ImportError:                                              # renamed layer_norm in mamba_ssm 2.x
        ln = types.ModuleType("mamba_ssm.ops.triton.layernorm")
        ln.RMSNorm = importlib.import_module("mamba_ssm.ops.triton.layer_norm").RMSNorm
        sys.modules["mamba_ssm.ops.triton.layernorm"] = ln
    pm = _file("plain_mamba_layer", BASE / "PlainMamba/mmcls/plain_mamba_dev/models/plain_mamba/plain_mamba_layer.py")

    class Zigzag(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = pm.PlainMamba2D(d_model=C, d_state=16, expand=2, conv_size=7, conv_bias=True, bias=True,
                                     default_hw_shape=(n, n) if n else None)        # PlainMamba-L1 settings

        def forward(self, x):
            B, _, H, W = x.shape
            y = self.m(x.flatten(2).transpose(1, 2), (H, W))
            return y.transpose(1, 2).reshape(B, C, H, W)
    return Zigzag()


# ----------------------------------------------------------------------------- local windows
LOCAL_DIRECTIONS = [("h", "h_flip", "w7", "w7_flip"), ("h_flip", "v_flip", "w2", "w2_flip")]   # LocalVMamba-T stage 1


def localmamba(C, layer=0):
    sys.path.insert(0, str(BASE / "LocalMamba/classification"))
    lv = importlib.import_module("lib.models.local_vmamba")
    op = lv.SS2D(d_model=C, d_state=16, ssm_ratio=2.0, dt_rank="auto", d_conv=3, conv_bias=True,
                 directions=list(LOCAL_DIRECTIONS[layer % 2]))
    return _ChannelsLast(op)


# ----------------------------------------------------------------------------- atrous (skip-sampled) scans
def effvmamba(C):
    sys.path.insert(0, str(BASE / "EfficientVMamba/classification"))
    ev = _pkg("evm_models", BASE / "EfficientVMamba/classification/models", "vmamba_efficient")
    return _ChannelsLast(ev.SS2D(d_model=C, d_state=16, ssm_ratio=2.0, dt_rank="auto", d_conv=3, conv_bias=True,
                                 forward_type="v2", step_size=2))


# ----------------------------------------------------------------------------- multi-scale scans
def msvmamba(C):
    ms = _pkg("msv_models", BASE / "MSVMamba/classification/models", "vmamba")
    return _ChannelsLast(ms.SS2D(d_model=C, d_state=1, ssm_ratio=2.0, dt_rank="auto", d_conv=3, conv_bias=True,
                                 forward_type="v2", sscore_type="multiscale_4scan_12", ms_stage=[0],
                                 current_layer=0, ms_split=[1, 3], add_se=True))


# ----------------------------------------------------------------------------- learnable quadtree windows
def quadmamba(C, layer=0):
    cudnn = torch.backends.cudnn
    saved = (cudnn.enabled, cudnn.benchmark, cudnn.deterministic)
    qm = _pkg("qm_models", BASE / "QuadMamba/classification/models", "vmamba")
    cudnn.enabled, cudnn.benchmark, cudnn.deterministic = saved      # vmamba.py sets these globally
    cs = sys.modules["qm_models.csms6s"]
    op = qm.SS2D(d_model=C, d_state=1, ssm_ratio=2.0, dt_rank="auto", d_conv=3, conv_bias=False,
                 forward_type="v05_noz", channel_first=False, stage_num=0, depth_num=layer, block_depth=[2, 2, 5, 2])
    op.forward_core = partial(op.forward_core, SelectiveScan=cs.SelectiveScanMamba)   # mamba_ssm kernel
    return _ChannelsLast(op)


# ----------------------------------------------------------------------------- deformable scan order
def defmamba(C):
    dm = _pkg("defm_models", BASE / "DefMamba/classification/models", "vmamba")
    return _ChannelsLast(dm.DSSM(d_model=C, d_state=16, ssm_ratio=1.0, dt_rank="auto", act_layer=nn.SiLU, d_conv=3,
                                 conv_bias=True, dropout=0.0, stage=0))


# ----------------------------------------------------------------------------- minimum-spanning-tree topology
def grootv(C):
    sys.path.insert(0, str(BASE / "GrootVL/GrootV/third-party/TreeScan"))
    ts = _pkg("grootv_models", BASE / "GrootVL/GrootV/classification/models", "tree_scanning")
    return _ChannelsLast(ts.Tree_SSM(d_model=C, d_state=1, ssm_ratio=2, ssm_rank_ratio=2, dt_rank="auto",
                                     act_layer=nn.SiLU, d_conv=3, conv_bias=False, dropout=0.0))


# ----------------------------------------------------------------------------- 2-D FM recurrence (one orthant per layer)
def mamba2d(C, layer=0, groups=1, flip=False):
    """Mamba2D's released high-resolution mixer ("2D_local": one wavefront sweep + SepConv local path, scan start TL on
    even and BR on odd blocks, inter_flip). groups=2 splits the channels over two half-width official blocks; flip=True
    runs the second group on the horizontally flipped map, so every layer also sweeps the anti-diagonal orthant
    (the orthant-completion remedy); groups=2, flip=False is its parameter- and compute-matched control."""
    m2d = _pkg("m2d_models", BASE / "Mamba2D/models", "mamba2d")
    start = "BR" if layer % 2 else "TL"

    class M2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(m2d.M2DBlock(C // groups, scan_start=start, local_path=True)
                                        for _ in range(groups))

        def forward(self, x):
            xs = x.permute(0, 2, 3, 1).contiguous().chunk(groups, -1)
            ys = []
            for g, (blk, xg) in enumerate(zip(self.blocks, xs)):
                dims = ([2] if g & 1 else []) + ([1] if g & 2 else []) if flip else []   # g: none, W, H, H and W
                ys.append(blk(xg.flip(dims)).flip(dims) if dims else blk(xg))
            return torch.cat(ys, -1).permute(0, 3, 1, 2).contiguous()
    return M2D()


# ----------------------------------------------------------------------------- shifted-window attention
def swin(C, n, layer=0, window=8, heads=4):
    from timm.models.swin_transformer import SwinTransformerBlock
    blk = SwinTransformerBlock(dim=C, input_resolution=(n, n), num_heads=heads, window_size=window,
                               shift_size=window // 2 if layer % 2 else 0, dynamic_mask=True)
    for name in ("norm1", "norm2", "mlp", "drop_path1", "drop_path2"):      # keep only the attention
        if hasattr(blk, name):
            setattr(blk, name, nn.Identity())

    class Swin(nn.Module):
        def __init__(self):
            super().__init__()
            self.blk = blk

        def forward(self, x):
            return self.blk._attn(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
    return Swin()


# ----------------------------------------------------------------------------- gated CNN
class GatedCNN(nn.Module):
    """MambaOut GatedCNNBlock (models/mambaout.py) without its norm, drop path and residual; same parameter names."""

    def __init__(self, dim, expansion_ratio=8 / 3, kernel_size=7, conv_ratio=1.0):
        super().__init__()
        hidden, conv_channels = int(expansion_ratio * dim), int(conv_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = nn.GELU()
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = self.conv(c.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.fc2(self.act(g) * torch.cat((i, c), dim=-1)).permute(0, 3, 1, 2)


KINDS = ("zigzag", "localmamba", "effvmamba", "msvmamba", "quadmamba", "defmamba", "grootv", "swin", "swin16", "mambaout",
         "m2d", "m2d_g", "m2d_oc", "m2d_g4", "m2d_oc4")


def build_mixer(kind, C, n, layer=0):
    if kind == "zigzag":
        return zigzag(C, n)
    if kind == "localmamba":
        return localmamba(C, layer)
    if kind == "effvmamba":
        return effvmamba(C)
    if kind == "msvmamba":
        return msvmamba(C)
    if kind == "quadmamba":
        return quadmamba(C, layer)
    if kind == "defmamba":
        return defmamba(C)
    if kind == "grootv":
        return grootv(C)
    if kind == "swin":
        return swin(C, n, layer)
    if kind == "swin16":                                                    # window-size control
        return swin(C, n, layer, window=16)
    if kind == "mambaout":
        return GatedCNN(C)
    if kind == "m2d":
        return mamba2d(C, layer)
    if kind == "m2d_g":
        return mamba2d(C, layer, groups=2)
    if kind == "m2d_oc":
        return mamba2d(C, layer, groups=2, flip=True)
    if kind == "m2d_g4":
        return mamba2d(C, layer, groups=4)
    if kind == "m2d_oc4":                               # all four sweep orthants in every layer
        return mamba2d(C, layer, groups=4, flip=True)
    raise ValueError(kind)


if __name__ == "__main__":                              # python explore3d/zoo2d_ext.py <kind>: fwd/bwd check, time
    import time
    kind = sys.argv[1]
    torch.manual_seed(0)
    mods = nn.ModuleList(build_mixer(kind, 32, 64, layer=i) for i in range(2)).cuda()
    x = torch.randn(4, 32, 64, 64, device="cuda", requires_grad=True)
    for it in range(4):
        if it == 1:
            torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = x
            for m in mods:
                y = y + m(y)
        y.float().square().mean().backward()
    torch.cuda.synchronize()
    g = x.grad
    print(f"{kind}: out {tuple(y.shape)} finite {bool(torch.isfinite(y).all())} grad finite {bool(torch.isfinite(g).all())} "
          f"params/layer {sum(p.numel() for p in mods[0].parameters()) / 1e3:.1f}k  {(time.time() - t0) / 3 * 1e3:.0f} ms/iter")
