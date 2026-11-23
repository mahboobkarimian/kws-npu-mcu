"""int8 quantisation that is EXACTLY the same on PyTorch (training), the MCU (C code) and the Phoenix NPU.

The number format (chosen for the Phoenix DPU, which only does this; also trivial on a Cortex-M):
  - every tensor is int8 with a power-of-two scale and zero point 0:  real value = q * 2^-f   (q in -128..127)
  - one scale per tensor (weights included), biases stored in int8 too (|b_q| <= 127, a Phoenix DPU limit)
  - a layer computes  acc = sum(x_q * w_q) + (b_q << bias_shift)        (int32 accumulator)
                      y_q = clamp( (acc + 2^(sh-1)) >> sh , lo, 127 )    sh = f_in + f_w - f_out >= 0
    i.e. "round half UP" (the DPU's rounding, verified bit-exact on DFN3-NPU, 17 Nov). lo = 0 when ReLU is fused.
  - the global average pool is  y_q = round_half_up(sum(x_q) * 33 / 4096 * 2^(f_out - f_in)).
    33/4096 = 1/124.1 is the Phoenix DPU's fixed-point version of 1/125 (25 x 5 positions). We MEASURED it on the NPU
    (23 Nov, npu probe: 100 % of 2000 random inputs match this formula, only 94 % match an exact /125), and train and
    run the MCU with the same formula, so all three targets stay bit-identical.
Where the rules came from: our DFN3-NPU work (dfn/train/dfnpu_quant.py), which ran 100 % bit-exact on this NPU.

Pipeline:   float model (with BatchNorm)
              --fold_batchnorm-->  convs with bias
              --calibrate-->       activation scales from the 99.9th percentile (max |x| wastes range on outliers)
              --QAT-->             a few epochs with fake quantisation in the forward pass (straight-through gradient)
              --export_int8-->     integer weights + shifts (a dict) = what the MCU header and the NPU QDQ graph contain
SPDX-License-Identifier: MIT
"""
import copy
import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DSCNN


# ----------------------------------------------------------------------------------------------- number helpers
def p2(maxabs: float) -> float:
    """Smallest power-of-two scale s with maxabs / s <= 127."""
    return float(2.0 ** math.ceil(math.log2(max(float(maxabs), 1e-8) / 127.0)))


def fix(scale: float) -> int:
    """scale 2^-f -> f (the 'fix position')."""
    f = -math.log2(scale)
    assert abs(f - round(f)) < 1e-9, scale
    return int(round(f))


def round_half_up(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5)


def fake_quant(x: torch.Tensor, s: float, lo: int = -128, hi: int = 127) -> torch.Tensor:
    """Quantise to the int8 grid and back to float; the gradient passes straight through (STE)."""
    q = torch.clamp(round_half_up(x / s), lo, hi) * s
    return x + (q - x).detach()


# ----------------------------------------------------------------------------------------------- BatchNorm folding
def fold_batchnorm(model: DSCNN) -> DSCNN:
    """Return a copy in which every conv+BN pair is one conv with bias (BN replaced by Identity). Same function."""
    m = copy.deepcopy(model).eval()
    for layer in [m.stem] + [b.depthwise for b in m.blocks] + [b.pointwise for b in m.blocks]:
        conv, bn = layer.conv, layer.bn
        k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        new = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                        groups=conv.groups, bias=True)
        new.weight.data = conv.weight.data * k.view(-1, 1, 1, 1)
        new.bias.data = bn.bias.data - bn.running_mean * k
        layer.conv, layer.bn = new, nn.Identity()
    return m


# ----------------------------------------------------------------------------------------------- QAT modules
class QConv(nn.Module):
    """A folded conv (+ optional fused ReLU) with fake-quantised weights, bias and output."""

    def __init__(self, conv: nn.Conv2d, relu: bool, s_in: float, s_out: float):
        super().__init__()
        self.conv, self.relu, self.s_in, self.s_out = conv, relu, s_in, s_out

    def scales(self):
        s_w = p2(self.conv.weight.detach().abs().max())
        s_b = max(self.s_in * s_w, p2(self.conv.bias.detach().abs().max()))      # int8 bias, never finer than acc
        return s_w, s_b

    def forward(self, x):
        s_w, s_b = self.scales()
        w = fake_quant(self.conv.weight, s_w)
        b = fake_quant(self.conv.bias, s_b)
        y = F.conv2d(x, w, b, self.conv.stride, self.conv.padding, groups=self.conv.groups)
        if self.relu:
            return fake_quant(F.relu(y), self.s_out, lo=0)
        return fake_quant(y, self.s_out)


POOL_N = 125                       # 25 x 5 positions after the stem
POOL_MUL, POOL_SHIFT = 33, 12      # the Phoenix DPU computes "mean" as sum * 33 / 2^12 (measured; 1/125 ~ 1/124.1)


class QAvgPool(nn.Module):
    """Global average pool with int8 output, using the NPU's multiplier: round_half_up(sum * 33/4096)."""

    def __init__(self, s_in: float, s_out: float):
        super().__init__()
        self.s_in, self.s_out = s_in, s_out

    def forward(self, x):
        assert x.shape[2] * x.shape[3] == POOL_N, "the 33/4096 multiplier was measured for 125 positions"
        y = x.sum(dim=(2, 3), keepdim=True) * (POOL_MUL / 2.0 ** POOL_SHIFT)   # exact in float32 (< 2^24)
        return fake_quant(y, self.s_out, lo=0)


class QInput(nn.Module):
    """The host quantises the normalised MFCC to int8 (round half up) before the network sees it."""

    def __init__(self, s: float):
        super().__init__()
        self.s = s

    def forward(self, x):
        return fake_quant(x, self.s)


class QDSCNN(nn.Module):
    """The int8 network as a chain of layers (same layer names as DSCNN)."""

    def __init__(self, folded: DSCNN, scales: Dict[str, float]):
        super().__init__()
        s = scales
        self.input = QInput(s["input"])
        self.stem = QConv(folded.stem.conv, True, s["input"], s["stem"])
        layers, prev = [], "stem"
        for i, b in enumerate(folded.blocks):
            layers.append(QConv(b.depthwise.conv, True, s[prev], s[f"block{i}.depthwise"]))
            layers.append(QConv(b.pointwise.conv, True, s[f"block{i}.depthwise"], s[f"block{i}.pointwise"]))
            prev = f"block{i}.pointwise"
        self.blocks = nn.Sequential(*layers)
        self.pool = QAvgPool(s[prev], s["pool"])
        self.classifier = QConv(folded.classifier.fc, False, s["pool"], s["classifier"])
        self.scale_table = dict(s)

    def forward(self, x):
        x = self.input(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.flatten(1)


# ----------------------------------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(folded: DSCNN, batches: List[torch.Tensor], pct: float = 99.9) -> Dict[str, float]:
    """Run the float (folded) model and pick a power-of-two scale for every activation tensor.
    Percentile, not max: one rare outlier would otherwise halve the resolution of the whole tensor
    (lesson from DFN3-NPU, 17 Nov: max ranges destroyed PTQ, 99.9 % ranges fixed it)."""
    acts: Dict[str, List[torch.Tensor]] = {}

    def keep(name):
        def h(_, __, out):
            acts.setdefault(name, []).append(out.detach().abs().flatten())
        return h
    hooks = [folded.stem.register_forward_hook(keep("stem")), folded.pool.register_forward_hook(keep("pool")),
             folded.classifier.register_forward_hook(keep("classifier"))]
    for i, b in enumerate(folded.blocks):
        hooks.append(b.depthwise.register_forward_hook(keep(f"block{i}.depthwise")))
        hooks.append(b.pointwise.register_forward_hook(keep(f"block{i}.pointwise")))
    xs = []
    for x in batches:
        folded(x)
        xs.append(x.abs().flatten())
    for h in hooks:
        h.remove()
    scales = {"input": p2(np.percentile(torch.cat(xs).numpy(), pct))}
    for k, v in acts.items():
        v = torch.cat(v).numpy()
        # the logits keep their full range (the top score must not saturate); the rest use the percentile
        scales[k] = p2(v.max() if k == "classifier" else np.percentile(v, pct))
    return scales


# ----------------------------------------------------------------------------------------------- int8 export + runner
def layer_tag(name: str) -> str:
    """Readable C / ONNX name: 'blocks.3' (the 4th conv inside the separable blocks) -> 'block1_pointwise'."""
    if name.startswith("blocks."):
        i = int(name.split(".")[1])
        return f"block{i // 2}_{'depthwise' if i % 2 == 0 else 'pointwise'}"
    return name


@torch.no_grad()
def export_int8(q: QDSCNN) -> Dict:
    """Integer parameters of every layer: exactly what the C header and the NPU graph hold."""
    layers = []
    for name, m in [("stem", q.stem)] + [(f"blocks.{i}", m) for i, m in enumerate(q.blocks)] + \
                   [("pool", q.pool), ("classifier", q.classifier)]:
        if isinstance(m, QAvgPool):
            layers.append(dict(name=name, op="avgpool", f_in=fix(m.s_in), f_out=fix(m.s_out)))
            continue
        s_w, s_b = m.scales()
        f_in, f_w, f_b, f_out = fix(m.s_in), fix(s_w), fix(s_b), fix(m.s_out)
        w_q = torch.clamp(round_half_up(m.conv.weight / s_w), -128, 127).to(torch.int8).numpy()
        b_q = torch.clamp(round_half_up(m.conv.bias / s_b), -128, 127).to(torch.int8).numpy()
        sh = f_in + f_w - f_out
        assert sh >= 0, f"{name}: output finer than the accumulator (shift {sh})"
        bias_shift = f_in + f_w - f_b
        assert bias_shift >= 0
        layers.append(dict(name=name, op="conv", w=w_q, b=b_q, f_in=f_in, f_w=f_w, f_b=f_b, f_out=f_out,
                           shift=sh, bias_shift=bias_shift, stride=tuple(m.conv.stride), pad=tuple(m.conv.padding),
                           groups=m.conv.groups, relu=m.relu))
    return dict(f_input=fix(q.input.s), layers=layers)


def quantize_input(x: np.ndarray, f_input: int) -> np.ndarray:
    """Host side: normalised MFCC (float) -> int8, round half up."""
    return np.clip(np.floor(x * 2.0 ** f_input + 0.5), -128, 127).astype(np.int8)


def _conv_int(x: np.ndarray, L: Dict) -> np.ndarray:
    """Integer conv, computed in float64 (exact: all values are integers far below 2^53). x [N,C,H,W] -> acc."""
    w = L["w"].astype(np.float64)
    co, ci_g, kh, kw = w.shape
    (sh_, sw_), (ph, pw), g = L["stride"], L["pad"], L["groups"]
    xp = np.pad(x.astype(np.float64), ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    N, Cin, H, W = xp.shape
    Ho, Wo = (H - kh) // sh_ + 1, (W - kw) // sw_ + 1
    cols = np.empty((N, Cin, kh, kw, Ho, Wo))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[:, :, i:i + sh_ * Ho:sh_, j:j + sw_ * Wo:sw_]
    if g == 1:
        acc = np.einsum("ncijhw,ocij->nohw", cols, w, optimize=True)
    else:                                                  # depthwise (groups == channels)
        acc = np.einsum("ncijhw,cij->nchw", cols, w[:, 0], optimize=True)
    return acc + (L["b"].astype(np.float64) * 2.0 ** L["bias_shift"])[None, :, None, None]


def run_int8(P: Dict, x_q: np.ndarray) -> np.ndarray:
    """The reference integer network. x_q int8 [N,1,49,10] -> logits int8 [N, classes]."""
    x = x_q.astype(np.float64)
    for L in P["layers"]:
        if L["op"] == "conv":
            acc = _conv_int(x, L)
            y = np.floor(acc / 2.0 ** L["shift"] + 0.5)                       # round half up
            x = np.clip(y, 0 if L["relu"] else -128, 127)
        else:                                                                 # global average pool (NPU formula)
            assert x.shape[2] * x.shape[3] == POOL_N
            s = x.sum(axis=(2, 3), keepdims=True) * POOL_MUL
            x = np.clip(np.floor(s * 2.0 ** (L["f_out"] - L["f_in"] - POOL_SHIFT) + 0.5), 0, 127)
    return x.reshape(x.shape[0], -1).astype(np.int8)


def param_bytes(P: Dict) -> int:
    return sum(L["w"].size + L["b"].size for L in P["layers"] if L["op"] == "conv")
