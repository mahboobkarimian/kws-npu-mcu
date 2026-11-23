r"""DS-CNN keyword spotter (depthwise-separable CNN), sized for Cortex-M microcontrollers and the AMD Phoenix NPU.

Origin of the design: "DS-CNN-S" from Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" (ARM, 2017),
which is also the MLPerf Tiny keyword-spotting reference model. We keep its shape (stem /2, 4 separable blocks,
64 channels, global average pool) and change only what the hardware needs: see "NPU / MCU notes" below.

The picture (shapes are [channels, time, cepstral coefficient]):

    MFCC image        [ 1, 49, 10]   1 s of audio, 49 frames x 10 MFCCs (computed on the host, features.py)
      |
    1 FeatureStem     [64, 25,  5]   full conv 9x4, stride 2: local spectro-temporal patterns (~200 ms x 4 coefs)
      |
    2 DSBlock 1  \                   each block =  DepthwiseConv 3x3  (per channel: "how does MY pattern move
    3 DSBlock 2   |   [64, 25,  5]                                        in time / across coefficients?")
    4 DSBlock 3   |                              + PointwiseConv 1x1  (per position: "which patterns occur
    5 DSBlock 4  /                                                      TOGETHER here?" = mixing channels)
      |
    6 GlobalAvgPool   [64,  1,  1]   "did these patterns occur anywhere in the second?"  (position-free)
      |
    7 Classifier      [12,  1,  1]   1x1 conv = fully connected layer: 64 evidence values -> one score per class
      |
    softmax (host)    probabilities: silence, unknown, yes, no, ...

Why depthwise + pointwise ("separable") instead of normal 3x3 convs: a normal 3x3 conv 64->64 needs 64*64*9 = 36,864
weights; the separable pair needs 64*9 + 64*64 = 4,672 (7.9x fewer) for almost the same accuracy (Hello Edge, MobileNet).

NPU / MCU notes (from FINDINGS.md section 11, the Phoenix op map, and our own probe on 23 Nov):
  - Stem kernel 9x4 instead of Hello Edge's 10x4: the Phoenix compiler only accepts kernel sizes 1..9
    ("kernel (4x10) is not in AMD_AIE2_Nx4_Overlay supported range {1..9}" -> that conv went to the CPU).
  - BatchNorm is used for training only; before int8 it is folded into the conv weights/bias (quant.py).
  - The classifier is a 1x1 Conv on a [1,64,1,1] tensor, not Gemm/Linear: Gemm falls back to the CPU on Phoenix.
  - Softmax stays on the host (Softmax over channels is a CPU op on Phoenix and is not needed to pick the class).
  - Only Conv, Relu, GlobalAveragePool: all int8, all on the NPU, all in CMSIS-NN / TFLite Micro.
SPDX-License-Identifier: MIT
"""
import torch
import torch.nn as nn

from . import config as C

CHANNELS = 64           # feature maps everywhere after the stem
N_BLOCKS = 4            # depthwise-separable blocks


class FeatureStem(nn.Module):
    """Layer 1: the only full convolution.  [1, 49, 10] -> [64, 25, 5]

    Responsibility: look at the raw MFCC image and detect 64 kinds of short sound patterns, e.g. "energy rising at a
    word onset", "F2 moving up" (the tell-tale transition between a consonant and a vowel), "a hiss starting".
    Kernel 9 frames x 4 coefficients = 200 ms of audio (8 x 20 ms hop + 40 ms frame) x 4 of the 10 coefficients:
    about one syllable. Stride 2 on both axes halves the map (49x10 -> 25x5), so every later layer costs 4x less.
    It must be a full conv (not depthwise) because there is only 1 input channel to start from.
    """

    def __init__(self, out_ch: int = CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(1, out_ch, kernel_size=(9, 4), stride=(2, 2), padding=(4, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseConv(nn.Module):
    """First half of a separable block: a 3x3 filter for EACH channel on its own (groups = channels).  [64,25,5] -> same

    Responsibility: context in time and across coefficients, per pattern. Channel 7 only looks at channel 7's map,
    3 frames x 3 coefficients around each position, e.g. "my pattern, then 20 ms later the same pattern one coefficient
    higher". Each one widens the view by 2 positions = 4 frames (the stem's stride 2); after the stem and 4 blocks one
    position sees 25 frames (~0.5 s) and all 10 coefficients. The global pool then covers the whole second.
    It never mixes channels: 9 weights per channel.
    """

    def __init__(self, ch: int = CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, groups=ch, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PointwiseConv(nn.Module):
    """Second half of a separable block: a 1x1 conv that mixes the 64 channels at every position.  [64,25,5] -> same

    Responsibility: combine evidence. At each time/coefficient position it computes 64 new features, each a weighted
    sum of all 64 old ones, e.g. "onset AND rising F2 AND no hiss" -> "probably the /j/ of 'yes'". It sees no
    neighbours (1x1): the depthwise conv before it already brought the neighbourhood in.
    """

    def __init__(self, ch: int = CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSBlock(nn.Module):
    """Layers 2-5: one depthwise-separable block = DepthwiseConv ("where/when") then PointwiseConv ("what")."""

    def __init__(self, ch: int = CHANNELS):
        super().__init__()
        self.depthwise = DepthwiseConv(ch)
        self.pointwise = PointwiseConv(ch)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class GlobalAvgPool(nn.Module):
    """Layer 6: average each of the 64 maps over all 25 x 5 positions.  [64,25,5] -> [64,1,1]

    Responsibility: make the decision independent of WHERE in the 1 s window the word is (early, late) and of small
    shifts along the cepstral axis. No weights.
    """

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class Classifier(nn.Module):
    """Layer 7: fully connected 64 -> classes, written as a 1x1 conv (the NPU-friendly form of a Linear layer).

    Responsibility: turn the 64 pooled evidence values into one score (logit) per class. Softmax is done by the host.
    """

    def __init__(self, num_classes: int, ch: int = CHANNELS):
        super().__init__()
        self.fc = nn.Conv2d(ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        return self.fc(x)


class DSCNN(nn.Module):
    """input [B, 1, 49, 10] normalised MFCC  ->  logits [B, num_classes]"""

    def __init__(self, num_classes: int = len(C.TASKS["commands12"]), ch: int = CHANNELS, n_blocks: int = N_BLOCKS):
        super().__init__()
        self.stem = FeatureStem(ch)                                          # layer 1
        self.blocks = nn.Sequential(*[DSBlock(ch) for _ in range(n_blocks)])  # layers 2-5
        self.pool = GlobalAvgPool()                                          # layer 6
        self.classifier = Classifier(num_classes, ch)                        # layer 7

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.flatten(1)


def count(model: nn.Module, input_shape=(1, 1, C.N_FRAMES, C.N_MFCC)):
    """Parameters (BN excluded: it is folded away before deployment) and multiply-accumulates per inference."""
    macs, params = 0, 0
    hooks = []

    def hook(m, inp, out):
        nonlocal macs, params
        k = m.weight.numel() // m.out_channels            # weights per output value
        macs += out.numel() // out.shape[0] * k
        params += m.weight.numel() + m.out_channels        # every conv has a bias once BN is folded into it
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model.eval()(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return params, macs
