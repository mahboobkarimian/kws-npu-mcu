"""Speaker pitch (f0) estimate per clip, with YIN (de Cheveigne & Kawahara, JASA 2002). Used only for EVALUATION:
it lets us split the test set into low / middle / high voices and check that accuracy holds for all of them.

YIN in short, per 32 ms frame: for every candidate period tau (2..16.7 ms = 500..60 Hz) measure how different the
signal is from itself shifted by tau: d(tau) = sum (x[j] - x[j+tau])^2. A voiced sound repeats every pitch period, so
d dips at tau = 1/f0. Normalise d by its running mean (d'), take the first dip below 0.20 -> f0 = fs / tau.
Clip f0 = median over the voiced frames (>= 4 frames within 30 dB of the clip's loudest frame).
SPDX-License-Identifier: MIT
"""
import numpy as np
import torch

FS = 16000
W, HOP = 512, 160
TAU_MIN, TAU_MAX = FS // 500, FS // 60          # 32 .. 266 samples
THRESH = 0.20                                   # 0.15 in the paper; 0.20 for noisy phone recordings


@torch.no_grad()
def clip_f0(waves: np.ndarray, batch: int = 500) -> np.ndarray:
    """waves [N, 16000] float -> median f0 per clip in Hz (NaN if fewer than 4 voiced frames)."""
    out = np.full(len(waves), np.nan)
    L = W + TAU_MAX
    for s in range(0, len(waves), batch):
        x = torch.from_numpy(np.asarray(waves[s:s + batch], np.float32))
        fr = x.unfold(1, L, HOP)                                        # [B, F, L]
        B, F_, _ = fr.shape
        d = torch.stack([((fr[..., :W] - fr[..., t:t + W]) ** 2).sum(-1) for t in range(1, TAU_MAX + 1)], -1)
        tau = torch.arange(1, TAU_MAX + 1, dtype=torch.float32)
        dn = d * tau / torch.cumsum(d, -1).clamp_min(1e-12)              # cumulative-mean-normalised difference
        dn[..., : TAU_MIN - 1] = 2.0                                     # outside the search range
        below = dn < THRESH
        first = torch.argmax(below.int(), -1)                            # first tau index below the threshold
        found = below.any(-1)
        idx = torch.arange(TAU_MAX).view(1, 1, -1)
        after = idx >= first.unsqueeze(-1)
        run = after & (torch.cumsum((~below & after).int(), -1) == 0)  # the first contiguous dip
        best = torch.where(run, dn, torch.full_like(dn, 9.0)).argmin(-1)
        f0 = FS / (best + 1).float()
        energy = (fr[..., :W] ** 2).mean(-1)
        loud = 10 * torch.log10(energy + 1e-12) > 10 * torch.log10(energy.max(-1, keepdim=True).values + 1e-12) - 30
        ok = found & loud
        for b in range(B):
            v = f0[b][ok[b]]
            if len(v) >= 4:
                out[s + b] = float(v.median())
    return out
