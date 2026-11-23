"""Audio front end: waveform -> MFCC "image" [49 frames x 10 coefficients].

This runs on the host (the MCU's Cortex-M core, or the laptop CPU next to the NPU), never inside the network.

Steps for every 40 ms frame (hop 20 ms):

    samples (640) --Hann window--> --FFT 1024--> power spectrum (513 bins)
        --40 triangular mel bands (20 Hz..4 kHz)--> --log--> --DCT-II, keep 10--> MFCC (10 numbers)

Why each step:
  - Hann window: stops the frame edges from smearing energy across frequency.
  - Power spectrum: phase does not matter for recognising a word, energy does.
  - Mel bands: narrow at low frequency, wide at high frequency, like the ear. A band wider than the gap
    between two pitch harmonics averages them together: above ~500 Hz for a 100 Hz voice, only above ~3 kHz for 300 Hz.
  - log: loudness becomes an ADDED constant (it mostly lands in coefficient 0) instead of a multiplication; it also
    compresses the 100 dB range of speech.
  - DCT, keep 10 ("cepstral liftering"): low coefficients = slow ripple across frequency = vocal-tract envelope
    (formants: WHAT is said). The pitch harmonics are a fast ripple = mostly high coefficients: dropped.
    Measured with synthetic vowels: this REDUCES pitch but does not remove it. The lowest mel bands are
    narrow (~40 Hz at 100 Hz), so low harmonics of high voices stay resolved, and a high voice has no energy below its
    f0, which mostly changes c1. Moving f0 100 -> 200..400 Hz moves the normalised c1..c9 (what the network sees)
    by 0.79-1.04 of the /a/ -> /i/ vowel change (raw c2..c9: 0.52-0.62).
    The rest of the pitch invariance is learned (2,618 speakers + speed/pitch augmentation).
  - Per-coefficient normalisation (mean/std from the training set) brings all 10 inputs to a similar range before int8.
SPDX-License-Identifier: MIT
"""
import math

import numpy as np
import torch
import torch.nn as nn

from . import config as C


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)   # HTK mel scale


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """[N_FFT/2+1, N_MELS] triangular filters, peak 1, edges equally spaced on the mel scale."""
    n_bins = C.N_FFT // 2 + 1
    freqs = np.arange(n_bins) * C.SAMPLE_RATE / C.N_FFT
    edges = mel_to_hz(np.linspace(hz_to_mel(C.F_MIN), hz_to_mel(C.F_MAX), C.N_MELS + 2))
    fb = np.zeros((n_bins, C.N_MELS))
    for m in range(C.N_MELS):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / (mid - lo)
        down = (hi - freqs) / (hi - mid)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down))
    return fb.astype(np.float32)


def dct_matrix() -> np.ndarray:
    """[N_MELS, N_MFCC] orthonormal DCT-II (only the first N_MFCC columns)."""
    n = np.arange(C.N_MELS)[:, None]
    k = np.arange(C.N_MFCC)[None, :]
    d = np.cos(math.pi / C.N_MELS * (n + 0.5) * k) * math.sqrt(2.0 / C.N_MELS)
    d[:, 0] *= 1.0 / math.sqrt(2.0)
    return d.astype(np.float32)


def hann_window() -> np.ndarray:
    """Periodic Hann window of FRAME_LEN samples (the usual choice for analysis with overlap)."""
    n = np.arange(C.FRAME_LEN)
    return (0.5 - 0.5 * np.cos(2.0 * math.pi * n / C.FRAME_LEN)).astype(np.float32)


class MFCC(nn.Module):
    """Batched torch version of the front end (used for training; same maths as mcu/mfcc.c)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.from_numpy(hann_window()))
        self.register_buffer("mel", torch.from_numpy(mel_filterbank()))
        self.register_buffer("dct", torch.from_numpy(dct_matrix()))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave [B, 16000] float in [-1, 1]  ->  mfcc [B, 49, 10] (not yet normalised)."""
        frames = wave.unfold(-1, C.FRAME_LEN, C.HOP) * self.window          # [B, 49, 640]
        spec = torch.fft.rfft(frames, n=C.N_FFT)                             # [B, 49, 513] complex
        power = spec.real ** 2 + spec.imag ** 2
        logmel = torch.log(power @ self.mel + C.LOG_EPS)                     # [B, 49, 40]
        return logmel @ self.dct                                             # [B, 49, 10]


class Normalizer(nn.Module):
    """(mfcc - mean) / std per coefficient. The statistics come from the training set (train.py) and are frozen."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(C.N_MFCC) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("std", torch.ones(C.N_MFCC) if std is None else torch.as_tensor(std).float())

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        return (mfcc - self.mean) / self.std


class FrontEnd(nn.Module):
    """waveform [B, 16000] -> network input [B, 1, 49, 10] (1 channel, H = time, W = cepstral coefficient)."""

    def __init__(self, mean=None, std=None):
        super().__init__()
        self.mfcc = MFCC()
        self.norm = Normalizer(mean, std)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mfcc(wave)).unsqueeze(1)
