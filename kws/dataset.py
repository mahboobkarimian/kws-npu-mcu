"""Speech Commands v2 for keyword spotting: class labels, the official speaker-disjoint splits, and augmentation.

Dataset: Google Speech Commands v0.02 (Warden 2018, CC-BY-4.0), packed by tools/prepare_data.py.
  train 84,843 / val 9,981 / test 11,005 clips (official lists, split by speaker), 35 words, 2,618 speakers.
  _background_noise_: 6 recordings (dishes, cat, exercise bike, pink, tap, white). First 80 % of each -> training
  noise, last 20 % -> test noise only (so the noise-robustness test uses noise the model never heard).

Classes: the task's target words, plus
  _unknown_ = every other word of the 35 (teaches "speech, but not my keyword": the main defence against false alarms)
  _silence_ = background noise only, no speech.

An "epoch" (Warden's recipe, input_data.py): all target-word clips + a fresh random draw of unknown clips and silence
clips (their number is a ratio of the target count), re-drawn every epoch so all unknown words get seen.

Augmentation, in the order applied (why):
  1. speed/pitch perturbation x0.85..1.15 (resampling): moves f0 AND formants, like a different (smaller/larger) speaker.
     This is our main tool for pitch robustness (Ko et al. 2015, "Audio augmentation for speech recognition").
  2. time shift +-100 ms: the word is not always centred in a real 1 s window.
  3. gain +-10 dB: microphones and distances differ (MFCC c0 follows the level).
  4. background noise, p = 0.8, SNR 0..25 dB.
SPDX-License-Identifier: MIT
"""
import os
from typing import List, Optional

import numpy as np
import torch

from . import config as C


def resample_fft(x: np.ndarray, n_out: int) -> np.ndarray:
    """Band-limited resampling by zero-padding / truncating the spectrum (no aliasing)."""
    X = np.fft.rfft(x)
    n_bins = n_out // 2 + 1
    Y = np.zeros(n_bins, np.complex128)
    k = min(len(X), n_bins)
    Y[:k] = X[:k]
    return np.fft.irfft(Y, n_out) * (n_out / len(x))


def speed_perturb(x: np.ndarray, factor: float) -> np.ndarray:
    """Play the clip `factor` times faster: pitch and formants x factor, duration / factor. Kept at 1 s, centred."""
    y = resample_fft(x, int(round(len(x) / factor)))
    out = np.zeros(len(x), np.float32)
    if len(y) >= len(x):
        s = (len(y) - len(x)) // 2
        out[:] = y[s:s + len(x)]
    else:
        s = (len(x) - len(y)) // 2
        out[s:s + len(y)] = y
    return out


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = np.mean(speech ** 2) + 1e-10
    pn = np.mean(noise ** 2) + 1e-10
    return speech + noise * np.sqrt(ps / (pn * 10.0 ** (snr_db / 10.0)))


class SpeechCommands:
    """Read-only view of the packed data. Labels follow config.TASKS[task]."""

    def __init__(self, packed: str, task: str):
        self.packed, self.task = packed, task
        self.classes: List[str] = C.TASKS[task]
        self.wave = np.load(os.path.join(packed, "pool_wave.npy"), mmap_mode="r")
        meta = np.load(os.path.join(packed, "pool_meta.npz"))
        self.word, self.split, self.speaker = meta["word"], meta["split"], meta["speaker"]
        self.noise = {"train": np.load(os.path.join(packed, "noise_train.npy")),
                      "test": np.load(os.path.join(packed, "noise_test.npy"))}
        cls = {w: i for i, w in enumerate(self.classes)}
        self.label = np.array([cls.get(w, 1) for w in self.word], np.int64)        # 1 = _unknown_

    def indices(self, split: int, target: Optional[bool] = None) -> np.ndarray:
        """split 0/1/2 = train/val/test; target True = only target words, False = only unknown words."""
        m = self.split == split
        if target is True:
            m &= self.label >= 2
        elif target is False:
            m &= self.label == 1
        return np.nonzero(m)[0]

    def clip(self, i: int) -> np.ndarray:
        return self.wave[i].astype(np.float32) / 32768.0

    def noise_segment(self, rng: np.random.Generator, which: str = "train") -> np.ndarray:
        n = self.noise[which]
        s = rng.integers(0, len(n) - C.CLIP_SAMPLES)
        return n[s:s + C.CLIP_SAMPLES].astype(np.float32)

    def silence_clip(self, rng: np.random.Generator, which: str = "train") -> np.ndarray:
        """A _silence_ item: 1 s of background noise at a random level, -70..-20 dBFS RMS (a quiet room .. a loud one)."""
        x = self.noise_segment(rng, which)
        return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10.0 ** (rng.uniform(-70, -20) / 20.0)).astype(np.float32)


class TrainEpoch(torch.utils.data.Dataset):
    """One epoch of augmented training items: (waveform [16000] float32, label)."""

    SILENCE_IDX = -1

    def __init__(self, sc: SpeechCommands, unknown_ratio: float, silence_ratio: float, seed: int,
                 split: int = 0, augment: bool = True):
        self.sc, self.augment, self.seed = sc, augment, seed
        rng = np.random.default_rng(seed)
        tgt = sc.indices(split, target=True)
        unk = sc.indices(split, target=False)
        n_unk = min(len(unk), int(round(unknown_ratio * len(tgt))))
        n_sil = int(round(silence_ratio * len(tgt)))
        self.items = np.concatenate([tgt, rng.choice(unk, n_unk, replace=False),
                                     np.full(n_sil, self.SILENCE_IDX)])
        rng.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k))
        i = self.items[k]
        if i == self.SILENCE_IDX:                                         # _silence_: noise only, any level
            return torch.from_numpy(self.sc.silence_clip(rng)), 0
        x = self.sc.clip(i)
        if self.augment:
            if rng.random() < 0.5:
                x = speed_perturb(x, rng.uniform(0.85, 1.15))
            sh = rng.integers(-1600, 1601)
            x = np.roll(x, sh)
            if sh > 0:
                x[:sh] = 0
            elif sh < 0:
                x[sh:] = 0
            x = x * 10.0 ** (rng.uniform(-10, 10) / 20.0)
            if rng.random() < 0.8:
                x = mix_at_snr(x, self.sc.noise_segment(rng), rng.uniform(0, 25))
        return torch.from_numpy(x.astype(np.float32)), int(self.sc.label[i])


def eval_set(sc: SpeechCommands, split: int, n_silence: int, seed: int = 1234, noise: str = "train"):
    """Every clip of the split (all target AND all unknown words) + n_silence noise-only clips. No augmentation.
    Returns waves [N,16000] float32, labels [N]."""
    idx = sc.indices(split)
    rng = np.random.default_rng(seed)
    waves = np.asarray(sc.wave[idx], np.float32) / 32768.0
    sil = np.stack([sc.silence_clip(rng, noise) for _ in range(n_silence)])
    return np.concatenate([waves, sil]), np.concatenate([sc.label[idx], np.zeros(n_silence, np.int64)])


def official_test_set(packed: str, task: str):
    """The 4,890-clip official v2 12-class test set (speech_commands_test_set_v0.02): fixed silence/unknown items."""
    w = np.load(os.path.join(packed, "official_test_wave.npy")).astype(np.float32) / 32768.0
    lab = np.load(os.path.join(packed, "official_test_meta.npz"))["label"]
    cls = {c: i for i, c in enumerate(C.TASKS[task])}
    return w, np.array([cls.get(l, 1) for l in lab], np.int64)
