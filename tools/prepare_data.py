"""Pack Google Speech Commands v2 into a few .npy files (one int16 row of 16000 samples per clip).

Dataset: Speech Commands v0.02, P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition",
arXiv:1804.03209 (2018). License CC-BY-4.0. 105,829 one-second utterances of 35 words by 2,618 speakers, 16 kHz.
  download: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz           (train/val/test pool)
            http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz  (the official 12-class test set)

Splits: the dataset ships validation_list.txt and testing_list.txt. They were made by hashing the SPEAKER id, so one
speaker is never in two splits (otherwise the test would partly measure "do I know this voice").

Output (data/packed/):
  pool_wave.npy  int16 [N, 16000]   every clip, zero-padded at the end to 1 s (as Warden's input_data.py)
  pool_meta.npz  word [N] (str), speaker [N] (str), split [N] (0 train, 1 val, 2 test)
  noise_train.npy / noise_test.npy  float32 1-D: the 6 _background_noise_ recordings, first 80 % / last 20 % of each
                 (the test-time noise is never heard in training)
  official_test_wave.npy int16 [4890, 16000] + official_test_meta.npz label [4890] (str: word, _silence_, _unknown_)
Usage: python -I tools/prepare_data.py data/speech_commands_v2 data/speech_commands_test_set_v2 data/packed
SPDX-License-Identifier: MIT
"""
import os
import sys
import wave

import numpy as np


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, path
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    out = np.zeros(16000, np.int16)
    out[: min(16000, len(x))] = x[:16000]
    return out


def main(src: str, src_test: str, dst: str):
    os.makedirs(dst, exist_ok=True)
    val = set(open(os.path.join(src, "validation_list.txt")).read().split())
    test = set(open(os.path.join(src, "testing_list.txt")).read().split())
    words = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and not d.startswith("_"))
    rel = [f"{w}/{f}" for w in words for f in sorted(os.listdir(os.path.join(src, w))) if f.endswith(".wav")]
    print(f"{len(words)} words, {len(rel)} clips")
    waves = np.zeros((len(rel), 16000), np.int16)
    for i, r in enumerate(rel):
        waves[i] = read_wav(os.path.join(src, r))
    np.save(os.path.join(dst, "pool_wave.npy"), waves)
    np.savez(os.path.join(dst, "pool_meta.npz"),
             word=np.array([r.split("/")[0] for r in rel]),
             speaker=np.array([r.split("/")[1].split("_nohash_")[0] for r in rel]),
             split=np.array([2 if r in test else 1 if r in val else 0 for r in rel], np.int8),
             path=np.array(rel))

    tr, te = [], []
    nd = os.path.join(src, "_background_noise_")
    for f in sorted(os.listdir(nd)):
        if f.endswith(".wav"):
            with wave.open(os.path.join(nd, f), "rb") as w:
                x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
            k = int(0.8 * len(x))
            tr.append(x[:k]); te.append(x[k:])
            print(f"noise {f}: {len(x) / 16000:.0f} s")
    np.save(os.path.join(dst, "noise_train.npy"), np.concatenate(tr))
    np.save(os.path.join(dst, "noise_test.npy"), np.concatenate(te))

    labs = sorted(d for d in os.listdir(src_test) if os.path.isdir(os.path.join(src_test, d)))
    trel = [f"{l}/{f}" for l in labs for f in sorted(os.listdir(os.path.join(src_test, l))) if f.endswith(".wav")]
    tw = np.stack([read_wav(os.path.join(src_test, r)) for r in trel])
    np.save(os.path.join(dst, "official_test_wave.npy"), tw)
    np.savez(os.path.join(dst, "official_test_meta.npz"), label=np.array([r.split("/")[0] for r in trel]),
             path=np.array(trel))
    s = np.load(os.path.join(dst, "pool_meta.npz"))["split"]
    print(f"train {np.sum(s == 0)}  val {np.sum(s == 1)}  test {np.sum(s == 2)}  official test {len(trel)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
