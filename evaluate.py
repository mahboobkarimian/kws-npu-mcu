"""Accuracy checks for a trained model -> <run>/eval.json (the report is built from it).

  python -I evaluate.py runs/commands12
  python -I evaluate.py runs/marvin          (adds the always-on streaming test: misses vs false alarms per hour)

Checks (all on speakers never seen in training):
  1. accuracy, float and int8, on the official 4,890-clip v2 test set (commands12) and on the full test split
     (11,005 clips: every target AND every other word, + 1,000 noise-only clips made from held-out noise)
  2. int8 PyTorch model == integer reference (the code that the MCU and NPU run), logit for logit
  3. per-class precision / recall + confusion matrix
  4. noise robustness: held-out noise at SNR 20 / 10 / 5 / 0 dB
  5. speaker pitch: accuracy per f0 band (YIN estimate per clip)
  6. pitch/formant shift: the test clips resampled x0.8 .. x1.25 (a lower / higher, smaller / larger voice)
  7. (wake word) streaming over 3 h of test audio, one decision every 100 ms: false rejects vs false alarms per hour
SPDX-License-Identifier: MIT
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kws import config as C                                                 # noqa: E402
from kws.dataset import SpeechCommands, eval_set, mix_at_snr, official_test_set, speed_perturb  # noqa: E402
from kws.features import FrontEnd                                          # noqa: E402
from kws.model import DSCNN                                                 # noqa: E402
from kws.pitch import clip_f0                                               # noqa: E402
from kws.quant import QDSCNN, fold_batchnorm, quantize_input, run_int8      # noqa: E402

F0_BANDS = [(0, 100), (100, 130), (130, 170), (170, 220), (220, 600)]
SNRS = [20, 10, 5, 0]
SPEEDS = [0.8, 0.9, 1.0, 1.12, 1.25]
# streaming wake-word detector (same numbers would go into the device firmware)
INFER_EVERY = 5          # frames: one decision every 100 ms
SMOOTH = 3               # average the "marvin" probability over the last 3 decisions (300 ms)
REFRACTORY_S = 1.0       # after a trigger, ignore 1 s


def load(run):
    f = torch.load(f"{run}/float_best.pt", weights_only=False)
    qd = torch.load(f"{run}/qat_best.pt", weights_only=False)
    classes = f["classes"]
    front = FrontEnd(f["mean"], f["std"])
    fm = DSCNN(len(classes)); fm.load_state_dict(f["model"]); fm.eval()
    qm = QDSCNN(fold_batchnorm(DSCNN(len(classes))), qd["scales"]); qm.load_state_dict(qd["model"]); qm.eval()
    with open(f"{run}/int8.pkl", "rb") as fh:
        P = pickle.load(fh)["int8"]
    return classes, front, fm, qm, P


@torch.no_grad()
def logits(model, front, waves, bs=1024):
    return torch.cat([model(front(torch.from_numpy(np.ascontiguousarray(waves[i:i + bs]))))
                      for i in range(0, len(waves), bs)]).numpy()


def acc(lg, y):
    return float(np.mean(lg.argmax(1) == y))


def confusion(pred, y, n):
    m = np.zeros((n, n), int)
    np.add.at(m, (y, pred), 1)
    return m


def streaming(sc, front, qm, classes, snr, seed=7):
    """Concatenate every test clip (random order) into one long recording, optionally add held-out noise, and run the
    detector every 100 ms on the last 1 s. A trigger whose window ends inside [start, start + 2 s) of a 'marvin'
    clip is a hit for that clip; any other trigger is a false alarm."""
    rng = np.random.default_rng(seed)
    idx = sc.indices(2)
    idx = idx[rng.permutation(len(idx))]
    target = sc.label[idx] == classes.index("marvin")
    stream = (np.asarray(sc.wave[idx], np.float32) / 32768.0).reshape(-1)
    if snr is not None:
        n = np.resize(sc.noise["test"], len(stream))
        # level: SNR against the average speech level of the clips
        stream = stream + n * np.sqrt(np.mean(stream ** 2) / (np.mean(n ** 2) * 10 ** (snr / 10)))
    hours = len(stream) / C.SAMPLE_RATE / 3600
    with torch.no_grad():
        mf = front.mfcc(torch.from_numpy(stream[None]))[0]                                   # [frames, 10]
        feats = front.norm(mf)
        win = feats.unfold(0, C.N_FRAMES, INFER_EVERY).permute(0, 2, 1).unsqueeze(1)       # [W, 1, 49, 10]
        lg = torch.cat([qm(win[i:i + 2048]) for i in range(0, len(win), 2048)])
    p = torch.softmax(lg, 1)[:, classes.index("marvin")].numpy()
    ps = np.convolve(p, np.ones(SMOOTH) / SMOOTH)[: len(p)]
    t_end = (np.arange(len(p)) * INFER_EVERY * C.HOP + C.CLIP_SAMPLES) / C.SAMPLE_RATE      # window end time (s)
    starts = np.nonzero(target)[0] * 1.0                                                    # clip k starts at k s
    words = sc.word[idx]                                                                    # word of clip k
    curve, fa_words = [], {}
    for th in np.round(np.concatenate([np.arange(0.05, 0.95, 0.05), [0.95, 0.97, 0.98, 0.99, 0.995]]), 3):
        trig, last = [], -1e9
        for k in np.nonzero(ps >= th)[0]:
            if t_end[k] - last >= REFRACTORY_S:
                trig.append(t_end[k]); last = t_end[k]
        trig = np.array(trig)
        hit = np.zeros(len(starts), bool)
        fa = 0
        for t in trig:
            j = np.searchsorted(starts, t, side="right") - 1
            if j >= 0 and t - starts[j] < 2.0:
                hit[j] = True
            else:
                fa += 1
                if th == 0.95:                                   # which word was being said when it fired?
                    w = words[min(len(words) - 1, int(t - 0.5))]
                    fa_words[w] = fa_words.get(w, 0) + 1
        curve.append(dict(threshold=float(th), frr=float(1 - hit.mean()), fa=int(fa), fa_per_hour=fa / hours))
    fa_words = dict(sorted(fa_words.items(), key=lambda kv: -kv[1]))
    return dict(hours=hours, n_keywords=int(target.sum()), n_windows=len(p), curve=curve, fa_words_at_0_95=fa_words)


def main(run):
    t0 = time.time()
    torch.set_num_threads(8)
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/packed")
    classes, front, fm, qm, P = load(run)
    task = "marvin" if classes[-1] == "marvin" and len(classes) == 3 else "commands12"
    sc = SpeechCommands(data, task)
    R = dict(task=task, classes=classes)

    # 1+2: official test set and full test split; int8 torch == integer reference
    if task == "commands12":
        ow, oy = official_test_set(data, task)
        lf, lq = logits(fm, front, ow), logits(qm, front, ow)
        ref = run_int8(P, quantize_input(front(torch.from_numpy(ow)).numpy(), P["f_input"]))
        same = np.all(np.round(lq * 2.0 ** P["layers"][-1]["f_out"]).astype(int) == ref, 1)
        R["official"] = dict(n=len(oy), float=acc(lf, oy), int8=acc(lq, oy), int8_equals_integer_ref=float(same.mean()))
        print("official test:", R["official"])
    tw, ty = eval_set(sc, split=2, n_silence=1000, noise="test")
    lf, lq = logits(fm, front, tw), logits(qm, front, tw)
    R["test_split"] = dict(n=len(ty), float=acc(lf, ty), int8=acc(lq, ty))
    print("full test split:", R["test_split"])

    # 3: per class
    pred = lq.argmax(1)
    cm = confusion(pred, ty, len(classes))
    R["confusion_int8"] = cm.tolist()
    R["per_class"] = {c: dict(recall=float(cm[i, i] / max(1, cm[i].sum())), precision=float(cm[i, i] / max(1, cm[:, i].sum())),
                              n=int(cm[i].sum())) for i, c in enumerate(classes)}

    # 4: noise (speech clips only; the noise-only clips would just become louder noise)
    sp = slice(0, len(ty) - 1000)
    w_sp, y_sp = tw[sp], ty[sp]
    rng = np.random.default_rng(3)
    R["noise"] = {"clean": dict(float=acc(lf[sp], y_sp), int8=acc(lq[sp], y_sp))}
    for snr in SNRS:
        noisy = np.stack([mix_at_snr(x, sc.noise_segment(rng, "test"), snr) for x in w_sp]).astype(np.float32)
        R["noise"][f"{snr} dB"] = dict(float=acc(logits(fm, front, noisy), y_sp), int8=acc(logits(qm, front, noisy), y_sp))
        print("SNR", snr, R["noise"][f"{snr} dB"])

    # 5: speaker pitch bands
    f0_file = os.path.join(data, "test_f0.npy")
    if not os.path.exists(f0_file):
        np.save(f0_file, clip_f0(w_sp))
    f0 = np.load(f0_file)
    R["f0_percentiles"] = np.nanpercentile(f0, [5, 25, 50, 75, 95]).tolist()
    R["pitch"] = {}
    cor_f, cor_q = lf[sp].argmax(1) == y_sp, lq[sp].argmax(1) == y_sp
    for lo, hi in F0_BANDS:
        m = (f0 >= lo) & (f0 < hi)
        R["pitch"][f"{lo}-{hi} Hz"] = dict(n=int(m.sum()), float=float(cor_f[m].mean()), int8=float(cor_q[m].mean()))
    m = np.isnan(f0)
    R["pitch"]["f0 not found"] = dict(n=int(m.sum()), float=float(cor_f[m].mean()), int8=float(cor_q[m].mean()))
    print("pitch:", R["pitch"])

    # 6: pitch + formant shift by resampling (x1.25 = +3.9 semitones, formants +25 %: closer to a child's voice)
    R["speed"] = {}
    for s in SPEEDS:
        ws = w_sp if s == 1.0 else np.stack([speed_perturb(x, s) for x in w_sp])
        R["speed"][f"x{s}"] = dict(float=acc(logits(fm, front, ws), y_sp), int8=acc(logits(qm, front, ws), y_sp))
        print("speed", s, R["speed"][f"x{s}"])

    # 7: always-on streaming (wake word)
    if task == "marvin":
        R["streaming"] = {"clean": streaming(sc, front, qm, classes, None), "10 dB": streaming(sc, front, qm, classes, 10)}
        for k, v in R["streaming"].items():
            print("streaming", k, v["hours"], "h", [c for c in v["curve"] if c["threshold"] in (0.5, 0.8, 0.9, 0.95)])
    for f in ("npu_check.json", "mcu_check.json"):
        if os.path.exists(f"{run}/{f}"):
            R[f[:-5]] = json.load(open(f"{run}/{f}"))
    json.dump(R, open(f"{run}/eval.json", "w"), indent=1)
    print(f"-> {run}/eval.json  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1])
