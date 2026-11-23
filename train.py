"""Train the DS-CNN keyword spotter: float training -> BatchNorm folding + int8 calibration -> quantisation-aware training.

  python -I train.py --task commands12 --out runs/commands12          (12-class Speech Commands benchmark)
  python -I train.py --task marvin     --out runs/marvin              (wake word "marvin" vs everything else)

Outputs in --out:  log.txt, float_best.pt, qat_best.pt, int8.pkl (integer model for the MCU / NPU + test inputs)
Runs on the CPU (the model is 2.7 M MAC per clip; an epoch is about a minute on 8 threads).
SPDX-License-Identifier: MIT
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works with python -I too
from kws import config as C  # noqa: E402
from kws.dataset import SpeechCommands, TrainEpoch, eval_set, official_test_set
from kws.features import MFCC, FrontEnd
from kws.model import DSCNN, count
from kws.quant import QDSCNN, calibrate, export_int8, fold_batchnorm, param_bytes, quantize_input

TASK_RATIOS = {                       # (unknown_ratio, silence_ratio) per epoch, relative to the target-word count
    "commands12": (0.1, 0.1),         # Warden's recipe: unknown and silence each ~ one word class
    "marvin": (4.0, 1.0),             # wake word: negatives dominate in real life, so show many unknown words
}


def log(f, msg):
    print(msg, flush=True)
    f.write(msg + "\n"); f.flush()


@torch.no_grad()
def predict(model, front, waves, bs=512):
    model.eval()
    out = []
    for i in range(0, len(waves), bs):
        out.append(model(front(torch.from_numpy(waves[i:i + bs]))))
    return torch.cat(out).numpy()


def scores(model, front, waves, y):
    """(balanced accuracy, plain accuracy). Checkpoints are chosen by BALANCED accuracy = mean recall over the classes.
    Plain accuracy on our validation set (every val clip; ~60 % are other words) mostly measures "_unknown_" and
    rewards a model biased to say unknown -- the first runs (runs/archive_selection_by_plain_acc) picked checkpoints
    that way: int8 looked better than float there, but was 0.5 % worse on the balanced official test set."""
    pred = predict(model, front, waves).argmax(1)
    rec = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(rec)), float(np.mean(pred == y))


def run_epoch(model, front, loader, opt, sched):
    model.train()
    lossf = nn.CrossEntropyLoss()
    tot, n, correct = 0.0, 0, 0
    for wave, y in loader:
        with torch.no_grad():
            x = front(wave)
        logits = model(x)
        loss = lossf(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(y); n += len(y)
        correct += (logits.argmax(1) == y).sum().item()
    return tot / n, correct / n


def cosine(opt, total_steps, warmup):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="commands12", choices=list(C.TASKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/packed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--qat_epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--qat_lr", type=float, default=2e-4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    lf = open(os.path.join(a.out, "log.txt"), "a")
    sc = SpeechCommands(a.data, a.task)
    classes = sc.classes
    unk_r, sil_r = TASK_RATIOS[a.task]
    log(lf, f"task {a.task}: classes {classes}; args {vars(a)}")

    # ---- feature normalisation statistics (from augmented training audio, so they match what the net sees)
    stats_ds = TrainEpoch(sc, unk_r, sil_r, seed=999)
    sub = torch.utils.data.Subset(stats_ds, range(min(6000, len(stats_ds))))
    mf = MFCC()
    feats = torch.cat([mf(w) for w, _ in torch.utils.data.DataLoader(sub, batch_size=500, num_workers=a.workers)])
    mean, std = feats.mean(dim=(0, 1)), feats.std(dim=(0, 1))
    front = FrontEnd(mean, std)
    log(lf, f"MFCC mean {np.round(mean.numpy(), 2)}  std {np.round(std.numpy(), 2)}")

    val_w, val_y = eval_set(sc, split=1, n_silence=400)
    log(lf, f"validation items {len(val_y)} (all val clips + 400 silence)")

    # ---- stage 1: float training
    model = DSCNN(len(classes))
    params, macs = count(model)
    log(lf, f"DS-CNN: {params} parameters (after BN folding), {macs / 1e6:.2f} M MAC per inference")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(stats_ds) / a.bs)
    sched = cosine(opt, steps, warmup=300)
    best = -1
    for ep in range(a.epochs):
        t0 = time.time()
        ds = TrainEpoch(sc, unk_r, sil_r, seed=a.seed * 1000 + ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
        loss, acc = run_epoch(model, front, dl, opt, sched)
        vbal, vacc = scores(model, front, val_w, val_y)
        if vbal > best:
            best = vbal
            torch.save(dict(model=model.state_dict(), mean=mean, std=std, classes=classes), f"{a.out}/float_best.pt")
        log(lf, f"float ep {ep:2d}  loss {loss:.4f}  train acc {acc:.4f}  val bal {vbal:.4f}  val acc {vacc:.4f}  "
                f"(best bal {best:.4f})  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f} s")

    # ---- stage 2: fold BatchNorm, calibrate int8 scales
    model.load_state_dict(torch.load(f"{a.out}/float_best.pt", weights_only=False)["model"])
    folded = fold_batchnorm(model)
    fa, _ = scores(folded, front, val_w, val_y)
    cal_ds = TrainEpoch(sc, unk_r, sil_r, seed=4242)
    cal = [front(w) for k, (w, _) in zip(range(30), torch.utils.data.DataLoader(cal_ds, batch_size=128,
                                                                                    shuffle=True, num_workers=a.workers))]
    scales = calibrate(folded, cal)
    q = QDSCNN(folded, scales)
    pa, _ = scores(q, front, val_w, val_y)
    log(lf, f"folded float val bal {fa:.4f};  int8 before QAT (PTQ) {pa:.4f};  scales {scales}")

    # ---- stage 3: quantisation-aware training (the int8 network is trained directly; scales stay fixed)
    opt = torch.optim.AdamW(q.parameters(), lr=a.qat_lr, weight_decay=0.0)
    sched = cosine(opt, a.qat_epochs * math.ceil(len(stats_ds) / a.bs), warmup=100)
    best = pa
    torch.save(dict(model=q.state_dict(), scales=scales, mean=mean, std=std, classes=classes), f"{a.out}/qat_best.pt")
    for ep in range(a.qat_epochs):
        t0 = time.time()
        ds = TrainEpoch(sc, unk_r, sil_r, seed=a.seed * 1000 + 500 + ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True)
        loss, acc = run_epoch(q, front, dl, opt, sched)
        vbal, vacc = scores(q, front, val_w, val_y)
        if vbal > best:
            best = vbal
            torch.save(dict(model=q.state_dict(), scales=scales, mean=mean, std=std, classes=classes),
                       f"{a.out}/qat_best.pt")
        log(lf, f"QAT ep {ep:2d}  loss {loss:.4f}  train acc {acc:.4f}  val bal {vbal:.4f}  val acc {vacc:.4f}  "
                f"(best bal {best:.4f})  {time.time() - t0:.0f} s")

    # ---- export the integer model (+ the official test inputs, already quantised, for the MCU / NPU checks)
    q.load_state_dict(torch.load(f"{a.out}/qat_best.pt", weights_only=False)["model"])
    P = export_int8(q)
    tw, _ = official_test_set(a.data, a.task)
    x_q = quantize_input(front(torch.from_numpy(tw)).numpy(), P["f_input"])
    with open(f"{a.out}/int8.pkl", "wb") as fh:
        pickle.dump(dict(int8=P, mean=mean.numpy(), std=std.numpy(), classes=classes, test_inputs=x_q), fh)
    log(lf, f"int8 model exported: {param_bytes(P)} bytes of weights+biases -> {a.out}/int8.pkl")


if __name__ == "__main__":
    main()
