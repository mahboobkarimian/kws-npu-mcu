"""Compile the int8 DS-CNN for the Phoenix NPU, run it on the NPU, and compare every output with the integer reference.

  python -I npu/npu_check.py <int8.pkl | random> <out_dir> [--n 2000]

  random   : untrained weights (a probe: does every op land on the NPU, is it bit-exact?)
  int8.pkl : a trained model exported by train.py (also holds real test inputs)

Uses the repository's Ryzen AI 1.8 Linux toolchain (scripts/npu.sh, 1x4 overlay = work/configs/phx_18_1x4.json).
SPDX-License-Identifier: MIT
"""
import argparse
import os
import pickle
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # XDNA1NPU
from kws import config as C                               # noqa: E402
from kws.onnx_export import build_qdq                     # noqa: E402
from kws.quant import run_int8                            # noqa: E402

ENV = dict(os.environ, PHX_CFG=os.path.join(ROOT, "work/configs/phx_18_1x4.json"))


def random_model(num_classes=12, seed=0):
    import torch
    from kws.model import DSCNN
    from kws.quant import QDSCNN, calibrate, export_int8, fold_batchnorm
    torch.manual_seed(seed)
    m = DSCNN(num_classes)
    for mod in m.modules():                                   # non-trivial BN statistics
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2); mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.uniform_(-0.2, 0.2)
    f = fold_batchnorm(m)
    batches = [torch.randn(64, 1, C.N_FRAMES, C.N_MFCC) for _ in range(4)]
    q = QDSCNN(f, calibrate(f, batches))
    return export_int8(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.model == "random":
        P = random_model()
        rng = np.random.default_rng(1)
        x_q = np.clip(np.round(rng.normal(0, 2.0 ** P["f_input"], (a.n, 1, C.N_FRAMES, C.N_MFCC))), -128, 127)
        x_q = x_q.astype(np.int8)
    else:
        with open(a.model, "rb") as fh:
            d = pickle.load(fh)
        P, x_q = d["int8"], d["test_inputs"][: a.n]
    n_cls = P["layers"][-1]["w"].shape[0]
    onnx_path = os.path.join(a.out, "dscnn_int8.onnx")
    build_qdq(P, n_cls, onnx_path)

    # 1) compile (one-time; the context model holds the NPU program)
    r = subprocess.run([os.path.join(ROOT, "scripts/npu.sh"), "compile", "18", onnx_path, a.out], env=ENV,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    # saved without this machine's absolute paths (the json is published with the results)
    r_compile = r.stdout.strip().replace(os.path.dirname(HERE) + os.sep, "").splitlines()
    ctx = re.search(r"output: (\S+)", r.stdout).group(1)

    # 2) run all inputs on the NPU in one session; inputs are exact int8 grid values as float32
    inp = os.path.join(a.out, "inputs.f32")
    (x_q.astype(np.float32) * 2.0 ** -P["f_input"]).tofile(inp)
    r = subprocess.run([os.path.join(ROOT, "scripts/npu.sh"), "run", ctx, f"--in={inp}", f"--frames={len(x_q)}",
                        f"--out={os.path.join(a.out, 'npu')}"], env=ENV, capture_output=True, text=True)
    run_lines = [l for l in r.stdout.splitlines() if "RESULT" in l or "ERROR" in l]
    print("\n".join(run_lines))
    npu = np.fromfile(os.path.join(a.out, "npu_0.bin"), np.float32).reshape(len(x_q), n_cls)
    npu_q = np.round(npu * 2.0 ** P["layers"][-1]["f_out"]).astype(np.int32)

    # 3) compare with the integer reference (the same integers the MCU code computes)
    ref = run_int8(P, x_q).astype(np.int32)
    diff = np.abs(npu_q - ref)
    print(f"NPU vs integer reference: {np.mean(diff == 0) * 100:.3f} % of logits identical, "
          f"{np.mean(np.all(diff == 0, 1)) * 100:.2f} % of inputs fully identical, max |diff| {diff.max()} LSB, "
          f"argmax agree {np.mean(npu_q.argmax(1) == ref.argmax(1)) * 100:.2f} %")
    os.remove(inp)
    if a.model != "random":
        import json
        from kws.dataset import official_test_set
        _, labels = official_test_set(os.path.join(os.path.dirname(HERE), "data/packed"), "commands12")
        labels = labels[: len(x_q)]
        npu_acc = float(np.mean(npu_q.argmax(1) == labels)) if n_cls == 12 else None
        if npu_acc is not None:
            print(f"accuracy of the NPU's own outputs on the official test set: {npu_acc * 100:.2f} %")
        res = dict(n=len(x_q), logits_identical_pct=float(np.mean(diff == 0) * 100),
                   inputs_identical_pct=float(np.mean(np.all(diff == 0, 1)) * 100), max_diff=int(diff.max()),
                   compile=r_compile, run=run_lines, npu_official_test_acc=npu_acc)
        json.dump(res, open(os.path.join(os.path.dirname(a.model), "npu_check.json"), "w"), indent=1)
    return P, x_q, npu_q, ref


if __name__ == "__main__":
    main()
