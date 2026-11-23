# DS-CNN keyword spotting for Arm Cortex-M and the AMD Phoenix NPU

## Disclaimer
**This coder is an AI agent for this project. No human review of the code has been done, but human supervision was in place. Before using the code and models, read and check them.**


A depthwise-separable CNN (DS-CNN) that listens to the last 1 second of 16 kHz audio and says which keyword it heard.
It has about 22 k weights and needs 2.6 M multiply-adds per decision.

The same int8 model runs, with **bit-identical outputs**, on three targets:

| target | how |
|---|---|
| PyTorch | float training, then int8 quantisation-aware training |
| Arm Cortex-M microcontrollers (e.g. STM32) | plain C99, int8 |
| AMD Phoenix NPU (XDNA1, Ryzen 7040/8040) | int8 QDQ ONNX compiled with Ryzen AI 1.8 (Linux) |

## Trained models

| model | classes |
|---|---|
| `commands12` | `_silence_`, `_unknown_`, yes, no, up, down, left, right, on, off, stop, go |
| `marvin` (wake word) | `_silence_`, `_unknown_`, marvin |

Both are trained on **Google Speech Commands v0.02** (Warden 2018, arXiv:1804.03209, CC BY 4.0), using the
dataset's official speaker-separated train / validation / test splits.

## Processing chain

```
 16 kHz audio
   │  every 20 ms
   ▼
 MFCC front end (float)          640-sample frame → Hann → FFT 1024 → power → 40 mel bands (20 Hz–4 kHz)
                                 → log → DCT → 10 coefficients → per-coefficient normalisation → int8
   │  last 49 frames = 1 s = a 49 × 10 input
   ▼  every 100 ms
 DS-CNN (int8)                   → one score (logit) per class
   │
   ▼
 softmax → average of the last 3 decisions → threshold → detection (then 1 s hold-off)
```

## Network (`kws/model.py`)

Shapes are [channels, time frames, MFCC coefficients].

| # | layer | operation | output | weights | MACs |
|---|---|---|---|---|---|
| in | input | normalised MFCC | [1, 49, 10] | | |
| 1 | `FeatureStem` | conv 9×4, stride 2, ReLU | [64, 25, 5] | 2,368 | 288,000 |
| 2–5 | `DSBlock` ×4: `DepthwiseConv` | 3×3 per channel, stride 1, ReLU | [64, 25, 5] | 640 each | 72,000 each |
| | `DSBlock` ×4: `PointwiseConv` | 1×1, 64 → 64, ReLU | [64, 25, 5] | 4,160 each | 512,000 each |
| 6 | `GlobalAvgPool` | mean over 25 × 5 | [64, 1, 1] | | |
| 7 | `Classifier` | 1×1 conv, 64 → classes | [classes] | 780 | 768 |
| | **total** (12 classes) | | | **22,348** | **2.62 M** |

What each layer does:
- **FeatureStem:** detects 64 kinds of short sound patterns (about 200 ms × 4 coefficients) and halves both axes.
- **DepthwiseConv:** adds context in time and across coefficients, one channel at a time.
- **PointwiseConv:** mixes the 64 channels at each position.
- **After 4 blocks:** each position sees about 0.5 s.
- **GlobalAvgPool:** makes the result independent of where the word is in the window.

The architecture follows DS-CNN-S from *Hello Edge* (Zhang et al., arXiv:1711.07128), with these hardware
adaptations:

- **Stem kernel 9×4** instead of 10×4: the Phoenix NPU compiler accepts kernel sizes 1 to 9 only.
- **Classifier as a 1×1 conv** instead of a Linear layer, so it runs on the NPU.
- **Softmax on the host:** it is not needed to pick the class.
- **Average pool computed as `round(sum × 33 / 4096)`:** this is the NPU's fixed-point 1/125, and training and the C
  code use it too.
- **BatchNorm folded into the conv weights** before quantisation.

## int8 format

- Weights, biases and activations are all int8.
- Scales are powers of two, with zero point 0.
- Rounding is round-half-up.
- The int32 accumulator is shifted back to int8.

Weights are trained with quantisation-aware training, and activation ranges are calibrated at the 99.9th percentile.
`kws/quant.py` holds the integer reference (`run_int8`), and the C code and the NPU reproduce it exactly.

## Results (commands12, official 4,890-clip test set)

| | accuracy |
|---|---|
| float | 94.2 % |
| int8 (PyTorch, C, NPU: identical outputs) | 93.8 % |

The marvin model recognises 96.9 % of "marvin" test clips and rejects 98.9 % of other words. Full evaluation:
[REPORT.md](REPORT.md).

| deployment | cost |
|---|---|
| Phoenix NPU | 0.6–0.7 ms per inference; the whole network runs as one NPU partition |
| Cortex-M4 (clang `-Os`) | ≈ 32 KB flash (weights 23.5 KB) and ≈ 34 KB RAM (activations 16 KB, FFT 14 KB) |

## Files

```
kws/config.py        front-end constants, class lists
kws/features.py      MFCC front end (PyTorch)
kws/model.py         the DS-CNN, one class per layer type
kws/quant.py         BatchNorm folding, calibration, QAT, int8 export, integer reference
kws/onnx_export.py   int8 QDQ ONNX graph for the NPU
kws/dataset.py       Speech Commands loading and augmentation
kws/pitch.py         pitch (f0) estimator, used by the evaluation
train.py                  float training → QAT → int8 export
evaluate.py               accuracy, noise, pitch and streaming tests → eval.json
export_mcu.py             writes the C model (mcu/generated/) and checks it against PyTorch
npu/npu_check.py          compiles the ONNX model for the NPU, runs it and checks it
tools/prepare_data.py     packs the dataset into numpy arrays
tools/make_report.py      writes REPORT.md

runs/<model>/float_best.pt   float checkpoint: weights, normalisation mean/std, class names
runs/<model>/qat_best.pt     quantisation-aware checkpoint
runs/<model>/int8.pkl        integer model (int8 weights, shifts) + quantised test inputs
runs/<model>/npu/dscnn_int8.onnx                         int8 QDQ ONNX model
runs/<model>/npu/dscnn_int8.phx18_ctx.onnx (+ _VITISAI.bin)  compiled for the Phoenix NPU (Ryzen AI 1.8)

mcu/dscnn_int8.h/.c       int8 inference engine (conv, depthwise conv, average pool)
mcu/kws_frontend.h/.c     MFCC front end in C, whole clip or streaming 20 ms at a time
mcu/generated/            model weights and front-end tables, written by export_mcu.py
mcu/kws_test.c, Makefile  host test program; Cortex-M4/M7 size builds
```

## Usage

**Python (float model):**

```python
import torch
from kws.features import FrontEnd
from kws.model import DSCNN

ck = torch.load("runs/commands12/float_best.pt", weights_only=False)
front = FrontEnd(ck["mean"], ck["std"])
net = DSCNN(len(ck["classes"]))
net.load_state_dict(ck["model"])
net.eval()

# wave: float32 tensor [batch, 16000], 16 kHz, range -1 … 1
with torch.no_grad():
    scores = net(front(wave))
print([ck["classes"][i] for i in scores.argmax(1)])
```

**C (microcontroller):**

```c
#include "dscnn_int8.h"
#include "kws_frontend.h"
#include "generated/kws_model_data.h"

static int8_t arena[KWS_ARENA_BYTES];
static kws_stream_t stream;
int8_t logits[KWS_NUM_CLASSES];

kws_frontend_init();
kws_stream_init(&stream);
/* every 20 ms, with 320 new int16 samples: */
if (kws_stream_push(&stream, hop320) && ++n % 5 == 0)        /* every 100 ms */
    dscnn_run(stream.feat, logits, arena);                  /* logit = q * 2^-KWS_LOGIT_FIX */
```

`mcu/generated/` holds one model at a time. To switch models, run `python -I export_mcu.py runs/<model>/int8.pkl`.

**NPU:** the input of the compiled ONNX model is `mfcc` (float [1, 1, 49, 10], normalised MFCC) and the output is
`logits`. Run it with ONNX Runtime and the Vitis AI execution provider from Ryzen AI 1.8.

## Reproduce

```bash
# data (2.3 GB download)
mkdir -p data/speech_commands_v2 data/speech_commands_test_set_v2
curl -L http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz          | tar xz -C data/speech_commands_v2
curl -L http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz | tar xz -C data/speech_commands_test_set_v2
python -I tools/prepare_data.py data/speech_commands_v2 data/speech_commands_test_set_v2 data/packed

# train (CPU is enough: about 20 s per epoch)
python -I train.py --task commands12 --out runs/commands12
python -I train.py --task marvin --out runs/marvin --epochs 80 --qat_epochs 20

# deploy and check
python -I export_mcu.py runs/commands12/int8.pkl --check
python -I npu/npu_check.py runs/commands12/int8.pkl runs/commands12/npu --n 4890
python -I evaluate.py runs/commands12
```

Requirements: Python 3 with PyTorch 2.x, NumPy and ONNX. Building the C code needs clang or gcc. The NPU check needs
the Ryzen AI 1.8 Linux toolchain.

## License

The code is MIT. The dataset is Google Speech Commands v0.02, CC BY 4.0 (Warden 2018).
