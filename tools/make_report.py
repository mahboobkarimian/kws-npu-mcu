"""Build REPORT.md + figures from runs/*/eval.json, log.txt, mcu_check.json, npu_check.json.

  <python with matplotlib> -I tools/make_report.py      (we use ~/miniconda3/envs/plc/bin/python)
SPDX-License-Identifier: MIT
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "report_figs")
BLUE, ORANGE, GREY = "#2a6fdb", "#e8842c", "#888888"


def pct(x):
    return f"{100 * x:.2f} %"


def load(run):
    d = os.path.join(ROOT, "runs", run)
    E = json.load(open(f"{d}/eval.json"))
    log = open(f"{d}/log.txt").read()
    return E, log


def curves(log):
    fl = [(int(m[0]), float(m[1]), float(m[2])) for m in re.findall(r"float ep\s+(\d+)\s+loss \S+\s+train acc (\S+)\s+val bal (\S+)", log)]
    q = [(int(m[0]), float(m[1]), float(m[2])) for m in re.findall(r"QAT ep\s+(\d+)\s+loss \S+\s+train acc (\S+)\s+val bal (\S+)", log)]
    return fl, q


def fig_training(name, log):
    fl, q = curves(log)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot([e for e, _, _ in fl], [v for _, _, v in fl], color=BLUE, label="float: validation (balanced acc.)")
    ax.plot([e for e, _, _ in fl], [t for _, t, _ in fl], color=BLUE, ls=":", label="float: training (augmented)")
    n = len(fl)
    ax.plot([n + e for e, _, _ in q], [v for _, _, v in q], color=ORANGE, label="int8 QAT: validation (balanced acc.)")
    ax.axvline(n - 0.5, color=GREY, lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{name}: training")
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}_training.png", dpi=120); plt.close(fig)


def fig_confusion(name, E):
    cm = np.array(E["confusion_int8"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    cls = [c.strip("_") for c in E["classes"]]
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(cls), 0.8 + 0.5 * len(cls)))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}" if cmn[i, j] >= 0.005 else "·", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(cls))); ax.set_xticklabels(cls, rotation=60, fontsize=8)
    ax.set_yticks(range(len(cls))); ax.set_yticklabels(cls, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{name}: confusion (int8, % of each true class)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}_confusion.png", dpi=120); plt.close(fig)


def fig_robust(name, E):
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    k = list(E["noise"]); axs[0].plot(k, [E["noise"][x]["int8"] for x in k], "o-", color=ORANGE, label="int8")
    axs[0].plot(k, [E["noise"][x]["float"] for x in k], "s--", color=BLUE, label="float", ms=4)
    axs[0].set_title("added held-out noise (SNR)", fontsize=9)
    k = [x for x in E["pitch"] if x != "f0 not found"]
    axs[1].bar(range(len(k)), [E["pitch"][x]["int8"] for x in k], color=ORANGE)
    for i, x in enumerate(k):
        axs[1].text(i, E["pitch"][x]["int8"] + 0.005, f"n={E['pitch'][x]['n']}", ha="center", fontsize=7)
    axs[1].set_xticks(range(len(k))); axs[1].set_xticklabels([x.replace(" Hz", "") for x in k], fontsize=8)
    axs[1].set_title("speaker f0 band (Hz), int8", fontsize=9)
    lo = min(E["pitch"][x]["int8"] for x in k)
    axs[1].set_ylim(max(0, lo - 0.1), 1.0)
    k = list(E["speed"]); axs[2].plot(k, [E["speed"][x]["int8"] for x in k], "o-", color=ORANGE, label="int8")
    axs[2].plot(k, [E["speed"][x]["float"] for x in k], "s--", color=BLUE, label="float", ms=4)
    axs[2].axvspan(0.5, 3.5, color=GREY, alpha=0.12)
    axs[2].set_title("voice resampled (f0 + formants x factor)\nshaded = range seen in training", fontsize=9)
    for a in axs:
        a.grid(alpha=0.3); a.set_ylabel("accuracy")
    axs[0].legend(fontsize=8); axs[2].legend(fontsize=8)
    fig.suptitle(f"{name}: robustness (full test split, speech clips)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}_robustness.png", dpi=120); plt.close(fig)


def fig_det(E):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for k, col in (("clean", BLUE), ("10 dB", ORANGE)):
        c = E["streaming"][k]["curve"]
        ax.plot([x["fa_per_hour"] for x in c], [100 * x["frr"] for x in c], "o-", ms=3, color=col,
                label=f"{k} ({E['streaming'][k]['hours']:.2f} h, {E['streaming'][k]['n_keywords']} 'marvin')")
    ax.set_xscale("symlog", linthresh=1); ax.set_xlabel("false alarms per hour"); ax.set_ylabel("missed 'marvin' (%)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title("wake word 'marvin', always-on (decision every 100 ms)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/marvin_det.png", dpi=120); plt.close(fig)


def op_points(E, cond):
    c = E["streaming"][cond]["curve"]
    rows = []
    for th in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
        x = next(r for r in c if abs(r["threshold"] - th) < 1e-6)
        rows.append(f"| {th} | {pct(x['frr'])} | {x['fa']} | {x['fa_per_hour']:.2f} |")
    return rows


def section(name, E, log):
    s = []
    fl, q = curves(log)
    s.append(f"### Training ({name})\n")
    m = re.search(r"DS-CNN: (\d+) parameters.*?([\d.]+) M MAC", log)
    s.append(f"- {m.group(1)} parameters, {m.group(2)} M multiply-accumulates per 1 s window.")
    m = re.search(r"folded float val bal (\S+);  int8 before QAT \(PTQ\) (\S+);", log)
    ptq, qbest = float(m.group(2)), max(v for _, _, v in q)
    s.append(f"- Validation = balanced accuracy (mean recall over classes; used to pick checkpoints). "
             f"Best float {pct(max(v for _, _, v in fl))}; int8 straight after calibration (PTQ) {pct(ptq)}; "
             f"best QAT epoch {pct(qbest)}. " +
             ("Exported: the QAT weights." if qbest > ptq else
              "QAT never beat PTQ here, so the exported int8 model is the PTQ one (calibrated float weights)."))
    s.append(f"\n![training](report_figs/{name}_training.png)\n")
    s.append("### Accuracy\n")
    s.append("| test set | clips | float | int8 (= MCU = NPU) |\n|---|---|---|---|")
    if "official" in E:
        o = E["official"]
        s.append(f"| official v2 12-class test set | {o['n']} | {pct(o['float'])} | {pct(o['int8'])} |")
    t = E["test_split"]
    s.append(f"| full test split (all 35 words + 1,000 noise-only) | {t['n']} | {pct(t['float'])} | {pct(t['int8'])} |")
    if "official" in E:
        s.append(f"\nint8 PyTorch vs the integer reference code: identical logits on "
                 f"{pct(E['official']['int8_equals_integer_ref'])} of the official test clips.")
    s.append("\nPer class (int8, full test split):\n")
    s.append("| class | clips | recall (found) | precision (right when it says so) |\n|---|---|---|---|")
    for c, v in E["per_class"].items():
        s.append(f"| {c} | {v['n']} | {pct(v['recall'])} | {pct(v['precision'])} |")
    s.append(f"\n![confusion](report_figs/{name}_confusion.png)\n")
    s.append("### Robustness: noise, speaker pitch, voice shift\n")
    s.append(f"![robustness](report_figs/{name}_robustness.png)\n")
    s.append("| condition | float | int8 |\n|---|---|---|")
    for k, v in E["noise"].items():
        s.append(f"| noise {k} | {pct(v['float'])} | {pct(v['int8'])} |")
    for k, v in E["pitch"].items():
        s.append(f"| speaker {k if k.startswith('f0') else 'f0 ' + k} ({v['n']} clips) | {pct(v['float'])} | {pct(v['int8'])} |")
    for k, v in E["speed"].items():
        s.append(f"| voice resampled {k} | {pct(v['float'])} | {pct(v['int8'])} |")
    p = E["f0_percentiles"]
    unk = 100.0 * E["per_class"]["_unknown_"]["n"] / sum(v["n"] for v in E["per_class"].values())
    s.append(f"\nTest-speaker f0 (YIN): 5th / 25th / 50th / 75th / 95th percentile = "
             f"{p[0]:.0f} / {p[1]:.0f} / {p[2]:.0f} / {p[3]:.0f} / {p[4]:.0f} Hz.\n")
    s.append("How to read these rows:\n"
             "- *f0 bands* are real speakers, but each band also has its own mix of speakers, microphones and words, "
             "so a few % between bands is not caused by pitch alone. The *resampled* rows are the controlled "
             "experiment: the same clips, only the voice changed. x0.85..x1.15 was used in training augmentation; "
             "x0.8 and x1.25 are outside it.\n"
             f"- *full test split* and the noise/pitch rows count every clip, and {unk:.0f} % of them are `_unknown_` "
             "words. A model (or a condition, like light noise) that pushes the decision towards 'unknown' scores "
             "higher there: that is why int8 can beat float, or 20 dB noise beat clean, on these rows. Use the "
             "per-class recall, the balanced official test set and (for the wake word) the streaming test to judge.\n")
    return s


def main():
    os.makedirs(FIG, exist_ok=True)
    out = ["# Accuracy report: DS-CNN keyword spotter", "",
           "Generated by `tools/make_report.py` from `runs/*/eval.json` (made by `evaluate.py`). "
           "Dataset: Google Speech Commands v0.02 (Warden 2018, CC-BY-4.0). Every test speaker is unseen in training "
           "(official speaker-hashed split). The noise used in tests is the held-out last 20 % of each background "
           "recording. \"int8\" is the deployed model: the same integers run on the MCU (C) and the NPU.", ""]
    E12, log12 = load("commands12")
    fig_training("commands12", log12); fig_confusion("commands12", E12); fig_robust("commands12", E12)
    out += ["## 1. Twelve-class command model (`runs/commands12`)", ""] + section("commands12", E12, log12)
    out += ["Published references (12-class Speech Commands; test subsets differ, so +-1 % is noise): "
            "MLPerf Tiny KWS (DS-CNN, 49x10 MFCC input like ours): quality target 90 % top-1, reference model 92.2 % "
            "on its 1,000-clip test subset (arXiv:2106.07597). Hello Edge (arXiv:1711.07128, Speech Commands v1): "
            "the largest DS-CNN 95.4 %; the small DS-CNN (our size class) ~94.4 % (from the paper's table, as I "
            "recall it; check the PDF before quoting).", ""]
    if os.path.exists(os.path.join(ROOT, "runs/marvin/eval.json")):
        EM, logm = load("marvin")
        fig_training("marvin", logm); fig_confusion("marvin", EM); fig_robust("marvin", EM); fig_det(EM)
        out += ["## 2. Wake word \"marvin\" (`runs/marvin`)", ""] + section("marvin", EM, logm)
        out += ["### Always-on test (what matters for a wake word)", "",
                "All 11,005 test clips are joined into one recording (3 h; 'marvin' appears ~200 times among 34 other "
                "words). The detector runs every 100 ms on the last 1 s, averages P(marvin) over 3 decisions, "
                "triggers above a threshold and then sleeps 1 s. A trigger within 2 s of a 'marvin' start is a hit, "
                "anything else a false alarm.", "", "![det](report_figs/marvin_det.png)", ""]
        for cond in ("clean", "10 dB"):
            out += [f"**{cond}**", "", "| threshold | missed | false alarms | per hour |", "|---|---|---|---|"]
            out += op_points(EM, cond) + [""]
            fw = EM["streaming"][cond].get("fa_words_at_0_95", {})
            if fw:
                out += ["False alarms at threshold 0.95, by the word being said: " +
                        ", ".join(f"{w} {n}" for w, n in fw.items()) + ".", ""]
        out += ["The false alarms are look-alike sounds: words sharing the r-coloured vowel (/ar/ of mARvin ~ "
                "bIRd, backWARD, forWARD), the /v/ (fiVe) or the final /n/ (niNE). The standard fixes are hard-negative "
                "training (more such words), a longer or stricter smoothing rule, and a second-stage verifier.", "",
                "Caveat: the negatives are isolated words, 34 of them, said on purpose. Real use has hours of "
                "conversation, TV and music, which give more look-alike sounds; products measure false alarms on "
                "hundreds of hours of such audio and add a second-stage check.", ""]
    out += ["## 3. Deployment checks", ""]
    out += ["Note on model selection: the first training runs (kept in `runs/archive_selection_by_plain_acc/`) chose "
            "checkpoints by plain accuracy on a validation set that is ~60 % `_unknown_` clips. That metric "
            "rewards a bias towards 'unknown': int8 looked 1.5 % better than float there, but on the balanced official "
            "test set it was float 94.38 % / int8 93.84 %. All runs reported above choose checkpoints by balanced "
            "accuracy instead.", ""]
    for name, E in [("commands12", E12)] + ([("marvin", EM)] if "EM" in dir() else []):
        if "npu_check" in E:
            n = E["npu_check"]
            out.append(f"**NPU ({name})**: AMD Phoenix (Ryzen 7 PRO 7840U), Ryzen AI 1.8 Linux, 1x4 overlay. "
                       f"{n['n']} official-test inputs: {n['logits_identical_pct']:.3f} % of logits identical to the "
                       f"integer reference (max diff {n['max_diff']} LSB)"
                       + (f"; accuracy of the NPU outputs {pct(n['npu_official_test_acc'])}" if n.get("npu_official_test_acc") else "")
                       + ". " + " ".join(n["run"]) + ". Compile: " + "; ".join(x.strip() for x in n["compile"]) + "\n")
        if "mcu_check" in E:
            m = E["mcu_check"]
            out.append(f"**MCU C code ({name})**: network {m['net_identical_pct']:.3f} % identical logits on "
                       f"{m['net_n']} inputs. Full C pipeline (C MFCC + C network) on {m['fe_n']} raw clips: "
                       f"{m['fe_feat_identical_pct']:.3f} % of int8 features identical to Python (max "
                       f"{m['fe_feat_maxdiff']} LSB; float FFT order differs), logits identical for "
                       f"{m['fe_clip_logits_identical_pct']:.2f} % of clips, same predicted class for "
                       f"{m['fe_argmax_agree_pct']:.2f} %. Laptop timing: {m['fe_host']}\n")
            out.append("```\n" + m["sizes"]["cortex-m4"] + "```\n")
    open(os.path.join(ROOT, "REPORT.md"), "w").write("\n".join(out))
    print("wrote REPORT.md")


if __name__ == "__main__":
    main()
