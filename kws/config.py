"""Every number that defines the audio front end and the tasks lives here, in one place.

The C front end (mcu/mfcc.c) and the NPU host use the same numbers; export_mcu.py writes them into a header so the
two can never drift apart.

Where the numbers come from:
  - 16 kHz, 1 s windows, 40 ms frames, 20 ms hop, 40 mel bands 20 Hz..4 kHz, 10 MFCCs:
    Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers" (ARM, 2017), also the MLPerf Tiny KWS benchmark.
    We keep them so our accuracy can be compared with those papers.
  - 12 classes = 10 command words + "silence" + "unknown": Warden, "Speech Commands" (2018), the standard KWS task.
SPDX-License-Identifier: MIT
"""

# ---------------------------------------------------------------- audio front end
SAMPLE_RATE = 16000           # Hz. Speech Commands is recorded/stored at 16 kHz.
CLIP_SAMPLES = 16000          # 1.0 s analysis window: long enough for one word or a short phrase ("hey siri" ~0.7 s).
FRAME_LEN = 640               # 40 ms analysis frame: holds 4..16 pitch periods (f0 100..400 Hz).
HOP = 320                     # 20 ms hop: 50 % overlap.
N_FFT = 1024                  # next power of two >= FRAME_LEN (zero-padded; CMSIS-DSP arm_rfft_fast_f32 size).
N_FRAMES = 1 + (CLIP_SAMPLES - FRAME_LEN) // HOP   # = 49 frames per 1 s window
N_MELS = 40                   # triangular mel bands
F_MIN, F_MAX = 20.0, 4000.0   # Hz. Hello Edge range; the vowel formants F1/F2 and most consonant cues are below 4 kHz.
N_MFCC = 10                   # keep only the 10 lowest cepstral coefficients = the smooth spectral ENVELOPE (formants);
                              # most of the fine pitch-harmonic ripple lands in higher coefficients and is dropped.
                              # NOT all pitch is removed (measured with synthetic vowels, see features.py).
LOG_EPS = 1e-6                # floor inside log(mel energy); digital silence -> log(1e-6) = -13.8

# ---------------------------------------------------------------- tasks (class lists)
SILENCE = "_silence_"
UNKNOWN = "_unknown_"
TASKS = {
    # the standard benchmark (Warden 2018 / Hello Edge / MLPerf Tiny)
    "commands12": [SILENCE, UNKNOWN, "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    # a wake-word detector: "marvin" is a name, like "Siri" or "Alexa". Everything else is "unknown".
    "marvin": [SILENCE, UNKNOWN, "marvin"],
}

ALL_WORDS = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four", "go",
             "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven",
             "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]   # the 35 v2 words
