/* Audio front end for the MCU: 16 kHz int16 audio -> int8 MFCC frames for the DS-CNN. Single precision float
 * (Cortex-M4F/M7/M33/M55 have an FPU). Same maths and constants as kws/features.py (tables are generated).
 *
 *   every 20 ms (320 new samples):  640-sample frame -> Hann -> FFT 1024 -> power -> 40 mel -> log -> DCT (10)
 *                                   -> (x - mean) / std -> int8 (x 2^KWS_INPUT_FIX, round half up)
 *
 * The FFT here is a portable radix-2 one. On a real STM32 use CMSIS-DSP arm_rfft_fast_f32 (1024), ~4x faster.
 * SPDX-License-Identifier: MIT
 */
#ifndef KWS_FRONTEND_H
#define KWS_FRONTEND_H
#include <stdint.h>

#define KWS_FRAME_LEN 640
#define KWS_HOP 320
#define KWS_N_FRAMES 49
#define KWS_N_MFCC 10

void kws_frontend_init(void);

/* one 40 ms frame (640 int16 samples) -> 10 int8 features */
void kws_mfcc_frame(const int16_t *frame, int8_t *out10);

/* a whole 1 s clip (16000 samples) -> [49][10] int8 network input */
void kws_mfcc_clip(const int16_t *clip, int8_t *out490);

/* ---- streaming: push 20 ms at a time; the last 49 feature frames (1 s) are always ready in `feat` */
typedef struct {
  int16_t frame[KWS_FRAME_LEN];             /* last 640 samples */
  int8_t feat[KWS_N_FRAMES * KWS_N_MFCC];   /* last 49 MFCC frames, oldest first */
  int hops;                                 /* hops pushed so far */
} kws_stream_t;

void kws_stream_init(kws_stream_t *s);
/* returns 1 when a new feature frame was added (from the 2nd hop on) */
int kws_stream_push(kws_stream_t *s, const int16_t *hop320);

#endif
