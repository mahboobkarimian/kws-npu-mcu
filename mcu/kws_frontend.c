/* MFCC front end (see kws_frontend.h). Tables come from export_mcu.py, so they equal the Python ones.
 * SPDX-License-Identifier: MIT
 */
#include "kws_frontend.h"

#if defined(__STDC_HOSTED__) && __STDC_HOSTED__ == 0
/* bare-metal object build without libc headers (Makefile cortex-m4/m7 size check); on a real board newlib provides these */
float logf(float); float floorf(float); double cos(double); double sin(double);
void *memmove(void *, const void *, __SIZE_TYPE__); void *memcpy(void *, const void *, __SIZE_TYPE__);
void *memset(void *, int, __SIZE_TYPE__);
#else
#include <math.h>
#include <string.h>
#endif

#include "generated/kws_frontend_tables.h"

#define NFFT 1024
static float tw_re[NFFT / 2], tw_im[NFFT / 2];
static uint16_t bitrev[NFFT];

void kws_frontend_init(void) {
  for (int k = 0; k < NFFT / 2; ++k) {
    double a = -2.0 * 3.14159265358979323846 * k / NFFT;
    tw_re[k] = (float)cos(a);
    tw_im[k] = (float)sin(a);
  }
  for (int i = 0, j = 0; i < NFFT; ++i) {
    bitrev[i] = (uint16_t)j;
    int m = NFFT >> 1;
    while (m >= 1 && (j & m)) { j ^= m; m >>= 1; }
    j |= m;
  }
}

/* in-place iterative radix-2 complex FFT */
static void fft1024(float *re, float *im) {
  for (int i = 0; i < NFFT; ++i) {
    int j = bitrev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= NFFT; len <<= 1) {
    const int half = len >> 1, step = NFFT / len;
    for (int s = 0; s < NFFT; s += len)
      for (int k = 0; k < half; ++k) {
        const float wr = tw_re[k * step], wi = tw_im[k * step];
        const int a = s + k, b = a + half;
        const float xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr; im[b] = im[a] - xi;
        re[a] += xr; im[a] += xi;
      }
  }
}

void kws_mfcc_frame(const int16_t *frame, int8_t *out10) {
  static float re[NFFT], im[NFFT];          /* 8 KB; static so it is not on the stack */
  float power[NFFT / 2 + 1], logmel[KWS_N_MELS];
  for (int i = 0; i < NFFT; ++i) {
    re[i] = i < KWS_FRAME_LEN ? (frame[i] / 32768.0f) * kws_hann[i] : 0.0f;
    im[i] = 0.0f;
  }
  fft1024(re, im);
  for (int k = 0; k <= NFFT / 2; ++k) power[k] = re[k] * re[k] + im[k] * im[k];
  for (int m = 0; m < KWS_N_MELS; ++m) {                 /* sparse triangular filters */
    float e = 0.0f;
    for (int j = 0; j < kws_mel_len[m]; ++j) e += power[kws_mel_start[m] + j] * kws_mel_w[kws_mel_off[m] + j];
    logmel[m] = logf(e + KWS_LOG_EPS);
  }
  for (int c = 0; c < KWS_N_MFCC; ++c) {
    float v = 0.0f;
    for (int m = 0; m < KWS_N_MELS; ++m) v += logmel[m] * kws_dct[m * KWS_N_MFCC + c];
    v = (v - kws_mean[c]) / kws_std[c];
    float q = floorf(v * (float)(1 << KWS_INPUT_FIX) + 0.5f);
    out10[c] = (int8_t)(q > 127.f ? 127 : q < -128.f ? -128 : q);
  }
}

void kws_mfcc_clip(const int16_t *clip, int8_t *out490) {
  for (int t = 0; t < KWS_N_FRAMES; ++t) kws_mfcc_frame(clip + t * KWS_HOP, out490 + t * KWS_N_MFCC);
}

void kws_stream_init(kws_stream_t *s) { memset(s, 0, sizeof(*s)); }

int kws_stream_push(kws_stream_t *s, const int16_t *hop320) {
  memmove(s->frame, s->frame + KWS_HOP, (KWS_FRAME_LEN - KWS_HOP) * sizeof(int16_t));
  memcpy(s->frame + KWS_FRAME_LEN - KWS_HOP, hop320, KWS_HOP * sizeof(int16_t));
  if (++s->hops < 2) return 0;                          /* the first full 640-sample frame needs 2 hops */
  memmove(s->feat, s->feat + KWS_N_MFCC, (KWS_N_FRAMES - 1) * KWS_N_MFCC);
  kws_mfcc_frame(s->frame, s->feat + (KWS_N_FRAMES - 1) * KWS_N_MFCC);
  return 1;
}
