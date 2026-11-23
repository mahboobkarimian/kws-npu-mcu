/* DS-CNN int8 inference: one plain C function per layer type, the network is a table (generated/kws_model_data.h).
 *
 * Production note: the three loops below are exactly what CMSIS-NN's arm_convolve_s8, arm_depthwise_conv_s8 and
 * arm_avgpool_s8 do (with SIMD). They are written out here so every operation is visible and the result is bit-exact
 * with the NPU. Swap them for CMSIS-NN for ~3-5x speed on Cortex-M4/M7 (check: CMSIS-NN rounds negative ties away
 * from zero, we round them up; only the non-ReLU classifier layer can see a negative tie).
 * SPDX-License-Identifier: MIT
 */
#include "dscnn_int8.h"

#include "generated/kws_model_data.h"

/* round-half-up right shift + saturation to int8 (">>" of a negative int32 is arithmetic on GCC/Clang/Arm compilers) */
static inline int8_t requantize(int32_t acc, int shift, int relu) {
  int32_t y = shift > 0 ? (acc + (1 << (shift - 1))) >> shift : acc << (-shift);
  const int32_t lo = relu ? 0 : -128;
  if (y > 127) y = 127;
  if (y < lo) y = lo;
  return (int8_t)y;
}

/* Full convolution (stem, pointwise 1x1, classifier). Zero padding = skip out-of-range taps. */
static void conv_s8(const kws_layer_t *L, const int8_t *in, int8_t *out) {
  for (int oy = 0; oy < L->out_h; ++oy)
    for (int ox = 0; ox < L->out_w; ++ox)
      for (int oc = 0; oc < L->out_c; ++oc) {
        int32_t acc = (int32_t)L->b[oc] * (1 << L->bias_shift);
        for (int ky = 0; ky < L->k_h; ++ky) {
          const int iy = oy * L->stride_h - L->pad_h + ky;
          if (iy < 0 || iy >= L->in_h) continue;
          for (int kx = 0; kx < L->k_w; ++kx) {
            const int ix = ox * L->stride_w - L->pad_w + kx;
            if (ix < 0 || ix >= L->in_w) continue;
            const int8_t *px = in + (iy * L->in_w + ix) * L->in_c;
            const int8_t *pw = L->w + ((oc * L->k_h + ky) * L->k_w + kx) * L->in_c;
            for (int ic = 0; ic < L->in_c; ++ic) acc += (int32_t)px[ic] * pw[ic];
          }
        }
        out[(oy * L->out_w + ox) * L->out_c + oc] = requantize(acc, L->out_shift, L->relu);
      }
}

/* Depthwise convolution: channel c only sees channel c (9 weights per channel for 3x3). */
static void dwconv_s8(const kws_layer_t *L, const int8_t *in, int8_t *out) {
  const int C = L->in_c;
  for (int oy = 0; oy < L->out_h; ++oy)
    for (int ox = 0; ox < L->out_w; ++ox)
      for (int c = 0; c < C; ++c) {
        int32_t acc = (int32_t)L->b[c] * (1 << L->bias_shift);
        for (int ky = 0; ky < L->k_h; ++ky) {
          const int iy = oy * L->stride_h - L->pad_h + ky;
          if (iy < 0 || iy >= L->in_h) continue;
          for (int kx = 0; kx < L->k_w; ++kx) {
            const int ix = ox * L->stride_w - L->pad_w + kx;
            if (ix < 0 || ix >= L->in_w) continue;
            acc += (int32_t)in[(iy * L->in_w + ix) * C + c] * L->w[(ky * L->k_w + kx) * C + c];
          }
        }
        out[(oy * L->out_w + ox) * C + c] = requantize(acc, L->out_shift, L->relu);
      }
}

/* Global average pool, the NPU's way: sum * 33 / 2^12 (~ / 125), then the scale change, round half up. */
static void avgpool_s8(const kws_layer_t *L, const int8_t *in, int8_t *out) {
  const int n = L->in_h * L->in_w, C = L->in_c;
  for (int c = 0; c < C; ++c) {
    int32_t sum = 0;
    for (int i = 0; i < n; ++i) sum += in[i * C + c];
    out[c] = requantize(sum * 33, L->out_shift, 1);
  }
}

void dscnn_run(const int8_t *input, int8_t *logits, int8_t *arena) {
  int8_t *buf[2] = {arena, arena + KWS_ARENA_BYTES / 2};
  const int8_t *x = input;
  for (int i = 0; i < KWS_NUM_LAYERS; ++i) {
    const kws_layer_t *L = &KWS_LAYERS[i];
    int8_t *y = (i == KWS_NUM_LAYERS - 1) ? logits : buf[i & 1];
    switch (L->op) {
      case KWS_OP_CONV: conv_s8(L, x, y); break;
      case KWS_OP_DWCONV: dwconv_s8(L, x, y); break;
      default: avgpool_s8(L, x, y); break;
    }
    x = y;
  }
}
