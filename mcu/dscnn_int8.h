/* DS-CNN keyword spotter, int8 inference for Cortex-M (any MCU with a C99 compiler; no FPU needed for this part).
 *
 * Bit-exact with the PyTorch QAT model and with the AMD Phoenix NPU (same integers, same rounding):
 *   acc = sum(x_q * w_q) + (b_q << bias_shift)              int32 accumulator
 *   y_q = clamp((acc + 2^(shift-1)) >> shift, lo, 127)      round half up; lo = 0 when ReLU is fused
 *   global average pool:  y_q = round_half_up(sum * 33 >> pool_shift)   (the NPU's 1/125 ~ 33/4096)
 *
 * Memory: weights+biases ~22 KB in flash (const), activations 2 x 8,000 bytes of RAM (ping-pong arena).
 * Activation layout: HWC (channel last, like CMSIS-NN), H = time frame, W = cepstral coefficient.
 * SPDX-License-Identifier: MIT
 */
#ifndef DSCNN_INT8_H
#define DSCNN_INT8_H
#include <stdint.h>

enum { KWS_OP_CONV = 0, KWS_OP_DWCONV = 1, KWS_OP_AVGPOOL = 2 };

typedef struct {
  const char *name;
  int op;                          /* KWS_OP_* */
  int in_h, in_w, in_c;            /* input map */
  int out_h, out_w, out_c;         /* output map */
  int k_h, k_w, stride_h, stride_w, pad_h, pad_w;
  const int8_t *w;                 /* conv: [out_c][k_h][k_w][in_c];  depthwise: [k_h][k_w][c] */
  const int8_t *b;                 /* [out_c], int8 (Phoenix DPU limit), shifted left by bias_shift */
  int bias_shift;                  /* f_in + f_w - f_b */
  int out_shift;                   /* conv: f_in + f_w - f_out;  avgpool: 12 - (f_out - f_in) */
  int relu;                        /* 1: clamp at 0 (fused ReLU) */
} kws_layer_t;

#define KWS_ARENA_BYTES (2 * 25 * 5 * 64)   /* two largest activation maps */

/* input:  int8 [49][10] normalised, quantised MFCC (kws_frontend.h makes it)
 * logits: int8 [KWS_NUM_CLASSES]; multiply by 2^-KWS_LOGIT_FIX for real-valued logits
 * arena:  KWS_ARENA_BYTES of scratch RAM (word aligned) */
void dscnn_run(const int8_t *input, int8_t *logits, int8_t *arena);

#endif
