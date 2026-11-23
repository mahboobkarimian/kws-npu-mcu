/* Host test harness for the MCU code (builds with gcc/clang on the laptop; the same .c files build for Cortex-M).
 *
 *   kws_test net  <inputs.s8>  <n> <logits.s8>    network only: int8 [n][49][10] -> int8 [n][classes]
 *   kws_test clip <clips.s16>  <n> <feats.s8> <logits.s8>   full pipeline: int16 audio [n][16000] -> features + logits
 *
 * Python (export_mcu.py --check) writes the inputs and compares the outputs with the PyTorch / NPU results.
 * SPDX-License-Identifier: MIT
 */
#define _POSIX_C_SOURCE 199309L   /* clock_gettime with -std=c99 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dscnn_int8.h"
#include "generated/kws_model_data.h"
#include "kws_frontend.h"

static void *load(const char *path, size_t bytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  void *p = malloc(bytes);
  if (fread(p, 1, bytes, f) != bytes) { fprintf(stderr, "%s: short read\n", path); exit(1); }
  fclose(f);
  return p;
}

static void save(const char *path, const void *p, size_t bytes) {
  FILE *f = fopen(path, "wb");
  if (!f || fwrite(p, 1, bytes, f) != bytes) { perror(path); exit(1); }
  fclose(f);
}

static double now_s(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  static int8_t arena[KWS_ARENA_BYTES];
  if (argc < 5) { fprintf(stderr, "usage: see kws_test.c\n"); return 2; }
  const int n = atoi(argv[3]);
  const int in_sz = KWS_N_FRAMES * KWS_N_MFCC;
  int8_t *logits = malloc((size_t)n * KWS_NUM_CLASSES);
  int8_t *feats;
  double t_fe = 0, t_net = 0, t0;
  if (strcmp(argv[1], "net") == 0) {
    feats = load(argv[2], (size_t)n * in_sz);
  } else {
    int16_t *clips = load(argv[2], (size_t)n * 16000 * 2);
    feats = malloc((size_t)n * in_sz);
    kws_frontend_init();
    t0 = now_s();
    for (int i = 0; i < n; ++i) kws_mfcc_clip(clips + (size_t)i * 16000, feats + (size_t)i * in_sz);
    t_fe = now_s() - t0;
    save(argv[4], feats, (size_t)n * in_sz);
  }
  t0 = now_s();
  for (int i = 0; i < n; ++i) dscnn_run(feats + (size_t)i * in_sz, logits + (size_t)i * KWS_NUM_CLASSES, arena);
  t_net = now_s() - t0;
  save(argv[argc - 1], logits, (size_t)n * KWS_NUM_CLASSES);
  printf("RESULT %d inputs; host time per inference: network %.1f us", n, 1e6 * t_net / n);
  if (t_fe > 0) printf(", front end %.1f us per 1 s clip (49 frames)", 1e6 * t_fe / n);
  printf("\n");
  return 0;
}
