// Copyright (C) Mihai Preda

// #defines that allow fft_height and fft_width share common code in fftbase.cl
#define VARIANT       FFT_VARIANT_H
#define LDSPAD        LDSPAD_H
#define LDSSWIZ       LDSSWIZ_H
#define SHUFL_BYTES   SHUFL_BYTES_H
#define WGSZ          G_H                           // Change this to WG!!!
#define RADIX         NH

#include "math.cl"
#include "trig.cl"
#include "fftbase.cl"

#if SMALL_HEIGHT != 256 && SMALL_HEIGHT != 512 && SMALL_HEIGHT != 1024 && SMALL_HEIGHT != 4096
#error SMALL_HEIGHT must be one of: 256, 512, 1024, 4096
#endif

#if !INPLACE
u32 transPos(u32 k, u32 middle, u32 width) { return k / width + k % width * middle; }
#else
u32 transPos(u32 k, u32 middle, u32 width) { return k; }
#endif

#if FFT_FP64

void OVERLOAD fft_NH(T2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

#if FFT_VARIANT_H == 0

#if HEIGHT > 1024
#error FFT_VARIANT_H == 0 only supports HEIGHT <= 1024
#endif
#if !AMDGPU
#error FFT_VARIANT_H == 0 only supported by AMD GPUs
#endif

void OVERLOAD fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = SMALL_HEIGHT / NH;
  for (u32 s = 1; s < WG; s *= NH) {
    fft_NH(u);
    w = bcast(w, s);

    chainMul(NH, u, w, 1);

    shufl(WG, lds,  u, NH, s, numWG, sb, lowMe);
  }
  fft_NH(u);
}

#else

void OVERLOAD fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = SMALL_HEIGHT / NH;

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NH) {
    fft_NH(u);
    tabMul(WG, trig, u, NH, s, lowMe);
    shufl(WG, lds,  u, NH, s, numWG, sb, lowMe);
  }
  fft_NH(u);
}

#endif

void OVERLOAD new_fft_HEIGHT(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, const u32 sb, u32 lowMe, int callnum) {
  u32 WG = SMALL_HEIGHT / NH;

  // This line mimics shufl -- partition lds
  local T2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

// Custom code for various SMALL_HEIGHT values

#if SMALL_HEIGHT == 256 && NH == 4 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=256, NH=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NH, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NH, 16, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);

#elif SMALL_HEIGHT == 512 && NH == 8 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=512, NH=8

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + 2*WG*8;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NH, 8, numWG, sb, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 1);

#elif SMALL_HEIGHT == 1024 && NH == 4 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=1024, NH=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NH, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NH, 16, numWG, sb, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NH, 64, numWG, sb, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 64, numWG, lowMe, 1);

#else

  // Old version
  fft_HEIGHT(lds, u, trig, w, numWG, sb, lowMe);

#endif
}

void new_fft_HEIGHT1(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, const u32 sb, u32 lowMe)  { new_fft_HEIGHT(lds, u, trig, w, numWG, sb, lowMe, 1); }
void new_fft_HEIGHT2(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, const u32 sb, u32 lowMe)  { new_fft_HEIGHT(lds, u, trig, w, numWG, sb, lowMe, 2); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft_NH(F2 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

void OVERLOAD fft_HEIGHT(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = SMALL_HEIGHT / NH;

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NH) {
    fft_NH(u);
    tabMul(WG, trig, u, NH, s, lowMe);
    shufl(WG, lds, u, NH, s, numWG, sb, lowMe);
  }
  fft_NH(u);
}

void OVERLOAD new_fft_HEIGHT(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe, int callnum) {
  u32 WG = SMALL_HEIGHT / NH;

  // This line mimics shufl -- partition lds
  local F2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

// Custom code for various SMALL_HEIGHT values

#if ENABLE_FP32_VARIANT_2 && SMALL_HEIGHT == 256 && NH == 4 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=256, NH=4

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NH, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NH, 16, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);

#elif ENABLE_FP32_VARIANT_2 && SMALL_HEIGHT == 512 && NH == 8 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=512, NH=8

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + 2*WG*8;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NH, 8, numWG, sb, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 1);

#elif ENABLE_FP32_VARIANT_2 && SMALL_HEIGHT == 1024 && NH == 4 && FFT_VARIANT_H == 2

// Custom code for SMALL_HEIGHT=1024, NH=4

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NH, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NH, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NH, 16, numWG, sb, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NH, 64, numWG, sb, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 64, numWG, lowMe, 1);

#else

  // Old version
  fft_HEIGHT(lds, u, trig, numWG, sb, lowMe);

#endif
}

void new_fft_HEIGHT1(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe)  { new_fft_HEIGHT(lds, u, trig, numWG, sb, lowMe, 1); }
void new_fft_HEIGHT2(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe)  { new_fft_HEIGHT(lds, u, trig, numWG, sb, lowMe, 2); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft_NH(GF31 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

void OVERLOAD fft_HEIGHT(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = SMALL_HEIGHT / NH;

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NH) {
    fft_NH(u);
    tabMul(WG, trig, u, NH, s, lowMe);
    shufl(WG, lds, u, NH, s, numWG, sb, lowMe);
  }
  fft_NH(u);
}

void OVERLOAD new_fft_HEIGHT1(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe)  { fft_HEIGHT(lds, u, trig, numWG, sb, lowMe); }
void OVERLOAD new_fft_HEIGHT2(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe)  { fft_HEIGHT(lds, u, trig, numWG, sb, lowMe); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft_NH(GF61 *u) {
#if NH == 4
  fft4(u);
#elif NH == 8
  fft8(u);
#else
#error NH
#endif
}

void OVERLOAD fft_HEIGHT(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = SMALL_HEIGHT / NH;

#if !UNROLL_H
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NH) {
    fft_NH(u);
    tabMul(WG, trig, u, NH, s, lowMe);
    shufl(WG, lds, u, NH, s, numWG, sb, lowMe);
  }
  fft_NH(u);
}

void OVERLOAD new_fft_HEIGHT1(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe)  { fft_HEIGHT(lds, u, trig, numWG, sb, lowMe); }
void OVERLOAD new_fft_HEIGHT2(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe)  { fft_HEIGHT(lds, u, trig, numWG, sb, lowMe); }

#endif
