// Copyright (C) Mihai Preda

// #defines that allow fft_height and fft_width share common code in fftbase.cl
#define VARIANT       FFT_VARIANT_W
#define LDSPAD        LDSPAD_W
#define LDSSWIZ       LDSSWIZ_W
#define SHUFL_BYTES   SHUFL_BYTES_W
#define WGSZ          G_W                           // Change this to WG!!!
#define RADIX         NW

#include "math.cl"
#include "trig.cl"
#include "fftbase.cl"

#if WIDTH != 256 && WIDTH != 512 && WIDTH != 1024 && WIDTH != 4096 && WIDTH != 625
#error WIDTH must be one of: 256, 512, 1024, 4096, 625
#endif

#if FFT_FP64

void OVERLOAD fft_NW(T2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 5
  fft5(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

#if FFT_VARIANT_W == 0

#if WIDTH > 1024
#error FFT_VARIANT_W == 0 only supports WIDTH <= 1024
#endif
#if !AMDGPU
#error FFT_VARIANT_W == 0 only supported by AMD GPUs
#endif

void OVERLOAD fft_WIDTH(local T2 *lds, T2 *u, Trig trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = WIDTH / NW;

#if NW == 8
  T2 w = fancyTrig_N(ND / WIDTH * lowMe);
#else
  T2 w = slowTrig_N(ND / WIDTH * lowMe, ND / NW);
#endif

  for (u32 s = 1; s < WG; s *= NW) {
    fft_NW(u);
    w = bcast(w, s);

    chainMul(NW, u, w, 0);

    shufl(WG, lds,  u, NW, s, numWG, sb, lowMe);
  }
  fft_NW(u);
}

#else

void OVERLOAD fft_WIDTH(local T2 *lds, T2 *u, Trig trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = WIDTH / NW;

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NW) {
    fft_NW(u);
    tabMul(WG, trig, u, NW, s, lowMe);
    shufl(WG, lds,  u, NW, s, numWG, sb, lowMe);
  }
  fft_NW(u);
}

#endif


// New fft_WIDTH that uses more FMA instructions than the old fft_WIDTH.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void OVERLOAD new_fft_WIDTH(local T2 *lds, T2 *u, Trig trig, u32 numWG, const u32 sb, u32 lowMe, int callnum) {
  u32 WG = WIDTH / NW;

  // This line mimics shufl -- partition lds
  local T2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

// Custom code for various WIDTH values

#if WIDTH == 256 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=256, NW=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NW, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NW, 16, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);

#elif WIDTH == 512 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=512, NW=8

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NW, 8, numWG, sb, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#elif WIDTH == 1024 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=1024, NW=4

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NW, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NW, 16, numWG, sb, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NW, 64, numWG, sb, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 64, numWG, lowMe, 1);

#elif WIDTH == 4096 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=4K, NW=8

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NW, 8, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NW, 64, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old version
  fft_WIDTH(lds, u, trig, numWG, sb, lowMe);

#endif
}

// There are two version of new_fft_WIDTH in case we want to try saving some trig values from new_fft_WIDTH1 in LDS memory for later use in new_fft_WIDTH2.
void OVERLOAD new_fft_WIDTH1(local T2 *lds, T2 *u, Trig trig, u32 numWG, const u32 sb, u32 lowMe) { new_fft_WIDTH(lds, u, trig, numWG, sb, lowMe, 1); }
void OVERLOAD new_fft_WIDTH2(local T2 *lds, T2 *u, Trig trig, u32 numWG, const u32 sb, u32 lowMe) { new_fft_WIDTH(lds, u, trig, numWG, sb, lowMe, 2); }

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft_NW(F2 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

void OVERLOAD fft_WIDTH(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = WIDTH / NW;

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NW) {
    fft_NW(u);
    tabMul(WG, trig, u, NW, s, lowMe);
    shufl(WG, lds,  u, NW, s, numWG, sb, lowMe);
  }
  fft_NW(u);
}

// New fft_WIDTH that uses more FMA instructions than the old fft_WIDTH.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void OVERLOAD new_fft_WIDTH(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe, int callnum) {
  u32 WG = WIDTH / NW;

  // This line mimics shufl -- partition lds
  local F2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

// Custom code for various WIDTH values

#if ENABLE_FP32_VARIANT_2 && WIDTH == 256 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=256, NW=4

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NW, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NW, 16, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);

#elif ENABLE_FP32_VARIANT_2 && WIDTH == 512 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=512, NW=8

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NW, 8, numWG, sb, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#elif ENABLE_FP32_VARIANT_2 && WIDTH == 1024 && NW == 4 && FFT_VARIANT_W == 2

// Custom code for WIDTH=1024, NW=4

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(WG, lds, u, NW, 4, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(WG, lds, u, NW, 16, numWG, sb, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(WG, trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NW, 64, numWG, sb, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(WG, trig, preloads, u, 64, numWG, lowMe, 1);

#elif ENABLE_FP32_VARIANT_2 && WIDTH == 4096 && NW == 8 && FFT_VARIANT_W == 2

// Custom code for WIDTH=4K, NW=8

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(WG, trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(WG, lds, u, NW, 1, numWG, sb, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(WG, lds, u, NW, 8, numWG, sb, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(WG, trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(WG, partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(WG, lds, u, NW, 64, numWG, sb, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(WG, trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old version
  fft_WIDTH(lds, u, trig, numWG, sb, lowMe);

#endif
}

// There are two version of new_fft_WIDTH in case we want to try saving some trig values from new_fft_WIDTH1 in LDS memory for later use in new_fft_WIDTH2.
void OVERLOAD new_fft_WIDTH1(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe) { new_fft_WIDTH(lds, u, trig, numWG, sb, lowMe, 1); }
void OVERLOAD new_fft_WIDTH2(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, const u32 sb, u32 lowMe) { new_fft_WIDTH(lds, u, trig, numWG, sb, lowMe, 2); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft_NW(GF31 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

void OVERLOAD fft_WIDTH(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = WIDTH / NW;

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NW) {
    fft_NW(u);
    tabMul(WG, trig, u, NW, s, lowMe);
    shufl(WG, lds,  u, NW, s, numWG, sb, lowMe);
  }
  fft_NW(u);
}

void OVERLOAD new_fft_WIDTH1(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe) { fft_WIDTH(lds, u, trig, numWG, sb, lowMe); }
void OVERLOAD new_fft_WIDTH2(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, const u32 sb, u32 lowMe) { fft_WIDTH(lds, u, trig, numWG, sb, lowMe); }

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft_NW(GF61 *u) {
#if NW == 4
  fft4(u);
#elif NW == 8
  fft8(u);
#else
#error NW
#endif
}

void OVERLOAD fft_WIDTH(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe) {
  u32 WG = WIDTH / NW;

#if !UNROLL_W
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= NW) {
    fft_NW(u);
    tabMul(WG, trig, u, NW, s, lowMe);
    shufl(WG, lds,  u, NW, s, numWG, sb, lowMe);
  }
  fft_NW(u);
}

void OVERLOAD new_fft_WIDTH1(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe) { fft_WIDTH(lds, u, trig, numWG, sb, lowMe); }
void OVERLOAD new_fft_WIDTH2(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, const u32 sb, u32 lowMe) { fft_WIDTH(lds, u, trig, numWG, sb, lowMe); }

#endif
