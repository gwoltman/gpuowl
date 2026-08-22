// Copyright (C) Mihai Preda

#include "fft4.cl"
#include "fft8.cl"

// Calculate the LDS bytes used by shufl
#if LDSPAD && SHUFL_BYTES == 16 && RADIX == 8
#define LDS_BYTES     ((WG * RADIX + 7) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 16 && RADIX == 4
#define LDS_BYTES     ((WG * RADIX + 12) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 8 && RADIX == 8
#define LDS_BYTES     ((WG * RADIX + 56) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 8 && RADIX == 4
#define LDS_BYTES     ((WG * RADIX + 12) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 4 && RADIX == 8
#define LDS_BYTES     ((WG * RADIX + 56) * SHUFL_BYTES)
#elif LDSPAD && SHUFL_BYTES == 4 && RADIX == 4
#define LDS_BYTES     ((WG * RADIX + 12) * SHUFL_BYTES)
#else
#define LDS_BYTES     (WG * RADIX * SHUFL_BYTES)
#endif

#define INCLUDE_FILE "shufl.cl"
#include "expand.cl"

#if FFT_FP64

void OVERLOAD chainMul4(T2 *u, T2 w) {
  u[1] = cmul(u[1], w);

  T2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  double a = mul2(base.y);
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

#if 1
// This version of chainMul8 tries to minimize roundoff error even if more F64 ops are used.
// Trial and error looking at Z values on a WIDTH=512 FFT was used to determine when to switch from fancy to non-fancy powers of w.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2;
  // Rocm optimizer behaves weirdly. Using multiple mul2s instead of one mul2 in csqTrigFancy makes double-wide single-kernel variant 0 tailSquare inexplicably slower
  if (DOING_WIDTH || VARIANT != 0) {
    w2 = csqTrigFancy(w);
  } else {
    w2 = U2(mulminus2(w.y) * w.y, mul2(fma(w.x, w.y, w.y)));
  }
  u[2] = cmulFancy(u[2], w2);

  T2 w3;
  // Rocm optimizer behaves weirdly yet again. Using mul2 instead of 2.0* makes double-wide single-kernel tailSquare inexplicably slower
  // even though it is one fewer F64 op.
  if (DOING_WIDTH || VARIANT != 0) {
    w3 = ccubeTrigFancy(w2, w);
  } else {
    double a = 2*w2.y;
    w3 = U2(fma(a, -w.y, w.x), fma(a, w.x, a - w.y));
  }
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  T2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

#else
// This version of chainMul8 minimizes F64 ops even if that increases roundoff error.
// This version is faster on a Radeon 7 with worse roundoff.  However, FFT_width is even faster with better roundoff.
// This version is the same speed on a TitanV probably due to its great F64 throughput.
// This version is slower on R7Pro due to a rocm optimizer issue in double-wide single-kernel tailSquare using BCAST.  I could not find a work-around.
// Other GPUs???  This version might be useful.  If we decide to make this available, it will need a new width and height fft spec number.
// Consequently, an increase in the BPW table and increase work for -ztune and -tune.
void OVERLOAD chainMul8(T2 *u, T2 w) {
  u[1] = cmulFancy(u[1], w);

  T2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  T2 w3 = ccubeTrigDefancy(w2, w);
  u[3] = cmul(u[3], w3);

  T2 w4 = csqTrigDefancy(w2);
  u[4] = cmul(u[4], w4);

  T2 w6 = csqTrig(w3);
  T2 w5, w7; cmul_a_by_fancyb_and_conjfancyb(&w7, &w5, w6, w);
  u[5] = cmul(u[5], w5);
  u[6] = cmul(u[6], w6);
  u[7] = cmul(u[7], w7);
}
#endif

void OVERLOAD chainMul(T2 *u, T2 w) {
  // Do a length 4 chain mul, w must not be in Fancy format
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul, w must be in Fancy format
  if (RADIX == 8) chainMul8(u, w);
}


#if AMDGPU && VARIANT == 0

int bcast4(int x)  { return __builtin_amdgcn_mov_dpp(x, 0, 0xf, 0xf, false); }
int bcast8(int x)  { return __builtin_amdgcn_ds_swizzle(x, 0x0018); }
int bcast16(int x) { return __builtin_amdgcn_ds_swizzle(x, 0x0010); }
int bcast64(int x) { return __builtin_amdgcn_readfirstlane(x); }

int bcastAux(int x, u32 span) {
  return span == 4 ? bcast4(x) : span == 8 ? bcast8(x) : span == 16 ? bcast16(x) : span == 64 ? bcast64(x) : x;
}

T2 bcast(T2 src, u32 span) {
  int4 s = as_int4(src);
  for (int i = 0; i < 4; ++i) { s[i] = bcastAux(s[i], span); }
  return as_double2(s);
}

#endif

void OVERLOAD fft_RADIX(T2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 5
  fft5(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD tabMul(Trig trig, T2 *u, u32 f, u32 me) {
#if 0
  u32 p = me / f * f;
#else
  u32 p = me & ~(f - 1);
#endif

// Compute trigs from scratch every time.  This can't possibly be a good idea on any GPUs.
#if 0
  T2 w = slowTrig_N(ND / RADIX / WG * p, ND / RADIX);
  T2 base = w;
  for (int i = 1; i < RADIX; ++i) {
    u[i] = cmul(u[i], w);
    w = cmul(w, base);
  }
  return;
#endif

// This code uses chained complex multiplies which could be faster on GPUs with great DP throughput or poor memory bandwidth or caching.
// This ought to be the least accurate version of Tabmul.  In practice, this is just as accurate as reading precomputed values from memory.
// Apparently, chained Fancy muls at these short n=4 and n=8 lengths are very accurate.

  if (TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);
    chainMul(u, w);
    return;
  }

// Theoretically, maximum accuracy.  Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.
// Radeon VII loves this case, it is faster than the chainmul case.  nVidia Titan V hates this case.

  if (!TABMUL_CHAIN) {
    T2 w = TFLOAD(&trig[p]);

    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], w);
    } else {
      u[1] = cmul(u[1], w);
    }

    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Partial complex-multiply that delays the mul-by-cosine so it can be part of an FMA.
// We're trying to calculate u * U2(cosine,sine).
// real = (u.x - u.y*sine_over_cosine) * cosine
// imag = (u.x*sine_over_cosine + u.y) * cosine
T2 partial_cmul(T2 u, T sine_over_cosine) {
  return U2(fma(-u.y, sine_over_cosine, u.x), fma(u.x, sine_over_cosine, u.y));
}

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { T2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  Trig trig2 = (Trig) trig1;
  T2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(Trig trig, T *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingle trig1 = (TrigSingle) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    Trig trig2 = (Trig) (trig1 + (i-1)*WG);
    T2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local T2 *lds, Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me) {
  local T *lds1 = (local T *) lds;
  TrigSingle trig1 = (TrigSingle) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    T sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      Trig trig2 = (Trig) (trig1 + i*WG);
      T2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(Trig trig, T *preloads, T2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingle trig1 = (TrigSingle) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    T cosine1_SQRT1_2 = preloads[1] * M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2_apply_delay(u[4], u[5]);
    X2_apply_delay(u[6], u[7]);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}


void OVERLOAD fft_common(local T2 *lds, T2 *u, Trig trig, T2 w, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds for variant 2
  local T2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(T2);

// Variant 0 uses broadcast instructions.  Only available on AMD GPUs.

#if VARIANT == 0

#if WG * RADIX > 1024
#error VARIANT == 0 only supported for FFT size <= 1024
#endif
#if !AMDGPU
#error VARIANT == 0 only supported by AMD GPUs
#endif

// There is a slight difference between fft_WIDTH and fft_HEIGHT.  Tail square computes the trig values
// to be broadcast, while fft_WIDTH does not.  Compute the trig values now for fft_WIDTH,
#if DOING_WIDTH
#if RADIX == 8
  w = fancyTrig_N(ND / (WG * RADIX) * lowMe);
#else
  w = slowTrig_N(ND / (WG * RADIX) * lowMe, ND / RADIX);
#endif
#endif

  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    w = bcast(w, s);
    chainMul(u, w);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

// Variant 2 uses more FMA instructions than the original FFT code.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

// Variant 2 code for SIZE=256, RADIX=4
#elif WG == 64 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif WG == 64 && RADIX == 8 && VARIANT == 2

  T preloads[10];                       // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;   // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);  // We'd rather set save_one_more_mul to 1

// Variant 2 code for SIZE=1024, RADIX=4
#elif WG == 256 && RADIX == 4 && VARIANT == 2

  T preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Custom code for SIZE=4K, RADIX=8
#elif WG == 512 && RADIX == 8 && VARIANT == 2

  T preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*            Similar to above, but for an FFT based on FP32              */
/**************************************************************************/

#if FFT_FP32

void OVERLOAD fft_RADIX(F2 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(F2 *u, F2 w) {
  u[1] = cmul(u[1], w);

  F2 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  F a = mul2(base.y);
  base = U2(fma(a, -w.y, w.x), fma(a, w.x, -w.y));
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(F2 *u, F2 w) {
  u[1] = cmulFancy(u[1], w);
                                                  //GWBUG - see FP64 version for many possible optimizations
  F2 w2 = csqTrigFancy(w);
  u[2] = cmulFancy(u[2], w2);

  F2 w3 = ccubeTrigFancy(w2, w);
  u[3] = cmulFancy(u[3], w3);

  w3.x += 1;
  F2 base = cmulFancy(w3, w);
  for (int i = 4; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmulFancy(base, w);
  }
}

void OVERLOAD chainMul(F2 *u, F2 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigFP32 trig, F2 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN32) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN32) {
    if (RADIX >= 8) {
      u[1] = cmulFancy(u[1], TFLOAD(&trig[p]));
    } else {
      u[1] = cmul(u[1], TFLOAD(&trig[p]));
    }
    for (u32 i = 2; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-4 FFTs with more FMA instructions
//************************************************************************************

// Some OpenCL compilers are having trouble with fma on floats.  Specifically, line "X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);".
// Since we're not enabling FP32 variant 2 by default, don't include these "more FMA" routines.

#if ENABLE_FP32_VARIANT_2

// Partial complex-multiply that delays the mul-by-cosine so it can be part of an FMA.
// We're trying to calculate u * U2(cosine,sine).
// real = (u.x - u.y*sine_over_cosine) * cosine
// imag = (u.x*sine_over_cosine + u.y) * cosine
F2 partial_cmul(F2 u, F sine_over_cosine) {
  return U2(fma(-u.y, sine_over_cosine, u.x), fma(u.x, sine_over_cosine, u.y));
}

// Copy of macro from fft4 and fft8 with FMAs added
#define X2_via_FMA(a, c, b) { F2 t = a; a = fma(c, b, t); b = fma(-c, b, t); }

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul4_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 3 lines of sine/cosine values for the first fft4.  Read two of the lines as a pair as AMD likes T2 global memory reads
  TrigFP32 trig2 = (TrigFP32) trig1;
  F2 sine_over_cosines = TFLOAD(&trig2[me]);
  preloads[0] = sine_over_cosines.x;
  preloads[1] = sine_over_cosines.y;
  // Read 3rd line
  preloads[2] = TFLOAD(&trig1[2*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul4(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 4*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[4];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[5];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 4; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/4) + (me/f)*(f/4)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 4; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine1, cosine2, cosine3/cosine1
    if (f < WG/4) preloads[0] = lds1[WG + ((me/f) & 3) * WG/4 + (0 * WG + me)/(4*f) * f/4];
    preloads[2] = lds1[WG + ((me/f) & 3) * WG/4 + (2 * WG + me)/(4*f) * f/4];
    preloads[3] = lds1[WG + ((me/f) & 3) * WG/4 + (3 * WG + me)/(4*f) * f/4];
    preloads[1] = lds1[WG + ((me/f) & 3) * WG/4 + (1 * WG + me)/(4*f) * f/4];
  }
}

// Finish off a partial tabMul while doing next fft4 making more use of FMA.
void finish_tabMul4_fft4(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft4 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/4) u[0] = u[0] * preloads[0];

  // Apply cosine2, cosine3/cosine1 to u[2] and u[3] using FMA
  X2_via_FMA(u[0], preloads[2], u[2]);
  X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);

  // Preload one line of sine/cosines and one line of cosines for later tabMuls.  We'll later broadcast these values as needed using LDS.
  if (f == 1) {
    preloads[4] = TFLOAD(&trig1[3*WG + me]);             // Sine/cosines for later tabMuls
    preloads[5] = TFLOAD(&trig1[4*WG + 4*WG + me]);      // Cosines for later tabMuls
  }

  // Do the last level of fft4 applying cosine1
  X2_via_FMA(u[0], preloads[1], u[1]);
  X2_via_FMA(u[2], preloads[1], u[3]);

  // revbin [0, 2, 1, 3] undo
  SWAP(u[1], u[2]);
}

//************************************************************************************
// New fft WIDTH and HEIGHT macros to support radix-8 FFTs with more FMA instructions
//************************************************************************************

// Preload trig values for the first partial tabMul.  We load the sine/cosine values early so that F64 ops can hide the read latency.
void preload_tabMul8_trig(TrigFP32 trig, F *preloads, u32 f, u32 numWG, u32 me) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  // Read 7 lines of sine/cosine values for the first fft8.  Read six of the lines as pairs as AMD likes T2 global memory reads
  for (u32 i = 1; i < 7; i += 2) {
    TrigFP32 trig2 = (TrigFP32) (trig1 + (i-1)*WG);
    F2 sine_over_cosines = TFLOAD(&trig2[me]);
    preloads[i-1] = sine_over_cosines.x;
    preloads[i] = sine_over_cosines.y;
  }
  // Read 7th line
  preloads[6] = TFLOAD(&trig1[6*WG + me]);
}

// Do a partial tabMul.  Save the mul-by-cosine for later FMA instructions.
void partial_tabMul8(local F2 *lds, TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me) {
  local F *lds1 = (local F *) lds;
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;
  trig1 += 8*WG;                // Skip past sine_over_cosine values

  // Use LDS memory to distribute preloaded trig values.
  if (f > 1) {
    bar(WG);
    lds1[me] = preloads[8];     // Preloaded sine/cosine values
    lds1[WG+me] = preloads[9];  // Preloaded cosine values
  }

  // Apply sine/cosines
  bar(WG);
  for (u32 i = 1; i < 8; ++i) {
    F sine_over_cosine;
    if (f == 1) sine_over_cosine = preloads[i-1];
    else sine_over_cosine = lds1[i*(WG/8) + (me/f)*(f/8)];
    u[i] = partial_cmul(u[i], sine_over_cosine);
  }

  // Preload cosines for finishing first tabMul (done after using up preloaded sine/cosine values).  Hopefully, shufl will hide the latency.
  if (f == 1) {
    // Read pairs of lines to make AMD happy with T2 global memory loads
    for (u32 i = 0; i < 8; i += 2) {
      TrigFP32 trig2 = (TrigFP32) (trig1 + i*WG);
      F2 cosines = TFLOAD(&trig2[me]);
      preloads[i] = cosines.x;
      preloads[i+1] = cosines.y;
    }
  }
  else {
    // Load cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3, cosine2, cosine3/cosine1, cosine1
    // Load them in the order they will be used, though it probably won't matter.
    if (f < WG/8) preloads[0] = lds1[WG + ((me/f) & 7) * WG/8 + (0 * WG + me)/(8*f) * f/8];
    preloads[1] = lds1[WG + ((me/f) & 7) * WG/8 + (1 * WG + me)/(8*f) * f/8];
    preloads[4] = lds1[WG + ((me/f) & 7) * WG/8 + (4 * WG + me)/(8*f) * f/8];
    preloads[5] = lds1[WG + ((me/f) & 7) * WG/8 + (5 * WG + me)/(8*f) * f/8];
    preloads[6] = lds1[WG + ((me/f) & 7) * WG/8 + (6 * WG + me)/(8*f) * f/8];
    preloads[7] = lds1[WG + ((me/f) & 7) * WG/8 + (7 * WG + me)/(8*f) * f/8];
    preloads[2] = lds1[WG + ((me/f) & 7) * WG/8 + (2 * WG + me)/(8*f) * f/8];
    preloads[3] = lds1[WG + ((me/f) & 7) * WG/8 + (3 * WG + me)/(8*f) * f/8];
  }
}

// Finish off a partial tabMul while doing next fft8 making more use of FMA.
void finish_tabMul8_fft8(TrigFP32 trig, F *preloads, F2 *u, u32 f, u32 numWG, u32 me, u32 save_one_more_mul) {
  TrigSingleFP32 trig1 = (TrigSingleFP32) trig;

  //
  // Mimic a traditional fft8 but use FMA instructions to apply the cosine multiplies.
  //

  // Apply cosine0 to u[0]
  if (f < WG/8) u[0] = u[0] * preloads[0];

  if (save_one_more_mul) {   // This should always be the best option.  ROCm optimizer is doing something weird in fft_WIDTH case.

    // Apply cosine4, cosine5/cosine1, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3/cosine1
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8 applying cosine1
//TODO: Save this MUL by SQRT(1/2) by pre-computing cosine1*SQRTHALF
    F cosine1_SQRT1_2 = preloads[1] * (float) M_SQRT1_2;
    X2_via_FMA(u[0], preloads[1], u[1]);
    X2_via_FMA(u[2], preloads[1], u[3]);
    X2_via_FMA(u[4], cosine1_SQRT1_2, u[5]);
    X2_via_FMA(u[6], cosine1_SQRT1_2, u[7]);

  } else {

    // Apply cosine to u[1]
    u[1] = u[1] * preloads[1];

    // Apply cosine4, cosine5, cosine6/cosine2, cosine7/cosine3 to u[4] through u[7] using FMA
    X2_via_FMA(u[0], preloads[4], u[4]);
    X2_via_FMA(u[1], preloads[5], u[5]);  u[5] = mul_t8_delayed(u[5]);
    X2_via_FMA(u[2], preloads[6], u[6]);  u[6] = mul_t4(u[6]);
    X2_via_FMA(u[3], preloads[7], u[7]);  u[7] = mul_3t8_delayed(u[7]);

    // Preload one line of sine/cosines and one line of cosines for second tabMul.  We'll later broadcast these values as needed using LDS.
    if (f == 1) {
      preloads[8] = TFLOAD(&trig1[7*WG + me]);             // Sine/cosines for second tabMul
      preloads[9] = TFLOAD(&trig1[8*WG + 8*WG + me]);      // Cosines for second tabMul
    }

    // Do the fft4Core and fft4CoreSpecial applying cosine2, cosine3
    X2_via_FMA(u[0], preloads[2], u[2]);
    X2_via_FMA(u[4], preloads[2], u[6]);
    X2_via_FMA(u[1], preloads[3], u[3]);  u[3] = mul_t4(u[3]);
    X2_via_FMA(u[5], preloads[3], u[7]);  u[7] = mul_t4(u[7]);

    // Do last level of fft8
    X2(u[0], u[1]);
    X2(u[2], u[3]);
    X2_apply_delay(u[4], u[5]);
    X2_apply_delay(u[6], u[7]);
  }

  // revbin [0, 4, 2, 6, 1, 5, 3, 7] undo
  SWAP(u[1], u[4]);
  SWAP(u[3], u[6]);
}

#endif

// Variant 2 code uses more FMA instructions than the original fft version.
// The tabMul after fft8 only does a partial complex multiply, saving a mul-by-cosine for the next fft8 using FMA instructions.
// To maximize FMA opportunities we precompute trig values as cosine and sine/cosine rather than cosine and sine.
// The downside is sine/cosine cannot be computed with chained multiplies.

void OVERLOAD fft_common(local F2 *lds, F2 *u, TrigFP32 trig, u32 numWG, u32 lowMe, int callnum) {

  // This line mimics shufl -- partition lds
  local F2* partitioned_lds = lds;
  if (numWG > 1) partitioned_lds += ((u32) get_local_id(0) / WG) * LDS_BYTES / sizeof(F2);

// Variant 2 code for SIZE=256, RADIX=4
#if ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish third tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);

// Variant 2 code for SIZE=512, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 64 && RADIX == 8 && VARIANT == 2

  F preloads[10];                        // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8 + SAVE_ONE_MUL*2*WG*8;    // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, SAVE_ONE_MUL);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish second tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, SAVE_ONE_MUL);

// Variant 2 code for SIZE=1024, RADIX=4
#elif ENABLE_FP32_VARIANT_2 && WG == 256 && RADIX == 4 && VARIANT == 2

  F preloads[6];              // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*4 + 2*WG*4;      // Skip past old FFT_width trig values.  Also skip past !save_one_more_mul trig values.

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul4_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft4, partial tabMul, and shufl.
  fft4(u);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft4.  Do second partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 1, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 4, numWG, lowMe);
  shufl(lds, u, 4, numWG, lowMe);

  // Finish the second tabMul and perform third fft4.  Do third partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 4, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 16, numWG, lowMe);
  shufl(lds, u, 16, numWG, lowMe);

  // Finish the third tabMul and perform fourth fft4.  Do fourth partial tabMul and shufl.
  finish_tabMul4_fft4(trig, preloads, u, 16, numWG, lowMe, 1);
  partial_tabMul4(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish fourth tabMul and perform final fft4.
  finish_tabMul4_fft4(trig, preloads, u, 64, numWG, lowMe, 1);

// Variant 2 code for SIZE=4K, RADIX=8
#elif ENABLE_FP32_VARIANT_2 && WG == 512 && RADIX == 8 && VARIANT == 2

  F preloads[10];             // Place to store preloaded trig values.  We want F64 ops to hide load latencies without creating register pressure.
  trig += WG*8;               // Skip past old FFT_width trig values to the !save_one_more_mul trig values

  // Preload trig values to hide global memory latencies.  As the preloads are used, the next set of trig values are preloaded.
  preload_tabMul8_trig(trig, preloads, 1, numWG, lowMe);

  // Do first fft8, partial tabMul, and shufl.
  fft8(u);
  partial_tabMul8(partitioned_lds, trig, preloads, u, 1, numWG, lowMe);
  shufl(lds, u, 1, numWG, lowMe);

  // Finish the first tabMul and perform second fft8.  Do second partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 1, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 8, numWG, lowMe);
  shufl(lds, u, 8, numWG, lowMe);

  // Finish the second tabMul and perform third fft8.  Do third partial tabMul and shufl.
  finish_tabMul8_fft8(trig, preloads, u, 8, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1
  partial_tabMul8(partitioned_lds, trig, preloads, u, 64, numWG, lowMe);
  shufl(lds, u, 64, numWG, lowMe);

  // Finish third tabMul and perform final fft8.
  finish_tabMul8_fft8(trig, preloads, u, 64, numWG, lowMe, 0);  // We'd rather set save_one_more_mul to 1

#else

  // Old / original version

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);

#endif
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

void OVERLOAD fft_RADIX(GF31 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF31 *u, GF31 w) {
  u[1] = cmul(u[1], w);

  GF31 base = csqTrig(w);
  u[2] = cmul(u[2], base);

  base = ccubeTrig(base, w);
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF31 *u, GF31 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF31 trig, GF31 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN31) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN31) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

void OVERLOAD fft_common(local GF31 *lds, GF31 *u, TrigGF31 trig, u32 numWG, u32 lowMe) {

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

void OVERLOAD fft_RADIX(GF61 *u) {
#if RADIX == 4
  fft4(u);
#elif RADIX == 8
  fft8(u);
#else
#error RADIX
#endif
}

void OVERLOAD chainMul4(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 base = csq(w);
  u[2] = cmul(u[2], base);

  base = cmul(base, w);                 //GWBUG - see FP64 version for possible optimization
  u[3] = cmul(u[3], base);
}

void OVERLOAD chainMul8(GF61 *u, GF61 w) {
  u[1] = cmul(u[1], w);

  GF61 w2 = csq(w);
  u[2] = cmul(u[2], w2);

  GF61 base = cmul(w2, w);              //GWBUG - see FP64 version for many possible optimizations
  for (int i = 3; i < 8; ++i) {
    u[i] = cmul(u[i], base);
    base = cmul(base, w);
  }
}

void OVERLOAD chainMul(GF61 *u, GF61 w) {
  // Do a length 4 chain mul
  if (RADIX == 4) chainMul4(u, w);
  // Do a length 8 chain mul
  if (RADIX == 8) chainMul8(u, w);
}

void OVERLOAD tabMul(TrigGF61 trig, GF61 *u, u32 f, u32 me) {
  u32 p = me & ~(f - 1);

// This code uses chained complex multiplies which could be faster on GPUs with great mul throughput or poor memory bandwidth or caching.

  if (TABMUL_CHAIN61) {
    chainMul(u, TFLOAD(&trig[p]));
    return;
  }

// Use memory accesses (probably cached) to reduce complex muls.  Beneficial when memory bandwidth is not the bottleneck.

  if (!TABMUL_CHAIN61) {
    for (u32 i = 1; i < RADIX; ++i) {
      u[i] = cmul(u[i], TFLOAD(&trig[(i-1)*WG + p]));
    }
    return;
  }
}

void OVERLOAD fft_common(local GF61 *lds, GF61 *u, TrigGF61 trig, u32 numWG, u32 lowMe) {

#if !UNROLL
  __attribute__((opencl_unroll_hint(1)))
#endif
  for (u32 s = 1; s < WG; s *= RADIX) {
    fft_RADIX(u);
    tabMul(trig, u, s, lowMe);
    shufl(lds, u, s, numWG, lowMe);
  }
  fft_RADIX(u);
}

#endif
