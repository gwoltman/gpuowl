#ifndef NTT_TRANSPOSE31
#define NTT_TRANSPOSE31 0
#endif
#ifndef NTT_TRANSPOSE61
#define NTT_TRANSPOSE61 0
#endif
// Copyright (C) Mihai Preda and George Woltman

#include "base.cl"
#include "fft-middle.cl"

#define INCLUDE_FILE "middle.cl"
#include "expand.cl"

#if !INPLACE                   // Original implementation (not in place)

#if FFT_FP64

KERNEL_CAP(OUT_WG) fftMiddleOut(P(T2) out, CP(T2) in, u32 base, Trig trig) {
  T2 u[MIDDLE];

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareFP64 that launched dependents before writing FP64 data

  readMiddleOutLine(u, in, y, x);

#if !PFA
  middleMul(u, x, trig);

  fft_MIDDLE(u);
#endif

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

#if PFA
  pfaMiddleOut(u, y, x, factor, trig, !MIDDLE_OUT_LDS_TRANSPOSE);       // y is the width position, x the height position
#else
  middleMul2(u, y, x, factor, trig);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local T lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX, true, x);   // With PFA, also rotates the rows (see pfaMiddleOut)
  out += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if FFT_FP32

KERNEL_CAP(OUT_WG) fftMiddleOut(P(T2) out, CP(T2) in, u32 base, Trig trig) {
  F2 u[MIDDLE];

  CP(F2) inF2 = (CP(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareFP32 that launched dependents before writing FP64 data

  readMiddleOutLine(u, inF2, y, x);

#if !PFA
  middleMul(u, x, trigF2);

  fft_MIDDLE(u);
#endif

  // FFT results come out multiplied by the FFT length (NWORDS * 2).
  const float factor = 1.0f / (NWORDS * 2);

#if PFA
  pfaMiddleOut(u, y, x, factor, trigF2, !MIDDLE_OUT_LDS_TRANSPOSE || MIDDLE > 16);       // y is the width position, x the height position
#else
  middleMul2(u, y, x, factor, trigF2);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local F lds[OUT_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX, MIDDLE <= 16, x);   // With PFA, also rotates the rows (not above MIDDLE 16, see pfaMiddleOut)
  outF2 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  outF2 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(outF2, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL_CAP(OUT_WG) fftMiddleOutGF31(P(T2) out, CP(T2) in, u32 base, Trig trig) {
  GF31 u[MIDDLE];

  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquareGF31 that launched dependents before writing GF31 data

  readMiddleOutLine(u, in31, y, x);

#if PFA
  pfaMiddleOut(u, y, x, trig31, !MIDDLE_OUT_LDS_TRANSPOSE || MIDDLE > 16);       // y is the width position, x the height position
#else
  middleMul(u, x, trig31);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig31);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z31 lds[OUT_WG / 2 * (MIDDLE <= 16 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX, MIDDLE <= 16, x);   // With PFA, also rotates the rows (not above MIDDLE 16, see pfaMiddleOut)
  out31 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out31 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out31, u, gy, gx);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL_CAP(OUT_WG) fftMiddleOutGF61(P(T2) out, CP(T2) in, u32 base, Trig trig) {
  GF61 u[MIDDLE];

  CP(GF61) in61 = (CP(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 SIZEY = OUT_WG / OUT_SIZEX;

  u32 N = SMALL_HEIGHT / OUT_SIZEX;

  u32 g = get_group_id(0);
  u32 gx = g % N;
  u32 gy = g / N;

  u32 me = get_local_id(0);
  u32 mx = me % OUT_SIZEX;
  u32 my = me / OUT_SIZEX;

  // Kernels read OUT_SIZEX consecutive T2.
  // Each WG-thread kernel processes OUT_SIZEX columns from a needed SMALL_HEIGHT columns
  // Each WG-thread kernel processes SIZEY rows out of a needed WIDTH rows

  u32 startx = gx * OUT_SIZEX;  // Each input column increases FFT element by one
  u32 starty = gy * SIZEY;  // Each input row increases FFT element by BIG_HEIGHT

  u32 x = startx + mx;
  u32 y = starty + my;

  dependentLaunchWait();   // Previous kernel was tailSquare61 that launched dependents before writing GF61 data

  readMiddleOutLine(u, in61, y, x);

#if PFA
  pfaMiddleOut(u, y, x, trig61, !MIDDLE_OUT_LDS_TRANSPOSE);       // y is the width position, x the height position
#else
  middleMul(u, x, trig61);

  fft_MIDDLE(u);

  middleMul2(u, y, x, trig61);
#endif

  dependentLaunch();       // Next kernel will be carryfused which must dependentLaunchWait before reading data

#if MIDDLE_OUT_LDS_TRANSPOSE
  // Transpose the x and y values
  local Z61 lds[OUT_WG / 2 * (MIDDLE <= 8 ? 2 * MIDDLE : MIDDLE)];
  middleShuffle(lds, u, OUT_WG, OUT_SIZEX, true, x);   // With PFA, also rotates the rows (see pfaMiddleOut)
  out61 += me;  // Threads write sequentially to memory since x and y values are already transposed
#else
  // Adjust out pointer to effect a transpose of x and y values
  out61 += mx * SIZEY + my;
#endif

  writeMiddleOutLine(out61, u, gy, gx);
}

#endif



// fftMiddleOut processes lines output by tailSquare or tailMul.  Call this the x coordinate with range 0..SMALL_HEIGHT-1
// fftMiddleOut outputs lines for carryFused or fftW.  Call this the y coordinate with range 0..WIDTH-1
// In place transpose processeses blocks of 16 x coordinates by 16 y coordinates.
// fftMiddleOut processes processes MIDDLE blocks at a time.
//
// fftMiddleOut can work on all the FFT data, in which case we can process blocks in any order.  Sequentially through memory by increasing x coordinates first might be best.
// More interestingly, fftMiddleOut can work on smaller amounts of FFT data in hopes that the data has stayed in the L2 cache during the fftMiddleIn/tailSquare/fftMiddleOut kernels.
// In this case we must work through all the x coordinates to read complete lines from tailSquare.  We must also read y and N-y due to Hermetian symmetry.


#else           // in place transpose

// L2 striping processes both base_lo and base_hi (see Gpu.cpp) in one kernel call to reduce kernel launch overhead.  There is also some special handling required for the
// first and last stripe groups.  This results in some more complicated to map group_id into startx and starty coordinates.
#if L2_STRIPING
void map_striping_group_id(u32 base_lo, u32 g, u32 *startx, u32 *starty) {
  // Old, simple L2 striping code
  // u32 N = SMALL_HEIGHT / INPLACE_TILE;
  // u32 startx = g % N * INPLACE_TILE;
  // u32 starty = base + g / N * INPLACE_TILE;

  // Process stripe group from base_lo.
  u32 oneStripeKernelsToExecute = SMALL_HEIGHT / INPLACE_TILE;
  u32 stripe_group_size = L2_STRIPING;
  u32 kernelsToExecute = stripe_group_size * oneStripeKernelsToExecute;
  if (g < kernelsToExecute) {
    *startx = g % oneStripeKernelsToExecute * INPLACE_TILE;
    *starty = base_lo + g / oneStripeKernelsToExecute * INPLACE_TILE;
    return;
  }
  g -= kernelsToExecute;

  // Process stripe group from base_hi.  The first stripe in the base_hi stripe group is not ready for output (except in the last group).
  // The last group processes the stripe that was skipped in the first base_hi group.
  u32 base_hi = WIDTH - stripe_group_size * INPLACE_TILE - base_lo;
  u32 base = (base_hi == WIDTH / 2 || (MULTI_Q && base_hi == 3 * WIDTH / 4)) ? base_hi : base_hi + INPLACE_TILE;  // Skip first stripe in base_hi (usually)
  *startx = g % oneStripeKernelsToExecute * INPLACE_TILE;
  *starty = base + g / oneStripeKernelsToExecute * INPLACE_TILE;
}
#endif

#if FFT_FP64

KERNEL_CAP((INPLACE_TILE*INPLACE_TILE)) fftMiddleOut(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  T2 u[MIDDLE];

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / INPLACE_TILE;
  u32 startx = g % N * INPLACE_TILE;
  u32 starty = g / N * INPLACE_TILE;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / INPLACE_TILE;
  u32 starty = g % N * INPLACE_TILE;
  u32 startx = g / N * INPLACE_TILE;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % INPLACE_TILE;
  u32 y = starty + me / INPLACE_TILE;

  dependentLaunchWait();   // Previous kernel was tailSquareFP64 that launched dependents before writing FP64 data

  readMiddleOutLine(u, in, y, x);

#if !PFA
  middleMul(u, x, trig);

  fft_MIDDLE(u);
#endif

  // FFT results come out multiplied by the FFT length (NWORDS).  Also, for performance reasons
  // weights and invweights are doubled meaning we need to divide by another 2^2 and 2^2.
  // Finally, roundoff errors are sometimes improved if we use the next lower double precision
  // number.  This may be due to roundoff errors introduced by applying inexact TWO_TO_N_8TH weights.
  double factor = 1.0 / (4 * 4 * NWORDS);

#if PFA
  pfaMiddleOut(u, y, x, factor, trig, true);       // y is the width position, x the height position
#else
  middleMul2(u, y, x, factor, trig);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local T2 lds[(INPLACE_TILE*INPLACE_TILE)];
  middleShuffle(lds, u);

  writeMiddleOutLine(out, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if FFT_FP32

KERNEL_CAP((INPLACE_TILE*INPLACE_TILE)) fftMiddleOut(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  F2 u[MIDDLE];

  P(F2) inF2 = (P(F2)) in;
  P(F2) outF2 = (P(F2)) out;
  TrigFP32 trigF2 = (TrigFP32) trig;

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / INPLACE_TILE;
  u32 startx = g % N * INPLACE_TILE;
  u32 starty = g / N * INPLACE_TILE;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / INPLACE_TILE;
  u32 starty = g % N * INPLACE_TILE;
  u32 startx = g / N * INPLACE_TILE;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % INPLACE_TILE;
  u32 y = starty + me / INPLACE_TILE;

  dependentLaunchWait();   // Previous kernel was tailSquareFP32 that launched dependents before writing FP64 data

  readMiddleOutLine(u, inF2, y, x);

#if !PFA
  middleMul(u, x, trigF2);

  fft_MIDDLE(u);
#endif

  // FFT results come out multiplied by the FFT length (NWORDS * 2).
  const float factor = 1.0f / (NWORDS * 2);

#if PFA
  pfaMiddleOut(u, y, x, factor, trigF2, true);       // y is the width position, x the height position
#else
  middleMul2(u, y, x, factor, trigF2);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
  local F2 lds[(INPLACE_TILE*INPLACE_TILE)];
  middleShuffle(lds, u);

  writeMiddleOutLine(outF2, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M31^2)           */
/**************************************************************************/

#if NTT_GF31

KERNEL_CAP((INPLACE_TILE31*INPLACE_TILE31)) fftMiddleOutGF31(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  GF31 u[MIDDLE];

  P(GF31) in31 = (P(GF31)) (in + DISTGF31);
  P(GF31) out31 = (P(GF31)) (out + DISTGF31);
  TrigGF31 trig31 = (TrigGF31) (trig + DISTMTRIGGF31);

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / INPLACE_TILE31;
  u32 startx = g % N * INPLACE_TILE31;
  u32 starty = g / N * INPLACE_TILE31;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / INPLACE_TILE31;
  u32 starty = g % N * INPLACE_TILE31;
  u32 startx = g / N * INPLACE_TILE31;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % INPLACE_TILE31;
  u32 y = starty + me / INPLACE_TILE31;

  dependentLaunchWait();   // Previous kernel was tailSquareGF31 that launched dependents before writing GF31 data

  readMiddleOutLine(u, in31, y, x);

#if !PFA
  middleMul(u, x, trig31);

  fft_MIDDLE(u);
#endif

#if PFA
  pfaMiddleOut(u, y, x, trig31, !(PFA == 7 && MIDDLE == 7 && NTT_TRANSPOSE31));       // y is the width position, x the height position
#else
  middleMul2(u, y, x, trig31);
#endif

  dependentLaunch();       // Next kernel will be carryFused which must dependentLaunchWait before reading data

  // Transpose the x and y values
#if PFA == 7 && MIDDLE == 7 && NTT_TRANSPOSE31
#if NTT_TRANSPOSE31 == 1
  local GF31 lds[(INPLACE_TILE31*INPLACE_TILE31)*MIDDLE];
  middleShuffleTile31(lds,u,true,x);
#else
  local Z31 lds[(INPLACE_TILE31*INPLACE_TILE31)*MIDDLE];
  middleShuffle(lds,u,(INPLACE_TILE31*INPLACE_TILE31),INPLACE_TILE31,true,x);
#endif
#else
  local GF31 lds[(INPLACE_TILE31*INPLACE_TILE31)];
  middleShuffle(lds, u);
#endif

  writeMiddleOutLine(out31, u, y, x);
}

#endif


/**************************************************************************/
/*          Similar to above, but for an NTT based on GF(M61^2)           */
/**************************************************************************/

#if NTT_GF61

KERNEL_CAP((INPLACE_TILE61*INPLACE_TILE61)) fftMiddleOutGF61(P(T2) out, P(T2) in, u32 base, Trig trig) {
  assert(out == in);
  GF61 u[MIDDLE];

  P(GF61) in61 = (P(GF61)) (in + DISTGF61);
  P(GF61) out61 = (P(GF61)) (out + DISTGF61);
  TrigGF61 trig61 = (TrigGF61) (trig + DISTMTRIGGF61);

  u32 g = get_group_id(0);
#if L2_STRIPING
  u32 startx, starty;
  map_striping_group_id(base, g, &startx, &starty);
#elif INPLACE == 1                                 // nVidia friendly padding
  u32 N = SMALL_HEIGHT / INPLACE_TILE61;
  u32 startx = g % N * INPLACE_TILE61;
  u32 starty = g / N * INPLACE_TILE61;
#else                                              // AMD friendly padding, vary x fast?  I've no explanation for why that would be better
  u32 N = WIDTH / INPLACE_TILE61;
  u32 starty = g % N * INPLACE_TILE61;
  u32 startx = g / N * INPLACE_TILE61;
#endif

  u32 me = get_local_id(0);
  u32 x = startx + me % INPLACE_TILE61;
  u32 y = starty + me / INPLACE_TILE61;

  dependentLaunchWait();   // Previous kernel was tailSquareGF61 that launched dependents before writing GF61 data

  readMiddleOutLine(u, in61, y, x);

#if !PFA
  middleMul(u, x, trig61);

  fft_MIDDLE(u);
#endif

#if PFA
  pfaMiddleOut(u, y, x, trig61, !(PFA == 7 && MIDDLE == 7 && NTT_TRANSPOSE61));       // y is the width position, x the height position
#else
  middleMul2(u, y, x, trig61);
#endif

  dependentLaunch();       // Next kernel will be carryfused which must dependentLaunchWait before reading data

  // Transpose the x and y values
#if PFA == 7 && MIDDLE == 7 && NTT_TRANSPOSE61
#if NTT_TRANSPOSE61 == 1
  local GF61 lds[(INPLACE_TILE61*INPLACE_TILE61)*MIDDLE];
  middleShuffleTile61(lds,u,true,x);
#else
  local Z61 lds[(INPLACE_TILE61*INPLACE_TILE61)*MIDDLE];
  middleShuffle(lds,u,(INPLACE_TILE61*INPLACE_TILE61),INPLACE_TILE61,true,x);
#endif
#else
  local GF61 lds[(INPLACE_TILE61*INPLACE_TILE61)];
  middleShuffle(lds, u);
#endif

  writeMiddleOutLine(out61, u, y, x);
}

#endif

#endif
